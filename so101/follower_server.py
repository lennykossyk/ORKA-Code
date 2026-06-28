"""
follower_server.py - läuft auf dem Raspberry Pi

Aufgabe:
- Empfängt Leader-Positionen per ZMQ vom PC
- Steuert den SO-101 Follower-Arm
- Schützt den Arm bei Start, WLAN-Ausfall, Hindernissen und Teleop-Stuck

Finale Sicherheitslogik:
1. Start:
   - Follower fährt in Parkposition
   - Danach Torque aus
   - Wartet auf Leader

2. Teleop-Freigabe:
   - Leader muss ungefähr in Parkposition sein
   - Follower muss ebenfalls ungefähr in Parkposition sein
   - Erst dann wird Torque aktiviert und Teleop gestartet

3. WLAN/Leader-Ausfall:
   - Wenn 500 ms kein Paket kommt:
     Follower fährt in Parkposition und deaktiviert Torque

4. Reconnect:
   - Keine automatische Recovery-Bewegung
   - Keine Soft-Start-Bewegung
   - Teleop wird nur wieder aktiviert, wenn Leader UND Follower in Parkposition sind

5. Gripper:
   - Wird in normaler Teleop weiter gesteuert
   - Wird bei Park-Check, Load-Check und Stuck-Detection ignoriert

Usage:
    cd ~/ORKA-Code/lerobot
    source .venv/bin/activate
    python follower_server.py --port /dev/ttyACM0
"""

import argparse
import json
import os
import signal
import tempfile
import time
import zmq

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


def _handle_stop_signal(signum, frame):
    # SIGTERM/SIGINT sollen den finally-Block erreichen, damit der Arm sicher parkt.
    raise KeyboardInterrupt


signal.signal(signal.SIGTERM, _handle_stop_signal)


# -----------------------------
# Konfiguration
# -----------------------------

WATCHDOG_TIMEOUT_MS = 500
LOG_THROTTLE_S = 1.0

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Für Safety/Parken ignorieren wir den Gripper.
# Er wird aber in normaler Teleop weiterhin gesendet und gesteuert.
ARM_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]

# Gemessene sichere Parkposition.
# Der Gripper-Wert bleibt hier nur zur Dokumentation drin,
# wird aber für Safety/Parken nicht genutzt.
PARK_POSITION = {
    "shoulder_pan": -4.0,
    "shoulder_lift": -111.4,
    "elbow_flex": 96.7,
    "wrist_flex": -68.3,
    "wrist_roll": -0.1,
    "gripper": 75.1,
}

# Parken: vorsichtig, aber nicht zu empfindlich.
PARK_LOAD = 200
PARK_STEPS = 200
PARK_DURATION = 6.0

# Park-Toleranz.
# Gripper steht drin, wird aber beim Safety-Check ignoriert.
PARK_TOLERANCE = {
    "shoulder_pan": 15.0,
    "shoulder_lift": 15.0,
    "elbow_flex": 15.0,
    "wrist_flex": 15.0,
    "wrist_roll": 15.0,
    "gripper": 45.0,
}

# Teleop-Stuck-Erkennung:
# Stuck heißt:
# - hohe Last
# - kaum Bewegung
# - Zielposition ist noch deutlich entfernt
TELEOP_STUCK_LOAD = 380
TELEOP_STUCK_TIME = 1.0
TELEOP_STUCK_POS_EPS = 1.5
TELEOP_TARGET_ERROR = 5.0



# -----------------------------
# Webstatus für ORKA Flask UI
# -----------------------------

STATUS_FILE = "/tmp/orka_arm_status.json"
_status_cache = {}

def write_status(**updates):
    """Schreibt einen kleinen JSON-Status, den der Flask-Server lesen kann."""
    global _status_cache

    _status_cache.update(updates)
    _status_cache["updated_at"] = time.time()

    path = STATUS_FILE
    tmp = path + ".tmp"

    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_status_cache, f, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        # Status darf niemals den Arm stoppen.
        pass


def decode_zmq_message(message):
    """
    Unterstützt zwei Formate:
    1) alte Leader-Version: [pos0, pos1, ...]
    2) neue Leader-Version: {"type":"heartbeat"|"positions", "positions":[...]}
    """
    if isinstance(message, dict):
        msg_type = str(message.get("type", "positions"))
        positions = message.get("positions") or []
        return msg_type, positions, message

    return "positions", message, {}


# -----------------------------
# Hilfsfunktionen
# -----------------------------

def log_header(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def make_action_from_positions(positions):
    """
    Wandelt die vom PC empfangene Positionsliste in ein LeRobot-action-dict um.

    Erwartet:
        [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]

    Gibt zurück:
        {"shoulder_pan.pos": ..., ...}
    """
    if len(positions) < len(JOINT_NAMES):
        raise ValueError(f"Zu wenige Positionen empfangen: {len(positions)}")

    return {
        f"{JOINT_NAMES[i]}.pos": float(positions[i])
        for i in range(len(JOINT_NAMES))
    }


def action_is_near_park(action) -> bool:
    """
    Prüft, ob die Arm-Gelenke ungefähr in Parkposition sind.
    Der Gripper wird bewusst ignoriert.
    """
    for joint in ARM_JOINTS:
        key = f"{joint}.pos"

        if key not in action:
            return False

        diff = abs(action[key] - PARK_POSITION[joint])
        tolerance = PARK_TOLERANCE[joint]

        if diff > tolerance:
            return False

    return True


def get_worst_park_diff(action):
    """
    Gibt das Gelenk mit der größten Abweichung zur Parkposition zurück.
    Nur Arm-Gelenke, kein Gripper.
    """
    worst_joint = None
    worst_diff = 0.0

    for joint in ARM_JOINTS:
        key = f"{joint}.pos"

        if key not in action:
            return joint, 999.0

        diff = abs(action[key] - PARK_POSITION[joint])

        if diff > worst_diff:
            worst_diff = diff
            worst_joint = joint

    return worst_joint, worst_diff


def get_max_arm_load(follower) -> float:
    """
    Liest Present_Load und gibt den größten Absolutwert der Arm-Gelenke zurück.
    Der Gripper wird ignoriert, weil er oft Sonderwerte liefert.
    """
    load = follower.bus.sync_read("Present_Load")

    values = [
        abs(value)
        for joint, value in load.items()
        if joint in ARM_JOINTS
    ]

    if not values:
        return 0.0

    return max(values)


def get_target_error(obs, target_action) -> float:
    """
    Wie weit ist der Follower noch von der aktuellen Leader-Zielposition entfernt?
    Nur Arm-Gelenke, kein Gripper.
    """
    if obs is None:
        return 999.0

    errors = []

    for joint in ARM_JOINTS:
        key = f"{joint}.pos"

        if key in obs and key in target_action:
            errors.append(abs(obs[key] - target_action[key]))

    if not errors:
        return 0.0

    return max(errors)


def get_position_delta(obs_a, obs_b) -> float:
    """
    Maximale Positionsänderung zwischen zwei Beobachtungen.
    Nur Arm-Gelenke, kein Gripper.
    """
    if obs_a is None or obs_b is None:
        return 999.0

    deltas = []

    for joint in ARM_JOINTS:
        key = f"{joint}.pos"

        if key in obs_a and key in obs_b:
            deltas.append(abs(obs_a[key] - obs_b[key]))

    if not deltas:
        return 999.0

    return max(deltas)


def observation_to_action(obs):
    """
    Wandelt eine Follower-Observation in ein action-artiges dict um,
    damit action_is_near_park() wiederverwendet werden kann.
    """
    return {
        key: value
        for key, value in obs.items()
        if key.endswith(".pos")
    }


def park_and_disable(follower) -> bool:
    """
    Fährt die Arm-Gelenke sanft in Parkposition.
    Der Gripper wird nicht bewegt.

    Bei Hindernis:
        Torque sofort aus.

    Am Ende:
        Torque aus.
    """
    print("[PARK] Fahre Arm-Gelenke sanft in Parkposition...")
    write_status(mode="parking", message="Follower parkt", teleop_active=False)

    obs_start = follower.get_observation()

    start = {
        key.replace(".pos", ""): value
        for key, value in obs_start.items()
        if key.endswith(".pos")
    }

    dt = PARK_DURATION / PARK_STEPS

    for step in range(1, PARK_STEPS + 1):
        t = step / PARK_STEPS
        action = {}

        for joint in ARM_JOINTS:
            if joint not in start:
                continue

            current = start[joint]
            target = PARK_POSITION[joint]
            action[f"{joint}.pos"] = current + t * (target - current)

        try:
            max_load = get_max_arm_load(follower)

            if max_load > PARK_LOAD:
                print(f"[WARN] Parken abgebrochen: Hindernis erkannt (Load={max_load:.0f} > {PARK_LOAD})")
                follower.bus.disable_torque()
                write_status(mode="park_blocked", message=f"Hindernis beim Parken, Load={max_load:.0f}", torque_active=False, teleop_active=False)
                print("[TORQUE] Torque deaktiviert.")
                return False

        except Exception as e:
            print(f"[WARN] Park-Load-Check übersprungen: {e}")

        follower.send_action(action)
        time.sleep(dt)

    follower.bus.disable_torque()
    write_status(mode="parked", message="Parkposition erreicht, Torque aus", torque_active=False, teleop_active=False)
    print("[OK] Parkposition erreicht. Torque deaktiviert.")
    return True


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zmq-port", type=int, default=5555)
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--status-file", default="/tmp/orka_arm_status.json")
    args = parser.parse_args()

    global STATUS_FILE
    STATUS_FILE = args.status_file
    write_status(
        process="starting",
        mode="starting",
        message="Follower-Server startet",
        follower_connected=False,
        leader_connected=False,
        leader_near_park=False,
        teleop_active=False,
        torque_active=False,
        port=args.port,
        zmq_port=args.zmq_port,
    )

    log_header("ORKA SO-101 Follower Server")

    # ZMQ Subscriber: empfängt Leader-Positionen vom PC.
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.bind(f"tcp://0.0.0.0:{args.zmq_port}")
    sock.setsockopt_string(zmq.SUBSCRIBE, "")

    # Nur das neueste Paket behalten.
    # Dadurch arbeitet der Follower nach Pausen keine alte Queue ab.
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.setsockopt(zmq.RCVTIMEO, WATCHDOG_TIMEOUT_MS)

    print(f"[ZMQ] Warte auf Leader-Pakete auf Port {args.zmq_port}")

    # Follower initialisieren.
    config = SO101FollowerConfig(port=args.port, id="orka_follower")
    follower = SO101Follower(config)
    follower.connect()
    write_status(process="running", mode="follower_connected", message="Follower verbunden", follower_connected=True, port=args.port)

    print(f"[ARM] Follower verbunden: {args.port}")
    print(f"[SAFETY] Park-Load-Limit: {PARK_LOAD}")
    print(
        f"[SAFETY] Teleop-Stuck: Load>{TELEOP_STUCK_LOAD}, "
        f"Bewegung<{TELEOP_STUCK_POS_EPS}°, Ziel-Fehler>{TELEOP_TARGET_ERROR}° "
        f"für {TELEOP_STUCK_TIME}s"
    )
    print("[INFO] Gripper wird bei Safety/Parken ignoriert, aber in Teleop normal gesteuert.")

    torque_active = True
    stuck_since = None
    last_obs = None
    last_wait_print = 0.0

    try:
        # Startup-Sicherheit:
        # Follower fährt zuerst in Parkposition und deaktiviert Torque.
        print("\n[START] Follower parkt zuerst sicher.")
        park_and_disable(follower)
        torque_active = False
        write_status(mode="waiting_leader", message="Follower geparkt, warte auf Leader", torque_active=False, teleop_active=False)
        print("[OK] Startup fertig: Follower ist geparkt und Torque ist aus.")
        print("[WAIT] Warte auf Leader in Parkposition...")

        while True:
            try:
                message = sock.recv_json()
                msg_type, positions, meta = decode_zmq_message(message)

                if msg_type == "heartbeat":
                    write_status(
                        process="running",
                        mode="leader_heartbeat",
                        message="Leader verbunden, wartet auf Parkposition" if not meta.get("near_park") else "Leader in Parkposition",
                        leader_connected=True,
                        leader_near_park=bool(meta.get("near_park", False)),
                        leader_worst_joint=meta.get("worst_joint"),
                        leader_worst_diff=meta.get("worst_diff"),
                        last_packet_time=time.time(),
                        teleop_active=bool(torque_active),
                        torque_active=bool(torque_active),
                    )
                    continue

                write_status(
                    process="running",
                    leader_connected=True,
                    last_packet_time=time.time(),
                )
                target_action = make_action_from_positions(positions)

                # -----------------------------------------------------
                # Zustand: Torque ist aus.
                # Teleop darf nur starten, wenn Leader UND Follower in Parkposition sind.
                # Keine Soft-Start-Bewegung, keine Recovery-Bewegung.
                # -----------------------------------------------------
                if not torque_active:
                    if not action_is_near_park(target_action):
                        now = time.time()
                        if now - last_wait_print > LOG_THROTTLE_S:
                            joint, diff = get_worst_park_diff(target_action)
                            print(
                                f"[WAIT] Leader nicht in Parkposition "
                                f"({joint}: {diff:.1f}° Abweichung). Torque bleibt aus."
                            )
                            last_wait_print = now
                        write_status(
                            mode="waiting_leader_park",
                            message=f"Leader nicht in Parkposition: {joint} {diff:.1f}°",
                            leader_connected=True,
                            leader_near_park=False,
                            torque_active=False,
                            teleop_active=False,
                        )
                        continue

                    follower_obs = follower.get_observation()
                    follower_action = observation_to_action(follower_obs)

                    if not action_is_near_park(follower_action):
                        now = time.time()
                        if now - last_wait_print > LOG_THROTTLE_S:
                            joint, diff = get_worst_park_diff(follower_action)
                            print(
                                f"[BLOCKED] Leader passt, aber Follower nicht in Parkposition "
                                f"({joint}: {diff:.1f}° Abweichung)."
                            )
                            print("[INFO] Follower manuell in Parkposition bringen oder Script neu starten.")
                            last_wait_print = now
                        write_status(
                            mode="blocked_follower_not_park",
                            message=f"Follower nicht in Parkposition: {joint} {diff:.1f}°",
                            leader_connected=True,
                            leader_near_park=True,
                            torque_active=False,
                            teleop_active=False,
                        )
                        continue

                    print("[OK] Leader und Follower sind in Parkposition. Teleop wird aktiviert.")
                    follower.bus.enable_torque()
                    torque_active = True
                    stuck_since = None
                    last_obs = follower.get_observation()

                    # Erstes Paket ist sicher, weil beide in Parkposition sind.
                    follower.send_action(target_action)
                    write_status(
                        mode="teleop",
                        message="Teleop aktiv",
                        leader_connected=True,
                        leader_near_park=True,
                        torque_active=True,
                        teleop_active=True,
                        last_packet_time=time.time(),
                    )
                    continue

                # -----------------------------------------------------
                # Zustand: normale Teleop.
                # -----------------------------------------------------
                follower.send_action(target_action)
                write_status(
                    mode="teleop",
                    message="Teleop aktiv",
                    leader_connected=True,
                    torque_active=True,
                    teleop_active=True,
                    last_packet_time=time.time(),
                )

                try:
                    max_load = get_max_arm_load(follower)
                    obs = follower.get_observation()

                    pos_delta = get_position_delta(obs, last_obs)
                    target_error = get_target_error(obs, target_action)
                    last_obs = obs

                    write_status(max_load=round(float(max_load), 1), target_error=round(float(target_error), 2), position_delta=round(float(pos_delta), 2))

                    stuck_condition = (
                        max_load > TELEOP_STUCK_LOAD
                        and pos_delta < TELEOP_STUCK_POS_EPS
                        and target_error > TELEOP_TARGET_ERROR
                    )

                    if stuck_condition:
                        if stuck_since is None:
                            stuck_since = time.time()
                        elif time.time() - stuck_since > TELEOP_STUCK_TIME:
                            print(
                                f"[STOP] Teleop-Stuck erkannt: "
                                f"Load={max_load:.0f}, "
                                f"Bewegung={pos_delta:.2f}°, "
                                f"Ziel-Fehler={target_error:.2f}°."
                            )
                            follower.bus.disable_torque()
                            write_status(mode="stuck_stop", message="Teleop-Stuck erkannt, Torque aus", torque_active=False, teleop_active=False)
                            print("[TORQUE] Torque deaktiviert. Leader und Follower müssen wieder in Parkposition.")
                            torque_active = False
                            stuck_since = None
                            last_obs = None
                    else:
                        stuck_since = None

                except Exception as e:
                    print(f"[WARN] Teleop-Stuck-Check übersprungen: {e}")

            except zmq.Again:
                # Keine Pakete vom PC: WLAN weg, PC aus, Leader-Script gestoppt.
                write_status(leader_connected=False, leader_near_park=False, last_packet_age_ms=WATCHDOG_TIMEOUT_MS, message="Kein Leader-Signal")
                if torque_active:
                    print("\n[WARN] Kein Leader-Signal. Follower parkt sicher...")
                    park_and_disable(follower)
                    torque_active = False
                    stuck_since = None
                    last_obs = None
                    write_status(mode="waiting_leader", message="Leader-Signal verloren, Follower geparkt", torque_active=False, teleop_active=False, leader_connected=False)
                    print("[WAIT] Warte erneut auf Leader und Follower in Parkposition...")

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C erkannt. Beende sicher...")

    finally:
        if torque_active:
            park_and_disable(follower)

        follower.disconnect()
        write_status(process="stopped", mode="stopped", message="Follower-Server beendet", follower_connected=False, leader_connected=False, torque_active=False, teleop_active=False)
        sock.close()
        ctx.term()
        print("[OK] Follower-Server beendet.")


if __name__ == "__main__":
    main()
