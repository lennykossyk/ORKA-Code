"""
leader_client.py - läuft auf dem PC

Aufgabe:
- Liest den SO-101 Leader-Arm
- Sendet Gelenkpositionen per ZMQ an den Raspberry Pi
- Sendet am Anfang nichts, bis der Leader ungefähr in Parkposition ist

Finale Sicherheitslogik:
1. Leader verbindet sich
2. Script wartet, bis der Leader in Parkposition ist
3. Erst dann werden Positionen an den Follower gesendet
4. Bei 5 Leader-Fehlern in Folge beendet sich das Script
5. Der Pi-Watchdog erkennt dann Paketverlust und parkt den Follower

Usage:
    cd C:\\Users\\Thales\\Documents\\Development\\SOARM101\\lerobot
    .venv\\Scripts\\Activate.ps1
    python leader_client.py --pi-ip 192.168.178.85 --port COM3
"""

import argparse
import time
import zmq

from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig


# -----------------------------
# Konfiguration
# -----------------------------

MAX_ERRORS = 5
LOG_THROTTLE_S = 1.0

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Für den Start-Check ignorieren wir den Gripper.
# Er wird weiterhin normal gesendet und gesteuert.
ARM_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]

# Muss zur Parkposition im follower_server.py passen.
PARK_POSITION = {
    "shoulder_pan": -4.0,
    "shoulder_lift": -111.4,
    "elbow_flex": 96.7,
    "wrist_flex": -68.3,
    "wrist_roll": -0.1,
    "gripper": 75.1,
}

# Nur Arm-Gelenke werden geprüft.
# Gripper wird beim Park-Check ignoriert.
PARK_TOLERANCE = {
    "shoulder_pan": 15.0,
    "shoulder_lift": 15.0,
    "elbow_flex": 15.0,
    "wrist_flex": 15.0,
    "wrist_roll": 15.0,
    "gripper": 45.0,
}


# -----------------------------
# Hilfsfunktionen
# -----------------------------

def log_header(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def extract_positions(action):
    """
    Extrahiert die 6 Gelenkpositionen aus leader.get_action().

    Unterstützt beide Key-Formate:
    - "shoulder_pan.pos"
    - "shoulder_pan"

    Falls diese Keys nicht vorhanden sind, wird als Fallback die dict-Reihenfolge genutzt.
    """
    positions = []

    for joint in JOINT_NAMES:
        key_with_pos = f"{joint}.pos"
        key_without_pos = joint

        if key_with_pos in action:
            positions.append(float(action[key_with_pos]))
        elif key_without_pos in action:
            positions.append(float(action[key_without_pos]))
        else:
            positions = []
            break

    if len(positions) == len(JOINT_NAMES):
        return positions

    values = list(action.values())

    if len(values) < len(JOINT_NAMES):
        raise ValueError(f"Leader liefert zu wenige Werte: {len(values)}")

    return [float(v) for v in values[:len(JOINT_NAMES)]]


def positions_to_dict(positions):
    """
    Wandelt Positionsliste in dict um.
    """
    return {
        joint: float(value)
        for joint, value in zip(JOINT_NAMES, positions)
    }


def leader_is_near_park(positions):
    """
    Prüft, ob der Leader ungefähr in Parkposition steht.
    Der Gripper wird ignoriert.

    Rückgabe:
        (ok, worst_joint, worst_diff)
    """
    pos = positions_to_dict(positions)

    worst_joint = None
    worst_diff = 0.0

    for joint in ARM_JOINTS:
        diff = abs(pos[joint] - PARK_POSITION[joint])
        tolerance = PARK_TOLERANCE[joint]

        if diff > worst_diff:
            worst_diff = diff
            worst_joint = joint

        if diff > tolerance:
            return False, worst_joint, worst_diff

    return True, worst_joint, worst_diff


def wait_until_leader_in_park(leader):
    """
    Wartet, bis der Leader in Parkposition ist.
    Währenddessen wird nichts an den Pi gesendet.
    """
    print("\n[START] Start-Sicherheit")
    print("Bitte Leader-Arm in Parkposition bringen.")
    print("Toleranz: Arm-Gelenke ±15°. Gripper wird ignoriert.")
    print("Solange der Leader nicht passt, wird nichts an den Follower gesendet.")

    last_print = 0.0
    error_count = 0

    while True:
        try:
            action = leader.get_action()
            positions = extract_positions(action)
            ok, worst_joint, worst_diff = leader_is_near_park(positions)

            if ok:
                print("[OK] Leader ist in Parkposition. Teleop darf starten.")
                return positions

            now = time.time()
            if now - last_print > LOG_THROTTLE_S:
                print(
                    f"[WAIT] Leader noch nicht in Parkposition "
                    f"({worst_joint}: {worst_diff:.1f}° Abweichung)."
                )
                last_print = now

            error_count = 0
            time.sleep(0.1)

        except KeyboardInterrupt:
            raise

        except Exception as e:
            error_count += 1
            print(f"[WARN] Leader-Fehler beim Park-Check ({error_count}/{MAX_ERRORS}): {e}")

            if error_count >= MAX_ERRORS:
                raise RuntimeError("Leader nicht erreichbar beim Park-Check.")

            time.sleep(0.2)


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pi-ip", default="192.168.178.85")
    parser.add_argument("--zmq-port", type=int, default=5555)
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--fps", type=int, default=50)
    args = parser.parse_args()

    log_header("ORKA SO-101 Leader Client")

    # ZMQ Publisher: sendet an den Pi.
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)

    # Nur das neueste Paket relevant halten.
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.connect(f"tcp://{args.pi_ip}:{args.zmq_port}")

    print(f"[ZMQ] Verbunden mit Pi: {args.pi_ip}:{args.zmq_port}")

    # Leader initialisieren.
    config = SO101LeaderConfig(port=args.port, id="orka_leader")
    leader = SO101Leader(config)
    leader.connect()

    print(f"[LEADER] Leader verbunden: {args.port}")
    print(f"[SAFETY] Fehlerlimit: {MAX_ERRORS} Fehler in Folge → Script stoppt.")
    print("[INFO] Am Anfang wird nichts gesendet, bis der Leader in Parkposition ist.")

    dt = 1.0 / args.fps
    error_count = 0

    try:
        # Erst warten, bis Leader in Parkposition ist.
        first_positions = wait_until_leader_in_park(leader)

        # Kurze Pause, damit der ZMQ-Subscriber sicher bereit ist.
        time.sleep(0.3)

        # Erstes sicheres Paket senden.
        sock.send_json(first_positions)
        print("[SEND] Erstes sicheres Leader-Paket gesendet.")
        print("[OK] Teleop läuft. Bewege den Leader-Arm. Ctrl+C zum Beenden.")

        while True:
            t0 = time.perf_counter()

            try:
                action = leader.get_action()
                positions = extract_positions(action)

                sock.send_json(positions)
                error_count = 0

            except Exception as e:
                error_count += 1
                print(f"[WARN] Leader-Fehler ({error_count}/{MAX_ERRORS}): {e}")

                if error_count >= MAX_ERRORS:
                    print("[ERROR] Leader nicht erreichbar. Beende Script.")
                    print("   Pi-Watchdog erkennt Paketverlust und parkt den Follower.")
                    break

            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, dt - elapsed))

    except KeyboardInterrupt:
        print("\nCtrl+C erkannt. Leader-Client wird beendet...")

    except Exception as e:
        print(f"[ERROR] Fehler: {e}")

    finally:
        try:
            leader.disconnect()
        except Exception:
            pass

        sock.close()
        ctx.term()
        print("[OK] Leader-Client beendet.")


if __name__ == "__main__":
    main()
