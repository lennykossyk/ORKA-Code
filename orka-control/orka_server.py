from flask import Flask, Response, jsonify, request, send_file
import serial
import threading
import queue
import time
import os
import sys
import signal
import json
import subprocess
import psutil
import cv2
import numpy as np
from collections import deque

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

TARGET_FPS = 20
YOLO_CAM = 0
JPEG_QUALITY = 70
CAM_RES = (640, 480)

# Wichtig:
# Erst "RGB" testen.
# Wenn der Feed weiter blau/violett ist, NUR diese Zeile ändern zu:
# Blau/violett-Fix: Picamera2 RGB888 kommt auf deinem Setup effektiv BGR an.
# Wenn Farben später falsch andersherum sind, wieder auf "RGB" stellen.
COLOR_MODE = "BGR"

# Kameraformat. Erst so lassen.
CAM_FORMAT = "RGB888"

# Kamera-Bildrotation pro Kamera.
#
# Warum hier in Python und nicht nur per CSS/HTML?
# - Der Browser-Stream ist gedreht.
# - YOLO/Hailo bekommt ebenfalls das korrekt ausgerichtete Bild.
# - Gespeicherte JPEG-Frames sind ebenfalls korrekt ausgerichtet.
#
# Erlaubte Werte pro Kamera:
#   "none" = nicht drehen
#   "180"  = auf den Kopf drehen
#   "cw90" = 90 Grad im Uhrzeigersinn
#   "ccw90"= 90 Grad gegen den Uhrzeigersinn
#
# Typischer Fall, wenn beide Kameras kopfüber montiert sind:
#   {0: "180", 1: "180"}
#
# Wenn nur Kamera 0 gedreht werden soll:
#   {0: "180", 1: "none"}
#
# Wenn die Drehung falsch ist, nur diese Werte ändern und den Server neu starten.
CAM_ROTATION = {
    0: "180",
    1: "180",
}


# Hailo-Modell
HAILO_HEF = "/usr/share/hailo-models/yolov8s_h8l.hef"
CONF_THRESH = 0.35

# Wenn Hailo Probleme macht, testweise False setzen.
ENABLE_YOLO_INIT = True

# SO-101 Arm / Leader-Follower Integration
# Wichtig: Der Follower muss im LeRobot-venv laufen, nicht im ORKA-Flask-venv.
# Standard: /home/orca/ORKA-Code/lerobot/.venv/bin/python
# Bei anderem Pfad vor dem Serverstart setzen:
#   export ORKA_ARM_PYTHON=/pfad/zu/lerobot/.venv/bin/python
ARM_ENABLE_AUTOSTART = os.environ.get("ORKA_ARM_AUTOSTART", "1") == "1"
LEROBOT_DIR = os.path.expanduser("~/ORKA-Code/lerobot")
ARM_SCRIPT = os.environ.get("ORKA_ARM_SCRIPT", os.path.join(LEROBOT_DIR, "follower_server.py"))
ARM_PYTHON = os.environ.get("ORKA_ARM_PYTHON", os.path.join(LEROBOT_DIR, ".venv", "bin", "python"))
ARM_WORKDIR = os.environ.get("ORKA_ARM_WORKDIR", LEROBOT_DIR)
ARM_PORT = os.environ.get("ORKA_ARM_PORT", "/dev/ttyACM0")
ARM_ZMQ_PORT = int(os.environ.get("ORKA_ARM_ZMQ_PORT", "5555"))
ARM_STATUS_FILE = os.environ.get("ORKA_ARM_STATUS_FILE", "/tmp/orka_arm_status.json")

# ─────────────────────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────────────────────

esp32 = None
esp32_lock = threading.Lock()

cams = [None, None]
fps = [0, 0]
detections = [0, 0]
latest_frames = [None, None]
frame_locks = [threading.Lock(), threading.Lock()]

yolo_active = False
hailo_runner = None

latest_status = {
    "cmd": "s",
    "cam": 90,
}

telemetry_lock = threading.Lock()
telemetry_state = {
    "ok": False,
    "v24": None,
    "raw": "",
    "last_update": 0.0,
    "error": "",
}
telemetry_history = deque(maxlen=180)

arm_state = {
    "base": 90,
    "shoulder": 90,
    "elbow": 90,
    "wrist": 90,
    "gripper": 90,
}

yolo_in_q = queue.Queue(maxsize=2)
yolo_out_q = queue.Queue(maxsize=2)


arm_process = None
arm_process_lock = threading.Lock()
arm_log = deque(maxlen=100)


# ─────────────────────────────────────────────────────────────────────────────
# ESP32
# ─────────────────────────────────────────────────────────────────────────────

def connect_esp32():
    global esp32

    for port in ["/dev/ttyUSB0", "/dev/ttyACM0"]:
        try:
            esp32 = serial.Serial(port, 115200, timeout=1)
            time.sleep(2)
            esp32.reset_input_buffer()
            esp32.reset_output_buffer()
            print(f"ESP32 verbunden auf {port}")
            return True
        except Exception as e:
            print(f"Kein ESP32 auf {port}: {e}")

    esp32 = None
    print("ESP32 nicht gefunden")
    return False


def send_to_esp32(cmd):
    if not esp32 or not esp32.is_open:
        return "ERR:not_connected"

    try:
        with esp32_lock:
            esp32.reset_input_buffer()
            print(">>", cmd)
            esp32.write((cmd + "\n").encode())
            esp32.flush()
            response = esp32.readline().decode(errors="ignore").strip()
            print("<<", response)
            return response if response else "OK:sent"
    except Exception as e:
        print("ESP32 Fehler:", e)
        return "ERR:exception"


def parse_esp_telemetry(response):
    """
    Erwartet vom ESP32 z.B.:
    OK:telemetry:V24=24.39,CAM=90,CMD=s,MOTORS=1,SERVO=1
    oder:
    OK:volt:24.39
    """
    result = {}

    if not response:
        return result

    response = response.strip()

    if response.startswith("OK:volt:"):
        try:
            result["v24"] = float(response.split(":")[-1])
        except Exception:
            pass
        return result

    if response.startswith("OK:telemetry:"):
        payload = response.split("OK:telemetry:", 1)[1]
        for part in payload.split(","):
            if "=" not in part:
                continue
            key, val = part.split("=", 1)
            key = key.strip().upper()
            val = val.strip()

            if key == "V24":
                try:
                    result["v24"] = float(val)
                except Exception:
                    pass
            elif key == "CAM":
                try:
                    result["cam"] = int(float(val))
                except Exception:
                    pass
            elif key == "CMD":
                result["cmd"] = val
            elif key == "MOTORS":
                result["motors"] = val == "1"
            elif key == "SERVO":
                result["servo"] = val == "1"

    return result


def poll_esp32_telemetry(force=False):
    """Fragt ESP32-Telemetrie ab und speichert 24V-Verlauf."""
    now = time.time()

    with telemetry_lock:
        age = now - float(telemetry_state.get("last_update") or 0.0)
        if not force and age < 0.8:
            return dict(telemetry_state)

    response = send_to_esp32("telemetry")
    parsed = parse_esp_telemetry(response)

    with telemetry_lock:
        telemetry_state["raw"] = response
        telemetry_state["last_update"] = now

        if response.startswith("ERR:") or not parsed:
            telemetry_state["ok"] = False
            telemetry_state["error"] = response
        else:
            telemetry_state["ok"] = True
            telemetry_state["error"] = ""
            if "v24" in parsed:
                telemetry_state["v24"] = parsed["v24"]
                telemetry_history.append({
                    "t": round(now, 3),
                    "v": parsed["v24"],
                })
            if "cam" in parsed:
                latest_status["cam"] = parsed["cam"]
            if "cmd" in parsed:
                latest_status["cmd"] = parsed["cmd"]

        return dict(telemetry_state)



# ─────────────────────────────────────────────────────────────────────────────
# SO-101 ARM PROCESS
# ─────────────────────────────────────────────────────────────────────────────

def _append_arm_log(line):
    line = str(line).rstrip()
    if not line:
        return
    arm_log.append({"t": round(time.time(), 3), "line": line[-500:]})
    print("[ARM]", line, flush=True)


def _arm_log_reader(proc):
    try:
        for line in iter(proc.stdout.readline, ""):
            if not line:
                break
            _append_arm_log(line)
    except Exception as e:
        _append_arm_log(f"log_reader_error: {e}")


def _read_arm_status_file():
    try:
        if not os.path.exists(ARM_STATUS_FILE):
            return {}
        with open(ARM_STATUS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"status_file_error": str(e)}


def arm_process_running():
    global arm_process
    return arm_process is not None and arm_process.poll() is None


def _validate_arm_runtime():
    """Prüft, ob der Follower wirklich im LeRobot-venv gestartet werden kann."""
    if not os.path.exists(ARM_SCRIPT):
        return False, f"Arm-Script nicht gefunden: {ARM_SCRIPT}"

    if not os.path.exists(ARM_PYTHON):
        return False, (
            "LeRobot-venv Python nicht gefunden: " + ARM_PYTHON +
            " | Setze z.B.: export ORKA_ARM_PYTHON=/home/orca/ORKA-Code/lerobot/.venv/bin/python"
        )

    if not os.access(ARM_PYTHON, os.X_OK):
        return False, f"LeRobot-venv Python ist nicht ausführbar: {ARM_PYTHON}"

    return True, "ok"


def start_arm_process():
    """Startet follower_server.py im LeRobot-venv als separaten Prozess."""
    global arm_process

    with arm_process_lock:
        if arm_process_running():
            return True, "already_running"

        ok, msg = _validate_arm_runtime()
        if not ok:
            _append_arm_log(msg)
            return False, msg

        cmd = [
            ARM_PYTHON,
            ARM_SCRIPT,
            "--port", ARM_PORT,
            "--zmq-port", str(ARM_ZMQ_PORT),
            "--status-file", ARM_STATUS_FILE,
        ]

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        # Damit Python-Pakete und native Libraries sicher aus dem LeRobot-venv kommen.
        venv_dir = os.path.dirname(os.path.dirname(ARM_PYTHON))
        env["VIRTUAL_ENV"] = venv_dir
        env["PATH"] = os.path.join(venv_dir, "bin") + os.pathsep + env.get("PATH", "")

        workdir = ARM_WORKDIR if os.path.isdir(ARM_WORKDIR) else (os.path.dirname(ARM_SCRIPT) or BASE_DIR)
        if workdir != ARM_WORKDIR:
            _append_arm_log(f"WARN: ARM_WORKDIR nicht gefunden ({ARM_WORKDIR}), nutze {workdir}")

        try:
            _append_arm_log("Starte Arm-Prozess im LeRobot-venv: " + " ".join(cmd))
            arm_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=workdir,
                env=env,
            )
            threading.Thread(target=_arm_log_reader, args=(arm_process,), daemon=True).start()
            return True, "started"
        except Exception as e:
            arm_process = None
            _append_arm_log(f"Start fehlgeschlagen: {e}")
            return False, str(e)


def stop_arm_process():
    """
    Stoppt den Arm-Prozess sicher.
    Wichtig: Zuerst SIGINT senden, damit follower_server.py im finally IMMER
    versucht, den Follower in Parkposition zu fahren und danach Torque zu deaktivieren.
    Erst danach härter beenden.
    """
    global arm_process

    with arm_process_lock:
        if not arm_process_running():
            return True, "not_running"

        try:
            _append_arm_log("ARM AUS: fordere Parkposition an und stoppe Arm-Prozess sicher per SIGINT...")
            arm_process.send_signal(signal.SIGINT)
            try:
                # Parken dauert im Follower-Code ca. PARK_DURATION=6s. Extra Zeit für Enable-Torque/LeRobot geben.
                arm_process.wait(timeout=25)
                return True, "stopped_safe_sigint"
            except subprocess.TimeoutExpired:
                _append_arm_log("Arm-Prozess reagiert nicht auf SIGINT, sende SIGTERM...")
                arm_process.terminate()
                try:
                    arm_process.wait(timeout=6)
                    return True, "stopped_sigterm"
                except subprocess.TimeoutExpired:
                    _append_arm_log("Arm-Prozess reagiert nicht, kill...")
                    arm_process.kill()
                    arm_process.wait(timeout=3)
                    return True, "stopped_killed"
        except Exception as e:
            return False, str(e)


def get_arm_process_status():
    file_status = _read_arm_status_file()
    running = arm_process_running()
    exit_code = None
    if arm_process is not None and arm_process.poll() is not None:
        exit_code = arm_process.poll()

    now = time.time()
    last_packet_time = file_status.get("last_packet_time")
    if last_packet_time:
        try:
            file_status["last_packet_age_ms"] = int((now - float(last_packet_time)) * 1000)
        except Exception:
            pass

    return {
        "ok": True,
        "process_running": running,
        "exit_code": exit_code,
        "autostart": ARM_ENABLE_AUTOSTART,
        "script": ARM_SCRIPT,
        "python": ARM_PYTHON,
        "workdir": ARM_WORKDIR,
        "venv_ok": os.path.exists(ARM_PYTHON),
        "port": ARM_PORT,
        "zmq_port": ARM_ZMQ_PORT,
        "status_file": ARM_STATUS_FILE,
        "status": file_status,
        "log": list(arm_log)[-25:],
    }

# ─────────────────────────────────────────────────────────────────────────────
# CAMERA
# ─────────────────────────────────────────────────────────────────────────────

def close_cameras():
    global cams

    for i, cam in enumerate(cams):
        try:
            if cam is not None:
                cam.stop()
                cam.close()
                print(f"Kamera {i} geschlossen")
        except Exception as e:
            print(f"Kamera {i} close Fehler:", e)

    cams = [None, None]


def init_cameras():
    global cams

    try:
        from picamera2 import Picamera2
    except Exception as e:
        print("Picamera2 Import Fehler:", e)
        return False

    close_cameras()
    time.sleep(0.5)

    try:
        info = Picamera2.global_camera_info()
        print("Gefundene Kameras:", info)
    except Exception as e:
        print("Kamera-Info Fehler:", e)
        info = []

    any_started = False

    for i in range(2):
        cams[i] = None

        if i >= len(info):
            print(f"Kamera {i}: nicht vorhanden")
            continue

        try:
            cam = Picamera2(i)

            config = cam.create_video_configuration(
                main={
                    "size": CAM_RES,
                    "format": CAM_FORMAT,
                }
            )

            cam.configure(config)
            cam.start()
            time.sleep(0.5)

            cams[i] = cam
            any_started = True

            print(f"Kamera {i} gestartet")
            try:
                print(f"Kamera {i} Config:", cam.camera_configuration())
            except Exception:
                pass

        except Exception as e:
            print(f"Kamera {i}: Start fehlgeschlagen:", e)
            try:
                cam.close()
            except Exception:
                pass
            cams[i] = None

    if not any_started:
        print("WARNUNG: Keine Kamera gestartet. Server läuft ohne Kamerastream.")

    return any_started


def _camera_rotation_to_cv2(rotation):
    """
    Wandelt die lesbare Rotation aus CAM_ROTATION in einen OpenCV-Code um.

    Rückgabe:
    - None: keine Rotation
    - cv2.ROTATE_*: Rotation, die cv2.rotate() versteht

    Dadurch bleibt die Config oben lesbar: "180", "cw90", "ccw90", "none".
    """
    rotation = str(rotation or "none").strip().lower()

    if rotation in ("none", "0", "off", "false", "no", ""):
        return None

    if rotation in ("180", "rotate_180", "upside_down"):
        return cv2.ROTATE_180

    if rotation in ("cw90", "90cw", "right", "rechts"):
        return cv2.ROTATE_90_CLOCKWISE

    if rotation in ("ccw90", "90ccw", "left", "links"):
        return cv2.ROTATE_90_COUNTERCLOCKWISE

    print(f"WARN: Unbekannte Kamerarotation '{rotation}', nutze keine Rotation")
    return None


def rotate_rgb_frame(rgb_frame, cam_id):
    """
    Dreht ein bereits nach RGB konvertiertes Kamerabild.

    Wichtig:
    - Die Rotation passiert nach der Farbkorrektur.
    - Dadurch bleiben Farben korrekt.
    - YOLO, JPEG-Stream und gespeicherte Frames nutzen dieselbe Ausrichtung.
    """
    if rgb_frame is None:
        return None

    cv2_rotation = _camera_rotation_to_cv2(CAM_ROTATION.get(cam_id, "none"))

    if cv2_rotation is None:
        return rgb_frame

    return cv2.rotate(rgb_frame, cv2_rotation)


def capture_frame_as_rgb(cam_id):
    """
    Holt ein Bild von Picamera2, korrigiert die Farbreihenfolge und dreht es optional.

    Ablauf:
    1. Picamera2 liefert ein Rohbild mit capture_array().
    2. COLOR_MODE sorgt dafür, dass das Ergebnis intern als RGB vorliegt.
    3. CAM_ROTATION dreht das RGB-Bild pro Kamera.
    4. Rückgabe ist immer das Bild, mit dem der Rest des Servers arbeitet.

    Warum immer RGB zurückgeben?
    - Hailo bekommt in diesem Code RGB-Frames.
    - Die Bounding-Box-Zeichnung arbeitet auf RGB.
    - Erst rgb_to_jpeg() wandelt für OpenCV/JPEG wieder nach BGR.
    """
    frame = cams[cam_id].capture_array()

    if frame is None:
        return None

    # Fall 1: Picamera2 liefert 4 Kanäle, z.B. RGBA/BGRA.
    # Bei CAM_FORMAT="RGB888" sollte das normalerweise nicht passieren,
    # aber diese Behandlung macht den Code robuster.
    if frame.ndim == 3 and frame.shape[2] == 4:
        if COLOR_MODE == "BGR":
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

    # Fall 2: Normalfall mit 3 Kanälen.
    # COLOR_MODE="BGR" bedeutet: Das Bild sieht trotz RGB888 effektiv wie BGR aus
    # und muss für korrekte Farben nach RGB konvertiert werden.
    elif frame.ndim == 3 and frame.shape[2] == 3:
        if COLOR_MODE == "BGR":
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb = frame

    # Fall 3: Unerwartetes Format, z.B. Graustufenbild.
    # Dann nicht konvertieren, sondern unverändert weitergeben.
    else:
        rgb = frame

    return rotate_rgb_frame(rgb, cam_id)


def rgb_to_jpeg(rgb_frame):
    """
    OpenCV imencode erwartet BGR.
    Browser/JPEG bekommt dadurch korrekte Farben.
    """
    if rgb_frame is None:
        return None

    if rgb_frame.ndim == 3 and rgb_frame.shape[2] == 3:
        bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    else:
        bgr = rgb_frame

    ok, buf = cv2.imencode(
        ".jpg",
        bgr,
        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    )

    if not ok:
        return None

    return buf.tobytes()


# ─────────────────────────────────────────────────────────────────────────────
# HAILO / YOLO
# ─────────────────────────────────────────────────────────────────────────────

class HailoRunner:
    """
    Hailo YOLO Runner.

    Wichtige Änderung:
    Input ist UINT8, nicht float32/0..1.
    Viele Hailo-HEF YOLO-Modelle sind quantisiert und erwarten UINT8.
    """

    def __init__(self, hef_path):
        import hailo_platform as hpf

        self._lock = threading.Lock()

        self._target = hpf.VDevice()
        self._hef = hpf.HEF(hef_path)

        params = hpf.ConfigureParams.create_from_hef(
            self._hef,
            interface=hpf.HailoStreamInterface.PCIe
        )

        self._ng = self._target.configure(self._hef, params)[0]
        self._ngp = self._ng.create_params()

        # Wichtig: Input UINT8
        self._in_p = hpf.InputVStreamParams.make(
            self._ng,
            format_type=hpf.FormatType.UINT8
        )

        # Output FLOAT32 für bequemes Parsing
        self._out_p = hpf.OutputVStreamParams.make(
            self._ng,
            format_type=hpf.FormatType.FLOAT32
        )

        in_info = self._hef.get_input_vstream_infos()[0]
        out_info = self._hef.get_output_vstream_infos()[0]

        self._in_name = in_info.name
        self._out_name = out_info.name

        self._in_h = in_info.shape[0]
        self._in_w = in_info.shape[1]

        print("Hailo geladen:", hef_path)
        print("Hailo Input:", self._in_name, in_info.shape)
        print("Hailo Output:", self._out_name, out_info.shape)

    def infer(self, rgb_frame):
        orig_h, orig_w = rgb_frame.shape[:2]

        resized = cv2.resize(rgb_frame, (self._in_w, self._in_h))

        # Hailo bekommt UINT8 RGB, Shape (1,H,W,3)
        inp = resized.astype(np.uint8)[np.newaxis, ...]

        with self._lock:
            from hailo_platform import InferVStreams

            with self._ng.activate(self._ngp):
                with InferVStreams(self._ng, self._in_p, self._out_p) as pipeline:
                    raw = pipeline.infer({self._in_name: inp})

        out = np.array(raw[self._out_name])

        boxes = self._parse_output(out, orig_w, orig_h)
        annotated = self._draw_boxes(rgb_frame.copy(), boxes)

        return annotated, len(boxes)

    def _parse_output(self, out, orig_w, orig_h):
        """
        Erwartet typisches Hailo YOLO NMS-Output:
        häufig (1, 80, 5, 100) oder (80, 5, 100).

        Falls dein HEF ein anderes Output-Format hat, wird hier nichts erkannt.
        Dann brauchen wir die gedruckte Output-Shape.
        """
        arr = np.array(out)

        if arr.ndim == 4:
            arr = arr[0]

        results = []

        if arr.ndim == 3 and arr.shape[1] == 5:
            num_classes, _, max_det = arr.shape

            for cls in range(num_classes):
                for det in range(max_det):
                    score = float(arr[cls, 4, det])
                    if score < CONF_THRESH:
                        continue

                    x1 = int(float(arr[cls, 0, det]) * orig_w)
                    y1 = int(float(arr[cls, 1, det]) * orig_h)
                    x2 = int(float(arr[cls, 2, det]) * orig_w)
                    y2 = int(float(arr[cls, 3, det]) * orig_h)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    results.append((x1, y1, x2, y2, score, cls))

            return results

        print("WARN: Unbekanntes Hailo Output Format:", arr.shape)
        return results

    def _draw_boxes(self, rgb_frame, boxes):
        for x1, y1, x2, y2, score, cls in boxes:
            cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(
                rgb_frame,
                f"{cls} {score:.2f}",
                (x1, max(y1 - 5, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1
            )

        return rgb_frame


class UltralyticsRunner:
    def __init__(self):
        from ultralytics import YOLO
        self._model = YOLO("yolov8n.pt")
        self._lock = threading.Lock()
        print("Ultralytics CPU-Fallback geladen")

    def infer(self, rgb_frame):
        # Ultralytics/OpenCV arbeitet gut mit BGR
        bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        with self._lock:
            results = self._model(bgr, verbose=False)

        annotated_bgr = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        return annotated_rgb, len(results[0].boxes)


def init_yolo():
    global hailo_runner

    if not ENABLE_YOLO_INIT:
        print("YOLO Init deaktiviert")
        hailo_runner = None
        return

    try:
        hailo_runner = HailoRunner(HAILO_HEF)
        print("Hailo Runner aktiv")
        return
    except Exception as e:
        print("Hailo nicht verfügbar:", e)

    try:
        hailo_runner = UltralyticsRunner()
        print("Ultralytics Fallback aktiv")
        return
    except Exception as e:
        print("Ultralytics nicht verfügbar:", e)

    hailo_runner = None
    print("Kein YOLO verfügbar")


# ─────────────────────────────────────────────────────────────────────────────
# THREADS
# ─────────────────────────────────────────────────────────────────────────────

def camera_thread(cam_id):
    global fps, detections

    frame_count = 0
    fps_start = time.time()

    while True:
        loop_start = time.time()

        try:
            if cams[cam_id] is None:
                time.sleep(0.1)
                continue

            rgb = capture_frame_as_rgb(cam_id)

            if rgb is None:
                time.sleep(0.05)
                continue

            if cam_id == YOLO_CAM and yolo_active and hailo_runner:
                try:
                    yolo_in_q.put_nowait(rgb)
                except queue.Full:
                    pass

                try:
                    annotated, num_det = yolo_out_q.get_nowait()
                    rgb = annotated
                    detections[cam_id] = num_det
                except queue.Empty:
                    pass
            else:
                detections[cam_id] = 0

            frame_count += 1
            if frame_count % 10 == 0:
                elapsed = time.time() - fps_start
                fps[cam_id] = round(10 / elapsed, 1) if elapsed > 0 else 0
                fps_start = time.time()

            jpeg = rgb_to_jpeg(rgb)
            if jpeg:
                with frame_locks[cam_id]:
                    latest_frames[cam_id] = jpeg

            if cam_id == YOLO_CAM and yolo_active:
                sleep_t = (1.0 / TARGET_FPS) - (time.time() - loop_start)
                if sleep_t > 0:
                    time.sleep(sleep_t)

        except Exception as e:
            print(f"Kamera Thread {cam_id} Fehler:", e)
            time.sleep(0.1)


def hailo_thread():
    while True:
        try:
            rgb = yolo_in_q.get(timeout=1.0)

            if hailo_runner is None:
                continue

            annotated, num_det = hailo_runner.infer(rgb)

            while not yolo_out_q.empty():
                try:
                    yolo_out_q.get_nowait()
                except queue.Empty:
                    break

            yolo_out_q.put_nowait((annotated, num_det))

        except queue.Empty:
            continue
        except Exception as e:
            print("YOLO Thread Fehler:", e)
            time.sleep(0.1)


def generate_frames(cam_id):
    while True:
        with frame_locks[cam_id]:
            frame = latest_frames[cam_id]

        if frame:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                frame +
                b"\r\n"
            )
        else:
            time.sleep(0.05)


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM INFO
# ─────────────────────────────────────────────────────────────────────────────

def get_temp_c():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return round(int(f.read().strip()) / 1000, 1)
    except Exception:
        return None


def get_throttled():
    try:
        return subprocess.check_output(
            ["vcgencmd", "get_throttled"],
            text=True
        ).strip()
    except Exception:
        return "unknown"


def get_uptime():
    seconds = int(time.time() - psutil.boot_time())
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def estimate_pi5_watts(cpu):
    return round(3.0 + (cpu / 100.0) * 10.0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_file(os.path.join(BASE_DIR, "orka_controller.html"))


@app.route("/status")
@app.route("/api/status")
def status():
    cam_available = [
        cams[0] is not None,
        cams[1] is not None,
    ]

    return jsonify({
        "connected": esp32 is not None and esp32.is_open,
        "yolo_active": yolo_active,
        "fps0": fps[0],
        "fps1": fps[1],
        "detections0": detections[0],
        "detections1": detections[1],
        "cmd": latest_status["cmd"],
        "cam": latest_status["cam"],
        "hailo": isinstance(hailo_runner, HailoRunner),
        "color_mode": COLOR_MODE,
        "cam_rotation": CAM_ROTATION,

        "cam_available": cam_available,
        "cam0_available": cam_available[0],
        "cam1_available": cam_available[1],
    })


@app.route("/system")
def system():
    cpu = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory()
    throttled = get_throttled()

    return jsonify({
        "cpu_percent": round(cpu, 1),
        "ram_percent": round(ram.percent, 1),
        "ram_used_mb": round(ram.used / 1024 / 1024),
        "ram_total_mb": round(ram.total / 1024 / 1024),
        "temp_c": get_temp_c(),
        "uptime": get_uptime(),
        "power_est_w": estimate_pi5_watts(cpu),
        "throttled": throttled,
        "power_ok": throttled in ["throttled=0x0", "unknown"],
    })


@app.route("/cmd", methods=["POST"])
def cmd():
    data = request.get_json(silent=True) or {}

    command = data.get("cmd", "s")
    speed = int(data.get("speed", 150))
    speed = max(0, min(255, speed))

    if command in ["s", "stop", "testservo"]:
        serial_cmd = command
    else:
        serial_cmd = f"{command}:{speed}"

    response = send_to_esp32(serial_cmd)
    latest_status["cmd"] = command

    return jsonify({
        "response": response,
        "cmd": command,
        "speed": speed,
    })


@app.route("/servo", methods=["POST"])
def servo():
    data = request.get_json(silent=True) or {}

    angle = int(data.get("angle", 90))
    angle = max(10, min(170, angle))

    response = send_to_esp32(f"cam:{angle}")
    latest_status["cam"] = angle

    return jsonify({
        "response": response,
        "angle": angle,
    })


@app.route("/yolo", methods=["POST"])
def yolo():
    global yolo_active

    data = request.get_json(silent=True) or {}
    yolo_active = bool(data.get("active", False))

    return jsonify({
        "yolo_active": yolo_active
    })


@app.route("/telemetry")
@app.route("/api/telemetry")
def telemetry():
    state = poll_esp32_telemetry(force=False)

    with telemetry_lock:
        history = list(telemetry_history)

    return jsonify({
        "ok": bool(state.get("ok")),
        "v24": state.get("v24"),
        "raw": state.get("raw", ""),
        "error": state.get("error", ""),
        "last_update": state.get("last_update", 0.0),
        "history": history,
    })


@app.route("/arm", methods=["POST"])
@app.route("/api/arm", methods=["POST"])
def arm():
    data = request.get_json(silent=True) or {}

    joint = str(data.get("joint", "")).lower().strip()
    angle = int(data.get("angle", 90))
    angle = max(0, min(180, angle))

    if joint not in arm_state:
        return jsonify({
            "ok": False,
            "error": "unknown_joint",
            "allowed": list(arm_state.keys()),
        }), 400

    arm_state[joint] = angle

    # ESP32 hat aktuell nur einen Platzhalter für arm:<joint>:<angle>.
    response = send_to_esp32(f"arm:{joint}:{angle}")

    return jsonify({
        "ok": response.startswith("OK:"),
        "response": response,
        "joint": joint,
        "angle": angle,
        "arm": arm_state,
    })


@app.route("/arm")
@app.route("/api/arm")
@app.route("/arm/status")
@app.route("/api/arm/status")
def arm_status():
    return jsonify(get_arm_process_status())


@app.route("/arm/start", methods=["POST"])
@app.route("/api/arm/start", methods=["POST"])
def arm_start():
    ok, msg = start_arm_process()
    return jsonify({"ok": ok, "message": msg, "arm": get_arm_process_status()})


@app.route("/arm/stop", methods=["POST"])
@app.route("/api/arm/stop", methods=["POST"])
@app.route("/arm/off", methods=["POST"])
@app.route("/api/arm/off", methods=["POST"])
def arm_stop():
    ok, msg = stop_arm_process()
    return jsonify({"ok": ok, "message": msg, "arm": get_arm_process_status()})


@app.route("/server_stop", methods=["POST"])
@app.route("/api/server_stop", methods=["POST"])
def server_stop():
    try:
        send_to_esp32("s")
    except Exception:
        pass
    try:
        stop_arm_process()
    except Exception:
        pass

    def _stop():
        time.sleep(0.4)
        os._exit(0)

    threading.Thread(target=_stop, daemon=True).start()
    return jsonify({"status": "server_stopping"})


@app.route("/shutdown", methods=["POST"])
@app.route("/api/shutdown", methods=["POST"])
def shutdown():
    try:
        send_to_esp32("s")
        try:
            stop_arm_process()
        except Exception:
            pass
        time.sleep(0.2)
        os.system("sudo shutdown -h now")
        return jsonify({"status": "shutting_down"})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)})


@app.route("/video_feed/<int:cam_id>")
def video_feed(cam_id):
    if cam_id not in [0, 1] or cams[cam_id] is None:
        return "Invalid camera", 404

    return Response(
        generate_frames(cam_id),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ─────────────────────────────────────────────────────────────────────────────
# START
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    connect_esp32()

    if not init_cameras():
        print("WARNUNG: Keine Kameras gestartet")

    # Kleine Pause hilft oft gegen Init-Race bei Kamera/Hailo
    time.sleep(1.0)

    init_yolo()

    for i in range(2):
        if cams[i] is not None:
            t = threading.Thread(target=camera_thread, args=(i,), daemon=True)
            t.start()
            print(f"Kamera Thread {i} gestartet")

    if hailo_runner is not None:
        ht = threading.Thread(target=hailo_thread, daemon=True)
        ht.start()
        print("YOLO Thread gestartet")

    if ARM_ENABLE_AUTOSTART:
        print("Arm-Autostart aktiviert")
        start_arm_process()
    else:
        print("Arm-Autostart deaktiviert")

    print("ORKA Server läuft auf http://0.0.0.0:5000")
    print("Farbmodus:", COLOR_MODE)
    print("Kamera-Rotation:", CAM_ROTATION)
    print("Wenn Farben wieder falsch sind: COLOR_MODE zwischen 'RGB' und 'BGR' wechseln.")
    print("Wenn das Bild falsch herum ist: CAM_ROTATION oben anpassen, z.B. 'none', '180', 'cw90', 'ccw90'.")

    app.run(host="0.0.0.0", port=5000, threaded=True)