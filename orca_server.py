from flask import Flask, Response, jsonify, request, send_file
import serial
import threading
import queue
import time
import os
import subprocess
import psutil
import cv2
import numpy as np

app = Flask(__name__)

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
# COLOR_MODE = "BGR"
COLOR_MODE = "RGB"

# Kameraformat. Erst so lassen.
CAM_FORMAT = "RGB888"

# Hailo-Modell
HAILO_HEF = "/usr/share/hailo-models/yolov8s_h8l.hef"
CONF_THRESH = 0.35

# Wenn Hailo Probleme macht, testweise False setzen.
ENABLE_YOLO_INIT = True

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

yolo_in_q = queue.Queue(maxsize=2)
yolo_out_q = queue.Queue(maxsize=2)


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

    for attempt in range(1, 4):
        try:
            close_cameras()
            time.sleep(0.5)

            info = Picamera2.global_camera_info()
            print("Gefundene Kameras:", info)

            count = min(2, len(info))

            for i in range(count):
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

                print(f"Kamera {i} gestartet")
                try:
                    print(f"Kamera {i} Config:", cam.camera_configuration())
                except Exception:
                    pass

            return True

        except Exception as e:
            print(f"Kamera Init Fehler Versuch {attempt}/3:", e)
            close_cameras()
            time.sleep(1.5)

    print("Kameras konnten nicht gestartet werden")
    return False


def capture_frame_as_rgb(cam_id):
    """
    Gibt ein RGB-Bild zurück.

    COLOR_MODE:
    - "RGB": capture_array wird als RGB interpretiert
    - "BGR": capture_array wird als BGR interpretiert und zu RGB konvertiert

    Wenn Feed blau/violett ist:
    oben COLOR_MODE = "BGR" setzen und Server neu starten.
    """
    frame = cams[cam_id].capture_array()

    if frame is None:
        return None

    if frame.ndim == 3 and frame.shape[2] == 4:
        # Falls doch 4 Kanäle kommen
        # Bei RGB888 sollte das nicht passieren.
        if COLOR_MODE == "BGR":
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        return cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

    if frame.ndim == 3 and frame.shape[2] == 3:
        if COLOR_MODE == "BGR":
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    return frame


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
    return send_file("orca_controller.html")


@app.route("/status")
def status():
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


@app.route("/shutdown", methods=["POST"])
def shutdown():
    try:
        send_to_esp32("s")
        time.sleep(0.2)
        os.system("sudo shutdown -h now")
        return jsonify({"status": "shutting_down"})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)})


@app.route("/video_feed/<int:cam_id>")
def video_feed(cam_id):
    if cam_id not in [0, 1]:
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

    print("ORKA Server läuft auf http://0.0.0.0:5000")
    print("Farbmodus:", COLOR_MODE)
    print("Wenn Bild blau/violett ist: COLOR_MODE von 'RGB' auf 'BGR' ändern.")

    app.run(host="0.0.0.0", port=5000, threaded=True)