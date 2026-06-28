# ORKA-Code

Software repository for **ORKA**, an omnidirectional mobile robot platform with camera-based perception, ESP32-based drive control, ORKA-Control web interface, and SO-101 teleoperation experiments.

## Structure

| Path | Purpose |
|---|---|
| `orka-control/` | Raspberry Pi ORKA-Control server, web UI, camera stream, telemetry, YOLO/Hailo experiments and ESP32 communication |
| `esp32/` | ESP32 drive-control firmware |
| `so101/` | SO-101 Leader/Follower teleoperation helper scripts |

## Main Components

- `orka-control/orka_server.py` — Raspberry Pi server for ORKA-Control
- `orka-control/orka_controller.html` — browser-based control interface
- `orka-control/start_orka.sh` — startup helper script
- `esp32/orka_esp32.ino` — ESP32 drive-control firmware
- `so101/leader_client.py` — leader-side SO-101 control client
- `so101/follower_server.py` — follower-side SO-101 control server for Raspberry Pi

## Hardware Context

ORKA currently uses:

- Raspberry Pi 5 8GB
- Raspberry Pi Camera Module 3
- Hailo-8L M.2 HAT
- ESP32 DevKit
- L298 dual H-bridge motor controllers
- JGB37-520 24V gear motors
- 36V battery pack with 24V, 12V, and 5V rails
- SO-101 robotic arm for teleoperation experiments

## Status

Current status as of 2026-06-28:

- ORKA-Control web interface is active
- camera stream is integrated
- ESP32 telemetry and drive communication are implemented
- voltage monitoring is shown through the control interface
- SO-101 Leader/Follower teleoperation is tested
- YOLO / Hailo experiments are integrated as optional perception paths

## Notes

Large local model files such as `.pt`, `.onnx`, `.engine`, Hailo artifacts and generated YOLO output folders are intentionally not tracked in Git.

## Related Repository

Main hardware and engineering documentation:

- https://github.com/lennykossyk/ORKA
