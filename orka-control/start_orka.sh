#!/bin/bash

export ORKA_ARM_SCRIPT=/home/orka/ORKA-Code/lerobot/follower_server_webstatus.py
export ORKA_ARM_PYTHON=/home/orka/ORKA-Code/lerobot/.venv/bin/python
export ORKA_ARM_WORKDIR=/home/orka/ORKA-Code/lerobot
export ORKA_ARM_PORT=/dev/ttyACM0

cd /home/orka/ORKA-Code
python3 orka_server.py
