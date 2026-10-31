"""Emulated freedrive via force mode (ur_rtde). No MuJoCo, no visuals.

Each selected arm is put in force mode with a zero wrench on all six axes, so it
yields to hand pressure. forceMode() must be re-sent every cycle or the robot
leaves force mode.

    python3 freedrive_force_mode.py          # both arms
    python3 freedrive_force_mode.py left     # one arm
    python3 freedrive_force_mode.py right

Ctrl+C stops force mode on every connected arm.
"""
import signal
import sys
import time

import numpy as np
import rtde_control
import rtde_receive

ROBOT_IPS = {
    "left": "192.168.0.1",   # UR3
    "right": "192.168.0.3",  # UR3e
}

# Optional: set these to the real tool values if a tool is mounted.
# A wrong payload/CoG shows up as a phantom force, which force mode turns into drift.
PAYLOAD_MASS_KG = None            # e.g. 0.5
PAYLOAD_COG_M = [0.0, 0.0, 0.0]   # centre of gravity from the flange, metres
TCP_OFFSET = None                 # e.g. [0, 0, 0.2, 0, 0, 0]

# Force mode: zero wrench, every axis compliant.
TASK_FRAME = [0, 0, 0, 0, 0, 0]
SELECTION_VECTOR = [1, 1, 1, 1, 1, 1]
WRENCH = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
FORCE_TYPE = 2
LIMITS = [0.1, 0.1, 1.5, 3.14, 3.14, 0.5]  # speed caps on the compliant axes
DAMPING = 0.004
GAIN_SCALING = 1.0

CONTROL_PERIOD_S = 0.002
REPORT_PERIOD_S = 0.5


def main():
    sides = sys.argv[1:] or list(ROBOT_IPS)
    for side in sides:
        if side not in ROBOT_IPS:
            raise SystemExit(f"Unknown arm '{side}'. Choose from: {', '.join(ROBOT_IPS)}")

    controllers = {}
    receivers = {}

    try:
        for side in sides:
            ip = ROBOT_IPS[side]
            print(f"Connecting to {side} robot at {ip}...")
            controllers[side] = rtde_control.RTDEControlInterface(ip)
            receivers[side] = rtde_receive.RTDEReceiveInterface(ip)

        for c in controllers.values():
            if TCP_OFFSET is not None:
                c.setTcp(TCP_OFFSET)
            if PAYLOAD_MASS_KG is not None:
                c.setPayload(PAYLOAD_MASS_KG, PAYLOAD_COG_M)

        # Zero the F/T sensor only while the arm is still and nobody is touching it,
        # otherwise the operator's load gets baked into the bias.
        input("Put the arm(s) in a clean pose, hands off, then press Enter to zero the sensor...")
        for c in controllers.values():
            c.zeroFtSensor()
            c.forceModeSetDamping(DAMPING)
            c.forceModeSetGainScaling(GAIN_SCALING)

        running = True

        def handle_sigint(signum, frame):
            nonlocal running
            print("\nCtrl+C received, stopping...")
            running = False

        signal.signal(signal.SIGINT, handle_sigint)

        print("Force mode starting. The arm(s) become compliant now, so support them.")
        last_report = time.perf_counter()

        while running:
            loop_start = time.perf_counter()

            for side, c in controllers.items():
                if not c.forceMode(TASK_FRAME, SELECTION_VECTOR, WRENCH, FORCE_TYPE, LIMITS):
                    print(f"[{side}] forceMode() returned False, stopping.")
                    running = False

            # Drift diagnostic: at rest, |F| should be near 0 and |v| should stay near 0.
            # |v| creeping up while |F| is near 0 means a phantom force (bias or payload error).
            if loop_start - last_report >= REPORT_PERIOD_S:
                last_report = loop_start
                for side, r in receivers.items():
                    force = np.linalg.norm(r.getActualTCPForce()[:3])
                    speed = np.linalg.norm(r.getActualTCPSpeed()[:3])
                    print(f"[{side}] |F|={force:6.2f} N   |v|={speed * 1000:6.2f} mm/s")

            remaining = CONTROL_PERIOD_S - (time.perf_counter() - loop_start)
            if remaining > 0:
                time.sleep(remaining)

    finally:
        print("Stopping force mode and disconnecting...")
        for side, c in controllers.items():
            try:
                c.forceModeStop()
                c.stopScript()
            except Exception as e:
                print(f"[{side}] error stopping force mode: {e}")
        for side, r in receivers.items():
            try:
                r.disconnect()
            except Exception as e:
                print(f"[{side}] error disconnecting: {e}")
        print("Done.")


if __name__ == "__main__":
    main()