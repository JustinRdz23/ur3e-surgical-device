"""Emulated freedrive via force mode (ur_rtde) with a bias-cancelling wrench offset.

Same idea as freedrive_force_mode.py, plus: the TCP force the sensor reports while the
arm is nearly still is tracked as a slow "bias" estimate, and the commanded wrench is
offset against it (clamped to a few N) so that phantom force does not turn into drift.

    python3 freedrive_force_mode_offset.py                   # both arms, offset on
    python3 freedrive_force_mode_offset.py left              # one arm
    python3 freedrive_force_mode_offset.py left --baseline   # offset off, for A/B comparison

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
PAYLOAD_MASS_KG = None            # e.g. 0.5
PAYLOAD_COG_M = [0.0, 0.0, 0.0]   # centre of gravity from the flange, metres
TCP_OFFSET = None                 # e.g. [0, 0, 0.2, 0, 0, 0]

# Force mode: zero base wrench, every axis compliant.
TASK_FRAME = [0, 0, 0, 0, 0, 0]
SELECTION_VECTOR = [1, 1, 1, 1, 1, 1]
WRENCH = np.zeros(6)
FORCE_TYPE = 2
LIMITS = [0.1, 0.1, 1.5, 3.14, 3.14, 0.5]  # speed caps on the compliant axes
DAMPING = 0.004
GAIN_SCALING = 1.0

# Bias-cancelling offset.
# OFFSET_SIGN = -1 assumes the sensor reports the force acting ON the tool while the
# commanded wrench is the force the robot applies to its environment (opposite signs).
# This is NOT verified: if the arm drifts MORE with the offset than with --baseline,
# set it to +1.
OFFSET_SIGN = -1.0
BIAS_TAU_S = 1.0                       # time constant of the bias estimate
QUIET_LIN_SPEED = 0.005                # m/s: below this the arm counts as "still"
QUIET_ANG_SPEED = 0.05                 # rad/s
CLAMP = np.array([2.0, 2.0, 2.0, 0.2, 0.2, 0.2])   # max offset: N, N, N, Nm, Nm, Nm

CONTROL_PERIOD_S = 0.002
REPORT_PERIOD_S = 0.5


def main():
    args = sys.argv[1:]
    use_offset = "--baseline" not in args
    sides = [a for a in args if not a.startswith("--")] or list(ROBOT_IPS)
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

        # Zero the F/T sensor only while the arm is still and nobody is touching it.
        input("Put the arm(s) in a clean pose, hands off, then press Enter to zero the sensor...")
        for c in controllers.values():
            c.zeroFtSensor()
            c.forceModeSetDamping(DAMPING)
            c.forceModeSetGainScaling(GAIN_SCALING)

        state = {
            side: {
                "bias": np.zeros(6),
                "force": np.zeros(6),
                "speed": np.zeros(6),
                "offset": np.zeros(6),
            }
            for side in sides
        }

        running = True

        def handle_sigint(signum, frame):
            nonlocal running
            print("\nCtrl+C received, stopping...")
            running = False

        signal.signal(signal.SIGINT, handle_sigint)

        mode = "OFFSET ON" if use_offset else "BASELINE (offset off)"
        print(f"Force mode starting [{mode}]. The arm(s) become compliant now, so support them.")
        last_report = last_loop = time.perf_counter()

        while running:
            loop_start = time.perf_counter()
            dt = loop_start - last_loop
            last_loop = loop_start
            alpha = min(1.0, dt / BIAS_TAU_S)

            for side, c in controllers.items():
                s = state[side]
                r = receivers[side]

                s["force"] = np.array(r.getActualTCPForce())
                s["speed"] = np.array(r.getActualTCPSpeed())

                if use_offset:
                    quiet = (
                        np.linalg.norm(s["speed"][:3]) < QUIET_LIN_SPEED
                        and np.linalg.norm(s["speed"][3:]) < QUIET_ANG_SPEED
                    )
                    # Only learn the bias while the arm is still, so deliberate
                    # operator motion and pushes are not cancelled.
                    if quiet:
                        s["bias"] += alpha * (s["force"] - s["bias"])
                    s["offset"] = OFFSET_SIGN * np.clip(s["bias"], -CLAMP, CLAMP)

                wrench = (WRENCH + s["offset"]).tolist()
                if not c.forceMode(TASK_FRAME, SELECTION_VECTOR, wrench, FORCE_TYPE, LIMITS):
                    print(f"[{side}] forceMode() returned False, stopping.")
                    running = False

            if loop_start - last_report >= REPORT_PERIOD_S:
                last_report = loop_start
                for side, s in state.items():
                    f, v, off = s["force"], s["speed"], s["offset"]
                    print(
                        f"[{side}] v={np.linalg.norm(v[:3]) * 1000:6.2f} mm/s  "
                        f"F=[{f[0]:+6.2f} {f[1]:+6.2f} {f[2]:+6.2f}] N  "
                        f"off=[{off[0]:+5.2f} {off[1]:+5.2f} {off[2]:+5.2f}] N"
                    )

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