"""Freedrive del UR3e por ur_rtde (sin ROS).

Pone el brazo en freedrive para moverlo a mano mientras corre mujoco_unity_bridge.py --robot-ip.
El robot debe estar en modo Remote Control (e-Series) para aceptar el RTDEControlInterface.

    python3 ur_freedrive.py                     # todas las juntas libres
    python3 ur_freedrive.py --axes 0 0 0 1 1 1  # solo rotaciones del TCP libres

Enter o Ctrl+C sale de freedrive y libera el control.
"""
import argparse
import threading
import time

import rtde_control
import rtde_receive

ROBOT_IP = "192.168.0.3"  # UR3e
# ROBOT_IP = "192.168.0.1"  # UR3

STATUS_PERIOD_S = 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default=ROBOT_IP)
    ap.add_argument("--axes", type=int, nargs=6, default=[1, 1, 1, 1, 1, 1], metavar="A",
                    help="ejes libres x y z rx ry rz (1 = libre, 0 = bloqueado)")
    args = ap.parse_args()

    print(f"Connecting to robot at {args.ip}...")
    rtde_c = rtde_control.RTDEControlInterface(args.ip)
    rtde_r = rtde_receive.RTDEReceiveInterface(args.ip)

    stop = threading.Event()
    threading.Thread(target=lambda: (input(), stop.set()), daemon=True).start()

    try:
        if not rtde_c.freedriveMode(args.axes, [0, 0, 0, 0, 0, 0]):
            print("El robot rechazo freedrive (¿protective stop o no esta en Remote?).")
            return
        print(f"Freedrive activo, ejes libres {args.axes}. Enter o Ctrl+C para salir.")
        while not stop.is_set():
            # the robot can kill our control script (pendant, protective stop, another client)
            if not rtde_c.isProgramRunning():
                print("\nEl robot detuvo el script de control: freedrive ya NO esta activo.")
                break
            q = rtde_r.getActualQ()
            print("  q [deg]: " + "  ".join(f"{v * 57.2958:7.1f}" for v in q), end="\r")
            time.sleep(STATUS_PERIOD_S)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nEnding freedrive and disconnecting...")
        try:
            rtde_c.endFreedriveMode()
            rtde_c.stopScript()
        except RuntimeError as e:
            print(f"  (no se pudo cerrar limpio: {e})")
        rtde_c.disconnect()
        rtde_r.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()
