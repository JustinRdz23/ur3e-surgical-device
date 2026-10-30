"""Freedrive del UR3 por sockets (sin ROS, sin ur_rtde).

Manda URScript por el puerto 30002 y lee las juntas del stream de 30003.
Funciona en CB3 y e-Series (en e-Series, ponerlo en Remote Control si hace falta).

    python3 ur_freedrive.py                     # todas las juntas libres
    python3 ur_freedrive.py --robot both        # UR3 y UR3e a la vez
    python3 ur_freedrive.py --robot ur3e        # seleccionar el UR3e
    python3 ur_freedrive.py --axes 0 0 0 1 1 1  # solo rotaciones libres (requiere software reciente)

Enter o Ctrl+C sale de freedrive.
"""
import argparse
import socket
import struct
import threading
import time

ROBOT_IPS = {
    "ur3": "192.168.0.1",
    "ur3e": "192.168.0.3",
}

SCRIPT_PORT = 30002
STREAM_PORT = 30003
STATUS_PERIOD_S = 0.5
Q_ACTUAL_OFFSET = 252  # byte offset de q_actual[6] en el paquete de 30003


def freedrive_script(axes):
    if all(axes):
        call = "freedrive_mode()"
    else:
        # freeAxes/feature: solo en software reciente (5.x)
        call = f"freedrive_mode(freeAxes={list(axes)}, feature=\"base\")"
    return (
        "def fd():\n"
        f"  {call}\n"
        "  while True:\n"
        "    sleep(0.1)\n"
        "  end\n"
        "end\n"
        "fd()\n"
    )


STOP_SCRIPT = (
    "def stop_fd():\n"
    "  end_freedrive_mode()\n"
    "end\n"
    "stop_fd()\n"
)


def recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("stream cerrado por el robot")
        buf += chunk
    return buf


def read_q_actual(sock):
    """Lee un paquete del stream de 30003 y devuelve q_actual (rad)."""
    (length,) = struct.unpack("!i", recv_exact(sock, 4))
    body = recv_exact(sock, length - 4)
    data = struct.pack("!i", length) + body
    return struct.unpack_from("!6d", data, Q_ACTUAL_OFFSET)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--robot",
        choices=(*ROBOT_IPS, "both"),
        default="ur3",
        help="robot to connect to, or both (default: ur3)",
    )
    ap.add_argument(
        "--ip",
        help="override the selected robot's IP address",
    )
    ap.add_argument("--axes", type=int, nargs=6, default=[1, 1, 1, 1, 1, 1], metavar="A",
                    help="ejes libres x y z rx ry rz (1 = libre, 0 = bloqueado)")
    args = ap.parse_args()

    if args.robot == "both" and args.ip:
        ap.error("--ip cannot be used with --robot both")

    targets = (
        list(ROBOT_IPS.items())
        if args.robot == "both"
        else [(args.robot, args.ip or ROBOT_IPS[args.robot])]
    )
    command_sockets = {}
    stream_sockets = {}
    stop = threading.Event()

    try:
        for robot, robot_ip in targets:
            print(f"Connecting to {robot.upper()} at {robot_ip}...")
            command_sockets[robot] = socket.create_connection(
                (robot_ip, SCRIPT_PORT),
                timeout=5,
            )
            stream_sockets[robot] = socket.create_connection(
                (robot_ip, STREAM_PORT),
                timeout=5,
            )

        script = freedrive_script(args.axes).encode()
        for cmd in command_sockets.values():
            cmd.sendall(script)

        threading.Thread(target=lambda: (input(), stop.set()), daemon=True).start()
        print(
            f"Freedrive active on {', '.join(robot.upper() for robot, _ in targets)}; "
            f"axes {args.axes}. Press Enter or Ctrl+C to exit."
        )
        last_print = 0.0
        while not stop.is_set():
            joint_positions = {
                robot: read_q_actual(stream)
                for robot, stream in stream_sockets.items()
            }
            now = time.monotonic()
            if now - last_print >= STATUS_PERIOD_S:
                status = " | ".join(
                    f"{robot.upper()} q [deg]: "
                    + "  ".join(f"{value * 57.2958:7.1f}" for value in joints)
                    for robot, joints in joint_positions.items()
                )
                print(status, end="\r")
                last_print = now
    except KeyboardInterrupt:
        pass
    except (ConnectionError, socket.timeout, OSError) as e:
        print(f"\nConnection error: {e}")
    finally:
        print("\nEnding freedrive and disconnecting...")
        for robot, cmd in command_sockets.items():
            try:
                # Closing a socket alone does not stop freedrive; send URScript to end it.
                cmd.sendall(STOP_SCRIPT.encode())
            except OSError as e:
                print(f"  Could not stop {robot.upper()} cleanly: {e}")
        if command_sockets:
            time.sleep(0.5)
        for cmd in command_sockets.values():
            cmd.close()
        for stream in stream_sockets.values():
            stream.close()
        print("Done.")


if __name__ == "__main__":
    main()