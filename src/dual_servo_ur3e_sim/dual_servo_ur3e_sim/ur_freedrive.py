"""Freedrive del UR3 por sockets (sin ROS, sin ur_rtde).

Manda URScript por el puerto 30002 y lee las juntas del stream de 30003.
Funciona en CB3 y e-Series (en e-Series, ponerlo en Remote Control si hace falta).

    python3 ur_freedrive.py                     # todas las juntas libres
    python3 ur_freedrive.py --axes 0 0 0 1 1 1  # solo rotaciones libres (requiere software reciente)

Enter o Ctrl+C sale de freedrive.
"""
import argparse
import socket
import struct
import threading
import time

ROBOT_IP = "192.168.0.3"  # UR3e
# ROBOT_IP = "192.168.0.1"  # UR3

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
    ap.add_argument("--ip", default=ROBOT_IP)
    ap.add_argument("--axes", type=int, nargs=6, default=[1, 1, 1, 1, 1, 1], metavar="A",
                    help="ejes libres x y z rx ry rz (1 = libre, 0 = bloqueado)")
    args = ap.parse_args()

    print(f"Connecting to robot at {args.ip}...")
    cmd = socket.create_connection((args.ip, SCRIPT_PORT), timeout=5)
    stream = socket.create_connection((args.ip, STREAM_PORT), timeout=5)

    stop = threading.Event()
    threading.Thread(target=lambda: (input(), stop.set()), daemon=True).start()

    try:
        cmd.sendall(freedrive_script(args.axes).encode())
        print(f"Freedrive activo, ejes libres {args.axes}. Enter o Ctrl+C para salir.")
        last_print = 0.0
        while not stop.is_set():
            q = read_q_actual(stream)  # bloquea ~8 ms (125 Hz en CB3)
            now = time.monotonic()
            if now - last_print >= STATUS_PERIOD_S:
                print("  q [deg]: " + "  ".join(f"{v * 57.2958:7.1f}" for v in q), end="\r")
                last_print = now
    except KeyboardInterrupt:
        pass
    except (ConnectionError, socket.timeout, OSError) as e:
        print(f"\nError de conexion: {e}")
    finally:
        print("\nEnding freedrive and disconnecting...")
        try:
            # enviar un script nuevo reemplaza al que corre; cerrar el socket NO detiene freedrive
            cmd.sendall(STOP_SCRIPT.encode())
            time.sleep(0.5)
        except OSError as e:
            print(f"  (no se pudo cerrar limpio: {e})")
        cmd.close()
        stream.close()
        print("Done.")


if __name__ == "__main__":
    main()