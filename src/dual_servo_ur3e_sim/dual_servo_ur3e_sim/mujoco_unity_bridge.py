import argparse
import collections
import csv
import json
import math
import os
import socket
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np

UNITY_HOST = "127.0.0.1"
STATE_PORT = 5005
CMD_PORT = 5006
SEND_HZ = 60.0
SCHEMA = 1
SCENE_EVERY = 30
MAX_CONTACTS = 32
TOOL_JOINTS = ["joint_roll", "joint_pitch", "joint_gripper_left", "joint_gripper_right"]

UR_JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
             "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

DEFAULT_MAP = {"joint_roll": ("wrist_3_joint", 1.0),
               "joint_pitch": ("wrist_2_joint", 1.0)}

JAW_SPEED = 1.5
JAW_MAX = 1.0472

UR3E_DH = {"d": [0.15185, 0, 0, 0.13105, 0.08535, 0.0921],
           "a": [0, -0.24355, -0.2132, 0, 0, 0],
           "alpha": [math.pi / 2, 0, 0, math.pi / 2, -math.pi / 2, 0]}


def pos_to_unity(p):
    return [float(p[0]), float(p[2]), float(p[1])]


def quat_to_unity(q):
    w, x, y, z = q
    return [float(-x), float(-z), float(-y), float(w)]


def ur3e_fk_frames(q):
    T = np.eye(4)
    frames = [T.copy()]
    for i in range(6):
        d, a, al = UR3E_DH["d"][i], UR3E_DH["a"][i], UR3E_DH["alpha"][i]
        ct, st = math.cos(q[i]), math.sin(q[i])
        ca, sa = math.cos(al), math.sin(al)
        A = np.array([[ct, -st * ca, st * sa, a * ct],
                      [st, ct * ca, -ct * sa, a * st],
                      [0, sa, ca, d],
                      [0, 0, 0, 1]])
        T = T @ A
        frames.append(T.copy())
    return frames


def ur3e_jacobian(q):
    F = ur3e_fk_frames(q)
    pe = F[6][:3, 3]
    J = np.zeros((6, 6))
    for i in range(6):
        z, p = F[i][:3, 2], F[i][:3, 3]
        J[:3, i] = np.cross(z, pe - p)
        J[3:, i] = z
    return J


def ur3e_singularity(q):
    wrist = abs(math.sin(q[4]))
    elbow = abs(math.sin(q[2]))
    F = ur3e_fk_frames(q)
    wc = F[4][:3, 3]
    shoulder = min(1.0, math.hypot(wc[0], wc[1]) / 0.10)
    J = ur3e_jacobian(q)
    s = np.linalg.svd(J, compute_uv=False)
    return {"wrist": wrist, "elbow": elbow, "shoulder": shoulder,
            "sigma_min": float(s[-1]), "cond": float(s[0] / max(s[-1], 1e-9))}


def short_name(n):
    i = n.find("_Shape")
    return n[:i] if i > 0 else n


class DemoMaster:
    # driven by sim time so the trajectory is reproducible run to run
    def spin(self):
        pass

    # jaw is left to the E/D keys, like with the real robot
    def jaw(self, t):
        return None

    def joints(self, t):
        return {"shoulder_pan_joint": 0.0, "shoulder_lift_joint": -1.57,
                "elbow_joint": 1.2 + 0.3 * math.sin(0.2 * t), "wrist_1_joint": -1.2,
                "wrist_2_joint": 1.57 + 0.6 * math.sin(0.5 * t),
                "wrist_3_joint": 1.0 * math.sin(0.3 * t)}


class RosMaster:
    def __init__(self, arm, topic):
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        self.rclpy = rclpy
        rclpy.init()
        self.prefix = f"{arm}_" if arm else ""
        self.q = {}
        self.stamp = 0.0
        outer = self

        class N(Node):
            def __init__(self):
                super().__init__("ur3e_mujoco_unity_bridge")
                self.create_subscription(JointState, topic, self.cb, 10)

            def cb(self, msg):
                for name, pos in zip(msg.name, msg.position):
                    if name.startswith(outer.prefix):
                        outer.q[name[len(outer.prefix):]] = pos
                outer.stamp = time.perf_counter()

        self.node = N()

    def spin(self):
        self.rclpy.spin_once(self.node, timeout_sec=0.0)

    def jaw(self, t):
        return None

    def joints(self, t):
        if time.perf_counter() - self.stamp > 0.5 or not all(j in self.q for j in UR_JOINTS):
            return None
        return dict(self.q)

    def close(self):
        self.node.destroy_node()
        self.rclpy.shutdown()


class Teleop:
    def __init__(self, model, mapping):
        self.model = model
        self.map = mapping
        self.act = {model.actuator(i).trnid[0]: i for i in range(model.nu)}
        self.clutch = True
        self.q_master_ref = None
        self.q_tool_ref = {}
        self.jaw = 0.0

    def _jid(self, name):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)

    def update(self, data, qm, buttons, dt, jaw_cmd=None):
        if buttons.get("jaw_open"):
            self.jaw = min(JAW_MAX, self.jaw + JAW_SPEED * dt)
        elif buttons.get("jaw_close"):
            self.jaw = max(0.0, self.jaw - JAW_SPEED * dt)
        elif jaw_cmd is not None:
            self.jaw = jaw_cmd
        for jn in ("joint_gripper_left", "joint_gripper_right"):
            j = self._jid(jn)
            if j >= 0 and j in self.act:
                data.ctrl[self.act[j]] = self.jaw

        want_clutch = bool(buttons.get("clutch")) or qm is None
        if want_clutch:
            self.clutch = True
            return
        if self.clutch:
            self.clutch = False
            self.q_master_ref = dict(qm)
            self.q_tool_ref = {tj: float(data.ctrl[self.act[self._jid(tj)]]) for tj in self.map
                               if self._jid(tj) in self.act}
        for tj, (mj, scale) in self.map.items():
            j = self._jid(tj)
            if j < 0 or j not in self.act or mj not in qm:
                continue
            a = self.act[j]
            target = self.q_tool_ref[tj] + scale * (qm[mj] - self.q_master_ref[mj])
            lo, hi = self.model.actuator_ctrlrange[a]
            data.ctrl[a] = float(np.clip(target, lo, hi))


class CmdReceiver:
    # own thread so acks are timestamped on arrival, not when the main loop wakes up
    def __init__(self, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", port))
        self.sock.settimeout(0.2)
        self.buttons = {}
        self.reset = False
        self.acks = collections.deque()
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            try:
                raw = self.sock.recvfrom(4096)[0]
            except socket.timeout:
                continue
            except OSError:
                return
            t = time.perf_counter()
            try:
                msg = json.loads(raw.decode())
            except ValueError:
                continue
            self.buttons = msg.get("buttons", self.buttons)
            if msg.get("reset"):
                self.reset = True
            if "ack" in msg:
                self.acks.append((msg["ack"], float(msg.get("hold_ms", 0.0)), t))

    def close(self):
        self.running = False
        self.sock.close()


def r6(v):
    return [round(float(x), 6) for x in v]


def read_contacts(model, data, names, jaw_ids):
    out, jaw_force = [], 0.0
    f = np.zeros(6)
    for k in range(data.ncon):
        c = data.contact[k]
        b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
        mujoco.mj_contactForce(model, data, k, f)
        if b1 in jaw_ids or b2 in jaw_ids:
            jaw_force += abs(f[0])
        if len(out) < MAX_CONTACTS:
            fw = c.frame.reshape(3, 3).T @ f[:3]  # contact frame -> world
            out.append([names[b1], names[b2]] + r6(pos_to_unity(c.pos))
                       + [round(float(f[0]), 4)] + r6(pos_to_unity(fw)))
    return out, jaw_force


PRIM_TYPES = {int(mujoco.mjtGeom.mjGEOM_BOX): "box", int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere"}


def body_poses(data, body_names):
    return {n: r6(pos_to_unity(data.xpos[i + 1]) + quat_to_unity(data.xquat[i + 1]))
            for i, n in enumerate(body_names)}


def scene_prims(model, data):
    # primitive geoms not in the Unity prefab (e.g. tissue); group 3 = collision-only, skipped
    out = []
    q = np.zeros(4)
    for g in range(model.ngeom):
        kind = PRIM_TYPES.get(int(model.geom_type[g]))
        if kind is None or model.geom_group[g] >= 3:
            continue
        s = model.geom_size[g]
        mujoco.mju_mat2Quat(q, data.geom_xmat[g])
        out.append({"name": model.geom(g).name, "type": kind,
                    "size": r6([s[0], s[2], s[1]]),
                    "pose": r6(pos_to_unity(data.geom_xpos[g]) + quat_to_unity(q)),
                    "rgba": [round(float(v), 3) for v in model.geom_rgba[g]]})
    return out


class BridgeLog:
    def __init__(self, folder):
        os.makedirs(folder, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.files = [open(os.path.join(folder, f"{k}_{stamp}.csv"), "w", newline="", buffering=1)
                      for k in ("mujoco", "acks")]
        self.state, self.ack = (csv.writer(f) for f in self.files)
        self.state.writerow(["seq", "t_sim", "t_send", "loop_ms", "step_ms", "n_steps", "bytes",
                             "ncon", "n_contacts_sent", "jaw_force"] + TOOL_JOINTS)
        self.ack.writerow(["seq", "t_send", "t_ack", "rtt_ms", "unity_hold_ms", "net_rtt_ms",
                           "est_latency_ms"])
        print(f"[bridge] log -> {os.path.abspath(folder)}  ({stamp})")

    def close(self):
        for f in self.files:
            f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../mjcf/hello_unity.xml")
    ap.add_argument("--arm", default="justi", help="prefijo del brazo maestro: justi | gracie | ''")
    ap.add_argument("--topic", default="/joint_states")
    ap.add_argument("--scale", type=float, default=1.0, help="escala de movimiento maestro->esclavo")
    ap.add_argument("--demo", action="store_true", help="maestro sintetico (sin ROS2/robot)")
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--host", default=UNITY_HOST, help="IP de la maquina con Unity")
    ap.add_argument("--log", metavar="DIR", help="guarda CSVs de estado y acks en DIR")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="segundos de sim y termina (0 = infinito)")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)
    mapping = {k: (v[0], v[1] * args.scale) for k, v in DEFAULT_MAP.items()}
    teleop = Teleop(model, mapping)
    master = DemoMaster() if args.demo else RosMaster(args.arm, args.topic)

    names = [short_name(model.body(i).name) for i in range(model.nbody)]
    body_names = [model.body(i).name for i in range(1, model.nbody)]
    jaw_ids = {i for i in range(model.nbody) if "gripper" in model.body(i).name}
    tool_adr = [model.joint(n).qposadr[0] for n in TOOL_JOINTS]
    # body poses at qpos0: Unity calibrates against these, not against whatever pose it sees first
    d0 = mujoco.MjData(model)
    mujoco.mj_forward(model, d0)
    rest = body_poses(d0, body_names)

    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx = CmdReceiver(CMD_PORT)
    log = BridgeLog(args.log) if args.log else None
    sent = collections.OrderedDict()
    viewer = mujoco.viewer.launch_passive(model, data) if args.viewer else None

    period = 1.0 / SEND_HZ
    max_steps = 4 * int(math.ceil(period / model.opt.timestep))
    pitch_j = model.joint("joint_pitch")
    print(f"[bridge] maestro={'DEMO' if args.demo else 'UR3e ' + args.arm + ' ' + args.topic}  "
          f"estado->{args.host}:{STATE_PORT}  botones<-:{CMD_PORT}")

    seq = 0
    t_start = next_tick = time.perf_counter()
    try:
        while viewer is None or viewer.is_running():
            wall0 = time.perf_counter()
            if args.duration and data.time >= args.duration:
                break
            if rx.reset:
                rx.reset = False
                mujoco.mj_resetData(model, data)
                teleop.clutch, teleop.jaw = True, 0.0
                t_start = wall0

            master.spin()
            qm = master.joints(data.time)
            teleop.update(data, qm, rx.buttons, period, master.jaw(data.time))

            # real-time sync: step until sim time catches up with wall time
            step0 = time.perf_counter()
            n = 0
            while data.time < wall0 - t_start and n < max_steps:
                mujoco.mj_step(model, data)
                n += 1
            if n == max_steps:
                t_start = wall0 - data.time  # can't keep up: fall behind instead of spiraling
            step_ms = (time.perf_counter() - step0) * 1e3
            if viewer is not None:
                viewer.sync()

            contacts, jaw_force = read_contacts(model, data, names, jaw_ids)
            tool_q = [float(data.qpos[a]) for a in tool_adr]
            master_state = None
            if qm is not None:
                qv = [qm[j] for j in UR_JOINTS]
                master_state = {"q": r6(qv), "sing": ur3e_singularity(qv)}

            seq += 1
            state = {
                "v": SCHEMA,
                "seq": seq,
                "t_sim": round(float(data.time), 6),
                "master_ok": qm is not None,
                "clutch": teleop.clutch,
                "master": master_state,
                "tool_q": r6(tool_q),
                "bodies": body_poses(data, body_names),
                "contacts": contacts,
                "pitch_margin": float(1.0 - abs(data.qpos[pitch_j.qposadr[0]]) / pitch_j.range[1]),
                "jaw_contact_force": float(jaw_force),
                "ncon": int(data.ncon),
            }
            if seq % SCENE_EVERY == 1:
                state["scene"] = scene_prims(model, data)
                state["rest"] = rest
            # same clock as Unity's Stopwatch on Windows (QueryPerformanceCounter)
            t_send = state["t_send"] = time.perf_counter()
            payload = json.dumps(state, separators=(",", ":")).encode()
            tx.sendto(payload, (args.host, STATE_PORT))

            sent[seq] = t_send
            if len(sent) > 600:
                sent.popitem(last=False)
            while rx.acks:
                aseq, hold, t_ack = rx.acks.popleft()
                t0 = sent.pop(aseq, None)
                if t0 is None or log is None:
                    continue
                rtt = (t_ack - t0) * 1e3
                log.ack.writerow([aseq, f"{t0:.6f}", f"{t_ack:.6f}", f"{rtt:.3f}", f"{hold:.3f}",
                                  f"{rtt - hold:.3f}", f"{(rtt - hold) / 2 + hold:.3f}"])

            loop_ms = (time.perf_counter() - wall0) * 1e3
            if log is not None:
                log.state.writerow([seq, f"{data.time:.6f}", f"{t_send:.6f}", f"{loop_ms:.3f}",
                                    f"{step_ms:.3f}", n, len(payload), data.ncon, len(contacts),
                                    f"{jaw_force:.4f}"] + [f"{q:.6f}" for q in tool_q])

            # absolute deadlines: sleep overshoot doesn't accumulate into rate drift
            next_tick += period
            wait = next_tick - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            else:
                next_tick = time.perf_counter()
    except KeyboardInterrupt:
        print("\n[bridge] detenido")
    finally:
        rx.close()
        if log is not None:
            log.close()
        if viewer is not None:
            viewer.close()
        if hasattr(master, "close"):
            master.close()


if __name__ == "__main__":
    main()
