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
SCHEMA = 2  # v2: per-tool state under "tools"; v1 top-level fields kept for the first tool
SCENE_EVERY = 30
MAX_CONTACTS = 32
TOOL_JOINTS = ["joint_roll", "joint_pitch", "joint_gripper_left", "joint_gripper_right"]
# Scenes with several tools prefix each tool's names (dual_tool_scene.xml: left_, right_).
DEFAULT_ARMS = {"left": "justi", "right": "gracie"}

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
    # driven by sim time so the trajectory is reproducible run to run;
    # phase keeps several demo tools from moving in lockstep
    def __init__(self, phase=0.0):
        self.phase = phase

    def spin(self):
        pass

    # jaw is left to the E/D keys, like with the real robot
    def jaw(self, t):
        return None

    def joints(self, t):
        t += self.phase
        return {"shoulder_pan_joint": 0.0, "shoulder_lift_joint": -1.57,
                "elbow_joint": 1.2 + 0.3 * math.sin(0.2 * t), "wrist_1_joint": -1.2,
                "wrist_2_joint": 1.57 + 0.6 * math.sin(0.5 * t),
                "wrist_3_joint": 1.0 * math.sin(0.3 * t)}


class RtdeMaster:
    # reads the UR3e directly over RTDE (port 30004), no ROS needed; receive-only, never commands the arm
    RECONNECT_PERIOD_S = 2.0

    def __init__(self, ip):
        import rtde_receive
        self.r = rtde_receive.RTDEReceiveInterface(ip)
        self.next_retry = 0.0

    def spin(self):
        # the receive link drops when the robot restarts its control script; retry without stalling the sim
        now = time.perf_counter()
        if self.r.isConnected() or now < self.next_retry:
            return
        self.next_retry = now + self.RECONNECT_PERIOD_S
        print("[bridge] RTDE desconectado, reintentando...")
        try:
            if self.r.reconnect():
                print("[bridge] RTDE reconectado")
        except RuntimeError as e:
            print(f"[bridge] reconexion fallo: {e}")

    def jaw(self, t):
        return None

    def joints(self, t):
        if not self.r.isConnected():
            return None
        try:
            return dict(zip(UR_JOINTS, self.r.getActualQ()))
        except RuntimeError:
            return None

    def close(self):
        self.r.disconnect()


class RosHub:
    # one node for every arm: rclpy.init() may only run once per process, and
    # both arms publish on the same /joint_states with "<arm>_" prefixes
    def __init__(self, topic):
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        self.rclpy = rclpy
        rclpy.init()
        self.q = {}
        self.stamp = {}
        self.closed = False
        outer = self

        class N(Node):
            def __init__(self):
                super().__init__("ur3e_mujoco_unity_bridge")
                self.create_subscription(JointState, topic, self.cb, 10)

            def cb(self, msg):
                now = time.perf_counter()
                for name, pos in zip(msg.name, msg.position):
                    outer.q[name] = pos
                    outer.stamp[name] = now

        self.node = N()

    def spin(self):
        self.rclpy.spin_once(self.node, timeout_sec=0.0)

    def close(self):
        if not self.closed:
            self.closed = True
            self.node.destroy_node()
            self.rclpy.shutdown()


class RosMaster:
    def __init__(self, hub, arm):
        self.hub = hub
        self.prefix = f"{arm}_" if arm else ""

    def spin(self):
        self.hub.spin()

    def jaw(self, t):
        return None

    def joints(self, t):
        names = [self.prefix + j for j in UR_JOINTS]
        now = time.perf_counter()
        if not all(now - self.hub.stamp.get(n, -1e9) <= 0.5 for n in names):
            return None
        return {j: self.hub.q[n] for j, n in zip(UR_JOINTS, names)}

    def close(self):
        self.hub.close()


class Teleop:
    def __init__(self, model, mapping, prefix=""):
        self.model = model
        self.prefix = prefix
        self.map = mapping
        self.act = {model.actuator(i).trnid[0]: i for i in range(model.nu)}
        self.clutch = True
        self.q_master_ref = None
        self.q_tool_ref = {}
        self.jaw = 0.0

    def _jid(self, name):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.prefix + name)

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
        self.buttons = {}  # tool prefix -> buttons; "" = sender didn't say which tool
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
            if "buttons" in msg:
                self.buttons[msg.get("tool", "")] = msg["buttons"]
            if msg.get("reset"):
                self.reset = True
            if "ack" in msg:
                self.acks.append((msg["ack"], float(msg.get("hold_ms", 0.0)), t))

    def buttons_for(self, prefix):
        # older Unity scenes send no "tool": their buttons drive every tool
        return self.buttons.get(prefix, self.buttons.get("", {}))

    def close(self):
        self.running = False
        self.sock.close()


def r6(v):
    return [round(float(x), 6) for x in v]


def read_contacts(model, data, names, jaw_ids):
    # jaw_ids: tool prefix -> body ids of that tool's jaws
    out, jaw_force = [], {p: 0.0 for p in jaw_ids}
    f = np.zeros(6)
    for k in range(data.ncon):
        c = data.contact[k]
        b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
        mujoco.mj_contactForce(model, data, k, f)
        for p, ids in jaw_ids.items():
            if b1 in ids or b2 in ids:
                jaw_force[p] += abs(f[0])
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
    def __init__(self, folder, prefixes):
        os.makedirs(folder, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.files = [open(os.path.join(folder, f"{k}_{stamp}.csv"), "w", newline="", buffering=1)
                      for k in ("mujoco", "acks")]
        self.state, self.ack = (csv.writer(f) for f in self.files)
        self.state.writerow(["seq", "t_sim", "t_send", "loop_ms", "step_ms", "n_steps", "bytes",
                             "ncon", "n_contacts_sent"]
                            + [f"{p}{c}" for p in prefixes for c in ["jaw_force"] + TOOL_JOINTS])
        self.ack.writerow(["seq", "t_send", "t_ack", "rtt_ms", "unity_hold_ms", "net_rtt_ms",
                           "est_latency_ms"])
        print(f"[bridge] log -> {os.path.abspath(folder)}  ({stamp})")

    def close(self):
        for f in self.files:
            f.close()


def tool_prefixes(model):
    """Prefix of every tool in the scene, in model order: [""] for hello_unity.xml,
    ["left_", "right_"] for dual_tool_scene.xml."""
    out = []
    for j in range(model.njnt):
        name = model.joint(j).name
        if name.endswith(TOOL_JOINTS[0]):
            out.append(name[:-len(TOOL_JOINTS[0])])
    if not out:
        raise SystemExit(f"[bridge] no '{TOOL_JOINTS[0]}' joint in the model: is this a dVRK tool scene?")
    return out


def parse_pairs(items, what):
    out = {}
    for item in items or []:
        key, sep, val = item.partition("=")
        if not sep:
            raise SystemExit(f"[bridge] {what}: expected TOOL=VALUE, got '{item}'")
        out[key] = val
    return out


class ToolChannel:
    """Everything one simulated tool needs: its master, teleop state and joints."""

    def __init__(self, model, prefix, mapping, master, desc):
        self.prefix, self.master, self.desc = prefix, master, desc
        self.teleop = Teleop(model, mapping, prefix)
        self.adr = [model.joint(prefix + n).qposadr[0] for n in TOOL_JOINTS]
        self.pitch = model.joint(prefix + "joint_pitch")
        self.jaw_left = model.joint(prefix + "joint_gripper_left")
        self.jaw_ids = {i for i in range(model.nbody)
                        if model.body(i).name.startswith(prefix) and "gripper" in model.body(i).name}
        self.qm = None


def make_masters(args, prefixes):
    """One master per tool. Single-tool scenes keep --arm / --robot-ip; multi-tool
    scenes use --arms / --robot-ips (defaults: left=justi, right=gracie)."""
    tools = [p.rstrip("_") for p in prefixes]
    arms = {**DEFAULT_ARMS, **parse_pairs(args.arms, "--arms")}
    ips = parse_pairs(args.robot_ips, "--robot-ips")
    hub, out = None, []
    for k, (prefix, tool) in enumerate(zip(prefixes, tools)):
        if args.demo:
            out.append((DemoMaster(phase=7.0 * k), "DEMO"))
            continue
        ip = args.robot_ip if prefix == "" else ips.get(tool)
        if ip:
            out.append((RtdeMaster(ip), f"UR3e RTDE {ip}"))
            continue
        arm = args.arm if prefix == "" else arms.get(tool, tool)
        hub = hub or RosHub(args.topic)
        out.append((RosMaster(hub, arm), f"UR3e {arm} {args.topic}"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../mjcf/hello_unity.xml")
    ap.add_argument("--arm", default="justi", help="prefijo del brazo maestro: justi | gracie | ''")
    ap.add_argument("--arms", nargs="*", metavar="TOOL=ARM",
                    help="multi-tool scenes: master arm per tool, e.g. left=justi right=gracie (default)")
    ap.add_argument("--topic", default="/joint_states")
    ap.add_argument("--scale", type=float, default=1.0, help="escala de movimiento maestro->esclavo")
    ap.add_argument("--demo", action="store_true", help="maestro sintetico (sin ROS2/robot)")
    ap.add_argument("--robot-ip", help="IP del UR3e: lee las juntas por RTDE directo, sin ROS2")
    ap.add_argument("--robot-ips", nargs="*", metavar="TOOL=IP",
                    help="multi-tool scenes: RTDE IP per tool, e.g. left=192.168.56.101")
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--host", default=UNITY_HOST, help="IP de la maquina con Unity")
    ap.add_argument("--log", metavar="DIR", help="guarda CSVs de estado y acks en DIR")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="segundos de sim y termina (0 = infinito)")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)
    mapping = {k: (v[0], v[1] * args.scale) for k, v in DEFAULT_MAP.items()}
    prefixes = tool_prefixes(model)
    tools = [ToolChannel(model, p, mapping, m, desc)
             for p, (m, desc) in zip(prefixes, make_masters(args, prefixes))]
    masters = list({id(t.master): t.master for t in tools}.values())

    # scenes may define a "home" keyframe (e.g. dual_tool_scene.xml: jaws open, pieces in place)
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")

    def reset():
        if home >= 0:
            mujoco.mj_resetDataKeyframe(model, data, home)
        else:
            mujoco.mj_resetData(model, data)
        for t in tools:
            t.teleop.clutch = True
            # start the jaw command where the scene left the jaws, or they snap shut on the first step
            t.teleop.jaw = float(data.qpos[t.jaw_left.qposadr[0]])

    reset()

    names = [short_name(model.body(i).name) for i in range(model.nbody)]
    body_names = [model.body(i).name for i in range(1, model.nbody)]
    jaw_ids = {t.prefix: t.jaw_ids for t in tools}
    # body poses at qpos0: Unity calibrates against these, not against whatever pose it sees first
    d0 = mujoco.MjData(model)
    mujoco.mj_forward(model, d0)
    rest = body_poses(d0, body_names)

    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx = CmdReceiver(CMD_PORT)
    log = BridgeLog(args.log, prefixes) if args.log else None
    sent = collections.OrderedDict()
    viewer = mujoco.viewer.launch_passive(model, data) if args.viewer else None

    period = 1.0 / SEND_HZ
    max_steps = 4 * int(math.ceil(period / model.opt.timestep))
    maestros = "  ".join(f"{t.prefix or 'tool'}<-{t.desc}" for t in tools)
    print(f"[bridge] maestro={maestros}  "
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
                reset()
                t_start = wall0

            for m in masters:
                m.spin()
            for t in tools:
                t.qm = t.master.joints(data.time)
                t.teleop.update(data, t.qm, rx.buttons_for(t.prefix), period, t.master.jaw(data.time))

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
            tool_state = {}
            for t in tools:
                master_state = None
                if t.qm is not None:
                    qv = [t.qm[j] for j in UR_JOINTS]
                    master_state = {"q": r6(qv), "sing": ur3e_singularity(qv)}
                tool_state[t.prefix] = {
                    "master_ok": t.qm is not None,
                    "clutch": t.teleop.clutch,
                    "master": master_state,
                    "tool_q": r6([data.qpos[a] for a in t.adr]),
                    "pitch_margin": float(1.0 - abs(data.qpos[t.pitch.qposadr[0]]) / t.pitch.range[1]),
                    "jaw_contact_force": float(jaw_force[t.prefix]),
                }

            seq += 1
            state = {
                "v": SCHEMA,
                "seq": seq,
                "t_sim": round(float(data.time), 6),
                **tool_state[tools[0].prefix],  # v1 fields: single-tool Unity scenes read these
                "tools": tool_state,
                "bodies": body_poses(data, body_names),
                "contacts": contacts,
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
                t0 = sent.pop(aseq, None)  # several Unity tools ack the same packet: log the first
                if t0 is None or log is None:
                    continue
                rtt = (t_ack - t0) * 1e3
                log.ack.writerow([aseq, f"{t0:.6f}", f"{t_ack:.6f}", f"{rtt:.3f}", f"{hold:.3f}",
                                  f"{rtt - hold:.3f}", f"{(rtt - hold) / 2 + hold:.3f}"])

            loop_ms = (time.perf_counter() - wall0) * 1e3
            if log is not None:
                per_tool = []
                for t in tools:
                    ts = tool_state[t.prefix]
                    per_tool += [f"{ts['jaw_contact_force']:.4f}"] + [f"{q:.6f}" for q in ts["tool_q"]]
                log.state.writerow([seq, f"{data.time:.6f}", f"{t_send:.6f}", f"{loop_ms:.3f}",
                                    f"{step_ms:.3f}", n, len(payload), data.ncon, len(contacts)] + per_tool)

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
        for m in masters:
            if hasattr(m, "close"):
                m.close()


if __name__ == "__main__":
    main()
