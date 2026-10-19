import argparse
import json
import math
import socket
import time

import mujoco
import mujoco.viewer
import numpy as np

UNITY_HOST = "127.0.0.1"
STATE_PORT = 5005
CMD_PORT = 5006
SEND_HZ = 60.0

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


class DemoMaster:
    def __init__(self):
        self.t0 = time.perf_counter()

    def spin(self):
        pass

    def joints(self):
        t = time.perf_counter() - self.t0
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

    def joints(self):
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

    def update(self, data, qm, buttons, dt):
        if buttons.get("jaw_open"):
            self.jaw = min(JAW_MAX, self.jaw + JAW_SPEED * dt)
        if buttons.get("jaw_close"):
            self.jaw = max(0.0, self.jaw - JAW_SPEED * dt)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../mjcf/hello_unity.xml")
    ap.add_argument("--arm", default="justi", help="prefijo del brazo maestro: justi | gracie | ''")
    ap.add_argument("--topic", default="/joint_states")
    ap.add_argument("--scale", type=float, default=1.0, help="escala de movimiento maestro->esclavo")
    ap.add_argument("--demo", action="store_true", help="maestro sintetico (sin ROS2/robot)")
    ap.add_argument("--viewer", action="store_true")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)
    mapping = {k: (v[0], v[1] * args.scale) for k, v in DEFAULT_MAP.items()}
    teleop = Teleop(model, mapping)
    master = DemoMaster() if args.demo else RosMaster(args.arm, args.topic)

    body_names = [model.body(i).name for i in range(1, model.nbody)]
    joint_names = [model.joint(i).name for i in range(model.njnt)]
    jaw_ids = {i for i in range(model.nbody) if "gripper" in model.body(i).name}

    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("0.0.0.0", CMD_PORT))
    rx.setblocking(False)
    buttons = {}
    viewer = mujoco.viewer.launch_passive(model, data) if args.viewer else None

    period = 1.0 / SEND_HZ
    n_steps = max(1, int(round(period / model.opt.timestep)))
    pitch_j = model.joint("joint_pitch")
    print(f"[bridge] maestro={'DEMO' if args.demo else 'UR3e ' + args.arm + ' ' + args.topic}  "
          f"estado->:{STATE_PORT}  botones<-:{CMD_PORT}")

    try:
        while viewer is None or viewer.is_running():
            wall0 = time.perf_counter()

            while True:
                try:
                    msg = json.loads(rx.recvfrom(4096)[0].decode())
                    buttons = msg.get("buttons", buttons)
                    if msg.get("reset"):
                        mujoco.mj_resetData(model, data)
                        teleop.clutch, teleop.jaw = True, 0.0
                except BlockingIOError:
                    break
                except ValueError:
                    break

            master.spin()
            qm = master.joints()
            teleop.update(data, qm, buttons, period)

            for _ in range(n_steps):
                mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.sync()

            jaw_force = 0.0
            for k in range(data.ncon):
                c = data.contact[k]
                if model.geom_bodyid[c.geom1] in jaw_ids or model.geom_bodyid[c.geom2] in jaw_ids:
                    f = np.zeros(6)
                    mujoco.mj_contactForce(model, data, k, f)
                    jaw_force += abs(f[0])

            master_state = None
            if qm is not None:
                qv = [qm[j] for j in UR_JOINTS]
                master_state = {"q": qv, "sing": ur3e_singularity(qv)}

            state = {
                "t": float(data.time),
                "master_ok": qm is not None,
                "clutch": teleop.clutch,
                "master": master_state,
                "q": {n: float(data.qpos[model.joint(n).qposadr[0]]) for n in joint_names},
                "bodies": {n: pos_to_unity(data.xpos[i + 1]) + quat_to_unity(data.xquat[i + 1])
                           for i, n in enumerate(body_names)},
                "pitch_margin": float(1.0 - abs(data.qpos[pitch_j.qposadr[0]]) / pitch_j.range[1]),
                "jaw_contact_force": float(jaw_force),
                "ncon": int(data.ncon),
            }
            tx.sendto(json.dumps(state).encode(), (UNITY_HOST, STATE_PORT))

            dt = time.perf_counter() - wall0
            if dt < period:
                time.sleep(period - dt)
    except KeyboardInterrupt:
        print("\n[bridge] detenido")
    finally:
        if viewer is not None:
            viewer.close()
        if hasattr(master, "close"):
            master.close()


if __name__ == "__main__":
    main()
