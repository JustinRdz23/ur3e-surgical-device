"""Dual-arm teleop + haptics for teleop_dual_tool_scene.xml (ur_rtde, no ROS).

Each robot's TCP pose drives a mocap body (left_arm_mocap / right_arm_mocap) that is welded
to a tool in the scene.  forceMode() with a zero wrench keeps both arms freely movable by hand;
contact forces on each tool are sent back as that robot's wrench.

    python3 dual_arm_teleop_haptic.py                 # both arms, haptics on
    python3 dual_arm_teleop_haptic.py --no-haptics    # just teleop (zero wrench)
    python3 dual_arm_teleop_haptic.py --force-scale 0.1

Ctrl+C or closing the viewer stops force mode on both robots and disconnects.
"""
import argparse
import signal
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer
import rtde_control
import rtde_receive

LEFT_IP = "192.168.0.1"    # UR3
RIGHT_IP = "192.168.0.3"   # UR3e

MJCF_PATH = str(Path(__file__).resolve().parent.parent / "mjcf" / "teleop_dual_tool_scene.xml")
MOCAP_BODIES = {"left": "left_arm_mocap", "right": "right_arm_mocap"}

# forceMode settings (same as the single-arm script that works)
TASK_FRAME = [0, 0, 0, 0, 0, 0]
SELECTION_VECTOR = [1, 1, 1, 1, 1, 1]
FORCE_TYPE = 2
LIMITS = [0.1, 0.1, 1.5, 3.14, 3.14, 0.5]
DAMPING = 0.004
GAIN_SCALING = 1.0

CONTROL_PERIOD_S = 0.002
RENDER_EVERY_N_LOOPS = 10

# Fixed 180-degree rotation about Y, quaternion (w, x, y, z), robot base -> MuJoCo world.
Q_OFFSET = np.array([0.0, 0.0, 1.0, 0.0])
Q_OFFSET_INV = np.array([Q_OFFSET[0], -Q_OFFSET[1], -Q_OFFSET[2], -Q_OFFSET[3]])

# Where each robot's base origin sits in MuJoCo world coordinates.  CALIBRATE these.
FRAME_OFFSETS = {
    "left": np.array([0.0, 0.0, 0.8]),
    "right": np.array([0.0, 0.0, 0.8]),
}

FORCE_SCALE = 0.4      # fraction of the simulated contact force rendered on the robot
FORCE_FILTER = 0.5     # low-pass per loop (1 = no filtering)
MAX_FORCE_N = 10.0     # hard clamp per axis sent to the robot

ARROW_LENGTH_SCALE = 0.01
ARROW_WIDTH = 0.004


def get_mocap_body(model, body_name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body not found: {body_name}")
    if model.body_mocapid[body_id] < 0:
        raise ValueError(f"Body is not a mocap body: {body_name}")
    return body_id, model.body_mocapid[body_id]


def find_welded_body(model, body_id):
    """Body that the given mocap body is welded to (via <equality><weld>)."""
    for i in range(model.neq):
        if model.eq_type[i] != mujoco.mjtEq.mjEQ_WELD:
            continue
        a, b = int(model.eq_obj1id[i]), int(model.eq_obj2id[i])
        if a == body_id:
            return b
        if b == body_id:
            return a
    raise ValueError(f"No weld constraint found for mocap body id {body_id}")


def descendants(model, root):
    """Body ids of root and everything below it (children always have larger ids)."""
    ids = {root}
    for b in range(root + 1, model.nbody):
        if model.body_parentid[b] in ids:
            ids.add(b)
    return ids


def transform_pose(pose, frame_offset):
    pos = np.zeros(3)
    mujoco.mju_rotVecQuat(pos, np.array(pose[:3]), Q_OFFSET)
    pos += frame_offset

    rot_vec = np.array(pose[3:6])
    angle = np.linalg.norm(rot_vec)
    quat_robot = np.array([1.0, 0.0, 0.0, 0.0])
    if angle > 1e-9:
        mujoco.mju_axisAngle2Quat(quat_robot, rot_vec / angle, angle)

    quat = np.zeros(4)
    mujoco.mju_mulQuat(quat, Q_OFFSET, quat_robot)
    return pos, quat


class Arm:
    def __init__(self, model, side, ip):
        self.side, self.ip = side, ip
        body_id, self.mocap_id = get_mocap_body(model, MOCAP_BODIES[side])
        tool_root = find_welded_body(model, body_id)
        self.tool_bodies = descendants(model, tool_root)
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, tool_root)
        print(f"[{side}] mocap '{MOCAP_BODIES[side]}' is welded to '{name}' "
              f"({len(self.tool_bodies)} bodies in the tool)")

        self.wrench = [0.0] * 6
        self.f_filt = np.zeros(3)
        self.arrow_start = None
        self.arrow_vec = np.zeros(3)
        self.rtde_c = None
        self.rtde_r = None

    def connect(self):
        print(f"[{self.side}] connecting to {self.ip}...")
        self.rtde_c = rtde_control.RTDEControlInterface(self.ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(self.ip)

    def setup_force_mode(self):
        self.rtde_c.zeroFtSensor()
        self.rtde_c.forceModeSetDamping(DAMPING)
        self.rtde_c.forceModeSetGainScaling(GAIN_SCALING)

    def send_force(self):
        self.rtde_c.forceMode(TASK_FRAME, SELECTION_VECTOR, self.wrench, FORCE_TYPE, LIMITS)

    def update_mocap(self, data):
        pose = self.rtde_r.getActualTCPPose()
        pos, quat = transform_pose(pose, FRAME_OFFSETS[self.side])
        data.mocap_pos[self.mocap_id] = pos
        data.mocap_quat[self.mocap_id] = quat

    def compute_force(self, model, data, enabled):
        total = np.zeros(3)
        point_acc = np.zeros(3)
        weight = 0.0
        res = np.zeros(6)

        for i in range(data.ncon):
            c = data.contact[i]
            in1 = model.geom_bodyid[c.geom1] in self.tool_bodies
            in2 = model.geom_bodyid[c.geom2] in self.tool_bodies
            if in1 == in2:        # no tool involved, or the tool touching itself
                continue
            mujoco.mj_contactForce(model, data, i, res)
            f = np.array(c.frame).reshape(3, 3).T @ res[:3]   # force on geom2, world frame
            if in1:               # tool is geom1: take the reaction
                f = -f
            total += f
            n = np.linalg.norm(f)
            point_acc += np.array(c.pos) * n
            weight += n

        self.arrow_start = point_acc / weight if weight > 1e-9 else None
        self.arrow_vec = total

        if not enabled:
            self.wrench[0:3] = [0.0, 0.0, 0.0]
            return

        # Rotation only (no translation) for a force: world -> robot base frame.
        f_robot = np.zeros(3)
        mujoco.mju_rotVecQuat(f_robot, total, Q_OFFSET_INV)
        self.f_filt += FORCE_FILTER * (f_robot - self.f_filt)
        self.wrench[0:3] = np.clip(self.f_filt * FORCE_SCALE, -MAX_FORCE_N, MAX_FORCE_N).tolist()

    def close(self):
        if self.rtde_c is not None:
            try:
                self.rtde_c.forceModeStop()
                self.rtde_c.stopScript()
            except Exception as e:
                print(f"[{self.side}] error stopping force mode: {e}")
        if self.rtde_r is not None:
            try:
                self.rtde_r.disconnect()
            except Exception as e:
                print(f"[{self.side}] error disconnecting: {e}")


def draw_arrows(viewer, arms):
    scn = viewer.user_scn
    scn.ngeom = 0
    for arm in arms:
        if arm.arrow_start is None or np.linalg.norm(arm.arrow_vec) < 1e-6:
            continue
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                            np.eye(3).flatten(), np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, ARROW_WIDTH,
                             arm.arrow_start, arm.arrow_start + arm.arrow_vec * ARROW_LENGTH_SCALE)
        scn.ngeom += 1


def main():
    global FORCE_SCALE
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=MJCF_PATH)
    ap.add_argument("--no-haptics", action="store_true", help="teleop only, send zero wrench")
    ap.add_argument("--force-scale", type=float, default=FORCE_SCALE)
    args = ap.parse_args()
    FORCE_SCALE = args.force_scale
    haptics = not args.no_haptics

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    arms = [Arm(model, "left", LEFT_IP), Arm(model, "right", RIGHT_IP)]
    steps_per_loop = max(1, round(CONTROL_PERIOD_S / model.opt.timestep))

    running = True

    def handle_sigint(signum, frame):
        nonlocal running
        print("\nCtrl+C received, stopping...")
        running = False

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        for arm in arms:
            arm.connect()
        time.sleep(1.0)
        for arm in arms:
            arm.setup_force_mode()
        print("Connected. Both arms become compliant now, so support them.")

        with mujoco.viewer.launch_passive(model, data) as viewer:
            print(f"Dual-arm loop running (haptics {'ON' if haptics else 'OFF'}).")
            loop = 0
            while running and viewer.is_running():
                loop_start = time.perf_counter()

                # Compliant mode: must be re-sent every loop, as in the single-arm script.
                for arm in arms:
                    arm.send_force()

                with viewer.lock():
                    for arm in arms:
                        arm.update_mocap(data)
                    for _ in range(steps_per_loop):
                        mujoco.mj_step(model, data)
                    for arm in arms:
                        arm.compute_force(model, data, haptics)

                loop += 1
                if loop % RENDER_EVERY_N_LOOPS == 0:
                    with viewer.lock():
                        draw_arrows(viewer, arms)
                    viewer.sync()

                remaining = CONTROL_PERIOD_S - (time.perf_counter() - loop_start)
                if remaining > 0:
                    time.sleep(remaining)

    finally:
        print("Stopping force mode and disconnecting...")
        for arm in arms:
            arm.close()
        print("Done.")


if __name__ == "__main__":
    main()