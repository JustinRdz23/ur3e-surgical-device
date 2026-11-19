import json
import signal
import socket
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import rtde_control
import rtde_receive
from IK_solver import JOINT_SUFFIXES, PoseIK, mat_to_quat
from mujoco_unity_bridge import SCHEMA, STATE_PORT, body_poses, r6
from pynput import keyboard
from scipy.spatial.transform import Rotation as R

# =====================================================================
# Robot configuration
# =====================================================================

LEFT_IP = "192.168.0.1"
RIGHT_IP = "192.168.0.3"

MJCF_PATH = str(Path(__file__).resolve().parent.parent / "mjcf" / "teleop_dual_IK.xml")

MOCAP_BODIES = {
    "left": "left_arm_mocap",
    "right": "right_arm_mocap",
}

PORT_BODIES = {
    "left": "left_port",
    "right": "right_port",
}

# =====================================================================
# Symmetric 1-DOF gripper toggle & grip strength configuration
# =====================================================================

GRIPPER_CLOSED_POS = -0.05
GRIPPER_OPEN_POS = 0.3
GRIPPER_INITIAL_OPEN = {"left": True, "right": True}

JAW_SUFFIX_SIGNS = {
    "joint_gripper_left": 1.0,
    "joint_gripper_right": 1.0,
}

GRIP_KP_SCALE = 3.0
GRIP_FORCE_SCALE = 4.0
JAW_CONDIM = 4
JAW_FRICTION = np.array([2.0, 0.05, 0.01])

# =====================================================================
# Unity bridge configuration
# =====================================================================

UNITY_HOST = "192.168.0.101"
STATE_SEND_HZ = 60.0

UNITY_TOOL_JOINTS = [
    "joint_roll",
    "joint_pitch",
    "joint_gripper_left",
    "joint_gripper_right",
]

UNITY_PORT_JOINTS = [
    "port_yaw",
    "port_pitch",
    "port_insert",
]

UNITY_BODY_SUFFIXES = (
    "tool_main_insert",
    "tool_roll_link",
    "tool_pitch_link",
    "tool_yaw_link",
    "gripper_left",
    "gripper_right",
)

# =====================================================================
# forceMode configuration
# =====================================================================

TASK_FRAME = [0, 0, 0, 0, 0, 0]
SELECTION_VECTOR = [1, 1, 1, 1, 1, 1]
WRENCH = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
FORCE_TYPE = 2
LIMITS = [0.1, 0.1, 1.5, 1.5, 1.5, 0.5]
DAMPING = 0.004
GAIN_SCALING = 1.0

CONTROL_PERIOD_S = 0.008

# =====================================================================
# Coordinate transforms
# =====================================================================

Q_OFFSET = np.array([0.0, 0.0, 1.0, 0.0])

FRAME_OFFSETS = {
    "left": np.array([-0.3, 0.0, 0.8]),
    "right": np.array([0.3, 0.0, 0.8]),
}

R_WORLD_H = {
    "left": np.diag([-1.0, 1.0, -1.0]),
    "right": np.diag([-1.0, 1.0, -1.0]),
}

SCALE = 0.5

# =====================================================================
# Haptic feedback (Tip-aware God-object proxy + virtual coupling fallback)
# =====================================================================

HAPTICS_ENABLED = True
GOD_OBJECT_ENABLED = True
TIP_HAPTICS_ENABLED = True

# Distance (m) from each jaw body origin toward its tip along local +X (adjust to match jaw length)
TIP_FORWARD_OFFSET_M = 0.012

STIFFNESS_LIN = 4000.0      # N/m
DAMPING_LIN = 20.0        # N/(m/s)
GAP_DEADBAND_M = 0.0005     # m
PROXY_RADIUS_M = 0.015      # m expansion around obstacle surfaces in {RB}
INTERNAL_MAX_FORCE = 100.0  # N per axis
FORCE_SCALE = 0.6           # gain from {RB} force to hand/UR frame
MAX_FORCE = 100.0           # N per axis

# Optional obstacle definitions per arm in {RB} frame:
# Each box entry: {"center": np.array([x, y, z]), "half_size": np.array([hx, hy, hz])}
GOD_OBJECT_BOXES_RB = {
    "left": [],
    "right": [],
}

# Optional floor height in {RB} frame per arm (None to disable)
GOD_OBJECT_FLOOR_Z_RB = {
    "left": None,
    "right": None,
}

HAPTIC_STARTUP_DELAY_S = 2.0
HAPTIC_RAMP_S = 1.0
HAPTIC_MAX_DT_S = 0.05
HAPTIC_PRINT_EVERY_N_LOOPS = 100

# =====================================================================
# Torque haptics (rotational virtual coupling)
# =====================================================================

TORQUE_HAPTICS_ENABLED = True

STIFFNESS_ROT = 6.0
DAMPING_ROT = 0.05
ROT_DEADBAND_RAD = 0.05
TORQUE_VEL_FILTER = 0.3
INTERNAL_MAX_TORQUE = 5.0
TORQUE_SCALE = 0.5
MAX_TORQUE = 1.0

MAX_TORQUE_RATE = 10.0
OMEGA_SOFT = 0.6
OMEGA_HARD = 1.5

# =====================================================================
# Keyboard gripper toggle (Space = left tool, Ctrl = right tool)
# =====================================================================

CTRL_KEYS = {
    keyboard.Key.ctrl,
    keyboard.Key.ctrl_l,
    keyboard.Key.ctrl_r,
}


class KeyboardGripperController:
    """Edge-triggered keyboard listener: Space toggles left tool, Ctrl toggles right tool."""

    def __init__(self, initial_open: dict[str, bool]):
        self._lock = threading.Lock()
        self._is_open = {
            "left": bool(initial_open.get("left", True)),
            "right": bool(initial_open.get("right", True)),
        }
        self._toggle_pending = {"left": False, "right": False}
        self._held = {"left": False, "right": False}
        self._listener: keyboard.Listener | None = None

    def _on_press(self, key):
        side = None
        if key == keyboard.Key.space:
            side = "left"
        elif key in CTRL_KEYS:
            side = "right"

        if side is None:
            return

        with self._lock:
            if not self._held[side]:
                self._held[side] = True
                self._toggle_pending[side] = True

    def _on_release(self, key):
        side = None
        if key == keyboard.Key.space:
            side = "left"
        elif key in CTRL_KEYS:
            side = "right"

        if side is None:
            return

        with self._lock:
            self._held[side] = False

    def start(self):
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.daemon = True
        self._listener.start()
        print("[Gripper keys] SPACE -> toggle LEFT tool | CTRL -> toggle RIGHT tool")

    def stop(self):
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def consume_toggles_and_get_states(self) -> dict[str, bool]:
        changed: list[str] = []
        with self._lock:
            for side in ("left", "right"):
                if self._toggle_pending[side]:
                    self._is_open[side] = not self._is_open[side]
                    self._toggle_pending[side] = False
                    changed.append(side)
            states = dict(self._is_open)

        for side in changed:
            print(f"[Gripper] {side.upper()} tool -> {'OPEN' if states[side] else 'CLOSED'}")
        return states


def get_jaw_actuator_ids(model, side: str) -> dict[str, int]:
    """Resolve actuator IDs for left and right gripper joints of a given tool side."""
    aids: dict[str, int] = {}
    for suffix in JAW_SUFFIX_SIGNS:
        joint_name = f"{side}_{suffix}"
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            raise ValueError(f"Gripper joint not found: '{joint_name}'")

        found = [
            actuator_id
            for actuator_id in range(model.nu)
            if int(model.actuator_trntype[actuator_id]) == int(mujoco.mjtTrn.mjTRN_JOINT)
            and int(model.actuator_trnid[actuator_id, 0]) == jid
        ]
        if not found:
            raise ValueError(f"No actuator drives gripper joint '{joint_name}'.")
        aids[suffix] = found[0]
    return aids


def boost_grip_strength(model, jaw_act_ids: dict[str, dict[str, int]]):
    """Increase gripper actuator stiffness/force limits and jaw contact friction."""
    for side, side_aids in jaw_act_ids.items():
        for suffix, aid in side_aids.items():
            model.actuator_gainprm[aid, 0] *= GRIP_KP_SCALE
            if abs(model.actuator_biasprm[aid, 1]) > 1e-9:
                model.actuator_biasprm[aid, 1] *= GRIP_KP_SCALE

            if model.actuator_forcerange[aid, 0] != 0.0 or model.actuator_forcerange[aid, 1] != 0.0:
                model.actuator_forcerange[aid, 0] *= GRIP_FORCE_SCALE
                model.actuator_forcerange[aid, 1] *= GRIP_FORCE_SCALE

            if model.actuator_ctrlrange[aid, 0] != 0.0 or model.actuator_ctrlrange[aid, 1] != 0.0:
                lo, hi = model.actuator_ctrlrange[aid, 0], model.actuator_ctrlrange[aid, 1]
                model.actuator_ctrlrange[aid, 0] = min(lo, GRIPPER_CLOSED_POS - 0.05)
                model.actuator_ctrlrange[aid, 1] = max(hi, GRIPPER_OPEN_POS + 0.05)

            jid = int(model.actuator_trnid[aid, 0])
            body_id = int(model.jnt_bodyid[jid])
            for gid in range(model.ngeom):
                if int(model.geom_bodyid[gid]) == body_id:
                    model.geom_condim[gid] = max(int(model.geom_condim[gid]), JAW_CONDIM)
                    model.geom_friction[gid, 0] = max(float(model.geom_friction[gid, 0]), JAW_FRICTION[0])
                    model.geom_friction[gid, 1] = max(float(model.geom_friction[gid, 1]), JAW_FRICTION[1])
                    model.geom_friction[gid, 2] = max(float(model.geom_friction[gid, 2]), JAW_FRICTION[2])

    print(
        f"[Grip] Boosted gripper actuators (kp x{GRIP_KP_SCALE}, force x{GRIP_FORCE_SCALE}) "
        f"and jaw friction (condim>={JAW_CONDIM}, mu>={JAW_FRICTION[0]})."
    )


def apply_symmetric_gripper_ctrl(data, jaw_act_ids: dict[str, dict[str, int]], gripper_open: dict[str, bool]):
    """Drive both jaws of each tool symmetrically from a single 1-DOF open/closed state."""
    for side in ("left", "right"):
        mag = GRIPPER_OPEN_POS if gripper_open[side] else GRIPPER_CLOSED_POS
        for suffix, sign in JAW_SUFFIX_SIGNS.items():
            aid = jaw_act_ids[side][suffix]
            data.ctrl[aid] = sign * mag


# =====================================================================
# MuJoCo helpers
# =====================================================================


def get_mocap_id(model, body_name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body not found: {body_name}")

    mocap_id = model.body_mocapid[body_id]
    if mocap_id < 0:
        raise ValueError(f"Body is not a mocap body: {body_name}")

    return mocap_id


def get_jaw_body_ids(model, side: str) -> list[int]:
    """Find MuJoCo body IDs corresponding to the left/right gripper jaws for `side`."""
    prefix = f"{side}_"
    jaw_ids: list[int] = []
    for body_id in range(1, model.nbody):
        name = model.body(body_id).name
        if not name.startswith(prefix):
            continue
        own_name = name[len(prefix) :]
        short_name = own_name.split("_Shape")[0]
        if short_name in ("gripper_left", "gripper_right"):
            jaw_ids.append(body_id)
    return jaw_ids


def transform_pose(pose, frame_offset):
    """Convert UR TCP pose [x, y, z, rx, ry, rz] into the MuJoCo world frame."""
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


def vector_to_htm(tcp_vector):
    mat = np.eye(4)
    mat[:3, 3] = tcp_vector[:3]
    mat[:3, :3] = R.from_rotvec(tcp_vector[3:]).as_matrix()
    return mat


def site_pose_rb(ik, data):
    """End-effector pose expressed in the fixed RB / port frame."""
    p_world = data.site_xpos[ik.site_id]
    R_world = data.site_xmat[ik.site_id].reshape(3, 3)
    return ik.R_rb.T @ (p_world - ik.p_rb), ik.R_rb.T @ R_world


def jaws_tip_pose_rb(ik, data, jaw_body_ids: list[int], tip_offset_m: float):
    """Compute mean jaw tip position in {RB} and the offset from site_pose_rb."""
    p_ee_rb, R_ee_rb = site_pose_rb(ik, data)
    if not jaw_body_ids:
        return p_ee_rb, R_ee_rb, np.zeros(3)

    tip_points_rb: list[np.ndarray] = []
    for bid in jaw_body_ids:
        p_jaw_w = data.xpos[bid]
        R_jaw_w = data.xmat[bid].reshape(3, 3)
        p_tip_w = p_jaw_w + R_jaw_w @ np.array([tip_offset_m, 0.0, 0.0])
        p_tip_rb = ik.R_rb.T @ (p_tip_w - ik.p_rb)
        tip_points_rb.append(p_tip_rb)

    p_tip_mean_rb = np.mean(tip_points_rb, axis=0)
    tip_offset_rb = p_tip_mean_rb - p_ee_rb
    return p_tip_mean_rb, R_ee_rb, tip_offset_rb


def get_actuator_ids_for_joints(model, side):
    ids = []
    for suffix in JOINT_SUFFIXES:
        name = f"{side}_{suffix}"
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

        found = [
            actuator_id
            for actuator_id in range(model.nu)
            if int(model.actuator_trntype[actuator_id]) == int(mujoco.mjtTrn.mjTRN_JOINT)
            and int(model.actuator_trnid[actuator_id, 0]) == jid
        ]

        if not found:
            raise ValueError(f"No actuator drives joint '{name}'.")
        ids.append(found[0])

    return ids


def disable_mocap_welds(model, data, mocap_body_ids):
    if not hasattr(data, "eq_active"):
        print(
            "WARNING: this MuJoCo version has no data.eq_active. Remove the mocap welds "
            "from the XML manually or they will fight the IK."
        )
        return

    disabled = 0
    for i in range(model.neq):
        if int(model.eq_type[i]) != int(mujoco.mjtEq.mjEQ_WELD):
            continue
        if (
            int(model.eq_obj1id[i]) in mocap_body_ids
            or int(model.eq_obj2id[i]) in mocap_body_ids
        ):
            data.eq_active[i] = 0
            disabled += 1

    print(f"Disabled {disabled} mocap weld(s).")


# =====================================================================
# Haptic helpers (Tip-aware God-object proxy + virtual coupling fallback)
# =====================================================================


def clip_norm(vec, limit):
    """Scales vec down so its length is at most `limit`. Keeps its direction."""
    n = np.linalg.norm(vec)
    if n > limit:
        return vec * (limit / n)
    return vec


def slew_limit_norm(current, target, max_step):
    """Moves `current` toward `target`, but by no more than `max_step` (in length) per call."""
    delta = target - current
    n = np.linalg.norm(delta)
    if n > max_step:
        delta = delta * (max_step / n)
    return current + delta


def speed_taper(omega):
    """1 at slow rotation speeds, falling linearly to 0 at OMEGA_HARD."""
    if omega <= OMEGA_SOFT:
        return 1.0
    if omega >= OMEGA_HARD:
        return 0.0
    return (OMEGA_HARD - omega) / (OMEGA_HARD - OMEGA_SOFT)


def project_god_object_rb(p_query, boxes_rb, floor_z_rb, radius_m):
    """Projects a query position (tip or site) in {RB} onto/outside configured constraints.

    Returns (p_proxy, in_contact, normal_out).
    """
    p_proxy = p_query.copy()
    in_contact = False
    normal_sum = np.zeros(3)

    if floor_z_rb is not None:
        min_z = float(floor_z_rb) + radius_m
        if p_proxy[2] < min_z:
            p_proxy[2] = min_z
            in_contact = True
            normal_sum += np.array([0.0, 0.0, 1.0])

    for box in boxes_rb or []:
        center = np.asarray(box["center"], dtype=np.float64)
        half = np.asarray(box["half_size"], dtype=np.float64) + radius_m
        b_min = center - half
        b_max = center + half

        if (
            b_min[0] < p_proxy[0] < b_max[0]
            and b_min[1] < p_proxy[1] < b_max[1]
            and b_min[2] < p_proxy[2] < b_max[2]
        ):
            in_contact = True
            faces = [
                (abs(p_proxy[0] - b_min[0]), np.array([b_min[0], p_proxy[1], p_proxy[2]]), np.array([-1.0, 0.0, 0.0])),
                (abs(b_max[0] - p_proxy[0]), np.array([b_max[0], p_proxy[1], p_proxy[2]]), np.array([1.0, 0.0, 0.0])),
                (abs(p_proxy[1] - b_min[1]), np.array([p_proxy[0], b_min[1], p_proxy[2]]), np.array([0.0, -1.0, 0.0])),
                (abs(b_max[1] - p_proxy[1]), np.array([p_proxy[0], b_max[1], p_proxy[2]]), np.array([0.0, 1.0, 0.0])),
                (abs(p_proxy[2] - b_min[2]), np.array([p_proxy[0], p_proxy[1], b_min[2]]), np.array([0.0, 0.0, -1.0])),
                (abs(b_max[2] - p_proxy[2]), np.array([p_proxy[0], p_proxy[1], b_max[2]]), np.array([0.0, 0.0, 1.0])),
            ]
            _, p_proj, n_face = min(faces, key=lambda item: item[0])

            if floor_z_rb is not None:
                min_z = float(floor_z_rb) + radius_m
                if p_proj[2] < min_z:
                    p_proj[2] = min_z
                    normal_sum += np.array([0.0, 0.0, 1.0])

            p_proxy = p_proj
            normal_sum += n_face

    n_norm = np.linalg.norm(normal_sum)
    normal_out = normal_sum / n_norm if n_norm > 1e-9 else np.zeros(3)
    return p_proxy, in_contact, normal_out


def god_object_force_on_hand(p_query, p_proxy, prev_query, normal_out, dt, M):
    """Computes force on hand in robot/hand frame when god-object proxy is in contact."""
    query_vel = (p_query - prev_query) / dt
    penetration_vec = p_proxy - p_query

    force_on_hand_rb = STIFFNESS_LIN * penetration_vec - DAMPING_LIN * query_vel

    if np.linalg.norm(normal_out) > 1e-6:
        n_dot = float(np.dot(force_on_hand_rb, normal_out))
        if n_dot < 0.0:
            force_on_hand_rb -= n_dot * normal_out

    force_on_hand_rb = np.clip(force_on_hand_rb, -INTERNAL_MAX_FORCE, INTERNAL_MAX_FORCE)
    force_on_hand_h = FORCE_SCALE * (M.T @ force_on_hand_rb)
    force_on_hand_h = np.clip(force_on_hand_h, -MAX_FORCE, MAX_FORCE)
    return force_on_hand_h, force_on_hand_rb


def coupling_force_on_hand(p_target, p_ee, prev_gap, dt, M):
    """Virtual coupling fallback between the IK target and the tool's end effector/tip."""
    gap = p_target - p_ee
    gap_vel = (gap - prev_gap) / dt

    gap_norm = np.linalg.norm(gap)
    if gap_norm > GAP_DEADBAND_M:
        gap_eff = gap * (gap_norm - GAP_DEADBAND_M) / gap_norm
        proxy_force = STIFFNESS_LIN * gap_eff + DAMPING_LIN * gap_vel
    else:
        proxy_force = np.zeros(3)

    proxy_force = np.clip(proxy_force, -INTERNAL_MAX_FORCE, INTERNAL_MAX_FORCE)
    force_on_hand_rb = -proxy_force

    force_on_hand_h = FORCE_SCALE * (M.T @ force_on_hand_rb)
    force_on_hand_h = np.clip(force_on_hand_h, -MAX_FORCE, MAX_FORCE)

    return force_on_hand_h, force_on_hand_rb, gap


def coupling_torque_on_hand(R_target, R_ee, prev_gap, prev_vel, dt, M):
    """Rotational virtual coupling between the IK target orientation and the tool's."""
    gap = R.from_matrix(R_target @ R_ee.T).as_rotvec()
    gap_vel = TORQUE_VEL_FILTER * (gap - prev_gap) / dt + (1.0 - TORQUE_VEL_FILTER) * prev_vel

    gap_norm = np.linalg.norm(gap)
    if gap_norm > ROT_DEADBAND_RAD:
        gap_eff = gap * (gap_norm - ROT_DEADBAND_RAD) / gap_norm
        proxy_torque = STIFFNESS_ROT * gap_eff + DAMPING_ROT * gap_vel
    else:
        proxy_torque = np.zeros(3)

    proxy_torque = clip_norm(proxy_torque, INTERNAL_MAX_TORQUE)
    torque_on_hand_rb = -proxy_torque

    torque_on_hand_h = TORQUE_SCALE * (M.T @ torque_on_hand_rb)
    torque_on_hand_h = clip_norm(torque_on_hand_h, MAX_TORQUE)

    return torque_on_hand_h, gap, gap_vel


# =====================================================================
# Unity helpers
# =====================================================================


def get_joint_qpos(model, data, joint_name, default=0.0):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        return float(default)
    adr = int(model.jnt_qposadr[jid])
    return float(data.qpos[adr])


def get_joint_range(model, joint_name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        return None
    return float(model.jnt_range[jid, 0]), float(model.jnt_range[jid, 1])


def build_unity_tool_state(model, data, side):
    prefix = f"{side}_"
    tool_q = [get_joint_qpos(model, data, prefix + joint_name) for joint_name in UNITY_TOOL_JOINTS]
    port_q = [get_joint_qpos(model, data, prefix + joint_name) for joint_name in UNITY_PORT_JOINTS]

    pitch_name = prefix + "joint_pitch"
    pitch = get_joint_qpos(model, data, pitch_name)
    pitch_range = get_joint_range(model, pitch_name)
    pitch_margin = 1.0

    if pitch_range is not None:
        limit = max(abs(pitch_range[0]), abs(pitch_range[1]))
        if limit > 1e-9:
            pitch_margin = 1.0 - min(1.0, abs(pitch) / limit)

    return {
        "mode": "ik",
        "master_ok": True,
        "clutch": False,
        "master": None,
        "tool_q": r6(tool_q),
        "pitch_margin": float(pitch_margin),
        "jaw_contact_force": 0.0,
        "port_q": r6(port_q),
    }


def get_unity_tool_bodies(model, side):
    prefix = f"{side}_"
    selected = []

    for body_id in range(1, model.nbody):
        name = model.body(body_id).name
        if not name.startswith(prefix):
            continue
        own_name = name[len(prefix) :]
        short_name = own_name.split("_Shape")[0]

        if short_name in UNITY_BODY_SUFFIXES:
            selected.append(name)

    return selected


# =====================================================================
# Main
# =====================================================================


def main():
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    jaw_act_ids = {side: get_jaw_actuator_ids(model, side) for side in ("left", "right")}
    boost_grip_strength(model, jaw_act_ids)
    data = mujoco.MjData(model)

    gripper_ctrl = KeyboardGripperController(GRIPPER_INITIAL_OPEN)
    gripper_ctrl.start()

    unity_tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    unity_period = 1.0 / STATE_SEND_HZ
    next_unity_send = time.perf_counter()
    unity_seq = 0

    left_unity_bodies = get_unity_tool_bodies(model, "left")
    right_unity_bodies = get_unity_tool_bodies(model, "right")
    jaw_body_ids = {
        side: get_jaw_body_ids(model, side) for side in ("left", "right")
    }

    print(f"\n[Unity bridge] state -> {UNITY_HOST}:{STATE_PORT} @ {STATE_SEND_HZ:.0f} Hz")
    print(f"[Jaw bodies] left={jaw_body_ids['left']} | right={jaw_body_ids['right']}")

    mocap_ids = {
        side: get_mocap_id(model, body_name) for side, body_name in MOCAP_BODIES.items()
    }
    ips = {"left": LEFT_IP, "right": RIGHT_IP}

    steps_per_loop = max(1, round(CONTROL_PERIOD_S / model.opt.timestep))
    iks = {
        side: PoseIK(model, side, port_body=PORT_BODIES[side]) for side in ips
    }
    act_ids = {side: get_actuator_ids_for_joints(model, side) for side in ips}
    mocap_body_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        for body_name in MOCAP_BODIES.values()
    ]

    controllers = {}
    receivers = {}
    running = True

    def handle_sigint(signum, frame):
        nonlocal running
        print("\nCtrl+C received, stopping...")
        running = False

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        for side, ip in ips.items():
            print(f"Connecting to {side} robot at {ip}...")
            controllers[side] = rtde_control.RTDEControlInterface(ip)
            receivers[side] = rtde_receive.RTDEReceiveInterface(ip)

        time.sleep(1.0)

        for controller in controllers.values():
            controller.zeroFtSensor()
            controller.forceModeSetDamping(DAMPING)
            controller.forceModeSetGainScaling(GAIN_SCALING)

        print("Connected. Launching MuJoCo viewer alongside Unity transmission...")

        W_start = {side: receiver.getActualTCPPose() for side, receiver in receivers.items()}

        if model.nkey > 0:
            mujoco.mj_resetDataKeyframe(model, data, 0)

        disable_mocap_welds(model, data, mocap_body_ids)
        mujoco.mj_forward(model, data)

        P_start = {side: site_pose_rb(iks[side], data) for side in ips}
        mujoco.mj_forward(model, data)

        for side in ips:
            Rw = R_WORLD_H[side]
            valid_rotation = np.allclose(Rw.T @ Rw, np.eye(3)) and np.isclose(
                np.linalg.det(Rw), 1.0
            )
            if not valid_rotation:
                raise ValueError(
                    f"R_WORLD_H['{side}'] must be a proper rotation (orthonormal, det +1)."
                )

        R_RB_H = {side: iks[side].R_rb.T @ R_WORLD_H[side] for side in ips}

        hand_start = {}
        for side in ips:
            H0 = vector_to_htm(W_start[side])
            hand_start[side] = (H0[:3, 3].copy(), H0[:3, :3].copy())

        tool_start = P_start
        q_cmd = {side: data.qpos[iks[side].qpos_ids].copy() for side in ips}

        for side in ips:
            iks[side].q_last[:] = q_cmd[side]

        ik_ok = {side: True for side in ips}
        ik_calls = 0
        loop = 0

        # --- HAPTICS: per-arm state ---
        wrench = {side: list(WRENCH) for side in ips}
        p_targets = {side: tool_start[side][0].copy() for side in ips}
        prev_p_queries = {side: tool_start[side][0].copy() for side in ips}
        p_proxies = {side: tool_start[side][0].copy() for side in ips}
        prev_gap = {side: np.zeros(3) for side in ips}

        # --- TORQUE HAPTICS: per-arm state ---
        R_targets = {side: tool_start[side][1].copy() for side in ips}
        prev_rot_gap = {side: np.zeros(3) for side in ips}
        rot_gap_vel = {side: np.zeros(3) for side in ips}
        torque_cmd = {side: np.zeros(3) for side in ips}

        haptic_t0 = time.perf_counter()
        prev_time = haptic_t0

        with mujoco.viewer.launch_passive(model, data) as viewer:
            while running and viewer.is_running():
                loop_start = time.perf_counter()

                # --- HAPTICS: each arm gets its own wrench ---
                for side, controller in controllers.items():
                    controller.forceMode(
                        TASK_FRAME, SELECTION_VECTOR, wrench[side], FORCE_TYPE, LIMITS
                    )

                poses = {side: receiver.getActualTCPPose() for side, receiver in receivers.items()}

                # IK Solver loop (kept anchored to ik.site_id)
                for side, pose in poses.items():
                    H1 = vector_to_htm(pose)
                    p_h1 = H1[:3, 3]
                    R_h1 = H1[:3, :3]

                    p_h0, R_h0 = hand_start[side]
                    p_rb0, R_rb0 = tool_start[side]
                    M = R_RB_H[side]

                    p_target = p_rb0 + SCALE * (M @ (p_h1 - p_h0))
                    R_target = (M @ (R_h1 @ R_h0.T) @ M.T) @ R_rb0
                    p_targets[side] = p_target
                    R_targets[side] = R_target

                    target_quat = mat_to_quat(R_target)
                    result = iks[side].solve(p_target, target_quat, frame="rb")

                    if result.converged:
                        q_cmd[side] = result.q

                    if result.converged != ik_ok[side]:
                        ik_ok[side] = result.converged
                        status = "recovered" if result.converged else "FAILED"
                        print(f"[{side}] IK {status}")

                ik_calls += 1

                # MuJoCo Simulation Update
                for side, pose in poses.items():
                    pos, quat = transform_pose(pose, FRAME_OFFSETS[side])
                    data.mocap_pos[mocap_ids[side]] = pos
                    data.mocap_quat[mocap_ids[side]] = quat

                for side in ips:
                    data.ctrl[act_ids[side]] = q_cmd[side]

                gripper_states = gripper_ctrl.consume_toggles_and_get_states()
                apply_symmetric_gripper_ctrl(data, jaw_act_ids, gripper_states)

                for _ in range(steps_per_loop):
                    mujoco.mj_step(model, data)

                viewer.sync()

                # --- HAPTICS: tip-aware god-object proxy + virtual coupling fallback ---
                now_h = time.perf_counter()
                dt_h = max(now_h - prev_time, 1e-4)
                prev_time = now_h

                ramp_s = max(HAPTIC_RAMP_S, 1e-6)
                ramp = min(1.0, max(0.0, (now_h - haptic_t0 - HAPTIC_STARTUP_DELAY_S) / ramp_s))
                stalled = dt_h > HAPTIC_MAX_DT_S

                for side in ips:
                    p_ee_rb, R_ee_rb = site_pose_rb(iks[side], data)

                    if TIP_HAPTICS_ENABLED:
                        p_tip_rb, _, tip_offset_rb = jaws_tip_pose_rb(
                            iks[side], data, jaw_body_ids[side], TIP_FORWARD_OFFSET_M
                        )
                        p_query_target = p_targets[side] + tip_offset_rb
                        p_query_actual = p_tip_rb
                    else:
                        p_query_target = p_targets[side]
                        p_query_actual = p_ee_rb

                    if GOD_OBJECT_ENABLED:
                        p_proxy_next, in_contact, n_out = project_god_object_rb(
                            p_query=p_query_target,
                            boxes_rb=GOD_OBJECT_BOXES_RB.get(side, []),
                            floor_z_rb=GOD_OBJECT_FLOOR_Z_RB.get(side, None),
                            radius_m=PROXY_RADIUS_M,
                        )
                        p_proxies[side] = p_proxy_next

                        if in_contact:
                            force_h, force_rb = god_object_force_on_hand(
                                p_query=p_query_target,
                                p_proxy=p_proxy_next,
                                prev_query=prev_p_queries[side],
                                normal_out=n_out,
                                dt=dt_h,
                                M=R_RB_H[side],
                            )
                            gap = p_proxy_next - p_query_target
                        else:
                            force_h, force_rb, gap = coupling_force_on_hand(
                                p_query_target, p_query_actual, prev_gap[side], dt_h, R_RB_H[side]
                            )
                        prev_gap[side] = gap
                    else:
                        p_proxies[side] = p_query_target
                        force_h, force_rb, gap = coupling_force_on_hand(
                            p_query_target, p_query_actual, prev_gap[side], dt_h, R_RB_H[side]
                        )
                        prev_gap[side] = gap

                    prev_p_queries[side] = p_query_target.copy()

                    if not HAPTICS_ENABLED or stalled:
                        force_h = np.zeros(3)

                    wrench[side][0:3] = (ramp * force_h).tolist()

                    torque_h, rot_gap, rot_gap_vel[side] = coupling_torque_on_hand(
                        R_targets[side],
                        R_ee_rb,
                        prev_rot_gap[side],
                        rot_gap_vel[side],
                        dt_h,
                        R_RB_H[side],
                    )
                    prev_rot_gap[side] = rot_gap

                    omega = float(np.linalg.norm(receivers[side].getActualTCPSpeed()[3:6]))
                    torque_h = torque_h * speed_taper(omega)

                    if not TORQUE_HAPTICS_ENABLED or stalled:
                        torque_cmd[side] = np.zeros(3)
                    else:
                        torque_cmd[side] = slew_limit_norm(
                            torque_cmd[side], torque_h, MAX_TORQUE_RATE * dt_h
                        )

                    wrench[side][3:6] = (ramp * torque_cmd[side]).tolist()

                    if loop % HAPTIC_PRINT_EVERY_N_LOOPS == 0:
                        print(
                            f"[{side}] tip={TIP_HAPTICS_ENABLED} | |gap| {np.linalg.norm(gap) * 1000:.2f} mm | "
                            f"F_hand {np.round(wrench[side][0:3], 2)} N | "
                            f"T_hand {np.round(wrench[side][3:6], 3)} N·m"
                        )

                loop += 1

                # Unity UDP Publication
                now = time.perf_counter()
                if now >= next_unity_send:
                    tools = {
                        "left_": build_unity_tool_state(model, data, "left"),
                        "right_": build_unity_tool_state(model, data, "right"),
                    }
                    unity_seq += 1

                    if unity_seq % 2 == 1:
                        packet_side = "left"
                        packet_bodies = left_unity_bodies
                    else:
                        packet_side = "right"
                        packet_bodies = right_unity_bodies

                    state = {
                        "v": SCHEMA,
                        "seq": unity_seq,
                        "t_sim": round(float(data.time), 6),
                        "ncon": int(data.ncon),
                        "tools": tools,
                        "bodies": body_poses(model, data, packet_bodies),
                        "contacts": [],
                    }

                    state.update(tools["left_"])
                    state["t_send"] = now

                    payload = json.dumps(state, separators=(",", ":")).encode()
                    unity_tx.sendto(payload, (UNITY_HOST, STATE_PORT))
                    next_unity_send += unity_period

                    if next_unity_send < (now - unity_period):
                        next_unity_send = now + unity_period

                # Maintain loop pacing
                remaining = CONTROL_PERIOD_S - (time.perf_counter() - loop_start)
                if remaining > 0:
                    time.sleep(remaining)

    finally:
        print("Stopping force mode and disconnecting...")
        gripper_ctrl.stop()
        try:
            unity_tx.close()
        except Exception:
            pass

        for side, controller in controllers.items():
            try:
                controller.forceModeStop()
                controller.stopScript()
            except Exception as exc:
                print(f"[{side}] error stopping force mode: {exc}")

        for side, receiver in receivers.items():
            try:
                receiver.disconnect()
            except Exception as exc:
                print(f"[{side}] error disconnecting: {exc}")

        print("Done.")


if __name__ == "__main__":
    main()