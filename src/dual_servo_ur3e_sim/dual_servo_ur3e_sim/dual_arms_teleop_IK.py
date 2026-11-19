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

# Single open/close target angle (rad) per tool.
# Slightly negative GRIPPER_CLOSED_POS applies firm closing preload onto grasped objects
# (automatically clamped to joint limits).
GRIPPER_CLOSED_POS = -0.05
GRIPPER_OPEN_POS = 0.3
GRIPPER_INITIAL_OPEN = {"left": True, "right": True}

# Sign multiplier applied to each jaw joint when setting symmetric opening/closing.
# Flip a jaw to -1.0 if its joint axis opens in the opposite direction in hello_unity_7dof.xml.
JAW_SUFFIX_SIGNS = {
    "joint_gripper_left": 1.0,
    "joint_gripper_right": 1.0,
}

# Grip strength multipliers applied directly to MuJoCo gripper actuators & jaw contacts:
GRIP_KP_SCALE = 3.0       # multiplies position gain kp on gripper actuators
GRIP_FORCE_SCALE = 4.0    # multiplies actuator forcerange limit on gripper actuators
JAW_CONDIM = 4            # 4 = sliding + torsional friction; 6 = full 3D + spin/roll
JAW_FRICTION = np.array([2.0, 0.05, 0.01])

# =====================================================================
# Unity bridge configuration
# =====================================================================

# Windows Ethernet address reachable from WSL.
UNITY_HOST = "192.168.0.101"
STATE_SEND_HZ = 60.0

# Each arm receives its visual body poses at half the packet rate:
# packet 1 -> left bodies | packet 2 -> right bodies | etc.
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
WRENCH = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # zero wrench: the starting point for every arm
FORCE_TYPE = 2
# Speed caps on the compliant axes (m/s for the first three, rad/s for the last three).
# The rotation caps were 3.14, 3.14 and are now 1.5: a hard ceiling, enforced by the robot
# controller itself, on how fast a torque can spin the wrist.
LIMITS = [0.1, 0.1, 1.5, 1.5, 1.5, 0.5]
DAMPING = 0.004
GAIN_SCALING = 1.0

CONTROL_PERIOD_S = 0.008

# =====================================================================
# Coordinate transforms
# =====================================================================

# Fixed 180-degree rotation around Y. Quaternion order used by MuJoCo: w, x, y, z.
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
# Haptic feedback (virtual coupling, as in the sphere test)
# =====================================================================

HAPTICS_ENABLED = True

STIFFNESS_LIN = 2800.0  # N/m
DAMPING_LIN = 15.0  # N/(m/s)
GAP_DEADBAND_M = 0.002  # gaps below this render no force (hides the tool's tracking lag)
INTERNAL_MAX_FORCE = 100.0  # clip of the coupling force, per axis (N)
FORCE_SCALE = 0.4  # gain from tool force to hand force
MAX_FORCE = 100.0  # clip of the force sent to each robot, per axis (N). Raise gradually.

HAPTIC_STARTUP_DELAY_S = 2.0  # zero force for this long after the loop starts
HAPTIC_RAMP_S = 1.0  # then ramp the force in over this long
HAPTIC_MAX_DT_S = 0.05  # a loop cycle slower than this sends zero force (stall protection)
HAPTIC_PRINT_EVERY_N_LOOPS = 100

# =====================================================================
# Torque haptics (rotational virtual coupling)
# =====================================================================

# The same idea as the force coupling, for orientation: a torsional spring-damper couples the
# target orientation (from the hand) to the tool's end-effector orientation. When the tool
# cannot follow the rotation, the reaction torque is sent to the UR arm in WRENCH[3:6].
TORQUE_HAPTICS_ENABLED = True

STIFFNESS_ROT = 6.0  # N·m/rad
DAMPING_ROT = 0.05  # N·m/(rad/s)
ROT_DEADBAND_RAD = 0.05  # rotation gaps below this render no torque (about 3 degrees)
TORQUE_VEL_FILTER = 0.3  # 0..1: weight of the newest rate sample (smaller = smoother)
INTERNAL_MAX_TORQUE = 5.0  # clip of the coupling torque, by norm (N·m)
TORQUE_SCALE = 0.5  # gain from tool torque to hand torque
MAX_TORQUE = 1.0  # clip of the torque sent to each robot, by norm (N·m). Raise gradually.

# Protection against harsh or runaway rotation:
MAX_TORQUE_RATE = 10.0  # N·m/s: the sent torque can change no faster than this
OMEGA_SOFT = 0.6  # rad/s: above this TCP rotation speed the torque starts to fade out
OMEGA_HARD = 1.5  # rad/s: at this speed (or faster) the torque is zero

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
            # For MuJoCo position actuators, gainprm[0] is kp and biasprm[1] is -kp.
            model.actuator_gainprm[aid, 0] *= GRIP_KP_SCALE
            if abs(model.actuator_biasprm[aid, 1]) > 1e-9:
                model.actuator_biasprm[aid, 1] *= GRIP_KP_SCALE

            if model.actuator_forcerange[aid, 0] != 0.0 or model.actuator_forcerange[aid, 1] != 0.0:
                model.actuator_forcerange[aid, 0] *= GRIP_FORCE_SCALE
                model.actuator_forcerange[aid, 1] *= GRIP_FORCE_SCALE

            # Also widen actuator ctrlrange slightly if needed so over-closure isn't clipped.
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
# Haptic helpers
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


def coupling_force_on_hand(p_target, p_ee, prev_gap, dt, M):
    """Virtual coupling between the IK target and the tool's end effector."""
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

    print(f"\n[Unity bridge] state -> {UNITY_HOST}:{STATE_PORT} @ {STATE_SEND_HZ:.0f} Hz")

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

        # --- HAPTICS: per-arm state ---
        wrench = {side: list(WRENCH) for side in ips}
        p_targets = {side: tool_start[side][0].copy() for side in ips}
        prev_gap = {side: np.zeros(3) for side in ips}

        # --- TORQUE HAPTICS: per-arm state ---
        R_targets = {side: tool_start[side][1].copy() for side in ips}
        prev_rot_gap = {side: np.zeros(3) for side in ips}
        rot_gap_vel = {side: np.zeros(3) for side in ips}
        torque_cmd = {side: np.zeros(3) for side in ips}

        haptic_t0 = time.perf_counter()
        prev_time = haptic_t0

        # Launch the MuJoCo passive viewer
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while running and viewer.is_running():
                loop_start = time.perf_counter()

                # --- HAPTICS: each arm gets its own wrench ---
                for side, controller in controllers.items():
                    controller.forceMode(
                        TASK_FRAME, SELECTION_VECTOR, wrench[side], FORCE_TYPE, LIMITS
                    )

                poses = {side: receiver.getActualTCPPose() for side, receiver in receivers.items()}

                # IK Solver loop
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

                # Symmetric 1-DOF jaw toggle per tool (Space = left, Ctrl = right)
                gripper_states = gripper_ctrl.consume_toggles_and_get_states()
                apply_symmetric_gripper_ctrl(data, jaw_act_ids, gripper_states)

                for _ in range(steps_per_loop):
                    mujoco.mj_step(model, data)

                # Sync MuJoCo passive viewer graphics
                viewer.sync()

                # --- HAPTICS: virtual coupling calculations ---
                now_h = time.perf_counter()
                dt_h = max(now_h - prev_time, 1e-4)
                prev_time = now_h

                ramp_s = max(HAPTIC_RAMP_S, 1e-6)
                ramp = min(1.0, max(0.0, (now_h - haptic_t0 - HAPTIC_STARTUP_DELAY_S) / ramp_s))
                stalled = dt_h > HAPTIC_MAX_DT_S

                for side in ips:
                    p_ee_rb, R_ee_rb = site_pose_rb(iks[side], data)

                    force_h, force_rb, gap = coupling_force_on_hand(
                        p_targets[side], p_ee_rb, prev_gap[side], dt_h, R_RB_H[side]
                    )
                    prev_gap[side] = gap

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