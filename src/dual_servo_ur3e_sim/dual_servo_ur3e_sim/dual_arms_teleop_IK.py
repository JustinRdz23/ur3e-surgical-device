"""Dual-arm teleoperation with inverse kinematics, streaming its state to Unity.

Each UR arm is a master. Its TCP motion, measured from where it was when the program
started, moves the matching simulated tool in MuJoCo through inverse kinematics (the
incremental update P1 = dW * P0 from the paper). Both arms are kept compliant with a
zero-wrench forceMode(). The MuJoCo state is streamed to Unity over UDP.

Needs IK_solver.py and mujoco_unity_bridge.py in the same folder.

    python3 dual_arms_teleop_IK_unity.py

Ctrl+C or closing the viewer stops force mode on both robots and disconnects.
"""
import json
import signal
import socket
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer
import rtde_control
import rtde_receive
from scipy.spatial.transform import Rotation

from IK_solver import PoseIK, JOINT_SUFFIXES, mat_to_quat
from mujoco_unity_bridge import STATE_PORT, SCHEMA, body_poses, r6


# ════════════════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════════════════

# ── Robots and files ─────────────────────────────────────────────────────────
LEFT_IP = "192.168.0.1"    # UR3
RIGHT_IP = "192.168.0.3"   # UR3e
ROBOT_IPS = {"left": LEFT_IP, "right": RIGHT_IP}

MJCF_PATH = str(Path(__file__).resolve().parent.parent / "mjcf" / "teleop_dual_IK.xml")
MOCAP_BODIES = {"left": "left_arm_mocap", "right": "right_arm_mocap"}
PORT_BODIES = {"left": "left_port", "right": "right_port"}

# ── forceMode with a zero wrench (keeps the arms compliant) ──────────────────
TASK_FRAME = [0, 0, 0, 0, 0, 0]
SELECTION_VECTOR = [1, 1, 1, 1, 1, 1]
WRENCH = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
FORCE_TYPE = 2
LIMITS = [0.1, 0.1, 1.5, 3.14, 3.14, 0.5]
DAMPING = 0.004
GAIN_SCALING = 1.0

# ── Loop timing ──────────────────────────────────────────────────────────────
CONTROL_PERIOD_S = 0.008      # 8 ms = 125 Hz (the IK-only version used 2 ms)
RENDER_EVERY_N_LOOPS = 10
IK_REPORT_EVERY = 500         # print IK timing statistics every N loops

# ── Coordinate frames and hand -> tool mapping ───────────────────────────────
# Fixed 180-degree rotation about Y, quaternion (w, x, y, z), robot base -> MuJoCo world.
# Only used to place the visual mocap markers, not by the IK.
Q_OFFSET = np.array([0.0, 0.0, 1.0, 0.0])

# Where each robot's base origin sits in MuJoCo world coordinates (markers only).
FRAME_OFFSETS = {
    "left": np.array([-0.3, 0.0, 0.8]),
    "right": np.array([0.3, 0.0, 0.8]),
}

# Rotation from each robot's base axes to the MuJoCo world axes: a chosen convention.
# Must be a proper rotation (orthonormal, determinant +1).
R_WORLD_H = {
    "left": np.diag([-1.0, 1.0, -1.0]),
    "right": np.diag([-1.0, 1.0, -1.0]),
}

SCALE = 0.5    # tool motion per unit of hand motion

# ── Unity bridge ─────────────────────────────────────────────────────────────
UNITY_HOST = "192.168.0.101"    # Windows Ethernet address reachable from WSL
STATE_SEND_HZ = 60.0
MAX_UDP_BYTES = 1450            # larger datagrams fragment between WSL and Windows

# Each packet carries both arms' tool state but only ONE arm's visual bodies, alternating:
#   odd packets -> left bodies, even packets -> right bodies.
# At 60 Hz that gives 60 Hz tool state and 30 Hz visual bodies per arm.

# NOTE: joint_yaw is not in this list, but the IK drives it. Check whether Unity needs it.
UNITY_TOOL_JOINTS = ["joint_roll", "joint_pitch", "joint_gripper_left", "joint_gripper_right"]
UNITY_PORT_JOINTS = ["port_yaw", "port_pitch", "port_insert"]
UNITY_BODY_SUFFIXES = (
    "tool_main_insert",
    "tool_roll_link",
    "tool_pitch_link",
    "tool_yaw_link",
    "gripper_left",
    "gripper_right",
)


# ════════════════════════════════════════════════════════════════════════════
# MuJoCo lookups
# ════════════════════════════════════════════════════════════════════════════

def get_mocap_id(model, body_name):
    """Index of a mocap body in data.mocap_pos / data.mocap_quat. Raises if it is not one."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body not found: {body_name}")
    mocap_id = model.body_mocapid[body_id]
    if mocap_id < 0:
        raise ValueError(f"Body is not a mocap body: {body_name}")
    return mocap_id


def get_actuator_ids_for_joints(model, side):
    """Finds the actuator that drives each of the six IK joints, in JOINT_SUFFIXES order.

    It looks the actuators up by which JOINT they drive, so their names do not matter:
    for a joint actuator, model.actuator_trnid[a, 0] holds the joint's id.
    """
    ids = []
    for suffix in JOINT_SUFFIXES:
        name = f"{side}_{suffix}"
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        found = [
            a for a in range(model.nu)
            if int(model.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_JOINT)
            and int(model.actuator_trnid[a, 0]) == jid
        ]
        if not found:
            raise ValueError(f"No actuator drives joint '{name}'.")
        ids.append(found[0])
    return ids


def disable_mocap_welds(model, data, mocap_body_ids):
    """Switches off every weld that touches a mocap body, so nothing fights the IK."""
    if not hasattr(data, "eq_active"):
        print("WARNING: this MuJoCo version has no data.eq_active. Remove the mocap welds "
              "from the XML by hand or they will fight the IK.")
        return

    disabled = 0
    for i in range(model.neq):
        if int(model.eq_type[i]) != int(mujoco.mjtEq.mjEQ_WELD):
            continue
        if int(model.eq_obj1id[i]) in mocap_body_ids or int(model.eq_obj2id[i]) in mocap_body_ids:
            data.eq_active[i] = 0
            disabled += 1
    print(f"Disabled {disabled} mocap weld(s).")


# ════════════════════════════════════════════════════════════════════════════
# Coordinate conversions
# ════════════════════════════════════════════════════════════════════════════

def vector_to_htm(tcp_vector):
    """UR pose [x, y, z, rx, ry, rz] (rotation vector) -> 4x4 homogeneous transform."""
    htm = np.eye(4)
    htm[:3, 3] = tcp_vector[:3]
    htm[:3, :3] = Rotation.from_rotvec(tcp_vector[3:]).as_matrix()
    return htm


def transform_pose(pose, frame_offset):
    """UR TCP pose -> (position, quaternion) in the MuJoCo world. Used for the markers only."""
    # Rotate the position 180 degrees about Y, then shift by the robot's base location.
    pos = np.zeros(3)
    mujoco.mju_rotVecQuat(pos, np.array(pose[:3]), Q_OFFSET)
    pos += frame_offset

    # Rotation vector -> quaternion, then apply the same fixed 180-degree rotation.
    rot_vec = np.array(pose[3:6])
    angle = np.linalg.norm(rot_vec)
    quat_robot = np.array([1.0, 0.0, 0.0, 0.0])
    if angle > 1e-9:
        mujoco.mju_axisAngle2Quat(quat_robot, rot_vec / angle, angle)

    quat = np.zeros(4)
    mujoco.mju_mulQuat(quat, Q_OFFSET, quat_robot)
    return pos, quat


def site_pose_rb(ik, data):
    """The TOOL's end-effector pose expressed in {RB} (the port frame at zero joints).

    This is P0 in the paper's formula: the pose of the simulated tool, not of the hand.
    {RB} is ik.p_rb / ik.R_rb, the fixed pose of the port body (the RCM).
    Returns (position 3-vector, rotation matrix 3x3). Call after mj_forward.
    """
    p_world = data.site_xpos[ik.site_id]
    R_world = data.site_xmat[ik.site_id].reshape(3, 3)
    return ik.R_rb.T @ (p_world - ik.p_rb), ik.R_rb.T @ R_world


def check_proper_rotation(R, name):
    """Raises unless R is orthonormal with determinant +1. A determinant of -1 is a mirror."""
    ok = np.allclose(R.T @ R, np.eye(3)) and np.isclose(np.linalg.det(R), 1.0)
    if not ok:
        raise ValueError(f"{name} must be a proper rotation (orthonormal, det +1).")


def hand_to_tool_target(pose, hand0, tool0, M):
    """The paper's incremental update: where the tool should be, in {RB}.

    pose  : current UR TCP pose [x, y, z, rx, ry, rz]
    hand0 : (position, rotation matrix) of the hand when engaged, in the UR base frame
    tool0 : (position, rotation matrix) of the tool when engaged, in {RB}
    M     : rotation from the UR base axes to {RB} (R_RB_H)
    Returns (target position, target rotation matrix), both in {RB}.
    """
    H1 = vector_to_htm(pose)
    p_h0, R_h0 = hand0
    p_rb0, R_rb0 = tool0

    p_target = p_rb0 + SCALE * (M @ (H1[:3, 3] - p_h0))
    R_target = (M @ (H1[:3, :3] @ R_h0.T) @ M.T) @ R_rb0
    return p_target, R_target


# ════════════════════════════════════════════════════════════════════════════
# Unity packets
# ════════════════════════════════════════════════════════════════════════════

def get_joint_qpos(model, data, joint_name, default=0.0):
    """Current value of a joint by name, or `default` if the joint does not exist."""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        return float(default)
    return float(data.qpos[int(model.jnt_qposadr[jid])])


def get_joint_range(model, joint_name):
    """(low, high) limits of a joint by name, or None if the joint does not exist."""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        return None
    return float(model.jnt_range[jid, 0]), float(model.jnt_range[jid, 1])


def get_unity_tool_bodies(model, side):
    """The six MuJoCo bodies per arm that MujocoBridge.cs actually uses for one dVRK tool.

    Filtering to six keeps each UDP datagram below the fragmentation threshold observed
    between WSL and Windows.
    """
    prefix = f"{side}_"
    selected = []

    for body_id in range(1, model.nbody):
        name = model.body(body_id).name
        if not name.startswith(prefix):
            continue
        short_name = name[len(prefix):].split("_Shape")[0]
        if short_name in UNITY_BODY_SUFFIXES:
            selected.append(name)

    if len(selected) != 6:
        print(f"WARNING: expected 6 Unity bodies for {side}, found {len(selected)}")
        for name in selected:
            print(f"  {name}")
    return selected


def build_unity_tool_state(model, data, side):
    """Joint values and status for one tool, in the fields Unity expects."""
    prefix = f"{side}_"

    tool_q = [get_joint_qpos(model, data, prefix + j) for j in UNITY_TOOL_JOINTS]
    port_q = [get_joint_qpos(model, data, prefix + j) for j in UNITY_PORT_JOINTS]

    # How far the wrist pitch is from its limit: 1.0 = centred, 0.0 = at the limit.
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


def build_unity_packet(model, data, seq, bodies, t_send):
    """One UDP payload: both tools' state plus the visual body poses in `bodies`."""
    tools = {
        "left_": build_unity_tool_state(model, data, "left"),
        "right_": build_unity_tool_state(model, data, "right"),
    }

    state = {
        "v": SCHEMA,
        "seq": seq,
        "t_sim": round(float(data.time), 6),
        "ncon": int(data.ncon),
        "tools": tools,
        "bodies": body_poses(model, data, bodies),
        "contacts": [],
    }
    state.update(tools["left_"])    # legacy v1 fields
    state["t_send"] = t_send

    return json.dumps(state, separators=(",", ":")).encode()


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    # ── 1. Model, IK solvers and lookups ────────────────────────────────────
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)

    mocap_ids = {side: get_mocap_id(model, name) for side, name in MOCAP_BODIES.items()}
    mocap_body_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in MOCAP_BODIES.values()
    ]
    iks = {side: PoseIK(model, side, port_body=PORT_BODIES[side]) for side in ROBOT_IPS}
    act_ids = {side: get_actuator_ids_for_joints(model, side) for side in ROBOT_IPS}
    steps_per_loop = max(1, round(CONTROL_PERIOD_S / model.opt.timestep))

    # ── 2. Unity UDP ────────────────────────────────────────────────────────
    unity_bodies = {side: get_unity_tool_bodies(model, side) for side in ROBOT_IPS}
    print("\nUnity bodies:")
    for side, names in unity_bodies.items():
        print(f"{side.upper()}:")
        for name in names:
            print(f"  {name}")

    print(f"\n[Unity bridge] state -> {UNITY_HOST}:{STATE_PORT} @ {STATE_SEND_HZ:.0f} Hz")
    print("[Unity bridge] alternating left/right visual bodies to avoid UDP fragmentation.")

    unity_tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    unity_period = 1.0 / STATE_SEND_HZ
    next_unity_send = time.perf_counter()
    unity_seq = 0

    # ── 3. Ctrl+C safety stop ───────────────────────────────────────────────
    controllers = {}
    receivers = {}
    running = True

    def handle_sigint(signum, frame):
        nonlocal running
        print("\nCtrl+C received, stopping...")
        running = False

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        # ── 4. Connect to the robots ────────────────────────────────────────
        for side, ip in ROBOT_IPS.items():
            print(f"Connecting to {side} robot at {ip}...")
            controllers[side] = rtde_control.RTDEControlInterface(ip)
            receivers[side] = rtde_receive.RTDEReceiveInterface(ip)

        time.sleep(1.0)
        for controller in controllers.values():
            controller.zeroFtSensor()    # resets the force/torque sensor at the TCP
            controller.forceModeSetDamping(DAMPING)
            controller.forceModeSetGainScaling(GAIN_SCALING)
        print("Connected. Both arms become compliant now.")

        with mujoco.viewer.launch_passive(model, data) as viewer:
            print("Dual-arm teleoperation running.")

            # ── 5. Latch the starting poses (hand and tool) ─────────────────
            # Hand: each arm's TCP pose, in its own base frame.
            W_start = {side: receiver.getActualTCPPose() for side, receiver in receivers.items()}

            # Tool: reset the sim to its home pose, then read where each tool's end effector is.
            if model.nkey > 0:
                mujoco.mj_resetDataKeyframe(model, data, 0)
            disable_mocap_welds(model, data, mocap_body_ids)    # after the reset, which restores eq_active
            mujoco.mj_forward(model, data)
            tool_start = {side: site_pose_rb(iks[side], data) for side in ROBOT_IPS}

            hand_start = {}
            for side in ROBOT_IPS:
                H0 = vector_to_htm(W_start[side])
                hand_start[side] = (H0[:3, 3].copy(), H0[:3, :3].copy())

            # ── 6. Fixed rotation from the UR base to {RB} ──────────────────
            # R(RB<-H) = R(RB<-World) . R(World<-H)
            for side in ROBOT_IPS:
                check_proper_rotation(R_WORLD_H[side], f"R_WORLD_H['{side}']")
            R_RB_H = {side: iks[side].R_rb.T @ R_WORLD_H[side] for side in ROBOT_IPS}
            print("Computed R_RB_H successfully:", R_RB_H)

            # ── 7. Joint commands, warm-started from the sim's starting pose ─
            q_cmd = {side: data.qpos[iks[side].qpos_ids].copy() for side in ROBOT_IPS}
            for side in ROBOT_IPS:
                iks[side].q_last[:] = q_cmd[side]

            ik_ok = {side: True for side in ROBOT_IPS}
            ik_ms_sum = 0.0
            ik_ms_max = 0.0
            ik_calls = 0
            loop = 0

            # ── 8. Teleoperation loop ───────────────────────────────────────
            while running and viewer.is_running():
                loop_start = time.perf_counter()

                # a) Keep both physical arms compliant (must be re-sent every loop).
                for controller in controllers.values():
                    controller.forceMode(TASK_FRAME, SELECTION_VECTOR, WRENCH, FORCE_TYPE, LIMITS)

                # b) Read both TCP poses.
                poses = {side: receiver.getActualTCPPose() for side, receiver in receivers.items()}

                # c) Inverse kinematics: hand motion -> tool joint values.
                ik_t0 = time.perf_counter()
                for side, pose in poses.items():
                    p_target, R_target = hand_to_tool_target(
                        pose, hand_start[side], tool_start[side], R_RB_H[side]
                    )
                    result = iks[side].solve(p_target, mat_to_quat(R_target), frame="rb")

                    # Good solution: use it. Otherwise hold the previous command.
                    if result.converged:
                        q_cmd[side] = result.q

                    # Report only when the status changes.
                    if result.converged != ik_ok[side]:
                        ik_ok[side] = result.converged
                        if result.converged:
                            status = "recovered"
                        else:
                            status = (
                                "FAILED (target unreachable? "
                                f"pos_err={result.pos_err * 1000:.1f} mm, "
                                f"rot_err={result.rot_err * 1000:.0f} mrad), "
                                "holding last command"
                            )
                        print(f"[{side}] IK {status}")

                ik_ms = (time.perf_counter() - ik_t0) * 1000.0
                ik_ms_sum += ik_ms
                ik_ms_max = max(ik_ms_max, ik_ms)
                ik_calls += 1
                if ik_calls % IK_REPORT_EVERY == 0:
                    # The loop budget is CONTROL_PERIOD_S. If the average approaches it,
                    # lower MAX_ITERS in IK_solver.py or lengthen the period.
                    print(
                        f"IK time for both arms: avg {ik_ms_sum / IK_REPORT_EVERY:.2f} ms, "
                        f"max {ik_ms_max:.2f} ms (budget {CONTROL_PERIOD_S * 1000:.0f} ms)"
                    )
                    ik_ms_sum = 0.0
                    ik_ms_max = 0.0

                # d) Update the simulation: markers, joint commands, physics.
                with viewer.lock():
                    for side, pose in poses.items():
                        pos, quat = transform_pose(pose, FRAME_OFFSETS[side])
                        data.mocap_pos[mocap_ids[side]] = pos
                        data.mocap_quat[mocap_ids[side]] = quat
                    for side in ROBOT_IPS:
                        data.ctrl[act_ids[side]] = q_cmd[side]
                    for _ in range(steps_per_loop):
                        mujoco.mj_step(model, data)

                # e) Refresh the viewer now and then.
                loop += 1
                if loop % RENDER_EVERY_N_LOOPS == 0:
                    viewer.sync()

                # f) Publish to Unity at STATE_SEND_HZ.
                now = time.perf_counter()
                if now >= next_unity_send:
                    unity_seq += 1

                    # Odd packets carry the left bodies, even packets the right ones.
                    packet_side = "left" if unity_seq % 2 == 1 else "right"

                    payload = build_unity_packet(
                        model, data, unity_seq, unity_bodies[packet_side], now
                    )

                    if unity_seq <= 2:    # startup check only
                        print(f"[Unity bridge] seq={unity_seq} side={packet_side} bytes={len(payload)}")
                    if len(payload) > MAX_UDP_BYTES:
                        print(f"WARNING: Unity packet {unity_seq} is {len(payload)} bytes "
                              f"(>{MAX_UDP_BYTES}).")

                    unity_tx.sendto(payload, (UNITY_HOST, STATE_PORT))

                    next_unity_send += unity_period
                    # If execution stalled, resynchronize instead of sending a burst to catch up.
                    if next_unity_send < now - unity_period:
                        next_unity_send = now + unity_period

                # g) Sleep off the rest of the control period.
                remaining = CONTROL_PERIOD_S - (time.perf_counter() - loop_start)
                if remaining > 0:
                    time.sleep(remaining)

    finally:
        print("Stopping force mode and disconnecting...")

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