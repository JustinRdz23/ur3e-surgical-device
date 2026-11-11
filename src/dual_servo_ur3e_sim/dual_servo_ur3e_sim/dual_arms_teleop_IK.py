import signal
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer
import rtde_control
import rtde_receive

from scipy.spatial.transform import Rotation as R

from IK_solver import PoseIK, JOINT_SUFFIXES, mat_to_quat
import numpy as np

LEFT_IP = "192.168.0.1"    # UR3
RIGHT_IP = "192.168.0.3"   # UR3e

MJCF_PATH = str(Path(__file__).resolve().parent.parent / "mjcf" / "teleop_dual_IK.xml")
MOCAP_BODIES = {"left": "left_arm_mocap", "right": "right_arm_mocap"}
PORT_BODIES = {"left": "left_port", "right": "right_port"}

# forceMode with a zero wrench
TASK_FRAME = [0, 0, 0, 0, 0, 0]
SELECTION_VECTOR = [1, 1, 1, 1, 1, 1]
WRENCH = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
FORCE_TYPE = 2
LIMITS = [0.1, 0.1, 1.5, 3.14, 3.14, 0.5]
DAMPING = 0.004
GAIN_SCALING = 1.0

CONTROL_PERIOD_S = 0.002
RENDER_EVERY_N_LOOPS = 10


# Fixed 180-degree rotation about Y, quaternion (w, x, y, z), robot base -> MuJoCo world.
Q_OFFSET = np.array([0.0, 0.0, 1.0, 0.0])


# Where each robot's base origin sits in MuJoCo world coordinates. 
FRAME_OFFSETS = {
    "left": np.array([-0.3, 0.0, 0.8]),
    "right": np.array([0.3, 0.0, 0.8]),
}

R_WORLD_H = {
    "left": np.diag([-1.0, 1.0, -1.0]),
    "right": np.diag([-1.0, 1.0, -1.0]),
}

SCALE = 0.5

# Function to detect the mocap bodies and update their position. Validates if the body exists
# if it does then it is evaluated as a mocap or not. If it does it raises True if not, error.
def get_mocap_id(model, body_name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0: #Identifies body exisiting inside the simulation
        raise ValueError(f"Body not found: {body_name}")
    mocap_id = model.body_mocapid[body_id]
    if mocap_id < 0: #Identifies body as a mocap
        raise ValueError(f"Body is not a mocap body: {body_name}")
    return mocap_id

#Converts the UR TCP frame into a format that corresponds correctly into Mujoco's simulation.
#Recieves the pose vector [x,y,z,rx,ry,rz] and the offset established before hand corresponding to each arms phyisical location to each other.
def transform_pose(pose, frame_offset):
    pos = np.zeros(3)
    mujoco.mju_rotVecQuat(pos, np.array(pose[:3]), Q_OFFSET) #Takes position of the respective TCP, then rotates it 180° on Y to adjust the position into Mujocos World frame.
    pos += frame_offset 

    #Since Mujoco requires quaternions, turns the corresponding vector of the TCP into quat
    rot_vec = np.array(pose[3:6])
    angle = np.linalg.norm(rot_vec) #Obtains the magnitude of the rotation vector from the TCP
    quat_robot = np.array([1.0, 0.0, 0.0, 0.0]) 
    if angle > 1e-9:
        mujoco.mju_axisAngle2Quat(quat_robot, rot_vec / angle, angle) #Defines the quaternion of rotation from the TCP vector.
        #Turns a rotation axis into quaternion. Uses the sole axis (rot_vec) and the magnitude (angle) to create a quat (quat_robot)

    quat = np.zeros(4)
    mujoco.mju_mulQuat(quat, Q_OFFSET, quat_robot) #implements the same offset as the position vector. The Y rotation of 180°
    return pos, quat

#Transforms from vector to Homogenous Transformation Matrix
def vector_to_htm(tcp_vector):
    # tcp_vector = [x, y, z, rx, ry, rz]
    mat = np.eye(4)
    mat[:3, 3] = tcp_vector[:3]
    mat[:3, :3] = R.from_rotvec(tcp_vector[3:]).as_matrix()
    return mat

import numpy as np
import mujoco

def transform_world_point_to_site_frame(model, data, site_name: str, p_world: np.ndarray) -> np.ndarray:

    mujoco.mj_forward(model, data)
    
    # 1. Get site ID (works for sites, can use mjOBJ_BODY for bodies)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id == -1:
        raise ValueError(f"Site '{site_name}' not found.")
        
    # 2. Extract position and rotation from MuJoCo data
    p_rb_world = data.site_xpos[site_id]             # Position [3x1]
    R_world_rb = data.site_xmat[site_id].reshape(3, 3) # Rotation matrix [3x3]
    
    # 3. Build the 4x4 Forward HTM (^W T_RB)
    T_world_rb = np.eye(4)
    T_world_rb[:3, :3] = R_world_rb
    T_world_rb[:3, 3] = p_rb_world
    
    # 4. Invert the HTM to get (^RB T_W)
    # Note: For rigid transformations, numpy's linalg.inv is safe, 
    # but block-inverse is mathematically cleaner:
    R_transpose = R_world_rb.T
    T_rb_world = np.eye(4)
    T_rb_world[:3, :3] = R_transpose
    T_rb_world[:3, 3] = -R_transpose @ p_rb_world
    
    # 5. Convert world point to homogeneous vector [x, y, z, 1]^T
    p_world_homog = np.array([p_world[0], p_world[1], p_world[2], 1.0])
    
    # 6. Apply transformation matrix
    p_rb_homog = T_rb_world @ p_world_homog
    
    # Return just the 3D coordinates (drop the trailing 1)
    return p_rb_homog[:3]


def site_pose_rb(ik, data):
    """The TOOL's end-effector pose expressed in {RB} (the port frame at zero joints).

    This is P0 in the paper's formula. It is the pose of the simulated tool, NOT of the
    hand. {RB} is ik.p_rb / ik.R_rb, the fixed pose of the port body (the RCM).
    Returns (position 3-vector, rotation matrix 3x3). Call after mj_forward.
    """
    p_world = data.site_xpos[ik.site_id]
    R_world = data.site_xmat[ik.site_id].reshape(3, 3)
    return ik.R_rb.T @ (p_world - ik.p_rb), ik.R_rb.T @ R_world


def get_actuator_ids_for_joints(model, side):
    """Finds the actuator that drives each of the six IK joints, in JOINT_SUFFIXES order.

    It looks the actuators up by which JOINT they drive, so you do not have to know their
    names: for a joint actuator, model.actuator_trnid[a, 0] holds the joint's id.
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
def main():
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)

    #Obtains the Mocap IDs and creates a new dictionary that integrates only mocap
    mocap_ids = {side: get_mocap_id(model, name) for side, name in MOCAP_BODIES.items()}
    ips = {"left": LEFT_IP, "right": RIGHT_IP}
    steps_per_loop = max(1, round(CONTROL_PERIOD_S / model.opt.timestep))

    iks = {side: PoseIK(model, side, port_body=PORT_BODIES[side]) for side in ips}
    act_ids = {side: get_actuator_ids_for_joints(model, side) for side in ips}
    mocap_body_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in MOCAP_BODIES.values()
    ]
    controllers = {}
    receivers = {}

    running = True

    #Safety Stop
    def handle_sigint(signum, frame):
        nonlocal running
        print("\nCtrl+C received, stopping...")
        running = False

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        #Runs over the IPs and enables the controller and reciever port of each.
        for side, ip in ips.items():
            print(f"Connecting to {side} robot at {ip}...")
            controllers[side] = rtde_control.RTDEControlInterface(ip)
            receivers[side] = rtde_receive.RTDEReceiveInterface(ip)

        time.sleep(1.0)
        for c in controllers.values():
            c.zeroFtSensor() #Resets the force/torque sensor at the TCP
            c.forceModeSetDamping(DAMPING) 
            c.forceModeSetGainScaling(GAIN_SCALING)
        print("Connected. Both arms become compliant now.")

        #Enables Mujoco simulation scene. Begins Teleoperation Loop
        with mujoco.viewer.launch_passive(model, data) as viewer:
            print("Dual-arm teleoperation running.")

            #Read initial position of the tool in UR space
            W_start = {side: r.getActualTCPPose() for side, r in receivers.items()}

            if model.nkey > 0:
                mujoco.mj_resetDataKeyframe(model, data, 0)
            disable_mocap_welds(model, data, mocap_body_ids)   # after the reset: it restores eq_active
            mujoco.mj_forward(model, data)
            #Read initial position of the tool in Mujoco
            P_start = {side: site_pose_rb(iks[side], data) for side in ips}
        
            mujoco.mj_forward(model, data)

        #R(RB←H) = R(RB←World) · R(World←H)
            for side in ips:
                Rw = R_WORLD_H[side]
                if not (np.allclose(Rw.T @ Rw, np.eye(3)) and np.isclose(np.linalg.det(Rw), 1.0)):
                    raise ValueError(
                        f"R_WORLD_H['{side}'] must be a proper rotation (orthonormal, det +1). "
                        "A determinant of -1 is a mirror, which was the original inversion bug."
                    )
            R_RB_H = {side: iks[side].R_rb.T @ R_WORLD_H[side] for side in ips}
            print("Computed R_RB_H successfully:", R_RB_H)

            hand_start = {}
            for side in ips:
                H0 = vector_to_htm(W_start[side])
                hand_start[side] = (H0[:3, 3].copy(), H0[:3, :3].copy())
            tool_start = P_start

            q_cmd = {side: data.qpos[iks[side].qpos_ids].copy() for side in ips}
            for side in ips:
                iks[side].q_last[:] = q_cmd[side]   # warm-start the solver from the same place
            ik_ok = {side: True for side in ips}
            ik_ms_sum = 0.0
            ik_ms_max = 0.0
            ik_calls = 0
            loop = 0
            while running and viewer.is_running():
                loop_start = time.perf_counter()

                # Must be re-sent every loop to keep the arms compliant.
                for c in controllers.values():
                    c.forceMode(TASK_FRAME, SELECTION_VECTOR, WRENCH, FORCE_TYPE, LIMITS) #Enables force mode with a zero wrench and the previously placed parameters for this function.

                poses = {side: r.getActualTCPPose() for side, r in receivers.items()} #New dictionary that stores the pose of the TCP for each arm.

                ik_t0 = time.perf_counter()
                for side, pose in poses.items():
                    H1 = vector_to_htm(pose)
                    p_h1, R_h1 = H1[:3, 3], H1[:3, :3]
                    p_h0, R_h0 = hand_start[side]
                    p_rb0, R_rb0 = tool_start[side]
                    M = R_RB_H[side]

                    p_target = p_rb0 + SCALE * (M @ (p_h1 - p_h0))
                    R_target = (M @ (R_h1 @ R_h0.T) @ M.T) @ R_rb0

                    res = iks[side].solve(p_target, mat_to_quat(R_target), frame="rb")
                    if res.converged:
                        q_cmd[side] = res.q          # good solution: use it
                    # else: keep the previous q_cmd (hold position) instead of sending garbage
                    if res.converged != ik_ok[side]:
                        ik_ok[side] = res.converged
                        state = "recovered" if res.converged else (
                            f"FAILED (target unreachable? pos_err={res.pos_err * 1000:.1f} mm, "
                            f"rot_err={res.rot_err * 1000:.0f} mrad), holding last command")
                        print(f"[{side}] IK {state}")
                ik_ms = (time.perf_counter() - ik_t0) * 1000.0
                ik_ms_sum += ik_ms
                ik_ms_max = max(ik_ms_max, ik_ms)
                ik_calls += 1
                if ik_calls % 500 == 0:
                    # Your loop budget is CONTROL_PERIOD_S (2 ms). If the average is above
                    # about 1 ms, lower MAX_ITERS in IK_solver.py or slow the loop down.
                    print(f"IK time for both arms: avg {ik_ms_sum / 500:.2f} ms, max {ik_ms_max:.2f} ms")
                    ik_ms_sum = 0.0
                    ik_ms_max = 0.0
                with viewer.lock():
                    for side, pose in poses.items():
                        pos, quat = transform_pose(pose, FRAME_OFFSETS[side]) #Unpacks the tuple for the position and rotation quaternion, stores them new
                        data.mocap_pos[mocap_ids[side]] = pos #mocap_pos is an array that holds the position of each of the mocap bodies indentified.
                        data.mocap_quat[mocap_ids[side]] = quat #mocap_quat is the array that holds the rotation quaternion of each as well.
                    for side in ips:
                        data.ctrl[act_ids[side]] = q_cmd[side]
                    for _ in range(steps_per_loop):
                        mujoco.mj_step(model, data)

                loop += 1
                if loop % RENDER_EVERY_N_LOOPS == 0:
                    viewer.sync()

                remaining = CONTROL_PERIOD_S - (time.perf_counter() - loop_start)
                if remaining > 0:
                    time.sleep(remaining)

    finally:
        print("Stopping force mode and disconnecting...")
        for side, c in controllers.items():
            try:
                c.forceModeStop()
                c.stopScript()
            except Exception as e:
                print(f"[{side}] error stopping force mode: {e}")
        for side, r in receivers.items():
            try:
                r.disconnect()
            except Exception as e:
                print(f"[{side}] error disconnecting: {e}")
        print("Done.")


if __name__ == "__main__":
    main()