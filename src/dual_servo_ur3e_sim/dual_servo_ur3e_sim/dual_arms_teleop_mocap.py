"""Dual-arm teleop for teleop_dual_tool_scene.xml (ur_rtde, no ROS, no haptics).

Each robot's TCP pose drives a mocap body (left_arm_mocap / right_arm_mocap) that is welded
to a tool in the scene.  forceMode() with a zero wrench keeps both arms freely movable by hand
(this is what makes them compliant); nothing from the sim is sent back to the robots.

    python3 dual_arm_teleop.py

Ctrl+C or closing the viewer stops force mode on both robots and disconnects.
"""
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


def main():
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)

    #Obtains the Mocap IDs and creates a new dictionary that integrates only mocap
    mocap_ids = {side: get_mocap_id(model, name) for side, name in MOCAP_BODIES.items()}
    ips = {"left": LEFT_IP, "right": RIGHT_IP}
    steps_per_loop = max(1, round(CONTROL_PERIOD_S / model.opt.timestep))

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
            loop = 0
            while running and viewer.is_running():
                loop_start = time.perf_counter()

                # Must be re-sent every loop to keep the arms compliant.
                for c in controllers.values():
                    c.forceMode(TASK_FRAME, SELECTION_VECTOR, WRENCH, FORCE_TYPE, LIMITS) #Enables force mode with a zero wrench and the previously placed parameters for this function.

                poses = {side: r.getActualTCPPose() for side, r in receivers.items()} #New dictionary that stores the pose of the TCP for each arm.

                with viewer.lock():
                    for side, pose in poses.items():
                        pos, quat = transform_pose(pose, FRAME_OFFSETS[side]) #Unpacks the tuple for the position and rotation quaternion, stores them new
                        data.mocap_pos[mocap_ids[side]] = pos #mocap_pos is an array that holds the position of each of the mocap bodies indentified.
                        data.mocap_quat[mocap_ids[side]] = quat #mocap_quat is the array that holds the rotation quaternion of each as well.
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