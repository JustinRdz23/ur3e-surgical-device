import time
import signal
import sys
import numpy as np
import mujoco
import mujoco.viewer
import rtde_control
import rtde_receive

ROBOT_IP = "192.168.0.3" #UR3e
# ROBOT_IP = "192.168.0.1"  #UR3

MJCF_PATH = "/home/justinrc/workspace/src/dual_servo_ur3e_sim/mjcf/hello.xml"

TASK_FRAME = [0, 0, 0, 0, 0, 0]
SELECTION_VECTOR = [1, 1, 1, 1, 1, 1]
WRENCH = [0, 0, 0, 0, 0, 0]
FORCE_TYPE = 2
LIMITS = [0.1, 0.1, 1.5, 3.14, 3.14, 0.5]
DAMPING = 0.004
GAIN_SCALING = 1.0

CONTROL_PERIOD_S = 0.002    # real 500Hz, not a busy-loop
RENDER_EVERY_N_STEPS = 20   # ~25Hz render off a 500Hz control loop

# Fixed 180-degree rotation about Y, as a quaternion (w, x, y, z).
Q_OFFSET = np.array([0.0, 0.0, 1.0, 0.0])  # cos(90deg)=0, axis=(0,1,0)*sin(90deg)=1

# Translation offset: where the robot's base frame origin sits in MuJoCo
# world coordinates. Tune this empirically -- command the robot to a known
# pose, see where it lands in the scene, adjust until it matches.
FRAME_OFFSET_POS = np.array([0.0, 0.0, 0.5])


def main():
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)

    print(f"Connecting to robot at {ROBOT_IP}...")
    rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
    rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
    print("Connected.")

    rtde_c.zeroFtSensor()
    rtde_c.forceModeSetDamping(DAMPING)
    rtde_c.forceModeSetGainScaling(GAIN_SCALING)

    running = True

    def handle_sigint(signum, frame):
        nonlocal running
        print("\nCtrl+C received, stopping...")
        running = False

    signal.signal(signal.SIGINT, handle_sigint)

    step_count = 0
    prev_quat = np.array([1.0, 0.0, 0.0, 0.0])  # for sign-continuity fix

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            print("Entering force mode + MuJoCo loop...")

            while running and viewer.is_running():
                loop_start = time.time()

                # --- Read robot state FIRST, before using it ---
                rtde_c.forceMode(TASK_FRAME, SELECTION_VECTOR, WRENCH, FORCE_TYPE, LIMITS)
                q = rtde_r.getActualQ()  # [base, shoulder, elbow, wrist1, wrist2, wrist3]
                current_pose = rtde_r.getActualTCPPose()  # [x, y, z, rx, ry, rz]
            
                pos_transformed = np.zeros(3)
                mujoco.mju_rotVecQuat(pos_transformed, np.array(current_pose[0:3]), Q_OFFSET)
                pos_transformed += FRAME_OFFSET_POS
                data.mocap_pos[0] = pos_transformed

                # --- Orientation: convert to quaternion, compose with offset ---
                rot_vec = np.array(current_pose[3:6])
                angle = np.linalg.norm(rot_vec)

                quat_robot = np.array([1.0, 0.0, 0.0, 0.0])
                if angle > 1e-9:
                    axis = rot_vec / angle
                    mujoco.mju_axisAngle2Quat(quat_robot, axis, angle)

                quat_final = np.zeros(4)
                mujoco.mju_mulQuat(quat_final, Q_OFFSET, quat_robot)

                # --- Sign-continuity fix (avoids the weld jiggle) ---
                if np.dot(quat_final, prev_quat) < 0:
                    quat_final = -quat_final
                prev_quat = quat_final.copy()

                data.mocap_quat[0] = quat_final

                # data.ctrl[0] = q[5]  # wrist_3 -> tool roll
                # data.ctrl[1] = q[3]  # wrist_2 -> tool yaw

                step_count += 1
                if step_count % RENDER_EVERY_N_STEPS == 0:
                    mujoco.mj_step(model, data)
                    viewer.sync()

                elapsed = time.time() - loop_start
                sleep_time = CONTROL_PERIOD_S - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    finally:
        print("Stopping force mode and disconnecting...")
        rtde_c.forceModeStop()
        rtde_c.stopScript()
        print("Done.")


if __name__ == "__main__":
    main()