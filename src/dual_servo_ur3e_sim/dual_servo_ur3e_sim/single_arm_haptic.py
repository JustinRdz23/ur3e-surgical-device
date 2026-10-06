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

MJCF_PATH = "/home/justinrc/workspace/src/dual_servo_ur3e_sim/mjcf/simpletool.xml"

TASK_FRAME = [0, 0, 0, 0, 0, 0] #Frame that will applied robot
SELECTION_VECTOR = [1, 1, 1, 1, 1, 1]
WRENCH = [0, 0, 0, 0, 0, 0]
FORCE_TYPE = 2
LIMITS = [0.1, 0.1, 1.5, 3.14, 3.14, 0.5]
DAMPING = 0.004
GAIN_SCALING = 1.0

CONTROL_PERIOD_S = 0.002    
RENDER_EVERY_N_STEPS = 20   

# Fixed 180-degree rotation about Y, as a quaternion (w, x, y, z).
Q_OFFSET = np.array([0.0, 0.0, 1.0, 0.0])  # cos(90deg)=0, axis=(0,1,0)*sin(90deg)=1

# Translation offset: where the robot's base frame origin sits in MuJoCo
# world coordinates.
FRAME_OFFSET_POS = np.array([0.0, 0.0, 0.8])

# Arrow visualization scale/width
ARROW_LENGTH_SCALE = 0.01
ARROW_WIDTH = 0.004

# How strongly the measured contact force is applied to the real robot
FORCE_SCALE = 0.4


def draw_force_arrow(viewer, start, force_vec):
    scn = viewer.user_scn

    if start is None or np.linalg.norm(force_vec) < 1e-6:
        scn.ngeom = 0
        return

    end = start + force_vec * ARROW_LENGTH_SCALE

    mujoco.mjv_initGeom(
        scn.geoms[0],
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        size=np.zeros(3),
        pos=np.zeros(3),
        mat=np.eye(3).flatten(),
        rgba=np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )
    mujoco.mjv_connector(
        scn.geoms[0],
        mujoco.mjtGeom.mjGEOM_ARROW,
        ARROW_WIDTH,
        start,
        end,
    )
    scn.ngeom = 1


def main():
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)

    print(f"Connecting to robot at {ROBOT_IP}...")
    rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
    rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
    print("Connected.")

    time.sleep(1.0)  
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
    # prev_quat = np.array([1.0, 0.0, 0.0, 0.0])  # for sign-continuity fix

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            print("Entering force mode + MuJoCo loop...")

            while running and viewer.is_running():
                loop_start = time.time()

                #TCP Position Reading
                rtde_c.forceMode(TASK_FRAME, SELECTION_VECTOR, WRENCH, FORCE_TYPE, LIMITS)
                current_pose = rtde_r.getActualTCPPose()  # [x, y, z, rx, ry, rz]
                #Position offset transformation, quaternion rotation to describe the positions accordingly in Mujoco
                pos_transformed = np.zeros(3)
                mujoco.mju_rotVecQuat(pos_transformed, np.array(current_pose[0:3]), Q_OFFSET)
                pos_transformed += FRAME_OFFSET_POS
                data.mocap_pos[0] = pos_transformed

                #Orientation transformation
                rot_vec = np.array(current_pose[3:6])
                angle = np.linalg.norm(rot_vec) #we obtain the rotation vector from the (rx,ry,yz) to create the robot rotation quaternion
                #this to operate this quaternion with the Q_OFFSET and rotate the frame.

                quat_robot = np.array([1.0, 0.0, 0.0, 0.0])
                if angle > 1e-9:
                    axis = rot_vec / angle
                    mujoco.mju_axisAngle2Quat(quat_robot, axis, angle)

                quat_final = np.zeros(4) #quaternion rotation formula simplifies since rotation doen't require the compensation of a conjugate quaternion
                mujoco.mju_mulQuat(quat_final, Q_OFFSET, quat_robot)

                # # sign-continuity fix (fixes weldment issue of vibration)
                # if np.dot(quat_final, prev_quat) < 0:
                #     quat_final = -quat_final
                # prev_quat = quat_final.copy()

                #Contact Force Measure
                #ID of interacted bodies
                b1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
                b2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_roll_link_Shape_IndexedFaceSet")
                #result vector for contact force
                total_force = np.zeros(3)
                contact_point = None

                #Find contact of interest
                for i in range(data.ncon):
                    contact = data.contact[i]
                    b1 = model.geom_bodyid[contact.geom1]
                    b2 = model.geom_bodyid[contact.geom2]
                #Sum of force vectors
                    if {b1,b2} == {b1_id,b2_id}:
                        result_force = np.zeros(6)
                        mujoco.mj_contactForce(model, data,i,result_force )
                        #Force transformation into world coordinates
                        R = np.array(contact.frame).reshape(3, 3)
                        force_world = R.T @ result_force[0:3]

                        total_force += force_world
                        contact_point = np.array(contact.pos)

                #Force transformation into world coordinates.
                #Since its a force, no translation is required, just rotation. Therefore using the cinjugate of the Q_OFFSET quaternion:
                Q_OFFSET_INV = np.array([Q_OFFSET[0], -Q_OFFSET[1], -Q_OFFSET[2], -Q_OFFSET[3]])

                #Rotate the world-frame force back into the robot's base frame
                force_robot_frame = np.zeros(3)
                mujoco.mju_rotVecQuat(force_robot_frame, total_force, Q_OFFSET_INV)
                WRENCH[0:3] = (force_robot_frame * FORCE_SCALE).tolist()

                data.mocap_quat[0] = quat_final

                # data.ctrl[0] = q[5]  # wrist_3 -> tool roll
                # data.ctrl[1] = q[3]  # wrist_2 -> tool yaw

                step_count += 1
                if step_count % RENDER_EVERY_N_STEPS == 0:
                    mujoco.mj_step(model, data)
                    with viewer.lock():
                        draw_force_arrow(viewer, contact_point, total_force)
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