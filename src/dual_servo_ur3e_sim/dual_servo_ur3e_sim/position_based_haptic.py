import time
import numpy as np
import mujoco
import mujoco.viewer
import rtde_control
import rtde_receive

ROBOT_IP = "192.168.0.3"
MJCF_PATH = "/home/justinrc/workspace/src/dual_servo_ur3e_sim/mjcf/sphere_test.xml"

TASK_FRAME = [0, 0, 0, 0, 0, 0]
SELECTION_VECTOR = [1, 1, 1, 1, 1, 1]
WRENCH = [0, 0, 0, 0, 0, 0]
FORCE_TYPE = 2
LIMITS = [0.1, 0.1, 1.5, 3.14, 3.14, 0.5]
DAMPING = 0.004
GAIN_SCALING = 1.0

CONTROL_PERIOD_S = 0.002
RENDER_EVERY_N_STEPS = 20
PRINT_EVERY_N_STEPS = 100

STARTUP_DELAY_S = 2.0

Q_OFFSET = np.array([0.0, 0.0, 1.0, 0.0])
Q_OFFSET_INV = np.array([
    Q_OFFSET[0],
    -Q_OFFSET[1],
    -Q_OFFSET[2],
    -Q_OFFSET[3]
])  

FRAME_OFFSET_POS = np.array([0.0, 0.0, 0.8])

ARROW_LENGTH_SCALE = 0.01
ARROW_WIDTH = 0.004

STIFFNESS_LIN = 2200.0
DAMPING_LIN = 15.0
FORCE_SCALE = 0.4
MAX_FORCE = 100.0
INTERNAL_MAX_FORCE = 100.0



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
        rgba=np.array(
            [1.0, 0.0, 0.0, 1.0],
            dtype=np.float32
        ),
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

    sphere_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "sphere"
    )

    if sphere_id < 0:
        raise RuntimeError(
            "Could not find body named 'sphere' in the MJCF."
        )

    rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
    rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)

    time.sleep(1.0)

    rtde_c.zeroFtSensor()
    rtde_c.forceModeSetDamping(DAMPING)
    rtde_c.forceModeSetGainScaling(GAIN_SCALING)

    running = True
    startup_complete = False
    step_count = 0

    prev_gap_pos = None
    prev_time = None

    program_start_time = time.time()

    try:
        with mujoco.viewer.launch_passive(
            model,
            data
        ) as viewer:

            while running and viewer.is_running():

                loop_start = time.time()

                current_pose = rtde_r.getActualTCPPose()

                pos_transformed = np.zeros(3)

                mujoco.mju_rotVecQuat(
                    pos_transformed,
                    np.array(current_pose[0:3]),
                    Q_OFFSET
                )

                pos_transformed += FRAME_OFFSET_POS

                data.mocap_pos[0] = pos_transformed

                elapsed_since_start = (
                    time.time() - program_start_time
                )

                if not startup_complete:

                    WRENCH[0:3] = [0.0, 0.0, 0.0]

                    data.xfrc_applied[
                        sphere_id,
                        0:3
                    ] = 0.0

                    data.xfrc_applied[
                        sphere_id,
                        3:6
                    ] = 0.0

                    if elapsed_since_start >= STARTUP_DELAY_S:

                        data.qpos[0:3] = data.mocap_pos[0]

                        data.qvel[0:3] = 0.0
                        data.qvel[3:6] = 0.0

                        mujoco.mj_forward(
                            model,
                            data
                        )

                        sphere_pos = (
                            data.xpos[sphere_id].copy()
                        )

                        prev_gap_pos = (
                            data.mocap_pos[0].copy()
                            - sphere_pos
                        )

                        prev_time = time.time()

                        startup_complete = True

                        print()
                        print("==========================================")
                        print("VIRTUAL COUPLING STARTED")
                        print("==========================================")
                        print(
                            f"Stiffness:          "
                            f"{STIFFNESS_LIN:.2f} N/m"
                        )
                        print(
                            f"Damping:            "
                            f"{DAMPING_LIN:.2f} N/(m/s)"
                        )
                        print(
                            f"Initial gap:        "
                            f"{prev_gap_pos}"
                        )
                        print(
                            f"Initial |gap|:      "
                            f"{np.linalg.norm(prev_gap_pos):.6f} m"
                        )
                        print("==========================================")
                        print()

                    elapsed = time.time() - loop_start

                    sleep_time = (
                        CONTROL_PERIOD_S - elapsed
                    )

                    if sleep_time > 0:
                        time.sleep(sleep_time)

                    continue

                rtde_c.forceMode(
                    TASK_FRAME,
                    SELECTION_VECTOR,
                    WRENCH,
                    FORCE_TYPE,
                    LIMITS
                )

                mocap_pos = (
                    data.mocap_pos[0].copy()
                )

                sphere_pos = (
                    data.xpos[sphere_id].copy()
                )

                gap_pos = (
                    mocap_pos - sphere_pos
                )

                now = time.time()

                dt = max(
                    now - prev_time,
                    1e-4
                )

                gap_vel = (
                    gap_pos - prev_gap_pos
                ) / dt

                prev_gap_pos = gap_pos.copy()
                prev_time = now

                spring_force = (
                    STIFFNESS_LIN * gap_pos
                )

                damping_force = (
                    DAMPING_LIN * gap_vel
                )

                spring_force_on_proxy = (
                    spring_force
                    + damping_force
                )

                proxy_force_limited = np.clip(
                    spring_force_on_proxy,
                    -INTERNAL_MAX_FORCE,
                    INTERNAL_MAX_FORCE
                )

                data.xfrc_applied[
                    sphere_id,
                    0:3
                ] = proxy_force_limited

                data.xfrc_applied[
                    sphere_id,
                    3:6
                ] = 0.0

                mujoco.mj_step(
                    model,
                    data
                )

                force_on_device_world = (
                    -proxy_force_limited
                )
                force_robot_frame = np.zeros(3)

                mujoco.mju_rotVecQuat(
                    force_robot_frame,
                    force_on_device_world,
                    Q_OFFSET_INV
                )

                force_robot_scaled = (
                    force_robot_frame * FORCE_SCALE
                )

                force_robot_limited = np.clip(
                    force_robot_scaled,
                    -MAX_FORCE,
                    MAX_FORCE
                )

                WRENCH[0:3] = (
                    force_robot_limited.tolist()
                )

                step_count += 1

                if step_count % PRINT_EVERY_N_STEPS == 0:

                    print()
                    print("--------------- DEBUG ---------------")
                    print(
                        f"Step:               "
                        f"{step_count}"
                    )
                    print(
                        f"dt:                 "
                        f"{dt:.6f} s"
                    )
                    print(
                        f"Gap:                "
                        f"{gap_pos}"
                    )
                    print(
                        f"|Gap|:              "
                        f"{np.linalg.norm(gap_pos):.6f} m"
                    )
                    print(
                        f"Gap velocity:       "
                        f"{gap_vel}"
                    )
                    print(
                        f"|velocity|:         "
                        f"{np.linalg.norm(gap_vel):.6f} m/s"
                    )
                    print(
                        f"Spring force:       "
                        f"{spring_force}"
                    )
                    print(
                        f"Spring |F|:         "
                        f"{np.linalg.norm(spring_force):.4f} N"
                    )
                    print(
                        f"Damping force:      "
                        f"{damping_force}"
                    )
                    print(
                        f"Damping |F|:        "
                        f"{np.linalg.norm(damping_force):.4f} N"
                    )
                    print(
                        f"Total proxy force:  "
                        f"{spring_force_on_proxy}"
                    )
                    print(
                        f"Total |F|:          "
                        f"{np.linalg.norm(spring_force_on_proxy):.4f} N"
                    )
                    print(
                        f"Robot force:        "
                        f"{force_robot_limited}"
                    )
                    print(
                        f"Robot |F|:          "
                        f"{np.linalg.norm(force_robot_limited):.4f} N"
                    )
                    print("--------------------------------------")

                if step_count % RENDER_EVERY_N_STEPS == 0:

                    sphere_pos = (
                        data.xpos[sphere_id].copy()
                    )

                    with viewer.lock():
                        draw_force_arrow(
                            viewer,
                            sphere_pos,
                            force_on_device_world
                        )

                    viewer.sync()

                elapsed = time.time() - loop_start

                sleep_time = (
                    CONTROL_PERIOD_S - elapsed
                )

                if sleep_time > 0:
                    time.sleep(sleep_time)

    finally:

        rtde_c.forceModeStop()
        rtde_c.stopScript()


if __name__ == "__main__":
    main()