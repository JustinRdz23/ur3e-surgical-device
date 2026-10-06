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

STIFFNESS_LIN = 1000.0
DAMPING_LIN = 15.0
FORCE_SCALE = 0.4
MAX_FORCE = 20.0

VELOCITY_FILTER_ALPHA = 0.15


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
    prev_gap_vel = np.zeros(3)
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

                        prev_gap_vel = np.zeros(3)
                        prev_time = time.time()

                        startup_complete = True

                        print()
                        print("==========================================")
                        print("VIRTUAL COUPLING STARTED")
                        print("==========================================")
                        print(f"Stiffness:          {STIFFNESS_LIN:.2f} N/m")
                        print(f"Damping:            {DAMPING_LIN:.2f} N/(m/s)")
                        print(f"Velocity filter:    {VELOCITY_FILTER_ALPHA:.2f}")
                        print(f"Initial gap:        {prev_gap_pos}")
                        print(f"Initial |gap|:      {np.linalg.norm(prev_gap_pos):.6f} m")
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

                raw_gap_vel = (
                    gap_pos - prev_gap_pos
                ) / dt

                gap_vel = (
                    VELOCITY_FILTER_ALPHA * raw_gap_vel
                    + (1.0 - VELOCITY_FILTER_ALPHA) * prev_gap_vel
                )

                prev_gap_pos = gap_pos.copy()
                prev_gap_vel = gap_vel.copy()
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

                data.xfrc_applied[
                    sphere_id,
                    0:3
                ] = spring_force_on_proxy

                data.xfrc_applied[
                    sphere_id,
                    3:6
                ] = 0.0

                mujoco.mj_step(
                    model,
                    data
                )

                force_on_device_world = (
                    -spring_force_on_proxy
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
                    print(f"Step:               {step_count}")
                    print(f"dt:                 {dt:.6f} s")
                    print(f"Gap:                {gap_pos}")
                    print(f"|Gap|:              {np.linalg.norm(gap_pos):.6f} m")
                    print(f"Raw gap velocity:   {raw_gap_vel}")
                    print(f"Raw |velocity|:     {np.linalg.norm(raw_gap_vel):.6f} m/s")
                    print(f"Filtered velocity:  {gap_vel}")
                    print(f"Filtered |velocity|:{np.linalg.norm(gap_vel):.6f} m/s")
                    print(f"Spring force:       {spring_force}")
                    print(f"Spring |F|:         {np.linalg.norm(spring_force):.4f} N")
                    print(f"Damping force:      {damping_force}")
                    print(f"Damping |F|:        {np.linalg.norm(damping_force):.4f} N")
                    print(f"Total proxy force:  {spring_force_on_proxy}")
                    print(f"Total |F|:          {np.linalg.norm(spring_force_on_proxy):.4f} N")
                    print(f"Robot force:        {force_robot_limited}")
                    print(f"Robot |F|:          {np.linalg.norm(force_robot_limited):.4f} N")
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