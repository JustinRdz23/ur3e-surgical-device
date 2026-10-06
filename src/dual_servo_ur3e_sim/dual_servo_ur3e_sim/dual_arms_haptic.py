import time
import signal
import numpy as np
import mujoco
import mujoco.viewer
import rtde_control
import rtde_receive


# ============================================================
# ROBOTS
# ============================================================

ROBOT_IPS = [
    "192.168.0.3",   # Robot 1 - UR3e
    "192.168.0.1",   # Robot 2 - UR3e
]

MOCAP_IDS = [0, 1]

TOOL_BODY_NAMES = [
    "tool_roll_link_Shape_IndexedFaceSet",
    "tool_roll_link_Shape_IndexedFaceSet_2",
]

BOX_BODY_NAME = "box"


# ============================================================
# FORCE MODE
# ============================================================

TASK_FRAME = [0, 0, 0, 0, 0, 0]

SELECTION_VECTOR = [1, 1, 1, 1, 1, 1]

WRENCHES = [
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
]

FORCE_TYPE = 2

LIMITS = [
    [0.1, 0.1, 1.5, 3.14, 3.14, 0.5],
    [0.1, 0.1, 1.5, 3.14, 3.14, 0.5],
]

DAMPING = 0.004
GAIN_SCALING = 1.0
FORCE_SCALE = 0.4


# ============================================================
# TIMING
# ============================================================

CONTROL_PERIOD_S = 0.002

# MuJoCo rendering frequency
RENDER_EVERY_N_STEPS = 20


# ============================================================
# COORDINATE TRANSFORMATION
# ============================================================

Q_OFFSET = np.array([
    0.0,
    0.0,
    1.0,
    0.0
])


# Base translation of each robot in MuJoCo
FRAME_OFFSET_POS = [
    np.array([0.0, 0.0, 0.5]),    # Robot 1
    np.array([0.0, 0.0, 0.5]),    # Robot 2
]


# Additional separation between the two robots.
# Increase this value if you want them farther apart.
ROBOT_SEPARATION = 0.15


# ============================================================
# FORCE ARROW VISUALIZATION
# ============================================================

ARROW_LENGTH_SCALE = 0.01
ARROW_WIDTH = 0.004


def draw_force_arrows(viewer, arrows):

    scn = viewer.user_scn
    scn.ngeom = 0

    for start, force_vec in arrows:

        if start is None:
            continue

        if np.linalg.norm(force_vec) < 1e-6:
            continue

        geom = scn.geoms[scn.ngeom]

        end = start + force_vec * ARROW_LENGTH_SCALE

        mujoco.mjv_initGeom(
            geom,
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
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            ARROW_WIDTH,
            start,
            end,
        )

        scn.ngeom += 1


# ============================================================
# ROBOT POSE -> MUJOCO POSE
# ============================================================

def transform_robot_pose_to_mujoco(current_pose, robot_id):

    # --------------------------------------------------------
    # Position
    # --------------------------------------------------------

    pos_transformed = np.zeros(3)

    mujoco.mju_rotVecQuat(
        pos_transformed,
        np.array(current_pose[0:3]),
        Q_OFFSET
    )

    # --------------------------------------------------------
    # Mirror the two robots
    #
    # Robot 1 is mirrored in X.
    #
    # Then move each robot outward by ROBOT_SEPARATION.
    # --------------------------------------------------------

    if robot_id == 0:

        pos_transformed[0] = (
            -pos_transformed[0]
            - ROBOT_SEPARATION
        )

    else:

        pos_transformed[0] = (
            pos_transformed[0]
            + ROBOT_SEPARATION
        )

    # --------------------------------------------------------
    # Base frame translation
    # --------------------------------------------------------

    pos_transformed += FRAME_OFFSET_POS[robot_id]

    # --------------------------------------------------------
    # Orientation
    # --------------------------------------------------------

    rot_vec = np.array(current_pose[3:6])

    angle = np.linalg.norm(rot_vec)

    quat_robot = np.array([
        1.0,
        0.0,
        0.0,
        0.0
    ])

    if angle > 1e-9:

        axis = rot_vec / angle

        mujoco.mju_axisAngle2Quat(
            quat_robot,
            axis,
            angle
        )

    # Apply coordinate-system rotation
    quat_final = np.zeros(4)

    mujoco.mju_mulQuat(
        quat_final,
        Q_OFFSET,
        quat_robot
    )

    return pos_transformed, quat_final


# ============================================================
# CONTACT FORCE
# ============================================================

def get_contact_force(
    model,
    data,
    box_body_id,
    tool_body_id
):

    total_force = np.zeros(3)

    contact_point = None

    for i in range(data.ncon):

        contact = data.contact[i]

        b1 = model.geom_bodyid[contact.geom1]
        b2 = model.geom_bodyid[contact.geom2]

        # Check whether this contact is between
        # the box and the current tool.
        if {b1, b2} == {box_body_id, tool_body_id}:

            result_force = np.zeros(6)

            mujoco.mj_contactForce(
                model,
                data,
                i,
                result_force
            )

            # Contact frame -> world frame
            R = np.array(
                contact.frame
            ).reshape(3, 3)

            force_world = R.T @ result_force[0:3]

            total_force += force_world

            contact_point = np.array(
                contact.pos
            )

    return total_force, contact_point


# ============================================================
# WORLD FORCE -> ROBOT FRAME
# ============================================================

def world_force_to_robot_frame(force_world):

    Q_OFFSET_INV = np.array([
        Q_OFFSET[0],
        -Q_OFFSET[1],
        -Q_OFFSET[2],
        -Q_OFFSET[3]
    ])

    force_robot_frame = np.zeros(3)

    mujoco.mju_rotVecQuat(
        force_robot_frame,
        force_world,
        Q_OFFSET_INV
    )

    return force_robot_frame


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load MuJoCo model
    # --------------------------------------------------------

    MJCF_PATH = (
        "/home/justinrc/workspace/src/"
        "dual_servo_ur3e_sim/mjcf/goodbye.xml"
    )

    model = mujoco.MjModel.from_xml_path(
        MJCF_PATH
    )

    data = mujoco.MjData(model)

    # --------------------------------------------------------
    # Find box
    # --------------------------------------------------------

    box_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        BOX_BODY_NAME
    )

    # --------------------------------------------------------
    # Find tool bodies
    # --------------------------------------------------------

    tool_body_ids = []

    for name in TOOL_BODY_NAMES:

        tool_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            name
        )

        tool_body_ids.append(tool_id)

    print("MuJoCo bodies:")
    print("Box:", box_body_id)
    print("Tools:", tool_body_ids)

    # --------------------------------------------------------
    # Connect to both robots
    # --------------------------------------------------------

    rtde_controls = []
    rtde_receives = []

    for ip in ROBOT_IPS:

        print(
            f"Connecting to robot at {ip}..."
        )

        rtde_c = (
            rtde_control.RTDEControlInterface(ip)
        )

        rtde_r = (
            rtde_receive.RTDEReceiveInterface(ip)
        )

        rtde_controls.append(rtde_c)
        rtde_receives.append(rtde_r)

        print(
            f"Connected to {ip}"
        )

        rtde_c.zeroFtSensor()

        rtde_c.forceModeSetDamping(
            DAMPING
        )

        rtde_c.forceModeSetGainScaling(
            GAIN_SCALING
        )

    # --------------------------------------------------------
    # Signal handling
    # --------------------------------------------------------

    running = True

    def handle_sigint(signum, frame):

        nonlocal running

        print(
            "\nCtrl+C received, stopping..."
        )

        running = False

    signal.signal(
        signal.SIGINT,
        handle_sigint
    )

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------

    step_count = 0

    try:

        with mujoco.viewer.launch_passive(
            model,
            data
        ) as viewer:

            print(
                "Entering dual robot "
                "force-mode + MuJoCo loop..."
            )

            while (
                running
                and viewer.is_running()
            ):

                loop_start = time.time()

                arrows = []

                # ====================================================
                # UPDATE BOTH ROBOTS
                # ====================================================

                for robot_id in range(2):

                    rtde_c = (
                        rtde_controls[robot_id]
                    )

                    rtde_r = (
                        rtde_receives[robot_id]
                    )

                    # ------------------------------------------------
                    # Apply current force command
                    # ------------------------------------------------

                    rtde_c.forceMode(
                        TASK_FRAME,
                        SELECTION_VECTOR,
                        WRENCHES[robot_id],
                        FORCE_TYPE,
                        LIMITS[robot_id]
                    )

                    # ------------------------------------------------
                    # Read physical robot TCP pose
                    # ------------------------------------------------

                    current_pose = (
                        rtde_r.getActualTCPPose()
                    )

                    # ------------------------------------------------
                    # Convert to MuJoCo coordinates
                    # ------------------------------------------------

                    (
                        pos_transformed,
                        quat_final
                    ) = transform_robot_pose_to_mujoco(
                        current_pose,
                        robot_id
                    )

                    # ------------------------------------------------
                    # Move corresponding mocap
                    # ------------------------------------------------

                    mocap_id = (
                        MOCAP_IDS[robot_id]
                    )

                    data.mocap_pos[mocap_id] = (
                        pos_transformed
                    )

                    data.mocap_quat[mocap_id] = (
                        quat_final
                    )

                    # ------------------------------------------------
                    # Get contact force
                    # ------------------------------------------------

                    (
                        total_force,
                        contact_point
                    ) = get_contact_force(
                        model,
                        data,
                        box_body_id,
                        tool_body_ids[robot_id]
                    )

                    # ------------------------------------------------
                    # Convert force to robot frame
                    # ------------------------------------------------

                    force_robot_frame = (
                        world_force_to_robot_frame(
                            total_force
                        )
                    )

                    # ------------------------------------------------
                    # Scale force sent to robot
                    # ------------------------------------------------

                    WRENCHES[robot_id][0:3] = (
                        force_robot_frame
                        * FORCE_SCALE
                    ).tolist()

                    # ------------------------------------------------
                    # Store force arrow
                    # ------------------------------------------------

                    arrows.append(
                        (
                            contact_point,
                            total_force
                        )
                    )

                # ====================================================
                # MUJOCO STEP / RENDER
                # ====================================================

                step_count += 1

                if (
                    step_count
                    % RENDER_EVERY_N_STEPS
                    == 0
                ):

                    mujoco.mj_step(
                        model,
                        data
                    )

                    with viewer.lock():

                        draw_force_arrows(
                            viewer,
                            arrows
                        )

                    viewer.sync()

                # ====================================================
                # MAINTAIN 2 ms LOOP
                # ====================================================

                elapsed = (
                    time.time()
                    - loop_start
                )

                sleep_time = (
                    CONTROL_PERIOD_S
                    - elapsed
                )

                if sleep_time > 0:

                    time.sleep(
                        sleep_time
                    )

    # ============================================================
    # CLEANUP
    # ============================================================

    finally:

        print(
            "Stopping force mode "
            "and disconnecting..."
        )

        for rtde_c in rtde_controls:

            try:

                rtde_c.forceModeStop()

                rtde_c.stopScript()

            except Exception as e:

                print(
                    f"Error stopping robot: {e}"
                )

        print("Done.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()