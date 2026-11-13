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

from scipy.spatial.transform import Rotation as R

from IK_solver import PoseIK, JOINT_SUFFIXES, mat_to_quat

from mujoco_unity_bridge import (
    STATE_PORT,
    SCHEMA,
    body_poses,
    r6,
)


# =====================================================================
# Robot configuration
# =====================================================================

LEFT_IP = "192.168.0.1"
RIGHT_IP = "192.168.0.3"

MJCF_PATH = str(
    Path(__file__).resolve().parent.parent
    / "mjcf"
    / "teleop_dual_IK.xml"
)

MOCAP_BODIES = {
    "left": "left_arm_mocap",
    "right": "right_arm_mocap",
}

PORT_BODIES = {
    "left": "left_port",
    "right": "right_port",
}


# =====================================================================
# Unity bridge configuration
# =====================================================================

# Windows Ethernet address reachable from WSL.
UNITY_HOST = "192.168.0.101"

STATE_SEND_HZ = 60.0

# Each arm receives its visual body poses at half the packet rate:
#
# packet 1 -> left bodies
# packet 2 -> right bodies
# packet 3 -> left bodies
# ...
#
# Therefore:
# tool state = 60 Hz
# left visual bodies = 30 Hz
# right visual bodies = 30 Hz

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

TASK_FRAME = [
    0,
    0,
    0,
    0,
    0,
    0,
]

SELECTION_VECTOR = [
    1,
    1,
    1,
    1,
    1,
    1,
]

WRENCH = [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]

FORCE_TYPE = 2

LIMITS = [
    0.1,
    0.1,
    1.5,
    3.14,
    3.14,
    0.5,
]

DAMPING = 0.004
GAIN_SCALING = 1.0

CONTROL_PERIOD_S = 0.008
RENDER_EVERY_N_LOOPS = 10


# =====================================================================
# Coordinate transforms
# =====================================================================

# Fixed 180-degree rotation around Y.
# Quaternion order used by MuJoCo: w, x, y, z.
Q_OFFSET = np.array([
    0.0,
    0.0,
    1.0,
    0.0,
])


FRAME_OFFSETS = {
    "left": np.array([
        -0.3,
        0.0,
        0.8,
    ]),
    "right": np.array([
        0.3,
        0.0,
        0.8,
    ]),
}


R_WORLD_H = {
    "left": np.diag([
        -1.0,
        1.0,
        -1.0,
    ]),
    "right": np.diag([
        -1.0,
        1.0,
        -1.0,
    ]),
}


SCALE = 0.5


# =====================================================================
# MuJoCo helpers
# =====================================================================

def get_mocap_id(model, body_name):
    body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_name,
    )

    if body_id < 0:
        raise ValueError(
            f"Body not found: {body_name}"
        )

    mocap_id = model.body_mocapid[
        body_id
    ]

    if mocap_id < 0:
        raise ValueError(
            f"Body is not a mocap body: {body_name}"
        )

    return mocap_id


def transform_pose(
    pose,
    frame_offset,
):
    """
    Convert UR TCP pose:

        [x, y, z, rx, ry, rz]

    into the MuJoCo world frame.
    """

    pos = np.zeros(3)

    mujoco.mju_rotVecQuat(
        pos,
        np.array(
            pose[:3]
        ),
        Q_OFFSET,
    )

    pos += frame_offset

    rot_vec = np.array(
        pose[3:6]
    )

    angle = np.linalg.norm(
        rot_vec
    )

    quat_robot = np.array([
        1.0,
        0.0,
        0.0,
        0.0,
    ])

    if angle > 1e-9:
        mujoco.mju_axisAngle2Quat(
            quat_robot,
            rot_vec / angle,
            angle,
        )

    quat = np.zeros(4)

    mujoco.mju_mulQuat(
        quat,
        Q_OFFSET,
        quat_robot,
    )

    return pos, quat


def vector_to_htm(
    tcp_vector,
):
    mat = np.eye(4)

    mat[:3, 3] = (
        tcp_vector[:3]
    )

    mat[:3, :3] = (
        R.from_rotvec(
            tcp_vector[3:]
        ).as_matrix()
    )

    return mat


def transform_world_point_to_site_frame(
    model,
    data,
    site_name: str,
    p_world: np.ndarray,
) -> np.ndarray:

    mujoco.mj_forward(
        model,
        data,
    )

    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        site_name,
    )

    if site_id == -1:
        raise ValueError(
            f"Site '{site_name}' not found."
        )

    p_rb_world = (
        data.site_xpos[
            site_id
        ]
    )

    R_world_rb = (
        data.site_xmat[
            site_id
        ]
        .reshape(
            3,
            3,
        )
    )

    R_transpose = (
        R_world_rb.T
    )

    T_rb_world = np.eye(4)

    T_rb_world[
        :3,
        :3,
    ] = R_transpose

    T_rb_world[
        :3,
        3,
    ] = (
        -R_transpose
        @ p_rb_world
    )

    p_world_homog = np.array([
        p_world[0],
        p_world[1],
        p_world[2],
        1.0,
    ])

    p_rb_homog = (
        T_rb_world
        @ p_world_homog
    )

    return p_rb_homog[:3]


def site_pose_rb(
    ik,
    data,
):
    """
    End-effector pose expressed in the fixed RB / port frame.
    """

    p_world = (
        data.site_xpos[
            ik.site_id
        ]
    )

    R_world = (
        data.site_xmat[
            ik.site_id
        ]
        .reshape(
            3,
            3,
        )
    )

    return (
        ik.R_rb.T
        @ (
            p_world
            - ik.p_rb
        ),
        ik.R_rb.T
        @ R_world,
    )


def get_actuator_ids_for_joints(
    model,
    side,
):
    ids = []

    for suffix in JOINT_SUFFIXES:

        name = (
            f"{side}_{suffix}"
        )

        jid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        found = [
            actuator_id

            for actuator_id
            in range(
                model.nu
            )

            if int(
                model.actuator_trntype[
                    actuator_id
                ]
            )
            == int(
                mujoco.mjtTrn.mjTRN_JOINT
            )

            and int(
                model.actuator_trnid[
                    actuator_id,
                    0,
                ]
            )
            == jid
        ]

        if not found:
            raise ValueError(
                f"No actuator drives joint '{name}'."
            )

        ids.append(
            found[0]
        )

    return ids


def disable_mocap_welds(
    model,
    data,
    mocap_body_ids,
):
    if not hasattr(
        data,
        "eq_active",
    ):
        print(
            "WARNING: this MuJoCo version has no "
            "data.eq_active. Remove the mocap welds "
            "from the XML manually or they will "
            "fight the IK."
        )

        return

    disabled = 0

    for i in range(
        model.neq
    ):

        if (
            int(
                model.eq_type[i]
            )
            != int(
                mujoco.mjtEq.mjEQ_WELD
            )
        ):
            continue

        if (
            int(
                model.eq_obj1id[i]
            )
            in mocap_body_ids
            or int(
                model.eq_obj2id[i]
            )
            in mocap_body_ids
        ):
            data.eq_active[i] = 0

            disabled += 1

    print(
        f"Disabled {disabled} mocap weld(s)."
    )


# =====================================================================
# Unity helpers
# =====================================================================

def get_joint_qpos(
    model,
    data,
    joint_name,
    default=0.0,
):
    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name,
    )

    if jid < 0:
        return float(
            default
        )

    adr = int(
        model.jnt_qposadr[
            jid
        ]
    )

    return float(
        data.qpos[
            adr
        ]
    )


def get_joint_range(
    model,
    joint_name,
):
    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name,
    )

    if jid < 0:
        return None

    return (
        float(
            model.jnt_range[
                jid,
                0,
            ]
        ),
        float(
            model.jnt_range[
                jid,
                1,
            ]
        ),
    )


def build_unity_tool_state(
    model,
    data,
    side,
):
    prefix = (
        f"{side}_"
    )

    tool_q = [
        get_joint_qpos(
            model,
            data,
            prefix + joint_name,
        )

        for joint_name
        in UNITY_TOOL_JOINTS
    ]

    port_q = [
        get_joint_qpos(
            model,
            data,
            prefix + joint_name,
        )

        for joint_name
        in UNITY_PORT_JOINTS
    ]

    pitch_name = (
        prefix
        + "joint_pitch"
    )

    pitch = get_joint_qpos(
        model,
        data,
        pitch_name,
    )

    pitch_range = (
        get_joint_range(
            model,
            pitch_name,
        )
    )

    pitch_margin = 1.0

    if pitch_range is not None:

        limit = max(
            abs(
                pitch_range[0]
            ),
            abs(
                pitch_range[1]
            ),
        )

        if limit > 1e-9:
            pitch_margin = (
                1.0
                - min(
                    1.0,
                    abs(
                        pitch
                    )
                    / limit,
                )
            )

    return {
        "mode": "ik",

        "master_ok": True,

        "clutch": False,

        "master": None,

        "tool_q": r6(
            tool_q
        ),

        "pitch_margin": float(
            pitch_margin
        ),

        "jaw_contact_force": 0.0,

        "port_q": r6(
            port_q
        ),
    }


def get_unity_tool_bodies(
    model,
    side,
):
    """
    Return only the six MuJoCo bodies that MujocoBridge.cs
    actually uses for one dVRK instrument.

    This is intentionally filtered to keep UDP datagrams
    below the fragmentation threshold observed between
    WSL and Windows.
    """

    prefix = (
        f"{side}_"
    )

    selected = []

    for body_id in range(
        1,
        model.nbody,
    ):

        name = (
            model.body(
                body_id
            ).name
        )

        if not name.startswith(
            prefix
        ):
            continue

        own_name = (
            name[
                len(prefix):
            ]
        )

        short_name = (
            own_name
            .split(
                "_Shape"
            )[0]
        )

        if (
            short_name
            in UNITY_BODY_SUFFIXES
        ):
            selected.append(
                name
            )

    if len(
        selected
    ) != 6:
        print(
            f"WARNING: expected 6 Unity bodies "
            f"for {side}, found {len(selected)}"
        )

        for name in selected:
            print(
                f"  {name}"
            )

    return selected


# =====================================================================
# Main
# =====================================================================

def main():

    # -----------------------------------------------------------------
    # Load MuJoCo model
    # -----------------------------------------------------------------

    model = mujoco.MjModel.from_xml_path(
        MJCF_PATH
    )

    data = mujoco.MjData(
        model
    )


    # -----------------------------------------------------------------
    # Unity UDP
    # -----------------------------------------------------------------

    unity_tx = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM,
    )

    unity_period = (
        1.0
        / STATE_SEND_HZ
    )

    next_unity_send = (
        time.perf_counter()
    )

    unity_seq = 0


    left_unity_bodies = (
        get_unity_tool_bodies(
            model,
            "left",
        )
    )

    right_unity_bodies = (
        get_unity_tool_bodies(
            model,
            "right",
        )
    )


    print(
        "\nUnity bodies:"
    )

    print(
        "LEFT:"
    )

    for name in left_unity_bodies:
        print(
            f"  {name}"
        )

    print(
        "RIGHT:"
    )

    for name in right_unity_bodies:
        print(
            f"  {name}"
        )


    print(
        f"\n[Unity bridge] "
        f"state -> "
        f"{UNITY_HOST}:{STATE_PORT} "
        f"@ {STATE_SEND_HZ:.0f} Hz"
    )

    print(
        "[Unity bridge] alternating "
        "left/right visual bodies "
        "to avoid UDP fragmentation."
    )


    # -----------------------------------------------------------------
    # Teleoperation initialization
    # -----------------------------------------------------------------

    mocap_ids = {
        side:
        get_mocap_id(
            model,
            body_name,
        )

        for side, body_name
        in MOCAP_BODIES.items()
    }


    ips = {
        "left": LEFT_IP,
        "right": RIGHT_IP,
    }


    steps_per_loop = max(
        1,
        round(
            CONTROL_PERIOD_S
            / model.opt.timestep
        ),
    )


    iks = {
        side:
        PoseIK(
            model,
            side,
            port_body=(
                PORT_BODIES[
                    side
                ]
            ),
        )

        for side
        in ips
    }


    act_ids = {
        side:
        get_actuator_ids_for_joints(
            model,
            side,
        )

        for side
        in ips
    }


    mocap_body_ids = [
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_name,
        )

        for body_name
        in MOCAP_BODIES.values()
    ]


    controllers = {}
    receivers = {}

    running = True


    # -----------------------------------------------------------------
    # Ctrl+C safety stop
    # -----------------------------------------------------------------

    def handle_sigint(
        signum,
        frame,
    ):
        nonlocal running

        print(
            "\nCtrl+C received, stopping..."
        )

        running = False


    signal.signal(
        signal.SIGINT,
        handle_sigint,
    )


    try:

        # -------------------------------------------------------------
        # Connect robots
        # -------------------------------------------------------------

        for side, ip in ips.items():

            print(
                f"Connecting to "
                f"{side} robot "
                f"at {ip}..."
            )

            controllers[
                side
            ] = (
                rtde_control
                .RTDEControlInterface(
                    ip
                )
            )

            receivers[
                side
            ] = (
                rtde_receive
                .RTDEReceiveInterface(
                    ip
                )
            )


        time.sleep(
            1.0
        )


        for controller in (
            controllers.values()
        ):

            controller.zeroFtSensor()

            controller.forceModeSetDamping(
                DAMPING
            )

            controller.forceModeSetGainScaling(
                GAIN_SCALING
            )


        print(
            "Connected. Both arms "
            "become compliant now."
        )


        # -------------------------------------------------------------
        # MuJoCo simulation
        # -------------------------------------------------------------

        with mujoco.viewer.launch_passive(
            model,
            data,
        ) as viewer:

            print(
                "Dual-arm teleoperation running."
            )


            W_start = {
                side:
                receiver.getActualTCPPose()

                for side, receiver
                in receivers.items()
            }


            if model.nkey > 0:

                mujoco.mj_resetDataKeyframe(
                    model,
                    data,
                    0,
                )


            disable_mocap_welds(
                model,
                data,
                mocap_body_ids,
            )


            mujoco.mj_forward(
                model,
                data,
            )


            P_start = {
                side:
                site_pose_rb(
                    iks[
                        side
                    ],
                    data,
                )

                for side
                in ips
            }


            mujoco.mj_forward(
                model,
                data,
            )


            # ---------------------------------------------------------
            # Validate hand -> robot-base transforms
            # ---------------------------------------------------------

            for side in ips:

                Rw = (
                    R_WORLD_H[
                        side
                    ]
                )

                valid_rotation = (
                    np.allclose(
                        Rw.T @ Rw,
                        np.eye(3),
                    )
                    and
                    np.isclose(
                        np.linalg.det(
                            Rw
                        ),
                        1.0,
                    )
                )

                if not valid_rotation:
                    raise ValueError(
                        f"R_WORLD_H['{side}'] "
                        "must be a proper rotation "
                        "(orthonormal, det +1)."
                    )


            R_RB_H = {
                side:
                (
                    iks[
                        side
                    ].R_rb.T
                    @ R_WORLD_H[
                        side
                    ]
                )

                for side
                in ips
            }


            print(
                "Computed R_RB_H successfully:",
                R_RB_H,
            )


            hand_start = {}

            for side in ips:

                H0 = vector_to_htm(
                    W_start[
                        side
                    ]
                )

                hand_start[
                    side
                ] = (
                    H0[
                        :3,
                        3,
                    ].copy(),

                    H0[
                        :3,
                        :3,
                    ].copy(),
                )


            tool_start = (
                P_start
            )


            q_cmd = {
                side:
                data.qpos[
                    iks[
                        side
                    ].qpos_ids
                ].copy()

                for side
                in ips
            }


            for side in ips:

                iks[
                    side
                ].q_last[:] = (
                    q_cmd[
                        side
                    ]
                )


            ik_ok = {
                side: True
                for side
                in ips
            }


            ik_ms_sum = 0.0
            ik_ms_max = 0.0
            ik_calls = 0

            loop = 0


            # ---------------------------------------------------------
            # Main teleoperation loop
            # ---------------------------------------------------------

            while (
                running
                and viewer.is_running()
            ):

                loop_start = (
                    time.perf_counter()
                )


                # -----------------------------------------------------
                # Keep both physical arms compliant
                # -----------------------------------------------------

                for controller in (
                    controllers.values()
                ):

                    controller.forceMode(
                        TASK_FRAME,
                        SELECTION_VECTOR,
                        WRENCH,
                        FORCE_TYPE,
                        LIMITS,
                    )


                poses = {
                    side:
                    receiver.getActualTCPPose()

                    for side, receiver
                    in receivers.items()
                }


                # -----------------------------------------------------
                # IK
                # -----------------------------------------------------

                ik_t0 = (
                    time.perf_counter()
                )


                for side, pose in (
                    poses.items()
                ):

                    H1 = vector_to_htm(
                        pose
                    )

                    p_h1 = (
                        H1[
                            :3,
                            3,
                        ]
                    )

                    R_h1 = (
                        H1[
                            :3,
                            :3,
                        ]
                    )


                    p_h0, R_h0 = (
                        hand_start[
                            side
                        ]
                    )

                    p_rb0, R_rb0 = (
                        tool_start[
                            side
                        ]
                    )

                    M = (
                        R_RB_H[
                            side
                        ]
                    )


                    p_target = (
                        p_rb0
                        + SCALE
                        * (
                            M
                            @ (
                                p_h1
                                - p_h0
                            )
                        )
                    )


                    R_target = (
                        (
                            M
                            @ (
                                R_h1
                                @ R_h0.T
                            )
                            @ M.T
                        )
                        @ R_rb0
                    )


                    result = (
                        iks[
                            side
                        ].solve(
                            p_target,
                            mat_to_quat(
                                R_target
                            ),
                            frame="rb",
                        )
                    )


                    if result.converged:

                        q_cmd[
                            side
                        ] = (
                            result.q
                        )


                    if (
                        result.converged
                        != ik_ok[
                            side
                        ]
                    ):

                        ik_ok[
                            side
                        ] = (
                            result.converged
                        )


                        if result.converged:

                            status = (
                                "recovered"
                            )

                        else:

                            status = (
                                "FAILED "
                                "(target unreachable? "
                                f"pos_err="
                                f"{result.pos_err * 1000:.1f} mm, "
                                f"rot_err="
                                f"{result.rot_err * 1000:.0f} mrad), "
                                "holding last command"
                            )


                        print(
                            f"[{side}] "
                            f"IK {status}"
                        )


                ik_ms = (
                    (
                        time.perf_counter()
                        - ik_t0
                    )
                    * 1000.0
                )


                ik_ms_sum += (
                    ik_ms
                )

                ik_ms_max = max(
                    ik_ms_max,
                    ik_ms,
                )

                ik_calls += 1


                if (
                    ik_calls
                    % 500
                    == 0
                ):

                    print(
                        "IK time for both arms: "
                        f"avg "
                        f"{ik_ms_sum / 500:.2f} ms, "
                        f"max "
                        f"{ik_ms_max:.2f} ms"
                    )

                    ik_ms_sum = 0.0
                    ik_ms_max = 0.0


                # -----------------------------------------------------
                # Update the same MuJoCo simulation used by Unity
                # -----------------------------------------------------

                with viewer.lock():

                    for side, pose in (
                        poses.items()
                    ):

                        pos, quat = (
                            transform_pose(
                                pose,
                                FRAME_OFFSETS[
                                    side
                                ],
                            )
                        )

                        data.mocap_pos[
                            mocap_ids[
                                side
                            ]
                        ] = pos

                        data.mocap_quat[
                            mocap_ids[
                                side
                            ]
                        ] = quat


                    for side in ips:

                        data.ctrl[
                            act_ids[
                                side
                            ]
                        ] = (
                            q_cmd[
                                side
                            ]
                        )


                    for _ in range(
                        steps_per_loop
                    ):

                        mujoco.mj_step(
                            model,
                            data,
                        )


                loop += 1


                if (
                    loop
                    % RENDER_EVERY_N_LOOPS
                    == 0
                ):

                    viewer.sync()


                # -----------------------------------------------------
                # Unity UDP publication
                # -----------------------------------------------------

                now = (
                    time.perf_counter()
                )


                if (
                    now
                    >= next_unity_send
                ):

                    tools = {
                        "left_":
                        build_unity_tool_state(
                            model,
                            data,
                            "left",
                        ),

                        "right_":
                        build_unity_tool_state(
                            model,
                            data,
                            "right",
                        ),
                    }


                    unity_seq += 1


                    # Alternate the six visual bodies.
                    #
                    # Odd packet:
                    #     left bodies
                    #
                    # Even packet:
                    #     right bodies
                    #
                    # This keeps each UDP datagram below
                    # the ~1450-byte limit observed between
                    # WSL and Windows.

                    if (
                        unity_seq
                        % 2
                        == 1
                    ):

                        packet_side = (
                            "left"
                        )

                        packet_bodies = (
                            left_unity_bodies
                        )

                    else:

                        packet_side = (
                            "right"
                        )

                        packet_bodies = (
                            right_unity_bodies
                        )


                    state = {
                        "v": SCHEMA,

                        "seq":
                        unity_seq,

                        "t_sim":
                        round(
                            float(
                                data.time
                            ),
                            6,
                        ),

                        "ncon":
                        int(
                            data.ncon
                        ),

                        "tools":
                        tools,

                        "bodies":
                        body_poses(
                            data,
                            packet_bodies,
                        ),

                        "contacts":
                        [],
                    }


                    # Preserve legacy v1 fields.
                    state.update(
                        tools[
                            "left_"
                        ]
                    )


                    state[
                        "t_send"
                    ] = now


                    payload = (
                        json.dumps(
                            state,
                            separators=(
                                ",",
                                ":",
                            ),
                        )
                        .encode()
                    )


                    # Debug only on startup.
                    if (
                        unity_seq
                        <= 2
                    ):

                        print(
                            f"[Unity bridge] "
                            f"seq={unity_seq} "
                            f"side={packet_side} "
                            f"bytes={len(payload)}"
                        )


                    # Safety warning:
                    # do not silently return to fragmented UDP.
                    if (
                        len(payload)
                        > 1450
                    ):

                        print(
                            f"WARNING: Unity packet "
                            f"{unity_seq} is "
                            f"{len(payload)} bytes "
                            f"(>1450)."
                        )


                    unity_tx.sendto(
                        payload,
                        (
                            UNITY_HOST,
                            STATE_PORT,
                        ),
                    )


                    next_unity_send += (
                        unity_period
                    )


                    # If execution stalls, resynchronize
                    # instead of trying to catch up with
                    # a burst of UDP packets.
                    if (
                        next_unity_send
                        < (
                            now
                            - unity_period
                        )
                    ):

                        next_unity_send = (
                            now
                            + unity_period
                        )


                # -----------------------------------------------------
                # Maintain approximately 500-Hz teleoperation loop
                # -----------------------------------------------------

                remaining = (
                    CONTROL_PERIOD_S
                    - (
                        time.perf_counter()
                        - loop_start
                    )
                )


                if (
                    remaining
                    > 0
                ):

                    time.sleep(
                        remaining
                    )


    finally:

        print(
            "Stopping force mode "
            "and disconnecting..."
        )


        try:
            unity_tx.close()

        except Exception:
            pass


        for side, controller in (
            controllers.items()
        ):

            try:

                controller.forceModeStop()

                controller.stopScript()

            except Exception as exc:

                print(
                    f"[{side}] error "
                    f"stopping force mode: "
                    f"{exc}"
                )


        for side, receiver in (
            receivers.items()
        ):

            try:

                receiver.disconnect()

            except Exception as exc:

                print(
                    f"[{side}] error "
                    f"disconnecting: "
                    f"{exc}"
                )


        print(
            "Done."
        )


if __name__ == "__main__":
    main()