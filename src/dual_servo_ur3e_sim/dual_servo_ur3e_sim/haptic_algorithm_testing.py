import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import rtde_control
import rtde_receive

ROBOT_IPS = {
    "ur3": "192.168.0.1",
    "ur3e": "192.168.0.3",
}
MJCF_PATH = Path(__file__).resolve().parent.parent / "mjcf" / "sphere_test.xml"

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
    -Q_OFFSET[3],
])

FRAME_OFFSET_POS = np.array([0.0, 0.0, 0.8])

ARROW_LENGTH_SCALE = 0.01
ARROW_WIDTH = 0.004

# Matches sphere_haptic_test.xml
SPHERE_RADIUS = 0.03
BOX_CENTER = np.array([0.0, -0.3, 0.2])
BOX_HALF_SIZE = np.array([0.2, 0.2, 0.2])

# God-object spring-damper gains (tune higher for firmer wall feel)
STIFFNESS_LIN = 4000.0
DAMPING_LIN = 20.0
FORCE_SCALE = 0.6
MAX_FORCE = 100.0
INTERNAL_MAX_FORCE = 100.0


def rotvec_to_quat(rotvec):
    """Converts a rotation vector (Axis-Angle) from UR RTDE to a MuJoCo quaternion [w, x, y, z]."""
    angle = np.linalg.norm(rotvec)
    quat = np.zeros(4)
    if angle < 1e-6:
        quat[0] = 1.0
    else:
        axis = rotvec / angle
        half_angle = angle / 2.0
        quat[0] = np.cos(half_angle)
        quat[1:4] = axis * np.sin(half_angle)
    return quat


def project_god_object(p_device, p_proxy_prev, box_center, box_half_size, sphere_radius):
    """Projects a god-object proxy onto/outside the floor and axis-aligned box.

    Returns (p_proxy, in_contact, normal_out).
    """
    in_contact = False
    normal_sum = np.zeros(3)

    # 1. Floor plane at z = 0 -> sphere center min z = sphere_radius
    p_proxy = p_device.copy()
    floor_min_z = sphere_radius
    if p_proxy[2] < floor_min_z:
        p_proxy[2] = floor_min_z
        in_contact = True
        normal_sum += np.array([0.0, 0.0, 1.0])

    # 2. Axis-aligned box expanded by sphere_radius (Minkowski sum for sphere vs AABB)
    ext = box_half_size + sphere_radius
    b_min = box_center - ext
    b_max = box_center + ext

    inside_x = b_min[0] < p_proxy[0] < b_max[0]
    inside_y = b_min[1] < p_proxy[1] < b_max[1]
    inside_z = b_min[2] < p_proxy[2] < b_max[2]

    if inside_x and inside_y and inside_z:
        in_contact = True
        # Find nearest face of expanded AABB to project onto
        pen_faces = [
            (abs(p_proxy[0] - b_min[0]), np.array([b_min[0], p_proxy[1], p_proxy[2]]), np.array([-1.0, 0.0, 0.0])),
            (abs(b_max[0] - p_proxy[0]), np.array([b_max[0], p_proxy[1], p_proxy[2]]), np.array([1.0, 0.0, 0.0])),
            (abs(p_proxy[1] - b_min[1]), np.array([p_proxy[0], b_min[1], p_proxy[2]]), np.array([0.0, -1.0, 0.0])),
            (abs(b_max[1] - p_proxy[1]), np.array([p_proxy[0], b_max[1], p_proxy[2]]), np.array([0.0, 1.0, 0.0])),
            (abs(p_proxy[2] - b_min[2]), np.array([p_proxy[0], p_proxy[1], b_min[2]]), np.array([0.0, 0.0, -1.0])),
            (abs(b_max[2] - p_proxy[2]), np.array([p_proxy[0], p_proxy[1], b_max[2]]), np.array([0.0, 0.0, 1.0])),
        ]
        _, p_proj, n_face = min(pen_faces, key=lambda item: item[0])

        # Respect floor constraint on projection as well
        if p_proj[2] < floor_min_z:
            p_proj[2] = floor_min_z
            normal_sum += np.array([0.0, 0.0, 1.0])

        p_proxy = p_proj
        normal_sum += n_face

    norm_len = np.linalg.norm(normal_sum)
    if norm_len > 1e-9:
        normal_out = normal_sum / norm_len
    else:
        normal_out = np.zeros(3)

    return p_proxy, in_contact, normal_out


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
    parser = argparse.ArgumentParser(
        description="Run true surface-projected god-object proxy with MuJoCo and UR RTDE."
    )
    parser.add_argument(
        "--robot",
        choices=ROBOT_IPS,
        default="ur3e",
        help="robot to connect to (default: ur3e)",
    )
    args = parser.parse_args()
    robot_ip = ROBOT_IPS[args.robot]

    model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    data = mujoco.MjData(model)

    sphere_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "sphere")
    if sphere_id < 0:
        raise RuntimeError("Could not find body named 'sphere' in the MJCF.")

    print(f"Connecting to {args.robot.upper()} at {robot_ip}...")
    rtde_c = rtde_control.RTDEControlInterface(robot_ip)
    rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)

    time.sleep(1.0)

    rtde_c.zeroFtSensor()
    rtde_c.forceModeSetDamping(DAMPING)
    rtde_c.forceModeSetGainScaling(GAIN_SCALING)

    running = True
    startup_complete = False
    step_count = 0

    p_proxy = None
    prev_mocap_pos = None
    prev_time = None
    program_start_time = time.time()

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while running and viewer.is_running():
                loop_start = time.time()

                current_pose = rtde_r.getActualTCPPose()

                # --- 1. Device (unconstrained hand target) Position ---
                pos_transformed = np.zeros(3)
                mujoco.mju_rotVecQuat(pos_transformed, np.array(current_pose[0:3]), Q_OFFSET)
                pos_transformed += FRAME_OFFSET_POS
                data.mocap_pos[0] = pos_transformed

                # --- 2. Rotation Transformation ---
                quat_tcp = rotvec_to_quat(np.array(current_pose[3:6]))
                quat_transformed = np.zeros(4)
                mujoco.mju_mulQuat(quat_transformed, Q_OFFSET, quat_tcp)
                data.mocap_quat[0] = quat_transformed

                elapsed_since_start = time.time() - program_start_time

                if not startup_complete:
                    WRENCH[0:3] = [0.0, 0.0, 0.0]
                    data.xfrc_applied[sphere_id, :] = 0.0

                    if elapsed_since_start >= STARTUP_DELAY_S:
                        p_proxy, _, _ = project_god_object(
                            p_device=data.mocap_pos[0].copy(),
                            p_proxy_prev=data.mocap_pos[0].copy(),
                            box_center=BOX_CENTER,
                            box_half_size=BOX_HALF_SIZE,
                            sphere_radius=SPHERE_RADIUS,
                        )
                        data.qpos[0:3] = p_proxy
                        data.qpos[3:7] = data.mocap_quat[0]
                        data.qvel[0:3] = 0.0
                        data.qvel[3:6] = 0.0
                        mujoco.mj_forward(model, data)

                        prev_mocap_pos = data.mocap_pos[0].copy()
                        prev_time = time.time()
                        startup_complete = True

                        print("\n==========================================")
                        print("TRUE GOD-OBJECT PROXY STARTED")
                        print("==========================================")
                        print(f"Stiffness:    {STIFFNESS_LIN:.2f} N/m")
                        print(f"Damping:      {DAMPING_LIN:.2f} N/(m/s)")
                        print(f"Box center:   {BOX_CENTER}")
                        print(f"Box half-sz:  {BOX_HALF_SIZE}")
                        print("==========================================\n")

                    elapsed = time.time() - loop_start
                    sleep_time = CONTROL_PERIOD_S - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    continue

                rtde_c.forceMode(
                    TASK_FRAME,
                    SELECTION_VECTOR,
                    WRENCH,
                    FORCE_TYPE,
                    LIMITS,
                )

                now = time.time()
                dt = max(now - prev_time, 1e-4)
                prev_time = now

                p_device = data.mocap_pos[0].copy()
                mocap_vel = (p_device - prev_mocap_pos) / dt
                prev_mocap_pos = p_device.copy()

                p_proxy_next, in_contact, n_out = project_god_object(
                    p_device=p_device,
                    p_proxy_prev=p_proxy,
                    box_center=BOX_CENTER,
                    box_half_size=BOX_HALF_SIZE,
                    sphere_radius=SPHERE_RADIUS,
                )
                proxy_vel = (p_proxy_next - p_proxy) / dt
                p_proxy = p_proxy_next

                # Penetration is how far the unconstrained device sits on the interior side of p_proxy
                penetration_vec = p_proxy - p_device
                pen_depth = float(np.linalg.norm(penetration_vec))

                if in_contact and pen_depth > 1e-6:
                    # Push device back toward p_proxy (out of the obstacle) with velocity damping
                    force_on_device_world = (
                        STIFFNESS_LIN * penetration_vec - DAMPING_LIN * mocap_vel
                    )
                    # Prevent negative/pulling force into the wall
                    if np.linalg.norm(n_out) > 1e-6 and np.dot(force_on_device_world, n_out) < 0.0:
                        force_on_device_world -= np.dot(force_on_device_world, n_out) * n_out
                else:
                    force_on_device_world = np.zeros(3)

                force_on_device_world = np.clip(
                    force_on_device_world,
                    -INTERNAL_MAX_FORCE,
                    INTERNAL_MAX_FORCE,
                )

                # Constrain visual god-object proxy directly in MuJoCo without dynamic penetration
                data.qpos[0:3] = p_proxy
                data.qpos[3:7] = data.mocap_quat[0]
                data.qvel[0:3] = proxy_vel
                data.qvel[3:6] = 0.0
                data.xfrc_applied[sphere_id, :] = 0.0
                mujoco.mj_forward(model, data)

                force_robot_frame = np.zeros(3)
                mujoco.mju_rotVecQuat(
                    force_robot_frame,
                    force_on_device_world,
                    Q_OFFSET_INV,
                )

                force_robot_scaled = force_robot_frame * FORCE_SCALE
                force_robot_limited = np.clip(
                    force_robot_scaled,
                    -MAX_FORCE,
                    MAX_FORCE,
                )

                WRENCH[0:3] = force_robot_limited.tolist()

                step_count += 1

                if step_count % PRINT_EVERY_N_STEPS == 0:
                    print(
                        f"Step {step_count} | contact={in_contact} | "
                        f"pen={pen_depth * 1000:.2f} mm | "
                        f"F_world={np.round(force_on_device_world, 2)} N | "
                        f"WRENCH={np.round(WRENCH[0:3], 2)} N"
                    )

                if step_count % RENDER_EVERY_N_STEPS == 0:
                    with viewer.lock():
                        draw_force_arrow(viewer, p_proxy, force_on_device_world)
                    viewer.sync()

                elapsed = time.time() - loop_start
                sleep_time = CONTROL_PERIOD_S - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    finally:
        rtde_c.forceModeStop()
        rtde_c.stopScript()


if __name__ == "__main__":
    main()