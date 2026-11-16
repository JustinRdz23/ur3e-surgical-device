from pathlib import Path
import time
import numpy as np
import scipy as sp
import scipy.linalg
import mujoco
import mujoco.viewer

# --- MODEL SETUP FOR STANDALONE TESTING ---
MJCF_PATH = str(Path(__file__).resolve().parent.parent / "mjcf" / "teleop_dual_IK.xml")
model = mujoco.MjModel.from_xml_path(MJCF_PATH)
data = mujoco.MjData(model)

site_name = "left_ee"
port_name = "left_port"

site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
port_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, port_name)

joint_names = [
    "left_port_yaw", "left_port_pitch", "left_port_insert",
    "left_joint_roll", "left_joint_pitch", "left_joint_yaw"
]
joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
qpos_ids = model.jnt_qposadr[joint_ids]
dof_ids = model.jnt_dofadr[joint_ids]
ranges = model.jnt_range[joint_ids]    # joint limits, used to clip theta

# Pre-compute base-to-world transform (T_rb_world)
scratch = mujoco.MjData(model)
mujoco.mj_kinematics(model, scratch)
T_world_rb = np.eye(4)
T_world_rb[:3, :3] = scratch.xmat[port_id].reshape(3, 3)
T_world_rb[:3, 3]  = scratch.xpos[port_id]
T_rb_world = sp.linalg.inv(T_world_rb)

# Error tolerances
omega_epsilon = 0.0001
vel_epsilon = 0.0001
max_iter = 1000


def get_T_sb(theta):
    """Computes the Base-to-End-Effector homogeneous transformation matrix."""
    data.qpos[qpos_ids] = theta
    mujoco.mj_kinematics(model, data)

    T_world_ee = np.eye(4)
    T_world_ee[:3, :3] = data.site_xmat[site_id].reshape(3, 3)
    T_world_ee[:3, 3] = data.site_xpos[site_id]

    return T_rb_world @ T_world_ee


def twist_error(theta, T_sd):
    """Computes the 6D spatial twist error vector between current pose and desired T_sd."""
    T_curr = get_T_sb(theta)
    T_bd = sp.linalg.inv(T_curr) @ T_sd
    V_mat = sp.linalg.logm(T_bd).real
    
    V_vector = np.zeros(6)
    V_vector[0] = V_mat[2, 1]      # ω1
    V_vector[1] = V_mat[0, 2]      # ω2
    V_vector[2] = V_mat[1, 0]      # ω3
    V_vector[3:] = V_mat[:3, 3]    # linear velocity / position error
    return V_vector


def Jacobian(theta):
    """Computes the 6x6 Jacobian for the active left arm joints, in the end-effector frame."""
    data.qpos[qpos_ids] = theta
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

    R_ee = data.site_xmat[site_id].reshape(3, 3)
    jacp_filtered = R_ee.T @ jacp[:, dof_ids]
    jacr_filtered = R_ee.T @ jacr[:, dof_ids]
    
    return np.vstack((jacr_filtered, jacp_filtered))


def Newton_Raphson_IK(T_sd, theta, viewer=None):
    """Runs Newton-Raphson inverse kinematics, optionally updating a live viewer each step."""
    theta = np.clip(np.array(theta, dtype=float), ranges[:, 0], ranges[:, 1])
    i = 0
    ok = False
    
    while i < max_iter:
        V_vector = twist_error(theta, T_sd)

        err_omega = np.linalg.norm(V_vector[:3])
        err_vel = np.linalg.norm(V_vector[3:])
        
        if err_omega < omega_epsilon and err_vel < vel_epsilon:
            ok = True
            break

        J = Jacobian(theta)
        theta = theta + np.linalg.pinv(J) @ V_vector
        theta = np.clip(theta, ranges[:, 0], ranges[:, 1])
        
        # Update kinematics and viewer visually (without physics overriding qpos)
        if viewer is not None:
            data.qpos[qpos_ids] = theta
            mujoco.mj_kinematics(model, data)
            viewer.sync()
            time.sleep(0.02)
            
        i += 1

    return theta, ok


# --- STANDALONE TEST SCRIPT WITH VISUALIZATION ---
if __name__ == "__main__":
    # 1. Forward kinematics & joint limits check
    T0 = get_T_sb(np.zeros(6))
    R0 = T0[:3, :3]
    print("R orthonormal:", np.allclose(R0 @ R0.T, np.eye(3)), "| det:", round(float(np.linalg.det(R0)), 6))
    print("Joint limits (min, max):\n", ranges)

    # Target angles matching the MuJoCo Control panel in the image
    theta_true = np.array([-0.6, 0.6, -0.06, 3.14, 1.57, 1.57])
    theta_true_clipped = np.clip(theta_true, ranges[:, 0], ranges[:, 1])
    T_sd = get_T_sb(theta_true_clipped)

    # Start slightly off-zero on roll (0.1 rad) to avoid the 180-deg (3.14 rad) logm branch cut
    theta_init = np.array([0.0, 0.0, 0.0, 0.1, 0.0, 0.0])

    print("\nLaunching MuJoCo 3D Viewer...")
    print("Watch the robot arm solve IK and move to the target pose.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Reset to starting configuration in the viewer
        data.qpos[qpos_ids] = theta_init
        mujoco.mj_kinematics(model, data)
        viewer.sync()
        time.sleep(1.0)

        # Run IK while streaming updates to the live window
        theta_found, ok = Newton_Raphson_IK(T_sd, theta_init, viewer=viewer)
        
        err = twist_error(theta_found, T_sd)
        print("\n--- Solver Results ---")
        print("Solver converged:", ok, "| final twist norm:", float(np.linalg.norm(err)))
        print("target theta (clipped):", np.round(theta_true_clipped, 4))
        print("found  theta          :", np.round(theta_found, 4))

        # Keep viewer open and hold the final solved pose kinematically
        print("\nSimulation finished converging. Close the window to exit.")
        while viewer.is_running():
            data.qpos[qpos_ids] = theta_found
            mujoco.mj_kinematics(model, data)
            viewer.sync()
            time.sleep(0.01)