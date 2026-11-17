import numpy as np
import mujoco

MOCAP_BODIES = {"left": "left_arm_mocap", "right": "right_arm_mocap"}


def RB_T_H(model, data, T_RB_world, side="left"):
    """
    Pose of the virtual hand (the mocap body of `side`) expressed in {RB}:
    P_RB = RB_T_H @ P_H
    RB_T_H = (World_T_RB)^-1 @ (World_T_H)

    T_RB_world is the inverse of the fixed port pose (4x4). Call it inside viewer.lock(),
    since it recomputes the kinematics of `data`.
    """
    # Ensure kinematics are up to date if mocap_pos/quat were just modified
    mujoco.mj_kinematics(model, data)

    # Use body_id for data.xpos and data.xmat
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, MOCAP_BODIES[side])

    T_world_H = np.eye(4)
    T_world_H[:3, :3] = data.xmat[body_id].reshape(3, 3)  # Rotation (3x3)
    T_world_H[:3, 3] = data.xpos[body_id]                 # Translation (3,)

    return T_RB_world @ T_world_H