"""Reusable 6D pose IK for one tool arm of the dual-tool scene.

Give it a target pose for the <side>_ee site and it returns the six joint values
(port yaw, pitch, insert + wrist roll, pitch, yaw). It works on a private copy of the sim
data, so it never disturbs a running simulation.

Run it directly for a self-check and a few example waypoints:

    python3 ik_solver.py                  # left arm, default scene
    python3 ik_solver.py --right          # right arm
    python3 ik_solver.py --view           # also play the waypoints in the viewer
    python3 ik_solver.py path/to/scene.xml

Use it from another script (this file next to yours):

    from ik_solver import PoseIK

    ik = PoseIK(model, "left")                      # build once, before your loop
    result = ik.solve(pos, quat)                    # target in MuJoCo world coordinates
    result = ik.solve(pos, quat, frame="rb")        # target in the port (RCM) frame
    result = ik.solve(pos)                          # position only: keeps current orientation
    if result.converged:
        joint_values = result.q                     # (yaw, pitch, insert, roll, w_pitch, w_yaw)

Quaternions are (w, x, y, z), positions are metres.
"""
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import mujoco

# Assumes this file sits next to your other scripts, one level below the package root.
DEFAULT_XML = Path(__file__).resolve().parent.parent / "mjcf" / "teleop_dual_IK.xml"

# Solution vector order.
JOINT_SUFFIXES = (
    "port_yaw",
    "port_pitch",
    "port_insert",
    "joint_roll",
    "joint_pitch",
    "joint_yaw",
)

MAX_ITERS = 100
TOL_POS = 1e-4     # metres
TOL_ROT = 1e-3     # radians
DAMPING = 0.01     # damped least squares: keeps steps sane near singularities
MAX_STEP = 0.05    # largest change per iteration, rad (hinges) or m (slide)


# ── Small quaternion helpers (all quaternions are (w, x, y, z)) ───────────────

def mat_to_quat(mat):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(mat, dtype=float).flatten())
    return q


def quat_to_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, dtype=float))
    return m.reshape(3, 3)


def quat_mul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return out


def rotvec_to_quat(rotvec):
    """Rotation vector (axis * angle, radians) -> quaternion."""
    rotvec = np.asarray(rotvec, dtype=float)
    angle = np.linalg.norm(rotvec)
    q = np.array([1.0, 0.0, 0.0, 0.0])
    if angle > 1e-12:
        mujoco.mju_axisAngle2Quat(q, rotvec / angle, angle)
    return q


def quat_error(q_target, q_current):
    """3D rotation vector (world frame) that takes q_current to q_target."""
    q_inv = np.zeros(4)
    mujoco.mju_negQuat(q_inv, q_current)
    q_err = quat_mul(q_target, q_inv)
    err = np.zeros(3)
    mujoco.mju_quat2Vel(err, q_err, 1.0)
    return err


@dataclass
class IKResult:
    q: np.ndarray        # six joint values, in JOINT_SUFFIXES order
    converged: bool      # False = best effort only (target unreachable or iteration limit)
    pos_err: float       # metres
    rot_err: float       # radians
    iters: int


class PoseIK:
    """IK for one arm. Build it once, then call solve() as often as you like."""

    def __init__(self, model, side, port_body=None):
        self.model = model
        self.data = mujoco.MjData(model)   # private scratch copy of the sim state

        site_name = f"{side}_ee"
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id < 0:
            raise ValueError(f"Site '{site_name}' not found in the model.")

        joint_ids = []
        for suffix in JOINT_SUFFIXES:
            name = f"{side}_{suffix}"
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"Joint '{name}' not found in the model.")
            joint_ids.append(jid)

        # qpos index = where a joint's value lives; dof index = its Jacobian column.
        # They differ because free joints use 7 qpos entries but only 6 dofs.
        self.qpos_ids = model.jnt_qposadr[joint_ids]
        self.dof_ids = model.jnt_dofadr[joint_ids]
        self.ranges = model.jnt_range[joint_ids]

        # {RB}: the port body's pose with all joints at zero, expressed in the world.
        port_name = port_body or f"{side}_port"
        self.port_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, port_name)
        if self.port_id >= 0:
            self.p_rb = model.body_pos[self.port_id].copy()
            self.q_rb = model.body_quat[self.port_id].copy()
            self.R_rb = quat_to_mat(self.q_rb)
        else:
            self.p_rb = self.q_rb = self.R_rb = None   # frame="rb" will refuse to run

        # Warm start: each solve begins from the previous solution.
        self.q_last = np.clip(np.zeros(6), self.ranges[:, 0], self.ranges[:, 1])

    def forward(self, q):
        """Forward kinematics: six joint values -> (site position, site quaternion)."""
        self.data.qpos[self.qpos_ids] = q
        mujoco.mj_kinematics(self.model, self.data)
        pos = self.data.site_xpos[self.site_id].copy()
        quat = mat_to_quat(self.data.site_xmat[self.site_id].reshape(3, 3))
        return pos, quat

    def solve(self, target_pos, target_quat=None, q_init=None, frame="world"):
        if frame not in ("world", "rb"):
            raise ValueError("frame must be 'world' or 'rb'")

        target_pos = np.asarray(target_pos, dtype=float)
        if target_quat is not None:
            target_quat = np.asarray(target_quat, dtype=float)

        if frame == "rb":
            if self.p_rb is None:
                raise ValueError("Port body not found, so frame='rb' is unavailable.")
            target_pos = self.p_rb + self.R_rb @ target_pos
            if target_quat is not None:
                target_quat = quat_mul(self.q_rb, target_quat)

        if target_quat is None:
            _, target_quat = self.forward(self.q_last)   # keep the current orientation

        start = self.q_last if q_init is None else q_init
        q = np.clip(np.asarray(start, dtype=float), self.ranges[:, 0], self.ranges[:, 1])

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        pos_err = rot_err = float("inf")

        for it in range(MAX_ITERS + 1):
            pos, quat = self.forward(q)
            e_pos = target_pos - pos
            e_rot = quat_error(target_quat, quat)
            pos_err = float(np.linalg.norm(e_pos))
            rot_err = float(np.linalg.norm(e_rot))

            if pos_err < TOL_POS and rot_err < TOL_ROT:
                self.q_last = q.copy()
                return IKResult(q, True, pos_err, rot_err, it)
            if it == MAX_ITERS:
                break

            # The Jacobian needs mj_kinematics (done in forward) and mj_comPos first.
            mujoco.mj_comPos(self.model, self.data)
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
            J = np.vstack((jacp[:, self.dof_ids], jacr[:, self.dof_ids]))   # 6 x 6
            err = np.concatenate((e_pos, e_rot))

            dq = J.T @ np.linalg.solve(J @ J.T + DAMPING**2 * np.eye(6), err)
            dq = np.clip(dq, -MAX_STEP, MAX_STEP)
            q = np.clip(q + dq, self.ranges[:, 0], self.ranges[:, 1])

        self.q_last = q.copy()   # best effort, kept as the next warm start
        return IKResult(q, False, pos_err, rot_err, MAX_ITERS)


# ── Demo / self-check ─────────────────────────────────────────────────────────

def describe(label, r):
    status = "OK  " if r.converged else "FAIL"
    print(
        f"{label:<22} {status} iters={r.iters:>3}  "
        f"pos_err={r.pos_err * 1000:8.4f} mm  rot_err={r.rot_err * 1000:8.4f} mrad"
    )
    print(f"{'':<22} q = {np.round(r.q, 4)}")


def main():
    argv = sys.argv[1:]
    view = "--view" in argv
    side = "right" if "--right" in argv else "left"
    paths = [a for a in argv if not a.startswith("--")]
    xml_path = Path(paths[0]) if paths else DEFAULT_XML

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    ik = PoseIK(model, side)
    print(f"Loaded {xml_path}; side={side}\n")

    # 1. {RB} frame check: the port body at zero joints should sit exactly at model.body_pos.
    if ik.p_rb is not None:
        mujoco.mj_resetData(model, ik.data)
        mujoco.mj_kinematics(model, ik.data)
        diff = np.abs(ik.data.xpos[ik.port_id] - ik.p_rb).max()
        print(f"{{RB}} frame check: max |xpos - body_pos| = {diff:.2e} m (expect ~0)\n")

    # 2. Round trip: build a pose with forward kinematics, then recover it from zero.
    q_true = np.array([0.25, 0.15, 0.03, 0.5, -0.4, 0.3])
    pos, quat = ik.forward(q_true)
    ik.q_last = np.zeros(6)
    result = ik.solve(pos, quat)
    describe("round trip", result)
    print(f"{'':<22} true = {q_true}")
    print("   (a different q can still be right: convergence of the POSE is what counts)\n")

    # 3. Example waypoints, defined as offsets from the pose at all-zero joints.
    ik.q_last = np.zeros(6)
    home_pos, home_quat = ik.forward(np.zeros(6))
    waypoints = [
        ("home", [0, 0, 0], [0, 0, 0]),
        ("+1 cm world x", [0.01, 0, 0], [0, 0, 0]),
        ("+1 cm world y", [0, 0.01, 0], [0, 0, 0]),
        ("+1 cm world z", [0, 0, 0.01], [0, 0, 0]),
        ("twist 0.3 rad (tool x)", [0, 0, 0], [0.3, 0, 0]),
        ("unreachable (+30 cm)", [0.3, 0, 0], [0, 0, 0]),
    ]
    solved = []
    for label, dpos, rotvec in waypoints:
        target_pos = home_pos + np.array(dpos)
        target_quat = quat_mul(home_quat, rotvec_to_quat(rotvec))
        r = ik.solve(target_pos, target_quat)
        describe(label, r)
        solved.append((label, r.q.copy()))

    # 4. Optional: show each solution in the viewer (pure kinematics, no physics).
    if view:
        import mujoco.viewer as mj_viewer

        data = mujoco.MjData(model)
        if model.nkey > 0:
            mujoco.mj_resetDataKeyframe(model, data, 0)
        with mj_viewer.launch_passive(model, data) as viewer:
            for label, q in solved:
                if not viewer.is_running():
                    break
                print(f"showing: {label}")
                data.qpos[ik.qpos_ids] = q
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(1.5)
            while viewer.is_running():
                time.sleep(0.1)


if __name__ == "__main__":
    main()