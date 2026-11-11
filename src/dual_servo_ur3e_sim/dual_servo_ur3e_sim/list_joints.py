"""Step 1 of the IK build: list the joints of the compiled MuJoCo model.

    python3 list_joints.py                      # default scene
    python3 list_joints.py path/to/scene.xml    # another scene

Free joints (the loose board pieces) are counted but not listed.
"""
import sys
from pathlib import Path

import mujoco

# Assumes this file sits next to your other scripts, one level below the package root.
DEFAULT_XML = Path(__file__).resolve().parent.parent / "mjcf" / "dual_tool_scene.xml"

# The six joints the IK will solve for, per side. Adjust the names if yours differ.
SIDES = ("left", "right")
IK_JOINTS = (
    "port_yaw",
    "port_pitch",
    "port_insert",
    "joint_roll",
    "joint_pitch",
    "joint_yaw",
)

# MuJoCo's joint type codes.
JOINT_TYPES = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}


def main():
    xml_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_XML
    model = mujoco.MjModel.from_xml_path(str(xml_path))

    print(f"Loaded {xml_path}")
    print(
        f"joints={model.njnt}  qpos size={model.nq}  "
        f"dofs={model.nv}  actuators={model.nu}\n"
    )

    print(f"{'id':>3}  {'name':<34} {'type':<6} {'qpos':>4} {'dof':>4}  range")
    hidden_free = 0
    for j in range(model.njnt):
        jtype = JOINT_TYPES[int(model.jnt_type[j])]
        if jtype == "free":
            hidden_free += 1
            continue
        name = model.joint(j).name or "(unnamed)"
        lo, hi = model.jnt_range[j]
        rng = f"[{lo:+.3f}, {hi:+.3f}]" if model.jnt_limited[j] else "unlimited"
        print(
            f"{j:>3}  {name:<34} {jtype:<6} "
            f"{model.jnt_qposadr[j]:>4} {model.jnt_dofadr[j]:>4}  {rng}"
        )
    print(f"\n({hidden_free} free joints not listed)")

    print("\nExpected IK joints:")
    for side in SIDES:
        for suffix in IK_JOINTS:
            name = f"{side}_{suffix}"
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            status = f"found (id {jid}, dof {model.jnt_dofadr[jid]})" if jid >= 0 else "MISSING"
            print(f"  {name:<24} {status}")


if __name__ == "__main__":
    main()