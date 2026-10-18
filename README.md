# UR3e Surgical Device

Dual-arm **UR3e** robotic cell (*Justi* and *Gracie*) for surgical teleoperation with a dVRK-style instrument (Large Needle Driver: roll, pitch and gripper).

## Components

| Part | What it does |
|---|---|
| `src/servo_ur3e` | Keyboard teleoperation of a single UR3e with MoveIt Servo |
| `src/dual_servo_ur3e` | Dual cell: URDF/xacro, controllers, per-arm calibration and RViz visualization |
| `src/dual_servo_ur3e_sim` *(in development)* | **MuJoCo** simulation of the instrument: contacts, object grasping (needle, ring) and the bridge to Unity |
| `unity-sim/` *(in development)* | **Unity** scene that only renders what MuJoCo computes |

**All physics comes from MuJoCo; Unity only renders.** The real UR3e acts as the master and the simulated instrument follows it.

## Requirements

ROS 2, `ur_robot_driver`, `ur_description` and MoveIt. The simulation also needs Python with `mujoco` and Unity 6.

## Branches

- `main`: dual cell in ROS 2.
- `implementationUnity`: MuJoCo + Unity simulation, not yet merged into `main`.
- `*-objectCreation`: simulation objects (needle, ring).
