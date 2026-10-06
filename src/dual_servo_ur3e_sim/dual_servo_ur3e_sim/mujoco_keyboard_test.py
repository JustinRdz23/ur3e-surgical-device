"""
No-robot test version of the mocap-driven tool.

Instead of reading the real robot's TCP pose over RTDE, this drives the
mocap body directly from the keyboard, so the weld constraint + box
interaction can be tested without the arm connected.

Controls:
  Arrow keys   -> move in X/Y (world plane)
  Page Up/Down -> move in Z (up/down)
  I / K        -> pitch (rotate about local X)
  J / L        -> yaw   (rotate about local Z)
  U / O        -> roll  (rotate about local Y)
  R            -> reset to the mocap body's starting pose
  Esc / close window -> quit
"""

import time
import numpy as np
import mujoco
import mujoco.viewer
import glfw

MJCF_PATH = "/home/xpatricia-garcia/repos/medicalUR/ur3e-surgical-device/src/dual_servo_ur3e_sim/mjcf/hello.xml"

MOCAP_ID = 0  # index of the mocap body in mocap_pos/mocap_quat

POS_STEP = 0.005    # meters moved per key press
ANGLE_STEP = np.deg2rad(3.0)  # radians rotated per key press

RENDER_PERIOD_S = 1.0 / 60.0  # ~60Hz, no RTDE constraint here so just render smoothly


class TeleopState:
    """Holds the mocap target pose, updated from key presses."""

    def __init__(self, initial_pos, initial_quat):
        self.initial_pos = np.array(initial_pos, dtype=float)
        self.initial_quat = np.array(initial_quat, dtype=float)
        self.pos = self.initial_pos.copy()
        self.quat = self.initial_quat.copy()

    def reset(self):
        self.pos = self.initial_pos.copy()
        self.quat = self.initial_quat.copy()

    def move(self, dx=0.0, dy=0.0, dz=0.0):
        self.pos += np.array([dx, dy, dz])

    def rotate_local(self, axis, angle):
        """Rotate the current orientation by `angle` about local `axis`."""
        delta_quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(delta_quat, np.array(axis, dtype=float), angle)
        new_quat = np.zeros(4)
        # Local-frame rotation: current_quat * delta_quat
        mujoco.mju_mulQuat(new_quat, self.quat, delta_quat)
        self.quat = new_quat


def make_key_callback(state: TeleopState):
    def key_callback(keycode):
        # Position: arrow keys (X/Y), Page Up/Down (Z)
        if keycode == glfw.KEY_UP:
            state.move(dy=POS_STEP)
        elif keycode == glfw.KEY_DOWN:
            state.move(dy=-POS_STEP)
        elif keycode == glfw.KEY_LEFT:
            state.move(dx=-POS_STEP)
        elif keycode == glfw.KEY_RIGHT:
            state.move(dx=POS_STEP)
        elif keycode == glfw.KEY_PAGE_UP:
            state.move(dz=POS_STEP)
        elif keycode == glfw.KEY_PAGE_DOWN:
            state.move(dz=-POS_STEP)

        # Orientation: I/K pitch, J/L yaw, U/O roll
        elif keycode == glfw.KEY_I:
            state.rotate_local(axis=[1, 0, 0], angle=ANGLE_STEP)
        elif keycode == glfw.KEY_K:
            state.rotate_local(axis=[1, 0, 0], angle=-ANGLE_STEP)
        elif keycode == glfw.KEY_J:
            state.rotate_local(axis=[0, 0, 1], angle=ANGLE_STEP)
        elif keycode == glfw.KEY_L:
            state.rotate_local(axis=[0, 0, 1], angle=-ANGLE_STEP)
        elif keycode == glfw.KEY_U:
            state.rotate_local(axis=[0, 1, 0], angle=ANGLE_STEP)
        elif keycode == glfw.KEY_O:
            state.rotate_local(axis=[0, 1, 0], angle=-ANGLE_STEP)

        elif keycode == glfw.KEY_R:
            state.reset()

    return key_callback


def main():
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)

    # Start the teleop state from wherever the mocap body is placed in the XML
    state = TeleopState(
        initial_pos=data.mocap_pos[MOCAP_ID].copy(),
        initial_quat=data.mocap_quat[MOCAP_ID].copy(),
    )

    print("Keyboard teleop test (no robot connected).")
    print("  Arrows: move X/Y | PageUp/PageDown: move Z")
    print("  I/K: pitch | J/L: yaw | U/O: roll | R: reset | Esc: quit")

    with mujoco.viewer.launch_passive(
        model, data, key_callback=make_key_callback(state)
    ) as viewer:
        while viewer.is_running():
            loop_start = time.time()

            data.mocap_pos[MOCAP_ID] = state.pos
            data.mocap_quat[MOCAP_ID] = state.quat

            mujoco.mj_step(model, data)
            viewer.sync()

            elapsed = time.time() - loop_start
            sleep_time = RENDER_PERIOD_S - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    print("Done.")


if __name__ == "__main__":
    main()