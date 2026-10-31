
import sys

import rtde_control
import rtde_receive


ROBOT_IP = sys.argv[1] if len(sys.argv) > 1 else "192.168.0.3"

# Control parameters
FORCE_GAIN = 0.006       # m/s per N
TORQUE_GAIN = 0.30       # rad/s per Nm

FORCE_DEADBAND = 3.0     # N
TORQUE_DEADBAND = 0.5    # Nm

MAX_LINEAR_SPEED = 0.03  # m/s
MAX_ANGULAR_SPEED = 0.15 # rad/s

ACCELERATION = 0.5
COMMAND_DURATION = 0.02


def clamp(value, limit):
    return max(-limit, min(limit, value))


def main():
    print(f"Connecting to robot: {ROBOT_IP}")
    controller = None
    receiver = None

    try:
        controller = rtde_control.RTDEControlInterface(ROBOT_IP)
        receiver = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
        input("Make sure the robot is stationary and untouched. Press Enter to zero the FT sensor...")

        controller.zeroFtSensor()

        print("Starting force-to-velocity control.")
        print("Press Ctrl+C to stop.")

        while True:
            wrench = receiver.getActualTCPForce()
            velocity = []

            for force in wrench[:3]:
                if abs(force) < FORCE_DEADBAND:
                    force = 0.0
                velocity.append(clamp(force * FORCE_GAIN, MAX_LINEAR_SPEED))

            for torque in wrench[3:6]:
                if abs(torque) < TORQUE_DEADBAND:
                    torque = 0.0
                velocity.append(clamp(torque * TORQUE_GAIN, MAX_ANGULAR_SPEED))

            controller.speedL(velocity, ACCELERATION, COMMAND_DURATION)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        if controller is not None:
            try:
                controller.speedStop(ACCELERATION)
            except Exception as exc:
                print(f"Could not send speedStop: {exc}")
            try:
                controller.stopScript()
            except Exception as exc:
                print(f"Could not stop the robot script: {exc}")
            try:
                controller.disconnect()
            except Exception as exc:
                print(f"Could not disconnect the control interface: {exc}")

        if receiver is not None:
            try:
                receiver.disconnect()
            except Exception as exc:
                print(f"Could not disconnect the receive interface: {exc}")

        print("Shutdown commands sent; verify the robot is stationary.")


if __name__ == "__main__":
    main()
