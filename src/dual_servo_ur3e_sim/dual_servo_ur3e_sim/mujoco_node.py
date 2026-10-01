import mujoco
import mujoco.viewer
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class UR2MujocoBridge(Node):
    def __init__(self,data):
        super().__init__('ur2mujoco_bridge')
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.data = data

    def joint_state_callback(self, msg):
        self.data.ctrl[0] = msg.position[11] #wrist joint enables rotation in roll axis for tool
        self.data.ctrl[1] = msg.position[10] #wrist joint 2 enables rotation in yaw

def main(args=None):
    model = mujoco.MjModel.from_xml_path("/home/justinrc/workspace/src/dual_servo_ur3e_sim/mjcf/hello.xml")
    data = mujoco.MjData(model)
    
    rclpy.init(args=args)

    UR2MBridge = UR2MujocoBridge(data)
    try:

     with mujoco.viewer.launch_passive(model, data) as viewer:

        while viewer.is_running():
            rclpy.spin_once(UR2MBridge)
            mujoco.mj_step(model,data)
            viewer.sync()
     UR2MBridge.destroy_node()
     rclpy.shutdown()
        
    except KeyboardInterrupt:
        print("\nSimulation stopped.")

    finally:
        viewer.close()
        print("MuJoCo closed.")


if __name__ == '__main__':
    main()