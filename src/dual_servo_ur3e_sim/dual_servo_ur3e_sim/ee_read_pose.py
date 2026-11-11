import mujoco
import mujoco.viewer
import time

model = mujoco.MjModel.from_xml_path("/home/justinrc/workspace/src/dual_servo_ur3e_sim/mjcf/teleop_dual_IK.xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        
        # Read site position [x, y, z] and orientation rotation matrix (3x3 flattened) live
        site_pos = data.site("left_ee").xpos
        site_mat = data.site("left_ee").xmat
        
        print(f"Live Position: {site_pos}")
        
        viewer.sync()
        time.sleep(0.002)