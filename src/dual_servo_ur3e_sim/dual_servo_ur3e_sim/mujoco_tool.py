import mujoco
import mujoco.viewer
import numpy as np
model = mujoco.MjModel.from_xml_path("/home/justinrc/workspace/src/dual_servo_ur3e_sim/mjcf/hello.xml")
data = mujoco.MjData(model)

try:

    with mujoco.viewer.launch_passive(model, data) as viewer:

        while viewer.is_running():

            t = data.time
            data.ctrl[0] = 0.5*np.sin(t)
            data.ctrl[1] = 0.5*np.sin(t)
            data.ctrl[2] = 0.5*np.sin(t)
            data.ctrl[3] = 0.5*np.sin(t)

            

            mujoco.mj_step(model, data)

            viewer.sync()

except KeyboardInterrupt:
    print("\nSimulation stopped.")

finally:
    viewer.close()
    print("MuJoCo closed.")