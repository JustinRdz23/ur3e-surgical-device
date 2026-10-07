import time
import numpy as np
import rtde_control
import rtde_receive

ROBOT_IP = "192.168.0.3"  # change as needed

TASK_FRAME = [0, 0, 0, 0, 0, 0]
SELECTION_VECTOR = [1, 1, 1, 1, 1, 1]
WRENCH = [0, 0, 0, 0, 0, 0]
FORCE_TYPE = 2
LIMITS = [0.1, 0.1, 1.5, 3.14, 3.14, 0.5]
DAMPING = 0.004
GAIN_SCALING = 1.0

CONTROL_PERIOD_S = 0.002

BIAS_SAMPLE_DURATION_S = 3.0  # how long to sample the residual force at startup

rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)

print("Settling before zeroing F/T sensor -- do not touch the robot...")
time.sleep(2.0)
rtde_c.zeroFtSensor()
rtde_c.forceModeSetDamping(DAMPING)
rtde_c.forceModeSetGainScaling(GAIN_SCALING)

start_pose = rtde_r.getActualTCPPose()
print(f"Start pose: {[round(v, 4) for v in start_pose]}\n")

# --- Bias sampling phase ---
# Enter force mode with zero wrench, DO NOT touch the robot, and collect
# force readings over a short window to estimate whatever residual bias
# remains after zeroFtSensor().
print(f"Sampling residual force bias for {BIAS_SAMPLE_DURATION_S}s. DO NOT TOUCH THE ROBOT.\n")

bias_samples = []
sample_start = time.time()

while time.time() - sample_start < BIAS_SAMPLE_DURATION_S:
    loop_start = time.time()
    rtde_c.forceMode(TASK_FRAME, SELECTION_VECTOR, WRENCH, FORCE_TYPE, LIMITS)
    bias_samples.append(rtde_r.getActualTCPForce())

    elapsed = time.time() - loop_start
    sleep_time = CONTROL_PERIOD_S - elapsed
    if sleep_time > 0:
        time.sleep(sleep_time)

BIAS_FORCE = np.mean(np.array(bias_samples), axis=0)
print("==========================================")
print(f"Estimated bias force (will be subtracted): {[round(v, 4) for v in BIAS_FORCE]}")
print(f"Samples used: {len(bias_samples)}")
print("==========================================\n")

# --- Main loop: hold zero wrench, continuously compensating for the
# measured bias, and report drift + raw/compensated force over time ---
print("Entering compensated force mode. Do not touch the arm.")
print("Watching TCP pose and TCP force every second. Ctrl+C to stop.\n")

try:
    last_print = time.time()
    while True:
        loop_start = time.time()

        rtde_c.forceMode(TASK_FRAME, SELECTION_VECTOR, WRENCH, FORCE_TYPE, LIMITS)

        if time.time() - last_print >= 1.0:
            pose = rtde_r.getActualTCPPose()
            raw_force = np.array(rtde_r.getActualTCPForce())
            compensated_force = raw_force - BIAS_FORCE

            drift = [round(pose[i] - start_pose[i], 4) for i in range(6)]
            print(f"pose_drift       = {drift}")
            print(f"raw_force        = {[round(f, 3) for f in raw_force]}")
            print(f"compensated_force= {[round(f, 3) for f in compensated_force]}\n")
            last_print = time.time()

        elapsed = time.time() - loop_start
        sleep_time = CONTROL_PERIOD_S - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    rtde_c.forceModeStop()
    rtde_c.stopScript()
    print("Done.")