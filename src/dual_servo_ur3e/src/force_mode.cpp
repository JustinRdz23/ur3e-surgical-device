// //Add force_mode test executable to dual_servo_ur3e

// Copies UR client library's force_mode_example.cpp as a standalone
// test for validating force_mode as a freedrive substitute (freedrive
// can't be combined with external force/torque commands per driver docs).

// Changes from original example:
// - Bypassed checkCalibration() check (calibration constant not matched
//   to this URDF; not needed for this exploratory test)
// - Set startForceMode() to zero wrench, fully compliant on all 6 axes,
//   to approximate freedrive-like transparency as a baseline

// NOTE: must be run from the Universal_Robots_Client_Library workspace
// directory (not this package's directory), since RTDE recipe files and
// the external_control.urp reference are resolved via relative paths
// inherited from the original example:

//   cd ~/workspace/ros_ur_driver/src/Universal_Robots_Client_Library
//   ros2 run dual_servo_ur3e force_mode <robot_ip> <seconds>

// TODO: external_control.urp must exist in the target URSim container's
// program storage (created via PolyScope) or the run will fail with
// "Could not open script file"

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <ur_client_library/example_robot_wrapper.h>
#include <ur_client_library/ur/dashboard_client.h>
#include <ur_client_library/ur/ur_driver.h>
#include <ur_client_library/types.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <thread>
#include "ur_client_library/control/reverse_interface.h"

using namespace urcl;
const std::string DEFAULT_ROBOT_IP = "192.168.56.101";

// const std::string PACKAGE_SHARE = ament_index_cpp::get_package_share_directory("dual_servo_ur3e");

// const std::string SCRIPT_FILE = PACKAGE_SHARE + "/resources/external_control.urscript";
const std::string OUTPUT_RECIPE = "examples/resources/rtde_output_recipe.txt";
const std::string INPUT_RECIPE = "examples/resources/rtde_input_recipe.txt";

const std::string CALIBRATION_CHECKSUM = "calib_12788084448423163542";

std::unique_ptr<ExampleRobotWrapper> g_my_robot;

void sendFreedriveMessageOrDie(const control::FreedriveControlMessage freedrive_action)
{
  bool ret = g_my_robot->getUrDriver()->writeFreedriveControlMessage(freedrive_action);
  if (!ret)
  {
    URCL_LOG_ERROR("Could not send joint command. Is the robot in remote control?");
    exit(1);
  }
}

int main(int argc, char* argv[])
{
  urcl::setLogLevel(urcl::LogLevel::INFO);
  // Parse the ip arguments if given
  std::string robot_ip = DEFAULT_ROBOT_IP;
  if (argc > 1)
  {
    robot_ip = std::string(argv[1]);
  }

  // Parse how many seconds to run
  auto second_to_run = std::chrono::seconds(0);
  if (argc > 2)
  {
    second_to_run = std::chrono::seconds(std::stoi(argv[2]));
  }

  bool headless_mode = true;
  g_my_robot = std::make_unique<ExampleRobotWrapper>(robot_ip, OUTPUT_RECIPE, INPUT_RECIPE, headless_mode,
                                                     "external_control.urp");

  if (!g_my_robot->isHealthy())
  {
    URCL_LOG_ERROR("Something in the robot initialization went wrong. Exiting. Please check the output above.");
    return 1;
  }
//   if (!g_my_robot->getUrDriver()->checkCalibration(CALIBRATION_CHECKSUM))
//   {
//     URCL_LOG_ERROR("Calibration checksum does not match actual robot.");
//     URCL_LOG_ERROR("Use the ur_calibration tool to extract the correct calibration from the robot and pass that into "
//                    "the description. See "
//                    "[https://github.com/UniversalRobots/Universal_Robots_ROS_Driver#extract-calibration-information] "
//                    "for details.");
//   }

  // End of initialization -- We've started the external control program, which means we have to
  // write keepalive signals from now on. Otherwise the connection will be dropped.

  // Start force mode
  // Task frame at the robot's base with limits being large enough to cover the whole workspace
  // Compliance in z axis and rotation around z axis

  auto test_start = std::chrono::steady_clock::now();
  std::chrono::duration<double> time_done(0);
  std::chrono::duration<double> timeout(second_to_run);

  int iteration = 0;
  auto last_iter_time = std::chrono::steady_clock::now();

  URCL_LOG_INFO("Entering Force Mode test loop...");

  while (time_done < timeout || second_to_run.count() == 0)
  {
    double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - test_start).count();
    double fz = -2.0 + std::sin(t * 2.0 * M_PI);

    auto call_start = std::chrono::steady_clock::now();
    bool success = g_my_robot->getUrDriver()->startForceMode(
        { 0, 0, 0, 0, 0, 0 },
        { 1, 1, 1, 1, 1, 1 },
        { 0, fz, 0, 0, 0, 0 },
        2,
        { 0.1, 0.1, 1.5, 3.14, 3.14, 0.5 },
        0.005,
        1.0
    );
    auto call_end = std::chrono::steady_clock::now();

    if (!success)
    {
      URCL_LOG_ERROR("Failed to set force mode frame at iteration %d.", iteration);
      return 1;
    }

    g_my_robot->getUrDriver()->writeKeepalive();

    // Measure actual loop period, not just the sleep duration
    auto now = std::chrono::steady_clock::now();
    double call_ms = std::chrono::duration<double, std::milli>(call_end - call_start).count();
    double period_ms = std::chrono::duration<double, std::milli>(now - last_iter_time).count();
    last_iter_time = now;

    if (iteration % 100 == 0)  // log every 100th iteration, not every single one
    {
      URCL_LOG_INFO("iter=%d call=%.3fms period=%.3fms", iteration, call_ms, period_ms);
    }
    iteration++;

    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    time_done = std::chrono::steady_clock::now() - test_start;
  }

  URCL_LOG_INFO("Timeout reached. Exiting force mode...");
  g_my_robot->getUrDriver()->endForceMode();

  return 0;
}