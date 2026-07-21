cmake_minimum_required(VERSION 3.8)
project(servo_ur3e)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(control_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)

add_executable(servo_keyboard_input src/servo_keyboard.cpp)
ament_target_dependencies(servo_keyboard_input
  rclcpp
  geometry_msgs
  control_msgs
  moveit_msgs
)

install(TARGETS servo_keyboard_input
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()