cmake_minimum_required(VERSION 3.8)
project(dual_servo_ur3e)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(control_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)

add_executable(dual_servo_keyboard src/dual_servo_keyboard.cpp)
ament_target_dependencies(dual_servo_keyboard
  rclcpp
  geometry_msgs
  control_msgs
  moveit_msgs
)
target_include_directories(dual_servo_keyboard PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

install(TARGETS dual_servo_keyboard
  DESTINATION lib/${PROJECT_NAME}
)

# Install non-code assets so $(find dual_servo_ur3e) / FindPackageShare(...)
# can locate them at runtime after this package is built.
install(DIRECTORY
  urdf
  launch
  config
  rviz
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()