import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Launch Configurations
    justi_ip = LaunchConfiguration("justi_ip")
    gracie_ip = LaunchConfiguration("gracie_ip")
    justi_kinematics = LaunchConfiguration("justi_kinematics_parameters_file")
    gracie_kinematics = LaunchConfiguration("gracie_kinematics_parameters_file")
    use_mock_hardware = LaunchConfiguration("use_mock_hardware")
    start_rviz = LaunchConfiguration("start_rviz")

    # Command to process Xacro -> URDF string
    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name="xacro")]), " ",
        PathJoinSubstitution([FindPackageShare("dual_servo_ur3e"), "urdf", "dual_robot_cell.urdf.xacro"]), " ",
        "justi_ip:=", justi_ip, " ",
        "gracie_ip:=", gracie_ip, " ",
        "justi_kinematics_parameters_file:=", justi_kinematics, " ",
        "gracie_kinematics_parameters_file:=", gracie_kinematics, " ",
        "use_mock_hardware:=", use_mock_hardware,
    ])
    
    robot_description = {"robot_description": robot_description_content}

    # Path to controllers YAML
    controllers_yaml = PathJoinSubstitution(
        [FindPackageShare("dual_servo_ur3e"), "config", "dual_controllers.yaml"]
    )

    # Path to RViz configuration
    rviz_config_file = PathJoinSubstitution(
        [FindPackageShare("dual_servo_ur3e"), "rviz", "view_cell.rviz"]
    )

    # ------------------ Controller Spawners ------------------
    # Justi Spawners
    justi_jsb_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["justi_joint_state_broadcaster", "-c", "/controller_manager"],
    )

    justi_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["justi_scaled_joint_trajectory_controller", "-c", "/controller_manager"],
    )

    # Gracie Spawners
    gracie_jsb_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["gracie_joint_state_broadcaster", "-c", "/controller_manager"],
    )

    gracie_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["gracie_scaled_joint_trajectory_controller", "-c", "/controller_manager"],
    )

    return LaunchDescription([
        # Declare Launch Arguments
        DeclareLaunchArgument("justi_ip", default_value="192.168.56.101"),
        DeclareLaunchArgument("gracie_ip", default_value="192.168.56.102"),
        DeclareLaunchArgument(
            "justi_kinematics_parameters_file",
            default_value=PathJoinSubstitution([FindPackageShare("dual_servo_ur3e"), "config", "justi_calibration.yaml"])
        ),
        DeclareLaunchArgument(
            "gracie_kinematics_parameters_file",
            default_value=PathJoinSubstitution([FindPackageShare("dual_servo_ur3e"), "config", "gracie_calibration.yaml"])
        ),
        DeclareLaunchArgument("use_mock_hardware", default_value="false"),
        DeclareLaunchArgument("start_rviz", default_value="true"),

        # Core Nodes
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="both",
            parameters=[robot_description],
        ),

        Node(
            package="controller_manager",
            executable="ros2_control_node",
            parameters=[robot_description, controllers_yaml],
            output="screen",
        ),

        # Visualization
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="log",
            arguments=["-d", rviz_config_file],
            condition=IfCondition(start_rviz),
        ),

        # Spawners
        justi_jsb_spawner,
        justi_controller_spawner,
        gracie_jsb_spawner,
        gracie_controller_spawner,
    ])