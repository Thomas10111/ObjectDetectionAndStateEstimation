"""Start Gazebo with the LEGO car in the test room and bridge it to ROS 2.

    ros2 launch odse_sim sim.launch.py                  # with Gazebo window
    ros2 launch odse_sim sim.launch.py gui:=false       # server only
    ros2 launch odse_sim sim.launch.py rviz:=true       # also open RViz

Drive it from a second terminal:

    ros2 run teleop_twist_keyboard teleop_twist_keyboard
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    share = Path(get_package_share_directory("odse_sim"))
    gz_launch = Path(get_package_share_directory("ros_gz_sim")) / "launch" / "gz_sim.launch.py"

    gui = LaunchConfiguration("gui")
    rviz = LaunchConfiguration("rviz")
    world = LaunchConfiguration("world")
    camera_x = LaunchConfiguration("camera_x")
    camera_z = LaunchConfiguration("camera_z")
    cameras = LaunchConfiguration("cameras")
    car_x = LaunchConfiguration("car_x")
    car_y = LaunchConfiguration("car_y")
    car_yaw = LaunchConfiguration("car_yaw")

    robot_description = ParameterValue(
        Command(
            [
                "xacro ",
                str(share / "urdf" / "lego_car.urdf.xacro"),
                " camera_x:=",
                camera_x,
                " camera_z:=",
                camera_z,
                " cameras:=",
                cameras,
            ]
        ),
        value_type=str,
    )

    # "-r" starts the simulation running; "-s" runs the server without a window.
    gz_args = PythonExpression(
        ["'-r ' + ('' if '", gui, "' == 'true' else '-s ') + '", world, "'"]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("gui", default_value="true"),
            DeclareLaunchArgument("rviz", default_value="false"),
            DeclareLaunchArgument("world", default_value=str(share / "worlds" / "room.sdf")),
            DeclareLaunchArgument("camera_x", default_value="0.25"),
            DeclareLaunchArgument("camera_z", default_value="0.15"),
            DeclareLaunchArgument("car_x", default_value="0.0", description="start position of the car, m"),
            DeclareLaunchArgument("car_y", default_value="0.0", description="start position of the car, m"),
            DeclareLaunchArgument("car_yaw", default_value="0.0", description="start heading of the car, rad (0 = +x)"),
            DeclareLaunchArgument(
                "cameras", default_value="true", description="false: no rendered cameras, faster"
            ),
            # Gazebo's own transport finds its peers by multicast, which fails with
            # WSL mirrored networking ("Network is unreachable"). Keep it on this
            # machine; ROS 2 traffic to the Orin is not affected.
            SetEnvironmentVariable("GZ_IP", "127.0.0.1"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(gz_launch)),
                launch_arguments={"gz_args": gz_args, "on_exit_shutdown": "true"}.items(),
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                parameters=[{"robot_description": robot_description, "use_sim_time": True}],
                output="screen",
            ),
            Node(
                package="ros_gz_sim",
                executable="create",
                arguments=[
                    "-param", "robot_description", "-name", "lego_car",
                    "-x", car_x, "-y", car_y, "-z", "0.01", "-Y", car_yaw,
                ],
                parameters=[{"robot_description": robot_description}],
                output="screen",
            ),
            Node(
                package="ros_gz_bridge",
                executable="parameter_bridge",
                parameters=[
                    {"config_file": str(share / "config" / "bridge.yaml"), "use_sim_time": True}
                ],
                output="screen",
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                arguments=["-d", str(share / "config" / "sim.rviz")],
                parameters=[{"use_sim_time": True}],
                condition=IfCondition(rviz),
            ),
        ]
    )
