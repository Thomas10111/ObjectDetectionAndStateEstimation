"""Driving experiment in Gazebo: the trained network drives the car to the ball.

    ros2 launch odse_driving experiment.launch.py                         # PPO, ball ahead, room world
    ros2 launch odse_driving experiment.launch.py model:=ppo_search world:=arena target_x:=-1.0 target_y:=1.2
    ros2 launch odse_driving experiment.launch.py model:=bc               # imitation network
    ros2 launch odse_driving experiment.launch.py gui:=false              # no window, faster
    ros2 launch odse_driving experiment.launch.py noise:=1.0 dropout:=0.1 # training like detections
    ros2 launch odse_driving experiment.launch.py policy:=orin            # network runs on the Orin
    ros2 launch odse_driving experiment.launch.py policy:=orin target:=orin   # and YOLO too

With policy:=orin no policy node is started here: run policy.launch.py on the
Orin (scripts/orin_policy_windows.py run), which picks /target up over Zenoh
and sends /drive/command back. The model and start_delay arguments then have
to be given on the Orin. With target:=orin, YOLO and the target node on the
Orin (policy.launch.py target:=yolo) turn the rendered camera images into
/target: the cameras are switched on and sim_target is not started.

Starts the simulation (odse_sim, cameras off by default for speed), the
stand in for camera and YOLO (sim_target), the policy, the adapter to
Gazebo's /cmd_vel, and the episode monitor, which reports the outcome and
writes a CSV log to ~/odse_experiments/. The car starts at the origin facing
+x unless car_x, car_y and car_yaw say otherwise; target_x and target_y place
the ball. Ball 3 m from the camera, in the larger arena:

    ros2 launch odse_driving experiment.launch.py world:=arena car_x:=-1.45
"""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

MODELS = {
    "ppo": "ppo_policy.onnx",
    "ppo_search": "ppo_search_policy.onnx",
    "bc": "bc_policy.onnx",
    # 4 hidden units instead of 64 x 64, section 8.1 of docs/how_it_works.md
    "ppo_h4": "ppo_h4_policy.onnx",
    "ppo_search_h4": "ppo_search_h4_policy.onnx",
}
BALL_POSE_IN_WORLD_FILES = "<pose>1.8 -0.3 0.1 0 0 0</pose>"


def world_with_ball(world: str, x: float, y: float) -> str:
    """Copy of the world file with the target ball moved to (x, y)."""
    source = Path(get_package_share_directory("odse_sim")) / "worlds" / f"{world}.sdf"
    text = source.read_text()
    if BALL_POSE_IN_WORLD_FILES not in text:
        raise RuntimeError(f"{source} has no target_ball at the default pose")
    path = Path(f"/tmp/odse_{world}_{os.getpid()}.sdf")
    path.write_text(text.replace(BALL_POSE_IN_WORLD_FILES, f"<pose>{x} {y} 0.1 0 0 0</pose>"))
    return str(path)


def nodes(context):
    def arg(name: str) -> str:
        return LaunchConfiguration(name).perform(context)

    share = Path(get_package_share_directory("odse_driving"))
    model_path = str(share / "models" / MODELS.get(arg("model"), arg("model")))
    venv_python = os.path.expanduser(arg("venv")) + "/bin/python"
    world = world_with_ball(arg("world"), float(arg("target_x")), float(arg("target_y")))
    target = {"target_model": "target_ball", "target_radius": 0.1, "use_sim_time": True}
    yolo_on_orin = arg("target") == "orin"
    sim_launch = Path(get_package_share_directory("odse_sim")) / "launch" / "sim.launch.py"
    policy = Node(
        package="odse_driving",
        executable="policy",
        prefix=venv_python,  # onnxruntime lives in this environment
        parameters=[
            {
                "model": model_path,
                "speed_limit": 0.4,
                "start_delay": float(arg("start_delay")),
                "use_sim_time": True,
            }
        ],
        output="screen",
    )
    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(sim_launch)),
            launch_arguments={
                "gui": arg("gui"),
                "rviz": arg("rviz"),
                "cameras": "true" if yolo_on_orin else arg("cameras"),
                "world": world,
                "car_x": arg("car_x"),
                "car_y": arg("car_y"),
                "car_yaw": arg("car_yaw"),
            }.items(),
        ),
        *([] if yolo_on_orin else [
            Node(
                package="odse_driving",
                executable="sim_target",
                parameters=[target, {"noise": float(arg("noise")), "dropout": float(arg("dropout"))}],
                output="screen",
            )
        ]),
        *([policy] if arg("policy") == "laptop" else []),
        Node(
            package="odse_driving",
            executable="sim_adapter",
            parameters=[{"use_sim_time": True}],
            output="screen",
        ),
        Node(
            package="odse_driving",
            executable="episode_monitor",
            parameters=[target],
            output="screen",
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("model", default_value="ppo", description="ppo, ppo_search, bc, ppo_h4, ppo_search_h4, or an ONNX file name in models/"),
            DeclareLaunchArgument("world", default_value="room", description="room (with other objects) or arena (empty, 6 x 6 m)"),
            DeclareLaunchArgument("target_x", default_value="1.8", description="ball position, m"),
            DeclareLaunchArgument("target_y", default_value="-0.3", description="ball position, m"),
            DeclareLaunchArgument("car_x", default_value="0.0", description="car start position, m"),
            DeclareLaunchArgument("car_y", default_value="0.0", description="car start position, m"),
            DeclareLaunchArgument("car_yaw", default_value="0.0", description="car start heading, rad (0 = +x)"),
            DeclareLaunchArgument("gui", default_value="true"),
            DeclareLaunchArgument("rviz", default_value="false"),
            DeclareLaunchArgument("cameras", default_value="false", description="rendered cameras (slow)"),
            DeclareLaunchArgument("noise", default_value="0.0", description="detection noise, 1.0 = training level"),
            DeclareLaunchArgument("dropout", default_value="0.0", description="share of missed detections"),
            DeclareLaunchArgument("venv", default_value="~/.venvs/odse_ros"),
            DeclareLaunchArgument(
                "policy", default_value="laptop", choices=["laptop", "orin"], description="where the network runs"
            ),
            DeclareLaunchArgument(
                "target", default_value="sim", choices=["sim", "orin"],
                description="sim: sim_target computes /target from true positions; orin: YOLO on the Orin sees the camera",
            ),
            DeclareLaunchArgument(
                "start_delay", default_value="5", description="simulated seconds before the car starts"
            ),
            OpaqueFunction(function=nodes),
        ]
    )
