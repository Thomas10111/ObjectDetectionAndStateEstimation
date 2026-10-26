"""The driving policy alone, for the Orin while Gazebo runs on the laptop.

    ros2 launch odse_driving policy.launch.py                      # PPO, ball ahead
    ros2 launch odse_driving policy.launch.py model:=ppo_search
    ros2 launch odse_driving policy.launch.py target:=yolo         # also YOLO and the target node

The laptop starts everything else with experiment.launch.py policy:=orin
(and target:=orin with target:=yolo here, so Gazebo renders the camera and
sim_target stays off).
The node follows Gazebo's /clock over Zenoh and can stay running across
experiments: a restarted simulation starts a new episode. They run with the
system Python, which has onnxruntime and OpenCV in the Orin container.
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
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


def nodes(context):
    def arg(name: str) -> str:
        return LaunchConfiguration(name).perform(context)

    share = Path(get_package_share_directory("odse_driving"))
    model_path = str(share / "models" / MODELS.get(arg("model"), arg("model")))
    perception = [
        Node(
            package="odse_driving",
            executable="yolo",
            parameters=[{"model": str(share / "models" / arg("yolo_model")), "use_sim_time": True}],
            output="screen",
        ),
        Node(
            package="odse_driving",
            executable="target",
            parameters=[{"target_class": arg("target_class"), "use_sim_time": True}],
            output="screen",
        ),
    ]
    return [
        *(perception if arg("target") == "yolo" else []),
        Node(
            package="odse_driving",
            executable="policy",
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
    ]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("model", default_value="ppo", description="ppo, ppo_search, bc, ppo_h4, ppo_search_h4, or an ONNX file name in models/"),
            DeclareLaunchArgument(
                "start_delay", default_value="5", description="simulated seconds before the car starts"
            ),
            DeclareLaunchArgument(
                "target", default_value="sim", choices=["sim", "yolo"],
                description="sim: /target comes from sim_target on the laptop; yolo: from YOLO and the target node here",
            ),
            DeclareLaunchArgument("yolo_model", default_value="yolov8n_384x640.onnx", description="ONNX file in models/"),
            DeclareLaunchArgument("target_class", default_value="sports ball", description="COCO class to drive to"),
            OpaqueFunction(function=nodes),
        ]
    )
