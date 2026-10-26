from glob import glob

from setuptools import find_packages, setup

package_name = "odse_driving"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/models", glob("models/*.onnx")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="fisch",
    maintainer_email="mrfisch27@gmail.com",
    description="Driving policy nodes and the Gazebo driving experiment.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "sim_target = odse_driving.sim_target_node:main",
            "policy = odse_driving.policy_node:main",
            "sim_adapter = odse_driving.sim_adapter_node:main",
            "episode_monitor = odse_driving.episode_monitor:main",
            "yolo = odse_driving.yolo_node:main",
            "target = odse_driving.target_node:main",
        ],
    },
)
