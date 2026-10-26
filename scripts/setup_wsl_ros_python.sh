#!/usr/bin/env bash
# Python environment for ROS 2 nodes that need extra packages (the policy node
# needs onnxruntime). Run in WSL, no sudo needed:
#
#   bash /mnt/c/Users/fisch/MyGit/ObjectDetectionAndStateEstimation/scripts/setup_wsl_ros_python.sh
#
# It uses the system Python that ROS 2 Lyrical is built for (3.14), with
# access to the system packages, so rclpy and the message packages still
# import. The training environment (~/.venvs/odse_policy, Python 3.12) cannot
# load ROS 2; the ONNX file connects the two.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
VENV=~/.venvs/odse_ros

[[ -d $VENV ]] || uv venv --python /usr/bin/python3 --system-site-packages "$VENV"
uv pip install --python "$VENV/bin/python" onnxruntime

set +u  # the ROS setup script reads unset variables
source /opt/ros/lyrical/setup.bash
"$VENV/bin/python" -c 'import sys, rclpy, onnxruntime; print(f"Python {sys.version.split()[0]}, rclpy ok, onnxruntime {onnxruntime.__version__}")'
