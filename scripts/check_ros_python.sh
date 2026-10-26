#!/usr/bin/env bash
# Check which inference packages exist for the Python that ROS 2 Lyrical uses
# (Ubuntu 26.04 system Python). ROS nodes must run on that Python, because
# rclpy is built for it; the training environment (Python 3.12) cannot import rclpy.
export PATH="$HOME/.local/bin:$PATH"
source /opt/ros/lyrical/setup.bash
py=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "ROS Python: $py"
python3 -c 'import rclpy, cv_bridge, numpy; print("rclpy, cv_bridge, numpy: ok")' 2>&1 | tail -1
for p in onnxruntime openvino ultralytics opencv-python-headless; do
    printf '%-24s ' "$p"
    echo "$p" | uv pip compile - --python-version "$py" --python-platform x86_64-manylinux_2_28 \
        --no-header --no-annotate -q 2>&1 | grep -iE "^$p==|error|no solution" | head -1
done
