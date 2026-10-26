#!/usr/bin/env bash
# Check that the ROS 2 message packages used by docs/interfaces.md are installed.
source /opt/ros/lyrical/setup.bash
for p in vision_msgs ackermann_msgs cv_bridge sensor_msgs geometry_msgs; do
    if ros2 pkg prefix "$p" > /dev/null 2>&1; then
        echo "$p: installed"
    else
        echo "$p: MISSING ($(apt-cache policy "ros-lyrical-${p//_/-}" 2>/dev/null | grep -m1 Candidate || echo 'no apt info'))"
    fi
done
