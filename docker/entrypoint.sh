#!/usr/bin/env bash
# Start the Zenoh router in the background, then run the container command.
# ROS 2 nodes on this machine find each other (and the laptop) through it.
set -e
source /opt/ros/lyrical/setup.bash
if ! pgrep -f rmw_zenohd > /dev/null; then
    nohup ros2 run rmw_zenoh_cpp rmw_zenohd > /tmp/zenohd.log 2>&1 &
fi
exec "$@"
