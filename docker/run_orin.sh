#!/usr/bin/env bash
# Build the image and (re)create the ros2_lyrical container on the Orin.
#
#   bash run_orin.sh            # in the docker/ folder on the Orin
#
# Removes an existing ros2_lyrical container first. Files in ~/ros2_ws stay;
# anything installed by hand inside the old container is lost.
set -euo pipefail
cd "$(dirname "$0")"

IMAGE=odse/ros:lyrical-zenoh
NAME=ros2_lyrical

docker build -t "$IMAGE" .
docker rm -f "$NAME" 2> /dev/null || true

# --ipc=host: shared memory for Zenoh without Docker's 64 MB /dev/shm limit.
# --restart unless-stopped: start again after a reboot, until stopped by hand.
docker run -dit \
    --name "$NAME" \
    --restart unless-stopped \
    --runtime=nvidia \
    --network=host \
    --ipc=host \
    --privileged \
    -e DISPLAY=:0 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /home/fish/ros2_ws:/root/ros2_ws \
    "$IMAGE" \
    bash

sleep 3
docker exec "$NAME" bash -c 'pgrep -f rmw_zenohd > /dev/null && echo "Zenoh router running" || echo "Zenoh router NOT running, see /tmp/zenohd.log"'
