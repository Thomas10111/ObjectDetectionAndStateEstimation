#!/usr/bin/env bash
# Make Zenoh the ROS 2 middleware in WSL. Run in WSL as your normal user (no sudo):
#
#   bash /mnt/c/Users/fisch/MyGit/ObjectDetectionAndStateEstimation/scripts/wsl/setup_zenoh.sh
#
# 1. ~/.bashrc: RMW_IMPLEMENTATION=rmw_zenoh_cpp, shared memory on, and the
#    old Fast DDS setting ROS_STATIC_PEERS removed.
# 2. A systemd user service runs the Zenoh router, connected to the Orin.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"

# 1. Shell setup
sed -i '/ROS_STATIC_PEERS/d' ~/.bashrc
if ! grep -q 'RMW_IMPLEMENTATION=rmw_zenoh_cpp' ~/.bashrc; then
    cat >> ~/.bashrc <<'EOF'
# ROS 2 middleware: Zenoh (docs/rmw_comparison.md). The router runs as the
# systemd user service zenohd: systemctl --user status zenohd
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=true'
EOF
fi

# 2. Router service
mkdir -p ~/.config/systemd/user
cp "$here/zenohd.service" ~/.config/systemd/user/zenohd.service
systemctl --user daemon-reload
systemctl --user enable --now zenohd.service
systemctl --user restart zenohd.service
sleep 3
systemctl --user --no-pager status zenohd.service | head -5
