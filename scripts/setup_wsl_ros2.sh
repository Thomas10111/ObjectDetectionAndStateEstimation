#!/usr/bin/env bash
# Install ROS 2 Lyrical, Gazebo (via ros_gz) and Nav2 on Ubuntu 26.04 in WSL2.
#
# This matches the ros:lyrical container on the Jetson Orin, so both machines
# speak the same ROS 2 version. Run it inside the Ubuntu-26.04 WSL distribution
# as your normal user; it asks for your sudo password once:
#
#   bash /mnt/c/Users/fisch/MyGit/ObjectDetectionAndStateEstimation/scripts/setup_wsl_ros2.sh
#
# Safe to run again: steps that are already done are skipped or repeated
# harmlessly.

set -euo pipefail

ROS_DISTRO=lyrical
EXPECTED_CODENAME=resolute  # Ubuntu 26.04

# Required: the script fails if any of these cannot be installed.
CORE_PACKAGES=(
    "ros-${ROS_DISTRO}-desktop"  # ROS 2, RViz, demos
    "ros-${ROS_DISTRO}-ros-gz"   # Gazebo plus the ROS 2 bridge
    ros-dev-tools                # colcon, rosdep, vcstool
)

# Wanted, but a new distro may not have all of them yet. Missing ones are
# reported at the end instead of stopping the install.
OPTIONAL_PACKAGES=(
    "ros-${ROS_DISTRO}-navigation2"
    "ros-${ROS_DISTRO}-nav2-bringup"
    "ros-${ROS_DISTRO}-xacro"
    "ros-${ROS_DISTRO}-ackermann-msgs"
    "ros-${ROS_DISTRO}-teleop-twist-keyboard"
    "ros-${ROS_DISTRO}-rmw-zenoh-cpp"
)

step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

# 1. Sanity checks

. /etc/os-release
if [[ "${VERSION_CODENAME:-}" != "$EXPECTED_CODENAME" ]]; then
    echo "This script targets Ubuntu 26.04 ($EXPECTED_CODENAME), found ${PRETTY_NAME}." >&2
    exit 1
fi
if [[ $EUID -eq 0 ]]; then
    echo "Run as your normal user, not root; the script uses sudo where needed." >&2
    exit 1
fi
sudo -v  # Ask for the password once, up front.

# 2. Locale with UTF-8, required by ROS 2

step "Locale"
if ! locale | grep -qi 'utf-8'; then
    sudo apt-get update
    sudo apt-get install -y locales
    sudo locale-gen en_US en_US.UTF-8
    sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
    export LANG=en_US.UTF-8
fi

# 3. ROS 2 apt repository (official ros-apt-source package)

step "ROS 2 apt repository"
sudo apt-get update
sudo apt-get install -y software-properties-common curl
sudo add-apt-repository -y universe
if ! dpkg -s ros2-apt-source > /dev/null 2>&1; then
    version=$(curl -fsSL https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
        | grep -F '"tag_name"' | awk -F'"' '{print $4}')
    deb="/tmp/ros2-apt-source.deb"
    curl -fL -o "$deb" \
        "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${version}/ros2-apt-source_${version}.${VERSION_CODENAME}_all.deb"
    sudo dpkg -i "$deb"
    rm -f "$deb"
fi
sudo apt-get update
sudo apt-get upgrade -y

# 4. Packages

step "Core packages: ${CORE_PACKAGES[*]}"
sudo apt-get install -y "${CORE_PACKAGES[@]}"

step "Optional packages"
missing=()
for pkg in "${OPTIONAL_PACKAGES[@]}"; do
    if apt-cache show "$pkg" > /dev/null 2>&1; then
        sudo apt-get install -y "$pkg"
    else
        missing+=("$pkg")
    fi
done

# 5. rosdep (resolves dependencies of packages built from source)

step "rosdep"
if [[ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]]; then
    sudo rosdep init
fi
rosdep update

# 6. Shell setup

step "Shell setup"
line="source /opt/ros/${ROS_DISTRO}/setup.bash"
if ! grep -qxF "$line" ~/.bashrc; then
    printf '\n# ROS 2\n%s\n' "$line" >> ~/.bashrc
fi

# 7. Summary

set +u  # ROS setup scripts read unset variables.
# shellcheck disable=SC1090
source "/opt/ros/${ROS_DISTRO}/setup.bash"
set -u

step "Done"
echo "ROS_DISTRO: ${ROS_DISTRO}"
printf 'Gazebo:     '
gz sim --versions 2>/dev/null || echo "not found on PATH (try: ros2 run ros_gz_sim gz_sim)"
if ((${#missing[@]})); then
    echo
    echo "Not yet available for ${ROS_DISTRO}, install later or build from source:"
    printf '  %s\n' "${missing[@]}"
fi
echo
echo "Open a new terminal (or run 'source ~/.bashrc'), then try:"
echo "  ros2 run demo_nodes_cpp talker"
echo "  ros2 launch ros_gz_sim gz_sim.launch.py gz_args:=shapes.sdf"
