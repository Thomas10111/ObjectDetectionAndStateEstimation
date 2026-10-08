#!/usr/bin/env bash
# Image transport benchmark between the laptop and the Orin: publisher in WSL,
# subscriber in the ros2_lyrical container. Run in WSL:
#
#   run_cross.sh [all|zenoh|reliable]      LINK=usb run_cross.sh ...
#
# all       Fast DDS, Fast DDS LARGE_DATA and Zenoh, three message sizes
# zenoh     Zenoh only, three sizes plus 6.2 MB with reliable QoS
# reliable  6.2 MB with reliable QoS, Fast DDS and Zenoh
#
# LINK=ethernet (default) or usb. Fast DDS needs the WSL firewall rule
# ROS2-Orin-USB, which only covers the USB network, so over Ethernet use
# "zenoh". Zenoh uses the routers that run permanently on both sides (WSL
# service zenohd, container entrypoint); the WSL router must point at the
# chosen link (scripts/wsl/zenohd.service).
#
# Needs the bench scripts in /root/bench in the container. Uses the Windows
# ssh.exe, which holds the key for the Orin.
source /opt/ros/lyrical/setup.bash
B="$(cd "$(dirname "$0")" && pwd)"
MODE=${1:-all}
case ${LINK:-ethernet} in
    ethernet) ORIN_IP=192.168.56.1 LAPTOP_IP=192.168.56.100 ;;
    usb) ORIN_IP=192.168.55.1 LAPTOP_IP=192.168.55.100 ;;
    *) echo "LINK must be ethernet or usb" >&2; exit 1 ;;
esac
SSH=(/mnt/c/Windows/System32/OpenSSH/ssh.exe -o BatchMode=yes -o ConnectionAttempts=5 "fish@$ORIN_IP")
DOMAIN=78
SECONDS_PER_RUN=15

orin() { "${SSH[@]}" "$@" 2>&1 | tr -d '\r'; }

run() {  # label, size, hz, then extra env for both sides as NAME=VALUE
    local label=$1 size=$2 hz=$3
    shift 3
    local docker_env=() kv
    for kv in "$@"; do docker_env+=(-e "$kv"); done
    orin docker exec -d -e ROS_DOMAIN_ID=$DOMAIN -e ROS_STATIC_PEERS=$LAPTOP_IP "${docker_env[@]}" ros2_lyrical \
        bash -c "'source /opt/ros/lyrical/setup.bash; python3 /root/bench/bench_sub.py $((SECONDS_PER_RUN + 12)) > /tmp/sub.out 2>&1'"
    sleep 4
    local pub
    pub=$(env ROS_DOMAIN_ID=$DOMAIN ROS_STATIC_PEERS=$ORIN_IP "$@" \
        python3 "$B/bench_pub.py" "$size" "$hz" $SECONDS_PER_RUN 2>&1 | grep PUB | cut -d' ' -f4-)
    sleep 8
    echo "$label size=$size hz=$hz | ${pub:-no publisher output} | $(orin docker exec ros2_lyrical cat /tmp/sub.out | cut -d' ' -f2-)"
}

cases() {  # label, then env
    local label=$1
    shift
    run "$label" 614400 30 "$@"
    run "$label" 1228800 30 "$@"
    run "$label" 6220800 5 "$@"
}

check_zenoh() {
    systemctl --user is-active --quiet zenohd || { echo "WSL router zenohd not running" >&2; exit 1; }
    orin docker exec ros2_lyrical pgrep -f rmw_zenohd > /dev/null || { echo "Orin router not running" >&2; exit 1; }
}

FASTDDS=RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ZENOH=RMW_IMPLEMENTATION=rmw_zenoh_cpp
echo "link: ${LINK:-ethernet} (Orin $ORIN_IP)"

if [[ $MODE == all ]]; then
    echo "=== Fast DDS, default"
    cases fastdds $FASTDDS
    echo "=== Fast DDS, LARGE_DATA"
    cases fastdds_large $FASTDDS FASTDDS_BUILTIN_TRANSPORTS=LARGE_DATA
fi
if [[ $MODE == all || $MODE == zenoh ]]; then
    check_zenoh
    echo "=== Zenoh"
    cases zenoh $ZENOH
fi
if [[ $MODE == zenoh || $MODE == reliable ]]; then
    echo "=== 6.2 MB at 5 Hz, reliable QoS"
    [[ $MODE == reliable ]] && run fastdds_reliable 6220800 5 $FASTDDS BENCH_RELIABLE=1
    check_zenoh
    run zenoh_reliable 6220800 5 $ZENOH BENCH_RELIABLE=1
fi
echo "=== done"
