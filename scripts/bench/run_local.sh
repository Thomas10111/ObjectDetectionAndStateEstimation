#!/usr/bin/env bash
# Image transport benchmark inside the ros2_lyrical container: publisher and
# subscriber on the same Orin, Fast DDS versus Zenoh. Prints one line per run.
source /opt/ros/lyrical/setup.bash
unset ROS_STATIC_PEERS
export ROS_DOMAIN_ID=77
B=/root/bench
SECONDS_PER_RUN=15

cpu_ticks() { awk '{print $14 + $15}' "/proc/$1/stat" 2>/dev/null || echo 0; }

run() {  # label, size
    python3 $B/bench_sub.py $((SECONDS_PER_RUN + 8)) > /tmp/sub.out 2>&1 &
    SUB=$!
    sleep 2
    python3 $B/bench_pub.py "$2" 30 $SECONDS_PER_RUN > /tmp/pub.out 2>&1
    wait $SUB
    echo "$1 size=$2 | $(grep PUB /tmp/pub.out | cut -d' ' -f4-) | $(grep SUB /tmp/sub.out | cut -d' ' -f2-)"
}

SIZES="614400 1228800 6220800"  # F200 depth 16 bit, sim depth float, F200 color 1080p RGB

echo "=== Fast DDS (default)"
unset RMW_IMPLEMENTATION
for s in $SIZES; do run fastdds "$s"; done

echo "=== Zenoh"
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
ros2 run rmw_zenoh_cpp rmw_zenohd > /tmp/zenohd.log 2>&1 &
ZD=$!
sleep 3
t0=$(cpu_ticks "$(pgrep -f '[r]mw_zenohd' | tail -1)")
for s in $SIZES; do run zenoh "$s"; done
t1=$(cpu_ticks "$(pgrep -f '[r]mw_zenohd' | tail -1)")
echo "zenoh router cpu during the runs: $(( (t1 - t0) )) ticks (100 ticks = 1 cpu second)"

echo "=== Zenoh with shared memory"
export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=true'
for s in $SIZES; do run zenoh_shm "$s"; done
kill $ZD
echo "=== done"
