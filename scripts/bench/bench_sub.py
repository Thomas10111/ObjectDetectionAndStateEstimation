"""Receive image sized messages, ack each one, report rate and CPU.

Usage: bench_sub.py SECONDS
"""

import os
import sys
import time

import rclpy
from rclpy.qos import qos_profile_sensor_data, qos_profile_system_default
from sensor_msgs.msg import Image
from std_msgs.msg import Header

seconds = float(sys.argv[1])

# BENCH_RELIABLE=1 switches from sensor data QoS (best effort) to reliable.
QOS = qos_profile_system_default if os.environ.get("BENCH_RELIABLE") == "1" else qos_profile_sensor_data

rclpy.init()
node = rclpy.create_node("bench_sub")
ack = node.create_publisher(Header, "/bench_ack", 50)
stats = {"n": 0, "bytes": 0, "first": None, "last": None, "cpu0": None}


def on_image(msg: Image) -> None:
    now = time.monotonic()
    if stats["first"] is None:
        stats["first"], stats["cpu0"] = now, sum(os.times()[:2])
    stats["n"] += 1
    stats["bytes"] += len(msg.data)
    stats["last"] = now
    ack.publish(msg.header)


node.create_subscription(Image, "/bench_image", on_image, QOS)
end = time.monotonic() + seconds
while time.monotonic() < end:
    rclpy.spin_once(node, timeout_sec=0.1)

if stats["n"] > 1:
    span = stats["last"] - stats["first"]
    cpu = sum(os.times()[:2]) - stats["cpu0"]
    print(
        f"SUB received={stats['n']} rate={(stats['n'] - 1) / span:.1f}Hz "
        f"throughput={stats['bytes'] / span / 1e6:.1f}MB/s cpu={100 * cpu / span:.0f}%"
    )
else:
    print(f"SUB received={stats['n']}")
node.destroy_node()
rclpy.shutdown()
