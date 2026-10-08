"""Publish image sized messages and measure round trip time via acks.

Usage: bench_pub.py SIZE_BYTES HZ SECONDS
"""

import array
import os
import statistics
import sys
import time

import rclpy
from rclpy.qos import qos_profile_sensor_data, qos_profile_system_default
from sensor_msgs.msg import Image
from std_msgs.msg import Header

size, hz, seconds = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])

# BENCH_RELIABLE=1 switches from sensor data QoS (best effort) to reliable.
QOS = qos_profile_system_default if os.environ.get("BENCH_RELIABLE") == "1" else qos_profile_sensor_data

rclpy.init()
node = rclpy.create_node("bench_pub")
pub = node.create_publisher(Image, "/bench_image", QOS)
rtts = []


def on_ack(msg: Header) -> None:
    sent = msg.stamp.sec + msg.stamp.nanosec * 1e-9
    rtts.append(time.time() - sent)


node.create_subscription(Header, "/bench_ack", on_ack, 50)

msg = Image()
msg.height, msg.width, msg.encoding = 1, size, "mono8"
msg.step = size
msg.data = array.array("B", bytes(size))

# Discovery
end = time.monotonic() + 3
while time.monotonic() < end:
    rclpy.spin_once(node, timeout_sec=0.05)

sent = 0
cpu0, wall0 = sum(os.times()[:2]), time.monotonic()
period = 1.0 / hz
next_t = wall0
while time.monotonic() - wall0 < seconds:
    now = time.time()
    msg.header.stamp.sec = int(now)
    msg.header.stamp.nanosec = int((now % 1) * 1e9)
    msg.header.frame_id = str(sent)
    pub.publish(msg)
    sent += 1
    next_t += period
    while True:
        remaining = next_t - time.monotonic()
        if remaining <= 0:
            break
        rclpy.spin_once(node, timeout_sec=remaining)
wall = time.monotonic() - wall0
cpu = sum(os.times()[:2]) - cpu0
end = time.monotonic() + 2
while time.monotonic() < end:
    rclpy.spin_once(node, timeout_sec=0.05)

rtt_ms = sorted(r * 1000 for r in rtts)
p95 = rtt_ms[int(len(rtt_ms) * 0.95)] if rtt_ms else float("nan")
med = statistics.median(rtt_ms) if rtt_ms else float("nan")
print(
    f"PUB size={size} target={hz:.0f}Hz sent={sent} rate={sent / wall:.1f}Hz "
    f"acks={len(rtts)} delivered={100 * len(rtts) / max(sent, 1):.0f}% "
    f"rtt_median={med:.1f}ms rtt_p95={p95:.1f}ms cpu={100 * cpu / wall:.0f}%"
)
node.destroy_node()
rclpy.shutdown()
