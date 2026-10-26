"""Watch a driving experiment in Gazebo: progress, outcome, and a CSV log.

Uses the same arrival and collision rules as the training simulator, so the
outcome means the same thing in both. Prints once per simulated second and
when the outcome is decided; logs every car pose update to
~/odse_experiments/<start time>.csv for plotting.
"""

import csv
import time
from pathlib import Path

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_msgs.msg import TFMessage

from odse_driving.spin import spin
from odse_driving.camera_model import arrived, collided, gap, yaw_from_quaternion

TIMEOUT = 40.0  # s, as MAX_STEPS in training


class EpisodeMonitor(Node):
    def __init__(self) -> None:
        super().__init__("episode_monitor")
        self.target_model = self.declare_parameter("target_model", "target_ball").value
        self.radius = self.declare_parameter("target_radius", 0.1).value
        self.target = None
        self.command = (0.0, 0.0)
        self.start = None
        self.last_print = -1.0
        self.outcome = None
        folder = Path.home() / "odse_experiments"
        folder.mkdir(exist_ok=True)
        self.csv_path = folder / f"{time.strftime('%Y%m%d_%H%M%S')}.csv"
        self.csv_file = self.csv_path.open("w", newline="")
        self.csv = csv.writer(self.csv_file)
        self.csv.writerow(["t", "x", "y", "yaw", "gap", "speed", "steering_angle", "speed_command"])
        self.create_subscription(TFMessage, "/world_poses", self.on_poses, 10)
        self.create_subscription(Odometry, "/ground_truth/odom", self.on_car, 10)
        self.create_subscription(AckermannDriveStamped, "/drive/command", self.on_command, 10)
        self.get_logger().info(f"logging to {self.csv_path}")

    def on_poses(self, msg: TFMessage) -> None:
        for t in msg.transforms:
            if t.child_frame_id == self.target_model:
                self.target = (t.transform.translation.x, t.transform.translation.y)

    def on_command(self, msg: AckermannDriveStamped) -> None:
        self.command = (msg.drive.steering_angle, msg.drive.speed)
        if self.start is None:  # the experiment starts with the first command
            self.start = self.get_clock().now()

    def on_car(self, msg: Odometry) -> None:
        if self.start is None or self.target is None:
            return
        t = (self.get_clock().now() - self.start).nanoseconds * 1e-9
        p = msg.pose.pose
        pose = (p.position.x, p.position.y, yaw_from_quaternion(p.orientation))
        speed = msg.twist.twist.linear.x
        g = gap(pose, self.target, self.radius)
        self.csv.writerow([f"{t:.3f}", f"{pose[0]:.4f}", f"{pose[1]:.4f}", f"{pose[2]:.4f}",
                           f"{g:.4f}", f"{speed:.4f}", f"{self.command[0]:.4f}", f"{self.command[1]:.4f}"])

        if self.outcome is None:
            if collided(pose, self.target, self.radius):
                self.outcome = "collision"
            elif arrived(pose, self.target, self.radius, speed):
                self.outcome = "arrived"
            elif t > TIMEOUT:
                self.outcome = "timeout"
            if self.outcome:
                self.csv_file.flush()
                self.get_logger().info(
                    f"OUTCOME: {self.outcome.upper()} after {t:.1f} s simulated time, gap {g:.2f} m "
                    f"(arrival means 0.30 +/- 0.15 m and stopped)"
                )
        if t - self.last_print >= 1.0:
            self.last_print = t
            self.get_logger().info(
                f"t {t:5.1f} s | gap to target {g:5.2f} m | speed {speed:5.2f} m/s | "
                f"steering {self.command[0]:+.2f} rad"
            )


def main() -> None:
    rclpy.init()
    node = EpisodeMonitor()
    try:
        spin(node)
    finally:
        node.csv_file.close()


if __name__ == "__main__":
    main()
