"""Turn /drive/command into /cmd_vel for the Gazebo car (M5a of docs/interfaces.md).

Gazebo's Ackermann plugin takes a speed and a yaw rate; it turns the yaw rate
back into a steering angle. At speed 0 the steering command is lost (F14).

Every 50 commands it logs how old they were on arrival (simulated time). With
the policy on the Orin this is the delay of the way back over the link.
"""

import math
import statistics

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.time import Time

from odse_driving.spin import spin
from odse_driving.camera_model import WHEELBASE


class SimAdapter(Node):
    def __init__(self) -> None:
        super().__init__("sim_adapter")
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_subscription(AckermannDriveStamped, "/drive/command", self.on_command, 10)
        self.ages: list[float] = []  # ms, simulated time

    def on_command(self, msg: AckermannDriveStamped) -> None:
        twist = Twist()
        twist.linear.x = msg.drive.speed
        twist.angular.z = msg.drive.speed * math.tan(msg.drive.steering_angle) / WHEELBASE
        self.pub.publish(twist)

        age = (self.get_clock().now() - Time.from_msg(msg.header.stamp)).nanoseconds / 1e6
        if age < 0:  # a command from the previous run, stamped before a restart
            return
        self.ages.append(age)
        if len(self.ages) == 50:
            self.get_logger().info(
                f"command age {statistics.median(self.ages):.0f} ms median, {max(self.ages):.0f} ms max (simulated time)"
            )
            self.ages.clear()


def main() -> None:
    rclpy.init()
    spin(SimAdapter())


if __name__ == "__main__":
    main()
