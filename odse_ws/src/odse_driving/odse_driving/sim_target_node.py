"""Stand in for camera, YOLO and depth lookup in Gazebo (M3 of docs/interfaces.md).

Takes the true pose of the car (/ground_truth/odom) and of the target object
(/world_poses) from Gazebo and publishes what the real chain would report on
/target, using the same camera geometry as the training simulator
(camera_model.py). Noise and missed detections can be switched on to match
training.
"""

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_msgs.msg import TFMessage

from odse_driving.spin import spin
from odse_driving.camera_model import observe, yaw_from_quaternion
from odse_interfaces.msg import TargetObservation


class SimTarget(Node):
    def __init__(self) -> None:
        super().__init__("sim_target")
        self.target_model = self.declare_parameter("target_model", "target_ball").value
        self.radius = self.declare_parameter("target_radius", 0.1).value
        self.class_id = self.declare_parameter("class_id", "sports ball").value
        self.rate = self.declare_parameter("rate", 30.0).value  # like the camera
        self.noise = self.declare_parameter("noise", 0.0).value  # 1.0 = training level
        self.dropout = self.declare_parameter("dropout", 0.0).value
        self.rng = np.random.default_rng(0)
        self.target = None
        self.pub = self.create_publisher(TargetObservation, "/target", 10)
        self.create_subscription(TFMessage, "/world_poses", self.on_poses, 10)
        self.create_subscription(Odometry, "/ground_truth/odom", self.on_car, 10)
        self.create_timer(1.0 / self.rate, self.publish)
        self.car = None
        self.get_logger().info(
            f"target '{self.target_model}' (radius {self.radius} m), noise {self.noise}, dropout {self.dropout}"
        )

    def on_poses(self, msg: TFMessage) -> None:
        for t in msg.transforms:
            if t.child_frame_id == self.target_model:
                self.target = (t.transform.translation.x, t.transform.translation.y)

    def on_car(self, msg: Odometry) -> None:
        p = msg.pose.pose
        self.car = (p.position.x, p.position.y, yaw_from_quaternion(p.orientation))

    def publish(self) -> None:
        if self.car is None or self.target is None:
            return
        det = observe(self.car, self.target, self.radius, self.rng, noise=self.noise, dropout=self.dropout)
        out = TargetObservation()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = "camera_color_optical_frame"
        out.visible, out.z_valid = det.visible, det.z_valid
        out.x, out.y, out.w, out.h, out.z = det.x, det.y, det.w, det.h, det.z
        out.score = 1.0 if det.visible else 0.0
        out.class_id = self.class_id if det.visible else ""
        self.pub.publish(out)


def main() -> None:
    rclpy.init()
    spin(SimTarget())


if __name__ == "__main__":
    main()
