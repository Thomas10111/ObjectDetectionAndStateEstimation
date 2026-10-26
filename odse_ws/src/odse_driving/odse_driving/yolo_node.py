"""YOLO detector as a ROS 2 node (M2 of docs/interfaces.md).

Runs YOLOv8n (ONNX, yolo.py) on every color image it gets and publishes all
boxes on /detections. The subscription keeps only the newest image, so when
YOLO is slower than the camera, frames are skipped rather than queued and the
detections never fall behind.

Every 5 s it logs the detection rate, how long YOLO took (wall clock) and how
old the image was when its detections went out (simulated time in Gazebo).
"""

import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from odse_driving.spin import spin
from odse_driving.yolo import Yolo


class YoloNode(Node):
    def __init__(self) -> None:
        super().__init__("yolo")
        model = self.declare_parameter("model", "").value
        # 0.25 rather than the 0.5 of robot_detection.py: the Gazebo ball often
        # scores between the two (section 9.5 of docs/how_it_works.md)
        self.conf = self.declare_parameter("conf", 0.25).value
        threads = self.declare_parameter("threads", 0).value
        self.yolo = Yolo(model, threads)
        self.pub = self.create_publisher(Detection2DArray, "/detections", 10)
        self.create_subscription(Image, "/camera/color/image_raw", self.on_image, qos_profile_sensor_data)
        self.inference_times: list[float] = []  # ms, wall clock
        self.image_ages: list[float] = []  # ms, simulated time
        self.frames = 0
        self.last_log = time.monotonic()
        self.get_logger().info(
            f"model {model}, input {self.yolo.input_w} x {self.yolo.input_h}, confidence {self.conf}"
        )

    def on_image(self, msg: Image) -> None:
        if msg.encoding not in ("rgb8", "bgr8"):
            self.get_logger().error(f"unsupported encoding {msg.encoding}", throttle_duration_sec=5.0)
            return
        image = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, : msg.width * 3]
        image = image.reshape(msg.height, msg.width, 3)
        if msg.encoding == "bgr8":
            image = image[:, :, ::-1]
        start = time.perf_counter()
        boxes = self.yolo.detect(image, conf=self.conf)
        self.inference_times.append((time.perf_counter() - start) * 1e3)

        out = Detection2DArray()
        out.header = msg.header
        for b in boxes:
            det = Detection2D()
            det.header = msg.header
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = b.class_id
            hypothesis.hypothesis.score = b.score
            det.results.append(hypothesis)
            det.bbox.center.position.x, det.bbox.center.position.y = b.u, b.v
            det.bbox.size_x, det.bbox.size_y = b.width, b.height
            out.detections.append(det)
        self.pub.publish(out)
        self.image_ages.append((self.get_clock().now() - Time.from_msg(msg.header.stamp)).nanoseconds / 1e6)
        self.frames += 1
        self.log_timing(boxes)

    def log_timing(self, boxes) -> None:
        now = time.monotonic()
        if now - self.last_log < 5.0:
            return
        inference, ages = np.array(self.inference_times), np.array(self.image_ages)
        found = ", ".join(f"{b.class_id} {b.score:.2f}" for b in boxes[:3]) or "nothing"
        self.get_logger().info(
            f"{self.frames / (now - self.last_log):.1f} frames/s | YOLO {np.median(inference):.0f} ms median, "
            f"{inference.max():.0f} ms max | image age {np.median(ages):.0f} ms median (simulated time) | last: {found}"
        )
        self.inference_times.clear()
        self.image_ages.clear()
        self.frames = 0
        self.last_log = now


def main() -> None:
    rclpy.init()
    spin(YoloNode())


if __name__ == "__main__":
    main()
