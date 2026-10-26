"""Target observation from YOLO boxes and the depth image (M3 of docs/interfaces.md).

For every /detections message: pick the box of target_class with the highest
score, normalize it with the color image size, look up the depth in the
central part of the box and publish /target, the observation the policy was
trained on. Replaces sim_target when the real chain runs.

Depth: the box center region is mapped from the color into the depth image
with both camera infos. That is exact in Gazebo, where both cameras sit at
the same point; the real F200 also needs the few cm between its two cameras
(not done yet). Accepts 32FC1 depth in m (Gazebo) and 16UC1 in mm (F200).

Depth hold: YOLO loses the ball when it is close and fills the image
(below about 0.6 m, section 9.5 of docs/how_it_works.md), exactly where the
car has to stop. If the target was seen with a valid depth (so closer than
1 m) in the last hold_for seconds and YOLO misses it now, the node measures the
depth again where the box was. If that depth is still plausible (moved less
than MAX_HOLD_JUMP), it reports the target as visible with the box scaled to
the new distance (pinhole: offsets from the image center and sizes grow
with 1 / distance) and score 0, which marks a held observation. Every such
depth reading renews the hold, so a car stopped in front of the ball keeps
seeing it; the hold ends when the depth jumps (the ball is gone) or no
reading succeeds for hold_for seconds.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import Detection2DArray

from odse_driving.camera_model import DEPTH_MAX, DEPTH_MIN
from odse_driving.spin import spin
from odse_interfaces.msg import TargetObservation

CENTER_SHARE = 0.2  # side length of the depth window, as a share of the box
MAX_DEPTH_LAG = 0.2  # s, depth image at most this much older than the color image
MAX_HOLD_JUMP = 0.15  # m, larger depth changes between frames end the hold


class Target(Node):
    def __init__(self) -> None:
        super().__init__("target")
        self.target_class = self.declare_parameter("target_class", "sports ball").value
        self.hold_for = self.declare_parameter("hold_for", 1.0).value  # s, 0 switches the hold off
        self.last: tuple[float, float, float, float, float, Time] | None = None  # u, v, box w, h, z, stamp
        self.color_info: CameraInfo | None = None
        self.depth_info: CameraInfo | None = None
        self.depth: Image | None = None
        self.pub = self.create_publisher(TargetObservation, "/target", 10)
        self.create_subscription(Detection2DArray, "/detections", self.on_detections, 10)
        self.create_subscription(Image, "/camera/depth/image_rect_raw", self.on_depth, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, "/camera/color/camera_info", self.on_color_info, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, "/camera/depth/camera_info", self.on_depth_info, qos_profile_sensor_data)
        self.get_logger().info(
            f"target class '{self.target_class}', depth hold {self.hold_for} s"
        )

    def on_depth(self, msg: Image) -> None:
        self.depth = msg

    def on_color_info(self, msg: CameraInfo) -> None:
        self.color_info = msg

    def on_depth_info(self, msg: CameraInfo) -> None:
        self.depth_info = msg

    def on_detections(self, msg: Detection2DArray) -> None:
        out = TargetObservation()
        out.header = msg.header
        if self.color_info is None:
            self.get_logger().warn("no /camera/color/camera_info yet", throttle_duration_sec=5.0)
            return
        stamp = Time.from_msg(msg.header.stamp)
        candidates = [d for d in msg.detections if d.results and d.results[0].hypothesis.class_id == self.target_class]
        if candidates:
            best = max(candidates, key=lambda d: d.results[0].hypothesis.score)
            u, v = best.bbox.center.position.x, best.bbox.center.position.y
            box_w, box_h = best.bbox.size_x, best.bbox.size_y
            z = self.depth_at(u, v, box_w, box_h, stamp)
            z = z if z is not None and DEPTH_MIN <= z <= DEPTH_MAX else None
            self.fill(out, u, v, box_w, box_h, z, best.results[0].hypothesis.score)
            self.last = (u, v, box_w, box_h, z, stamp) if z is not None else None
        elif (held := self.hold(stamp)) is not None:
            self.fill(out, *held[:5], score=0.0)
            self.last = held
        else:
            self.last = None
        self.pub.publish(out)

    def fill(self, out: TargetObservation, u: float, v: float, box_w: float, box_h: float,
             z: float | None, score: float) -> None:
        W, H = self.color_info.width, self.color_info.height
        out.visible = True
        out.x, out.y = float(np.clip(2 * u / W - 1, -1, 1)), float(np.clip(2 * v / H - 1, -1, 1))
        out.w, out.h = min(box_w / W, 1.0), min(box_h / H, 1.0)
        out.score = score
        out.class_id = self.target_class
        if z is not None:
            out.z_valid, out.z = True, z

    def hold(self, stamp: Time) -> tuple[float, float, float, float, float, Time] | None:
        """The last box moved to the depth measured there now, or None (see module docstring)."""
        if self.last is None or not self.hold_for:
            return None
        u, v, box_w, box_h, z_last, seen = self.last
        if (stamp - seen).nanoseconds > self.hold_for * 1e9:
            return None
        z = self.depth_at(u, v, box_w, box_h, stamp)
        if z is None or not DEPTH_MIN <= z <= DEPTH_MAX or abs(z - z_last) > MAX_HOLD_JUMP:
            return None
        scale = z_last / z
        cx, cy = self.color_info.k[2], self.color_info.k[5]
        return cx + (u - cx) * scale, cy + (v - cy) * scale, box_w * scale, box_h * scale, z, stamp

    def depth_at(self, u: float, v: float, box_w: float, box_h: float, stamp: Time) -> float | None:
        """Median depth in m of the box center region, or None."""
        d, ci, di = self.depth, self.color_info, self.depth_info
        if d is None or di is None:
            return None
        if abs((stamp - Time.from_msg(d.header.stamp)).nanoseconds) > MAX_DEPTH_LAG * 1e9:
            return None
        # color pixel -> ray -> depth pixel (shared origin, see module docstring)
        fx_c, fy_c, cx_c, cy_c = ci.k[0], ci.k[4], ci.k[2], ci.k[5]
        fx_d, fy_d, cx_d, cy_d = di.k[0], di.k[4], di.k[2], di.k[5]
        half_w, half_h = CENTER_SHARE * box_w / 2, CENTER_SHARE * box_h / 2
        u0 = int(np.floor(fx_d * (u - half_w - cx_c) / fx_c + cx_d))
        u1 = int(np.ceil(fx_d * (u + half_w - cx_c) / fx_c + cx_d))
        v0 = int(np.floor(fy_d * (v - half_h - cy_c) / fy_c + cy_d))
        v1 = int(np.ceil(fy_d * (v + half_h - cy_c) / fy_c + cy_d))
        u0, u1 = max(u0, 0), min(max(u1, u0 + 1), d.width)
        v0, v1 = max(v0, 0), min(max(v1, v0 + 1), d.height)
        if u0 >= u1 or v0 >= v1:
            return None
        if d.encoding == "32FC1":
            image, scale = np.frombuffer(d.data, np.float32).reshape(d.height, d.step // 4), 1.0
        elif d.encoding == "16UC1":
            image, scale = np.frombuffer(d.data, np.uint16).reshape(d.height, d.step // 2), 0.001
        else:
            self.get_logger().error(f"unsupported depth encoding {d.encoding}", throttle_duration_sec=5.0)
            return None
        window = image[v0:v1, u0:u1].astype(np.float32) * scale
        valid = window[np.isfinite(window) & (window > 0)]
        return float(np.median(valid)) if valid.size else None


def main() -> None:
    rclpy.init()
    spin(Target())


if __name__ == "__main__":
    main()
