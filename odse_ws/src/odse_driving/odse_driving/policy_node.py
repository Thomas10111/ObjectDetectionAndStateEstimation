"""The driving policy as a ROS 2 node (M4 of docs/interfaces.md).

Every 0.1 s: build the 9 number observation from the latest /target and the
previous command, run the ONNX network, apply the speed limit and publish
/drive/command (steering angle in rad, positive left; speed in m/s).

Needs onnxruntime: on the laptop it runs with the Python of ~/.venvs/odse_ros
(see the package README), on the Orin with the system Python of the container.

The node may run on the Orin while Gazebo runs on the laptop, and stay up
across several experiments. When the simulated time jumps back (Gazebo was
restarted), it forgets the old target and its previous command and waits
start_delay again. Every 5 s it logs how old the target was when used (time
from sim_target to the network, in simulated time) and how long the network
took (wall clock).
"""

import time

import numpy as np
import onnxruntime as ort
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.node import Node
from rclpy.time import Time

from odse_driving.spin import spin
from odse_driving.camera_model import MAX_STEER
from odse_interfaces.msg import TargetObservation


class Policy(Node):
    def __init__(self) -> None:
        super().__init__("policy")
        model = self.declare_parameter("model", "").value
        self.rate = self.declare_parameter("rate", 10.0).value
        self.speed_limit = self.declare_parameter("speed_limit", 0.4).value
        self.speed_per_command = self.declare_parameter("speed_per_command", 1.0).value  # m/s
        self.stale_after = self.declare_parameter("stale_after", 0.5).value  # s
        # Wait this long (simulated time) before driving, to have time to watch
        self.start_delay = self.declare_parameter("start_delay", 0.0).value  # s
        self.start_time = None
        self.last_now = None
        self.target_ages: list[float] = []  # ms, simulated time
        self.inference_times: list[float] = []  # ms, wall clock
        self.session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
        self.prev = np.zeros(2, dtype=np.float32)
        self.target: TargetObservation | None = None
        self.target_time = None
        self.pub = self.create_publisher(AckermannDriveStamped, "/drive/command", 10)
        self.create_subscription(TargetObservation, "/target", self.on_target, 10)
        self.create_timer(1.0 / self.rate, self.step)
        self.steps = 0
        self.get_logger().info(f"model {model}, {self.rate} Hz, speed limit {self.speed_limit}")

    def reset(self) -> None:
        """Forget the last episode, for a restarted simulation."""
        self.start_time = None
        self.prev = np.zeros(2, dtype=np.float32)
        self.target = self.target_time = None
        self.target_ages.clear()
        self.inference_times.clear()
        self.steps = 0

    def on_target(self, msg: TargetObservation) -> None:
        self.target, self.target_time = msg, self.get_clock().now()

    def observation(self) -> np.ndarray:
        t = self.target
        fresh = t is not None and (self.get_clock().now() - self.target_time).nanoseconds < self.stale_after * 1e9
        if not fresh or not t.visible:
            return np.array([0, 0, 0, 0, 0, 0, 0, *self.prev], dtype=np.float32)
        return np.array(
            [1, t.x, t.y, t.w, t.h, float(t.z_valid), t.z, *self.prev], dtype=np.float32
        )

    def step(self) -> None:
        now = self.get_clock().now()
        if self.last_now is not None and now < self.last_now:
            self.get_logger().info("simulated time jumped back, new episode")
            self.reset()
        self.last_now = now
        if self.start_time is None:
            self.start_time = now
            if self.start_delay:
                self.get_logger().info(f"driving starts in {self.start_delay:.0f} s (simulated time)")
        if (now - self.start_time).nanoseconds < self.start_delay * 1e9:
            return
        obs = self.observation()
        if obs[0]:
            age = now - Time.from_msg(self.target.header.stamp)
            self.target_ages.append(age.nanoseconds / 1e6)
        start = time.perf_counter()
        action = self.session.run(["action"], {"obs": obs[None, :]})[0][0]
        self.inference_times.append((time.perf_counter() - start) * 1e3)
        action = np.clip(action, -1.0, 1.0)
        self.prev = action.astype(np.float32)  # the network sees its own raw output
        steer, speed = float(action[0]), float(np.clip(action[1], -self.speed_limit, self.speed_limit))

        cmd = AckermannDriveStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = "base_link"
        cmd.drive.steering_angle = -steer * MAX_STEER  # positive = left (REP-103)
        cmd.drive.speed = speed * self.speed_per_command
        self.pub.publish(cmd)

        self.steps += 1
        if self.steps % int(self.rate) == 0:
            self.get_logger().info(
                f"visible {int(obs[0])} x {obs[1]:+.2f} z {'%.2f' % obs[6] if obs[5] else '  - '} "
                f"| steer {steer:+.2f} speed {speed:.2f}"
            )
        if self.steps % int(5 * self.rate) == 0:
            self.log_timing()

    def log_timing(self) -> None:
        ages, inference = np.array(self.target_ages), np.array(self.inference_times)
        text = f"network {np.median(inference):.2f} ms median, {inference.max():.2f} ms max"
        if len(ages):
            text = (f"target age {np.median(ages):.0f} ms median, {ages.max():.0f} ms max "
                    f"(simulated time) | {text}")
        self.get_logger().info(text)


def main() -> None:
    rclpy.init()
    spin(Policy())


if __name__ == "__main__":
    main()
