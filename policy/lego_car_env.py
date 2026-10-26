"""Gymnasium environment: drive the LEGO car to a detected object.

A fast, render free stand in for camera, YOLO and car, used to train the
driving policy. The observation is what the real pipeline delivers: the YOLO
box of the object (position and size in the image), the F200 depth at the box,
and the policy's previous command. The action is steering and speed for the
LEGO hub.

Car and camera values are measured on the real car (odse_sim/README.md,
docs/camera_f200.md). Coordinates: base_link on the floor below the center of
the rear axle, x forward, y left, heading counterclockwise (REP-103).
"""

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Car, measured
WHEELBASE = 0.25  # m
MAX_STEER = 0.395  # rad (22.6 deg), from the 1.20 m turning circle
BODY_REAR, BODY_FRONT, BODY_HALF_WIDTH = -0.085, 0.335, 0.14  # m, from base_link
# Speed command 1.0 = 100 % motor power. 30 % gave 0.31 m/s on the test floor.
SPEED_PER_COMMAND = 1.0  # m/s per unit of speed command
# Speed commands are clipped to this until faster speeds are measured on the
# car (docs/interfaces.md, F8). The policy node and the hub bridge apply the
# same limit. Without it, PPO learned to drive at 100 % (1.1 m/s).
SPEED_LIMIT = 0.4

# Actuators, estimated
STEER_RATE = 1.5  # rad/s, how fast the front wheels follow the command
SPEED_TAU = 0.25  # s, time constant of the drive motors

# Camera (RealSense F200), docs/camera_f200.md. YOLO runs on the color camera,
# so boxes use the color field of view; the depth camera only decides whether
# a depth reading exists. Interfaces: docs/interfaces.md.
CAMERA_X, CAMERA_HEIGHT = 0.25, 0.15  # m, over the front axle
HFOV, VFOV = math.radians(70), math.radians(43)  # color camera, 16:9
DEPTH_HFOV = math.radians(68)  # depth camera, 4:3 (vertically wider than color)
DEPTH_MIN, DEPTH_MAX = 0.2, 1.0  # m
IMAGE_HEIGHT_PX = 360  # image height as YOLO sees it (640 wide input, 16:9)
MIN_BOX_PX = 10  # YOLO misses objects smaller than this

# Task
DT = 0.1  # s per step (10 commands per second)
SUBSTEPS = 5
ARRIVE_GAP = 0.30  # m, car front to object surface
ARRIVE_TOLERANCE = 0.15  # m
ARRIVE_MAX_SPEED = 0.05  # m/s, must be (nearly) stopped
MAX_STEPS = 400  # 40 s
LOST_STEPS = 30  # 3 s out of view ends the episode
MAX_DISTANCE = 4.0  # m from the object ends the episode
STEER_CHANGE_COST = 0.05  # reward cost per unit of steering change per step

# Observation layout
OBS_NAMES = ("visible", "x", "y", "w", "h", "z_valid", "z", "prev_steer", "prev_speed")


class LegoCarEnv(gym.Env):
    """Drive to an object seen by the camera and stop in front of it.

    Observation (9 floats, all roughly in -1 .. 1):
        visible     1 if YOLO reports the object this step, else 0
        x, y        box center in the image, -1 left/top .. 1 right/bottom
        w, h        box size as a fraction of the image width/height
        z_valid     1 if the F200 has a depth reading (0.2 to 1.0 m)
        z           depth to the object surface in meters, 0 if not valid
        prev_steer, prev_speed   the previous action
    The detection fields are 0 when visible is 0.

    Action (2 floats in -1 .. 1):
        steer       -1 full left .. 1 full right
        speed       -1 full reverse .. 1 full forward (fraction of motor power)

    Episodes end on arrival (+10), collision (-10), driving 4 m away (-5), or
    after 40 s (truncated). In the "ahead" scenario, losing the object for 3 s
    also ends the episode (-5); in "anywhere" the car has to search for it.
    """

    metadata = {"render_modes": []}

    SCENARIOS = ("ahead", "anywhere", "mixed")

    def __init__(self, randomize: bool = True, scenario: str = "ahead") -> None:
        """Create the environment.

        Args:
            randomize: Vary car and sensor properties per episode (domain
                randomization), so the policy does not rely on exact values.
            scenario: Where the object starts. "ahead": inside the camera
                view, 0.8 to 3 m away. "anywhere": any direction around the
                car, 0.8 to 2.5 m away, so it often has to be searched for.
                "mixed": half of the episodes each.
        """
        if scenario not in self.SCENARIOS:
            raise ValueError(f"scenario must be one of {self.SCENARIOS}")
        self.randomize = randomize
        self.scenario = scenario
        self.observation_space = spaces.Box(-1.5, 1.5, shape=(len(OBS_NAMES),), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    # Episode

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        rng = self.np_random

        # Car at the origin facing +x; object ahead in view, or anywhere around.
        self.x = self.y = self.heading = 0.0
        self.speed = self.steer_angle = 0.0
        self.radius = rng.uniform(0.06, 0.15)
        scenario = self.scenario
        if scenario == "mixed":
            scenario = "anywhere" if rng.random() < 0.5 else "ahead"
        self.episode_scenario = scenario
        if scenario == "ahead":
            distance = rng.uniform(0.8, 3.0)
            bearing = rng.uniform(-0.8, 0.8) * HFOV / 2
        else:
            distance = rng.uniform(0.8, 2.5)
            bearing = rng.uniform(-math.pi, math.pi)
        self.obj = np.array(
            [CAMERA_X + distance * math.cos(bearing), distance * math.sin(bearing)]
        )

        # Per episode variation of the real world
        r = self.randomize
        self.speed_scale = rng.uniform(0.85, 1.15) if r else 1.0  # battery, floor
        self.steer_scale = rng.uniform(0.9, 1.1) if r else 1.0
        self.steer_offset = rng.uniform(-0.035, 0.035) if r else 0.0  # calibration, 2 deg
        self.delay_steps = int(rng.integers(0, 3)) if r else 1  # Bluetooth plus YOLO, 0 to 0.2 s
        self.dropout = rng.uniform(0.0, 0.15) if r else 0.05  # missed detections
        self.noise = rng.uniform(0.5, 1.5) if r else 1.0

        # A fixed scenario, for example to repeat a Gazebo experiment:
        # reset(options={"object": (1.8, -0.3), "radius": 0.1, "delay_steps": 0, ...})
        options = options or {}
        if "object" in options:
            self.obj = np.array(options["object"], dtype=float)
        for key in ("radius", "delay_steps", "dropout", "noise", "speed_scale", "steer_scale", "steer_offset"):
            if key in options:
                setattr(self, key, options[key])

        self.pending = [np.zeros(2, dtype=np.float32)] * self.delay_steps
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.steps = self.lost = 0
        self.prev_gap = self._gap()
        return self._observe(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        # Commands reach the motors delay_steps later
        self.pending.append(action)
        applied = self.pending.pop(0)
        self._drive(applied)
        self.steps += 1

        gap = self._gap()
        in_view = self._in_view()
        self.lost = 0 if in_view else self.lost + 1
        obs = self._observe()

        # Reward: progress toward the stop point, keep the object centered,
        # smooth steering, and a small cost per step.
        reward = 5.0 * (abs(self.prev_gap - ARRIVE_GAP) - abs(gap - ARRIVE_GAP))
        reward -= 0.01 + 0.02 * abs(float(obs[1])) * float(obs[0])
        reward -= STEER_CHANGE_COST * abs(float(action[0] - self.prev_action[0]))
        self.prev_gap = gap
        self.prev_action = action

        terminated, outcome = False, None
        if self._collided():
            terminated, outcome, reward = True, "collision", reward - 10.0
        elif self._arrived(gap):
            terminated, outcome, reward = True, "arrived", reward + 10.0
        elif self.episode_scenario == "ahead" and self.lost >= LOST_STEPS:
            terminated, outcome, reward = True, "lost", reward - 5.0
        elif self._distance() > MAX_DISTANCE:
            terminated, outcome, reward = True, "away", reward - 5.0
        truncated = not terminated and self.steps >= MAX_STEPS
        if truncated:
            outcome = "timeout"

        info = {"outcome": outcome, "pose": (self.x, self.y, self.heading), "gap": gap}
        return obs, float(reward), terminated, truncated, info

    # Car

    def _drive(self, action: np.ndarray) -> None:
        """Bicycle model with rate limited steering and lagging motors."""
        target_angle = -float(action[0]) * MAX_STEER * self.steer_scale + self.steer_offset
        speed_command = float(np.clip(action[1], -SPEED_LIMIT, SPEED_LIMIT))
        target_speed = speed_command * SPEED_PER_COMMAND * self.speed_scale
        h = DT / SUBSTEPS
        for _ in range(SUBSTEPS):
            delta = np.clip(target_angle - self.steer_angle, -STEER_RATE * h, STEER_RATE * h)
            self.steer_angle = float(np.clip(self.steer_angle + delta, -MAX_STEER, MAX_STEER))
            self.speed += (target_speed - self.speed) * h / SPEED_TAU
            self.x += self.speed * math.cos(self.heading) * h
            self.y += self.speed * math.sin(self.heading) * h
            self.heading += self.speed / WHEELBASE * math.tan(self.steer_angle) * h

    def _object_in_car(self) -> tuple[float, float]:
        """Object center in the car frame (x forward, y left)."""
        dx, dy = self.obj[0] - self.x, self.obj[1] - self.y
        c, s = math.cos(self.heading), math.sin(self.heading)
        return c * dx + s * dy, -s * dx + c * dy

    def _gap(self) -> float:
        """Distance from the car front to the object surface, straight ahead."""
        ox, oy = self._object_in_car()
        return math.hypot(ox - BODY_FRONT, oy) - self.radius

    def _distance(self) -> float:
        return math.hypot(self.obj[0] - self.x, self.obj[1] - self.y)

    def _collided(self) -> bool:
        ox, oy = self._object_in_car()
        nearest_x = min(max(ox, BODY_REAR), BODY_FRONT)
        nearest_y = min(max(oy, -BODY_HALF_WIDTH), BODY_HALF_WIDTH)
        return math.hypot(ox - nearest_x, oy - nearest_y) < self.radius

    def _arrived(self, gap: float) -> bool:
        ox, oy = self._object_in_car()
        return (
            abs(gap - ARRIVE_GAP) <= ARRIVE_TOLERANCE
            and abs(oy) < BODY_HALF_WIDTH
            and ox > BODY_FRONT
            and abs(self.speed) < ARRIVE_MAX_SPEED
        )

    # Camera and YOLO

    def _in_view(self) -> bool:
        """Whether the object is in the image and large enough for YOLO."""
        ox, oy = self._object_in_car()
        forward = ox - CAMERA_X
        if forward <= self.radius:
            return False
        if abs(math.atan2(oy, forward)) > HFOV / 2:
            return False
        box_px = 2 * self.radius / forward / (2 * math.tan(VFOV / 2)) * IMAGE_HEIGHT_PX
        return box_px >= MIN_BOX_PX

    def _observe(self) -> np.ndarray:
        prev = self.prev_action
        if not self._in_view() or self.np_random.random() < self.dropout:
            return np.array([0, 0, 0, 0, 0, 0, 0, prev[0], prev[1]], dtype=np.float32)

        rng, n = self.np_random, self.noise
        ox, oy = self._object_in_car()
        forward = ox - CAMERA_X
        tan_h, tan_v = math.tan(HFOV / 2), math.tan(VFOV / 2)
        x = -(oy / forward) / tan_h + rng.normal(0, 0.02 * n)  # right is positive
        y = (CAMERA_HEIGHT - self.radius) / forward / tan_v + rng.normal(0, 0.02 * n)
        size = 2 * self.radius / forward * rng.normal(1, 0.05 * n)
        w, h = size / (2 * tan_h), size / (2 * tan_v)
        z = forward - self.radius + rng.normal(0, 0.01 * n)
        in_depth_view = abs(math.atan2(oy, forward)) <= DEPTH_HFOV / 2
        z_valid = in_depth_view and DEPTH_MIN <= z <= DEPTH_MAX
        return np.array(
            [
                1.0,
                np.clip(x, -1, 1),
                np.clip(y, -1, 1),
                np.clip(w, 0, 1),
                np.clip(h, 0, 1),
                float(z_valid),
                z if z_valid else 0.0,
                prev[0],
                prev[1],
            ],
            dtype=np.float32,
        )
