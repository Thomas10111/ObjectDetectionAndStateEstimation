"""What camera and YOLO would report for an object at a known position.

The same geometry as the training simulator (policy/lego_car_env.py,
LegoCarEnv._in_view and _observe), without noise unless asked for. Keep the
two in step: the policy only works if Gazebo shows it what training showed it.
Units and conventions: docs/interfaces.md.
"""

import math
from dataclasses import dataclass

import numpy as np

# Car and camera, as in lego_car_env.py
WHEELBASE = 0.25
MAX_STEER = 0.395
BODY_REAR, BODY_FRONT, BODY_HALF_WIDTH = -0.085, 0.335, 0.14
CAMERA_X, CAMERA_HEIGHT = 0.25, 0.15
HFOV, VFOV = math.radians(70), math.radians(43)
DEPTH_HFOV = math.radians(68)
DEPTH_MIN, DEPTH_MAX = 0.2, 1.0
IMAGE_HEIGHT_PX = 360
MIN_BOX_PX = 10
ARRIVE_GAP, ARRIVE_TOLERANCE, ARRIVE_MAX_SPEED = 0.30, 0.15, 0.05


@dataclass
class Detection:
    visible: bool = False
    x: float = 0.0
    y: float = 0.0
    w: float = 0.0
    h: float = 0.0
    z_valid: bool = False
    z: float = 0.0


def object_in_car(car: tuple[float, float, float], obj: tuple[float, float]) -> tuple[float, float]:
    """Object center in the car frame (x forward, y left); car = (x, y, yaw)."""
    cx, cy, yaw = car
    dx, dy = obj[0] - cx, obj[1] - cy
    c, s = math.cos(yaw), math.sin(yaw)
    return c * dx + s * dy, -s * dx + c * dy


def observe(car, obj, radius: float, rng: np.random.Generator | None = None,
            noise: float = 0.0, dropout: float = 0.0) -> Detection:
    """Detection of a round object of the given radius, as LegoCarEnv computes it."""
    ox, oy = object_in_car(car, obj)
    forward = ox - CAMERA_X
    if forward <= radius or abs(math.atan2(oy, forward)) > HFOV / 2:
        return Detection()
    box_px = 2 * radius / forward / (2 * math.tan(VFOV / 2)) * IMAGE_HEIGHT_PX
    if box_px < MIN_BOX_PX:
        return Detection()
    rng = rng or np.random.default_rng()
    if dropout and rng.random() < dropout:
        return Detection()

    def jitter(sigma: float) -> float:
        return rng.normal(0, sigma * noise) if noise else 0.0

    tan_h, tan_v = math.tan(HFOV / 2), math.tan(VFOV / 2)
    x = -(oy / forward) / tan_h + jitter(0.02)
    y = (CAMERA_HEIGHT - radius) / forward / tan_v + jitter(0.02)
    size = 2 * radius / forward * (1 + jitter(0.05))
    z = forward - radius + jitter(0.01)
    z_valid = abs(math.atan2(oy, forward)) <= DEPTH_HFOV / 2 and DEPTH_MIN <= z <= DEPTH_MAX
    return Detection(
        visible=True,
        x=float(np.clip(x, -1, 1)),
        y=float(np.clip(y, -1, 1)),
        w=float(np.clip(size / (2 * tan_h), 0, 1)),
        h=float(np.clip(size / (2 * tan_v), 0, 1)),
        z_valid=bool(z_valid),
        z=float(z) if z_valid else 0.0,
    )


def gap(car, obj, radius: float) -> float:
    """Distance from the car front to the object surface."""
    ox, oy = object_in_car(car, obj)
    return math.hypot(ox - BODY_FRONT, oy) - radius


def collided(car, obj, radius: float) -> bool:
    ox, oy = object_in_car(car, obj)
    nx = min(max(ox, BODY_REAR), BODY_FRONT)
    ny = min(max(oy, -BODY_HALF_WIDTH), BODY_HALF_WIDTH)
    return math.hypot(ox - nx, oy - ny) < radius


def arrived(car, obj, radius: float, speed: float) -> bool:
    ox, oy = object_in_car(car, obj)
    return (
        abs(gap(car, obj, radius) - ARRIVE_GAP) <= ARRIVE_TOLERANCE
        and abs(oy) < BODY_HALF_WIDTH
        and ox > BODY_FRONT
        and abs(speed) < ARRIVE_MAX_SPEED
    )


def yaw_from_quaternion(q) -> float:
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
