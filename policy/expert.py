"""Rule based driving controller: the benchmark and teacher for the network.

Steers toward the object, drives slower when it gets close, and stops at the
arrival distance using the F200 depth. Without a detection it drives a slow
full lock circle to search. Works on the same observation as the
network (lego_car_env.OBS_NAMES).
"""

import numpy as np

from lego_car_env import ARRIVE_GAP, BODY_FRONT, CAMERA_X

STEER_GAIN = 2.0
CRUISE_SPEED = 0.35  # 35 % motor power
APPROACH_GAIN = 0.8
SEARCH_SPEED = 0.2  # driving a full lock circle (about 1.2 m) sweeps the camera around
# The camera sits behind the car front, so the depth at arrival is larger.
STOP_DEPTH = ARRIVE_GAP + (BODY_FRONT - CAMERA_X)


def expert_action(obs: np.ndarray) -> np.ndarray:
    """Return [steer, speed] for one observation."""
    visible, x, _y, w, _h, z_valid, z, prev_steer, _prev_speed = obs
    if not visible:
        # Search: full lock to the side it was last steering toward (left if
        # none), and drive slowly in a circle until the object shows up.
        side = 1.0 if prev_steer > 0.05 else -1.0
        return np.array([side, SEARCH_SPEED], dtype=np.float32)
    steer = np.clip(STEER_GAIN * x, -1.0, 1.0)
    if z_valid:
        speed = np.clip(APPROACH_GAIN * (z - STOP_DEPTH), -0.1, CRUISE_SPEED)
    else:
        # Farther than about 1 m: the box size says how far.
        speed = CRUISE_SPEED if w < 0.25 else 0.15
    return np.array([steer, speed], dtype=np.float32)
