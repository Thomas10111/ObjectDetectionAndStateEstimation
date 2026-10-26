"""Make the figures of docs/how_it_works.md and copy the plots and animations there.

    source ~/.venvs/odse_policy/bin/activate
    python make_report_figures.py

Draws the camera geometry, the bicycle model and the PPO learning curves (from
the TensorBoard logs in out/tb), and copies the plots and GIFs that the
training and comparison scripts wrote to out/.
"""

import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, Polygon, Rectangle
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from evaluate import car_outline
from lego_car_env import (
    BODY_FRONT,
    BODY_HALF_WIDTH,
    BODY_REAR,
    CAMERA_HEIGHT,
    CAMERA_X,
    DEPTH_HFOV,
    HFOV,
    MAX_STEER,
    VFOV,
    WHEELBASE,
)

OUT = Path(__file__).parent / "out"
DOCS = Path(__file__).parent.parent / "docs" / "images"

COPIES = {
    "drive.gif": "drive_bc.gif",
    "search.gif": "search_ppo.gif",
    "expert_paths_ahead.png": "paths_rule_based_ahead.png",
    "expert_paths_anywhere.png": "paths_rule_based_anywhere.png",
    "bc_paths.png": "paths_imitation.png",
    "ppo_paths.png": "paths_ppo.png",
    "ppo_search_paths_anywhere.png": "paths_ppo_search_anywhere.png",
    "compare_gazebo_ppo.png": "gazebo_compare_ppo.png",
    "compare_gazebo_bc.png": "gazebo_compare_bc.png",
    "compare_gazebo.png": "gazebo_compare_search.png",
}

RUNS = {
    "ppo_2": "PPO, no speed limit (1 M steps)",
    "ppo_3": "PPO, 40 % speed limit (400 k steps)",
    "ppo_search_1": "PPO search, mixed scenario (2 M steps)",
}


def camera_geometry() -> None:
    """Top view: fields of view, forward and lateral distance; side view: height."""
    fig, (top, side) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.4, 1]})

    # Top view
    outline = car_outline(0, 0, 0)
    top.add_patch(Polygon(outline, closed=True, fill=False, lw=1.5))
    top.plot(CAMERA_X, 0, "ks")
    top.annotate("camera", (CAMERA_X, 0), (CAMERA_X - 0.15, -0.28), fontsize=9)
    top.plot(0, 0, "k^")
    top.annotate("base_link\n(rear axle)", (0, 0), (-0.25, 0.22), fontsize=8)
    reach = 2.0
    for half, color, label in ((HFOV / 2, "tab:orange", "color 70°"), (DEPTH_HFOV / 2, "tab:blue", "depth 68°")):
        for s in (-1, 1):
            top.plot([CAMERA_X, CAMERA_X + reach * math.cos(half)], [0, s * reach * math.sin(half)],
                     color=color, ls="--", lw=1, label=label if s == 1 else None)
    ball, r = np.array([1.6, -0.45]), 0.12
    top.add_patch(Circle(ball, r, color="tab:green", alpha=0.5))
    f, l = ball[0] - CAMERA_X, ball[1]
    top.plot([CAMERA_X, ball[0]], [0, 0], color="gray", lw=1)
    top.plot([ball[0], ball[0]], [0, l], color="gray", lw=1)
    top.plot([CAMERA_X, ball[0]], [0, l], color="tab:green", lw=1)
    top.text((CAMERA_X + ball[0]) / 2, 0.05, "f (forward)", ha="center", fontsize=9)
    top.text(ball[0] + 0.05, l / 2, "l (lateral)", fontsize=9)
    bearing = math.atan2(l, f)
    top.add_patch(Arc((CAMERA_X, 0), 0.9, 0.9, theta1=math.degrees(bearing), theta2=0, color="tab:green"))
    top.text(CAMERA_X + 0.5, -0.12, "β", color="tab:green", fontsize=11)
    top.text(ball[0] - 0.05, l - 0.25, "object, radius r", fontsize=9, color="tab:green")
    top.set_aspect("equal")
    top.set_xlim(-0.4, 2.3)
    top.set_ylim(-1.0, 1.0)
    top.set_xlabel("x (m), forward")
    top.set_ylabel("y (m), left")
    top.set_title("Top view: x = -(l / f) / tan 35°, box width w = (2r / f) / (2 tan 35°)")
    top.legend(loc="upper left", fontsize=8)

    # Side view
    side.add_patch(Rectangle((BODY_REAR, 0.03), BODY_FRONT - BODY_REAR, 0.12, fill=False, lw=1.5))
    side.add_patch(Circle((0, 0.0425), 0.0425, fill=False))
    side.add_patch(Circle((WHEELBASE, 0.0425), 0.0425, fill=False))
    side.plot(CAMERA_X, CAMERA_HEIGHT, "ks")
    reach = 1.6
    for s in (-1, 1):
        side.plot([CAMERA_X, CAMERA_X + reach], [CAMERA_HEIGHT, CAMERA_HEIGHT + s * reach * math.tan(VFOV / 2)],
                  color="tab:orange", ls="--", lw=1, label="color 43° vertical" if s == 1 else None)
    side.add_patch(Circle((1.4, 0.1), 0.1, color="tab:green", alpha=0.5))
    side.plot([CAMERA_X, 1.4], [CAMERA_HEIGHT, 0.1], color="tab:green", lw=1)
    side.axhline(0, color="black", lw=0.8)
    side.text(0.6, 0.2, "y = (h_cam - r) / f / tan 21.5°", fontsize=9)
    side.set_aspect("equal")
    side.set_xlim(-0.2, 1.9)
    side.set_ylim(-0.1, 0.8)
    side.set_xlabel("x (m)")
    side.set_ylabel("height (m)")
    side.set_title("Side view: camera 0.15 m high")
    side.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(DOCS / "camera_geometry.png", dpi=120)
    plt.close(fig)


def bicycle_model() -> None:
    """Front axle steering: turning radius R = L / tan(delta)."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    delta = MAX_STEER
    radius = WHEELBASE / math.tan(delta)
    center = (0.0, radius)  # left turn: center to the left of the rear axle
    ax.add_patch(Polygon(car_outline(0, 0, 0), closed=True, fill=False, lw=1.5))
    ax.plot([0, WHEELBASE], [0, 0], "k-", lw=2)
    ax.plot(0, 0, "ko")
    ax.plot(WHEELBASE, 0, "ko")
    for x0, ang in ((0, 0.0), (WHEELBASE, delta)):  # wheels
        dx, dy = 0.06 * math.cos(ang), 0.06 * math.sin(ang)
        ax.plot([x0 - dx, x0 + dx], [-dy, dy], color="tab:red", lw=5, solid_capstyle="butt")
    ax.plot(*center, "k+", ms=12)
    ax.plot([0, center[0]], [0, center[1]], color="gray", ls="--")
    ax.plot([WHEELBASE, center[0]], [0, center[1]], color="gray", ls="--")
    theta = np.linspace(-math.pi / 2, 0.3, 100)
    ax.plot(center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta), color="tab:blue",
            label=f"rear axle path, R = L / tan δ = {radius:.2f} m")
    ax.text(-0.08, radius / 2, "R", fontsize=12)
    ax.text(WHEELBASE / 2, -0.06, "L = 0.25 m", ha="center", fontsize=10)
    ax.text(WHEELBASE + 0.07, 0.03, f"δ = {math.degrees(delta):.1f}°", color="tab:red", fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlim(-0.75, 0.75)
    ax.set_ylim(-0.25, 1.25)
    ax.set_title("Bicycle model at full left lock: 1.20 m circle, as measured")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.tight_layout()
    fig.savefig(DOCS / "bicycle_model.png", dpi=120)
    plt.close(fig)


def learning_curves() -> None:
    """Test arrival rate and mean episode reward over training steps."""
    fig, (ax_arr, ax_rew) = plt.subplots(1, 2, figsize=(12, 4.2))
    for run, label in RUNS.items():
        folder = OUT / "tb" / run
        if not folder.exists():
            continue
        acc = EventAccumulator(str(folder))
        acc.Reload()
        tags = acc.Tags()["scalars"]
        if "check/arrived" in tags:
            events = acc.Scalars("check/arrived")
            ax_arr.plot([e.step for e in events], [100 * e.value for e in events], marker="o", ms=3, label=label)
        if "rollout/ep_rew_mean" in tags:
            events = acc.Scalars("rollout/ep_rew_mean")
            ax_rew.plot([e.step for e in events], [e.value for e in events], label=label)
    ax_arr.set_xlabel("training steps")
    ax_arr.set_ylabel("arrived in 100 test episodes (%)")
    ax_arr.set_title("Progress checks during training")
    ax_rew.set_xlabel("training steps")
    ax_rew.set_ylabel("mean episode reward (training episodes)")
    ax_rew.set_title("Reward of the episodes PPO collected")
    for ax in (ax_arr, ax_rew):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(DOCS / "learning_curves.png", dpi=120)
    plt.close(fig)


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    camera_geometry()
    bicycle_model()
    learning_curves()
    for src, dst in COPIES.items():
        if (OUT / src).exists():
            shutil.copy(OUT / src, DOCS / dst)
        else:
            print(f"missing: out/{src}")
    print(f"figures in {DOCS}:")
    for p in sorted(DOCS.iterdir()):
        print(f"  {p.name:36s} {p.stat().st_size // 1024:6d} KB")


if __name__ == "__main__":
    main()
