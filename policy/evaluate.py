"""Run a controller in LegoCarEnv and summarize or plot what it did.

Used by test_env.py (rule based controller) and train_bc.py (network).
A controller is any function that maps one observation to [steer, speed].
"""

import collections
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lego_car_env import BODY_FRONT, BODY_HALF_WIDTH, BODY_REAR, DT, LegoCarEnv

Controller = Callable[[np.ndarray], np.ndarray]
OUTCOMES = ("arrived", "collision", "lost", "away", "timeout")
COLORS = {
    "arrived": "tab:green",
    "collision": "tab:red",
    "lost": "tab:orange",
    "away": "tab:purple",
    "timeout": "tab:gray",
}


def front(x: float, y: float, heading: float) -> tuple[float, float]:
    """Center of the car front, which is what stops in front of the object."""
    return x + BODY_FRONT * np.cos(heading), y + BODY_FRONT * np.sin(heading)


def car_outline(x: float, y: float, heading: float) -> np.ndarray:
    corners = np.array(
        [
            [BODY_REAR, -BODY_HALF_WIDTH],
            [BODY_FRONT, -BODY_HALF_WIDTH],
            [BODY_FRONT, BODY_HALF_WIDTH],
            [BODY_REAR, BODY_HALF_WIDTH],
            [BODY_REAR, -BODY_HALF_WIDTH],
        ]
    )
    c, s = np.cos(heading), np.sin(heading)
    return corners @ np.array([[c, s], [-s, c]]) + [x, y]


def run_episode(env: LegoCarEnv, seed: int, controller: Controller) -> dict:
    """Drive one episode and return its outcome and path."""
    obs, _ = env.reset(seed=seed)
    path, total, done = [front(env.x, env.y, env.heading)], 0.0, False
    while not done:
        obs, reward, terminated, truncated, info = env.step(controller(obs))
        path.append(front(*info["pose"]))
        total += reward
        done = terminated or truncated
    return {
        "outcome": info["outcome"],
        "steps": env.steps,
        "reward": total,
        "path": np.array(path),
        "final_pose": info["pose"],
        "object": env.obj.copy(),
        "radius": env.radius,
    }


def evaluate(controller: Controller, seeds: range, env: LegoCarEnv | None = None) -> list[dict]:
    env = env or LegoCarEnv()
    return [run_episode(env, seed, controller) for seed in seeds]


def summarize(results: list[dict], label: str) -> dict:
    """Print outcome shares, time to arrive and mean reward; return them."""
    counts = collections.Counter(r["outcome"] for r in results)
    arrived = [r["steps"] * DT for r in results if r["outcome"] == "arrived"]
    summary = {
        "arrived": counts["arrived"] / len(results),
        "median_time": float(np.median(arrived)) if arrived else float("nan"),
        "reward": float(np.mean([r["reward"] for r in results])),
        "outcomes": {o: counts[o] / len(results) for o in OUTCOMES},
    }
    shares = "  ".join(
        f"{o} {100 * counts[o] / len(results):.1f}%" for o in OUTCOMES if counts[o]
    )
    print(
        f"{label}: {shares} | time to arrive median {summary['median_time']:.1f} s "
        f"| mean reward {summary['reward']:.2f}"
    )
    return summary


def plot_paths(results: list[dict], path: Path, title: str) -> None:
    """Top view of a few episodes: path of the car front, car at the end."""
    fig, ax = plt.subplots(figsize=(7, 6))
    shown = results[:8] + [r for r in results[8:] if r["outcome"] != "arrived"][:6]
    for r in shown:
        color = COLORS[r["outcome"]]
        ax.plot(r["path"][:, 0], r["path"][:, 1], color=color, lw=1)
        outline = car_outline(*r["final_pose"])
        ax.plot(outline[:, 0], outline[:, 1], color=color, lw=0.8)
        ax.add_patch(plt.Circle(r["object"], r["radius"], color=color, alpha=0.3))
    for name, color in COLORS.items():
        ax.plot([], [], color=color, label=name)
    ax.plot(0, 0, "k^", label="start")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
