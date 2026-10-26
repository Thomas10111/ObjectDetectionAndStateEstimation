"""Compare a Gazebo driving experiment with the same scenario in the training simulator.

    source ~/.venvs/odse_policy/bin/activate
    python compare_gazebo.py ~/odse_experiments/<run>.csv
    python compare_gazebo.py <run>.csv --model out/bc_policy.onnx --object 1.8 -0.3

The training simulator drives the same ONNX network with the ball at the same
place, without random variation, delay or detection noise (as in the default
Gazebo experiment). Writes out/compare_gazebo.png: the paths from above, and
gap, speed and steering over time for both.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort

from evaluate import car_outline, front
from lego_car_env import DT, MAX_STEER, SPEED_LIMIT, LegoCarEnv

OUT = Path(__file__).parent / "out"


def load_gazebo(path: Path) -> dict:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    data = {key: np.array([float(r[key]) for r in rows]) for key in rows[0]}
    return data


def run_training_sim(model: Path, obj: tuple[float, float], radius: float, seconds: float) -> dict:
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    env = LegoCarEnv(randomize=False, scenario="anywhere")  # no end when the ball is out of view
    obs, _ = env.reset(
        seed=0,
        options={"object": obj, "radius": radius, "delay_steps": 0, "dropout": 0.0, "noise": 0.0},
    )
    log = {key: [] for key in ("t", "x", "y", "yaw", "gap", "speed", "steering_angle")}
    outcome, done = None, False
    while not done and env.steps * DT <= seconds:
        action = session.run(["action"], {"obs": obs[None, :]})[0][0]
        for key, value in zip(log, (env.steps * DT, env.x, env.y, env.heading, env._gap(), env.speed,
                                    -float(np.clip(action[0], -1, 1)) * MAX_STEER)):
            log[key].append(value)
        obs, _, terminated, truncated, info = env.step(action)
        done, outcome = terminated or truncated, info["outcome"]
    return {key: np.array(value) for key, value in log.items()} | {"outcome": outcome}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", type=Path, help="CSV log from episode_monitor")
    parser.add_argument("--model", type=Path, default=OUT / "ppo_policy.onnx")
    parser.add_argument("--object", type=float, nargs=2, default=(1.8, -0.3), help="ball position in Gazebo")
    parser.add_argument("--radius", type=float, default=0.1)
    args = parser.parse_args()

    gz = load_gazebo(args.csv.expanduser())
    seconds = float(gz["t"][-1])
    sim = run_training_sim(args.model, tuple(args.object), args.radius, seconds)
    print(f"training simulator: {sim['outcome']} after {sim['t'][-1] + DT:.1f} s, final gap {sim['gap'][-1]:.2f} m")
    stopped = np.nonzero((gz["speed"] < 0.05) & (gz["gap"] < 0.45) & (gz["t"] > 0.5))[0]
    if stopped.size:
        i = stopped[0]
        print(f"Gazebo:             stopped after {gz['t'][i]:.1f} s, gap {gz['gap'][i]:.2f} m")
    else:
        print(f"Gazebo:             not stopped near the ball within {seconds:.1f} s")
    for name, d in (("training simulator", sim), ("Gazebo", gz)):
        changes = np.abs(np.diff(d["steering_angle"]))
        print(f"{name:19s} max speed {d['speed'].max():.2f} m/s, steering changes per step: mean "
              f"{changes.mean() / MAX_STEER:.3f} (in units of full lock), sign flips {np.sum(np.diff(np.sign(d['steering_angle'])) != 0)}")

    fig = plt.figure(figsize=(12, 6.5))
    grid = fig.add_gridspec(3, 2, width_ratios=[1.2, 1])
    top = fig.add_subplot(grid[:, 0])
    ax_gap, ax_speed, ax_steer = (fig.add_subplot(grid[i, 1]) for i in range(3))
    styles = {"training simulator": ("tab:blue", sim), "Gazebo": ("tab:red", gz)}
    for name, (color, d) in styles.items():
        fx, fy = front(d["x"], d["y"], d["yaw"])
        top.plot(fx, fy, color=color, label=f"{name}, car front")
        outline = car_outline(d["x"][-1], d["y"][-1], d["yaw"][-1])
        top.plot(outline[:, 0], outline[:, 1], color=color, lw=0.8)
        ax_gap.plot(d["t"], d["gap"], color=color, label=name)
        ax_speed.plot(d["t"], d["speed"], color=color)
        ax_steer.plot(d["t"], np.degrees(d["steering_angle"]), color=color)
    top.add_patch(plt.Circle(args.object, args.radius, color="tab:orange", alpha=0.6, label="ball"))
    top.plot(0, 0, "k^", label="start (rear axle)")
    top.set_aspect("equal")
    top.set_xlabel("x (m)")
    top.set_ylabel("y (m)")
    top.set_title(f"Same scenario, {args.model.name}")
    top.legend(loc="upper left", fontsize=8)
    ax_gap.axhspan(0.15, 0.45, color="tab:green", alpha=0.15, label="arrival zone")
    ax_gap.set_ylabel("gap to ball (m)")
    ax_gap.legend(fontsize=8)
    ax_speed.axhline(SPEED_LIMIT, color="gray", ls="--", lw=0.8)
    ax_speed.set_ylabel("speed (m/s)")
    ax_steer.set_ylabel("steering command (deg)\n+ = left")
    ax_steer.set_xlabel("time (s)")
    for ax in (ax_gap, ax_speed, ax_steer):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "compare_gazebo.png", dpi=120)
    print(f"plot: {OUT / 'compare_gazebo.png'}")


if __name__ == "__main__":
    main()
