"""Animate the trained network driving in the training simulator.

    source ~/.venvs/odse_policy/bin/activate
    python animate.py                       # out/bc_policy.onnx, 3 episodes
    python animate.py --episodes 5 --seed 7
    python animate.py --model out/ppo_search_policy.onnx --scenario anywhere

Runs the exported ONNX file with ONNX Runtime (no PyTorch), so it also checks
that the file works on its own. Writes out/drive.gif with three panels per
step: the car seen from above, the camera image as the network sees it (the
YOLO box and the depth), and the network's commands.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
from matplotlib.animation import FuncAnimation, PillowWriter

from evaluate import car_outline, front
from lego_car_env import DT, HFOV, LegoCarEnv

OUT = Path(__file__).parent / "out"
CAMERA_ASPECT = 16 / 9


def record(onnx_path: Path, episodes: int, seed: int, scenario: str) -> list[dict]:
    """Drive episodes with the ONNX network and record every step."""
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    env = LegoCarEnv(scenario=scenario)
    frames = []
    episode_limits = {}
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        done, path = False, []
        while not done:
            action = session.run(["action"], {"obs": obs[None, :]})[0][0]
            path.append(front(env.x, env.y, env.heading))
            limits = episode_limits.setdefault(episode, [env.obj[0], env.obj[0], env.obj[1], env.obj[1]])
            fx, fy = path[-1]
            limits[:] = [min(limits[0], fx, 0), max(limits[1], fx, 0), min(limits[2], fy, 0), max(limits[3], fy, 0)]
            frames.append(
                {
                    "episode": episode,
                    "time": env.steps * DT,
                    "pose": (env.x, env.y, env.heading),
                    "path": list(path),
                    "object": env.obj.copy(),
                    "radius": env.radius,
                    "obs": obs.copy(),
                    "action": action.copy(),
                    "outcome": None,
                }
            )
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        frames[-1]["outcome"] = info["outcome"]
        frames.extend([dict(frames[-1])] * 10)  # hold the last frame for 1 s
    for f in frames:
        f["limits"] = episode_limits[f["episode"]]
    return frames


def animate(frames: list[dict], path: Path) -> None:
    fig = plt.figure(figsize=(11, 4.2))
    grid = fig.add_gridspec(2, 3, width_ratios=[1.3, 1.1, 0.6], height_ratios=[1, 1])
    top = fig.add_subplot(grid[:, 0])
    cam = fig.add_subplot(grid[0, 1])
    info = fig.add_subplot(grid[1, 1])
    bars = fig.add_subplot(grid[:, 2])
    fig.subplots_adjust(bottom=0.2, wspace=0.35)

    def draw(i: int) -> None:
        f = frames[i]
        visible, x, y, w, h, z_valid, z, prev_steer, prev_speed = f["obs"]
        steer, speed = f["action"]

        # Top view, centered on the object
        top.clear()
        ox, oy = f["object"]
        top.add_patch(plt.Circle((ox, oy), f["radius"], color="tab:green", alpha=0.5))
        p = np.array(f["path"])
        top.plot(p[:, 0], p[:, 1], color="tab:blue", lw=1)
        outline = car_outline(*f["pose"])
        top.plot(outline[:, 0], outline[:, 1], color="black", lw=1.2)
        cx, cy, heading = f["pose"]
        fx, fy = front(cx, cy, heading)
        for side in (-1, 1):  # camera field of view
            angle = heading + side * HFOV / 2
            top.plot([fx, fx + 1.2 * np.cos(angle)], [fy, fy + 1.2 * np.sin(angle)], color="tab:orange", lw=0.6, ls="--")
        x0, x1, y0, y1 = f["limits"]
        half = max(x1 - x0, y1 - y0) / 2 + 0.6
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        top.set_xlim(mx - half, mx + half)
        top.set_ylim(my - half, my + half)
        top.set_aspect("equal")
        top.set_title(f"episode {f['episode'] + 1}, t = {f['time']:.1f} s" + (f"  ({f['outcome']})" if f["outcome"] else ""))
        top.set_xlabel("x (m)")
        top.set_ylabel("y (m)")

        # Camera image as the network sees it: just the YOLO box
        cam.clear()
        cam.set_xlim(-1, 1)
        cam.set_ylim(1, -1)  # image y points down
        cam.set_aspect(1 / CAMERA_ASPECT)
        cam.set_xticks([])
        cam.set_yticks([])
        cam.set_facecolor("#f4f4f4")
        cam.axvline(0, color="gray", lw=0.5)
        if visible:
            cam.add_patch(plt.Rectangle((x - w, y - h), 2 * w, 2 * h, fill=False, color="tab:green", lw=2))
            cam.plot(x, y, "r+")
            cam.set_title("camera: YOLO box")
        else:
            cam.set_title("camera: nothing detected")

        info.clear()
        info.axis("off")
        depth = f"z = {z:.2f} m" if z_valid else "z = unknown (> 1 m or out of view)"
        info.text(0, 0.8, f"x = {x:+.2f}   y = {y:+.2f}", fontsize=10, family="monospace")
        info.text(0, 0.55, f"w = {w:.2f}    h = {h:.2f}", fontsize=10, family="monospace")
        info.text(0, 0.3, depth, fontsize=10, family="monospace")
        info.text(0, 0.0, "network input: 9 numbers", fontsize=8, color="gray")

        # Network output
        bars.clear()
        bars.barh(["speed", "steer"], [speed, steer], color=["tab:blue", "tab:orange"])
        bars.set_xlim(-1, 1)
        bars.axvline(0, color="black", lw=0.5)
        bars.set_title("network output")
        bars.set_xlabel("steer: left -1 .. +1 right\nspeed: back -1 .. +1 forward", fontsize=8)

    animation = FuncAnimation(fig, draw, frames=len(frames), interval=DT * 1000)
    path.parent.mkdir(exist_ok=True)
    animation.save(str(path), writer=PillowWriter(fps=round(1 / DT)))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, default=OUT / "bc_policy.onnx")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=300_000)
    parser.add_argument("--scenario", default="ahead", choices=LegoCarEnv.SCENARIOS)
    parser.add_argument("--out", type=Path, default=OUT / "drive.gif")
    args = parser.parse_args()
    frames = record(args.model, args.episodes, args.seed, args.scenario)
    outcomes = [f["outcome"] for f in frames if f["outcome"]]
    print(f"{args.episodes} episodes, outcomes: {sorted(set(outcomes))}")
    animate(frames, args.out)
    print(f"animation: {args.out} ({len(frames)} frames at {round(1 / DT)} per second)")


if __name__ == "__main__":
    main()
