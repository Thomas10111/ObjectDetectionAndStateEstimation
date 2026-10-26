"""Train the driving network by imitating the rule based controller.

    source ~/.venvs/odse_policy/bin/activate
    python train_bc.py

Imitation learning with DAgger: round 0 records the rule based controller
(with some action noise, to see a few off course situations). In the later
rounds the network drives, and the rule based controller labels every
observation with the action it would have taken. All labels are pooled, and the
network is retrained after each round. This teaches it to recover from its own
mistakes, which plain copying does not.

Writes to out/:
    bc_policy.pt      PyTorch weights (state dict)
    bc_policy.onnx    the same network for ONNX Runtime and TensorRT; input
                      "obs" float32[N, 9], output "action" float32[N, 2]
                      (docs/interfaces.md, M4)
    bc_paths.png      a few episodes driven by the network
    bc_results.json   size, training time and test results

"bc" is the default --name; --hidden 16,16 trains a smaller network.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from evaluate import evaluate, plot_paths, summarize
from expert import expert_action
from lego_car_env import OBS_NAMES, LegoCarEnv

OUT = Path(__file__).parent / "out"
OBS_DIM, ACT_DIM = len(OBS_NAMES), 2
EVAL_SEEDS = range(100_000, 100_500)  # never used for training


class PolicyNet(nn.Module):
    """9 observations in, [steer, speed] out, each squashed to -1 .. 1.

    hidden lists the width of each hidden layer, (64, 64) by default.
    """

    def __init__(self, hidden: tuple[int, ...] = (64, 64)) -> None:
        super().__init__()
        layers, width = [], OBS_DIM
        for size in hidden:
            layers += [nn.Linear(width, size), nn.Tanh()]
            width = size
        self.layers = nn.Sequential(*layers, nn.Linear(width, ACT_DIM), nn.Tanh())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.layers(obs)


def parse_hidden(text: str) -> tuple[int, ...]:
    """'16,16' gives (16, 16): the width of each hidden layer."""
    return tuple(int(n) for n in text.split(",") if n)


def count_parameters(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())


def as_controller(net: PolicyNet):
    """Wrap the network as a controller function for evaluate.py."""

    @torch.no_grad()
    def act(obs: np.ndarray) -> np.ndarray:
        return net(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()

    return act


def collect(env: LegoCarEnv, net: PolicyNet | None, episodes: int, seed0: int,
            beta: float, noise: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Drive episodes and record (observation, rule based label) pairs.

    Each step the rule based controller drives with probability beta (plus
    Gaussian action noise), otherwise the network drives.
    """
    act = as_controller(net) if net is not None else None
    observations, labels = [], []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed0 + episode)
        done = False
        while not done:
            label = expert_action(obs)
            observations.append(obs)
            labels.append(label)
            if act is None or rng.random() < beta:
                action = np.clip(label + rng.normal(0, noise, ACT_DIM), -1, 1).astype(np.float32)
            else:
                action = act(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    return np.array(observations, dtype=np.float32), np.array(labels, dtype=np.float32)


def fit(net: PolicyNet, obs: np.ndarray, labels: np.ndarray, epochs: int) -> float:
    """Supervised training on all pairs collected so far; returns the last loss."""
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    x, y = torch.from_numpy(obs), torch.from_numpy(labels)
    loss = torch.tensor(0.0)
    for _ in range(epochs):
        for idx in torch.randperm(len(x)).split(256):
            loss = nn.functional.mse_loss(net(x[idx]), y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return loss.item()


def export_onnx(net: PolicyNet, path: Path) -> float:
    """Export to ONNX and return the largest difference to PyTorch on test inputs."""
    net.eval()
    example = torch.zeros(1, OBS_DIM)
    torch.onnx.export(
        net,
        (example,),
        str(path),
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    test = np.random.default_rng(0).uniform(-1, 1, (1000, OBS_DIM)).astype(np.float32)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    onnx_out = session.run(["action"], {"obs": test})[0]
    with torch.no_grad():
        torch_out = net(torch.from_numpy(test)).numpy()
    return float(np.abs(onnx_out - torch_out).max())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rounds", type=int, default=6, help="DAgger rounds after round 0")
    parser.add_argument("--episodes", type=int, default=100, help="episodes per round")
    parser.add_argument("--epochs", type=int, default=15, help="training epochs per round")
    parser.add_argument("--hidden", default="64,64", help="hidden layer widths, for example 16,16 or 8")
    parser.add_argument("--name", default="bc", help="output file prefix in out/")
    parser.add_argument("--threads", type=int, default=4, help="PyTorch threads")
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(0)
    env = LegoCarEnv()
    hidden = parse_hidden(args.hidden)
    net = PolicyNet(hidden)
    print(f"network: 9 -> {' -> '.join(map(str, hidden))} -> 2, {count_parameters(net)} parameters")
    t0 = time.perf_counter()

    obs, labels = collect(env, None, args.episodes, seed0=0, beta=1.0, noise=0.15, rng=rng)
    for round_ in range(args.rounds + 1):
        if round_ > 0:
            beta = 0.5**round_
            new_obs, new_labels = collect(
                env, net, args.episodes, seed0=round_ * 10_000, beta=beta, noise=0.15, rng=rng
            )
            obs, labels = np.concatenate([obs, new_obs]), np.concatenate([labels, new_labels])
        loss = fit(net, obs, labels, args.epochs)
        quick = evaluate(as_controller(net), range(200_000, 200_050), env)
        arrived = sum(r["outcome"] == "arrived" for r in quick) / len(quick)
        print(f"round {round_}: {len(obs):6d} samples, loss {loss:.4f}, arrived {100 * arrived:.0f}% of 50")
    train_seconds = time.perf_counter() - t0
    print(f"training took {train_seconds:.0f} s")

    print(f"--- final evaluation, {len(EVAL_SEEDS)} unseen episodes")
    expert = summarize(evaluate(expert_action, EVAL_SEEDS, env), "rule based controller")
    results = evaluate(as_controller(net), EVAL_SEEDS, env)
    learned = summarize(results, "network            ")
    print(f"network reaches {100 * learned['arrived'] / max(expert['arrived'], 1e-9):.0f}% of the rule based success rate")

    OUT.mkdir(exist_ok=True)
    name = args.name
    torch.save(net.state_dict(), OUT / f"{name}_policy.pt")
    diff = export_onnx(net, OUT / f"{name}_policy.onnx")
    print(f"ONNX export: largest difference to PyTorch on 1000 inputs {diff:.2e}")
    plot_paths(results, OUT / f"{name}_paths.png", f"Network (imitation, {name}): path of the car front, car at the end")
    summary = {"name": name, "method": "bc", "hidden": hidden, "parameters": count_parameters(net),
               "train_seconds": train_seconds, "onnx_diff": diff, "eval": {"ahead": learned}}
    (OUT / f"{name}_results.json").write_text(json.dumps(summary, indent=1))
    print(f"saved: {OUT}/{name}_policy.pt, _policy.onnx, _paths.png, _results.json")


if __name__ == "__main__":
    main()
