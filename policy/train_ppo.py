"""Train the driving network with reinforcement learning (PPO), without a teacher.

    source ~/.venvs/odse_policy/bin/activate
    python train_ppo.py                        # 1 million steps
    python train_ppo.py --steps 3000000
    python train_ppo.py --scenario mixed --name ppo_search   # also learn to search
    tensorboard --logdir out/tb                # learning curves, in a second terminal

The network starts random and learns only from the rewards of LegoCarEnv
(getting closer, keeping the object centered, arriving, not crashing). PPO
trains two networks: the actor, which drives (same size as the imitation
network: 9 inputs, 2 x 64, 2 outputs), and the critic, which estimates how
good a situation is and is only needed during training.

Every --eval-every steps the current actor drives 100 fixed test episodes; the
best version is kept. At the end it is compared with the rule based controller
and the imitation network on the same 500 unseen episodes as train_bc.py.

Writes to out/:
    ppo_best.zip      best Stable-Baselines3 model (actor and critic)
    ppo_policy.onnx   the actor only, deterministic: input "obs" float32[N, 9],
                      output "action" float32[N, 2], clipped to -1 .. 1
    ppo_paths.png     a few episodes driven by the actor
    ppo_results.json  size, training time and final test results
    tb/               TensorBoard logs

--hidden sets the actor's hidden layers (64,64 by default). The critic keeps
--critic (64,64): it only runs during training, so its size costs nothing on
the Orin, and a large critic gives the small actor good advantage estimates.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from torch import nn

from evaluate import evaluate, plot_paths, summarize
from expert import expert_action
from lego_car_env import LegoCarEnv
from train_bc import EVAL_SEEDS, PolicyNet, as_controller, count_parameters, export_onnx, parse_hidden

OUT = Path(__file__).parent / "out"
TEST_SEEDS = range(200_000, 200_100)  # progress checks during training


class DeterministicActor(nn.Module):
    """The PPO actor as a plain network: mean action, clipped to -1 .. 1.

    During training PPO adds random exploration around this mean; when
    driving for real only the mean is used.
    """

    def __init__(self, model: PPO) -> None:
        super().__init__()
        policy = model.policy
        self.features = policy.features_extractor
        self.mlp = policy.mlp_extractor
        self.action_net = policy.action_net

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        latent = self.mlp.forward_actor(self.features(obs))
        return torch.clamp(self.action_net(latent), -1.0, 1.0)


def ppo_controller(model: PPO):
    def act(obs: np.ndarray) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=True)
        return np.clip(action, -1, 1).astype(np.float32)

    return act


class ProgressCheck(BaseCallback):
    """Every few steps: drive the test episodes, print, keep the best model."""

    def __init__(self, every: int, path: Path, scenario: str) -> None:
        super().__init__()
        self.every, self.path, self.scenario = every, path, scenario
        self.next_check, self.best, self.t0 = every, -1.0, time.perf_counter()

    def _on_step(self) -> bool:
        if self.num_timesteps < self.next_check:
            return True
        self.next_check += self.every
        results = evaluate(ppo_controller(self.model), TEST_SEEDS, LegoCarEnv(scenario=self.scenario))
        arrived = np.mean([r["outcome"] == "arrived" for r in results])
        reward = np.mean([r["reward"] for r in results])
        self.logger.record("check/arrived", arrived)
        self.logger.record("check/reward", reward)
        mark = ""
        if arrived > self.best or (arrived == self.best and reward > self.best_reward):
            self.best, self.best_reward = arrived, reward
            self.model.save(self.path)
            mark = "  (best so far, saved)"
        elapsed = time.perf_counter() - self.t0
        print(
            f"{self.num_timesteps:9,d} steps, {elapsed:5.0f} s: arrived {100 * arrived:5.1f}%, "
            f"reward {reward:6.2f}{mark}",
            flush=True,
        )
        return True

    best_reward = -np.inf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--envs", type=int, default=8, help="simulators running side by side")
    parser.add_argument("--eval-every", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenario", default="ahead", choices=LegoCarEnv.SCENARIOS)
    parser.add_argument("--name", default="ppo", help="output file prefix in out/")
    parser.add_argument("--hidden", default="64,64", help="actor hidden layer widths, for example 16,16 or 8")
    parser.add_argument("--critic", default="64,64", help="critic hidden layer widths")
    parser.add_argument("--threads", type=int, default=4, help="PyTorch threads")
    args = parser.parse_args()
    hidden, critic = parse_hidden(args.hidden), parse_hidden(args.critic)

    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(args.threads)
    envs = make_vec_env(LegoCarEnv, n_envs=args.envs, seed=args.seed, env_kwargs={"scenario": args.scenario})
    model = PPO(
        "MlpPolicy",
        envs,
        policy_kwargs={"net_arch": {"pi": list(hidden), "vf": list(critic)}, "activation_fn": nn.Tanh},
        n_steps=1024,  # per simulator, so 8192 steps per update
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        seed=args.seed,
        device="cpu",
        tensorboard_log=str(OUT / "tb"),
        verbose=0,
    )
    print(f"PPO: {args.steps:,d} steps, {args.envs} simulators, scenario {args.scenario}, check every {args.eval_every:,d} steps")
    t0 = time.perf_counter()
    model.learn(args.steps, callback=ProgressCheck(args.eval_every, OUT / f"{args.name}_best", args.scenario), tb_log_name=args.name)
    train_seconds = time.perf_counter() - t0
    print(f"training took {train_seconds:.0f} s")

    best = PPO.load(OUT / f"{args.name}_best", device="cpu")
    bc_path = OUT / "bc_policy.pt"
    finals = {}
    for scenario in ("ahead", "anywhere"):
        env = LegoCarEnv(scenario=scenario)
        print(f"--- final evaluation, {len(EVAL_SEEDS)} unseen episodes, scenario {scenario}")
        summarize(evaluate(expert_action, EVAL_SEEDS, env), "rule based controller")
        if bc_path.exists():
            bc = PolicyNet()
            bc.load_state_dict(torch.load(bc_path))
            summarize(evaluate(as_controller(bc), EVAL_SEEDS, env), "imitation network    ")
        results = evaluate(ppo_controller(best), EVAL_SEEDS, env)
        finals[scenario] = summarize(results, f"{args.name} network".ljust(21))
        plot_paths(results, OUT / f"{args.name}_paths_{scenario}.png", f"Network ({args.name}, {scenario}): path of the car front, car at the end")

    actor = DeterministicActor(best)
    diff = export_onnx(actor, OUT / f"{args.name}_policy.onnx")
    print(f"ONNX export: largest difference to PyTorch on 1000 inputs {diff:.2e}")
    summary = {"name": args.name, "method": "ppo", "scenario": args.scenario, "seed": args.seed,
               "steps": args.steps, "hidden": hidden, "critic": critic, "parameters": count_parameters(actor.mlp.policy_net) + count_parameters(actor.action_net),
               "train_seconds": train_seconds, "onnx_diff": diff, "eval": finals}
    (OUT / f"{args.name}_results.json").write_text(json.dumps(summary, indent=1))
    print(f"saved: {OUT / f'{args.name}_best.zip'}, {OUT / f'{args.name}_policy.onnx'}, {OUT}/{args.name}_paths_*.png")


if __name__ == "__main__":
    main()
