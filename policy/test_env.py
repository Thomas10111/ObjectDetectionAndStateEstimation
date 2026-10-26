"""Check the environment and measure the rule based controller in it.

    source ~/.venvs/odse_policy/bin/activate
    python test_env.py                 # 200 episodes
    python test_env.py --episodes 1000

Prints outcome statistics and writes out/expert_paths.png with a few driven
paths seen from above.
"""

import argparse
import time
from pathlib import Path

from gymnasium.utils.env_checker import check_env

from evaluate import evaluate, plot_paths, summarize
from expert import expert_action
from lego_car_env import LegoCarEnv

OUT = Path(__file__).parent / "out"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--scenario", default="ahead", choices=LegoCarEnv.SCENARIOS)
    args = parser.parse_args()

    check_env(LegoCarEnv(), skip_render_check=True)
    print("Gymnasium env_checker: ok")

    t0 = time.perf_counter()
    results = evaluate(expert_action, range(args.episodes), LegoCarEnv(scenario=args.scenario))
    elapsed = time.perf_counter() - t0
    steps = sum(r["steps"] for r in results)
    print(f"{args.episodes} episodes, {steps} steps in {elapsed:.1f} s ({steps / elapsed:.0f} steps/s)")
    summarize(results, f"rule based controller ({args.scenario})")

    plot_paths(results, OUT / f"expert_paths_{args.scenario}.png", f"Rule based controller ({args.scenario}): path of the car front, car at the end")
    print(f"plot: {OUT / f'expert_paths_{args.scenario}.png'}")


if __name__ == "__main__":
    main()
