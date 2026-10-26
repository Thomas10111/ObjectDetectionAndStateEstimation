"""Table and plot of the network size sweep (sweep_network_size.sh).

    source ~/.venvs/odse_policy/bin/activate
    python summarize_sweep.py

Reads out/*_results.json and writes out/network_size.png and
out/network_size.md (the table, also printed).
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent / "out"
EXPERIMENTS = {  # file prefix: (label, scenario evaluated)
    "bc": ("Imitation, ball ahead", "ahead"),
    "ppo": ("PPO, ball ahead", "ahead"),
    "search": ("PPO search, ball anywhere", "anywhere"),
}


def load() -> dict:
    """{experiment: {hidden: [result per seed]}}"""
    runs = defaultdict(lambda: defaultdict(list))
    for path in sorted(OUT.glob("*_h*_results.json")):
        result = json.loads(path.read_text())
        prefix = path.name.split("_h")[0]
        if prefix in EXPERIMENTS:
            runs[prefix][tuple(result["hidden"])].append(result)
    return runs


def label(hidden: tuple) -> str:
    return " × ".join(map(str, hidden))


def stats(results: list, scenario: str, key: str) -> tuple[float, float, float]:
    """Mean, min and max over seeds."""
    if key == "collision":
        values = [r["eval"][scenario]["outcomes"]["collision"] for r in results]
    else:
        values = [r["eval"][scenario][key] for r in results]
    values = np.array(values, dtype=float)
    return float(np.nanmean(values)), float(np.nanmin(values)), float(np.nanmax(values))


def main() -> None:
    runs = load()
    lines = ["| Network | Parameters | Experiment | Seeds | Arrived (mean, min to max) | Collisions | Median time to arrive |",
             "|---|---|---|---|---|---|---|"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, (prefix, (title, scenario)) in zip(axes, EXPERIMENTS.items()):
        sizes = sorted(runs[prefix], key=lambda h: runs[prefix][h][0]["parameters"])
        for hidden in sorted(sizes, key=lambda h: -runs[prefix][h][0]["parameters"]):
            results = runs[prefix][hidden]
            arr, lo, hi = stats(results, scenario, "arrived")
            col = stats(results, scenario, "collision")[0]
            time_ = stats(results, scenario, "median_time")[0]
            spread = f"{100 * arr:.1f}%" if len(results) == 1 else f"{100 * arr:.1f}% ({100 * lo:.1f} to {100 * hi:.1f})"
            lines.append(f"| {label(hidden)} | {results[0]['parameters']:,} | {title} | {len(results)} | {spread} "
                         f"| {100 * col:.1f}% | {time_:.1f} s |")
        params = [runs[prefix][h][0]["parameters"] for h in sizes]
        for k, hidden in enumerate(sizes):
            values = [100 * r["eval"][scenario]["arrived"] for r in runs[prefix][hidden]]
            ax.scatter([params[k]] * len(values), values, color="tab:blue", alpha=0.5, s=18)
            ax.annotate(label(hidden), (params[k], np.mean(values)), (4, -12), textcoords="offset points", fontsize=7)
        if sizes:
            ax.plot(params, [100 * stats(runs[prefix][h], scenario, "arrived")[0] for h in sizes], color="tab:blue")
        ax.set_xscale("log")
        ax.set_xlabel("parameters of the driving network")
        ax.set_ylabel("arrived in 500 test episodes (%)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    fig.suptitle("Network size sweep: dots are single training runs (seeds), the line is their mean")
    fig.tight_layout()
    fig.savefig(OUT / "network_size.png", dpi=120)
    table = "\n".join(lines)
    (OUT / "network_size.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
