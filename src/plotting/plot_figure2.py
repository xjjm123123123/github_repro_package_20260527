"""Generate a simple condition-by-task heatmap from aggregate CSV results."""

from __future__ import annotations
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="results/main/table1_metaworld.csv")
    p.add_argument("--output", default="figures/figure2_goal_dependency.png")
    p.add_argument("--policy", default="no_mem")
    args = p.parse_args()

    rows = list(csv.DictReader(Path(args.input).open("r", encoding="utf-8")))
    grouped = defaultdict(list)
    tasks = []
    conditions = []
    for row in rows:
        if row.get("policy") != args.policy:
            continue
        task = row.get("task", "")
        cond = row.get("condition", "")
        if task not in tasks:
            tasks.append(task)
        if cond not in conditions:
            conditions.append(cond)
        try:
            grouped[(task, cond)].append(float(row.get("success_rate", 0) or 0))
        except ValueError:
            grouped[(task, cond)].append(0.0)

    if not tasks or not conditions:
        raise SystemExit("No matching rows found for requested policy.")

    matrix = []
    for task in tasks:
        row_vals = []
        for cond in conditions:
            vals = grouped.get((task, cond), [0.0])
            row_vals.append(sum(vals) / max(1, len(vals)))
        matrix.append(row_vals)

    fig, ax = plt.subplots(figsize=(1.6 * len(conditions) + 2, 0.5 * len(tasks) + 2))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(conditions)), conditions)
    ax.set_yticks(range(len(tasks)), tasks)
    for i in range(len(tasks)):
        for j in range(len(conditions)):
            ax.text(j, i, f"{matrix[i][j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(f"{args.policy} success rates by task and condition")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="success rate")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
