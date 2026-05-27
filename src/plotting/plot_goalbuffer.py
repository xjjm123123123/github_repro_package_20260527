"""Aggregate GoalBuffer eval JSON files and optionally render a grouped bar chart."""

from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default="results/baselines/goal_buffer")
    p.add_argument("--csv-out", default="results/baselines/goalbuffer_results.csv")
    p.add_argument("--fig-out", default="figures/figure4_goalbuffer.png")
    args = p.parse_args()

    rows = []
    for path in sorted(Path(args.input_dir).glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            rows.append(json.load(f))

    csv_out = Path(args.csv_out)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = sorted({k for row in rows for k in row.keys()})
        with csv_out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in rows:
                w.writerow(row)

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.get("task", ""), row.get("condition", ""), row.get("memory_mode", "goal_buffer"))].append(
            float(row.get("success_rate", 0) or 0)
        )

    tasks = sorted({k[0] for k in grouped})
    conditions = sorted({k[1] for k in grouped})
    if tasks and conditions:
        fig, ax = plt.subplots(figsize=(1.5 * len(tasks) + 2, 4))
        width = 0.8 / max(1, len(conditions))
        xs = list(range(len(tasks)))
        for idx, cond in enumerate(conditions):
            vals = []
            for task in tasks:
                runs = grouped.get((task, cond, "goal_buffer"), [0.0])
                vals.append(sum(runs) / max(1, len(runs)))
            ax.bar([x + idx * width for x in xs], vals, width=width, label=cond)
        ax.set_xticks([x + width * (len(conditions) - 1) / 2 for x in xs], tasks, rotation=20, ha="right")
        ax.set_ylabel("success rate")
        ax.set_ylim(0, 1)
        ax.legend()
        fig.tight_layout()
        fig_out = Path(args.fig_out)
        fig_out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_out, dpi=300)
        print(f"Wrote: {fig_out}")
    print(f"Wrote: {csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
