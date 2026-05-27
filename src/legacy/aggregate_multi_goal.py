#!/usr/bin/env python3
"""
Aggregate multi-goal run summaries into a single CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std_sample(xs: list[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _read_json(fp: Path) -> dict:
    return json.loads(fp.read_text())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=str, default="runs/multi_goal")
    p.add_argument("--out-dir", type=str, default="artifacts/multi_goal")
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, object]] = []
    for summary_fp in sorted(input_dir.glob("**/summary.json")):
        try:
            summary = _read_json(summary_fp)
            config = _read_json(summary_fp.parent / "config.json")
        except Exception:
            continue
        run_rows.append(
            {
                "task": config.get("task", ""),
                "policy": config.get("policy", ""),
                "seed": int(config.get("seed", -1)),
                "episodes": int(config.get("episodes", 0)),
                "flash_every": int(config.get("flash_every", 0)),
                "flash_len": int(config.get("flash_len", 0)),
                "success_rate": float(summary.get("success_rate", 0.0)),
                "mean_reward": float(summary.get("avg_reward", summary.get("mean_reward", 0.0))),
                "run_path": str(summary_fp.parent),
            }
        )

    runs_csv = out_dir / "multi_goal_runs.csv"
    with runs_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "task",
                "policy",
                "seed",
                "episodes",
                "flash_every",
                "flash_len",
                "success_rate",
                "mean_reward",
                "run_path",
            ],
        )
        w.writeheader()
        w.writerows(run_rows)

    grouped_sr: dict[tuple[str, str], list[float]] = defaultdict(list)
    grouped_mr: dict[tuple[str, str], list[float]] = defaultdict(list)
    grouped_meta: dict[tuple[str, str], dict[str, int]] = {}
    for row in run_rows:
        key = (str(row["task"]), str(row["policy"]))
        grouped_sr[key].append(float(row["success_rate"]))
        grouped_mr[key].append(float(row["mean_reward"]))
        grouped_meta[key] = {
            "episodes": int(row["episodes"]),
            "flash_every": int(row["flash_every"]),
            "flash_len": int(row["flash_len"]),
        }

    summary_rows: list[dict[str, object]] = []
    for (task, policy), xs in sorted(grouped_sr.items()):
        meta = grouped_meta[(task, policy)]
        mr = grouped_mr[(task, policy)]
        summary_rows.append(
            {
                "task": task,
                "policy": policy,
                "n_seeds": len(xs),
                "episodes_per_seed": meta["episodes"],
                "flash_every": meta["flash_every"],
                "flash_len": meta["flash_len"],
                "success_rate_mean": _mean(xs),
                "success_rate_std": _std_sample(xs),
                "mean_reward_mean": _mean(mr),
                "mean_reward_std": _std_sample(mr),
            }
        )

    summary_csv = out_dir / "multi_goal_summary.csv"
    with summary_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "task",
                "policy",
                "n_seeds",
                "episodes_per_seed",
                "flash_every",
                "flash_len",
                "success_rate_mean",
                "success_rate_std",
                "mean_reward_mean",
                "mean_reward_std",
            ],
        )
        w.writeheader()
        w.writerows(summary_rows)

    print(f"Wrote: {runs_csv}")
    print(f"Wrote: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

