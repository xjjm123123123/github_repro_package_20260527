#!/usr/bin/env python3
"""
Aggregate SAC baseline eval logs into a single CSV.

Input:
  - artifacts/sac_baselines/*__FO.eval.json
  - artifacts/sac_baselines/*__WeakPO.eval.json
  - artifacts/sac_baselines/*__StrongPO.eval.json

Output:
  - artifacts/strong_po/strong_po_eval_runs.csv (per-seed)
  - artifacts/strong_po/strong_po_eval_summary.csv (mean/std/n)
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


def _iter_eval_json(input_dir: Path) -> list[Path]:
    return sorted(input_dir.glob("*__*.eval.json"))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=str, default="artifacts/sac_baselines")
    p.add_argument("--out-dir", type=str, default="artifacts/strong_po")
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, object]] = []
    for fp in _iter_eval_json(input_dir):
        try:
            data = json.loads(fp.read_text())
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        # Required keys produced by train_sac_baseline.py
        task = str(data.get("task", ""))
        policy = str(data.get("policy", ""))
        condition = str(data.get("condition", ""))
        seed = int(data.get("seed", -1))
        success_rate = float(data.get("success_rate", 0.0))
        mean_reward = float(data.get("mean_reward", 0.0))
        flash_every = int(data.get("flash_every", 0))
        flash_len = int(data.get("flash_len", 0))
        episodes = int(data.get("episodes", 0))

        if not task or not policy or seed < 0 or not condition:
            continue

        run_rows.append(
            {
                "task": task,
                "policy": policy,
                "condition": condition,
                "seed": seed,
                "episodes": episodes,
                "flash_every": flash_every,
                "flash_len": flash_len,
                "success_rate": success_rate,
                "mean_reward": mean_reward,
                "source_file": str(fp),
            }
        )

    runs_csv = out_dir / "strong_po_eval_runs.csv"
    with runs_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "task",
                "policy",
                "condition",
                "seed",
                "episodes",
                "flash_every",
                "flash_len",
                "success_rate",
                "mean_reward",
                "source_file",
            ],
        )
        w.writeheader()
        w.writerows(run_rows)

    grouped_sr: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    grouped_mr: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    grouped_meta: dict[tuple[str, str, str], dict[str, int]] = {}

    for r in run_rows:
        key = (str(r["task"]), str(r["policy"]), str(r["condition"]))
        grouped_sr[key].append(float(r["success_rate"]))
        grouped_mr[key].append(float(r["mean_reward"]))
        grouped_meta[key] = {
            "episodes": int(r["episodes"]),
            "flash_every": int(r["flash_every"]),
            "flash_len": int(r["flash_len"]),
        }

    summary_rows: list[dict[str, object]] = []
    for (task, policy, condition), xs in sorted(grouped_sr.items()):
        meta = grouped_meta.get((task, policy, condition), {})
        ms = grouped_mr[(task, policy, condition)]
        summary_rows.append(
            {
                "task": task,
                "policy": policy,
                "condition": condition,
                "n_seeds": len(xs),
                "episodes_per_seed": int(meta.get("episodes", 0)),
                "flash_every": int(meta.get("flash_every", 0)),
                "flash_len": int(meta.get("flash_len", 0)),
                "success_rate_mean": _mean(xs),
                "success_rate_std": _std_sample(xs),
                "mean_reward_mean": _mean(ms),
                "mean_reward_std": _std_sample(ms),
            }
        )

    summary_csv = out_dir / "strong_po_eval_summary.csv"
    with summary_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "task",
                "policy",
                "condition",
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

