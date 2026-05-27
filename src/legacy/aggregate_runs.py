#!/usr/bin/env python3
"""
Aggregate MetaWorld run summaries to CSV.

Reads:
  <runs_root>/**/summary.json
Writes:
  CSV with one row per run directory.

Usage:
  python3 scripts/aggregate_runs.py --runs runs --out artifacts/results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs", help="Root runs directory")
    p.add_argument("--out", default="artifacts/metaworld_results.csv", help="Output CSV path")
    return p.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid json object: {path}")
    return data  # type: ignore[return-value]


def main() -> int:
    args = _parse_args()
    runs_root = Path(args.runs)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summaries = sorted(runs_root.glob("**/summary.json"))
    rows: list[dict[str, Any]] = []
    for sp in summaries:
        try:
            s = _read_json(sp)
        except Exception:
            continue
        config_path = sp.parent / "config.json"
        condition = "PO"
        if config_path.exists():
            try:
                config = _read_json(config_path)
                if config.get("full_observable", False):
                    condition = "FO"
            except Exception:
                pass

        row = {
            "run_dir": str(sp.parent),
            "exp_id": s.get("exp_id", ""),
            "task": s.get("task", ""),
            "policy": s.get("policy", ""),
            "condition": condition,
            "controller": s.get("controller", "PD"),
            "episodes": s.get("episodes", ""),
            "success_rate": s.get("success_rate", ""),
            "avg_steps": s.get("avg_steps", ""),
            "avg_reward": s.get("avg_reward", ""),
            "avg_grasp_attempts": s.get("avg_grasp_attempts", 0),
            "avg_grasp_success": s.get("avg_grasp_success", 0),
            "avg_retry_count": s.get("avg_retry_count", 0),
            "flash_every": s.get("flash_every", 0),
            "flash_len": s.get("flash_len", 0),
            "wall_time_sec": s.get("wall_time_sec", ""),
            "generated": s.get("generated", ""),
        }
        rows.append(row)

    fieldnames = [
        "run_dir",
        "exp_id",
        "task",
        "policy",
        "condition",
        "controller",
        "episodes",
        "success_rate",
        "avg_steps",
        "avg_reward",
        "avg_grasp_attempts",
        "avg_grasp_success",
        "avg_retry_count",
        "flash_every",
        "flash_len",
        "wall_time_sec",
        "generated",
    ]

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote: {out_path} (rows={len(rows)})")

    if rows:
        by_task_policy: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for r in rows:
            key = (r["task"], r["policy"])
            by_task_policy.setdefault(key, []).append(r)

        print("\n=== Success Rate by Task × Policy ===")
        tasks_seen: list[str] = []
        policies_seen: list[str] = []
        for (task, policy), group in sorted(by_task_policy.items()):
            if task not in tasks_seen:
                tasks_seen.append(task)
            if policy not in policies_seen:
                policies_seen.append(policy)

        header = f"{'task':<28} " + " ".join(f"{p:<12}" for p in policies_seen)
        print(header)
        for task in tasks_seen:
            line = f"{task:<28} "
            for policy in policies_seen:
                group = by_task_policy.get((task, policy), [])
                if group:
                    avg_sr = sum(g["success_rate"] for g in group) / len(group)
                    line += f"{avg_sr:<12.4f} "
                else:
                    line += f"{'—':<12} "
            print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
