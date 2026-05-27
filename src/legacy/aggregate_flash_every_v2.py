#!/usr/bin/env python3
"""Aggregate flash_every ablation results from a SPECIFIED directory only.

P0-2 Data Governance: paper data only comes from artifacts/, never from
scanning runs/ broadly.  This script reads ONLY from the --input-dir
directory and validates config.json for every run before inclusion.

Reads:
  <input-dir>/{task}/{policy}/flash{flash_every}/seed{seed}/config.json
  <input-dir>/{task}/{policy}/flash{flash_every}/seed{seed}/summary.json

Writes:
  <output-dir>/flash_every_results_sweep_door.csv
  <output-dir>/flash_every_results_combined.csv  (if --merge-existing)

Usage:
  python3 scripts/aggregate_flash_every_v2.py
  python3 scripts/aggregate_flash_every_v2.py --input-dir runs/batch/flash_every_sweep_door/
  python3 scripts/aggregate_flash_every_v2.py --merge-existing
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate flash_every ablation results (scoped read, no runs/ scan)"
    )
    p.add_argument(
        "--input-dir",
        default="runs/batch/flash_every_sweep_door/",
        help="Specific batch output directory to read from (default: runs/batch/flash_every_sweep_door/)",
    )
    p.add_argument(
        "--output-dir",
        default="artifacts/flash_every_ablation/",
        help="Output directory for CSV artifacts (default: artifacts/flash_every_ablation/)",
    )
    p.add_argument(
        "--merge-existing",
        action="store_true",
        help="Merge with existing flash_every_results_5seeds.csv into flash_every_results_combined.csv",
    )
    return p.parse_args()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        return data
    except Exception:
        return None


def _collect_runs(input_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    validated = 0
    skipped_config = 0
    skipped_summary = 0

    for config_path in sorted(input_dir.glob("*/*/*/*/config.json")):
        parts = config_path.relative_to(input_dir).parts
        if len(parts) < 5:
            skipped_config += 1
            continue

        task_dir = parts[0]
        policy_dir = parts[1]
        flash_dir = parts[2]
        seed_dir = parts[3]

        config = _read_json(config_path)
        if config is None:
            skipped_config += 1
            continue

        cfg_task = config.get("task", "")
        cfg_policy = config.get("policy", "")
        cfg_flash = config.get("flash_every")
        cfg_seed = config.get("seed")

        if cfg_task != task_dir or cfg_policy != policy_dir:
            skipped_config += 1
            continue

        if not flash_dir.startswith("flash"):
            skipped_config += 1
            continue
        try:
            flash_every_from_dir = int(flash_dir.replace("flash", ""))
        except ValueError:
            skipped_config += 1
            continue

        if cfg_flash is not None and cfg_flash != flash_every_from_dir:
            skipped_config += 1
            continue

        if not seed_dir.startswith("seed"):
            skipped_config += 1
            continue
        try:
            seed_from_dir = int(seed_dir.replace("seed", ""))
        except ValueError:
            skipped_config += 1
            continue

        if cfg_seed is not None and cfg_seed != seed_from_dir:
            skipped_config += 1
            continue

        summary_path = None
        for candidate in sorted(config_path.parent.iterdir()):
            if candidate.is_dir() and candidate.name.startswith("mw-"):
                sp = candidate / "summary.json"
                if sp.exists():
                    summary_path = sp
                    break
        if summary_path is None:
            summary_path = config_path.parent / "summary.json"
        summary = _read_json(summary_path)
        if summary is None:
            skipped_summary += 1
            continue

        success_rate = summary.get("success_rate")
        if success_rate is None:
            skipped_summary += 1
            continue

        validated += 1
        rows.append({
            "task": cfg_task,
            "policy": cfg_policy,
            "flash_every": flash_every_from_dir,
            "seed": seed_from_dir,
            "success_rate": float(success_rate),
            "avg_steps": summary.get("avg_steps", 0),
            "avg_reward": summary.get("avg_reward", 0),
        })

    print(f"Run collection: validated={validated}, skipped_config={skipped_config}, skipped_summary={skipped_summary}")
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for r in rows:
        key = (r["task"], r["policy"], r["flash_every"])
        grouped[key].append(r["success_rate"])

    result: list[dict[str, Any]] = []
    for (task, policy, flash_every), srs in sorted(grouped.items()):
        n = len(srs)
        mean = sum(srs) / n
        std = math.sqrt(sum((x - mean) ** 2 for x in srs) / n) if n > 0 else 0.0
        std_ddof1 = math.sqrt(sum((x - mean) ** 2 for x in srs) / (n - 1)) if n > 1 else 0.0
        result.append({
            "task": task,
            "policy": policy,
            "flash_every": flash_every,
            "mean": round(mean, 6),
            "std": round(std_ddof1, 6),
            "n_seeds": n,
        })
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote: {path} (rows={len(rows)})")


def _merge_existing(output_dir: Path, new_rows: list[dict[str, Any]]) -> None:
    existing_path = output_dir / "flash_every_results_5seeds.csv"
    if not existing_path.exists():
        print(f"Merge skipped: {existing_path} not found")
        return

    existing_rows: list[dict[str, Any]] = []
    with existing_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing_rows.append({
                "task": row["task"],
                "policy": row["policy"],
                "flash_every": int(row["flash_every"]),
                "mean": float(row["mean"]),
                "std": float(row["std"]),
                "n_seeds": int(row["n_seeds"]),
            })

    combined: dict[tuple[str, str, int], dict[str, Any]] = {}
    for r in existing_rows:
        key = (r["task"], r["policy"], r["flash_every"])
        combined[key] = r
    for r in new_rows:
        key = (r["task"], r["policy"], r["flash_every"])
        if key in combined:
            print(f"  Merge conflict for {key}: keeping existing, skipping new")
        else:
            combined[key] = r

    sorted_rows = [combined[k] for k in sorted(combined.keys())]
    combined_path = output_dir / "flash_every_results_combined.csv"
    _write_csv(combined_path, sorted_rows, ["task", "policy", "flash_every", "mean", "std", "n_seeds"])


def _print_summary(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No aggregated rows to summarize.")
        return

    tasks = sorted(set(r["task"] for r in rows))
    policies = sorted(set(r["policy"] for r in rows))
    flash_values = sorted(set(r["flash_every"] for r in rows))

    print(f"\n=== Flash-Every Ablation Summary ===")
    print(f"Tasks: {tasks}")
    print(f"Policies: {policies}")
    print(f"Flash-every values: {flash_values}")
    print()

    for task in tasks:
        print(f"--- {task} ---")
        header = f"{'policy':<14} " + " ".join(f"fe={fe:<8}" for fe in flash_values)
        print(header)
        for policy in policies:
            parts = []
            for fe in flash_values:
                match = [r for r in rows if r["task"] == task and r["policy"] == policy and r["flash_every"] == fe]
                if match:
                    r = match[0]
                    parts.append(f"{r['mean']:.3f}±{r['std']:.3f}(n={r['n_seeds']})")
                else:
                    parts.append("—")
            line = f"{policy:<14} " + " ".join(f"{p:<14}" for p in parts)
            print(line)
        print()


def main() -> int:
    args = _parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        print(f"ERROR: input directory does not exist: {input_dir}")
        print("This script only reads from the specified --input-dir. It does NOT scan runs/ broadly.")
        return 1

    print(f"Input dir:  {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Merge existing: {args.merge_existing}")
    print()

    rows = _collect_runs(input_dir)
    if not rows:
        print("No valid runs found in the specified input directory.")
        return 1

    aggregated = _aggregate(rows)

    csv_path = output_dir / "flash_every_results_sweep_door.csv"
    _write_csv(csv_path, aggregated, ["task", "policy", "flash_every", "mean", "std", "n_seeds"])

    if args.merge_existing:
        _merge_existing(output_dir, aggregated)

    _print_summary(aggregated)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
