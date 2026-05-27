#!/usr/bin/env python3
"""
Batch runner for MetaWorld experiments based on an experiment plan YAML.

This script does NOT call any LLM.
It reads:
  experiments/metaworld_structmem_plan.yaml
and executes batches by spawning:
  scripts/run_metaworld_mem.py

Why subprocess:
  - Keeps the single-run script as the canonical runner.
  - Avoids duplicating env/policy/logging logic.

Usage:
  # Full batch from plan
  python3 scripts/run_metaworld_batch.py \
    --plan experiments/metaworld_structmem_plan.yaml \
    --policies no_mem,text_buf,struct_mem

  # Small-pool validation (4 tasks, 60 eps)
  python3 scripts/run_metaworld_batch.py \
    --small-pool \
    --policies no_mem,text_buf,struct_mem

Then aggregate:
  python3 scripts/aggregate_runs.py --runs runs --out artifacts/metaworld_results.csv
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SMALL_POOL_TASKS = ["reach-v3", "push-v3", "door-open-v3", "pick-place-v3"]
SMALL_POOL_SEEDS = [0, 1, 2, 3, 4]
SMALL_POOL_EPISODES = 60


def _normalize_task_name(task_name: str) -> str:
    t = (task_name or "").strip()
    if t.endswith("-v2"):
        return t[:-3] + "-v3"
    return t


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception:
        raise SystemExit(
            "Missing dependency: pyyaml\n"
            "Install with:\n"
            "  python3 -m pip install pyyaml\n"
        )
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"Invalid plan YAML: {path}")
    return data  # type: ignore[return-value]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--plan", default="experiments/metaworld_structmem_plan.yaml")
    p.add_argument("--policies", default="no_mem,text_buf,struct_mem")
    p.add_argument("--run-dir", default="runs/batch")
    p.add_argument("--limit-tasks", type=int, default=0, help="If >0, only run first N tasks per split (smoke)")
    p.add_argument("--dry-run", action="store_true", help="Print commands without running")
    p.add_argument("--small-pool", action="store_true", help="Use small-pool validation: 4 tasks, 60 eps, 5 seeds")
    p.add_argument("--controller", choices=["PD", "PI"], default="PD", help="Controller type")
    p.add_argument("--flash-every", type=int, default=20, help="Flash goal every N steps")
    p.add_argument("--flash-len", type=int, default=1, help="Flash goal for N steps each cycle")
    p.add_argument("--po-noise-std", type=float, default=0.01)
    p.add_argument("--max-retries", type=int, default=3, help="Max grasp retry attempts (0=disable retry)")
    p.add_argument("--no-perturbation", action="store_true", help="Disable XY perturbation on retry")
    p.add_argument("--full-observable", action="store_true", help="Make goal always visible (full observable control)")
    return p.parse_args()


def _pick_policies(s: str) -> list[str]:
    out = []
    for x in (s or "").split(","):
        x = x.strip()
        if not x:
            continue
        out.append(x)
    return out or ["no_mem", "text_buf", "struct_mem"]


def _extract_tasks(plan: dict[str, Any]) -> dict[str, list[str]]:
    tasks = plan.get("tasks") or {}
    if not isinstance(tasks, dict):
        raise SystemExit("plan.tasks must be a mapping")
    train = tasks.get("train") or []
    test = tasks.get("test_unseen") or []
    if not isinstance(train, list) or not isinstance(test, list):
        raise SystemExit("plan.tasks.train/test_unseen must be lists")
    return {
        "train": [_normalize_task_name(str(x)) for x in train],
        "test_unseen": [_normalize_task_name(str(x)) for x in test],
    }


def _extract_eval(plan: dict[str, Any]) -> tuple[list[int], int]:
    evaluation = plan.get("evaluation") or {}
    if not isinstance(evaluation, dict):
        raise SystemExit("plan.evaluation must be a mapping")
    seeds = evaluation.get("seeds") or [0]
    eps = evaluation.get("episodes_per_task") or 10
    if not isinstance(seeds, list):
        raise SystemExit("plan.evaluation.seeds must be a list")
    seeds_i = [int(s) for s in seeds]
    return seeds_i, int(eps)


def _extract_po(plan: dict[str, Any]) -> float:
    env = plan.get("environment") or {}
    po = (env.get("partial_observability") if isinstance(env, dict) else {}) or {}
    if not isinstance(po, dict):
        po = {}
    noise = po.get("noise") or {}
    if not isinstance(noise, dict):
        noise = {}
    return float(noise.get("std") or 0.0)


def _run_one(cmd: list[str], *, dry_run: bool) -> int:
    print("$ " + " ".join(cmd))
    if dry_run:
        return 0
    p = subprocess.run(cmd, check=False)
    return int(p.returncode)


def main() -> int:
    args = _parse_args()
    policies = _pick_policies(str(args.policies))
    controller = str(args.controller)
    flash_every = int(args.flash_every)
    flash_len = int(args.flash_len)
    po_std = float(args.po_noise_std)
    max_retries = int(args.max_retries)
    no_pert = bool(args.no_perturbation)
    full_obs = bool(args.full_observable)

    if args.small_pool:
        tasks_dict: dict[str, list[str]] = {"train": SMALL_POOL_TASKS, "test_unseen": []}
        seeds = SMALL_POOL_SEEDS
        episodes = SMALL_POOL_EPISODES
    else:
        plan_path = Path(args.plan)
        plan = _load_yaml(plan_path)
        tasks_dict = _extract_tasks(plan)
        seeds, episodes = _extract_eval(plan)
        po_std = _extract_po(plan)

        limit = int(args.limit_tasks)
        if limit > 0:
            tasks_dict = {k: v[:limit] for k, v in tasks_dict.items()}

    batch_id = f"mw-batch-{time.strftime('%Y%m%d-%H%M%S', time.localtime())}"
    if args.small_pool:
        batch_id = f"mw-smallpool-{time.strftime('%Y%m%d-%H%M%S', time.localtime())}"
    run_root = Path(args.run_dir) / batch_id
    run_root.mkdir(parents=True, exist_ok=True)

    (run_root / "batch_config.txt").write_text(
        "\n".join(
            [
                f"mode={'small_pool' if args.small_pool else 'plan'}",
                f"policies={policies}",
                f"seeds={seeds}",
                f"episodes_per_task={episodes}",
                f"controller={controller}",
                f"po_noise_std={po_std}",
                f"flash_every={flash_every}",
                f"flash_len={flash_len}",
                f"tasks.train={tasks_dict['train']}",
                f"tasks.test_unseen={tasks_dict['test_unseen']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    script = Path("scripts/run_metaworld_mem.py")
    if not script.exists():
        raise SystemExit(f"Missing runner script: {script}")

    total_runs = 0
    completed_runs = 0
    failed_runs = 0

    for split_name, split_tasks in tasks_dict.items():
        for task in split_tasks:
            for pol in policies:
                for seed in seeds:
                    total_runs += 1

    for split_name, split_tasks in tasks_dict.items():
        for task in split_tasks:
            for pol in policies:
                for seed in seeds:
                    cmd = [
                        sys.executable,
                        str(script),
                        "--task", task,
                        "--episodes", str(episodes),
                        "--seed", str(seed),
                        "--policy", pol,
                        "--controller", controller,
                        "--po-noise-std", str(po_std),
                        "--flash-every", str(flash_every),
                        "--flash-len", str(flash_len),
                        "--max-retries", str(max_retries),
                        "--run-dir", str(run_root),
                    ]
                    if no_pert:
                        cmd.append("--no-perturbation")
                    if full_obs:
                        cmd.append("--full-observable")
                    rc = _run_one(cmd, dry_run=bool(args.dry_run))
                    if rc != 0:
                        failed_runs += 1
                        print(f"FAILED rc={rc} split={split_name} task={task} policy={pol} seed={seed}")
                    else:
                        completed_runs += 1

    print(f"\nDone. Runs written under: {run_root}")
    print(f"Completed: {completed_runs}/{total_runs}, Failed: {failed_runs}/{total_runs}")
    print("Next: aggregate")
    print(f"  {sys.executable} scripts/aggregate_runs.py --runs {run_root} --out artifacts/metaworld_results.csv")
    return 1 if failed_runs > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
