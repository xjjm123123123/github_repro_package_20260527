#!/usr/bin/env python3
"""Fine-grained retry ablation for pick-place-v3.

Purpose:
  Sweep max_retries over {0, 1, 2, 3, 5} under PO condition
  to measure how retry budget affects text_buf policy performance
  at fine-grained granularity.

Design:
  - Task: pick-place-v3
  - Policy: text_buf
  - max_retries: 0, 1, 2, 3, 5
  - Seeds: 0, 1, 2, 3, 4
  - Episodes: 60
  - Condition: PO only (full_observable=False)
  - Total runs: 5 x 5 = 25

Notes:
  pick-place-v3 uses V1 FSM (fallback from V2), so --force-aware
  is NOT passed.

Output:
  runs/batch/retry_fine_grained/{task}/{policy}/retry{retry}/seed{seed}/

Usage:
  export MUJOCO_GL=disable
  cd /root/AutoResearchClaw
  python3 scripts/run_retry_fine_grained.py
  python3 scripts/run_retry_fine_grained.py --dry-run
  python3 scripts/run_retry_fine_grained.py --parallel 8
"""
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

PYTHON = "python scripts/run_metaworld_mem.py "
TASK = "pick-place-v3"
POLICY = "text_buf"
RETRY_VALUES = [0, 1, 2, 3, 5]
SEEDS = [0, 1, 2, 3, 4]
EPS = 60
BASE_DIR = "runs/batch/retry_fine_grained"


def _run_dir(task, policy, retry, seed):
    return f"{BASE_DIR}/{task}/{policy}/retry{retry}/seed{seed}"


def _build_cmd(task, policy, retry, seed):
    run_dir = _run_dir(task, policy, retry, seed)
    return (
        f"{PYTHON} --task {task} --episodes {EPS} --seed {seed} "
        f"--policy {policy} --controller PD "
        f"--flash-every 20 --flash-len 1 --po-noise-std 0.01 "
        f"--max-retries {retry} "
        f"--run-dir {run_dir}"
    )


def _write_config(task, policy, retry, seed, dry_run=False):
    cfg = {
        "task": task,
        "policy": policy,
        "max_retries": retry,
        "flash_every": 20,
        "flash_len": 1,
        "seed": seed,
        "episodes": EPS,
        "controller": "PD",
        "full_observable": False,
        "po_noise_std": 0.01,
        "fsm_version": 1,
        "force_aware": False,
    }
    run_dir = _run_dir(task, policy, retry, seed)
    cfg_path = os.path.join(run_dir, "config.json")
    if dry_run:
        print(f"  would write {cfg_path}")
        return
    os.makedirs(run_dir, exist_ok=True)
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)


def run_single(cmd, label, dry_run=False):
    if dry_run:
        print(f"  $ {cmd}")
        return 0
    p = subprocess.run(cmd, shell=True, check=False,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        print(f"FAILED: {label} (rc={p.returncode})")
        if p.stderr:
            print(p.stderr.decode(errors="replace")[:500])
    return p.returncode


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fine-grained retry ablation for pick-place-v3")
    parser.add_argument("--dry-run", action="store_true", help="Show commands without executing")
    parser.add_argument("--parallel", type=int, default=4, help="Number of parallel workers (default 4)")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--retries", type=str, default="0,1,2,3,5",
                        help="Comma-separated max_retries values")
    args = parser.parse_args()

    global SEEDS, EPS, RETRY_VALUES
    SEEDS = [int(s) for s in args.seeds.split(",")]
    EPS = args.episodes
    RETRY_VALUES = [int(v) for v in args.retries.split(",")]

    os.environ["MUJOCO_GL"] = "disable"

    jobs = []
    for retry in RETRY_VALUES:
        for seed in SEEDS:
            label = f"{TASK} {POLICY} retry{retry} seed{seed}"
            cmd = _build_cmd(TASK, POLICY, retry, seed)
            jobs.append((cmd, label, retry, seed))

    total = len(jobs)
    print(f"Fine-grained retry ablation: {total} runs")
    print(f"  Task: {TASK}")
    print(f"  Policy: {POLICY}")
    print(f"  max_retries: {RETRY_VALUES}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Episodes: {EPS}")
    print(f"  Condition: PO (full_observable=False)")
    print(f"  Parallel: {args.parallel}")
    print(f"  Output: {BASE_DIR}/")
    print()

    for cmd, label, retry, seed in jobs:
        _write_config(TASK, POLICY, retry, seed, dry_run=args.dry_run)

    t0 = time.time()
    failed = 0
    done = 0

    if args.parallel <= 1 or args.dry_run:
        for cmd, label, *_ in jobs:
            print(f"\n{'='*60}")
            print(f"  {label}")
            print(f"{'='*60}")
            rc = run_single(cmd, label, dry_run=args.dry_run)
            failed += (1 if rc != 0 else 0)
            done += 1
            if done % 5 == 0:
                elapsed = time.time() - t0
                print(f"  Progress: {done}/{total} ({elapsed:.0f}s elapsed)")
    else:
        with ProcessPoolExecutor(max_workers=args.parallel) as pool:
            futures = {}
            for cmd, label, *_ in jobs:
                f = pool.submit(run_single, cmd, label, dry_run=False)
                futures[f] = label
            for f in as_completed(futures):
                label = futures[f]
                try:
                    rc = f.result()
                except Exception as exc:
                    print(f"EXCEPTION: {label} -> {exc}")
                    rc = 1
                failed += (1 if rc != 0 else 0)
                done += 1
                if done % 5 == 0:
                    elapsed = time.time() - t0
                    print(f"  Progress: {done}/{total} ({elapsed:.0f}s elapsed, {failed} failed)")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Fine-grained retry ablation done in {elapsed:.0f}s")
    print(f"  Completed: {done - failed}/{total}, Failed: {failed}/{total}")
    print(f"  Output: {BASE_DIR}/")
    print(f"  Next: python3 scripts/aggregate_runs.py --runs {BASE_DIR} --out artifacts/retry_fine_grained_results.csv")
    print(f"{'='*60}")
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
