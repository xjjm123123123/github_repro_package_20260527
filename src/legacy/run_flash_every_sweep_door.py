#!/usr/bin/env python3
"""Flash-every ablation for sweep-v3 and door-open-v3 tasks.

Purpose:
  Sweep flash_every over {1, 5, 10, 20, 50} under PO condition
  to measure how goal-visibility frequency affects policy performance.
  flash_every=1 means goal is visible every step (FO equivalent).

Design:
  - Tasks: sweep-v3, door-open-v3
  - Policies: no_mem, text_buf, struct_mem
  - flash_every: 1, 5, 10, 20, 50
  - Seeds: 0, 1, 2, 3, 4
  - Episodes: 60
  - Condition: PO only (full_observable=False)
  - Total runs: 2 x 3 x 5 x 5 = 150

Output:
  runs/batch/flash_every_sweep_door/{task}/{policy}/flash{flash_every}/seed{seed}/

Usage (AutoDL):
  export MUJOCO_GL=disable
  export PATH=/root/miniconda3/bin:$PATH
  cd /root/AutoResearchClaw
  python3 scripts/run_flash_every_sweep_door.py
  python3 scripts/run_flash_every_sweep_door.py --dry-run
  python3 scripts/run_flash_every_sweep_door.py --parallel 8
"""
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

PYTHON = "python3 scripts/run_metaworld_mem.py "
TASKS = ["sweep-v3", "door-open-v3"]
POLICIES = ["no_mem", "text_buf", "struct_mem"]
FLASH_EVERY_VALUES = [1, 5, 10, 20, 50]
SEEDS = [0, 1, 2, 3, 4]
EPS = 60
BASE_DIR = "runs/batch/flash_every_sweep_door"


def _run_dir(task, policy, flash_every, seed):
    return f"{BASE_DIR}/{task}/{policy}/flash{flash_every}/seed{seed}"


def _build_cmd(task, policy, flash_every, seed):
    run_dir = _run_dir(task, policy, flash_every, seed)
    return (
        f"{PYTHON} --task {task} --episodes {EPS} --seed {seed} "
        f"--policy {policy} --controller PD "
        f"--flash-every {flash_every} --flash-len 1 --po-noise-std 0.01 "
        f"--max-retries 0 --force-aware "
        f"--run-dir {run_dir}"
    )


def _write_config(task, policy, flash_every, seed, dry_run=False):
    cfg = {
        "task": task,
        "policy": policy,
        "flash_every": flash_every,
        "flash_len": 1,
        "seed": seed,
        "episodes": EPS,
        "controller": "PD",
        "full_observable": False,
        "po_noise_std": 0.01,
        "max_retries": 0,
        "fsm_version": 2,
        "force_aware": True,
    }
    run_dir = _run_dir(task, policy, flash_every, seed)
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
    parser = argparse.ArgumentParser(description="Flash-every ablation for sweep/door tasks")
    parser.add_argument("--dry-run", action="store_true", help="Show commands without executing")
    parser.add_argument("--parallel", type=int, default=4, help="Number of parallel workers (default 4)")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--flash-every", type=str, default="1,5,10,20,50",
                        help="Comma-separated flash_every values")
    args = parser.parse_args()

    global SEEDS, EPS, FLASH_EVERY_VALUES
    SEEDS = [int(s) for s in args.seeds.split(",")]
    EPS = args.episodes
    FLASH_EVERY_VALUES = [int(v) for v in args.flash_every.split(",")]

    os.environ["MUJOCO_GL"] = "disable"

    jobs = []
    for task in TASKS:
        for policy in POLICIES:
            for flash_every in FLASH_EVERY_VALUES:
                for seed in SEEDS:
                    label = f"{task} {policy} flash{flash_every} seed{seed}"
                    cmd = _build_cmd(task, policy, flash_every, seed)
                    jobs.append((cmd, label, task, policy, flash_every, seed))

    total = len(jobs)
    print(f"Flash-every ablation: {total} runs")
    print(f"  Tasks: {TASKS}")
    print(f"  Policies: {POLICIES}")
    print(f"  flash_every: {FLASH_EVERY_VALUES}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Episodes: {EPS}")
    print(f"  Parallel: {args.parallel}")
    print(f"  Output: {BASE_DIR}/")
    print()

    for cmd, label, task, policy, flash_every, seed in jobs:
        _write_config(task, policy, flash_every, seed, dry_run=args.dry_run)

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
            if done % 10 == 0:
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
                if done % 10 == 0:
                    elapsed = time.time() - t0
                    print(f"  Progress: {done}/{total} ({elapsed:.0f}s elapsed, {failed} failed)")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Flash-every ablation done in {elapsed:.0f}s")
    print(f"  Completed: {done - failed}/{total}, Failed: {failed}/{total}")
    print(f"  Output: {BASE_DIR}/")
    print(f"  Next: python3 scripts/aggregate_runs.py --runs {BASE_DIR} --out artifacts/flash_every_sweep_door_results.csv")
    print(f"{'='*60}")
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
