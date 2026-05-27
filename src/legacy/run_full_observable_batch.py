#!/usr/bin/env python3
"""
Full-Observable control experiment batch runner.

Purpose:
  Verify that the performance gap between NoMemory and memory-augmented
  policies is caused by partial observability, not by intrinsic policy
  differences.

Design:
  - Tasks: reach-v3, push-v3, pick-place-v3
  - Policies: no_mem, text_buf, struct_mem
  - Config: goal always visible (--full-observable)
  - Scale: 3 seeds x 60 eps
  - Controller: PD
  - max_retries=0 (preferred, based on P3B ablation)

Output:
  runs/batch/full_observable/... -> aggregate to artifacts/full_observable_results.csv

Usage (AutoDL):
  export PATH=/root/miniconda3/bin:$PATH
  export PYOPENGL_PLATFORM=egl
  export MUJOCO_GL=egl
  cd /root/AutoResearchClaw
  python3 scripts/run_full_observable_batch.py

Usage (dry-run):
  python3 scripts/run_full_observable_batch.py --dry-run
"""
import subprocess
import sys
import time

ENV_PREFIX = ""
PYTHON = "python3 scripts/run_metaworld_mem.py "
TASKS = ["reach-v3", "push-v3", "pick-place-v3"]
POLICIES = ["no_mem", "text_buf", "struct_mem"]
SEEDS = [0, 1, 2]
EPS = 60


def run(cmd, label, dry_run=False):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    if dry_run:
        print(f"  $ {cmd}")
        return 0
    full = ENV_PREFIX + cmd
    p = subprocess.run(full, shell=True, check=False)
    if p.returncode != 0:
        print(f"FAILED: {label} (rc={p.returncode})")
    return p.returncode


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-retries", type=int, default=0, help="0=disable retry (preferred)")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--episodes", type=int, default=60)
    args = parser.parse_args()

    global SEEDS, EPS
    SEEDS = [int(s) for s in args.seeds.split(",")]
    EPS = args.episodes
    max_retries = args.max_retries

    t0 = time.time()
    failed = 0
    total = len(TASKS) * len(POLICIES) * len(SEEDS)
    done = 0

    for task in TASKS:
        for policy in POLICIES:
            for seed in SEEDS:
                cmd = (
                    f"{PYTHON} --task {task} --episodes {EPS} --seed {seed} "
                    f"--policy {policy} --controller PD "
                    f"--flash-every 20 --flash-len 1 --po-noise-std 0.01 "
                    f"--max-retries {max_retries} "
                    f"--full-observable "
                    f"--run-dir runs/batch/full_observable"
                )
                label = f"FullObs: {task} {policy} seed{seed} retries={max_retries}"
                rc = run(cmd, label, dry_run=args.dry_run)
                failed += (1 if rc != 0 else 0)
                done += 1
                print(f"  Progress: {done}/{total}")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Full-Observable experiment done in {elapsed:.0f}s")
    print(f"  Completed: {done - failed}/{total}, Failed: {failed}/{total}")
    print(f"  Next: python3 scripts/aggregate_runs.py --runs runs/batch/full_observable --out artifacts/full_observable_results.csv")
    print(f"{'='*60}")
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
