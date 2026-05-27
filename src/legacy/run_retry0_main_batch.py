#!/usr/bin/env python3
"""
Main experiment re-run with max_retries=0.

Purpose:
  Replace or extend the main experiment results with a configurable
  retry budget, based on the P3B ablation finding that retry can
  significantly hurt pick-place-v3 success rate.

Design:
  - Default tasks: reach-v3, push-v3, pick-place-v3, door-open-v3
  - Default policies: no_mem, text_buf, struct_mem
  - Config: flash_every=20, flash_len=1, controller=PD
  - Scale: 5 seeds x 60 eps

Output:
  Configurable run root -> aggregate with scripts/aggregate_runs.py

Usage (AutoDL):
  export PATH=/root/miniconda3/bin:$PATH
  export PYOPENGL_PLATFORM=egl
  export MUJOCO_GL=egl
  cd /root/AutoResearchClaw
  python3 scripts/run_retry0_main_batch.py

Usage (dry-run):
  python3 scripts/run_retry0_main_batch.py --dry-run
"""
import subprocess
import sys
import time

ENV_PREFIX = ""
PYTHON = "python3 scripts/run_metaworld_mem.py "
DEFAULT_TASKS = ["reach-v3", "push-v3", "pick-place-v3", "door-open-v3"]
DEFAULT_POLICIES = ["no_mem", "text_buf", "struct_mem"]
TASKS = list(DEFAULT_TASKS)
POLICIES = list(DEFAULT_POLICIES)
SEEDS = [0, 1, 2, 3, 4]
EPS = 60
MAX_RETRIES = 0
RUN_DIR = "runs/batch/retry0_main"


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
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--tasks", type=str, default=",".join(DEFAULT_TASKS))
    parser.add_argument("--policies", type=str, default=",".join(DEFAULT_POLICIES))
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--run-dir", type=str, default="runs/batch/retry0_main")
    args = parser.parse_args()

    global TASKS, POLICIES, SEEDS, EPS, MAX_RETRIES, RUN_DIR
    TASKS = [t.strip() for t in args.tasks.split(",") if t.strip()]
    POLICIES = [p.strip() for p in args.policies.split(",") if p.strip()]
    SEEDS = [int(s) for s in args.seeds.split(",")]
    EPS = args.episodes
    MAX_RETRIES = args.max_retries
    RUN_DIR = args.run_dir

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
                    f"--max-retries {MAX_RETRIES} "
                    f"--run-dir {RUN_DIR}"
                )
                label = f"Retry{MAX_RETRIES}: {task} {policy} seed{seed}"
                rc = run(cmd, label, dry_run=args.dry_run)
                failed += (1 if rc != 0 else 0)
                done += 1
                print(f"  Progress: {done}/{total}")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Batch experiment done in {elapsed:.0f}s")
    print(f"  Completed: {done - failed}/{total}, Failed: {failed}/{total}")
    print(f"  Next: python3 scripts/aggregate_runs.py --runs {RUN_DIR} --out artifacts/results.csv")
    print(f"{'='*60}")
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
