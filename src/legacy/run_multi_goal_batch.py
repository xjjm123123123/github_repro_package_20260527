#!/usr/bin/env python3
"""
Run sequential multi-goal experiments for rule-based policies.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time


DEFAULT_TASKS = ["reach-v3", "push-v3", "pick-place-v3"]
DEFAULT_POLICIES = ["no_mem", "text_buf", "struct_mem"]
SEEDS = [0, 1, 2, 3, 4]
EPISODES = 60
MAX_STEPS = 200


def run(cmd: str, label: str, dry_run: bool = False) -> int:
    print(f"\n>>> {label}")
    print(cmd)
    if dry_run:
        return 0
    return subprocess.run(cmd, shell=True).returncode


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=str, default=",".join(DEFAULT_TASKS))
    p.add_argument("--policies", type=str, default=",".join(DEFAULT_POLICIES))
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--episodes", type=int, default=EPISODES)
    p.add_argument("--max-steps", type=int, default=MAX_STEPS)
    p.add_argument("--flash-every", type=int, default=5)
    p.add_argument("--flash-len", type=int, default=1)
    p.add_argument("--run-dir", type=str, default="runs/multi_goal")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    python = "/root/miniconda3/bin/python -u"
    script = "scripts/run_metaworld_mem.py"
    start = time.time()
    total = len(tasks) * len(policies) * len(seeds)
    done = 0
    failed = 0

    for task in tasks:
        for policy in policies:
            for seed in seeds:
                cmd = (
                    f"PYOPENGL_PLATFORM=egl MUJOCO_GL=egl {python} {script} "
                    f"--task {task} --episodes {args.episodes} --max-steps {args.max_steps} "
                    f"--seed {seed} --policy {policy} --controller PD "
                    f"--po-mode weak --flash-every {args.flash_every} --flash-len {args.flash_len} "
                    f"--multi-goal --multi-goal-flash-len 1 --max-retries 0 "
                    f"--run-dir {args.run_dir}"
                )
                rc = run(cmd, f"MultiGoal: {task} {policy} seed{seed}", dry_run=args.dry_run)
                done += 1
                if rc != 0:
                    failed += 1

    elapsed = time.time() - start
    print(f"\nCompleted {done}/{total}, failed={failed}, elapsed={elapsed:.0f}s")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

