#!/usr/bin/env python3
"""
Run all P2 + P3 ablation experiments sequentially.

P2: door-open-v3 PI controller comparison
P3-1: No goal flashing (flash_every=0) on reach/push/pick-place
P3-2: No retry (max_retries=0) on pick-place/door-open
P3-3: No XY perturbation (--no-perturbation) on pick-place/door-open
"""
import subprocess
import sys
import time

ENV = "export PATH=/root/miniconda3/bin:$PATH && export PYOPENGL_PLATFORM=egl && export MUJOCO_GL=egl && cd /root/AutoResearchClaw && "
PYTHON = "python3 scripts/run_metaworld_mem.py "
SEEDS = [0, 1, 2]
EPS = 60

def run(cmd, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    full = ENV + cmd
    p = subprocess.run(full, shell=True, check=False)
    if p.returncode != 0:
        print(f"FAILED: {label} (rc={p.returncode})")
    return p.returncode

def main():
    t0 = time.time()
    failed = 0

    # P2: door-open-v3 PI controller
    for policy in ["text_buf", "struct_mem"]:
        for ctrl in ["PD", "PI"]:
            for seed in SEEDS:
                cmd = (f"{PYTHON} --task door-open-v3 --episodes {EPS} --seed {seed} "
                       f"--policy {policy} --controller {ctrl} "
                       f"--flash-every 20 --flash-len 1 --po-noise-std 0.01 "
                       f"--run-dir runs/ablation/p2_door_pi")
                rc = run(cmd, f"P2: door-open {policy} {ctrl} seed{seed}")
                failed += (1 if rc != 0 else 0)

    # P3-1: No goal flashing
    for task in ["reach-v3", "push-v3", "pick-place-v3"]:
        for policy in ["no_mem", "text_buf", "struct_mem"]:
            for seed in SEEDS:
                cmd = (f"{PYTHON} --task {task} --episodes {EPS} --seed {seed} "
                       f"--policy {policy} --controller PD "
                       f"--flash-every 0 --flash-len 0 --po-noise-std 0.01 "
                       f"--run-dir runs/ablation/p3a_no_flash")
                rc = run(cmd, f"P3a: {task} {policy} no-flash seed{seed}")
                failed += (1 if rc != 0 else 0)

    # P3-2: No retry
    for task in ["pick-place-v3", "door-open-v3"]:
        for policy in ["text_buf", "struct_mem"]:
            for seed in SEEDS:
                cmd = (f"{PYTHON} --task {task} --episodes {EPS} --seed {seed} "
                       f"--policy {policy} --controller PD "
                       f"--flash-every 20 --flash-len 1 --po-noise-std 0.01 "
                       f"--max-retries 0 "
                       f"--run-dir runs/ablation/p3b_no_retry")
                rc = run(cmd, f"P3b: {task} {policy} no-retry seed{seed}")
                failed += (1 if rc != 0 else 0)

    # P3-3: No XY perturbation
    for task in ["pick-place-v3", "door-open-v3"]:
        for seed in SEEDS:
            cmd = (f"{PYTHON} --task {task} --episodes {EPS} --seed {seed} "
                   f"--policy struct_mem --controller PD "
                   f"--flash-every 20 --flash-len 1 --po-noise-std 0.01 "
                   f"--no-perturbation "
                   f"--run-dir runs/ablation/p3c_no_perturb")
            rc = run(cmd, f"P3c: {task} struct_mem no-perturb seed{seed}")
            failed += (1 if rc != 0 else 0)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  All ablation experiments done in {elapsed:.0f}s")
    print(f"  Failed: {failed}")
    print(f"{'='*60}")
    return 1 if failed > 0 else 0

if __name__ == "__main__":
    raise SystemExit(main())
