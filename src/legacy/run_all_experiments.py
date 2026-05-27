#!/usr/bin/env python3
"""
One-click runner for the full experiment pipeline on AutoDL.

Execution order:
  Step 1: Full-Observable control experiment (3 seeds x 60 eps)
  Step 2: Retry=0 main experiment (5 seeds x 60 eps)
  Step 3: Aggregate results
  Step 4: Compare and generate analysis tables
  Step 5: Generate paper figures

Usage (AutoDL):
  export PATH=/root/miniconda3/bin:$PATH
  export PYOPENGL_PLATFORM=egl
  export MUJOCO_GL=egl
  cd /root/AutoResearchClaw
  python3 scripts/run_all_experiments.py

Usage (dry-run, just print commands):
  python3 scripts/run_all_experiments.py --dry-run

Usage (skip to analysis only):
  python3 scripts/run_all_experiments.py --skip-experiments
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run(cmd: str, label: str, dry_run: bool = False) -> int:
    print(f"\n{'='*70}")
    print(f"  STEP: {label}")
    print(f"  CMD: {cmd}")
    print(f"{'='*70}")
    if dry_run:
        return 0
    p = subprocess.run(cmd, shell=True, check=False)
    if p.returncode != 0:
        print(f"FAILED: {label} (rc={p.returncode})")
    return p.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-experiments", action="store_true", help="Skip running experiments, go straight to analysis")
    parser.add_argument("--only-full-obs", action="store_true", help="Only run full-observable experiment")
    parser.add_argument("--only-retry0", action="store_true", help="Only run retry=0 main experiment")
    args = parser.parse_args()

    t0 = time.time()
    failed = 0

    if not args.skip_experiments:
        # Step 1: Full-Observable control experiment
        if not args.only_retry0:
            rc = run(
                "python3 scripts/run_full_observable_batch.py",
                "Step 1: Full-Observable Control (3 seeds x 60 eps)",
                dry_run=args.dry_run,
            )
            failed += (1 if rc != 0 else 0)

            if rc == 0 and not args.dry_run:
                run(
                    "python3 scripts/aggregate_runs.py --runs runs/batch/full_observable --out artifacts/full_observable_results.csv",
                    "Step 1b: Aggregate Full-Observable results",
                    dry_run=args.dry_run,
                )

        # Step 2: Retry=0 main experiment
        if not args.only_full_obs:
            rc = run(
                "python3 scripts/run_retry0_main_batch.py",
                "Step 2: Retry=0 Main Experiment (5 seeds x 60 eps)",
                dry_run=args.dry_run,
            )
            failed += (1 if rc != 0 else 0)

            if rc == 0 and not args.dry_run:
                run(
                    "python3 scripts/aggregate_runs.py --runs runs/batch/retry0_main --out artifacts/retry0_main_results.csv",
                    "Step 2b: Aggregate Retry=0 results",
                    dry_run=args.dry_run,
                )

    # Step 3: Compare results and generate analysis
    rc = run(
        "python3 scripts/compare_results.py",
        "Step 3: Compare Results & Generate Analysis Tables",
        dry_run=args.dry_run,
    )

    # Step 4: Generate paper figures
    rc = run(
        "python3 scripts/generate_paper_figures.py",
        "Step 4: Generate Paper Figures",
        dry_run=args.dry_run,
    )

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  Pipeline completed in {elapsed:.0f}s")
    print(f"  Failed steps: {failed}")
    print(f"{'='*70}")

    print("\nOutput files:")
    print("  artifacts/full_observable_results.csv  — Full-Obs raw results")
    print("  artifacts/retry0_main_results.csv       — Retry=0 raw results")
    print("  artifacts/comparison_po_vs_fullobs.csv  — PO vs Full-Obs comparison")
    print("  artifacts/comparison_retry0_vs_old.csv  — Retry=0 vs Old comparison")
    print("  artifacts/decision_struct_diff.md        — Struct diff decision")
    print("  artifacts/fig1_po_main_success_rate.png  — Fig 1: PO main")
    print("  artifacts/fig2_retry_comparison.png      — Fig 2: Retry comparison")
    print("  artifacts/fig3_po_vs_fullobs.png         — Fig 3: PO vs Full-Obs")
    print("  artifacts/table1_main_results.tex        — Table 1: Main results")
    print("  artifacts/table2_ablation.tex            — Table 2: Ablation")
    print("  artifacts/table3_failure.tex             — Table 3: Failure case")

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
