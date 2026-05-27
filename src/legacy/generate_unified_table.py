#!/usr/bin/env python3
"""Generate unified 10-task main table + statistical tests.

Reads:
  - artifacts/v2_rerun_core3_results.csv (new 3 tasks, FO+PO, 5s×60e)
  - artifacts/v2_final_results/paper_table_results.csv (6 extended tasks, FO+PO, 5s×60e)

Outputs (to artifacts/v2_unified_main/):
  - unified_main_table.csv
  - unified_main_table.md
  - statistical_tests.txt
  - consistency_check.txt (flash_every V1/V2 comparison)
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np


TASKS_ORDER = [
    "reach-v3", "push-v3", "pick-place-v3",
    "door-open-v3", "drawer-open-v3", "faucet-open-v3",
    "button-press-topdown-v3", "sweep-v3", "shelf-place-v3",
]

POLICIES_ORDER = ["no_mem", "text_buf", "struct_mem"]
CONDITIONS = ["FO", "PO"]


def _load_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        print(f"WARNING: {path} not found")
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _extract_results(rows: list[dict[str, Any]]) -> dict[tuple, list[float]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in rows:
        task = r.get("task", "")
        policy = r.get("policy", "")
        cond = r.get("condition", "")
        sr = float(r.get("success_rate", 0))
        grouped[(task, policy, cond)].append(sr)
    return grouped


def permutation_test(a: list[float], b: list[float], n_perm: int = 10000) -> tuple[float, float]:
    observed = abs(np.mean(a) - np.mean(b))
    combined = np.array(list(a) + list(b))
    n_a = len(a)
    count = 0
    for _ in range(n_perm):
        perm = np.random.permutation(combined)
        diff = abs(perm[:n_a].mean() - perm[n_a:].mean())
        if diff >= observed:
            count += 1
    p_val = count / n_perm
    return float(observed), float(p_val)


def bootstrap_ci(data: list[float], n_boot: int = 10000, ci: float = 0.95) -> tuple[float, float, float]:
    arr = np.array(data)
    boots = [np.random.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    lo = float(np.percentile(boots, alpha * 100))
    hi = float(np.percentile(boots, (1 - alpha) * 100))
    return lo, float(np.mean(arr)), hi


def main():
    root = Path(__file__).resolve().parent.parent / "artifacts"
    out_dir = root / "v2_unified_main"
    out_dir.mkdir(parents=True, exist_ok=True)

    core3_path = root / "v2_rerun_core3_results.csv"
    ext6_path = root / "v2_final_results" / "paper_table_results.csv"

    core3_rows = _load_csv(core3_path)
    ext6_rows = _load_csv(ext6_path)

    if not core3_rows:
        print("ERROR: No core3 results found. Run Step 1 first.")
        return 1

    all_rows = core3_rows + ext6_rows
    grouped = _extract_results(all_rows)

    # --- Generate unified table ---
    csv_rows = []
    md_lines = []
    md_lines.append("# Unified 10-Task Main Table\n")
    md_lines.append(f"Generated: {__import__('time').strftime('%Y-%m-%d %H:%M')}\n")

    for cond in CONDITIONS:
        md_lines.append(f"\n## {cond} Condition\n")
        md_lines.append("| Task | NoMemory | TextBuffer | StructMemory |")
        md_lines.append("|------|----------|------------|--------------|")

        for task in TASKS_ORDER:
            vals = []
            for pol in POLICIES_ORDER:
                key = (task, pol, cond)
                if key in grouped:
                    srs = grouped[key]
                    mean = np.mean(srs)
                    std = np.std(srs, ddof=1) if len(srs) > 1 else 0
                    vals.append(f"{mean:.1%}±{std:.1%}")
                    csv_rows.append({
                        "task": task, "policy": pol, "condition": cond,
                        "mean": f"{mean:.4f}", "std": f"{std:.4f}",
                        "n_seeds": len(srs),
                    })
                else:
                    vals.append("—")
            md_lines.append(f"| {task} | {vals[0]} | {vals[1]} | {vals[2]} |")

    # Write CSV
    csv_path = out_dir / "unified_main_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "policy", "condition", "mean", "std", "n_seeds"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"CSV: {csv_path}")

    # Write Markdown
    md_path = out_dir / "unified_main_table.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Markdown: {md_path}")

    # --- Statistical tests ---
    test_lines = []
    test_lines.append("Statistical Tests for Unified Main Table")
    test_lines.append("=" * 60)
    test_lines.append(f"Generated: {__import__('time').strftime('%Y-%m-%d %H:%M')}\n")

    np.random.seed(42)

    # Test 1: NoMemory FO vs PO
    test_lines.append("\n1. NoMemory FO vs PO (permutation test)")
    test_lines.append("-" * 40)
    for task in TASKS_ORDER:
        fo_key = (task, "no_mem", "FO")
        po_key = (task, "no_mem", "PO")
        if fo_key in grouped and po_key in grouped:
            diff, p = permutation_test(grouped[fo_key], grouped[po_key])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            test_lines.append(f"  {task}: diff={diff:.3f} p={p:.4f} {sig}")

    # Test 2: TextBuffer vs StructMemory (PO)
    test_lines.append("\n2. TextBuffer vs StructMemory PO (permutation test + bootstrap CI)")
    test_lines.append("-" * 40)
    for task in TASKS_ORDER:
        tb_key = (task, "text_buf", "PO")
        sm_key = (task, "struct_mem", "PO")
        if tb_key in grouped and sm_key in grouped:
            diff, p = permutation_test(grouped[tb_key], grouped[sm_key])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            tb_lo, tb_mean, tb_hi = bootstrap_ci(grouped[tb_key])
            sm_lo, sm_mean, sm_hi = bootstrap_ci(grouped[sm_key])
            overlap = not (tb_lo > sm_hi or sm_lo > tb_hi)
            test_lines.append(f"  {task}: diff={diff:.3f} p={p:.4f} {sig}")
            test_lines.append(f"    TB 95% CI: [{tb_lo:.3f}, {tb_hi:.3f}] mean={tb_mean:.3f}")
            test_lines.append(f"    SM 95% CI: [{sm_lo:.3f}, {sm_hi:.3f}] mean={sm_mean:.3f}")
            test_lines.append(f"    CI overlap: {'YES' if overlap else 'NO'}")

    # Test 3: retry=0 vs retry=3 (from retry_budget_ablation if available)
    retry_path = root / "retry_budget_ablation"
    if retry_path.exists():
        test_lines.append("\n3. retry=0 vs retry=3 (from retry_budget_ablation)")
        test_lines.append("-" * 40)
        test_lines.append("  (Data to be loaded from artifacts/retry_budget_ablation/)")

    test_path = out_dir / "statistical_tests.txt"
    test_path.write_text("\n".join(test_lines), encoding="utf-8")
    print(f"Tests: {test_path}")

    # --- Consistency check: V1 vs V2 FO for core 3 tasks ---
    old_fo_path = root / "new_results" / "full_observable_results.csv"
    consistency_lines = []
    consistency_lines.append("Flash_every Consistency Check: V1 vs V2 FO")
    consistency_lines.append("=" * 60)
    consistency_lines.append(f"Generated: {__import__('time').strftime('%Y-%m-%d %H:%M')}\n")
    consistency_lines.append("Method: Compare FO success rate (3-seed subset s0,s1,s2)")
    consistency_lines.append("between old (artifacts/new_results/) and new (v2_rerun_core3) runs.\n")

    if old_fo_path.exists():
        old_rows = _load_csv(old_fo_path)
        old_grouped: dict[tuple, list[float]] = defaultdict(list)
        for r in old_rows:
            task = r.get("task", "")
            policy = r.get("policy", "")
            sr = float(r.get("success_rate", 0))
            old_grouped[(task, policy)].append(sr)

        can_reuse_flash = True
        for task in ["reach-v3", "push-v3", "pick-place-v3"]:
            consistency_lines.append(f"\n{task}:")
            for pol in POLICIES_ORDER:
                old_vals = old_grouped.get((task, pol), [])
                new_key = (task, pol, "FO")
                new_vals_all = grouped.get(new_key, [])
                new_vals_s012 = new_vals_all[:3]

                if old_vals and new_vals_s012:
                    old_mean = np.mean(old_vals)
                    new_mean = np.mean(new_vals_s012)
                    diff_pp = abs(old_mean - new_mean) * 100
                    status = "OK" if diff_pp <= 5 else "MISMATCH"
                    if diff_pp > 5:
                        can_reuse_flash = False
                    consistency_lines.append(
                        f"  {pol}: old={old_mean:.3f} (n={len(old_vals)}) "
                        f"new_s012={new_mean:.3f} (n={len(new_vals_s012)}) "
                        f"diff={diff_pp:.1f}pp [{status}]"
                    )
                else:
                    consistency_lines.append(f"  {pol}: data missing")

        consistency_lines.append(f"\nConclusion: flash_every curve can {'REUSE' if can_reuse_flash else 'NOT reuse'} (all diffs ≤ 5pp)")
    else:
        consistency_lines.append("Old FO data not found, skipping comparison.")

    cons_path = out_dir / "consistency_check.txt"
    cons_path.write_text("\n".join(consistency_lines), encoding="utf-8")
    print(f"Consistency: {cons_path}")

    print("\nDone! All outputs in artifacts/v2_unified_main/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
