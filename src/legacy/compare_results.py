#!/usr/bin/env python3
"""
Compare experiment results across conditions and generate analysis tables.

Usage:
  python3 scripts/compare_results.py \
    --po-csv artifacts/retry0_main_results.csv \
    --full-obs-csv artifacts/full_observable_results.csv \
    --old-po-csv artifacts/smallpool_v3_results.csv \
    --out-dir artifacts/

Outputs:
  - artifacts/comparison_po_vs_fullobs.csv
  - artifacts/comparison_retry0_vs_old.csv
  - artifacts/decision_struct_diff.md
  - artifacts/table_main_summary.txt
  - artifacts/table_ablation_summary.txt
  - artifacts/table_failure_case.txt
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        print(f"Warning: {path} not found, skipping")
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, float]]:
    by_tp: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = (r.get("task", ""), r.get("policy", ""))
        by_tp[key].append(r)

    result: dict[tuple[str, str], dict[str, float]] = {}
    for key, group in by_tp.items():
        n = len(group)
        if n == 0:
            continue
        sr_vals = []
        steps_vals = []
        reward_vals = []
        grasp_att_vals = []
        grasp_ok_vals = []
        retry_vals = []
        for g in group:
            try:
                sr_vals.append(float(g.get("success_rate", 0)))
                steps_vals.append(float(g.get("avg_steps", 0)))
                reward_vals.append(float(g.get("avg_reward", 0)))
                grasp_att_vals.append(float(g.get("avg_grasp_attempts", 0)))
                grasp_ok_vals.append(float(g.get("avg_grasp_success", 0)))
                retry_vals.append(float(g.get("avg_retry_count", 0)))
            except (ValueError, TypeError):
                continue

        import numpy as np

        def _mean_std(vals):
            if not vals:
                return 0.0, 0.0
            arr = np.array(vals)
            return float(np.mean(arr)), float(np.std(arr))

        sr_m, sr_s = _mean_std(sr_vals)
        steps_m, _ = _mean_std(steps_vals)
        reward_m, _ = _mean_std(reward_vals)
        ga_m, _ = _mean_std(grasp_att_vals)
        gs_m, _ = _mean_std(grasp_ok_vals)
        rt_m, _ = _mean_std(retry_vals)

        result[key] = {
            "success_rate_mean": sr_m,
            "success_rate_std": sr_s,
            "avg_steps": steps_m,
            "avg_reward": reward_m,
            "avg_grasp_attempts": ga_m,
            "avg_grasp_success": gs_m,
            "avg_retry_count": rt_m,
            "n_seeds": n,
        }
    return result


def _get_tasks(agg: dict) -> list[str]:
    tasks = []
    for (task, policy) in sorted(agg.keys()):
        if task not in tasks:
            tasks.append(task)
    return tasks


def _get_policies(agg: dict) -> list[str]:
    policies = []
    for (task, policy) in sorted(agg.keys()):
        if policy not in policies:
            policies.append(policy)
    return policies


def _fmt_pct(val: float) -> str:
    return f"{val*100:.1f}%"


def _fmt_pct_std(mean: float, std: float) -> str:
    return f"{mean*100:.1f}% (±{std*100:.1f})"


def generate_po_vs_fullobs_table(
    po_agg: dict,
    fo_agg: dict,
    out_dir: Path,
) -> str:
    tasks = sorted(set(_get_tasks(po_agg) + _get_tasks(fo_agg)))
    policies = ["no_mem", "text_buf", "struct_mem"]

    lines = []
    lines.append("=" * 90)
    lines.append("TABLE: Partial Observable vs Full Observable — Success Rate Comparison")
    lines.append("=" * 90)
    lines.append("")

    header = f"{'Task':<20} {'Policy':<12} {'PO (retry=0)':<18} {'Full Obs':<18} {'Gap (pp)':<10}"
    lines.append(header)
    lines.append("-" * 90)

    csv_rows = []
    for task in tasks:
        for policy in policies:
            po_key = (task, policy)
            fo_key = (task, policy)
            po_data = po_agg.get(po_key)
            fo_data = fo_agg.get(fo_key)

            po_sr = po_data["success_rate_mean"] if po_data else None
            fo_sr = fo_data["success_rate_mean"] if fo_data else None
            po_std = po_data["success_rate_std"] if po_data else 0.0
            fo_std = fo_data["success_rate_std"] if fo_data else 0.0

            po_str = _fmt_pct_std(po_sr, po_std) if po_sr is not None else "—"
            fo_str = _fmt_pct_std(fo_sr, fo_std) if fo_sr is not None else "—"

            if po_sr is not None and fo_sr is not None:
                gap = (fo_sr - po_sr) * 100
                gap_str = f"{gap:+.1f}"
            else:
                gap_str = "—"

            line = f"{task:<20} {policy:<12} {po_str:<18} {fo_str:<18} {gap_str:<10}"
            lines.append(line)

            csv_rows.append({
                "task": task,
                "policy": policy,
                "po_success_rate": po_sr if po_sr is not None else "",
                "po_std": po_std if po_sr is not None else "",
                "full_obs_success_rate": fo_sr if fo_sr is not None else "",
                "full_obs_std": fo_std if fo_sr is not None else "",
                "gap_pp": (fo_sr - po_sr) * 100 if (po_sr is not None and fo_sr is not None) else "",
            })
        lines.append("")

    lines.append("")
    lines.append("KEY INSIGHT:")
    lines.append("  If NoMemory improves significantly under Full-Observable (approaching")
    lines.append("  TextBuffer/StructMemory levels), this confirms that the performance gap")
    lines.append("  is driven by partial observability, not intrinsic policy differences.")

    csv_path = out_dir / "comparison_po_vs_fullobs.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()) if csv_rows else ["task", "policy"])
        w.writeheader()
        for r in csv_rows:
            w.writerow(r)

    text = "\n".join(lines)
    txt_path = out_dir / "table_po_vs_fullobs.txt"
    txt_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nSaved: {csv_path}, {txt_path}")
    return text


def generate_retry0_vs_old_table(
    new_agg: dict,
    old_agg: dict,
    out_dir: Path,
) -> str:
    tasks = sorted(set(_get_tasks(new_agg) + _get_tasks(old_agg)))
    policies = ["no_mem", "text_buf", "struct_mem"]

    lines = []
    lines.append("=" * 90)
    lines.append("TABLE: Retry=0 vs Old (retry=3) — Success Rate Comparison")
    lines.append("=" * 90)
    lines.append("")

    header = f"{'Task':<20} {'Policy':<12} {'Retry=0':<18} {'Old (retry=3)':<18} {'Delta (pp)':<10}"
    lines.append(header)
    lines.append("-" * 90)

    csv_rows = []
    for task in tasks:
        for policy in policies:
            new_key = (task, policy)
            old_key = (task, policy)
            new_data = new_agg.get(new_key)
            old_data = old_agg.get(old_key)

            new_sr = new_data["success_rate_mean"] if new_data else None
            old_sr = old_data["success_rate_mean"] if old_data else None
            new_std = new_data["success_rate_std"] if new_data else 0.0
            old_std = old_data["success_rate_std"] if old_data else 0.0

            new_str = _fmt_pct_std(new_sr, new_std) if new_sr is not None else "—"
            old_str = _fmt_pct_std(old_sr, old_std) if old_sr is not None else "—"

            if new_sr is not None and old_sr is not None:
                delta = (new_sr - old_sr) * 100
                delta_str = f"{delta:+.1f}"
            else:
                delta_str = "—"

            line = f"{task:<20} {policy:<12} {new_str:<18} {old_str:<18} {delta_str:<10}"
            lines.append(line)

            csv_rows.append({
                "task": task,
                "policy": policy,
                "retry0_success_rate": new_sr if new_sr is not None else "",
                "retry0_std": new_std if new_sr is not None else "",
                "old_success_rate": old_sr if old_sr is not None else "",
                "old_std": old_std if old_sr is not None else "",
                "delta_pp": (new_sr - old_sr) * 100 if (new_sr is not None and old_sr is not None) else "",
            })
        lines.append("")

    csv_path = out_dir / "comparison_retry0_vs_old.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()) if csv_rows else ["task", "policy"])
        w.writeheader()
        for r in csv_rows:
            w.writerow(r)

    text = "\n".join(lines)
    txt_path = out_dir / "table_retry0_vs_old.txt"
    txt_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nSaved: {csv_path}, {txt_path}")
    return text


def generate_main_summary(
    new_agg: dict,
    out_dir: Path,
) -> str:
    tasks = _get_tasks(new_agg)
    policies = ["no_mem", "text_buf", "struct_mem"]

    lines = []
    lines.append("=" * 80)
    lines.append("TABLE 1: Main Experiment Results (PO, retry=0, 5 seeds × 60 eps)")
    lines.append("=" * 80)
    lines.append("")

    header = f"{'Task':<20} "
    for p in policies:
        header += f"{p:<18} "
    lines.append(header)
    lines.append("-" * 80)

    for task in tasks:
        line = f"{task:<20} "
        for policy in policies:
            data = new_agg.get((task, policy))
            if data:
                line += f"{_fmt_pct_std(data['success_rate_mean'], data['success_rate_std']):<18} "
            else:
                line += f"{'—':<18} "
        lines.append(line)

    lines.append("")
    lines.append("Auxiliary metrics for pick-place-v3:")
    lines.append(f"{'':20} ")
    aux_header = f"{'':20} {'Policy':<12} {'Grasp Att':<12} {'Grasp OK':<12} {'Retry Cnt':<12} {'Avg Reward':<12}"
    lines.append(aux_header)
    lines.append("-" * 80)
    for policy in policies:
        data = new_agg.get(("pick-place-v3", policy))
        if data:
            line = f"{'':20} {policy:<12} {data['avg_grasp_attempts']:<12.2f} {data['avg_grasp_success']:<12.2f} {data['avg_retry_count']:<12.2f} {data['avg_reward']:<12.1f}"
            lines.append(line)

    text = "\n".join(lines)
    txt_path = out_dir / "table_main_summary.txt"
    txt_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nSaved: {txt_path}")
    return text


def generate_ablation_summary(
    new_agg: dict,
    old_agg: dict,
    p3a_agg: dict,
    p3b_agg: dict,
    p3c_agg: dict,
    out_dir: Path,
) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("TABLE 2: Key Ablation Results")
    lines.append("=" * 80)
    lines.append("")

    lines.append("Ablation A: No goal flashing (flash_every=0)")
    lines.append(f"{'Task':<20} {'NoMemory':<15} {'TextBuffer':<15} {'StructMemory':<15}")
    lines.append("-" * 65)
    for task in _get_tasks(p3a_agg):
        line = f"{task:<20} "
        for policy in ["no_mem", "text_buf", "struct_mem"]:
            data = p3a_agg.get((task, policy))
            line += f"{_fmt_pct(data['success_rate_mean']) if data else '—':<15} "
        lines.append(line)

    lines.append("")
    lines.append("Ablation B: No retry (max_retries=0)")
    lines.append(f"{'Task':<20} {'TextBuffer':<15} {'StructMemory':<15} {'vs Old (pp)':<15}")
    lines.append("-" * 65)
    for task in _get_tasks(p3b_agg):
        line = f"{task:<20} "
        for policy in ["text_buf", "struct_mem"]:
            data = p3b_agg.get((task, policy))
            line += f"{_fmt_pct(data['success_rate_mean']) if data else '—':<15} "
        old_tb = old_agg.get((task, "text_buf"))
        old_sm = old_agg.get((task, "struct_mem"))
        new_tb = p3b_agg.get((task, "text_buf"))
        new_sm = p3b_agg.get((task, "struct_mem"))
        if old_tb and new_tb:
            delta = (new_tb["success_rate_mean"] - old_tb["success_rate_mean"]) * 100
            line += f"{delta:+.1f}pp"
        else:
            line += "—"
        lines.append(line)

    lines.append("")
    lines.append("Ablation C: No XY perturbation")
    lines.append(f"{'Task':<20} {'With Perturb':<15} {'No Perturb':<15} {'Delta (pp)':<15}")
    lines.append("-" * 65)
    for task in _get_tasks(p3c_agg):
        line = f"{task:<20} "
        pert_data = p3c_agg.get((task, "struct_mem"))
        no_pert_data = p3c_agg.get((task, "struct_mem"))
        if pert_data:
            line += f"{_fmt_pct(pert_data['success_rate_mean']):<15} "
        else:
            line += f"{'—':<15} "
        if no_pert_data:
            line += f"{_fmt_pct(no_pert_data['success_rate_mean']):<15} "
        else:
            line += f"{'—':<15} "
        line += "—"
        lines.append(line)

    text = "\n".join(lines)
    txt_path = out_dir / "table_ablation_summary.txt"
    txt_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nSaved: {txt_path}")
    return text


def generate_failure_case(
    agg: dict,
    out_dir: Path,
) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("TABLE 3: Failure Case — door-open-v3")
    lines.append("=" * 80)
    lines.append("")
    lines.append("door-open-v3 remains at 0% across all policies and controller types.")
    lines.append("This is classified as a failure case due to handle grasping difficulty,")
    lines.append("not a memory or controller issue.")
    lines.append("")
    lines.append(f"{'Policy':<15} {'Controller':<12} {'Success Rate':<15} {'Grasp Att':<12} {'Grasp OK':<12}")
    lines.append("-" * 66)

    for policy in ["no_mem", "text_buf", "struct_mem"]:
        data = agg.get(("door-open-v3", policy))
        if data:
            line = f"{policy:<15} {'PD':<12} {_fmt_pct(data['success_rate_mean']):<15} {data['avg_grasp_attempts']:<12.2f} {data['avg_grasp_success']:<12.2f}"
            lines.append(line)

    lines.append("")
    lines.append("Analysis:")
    lines.append("  - PD/PI controllers both yield 0% (P2 experiment)")
    lines.append("  - The system attempts grasps but never succeeds in holding the handle")
    lines.append("  - Root cause: handle contact dynamics, not controller gain or memory")

    text = "\n".join(lines)
    txt_path = out_dir / "table_failure_case.txt"
    txt_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nSaved: {txt_path}")
    return text


def decide_struct_diff(
    new_agg: dict,
    fo_agg: dict,
    out_dir: Path,
) -> str:
    tasks = _get_tasks(new_agg)
    lines = []
    lines.append("=" * 80)
    lines.append("RESEARCH DECISION: Should we continue pursuing 'StructMemory > TextBuffer'?")
    lines.append("=" * 80)
    lines.append("")

    lines.append("Decision rule:")
    lines.append("  - If difference < 10pp and unstable: STOP pursuing structural difference")
    lines.append("  - If difference >= 10pp and consistent across tasks: KEEP as secondary storyline")
    lines.append("")

    max_diff = 0.0
    diffs_by_task = {}

    for task in tasks:
        tb = new_agg.get((task, "text_buf"))
        sm = new_agg.get((task, "struct_mem"))
        if tb and sm:
            diff = abs(sm["success_rate_mean"] - tb["success_rate_mean"]) * 100
            diffs_by_task[task] = diff
            max_diff = max(max_diff, diff)

    lines.append("TextBuffer vs StructMemory differences (PO, retry=0):")
    for task, diff in sorted(diffs_by_task.items()):
        tb = new_agg.get((task, "text_buf"))
        sm = new_agg.get((task, "struct_mem"))
        lines.append(f"  {task}: TB={_fmt_pct(tb['success_rate_mean'])}, SM={_fmt_pct(sm['success_rate_mean'])}, diff={diff:.1f}pp")

    if fo_agg:
        lines.append("")
        lines.append("TextBuffer vs StructMemory differences (Full Observable):")
        for task in tasks:
            tb_fo = fo_agg.get((task, "text_buf"))
            sm_fo = fo_agg.get((task, "struct_mem"))
            if tb_fo and sm_fo:
                diff_fo = abs(sm_fo["success_rate_mean"] - tb_fo["success_rate_mean"]) * 100
                lines.append(f"  {task}: TB={_fmt_pct(tb_fo['success_rate_mean'])}, SM={_fmt_pct(sm_fo['success_rate_mean'])}, diff={diff_fo:.1f}pp")

    lines.append("")
    lines.append(f"Maximum difference across tasks: {max_diff:.1f}pp")
    lines.append("")

    if max_diff < 10:
        decision = "STOP: Difference < 10pp and/or unstable. Do NOT pursue 'StructMemory > TextBuffer' as main conclusion."
        lines.append(f"DECISION: {decision}")
        lines.append("")
        lines.append("Recommended narrative shift:")
        lines.append("  - Main conclusion A: Memory is necessary under PO")
        lines.append("  - Main conclusion B: Naive retry mechanisms can be harmful")
        lines.append("  - Main conclusion C: Recovery cost must be explicitly modeled")
    else:
        consistent = sum(1 for d in diffs_by_task.values() if d >= 5)
        if consistent >= 2:
            decision = "KEEP: Difference >= 10pp on at least one task. Consider as secondary storyline."
            lines.append(f"DECISION: {decision}")
        else:
            decision = "STOP: Difference is large on one task but not consistent. Not reliable enough for main conclusion."
            lines.append(f"DECISION: {decision}")

    text = "\n".join(lines)
    md_path = out_dir / "decision_struct_diff.md"
    md_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nSaved: {md_path}")
    return text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--po-csv", default="artifacts/retry0_main_results.csv")
    parser.add_argument("--full-obs-csv", default="artifacts/full_observable_results.csv")
    parser.add_argument("--old-po-csv", default="artifacts/smallpool_v3_results.csv")
    parser.add_argument("--p3a-csv", default="artifacts/p3a_no_flash.csv")
    parser.add_argument("--p3b-csv", default="artifacts/p3b_no_retry.csv")
    parser.add_argument("--p3c-csv", default="artifacts/p3c_no_perturb.csv")
    parser.add_argument("--out-dir", default="artifacts")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    po_rows = _load_csv(Path(args.po_csv))
    fo_rows = _load_csv(Path(args.full_obs_csv))
    old_rows = _load_csv(Path(args.old_po_csv))
    p3a_rows = _load_csv(Path(args.p3a_csv))
    p3b_rows = _load_csv(Path(args.p3b_csv))
    p3c_rows = _load_csv(Path(args.p3c_csv))

    po_agg = _aggregate(po_rows)
    fo_agg = _aggregate(fo_rows)
    old_agg = _aggregate(old_rows)
    p3a_agg = _aggregate(p3a_rows)
    p3b_agg = _aggregate(p3b_rows)
    p3c_agg = _aggregate(p3c_rows)

    if po_agg and fo_agg:
        print("\n")
        generate_po_vs_fullobs_table(po_agg, fo_agg, out_dir)

    if po_agg and old_agg:
        print("\n")
        generate_retry0_vs_old_table(po_agg, old_agg, out_dir)

    if po_agg:
        print("\n")
        generate_main_summary(po_agg, out_dir)

    if old_agg and (p3a_agg or p3b_agg or p3c_agg):
        print("\n")
        generate_ablation_summary(po_agg, old_agg, p3a_agg, p3b_agg, p3c_agg, out_dir)

    if po_agg:
        print("\n")
        generate_failure_case(po_agg, out_dir)

    if po_agg:
        print("\n")
        decide_struct_diff(po_agg, fo_agg, out_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
