#!/usr/bin/env python3
"""
Generate paper-quality figures and tables from experiment results.

Required figures:
  Fig 1: PO main experiment success rate bar chart
  Fig 2: retry=0 vs original comparison (focus on pick-place-v3)
  Fig 3: Full Observable vs PO comparison

Required tables:
  Table 1: Main experiment success rate summary
  Table 2: Key ablation summary
  Table 3: Failure case (door-open-v3)

Usage:
  python3 scripts/generate_paper_figures.py \
    --po-csv artifacts/retry0_main_results.csv \
    --full-obs-csv artifacts/full_observable_results.csv \
    --old-po-csv artifacts/smallpool_v3_results.csv \
    --p3a-csv artifacts/p3a_no_flash.csv \
    --p3b-csv artifacts/p3b_no_retry.csv \
    --p3c-csv artifacts/p3c_no_perturb.csv \
    --out-dir artifacts/
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np
    HAS_MPL = True
except Exception:
    HAS_MPL = False


POLICY_LABELS = {"no_mem": "NoMemory", "text_buf": "TextBuffer", "struct_mem": "StructMemory"}
POLICY_COLORS = {"no_mem": "#e74c3c", "text_buf": "#3498db", "struct_mem": "#2ecc71"}
POLICY_HATCH = {"no_mem": "", "text_buf": "//", "struct_mem": "\\\\"}


def _load_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        print(f"Warning: {path} not found")
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _aggregate(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, float]]:
    by_tp: dict[tuple[str, str], list] = defaultdict(list)
    for r in rows:
        by_tp[(r.get("task", ""), r.get("policy", ""))].append(r)

    result: dict[tuple[str, str], dict[str, float]] = {}
    for key, group in by_tp.items():
        n = len(group)
        if n == 0:
            continue
        srs = [float(g.get("success_rate", 0)) for g in group]
        steps = [float(g.get("avg_steps", 0)) for g in group]
        rewards = [float(g.get("avg_reward", 0)) for g in group]
        gras = [float(g.get("avg_grasp_attempts", 0)) for g in group]
        groks = [float(g.get("avg_grasp_success", 0)) for g in group]
        rets = [float(g.get("avg_retry_count", 0)) for g in group]

        result[key] = {
            "success_rate_mean": sum(srs) / n,
            "success_rate_std": (sum((s - sum(srs)/n)**2 for s in srs) / n) ** 0.5 if n > 1 else 0.0,
            "avg_steps": sum(steps) / n,
            "avg_reward": sum(rewards) / n,
            "avg_grasp_attempts": sum(gras) / n,
            "avg_grasp_success": sum(groks) / n,
            "avg_retry_count": sum(rets) / n,
            "n_seeds": n,
        }
    return result


def _get_tasks(agg: dict) -> list[str]:
    seen = []
    for (task, _) in sorted(agg.keys()):
        if task not in seen:
            seen.append(task)
    return seen


def fig1_po_main(po_agg: dict, out_dir: Path) -> None:
    if not HAS_MPL:
        print("matplotlib not available, skip Fig 1")
        return

    tasks = [t for t in _get_tasks(po_agg) if t != "door-open-v3"]
    policies = ["no_mem", "text_buf", "struct_mem"]

    x = np.arange(len(tasks))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, pol in enumerate(policies):
        means = [po_agg.get((t, pol), {}).get("success_rate_mean", 0.0) * 100 for t in tasks]
        stds = [po_agg.get((t, pol), {}).get("success_rate_std", 0.0) * 100 for t in tasks]
        bars = ax.bar(
            x + i * width, means, width,
            label=POLICY_LABELS[pol], color=POLICY_COLORS[pol],
            yerr=stds, capsize=3, edgecolor="white", linewidth=0.5,
            hatch=POLICY_HATCH[pol],
        )
        for bar, m in zip(bars, means):
            if m > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                        f"{m:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_xlabel("Task", fontsize=12)
    ax.set_xticks(x + width)
    ax.set_xticklabels(tasks, fontsize=10)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10, loc="upper right")
    ax.set_title("Fig 1: PO Main Experiment (goal flashing, retry=0)", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_po_main_success_rate.png", dpi=200)
    fig.savefig(out_dir / "fig1_po_main_success_rate.pdf")
    plt.close(fig)
    print(f"Saved: {out_dir / 'fig1_po_main_success_rate.png'}")


def fig2_retry_comparison(new_agg: dict, old_agg: dict, out_dir: Path) -> None:
    if not HAS_MPL:
        print("matplotlib not available, skip Fig 2")
        return

    tasks = [t for t in _get_tasks(new_agg) if t != "door-open-v3"]
    policies = ["text_buf", "struct_mem"]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(tasks))
    width = 0.18
    colors = {
        ("text_buf", "new"): "#3498db",
        ("text_buf", "old"): "#85c1e9",
        ("struct_mem", "new"): "#2ecc71",
        ("struct_mem", "old"): "#82e0aa",
    }

    offsets = [0, width, width * 2, width * 3]
    labels_done = set()

    for i, pol in enumerate(policies):
        for j, (agg, version, alpha) in enumerate([
            (new_agg, "retry=0", 1.0),
            (old_agg, "retry=3", 0.5),
        ]):
            means = [agg.get((t, pol), {}).get("success_rate_mean", 0.0) * 100 for t in tasks]
            stds = [agg.get((t, pol), {}).get("success_rate_std", 0.0) * 100 for t in tasks]
            label = f"{POLICY_LABELS[pol]} ({version})"
            if label in labels_done:
                label = None
            else:
                labels_done.add(label)
            color_key = (pol, "new" if "0" in version else "old")
            ax.bar(
                x + offsets[i * 2 + j], means, width,
                label=label, color=colors[color_key],
                yerr=stds, capsize=2, edgecolor="white", linewidth=0.5,
                alpha=alpha,
            )

    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_xlabel("Task", fontsize=12)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(tasks, fontsize=10)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title("Fig 2: Retry=0 vs Retry=3 Comparison", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_retry_comparison.png", dpi=200)
    fig.savefig(out_dir / "fig2_retry_comparison.pdf")
    plt.close(fig)
    print(f"Saved: {out_dir / 'fig2_retry_comparison.png'}")


def fig3_fullobs_vs_po(po_agg: dict, fo_agg: dict, out_dir: Path) -> None:
    if not HAS_MPL:
        print("matplotlib not available, skip Fig 3")
        return
    if not fo_agg:
        print("No full-observable data, skip Fig 3")
        return

    tasks = [t for t in _get_tasks(po_agg) if t != "door-open-v3" and (t, "no_mem") in fo_agg]
    policies = ["no_mem", "text_buf", "struct_mem"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, (agg, title) in zip(axes, [
        (po_agg, "Partial Observable"),
        (fo_agg, "Full Observable"),
    ]):
        x = np.arange(len(tasks))
        width = 0.25
        for i, pol in enumerate(policies):
            means = [agg.get((t, pol), {}).get("success_rate_mean", 0.0) * 100 for t in tasks]
            stds = [agg.get((t, pol), {}).get("success_rate_std", 0.0) * 100 for t in tasks]
            ax.bar(
                x + i * width, means, width,
                label=POLICY_LABELS[pol], color=POLICY_COLORS[pol],
                yerr=stds, capsize=3, edgecolor="white", linewidth=0.5,
            )
        ax.set_ylabel("Success Rate (%)", fontsize=12)
        ax.set_xlabel("Task", fontsize=12)
        ax.set_xticks(x + width)
        ax.set_xticklabels(tasks, fontsize=9)
        ax.set_ylim(0, 105)
        ax.set_title(title, fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].legend(fontsize=9)
    fig.suptitle("Fig 3: Partial Observable vs Full Observable", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_po_vs_fullobs.png", dpi=200)
    fig.savefig(out_dir / "fig3_po_vs_fullobs.pdf")
    plt.close(fig)
    print(f"Saved: {out_dir / 'fig3_po_vs_fullobs.png'}")


def table1_main(po_agg: dict, out_dir: Path) -> None:
    tasks = _get_tasks(po_agg)
    policies = ["no_mem", "text_buf", "struct_mem"]

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Main experiment success rates (\\%) under partial observability with retry=0. Mean $\\pm$ std over 5 seeds, 60 episodes per seed.}")
    lines.append("\\label{tab:main_results}")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\toprule")
    lines.append("Task & NoMemory & TextBuffer & StructMemory \\\\")
    lines.append("\\midrule")

    for task in tasks:
        vals = []
        for pol in policies:
            d = po_agg.get((task, pol))
            if d:
                vals.append(f"{d['success_rate_mean']*100:.1f} $\\pm$ {d['success_rate_std']*100:.1f}")
            else:
                vals.append("—")
        lines.append(f"{task} & {vals[0]} & {vals[1]} & {vals[2]} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    text = "\n".join(lines)
    (out_dir / "table1_main_results.tex").write_text(text, encoding="utf-8")
    print(f"Saved: {out_dir / 'table1_main_results.tex'}")


def table2_ablation(po_agg: dict, old_agg: dict, p3a_agg: dict, p3b_agg: dict, p3c_agg: dict, out_dir: Path) -> None:
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Ablation study results (success rate \\%). P3A: no goal flashing; P3B: no retry; P3C: no XY perturbation.}")
    lines.append("\\label{tab:ablation}")
    lines.append("\\begin{tabular}{llccc}")
    lines.append("\\toprule")
    lines.append("Ablation & Task & NoMemory & TextBuffer & StructMemory \\\\")
    lines.append("\\midrule")

    if p3a_agg:
        for task in _get_tasks(p3a_agg):
            vals = []
            for pol in ["no_mem", "text_buf", "struct_mem"]:
                d = p3a_agg.get((task, pol))
                vals.append(f"{d['success_rate_mean']*100:.1f}" if d else "—")
            lines.append(f"No Flash & {task} & {vals[0]} & {vals[1]} & {vals[2]} \\\\")

    if p3b_agg:
        lines.append("\\midrule")
        for task in _get_tasks(p3b_agg):
            vals = []
            for pol in ["text_buf", "struct_mem"]:
                d = p3b_agg.get((task, pol))
                vals.append(f"{d['success_rate_mean']*100:.1f}" if d else "—")
            lines.append(f"No Retry & {task} & — & {vals[0]} & {vals[1]} \\\\")

    if p3c_agg:
        lines.append("\\midrule")
        for task in _get_tasks(p3c_agg):
            d = p3c_agg.get((task, "struct_mem"))
            val = f"{d['success_rate_mean']*100:.1f}" if d else "—"
            lines.append(f"No Perturb & {task} & — & — & {val} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    text = "\n".join(lines)
    (out_dir / "table2_ablation.tex").write_text(text, encoding="utf-8")
    print(f"Saved: {out_dir / 'table2_ablation.tex'}")


def table3_failure(po_agg: dict, out_dir: Path) -> None:
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Failure case analysis: door-open-v3. All policies achieve 0\\% success rate. The bottleneck is handle grasping dynamics, not memory or controller design.}")
    lines.append("\\label{tab:failure}")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("Policy & Success Rate & Grasp Attempts & Grasp Success & Avg Reward \\\\")
    lines.append("\\midrule")

    for pol in ["no_mem", "text_buf", "struct_mem"]:
        d = po_agg.get(("door-open-v3", pol))
        if d:
            lines.append(
                f"{POLICY_LABELS[pol]} & {d['success_rate_mean']*100:.1f}\\% & "
                f"{d['avg_grasp_attempts']:.2f} & {d['avg_grasp_success']:.2f} & "
                f"{d['avg_reward']:.1f} \\\\"
            )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    text = "\n".join(lines)
    (out_dir / "table3_failure.tex").write_text(text, encoding="utf-8")
    print(f"Saved: {out_dir / 'table3_failure.tex'}")


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

    po_agg = _aggregate(_load_csv(Path(args.po_csv)))
    fo_agg = _aggregate(_load_csv(Path(args.full_obs_csv)))
    old_agg = _aggregate(_load_csv(Path(args.old_po_csv)))
    p3a_agg = _aggregate(_load_csv(Path(args.p3a_csv)))
    p3b_agg = _aggregate(_load_csv(Path(args.p3b_csv)))
    p3c_agg = _aggregate(_load_csv(Path(args.p3c_csv)))

    if po_agg:
        fig1_po_main(po_agg, out_dir)
        table1_main(po_agg, out_dir)
        table3_failure(po_agg, out_dir)

    if po_agg and old_agg:
        fig2_retry_comparison(po_agg, old_agg, out_dir)

    if po_agg and fo_agg:
        fig3_fullobs_vs_po(po_agg, fo_agg, out_dir)

    if po_agg:
        table2_ablation(po_agg, old_agg, p3a_agg, p3b_agg, p3c_agg, out_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
