#!/usr/bin/env python3
"""
Generate result tables and figures from smallpool_v3_results.csv.

Outputs:
  artifacts/fig_success_rate.png   — grouped bar chart
  artifacts/fig_grasp_metrics.png  — grasp success & retry by task×policy
  artifacts/table_success_rate.txt — plain text table
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MPL = True
except Exception:
    HAS_MPL = False


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def aggregate(rows: list[dict[str, str]]) -> dict:
    by_tp: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_tp[(r["task"], r["policy"])].append(r)

    out: dict = {}
    for (task, policy), group in sorted(by_tp.items()):
        srs = [float(g["success_rate"]) for g in group]
        gras = [float(g["avg_grasp_attempts"]) for g in group]
        groks = [float(g["avg_grasp_success"]) for g in group]
        rets = [float(g["avg_retry_count"]) for g in group]
        rewards = [float(g["avg_reward"]) for g in group]
        n = len(srs)
        out[(task, policy)] = {
            "success_rate_mean": sum(srs) / n,
            "success_rate_std": (sum((s - sum(srs)/n)**2 for s in srs) / n) ** 0.5 if n > 1 else 0.0,
            "grasp_attempts_mean": sum(gras) / n,
            "grasp_success_mean": sum(groks) / n,
            "retry_mean": sum(rets) / n,
            "reward_mean": sum(rewards) / n,
            "n_seeds": n,
        }
    return out


def plot_success_rate(agg: dict, out_path: Path) -> None:
    if not HAS_MPL:
        print("matplotlib not available, skip figure")
        return

    tasks = sorted(set(k[0] for k in agg))
    policies = ["no_mem", "text_buf", "struct_mem"]
    policy_labels = {"no_mem": "NoMemory", "text_buf": "TextBuffer", "struct_mem": "StructMemory"}
    colors = {"no_mem": "#e74c3c", "text_buf": "#3498db", "struct_mem": "#2ecc71"}

    x = np.arange(len(tasks))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, pol in enumerate(policies):
        means = [agg.get((t, pol), {}).get("success_rate_mean", 0.0) * 100 for t in tasks]
        stds = [agg.get((t, pol), {}).get("success_rate_std", 0.0) * 100 for t in tasks]
        bars = ax.bar(x + i * width, means, width, label=policy_labels[pol],
                      color=colors[pol], yerr=stds, capsize=3, edgecolor="white", linewidth=0.5)
        for bar, m in zip(bars, means):
            if m > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                        f"{m:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_xlabel("Task", fontsize=12)
    ax.set_xticks(x + width)
    ax.set_xticklabels(tasks, fontsize=10)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10)
    ax.set_title("Success Rate by Task × Policy\n(Goal Flashing: every 20 steps, 1 step visible)", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_grasp_metrics(agg: dict, out_path: Path) -> None:
    if not HAS_MPL:
        return

    grasp_tasks = [t for t in sorted(set(k[0] for k in agg))
                   if any(agg.get((t, p), {}).get("grasp_attempts_mean", 0) > 0 for p in ["text_buf", "struct_mem"])]
    if not grasp_tasks:
        print("No grasp tasks, skip grasp figure")
        return

    policies = ["text_buf", "struct_mem"]
    policy_labels = {"text_buf": "TextBuffer", "struct_mem": "StructMemory"}
    colors = {"text_buf": "#3498db", "struct_mem": "#2ecc71"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    metrics = [
        ("grasp_attempts_mean", "Avg Grasp Attempts"),
        ("grasp_success_mean", "Avg Grasp Success"),
        ("retry_mean", "Avg Retry Count"),
    ]

    for ax, (metric_key, metric_label) in zip(axes, metrics):
        x = np.arange(len(grasp_tasks))
        width = 0.3
        for i, pol in enumerate(policies):
            vals = [agg.get((t, pol), {}).get(metric_key, 0.0) for t in grasp_tasks]
            ax.bar(x + i * width, vals, width, label=policy_labels[pol], color=colors[pol],
                   edgecolor="white", linewidth=0.5)
        ax.set_ylabel(metric_label, fontsize=10)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(grasp_tasks, fontsize=9)
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Grasp & Recovery Metrics by Task × Policy", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def print_table(agg: dict, out_path: Path) -> None:
    tasks = sorted(set(k[0] for k in agg))
    policies = ["no_mem", "text_buf", "struct_mem"]
    policy_labels = {"no_mem": "NoMemory", "text_buf": "TextBuffer", "struct_mem": "StructMemory"}

    lines = []
    lines.append("=" * 90)
    lines.append("Table 1: Success Rate (%) by Task × Policy (mean ± std over 5 seeds, 60 eps/seed)")
    lines.append("=" * 90)
    header = f"{'Task':<22} {'NoMemory':>15} {'TextBuffer':>15} {'StructMemory':>15} {'NM vs TB':>10} {'NM vs SM':>10}"
    lines.append(header)
    lines.append("-" * 90)

    for task in tasks:
        vals = {}
        for pol in policies:
            d = agg.get((task, pol), {})
            m = d.get("success_rate_mean", 0.0) * 100
            s = d.get("success_rate_std", 0.0) * 100
            vals[pol] = (m, s)

        nm = vals["no_mem"][0]
        tb = vals["text_buf"][0]
        sm = vals["struct_mem"][0]
        diff_tb = tb - nm
        diff_sm = sm - nm

        line = (f"{task:<22} "
                f"{vals['no_mem'][0]:>6.1f}±{vals['no_mem'][1]:<5.1f} "
                f"{vals['text_buf'][0]:>6.1f}±{vals['text_buf'][1]:<5.1f} "
                f"{vals['struct_mem'][0]:>6.1f}±{vals['struct_mem'][1]:<5.1f} "
                f"{diff_tb:>+8.1f} {diff_sm:>+8.1f}")
        lines.append(line)

    lines.append("-" * 90)
    lines.append("")

    grasp_tasks = [t for t in tasks if any(agg.get((t, p), {}).get("grasp_attempts_mean", 0) > 0 for p in policies)]
    if grasp_tasks:
        lines.append("=" * 90)
        lines.append("Table 2: Grasp & Recovery Metrics (mean over 5 seeds)")
        lines.append("=" * 90)
        header2 = f"{'Task':<22} {'Policy':<14} {'GraspAtt':>10} {'GraspOk':>10} {'Retry':>10} {'Reward':>10}"
        lines.append(header2)
        lines.append("-" * 90)
        for task in grasp_tasks:
            for pol in ["no_mem", "text_buf", "struct_mem"]:
                d = agg.get((task, pol), {})
                ga = d.get("grasp_attempts_mean", 0.0)
                gs = d.get("grasp_success_mean", 0.0)
                ry = d.get("retry_mean", 0.0)
                rw = d.get("reward_mean", 0.0)
                lines.append(f"{task:<22} {policy_labels[pol]:<14} {ga:>10.2f} {gs:>10.2f} {ry:>10.2f} {rw:>10.1f}")
            lines.append("-" * 90)

    text = "\n".join(lines)
    out_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nSaved: {out_path}")


def main() -> int:
    csv_path = Path("artifacts/smallpool_v3_results.csv")
    if not csv_path.exists():
        print(f"Missing: {csv_path}")
        return 1

    rows = load_csv(csv_path)
    agg = aggregate(rows)

    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    print_table(agg, out_dir / "table_success_rate.txt")
    plot_success_rate(agg, out_dir / "fig_success_rate.png")
    plot_grasp_metrics(agg, out_dir / "fig_grasp_metrics.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
