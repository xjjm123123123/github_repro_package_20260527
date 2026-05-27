#!/usr/bin/env python3
"""Generate paper-quality figures from unified data sources.

Replaces the old generate_paper_figures.py which was based on the old 4-task
data structure.  This version reads from the unified 9-task data pipeline.

Figures:
  Fig 1: FO vs PO Main Results Bar Chart  (9 tasks x 4 groups)
  Fig 2: Memory Necessity Curve            (flash_every ablation, 95% CI)
  Fig 3: Retry Budget Consumption          (retry=0 vs retry=3)

Usage:
  python3 scripts/generate_unified_figures.py
  python3 scripts/generate_unified_figures.py --out-dir artifacts/v2_unified_main/figures/
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


TASKS_ORDER = [
    "reach-v3", "push-v3", "pick-place-v3",
    "door-open-v3", "drawer-open-v3", "faucet-open-v3",
    "button-press-topdown-v3", "sweep-v3", "shelf-place-v3",
]

POLICIES_ORDER = ["no_mem", "text_buf", "struct_mem"]

POLICY_LABELS = {"no_mem": "NoMemory", "text_buf": "TextBuffer", "struct_mem": "StructMemory"}
POLICY_COLORS = {"no_mem": "#e74c3c", "text_buf": "#3498db", "struct_mem": "#2ecc71"}

FO_COLOR = "#7f8c8d"
FO_LABEL = "FO Baseline"

FLASH_EVERY_XS = [1, 5, 10, 20, 50]


def _load_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        print(f"  WARNING: {path} not found")
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _save_fig(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / name}.png / .pdf")


def bootstrap_ci(data: list[float], n_boot: int = 10000, ci: float = 0.95) -> tuple[float, float]:
    arr = np.array(data, dtype=float)
    if len(arr) < 2:
        v = float(arr[0]) if len(arr) == 1 else 0.0
        return v, v
    boots = [
        float(np.random.choice(arr, size=len(arr), replace=True).mean())
        for _ in range(n_boot)
    ]
    alpha = (1 - ci) / 2
    lo = float(np.percentile(boots, alpha * 100))
    hi = float(np.percentile(boots, (1 - alpha) * 100))
    return lo, hi


def _parse_statistical_tests(stat_path: Path) -> dict[str, str]:
    sig_map: dict[str, str] = {}
    if not stat_path.exists():
        return sig_map
    text = stat_path.read_text(encoding="utf-8")
    in_section = False
    for line in text.split("\n"):
        if "NoMemory FO vs PO" in line:
            in_section = True
            continue
        if in_section and re.match(r"\s*\d+\.", line):
            break
        if in_section:
            m = re.match(
                r"\s+(\S+):\s+diff=[\d.]+\s+p=[\d.]+\s+(\*+|n\.s\.)", line
            )
            if m:
                sig_map[m.group(1)] = m.group(2)
    return sig_map


def _read_flash_every_seed_data(root: Path) -> dict[tuple[str, str, int], list[float]]:
    result: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    search_dirs = [
        root / "runs" / "flash_every_ablation",
        root / "runs" / "batch" / "flash_every_sweep_door",
    ]
    for base in search_dirs:
        if not base.is_dir():
            continue
        for config_path in sorted(base.rglob("config.json")):
            try:
                with config_path.open("r", encoding="utf-8") as f:
                    config = json.load(f)
            except Exception:
                continue
            task = config.get("task", "")
            policy = config.get("policy", "")
            fe = config.get("flash_every")
            if not task or not policy or fe is None:
                continue
            try:
                fe = int(fe)
            except (ValueError, TypeError):
                continue
            if fe not in FLASH_EVERY_XS:
                continue
            if policy not in POLICIES_ORDER:
                continue
            summary_path = config_path.parent / "summary.json"
            if not summary_path.exists():
                continue
            try:
                with summary_path.open("r", encoding="utf-8") as f:
                    summary = json.load(f)
            except Exception:
                continue
            sr = float(summary.get("success_rate", 0))
            result[(task, policy, fe)].append(sr)
    return dict(result)


def _read_retry3_from_runs(root: Path) -> dict[tuple[str, str], list[float]]:
    csv_path = root / "retry3_ablation_5s.csv"
    if csv_path.exists():
        result: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in _load_csv(csv_path):
            task = row.get("task", "")
            policy = row.get("policy", "")
            if not task or policy not in POLICIES_ORDER:
                continue
            try:
                sr = float(row.get("success_rate", 0))
            except (TypeError, ValueError):
                continue
            result[(task, policy)].append(sr)
        if result:
            return dict(result)

    for candidate in [
        root / "runs" / "retry_budget_ablation" / "retry_budget_ablation" / "retry3",
        root / "runs" / "retry_budget_ablation" / "retry3",
    ]:
        if candidate.is_dir():
            runs_dir = candidate
            break
    else:
        print("  WARNING: retry3 runs directory not found")
        return {}

    result: dict[tuple[str, str], list[float]] = defaultdict(list)
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        try:
            with summary_path.open("r", encoding="utf-8") as f:
                summary = json.load(f)
        except Exception:
            continue
        exp_id = summary.get("exp_id", "")
        m = re.match(
            r"mw-(.+)-(no_mem|text_buf|struct_mem)-\d{8}-\d{6}-seed(\d+)", exp_id
        )
        if not m:
            continue
        task, policy = m.group(1), m.group(2)
        result[(task, policy)].append(float(summary.get("success_rate", 0)))
    return dict(result)


# ---------------------------------------------------------------------------
# Figure 1: FO vs PO Main Results Bar Chart
# ---------------------------------------------------------------------------

def fig1_fo_vs_po_main(root: Path, out_dir: Path) -> None:
    print("\n[Fig 1] FO vs PO Main Results Bar Chart")
    csv_path = root / "v2_unified_main" / "unified_main_table.csv"
    stat_path = root / "v2_unified_main" / "statistical_tests.txt"

    rows = _load_csv(csv_path)
    if not rows:
        print("  SKIP: no unified_main_table.csv data")
        return

    grouped: dict[tuple[str, str, str], tuple[float, float]] = {}
    for r in rows:
        key = (r["task"], r["policy"], r["condition"])
        grouped[key] = (float(r["mean"]), float(r["std"]))

    sig_map = _parse_statistical_tests(stat_path)
    tasks = [t for t in TASKS_ORDER if any(k[0] == t for k in grouped)]

    fig, ax = plt.subplots(figsize=(14, 5.5))
    x = np.arange(len(tasks))
    width = 0.18

    bar_specs = [
        (FO_LABEL, FO_COLOR, "no_mem", "FO", 0.55),
        ("PO-NoMemory", POLICY_COLORS["no_mem"], "no_mem", "PO", 0.90),
        ("PO-TextBuffer", POLICY_COLORS["text_buf"], "text_buf", "PO", 0.90),
        ("PO-StructMemory", POLICY_COLORS["struct_mem"], "struct_mem", "PO", 0.90),
    ]

    for i, (label, color, pol, cond, alpha) in enumerate(bar_specs):
        means, stds = [], []
        for t in tasks:
            d = grouped.get((t, pol, cond), (0.0, 0.0))
            means.append(d[0] * 100)
            stds.append(d[1] * 100)
        ax.bar(
            x + i * width, means, width,
            label=label, color=color, alpha=alpha,
            yerr=stds, capsize=3, edgecolor="white", linewidth=0.5,
        )

    for j, t in enumerate(tasks):
        fo_d = grouped.get((t, "no_mem", "FO"))
        po_d = grouped.get((t, "no_mem", "PO"))
        if fo_d and po_d:
            gap = (fo_d[0] - po_d[0]) * 100
            if gap > 5:
                y_top = fo_d[0] * 100 + fo_d[1] * 100 + 4
                x_left = x[j]
                x_right = x[j] + width
                ax.plot(
                    [x_left, x_left, x_right, x_right],
                    [y_top - 1, y_top, y_top, y_top - 1],
                    lw=0.8, color="#555555",
                )
                ax.text(
                    (x_left + x_right) / 2, y_top + 1,
                    f"Δ{gap:.0f}pp",
                    ha="center", va="bottom", fontsize=7,
                    fontweight="bold", color="#555555",
                )

    for j, t in enumerate(tasks):
        sig = sig_map.get(t, "")
        if sig and sig != "n.s.":
            y_max = max(
                grouped.get((t, pol, cond), (0.0, 0.0))[0] * 100
                + grouped.get((t, pol, cond), (0.0, 0.0))[1] * 100
                for _, _, pol, cond, _ in bar_specs
            )
            ax.text(
                x[j] + 1.5 * width, y_max + 6,
                sig, ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#c0392b",
            )

    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_xlabel("Task", fontsize=12)
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(tasks, fontsize=10, rotation=30, ha="right")
    ax.set_ylim(0, 115)
    ax.legend(fontsize=9, loc="upper right", ncol=2)
    ax.set_title("FO vs PO: Main Results across 9 Tasks", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    fig.tight_layout()
    _save_fig(fig, out_dir, "fig1_fo_vs_po_main")


# ---------------------------------------------------------------------------
# Figure 2: Memory Necessity Curve (flash_every ablation)
# ---------------------------------------------------------------------------

def fig2_flash_every_curve(root: Path, out_dir: Path) -> None:
    print("\n[Fig 2] Memory Necessity Curve (flash_every)")
    csv_5seeds = root / "flash_every_ablation" / "flash_every_results_5seeds.csv"
    csv_sweep = root / "flash_every_ablation" / "flash_every_results_sweep_door.csv"

    rows_5 = _load_csv(csv_5seeds)
    rows_sw = _load_csv(csv_sweep) if csv_sweep.exists() else []

    if not rows_5 and not rows_sw:
        print("  SKIP: no flash_every data")
        return

    seed_data = _read_flash_every_seed_data(root)

    np.random.seed(42)

    ci_data: dict[tuple[str, str, int], dict[str, float]] = {}

    for r in rows_5:
        fe_raw = r.get("flash_every", "")
        try:
            fe = int(fe_raw)
        except ValueError:
            continue
        if fe not in FLASH_EVERY_XS:
            continue
        task, policy = r["task"], r["policy"]
        mean_val = float(r.get("success_rate_mean", r.get("mean", 0)))
        std_val = float(r.get("success_rate_std", r.get("std", 0)))
        n = int(r.get("n_seeds", 0))

        seed_key = (task, policy, fe)
        if seed_key in seed_data and len(seed_data[seed_key]) >= 2:
            lo, hi = bootstrap_ci(seed_data[seed_key], n_boot=10000)
        else:
            ci_lo_csv = float(r.get("ci_lo", 0))
            ci_hi_csv = float(r.get("ci_hi", 0))
            if ci_lo_csv > 0 or ci_hi_csv > 0:
                lo, hi = ci_lo_csv, ci_hi_csv
            elif n > 1:
                se = std_val / (n ** 0.5)
                lo = max(0.0, mean_val - 1.96 * se)
                hi = min(1.0, mean_val + 1.96 * se)
            else:
                lo, hi = mean_val, mean_val

        ci_data[(task, policy, fe)] = {"mean": mean_val, "ci_lo": lo, "ci_hi": hi}

    for r in rows_sw:
        fe_raw = r.get("flash_every", "")
        try:
            fe = int(fe_raw)
        except ValueError:
            continue
        if fe not in FLASH_EVERY_XS:
            continue
        task, policy = r["task"], r["policy"]
        mean_val = float(r.get("mean", 0))
        std_val = float(r.get("std", 0))
        n = int(r.get("n_seeds", 0))

        seed_key = (task, policy, fe)
        if seed_key in seed_data and len(seed_data[seed_key]) >= 2:
            lo, hi = bootstrap_ci(seed_data[seed_key], n_boot=10000)
        elif "ci_lo" in r and r["ci_lo"] and "ci_hi" in r and r["ci_hi"]:
            lo = float(r["ci_lo"])
            hi = float(r["ci_hi"])
        elif n > 1:
            se = std_val / (n ** 0.5)
            lo = max(0.0, mean_val - 1.96 * se)
            hi = min(1.0, mean_val + 1.96 * se)
        else:
            lo, hi = mean_val, mean_val

        ci_data[(task, policy, fe)] = {"mean": mean_val, "ci_lo": lo, "ci_hi": hi}

    all_tasks = sorted(set(k[0] for k in ci_data))
    tasks = [
        t for t in all_tasks
        if any((t, p, fe) in ci_data for p in POLICIES_ORDER for fe in FLASH_EVERY_XS)
    ]
    if not tasks:
        print("  SKIP: no plottable flash_every data")
        return

    n_tasks = len(tasks)
    ncols = min(3, n_tasks)
    nrows = (n_tasks + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False
    )

    for idx, task in enumerate(tasks):
        ax = axes[idx // ncols][idx % ncols]
        for policy in POLICIES_ORDER:
            fe_vals, means, ci_los, ci_his = [], [], [], []
            for fe in FLASH_EVERY_XS:
                key = (task, policy, fe)
                if key in ci_data:
                    d = ci_data[key]
                    fe_vals.append(fe)
                    means.append(d["mean"] * 100)
                    ci_los.append(d["ci_lo"] * 100)
                    ci_his.append(d["ci_hi"] * 100)
            if not fe_vals:
                continue
            ax.plot(
                fe_vals, means, "o-",
                color=POLICY_COLORS[policy], label=POLICY_LABELS[policy],
                linewidth=2, markersize=5,
            )
            ax.fill_between(
                fe_vals, ci_los, ci_his,
                color=POLICY_COLORS[policy], alpha=0.15,
            )

        ax.set_xscale("log")
        ax.set_xticks(FLASH_EVERY_XS)
        ax.set_xticklabels([str(v) for v in FLASH_EVERY_XS], fontsize=9)
        ax.set_xlabel("flash_every", fontsize=10)
        ax.set_ylabel("Success Rate (%)", fontsize=10)
        ax.set_title(task, fontsize=11, fontweight="bold")
        ax.set_ylim(-5, 105)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx == 0:
            ax.legend(fontsize=8, loc="lower left")

    for idx in range(n_tasks, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(
        "Memory Necessity: Success Rate vs Goal-Visibility Frequency\n"
        "(shading = 95% CI via bootstrap, 10000 resamples)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    _save_fig(fig, out_dir, "fig2_flash_every_necessity_curve")


# ---------------------------------------------------------------------------
# Figure 3: Retry Budget Consumption
# ---------------------------------------------------------------------------

def fig3_retry_budget(root: Path, out_dir: Path) -> None:
    print("\n[Fig 3] Retry Budget Consumption")
    csv_path = root / "v2_unified_main" / "unified_main_table.csv"
    rows = _load_csv(csv_path)

    retry0: dict[tuple[str, str], tuple[float, float]] = {}
    for r in rows:
        if r.get("condition") != "PO":
            continue
        retry0[(r["task"], r["policy"])] = (float(r["mean"]), float(r["std"]))

    retry3_raw = _read_retry3_from_runs(root)
    retry3: dict[tuple[str, str], tuple[float, float]] = {}
    for (task, policy), srs in retry3_raw.items():
        n = len(srs)
        if n == 0:
            continue
        mean = sum(srs) / n
        std = (sum((s - mean) ** 2 for s in srs) / (n - 1)) ** 0.5 if n > 1 else 0.0
        retry3[(task, policy)] = (mean, std)

    if not retry0 and not retry3:
        print("  SKIP: no retry budget data")
        return

    all_keys = set(retry0.keys()) | set(retry3.keys())
    tasks = sorted(set(k[0] for k in all_keys))
    policies = ["text_buf", "struct_mem"]

    if "pick-place-v3" in tasks:
        tasks = ["pick-place-v3"] + [t for t in tasks if t != "pick-place-v3"]

    fig, ax = plt.subplots(figsize=(max(8, len(tasks) * 2.5), 5))
    x = np.arange(len(tasks))
    width = 0.18

    color_map = {
        ("text_buf", 0): "#3498db",
        ("text_buf", 3): "#85c1e9",
        ("struct_mem", 0): "#2ecc71",
        ("struct_mem", 3): "#82e0aa",
    }
    offsets = [0, width, 2 * width, 3 * width]
    seen_labels: set[str] = set()

    for i, pol in enumerate(policies):
        for j, (src, rv) in enumerate([(retry0, 0), (retry3, 3)]):
            means, stds = [], []
            for t in tasks:
                d = src.get((t, pol))
                means.append(d[0] * 100 if d else 0.0)
                stds.append(d[1] * 100 if d else 0.0)
            lbl = f"{POLICY_LABELS[pol]} (retry={rv})"
            if lbl in seen_labels:
                lbl = None
            else:
                seen_labels.add(lbl)
            ax.bar(
                x + offsets[i * 2 + j], means, width,
                label=lbl, color=color_map.get((pol, rv), "#999999"),
                yerr=stds, capsize=2, edgecolor="white", linewidth=0.5,
                alpha=1.0 if rv == 0 else 0.65,
            )

    if "pick-place-v3" in tasks:
        pp_idx = tasks.index("pick-place-v3")
        ax.axvspan(pp_idx - 0.4, pp_idx + 3.5 * width + 0.2, color="#f39c12", alpha=0.08)
        y_max = max(
            max(src.get((t, pol), (0, 0))[0] * 100 for t in tasks)
            for src in [retry0, retry3] for pol in policies
        )
        ax.text(
            pp_idx + 1.5 * width, y_max * 100 + 8 if y_max < 1 else y_max + 8,
            "★ primary", ha="center", va="bottom", fontsize=8,
            color="#f39c12", fontweight="bold",
        )

    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_xlabel("Task", fontsize=12)
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(tasks, fontsize=10)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title("Retry Budget: retry=0 vs retry=3", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save_fig(fig, out_dir, "fig3_retry_budget_consumption")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Generate unified paper figures")
    parser.add_argument("--root", default="artifacts")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_absolute():
        root = Path(__file__).resolve().parent.parent / root

    out_dir = (
        Path(args.out_dir) if args.out_dir
        else root / "v2_unified_main" / "figures"
    )
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parent.parent / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Root:   {root}")
    print(f"Output: {out_dir}")

    fig1_fo_vs_po_main(root, out_dir)
    fig2_flash_every_curve(root, out_dir)
    fig3_retry_budget(root, out_dir)

    print(f"\nDone! All figures saved to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
