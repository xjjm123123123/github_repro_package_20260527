#!/usr/bin/env python3
"""
Compute simple permutation tests and bootstrap confidence intervals for
Strong PO SAC evaluations.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _percentile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    idx = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return xs[idx]


def bootstrap_ci(xs: list[float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sample = [xs[rng.randrange(len(xs))] for _ in range(len(xs))]
        means.append(_mean(sample))
    return _percentile(means, 0.025), _percentile(means, 0.975)


def permutation_pvalue(xs: list[float], ys: list[float], n_perm: int = 10000, seed: int = 0) -> float:
    if not xs or not ys:
        return 1.0
    rng = random.Random(seed)
    observed = abs(_mean(xs) - _mean(ys))
    pooled = list(xs) + list(ys)
    nx = len(xs)
    ge = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        a = pooled[:nx]
        b = pooled[nx:]
        if abs(_mean(a) - _mean(b)) >= observed - 1e-12:
            ge += 1
    return (ge + 1) / (n_perm + 1)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input-csv", default="artifacts/strong_po/strong_po_eval_runs.csv")
    p.add_argument("--out-csv", default="artifacts/strong_po/strong_po_stats.csv")
    args = p.parse_args()

    rows = list(csv.DictReader(open(args.input_csv)))
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in rows:
        grouped[(r["task"], r["policy"], r["condition"])].append(float(r["success_rate"]))

    out_rows: list[dict[str, object]] = []

    # Within-policy comparisons
    for task in sorted({k[0] for k in grouped}):
        for policy in sorted({k[1] for k in grouped if k[0] == task}):
            fo = grouped.get((task, policy, "FO"), [])
            weak = grouped.get((task, policy, "WeakPO"), [])
            strong = grouped.get((task, policy, "StrongPO"), [])
            for lhs_name, lhs_vals, rhs_name, rhs_vals in [
                ("FO", fo, "StrongPO", strong),
                ("WeakPO", weak, "StrongPO", strong),
            ]:
                if not lhs_vals or not rhs_vals:
                    continue
                lo, hi = bootstrap_ci(rhs_vals, seed=17)
                out_rows.append(
                    {
                        "task": task,
                        "comparison": f"{policy}:{lhs_name}_vs_{rhs_name}",
                        "lhs_mean": _mean(lhs_vals),
                        "rhs_mean": _mean(rhs_vals),
                        "delta": _mean(lhs_vals) - _mean(rhs_vals),
                        "p_value_perm": permutation_pvalue(lhs_vals, rhs_vals, seed=17),
                        "rhs_ci95_lo": lo,
                        "rhs_ci95_hi": hi,
                        "n_lhs": len(lhs_vals),
                        "n_rhs": len(rhs_vals),
                    }
                )

        # Cross-policy comparison under StrongPO
        mlp = grouped.get((task, "mlp", "StrongPO"), [])
        lstm = grouped.get((task, "lstm", "StrongPO"), [])
        if mlp and lstm:
            lo, hi = bootstrap_ci(lstm, seed=29)
            out_rows.append(
                {
                    "task": task,
                    "comparison": "StrongPO:mlp_vs_lstm",
                    "lhs_mean": _mean(mlp),
                    "rhs_mean": _mean(lstm),
                    "delta": _mean(lstm) - _mean(mlp),
                    "p_value_perm": permutation_pvalue(mlp, lstm, seed=29),
                    "rhs_ci95_lo": lo,
                    "rhs_ci95_hi": hi,
                    "n_lhs": len(mlp),
                    "n_rhs": len(lstm),
                }
            )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "task",
                "comparison",
                "lhs_mean",
                "rhs_mean",
                "delta",
                "p_value_perm",
                "rhs_ci95_lo",
                "rhs_ci95_hi",
                "n_lhs",
                "n_rhs",
            ],
        )
        w.writeheader()
        w.writerows(out_rows)

    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

