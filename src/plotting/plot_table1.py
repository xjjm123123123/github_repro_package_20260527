"""Create a compact Table-1 style summary from aggregate MetaWorld results."""

from __future__ import annotations
import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="results/main/table1_metaworld.csv")
    p.add_argument("--output", default="results/tables/table1_summary.csv")
    args = p.parse_args()

    with Path(args.input).open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    grouped = defaultdict(list)
    for row in rows:
        key = (row.get("task", ""), row.get("policy", ""), row.get("condition", ""))
        try:
            grouped[key].append(float(row.get("success_rate", 0) or 0))
        except ValueError:
            grouped[key].append(0.0)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "policy", "condition", "mean_success_rate"])
        for key in sorted(grouped):
            vals = grouped[key]
            w.writerow([*key, sum(vals) / max(1, len(vals))])

    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
