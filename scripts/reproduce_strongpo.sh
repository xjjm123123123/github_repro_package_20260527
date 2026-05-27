#!/usr/bin/env bash
set -euo pipefail

python scripts/aggregate_strong_po.py || true
echo "StrongPO summaries are stored under results/main/strongpo_full.csv and results/main/strongpo_summary.csv"
