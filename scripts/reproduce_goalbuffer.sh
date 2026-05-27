#!/usr/bin/env bash
set -euo pipefail

python src/plotting/plot_goalbuffer.py       --input-dir results/baselines/goal_buffer       --csv-out results/baselines/goalbuffer_results.csv       --fig-out figures/figure4_goalbuffer.png
