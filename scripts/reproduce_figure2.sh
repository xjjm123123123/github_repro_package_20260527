#!/usr/bin/env bash
set -euo pipefail

python src/plotting/plot_figure2.py       --input results/main/table1_metaworld.csv       --output figures/figure2_goal_dependency.png       --policy no_mem
