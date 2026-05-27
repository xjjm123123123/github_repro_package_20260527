#!/usr/bin/env bash
set -euo pipefail

python scripts/run_metaworld_batch.py       --plan configs/metaworld_weakpo.yaml       --policies no_mem,text_buf,struct_mem       --run-dir results/generated_runs/table1

python scripts/aggregate_runs.py       --runs results/generated_runs/table1       --out results/main/table1_metaworld_reproduced.csv

python src/plotting/plot_table1.py       --input results/main/table1_metaworld_reproduced.csv       --output results/tables/table1_summary_reproduced.csv
