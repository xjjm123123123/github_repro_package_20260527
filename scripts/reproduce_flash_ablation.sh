#!/usr/bin/env bash
set -euo pipefail

python scripts/run_flash_every_sweep_door.py --out-dir results/generated_runs/flash_every
python scripts/aggregate_flash_every_v2.py
