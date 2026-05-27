#!/usr/bin/env bash
set -euo pipefail

python scripts/precheck_gymrobot.py
python scripts/run_gymrobot_rulebased.py --flash-every 5 --flash-len 1 --out-dir results/fetch/generated
