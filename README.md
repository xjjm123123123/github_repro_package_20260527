# GitHub Attachment Package

This folder contains a lightweight supplementary package prepared for GitHub upload alongside the paper submission.

## Contents

- `paper/`
  - `corl2026_camera_ready_final.pdf`: final camera-ready paper
  - `main.tex`, `macros.tex`, `corl_2026.sty`, `corlabbrvnat.bst`
  - `sections/`: LaTeX section sources
  - `figures/`: main figure source files used in the paper
  - `tables/`: table source files used in the paper
- `data/`
  - selected experiment metadata and result summaries copied from `runs/`
  - includes only lightweight files such as `config.json`, `summary.json`, and `batch_config.txt`
- `remote_artifacts/`
  - selected lightweight files pulled from the remote AutoResearchClaw server
  - includes aggregated `csv`, `txt`, `tex`, `pdf`, `png`, and evaluation `json` files
  - excludes heavy training logs, step-level traces, and model checkpoints

## Included Experiment Blocks

- `ablation/`
- `flash_every_ablation/`
- `batch/full_observable/`
- `batch/mw-batch-20260410-171844/`
- `batch/retry0_main/`
- `batch/retry3_ablation_5s/`
- `batch/retry_fine_grained/`
- `batch/flash_every_sweep_door/`
- `batch/v2_rerun_core3/`

## Excluded Files

To keep the GitHub attachment package lightweight and easy to browse, the following files are intentionally excluded:

- step-level logs such as `events.jsonl`
- episode-level logs such as `metrics.jsonl`
- rendered media files
- backup directories
- temporary LaTeX build artifacts
- remote training logs such as `*.train.log` and `*.eval.log`
- remote model checkpoint directories

## Suggested Use

- upload this folder as the supplementary GitHub repository contents
- use `paper/corl2026_camera_ready_final.pdf` as the paper artifact
- use the `data/` directory for lightweight reproducibility evidence and experiment configuration tracing
- use the `remote_artifacts/` directory for server-side aggregated result files and exported figures
