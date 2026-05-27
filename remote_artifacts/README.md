# Remote Artifacts

This directory contains lightweight artifacts copied from the remote AutoResearchClaw server under `/root/AutoResearchClaw/artifacts`.

## Included Content

- aggregated result tables in `csv`
- exported figure files in `pdf` and `png`
- lightweight evaluation outputs in `json`
- helper notes and table fragments in `txt`, `md`, and `tex`

## Included Subdirectories

- `flash_every_ablation/`
- `goal_buffer/`
- `multi_goal/`
- `retry_fine_grained/`
- `sac_baselines/`
- `strong_po/`

## Included Top-Level Files

- `metaworld_results.csv`
- `full_observable_results.csv`
- `retry0_main_results.csv`
- `retry3_ablation_5s.csv`
- `v2_rerun_core3_results.csv`
- `comparison_po_vs_fullobs.csv`
- `comparison_retry0_vs_old.csv`
- exported figure files such as `fig1_po_main_success_rate.pdf`
- table summary files such as `table1_main_results.tex`

## Excluded Content

To keep the package small and GitHub-friendly, the following remote content is excluded:

- `*.train.log`
- `*.eval.log`
- raw per-step traces
- model checkpoints
- unrelated hidden files
