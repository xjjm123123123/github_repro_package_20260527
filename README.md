# When Does Memory Matter?

Code and lightweight reproducibility artifacts for the CoRL 2026 paper:

**When Does Memory Matter? A Diagnostic Framework for Partially Observable Manipulation**

## Overview

This repository implements controlled partial-observability diagnostics for robotic manipulation. It includes:

- WeakPO and StrongPO wrappers
- NoMemory, TextBuffer, and StructMemory rule-based variants
- GoalBuffer evaluation for learned policies
- SAC/MLP and SAC/LSTM baseline evaluation artifacts
- MetaWorld main results, StrongPO summaries, flash-frequency ablations, and Fetch sanity-check scaffolds

The primary reproducible results are the controlled rule-based diagnostics. Learned-policy baselines are included as supplementary experiments and lightweight evaluation artifacts.

## Repository Layout

```text
README.md
LICENSE
requirements.txt
environment.yml
paper/
src/
configs/
scripts/
results/
figures/
```

## Installation

### Conda

```bash
conda env create -f environment.yml
conda activate memory-diagnostic
```

### Pip

```bash
python -m pip install -r requirements.txt
```

## Main Components

- `src/envs/po_wrapper.py`: WeakPO and StrongPO wrappers
- `src/memory/`: NoMemory, TextBuffer, StructMemory, and GoalBuffer helpers
- `src/controllers/`: controller exports used by the rule-based policies
- `src/eval/`: evaluation entry points
- `src/legacy/`: preserved experiment scripts used to generate the included artifacts
- `configs/`: minimal configs for main experiments plus a preserved legacy plan
- `results/`: aggregate CSVs, evaluation JSONs, table fragments, and run summaries
- `figures/`: exported analysis figures and paper figure source files

## Reproducing Main Results

### Table 1: MetaWorld WeakPO results

```bash
bash scripts/reproduce_table1.sh
```

This runs the rule-based MetaWorld batch and re-aggregates the output to `results/main/table1_metaworld_reproduced.csv`.

### Figure 2: task-by-condition summary from aggregate results

```bash
bash scripts/reproduce_figure2.sh
```

This uses the included aggregate CSV and regenerates a heatmap-style summary figure.

### StrongPO summary

```bash
bash scripts/reproduce_strongpo.sh
```

Bundled StrongPO summaries are stored in:

- `results/main/strongpo_full.csv`
- `results/main/strongpo_summary.csv`

### GoalBuffer evaluation

```bash
bash scripts/reproduce_goalbuffer.sh
```

This aggregates bundled GoalBuffer evaluation JSON files and regenerates a summary CSV/figure.

### Flash-frequency ablation

```bash
bash scripts/reproduce_flash_ablation.sh
```

### Fetch cross-platform sanity check

```bash
bash scripts/reproduce_fetch.sh
```

The lightweight package includes the Fetch evaluation scaffold and the cross-platform table source. Raw Fetch run logs are not bundled.

## Results Mapping

- `results/main/table1_metaworld.csv`: aggregate MetaWorld results used for the main table
- `results/ablations/flash_ablation.csv`: flash-frequency ablation summary
- `results/ablations/retry_ablation.csv`: retry ablation summary
- `results/baselines/goalbuffer_results.csv`: aggregated GoalBuffer eval JSONs
- `results/baselines/sac_baselines_results.csv`: aggregated SAC/LSTM eval JSONs
- `results/multi_goal/`: multi-goal summaries
- `results/tables/`: paper-oriented table fragments and text summaries

## Notes

- This repository is organized as a reproducibility-oriented supplement, not as a full training platform.
- The `src/legacy/` directory preserves the original experiment scripts so the repository stays close to the code that produced the bundled artifacts.
- The camera-ready paper source and PDF remain under `paper/` for traceability, but the core supplementary value is the code, configs, and results.

## Citation

```bibtex
@inproceedings{xu2026memory,
  title={When Does Memory Matter? A Diagnostic Framework for Partially Observable Manipulation},
  author={Xu, Jiaming and Chang, Danni},
  booktitle={Conference on Robot Learning},
  year={2026}
}
```
