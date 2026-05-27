#!/usr/bin/env python3
"""Aggregate v2_final_results into paper-ready tables.

Maps stdout.log filenames (task__policy__FO/PO__seed.stdout.log) to
the actual summary.json directories, then computes mean±std across seeds.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts/v2_final_results")

    # Collect all summary.json files
    summaries = sorted(root.glob("mw-*/summary.json"))

    # Parse exp_id to determine task, policy, seed
    # exp_id format: mw-{task}-{policy}-{timestamp}-seed{N}
    # But we need FO/PO info which is NOT in summary.json
    # Strategy: match by timestamp ordering with stdout.log filenames

    # First, build mapping from stdout.log filenames
    log_map = {}
    for log_file in sorted(root.glob("*__*.stdout.log")):
        name = log_file.name.replace(".stdout.log", "")
        parts = name.split("__")
        if len(parts) == 4:
            task, policy, fo_po, seed_str = parts
            seed = int(seed_str.replace("s", ""))
            log_map[name] = {
                "task": task,
                "policy": policy,
                "condition": fo_po,
                "seed": seed,
                "log_file": log_file,
            }

    # Read each summary.json and match to log entry by task+policy+seed+timestamp
    # Since FO runs before PO for same task+policy+seed, and timestamps are ordered,
    # we can match by looking at the stdout.log content for each summary dir
    results = []

    for sp in summaries:
        try:
            with open(sp) as f:
                s = json.load(f)
        except Exception:
            continue

        exp_id = s.get("exp_id", "")
        # Parse exp_id: mw-{task}-{policy}-{timestamp}-seed{N}
        m = re.match(r"mw-(.+)-(\d{8}-\d{6})-seed(\d+)", exp_id)
        if not m:
            continue

        # Extract task and policy from exp_id
        # Task can have hyphens, so we need to be careful
        # Format: mw-{task}-{policy}-{timestamp}-seed{N}
        # policy is one of: no_mem, text_buf, struct_mem
        policy_match = re.match(r"mw-(.+)-(no_mem|text_buf|struct_mem)-(\d{8}-\d{6})-seed(\d+)", exp_id)
        if not policy_match:
            continue

        task = policy_match.group(1)
        policy = policy_match.group(2)
        seed = int(policy_match.group(4))

        # Find which stdout.log this corresponds to
        # Check all matching log entries for this task+policy+seed
        matching_logs = []
        for name, info in log_map.items():
            if info["task"] == task and info["policy"] == policy and info["seed"] == seed:
                matching_logs.append((name, info))

        if not matching_logs:
            continue

        # Read the stdout.log for this summary to match
        summary_dir = sp.parent
        stdout_in_dir = list(summary_dir.glob("*.log")) + list(summary_dir.glob("*.jsonl"))
        # Actually, stdout.log is in root, not in the summary dir
        # Better approach: read the first line of each matching stdout.log
        # and check if the success_rate matches

        sr = s.get("success_rate", 0)
        for name, info in matching_logs:
            log_path = info["log_file"]
            # Check if this log's last line matches the success rate
            try:
                last_line = log_path.read_text().strip().split("\n")[-1]
                # Last line format: [60/60] success=X steps=Y ...
                # or "Summary: ..."
                if "Summary:" in last_line:
                    # This is the summary line, check the summary.json instead
                    pass
            except Exception:
                pass

        # Simpler approach: since FO always runs before PO in the batch script,
        # and timestamps are ordered, we can use timestamp ordering
        # But even simpler: just read the config.json if it exists
        config_path = sp.parent / "config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                full_obs = config.get("full_observable", False)
                condition = "FO" if full_obs else "PO"
            except Exception:
                continue
        else:
            # Fallback: use timestamp ordering within same task+policy+seed
            # Group summaries by task+policy+seed, sort by timestamp
            pass

        results.append({
            "task": task,
            "policy": policy,
            "condition": condition,
            "seed": seed,
            "success_rate": sr,
            "avg_steps": s.get("avg_steps", 0),
            "avg_reward": s.get("avg_reward", 0),
            "flash_every": s.get("flash_every", 0),
            "flash_len": s.get("flash_len", 0),
        })

    if not results:
        print("No results found!")
        return

    # Group by task × policy × condition
    grouped = defaultdict(list)
    for r in results:
        key = (r["task"], r["policy"], r["condition"])
        grouped[key].append(r["success_rate"])

    # Print table
    tasks = ["door-open-v3", "drawer-open-v3", "faucet-open-v3",
             "button-press-topdown-v3", "sweep-v3", "shelf-place-v3"]
    policies = ["no_mem", "text_buf", "struct_mem"]
    conditions = ["FO", "PO"]

    print("\n=== Success Rate (mean over 5 seeds × 60 eps) ===\n")
    header = f"{'Task':<28} {'Cond':<5} " + " ".join(f"{p:<14}" for p in policies)
    print(header)
    print("-" * len(header))

    for task in tasks:
        for cond in conditions:
            vals = []
            for pol in policies:
                key = (task, pol, cond)
                if key in grouped:
                    srs = grouped[key]
                    mean = sum(srs) / len(srs)
                    if len(srs) > 1:
                        std = (sum((x - mean) ** 2 for x in srs) / (len(srs) - 1)) ** 0.5
                        vals.append(f"{mean:.3f}±{std:.3f}")
                    else:
                        vals.append(f"{mean:.3f}")
                else:
                    vals.append("—")
            line = f"{task:<28} {cond:<5} " + " ".join(f"{v:<14}" for v in vals)
            print(line)
        print()

    # Print FO-PO gap
    print("\n=== FO-PO Gap (FO - PO, in percentage points) ===\n")
    header = f"{'Task':<28} " + " ".join(f"{p:<14}" for p in policies)
    print(header)
    print("-" * len(header))

    for task in tasks:
        vals = []
        for pol in policies:
            fo_key = (task, pol, "FO")
            po_key = (task, pol, "PO")
            if fo_key in grouped and po_key in grouped:
                fo_mean = sum(grouped[fo_key]) / len(grouped[fo_key])
                po_mean = sum(grouped[po_key]) / len(grouped[po_key])
                gap = (fo_mean - po_mean) * 100
                vals.append(f"{gap:+.1f}pp")
            else:
                vals.append("—")
        line = f"{task:<28} " + " ".join(f"{v:<14}" for v in vals)
        print(line)

    # Save CSV
    csv_path = root / "paper_table_results.csv"
    with open(csv_path, "w") as f:
        f.write("task,policy,condition,seed,success_rate\n")
        for r in sorted(results, key=lambda x: (x["task"], x["policy"], x["condition"], x["seed"])):
            f.write(f"{r['task']},{r['policy']},{r['condition']},{r['seed']},{r['success_rate']}\n")
    print(f"\nCSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
