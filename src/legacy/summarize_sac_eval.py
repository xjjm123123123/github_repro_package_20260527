import json, glob
from collections import defaultdict

results = []
for f in sorted(glob.glob("artifacts/sac_baselines/*__FO.eval.json")
              + glob.glob("artifacts/sac_baselines/*__WeakPO.eval.json")
              + glob.glob("artifacts/sac_baselines/*__StrongPO.eval.json")):
    d = json.load(open(f))
    results.append(d)

table = defaultdict(dict)
for r in results:
    key = (r["task"], r["policy"], r["seed"])
    table[key][r["condition"]] = r["success_rate"]

print(f'{"task":<30} {"policy":<6} {"seed":<5} {"FO":>8} {"WeakPO":>8} {"StrongPO":>8}')
print("-" * 75)
for (task, pol, seed), conds in sorted(table.items()):
    fo = conds.get("FO", -1)
    wpo = conds.get("WeakPO", -1)
    spo = conds.get("StrongPO", -1)
    print(f"{task:<30} {pol:<6} {seed:<5} {fo:>8.3f} {wpo:>8.3f} {spo:>8.3f}")

print("\n=== Averaged over seeds ===")
avg_table = defaultdict(lambda: defaultdict(list))
for r in results:
    avg_table[(r["task"], r["policy"])][r["condition"]].append(r["success_rate"])

print(f'{"task":<30} {"policy":<6} {"FO":>12} {"WeakPO":>12} {"StrongPO":>12}')
print("-" * 78)
for (task, pol), conds in sorted(avg_table.items()):
    import numpy as np
    fo_vals = conds.get("FO", [0])
    wpo_vals = conds.get("WeakPO", [0])
    spo_vals = conds.get("StrongPO", [0])
    fo_str = f"{np.mean(fo_vals):.3f}±{np.std(fo_vals):.3f}"
    wpo_str = f"{np.mean(wpo_vals):.3f}±{np.std(wpo_vals):.3f}"
    spo_str = f"{np.mean(spo_vals):.3f}±{np.std(spo_vals):.3f}"
    print(f"{task:<30} {pol:<6} {fo_str:>12} {wpo_str:>12} {spo_str:>12}")
