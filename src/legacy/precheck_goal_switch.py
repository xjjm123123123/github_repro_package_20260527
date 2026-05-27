#!/usr/bin/env python3
"""
Precheck whether MetaWorld tasks support mid-episode goal switching.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def make_env(task_name: str, seed: int):
    import metaworld

    ml1 = metaworld.ML1(task_name, seed=seed)
    env = ml1.train_classes[task_name]()
    env.set_task(ml1.train_tasks[0])
    env._partially_observable = False
    env._freeze_rand_vec = False
    env._seed = seed
    return env


def try_switch_goal(env, task_name: str) -> dict[str, object]:
    obs_raw = env.reset()
    obs = obs_raw[0] if isinstance(obs_raw, tuple) else obs_raw
    before_goal = np.asarray(obs[36:39], dtype=np.float32).copy() if len(obs) >= 39 else np.zeros(3, dtype=np.float32)
    step_zero = np.zeros(4, dtype=np.float32)

    candidates = []
    if np.linalg.norm(before_goal) > 1e-6:
        candidates.append(before_goal + np.array([0.03, -0.02, 0.01], dtype=np.float32))
        candidates.append(before_goal + np.array([-0.02, 0.01, 0.0], dtype=np.float32))
    else:
        candidates.append(np.array([0.1, 0.7, 0.15], dtype=np.float32))

    methods = []
    if hasattr(env, "_target_pos"):
        methods.append("_target_pos")
    if hasattr(env, "goal"):
        methods.append("goal")

    results = {
        "task": task_name,
        "before_goal": before_goal.tolist(),
        "methods_tried": methods,
        "checks": [],
    }

    for method in methods:
        for cand in candidates:
            ok = False
            err = None
            try:
                if method == "_target_pos":
                    env._target_pos = np.asarray(cand, dtype=np.float32).copy()
                elif method == "goal":
                    env.goal = np.asarray(cand, dtype=np.float32).copy()
                step_out = env.step(step_zero)
                if len(step_out) == 5:
                    obs2, _, _, _, _ = step_out
                else:
                    obs2, _, _, _ = step_out
                after_goal = np.asarray(obs2[36:39], dtype=np.float32).copy() if len(obs2) >= 39 else np.zeros(3, dtype=np.float32)
                internal_goal = None
                if hasattr(env, "_target_pos"):
                    try:
                        internal_goal = np.asarray(env._target_pos, dtype=np.float32).copy()
                    except Exception:
                        internal_goal = None
                ok = bool(np.linalg.norm(after_goal - cand) < 1e-3 or (internal_goal is not None and np.linalg.norm(internal_goal - cand) < 1e-3))
                results["checks"].append(
                    {
                        "method": method,
                        "candidate_goal": cand.tolist(),
                        "after_goal": after_goal.tolist(),
                        "internal_goal": None if internal_goal is None else internal_goal.tolist(),
                        "ok": ok,
                    }
                )
            except Exception as e:
                err = str(e)
                results["checks"].append(
                    {
                        "method": method,
                        "candidate_goal": cand.tolist(),
                        "ok": False,
                        "error": err,
                    }
                )

    return results


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=str, default="reach-v3,push-v3,pick-place-v3")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="artifacts/multi_goal/precheck_goal_switch.json")
    args = p.parse_args()

    out = []
    for task in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        env = make_env(task, args.seed)
        try:
            out.append(try_switch_goal(env, task))
        finally:
            env.close()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

