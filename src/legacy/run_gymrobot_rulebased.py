#!/usr/bin/env python3
"""Rule-based Gymnasium-Robotics batch evaluation under PO.

Runs NM / TB / SM x FO / WeakPO / StrongPO x 3 seeds x 60 episodes
for FetchReach-v4, FetchPush-v4, FetchPickAndPlace-v4.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gymrobot_po_wrapper import GoalMaskedGoalEnv

gym.register_envs(gymnasium_robotics)

TASKS = ["FetchReach-v4", "FetchPush-v4", "FetchPickAndPlace-v4"]
CONTROLLERS = ["no_memory", "text_buffer", "struct_memory"]
CONDITIONS = [("FO", "none"), ("WeakPO", "weak"), ("StrongPO", "strong")]
N_SEEDS = 3
N_EPS = 60
MAX_STEPS = 50


def make_env(task, po_mode, flash_every, flash_len):
    env = gym.make(task)
    if po_mode != "none":
        env = GoalMaskedGoalEnv(
            env, po_mode=po_mode, flash_every=flash_every, flash_len=flash_len, mask_goal_on_reset=False
        )
    return env


def _goal_from_obs(obs, cached_goal):
    goal = np.asarray(obs["desired_goal"], dtype=np.float32)
    if np.linalg.norm(goal) > 1e-8:
        return goal.copy()
    if cached_goal is not None:
        return cached_goal.copy()
    return goal.copy()


def reach_controller(grip, target):
    action = np.zeros(4, dtype=np.float32)
    action[:3] = np.clip(10.0 * (target - grip), -1.0, 1.0)
    return action


def push_controller(obs, goal):
    action = np.zeros(4, dtype=np.float32)
    grip = np.asarray(obs["observation"][:3], dtype=np.float32)
    obj = np.asarray(obs["achieved_goal"], dtype=np.float32)
    delta_xy = goal[:2] - obj[:2]
    norm = np.linalg.norm(delta_xy)
    push_dir = delta_xy / (norm + 1e-8)
    behind = obj.copy()
    behind[:2] -= 0.06 * push_dir
    behind[2] = obj[2] + 0.02
    if np.linalg.norm(grip[:2] - behind[:2]) > 0.03 or abs(grip[2] - behind[2]) > 0.02:
        target = behind
    else:
        target = np.array([obj[0] + 0.04 * push_dir[0], obj[1] + 0.04 * push_dir[1], obj[2]], dtype=np.float32)
    action[:3] = np.clip(8.0 * (target - grip), -1.0, 1.0)
    return action


def pick_place_controller(obs, goal, phase, grip_open):
    action = np.zeros(4, dtype=np.float32)
    grip = np.asarray(obs["observation"][:3], dtype=np.float32)
    obj = np.asarray(obs["achieved_goal"], dtype=np.float32)
    grip_state = float(obs["observation"][3])

    if phase == "approach":
        target = obj.copy()
        target[2] += 0.04
        action[:3] = np.clip(8.0 * (target - grip), -1.0, 1.0)
        action[3] = 1.0
        if np.linalg.norm(grip[:2] - obj[:2]) < 0.04 and abs(grip[2] - (obj[2] + 0.04)) < 0.03:
            phase = "descend"

    elif phase == "descend":
        target = obj.copy()
        action[:3] = np.clip(8.0 * (target - grip), -1.0, 1.0)
        action[3] = 1.0
        if np.linalg.norm(grip - obj) < 0.04:
            phase = "grasp"

    elif phase == "grasp":
        action[:3] = np.clip(4.0 * (obj - grip), -1.0, 1.0)
        action[3] = -1.0
        phase = "lift"

    elif phase == "lift":
        target = obj.copy()
        target[2] += 0.1
        action[:3] = np.clip(8.0 * (target - grip), -1.0, 1.0)
        action[3] = -1.0
        if grip[2] > obj[2] + 0.05:
            phase = "move"

    elif phase == "move":
        target = goal.copy()
        target[2] += 0.05
        action[:3] = np.clip(6.0 * (target - grip), -1.0, 1.0)
        action[3] = -1.0
        if np.linalg.norm(grip[:2] - goal[:2]) < 0.04:
            phase = "release"

    elif phase == "release":
        action[:3] = 0.0
        action[3] = 1.0
        phase = "done"

    return action, phase


def get_action(task, obs, goal, phase):
    grip = np.asarray(obs["observation"][:3], dtype=np.float32)
    if task.startswith("FetchReach"):
        return reach_controller(grip, goal), phase
    elif task.startswith("FetchPush"):
        return push_controller(obs, goal), phase
    elif task.startswith("FetchPickAndPlace"):
        return pick_place_controller(obs, goal, phase, None)
    return np.zeros(4, dtype=np.float32), phase


def run_batch(flash_every, flash_len, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    start = time.time()

    for task in TASKS:
        for cond_tag, po_mode in CONDITIONS:
            for controller in CONTROLLERS:
                for seed in range(N_SEEDS):
                    tag = f"{task}__{controller}__{cond_tag}__s{seed}"
                    out_file = out_dir / f"{tag}.json"
                    if out_file.exists():
                        print(f"[SKIP] {tag}")
                        continue

                    env = make_env(task, po_mode, flash_every, flash_len)
                    successes = []
                    rewards = []

                    for ep in range(N_EPS):
                        obs, info = env.reset(seed=seed * 10000 + ep)
                        cached_goal = None
                        struct_goal = None
                        phase = "approach" if "Pick" in task else None
                        ep_reward = 0.0
                        ep_success = 0.0

                        for step in range(MAX_STEPS):
                            if controller == "no_memory":
                                goal = np.asarray(obs["desired_goal"], dtype=np.float32)
                            elif controller == "text_buffer":
                                goal = _goal_from_obs(obs, cached_goal)
                                if np.linalg.norm(np.asarray(obs["desired_goal"], dtype=np.float32)) > 1e-8:
                                    cached_goal = goal.copy()
                            elif controller == "struct_memory":
                                goal = _goal_from_obs(obs, struct_goal)
                                if np.linalg.norm(np.asarray(obs["desired_goal"], dtype=np.float32)) > 1e-8:
                                    struct_goal = goal.copy()

                            action, phase = get_action(task, obs, goal, phase)
                            obs, reward, terminated, truncated, info = env.step(action)
                            ep_reward += float(reward)
                            ep_success = max(ep_success, float(info.get("is_success", 0.0) or 0.0))
                            if terminated or truncated:
                                break

                        successes.append(ep_success)
                        rewards.append(ep_reward)

                    env.close()
                    result = {
                        "task": task,
                        "controller": controller,
                        "condition": cond_tag,
                        "po_mode": po_mode,
                        "seed": seed,
                        "episodes": N_EPS,
                        "success_rate": float(np.mean(successes)),
                        "mean_reward": float(np.mean(rewards)),
                    }
                    all_results.append(result)
                    out_file.write_text(json.dumps(result, indent=2))
                    print(f"[{cond_tag}] {task} {controller} s{seed}: success={result['success_rate']:.3f}")

    elapsed = time.time() - start
    summary_file = out_dir / "summary.json"
    summary_file.write_text(json.dumps(all_results, indent=2))
    print(f"\nDone in {elapsed:.0f}s. {len(all_results)} results written to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flash-every", type=int, default=5)
    parser.add_argument("--flash-len", type=int, default=1)
    parser.add_argument("--out-dir", type=str, default="artifacts/gymrobot_rulebased")
    args = parser.parse_args()
    run_batch(args.flash_every, args.flash_len, args.out_dir)
