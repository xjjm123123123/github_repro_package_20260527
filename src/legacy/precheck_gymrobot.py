#!/usr/bin/env python3
"""Smoke-test Gymnasium-Robotics tasks under FO and WeakPO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
import numpy as np

from gymrobot_po_wrapper import GoalMaskedGoalEnv


gym.register_envs(gymnasium_robotics)


def make_env(task: str, po_mode: str, flash_every: int, flash_len: int):
    env = gym.make(task)
    if po_mode != "none":
        env = GoalMaskedGoalEnv(
            env,
            po_mode=po_mode,
            flash_every=flash_every,
            flash_len=flash_len,
            mask_goal_on_reset=False,
        )
    return env


def _goal_from_obs(obs: dict[str, np.ndarray], cached_goal: np.ndarray | None) -> np.ndarray:
    goal = np.asarray(obs["desired_goal"], dtype=np.float32)
    if np.linalg.norm(goal) > 1e-8:
        return goal.copy()
    if cached_goal is not None:
        return cached_goal.copy()
    return goal.copy()


def policy_no_memory(task: str, obs: dict[str, np.ndarray]) -> np.ndarray:
    goal = np.asarray(obs["desired_goal"], dtype=np.float32)
    action = np.zeros(4, dtype=np.float32)

    if task.startswith("FetchReach"):
        current = np.asarray(obs["achieved_goal"], dtype=np.float32)
        action[:3] = np.clip(10.0 * (goal - current), -1.0, 1.0)
        return action

    if task.startswith("FetchPush"):
        return push_controller(obs, goal)

    return action


def policy_text_buffer(task: str, obs: dict[str, np.ndarray], cached_goal: np.ndarray | None):
    goal = _goal_from_obs(obs, cached_goal)
    action = np.zeros(4, dtype=np.float32)

    if task.startswith("FetchReach"):
        current = np.asarray(obs["achieved_goal"], dtype=np.float32)
        action[:3] = np.clip(10.0 * (goal - current), -1.0, 1.0)
        return action, goal

    if task.startswith("FetchPush"):
        return push_controller(obs, goal), goal

    return action, goal


def push_controller(obs: dict[str, np.ndarray], goal: np.ndarray) -> np.ndarray:
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
    action[3] = 0.0
    return action


def run_episodes(task: str, controller: str, po_mode: str, episodes: int, flash_every: int, flash_len: int) -> dict[str, object]:
    env = make_env(task, po_mode, flash_every, flash_len)
    successes = []
    rewards = []

    try:
        for ep in range(episodes):
            obs, info = env.reset(seed=ep)
            cached_goal = None
            ep_reward = 0.0
            ep_success = 0.0

            for _ in range(getattr(env, "_max_episode_steps", 50)):
                if controller == "no_memory":
                    action = policy_no_memory(task, obs)
                else:
                    action, cached_goal = policy_text_buffer(task, obs, cached_goal)

                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += float(reward)
                ep_success = max(ep_success, float(info.get("is_success", 0.0) or 0.0))
                if terminated or truncated:
                    break

            successes.append(ep_success)
            rewards.append(ep_reward)
    finally:
        env.close()

    return {
        "task": task,
        "controller": controller,
        "condition": "FO" if po_mode == "none" else po_mode.upper(),
        "episodes": episodes,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=str, default="FetchReach-v4,FetchPush-v4")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--flash-every", type=int, default=5)
    parser.add_argument("--flash-len", type=int, default=1)
    parser.add_argument("--out", type=str, default="artifacts/gymrobot/precheck_gymrobot.json")
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    results = []
    for task in tasks:
        for controller in ["no_memory", "text_buffer"]:
            results.append(run_episodes(task, controller, "none", args.episodes, args.flash_every, args.flash_len))
            results.append(run_episodes(task, controller, "weak", args.episodes, args.flash_every, args.flash_len))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
