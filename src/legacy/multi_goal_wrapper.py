#!/usr/bin/env python3
"""
Sequential multi-goal wrapper for MetaWorld tasks.

Stage 1: solve original goal A.
Stage 2: switch to goal B mid-episode and flash the new goal once.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class MultiGoalWrapper:
    def __init__(self, env: Any, *, task_name: str, flash_len: int = 1) -> None:
        self.env = env
        self.task_name = task_name
        self.flash_len = int(flash_len)
        self._goal_a: np.ndarray | None = None
        self._goal_b: np.ndarray | None = None
        self._goal_stage = 1
        self._switched = False
        self._flash_remaining = 0

    def _make_goal_b(self, goal_a: np.ndarray) -> np.ndarray:
        offsets = {
            "reach-v3": np.array([0.08, -0.04, 0.0], dtype=np.float32),
            "push-v3": np.array([-0.06, 0.05, 0.0], dtype=np.float32),
            "pick-place-v3": np.array([0.05, -0.04, 0.02], dtype=np.float32),
        }
        offset = offsets.get(self.task_name, np.array([0.06, -0.03, 0.0], dtype=np.float32))
        goal_b = goal_a.astype(np.float32).copy() + offset
        goal_b[2] = max(0.02, float(goal_b[2]))
        return goal_b

    def _set_goal(self, goal: np.ndarray) -> None:
        if hasattr(self.env, "_target_pos"):
            self.env._target_pos = goal.astype(np.float32).copy()
        if hasattr(self.env, "_last_goal"):
            self.env._last_goal = goal.astype(np.float32).copy()

    def _inject_flash(self, obs: Any) -> Any:
        try:
            x = np.asarray(obs, dtype=np.float32).copy()
        except Exception:
            return obs
        if self._flash_remaining > 0 and self._goal_b is not None and len(x) >= 39:
            x[36:39] = self._goal_b.copy()
            self._flash_remaining -= 1
        return x

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        out = self.env.reset(*args, **kwargs)
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs, info = out, {}

        goal_a = None
        if hasattr(self.env, "last_goal") and self.env.last_goal is not None:
            goal_a = np.asarray(self.env.last_goal, dtype=np.float32).copy()
        elif hasattr(self.env, "_target_pos"):
            goal_a = np.asarray(self.env._target_pos, dtype=np.float32).copy()
        else:
            goal_a = np.asarray(obs[36:39], dtype=np.float32).copy()

        self._goal_a = goal_a
        self._goal_b = self._make_goal_b(goal_a)
        self._goal_stage = 1
        self._switched = False
        self._flash_remaining = 0
        info = dict(info or {})
        info["goal_stage"] = self._goal_stage
        return obs, info

    def step(self, action: Any) -> Any:
        obs, reward, done, info = self.env.step(action)
        info = dict(info or {})

        if not self._switched and float(info.get("success", 0.0) or 0.0) > 0.0 and self._goal_a is not None and self._goal_b is not None:
            prev_goal = self._goal_a.copy()
            new_goal = self._goal_b.copy()
            self._set_goal(new_goal)
            self._switched = True
            self._goal_stage = 2
            self._flash_remaining = self.flash_len
            info["goal_switched"] = True
            info["prev_goal"] = prev_goal.tolist()
            info["new_goal"] = new_goal.tolist()
            info["success"] = 0.0
            done = False

        obs = self._inject_flash(obs)
        info["goal_stage"] = self._goal_stage
        return obs, reward, done, info

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

