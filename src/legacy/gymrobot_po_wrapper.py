#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class GoalMaskSpec:
    goal_key: str = "desired_goal"
    achieved_key: str = "achieved_goal"
    observation_key: str = "observation"


class GoalMaskedGoalEnv(gym.Wrapper):
    """Weak-PO wrapper for GoalEnv-style robotics tasks.

    The wrapper preserves the original dict structure and only masks
    goal-related entries between flashes. This makes it easy to test
    whether a controller truly remembers the goal instead of reading it
    every step.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        flash_every: int = 5,
        flash_len: int = 1,
        po_mode: str = "weak",
        mask_goal_on_reset: bool = False,
        spec: GoalMaskSpec | None = None,
    ) -> None:
        super().__init__(env)
        self.flash_every = int(flash_every)
        self.flash_len = int(flash_len)
        self.po_mode = str(po_mode).strip().lower()
        self.mask_goal_on_reset = bool(mask_goal_on_reset)
        self.mask_spec = spec or GoalMaskSpec()
        self._step_count = 0

        if self.po_mode not in {"none", "weak", "strong"}:
            raise ValueError(f"Unsupported po_mode: {self.po_mode}")

    def reset(self, **kwargs: Any):
        obs, info = self.env.reset(**kwargs)
        self._step_count = 0
        return self._transform(obs, is_reset=True), info

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1
        return self._transform(obs, is_reset=False), reward, terminated, truncated, info

    def _goal_visible(self, *, is_reset: bool) -> bool:
        if self.po_mode == "none":
            return True
        if is_reset:
            return not self.mask_goal_on_reset
        if self.flash_every <= 0:
            return False
        return (self._step_count % self.flash_every) < self.flash_len

    def _transform(self, obs: Any, *, is_reset: bool) -> Any:
        if not isinstance(obs, dict):
            return obs

        visible = self._goal_visible(is_reset=is_reset)
        out = {
            key: (np.asarray(value).copy() if isinstance(value, np.ndarray) else value)
            for key, value in obs.items()
        }

        if not visible and self.mask_spec.goal_key in out:
            out[self.mask_spec.goal_key] = np.zeros_like(np.asarray(out[self.mask_spec.goal_key]))

        if self.po_mode == "strong" and not visible and self.mask_spec.achieved_key in out:
            out[self.mask_spec.achieved_key] = np.zeros_like(np.asarray(out[self.mask_spec.achieved_key]))

        return out
