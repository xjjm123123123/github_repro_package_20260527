#!/usr/bin/env python3
"""
MetaWorld experiment scaffold: partial observability + (no_mem/text_buf/struct_mem) policies.

Design goals:
- No LLM calls.
- Minimal, reproducible CLI to run episodes and log metrics/events.
- Works even if MetaWorld is not installed: prints actionable install hints.

What this script provides:
- Environment creation helpers for MetaWorld tasks
- Partial-observability wrappers with goal flashing support
- Policy implementations (imported from metaworld_policies.py):
  - NoMemoryPolicy: FSM+primitives, persist_goal=False, no failure recovery
  - TextBufferPolicy: FSM+primitives with text buffer, persist_goal=True
  - StructMemoryPolicy: FSM+primitives with structured memory, persist_goal=True
- Logging: runs/<exp_id>/{metrics.jsonl,events.jsonl,summary.json,config.json}
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metaworld_policies import (
    NoMemoryPolicy,
    StructMemoryPolicy,
    TextBufferPolicy,
)


def _now_tag() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def _jsonl_append(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _try_import_metaworld() -> Any | None:
    try:
        import metaworld  # type: ignore
        return metaworld
    except Exception:
        return None


def _install_hint() -> str:
    return (
        "MetaWorld is not installed.\n"
        "Suggested install:\n"
        "  1) Create a clean venv\n"
        "  2) pip install 'metaworld @ git+https://github.com/Farama-Foundation/Metaworld.git'\n"
        "  3) pip install mujoco gymnasium numpy\n"
        "If you already have MuJoCo issues, check that MUJOCO_GL is set correctly.\n"
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np  # type: ignore
        np.random.seed(seed)
    except Exception:
        pass


def _step_env(env: Any, action: Any) -> tuple[Any, float, bool, dict[str, Any]]:
    out = env.step(action)
    if isinstance(out, tuple) and len(out) == 4:
        obs, reward, done, info = out
        return obs, float(reward), bool(done), dict(info or {})
    if isinstance(out, tuple) and len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated) or bool(truncated)
        return obs, float(reward), done, dict(info or {})
    raise RuntimeError(f"Unexpected env.step() return: {type(out)} len={len(out) if isinstance(out, tuple) else 'n/a'}")


def _reset_env(env: Any, seed: int | None) -> tuple[Any, dict[str, Any]]:
    if seed is None:
        out = env.reset()
    else:
        out = env.reset(seed=seed)
    if isinstance(out, tuple) and len(out) == 2:
        obs, info = out
        return obs, dict(info or {})
    return out, {}


class PartialObsWrapper:
    """Partial observability wrapper for MetaWorld with goal flashing.

    PO setting:
    - On reset (step 0): goal position is visible in obs[36:39]
    - Goal flashing: every `flash_every` steps, goal is visible for
      `flash_len` steps. Between flashes, goal is zeroed out.
    - If flash_every=0: goal is masked after reset (original behavior)
    - Gaussian noise (std) is added to all observations
    - Optional index masking via keep_indices

    This design forces policies to *remember* the goal across steps,
    which is the core experimental manipulation:
    - NoMemory: persist_goal=False → goal_pos returns zeros when masked → poor navigation
    - TextBuffer: persist_goal=True → goal stored in ObsParser → can navigate
    - StructMemory: persist_goal=True → goal stored in ObsParser → can navigate
    The difference between TextBuffer and StructMemory emerges in failure
    recovery: StructMemory detects grasp failures more reliably and retries
    with XY perturbation.
    """

    GOAL_INDICES = list(range(36, 39))
    OBJECT_POS_INDICES = list(range(4, 7))

    def __init__(
        self,
        env: Any,
        *,
        po_mode: str = "weak",
        memory_mode: str = "none",
        noise_std: float = 0.0,
        keep_indices: list[int] | None = None,
        mask_goal_after_reset: bool = True,
        flash_every: int = 0,
        flash_len: int = 1,
        mask_goal_on_reset: bool = False,
    ) -> None:
        self.env = env
        self.po_mode = str(po_mode or "weak").strip().lower()
        self.memory_mode = str(memory_mode or "none").strip().lower()
        self.noise_std = float(noise_std)
        self.keep_indices = keep_indices
        self.mask_goal_after_reset = mask_goal_after_reset
        self.flash_every = int(flash_every)
        self.flash_len = int(flash_len)
        self.mask_goal_on_reset = mask_goal_on_reset
        self._step_count = 0
        self._last_goal: Any = None

    def reset(self, **kwargs: Any) -> Any:
        obs, info = _reset_env(self.env, kwargs.get("seed"))
        self._step_count = 0
        obs_out = self._transform(obs, is_reset=True)
        return obs_out, info

    def step(self, action: Any) -> Any:
        obs, reward, done, info = _step_env(self.env, action)
        self._step_count += 1
        obs_out = self._transform(obs, is_reset=False)
        return obs_out, reward, done, info

    def _is_goal_visible(self, *, is_reset: bool) -> bool:
        if is_reset:
            return not self.mask_goal_on_reset
        if not self.mask_goal_after_reset:
            return True
        if self.flash_every <= 0:
            return False
        # Step count is incremented in step() before calling _transform().
        # We define flashes to occur when step_count % flash_every == 0.
        # For flash_len>1, the goal stays visible for flash_len consecutive steps
        # starting at that flash step.
        if self._step_count <= 0:
            return False
        step_in_cycle = self._step_count % self.flash_every
        return step_in_cycle < self.flash_len

    def _transform(self, obs: Any, *, is_reset: bool = False) -> Any:
        try:
            import numpy as np  # type: ignore
            x = np.asarray(obs, dtype=np.float32).copy()
        except Exception:
            return obs

        if len(x) >= 39:
            env_target = None
            try:
                env_target = np.asarray(self.env._target_pos, dtype=np.float32)
            except Exception:
                pass

            if is_reset and env_target is not None:
                self._last_goal = env_target.copy()

            if self._last_goal is None and len(x) >= 39:
                self._last_goal = x[36:39].copy()

        goal_visible = self._is_goal_visible(is_reset=is_reset)

        if len(x) >= 39:
            if goal_visible and self._last_goal is not None:
                x[36:39] = self._last_goal.copy()
            elif self.memory_mode == "goal_buffer" and self._last_goal is not None:
                x[36:39] = self._last_goal.copy()
            elif self.mask_goal_after_reset:
                x[36:39] = 0.0

        # Strong PO: additionally mask object position when goal is not visible.
        # We only apply this under the PO regime (mask_goal_after_reset=True).
        if self.po_mode == "strong" and (not goal_visible) and self.mask_goal_after_reset:
            if len(x) >= 7:
                x[4:7] = 0.0

        if self.keep_indices is not None and len(self.keep_indices) > 0:
            x2 = x[self.keep_indices]
        else:
            x2 = x

        if self.noise_std > 0:
            try:
                import numpy as np  # type: ignore
                x2 = x2 + np.random.normal(0.0, self.noise_std, size=x2.shape).astype(x2.dtype)
            except Exception:
                pass

        return x2

    @property
    def last_goal(self) -> Any:
        return self._last_goal

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)


@dataclass
class Event:
    t: float
    step: int
    action: list[float]
    reward: float
    done: bool
    success: float
    fsm_state: str = ""
    fsm_idx: int = -1
    retry_count: int = 0
    did_rollback: bool = False


@dataclass
class EpisodeMetrics:
    task: str
    seed: int
    episode: int
    steps: int
    total_reward: float
    success: float
    wall_time_sec: float
    grasp_attempts: int
    grasp_success: int
    retry_count: int


def _make_env(metaworld: Any, task_name: str) -> Any:
    """Create a MetaWorld env instance for a given task.

    Disables MetaWorld's built-in partially_observable so that goal
    appears in obs[36:39] on reset; our PartialObsWrapper handles masking.
    """
    norm = _normalize_task_name(task_name)
    ml1 = metaworld.ML1(norm)
    env = ml1.train_classes[norm]()
    task = ml1.train_tasks[0]
    if hasattr(env, "_set_task"):
        env._set_task(task)
    elif hasattr(env, "set_task"):
        env.set_task(task)
    else:
        raise RuntimeError("MetaWorld env lacks set_task/_set_task; version mismatch?")
    if hasattr(env, "_partially_observable"):
        env._partially_observable = False
    return env


def _action_dim(env: Any) -> int:
    space = getattr(env, "action_space", None)
    if space is None:
        return 4
    shape = getattr(space, "shape", None)
    if shape and len(shape) == 1:
        return int(shape[0])
    return 4


def _parse_keep_indices(s: str) -> list[int] | None:
    s = (s or "").strip()
    if not s:
        return None
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _normalize_task_name(task_name: str) -> str:
    t = (task_name or "").strip()
    if t.endswith("-v2"):
        return t[:-3] + "-v3"
    return t


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="reach-v3", help="MetaWorld task name (e.g. reach-v3 or reach-v2)")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--policy", choices=["no_mem", "text_buf", "struct_mem"], default="struct_mem")
    p.add_argument("--text-hist", type=int, default=20, help="TextBuffer history length")
    p.add_argument("--mem-k", type=int, default=5, help="StructMemory top-k retrieval")
    p.add_argument("--controller", choices=["PD", "PI"], default="PD", help="Controller type")
    p.add_argument("--po-noise-std", type=float, default=0.01)
    p.add_argument("--po-keep-indices", default="", help="Optional comma-separated indices to keep")
    p.add_argument("--po-mode", choices=["weak", "strong"], default="weak", help="PO protocol: weak masks goal only; strong also masks object position on non-flash steps")
    p.add_argument("--flash-every", type=int, default=20, help="Flash goal every N steps (0=never after reset)")
    p.add_argument("--flash-len", type=int, default=1, help="Flash goal for N steps each cycle")
    p.add_argument("--mask-goal-on-reset", action="store_true", help="Mask goal even on reset under PO (forces memory reliance)")
    p.add_argument("--max-retries", type=int, default=3, help="Max grasp retry attempts (0=disable retry)")
    p.add_argument("--no-perturbation", action="store_true", help="Disable XY perturbation on retry (ablation)")
    p.add_argument("--full-observable", action="store_true", help="Make goal always visible (full observable control)")
    p.add_argument("--run-dir", default="runs", help="Root directory for run outputs")
    p.add_argument("--fsm-version", type=int, default=1, help="FSM builder version (1=original, 2=force-aware)")
    p.add_argument("--force-aware", action="store_true", help="Shortcut for --fsm-version 2")
    p.add_argument("--multi-goal", action="store_true", help="Enable sequential goal A->B switching within an episode")
    p.add_argument("--multi-goal-flash-len", type=int, default=1, help="Visible steps for the new goal immediately after switching")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    metaworld = _try_import_metaworld()
    if metaworld is None:
        print(_install_hint())
        return 2

    _set_seed(int(args.seed))

    env = _make_env(metaworld, str(args.task))
    env = PartialObsWrapper(
        env,
        po_mode=str(args.po_mode),
        noise_std=float(args.po_noise_std),
        keep_indices=_parse_keep_indices(str(args.po_keep_indices)),
        flash_every=int(args.flash_every),
        flash_len=int(args.flash_len),
        mask_goal_after_reset=not bool(args.full_observable),
        mask_goal_on_reset=bool(args.mask_goal_on_reset) if hasattr(args, 'mask_goal_on_reset') else False,
    )
    if bool(args.multi_goal):
        from multi_goal_wrapper import MultiGoalWrapper

        env = MultiGoalWrapper(
            env,
            task_name=str(args.task),
            flash_len=int(args.multi_goal_flash_len),
        )

    adim = _action_dim(env)
    ctrl = str(args.controller)
    max_ret = int(args.max_retries)
    no_pert = bool(args.no_perturbation)
    fsm_ver = 2 if bool(args.force_aware) else int(args.fsm_version)
    if args.policy == "no_mem":
        policy: NoMemoryPolicy | TextBufferPolicy | StructMemoryPolicy = NoMemoryPolicy(adim, controller_type=ctrl, max_retries=max_ret, fsm_version=fsm_ver)
    elif args.policy == "text_buf":
        policy = TextBufferPolicy(adim, history=int(args.text_hist), controller_type=ctrl, max_retries=max_ret, fsm_version=fsm_ver)
    else:
        policy = StructMemoryPolicy(adim, top_k=int(args.mem_k), controller_type=ctrl, max_retries=max_ret, enable_perturbation=not no_pert, fsm_version=fsm_ver)

    exp_id = f"mw-{args.task}-{args.policy}-{_now_tag()}-seed{args.seed}"
    run_dir = Path(args.run_dir) / exp_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    metrics_path = run_dir / "metrics.jsonl"
    events_path = run_dir / "events.jsonl"

    episode_results: list[EpisodeMetrics] = []
    t_start = time.time()

    for ep in range(int(args.episodes)):
        ep_seed = int(args.seed) + ep
        obs, _ = env.reset(seed=ep_seed)
        policy.reset(str(args.task))
        policy.set_env_ref(env)

        total_reward = 0.0
        success = 0.0
        ep_t0 = time.time()

        prev_fsm_idx = -1
        prev_retry_count = 0

        for step in range(int(args.max_steps)):
            action = policy.act(obs)
            obs2, reward, done, info = env.step(action)
            total_reward += float(reward)
            success = max(success, float(info.get("success", 0.0) or 0.0))

            policy.observe(obs2, action, reward, info)
            if isinstance(policy, StructMemoryPolicy):
                policy.update(obs=obs2, action=action, reward=reward, info=info, step=step)

            fsm_state_name = ""
            fsm_idx_now = -1
            if hasattr(policy, "fsm") and policy.fsm is not None:
                fsm_state_name = policy.fsm.current_state_name or ""
                fsm_idx_now = policy.fsm.current_idx

            cur_retry = int(policy.retry_count)
            did_rollback = fsm_idx_now < prev_fsm_idx and cur_retry > prev_retry_count

            act_list: list[float]
            try:
                act_list = [float(x) for x in (action.tolist() if hasattr(action, "tolist") else action)]
            except Exception:
                act_list = []
            _jsonl_append(events_path, asdict(Event(
                t=time.time(),
                step=step,
                action=act_list,
                reward=float(reward),
                done=bool(done),
                success=float(info.get("success", 0.0) or 0.0),
                fsm_state=fsm_state_name,
                fsm_idx=fsm_idx_now,
                retry_count=cur_retry,
                did_rollback=did_rollback,
            )))

            prev_fsm_idx = fsm_idx_now
            prev_retry_count = cur_retry

            obs = obs2
            if done:
                break

        em = EpisodeMetrics(
            task=str(args.task),
            seed=ep_seed,
            episode=ep,
            steps=step + 1,
            total_reward=float(total_reward),
            success=float(success),
            wall_time_sec=float(time.time() - ep_t0),
            grasp_attempts=int(policy.grasp_attempts),
            grasp_success=int(policy.grasp_success),
            retry_count=int(policy.retry_count),
        )
        episode_results.append(em)
        _jsonl_append(metrics_path, asdict(em))
        print(f"[{ep+1}/{args.episodes}] success={em.success:.0f} steps={em.steps} reward={em.total_reward:.2f} grasp_att={em.grasp_attempts} grasp_ok={em.grasp_success} retry={em.retry_count}")

    succ_rate = sum(1 for e in episode_results if e.success >= 1.0) / max(1, len(episode_results))
    avg_grasp_attempts = sum(e.grasp_attempts for e in episode_results) / max(1, len(episode_results))
    avg_grasp_success = sum(e.grasp_success for e in episode_results) / max(1, len(episode_results))
    avg_retry = sum(e.retry_count for e in episode_results) / max(1, len(episode_results))
    summary = {
        "exp_id": exp_id,
        "task": str(args.task),
        "policy": str(args.policy),
        "controller": str(args.controller),
        "episodes": int(args.episodes),
        "success_rate": succ_rate,
        "avg_steps": sum(e.steps for e in episode_results) / max(1, len(episode_results)),
        "avg_reward": sum(e.total_reward for e in episode_results) / max(1, len(episode_results)),
        "avg_grasp_attempts": avg_grasp_attempts,
        "avg_grasp_success": avg_grasp_success,
        "avg_retry_count": avg_retry,
        "flash_every": int(args.flash_every),
        "flash_len": int(args.flash_len),
        "wall_time_sec": float(time.time() - t_start),
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary: {run_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
