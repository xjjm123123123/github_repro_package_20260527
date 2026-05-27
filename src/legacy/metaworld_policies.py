#!/usr/bin/env python3
"""
MetaWorld policy implementations: PD/PI controller, primitives, FSM, and memory-augmented policies.

No LLM calls — everything is rule-based.

Observation layout (standard 39-dim):
  [0:3]   end-effector XYZ
  [3]     gripper open/close state (1.0=open, <0.6=closed)
  [4:7]   object1 XYZ position
  [7:11]  object1 quaternion
  [11:14] object2 XYZ position
  [14:18] object2 quaternion
  [18:21] prev end-effector XYZ
  [21:24] prev object1 XYZ
  [24:36] additional prev info
  [36:39] goal position (zeroed in partially_observable mode)

Gripper convention (MetaWorld Sawyer):
  obs[3] = 1.0 → fully open
  obs[3] < 0.6 → closed
  action[3] > 0 → close gripper
  action[3] < 0 → open gripper
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np


KP = 12.0
KD = 0.2
PI_KP = 25.0
PI_KI = 2.0
GRIPPER_CLOSE_THRESHOLD = 0.6
DIST_THRESHOLD = 0.02
REACH_THRESHOLD = 0.04
ACTION_DIM = 4
GRASP_HOLD_STEPS = 20
RELEASE_HOLD_STEPS = 10
APPROACH_HEIGHT = 0.06
DESCEND_SPEED_SCALE = 0.3
MAX_GRASP_RETRIES = 3
GRASP_SUCCESS_EE_DIST = 0.025
GRASP_SUCCESS_LIFT_Z = 0.01
PERTURB_XY_RANGE = 0.02
FORCE_GRASP_HOLD_STEPS = 20
IMPEDANCE_PULL_STEPS = 150
FORCE_PRESS_STEPS = 100
ARC_TURN_STEPS = 80


class Controller(Protocol):
    def compute(
        self,
        target_pos: np.ndarray,
        ee_pos: np.ndarray,
        ee_vel: np.ndarray | None = None,
    ) -> np.ndarray: ...


class ObsParser:
    """Safely extract fields from a MetaWorld observation vector.

    When persist_goal=True (default), the goal position is stored on first
    sight and returned even when the current observation has it zeroed out.
    When persist_goal=False (NoMemory), goal_pos returns zeros whenever
    the current observation does not contain a visible goal.
    """

    def __init__(
        self, obs: np.ndarray | list[float] | None = None, persist_goal: bool = True
    ) -> None:
        self._obs = (
            np.asarray(obs, dtype=np.float32)
            if obs is not None
            else np.zeros(0, dtype=np.float32)
        )
        self._goal_stored: np.ndarray | None = None
        self._persist_goal = persist_goal
        self._goal_updates_locked = False

    def update(self, obs: np.ndarray | list[float]) -> None:
        self._obs = np.asarray(obs, dtype=np.float32)
        g = self._extract_goal()
        if g is not None and np.linalg.norm(g) > 0.05 and not self._goal_updates_locked:
            self._goal_stored = g.copy()

    def set_goal(self, goal: np.ndarray) -> None:
        if np.linalg.norm(goal) > 1e-6:
            self._goal_stored = np.asarray(goal, dtype=np.float32).copy()

    def lock_goal_updates(self, goal: np.ndarray | None = None) -> None:
        if goal is not None and np.linalg.norm(goal) > 1e-6:
            self._goal_stored = np.asarray(goal, dtype=np.float32).copy()
        self._goal_updates_locked = True

    def unlock_goal_updates(self) -> None:
        self._goal_updates_locked = False

    def _extract_goal(self) -> np.ndarray | None:
        if len(self._obs) >= 39:
            return self._obs[36:39].copy()
        return None

    @property
    def ee_pos(self) -> np.ndarray:
        if len(self._obs) >= 3:
            return self._obs[0:3].copy()
        return np.zeros(3, dtype=np.float32)

    @property
    def gripper(self) -> float:
        if len(self._obs) >= 4:
            return float(self._obs[3])
        return 0.0

    @property
    def obj1_pos(self) -> np.ndarray:
        if len(self._obs) >= 7:
            return self._obs[4:7].copy()
        return np.zeros(3, dtype=np.float32)

    @property
    def obj1_quat(self) -> np.ndarray:
        if len(self._obs) >= 11:
            return self._obs[7:11].copy()
        return np.zeros(4, dtype=np.float32)

    @property
    def obj2_pos(self) -> np.ndarray:
        if len(self._obs) >= 14:
            return self._obs[11:14].copy()
        return np.zeros(3, dtype=np.float32)

    @property
    def prev_ee_pos(self) -> np.ndarray:
        if len(self._obs) >= 21:
            return self._obs[18:21].copy()
        return self.ee_pos.copy()

    @property
    def prev_obj1_pos(self) -> np.ndarray:
        if len(self._obs) >= 24:
            return self._obs[21:24].copy()
        return self.obj1_pos.copy()

    @property
    def goal_pos(self) -> np.ndarray:
        g = self._extract_goal()
        if g is not None and np.linalg.norm(g) > 0.05:
            return g
        if self._persist_goal and self._goal_stored is not None:
            return self._goal_stored.copy()
        return np.zeros(3, dtype=np.float32)

    @property
    def ee_velocity(self) -> np.ndarray:
        return self.ee_pos - self.prev_ee_pos

    @property
    def raw(self) -> np.ndarray:
        return self._obs

    @property
    def has_obj1(self) -> bool:
        return len(self._obs) >= 7 and np.linalg.norm(self._obs[4:7]) > 1e-6

    @property
    def has_goal(self) -> bool:
        g = self.goal_pos
        return np.linalg.norm(g) > 0.05

    @property
    def gripper_is_closed(self) -> bool:
        return self.gripper < GRIPPER_CLOSE_THRESHOLD

    @property
    def gripper_is_open(self) -> bool:
        return self.gripper >= GRIPPER_CLOSE_THRESHOLD

    def ee_obj_distance(self) -> float:
        return float(np.linalg.norm(self.ee_pos - self.obj1_pos))

    def obj_grasped(self) -> bool:
        return self.gripper_is_closed and self.ee_obj_distance() < 0.06


class CartesianPDController:
    def __init__(self, kp: float = KP, kd: float = KD) -> None:
        self.kp = kp
        self.kd = kd

    def compute(
        self,
        target_pos: np.ndarray,
        ee_pos: np.ndarray,
        ee_vel: np.ndarray | None = None,
    ) -> np.ndarray:
        error = target_pos - ee_pos
        delta = self.kp * error
        if ee_vel is not None:
            delta -= self.kd * ee_vel
        delta = np.clip(delta, -1.0, 1.0)
        return delta.astype(np.float32)


class CartesianPIController:
    def __init__(self, kp: float = PI_KP, ki: float = PI_KI) -> None:
        self.kp = kp
        self.ki = ki
        self._integral = np.zeros(3, dtype=np.float32)
        self._last_target: np.ndarray | None = None

    def compute(
        self,
        target_pos: np.ndarray,
        ee_pos: np.ndarray,
        ee_vel: np.ndarray | None = None,
    ) -> np.ndarray:
        error = target_pos - ee_pos
        if (
            self._last_target is not None
            and np.linalg.norm(target_pos - self._last_target) > 0.05
        ):
            self._integral = np.zeros(3, dtype=np.float32)
        self._integral += error * 0.01
        self._integral = np.clip(self._integral, -0.5, 0.5)
        delta = self.kp * error + self.ki * self._integral
        delta = np.clip(delta, -1.0, 1.0)
        self._last_target = target_pos.copy()
        return delta.astype(np.float32)

    def reset(self) -> None:
        self._integral = np.zeros(3, dtype=np.float32)
        self._last_target = None


def _make_controller(controller_type: str = "PD") -> Controller:
    if controller_type == "PI":
        return CartesianPIController()
    return CartesianPDController()


def reach(
    target_pos: np.ndarray,
    obs: ObsParser,
    controller: Controller,
) -> np.ndarray:
    delta = controller.compute(target_pos, obs.ee_pos, obs.ee_velocity)
    return np.array(
        [delta[0], delta[1], delta[2], -1.0], dtype=np.float32
    )


def approach_above(
    target_pos: np.ndarray,
    obs: ObsParser,
    controller: Controller,
    height: float = APPROACH_HEIGHT,
) -> np.ndarray:
    above = target_pos.copy()
    above[2] += height
    delta = controller.compute(above, obs.ee_pos, obs.ee_velocity)
    return np.array(
        [delta[0], delta[1], delta[2], -1.0], dtype=np.float32
    )


def descend_slow(
    target_pos: np.ndarray,
    obs: ObsParser,
    controller: Controller,
    speed_scale: float = DESCEND_SPEED_SCALE,
) -> np.ndarray:
    delta = controller.compute(target_pos, obs.ee_pos, obs.ee_velocity)
    delta[:3] *= speed_scale
    delta[2] *= 0.5
    delta = np.clip(delta, -1.0, 1.0)
    return np.array(
        [delta[0], delta[1], delta[2], -1.0], dtype=np.float32
    )


def grasp(obs: ObsParser) -> np.ndarray:
    return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def grasp_hold(obs: ObsParser) -> np.ndarray:
    return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def release(obs: ObsParser) -> np.ndarray:
    return np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)


def move_to(
    target_pos: np.ndarray,
    obs: ObsParser,
    controller: Controller,
) -> np.ndarray:
    delta = controller.compute(target_pos, obs.ee_pos, obs.ee_velocity)
    return np.array(
        [delta[0], delta[1], delta[2], 1.0], dtype=np.float32
    )


def lift(
    height: float,
    obs: ObsParser,
    controller: Controller,
) -> np.ndarray:
    target = obs.ee_pos.copy()
    target[2] += height
    delta = controller.compute(target, obs.ee_pos, obs.ee_velocity)
    return np.array(
        [delta[0], delta[1], delta[2], 1.0], dtype=np.float32
    )


class AfterStepsTransition:
    def __init__(self, n: int) -> None:
        self.n = n
        self.count = 0

    def __call__(self, obs: ObsParser) -> bool:
        self.count += 1
        return self.count >= self.n

    def reset(self) -> None:
        self.count = 0


@dataclass
class FSMState:
    name: str
    action_fn: Callable[..., np.ndarray]
    transition_fn: Callable[[ObsParser], bool]


class FSM:
    """Finite state machine that transitions through subgoals."""

    def __init__(self, states: list[FSMState]) -> None:
        self.states = states
        self.current_idx = 0
        self.completed = False
        self._step_count = 0

    def reset(self) -> None:
        self.current_idx = 0
        self.completed = False
        self._step_count = 0
        for s in self.states:
            if isinstance(s.transition_fn, AfterStepsTransition):
                s.transition_fn.reset()

    @property
    def current_state(self) -> FSMState | None:
        if self.current_idx < len(self.states):
            return self.states[self.current_idx]
        return None

    @property
    def current_state_name(self) -> str:
        s = self.current_state
        return s.name if s else "done"

    @property
    def remaining_states(self) -> list[str]:
        return [s.name for s in self.states[self.current_idx :]]

    def step(self, obs: ObsParser) -> np.ndarray:
        self._step_count += 1
        state = self.current_state
        if state is None:
            self.completed = True
            return np.zeros(ACTION_DIM, dtype=np.float32)

        action = state.action_fn(obs)

        if state.transition_fn(obs):
            self.current_idx += 1
            if self.current_idx >= len(self.states):
                self.completed = True

        return action

    def retry_current(self) -> None:
        self.current_idx = max(0, self.current_idx)
        state = self.current_state
        if state is not None and isinstance(state.transition_fn, AfterStepsTransition):
            state.transition_fn.reset()

    def go_back_to(self, state_name: str) -> bool:
        for i, s in enumerate(self.states):
            if s.name == state_name:
                self.current_idx = i
                self.completed = False
                if isinstance(s.transition_fn, AfterStepsTransition):
                    s.transition_fn.reset()
                for j in range(i + 1, len(self.states)):
                    if isinstance(self.states[j].transition_fn, AfterStepsTransition):
                        self.states[j].transition_fn.reset()
                return True
        return False


def distance_reached(
    target: np.ndarray, threshold: float = DIST_THRESHOLD
) -> Callable[[ObsParser], bool]:
    def check(obs: ObsParser) -> bool:
        return bool(np.linalg.norm(obs.ee_pos - target) < threshold)
    return check


def gripper_closed(obs: ObsParser) -> bool:
    return obs.gripper < GRIPPER_CLOSE_THRESHOLD


def gripper_opened(obs: ObsParser) -> bool:
    return obs.gripper >= GRIPPER_CLOSE_THRESHOLD


def always_true(obs: ObsParser) -> bool:
    return True


def after_steps(n: int) -> AfterStepsTransition:
    return AfterStepsTransition(n)


def obj_at_target(
    target: np.ndarray, threshold: float = 0.05
) -> Callable[[ObsParser], bool]:
    def check(obs: ObsParser) -> bool:
        return bool(np.linalg.norm(obs.obj1_pos - target) < threshold)
    return check


def _goal_reached(threshold: float = REACH_THRESHOLD) -> Callable[[ObsParser], bool]:
    def check(obs: ObsParser) -> bool:
        if not obs.has_goal:
            return False
        return bool(np.linalg.norm(obs.ee_pos - obs.goal_pos) < threshold)
    return check


def _obj_at_goal(threshold: float = 0.05) -> Callable[[ObsParser], bool]:
    def check(obs: ObsParser) -> bool:
        if not obs.has_goal:
            return False
        return bool(np.linalg.norm(obs.obj1_pos - obs.goal_pos) < threshold)
    return check


def _push_action(obs: ObsParser, controller: Controller) -> np.ndarray:
    obj_pos = obs.obj1_pos
    goal = obs.goal_pos
    if not obs.has_goal:
        return np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)
    push_dir = goal - obj_pos
    push_dir[2] = 0.0
    norm = np.linalg.norm(push_dir[:2])
    if norm > 1e-6:
        push_dir[:2] = push_dir[:2] / norm * 0.2
    push_target = obj_pos + push_dir
    delta = controller.compute(push_target, obs.ee_pos, obs.ee_velocity)
    return np.array([delta[0], delta[1], delta[2], -1.0], dtype=np.float32)


def _build_reach_fsm(
    obs: ObsParser, controller: Controller, goal: np.ndarray
) -> FSM:
    return FSM([
        FSMState(
            "reach",
            lambda o: reach(o.goal_pos, o, controller) if o.has_goal else np.zeros(ACTION_DIM, dtype=np.float32),
            _goal_reached(REACH_THRESHOLD),
        ),
    ])


def _build_push_fsm(
    obs: ObsParser, controller: Controller, goal: np.ndarray
) -> FSM:
    obj_pos = obs.obj1_pos.copy()
    above_obj = obj_pos.copy()
    above_obj[2] += APPROACH_HEIGHT
    obj_level = obj_pos.copy()

    return FSM([
        FSMState(
            "reach_above_obj",
            lambda o: reach(above_obj, o, controller),
            distance_reached(above_obj, REACH_THRESHOLD),
        ),
        FSMState(
            "reach_obj_level",
            lambda o: reach(obj_level, o, controller),
            distance_reached(obj_level, REACH_THRESHOLD),
        ),
        FSMState(
            "push_to_goal",
            lambda o: _push_action(o, controller),
            _obj_at_goal(0.05),
        ),
    ])


def _build_pick_place_fsm(
    obs: ObsParser, controller: Controller, goal: np.ndarray
) -> FSM:
    obj_pos = obs.obj1_pos.copy()
    above_obj = obj_pos.copy()
    above_obj[2] += APPROACH_HEIGHT
    lift_target = obj_pos.copy()
    lift_target[2] += 0.15

    return FSM([
        FSMState(
            "approach_above_obj",
            lambda o: approach_above(obj_pos, o, controller),
            distance_reached(above_obj, REACH_THRESHOLD),
        ),
        FSMState(
            "descend_slow",
            lambda o: descend_slow(obj_pos, o, controller),
            distance_reached(obj_pos, 0.03),
        ),
        FSMState(
            "grasp_hold",
            lambda o: grasp_hold(o),
            after_steps(GRASP_HOLD_STEPS),
        ),
        FSMState(
            "lift",
            lambda o: lift(0.15, o, controller),
            distance_reached(lift_target, REACH_THRESHOLD),
        ),
        FSMState(
            "move_to_goal",
            lambda o: move_to(
                np.array([o.goal_pos[0], o.goal_pos[1], lift_target[2]], dtype=np.float32) if o.has_goal else lift_target,
                o, controller,
            ),
            lambda o: o.has_goal and bool(np.linalg.norm(o.ee_pos - np.array([o.goal_pos[0], o.goal_pos[1], lift_target[2]])) < 0.06),
        ),
        FSMState(
            "release",
            lambda o: release(o),
            after_steps(RELEASE_HOLD_STEPS),
        ),
    ])


def _build_drawer_open_fsm(
    obs: ObsParser, controller: Controller, goal: np.ndarray
) -> FSM:
    handle_pos = obs.obj1_pos.copy()
    above_handle = handle_pos.copy()
    above_handle[2] += APPROACH_HEIGHT
    pull_target = handle_pos.copy()
    pull_target[0] -= 0.15

    return FSM([
        FSMState(
            "approach_above_handle",
            lambda o: approach_above(handle_pos, o, controller),
            distance_reached(above_handle, REACH_THRESHOLD),
        ),
        FSMState(
            "descend_slow",
            lambda o: descend_slow(handle_pos, o, controller),
            distance_reached(handle_pos, 0.03),
        ),
        FSMState(
            "grasp_hold",
            lambda o: grasp_hold(o),
            after_steps(GRASP_HOLD_STEPS),
        ),
        FSMState(
            "pull_back",
            lambda o: move_to(pull_target, o, controller),
            distance_reached(pull_target, 0.06),
        ),
    ])


def _build_sweep_fsm(
    obs: ObsParser, controller: Controller, goal: np.ndarray
) -> FSM:
    obj_pos = obs.obj1_pos.copy()
    sweep_dir = goal - obj_pos if np.linalg.norm(goal) > 1e-6 else np.array([0.1, 0.0, 0.0])
    sweep_dir[2] = 0.0
    norm = np.linalg.norm(sweep_dir[:2])
    if norm > 1e-6:
        unit = sweep_dir[:2] / norm
    else:
        unit = np.array([1.0, 0.0])
    side_pos = obj_pos.copy()
    side_pos[:2] -= unit * 0.05
    side_pos[2] = obj_pos[2] + 0.01
    push_target = obj_pos.copy()
    push_target[:2] += unit * 0.2

    return FSM([
        FSMState(
            "reach_side",
            lambda o: reach(side_pos, o, controller),
            distance_reached(side_pos, REACH_THRESHOLD),
        ),
        FSMState(
            "push_to_goal",
            lambda o: reach(push_target, o, controller),
            distance_reached(push_target, 0.06),
        ),
    ])


def _build_door_open_fsm(
    obs: ObsParser, controller: Controller, goal: np.ndarray
) -> FSM:
    handle_pos = obs.obj1_pos.copy()
    above_handle = handle_pos.copy()
    above_handle[2] += APPROACH_HEIGHT
    pull_target = handle_pos.copy()
    pull_target[0] -= 0.15
    pull_target[1] += 0.1

    return FSM([
        FSMState(
            "approach_above_handle",
            lambda o: approach_above(handle_pos, o, controller),
            distance_reached(above_handle, REACH_THRESHOLD),
        ),
        FSMState(
            "descend_slow",
            lambda o: descend_slow(handle_pos, o, controller),
            distance_reached(handle_pos, 0.03),
        ),
        FSMState(
            "grasp_hold",
            lambda o: grasp_hold(o),
            after_steps(GRASP_HOLD_STEPS),
        ),
        FSMState(
            "pull",
            lambda o: move_to(pull_target, o, controller),
            distance_reached(pull_target, 0.06),
        ),
    ])


def _build_button_press_topdown_fsm(
    obs: ObsParser, controller: Controller, goal: np.ndarray
) -> FSM:
    button_pos = obs.obj1_pos.copy()
    above_button = button_pos.copy()
    above_button[2] += APPROACH_HEIGHT
    press_target = button_pos.copy()
    press_target[2] -= 0.02

    return FSM([
        FSMState(
            "reach_above",
            lambda o: reach(above_button, o, controller),
            distance_reached(above_button, REACH_THRESHOLD),
        ),
        FSMState(
            "press_down",
            lambda o: descend_slow(press_target, o, controller, speed_scale=0.5),
            distance_reached(press_target, REACH_THRESHOLD),
        ),
    ])


def _build_faucet_open_fsm(
    obs: ObsParser, controller: Controller, goal: np.ndarray
) -> FSM:
    faucet_pos = obs.obj1_pos.copy()
    above_faucet = faucet_pos.copy()
    above_faucet[2] += APPROACH_HEIGHT
    turn_target = faucet_pos.copy()
    turn_target[1] += 0.08
    turn_target[0] -= 0.05

    return FSM([
        FSMState(
            "approach_above_faucet",
            lambda o: approach_above(faucet_pos, o, controller),
            distance_reached(above_faucet, REACH_THRESHOLD),
        ),
        FSMState(
            "descend_slow",
            lambda o: descend_slow(faucet_pos, o, controller),
            distance_reached(faucet_pos, 0.03),
        ),
        FSMState(
            "grasp_hold",
            lambda o: grasp_hold(o),
            after_steps(GRASP_HOLD_STEPS),
        ),
        FSMState(
            "turn",
            lambda o: move_to(turn_target, o, controller),
            distance_reached(turn_target, 0.06),
        ),
    ])


def _build_shelf_place_fsm(
    obs: ObsParser, controller: Controller, goal: np.ndarray
) -> FSM:
    obj_pos = obs.obj1_pos.copy()
    above_obj = obj_pos.copy()
    above_obj[2] += APPROACH_HEIGHT
    lift_target = obj_pos.copy()
    lift_target[2] += 0.18

    return FSM([
        FSMState(
            "approach_above_obj",
            lambda o: approach_above(obj_pos, o, controller),
            distance_reached(above_obj, REACH_THRESHOLD),
        ),
        FSMState(
            "descend_slow",
            lambda o: descend_slow(obj_pos, o, controller),
            distance_reached(obj_pos, 0.03),
        ),
        FSMState(
            "grasp_hold",
            lambda o: grasp_hold(o),
            after_steps(GRASP_HOLD_STEPS),
        ),
        FSMState(
            "lift",
            lambda o: lift(0.18, o, controller),
            distance_reached(lift_target, REACH_THRESHOLD),
        ),
        FSMState(
            "move_to_shelf",
            lambda o: move_to(
                np.array([o.goal_pos[0], o.goal_pos[1], lift_target[2]], dtype=np.float32) if o.has_goal else lift_target,
                o, controller,
            ),
            lambda o: o.has_goal and bool(np.linalg.norm(o.ee_pos - np.array([o.goal_pos[0], o.goal_pos[1], lift_target[2]])) < 0.06),
        ),
        FSMState(
            "release",
            lambda o: release(o),
            after_steps(RELEASE_HOLD_STEPS),
        ),
    ])


def _build_basketball_fsm(
    obs: ObsParser, controller: Controller, goal: np.ndarray
) -> FSM:
    ball_pos = obs.obj1_pos.copy()
    above_ball = ball_pos.copy()
    above_ball[2] += APPROACH_HEIGHT
    lift_target = ball_pos.copy()
    lift_target[2] += 0.22

    return FSM([
        FSMState(
            "approach_above_ball",
            lambda o: approach_above(ball_pos, o, controller),
            distance_reached(above_ball, REACH_THRESHOLD),
        ),
        FSMState(
            "descend_slow",
            lambda o: descend_slow(ball_pos, o, controller),
            distance_reached(ball_pos, 0.03),
        ),
        FSMState(
            "grasp_hold",
            lambda o: grasp_hold(o),
            after_steps(GRASP_HOLD_STEPS),
        ),
        FSMState(
            "lift",
            lambda o: lift(0.22, o, controller),
            distance_reached(lift_target, REACH_THRESHOLD),
        ),
        FSMState(
            "reach_hoop",
            lambda o: move_to(
                np.array([o.goal_pos[0], o.goal_pos[1], lift_target[2] + APPROACH_HEIGHT], dtype=np.float32) if o.has_goal else lift_target,
                o, controller,
            ),
            lambda o: o.has_goal and bool(np.linalg.norm(o.ee_pos - np.array([o.goal_pos[0], o.goal_pos[1], lift_target[2] + APPROACH_HEIGHT])) < 0.06),
        ),
        FSMState(
            "release",
            lambda o: release(o),
            after_steps(RELEASE_HOLD_STEPS),
        ),
    ])


TASK_FSM_BUILDERS: dict[
    str, Callable[[ObsParser, Controller, np.ndarray], FSM]
] = {
    "reach-v3": _build_reach_fsm,
    "push-v3": _build_push_fsm,
    "pick-place-v3": _build_pick_place_fsm,
    "drawer-open-v3": _build_drawer_open_fsm,
    "sweep-v3": _build_sweep_fsm,
    "door-open-v3": _build_door_open_fsm,
    "button-press-topdown-v3": _build_button_press_topdown_fsm,
    "faucet-open-v3": _build_faucet_open_fsm,
    "shelf-place-v3": _build_shelf_place_fsm,
    "basketball-v3": _build_basketball_fsm,
}

GRASP_STATES = frozenset({
    "approach_above_obj", "approach_above_handle", "approach_above_ball",
    "approach_above_faucet",
    "descend_slow", "descend_to_handle", "descend_to_obj",
    "descend_to_faucet", "descend_to_button",
    "grasp_hold", "close_gripper",
})

RETRY_ORIGIN_STATES = {
    "pick-place-v3": "approach_above_obj",
    "drawer-open-v3": "approach_above_handle",
    "door-open-v3": "approach_above_handle",
    "faucet-open-v3": "approach_above_faucet",
    "shelf-place-v3": "approach_above_obj",
    "basketball-v3": "approach_above_ball",
}

POST_GRASP_STATES = frozenset({
    "lift", "move_to_goal", "move_to_shelf", "reach_hoop",
    "pull_back", "pull", "turn",
    "impedance_pull", "continuous_pull", "arc_turn",
    "force_press", "contact_sweep", "descend_to_shelf",
})


class MuJoCoForceSensor:
    """Access MuJoCo contact/joint data from a MetaWorld env.

    MetaWorld V3 exposes env.model (MjModel) and env.data (MjData).
    This helper provides force-aware queries for v2 FSM builders.
    """

    GRIPPER_GEOMS = {"rightpad_geom", "leftpad_geom", "rightclaw_it", "leftclaw_it"}
    HANDLE_GEOMS = {"handle", "objGeom", "btnGeom"}

    def __init__(self, env: Any) -> None:
        self._env = env

    @property
    def _model(self) -> Any:
        return self._env.model

    @property
    def _data(self) -> Any:
        return self._env.data

    def _geom_id(self, name: str) -> int | None:
        try:
            return int(self._model.geom(name).id)
        except Exception:
            return None

    def _geom_name(self, gid: int) -> str:
        try:
            return self._model.geom(gid).name
        except Exception:
            return ""

    def has_contact(self, geom_a: str, geom_b: str) -> bool:
        id_a = self._geom_id(geom_a)
        id_b = self._geom_id(geom_b)
        if id_a is None or id_b is None:
            return False
        for i in range(self._data.ncon):
            c = self._data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 == id_a and g2 == id_b) or (g1 == id_b and g2 == id_a):
                return True
        return False

    def gripper_has_contact(self, target_geom: str) -> bool:
        for g in self.GRIPPER_GEOMS:
            if self.has_contact(g, target_geom):
                return True
        return False

    def any_gripper_contact(self) -> bool:
        target_ids = set()
        for i in range(self._data.ncon):
            c = self._data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            n1 = self._geom_name(g1)
            n2 = self._geom_name(g2)
            if n1 in self.GRIPPER_GEOMS or n2 in self.GRIPPER_GEOMS:
                return True
        return False

    def get_joint_pos(self, joint_name: str) -> float:
        try:
            jnt = self._model.joint(joint_name)
            addr = int(jnt.qposadr[0]) if hasattr(jnt.qposadr, '__len__') else int(jnt.qposadr)
            return float(self._data.qpos[addr])
        except Exception:
            return 0.0

    def get_contact_force_magnitude(self, geom_a: str, geom_b: str) -> float:
        id_a = self._geom_id(geom_a)
        id_b = self._geom_id(geom_b)
        if id_a is None or id_b is None:
            return 0.0
        total = 0.0
        for i in range(self._data.ncon):
            c = self._data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 == id_a and g2 == id_b) or (g1 == id_b and g2 == id_a):
                try:
                    force = np.linalg.norm(self._data.efc_force[c.efc_address: c.efc_address + 1])
                    total += force
                except Exception:
                    pass
        return total


class ContinuousPull:
    """Stateful continuous pull action for door/drawer tasks.

    Instead of moving to a fixed target, applies continuous velocity
    in the pull direction while tracking the handle position to
    maintain contact. Key insight: the handle moves as the door/drawer
    opens, so we must track it continuously.
    """

    def __init__(
        self,
        pull_dir: np.ndarray,
        controller: Controller,
        n_steps: int = IMPEDANCE_PULL_STEPS,
        pull_speed: float = 0.4,
        track_gain: float = 0.5,
    ) -> None:
        self.pull_dir = pull_dir / max(np.linalg.norm(pull_dir), 1e-6)
        self.controller = controller
        self.n_steps = n_steps
        self.pull_speed = pull_speed
        self.track_gain = track_gain
        self.count = 0

    def __call__(self, obs: ObsParser) -> np.ndarray:
        self.count += 1
        handle_pos = obs.obj1_pos
        ee = obs.ee_pos

        pull_component = self.pull_dir * self.pull_speed

        track_error = handle_pos - ee
        track_correction = track_error * self.track_gain
        pull_axis_proj = np.dot(track_correction, self.pull_dir)
        track_correction -= pull_axis_proj * self.pull_dir

        delta = pull_component + track_correction
        delta = np.clip(delta, -1.0, 1.0)

        return np.array([delta[0], delta[1], delta[2], 1.0], dtype=np.float32)

    def is_done(self, obs: ObsParser) -> bool:
        return self.count >= self.n_steps

    def reset(self) -> None:
        self.count = 0


class ForcePress:
    """Stateful force-controlled press action for button tasks.

    Descends slowly with increasing force until button is pressed.
    Uses reward feedback to detect success.
    """

    def __init__(
        self,
        target_pos: np.ndarray,
        controller: Controller,
        n_steps: int = FORCE_PRESS_STEPS,
        press_speed: float = 0.15,
    ) -> None:
        self.target_pos = target_pos.copy()
        self.controller = controller
        self.n_steps = n_steps
        self.press_speed = press_speed
        self.count = 0

    def __call__(self, obs: ObsParser) -> np.ndarray:
        self.count += 1
        ee = obs.ee_pos
        error = self.target_pos - ee
        error[:2] *= 0.5
        error[2] *= self.press_speed * min(1.0 + self.count * 0.02, 3.0)
        delta = self.controller.kp * error
        delta = np.clip(delta, -1.0, 1.0)
        return np.array([delta[0], delta[1], delta[2], -1.0], dtype=np.float32)

    def is_done(self, obs: ObsParser) -> bool:
        return self.count >= self.n_steps

    def reset(self) -> None:
        self.count = 0


class ArcTurn:
    """Stateful arc motion for faucet/rotation tasks.

    Moves the end-effector along a circular arc around a center point.
    """

    def __init__(
        self,
        center: np.ndarray,
        radius: float,
        start_angle: float,
        sweep_angle: float,
        controller: Controller,
        n_steps: int = ARC_TURN_STEPS,
        height: float = 0.0,
    ) -> None:
        self.center = center.copy()
        self.radius = radius
        self.start_angle = start_angle
        self.sweep_angle = sweep_angle
        self.controller = controller
        self.n_steps = n_steps
        self.height = height
        self.count = 0

    def __call__(self, obs: ObsParser) -> np.ndarray:
        self.count += 1
        t = self.count / max(self.n_steps, 1)
        angle = self.start_angle + self.sweep_angle * t
        target = self.center.copy()
        target[0] += self.radius * np.cos(angle)
        target[1] += self.radius * np.sin(angle)
        target[2] += self.height
        delta = self.controller.compute(target, obs.ee_pos, obs.ee_velocity)
        return np.array([delta[0], delta[1], delta[2], 1.0], dtype=np.float32)

    def is_done(self, obs: ObsParser) -> bool:
        return self.count >= self.n_steps

    def reset(self) -> None:
        self.count = 0


class ContactSweep:
    """Stateful contact-maintaining sweep for sweep task.

    Pushes laterally while maintaining downward contact force.
    """

    def __init__(
        self,
        push_dir: np.ndarray,
        controller: Controller,
        n_steps: int = 80,
        push_speed: float = 0.3,
        contact_z_offset: float = -0.005,
    ) -> None:
        self.push_dir = push_dir / max(np.linalg.norm(push_dir), 1e-6)
        self.controller = controller
        self.n_steps = n_steps
        self.push_speed = push_speed
        self.contact_z_offset = contact_z_offset
        self.count = 0

    def __call__(self, obs: ObsParser) -> np.ndarray:
        self.count += 1
        obj_pos = obs.obj1_pos
        target = obj_pos + self.push_dir * 0.05
        target[2] = obj_pos[2] + self.contact_z_offset
        delta = self.controller.compute(target, obs.ee_pos, obs.ee_velocity)
        delta[:2] = np.clip(delta[:2], -1.0, 1.0)
        delta[2] = min(delta[2], 0.05)
        return np.array([delta[0], delta[1], delta[2], -1.0], dtype=np.float32)

    def is_done(self, obs: ObsParser) -> bool:
        return self.count >= self.n_steps

    def reset(self) -> None:
        self.count = 0


def _build_door_open_v2_fsm(
    obs: ObsParser, controller: Controller, goal: np.ndarray, env_ref: Any = None,
) -> FSM:
    handle_pos = obs.obj1_pos.copy()
    above_handle = handle_pos.copy()
    above_handle[2] += APPROACH_HEIGHT

    PULL_FORCE = 0.5

    def impedance_pull_action(o: ObsParser) -> np.ndarray:
        goal_pos = o.goal_pos
        if np.linalg.norm(goal_pos) < 0.05:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        ee = o.ee_pos
        error = goal_pos - ee
        norm = np.linalg.norm(error)
        if norm < 1e-6:
            direction = np.array([-1.0, 0.05, 0.0])
        else:
            direction = error / norm
        action_xyz = PULL_FORCE * direction
        return np.array([action_xyz[0], action_xyz[1], action_xyz[2], 1.0], dtype=np.float32)

    pull_counter = [0]

    def door_open_check(o: ObsParser) -> bool:
        pull_counter[0] += 1
        return pull_counter[0] >= IMPEDANCE_PULL_STEPS

    return FSM([
        FSMState(
            "approach_above_handle",
            lambda o: approach_above(handle_pos, o, controller),
            distance_reached(above_handle, REACH_THRESHOLD),
        ),
        FSMState(
            "descend_to_handle",
            lambda o: reach(handle_pos, o, controller),
            distance_reached(handle_pos, 0.015),
        ),
        FSMState(
            "grasp_hold",
            lambda o: grasp_hold(o),
            after_steps(FORCE_GRASP_HOLD_STEPS),
        ),
        FSMState(
            "impedance_pull",
            impedance_pull_action,
            door_open_check,
        ),
    ])


def _build_button_press_v2_fsm(
    obs: ObsParser, controller: Controller, goal: np.ndarray, env_ref: Any = None,
) -> FSM:
    button_pos = obs.obj1_pos.copy()
    above_button = button_pos.copy()
    above_button[2] += APPROACH_HEIGHT

    PRESS_FORCE = 0.3
    press_counter = [0]

    def force_press_action(o: ObsParser) -> np.ndarray:
        press_counter[0] += 1
        goal_pos = o.goal_pos
        ee = o.ee_pos
        if np.linalg.norm(goal_pos) > 0.05:
            xy_error = goal_pos[:2] - ee[:2]
            xy_correction = np.clip(xy_error * 3.0, -0.3, 0.3)
            return np.array([xy_correction[0], xy_correction[1], -PRESS_FORCE, -1.0], dtype=np.float32)
        return np.array([0.0, 0.0, -PRESS_FORCE, -1.0], dtype=np.float32)

    def button_pressed_check(o: ObsParser) -> bool:
        return press_counter[0] >= FORCE_PRESS_STEPS

    return FSM([
        FSMState(
            "reach_above",
            lambda o: reach(above_button, o, controller),
            distance_reached(above_button, REACH_THRESHOLD),
        ),
        FSMState(
            "descend_to_button",
            lambda o: reach(button_pos, o, controller),
            distance_reached(button_pos, 0.02),
        ),
        FSMState(
            "force_press",
            force_press_action,
            button_pressed_check,
        ),
    ])


def _build_faucet_open_v2_fsm(
    obs: ObsParser, controller: Controller, goal: np.ndarray, env_ref: Any = None,
) -> FSM:
    faucet_pos = obs.obj1_pos.copy()
    above_faucet = faucet_pos.copy()
    above_faucet[2] += APPROACH_HEIGHT

    TURN_FORCE = 0.5
    turn_counter = [0]

    def arc_turn_action(o: ObsParser) -> np.ndarray:
        turn_counter[0] += 1
        goal_pos = o.goal_pos
        if np.linalg.norm(goal_pos) > 0.05:
            ee = o.ee_pos
            error = goal_pos - ee
            norm = np.linalg.norm(error)
            if norm < 1e-6:
                direction = np.array([0.4, 0.4, 0.0])
            else:
                direction = error / norm
            action_xyz = TURN_FORCE * direction
            return np.array([action_xyz[0], action_xyz[1], action_xyz[2], 1.0], dtype=np.float32)
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    def faucet_open_check(o: ObsParser) -> bool:
        return turn_counter[0] >= ARC_TURN_STEPS

    return FSM([
        FSMState(
            "approach_above_faucet",
            lambda o: approach_above(faucet_pos, o, controller),
            distance_reached(above_faucet, REACH_THRESHOLD),
        ),
        FSMState(
            "descend_to_faucet",
            lambda o: reach(faucet_pos, o, controller),
            distance_reached(faucet_pos, 0.015),
        ),
        FSMState(
            "grasp_hold",
            lambda o: grasp_hold(o),
            after_steps(FORCE_GRASP_HOLD_STEPS),
        ),
        FSMState(
            "arc_turn",
            arc_turn_action,
            faucet_open_check,
        ),
    ])


def _build_drawer_open_v2_fsm(
    obs: ObsParser, controller: Controller, goal: np.ndarray, env_ref: Any = None,
) -> FSM:
    handle_pos = obs.obj1_pos.copy()
    above_handle = handle_pos.copy()
    above_handle[2] += APPROACH_HEIGHT

    PULL_FORCE = 0.5

    def continuous_pull_action(o: ObsParser) -> np.ndarray:
        goal_pos = o.goal_pos
        if np.linalg.norm(goal_pos) > 0.05:
            ee = o.ee_pos
            error = goal_pos - ee
            error[0] = 0.0
            error[2] = 0.0
            norm = np.linalg.norm(error)
            if norm < 1e-6:
                direction = np.array([0.0, -1.0, 0.0])
            else:
                direction = error / norm
            action_xyz = PULL_FORCE * direction
            return np.array([action_xyz[0], action_xyz[1], action_xyz[2], 1.0], dtype=np.float32)
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    pull_counter = [0]

    def drawer_open_check(o: ObsParser) -> bool:
        pull_counter[0] += 1
        return pull_counter[0] >= IMPEDANCE_PULL_STEPS

    return FSM([
        FSMState(
            "approach_above_handle",
            lambda o: approach_above(handle_pos, o, controller),
            distance_reached(above_handle, REACH_THRESHOLD),
        ),
        FSMState(
            "descend_to_handle",
            lambda o: reach(handle_pos, o, controller),
            distance_reached(handle_pos, 0.015),
        ),
        FSMState(
            "grasp_hold",
            lambda o: grasp_hold(o),
            after_steps(FORCE_GRASP_HOLD_STEPS),
        ),
        FSMState(
            "continuous_pull",
            continuous_pull_action,
            drawer_open_check,
        ),
    ])


def _build_sweep_v2_fsm(
    obs: ObsParser, controller: Controller, goal: np.ndarray, env_ref: Any = None,
) -> FSM:
    obj_pos = obs.obj1_pos.copy()

    target_pos = goal.copy()
    if np.linalg.norm(target_pos) < 1e-3:
        target_pos = obj_pos + np.array([0.3, 0.0, 0.0])

    push_dir = target_pos - obj_pos
    push_dir[2] = 0.0
    if np.linalg.norm(push_dir[:2]) < 1e-6:
        push_dir = np.array([1.0, 0.0, 0.0])
    push_dir_unit = push_dir / np.linalg.norm(push_dir)

    PUSH_FORCE = 0.5
    MAX_PUSH_STEPS = 180
    GOAL_DIST_THRESHOLD = 0.04
    push_counter = [0]

    def _get_push_direction(o: ObsParser) -> np.ndarray:
        goal_pos = o.goal_pos
        if np.linalg.norm(goal_pos) > 0.05:
            d = goal_pos - o.obj1_pos
            d[2] = 0.0
            n = np.linalg.norm(d[:2])
            if n > 1e-6:
                return d[:2] / n
        return np.array([0.0, 0.0])

    def _get_behind_pos(o: ObsParser) -> np.ndarray:
        direction = _get_push_direction(o)
        behind = o.obj1_pos.copy()
        behind[:2] -= direction * 0.03
        behind[2] = 0.046
        return behind

    def _get_above_behind(o: ObsParser) -> np.ndarray:
        behind = _get_behind_pos(o)
        behind[2] = o.obj1_pos[2] + APPROACH_HEIGHT
        return behind

    def reach_above_behind_action(o: ObsParser) -> np.ndarray:
        above = _get_above_behind(o)
        return reach(above, o, controller)

    def reach_above_behind_done(o: ObsParser) -> bool:
        above = _get_above_behind(o)
        return bool(np.linalg.norm(o.ee_pos - above) < 0.05)

    def descend_to_push_action(o: ObsParser) -> np.ndarray:
        behind = _get_behind_pos(o)
        return reach(behind, o, controller)

    def descend_to_push_done(o: ObsParser) -> bool:
        behind = _get_behind_pos(o)
        xy_close = np.linalg.norm(o.ee_pos[:2] - behind[:2]) < 0.03
        z_low = o.ee_pos[2] < 0.05
        return bool(xy_close and z_low)

    def contact_sweep_action(o: ObsParser) -> np.ndarray:
        push_counter[0] += 1
        direction = _get_push_direction(o)
        obj_pos_cur = o.obj1_pos
        z_target = max(obj_pos_cur[2] + 0.01, 0.025)
        z_error = z_target - o.ee_pos[2]
        z_action = np.clip(z_error * 8.0, -0.3, 0.3)
        return np.array([direction[0] * PUSH_FORCE, direction[1] * PUSH_FORCE, z_action, 1.0], dtype=np.float32)

    def sweep_done_check(o: ObsParser) -> bool:
        goal_pos = o.goal_pos
        if np.linalg.norm(goal_pos) > 0.05:
            dist = np.linalg.norm(o.obj1_pos[:2] - goal_pos[:2])
            if dist < GOAL_DIST_THRESHOLD:
                return True
        return push_counter[0] >= MAX_PUSH_STEPS

    return FSM([
        FSMState(
            "reach_above_behind",
            reach_above_behind_action,
            reach_above_behind_done,
        ),
        FSMState(
            "descend_to_push",
            descend_to_push_action,
            descend_to_push_done,
        ),
        FSMState(
            "close_gripper",
            lambda o: grasp_hold(o),
            after_steps(15),
        ),
        FSMState(
            "contact_sweep",
            contact_sweep_action,
            sweep_done_check,
        ),
    ])


def _build_shelf_place_v2_fsm(
    obs: ObsParser, controller: Controller, goal: np.ndarray, env_ref: Any = None,
) -> FSM:
    obj_pos = obs.obj1_pos.copy()
    above_obj = obj_pos.copy()
    above_obj[2] += APPROACH_HEIGHT

    target_pos = goal.copy()
    if np.linalg.norm(target_pos) < 1e-3:
        target_pos = np.array([0.0, 0.8, 0.3], dtype=np.float32)

    shelf_z = target_pos[2]
    lift_height = max(0.22, shelf_z - obj_pos[2] + 0.08)
    lift_target = obj_pos.copy()
    lift_target[2] += lift_height

    obj_pos_xy = obj_pos[:2].copy()

    def _get_shelf_above(o: ObsParser) -> np.ndarray | None:
        goal_pos = o.goal_pos
        if np.linalg.norm(goal_pos) > 0.05:
            shelf_above = goal_pos.copy()
            shelf_above[2] = lift_target[2]
            return shelf_above
        return None

    def _get_shelf_target(o: ObsParser) -> np.ndarray | None:
        goal_pos = o.goal_pos
        if np.linalg.norm(goal_pos) > 0.05:
            return goal_pos.copy()
        return None

    def move_to_shelf_action(o: ObsParser) -> np.ndarray:
        shelf_above = _get_shelf_above(o)
        if shelf_above is None:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        delta = controller.compute(shelf_above, o.ee_pos, o.ee_velocity)
        return np.array([delta[0], delta[1], delta[2], 1.0], dtype=np.float32)

    def move_to_shelf_done(o: ObsParser) -> bool:
        shelf_above = _get_shelf_above(o)
        if shelf_above is None:
            return False
        return bool(np.linalg.norm(o.ee_pos - shelf_above) < 0.04)

    def descend_to_shelf_action(o: ObsParser) -> np.ndarray:
        shelf_target = _get_shelf_target(o)
        if shelf_target is None:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        delta = controller.compute(shelf_target, o.ee_pos, o.ee_velocity)
        delta[:3] *= 0.3
        delta = np.clip(delta, -1.0, 1.0)
        return np.array([delta[0], delta[1], delta[2], 1.0], dtype=np.float32)

    def descend_to_shelf_done(o: ObsParser) -> bool:
        shelf_target = _get_shelf_target(o)
        if shelf_target is None:
            return False
        return bool(o.ee_pos[2] < (shelf_target[2] + 0.04))

    return FSM([
        FSMState(
            "approach_above_obj",
            lambda o: approach_above(obj_pos, o, controller),
            distance_reached(above_obj, REACH_THRESHOLD),
        ),
        FSMState(
            "descend_to_obj",
            lambda o: reach(obj_pos, o, controller),
            lambda o: o.ee_pos[2] < (obj_pos[2] + 0.03) and np.linalg.norm(o.ee_pos[:2] - obj_pos_xy) < 0.025,
        ),
        FSMState(
            "grasp_hold",
            lambda o: grasp_hold(o),
            after_steps(FORCE_GRASP_HOLD_STEPS + 5),
        ),
        FSMState(
            "lift",
            lambda o: move_to(lift_target, o, controller),
            distance_reached(lift_target, 0.03),
        ),
        FSMState(
            "move_to_shelf",
            move_to_shelf_action,
            move_to_shelf_done,
        ),
        FSMState(
            "descend_to_shelf",
            descend_to_shelf_action,
            descend_to_shelf_done,
        ),
        FSMState(
            "release",
            lambda o: release(o),
            after_steps(RELEASE_HOLD_STEPS + 5),
        ),
    ])


TASK_FSM_BUILDERS_V2: dict[
    str, Callable[[ObsParser, Controller, np.ndarray, Any], FSM]
] = {
    "door-open-v3": _build_door_open_v2_fsm,
    "button-press-topdown-v3": _build_button_press_v2_fsm,
    "faucet-open-v3": _build_faucet_open_v2_fsm,
    "drawer-open-v3": _build_drawer_open_v2_fsm,
    "sweep-v3": _build_sweep_v2_fsm,
    "shelf-place-v3": _build_shelf_place_v2_fsm,
}


def build_fsm(
    task: str,
    obs: ObsParser,
    controller: Controller,
    goal: np.ndarray,
    env_ref: Any = None,
    fsm_version: int = 1,
) -> FSM:
    norm = task.strip()
    if norm.endswith("-v2"):
        norm = norm[:-3] + "-v3"
    if fsm_version >= 2:
        builder_v2 = TASK_FSM_BUILDERS_V2.get(norm)
        if builder_v2 is not None:
            return builder_v2(obs, controller, goal, env_ref)
    builder = TASK_FSM_BUILDERS.get(norm)
    if builder is None:
        return _build_reach_fsm(obs, controller, goal)
    return builder(obs, controller, goal)


@dataclass
class ObjectSlot:
    last_pose: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    contact: bool = False
    grasped: bool = False


@dataclass
class Relation:
    subject: str
    relation_type: str
    obj: str


@dataclass
class Subgoal:
    name: str
    done: bool = False


class StructMemory:
    """Enhanced structured memory with objects, relations, plan, and events."""

    def __init__(self) -> None:
        self.objects: dict[str, ObjectSlot] = {}
        self.relations: list[Relation] = []
        self.plan: list[Subgoal] = []
        self.events: list[dict[str, Any]] = []
        self._stored_goal: np.ndarray | None = None

    def init_plan(self, state_names: list[str]) -> None:
        self.plan = [Subgoal(name=n, done=False) for n in state_names]

    def store_goal(self, goal: np.ndarray) -> None:
        if np.linalg.norm(goal) > 1e-6:
            self._stored_goal = goal.copy()

    def get_stored_goal(self) -> np.ndarray | None:
        return self._stored_goal

    def update(
        self,
        *,
        obs: ObsParser,
        action: np.ndarray,
        reward: float,
        info: dict[str, Any],
        step: int,
    ) -> None:
        ee = obs.ee_pos
        obj1 = obs.obj1_pos
        grip = obs.gripper

        if "ee" not in self.objects:
            self.objects["ee"] = ObjectSlot()
        if "obj1" not in self.objects:
            self.objects["obj1"] = ObjectSlot()

        self.objects["ee"].last_pose = ee.copy()
        self.objects["obj1"].last_pose = obj1.copy()

        ee_obj_dist = float(np.linalg.norm(ee - obj1))
        self.objects["ee"].contact = ee_obj_dist < 0.04
        self.objects["obj1"].contact = ee_obj_dist < 0.04
        self.objects["obj1"].grasped = obs.obj_grasped()

        self.relations = [
            r
            for r in self.relations
            if r.relation_type not in ("in_contact", "adjacent", "contained")
        ]
        if self.objects["ee"].contact:
            self.relations.append(Relation("ee", "in_contact", "obj1"))
        if ee_obj_dist < 0.08:
            self.relations.append(Relation("ee", "adjacent", "obj1"))
        if self.objects["obj1"].grasped:
            self.relations.append(Relation("ee", "contained", "obj1"))

        obs_summary = (
            f"ee({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}) "
            f"grip={grip:.3f} "
            f"obj1({obj1[0]:.3f},{obj1[1]:.3f},{obj1[2]:.3f}) "
            f"r={reward:.4f}"
        )
        action_summary = (
            f"[{action[0]:.3f},{action[1]:.3f},{action[2]:.3f},{action[3]:.3f}]"
        )
        outcome = (
            "grasped"
            if self.objects["obj1"].grasped
            else ("contact" if self.objects["ee"].contact else "none")
        )

        self.events.append(
            {
                "t": time.time(),
                "step": step,
                "obs_summary": obs_summary,
                "action": action_summary,
                "outcome": outcome,
                "reward": float(reward),
                "success": float(info.get("success", 0.0) or 0.0),
            }
        )
        if len(self.events) > 200:
            self.events = self.events[-200:]

    def retrieve(
        self, subgoal: str | None, *, k: int = 5
    ) -> list[dict[str, Any]]:
        if k <= 0:
            return []
        if subgoal is None:
            return list(self.events[-k:])
        relevant: list[dict[str, Any]] = []
        for ev in reversed(self.events):
            if subgoal in ev.get("obs_summary", "") or subgoal in ev.get(
                "outcome", ""
            ):
                relevant.append(ev)
                if len(relevant) >= k:
                    break
        if len(relevant) < k:
            seen_ids = {id(e) for e in relevant}
            remaining = [
                ev for ev in reversed(self.events) if id(ev) not in seen_ids
            ]
            relevant.extend(remaining[: k - len(relevant)])
        return relevant

    def mark_current_subgoal_done(self) -> None:
        for sg in self.plan:
            if not sg.done:
                sg.done = True
                break

    def current_subgoal(self) -> str | None:
        for sg in self.plan:
            if not sg.done:
                return sg.name
        return None

    def is_obj_grasped(self) -> bool:
        return self.objects.get("obj1", ObjectSlot()).grasped

    def is_ee_in_contact(self) -> bool:
        return self.objects.get("ee", ObjectSlot()).contact


class TextBuffer:
    """Enhanced text buffer with rule-based pattern matching."""

    def __init__(self, history: int = 20) -> None:
        self.history = history
        self.buf: list[str] = []
        self._stored_goal: np.ndarray | None = None

    def reset(self) -> None:
        self.buf = []
        self._stored_goal = None

    def store_goal(self, goal: np.ndarray) -> None:
        if np.linalg.norm(goal) > 1e-6:
            self._stored_goal = goal.copy()

    def get_stored_goal(self) -> np.ndarray | None:
        return self._stored_goal

    def append(
        self, obs: ObsParser, action: np.ndarray, reward: float
    ) -> None:
        ee = obs.ee_pos
        grip = obs.gripper
        obj1 = obs.obj1_pos
        goal = obs.goal_pos
        line = (
            f"ee({ee[0]:.2f},{ee[1]:.2f},{ee[2]:.2f}) "
            f"grip={grip:.2f} "
            f"obj({obj1[0]:.2f},{obj1[1]:.2f},{obj1[2]:.2f}) "
            f"goal({goal[0]:.2f},{goal[1]:.2f},{goal[2]:.2f}) "
            f"r={reward:.3f}"
        )
        self.buf.append(line)
        if len(self.buf) > self.history:
            self.buf = self.buf[-self.history :]

    def match_pattern(self, pattern: str) -> bool:
        for line in reversed(self.buf):
            if pattern in line:
                return True
        return False

    def gripper_recently_closed(self, n: int = 5) -> bool:
        for line in self.buf[-n:]:
            if "grip=" in line:
                try:
                    idx = line.index("grip=") + 5
                    val = float(line[idx : idx + 4])
                    if val < GRIPPER_CLOSE_THRESHOLD:
                        return True
                except (ValueError, IndexError):
                    continue
        return False

    def object_moving(self, n: int = 5) -> bool:
        if len(self.buf) < 2:
            return False
        recent = self.buf[-n:]
        obj_positions: list[list[float]] = []
        for line in recent:
            try:
                idx = line.index("obj(")
                end = line.index(")", idx)
                coords = line[idx + 4 : end].split(",")
                obj_positions.append([float(c) for c in coords])
            except (ValueError, IndexError):
                continue
        if len(obj_positions) < 2:
            return False
        first = np.array(obj_positions[0])
        last = np.array(obj_positions[-1])
        return float(np.linalg.norm(last - first)) > 0.01

    def reward_increasing(self, n: int = 5) -> bool:
        if len(self.buf) < 2:
            return False
        recent = self.buf[-n:]
        rewards: list[float] = []
        for line in recent:
            try:
                idx = line.index("r=") + 2
                val = float(line[idx : idx + 6])
                rewards.append(val)
            except (ValueError, IndexError):
                continue
        if len(rewards) < 2:
            return False
        return rewards[-1] > rewards[0]

    def object_lifted_after_grasp(self, n: int = 10) -> bool:
        if len(self.buf) < 2:
            return False
        recent = self.buf[-n:]
        obj_z_values: list[float] = []
        for line in recent:
            try:
                idx = line.index("obj(")
                end = line.index(")", idx)
                coords = line[idx + 4 : end].split(",")
                obj_z_values.append(float(coords[2]))
            except (ValueError, IndexError):
                continue
        if len(obj_z_values) < 2:
            return False
        return obj_z_values[-1] - obj_z_values[0] > GRASP_SUCCESS_LIFT_Z

    @property
    def last_line(self) -> str:
        return self.buf[-1] if self.buf else ""

    @property
    def lines(self) -> list[str]:
        return list(self.buf)


class FSMPolicyBase:
    """Base class for FSM-driven policies.

    Shared FSM + primitives across NoMemory, TextBuffer, and StructMemory.
    Subclasses override `_get_context_action` to inject memory-based adjustments.
    """

    def __init__(
        self,
        action_dim: int = ACTION_DIM,
        controller_type: str = "PD",
        persist_goal: bool = True,
        max_retries: int = MAX_GRASP_RETRIES,
        fsm_version: int = 1,
    ) -> None:
        self.action_dim = action_dim
        self.controller = _make_controller(controller_type)
        self.obs_parser = ObsParser(persist_goal=persist_goal)
        self.fsm: FSM | None = None
        self.task: str = ""
        self._initialized = False
        self._grasp_attempts = 0
        self._grasp_success = False
        self._retry_count = 0
        self._max_retries = max_retries
        self._persist_goal = persist_goal
        self._env_ref: Any = None
        self._fsm_version = fsm_version

    def set_env_ref(self, env: Any) -> None:
        self._env_ref = env

    def reset(self, task: str) -> None:
        self.task = task
        self.fsm = None
        self._initialized = False
        self.obs_parser = ObsParser(persist_goal=self._persist_goal)
        self._grasp_attempts = 0
        self._grasp_success = False
        self._retry_count = 0

    def _ensure_fsm(self, obs: Any) -> None:
        if self._initialized:
            return
        self.obs_parser.update(obs)
        goal = self.obs_parser.goal_pos
        self.fsm = build_fsm(self.task, self.obs_parser, self.controller, goal, self._env_ref, self._fsm_version)
        self._initialized = True
        self._on_fsm_built()

    def _on_fsm_built(self) -> None:
        pass

    def _on_fsm_transition(self) -> None:
        pass

    def _reset_fsm_for_goal(self, goal: np.ndarray) -> None:
        self.obs_parser.set_goal(goal)
        self.fsm = build_fsm(self.task, self.obs_parser, self.controller, goal, self._env_ref, self._fsm_version)
        self._initialized = True
        self._retry_count = 0
        self._on_fsm_built()

    def act(self, obs: Any) -> Any:
        self._ensure_fsm(obs)
        self.obs_parser.update(obs)
        if self.fsm is None:
            return np.zeros(self.action_dim, dtype=np.float32)

        prev_idx = self.fsm.current_idx
        action = self.fsm.step(self.obs_parser)
        if self.fsm.current_idx > prev_idx:
            self._on_fsm_transition()
        context_action = self._get_context_action(action)
        return context_action

    def _get_context_action(self, action: np.ndarray) -> np.ndarray:
        return action

    def observe(
        self, obs: Any, action: Any, reward: float, info: dict[str, Any]
    ) -> None:
        pass

    @property
    def grasp_attempts(self) -> int:
        return self._grasp_attempts

    @property
    def grasp_success(self) -> bool:
        return self._grasp_success

    @property
    def retry_count(self) -> int:
        return self._retry_count


class NoMemoryPolicy(FSMPolicyBase):
    """Stateless FSM+primitives policy — no history context.

    persist_goal=False: ObsParser does not store goal across steps.
    Under goal flashing, NoMemory can only navigate when the goal is
    visible in the current observation. Between flashes, goal_pos
    returns zeros and the FSM action functions produce zero/no-op actions.
    No failure recovery: if a grasp fails, the FSM proceeds regardless.
    """

    def __init__(
        self, action_dim: int = ACTION_DIM, controller_type: str = "PD",
        max_retries: int = 0, fsm_version: int = 1,
    ) -> None:
        super().__init__(
            action_dim=action_dim,
            controller_type=controller_type,
            persist_goal=False,
            max_retries=max_retries,
            fsm_version=fsm_version,
        )


class TextBufferPolicy(FSMPolicyBase):
    """FSM+primitives with text buffer context.

    persist_goal=True: ObsParser stores goal, so navigation works across flashes.
    Additionally stores goal in text buffer for pattern-based retrieval.
    Checks text patterns to detect stuck states and grasp failures,
    then retries from approach_above with perturbation.
    """

    def __init__(
        self,
        action_dim: int = ACTION_DIM,
        history: int = 20,
        controller_type: str = "PD",
        max_retries: int = MAX_GRASP_RETRIES,
        fsm_version: int = 1,
    ) -> None:
        super().__init__(
            action_dim=action_dim,
            controller_type=controller_type,
            persist_goal=True,
            max_retries=max_retries,
            fsm_version=fsm_version,
        )
        self.history = history
        self.text_buf = TextBuffer(history=history)
        self._stuck_counter = 0
        self._grasp_retry_count = 0

    def reset(self, task: str) -> None:
        super().reset(task)
        self.text_buf.reset()
        self._stuck_counter = 0
        self._grasp_retry_count = 0

    def _on_fsm_built(self) -> None:
        if self.fsm is not None:
            self.text_buf.reset()
            if self.obs_parser.has_goal:
                self.text_buf.store_goal(self.obs_parser.goal_pos)

    def observe(
        self, obs: Any, action: Any, reward: float, info: dict[str, Any]
    ) -> None:
        self.obs_parser.update(obs)
        if bool(info.get("goal_switched", False)):
            prev_goal = np.asarray(info.get("prev_goal", self.obs_parser.goal_pos), dtype=np.float32)
            self.obs_parser.lock_goal_updates(prev_goal)
            self._reset_fsm_for_goal(prev_goal)
        if self.obs_parser.has_goal:
            self.text_buf.store_goal(self.obs_parser.goal_pos)
        act_arr = (
            np.asarray(action, dtype=np.float32)
            if action is not None
            else np.zeros(ACTION_DIM, dtype=np.float32)
        )
        self.text_buf.append(self.obs_parser, act_arr, float(reward))

    def _get_context_action(self, action: np.ndarray) -> np.ndarray:
        if self.fsm is None:
            return action

        state_name = self.fsm.current_state_name

        if state_name in POST_GRASP_STATES:
            if not self.obs_parser.obj_grasped() and self._grasp_retry_count < self._max_retries:
                self._grasp_retry_count += 1
                self._retry_count += 1
                self._grasp_attempts += 1
                retry_origin = RETRY_ORIGIN_STATES.get(self.task)
                if retry_origin and self.fsm.go_back_to(retry_origin):
                    return action

        if state_name not in ("done",) and not self.text_buf.reward_increasing(
            n=10
        ):
            self._stuck_counter += 1
        else:
            self._stuck_counter = 0

        if self._stuck_counter > 30 and self.fsm.current_idx > 0:
            if state_name not in ("grasp_hold", "release"):
                prev_name = self.fsm.states[max(0, self.fsm.current_idx - 1)].name
                self.fsm.go_back_to(prev_name)
                self._stuck_counter = 0

        if state_name in ("lift",) and self.text_buf.object_lifted_after_grasp():
            self._grasp_success = True

        return action


class StructMemoryPolicy(FSMPolicyBase):
    """FSM+primitives with structured memory context.

    persist_goal=True: ObsParser stores goal, so navigation works across flashes.
    Additionally stores goal in structured memory for reliable retrieval.
    Queries structured memory to detect grasp failures and retry with
    XY perturbation from approach_above.
    """

    def __init__(
        self,
        action_dim: int = ACTION_DIM,
        top_k: int = 5,
        controller_type: str = "PD",
        max_retries: int = MAX_GRASP_RETRIES,
        enable_perturbation: bool = True,
        fsm_version: int = 1,
    ) -> None:
        super().__init__(
            action_dim=action_dim,
            controller_type=controller_type,
            persist_goal=True,
            max_retries=max_retries,
            fsm_version=fsm_version,
        )
        self.top_k = top_k
        self.mem = StructMemory()
        self._grasp_retry_count = 0
        self._perturbation: np.ndarray | None = None
        self._enable_perturbation = enable_perturbation

    def reset(self, task: str) -> None:
        super().reset(task)
        self.mem = StructMemory()
        self._grasp_retry_count = 0
        self._perturbation = None

    def _on_fsm_built(self) -> None:
        if self.fsm is not None:
            state_names = [s.name for s in self.fsm.states]
            self.mem.init_plan(state_names)
            if self.obs_parser.has_goal:
                self.mem.store_goal(self.obs_parser.goal_pos)

    def _on_fsm_transition(self) -> None:
        self.mem.mark_current_subgoal_done()

    def observe(
        self, obs: Any, action: Any, reward: float, info: dict[str, Any]
    ) -> None:
        pass

    def update(
        self,
        *,
        obs: Any,
        action: Any,
        reward: float,
        info: dict[str, Any],
        step: int,
    ) -> None:
        self.obs_parser.update(obs)
        if bool(info.get("goal_switched", False)):
            new_goal = np.asarray(info.get("new_goal", self.obs_parser.goal_pos), dtype=np.float32)
            self.obs_parser.unlock_goal_updates()
            self.obs_parser.set_goal(new_goal)
            self.mem.store_goal(new_goal)
            self._reset_fsm_for_goal(new_goal)
        elif self.obs_parser.has_goal:
            self.mem.store_goal(self.obs_parser.goal_pos)
        act_arr = (
            np.asarray(action, dtype=np.float32)
            if action is not None
            else np.zeros(ACTION_DIM, dtype=np.float32)
        )
        self.mem.update(
            obs=self.obs_parser,
            action=act_arr,
            reward=float(reward),
            info=info,
            step=step,
        )

    def _get_context_action(self, action: np.ndarray) -> np.ndarray:
        if self.fsm is None:
            return action

        state_name = self.fsm.current_state_name

        if state_name in POST_GRASP_STATES:
            if not self.mem.is_obj_grasped() and self._grasp_retry_count < self._max_retries:
                self._grasp_retry_count += 1
                self._retry_count += 1
                self._grasp_attempts += 1
                if self._enable_perturbation:
                    self._perturbation = np.random.uniform(
                        -PERTURB_XY_RANGE, PERTURB_XY_RANGE, size=2
                    ).astype(np.float32)
                else:
                    self._perturbation = None
                retry_origin = RETRY_ORIGIN_STATES.get(self.task)
                if retry_origin and self.fsm.go_back_to(retry_origin):
                    return action

        if state_name in ("approach_above_obj", "approach_above_handle",
                          "approach_above_ball", "approach_above_faucet"):
            if self._perturbation is not None:
                perturbed = action.copy()
                perturbed[0] += self._perturbation[0] * 0.5
                perturbed[1] += self._perturbation[1] * 0.5
                self._perturbation = None
                return perturbed

        if state_name == "grasp_hold":
            _ctx = self.mem.retrieve("grasp", k=self.top_k)

        if state_name in ("lift",) and self.mem.is_obj_grasped():
            self._grasp_success = True

        return action
