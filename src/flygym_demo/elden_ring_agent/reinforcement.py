"""Gymnasium environment for reinforcement-learning the Elden Ring action policy.

This is a compact combat proxy, not a hidden connection to the game process.
The real game does not expose a resettable state or reward through the current
screen adapter. Use this environment to learn action timing and then evaluate
the policy in the live controller.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .policy import Action

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as error:  # pragma: no cover - optional dependency
    raise ImportError(
        "Elden Ring reinforcement learning requires `gymnasium`; "
        "install with `uv pip install 'flygym[rl]'`."
    ) from error


@dataclass(slots=True)
class CombatState:
    enemy_distance: float = 0.7
    enemy_direction: float = 0.0
    enemy_attacking: float = 0.0
    health: float = 1.0
    stamina: float = 1.0
    target_direction: float = 0.0


class EldenRingCombatEnv(gym.Env):
    """Small continuous-observation proxy for PPO training.

    Observation order is ``distance, enemy_direction, enemy_attacking,
    health, stamina, target_direction``. The action is ``(action_id, turn)``;
    ``action_id`` is rounded to one of the public :class:`Action` values.
    """

    metadata = {"render_modes": []}

    def __init__(self, *, max_steps: int = 500, seed: int | None = None) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.action_names = list(Action)
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([len(self.action_names) - 1, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.array([0.0, -1.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32),
            high=np.ones(6, dtype=np.float32),
            dtype=np.float32,
        )
        self._rng = np.random.default_rng(seed)
        self.state = CombatState()
        self.steps = 0

    def _observation(self) -> np.ndarray:
        state = self.state
        return np.asarray(
            [
                state.enemy_distance,
                state.enemy_direction,
                state.enemy_attacking,
                state.health,
                state.stamina,
                state.target_direction,
            ],
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        rng = self.np_random if seed is not None else self._rng
        self.state = CombatState(
            enemy_distance=float(rng.uniform(0.15, 0.9)),
            enemy_direction=float(rng.uniform(-1.0, 1.0)),
            enemy_attacking=float(rng.integers(0, 2)),
            health=float(rng.uniform(0.65, 1.0)),
            stamina=float(rng.uniform(0.5, 1.0)),
            target_direction=float(rng.uniform(-1.0, 1.0)),
        )
        self.steps = 0
        return self._observation(), {}

    def step(self, action: np.ndarray):
        action_id = int(np.clip(np.rint(float(action[0])), 0, len(self.action_names) - 1))
        turn = float(np.clip(action[1], -1.0, 1.0))
        selected = self.action_names[action_id]
        state = self.state
        reward = 0.02

        if selected == Action.ATTACK and state.enemy_distance < 0.3:
            reward += 1.2
            state.enemy_distance = min(1.0, state.enemy_distance + 0.2)
            state.stamina = max(0.0, state.stamina - 0.12)
        elif selected == Action.DODGE and state.enemy_attacking > 0.5:
            reward += 1.0
            state.enemy_attacking = 0.0
            state.stamina = max(0.0, state.stamina - 0.1)
        elif selected == Action.HEAL and state.health < 0.45 and state.stamina > 0.1:
            reward += 0.8
            state.health = min(1.0, state.health + 0.25)
            state.stamina = max(0.0, state.stamina - 0.2)
        elif selected in (Action.TURN_LEFT, Action.TURN_RIGHT):
            state.target_direction = float(np.clip(state.target_direction - turn * 0.25, -1.0, 1.0))
            reward += 0.15 * (1.0 - abs(state.target_direction))
        elif selected == Action.ADVANCE:
            state.enemy_distance = max(0.05, state.enemy_distance - 0.06)
            reward += 0.08
        elif selected == Action.RETREAT:
            state.enemy_distance = min(1.0, state.enemy_distance + 0.08)

        if state.enemy_attacking > 0.5 and selected != Action.DODGE:
            state.health = max(0.0, state.health - 0.08)
            reward -= 0.5
        state.enemy_attacking = float(self._rng.random() < 0.15 + 0.4 * (state.enemy_distance < 0.3))
        state.enemy_direction = float(np.clip(state.enemy_direction + self._rng.normal(0, 0.08), -1, 1))
        state.stamina = min(1.0, state.stamina + 0.04)
        self.steps += 1
        terminated = state.health <= 0.0
        truncated = self.steps >= self.max_steps
        return self._observation(), float(reward), terminated, truncated, {"action": selected.value}