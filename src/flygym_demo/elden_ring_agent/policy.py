"""High-level game policy that can drive a FlyGym locomotion controller.

This module deliberately does not capture screens or inject game input.  It
defines the small, testable contract between perception, decision-making, and
the fly controller.  A screen reader or trained policy can replace
``RuleBasedFlyPolicy`` without changing the rest of an integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Action(str, Enum):
    """High-level actions exposed to a game-input adapter."""

    IDLE = "idle"
    ADVANCE = "advance"
    RETREAT = "retreat"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    DODGE = "dodge"
    ATTACK = "attack"
    HEAL = "heal"


@dataclass(frozen=True, slots=True)
class FlyGameObservation:
    """Perception available to the decision policy.

    Values are normalized to the range [-1, 1] where appropriate.  A vision
    adapter should produce this object from a frame, simulator state, or a
    human-labelled recording.
    """

    enemy_visible: bool = False
    enemy_distance: float = 1.0
    enemy_direction: float = 0.0
    enemy_attacking: bool = False
    health: float = 1.0
    stamina: float = 1.0
    target_direction: float = 0.0

    def __post_init__(self) -> None:
        for name in ("enemy_distance", "health", "stamina"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        for name in ("enemy_direction", "target_direction"):
            value = getattr(self, name)
            if not -1.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between -1 and 1")


@dataclass(frozen=True, slots=True)
class ActionCommand:
    """One high-level command for the fly/game adapter."""

    action: Action
    turn: float = 0.0

    def __post_init__(self) -> None:
        if not -1.0 <= self.turn <= 1.0:
            raise ValueError("turn must be between -1 and 1")


class RuleBasedFlyPolicy:
    """Small baseline policy for validating the full integration loop.

    This is intentionally not presented as a learned or biological brain. It
    provides predictable behavior while perception and training infrastructure
    are being developed.
    """

    def decide(self, observation: FlyGameObservation) -> ActionCommand:
        if observation.health < 0.2 and observation.stamina > 0.1:
            return ActionCommand(Action.HEAL)

        if observation.enemy_attacking and observation.stamina > 0.2:
            turn = -observation.enemy_direction
            return ActionCommand(Action.DODGE, turn=turn)

        if observation.enemy_visible and observation.enemy_distance < 0.2:
            return ActionCommand(Action.ATTACK, turn=observation.enemy_direction)

        if abs(observation.target_direction) > 0.2:
            action = Action.TURN_RIGHT if observation.target_direction > 0 else Action.TURN_LEFT
            return ActionCommand(action, turn=observation.target_direction)

        return ActionCommand(Action.ADVANCE)