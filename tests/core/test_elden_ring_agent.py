import pytest

from flygym_demo.elden_ring_agent import (
    ActionCommand,
    FlyGameObservation,
    RuleBasedFlyPolicy,
)
from flygym_demo.elden_ring_agent.policy import Action


def test_policy_dodges_attacks_away_from_enemy():
    observation = FlyGameObservation(
        enemy_visible=True,
        enemy_attacking=True,
        enemy_direction=0.75,
        stamina=0.8,
    )

    assert RuleBasedFlyPolicy().decide(observation) == ActionCommand(
        Action.DODGE,
        turn=-0.75,
    )


def test_policy_attacks_close_enemy():
    observation = FlyGameObservation(
        enemy_visible=True,
        enemy_distance=0.1,
        enemy_direction=-0.5,
    )

    assert RuleBasedFlyPolicy().decide(observation) == ActionCommand(
        Action.ATTACK,
        turn=-0.5,
    )


def test_observation_rejects_out_of_range_values():
    with pytest.raises(ValueError, match="health"):
        FlyGameObservation(health=1.1)