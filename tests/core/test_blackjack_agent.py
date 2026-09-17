import random

import pytest

from flygym_demo.blackjack_agent import BlackjackEnv, BlackjackQLearner
from flygym_demo.blackjack_agent.blackjack import HIT, STICK, hand_value


def test_hand_value_prefers_usable_ace_without_busting():
    assert hand_value([1, 6]) == (17, True)
    assert hand_value([1, 9, 10]) == (20, False)


def test_environment_returns_valid_state_after_reset():
    state = BlackjackEnv(rng=random.Random(3)).reset()

    assert len(state) == 3
    assert 4 <= state[0] <= 21
    assert 1 <= state[1] <= 10
    assert isinstance(state[2], bool)


def test_learner_updates_terminal_action():
    learner = BlackjackQLearner(rng=random.Random(0))
    state = (16, 10, False)

    learner.update(state, STICK, -1.0, None, True)

    assert learner.q_values[state][STICK] < 0.0


def test_invalid_action_is_rejected():
    environment = BlackjackEnv(rng=random.Random(0))
    environment.reset()

    with pytest.raises(ValueError):
        environment.step(99)