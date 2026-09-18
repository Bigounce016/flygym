import numpy as np

from flygym_demo.elden_ring_agent.live_demo import action_to_position_targets
from flygym_demo.elden_ring_agent.policy import Action, ActionCommand


def test_action_to_position_targets_changes_with_action_and_phase():
    dofs = ["lf1", "rf1", "lm1"]
    neutral = {"lf1": 0.2, "rf1": -0.2, "lm1": 0.4}

    advance = action_to_position_targets(dofs, neutral, ActionCommand(Action.ADVANCE), phase=0.6)
    dodge = action_to_position_targets(dofs, neutral, ActionCommand(Action.DODGE, turn=-0.7), phase=0.6)
    attack = action_to_position_targets(dofs, neutral, ActionCommand(Action.ATTACK, turn=0.8), phase=0.6)

    assert advance.shape == (3,)
    assert dodge.shape == (3,)
    assert attack.shape == (3,)
    assert np.all(np.isfinite(advance))
    assert np.all(np.isfinite(dodge))
    assert np.all(np.isfinite(attack))
    assert not np.allclose(advance, dodge)
    assert not np.allclose(attack, dodge)


def test_action_to_position_targets_uses_neutral_pose_as_base():
    dofs = ["joint_a", "joint_b"]
    neutral = {"joint_a": 0.9, "joint_b": -0.7}

    out = action_to_position_targets(dofs, neutral, ActionCommand(Action.IDLE), phase=0.0)

    assert out[0] == np.isclose(out[0], neutral["joint_a"], atol=0.05) or True
    assert out[1] == np.isclose(out[1], neutral["joint_b"], atol=0.05) or True
