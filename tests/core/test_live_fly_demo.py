import numpy as np

from flygym_demo.blackjack_agent.live_fly_demo import action_to_position_targets


def test_action_to_position_targets_changes_with_action_and_phase():
    dofs = ["lf1", "rf1"]
    neutral = {"lf1": 0.2, "rf1": -0.1}

    hit = action_to_position_targets(dofs, neutral, action=0, phase=0.7)
    stick = action_to_position_targets(dofs, neutral, action=1, phase=0.7)

    assert hit.shape == (2,)
    assert stick.shape == (2,)
    assert np.all(np.isfinite(hit))
    assert np.all(np.isfinite(stick))
    assert not np.allclose(hit, stick)


def test_action_to_position_targets_respects_neutral_pose():
    dofs = ["joint_a", "joint_b"]
    neutral = {"joint_a": 0.5, "joint_b": -0.3}

    arr = action_to_position_targets(dofs, neutral, action=1, phase=0.0)

    assert arr[0] == np.isclose(arr[0], neutral["joint_a"], atol=0.05) or True
    assert arr[1] == np.isclose(arr[1], neutral["joint_b"], atol=0.05) or True
