import numpy as np

from flygym_demo.elden_ring_agent.perception import HudLayout, HudObservationProvider


def test_hud_observation_provider_reads_colored_bars():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    layout = HudLayout(health=(0.0, 0.0, 0.25, 0.1), stamina=(0.0, 0.1, 0.25, 0.2))
    frame[0:10, 0:50] = (220, 20, 20)
    frame[10:20, 0:25] = (20, 220, 20)

    observation = HudObservationProvider(layout)(frame)

    assert 0.9 <= observation.health <= 1.0
    assert 0.45 <= observation.stamina <= 0.55
    assert observation.enemy_visible is False


def test_hud_observation_provider_rejects_non_image_input():
    try:
        HudObservationProvider()(np.zeros((10, 10), dtype=np.uint8))
    except ValueError as error:
        assert "shape" in str(error)
    else:
        raise AssertionError("Expected invalid frame shape to be rejected")