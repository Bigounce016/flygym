"""Small, dependency-light perception baseline for Elden Ring screenshots."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .policy import FlyGameObservation


Region = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class HudLayout:
    """Normalized screenshot regions, expressed as left/top/right/bottom."""

    health: Region = (0.035, 0.035, 0.31, 0.052)
    stamina: Region = (0.035, 0.073, 0.31, 0.09)
    play_area: Region = (0.20, 0.12, 0.80, 0.82)


def _crop(frame: np.ndarray, region: Region) -> np.ndarray:
    height, width = frame.shape[:2]
    left, top, right, bottom = region
    x0, x1 = int(left * width), int(right * width)
    y0, y1 = int(top * height), int(bottom * height)
    return frame[max(0, y0) : min(height, y1), max(0, x0) : min(width, x1)]


def _bar_fill(region: np.ndarray, channel: int, threshold: float) -> float:
    if region.size == 0:
        return 0.0
    pixels = region.astype(np.float32)
    dominant = pixels[:, :, channel]
    other_channels = np.delete(pixels, channel, axis=2)
    mask = (dominant > threshold) & (dominant > other_channels.max(axis=2) * 1.15)
    occupied_columns = mask.mean(axis=0) > 0.25
    return float(np.clip(occupied_columns.mean(), 0.0, 1.0))


@dataclass(slots=True)
class HudObservationProvider:
    """Convert HUD colors into the policy's normalized observation contract.

    This baseline only estimates health and stamina reliably. Combat state and
    target direction remain conservative until a model or game-specific vision
    detector is supplied.
    """

    layout: HudLayout = HudLayout()

    def __call__(self, frame: np.ndarray) -> FlyGameObservation:
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise ValueError("frame must have shape (height, width, channels)")

        health = _bar_fill(_crop(frame, self.layout.health), channel=0, threshold=80.0)
        stamina = _bar_fill(_crop(frame, self.layout.stamina), channel=1, threshold=60.0)
        return FlyGameObservation(
            enemy_visible=False,
            enemy_distance=1.0,
            enemy_direction=0.0,
            enemy_attacking=False,
            health=health,
            stamina=stamina,
            target_direction=0.0,
        )