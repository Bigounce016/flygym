"""Video dataset and behavior-cloning helpers for the Elden Ring demo.

Videos must be paired with action intervals. Pixels alone do not reveal which
controller action the player intended, so the manifest is the training label.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import imageio.v3 as iio
import numpy as np

from .policy import Action, ActionCommand


@dataclass(frozen=True, slots=True)
class ActionInterval:
    start: float
    end: float
    action: Action
    turn: float = 0.0

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("action interval must have 0 <= start < end")
        if not -1.0 <= self.turn <= 1.0:
            raise ValueError("turn must be between -1 and 1")


@dataclass(frozen=True, slots=True)
class VideoRecording:
    video: Path
    fps: float
    intervals: tuple[ActionInterval, ...]


def load_recordings(manifest_path: str | Path) -> list[VideoRecording]:
    """Load a JSON manifest containing videos and timestamped action labels."""
    manifest_path = Path(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    recordings = []
    for item in payload["recordings"]:
        video = Path(item["video"])
        if not video.is_absolute():
            video = manifest_path.parent / video
        intervals = tuple(
            ActionInterval(
                start=float(segment["start"]),
                end=float(segment["end"]),
                action=Action(segment["action"]),
                turn=float(segment.get("turn", 0.0)),
            )
            for segment in item["actions"]
        )
        recordings.append(VideoRecording(video, float(item["fps"]), intervals))
    return recordings


def _label_at(intervals: tuple[ActionInterval, ...], timestamp: float) -> ActionCommand | None:
    for interval in intervals:
        if interval.start <= timestamp < interval.end:
            return ActionCommand(interval.action, interval.turn)
    return None


def iter_labeled_frames(
    recordings: list[VideoRecording], *, stride: int = 1
) -> Iterator[tuple[np.ndarray, ActionCommand]]:
    """Yield RGB video frames and their nearest timestamped action labels."""
    if stride < 1:
        raise ValueError("stride must be at least 1")
    for recording in recordings:
        for frame_index, frame in enumerate(iio.imiter(recording.video)):
            if frame_index % stride:
                continue
            label = _label_at(recording.intervals, frame_index / recording.fps)
            if label is not None:
                yield np.asarray(frame)[..., :3], label


def build_dataset(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    image_size: tuple[int, int] = (96, 96),
    stride: int = 4,
) -> int:
    """Extract labeled frames into a compact ``.npz`` dataset."""
    from PIL import Image

    frames: list[np.ndarray] = []
    actions: list[int] = []
    turns: list[float] = []
    action_values = list(Action)
    for frame, command in iter_labeled_frames(load_recordings(manifest_path), stride=stride):
        image = Image.fromarray(frame).resize((image_size[1], image_size[0]))
        frames.append(np.asarray(image, dtype=np.uint8))
        actions.append(action_values.index(command.action))
        turns.append(command.turn)
    if not frames:
        raise ValueError("manifest produced no labeled frames")
    np.savez_compressed(
        output_path,
        frames=np.stack(frames),
        actions=np.asarray(actions, dtype=np.int64),
        turns=np.asarray(turns, dtype=np.float32),
        action_names=np.asarray([action.value for action in action_values]),
    )
    return len(frames)


class LearnedVideoPolicy:
    """Load a trained PyTorch frame classifier as a live ActionCommand policy."""

    def __init__(self, model_path: str | Path, *, device: str = "cpu") -> None:
        try:
            import torch
            from .train_video_policy import FramePolicy
        except ImportError as error:
            raise RuntimeError(
                "LearnedVideoPolicy requires PyTorch; install the training extra."
            ) from error
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        self._torch = torch
        self._device = torch.device(device)
        self._model = FramePolicy(len(checkpoint["action_names"])).to(self._device)
        self._model.load_state_dict(checkpoint["state_dict"])
        self._model.eval()
        self._actions = [Action(name) for name in checkpoint["action_names"]]

    def __call__(self, frame: np.ndarray) -> ActionCommand:
        image = self._torch.from_numpy(frame[..., :3]).float().permute(2, 0, 1) / 255.0
        image = image.unsqueeze(0).to(self._device)
        with self._torch.no_grad():
            logits, turn = self._model(image)
        action = self._actions[int(logits.argmax(dim=1).item())]
        return ActionCommand(action, turn=float(turn.squeeze().clamp(-1, 1).item()))