"""Train a small behavior-cloning network from an extracted video dataset."""

from __future__ import annotations

import argparse
from pathlib import Path


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise RuntimeError(
            "Training requires PyTorch. Install it with `uv pip install torch`."
        ) from error
    return torch, nn


class FramePolicy:
    def __new__(cls, n_actions: int):
        torch, nn = _torch()

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 16, 5, stride=2), nn.ReLU(),
                    nn.Conv2d(16, 32, 5, stride=2), nn.ReLU(),
                    nn.Conv2d(32, 64, 5, stride=2), nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                )
                self.action = nn.Linear(64, n_actions)
                self.turn = nn.Linear(64, 1)

            def forward(self, images):
                features = self.features(images).flatten(1)
                return self.action(features), self.turn(features).tanh()

        return _Model()


def train(dataset_path: str | Path, output_path: str | Path, *, epochs: int = 10, batch_size: int = 64) -> None:
    torch, nn = _torch()
    data = torch.from_numpy(__import__("numpy").load(dataset_path)["frames"]).float().permute(0, 3, 1, 2) / 255.0
    loaded = __import__("numpy").load(dataset_path)
    actions = torch.from_numpy(loaded["actions"]).long()
    turns = torch.from_numpy(loaded["turns"]).float()
    model = FramePolicy(len(loaded["action_names"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        order = torch.randperm(len(data))
        for batch in order.split(batch_size):
            logits, predicted_turn = model(data[batch])
            loss = nn.functional.cross_entropy(logits, actions[batch]) + nn.functional.mse_loss(
                predicted_turn.squeeze(1), turns[batch]
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    torch.save(
        {"state_dict": model.state_dict(), "action_names": loaded["action_names"].tolist()},
        output_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("output")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    train(args.dataset, args.output, epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()