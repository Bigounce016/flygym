"""Prepare labeled Elden Ring videos or train a frame policy from them."""

from __future__ import annotations

import argparse

from flygym_demo.elden_ring_agent.video_learning import build_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("manifest")
    prepare.add_argument("output")
    prepare.add_argument("--stride", type=int, default=4)
    train = subparsers.add_parser("train")
    train.add_argument("dataset")
    train.add_argument("output")
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.command == "prepare":
        count = build_dataset(args.manifest, args.output, stride=args.stride)
        print(f"Wrote {count} labeled frames to {args.output}")
    else:
        from flygym_demo.elden_ring_agent.train_video_policy import train

        train(args.dataset, args.output, epochs=args.epochs, batch_size=args.batch_size)
        print(f"Wrote trained policy to {args.output}")


if __name__ == "__main__":
    main()