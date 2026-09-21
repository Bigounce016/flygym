"""Run the screen-driven Elden Ring FlyGym controller on Windows."""

from __future__ import annotations

import argparse

from flygym_demo.elden_ring_agent.io import ScreenCapturer, VirtualXboxController
from flygym_demo.elden_ring_agent.perception import HudObservationProvider
from flygym_demo.elden_ring_agent.video_learning import LearnedVideoPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--region",
        type=int,
        nargs=4,
        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
        required=True,
        help="Elden Ring screen bounds in desktop pixels",
    )
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.08)
    parser.add_argument(
        "--model",
        help="Optional behavior-cloning checkpoint trained from labeled videos.",
    )
    args = parser.parse_args()

    from flygym_demo.elden_ring_agent.live_demo import run_live_elden_ring_demo

    action_provider = LearnedVideoPolicy(args.model) if args.model else None
    run_live_elden_ring_demo(
        render=not args.no_render,
        sleep_time=args.sleep,
        frame_source=ScreenCapturer(region=tuple(args.region)),
        observation_provider=None if action_provider else HudObservationProvider(),
        controller=VirtualXboxController(),
        action_provider=action_provider,
    )


if __name__ == "__main__":
    main()