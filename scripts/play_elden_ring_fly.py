"""Run the screen-driven Elden Ring FlyGym controller on Windows."""

from __future__ import annotations

import argparse

from flygym_demo.elden_ring_agent.io import ScreenCapturer, VirtualXboxController
from flygym_demo.elden_ring_agent.perception import HudObservationProvider


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
    args = parser.parse_args()

    from flygym_demo.elden_ring_agent.live_demo import run_live_elden_ring_demo

    run_live_elden_ring_demo(
        render=not args.no_render,
        sleep_time=args.sleep,
        frame_source=ScreenCapturer(region=tuple(args.region)),
        observation_provider=HudObservationProvider(),
        controller=VirtualXboxController(),
    )


if __name__ == "__main__":
    main()