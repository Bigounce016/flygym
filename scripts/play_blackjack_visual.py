"""Animate a fly actually playing blackjack.

The fly uses the repository's Blackjack Q-learning implementation to choose
HIT/STICK. This script then renders a live-looking card-table animation in
which the fly sits at the table and makes decisions across several hands.

Run with:

    python scripts/play_blackjack_visual.py --episodes 20000 --seed 0 --hands 5 --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse, FancyBboxPatch

from flygym_demo.blackjack_agent import BlackjackEnv, train
from flygym_demo.blackjack_agent.blackjack import HIT, STICK


def draw_card(ax, x, y, value: int, *, color: str = "black") -> None:
    card = FancyBboxPatch(
        (x - 0.45, y - 0.7),
        0.9,
        1.4,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        linewidth=1.5,
        edgecolor="black",
        facecolor="white",
    )
    ax.add_patch(card)

    suit = "♠" if value in (1, 10) else "♥"
    label = "A" if value == 1 else str(value)
    ax.text(x, y + 0.34, label, ha="center", va="center", fontsize=16, fontweight="bold", color=color)
    ax.text(x, y - 0.28, suit, ha="center", va="center", fontsize=16, color=color)


def draw_fly(ax, x: float, y: float, *, scale: float = 1.0) -> None:
    body = Ellipse((x, y), 0.8 * scale, 0.5 * scale, color="#7fdc5d", ec="black", lw=1.5)
    ax.add_patch(body)
    ax.add_patch(Ellipse((x - 0.45 * scale, y + 0.1 * scale), 0.55 * scale, 0.2 * scale, angle=20, color="#dff7ff", alpha=0.8, ec="black", lw=1))
    ax.add_patch(Ellipse((x + 0.45 * scale, y + 0.1 * scale), 0.55 * scale, 0.2 * scale, angle=-20, color="#dff7ff", alpha=0.8, ec="black", lw=1))
    ax.plot([x - 0.2 * scale, x - 0.9 * scale], [y - 0.1 * scale, y - 0.5 * scale], color="black", lw=1.5)
    ax.plot([x + 0.2 * scale, x + 0.9 * scale], [y - 0.1 * scale, y - 0.5 * scale], color="black", lw=1.5)
    ax.plot([x - 0.15 * scale, x - 0.5 * scale], [y + 0.1 * scale, y + 0.7 * scale], color="black", lw=1.5)
    ax.plot([x + 0.15 * scale, x + 0.5 * scale], [y + 0.1 * scale, y + 0.7 * scale], color="black", lw=1.5)
    ax.text(x, y - 0.12 * scale, "fly", ha="center", va="center", fontsize=9, color="black")


def simulate_hand(learner, env: BlackjackEnv) -> list[tuple[str, tuple[int, int, bool], list[int], int]]:
    state = env.reset()
    history: list[tuple[str, tuple[int, int, bool], list[int], int]] = []
    done = False
    while not done:
        action = learner.choose_action(state, explore=False)
        action_name = "HIT" if action == HIT else "STICK"
        history.append((action_name, state, list(env.player), env.dealer[0]))
        next_state, _, done = env.step(action)
        if next_state is not None:
            state = next_state
    return history


def render_animation(learner, output_path: Path, hands: int = 4, *, show: bool = False) -> None:
    env = BlackjackEnv()
    all_hands = [simulate_hand(learner, env) for _ in range(hands)]
    frames: list[tuple[str, int, list[int], int]] = []
    for hand in all_hands:
        for action_name, state, player_cards, dealer_up in hand:
            frames.append((action_name, state[0], player_cards, dealer_up))

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0d3b2f")
    ax.set_facecolor("#0d3b2f")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    table = FancyBboxPatch(
        (0.5, 0.5),
        9.0,
        5.0,
        boxstyle="round,pad=0.2,rounding_size=0.2",
        linewidth=2,
        edgecolor="#d9c58a",
        facecolor="#1f6b46",
    )

    def draw_scene(frame_idx: int) -> None:
        ax.clear()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis("off")
        ax.set_facecolor("#0d3b2f")
        ax.add_patch(table)
        ax.text(5, 5.55, "FLY BLACKJACK", ha="center", va="center", fontsize=22, fontweight="bold", color="white")

        action_name, player_total, player_cards, dealer_up = frames[frame_idx % len(frames)]
        ax.text(0.8, 4.8, f"Fly chooses {action_name} at {player_total} vs dealer {dealer_up}", ha="left", va="center", fontsize=12, color="#f4e7a1", fontweight="bold")
        draw_fly(ax, 8.2, 1.2, scale=1.4)

        draw_card(ax, 2.2, 2.3, dealer_up, color="red")
        draw_card(ax, 3.7, 2.3, 9, color="black")

        for i, card in enumerate(player_cards[:4]):
            draw_card(ax, 1.8 + i * 1.1, 1.0, card, color="black")

        if action_name == "HIT":
            ax.text(1.5, 0.7, "HIT!", fontsize=18, color="#ffd166", fontweight="bold")
        else:
            ax.text(1.4, 0.7, "STICK", fontsize=18, color="#8fe388", fontweight="bold")

    def _animate(frame_idx: int):
        draw_scene(frame_idx)
        return []

    anim = FuncAnimation(fig, _animate, frames=max(1, len(frames)), interval=700, blit=False)
    anim.save(output_path, writer="pillow", fps=2)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hands", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("blackjack_fly_playing.gif"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    learner = train(episodes=args.episodes, seed=args.seed)
    render_animation(learner, args.out, hands=args.hands, show=args.show)
    print(f"Saved fly blackjack animation to {args.out.resolve()}")


if __name__ == "__main__":
    main()
