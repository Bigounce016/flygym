"""Train and inspect the tabular Blackjack agent.

Run with::

    python scripts/train_blackjack.py --episodes 100000
"""

from __future__ import annotations

import argparse

from flygym_demo.blackjack_agent import BlackjackEnv, train
from flygym_demo.blackjack_agent.blackjack import HIT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    learner = train(episodes=args.episodes, seed=args.seed)
    print(f"trained {args.episodes} episodes; epsilon={learner.epsilon:.3f}")
    print("state (player total, dealer card, usable ace) -> action")
    for state in ((12, 10, False), (16, 10, False), (20, 6, False), (18, 6, True)):
        action = "hit" if learner.choose_action(state, explore=False) == HIT else "stick"
        print(f"{state} -> {action}")

    environment = BlackjackEnv()
    wins = 0.0
    evaluation_hands = 1_000
    for _ in range(evaluation_hands):
        state = environment.reset()
        done = False
        reward = 0.0
        while not done:
            action = learner.choose_action(state, explore=False)
            next_state, reward, done = environment.step(action)
            if next_state is not None:
                state = next_state
        wins += reward
    print(f"evaluation average reward: {wins / evaluation_hands:.3f}")


if __name__ == "__main__":
    main()