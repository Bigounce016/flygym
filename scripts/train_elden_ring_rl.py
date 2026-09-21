"""Train an Elden Ring action policy with PPO on the combat proxy."""

from __future__ import annotations

import argparse
from pathlib import Path

from flygym_demo.elden_ring_agent.reinforcement import EldenRingCombatEnv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--output", type=Path, default=Path("runs/elden_ring_ppo"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
    except ImportError as error:
        raise SystemExit(
            "Install RL dependencies first: uv pip install 'flygym[rl]'"
        ) from error

    args.output.parent.mkdir(parents=True, exist_ok=True)
    env = EldenRingCombatEnv(seed=args.seed)
    model = PPO("MlpPolicy", env, verbose=1, seed=args.seed, tensorboard_log=str(args.output / "tb"))
    model.learn(total_timesteps=args.total_timesteps)
    model.save(args.output)
    env.close()
    print(f"Saved PPO policy to {args.output}.zip")


if __name__ == "__main__":
    main()