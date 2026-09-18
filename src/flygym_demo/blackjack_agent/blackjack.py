"""Tabular Q-learning for a simple, resettable Blackjack environment."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import random

State = tuple[int, int, bool]
Action = int
HIT: Action = 0
STICK: Action = 1
ACTIONS = (HIT, STICK)


def hand_value(hand: list[int]) -> tuple[int, bool]:
    """Return the best hand value not exceeding 21, if possible, and ace usage."""
    value = sum(hand)
    aces = hand.count(1)
    usable_ace = aces > 0 and value + 10 <= 21
    if usable_ace:
        value += 10
    return value, usable_ace


@dataclass
class BlackjackEnv:
    """A compact infinite-deck Blackjack environment.

    Cards are sampled with replacement. Face cards are represented by 10 and
    aces by 1. Rewards are +1 for a win, 0 for a draw, and -1 for a loss.
    """

    rng: random.Random | None = None
    player: list[int] = field(default_factory=list)
    dealer: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.rng = self.rng or random.Random()

    def _draw(self) -> int:
        return self.rng.randint(1, 10)

    def _state(self) -> State:
        player_value, usable_ace = hand_value(self.player)
        return player_value, self.dealer[0], usable_ace

    def reset(self) -> State:
        self.player = [self._draw(), self._draw()]
        self.dealer = [self._draw(), self._draw()]
        return self._state()

    def step(self, action: Action) -> tuple[State | None, float, bool]:
        if action not in ACTIONS:
            raise ValueError(f"action must be one of {ACTIONS}")

        if action == HIT:
            self.player.append(self._draw())
            player_value, _ = hand_value(self.player)
            if player_value > 21:
                return None, -1.0, True
            return self._state(), 0.0, False

        while hand_value(self.dealer)[0] < 17:
            self.dealer.append(self._draw())

        player_value, _ = hand_value(self.player)
        dealer_value, _ = hand_value(self.dealer)
        if dealer_value > 21 or player_value > dealer_value:
            reward = 1.0
        elif player_value == dealer_value:
            reward = 0.0
        else:
            reward = -1.0
        return None, reward, True


class BlackjackQLearner:
    """Epsilon-greedy tabular Q-learning agent."""

    def __init__(
        self,
        *,
        learning_rate: float = 0.1,
        discount: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.9995,
        minimum_epsilon: float = 0.05,
        rng: random.Random | None = None,
    ) -> None:
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.minimum_epsilon = minimum_epsilon
        self.rng = rng or random.Random()
        self.q_values: defaultdict[State, list[float]] = defaultdict(lambda: [0.0, 0.0])

    def choose_action(self, state: State, *, explore: bool = True) -> Action:
        if explore and self.rng.random() < self.epsilon:
            return self.rng.choice(ACTIONS)
        values = self.q_values[state]
        return HIT if values[HIT] >= values[STICK] else STICK

    def update(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: State | None,
        done: bool,
    ) -> None:
        future = 0.0 if done else max(self.q_values[next_state])
        target = reward + self.discount * future
        current = self.q_values[state][action]
        self.q_values[state][action] += self.learning_rate * (target - current)

    def learn_episode(self, environment: BlackjackEnv) -> float:
        state = environment.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = self.choose_action(state)
            next_state, reward, done = environment.step(action)
            self.update(state, action, reward, next_state, done)
            total_reward += reward
            if next_state is not None:
                state = next_state
        self.epsilon = max(self.minimum_epsilon, self.epsilon * self.epsilon_decay)
        return total_reward


def train(episodes: int = 100_000, seed: int | None = 0) -> BlackjackQLearner:
    """Train and return a learner using reproducible randomness by default."""
    rng = random.Random(seed)
    environment = BlackjackEnv(rng=random.Random(rng.random()))
    learner = BlackjackQLearner(rng=random.Random(rng.random()))
    for _ in range(episodes):
        learner.learn_episode(environment)
    return learner