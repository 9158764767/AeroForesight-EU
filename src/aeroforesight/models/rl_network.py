"""Reinforcement-learning network optimiser (RL component).

Problem: an airline network planner has a limited pool of *schedule buffer*
(extra minutes / gate slack) to distribute across congested European hubs each
morning. Too little buffer at a hot hub → cascading delays; too much everywhere
→ wasted aircraft utilisation. The agent learns *where* to invest the buffer.

Formulation (tabular Q-learning):
  state   : discretised congestion band (low/med/high) for each modelled hub
  action  : which hub receives the next buffer unit (or "hold")
  reward  : avoided delay-cost minus the utilisation cost of the buffer

Tabular Q-learning keeps it transparent and fast; the same interface would back
a DQN drop-in if the state space grew.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class HubEnv:
    """Lightweight simulator of buffer-allocation on an N-hub network."""

    n_hubs: int = 4
    buffer_units: int = 5              # units to place per episode
    delay_cost_eur: float = 1500.0     # cost of one congestion-hour
    buffer_cost_eur: float = 300.0     # cost of one buffer unit
    seed: int = 0
    _rng: np.random.Generator = field(default=None, repr=False)
    _congestion: np.ndarray = field(default=None, repr=False)
    _placed: int = 0

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    # congestion band per hub: 0 low, 1 med, 2 high
    def reset(self) -> tuple:
        self._congestion = self._rng.integers(0, 3, size=self.n_hubs)
        self._placed = 0
        self._buffers = np.zeros(self.n_hubs, dtype=int)
        return tuple(self._congestion.tolist())

    @property
    def n_actions(self) -> int:
        return self.n_hubs + 1  # +1 = "hold" (place nothing this step)

    def step(self, action: int):
        done = False
        reward = 0.0
        if action < self.n_hubs and self._placed < self.buffer_units:
            hub = action
            band = self._congestion[hub]
            # diminishing returns: first buffer unit at a hot hub avoids the most delay
            marginal = max(0, band * 1.0 - 0.5 * self._buffers[hub])
            reward += marginal * self.delay_cost_eur - self.buffer_cost_eur
            self._buffers[hub] += 1
            self._placed += 1
        else:
            # "hold": mild penalty so the agent doesn't dawdle when buffer is useful
            reward -= 50.0
        if self._placed >= self.buffer_units:
            done = True
        return tuple(self._congestion.tolist()), reward, done


class QLearningAgent:
    def __init__(self, n_actions: int, gamma: float = 0.95, alpha: float = 0.10, epsilon: float = 0.20, seed: int = 0):
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.q: dict[tuple, np.ndarray] = {}
        self._rng = np.random.default_rng(seed)
        self.history: list[float] = []

    def _row(self, s: tuple) -> np.ndarray:
        return self.q.setdefault(s, np.zeros(self.n_actions))

    def act(self, s: tuple, greedy: bool = False) -> int:
        if not greedy and self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.n_actions))
        return int(np.argmax(self._row(s)))

    def update(self, s, a, r, s2, done):
        row = self._row(s)
        target = r + (0 if done else self.gamma * float(np.max(self._row(s2))))
        row[a] += self.alpha * (target - row[a])


def train_agent(env: HubEnv, episodes: int = 400, **agent_kwargs) -> QLearningAgent:
    agent = QLearningAgent(env.n_actions, **agent_kwargs)
    for _ in range(episodes):
        s = env.reset()
        total, done = 0.0, False
        while not done:
            a = agent.act(s)
            s2, r, done = env.step(a)
            agent.update(s, a, r, s2, done)
            s, total = s2, total + r
        agent.history.append(total)
    return agent


def recommend_allocation(agent: QLearningAgent, congestion_bands: list[int]) -> dict:
    """Greedy roll-out: given today's hub congestion, where should buffer go?"""
    env = HubEnv(n_hubs=len(congestion_bands))
    s = tuple(int(b) for b in congestion_bands)
    env._congestion = np.array(s)
    env._placed = 0
    env._buffers = np.zeros(env.n_hubs, dtype=int)
    plan, total = [], 0.0
    done = False
    while not done:
        a = agent.act(s, greedy=True)
        _, r, done = env.step(a)
        total += r
        plan.append("hold" if a == env.n_hubs else f"hub_{a}")
    return {"allocation": plan, "expected_net_benefit_eur": round(total, 2), "buffers": env._buffers.tolist()}
