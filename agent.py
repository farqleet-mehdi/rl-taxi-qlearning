import numpy as np


class QLearningAgent:
    """Tabular Q-learning agent with epsilon-greedy exploration."""

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99,
                 eps_start=1.0, eps_end=0.05, eps_decay=0.999, seed=0):
        self.n_actions = n_actions
        self.alpha = alpha            # learning rate
        self.gamma = gamma            # discount factor
        self.epsilon = eps_start      # current exploration rate
        self.eps_end = eps_end        # minimum exploration rate
        self.eps_decay = eps_decay    # multiplied after every episode
        self.rng = np.random.default_rng(seed)
        self.q = np.zeros((n_states, n_actions))   # the Q-table

    def select_action(self, state, greedy=False):
        # explore: random action with probability epsilon
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        # exploit: best known action (ties broken randomly)
        best = np.flatnonzero(self.q[state] == self.q[state].max())
        return int(self.rng.choice(best))

    def update(self, s, a, r, s_next, terminated):
        # no bootstrapping from a terminal state
        target = r if terminated else r + self.gamma * self.q[s_next].max()
        self.q[s, a] += self.alpha * (target - self.q[s, a])

    def decay_epsilon(self):
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

    def save(self, path):
        np.save(path, self.q)

    def load(self, path):
        self.q = np.load(path)