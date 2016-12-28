# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import, unicode_literals

from collections import deque
import logging

from gymkit.agent import GymKitAgent

import numpy as np


logger = logging.getLogger('q_agent')


class QLearning(GymKitAgent):
    def __init__(self, env, epsilon=0.1, alpha=0.1, gamma=0.9, epsilon_decay=None, alpha_decay=None):
        super().__init__(env)
        self.n_actions = env.action_space.n

        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.alpha_decay = alpha_decay

        self.q_table = {}
        self.state = None
        self.episode = 0
        self.steps = deque(maxlen=100)

    def act(self, observation):
        if self.state is None:
            self.state = self.observation_to_state(observation)
        if np.random.random() < self.epsilon:
            self.action = np.random.choice(self.n_actions)
        else:
            actions = self.get_actions(self.state)
            self.action = np.argmax(actions)
        return self.action

    def fit(self, observation, reward, done, info):
        previous_actions = self.get_actions(self.state)
        current_state = self.observation_to_state(observation)
        current_actions = self.get_actions(current_state)
        future = 0 if done else np.max(current_actions)
        value = previous_actions[self.action]
        previous_actions[self.action] += self.alpha * (reward + self.gamma * future - value)

        self.state = current_state
        self.values.append(value)
        self.step += 1

    def __enter__(self):
        self.episode += 1
        self.step = 0
        self.values = []
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.steps.append(self.step)
        mean_value = np.mean(self.values)
        mean_step = np.mean(self.steps)
        logger.info("{} steps(avg {}), epsilon={:.2f}, alpha={:.2f}, q={:.2f}".format(self.step,
                                                                                      mean_step,
                                                                                      self.epsilon,
                                                                                      self.alpha,
                                                                                      mean_value))

        if self.epsilon_decay is not None:
            self.epsilon = self.epsilon_decay(self.epsilon, self.episode)
        if self.alpha_decay is not None:
            self.alpha = self.alpha_decay(self.alpha, self.episode)

    def get_actions(self, state):
        actions = self.q_table.get(state)
        if actions is None:
            actions = np.zeros(self.n_actions)
            self.q_table[state] = actions
        return actions

    def observation_to_state(self, observation):
        state = ''
        bins = self.get_state_bins()
        for d, o in enumerate(observation.flatten()):
            state += str(np.digitize(o, bins[d]))
        return state

    def get_state_bins(self, bin_sizes=None, low_bound=None, high_bound=None):
        if hasattr(self, '_dimension_bins'):
            return self._dimension_bins

        self._dimension_bins = []
        lows = self.env.observation_space.low
        highs = self.env.observation_space.high
        for i in range(len(lows)):
            low = lows[i]
            if low_bound is not None:
                _low_bound = low_bound if not isinstance(low_bound, list) else low_bound[i]
                low = low if _low_bound is None else max(low, _low_bound)

            high = highs[i]
            if high_bound is not None:
                _high_bound = high_bound if not isinstance(high_bound, list) else high_bound[i]
                high = high if _high_bound is None else min(high, _high_bound)

            r = (float(high) - float(low)) / (bin_sizes[i] - 1)
            bins = np.arange(low + r / 2, high, r)
            if min(bins) < 0 and 0 not in bins:
                bins = np.sort(np.append(bins, [0]))
            self._dimension_bins.append(bins)

        return self._dimension_bins
