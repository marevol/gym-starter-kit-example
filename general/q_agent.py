# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import, unicode_literals

import logging
import math
import sys

import gymkit
from gymkit.agent import GymKitAgent

import numpy as np


logger = logging.getLogger('q_agent')


class QAgent(GymKitAgent):
    def __init__(self, env):
        super().__init__(env)
        self.n_actions = env.action_space.n
        self._observation_dimension = 1
        for d in env.observation_space.shape:
            self._observation_dimension *= d

        self.epsilon = 0.05
        self.lr = 0.5
        self.gamma = 0.99
        bin_size = [10, 10, 10, 10]
        low_bound = [None, -0.5, None, -math.radians(50)]
        high_bound = [None, 0.5, None, math.radians(50)]

        self._bin_sizes = bin_size if isinstance(bin_size, list) else [bin_size] * self._observation_dimension
        self._dimension_bins = []
        for i, low, high in self._low_high_iter(env.observation_space, low_bound, high_bound):
            b_size = self._bin_sizes[i]
            bins = self._make_bins(low, high, b_size)
            self._dimension_bins.append(bins)

        self.q_table = {}
        self.state = None

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
        previous_state = self.state
        current_state = self.observation_to_state(observation)
        previous_actions = self.get_actions(previous_state)
        current_actions = self.get_actions(current_state)
        future = 0 if done else np.max(current_actions)
        value = previous_actions[self.action]
        previous_actions[self.action] += self.lr * (reward + self.gamma * future - value)
        self.state = current_state
        self.values.append(value)
        self.step += 1

    def __enter__(self):
        self.step = 0
        self.values = []
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        mean = np.mean(self.values)
        logger.info("Step {}, Mean Q value={:.2f}".format(self.step, mean))

    def get_actions(self, state):
        actions = self.q_table.get(state)
        if actions is None:
            actions = np.zeros(self.n_actions)
            actions += 1 / self.n_actions
            self.q_table[state] = actions
        return actions

    def observation_to_state(self, observation):
        state = 0
        unit = max(self._bin_sizes)
        for d, o in enumerate(observation.flatten()):
            state = state + np.digitize(o, self._dimension_bins[d]) * pow(unit, d)
        return state

    @classmethod
    def _make_bins(cls, low, high, bin_size):
        r = (float(high) - float(low)) / (bin_size - 1)
        bins = np.arange(low + r / 2, high, r)
        return bins

    @classmethod
    def _low_high_iter(cls, observation_space, low_bound, high_bound):
        lows = observation_space.low
        highs = observation_space.high
        for i in range(len(lows)):
            low = lows[i]
            if low_bound is not None:
                _low_bound = low_bound if not isinstance(low_bound, list) else low_bound[i]
                low = low if _low_bound is None else max(low, _low_bound)

            high = highs[i]
            if high_bound is not None:
                _high_bound = high_bound if not isinstance(high_bound, list) else high_bound[i]
                high = high if _high_bound is None else min(high, _high_bound)

            yield i, low, high

if __name__ == '__main__':
    sys.exit(gymkit.main(['--agent', 'general.q_agent.QAgent']))
