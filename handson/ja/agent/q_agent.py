# -*- coding: utf-8 -*-

from collections import deque
import logging

from gymkit.agent import GymKitAgent

import numpy as np


logger = logging.getLogger('q_agent')


class QAgent(GymKitAgent):
    def __init__(self, env):
        super().__init__(env)
        self.n_actions = env.action_space.n

        self.epsilon = 0.1
        self.alpha = 0.1
        self.gamma = 0.9

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
        # Update Q(s, a)

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

    def get_actions(self, state):
        actions = self.q_table.get(state)
        if actions is None:
            actions = np.zeros(self.n_actions)
            self.q_table[state] = actions
        return actions

    def observation_to_state(self, observation):
        pass

    def get_state_bins(self, bin_sizes=[2, 2, 7, 4]):
        pass

