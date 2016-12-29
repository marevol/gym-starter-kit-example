# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import, unicode_literals

import sys

import gymkit

from general import QLearning
import math


class QAgent(QLearning):
    def __init__(self, env):
        super().__init__(env, epsilon=1.0, alpha=0.5, gamma=0.99, max_step=250,
                         alpha_decay=lambda a, t: max(0.1, min(0.5, 1.0 - math.log10((t + 1) / 25))),
                         epsilon_decay=lambda eps, t: max(0.01, min(1.0, 1.0 - math.log10((t + 1) / 25))))

    def get_state_bins(self):
        return super().get_state_bins(bin_sizes=[2, 2, 7, 4],
                                      low_bound=[None, -0.5, None, -math.radians(50)],
                                      high_bound=[None, 0.5, None, math.radians(50)])

if __name__ == '__main__':
    sys.exit(gymkit.main(['--agent', 'cartpole.q_agent.QAgent',
                          '--env-id', 'CartPole-v0',
                          '--try-count', '1000']))
