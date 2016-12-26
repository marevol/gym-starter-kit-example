# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import, unicode_literals

import sys

import gymkit
from gymkit.agent import GymKitAgent


class RandomAgent(GymKitAgent):
    def __init__(self, env):
        super().__init__(env)

    def act(self, observation):
        return self.env.action_space.sample()


if __name__ == '__main__':
    sys.exit(gymkit.main(['--agent', 'random_agent.RandomAgent']))
