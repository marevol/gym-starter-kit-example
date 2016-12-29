# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import, unicode_literals

import sys

import gymkit
from gymkit.agent import GymKitAgent


class SwingAgent(GymKitAgent):
    def __init__(self, env, interval=5):
        super().__init__(env)
        if env.spec.id != 'Pong-v0':
            raise ValueError('{} is not supported. Use Pong-v0.'.format(env.id))
        self._interval = interval
        self._plan = []
        self.init_plan()

    def act(self, observation):
        if len(self._plan) == 0:
            self.init_plan()
        return self._plan.pop(0)

    def init_plan(self):
        self._plan += [2] * self._interval
        self._plan += [3] * self._interval
        self._plan += [3] * self._interval
        self._plan += [2] * self._interval

if __name__ == '__main__':
    sys.exit(gymkit.main(['--agent', 'pong.swing_agent.SwingAgent', '--env-id', 'Pong-v0']))
