# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import, unicode_literals

import logging
import sys

import gymkit
from gymkit.agent import GymKitAgent

import numpy as np


PLAYER_COLOR = [92, 186, 92]
ENEMY_COLOR = [213, 130, 74]
BALL_COLOR = [236, 236, 236]

logger = logging.getLogger()


class TrackAgent(GymKitAgent):

    def __init__(self, env):
        super().__init__(env)
        if env.spec.id != 'Pong-v0':
            raise ValueError('{} is not supported. Use Pong-v0.'.format(env.id))
        self._past_action = 0

    def act(self, observation):
        player, enemy, ball = self.observation_to_state(observation)
        action = self._past_action
        if len(player) == len(ball) == 2:
            if player[0] < ball[0]:
                action = 3
            elif player[0] > ball[0]:
                action = 2
            else:
                action = 0
            self._past_action = action

        return action

    def observation_to_state(self, observation):
        area = observation[35:194]

        player = self.search_position(area, PLAYER_COLOR)
        enemy = self.search_position(area, ENEMY_COLOR)
        ball = self.search_position(area, BALL_COLOR)

        logger.debug("player:{} enemy:{} ball:{}".format(player, enemy, ball))
        return player, enemy, ball

    def search_position(self, area, color):
        position = []
        index = np.where(area == color)
        if len(index[0]) > 0:
            position = [index[0][0], index[1][0]]
        return position

if __name__ == '__main__':
    sys.exit(gymkit.main(['--agent', 'pong.track_agent.TrackAgent', '--env-id', 'Pong-v0']))
