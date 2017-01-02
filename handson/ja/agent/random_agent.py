from gymkit.agent import GymKitAgent


class RandomAgent(GymKitAgent):
    def __init__(self, env):
        super().__init__(env)

    def act(self, observation):
        return None

