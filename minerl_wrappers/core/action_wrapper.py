import gym

from gym.spaces import Box


class MineRLActionTransformationWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        if isinstance(self.action_space, gym.spaces.Dict):
            self.old_vec_space = self.action_space.spaces["vector"]
            self.vec_space = self.transform_vec_space(self.old_vec_space)
            self.action_space = gym.spaces.Dict({"vector": self.vec_space})
        else:
            assert isinstance(self.action_space, Box)
            self.old_vec_space = self.action_space
            self.vec_space = self.transform_vec_space(self.old_vec_space)
            self.action_space = self.vec_space

    def action(self, action):
        if isinstance(action, dict):
            return {"vector": self.transform_vec(action["vector"])}
        else:
            return self.transform_vec(action)

    def reverse_action(self, action):
        if isinstance(action, dict):
            return {"vector": self.reverse_transform_vec(action["vector"])}
        else:
            return self.reverse_transform_vec(action)

    def transform_vec_space(self, vec_space: Box) -> Box:
        raise NotImplementedError

    def transform_vec(self, vec):
        raise NotImplementedError

    def reverse_transform_vec(self, vec):
        raise NotImplementedError


class MineRLFlattenActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = self.action_space["vector"]

    def action(self, action):
        return dict(vector=action)

    def reverse_action(self, action):
        return action["vector"]
