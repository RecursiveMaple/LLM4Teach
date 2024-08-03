import posggym as gym
import numpy as np
from .format import flatten_tuple

def make_env_fn(env_key, render_mode=None, frame_stack=1):
    def _f():
        print("env_key", env_key)
        env = gym.make(env_key, render_mode=render_mode)
        # if frame_stack > 1:
        #     env = FrameStack(env, frame_stack)
        return env
    return _f


# Gives a vectorized interface to a single environment
class WrapEnv:
    def __init__(self, env_fn):
        self.env = env_fn()

    def __getattr__(self, attr):
        return getattr(self.env, attr)

    def step(self, actions):
        # assert action.ndim == 1
        env_return = self.env.step(actions)
        observations, rewards, terminations, truncations, all_done, infos = env_return
        observations = {k:np.array(tuple(flatten_tuple(v))) for k,v in observations.items()}
        return observations, rewards, terminations, truncations, all_done, infos

    def render(self):
        self.env.render()

    def reset(self, seed=None):
        observations, infos = self.env.reset(seed=seed)
        observations = {k:np.array(tuple(flatten_tuple(v))) for k,v in observations.items()}
        return observations, infos
        