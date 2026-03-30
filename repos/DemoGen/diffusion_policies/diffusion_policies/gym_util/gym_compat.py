import types

import numpy as np

try:
    import gym  # type: ignore
    from gym import spaces  # type: ignore
except ImportError:
    class Env:
        metadata = {}

        def reset(self, **kwargs):
            raise NotImplementedError

        def step(self, action):
            raise NotImplementedError

        def render(self, *args, **kwargs):
            raise NotImplementedError

        def close(self):
            return None

    class Wrapper(Env):
        def __init__(self, env):
            self.env = env
            self.action_space = getattr(env, "action_space", None)
            self.observation_space = getattr(env, "observation_space", None)
            self.metadata = getattr(env, "metadata", {})

        def reset(self, **kwargs):
            return self.env.reset(**kwargs)

        def step(self, action):
            return self.env.step(action)

        def render(self, *args, **kwargs):
            return self.env.render(*args, **kwargs)

        def close(self):
            return self.env.close()

        def __getattr__(self, name):
            return getattr(self.env, name)

    class Box:
        def __init__(self, low, high, shape=None, dtype=np.float32):
            self.dtype = np.dtype(dtype)
            if shape is None:
                low_arr = np.asarray(low, dtype=self.dtype)
                high_arr = np.asarray(high, dtype=self.dtype)
                self.shape = low_arr.shape
            else:
                self.shape = tuple(shape)
                low_arr = np.full(self.shape, low, dtype=self.dtype) if np.isscalar(low) else np.asarray(low, dtype=self.dtype)
                high_arr = np.full(self.shape, high, dtype=self.dtype) if np.isscalar(high) else np.asarray(high, dtype=self.dtype)
                low_arr = low_arr.reshape(self.shape)
                high_arr = high_arr.reshape(self.shape)

            self.low = low_arr.astype(self.dtype, copy=False)
            self.high = high_arr.astype(self.dtype, copy=False)

        def __eq__(self, other):
            return (
                isinstance(other, Box)
                and self.shape == other.shape
                and self.dtype == other.dtype
                and np.array_equal(self.low, other.low)
                and np.array_equal(self.high, other.high)
            )

    class Dict(dict):
        def __init__(self, spaces_dict=None):
            super().__init__(spaces_dict or {})

        @property
        def spaces(self):
            return self

        def __eq__(self, other):
            if not isinstance(other, Dict):
                return False
            if self.keys() != other.keys():
                return False
            return all(self[key] == other[key] for key in self.keys())

    spaces = types.SimpleNamespace(Box=Box, Dict=Dict)
    gym = types.SimpleNamespace(Env=Env, Wrapper=Wrapper, spaces=spaces)
