from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

import diffusion_policies.env.robosuite.robosuite_wrapper as base_wrapper
from diffusion_policies.env.robosuite.robosuite_wrapper import Robosuite3DEnv as BaseRobosuite3DEnv


REPRO_ROOT = Path(__file__).resolve().parents[4]
FORK_DIR = REPRO_ROOT / "forks" / "handlepress"

fork_dir_str = str(FORK_DIR)
if fork_dir_str not in sys.path:
    sys.path.insert(0, fork_dir_str)

import handlepress_env  # noqa: F401


base_wrapper.TASK_BOUDNS.setdefault(
    "HandlePress",
    [-0.55, -0.65, 0.75, 0.25, 0.20, 1.30],
)
base_wrapper.SINGLE_CEMERA_NAME.setdefault("HandlePress", "agentview")


class HandlePressRobosuite3DEnv(BaseRobosuite3DEnv):
    def reset_to(self, state):
        state = deepcopy(state)
        object_translation = state.pop("object_translation", None)

        obs = super().reset_to(state)

        if object_translation is None:
            return obs

        object_translation = np.asarray(object_translation, dtype=np.float32)
        if object_translation.shape[0] < 3:
            raise ValueError("object_translation must have at least 3 elements")

        root_body_name = self.env.door.root_body
        root_body_id = self.env.sim.model.body_name2id(root_body_name)
        self.env.sim.model.body_pos[root_body_id] = (
            self.env.sim.model.body_pos[root_body_id]
            + object_translation[:3]
        )
        self.env.sim.forward()

        obs_dict = self.get_observation()
        return self.process_obs_dict(obs_dict)
