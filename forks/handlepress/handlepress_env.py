#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from robosuite.environments.manipulation.door import Door


class HandlePress(Door):
    """
    Forked single-object handle-press task.

    This reuses the robosuite Door scene with the spring-loaded latch enabled,
    but treats handle rotation itself as the task objective instead of opening
    the whole door.
    """

    def __init__(
        self,
        *args,
        handle_success_threshold: float = 0.65,
        use_latch: bool = True,
        reward_shaping: bool = True,
        **kwargs,
    ):
        self.handle_success_threshold = float(handle_success_threshold)
        if self.handle_success_threshold <= 0:
            raise ValueError("handle_success_threshold must be positive")

        super().__init__(
            *args,
            use_latch=use_latch,
            reward_shaping=reward_shaping,
            **kwargs,
        )

        if not self.use_latch:
            raise ValueError("HandlePress requires use_latch=True")

    def _check_success(self):
        handle_qpos = float(self.sim.data.qpos[self.handle_qpos_addr])
        return abs(handle_qpos) >= self.handle_success_threshold

    def reward(self, action=None):
        reward = 0.0

        if self._check_success():
            reward = 1.0
        elif self.reward_shaping:
            dist = np.linalg.norm(self._gripper_to_handle)
            reaching_reward = 0.25 * (1 - np.tanh(10.0 * dist))
            handle_qpos = float(self.sim.data.qpos[self.handle_qpos_addr])
            handle_progress = min(1.0, abs(handle_qpos) / self.handle_success_threshold)
            rotating_reward = 0.75 * handle_progress
            reward = reaching_reward + rotating_reward

        if self.reward_scale is not None:
            reward *= self.reward_scale / 1.0
        return reward
