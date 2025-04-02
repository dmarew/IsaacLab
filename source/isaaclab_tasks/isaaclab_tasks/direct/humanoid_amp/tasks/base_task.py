# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch


import isaaclab.utils.math as math_utils
from .task_cfgs import TaskCfg

class Task:
    def __init__(self, cfg: TaskCfg):
        self.cfg = cfg

    def get_obs(self, env, env_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_reward(self, env, env_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_done(self, env) -> torch.Tensor:
        return (
            env.episode_length_buf
            >= int(self.cfg.task_episode_length_s / env.step_dt) - 1
        )

    def reset(self, env, env_ids: torch.Tensor):
        raise NotImplementedError


