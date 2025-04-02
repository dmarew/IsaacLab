# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch


from .task_cfgs import (
    DanceTaskCfg,
)
from .base_task import Task


class DanceTask(Task):
    def __init__(self, cfg: DanceTaskCfg, num_envs: int, device: torch.device):
        super().__init__(cfg)

    def get_obs(self, env, env_ids: torch.Tensor) -> torch.Tensor:
        root_pos = env.robot.data.body_pos_w[env_ids, env.ref_body_index][:, 0:2]
        return root_pos

    def get_reward(self, env, env_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros(len(env_ids), device=env.device)

    def reset(self, env, env_ids: torch.Tensor):
        return
