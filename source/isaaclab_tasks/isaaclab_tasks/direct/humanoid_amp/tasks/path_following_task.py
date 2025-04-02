# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from .task_cfgs import (
    PathFollowingTaskCfg,
)
from .base_task import Task
from .traj_generator import TrajGenerator
from .task_rewards import compute_traj_reward
from .task_observations import compute_traj_obs

class PathFollowingTask(Task):
    def __init__(self, cfg: PathFollowingTaskCfg, num_envs: int, device: torch.device):
        super().__init__(cfg)
        self.traj_generator = TrajGenerator(
            num_envs,
            cfg.task_episode_length_s,
            cfg.num_verts,
            device,
            cfg.dtheta_max,
            cfg.speed_min,
            cfg.speed_max,
            cfg.accel_max,
            cfg.sharp_turn_prob,
        )
        self.num_traj_samples = cfg.num_traj_samples
        self.traj_sample_timestep = cfg.traj_sample_timestep

    def get_obs(self, env, env_ids: torch.Tensor) -> torch.Tensor:

        timestep_beg = env.episode_length_buf[env_ids] * env.step_dt
        timesteps = torch.arange(
            self.num_traj_samples, device=env.device, dtype=torch.float
        )
        timesteps = timesteps * self.traj_sample_timestep
        traj_timesteps = timestep_beg.unsqueeze(-1) + timesteps

        env_ids_tiled = torch.broadcast_to(env_ids.unsqueeze(-1), traj_timesteps.shape)

        traj_samples_flat = self.traj_generator.calc_pos(
            env_ids_tiled.flatten(), traj_timesteps.flatten()
        )
        traj_samples = torch.reshape(
            traj_samples_flat,
            shape=(
                env_ids.shape[0],
                self.num_traj_samples,
                traj_samples_flat.shape[-1],
            ),
        )

        root_pos = env.robot.data.body_pos_w[env_ids, env.ref_body_index]
        root_rot = env.robot.data.body_quat_w[env_ids, env.ref_body_index]

        return compute_traj_obs(root_pos, root_rot, traj_samples)

    def get_reward(self, env, env_ids: torch.Tensor) -> torch.Tensor:
        root_pos = env.robot.data.body_pos_w[env_ids, env.ref_body_index]
        time = env.episode_length_buf[env_ids] * env.step_dt
        tar_pos = self.traj_generator.calc_pos(env_ids, time)
        return compute_traj_reward(root_pos, tar_pos)

    def reset(self, env, env_ids: torch.Tensor):
        root_pos = env.robot.data.body_pos_w[env_ids, env.ref_body_index]
        all_env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
        self.traj_generator.reset(all_env_ids, root_pos)




