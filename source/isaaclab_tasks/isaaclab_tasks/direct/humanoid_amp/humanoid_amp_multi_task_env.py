# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

from .humanoid_amp_env_cfg import HumanoidAmpMultiTaskEnvCfg
from .humanoid_amp_env import HumanoidAmpEnv
from .motions.motion_loader import TaskMultiMotionLoader
from .humanoid_amp_env import compute_obs
import isaaclab.utils.math as math_utils
from typing import Tuple
from .humanoid_amp_multitask_cfg import (
    TaskCfg,
    PathFollowingTaskCfg,
    DanceTaskCfg,
    TaskType,
)


class HumanoidAmpMultiTaskEnv(HumanoidAmpEnv):
    cfg: HumanoidAmpMultiTaskEnvCfg

    def __init__(
        self, cfg: HumanoidAmpMultiTaskEnvCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)
        del self._motion_loader

        self.task_probabilities = [task["cfg"].task_probability for task in cfg.tasks]
        self.task_dims = [task["cfg"].task_dim for task in cfg.tasks]
        self.motion_files = [task["cfg"].motion_files for task in cfg.tasks]

        self.task_obs_size = sum(self.task_dims)
        self.state_dim = self.cfg.observation_space - self.task_obs_size

        self.tasks = []
        for task in cfg.tasks:
            if task["type"] == TaskType.PATH_FOLLOWING:
                self.tasks.append(
                    PathFollowingTask(task["cfg"], self.num_envs, self.device)
                )
            elif task["type"] == TaskType.DANCE:
                self.tasks.append(DanceTask(task["cfg"], self.num_envs, self.device))

        self._motion_loader = TaskMultiMotionLoader(
            self.motion_files, self.task_probabilities, self.device
        )
        self.num_tasks = len(self.task_probabilities)
        assert len(self.task_dims) == self.num_tasks

        self.cfg.amp_observation_space = (
            self.num_tasks + self.cfg.amp_observation_space
        )  # adding one hot vector for the task
        self.amp_observation_size = (
            self.cfg.num_amp_observations * self.cfg.amp_observation_space
        )
        self.amp_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.amp_observation_size,)
        )
        self.amp_observation_buffer = torch.zeros(
            (
                self.num_envs,
                self.cfg.num_amp_observations,
                self.cfg.amp_observation_space,
            ),
            device=self.device,
        )
        self.task_onehot = torch.zeros(
            (self.num_envs, self.num_tasks), device=self.device
        )
        self.task_assignment = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.long
        )
        self.num_motions_by_task = self._motion_loader.num_motions_by_task

        for task in self.tasks:
            task.reset(
                self, torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            )

    def _get_observations(self) -> dict:
        # build task observation
        proprio_obs = compute_obs(
            self.robot.data.joint_pos,
            self.robot.data.joint_vel,
            self.robot.data.body_pos_w[:, self.ref_body_index],
            self.robot.data.body_quat_w[:, self.ref_body_index],
            self.robot.data.body_lin_vel_w[:, self.ref_body_index],
            self.robot.data.body_ang_vel_w[:, self.ref_body_index],
            self.robot.data.body_pos_w[:, self.key_body_indexes],
        )
        # update AMP observation history
        for i in reversed(range(self.cfg.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        # amp obs = proprio + motion_type_onehot concatenated
        self.amp_observation_buffer[:, 0] = torch.cat(
            (proprio_obs, self.task_onehot), dim=-1
        )

        task_obs = self._compute_task_obs()

        self.extras = {
            "amp_obs": self.amp_observation_buffer.view(-1, self.amp_observation_size)
        }

        obs = torch.cat((proprio_obs, task_obs), dim=-1)
        return {"policy": obs}

    def _compute_task_obs(self) -> torch.Tensor:
        task_obs = []
        for i, task in enumerate(self.tasks):
            env_ids = torch.where(self.task_assignment == i)[0]
            task_i_obs = torch.zeros(
                self.num_envs, self.task_dims[i], device=self.device
            )
            task_i_obs[env_ids] = task.get_obs(self, env_ids)
            task_obs.append(task_i_obs)
        return torch.cat(task_obs, dim=-1)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        return super()._get_dones()

    def _get_rewards(self) -> torch.Tensor:
        rewards = torch.zeros(self.num_envs, device=self.device)
        for i, task in enumerate(self.tasks):
            env_ids = torch.where(self.task_assignment == i)[0]
            rewards[env_ids] = task.get_reward(self, env_ids)
        return rewards

    def _reset_strategy_random(
        self, env_ids: torch.Tensor, start: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # sample random motion times (or zeros if start is True)
        num_samples = env_ids.shape[0]
        times_dict = self._motion_loader.sample_task_motion_and_times(num_samples)

        if start:
            for i in range(self.num_tasks):
                for j in range(self.num_motions_by_task[i]):
                    times_dict[(i, j)][:] = 0

        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
            task_assignment,
        ) = self._motion_loader.sample_motion(times_dict)
        task_onehot = torch.nn.functional.one_hot(
            task_assignment, num_classes=self.num_tasks
        ).float()

        self.task_assignment[env_ids] = task_assignment
        self.task_onehot[env_ids] = task_onehot
        # get root transforms (the humanoid torso)
        motion_torso_index = self._motion_loader.get_body_index(["torso"])[0]
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, 0:3] = (
            body_positions[:, motion_torso_index] + self.scene.env_origins[env_ids]
        )
        root_state[
            :, 2
        ] += 0.15  # lift the humanoid slightly to avoid collisions with the ground
        root_state[:, 3:7] = body_rotations[:, motion_torso_index]
        root_state[:, 7:10] = body_linear_velocities[:, motion_torso_index]
        root_state[:, 10:13] = body_angular_velocities[:, motion_torso_index]
        # get DOFs state
        dof_pos = dof_positions[:, self.motion_dof_indexes]
        dof_vel = dof_velocities[:, self.motion_dof_indexes]

        # update AMP observation
        amp_observations = self.collect_reference_motions(num_samples, times_dict)
        self.amp_observation_buffer[env_ids] = amp_observations.view(
            num_samples, self.cfg.num_amp_observations, -1
        )

        return root_state, dof_pos, dof_vel

    # env methods

    def collect_reference_motions(
        self,
        num_samples: int,
        times_dict: dict[Tuple[int, int], np.ndarray] | None = None,
    ) -> torch.Tensor:
        # If no times_dict is provided, sample from the motion loader.
        if times_dict is None:
            times_dict = self._motion_loader.sample_task_motion_and_times(num_samples)
        # Get the number of AMP observations per environment.
        num_amp = self.cfg.num_amp_observations
        dt = self._motion_loader.dt
        times_dict_expanded = {}

        for (task_idx, motion_idx), current_times in times_dict.items():
            times_dict_expanded[(task_idx, motion_idx)] = (
                np.expand_dims(current_times, axis=-1) - dt * np.arange(num_amp)
            ).flatten()
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
            task_assignment,
        ) = self._motion_loader.sample_motion(times_dict=times_dict_expanded)
        task_onehot = torch.nn.functional.one_hot(
            task_assignment, num_classes=self.num_tasks
        ).float()

        # Compute the AMP observation.
        amp_observation = compute_obs(
            dof_positions[:, self.motion_dof_indexes],
            dof_velocities[:, self.motion_dof_indexes],
            body_positions[:, self.motion_ref_body_index],
            body_rotations[:, self.motion_ref_body_index],
            body_linear_velocities[:, self.motion_ref_body_index],
            body_angular_velocities[:, self.motion_ref_body_index],
            body_positions[:, self.motion_key_body_indexes],
        )

        # Concatenate the task one-hot vector to the AMP observation.
        amp_observation = torch.cat((amp_observation, task_onehot), dim=-1)
        # Reshape the observation so that each environment's AMP observation is concatenated
        # into a single vector of size self.amp_observation_size.
        return amp_observation.view(num_samples, self.amp_observation_size)


# abstract class for Tasks each task should have obs and reward function


class Task:
    def __init__(self, cfg: TaskCfg):
        self.cfg = cfg

    def get_obs(
        self, env: HumanoidAmpMultiTaskEnv, env_ids: torch.Tensor
    ) -> torch.Tensor:
        pass

    def get_reward(
        self, env: HumanoidAmpMultiTaskEnv, env_ids: torch.Tensor
    ) -> torch.Tensor:
        pass

    def get_done(
        self, env: HumanoidAmpMultiTaskEnv, env_ids: torch.Tensor
    ) -> torch.Tensor:
        pass

    def reset(self, env: HumanoidAmpMultiTaskEnv, env_ids: torch.Tensor):
        pass


from .motions.traj_generator import TrajGenerator


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

    def get_obs(
        self, env: HumanoidAmpMultiTaskEnv, env_ids: torch.Tensor
    ) -> torch.Tensor:

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

        return compute_location_observations(root_pos, root_rot, traj_samples)

    def get_reward(
        self, env: HumanoidAmpMultiTaskEnv, env_ids: torch.Tensor
    ) -> torch.Tensor:
        root_pos = env.robot.data.body_pos_w[env_ids, env.ref_body_index]
        time = env.episode_length_buf[env_ids] * env.step_dt
        tar_pos = self.traj_generator.calc_pos(env_ids, time)
        return compute_location_reward(root_pos, tar_pos)

    def reset(self, env: HumanoidAmpMultiTaskEnv, env_ids: torch.Tensor):
        root_pos = env.robot.data.body_pos_w[env_ids, env.ref_body_index]
        all_env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
        self.traj_generator.reset(all_env_ids, root_pos)

class DanceTask(Task):
    def __init__(self, cfg: DanceTaskCfg, num_envs: int, device: torch.device):
        super().__init__(cfg)

    def get_obs(
        self, env: HumanoidAmpMultiTaskEnv, env_ids: torch.Tensor
    ) -> torch.Tensor:
        root_pos = env.robot.data.body_pos_w[env_ids, env.ref_body_index][:, 0:2]
        return root_pos

    def get_reward(
        self, env: HumanoidAmpMultiTaskEnv, env_ids: torch.Tensor
    ) -> torch.Tensor:
        return torch.zeros(len(env_ids), device=env.device)

    def get_done(
        self, env: HumanoidAmpMultiTaskEnv, env_ids: torch.Tensor
    ) -> torch.Tensor:
        return torch.zeros(len(env_ids), device=env.device)

    def reset(self, env: HumanoidAmpMultiTaskEnv, env_ids: torch.Tensor):
        pass


@torch.jit.script
def compute_location_reward(root_pos, tar_pos):
    # type: (Tensor, Tensor) -> Tensor
    pos_err_scale = 2.0

    pos_diff = tar_pos[..., 0:2] - root_pos[..., 0:2]
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)

    pos_reward = torch.exp(-pos_err_scale * pos_err)

    reward = pos_reward

    return reward


@torch.jit.script
def compute_location_observations(root_pos, root_rot, traj_samples):
    # type: (Tensor, Tensor, Tensor) -> Tensor

    heading_rot = math_utils.calc_heading_quat_inv(root_rot)
    heading_rot_exp = torch.broadcast_to(
        heading_rot.unsqueeze(-2),
        (heading_rot.shape[0], traj_samples.shape[1], heading_rot.shape[1]),
    )
    heading_rot_exp = torch.reshape(
        heading_rot_exp,
        (heading_rot_exp.shape[0] * heading_rot_exp.shape[1], heading_rot_exp.shape[2]),
    )
    traj_samples_delta = traj_samples - root_pos.unsqueeze(-2)
    traj_samples_delta_flat = torch.reshape(
        traj_samples_delta,
        (
            traj_samples_delta.shape[0] * traj_samples_delta.shape[1],
            traj_samples_delta.shape[2],
        ),
    )
    local_traj_pos = math_utils.my_quat_rotate(heading_rot_exp, traj_samples_delta_flat)
    local_traj_pos = local_traj_pos[..., 0:2]

    obs = torch.reshape(
        local_traj_pos,
        (traj_samples.shape[0], traj_samples.shape[1] * local_traj_pos.shape[1]),
    )
    return obs
