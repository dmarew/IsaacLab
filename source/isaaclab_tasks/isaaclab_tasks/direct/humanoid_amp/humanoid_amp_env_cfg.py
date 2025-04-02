# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import MISSING

from isaaclab_assets import HUMANOID_28_CFG

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass
from enum import Enum
from .humanoid_amp_multitask_cfg import PathFollowingTaskCfg, DanceTaskCfg, TaskType

MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")





@configclass
class HumanoidAmpEnvCfg(DirectRLEnvCfg):
    """Humanoid AMP environment config (base class)."""

    # env
    episode_length_s = 10.0
    decimation = 2

    # spaces
    observation_space = 81
    action_space = 28
    state_space = 0
    num_amp_observations = 2
    amp_observation_space = 81

    early_termination = True
    termination_height = 0.5

    motion_file: str = MISSING
    reference_body = "torso"
    reset_strategy = "random"  # default, random, random-start
    """Strategy to be followed when resetting each environment (humanoid's pose and joint states).

    * default: pose and joint states are set to the initial state of the asset.
    * random: pose and joint states are set by sampling motions at random, uniform times.
    * random-start: pose and joint states are set by sampling motion at the start (time zero).
    """

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=10.0, replicate_physics=True
    )

    # robot
    robot: ArticulationCfg = HUMANOID_28_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    ).replace(
        actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                velocity_limit=100.0,
                stiffness=None,
                damping=None,
            ),
        },
    )


@configclass
class HumanoidAmpDanceEnvCfg(HumanoidAmpEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "humanoid_dance.npz")


@configclass
class HumanoidAmpRunEnvCfg(HumanoidAmpEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "humanoid_run.npz")


@configclass
class HumanoidAmpWalkEnvCfg(HumanoidAmpEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "humanoid_walk.npz")


@configclass
class HumanoidAmpMultiTaskEnvCfg(HumanoidAmpEnvCfg):
    observation_space = 81 + 20 + 2
    motion_file = os.path.join(MOTIONS_DIR, "humanoid_walk.npz") # dummy motion file does not matter
    tasks = [
        {
            "type": TaskType.PATH_FOLLOWING,
            "cfg": PathFollowingTaskCfg(
                task_dim=20,
                task_episode_length_s=10.0,
                num_verts=101,
                dtheta_max=2.0,
                speed_min=0.5,
                speed_max=1.0,
                accel_max=1.0,
                sharp_turn_prob=0.15,
                num_traj_samples=10,
                traj_sample_timestep=0.5,
                motion_files=[
                    {
                        "motion_file": os.path.join(MOTIONS_DIR, "humanoid_walk.npz"),
                        "prob": 0.5,
                    },
                    {
                        "motion_file": os.path.join(MOTIONS_DIR, "humanoid_run.npz"),
                        "prob": 0.5,
                    },
                ],
                task_probability=0.5,
            ),
        },
        {
            "type": TaskType.DANCE,
            "cfg": DanceTaskCfg(
                task_episode_length_s=10.0,
                task_dim=2,
                motion_files=[
                    {
                        "motion_file": os.path.join(MOTIONS_DIR, "humanoid_dance.npz"),
                        "prob": 0.75,
                    },
                    {
                        "motion_file": os.path.join(MOTIONS_DIR, "humanoid_walk.npz"),
                        "prob": 0.25,
                    },
                ],
                task_probability=0.5,
            ),
        },
    ]
