from .tasks.task_cfgs import PathFollowingTaskCfg, DanceTaskCfg, TaskType
import os
from isaaclab.utils import configclass
from .humanoid_amp_env_cfg import HumanoidAmpEnvCfg

MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")
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
