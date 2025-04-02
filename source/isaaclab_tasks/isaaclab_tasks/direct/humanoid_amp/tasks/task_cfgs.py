from isaaclab.utils import configclass
from enum import Enum
from typing import List, Dict, Union
from dataclasses import MISSING
class TaskType(Enum):
    PATH_FOLLOWING = "path_following"
    BOX_CARRYING = "box_carrying"
    DANCE = "dance"


@configclass
class TaskCfg:
    task_dim: int = MISSING
    task_episode_length_s: float = MISSING
    motion_files: List[Dict[str, Union[str, float]]] = MISSING
    task_probability: float = MISSING

@configclass
class PathFollowingTaskCfg(TaskCfg):
    num_verts: int = MISSING
    dtheta_max: float = MISSING
    speed_min: float = MISSING
    speed_max: float = MISSING
    accel_max: float = MISSING
    sharp_turn_prob: float = MISSING
    num_traj_samples: int = MISSING
    traj_sample_timestep: float = MISSING


@configclass
class BoxCarryingTaskCfg(TaskCfg):
    pass

@configclass
class DanceTaskCfg(TaskCfg):
    pass
