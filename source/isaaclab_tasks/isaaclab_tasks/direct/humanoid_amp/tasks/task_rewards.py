
from __future__ import annotations

import torch
from isaaclab.utils.math import get_euler_xyz

@torch.jit.script
def compute_traj_reward(root_pos, tar_pos):
    # type: (Tensor, Tensor) -> Tensor
    pos_err_scale = 2.0

    pos_diff = tar_pos[..., 0:2] - root_pos[..., 0:2]
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)

    pos_reward = torch.exp(-pos_err_scale * pos_err)

    reward = pos_reward

    return reward

@torch.jit.script
def compute_sit_reward(root_pos, prev_root_pos, root_rot, prev_root_rot, 
                       object_root_pos, tar_pos, tar_speed, dt,
                       sit_vel_penalty, sit_vel_pen_coeff, sit_vel_penalty_thre, sit_ang_vel_pen_coeff, sit_ang_vel_penalty_thre,):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, bool, float, float, float, float,) -> Tensor

    d_obj2human_xy = torch.sum((object_root_pos[..., 0:2] - root_pos[..., 0:2]) ** 2, dim=-1)
    reward_far_pos = torch.exp(-0.5 * d_obj2human_xy)

    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    tar_dir = object_root_pos[..., 0:2] - root_pos[..., 0:2] # d* is a horizontal unit vector pointing from the root to the object's location
    tar_dir = torch.nn.functional.normalize(tar_dir, dim=-1)
    tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)
    tar_vel_err = tar_speed - tar_dir_speed
    reward_far_vel = torch.exp(-2.0 * tar_vel_err * tar_vel_err)

    reward_far_final = 0.0 * reward_far_pos + 1.0 * reward_far_vel
    dist_mask = (d_obj2human_xy <= 0.5 ** 2)
    reward_far_final[dist_mask] = 1.0

    # when humanoid is close to the object
    reward_near = torch.exp(-10.0 * torch.sum((tar_pos - root_pos) ** 2, dim=-1))

    reward = 0.7 * reward_near + 0.3 * reward_far_final

    if sit_vel_penalty:
        min_speed_penalty = sit_vel_penalty_thre
        root_vel_norm = torch.norm(root_vel, p=2, dim=-1)
        root_vel_norm = torch.clamp_min(root_vel_norm, min_speed_penalty)
        root_vel_err = min_speed_penalty - root_vel_norm
        root_vel_penalty = -1 * sit_vel_pen_coeff * (1 - torch.exp(-2.0 * (root_vel_err * root_vel_err)))
        dist_mask = (d_obj2human_xy <= 1.5 ** 2)
        root_vel_penalty[~dist_mask] = 0.0
        reward += root_vel_penalty

        root_z_ang_vel = torch.abs((get_euler_xyz(root_rot)[2] - get_euler_xyz(prev_root_rot)[2]) / dt)
        root_z_ang_vel = torch.clamp_min(root_z_ang_vel, sit_ang_vel_penalty_thre)
        root_z_ang_vel_err = sit_ang_vel_penalty_thre - root_z_ang_vel
        root_z_ang_vel_penalty = -1 * sit_ang_vel_pen_coeff * (1 - torch.exp(-0.5 * (root_z_ang_vel_err ** 2)))
        root_z_ang_vel_penalty[~dist_mask] = 0.0
        reward += root_z_ang_vel_penalty

    return reward

@torch.jit.script
def compute_climb_reward(root_pos, prev_root_pos, object_pos, dt, tar_pos, rigid_body_pos, feet_ids, char_h,
                         valid_radius,
                         climb_vel_penalty, climb_vel_pen_coeff, climb_vel_penalty_thre):
    # type: (Tensor, Tensor, Tensor, float, Tensor, Tensor, Tensor, float, Tensor, bool, float, float) -> Tensor

    pos_diff = object_pos[..., 0:2] - root_pos[..., 0:2]
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
    pos_reward = torch.exp(-0.5 * pos_err)

    min_speed = 1.5

    tar_dir = object_pos[..., 0:2] - root_pos[..., 0:2]
    tar_dir = torch.nn.functional.normalize(tar_dir, dim=-1)

    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)
    tar_vel_err = min_speed - tar_dir_speed
    # tar_vel_err = torch.clamp_min(tar_vel_err, 0.0) # constrain vel around the peak value
    vel_reward = torch.exp(-2.0 * (tar_vel_err * tar_vel_err))
    speed_mask = tar_dir_speed <= 0
    vel_reward[speed_mask] = 0

    dist_mask = (pos_err <= valid_radius ** 2)
    pos_reward[dist_mask] = 1.0
    vel_reward[dist_mask] = 1.0

    pos_reward_near = torch.exp(-10 * torch.sum((tar_pos - root_pos) ** 2, dim=-1))

    feet_height_err = (rigid_body_pos[:, feet_ids, -1].mean(dim=1) - (tar_pos[..., 2] - char_h)) ** 2 # height
    feet_height_reward = torch.exp(-50.0 * feet_height_err)
    feet_height_reward[~dist_mask] = 0.0

    reward = 0.0 * pos_reward + 0.2 * vel_reward + 0.5 * pos_reward_near + 0.3 * feet_height_reward

    if climb_vel_penalty:
        thre_tensor = torch.ones_like(valid_radius)
        thre_tensor[pos_err <= 1.5 ** 2] = 1.5
        thre_tensor[pos_err <= valid_radius ** 2] = climb_vel_penalty_thre
        min_speed_penalty = thre_tensor
        root_vel_norm = torch.norm(root_vel, p=2, dim=-1) # torch.abs(root_vel[..., -1]) # only consider Z
        root_vel_norm = torch.clamp_min(root_vel_norm, min_speed_penalty)
        root_vel_err = min_speed_penalty - root_vel_norm
        root_vel_penalty = -1 * climb_vel_pen_coeff * (1 - torch.exp(-2.0 * (root_vel_err * root_vel_err)))
        dist_mask = (pos_err <= 1.5 ** 2)
        root_vel_penalty[~dist_mask] = 0.0
        reward += root_vel_penalty

    return reward

@torch.jit.script
def compute_handheld_reward(humanoid_rigid_body_pos, box_pos, hands_ids, tar_pos, only_height):
    # type: (Tensor, Tensor, Tensor, Tensor, bool) -> Tensor
    if only_height:
        hands2box_pos_err = torch.sum((humanoid_rigid_body_pos[:, hands_ids, 2] - box_pos[:, 2].unsqueeze(-1)) ** 2, dim=-1) # height
    else:
        hands2box_pos_err = torch.sum((humanoid_rigid_body_pos[:, hands_ids].mean(dim=1) - box_pos) ** 2, dim=-1) # xyz
    hands2box = torch.exp(-5.0 * hands2box_pos_err)

    # box2tar = torch.sum((box_pos[..., 0:2] - tar_pos[..., 0:2]) ** 2, dim=-1) # 2d
    # hands2box[box2tar < 0.7 ** 2] = 1.0 # assume this reward is max when the box is close enough to its target location

    root_pos = humanoid_rigid_body_pos[:, 0, :]
    box2human = torch.sum((box_pos[..., 0:2] - root_pos[..., 0:2]) ** 2, dim=-1) # 2d
    hands2box[box2human > 0.7 ** 2] = 0 # disable this reward when the box is not close to the humanoid

    return 0.2 * hands2box

@torch.jit.script
def compute_walk_reward(root_pos, prev_root_pos, box_pos, dt, tar_vel, only_vel_reward):
    # type: (Tensor, Tensor, Tensor, float, float, bool) -> Tensor

    # this reward encourages the character to walk towards the box and stay close to it

    pos_diff = box_pos[..., 0:2] - root_pos[..., 0:2]
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
    pos_reward = torch.exp(-0.5 * pos_err)

    min_speed = tar_vel

    tar_dir = box_pos[..., 0:2] - root_pos[..., 0:2]
    tar_dir = torch.nn.functional.normalize(tar_dir, dim=-1)

    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)
    tar_vel_err = min_speed - tar_dir_speed
    # tar_vel_err = torch.clamp_min(tar_vel_err, 0.0) # constrain vel around the peak value
    vel_reward = torch.exp(-5.0 * (tar_vel_err * tar_vel_err))
    speed_mask = tar_dir_speed <= 0
    vel_reward[speed_mask] = 0

    dist_mask = pos_err < 0.5 ** 2
    pos_reward[dist_mask] = 1.0
    vel_reward[dist_mask] = 1.0

    if only_vel_reward:
        reward = 0.2 * vel_reward
    else:
        reward = 0.1 * pos_reward + 0.1 * vel_reward
    return reward

@torch.jit.script
def compute_carry_reward(box_pos, prev_box_pos, tar_box_pos, dt, tar_vel, box_size, only_vel_reward, box_vel_penalty, box_vel_pen_coeff, box_vel_penalty_thre):
    # type: (Tensor, Tensor, Tensor, float, float, Tensor, bool, bool, float, float) -> Tensor
    
    # this reward encourages the character to carry the box to a target position

    pos_diff = tar_box_pos - box_pos # xyz
    pos_err_xy = torch.sum(pos_diff[..., 0:2] ** 2, dim=-1)
    pos_err_xyz = torch.sum(pos_diff * pos_diff, dim=-1)
    pos_reward_far = torch.exp(-0.5 * pos_err_xy)
    pos_reward_near = torch.exp(-10.0 * pos_err_xyz)

    min_speed = tar_vel

    tar_dir = tar_box_pos[..., 0:2] - box_pos[..., 0:2]
    tar_dir = torch.nn.functional.normalize(tar_dir, dim=-1)

    delta_root_pos = box_pos - prev_box_pos
    root_vel = delta_root_pos / dt
    tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)
    tar_vel_err = min_speed - tar_dir_speed
    # tar_vel_err = torch.clamp_min(tar_vel_err, 0.0) # constrain vel around the peak value
    vel_reward = torch.exp(-5.0 * (tar_vel_err * tar_vel_err))
    speed_mask = tar_dir_speed <= 0
    vel_reward[speed_mask] = 0

    height_mask = box_pos[..., 2] <= (box_size[..., 2] / 2 + 0.2) # avoid learning to kick the box
    pos_reward_far[height_mask] = 0.0
    vel_reward[height_mask] = 0.0

    dist_mask = pos_err_xy < 0.5 ** 2
    pos_reward_far[dist_mask] = 1.0
    vel_reward[dist_mask] = 1.0

    if only_vel_reward:
        reward = 0.2 * vel_reward + 0.2 * pos_reward_near
    else:
        reward = 0.1 * pos_reward_far + 0.1 * vel_reward + 0.2 * pos_reward_near

    if box_vel_penalty:
        min_speed_penalty = box_vel_penalty_thre
        root_vel_norm = torch.norm(root_vel, p=2, dim=-1)
        root_vel_norm = torch.clamp_min(root_vel_norm, min_speed_penalty)
        root_vel_err = min_speed_penalty - root_vel_norm
        root_vel_penalty = -1 * box_vel_pen_coeff * (1 - torch.exp(-2.0 * (root_vel_err * root_vel_err)))
        reward += root_vel_penalty

    return reward

@torch.jit.script
def compute_putdown_reward(box_pos, tar_pos):
    # type: (Tensor, Tensor) -> Tensor

    reward = (torch.abs((box_pos[:, -1] - tar_pos[:, -1])) <= 0.001) * 1.0 # binary reward, 0.0 or 1.0
    
    pos_err_xy = torch.sum((tar_pos[..., :2] - box_pos[..., :2]) ** 2, dim=-1)
    reward[(pos_err_xy > 0.1 ** 2)] = 0.0
    
    return 0.2 * reward