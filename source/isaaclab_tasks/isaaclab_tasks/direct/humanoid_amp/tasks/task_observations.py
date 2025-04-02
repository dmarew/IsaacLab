from __future__ import annotations

import torch
from isaaclab.utils.math import calc_heading_quat_inv, my_quat_rotate, quat_mul, quat_rotate, quat_to_tan_norm


@torch.jit.script
def compute_location_observations(root_pos, root_rot, traj_samples):
    # type: (Tensor, Tensor, Tensor) -> Tensor

    heading_rot = calc_heading_quat_inv(root_rot)
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
    local_traj_pos = my_quat_rotate(heading_rot_exp, traj_samples_delta_flat)
    local_traj_pos = local_traj_pos[..., 0:2]

    obs = torch.reshape(
        local_traj_pos,
        (traj_samples.shape[0], traj_samples.shape[1] * local_traj_pos.shape[1]),
    )
    return obs


@torch.jit.script
def compute_traj_obs(root_pos: Tensor, root_rot: Tensor, traj_samples: Tensor) -> Tensor:
    # Compute heading rotation from root orientation.
    heading_rot = calc_heading_quat_inv(root_rot)
    
    B = root_pos.shape[0]
    NM = traj_samples.shape[1]
    
    # Expand heading rotation and root position for each trajectory sample.
    heading_rot_exp = torch.broadcast_to(heading_rot.unsqueeze(1), (B, NM, 4))
    root_pos_exp = torch.broadcast_to(root_pos.unsqueeze(1), (B, NM, 3))
    
    N = traj_samples.shape[1]
    # Rotate the difference (traj sample minus root position) into the local frame.
    local_traj = quat_rotate(
        heading_rot_exp[:, :N].reshape(-1, 4), 
        traj_samples.reshape(-1, 3) - root_pos_exp[:, :N].reshape(-1, 3)
    )[..., 0:2]
    
    return local_traj.reshape(B, -1)

@torch.jit.script
def compute_sit_obs(
    root_pos: Tensor,
    root_rot: Tensor,
    sit_tar_pos: Tensor, 
    sit_obj_pos: Tensor, 
    sit_obj_rot: Tensor, 
    sit_object_bps: Tensor, 
    sit_object_facings: Tensor
) -> Tensor:
    # Compute heading rotation from the root orientation.
    heading_rot = calc_heading_quat_inv(root_rot)
    
    # Compute the sit target position in the local frame.
    local_sit_tar_pos = quat_rotate(heading_rot, sit_tar_pos - root_pos)
    
    # For the sit object, use its separate position and rotation.
    local_sit_obj_root_pos = quat_rotate(heading_rot, sit_obj_pos - root_pos)
    local_sit_obj_root_rot = quat_mul(heading_rot, sit_obj_rot)
    local_sit_obj_root_rot = quat_to_tan_norm(local_sit_obj_root_rot)
    
    B = root_pos.shape[0]
    N = sit_object_bps.shape[1]
    
    # Broadcast sit object states to match the number of bounding points.
    sit_obj_pos_exp = torch.broadcast_to(sit_obj_pos.unsqueeze(1), (B, N, 3)).reshape(-1, 3)
    sit_obj_rot_exp = torch.broadcast_to(sit_obj_rot.unsqueeze(1), (B, N, 4)).reshape(-1, 4)
    
    # Transform bounding points from object local space to world space.
    sit_obj_bps_world = quat_rotate(sit_obj_rot_exp, sit_object_bps.reshape(-1, 3)) + sit_obj_pos_exp
    heading_rot_exp = torch.broadcast_to(heading_rot.unsqueeze(1), (B, N, 4)).reshape(-1, 4)
    root_pos_exp = torch.broadcast_to(root_pos.unsqueeze(1), (B, N, 3)).reshape(-1, 3)
    # Then bring them into the humanoid's local space.
    sit_obj_bps_local = quat_rotate(heading_rot_exp, sit_obj_bps_world - root_pos_exp).reshape(B, N * 3)
    
    # Rotate the object's facing vector into the local frame.
    sit_face_vec_world = quat_rotate(sit_obj_rot, sit_object_facings)
    sit_face_vec_local = quat_rotate(heading_rot, sit_face_vec_world)[..., 0:2]
    
    return torch.cat([
        local_sit_tar_pos, 
        sit_obj_bps_local, 
        sit_face_vec_local, 
        local_sit_obj_root_pos, 
        local_sit_obj_root_rot
    ], dim=-1)

@torch.jit.script
def compute_box_obs(
    root_pos: Tensor,
    root_rot: Tensor,
    box_pos: Tensor,
    box_rot: Tensor,
    box_vel: Tensor,
    box_ang_vel: Tensor,
    box_bps: Tensor, 
    box_tar_pos: Tensor
) -> Tensor:
    # Compute heading rotation.
    heading_rot = calc_heading_quat_inv(root_rot)
    
    # Transform box properties into the root's local frame.
    local_box_pos = quat_rotate(heading_rot, box_pos - root_pos)
    local_box_rot = quat_mul(heading_rot, box_rot)
    local_box_rot_obs = quat_to_tan_norm(local_box_rot)
    local_box_vel = quat_rotate(heading_rot, box_vel)
    local_box_ang_vel = quat_rotate(heading_rot, box_ang_vel)
    
    B = root_pos.shape[0]
    N = box_bps.shape[1]
    
    # Process box bounding points: from object local space -> world space -> local space.
    box_pos_exp = torch.broadcast_to(box_pos.unsqueeze(1), (B, N, 3))
    box_rot_exp = torch.broadcast_to(box_rot.unsqueeze(1), (B, N, 4))
    box_bps_world = quat_rotate(box_rot_exp.reshape(-1, 4), box_bps.reshape(-1, 3)) + box_pos_exp.reshape(-1, 3)
    heading_rot_exp = torch.broadcast_to(heading_rot.unsqueeze(1), (B, N, 4)).reshape(-1, 4)
    root_pos_exp = torch.broadcast_to(root_pos.unsqueeze(1), (B, N, 3)).reshape(-1, 3)
    box_bps_local = quat_rotate(heading_rot_exp, box_bps_world - root_pos_exp).reshape(B, N * 3)
    
    # Transform the target box position.
    local_box_tar_pos = quat_rotate(heading_rot, box_tar_pos - root_pos)
    
    return torch.cat([
        local_box_vel, 
        local_box_ang_vel, 
        local_box_pos, 
        local_box_rot_obs, 
        box_bps_local, 
        local_box_tar_pos
    ], dim=-1)

@torch.jit.script
def compute_climb_obs(
    root_pos: Tensor,
    root_rot: Tensor,
    climb_obj_pos: Tensor,
    climb_obj_rot: Tensor,
    climb_object_bps: Tensor, 
    climb_tar_pos: Tensor
) -> Tensor:
    # Compute heading rotation.
    heading_rot = calc_heading_quat_inv(root_rot)
    
    # Transform the climb target position.
    local_climb_tar_pos = quat_rotate(heading_rot, climb_tar_pos - root_pos)
    
    B = root_pos.shape[0]
    N = climb_object_bps.shape[1]
    
    # Broadcast climb object states.
    climb_obj_pos_exp = torch.broadcast_to(climb_obj_pos.unsqueeze(1), (B, N, 3)).reshape(-1, 3)
    climb_obj_rot_exp = torch.broadcast_to(climb_obj_rot.unsqueeze(1), (B, N, 4)).reshape(-1, 4)
    
    # Compute climb bounding points.
    climb_obj_bps_world = quat_rotate(climb_obj_rot_exp, climb_object_bps.reshape(-1, 3)) + climb_obj_pos_exp
    heading_rot_exp = torch.broadcast_to(heading_rot.unsqueeze(1), (B, N, 4)).reshape(-1, 4)
    root_pos_exp = torch.broadcast_to(root_pos.unsqueeze(1), (B, N, 3)).reshape(-1, 3)
    climb_obj_bps_local = quat_rotate(heading_rot_exp, climb_obj_bps_world - root_pos_exp).reshape(B, N * 3)
    
    return torch.cat([local_climb_tar_pos, climb_obj_bps_local], dim=-1)

@torch.jit.script
def compute_all_observations(
    # Root properties
    root_pos: Tensor,
    root_rot: Tensor,
    traj_samples: Tensor,
    # Sit task
    sit_tar_pos: Tensor,
    sit_obj_pos: Tensor,
    sit_obj_rot: Tensor,
    sit_object_bps: Tensor,
    sit_object_facings: Tensor,
    # Box (carry) task
    box_pos: Tensor,
    box_rot: Tensor,
    box_vel: Tensor,
    box_ang_vel: Tensor,
    box_bps: Tensor,
    box_tar_pos: Tensor,
    # Climb task
    climb_obj_pos: Tensor,
    climb_obj_rot: Tensor,
    climb_object_bps: Tensor,
    climb_tar_pos: Tensor,
    # Masking options
    task_mask: Tensor, 
    each_subtask_obs_mask: Tensor, 
    enable_apply_mask: bool
) -> Tensor:
    # Compute each sub-observation.
    traj_obs = compute_traj_obs(root_pos, root_rot, traj_samples)
    sit_obs = compute_sit_obs(root_pos, root_rot, sit_tar_pos, sit_obj_pos, sit_obj_rot, sit_object_bps, sit_object_facings)
    box_obs = compute_box_obs(root_pos, root_rot, box_pos, box_rot, box_vel, box_ang_vel, box_bps, box_tar_pos)
    climb_obs = compute_climb_obs(root_pos, root_rot, climb_obj_pos, climb_obj_rot, climb_object_bps, climb_tar_pos)
    
    # Concatenate all the components.
    obs = torch.cat([traj_obs, sit_obs, box_obs, climb_obs], dim=-1)
    
    if enable_apply_mask:
        mask = task_mask[:, None, :].float() @ torch.broadcast_to(
            each_subtask_obs_mask[None, :, :].float(),
            (root_pos.shape[0], each_subtask_obs_mask.shape[0], each_subtask_obs_mask.shape[1])
        )
        obs = obs * mask.squeeze(1)
        
    return obs
