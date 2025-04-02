# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import os
import torch
from typing import Optional, List, Tuple, Dict


class MotionLoader:
    """
    Helper class to load and sample motion data from NumPy-file format.
    """

    def __init__(self, motion_file: str, device: torch.device) -> None:
        """Load a motion file and initialize the internal variables.

        Args:
            motion_file: Motion file path to load.
            device: The device to which to load the data.

        Raises:
            AssertionError: If the specified motion file doesn't exist.
        """
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)

        self.device = device
        self._dof_names = data["dof_names"].tolist()
        self._body_names = data["body_names"].tolist()

        self.dof_positions = torch.tensor(
            data["dof_positions"], dtype=torch.float32, device=self.device
        )
        self.dof_velocities = torch.tensor(
            data["dof_velocities"], dtype=torch.float32, device=self.device
        )
        self.body_positions = torch.tensor(
            data["body_positions"], dtype=torch.float32, device=self.device
        )
        self.body_rotations = torch.tensor(
            data["body_rotations"], dtype=torch.float32, device=self.device
        )
        self.body_linear_velocities = torch.tensor(
            data["body_linear_velocities"], dtype=torch.float32, device=self.device
        )
        self.body_angular_velocities = torch.tensor(
            data["body_angular_velocities"], dtype=torch.float32, device=self.device
        )

        self.dt = 1.0 / data["fps"]
        self.num_frames = self.dof_positions.shape[0]
        self.duration = self.dt * (self.num_frames - 1)
        print(
            f"Motion loaded ({motion_file}): duration: {self.duration} sec, frames: {self.num_frames}"
        )

    @property
    def dof_names(self) -> list[str]:
        """Skeleton DOF names."""
        return self._dof_names

    @property
    def body_names(self) -> list[str]:
        """Skeleton rigid body names."""
        return self._body_names

    @property
    def num_dofs(self) -> int:
        """Number of skeleton's DOFs."""
        return len(self._dof_names)

    @property
    def num_bodies(self) -> int:
        """Number of skeleton's rigid bodies."""
        return len(self._body_names)

    def _interpolate(
        self,
        a: torch.Tensor,
        *,
        b: Optional[torch.Tensor] = None,
        blend: Optional[torch.Tensor] = None,
        start: Optional[np.ndarray] = None,
        end: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """Linear interpolation between consecutive values.

        Args:
            a: The first value. Shape is (N, X) or (N, M, X).
            b: The second value. Shape is (N, X) or (N, M, X).
            blend: Interpolation coefficient between 0 (a) and 1 (b).
            start: Indexes to fetch the first value. If both, ``start`` and ``end` are specified,
                the first and second values will be fetches from the argument ``a`` (dimension 0).
            end: Indexes to fetch the second value. If both, ``start`` and ``end` are specified,
                the first and second values will be fetches from the argument ``a`` (dimension 0).

        Returns:
            Interpolated values. Shape is (N, X) or (N, M, X).
        """
        if start is not None and end is not None:
            return self._interpolate(a=a[start], b=a[end], blend=blend)
        if a.ndim >= 2:
            blend = blend.unsqueeze(-1)
        if a.ndim >= 3:
            blend = blend.unsqueeze(-1)
        return (1.0 - blend) * a + blend * b

    def _slerp(
        self,
        q0: torch.Tensor,
        *,
        q1: Optional[torch.Tensor] = None,
        blend: Optional[torch.Tensor] = None,
        start: Optional[np.ndarray] = None,
        end: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """Interpolation between consecutive rotations (Spherical Linear Interpolation).

        Args:
            q0: The first quaternion (wxyz). Shape is (N, 4) or (N, M, 4).
            q1: The second quaternion (wxyz). Shape is (N, 4) or (N, M, 4).
            blend: Interpolation coefficient between 0 (q0) and 1 (q1).
            start: Indexes to fetch the first quaternion. If both, ``start`` and ``end` are specified,
                the first and second quaternions will be fetches from the argument ``q0`` (dimension 0).
            end: Indexes to fetch the second quaternion. If both, ``start`` and ``end` are specified,
                the first and second quaternions will be fetches from the argument ``q0`` (dimension 0).

        Returns:
            Interpolated quaternions. Shape is (N, 4) or (N, M, 4).
        """
        if start is not None and end is not None:
            return self._slerp(q0=q0[start], q1=q0[end], blend=blend)
        if q0.ndim >= 2:
            blend = blend.unsqueeze(-1)
        if q0.ndim >= 3:
            blend = blend.unsqueeze(-1)

        qw, qx, qy, qz = 0, 1, 2, 3  # wxyz
        cos_half_theta = (
            q0[..., qw] * q1[..., qw]
            + q0[..., qx] * q1[..., qx]
            + q0[..., qy] * q1[..., qy]
            + q0[..., qz] * q1[..., qz]
        )

        neg_mask = cos_half_theta < 0
        q1 = q1.clone()
        q1[neg_mask] = -q1[neg_mask]
        cos_half_theta = torch.abs(cos_half_theta)
        cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

        half_theta = torch.acos(cos_half_theta)
        sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)

        ratio_a = torch.sin((1 - blend) * half_theta) / sin_half_theta
        ratio_b = torch.sin(blend * half_theta) / sin_half_theta

        new_q_x = ratio_a * q0[..., qx : qx + 1] + ratio_b * q1[..., qx : qx + 1]
        new_q_y = ratio_a * q0[..., qy : qy + 1] + ratio_b * q1[..., qy : qy + 1]
        new_q_z = ratio_a * q0[..., qz : qz + 1] + ratio_b * q1[..., qz : qz + 1]
        new_q_w = ratio_a * q0[..., qw : qw + 1] + ratio_b * q1[..., qw : qw + 1]

        new_q = torch.cat(
            [new_q_w, new_q_x, new_q_y, new_q_z], dim=len(new_q_w.shape) - 1
        )
        new_q = torch.where(
            torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q
        )
        new_q = torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)
        return new_q

    def _compute_frame_blend(
        self, times: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the indexes of the first and second values, as well as the blending time
        to interpolate between them and the given times.

        Args:
            times: Times, between 0 and motion duration, to sample motion values.
                Specified times will be clipped to fall within the range of the motion duration.

        Returns:
            First value indexes, Second value indexes, and blending time between 0 (first value) and 1 (second value).
        """
        phase = np.clip(times / self.duration, 0.0, 1.0)
        index_0 = (phase * (self.num_frames - 1)).round(decimals=0).astype(int)
        index_1 = np.minimum(index_0 + 1, self.num_frames - 1)
        blend = ((times - index_0 * self.dt) / self.dt).round(decimals=5)
        return index_0, index_1, blend

    def sample_times(
        self, num_samples: int, duration: float | None = None
    ) -> np.ndarray:
        """Sample random motion times uniformly.

        Args:
            num_samples: Number of time samples to generate.
            duration: Maximum motion duration to sample.
                If not defined samples will be within the range of the motion duration.

        Raises:
            AssertionError: If the specified duration is longer than the motion duration.

        Returns:
            Time samples, between 0 and the specified/motion duration.
        """
        duration = self.duration if duration is None else duration
        assert (
            duration <= self.duration
        ), f"The specified duration ({duration}) is longer than the motion duration ({self.duration})"
        return duration * np.random.uniform(low=0.0, high=1.0, size=num_samples)

    def sample(
        self,
        num_samples: int,
        times: Optional[np.ndarray] = None,
        duration: float | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Sample motion data.

        Args:
            num_samples: Number of time samples to generate. If ``times`` is defined, this parameter is ignored.
            times: Motion time used for sampling.
                If not defined, motion data will be random sampled uniformly in time.
            duration: Maximum motion duration to sample.
                If not defined, samples will be within the range of the motion duration.
                If ``times`` is defined, this parameter is ignored.

        Returns:
            Sampled motion DOF positions (with shape (N, num_dofs)), DOF velocities (with shape (N, num_dofs)),
            body positions (with shape (N, num_bodies, 3)), body rotations (with shape (N, num_bodies, 4), as wxyz quaternion),
            body linear velocities (with shape (N, num_bodies, 3)) and body angular velocities (with shape (N, num_bodies, 3)).
        """
        times = self.sample_times(num_samples, duration) if times is None else times
        index_0, index_1, blend = self._compute_frame_blend(times)
        blend = torch.tensor(blend, dtype=torch.float32, device=self.device)

        return (
            self._interpolate(
                self.dof_positions, blend=blend, start=index_0, end=index_1
            ),
            self._interpolate(
                self.dof_velocities, blend=blend, start=index_0, end=index_1
            ),
            self._interpolate(
                self.body_positions, blend=blend, start=index_0, end=index_1
            ),
            self._slerp(self.body_rotations, blend=blend, start=index_0, end=index_1),
            self._interpolate(
                self.body_linear_velocities, blend=blend, start=index_0, end=index_1
            ),
            self._interpolate(
                self.body_angular_velocities, blend=blend, start=index_0, end=index_1
            ),
        )

    def get_dof_index(self, dof_names: list[str]) -> list[int]:
        """Get skeleton DOFs indexes by DOFs names.

        Args:
            dof_names: List of DOFs names.

        Raises:
            AssertionError: If the specified DOFs name doesn't exist.

        Returns:
            List of DOFs indexes.
        """
        indexes = []
        for name in dof_names:
            assert (
                name in self._dof_names
            ), f"The specified DOF name ({name}) doesn't exist: {self._dof_names}"
            indexes.append(self._dof_names.index(name))
        return indexes

    def get_body_index(self, body_names: list[str]) -> list[int]:
        """Get skeleton body indexes by body names.

        Args:
            dof_names: List of body names.

        Raises:
            AssertionError: If the specified body name doesn't exist.

        Returns:
            List of body indexes.
        """
        indexes = []
        for name in body_names:
            assert (
                name in self._body_names
            ), f"The specified body name ({name}) doesn't exist: {self._body_names}"
            indexes.append(self._body_names.index(name))
        return indexes


class MultiMotionLoader:
    """
    Helper class to load multiple motions and sample frames from them based on a fixed probability distribution.
    Each motion is loaded as an instance of MotionLoader.
    """

    def __init__(
        self, motion_files: List[str], probabilities: List[float], device: torch.device
    ) -> None:
        """
        Args:
            motion_files: List of motion file paths. Each file corresponds to a motion type.
            probabilities: List of sampling probabilities for each motion type.
                           Its length must equal the number of motion files.
                           For example: [0.0, 0.25, 0.25, 0.5]
            device: The device to which to load the data.
        """
        assert len(motion_files) > 0, "At least one motion file must be provided."
        self.device = device
        self.motion_loaders = [
            MotionLoader(motion_file, device) for motion_file in motion_files
        ]
        self.num_motions = len(self.motion_loaders)
        self.dt = self.motion_loaders[0].dt  # all motion loaders have the same dt

        # Check and normalize probabilities.
        assert (
            len(probabilities) == self.num_motions
        ), f"Expected {self.num_motions} probabilities, got {len(probabilities)}."
        probs = np.array(probabilities, dtype=np.float32)
        if probs.sum() <= 0:
            raise ValueError("The sum of probabilities must be greater than zero.")
        self.probabilities = (probs / probs.sum()).tolist()

    def get_body_index(self, body_names: list[str]) -> list[int]:
        """Get skeleton body indexes by body names.

        Args:
            dof_names: List of body names.

        Raises:
            AssertionError: If the specified body name doesn't exist.

        Returns:
            List of body indexes.
        """
        return self.motion_loaders[0].get_body_index(body_names)

    def sample_motion_type_and_times(
        self, num_samples: int, duration: Optional[float] = None
    ) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
        """
        First step: Sample a motion type for each sample (using the fixed probability distribution)
        and then, for each motion type, sample time points using that motion loader's sample_times method.

        Args:
            num_samples: Total number of samples.
            duration: Maximum motion duration to sample. If None, each motion uses its full duration.

        Returns:
            A tuple containing:
              - times_dict: A dictionary mapping motion type index to an array of sampled times for that type.
              - chosen_motion_indices: A numpy array (shape: [num_samples]) with the chosen motion type for each sample.
        """
        # For each sample, choose a motion type based on self.probabilities.
        chosen_motion_indices = np.random.choice(
            self.num_motions, size=num_samples, p=self.probabilities
        )

        # Group global sample indices by motion type.
        indices_by_motion: Dict[int, List[int]] = {}
        for sample_idx, motion_idx in enumerate(chosen_motion_indices):
            indices_by_motion.setdefault(motion_idx, []).append(sample_idx)

        # For each motion type, sample time points.
        times_dict: Dict[int, np.ndarray] = {}
        for motion_idx, sample_indices in indices_by_motion.items():
            n_samples_motion = len(sample_indices)
            times = self.motion_loaders[motion_idx].sample_times(
                n_samples_motion, duration=duration
            )
            times_dict[motion_idx] = times

        return times_dict, chosen_motion_indices

    def sample_motion(
        self, chosen_motion_indices: np.ndarray, times_dict: Dict[int, np.ndarray]
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Second step: Given the chosen motion types and their time samples, group the samples by motion type,
        run the sampling in batches, and reassemble the final output.

        Args:
            chosen_motion_indices: A numpy array (shape: [num_samples]) with the chosen motion type for each sample.
            times_dict: A dictionary mapping each motion type to an array of time samples.

        Returns:
            A tuple containing:
              - dof_positions: (num_samples, num_dofs)
              - dof_velocities: (num_samples, num_dofs)
              - body_positions: (num_samples, num_bodies, 3)
              - body_rotations: (num_samples, num_bodies, 4) (as wxyz quaternions)
              - body_linear_velocities: (num_samples, num_bodies, 3)
              - body_angular_velocities: (num_samples, num_bodies, 3)
              - motion_onehot: (num_samples, num_motions) one-hot vector for the motion type of each sample.
        """
        # Group global sample indices by motion type.
        indices_by_motion: Dict[int, List[int]] = {}
        for sample_idx, motion_idx in enumerate(chosen_motion_indices):
            indices_by_motion.setdefault(motion_idx, []).append(sample_idx)

        num_total_samples = len(chosen_motion_indices)
        # Placeholder list to store sampled data for each sample.
        result: List[
            Tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ]
        ] = [None] * num_total_samples

        # For each motion type, use the corresponding time samples to get motion data.
        for motion_idx, sample_indices in indices_by_motion.items():
            n_samples_motion = len(sample_indices)
            times = times_dict[motion_idx]
            samples = self.motion_loaders[motion_idx].sample(
                n_samples_motion, times=times
            )
            # Place the sampled data into the global result list.
            for local_idx, final_idx in enumerate(sample_indices):
                sample_tuple = tuple(field[local_idx] for field in samples)
                result[final_idx] = sample_tuple

        # Reassemble outputs so that each field is batched.
        output_fields = []
        for field_idx in range(6):  # 6 fields from MotionLoader.sample
            field_samples = [sample[field_idx] for sample in result]
            field_tensor = torch.stack(field_samples, dim=0)
            output_fields.append(field_tensor)

        # Create a one-hot tensor encoding the motion type for each sample.
        motion_onehot = torch.nn.functional.one_hot(
            torch.tensor(chosen_motion_indices, device=self.device),
            num_classes=self.num_motions,
        ).float()

        return tuple(output_fields) + (motion_onehot,)

    def sample(self, num_samples: int, duration: Optional[float] = None) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Unified sampling: First, sample motion types and time samples; then, sample motion data accordingly.

        Args:
            num_samples: Total number of frames to sample.
            duration: Maximum motion duration to sample (if None, uses each motion's full duration).

        Returns:
            A tuple with:
              - dof_positions, dof_velocities, body_positions, body_rotations,
                body_linear_velocities, body_angular_velocities, motion_onehot.
        """
        times_dict, chosen_motion_indices = self.sample_motion_type_and_times(
            num_samples, duration
        )
        return self.sample_motion(chosen_motion_indices, times_dict)


# Assume MotionLoader is defined elsewhere and imported.
# from motions.motion_loader import MotionLoader


class TaskMultiMotionLoader:
    """
    Helper class to load multiple tasks, each with multiple motions, and sample frames
    based on a two-level (task and motion) fixed probability distribution.
    """

    def __init__(
        self,
        motion_files: List[
            List[Dict]
        ],  # Each task is a list of dicts with keys: "motion_file", "prob"
        task_probabilities: List[float],
        device: torch.device,
    ) -> None:
        """
        Args:
            motion_files: A list of tasks, where each task is a list of motion dictionaries.
                          For example, motion_files[task][i] is a dict with keys "motion_file" and "prob".
            task_probabilities: A list of probabilities for each task. Length must equal number of tasks.
            device: The device to load the motions onto.
        """
        assert len(motion_files) > 0, "At least one task must be provided."
        self.device = device
        self.num_tasks = len(motion_files)

        # Process each task: create a list of MotionLoader instances and record probabilities.
        self.motion_loaders_by_task: List[List[MotionLoader]] = []
        self.motion_probabilities_by_task: List[List[float]] = []
        self.num_motions_by_task: List[int] = []

        for task_motions in motion_files:
            assert len(task_motions) > 0, "Each task must have at least one motion."
            loaders = []
            probs = []
            for motion_cfg in task_motions:
                loaders.append(MotionLoader(motion_cfg["motion_file"], device))
                probs.append(float(motion_cfg["prob"]))
            # Normalize motion probabilities for the task.
            probs = np.array(probs, dtype=np.float32)
            if probs.sum() <= 0:
                raise ValueError("Motion probabilities must sum to >0.")
            probs = (probs / probs.sum()).tolist()
            self.motion_loaders_by_task.append(loaders)
            self.motion_probabilities_by_task.append(probs)
            self.num_motions_by_task.append(len(loaders))

        # Normalize task probabilities.
        task_probs = np.array(task_probabilities, dtype=np.float32)
        if task_probs.sum() <= 0:
            raise ValueError("Task probabilities must sum to >0.")
        self.task_probabilities = (task_probs / task_probs.sum()).tolist()

        # For dt, assume all motions share the same dt; take it from the first task's first motion.
        self.dt = self.motion_loaders_by_task[0][0].dt

    def get_body_index(self, body_names: List[str]) -> List[int]:
        """Returns body indices by querying the first task's first motion loader."""
        return self.motion_loaders_by_task[0][0].get_body_index(body_names)

    def sample_task_motion_and_times(
        self, num_samples: int, duration: Optional[float] = None
    ) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Hierarchical sampling:
          1. For each sample, first sample a task index using self.task_probabilities.
          2. Within each task, sample a motion index using that task's motion probabilities.
          3. For each unique (task, motion) pair, sample time points using the corresponding motion loader.

        Returns:
            times_dict: Dictionary mapping (task, motion) -> np.ndarray of sampled times.
            chosen_task_indices: Array of shape (num_samples,) with the sampled task index per sample.
            chosen_motion_indices: Array of shape (num_samples,) with the sampled motion index (within the task) per sample.
        """
        # Sample tasks for each sample.
        chosen_task_indices = np.random.choice(
            self.num_tasks, size=num_samples, p=self.task_probabilities
        )
        times_dict: Dict[Tuple[int, int], np.ndarray] = {}

        # Process each task.
        for task in range(self.num_tasks):
            task_sample_indices = np.where(chosen_task_indices == task)[0]
            if task_sample_indices.size == 0:
                continue
            num_task_samples = len(task_sample_indices)
            # Sample motions for samples that chose this task.
            motion_probs = np.array(
                self.motion_probabilities_by_task[task], dtype=np.float32
            )
            motion_probs = motion_probs / motion_probs.sum()  # ensure normalization
            num_motions = self.num_motions_by_task[task]
            chosen_motions = np.random.choice(
                num_motions, size=num_task_samples, p=motion_probs
            )

            # For each unique motion in this task, sample times.
            unique_motions = np.unique(chosen_motions)
            for motion in unique_motions:
                indices_in_task = task_sample_indices[chosen_motions == motion]
                n_samples_motion = len(indices_in_task)
                times = self.motion_loaders_by_task[task][motion].sample_times(
                    n_samples_motion, duration=duration
                )
                times_dict[(task, motion)] = times

        return times_dict

    def sample_motion(self, times_dict: Dict[Tuple[int, int], np.ndarray]) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Second-stage sampling:
          1. For each unique (task, motion) pair in times_dict, obtain the corresponding time samples
             and sample motion data.
          2. Concatenate all results (the sample order will be based on the dictionary order).
          3. Create one-hot encoding for tasks (B x num_tasks) based on the task information.

        Returns:
            A tuple containing:
              - dof_positions, dof_velocities, body_positions, body_rotations,
                body_linear_velocities, body_angular_velocities (each concatenated along batch dimension)
              - task_onehot: Tensor of shape (total_samples, num_tasks)
        """
        dof_positions_list = []
        dof_velocities_list = []
        body_positions_list = []
        body_rotations_list = []
        body_linear_velocities_list = []
        body_angular_velocities_list = []
        task_assignment_list = []

        # Iterate over each unique (task, motion) pair.
        for (task, motion), times in times_dict.items():
            n_samples = len(times)
            # Sample motion data using the corresponding motion loader.
            samples = self.motion_loaders_by_task[task][motion].sample(
                n_samples, times=times
            )
            dof_positions_list.append(samples[0])
            dof_velocities_list.append(samples[1])
            body_positions_list.append(samples[2])
            body_rotations_list.append(samples[3])
            body_linear_velocities_list.append(samples[4])
            body_angular_velocities_list.append(samples[5])
            # Record the task assignment for these samples.
            task_assignment_list.append(
                torch.full((n_samples,), task, dtype=torch.long, device=self.device)
            )

        # Concatenate all sampled data.
        dof_positions = torch.cat(dof_positions_list, dim=0)
        dof_velocities = torch.cat(dof_velocities_list, dim=0)
        body_positions = torch.cat(body_positions_list, dim=0)
        body_rotations = torch.cat(body_rotations_list, dim=0)
        body_linear_velocities = torch.cat(body_linear_velocities_list, dim=0)
        body_angular_velocities = torch.cat(body_angular_velocities_list, dim=0)
        task_assignment = torch.cat(task_assignment_list, dim=0)

        # Create one-hot encoding for tasks.

        return (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
            task_assignment,
        )

    def sample(self, num_samples: int, duration: Optional[float] = None) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Unified hierarchical sampling: First, sample task and motion types along with time samples,
        then sample the motion data accordingly.

        Returns:
            A tuple containing:
              - dof_positions, dof_velocities, body_positions, body_rotations,
                body_linear_velocities, body_angular_velocities, task_onehot.
        """
        times_dict = self.sample_task_motion_and_times(num_samples, duration)
        return self.sample_motion(times_dict)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Motion file")
    args, _ = parser.parse_known_args()

    motion = MotionLoader(args.file, "cpu")

    print("- number of frames:", motion.num_frames)
    print("- number of DOFs:", motion.num_dofs)
    print("- number of bodies:", motion.num_bodies)

    multi_motion = MultiMotionLoader(
        [args.file, args.file, args.file], [0.75, 0.125, 0.125], "cpu"
    )
