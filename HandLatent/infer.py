"""Inference CLI for real trajectory retargeting and visualization."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import rerun as rr
import torch
from scipy.spatial.transform import Rotation as scipy_rotation

from HandLatent.ik import pink_align_arm
from HandLatent.model import (
    CrossEmbodimentTrainer,
    TrainerCacheState,
    TrainingConfig,
    clone_default_arm_cache_pose,
    compute_pinch_loss,
)
from HandLatent.visualize import visualize_hand_motion


@dataclass
class EvaluationConfig:
    """Inference-time IK hyper-parameters.

    Parameters
    ----------
    ik_pink_arm_initial_iterations : int
        Pink iterations for the first frame when no cache exists, shape=().
    ik_pink_arm_iterations : int
        Pink iterations for subsequent frames, shape=().
    ik_rotation_weight : float
        Wrist rotation cost weight, shape=().
    """

    ik_pink_arm_initial_iterations: int = 50
    ik_pink_arm_iterations: int = 5
    ik_rotation_weight: float = 0.01


def _normalize_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    """Normalize quaternion tensors and make scalar component non-negative.

    Parameters
    ----------
    quaternion : torch.Tensor, shape=(B, 4) or (4,), dtype=float32
        Quaternion in ``(w, x, y, z)`` convention.

    Returns
    -------
    torch.Tensor, shape matching input, dtype matching input
        Normalized quaternion tensor with non-negative ``w``.
    """

    squeeze = quaternion.ndim == 1
    batch = quaternion.unsqueeze(0) if squeeze else quaternion
    batch_cpu = batch.detach().to(device="cpu", dtype=torch.float64)
    xyzw = torch.cat([batch_cpu[:, 1:], batch_cpu[:, :1]], dim=1).numpy()
    normalized_xyzw = scipy_rotation.from_quat(xyzw).as_quat()
    wxyz = np.concatenate([normalized_xyzw[:, 3:4], normalized_xyzw[:, :3]], axis=1)
    sign = np.where(wxyz[:, :1] < 0.0, -1.0, 1.0)
    normalized = torch.from_numpy(wxyz * sign).to(
        device=batch.device, dtype=batch.dtype
    )
    return normalized[0] if squeeze else normalized


def _rotation_matrix_to_quaternion(rotation: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to quaternions in ``(w, x, y, z)`` order.

    Parameters
    ----------
    rotation : torch.Tensor, shape=(B, 3, 3) or (3, 3), dtype=float32
        Proper rotation matrices.

    Returns
    -------
    torch.Tensor, shape=(B, 4) or (4,), dtype matching input
        Normalized quaternions in ``(w, x, y, z)`` order.
    """

    squeeze = rotation.ndim == 2
    batch = rotation.unsqueeze(0) if squeeze else rotation
    matrices = batch.detach().to(device="cpu", dtype=torch.float64).numpy()
    xyzw = scipy_rotation.from_matrix(matrices).as_quat()
    wxyz = np.concatenate([xyzw[:, 3:4], xyzw[:, :3]], axis=1)
    sign = np.where(wxyz[:, :1] < 0.0, -1.0, 1.0)
    output = torch.from_numpy(wxyz * sign).to(device=batch.device, dtype=batch.dtype)
    return output[0] if squeeze else output


def _quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert quaternions in ``(w, x, y, z)`` order to rotation matrices.

    Parameters
    ----------
    quaternion : torch.Tensor, shape=(B, 4) or (4,), dtype=float32
        Quaternion tensor.

    Returns
    -------
    torch.Tensor, shape=(B, 3, 3) or (3, 3), dtype matching input
        Rotation matrices.
    """

    squeeze = quaternion.ndim == 1
    batch = quaternion.unsqueeze(0) if squeeze else quaternion
    batch_cpu = batch.detach().to(device="cpu", dtype=torch.float64)
    quaternion_xyzw = torch.cat([batch_cpu[:, 1:], batch_cpu[:, :1]], dim=1).numpy()
    matrices = scipy_rotation.from_quat(quaternion_xyzw).as_matrix()
    output = torch.from_numpy(matrices).to(device=batch.device, dtype=batch.dtype)
    return output[0] if squeeze else output


def compute_alignment_points(
    fingertip_positions: torch.Tensor,
    pinch_pairs: Sequence[Tuple[int, int]],
    weights: torch.Tensor,
) -> torch.Tensor:
    """Compute weighted pinch midpoints for each frame.

    Parameters
    ----------
    fingertip_positions : torch.Tensor, shape=(B, F, 3), dtype=float32
        Fingertip coordinates.
    pinch_pairs : Sequence[Tuple[int, int]], shape=(P, 2)
        Pinch pair indices.
    weights : torch.Tensor, shape=(B, P), dtype=float32
        Pair weights.

    Returns
    -------
    torch.Tensor, shape=(B, 3), dtype=float32
        Weighted midpoint coordinates.
    """

    if len(pinch_pairs) == 0 or weights.numel() == 0:
        return torch.zeros(
            fingertip_positions.shape[0],
            3,
            device=fingertip_positions.device,
            dtype=fingertip_positions.dtype,
        )
    pair_indices = torch.tensor(
        pinch_pairs, device=fingertip_positions.device, dtype=torch.long
    )
    first_indices = pair_indices[:, 0]
    second_indices = pair_indices[:, 1]
    midpoints = 0.5 * (
        fingertip_positions[:, first_indices, :]
        + fingertip_positions[:, second_indices, :]
    )
    normalized_weights = weights.to(
        device=fingertip_positions.device, dtype=fingertip_positions.dtype
    )
    normalizer = torch.clamp_min(normalized_weights.sum(dim=1, keepdim=True), 1.0e-8)
    return (midpoints * normalized_weights.unsqueeze(-1)).sum(dim=1) / normalizer


def encode_hand_sequence_eepose(
    trainer: CrossEmbodimentTrainer,
    hand_name: str,
    qpos: torch.Tensor,
) -> torch.Tensor:
    """Encode normalized qpos into EEPose latent representation.

    Parameters
    ----------
    trainer : CrossEmbodimentTrainer
        Trainer with loaded autoencoders and FK models.
    hand_name : str
        Source hand name with shape=().
    qpos : torch.Tensor, shape=(B, D), dtype=float32
        Normalized source qpos trajectory.

    Returns
    -------
    torch.Tensor, shape=(B, 7 + latent_dim_hand), dtype=float32
        Concatenated ``[alignment(3), wrist_quaternion(4), hand_latent]``.
    """

    cache_state = trainer.encode_state()
    arm_states, _ = trainer._split_qpos(hand_name, qpos)
    model = trainer.hand_models[hand_name]
    pinch_pairs = list(trainer.pinch_pairs_for_hand(hand_name))

    with torch.no_grad():
        _, mean_hand, _ = trainer.autoencoders[hand_name].encode(qpos)
        fingertip_positions, wrist_pose = model.forward_with_wrist_pose(qpos)
        wrist_rotation = wrist_pose[:, :3, :3]
        wrist_quaternion = _rotation_matrix_to_quaternion(wrist_rotation)
        _, _, weights = compute_pinch_loss(
            fingertip_positions,
            fingertip_positions,
            pinch_pairs,
            trainer.config.lambda_dis_exp,
        )
        alignment = compute_alignment_points(fingertip_positions, pinch_pairs, weights)

    if arm_states.shape[0] > 0:
        cache_state.eepose_encode_arm = (
            arm_states[-1].detach().to(device="cpu", dtype=torch.float32).clone()
        )

    return torch.cat(
        [alignment.detach(), wrist_quaternion.detach(), mean_hand.detach()], dim=1
    )


def decode_hand_sequence_eepose(
    trainer: CrossEmbodimentTrainer,
    hand_name: str,
    latents: torch.Tensor,
    evaluation_config: Optional[EvaluationConfig] = None,
    decode_state: Optional[TrainerCacheState] = None,
) -> torch.Tensor:
    """Decode EEPose latents to normalized qpos with Pink IK arm solve.

    Parameters
    ----------
    trainer : CrossEmbodimentTrainer
        Trainer with loaded autoencoders and FK models.
    hand_name : str
        Target hand name with shape=().
    latents : torch.Tensor, shape=(B, 7 + latent_dim_hand), dtype=float32
        EEPose latent sequence.
    evaluation_config : EvaluationConfig or None
        Optional IK config.
    decode_state : TrainerCacheState or None
        Optional decode cache override.

    Returns
    -------
    torch.Tensor, shape=(B, D_target), dtype=float32
        Decoded normalized target qpos sequence.
    """

    cache_state = (
        decode_state if decode_state is not None else trainer.decode_state(hand_name)
    )
    eval_config = (
        evaluation_config if evaluation_config is not None else EvaluationConfig()
    )
    device = trainer.config.device

    latents_device = latents.to(device=device, dtype=torch.float32)
    target_alignment = latents_device[:, :3]
    target_quaternion = _normalize_quaternion(latents_device[:, 3:7])
    target_rotations = _quaternion_to_rotation_matrix(target_quaternion)
    latent_hand = latents_device[:, 7:]

    autoencoder = trainer.autoencoders[hand_name]
    with torch.no_grad():
        qpos_hand = autoencoder.hand_decoder(latent_hand)

    cached_arm = (
        cache_state.eepose_decode_arm.to(device=device, dtype=qpos_hand.dtype)
        .detach()
        .clone()
        if cache_state.eepose_decode_arm is not None
        else None
    )
    model = trainer.hand_models[hand_name]
    pinch_pairs = list(trainer.pinch_pairs_for_hand(hand_name))
    batch = latents_device.shape[0]
    arm_solutions = torch.zeros(
        batch, trainer.arm_dof, dtype=qpos_hand.dtype, device=device
    )
    previous_arm = None if cached_arm is None else cached_arm.detach().clone()
    default_arm_seed = clone_default_arm_cache_pose().to(
        device=device, dtype=qpos_hand.dtype
    )

    for frame_index in range(batch):
        hand_frame = qpos_hand[frame_index]
        arm_seed = default_arm_seed if previous_arm is None else previous_arm
        combined_seed = trainer._merge_qpos(
            arm_seed.unsqueeze(0), hand_frame.unsqueeze(0)
        )
        tips_seed, _ = model.forward_with_wrist_pose(combined_seed)
        _, _, weights_seed = compute_pinch_loss(
            tips_seed,
            tips_seed,
            pinch_pairs,
            trainer.config.lambda_dis_exp,
        )
        solved = pink_align_arm(
            hand_name=hand_name,
            model=model,
            arm_seed=arm_seed,
            hand_fixed=hand_frame,
            target_alignment=target_alignment[frame_index],
            target_rotation=target_rotations[frame_index],
            pinch_pairs=pinch_pairs,
            pair_weights=weights_seed[0],
            rotation_weight=eval_config.ik_rotation_weight,
            iterations=eval_config.ik_pink_arm_iterations
            if (cached_arm is not None or frame_index > 0)
            else eval_config.ik_pink_arm_initial_iterations,
        )
        arm_solutions[frame_index] = solved
        previous_arm = solved.detach()

    combined = trainer._merge_qpos(arm_solutions, qpos_hand.detach())
    if batch > 0:
        cache_state.eepose_decode_arm = (
            arm_solutions[-1].detach().to(device="cpu", dtype=torch.float32).clone()
        )
    return combined.detach().clone()


def _find_latest_checkpoint(checkpoint_root: Path) -> Path:
    """Find latest checkpoint by lexicographic path order.

    Parameters
    ----------
    checkpoint_root : pathlib.Path
        Checkpoint root directory path with shape=().

    Returns
    -------
    pathlib.Path, shape=()
        Latest ``checkpoint_epoch_*.pt`` path.
    """

    candidates = sorted(checkpoint_root.rglob("checkpoint_epoch_*.pt"))
    return candidates[-1]


def main() -> None:
    """Run inference CLI on real trajectory and visualize origin plus four decodes.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Starts a Rerun session and logs trajectory playback.
    """

    parser = argparse.ArgumentParser(
        description="Retarget one real trajectory to four xarm hand embodiments."
    )
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--data", type=str, default="Dataset/demo.npz")
    parser.add_argument(
        "--side", type=str, default="both", choices=("right", "left", "both")
    )
    args = parser.parse_args()

    sides = ("right", "left") if args.side == "both" else (args.side,)
    trainer_hands = [
        hand_name
        for side in sides
        for hand_name in (
            f"xarm7_xhand_{side}",
            f"xarm7_ability_{side}",
            f"xarm7_inspire_{side}",
            f"xarm7_paxini_{side}",
        )
    ]

    config = TrainingConfig()
    trainer = CrossEmbodimentTrainer(trainer_hands, config)
    ckpt_path = (
        Path(args.ckpt).expanduser().resolve()
        if args.ckpt is not None
        else _find_latest_checkpoint(Path(config.checkpoint_dir))
    )
    payload = torch.load(ckpt_path, map_location="cpu")
    trainer.load_autoencoders_from_payload(payload)

    recording_name = f"hand_latent_real_data_{args.side}"
    rr.init(recording_name, spawn=True)
    recording = rr.get_global_data_recording()

    with np.load(Path(args.data).expanduser().resolve()) as dataset:
        for side in sides:
            source_hand = f"xarm7_inspire_{side}"
            target_hands = [
                f"xarm7_xhand_{side}",
                f"xarm7_ability_{side}",
                f"xarm7_inspire_{side}",
                f"xarm7_paxini_{side}",
            ]
            source_qpos = torch.as_tensor(dataset[f"{side}_qpos"], dtype=torch.float32)
            source_norm = trainer.normalized_qpos(source_hand, source_qpos).to(
                device=config.device
            )

            with torch.no_grad():
                latents = encode_hand_sequence_eepose(
                    trainer=trainer, hand_name=source_hand, qpos=source_norm
                )
                decoded = {
                    hand_name: decode_hand_sequence_eepose(
                        trainer=trainer,
                        hand_name=hand_name,
                        latents=latents,
                        evaluation_config=EvaluationConfig(),
                    )
                    .detach()
                    .cpu()
                    for hand_name in target_hands
                }

            side_offset = np.array(
                [0.0, 0.0 if side == "right" else 0.8, 0.0], dtype=np.float32
            )
            overlap_offsets = np.repeat(
                side_offset.reshape(1, 3), source_norm.shape[0], axis=0
            )
            source_array = source_norm.detach().cpu().numpy()
            visualize_hand_motion(
                hand_name=source_hand,
                joint_series=source_array,
                recording_name=recording_name,
                recording=recording,
                entity_path_prefix=f"{source_hand}_origin",
                per_frame_root_offsets=overlap_offsets,
            )

            for hand_name in target_hands:
                series = decoded[hand_name].numpy()
                visualize_hand_motion(
                    hand_name=hand_name,
                    joint_series=series,
                    recording_name=recording_name,
                    recording=recording,
                    entity_path_prefix=f"{hand_name}_decode",
                    per_frame_root_offsets=overlap_offsets,
                )


if __name__ == "__main__":
    main()
