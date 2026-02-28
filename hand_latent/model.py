"""Latent model and trainer for cross-embodiment xarm hand retargeting."""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from hand_latent.kinematics import MultiHandDifferentiableFK, solve_inverse_kinematics

PINCH_PAIR_DEFAULTS: Tuple[Tuple[int, int], ...] = ((0, 1), (0, 2), (0, 3), (0, 4))
DEFAULT_ARM_CACHE_POSE: torch.Tensor = torch.tensor(
    [0.0, 0.0, 0.0, -0.89, 0.5, 0.25, 0.0],
    dtype=torch.float32,
)


def clone_default_arm_cache_pose() -> torch.Tensor:
    """Return a detached copy of the default normalized arm pose.

    Parameters
    ----------
    None

    Returns
    -------
    torch.Tensor, shape=(7,), dtype=float32
        Default arm seed used by inference-time IK caches.
    """

    return DEFAULT_ARM_CACHE_POSE.detach().clone()


@dataclass
class TrainingConfig:
    """Training hyper-parameters for the minimal latent project.

    Parameters
    ----------
    device : torch.device
        Runtime device with shape=().
    latent_dim_hand : int
        Hand latent width with shape=().
    hand_hidden_dims : Sequence[int], shape=(L,)
        MLP hidden widths for encoder/decoder.
    batch_size : int
        Batch size with shape=().
    num_steps : int
        Number of optimizer steps with shape=().
    learning_rate : float
        Optimizer learning rate with shape=().
    rec_hand_weight : float
        Hand reconstruction loss weight with shape=().
    arm_dof : int
        Number of leading arm joints with shape=().
    lambda_dis : float
        Distance pinch loss weight with shape=().
    lambda_dir : float
        Direction pinch loss weight with shape=().
    lambda_dis_exp : float
        Exponential weighting coefficient with shape=().
    lambda_kl : float
        KL weight with shape=().
    pinch_pairs : Sequence[Tuple[int, int]], shape=(P, 2)
        Pinch fingertip index pairs.
    pinch_sampling_probability : float
        Probability of pinch-template samples with shape=().
    pinch_offset : Tuple[float, float, float], shape=(3,)
        Wrist-frame pinch target offset in meters.
    pinch_template_target_noise_std : float
        Target point noise standard deviation with shape=().
    pinch_template_joint_noise_std : float
        Joint noise standard deviation with shape=().
    pinch_template_iterations : int
        IK iterations for template generation with shape=().
    pinch_template_learning_rate : float
        IK learning rate for template generation with shape=().
    pinch_template_count : int
        Number of cached pinch templates with shape=().
    checkpoint_dir : str
        Directory path used for checkpoint storage with shape=().
    checkpoint_interval : int
        Save interval in steps with shape=().
    """

    device: torch.device = (
        torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
    )
    latent_dim_hand: int = 32
    hand_hidden_dims: Sequence[int] = (64, 128, 64)
    batch_size: int = 1024
    num_steps: int = 10_000
    learning_rate: float = 2e-3
    rec_hand_weight: float = 1.0
    arm_dof: int = 7
    lambda_dis: float = 2000.0
    lambda_dir: float = 5.0
    lambda_dis_exp: float = 12.0
    lambda_kl: float = 0.0
    pinch_pairs: Sequence[Tuple[int, int]] = field(default_factory=lambda: list(PINCH_PAIR_DEFAULTS))
    pinch_sampling_probability: float = 0.5
    pinch_offset: Tuple[float, float, float] = (0.07, 0.0, -0.08)
    pinch_template_target_noise_std: float = 0.01
    pinch_template_joint_noise_std: float = 0.01
    pinch_template_iterations: int = 100
    pinch_template_learning_rate: float = 0.05
    pinch_template_count: int = 2048
    checkpoint_dir: str = field(
        default_factory=lambda: os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Checkpoints"))
    )
    checkpoint_interval: int = 500


@dataclass
class TrainerCacheState:
    """Runtime cache container for encode/decode seeds.

    Parameters
    ----------
    eepose_encode_arm : torch.Tensor or None, shape=(7,), dtype=float32
        Last arm command used by EEPose encoder.
    eepose_decode_arm : torch.Tensor or None, shape=(7,), dtype=float32
        Last arm command solved by EEPose decoder.
    """

    eepose_encode_arm: Optional[torch.Tensor] = field(default_factory=clone_default_arm_cache_pose)
    eepose_decode_arm: Optional[torch.Tensor] = field(default_factory=clone_default_arm_cache_pose)

    def reset(self) -> None:
        """Reset cached arm seeds to defaults.

        Parameters
        ----------
        self : TrainerCacheState, shape=()
            Cache state object.

        Returns
        -------
        None
            Resets fields in-place.
        """

        self.eepose_encode_arm = clone_default_arm_cache_pose()
        self.eepose_decode_arm = clone_default_arm_cache_pose()


class HandAutoencoder(nn.Module):
    """Hand autoencoder with arm pass-through and VAE-style hand heads.

    Parameters
    ----------
    arm_dof : int
        Arm dof count with shape=().
    hand_dof : int
        Hand dof count with shape=().
    latent_dim_hand : int
        Hand latent width with shape=().
    hand_hidden_dims : Sequence[int], shape=(L,)
        Hidden widths for hand encoder/decoder MLP.
    """

    def __init__(
        self,
        arm_dof: int,
        hand_dof: int,
        latent_dim_hand: int,
        hand_hidden_dims: Sequence[int],
    ) -> None:
        """Build hand encoder/decoder networks.

        Parameters
        ----------
        arm_dof : int
            Arm dof count with shape=().
        hand_dof : int
            Hand dof count with shape=().
        latent_dim_hand : int
            Hand latent width with shape=().
        hand_hidden_dims : Sequence[int], shape=(L,)
            Hidden widths for MLP backbones.

        Returns
        -------
        None
            Initializes module parameters.
        """

        super().__init__()
        self.arm_dof = int(arm_dof)
        self.hand_dof = int(hand_dof)
        self.latent_dim_hand = int(latent_dim_hand)
        self.hand_encoder_backbone = self._make_mlp(self.hand_dof, hand_hidden_dims)
        hand_backbone_dim = self._infer_last_width(self.hand_dof, hand_hidden_dims)
        self.hand_mean_head = nn.Linear(hand_backbone_dim, self.latent_dim_hand)
        self.hand_logvar_head = nn.Linear(hand_backbone_dim, self.latent_dim_hand)
        self.hand_decoder = self._make_mlp(
            self.latent_dim_hand,
            hand_hidden_dims,
            output_dim=self.hand_dof,
            final_activation=nn.Tanh(),
        )

    @staticmethod
    def _infer_last_width(input_dim: int, hidden_dims: Sequence[int]) -> int:
        """Return terminal MLP width.

        Parameters
        ----------
        input_dim : int
            Input width with shape=().
        hidden_dims : Sequence[int], shape=(L,)
            Hidden widths.

        Returns
        -------
        int
            Output width of final hidden layer with shape=().
        """

        return hidden_dims[-1] if hidden_dims else input_dim

    @staticmethod
    def _make_mlp(
        input_dim: int,
        hidden_dims: Sequence[int],
        *,
        output_dim: Optional[int] = None,
        final_activation: Optional[nn.Module] = None,
    ) -> nn.Sequential:
        """Construct an MLP with LayerNorm and ReLU.

        Parameters
        ----------
        input_dim : int
            Input width with shape=().
        hidden_dims : Sequence[int], shape=(L,)
            Hidden widths.
        output_dim : int or None
            Optional output width with shape=().
        final_activation : nn.Module or None
            Optional final activation module.

        Returns
        -------
        torch.nn.Sequential, shape=()
            Constructed MLP module.
        """

        layers: List[nn.Module] = []
        current_dim = input_dim
        for width in hidden_dims:
            layers.append(nn.Linear(current_dim, width))
            layers.append(nn.LayerNorm(width))
            layers.append(nn.ReLU())
            current_dim = width
        if output_dim is not None:
            layers.append(nn.Linear(current_dim, output_dim))
            if final_activation is not None:
                layers.append(final_activation)
        return nn.Sequential(*layers)

    def _split_qpos(self, qpos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split concatenated qpos into arm and hand segments.

        Parameters
        ----------
        qpos : torch.Tensor, shape=(B, arm_dof + hand_dof), dtype=float32
            Concatenated normalized joints.

        Returns
        -------
        tuple
            - arm : torch.Tensor, shape=(B, arm_dof), dtype=float32
            - hand : torch.Tensor, shape=(B, hand_dof), dtype=float32
        """

        return qpos[..., : self.arm_dof], qpos[..., self.arm_dof :]

    def encode(self, qpos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode qpos into arm latent and hand posterior stats.

        Parameters
        ----------
        qpos : torch.Tensor, shape=(B, arm_dof + hand_dof), dtype=float32
            Concatenated normalized joints.

        Returns
        -------
        tuple
            - latent_arm : torch.Tensor, shape=(B, arm_dof), dtype=float32
            - mean_hand : torch.Tensor, shape=(B, latent_dim_hand), dtype=float32
            - logvar_hand : torch.Tensor, shape=(B, latent_dim_hand), dtype=float32
        """

        latent_arm, qpos_hand = self._split_qpos(qpos)
        hand_features = self.hand_encoder_backbone(qpos_hand)
        mean_hand = self.hand_mean_head(hand_features)
        logvar_hand = self.hand_logvar_head(hand_features)
        return latent_arm, mean_hand, logvar_hand

    @staticmethod
    def reparameterize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample latent vectors via reparameterization.

        Parameters
        ----------
        mean : torch.Tensor, shape=(B, L), dtype=float32
            Posterior mean.
        logvar : torch.Tensor, shape=(B, L), dtype=float32
            Posterior log variance.

        Returns
        -------
        torch.Tensor, shape=(B, L), dtype=float32
            Sampled latent vectors.
        """

        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mean + epsilon * std

    def decode_from_latents(self, latent_arm: torch.Tensor, latent_hand: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode arm and hand latents into normalized joints.

        Parameters
        ----------
        latent_arm : torch.Tensor, shape=(B, arm_dof), dtype=float32
            Arm latent vectors.
        latent_hand : torch.Tensor, shape=(B, latent_dim_hand), dtype=float32
            Hand latent vectors.

        Returns
        -------
        tuple
            - qpos_arm : torch.Tensor, shape=(B, arm_dof), dtype=float32
            - qpos_hand : torch.Tensor, shape=(B, hand_dof), dtype=float32
        """

        return latent_arm, self.hand_decoder(latent_hand)

    def forward(
        self, qpos: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with deterministic hand latent mean.

        Parameters
        ----------
        qpos : torch.Tensor, shape=(B, arm_dof + hand_dof), dtype=float32
            Concatenated normalized joints.

        Returns
        -------
        tuple
            - latent_arm : torch.Tensor, shape=(B, arm_dof), dtype=float32
            - latent_hand : torch.Tensor, shape=(B, latent_dim_hand), dtype=float32
            - qpos_arm : torch.Tensor, shape=(B, arm_dof), dtype=float32
            - qpos_hand : torch.Tensor, shape=(B, hand_dof), dtype=float32
            - hand_stats : tuple[torch.Tensor, torch.Tensor], shapes=((B, latent_dim_hand), (B, latent_dim_hand))
        """

        latent_arm, mean_hand, logvar_hand = self.encode(qpos)
        latent_hand = mean_hand
        qpos_arm, qpos_hand = self.decode_from_latents(latent_arm, latent_hand)
        return latent_arm, latent_hand, qpos_arm, qpos_hand, (mean_hand, logvar_hand)


def compute_pinch_loss(
    source_tips: torch.Tensor,
    target_tips: torch.Tensor,
    pinch_pairs: Sequence[Tuple[int, int]],
    lambda_dis_exp: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute pinch distance and direction losses plus exponential weights.

    Parameters
    ----------
    source_tips : torch.Tensor, shape=(B, F, 3), dtype=float32
        Source fingertip coordinates.
    target_tips : torch.Tensor, shape=(B, F, 3), dtype=float32
        Target fingertip coordinates.
    pinch_pairs : Sequence[Tuple[int, int]], shape=(P, 2)
        Pinch pair indices.
    lambda_dis_exp : float
        Distance exponential scale with shape=().

    Returns
    -------
    tuple
        - distance_term : torch.Tensor, shape=(B, P), dtype=float32
        - direction_term : torch.Tensor, shape=(B, P), dtype=float32
        - weight : torch.Tensor, shape=(B, P), dtype=float32
    """

    if len(pinch_pairs) == 0:
        empty = torch.zeros(source_tips.shape[0], 0, device=source_tips.device, dtype=source_tips.dtype)
        return empty, empty, empty

    pair_indices = torch.tensor(pinch_pairs, device=source_tips.device, dtype=torch.long)
    first_indices = pair_indices[:, 0]
    second_indices = pair_indices[:, 1]
    delta_source = source_tips[:, first_indices, :] - source_tips[:, second_indices, :]
    delta_target = target_tips[:, first_indices, :] - target_tips[:, second_indices, :]
    distance_source = torch.linalg.norm(delta_source, dim=-1)
    distance_target = torch.linalg.norm(delta_target, dim=-1)
    distance_term = torch.square(distance_target - distance_source)
    source_dir = F.normalize(delta_source, dim=-1, eps=1.0e-6)
    target_dir = F.normalize(delta_target, dim=-1, eps=1.0e-6)
    direction_term = 1.0 - (source_dir * target_dir).sum(dim=-1)
    weight = torch.exp(-lambda_dis_exp * distance_source)
    return distance_term, direction_term, weight


class CrossEmbodimentTrainer:
    """Trainer for multi-hand latent learning and retargeting.

    Parameters
    ----------
    hand_names : Sequence[str], shape=(H,)
        Hand embodiment names for training.
    config : TrainingConfig
        Hyper-parameter bundle.
    """

    def __init__(self, hand_names: Sequence[str], config: TrainingConfig) -> None:
        """Initialize models, optimizers, and caches.

        Parameters
        ----------
        hand_names : Sequence[str], shape=(H,)
            Hand embodiment names.
        config : TrainingConfig
            Hyper-parameter bundle.

        Returns
        -------
        None
            Initializes trainer state in-place.
        """

        self.hand_names: List[str] = list(hand_names)
        self.config = config
        self.registry = MultiHandDifferentiableFK(hand_names)
        self.hand_models = self.registry.models
        self.dof_per_hand: Dict[str, int] = {name: self.hand_models[name].dof_count() for name in self.hand_names}
        self.tip_count_per_hand: Dict[str, int] = {name: self.hand_models[name].tip_count() for name in self.hand_names}
        self.arm_dof = int(self.config.arm_dof)
        self.hand_dof_per_hand: Dict[str, int] = {name: self.dof_per_hand[name] - self.arm_dof for name in self.hand_names}
        self.autoencoders = nn.ModuleDict(
            {
                name: HandAutoencoder(
                    arm_dof=self.arm_dof,
                    hand_dof=self.hand_dof_per_hand[name],
                    latent_dim_hand=self.config.latent_dim_hand,
                    hand_hidden_dims=self.config.hand_hidden_dims,
                )
                for name in self.hand_names
            }
        )
        self.autoencoders.to(self.config.device)
        self.optimizer = torch.optim.AdamW(list(self.autoencoders.parameters()), lr=self.config.learning_rate)
        self.checkpoint_root_dir = os.path.abspath(self.config.checkpoint_dir)
        os.makedirs(self.checkpoint_root_dir, exist_ok=True)
        self._pinch_templates: Dict[str, torch.Tensor] = {}
        self._pinch_points: Dict[str, torch.Tensor] = {}
        self._pinch_pair_cache: Dict[Tuple[str, ...], Tuple[Tuple[int, int], ...]] = {}
        self._pinch_template_target_noise_std = float(self.config.pinch_template_target_noise_std)
        self._pinch_template_joint_noise_std = float(self.config.pinch_template_joint_noise_std)
        self._neutral_tips: Dict[str, torch.Tensor] = {}
        self._encode_state = TrainerCacheState()
        self._decode_states: Dict[str, TrainerCacheState] = {}

    def encode_state(self) -> TrainerCacheState:
        """Return encode cache state.

        Parameters
        ----------
        self : CrossEmbodimentTrainer, shape=()
            Trainer instance.

        Returns
        -------
        TrainerCacheState, shape=()
            Shared encode cache.
        """

        return self._encode_state

    def decode_state(self, hand_name: str) -> TrainerCacheState:
        """Return decode cache state for a hand.

        Parameters
        ----------
        hand_name : str
            Target hand name with shape=().

        Returns
        -------
        TrainerCacheState, shape=()
            Decode cache for ``hand_name``.
        """

        state = self._decode_states.get(hand_name)
        if state is None:
            state = TrainerCacheState()
            self._decode_states[hand_name] = state
        return state

    def load_autoencoders_from_payload(self, payload: Dict) -> None:
        """Load autoencoder state dict from a checkpoint payload.

        Parameters
        ----------
        payload : dict, shape=()
            Checkpoint payload containing key ``"autoencoders"``.

        Returns
        -------
        None
            Loads parameters and sets eval mode.
        """

        self.autoencoders.load_state_dict(payload["autoencoders"], strict=False)
        for left_name in self.autoencoders.keys():
            if "_left" in left_name:
                right_name = left_name.replace("_left", "_right")
                if right_name in self.autoencoders:
                    self.autoencoders[left_name].load_state_dict(self.autoencoders[right_name].state_dict())
        for autoencoder in self.autoencoders.values():
            autoencoder.eval()

    def normalized_qpos(self, hand_name: str, qpos: torch.Tensor) -> torch.Tensor:
        """Normalize raw joint angles for a selected hand.

        Parameters
        ----------
        hand_name : str
            Hand name with shape=().
        qpos : torch.Tensor, shape=(T, D) or (D,), dtype=float32 or float64
            Raw joint angles in radians.

        Returns
        -------
        torch.Tensor, shape matching ``qpos``, dtype=float32 or float64
            Normalized values in ``[-1, 1]``.
        """

        normalized = self.hand_models[hand_name].angles_to_normalized(qpos)
        return torch.clamp(normalized, -1.0, 1.0)

    def chunk_qpos(self, hand_name: str, qpos: torch.Tensor, chunk_size: int) -> List[torch.Tensor]:
        """Split qpos sequence into device chunks.

        Parameters
        ----------
        hand_name : str
            Hand name with shape=().
        qpos : torch.Tensor, shape=(T, D), dtype=float32
            Normalized qpos sequence.
        chunk_size : int
            Chunk length with shape=().

        Returns
        -------
        list[torch.Tensor], shape=(K,)
            Chunk list where each tensor has shape=(t_k, D).
        """

        qpos_tensor = qpos.reshape(-1, self.dof_per_hand[hand_name]).contiguous()
        return list(torch.split(qpos_tensor.to(device=self.config.device), chunk_size, dim=0))

    def _cache_pinch_templates(self) -> None:
        """Precompute pinch-oriented IK templates for each hand.

        Parameters
        ----------
        self : CrossEmbodimentTrainer, shape=()
            Trainer instance storing template caches.

        Returns
        -------
        None
            Updates template-related fields in-place.
        """

        base_offset = torch.tensor(self.config.pinch_offset, dtype=torch.float32)
        template_count = int(self.config.pinch_template_count)
        target_noise_std = self._pinch_template_target_noise_std
        for hand_name in self.hand_names:
            model = self.hand_models[hand_name]
            dof = model.dof_count()
            neutral = model.angles_to_normalized(torch.zeros(dof, dtype=torch.float32))
            tips, wrist_pose = model.forward_with_wrist_pose(neutral)
            neutral_tips = tips.detach()
            wrist_pose = wrist_pose.detach()
            self._neutral_tips[hand_name] = neutral_tips
            rotation = wrist_pose[:3, :3] if wrist_pose.ndim == 2 else wrist_pose[0, :3, :3]
            wrist_position = wrist_pose[:3, 3] if wrist_pose.ndim == 2 else wrist_pose[0, :3, 3]
            offset_world = torch.matmul(rotation, base_offset.to(device=rotation.device, dtype=rotation.dtype))

            target_sets: List[torch.Tensor] = []
            pinch_points: List[torch.Tensor] = []
            tip_count = self.hand_tip_count(hand_name)
            for finger_index in range(1, tip_count):
                pinch_point = wrist_position + offset_world
                base_target = neutral_tips.clone()
                base_target[0] = pinch_point
                base_target[finger_index] = pinch_point
                target_sets.append(base_target)
                pinch_points.append(pinch_point)

            if not target_sets:
                self._pinch_templates[hand_name] = torch.zeros(0, neutral.shape[0], dtype=torch.float32)
                self._pinch_points[hand_name] = torch.zeros(0, 3, dtype=torch.float32)
                continue

            base_targets_tensor = torch.stack(target_sets, dim=0)
            pinch_points_tensor = torch.stack(pinch_points, dim=0)
            repeats = math.ceil(template_count / base_targets_tensor.shape[0])
            tiled_targets = base_targets_tensor.repeat_interleave(repeats, dim=0)[:template_count]
            tiled_points = pinch_points_tensor.repeat_interleave(repeats, dim=0)[:template_count]

            noise = torch.randn_like(tiled_targets) * target_noise_std
            noise[:, 0, :] = 0.0
            perturbed_targets = tiled_targets + noise
            perturbed_targets[:, 0, :] = tiled_points

            history = solve_inverse_kinematics(
                model=model,
                target_positions=perturbed_targets,
                iterations=self.config.pinch_template_iterations,
                learning_rate=self.config.pinch_template_learning_rate,
            )
            solutions = history[-1].detach()
            self._pinch_templates[hand_name] = solutions.cpu()
            self._pinch_points[hand_name] = tiled_points.detach().cpu()

    def _sample_pinch_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample noisy joint configurations from cached pinch templates.

        Parameters
        ----------
        batch_size : int
            Number of samples per hand with shape=().

        Returns
        -------
        dict[str, torch.Tensor]
            Mapping hand->samples with each tensor shape=(B, D).
        """

        device = self.config.device
        pinch_batches: Dict[str, torch.Tensor] = {}
        for hand_name in self.hand_names:
            templates = self._pinch_templates[hand_name]
            dof = self.dof_per_hand[hand_name]
            if templates.shape[0] == 0:
                pinch_batches[hand_name] = torch.empty(0, dof, device=device)
                continue
            indices = torch.randint(0, templates.shape[0], (batch_size,), device=templates.device)
            sampled = templates[indices].to(device=device)
            noise = torch.randn_like(sampled) * self._pinch_template_joint_noise_std
            pinch_batches[hand_name] = torch.clamp(sampled + noise, -1.0, 1.0)
        return {name: tensor.to(device=device) for name, tensor in pinch_batches.items()}

    def _sample_qpos(self) -> Dict[str, torch.Tensor]:
        """Sample random normalized qpos for each hand.

        Parameters
        ----------
        self : CrossEmbodimentTrainer, shape=()
            Trainer instance.

        Returns
        -------
        dict[str, torch.Tensor]
            Mapping hand->random qpos with shape=(B, D).
        """

        batch: Dict[str, torch.Tensor] = {}
        for name, dof in self.dof_per_hand.items():
            batch[name] = torch.empty(self.config.batch_size, dof, device=self.config.device).uniform_(-1.0, 1.0)
        return batch

    def _sample_training_batch(self) -> Dict[str, torch.Tensor]:
        """Sample mixed random and pinch-centric batches.

        Parameters
        ----------
        self : CrossEmbodimentTrainer, shape=()
            Trainer instance.

        Returns
        -------
        dict[str, torch.Tensor]
            Mapping hand->training batch with shape=(B, D).
        """

        batch_size = self.config.batch_size
        pinch_count = int(round(batch_size * self.config.pinch_sampling_probability))
        uniform_count = batch_size - pinch_count
        uniform_samples = self._sample_qpos()
        if pinch_count <= 0:
            return uniform_samples
        pinch_samples = self._sample_pinch_batch(pinch_count)
        mixed: Dict[str, torch.Tensor] = {}
        for hand_name in self.hand_names:
            if uniform_count <= 0:
                mixed_batch = pinch_samples[hand_name]
            else:
                mixed_batch = torch.cat([uniform_samples[hand_name], pinch_samples[hand_name]], dim=0)
            mixed[hand_name] = mixed_batch[torch.randperm(mixed_batch.shape[0], device=self.config.device)]
        return mixed

    def _init_checkpoint_session_dir(self) -> str:
        """Create a timestamp-based run folder for checkpoints.

        Parameters
        ----------
        self : CrossEmbodimentTrainer, shape=()
            Trainer instance.

        Returns
        -------
        str
            Absolute run directory path with shape=().
        """

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self.checkpoint_root_dir, timestamp)
        suffix = 1
        while os.path.exists(run_dir):
            run_dir = os.path.join(self.checkpoint_root_dir, f"{timestamp}_{suffix:02d}")
            suffix += 1
        os.makedirs(run_dir, exist_ok=False)
        return run_dir

    def _split_qpos(self, hand_name: str, qpos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split qpos into arm and hand components.

        Parameters
        ----------
        hand_name : str
            Hand name with shape=().
        qpos : torch.Tensor, shape=(B, D), dtype=float32
            Normalized qpos.

        Returns
        -------
        tuple
            - qpos_arm : torch.Tensor, shape=(B, arm_dof), dtype=float32
            - qpos_hand : torch.Tensor, shape=(B, hand_dof), dtype=float32
        """

        hand_dof = self.hand_dof_per_hand[hand_name]
        return qpos[:, : self.arm_dof], qpos[:, self.arm_dof : self.arm_dof + hand_dof]

    @staticmethod
    def _merge_qpos(arm: torch.Tensor, hand: torch.Tensor) -> torch.Tensor:
        """Concatenate arm and hand qpos tensors.

        Parameters
        ----------
        arm : torch.Tensor, shape=(B, A), dtype=float32
            Arm segment.
        hand : torch.Tensor, shape=(B, H), dtype=float32
            Hand segment.

        Returns
        -------
        torch.Tensor, shape=(B, A + H), dtype=float32
            Concatenated qpos.
        """

        return torch.cat([arm, hand], dim=-1)

    def hand_tip_count(self, hand_name: str) -> int:
        """Return fingertip count for one hand.

        Parameters
        ----------
        hand_name : str
            Hand name with shape=().

        Returns
        -------
        int
            Fingertip count with shape=().
        """

        return self.tip_count_per_hand[hand_name]

    def pinch_pairs_for_hand(self, hand_name: str) -> Tuple[Tuple[int, int], ...]:
        """Return valid pinch pairs for one hand based on tip count.

        Parameters
        ----------
        hand_name : str
            Hand name with shape=().

        Returns
        -------
        tuple[tuple[int, int], ...], shape=(P, 2)
            Filtered pinch pairs valid for the hand.
        """

        cache_key = (hand_name,)
        if cache_key in self._pinch_pair_cache:
            return self._pinch_pair_cache[cache_key]
        tip_limit = self.hand_tip_count(hand_name)
        filtered = tuple(pair for pair in self.config.pinch_pairs if pair[0] < tip_limit and pair[1] < tip_limit)
        self._pinch_pair_cache[cache_key] = filtered
        return filtered

    def shared_pinch_pairs(self, hand_a: str, hand_b: str) -> Tuple[Tuple[int, int], ...]:
        """Return pinch pairs valid for both hands.

        Parameters
        ----------
        hand_a : str
            First hand name with shape=().
        hand_b : str
            Second hand name with shape=().

        Returns
        -------
        tuple[tuple[int, int], ...], shape=(P, 2)
            Pinch pairs below the minimum tip count.
        """

        cache_key = tuple(sorted((hand_a, hand_b)))
        if cache_key in self._pinch_pair_cache:
            return self._pinch_pair_cache[cache_key]
        limit = min(self.hand_tip_count(hand_a), self.hand_tip_count(hand_b))
        filtered = tuple(pair for pair in self.config.pinch_pairs if pair[0] < limit and pair[1] < limit)
        self._pinch_pair_cache[cache_key] = filtered
        return filtered

    def step(self) -> Dict[str, float]:
        """Run one optimization step.

        Parameters
        ----------
        self : CrossEmbodimentTrainer, shape=()
            Trainer instance.

        Returns
        -------
        dict[str, float], shape=(7,)
            Scalar metric dictionary for logging.
        """

        self.optimizer.zero_grad()
        batch_qpos = self._sample_training_batch()
        latents: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        latent_stats: Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = {}
        source_tips: Dict[str, torch.Tensor] = {}
        hand_recon_loss = torch.tensor(0.0, device=self.config.device)

        for name in self.hand_names:
            hand_input = batch_qpos[name]
            _, qpos_hand_gt = self._split_qpos(name, hand_input)
            latent_arm, latent_hand, _, qpos_hand_pred, hand_stats = self.autoencoders[name](hand_input)
            latents[name] = (latent_arm, latent_hand)
            latent_stats[name] = {"hand": hand_stats}
            hand_recon_loss = hand_recon_loss + F.mse_loss(qpos_hand_pred, qpos_hand_gt)
            source_tips[name] = self.hand_models[name].forward(hand_input)

        hand_recon_loss = hand_recon_loss / float(len(self.hand_names))
        reconstruction_loss = hand_recon_loss * self.config.rec_hand_weight

        pinch_distance_total = torch.tensor(0.0, device=self.config.device)
        pinch_direction_total = torch.tensor(0.0, device=self.config.device)
        exp_weight_total = torch.tensor(0.0, device=self.config.device)
        pair_count = 0

        for source_name in self.hand_names:
            source_fk = source_tips[source_name]
            latent_arm_source, latent_hand_source = latents[source_name]
            for target_name in self.hand_names:
                if target_name == source_name:
                    continue
                pinch_pairs = self.shared_pinch_pairs(source_name, target_name)
                if len(pinch_pairs) == 0:
                    continue
                target_arm_pred, target_hand_pred = self.autoencoders[target_name].decode_from_latents(
                    latent_arm_source,
                    latent_hand_source,
                )
                fk_hand = self.hand_models[target_name].forward(self._merge_qpos(target_arm_pred.detach(), target_hand_pred))
                distance_term, direction_term, exp_weight = compute_pinch_loss(
                    source_fk,
                    fk_hand,
                    pinch_pairs,
                    self.config.lambda_dis_exp,
                )
                pair_count += 1
                pinch_distance_total = pinch_distance_total + (distance_term * exp_weight).mean()
                pinch_direction_total = pinch_direction_total + (direction_term * exp_weight).mean()
                exp_weight_total = exp_weight_total + exp_weight.mean()

        if pair_count > 0:
            pinch_distance = pinch_distance_total / float(pair_count)
            pinch_direction = pinch_direction_total / float(pair_count)
            exp_weight_mean = exp_weight_total / float(pair_count)
        else:
            pinch_distance = torch.tensor(0.0, device=self.config.device)
            pinch_direction = torch.tensor(0.0, device=self.config.device)
            exp_weight_mean = torch.tensor(0.0, device=self.config.device)

        kl_total = torch.tensor(0.0, device=self.config.device)
        component_count = 0
        for stats in latent_stats.values():
            for mean, logvar in stats.values():
                kl_divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1)
                kl_total = kl_total + kl_divergence.mean()
                component_count += 1
        kl_loss = kl_total / float(component_count) if component_count > 0 else torch.tensor(0.0, device=self.config.device)

        total_loss = (
            reconstruction_loss
            + self.config.lambda_dis * pinch_distance
            + self.config.lambda_dir * pinch_direction
            + self.config.lambda_kl * kl_loss
        )
        total_loss.backward()
        self.optimizer.step()

        return {
            "loss_total": float(total_loss.detach().cpu()),
            "loss_rec_total": float(reconstruction_loss.detach().cpu()),
            "loss_rec_hand": float((hand_recon_loss * self.config.rec_hand_weight).detach().cpu()),
            "loss_pinch_dis": float(pinch_distance.detach().cpu()) * self.config.lambda_dis,
            "loss_pinch_dir": float(pinch_direction.detach().cpu()) * self.config.lambda_dir,
            "loss_kl": float(kl_loss.detach().cpu()) * self.config.lambda_kl,
            "exp_dis": float(exp_weight_mean.detach().cpu()),
        }

    def train(self) -> List[Dict[str, float]]:
        """Run full training loop and save periodic checkpoints.

        Parameters
        ----------
        self : CrossEmbodimentTrainer, shape=()
            Trainer instance.

        Returns
        -------
        list[dict[str, float]], shape=(num_steps,)
            Metric history across training steps.
        """

        self.checkpoint_dir = self._init_checkpoint_session_dir()
        self._cache_pinch_templates()
        history: List[Dict[str, float]] = []
        for autoencoder in self.autoencoders.values():
            autoencoder.train()
        for step_index in range(self.config.num_steps):
            metrics = self.step()
            history.append(metrics)
            print(
                f"Step {step_index + 1:04d} | "
                f"total={metrics['loss_total']:.4f} "
                f"rec_total={metrics['loss_rec_total']:.4f} "
                f"rec_hand={metrics['loss_rec_hand']:.4f} "
                f"pinch_dis={metrics['loss_pinch_dis']:.4f} "
                f"pinch_dir={metrics['loss_pinch_dir']:.4f} "
                f"exp_dis={metrics['exp_dis']:.4f} "
                f"kl={metrics['loss_kl']:.4f}"
            )
            epoch_index = step_index + 1
            if self.config.checkpoint_interval > 0 and epoch_index % self.config.checkpoint_interval == 0:
                checkpoint_path = self.save_checkpoint(epoch_index)
                print(f"Saved checkpoint to {checkpoint_path}")
        return history

    def save_checkpoint(self, epoch_index: int) -> str:
        """Save checkpoint payload at current epoch index.

        Parameters
        ----------
        epoch_index : int
            1-based step index with shape=().

        Returns
        -------
        str
            Checkpoint file path with shape=().
        """

        config_dict = asdict(self.config)
        config_dict["device"] = str(self.config.device)
        config_dict["checkpoint_dir"] = self.checkpoint_dir
        payload = {
            "epoch": epoch_index,
            "hand_names": self.hand_names,
            "config": config_dict,
            "autoencoders": self.autoencoders.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch_index:04d}.pt")
        torch.save(payload, checkpoint_path)
        return checkpoint_path
