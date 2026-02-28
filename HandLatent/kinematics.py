"""Differentiable forward kinematics and IK utilities for xarm hand embodiments.

This module keeps only the xarm variants used by the minimal training and
inference pipeline.
"""

from __future__ import annotations

import contextlib
import io
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from xml.etree import ElementTree as et

import torch
import torch.nn.functional as F
from urdf_parser_py import urdf as urdf_parser

PROJECT_ROOT: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ASSETS_DIR: str = os.path.join(PROJECT_ROOT, "Assets")

HAND_CONFIGS: Dict[str, Dict[str, Sequence[str] | str]] = {
    "xarm7_xhand_left": {
        "urdf_path": os.path.join(ASSETS_DIR, "xarm7_xhand", "xarm7_xhand_left_hand.urdf"),
        "root_link": "link_base",
        "wrist_link": "link7",
        "tip_links": (
            "left_hand_thumb_rota_tip",
            "left_hand_index_rota_tip",
            "left_hand_mid_tip",
            "left_hand_ring_tip",
            "left_hand_pinky_tip",
        ),
    },
    "xarm7_xhand_right": {
        "urdf_path": os.path.join(ASSETS_DIR, "xarm7_xhand", "xarm7_xhand_right_hand.urdf"),
        "root_link": "link_base",
        "wrist_link": "link7",
        "tip_links": (
            "right_hand_thumb_rota_tip",
            "right_hand_index_rota_tip",
            "right_hand_mid_tip",
            "right_hand_ring_tip",
            "right_hand_pinky_tip",
        ),
    },
    "xarm7_inspire_left": {
        "urdf_path": os.path.join(ASSETS_DIR, "xarm7_inspire", "xarm7_inspire_left_hand.urdf"),
        "root_link": "link_base",
        "wrist_link": "link7",
        "tip_links": (
            "thumb_tip",
            "index_tip",
            "middle_tip",
            "ring_tip",
            "pinky_tip",
        ),
    },
    "xarm7_inspire_right": {
        "urdf_path": os.path.join(ASSETS_DIR, "xarm7_inspire", "xarm7_inspire_right_hand.urdf"),
        "root_link": "link_base",
        "wrist_link": "link7",
        "tip_links": (
            "thumb_tip",
            "index_tip",
            "middle_tip",
            "ring_tip",
            "pinky_tip",
        ),
    },
    "xarm7_ability_left": {
        "urdf_path": os.path.join(ASSETS_DIR, "xarm7_ability", "xarm7_ability_left_hand.urdf"),
        "root_link": "link_base",
        "wrist_link": "link7",
        "tip_links": (
            "thumb_tip",
            "index_tip",
            "middle_tip",
            "ring_tip",
            "pinky_tip",
        ),
    },
    "xarm7_ability_right": {
        "urdf_path": os.path.join(ASSETS_DIR, "xarm7_ability", "xarm7_ability_right_hand.urdf"),
        "root_link": "link_base",
        "wrist_link": "link7",
        "tip_links": (
            "thumb_tip",
            "index_tip",
            "middle_tip",
            "ring_tip",
            "pinky_tip",
        ),
    },
    "xarm7_paxini_left": {
        "urdf_path": os.path.join(ASSETS_DIR, "xarm7_paxini", "xarm7_pxdh13_left_hand_glb.urdf"),
        "root_link": "link_base",
        "wrist_link": "link7",
        "tip_links": (
            "thumb_tip",
            "index_tip",
            "middle_tip",
            "ring_tip",
        ),
    },
    "xarm7_paxini_right": {
        "urdf_path": os.path.join(ASSETS_DIR, "xarm7_paxini", "xarm7_pxdh13_right_hand_glb.urdf"),
        "root_link": "link_base",
        "wrist_link": "link7",
        "tip_links": (
            "thumb_tip",
            "index_tip",
            "middle_tip",
            "ring_tip",
        ),
    },
}


@dataclass(frozen=True)
class JointSpec:
    """URDF joint metadata used for forward kinematics.

    Parameters
    ----------
    name : str
        Joint identifier in URDF.
    parent : str
        Parent link name.
    child : str
        Child link name.
    kind : str
        URDF joint type string.
    origin_transform : torch.Tensor, shape=(4, 4), dtype=float32
        Homogeneous transform from parent link frame to joint frame.
    axis : torch.Tensor or None, shape=(3,), dtype=float32
        Joint axis for revolute joints.
    lower : float or None
        Lower joint limit in radians.
    upper : float or None
        Upper joint limit in radians.
    mimic_parent : str or None
        Parent joint name for mimic relation.
    mimic_multiplier : float
        Mimic multiplier.
    mimic_offset : float
        Mimic offset in radians.
    """

    name: str
    parent: str
    child: str
    kind: str
    origin_transform: torch.Tensor
    axis: Optional[torch.Tensor]
    lower: Optional[float]
    upper: Optional[float]
    mimic_parent: Optional[str]
    mimic_multiplier: float
    mimic_offset: float


def _strip_disallowed_name_attributes(root: et.Element) -> int:
    """Remove unsupported ``name`` attributes on visual and collision elements.

    Parameters
    ----------
    root : xml.etree.ElementTree.Element
        Root XML element for a URDF document.

    Returns
    -------
    int
        Number of removed attributes as scalar count with shape=().
    """

    removed = 0
    for element in root.iter():
        if element.tag in {"visual", "collision"} and "name" in element.attrib:
            del element.attrib["name"]
            removed += 1
    return removed


def load_urdf_silent(urdf_path: str) -> urdf_parser.URDF:
    """Load a URDF document while suppressing parser stdout and stderr.

    Parameters
    ----------
    urdf_path : str
        Absolute or relative filesystem path with shape=() to a ``.urdf`` file.

    Returns
    -------
    urdf_parser_py.urdf.URDF, shape=()
        Parsed URDF model object.
    """

    xml_tree = et.parse(urdf_path)
    root = xml_tree.getroot()
    _strip_disallowed_name_attributes(root)
    sanitized_xml = et.tostring(root, encoding="unicode")
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        return urdf_parser.URDF.from_xml_string(sanitized_xml)


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> torch.Tensor:
    """Convert XYZ Euler angles to a rotation matrix.

    Parameters
    ----------
    roll : float
        Rotation around x-axis in radians with shape=().
    pitch : float
        Rotation around y-axis in radians with shape=().
    yaw : float
        Rotation around z-axis in radians with shape=().

    Returns
    -------
    torch.Tensor, shape=(3, 3), dtype=float32
        Rotation matrix built from yaw-pitch-roll composition.
    """

    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    return torch.tensor(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=torch.float32,
    )


def _make_transform(rotation: torch.Tensor, translation: Sequence[float]) -> torch.Tensor:
    """Construct a homogeneous transform from rotation and translation.

    Parameters
    ----------
    rotation : torch.Tensor, shape=(3, 3), dtype=float32
        Rotation matrix component.
    translation : Sequence[float], shape=(3,)
        XYZ translation in meters.

    Returns
    -------
    torch.Tensor, shape=(4, 4), dtype=float32
        Homogeneous transform matrix.
    """

    transform = torch.eye(4, dtype=torch.float32)
    transform[:3, :3] = rotation
    transform[:3, 3] = torch.tensor(list(translation), dtype=torch.float32)
    return transform


def axis_angle_to_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Convert batched axis-angle representation to batched rotation matrices.

    Parameters
    ----------
    axis : torch.Tensor, shape=(B, 3), dtype=float32 or float64
        Rotation axes.
    angle : torch.Tensor, shape=(B,), dtype matching ``axis``
        Rotation angles in radians.

    Returns
    -------
    torch.Tensor, shape=(B, 3, 3), dtype matching ``axis``
        Rotation matrices corresponding to each axis-angle pair.
    """

    axis = torch.nn.functional.normalize(axis, dim=-1)
    angle = angle.unsqueeze(-1)
    cos_term = torch.cos(angle)
    sin_term = torch.sin(angle)
    one_minus_cos = 1.0 - cos_term
    x = axis[..., 0:1]
    y = axis[..., 1:2]
    z = axis[..., 2:3]
    xy = x * y
    yz = y * z
    zx = z * x
    rotation = torch.zeros(axis.shape[0], 3, 3, dtype=axis.dtype, device=axis.device)
    rotation[:, 0, 0] = (cos_term + x * x * one_minus_cos).squeeze(-1)
    rotation[:, 0, 1] = (xy * one_minus_cos - z * sin_term).squeeze(-1)
    rotation[:, 0, 2] = (zx * one_minus_cos + y * sin_term).squeeze(-1)
    rotation[:, 1, 0] = (xy * one_minus_cos + z * sin_term).squeeze(-1)
    rotation[:, 1, 1] = (cos_term + y * y * one_minus_cos).squeeze(-1)
    rotation[:, 1, 2] = (yz * one_minus_cos - x * sin_term).squeeze(-1)
    rotation[:, 2, 0] = (zx * one_minus_cos - y * sin_term).squeeze(-1)
    rotation[:, 2, 1] = (yz * one_minus_cos + x * sin_term).squeeze(-1)
    rotation[:, 2, 2] = (cos_term + z * z * one_minus_cos).squeeze(-1)
    return rotation


class HandKinematicsModel:
    """Differentiable FK model for one hand-arm embodiment.

    Parameters
    ----------
    hand_name : str
        Embodiment identifier with shape=().
    urdf_path : str
        URDF filesystem path with shape=().
    root_link : str
        Root link name used as FK origin with shape=().
    tip_links : Sequence[str], shape=(F,)
        Ordered fingertip link names.
    wrist_link : str or None
        Wrist link name with shape=(); defaults to ``root_link``.
    """

    def __init__(
        self,
        hand_name: str,
        urdf_path: str,
        root_link: str,
        tip_links: Sequence[str],
        wrist_link: Optional[str] = None,
    ) -> None:
        """Initialize FK buffers and parse URDF kinematic graph.

        Parameters
        ----------
        hand_name : str
            Embodiment identifier with shape=().
        urdf_path : str
            URDF filesystem path with shape=().
        root_link : str
            Root link name with shape=().
        tip_links : Sequence[str], shape=(F,)
            Fingertip link names used for output ordering.
        wrist_link : str or None
            Wrist link name with shape=(); when ``None``, equals ``root_link``.

        Returns
        -------
        None
            Initializes object state in-place.
        """

        self.hand_name = hand_name
        self.tip_links: List[str] = list(tip_links)
        self.root_link = root_link
        self.wrist_link = wrist_link if wrist_link is not None else root_link
        self.urdf = load_urdf_silent(urdf_path)
        self.joint_specs: Dict[str, JointSpec] = {}
        self.children_by_parent: Dict[str, List[str]] = {}
        self.dof_joints: List[str] = []
        self._lower = torch.tensor([], dtype=torch.float32)
        self._upper = torch.tensor([], dtype=torch.float32)
        self._parse_urdf()
        self.traversal_order: List[str] = self._compute_traversal()

    def _parse_urdf(self) -> None:
        """Parse URDF joints into static tensors and mimic metadata.

        Parameters
        ----------
        self : HandKinematicsModel, shape=()
            Instance storing parsed joint metadata.

        Returns
        -------
        None
            Updates ``joint_specs``, ``dof_joints``, ``_lower``, and ``_upper``.
        """

        joints: List[urdf_parser.Joint] = list(self.urdf.joints)
        dof_lowers: List[float] = []
        dof_uppers: List[float] = []
        for joint in joints:
            if joint.origin is not None:
                xyz = joint.origin.xyz if joint.origin.xyz is not None else [0.0, 0.0, 0.0]
                rpy = joint.origin.rpy if joint.origin.rpy is not None else [0.0, 0.0, 0.0]
            else:
                xyz = [0.0, 0.0, 0.0]
                rpy = [0.0, 0.0, 0.0]
            origin_transform = _make_transform(_rpy_to_matrix(rpy[0], rpy[1], rpy[2]), xyz)
            axis_tensor: Optional[torch.Tensor] = None
            lower: Optional[float] = None
            upper: Optional[float] = None
            mimic_parent: Optional[str] = None
            mimic_multiplier = 1.0
            mimic_offset = 0.0
            if joint.type == "revolute":
                axis_list = joint.axis if joint.axis is not None else [0.0, 0.0, 1.0]
                axis_tensor = torch.tensor(axis_list, dtype=torch.float32)
                axis_tensor = axis_tensor / torch.linalg.norm(axis_tensor)
                if joint.limit is not None:
                    lower = float(joint.limit.lower)
                    upper = float(joint.limit.upper)
                if joint.mimic is not None:
                    mimic_parent = joint.mimic.joint
                    mimic_multiplier = float(joint.mimic.multiplier)
                    mimic_offset = float(joint.mimic.offset)
                else:
                    self.dof_joints.append(joint.name)
                    dof_lowers.append(lower if lower is not None else 0.0)
                    dof_uppers.append(upper if upper is not None else 0.0)
            self.joint_specs[joint.name] = JointSpec(
                name=joint.name,
                parent=joint.parent,
                child=joint.child,
                kind=joint.type,
                origin_transform=origin_transform,
                axis=axis_tensor,
                lower=lower,
                upper=upper,
                mimic_parent=mimic_parent,
                mimic_multiplier=mimic_multiplier,
                mimic_offset=mimic_offset,
            )
            self.children_by_parent.setdefault(joint.parent, []).append(joint.name)
        self._lower = torch.tensor(dof_lowers, dtype=torch.float32)
        self._upper = torch.tensor(dof_uppers, dtype=torch.float32)

    def _compute_traversal(self) -> List[str]:
        """Compute BFS traversal order for joint propagation.

        Parameters
        ----------
        self : HandKinematicsModel, shape=()
            Instance storing kinematic tree.

        Returns
        -------
        list[str], shape=(J,)
            Joint names in traversal order from ``root_link``.
        """

        order: List[str] = []
        queue: List[str] = [self.root_link]
        while queue:
            link = queue.pop(0)
            for joint_name in self.children_by_parent.get(link, []):
                order.append(joint_name)
                queue.append(self.joint_specs[joint_name].child)
        return order

    def dof_count(self) -> int:
        """Return number of independent revolute DoFs.

        Parameters
        ----------
        self : HandKinematicsModel, shape=()
            Instance queried for DoF count.

        Returns
        -------
        int
            Scalar DoF count with shape=().
        """

        return len(self.dof_joints)

    def joint_name_order(self) -> List[str]:
        """Return ordered independent joint names.

        Parameters
        ----------
        self : HandKinematicsModel, shape=()
            Instance queried for joint ordering.

        Returns
        -------
        list[str], shape=(D,)
            Independent joint names where ``D == dof_count()``.
        """

        return list(self.dof_joints)

    def tip_count(self) -> int:
        """Return fingertip count.

        Parameters
        ----------
        self : HandKinematicsModel, shape=()
            Instance queried for fingertip count.

        Returns
        -------
        int
            Scalar fingertip count with shape=().
        """

        return len(self.tip_links)

    def _normalized_to_all_joint_angles(self, normalized: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convert normalized independent joints to all revolute joint angles.

        Parameters
        ----------
        normalized : torch.Tensor, shape=(B, D), dtype=float32 or float64
            Normalized independent joints in ``[-1, 1]``.

        Returns
        -------
        dict[str, torch.Tensor]
            Mapping from revolute joint name to angle tensor with shape=(B,).
        """

        clipped = torch.clamp(normalized, -1.0, 1.0)
        lower = self._lower.to(device=clipped.device, dtype=clipped.dtype)
        upper = self._upper.to(device=clipped.device, dtype=clipped.dtype)
        angles = (clipped + 1.0) * 0.5 * (upper - lower) + lower
        angle_map: Dict[str, torch.Tensor] = {}
        for index, joint_name in enumerate(self.dof_joints):
            angle_map[joint_name] = angles[:, index]
        for joint_name, spec in self.joint_specs.items():
            if spec.kind == "revolute" and spec.mimic_parent is not None:
                angle_map[joint_name] = (
                    angle_map[spec.mimic_parent] * spec.mimic_multiplier + spec.mimic_offset
                )
        for joint_name, spec in self.joint_specs.items():
            if spec.kind == "revolute" and joint_name in angle_map:
                lower_bound = float(spec.lower) if spec.lower is not None else float("-inf")
                upper_bound = float(spec.upper) if spec.upper is not None else float("inf")
                angle_map[joint_name] = torch.clamp(angle_map[joint_name], min=lower_bound, max=upper_bound)
        return angle_map

    def angles_to_normalized(self, angles: torch.Tensor) -> torch.Tensor:
        """Convert independent joint angles to normalized range ``[-1, 1]``.

        Parameters
        ----------
        angles : torch.Tensor, shape=(D,) or (B, D), dtype=float32 or float64
            Independent joint angles in radians.

        Returns
        -------
        torch.Tensor, shape matching ``angles``, dtype matching ``angles``
            Normalized joints.
        """

        squeeze = angles.ndim == 1
        batch = angles.unsqueeze(0) if squeeze else angles
        lower = self._lower.to(device=batch.device, dtype=batch.dtype)
        upper = self._upper.to(device=batch.device, dtype=batch.dtype)
        span = upper - lower
        safe_span = torch.where(span.abs() < 1.0e-8, torch.ones_like(span), span)
        normalized = 2.0 * (batch - lower) / safe_span - 1.0
        zero_span = span.abs() < 1.0e-8
        normalized = torch.where(zero_span.unsqueeze(0), torch.zeros_like(normalized), normalized)
        return normalized[0] if squeeze else normalized

    def _forward_internal(self, normalized_qpos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """Run FK and return fingertip positions and wrist transform.

        Parameters
        ----------
        normalized_qpos : torch.Tensor, shape=(D,) or (B, D), dtype=float32 or float64
            Normalized independent joints.

        Returns
        -------
        tuple
            - tips : torch.Tensor, shape=(B, F, 3), dtype matching input
            - wrist : torch.Tensor, shape=(B, 4, 4), dtype matching input
            - squeeze_output : bool, shape=()
        """

        squeeze_output = normalized_qpos.ndim == 1
        batch_qpos = normalized_qpos.unsqueeze(0) if squeeze_output else normalized_qpos
        batch = batch_qpos.shape[0]
        dtype = batch_qpos.dtype
        device = batch_qpos.device
        angles_map = self._normalized_to_all_joint_angles(batch_qpos)
        transforms: Dict[str, torch.Tensor] = {
            self.root_link: torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(batch, 1, 1)
        }
        for joint_name in self.traversal_order:
            spec = self.joint_specs[joint_name]
            parent_transform = transforms[spec.parent]
            origin = spec.origin_transform.to(dtype=dtype, device=device).unsqueeze(0).repeat(batch, 1, 1)
            joint_transform = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(batch, 1, 1)
            if spec.kind == "revolute":
                axis = spec.axis.to(dtype=dtype, device=device).unsqueeze(0).repeat(batch, 1)
                angle = angles_map[joint_name].to(dtype=dtype, device=device)
                joint_transform[:, :3, :3] = axis_angle_to_matrix(axis, angle)
            transforms[spec.child] = parent_transform @ origin @ joint_transform
        tips = torch.stack([transforms[tip][:, :3, 3] for tip in self.tip_links], dim=1)
        wrist = transforms[self.wrist_link]
        return tips, wrist, squeeze_output

    def forward(self, normalized_qpos: torch.Tensor) -> torch.Tensor:
        """Compute fingertip positions from normalized joint commands.

        Parameters
        ----------
        normalized_qpos : torch.Tensor, shape=(D,) or (B, D), dtype=float32 or float64
            Normalized independent joints.

        Returns
        -------
        torch.Tensor, shape=(F, 3) or (B, F, 3), dtype matching input
            Fingertip Cartesian coordinates in root frame.
        """

        tips, _, squeeze_output = self._forward_internal(normalized_qpos)
        return tips[0] if squeeze_output else tips

    def forward_with_wrist_pose(self, normalized_qpos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute fingertips and wrist homogeneous pose from normalized joints.

        Parameters
        ----------
        normalized_qpos : torch.Tensor, shape=(D,) or (B, D), dtype=float32 or float64
            Normalized independent joints.

        Returns
        -------
        tuple
            - tips : torch.Tensor, shape=(F, 3) or (B, F, 3), dtype matching input
            - wrist : torch.Tensor, shape=(4, 4) or (B, 4, 4), dtype matching input
        """

        tips, wrist, squeeze_output = self._forward_internal(normalized_qpos)
        if squeeze_output:
            return tips[0], wrist[0]
        return tips, wrist


class MultiHandDifferentiableFK:
    """Registry wrapper that builds FK models for a set of hands.

    Parameters
    ----------
    hand_names : Iterable[str] or None
        Hand names with shape=(H,). ``None`` loads every key in ``HAND_CONFIGS``.
    """

    def __init__(self, hand_names: Optional[Iterable[str]] = None) -> None:
        """Initialize FK model registry.

        Parameters
        ----------
        hand_names : Iterable[str] or None
            Hand names with shape=(H,). ``None`` means all configured hands.

        Returns
        -------
        None
            Builds ``self.models`` in-place.
        """

        targets = list(HAND_CONFIGS.keys()) if hand_names is None else list(hand_names)
        self.models: Dict[str, HandKinematicsModel] = {}
        for name in targets:
            config = HAND_CONFIGS[name]
            self.models[name] = HandKinematicsModel(
                hand_name=name,
                urdf_path=str(config["urdf_path"]),
                root_link=str(config["root_link"]),
                tip_links=tuple(config["tip_links"]),
                wrist_link=str(config["wrist_link"]),
            )

    def supported_hands(self) -> List[str]:
        """Return registry hand identifiers.

        Parameters
        ----------
        self : MultiHandDifferentiableFK, shape=()
            Registry instance.

        Returns
        -------
        list[str], shape=(H,)
            Registered hand names.
        """

        return list(self.models.keys())


def solve_inverse_kinematics(
    model: HandKinematicsModel,
    target_positions: torch.Tensor,
    iterations: int = 100,
    learning_rate: float = 0.05,
    initial_qpos: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Solve fingertip IK with gradient descent on unconstrained latent joints.

    Parameters
    ----------
    model : HandKinematicsModel
        FK model of a single hand embodiment.
    target_positions : torch.Tensor, shape=(F, 3) or (B, F, 3), dtype=float32 or float64
        Target fingertip coordinates.
    iterations : int
        Number of optimization updates with shape=().
    learning_rate : float
        AdamW learning rate with shape=().
    initial_qpos : torch.Tensor or None, shape=(D,) or (B, D), dtype=float32 or float64
        Optional normalized IK seed.

    Returns
    -------
    torch.Tensor, shape=(iterations + 1, D) or (iterations + 1, B, D), dtype matching target
        Optimization trajectory in normalized joint space.
    """

    finger_count = len(model.tip_links)
    squeeze_batch = target_positions.ndim == 2
    targets = target_positions.unsqueeze(0) if squeeze_batch else target_positions
    dof = model.dof_count()
    batch_size = targets.shape[0]
    dtype = targets.dtype if targets.is_floating_point() else torch.float32
    device = targets.device

    if initial_qpos is None:
        unconstrained_start = torch.zeros(batch_size, dof, dtype=dtype, device=device)
    else:
        seed = initial_qpos.to(dtype=dtype, device=device)
        seed = seed.unsqueeze(0) if seed.ndim == 1 else seed
        seed_clamped = torch.clamp(seed, -1.0 + 1.0e-6, 1.0 - 1.0e-6)
        unconstrained_start = torch.atanh(seed_clamped)

    current = unconstrained_start.clone().detach().requires_grad_(True)
    trajectory: List[torch.Tensor] = [torch.tanh(current).detach().clone()]
    optimizer = torch.optim.AdamW([current], lr=learning_rate)
    target = targets.to(dtype=dtype, device=device)

    for _ in range(iterations):
        optimizer.zero_grad()
        normalized = torch.tanh(current)
        tips = model.forward(normalized)
        loss = F.mse_loss(tips, target)
        loss.backward()
        optimizer.step()
        trajectory.append(torch.tanh(current).detach().clone())

    result = torch.stack(trajectory, dim=0)
    return result[:, 0, :] if squeeze_batch else result
