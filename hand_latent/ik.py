"""Minimal Pink IK utilities used by EEPose decoding."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, cast

import numpy as np
import pinocchio as pin
import torch
from pink import Configuration, solve_ik
from pink.tasks import FrameTask
from qpsolvers import available_solvers

from hand_latent.kinematics import ASSETS_DIR, HAND_CONFIGS, HandKinematicsModel

PINK_DT: float = 0.1
PINK_ARM_WEIGHT_EXPONENT: float = 1.5


@dataclass
class PinkArmIkContext:
    """Cached Pinocchio/Pink context for one hand embodiment.

    Parameters
    ----------
    robot : pin.robot_wrapper.RobotWrapper
        Pinocchio robot wrapper object with shape=().
    joint_indices : dict[str, int], shape=(J,)
        Mapping from joint name to Pinocchio q index.
    tip_links : Sequence[str], shape=(F,)
        Fingertip link names used for pinch midpoint calculation.
    wrist_link : str
        Wrist frame name with shape=().
    root_link : str
        Root frame name with shape=().
    mimic_pairs : Sequence[Tuple[int, int, float, float]], shape=(M, 4)
        Mimic relation tuples ``(mimic_idx, parent_idx, multiplier, offset)``.
    solver_name : str
        QP solver name passed to Pink.
    """

    robot: pin.robot_wrapper.RobotWrapper
    joint_indices: Dict[str, int]
    tip_links: Sequence[str]
    wrist_link: str
    root_link: str
    mimic_pairs: Sequence[Tuple[int, int, float, float]]
    solver_name: str


_PINK_CONTEXT: Dict[str, PinkArmIkContext] = {}
_PINK_SOLVER_NAME: Optional[str] = None


def select_solver(requested: Optional[str]) -> str:
    """Resolve QP solver name for Pink.

    Parameters
    ----------
    requested : str or None
        Requested solver name with shape=(). ``None`` picks ``osqp`` if available.

    Returns
    -------
    str
        Solver name with shape=().
    """

    installed = list(available_solvers)
    if requested is not None:
        return requested
    if "osqp" in installed:
        return "osqp"
    return installed[0]


def build_pin_robot(hand_name: str) -> pin.robot_wrapper.RobotWrapper:
    """Build Pinocchio robot wrapper from hand URDF.

    Parameters
    ----------
    hand_name : str
        Hand name key in ``HAND_CONFIGS`` with shape=().

    Returns
    -------
    pin.robot_wrapper.RobotWrapper, shape=()
        Loaded Pinocchio robot wrapper.
    """

    config = HAND_CONFIGS[hand_name]
    package_dirs = [os.path.dirname(cast(str, config["urdf_path"])), ASSETS_DIR]
    return pin.RobotWrapper.BuildFromURDF(
        filename=cast(str, config["urdf_path"]),
        package_dirs=package_dirs,
        root_joint=None,
    )


def compute_joint_indices(robot: pin.robot_wrapper.RobotWrapper, joint_names: Sequence[str]) -> Dict[str, int]:
    """Map joint names to Pinocchio configuration indices.

    Parameters
    ----------
    robot : pin.robot_wrapper.RobotWrapper
        Pinocchio wrapper with shape=().
    joint_names : Sequence[str], shape=(J,)
        Joint names in model order.

    Returns
    -------
    dict[str, int], shape=(J,)
        Mapping name->q index.
    """

    model = robot.model
    return {name: model.joints[model.getJointId(name)].idx_q for name in joint_names}


def normalized_to_configuration(
    model: HandKinematicsModel,
    robot: pin.robot_wrapper.RobotWrapper,
    joint_indices: Dict[str, int],
    normalized: torch.Tensor,
) -> Configuration:
    """Convert normalized qpos to Pink configuration.

    Parameters
    ----------
    model : HandKinematicsModel
        FK model with matching joint order.
    robot : pin.robot_wrapper.RobotWrapper
        Pinocchio robot wrapper.
    joint_indices : dict[str, int], shape=(J,)
        Joint name to q index map.
    normalized : torch.Tensor, shape=(D,), dtype=float32
        Normalized independent joints.

    Returns
    -------
    pink.Configuration, shape=()
        Pink configuration object.
    """

    clamped = torch.clamp(normalized, -1.0 + 1.0e-3, 1.0 - 1.0e-3)
    angles_map = model._normalized_to_all_joint_angles(clamped.unsqueeze(0))
    q = robot.q0.copy()
    model_struct = robot.model
    for joint_name, angles in angles_map.items():
        if joint_name in joint_indices:
            index = joint_indices[joint_name]
        else:
            joint_id = model_struct.getJointId(joint_name)
            index = model_struct.joints[joint_id].idx_q
            joint_indices[joint_name] = index
        q[index] = angles[0].item()
    return Configuration(robot.model, robot.data, q)


def configuration_to_normalized(
    model: HandKinematicsModel,
    joint_indices: Dict[str, int],
    configuration: Configuration,
) -> torch.Tensor:
    """Convert Pink configuration back to normalized independent joints.

    Parameters
    ----------
    model : HandKinematicsModel
        FK model with joint order.
    joint_indices : dict[str, int], shape=(J,)
        Joint name to q index map.
    configuration : pink.Configuration
        Current Pink configuration with shape=().

    Returns
    -------
    torch.Tensor, shape=(D,), dtype=float32
        Normalized independent joints.
    """

    angles = torch.tensor([configuration.q[joint_indices[name]] for name in model.joint_name_order()], dtype=torch.float32)
    return model.angles_to_normalized(angles)


def _resolve_pink_solver_name() -> str:
    """Return cached Pink solver name.

    Parameters
    ----------
    None

    Returns
    -------
    str
        Solver name with shape=().
    """

    global _PINK_SOLVER_NAME
    if _PINK_SOLVER_NAME is None:
        _PINK_SOLVER_NAME = select_solver(None)
    return _PINK_SOLVER_NAME


def get_pink_arm_context(hand_name: str, model: HandKinematicsModel) -> PinkArmIkContext:
    """Build or fetch cached Pink context for one hand.

    Parameters
    ----------
    hand_name : str
        Hand name key with shape=().
    model : HandKinematicsModel
        FK model instance for this hand.

    Returns
    -------
    PinkArmIkContext, shape=()
        Cached context object.
    """

    cached = _PINK_CONTEXT.get(hand_name)
    if cached is not None:
        return cached

    config = HAND_CONFIGS[hand_name]
    robot = build_pin_robot(hand_name)
    joint_indices = compute_joint_indices(robot, model.joint_name_order())
    tip_links = tuple(cast(Sequence[str], config["tip_links"]))
    wrist_link = cast(str, config["wrist_link"])
    root_link = cast(str, config["root_link"])
    model_struct = robot.model
    mimic_pairs = tuple(
        (
            model_struct.joints[model_struct.getJointId(spec.name)].idx_q,
            model_struct.joints[model_struct.getJointId(cast(str, spec.mimic_parent))].idx_q,
            spec.mimic_multiplier,
            spec.mimic_offset,
        )
        for spec in model.joint_specs.values()
        if spec.mimic_parent is not None
    )
    context = PinkArmIkContext(
        robot=robot,
        joint_indices=joint_indices,
        tip_links=tip_links,
        wrist_link=wrist_link,
        root_link=root_link,
        mimic_pairs=mimic_pairs,
        solver_name=_resolve_pink_solver_name(),
    )
    _PINK_CONTEXT[hand_name] = context
    return context


def pink_align_arm(
    hand_name: str,
    model: HandKinematicsModel,
    arm_seed: torch.Tensor,
    hand_fixed: torch.Tensor,
    target_alignment: torch.Tensor,
    target_rotation: torch.Tensor,
    pinch_pairs: Sequence[Tuple[int, int]],
    pair_weights: Optional[torch.Tensor],
    *,
    rotation_weight: float,
    iterations: int,
) -> torch.Tensor:
    """Solve arm IK to match weighted pinch midpoint and wrist rotation.

    Parameters
    ----------
    hand_name : str
        Hand name key with shape=().
    model : HandKinematicsModel
        FK model instance.
    arm_seed : torch.Tensor, shape=(A,), dtype=float32
        Initial normalized arm qpos.
    hand_fixed : torch.Tensor, shape=(H,), dtype=float32
        Fixed normalized hand qpos.
    target_alignment : torch.Tensor, shape=(3,), dtype=float32
        Desired pinch midpoint in world frame.
    target_rotation : torch.Tensor, shape=(3, 3), dtype=float32
        Desired wrist rotation matrix.
    pinch_pairs : Sequence[Tuple[int, int]], shape=(P, 2)
        Pinch pair indices.
    pair_weights : torch.Tensor or None, shape=(P,), dtype=float32
        Optional pair weights.
    rotation_weight : float
        Wrist orientation cost scalar with shape=().
    iterations : int
        Pink iteration count with shape=().

    Returns
    -------
    torch.Tensor, shape=(A,), dtype=float32
        Solved normalized arm qpos.
    """

    context = get_pink_arm_context(hand_name, model)
    dtype = arm_seed.dtype
    device = arm_seed.device
    arm_clamped = arm_seed.detach().clone()
    hand_clamped = hand_fixed.detach().clone()
    combined_seed = torch.cat([arm_clamped, hand_clamped], dim=0)
    arm_dof = int(arm_seed.shape[0])
    hand_dof = int(hand_fixed.shape[0])

    configuration = normalized_to_configuration(
        model=model,
        robot=context.robot,
        joint_indices=context.joint_indices,
        normalized=combined_seed.to(device="cpu", dtype=torch.float32),
    )

    wrist_current = configuration.get_transform_frame_to_world(context.wrist_link)
    wrist_target = pin.SE3(wrist_current.rotation.copy(), wrist_current.translation.copy())
    target_align_np = target_alignment.detach().to(device="cpu", dtype=torch.float64).numpy()
    target_rot_np = target_rotation.detach().to(device="cpu", dtype=torch.float64).numpy()

    fingertip_transforms = [configuration.get_transform_frame_to_world(tip_name) for tip_name in context.tip_links]
    if pair_weights is None:
        if len(pinch_pairs) > 0:
            weights_np = np.ones(len(pinch_pairs), dtype=np.float64)
        else:
            weights_np = np.zeros(0, dtype=np.float64)
    else:
        weights_np = pair_weights.detach().to(device="cpu", dtype=torch.float64).numpy()

    if weights_np.size > 0:
        weight_sum = float(np.sum(weights_np))
        weights_np = weights_np / weight_sum if weight_sum > 1.0e-8 else np.full_like(weights_np, 1.0 / max(1, weights_np.size))

    if len(pinch_pairs) == 0:
        pinch_world = wrist_current.translation.copy()
    else:
        pinch_world = np.zeros(3, dtype=np.float64)
        for weight, (thumb_idx, finger_idx) in zip(weights_np, pinch_pairs):
            thumb_pos = fingertip_transforms[thumb_idx].translation
            finger_pos = fingertip_transforms[finger_idx].translation
            pinch_world += weight * 0.5 * (thumb_pos + finger_pos)

    wrist_target.translation = wrist_current.translation + (target_align_np - pinch_world)
    wrist_target.rotation = target_rot_np

    wrist_task = FrameTask(
        frame=context.wrist_link,
        position_cost=1.0,
        orientation_cost=max(0.0, float(rotation_weight)),
    )
    wrist_task.set_target(wrist_target)
    tasks = [wrist_task]

    progression = torch.linspace(1.0 / max(1, arm_dof), 1.0, steps=arm_dof, dtype=torch.float64)
    arm_velocity_weights = progression.pow(float(PINK_ARM_WEIGHT_EXPONENT)).numpy()

    dt = PINK_DT
    iteration_count = max(1, int(iterations))
    q_min = configuration.model.lowerPositionLimit
    q_max = configuration.model.upperPositionLimit
    limit_mask = q_max > q_min + 1.0e-6

    for _ in range(iteration_count):
        velocity = solve_ik(configuration, tasks, dt, solver=context.solver_name)
        velocity_np = np.asarray(velocity, dtype=np.float64).copy()
        velocity_np[:arm_dof] *= arm_velocity_weights
        if hand_dof > 0:
            velocity_np[arm_dof : arm_dof + hand_dof] = 0.0
        if np.linalg.norm(velocity_np) < 1.0e-6:
            break
        q = pin.integrate(configuration.model, configuration.q, velocity_np * dt)
        for mimic_idx, parent_idx, multiplier, offset in context.mimic_pairs:
            q[mimic_idx] = q[parent_idx] * multiplier + offset
        q[limit_mask] = np.clip(q[limit_mask], q_min[limit_mask], q_max[limit_mask])
        configuration.update(q)

    normalized_solution = configuration_to_normalized(
        model=model,
        joint_indices=context.joint_indices,
        configuration=configuration,
    )
    arm_solution = normalized_solution[: arm_clamped.shape[0]]
    return torch.clamp(arm_solution.to(device=device, dtype=dtype), -1.0, 1.0)
