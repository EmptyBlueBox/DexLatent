"""Training entry point for minimal hand latent project."""

from __future__ import annotations

import argparse

import torch

from HandLatent.model import CrossEmbodimentTrainer, TrainingConfig


def main() -> None:
    """Run latent training with default parameters matching the reference repo.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Trains the model and writes checkpoints.
    """

    parser = argparse.ArgumentParser(description="Train minimal cross-embodiment hand latent model.")
    parser.add_argument("--num_steps", type=int, default=10_000)
    parser.add_argument("--checkpoint_interval", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--pinch_template_count", type=int, default=2048)
    parser.add_argument("--pinch_template_iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    hand_names = [
        "xarm7_xhand_right",
        "xarm7_ability_right",
        "xarm7_inspire_right",
        "xarm7_paxini_right",
    ]
    config = TrainingConfig(
        num_steps=args.num_steps,
        checkpoint_interval=args.checkpoint_interval,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        pinch_template_count=args.pinch_template_count,
        pinch_template_iterations=args.pinch_template_iterations,
    )
    trainer = CrossEmbodimentTrainer(hand_names, config)
    trainer.train()


if __name__ == "__main__":
    main()
