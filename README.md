# HandLatent

A minimal standalone project for xarm-only latent training and inference.

## Supported Hands

- `xarm7_xhand_left` / `xarm7_xhand_right`
- `xarm7_ability_left` / `xarm7_ability_right`
- `xarm7_inspire_left` / `xarm7_inspire_right`
- `xarm7_paxini_left` / `xarm7_paxini_right`

## Install

```bash
uv sync
```

## Train

Run with default parameters (aligned with the reference repo):

```bash
uv run -m hand_latent.train
```

Example with custom arguments:

```bash
uv run -m hand_latent.train --num_steps 2000 --checkpoint_interval 200
```

Checkpoints are written to:

- `Checkpoints/<timestamp>/checkpoint_epoch_XXXX.pt`

## Inference and Visualization

By default, inference reads `Dataset/inspire-11_08.npz` and visualizes:

- source trajectory (origin)
- four decoded trajectories (`xhand`, `ability`, `inspire`, `paxini`)

All five trajectories are rendered with zero offset (overlapped).

Run with a specific checkpoint:

```bash
uv run -m hand_latent.infer --ckpt Checkpoints/<timestamp>/checkpoint_epoch_XXXX.pt
```

Or use the latest checkpoint automatically:

```bash
uv run -m hand_latent.infer
```

Common option:

```bash
uv run -m hand_latent.infer --side right
```
