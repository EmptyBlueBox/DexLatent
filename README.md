# HandLatent

Hand latent representation implementation of the paper "Cross-Hand Latent Representation for Vision-Language-Action Models"

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
uv run -m hand_latent.train --num_steps 5000 --checkpoint_interval 1000
```

Checkpoints are written to:

- `Checkpoints/<timestamp>/checkpoint_epoch_XXXX.pt`

## Inference and Visualization

By default, inference reads `Dataset/demo.npz` and visualizes:

- source trajectory (origin)
- four decoded trajectories (`xhand`, `ability`, `inspire`, `paxini`)

Run with a specific checkpoint:

```bash
uv run -m hand_latent.infer --ckpt Checkpoints/<timestamp>/checkpoint_epoch_XXXX.pt
```
