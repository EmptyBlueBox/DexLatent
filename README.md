# DexLatent

Official implementation of DexLatent for the paper **XL-VLA: Cross-Hand Latent Representation for Vision-Language-Action Models** (CVPR 2026 Highlight ✨).

![teaser](Assets/videos/teaser.gif)

## News!

- [x] `2026.04.08`: Selected as a CVPR 2026 Highlight ✨.
- [x] `2026.02.27`: Released code and [project website](https://xl-vla.github.io).
- [x] `2026.02.22`: Accepted to CVPR 2026.

## Inference and Visualization

Run with our pretrained checkpoint:

```bash
uv run -m HandLatent.infer
```

By default, inference reads `Dataset/demo.npz` and visualizes:

- source trajectory (origin)
- four decoded trajectories (`xhand`, `ability`, `inspire`, `paxini`)

## Train

Run the training script with:

```bash
uv run -m HandLatent.train
```

Checkpoints are written to:

- `Checkpoints/<timestamp>/checkpoint_epoch_XXXX.pt`
