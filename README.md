# BridgeRAE

`BridgeRAE` is a working directory for a point cloud completion baseline that combines:

- `BridgeShape`: latent-space completion and transport-oriented conditional generation.
- `RAE`: frozen pretrained encoder + trainable decoder as the stage-1 representation autoencoder recipe.
- `Point-MAE`: grouped point tokenizer and transformer encoder as the first point-cloud encoder candidate.

## Baseline Goal

Input: partial point cloud `X_p \in R^{N_p x 3}`

Output: complete point cloud `X_c \in R^{N_c x 3}`

The first baseline is intentionally simple:

1. Use a Point-MAE-style grouped encoder as the frozen representation encoder.
2. Train a completion decoder from the encoder latent to the full point cloud.
3. Add latent normalization and latent noise augmentation following the RAE recipe.
4. Keep the latent-space diffusion / Schr\"odinger bridge stage as the next step after the deterministic baseline is stable.

## Current Files

- `docs/paper_notes.md`: paper and code reading notes.
- `configs/pointmae_rae_baseline.yaml`: first baseline config draft.

## Recommended Implementation Order

1. Build a paired dataset loader: partial / complete point cloud.
2. Lift Point-MAE encoder code into `BridgeRAE`.
3. Freeze the encoder and train only the completion decoder.
4. Train a deterministic latent-to-shape baseline with Chamfer + density-aware loss.
5. After the autoencoding baseline is stable, add latent bridge / diffusion.
