# BridgeRAE Implementation Plan

## Goal

Build a point cloud completion baseline that combines:

- BridgeShape's latent-space completion perspective
- RAE's frozen-encoder and trainable-decoder stage-1 design
- Point-MAE's grouped point encoder

## Recommended Order

### 1. Build the paired dataset

Implement a dataset that returns:

- partial point cloud
- complete point cloud
- optional taxonomy / model id

Requirements:

- support train / val split
- normalize both partial and complete shapes consistently
- support controllable partial generation strategy if raw partial data is unavailable

### 2. Port the Point-MAE encoder into BridgeRAE

Reuse the following modules first:

- FPS + KNN grouping
- local group encoder
- transformer encoder
- center positional embedding

Goal:

- encode a partial point cloud into latent tokens reliably

### 3. Freeze the encoder and train only the completion decoder

Follow the RAE stage-1 recipe:

- frozen encoder
- trainable decoder
- optional EMA
- latent normalization
- latent noise augmentation

Goal:

- establish a point-cloud version of RAE for completion

### 4. Train a deterministic completion baseline

Baseline form:

- input: partial point cloud
- latent: Point-MAE encoder tokens
- output: complete point cloud

Recommended losses:

- Chamfer L1 or L2
- optional density / repulsion regularization
- optional partial-preservation regularization

Goal:

- verify that the latent representation is sufficient for completion

### 5. Add latent bridge / diffusion after the autoencoding baseline is stable

After stage-1 works, add a stage-2 latent model inspired by BridgeShape:

- encode partial shape to partial latent
- encode complete shape to complete latent
- learn a conditional latent transition from partial to complete

Goal:

- bring in BridgeShape's transport / generative formulation on top of a stable representation baseline

## Practical Rule

Do not start from the full BridgeShape Schrödinger bridge implementation.

Start from the point-cloud RAE baseline first:

1. paired dataset
2. Point-MAE encoder
3. frozen encoder + trainable decoder
4. deterministic completion training
5. latent bridge / diffusion
