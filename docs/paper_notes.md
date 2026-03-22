# Paper Notes

## 1. Papers

### 1.1 BridgeShape

Paper title: `BridgeShape: Latent Diffusion Schrödinger Bridge for 3D Shape Completion`

Latest arXiv page checked: `2026-03-22 UTC`

- Local PDF in workspace: `BridgeShape.pdf`
- arXiv latest version is `v2`, updated on `2026-03-17`
- arXiv URL: <https://arxiv.org/abs/2506.23205>

Core takeaways:

1. It does not complete shapes directly in voxel space.
2. It first learns a compact latent representation of shape.
3. Completion is treated as a transport problem from incomplete shape distribution to complete shape distribution.
4. The paper uses a latent diffusion Schr\"odinger bridge to model this transition explicitly.
5. A stronger shape encoder matters; the paper emphasizes geometry-aware latent quality, not only the bridge process itself.

What is useful for us:

1. Work in latent space instead of direct point regression as the final target formulation.
2. Keep the condition path explicit: partial shape latent should not only be concatenated late; it should define the source distribution.
3. The baseline should separate:
   - stage-1 representation learning / autoencoding
   - stage-2 conditional latent generation or transport

What we do not need in the first baseline:

1. VQ-VAE discretization is not mandatory.
2. Schr\"odinger bridge is not required for the first working version.
3. Voxel representation is not required because our target is point cloud completion.

### 1.2 RAE

Paper title: `Diffusion Transformers with Representation Autoencoders`

Latest arXiv page checked: `2026-03-22 UTC`

- Local PDF in workspace: `RAE.pdf`
- arXiv URL: <https://arxiv.org/abs/2510.11690>

Core takeaways:

1. Replace a jointly-trained compressive VAE encoder with a frozen pretrained representation encoder.
2. Train only the decoder in stage 1.
3. High-dimensional semantic latents are usable if architecture and training are adjusted properly.
4. Decoder training should be robust to noisy continuous latents.
5. Latent normalization and decoder noise augmentation are important implementation details.

The parts that transfer well to point cloud completion:

1. Frozen encoder, trainable decoder.
2. Latent normalization statistics.
3. Latent noise augmentation during decoder training.
4. Stage split:
   - stage-1: representation autoencoder / reconstruction
   - stage-2: generative model in latent space

The parts that do not transfer directly:

1. Image ViT decoder does not map directly to point clouds.
2. LPIPS / GAN losses are image-specific.
3. DiT width and noise-schedule analysis are tied to image latent diffusion, not directly to point completion.

## 2. RAE Code Reading

Repository: `/root/autodl-tmp/projects/RAE`

### 2.1 Stage-1 core module

File: `/root/autodl-tmp/projects/RAE/src/stage1/rae.py`

Important design points:

1. `RAE.encode()`:
   - normalizes input
   - runs a frozen pretrained encoder
   - optionally adds latent noise during training
   - reshapes token sequence to latent grid
   - optionally normalizes latent with dataset statistics
2. `RAE.decode()`:
   - unnormalizes latent if needed
   - maps latent tokens through a trainable decoder
   - reconstructs to input space
3. `RAE.forward()` is just `encode -> decode`

This is the exact pattern we want for the point-cloud baseline.

### 2.2 Stage-1 training script

File: `/root/autodl-tmp/projects/RAE/src/train_stage1.py`

Important design points:

1. Encoder is explicitly frozen.
2. Decoder is explicitly trainable.
3. EMA is maintained on the stage-1 model.
4. Reconstruction training is enhanced with perceptual and adversarial losses for images.

For point clouds, we should keep:

1. frozen encoder
2. trainable decoder
3. optional EMA
4. online reconstruction evaluation

For point clouds, we should replace image losses with:

1. Chamfer L1 or L2
2. density-aware loss or repulsion regularization
3. optional coarse-to-fine reconstruction loss

## 3. Point-MAE Code Reading

Repository: `/root/autodl-tmp/projects/Point-MAE`

### 3.1 Main reusable pieces

File: `/root/autodl-tmp/projects/Point-MAE/models/Point_MAE.py`

Important reusable modules:

1. `Group`
   - FPS samples group centers
   - KNN gathers local neighborhoods
2. `Encoder`
   - encodes each local point group into a group token
3. `TransformerEncoder`
   - processes visible tokens with positional embedding
4. `TransformerDecoder`
   - reconstructs masked groups from visible tokens + mask tokens

For our task, the strongest reusable idea is:

1. Convert a point cloud into local group tokens.
2. Encode the partial shape into a latent token sequence.
3. Decode from these latent tokens to a complete point set.

### 3.2 What should be reused as-is

1. Grouping strategy: FPS + KNN
2. Group encoder
3. Positional embedding from 3D center
4. Transformer encoder block definitions

### 3.3 What should not be reused as-is

1. Masked autoencoding objective
2. Reconstruction head that predicts only masked local patches
3. Pretraining runner logic tied to self-supervised masking
4. Dataset loader that only returns a single complete shape

## 4. Proposed Baseline

### 4.1 High-level architecture

`partial point cloud -> frozen Point-MAE encoder -> latent tokens -> trainable completion decoder -> complete point cloud`

Concretely:

1. Input partial point cloud `X_p`
2. Use Point-MAE `Group` to get:
   - local neighborhoods
   - local centers
3. Use Point-MAE local `Encoder` to get group tokens
4. Use Point-MAE transformer encoder to get latent tokens `Z_p`
5. Apply RAE-style latent normalization and optional latent noise augmentation
6. Use a trainable completion decoder to predict the full shape `X_c_hat`

### 4.2 Decoder choice for baseline-v1

A practical first decoder is:

1. transformer latent decoder
2. coarse query set generation
3. MLP head to output coarse complete points

Two reasonable variants:

1. Global decoder:
   - pool latent tokens
   - predict `M` coarse points directly
   - simplest baseline, fast to debug
2. Query-based decoder:
   - create `Q` learned query tokens
   - cross-attend to encoder latents
   - predict `Q` output points
   - closer to latent generation design

Recommendation:

- Start with query-based decoder because it preserves tokenized latent structure better and is more aligned with later latent bridge/diffusion.

### 4.3 Losses

First baseline losses:

1. `ChamferDistanceL1(X_c_hat, X_c)`
2. optional `ChamferDistanceL1(X_partial_preserved, X_p)` style preservation regularizer
3. optional repulsion / density regularizer

### 4.4 Stage split

#### Stage 1

Deterministic partial-to-complete reconstruction with frozen encoder and trainable decoder.

Goal:

- verify that Point-MAE latent is sufficient for completion

#### Stage 2

Latent conditional generation / bridge model.

Suggested formulation:

1. encode partial to `Z_p`
2. encode complete to `Z_c`
3. train a latent model that maps source latent distribution around `Z_p` toward target latent `Z_c`

This is where BridgeShape ideas should enter more explicitly.

## 5. Minimal Experimental Plan

### 5.1 Baseline-A

Frozen Point-MAE encoder + trainable query decoder, supervised by complete-shape Chamfer loss.

Purpose:

- verify pure RAE idea on point completion

### 5.2 Baseline-B

Same as A, plus latent normalization and latent noise augmentation.

Purpose:

- test whether RAE training tricks improve stability and generalization

### 5.3 Baseline-C

Train an auxiliary complete-shape encoder branch and match partial latent to complete latent with latent alignment loss.

Purpose:

- make the future bridge stage easier

### 5.4 Stage-2 Candidate

Latent diffusion / Schr\"odinger-bridge-like model over latent tokens.

Purpose:

- inject BridgeShape’s transport idea after stage-1 baseline is reliable

## 6. Practical Conclusion

The best first implementation path is:

1. do not start from BridgeShape’s full diffusion bridge
2. first build a point-cloud RAE:
   - frozen Point-MAE encoder
   - trainable completion decoder
   - latent normalization
   - latent noise augmentation
3. only after that, add a latent conditional generative model between partial and complete latents

This keeps the first baseline aligned with both papers while staying technically manageable.

## 7. Encoder Structure Note

### 7.1 Point-MAE Encoder Data Flow

The current BridgeRAE Point-MAE encoder uses the following structure:

```text
partial point cloud: (B, N, 3)
    |
    |-- FPS sample 64 anchors
    v
centers: (B, 64, 3)
    |
    |-- kNN / nearest-neighbor grouping, 32 points per anchor
    v
local point groups: (B, 64, 32, 3)
    |
    |-- subtract center coordinate
    v
neighborhoods: (B, 64, 32, 3)
    |
    |-- local point encoder
    v
group tokens: (B, 64, 384)
    |
    |-- add positional embedding from centers
    |-- transformer encoder
    v
latent tokens: (B, 64, 384)
```

### 7.2 Meaning of Each Tensor

- `centers`: global anchor positions of local patches
- `neighborhoods`: local patch geometry, already centered by subtracting `centers`
- `tokens`: latent patch representations after local encoding and transformer aggregation

A useful way to read them is:

- `centers` answer `where`
- `neighborhoods` answer `what local geometry is here`
- `tokens` answer `what this local region means after contextual encoding`

### 7.3 BridgeRAE Decoder Interface

The completion decoder should consume:

- encoder latent tokens `(B, G, C)`
- encoder centers `(B, G, 3)`

and predict:

- complete point cloud `(B, M, 3)`

This is why the current BridgeRAE encoder returns all three tensors:

- `tokens`
- `centers`
- `neighborhoods`

For stage-1 completion training, the decoder mainly needs `tokens` and `centers`.
