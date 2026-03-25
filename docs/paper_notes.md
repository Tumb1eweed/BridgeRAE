# Project Notes

## 1. Reference Papers

### 1.1 RAE

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
3. DiT width and image-specific noise-schedule analysis need revalidation for point completion.

### 1.2 Point-MAE

Repository: `/root/autodl-tmp/projects/Point-MAE`

Main reusable pieces from `models/Point_MAE.py`:

1. `Group`
   - FPS samples group centers
   - KNN gathers local neighborhoods
2. `Encoder`
   - encodes each local point group into a group token
3. `TransformerEncoder`
   - processes visible tokens with positional embedding
4. Center-based positional encoding

For this project, the strongest reusable idea is:

1. Convert a point cloud into local group tokens.
2. Encode the partial shape into a latent token sequence.
3. Decode from these latent tokens to a complete point set.

## 2. Stage-1 Baseline

### 2.1 High-level architecture

`partial point cloud -> frozen Point-MAE encoder -> latent tokens -> trainable completion decoder -> complete point cloud`

Concretely:

1. Input partial point cloud `X_p`
2. Use Point-MAE `Group` to get local neighborhoods and local centers
3. Use the Point-MAE local encoder to get group tokens
4. Use the Point-MAE transformer encoder to get latent tokens `Z_p`
5. Apply RAE-style latent normalization and optional latent noise augmentation
6. Use a trainable completion decoder to predict the full shape `X_c_hat`

### 2.2 Decoder choice

A practical decoder is:

1. transformer latent decoder
2. coarse query set generation
3. MLP head to output coarse complete points

Recommendation:

- Use a query-based decoder because it preserves tokenized latent structure and matches the later latent diffusion stage.

### 2.3 Losses

First baseline losses:

1. `ChamferDistanceL1(X_c_hat, X_c)` or `ChamferDistanceL2(X_c_hat, X_c)`
2. optional partial-preservation regularizer
3. optional repulsion or density regularizer

## 3. Stage-2 Diffusion Model

### 3.1 Formulation

Stage-2 uses conditional latent diffusion with a DiT denoiser.

1. Encode partial shapes to conditional latent tokens `Z_p`.
2. Encode complete shapes to aligned target latent tokens `Z_c`.
3. Sample a timestep `t` and corrupt `Z_c` with Gaussian noise.
4. Train the DiT model to predict the injected noise from the noisy latent and the partial latent condition.
5. Decode predicted clean latents with the frozen stage-1 decoder.

### 3.2 Practical training notes

1. Keep the encoder frozen in stage 2.
2. Train the latent DiT and reuse the stage-1 decoder for reconstruction supervision.
3. Use direct `x0` reconstruction loss during training so the reconstruction term backpropagates through the denoiser.
4. Use iterative denoising only for validation and test-time sampling.

## 4. Experimental Plan

### 4.1 Baseline-A

Frozen Point-MAE encoder + trainable query decoder, supervised by complete-shape Chamfer loss.

Purpose:

- verify the point-cloud RAE baseline

### 4.2 Baseline-B

Same as A, plus latent normalization and latent noise augmentation.

Purpose:

- test whether RAE training tricks improve stability and generalization

### 4.3 Baseline-C

Conditional latent diffusion with DiT over latent tokens.

Purpose:

- model one-to-many completion uncertainty in latent space

## 5. Practical Conclusion

The implementation path is:

1. first build a point-cloud RAE:
   - frozen Point-MAE encoder
   - trainable completion decoder
   - latent normalization
   - latent noise augmentation
2. then add a conditional latent diffusion model between partial and complete latents

This keeps the project technically manageable while preserving a strong latent generative stage.

## 6. Encoder Structure Note

### 6.1 Point-MAE Encoder Data Flow

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

### 6.2 Meaning of Each Tensor

- `centers`: global anchor positions of local patches
- `neighborhoods`: local patch geometry, already centered by subtracting `centers`
- `tokens`: latent patch representations after local encoding and transformer aggregation

A useful way to read them is:

- `centers` answer `where`
- `neighborhoods` answer `what local geometry is here`
- `tokens` answer `what this local region means after contextual encoding`
