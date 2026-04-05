# BridgeRAE Agent Notes

## Code Layout

- `bridgerae/datasets/shapenet_pairs.py` contains the ShapeNet55 PoinTrPairs point-pair loader.
- `bridgerae/models/` contains Point-MAE encoder wrapping, completion decoder, latent transport model, and latent normalizer.
- `bridgerae/models/latent_normalizer.py` — `LatentNormalizer`: channel-wise mean/std normalization for encoder latent tokens. Stats stored as registered buffers, saved/loaded with checkpoints.
- `bridgerae/training/` contains stage-1 and stage-2 train/eval entry points.

## Dataset

- Expected dataset root: `/root/autodl-tmp/datasets/ShapeNet55_PoinTrPairs`.
- Training data layout: `train/complete/<taxonomy_id>/<model_id>.npy` paired with `train/partial/<taxonomy_id>/<model_id>/<view_id>.npy`.
- Test data layout mirrors the same structure under `test/`.
- Each partial view file is treated as an independent sample.

## Defaults

- Default input points: `2048`
- Default complete points: `8192`
- Default behavior trains on all categories unless `--class-id` is provided.

## Two-Stage Training Architecture

- **Stage-1 (autoencoder):** `complete → encoder → normalize → (noise) → decoder → complete`. Decoder learns to reconstruct complete shapes from complete latent tokens. Encoder is frozen (Point-MAE pretrained); only decoder is trainable.
- **Stage-2 (transport):** `partial → encoder → normalize → transport → denormalize → decoder → complete`. Transport model learns to map partial latent to complete latent space. Decoder receives the same distribution it was trained on.
- Latent stats are collected over **complete** point clouds (not partial) to match the decoder's operating space.
- Stage-1 does NOT use partial points at all; the paired partial data is only used in stage-2.

## Latent Normalization & Noise Augmentation

- Stage-1 supports `--latent-normalize` to enable channel-wise latent normalization (RAE recipe).
- Stage-1 supports `--latent-noise-std` (e.g. `0.1`) to add Gaussian noise to normalized latents during training; improves decoder robustness for stage-2.
- `--latent-stats-batches` controls how many batches are used for stats collection (default: full training set).
- Normalizer state is saved inside stage-1 checkpoints under key `normalizer`.
- Stage-2 automatically loads the normalizer from the stage-1 checkpoint if present; no extra flags needed.
- Data flow with normalizer: `encoder → normalize → (noise, train only) → decoder` (stage-1), `encoder → normalize → transport → denormalize → decoder` (stage-2).

## Latent Transport

- `LatentTransportModel.transport_train()` — Euler integration **with** gradients, used in training so `recon_loss` backprops through the transport model.
- `LatentTransportModel.transport()` — Euler integration **without** gradients (`@torch.no_grad()`), used for inference/eval only.
- `TimeEmbedding` uses sinusoidal positional encoding → MLP (not raw scalar input).

## Training Extras

- **LR Scheduler:** Both stages support `--lr-scheduler cosine` (default) with `--warmup-epochs` (default 5) and `--lr-min` (default 1e-6). Use `--lr-scheduler none` for constant LR.
- **Repulsion Loss:** Stage-1 supports `--repulsion-weight` (default 0) and `--repulsion-k` (default 8) to penalize point clustering.

## Default Training Recipe

- Default PCN recipe should use `--latent-normalize` for stage-1.
- Default stage-1 batch size should be `256` unless a run explicitly overrides it.
- Default stage-1 recipe should use latent noise augmentation with `--latent-noise-std 0.1`.
- Default transport recipe should use the zero-init transport head in `bridgerae/models/latent_transport.py`, so stage-2 starts close to identity and learns residual corrections.
- Default stage-2 recipe should jointly fine-tune the decoder tail with `--decoder-train-mode last_n` and keep `--decoder-train-last-n 2` unless a run explicitly overrides it.
- Default stage-2 batch size should be `128` unless a run explicitly overrides it.
- For long runs, prefer `--no-save-epoch-checkpoints` and keep only `best.pth` / `last.pth` plus logs and exported CSV/JSON/PNG artifacts to avoid filling `/root/autodl-tmp`.
- For comparison experiments, always record both stage train curves and val curves, and include at least `loss`, `CD-L2`, `F1@1%`, and `IOU`. Include `EMD` when runtime permits.
