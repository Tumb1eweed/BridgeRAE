# BridgeRAE

`BridgeRAE` is a point cloud completion baseline that combines:

- `BridgeShape`: latent-space completion and transport-oriented conditional generation.
- `RAE`: frozen pretrained encoder + trainable decoder as the stage-1 recipe.
- `Point-MAE`: grouped point tokenizer and transformer encoder as the encoder backbone.

## Goal

Input: partial point cloud `X_p \in R^{N_p x 3}`

Output: complete point cloud `X_c \in R^{N_c x 3}`

Current pipeline has two stages:

1. `Stage 1`: frozen Point-MAE encoder + trainable completion decoder.
2. `Stage 2`: latent transport from partial latent to complete latent, then decode with the stage-1 decoder.

## Dataset

Current experiments use `ShapeNet55_PoinTrPairs`, following the processed `ShapeNet55-34` point-pair layout used by PoinTr.

Dataset root used by the code:

```bash
/root/autodl-tmp/datasets/ShapeNet55_PoinTrPairs
```

The dataset loader is implemented in `bridgerae/datasets/shapenet_pairs.py`.

Expected directory layout:

```bash
/root/autodl-tmp/datasets/ShapeNet55_PoinTrPairs/
├── train/
│   ├── complete/<taxonomy_id>/<model_id>.npy
│   └── partial/<taxonomy_id>/<model_id>/<view_id>.npy
└── test/
    ├── complete/<taxonomy_id>/<model_id>.npy
    └── partial/<taxonomy_id>/<model_id>/<view_id>.npy
```

BridgeRAE treats each partial view file such as `00.npy` to `07.npy` as an independent sample paired with the matching complete shape.

Typical shapes in the local dataset:

- `partial`: `(2048, 3)`
- `complete`: `(8192, 3)`

BridgeRAE currently samples them to:

- `partial_points`: `2048`
- `complete_points`: `8192`

The partial and complete clouds are normalized jointly using the complete shape centroid and scale.

## Project Layout

```text
BridgeRAE/
├── bridgerae/
│   ├── datasets/
│   │   └── shapenet_pairs.py
│   ├── models/
│   │   ├── pointmae_encoder.py
│   │   ├── completion_decoder.py
│   │   └── latent_transport.py
│   └── training/
│       ├── stage1_train.py
│       ├── stage1_eval.py
│       ├── stage1_eval_all.py
│       ├── stage2_train.py
│       ├── stage2_eval.py
│       ├── losses.py
│       └── metrics.py
├── configs/
├── docs/
└── README.md
```

## Code Flow

### Stage 1

1. Load `partial_points` and `complete_points` from `ShapeNetPointCloudDataset`.
2. Encode `partial_points` with `PointMAEEncoder`.
3. Decode latent tokens with `QueryCompletionDecoder`.
4. Compute point-set reconstruction loss in `bridgerae/training/losses.py`.
5. Train only the decoder while the encoder stays frozen.

### Stage 2

1. Encode `partial_points` to source latent tokens.
2. Reuse the same patch centers to encode `complete_points` to aligned target latent tokens.
3. Train `LatentTransportModel` to predict latent transport velocity.
4. Decode transported latent tokens with the frozen stage-1 decoder.
5. Optimize latent flow loss and point reconstruction loss together.

## Training

Stage-1 example:

```bash
PYTHONPATH=/root/autodl-tmp/projects/BridgeRAE:/root/autodl-tmp/projects/Point-MAE/extensions/emd \
conda run -n bridgerae python -m bridgerae.training.stage1_train \
  --epochs 30 \
  --batch-size 256 \
  --num-workers 8 \
  --log-every 25 \
  --amp \
  --save-dir /root/autodl-tmp/projects/BridgeRAE/outputs/stage1_ep30_bs256_tqdm
```

Stage-2 example:

```bash
PYTHONPATH=/root/autodl-tmp/projects/BridgeRAE:/root/autodl-tmp/projects/Point-MAE/extensions/emd \
conda run -n bridgerae python -m bridgerae.training.stage2_train \
  --stage1-ckpt /root/autodl-tmp/projects/BridgeRAE/outputs/stage1_ep30_bs256_tqdm/best.pth \
  --epochs 30 \
  --batch-size 128 \
  --num-workers 8 \
  --log-every 25 \
  --amp \
  --save-dir /root/autodl-tmp/projects/BridgeRAE/outputs/stage2_ep30
```

Default training and evaluation use all categories. You can still pass `--class-id <taxonomy_id>` to restrict to one ShapeNet category.

Existing 2048-point stage-1 or stage-2 checkpoints are not shape-compatible with the new 8192-point decoder and should be retrained.
