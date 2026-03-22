# BridgeRAE

`BridgeRAE` is a point cloud completion baseline that combines:

- `BridgeShape`: latent-space completion and transport-oriented conditional generation.
- `RAE`: frozen pretrained encoder + trainable decoder as the stage-1 recipe.
- `Point-MAE`: grouped point tokenizer and transformer encoder as the first point-cloud encoder candidate.

## Goal

Input: partial point cloud `X_p \in R^{N_p x 3}`

Output: complete point cloud `X_c \in R^{N_c x 3}`

Current pipeline has two stages:

1. `Stage 1`: frozen Point-MAE encoder + trainable completion decoder.
2. `Stage 2`: latent transport from partial latent to complete latent, then decode with the stage-1 decoder.

## Environment

Recommended conda environment: `bridgerae`

Activate it with:

```bash
conda activate bridgerae
```

Environment notes are in [bridgerae_env.md](bridgerae_env.md).

Core dependencies used by the current code:

- `python 3.10`
- `torch 2.5.1+cu124`
- `torchvision 0.20.1+cu124`
- `torchaudio 2.5.1+cu124`
- `timm 0.4.5`
- `einops`
- `omegaconf`
- `hydra-core`
- `open3d`
- `trimesh`
- `mcubes`
- `scikit-image`
- `opencv-python`
- `pointnet2_ops`
- Point-MAE `emd` CUDA extension

If you need to rebuild the environment manually, the main external code dependencies are:

- Point-MAE repo: `/root/autodl-tmp/projects/Point-MAE`
- DiffComplete repo: `/root/autodl-tmp/projects/DiffComplete`
- RAE repo: `/root/autodl-tmp/projects/RAE`

## Dataset

Current experiments use `3D-EPN` from DiffComplete.

Dataset root used by the code:

```bash
/root/autodl-tmp/projects/DiffComplete/data/3d_epn
```

### Dataset format

The voxel-level source dataset is read by [epn_voxel.py](bridgerae/datasets/epn_voxel.py).

Each sample is a `.pth` file storing:

- `input_sdf`: shape `(2, 32, 32, 32)`
  - channel 0: `abs(sdf)`
  - channel 1: `sign(sdf)`
- `target_df`: shape `(32, 32, 32)`
  - complete-shape distance field

Split files are expected at:

```bash
/root/autodl-tmp/projects/DiffComplete/data/3d_epn/splits
```

Current code supports per-class splits such as:

- `train_03001627.txt`
- `test_03001627.txt`

### Point-cloud conversion

The point-level dataset is built in [epn_point.py](bridgerae/datasets/epn_point.py).

Current default point counts are:

- `partial_points`: `768`
- `complete_points`: `2048`

Generated point clouds are normalized jointly using the complete shape centroid and scale.

## Project Layout

```text
BridgeRAE/
├── bridgerae/
│   ├── datasets/
│   │   ├── epn_voxel.py
│   │   └── epn_point.py
│   ├── models/
│   │   ├── pointmae_encoder.py
│   │   ├── completion_decoder.py
│   │   └── latent_transport.py
│   └── training/
│       ├── stage1_train.py
│       ├── stage1_eval.py
│       ├── stage2_train.py
│       ├── stage2_eval.py
│       ├── losses.py
│       └── metrics.py
├── configs/
├── docs/
├── README.md
└── implementation_plan.md
```

## Code Flow

### Stage 1

Training flow:

1. Load `partial_points` and `complete_points` from `EPNPointCloudDataset`.
2. Encode `partial_points` with [PointMAEEncoder](bridgerae/models/pointmae_encoder.py).
3. Decode latent tokens with [QueryCompletionDecoder](bridgerae/models/completion_decoder.py).
4. Compute point-set reconstruction loss in [losses.py](bridgerae/training/losses.py).
5. Train only the decoder. The encoder stays frozen.

Inference flow:

1. `partial_points -> PointMAEEncoder`
2. `latent tokens -> QueryCompletionDecoder`
3. output `complete_points_hat`

### Stage 2

Training flow:

1. Encode `partial_points` to source latent tokens.
2. Reuse the same patch centers to encode `complete_points` to aligned target latent tokens.
3. Train [LatentTransportModel](bridgerae/models/latent_transport.py) to predict latent transport / velocity.
4. Transported latent tokens are passed through the frozen stage-1 decoder.
5. Optimize both:
   - latent flow loss
   - point reconstruction loss

Inference flow:

1. `partial_points -> PointMAEEncoder`
2. `source latent -> LatentTransportModel.transport()`
3. `transported latent -> stage-1 decoder`
4. output `complete_points_hat`

## Metrics

Metrics are defined in [metrics.py](bridgerae/training/metrics.py).

Current eval outputs include:

- `chamfer_distance`
  - reported as `original_value * 100`
- `chamfer_distance_l1`
- `emd`
- `iou`
  - reported as `original_value * 100`

## Training

### Stage 1 training

Formal 30-epoch training example:

```bash
PYTHONPATH=/root/autodl-tmp/projects/BridgeRAE:/root/autodl-tmp/projects/Point-MAE/extensions/emd \
conda run -n bridgerae python -m bridgerae.training.stage1_train \
  --class-id 03001627 \
  --epochs 30 \
  --batch-size 256 \
  --num-workers 8 \
  --log-every 25 \
  --amp \
  --save-dir /root/autodl-tmp/projects/BridgeRAE/outputs/stage1_ep30_bs256_tqdm
```

### Stage 2 training

Formal 30-epoch training example:

```bash
PYTHONPATH=/root/autodl-tmp/projects/BridgeRAE:/root/autodl-tmp/projects/Point-MAE/extensions/emd \
conda run -n bridgerae python -m bridgerae.training.stage2_train \
  --class-id 03001627 \
  --stage1-ckpt /root/autodl-tmp/projects/BridgeRAE/outputs/stage1_ep30_bs256_tqdm/best.pth \
  --epochs 30 \
  --batch-size 128 \
  --num-workers 8 \
  --log-every 25 \
  --amp \
  --save-dir /root/autodl-tmp/projects/BridgeRAE/outputs/stage2_ep30
```

## Testing / Evaluation

### Stage 1 eval

```bash
PYTHONPATH=/root/autodl-tmp/projects/BridgeRAE:/root/autodl-tmp/projects/Point-MAE/extensions/emd \
conda run -n bridgerae python -m bridgerae.training.stage1_eval \
  --checkpoint /root/autodl-tmp/projects/BridgeRAE/outputs/stage1_ep30_bs256_tqdm/best.pth \
  --class-id 03001627 \
  --split test \
  --batch-size 128 \
  --num-workers 8
```

### Stage 2 eval

```bash
PYTHONPATH=/root/autodl-tmp/projects/BridgeRAE:/root/autodl-tmp/projects/Point-MAE/extensions/emd \
conda run -n bridgerae python -m bridgerae.training.stage2_eval \
  --class-id 03001627 \
  --stage2-ckpt /root/autodl-tmp/projects/BridgeRAE/outputs/stage2_ep30/best.pth \
  --stage1-ckpt /root/autodl-tmp/projects/BridgeRAE/outputs/stage1_ep30_bs256_tqdm/best.pth \
  --split test \
  --batch-size 128 \
  --num-workers 8
```

### Evaluate all stage-1 checkpoints and plot curves

```bash
PYTHONPATH=/root/autodl-tmp/projects/BridgeRAE:/root/autodl-tmp/projects/Point-MAE/extensions/emd \
conda run -n bridgerae python -m bridgerae.training.stage1_eval_all \
  --ckpt-dir /root/autodl-tmp/projects/BridgeRAE/outputs/stage1_ep30_bs256_tqdm \
  --class-id 03001627 \
  --split test \
  --batch-size 128 \
  --num-workers 8
```

```bash
PYTHONPATH=/root/autodl-tmp/projects/BridgeRAE:/root/autodl-tmp/projects/Point-MAE/extensions/emd \
conda run -n bridgerae python -m bridgerae.training.plot_stage1_curves \
  --ckpt-dir /root/autodl-tmp/projects/BridgeRAE/outputs/stage1_ep30_bs256_tqdm \
  --eval-jsonl /root/autodl-tmp/projects/BridgeRAE/outputs/stage1_ep30_bs256_tqdm/test_eval_metrics.jsonl \
  --output /root/autodl-tmp/projects/BridgeRAE/outputs/stage1_ep30_bs256_tqdm/loss_curves.png
```

## Current Best Results

Class: `03001627`

### Stage 1 best checkpoint

- `chamfer_distance`: `7.3593`
- `chamfer_distance_l1`: `0.1080`
- `emd`: `0.0604`
- `iou`: `47.0546`

### Stage 2 best checkpoint

- `chamfer_distance`: `7.9999`
- `chamfer_distance_l1`: `0.1171`
- `emd`: `0.0646`
- `iou`: `41.9743`

At the current state, `stage1` is still stronger than `stage2`.

## Notes

- Generated checkpoints and outputs are ignored by `.gitignore`.
- The code expects CUDA.
- `torch.load` currently prints `FutureWarning` in several places; this does not block training or evaluation.
