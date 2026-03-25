# BridgeRAE Agent Notes

## Code Layout

- `bridgerae/datasets/shapenet_pairs.py` contains the ShapeNet55 PoinTrPairs point-pair loader.
- `bridgerae/models/` contains the Point-MAE encoder wrapper, completion decoder, and latent DiT model.
- `bridgerae/training/` contains the stage-1 and stage-2 train and eval entry points.

## Dataset

- Expected dataset root: `/root/autodl-tmp/datasets/ShapeNet55_PoinTrPairs`.
- Training data layout: `train/complete/<taxonomy_id>/<model_id>.npy` paired with `train/partial/<taxonomy_id>/<model_id>/<view_id>.npy`.
- Test data layout mirrors the same structure under `test/`.
- Each partial view file is treated as an independent sample.

## Defaults

- Default input points: `2048`
- Default complete points: `8192`
- Default behavior trains on all categories unless `--class-id` is provided.
