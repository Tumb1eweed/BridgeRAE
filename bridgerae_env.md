# bridgerae Conda Environment

## Environment Name

`bridgerae`

## Activate

```bash
conda activate bridgerae
```

## Current Location

`/root/autodl-tmp/.cache/.conda/envs/bridgerae`

## Verified Core Versions

- Python: `3.10.x`
- torch: `2.5.1+cu124`
- torchvision: `0.20.1+cu124`
- torchaudio: `2.5.1+cu124`
- open3d: `0.19.0`
- timm: `0.4.5`
- omegaconf: `2.3.0`
- hydra-core: `1.3.2`
- transformers: `4.56.2`
- accelerate: `0.23.0`
- einops: `0.8.2`
- wandb: `0.25.1`
- torchdiffeq: `0.2.5`
- torchmetrics: `1.9.0`
- trimesh: `4.4.9`
- PyMCubes / `mcubes`: `0.1.4`
- scikit-image: `0.22.0`
- pandas: `2.2.2`
- opencv-python / `cv2`: `4.13.0`

## Point-MAE Related Extensions

- `pointnet2_ops`: import verified
- `chamfer`: import verified
- `knn_cuda`: package present; runtime check failed only because `torch.cuda.is_available()` was `False` during the import test

## Notes

- This environment was built by recreating the `pointmae` base stack and then adding the missing `RAE` and `DiffComplete` dependencies.
- We downgraded `numpy` to `1.26.4` to resolve ABI conflicts for `mcubes` and `scikit-image`.
- Future BridgeRAE development should use this environment by default.
