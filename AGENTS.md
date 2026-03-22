# Repository Guidelines

## Project Structure & Module Organization
- Core code lives in `bridgerae/`.
- `bridgerae/datasets/` contains `3D-EPN` loaders: voxel input in `epn_voxel.py`, point-cloud conversion in `epn_point.py`.
- `bridgerae/models/` contains the frozen Point-MAE encoder, stage-1 completion decoder, and stage-2 latent transport model.
- `bridgerae/training/` contains train/eval scripts, losses, metrics, and plotting utilities.
- `configs/` stores baseline config drafts. `docs/` stores paper and design notes.
- Generated artifacts belong under `outputs/` and are ignored by git.

## Build, Test, and Development Commands
- Activate the expected environment: `conda activate bridgerae`
- Stage 1 train: `PYTHONPATH=/root/autodl-tmp/projects/BridgeRAE:/root/autodl-tmp/projects/Point-MAE/extensions/emd conda run -n bridgerae python -m bridgerae.training.stage1_train --class-id 03001627 --epochs 30 --batch-size 256 --amp`
- Stage 1 eval: `conda run -n bridgerae python -m bridgerae.training.stage1_eval --checkpoint outputs/stage1_ep30_bs256_tqdm/best.pth --class-id 03001627 --split test`
- Stage 2 train: `conda run -n bridgerae python -m bridgerae.training.stage2_train --stage1-ckpt outputs/stage1_ep30_bs256_tqdm/best.pth --class-id 03001627 --epochs 30 --batch-size 128 --amp`
- Stage 2 eval: `conda run -n bridgerae python -m bridgerae.training.stage2_eval --stage2-ckpt outputs/stage2_ep30/best.pth --stage1-ckpt outputs/stage1_ep30_bs256_tqdm/best.pth --class-id 03001627 --split test`

## Coding Style & Naming Conventions
- Use Python with 4-space indentation and type hints where practical.
- Prefer small, explicit modules over large utility files.
- Use `snake_case` for files, functions, and variables; use `PascalCase` for classes.
- Keep training logs JSON-like and machine-readable. Do not hardcode output files outside `outputs/`.

## Testing Guidelines
- No dedicated unit-test suite exists yet; use smoke runs as regression checks.
- For loader/model changes, run a short train sanity check with `--max-steps 1` or `--max-steps 3`.
- For metric changes, run `stage1_eval.py` or `stage2_eval.py` with `--max-batches 1` before full evaluation.

## Commit & Pull Request Guidelines
- Follow the current git style: short, imperative commit subjects such as `Initial BridgeRAE baseline` or `Improve README init guide and metric scaling`.
- Keep commits scoped to one logical change.
- PRs should include: purpose, changed scripts/modules, exact train/eval commands used, and key metrics if behavior changes.

## Security & Configuration Tips
- Expected dataset root: `/root/autodl-tmp/projects/DiffComplete/data/3d_epn`.
- Expected pretrained encoder checkpoint: `/root/autodl-tmp/projects/Point-MAE/checkpoint/pretrain.pth`.
- Do not commit checkpoints, cached outputs, or dataset files.
