# pw-hiera

2D reimplementation of [Hiera](https://github.com/facebookresearch/hiera) with MAE pretraining, applied to Gray-Scott reaction-diffusion simulation data.

## Project structure

- `src/pw_hiera/hiera/` — Hiera encoder and MAE decoder implementation
- `src/pw_hiera/experiments/` — training, data loading, visualization
- `configs/` — JSON configs for encoder and decoder
- `tests/` — unit tests

## Usage

1. Prepare a dataset directory with `train.hdf5` and `val.hdf5`.
   Expected HDF5 structure: `/sims/sim* -> (time, channels, height, width)`

2. Run MAE pretraining:
```bash
uv run train-mae \
    --mae-config configs/mae/basic_decoder.json \
    --hiera-config configs/hiera/gray_scott_hiera_tiny.json \
    --data-path training/data/gray_scott \
    --batch-size 32 \
    --n-epochs 200 \
    --n-warmup-epochs 10 \
    --lr 0.0005 \
    --min-lr 0.000001 \
    --weight-decay 0.05 \
    --mask-ratio 0.6 \
    --seed 42 \
    --output-dir runs \
    --name gs_hiera_tiny_2
```

