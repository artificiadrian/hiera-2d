# hiera-2d

A minimal from scratch reimplementation of [Hiera](https://github.com/facebookresearch/hiera) in 2D along with MAE pretraining functionality and config system.

---

## Usage

Run MAE pretraining:

```bash
uv run train-mae \
    --mae-config configs/mae/basic.json \
    --hiera-config configs/hiera/tiny.json \
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
    --name gs_hiera_tiny
```
