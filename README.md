# PC-ALM

![PC-ALM Figure 1](assets/fig1.png)

> **PC-ALM** aligns local predictive-coding updates with backpropagation by
> accumulating layer-local constraint errors in Lagrange multipliers.

PyTorch port of the official JAX reference implementation of
**Augmented Lagrangian Predictive Coding**.

[![arXiv](https://img.shields.io/badge/arXiv-2605.31022-b31b1b?style=flat-square)](https://arxiv.org/abs/2605.31022)
[![Blog](https://img.shields.io/badge/Blog-Sakana%20AI-1f6feb?style=flat-square)](https://pub.sakana.ai/pc-alm/)

This minimal reference covers the paper's residual MLP width/depth grid on MNIST and
Fashion-MNIST, with BP, PC, and PC-ALM on the same architecture. It uses the
paper's fixed `gamma0=1` parameterization.

## Installation

Run from a source checkout. Install
[uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
uv sync
```

To run the test suite:

```bash
uv sync --extra test
uv run pytest
```

The default environment uses CPU PyTorch. On a Linux NVIDIA host, install the
locked CUDA 13 build instead:

```bash
uv sync --no-group cpu --group gpu --extra test
uv run --no-sync pytest
```

Training selects CUDA when it is available and otherwise falls back to CPU.
Pass `--device cpu` or `--device cuda` to require a particular device.

## Data

The synthetic smoke test does not require downloaded data:

```bash
uv run python train.py --config configs/smoke.yaml
```

Download MNIST and [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)
from the repository root:

```bash
mkdir -p data/MNIST/raw data/FashionMNIST/raw
for file in train-images-idx3-ubyte train-labels-idx1-ubyte \
            t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  curl -fL "https://storage.googleapis.com/cvdf-datasets/mnist/${file}.gz" \
    -o "data/MNIST/raw/${file}.gz" || break
  curl -fL "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/${file}.gz" \
    -o "data/FashionMNIST/raw/${file}.gz" || break
done
```

Both uncompressed IDX files and `.gz` IDX files are supported. Use `--data-dir`
to point to another directory containing `MNIST/raw/` and `FashionMNIST/raw/`.

## Training

Reproduce one Fashion-MNIST cell: width `N=32`, depth `L=32`, ReLU, seed 0,
one epoch on the full dataset, and inference budget `T=2L`:

```bash
uv run python scripts/run_headline_grid.py --config configs/headline_fashion.yaml \
  --widths 32 --depths 32 --activations relu --seeds 0 --methods bp,pc,pcalm \
  --budget-rule 2L --state-lr-table configs/eta_best_by_cell.csv \
  --output-dir results/repro_fashion_n32_l32 --data-dir data
```

Results from the PyTorch CUDA reproduction at commit `6d1efd9` are shown below.
The run used one epoch over the full Fashion-MNIST training set and the command
above on the Flourish GPU `experiments` queue (run
`pc-alm-fashion-n32-l32-1e2a`):

| Method | Test accuracy | Gradient cosine to BP |
|---|---:|---:|
| BP | 77.02% | 1.000 |
| PC | 63.76% | 0.548 |
| PC-ALM | 76.31% | 0.942 |

`configs/eta_best_by_cell.csv` contains the paper's frozen activity step sizes
(`eta_h = 1/lambda_max`, median over seeds) for each dataset/activation/width/depth.
For custom runs, edit a YAML config or use `uv run python train.py --help` for
single-run options.

<details>
<summary>Full Fashion-MNIST grid (675 runs)</summary>

```bash
uv run python scripts/run_headline_grid.py --config configs/headline_fashion.yaml \
  --widths 8,16,32,64,128 --depths 8,16,32,64,128 \
  --activations linear,tanh,relu --seeds 0,1,2 --methods bp,pc,pcalm \
  --budget-rule 2L --state-lr-table configs/eta_best_by_cell.csv \
  --output-dir results/headline_fashion --data-dir data
```

</details>

Use `configs/headline_mnist.yaml` for MNIST.

## Evaluation

Each run writes:

```text
results/path-to-run/
  config.json
  metrics.csv
  summary.json
```

For grid runs, `cells.csv` summarizes all methods and seeds. A compact heatmap
can be generated with:

```bash
uv run python scripts/plot_headline_grid.py \
  --input results/repro_fashion_n32_l32/cells.csv \
  --output results/repro_fashion_n32_l32/gain_pcalm_minus_pc.png \
  --activation relu
```

Use the full grid's `cells.csv` for a heatmap across widths and depths.

## Citation

If you use this code, please cite:

```bibtex
@misc{seely2026pc-alm,
  title         = {Augmented Lagrangian Predictive Coding},
  author        = {Jeffrey Seely and Julian Gould},
  year          = {2026},
  eprint        = {2605.31022},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2605.31022},
}
```

## License

This project is released under the MIT License.
