<div align="center">

<h1>FAV for Image Generation</h1>

</div>

## Overview

Official implementation of **FAV** (Aligning Few Step Generative Model via Amortizing Sample Based Variational Inference)
for reward alignment of **class-conditional image generators**. The repository reproduces the paper results across four
generative backbones and four alignment algorithms, with three reward functions
(aesthetic score, JPEG compressibility, JPEG incompressibility).

| Backbone | FAV | DRaFT | Flow-GRPO | Adjoint Matching |
|---|:---:|:---:|:---:|:---:|
| iMF (PyTorch)         | ✓ | ✓ | ✓ | ✓ |
| IMM (PyTorch)         | ✓ | ✓ | — | — |
| StyleGAN-XL (PyTorch) | ✓ | ✓ | — | — |
| Drifting (JAX)        | ✓ | ✓ | — | — |


## Installation

FAV is built on PyTorch (all backbones except Drifting, which is JAX-native).

```bash
conda create -n fav-image python=3.10
conda activate fav-image
pip install -e .

# JAX backend (Drifting backbone only) — install separately:
pip install -e ".[jax]"
```

### Pre-trained models

Download the backbone checkpoints for ImageNet256 and the reward MLP, and place them under `assets/`:

| File | Backbone / use | Source |
|---|---|---|
| `iMF-XL-2.pth`                  | iMF         | https://github.com/Lyy-iiis/imeanflow/tree/torch |
| `imm-im256.pkl`                 | IMM         | https://github.com/lumalabs/imm |
| `stylegan_xl_imagenet256.pkl`   | StyleGAN-XL | https://github.com/autonomousvision/stylegan-xl |
| `drifting_latent_L_sota/`       | Drifting    | https://github.com/lambertae/drifting (or `init_from=hf://latent_L_sota`) |
| `sac+logos+ava1-l14-linearMSE.pth` | aesthetic reward MLP | [LAION-AI/aesthetic-predictor](https://github.com/LAION-AI/aesthetic-predictor) |

Checkpoint paths are set per backbone in `configs/backbone/*.yaml` and the reward
MLP in `configs/reward/aesthetic.yaml`.

## Usage

```bash
# FAV on iMF with the aesthetic reward (single process)
python -m scripts.train +experiment=imf_fav_aesthetic seed=0

# Multi-GPU via accelerate (per-experiment launcher; override GPUs/port via env)
bash bash_scripts/imf_fav_aesthetic.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 MAIN_PORT=29501 bash bash_scripts/imf_fav_aesthetic.sh

# Drifting is JAX-native — use the JAX entrypoint and env:
python -m scripts.train_drifting +experiment=drifting_fav_aesthetic seed=0
```

Hyperparameters for each run are fixed in the experiment overlays under
`configs/experiment/`; any field can be overridden on the command line, e.g.
`python -m scripts.train +experiment=imf_fav_aesthetic algorithm.beta=1.0`.

## Reproducing the paper

Each command below reproduces one experiment (hyperparameters set in the overlay).
For multi-GPU runs use the matching launcher in `bash_scripts/`.

```bash
# ── iMF ─────────────────────────────────────────────────────────────
python -m scripts.train +experiment=imf_fav_aesthetic               seed=0
python -m scripts.train +experiment=imf_draft_aesthetic             seed=0
python -m scripts.train +experiment=imf_flow_grpo_aesthetic         seed=0
python -m scripts.train +experiment=imf_adjoint_matching_aesthetic  seed=0
python -m scripts.train +experiment=imf_fav_jpeg_compress           seed=0   # FAV + NES estimator
python -m scripts.train +experiment=imf_fav_jpeg_incompress         seed=0
python -m scripts.train +experiment=imf_flow_grpo_jpeg_compress     seed=0
python -m scripts.train +experiment=imf_flow_grpo_jpeg_incompress   seed=0

# ── IMM ─────────────────────────────────────────────────────────────
python -m scripts.train +experiment=imm_fav_aesthetic    seed=0
python -m scripts.train +experiment=imm_draft_aesthetic  seed=0

# ── StyleGAN-XL ─────────────────────────────────────────────────────
python -m scripts.train +experiment=stylegan_xl_fav_aesthetic    seed=0
python -m scripts.train +experiment=stylegan_xl_draft_aesthetic  seed=0

# ── Drifting (JAX) ──────────────────────────────────────────────────
python -m scripts.train_drifting +experiment=drifting_fav_aesthetic    seed=0
python -m scripts.train_drifting +experiment=drifting_draft_aesthetic  seed=0
```

The non-differentiable JPEG rewards use a zeroth-order NES gradient estimator
(`algorithm.gradient_estimator.enabled=true`, set in the JPEG overlays).

## Acknowledgments

The backbones are adapted from their official releases:

- **iMF** — [Lyy-iiis/imeanflow](https://github.com/Lyy-iiis/imeanflow/tree/torch)
- **IMM** — [lumalabs/imm](https://github.com/lumalabs/imm)
- **StyleGAN-XL** — [autonomousvision/stylegan-xl](https://github.com/autonomousvision/stylegan-xl)
- **Drifting** — [lambertae/drifting](https://github.com/lambertae/drifting)

The DRaFT, Flow-GRPO, and Adjoint Matching baselines follow their respective papers.
