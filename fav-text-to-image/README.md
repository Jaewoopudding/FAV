<div align="center">

<h1>FAV for Text-to-Image Generation</h1>

</div>

## Overview

Official implementation of **FAV** (Aligning Few Step Generative Model via Amortizing Sample Based Variational Inference)
for reward alignment of **text-to-image image generators**. 


## Installation

```bash
conda create -n fav python=3.10
conda activate fav
pip install -e .   # torch + diffusers (>=0.32, for Sana-Sprint / SCM) + hpsv2 + transformers
```

### Pre-trained models

All weights are fetched from HuggingFace on first run and cached locally:

| Component | Source |
|---|---|
| Sana-Sprint  | `backbone.model_repo` (default `Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers`; experiment overlays use the 1.6B checkpoint) |
| HPSv2 reward | `hpsv2` package (auto-download) |
| NSFW reward  | `Falconsai/nsfw_image_detection` (auto-download) |

Select a different checkpoint by overriding `backbone.model_repo`.

The NSFW experiments additionally need the NSFW prompt set. Get it from
[Yuchen413/text2image_safety](https://github.com/Yuchen413/text2image_safety)
and place it under `assets/prompts/`.

## Usage

```bash
# FAV with the HPSv2 reward (single process)
python -m scripts.train +experiment=sana_fav_hpsv2 seed=0

# Multi-GPU via accelerate (per-experiment launcher; override GPUs/port via env).
# Same effective batch and gradient on any GPU count: multi-GPU shards the
# per-prompt particles, single-GPU chunks them.
bash bash_scripts/sana_fav_hpsv2.sh
CUDA_VISIBLE_DEVICES=0 bash bash_scripts/sana_fav_hpsv2.sh
```

Hyperparameters for each run are fixed in the experiment overlays under
`configs/experiment/`; any field can be overridden on the command line, e.g.
`python -m scripts.train +experiment=sana_fav_hpsv2 algorithm.beta=2.0 runtime.micro_batch_size=2`.

## Reproducing the paper

Each command below reproduces one experiment (hyperparameters set in the overlay).
For multi-GPU runs use the matching launcher in `bash_scripts/`.

```bash
# FAV
python -m scripts.train +experiment=sana_fav_hpsv2  seed=0
python -m scripts.train +experiment=sana_fav_nsfw   seed=0

# DRaFT
python -m scripts.train +experiment=sana_draft_hpsv2  seed=0
python -m scripts.train +experiment=sana_draft_nsfw   seed=0
```


## Acknowledgments

The backbone and rewards are adapted from their official releases:

- **Sana-Sprint** — [NVlabs/Sana](https://github.com/NVlabs/Sana)
- **HPSv2**  — [tgxs002/HPSv2](https://github.com/tgxs002/HPSv2)
- **NSFW**  — [Falconsai/nsfw_image_detection](https://huggingface.co/Falconsai/nsfw_image_detection)
