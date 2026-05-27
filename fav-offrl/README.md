<div align="center">

<h1>FAV for Reinforcement Learning</h1>

</div>

## Overview

  JAX/Flax implementation of **FAV**, a single-step generative policy that samples from the Q-tilted distribution by amortizing SVGD particle updates into the actor via fixed-point regression. The repository reproduces the offline (56 tasks; OGBench + D4RL AntMaze and offline-to-online (30 OGBench tasks) results from the paper.


## Installation

FAV requires **Python 3.11+** (for `jax==0.7.1`) and is based on JAX. The main
dependencies are `jax >= 0.4.26`, `ogbench == 1.1.0`, and `gymnasium == 0.29.1`.

```bash
conda create -n fav python=3.11
conda activate fav
pip install -r requirements.txt
```

> [!NOTE]
> To use D4RL environments, you additionally need to set up MuJoCo 2.1.0.

## Usage

The FAV agent lives in [agents/fav.py](agents/fav.py), and reference
implementations of four baselines (IQL, ReBRAC, IFQL, RLPD) live alongside it.
A short example:

```bash
# FAV on OGBench scene-play (offline-to-online)
python main.py --agent=agents/fav.py \
  --env_name=scene-play-singletask-v0 \
  --offline_steps=1000000 --online_steps=1000000 \
  --agent.bandwidth_method=fixed --agent.bandwidth=0.1 --agent.beta=0.5

# FAV-Adaptive on the same task
python main.py --agent=agents/fav.py \
  --env_name=scene-play-singletask-v0 \
  --offline_steps=1000000 --online_steps=1000000 \
  --agent.bandwidth_method=scott --agent.beta=0.5
```

## Hyperparameters

The agent has two variants:

| Variant         | Bandwidth                  | Temperature       |
|-----------------|----------------------------|----------------|
| **FAV**          | fixed `σ`                  | fixed `β`      |
| **FAV-Adaptive** | Scott-rule `σ` (auto)      | fixed `β`      |

Set `--agent.bandwidth_method=fixed --agent.bandwidth=<σ>` for FAV, or
`--agent.bandwidth_method=scott` for FAV-Adaptive.

`σ` is ignored for FAV-Adaptive — the bandwidth is set automatically by the
Scott rule.

### OGBench (10 environments × 5 tasks each = 50 tasks)

| Environment                       | FAV (σ, β)  | FAV-Adaptive β |
|-----------------------------------|:-----------:|:--------------:|
| `antmaze-large-navigate`          | (0.5, 2)    | 1              |
| `antmaze-giant-navigate`          | (0.5, 1)    | 1              |
| `humanoidmaze-medium-navigate`    | (0.5, 1)    | 2              |
| `humanoidmaze-large-navigate`     | (0.5, 1)    | 2              |
| `antsoccer-arena-navigate`        | (1, 1)      | 2              |
| `cube-single-play`                | (0.05, 1)   | 0.5            |
| `cube-double-play`                | (0.1, 0.5)  | 0.5            |
| `scene-play`                      | (0.1, 0.5)  | 0.5            |
| `puzzle-3x3-play`                 | (0.5, 0.5)  | 2              |
| `puzzle-4x4-play`                 | (1, 0.5)    | 0.1            |

### D4RL AntMaze (6 single-task environments)

| Environment                  | FAV (σ, β) | FAV-Adaptive β |
|------------------------------|:----------:|:--------------:|
| `antmaze-umaze-v2`           | (0.5, 1)   | 3              |
| `antmaze-umaze-diverse-v2`   | (1, 1)     | 3              |
| `antmaze-medium-play-v2`     | (1, 3)     | 2              |
| `antmaze-medium-diverse-v2`  | (1, 3)     | 3              |
| `antmaze-large-play-v2`      | (1, 3)     | 5              |
| `antmaze-large-diverse-v2`   | (1, 3)     | 5              |

### Per-environment flags

| Environment family               | Extra flags                            |
|----------------------------------|----------------------------------------|
| `antmaze-large-navigate-*`       | `--agent.q_agg=min`                    |
| `antmaze-giant-navigate-*`       | `--agent.q_agg=min --agent.discount=0.995` |
| `humanoidmaze-*-navigate-*`      | `--agent.discount=0.995`               |
| `antsoccer-arena-navigate-*`     | `--agent.discount=0.995`               |
| (others)                         | (none)                                 |

For new environments we recommend starting with FAV-Adaptive and sweeping
`β ∈ [0.1, 0.5, 1, 2]`.

## Reproducing the offline benchmark (56 tasks)

Each command below runs 1 M offline gradient steps with no online phase.

### FAV (fixed bandwidth) — 56 commands

```bash
# OGBench (50 tasks)
# antmaze-large-navigate (σ=0.5, β=2)
python main.py --agent=agents/fav.py --env_name=antmaze-large-navigate-singletask-task1-v0 --offline_steps=1000000 --agent.q_agg=min --agent.bandwidth_method=fixed --agent.bandwidth=0.5 --agent.beta=2
python main.py --agent=agents/fav.py --env_name=antmaze-large-navigate-singletask-task2-v0 --offline_steps=1000000 --agent.q_agg=min --agent.bandwidth_method=fixed --agent.bandwidth=0.5 --agent.beta=2
python main.py --agent=agents/fav.py --env_name=antmaze-large-navigate-singletask-task3-v0 --offline_steps=1000000 --agent.q_agg=min --agent.bandwidth_method=fixed --agent.bandwidth=0.5 --agent.beta=2
python main.py --agent=agents/fav.py --env_name=antmaze-large-navigate-singletask-task4-v0 --offline_steps=1000000 --agent.q_agg=min --agent.bandwidth_method=fixed --agent.bandwidth=0.5 --agent.beta=2
python main.py --agent=agents/fav.py --env_name=antmaze-large-navigate-singletask-task5-v0 --offline_steps=1000000 --agent.q_agg=min --agent.bandwidth_method=fixed --agent.bandwidth=0.5 --agent.beta=2

# antmaze-giant-navigate (σ=0.5, β=1)
python main.py --agent=agents/fav.py --env_name=antmaze-giant-navigate-singletask-task1-v0 --offline_steps=1000000 --agent.discount=0.995 --agent.q_agg=min --agent.bandwidth_method=fixed --agent.bandwidth=0.5 --agent.beta=1
python main.py --agent=agents/fav.py --env_name=antmaze-giant-navigate-singletask-task2-v0 --offline_steps=1000000 --agent.discount=0.995 --agent.q_agg=min --agent.bandwidth_method=fixed --agent.bandwidth=0.5 --agent.beta=1
python main.py --agent=agents/fav.py --env_name=antmaze-giant-navigate-singletask-task3-v0 --offline_steps=1000000 --agent.discount=0.995 --agent.q_agg=min --agent.bandwidth_method=fixed --agent.bandwidth=0.5 --agent.beta=1
python main.py --agent=agents/fav.py --env_name=antmaze-giant-navigate-singletask-task4-v0 --offline_steps=1000000 --agent.discount=0.995 --agent.q_agg=min --agent.bandwidth_method=fixed --agent.bandwidth=0.5 --agent.beta=1
python main.py --agent=agents/fav.py --env_name=antmaze-giant-navigate-singletask-task5-v0 --offline_steps=1000000 --agent.discount=0.995 --agent.q_agg=min --agent.bandwidth_method=fixed --agent.bandwidth=0.5 --agent.beta=1

# humanoidmaze-medium-navigate (σ=0.5, β=1)
python main.py --agent=agents/fav.py --env_name=humanoidmaze-medium-navigate-singletask-task1-v0 --offline_steps=1000000 --agent.discount=0.995 --agent.bandwidth_method=fixed --agent.bandwidth=0.5 --agent.beta=1
python main.py --agent=agents/fav.py --env_name=humanoidmaze-medium-navigate-singletask-task2-v0 --offline_steps=1000000 --agent.discount=0.995 --agent.bandwidth_method=fixed --agent.bandwidth=0.5 --agent.beta=1
python main.py --agent=agents/fav.py --env_name=humanoidmaze-medium-navigate-singletask-task3-v0 --offline_steps=1000000 --agent.discount=0.995 --agent.bandwidth_method=fixed --agent.bandwidth=0.5 --agent.beta=1
python main.py --agent=agents/fav.py --env_name=humanoidmaze-medium-navigate-singletask-task4-v0 --offline_steps=1000000 --agent.discount=0.995 --agent.bandwidth_method=fixed --agent.bandwidth=0.5 --agent.beta=1
python main.py --agent=agents/fav.py --env_name=humanoidmaze-medium-navigate-singletask-task5-v0 --offline_steps=1000000 --agent.discount=0.995 --agent.bandwidth_method=fixed --agent.bandwidth=0.5 --agent.beta=1

# humanoidmaze-large-navigate (σ=0.5, β=1)
python main.py --agent=agents/fav.py --env_name=humanoidmaze-large-navigate-singletask-task1-v0 --offline_steps=1000000 --agent.discount=0.995 --agent.bandwidth_method=fixed --agent.bandwidth=0.5 --agent.beta=1
python main.py --agent=agents/fav.py --env_name=humanoidmaze-large-navigate-singletask-task2-v0 --offline_steps=1000000 --agent.discount=0.995 --agent.bandwidth_method=fixed --agent.bandwidth=0.5 --agent.beta=1
python main.py --agent=agents/fav.py --env_name=humanoidmaze-large-navigate-singletask-task3-v0 --offline_steps=1000000 --agent.discount=0.995 --agent.bandwidth_method=fixed --agent.bandwidth=0.5 --agent.beta=1
python main.py --agent=agents/fav.py --env_name=humanoidmaze-large-navigate-singletask-task4-v0 --offline_steps=1000000 --agent.discount=0.995 --agent.bandwidth_method=fixed --agent.bandwidth=0.5 --agent.beta=1
python main.py --agent=agents/fav.py --env_name=humanoidmaze-large-navigate-singletask-task5-v0 --offline_steps=1000000 --agent.discount=0.995 --agent.bandwidth_method=fixed --agent.bandwidth=0.5 --agent.beta=1

# antsoccer-arena-navigate (σ=1, β=1)
python main.py --agent=agents/fav.py --env_name=antsoccer-arena-navigate-singletask-task1-v0 --offline_steps=1000000 --agent.discount=0.995 --agent.bandwidth_method=fixed --agent.bandwidth=1 --agent.beta=1
python main.py --agent=agents/fav.py --env_name=antsoccer-arena-navigate-singletask-task2-v0 --offline_steps=1000000 --agent.discount=0.995 --agent.bandwidth_method=fixed --agent.bandwidth=1 --agent.beta=1
python main.py --agent=agents/fav.py --env_name=antsoccer-arena-navigate-singletask-task3-v0 --offline_steps=1000000 --agent.discount=0.995 --agent.bandwidth_method=fixed --agent.bandwidth=1 --agent.beta=1
python main.py --agent=agents/fav.py --env_name=antsoccer-arena-navigate-singletask-task4-v0 --offline_steps=1000000 --agent.discount=0.995 --agent.bandwidth_method=fixed --agent.bandwidth=1 --agent.beta=1
python main.py --agent=agents/fav.py --env_name=antsoccer-arena-navigate-singletask-task5-v0 --offline_steps=1000000 --agent.discount=0.995 --agent.bandwidth_method=fixed --agent.bandwidth=1 --agent.beta=1

# cube-single-play (σ=0.05, β=1)
python main.py --agent=agents/fav.py --env_name=cube-single-play-singletask-task1-v0 --offline_steps=1000000 --agent.bandwidth_method=fixed --agent.bandwidth=0.05 --agent.beta=1
python main.py --agent=agents/fav.py --env_name=cube-single-play-singletask-task2-v0 --offline_steps=1000000 --agent.bandwidth_method=fixed --agent.bandwidth=0.05 --agent.beta=1
python main.py --agent=agents/fav.py --env_name=cube-single-play-singletask-task3-v0 --offline_steps=1000000 --agent.bandwidth_method=fixed --agent.bandwidth=0.05 --agent.beta=1
python main.py --agent=agents/fav.py --env_name=cube-single-play-singletask-task4-v0 --offline_steps=1000000 --agent.bandwidth_method=fixed --agent.bandwidth=0.05 --agent.beta=1
python main.py --agent=agents/fav.py --env_name=cube-single-play-singletask-task5-v0 --offline_steps=1000000 --agent.bandwidth_method=fixed --agent.bandwidth=0.05 --agent.beta=1

# cube-double-play (σ=0.1, β=0.5)
python main.py --agent=agents/fav.py --env_name=cube-double-play-singletask-task1-v0 --offline_steps=1000000 --agent.bandwidth_method=fixed --agent.bandwidth=0.1 --agent.beta=0.5
python main.py --agent=agents/fav.py --env_name=cube-double-play-singletask-task2-v0 --offline_steps=1000000 --agent.bandwidth_method=fixed --agent.bandwidth=0.1 --agent.beta=0.5
python main.py --agent=agents/fav.py --env_name=cube-double-play-singletask-task3-v0 --offline_steps=1000000 --agent.bandwidth_method=fixed --agent.bandwidth=0.1 --agent.beta=0.5
python main.py --agent=agents/fav.py --env_name=cube-double-play-singletask-task4-v0 --offline_steps=1000000 --agent.bandwidth_method=fixed --agent.bandwidth=0.1 --agent.beta=0.5
python main.py --agent=agents/fav.py --env_name=cube-double-play-singletask-task5-v0 --offline_steps=1000000 --agent.bandwidth_method=fixed --agent.bandwidth=0.1 --agent.beta=0.5

# scene-play (σ=0.1, β=0.5)
python main.py --agent=agents/fav.py --env_name=scene-play-singletask-task1-v0 --offline_steps=1000000 --agent.bandwidth_method=fixed --agent.bandwidth=0.1 --agent.beta=0.5
python main.py --agent=agents/fav.py --env_name=scene-play-singletask-task2-v0 --offline_steps=1000000 --agent.bandwidth_method=fixed --agent.bandwidth=0.1 --agent.beta=0.5
python main.py --agent=agents/fav.py --env_name=scene-play-singletask-task3-v0 --offline_steps=1000000 --agent.bandwidth_method=fixed --agent.bandwidth=0.1 --agent.beta=0.5
python main.py --agent=agents/fav.py --env_name=scene-play-singletask-task4-v0 --offline_steps=1000000 --agent.bandwidth_method=fixed --agent.bandwidth=0.1 --agent.beta=0.5
python main.py --agent=agents/fav.py --env_name=scene-play-singletask-task5-v0 --offline_steps=1000000 --agent.bandwidth_method=fixed --agent.bandwidth=0.1 --agent.beta=0.5

# puzzle-3x3-play (σ=0.5, β=0.5)
python main.py --agent=agents/fav.py --env_name=puzzle-3x3-play-singletask-task1-v0 --offline_steps=1000000 --agent.bandwidth_method=fixed --agent.bandwidth=0.5 --agent.beta=0.5
python main.py --agent=agents/fav.py --env_name=puzzle-3x3-play-singletask-task2-v0 --offline_steps=1000000 --agent.bandwidth_method=fixed --agent.bandwidth=0.5 --agent.beta=0.5
python main.py --agent=agents/fav.py --env_name=puzzle-3x3-play-singletask-task3-v0 --offline_steps=1000000 --agent.bandwidth_method=fixed --agent.bandwidth=0.5 --agent.beta=0.5
python main.py --agent=agents/fav.py --env_name=puzzle-3x3-play-singletask-task4-v0 --offline_steps=1000000 --agent.bandwidth_method=fixed --agent.bandwidth=0.5 --agent.beta=0.5
python main.py --agent=agents/fav.py --env_name=puzzle-3x3-play-singletask-task5-v0 --offline_steps=1000000 --agent.bandwidth_method=fixed --agent.bandwidth=0.5 --agent.beta=0.5

# puzzle-4x4-play (σ=1, β=0.5)
python main.py --agent=agents/fav.py --env_name=puzzle-4x4-play-singletask-task1-v0 --offline_steps=1000000 --agent.bandwidth_method=fixed --agent.bandwidth=1 --agent.beta=0.5
python main.py --agent=agents/fav.py --env_name=puzzle-4x4-play-singletask-task2-v0 --offline_steps=1000000 --agent.bandwidth_method=fixed --agent.bandwidth=1 --agent.beta=0.5
python main.py --agent=agents/fav.py --env_name=puzzle-4x4-play-singletask-task3-v0 --offline_steps=1000000 --agent.bandwidth_method=fixed --agent.bandwidth=1 --agent.beta=0.5
python main.py --agent=agents/fav.py --env_name=puzzle-4x4-play-singletask-task4-v0 --offline_steps=1000000 --agent.bandwidth_method=fixed --agent.bandwidth=1 --agent.beta=0.5
python main.py --agent=agents/fav.py --env_name=puzzle-4x4-play-singletask-task5-v0 --offline_steps=1000000 --agent.bandwidth_method=fixed --agent.bandwidth=1 --agent.beta=0.5

# D4RL AntMaze (6 tasks)
python main.py --agent=agents/fav.py --env_name=antmaze-umaze-v2          --offline_steps=1000000 --agent.bandwidth_method=fixed --agent.bandwidth=0.5 --agent.beta=1
python main.py --agent=agents/fav.py --env_name=antmaze-umaze-diverse-v2  --offline_steps=1000000 --agent.bandwidth_method=fixed --agent.bandwidth=1   --agent.beta=1
python main.py --agent=agents/fav.py --env_name=antmaze-medium-play-v2    --offline_steps=1000000 --agent.bandwidth_method=fixed --agent.bandwidth=1   --agent.beta=3
python main.py --agent=agents/fav.py --env_name=antmaze-medium-diverse-v2 --offline_steps=1000000 --agent.bandwidth_method=fixed --agent.bandwidth=1   --agent.beta=3
python main.py --agent=agents/fav.py --env_name=antmaze-large-play-v2     --offline_steps=1000000 --agent.bandwidth_method=fixed --agent.bandwidth=1   --agent.beta=3
python main.py --agent=agents/fav.py --env_name=antmaze-large-diverse-v2  --offline_steps=1000000 --agent.bandwidth_method=fixed --agent.bandwidth=1   --agent.beta=3
```

### FAV-Adaptive (Scott-rule bandwidth) — 56 commands

```bash
# OGBench (50 tasks)
# antmaze-large-navigate (β=1)
python main.py --agent=agents/fav.py --env_name=antmaze-large-navigate-singletask-task1-v0 --offline_steps=1000000 --agent.q_agg=min --agent.bandwidth_method=scott --agent.beta=1
python main.py --agent=agents/fav.py --env_name=antmaze-large-navigate-singletask-task2-v0 --offline_steps=1000000 --agent.q_agg=min --agent.bandwidth_method=scott --agent.beta=1
python main.py --agent=agents/fav.py --env_name=antmaze-large-navigate-singletask-task3-v0 --offline_steps=1000000 --agent.q_agg=min --agent.bandwidth_method=scott --agent.beta=1
python main.py --agent=agents/fav.py --env_name=antmaze-large-navigate-singletask-task4-v0 --offline_steps=1000000 --agent.q_agg=min --agent.bandwidth_method=scott --agent.beta=1
python main.py --agent=agents/fav.py --env_name=antmaze-large-navigate-singletask-task5-v0 --offline_steps=1000000 --agent.q_agg=min --agent.bandwidth_method=scott --agent.beta=1

# antmaze-giant-navigate (β=1)
python main.py --agent=agents/fav.py --env_name=antmaze-giant-navigate-singletask-task1-v0 --offline_steps=1000000 --agent.discount=0.995 --agent.q_agg=min --agent.bandwidth_method=scott --agent.beta=1
python main.py --agent=agents/fav.py --env_name=antmaze-giant-navigate-singletask-task2-v0 --offline_steps=1000000 --agent.discount=0.995 --agent.q_agg=min --agent.bandwidth_method=scott --agent.beta=1
python main.py --agent=agents/fav.py --env_name=antmaze-giant-navigate-singletask-task3-v0 --offline_steps=1000000 --agent.discount=0.995 --agent.q_agg=min --agent.bandwidth_method=scott --agent.beta=1
python main.py --agent=agents/fav.py --env_name=antmaze-giant-navigate-singletask-task4-v0 --offline_steps=1000000 --agent.discount=0.995 --agent.q_agg=min --agent.bandwidth_method=scott --agent.beta=1
python main.py --agent=agents/fav.py --env_name=antmaze-giant-navigate-singletask-task5-v0 --offline_steps=1000000 --agent.discount=0.995 --agent.q_agg=min --agent.bandwidth_method=scott --agent.beta=1

# humanoidmaze-medium-navigate (β=2)
python main.py --agent=agents/fav.py --env_name=humanoidmaze-medium-navigate-singletask-task1-v0 --offline_steps=1000000 --agent.discount=0.995 --agent.bandwidth_method=scott --agent.beta=2
python main.py --agent=agents/fav.py --env_name=humanoidmaze-medium-navigate-singletask-task2-v0 --offline_steps=1000000 --agent.discount=0.995 --agent.bandwidth_method=scott --agent.beta=2
python main.py --agent=agents/fav.py --env_name=humanoidmaze-medium-navigate-singletask-task3-v0 --offline_steps=1000000 --agent.discount=0.995 --agent.bandwidth_method=scott --agent.beta=2
python main.py --agent=agents/fav.py --env_name=humanoidmaze-medium-navigate-singletask-task4-v0 --offline_steps=1000000 --agent.discount=0.995 --agent.bandwidth_method=scott --agent.beta=2
python main.py --agent=agents/fav.py --env_name=humanoidmaze-medium-navigate-singletask-task5-v0 --offline_steps=1000000 --agent.discount=0.995 --agent.bandwidth_method=scott --agent.beta=2

# humanoidmaze-large-navigate (β=2)
python main.py --agent=agents/fav.py --env_name=humanoidmaze-large-navigate-singletask-task1-v0 --offline_steps=1000000 --agent.discount=0.995 --agent.bandwidth_method=scott --agent.beta=2
python main.py --agent=agents/fav.py --env_name=humanoidmaze-large-navigate-singletask-task2-v0 --offline_steps=1000000 --agent.discount=0.995 --agent.bandwidth_method=scott --agent.beta=2
python main.py --agent=agents/fav.py --env_name=humanoidmaze-large-navigate-singletask-task3-v0 --offline_steps=1000000 --agent.discount=0.995 --agent.bandwidth_method=scott --agent.beta=2
python main.py --agent=agents/fav.py --env_name=humanoidmaze-large-navigate-singletask-task4-v0 --offline_steps=1000000 --agent.discount=0.995 --agent.bandwidth_method=scott --agent.beta=2
python main.py --agent=agents/fav.py --env_name=humanoidmaze-large-navigate-singletask-task5-v0 --offline_steps=1000000 --agent.discount=0.995 --agent.bandwidth_method=scott --agent.beta=2

# antsoccer-arena-navigate (β=2)
python main.py --agent=agents/fav.py --env_name=antsoccer-arena-navigate-singletask-task1-v0 --offline_steps=1000000 --agent.discount=0.995 --agent.bandwidth_method=scott --agent.beta=2
python main.py --agent=agents/fav.py --env_name=antsoccer-arena-navigate-singletask-task2-v0 --offline_steps=1000000 --agent.discount=0.995 --agent.bandwidth_method=scott --agent.beta=2
python main.py --agent=agents/fav.py --env_name=antsoccer-arena-navigate-singletask-task3-v0 --offline_steps=1000000 --agent.discount=0.995 --agent.bandwidth_method=scott --agent.beta=2
python main.py --agent=agents/fav.py --env_name=antsoccer-arena-navigate-singletask-task4-v0 --offline_steps=1000000 --agent.discount=0.995 --agent.bandwidth_method=scott --agent.beta=2
python main.py --agent=agents/fav.py --env_name=antsoccer-arena-navigate-singletask-task5-v0 --offline_steps=1000000 --agent.discount=0.995 --agent.bandwidth_method=scott --agent.beta=2

# cube-single-play (β=0.5)
python main.py --agent=agents/fav.py --env_name=cube-single-play-singletask-task1-v0 --offline_steps=1000000 --agent.bandwidth_method=scott --agent.beta=0.5
python main.py --agent=agents/fav.py --env_name=cube-single-play-singletask-task2-v0 --offline_steps=1000000 --agent.bandwidth_method=scott --agent.beta=0.5
python main.py --agent=agents/fav.py --env_name=cube-single-play-singletask-task3-v0 --offline_steps=1000000 --agent.bandwidth_method=scott --agent.beta=0.5
python main.py --agent=agents/fav.py --env_name=cube-single-play-singletask-task4-v0 --offline_steps=1000000 --agent.bandwidth_method=scott --agent.beta=0.5
python main.py --agent=agents/fav.py --env_name=cube-single-play-singletask-task5-v0 --offline_steps=1000000 --agent.bandwidth_method=scott --agent.beta=0.5

# cube-double-play (β=0.5)
python main.py --agent=agents/fav.py --env_name=cube-double-play-singletask-task1-v0 --offline_steps=1000000 --agent.bandwidth_method=scott --agent.beta=0.5
python main.py --agent=agents/fav.py --env_name=cube-double-play-singletask-task2-v0 --offline_steps=1000000 --agent.bandwidth_method=scott --agent.beta=0.5
python main.py --agent=agents/fav.py --env_name=cube-double-play-singletask-task3-v0 --offline_steps=1000000 --agent.bandwidth_method=scott --agent.beta=0.5
python main.py --agent=agents/fav.py --env_name=cube-double-play-singletask-task4-v0 --offline_steps=1000000 --agent.bandwidth_method=scott --agent.beta=0.5
python main.py --agent=agents/fav.py --env_name=cube-double-play-singletask-task5-v0 --offline_steps=1000000 --agent.bandwidth_method=scott --agent.beta=0.5

# scene-play (β=0.5)
python main.py --agent=agents/fav.py --env_name=scene-play-singletask-task1-v0 --offline_steps=1000000 --agent.bandwidth_method=scott --agent.beta=0.5
python main.py --agent=agents/fav.py --env_name=scene-play-singletask-task2-v0 --offline_steps=1000000 --agent.bandwidth_method=scott --agent.beta=0.5
python main.py --agent=agents/fav.py --env_name=scene-play-singletask-task3-v0 --offline_steps=1000000 --agent.bandwidth_method=scott --agent.beta=0.5
python main.py --agent=agents/fav.py --env_name=scene-play-singletask-task4-v0 --offline_steps=1000000 --agent.bandwidth_method=scott --agent.beta=0.5
python main.py --agent=agents/fav.py --env_name=scene-play-singletask-task5-v0 --offline_steps=1000000 --agent.bandwidth_method=scott --agent.beta=0.5

# puzzle-3x3-play (β=2)
python main.py --agent=agents/fav.py --env_name=puzzle-3x3-play-singletask-task1-v0 --offline_steps=1000000 --agent.bandwidth_method=scott --agent.beta=2
python main.py --agent=agents/fav.py --env_name=puzzle-3x3-play-singletask-task2-v0 --offline_steps=1000000 --agent.bandwidth_method=scott --agent.beta=2
python main.py --agent=agents/fav.py --env_name=puzzle-3x3-play-singletask-task3-v0 --offline_steps=1000000 --agent.bandwidth_method=scott --agent.beta=2
python main.py --agent=agents/fav.py --env_name=puzzle-3x3-play-singletask-task4-v0 --offline_steps=1000000 --agent.bandwidth_method=scott --agent.beta=2
python main.py --agent=agents/fav.py --env_name=puzzle-3x3-play-singletask-task5-v0 --offline_steps=1000000 --agent.bandwidth_method=scott --agent.beta=2

# puzzle-4x4-play (β=0.1)
python main.py --agent=agents/fav.py --env_name=puzzle-4x4-play-singletask-task1-v0 --offline_steps=1000000 --agent.bandwidth_method=scott --agent.beta=0.1
python main.py --agent=agents/fav.py --env_name=puzzle-4x4-play-singletask-task2-v0 --offline_steps=1000000 --agent.bandwidth_method=scott --agent.beta=0.1
python main.py --agent=agents/fav.py --env_name=puzzle-4x4-play-singletask-task3-v0 --offline_steps=1000000 --agent.bandwidth_method=scott --agent.beta=0.1
python main.py --agent=agents/fav.py --env_name=puzzle-4x4-play-singletask-task4-v0 --offline_steps=1000000 --agent.bandwidth_method=scott --agent.beta=0.1
python main.py --agent=agents/fav.py --env_name=puzzle-4x4-play-singletask-task5-v0 --offline_steps=1000000 --agent.bandwidth_method=scott --agent.beta=0.1

# D4RL AntMaze (6 tasks)
python main.py --agent=agents/fav.py --env_name=antmaze-umaze-v2          --offline_steps=1000000 --agent.bandwidth_method=scott --agent.beta=3
python main.py --agent=agents/fav.py --env_name=antmaze-umaze-diverse-v2  --offline_steps=1000000 --agent.bandwidth_method=scott --agent.beta=3
python main.py --agent=agents/fav.py --env_name=antmaze-medium-play-v2    --offline_steps=1000000 --agent.bandwidth_method=scott --agent.beta=2
python main.py --agent=agents/fav.py --env_name=antmaze-medium-diverse-v2 --offline_steps=1000000 --agent.bandwidth_method=scott --agent.beta=3
python main.py --agent=agents/fav.py --env_name=antmaze-large-play-v2     --offline_steps=1000000 --agent.bandwidth_method=scott --agent.beta=5
python main.py --agent=agents/fav.py --env_name=antmaze-large-diverse-v2  --offline_steps=1000000 --agent.bandwidth_method=scott --agent.beta=5
```

## Reproducing the offline-to-online benchmark (30 tasks)

The off2on benchmark is the 6-environment subset of the offline benchmark
(`antmaze-giant-navigate`, `humanoidmaze-large-navigate`,
`antsoccer-arena-navigate`, `cube-double-play`, `scene-play`,
`puzzle-4x4-play`). Each command runs 1 M offline steps followed by 1 M online
steps. To reproduce in offline-only mode, simply drop `--online_steps=1000000`.

### FAV (fixed bandwidth) — 30 commands

```bash
# antmaze-giant-navigate (σ=0.5, β=1)
python main.py --agent=agents/fav.py --env_name=antmaze-giant-navigate-singletask-task1-v0 --offline_steps=1000000 --online_steps=1000000 --agent.discount=0.995 --agent.q_agg=min --agent.bandwidth_method=fixed --agent.bandwidth=0.5 --agent.beta=1
python main.py --agent=agents/fav.py --env_name=antmaze-giant-navigate-singletask-task2-v0 --offline_steps=1000000 --online_steps=1000000 --agent.discount=0.995 --agent.q_agg=min --agent.bandwidth_method=fixed --agent.bandwidth=0.5 --agent.beta=1
python main.py --agent=agents/fav.py --env_name=antmaze-giant-navigate-singletask-task3-v0 --offline_steps=1000000 --online_steps=1000000 --agent.discount=0.995 --agent.q_agg=min --agent.bandwidth_method=fixed --agent.bandwidth=0.5 --agent.beta=1
python main.py --agent=agents/fav.py --env_name=antmaze-giant-navigate-singletask-task4-v0 --offline_steps=1000000 --online_steps=1000000 --agent.discount=0.995 --agent.q_agg=min --agent.bandwidth_method=fixed --agent.bandwidth=0.5 --agent.beta=1
python main.py --agent=agents/fav.py --env_name=antmaze-giant-navigate-singletask-task5-v0 --offline_steps=1000000 --online_steps=1000000 --agent.discount=0.995 --agent.q_agg=min --agent.bandwidth_method=fixed --agent.bandwidth=0.5 --agent.beta=1

# humanoidmaze-large-navigate (σ=0.5, β=1)
python main.py --agent=agents/fav.py --env_name=humanoidmaze-large-navigate-singletask-task1-v0 --offline_steps=1000000 --online_steps=1000000 --agent.discount=0.995 --agent.bandwidth_method=fixed --agent.bandwidth=0.5 --agent.beta=1
python main.py --agent=agents/fav.py --env_name=humanoidmaze-large-navigate-singletask-task2-v0 --offline_steps=1000000 --online_steps=1000000 --agent.discount=0.995 --agent.bandwidth_method=fixed --agent.bandwidth=0.5 --agent.beta=1
python main.py --agent=agents/fav.py --env_name=humanoidmaze-large-navigate-singletask-task3-v0 --offline_steps=1000000 --online_steps=1000000 --agent.discount=0.995 --agent.bandwidth_method=fixed --agent.bandwidth=0.5 --agent.beta=1
python main.py --agent=agents/fav.py --env_name=humanoidmaze-large-navigate-singletask-task4-v0 --offline_steps=1000000 --online_steps=1000000 --agent.discount=0.995 --agent.bandwidth_method=fixed --agent.bandwidth=0.5 --agent.beta=1
python main.py --agent=agents/fav.py --env_name=humanoidmaze-large-navigate-singletask-task5-v0 --offline_steps=1000000 --online_steps=1000000 --agent.discount=0.995 --agent.bandwidth_method=fixed --agent.bandwidth=0.5 --agent.beta=1

# antsoccer-arena-navigate (σ=1, β=1)
python main.py --agent=agents/fav.py --env_name=antsoccer-arena-navigate-singletask-task1-v0 --offline_steps=1000000 --online_steps=1000000 --agent.discount=0.995 --agent.bandwidth_method=fixed --agent.bandwidth=1 --agent.beta=1
python main.py --agent=agents/fav.py --env_name=antsoccer-arena-navigate-singletask-task2-v0 --offline_steps=1000000 --online_steps=1000000 --agent.discount=0.995 --agent.bandwidth_method=fixed --agent.bandwidth=1 --agent.beta=1
python main.py --agent=agents/fav.py --env_name=antsoccer-arena-navigate-singletask-task3-v0 --offline_steps=1000000 --online_steps=1000000 --agent.discount=0.995 --agent.bandwidth_method=fixed --agent.bandwidth=1 --agent.beta=1
python main.py --agent=agents/fav.py --env_name=antsoccer-arena-navigate-singletask-task4-v0 --offline_steps=1000000 --online_steps=1000000 --agent.discount=0.995 --agent.bandwidth_method=fixed --agent.bandwidth=1 --agent.beta=1
python main.py --agent=agents/fav.py --env_name=antsoccer-arena-navigate-singletask-task5-v0 --offline_steps=1000000 --online_steps=1000000 --agent.discount=0.995 --agent.bandwidth_method=fixed --agent.bandwidth=1 --agent.beta=1

# cube-double-play (σ=0.1, β=0.5)
python main.py --agent=agents/fav.py --env_name=cube-double-play-singletask-task1-v0 --offline_steps=1000000 --online_steps=1000000 --agent.bandwidth_method=fixed --agent.bandwidth=0.1 --agent.beta=0.5
python main.py --agent=agents/fav.py --env_name=cube-double-play-singletask-task2-v0 --offline_steps=1000000 --online_steps=1000000 --agent.bandwidth_method=fixed --agent.bandwidth=0.1 --agent.beta=0.5
python main.py --agent=agents/fav.py --env_name=cube-double-play-singletask-task3-v0 --offline_steps=1000000 --online_steps=1000000 --agent.bandwidth_method=fixed --agent.bandwidth=0.1 --agent.beta=0.5
python main.py --agent=agents/fav.py --env_name=cube-double-play-singletask-task4-v0 --offline_steps=1000000 --online_steps=1000000 --agent.bandwidth_method=fixed --agent.bandwidth=0.1 --agent.beta=0.5
python main.py --agent=agents/fav.py --env_name=cube-double-play-singletask-task5-v0 --offline_steps=1000000 --online_steps=1000000 --agent.bandwidth_method=fixed --agent.bandwidth=0.1 --agent.beta=0.5

# scene-play (σ=0.1, β=0.5)
python main.py --agent=agents/fav.py --env_name=scene-play-singletask-task1-v0 --offline_steps=1000000 --online_steps=1000000 --agent.bandwidth_method=fixed --agent.bandwidth=0.1 --agent.beta=0.5
python main.py --agent=agents/fav.py --env_name=scene-play-singletask-task2-v0 --offline_steps=1000000 --online_steps=1000000 --agent.bandwidth_method=fixed --agent.bandwidth=0.1 --agent.beta=0.5
python main.py --agent=agents/fav.py --env_name=scene-play-singletask-task3-v0 --offline_steps=1000000 --online_steps=1000000 --agent.bandwidth_method=fixed --agent.bandwidth=0.1 --agent.beta=0.5
python main.py --agent=agents/fav.py --env_name=scene-play-singletask-task4-v0 --offline_steps=1000000 --online_steps=1000000 --agent.bandwidth_method=fixed --agent.bandwidth=0.1 --agent.beta=0.5
python main.py --agent=agents/fav.py --env_name=scene-play-singletask-task5-v0 --offline_steps=1000000 --online_steps=1000000 --agent.bandwidth_method=fixed --agent.bandwidth=0.1 --agent.beta=0.5

# puzzle-4x4-play (σ=1, β=0.5)
python main.py --agent=agents/fav.py --env_name=puzzle-4x4-play-singletask-task1-v0 --offline_steps=1000000 --online_steps=1000000 --agent.bandwidth_method=fixed --agent.bandwidth=1 --agent.beta=0.5
python main.py --agent=agents/fav.py --env_name=puzzle-4x4-play-singletask-task2-v0 --offline_steps=1000000 --online_steps=1000000 --agent.bandwidth_method=fixed --agent.bandwidth=1 --agent.beta=0.5
python main.py --agent=agents/fav.py --env_name=puzzle-4x4-play-singletask-task3-v0 --offline_steps=1000000 --online_steps=1000000 --agent.bandwidth_method=fixed --agent.bandwidth=1 --agent.beta=0.5
python main.py --agent=agents/fav.py --env_name=puzzle-4x4-play-singletask-task4-v0 --offline_steps=1000000 --online_steps=1000000 --agent.bandwidth_method=fixed --agent.bandwidth=1 --agent.beta=0.5
python main.py --agent=agents/fav.py --env_name=puzzle-4x4-play-singletask-task5-v0 --offline_steps=1000000 --online_steps=1000000 --agent.bandwidth_method=fixed --agent.bandwidth=1 --agent.beta=0.5
```

## Acknowledgments

This codebase is built on top of [OGBench](https://github.com/seohongpark/ogbench)
and the [Flow Q-Learning](https://github.com/seohongpark/fql) reference
implementation.
