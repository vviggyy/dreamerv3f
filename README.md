# DreamerV3f — Spatial Representations in World Models

Fork of [DreamerV3][paper] for studying spatial representations learned by world model agents in Crafter. Includes tools for trajectory recording, position decoding, layer-wise analysis, spatial tuning curves, and dream decoding.

Based on [Hafner et al. 2025][paper], "Mastering Diverse Domains through World Models."

## Setup

Requires Python 3.11+. Tested on Linux (SLURM clusters with A100/H100 GPUs).

```sh
pip install -U -r requirements.txt
```

For CPU-only machines (e.g. local Mac):
```sh
pip install jax==0.4.35 jaxlib==0.4.35 chex==0.1.87 optax==0.2.3
```

> **JAX platform:** Default is `cuda`. On CPU-only machines, always pass `--jax.platform cpu` or JAX will fail to initialize.

## Pipeline Overview

The typical workflow is: **train** → **eval trajectory** → **analysis** (decoding, tuning curves, plotting).

All analysis scripts log provenance to `run_info.json` in their output directory via `dreamerv3/run_info.py`. Each entry records the timestamp, git SHA, full command line, SLURM job ID, and all arguments. Training runs also save `hyperparams.txt` and `config.yaml` to the logdir.

### Train

```sh
python dreamerv3/main.py \
  --configs crafter_small size25m \
  --logdir ./logdir/my_run \
  --env.crafter.random_spawn True \
  --env.crafter.fixed_seed False
```

All config options are in `dreamerv3/configs.yaml`. Use `--configs` to stack presets (applied left to right), then override individual flags. Key Crafter env flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--env.crafter.fixed_seed` | `True` | Same world every episode (resets crafter's internal episode counter) |
| `--env.crafter.random_spawn` | `False` | Relocate player to random walkable tile each episode |
| `--env.crafter.egocentric_view` | `0` | N×N ego-centered view (odd int, e.g. 7); 0 = disabled |
| `--env.crafter.disable_mobs` | `False` | Remove hostile mobs (zombies, skeletons) |
| `--env.crafter.area` | `[64,64]` | World size in tiles (`crafter_small` preset uses `[32,32]`) |
| `--seed` | `0` | Global seed. Env seed derived as `hash((seed, env_index))` |

**Seed mechanics:** Crafter generates worlds via `hash((env_seed, episode_number))`. With `fixed_seed=True`, the episode counter is reset to 0 before each `reset()`, so every episode gets the same world. With `fixed_seed=False`, episode numbers increment naturally (1, 2, 3, ...) giving a different world each episode. Different `--seed` values produce different environments even with `fixed_seed=True`.

Size presets: `size1m`, `size12m`, `size25m`, `size50m`, `size100m`, `size200m`, `size400m`.

### Eval Trajectory

Records agent positions, world model activations (all layers), images, and achievements per timestep. Saves per-episode `.pkl` files plus `all_episodes.pkl` with metadata.

```sh
python dreamerv3/main.py \
  --configs crafter_small size25m \
  --logdir ./logdir/my_run \
  --script eval_trajectory \
  --run.from_checkpoint ./logdir/my_run/ckpt/TIMESTAMP_DIR \
  --eval_trajectory.num_episodes 100 \
  --eval_trajectory.save_path ./logdir/my_run/trajectories \
  --seed 42 --jax.platform cpu
```

Checkpoint is auto-detected from `logdir/ckpt/latest` if `--run.from_checkpoint` is omitted. Env settings (egocentric_view, random_spawn, etc.) are inherited from the training run's saved `config.yaml`.

> **GPU eval:** Drop `--jax.platform cpu` on GPU nodes for 50-100x speedup (important for 400M param models).

### Plot Trajectories

```sh
MPLBACKEND=Agg python dreamerv3/plot_trajectories.py \
  --data ./logdir/my_run/trajectories \
  --plot all \
  --save ./logdir/my_run/plots
```

Plot types: `trajectories`, `heatmap`, `activation`, `spatial`, `world`, `fullworld`, `animate`, `animate_world`, `worldview`, `world_only`, `all`.

Animation/worldview options: `--egocentric_view N`, `--view_half N`, `--window_tiles N`, `--step_ms N`, `--mp4` (MP4 instead of GIF, requires ffmpeg).

### Decode Position (Standard)

Trains pRNN-style linear classification decoders to predict (x,y) from latent representations using cross-validation.

```sh
MPLBACKEND=Agg python dreamerv3/decode_position.py \
  --data ./logdir/my_run/trajectories \
  --save ./logdir/my_run/decoder_results \
  --n_jobs -1
```

Key options:
- `--repr {deter,stoch,combined,all}` — which representation to decode
- `--save_model` — save fitted decoders (needed for dream decode)
- `--n_jobs N` — parallel workers (`-1` = all CPUs)
- `--device {cpu,cuda}` — torch device for classification
- `--n_iters N` — classification training iterations (default 5000)
- `--patience N` — early stopping patience (default 500, 0 = disable)
- `--min_bbox N` — filter episodes with bounding-box area < N tiles² (0 = no filter)

Outputs: `classification_*.png`, `probmap_*.png`, `occupancy_vs_error_*.png`, `decode_results.pkl`.

### Layer-wise Decoding

Scans every recorded layer (encoder, dynamics, policy, value) and produces a per-layer comparison boxplot.

```sh
# Holdout mode (fastest — train on one set, eval on another):
MPLBACKEND=Agg python dreamerv3/decode_position.py \
  --data ./trajectories_train --test_data ./trajectories_test \
  --save ./decoder_results \
  --mode layers --device cuda --n_jobs -1

# CV mode (single trajectory set, 5-fold KFold):
MPLBACKEND=Agg python dreamerv3/decode_position.py \
  --data ./trajectories --save ./decoder_results \
  --mode layers --device cuda --n_jobs -1 --holdout_frac 0

# Save trained decoders for reuse:
MPLBACKEND=Agg python dreamerv3/decode_position.py \
  --data ./trajectories --save ./decoder_results \
  --mode layers --device cuda --n_jobs -1 --save_model

# Eval saved decoders on new data (no training):
MPLBACKEND=Agg python dreamerv3/decode_position.py \
  --data ./new_trajectories --save ./new_results \
  --mode layers --from_model ./decoder_results/layer_decoders
```

Layer-specific options:
- `--max_samples N` — subsample timesteps (default 10000)
- `--n_cv_folds N` — KFold folds (default 5)
- `--holdout_frac F` — auto train/test split fraction (default 0.2, use 0 for CV mode)
- `--test_data PATH` — separate held-out trajectory dir (overrides holdout_frac)
- `--resume PATH` — resume from partial checkpoint
- `--from_model PATH` — eval-only from saved `layer_decoders/` dir

Outputs: `layer_comparison.png`, `layer_decode_results.pkl`, `layer_decode_checkpoint.pkl`.

### Dream Decode

Applies a pretrained classification position decoder to imagined (dream) rollouts to test spatial coherence.

```sh
# Step 1: Save decoder model (if not done already)
MPLBACKEND=Agg python dreamerv3/decode_position.py \
  --data ./logdir/my_run/trajectories \
  --save ./logdir/my_run/decoder_results \
  --save_model

# Step 2: Run dream decode
python dreamerv3/main.py \
  --configs crafter_small size25m \
  --logdir ./logdir/my_run \
  --script dream_decode \
  --run.from_checkpoint ./logdir/my_run/ckpt/TIMESTAMP_DIR \
  --dream_decode.decoder_model ./logdir/my_run/decoder_results/classifier_deter.pkl \
  --dream_decode.save_path ./logdir/my_run/dream_results
```

Options: `--dream_decode.num_batches N`, `--dream_decode.num_episodes N`.

Outputs: `dream_trajectories_world.png`, `dream_probmap_*.png`, `dream_vs_real.png`, `dream_results.pkl`.

### Plot Training Progress

```sh
MPLBACKEND=Agg python dreamerv3/plot_training.py \
  --logdir ./logdir/my_run \
  --save ./logdir/my_run/plots \
  --smooth 50
```

Reads `scores.jsonl` and `metrics.jsonl` from logdir. Produces `training_progress.png` with panels for episode score, cumulative reward, Crafter score (geometric mean of achievement success rates), and per-achievement unlock rates.

Options: `--no_achievements` (skip per-achievement panel), `--no_losses` (skip loss/reward/value panels).

### Tuning Curve Analysis

Classifies neurons into spatial cell types (place, border, HD, etc.) using pynapple. Analyzes all recorded layers.

```sh
MPLBACKEND=Agg python dreamerv3/tuning_curve.py \
  --data ./logdir/my_run/trajectories \
  --save ./logdir/my_run/tuning_results \
  --n_jobs -1
```

Options:
- `--layers LAYER [...]` — filter to specific layers (default: all)
- `--test_data PATH` — held-out trajectories for EV reliability
- `--max_neurons N` — subsample large layers (0 = all)
- `--no_hd` — skip head direction analysis
- `--no_plots` — skip plot generation
- `--interactive` — show interactive SI vs EV scatter
- `--min_bbox N` — filter episodes by bounding-box area
- `--SI_thresh`, `--EV_thresh`, `--EV_unthresh`, `--HD_thresh` — classification thresholds

Outputs: `tuning_results.pkl`, `{layer}_si_ev_scatter.png`, `{layer}_cell_types.png`, `{layer}_example_tuning_curves.png`, `layer_summary.png`.

**Interactive viewer** (from precomputed results):
```sh
python dreamerv3/tuning_curve.py --from_pkl tuning_results.pkl
python dreamerv3/tuning_curve.py --from_pkl tuning_results.pkl --layers dyn/deter
```

## SLURM Scripts

Shell scripts for cluster submission are in the repo root:

| Script | Purpose | GPU | Key settings |
|--------|---------|-----|-------------|
| `run_Crafter.sh` | Train model | H100 | Parametrized env/size/activation/wd |
| `run_Trajectory.sh` | Eval trajectories | CPU | Matches training hyperparams |
| `run_Loop.sh` | Full pipeline (train → eval → all analysis) | A100 | End-to-end |
| `run_Decoding.sh` | Standard position decoding | GPU | method, repr, device |
| `run_LayerDecoding.sh` | Layer-wise decoding | A100 | mode, ridge/classification, holdout |
| `run_Plotting.sh` | Plot trajectories + training | CPU | plot type, animation settings |
| `run_Tuning.sh` | Tuning curve analysis | A100 | layers, thresholds, n_jobs |

Training scripts save `hyperparams.txt` to the logdir with all SLURM-level settings.

## Run Provenance

Every analysis script (`decode_position.py`, `plot_trajectories.py`, `plot_training.py`, `tuning_curve.py`) appends a JSON entry to `<save_dir>/run_info.json` via `dreamerv3/run_info.py`. Each entry contains:
- `stage` — pipeline stage name
- `timestamp` — ISO timestamp
- `git_sha` — short git SHA of HEAD
- `command` — full command line
- `slurm_job_id` — SLURM job ID (if running under SLURM)
- `args` — all parsed arguments
- `outputs` — list of output files produced
- `extra` — additional metadata (varies by stage)

Training runs additionally save:
- `config.yaml` — full resolved DreamerV3 config (all hyperparams)
- `hyperparams.txt` — SLURM script-level settings (when using `run_Crafter.sh` or `run_Loop.sh`)

## Tips

- All config options are in `dreamerv3/configs.yaml`. Override as CLI flags.
- Stack config presets: `--configs crafter_small size50m` (applied left to right).
- The `debug` preset reduces network size and batch size for fast iteration.
- Use `MPLBACKEND=Agg` for headless plotting on cluster nodes.
- Checkpoint path: pass the timestamped **directory** (e.g. `ckpt/20260129T183613F519148`), not the `latest` file.
- If you get `Too many leaves for PyTreeDef`, the checkpoint doesn't match the current config.
- CUDA errors: scroll up — often caused by OOM or JAX/CUDA version mismatch. Try `--batch_size 1`.
- To resume training, rerun the same command with the same `--logdir`.

## Tests

```sh
PYTHONPATH=. python embodied/tests/test_crafter_world.py
```

Tests fixed_seed determinism, walkable spawn validation, and world consistency.

## Citation

```
@article{hafner2025dreamerv3,
  title={Mastering diverse control tasks through world models},
  author={Hafner, Danijar and Pasukonis, Jurgis and Ba, Jimmy and Lillicrap, Timothy},
  journal={Nature},
  pages={1--7},
  year={2025},
  publisher={Nature Publishing Group}
}
```

[paper]: https://arxiv.org/pdf/2301.04104
[website]: https://danijar.com/dreamerv3
