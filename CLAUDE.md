# dreamerv3f repo guide

## arch
- `dreamerv3/main.py` — entry point, dispatches on `--script {train,train_eval,eval_only,eval_trajectory,dream_decode,parallel,parallel_env,parallel_envs,parallel_replay}`
- `dreamerv3/agent.py` — JAX agent (DreamerV3 world model)
- `dreamerv3/configs.yaml` — all configs. presets: `defaults`, `crafter`, `crafter_small`, `size{1m,12m,25m,50m,100m,200m,400m}`, `debug`, `atari`, `atari100k`, `procgen`, `minecraft`, `dmlab`, `dmc_proprio`, `dmc_vision`, `bsuite`, `loconav`, `multicpu`
- `dreamerv3/eval_trajectory.py` — records pos/activations/images per step, saves pkl
- `dreamerv3/plot_trajectories.py` — plots: trajectories, heatmap, activation, world overlay, fullworld, animations
- `dreamerv3/decode_position.py` — linear decoders (Ridge + classification) to predict (x,y) from deter/stoch. Includes `plot_probmap_on_world` which overlays decoder P(pos) heatmap on rendered Crafter world map with trajectory. `plot_occupancy_vs_error` produces a two-panel figure: faint world map + occupancy hotspots (left), per-timestep Manhattan error vs tile visit count scatter (right). `--save_model` saves fitted decoders for use by `dream_decode.py`. Layer-wise decoding (`--mode layers`): Ridge uses R², classification uses mean Manhattan distance (tiles).
- `dreamerv3/dream_decode.py` — applies pretrained position decoder to policy-based imagination (dream) rollouts. Tests spatial coherence of dreamed trajectories
- `dreamerv3/tuning_curve.py` — spatial tuning curve analysis: classifies neurons into cell types (place, border, HD, etc.) using pynapple. Computes per-neuron spatial info, EV reliability, autocorrelation metrics, and HD mutual info across all recorded layers
- `embodied/envs/crafter.py` — Crafter wrapper. `fixed_seed=True` resets `_episode=0` before each reset so same world. `random_spawn=True` relocates player to random walkable tile each episode. `egocentric_view=N` (odd int, e.g. 7) renders N×N egocentric view centered on player facing direction; inventory bar is copied from the standard render into the bottom rows. Also exposes: `log/player_facing_x`, `log/player_facing_y` (facing direction as ±1/0 ints), `log/achievement_*` (per-achievement binary), `log/reward` (raw crafter reward). Writes `stats.jsonl` to logdir if configured.
- `embodied/jax/agent.py` — JAX Agent.__new__ calls internal.setup() then __init__. Line 72: jax.devices()
- `embodied/jax/internal.py` — setup() sets jax platform, XLA flags. Line 34: `platform and jax.config.update('jax_platforms', platform)`
- `embodied/tests/test_crafter_world.py` — world consistency tests (fixed_seed, walkable spawn, determinism)

## critical gotchas
- **jax.platform**: default is `cuda` (configs.yaml L78). On mac/cpu: `--jax.platform cpu` or get AssertionError in jax backends
- **checkpoint path**: pass the timestamped DIR not the `latest` file. e.g. `./logdir/.../ckpt/20260129T183613F519148`
- **crafter area**: `crafter_small` uses `area=[32,32]` not default 64x64. plot_trajectories.py reads area from metadata
- **world seed**: crafter derives seed as `hash((self._seed, self._episode))`. fixed_seed resets _episode=0 so same world each ep
- **egocentric_view**: must be an odd integer (e.g. 7). `egocentric_view=0` disables it (default). Set in configs.yaml under `env.crafter.egocentric_view` or pass `--env.crafter.egocentric_view 7`
- **layer decode checkpoint**: checkpoint files record which metric was used (`r2`, `manhattan`, `ce_loss`). Mismatched metric triggers a warning and checkpoint is ignored — delete stale checkpoint if switching modes
- **layer decode metric**: `--ridge_layers` → R²; classification (no flag) → mean Manhattan distance in tiles (joint x,y, lower = better)
- **layer decode holdout default**: `--holdout_frac` defaults to 0.2 (auto 80/20 episode split, no CV). Use `--holdout_frac 0` for CV mode. Classification uses Manhattan-based early stopping on a validation split (not CE loss)

## commands

### train
```
python dreamerv3/main.py --configs crafter_small size1m --logdir ./logdir/crafter_small_1m --jax.platform cpu
```

### eval trajectory
```
python dreamerv3/main.py \
  --configs crafter_small size1m \
  --logdir ./logdir/crafter_small_1m \
  --script eval_trajectory \
  --run.from_checkpoint ./logdir/crafter_small_1m/ckpt/TIMESTAMP_DIR \
  --eval_trajectory.num_episodes 5 \
  --eval_trajectory.save_path ./logdir/crafter_small_1m/trajectories \
  --seed 42 --jax.platform cpu
```
Saves: per-episode pkl + all_episodes.pkl with metadata {env_seed, world_seed, fixed_seed, task, area}

### plot
```
MPLBACKEND=Agg python dreamerv3/plot_trajectories.py \
  --data ./logdir/crafter_small_1m/trajectories --plot all --save ./logdir/crafter_small_1m/plots
```
Drop MPLBACKEND=Agg if running with display. Plot types: trajectories, heatmap, activation, spatial, world, fullworld, animate, animate_world, worldview, all. Extra args for worldview/animate: `--egocentric_view N`, `--view_half N`, `--window_tiles N`, `--step_ms N`, `--mp4` (save as MP4 instead of GIF).

### decode position
```
MPLBACKEND=Agg python dreamerv3/decode_position.py \
  --data ./logdir/crafter_small_1m/trajectories \
  --save ./logdir/crafter_small_1m/decoder_results \
  --method both
```
Methods: regression (Ridge), classification (pRNN-style cross-entropy), both. Add `--repr {deter,stoch,combined,all}` to select representation. Add `--no_per_neuron` to skip per-neuron R² analysis. Add `--save_model` to save fitted decoders for dream_decode. Add `--n_jobs N` for parallel jobs (`-1` = all CPUs). Add `--device cuda` for GPU classification (multi-GPU round-robin with `n_jobs>1`). Add `--n_iters N` for classification training iterations (default 5000). Add `--resume PATH` for layer mode to resume from partial checkpoint. Auto-generates `occupancy_vs_error_{repr}.png` (world-overlay occupancy heatmap + per-timestep Manhattan error vs visit count scatter).

### layer-wise decoding

**Fastest — holdout mode (no CV), requires two trajectory sets:**
```
MPLBACKEND=Agg python dreamerv3/decode_position.py \
  --data ./trajectories_train --test_data ./trajectories_test \
  --save ./decoder_results \
  --mode layers --ridge_layers --n_jobs -1
```
Trains Ridge on `--data`, evaluates on `--test_data`. No folds at all. Completes in seconds.

**Save trained decoders for reuse:**
```
MPLBACKEND=Agg python dreamerv3/decode_position.py \
  --data ./trajectories --save ./decoder_results \
  --mode layers --ridge_layers --n_jobs -1 --save_model
```
After decoding, retrains each layer on full data and saves to `<save>/layer_decoders/` (per-layer `.pkl` files + `manifest.pkl`).

**Eval saved decoders on new trajectories (no training):**
```
MPLBACKEND=Agg python dreamerv3/decode_position.py \
  --data ./new_trajectories --save ./new_results \
  --mode layers --from_model ./decoder_results/layer_decoders
```
Loads pretrained decoders, evaluates on `--data`, produces `layer_comparison.png` and `layer_decode_results.pkl`.

**CV mode (one trajectory set):**
```
MPLBACKEND=Agg python dreamerv3/decode_position.py \
  --data ./trajectories --save ./decoder_results \
  --mode layers --ridge_layers --n_jobs -1
```
Uses 5-fold KFold (NOT LOGO — LOGO creates O(N_eps × N_layers) jobs, crushingly slow with randspawn).
Key args: `--max_samples 10000`, `--max_dims 256` (truncated PCA, critical for 4096-dim deter), `--n_cv_folds 5`.

### eval trajectory on GPU (fast for 400M param models)
Drop `--jax.platform cpu` on GPU nodes — JAX uses CUDA by default, 50-100x faster.
400M = 400 million parameters (weights) total across the full model.

### dream decode
```
# Step 1: Save decoder model
MPLBACKEND=Agg python dreamerv3/decode_position.py \
  --data ./logdir/crafter_small_1m/trajectories \
  --save ./logdir/crafter_small_1m/decoder_results \
  --method both --save_model

# Step 2: Run dream decode
python dreamerv3/main.py \
  --configs crafter_small size1m \
  --logdir ./logdir/crafter_small_1m \
  --script dream_decode \
  --run.from_checkpoint ./logdir/crafter_small_1m/ckpt/TIMESTAMP_DIR \
  --dream_decode.decoder_model ./logdir/crafter_small_1m/decoder_results/ridge_deter.pkl \
  --dream_decode.save_path ./logdir/crafter_small_1m/dream_results \
  --jax.platform cpu
```
Outputs: dream_trajectories_world.png, dream_probmap_*.png (classifier only), dream_vs_real.png, dream_results.pkl
Additional dream_decode configs: `--dream_decode.num_batches N`, `--dream_decode.decoder_type {ridge,...}`, `--dream_decode.num_episodes N`.

### plot training progress
```
MPLBACKEND=Agg python dreamerv3/plot_training.py \
  --logdir ./logdir/crafter_small_1m \
  --save ./logdir/crafter_small_1m/plots \
  --smooth 50
```
Reads `scores.jsonl` (episode score + per-episode achievement success) and `metrics.jsonl` (per-achievement stats) from logdir. Produces `training_progress.png` with up to 4 panels: episode score, cumulative reward, Crafter score (geometric mean of achievement success rates), per-achievement unlock rate. Add `--no_achievements` to skip the per-achievement panel. Add `--no_losses` to skip loss/reward/value panels. The Crafter score panel appears automatically when `scores.jsonl` contains per-episode achievement data (requires training with the updated logger).

### tuning curve analysis
```
MPLBACKEND=Agg python dreamerv3/tuning_curve.py \
  --data ./logdir/crafter_small_1m/trajectories \
  --save ./logdir/crafter_small_1m/tuning_results \
  --n_jobs -1
```
Analyzes all recorded layers (enc/*, dyn/*, pol/*, val/*). Computes per-neuron: 2D spatial tuning curves + spatial information (pynapple), HD tuning + mutual info, EV reliability, autocorrelation peaks/field size/asymmetry. Classifies neurons into 7 types: untuned, HD_cells, single_field, border_cells, spatial_HD, complex_cells, dead. Add `--test_data` for held-out EV reliability. Add `--layers dyn/deter dyn/stoch` to filter layers. Add `--no_hd` to skip HD analysis. Add `--no_plots` to skip plots. Add `--max_neurons N` to subsample large layers (0=all). Add `--interactive` to show interactive SI vs EV scatter during analysis. Threshold overrides: `--SI_thresh`, `--EV_thresh`, `--EV_unthresh`, `--HD_thresh`.

Outputs: `tuning_results.pkl` (per-layer tuning curves, metrics, cell groups), `{layer}_si_ev_scatter.png`, `{layer}_cell_types.png`, `{layer}_example_tuning_curves.png`, `layer_summary.png`.

### interactive tuning viewer (from precomputed pkl)
```
python dreamerv3/tuning_curve.py --from_pkl ./logdir/.../tuning_results/tuning_results.pkl
python dreamerv3/tuning_curve.py --from_pkl tuning_results.pkl --layers dyn/deter
```
Loads precomputed `tuning_results.pkl` and launches an interactive matplotlib SI vs EV scatter. Click any neuron to display its tuning curve in a side panel. No `--data` or `--save` required. If multiple layers exist, prompts for layer selection (or pass `--layers` to filter).

### tests
```
PYTHONPATH=/Users/viggy/Desktop/dreamerv3f python embodied/tests/test_crafter_world.py
```

## deps not in requirements.txt that were needed
All now added: crafter, matplotlib, pandas, Pillow, pynapple, ruamel.yaml. Install with `pip install -r requirements.txt`.
jax pinned to cuda in requirements but use `pip install jax==0.4.35 jaxlib==0.4.35 chex==0.1.87 optax==0.2.3` for cpu-compatible versions.

## current state
- checkpoint: `logdir/crafter_small_1m/ckpt/20260129T183613F519148` (1M steps, crafter_small size1m)
- trajectories: 5 eps saved, fixed_seed=True, area=(32,32), spawn=(16,16), env_seed=790160138
- plots: all generated in `logdir/crafter_small_1m/plots/`
- spatial units: deter[436] y_corr=0.875, deter[245] x_corr=-0.789 (top place cells)

## agent internals (compressed reference)

### RSSM world model (`dreamerv3/rssm.py`)
- `observe(carry, tokens, action, reset)` → carry, entries, feat. Carry = {deter, stoch}
- `imagine(carry, policy_or_actions, length)` → carry, feat, action. feat = {deter, stoch, logit}
- `starts(entries, carry, nlast)` → starting states for imagination (last nlast timesteps of entries, reshaped to batch)
- `_core(deter, stoch, actemb)` → next deter via block-wise GRU (8 blocks). Deter shape: (B, 4096) default
- `_prior(deter)` → stoch logits from deter alone (used during imagination)
- `_post(deter, tokens)` → stoch logits from deter + encoder tokens (used during observation)
- Stoch: categorical, shape (B, 32, 32) = 32 categories × 32 classes

### Agent training flow (`dreamerv3/agent.py::loss()`)
1. Encode: `enc(obs)` → tokens
2. Observe: `dyn.observe(tokens, actions)` → repfeat (posterior)
3. Decode: `dec(repfeat)` → reconstruction loss
4. Heads: rew(repfeat), con(repfeat) → reward/continuation losses
5. Imagination: `starts = dyn.starts(entries, carry, K)`, `dyn.imagine(starts, policyfn, H)` → imgfeat
6. Actor-critic: rew/con/pol/val on imgfeat → policy loss, value loss

### Agent report flow (`dreamerv3/agent.py::report()`)
1. `_apply_replay_context` → carry, obs, prevact
2. `loss()` for train metrics
3. Observe first half: `dyn.observe(firsthalf(tokens), firsthalf(prevact))`
4. Open-loop imagine: `dyn.imagine(dyn_carry, secondhalf(prevact))` (uses real actions)
5. Decode both halves → openloop video comparisons in metrics
6. Policy imagination: `dyn.imagine(dyn_carry, policyfn, H)` → dream/deter, dream/stoch in metrics

### Outer agent JIT wrapping (`embodied/jax/agent.py`)
- Inner agent methods wrapped via `transform.apply(nj.pure(model.method), mesh, shardings)`
- `_train`, `_report`, `_policy` are JIT-compiled transforms
- report() output: (carry, mets) — mets dict flows through `_take_outs` → numpy
- Adding keys to mets dict in inner report() automatically flows through (recompiles JIT once)

### Key configs (`dreamerv3/configs.yaml`)
- `imag_length: 15` — imagination horizon
- `imag_last: 0` — use all T batch timesteps as starts (0 = all)
- `batch_length: 64`, `report_length: 64` — sequence lengths
- `batch_size: 16` — batch size
- `replay_context: 0` — replay context window

### Checkpoint loading pattern
```python
agent = make_agent()
cp = elements.Checkpoint()
cp.agent = agent
cp.load(args.from_checkpoint, keys=['agent'])
```

### Data flow for report()
- Data must have all obs_space + act_space + ext_space keys
- Shape: (batch_size, report_length, ...)
- Created by: `make_stream(replay, 'report')` from replay buffer
- `agent.stream(stream)` wraps with device_put + seed injection
