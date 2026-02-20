# dreamerv3f repo guide

## arch
- `dreamerv3/main.py` — entry point, dispatches on `--script {train,eval_only,eval_trajectory,dream_decode,parallel}`
- `dreamerv3/agent.py` — JAX agent (DreamerV3 world model)
- `dreamerv3/configs.yaml` — all configs. presets: `defaults`, `crafter`, `crafter_small`, `size1m`, `debug`
- `dreamerv3/eval_trajectory.py` — records pos/activations/images per step, saves pkl
- `dreamerv3/plot_trajectories.py` — plots: trajectories, heatmap, activation, world overlay, fullworld, animations
- `dreamerv3/decode_position.py` — linear decoders (Ridge + classification) to predict (x,y) from deter/stoch. Includes `plot_probmap_on_world` which overlays decoder P(pos) heatmap on rendered Crafter world map with trajectory. `--save_model` saves fitted decoders for use by `dream_decode.py`
- `dreamerv3/dream_decode.py` — applies pretrained position decoder to policy-based imagination (dream) rollouts. Tests spatial coherence of dreamed trajectories
- `embodied/envs/crafter.py` — Crafter wrapper. `fixed_seed=True` resets `_episode=0` before each reset so same world
- `embodied/jax/agent.py` — JAX Agent.__new__ calls internal.setup() then __init__. Line 72: jax.devices()
- `embodied/jax/internal.py` — setup() sets jax platform, XLA flags. Line 34: `platform and jax.config.update('jax_platforms', platform)`
- `embodied/tests/test_crafter_world.py` — world consistency tests (fixed_seed, walkable spawn, determinism)

## critical gotchas
- **jax.platform**: default is `cuda` (configs.yaml L78). On mac/cpu: `--jax.platform cpu` or get AssertionError in jax backends
- **checkpoint path**: pass the timestamped DIR not the `latest` file. e.g. `./logdir/.../ckpt/20260129T183613F519148`
- **crafter area**: `crafter_small` uses `area=[32,32]` not default 64x64. plot_trajectories.py reads area from metadata
- **world seed**: crafter derives seed as `hash((self._seed, self._episode))`. fixed_seed resets _episode=0 so same world each ep

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
Drop MPLBACKEND=Agg if running with display. Plot types: trajectories, heatmap, activation, spatial, world, fullworld, animate, animate_world, all

### decode position
```
MPLBACKEND=Agg python dreamerv3/decode_position.py \
  --data ./logdir/crafter_small_1m/trajectories \
  --save ./logdir/crafter_small_1m/decoder_results \
  --method both
```
Methods: regression (Ridge), classification (pRNN-style cross-entropy), both. Add `--no_per_neuron` to skip per-neuron R² analysis. Add `--save_model` to save fitted decoders for dream_decode. Add `--n_jobs N` for parallel jobs (`-1` = all CPUs). Add `--device cuda` for GPU classification (multi-GPU round-robin with `n_jobs>1`).

### layer-wise decoding (fast, recommended for cluster)
```
MPLBACKEND=Agg python dreamerv3/decode_position.py \
  --data ./logdir/crafter_small_1m/trajectories \
  --save ./logdir/crafter_small_1m/decoder_results \
  --mode layers --ridge_layers --n_jobs -1
```
`--ridge_layers`: closed-form Ridge instead of gradient descent. Metric is R² (higher=better). All (layer × fold) pairs run as one flat parallel pool with threads (no pickling). Auto-saves checkpoint to `<save>/layer_decode_checkpoint.pkl`. Add `--resume <path>` to continue from a partial run.
`--max_samples N` (default 10000): subsample timesteps before fitting — eliminates O(N) scaling.
`--max_dims D` (default 256): truncated PCA before Ridge — eliminates O(D²) scaling (critical for deter at 4096 dims). Set 0 to disable either. Without `--ridge_layers`, uses PyTorch classifier (CE loss, lower=better) with `n_iters=500`.

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

### plot training progress
```
MPLBACKEND=Agg python dreamerv3/plot_training.py \
  --logdir ./logdir/crafter_small_1m \
  --save ./logdir/crafter_small_1m/plots \
  --smooth 50
```
Reads `scores.jsonl` (episode score) and `metrics.jsonl` (per-achievement stats) from logdir. Produces `training_progress.png` with 3 panels: episode score, cumulative reward, per-achievement unlock rate. Add `--no_achievements` to skip panel 3.

### tests
```
PYTHONPATH=/Users/viggy/Desktop/dreamerv3f python embodied/tests/test_crafter_world.py
```

## deps not in requirements.txt that were needed
All now added: crafter, matplotlib, pandas, Pillow, ruamel.yaml. Install with `pip install -r requirements.txt`.
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
