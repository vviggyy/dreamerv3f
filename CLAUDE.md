# dreamerv3f repo guide

## arch
- `dreamerv3/main.py` — entry point, dispatches on `--script {train,eval_only,eval_trajectory,parallel}`
- `dreamerv3/agent.py` — JAX agent (DreamerV3 world model)
- `dreamerv3/configs.yaml` — all configs. presets: `defaults`, `crafter`, `crafter_small`, `size1m`, `debug`
- `dreamerv3/eval_trajectory.py` — records pos/activations/images per step, saves pkl
- `dreamerv3/plot_trajectories.py` — plots: trajectories, heatmap, activation, world overlay, fullworld, animations
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
