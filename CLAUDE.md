# dreamerv3f repo guide

## arch
- `dreamerv3/main.py` — entry point, dispatches on `--script {train,train_eval,eval_only,eval_trajectory,dream_decode,dream_vs_future,replay_activations,parallel,parallel_env,parallel_envs,parallel_replay}`
- `dreamerv3/agent.py` — JAX agent (DreamerV3 world model)
- `dreamerv3/configs.yaml` — all configs. presets: `defaults`, `crafter`, `crafter_small`, `size{1m,12m,25m,50m,100m,200m,400m}`, `debug`, `atari`, `atari100k`, `procgen`, `minecraft`, `dmlab`, `dmc_proprio`, `dmc_vision`, `bsuite`, `loconav`, `multicpu`
- `dreamerv3/eval_trajectory.py` — records pos/activations/images per step, saves pkl. Auto-detects checkpoint from `logdir/ckpt/latest` if `--run.from_checkpoint` is empty. Inherits env settings from training run's saved `config.yaml`
- `dreamerv3/plot_trajectories.py` — plots: trajectories, heatmap, activation, world overlay, fullworld, worldview, world_only, animations (GIF/MP4)
- `dreamerv3/decode_position.py` — linear classification decoders to predict (x,y) from deter/stoch. Two modes: `standard` (single-repr decoding with CV) and `layers` (per-layer comparison boxplot). Includes `plot_probmap_on_world` (P(pos) heatmap on world map) and `plot_occupancy_vs_error` (occupancy hotspots + Manhattan error scatter). `--save_model` saves fitted decoders for dream_decode. Metric: mean Manhattan distance (tiles)
- `dreamerv3/dream_decode.py` — applies pretrained classification position decoder to policy-based imagination (dream) rollouts. Tests spatial coherence of dreamed trajectories
- `dreamerv3/dream_vs_future.py` — decodes policy-driven dream rollouts with a saved position decoder and compares them against the agent's *real future* trajectory from the same replay start (exported as `dream/future_pos` in report). Produces a divergence-vs-horizon curve (Manhattan tiles) with shuffled-future chance baseline. Uses policy actions (not real actions), so divergence conflates dynamics drift + policy/behavior mismatch — by design. Does NOT require `include_position` (player_pos is carried in obs for logging, never fed to the encoder)
- `dreamerv3/dream_seed_ablation.py` — four-condition dream **seeding** ablation (crossed 2×2: `deter ∈ {real, noise}` × `stoch ∈ {posterior(image), prior}`). A=both (current), B=just-latent, C=just-image (noise deter+real image), D=neither (noise). Optional 5th condition **E (blank-slate)** via `--dream_seed_ablation.include_e`: all-zero deter with NO warmup + prior stoch, rolled out under the policy (what the WM hallucinates from nothing). Unlike A′, E is a full member of the standard panels (appended to `CONDITIONS` by `_activate_e()`) and dumps `dream_deter_E.pkl`. Seed construction lives in `agent.report()` behind `agent.report_seed_ablation`; uses `RSSM.post_from_deter` to inject an arbitrary deter into the posterior. Each seed tiled `seed_ablation_nseeds` times → cross-seed variability. Decodes every dreamed latent with the saved position decoder. Produces `seed_ablation_curves.png` (6 panels: displacement-from-own-start, distance-from-real-start, cross-seed variability, step-to-step displacement/continuity — decoded distance from previous tile per step; read with central=mean since median collapses to 0 — plus P[dx=0] (per-step stay probability) and E[dx|dx>0] (mean jump size over moving steps), both always mean-aggregated regardless of `central`). Full walkthrough in `docs/training_and_dream_loop.md` §11+§13
- `dreamerv3/replay_activations.py` — replays saved trajectory observations through a (possibly untrained) agent to record activations. Used as control: does spatial decoding require learned representations or is it trivially present? Supports `--replay_activations.load_checkpoint False` for random-weight control
- `dreamerv3/state_probe.py` — interoceptive state probe: fix the visual scene, sweep the status vitals shown in the image (health/food/drink/energy), and measure how the policy shifts (action distribution + value). Mutates crafter's `player.inventory` and re-renders (terrain untouched, on-distribution icons) rather than editing pixels. Drives the real policy, capturing a frame every `warmup_steps`; at each frame runs the full factorial grid over `vitals` × `levels`. `--state_probe.carry_mode {fresh,history}` (default `fresh`): `fresh` = single-frame stimulus with is_first=True resetting the recurrent state (clean per-value counterfactual); `history` = probe from the live drive carry (is_first=False) for a contextualized read, but confounded by the real vitals already in the carry (JAX-functional, so the drive is unperturbed). Requires `agent.probe_policy` (set by main.py) which makes `agent.policy` emit `policy_logits` + `value`. Outputs: `value_vs_vitals.png`, `action_probs_vs_vitals.png`, `value_heatmaps.png` (vital pairs), `probe_frames.png`, `state_probe_results.pkl` — `fresh` keeps these canonical names, `history` (and any non-fresh mode) gets a `_{mode}` filename suffix + `[carry=mode]` title tag so both coexist in one dir; `carry_mode` is stored in the pkl `metadata`. Replot via `--state_probe.from_pkl`. NOTE `value` is the head's raw pred() (normalized space) — read its shape vs vitals, not absolute return
- `dreamerv3/inspect_replay.py` — dumps a **faithful** report-batch from the actual replay buffer (fresh eval rollouts → `make_stream('report')`): recomputed posterior deter (with the loaded checkpoint), the real replay images, player_pos, and the exact per-dim `mu`/`sd` the seed ablation's noise deter is built from. Emitted via `agent.report_dump_batch` as `dump/{deter,mu,sd,player_pos,image}`. Saves `replay_batch.pkl` for `inspect_replay_deter.ipynb` (set `SOURCE='replay_dump'`). Distinct from reusing saved trajectories (which record deter from eval_trajectory's own rollout)
- `dreamerv3/inspect_replay_deter.ipynb` — notebook to sample a report-style batch (from saved trajectories OR the faithful `replay_batch.pkl` via `SOURCE`), inspect the 16 sequence start locations + start frames on the world map, the per-dim deter `mu`/`sd`, the noise-null decode comparison (matched/matched_relu/truncnorm/shuffle/zero-mean), and per-sequence deter histograms
- `dreamerv3/tuning_curve.py` — spatial tuning curve analysis: classifies neurons into cell types (place, border, HD, etc.) using pynapple. Computes per-neuron spatial info, EV reliability, autocorrelation metrics, and HD mutual info across all recorded layers. Interactive viewer via `--from_pkl`
- `dreamerv3/analyze_tuning.py` — tuning curve analysis: clustering, metric-space embedding, distributions. Three modes: `autocorr` (PCA/t-SNE/UMAP on autocorrelation maps + HDBSCAN), `metrics` (Isomap on per-neuron feature vectors), `distributions` (per-metric histogram + example tuning curves at quantile positions). Loads `tuning_results.pkl`. Requires `umap-learn` and `hdbscan` for full autocorr pipeline (graceful fallback to PCA+t-SNE if missing)
- `dreamerv3/manifold_analysis.py` — neural manifold analysis: sRSA (spatial structure in reps), Isomap (2D manifold visualization), SW distance (dream-to-wake proximity). Compares wake vs dream activations on dyn/deter and dyn/stoch layers
- `dreamerv3/compare_conditions.py` — cross-condition comparison: loads layer_decode_results.pkl and tuning_results.pkl from N runs, produces heatmaps, line plots, cell type composition charts, and summary CSV
- `dreamerv3/plot_training.py` — reads `scores.jsonl` + `metrics.jsonl`, produces training_progress.png with episode score, cumulative reward, Crafter score, per-achievement unlock rates, and optionally loss/reward/value panels
- `dreamerv3/run_info.py` — lightweight run provenance logger. `log_run_info(save_dir, stage, args, outputs, extra)` appends a JSON entry to `<save_dir>/run_info.json` with timestamp, git SHA, command line, SLURM job ID, and all args. Integrated into decode_position, plot_trajectories, plot_training, and tuning_curve
- `embodied/envs/crafter.py` — Crafter wrapper. `fixed_seed=True` resets `_episode=0` before each reset so same world. `random_spawn=True` relocates player to random walkable tile each episode. `egocentric_view=N` (odd int, e.g. 7) renders N×N egocentric view centered on player facing direction; inventory bar is copied from the standard render into the bottom rows. Also exposes: `log/player_facing_x`, `log/player_facing_y` (facing direction as ±1/0 ints), `log/achievement_*` (per-achievement binary), `log/reward` (raw crafter reward). Writes `stats.jsonl` to logdir if configured
- `embodied/jax/agent.py` — JAX Agent.__new__ calls internal.setup() then __init__. Line 72: jax.devices()
- `embodied/jax/internal.py` — setup() sets jax platform, XLA flags. Line 34: `platform and jax.config.update('jax_platforms', platform)`
- `embodied/tests/test_crafter_world.py` — world consistency tests (fixed_seed, walkable spawn, determinism)

## SLURM scripts (repo root)
- `run_Crafter.sh` — train model (H100, parametrized env/size/activation/wd). Saves `hyperparams.txt` to logdir
- `run_Trajectory.sh` — eval trajectories (CPU, matches training hyperparams). Auto-resolves checkpoint from `$LOGDIR/ckpt/latest`
- `run_Loop.sh` — full pipeline: train → eval trajectory → plot → layer decoding → tuning (A100). Saves `hyperparams.txt`
- `run_Decoding.sh` — standard position decoding (GPU). Settings: repr, device, n_jobs
- `run_DreamVsFuture.sh` — decode policy dreams and compare to real future trajectory (GPU). Saves decoder if missing, then runs `dream_vs_future`. Settings: LOGDIR, DREAM_EPISODES, DREAM_BATCHES
- `run_DreamSeedAblation.sh` — four-condition dream seeding ablation (GPU). Saves decoder if missing, then runs `dream_seed_ablation`. Settings: LOGDIR, DREAM_EPISODES, DREAM_BATCHES, N_SEEDS
- `run_DSA_Aprime.sh` — dream seeding ablation **with condition A′ (perturbed state)** enabled (GPU, 3-net array: worlds_test_45/masked_k5/rollout_k5). Saves decoder if missing, then runs `dream_seed_ablation --perturb_state True` on the egocentric env; emits `condition_compare_ADAprime.png` (A vs D vs A′). Settings: PERTURB_VITALS, PERTURB_POOL, HORIZON, N_SEEDS, SAVE_ACTIVATIONS (for Phase-2 SW-dist), FROM_PKL (replot). Phase-1 only (no manifold step)
- `run_StateProbe.sh` — interoceptive state probe (GPU/CPU). Sweeps status vitals in the image, measures policy/value shift. Settings: LOGDIR, NUM_FRAMES, WARMUP_STEPS, VITALS, LEVELS, FROM_PKL
- `run_LayerDecoding.sh` — layer-wise decoding (A100). Settings: mode, holdout, resume
- `run_Plotting.sh` — plot trajectories + training progress (CPU). Settings: plot type, animation, smoothing
- `run_Tuning.sh` — tuning curve analysis (A100). Settings: layers, thresholds, n_jobs
- `run_Clustering.sh` — tuning curve analysis via PCA/t-SNE/UMAP + HDBSCAN / metric distributions (CPU, day partition). Settings: FROM_PKL, n_components, perplexity, umap_neighbors, min_cluster_size
- `run_Manifold.sh` — neural manifold analysis (CPU, day partition). Settings: LOGDIR, DREAM_DATA, LAYERS, MAX_WAKE_SAMPLES, N_NEIGHBORS, NO_ISOMAP, SEED_ABLATION_DIR (set → condition-overlay SW-over-time mode)

## run provenance
Every analysis script appends to `<save_dir>/run_info.json` via `dreamerv3/run_info.py`:
- `stage`, `timestamp`, `git_sha`, `command`, `slurm_job_id`, `args`, `outputs`, `extra`

Training runs also produce:
- `<logdir>/config.yaml` — full resolved DreamerV3 config (saved by `main.py`)
- `<logdir>/hyperparams.txt` — SLURM-level settings (saved by `run_Crafter.sh` / `run_Loop.sh`)

## critical gotchas
- **jax.platform**: default is `cuda` (configs.yaml). On mac/cpu: `--jax.platform cpu` or get AssertionError in jax backends
- **checkpoint path**: pass the timestamped DIR not the `latest` file. e.g. `./logdir/.../ckpt/20260129T183613F519148`. eval_trajectory auto-detects from `logdir/ckpt/latest` if omitted
- **crafter area**: `crafter_small` uses `area=[32,32]` not default 64x64. plot_trajectories.py reads area from metadata
- **seed flow**: `--seed` sets `config.seed` (default 0). `make_env` computes `env_seed = hash((config.seed, env_index)) % (2**32 - 1)` (see `main.py:299`). Crafter world seed = `hash((env_seed, episode_number)) % (2**31 - 1)` (inside `crafter.Env.reset`). `fixed_seed=True` resets episode_number to 0 each reset → same world. `fixed_seed=False` increments naturally → different world each episode. Different `--seed` → different world even with `fixed_seed=True`
- **egocentric_view**: must be an odd integer (e.g. 7). `egocentric_view=0` disables it (default). Set in configs.yaml under `env.crafter.egocentric_view` or pass `--env.crafter.egocentric_view 7`
- **layer decode checkpoint**: checkpoint files record which metric was used (`manhattan`, `ce_loss`). Mismatched metric triggers a warning and checkpoint is ignored — delete stale checkpoint if switching modes
- **layer decode metric**: mean Manhattan distance in tiles (joint x,y, lower = better)
- **layer decode holdout default**: `--holdout_frac` defaults to 0.2 (auto 80/20 episode split, no CV). Use `--holdout_frac 0` for CV mode. Classification uses Manhattan-based early stopping on a validation split (not CE loss)

## commands

### train
```
python dreamerv3/main.py --configs crafter_small size25m --logdir ./logdir/my_run \
  --env.crafter.random_spawn True --env.crafter.fixed_seed False
```

### eval trajectory
```
python dreamerv3/main.py \
  --configs crafter_small size25m \
  --logdir ./logdir/my_run \
  --script eval_trajectory \
  --run.from_checkpoint ./logdir/my_run/ckpt/TIMESTAMP_DIR \
  --eval_trajectory.num_episodes 100 \
  --eval_trajectory.save_path ./logdir/my_run/trajectories \
  --seed 42 --jax.platform cpu
```
Saves: per-episode pkl + all_episodes.pkl with metadata {env_seed, world_seed, fixed_seed, task, area}

### plot trajectories
```
MPLBACKEND=Agg python dreamerv3/plot_trajectories.py \
  --data ./logdir/my_run/trajectories --plot all --save ./logdir/my_run/plots
```
Drop MPLBACKEND=Agg if running with display. Plot types: trajectories, heatmap, activation, spatial, world, fullworld, animate, animate_world, worldview, world_only, all. Extra args for worldview/animate: `--egocentric_view N`, `--view_half N`, `--window_tiles N`, `--step_ms N`, `--mp4` (save as MP4 instead of GIF).

### decode position (standard)
```
MPLBACKEND=Agg python dreamerv3/decode_position.py \
  --data ./logdir/my_run/trajectories \
  --save ./logdir/my_run/decoder_results \
  --n_jobs -1
```
Uses pRNN-style classification (linear layer + CrossEntropyLoss over grid cells). Key args: `--repr {deter,stoch,combined,all}`, `--save_model`, `--n_jobs N`, `--device cuda`, `--n_iters N` (default 5000), `--patience N` (default 500), `--min_bbox N`.

### layer-wise decoding

**Fastest — holdout mode (no CV), requires two trajectory sets:**
```
MPLBACKEND=Agg python dreamerv3/decode_position.py \
  --data ./trajectories_train --test_data ./trajectories_test \
  --save ./decoder_results \
  --mode layers --n_jobs -1 --device cuda
```
Trains classifier on `--data`, evaluates on `--test_data`. No folds at all.

**Save trained decoders for reuse:**
```
MPLBACKEND=Agg python dreamerv3/decode_position.py \
  --data ./trajectories --save ./decoder_results \
  --mode layers --n_jobs -1 --device cuda --save_model
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
  --mode layers --n_jobs -1 --device cuda --holdout_frac 0
```
Uses 5-fold KFold (NOT LOGO — LOGO creates O(N_eps × N_layers) jobs, crushingly slow with randspawn).
Key args: `--max_samples 10000`, `--n_cv_folds 5`.

### dream decode
```
# Step 1: Save decoder model
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
Outputs: dream_trajectories_world.png, dream_probmap_*.png, dream_vs_real.png, dream_results.pkl
Additional dream_decode configs: `--dream_decode.num_batches N`, `--dream_decode.num_episodes N`.

### dream vs future (trajectory divergence)
```
# Step 1: save a position decoder (as in dream decode)
MPLBACKEND=Agg python dreamerv3/decode_position.py \
  --data ./logdir/my_run/trajectories \
  --save ./logdir/my_run/decoder_results --repr deter --save_model

# Step 2: decode policy dreams and compare to the real future trajectory
python dreamerv3/main.py \
  --configs crafter_small size25m \
  --logdir ./logdir/my_run \
  --script dream_vs_future \
  --run.from_checkpoint ./logdir/my_run/ckpt/TIMESTAMP_DIR \
  --dream_vs_future.decoder_model ./logdir/my_run/decoder_results/classifier_deter.pkl \
  --dream_vs_future.save_path ./logdir/my_run/dream_vs_future \
  --dream_vs_future.num_batches 50 --seed 42 --jax.platform cpu
```
Imagines policy-driven rollouts from replay states, decodes each latent step to (x,y), and compares to the agent's real future positions from the same start (`obs['player_pos'][T//2 : T//2+H]`, exported as `dream/future_pos`). Measures whether the dream stays spatially coherent or diverges. Uses **policy actions**, so divergence mixes dynamics drift + policy/behavior mismatch. N rollouts ≈ RB(=6) × num_batches. Does NOT need `include_position` (player_pos is logged but never encoded). Requires `report_length ≥ T//2 + imag_length` (default 32 ≥ 16+15 ✓). Prints step-0 calibration diagnostic (decoder error floor).
Outputs: divergence_vs_horizon.png (mean±IQR dream-vs-real, shuffled-future chance baseline, real displacement), dream_vs_real.png, dream_vs_future_results.pkl.
Additional configs: `--dream_vs_future.num_episodes N`, `--dream_vs_future.num_batches N`, `--dream_vs_future.warmup W` (observed steps before dream, 0→report_length//2), `--dream_vs_future.horizon H` (dream length, 0→imag_length; W+H≤report_length). NOTE: results generated before the report() T-clobber fix (docs §13) were misaligned ~5 steps and should be regenerated.

### dream seed ablation (four-condition crossed 2×2)
```
# Step 1: save a position decoder (as in dream decode / dream vs future)
# Step 2: run the ablation
python dreamerv3/main.py \
  --configs crafter_small size25m \
  --logdir ./logdir/my_run \
  --script dream_seed_ablation \
  --run.from_checkpoint ./logdir/my_run/ckpt/TIMESTAMP_DIR \
  --dream_seed_ablation.decoder_model ./logdir/my_run/decoder_results/classifier_deter.pkl \
  --dream_seed_ablation.save_path ./logdir/my_run/dream_seed_ablation \
  --dream_seed_ablation.num_batches 10 --dream_seed_ablation.n_seeds 16 \
  --seed 42 --jax.platform cpu
```
Seeds policy dreams four ways and decodes each: A=both (real deter+image, current), B=just-latent (real deter+prior), C=just-image (noise deter+real image), D=neither (noise deter+prior). A/B share the real deter; C/D share the same matched-stats noise deter. Seeds come from the full-sequence posterior recomputed with the loaded checkpoint (matches training's dyn.starts, in the decoder's space — NOT the buffer's stored latents). **Dense warmups by default** (`warmup 0` → seed at every W in [1, report_length−horizon], mirroring training; `warmup N>0` → single warmup N), pooled over warmups. Each seed tiled `n_seeds` times for cross-seed variability. Does NOT need `include_position` (player_pos carried for logging only). Outputs: `seed_ablation_curves.png` (7 panels, laid out in 2 rows: displacement-from-own-start, distance-from-real-start, cross-seed variability, step-to-step displacement/continuity — decoded distance from previous tile per step; read with central=mean since median collapses to 0 — plus P[dx=0] (per-step stay probability) and E[dx|dx>0] (mean jump size over moving steps), both always mean-aggregated regardless of `central`, plus **G) deviation-from-real-trajectory by warmup** — per-step |decoded_dream_t − real_t|, a tracking-fidelity metric distinct from panel B's distance-from-real-*start*, grouped into `DEFAULT_N_WARMUP_BINS` (=5) equal-count warmup bins auto-derived from the actual warmups present (`_auto_warmup_bins`, so the full range is covered on any horizon — no warmup dropped): A/B binned by warmup (color = warmup group dark→light, linestyle A '-' / B ':'), C/D drawn as a single POOLED gray line each (C '--', D '-.'). The warmup axis is only meaningful for A/B (real deter accumulates observation 1..W); C depends on W only via the single seed image and D not at all, and since the position decoder reads from the (noise) deter, C/D both start ~4 tiles off — so they're shown pooled as flat null baselines rather than spuriously binned. `bin_conditions`/`pooled_conditions` in `_draw_deviation_by_warmup` control the split. Lines-only with its own two-part legend, honors `central`; drawn when raw decoded positions are available so it regenerates under `--from_pkl`), `error_vs_warmup.png`, `decoded_dreams_by_condition.png` (4×N grid on the world map: rows = conditions A/B/C/D, columns = N distinct example dreams each with its own start/warmup seed frame — shared down a column so conditions are directly comparable; all seeds drawn per cell, colored by dream step — reconstructed from the pkl's raw decoded positions + `metadata` env_seed, so it also regenerates under `--from_pkl`), `decoded_dreams_animation.gif` (uncluttered counterpart: 1×4 condition panels for ONE example start/warmup, stepping through the seeds one dream at a time over the bold cyan real trajectory; controlled by `--dream_seed_ablation.make_gif`/`gif_col`/`gif_fps`; also regenerates under `--from_pkl`), `dream_seed_ablation_results.pkl`. Additional configs: `--dream_seed_ablation.num_episodes N`, `--dream_seed_ablation.warmup W`, `--dream_seed_ablation.warmup_stride S` (subsample dense warmups; memory ~ Nw×n_seeds×4), `--dream_seed_ablation.horizon H` (0→imag_length; W+H≤report_length), `--dream_seed_ablation.n_example_dreams N` (# columns in the decoded_dreams grid; 0 disables), `--dream_seed_ablation.noise_mode {matched,matched_relu,truncnorm,shuffle}` (noise-deter null for conditions C/D — `matched`=mu+sd·N default but ~36% negative/off the relu manifold, `matched_relu`=clipped ≥0, `truncnorm`=N(mu,sd)≥0, `shuffle`=per-dim bootstrap of real deter preserving marginal+non-negativity; requires a fresh run, baked into the rollout), `--dream_seed_ablation.perturb_state` (bool, default False → adds **condition A′ (Aprime)**: real deter + posterior from a seed image whose interoceptive status bar is overwritten with random vitals — isolates the causal effect of the interoceptive channel on the dream. Real deter ⇒ step-0 decodes identically to A; divergence beyond A ⇒ vitals steer the dream. Each of the `n_seeds` seeds gets an INDEPENDENT uniform vital draw (redrawn per batch), so A′'s cross-seed spread measures state-perturbation sensitivity — unlike A/B/C/D whose seeds share one posterior and differ only by the z-sample. Built via `post_from_deter(real_deter, enc(perturbed_image))` in `agent.report()`; the bar pool is rendered Python-side in the driver — reusing `state_probe`'s crafter unwrap — and baked onto the model as `agent.model.aprime_pool` (crafter rendering can't run in the jitted report). Companion flags `--dream_seed_ablation.perturb_vitals` (default `health,food,drink,energy`; each uniform int 0–9) and `--dream_seed_ablation.perturb_pool N` (# pre-rendered bars sampled with replacement, default 1024). Emits an extra output `condition_compare_ADAprime.png` — a 4-panel **A vs D vs A′** overlay (distance-from-real-trajectory, displacement-from-own-start, step-to-step continuity, cross-seed variability) with A/D as full-real/pure-noise bookends; A′ is threaded into `decoded`/curves/summary/pkl (as a 5th condition + `aprime_pool_idx`) but kept OUT of the standard 4-condition panels. Regenerates under `--from_pkl`. Requires the egocentric env with an inventory bar), `--dream_seed_ablation.include_e` (bool, default False → adds **condition E (blank-slate)**: a completely fresh all-zero deter with NO warmup, stoch sampled from the prior over that zero deter, then rolled out under the policy — tests what the world model hallucinates from nothing. Warmup-independent by construction (the zero deter is identical across warmup slots; only the per-seed z varies), so it reads as a flat baseline on the warmup axis. Unlike A′, E IS a first-class member of the standard panels — `_activate_e()` appends `'E'` to the module `CONDITIONS`, so it appears in `seed_ablation_curves.png`, the decoded-dreams grid/gif (which widen to `len(CONDITIONS)` rows/panels), the summary, and the pkl; with `--save_activations` it also dumps `dream_deter_E.pkl` (picked up by `manifold_analysis.py` per-condition and `--seed_ablation_dir` overlay, auto-detected). Regenerates under `--from_pkl`), `--dream_seed_ablation.real_style {bold,ghost,none}` (real trajectory in the decoded-dream grid/gif: `bold`=thick cyan on top (can occlude the dream when they track closely), `ghost`=thin dashed cyan under the dream so the dream stays visible (default), `none`=start marker only; plot-only, works via `--from_pkl`), `--dream_seed_ablation.central {median,mean}` (curve central tendency; spread pairs automatically: median→IQR, mean→±1 std; applies to all panels, error_vs_warmup, and printed summary), `--dream_seed_ablation.shading {band,bars,none}` (spread display: filled band, error bars, or line only). **Replot without rerunning:** the results pkl retains raw decoded dream positions + start/future positions, so `--dream_seed_ablation.from_pkl <dream_seed_ablation_results.pkl>` regenerates both plots under a new `central`/`shading` in seconds — no checkpoint, decoder, or rollouts needed (plots go to `--dream_seed_ablation.save_path` or the pkl's own dir). Decoders themselves are also never retrained here — they're loaded from the saved `--dream_seed_ablation.decoder_model` pkl. Only extending `horizon` requires a fresh full run. Design/rationale: `docs/training_and_dream_loop.md` §11+§13.

### replay activations (untrained control)
```
# Untrained control (random weights):
python dreamerv3/main.py \
  --configs crafter_small size25m \
  --logdir ./logdir/my_run \
  --script replay_activations \
  --replay_activations.source_data ./logdir/my_run/trajectories \
  --replay_activations.save_path ./logdir/my_run/untrained_activations \
  --replay_activations.load_checkpoint False \
  --jax.platform cpu

# Trained replay (same obs sequence, trained weights):
python dreamerv3/main.py \
  --configs crafter_small size25m \
  --logdir ./logdir/my_run \
  --script replay_activations \
  --run.from_checkpoint ./logdir/my_run/ckpt/TIMESTAMP \
  --replay_activations.source_data ./logdir/my_run/trajectories \
  --replay_activations.save_path ./logdir/my_run/trained_replay \
  --jax.platform cpu
```
Output format is identical to eval_trajectory — run decode_position/tuning_curve on the output as usual. Key args: `--replay_activations.max_episodes N` (0=all), `--replay_activations.load_checkpoint {True,False}`.

### state probe (interoceptive stimulus manipulation)
```
python dreamerv3/main.py \
  --configs crafter_small size25m \
  --logdir ./logdir/my_run \
  --script state_probe \
  --run.from_checkpoint ./logdir/my_run/ckpt/TIMESTAMP_DIR \
  --state_probe.save_path ./logdir/my_run/state_probe \
  --state_probe.num_frames 12 --state_probe.warmup_steps 20 \
  --state_probe.vitals food,drink,energy --state_probe.levels 0,3,6,9 \
  --seed 45 --jax.platform cpu
```
Fixes each captured scene and re-renders it with every combination of the status vitals (factorial `vitals` × `levels`), then runs the policy on each and records the action distribution + value. Produces `value_vs_vitals.png`, `action_probs_vs_vitals.png`, `value_heatmaps.png`, `probe_frames.png`, `state_probe_results.pkl`. Replot without rerunning: `--state_probe.from_pkl <state_probe_results.pkl>`. Config: `--state_probe.{num_frames,warmup_steps,vitals,levels,carry_mode,load_checkpoint}`. `--state_probe.carry_mode {fresh,history}` (default `fresh`) chooses whether each probe resets the recurrent state (fresh, clean counterfactual) or reads from the live drive carry (history, contextualized but confounded); non-fresh modes suffix their output filenames with `_{mode}` and tag titles `[carry=mode]` so they don't overwrite the fresh outputs. Reads the status bar the model was trained on (mutates `player.inventory`, terrain untouched); a flat response = the policy ignores the interoceptive bar (plausible for random-spawn navigation runs with `disable_mobs`). NOTE the field is `levels` not `values` (the latter collides with dict `.values()` on the config object).

### inspect replay (faithful report-batch dump)
```
python dreamerv3/main.py \
  --configs crafter_small size25m \
  --logdir ./logdir/my_run \
  --script inspect_replay \
  --run.from_checkpoint ./logdir/my_run/ckpt/TIMESTAMP_DIR \
  --inspect_replay.save_path ./logdir/my_run/replay_inspection \
  --inspect_replay.num_episodes 5 --inspect_replay.num_batches 2 \
  --seed 45 --jax.platform cpu
```
Dumps `replay_batch.pkl` (recomputed posterior deter + real replay images + player_pos + per-dim mu/sd) from the actual replay report stream. Load it in `inspect_replay_deter.ipynb` with `SOURCE='replay_dump'`. Match the training run's activations (pass `--agent.*.act` from `hyperparams.txt`) and env flags, same as the seed-ablation runner. Needs the same agent-block inheritance as other INFERENCE_SCRIPTS.

### plot training progress
```
MPLBACKEND=Agg python dreamerv3/plot_training.py \
  --logdir ./logdir/my_run \
  --save ./logdir/my_run/plots \
  --smooth 50
```
Reads `scores.jsonl` (episode score + per-episode achievement success) and `metrics.jsonl` (per-achievement stats) from logdir. Produces `training_progress.png` with up to 4 panels: episode score, cumulative reward, Crafter score (geometric mean of achievement success rates), per-achievement unlock rate. Add `--no_achievements` to skip the per-achievement panel. Add `--no_losses` to skip loss/reward/value panels. The Crafter score panel appears automatically when `scores.jsonl` contains per-episode achievement data (requires training with the updated logger).

### tuning curve analysis
```
MPLBACKEND=Agg python dreamerv3/tuning_curve.py \
  --data ./logdir/my_run/trajectories \
  --save ./logdir/my_run/tuning_results \
  --n_jobs -1
```
Analyzes all recorded layers (enc/*, dyn/*, pol/*, val/*). Computes per-neuron: 2D spatial tuning curves + spatial information (pynapple), HD tuning + mutual info, EV reliability, autocorrelation peaks/field size/asymmetry. Classifies neurons into 7 types: untuned, HD_cells, single_field, border_cells, spatial_HD, complex_cells, dead. Add `--test_data` for held-out EV reliability. Add `--layers dyn/deter dyn/stoch` to filter layers. Add `--no_hd` to skip HD analysis. Add `--no_plots` to skip plots. Add `--max_neurons N` to subsample large layers (0=all). Add `--interactive` to show interactive SI vs EV scatter during analysis. Add `--min_bbox N` to filter small-bbox episodes. Threshold overrides: `--SI_thresh`, `--EV_thresh`, `--EV_unthresh`, `--HD_thresh`.

Outputs: `tuning_results.pkl` (per-layer tuning curves, metrics, cell groups), `{layer}_si_ev_scatter.png`, `{layer}_cell_types.png`, `{layer}_example_tuning_curves.png`, `layer_summary.png`.

### compare conditions
```
MPLBACKEND=Agg python dreamerv3/compare_conditions.py \
  --conditions ./logdir/run1:label1 ./logdir/run2:label2 \
  --save ./logdir/comparison_plots
```
Loads `layer_decode_results.pkl` and `tuning_results.pkl` from N conditions and produces comparison plots. Key args: `--decode_subdir NAME` (default `layer_decoder_results`), `--tuning_subdir NAME` (default `tuning_results`), `--layers dyn/deter dyn/stoch` (optional filter), `--no_decode` / `--no_tuning` (skip one analysis type).

Outputs: `decode_heatmap.png` (conditions × layers heatmap), `decode_lineplot.png` (line plot with IQR bands), `tuning_si_heatmap.png`, `tuning_ev_heatmap.png`, `tuning_celltypes.png` (grouped stacked bar), `summary.csv`.

### tuning curve clustering
```
MPLBACKEND=Agg python dreamerv3/analyze_tuning.py \
  --from_pkl ./logdir/my_run/tuning_results/tuning_results.pkl \
  --save ./logdir/my_run/tuning_results/cluster_plots
```
Three modes via `--mode`:

**autocorr** (default): PCA/t-SNE/UMAP on spatial autocorrelation maps of tuning curves + HDBSCAN clustering on UMAP embedding. Key args: `--n_components 50` (PCA dims), `--perplexity 30` (t-SNE), `--umap_neighbors 15`, `--min_cluster_size 20` (HDBSCAN). Requires `umap-learn` + `hdbscan` for full pipeline; falls back to PCA + t-SNE if missing. Outputs per layer: `{layer}_scree.svg`, `{layer}_pca.svg`, `{layer}_tsne.svg`, `{layer}_umap.svg`, `{layer}_cluster_examples.svg`, `cluster_results.pkl`.

**metrics**: Isomap on per-neuron metric feature vectors (SI, EV, Moran's I, Geary's C, Getis-Ord G, field size, pf_peaks). Neurons with NaN in any metric are dropped. Key args: `--isomap_neighbors 15`, `--interactive` (launches click-to-inspect viewer: Isomap scatter on left, tuning curve + metric values on right). Outputs per layer: `{layer}_isomap_metrics_celltype.svg`, `{layer}_isomap_metrics_si.svg`, `metric_cluster_results.pkl`.

**distributions**: Per-metric histogram + example tuning curves at quantile positions (10%, 30%, 50%, 70%, 90%). Shows 3 neurons near each quantile. Outputs per layer per metric: `{layer}_dist_{metric}.png`, `distribution_results.pkl`.

Common args: `--layers dyn/deter dyn/stoch` (optional filter), `--no_normalize` (skip z-score).

```bash
# Metric-space interactive viewer
python dreamerv3/analyze_tuning.py \
  --from_pkl ./tuning_results/tuning_results.pkl \
  --save ./cluster_plots \
  --mode metrics --interactive --layers dyn/deter
```

### manifold analysis (sRSA, Isomap, SW distance)
```
MPLBACKEND=Agg python dreamerv3/manifold_analysis.py \
  --data ./logdir/my_run/trajectories \
  --dream_data ./logdir/my_run/dream_results/dream_results.pkl \
  --save ./logdir/my_run/manifold_results

# Condition-overlay mode: SW-over-time for seed-ablation A/B/C/D vs shared wake
MPLBACKEND=Agg python dreamerv3/manifold_analysis.py \
  --data ./logdir/my_run/trajectories \
  --seed_ablation_dir ./logdir/my_run/dream_seed_ablation_16_manifold \
  --save ./logdir/my_run/manifold_conditions --layers dyn/deter
```
Compares wake (real trajectory) and dream (imagination) activations on the neural manifold, following pRNN's `representationalGeometryAnalysis.py`. Four metrics: sRSA (Spearman rank correlation of spatial vs neural distances on wake, shown as P[neural|spatial] conditional histogram), Hill fit (saturating Hill function fitted to binned neural-vs-spatial distance, extracting dh_0/dh_inf/dx_1/2), Isomap (2D manifold visualization of wake+dream), SW distance (median min cosine distance from dream to nearest wake point, shown as pRNN-style vertical column). `--dream_data` accepts either a `dream_results.pkl` (from dream_decode with `save_activations=True`) or a second trajectory directory (for wake-vs-wake control). Key args: `--layers dyn/deter dyn/stoch` (default), `--max_wake_samples 4000`, `--n_neighbors 150` (Isomap), `--no_isomap` (skip Isomap for speed), `--no_hill` (skip Hill fit), `--min_bbox N`. **Condition-overlay mode** (`--seed_ablation_dir <dream_seed_ablation folder with dream_deter_{A,B,C,D}.pkl>`, mutually exclusive with `--dream_data`): instead of the full suite, produces one `{layer}_sw_over_time_by_condition.png` per layer overlaying SW-distance-vs-dream-step for the four seed-ablation conditions A/B/C/D against a **single shared wake manifold** (loaded once, so the comparison is against an identical reference). Conditions are processed one at a time (dumps are 8–16 GB), retaining only the per-point min-dists + timestep labels; results saved to `manifold_condition_results.pkl`. Requires the seed-ablation run to have been done with `--dream_seed_ablation.save_activations True`. `--shading {band,none}` controls the IQR band.

Outputs: `{layer}_srsa.png`, `{layer}_hillfit.png`, `{layer}_swdist.png` (SW-distance histogram, distance axis fixed to [0,1] — cosine dist on relu activations), `{layer}_sw_over_time.png` (median SW±IQR vs dream timestep; only when the dream source has a time axis, i.e. dream_deter dumps — shows drift onto/off the wake manifold as the dream rolls out), `{layer}_isomap_position.png`, `{layer}_isomap_wakedream.png`, `{layer}_wakesleep.png` (combined pRNN-style figure), `manifold_summary.png`, `manifold_results.pkl` (stores `sw_t_idx` per-row dream timestep for replotting the over-time curve). Condition-overlay mode instead emits `{layer}_sw_over_time_by_condition.png` + `manifold_condition_results.pkl` (per-condition min-dists + timestep labels).

### interactive tuning viewer (from precomputed pkl)
```
python dreamerv3/tuning_curve.py --from_pkl ./logdir/.../tuning_results/tuning_results.pkl
python dreamerv3/tuning_curve.py --from_pkl tuning_results.pkl --layers dyn/deter
```
Loads precomputed `tuning_results.pkl` and launches an interactive matplotlib SI vs EV scatter. Click any neuron to display its tuning curve in a side panel. No `--data` or `--save` required. If multiple layers exist, prompts for layer selection (or pass `--layers` to filter).

### tests
```
PYTHONPATH=. python embodied/tests/test_crafter_world.py
```

## deps
All in `requirements.txt`: crafter, matplotlib, pandas, Pillow, pynapple, ruamel.yaml, av (for MP4), etc.
JAX pinned to cuda in requirements but use `pip install jax==0.4.35 jaxlib==0.4.35 chex==0.1.87 optax==0.2.3` for cpu-compatible versions.

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
