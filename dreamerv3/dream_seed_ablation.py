"""
Dream seeding ablation: how does what you seed a dream with shape the rollout?

Four conditions form a clean crossed 2x2 over the seed carry {deter, stoch}:

               image (posterior stoch)     no image (prior stoch)
  real deter      A. both (current)          B. just-latent
  noise deter     C. just-image              D. neither

  - deter is either the real posterior deter or a matched-stats noise deter
    (per-dim Gaussian from real posterior deter stats). A/B share the real deter;
    C/D share the SAME noise draw.
  - stoch is either the posterior inferred from the real seed frame
    (posterior(deter, image)) or the dynamics prior _prior(deter) (no image).

To MATCH THE TRAINING PROCESS, seeds are drawn at MANY warmups: by default the
report block seeds a dream at *every* valid timestep W in [1, report_length -
horizon] (mirroring training's dyn.starts, where every timestep is a dream
start), and the curves pool over all warmups. Seeds come from the full-sequence
posterior recomputed with the loaded checkpoint (== what training imagines from,
and in the position decoder's space) — NOT the buffer's stored latents. Set
--dream_seed_ablation.warmup N to pin a single warmup instead.

Each seed is tiled Nseed times (stochastic rollouts) for cross-seed variability.
Every dreamed latent step is decoded to (x, y) with a pretrained position decoder
(Manhattan tiles). Plots (all conditions; real trajectory on A and B):
  A) displacement from each condition's own start (every curve starts at 0)
  B) distance from the real start position
  C) cross-seed variability within each seed (no real curve)
  + error_vs_warmup: final-step distance-from-real-start as a function of warmup

Seed construction lives in agent.report() behind report_seed_ablation. See
docs/training_and_dream_loop.md 11+13.

Rollouts and decoding are the expensive part; results (raw decoded dream
positions + start/future positions) are saved to
dream_seed_ablation_results.pkl. To re-render plots with a different central
tendency / shading WITHOUT rerunning the agent, pass
--dream_seed_ablation.from_pkl <that pkl> (no checkpoint/decoder needed).
Extending the dream horizon still requires a fresh run.

Usage:
  python dreamerv3/main.py \
    --configs crafter_small size25m --logdir ./logdir/my_run \
    --script dream_seed_ablation \
    --run.from_checkpoint ./logdir/my_run/ckpt/TIMESTAMP_DIR \
    --dream_seed_ablation.decoder_model ./logdir/my_run/decoder_results/classifier_deter.pkl \
    --dream_seed_ablation.save_path ./logdir/my_run/dream_seed_ablation \
    --dream_seed_ablation.num_batches 10 --dream_seed_ablation.n_seeds 16 \
    --seed 42 --jax.platform cpu
"""

import pickle
from functools import partial as bind
from pathlib import Path

import elements
import embodied
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


CONDITIONS = ['A', 'B', 'C', 'D']
LABELS = {
    'A': 'A: both (real deter + image)',
    'B': 'B: just-latent (real deter + prior)',
    'C': 'C: just-image (noise deter + image)',
    'D': 'D: neither (noise deter + prior)',
}
COLORS = {'A': 'crimson', 'B': 'steelblue', 'C': 'darkorange', 'D': 'dimgray'}


def _manhattan(a, b):
    return np.abs(a - b).sum(axis=-1)


def _band(ax, data, color, label, ls='-', central='median', shading='band'):
    """Central-tendency curve + spread over rollouts. data: (N, steps).

    central: 'median' (spread = IQR) or 'mean' (spread = +/-1 std).
    shading: 'band' (filled region), 'bars' (error bars), or 'none' (line only).
    """
    steps = np.arange(data.shape[1])
    if central == 'mean':
        mid = np.nanmean(data, axis=0)
        sd = np.nanstd(data, axis=0)
        lo, hi = mid - sd, mid + sd
    else:
        mid = np.nanmedian(data, axis=0)
        lo = np.nanpercentile(data, 25, axis=0)
        hi = np.nanpercentile(data, 75, axis=0)
    if shading == 'bars':
        ax.errorbar(steps, mid, yerr=[mid - lo, hi - mid], fmt=ls, color=color,
                    linewidth=2.0, label=label, capsize=2, elinewidth=1.0)
    else:
        ax.plot(steps, mid, ls, color=color, linewidth=2.0, label=label)
        if shading == 'band':
            ax.fill_between(steps, lo, hi, color=color, alpha=0.15)


def compute_curves(decoded, start_pos, future_pos, central='median'):
    """Derive plot curves from raw decoded dream positions (pure, no I/O).

    decoded: {cond: (S, Nw, Nseed, H+1, 2)}, start_pos: (S, Nw, 2),
    future_pos: (S, Nw, H, 2). Returns dispA, distB, varC, realA, distB_by_w.
    distB_by_w uses `central` (median/mean); dispA/distB/varC/realA are raw
    per-rollout matrices (central tendency applied later at plot time).
    """
    S, Nw, Ns, Hp1, _ = decoded['A'].shape
    _agg = np.nanmean if central == 'mean' else np.nanmedian
    rs = start_pos[:, :, None, None, :]                  # (S, Nw, 1, 1, 2)
    dispA, distB, varC, distB_by_w = {}, {}, {}, {}
    for c in CONDITIONS:
        pos = decoded[c]                                 # (S, Nw, Ns, H+1, 2)
        dA = _manhattan(pos, pos[:, :, :, :1, :])        # (S, Nw, Ns, H+1)
        dB = _manhattan(pos, rs)                         # (S, Nw, Ns, H+1)
        dispA[c] = dA.reshape(S * Nw * Ns, Hp1)
        distB[c] = dB.reshape(S * Nw * Ns, Hp1)
        centroid = pos.mean(axis=2, keepdims=True)       # (S, Nw, 1, H+1, 2)
        varC[c] = _manhattan(pos, centroid).mean(axis=2).reshape(S * Nw, Hp1)
        distB_by_w[c] = _agg(dB[..., -1], axis=(0, 2))   # (Nw,)
    realA = np.concatenate([
        np.zeros((S, Nw, 1)),
        _manhattan(future_pos, start_pos[:, :, None, :]),      # (S, Nw, H)
    ], axis=2).reshape(S * Nw, Hp1)
    return dispA, distB, varC, realA, distB_by_w


def _print_summary(dispA, distB, varC, realA, central):
    _agg = np.nanmean if central == 'mean' else np.nanmedian
    print(f"\n[summary] {central} final-step (H) values, pooled over warmups:")
    for c in CONDITIONS:
        print(f"  {c}: own-disp={_agg(dispA[c][:, -1]):5.2f}  "
              f"real-dist={_agg(distB[c][:, -1]):5.2f}  "
              f"seed-var={_agg(varC[c][:, -1]):5.2f} tiles")
    print(f"  real: own-disp={_agg(realA[:, -1]):5.2f} tiles")


def replot_from_pkl(pkl_path, save_dir, central='median', shading='band'):
    """Regenerate plots from a saved results pkl — no agent/rollouts/decode."""
    pkl_path = Path(pkl_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Replotting from {pkl_path} (central={central}, shading={shading})")
    with open(pkl_path, 'rb') as f:
        r = pickle.load(f)
    dispA, distB, varC, realA, distB_by_w = compute_curves(
        r['decoded'], r['start_pos'], r['future_pos'], central)
    warmups = r['warmups']
    _print_summary(dispA, distB, varC, realA, central)
    print("\nGenerating plots...")
    plot_seed_ablation(dispA, distB, varC, realA, save_dir,
                       central=central, shading=shading)
    plot_error_vs_warmup(distB_by_w, warmups, save_dir, central=central)
    print(f"Done. Plots written to {save_dir}")


def plot_seed_ablation(dispA, distB, varC, realA, save_dir,
                       central='median', shading='band'):
    band = bind(_band, central=central, shading=shading)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    ax = axes[0]
    for c in CONDITIONS:
        band(ax, dispA[c], COLORS[c], LABELS[c])
    band(ax, realA, 'black', 'real trajectory', ls='--')
    ax.set_title('A) displacement from own start')
    ax.set_ylabel('Manhattan distance from start (tiles)')

    ax = axes[1]
    for c in CONDITIONS:
        band(ax, distB[c], COLORS[c], LABELS[c])
    band(ax, realA, 'black', 'real trajectory', ls='--')
    ax.set_title('B) distance from real start')
    ax.set_ylabel('Manhattan distance from real start (tiles)')

    ax = axes[2]
    for c in CONDITIONS:
        band(ax, varC[c], COLORS[c], LABELS[c])
    ax.set_title('C) cross-seed variability')
    ax.set_ylabel('Mean seed dispersion (tiles)')

    for ax in axes:
        ax.set_xlabel('Dream step (0 = seed)')
        ax.grid(alpha=0.3)
        ax.set_xlim(0, dispA['A'].shape[1] - 1)
        ax.set_ylim(bottom=0)
        ax.legend(loc='upper left', fontsize=8)

    fig.tight_layout()
    out = save_dir / 'seed_ablation_curves.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out.name}")


def plot_error_vs_warmup(distB_by_w, warmups, save_dir, central='median'):
    """Final-step distance-from-real-start as a function of warmup length."""
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for c in CONDITIONS:
        ax.plot(warmups, distB_by_w[c], '-o', color=COLORS[c],
                label=LABELS[c], markersize=3)
    ax.set_xlabel('Warmup (observed steps before dream)')
    ax.set_ylabel(f'{central.capitalize()} final-step distance from real start '
                  '(tiles)')
    ax.set_title('Dream fidelity vs warmup length')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    out = save_dir / 'error_vs_warmup.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out.name}")


def dream_seed_ablation(make_agent, make_env, make_replay, make_stream,
                        make_logger, args):
    cfg = args.dream_seed_ablation
    assert cfg.central in ('median', 'mean'), \
        f"central must be 'median' or 'mean', got {cfg.central!r}"
    assert cfg.shading in ('band', 'bars', 'none'), \
        f"shading must be 'band', 'bars', or 'none', got {cfg.shading!r}"

    # Fast path: replot from a saved results pkl (no agent/rollouts/decode).
    if cfg.from_pkl:
        save_dir = Path(cfg.save_path or str(Path(cfg.from_pkl).parent))
        replot_from_pkl(cfg.from_pkl, save_dir, cfg.central, cfg.shading)
        return

    assert cfg.decoder_model, "Must provide --dream_seed_ablation.decoder_model"
    n_seeds = cfg.n_seeds

    save_dir = Path(cfg.save_path or str(
        elements.Path(args.logdir) / 'dream_seed_ablation'))
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Agent + checkpoint (params only, skipping opt/ — see dream_vs_future).
    print("Creating agent...")
    agent = make_agent()
    logger = make_logger()
    import pickle as _pickle, pathlib as _pl
    with open(_pl.Path(args.from_checkpoint) / 'agent.pkl', 'rb') as _f:
        agent.load(_pickle.load(_f), regex=r'^(?!opt/)')
    print(f"Loaded checkpoint (params only): {args.from_checkpoint}")

    # 2. Collect fresh eval rollouts (fixed_seed world).
    print("Collecting replay data...")
    env = make_env(0, fixed_seed=True)
    try:
        env_seed = env._seed
    except (AttributeError, ValueError):
        env_seed = None
    try:
        area = tuple(env._env._world._mat_map.shape)
    except (AttributeError, ValueError):
        area = None
    metadata = {
        'env_seed': env_seed,
        'world_seed': hash((env_seed, 1)) if env_seed is not None else None,
        'fixed_seed': True, 'task': 'crafter', 'area': area,
    }
    env.close()

    replay = make_replay()
    fns = [bind(make_env, i, fixed_seed=True) for i in range(1)]
    driver = embodied.Driver(fns, parallel=False)
    driver.on_step(replay.add)
    episode_count = [0]
    driver.on_step(lambda tran, w: episode_count.__setitem__(
        0, episode_count[0] + int(tran['is_last'])))
    policy = lambda *a: agent.policy(*a, mode='eval')
    driver.reset(agent.init_policy)
    print(f"Running {cfg.num_episodes} eval episodes...")
    while episode_count[0] < cfg.num_episodes:
        driver(policy, steps=100)
    print(f"  Collected {episode_count[0]} episodes, replay size: {len(replay)}")

    # 3. Load decoder up front (we decode each batch as it arrives to keep the
    #    big (RB, Nw, Nseed, H+1, D) deter tensors from piling up in memory).
    print(f"\nLoading decoder from {cfg.decoder_model}...")
    from .decode_position import load_classifier_model
    clf, dec_meta = load_classifier_model(Path(cfg.decoder_model))
    width, height = dec_meta['width'], dec_meta['height']
    print(f"  Decoder: {dec_meta.get('repr_name', '?')}, grid={width}x{height}")

    def decode(deter):  # (..., D) -> (..., 2)
        lead, D = deter.shape[:-1], deter.shape[-1]
        cls = clf.predict_proba(deter.reshape(-1, D)).argmax(axis=1)
        xy = np.stack(np.unravel_index(cls, (width, height)), axis=1)
        return xy.reshape(*lead, 2).astype(float)

    # 4. Run report() per batch; decode dreamed latents immediately.
    print("Running report() for four-condition dense-warmup dreams...")
    stream = iter(agent.stream(make_stream(replay, 'report')))
    carry = agent.init_report(args.batch_size)

    decoded = {c: [] for c in CONDITIONS}   # each: list of (RB, Nw, Nseed, H+1, 2)
    starts, futures = [], []
    warmups = None
    for b in range(cfg.num_batches):
        batch = next(stream)
        carry, mets = agent.report(carry, batch)
        if 'seedabl/A/deter' not in mets:
            print("ERROR: seedabl/* missing. Is agent.report_seed_ablation on, "
                  "and does obs carry player_pos?")
            return
        for c in CONDITIONS:
            decoded[c].append(decode(np.array(mets[f'seedabl/{c}/deter'])))
        starts.append(np.array(mets['seedabl/start_pos']))    # (RB, Nw, 2)
        futures.append(np.array(mets['seedabl/future_pos']))  # (RB, Nw, H, 2)
        if warmups is None:
            warmups = np.array(mets['seedabl/warmups'])       # (Nw,)
        print(f"  Batch {b + 1}/{cfg.num_batches} done")

    for c in CONDITIONS:
        decoded[c] = np.concatenate(decoded[c], axis=0)  # (S, Nw, Nseed, H+1, 2)
    start_pos = np.concatenate(starts, axis=0)           # (S, Nw, 2)
    future_pos = np.concatenate(futures, axis=0)         # (S, Nw, H, 2)
    S, Nw, Ns, Hp1, _ = decoded['A'].shape
    H = future_pos.shape[2]
    print(f"Pooled {S} start rows x {Nw} warmups {list(warmups)} x {Ns} seeds, "
          f"H+1={Hp1}")

    # 5. Curves (pooled over start rows and warmups; Nseed drives variability).
    central = cfg.central
    dispA, distB, varC, realA, distB_by_w = compute_curves(
        decoded, start_pos, future_pos, central)
    _print_summary(dispA, distB, varC, realA, central)

    # 6. Plots.
    print("\nGenerating plots...")
    plot_seed_ablation(dispA, distB, varC, realA, save_dir,
                       central=central, shading=cfg.shading)
    plot_error_vs_warmup(distB_by_w, warmups, save_dir, central=central)

    # 7. Save.
    results = {
        'decoded': decoded,          # {cond: (S, Nw, Nseed, H+1, 2)}
        'start_pos': start_pos, 'future_pos': future_pos, 'warmups': warmups,
        'dispA': dispA, 'distB': distB, 'varC': varC, 'realA': realA,
        'distB_by_warmup': distB_by_w,
        'S': S, 'Nw': Nw, 'n_seeds': Ns, 'H': H, 'Hp1': Hp1,
        'decoder_path': str(cfg.decoder_model), 'decoder_metadata': dec_meta,
        'metadata': metadata, 'conditions': CONDITIONS, 'labels': LABELS,
    }
    with open(save_dir / 'dream_seed_ablation_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {save_dir / 'dream_seed_ablation_results.pkl'}")

    try:
        from .run_info import log_run_info
        log_run_info(
            save_dir, 'dream_seed_ablation',
            args={'decoder_model': str(cfg.decoder_model),
                  'num_batches': cfg.num_batches, 'num_episodes': cfg.num_episodes,
                  'n_seeds': n_seeds, 'warmup': cfg.warmup,
                  'warmup_stride': cfg.warmup_stride, 'horizon': cfg.horizon,
                  'central': cfg.central, 'shading': cfg.shading,
                  'from_checkpoint': args.from_checkpoint},
            outputs=['seed_ablation_curves.png', 'error_vs_warmup.png',
                     'dream_seed_ablation_results.pkl'],
            extra={'n_start_rows': int(S), 'n_warmups': int(Nw),
                   'warmups': [int(w) for w in warmups], 'horizon': int(H)})
    except Exception as e:
        print(f"  (run_info logging skipped: {e})")

    logger.close()
    print("Done.")
