"""
Dream seeding ablation: how does what you seed a dream with shape the rollout?

Four conditions form a clean crossed 2x2 over the seed carry {deter, stoch}:

               image (posterior stoch)     no image (prior stoch)
  real deter      A. both (current)          B. just-latent
  noise deter     C. just-image              D. neither

  - deter is either the real posterior deter at the seed timestep, or a
    matched-stats noise deter (per-dim Gaussian from real posterior deter stats).
    A/B share the real deter; C/D share the SAME noise draw.
  - stoch is either the posterior inferred from the real seed frame
    (posterior(deter, image)) or the dynamics prior _prior(deter) (no image).

All four are rolled out with identical policy imagination for the same horizon,
each seed tiled Nseed times so per-condition cross-seed variability can be read
off. Every dreamed latent step is decoded to (x, y) with a pretrained position
decoder (Manhattan tiles).

Three plots, all conditions (+ the real trajectory on A and B) overlaid:
  A) displacement from each condition's own start (every curve starts at 0)
  B) distance from the real start position (start_pos); A/B start near the
     decoder error floor, C/D start displaced
  C) cross-seed variability within each condition (no real curve)

The seed construction lives in agent.report() behind report_seed_ablation; this
script drives it, decodes, and plots. See docs/training_and_dream_loop.md 11+13.

Usage:
  python dreamerv3/main.py \
    --configs crafter_small size25m \
    --logdir ./logdir/my_run \
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


def _band(ax, data, color, label, ls='-'):
    """Median + IQR band over rollouts. data: (N, steps)."""
    steps = np.arange(data.shape[1])
    med = np.nanmedian(data, axis=0)
    q1 = np.nanpercentile(data, 25, axis=0)
    q3 = np.nanpercentile(data, 75, axis=0)
    ax.plot(steps, med, ls, color=color, linewidth=2.0, label=label)
    ax.fill_between(steps, q1, q3, color=color, alpha=0.15)


def plot_seed_ablation(dispA, distB, varC, realA, save_dir):
    """Three-panel figure.

    dispA/distB[cond]: (N, H+1) per-rollout curves. varC[cond]: (S, H+1)
    per-start cross-seed dispersion. realA: (S, H+1) real displacement from start.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    # A) displacement from own start
    ax = axes[0]
    for c in CONDITIONS:
        _band(ax, dispA[c], COLORS[c], LABELS[c])
    _band(ax, realA, 'black', 'real trajectory', ls='--')
    ax.set_title('A) displacement from own start')
    ax.set_ylabel('Manhattan distance from start (tiles)')

    # B) distance from real start
    ax = axes[1]
    for c in CONDITIONS:
        _band(ax, distB[c], COLORS[c], LABELS[c])
    _band(ax, realA, 'black', 'real trajectory', ls='--')
    ax.set_title('B) distance from real start')
    ax.set_ylabel('Manhattan distance from real start (tiles)')

    # C) cross-seed variability (no real curve)
    ax = axes[2]
    for c in CONDITIONS:
        _band(ax, varC[c], COLORS[c], LABELS[c])
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


def dream_seed_ablation(make_agent, make_env, make_replay, make_stream,
                        make_logger, args):
    cfg = args.dream_seed_ablation
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

    # 2. Collect replay data (fixed_seed world, eval policy).
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

    # 3. Run report() to get per-condition dream latents.
    print("Running report() to get four-condition dream rollouts...")
    stream = iter(agent.stream(make_stream(replay, 'report')))
    carry = agent.init_report(args.batch_size)

    cond_deter = {c: [] for c in CONDITIONS}
    starts, futures = [], []
    for b in range(cfg.num_batches):
        batch = next(stream)
        carry, mets = agent.report(carry, batch)
        if f'seedabl/A/deter' not in mets:
            print("ERROR: seedabl/* missing from report metrics. "
                  "Is agent.report_seed_ablation enabled? Does obs carry "
                  "player_pos?")
            return
        for c in CONDITIONS:
            cond_deter[c].append(np.array(mets[f'seedabl/{c}/deter']))
        starts.append(np.array(mets['seedabl/start_pos']))
        futures.append(np.array(mets['seedabl/future_pos']))
        print(f"  Batch {b + 1}/{cfg.num_batches} done")

    start_pos = np.concatenate(starts, axis=0)            # (S, 2)
    future_pos = np.concatenate(futures, axis=0)          # (S, H, 2)
    S = start_pos.shape[0]
    RB = starts[0].shape[0]
    # Each report batch gives (RB*Nseed, H+1, D); regroup to (RB, Nseed, H+1, D)
    # then concat over batches on the start axis -> (S, Nseed, H+1, D).
    for c in CONDITIONS:
        per_batch = [a.reshape(RB, n_seeds, *a.shape[1:]) for a in cond_deter[c]]
        cond_deter[c] = np.concatenate(per_batch, axis=0)  # (S, Nseed, H+1, D)
    Hp1 = cond_deter['A'].shape[2]
    D = cond_deter['A'].shape[3]
    H = future_pos.shape[1]
    print(f"Collected {S} start states x {n_seeds} seeds, H+1={Hp1}, deter={D}")

    # 4. Decode every latent to (x, y) with the pretrained position decoder.
    print(f"\nLoading decoder from {cfg.decoder_model}...")
    from .decode_position import load_classifier_model
    clf, dec_meta = load_classifier_model(Path(cfg.decoder_model))
    width, height = dec_meta['width'], dec_meta['height']
    print(f"  Decoder: {dec_meta.get('repr_name', '?')}, grid={width}x{height}")

    decoded = {}
    for c in CONDITIONS:
        flat = cond_deter[c].reshape(-1, D)
        cls = clf.predict_proba(flat).argmax(axis=1)
        xy = np.stack(np.unravel_index(cls, (width, height)), axis=1)
        decoded[c] = xy.reshape(S, n_seeds, Hp1, 2).astype(float)

    # 5. Curves.
    # Graph A: displacement from own (per-rollout) start.
    dispA = {c: _manhattan(decoded[c], decoded[c][:, :, :1, :]).reshape(
        S * n_seeds, Hp1) for c in CONDITIONS}
    # Graph B: distance from the real start position.
    rs = start_pos[:, None, None, :]                      # (S,1,1,2)
    distB = {c: _manhattan(decoded[c], rs).reshape(S * n_seeds, Hp1)
             for c in CONDITIONS}
    # Graph C: within-start cross-seed dispersion (mean distance to seed centroid).
    varC = {}
    for c in CONDITIONS:
        centroid = decoded[c].mean(axis=1, keepdims=True)  # (S,1,H+1,2)
        varC[c] = _manhattan(decoded[c], centroid).mean(axis=1)  # (S,H+1)
    # Real trajectory displacement from its start (aligned: step0=0, 1..H=future).
    realA = np.concatenate([
        np.zeros((S, 1)),
        _manhattan(future_pos, start_pos[:, None, :]),     # (S, H)
    ], axis=1)                                             # (S, H+1)

    print("\n[summary] median final-step (H) values, per condition:")
    for c in CONDITIONS:
        print(f"  {c}: own-disp={np.median(dispA[c][:, -1]):5.2f}  "
              f"real-dist={np.median(distB[c][:, -1]):5.2f}  "
              f"seed-var={np.median(varC[c][:, -1]):5.2f} tiles")
    print(f"  real: own-disp={np.median(realA[:, -1]):5.2f} tiles")

    # 6. Plot.
    print("\nGenerating plots...")
    plot_seed_ablation(dispA, distB, varC, realA, save_dir)

    # 7. Save.
    results = {
        'decoded': decoded,          # {cond: (S, Nseed, H+1, 2)}
        'start_pos': start_pos,      # (S, 2)
        'future_pos': future_pos,    # (S, H, 2)
        'dispA': dispA, 'distB': distB, 'varC': varC, 'realA': realA,
        'S': S, 'n_seeds': n_seeds, 'H': H, 'Hp1': Hp1,
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
                  'n_seeds': n_seeds, 'from_checkpoint': args.from_checkpoint},
            outputs=['seed_ablation_curves.png',
                     'dream_seed_ablation_results.pkl'],
            extra={'n_starts': int(S), 'horizon': int(H)})
    except Exception as e:
        print(f"  (run_info logging skipped: {e})")

    logger.close()
    print("Done.")
