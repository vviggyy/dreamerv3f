"""
Dream vs Future: does the policy-driven dream diverge from the real trajectory?

The world model imagines rollouts from replay-buffer states using freshly
sampled *policy* actions (exactly what actor-critic training imagines on). This
script decodes each imagined latent step to (x, y) with a pretrained position
decoder and compares it against the agent's *real* future trajectory from the
same start (recorded in the replay buffer). The headline output is a
divergence-vs-horizon curve: how fast, and how far, the dream drifts from
reality.

Because the dream samples new policy actions rather than replaying the agent's
real actions, the divergence conflates world-model dynamics drift with
policy-vs-behavior action mismatch. That is intentional — the question here is
simply whether the imagined trajectory stays spatially coherent or flies off.

Position (player_pos) is carried in obs for logging only; with
include_position=False it is never fed to the encoder/decoder, so this analysis
adds no positional bias to the model.

Usage:
  python dreamerv3/main.py \
    --configs crafter_small size1m \
    --logdir ./logdir/my_run \
    --script dream_vs_future \
    --run.from_checkpoint ./logdir/my_run/ckpt/TIMESTAMP_DIR \
    --dream_vs_future.decoder_model ./logdir/my_run/decoder_results/classifier_deter.pkl \
    --dream_vs_future.save_path ./logdir/my_run/dream_vs_future \
    --jax.platform cpu
"""

import pickle
from functools import partial as bind
from pathlib import Path

import elements
import embodied
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_divergence_vs_horizon(err, baseline, real_disp, save_dir):
    """Mean +/- IQR Manhattan divergence between dream and real, per horizon step.

    Args:
        err: (N, H) per-rollout Manhattan distance dream-vs-real at each step
        baseline: (N, H) dream-vs-shuffled-real (chance / decoder-floor reference)
        real_disp: (N, H) how far the real trajectory itself moves from its start
        save_dir: Path
    """
    H = err.shape[1]
    steps = np.arange(1, H + 1)

    def band(ax, data, color, label):
        med = np.nanmedian(data, axis=0)
        q1 = np.nanpercentile(data, 25, axis=0)
        q3 = np.nanpercentile(data, 75, axis=0)
        ax.plot(steps, med, '-', color=color, linewidth=2.0, label=label)
        ax.fill_between(steps, q1, q3, color=color, alpha=0.2)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    band(ax, err, 'crimson', 'dream vs real future')
    band(ax, baseline, 'grey', 'dream vs shuffled future (chance)')
    band(ax, real_disp, 'steelblue', 'real displacement from start')

    ax.set_xlabel('Imagination step (horizon)')
    ax.set_ylabel('Manhattan distance (tiles)')
    ax.set_title('Dream vs real trajectory divergence (policy actions)')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(1, H)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    out = save_dir / 'divergence_vs_horizon.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def dream_vs_future(make_agent, make_env, make_replay, make_stream,
                    make_logger, args):
    """Compare policy-driven dream rollouts against the real future trajectory."""
    dvf_config = args.dream_vs_future
    assert dvf_config.decoder_model, \
        "Must provide --dream_vs_future.decoder_model"

    save_dir = Path(dvf_config.save_path or str(
        elements.Path(args.logdir) / 'dream_vs_future'))
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create agent and load checkpoint
    print("Creating agent...")
    agent = make_agent()
    logger = make_logger()

    cp = elements.Checkpoint()
    cp.agent = agent
    cp.load(args.from_checkpoint, keys=['agent'])
    print(f"Loaded checkpoint: {args.from_checkpoint}")

    # 2. Create env with fixed_seed, collect replay data
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
    world_seed = hash((env_seed, 1)) if env_seed is not None else None
    metadata = {
        'env_seed': env_seed,
        'world_seed': world_seed,
        'fixed_seed': True,
        'task': 'crafter',
        'area': area,
    }

    replay = make_replay()
    num_episodes = dvf_config.num_episodes
    fns = [bind(make_env, i, fixed_seed=True) for i in range(1)]
    driver = embodied.Driver(fns, parallel=False)
    driver.on_step(replay.add)

    policy = lambda *a: agent.policy(*a, mode='eval')
    driver.reset(agent.init_policy)

    episode_count = [0]
    def count_episodes(tran, worker):
        if tran['is_last']:
            episode_count[0] += 1
    driver.on_step(count_episodes)

    print(f"Running {num_episodes} eval episodes...")
    while episode_count[0] < num_episodes:
        driver(policy, steps=100)
    print(f"  Collected {episode_count[0]} episodes, "
          f"replay size: {len(replay)}")

    # 3. Create report stream and collect dream outputs
    print("Running report() to get dream rollouts + real futures...")
    stream = make_stream(replay, 'report')
    stream = agent.stream(stream)
    stream = iter(stream)

    carry = agent.init_report(args.batch_size)
    num_batches = dvf_config.num_batches

    all_dream_deter, all_start_pos, all_future_pos = [], [], []

    for batch_idx in range(num_batches):
        batch = next(stream)
        carry, mets = agent.report(carry, batch)

        if 'dream/deter' in mets:
            all_dream_deter.append(np.array(mets['dream/deter']))
        if 'dream/start_pos' in mets:
            all_start_pos.append(np.array(mets['dream/start_pos']))
        if 'dream/future_pos' in mets:
            all_future_pos.append(np.array(mets['dream/future_pos']))

        print(f"  Batch {batch_idx+1}/{num_batches} done")

    if not all_dream_deter:
        print("ERROR: No dream/deter in report metrics. "
              "Is agent.report_dream_feats enabled?")
        return
    if not all_future_pos:
        print("ERROR: No dream/future_pos in report metrics. "
              "player_pos must be present in obs (it is carried for logging "
              "even with include_position=False).")
        return

    dream_deter = np.concatenate(all_dream_deter, axis=0)   # (N, H, D)
    start_pos = np.concatenate(all_start_pos, axis=0)       # (N, 2)
    future_pos = np.concatenate(all_future_pos, axis=0)     # (N, L, 2)

    N, H, D = dream_deter.shape
    L = future_pos.shape[1]
    horizon = min(H, L)
    if H != L:
        print(f"  Note: dream horizon H={H} != future length L={L}; "
              f"comparing over first {horizon} steps.")
    dream_deter = dream_deter[:, :horizon]
    future_pos = future_pos[:, :horizon]
    print(f"Dream data: {N} rollouts, horizon={horizon}, deter_dim={D}")

    # 4. Load decoder and decode positions from imagined latents
    print(f"\nLoading decoder from {dvf_config.decoder_model}...")
    decoder_path = Path(dvf_config.decoder_model)
    from .decode_position import load_classifier_model
    clf, dec_meta = load_classifier_model(decoder_path)
    width, height = dec_meta['width'], dec_meta['height']
    print(f"  Classifier decoder: {dec_meta.get('repr_name', '?')}, "
          f"grid={width}x{height}")

    flat = dream_deter.reshape(-1, D)
    proba = clf.predict_proba(flat)                      # (N*horizon, W*H_grid)
    pred_cls = proba.argmax(axis=1)
    pred_xy = np.stack(np.unravel_index(pred_cls, (width, height)), axis=1)
    decoded_pos = pred_xy.reshape(N, horizon, 2).astype(float)  # (N, H, 2)

    # 5. Divergence metrics (Manhattan distance, tiles)
    def manhattan(a, b):
        return np.abs(a - b).sum(axis=-1)

    err = manhattan(decoded_pos, future_pos)             # (N, horizon)

    # Shuffled-future baseline: compare each dream to a different rollout's
    # future. Brackets the decoder error floor + world scale (chance level).
    perm = (np.arange(N) + max(1, N // 2)) % N           # deterministic derangement-ish
    baseline = manhattan(decoded_pos, future_pos[perm])  # (N, horizon)

    # How far the real trajectory itself moves from its own start (context).
    real_disp = manhattan(future_pos, future_pos[:, :1])  # (N, horizon)

    # Decoder calibration diagnostic: step-0 dream and real share (nearly) the
    # same starting state, so step-0 error ~ the decoder's own error floor.
    step0_err = float(np.median(err[:, 0]))
    start_err = float(np.median(manhattan(decoded_pos[:, 0], start_pos)))
    print(f"\n[calibration] median step-0 divergence: {step0_err:.2f} tiles "
          f"(decoder floor reference)")
    print(f"[calibration] median decoded step-0 vs real start: "
          f"{start_err:.2f} tiles")
    print(f"[divergence]  median final-step ({horizon}) divergence: "
          f"{np.median(err[:, -1]):.2f} tiles "
          f"(chance {np.median(baseline[:, -1]):.2f})")

    # 6. Plots
    print("\nGenerating plots...")
    plot_divergence_vs_horizon(err, baseline, real_disp, save_dir)

    # Overlay dream vs real *future* on the world map (reuse dream_decode's
    # plotter, but pass the future trajectory as the "real" line).
    if start_pos is not None:
        from .dream_decode import plot_dream_vs_real
        plot_dream_vs_real(decoded_pos, start_pos, future_pos, metadata, save_dir)

    # 7. Save results
    results = {
        'decoded_pos': decoded_pos,       # (N, H, 2) decoded dream positions
        'future_pos': future_pos,         # (N, H, 2) real future positions
        'start_pos': start_pos,           # (N, 2)
        'err': err,                       # (N, H) dream-vs-real Manhattan
        'baseline': baseline,             # (N, H) shuffled-future chance
        'real_disp': real_disp,           # (N, H) real displacement from start
        'horizon': horizon,
        'median_step0_err': step0_err,
        'median_final_err': float(np.median(err[:, -1])),
        'median_final_baseline': float(np.median(baseline[:, -1])),
        'decoder_path': str(decoder_path),
        'decoder_metadata': dec_meta,
        'metadata': metadata,
        'num_batches': num_batches,
        'num_episodes': num_episodes,
        'action_source': 'policy',
    }
    results_file = save_dir / 'dream_vs_future_results.pkl'
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {results_file}")

    try:
        from .run_info import log_run_info
        log_run_info(
            save_dir, 'dream_vs_future',
            args={
                'decoder_model': str(decoder_path),
                'num_batches': num_batches,
                'num_episodes': num_episodes,
                'from_checkpoint': args.from_checkpoint,
            },
            outputs=['divergence_vs_horizon.png', 'dream_vs_real.png',
                     'dream_vs_future_results.pkl'],
            extra={
                'n_rollouts': int(N), 'horizon': int(horizon),
                'median_step0_err': step0_err,
                'median_final_err': float(np.median(err[:, -1])),
            })
    except Exception as e:
        print(f"  (run_info logging skipped: {e})")

    logger.close()
    print("Done.")
