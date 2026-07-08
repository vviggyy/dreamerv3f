"""
Dream Decode: Position decoding from imagined (dream) rollouts.

Loads a trained classification position decoder and applies it to
the world model's policy-based imagination outputs. This tests whether
the world model maintains spatial coherence during dreaming.

Usage:
  python dreamerv3/main.py \
    --configs crafter_small size1m \
    --logdir ./logdir/crafter_small_1m \
    --script dream_decode \
    --run.from_checkpoint ./logdir/crafter_small_1m/ckpt/TIMESTAMP_DIR \
    --dream_decode.decoder_model ./logdir/crafter_small_1m/decoder_results/classifier_deter.pkl \
    --dream_decode.save_path ./logdir/crafter_small_1m/dream_results \
    --jax.platform cpu
"""

import pickle
from collections import defaultdict
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

def _get_render_world():
    """Import _render_crafter_world from plot_trajectories."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from plot_trajectories import _render_crafter_world
    return _render_crafter_world


def plot_dream_trajectories_on_world(decoded_pos, start_pos, metadata,
                                      save_dir, tile_size=8):
    """Plot decoded dream trajectories on rendered Crafter world map.

    Args:
        decoded_pos: (N, H, 2) decoded (x, y) positions per dream step
        start_pos: (N, 2) starting positions of dreams
        metadata: dict with env_seed, area, etc.
        save_dir: Path to save output
        tile_size: pixel size per grid cell
    """
    _render_crafter_world = _get_render_world()
    world_img, env_seed, tile_size = _render_crafter_world(metadata, tile_size)
    if world_img is None:
        print("Could not render world for dream trajectories — skipping")
        return
    img_h, img_w = world_img.shape[:2]

    def pos_to_px(p):
        px = p[..., 0] * tile_size + tile_size // 2
        py = img_h - (p[..., 1] * tile_size + tile_size // 2)
        return px, py

    N, H, _ = decoded_pos.shape
    n_show = min(N, 12)
    ncols = min(4, n_show)
    nrows = int(np.ceil(n_show / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows),
                             facecolor='#1a1a1a')
    axes = np.atleast_2d(axes)

    # Compute zoom bounds from all positions
    all_pos = np.concatenate([decoded_pos.reshape(-1, 2), start_pos], axis=0)
    all_px, all_py = pos_to_px(all_pos)
    pad = tile_size * 5
    x_lo = max(0, all_px.min() - pad)
    x_hi = min(img_w, all_px.max() + pad)
    y_lo = max(0, all_py.min() - pad)
    y_hi = min(img_h, all_py.max() + pad)

    cmap = plt.cm.plasma

    for idx, ax in enumerate(axes.flat):
        ax.set_facecolor('#1a1a1a')
        if idx >= n_show:
            ax.axis('off')
            continue

        ax.imshow((world_img * 0.6).astype(np.uint8))

        # Dream trajectory colored by timestep
        traj = decoded_pos[idx]  # (H, 2)
        traj_px, traj_py = pos_to_px(traj)
        colors = cmap(np.linspace(0, 1, H))
        for t in range(H - 1):
            ax.plot([traj_px[t], traj_px[t+1]], [traj_py[t], traj_py[t+1]],
                    '-', color=colors[t], linewidth=2.0, alpha=0.8, zorder=3)

        # Start position
        sp = start_pos[idx]
        spx, spy = pos_to_px(sp[np.newaxis])
        ax.plot(spx, spy, 'o', color='cyan', markersize=10,
                markeredgecolor='white', markeredgewidth=1.5, zorder=5,
                label='start' if idx == 0 else '')

        # End position
        ax.plot(traj_px[-1], traj_py[-1], 's', color='lime', markersize=8,
                markeredgecolor='white', markeredgewidth=1.0, zorder=5,
                label='end' if idx == 0 else '')

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_hi, y_lo)
        ax.set_title(f'Dream {idx+1}', fontsize=9, fontweight='bold',
                     color='white',
                     bbox=dict(facecolor='black', alpha=0.6, pad=2))
        ax.axis('off')
        if idx == 0:
            ax.legend(fontsize=7, loc='upper left',
                      facecolor='black', edgecolor='grey', labelcolor='white')

    fig.suptitle(f'Dream trajectories on world (seed={env_seed})',
                 fontsize=13, color='white')

    # Add colorbar for timestep
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, H - 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02, aspect=30)
    cbar.set_label('Imagination timestep', color='white', fontsize=10)
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.ax.tick_params(labelcolor='white')

    fig.tight_layout()
    out = save_dir / 'dream_trajectories_world.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
    plt.close(fig)
    print(f"  Saved {out.name}")


def plot_dream_probmaps(dream_proba, start_pos, metadata, save_dir,
                         n_dreams=4, n_steps=6, tile_size=8):
    """Per-step probability heatmaps for selected dreams overlaid on world map.

    Args:
        dream_proba: (N, H, W_grid, H_grid) probability maps
        start_pos: (N, 2)
        metadata: dict with env_seed, area
        save_dir: Path
        n_dreams: how many dreams to show
        n_steps: how many timesteps per dream
        tile_size: pixel size per grid cell
    """
    _render_crafter_world = _get_render_world()
    world_img, env_seed, tile_size = _render_crafter_world(metadata, tile_size)
    if world_img is None:
        print("Could not render world for dream probmaps — skipping")
        return
    img_h, img_w = world_img.shape[:2]

    N, H, W_grid, H_grid = dream_proba.shape
    n_dreams = min(n_dreams, N)
    n_steps = min(n_steps, H)
    step_indices = np.linspace(0, H - 1, n_steps, dtype=int)

    cmap = plt.cm.magma

    for di in range(n_dreams):
        fig, axes = plt.subplots(1, n_steps, figsize=(4.0 * n_steps, 4.0),
                                 facecolor='#1a1a1a')
        if n_steps == 1:
            axes = [axes]

        # Zoom bounds from start position + all argmax decoded positions
        sp = start_pos[di]
        # Collect all argmax positions for this dream
        all_cx = [sp[0] * tile_size + tile_size // 2]
        all_cy = [img_h - (sp[1] * tile_size + tile_size // 2)]
        for t in step_indices:
            p_grid_t = dream_proba[di, t]
            dx, dy = np.unravel_index(p_grid_t.argmax(), p_grid_t.shape)
            all_cx.append(dx * tile_size + tile_size // 2)
            all_cy.append(img_h - (dy * tile_size + tile_size // 2))
        pad = tile_size * 5
        x_lo = max(0, min(all_cx) - pad)
        x_hi = min(img_w, max(all_cx) + pad)
        y_lo = max(0, min(all_cy) - pad)
        y_hi = min(img_h, max(all_cy) + pad)

        for si, ax in enumerate(axes):
            ax.set_facecolor('#1a1a1a')
            t = step_indices[si]
            p_grid = dream_proba[di, t]  # (W_grid, H_grid)

            ax.imshow((world_img * 0.6).astype(np.uint8))

            # Upscale probability grid
            p_up = np.repeat(np.repeat(p_grid, tile_size, axis=0),
                             tile_size, axis=1)
            p_up = p_up.T[::-1]
            p_max = p_up.max()
            if p_max > 0:
                p_norm = p_up / p_max
            else:
                p_norm = p_up
            overlay_rgba = cmap(p_norm)
            overlay_rgba[..., 3] = p_norm * 0.85
            ax.imshow(overlay_rgba)

            # Start position marker
            spx = sp[0] * tile_size + tile_size // 2
            spy = img_h - (sp[1] * tile_size + tile_size // 2)
            ax.plot(spx, spy, 'o', color='cyan', markersize=7,
                    markeredgecolor='white', markeredgewidth=1.0, zorder=5)

            # Argmax decoded position
            dec_x, dec_y = np.unravel_index(p_grid.argmax(), p_grid.shape)
            dpx = dec_x * tile_size + tile_size // 2
            dpy = img_h - (dec_y * tile_size + tile_size // 2)
            ax.plot(dpx, dpy, '*', color='lime', markersize=11,
                    markeredgecolor='white', markeredgewidth=0.7, zorder=5)

            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(y_hi, y_lo)
            ax.set_title(f't={t}', fontsize=9, fontweight='bold', color='white',
                         bbox=dict(facecolor='black', alpha=0.6, pad=2))
            ax.axis('off')
            if si == 0:
                ax.plot([], [], 'o', color='cyan', markersize=7,
                        markeredgecolor='white', label='start')
                ax.plot([], [], '*', color='lime', markersize=11,
                        markeredgecolor='white', label='argmax pos')
                ax.legend(fontsize=7, loc='upper left',
                          facecolor='black', edgecolor='grey', labelcolor='white')

        fig.suptitle(f'Dream {di+1} P(pos) (seed={env_seed})',
                     fontsize=12, color='white')
        fig.tight_layout()
        out = save_dir / f'dream_probmap_{di+1}.png'
        fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
        plt.close(fig)
        print(f"  Saved {out.name}")


def plot_dream_vs_real(decoded_pos, start_pos, obs_pos, metadata,
                        save_dir, tile_size=8, real_label='observed',
                        title='Dream vs Real'):
    """Overlay the decoded dream trajectory against a reference trajectory.

    The reference (white line, `obs_pos`) means different things per caller:
      - dream_decode passes the observed PAST (the path into the dream start)
        -> a plausibility check ("does the dream head off sensibly?").
      - dream_vs_future passes the real FUTURE (where the agent actually went)
        -> an accuracy check ("does the dream match reality?").
    `real_label` / `title` let each caller say which one it is.

    Args:
        decoded_pos: (N, H, 2) decoded dream positions
        start_pos: (N, 2)
        obs_pos: (N, T_ref, 2) reference trajectory to overlay (observed past
            for dream_decode, real future for dream_vs_future)
        metadata: dict
        save_dir: Path
        tile_size: pixel size per grid cell
        real_label: legend label for the white reference line
        title: figure suptitle prefix
    """
    _render_crafter_world = _get_render_world()
    world_img, env_seed, tile_size = _render_crafter_world(metadata, tile_size)
    if world_img is None:
        print("Could not render world for dream vs real — skipping")
        return
    img_h, img_w = world_img.shape[:2]

    def pos_to_px(p):
        px = p[..., 0] * tile_size + tile_size // 2
        py = img_h - (p[..., 1] * tile_size + tile_size // 2)
        return px, py

    N = min(decoded_pos.shape[0], 6)
    fig, axes = plt.subplots(1, N, figsize=(5.0 * N, 5.0), facecolor='#1a1a1a')
    if N == 1:
        axes = [axes]

    # Zoom from all positions
    all_pts = []
    for i in range(N):
        all_pts.append(obs_pos[i])
        all_pts.append(decoded_pos[i])
        all_pts.append(start_pos[i:i+1])
    all_pts = np.concatenate(all_pts, axis=0)
    all_px, all_py = pos_to_px(all_pts)
    pad = tile_size * 5
    x_lo = max(0, all_px.min() - pad)
    x_hi = min(img_w, all_px.max() + pad)
    y_lo = max(0, all_py.min() - pad)
    y_hi = min(img_h, all_py.max() + pad)

    for idx, ax in enumerate(axes):
        ax.set_facecolor('#1a1a1a')
        ax.imshow((world_img * 0.6).astype(np.uint8))

        # Reference trajectory (white) — observed past or real future per caller
        rpx, rpy = pos_to_px(obs_pos[idx])
        ax.plot(rpx, rpy, '-', color='white', linewidth=1.5, alpha=0.7,
                zorder=3, label=real_label if idx == 0 else '')

        # Dream trajectory (colored by imagination step)
        dpx, dpy = pos_to_px(decoded_pos[idx])
        colors = plt.cm.plasma(np.linspace(0, 1, decoded_pos.shape[1]))
        for t in range(decoded_pos.shape[1] - 1):
            ax.plot([dpx[t], dpx[t+1]], [dpy[t], dpy[t+1]],
                    '-', color=colors[t], linewidth=2.0, alpha=0.8, zorder=4,
                    label='dream (policy)' if (idx == 0 and t == 0) else '')

        # Start position
        spx, spy = pos_to_px(start_pos[idx:idx+1])
        ax.plot(spx, spy, 'o', color='cyan', markersize=10,
                markeredgecolor='white', markeredgewidth=1.5, zorder=5,
                label='dream start' if idx == 0 else '')

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_hi, y_lo)
        ax.set_title(f'Dream {idx+1}', fontsize=9, fontweight='bold',
                     color='white',
                     bbox=dict(facecolor='black', alpha=0.6, pad=2))
        ax.axis('off')
        if idx == 0:
            ax.legend(fontsize=7, loc='upper left',
                      facecolor='black', edgecolor='grey', labelcolor='white')

    fig.suptitle(f'{title} (seed={env_seed})', fontsize=13, color='white')
    fig.tight_layout()
    out = save_dir / 'dream_vs_real.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
    plt.close(fig)
    print(f"  Saved {out.name}")


# ---------------------------------------------------------------------------
# Main dream decode function
# ---------------------------------------------------------------------------

def dream_decode(make_agent, make_env, make_replay, make_stream,
                 make_logger, args):
    """Run position decoding on imagined (dream) rollouts."""
    dd_config = args.dream_decode
    assert dd_config.decoder_model, "Must provide --dream_decode.decoder_model"

    save_dir = Path(dd_config.save_path or str(
        elements.Path(args.logdir) / 'dream_results'))
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create agent and load checkpoint
    print("Creating agent...")
    agent = make_agent()
    logger = make_logger()

    # Load model params only, skipping optimizer state (opt/). A run trained
    # with extra regularization (weight_decay / l1_wm) has a different optax
    # chain, so its saved opt/ tree does not match a freshly built agent and
    # would trip chex's tree-structure assert. Dream inference never uses opt/.
    # (Mirrors eval_trajectory.py's regex-skip load.)
    import pickle as _pickle, pathlib as _pl
    _ckpt_file = _pl.Path(args.from_checkpoint) / 'agent.pkl'
    with open(_ckpt_file, 'rb') as _f:
      agent.load(_pickle.load(_f), regex=r'^(?!opt/)')
    print(f"Loaded checkpoint (params only, skipping opt/): {args.from_checkpoint}")

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
    if env_seed is not None:
        world_seed = hash((env_seed, 1))
    else:
        world_seed = None

    metadata = {
        'env_seed': env_seed,
        'world_seed': world_seed,
        'fixed_seed': True,
        'task': 'crafter',
        'area': area,
    }

    # Create replay and populate with eval episodes
    replay = make_replay()
    num_episodes = dd_config.num_episodes
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
    print("Running report() to get dream rollouts...")
    stream = make_stream(replay, 'report')
    stream = agent.stream(stream)
    stream = iter(stream)

    carry = agent.init_report(args.batch_size)
    num_batches = dd_config.num_batches

    all_dream_deter = []
    all_dream_stoch = []
    all_start_pos = []
    all_obs_pos = []

    for batch_idx in range(num_batches):
        batch = next(stream)
        carry, mets = agent.report(carry, batch)

        if 'dream/deter' in mets:
            all_dream_deter.append(np.array(mets['dream/deter']))
        if 'dream/stoch' in mets:
            all_dream_stoch.append(np.array(mets['dream/stoch']))
        if 'dream/start_pos' in mets:
            all_start_pos.append(np.array(mets['dream/start_pos']))
        if 'dream/obs_pos' in mets:
            all_obs_pos.append(np.array(mets['dream/obs_pos']))

        print(f"  Batch {batch_idx+1}/{num_batches} done")

    if not all_dream_deter:
        print("ERROR: No dream/deter in report metrics. "
              "Check agent.py report() modification.")
        return

    dream_deter = np.concatenate(all_dream_deter, axis=0)  # (N, H, D)
    dream_stoch = np.concatenate(all_dream_stoch, axis=0) if all_dream_stoch else None
    start_pos = np.concatenate(all_start_pos, axis=0) if all_start_pos else None
    obs_pos = np.concatenate(all_obs_pos, axis=0) if all_obs_pos else None

    N, H, D = dream_deter.shape
    print(f"Dream data: {N} rollouts, {H} steps, deter_dim={D}")
    if start_pos is not None:
        print(f"  Start positions available: {start_pos.shape}")
    if obs_pos is not None:
        print(f"  Observed positions available: {obs_pos.shape}")

    # 4. Load decoder and decode positions
    print(f"\nLoading decoder from {dd_config.decoder_model}...")
    decoder_path = Path(dd_config.decoder_model)

    from .decode_position import load_classifier_model
    clf, dec_meta = load_classifier_model(decoder_path)
    width = dec_meta['width']
    height = dec_meta['height']
    print(f"  Classifier decoder: {dec_meta.get('repr_name', '?')}, "
          f"grid={width}x{height}")

    flat_deter = dream_deter.reshape(-1, D)
    proba = clf.predict_proba(flat_deter)  # (N*H, W*H_grid)
    dream_proba = proba.reshape(N, H, width, height)

    # Argmax positions
    pred_cls = proba.argmax(axis=1)
    pred_xy = np.stack(np.unravel_index(pred_cls, (width, height)), axis=1)
    decoded_pos = pred_xy.reshape(N, H, 2).astype(float)
    print(f"  Decoded grid positions range: "
          f"x=[{decoded_pos[:,:,0].min():.0f}, {decoded_pos[:,:,0].max():.0f}], "
          f"y=[{decoded_pos[:,:,1].min():.0f}, {decoded_pos[:,:,1].max():.0f}]")

    # Compute dream continuity: median Euclidean distance between consecutive steps
    step_dists = np.linalg.norm(np.diff(decoded_pos, axis=1), axis=-1)  # (N, H-1)
    median_step_dist = float(np.median(step_dists))
    mean_step_dist = float(np.mean(step_dists))
    print(f"  Dream continuity — median step dist: {median_step_dist:.2f} tiles, "
          f"mean: {mean_step_dist:.2f} tiles")

    # 5. Plot
    print("\nGenerating plots...")

    if start_pos is not None:
        plot_dream_trajectories_on_world(
            decoded_pos, start_pos, metadata, save_dir)

        if dream_proba is not None:
            plot_dream_probmaps(
                dream_proba, start_pos, metadata, save_dir)

        if obs_pos is not None:
            plot_dream_vs_real(
                decoded_pos, start_pos, obs_pos, metadata, save_dir,
                real_label='observed past (context)',
                title='Dream vs observed past — plausibility')
    else:
        print("  No start positions available — skipping world overlay plots")

    # 6. Save results
    results = {
        'decoded_pos': decoded_pos,
        'dream_deter_shape': dream_deter.shape,
        'start_pos': start_pos,
        'obs_pos': obs_pos,
        'decoder_path': str(decoder_path),
        'decoder_metadata': dec_meta,
        'metadata': metadata,
        'num_batches': num_batches,
        'num_episodes': num_episodes,
        'step_dists': step_dists,
        'median_step_dist': median_step_dist,
        'mean_step_dist': mean_step_dist,
    }
    if dream_proba is not None:
        results['dream_proba'] = dream_proba

    # Save raw dream activations for downstream manifold analysis
    if dd_config.save_activations:
        results['dream_deter'] = dream_deter  # (N, H, D)
        if dream_stoch is not None:
            results['dream_stoch'] = dream_stoch  # (N, H, ...)
        print(f"  Saved raw dream activations (save_activations=True)")

    results_file = save_dir / 'dream_results.pkl'
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {results_file}")

    logger.close()
    print("Done.")
