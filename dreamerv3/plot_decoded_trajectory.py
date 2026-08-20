"""Plot a real agent trajectory vs its decoded trajectory on the world map.

Loads a saved position classifier (decode_position.py --save_model) and an eval
trajectory episode, decodes each timestep's `deter` -> grid (x, y), and renders
three panels on the Crafter world:
  1. Real trajectory   (colored by time)
  2. Decoded trajectory (colored by time)
  3. Overlay: real (solid) + decoded (dashed) + faint per-step error connectors

Usage:
  MPLBACKEND=Agg python dreamerv3/plot_decoded_trajectory.py \
      --data logdir/masked_k5_invert_reg_wrld45/trajectories \
      --decoder logdir/masked_k5_invert_reg_wrld45/decoder_results/classifier_deter.pkl \
      --episode 10 \
      --save logdir/masked_k5_invert_reg_wrld45/plots/decoded_trajectory_ep10.svg
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decode_position import load_classifier_model  # noqa: E402
from render_poster_frame import render_world, world_to_img  # noqa: E402


def decode_episode(ep, clf, meta):
    """Return (real_xy (T,2), decoded_xy (T,2), mean_manhattan)."""
    width, height = meta['width'], meta['height']
    X = np.asarray(ep['deter'], dtype=np.float32)
    pred_cls = clf.predict(X)
    dec = np.stack(np.unravel_index(pred_cls, (width, height)), axis=1)  # (T,2)
    real = np.asarray(ep['player_pos'], dtype=float)[:len(dec)]
    err = np.abs(dec - real.astype(int)).sum(axis=1).mean()
    return real, dec.astype(float), err


def _time_colored_path(ax, xy, tile_size, cmap, lw, ls='-', alpha=1.0, label=None):
    """Draw a path colored by timestep as a LineCollection."""
    pts = np.array([world_to_img(x, y, tile_size) for x, y in xy])  # (T,2)=(col,row)
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    t = np.linspace(0, 1, len(segs))
    lc = LineCollection(segs, cmap=cmap, array=t, linewidths=lw,
                        linestyles=ls, alpha=alpha)
    ax.add_collection(lc)
    if label:
        ax.plot([], [], color=plt.get_cmap(cmap)(0.6), lw=lw, ls=ls, label=label)
    # start / end markers
    ax.plot(*pts[0], 'o', color='lime', ms=7, mec='black', mew=1, zorder=11)
    ax.plot(*pts[-1], 's', color='red', ms=7, mec='black', mew=1, zorder=11)
    return lc


def main():
    ap = argparse.ArgumentParser(description='Real vs decoded trajectory on the world')
    ap.add_argument('--data', required=True, help='trajectory dir (episode_*.pkl)')
    ap.add_argument('--decoder', required=True, help='classifier_*.pkl (from --save_model)')
    ap.add_argument('--episode', type=int, default=1, help='episode number (1-indexed)')
    ap.add_argument('--save', default=None, help='output path (svg/pdf/png)')
    ap.add_argument('--seed', type=int, default=None, help='config seed (auto from config.yaml)')
    ap.add_argument('--area', type=int, nargs=2, default=None, help='world area (auto)')
    ap.add_argument('--tile_size', type=int, default=12, help='px per tile for world render')
    ap.add_argument('--subsample', type=int, default=1, help='keep every Nth step (thin dense paths)')
    ap.add_argument('--drop_first', type=int, default=1,
                    help='drop the first N steps (t=0 deter is the uninformative '
                         'reset state and decodes to a fixed corner). Default 1.')
    args = ap.parse_args()

    data = Path(args.data)
    ep_file = data / f'episode_{args.episode:03d}.pkl'
    ep = pickle.load(open(ep_file, 'rb'))
    clf, meta = load_classifier_model(args.decoder)
    print(f"Loaded {ep_file} ({ep['length']} steps); decoder grid={meta['width']}x{meta['height']}")

    # Resolve seed/area (config.yaml can be clobbered by eval; fall back to defaults)
    seed, area = args.seed, tuple(args.area) if args.area else None
    if seed is None or area is None:
        cfg_path = data.parent / 'config.yaml'
        if cfg_path.exists():
            from ruamel.yaml import YAML
            cfg = YAML().load(open(cfg_path))
            if seed is None:
                seed = cfg.get('seed', 0)
            if area is None:
                area = tuple(cfg.get('env', {}).get('crafter', {}).get('area', [32, 32]))
    seed = seed if seed is not None else 0
    area = area or (32, 32)
    env_seed = hash((seed, 0)) % (2**32 - 1)
    print(f"seed={seed} -> env_seed={env_seed}, area={area}")

    real, dec, _ = decode_episode(ep, clf, meta)
    if args.drop_first > 0:
        real, dec = real[args.drop_first:], dec[args.drop_first:]
    if args.subsample > 1:
        real, dec = real[::args.subsample], dec[::args.subsample]
    err = np.abs(dec.astype(int) - real.astype(int)).sum(axis=1).mean()
    print(f"episode {args.episode}: mean Manhattan error = {err:.2f} tiles "
          f"({len(real)} steps, dropped first {args.drop_first})")

    ts = args.tile_size
    world = render_world(env_seed, area=area, tile_size=ts)

    fig, axes = plt.subplots(1, 3, figsize=(21, 7), constrained_layout=True)
    titles = ['Real trajectory', 'Decoded trajectory',
              f'Overlay (mean err = {err:.1f} tiles)']
    for ax, title in zip(axes, titles):
        ax.imshow(world, origin='upper', interpolation='nearest')
        ax.axis('off')
        ax.set_title(title, fontsize=13)

    # Panel 1: real (viridis by time)
    _time_colored_path(axes[0], real, ts, 'viridis', lw=2.5)
    # Panel 2: decoded (plasma by time)
    _time_colored_path(axes[1], dec, ts, 'plasma', lw=2.5)
    # Panel 3: overlay — real cyan solid, decoded magenta dashed, error connectors
    rp = np.array([world_to_img(x, y, ts) for x, y in real])
    dp = np.array([world_to_img(x, y, ts) for x, y in dec])
    err_segs = np.stack([rp, dp], axis=1)
    axes[2].add_collection(LineCollection(err_segs, colors='white', linewidths=0.5, alpha=0.35))
    axes[2].plot(rp[:, 0], rp[:, 1], '-', color='cyan', lw=2.2, alpha=0.9, label='real')
    axes[2].plot(dp[:, 0], dp[:, 1], '--', color='magenta', lw=1.8, alpha=0.9, label='decoded')
    axes[2].plot(rp[0, 0], rp[0, 1], 'o', color='lime', ms=7, mec='black', zorder=11)
    axes[2].plot(rp[-1, 0], rp[-1, 1], 's', color='red', ms=7, mec='black', zorder=11)
    handles = [
        Line2D([0], [0], color='cyan', lw=2.2, label='real'),
        Line2D([0], [0], color='magenta', lw=1.8, ls='--', label='decoded'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='lime',
               mec='black', ms=8, label='start (t=%d)' % args.drop_first),
        Line2D([0], [0], marker='s', color='none', markerfacecolor='red',
               mec='black', ms=8, label='end'),
    ]
    axes[2].legend(handles=handles, loc='upper right', fontsize=9, framealpha=0.85)

    fig.suptitle(f'{data.parent.name} — episode {args.episode} '
                 f'(deter decoder, {meta["width"]}x{meta["height"]} grid)',
                 fontsize=12, y=1.02)
    out = args.save or str(data.parent / f'decoded_trajectory_ep{args.episode}.svg')
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches='tight')
    print(f"Saved {out}")


if __name__ == '__main__':
    main()
