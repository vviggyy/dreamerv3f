"""Localize the vital-bar pixels in the rendered Crafter observation.

STEP 1 of the "does the world model represent the vital-bar pixels better than
other pixels?" experiment (companion to decode_vitals.py). Before we can train a
decoder deter -> {vital pixel values} and compare its error against other pixels,
we have to know WHICH pixels belong to each vital's status display and HOW MANY.

Approach (empirical, no crafter-layout arithmetic): Crafter draws the bottom
status bar from `env._env._player.inventory` (ints) and copies it into the
observation (crafter.py:_render_egocentric -> result[-inv_rows:] = raw[-inv_rows:]).
So for each vital we mutate ONLY that vital, re-render, and take the set of pixels
that CHANGE across its levels -- those are exactly the pixels that encode the
vital's value (digit/pips). We also do this for the other inventory icons
(wood/stone/coal/... ) to get a within-bar control set of comparable pixels.

This needs NO trained agent, NO checkpoint, NO JAX -- only the crafter env, built
from a training run's saved config.yaml so the image is pixel-identical to what
the model was trained on (egocentric_view, size, etc. all inherited).

Outputs (to --save):
  vital_pixel_masks.pkl   -- per-vital + per-item boolean masks (H,W), counts,
                             tile bounding boxes, the bar row-range, img shape
  vital_pixel_localization.svg -- sample frame + zoomed bar with masks overlaid

Usage:
  python dreamerv3/localize_vital_pixels.py \
      --logdir ./logdir/masked_k5_invert_reg_wrld45 \
      --save   ./logdir/masked_k5_invert_reg_wrld45/vital_pixels \
      --num_frames 8 --warmup 15
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

VITALS = ['health', 'food', 'drink', 'energy']
VITAL_LEVELS = list(range(10))                 # vitals are ints 0..9
# Other status-bar icons (materials + tools). Swept over a few nonzero counts to
# reveal their footprint; items not displayed never change and are auto-dropped.
INV_ITEMS = ['sapling', 'wood', 'stone', 'coal', 'iron', 'diamond',
             'wood_pickaxe', 'stone_pickaxe', 'iron_pickaxe',
             'wood_sword', 'stone_sword', 'iron_sword']
ITEM_LEVELS = [0, 1, 4, 9]
VITAL_COLORS = {'health': '#d62728', 'food': '#2ca02c',
                'drink': '#1f77b4', 'energy': '#ff7f0e'}


# ---------------------------------------------------------------------------
# Env (standalone, from a run's config.yaml)
# ---------------------------------------------------------------------------
def build_env(logdir):
    """Instantiate the bare Crafter wrapper with the run's saved env settings."""
    import ruamel.yaml as yaml
    cfg_path = Path(logdir) / 'config.yaml'
    if not cfg_path.exists():
        raise SystemExit(f"No config.yaml at {cfg_path}")
    with open(cfg_path) as f:
        cfg = yaml.YAML(typ='safe').load(f)
    suite, task = cfg['task'].split('_', 1)
    if suite != 'crafter':
        raise SystemExit(f"task is {cfg['task']!r}; this tool is crafter-only")
    ck = dict(cfg['env']['crafter'])
    use_seed = ck.pop('use_seed', False)
    ck.pop('use_logdir', None)
    for k in ('size', 'area'):
        if isinstance(ck.get(k), list):
            ck[k] = tuple(ck[k])
    seed = hash((cfg.get('seed', 0), 0)) % (2 ** 32 - 1) if use_seed else None
    from embodied.envs.crafter import Crafter
    print(f"Building Crafter(task={task!r}) with: egocentric_view="
          f"{ck.get('egocentric_view')}, size={ck.get('size')}, "
          f"area={ck.get('area')}, fixed_seed={ck.get('fixed_seed')}")
    return Crafter(task, seed=seed, **ck)


def render_with(env, overrides):
    """Re-render the current scene with inventory entries overwritten (and then
    restored). Returns a uint8 (H,W,3) image -- the exact observation image."""
    player = env._env._player
    saved = {k: player.inventory.get(k) for k in overrides}
    for k, v in overrides.items():
        player.inventory[k] = int(v)
    raw = env._env._obs()
    ego = getattr(env, '_egocentric_view', None)
    img = env._render_egocentric(raw) if ego else raw
    for k, v in saved.items():                 # restore exactly
        player.inventory[k] = v
    return np.asarray(img, np.uint8)


def capture_frames(env, num_frames, warmup):
    """Drive random actions, capturing the live player state at each frame.

    We keep the env itself (so the live inventory/scene is current when we probe)
    rather than snapshots: localization re-renders in place at probe time."""
    n_act = env.act_space['action'].high
    rng = np.random.RandomState(0)
    env.step({'action': np.int32(0), 'reset': np.bool_(True)})
    frames = []       # list of base images (for visualization only)
    probe_states = 0
    step = 0
    max_steps = num_frames * max(warmup, 1) * 5
    while probe_states < num_frames and step < max_steps:
        if step % max(warmup, 1) == 0:
            # record current base image + probe all vitals/items HERE while the
            # live scene is current.
            frames.append(_probe_current(env))
            probe_states += 1
            print(f"  probed frame {probe_states}/{num_frames} (step {step})")
        a = int(rng.randint(0, n_act))
        obs = env.step({'action': np.int32(a), 'reset': np.bool_(False)})
        if bool(np.asarray(obs['is_last'])):
            env.step({'action': np.int32(0), 'reset': np.bool_(True)})
        step += 1
    return frames


def _probe_current(env):
    """At the CURRENT live scene, render across each vital's/item's levels and
    return the base image plus per-key stacks (used to compute change masks)."""
    base = render_with(env, {})
    rec = {'base': base, 'vital_stacks': {}, 'item_stacks': {}}
    for v in VITALS:
        rec['vital_stacks'][v] = np.stack(
            [render_with(env, {v: lv}) for lv in VITAL_LEVELS])   # (L,H,W,3)
    for it in INV_ITEMS:
        rec['item_stacks'][it] = np.stack(
            [render_with(env, {it: lv}) for lv in ITEM_LEVELS])
    return rec


# ---------------------------------------------------------------------------
# Change masks
# ---------------------------------------------------------------------------
def change_mask(stack, thresh=0):
    """(L,H,W,3) -> (H,W) bool: pixel varies across the L levels (any channel).

    thresh=0 means strictly any change; raise it to ignore 1-2 level dithering."""
    span = stack.max(0).astype(np.int16) - stack.min(0).astype(np.int16)  # (H,W,3)
    return (span.max(-1) > thresh)


def union_over_frames(frames, group, key, thresh=0):
    m = None
    for rec in frames:
        cm = change_mask(rec[group][key], thresh=thresh)
        m = cm if m is None else (m | cm)
    return m


def bbox(mask):
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()))


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_localization(frames, masks, bar_row0, save_dir):
    base = frames[0]['base']
    H, W = base.shape[:2]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))

    # (a) full frame
    axes[0].imshow(base)
    axes[0].axhline(bar_row0 - 0.5, color='yellow', lw=1.0, ls='--')
    axes[0].set_title(f'sample frame ({H}x{W})\nbar = rows {bar_row0}..{H-1}')

    # (b) full frame with vital masks overlaid (colored)
    over = base.copy()
    for v in VITALS:
        m = masks['vitals'][v]
        if m is not None and m.any():
            col = np.array(_hex(VITAL_COLORS[v]))
            over[m] = (0.35 * over[m] + 0.65 * col).astype(np.uint8)
    axes[1].imshow(over)
    axes[1].set_title('vital pixels (change across levels 0-9)')

    # (c) zoom on the bar region, item masks (gray) + vital masks (colored)
    bar = base[bar_row0:].copy().astype(np.float32)
    for it in masks['items']:
        m = masks['items'][it]
        if m is not None:
            mb = m[bar_row0:]
            bar[mb] = 0.5 * bar[mb] + 0.5 * np.array([180, 180, 180])
    for v in VITALS:
        m = masks['vitals'][v]
        if m is not None:
            mb = m[bar_row0:]
            bar[mb] = 0.3 * bar[mb] + 0.7 * np.array(_hex(VITAL_COLORS[v]))
    axes[2].imshow(bar.astype(np.uint8), interpolation='nearest')
    axes[2].set_title('bar zoom: vitals=color, other items=gray')

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    # legend
    handles = [plt.Line2D([0], [0], marker='s', ls='', mfc=VITAL_COLORS[v],
                          mec='k', ms=10,
                          label=f"{v}: {int(masks['vitals'][v].sum()) if masks['vitals'][v] is not None else 0} px")
               for v in VITALS]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False)
    fig.suptitle('Vital-bar pixel localization', fontsize=14)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = Path(save_dir) / 'vital_pixel_localization.svg'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")


def _hex(h):
    h = h.lstrip('#')
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--logdir', required=True, help='run dir with config.yaml')
    ap.add_argument('--save', default='', help='output dir (default: logdir/vital_pixels)')
    ap.add_argument('--num_frames', type=int, default=8)
    ap.add_argument('--warmup', type=int, default=15, help='steps between probes')
    ap.add_argument('--thresh', type=int, default=0,
                    help='min per-channel span (0-255) to count a pixel as changing')
    args = ap.parse_args()

    save_dir = Path(args.save or (Path(args.logdir) / 'vital_pixels'))
    save_dir.mkdir(parents=True, exist_ok=True)

    env = build_env(args.logdir)
    print(f"Capturing {args.num_frames} frames (warmup {args.warmup})...")
    frames = capture_frames(env, args.num_frames, args.warmup)
    env.close()
    if not frames:
        raise SystemExit("No frames captured")

    H, W = frames[0]['base'].shape[:2]
    masks = {'vitals': {}, 'items': {}}
    for v in VITALS:
        masks['vitals'][v] = union_over_frames(frames, 'vital_stacks', v, args.thresh)
    for it in INV_ITEMS:
        m = union_over_frames(frames, 'item_stacks', it, args.thresh)
        masks['items'][it] = m if (m is not None and m.any()) else None

    # bar row start: topmost row touched by ANY bar key (vitals + items)
    all_bar = np.zeros((H, W), bool)
    for grp in masks.values():
        for m in grp.values():
            if m is not None:
                all_bar |= m
    bar_row0 = int(np.where(all_bar.any(1))[0].min()) if all_bar.any() else H

    # report
    print(f"\nImage {H}x{W}.  Status bar starts at row {bar_row0} "
          f"({H - bar_row0} rows).\n")
    print("=== vital pixel footprints (pixels that change across levels 0-9) ===")
    for v in VITALS:
        m = masks['vitals'][v]
        n = int(m.sum()) if m is not None else 0
        bb = bbox(m) if m is not None else None
        print(f"  {v:7s}: {n:4d} px   bbox(y0,y1,x0,x1)={bb}")
    nz = [int(masks['vitals'][v].sum()) for v in VITALS if masks['vitals'][v] is not None]
    if nz:
        print(f"  -> per-vital px: min {min(nz)}, max {max(nz)}, "
              f"median {int(np.median(nz))}  (use this as N)")
    print("\n=== other bar-item footprints (within-bar control) ===")
    shown = [it for it in INV_ITEMS if masks['items'][it] is not None]
    for it in shown:
        m = masks['items'][it]
        print(f"  {it:14s}: {int(m.sum()):4d} px   bbox={bbox(m)}")
    if not shown:
        print("  (no other items displayed in these frames; drive longer or "
              "collect items first)")

    out = {
        'img_shape': (H, W),
        'bar_row0': bar_row0,
        'vital_masks': {v: masks['vitals'][v] for v in VITALS},
        'item_masks': {it: masks['items'][it] for it in shown},
        'vital_levels': VITAL_LEVELS,
        'item_levels': ITEM_LEVELS,
        'vital_counts': {v: (int(masks['vitals'][v].sum())
                             if masks['vitals'][v] is not None else 0)
                         for v in VITALS},
        'item_counts': {it: int(masks['items'][it].sum()) for it in shown},
        'args': vars(args),
    }
    pkl = save_dir / 'vital_pixel_masks.pkl'
    with open(pkl, 'wb') as f:
        pickle.dump(out, f)
    print(f"\nSaved {pkl}")

    plot_localization(frames, masks, bar_row0, save_dir)

    try:
        from run_info import log_run_info
        log_run_info(str(save_dir), 'localize_vital_pixels', vars(args),
                     [str(pkl)], extra={'bar_row0': bar_row0,
                                        'vital_counts': out['vital_counts']})
    except Exception as e:
        print(f"(run_info not logged: {e})")


if __name__ == '__main__':
    main()
