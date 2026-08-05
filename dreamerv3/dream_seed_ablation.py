"""
Dream seeding ablation: how does what you seed a dream with shape the rollout?

Four conditions form a clean crossed 2x2 over the seed carry {deter, stoch}:

               image (posterior stoch)     no image (prior stoch)
  real deter      A. both (current)          B. just-latent
  noise deter     C. just-image              D. neither

  - deter is either the real posterior deter or a noise deter. A/B share the
    real deter; C/D share the SAME noise draw. The noise null is set by
    --dream_seed_ablation.noise_mode: matched (mu+sd*N, default; can go negative,
    off the relu manifold), matched_relu (clipped >=0), truncnorm (N(mu,sd)>=0),
    or shuffle (per-dim bootstrap of real deter — exact marginal + non-negative,
    correlations destroyed). Requires a fresh run (baked into the rollout).
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
(Manhattan tiles). Plots (all conditions; real trajectory on A, B and D):
  A) displacement from each condition's own start (every curve starts at 0)
  B) distance from the real start position
  C) cross-seed variability within each seed (no real curve)
  D) step-to-step displacement (continuity): decoded Manhattan distance from the
     previous tile at each dream step. Low + flat = spatially continuous dream;
     large early values = teleporting/incoherent decode. Best read with
     central=mean (median collapses to 0 since most steps move 0-1 tiles).
  E) P[dx=0] (stay probability): fraction of rollouts whose decoded position did
     not change from the previous step (agent acted but stayed on the same tile).
     Always mean-aggregated (a probability), independent of --central.
  F) E[dx | dx>0] (jump size when moving): mean decoded displacement over the
     steps that DID move (dx==0 steps excluded). Always mean-aggregated.
     D factorizes as E[dx] = P[dx>0] * E[dx | dx>0]; E and F split those apart.
  G) deviation from the REAL trajectory by warmup: per-step |decoded_dream_t -
     real_t| (tracking-fidelity, cf. dream_vs_future — distinct from panel B's
     distance from the real *start*), grouped into DEFAULT_N_WARMUP_BINS (=5)
     equal-count warmup bins derived from the actual warmups (covers the full
     range on any horizon). A/B are drawn per warmup bin (color = warmup group
     light->dark, linestyle A '-' / B ':'); C/D are drawn as a single POOLED
     gray line each (C '--', D '-.'), since their warmup axis is meaningless
     (noise deter is warmup-independent and the decoder reads position from the
     deter, so C/D start ~4 tiles off regardless of W) — flat null baselines.
  The 7 panels are laid out in 2 rows.
  + error_vs_warmup: final-step distance-from-real-start as a function of warmup
  + decoded_dreams_by_condition: 4xN grid on the world map (rows = conditions
    A/B/C/D, columns = distinct example dreams / seed frames), all seeds drawn
    per cell, colored by dream step; columns share start/warmup across rows
  + decoded_dreams_animation.gif: 1x4 conditions for ONE example (start,warmup),
    stepping through seeds one dream at a time over the real trajectory
    (uncluttered counterpart to the all-seeds grid)

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
import warnings
from functools import partial as bind
from pathlib import Path

import elements
import embodied
import matplotlib
import matplotlib.patheffects as pe
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

# Linestyle per condition for the warmup-grouped deviation plot (color there
# encodes the warmup group, so the condition must be carried by linestyle).
COND_LS = {'A': '-', 'B': ':', 'C': '--', 'D': '-.'}

# Panel G groups warmups into this many equal-count bins by default. Bins are
# derived from the ACTUAL warmups present (see _auto_warmup_bins), so they cover
# the full [1, report_length - horizon] range on any horizon — no warmup is ever
# dropped, unlike a fixed range list.
DEFAULT_N_WARMUP_BINS = 5


def _auto_warmup_bins(warmups, n_bins=DEFAULT_N_WARMUP_BINS):
    """Split the observed warmups into <= n_bins equal-count (quantile) groups.

    Returns a list of (lo, hi, label) inclusive ranges over the distinct warmup
    values, so every bin is guaranteed non-empty and the whole range is covered
    regardless of horizon. Labels read 'w{lo}' or 'w{lo}-{hi}'."""
    uniq = np.unique(np.asarray(warmups))
    bins = []
    for ch in np.array_split(uniq, min(n_bins, len(uniq))):
        if not len(ch):
            continue
        lo, hi = int(ch[0]), int(ch[-1])
        lab = f'w{lo}' if lo == hi else f'w{lo}-{hi}'
        bins.append((lo, hi, lab))
    return bins


def _manhattan(a, b):
    return np.abs(a - b).sum(axis=-1)


def _render_world(metadata, tile_size=8):
    """Reconstruct the Crafter world map from metadata (env_seed/area).

    Returns (world_img, env_seed, tile_size) or (None, None, tile_size)."""
    try:
        import sys
        from pathlib import Path as _P
        sys.path.insert(0, str(_P(__file__).parent))
        from plot_trajectories import _render_crafter_world
        return _render_crafter_world(metadata, tile_size)
    except Exception as e:
        print(f"  (world render unavailable: {e})")
        return None, None, tile_size


def _select_columns(S, Nw, ncol):
    """Deterministic (start_row, warmup) picks spread across starts & warmups.

    Columns vary warmup where possible (informative when Nw > 1) and start row,
    so each column is a distinct seed frame / dream."""
    if ncol <= 0:
        return []
    ws = (np.linspace(0, Nw - 1, ncol).round().astype(int)
          if Nw > 1 else np.zeros(ncol, int))
    ss = np.linspace(0, S - 1, ncol).round().astype(int)
    return list(zip(ss.tolist(), ws.tolist()))


def _to_px_fn(world_img, tile_size):
    """Return a (x, y)-tile -> pixel mapper for the rendered world (or identity
    in tile coords when no world image is available)."""
    if world_img is not None:
        img_h = world_img.shape[0]
        def to_px(p):
            px = p[..., 0] * tile_size + tile_size // 2
            py = img_h - (p[..., 1] * tile_size + tile_size // 2)
            return np.stack([px, py], axis=-1)
    else:
        def to_px(p):
            return p.astype(float)
    return to_px


def _draw_real_traj(ax, start_xy2, future_xy, to_px, style='ghost'):
    """Real trajectory + cyan start marker. style:
      'bold'  — thick cyan on top (can occlude the dream when they track closely),
      'ghost' — thin dashed cyan *under* the dream (default; dream stays visible),
      'none'  — start marker only, no line."""
    real_path = np.concatenate([start_xy2[None], future_xy])
    fp = to_px(real_path)
    if style == 'bold':
        ax.plot(fp[:, 0], fp[:, 1], '-', color='#00e5ff', linewidth=2.6,
                alpha=0.95, zorder=4,
                path_effects=[pe.Stroke(linewidth=4.6, foreground='black'),
                              pe.Normal()])
    elif style == 'ghost':
        ax.plot(fp[:, 0], fp[:, 1], '--', color='#00e5ff', linewidth=1.4,
                alpha=0.6, zorder=2)
    ax.plot(fp[0, 0], fp[0, 1], 'o', color='#00e5ff', markersize=8,
            markeredgecolor='white', markeredgewidth=1.2, zorder=6)


def _marker_handles(cmap):
    """Legend proxies: real start (cyan), decoded start (plasma-0), decoded end
    (plasma-max)."""
    from matplotlib.lines import Line2D
    mk = lambda m, fc, lab: Line2D([0], [0], marker=m, linestyle='none',
                                   markerfacecolor=fc, markeredgecolor='white',
                                   markersize=9, label=lab)
    ring = Line2D([0], [0], marker='o', linestyle='none', markerfacecolor='none',
                  markeredgecolor='magenta', markeredgewidth=2.2, markersize=12,
                  label='decoded start')  # hollow ring (sits over real start for A/B)
    return [mk('o', '#00e5ff', 'real start'), ring, mk('s', cmap(1.0), 'decoded end')]


def _draw_decoded_start(ax, xy2, to_px):
    """Decoded seed position as a hollow ring on top — visible even when it lands
    exactly on the cyan real-start marker (the A/B case)."""
    p = to_px(xy2[None])[0]
    ax.plot(p[0], p[1], 'o', markerfacecolor='none', markeredgecolor='magenta',
            markersize=13, markeredgewidth=2.2, zorder=7,
            path_effects=[pe.Stroke(linewidth=3.6, foreground='black'), pe.Normal()])


def plot_decoded_dreams(decoded, start_pos, future_pos, metadata, save_dir,
                        n_cols=4, warmups=None, real_style='ghost'):
    """4xN grid of decoded dream trajectories: rows = conditions (A/B/C/D),
    columns = distinct example dreams (each a different start/warmup seed frame).

    Every cell draws all seeds for that (condition, start, warmup), colored by
    dream step. Columns share their (start, warmup) across rows so conditions
    are directly comparable within a column. The real trajectory (shared across
    conditions) is drawn as a bold cyan line starting at the cyan start marker.
    World-map overlay when available, else tile coords.
    """
    S, Nw, Ns, Hp1, _ = decoded['A'].shape
    cols = _select_columns(S, Nw, n_cols)
    if not cols:
        return
    world_img, env_seed, tile_size = _render_world(metadata)
    to_px = _to_px_fn(world_img, tile_size)

    # Shared zoom bounds over every position that will be drawn.
    dream_xy = np.stack([decoded[c][s, w] for c in CONDITIONS
                         for (s, w) in cols], axis=0)          # (C*ncol, Ns, Hp1, 2)
    real_xy = np.stack([future_pos[s, w] for (s, w) in cols], axis=0)
    start_xy = np.stack([start_pos[s, w] for (s, w) in cols], axis=0)
    allp = to_px(np.concatenate(
        [dream_xy.reshape(-1, 2), real_xy.reshape(-1, 2), start_xy], axis=0))
    pad = (tile_size * 4) if world_img is not None else 2
    xlo, xhi = allp[:, 0].min() - pad, allp[:, 0].max() + pad
    ylo, yhi = allp[:, 1].min() - pad, allp[:, 1].max() + pad

    cmap = plt.cm.plasma
    fc = '#1a1a1a' if world_img is not None else 'white'
    txt = 'white' if world_img is not None else 'black'
    nc = len(cols)

    fig, axes = plt.subplots(4, nc, figsize=(4.2 * nc, 4.4 * 4), facecolor=fc,
                             squeeze=False)
    for ri, c in enumerate(CONDITIONS):
        for ci, (s, w) in enumerate(cols):
            ax = axes[ri, ci]
            ax.set_facecolor(fc)
            if world_img is not None:
                ax.imshow((world_img * 0.6).astype(np.uint8))

            for k in range(Ns):
                tp = to_px(decoded[c][s, w, k])                # (Hp1, 2)
                for t in range(Hp1 - 1):
                    ax.plot(tp[t:t + 2, 0], tp[t:t + 2, 1], '-',
                            color=cmap(t / max(Hp1 - 1, 1)), linewidth=1.4,
                            alpha=0.55, zorder=3)

            # Real trajectory (ground truth): style set by real_style.
            _draw_real_traj(ax, start_pos[s, w], future_pos[s, w], to_px, real_style)
            # Decoded start (step 0, shared across seeds) as a magenta ring on top.
            _draw_decoded_start(ax, decoded[c][s, w, 0, 0], to_px)

            ax.set_xlim(xlo, xhi)
            ax.set_ylim(yhi, ylo)
            if ri == 0:
                wlab = int(warmups[w]) if warmups is not None else w
                ax.set_title(f'dream {ci + 1}  (start {s}, warmup {wlab})',
                             fontsize=9, color=txt)
            if ci == 0:
                ax.set_ylabel(LABELS[c], fontsize=9, color=txt)
            if world_img is not None:
                ax.set_xticks([]); ax.set_yticks([])
            else:
                ax.set_aspect('equal')
            for sp_ in ax.spines.values():
                sp_.set_color('0.4')

    ttl = 'Decoded dream trajectories (rows: conditions, cols: example dreams'
    ttl += f'; all {Ns} seeds/cell'
    ttl += f', seed={env_seed})' if env_seed is not None else ')'
    fig.suptitle(ttl, fontsize=13, color=txt)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, Hp1 - 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.015, pad=0.02, aspect=40)
    cbar.set_label('Dream step', color=txt, fontsize=10)
    cbar.ax.yaxis.set_tick_params(color=txt)
    cbar.ax.tick_params(labelcolor=txt)
    fig.legend(handles=_marker_handles(cmap), loc='lower center', ncol=3,
               fontsize=9, facecolor='black', edgecolor='0.4', labelcolor=txt,
               framealpha=0.6, bbox_to_anchor=(0.5, -0.01))

    out = save_dir / 'decoded_dreams_by_condition.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=fc)
    plt.close(fig)
    print(f"  Saved {out.name}")


def animate_decoded_dreams(decoded, start_pos, future_pos, metadata, save_dir,
                           col_idx=0, fps=2, warmups=None, real_style='ghost'):
    """GIF: 1x4 condition panels for ONE example (start, warmup), stepping
    through the seeds one dream at a time over the bold cyan real trajectory.

    Uncluttered counterpart to the all-seeds grid — each frame is a single
    stochastic dream so its shape vs the real path is legible. col_idx picks
    which selected column (start/warmup) to animate.
    """
    try:
        import io
        from PIL import Image
    except Exception as e:
        print(f"  (gif skipped, Pillow unavailable: {e})")
        return
    S, Nw, Ns, Hp1, _ = decoded['A'].shape
    cols = _select_columns(S, Nw, max(col_idx + 1, 1))
    s, w = cols[min(col_idx, len(cols) - 1)]
    wlab = int(warmups[w]) if warmups is not None else w
    world_img, env_seed, tile_size = _render_world(metadata)
    to_px = _to_px_fn(world_img, tile_size)

    # Fixed bounds over all seeds of this example (across conditions) + real path.
    dxy = np.concatenate([decoded[c][s, w].reshape(-1, 2) for c in CONDITIONS])
    real_path = np.concatenate([start_pos[s, w][None], future_pos[s, w]])
    allp = to_px(np.concatenate([dxy, real_path], axis=0))
    pad = (tile_size * 4) if world_img is not None else 2
    xlo, xhi = allp[:, 0].min() - pad, allp[:, 0].max() + pad
    ylo, yhi = allp[:, 1].min() - pad, allp[:, 1].max() + pad

    cmap = plt.cm.plasma
    fc = '#1a1a1a' if world_img is not None else 'white'
    txt = 'white' if world_img is not None else 'black'

    frames = []
    for k in range(Ns):
        # Fixed figure size + dpi (no tight bbox) so every frame is identical px.
        fig, axes = plt.subplots(1, 4, figsize=(16.8, 4.7), facecolor=fc)
        for ax, c in zip(axes, CONDITIONS):
            ax.set_facecolor(fc)
            if world_img is not None:
                ax.imshow((world_img * 0.6).astype(np.uint8))
            _draw_real_traj(ax, start_pos[s, w], future_pos[s, w], to_px, real_style)
            tp = to_px(decoded[c][s, w, k])                    # (Hp1, 2)
            for t in range(Hp1 - 1):
                ax.plot(tp[t:t + 2, 0], tp[t:t + 2, 1], '-',
                        color=cmap(t / max(Hp1 - 1, 1)), linewidth=2.2,
                        alpha=0.95, zorder=3)
            ax.plot(tp[-1, 0], tp[-1, 1], 's', color=cmap(1.0), markersize=8,
                    markeredgecolor='white', markeredgewidth=0.8, zorder=5)
            _draw_decoded_start(ax, decoded[c][s, w, k, 0], to_px)
            if c == CONDITIONS[0]:
                ax.legend(handles=_marker_handles(cmap), loc='upper left',
                          fontsize=7, facecolor='black', edgecolor='0.4',
                          labelcolor=txt, framealpha=0.5)
            ax.set_title(LABELS[c], fontsize=9, color=txt)
            ax.set_xlim(xlo, xhi)
            ax.set_ylim(yhi, ylo)
            if world_img is not None:
                ax.set_xticks([]); ax.set_yticks([])
            else:
                ax.set_aspect('equal')
        st = f'Dream seed {k + 1}/{Ns}  —  start {s}, warmup {wlab}'
        st += f'  (world seed={env_seed})' if env_seed is not None else ''
        fig.suptitle(st, fontsize=13, color=txt)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.02,
                            wspace=0.05)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=110, facecolor=fc)
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert('RGB'))

    out = save_dir / 'decoded_dreams_animation.gif'
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=int(1000 / max(fps, 1)), loop=0)
    print(f"  Saved {out.name} ({Ns} frames, start {s}/warmup {wlab})")


def _band(ax, data, color, label, ls='-', central='median', shading='band'):
    """Central-tendency curve + spread over rollouts. data: (N, steps).

    central: 'median' (spread = IQR) or 'mean' (spread = +/-1 std).
    shading: 'band' (filled region), 'bars' (error bars), or 'none' (line only).
    """
    steps = np.arange(data.shape[1])
    # Continuity panels pass an all-NaN step-0 column (no predecessor); nan-agg
    # over it is intentional and yields NaN (skipped by plot). Silence the noise.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
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


def _pzero(stepD):
    """Per-step stay indicator from step displacement (N, H+1): 1.0 where the
    decoded position didn't change (dx==0), 0.0 where it moved, NaN at step 0
    (no predecessor). nanmean over rollouts -> P[dx=0] at each step."""
    return np.where(np.isnan(stepD), np.nan, (stepD == 0).astype(float))


def _moved(stepD):
    """Step displacement masked to moving steps only (dx>0); dx==0 and step 0
    become NaN. nanmean over rollouts -> E[dx | dx>0] at each step."""
    return np.where(stepD > 0, stepD, np.nan)


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
    dispA, distB, varC, distB_by_w, stepD = {}, {}, {}, {}, {}
    for c in CONDITIONS:
        pos = decoded[c]                                 # (S, Nw, Ns, H+1, 2)
        dA = _manhattan(pos, pos[:, :, :, :1, :])        # (S, Nw, Ns, H+1)
        dB = _manhattan(pos, rs)                         # (S, Nw, Ns, H+1)
        dispA[c] = dA.reshape(S * Nw * Ns, Hp1)
        distB[c] = dB.reshape(S * Nw * Ns, Hp1)
        centroid = pos.mean(axis=2, keepdims=True)       # (S, Nw, 1, H+1, 2)
        varC[c] = _manhattan(pos, centroid).mean(axis=2).reshape(S * Nw, Hp1)
        distB_by_w[c] = _agg(dB[..., -1], axis=(0, 2))   # (Nw,)
        # Step-to-step decoded displacement (continuity): distance from previous
        # tile at each dream step. Step 0 has no predecessor -> NaN (keeps x-axis
        # aligned 0..H with the other panels; nan-aggregation skips it).
        step = _manhattan(pos[:, :, :, 1:, :], pos[:, :, :, :-1, :])  # (S,Nw,Ns,H)
        stepD[c] = np.concatenate([
            np.full((S, Nw, Ns, 1), np.nan), step], axis=3).reshape(S * Nw * Ns, Hp1)
    realA = np.concatenate([
        np.zeros((S, Nw, 1)),
        _manhattan(future_pos, start_pos[:, :, None, :]),      # (S, Nw, H)
    ], axis=2).reshape(S * Nw, Hp1)
    # Real ground-truth step distance for the continuity panel: consecutive
    # Manhattan along [start, future_1..future_H]. No decoder jitter -> a floor.
    real_path = np.concatenate([start_pos[:, :, None, :], future_pos], axis=2)  # (S,Nw,H+1,2)
    realStep = np.concatenate([
        np.full((S, Nw, 1), np.nan),
        _manhattan(real_path[:, :, 1:, :], real_path[:, :, :-1, :]),  # (S, Nw, H)
    ], axis=2).reshape(S * Nw, Hp1)
    return dispA, distB, varC, realA, distB_by_w, stepD, realStep


def _print_summary(dispA, distB, varC, realA, central, stepD=None, realStep=None):
    _agg = np.nanmean if central == 'mean' else np.nanmedian
    print(f"\n[summary] {central} final-step (H) values, pooled over warmups:")
    for c in CONDITIONS:
        # step-dist / P[dx=0] / E[dx|dx>0] pooled over ALL steps (continuity),
        # not just the final step. P and conditional-mean always use nanmean.
        if stepD is not None:
            sd = (f"  step-dist={_agg(stepD[c]):5.2f}"
                  f"  P[dx=0]={np.nanmean(_pzero(stepD[c])):4.2f}"
                  f"  E[dx|dx>0]={np.nanmean(_moved(stepD[c])):5.2f}")
        else:
            sd = ""
        print(f"  {c}: own-disp={_agg(dispA[c][:, -1]):5.2f}  "
              f"real-dist={_agg(distB[c][:, -1]):5.2f}  "
              f"seed-var={_agg(varC[c][:, -1]):5.2f}{sd} tiles")
    if realStep is not None:
        rstep = (f"  step-dist={_agg(realStep):5.2f}"
                 f"  P[dx=0]={np.nanmean(_pzero(realStep)):4.2f}"
                 f"  E[dx|dx>0]={np.nanmean(_moved(realStep)):5.2f}")
    else:
        rstep = ""
    print(f"  real: own-disp={_agg(realA[:, -1]):5.2f}{rstep} tiles")


def replot_from_pkl(pkl_path, save_dir, central='median', shading='band',
                    n_example_dreams=24, make_gif=True, gif_col=0, gif_fps=2,
                    real_style='ghost'):
    """Regenerate plots from a saved results pkl — no agent/rollouts/decode."""
    pkl_path = Path(pkl_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Replotting from {pkl_path} (central={central}, shading={shading})")
    with open(pkl_path, 'rb') as f:
        r = pickle.load(f)
    dispA, distB, varC, realA, distB_by_w, stepD, realStep = compute_curves(
        r['decoded'], r['start_pos'], r['future_pos'], central)
    warmups = r['warmups']
    _print_summary(dispA, distB, varC, realA, central, stepD, realStep)
    print("\nGenerating plots...")
    plot_seed_ablation(dispA, distB, varC, realA, save_dir,
                       central=central, shading=shading,
                       stepD=stepD, realStep=realStep,
                       decoded=r['decoded'], start_pos=r['start_pos'],
                       future_pos=r['future_pos'], warmups=warmups)
    plot_error_vs_warmup(distB_by_w, warmups, save_dir, central=central)
    if n_example_dreams > 0:
        plot_decoded_dreams(r['decoded'], r['start_pos'], r['future_pos'],
                            r['metadata'], save_dir, n_cols=n_example_dreams,
                            warmups=warmups, real_style=real_style)
    if make_gif:
        animate_decoded_dreams(r['decoded'], r['start_pos'], r['future_pos'],
                               r['metadata'], save_dir, col_idx=gif_col,
                               fps=gif_fps, warmups=warmups, real_style=real_style)
    print(f"Done. Plots written to {save_dir}")


def plot_seed_ablation(dispA, distB, varC, realA, save_dir,
                       central='median', shading='band', stepD=None,
                       realStep=None, decoded=None, start_pos=None,
                       future_pos=None, warmups=None):
    band = bind(_band, central=central, shading=shading)
    # P[dx=0] and E[dx|dx>0] are expectations: always mean-aggregated over
    # rollouts, regardless of `central` (median of a 0/1 indicator is meaningless).
    band_mean = bind(_band, central='mean', shading=shading)

    # Panel G (deviation-by-warmup) is included when the raw decoded positions
    # are available (live run or --from_pkl). It carries its own two-part legend
    # and lines-only styling, so it's excluded from the shared legend loop below.
    have_G = decoded is not None and start_pos is not None \
        and future_pos is not None and warmups is not None
    n_panels = (6 if stepD is not None else 3) + (1 if have_G else 0)
    # Reflow into 2 rows once we exceed 4 panels so the figure isn't absurdly wide.
    ncol = int(np.ceil(n_panels / 2)) if n_panels > 4 else n_panels
    nrow = int(np.ceil(n_panels / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 5.2 * nrow),
                             squeeze=False)
    flat = axes.ravel()

    ax = flat[0]
    for c in CONDITIONS:
        band(ax, dispA[c], COLORS[c], LABELS[c])
    band(ax, realA, 'black', 'real trajectory', ls='--')
    ax.set_title('A) displacement from own start')
    ax.set_ylabel('Manhattan distance from start (tiles)')

    ax = flat[1]
    for c in CONDITIONS:
        band(ax, distB[c], COLORS[c], LABELS[c])
    band(ax, realA, 'black', 'real trajectory', ls='--')
    ax.set_title('B) distance from real start')
    ax.set_ylabel('Manhattan distance from real start (tiles)')

    ax = flat[2]
    for c in CONDITIONS:
        band(ax, varC[c], COLORS[c], LABELS[c])
    ax.set_title('C) cross-seed variability')
    ax.set_ylabel('Mean seed dispersion (tiles)')

    pzero_ax = None
    if stepD is not None:
        ax = flat[3]
        for c in CONDITIONS:
            band(ax, stepD[c], COLORS[c], LABELS[c])
        if realStep is not None:
            band(ax, realStep, 'black', 'real trajectory', ls='--')
        ax.set_title('D) step-to-step displacement (continuity)')
        ax.set_ylabel('Manhattan distance from previous tile (tiles)')

        ax = flat[4]
        pzero_ax = ax
        for c in CONDITIONS:
            band_mean(ax, _pzero(stepD[c]), COLORS[c], LABELS[c])
        if realStep is not None:
            band_mean(ax, _pzero(realStep), 'black', 'real trajectory', ls='--')
        ax.set_title('E) P[dx=0] (stay probability)')
        ax.set_ylabel('P(no move to a new tile)')

        ax = flat[5]
        for c in CONDITIONS:
            band_mean(ax, _moved(stepD[c]), COLORS[c], LABELS[c])
        if realStep is not None:
            band_mean(ax, _moved(realStep), 'black', 'real trajectory', ls='--')
        ax.set_title('F) E[dx | dx>0] (jump size when moving)')
        ax.set_ylabel('Mean displacement over moving steps (tiles)')

    # Panel G: deviation-from-real-trajectory by warmup (bespoke legend).
    dev_ax = None
    if have_G:
        dev_ax = flat[n_panels - 1]
        drew = _draw_deviation_by_warmup(
            dev_ax, decoded, start_pos, future_pos, warmups, central=central)
        if not drew:
            dev_ax = None  # nothing drawn -> treat like a normal (empty) axis

    for i, ax in enumerate(flat):
        if i >= n_panels:
            ax.set_visible(False)  # hide unused slots in the 2-row grid
            continue
        ax.set_xlabel('Dream step (0 = seed)')
        ax.grid(alpha=0.3)
        ax.set_xlim(0, dispA['A'].shape[1] - 1)
        ax.set_ylim(bottom=0)
        if ax is not dev_ax:  # panel G manages its own two-part legend
            ax.legend(loc='upper left', fontsize=8)
    if pzero_ax is not None:
        pzero_ax.set_ylim(0, 1)  # P[dx=0] is a probability

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


# Gray shades for the pooled (warmup-independent) reference conditions in
# panel G. Kept off the viridis ramp so they read as baselines, not warmup bins.
POOLED_GRAY = {'C': '0.55', 'D': '0.3'}


def _draw_deviation_by_warmup(ax, decoded, start_pos, future_pos, warmups,
                              bin_conditions=('A', 'B'),
                              pooled_conditions=('C', 'D'), central='median',
                              shading='none', bins=None, n_bins=None):
    """Draw panel G onto `ax`: per-step deviation from the REAL trajectory.

    Deviation at dream step t = |decoded_dream_t - real_t| (Manhattan tiles),
    with real_0 = start and real_t = future[t-1] (same alignment as
    dream_vs_future). Unlike panel B (distance from the real *start*, which
    grows simply because both dream and agent walk away from the seed), this is
    a tracking-fidelity metric: it rises as the dream drifts off the real path,
    and should sit LOWER at longer warmups if warmup helps seed the dream.

    `bin_conditions` (A/B) are split by warmup: `n_bins` equal-count groups over
    the ACTUAL warmups present (default DEFAULT_N_WARMUP_BINS=5, so every bin is
    non-empty and the full range is covered on any horizon). Color = warmup
    group (light=short -> dark=long), linestyle = condition (A '-', B ':').
    `pooled_conditions` (C/D) are drawn as a SINGLE gray line each, pooled over
    all warmups — their warmup axis is meaningless (noise deter is warmup-
    independent, and the position decoder reads from the deter, so C/D start
    ~4 tiles off regardless of W), so they serve as flat null baselines rather
    than a warmup story. Pass `bins` to override the auto ranges. Returns True
    if anything was drawn (caller manages the bespoke two-part legend), else
    False.
    """
    from matplotlib.lines import Line2D
    warmups = np.asarray(warmups)
    if bins is None:
        bins = _auto_warmup_bins(warmups, n_bins or DEFAULT_N_WARMUP_BINS)
    # Real trajectory as (S, Nw, H+1, 2): [start, future_1..future_H].
    real_path = np.concatenate([start_pos[:, :, None, :], future_pos], axis=2)
    Hp1 = real_path.shape[2]

    def dev_of(c):  # (S, Nw, Ns, H+1) per-step deviation from the real path
        return _manhattan(decoded[c], real_path[:, :, None, :, :])

    # Populate bins with the warmup indices whose warmup value falls in range.
    groups = []
    for lo, hi, lab in bins:
        idx = np.where((warmups >= lo) & (warmups <= hi))[0]
        if len(idx):
            groups.append((lab, idx))
        else:
            print(f"  (warmup bin '{lab}' [{lo},{hi}] empty, skipped)")

    drew = False
    cmap = plt.cm.viridis
    fracs = (np.linspace(0.12, 0.88, len(groups)) if len(groups) > 1
             else np.array([0.5]))

    # A/B: one warmup-colored line per (bin, condition).
    for gi, (lab, idx) in enumerate(groups):
        color = cmap(fracs[gi])
        for c in bin_conditions:
            dev_bin = dev_of(c)[:, idx].reshape(-1, Hp1)
            _band(ax, dev_bin, color, None, ls=COND_LS.get(c, '-'),
                  central=central, shading=shading)
            drew = True

    # C/D: single gray line each, pooled over ALL warmups (warmup-independent).
    for c in pooled_conditions:
        dev_all = dev_of(c).reshape(-1, Hp1)
        _band(ax, dev_all, POOLED_GRAY.get(c, '0.4'), None,
              ls=COND_LS.get(c, '-'), central=central, shading=shading)
        drew = True

    if not drew:
        print("  (nothing to draw; deviation panel left blank)")
        return False

    ax.set_title('G) deviation from real trajectory by warmup')
    ax.set_ylabel('Manhattan deviation from real trajectory (tiles)')

    # Two-part legend: warmup group (color, A/B) + condition (linestyle). A/B
    # entries are neutral gray (color carried by the warmup legend); C/D entries
    # use their pooled gray and are marked "(pooled)".
    grp_handles = [Line2D([0], [0], color=cmap(fracs[gi]), lw=2.5, label=lab)
                   for gi, (lab, _) in enumerate(groups)]
    cond_handles = [Line2D([0], [0], color='0.3', lw=2.0, ls=COND_LS.get(c, '-'),
                           label=LABELS[c]) for c in bin_conditions]
    cond_handles += [Line2D([0], [0], color=POOLED_GRAY.get(c, '0.4'), lw=2.0,
                            ls=COND_LS.get(c, '-'),
                            label=LABELS[c] + ' [pooled]')
                     for c in pooled_conditions]
    if grp_handles:
        leg1 = ax.legend(handles=grp_handles, title='Warmup group (A/B)',
                         loc='upper left', fontsize=7, title_fontsize=8)
        ax.add_artist(leg1)
    ax.legend(handles=cond_handles, title='Condition', loc='lower right',
              fontsize=7, title_fontsize=8)
    return True


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
        replot_from_pkl(cfg.from_pkl, save_dir, cfg.central, cfg.shading,
                        cfg.n_example_dreams, cfg.make_gif, cfg.gif_col,
                        cfg.gif_fps, cfg.real_style)
        return

    assert cfg.decoder_model, "Must provide --dream_seed_ablation.decoder_model"
    n_seeds = cfg.n_seeds

    save_dir = Path(cfg.save_path or str(
        elements.Path(args.logdir) / 'dream_seed_ablation'))
    save_dir.mkdir(parents=True, exist_ok=True)

    # Optional: dump raw dream deter per condition for manifold_analysis.py.
    # We stream each batch to disk (per-condition .npy) so peak RAM is one batch,
    # then concatenate into dream_deter_{cond}.pkl at the end. NOT subsampled.
    act_dir = None
    if cfg.save_activations:
        act_dir = save_dir / 'dream_activations_tmp'
        act_dir.mkdir(parents=True, exist_ok=True)

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
            raw = np.array(mets[f'seedabl/{c}/deter'])  # (RB, Nw, Nseed, H+1, Dt)
            decoded[c].append(decode(raw))
            if act_dir is not None:
                # collapse the start axes -> (N, H+1, Dt), N = RB*Nw*Nseed
                flat = raw.reshape(-1, *raw.shape[3:]).astype(np.float32)
                np.save(act_dir / f'{c}_b{b}.npy', flat)
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

    # Finalize raw-deter dumps: concat per-batch .npy -> dream_deter_{cond}.pkl
    # (one condition in RAM at a time). Format matches manifold_analysis.py's
    # loader (key 'dream_deter', shape (N, H+1, D) flattened to (N*(H+1), D)).
    if act_dir is not None:
        print("Saving raw dream deter per condition (for manifold analysis)...")
        for c in CONDITIONS:
            parts = [np.load(act_dir / f'{c}_b{b}.npy')
                     for b in range(cfg.num_batches)]
            arr = np.concatenate(parts, axis=0)  # (N_total, H+1, Dt)
            del parts
            out = save_dir / f'dream_deter_{c}.pkl'
            with open(out, 'wb') as f:
                pickle.dump({
                    'dream_deter': arr,
                    'dream_deter_shape': arr.shape,
                    'condition': c, 'label': LABELS.get(c, c),
                    'horizon': int(H), 'warmups': warmups,
                    'metadata': metadata, 'decoder_path': str(cfg.decoder_model),
                }, f)
            print(f"  {c} ({LABELS.get(c, c)}): {out.name}  "
                  f"shape={arr.shape}  {arr.nbytes / 1e9:.1f} GB")
            del arr
            for b in range(cfg.num_batches):
                (act_dir / f'{c}_b{b}.npy').unlink()
        try:
            act_dir.rmdir()
        except OSError:
            pass
    print(f"Pooled {S} start rows x {Nw} warmups {list(warmups)} x {Ns} seeds, "
          f"H+1={Hp1}")

    # 5. Curves (pooled over start rows and warmups; Nseed drives variability).
    central = cfg.central
    dispA, distB, varC, realA, distB_by_w, stepD, realStep = compute_curves(
        decoded, start_pos, future_pos, central)
    _print_summary(dispA, distB, varC, realA, central, stepD, realStep)

    # 6. Plots.
    print("\nGenerating plots...")
    plot_seed_ablation(dispA, distB, varC, realA, save_dir,
                       central=central, shading=cfg.shading,
                       stepD=stepD, realStep=realStep,
                       decoded=decoded, start_pos=start_pos,
                       future_pos=future_pos, warmups=warmups)
    plot_error_vs_warmup(distB_by_w, warmups, save_dir, central=central)
    if cfg.n_example_dreams > 0:
        plot_decoded_dreams(decoded, start_pos, future_pos, metadata, save_dir,
                            n_cols=cfg.n_example_dreams, warmups=warmups,
                            real_style=cfg.real_style)
    if cfg.make_gif:
        animate_decoded_dreams(decoded, start_pos, future_pos, metadata,
                               save_dir, col_idx=cfg.gif_col, fps=cfg.gif_fps,
                               warmups=warmups, real_style=cfg.real_style)

    # 7. Save.
    results = {
        'decoded': decoded,          # {cond: (S, Nw, Nseed, H+1, 2)}
        'start_pos': start_pos, 'future_pos': future_pos, 'warmups': warmups,
        'dispA': dispA, 'distB': distB, 'varC': varC, 'realA': realA,
        'distB_by_warmup': distB_by_w,
        'S': S, 'Nw': Nw, 'n_seeds': Ns, 'H': H, 'Hp1': Hp1,
        'decoder_path': str(cfg.decoder_model), 'decoder_metadata': dec_meta,
        'metadata': metadata, 'conditions': CONDITIONS, 'labels': LABELS,
        'noise_mode': cfg.noise_mode,
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
                  'n_example_dreams': cfg.n_example_dreams,
                  'noise_mode': cfg.noise_mode,
                  'from_checkpoint': args.from_checkpoint},
            outputs=['seed_ablation_curves.png', 'error_vs_warmup.png',
                     'decoded_dreams_by_condition.png',
                     'dream_seed_ablation_results.pkl'],
            extra={'n_start_rows': int(S), 'n_warmups': int(Nw),
                   'warmups': [int(w) for w in warmups], 'horizon': int(H)})
    except Exception as e:
        print(f"  (run_info logging skipped: {e})")

    logger.close()
    print("Done.")
