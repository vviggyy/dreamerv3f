"""Day/night-conditioned spatial place fields: does time-of-day remap the world
model's spatial code, or just scale it with screen brightness?

The day/night analog of `vital_place_fields.py`. For the top spatially-tuned
`dyn/deter` neurons we build, for each, its 2D place field at several points in
crafter's day/night cycle (morning / noon / evening / midnight) and ask whether
the field reorganizes across the day.

Why day/night is a cleaner conditioning axis than a vital (the motivation):
crafter's daylight is an EXOGENOUS clock -- `daylight = 1 - |cos(pi*((step/300)%1
+ 0.3))|**3`, period 300 steps -- uncorrelated with WHERE the agent is. So every
location is sampled across the whole cycle, the phase groups are well balanced,
and the field comparison is not confounded by occupancy/behavior (unlike vitals,
which the agent regulates AT specific places). We can therefore afford 4 phase
bins and still retain a large co-visited set.

Two effects, kept distinct (same split as the vital analysis):
  * GAIN (brightness) -- the field keeps its shape but scales with daylight. This
    is EXPECTED here, because crafter literally dims the screen at night
    (`engine._light`). Measured by the day/night mean-activation ratio over
    co-visited bins. A pure-gain cell is the day/night null-but-interesting case.
  * SPATIAL REMAPPING (shape) -- the field moves / reorganizes with time of day,
    beyond any brightness gain. Measured by the gain-invariant spatial correlation
    of the day-vs-night maps over CO-VISITED bins (mean-removed, so pure gain
    divides out). Low r = genuine remapping. THIS is the strong claim.

Significance: splitting data day/night always makes two maps differ from sampling
noise. Null = shuffle the day/night labels WITHIN each position bin (preserves
occupancy, breaks the daylight<->activation link), recompute the maps, collect the
correlation. Real r below the null percentile => genuine time-of-day remapping.

The r/gain/p statistics use the bright-vs-dark EXTREME contrast (day = daylight >=
--bright, night = daylight <= --dark); the multi-phase maps (--nphases) are the
"same neuron through the day" visualization. Needs a trajectory set with
`dyn/deter` + per-step `daylight` + `env_step` (eval_trajectory records all three
since 2026-10-17). NOT vitals_only.

Usage:
  MPLBACKEND=Agg python dreamerv3/daynight_place_fields.py \
      --data ./logdir/my_run/trajectories \
      --save ./logdir/my_run/daynight_place_fields \
      --layer dyn/deter --nphases 4 --top_n 10

  # replot from the results pkl (cosmetic re-render only):
  python dreamerv3/daynight_place_fields.py --from_pkl ./.../daynight_place_fields_results.pkl
"""
import argparse
import glob
import json
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plot_presentable as pp

# Reuse the vital analysis's binning / EV / conditioning / selection machinery --
# the day/night stats are literally the vital lo/hi contrast with lo=night,
# hi=day, so the math is shared rather than reimplemented.
from vital_place_fields import (  # noqa: E402
    _flatbin, _maps, _ev, _condition, _allneuron_remap, _select,
    _grid, _occ_grid, _occ_prob_grid, _area, OCC_MIN_VISITS, OCC_UNDER_COLOR)

# crafter day/night cycle (crafter/env.py::_update_time): daylight peaks (noon)
# at clock-phase ~0.2 and bottoms (midnight) at ~0.7; period 300 steps. We bin by
# the cyclic clock phase (from env_step) so morning/evening -- which share a
# daylight VALUE on the rising/falling limbs -- are separated. _DAWN_OFFSET aligns
# bin 0 to early morning (phase ~0.95, where daylight is rising), so for nphases=4
# the chronological bins are morning, noon, evening, midnight.
_PERIOD = 300.0
_DAWN_OFFSET = 0.825
_PHASE4_LABELS = ['morning', 'noon', 'evening', 'midnight']
_PHASE2_LABELS = ['day', 'night']
# viridis-on-a-clock: warm at midday, cool/dark at night
_PHASE4_COLORS = {'morning': '#f0a202', 'noon': '#d62728',
                  'evening': '#6a51a3', 'midnight': '#2c3e70',
                  'day': '#e6a817', 'night': '#2c3e70'}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _load(data_dir, layer):
    """Stream per-episode: concatenate deter (N,D), pos (N,2), daylight (N,),
    env_step (N,). Reads act/<layer> (falls back to top-level 'deter'). Loads one
    layer + pos + daylight + env_step per episode, so RAM ~ one episode."""
    act_key = f'act/{layer}'
    files = sorted(glob.glob(str(Path(data_dir) / 'episode_*.pkl')))
    if not files:
        raise SystemExit(f'No episode_*.pkl in {data_dir}')
    X, pos, day, step = [], [], [], []
    for f in files:
        with open(f, 'rb') as fh:
            ep = pickle.load(fh)
        a = ep.get(act_key)
        if a is None and layer == 'dyn/deter':
            a = ep.get('deter')
        p = ep.get('player_pos')
        d = ep.get('daylight')
        s = ep.get('env_step')
        if a is None or p is None or d is None or len(a) == 0:
            continue
        a = np.asarray(a, np.float32)
        if a.ndim > 2:
            a = a.reshape(len(a), -1)
        p = np.asarray(p, np.float32)
        d = np.asarray(d, np.float64)
        # env_step resets to 0 each episode; fall back to the step index (which
        # equals crafter's _step at episode start -- see CLAUDE.md daylight note).
        s = np.asarray(s, np.float64) if s is not None else np.arange(len(a), dtype=np.float64)
        T = min(len(a), len(p), len(d), len(s))
        X.append(a[:T]); pos.append(p[:T]); day.append(d[:T]); step.append(s[:T])
        del ep
    if not X:
        raise SystemExit('No episodes with activations + daylight. Needs a '
                         'light_activations (or full) eval that records daylight.')
    X = np.concatenate(X); pos = np.concatenate(pos)
    day = np.concatenate(day); step = np.concatenate(step)
    print(f'Loaded {len(X)} timesteps, {X.shape[1]} {layer} units, '
          f'{len(files)} episode files; daylight range '
          f'[{day.min():.2f}, {day.max():.2f}]')
    return X, pos, day, step


# ---------------------------------------------------------------------------
# Phase binning
# ---------------------------------------------------------------------------
def _phase_bins(step, day, nphases, bright=0.8, dark=0.2):
    """Time-of-day bins. Returns (labels, masks, mean_daylight_per_bin).

    nphases == 2: DAY/NIGHT by daylight VALUE windows around the peaks — day =
    daylight >= bright, night = daylight <= dark — the SAME definition the remap
    stat uses, so the maps and the statistic are the identical day-vs-night
    contrast. Twilight (dark < daylight < bright) is excluded from both maps.

    nphases != 2: contiguous cyclic clock-phase slices (bin 0 at dawn,
    _DAWN_OFFSET); nphases=4 -> morning/noon/evening/midnight. These are the
    "same neuron through the day" visualization (all steps assigned)."""
    if nphases == 2:
        masks = [day >= bright, day <= dark]          # day, night (peak windows)
        labels = list(_PHASE2_LABELS)
        mday = [float(day[m].mean()) if m.any() else np.nan for m in masks]
        return labels, masks, mday
    phase = (step / _PERIOD) % 1.0
    shifted = (phase - _DAWN_OFFSET) % 1.0
    idx = np.minimum((shifted * nphases).astype(int), nphases - 1)
    masks, mday = [], []
    for k in range(nphases):
        m = idx == k
        masks.append(m)
        mday.append(float(day[m].mean()) if m.any() else np.nan)
    if nphases == 4:
        labels = list(_PHASE4_LABELS)
    else:
        labels = [f'p{k}~{md:.2f}' for k, md in enumerate(mday)]
    return labels, masks, mday


def _phase_maps(X_sel, flatbin, masks, nbins):
    """Per-phase mean place-field maps + visit counts for the selected neurons.

    Returns (tc (P, nbins, k), occ (P, nbins)) with NaN on unvisited bins."""
    P = len(masks)
    tc = np.full((P, nbins, X_sel.shape[1]), np.nan)
    occ = np.zeros((P, nbins))
    for i, m in enumerate(masks):
        mm, cc = _maps(flatbin[m], X_sel[m], nbins)
        tc[i] = mm; occ[i] = cc
    return tc, occ


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def _get_world_img(results, frame='tuning'):
    """Render the faint Crafter world RGB for the map backgrounds (None on any
    failure -> falls back to dark-grey unvisited tiles). Reuses tuning_curve's
    reconstruction so the terrain frame matches the occupancy/tuning figures.
    ``frame`` ('tuning' y-up / 'native' y-down) must match the map orientation."""
    wm = results.get('world_meta')
    if not wm:
        return None
    try:
        from tuning_curve import _render_world_img
        return _render_world_img(wm, tuple(results['area']), frame=frame)
    except Exception as e:
        print(f'  (world background skipped: {e})')
        return None


def plot(results, save_dir, fmt='svg', world_bg=True, native=False,
         tuning_cmap=False):
    area = tuple(results['area'])
    labels = results['phase_labels']
    sel = results['selected']
    ev = results['ev']
    P = len(labels)
    cond = results['cond']['daynight']
    world_img = _get_world_img(results, frame='native' if native else 'tuning') \
        if world_bg else None

    cols = ['full'] + labels
    nmap = len(cols)                 # map columns (full + phases)
    nrow, ncol = len(sel) + 1, nmap + 1   # + a thin trailing colorbar column
    fig, ax = plt.subplots(nrow, ncol, figsize=(2.1 * nmap + 0.8, 2.1 * nrow),
                           gridspec_kw={'width_ratios': [1] * nmap + [0.12]},
                           squeeze=False)

    # native=True => Crafter's upright gameplay orientation (y down); else the
    # y-up "tuning" frame. world_img must already be rendered in the same frame.
    if native:
        extent = [-0.5, area[0] - 0.5, area[1] - 0.5, -0.5]
        grid_origin = 'upper'
    else:
        extent = [-0.5, area[0] - 0.5, -0.5, area[1] - 0.5]
        grid_origin = 'lower'

    def show(a, grid, vmin, vmax, cmap='viridis', visit_counts=None):
        # With a world background, unvisited (NaN) tiles are left transparent so
        # the faint Crafter terrain shows through (matches occupancy_fullset /
        # the EV-sorted tuning maps); otherwise they stay dark grey.
        # tuning_cmap=True uses the exact black->red->orange tuning/occupancy
        # colormap for ALL panels so these figures match the poster occupancy
        # & tuning instead of viridis/magma.
        if tuning_cmap:
            from tuning_curve import _tc_cmap
            cm = _tc_cmap(world_bg=world_img is not None)   # set_bad handled inside
        else:
            cm = plt.get_cmap(cmap).copy()
            cm.set_bad(alpha=0.0) if world_img is not None else cm.set_bad('#4d4d4d')
        if world_img is not None:
            a.imshow(world_img, origin='upper', aspect='equal', extent=extent,
                     alpha=0.4, zorder=0)
        im = a.imshow(np.ma.masked_invalid(grid.T), origin=grid_origin, cmap=cm,
                      vmin=vmin, vmax=vmax, interpolation='nearest',
                      extent=extent, aspect='equal', zorder=1)
        # Occupancy panels: tiles with 1..OCC_MIN_VISITS-1 visits render grey
        # (matches plot_fullset_occupancy's set_under); unvisited (0) stays
        # transparent so the world map shows through.
        if visit_counts is not None:
            from matplotlib.colors import ListedColormap
            cg = _grid(visit_counts, area)
            low = np.where((cg > 0) & (cg < OCC_MIN_VISITS), 1.0, np.nan)
            gcm = ListedColormap([OCC_UNDER_COLOR]); gcm.set_bad(alpha=0.0)
            a.imshow(np.ma.masked_invalid(low.T), origin=grid_origin, cmap=gcm,
                     vmin=0, vmax=1, interpolation='nearest', extent=extent,
                     aspect='equal', zorder=2)
        a.set_xticks([]); a.set_yticks([])
        return im

    def cbar(im, r, label):
        cb = fig.colorbar(im, cax=ax[r, -1])
        cb.ax.tick_params(labelsize=6)
        cb.set_label(label, fontsize=7)

    # header row: occupancy (full + per-phase), SHARED scale across columns so
    # the midnight huddle vs noon spread is directly comparable.
    # Phase occupancy panels share ONE scale so they're comparable (midnight
    # huddle vs noon spread). The FULL-occupancy panel is a different total (sum
    # over phases) so it keeps its own autoscale. Cap the shared scale at the 98th
    # pct of pooled phase visits so one hot bin (spawn/huddle) doesn't wash the
    # rest to black.
    # Probability-normalized occupancy (count/total per phase) so day vs night
    # coverage is comparable even if the two phases contain different # steps.
    full_occ = _occ_prob_grid(results['occ_full'], area)
    phase_occ = [_occ_prob_grid(o, area) for o in results['phase_occ']]
    pooled = np.concatenate([g[np.isfinite(g)] for g in phase_occ
                             if np.isfinite(g).any()]) if phase_occ else np.array([])
    occ_vmax = np.percentile(pooled, 98) if pooled.size else 1
    ff = full_occ[np.isfinite(full_occ)]
    show(ax[0, 0], full_occ, 0, np.percentile(ff, 98) if ff.size else None, 'magma',
         visit_counts=results['occ_full'])
    ax[0, 0].set_title('occupancy (all)', fontsize=8)
    ax[0, 0].set_ylabel('OCCUPANCY', fontsize=8)
    im_occ = None
    for i, lab in enumerate(labels):
        im_occ = show(ax[0, i + 1], phase_occ[i], 0, occ_vmax, 'magma',
                      visit_counts=results['phase_occ'][i])
        col = _PHASE4_COLORS.get(lab, 'k')
        ax[0, i + 1].set_title(f'{lab}\n(daylight {results["phase_daylight"][i]:.2f})',
                               fontsize=8, color=col)
    if im_occ is not None:
        cbar(im_occ, 0, 'visit prob (phases, shared)')

    for r, nidx in enumerate(sel, start=1):
        full_map = _grid(results['tc_full'][:, r - 1], area)
        finite = full_map[np.isfinite(full_map)]
        vmin = np.percentile(finite, 2) if finite.size else 0
        vmax = np.percentile(finite, 98) if finite.size else 1
        im = show(ax[r, 0], full_map, vmin, vmax)
        rsn = results.get('reason', {}).get(nidx, '')
        rmp = results.get('remap', {})
        rmp_v = rmp[nidx] if len(rmp) else np.nan
        ax[r, 0].set_ylabel(f'n{nidx} [{rsn}]\nEV={ev[nidx]:.2f} rmp={rmp_v:.2f}',
                            fontsize=7)
        # per-phase maps, shared vmin/vmax WITHIN the row so brightness-gain is visible
        for i in range(P):
            im = show(ax[r, i + 1], _grid(results['phase_tc'][i][:, r - 1], area),
                      vmin, vmax)
        # day/night contrast stats on the last (midnight) panel title
        pr, pp_val = cond['r'][r - 1], cond['p'][r - 1]
        nm = cond.get('null_med', np.full_like(cond['r'], np.nan))[r - 1]
        g = cond['gain'][r - 1]
        star = '*' if pp_val < 0.05 else ''
        ax[r, P].set_title(f'day/night r={pr:.2f}{star}\nnull={nm:.2f} gain={g:.2f}',
                           fontsize=6.5, color='#2c3e70')
        cbar(im, r, 'mean act')

    fig.suptitle(
        f'Day/night-conditioned place fields ({results["layer"]}, top-{len(sel)})\n'
        'maps share a color scale (night dimmer = brightness gain); day/night r (*) '
        '= field remaps across the day (shape, gain-invariant); gain = day/night ratio',
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = (Path(save_dir) /
         f'daynight_place_fields_{results["layer"].replace("/", "_")}.{fmt}')
    fig.savefig(p, bbox_inches='tight', pad_inches=0.3, dpi=130); plt.close(fig)
    print(f'Saved {p}')


def plot_population(results, save_dir, fmt='svg'):
    """Population view: EV vs day/night-remap scatter (selected highlighted) +
    remap-score histogram over all neurons."""
    if 'remap' not in results or not len(results['remap']):
        return
    ev = np.asarray(results['ev']); remap = np.asarray(results['remap'])
    min_ev = float(results.get('args', {}).get('ev_floor', 0.0) or 0.0)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4.2))
    fin = np.isfinite(ev) & np.isfinite(remap)
    # Every neuron drawn identically — no highlight of the shown/selected cells.
    a0.scatter(ev[fin], remap[fin], s=10, c='0.55', alpha=0.5,
               edgecolors='none', label='all neurons')
    if min_ev > 0:
        a0.axvline(min_ev, color='k', ls='--', lw=1.2, label=f'EV = {min_ev:g}')
    a0.set_xlabel('EV (spatial reliability)')
    a0.set_ylabel('day/night remap score (1 - r, day vs night)')
    a0.set_title('Spatial reliability vs day/night remapping')
    a0.legend(fontsize=8, frameon=False); a0.grid(alpha=0.3)
    a1.hist(remap[np.isfinite(remap)], bins=60, color='0.6')
    a1.set_xlabel('day/night remap score (1 - r)'); a1.set_ylabel('neurons')
    a1.set_title('Remap-score distribution (all neurons)')
    a1.grid(alpha=0.3)
    fig.suptitle(f'{results["layer"]}: does time of day remap the spatial code?',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = (Path(save_dir) /
         f'daynight_place_fields_population_{results["layer"].replace("/", "_")}.{fmt}')
    fig.savefig(p, bbox_inches='tight', pad_inches=0.3, dpi=130); plt.close(fig)
    print(f'Saved {p}')


def print_summary(results):
    cond = results['cond']['daynight']
    print('\n=== day/night remapping summary (shape corr day-vs-night; '
          'null=null-median r) ===')
    print(f'day (daylight>={results["bright"]}): n={cond["n_hi"]}  '
          f'night (daylight<={results["dark"]}): n={cond["n_lo"]}')
    nmed = cond.get('null_med', np.full_like(cond['r'], np.nan))
    for r, nidx in enumerate(results['selected']):
        sig = '*' if cond['p'][r] < 0.05 else ' '
        rsn = results.get('reason', {}).get(nidx, '')
        print(f'   n{nidx:<4d}[{rsn:>5s}] EV={results["ev"][nidx]:.2f}  '
              f'r={cond["r"][r]:+.2f} (null {nmed[r]:+.2f}) '
              f'p={cond["p"][r]:.3f}{sig}  gain(day/night)={cond["gain"][r]:.2f}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', help='trajectory dir with act/<layer> + daylight')
    ap.add_argument('--save', default='./daynight_place_fields')
    ap.add_argument('--layer', default='dyn/deter')
    ap.add_argument('--nphases', type=int, default=2,
                    help='# day-cycle bins to VISUALIZE. 2 (default) = day/night by '
                         'daylight VALUE windows (day>=bright, night<=dark), the '
                         'SAME contrast as the stat. 4 = morning/noon/evening/'
                         'midnight contiguous clock-phase slices (all steps).')
    ap.add_argument('--top_n', type=int, default=10, help='neurons to show')
    ap.add_argument('--select', default='both',
                    choices=['ev', 'remap', 'conj', 'both', 'medium'],
                    help='ev=top spatial (stable baseline), remap=top day/night-'
                    'remappers, conj=jointly high EV AND remap, both, medium=band')
    ap.add_argument('--ev_band', default='0.5,0.8',
                    help='[--select medium] EV window "lo,hi"')
    ap.add_argument('--remap_band', default='0.4,0.8',
                    help='[--select medium] remap window "lo,hi"')
    ap.add_argument('--remap_ev_pct', type=float, default=75.0,
                    help='remap candidates must have EV >= this percentile of EV')
    ap.add_argument('--ev_floor', type=float, default=0.0,
                    help='HARD absolute EV floor (e.g. 0.4): only EV>=this are '
                    'selectable, and the remap-vs-EV scatter marks the threshold')
    ap.add_argument('--bright', type=float, default=0.8,
                    help='daylight >= bright  => DAY (hi) bin for the contrast')
    ap.add_argument('--dark', type=float, default=0.2,
                    help='daylight <= dark    => NIGHT (lo) bin for the contrast')
    ap.add_argument('--min_visits', type=int, default=5,
                    help='min visits per bin in BOTH day & night to count co-visited')
    ap.add_argument('--n_shuffle', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--fmt', default='svg', choices=['svg', 'png', 'pdf'])
    ap.add_argument('--from_pkl', default='', help='replot from results pkl')
    ap.add_argument('--world_bg', action='store_true', default=True,
                    help='draw the faint Crafter world map behind the maps '
                         '(unvisited tiles transparent); default on')
    ap.add_argument('--no_world_bg', dest='world_bg', action='store_false',
                    help='dark-grey unvisited tiles instead of the world map')
    ap.add_argument('--native', action='store_true',
                    help="Crafter's upright gameplay orientation (y down) instead "
                         'of the y-up tuning frame (vertical reflection)')
    ap.add_argument('--tuning_cmap', action='store_true',
                    help='Use the black->red->orange tuning/occupancy colormap for '
                         'all map panels (match the poster occupancy & tuning) '
                         'instead of viridis/magma')
    pp.add_presentable_arg(ap)
    args = ap.parse_args()
    pp.maybe_apply(args, small_figsize=False)  # Tier-1 style + --out_fmt; no-op otherwise

    if args.from_pkl:
        with open(args.from_pkl, 'rb') as f:
            results = pickle.load(f)
        sd = args.save if args.save != './daynight_place_fields' else str(Path(args.from_pkl).parent)
        Path(sd).mkdir(parents=True, exist_ok=True)
        plot(results, sd, args.fmt, world_bg=args.world_bg, native=args.native,
             tuning_cmap=args.tuning_cmap)
        plot_population(results, sd, args.fmt)
        print_summary(results)
        print(f'\nReplotted {args.from_pkl} -> {sd}')
        return

    if not args.data:
        raise SystemExit('--data required (unless --from_pkl)')
    save_dir = Path(args.save); save_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    X, pos, day, step = _load(args.data, args.layer)
    area = _area(args.data, pos)
    nbins = area[0] * area[1]
    flatbin = _flatbin(pos, area)
    print(f'Area {area}, {nbins} bins')

    labels, masks, mday = _phase_bins(step, day, args.nphases,
                                      args.bright, args.dark)
    print('Phases: ' + ', '.join(f'{lb}(daylight~{md:.2f}, n={m.sum()})'
                                  for lb, m, md in zip(labels, masks, mday)))

    # EV (spatial reliability) + all-neuron day/night remap score. The remap
    # contrast is the vital lo/hi machinery with vit=daylight, lo=dark, hi=bright.
    ev = _ev(X, flatbin, nbins)
    vit = {'daynight': day}
    allr, remap = _allneuron_remap(X, flatbin, vit, nbins, args.dark, args.bright,
                                   args.min_visits)
    _ev_band = tuple(float(x) for x in args.ev_band.split(','))
    _remap_band = tuple(float(x) for x in args.remap_band.split(','))
    selected, reason, ev_floor = _select(ev, remap, args.top_n, args.select,
                                         args.remap_ev_pct, _ev_band, _remap_band,
                                         min_ev=args.ev_floor)
    print(f'Selected {len(selected)} neurons (mode={args.select}, EV floor '
          f'{ev_floor:.2f}):\n  ' + '\n  '.join(
              f'n{i}[{reason[i]}] EV={ev[i]:.2f} remap={remap[i]:.2f}'
              for i in selected))

    X_sel = X[:, selected]
    tc_full, occ_full = _maps(flatbin, X_sel, nbins)
    phase_tc, phase_occ = _phase_maps(X_sel, flatbin, masks, nbins)

    # day/night contrast stats (shape corr + gain + within-bin shuffle null)
    print(f'conditioning day/night (day>= {args.bright}, night<= {args.dark}, '
          f'{args.n_shuffle} shuffles) ...')
    cond = {'daynight': _condition(X_sel, flatbin, day, nbins, args.dark,
                                   args.bright, args.min_visits, args.n_shuffle, rng)}

    try:
        from tuning_curve import _load_world_meta
        world_meta = _load_world_meta(args.data)
    except Exception as e:
        print(f'(world_meta not loaded: {e})')
        world_meta = None
    results = dict(layer=args.layer, area=list(area), selected=selected,
                   reason=reason, ev=ev, remap=remap, ev_floor=ev_floor,
                   tc_full=tc_full, occ_full=occ_full,
                   phase_labels=labels, phase_daylight=mday,
                   phase_tc=phase_tc, phase_occ=phase_occ,
                   bright=args.bright, dark=args.dark, cond=cond, args=vars(args),
                   world_meta=world_meta)
    pkl = save_dir / 'daynight_place_fields_results.pkl'
    with open(pkl, 'wb') as f:
        pickle.dump(results, f)
    print(f'Saved {pkl}')

    plot(results, str(save_dir), args.fmt, world_bg=args.world_bg, native=args.native,
         tuning_cmap=args.tuning_cmap)
    plot_population(results, str(save_dir), args.fmt)
    print_summary(results)

    try:
        from run_info import log_run_info
        log_run_info(str(save_dir), 'daynight_place_fields', vars(args), [str(pkl)])
    except Exception as e:
        print(f'(run_info not logged: {e})')


if __name__ == '__main__':
    main()
