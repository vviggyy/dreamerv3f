"""Vital-conditioned spatial place fields: does internal (vital) state remap the
world model's spatial code?

Hypothesis (model analog of context-dependent / "splitter" place cells): a
neuron's spatial tuning reorganizes depending on the agent's interoceptive state.
We test this by taking the top-EV spatially-tuned `dyn/deter` neurons and, for
each, building its 2D place field CONDITIONED on a vital being low vs high, then
asking whether the field changes.

Two effects, kept distinct (classic hippocampal distinction):
  * GLOBAL remapping (shape) -- the field moves / changes shape. Measured by the
    spatial correlation of the low-vs-high maps over CO-VISITED bins (gain-
    invariant). Low r = remapping. THIS is the strong claim: it can't be explained
    by the neuron merely tracking the on-screen status bar (that's a gain change).
  * RATE remapping (gain) -- same shape, scaled up/down with the vital. Measured by
    the ratio of mean activation (high vs low) over co-visited bins.

Significance: splitting data into low/high always makes two maps differ from
sampling noise. Null = shuffle the low/high labels WITHIN each position bin
(preserves occupancy, breaks the vital<->activation link), recompute the two maps,
and collect the correlation. Real r below the null percentile => genuine
vital-dependent remapping beyond sampling.

Conditioning vitals default to food + drink (health is near-constant; see
vital_dynamics distributions). Needs a trajectory set with BOTH `dyn/deter` and
per-step `vital_*` (e.g. a light_activations eval). NOT vitals_only.

Usage:
  MPLBACKEND=Agg python dreamerv3/vital_place_fields.py \
      --data ./logdir/my_run/trajectories_acts \
      --save ./logdir/my_run/vital_place_fields \
      --layer dyn/deter --vitals food,drink --top_n 10

  # replot (new top_n / lo / hi are NOT re-derivable; only cosmetic re-render):
  python dreamerv3/vital_place_fields.py --from_pkl ./.../vital_place_fields_results.pkl
"""
import argparse
import glob
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plot_presentable as pp

# Field-shape classification thresholds (gating vs relocation)
STRENGTH_FLOOR = 0.35   # weaker condition's peak < this x stronger's -> "gated off"
SHIFT_THRESH = 3.0      # peak/COM shift (tiles) above which a field counts as moved
FIELD_SMOOTH = 1.0      # gaussian sigma (tiles) before peak/COM location

VITAL_COLORS = {'health': '#d62728', 'food': '#2ca02c',
                'drink': '#1f77b4', 'energy': '#ff7f0e'}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _load(data_dir, layer, vitals):
    """Stream per-episode: concatenate deter (N,D), pos (N,2), vitals {v:(N,)}.

    Reads act/<layer> (falls back to top-level 'deter' for dyn/deter). Only loads
    the one layer + pos + vitals per episode, so RAM ~ one episode."""
    act_key = f'act/{layer}'
    files = sorted(glob.glob(str(Path(data_dir) / 'episode_*.pkl')))
    if not files:
        raise SystemExit(f'No episode_*.pkl in {data_dir}')
    X, pos = [], []
    vit = {v: [] for v in vitals}
    for f in files:
        with open(f, 'rb') as fh:
            ep = pickle.load(fh)
        a = ep.get(act_key)
        if a is None and layer == 'dyn/deter':
            a = ep.get('deter')
        p = ep.get('player_pos')
        if a is None or p is None or len(a) == 0:
            continue
        if any(f'vital_{v}' not in ep for v in vitals):
            continue
        a = np.asarray(a, np.float32)
        if a.ndim > 2:
            a = a.reshape(len(a), -1)
        p = np.asarray(p, np.float32)
        T = min(len(a), len(p), *[len(ep[f'vital_{v}']) for v in vitals])
        X.append(a[:T]); pos.append(p[:T])
        for v in vitals:
            vit[v].append(np.asarray(ep[f'vital_{v}'])[:T].astype(int))
        del ep
    if not X:
        raise SystemExit('No episodes with both activations and vitals. '
                         'Needs a light_activations (or full) eval, not vitals_only.')
    X = np.concatenate(X); pos = np.concatenate(pos)
    vit = {v: np.concatenate(a) for v, a in vit.items()}
    print(f'Loaded {len(X)} timesteps, {X.shape[1]} {layer} units, '
          f'{len(files)} episode files')
    return X, pos, vit


def _area(data_dir, pos):
    meta = Path(data_dir) / 'trajectory_meta.json'
    if meta.exists():
        a = json.loads(meta.read_text()).get('area')
        if a:
            return int(a[0]), int(a[1])
    return int(pos[:, 0].max()) + 1, int(pos[:, 1].max()) + 1


# ---------------------------------------------------------------------------
# Binned-mean place fields
# ---------------------------------------------------------------------------
def _flatbin(pos, area):
    bx = np.clip(pos[:, 0].astype(int), 0, area[0] - 1)
    by = np.clip(pos[:, 1].astype(int), 0, area[1] - 1)
    return bx * area[1] + by  # flat index into (area_x, area_y)


def _maps(flatbin, vals, nbins):
    """vals (M,k) -> (mean (nbins,k) with NaN on unvisited, count (nbins,))."""
    count = np.bincount(flatbin, minlength=nbins).astype(float)
    s = np.zeros((nbins, vals.shape[1]), np.float64)
    np.add.at(s, flatbin, vals)
    mean = np.full((nbins, vals.shape[1]), np.nan)
    nz = count > 0
    mean[nz] = s[nz] / count[nz, None]
    return mean, count


def _ev(X, flatbin, nbins):
    """EV reliability per neuron from the full (unconditioned) field."""
    tc, _ = _maps(flatbin, X, nbins)              # (nbins, D)
    pred = np.nan_to_num(tc[flatbin])             # (N, D)
    var_real = X.var(0)
    var_res = (X - pred).var(0)
    ev = 1.0 - var_res / np.where(var_real > 0, var_real, 1.0)
    ev[var_real == 0] = 0.0
    return np.clip(ev, -1, 1)


def _corr_covis(m_lo, m_hi, covis):
    """Pearson r between two flat maps over co-visited bins (per neuron)."""
    a, b = m_lo[covis], m_hi[covis]               # (nco, k)
    a = a - a.mean(0); b = b - b.mean(0)
    num = (a * b).sum(0)
    den = np.sqrt((a * a).sum(0) * (b * b).sum(0))
    return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)


# ---------------------------------------------------------------------------
# Per-vital conditioning + shuffle null
# ---------------------------------------------------------------------------
def _condition(X_sel, flatbin, vital, nbins, lo, hi, min_visits, n_shuffle, rng):
    """For one vital, selected neurons X_sel (N,k): low/high maps, co-visited
    mask, shape corr + gain, and a within-bin-shuffle null on the corr."""
    is_lo = vital <= lo
    is_hi = vital >= hi
    m_lo, c_lo = _maps(flatbin[is_lo], X_sel[is_lo], nbins)
    m_hi, c_hi = _maps(flatbin[is_hi], X_sel[is_hi], nbins)
    covis = (c_lo >= min_visits) & (c_hi >= min_visits)
    r_real = _corr_covis(m_lo, m_hi, covis)
    with np.errstate(invalid='ignore'):
        gain = np.nanmean(np.where(covis[:, None], m_hi, np.nan), 0) / \
               np.nanmean(np.where(covis[:, None], m_lo, np.nan), 0)

    # within-bin shuffle of lo/hi labels (preserves occupancy)
    sel = is_lo | is_hi
    sb = flatbin[sel]; svals = X_sel[sel]
    shi = is_hi[sel].copy()
    # group member row-indices by bin (once)
    order = np.argsort(sb, kind='stable')
    sb_sorted = sb[order]
    bin_edges = np.searchsorted(sb_sorted, np.arange(nbins + 1))
    groups = [order[bin_edges[b]:bin_edges[b + 1]] for b in range(nbins)]
    groups = [g for g in groups if len(g) > 1]    # bins with 1 member can't shuffle
    null = np.full((n_shuffle, X_sel.shape[1]), np.nan)
    for s in range(n_shuffle):
        sh = shi.copy()
        for g in groups:
            sh[g] = rng.permutation(sh[g])
        ml, _ = _maps(sb[~sh], svals[~sh], nbins)
        mh, _ = _maps(sb[sh], svals[sh], nbins)
        null[s] = _corr_covis(ml, mh, covis)
    # one-sided p: real r in the LOW tail of the null (more remapped than chance)
    p = np.nanmean(null <= r_real[None, :], 0)
    null_med = np.nanmedian(null, 0)              # effect-size reference
    return dict(m_lo=m_lo, m_hi=m_hi, c_lo=c_lo, c_hi=c_hi, covis=covis,
                r=r_real, gain=gain, null=null, p=p, null_med=null_med,
                n_lo=int(is_lo.sum()), n_hi=int(is_hi.sum()),
                lo=lo, hi=hi)


def _allneuron_remap(X, flatbin, vit, nbins, lo, hi, min_visits):
    """Real (no-shuffle) low-vs-high map correlation for EVERY neuron, per vital.

    Returns (allr {vital:(D,) r over co-visited bins}, remap_score (D,) = mean
    (1-r) across vitals). Cheap -- used for remap-based selection + the population
    view. Noise cells also score high here (random fields decorrelate), so remap
    selection is gated on EV (see _select) and confirmed by the shuffle null."""
    allr = {}
    for v, vv in vit.items():
        lom, him = vv <= lo, vv >= hi
        m_lo, c_lo = _maps(flatbin[lom], X[lom], nbins)
        m_hi, c_hi = _maps(flatbin[him], X[him], nbins)
        covis = (c_lo >= min_visits) & (c_hi >= min_visits)
        allr[v] = _corr_covis(m_lo, m_hi, covis)
    remap = np.nanmean(np.stack([1.0 - allr[v] for v in vit]), 0)
    return allr, remap


def _pctrank(x):
    """Percentile rank in [0,1] (NaN -> 0). Scale-free, for conjunctive scoring."""
    x = np.nan_to_num(np.asarray(x, float), nan=-np.inf)
    order = np.argsort(x, kind='stable')
    rank = np.empty(len(x)); rank[order] = np.arange(len(x))
    return rank / max(len(x) - 1, 1)


def _select(ev, remap, top_n, mode, remap_ev_pct, ev_band=(0.5, 0.8),
            remap_band=(0.4, 0.8)):
    """Pick neurons to display. Returns (indices, reason-per-index dict).

    mode 'ev'    -> top EV (stable spatial baseline).
    mode 'remap' -> top remap score among EV-eligible neurons (candidate
                    remappers; EV floor keeps pure-noise cells out).
    mode 'conj'  -> JOINT top-right: high EV AND high remap, ranked by the product
                    of EV & remap percentile ranks (candidate conjunctive/
                    "splitter" place x vital cells). Still EV-floor gated.
    mode 'both'  -> half from each, unioned (dedup, EV-pick fills any shortfall)."""
    ev_floor = np.nanpercentile(ev, remap_ev_pct)
    eligible = np.where(ev >= ev_floor, remap, -np.inf)       # gate remap on EV
    ev_rank = np.argsort(ev)[::-1]
    remap_rank = np.argsort(np.nan_to_num(eligible, nan=-np.inf))[::-1]
    reason = {}
    if mode == 'ev':
        picks = list(ev_rank[:top_n])
        for i in picks: reason[int(i)] = 'ev'
    elif mode == 'remap':
        picks = list(remap_rank[:top_n])
        for i in picks: reason[int(i)] = 'remap'
    elif mode == 'conj':
        conj = _pctrank(ev) * _pctrank(remap)                 # high only if BOTH high
        conj = np.where(ev >= ev_floor, conj, -np.inf)        # keep the noise gate
        picks = list(np.argsort(conj)[::-1][:top_n])
        for i in picks: reason[int(i)] = 'conj'
    elif mode == 'medium':
        lo_e, hi_e = ev_band; lo_r, hi_r = remap_band
        band = ((ev > lo_e) & (ev < hi_e) & (remap > lo_r) & (remap < hi_r)
                & np.isfinite(ev) & np.isfinite(remap))
        cand = np.where(band)[0]
        ce, cr = (lo_e + hi_e) / 2, (lo_r + hi_r) / 2   # rank by centrality in band
        order = np.argsort((ev[cand] - ce) ** 2 + (remap[cand] - cr) ** 2)
        picks = list(cand[order][:top_n])
        for i in picks: reason[int(i)] = 'medium'
    else:  # both
        picks = []
        ne = top_n // 2
        for i in remap_rank[:top_n - ne]:
            picks.append(int(i)); reason[int(i)] = 'remap'
        for i in ev_rank:
            if len(picks) >= top_n: break
            if int(i) not in reason:
                picks.append(int(i)); reason[int(i)] = 'ev'
    return [int(i) for i in picks[:top_n]], reason, float(ev_floor)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def _grid(flat, area):
    return np.asarray(flat, float).reshape(area[0], area[1])


def _occ_grid(count, area):
    g = _grid(count, area)
    return np.where(g > 0, g, np.nan)


def _occ_prob_grid(count, area):
    """Occupancy as a visit-PROBABILITY grid (count / total visits), NaN on
    unvisited. Per-condition normalization makes coverage comparable across
    conditions of different total length (vital lo vs hi, day vs night) — raw
    counts would otherwise just reflect how many steps fell in that condition,
    not where the agent actually went."""
    c = np.asarray(count, float)
    tot = c.sum()
    g = _grid(c / tot if tot > 0 else c, area)
    return np.where(g > 0, g, np.nan)


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


def _plot_one_vital(results, v, save_dir, fmt, world_img=None, native=False,
                    tuning_cmap=False):
    """One figure for a single vital: occupancy (lo/hi, probability-normalized) +
    per-neuron full/lo/hi place-field maps for THIS vital's own selected neurons."""
    area = tuple(results['area'])
    ev = results['ev']
    pv = results['per_vital'][v]
    sel = pv['selected']; reason = pv['reason']; cv = pv['cond']
    remap = np.asarray(pv['remap'])
    nmap = 3                               # full + lo + hi
    nrow, ncol = len(sel) + 1, nmap + 1    # + a thin trailing colorbar column
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

    def show(a, grid, vmin, vmax, cmap='viridis'):
        # With a world background, unvisited (NaN) tiles are left transparent so
        # the faint Crafter terrain shows through (matches occupancy_fullset /
        # the EV-sorted tuning maps); otherwise they stay dark grey.
        # tuning_cmap=True uses the exact black->red->orange tuning/occupancy
        # colormap for ALL panels (occupancy header + neuron maps) instead of
        # viridis/magma, so these figures match the poster occupancy & tuning.
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
        a.set_xticks([]); a.set_yticks([])
        return im

    def cbar(im, r, label):
        cb = fig.colorbar(im, cax=ax[r, -1])
        cb.ax.tick_params(labelsize=6)
        cb.set_label(label, fontsize=7)

    # occupancy header: probability-normalized (count/total) so lo vs hi coverage
    # is comparable despite differing numbers of lo/hi steps. lo/hi share one
    # scale; the all-steps panel keeps its own autoscale.
    g_lo = _occ_prob_grid(cv['c_lo'], area)
    g_hi = _occ_prob_grid(cv['c_hi'], area)
    pooled = np.concatenate([g[np.isfinite(g)] for g in (g_lo, g_hi)
                             if np.isfinite(g).any()]) \
        if (np.isfinite(g_lo).any() or np.isfinite(g_hi).any()) else np.array([])
    occ_vmax = np.percentile(pooled, 98) if pooled.size else 1
    full_occ = _occ_prob_grid(results['occ_full'], area)
    ff = full_occ[np.isfinite(full_occ)]
    show(ax[0, 0], full_occ, 0, np.percentile(ff, 98) if ff.size else None, 'magma')
    ax[0, 0].set_title('occupancy (all)', fontsize=8)
    ax[0, 0].set_ylabel('OCCUPANCY', fontsize=8)
    show(ax[0, 1], g_lo, 0, occ_vmax, 'magma')
    ax[0, 1].set_title(f'{v}_lo', fontsize=8, color=VITAL_COLORS.get(v, 'k'))
    im_occ = show(ax[0, 2], g_hi, 0, occ_vmax, 'magma')
    ax[0, 2].set_title(f'{v}_hi', fontsize=8, color=VITAL_COLORS.get(v, 'k'))
    cbar(im_occ, 0, 'visit prob (lo/hi, shared)')

    for r, nidx in enumerate(sel, start=1):
        full_map = _grid(pv['tc_full'][:, r - 1], area)
        finite = full_map[np.isfinite(full_map)]
        vmin = np.percentile(finite, 2) if finite.size else 0
        vmax = np.percentile(finite, 98) if finite.size else 1
        show(ax[r, 0], full_map, vmin, vmax)
        rsn = reason.get(nidx, '')
        ax[r, 0].set_ylabel(f'n{nidx} [{rsn}]\nEV={ev[nidx]:.2f} '
                            f'rmp={remap[nidx]:.2f}', fontsize=7)
        show(ax[r, 1], _grid(cv['m_lo'][:, r - 1], area), vmin, vmax)
        im = show(ax[r, 2], _grid(cv['m_hi'][:, r - 1], area), vmin, vmax)
        pr, pvp = cv['r'][r - 1], cv['p'][r - 1]
        nm = cv.get('null_med', np.full_like(cv['r'], np.nan))[r - 1]
        star = '*' if pvp < 0.05 else ''
        ax[r, 2].set_title(
            f'{v} r={pr:.2f}{star}\n null={nm:.2f} g={cv["gain"][r-1]:.2f}',
            fontsize=6.5, color=VITAL_COLORS.get(v, 'k'))
        cbar(im, r, 'mean act')

    sel_mode = results.get('args', {}).get('select', '')
    fig.suptitle(
        f'{v}-conditioned place fields ({results["layer"]}, '
        f'top-{len(sel)} by {sel_mode})\n'
        'low r (*) = field remaps with vital (shape); gain = hi/lo activation ratio',
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = (Path(save_dir) /
         f'vital_place_fields_{results["layer"].replace("/", "_")}_{v}.{fmt}')
    fig.savefig(p, bbox_inches='tight', pad_inches=0.3, dpi=130); plt.close(fig)
    print(f'Saved {p}')


def plot(results, save_dir, fmt='svg', world_bg=True, native=False,
         tuning_cmap=False):
    """One place-field figure PER vital (each with its own selected neurons)."""
    world_img = _get_world_img(results, frame='native' if native else 'tuning') \
        if world_bg else None
    for v in results['vitals']:
        _plot_one_vital(results, v, save_dir, fmt, world_img=world_img,
                        native=native, tuning_cmap=tuning_cmap)


def plot_population(results, save_dir, fmt='svg'):
    """Population view PER vital: EV vs that vital's remap-score scatter (selected
    highlighted) + its remap-score histogram over all neurons. One r per vital —
    no averaging across vitals — so remapping is vital-specific."""
    ev = np.asarray(results['ev'])
    for v in results['vitals']:
        pv = results['per_vital'][v]
        remap = np.asarray(pv['remap'])
        sel = pv['selected']; reason = pv['reason']
        fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4.2))
        fin = np.isfinite(ev) & np.isfinite(remap)
        # Monochrome: all neurons one grey; the selected (shown) neurons kept
        # visible via a darker grey + black edge, but NOT color-coded.
        a0.scatter(ev[fin], remap[fin], s=4, c='0.7', alpha=0.4, label='all neurons')
        sel_fin = [i for i in sel if np.isfinite(ev[i]) and np.isfinite(remap[i])]
        if sel_fin:
            a0.scatter(ev[sel_fin], remap[sel_fin], s=42, c='0.35', edgecolor='k',
                       label='shown', zorder=3)
        if np.isfinite(pv.get('ev_floor', np.nan)):
            a0.axvline(pv['ev_floor'], color='0.4', ls='--', lw=1,
                       label='remap EV floor')
        a0.set_xlabel('EV (spatial reliability)')
        a0.set_ylabel(f'remap score  (1 − r,  {v} low vs high)')
        a0.set_title(f'Spatial reliability vs {v}-remapping')
        a0.legend(fontsize=8, frameon=False); a0.grid(alpha=0.3)
        a1.hist(remap[np.isfinite(remap)], bins=60, color='0.6')
        a1.set_xlabel(f'remap score (1 − r, {v})'); a1.set_ylabel('neurons')
        a1.set_title(f'{v} remap-score distribution (all neurons)')
        a1.grid(alpha=0.3)
        fig.suptitle(f'{results["layer"]}: does {v} remap the spatial code?',
                     fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        p = (Path(save_dir) /
             f'vital_place_fields_population_{results["layer"].replace("/", "_")}'
             f'_{v}.{fmt}')
        fig.savefig(p, bbox_inches='tight', pad_inches=0.3, dpi=130); plt.close(fig)
        print(f'Saved {p}')


def print_summary(results):
    print('\n=== remapping summary (shape corr low-vs-high; null=null-median r) ===')
    for v in results['vitals']:
        pv = results['per_vital'][v]
        cv = pv['cond']; sel = pv['selected']; reason = pv['reason']
        print(f'[{v}]  n_lo={cv["n_lo"]} n_hi={cv["n_hi"]}  '
              f'(lo<={cv["lo"]}, hi>={cv["hi"]})')
        nmed = cv.get('null_med', np.full_like(cv['r'], np.nan))
        for r, nidx in enumerate(sel):
            sig = '*' if cv['p'][r] < 0.05 else ' '
            rsn = reason.get(nidx, '')
            print(f'   n{nidx:<4d}[{rsn:>5s}] EV={results["ev"][nidx]:.2f}  '
                  f'r={cv["r"][r]:+.2f} (null {nmed[r]:+.2f}) '
                  f'p={cv["p"][r]:.3f}{sig}  gain={cv["gain"][r]:.2f}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', help='trajectory dir with act/<layer> + vital_*')
    ap.add_argument('--save', default='./vital_place_fields')
    ap.add_argument('--layer', default='dyn/deter')
    ap.add_argument('--vitals', default='food,drink')
    ap.add_argument('--top_n', type=int, default=10, help='neurons to show')
    ap.add_argument('--select', default='both',
                    choices=['ev', 'remap', 'conj', 'both', 'medium'],
                    help='ev=top spatial (stable baseline), remap=top vital-remappers, '
                    'conj=jointly high EV AND remap (candidate splitter cells), both, '
                    'medium=intermediate band (EV & remap both mid; --ev_band/--remap_band)')
    ap.add_argument('--ev_band', default='0.5,0.8',
                    help='[--select medium] EV window "lo,hi" (default 0.5,0.8)')
    ap.add_argument('--remap_band', default='0.4,0.8',
                    help='[--select medium] remap window "lo,hi" (default 0.4,0.8)')
    ap.add_argument('--remap_ev_pct', type=float, default=75.0,
                    help='remap candidates must have EV >= this percentile of EV '
                    '(gates out weakly-spatial vital-gain cells; raise toward 90 if '
                    'remap picks still look non-spatial)')
    ap.add_argument('--lo', type=int, default=3, help='vital<=lo => low bin')
    ap.add_argument('--hi', type=int, default=7, help='vital>=hi => high bin')
    ap.add_argument('--min_visits', type=int, default=5,
                    help='min visits per bin in BOTH conditions to count as co-visited')
    ap.add_argument('--n_shuffle', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--fmt', default='svg', choices=['svg', 'png', 'pdf'],
                    help='figure format (svg=vector for Illustrator; png=quick view)')
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
    pp.maybe_apply(args, small_figsize=False)  # Tier-1 style; no-op unless --presentable

    if args.from_pkl:
        with open(args.from_pkl, 'rb') as f:
            results = pickle.load(f)
        sd = args.save if args.save != './vital_place_fields' else str(Path(args.from_pkl).parent)
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
    vitals = [v.strip() for v in args.vitals.split(',') if v.strip()]
    rng = np.random.RandomState(args.seed)

    X, pos, vit = _load(args.data, args.layer, vitals)
    area = _area(args.data, pos)
    nbins = area[0] * area[1]
    flatbin = _flatbin(pos, area)
    print(f'Area {area}, {nbins} bins')

    # EV (spatial reliability) + PER-VITAL remap r (one r per vital, NOT averaged
    # across vitals — so selection + remap score are vital-specific).
    ev = _ev(X, flatbin, nbins)
    allr, _ = _allneuron_remap(X, flatbin, vit, nbins, args.lo, args.hi,
                               args.min_visits)
    _ev_band = tuple(float(x) for x in args.ev_band.split(','))
    _remap_band = tuple(float(x) for x in args.remap_band.split(','))
    # all-steps occupancy (visit counts) — depends only on position, shared by all
    occ_full = np.bincount(flatbin, minlength=nbins).astype(float)

    per_vital = {}
    for v in vitals:
        remap_v = 1.0 - np.asarray(allr[v])       # this vital's remap score (D,)
        selected, reason, ev_floor = _select(ev, remap_v, args.top_n, args.select,
                                             args.remap_ev_pct, _ev_band, _remap_band)
        print(f'[{v}] selected {len(selected)} neurons (mode={args.select}, EV '
              f'floor {ev_floor:.2f}): ' + ', '.join(
                  f'n{i}[{reason[i]}]' for i in selected))
        X_sel = X[:, selected]
        tc_full, _ = _maps(flatbin, X_sel, nbins)
        print(f'conditioning on {v} (lo<={args.lo}, hi>={args.hi}, '
              f'{args.n_shuffle} shuffles) ...')
        cond_v = _condition(X_sel, flatbin, vit[v], nbins, args.lo, args.hi,
                            args.min_visits, args.n_shuffle, rng)
        per_vital[v] = dict(remap=remap_v, selected=selected, reason=reason,
                            ev_floor=ev_floor, tc_full=tc_full, cond=cond_v)

    try:
        from tuning_curve import _load_world_meta
        world_meta = _load_world_meta(args.data)
    except Exception as e:
        print(f'(world_meta not loaded: {e})')
        world_meta = None
    results = dict(layer=args.layer, area=list(area), vitals=vitals, ev=ev,
                   occ_full=occ_full, per_vital=per_vital, args=vars(args),
                   world_meta=world_meta)
    pkl = save_dir / 'vital_place_fields_results.pkl'
    with open(pkl, 'wb') as f:
        pickle.dump(results, f)
    print(f'Saved {pkl}')

    plot(results, str(save_dir), args.fmt, world_bg=args.world_bg, native=args.native,
         tuning_cmap=args.tuning_cmap)
    plot_population(results, str(save_dir), args.fmt)
    print_summary(results)

    try:
        from run_info import log_run_info
        log_run_info(str(save_dir), 'vital_place_fields', vars(args), [str(pkl)])
    except Exception as e:
        print(f'(run_info not logged: {e})')


if __name__ == '__main__':
    main()
