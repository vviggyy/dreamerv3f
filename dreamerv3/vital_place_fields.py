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


def _select(ev, remap, top_n, mode, remap_ev_pct):
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


def plot(results, save_dir, fmt='svg'):
    area = tuple(results['area'])
    vitals = results['vitals']
    sel = results['selected']
    ev = results['ev']
    nbins = area[0] * area[1]
    cols = ['full'] + [f'{v}_{lab}' for v in vitals for lab in ('lo', 'hi')]
    nrow, ncol = len(sel) + 1, len(cols)
    fig, ax = plt.subplots(nrow, ncol, figsize=(2.1 * ncol, 2.1 * nrow),
                           squeeze=False)

    def show(a, grid, vmin, vmax, cmap='viridis'):
        cm = plt.get_cmap(cmap).copy(); cm.set_bad('#4d4d4d')
        a.imshow(grid.T, origin='lower', cmap=cm, vmin=vmin, vmax=vmax,
                 interpolation='nearest')
        a.set_xticks([]); a.set_yticks([])

    # occupancy header row
    show(ax[0, 0], _occ_grid(results['occ_full'], area), None, None, 'magma')
    ax[0, 0].set_title('occupancy', fontsize=8)
    ax[0, 0].set_ylabel('OCCUPANCY', fontsize=8)
    ci = 1
    for v in vitals:
        cv = results['cond'][v]
        for lab, ckey in (('lo', 'c_lo'), ('hi', 'c_hi')):
            show(ax[0, ci], _occ_grid(cv[ckey], area), None, None, 'magma')
            ax[0, ci].set_title(f'{v}_{lab}', fontsize=8,
                                color=VITAL_COLORS.get(v, 'k'))
            ci += 1

    for r, nidx in enumerate(sel, start=1):
        full_map = _grid(results['tc_full'][:, r - 1], area)
        finite = full_map[np.isfinite(full_map)]
        vmin = np.percentile(finite, 2) if finite.size else 0
        vmax = np.percentile(finite, 98) if finite.size else 1
        show(ax[r, 0], full_map, vmin, vmax)
        rsn = results.get('reason', {}).get(nidx, '')
        rmp = results.get('remap', {})
        rmp_v = rmp[nidx] if len(rmp) else np.nan
        ax[r, 0].set_ylabel(f'n{nidx} [{rsn}]\nEV={ev[nidx]:.2f} rmp={rmp_v:.2f}',
                            fontsize=7)
        ci = 1
        for v in vitals:
            cv = results['cond'][v]
            for mk in ('m_lo', 'm_hi'):
                show(ax[r, ci], _grid(cv[mk][:, r - 1], area), vmin, vmax)
                ci += 1
            # annotate shape corr vs null-median (effect size) + sig on hi panel
            pr, pp = cv['r'][r - 1], cv['p'][r - 1]
            nm = cv.get('null_med', np.full_like(cv['r'], np.nan))[r - 1]
            star = '*' if pp < 0.05 else ''
            ax[r, ci - 1].set_title(
                f'{v} r={pr:.2f}{star}\n null={nm:.2f} g={cv["gain"][r-1]:.2f}',
                fontsize=6.5, color=VITAL_COLORS.get(v, 'k'))

    fig.suptitle(
        f'Vital-conditioned place fields ({results["layer"]}, top-{len(sel)} EV)\n'
        'low r (*) = field remaps with vital (shape); gain = hi/lo activation ratio',
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = (Path(save_dir) /
         f'vital_place_fields_{results["layer"].replace("/", "_")}.{fmt}')
    fig.savefig(p, bbox_inches='tight', dpi=130); plt.close(fig)
    print(f'Saved {p}')


def plot_population(results, save_dir, fmt='svg'):
    """Population view: EV vs remap-score scatter (selected highlighted) +
    remap-score histogram over all neurons. Answers 'does ANY cell remap, and
    how much does the population?' independent of which cells we chose to show."""
    if 'remap' not in results or not len(results['remap']):
        return
    ev = np.asarray(results['ev']); remap = np.asarray(results['remap'])
    sel = results['selected']; reason = results.get('reason', {})
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4.2))
    fin = np.isfinite(ev) & np.isfinite(remap)
    a0.scatter(ev[fin], remap[fin], s=4, c='0.7', alpha=0.4, label='all neurons')
    for rsn, col in (('ev', '#1f77b4'), ('remap', '#d62728'), ('conj', '#2ca02c')):
        idx = [i for i in sel if reason.get(i) == rsn]
        if idx:
            a0.scatter(ev[idx], remap[idx], s=42, c=col, edgecolor='k',
                       label=f'shown ({rsn})', zorder=3)
    if np.isfinite(results.get('ev_floor', np.nan)):
        a0.axvline(results['ev_floor'], color='0.4', ls='--', lw=1,
                   label='remap EV floor')
    a0.set_xlabel('EV (spatial reliability)')
    a0.set_ylabel('remap score  (mean 1 − r,  low vs high)')
    a0.set_title('Spatial reliability vs vital-remapping')
    a0.legend(fontsize=8, frameon=False); a0.grid(alpha=0.3)
    a1.hist(remap[np.isfinite(remap)], bins=60, color='#8888cc')
    a1.set_xlabel('remap score (mean 1 − r)'); a1.set_ylabel('neurons')
    a1.set_title('Remap-score distribution (all neurons)')
    a1.grid(alpha=0.3)
    fig.suptitle(f'{results["layer"]}: does vital state remap the spatial code?',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = (Path(save_dir) /
         f'vital_place_fields_population_{results["layer"].replace("/", "_")}.{fmt}')
    fig.savefig(p, bbox_inches='tight', dpi=130); plt.close(fig)
    print(f'Saved {p}')


def print_summary(results):
    print('\n=== remapping summary (shape corr low-vs-high; null=null-median r) ===')
    for v in results['vitals']:
        cv = results['cond'][v]
        print(f'[{v}]  n_lo={cv["n_lo"]} n_hi={cv["n_hi"]}  '
              f'(lo<={cv["lo"]}, hi>={cv["hi"]})')
        nmed = cv.get('null_med', np.full_like(cv['r'], np.nan))
        for r, nidx in enumerate(results['selected']):
            sig = '*' if cv['p'][r] < 0.05 else ' '
            rsn = results.get('reason', {}).get(nidx, '')
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
                    choices=['ev', 'remap', 'conj', 'both'],
                    help='ev=top spatial (stable baseline), remap=top vital-remappers, '
                    'conj=jointly high EV AND remap (candidate splitter cells), both')
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
    args = ap.parse_args()

    if args.from_pkl:
        with open(args.from_pkl, 'rb') as f:
            results = pickle.load(f)
        sd = args.save if args.save != './vital_place_fields' else str(Path(args.from_pkl).parent)
        Path(sd).mkdir(parents=True, exist_ok=True)
        plot(results, sd, args.fmt); plot_population(results, sd, args.fmt)
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

    # EV (spatial reliability) + all-neuron remap score (selection + pop view)
    ev = _ev(X, flatbin, nbins)
    allr, remap = _allneuron_remap(X, flatbin, vit, nbins, args.lo, args.hi,
                                   args.min_visits)
    selected, reason, ev_floor = _select(ev, remap, args.top_n, args.select,
                                         args.remap_ev_pct)
    print(f'Selected {len(selected)} neurons (mode={args.select}, EV floor '
          f'{ev_floor:.2f}):\n  ' + '\n  '.join(
              f'n{i}[{reason[i]}] EV={ev[i]:.2f} remap={remap[i]:.2f}'
              for i in selected))

    X_sel = X[:, selected]
    tc_full, occ_full = _maps(flatbin, X_sel, nbins)

    cond = {}
    for v in vitals:
        print(f'conditioning on {v} (lo<={args.lo}, hi>={args.hi}, '
              f'{args.n_shuffle} shuffles) ...')
        cond[v] = _condition(X_sel, flatbin, vit[v], nbins, args.lo, args.hi,
                             args.min_visits, args.n_shuffle, rng)

    results = dict(layer=args.layer, area=list(area), vitals=vitals,
                   selected=selected, reason=reason, ev=ev, remap=remap,
                   ev_floor=ev_floor, tc_full=tc_full, occ_full=occ_full,
                   cond=cond, args=vars(args))
    pkl = save_dir / 'vital_place_fields_results.pkl'
    with open(pkl, 'wb') as f:
        pickle.dump(results, f)
    print(f'Saved {pkl}')

    plot(results, str(save_dir), args.fmt)
    plot_population(results, str(save_dir), args.fmt)
    print_summary(results)

    try:
        from run_info import log_run_info
        log_run_info(str(save_dir), 'vital_place_fields', vars(args), [str(pkl)])
    except Exception as e:
        print(f'(run_info not logged: {e})')


if __name__ == '__main__':
    main()
