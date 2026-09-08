"""
compare_scheme_mix.py — compare training progress across the replay-scheme
mixture sweep (run_TrainSchemeMix.sh).

Each scheme_mix run varies --agent.train_scheme_mix = weights (A, D, A') that
sum to 100, so the 10 runs live on a 3-axis simplex. This script reads each
run's scores.jsonl and produces, in one figure:

  LEFT  — a TERNARY (triangular) heatmap of final performance across the
          A / D / A' simplex (tricontourf over the 10 sampled mixtures, with
          each mixture scatter-labeled).
  RIGHT — smoothed performance-vs-training-step curves for all runs overlaid,
          so faster/earlier learners are visible.

Metric (--metric):
  achievements  exact # distinct achievements unlocked per episode
                (sum of the 22 episode/achievement_* flags)   [default]
  score         episode/score (Crafter reward proxy)
  crafter       rolling Crafter score (geom. mean of success rates, %)

Reuses plot_training.py's load_jsonl / records_to_series / smooth so parsing and
smoothing match training_progress.png exactly.

Usage:
  MPLBACKEND=Agg python dreamerv3/compare_scheme_mix.py \
      --logdirs './logdir/scheme_mix_*' \
      --save ./logdir/scheme_mix_comparison \
      --metric achievements --smooth 200
"""

import argparse
import glob
import pathlib
import re

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from plot_training import load_jsonl, records_to_series, smooth, fmt_millions

SQRT3_2 = np.sqrt(3) / 2.0
# Ternary vertices: A top, D bottom-left, A' bottom-right.
V_A = np.array([0.5, SQRT3_2])
V_D = np.array([0.0, 0.0])
V_AP = np.array([1.0, 0.0])


def bary_to_xy(a, d, ap):
    """Barycentric (a, d, ap) weights -> 2D cartesian in the A/D/A' triangle."""
    s = a + d + ap
    a, d, ap = a / s, d / s, ap / s
    return a * V_A + d * V_D + ap * V_AP


def parse_mix(logdir):
    """Read (A, D, A') weights + tag for a run from hyperparams.txt, falling
    back to the directory name (e.g. scheme_mix_A33D66)."""
    hp = pathlib.Path(logdir) / 'hyperparams.txt'
    tag = pathlib.Path(logdir).name.replace('scheme_mix_', '')
    if hp.exists():
        txt = hp.read_text()
        m = re.search(r'train_scheme_mix\s*=\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)', txt)
        if m:
            a, d, ap = (float(x) for x in m.groups())
            tm = re.search(r'tag=([A-Za-z0-9]+)', txt)
            if tm:
                tag = tm.group(1)
            return (a, d, ap), tag
    # Fallback: parse the tag itself (A33D66 / A100 / A33D33Ap33 ...).
    # Match Ap<num> first, strip it, then A<num> / D<num> (so "Ap" != "A").
    ap_m = re.search(r'Ap(\d+)', tag)
    ap = float(ap_m.group(1)) if ap_m else 0.0
    tag_noap = re.sub(r'Ap\d+', '', tag)
    a_m = re.search(r'A(\d+)', tag_noap)
    d_m = re.search(r'D(\d+)', tag_noap)
    a = float(a_m.group(1)) if a_m else 0.0
    d = float(d_m.group(1)) if d_m else 0.0
    if a + d + ap == 0:
        a = 100.0  # last-resort default
    return (a, d, ap), tag


def achievement_count_series(records):
    """Per-episode count of distinct achievements unlocked (sum of the 22
    episode/achievement_* binary flags), returned as (steps, counts)."""
    ach_keys = sorted({k for r in records for k in r
                       if k.startswith('episode/achievement_')})
    if not ach_keys:
        return np.array([]), np.array([])
    steps, counts = [], []
    for r in records:
        if 'step' not in r:
            continue
        steps.append(r['step'])
        counts.append(sum(float(r.get(k, 0.0)) for k in ach_keys))
    order = np.argsort(steps)
    return np.array(steps)[order], np.array(counts)[order]


def crafter_series(records, window):
    """Rolling Crafter score (%) vs step — mirrors plot_training.plot_crafter_score."""
    ach_keys = sorted({k for r in records for k in r
                       if k.startswith('episode/achievement_')})
    rec = [r for r in records if 'step' in r and ach_keys and ach_keys[0] in r]
    if not rec:
        return np.array([]), np.array([])
    steps = np.array([r['step'] for r in rec])
    succ = np.zeros((len(rec), len(ach_keys)))
    for j, key in enumerate(ach_keys):
        for i, r in enumerate(rec):
            succ[i, j] = r.get(key, 0.0)
    order = np.argsort(steps)
    steps, succ = steps[order], succ[order]
    eps = 1e-6
    w = min(max(window, 10), len(steps))
    out = np.full(len(steps), np.nan)
    for i in range(w - 1, len(steps)):
        rates = succ[i - w + 1:i + 1].mean(axis=0)
        out[i] = np.exp(np.mean(np.log(rates + eps))) * 100
    valid = ~np.isnan(out)
    return steps[valid], out[valid]


def get_series(records, metric, smooth_window):
    """Raw (steps, vals) for the chosen metric, before curve smoothing.
    crafter is already windowed; achievements/score are per-episode."""
    if metric == 'achievements':
        return achievement_count_series(records)
    if metric == 'score':
        return records_to_series(records, 'episode/score')
    if metric == 'crafter':
        return crafter_series(records, smooth_window)
    raise ValueError(f'unknown metric {metric}')


METRIC_LABEL = {
    'achievements': 'Achievements unlocked / episode',
    'score': 'Episode score (reward)',
    'crafter': 'Crafter score (%)',
}


def final_performance(steps, vals, frac):
    """Mean of the metric over the final `frac` fraction of episodes."""
    if len(vals) == 0:
        return np.nan
    n = max(1, int(len(vals) * frac))
    return float(np.mean(vals[-n:]))


def draw_ternary_heatmap(ax, runs, metric):
    """runs: list of dicts with 'mix' (a,d,ap), 'tag', 'perf'."""
    pts = np.array([bary_to_xy(*r['mix']) for r in runs])
    vals = np.array([r['perf'] for r in runs])

    import matplotlib.tri as mtri
    tri = mtri.Triangulation(pts[:, 0], pts[:, 1])
    tcf = ax.tricontourf(tri, vals, levels=14, cmap='viridis')
    cbar = plt.colorbar(tcf, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label(f'Final {METRIC_LABEL[metric]}', fontsize=9)

    # Triangle outline + light internal grid at 33/66 iso-lines.
    tri_xy = np.array([V_A, V_D, V_AP, V_A])
    ax.plot(tri_xy[:, 0], tri_xy[:, 1], color='k', lw=1.2)
    for f in (1 / 3, 2 / 3):
        # iso-a, iso-d, iso-ap lines
        ax.plot(*zip(bary_to_xy(f, 1 - f, 0), bary_to_xy(f, 0, 1 - f)),
                color='w', lw=0.4, alpha=0.5)
        ax.plot(*zip(bary_to_xy(1 - f, f, 0), bary_to_xy(0, f, 1 - f)),
                color='w', lw=0.4, alpha=0.5)
        ax.plot(*zip(bary_to_xy(1 - f, 0, f), bary_to_xy(0, 1 - f, f)),
                color='w', lw=0.4, alpha=0.5)

    # Sampled mixtures. Offset each value label TOWARD the triangle centroid so
    # corner points don't collide with the (outside) corner descriptors.
    centroid = (V_A + V_D + V_AP) / 3.0
    ax.scatter(pts[:, 0], pts[:, 1], s=45, c='white', edgecolors='k',
               linewidths=1.0, zorder=5)
    for r, pt in zip(runs, pts):
        d = centroid - pt
        n = np.linalg.norm(d)
        off = (d / n) * 16 if n > 1e-6 else np.array([0, 12])  # points -> offset px
        ax.annotate(f"{r['tag']}\n{r['perf']:.2f}", pt,
                    textcoords='offset points', xytext=(off[0], off[1]),
                    ha='center', va='center', fontsize=6.5, zorder=6)

    # Corner labels (outside the triangle, clear of the corner points' labels).
    ax.text(*(V_A + [0, 0.09]), "A\n(real posterior)", ha='center', va='bottom',
            fontsize=10, fontweight='bold')
    ax.text(*(V_D + [-0.05, -0.05]), "D\n(noise+prior)", ha='right', va='top',
            fontsize=10, fontweight='bold')
    ax.text(*(V_AP + [0.05, -0.05]), "A'\n(perturbed vitals)", ha='left', va='top',
            fontsize=10, fontweight='bold')

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(-0.16, 1.16)  # headroom so the A corner label clears the title
    ax.set_title(f'Final performance across A/D/A′ mixtures\n'
                 f'({METRIC_LABEL[metric]})', fontsize=12, y=1.04)


def draw_curves(ax, runs, metric, smooth_window):
    """Overlay smoothed metric-vs-step curves, one per run, sorted by final perf."""
    order = sorted(range(len(runs)), key=lambda i: -runs[i]['perf'])
    cmap = plt.cm.tab10
    for rank, i in enumerate(order):
        r = runs[i]
        steps, vals = r['steps'], r['vals']
        if len(steps) == 0:
            continue
        color = cmap(rank % 10)
        if smooth_window > 1 and len(steps) > smooth_window:
            sx, sy = smooth(steps, vals, smooth_window)
        else:
            sx, sy = steps, vals
        ax.plot(sx, sy, color=color, lw=1.8,
                label=f"{r['tag']} ({r['perf']:.2f})")

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_millions))
    ax.set_xlabel('Environment steps', fontsize=11)
    ax.set_ylabel(METRIC_LABEL[metric], fontsize=11)
    ax.set_title(f'Smoothed {METRIC_LABEL[metric]} over training '
                 f'(w={smooth_window})', fontsize=12)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=7.5, framealpha=0.85, ncol=2, title='mixture (final)',
              title_fontsize=8, loc='lower right')


def main():
    p = argparse.ArgumentParser(description='Compare training across scheme_mix runs')
    p.add_argument('--logdirs', default='./logdir/scheme_mix_*',
                   help="Glob (or space-joined list) of run logdirs "
                        "(default: ./logdir/scheme_mix_*)")
    p.add_argument('--save', default='./logdir/scheme_mix_comparison',
                   help='Output directory')
    p.add_argument('--metric', default='achievements',
                   choices=['achievements', 'score', 'crafter'])
    p.add_argument('--smooth', type=int, default=200,
                   help='Curve smoothing window in episodes (default: 200)')
    p.add_argument('--final_frac', type=float, default=0.1,
                   help='Fraction of final episodes averaged for the ternary '
                        'heatmap value (default: 0.1)')
    p.add_argument('--fmt', default='svg', choices=['svg', 'png', 'pdf'],
                   help='Figure format (default: svg — vector, per project pref)')
    args = p.parse_args()

    # Resolve logdirs (support a glob or a space-separated list).
    dirs = []
    for token in args.logdirs.split():
        dirs.extend(sorted(glob.glob(token)))
    dirs = [d for d in dirs if pathlib.Path(d, 'scores.jsonl').exists()]
    if not dirs:
        raise SystemExit(f'No runs with scores.jsonl matched: {args.logdirs}')
    print(f'Found {len(dirs)} runs')

    runs = []
    for d in dirs:
        recs = load_jsonl(pathlib.Path(d) / 'scores.jsonl')
        steps, vals = get_series(recs, args.metric, args.smooth)
        if len(steps) == 0:
            print(f'  SKIP {d}: no {args.metric} data')
            continue
        mix, tag = parse_mix(d)
        perf = final_performance(steps, vals, args.final_frac)
        runs.append(dict(dir=d, tag=tag, mix=mix, steps=steps, vals=vals, perf=perf))
        print(f'  {tag:12s} mix={mix}  n_ep={len(steps):5d}  final {args.metric}={perf:.3f}')

    if not runs:
        raise SystemExit('No usable runs.')

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(16, 7),
                                   gridspec_kw={'width_ratios': [1, 1.15]})
    fig.suptitle(f'Replay-scheme mixture sweep — {args.metric}',
                 fontsize=14, fontweight='bold')
    draw_ternary_heatmap(axL, runs, args.metric)
    draw_curves(axR, runs, args.metric, args.smooth)
    plt.tight_layout(rect=(0, 0, 1, 0.96))

    save_dir = pathlib.Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f'scheme_mix_comparison_{args.metric}.{args.fmt}'
    fig.savefig(out, bbox_inches='tight')
    print(f'Saved: {out}')

    # CSV summary.
    csv = save_dir / f'scheme_mix_summary_{args.metric}.csv'
    with open(csv, 'w') as f:
        f.write('tag,A,D,Ap,n_episodes,final_' + args.metric + '\n')
        for r in sorted(runs, key=lambda r: -r['perf']):
            a, d, ap = r['mix']
            f.write(f"{r['tag']},{a:g},{d:g},{ap:g},{len(r['steps'])},{r['perf']:.4f}\n")
    print(f'Saved: {csv}')


if __name__ == '__main__':
    main()
