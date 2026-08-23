"""Cross-condition layer comparison boxplot: same-world vs multi-world.

Three conditions per layer, all tested on World 45:
  A) Same world — train 45, test 45
  B) Different world — train 50, test 45
  C) Multi-world — train 104 (fixed_seed=False), test 45

Usage:
  MPLBACKEND=Agg python dreamerv3/plot_layer_comparison_multiworld.py
"""

import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import to_rgb
from pathlib import Path

# ============== CONFIG (defaults; override via CLI) ==============
import argparse

DEFAULTS = dict(
    base_45='logdir/worlds_test_45',
    base_50='logdir/worlds_test_50',
    base_104='logdir/worlds_test_104',
    test_seed=45,
    save='qbio_draft2/panel7',
    title='Cross-Condition Layer Decoding',
)


def build_conditions(base_45, base_50, base_104, test_seed):
    """Three conditions (train 45 / 50 / 104), each tested on world `test_seed`
    via its layer_decoder_results_seed{test_seed}/layer_decode_results.pkl."""
    sub = f'layer_decoder_results_seed{test_seed}'
    return {
        f'A: Same world (train 45)': Path(base_45) / sub / 'layer_decode_results.pkl',
        f'B: Diff world (train 50)': Path(base_50) / sub / 'layer_decode_results.pkl',
        f'C: Multi-world (train 104)': Path(base_104) / sub / 'layer_decode_results.pkl',
    }

# Layer section colors (matching layer_comparison.svg / decode_position.py)
SECTION_COLORS = {
    'enc/cnn': '#0055cc',
    'enc/mlp': '#0099ff',
    'enc/tok': '#44ccff',
    'dyn/sto': '#ff9900',
    'dyn/det': '#ff5500',
    'val/mlp': '#996600',
    'pol/mlp': '#33aa00',
}

def _section_color(layer_name):
    for prefix, color in SECTION_COLORS.items():
        if layer_name.startswith(prefix):
            return color
    return '#888888'

# Per-condition style: (alpha, hatch, lighten_factor)
# lighten_factor: 0 = original color, 1 = white
CONDITION_STYLES = [
    dict(alpha=0.85, hatch='', lighten=0.0, edgecolor=None, linestyle='-'),   # A: solid, full color
    dict(alpha=0.50, hatch='///', lighten=0.0, edgecolor=None, linestyle='-'), # B: hatched, half alpha
    dict(alpha=1.0, hatch='', lighten=0.55, edgecolor='black', linestyle='--', linewidth=1.5),  # C: lightened fill, dashed black edge
]
# ====================================

def _lighten(color, factor):
    """Mix color toward white by factor (0=unchanged, 1=white)."""
    r, g, b = to_rgb(color)
    return (r + (1 - r) * factor, g + (1 - g) * factor, b + (1 - b) * factor)


def main():
    ap = argparse.ArgumentParser(description='Cross-condition layer decoding boxplot')
    ap.add_argument('--base_45', default=DEFAULTS['base_45'], help='train-45 model logdir')
    ap.add_argument('--base_50', default=DEFAULTS['base_50'], help='train-50 model logdir')
    ap.add_argument('--base_104', default=DEFAULTS['base_104'], help='train-104 (multi) logdir')
    ap.add_argument('--test_seed', type=int, default=DEFAULTS['test_seed'],
                    help='eval world seed (uses layer_decoder_results_seed{N})')
    ap.add_argument('--save', default=DEFAULTS['save'], help='output directory')
    ap.add_argument('--title', default=DEFAULTS['title'], help='plot title (test world appended)')
    ap.add_argument('--cond', action='append', default=[],
                    help='explicit condition as "LABEL=path/to/layer_decode_results.pkl"; '
                         'repeatable. If any given, overrides the base_45/50/104 defaults. '
                         'Order sets left-to-right box order (max 3).')
    ap.add_argument('--out_name', default='cross_condition_layer_comparison_multiworld.svg',
                    help='output filename')
    args = ap.parse_args()

    if args.cond:
        conditions = {}
        for c in args.cond:
            if '=' not in c:
                raise SystemExit(f'--cond must be "LABEL=path", got: {c}')
            label, path = c.split('=', 1)
            conditions[label.strip()] = Path(path.strip())
    else:
        conditions = build_conditions(args.base_45, args.base_50, args.base_104, args.test_seed)

    data = {}
    ordered = None
    for label, p in conditions.items():
        with open(p, 'rb') as f:
            d = pickle.load(f)
        data[label] = d['layer_fold_values']
        if ordered is None:
            ordered = d['ordered']
        print(f'{label}: {len(d["layer_fold_values"])} layers')

    cond_labels = list(conditions.keys())
    n_conds = len(cond_labels)
    layers = [ln for ln in ordered if ln in data[cond_labels[0]]]
    # Reorder: value layers before policy layers (policy rightmost)
    val = [ln for ln in layers if ln.startswith('val/')]
    pol = [ln for ln in layers if ln.startswith('pol/')]
    rest = [ln for ln in layers if not ln.startswith('val/') and not ln.startswith('pol/')]
    layers = rest + val + pol
    n_layers = len(layers)

    fig, ax = plt.subplots(figsize=(max(12, n_layers * 1.5), 6))
    width = 0.7 / n_conds
    offsets = np.linspace(-(n_conds - 1) / 2 * width,
                          (n_conds - 1) / 2 * width, n_conds)
    whisker_tops = []

    for li, ln in enumerate(layers):
        layer_color = _section_color(ln)
        layer_max = 0
        for ci, cond in enumerate(cond_labels):
            sty = CONDITION_STYLES[ci]
            fill = _lighten(layer_color, sty['lighten'])
            edge = sty['edgecolor'] or layer_color
            pos = [li + offsets[ci]]
            vals = [np.asarray(data[cond][ln])]
            bp = ax.boxplot(vals, positions=pos, widths=width * 0.85,
                            patch_artist=True, showfliers=False, showmeans=True,
                            meanprops=dict(marker='D', markerfacecolor='white',
                                           markeredgecolor=layer_color, markersize=4),
                            medianprops=dict(color='white', linewidth=1.5),
                            whiskerprops=dict(color=layer_color, linewidth=1, alpha=sty['alpha']),
                            capprops=dict(color=layer_color, linewidth=1, alpha=sty['alpha']))
            for patch in bp['boxes']:
                patch.set_facecolor(fill)
                patch.set_alpha(sty['alpha'])
                patch.set_edgecolor(edge)
                patch.set_hatch(sty['hatch'])
                patch.set_linestyle(sty['linestyle'])
                if 'linewidth' in sty:
                    patch.set_linewidth(sty['linewidth'])
            wtop = max(w.get_ydata()[1] for w in bp['whiskers'][1::2])
            layer_max = max(layer_max, wtop)
        whisker_tops.append(layer_max)

    ax.set_xticks(range(n_layers))
    short_names = [ln.replace('mlp/', '') for ln in layers]
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=9)
    # Color x-tick labels by section
    for idx, ln in enumerate(layers):
        ax.get_xticklabels()[idx].set_color(_section_color(ln))

    ax.set_ylabel('Manhattan error (tiles)', fontsize=11)
    ax.set_title(f'{args.title} — tested on World {args.test_seed}',
                 fontsize=13, fontweight='bold')
    top_y = max(whisker_tops) + max(whisker_tops) * 0.08
    ax.set_ylim(ax.get_ylim()[0], top_y)

    # Condition legend
    cond_patches = []
    for i, cond in enumerate(cond_labels):
        sty = CONDITION_STYLES[i]
        fc = _lighten('#888888', sty['lighten'])
        cond_patches.append(Patch(
            facecolor=fc, alpha=sty['alpha'],
            edgecolor=sty['edgecolor'] or 'black',
            hatch=sty['hatch'], linestyle=sty['linestyle'],
            linewidth=sty.get('linewidth', 1.0),
            label=cond))
    ax.legend(handles=cond_patches, fontsize=8, loc='upper left')
    ax.grid(True, axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / args.out_name
    fig.savefig(str(out), bbox_inches='tight')
    print(f'Saved to {out}')


if __name__ == '__main__':
    main()
