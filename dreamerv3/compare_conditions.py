"""
Cross-condition comparison of layer decoding and tuning curve results.

Loads layer_decode_results.pkl and tuning_results.pkl from N conditions
and produces comparison heatmaps, line plots, cell type composition charts,
and a summary CSV.

Usage:
    MPLBACKEND=Agg python dreamerv3/compare_conditions.py \
      --conditions ./logdir/run1:label1 ./logdir/run2:label2 \
      --save ./logdir/comparison_plots
"""

import argparse
import csv
import os
import pickle
import sys
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy.stats import mannwhitneyu

# ---------------------------------------------------------------------------
# Constants (mirrored from decode_position.py and tuning_curve.py)
# ---------------------------------------------------------------------------

LAYER_ORDER = [
    'enc/cnn0', 'enc/cnn1', 'enc/cnn2', 'enc/cnn3',
    'enc/mlp0', 'enc/mlp1', 'enc/mlp2',
    'enc/tokens',
    'dyn/stoch', 'dyn/deter',
    'val/mlp/linear0', 'val/mlp/linear1', 'val/mlp/linear2',
    'pol/mlp/linear0', 'pol/mlp/linear1', 'pol/mlp/linear2',
]

SECTION_COLORS = {
    'enc/cnn': '#0055cc',
    'enc/mlp': '#0099ff',
    'enc/tok': '#44ccff',
    'dyn/sto': '#ff9900',
    'dyn/det': '#ff5500',
    'pol/mlp': '#33aa00',
    'val/mlp': '#996600',
}

GROUP_NAMES = [
    'untuned', 'HD_cells', 'single_field', 'border_cells',
    'spatial_HD', 'complex_cells', 'dead',
]


def _section_color(layer_name):
    for prefix, color in SECTION_COLORS.items():
        if layer_name.startswith(prefix):
            return color
    return '#888888'


def _order_layers(layer_names):
    """Sort layers by LAYER_ORDER, unknowns appended alphabetically."""
    order_map = {ln: i for i, ln in enumerate(LAYER_ORDER)}
    known = sorted([ln for ln in layer_names if ln in order_map],
                   key=lambda ln: order_map[ln])
    unknown = sorted(ln for ln in layer_names if ln not in order_map)
    return known + unknown


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def parse_conditions(raw):
    """Parse condition strings with optional per-condition subdir overrides.

    Format: path:label[:key=value ...]
    Supported keys: decode, tuning, manifold (override subdir names).
    Example:
        ./logdir/run1:vanilla:decode=layer_decoder_results_seed45:tuning=tuning_results_seed45
    """
    conditions = []
    for item in raw:
        parts = item.split(':')
        if len(parts) >= 2:
            path, label = parts[0], parts[1]
        else:
            path, label = parts[0], os.path.basename(parts[0].rstrip('/'))
        overrides = {}
        for part in parts[2:]:
            if '=' in part:
                k, v = part.split('=', 1)
                overrides[k] = v
        conditions.append((path, label, overrides))
    return conditions


def load_decode(path, subdir):
    fp = Path(path) / subdir / 'layer_decode_results.pkl'
    if not fp.exists():
        warnings.warn(f"Decode results not found: {fp}")
        return None
    with open(fp, 'rb') as f:
        return pickle.load(f)


def load_tuning(path, subdir):
    fp = Path(path) / subdir / 'tuning_results.pkl'
    if not fp.exists():
        warnings.warn(f"Tuning results not found: {fp}")
        return None
    with open(fp, 'rb') as f:
        return pickle.load(f)


def load_manifold(path, subdir):
    fp = Path(path) / subdir / 'manifold_results.pkl'
    if not fp.exists():
        warnings.warn(f"Manifold results not found: {fp}")
        return None
    with open(fp, 'rb') as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _condition_colors(n):
    cmap = plt.cm.tab10
    return [cmap(i % 10) for i in range(n)]


CONDITION_HATCHES = ['', '///', '...', 'xxx', '\\\\\\', '+++']
CONDITION_ALPHAS = [0.85, 0.50, 0.25, 0.15, 0.10, 0.08]


# ---------------------------------------------------------------------------
# Plot 1: Decode heatmap
# ---------------------------------------------------------------------------

def plot_decode_heatmap(decode_data, labels, layers, metric_name, save_dir):
    n_cond = len(labels)
    n_layers = len(layers)
    if n_layers == 0 or n_cond == 0:
        return

    matrix = np.full((n_cond, n_layers), np.nan)
    for i, dd in enumerate(decode_data):
        if dd is None:
            continue
        for j, ln in enumerate(layers):
            vals = dd['layer_fold_values'].get(ln)
            if vals is not None:
                matrix[i, j] = np.median(vals)

    fig, ax = plt.subplots(figsize=(max(8, n_layers * 0.7), max(3, n_cond * 0.5 + 1.5)))
    im = ax.imshow(matrix, aspect='auto', cmap='viridis')
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels(layers, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(n_cond))
    ax.set_yticklabels(labels, fontsize=9)

    # Color x-tick labels by section
    for idx, ln in enumerate(layers):
        ax.get_xticklabels()[idx].set_color(_section_color(ln))

    # Annotate cells
    for i in range(n_cond):
        for j in range(n_layers):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=7, color='white' if v < (np.nanmax(matrix) + np.nanmin(matrix)) / 2 else 'black')

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(metric_name, fontsize=9)
    ax.set_title(f'Layer Decoding — {metric_name}', fontsize=11)
    fig.tight_layout()
    out = Path(save_dir) / 'decode_heatmap.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out}")
    return str(out)


# ---------------------------------------------------------------------------
# Plot 2: Decode grouped boxplot
# ---------------------------------------------------------------------------

def _sig_label(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'


def plot_decode_boxplot(decode_data, labels, layers, metric_name, save_dir,
                        no_sig=False):
    """Grouped boxplot: side-by-side boxes per condition at each layer,
    colored by layer section, distinguished by hatching + alpha.
    Pairwise Mann-Whitney U with Bonferroni correction."""
    n_layers = len(layers)
    n_cond = len(labels)
    if n_layers == 0 or n_cond == 0:
        return

    # Reorder: value layers before policy layers (policy rightmost)
    val = [ln for ln in layers if ln.startswith('val/')]
    pol = [ln for ln in layers if ln.startswith('pol/')]
    rest = [ln for ln in layers if not ln.startswith('val/') and not ln.startswith('pol/')]
    layers = rest + val + pol
    n_layers = len(layers)

    fig, ax = plt.subplots(figsize=(max(12, n_layers * 1.5),
                                     6 if no_sig else 6 + n_cond * (n_cond - 1) // 2 * 0.6))
    width = 0.7 / n_cond
    offsets = np.linspace(-(n_cond - 1) / 2 * width,
                          (n_cond - 1) / 2 * width, n_cond)
    whisker_tops = []

    for li, ln in enumerate(layers):
        layer_color = _section_color(ln)
        layer_max = 0
        for ci, dd in enumerate(decode_data):
            if dd is None:
                continue
            vals = dd['layer_fold_values'].get(ln)
            if vals is None or len(vals) == 0:
                continue
            hatch = CONDITION_HATCHES[ci % len(CONDITION_HATCHES)]
            alpha = CONDITION_ALPHAS[ci % len(CONDITION_ALPHAS)]
            pos = [li + offsets[ci]]
            bp = ax.boxplot([np.asarray(vals)], positions=pos,
                            widths=width * 0.85,
                            patch_artist=True, showfliers=False, showmeans=True,
                            meanprops=dict(marker='D', markerfacecolor='white',
                                           markeredgecolor=layer_color, markersize=4),
                            medianprops=dict(color='white', linewidth=1.5),
                            whiskerprops=dict(color=layer_color, linewidth=1, alpha=alpha),
                            capprops=dict(color=layer_color, linewidth=1, alpha=alpha))
            for patch in bp['boxes']:
                patch.set_facecolor(layer_color)
                patch.set_alpha(alpha)
                patch.set_edgecolor(layer_color)
                patch.set_hatch(hatch)
            wtop = max(w.get_ydata()[1] for w in bp['whiskers'][1::2])
            layer_max = max(layer_max, wtop)
        whisker_tops.append(layer_max)

    # Significance brackets
    y_max_global = max(whisker_tops) if whisker_tops else 1
    bracket_step = y_max_global * 0.06
    n_pairs = n_cond * (n_cond - 1) // 2
    pair_indices = list(combinations(range(n_cond), 2))

    if not no_sig:
        for li, ln in enumerate(layers):
            base_y = whisker_tops[li] + y_max_global * 0.02
            for pi, (ci, cj) in enumerate(pair_indices):
                dd_i = decode_data[ci]
                dd_j = decode_data[cj]
                if dd_i is None or dd_j is None:
                    continue
                g1 = dd_i['layer_fold_values'].get(ln)
                g2 = dd_j['layer_fold_values'].get(ln)
                if g1 is None or g2 is None or len(g1) < 2 or len(g2) < 2:
                    continue
                g1, g2 = np.asarray(g1), np.asarray(g2)
                _, p_raw = mannwhitneyu(g1, g2, alternative='two-sided')
                p_corr = min(p_raw * n_pairs, 1.0)
                sl = _sig_label(p_corr)
                y = base_y + pi * bracket_step
                x1 = li + offsets[ci]
                x2 = li + offsets[cj]
                ax.plot([x1, x1, x2, x2],
                        [y, y + bracket_step * 0.3, y + bracket_step * 0.3, y],
                        color='0.3', linewidth=0.8)
                ax.text((x1 + x2) / 2, y + bracket_step * 0.3, sl,
                        ha='center', va='bottom', fontsize=6, fontweight='bold',
                        color='0.2')

    ax.set_xticks(range(n_layers))
    short_names = [ln.replace('mlp/', '') for ln in layers]
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=9)
    for idx, ln in enumerate(layers):
        ax.get_xticklabels()[idx].set_color(_section_color(ln))

    ax.set_ylabel(metric_name, fontsize=11)
    ax.set_title(f'Layer Decoding — {metric_name}', fontsize=13, fontweight='bold')
    if no_sig:
        ax.set_ylim(ax.get_ylim()[0], y_max_global * 1.1)
    else:
        top_bracket_y = (max(whisker_tops) + y_max_global * 0.02
                         + n_pairs * bracket_step + y_max_global * 0.05)
        ax.set_ylim(ax.get_ylim()[0], top_bracket_y)

    # Condition legend
    cond_patches = [Patch(facecolor='#888888',
                          alpha=CONDITION_ALPHAS[i % len(CONDITION_ALPHAS)],
                          edgecolor='black',
                          hatch=CONDITION_HATCHES[i % len(CONDITION_HATCHES)],
                          label=labels[i])
                    for i in range(n_cond)]
    if not no_sig:
        cond_patches.append(Patch(facecolor='none', edgecolor='none',
                                  label='Mann-Whitney U, Bonferroni'))
    ax.legend(handles=cond_patches, fontsize=8, loc='upper left')
    ax.grid(True, axis='y', alpha=0.3)
    _style_ax(ax)
    fig.tight_layout()
    out = Path(save_dir) / 'decode_boxplot.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out}")
    return str(out)


# ---------------------------------------------------------------------------
# Plot 3 & 4: Tuning heatmaps (SI / EV)
# ---------------------------------------------------------------------------

def _tuning_heatmap(tuning_data, labels, layers, metric_key, title, fname, save_dir):
    n_cond = len(labels)
    n_layers = len(layers)
    if n_layers == 0 or n_cond == 0:
        return

    matrix = np.full((n_cond, n_layers), np.nan)
    for i, td in enumerate(tuning_data):
        if td is None:
            continue
        ldict = td['layers']
        for j, ln in enumerate(layers):
            if ln in ldict and metric_key in ldict[ln]['metrics']:
                vals = ldict[ln]['metrics'][metric_key]
                matrix[i, j] = np.nanmean(vals)

    fig, ax = plt.subplots(figsize=(max(8, n_layers * 0.7), max(3, n_cond * 0.5 + 1.5)))
    im = ax.imshow(matrix, aspect='auto', cmap='viridis')
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels(layers, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(n_cond))
    ax.set_yticklabels(labels, fontsize=9)

    for idx, ln in enumerate(layers):
        ax.get_xticklabels()[idx].set_color(_section_color(ln))

    for i in range(n_cond):
        for j in range(n_layers):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                        fontsize=7, color='white' if v < (np.nanmax(matrix) + np.nanmin(matrix)) / 2 else 'black')

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(title, fontsize=9)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    out = Path(save_dir) / fname
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out}")
    return str(out)


def plot_tuning_si_heatmap(tuning_data, labels, layers, save_dir):
    return _tuning_heatmap(tuning_data, labels, layers,
                           'SI', 'Mean Spatial Information', 'tuning_si_heatmap.png', save_dir)


def plot_tuning_ev_heatmap(tuning_data, labels, layers, save_dir):
    return _tuning_heatmap(tuning_data, labels, layers,
                           'EVs', 'Mean Explained Variance', 'tuning_ev_heatmap.png', save_dir)


# ---------------------------------------------------------------------------
# Plot 5: Cell type composition
# ---------------------------------------------------------------------------

def plot_tuning_celltypes(tuning_data, labels, layers, save_dir):
    n_cond = len(labels)
    n_layers = len(layers)
    if n_layers == 0 or n_cond == 0:
        return

    # Build fraction matrix: (n_cond, n_layers, n_groups)
    n_groups = len(GROUP_NAMES)
    fracs = np.zeros((n_cond, n_layers, n_groups))
    for i, td in enumerate(tuning_data):
        if td is None:
            continue
        ldict = td['layers']
        for j, ln in enumerate(layers):
            if ln not in ldict:
                continue
            cg = ldict[ln]['cell_groups']
            total = len(next(iter(cg.values())))
            if total == 0:
                continue
            for g, gname in enumerate(GROUP_NAMES):
                if gname in cg:
                    fracs[i, j, g] = np.sum(cg[gname]) / total

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_groups))
    bar_width = 0.8 / n_cond
    fig, ax = plt.subplots(figsize=(max(10, n_layers * 1.2), 5))

    for i in range(n_cond):
        offsets = np.arange(n_layers) + i * bar_width - 0.4 + bar_width / 2
        hatch = CONDITION_HATCHES[i % len(CONDITION_HATCHES)]
        bottom = np.zeros(n_layers)
        for g in range(n_groups):
            label_str = GROUP_NAMES[g] if i == 0 else None
            ax.bar(offsets, fracs[i, :, g], bar_width, bottom=bottom,
                   color=colors[g], label=label_str, edgecolor='white',
                   linewidth=0.3, hatch=hatch)
            bottom += fracs[i, :, g]

    ax.set_xticks(np.arange(n_layers))
    ax.set_xticklabels(layers, rotation=45, ha='right', fontsize=8)
    for idx, ln in enumerate(layers):
        ax.get_xticklabels()[idx].set_color(_section_color(ln))

    ax.set_ylabel('Fraction', fontsize=10)
    ax.set_title('Cell Type Composition', fontsize=11)
    ax.set_ylim(0, 1.05)
    _style_ax(ax)

    # Legend for cell types
    handles, leg_labels = ax.get_legend_handles_labels()
    ax.legend(handles[:n_groups], GROUP_NAMES, fontsize=7, loc='upper right',
              ncol=2, framealpha=0.8)

    # Condition legend using hatching
    if n_cond > 1:
        from matplotlib.patches import Patch
        cond_patches = [Patch(facecolor='lightgrey', edgecolor='black',
                              hatch=CONDITION_HATCHES[i % len(CONDITION_HATCHES)],
                              label=labels[i])
                        for i in range(n_cond)]
        ax2 = ax.twinx()
        ax2.set_yticks([])
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.legend(handles=cond_patches, fontsize=7, loc='upper left',
                   framealpha=0.8, title='Condition', title_fontsize=8)

    fig.tight_layout()
    out = Path(save_dir) / 'tuning_celltypes.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out}")
    return str(out)


# ---------------------------------------------------------------------------
# Plot 6: Manifold heatmaps (sRSA / SW distance)
# ---------------------------------------------------------------------------

def _manifold_heatmap(manifold_data, labels, layers, metric_key, title, fname,
                      save_dir, cmap='viridis', fmt='.3f'):
    n_cond = len(labels)
    n_layers = len(layers)
    if n_layers == 0 or n_cond == 0:
        return

    matrix = np.full((n_cond, n_layers), np.nan)
    for i, md in enumerate(manifold_data):
        if md is None:
            continue
        for j, ln in enumerate(layers):
            if ln in md and metric_key in md[ln]:
                matrix[i, j] = md[ln][metric_key]

    fig, ax = plt.subplots(figsize=(max(8, n_layers * 0.7), max(3, n_cond * 0.5 + 1.5)))
    im = ax.imshow(matrix, aspect='auto', cmap=cmap)
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels(layers, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(n_cond))
    ax.set_yticklabels(labels, fontsize=9)

    for idx, ln in enumerate(layers):
        ax.get_xticklabels()[idx].set_color(_section_color(ln))

    for i in range(n_cond):
        for j in range(n_layers):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f'{v:{fmt}}', ha='center', va='center',
                        fontsize=7, color='white' if v < (np.nanmax(matrix) + np.nanmin(matrix)) / 2 else 'black')

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(title, fontsize=9)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    out = Path(save_dir) / fname
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out}")
    return str(out)


def plot_manifold_srsa_heatmap(manifold_data, labels, layers, save_dir):
    return _manifold_heatmap(manifold_data, labels, layers,
                             'srsa_rho', 'sRSA (Spearman ρ)',
                             'manifold_srsa_heatmap.png', save_dir)


def plot_manifold_sw_heatmap(manifold_data, labels, layers, save_dir):
    return _manifold_heatmap(manifold_data, labels, layers,
                             'sw_median', 'Median SW Distance (cosine)',
                             'manifold_sw_heatmap.png', save_dir,
                             cmap='viridis_r')


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------

def write_summary_csv(decode_data, tuning_data, manifold_data, labels,
                      decode_layers, tuning_layers, manifold_layers,
                      metric_name, save_dir):
    all_layers = list(dict.fromkeys(decode_layers + tuning_layers + manifold_layers))
    rows = []
    for i, label in enumerate(labels):
        dd = decode_data[i] if i < len(decode_data) else None
        td = tuning_data[i] if i < len(tuning_data) else None
        for ln in all_layers:
            row = {'condition': label, 'layer': ln}
            # Decode
            if dd is not None:
                vals = dd['layer_fold_values'].get(ln)
                if vals is not None and len(vals) > 0:
                    arr = np.array(vals)
                    row['decode_metric_median'] = f'{np.median(arr):.4f}'
                    row['decode_metric_std'] = f'{np.std(arr):.4f}'
                    row['decode_metric_name'] = metric_name
            # Tuning
            if td is not None and ln in td['layers']:
                lm = td['layers'][ln]['metrics']
                cg = td['layers'][ln]['cell_groups']
                n_neurons = len(next(iter(cg.values())))
                row['mean_SI'] = f'{np.nanmean(lm["SI"]):.4f}'
                row['mean_EV'] = f'{np.nanmean(lm["EVs"]):.4f}'
                # % spatial = single_field + border + spatial_HD + complex
                spatial_mask = (cg.get('single_field', np.zeros(n_neurons, dtype=bool)) |
                                cg.get('border_cells', np.zeros(n_neurons, dtype=bool)) |
                                cg.get('spatial_HD', np.zeros(n_neurons, dtype=bool)) |
                                cg.get('complex_cells', np.zeros(n_neurons, dtype=bool)))
                row['pct_spatial'] = f'{100 * np.mean(spatial_mask):.1f}'
                row['n_neurons'] = str(n_neurons)
            # Manifold
            md = manifold_data[i] if i < len(manifold_data) else None
            if md is not None and ln in md:
                row['srsa_rho'] = f'{md[ln]["srsa_rho"]:.4f}'
                row['sw_median'] = f'{md[ln]["sw_median"]:.4f}'
            rows.append(row)

    if not rows:
        return

    fieldnames = ['condition', 'layer', 'decode_metric_name', 'decode_metric_median',
                  'decode_metric_std', 'mean_SI', 'mean_EV', 'pct_spatial', 'n_neurons',
                  'srsa_rho', 'sw_median']
    out = Path(save_dir) / 'summary.csv'
    with open(out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {out}")
    return str(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Compare layer decoding and tuning results across conditions.')
    parser.add_argument('--conditions', nargs='+', required=True,
                        help='Space-separated path:label pairs')
    parser.add_argument('--save', required=True, help='Output directory')
    parser.add_argument('--decode_subdir', default='layer_decoder_results',
                        help='Subfolder containing layer_decode_results.pkl')
    parser.add_argument('--tuning_subdir', default='tuning_results',
                        help='Subfolder containing tuning_results.pkl')
    parser.add_argument('--layers', nargs='*', default=None,
                        help='Optional layer filter (e.g. dyn/deter dyn/stoch)')
    parser.add_argument('--no_decode', action='store_true', help='Skip decoding plots')
    parser.add_argument('--no_tuning', action='store_true', help='Skip tuning plots')
    parser.add_argument('--no_manifold', action='store_true', help='Skip manifold plots')
    parser.add_argument('--manifold_subdir', default='manifold_results',
                        help='Subfolder containing manifold_results.pkl')
    parser.add_argument('--no_sig', action='store_true', help='Skip significance brackets on boxplot')
    args = parser.parse_args()

    conditions = parse_conditions(args.conditions)
    labels = [label for _, label, _ in conditions]
    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    decode_data = []
    tuning_data = []
    manifold_data = []
    for path, label, overrides in conditions:
        dec_sub = overrides.get('decode', args.decode_subdir)
        tun_sub = overrides.get('tuning', args.tuning_subdir)
        man_sub = overrides.get('manifold', args.manifold_subdir)
        if not args.no_decode:
            dd = load_decode(path, dec_sub)
            decode_data.append(dd)
        else:
            decode_data.append(None)
        if not args.no_tuning:
            td = load_tuning(path, tun_sub)
            tuning_data.append(td)
        else:
            tuning_data.append(None)
        if not args.no_manifold:
            md = load_manifold(path, man_sub)
            manifold_data.append(md)
        else:
            manifold_data.append(None)

    # Determine layers
    decode_layers = []
    tuning_layers = []

    if not args.no_decode:
        all_decode_layers = set()
        for dd in decode_data:
            if dd is not None:
                all_decode_layers.update(dd.get('ordered', dd['layer_fold_values'].keys()))
        decode_layers = _order_layers(all_decode_layers)
        if args.layers:
            decode_layers = [ln for ln in decode_layers if ln in args.layers]

    if not args.no_tuning:
        all_tuning_layers = set()
        for td in tuning_data:
            if td is not None:
                all_tuning_layers.update(td['layers'].keys())
        tuning_layers = _order_layers(all_tuning_layers)
        if args.layers:
            tuning_layers = [ln for ln in tuning_layers if ln in args.layers]

    manifold_layers = []
    if not args.no_manifold:
        all_manifold_layers = set()
        for md in manifold_data:
            if md is not None:
                all_manifold_layers.update(md.keys())
        manifold_layers = _order_layers(all_manifold_layers)
        if args.layers:
            manifold_layers = [ln for ln in manifold_layers if ln in args.layers]

    # Determine metric name for decode
    metric_name = 'R²'
    for dd in decode_data:
        if dd is not None:
            m = dd.get('metric', 'r2')
            metric_name = 'R²' if m == 'r2' else 'Manhattan Distance (tiles)'
            break

    outputs = []

    # Decode plots
    if not args.no_decode and any(dd is not None for dd in decode_data):
        print("Generating decoding comparison plots...")
        out = plot_decode_heatmap(decode_data, labels, decode_layers, metric_name, save_dir)
        if out:
            outputs.append(out)
        out = plot_decode_boxplot(decode_data, labels, decode_layers, metric_name, save_dir,
                                  no_sig=args.no_sig)
        if out:
            outputs.append(out)
    elif not args.no_decode:
        print("No decode data found for any condition — skipping decode plots.")

    # Tuning plots
    if not args.no_tuning and any(td is not None for td in tuning_data):
        print("Generating tuning comparison plots...")
        out = plot_tuning_si_heatmap(tuning_data, labels, tuning_layers, save_dir)
        if out:
            outputs.append(out)
        out = plot_tuning_ev_heatmap(tuning_data, labels, tuning_layers, save_dir)
        if out:
            outputs.append(out)
        out = plot_tuning_celltypes(tuning_data, labels, tuning_layers, save_dir)
        if out:
            outputs.append(out)
    elif not args.no_tuning:
        print("No tuning data found for any condition — skipping tuning plots.")

    # Manifold plots
    if not args.no_manifold and any(md is not None for md in manifold_data):
        print("Generating manifold comparison plots...")
        out = plot_manifold_srsa_heatmap(manifold_data, labels, manifold_layers, save_dir)
        if out:
            outputs.append(out)
        out = plot_manifold_sw_heatmap(manifold_data, labels, manifold_layers, save_dir)
        if out:
            outputs.append(out)
    elif not args.no_manifold:
        print("No manifold data found for any condition — skipping manifold plots.")

    # Summary CSV
    print("Writing summary CSV...")
    out = write_summary_csv(decode_data, tuning_data, manifold_data, labels,
                            decode_layers, tuning_layers, manifold_layers,
                            metric_name, save_dir)
    if out:
        outputs.append(out)

    # Provenance
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from run_info import log_run_info
        log_run_info(
            save_dir=str(save_dir),
            stage='compare_conditions',
            args={
                'conditions': args.conditions,
                'decode_subdir': args.decode_subdir,
                'tuning_subdir': args.tuning_subdir,
                'layers': args.layers,
                'no_decode': args.no_decode,
                'no_tuning': args.no_tuning,
                'no_manifold': args.no_manifold,
                'manifold_subdir': args.manifold_subdir,
            },
            outputs=outputs,
        )
    except Exception as e:
        warnings.warn(f"Could not log run info: {e}")

    print("Done.")


if __name__ == '__main__':
    main()
