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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Constants (mirrored from decode_position.py and tuning_curve.py)
# ---------------------------------------------------------------------------

LAYER_ORDER = [
    'enc/cnn0', 'enc/cnn1', 'enc/cnn2', 'enc/cnn3',
    'enc/mlp0', 'enc/mlp1', 'enc/mlp2',
    'enc/tokens',
    'dyn/stoch', 'dyn/deter',
    'pol/mlp/linear0', 'pol/mlp/linear1', 'pol/mlp/linear2',
    'val/mlp/linear0', 'val/mlp/linear1', 'val/mlp/linear2',
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
    """Parse 'path:label' strings. Label defaults to basename of path."""
    conditions = []
    for item in raw:
        if ':' in item:
            path, label = item.rsplit(':', 1)
        else:
            path, label = item, os.path.basename(item.rstrip('/'))
        conditions.append((path, label))
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


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _condition_colors(n):
    cmap = plt.cm.tab10
    return [cmap(i % 10) for i in range(n)]


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
# Plot 2: Decode line plot
# ---------------------------------------------------------------------------

def plot_decode_lineplot(decode_data, labels, layers, metric_name, save_dir):
    n_layers = len(layers)
    if n_layers == 0:
        return

    colors = _condition_colors(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, n_layers * 0.6), 5))
    x = np.arange(n_layers)

    for i, (dd, label) in enumerate(zip(decode_data, labels)):
        if dd is None:
            continue
        medians = []
        lo = []
        hi = []
        for ln in layers:
            vals = dd['layer_fold_values'].get(ln)
            if vals is not None and len(vals) > 0:
                arr = np.array(vals)
                medians.append(np.median(arr))
                lo.append(np.percentile(arr, 25))
                hi.append(np.percentile(arr, 75))
            else:
                medians.append(np.nan)
                lo.append(np.nan)
                hi.append(np.nan)
        medians = np.array(medians)
        lo = np.array(lo)
        hi = np.array(hi)
        ax.plot(x, medians, 'o-', color=colors[i], label=label, markersize=4)
        ax.fill_between(x, lo, hi, alpha=0.15, color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=45, ha='right', fontsize=8)
    for idx, ln in enumerate(layers):
        ax.get_xticklabels()[idx].set_color(_section_color(ln))
    ax.set_ylabel(metric_name, fontsize=10)
    ax.set_title(f'Layer Decoding — {metric_name}', fontsize=11)
    ax.legend(fontsize=8)
    _style_ax(ax)
    fig.tight_layout()
    out = Path(save_dir) / 'decode_lineplot.png'
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
        bottom = np.zeros(n_layers)
        for g in range(n_groups):
            label_str = GROUP_NAMES[g] if i == 0 else None
            ax.bar(offsets, fracs[i, :, g], bar_width, bottom=bottom,
                   color=colors[g], label=label_str, edgecolor='white', linewidth=0.3)
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

    # Condition labels below legend
    if n_cond > 1:
        cond_colors = _condition_colors(n_cond)
        # Add condition indicator as a second legend
        from matplotlib.patches import Patch
        cond_patches = [Patch(facecolor=cond_colors[i], label=labels[i])
                        for i in range(n_cond)]
        ax2 = ax.twinx()
        ax2.set_yticks([])
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.legend(handles=cond_patches, fontsize=7, loc='upper left',
                   framealpha=0.8, title='Conditions', title_fontsize=8)

    fig.tight_layout()
    out = Path(save_dir) / 'tuning_celltypes.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out}")
    return str(out)


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------

def write_summary_csv(decode_data, tuning_data, labels, decode_layers,
                      tuning_layers, metric_name, save_dir):
    all_layers = list(dict.fromkeys(decode_layers + tuning_layers))
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
            rows.append(row)

    if not rows:
        return

    fieldnames = ['condition', 'layer', 'decode_metric_name', 'decode_metric_median',
                  'decode_metric_std', 'mean_SI', 'mean_EV', 'pct_spatial', 'n_neurons']
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
    args = parser.parse_args()

    conditions = parse_conditions(args.conditions)
    labels = [label for _, label in conditions]
    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    decode_data = []
    tuning_data = []
    for path, label in conditions:
        if not args.no_decode:
            dd = load_decode(path, args.decode_subdir)
            decode_data.append(dd)
        else:
            decode_data.append(None)
        if not args.no_tuning:
            td = load_tuning(path, args.tuning_subdir)
            tuning_data.append(td)
        else:
            tuning_data.append(None)

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
        out = plot_decode_lineplot(decode_data, labels, decode_layers, metric_name, save_dir)
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

    # Summary CSV
    print("Writing summary CSV...")
    out = write_summary_csv(decode_data, tuning_data, labels,
                            decode_layers, tuning_layers, metric_name, save_dir)
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
            },
            outputs=outputs,
        )
    except Exception as e:
        warnings.warn(f"Could not log run info: {e}")

    print("Done.")


if __name__ == '__main__':
    main()
