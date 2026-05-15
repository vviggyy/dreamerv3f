#!/usr/bin/env python3
"""Tuning curve analysis: clustering, metric-space embedding, distributions.

Three modes:
  autocorr (default): PCA / t-SNE / UMAP on spatial autocorrelation maps
    of tuning curves + HDBSCAN clustering.
  metrics: Isomap on per-neuron metric feature vectors (SI, EV, Moran's I,
    Geary's C, Getis-Ord G, field size, pf_peaks). Supports --interactive
    for a click-to-inspect viewer (scatter + tuning curve panel).
  distributions: Per-metric histogram + example tuning curves at quantile
    positions. Shows how each metric is distributed and what tuning curves
    look like at different metric values.

Usage:
    # Autocorrelation clustering (original)
    python dreamerv3/analyze_tuning.py \
        --from_pkl ./tuning_results/tuning_results.pkl \
        --save ./tuning_results/cluster_plots

    # Metric-space Isomap (static plots)
    python dreamerv3/analyze_tuning.py \
        --from_pkl ./tuning_results/tuning_results.pkl \
        --save ./tuning_results/cluster_plots \
        --mode metrics --layers dyn/deter

    # Metric distributions + example tuning curves
    python dreamerv3/analyze_tuning.py \
        --from_pkl ./tuning_results/tuning_results.pkl \
        --save ./tuning_results/dist_plots \
        --mode distributions --layers dyn/deter
"""

import argparse
import pickle
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate2d
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Reuse constants from sibling modules
# ---------------------------------------------------------------------------

GROUP_NAMES = [
    'untuned', 'HD_cells', 'single_field', 'border_cells',
    'spatial_HD', 'complex_cells', 'dead',
]

LAYER_ORDER = [
    'enc/cnn0', 'enc/cnn1', 'enc/cnn2', 'enc/cnn3',
    'enc/mlp0', 'enc/mlp1', 'enc/mlp2',
    'enc/tokens',
    'dyn/deter', 'dyn/stoch',
    'pol/mlp0', 'pol/mlp1',
    'val/mlp0', 'val/mlp1',
]


def _order_layers(layer_names):
    order_map = {ln: i for i, ln in enumerate(LAYER_ORDER)}
    known = sorted([ln for ln in layer_names if ln in order_map],
                   key=lambda ln: order_map[ln])
    unknown = sorted(ln for ln in layer_names if ln not in order_map)
    return known + unknown


# ---------------------------------------------------------------------------
# Autocorrelation (mirrored from tuning_curve.py)
# ---------------------------------------------------------------------------

def pf_autocorr(tuning_curves_array, peak_norm=True):
    """2D autocorrelation of each neuron's tuning curve."""
    results = []
    for tc in tuning_curves_array:
        ac = correlate2d(np.nan_to_num(tc), np.nan_to_num(tc), mode='full')
        if peak_norm and ac.max() > 0:
            ac = ac / ac.max()
        results.append(ac)
    return np.array(results)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _group_cmap(n=len(GROUP_NAMES)):
    return plt.get_cmap('tab10', n)


def _save(fig, path):
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {path}')


def plot_scree(explained_var, save_path):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(np.arange(1, len(explained_var) + 1),
            np.cumsum(explained_var), '-o', markersize=3)
    ax.set_xlabel('Number of components')
    ax.set_ylabel('Cumulative variance explained')
    ax.set_title('PCA scree plot')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    _save(fig, save_path)


def plot_scatter(embedding, labels, title, save_path, cmap=None,
                 is_continuous=False, label_names=None, xlabel='Dim 1',
                 ylabel='Dim 2'):
    fig, ax = plt.subplots(figsize=(6, 5))
    if is_continuous:
        sc = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, s=4,
                        alpha=0.6, cmap=cmap or 'viridis')
        plt.colorbar(sc, ax=ax, shrink=0.8)
    else:
        unique = np.unique(labels)
        cm = cmap or plt.get_cmap('tab10', max(len(unique), 1))
        for i, u in enumerate(unique):
            mask = labels == u
            name = label_names[u] if label_names and u < len(label_names) else str(u)
            color = cm(i % cm.N) if hasattr(cm, 'N') else cm(i / max(len(unique) - 1, 1))
            ax.scatter(embedding[mask, 0], embedding[mask, 1], s=4, alpha=0.6,
                       label=name, color=color)
        ax.legend(fontsize=6, markerscale=3, loc='best', framealpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _save(fig, save_path)


def plot_cluster_examples(tuning_curves, cluster_labels, si_values, save_path,
                          n_per_cluster=8, max_clusters=12):
    """Grid of top-SI tuning curves per HDBSCAN cluster."""
    unique_clusters = sorted(set(cluster_labels))
    # Remove noise label -1 from display if too many clusters
    if len(unique_clusters) > max_clusters:
        unique_clusters = [c for c in unique_clusters if c != -1][:max_clusters]

    n_clusters = len(unique_clusters)
    if n_clusters == 0:
        return

    fig, axes = plt.subplots(n_clusters, n_per_cluster,
                             figsize=(2 * n_per_cluster, 2 * n_clusters))
    if n_clusters == 1:
        axes = axes[np.newaxis, :]
    if n_per_cluster == 1:
        axes = axes[:, np.newaxis]

    for row, cl in enumerate(unique_clusters):
        mask = cluster_labels == cl
        idxs = np.where(mask)[0]
        # Sort by SI descending, pick top representatives
        order = np.argsort(-si_values[idxs])
        selected = idxs[order[:n_per_cluster]]
        for col in range(n_per_cluster):
            ax = axes[row, col]
            if col < len(selected):
                tc = tuning_curves[selected[col]]
                ax.imshow(tc, cmap='hot', interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                lbl = f'Cluster {cl}' if cl >= 0 else 'Noise'
                ax.set_ylabel(lbl, fontsize=8)

    fig.suptitle('Top-SI tuning curves per cluster', fontsize=11)
    fig.tight_layout()
    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def process_layer(layer_name, layer_data, save_dir, args):
    """Run full PCA → t-SNE → UMAP → HDBSCAN pipeline for one layer."""
    tc = layer_data['tuning_curves']  # (N, H, W)
    metrics = layer_data['metrics']
    group_ids = layer_data['group_ids']
    si = metrics['SI']
    N = tc.shape[0]
    print(f'\n--- {layer_name} ({N} neurons) ---')

    if N < 10:
        print(f'  Skipping {layer_name}: too few neurons ({N})')
        return None

    # 1. Autocorrelation maps
    autocorr = pf_autocorr(tc)  # (N, 2H-1, 2W-1)

    # 2. Flatten and preprocess
    flat = autocorr.reshape(N, -1)
    flat = np.nan_to_num(flat, nan=0.0)
    if args.normalize:
        flat = StandardScaler().fit_transform(flat)

    # 3. PCA
    n_comp = min(args.n_components, N - 1, flat.shape[1])
    pca = PCA(n_components=n_comp)
    pca_feats = pca.fit_transform(flat)

    prefix = layer_name.replace('/', '_')

    plot_scree(pca.explained_variance_ratio_,
               save_dir / f'{prefix}_scree.svg')

    # 4. t-SNE on PCA features
    perp = min(args.perplexity, N // 4, N - 1)
    if perp < 2:
        perp = 2
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42,
                init='pca', learning_rate='auto')
    tsne_feats = tsne.fit_transform(pca_feats)

    # 5. UMAP
    try:
        import umap
    except ImportError:
        print('  umap-learn not installed, skipping UMAP + HDBSCAN')
        umap_feats = None
    else:
        n_neighbors = min(args.umap_neighbors, N - 1)
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                            min_dist=0.1, random_state=42)
        umap_feats = reducer.fit_transform(pca_feats)

    # 6. HDBSCAN on UMAP
    cluster_labels = np.full(N, -1, dtype=int)
    if umap_feats is not None:
        try:
            import hdbscan
        except ImportError:
            print('  hdbscan not installed, skipping clustering')
        else:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=args.min_cluster_size,
                                        min_samples=args.min_samples)
            cluster_labels = clusterer.fit_predict(umap_feats)
            n_found = len(set(cluster_labels) - {-1})
            n_noise = (cluster_labels == -1).sum()
            print(f'  HDBSCAN: {n_found} clusters, {n_noise} noise points')

    # 7. Plots — PCA scatter
    plot_scatter(pca_feats[:, :2], group_ids,
                 f'{layer_name} — PCA (cell type)',
                 save_dir / f'{prefix}_pca.svg',
                 label_names=GROUP_NAMES, xlabel='PC1', ylabel='PC2')

    plot_scatter(pca_feats[:, :2], si,
                 f'{layer_name} — PCA (SI)',
                 save_dir / f'{prefix}_pca_si.svg',
                 is_continuous=True, xlabel='PC1', ylabel='PC2')

    # t-SNE scatter
    plot_scatter(tsne_feats, cluster_labels,
                 f'{layer_name} — t-SNE (HDBSCAN cluster)',
                 save_dir / f'{prefix}_tsne.svg',
                 xlabel='t-SNE 1', ylabel='t-SNE 2')

    plot_scatter(tsne_feats, group_ids,
                 f'{layer_name} — t-SNE (cell type)',
                 save_dir / f'{prefix}_tsne_celltype.svg',
                 label_names=GROUP_NAMES,
                 xlabel='t-SNE 1', ylabel='t-SNE 2')

    plot_scatter(tsne_feats, si,
                 f'{layer_name} — t-SNE (SI)',
                 save_dir / f'{prefix}_tsne_si.svg',
                 is_continuous=True,
                 xlabel='t-SNE 1', ylabel='t-SNE 2')

    # UMAP scatter
    if umap_feats is not None:
        plot_scatter(umap_feats, cluster_labels,
                     f'{layer_name} — UMAP (HDBSCAN cluster)',
                     save_dir / f'{prefix}_umap.svg',
                     xlabel='UMAP 1', ylabel='UMAP 2')

        plot_scatter(umap_feats, group_ids,
                     f'{layer_name} — UMAP (cell type)',
                     save_dir / f'{prefix}_umap_celltype.svg',
                     label_names=GROUP_NAMES,
                     xlabel='UMAP 1', ylabel='UMAP 2')

        plot_scatter(umap_feats, si,
                     f'{layer_name} — UMAP (SI)',
                     save_dir / f'{prefix}_umap_si.svg',
                     is_continuous=True,
                     xlabel='UMAP 1', ylabel='UMAP 2')

    # Cluster examples
    if (cluster_labels >= 0).any():
        plot_cluster_examples(tc, cluster_labels, np.nan_to_num(si),
                              save_dir / f'{prefix}_cluster_examples.svg')

    return {
        'pca_features': pca_feats,
        'pca_explained_var': pca.explained_variance_ratio_,
        'tsne_features': tsne_feats,
        'umap_features': umap_feats,
        'cluster_labels': cluster_labels,
        'group_ids': group_ids,
        'si': si,
    }


# ---------------------------------------------------------------------------
# Metric-space clustering (Isomap on per-neuron feature vectors)
# ---------------------------------------------------------------------------

METRIC_FEATURES = [
    'SI', 'EVs', 'morans_i', 'gearys_c', 'getis_ord_g', 'fieldsize', 'pf_peaks',
]

METRIC_DISPLAY = {
    'SI': 'SI', 'EVs': 'EV', 'morans_i': "Moran's I", 'gearys_c': "Geary's C",
    'getis_ord_g': "Getis-Ord G", 'fieldsize': 'Field size', 'pf_peaks': '# peaks',
}


def _build_feature_matrix(metrics):
    """Build (N, D) feature matrix from per-neuron metrics dict."""
    cols = []
    used_keys = []
    for key in METRIC_FEATURES:
        if key not in metrics:
            print(f'  Warning: metric {key!r} missing from pkl (old format?), skipping')
            continue
        v = np.asarray(metrics[key], dtype=float)
        cols.append(v)
        used_keys.append(key)
    if not cols:
        raise ValueError('No metric features found in pkl — re-run tuning_curve.py')
    X = np.column_stack(cols)  # (N, D)
    return X, used_keys


def plot_metric_covariance(X, used_keys, layer_name, save_path):
    """Heatmap of the Pearson correlation matrix across metric features.

    Low off-diagonal correlations indicate independent bases in the
    tuning-curve metric space.
    """
    # Correlation matrix (handles constant columns gracefully)
    with np.errstate(divide='ignore', invalid='ignore'):
        R = np.corrcoef(X, rowvar=False)
    R = np.nan_to_num(R, nan=0.0)

    display_names = [METRIC_DISPLAY.get(k, k) for k in used_keys]
    D = len(display_names)

    fig, ax = plt.subplots(figsize=(1.0 + 0.8 * D, 0.6 + 0.8 * D))
    im = ax.imshow(R, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Pearson r')

    ax.set_xticks(range(D))
    ax.set_yticks(range(D))
    ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(display_names, fontsize=8)

    # Annotate cells
    for i in range(D):
        for j in range(D):
            color = 'white' if abs(R[i, j]) > 0.6 else 'black'
            ax.text(j, i, f'{R[i, j]:.2f}', ha='center', va='center',
                    fontsize=7, color=color)

    # Summary stat: mean |r| of off-diagonal entries
    mask_offdiag = ~np.eye(D, dtype=bool)
    mean_abs_r = np.abs(R[mask_offdiag]).mean()
    ax.set_title(f'{layer_name} — metric correlation  '
                 f'(mean |r|_{{off-diag}} = {mean_abs_r:.3f})', fontsize=10)

    fig.tight_layout()
    _save(fig, save_path)


def process_layer_metrics(layer_name, layer_data, save_dir, args):
    """Isomap on metric feature vectors for one layer."""
    tc = layer_data['tuning_curves']
    metrics = layer_data['metrics']
    group_ids = layer_data['group_ids']
    si = metrics['SI']
    N = tc.shape[0]
    print(f'\n--- {layer_name} ({N} neurons) [metric-space] ---')

    if N < 10:
        print(f'  Skipping {layer_name}: too few neurons ({N})')
        return None

    # Build feature matrix and drop neurons with any NaN
    X, used_keys = _build_feature_matrix(metrics)
    valid = ~np.isnan(X).any(axis=1)
    X = X[valid]
    valid_idx = np.where(valid)[0]
    n_dropped = N - valid.sum()
    if n_dropped:
        print(f'  Dropped {n_dropped} neurons with NaN metrics')
    N_valid = X.shape[0]

    if N_valid < 10:
        print(f'  Skipping {layer_name}: too few valid neurons ({N_valid})')
        return None

    # Z-score normalize
    if args.normalize:
        X = StandardScaler().fit_transform(X)

    # Isomap
    n_neighbors = min(args.isomap_neighbors, N_valid - 1)
    isomap = Isomap(n_components=2, n_neighbors=n_neighbors)
    embedding = isomap.fit_transform(X)

    prefix = layer_name.replace('/', '_')

    # Static plots
    plot_scatter(embedding, group_ids[valid_idx],
                 f'{layer_name} — Isomap metrics (cell type)',
                 save_dir / f'{prefix}_isomap_metrics_celltype.svg',
                 label_names=GROUP_NAMES,
                 xlabel='Isomap 1', ylabel='Isomap 2')

    plot_scatter(embedding, si[valid_idx],
                 f'{layer_name} — Isomap metrics (SI)',
                 save_dir / f'{prefix}_isomap_metrics_si.svg',
                 is_continuous=True,
                 xlabel='Isomap 1', ylabel='Isomap 2')

    # Metric correlation heatmap (independence of bases)
    plot_metric_covariance(X, used_keys, layer_name,
                           save_dir / f'{prefix}_metric_corr.png')

    return {
        'feature_matrix': X,
        'isomap_embedding': embedding,
        'valid_idx': valid_idx,
        'group_ids': group_ids[valid_idx],
        'si': si[valid_idx],
        'metrics': {k: np.asarray(metrics[k], dtype=float)[valid_idx] for k in used_keys},
        'used_keys': used_keys,
    }


def _interactive_metric_scatter(layer_name, layer_data, result):
    """Interactive Isomap scatter — click a point to show tuning curve + metrics."""
    tc_array = layer_data['tuning_curves']  # full (N_total, H, W)
    embedding = result['isomap_embedding']
    valid_idx = result['valid_idx']         # maps embed row → original neuron id
    group_ids = result['group_ids']         # already subsetted to valid
    metrics = result['metrics']             # already subsetted to valid

    fig, (ax_scatter, ax_tc) = plt.subplots(
        1, 2, figsize=(13, 5.5), gridspec_kw={'width_ratios': [1, 1]})

    # Color by cell type
    cmap = _group_cmap()
    for gid, gname in enumerate(GROUP_NAMES):
        mask = group_ids == gid
        if not mask.any():
            continue
        ax_scatter.scatter(embedding[mask, 0], embedding[mask, 1],
                           s=12, alpha=0.5, color=cmap(gid),
                           label=gname, picker=True)
    ax_scatter.legend(fontsize=7, markerscale=2, loc='best', framealpha=0.7)
    ax_scatter.set_xlabel('Isomap 1')
    ax_scatter.set_ylabel('Isomap 2')
    ax_scatter.set_title(f'{layer_name}: metric-space Isomap  (click a point)')

    ax_tc.set_title('Tuning curve')
    ax_tc.axis('off')
    ax_tc.text(0.5, 0.5, 'Click a point\nin the scatter',
               ha='center', va='center', transform=ax_tc.transAxes,
               fontsize=12, color='grey')

    highlight = ax_scatter.scatter([], [], s=80, facecolors='none',
                                   edgecolors='red', linewidths=2, zorder=5)

    # Build a map from each scatter collection → indices into the valid subset.
    # matplotlib picker returns ind relative to the PathCollection, so we track
    # which valid-subset rows each collection holds.
    collection_valid_rows = []
    for gid in range(len(GROUP_NAMES)):
        mask = group_ids == gid
        if mask.any():
            collection_valid_rows.append(np.where(mask)[0])
    n_collections = len(collection_valid_rows)

    def on_pick(event):
        artist = event.artist
        scatter_collections = ax_scatter.collections[:n_collections]
        if artist not in scatter_collections:
            return
        coll_idx = scatter_collections.index(artist)
        local_ind = event.ind[0]
        valid_row = collection_valid_rows[coll_idx][local_ind]
        neuron_idx = valid_idx[valid_row]  # original neuron id

        # Highlight
        highlight.set_offsets([embedding[valid_row]])

        # Tuning curve from original array
        ax_tc.clear()
        tc = tc_array[neuron_idx]
        ax_tc.imshow(np.ma.masked_invalid(tc.T), origin='lower',
                     interpolation='nearest', cmap='hot')

        # Metric summary text
        lines = [f'Neuron {neuron_idx}  ({GROUP_NAMES[group_ids[valid_row]]})']
        for key in result.get('used_keys', METRIC_FEATURES):
            if key not in metrics:
                continue
            val = float(metrics[key][valid_row])
            lines.append(f'  {METRIC_DISPLAY.get(key, key)}: {val:.4f}')
        ax_tc.set_title('\n'.join(lines), fontsize=8, loc='left', family='monospace')
        ax_tc.set_xlabel('x')
        ax_tc.set_ylabel('y')
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('pick_event', on_pick)
    fig.tight_layout()
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Metric distributions (histogram + example tuning curves at quantiles)
# ---------------------------------------------------------------------------

# All metrics to iterate over in distributions mode
DIST_METRICS = [
    ('SI', 'Spatial Information'),
    ('EVs', 'Explained Variance'),
    ('morans_i', "Moran's I"),
    ('gearys_c', "Geary's C (lower = more spatial)"),
    ('getis_ord_g', "Getis-Ord G"),
    ('fieldsize', 'Field size'),
    ('pf_peaks', '# peaks'),
]


def plot_metric_distribution(tc_array, metric_values, metric_name,
                             display_name, layer_name, save_path,
                             n_per_quantile=3,
                             quantiles=(0.1, 0.3, 0.5, 0.7, 0.9)):
    """Combined figure: histogram of metric + example TCs at quantile positions.

    Top panel: density histogram with vertical lines at each quantile.
    Bottom panel: grid of example tuning curves (n_quantiles cols x n_per_quantile rows).
    """
    valid = np.isfinite(metric_values)
    if valid.sum() < 10:
        print(f'  Skipping {metric_name}: too few finite values ({valid.sum()})')
        return

    vals = metric_values[valid]
    valid_idx = np.where(valid)[0]

    n_q = len(quantiles)
    quantile_values = np.quantile(vals, quantiles)

    fig = plt.figure(figsize=(3.2 * n_q, 2.5 + 2.2 * n_per_quantile))
    gs = fig.add_gridspec(1 + n_per_quantile, n_q,
                          height_ratios=[2.5] + [2.0] * n_per_quantile,
                          hspace=0.4, wspace=0.3)

    # --- Top panel: histogram spanning all columns ---
    ax_hist = fig.add_subplot(gs[0, :])
    ax_hist.hist(vals, bins=min(50, max(20, len(vals) // 20)),
                 density=True, color='steelblue', alpha=0.7, edgecolor='white',
                 linewidth=0.5)
    for i, (q, qv) in enumerate(zip(quantiles, quantile_values)):
        ax_hist.axvline(qv, color='crimson', linestyle='--', linewidth=1, alpha=0.8)
        ax_hist.text(qv, ax_hist.get_ylim()[1] * 0.95, f'{int(q*100)}%',
                     ha='center', va='top', fontsize=7, color='crimson',
                     fontweight='bold')
    ax_hist.set_xlabel(display_name)
    ax_hist.set_ylabel('Density')
    ax_hist.set_title(f'{layer_name} — {display_name} distribution (N={len(vals)})')

    # --- Bottom panel: example tuning curves at each quantile ---
    for qi, (q, qv) in enumerate(zip(quantiles, quantile_values)):
        # Find neuron closest to this quantile value
        dists = np.abs(vals - qv)
        sorted_by_dist = np.argsort(dists)
        selected = sorted_by_dist[:n_per_quantile]

        for ri, sel in enumerate(selected):
            ax = fig.add_subplot(gs[1 + ri, qi])
            neuron_orig = valid_idx[sel]
            tc = tc_array[neuron_orig]
            ax.imshow(np.ma.masked_invalid(tc.T), origin='lower',
                      interpolation='nearest', cmap='hot')
            ax.set_xticks([])
            ax.set_yticks([])
            val_str = f'{vals[sel]:.3f}'
            ax.set_title(f'n{neuron_orig} ({val_str})', fontsize=7)
            if ri == 0:
                ax.text(0.5, 1.35, f'Q{int(q*100)} = {qv:.3f}',
                        ha='center', va='bottom', transform=ax.transAxes,
                        fontsize=8, fontweight='bold', color='crimson')

    _save(fig, save_path)


def process_layer_distributions(layer_name, layer_data, save_dir, args):
    """Generate metric distribution plots for one layer."""
    tc = layer_data['tuning_curves']
    metrics = layer_data['metrics']
    N = tc.shape[0]
    print(f'\n--- {layer_name} ({N} neurons) [distributions] ---')

    if N < 10:
        print(f'  Skipping {layer_name}: too few neurons ({N})')
        return None

    prefix = layer_name.replace('/', '_')
    summary = {}

    for metric_key, display_name in DIST_METRICS:
        if metric_key not in metrics:
            print(f'  {metric_key}: not found, skipping')
            continue
        vals = np.asarray(metrics[metric_key], dtype=float)
        save_path = save_dir / f'{prefix}_dist_{metric_key}.png'
        plot_metric_distribution(tc, vals, metric_key, display_name,
                                 layer_name, save_path)
        finite = vals[np.isfinite(vals)]
        if len(finite) > 0:
            summary[metric_key] = {
                'mean': float(np.mean(finite)),
                'std': float(np.std(finite)),
                'median': float(np.median(finite)),
                'n_finite': int(len(finite)),
                'n_total': int(len(vals)),
            }

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Tuning curve clustering via PCA/t-SNE/UMAP + HDBSCAN')
    parser.add_argument('--from_pkl', required=True,
                        help='Path to tuning_results.pkl')
    parser.add_argument('--save', required=True,
                        help='Output directory for plots and results')
    parser.add_argument('--layers', nargs='*', default=None,
                        help='Layer filter (e.g. dyn/deter dyn/stoch)')
    parser.add_argument('--mode', choices=['autocorr', 'metrics', 'distributions'],
                        default='autocorr',
                        help='autocorr: PCA/t-SNE/UMAP on autocorrelation maps. '
                             'metrics: Isomap on per-neuron metric feature vectors. '
                             'distributions: per-metric histogram + example TCs at '
                             'quantile positions')
    parser.add_argument('--interactive', action='store_true', default=False,
                        help='Launch interactive click-to-inspect viewer '
                             '(metrics mode only)')
    parser.add_argument('--n_components', type=int, default=50,
                        help='Number of PCA components for t-SNE/UMAP input')
    parser.add_argument('--perplexity', type=float, default=30,
                        help='t-SNE perplexity')
    parser.add_argument('--umap_neighbors', type=int, default=15,
                        help='UMAP n_neighbors')
    parser.add_argument('--isomap_neighbors', type=int, default=15,
                        help='Isomap n_neighbors (metrics mode)')
    parser.add_argument('--min_cluster_size', type=int, default=20,
                        help='HDBSCAN min_cluster_size')
    parser.add_argument('--min_samples', type=int, default=5,
                        help='HDBSCAN min_samples')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='Z-score normalize features')
    parser.add_argument('--no_normalize', action='store_false', dest='normalize')

    args = parser.parse_args()

    # Load
    print(f'Loading {args.from_pkl}')
    with open(args.from_pkl, 'rb') as f:
        data = pickle.load(f)

    layers_data = data['layers']
    all_layers = list(layers_data.keys())

    # Filter layers
    if args.layers:
        all_layers = [ln for ln in all_layers if ln in args.layers]
    all_layers = _order_layers(all_layers)

    if not all_layers:
        print('No matching layers found.')
        return

    print(f'Layers: {all_layers}')

    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == 'distributions':
        # ----- Metric distribution plots -----
        all_results = {}
        for ln in all_layers:
            result = process_layer_distributions(ln, layers_data[ln], save_dir, args)
            if result is not None:
                all_results[ln] = result

        out_path = save_dir / 'distribution_results.pkl'
        with open(out_path, 'wb') as f:
            pickle.dump({
                'metadata': {
                    'source_pkl': str(args.from_pkl),
                    'mode': 'distributions',
                },
                'layers': all_results,
            }, f)
        print(f'\nSaved distribution_results.pkl to {out_path}')

    elif args.mode == 'metrics':
        # ----- Metric-space Isomap pipeline -----
        all_results = {}
        for ln in all_layers:
            result = process_layer_metrics(ln, layers_data[ln], save_dir, args)
            if result is not None:
                all_results[ln] = result

        # Interactive viewer
        if args.interactive:
            # Pick layer
            available = list(all_results.keys())
            if len(available) == 1:
                chosen = available[0]
            else:
                print('\nAvailable layers:')
                for i, ln in enumerate(available):
                    print(f'  [{i}] {ln}')
                idx = int(input('Select layer index: '))
                chosen = available[idx]
            _interactive_metric_scatter(
                chosen, layers_data[chosen], all_results[chosen])

        # Save
        out_path = save_dir / 'metric_cluster_results.pkl'
        with open(out_path, 'wb') as f:
            pickle.dump({
                'metadata': {
                    'source_pkl': str(args.from_pkl),
                    'mode': 'metrics',
                    'isomap_neighbors': args.isomap_neighbors,
                    'normalize': args.normalize,
                    'features': list(set().union(*(r['used_keys'] for r in all_results.values() if r))),
                },
                'layers': all_results,
            }, f)
        print(f'\nSaved metric_cluster_results.pkl to {out_path}')

    else:
        # ----- Original autocorrelation pipeline -----
        all_results = {}
        for ln in all_layers:
            result = process_layer(ln, layers_data[ln], save_dir, args)
            if result is not None:
                all_results[ln] = result

        out_path = save_dir / 'cluster_results.pkl'
        with open(out_path, 'wb') as f:
            pickle.dump({
                'metadata': {
                    'source_pkl': str(args.from_pkl),
                    'n_components': args.n_components,
                    'perplexity': args.perplexity,
                    'umap_neighbors': args.umap_neighbors,
                    'min_cluster_size': args.min_cluster_size,
                    'min_samples': args.min_samples,
                    'normalize': args.normalize,
                },
                'layers': all_results,
            }, f)
        print(f'\nSaved cluster_results.pkl to {out_path}')


if __name__ == '__main__':
    main()
