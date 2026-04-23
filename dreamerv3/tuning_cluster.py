#!/usr/bin/env python3
"""Tuning curve clustering via dimensionality reduction.

Runs PCA / t-SNE / UMAP on spatial autocorrelation maps of tuning curves
to discover emergent clusters of spatial tuning patterns.  HDBSCAN is used
for automatic cluster discovery on the UMAP embedding.

Usage:
    python dreamerv3/tuning_cluster.py \
        --from_pkl ./tuning_results/tuning_results.pkl \
        --save ./tuning_results/cluster_plots \
        --layers dyn/deter dyn/stoch \
        --n_components 50 --perplexity 30 \
        --umap_neighbors 15 --min_cluster_size 20
"""

import argparse
import pickle
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate2d
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
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
    parser.add_argument('--n_components', type=int, default=50,
                        help='Number of PCA components for t-SNE/UMAP input')
    parser.add_argument('--perplexity', type=float, default=30,
                        help='t-SNE perplexity')
    parser.add_argument('--umap_neighbors', type=int, default=15,
                        help='UMAP n_neighbors')
    parser.add_argument('--min_cluster_size', type=int, default=20,
                        help='HDBSCAN min_cluster_size')
    parser.add_argument('--min_samples', type=int, default=5,
                        help='HDBSCAN min_samples')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='Z-score normalize flattened autocorrelations')
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

    # Process each layer
    all_results = {}
    for ln in all_layers:
        result = process_layer(ln, layers_data[ln], save_dir, args)
        if result is not None:
            all_results[ln] = result

    # Save combined results
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
