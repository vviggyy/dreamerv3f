"""
Spatial tuning curve analysis for DreamerV3 world model representations.

Classifies neurons into spatial cell types (place cells, border cells, HD cells,
etc.) using tuning curves and derived metrics, following the approach in the pRNN
repo (TuningCurveAnalysis.py). Uses pynapple for tuning curve and mutual
information computations.

Usage:
  # Full analysis (all layers, parallel):
  python dreamerv3/tuning_curve.py \
    --data ./logdir/crafter_small_1m/trajectories \
    --save ./logdir/crafter_small_1m/tuning_results \
    --n_jobs -1

  # With held-out data for EV reliability:
  python dreamerv3/tuning_curve.py \
    --data ./logdir/.../trajectories_train \
    --test_data ./logdir/.../trajectories_test \
    --save ./logdir/.../tuning_results

  # Filter specific layers:
  python dreamerv3/tuning_curve.py \
    --data ./logdir/.../trajectories \
    --save ./results --layers dyn/deter dyn/stoch

  # Interactive viewer from precomputed results (no recomputation):
  python dreamerv3/tuning_curve.py --from_pkl tuning_results.pkl
  python dreamerv3/tuning_curve.py --from_pkl tuning_results.pkl --layers dyn/deter
"""

import argparse
import gc
import pickle
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter, label, maximum_filter
from scipy.signal import correlate2d
def _tc_cmap():
    """Viridis colormap with white for NaN/masked (unvisited bins)."""
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('white')
    return cmap


# Reuse data loading from decode_position
from run_info import log_run_info
from decode_position import (
    LAYER_ORDER,
    _prepare_single_layer,
    filter_stuck_episodes,
    load_episodes,
    prepare_data,
    prepare_data_layers,
)


def _discover_layer_names(episodes):
    """Return sorted list of act/* layer names without loading activations."""
    keys = set()
    for ep in episodes:
        keys.update(k[len('act/'):] for k in ep if k.startswith('act/'))
    return sorted(keys)


def _strip_heavy_keys(episodes):
    """Drop image data from episodes in-place to free memory."""
    for ep in episodes:
        for k in ['image', 'images']:
            if k in ep:
                del ep[k]

# ---------------------------------------------------------------------------
# Autocorrelation metrics (from pRNN TuningCurveAnalysis.py)
# ---------------------------------------------------------------------------

def pf_autocorr(tuning_curves_array, peak_norm=True):
    """2D autocorrelation of each neuron's tuning curve.

    Args:
        tuning_curves_array: (N_neurons, H, W) array of tuning curves.
        peak_norm: Normalize each autocorrelation by its peak.

    Returns:
        (N_neurons, 2H-1, 2W-1) array of autocorrelations.
    """
    results = []
    for tc in tuning_curves_array:
        ac = correlate2d(np.nan_to_num(tc), np.nan_to_num(tc), mode='full')
        if peak_norm and ac.max() > 0:
            ac = ac / ac.max()
        results.append(ac)
    return np.array(results)


def count_autocorr_peaks(autocorr, size=3, threshold=0.15):
    """Count local maxima in autocorrelation map."""
    local_max = (maximum_filter(autocorr, size=size) == autocorr) & (
        autocorr > threshold
    )
    _, num_features = label(local_max)
    return num_features


def calculate_field_size(tc_autocorr, threshold=0.5):
    """Size of central field in autocorrelation (sqrt of area)."""
    field = tc_autocorr > threshold
    labeled, num_features = label(field)
    if num_features == 0:
        return np.nan
    whicharea = labeled[tc_autocorr == np.max(tc_autocorr)]
    centerlabeled = labeled == whicharea
    return np.sqrt(np.sum(centerlabeled))


def calculate_field_asymmetry(tc_autocorr, threshold=0.5):
    """Major/minor axis ratio of central autocorrelation field."""
    field = tc_autocorr > threshold
    labeled, num_features = label(field)
    if num_features == 0:
        return np.nan
    whicharea = labeled[tc_autocorr == np.max(tc_autocorr)]
    centerlabeled = labeled == whicharea
    coords = np.column_stack(np.nonzero(centerlabeled))
    if len(coords) < 2:
        return 1.0
    yc, xc = field.shape[0] // 2, field.shape[1] // 2
    coords = coords - [yc, xc]
    cov = np.cov(coords.T)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)[::-1]
    major_axis = 2 * np.sqrt(max(eigvals[0], 0))
    minor_axis = max(2 * np.sqrt(max(eigvals[1], 0)), 1.0)
    return major_axis / minor_axis


# ---------------------------------------------------------------------------
# EV reliability (from pRNN TuningCurveAnalysis.py makeFAKEdata)
# ---------------------------------------------------------------------------

def compute_ev_reliability(activations, positions, tuning_curves_array, area):
    """Explained variance of tuning curve predictions.

    For each neuron, look up the tuning curve value at visited positions
    and compute EV = 1 - Var(residual) / Var(real).

    Args:
        activations: (N, D) actual neural activations.
        positions: (N, 2) integer (x, y) positions.
        tuning_curves_array: (D, H, W) tuning curves.
        area: (width, height) of the map.

    Returns:
        ev: (D,) explained variance per neuron.
    """
    N, D = activations.shape
    px = np.clip(positions[:, 0].astype(int), 0, area[0] - 1)
    py = np.clip(positions[:, 1].astype(int), 0, area[1] - 1)
    predicted = np.zeros_like(activations)
    for d in range(D):
        tc = tuning_curves_array[d]
        if np.isnan(tc).all():
            continue
        predicted[:, d] = np.nan_to_num(tc[px, py])
    residual = activations - predicted
    var_real = np.var(activations, axis=0)
    var_residual = np.var(residual, axis=0)
    ev = 1.0 - var_residual / np.where(var_real > 0, var_real, 1.0)
    ev[var_real == 0] = 0.0
    ev[np.isinf(ev)] = 0.0
    return ev


# ---------------------------------------------------------------------------
# Cell classification (from pRNN TuningCurveAnalysis.py groupCells)
# ---------------------------------------------------------------------------

GROUP_NAMES = [
    'untuned', 'HD_cells', 'single_field', 'border_cells',
    'spatial_HD', 'complex_cells', 'dead',
]


def classify_cells(metrics, SI_thresh=0.5, EV_unthresh=0.15, HD_thresh=0.5,
                   border_thresh=0, border_symmetrythresh=3, EV_thresh=0.5,
                   place_symmetrythresh=3):
    """Classify neurons into cell types based on spatial metrics.

    Returns:
        groups: dict {group_name: bool array}
        group_ids: int array of group index per neuron
    """
    SI = metrics['SI']
    EVs = metrics['EVs']
    HD_info = metrics['HD_info']
    border_score = metrics['border_score']
    pf_peaks = metrics['pf_peaks']
    fieldasymmetry = metrics['fieldasymmetry']
    fieldsize = metrics['fieldsize']

    dead = np.isnan(fieldasymmetry) & np.isnan(fieldsize)
    untuned = (
        ~dead
        & (EVs <= EV_unthresh)
        & (SI <= SI_thresh)
        & (HD_info <= HD_thresh)
    )
    HD_cells = (
        ~dead
        & (EVs <= EV_unthresh)
        & (SI <= SI_thresh)
        & (HD_info > HD_thresh)
    )
    border_cells = (
        ~untuned & ~HD_cells & ~dead
        & (border_score > border_thresh)
        & (fieldasymmetry > border_symmetrythresh)
    )
    single_field = (
        ~untuned & ~HD_cells & ~border_cells & ~dead
        & (pf_peaks == 1)
        & (EVs > EV_thresh)
        & (fieldasymmetry < place_symmetrythresh)
    )
    spatial_HD = (
        ~untuned & ~HD_cells & ~border_cells & ~single_field & ~dead
        & (SI > SI_thresh)
        & (HD_info > HD_thresh)
    )
    complex_cells = (
        ~border_cells & ~single_field & ~untuned
        & ~HD_cells & ~spatial_HD & ~dead
    )

    groups = {
        'untuned': untuned,
        'HD_cells': HD_cells,
        'single_field': single_field,
        'border_cells': border_cells,
        'spatial_HD': spatial_HD,
        'complex_cells': complex_cells,
        'dead': dead,
    }
    group_ids = np.argmax(np.column_stack(list(groups.values())), axis=1)
    return groups, group_ids


# ---------------------------------------------------------------------------
# Pynapple-based tuning curve + mutual information computation
# ---------------------------------------------------------------------------

def build_pynapple_objects(positions, activations, groups, area, facing=None):
    """Build pynapple TsdFrame objects from concatenated episode data.

    Each episode gets a separate time epoch with gaps between episodes
    so pynapple treats them as separate intervals.

    Args:
        positions: (N, 2) array of (x, y) positions.
        activations: (N, D) array of neural activations.
        groups: (N,) array of episode indices.
        area: (width, height) tuple.
        facing: (N, 2) optional array of (dx, dy) facing direction.

    Returns:
        rates: nap.TsdFrame of activations
        position: nap.TsdFrame of (x, y)
        hd: nap.Tsd of head direction angle (or None)
        epoch: nap.IntervalSet
    """
    unique_eps = np.unique(groups)
    gap = 1000  # time gap between episodes

    timestamps = np.zeros(len(positions))
    starts, ends = [], []
    t_offset = 0
    for ep_idx in unique_eps:
        mask = groups == ep_idx
        n_steps = mask.sum()
        t_start = t_offset
        t_end = t_offset + n_steps - 1
        timestamps[mask] = np.arange(t_start, t_end + 1)
        starts.append(t_start)
        ends.append(t_end)
        t_offset = t_end + gap

    epoch = nap.IntervalSet(start=starts, end=ends)

    position = nap.TsdFrame(
        t=timestamps, d=positions.astype(float),
        columns=['x', 'y'], time_support=epoch,
    )
    rates = nap.TsdFrame(
        t=timestamps, d=activations.astype(float),
        time_support=epoch,
    )

    hd = None
    if facing is not None:
        angles = np.arctan2(facing[:, 1].astype(float),
                            facing[:, 0].astype(float))
        hd = nap.Tsd(t=timestamps, d=angles, time_support=epoch)

    return rates, position, hd, epoch


def compute_tuning_curves_and_si(rates, position, epoch, area, smooth_sigma=0):
    """Compute 2D spatial tuning curves and mutual information via pynapple.

    Args:
        smooth_sigma: Gaussian smoothing sigma (in bins) applied to tuning
            curves before computing SI. 0 = no smoothing.

    Returns:
        tc_array: (N_neurons, area_x, area_y) tuning curves
        si_values: (N_neurons,) spatial information in bits/spike
    """
    nb_bins = (area[0], area[1])
    minmax = (-0.5, area[0] - 0.5, -0.5, area[1] - 0.5)

    place_fields, _ = nap.compute_2d_tuning_curves_continuous(
        rates, position, ep=epoch,
        nb_bins=nb_bins, minmax=minmax,
    )

    # Optional Gaussian smoothing before SI computation
    if smooth_sigma > 0:
        for neuron_idx in place_fields:
            tc = place_fields[neuron_idx]
            tc = np.nan_to_num(tc, nan=0.0)
            place_fields[neuron_idx] = gaussian_filter(tc, sigma=smooth_sigma)

    si_df = nap.compute_2d_mutual_info(
        place_fields, position, position.time_support, bitssec=False, minmax=minmax
    )

    # Convert place_fields dict to array (keep NaN for unvisited bins)
    n_neurons = rates.shape[1]
    tc_array = np.full((n_neurons, area[0], area[1]), np.nan)
    for neuron_idx in place_fields:
        tc_array[neuron_idx] = place_fields[neuron_idx]

    si_values = si_df['SI'].values

    return tc_array, si_values


def compute_hd_info(rates, hd, epoch):
    """Compute head direction tuning curves and mutual information.

    Returns:
        hd_info: (N_neurons,) HD mutual information
    """
    if hd is None:
        return np.zeros(rates.shape[1])

    # 4 cardinal directions for Crafter: up/down/left/right
    # atan2 values: right=0, up=pi/2, left=pi, down=-pi/2
    hd_tc = nap.compute_1d_tuning_curves_continuous(
        rates, hd, ep=epoch,
        nb_bins=4, minmax=(-np.pi, np.pi),
    )
    hd_mi = nap.compute_1d_mutual_info(
        hd_tc, hd, hd.time_support,
    )
    return hd_mi['SI'].values


# ---------------------------------------------------------------------------
# Per-layer analysis pipeline
# ---------------------------------------------------------------------------

def analyze_layer(layer_name, activations, positions, groups, area,
                  facing=None, test_activations=None, test_positions=None,
                  test_groups=None, compute_hd=True, smooth_sigma=0):
    """Full tuning curve analysis for one layer.

    Args:
        layer_name: str identifier.
        activations: (N, D) array of neural activations (train set).
        positions: (N, 2) positions.
        groups: (N,) episode indices.
        area: (width, height).
        facing: (N, 2) optional facing direction.
        test_activations: optional held-out activations for EV.
        test_positions: optional held-out positions for EV.
        test_groups: optional held-out episode groups.
        compute_hd: whether to compute HD metrics.
        smooth_sigma: Gaussian smoothing sigma (bins) for tuning curves before SI.

    Returns:
        dict with tuning_curves, metrics, groups, group_ids.
    """
    n_neurons = activations.shape[1]
    print(f"  [{layer_name}] {n_neurons} neurons, {len(activations)} samples")

    # Build pynapple objects
    rates, position, hd, epoch = build_pynapple_objects(
        positions, activations, groups, area,
        facing=facing if compute_hd else None,
    )

    # Tuning curves + spatial info
    tc_array, si_values = compute_tuning_curves_and_si(
        rates, position, epoch, area, smooth_sigma=smooth_sigma,
    )

    # HD info
    if compute_hd and facing is not None:
        hd_info = compute_hd_info(rates, hd, epoch)
    else:
        hd_info = np.zeros(n_neurons)

    # EV reliability: use test data if available, else train data
    if test_activations is not None and test_positions is not None:
        ev_act = test_activations
        ev_pos = test_positions
    else:
        ev_act = activations
        ev_pos = positions
    evs = compute_ev_reliability(ev_act, ev_pos, tc_array, area)

    # Autocorrelation metrics
    autocorrs = pf_autocorr(tc_array, peak_norm=True)
    peaks = np.array([count_autocorr_peaks(ac) for ac in autocorrs])
    peaks = (peaks + 1) // 2
    fieldsizes = np.array([calculate_field_size(ac) for ac in autocorrs])
    fieldasymmetries = np.array([calculate_field_asymmetry(ac) for ac in autocorrs])

    # Border score: NaN (no terrain data at analysis time)
    border_score = np.full(n_neurons, np.nan)

    metrics = {
        'SI': si_values,
        'HD_info': hd_info,
        'EVs': evs,
        'border_score': border_score,
        'pf_peaks': peaks,
        'fieldsize': fieldsizes,
        'fieldasymmetry': fieldasymmetries,
    }

    # Cell classification
    cell_groups, group_ids = classify_cells(metrics)

    return {
        'layer_name': layer_name,
        'tuning_curves': tc_array,
        'metrics': metrics,
        'cell_groups': cell_groups,
        'group_ids': group_ids,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_si_ev_scatter(metrics, group_ids, layer_name, save_path,
                       tc_array=None, interactive=False):
    """SI vs EV scatter. If interactive=True and tc_array provided, click a
    point to display its tuning curve in a side panel."""
    if interactive and tc_array is not None:
        return _interactive_si_ev(metrics, layer_name, tc_array)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(metrics['SI'], metrics['EVs'], s=8, alpha=0.5, color='steelblue')
    ax.set_xlabel('Spatial Information (bits/spike)')
    ax.set_ylabel('Explained Variance')
    ax.set_title(f'{layer_name}: SI vs EV')
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def _interactive_si_ev(metrics, layer_name, tc_array):
    """Interactive SI vs EV scatter — click a point to show its tuning curve."""
    si = metrics['SI']
    ev = metrics['EVs']

    fig, (ax_scatter, ax_tc) = plt.subplots(1, 2, figsize=(11, 5),
                                             gridspec_kw={'width_ratios': [1, 1]})
    ax_scatter.scatter(si, ev, s=10, alpha=0.5, color='steelblue', picker=True)
    ax_scatter.set_xlabel('Spatial Information (bits/spike)')
    ax_scatter.set_ylabel('Explained Variance')
    ax_scatter.set_title(f'{layer_name}: SI vs EV  (click a point)')

    ax_tc.set_title('Tuning curve')
    ax_tc.axis('off')
    ax_tc.text(0.5, 0.5, 'Click a point\nin the scatter',
               ha='center', va='center', transform=ax_tc.transAxes,
               fontsize=12, color='grey')

    highlight = ax_scatter.scatter([], [], s=60, facecolors='none',
                                   edgecolors='red', linewidths=1.5, zorder=5)

    def on_pick(event):
        ind = event.ind[0]
        # Update highlight
        highlight.set_offsets([[si[ind], ev[ind]]])
        # Draw tuning curve
        ax_tc.clear()
        tc = tc_array[ind]
        im = ax_tc.imshow(np.ma.masked_invalid(tc.T), origin='lower',
                          interpolation='nearest', cmap=_tc_cmap())
        ax_tc.set_title(f'Neuron {ind}  SI={si[ind]:.3f}  EV={ev[ind]:.3f}')
        for spine in ax_tc.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(0.8)
        ax_tc.set_xlabel('x')
        ax_tc.set_ylabel('y')
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('pick_event', on_pick)
    fig.tight_layout()
    plt.show()
    return fig


def plot_cell_types(group_ids, layer_name, save_path):
    """Histogram of cell type fractions."""
    fig, ax = plt.subplots(figsize=(7, 4))
    cmap = plt.get_cmap('viridis', len(GROUP_NAMES))
    counts, bins, patches = ax.hist(
        group_ids,
        bins=np.arange(-0.5, len(GROUP_NAMES) + 0.5),
        density=True,
    )
    for gidx, patch in enumerate(patches):
        patch.set_facecolor(cmap(gidx))
    ax.set_xticks(range(len(GROUP_NAMES)))
    ax.set_xticklabels(GROUP_NAMES, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Fraction')
    ax.set_title(f'{layer_name}: Cell Type Distribution')
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def plot_example_tuning_curves(tc_array, metrics, group_ids, layer_name,
                               save_path, n_examples=20, area=None,
                               sort_by='SI'):
    """Grid of example tuning curves sorted by SI or EV."""
    n_neurons = tc_array.shape[0]
    n_show = min(n_examples, n_neurons)
    metric_key = 'EVs' if sort_by == 'EV' else 'SI'
    order = np.argsort(metrics[metric_key])[::-1]
    selected = order[:n_show]

    ncols = min(5, n_show)
    nrows = int(np.ceil(n_show / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    for idx, neuron_idx in enumerate(selected):
        r, c = idx // ncols, idx % ncols
        ax = axes[r, c]
        tc = tc_array[neuron_idx]
        im = ax.imshow(np.ma.masked_invalid(tc.T), origin='lower',
                       interpolation='nearest', cmap=_tc_cmap())
        val = metrics[metric_key][neuron_idx]
        ax.set_title(f'n{neuron_idx}\n{sort_by}={val:.2f}',
                     fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(0.8)

    # Hide unused axes
    for idx in range(n_show, nrows * ncols):
        r, c = idx // ncols, idx % ncols
        axes[r, c].axis('off')

    fig.suptitle(f'{layer_name}: Top Tuning Curves (by {sort_by})', fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def plot_layer_si_ev_grid(all_results, save_path, m=5, seed=None):
    """N×(1+M) grid: SI vs EV scatter (col 0) + M sampled tuning curves per layer.

    Sampled neurons are highlighted on the scatter with colored circles matching
    the tuning curve border color.

    Args:
        all_results: list of dicts with 'layer_name', 'metrics', 'tuning_curves'.
        save_path: output file path.
        m: number of sampled tuning curves per layer.
        seed: random seed for neuron sampling (None = random).
    """
    if seed is None:
        seed = np.random.randint(0, 2**31)
    layer_names = [r['layer_name'] for r in all_results]
    ordered = get_sorted_layers(layer_names)
    res_map = {r['layer_name']: r for r in all_results}
    display_order = [ln for ln in ordered if ln in res_map]
    n = len(display_order)
    if n == 0:
        return

    rng = np.random.RandomState(seed)
    ncols = 1 + m
    # Distinct colors for each sampled neuron
    sample_colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
                     '#911eb4', '#42d4f4', '#f032e6'][:m]
    fig, axes = plt.subplots(n, ncols, figsize=(2.5 * ncols, 2.5 * n),
                              squeeze=False)

    for row, ln in enumerate(display_order):
        res = res_map[ln]
        si = res['metrics']['SI']
        ev = res['metrics']['EVs']
        tc_array = res.get('tuning_curves')

        # Col 0: SI vs EV scatter
        ax = axes[row, 0]
        ax.scatter(si, ev, s=4, alpha=0.3, color='steelblue', rasterized=True)
        ax.set_xlabel('SI', fontsize=7)
        ax.set_ylabel('EV', fontsize=7)
        ax.set_title(ln, fontsize=8, fontweight='bold')
        ax.tick_params(labelsize=6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Cols 1..m: sampled tuning curves
        if tc_array is None or len(tc_array) == 0:
            for c in range(1, ncols):
                axes[row, c].axis('off')
            continue

        n_neurons = tc_array.shape[0]
        # Sample from top 10th percentile of EV
        ev_cutoff = np.percentile(ev, 90)
        top_ev_idx = np.where(ev >= ev_cutoff)[0]
        if len(top_ev_idx) == 0:
            top_ev_idx = np.arange(n_neurons)
        chosen = rng.choice(top_ev_idx, size=min(m, len(top_ev_idx)), replace=False)

        # Highlight chosen neurons on the scatter with colored circles
        for c_idx, neuron_idx in enumerate(chosen):
            color = sample_colors[c_idx % len(sample_colors)]
            ax.scatter(si[neuron_idx], ev[neuron_idx],
                       s=50, facecolors='none', edgecolors=color,
                       linewidths=1.5, zorder=5)

        for c_idx, neuron_idx in enumerate(chosen):
            color = sample_colors[c_idx % len(sample_colors)]
            tc_ax = axes[row, 1 + c_idx]
            tc = tc_array[neuron_idx]
            tc_ax.imshow(np.ma.masked_invalid(tc.T), origin='lower',
                         interpolation='nearest', aspect='equal',
                         cmap=_tc_cmap())
            tc_ax.set_title(
                f'n{neuron_idx}\nSI={si[neuron_idx]:.2f} EV={ev[neuron_idx]:.2f}',
                fontsize=6, color=color)
            tc_ax.tick_params(labelsize=5)
            # Colored border to match scatter highlight
            for spine in tc_ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2.5)

        # Hide unused columns
        for c_idx in range(len(chosen), m):
            axes[row, 1 + c_idx].axis('off')

    fig.suptitle(f'seed={seed}', fontsize=9, color='grey', y=1.0)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {save_path} (seed={seed})")


def plot_layer_si_ev(all_results, save_dir):
    """Horizontal boxplots of SI and EV across layers (like layer_comparison.svg).

    Produces two figures: layer_si.svg and layer_ev.svg, each with one box per
    layer ordered early → late.
    """
    # Collect per-layer arrays
    layer_names = [r['layer_name'] for r in all_results]
    ordered = get_sorted_layers(layer_names)
    display_order = [ln for ln in ordered if ln in layer_names]
    n = len(display_order)
    if n == 0:
        return

    # Map layer_name → result
    res_map = {r['layer_name']: r for r in all_results}

    section_colors = {
        'enc/cnn': '#0055cc',
        'enc/mlp': '#0099ff',
        'enc/tok': '#44ccff',
        'dyn/sto': '#ff9900',
        'dyn/det': '#ff5500',
        'pol/mlp': '#33aa00',
        'val/mlp': '#996600',
    }

    def _color(ln):
        for prefix, c in section_colors.items():
            if ln.startswith(prefix):
                return c
        return '#888888'

    for metric_key, metric_label, fname in [
        ('SI', 'Spatial Information (bits/spike)', 'layer_si.svg'),
        ('EVs', 'Explained Variance', 'layer_ev.svg'),
    ]:
        fig, ax = plt.subplots(figsize=(8, max(4, n * 0.5)))

        data_raw = [res_map[ln]['metrics'][metric_key] for ln in display_order]
        # Filter out NaN/Inf for clean boxplots
        data = [d[np.isfinite(d)] for d in data_raw]
        n_neurons = [len(d) for d in data]
        labels = [
            ln.replace('/', '/\n') + f' ({nn})'
            for ln, nn in zip(display_order, n_neurons)
        ]

        bp = ax.boxplot(data, vert=False, patch_artist=True,
                        labels=labels, widths=0.6, showfliers=False)

        for patch, ln in zip(bp['boxes'], display_order):
            patch.set_facecolor(_color(ln))
            patch.set_alpha(0.7)

        # Annotate means
        for i, (ln, d) in enumerate(zip(display_order, data), start=1):
            mean_v = np.nanmean(d)
            ax.text(mean_v, i, f' {mean_v:.3f}', va='center', fontsize=7,
                    color='black')

        ax.set_xlabel(metric_label, fontsize=11)
        ax.set_title(f'Per-Layer {metric_label}\n'
                     f'orange line = median, black number = mean',
                     fontsize=12)
        ax.grid(True, axis='x', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='y', labelsize=8)

        fig.tight_layout()
        out = save_dir / fname
        fig.savefig(out, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {out}")


def plot_layer_si_ev_filtered(all_results, save_dir, ev_thresh=0.4):
    """Like plot_layer_si_ev but only for neurons with EV > ev_thresh."""
    layer_names = [r['layer_name'] for r in all_results]
    ordered = get_sorted_layers(layer_names)
    display_order = [ln for ln in ordered if ln in layer_names]
    n = len(display_order)
    if n == 0:
        return

    res_map = {r['layer_name']: r for r in all_results}

    section_colors = {
        'enc/cnn': '#0055cc',
        'enc/mlp': '#0099ff',
        'enc/tok': '#44ccff',
        'dyn/sto': '#ff9900',
        'dyn/det': '#ff5500',
        'pol/mlp': '#33aa00',
        'val/mlp': '#996600',
    }

    def _color(ln):
        for prefix, c in section_colors.items():
            if ln.startswith(prefix):
                return c
        return '#888888'

    for metric_key, metric_label, fname in [
        ('SI', 'Spatial Information (bits/spike)',
         f'layer_si_ev_gt{ev_thresh}.svg'),
        ('EVs', 'Explained Variance',
         f'layer_ev_ev_gt{ev_thresh}.svg'),
    ]:
        fig, ax = plt.subplots(figsize=(8, max(4, n * 0.5)))

        data = []
        n_neurons = []
        n_total = []
        for ln in display_order:
            m = res_map[ln]['metrics']
            evs = m['EVs']
            vals = m[metric_key]
            mask = np.isfinite(evs) & (evs > ev_thresh) & np.isfinite(vals)
            data.append(vals[mask])
            n_neurons.append(int(mask.sum()))
            n_total.append(len(vals))

        labels = [
            ln.replace('/', '/\n') + f' ({nn}/{nt})'
            for ln, nn, nt in zip(display_order, n_neurons, n_total)
        ]

        bp = ax.boxplot(data, vert=False, patch_artist=True,
                        labels=labels, widths=0.6, showfliers=False)

        for patch, ln in zip(bp['boxes'], display_order):
            patch.set_facecolor(_color(ln))
            patch.set_alpha(0.7)

        for i, (ln, d) in enumerate(zip(display_order, data), start=1):
            if len(d) > 0:
                mean_v = np.nanmean(d)
                ax.text(mean_v, i, f' {mean_v:.3f}', va='center', fontsize=7,
                        color='black')

        ax.set_xlabel(metric_label, fontsize=11)
        ax.set_title(f'Per-Layer {metric_label} (EV > {ev_thresh})\n'
                     f'orange line = median, black number = mean',
                     fontsize=12)
        ax.grid(True, axis='x', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='y', labelsize=8)

        fig.tight_layout()
        out = save_dir / fname
        fig.savefig(out, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {out}")


def plot_layer_summary(all_results, save_path):
    """Cross-layer cell type fraction bar chart."""
    layer_names = [r['layer_name'] for r in all_results]
    n_groups = len(GROUP_NAMES)
    fractions = np.zeros((len(layer_names), n_groups))
    for i, res in enumerate(all_results):
        gids = res['group_ids']
        if len(gids) == 0:
            continue
        for g in range(n_groups):
            fractions[i, g] = np.mean(gids == g)

    fig, ax = plt.subplots(figsize=(max(8, len(layer_names) * 0.6), 5))
    cmap = plt.get_cmap('viridis', n_groups)
    x = np.arange(len(layer_names))
    bottom = np.zeros(len(layer_names))
    for g in range(n_groups):
        ax.bar(x, fractions[:, g], bottom=bottom, label=GROUP_NAMES[g],
               color=cmap(g), width=0.7)
        bottom += fractions[:, g]

    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Fraction')
    ax.set_title('Cell Type Distribution Across Layers')
    ax.legend(fontsize=7, loc='upper right', ncol=2)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_sorted_layers(layers):
    """Sort layer names by LAYER_ORDER, unknowns appended."""
    ordered = [ln for ln in LAYER_ORDER if ln in layers]
    ordered += sorted(k for k in layers if k not in LAYER_ORDER)
    return ordered


def extract_facing(episodes):
    """Extract facing direction from episodes, concatenated and aligned."""
    all_facing = []
    for ep in episodes:
        if 'player_facing' in ep and len(ep['player_facing']) > 0:
            f = np.array(ep['player_facing'], dtype=np.float32)
            all_facing.append(f)
        else:
            return None  # If any episode lacks facing, skip HD analysis
    if not all_facing:
        return None
    return np.concatenate(all_facing)


def extract_facing_aligned(episodes, groups):
    """Extract facing, aligned with the layer data (same episode filtering)."""
    all_facing = []
    unique_eps = np.unique(groups)
    valid_eps = []
    for i, ep in enumerate(episodes):
        if 'player_pos' not in ep:
            continue
        valid_eps.append(i)

    for ep_idx in valid_eps:
        ep = episodes[ep_idx]
        if 'player_facing' not in ep or len(ep['player_facing']) == 0:
            return None
        p = np.array(ep['player_pos'], dtype=np.float32)
        f = np.array(ep['player_facing'], dtype=np.float32)
        T = min(len(p), len(f))
        # Also check act/* key lengths to match prepare_data_layers truncation
        act_keys = [k for k in ep if k.startswith('act/')]
        for k in act_keys:
            arr = ep.get(k)
            if arr is not None and len(arr) > 0:
                T = min(T, len(arr))
        all_facing.append(f[:T])

    if not all_facing:
        return None
    return np.concatenate(all_facing)


def main():
    parser = argparse.ArgumentParser(
        description='Spatial tuning curve analysis for DreamerV3 representations')
    parser.add_argument('--data', default=None,
                        help='Path to trajectory data directory')
    parser.add_argument('--test_data', default=None,
                        help='Path to held-out trajectory data for EV reliability')
    parser.add_argument('--save', default=None,
                        help='Output directory for results')
    parser.add_argument('--layers', nargs='*', default=None,
                        help='Specific layers to analyze (default: all)')
    parser.add_argument('--max_neurons', type=int, default=0,
                        help='Max neurons per layer (0=all)')
    parser.add_argument('--no_hd', action='store_true',
                        help='Skip head direction analysis')
    parser.add_argument('--no_plots', action='store_true',
                        help='Skip plot generation')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='Parallel workers (-1=all CPUs)')
    # Classification thresholds
    parser.add_argument('--SI_thresh', type=float, default=0.5)
    parser.add_argument('--EV_thresh', type=float, default=0.5)
    parser.add_argument('--EV_unthresh', type=float, default=0.15)
    parser.add_argument('--HD_thresh', type=float, default=0.5)
    parser.add_argument('--smooth_sigma', type=float, default=0,
                        help='Gaussian smoothing sigma (bins) on tuning curves before SI (0=off)')
    parser.add_argument('--ev_filter', type=float, default=0.4,
                        help='EV cutoff for filtered SI/EV boxplots (default 0.4)')
    parser.add_argument('--min_bbox', type=float, default=0,
                        help='Min bounding-box area (tiles²) to keep an episode (0=no filter)')
    parser.add_argument('--interactive', action='store_true',
                        help='Show interactive SI vs EV scatter (click to see tuning curves)')
    parser.add_argument('--from_pkl', default=None,
                        help='Load precomputed tuning_results.pkl and show interactive viewer (no recomputation)')
    args = parser.parse_args()

    # Interactive viewer from precomputed pkl — no data/save needed
    if args.from_pkl:
        with open(args.from_pkl, 'rb') as f:
            results_dict = pickle.load(f)
        layers = results_dict['layers']
        layer_names = list(layers.keys())
        if len(layer_names) == 1:
            ln = layer_names[0]
        else:
            print("Available layers:")
            for i, ln in enumerate(layer_names):
                n = len(layers[ln]['metrics']['SI'])
                print(f"  [{i}] {ln} ({n} neurons)")
            choice = input("Select layer number (or 'all' for sequential): ").strip()
            if choice == 'all':
                for ln in layer_names:
                    ld = layers[ln]
                    _interactive_si_ev(ld['metrics'], ln, ld['tuning_curves'])
                print("Done.")
                return
            ln = layer_names[int(choice)]
        ld = layers[ln]
        _interactive_si_ev(ld['metrics'], ln, ld['tuning_curves'])
        return

    if not args.data or not args.save:
        parser.error("--data and --save are required unless using --from_pkl")

    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load episodes
    print(f"Loading episodes from {args.data}")
    episodes, metadata = load_episodes(args.data)
    print(f"  {len(episodes)} episodes loaded")

    if args.min_bbox > 0:
        n_before = len(episodes)
        episodes = filter_stuck_episodes(episodes, args.min_bbox)
        print(f"  {n_before} → {len(episodes)} episodes after bbox filter "
              f"(min_bbox={args.min_bbox})")

    area = metadata.get('area', (64, 64)) if metadata else (64, 64)
    if isinstance(area, (list, tuple)):
        area = tuple(area)
    print(f"  Area: {area}")

    # Drop image data to free memory (not needed for tuning analysis)
    _strip_heavy_keys(episodes)

    # Load test data if provided
    test_episodes = None
    if args.test_data:
        print(f"Loading test episodes from {args.test_data}")
        test_episodes, _ = load_episodes(args.test_data)
        print(f"  {len(test_episodes)} test episodes loaded")
        if args.min_bbox > 0:
            n_before = len(test_episodes)
            test_episodes = filter_stuck_episodes(test_episodes, args.min_bbox)
            print(f"  {n_before} → {len(test_episodes)} test episodes after bbox filter")
        _strip_heavy_keys(test_episodes)

    # Discover layer names and determine order
    layer_names = _discover_layer_names(episodes)
    if not layer_names:
        # Fallback: deter/stoch only
        print("  No act/* layers found, falling back to deter/stoch only")
        layer_names = ['dyn/deter', 'dyn/stoch']
    ordered = get_sorted_layers(layer_names)
    if args.layers:
        ordered = [ln for ln in ordered if ln in args.layers]
    print(f"  Layers to analyze: {ordered}")

    # Extract facing direction (once, small — just 2 floats per timestep)
    # Use a dummy single-layer load to get aligned groups for facing extraction
    _, _tmp_pos, _tmp_groups = _prepare_single_layer(episodes, ordered[0])
    facing = extract_facing_aligned(episodes, _tmp_groups) if not args.no_hd else None
    del _tmp_pos, _tmp_groups

    test_facing = None
    if test_episodes and not args.no_hd:
        _, _tp, _tg = _prepare_single_layer(test_episodes, ordered[0])
        test_facing = extract_facing_aligned(test_episodes, _tg)
        del _tp, _tg

    # Analyze each layer sequentially — load one at a time to keep memory low
    all_results = []

    for ln in ordered:
        print(f"\n--- Loading layer: {ln} ---")
        X, pos, groups = _prepare_single_layer(episodes, ln)
        if args.max_neurons > 0 and X.shape[1] > args.max_neurons:
            X = X[:, :args.max_neurons]
        mem_mb = X.nbytes / 1e6
        print(f"  {ln}: {X.shape} ({mem_mb:.0f} MB)")

        test_X, test_pos, test_groups = None, None, None
        if test_episodes is not None:
            test_X, test_pos, test_groups = _prepare_single_layer(test_episodes, ln)
            if args.max_neurons > 0 and test_X.shape[1] > args.max_neurons:
                test_X = test_X[:, :args.max_neurons]

        result = analyze_layer(
            ln, X, pos, groups, area,
            facing=facing,
            test_activations=test_X,
            test_positions=test_pos,
            test_groups=test_groups,
            compute_hd=not args.no_hd and facing is not None,
            smooth_sigma=args.smooth_sigma,
        )
        all_results.append(result)

        # Free the large activation array before next layer
        del X, pos, groups, test_X, test_pos, test_groups
        gc.collect()

    # Reclassify with custom thresholds if provided
    for res in all_results:
        res['cell_groups'], res['group_ids'] = classify_cells(
            res['metrics'],
            SI_thresh=args.SI_thresh,
            EV_thresh=args.EV_thresh,
            EV_unthresh=args.EV_unthresh,
            HD_thresh=args.HD_thresh,
        )

    # Save results
    results_dict = {
        'metadata': {
            'data_path': str(args.data),
            'test_data_path': str(args.test_data) if args.test_data else None,
            'area': area,
            'n_episodes': len(episodes),
            'n_test_episodes': len(test_episodes) if test_episodes else 0,
            'group_names': GROUP_NAMES,
            'thresholds': {
                'SI_thresh': args.SI_thresh,
                'EV_thresh': args.EV_thresh,
                'EV_unthresh': args.EV_unthresh,
                'HD_thresh': args.HD_thresh,
            },
        },
        'layers': {},
    }
    for res in all_results:
        ln = res['layer_name']
        results_dict['layers'][ln] = {
            'tuning_curves': res['tuning_curves'],
            'metrics': res['metrics'],
            'cell_groups': {k: v for k, v in res['cell_groups'].items()},
            'group_ids': res['group_ids'],
        }

    results_path = save_dir / 'tuning_results.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump(results_dict, f)
    print(f"\nResults saved to {results_path}")

    # Print summary
    print("\n=== Summary ===")
    for res in all_results:
        ln = res['layer_name']
        gids = res['group_ids']
        n = len(gids)
        print(f"\n  {ln} ({n} neurons):")
        for g, name in enumerate(GROUP_NAMES):
            count = np.sum(gids == g)
            pct = 100 * count / n if n > 0 else 0
            print(f"    {name:20s}: {count:5d} ({pct:5.1f}%)")

    # Plots
    if not args.no_plots:
        print("\nGenerating plots...")
        for res in all_results:
            ln = res['layer_name']
            safe_name = ln.replace('/', '_')
            layer_dir = save_dir / safe_name
            layer_dir.mkdir(parents=True, exist_ok=True)

            plot_si_ev_scatter(
                res['metrics'], res['group_ids'], ln,
                layer_dir / 'si_ev_scatter.svg',
                tc_array=res['tuning_curves'],
                interactive=args.interactive,
            )
            plot_cell_types(
                res['group_ids'], ln,
                layer_dir / 'cell_types.svg',
            )
            plot_example_tuning_curves(
                res['tuning_curves'], res['metrics'], res['group_ids'], ln,
                layer_dir / 'example_tuning_curves.svg',
                area=area, sort_by='SI',
            )
            plot_example_tuning_curves(
                res['tuning_curves'], res['metrics'], res['group_ids'], ln,
                layer_dir / 'example_tuning_curves_ev.svg',
                area=area, sort_by='EV',
            )

        if len(all_results) > 1:
            plot_layer_summary(all_results, save_dir / 'layer_summary.svg')
            plot_layer_si_ev(all_results, save_dir)
            plot_layer_si_ev_filtered(all_results, save_dir, ev_thresh=args.ev_filter)
            plot_layer_si_ev_grid(all_results, save_dir / 'layer_si_ev_grid.svg')

        print(f"Plots saved to {save_dir}")

    print("\nDone.")

    log_run_info(save_dir, 'tuning_curve', vars(args))


if __name__ == '__main__':
    main()
