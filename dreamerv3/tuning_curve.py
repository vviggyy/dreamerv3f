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


def _strip_activation_keys(episodes):
    """Drop all act/* keys from episodes in-place to free memory.

    This allows layer-by-layer reloading from individual episode files,
    keeping only lightweight keys (player_pos, reward, action, etc.) in RAM.
    """
    for ep in episodes:
        act_keys = [k for k in ep if k.startswith('act/')]
        for k in act_keys:
            del ep[k]
        # Also drop deter/stoch top-level keys (duplicated in act/dyn/*)
        for k in ['deter', 'stoch']:
            if k in ep:
                del ep[k]


def _reload_layer_from_files(episode_indices, layer_name, return_facing=False):
    """Reload a single layer's activations from individual episode files.

    Args:
        episode_indices: List of (episode_index, episode_file) tuples to load.
        layer_name: Layer name (e.g. 'enc/cnn0') — reads act/{layer_name}.
        return_facing: If True, also return aligned facing directions.

    Returns:
        X, pos, groups as concatenated arrays.
        If return_facing=True: X, pos, groups, facing (or None if no facing data).
    """
    act_key = f'act/{layer_name}'
    all_x, all_pos, all_groups, all_facing = [], [], [], []
    for i, ep_file in episode_indices:
        with open(ep_file, 'rb') as f:
            ep = pickle.load(f)
        arr = ep.get(act_key)
        p = ep.get('player_pos')
        if arr is None or p is None or len(arr) == 0:
            continue
        p = np.array(p, dtype=np.float32)
        a = np.array(arr, dtype=np.float32)
        T = min(len(p), len(a))
        if a.ndim > 2:
            a = a[:T].reshape(T, -1)
        else:
            a = a[:T]
        all_pos.append(p[:T])
        all_x.append(a)
        all_groups.append(np.full(T, i))
        if return_facing:
            f_arr = ep.get('player_facing')
            if f_arr is not None and len(f_arr) > 0:
                all_facing.append(np.array(f_arr, dtype=np.float32)[:T])
        del ep  # free immediately
    if not all_pos:
        raise ValueError(f"No valid episodes for layer {layer_name}")
    result = (np.concatenate(all_x), np.concatenate(all_pos), np.concatenate(all_groups))
    if return_facing:
        facing = np.concatenate(all_facing) if all_facing else None
        return result + (facing,)
    return result

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
# Global spatial autocorrelation metrics (Moran's I, Geary's C, Getis-Ord G)
# ---------------------------------------------------------------------------

def _inverse_distance_weights(valid_mask, dist_cutoff=0):
    """Build inverse-distance weight matrix for valid bins on a 2D grid.

    Args:
        valid_mask: (H, W) boolean array, True for non-NaN bins.
        dist_cutoff: Max distance in tiles. Pairs beyond this get weight 0.
            0 means no cutoff (all pairs weighted).
    Returns:
        W: (N, N) weight matrix, N = number of valid bins.
        valid_indices: (N, 2) array of (row, col) for each valid bin.
    """
    valid_indices = np.argwhere(valid_mask)  # (N, 2)
    N = len(valid_indices)
    if N < 2:
        return np.zeros((N, N)), valid_indices
    # Pairwise Euclidean distances
    diff = valid_indices[:, None, :] - valid_indices[None, :, :]  # (N, N, 2)
    dist = np.sqrt((diff ** 2).sum(axis=2))  # (N, N)
    # Inverse distance, diagonal = 0
    with np.errstate(divide='ignore'):
        W = np.where(dist > 0, 1.0 / dist, 0.0)
    # Apply distance cutoff
    if dist_cutoff > 0:
        W[dist > dist_cutoff] = 0.0
    return W, valid_indices


def global_morans_i(tc, dist_cutoff=0):
    """Global Moran's I for a single 2D tuning curve. NaN bins excluded.

    I = (N / W_sum) * (z^T @ W @ z) / (z^T @ z)
    where z = x - mean(x).

    Returns NaN if fewer than 3 valid bins or zero variance.
    """
    valid_mask = ~np.isnan(tc)
    N = valid_mask.sum()
    if N < 3:
        return np.nan
    vals = tc[valid_mask]
    z = vals - vals.mean()
    ss = (z ** 2).sum()
    if ss == 0:
        return np.nan
    W, _ = _inverse_distance_weights(valid_mask, dist_cutoff=dist_cutoff)
    W_sum = W.sum()
    if W_sum == 0:
        return np.nan
    I = (N / W_sum) * (z @ W @ z) / ss
    return float(I)


def gearys_c(tc, dist_cutoff=0):
    """Geary's C for a single 2D tuning curve. NaN bins excluded.

    C = ((N-1) / (2 * W_sum)) * sum_ij(w_ij * (x_i - x_j)^2) / sum(z_i^2)

    Returns NaN if fewer than 3 valid bins or zero variance.
    """
    valid_mask = ~np.isnan(tc)
    N = valid_mask.sum()
    if N < 3:
        return np.nan
    vals = tc[valid_mask]
    z = vals - vals.mean()
    ss = (z ** 2).sum()
    if ss == 0:
        return np.nan
    W, _ = _inverse_distance_weights(valid_mask, dist_cutoff=dist_cutoff)
    W_sum = W.sum()
    if W_sum == 0:
        return np.nan
    diff = vals[:, None] - vals[None, :]  # (N, N)
    numer = (W * diff ** 2).sum()
    C = ((N - 1) / (2 * W_sum)) * numer / ss
    return float(C)


def getis_ord_g(tc, dist_cutoff=0):
    """Getis-Ord General G for a single 2D tuning curve. NaN bins excluded.

    G = sum_ij(w_ij * x_i * x_j) / sum_ij(x_i * x_j), i != j

    Returns NaN if fewer than 3 valid bins or zero denominator.
    """
    valid_mask = ~np.isnan(tc)
    N = valid_mask.sum()
    if N < 3:
        return np.nan
    vals = tc[valid_mask]
    W, _ = _inverse_distance_weights(valid_mask, dist_cutoff=dist_cutoff)
    cross = vals[:, None] * vals[None, :]  # (N, N)
    # Exclude diagonal
    np.fill_diagonal(cross, 0)
    denom = cross.sum()
    if denom == 0:
        return np.nan
    G = (W * cross).sum() / denom
    return float(G)


def _compute_spatial_autocorr_metrics(tuning_curves, dist_cutoff=0):
    """Compute Moran's I, Geary's C, Getis-Ord G for an array of tuning curves.

    Args:
        tuning_curves: (N_neurons, H, W) array.
        dist_cutoff: Max distance in tiles for weight matrix (0=no cutoff).
    Returns:
        morans, gearys, getis: each (N_neurons,) arrays.
    """
    morans = np.array([global_morans_i(tc, dist_cutoff) for tc in tuning_curves])
    gearys = np.array([gearys_c(tc, dist_cutoff) for tc in tuning_curves])
    getis = np.array([getis_ord_g(tc, dist_cutoff) for tc in tuning_curves])
    return morans, gearys, getis


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
                  test_groups=None, compute_hd=True, smooth_sigma=0,
                  dist_cutoff=0):
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
        dist_cutoff: Max distance (tiles) for spatial autocorr weights (0=no cutoff).

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

    # Global spatial autocorrelation metrics
    morans, gearys, getis = _compute_spatial_autocorr_metrics(tc_array, dist_cutoff=dist_cutoff)

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
        'morans_i': morans,
        'gearys_c': gearys,
        'getis_ord_g': getis,
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
                               sort_by='SI', ev_filter=0.0):
    """Grid of example tuning curves sorted by a metric (SI, EV, morans_i, etc.).

    Args:
        ev_filter: For autocorr metrics (morans_i, gearys_c, getis_ord_g),
            only show neurons with EV > ev_filter. Set 0 to disable.
    """
    n_neurons = tc_array.shape[0]
    # Map sort_by name to metrics dict key
    _sort_key_map = {'EV': 'EVs', 'SI': 'SI',
                     'morans_i': 'morans_i', 'gearys_c': 'gearys_c',
                     'getis_ord_g': 'getis_ord_g', 'fieldsize': 'fieldsize',
                     'pf_peaks': 'pf_peaks'}
    metric_key = _sort_key_map.get(sort_by, sort_by)
    if metric_key not in metrics:
        print(f"  Warning: metric '{metric_key}' not found, falling back to SI")
        metric_key = 'SI'
        sort_by = 'SI'
    vals = metrics[metric_key]
    # Build candidate mask: exclude NaN/Inf values
    candidate_mask = np.isfinite(vals)
    # For autocorr metrics, also require EV above threshold
    _autocorr_metrics = {'morans_i', 'gearys_c', 'getis_ord_g', 'fieldsize', 'pf_peaks'}
    if sort_by in _autocorr_metrics and ev_filter > 0 and 'EVs' in metrics:
        candidate_mask &= np.isfinite(metrics['EVs']) & (metrics['EVs'] > ev_filter)
    candidates = np.where(candidate_mask)[0]
    if len(candidates) == 0:
        print(f"  Warning: no valid neurons for {sort_by} (ev_filter={ev_filter}), skipping")
        return
    n_show = min(n_examples, len(candidates))
    # For Geary's C, lower = more spatial structure, so sort ascending
    if sort_by == 'gearys_c':
        order = candidates[np.argsort(vals[candidates])]  # ascending — lowest C = most clustered
    else:
        order = candidates[np.argsort(vals[candidates])[::-1]]  # descending — highest = best
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
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_tuning_with_autocorr(tc_array, metrics, layer_name, save_path,
                              n_examples=10, sort_by='EV'):
    """Top tuning curves (row 1) with their spatial autocorrelations (row 2).

    Shows the top-N neurons by EV (or SI) with their 2D autocorrelation below,
    annotated with the number of peaks detected.

    Args:
        tc_array: (N_neurons, H, W) tuning curves.
        metrics: dict with 'SI', 'EVs', 'pf_peaks'.
        layer_name: name for title.
        save_path: output file path.
        n_examples: how many neurons to show.
        sort_by: 'EV' or 'SI'.
    """
    n_neurons = tc_array.shape[0]
    metric_key = 'EVs' if sort_by == 'EV' else 'SI'
    vals = metrics[metric_key]
    # Filter out NaN/Inf before sorting
    valid = np.where(np.isfinite(vals))[0]
    if len(valid) == 0:
        print(f"  Warning: no finite values for {sort_by} in {layer_name}, skipping")
        return
    n_show = min(n_examples, len(valid))
    order = valid[np.argsort(vals[valid])[::-1]]
    selected = order[:n_show]

    fig, axes = plt.subplots(2, n_show, figsize=(2 * n_show, 4.5),
                              squeeze=False)

    for col, neuron_idx in enumerate(selected):
        tc = tc_array[neuron_idx]
        ev_val = metrics['EVs'][neuron_idx]
        si_val = metrics['SI'][neuron_idx]
        peaks = int(metrics['pf_peaks'][neuron_idx]) if np.isfinite(
            metrics['pf_peaks'][neuron_idx]) else 0

        # Row 0: tuning curve
        ax_tc = axes[0, col]
        ax_tc.imshow(np.ma.masked_invalid(tc.T), origin='lower',
                     interpolation='nearest', cmap=_tc_cmap())
        ax_tc.set_title(f'n{neuron_idx}\nEV={ev_val:.2f} SI={si_val:.2f}',
                        fontsize=6)
        ax_tc.set_xticks([])
        ax_tc.set_yticks([])

        # Row 1: autocorrelation
        ax_ac = axes[1, col]
        ac = correlate2d(np.nan_to_num(tc), np.nan_to_num(tc), mode='full')
        if ac.max() > 0:
            ac = ac / ac.max()
        ax_ac.imshow(ac.T, origin='lower', interpolation='nearest',
                     cmap='RdBu_r', vmin=-0.2, vmax=1.0)
        ax_ac.set_title(f'peaks={peaks}', fontsize=6)
        ax_ac.set_xticks([])
        ax_ac.set_yticks([])

    axes[0, 0].set_ylabel('Tuning Curve', fontsize=8)
    axes[1, 0].set_ylabel('Autocorrelation', fontsize=8)

    fig.suptitle(f'{layer_name}: Top {n_show} by {sort_by} + Spatial Autocorrelation',
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.subplots_adjust(hspace=0.4)
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved {save_path}")


def plot_layer_si_ev_grid(all_results, save_path, m=5, seed=None, ev_thresh=None):
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
        # Sample from neurons above ev_thresh, or top 10th percentile if not set
        if ev_thresh is not None:
            ev_cutoff = ev_thresh
        else:
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
    parser.add_argument('--EV_thresh', type=float, default=0.5) #used for cell-type classificatoin Upper bound
    parser.add_argument('--EV_unthresh', type=float, default=0.15) #also used for cell type class. lower poind
    parser.add_argument('--HD_thresh', type=float, default=0.5)
    parser.add_argument('--smooth_sigma', type=float, default=0,
                        help='Gaussian smoothing sigma (bins) on tuning curves before SI (0=off)')
    parser.add_argument('--ev_filter', type=float, default=0.4,
                        help='EV cutoff for filtered SI/EV boxplots (default 0.4)') #used for "exmple tuning curves" 
    parser.add_argument('--dist_cutoff', type=float, default=7,
                        help='Max distance (tiles) for spatial autocorr weight matrix. '
                             'Pairs beyond this get weight 0. 0=no cutoff (default 7)')
    parser.add_argument('--min_bbox', type=float, default=0,
                        help='Min bounding-box area (tiles²) to keep an episode (0=no filter)')
    parser.add_argument('--interactive', action='store_true',
                        help='Show interactive SI vs EV scatter (click to see tuning curves)')
    parser.add_argument('--max_episodes', type=int, default=0,
                        help='Max episodes to use (0=all)')
    parser.add_argument('--area', type=int, nargs=2, default=None,
                        help='World area as W H (e.g. --area 32 32). '
                             'Overrides metadata. If not set, read from '
                             'metadata or inferred from positions.')
    parser.add_argument('--from_pkl', default=None,
                        help='Load precomputed tuning_results.pkl and show interactive viewer (no recomputation)')
    parser.add_argument('--plot_autocorr', action='store_true',
                        help='With --from_pkl: generate tuning+autocorr plots instead of interactive viewer')
    args = parser.parse_args()

    # Interactive viewer or autocorr plots from precomputed pkl
    if args.from_pkl:
        with open(args.from_pkl, 'rb') as f:
            results_dict = pickle.load(f)
        layers = results_dict['layers']
        layer_names = list(layers.keys())

        # Recompute spatial autocorrelation metrics (always, since dist_cutoff may differ)
        dc = getattr(args, 'dist_cutoff', 0)
        for ln in layer_names:
            ld = layers[ln]
            print(f"  Computing spatial autocorr metrics for {ln} (dist_cutoff={dc})...")
            m, g, go = _compute_spatial_autocorr_metrics(ld['tuning_curves'], dist_cutoff=dc)
            ld['metrics']['morans_i'] = m
            ld['metrics']['gearys_c'] = g
            ld['metrics']['getis_ord_g'] = go

        # --plot_autocorr: batch generate tuning+autocorr plots
        if args.plot_autocorr:
            pkl_path = Path(args.from_pkl)
            out_dir = Path(args.save) if args.save else pkl_path.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            ordered = get_sorted_layers(layer_names)
            if args.layers:
                ordered = [ln for ln in ordered if ln in args.layers]
            for ln in ordered:
                ld = layers[ln]
                safe_name = ln.replace('/', '_')
                layer_dir = out_dir / safe_name
                layer_dir.mkdir(parents=True, exist_ok=True)
                plot_tuning_with_autocorr(
                    ld['tuning_curves'], ld['metrics'], ln,
                    layer_dir / 'tuning_with_autocorr.pdf',
                    n_examples=10, sort_by='EV',
                )
                for metric_name in ('morans_i', 'gearys_c', 'getis_ord_g', 'fieldsize', 'pf_peaks'):
                    plot_example_tuning_curves(
                        ld['tuning_curves'], ld['metrics'],
                        ld.get('group_ids', np.zeros(len(ld['metrics']['SI']), dtype=int)),
                        ln,
                        layer_dir / f'example_tuning_curves_{metric_name}.pdf',
                        sort_by=metric_name,
                        ev_filter=args.ev_filter,
                    )
            print("Done.")
            return

        # Batch reclassify + replot when --save is provided (non-interactive)
        if args.save and not args.interactive:
            save_dir = Path(args.save)
            save_dir.mkdir(parents=True, exist_ok=True)

            # Rebuild all_results list from pkl layers
            all_results = []
            ordered = get_sorted_layers(layer_names)
            if args.layers:
                ordered = [ln for ln in ordered if ln in args.layers]
            for ln in ordered:
                ld = layers[ln]
                cell_groups, group_ids = classify_cells(
                    ld['metrics'],
                    SI_thresh=args.SI_thresh,
                    EV_thresh=args.EV_thresh,
                    EV_unthresh=args.EV_unthresh,
                    HD_thresh=args.HD_thresh,
                )
                all_results.append({
                    'layer_name': ln,
                    'tuning_curves': ld['tuning_curves'],
                    'metrics': ld['metrics'],
                    'cell_groups': cell_groups,
                    'group_ids': group_ids,
                })

            # Infer area from tuning curve shape
            tc0 = all_results[0]['tuning_curves']
            area = list(tc0.shape[1:])  # (n_neurons, H, W) -> [H, W]

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
                    )
                    plot_cell_types(
                        res['group_ids'], ln,
                        layer_dir / 'cell_types.svg',
                    )
                    plot_example_tuning_curves(
                        res['tuning_curves'], res['metrics'], res['group_ids'], ln,
                        layer_dir / 'example_tuning_curves.pdf',
                        area=area, sort_by='SI',
                    )
                    plot_example_tuning_curves(
                        res['tuning_curves'], res['metrics'], res['group_ids'], ln,
                        layer_dir / 'example_tuning_curves_ev.pdf',
                        area=area, sort_by='EV',
                    )
                if len(all_results) > 1:
                    plot_layer_summary(all_results, save_dir / 'layer_summary.svg')
                    plot_layer_si_ev(all_results, save_dir)
                    plot_layer_si_ev_filtered(all_results, save_dir, ev_thresh=args.ev_filter)
                    plot_layer_si_ev_grid(all_results, save_dir / 'layer_si_ev_grid.svg', ev_thresh=args.EV_thresh)
                print(f"Plots saved to {save_dir}")
            print("Done.")
            return

        # Interactive viewer
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

    # Load episodes — lightweight approach to avoid OOM.
    # 1) Identify episode files and discover layers from first file only
    # 2) Load lightweight data (pos, facing) from all episodes
    # 3) Per-layer analysis reloads activations one layer at a time from files
    data_path = Path(args.data)
    ep_files = sorted(data_path.glob('episode_*.pkl'))
    if args.max_episodes > 0:
        ep_files = ep_files[:args.max_episodes]
    if not ep_files:
        raise ValueError(f"No episode_*.pkl files found in {data_path}")
    print(f"Found {len(ep_files)} episode files in {data_path}")

    # Discover layer names from first episode only
    with open(ep_files[0], 'rb') as f:
        first_ep = pickle.load(f)
    layer_names = [k[len('act/'):] for k in first_ep if k.startswith('act/')]
    del first_ep
    if not layer_names:
        print("  No act/* layers found, falling back to deter/stoch only")
        layer_names = ['dyn/deter', 'dyn/stoch']
    ordered = get_sorted_layers(layer_names)
    if args.layers:
        ordered = [ln for ln in ordered if ln in args.layers]
    print(f"  Layers to analyze: {ordered}")

    # Load lightweight data only (pos, facing, achievements — no activations)
    print(f"Loading lightweight episode data...")
    lightweight_eps = []
    for ep_file in ep_files:
        with open(ep_file, 'rb') as f:
            ep = pickle.load(f)
        light = {k: ep[k] for k in ['player_pos', 'player_facing', 'reward',
                                      'action', 'achievements']
                 if k in ep}
        light['_source_file'] = ep_file
        lightweight_eps.append(light)
        del ep
    gc.collect()
    print(f"  {len(lightweight_eps)} episodes loaded (lightweight)")

    if args.min_bbox > 0:
        n_before = len(lightweight_eps)
        lightweight_eps = filter_stuck_episodes(lightweight_eps, args.min_bbox)
        print(f"  {n_before} → {len(lightweight_eps)} episodes after bbox filter "
              f"(min_bbox={args.min_bbox})")

    if args.area:
        area = tuple(args.area)
    else:
        all_pos = np.concatenate([ep['player_pos'] for ep in lightweight_eps])
        area = (int(all_pos[:, 0].max()) + 1, int(all_pos[:, 1].max()) + 1)
        print(f"  (area inferred from position data)")
    print(f"  Area: {area}")

    # Build episode file index from surviving (post-filter) episodes
    ep_file_index = [(i, ep['_source_file']) for i, ep in enumerate(lightweight_eps)]

    # Load test data if provided — same lightweight approach
    test_ep_file_index = None
    if args.test_data:
        test_data_path = Path(args.test_data)
        test_ep_files = sorted(test_data_path.glob('episode_*.pkl'))
        print(f"Found {len(test_ep_files)} test episode files in {test_data_path}")
        test_light = []
        for ep_file in test_ep_files:
            with open(ep_file, 'rb') as f:
                ep = pickle.load(f)
            light = {k: ep[k] for k in ['player_pos', 'player_facing', 'reward',
                                          'action', 'achievements']
                     if k in ep}
            light['_source_file'] = ep_file
            test_light.append(light)
            del ep
        gc.collect()
        if args.min_bbox > 0:
            n_before = len(test_light)
            test_light = filter_stuck_episodes(test_light, args.min_bbox)
            print(f"  {n_before} → {len(test_light)} test episodes after bbox filter")
        test_ep_file_index = [(i, ep['_source_file']) for i, ep in enumerate(test_light)]
        del test_light
        gc.collect()

    # Analyze each layer — reload activations from individual episode files
    all_results = []
    facing = None  # extracted from first layer load
    test_facing = None

    for li, ln in enumerate(ordered):
        print(f"\n--- Loading layer: {ln} ---")
        need_facing = (li == 0 and not args.no_hd)
        if need_facing:
            X, pos, groups, facing = _reload_layer_from_files(
                ep_file_index, ln, return_facing=True)
        else:
            X, pos, groups = _reload_layer_from_files(ep_file_index, ln)
        if args.max_neurons > 0 and X.shape[1] > args.max_neurons:
            X = X[:, :args.max_neurons]
        mem_mb = X.nbytes / 1e6
        print(f"  {ln}: {X.shape} ({mem_mb:.0f} MB)")

        test_X, test_pos, test_groups = None, None, None
        if test_ep_file_index is not None:
            if need_facing:
                test_X, test_pos, test_groups, test_facing = _reload_layer_from_files(
                    test_ep_file_index, ln, return_facing=True)
            else:
                test_X, test_pos, test_groups = _reload_layer_from_files(
                    test_ep_file_index, ln)
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
            dist_cutoff=args.dist_cutoff,
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
            'n_episodes': len(ep_file_index),
            'n_test_episodes': len(test_ep_file_index) if test_ep_file_index else 0,
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
                layer_dir / 'example_tuning_curves.pdf',
                area=area, sort_by='SI',
            )
            plot_example_tuning_curves(
                res['tuning_curves'], res['metrics'], res['group_ids'], ln,
                layer_dir / 'example_tuning_curves_ev.pdf',
                area=area, sort_by='EV',
            )
            plot_tuning_with_autocorr(
                res['tuning_curves'], res['metrics'], ln,
                layer_dir / 'tuning_with_autocorr.pdf',
                n_examples=10, sort_by='EV',
            )
            for metric_name in ('morans_i', 'gearys_c', 'getis_ord_g', 'fieldsize', 'pf_peaks'):
                plot_example_tuning_curves(
                    res['tuning_curves'], res['metrics'], res['group_ids'], ln,
                    layer_dir / f'example_tuning_curves_{metric_name}.pdf',
                    area=area, sort_by=metric_name,
                    ev_filter=args.ev_filter,
                )

        if len(all_results) > 1:
            plot_layer_summary(all_results, save_dir / 'layer_summary.svg')
            plot_layer_si_ev(all_results, save_dir)
            plot_layer_si_ev_filtered(all_results, save_dir, ev_thresh=args.ev_filter)
            plot_layer_si_ev_grid(all_results, save_dir / 'layer_si_ev_grid.svg', ev_thresh=args.EV_thresh)

        print(f"Plots saved to {save_dir}")

    print("\nDone.")

    log_run_info(save_dir, 'tuning_curve', vars(args))


if __name__ == '__main__':
    main()
