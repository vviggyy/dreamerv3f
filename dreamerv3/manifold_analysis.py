"""
Neural manifold analysis: sRSA, Isomap, and SW distance.

Compares wake (real trajectory) and dream (imagination) activations to assess
whether the world model's dreamed representations occupy the same neural
manifold as wake representations.

Metrics:
  - sRSA: Spearman rank correlation between pairwise position distances and
    pairwise neural cosine distances during wake — measures spatial structure.
  - Hill fit: Saturating Hill function fitted to binned neural-vs-spatial
    distance, extracting dh_0, dh_inf, dx_1/2 parameters.
  - Isomap: 2D manifold visualization of combined wake+dream activations.
  - SW Distance: Median minimum cosine distance from each dream point to its
    nearest wake neighbor — quantifies manifold proximity.

Usage:
  # Wake vs dream (from dream_decode output):
  MPLBACKEND=Agg python dreamerv3/manifold_analysis.py \\
    --data ./logdir/my_run/trajectories \\
    --dream_data ./logdir/my_run/dream_results/dream_results.pkl \\
    --save ./logdir/my_run/manifold_results

  # Wake vs wake control (second trajectory set):
  MPLBACKEND=Agg python dreamerv3/manifold_analysis.py \\
    --data ./logdir/my_run/trajectories_train \\
    --dream_data ./logdir/my_run/trajectories_test \\
    --save ./logdir/my_run/manifold_results

  # Filter layers:
  MPLBACKEND=Agg python dreamerv3/manifold_analysis.py \\
    --data ./logdir/my_run/trajectories \\
    --dream_data ./logdir/my_run/dream_results/dream_results.pkl \\
    --save ./logdir/my_run/manifold_results \\
    --layers dyn/deter dyn/stoch
"""

import argparse
import pickle
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.stats import spearmanr
from sklearn.manifold import Isomap

from run_info import log_run_info
from decode_position import load_episodes, _prepare_single_layer, filter_stuck_episodes

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

DREAM_LAYERS = ['dyn/deter', 'dyn/stoch']

# Seed-ablation conditions (mirrors dream_seed_ablation.py). D is lightened from
# the seed-ablation 'dimgray' to 'silver' so it reads on the dark manifold theme.
# A' (Aprime, perturbed-state) is an OPTIONAL 5th condition: it is included only
# when the seed-ablation dir holds a dream_deter_Aprime.pkl (a --perturb_state
# --save_activations run), auto-detected in run_seed_ablation_overlay. Kept OUT of
# the base list so ordinary A/B/C/D runs are unaffected.
SEED_ABLATION_CONDITIONS = ['A', 'B', 'C', 'D']
APRIME = 'Aprime'
ECOND = 'E'   # blank-slate: all-zero deter, no warmup, prior stoch, rolled out
# Canonical display order (A' and E drawn last so their curves compare on top).
SEED_ABLATION_ORDER = SEED_ABLATION_CONDITIONS + [APRIME, ECOND]
SEED_ABLATION_LABELS = {
    'A': 'A: both (real deter + image)',
    'B': 'B: just-latent (real deter + prior)',
    'C': 'C: just-image (noise deter + image)',
    'D': 'D: neither (noise deter + prior)',
    APRIME: "A': perturbed-state (real deter + vitals-perturbed image)",
    ECOND: 'E: blank-slate (zero deter + prior, no warmup)',
}
SEED_ABLATION_COLORS = {'A': 'crimson', 'B': 'steelblue',
                        'C': 'darkorange', 'D': 'silver',
                        APRIME: 'mediumseagreen', ECOND: 'mediumvioletred'}
# A and A' both seed from the REAL deter, so their SW-over-time curves can track
# closely; a dashed A' disambiguates them where color overlaps. A/B/C/D solid.
SEED_ABLATION_LINESTYLES = {'A': '-', 'B': '-', 'C': '-', 'D': '-',
                            APRIME: (0, (5, 1))}


def _assert_relu_space(X, layer_name, source):
    """Guard: wake/dream activations must be in relu (non-negative) space.

    deter/stoch/enc layers are all relu (or one-hot) and therefore non-negative.
    A signed array means the run was generated with a non-relu gru_act (e.g. the
    stock tanh default) — wake recorded with relu vs dream generated with tanh
    live in different activation spaces, so their cosine SW distances / isomap
    embeddings are meaningless (this is exactly the tanh-vs-relu mismatch that
    made vanilla dreams look off-manifold). Fail loudly rather than plot garbage.
    """
    mn = float(np.min(X))
    if mn < -1e-4:
        frac_neg = float((X < 0).mean())
        raise AssertionError(
            f"{source} activations for '{layer_name}' contain negative values "
            f"(min={mn:.3f}, {frac_neg:.1%} negative) -> NOT relu space. This run "
            f"was likely generated with gru_act=tanh; regenerate it with "
            f"gru_act=relu so wake and dream share one activation space.")


def load_wake_activations(data_path, layer_name, max_samples, min_bbox=0):
    """Load wake activations and positions for a single layer.

    Returns:
        X: (N, D) activations
        pos: (N, 2) positions
    """
    episodes, metadata = load_episodes(data_path)
    if min_bbox > 0:
        episodes = filter_stuck_episodes(episodes, min_bbox)
    X, pos, _groups = _prepare_single_layer(episodes, layer_name)

    # Subsample if needed (matching pRNN randSubSample)
    if max_samples > 0 and len(X) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), max_samples, replace=False)
        X, pos = X[idx], pos[idx]

    X = X.astype(np.float32)
    _assert_relu_space(X, layer_name, source='WAKE')
    return X, pos.astype(np.float32), metadata


def load_dream_activations(dream_path, layer_name):
    """Load dream activations from dream_results.pkl or a trajectory directory.

    If dream_path points to a .pkl file, loads dream_deter/dream_stoch from it.
    If dream_path is a directory, treats it as a second wake trajectory set and
    loads activations using _prepare_single_layer.

    Returns:
        X: (M, D) activations (flattened from (N, H, D) if from dream pkl)
        t_idx: (M,) dream-timestep index of each row (0..H) when the source has a
            time axis, else None. Row-major reshape means t = flat_index % T, so
            this survives later row subsampling as long as we index it in lockstep.
    """
    dream_path = Path(dream_path)

    # Case 1: pkl file from dream_decode
    if dream_path.suffix == '.pkl' or (dream_path.is_file() and not dream_path.is_dir()):
        with open(dream_path, 'rb') as f:
            data = pickle.load(f)

        key_map = {'dyn/deter': 'dream_deter', 'dyn/stoch': 'dream_stoch'}
        pkl_key = key_map.get(layer_name)
        if pkl_key is None or pkl_key not in data:
            raise ValueError(
                f"Layer '{layer_name}' not found in dream pkl. "
                f"Available: {[k for k in data if k.startswith('dream_') and not k.endswith('shape')]}")
        arr = np.array(data[pkl_key], dtype=np.float32)
        # Reshape from (N, H, D) or (N, H, S, C) to (N*H, D); H is the time axis.
        t_idx = None
        if arr.ndim == 4:
            N, H = arr.shape[:2]
            arr = arr.reshape(N * H, -1)
            t_idx = np.tile(np.arange(H), N)
        elif arr.ndim == 3:
            N, H, D = arr.shape
            arr = arr.reshape(N * H, D)
            t_idx = np.tile(np.arange(H), N)
        _assert_relu_space(arr, layer_name, source='DREAM')
        return arr, t_idx

    # Case 2: directory — treat as second wake trajectory set (no dream time axis)
    if dream_path.is_dir():
        episodes, _ = load_episodes(dream_path)
        X, _pos, _groups = _prepare_single_layer(episodes, layer_name)
        X = X.astype(np.float32)
        _assert_relu_space(X, layer_name, source='DREAM')
        return X, None

    raise ValueError(f"dream_data path not found or unrecognized: {dream_path}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_srsa(X, pos):
    """Spearman RSA: rank correlation between neural and spatial distances.

    Returns sRSA rho, p-value, and the 2D conditional histogram P[neural|spatial]
    matching pRNN's calculateRSA_space.

    Args:
        X: (N, D) activations
        pos: (N, 2) positions

    Returns:
        rho, pval, neural_dists, spatial_dists, hist2, neural_bins, spatial_bins
    """
    neural_dists = pdist(X, metric='cosine')
    spatial_dists = pdist(pos, metric='euclidean')

    # Filter NaN pairs (from zero-vector cosine distances)
    finite_mask = np.isfinite(neural_dists)
    rho, pval = spearmanr(neural_dists[finite_mask], spatial_dists[finite_mask])

    # 2D conditional histogram P[neural | spatial] (pRNN style)
    neural_bins = np.linspace(0, 1, 50)
    sp_max = np.percentile(spatial_dists, 99)
    spatial_bins = np.arange(-0.5, sp_max + 1.5, 1)
    hist2, _nb, _sb = np.histogram2d(neural_dists[finite_mask], spatial_dists[finite_mask],
                                      bins=[neural_bins, spatial_bins])
    # Normalize each spatial column to get P[neural | spatial]
    col_sums = hist2.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1
    hist2 = hist2 / col_sums

    return rho, pval, neural_dists, spatial_dists, hist2, neural_bins, spatial_bins


def _hill_saturating(t, x0, x_inf, t_half, n):
    """Hill saturating function: x0 + (x_inf - x0) * t^n / (t^n + t_half^n)."""
    return x0 + (x_inf - x0) * (t**n) / (t**n + t_half**n)


def compute_hill_fit(neural_dists, spatial_dists):
    """Fit Hill saturating function to binned neural-vs-spatial distance.

    Matches pRNN's calculateHillFit.

    Returns:
        hill_fit dict with x0, x_inf, t_half, n, sp_centers, hdist_means, hdist_stds
        or None if fitting fails.
    """
    sp_bins = np.arange(-0.5, 21.5, 1)
    sp_centers = (sp_bins[:-1] + sp_bins[1:]) / 2
    bin_indices = np.digitize(spatial_dists, sp_bins)

    hdist_means = []
    hdist_stds = []
    for i in range(1, len(sp_bins)):
        mask = bin_indices == i
        if mask.sum() > 0:
            hdist_means.append(neural_dists[mask].mean())
            hdist_stds.append(neural_dists[mask].std())
        else:
            hdist_means.append(np.nan)
            hdist_stds.append(np.nan)

    # Initial guesses (pRNN style)
    close_mask = spatial_dists < 1
    x0_guess = neural_dists[close_mask].mean() if close_mask.any() else 0.3
    t_half_guess = np.mean(spatial_dists)
    far_mask = spatial_dists > t_half_guess
    xinf_guess = neural_dists[far_mask].mean() if far_mask.any() else 0.8

    try:
        popt, _ = curve_fit(
            _hill_saturating, spatial_dists, neural_dists,
            p0=[x0_guess, xinf_guess, t_half_guess, 2.0],
            bounds=([0, 0, 0, 0.1], [1, 1, np.inf, 10]),
            maxfev=5000)
        return {
            'x0': float(popt[0]),
            'x_inf': float(popt[1]),
            't_half': float(popt[2]),
            'n': float(popt[3]),
            'sp_centers': sp_centers,
            'hdist_means': hdist_means,
            'hdist_stds': hdist_stds,
        }
    except (RuntimeError, ValueError) as e:
        print(f"  Hill fit failed: {e}")
        return None


def compute_sw_distance(X_wake, X_dream):
    """Median minimum cosine distance from each dream point to nearest wake point.

    Matches pRNN's calculateSleepWakeDist.

    Args:
        X_wake: (N, D) wake activations
        X_dream: (M, D) dream activations

    Returns:
        median_sw: median min cosine distance
        min_dists: (M,) per-dream-point minimum distances
    """
    # Compute in chunks to avoid OOM for large arrays
    chunk_size = 2000
    M = len(X_dream)
    min_dists = np.empty(M, dtype=np.float32)

    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        dists = cdist(X_dream[start:end], X_wake, metric='cosine')
        min_dists[start:end] = dists.min(axis=1)

    median_sw = float(np.nanmedian(min_dists))
    return median_sw, min_dists


def compute_isomap(X_wake, X_dream, n_neighbors=150, max_total=8000):
    """Fit Isomap on concatenated wake+dream, return 2D embedding.

    Matches pRNN's fitIsomap: fit on concatenated data, metric='cosine'.

    Args:
        X_wake: (N, D)
        X_dream: (M, D)
        n_neighbors: k-NN neighbors for Isomap
        max_total: max combined samples (subsample dream if needed)

    Returns:
        emb_wake: (N, 2)
        emb_dream: (M', 2)
        dream_idx: indices into original X_dream that were kept
    """
    N = len(X_wake)

    # Subsample dream if combined would be too large
    max_dream = max_total - N
    if max_dream <= 0:
        max_dream = min(len(X_dream), 2000)
    dream_idx = np.arange(len(X_dream))
    if len(X_dream) > max_dream:
        rng = np.random.RandomState(42)
        dream_idx = rng.choice(len(X_dream), max_dream, replace=False)

    X_combined = np.vstack([X_wake, X_dream[dream_idx]])

    # Adjust n_neighbors if dataset is small
    n_neighbors = min(n_neighbors, len(X_combined) - 1)

    iso = Isomap(n_neighbors=n_neighbors, n_components=2, metric='cosine')
    emb = iso.fit_transform(X_combined)

    emb_wake = emb[:N]
    emb_dream = emb[N:]
    return emb_wake, emb_dream, dream_idx


# ---------------------------------------------------------------------------
# Plotting (dark theme)
# ---------------------------------------------------------------------------

def _setup_dark_style():
    """Dark theme consistent with repo plotting style."""
    plt.rcParams.update({
        'figure.facecolor': '#1a1a1a',
        'axes.facecolor': '#1a1a1a',
        'axes.edgecolor': '#555555',
        'axes.labelcolor': 'white',
        'text.color': 'white',
        'xtick.color': '#cccccc',
        'ytick.color': '#cccccc',
        'grid.color': '#333333',
        'savefig.facecolor': '#1a1a1a',
    })


def plot_isomap_position(emb_wake, pos_wake, layer_name, save_dir,
                         mapcenter=None):
    """Isomap colored by arctan(x/y) position angle (viridis, pRNN style)."""
    _setup_dark_style()

    if mapcenter is None:
        mapcenter = [pos_wake[:, 0].mean(), pos_wake[:, 1].mean()]
    color = np.arctan((pos_wake[:, 0] - mapcenter[0]) /
                      (pos_wake[:, 1] - mapcenter[1]))

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    sc = ax.scatter(emb_wake[:, 0], emb_wake[:, 1], c=color, cmap='viridis',
                    s=4, alpha=0.6, edgecolors='none')
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label('arctan(x/y)', fontsize=9)
    ax.axis('off')
    ax.set_title(f'{layer_name} — position', fontsize=11, fontweight='bold')

    out = save_dir / f'{layer_name.replace("/", "_")}_isomap_position.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out.name}")


def plot_isomap_wakedream(emb_wake, emb_dream, layer_name, save_dir):
    """Isomap with wake (white) vs dream (red) overlay.

    Matches pRNN's isomapPanel with 'SleepWake' and 'Sleep' colorvar.
    """
    _setup_dark_style()

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    ax.scatter(emb_wake[:, 0], emb_wake[:, 1], c='white', s=4, alpha=0.3,
               edgecolors='none', label='wake', zorder=2)
    ax.scatter(emb_dream[:, 0], emb_dream[:, 1], c='#cc0000', s=4, alpha=0.5,
               edgecolors='none', label='dream', zorder=3)
    ax.legend(fontsize=9, loc='upper right', facecolor='#333333',
              edgecolor='grey', labelcolor='white')
    ax.set_xlabel('Isomap 1', fontsize=10)
    ax.set_ylabel('Isomap 2', fontsize=10)
    ax.set_title(f'{layer_name} — Wake vs Dream manifold', fontsize=11,
                 fontweight='bold')

    out = save_dir / f'{layer_name.replace("/", "_")}_isomap_wakedream.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out.name}")


def plot_srsa(hist2, neural_bins, spatial_bins, rho, layer_name, save_dir):
    """2D conditional probability histogram P[neural|spatial] (pRNN style).

    Matches pRNN's spatialRSApanel: imshow with 'binary' cmap, origin='lower',
    sRSA value overlaid in red.
    """
    _setup_dark_style()

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    vmax = np.median(hist2.max(axis=0))  # pRNN 'auto' vmax
    im = ax.imshow(hist2, origin='lower', aspect='auto',
                   extent=(spatial_bins[0], spatial_bins[-1],
                           neural_bins[0], neural_bins[-1]),
                   cmap='binary', vmin=0, vmax=vmax,
                   interpolation='none')
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('P[neural|space]', fontsize=9)
    ax.text(spatial_bins[1], neural_bins[-6],
            f'sRSA: {rho:.3f}', fontsize=12, color='red', fontweight='bold')
    ax.set_xlabel('Spatial distance (Euclidean)', fontsize=10)
    ax.set_ylabel('Neural distance (cosine)', fontsize=10)
    ax.set_title(f'{layer_name} — Spatial RSA', fontsize=11, fontweight='bold')

    out = save_dir / f'{layer_name.replace("/", "_")}_srsa.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out.name}")


def plot_hill_fit(hill_fit, layer_name, save_dir):
    """Neural distance vs spatial distance with Hill fit (pRNN style).

    Matches pRNN's hillFitPanel: error bars on binned data + fitted curve +
    marker annotations for dh_0, dh_inf, dx_1/2.
    """
    _setup_dark_style()

    sp_centers = hill_fit['sp_centers']
    hdist_means = np.array(hill_fit['hdist_means'], dtype=float)
    hdist_stds = np.array(hill_fit['hdist_stds'], dtype=float)

    # Filter NaN bins
    valid = ~np.isnan(hdist_means)
    sp_valid = sp_centers[valid]
    means_valid = hdist_means[valid]
    stds_valid = hdist_stds[valid]

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    # Data with error bars
    ax.errorbar(sp_valid, means_valid, yerr=stds_valid, fmt='none',
                capsize=4, alpha=0.6, color='#cccccc', label='Data')

    # Fitted curve
    t_plot = np.linspace(sp_valid.min(), sp_valid.max(), 100)
    y_fitted = _hill_saturating(t_plot, hill_fit['x0'], hill_fit['x_inf'],
                                 hill_fit['t_half'], hill_fit['n'])
    ax.plot(t_plot, y_fitted, '-', linewidth=2, color='#44aaff')

    # Marker annotations (pRNN style)
    ax.plot(-0.5, hill_fit['x0'], '<', color='#ff6666', markersize=10,
            label=f"dh_0: {hill_fit['x0']:.3f}")
    ax.plot(sp_valid.max() + 1, hill_fit['x_inf'], '<', color='#66ff66',
            markersize=10, label=f"dh_inf: {hill_fit['x_inf']:.3f}")
    ax.plot(hill_fit['t_half'], 1.05, 'v', color='#ffaa44', markersize=10,
            label=f"dx_1/2: {hill_fit['t_half']:.2f}")

    ax.set_xlabel('Spatial distance', fontsize=10)
    ax.set_ylabel('Neural distance (cosine)', fontsize=10)
    ax.set_title(f'{layer_name} — Hill fit', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, facecolor='#333333', edgecolor='grey',
              labelcolor='white', loc='lower right')

    out = save_dir / f'{layer_name.replace("/", "_")}_hillfit.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out.name}")


def plot_sw_histogram(min_dists, median_sw, neural_bins, layer_name, save_dir):
    """SW distance as thin vertical column image (pRNN style).

    Matches pRNN's sleepdistPanel: histogram of min distances displayed as
    imshow column with 'binary' cmap alongside a standard histogram.
    """
    _setup_dark_style()

    fig, axes = plt.subplots(1, 2, figsize=(9, 5),
                              gridspec_kw={'width_ratios': [1, 4]})

    # Filter NaNs for plotting
    finite_dists = min_dists[np.isfinite(min_dists)]
    n_nan = len(min_dists) - len(finite_dists)
    if n_nan > 0:
        print(f"  WARNING: {n_nan}/{len(min_dists)} SW distances are NaN (zero-vector cosine)")

    # Cosine distance on non-negative (relu) activations is bounded to [0, 1];
    # fix the distance axis + bins there so plots are comparable across runs.
    sw_bins = np.linspace(0.0, 1.0, 51)

    # Left panel: thin vertical column (pRNN sleepdistPanel style)
    ax = axes[0]
    if len(finite_dists) > 0:
        n, bins = np.histogram(finite_dists, bins=sw_bins)
        n_total = n.sum()
        n = n / n_total if n_total > 0 else n
        ax.imshow(np.expand_dims(n, axis=1), origin='lower', aspect='auto',
                  extent=(0, 0.1, 0.0, 1.0),
                  cmap='binary', interpolation='none')
    ax.set_xlabel('S-W', fontsize=9)
    ax.set_ylabel('Neural distance (cosine)', fontsize=10)
    ax.set_ylim(0.0, 1.0)
    ax.xaxis.set_ticklabels([])

    # Right panel: regular histogram with median line
    ax = axes[1]
    if len(finite_dists) > 0:
        ax.hist(finite_dists, bins=sw_bins, color='#ff6666', edgecolor='#cc3333',
                alpha=0.8, orientation='horizontal')
    ax.axhline(median_sw, color='cyan', linestyle='--', linewidth=2,
               label=f'median = {median_sw:.4f}')
    ax.set_xlabel('Count', fontsize=10)
    ax.set_ylabel('')
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=9, facecolor='#333333', edgecolor='grey',
              labelcolor='white')

    fig.suptitle(f'{layer_name} — SW distance', fontsize=11, fontweight='bold')
    fig.tight_layout()

    out = save_dir / f'{layer_name.replace("/", "_")}_swdist.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out.name}")


def plot_sw_over_time(min_dists, t_idx, layer_name, save_dir):
    """SW distance (dream -> nearest wake point) as a function of dream timestep.

    t=0 is the seed latent; t=1..H are the imagined steps. A rising curve = the
    dream drifts off the wake manifold as it rolls out; flat/falling = it stays
    on (or relaxes onto) the manifold. Bounded to [0, 1] (cosine distance on
    non-negative activations), matching the other SW panels.
    """
    _setup_dark_style()
    finite = np.isfinite(min_dists) & np.isfinite(t_idx)
    md, ti = min_dists[finite], t_idx[finite].astype(int)
    if len(md) == 0:
        print("  WARNING: no finite SW/timestep pairs; skipping sw_over_time")
        return
    T = int(ti.max()) + 1
    steps = np.arange(T)
    med = np.full(T, np.nan); q25 = np.full(T, np.nan); q75 = np.full(T, np.nan)
    for t in steps:
        v = md[ti == t]
        if len(v):
            med[t] = np.median(v)
            q25[t] = np.percentile(v, 25)
            q75[t] = np.percentile(v, 75)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(steps, q25, q75, color='#ff6666', alpha=0.2, label='IQR')
    ax.plot(steps, med, color='#ff6666', linewidth=2, label='median SW (dream→wake)')
    ax.set_xlabel('Dream step (0 = seed)', fontsize=10)
    ax.set_ylabel('SW distance (cosine)', fontsize=10)
    ax.set_xlim(0, T - 1)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f'{layer_name} — SW distance over dream time',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, facecolor='#333333', edgecolor='grey', labelcolor='white')
    fig.tight_layout()
    out = save_dir / f'{layer_name.replace("/", "_")}_sw_over_time.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out.name}")


def _sw_time_profile(min_dists, t_idx):
    """Per-timestep median + IQR of SW distances. Returns (steps, med, q25, q75).
    Mirrors the aggregation in plot_sw_over_time so both stay comparable."""
    finite = np.isfinite(min_dists) & np.isfinite(t_idx)
    md, ti = min_dists[finite], t_idx[finite].astype(int)
    if len(md) == 0:
        return None
    T = int(ti.max()) + 1
    steps = np.arange(T)
    med = np.full(T, np.nan); q25 = np.full(T, np.nan); q75 = np.full(T, np.nan)
    for t in steps:
        v = md[ti == t]
        if len(v):
            med[t] = np.median(v)
            q25[t] = np.percentile(v, 25)
            q75[t] = np.percentile(v, 75)
    return steps, med, q25, q75


def plot_sw_over_time_by_condition(cond_results, layer_name, save_dir,
                                   shading='band'):
    """Overlay SW-distance-over-time curves for seed-ablation conditions A/B/C/D
    (plus A' when a perturbed-state dump is present).

    cond_results: {cond: {'min_dists': (M,), 't_idx': (M,)}} against a SHARED wake
    manifold. Rising curve = that condition's dream drifts off the wake manifold as
    it rolls out; flat/low = stays on it. Conditions share one axis so real-deter
    (A/B/A') vs noise-deter (C/D) drift is directly comparable. A' vs A isolates
    whether perturbing the interoceptive channel pushes the dream further off the
    wake manifold. Bounded to [0, 1] (cosine distance on non-negative
    activations)."""
    _setup_dark_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    Tmax = 1
    for c in SEED_ABLATION_ORDER:
        if c not in cond_results:
            continue
        prof = _sw_time_profile(cond_results[c]['min_dists'],
                                cond_results[c]['t_idx'])
        if prof is None:
            print(f"  WARNING: no finite SW/timestep pairs for condition {c}")
            continue
        steps, med, q25, q75 = prof
        Tmax = max(Tmax, len(steps))
        color = SEED_ABLATION_COLORS[c]
        ls = SEED_ABLATION_LINESTYLES.get(c, '-')
        if shading == 'band':
            ax.fill_between(steps, q25, q75, color=color, alpha=0.15)
        ax.plot(steps, med, color=color, linewidth=2, linestyle=ls,
                label=SEED_ABLATION_LABELS[c])
    ax.set_xlabel('Dream step (0 = seed)', fontsize=10)
    ax.set_ylabel('SW distance to wake manifold (cosine)', fontsize=10)
    ax.set_xlim(0, Tmax - 1)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f'{layer_name} — SW distance over dream time by condition',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, facecolor='#333333', edgecolor='grey',
              labelcolor='white', loc='upper left')
    fig.tight_layout()
    out = save_dir / f'{layer_name.replace("/", "_")}_sw_over_time_by_condition.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out.name}")


def plot_wake_sleep_figure(layer_results, layer_name, save_dir):
    """Combined WakeSleep figure (pRNN WakeSleepFigure style).

    Layout: sRSA panel | SW column | Isomap position | Isomap wake+dream
    """
    _setup_dark_style()

    has_isomap = 'isomap_wake' in layer_results
    ncols = 4 if has_isomap else 2
    widths = [4, 1, 3, 3] if has_isomap else [4, 1]

    fig, axes = plt.subplots(1, ncols, figsize=(sum(widths) * 1.5, 5),
                              gridspec_kw={'width_ratios': widths})

    # Panel 1: sRSA conditional histogram
    ax = axes[0]
    r = layer_results
    hist2 = r['srsa_hist2']
    neural_bins = r['srsa_neural_bins']
    spatial_bins = r['srsa_spatial_bins']
    vmax = np.median(hist2.max(axis=0))
    im = ax.imshow(hist2, origin='lower', aspect='auto',
                   extent=(spatial_bins[0], spatial_bins[-1],
                           neural_bins[0], neural_bins[-1]),
                   cmap='binary', vmin=0, vmax=vmax, interpolation='none')
    ax.text(spatial_bins[1], neural_bins[-6],
            f"sRSA: {r['srsa_rho']:.3f}", fontsize=10, color='red',
            fontweight='bold')
    ax.set_xlabel('Spatial dist', fontsize=9)
    ax.set_ylabel('Neural dist (cosine)', fontsize=9)

    # Panel 2: SW distance column (cosine dist on relu activations is in [0, 1])
    ax = axes[1]
    min_dists = r['sw_min_dists']
    finite_dists = min_dists[np.isfinite(min_dists)]
    sw_bins = np.linspace(0.0, 1.0, 51)
    n, bins = np.histogram(finite_dists, bins=sw_bins) if len(finite_dists) > 0 else (np.zeros(len(sw_bins)-1), sw_bins)
    n_norm = n / max(n.sum(), 1)
    ax.imshow(np.expand_dims(n_norm, axis=1), origin='lower', aspect='auto',
              extent=(0, 0.1, 0.0, 1.0),
              cmap='binary', interpolation='none')
    ax.set_xlabel('S-W', fontsize=9)
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_ticklabels([])
    ax.xaxis.set_ticklabels([])

    if has_isomap:
        emb_wake = r['isomap_wake']
        emb_dream = r['isomap_dream']
        pos_wake = r['pos_wake']

        # Panel 3: Isomap colored by position angle (pRNN style)
        ax = axes[2]
        cx, cy = pos_wake[:, 0].mean(), pos_wake[:, 1].mean()
        color = np.arctan((pos_wake[:, 0] - cx) / (pos_wake[:, 1] - cy))
        sc = ax.scatter(emb_wake[:, 0], emb_wake[:, 1], c=color,
                        cmap='viridis', s=4, alpha=0.6, edgecolors='none')
        fig.colorbar(sc, ax=ax, shrink=0.7)
        ax.axis('off')
        ax.set_title('position', fontsize=9, color='white')

        # Panel 4: Isomap wake + dream overlay
        ax = axes[3]
        ax.scatter(emb_wake[:, 0], emb_wake[:, 1], c='white', s=4, alpha=0.3,
                   edgecolors='none')
        ax.scatter(emb_dream[:, 0], emb_dream[:, 1],
                   c=np.tile([0.7, 0, 0], (len(emb_dream), 1)),
                   s=4, alpha=0.5, edgecolors='none')
        ax.axis('off')
        ax.set_title('wake + dream', fontsize=9, color='white')

    fig.suptitle(f'{layer_name}', fontsize=12, fontweight='bold', color='white')
    fig.tight_layout()
    out = save_dir / f'{layer_name.replace("/", "_")}_wakesleep.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out.name}")


def plot_summary(results, save_dir):
    """Multi-panel summary figure across all layers."""
    _setup_dark_style()

    layers = list(results.keys())
    n_layers = len(layers)
    if n_layers == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: sRSA bar chart
    ax = axes[0, 0]
    srsa_vals = [results[l]['srsa_rho'] for l in layers]
    ax.bar(range(n_layers), srsa_vals, color='#44aaff', edgecolor='#2277cc')
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([l.replace('/', '\n') for l in layers], fontsize=8)
    ax.set_ylabel('Spearman rho', fontsize=10)
    ax.set_title('sRSA (spatial structure)', fontsize=11, fontweight='bold')
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.5)

    # Panel 2: SW distance bar chart
    ax = axes[0, 1]
    sw_vals = [results[l]['sw_median'] for l in layers]
    ax.bar(range(n_layers), sw_vals, color='#ff6666', edgecolor='#cc3333')
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([l.replace('/', '\n') for l in layers], fontsize=8)
    ax.set_ylabel('Median min cosine dist', fontsize=10)
    ax.set_title('SW distance (manifold proximity)', fontsize=11,
                 fontweight='bold')

    # Panel 3: Hill fit parameters (if available)
    ax = axes[1, 0]
    hill_data = {l: results[l].get('hill_fit') for l in layers}
    has_hill = any(h is not None for h in hill_data.values())
    if has_hill:
        x = np.arange(n_layers)
        w = 0.25
        dh0 = [hill_data[l]['x0'] if hill_data[l] else 0 for l in layers]
        dhinf = [hill_data[l]['x_inf'] if hill_data[l] else 0 for l in layers]
        thalf = [hill_data[l]['t_half'] if hill_data[l] else 0 for l in layers]
        ax.bar(x - w, dh0, w, color='#ff6666', label='dh_0')
        ax.bar(x, dhinf, w, color='#66ff66', label='dh_inf')
        ax.bar(x + w, thalf, w, color='#ffaa44', label='dx_1/2')
        ax.set_xticks(x)
        ax.set_xticklabels([l.replace('/', '\n') for l in layers], fontsize=8)
        ax.set_title('Hill fit parameters', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, facecolor='#333333', edgecolor='grey',
                  labelcolor='white')
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, 'Hill fit not available', transform=ax.transAxes,
                ha='center', va='center', fontsize=10)

    # Panel 4: text summary
    ax = axes[1, 1]
    ax.axis('off')
    lines = ['Layer Summary', '─' * 40]
    for l in layers:
        r = results[l]
        lines.append(f"{l}:")
        lines.append(f"  sRSA ρ = {r['srsa_rho']:.4f} (p={r['srsa_pval']:.2e})")
        lines.append(f"  SW median = {r['sw_median']:.4f}")
        hf = r.get('hill_fit')
        if hf:
            lines.append(f"  dh_0={hf['x0']:.3f}  dh_inf={hf['x_inf']:.3f}  "
                         f"dx_1/2={hf['t_half']:.2f}  n={hf['n']:.2f}")
        lines.append(f"  wake N = {r['n_wake']}, dream N = {r['n_dream']}")
    ax.text(0.05, 0.95, '\n'.join(lines), transform=ax.transAxes,
            fontsize=9, fontfamily='monospace', verticalalignment='top',
            color='white')

    fig.suptitle('Neural Manifold Analysis', fontsize=14, fontweight='bold',
                 color='white')
    fig.tight_layout()
    out = save_dir / 'manifold_summary.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out.name}")


# ---------------------------------------------------------------------------
# Seed-ablation condition overlay (SW-over-time for A/B/C/D vs shared wake)
# ---------------------------------------------------------------------------

def run_seed_ablation_overlay(args, save_dir, tuning_data):
    """SW-distance-over-time for the four seed-ablation conditions against one
    shared wake manifold. Consumes a dream_seed_ablation folder holding
    dream_deter_{A,B,C,D}.pkl (from a --save_activations run). One condition is
    held in memory at a time (dumps are 8-16 GB); only the small per-point
    min_dists + timestep labels are retained for plotting."""
    sa_dir = Path(args.seed_ablation_dir)
    all_results = {}
    output_files = []

    # A/B/C/D always; include A' iff its perturbed-state dump is present (from a
    # --perturb_state --save_activations run). Auto-detected once, up front.
    conditions = list(SEED_ABLATION_CONDITIONS)
    if (sa_dir / f'dream_deter_{APRIME}.pkl').exists():
        conditions.append(APRIME)
        print(f"Detected {sa_dir.name}/dream_deter_{APRIME}.pkl — including A' "
              "(perturbed-state) as a 5th condition")
    if (sa_dir / f'dream_deter_{ECOND}.pkl').exists():
        conditions.append(ECOND)
        print(f"Detected {sa_dir.name}/dream_deter_{ECOND}.pkl — including E "
              "(blank-slate) as an additional condition")

    for layer_name in args.layers:
        print(f"\n{'='*60}\nLayer: {layer_name}\n{'='*60}")

        # Wake manifold — loaded ONCE, reused for every condition so the
        # comparison is against an identical reference.
        print("Loading wake activations...")
        try:
            X_wake, pos_wake, _meta = load_wake_activations(
                args.data, layer_name, args.max_wake_samples,
                min_bbox=args.min_bbox)
        except (ValueError, KeyError) as e:
            print(f"  SKIP {layer_name}: {e}")
            continue
        print(f"  Wake: {X_wake.shape[0]} samples, dim={X_wake.shape[1]}")

        # Optional EV neuron subset (same neuron_idx applied to wake + every
        # condition so all live in the same neural subspace).
        neuron_idx = None
        tl = (tuning_data.get('layers', tuning_data)
              if tuning_data is not None else None)
        if tl is not None and layer_name in tl:
            evs = np.asarray(tl[layer_name]['metrics']['EVs'], dtype=float)
            ev_mask = np.isfinite(evs) & (evs > args.ev_thresh)
            if ev_mask.sum() < 10:
                print(f"  SKIP {layer_name}: only {ev_mask.sum()} neurons "
                      f"above EV > {args.ev_thresh}")
                continue
            neuron_idx = np.where(ev_mask)[0]
            print(f"  EV filter: {len(neuron_idx)}/{len(evs)} neurons")
            X_wake = X_wake[:, neuron_idx]

        cond_results = {}
        for c in conditions:
            pkl = sa_dir / f'dream_deter_{c}.pkl'
            if not pkl.exists():
                print(f"  condition {c}: {pkl.name} not found, skipping")
                continue
            print(f"  condition {c}: loading {pkl.name} ...")
            X_dream, t_dream = load_dream_activations(pkl, layer_name)
            if t_dream is None:
                print(f"    no time axis in {pkl.name}, skipping")
                continue
            if args.max_dream_samples > 0 and len(X_dream) > args.max_dream_samples:
                rng = np.random.RandomState(43)
                idx = rng.choice(len(X_dream), args.max_dream_samples, replace=False)
                X_dream, t_dream = X_dream[idx], t_dream[idx]
            if neuron_idx is not None:
                X_dream = X_dream[:, neuron_idx]
            median_sw, min_dists = compute_sw_distance(X_wake, X_dream)
            print(f"    {X_dream.shape[0]} dream pts, median SW={median_sw:.4f}")
            cond_results[c] = {
                'min_dists': min_dists, 't_idx': t_dream,
                'median_sw': float(median_sw), 'n_dream': int(len(min_dists)),
            }
            del X_dream  # free the big array before the next condition

        if not cond_results:
            print(f"  No conditions found for {layer_name}; skipping.")
            continue

        print("Generating condition overlay...")
        plot_sw_over_time_by_condition(cond_results, layer_name, save_dir,
                                       shading=args.shading)
        slug = layer_name.replace('/', '_')
        output_files.append(f'{slug}_sw_over_time_by_condition.png')
        all_results[layer_name] = {
            'n_wake': int(len(X_wake)),
            'conditions': {c: {k: v for k, v in cr.items()}
                           for c, cr in cond_results.items()},
        }

    if not all_results:
        print("\nNo layers/conditions successfully analyzed.")
        return

    results_file = save_dir / 'manifold_condition_results.pkl'
    with open(results_file, 'wb') as f:
        pickle.dump(all_results, f)
    output_files.append('manifold_condition_results.pkl')
    print(f"\nResults saved to {results_file}")

    log_run_info(
        save_dir=save_dir, stage='manifold_analysis_conditions',
        args=vars(args), outputs=output_files,
        extra={'seed_ablation_dir': str(sa_dir), 'layers': args.layers})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Neural manifold analysis (sRSA, Isomap, SW distance)')
    parser.add_argument('--data', required=True,
                        help='Wake trajectory directory (eval_trajectory output)')
    parser.add_argument('--dream_data', default=None,
                        help='Dream results pkl (from dream_decode) or second '
                             'wake trajectory directory. Mutually exclusive with '
                             '--seed_ablation_dir')
    parser.add_argument('--seed_ablation_dir', default=None,
                        help='dream_seed_ablation folder with dream_deter_'
                             '{A,B,C,D}.pkl (from a --save_activations run). '
                             'Switches to condition-overlay mode: one SW-over-'
                             'time figure per layer overlaying A/B/C/D against '
                             'the shared wake manifold. If dream_deter_Aprime.pkl '
                             'is also present (a --perturb_state run), A\' is '
                             'overlaid as a 5th condition. Only SW-over-time is '
                             'produced in this mode.')
    parser.add_argument('--shading', default='band', choices=['band', 'none'],
                        help='Condition overlay spread: filled IQR band or line '
                             'only (default: band)')
    parser.add_argument('--save', required=True,
                        help='Output directory')
    parser.add_argument('--layers', nargs='+', default=DREAM_LAYERS,
                        help='Layers to analyze (default: dyn/deter dyn/stoch)')
    parser.add_argument('--max_wake_samples', type=int, default=4000,
                        help='Max wake samples (0=all, default: 4000)')
    parser.add_argument('--max_dream_samples', type=int, default=0,
                        help='Max dream samples for SW distance (0=all)')
    parser.add_argument('--n_neighbors', type=int, default=150,
                        help='Isomap k-NN neighbors (default: 150)')
    parser.add_argument('--metric', default='cosine',
                        help='Distance metric for neural space (default: cosine)')
    parser.add_argument('--min_bbox', type=int, default=0,
                        help='Min bounding box area to filter stuck episodes')
    parser.add_argument('--no_isomap', action='store_true',
                        help='Skip Isomap (faster)')
    parser.add_argument('--no_hill', action='store_true',
                        help='Skip Hill fit')
    parser.add_argument('--ev_thresh', type=float, default=None,
                        help='Min EV threshold to subset neurons (requires --tuning_pkl)')
    parser.add_argument('--tuning_pkl', default=None,
                        help='Path to tuning_results.pkl (required for --ev_thresh)')
    args = parser.parse_args()

    if args.ev_thresh is not None and args.tuning_pkl is None:
        parser.error('--ev_thresh requires --tuning_pkl')
    if bool(args.dream_data) == bool(args.seed_ablation_dir):
        parser.error('provide exactly one of --dream_data or --seed_ablation_dir')

    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load tuning results for EV thresholding
    tuning_data = None
    if args.ev_thresh is not None:
        print(f"Loading tuning results from {args.tuning_pkl} ...")
        with open(args.tuning_pkl, 'rb') as f:
            tuning_data = pickle.load(f)
        print(f"  EV threshold: {args.ev_thresh}")

    # Condition-overlay mode: SW-over-time for A/B/C/D vs a shared wake manifold.
    if args.seed_ablation_dir:
        run_seed_ablation_overlay(args, save_dir, tuning_data)
        return

    all_results = {}
    output_files = []

    for layer_name in args.layers:
        print(f"\n{'='*60}")
        print(f"Layer: {layer_name}")
        print(f"{'='*60}")

        # --- Load wake data ---
        print("Loading wake activations...")
        try:
            X_wake, pos_wake, metadata = load_wake_activations(
                args.data, layer_name, args.max_wake_samples,
                min_bbox=args.min_bbox)
        except (ValueError, KeyError) as e:
            print(f"  SKIP {layer_name}: {e}")
            continue
        print(f"  Wake: {X_wake.shape[0]} samples, dim={X_wake.shape[1]}")

        # --- Load dream data ---
        print("Loading dream activations...")
        try:
            X_dream, t_dream = load_dream_activations(args.dream_data, layer_name)
        except (ValueError, KeyError) as e:
            print(f"  SKIP {layer_name}: {e}")
            continue

        if args.max_dream_samples > 0 and len(X_dream) > args.max_dream_samples:
            rng = np.random.RandomState(43)
            idx = rng.choice(len(X_dream), args.max_dream_samples, replace=False)
            X_dream = X_dream[idx]
            if t_dream is not None:
                t_dream = t_dream[idx]  # keep timestep labels aligned with rows
        print(f"  Dream: {X_dream.shape[0]} samples, dim={X_dream.shape[1]}")

        # --- EV threshold neuron filtering ---
        # tuning pkl has top-level keys {metadata, layers}; layer data is under 'layers'
        tuning_layers = tuning_data.get('layers', tuning_data) if tuning_data is not None else None
        if tuning_layers is not None and layer_name in tuning_layers:
            evs = np.asarray(tuning_layers[layer_name]['metrics']['EVs'], dtype=float)
            ev_mask = np.isfinite(evs) & (evs > args.ev_thresh)
            n_keep = ev_mask.sum()
            n_total = len(evs)
            if n_keep < 10:
                print(f"  SKIP {layer_name}: only {n_keep}/{n_total} neurons above EV > {args.ev_thresh}")
                continue
            print(f"  EV filter: {n_keep}/{n_total} neurons above {args.ev_thresh}")
            neuron_idx = np.where(ev_mask)[0]
            X_wake = X_wake[:, neuron_idx]
            X_dream = X_dream[:, neuron_idx]
        elif tuning_layers is not None and layer_name not in tuning_layers:
            print(f"  WARNING: layer '{layer_name}' not found in tuning pkl, skipping EV filter")

        # --- sRSA on wake ---
        print("Computing sRSA...")
        rho, pval, neural_dists, spatial_dists, hist2, neural_bins, spatial_bins = \
            compute_srsa(X_wake, pos_wake)
        print(f"  sRSA rho={rho:.4f}, p={pval:.2e}")

        # --- Hill fit ---
        hill_fit = None
        if not args.no_hill:
            print("Computing Hill fit...")
            hill_fit = compute_hill_fit(neural_dists, spatial_dists)
            if hill_fit:
                print(f"  dh_0={hill_fit['x0']:.3f}, dh_inf={hill_fit['x_inf']:.3f}, "
                      f"dx_1/2={hill_fit['t_half']:.2f}, n={hill_fit['n']:.2f}")

        # --- SW distance ---
        print("Computing SW distance...")
        median_sw, min_dists = compute_sw_distance(X_wake, X_dream)
        print(f"  SW median={median_sw:.4f}, "
              f"mean={min_dists.mean():.4f}, "
              f"std={min_dists.std():.4f}")

        # --- Isomap ---
        emb_wake, emb_dream = None, None
        if not args.no_isomap:
            print("Computing Isomap...")
            try:
                emb_wake, emb_dream, dream_idx = compute_isomap(
                    X_wake, X_dream, n_neighbors=args.n_neighbors)
                print(f"  Isomap: {len(emb_wake)} wake + {len(emb_dream)} dream points")
            except Exception as e:
                print(f"  Isomap failed: {e}")

        # --- Store results ---
        layer_results = {
            'srsa_rho': float(rho),
            'srsa_pval': float(pval),
            'srsa_hist2': hist2,
            'srsa_neural_bins': neural_bins,
            'srsa_spatial_bins': spatial_bins,
            'hill_fit': hill_fit,
            'sw_median': float(median_sw),
            'sw_mean': float(min_dists.mean()),
            'sw_std': float(min_dists.std()),
            'sw_min_dists': min_dists,
            'sw_t_idx': t_dream,  # per-row dream timestep (None if no time axis)
            'n_wake': len(X_wake),
            'n_dream': len(X_dream),
        }
        if emb_wake is not None:
            layer_results['isomap_wake'] = emb_wake
            layer_results['isomap_dream'] = emb_dream
            layer_results['pos_wake'] = pos_wake
        all_results[layer_name] = layer_results

        # --- Plots ---
        print("Generating plots...")
        slug = layer_name.replace('/', '_')

        plot_srsa(hist2, neural_bins, spatial_bins, rho, layer_name, save_dir)
        plot_sw_histogram(min_dists, median_sw, neural_bins, layer_name, save_dir)
        output_files.extend([f'{slug}_srsa.png', f'{slug}_swdist.png'])

        if t_dream is not None:
            plot_sw_over_time(min_dists, t_dream, layer_name, save_dir)
            output_files.append(f'{slug}_sw_over_time.png')

        if hill_fit:
            plot_hill_fit(hill_fit, layer_name, save_dir)
            output_files.append(f'{slug}_hillfit.png')

        if emb_wake is not None:
            plot_isomap_position(emb_wake, pos_wake, layer_name, save_dir)
            plot_isomap_wakedream(emb_wake, emb_dream, layer_name, save_dir)
            output_files.extend([
                f'{slug}_isomap_position.png',
                f'{slug}_isomap_wakedream.png',
            ])

        # Combined WakeSleep figure (pRNN style)
        plot_wake_sleep_figure(layer_results, layer_name, save_dir)
        output_files.append(f'{slug}_wakesleep.png')

    if not all_results:
        print("\nNo layers successfully analyzed.")
        return

    # --- Summary plot ---
    plot_summary(all_results, save_dir)
    output_files.append('manifold_summary.png')

    # --- Save results pkl ---
    results_file = save_dir / 'manifold_results.pkl'
    with open(results_file, 'wb') as f:
        pickle.dump(all_results, f)
    output_files.append('manifold_results.pkl')
    print(f"\nResults saved to {results_file}")

    # --- Log provenance ---
    log_run_info(
        save_dir=save_dir,
        stage='manifold_analysis',
        args=vars(args),
        outputs=output_files,
        extra={
            'layers': list(all_results.keys()),
            'srsa': {l: all_results[l]['srsa_rho'] for l in all_results},
            'sw_median': {l: all_results[l]['sw_median'] for l in all_results},
        },
    )

    # Print summary table
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Layer':<15} {'sRSA ρ':>10} {'SW median':>12} {'dh_0':>8} "
          f"{'dh_inf':>8} {'dx_1/2':>8} {'N_wake':>8} {'N_dream':>8}")
    print(f"{'─'*15} {'─'*10} {'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for l in all_results:
        r = all_results[l]
        hf = r.get('hill_fit')
        dh0 = f"{hf['x0']:.3f}" if hf else '—'
        dhinf = f"{hf['x_inf']:.3f}" if hf else '—'
        thalf = f"{hf['t_half']:.2f}" if hf else '—'
        print(f"{l:<15} {r['srsa_rho']:>10.4f} {r['sw_median']:>12.4f} "
              f"{dh0:>8} {dhinf:>8} {thalf:>8} "
              f"{r['n_wake']:>8d} {r['n_dream']:>8d}")


if __name__ == '__main__':
    main()
