"""Tests for the position decoding pipeline (decode_position.py).

Tests use synthetic data so they don't require trajectory files or
trained models. Covers:
  - prepare_data: shapes, stoch flattening, grouping, skipping, alignment
  - LinearClassifier: shapes, softmax normalization, overfitting
  - classification_decode: shapes, proba normalization, beats shuffle
  - plot_probmap_on_world: smoke test (no crafter needed — tests graceful skip)
"""

import sys
import os
import numpy as np

# Ensure repo root is on path so dreamerv3 is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dreamerv3.decode_position import (
    prepare_data,
    plot_probmap_on_world,
)

# Classification decoder and LinearClassifier require torch
try:
    import torch
    from dreamerv3.decode_position import (
        LinearClassifier,
        classification_decode,
        save_classifier_model,
        load_classifier_model,
    )
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_episodes(n_eps=5, T=100, deter_dim=512, stoch_shape=(32, 4),
                   grid_w=32, grid_h=32):
    """Create synthetic episodes with linearly-decodable positions.

    Uses uniform random positions (not random walks) so all episodes
    cover the same region, which is critical for leave-one-episode-out CV.
    """
    rng = np.random.RandomState(0)
    # Fixed linear embedding shared across episodes (so decoder can generalise)
    W = rng.randn(2, deter_dim).astype(np.float32) * 0.5
    episodes = []
    for i in range(n_eps):
        # Uniform positions — ensures overlapping coverage across episodes
        x = rng.uniform(1, grid_w - 2, T).astype(np.float32)
        y = rng.uniform(1, grid_h - 2, T).astype(np.float32)
        pos = np.stack([x, y], axis=1)

        # Deter: embed position linearly + noise (so it's decodable)
        deter = pos @ W + rng.randn(T, deter_dim).astype(np.float32) * 0.1

        # Stoch: random (not position-correlated)
        stoch = rng.randn(T, *stoch_shape).astype(np.float32)

        episodes.append({
            'deter': deter,
            'stoch': stoch,
            'player_pos': pos,
            'episode': i + 1,
            'total_reward': float(rng.randint(0, 20)),
        })
    return episodes


# ---------------------------------------------------------------------------
# prepare_data tests
# ---------------------------------------------------------------------------

def test_prepare_data_shapes():
    """Output arrays have correct shapes and dtypes."""
    episodes = _make_episodes(n_eps=3, T=50, deter_dim=64,
                               stoch_shape=(8, 4))
    deter, stoch, pos, groups = prepare_data(episodes)

    N = 3 * 50
    assert deter.shape == (N, 64), f"deter shape: {deter.shape}"
    assert stoch.shape == (N, 8 * 4), f"stoch shape: {stoch.shape}"
    assert pos.shape == (N, 2), f"pos shape: {pos.shape}"
    assert groups.shape == (N,), f"groups shape: {groups.shape}"
    assert deter.dtype == np.float32
    assert stoch.dtype == np.float32
    assert pos.dtype == np.float32


def test_prepare_data_stoch_flattening():
    """Stoch (T, S, C) is correctly flattened to (T, S*C)."""
    episodes = _make_episodes(n_eps=1, T=20, stoch_shape=(16, 3))
    _, stoch, _, _ = prepare_data(episodes)
    assert stoch.shape == (20, 16 * 3), f"stoch shape: {stoch.shape}"


def test_prepare_data_stoch_already_flat():
    """Stoch that's already 2D should pass through unchanged."""
    ep = _make_episodes(n_eps=1, T=10, stoch_shape=(32, 4))[0]
    # Pre-flatten stoch
    ep['stoch'] = ep['stoch'].reshape(10, -1)
    assert ep['stoch'].ndim == 2
    _, stoch, _, _ = prepare_data([ep])
    assert stoch.shape == (10, 128), f"stoch shape: {stoch.shape}"


def test_prepare_data_groups():
    """Groups array correctly assigns episode indices."""
    episodes = _make_episodes(n_eps=4, T=30)
    _, _, _, groups = prepare_data(episodes)
    for i in range(4):
        mask = groups == i
        assert mask.sum() == 30, f"Episode {i} has {mask.sum()} samples"
    assert len(np.unique(groups)) == 4


def test_prepare_data_skips_missing_keys():
    """Episodes missing deter or player_pos are skipped."""
    episodes = _make_episodes(n_eps=3, T=20)
    # Remove deter from episode 1
    del episodes[1]['deter']
    deter, stoch, pos, groups = prepare_data(episodes)
    # Only 2 episodes should remain
    assert deter.shape[0] == 2 * 20
    assert len(np.unique(groups)) == 2


def test_prepare_data_length_alignment():
    """Mismatched array lengths are truncated to the shortest."""
    ep = _make_episodes(n_eps=1, T=50)[0]
    # Make deter shorter
    ep['deter'] = ep['deter'][:40]
    deter, stoch, pos, groups = prepare_data([ep])
    assert deter.shape[0] == 40
    assert stoch.shape[0] == 40
    assert pos.shape[0] == 40


# ---------------------------------------------------------------------------
# LinearClassifier tests (require torch)
# ---------------------------------------------------------------------------

def test_classifier_predict_shape():
    """predict() returns (N,) integer array."""
    if not HAS_TORCH:
        print("  SKIP (no torch)")
        return
    n_units, n_classes, N = 32, 10, 50
    clf = LinearClassifier(n_units, n_classes)
    X = np.random.randn(N, n_units).astype(np.float32)
    y = np.random.randint(0, n_classes, N)
    clf.fit(X, y, n_iters=10, verbose=False)
    pred = clf.predict(X)
    assert pred.shape == (N,), f"predict shape: {pred.shape}"
    assert np.issubdtype(pred.dtype, np.integer)


def test_classifier_predict_proba_shape_and_sum():
    """predict_proba() returns (N, n_classes) summing to 1 per row."""
    if not HAS_TORCH:
        print("  SKIP (no torch)")
        return
    n_units, n_classes, N = 32, 10, 50
    clf = LinearClassifier(n_units, n_classes)
    X = np.random.randn(N, n_units).astype(np.float32)
    y = np.random.randint(0, n_classes, N)
    clf.fit(X, y, n_iters=10, verbose=False)
    proba = clf.predict_proba(X)
    assert proba.shape == (N, n_classes), f"proba shape: {proba.shape}"
    row_sums = proba.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-5,
                               err_msg="Probabilities don't sum to 1")
    assert (proba >= 0).all(), "Negative probabilities"


def test_classifier_can_overfit():
    """Classifier should perfectly fit a trivial identity mapping."""
    if not HAS_TORCH:
        print("  SKIP (no torch)")
        return
    n_classes = 5
    N = 100
    X = np.eye(n_classes, dtype=np.float32)
    X = np.tile(X, (N // n_classes, 1))  # (100, 5)
    y = np.tile(np.arange(n_classes), N // n_classes)  # (100,)
    clf = LinearClassifier(n_classes, n_classes, lr=0.01, weight_decay=0.0)
    clf.fit(X, y, n_iters=2000, verbose=False)
    pred = clf.predict(X)
    accuracy = (pred == y).mean()
    assert accuracy > 0.95, f"Failed to overfit: accuracy={accuracy:.3f}"


# ---------------------------------------------------------------------------
# classification_decode tests (require torch)
# ---------------------------------------------------------------------------

def test_classification_decode_shapes():
    """Output shapes from classification_decode are correct."""
    if not HAS_TORCH:
        print("  SKIP (no torch)")
        return
    episodes = _make_episodes(n_eps=3, T=40, deter_dim=32, grid_w=16, grid_h=16)
    deter, _, pos, groups = prepare_data(episodes)
    width, height = 16, 16
    errors, shuffle, pred_xy, true_xy, proba = classification_decode(
        deter, pos, groups, width, height, n_iters=200)
    N = 3 * 40
    assert errors.shape == (N,), f"errors shape: {errors.shape}"
    assert shuffle.shape == (N,), f"shuffle shape: {shuffle.shape}"
    assert pred_xy.shape == (N, 2), f"pred_xy shape: {pred_xy.shape}"
    assert true_xy.shape == (N, 2), f"true_xy shape: {true_xy.shape}"
    assert proba.shape == (N, width, height), f"proba shape: {proba.shape}"


def test_classification_decode_proba_normalized():
    """Each timestep's probability grid sums to ~1.0."""
    if not HAS_TORCH:
        print("  SKIP (no torch)")
        return
    episodes = _make_episodes(n_eps=3, T=30, deter_dim=32, grid_w=8, grid_h=8)
    deter, _, pos, groups = prepare_data(episodes)
    _, _, _, _, proba = classification_decode(
        deter, pos, groups, 8, 8, n_iters=100)
    sums = proba.sum(axis=(1, 2))
    np.testing.assert_allclose(sums, 1.0, atol=1e-4,
                               err_msg="Prob grids don't sum to 1")
    assert (proba >= 0).all(), "Negative probabilities in grid"


def test_classification_decode_errors_nonnegative():
    """Manhattan errors should be non-negative integers."""
    if not HAS_TORCH:
        print("  SKIP (no torch)")
        return
    episodes = _make_episodes(n_eps=3, T=30, deter_dim=32, grid_w=16, grid_h=16)
    deter, _, pos, groups = prepare_data(episodes)
    errors, shuffle, _, _, _ = classification_decode(
        deter, pos, groups, 16, 16, n_iters=100)
    assert (errors >= 0).all(), "Negative Manhattan errors"
    assert (shuffle >= 0).all(), "Negative shuffle errors"


def test_classification_decode_beats_shuffle():
    """Decoder on linearly-decodable data should beat shuffle baseline."""
    if not HAS_TORCH:
        print("  SKIP (no torch)")
        return
    # Use easy-to-decode data with small grid
    episodes = _make_episodes(n_eps=4, T=80, deter_dim=64, grid_w=10, grid_h=10)
    deter, _, pos, groups = prepare_data(episodes)
    errors, shuffle, _, _, _ = classification_decode(
        deter, pos, groups, 10, 10, n_iters=2000)
    assert errors.mean() < shuffle.mean(), (
        f"Decoder ({errors.mean():.2f}) should beat shuffle ({shuffle.mean():.2f})")


def test_classification_decode_pred_within_grid():
    """Predicted positions should be within the grid bounds."""
    if not HAS_TORCH:
        print("  SKIP (no torch)")
        return
    W, H = 12, 12
    episodes = _make_episodes(n_eps=3, T=30, deter_dim=32, grid_w=W, grid_h=H)
    deter, _, pos, groups = prepare_data(episodes)
    _, _, pred_xy, _, _ = classification_decode(
        deter, pos, groups, W, H, n_iters=100)
    assert (pred_xy[:, 0] >= 0).all() and (pred_xy[:, 0] < W).all(), \
        "Predicted x out of grid bounds"
    assert (pred_xy[:, 1] >= 0).all() and (pred_xy[:, 1] < H).all(), \
        "Predicted y out of grid bounds"


# ---------------------------------------------------------------------------
# plot_probmap_on_world smoke test
# ---------------------------------------------------------------------------

def test_probmap_on_world_no_crafter_graceful():
    """plot_probmap_on_world should not crash when crafter is unavailable."""
    import tempfile
    from pathlib import Path
    # Pass metadata without env_seed to trigger graceful failure path
    # (or crafter import failure)
    episodes = _make_episodes(n_eps=2, T=20, deter_dim=16, grid_w=8, grid_h=8)
    _, _, pos, groups = prepare_data(episodes)
    proba = np.random.rand(40, 8, 8).astype(np.float32)
    # Normalize each grid to sum to 1
    proba /= proba.sum(axis=(1, 2), keepdims=True)
    metadata = {'env_seed': 99999, 'area': (8, 8)}
    with tempfile.TemporaryDirectory() as tmpdir:
        # This may fail to render (no crafter or wrong seed), but should not crash
        try:
            plot_probmap_on_world(proba, pos, groups, metadata, 'test',
                                  Path(tmpdir), n_steps=4, episode=0)
        except Exception as e:
            # Acceptable: import error, render failure. Not acceptable: crash
            # from shape mismatches or logic errors
            acceptable = ('crafter', 'ModuleNotFoundError', 'render')
            if not any(kw in str(e).lower() for kw in acceptable):
                raise


# ---------------------------------------------------------------------------
# Integration: prepare_data -> classification_decode round-trip
# ---------------------------------------------------------------------------

def test_roundtrip_prepare_classify():
    """Full pipeline: make episodes -> prepare_data -> classify -> check."""
    if not HAS_TORCH:
        print("  SKIP (no torch)")
        return
    W, H = 8, 8
    episodes = _make_episodes(n_eps=3, T=50, deter_dim=32, grid_w=W, grid_h=H)
    deter, stoch, pos, groups = prepare_data(episodes)
    combined = np.concatenate([deter, stoch], axis=1)

    # Classification on combined
    errors, shuffle, pred_xy, true_xy, proba = classification_decode(
        combined, pos, groups, W, H, n_iters=500)

    # Verify all outputs are consistent
    N = deter.shape[0]
    assert errors.shape[0] == N
    assert proba.shape == (N, W, H)
    np.testing.assert_allclose(proba.sum(axis=(1, 2)), 1.0, atol=1e-4)
    assert (true_xy == pos.astype(int).clip(0, [W-1, H-1])).all(), \
        "true_xy should match clipped integer positions"


# ---------------------------------------------------------------------------
# Decoder model save/load tests
# ---------------------------------------------------------------------------

def test_save_load_classification_decoder():
    """Train classifier, save state_dict, reload, verify predictions match."""
    if not HAS_TORCH:
        print("  SKIP (no torch)")
        return
    import tempfile
    from pathlib import Path
    n_units, n_classes = 32, 64
    X = np.random.randn(100, n_units).astype(np.float32)
    y = np.random.randint(0, n_classes, 100)
    clf = LinearClassifier(n_units, n_classes)
    clf.fit(X, y, n_iters=100, verbose=False)
    pred_before = clf.predict(X[:10])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / 'clf_test.pkl'
        meta = {'n_units': n_units, 'n_classes': n_classes,
                'width': 8, 'height': 8, 'repr_name': 'deter'}
        save_classifier_model(clf, meta, path)
        loaded_clf, loaded_meta = load_classifier_model(path)

    pred_after = loaded_clf.predict(X[:10])
    np.testing.assert_array_equal(pred_before, pred_after,
                                  err_msg="Classifier predictions differ after save/load")
    assert loaded_meta['n_units'] == n_units


# ---------------------------------------------------------------------------
# Dream decode pipeline tests (synthetic, no JAX agent needed)
# ---------------------------------------------------------------------------

def test_decode_dream_classification_proba():
    """Classifier on dream deter → proba shape (N, H, W, H_grid), sums to 1."""
    if not HAS_TORCH:
        print("  SKIP (no torch)")
        return
    n_units = 32
    W, H_grid = 8, 8
    n_classes = W * H_grid
    N, H = 5, 10

    clf = LinearClassifier(n_units, n_classes)
    X_train = np.random.randn(200, n_units).astype(np.float32)
    y_train = np.random.randint(0, n_classes, 200)
    clf.fit(X_train, y_train, n_iters=50, verbose=False)

    dream_deter = np.random.randn(N, H, n_units).astype(np.float32)
    flat = dream_deter.reshape(-1, n_units)
    proba = clf.predict_proba(flat)  # (N*H, n_classes)
    proba_grid = proba.reshape(N, H, W, H_grid)

    assert proba_grid.shape == (N, H, W, H_grid)
    sums = proba_grid.sum(axis=(2, 3))
    np.testing.assert_allclose(sums, 1.0, atol=1e-4,
                               err_msg="Dream proba grids don't sum to 1")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    tests = [
        # prepare_data
        test_prepare_data_shapes,
        test_prepare_data_stoch_flattening,
        test_prepare_data_stoch_already_flat,
        test_prepare_data_groups,
        test_prepare_data_skips_missing_keys,
        test_prepare_data_length_alignment,
        # LinearClassifier
        test_classifier_predict_shape,
        test_classifier_predict_proba_shape_and_sum,
        test_classifier_can_overfit,
        # classification_decode
        test_classification_decode_shapes,
        test_classification_decode_proba_normalized,
        test_classification_decode_errors_nonnegative,
        test_classification_decode_beats_shuffle,
        test_classification_decode_pred_within_grid,
        # plotting
        test_probmap_on_world_no_crafter_graceful,
        # integration
        test_roundtrip_prepare_classify,
        # decoder model save/load
        test_save_load_classification_decoder,
        # dream decode pipeline
        test_decode_dream_classification_proba,
    ]

    passed = failed = errors = 0
    for test in tests:
        try:
            test()
            print(f"PASS: {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {test.__name__}: {type(e).__name__}: {e}")
            errors += 1

    print(f"\n{passed} passed, {failed} failed, {errors} errors "
          f"(out of {len(tests)} tests)")
    if failed or errors:
        sys.exit(1)
