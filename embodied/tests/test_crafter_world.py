"""Tests to verify Crafter world consistency across episodes and trials.

These tests ensure that:
1. The same seed produces identical worlds across resets (fixed_seed=True)
2. The player always spawns on walkable terrain
3. The world is deterministic given the same seed
"""

import numpy as np
import crafter


def _get_world_state(env):
    """Extract world material map and player position after reset."""
    return env._world._mat_map.copy(), np.array(env._player.pos)


def test_fixed_seed_same_world_across_resets():
    """With the same seed, resetting episode counter should produce identical worlds."""
    seed = 42
    area = (32, 32)
    env = crafter.Env(area=area, view=(9, 9), size=(64, 64), seed=seed)

    # First reset
    env.reset()
    mat_map_1, pos_1 = _get_world_state(env)

    # Simulate fixed_seed behavior: reset _episode to 0 before reset
    env._episode = 0
    env.reset()
    mat_map_2, pos_2 = _get_world_state(env)

    assert np.array_equal(mat_map_1, mat_map_2), (
        "Material maps differ across resets with fixed_seed behavior")
    assert np.array_equal(pos_1, pos_2), (
        "Player positions differ across resets with fixed_seed behavior")


def test_fixed_seed_same_world_many_resets():
    """World should be identical across many resets with fixed_seed."""
    seed = 123
    area = (32, 32)
    env = crafter.Env(area=area, view=(9, 9), size=(64, 64), seed=seed)

    # Get reference world
    env.reset()
    ref_mat_map, ref_pos = _get_world_state(env)

    # Reset 10 more times with fixed_seed behavior
    for i in range(10):
        env._episode = 0
        env.reset()
        mat_map, pos = _get_world_state(env)
        assert np.array_equal(mat_map, ref_mat_map), (
            f"Material map differs on reset {i+1}")
        assert np.array_equal(pos, ref_pos), (
            f"Player position differs on reset {i+1}")


def test_different_seeds_different_worlds():
    """Different seeds should produce different worlds."""
    area = (32, 32)
    env1 = crafter.Env(area=area, view=(9, 9), size=(64, 64), seed=42)
    env2 = crafter.Env(area=area, view=(9, 9), size=(64, 64), seed=99)

    env1.reset()
    env2.reset()

    mat_map_1 = env1._world._mat_map.copy()
    mat_map_2 = env2._world._mat_map.copy()

    assert not np.array_equal(mat_map_1, mat_map_2), (
        "Different seeds produced identical worlds")


def test_same_seed_separate_envs_same_world():
    """Two separate env instances with the same seed should produce identical worlds."""
    seed = 42
    area = (32, 32)

    env1 = crafter.Env(area=area, view=(9, 9), size=(64, 64), seed=seed)
    env2 = crafter.Env(area=area, view=(9, 9), size=(64, 64), seed=seed)

    env1.reset()
    env2.reset()

    mat_map_1, pos_1 = _get_world_state(env1)
    mat_map_2, pos_2 = _get_world_state(env2)

    assert np.array_equal(mat_map_1, mat_map_2), (
        "Same seed produced different material maps in separate env instances")
    assert np.array_equal(pos_1, pos_2), (
        "Same seed produced different player positions in separate env instances")


def test_player_spawns_on_walkable_terrain():
    """Player must always spawn on walkable terrain (grass/path/sand)."""
    walkable = {'grass', 'path', 'sand'}
    area = (32, 32)

    for seed in range(50):
        env = crafter.Env(area=area, view=(9, 9), size=(64, 64), seed=seed)
        env.reset()

        player_pos = tuple(env._player.pos)
        mat, _ = env._world[player_pos]
        assert mat in walkable, (
            f"Seed {seed}: player spawned on '{mat}' at {player_pos}, "
            f"expected one of {walkable}")


def test_player_spawns_on_walkable_large_world():
    """Player spawn check on larger world sizes."""
    walkable = {'grass', 'path', 'sand'}

    for area_size in [(32, 32), (64, 64)]:
        for seed in range(20):
            env = crafter.Env(area=area_size, view=(9, 9), size=(64, 64), seed=seed)
            env.reset()

            player_pos = tuple(env._player.pos)
            mat, _ = env._world[player_pos]
            assert mat in walkable, (
                f"Area {area_size}, seed {seed}: player spawned on '{mat}' "
                f"at {player_pos}")


def test_world_determinism_after_steps():
    """World material map at spawn should be the same regardless of prior steps."""
    seed = 42
    area = (32, 32)
    env = crafter.Env(area=area, view=(9, 9), size=(64, 64), seed=seed)

    # First run: just reset
    env.reset()
    ref_mat_map = env._world._mat_map.copy()

    # Second run: take some steps, then reset with fixed_seed
    env._episode = 0
    env.reset()
    for _ in range(50):
        env.step(1)  # move_left
    # Now reset again with fixed_seed
    env._episode = 0
    env.reset()
    mat_map_after_steps = env._world._mat_map.copy()

    assert np.array_equal(ref_mat_map, mat_map_after_steps), (
        "World differs after taking steps and resetting with fixed_seed")


def test_wrapper_fixed_seed():
    """Test the embodied Crafter wrapper's fixed_seed mechanism."""
    from embodied.envs.crafter import Crafter

    seed = 42
    env = Crafter('reward', size=(64, 64), area=(32, 32), seed=seed, fixed_seed=True)

    # First episode
    obs1 = env.step({'action': 0, 'reset': True})
    mat_map_1 = env._env._world._mat_map.copy()
    pos_1 = obs1['player_pos'].copy()

    # Second episode
    obs2 = env.step({'action': 0, 'reset': True})
    mat_map_2 = env._env._world._mat_map.copy()
    pos_2 = obs2['player_pos'].copy()

    # Third episode
    obs3 = env.step({'action': 0, 'reset': True})
    mat_map_3 = env._env._world._mat_map.copy()
    pos_3 = obs3['player_pos'].copy()

    assert np.array_equal(mat_map_1, mat_map_2), (
        "Wrapper: material maps differ between episode 1 and 2")
    assert np.array_equal(mat_map_2, mat_map_3), (
        "Wrapper: material maps differ between episode 2 and 3")
    assert np.array_equal(pos_1, pos_2), (
        "Wrapper: player positions differ between episode 1 and 2")
    assert np.array_equal(pos_2, pos_3), (
        "Wrapper: player positions differ between episode 2 and 3")


def test_wrapper_no_fixed_seed_different_worlds():
    """Without fixed_seed, each episode should have a different world."""
    from embodied.envs.crafter import Crafter

    seed = 42
    env = Crafter('reward', size=(64, 64), area=(32, 32), seed=seed, fixed_seed=False)

    env.step({'action': 0, 'reset': True})
    mat_map_1 = env._env._world._mat_map.copy()

    env._done = True  # Force reset
    env.step({'action': 0, 'reset': True})
    mat_map_2 = env._env._world._mat_map.copy()

    assert not np.array_equal(mat_map_1, mat_map_2), (
        "Without fixed_seed, consecutive episodes should have different worlds")


if __name__ == '__main__':
    tests = [
        test_fixed_seed_same_world_across_resets,
        test_fixed_seed_same_world_many_resets,
        test_different_seeds_different_worlds,
        test_same_seed_separate_envs_same_world,
        test_player_spawns_on_walkable_terrain,
        test_player_spawns_on_walkable_large_world,
        test_world_determinism_after_steps,
        test_wrapper_fixed_seed,
        test_wrapper_no_fixed_seed_different_worlds,
    ]
    for test in tests:
        try:
            test()
            print(f"PASS: {test.__name__}")
        except AssertionError as e:
            print(f"FAIL: {test.__name__}: {e}")
        except Exception as e:
            print(f"ERROR: {test.__name__}: {e}")
