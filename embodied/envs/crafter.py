import json

import crafter
import crafter.engine as crafter_engine
import elements
import embodied
import numpy as np


class Crafter(embodied.Env):

  def __init__(self, task, size=(64, 64), area=(64, 64), logs=False,
               logdir=None, seed=None, fixed_seed=False, random_spawn=False,
               egocentric_view=None):
    assert task in ('reward', 'noreward')
    self._env = crafter.Env(
        area=area, size=size, reward=(task == 'reward'), seed=seed)
    self._logs = logs
    self._logdir = logdir and elements.Path(logdir)
    self._logdir and self._logdir.mkdir()
    self._episode = 0
    self._length = None
    self._reward = None
    self._achievements = crafter.constants.achievements.copy()
    self._done = True
    self._fixed_seed = fixed_seed
    self._random_spawn = random_spawn
    self._seed = seed
    self._spawn_rng = np.random.RandomState(seed)
    # egocentric view setup
    self._pixel_size = size[0] if hasattr(size, '__len__') else size
    self._egocentric_view = egocentric_view if egocentric_view else None
    if self._egocentric_view is not None:
      assert egocentric_view % 2 == 1, 'egocentric_view must be odd'
      V = egocentric_view
      forward = V - 1   # tiles visible ahead of agent
      side = V // 2     # tiles visible to each side
      render_tiles = max(forward, side)
      self._ego_V = V
      self._ego_c = render_tiles            # center tile in the large render
      self._ego_unit = np.array([self._pixel_size // V, self._pixel_size // V])
      render_grid = 2 * render_tiles + 1
      # LocalView for the oversized centered render (terrain only, no inventory)
      self._ego_local_view = crafter_engine.LocalView(
          self._env._world, self._env._textures, [render_grid, render_grid])
      # Inventory bar: compute how many ego tiles the inventory needs,
      # so we can reduce the forward crop and keep the agent visible above it.
      import crafter as _crafter_mod
      # Standard view inventory (for pixel copy from raw_image)
      _std_item_rows = int(np.ceil(
          len(_crafter_mod.constants.items) / self._env._view[0]))
      self._ego_inv_rows = int(_std_item_rows * self._env._size[0] // self._env._view[0])
      # How many ego tiles the inventory occupies (to reduce forward crop)
      self._ego_inv_tiles = int(np.ceil(self._ego_inv_rows / int(self._ego_unit[0])))

  @property
  def obs_space(self):
    if self._egocentric_view is not None:
      u = self._ego_unit
      V = self._ego_V
      img_shape = (self._pixel_size, self._pixel_size, 3)
    else:
      img_shape = self._env.observation_space.shape
    spaces = {
        'image': elements.Space(np.uint8, img_shape),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
        'log/reward': elements.Space(np.float32),
        'player_pos': elements.Space(np.float32, (2,)),
        'log/player_facing': elements.Space(np.int32, (2,)),
    }
    # Include achievements for trajectory analysis (log/ prefix = ignored by agent)
    spaces.update({
        f'log/achievement_{k}': elements.Space(np.int32)
        for k in self._achievements})
    return spaces

  @property
  def act_space(self):
    return {
        'action': elements.Space(np.int32, (), 0, self._env.action_space.n),
        'reset': elements.Space(bool),
    }

  def step(self, action):
    if action['reset'] or self._done:
      self._episode += 1
      self._length = 0
      self._reward = 0
      self._done = False
      if self._fixed_seed:
        self._env._episode = 0
      image = self._env.reset()
      if self._random_spawn:
        self._relocate_player()
        image = self._env._obs()
      # Verify player spawned on walkable terrain
      player_pos = tuple(self._env._player.pos)
      mat, _ = self._env._world[player_pos]
      assert mat in ('grass', 'path', 'sand'), (
          f"Player spawned on non-walkable material '{mat}' at {player_pos}")
      return self._obs(image, 0.0, {}, is_first=True)
    image, reward, self._done, info = self._env.step(action['action'])
    self._reward += reward
    self._length += 1
    if self._done and self._logdir:
      self._write_stats(self._length, self._reward, info)
    return self._obs(
        image, reward, info,
        is_last=self._done,
        is_terminal=info['discount'] == 0)

  def _relocate_player(self):
    """Move player to a random walkable, unoccupied tile."""
    world = self._env._world
    player = self._env._player
    walkable_mats = ('grass', 'path', 'sand')
    mat_ids = [world._mat_ids[m] for m in walkable_mats if m in world._mat_ids]
    walkable = np.isin(world._mat_map, mat_ids) & (world._obj_map == 0)
    xs, ys = np.where(walkable)
    assert len(xs) > 0, 'No walkable tiles found for random spawn'
    idx = self._spawn_rng.randint(0, len(xs))
    new_pos = np.array([xs[idx], ys[idx]])
    world.move(player, new_pos)

  def _obs(
      self, image, reward, info,
      is_first=False, is_last=False, is_terminal=False):
    if self._egocentric_view is not None:
      image = self._render_egocentric(image)
    # Get player position and facing from internal crafter env
    player_pos = np.array(self._env._player.pos, dtype=np.float32)
    player_facing = np.array(self._env._player.facing, dtype=np.int32)
    obs = dict(
        image=image,
        reward=np.float32(reward),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
        player_pos=player_pos,
        **{'log/reward': np.float32(info['reward'] if info else 0.0)},
        **{'log/player_facing': player_facing},
    )
    # Include achievements for trajectory analysis (log/ prefix = ignored by agent)
    achievements = {
        f'log/achievement_{k}': np.int32(info['achievements'][k] if info else 0)
        for k in self._achievements}
    obs.update(achievements)
    return obs

  def _render_egocentric(self, raw_image):
    """Render egocentric observation with agent above the inventory bar.

    Renders a large centered grid (terrain only), crops asymmetrically based
    on facing direction. Forward vision is reduced by inv_tiles so the agent
    sits just above the inventory bar (matching allocentric behavior where
    inventory replaces bottom rows without covering the agent).

    Canvas coordinate convention (from crafter's LocalView — NOT transposed):
      axis-0 (rows) = world x-axis  (right/east = +row)
      axis-1 (cols) = world y-axis  (down/south = +col)

    Note: crafter's render() transposes the canvas to standard image form
    (rows=y, cols=x), but LocalView does not. We compensate with np.fliplr
    after rotation to correct the left-right mirror.

    Rotation mapping (agent ends up just above the inventory bar):
      facing left  (-1, 0) west:  crop rows backward, k=0 (forward already top)
      facing right (+1, 0) east:  crop rows forward,  k=2 (180°)
      facing up    ( 0,-1) north: crop cols backward, k=3 (90° CW)
      facing down  ( 0,+1) south: crop cols forward,  k=1 (90° CCW)
    """
    V = self._ego_V
    c = self._ego_c    # center tile index in the large render grid
    u = self._ego_unit  # 2-element array [upx, upx] required by crafter Textures
    inv_tiles = self._ego_inv_tiles
    forward = V - 1 - inv_tiles  # reduced so agent sits above inventory
    side = V // 2     # tiles to each side

    # Render oversized grid centered on the player (terrain only)
    canvas = self._ego_local_view(self._env._player, u)
    # canvas shape: (render_grid*upx, render_grid*upx, 3)

    upx = int(u[0])    # scalar pixels-per-tile for arithmetic
    cu = c * upx       # center pixel offset
    fu = forward * upx # forward span in pixels
    su = side * upx    # side span in pixels
    Vu = V * upx       # final image edge in pixels

    facing = tuple(self._env._player.facing)

    if facing == (-1, 0):    # left / west: forward = -row direction
      crop = canvas[cu - fu : cu + upx,    cu - su : cu - su + Vu]
      k = 0
    elif facing == (1, 0):   # right / east: forward = +row direction
      crop = canvas[cu       : cu + fu + upx, cu - su : cu - su + Vu]
      k = 2
    elif facing == (0, -1):  # up / north: forward = -col direction
      crop = canvas[cu - su : cu - su + Vu, cu - fu : cu + upx   ]
      k = 3
    else:                    # down / south (0, 1): forward = +col direction
      crop = canvas[cu - su : cu - su + Vu, cu       : cu + fu + upx]
      k = 1

    result = np.rot90(crop, k)
    # The LocalView canvas has axis-0=world-x, axis-1=world-y, but standard
    # images need axis-0=y, axis-1=x (crafter's render() transposes).
    # Since we rotate instead of transpose, the result is left-right mirrored.
    # A horizontal flip corrects this for all facing directions.
    result = np.fliplr(result)
    # Pad to pixel_size x pixel_size (crop may be smaller due to integer division)
    h, w = result.shape[:2]
    if h < self._pixel_size or w < self._pixel_size:
      padded = np.zeros((self._pixel_size, self._pixel_size, 3), dtype=np.uint8)
      padded[:h, :w] = result
      result = padded
    # Copy inventory bar from standard crafter image into bottom rows
    result[-self._ego_inv_rows:] = raw_image[-self._ego_inv_rows:]
    return result

  def _write_stats(self, length, reward, info):
    stats = {
        'episode': self._episode,
        'length': length,
        'reward': round(reward, 1),
        **{f'achievement_{k}': v for k, v in info['achievements'].items()},
    }
    filename = self._logdir / 'stats.jsonl'
    lines = filename.read() if filename.exists() else ''
    lines += json.dumps(stats) + '\n'
    filename.write(lines, mode='w')
    print(f'Wrote stats: {filename}')
