#!/usr/bin/env python3
"""Generate and save a Crafter world map image for a given seed.

Usage:
    python generate_world.py --seed 0
    python generate_world.py --seed 0 --area 32 32 --tile_size 16
    python generate_world.py --seed 0 --env_index 0 --save_dir ./worlds

The seed flow matches training:
    env_seed = hash((config_seed, env_index)) % (2**32 - 1)
    Crafter world seed = hash((env_seed, episode_number=0))

With fixed_seed=True (default for eval), episode_number is always 0,
so config_seed fully determines the world.
"""

import argparse
import os

import crafter
import numpy as np
from PIL import Image


def compute_env_seed(config_seed, env_index=0):
    return hash((config_seed, env_index)) % (2 ** 32 - 1)


def render_world(env_seed, area=(32, 32), tile_size=8):
    env = crafter.Env(area=area, view=(9, 9), size=(64, 64), seed=env_seed)
    env.reset()

    world = env._world
    textures = env._textures
    mat_map = world._mat_map
    mat_names = world._mat_names

    world_img = np.zeros(
        (mat_map.shape[0] * tile_size, mat_map.shape[1] * tile_size, 3),
        dtype=np.uint8,
    )
    for x in range(mat_map.shape[0]):
        for y in range(mat_map.shape[1]):
            mat_id = mat_map[x, y]
            mat_name = mat_names.get(mat_id, 'unknown')
            if mat_name:
                texture = textures.get(mat_name, (tile_size, tile_size))
                if texture.shape[-1] == 4:
                    texture = texture[..., :3]
                px, py = x * tile_size, y * tile_size
                world_img[px:px + tile_size, py:py + tile_size] = texture

    for obj in world.objects:
        texture = textures.get(obj.texture, (tile_size, tile_size))
        if texture.shape[-1] == 4:
            alpha = texture[..., 3:].astype(np.float32) / 255
            rgb = texture[..., :3].astype(np.float32)
            px = int(obj.pos[0]) * tile_size
            py = int(obj.pos[1]) * tile_size
            if (0 <= px < world_img.shape[0] - tile_size
                    and 0 <= py < world_img.shape[1] - tile_size):
                current = world_img[px:px + tile_size, py:py + tile_size].astype(np.float32)
                blended = alpha * rgb + (1 - alpha) * current
                world_img[px:px + tile_size, py:py + tile_size] = blended.astype(np.uint8)

    # Match plot_trajectories.py: transpose + flip to standard image orientation
    world_img = world_img.transpose(1, 0, 2)[::-1]
    return world_img


def main():
    parser = argparse.ArgumentParser(description='Generate a Crafter world map image')
    parser.add_argument('--seed', type=int, required=True, help='Config seed (as passed to training)')
    parser.add_argument('--env_index', type=int, default=0, help='Env index (default: 0)')
    parser.add_argument('--area', type=int, nargs=2, default=[32, 32], help='World area (default: 32 32)')
    parser.add_argument('--tile_size', type=int, default=16, help='Pixels per tile (default: 16)')
    parser.add_argument('--save_dir', type=str, default='./worlds', help='Output directory (default: ./worlds)')
    args = parser.parse_args()

    env_seed = compute_env_seed(args.seed, args.env_index)
    print(f"Config seed: {args.seed}, env_index: {args.env_index} -> env_seed: {env_seed}")
    print(f"Area: {args.area}, tile_size: {args.tile_size}")

    world_img = render_world(env_seed, area=tuple(args.area), tile_size=args.tile_size)

    os.makedirs(args.save_dir, exist_ok=True)
    filename = f"world_seed{args.seed}.png"
    save_path = os.path.join(args.save_dir, filename)
    Image.fromarray(world_img).save(save_path)
    print(f"Saved world image to {save_path} ({world_img.shape[1]}x{world_img.shape[0]}px)")


if __name__ == '__main__':
    main()
