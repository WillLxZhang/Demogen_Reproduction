#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import zarr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace one episode inside a concatenated DemoGen zarr dataset with "
            "the single episode stored in another zarr."
        )
    )
    parser.add_argument("--base-zarr", required=True)
    parser.add_argument("--patch-zarr", required=True)
    parser.add_argument("--target-episode", type=int, required=True)
    parser.add_argument("--output-zarr", required=True)
    return parser.parse_args()


def bounds_from_episode_ends(episode_ends: np.ndarray) -> list[tuple[int, int]]:
    bounds = []
    start = 0
    for end in np.asarray(episode_ends, dtype=np.int64):
        end = int(end)
        bounds.append((start, end))
        start = end
    return bounds


def create_dataset(group, key: str, data: np.ndarray, compressor) -> None:
    data = np.asarray(data)
    if data.ndim == 1:
        chunks = (min(100, data.shape[0]),)
    elif data.ndim == 2:
        chunks = (min(100, data.shape[0]), data.shape[1])
    elif data.ndim == 3:
        chunks = (min(100, data.shape[0]), data.shape[1], data.shape[2])
    else:
        raise ValueError(f"Unsupported ndim={data.ndim} for key={key}")

    group.create_dataset(
        key,
        data=data,
        chunks=chunks,
        dtype=str(data.dtype),
        overwrite=True,
        compressor=compressor,
    )


def main() -> None:
    args = parse_args()
    base_zarr = Path(args.base_zarr).expanduser().resolve()
    patch_zarr = Path(args.patch_zarr).expanduser().resolve()
    output_zarr = Path(args.output_zarr).expanduser().resolve()

    base_root = zarr.open(str(base_zarr), mode="r")
    patch_root = zarr.open(str(patch_zarr), mode="r")

    base_data = base_root["data"]
    base_meta = base_root["meta"]
    patch_data = patch_root["data"]
    patch_meta = patch_root["meta"]

    base_episode_ends = np.asarray(base_meta["episode_ends"][:], dtype=np.int64)
    if args.target_episode < 0 or args.target_episode >= len(base_episode_ends):
        raise IndexError(
            f"target episode {args.target_episode} out of range for {len(base_episode_ends)} episodes"
        )

    patch_episode_ends = np.asarray(patch_meta["episode_ends"][:], dtype=np.int64)
    if len(patch_episode_ends) != 1:
        raise ValueError("patch zarr must contain exactly one episode")

    base_bounds = bounds_from_episode_ends(base_episode_ends)
    patch_start, patch_end = bounds_from_episode_ends(patch_episode_ends)[0]
    base_start, base_end = base_bounds[args.target_episode]

    base_lengths = np.diff(np.concatenate([[0], base_episode_ends])).astype(np.int64)
    patch_len = int(patch_end - patch_start)
    new_lengths = base_lengths.copy()
    new_lengths[args.target_episode] = patch_len
    new_episode_ends = np.cumsum(new_lengths, dtype=np.int64)

    if output_zarr.exists():
        import shutil

        shutil.rmtree(output_zarr)

    out_root = zarr.group(str(output_zarr), overwrite=True)
    out_data = out_root.create_group("data")
    out_meta = out_root.create_group("meta")
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)

    for key in base_data.keys():
        base_arr = np.asarray(base_data[key], dtype=np.float32)
        patch_arr = np.asarray(patch_data[key][patch_start:patch_end], dtype=np.float32)
        new_arr = np.concatenate(
            [base_arr[:base_start], patch_arr, base_arr[base_end:]],
            axis=0,
        )
        create_dataset(out_data, key, new_arr, compressor)

    create_dataset(out_meta, "episode_ends", new_episode_ends, compressor)

    n_episodes = len(base_episode_ends)
    base_total_frames = int(base_episode_ends[-1])
    patch_total_frames = int(patch_episode_ends[-1])
    new_total_frames = int(new_episode_ends[-1])

    for key in base_meta.keys():
        if key == "episode_ends":
            continue
        base_arr = np.asarray(base_meta[key])

        if base_arr.ndim >= 1 and base_arr.shape[0] == n_episodes:
            new_arr = base_arr.copy()
            if key in patch_meta:
                patch_arr = np.asarray(patch_meta[key])
                if patch_arr.ndim >= 1 and patch_arr.shape[0] == 1:
                    new_arr[args.target_episode] = patch_arr[0]
            create_dataset(out_meta, key, new_arr, compressor)
        elif base_arr.ndim >= 1 and base_arr.shape[0] == base_total_frames:
            if key not in patch_meta:
                new_arr = np.concatenate(
                    [base_arr[:base_start], base_arr[base_end:]],
                    axis=0,
                )
            else:
                patch_arr = np.asarray(patch_meta[key])
                if patch_arr.ndim >= 1 and patch_arr.shape[0] == patch_total_frames:
                    new_arr = np.concatenate(
                        [base_arr[:base_start], patch_arr[patch_start:patch_end], base_arr[base_end:]],
                        axis=0,
                    )
                else:
                    raise ValueError(
                        f"Unsupported patch meta shape for key={key}: {patch_arr.shape}"
                    )
            if new_arr.shape[0] != new_total_frames:
                raise ValueError(f"Length mismatch after splicing key={key}")
            create_dataset(out_meta, key, new_arr, compressor)
        else:
            create_dataset(out_meta, key, base_arr, compressor)

    for key, value in dict(base_data.attrs).items():
        out_data.attrs[key] = value
    for key, value in dict(base_meta.attrs).items():
        out_meta.attrs[key] = value
    out_meta.attrs["patched_from_base_zarr"] = str(base_zarr)
    out_meta.attrs["patched_from_patch_zarr"] = str(patch_zarr)
    out_meta.attrs["patched_episode_idx"] = int(args.target_episode)

    print(f"Saved patched zarr to: {output_zarr}")
    print(f"Replaced episode {args.target_episode} ({base_end - base_start} -> {patch_len} frames)")


if __name__ == "__main__":
    main()
