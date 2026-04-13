#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import h5py
import numpy as np
import zarr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a renumbered subset bundle for the Lift pipeline: demo.hdf5, "
            "low_dim.hdf5, source.zarr, and matching sam_mask directories."
        )
    )
    parser.add_argument("--demo-hdf5", required=True, help="Input robosuite demo.hdf5")
    parser.add_argument("--demo-output", required=True, help="Output subset demo.hdf5")
    parser.add_argument("--low-dim-hdf5", required=True, help="Input robomimic low_dim.hdf5")
    parser.add_argument("--low-dim-output", required=True, help="Output subset low_dim.hdf5")
    parser.add_argument("--source-zarr", required=True, help="Input DemoGen source zarr")
    parser.add_argument("--source-output-zarr", required=True, help="Output subset source zarr")
    parser.add_argument(
        "--episodes",
        required=True,
        help="Comma-separated source demos to keep, such as '2,4,6,7' or 'demo_2,demo_4'",
    )
    parser.add_argument(
        "--input-source-name",
        default=None,
        help="Input source name for sam_mask lookup. Defaults to the input zarr stem.",
    )
    parser.add_argument(
        "--output-source-name",
        default=None,
        help="Output source name for sam_mask copy. Defaults to the output zarr stem.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    return parser.parse_args()


def natural_demo_key(name: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", name)
    return (int(match.group(1)) if match else -1, name)


def sorted_demo_keys(group) -> list[str]:
    return sorted(list(group.keys()), key=natural_demo_key)


def parse_selection(raw: str, available_keys: list[str]) -> list[str]:
    selected = []
    key_by_suffix = {}
    for key in available_keys:
        match = re.search(r"(\d+)$", key)
        if match:
            key_by_suffix[int(match.group(1))] = key

    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        if token in available_keys:
            selected.append(token)
            continue
        if token.startswith("demo_") and token in available_keys:
            selected.append(token)
            continue
        try:
            idx = int(token)
        except ValueError as exc:
            raise ValueError(f"Unknown episode specifier: {token}") from exc
        if idx not in key_by_suffix:
            raise KeyError(f"Episode {idx} not found in available demos: {available_keys}")
        selected.append(key_by_suffix[idx])

    if not selected:
        raise ValueError("No episodes selected")
    if len(set(selected)) != len(selected):
        raise ValueError(f"Duplicate episodes requested: {selected}")
    return selected


def ensure_overwrite(path: Path, overwrite: bool) -> None:
    if not path.exists():
        return
    if not overwrite:
        raise FileExistsError(f"Output already exists: {path}")
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def copy_attrs(src, dst) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def copy_h5_item(src, dst_parent, name: str) -> None:
    obj = src[name]
    if isinstance(obj, h5py.Dataset):
        dst = dst_parent.create_dataset(name, data=obj[()])
        copy_attrs(obj, dst)
        return

    dst_group = dst_parent.create_group(name)
    copy_attrs(obj, dst_group)
    for child_name in obj.keys():
        copy_h5_item(obj, dst_group, child_name)


def write_subset_hdf5(input_path: Path, output_path: Path, selected_keys: list[str]) -> None:
    with h5py.File(input_path, "r") as src, h5py.File(output_path, "w") as dst:
        copy_attrs(src, dst)

        if "data" not in src:
            raise KeyError(f"{input_path} missing /data")

        src_data = src["data"]
        dst_data = dst.create_group("data")
        copy_attrs(src_data, dst_data)

        total = 0
        key_mapping: dict[str, str] = {}
        for new_idx, old_key in enumerate(selected_keys, start=1):
            new_key = f"demo_{new_idx}"
            key_mapping[old_key] = new_key
            copy_h5_item(src_data, dst_data, old_key)
            dst_data.move(old_key, new_key)
            total += int(dst_data[new_key].attrs.get("num_samples", 0))

        if "total" in dst_data.attrs:
            dst_data.attrs["total"] = total

        if "mask" in src:
            dst_mask = dst.create_group("mask")
            src_mask = src["mask"]
            copy_attrs(src_mask, dst_mask)
            for mask_name in src_mask.keys():
                raw_names = []
                for entry in src_mask[mask_name][()]:
                    if isinstance(entry, bytes):
                        raw_names.append(entry.decode("utf-8"))
                    else:
                        raw_names.append(str(entry))
                filtered = [key_mapping[name] for name in raw_names if name in key_mapping]
                dtype = h5py.string_dtype(encoding="utf-8")
                dst_mask.create_dataset(mask_name, data=np.asarray(filtered, dtype=object), dtype=dtype)


def iter_episode_bounds(episode_ends: np.ndarray) -> list[tuple[int, int]]:
    bounds = []
    start = 0
    for end in np.asarray(episode_ends, dtype=np.int64):
        end = int(end)
        bounds.append((start, end))
        start = end
    return bounds


def build_selected_frame_spans(episode_ends: np.ndarray, selected_episode_indices: list[int]) -> tuple[list[slice], np.ndarray]:
    bounds = iter_episode_bounds(episode_ends)
    spans = []
    new_episode_ends = []
    count = 0
    for ep_idx in selected_episode_indices:
        start, end = bounds[ep_idx]
        spans.append(slice(start, end))
        count += end - start
        new_episode_ends.append(count)
    return spans, np.asarray(new_episode_ends, dtype=np.int64)


def subset_meta_array(src_arr, selected_episode_indices: list[int], spans: list[slice], total_frames: int):
    data = src_arr[:]
    if data.shape == (len(selected_episode_indices),):
        return data
    if data.shape[0] == len(spans):
        return data[selected_episode_indices]
    if data.shape[0] == total_frames:
        return np.concatenate([data[span] for span in spans], axis=0)
    return data


def write_subset_zarr(input_path: Path, output_path: Path, selected_episode_indices: list[int]) -> None:
    src_root = zarr.open(str(input_path), mode="r")
    if "data" not in src_root or "meta" not in src_root:
        raise KeyError(f"{input_path} is not a valid DemoGen zarr dataset")

    src_data = src_root["data"]
    src_meta = src_root["meta"]
    if "episode_ends" not in src_meta:
        raise KeyError(f"{input_path} missing meta/episode_ends")

    old_episode_ends = np.asarray(src_meta["episode_ends"][:], dtype=np.int64)
    spans, new_episode_ends = build_selected_frame_spans(old_episode_ends, selected_episode_indices)
    total_frames = int(old_episode_ends[-1]) if len(old_episode_ends) > 0 else 0

    dst_root = zarr.group(str(output_path), overwrite=True)
    dst_data = dst_root.create_group("data")
    dst_meta = dst_root.create_group("meta")
    for key, value in src_root.attrs.items():
        dst_root.attrs[key] = value
    for key, value in src_data.attrs.items():
        dst_data.attrs[key] = value
    for key, value in src_meta.attrs.items():
        dst_meta.attrs[key] = value

    for key in src_data.keys():
        src_arr = src_data[key]
        chunk = src_arr.chunks
        compressor = src_arr.compressor
        selected = np.concatenate([src_arr[span] for span in spans], axis=0)
        dst_data.create_dataset(
            key,
            data=selected,
            chunks=chunk,
            dtype=src_arr.dtype,
            compressor=compressor,
            overwrite=True,
        )

    dst_meta.create_dataset(
        "episode_ends",
        data=new_episode_ends,
        chunks=src_meta["episode_ends"].chunks,
        dtype=src_meta["episode_ends"].dtype,
        compressor=src_meta["episode_ends"].compressor,
        overwrite=True,
    )

    for key in src_meta.keys():
        if key == "episode_ends":
            continue
        src_arr = src_meta[key]
        subset = subset_meta_array(
            src_arr=src_arr,
            selected_episode_indices=selected_episode_indices,
            spans=spans,
            total_frames=total_frames,
        )
        dst_meta.create_dataset(
            key,
            data=subset,
            chunks=src_arr.chunks,
            dtype=src_arr.dtype,
            compressor=src_arr.compressor,
            overwrite=True,
        )


def copy_subset_sam_mask(
    input_source_name: str,
    output_source_name: str,
    selected_episode_indices: list[int],
    source_output_zarr: Path,
    overwrite: bool,
) -> None:
    data_root = source_output_zarr.parents[2]
    input_root = data_root / "sam_mask" / input_source_name
    output_root = data_root / "sam_mask" / output_source_name
    if not input_root.exists():
        return

    ensure_overwrite(output_root, overwrite)
    output_root.mkdir(parents=True, exist_ok=True)
    for new_idx, old_idx in enumerate(selected_episode_indices):
        src_dir = input_root / str(old_idx)
        if not src_dir.exists():
            raise FileNotFoundError(f"Missing source sam_mask dir: {src_dir}")
        shutil.copytree(src_dir, output_root / str(new_idx))


def main() -> None:
    args = parse_args()

    demo_hdf5 = Path(args.demo_hdf5).expanduser().resolve()
    demo_output = Path(args.demo_output).expanduser().resolve()
    low_dim_hdf5 = Path(args.low_dim_hdf5).expanduser().resolve()
    low_dim_output = Path(args.low_dim_output).expanduser().resolve()
    source_zarr = Path(args.source_zarr).expanduser().resolve()
    source_output_zarr = Path(args.source_output_zarr).expanduser().resolve()

    if not demo_hdf5.exists():
        raise FileNotFoundError(f"demo.hdf5 not found: {demo_hdf5}")
    if not low_dim_hdf5.exists():
        raise FileNotFoundError(f"low_dim.hdf5 not found: {low_dim_hdf5}")
    if not source_zarr.exists():
        raise FileNotFoundError(f"source zarr not found: {source_zarr}")

    demo_output.parent.mkdir(parents=True, exist_ok=True)
    low_dim_output.parent.mkdir(parents=True, exist_ok=True)
    source_output_zarr.parent.mkdir(parents=True, exist_ok=True)

    ensure_overwrite(demo_output, args.overwrite)
    ensure_overwrite(low_dim_output, args.overwrite)
    ensure_overwrite(source_output_zarr, args.overwrite)

    with h5py.File(demo_hdf5, "r") as demo_src:
        available_keys = sorted_demo_keys(demo_src["data"])
    selected_keys = parse_selection(args.episodes, available_keys)
    selected_episode_indices = [available_keys.index(key) for key in selected_keys]

    write_subset_hdf5(demo_hdf5, demo_output, selected_keys)
    write_subset_hdf5(low_dim_hdf5, low_dim_output, selected_keys)
    write_subset_zarr(source_zarr, source_output_zarr, selected_episode_indices)

    input_source_name = args.input_source_name or source_zarr.stem
    output_source_name = args.output_source_name or source_output_zarr.stem
    copy_subset_sam_mask(
        input_source_name=input_source_name,
        output_source_name=output_source_name,
        selected_episode_indices=selected_episode_indices,
        source_output_zarr=source_output_zarr,
        overwrite=args.overwrite,
    )

    print(
        {
            "selected_keys": selected_keys,
            "selected_episode_indices": selected_episode_indices,
            "demo_output": str(demo_output),
            "low_dim_output": str(low_dim_output),
            "source_output_zarr": str(source_output_zarr),
            "output_source_name": output_source_name,
        }
    )


if __name__ == "__main__":
    main()
