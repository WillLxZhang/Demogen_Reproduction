#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import h5py


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy a subset of /data/demo_* groups from a robomimic-style HDF5 into "
            "a new file, preserving top-level attrs for reproducible dataset forks."
        )
    )
    parser.add_argument("--input-hdf5", required=True)
    parser.add_argument("--output-hdf5", required=True)
    parser.add_argument(
        "--episodes",
        required=True,
        help="Comma-separated episode indices or demo keys, for example '0,1,2,3' or 'demo_1,demo_2'.",
    )
    parser.add_argument("--rename-sequential", action="store_true", help="Rename copied groups to demo_0..demo_N")
    return parser.parse_args()


def numeric_suffix(name: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", name)
    if match:
        return int(match.group(1)), name
    return math.inf, name


def sorted_demo_keys(data_group: h5py.Group) -> list[str]:
    return sorted(list(data_group.keys()), key=numeric_suffix)


def parse_episode_selection(raw: str, demo_keys: list[str]) -> list[str]:
    selected: list[str] = []
    for token in [part.strip() for part in raw.split(",") if part.strip()]:
        if token in demo_keys:
            selected.append(token)
            continue
        idx = int(token)
        if idx < 0 or idx >= len(demo_keys):
            raise IndexError(f"episode {idx} out of range for demos={demo_keys}")
        selected.append(demo_keys[idx])
    if not selected:
        raise ValueError("No episodes selected")
    return selected


def main() -> None:
    args = parse_args()
    input_hdf5 = Path(args.input_hdf5).expanduser().resolve()
    output_hdf5 = Path(args.output_hdf5).expanduser().resolve()
    output_hdf5.parent.mkdir(parents=True, exist_ok=True)
    if output_hdf5.exists():
        raise FileExistsError(f"Output already exists: {output_hdf5}")

    with h5py.File(input_hdf5, "r") as src_f, h5py.File(output_hdf5, "w") as dst_f:
        if "data" not in src_f:
            raise KeyError(f"{input_hdf5} missing /data group")
        src_data = src_f["data"]
        src_keys = sorted_demo_keys(src_data)
        selected = parse_episode_selection(args.episodes, src_keys)

        dst_data = dst_f.create_group("data")
        for key, value in src_data.attrs.items():
            dst_data.attrs[key] = value
        dst_data.attrs["source_hdf5"] = str(input_hdf5)
        dst_data.attrs["selected_demo_keys"] = json.dumps(selected)

        for out_idx, src_key in enumerate(selected):
            dst_key = f"demo_{out_idx}" if args.rename_sequential else src_key
            src_f.copy(src_data[src_key], dst_data, name=dst_key)

        print(
            json.dumps(
                {
                    "input_hdf5": str(input_hdf5),
                    "output_hdf5": str(output_hdf5),
                    "selected_demo_keys": selected,
                    "rename_sequential": bool(args.rename_sequential),
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
