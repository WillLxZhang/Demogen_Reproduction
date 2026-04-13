#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zarr


@dataclass
class BatchInfo:
    path: Path
    json_path: Path | None
    root: zarr.hierarchy.Group
    data_keys: list[str]
    meta_keys: list[str]
    episode_ends: np.ndarray
    episode_bounds: list[tuple[int, int]]
    n_episodes: int
    total_frames: int
    episode_global_indices: np.ndarray | None
    episode_summaries: list[dict[str, Any]] | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge multiple DemoGen batch zarr datasets into one ordered dataset."
    )
    parser.add_argument("--inputs", nargs="+", required=True, help="Input DemoGen .zarr paths")
    parser.add_argument(
        "--input-jsons",
        nargs="*",
        default=None,
        help="Optional batch summary json paths aligned with --inputs",
    )
    parser.add_argument("--output-zarr", required=True, help="Merged DemoGen .zarr output path")
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional merged summary json path",
    )
    return parser.parse_args()


def _json_equal(lhs: Any, rhs: Any) -> bool:
    return json.dumps(lhs, sort_keys=True, ensure_ascii=True) == json.dumps(
        rhs,
        sort_keys=True,
        ensure_ascii=True,
    )


def _episode_bounds(episode_ends: np.ndarray) -> list[tuple[int, int]]:
    bounds: list[tuple[int, int]] = []
    start = 0
    for raw_end in np.asarray(episode_ends, dtype=np.int64):
        end = int(raw_end)
        bounds.append((start, end))
        start = end
    return bounds


def _load_batch_summary(
    json_path: Path | None,
    n_episodes: int,
) -> tuple[np.ndarray | None, list[dict[str, Any]] | None]:
    if json_path is None:
        return None, None

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    episodes = payload.get("episodes")
    if not isinstance(episodes, list):
        return None, None
    if len(episodes) != n_episodes:
        raise ValueError(
            f"{json_path} has {len(episodes)} summaries, expected {n_episodes}"
        )
    if not all(isinstance(ep, dict) and "generated_episode_idx" in ep for ep in episodes):
        return None, episodes

    generated_episode_idx = np.asarray(
        [int(ep["generated_episode_idx"]) for ep in episodes],
        dtype=np.int64,
    )
    return generated_episode_idx, episodes


def load_batch(input_path: Path, json_path: Path | None) -> BatchInfo:
    root = zarr.open(str(input_path), mode="r")
    if "data" not in root or "meta" not in root:
        raise KeyError(f"{input_path} is not a valid DemoGen zarr dataset")

    meta = root["meta"]
    if "episode_ends" not in meta:
        raise KeyError(f"{input_path} is missing meta/episode_ends")

    episode_ends = np.asarray(meta["episode_ends"][:], dtype=np.int64)
    n_episodes = int(len(episode_ends))
    total_frames = int(episode_ends[-1]) if n_episodes > 0 else 0
    episode_bounds = _episode_bounds(episode_ends)
    episode_global_indices, episode_summaries = _load_batch_summary(json_path, n_episodes)

    return BatchInfo(
        path=input_path,
        json_path=json_path,
        root=root,
        data_keys=sorted(root["data"].keys()),
        meta_keys=sorted(root["meta"].keys()),
        episode_ends=episode_ends,
        episode_bounds=episode_bounds,
        n_episodes=n_episodes,
        total_frames=total_frames,
        episode_global_indices=episode_global_indices,
        episode_summaries=episode_summaries,
    )


def _infer_meta_kind(arr: zarr.Array, n_episodes: int, total_frames: int) -> str:
    if arr.ndim >= 1 and arr.shape[0] == n_episodes:
        return "episode"
    if arr.ndim >= 1 and arr.shape[0] == total_frames:
        return "frame"
    return "constant"


def _validate_batch_schema(batches: list[BatchInfo]) -> dict[str, str]:
    first = batches[0]
    if not batches:
        raise ValueError("No input batches provided")

    for batch in batches[1:]:
        if batch.data_keys != first.data_keys:
            raise ValueError(
                f"Data keys mismatch: {batch.path} has {batch.data_keys}, expected {first.data_keys}"
            )
        if batch.meta_keys != first.meta_keys:
            raise ValueError(
                f"Meta keys mismatch: {batch.path} has {batch.meta_keys}, expected {first.meta_keys}"
            )

    meta_kinds: dict[str, str] = {}
    for key in first.meta_keys:
        if key == "episode_ends":
            continue
        arr = first.root["meta"][key]
        kind = _infer_meta_kind(arr, first.n_episodes, first.total_frames)
        meta_kinds[key] = kind

    for key in first.data_keys:
        base = first.root["data"][key]
        for batch in batches[1:]:
            arr = batch.root["data"][key]
            if arr.ndim != base.ndim or arr.shape[1:] != base.shape[1:] or arr.dtype != base.dtype:
                raise ValueError(
                    f"Data schema mismatch for {key}: {batch.path} has shape {arr.shape}, dtype {arr.dtype}; "
                    f"expected trailing shape {base.shape[1:]}, dtype {base.dtype}"
                )

    for key, kind in meta_kinds.items():
        base = first.root["meta"][key]
        for batch in batches[1:]:
            arr = batch.root["meta"][key]
            batch_kind = _infer_meta_kind(arr, batch.n_episodes, batch.total_frames)
            if batch_kind != kind:
                raise ValueError(
                    f"Meta array kind mismatch for {key}: {batch.path} inferred {batch_kind}, expected {kind}"
                )
            if kind == "constant":
                if arr.shape != base.shape or arr.dtype != base.dtype:
                    raise ValueError(
                        f"Constant meta schema mismatch for {key}: {batch.path} has shape {arr.shape}, dtype {arr.dtype}; "
                        f"expected shape {base.shape}, dtype {base.dtype}"
                    )
                if not np.array_equal(np.asarray(arr[:]), np.asarray(base[:])):
                    raise ValueError(f"Constant meta values differ for key {key}")
            else:
                if arr.ndim != base.ndim or arr.shape[1:] != base.shape[1:] or arr.dtype != base.dtype:
                    raise ValueError(
                        f"Meta schema mismatch for {key}: {batch.path} has shape {arr.shape}, dtype {arr.dtype}; "
                        f"expected trailing shape {base.shape[1:]}, dtype {base.dtype}"
                    )

    for attr_owner in ("root", "data", "meta"):
        base_attrs = dict(
            first.root.attrs
            if attr_owner == "root"
            else first.root[attr_owner].attrs
        )
        for batch in batches[1:]:
            attrs = dict(batch.root.attrs if attr_owner == "root" else batch.root[attr_owner].attrs)
            if not _json_equal(attrs, base_attrs):
                raise ValueError(
                    f"{attr_owner} attrs mismatch between {first.path} and {batch.path}"
                )

    return meta_kinds


def _build_episode_entries(batches: list[BatchInfo]) -> list[dict[str, Any]]:
    require_global = any(batch.episode_global_indices is not None for batch in batches)
    if require_global and not all(batch.episode_global_indices is not None for batch in batches):
        raise ValueError("Either provide batch jsons for all inputs or for none of them")

    entries: list[dict[str, Any]] = []
    next_index = 0
    for batch in batches:
        for local_ep_idx, (start, end) in enumerate(batch.episode_bounds):
            global_idx = (
                int(batch.episode_global_indices[local_ep_idx])
                if batch.episode_global_indices is not None
                else next_index
            )
            summary = (
                batch.episode_summaries[local_ep_idx]
                if batch.episode_summaries is not None
                else None
            )
            entries.append(
                {
                    "batch": batch,
                    "local_ep_idx": local_ep_idx,
                    "start": start,
                    "end": end,
                    "global_idx": global_idx,
                    "summary": summary,
                }
            )
            next_index += 1

    entries.sort(key=lambda item: item["global_idx"])
    global_indices = [int(item["global_idx"]) for item in entries]
    if len(set(global_indices)) != len(global_indices):
        raise ValueError(f"Duplicate generated episode indices detected: {global_indices}")
    if global_indices and global_indices != list(range(global_indices[0], global_indices[0] + len(global_indices))):
        raise ValueError(f"Generated episode indices are not contiguous: {global_indices}")

    return entries


def _remove_existing_output(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _merge_arrays(
    batches: list[BatchInfo],
    meta_kinds: dict[str, str],
    entries: list[dict[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    first = batches[0]
    merged_data: dict[str, np.ndarray] = {}
    merged_meta: dict[str, np.ndarray] = {}

    episode_lengths: list[int] = []
    for key in first.data_keys:
        pieces = []
        for entry in entries:
            batch = entry["batch"]
            start = entry["start"]
            end = entry["end"]
            pieces.append(np.asarray(batch.root["data"][key][start:end]))
            if key == first.data_keys[0]:
                episode_lengths.append(end - start)
        merged_data[key] = np.concatenate(pieces, axis=0) if pieces else np.zeros((0,), dtype=first.root["data"][key].dtype)

    for key, kind in meta_kinds.items():
        first_arr = first.root["meta"][key]
        if kind == "episode":
            pieces = [
                np.asarray(entry["batch"].root["meta"][key][entry["local_ep_idx"] : entry["local_ep_idx"] + 1])
                for entry in entries
            ]
            merged_meta[key] = np.concatenate(pieces, axis=0) if pieces else np.zeros((0,), dtype=first_arr.dtype)
        elif kind == "frame":
            pieces = [
                np.asarray(entry["batch"].root["meta"][key][entry["start"] : entry["end"]])
                for entry in entries
            ]
            merged_meta[key] = np.concatenate(pieces, axis=0) if pieces else np.zeros((0,), dtype=first_arr.dtype)
        else:
            merged_meta[key] = np.asarray(first_arr[:])

    episode_ends = np.cumsum(np.asarray(episode_lengths, dtype=np.int64), dtype=np.int64)
    merged_meta["episode_ends"] = episode_ends
    merged_meta["generated_episode_idx"] = np.asarray(
        [int(entry["global_idx"]) for entry in entries],
        dtype=np.int64,
    )
    return merged_data, merged_meta, episode_ends


def _write_zarr(
    output_zarr: Path,
    batches: list[BatchInfo],
    merged_data: dict[str, np.ndarray],
    merged_meta: dict[str, np.ndarray],
) -> None:
    first = batches[0]
    _remove_existing_output(output_zarr)
    root = zarr.group(str(output_zarr), overwrite=True)
    data_group = root.create_group("data")
    meta_group = root.create_group("meta")

    for key, value in first.root.attrs.items():
        root.attrs[key] = value
    for key, value in first.root["data"].attrs.items():
        data_group.attrs[key] = value
    for key, value in first.root["meta"].attrs.items():
        meta_group.attrs[key] = value

    for key in first.data_keys:
        src_arr = first.root["data"][key]
        data_group.create_dataset(
            key,
            data=merged_data[key],
            chunks=src_arr.chunks,
            dtype=src_arr.dtype,
            compressor=src_arr.compressor,
            overwrite=True,
        )

    for key in first.meta_keys:
        src_arr = first.root["meta"][key]
        meta_group.create_dataset(
            key,
            data=merged_meta[key],
            chunks=src_arr.chunks,
            dtype=src_arr.dtype,
            compressor=src_arr.compressor,
            overwrite=True,
        )

    src_episode_ends = first.root["meta"]["episode_ends"]
    meta_group.create_dataset(
        "generated_episode_idx",
        data=merged_meta["generated_episode_idx"],
        chunks=src_episode_ends.chunks,
        dtype="int64",
        compressor=src_episode_ends.compressor,
        overwrite=True,
    )


def _write_hdf5(output_zarr: Path, merged_data: dict[str, np.ndarray], episode_ends: np.ndarray) -> Path:
    import h5py

    output_hdf5 = output_zarr.with_suffix(".hdf5")
    _remove_existing_output(output_hdf5)
    with h5py.File(output_hdf5, "w") as f:
        for key, value in merged_data.items():
            f.create_dataset(key, data=value, compression="gzip")
        f.create_dataset("episode_ends", data=episode_ends, compression="gzip")
    return output_hdf5


def _build_output_summary(
    output_zarr: Path,
    output_hdf5: Path,
    batches: list[BatchInfo],
    entries: list[dict[str, Any]],
    merged_data: dict[str, np.ndarray],
    merged_meta: dict[str, np.ndarray],
) -> dict[str, Any]:
    summaries = []
    motion1_rel = []
    motion2_rel = []
    for merged_ep_idx, entry in enumerate(entries):
        summary = entry["summary"]
        if summary is None:
            continue
        merged_summary = dict(summary)
        merged_summary["merged_episode_idx"] = merged_ep_idx
        merged_summary["source_batch_path"] = str(entry["batch"].path)
        merged_summary["source_batch_episode_idx"] = int(entry["local_ep_idx"])
        summaries.append(merged_summary)
        motion1 = merged_summary.get("motion1", {})
        motion2 = merged_summary.get("motion2", {})
        if "rel_final_err_norm" in motion1:
            motion1_rel.append(float(motion1["rel_final_err_norm"]))
        if "rel_final_err_norm" in motion2:
            motion2_rel.append(float(motion2["rel_final_err_norm"]))

    aggregate: dict[str, Any] = {
        "n_inputs": len(batches),
        "n_episodes": int(len(merged_meta["generated_episode_idx"])),
        "selected_episodes": merged_meta["generated_episode_idx"].astype(np.int64).tolist(),
        "total_frames": int(merged_meta["episode_ends"][-1]) if len(merged_meta["episode_ends"]) else 0,
        "data_keys": sorted(merged_data.keys()),
        "meta_keys": sorted(merged_meta.keys()),
    }
    if motion1_rel:
        aggregate["motion1_rel_final_err_norm_mean"] = float(np.mean(motion1_rel))
        aggregate["motion1_rel_final_err_norm_max"] = float(np.max(motion1_rel))
    if motion2_rel:
        aggregate["motion2_rel_final_err_norm_mean"] = float(np.mean(motion2_rel))
        aggregate["motion2_rel_final_err_norm_max"] = float(np.max(motion2_rel))

    result: dict[str, Any] = {
        "inputs": [str(batch.path) for batch in batches],
        "input_jsons": [None if batch.json_path is None else str(batch.json_path) for batch in batches],
        "output_zarr": str(output_zarr),
        "output_hdf5": str(output_hdf5),
        "aggregate": aggregate,
    }
    if summaries:
        result["episodes"] = summaries
    return result


def main() -> None:
    args = parse_args()
    input_paths = [Path(path).resolve() for path in args.inputs]
    json_paths = None
    if args.input_jsons is not None:
        if len(args.input_jsons) != len(input_paths):
            raise ValueError("--input-jsons must have the same length as --inputs")
        json_paths = [Path(path).resolve() for path in args.input_jsons]
    else:
        json_paths = [None] * len(input_paths)

    batches = [load_batch(path, json_path) for path, json_path in zip(input_paths, json_paths)]
    meta_kinds = _validate_batch_schema(batches)
    entries = _build_episode_entries(batches)
    merged_data, merged_meta, episode_ends = _merge_arrays(batches, meta_kinds, entries)

    output_zarr = Path(args.output_zarr).resolve()
    _write_zarr(output_zarr, batches, merged_data, merged_meta)
    output_hdf5 = _write_hdf5(output_zarr, merged_data, episode_ends)
    result = _build_output_summary(
        output_zarr=output_zarr,
        output_hdf5=output_hdf5,
        batches=batches,
        entries=entries,
        merged_data=merged_data,
        merged_meta=merged_meta,
    )

    if args.output_json:
        output_json = Path(args.output_json).resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
