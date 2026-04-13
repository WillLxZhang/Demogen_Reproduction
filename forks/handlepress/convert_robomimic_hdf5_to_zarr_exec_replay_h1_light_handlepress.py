#!/usr/bin/env python3
from __future__ import annotations

import runpy
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
REAL_WORLD_DIR = REPO_ROOT / "repos" / "DemoGen" / "real_world"
SOURCE_CONVERTER = REAL_WORLD_DIR / "convert_robomimic_hdf5_to_zarr_exec_replay_h1_light.py"

for path in (THIS_DIR, REAL_WORLD_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import handlepress_env  # noqa: F401


if __name__ == "__main__":
    runpy.run_path(str(SOURCE_CONVERTER), run_name="__main__")
