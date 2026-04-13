#!/usr/bin/env python3
from __future__ import annotations

import runpy
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
ROBOMIMIC_DATASET_TO_OBS = (
    REPO_ROOT
    / "repos"
    / "robomimic"
    / "robomimic"
    / "scripts"
    / "dataset_states_to_obs.py"
)

if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import handlepress_env  # noqa: F401


if __name__ == "__main__":
    runpy.run_path(str(ROBOMIMIC_DATASET_TO_OBS), run_name="__main__")
