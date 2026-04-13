#!/usr/bin/env python3
from __future__ import annotations

import runpy
from pathlib import Path

import handlepress_env  # noqa: F401


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    runpy.run_path(
        str(repo_root / "repos" / "robomimic" / "robomimic" / "scripts" / "train.py"),
        run_name="__main__",
    )
