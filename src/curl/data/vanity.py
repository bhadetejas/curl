"""Loader for the vanity-top waypoint CSV.

Columns are x, y, z (mm) and the unit normal nx, ny, nz. Consecutive duplicate rows are
stroke endpoints, so they collapse into one node: the raw rows become about 101 nodes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_vanity(path: str | Path, unit_scale: float = 1e-3):
    """Return (positions [m], normals, raw_rows) with consecutive duplicates removed."""
    raw = np.loadtxt(path, delimiter=",")
    positions = raw[:, :3] * unit_scale
    normals = raw[:, 3:6] / np.linalg.norm(raw[:, 3:6], axis=1, keepdims=True)

    keep = np.ones(len(raw), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(positions, axis=0), axis=1) > 1e-9
    return positions[keep], normals[keep], raw
