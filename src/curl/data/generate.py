
from __future__ import annotations

import numpy as np

TABLE_Z = 0.845


def _in_rect(pts, x0, y0, w, h):
    return (pts[:, 0] >= x0) & (pts[:, 0] <= x0 + w) & (pts[:, 1] >= y0) & (pts[:, 1] <= y0 + h)


def generate_scene(rng: np.random.Generator, cutout_prob: float = 0.7):
    width = rng.uniform(0.5, 1.5)
    depth = rng.uniform(0.35, 0.6)
    row_gap = rng.uniform(0.08, 0.11)
    step = rng.uniform(0.10, 0.22)
    xs = np.arange(0.0, width + 1e-9, step)
    ys = np.arange(0.0, depth + 1e-9, row_gap)
    grid = np.array([(x, y) for y in ys for x in xs], dtype=float)
    keep = np.ones(len(grid), dtype=bool)

    if rng.random() < cutout_prob:
        for _ in range(int(rng.integers(1, 3))):
            cw = rng.uniform(0.15, 0.35) * width
            ch = rng.uniform(0.15, 0.30) * depth
            cx = rng.uniform(0.0, width - cw)
            cy = rng.uniform(0.0, depth - ch)
            keep &= ~_in_rect(grid, cx, cy, cw, ch)

    if rng.random() < 0.3:
        nw = rng.uniform(0.10, 0.25) * width
        nh = rng.uniform(0.10, 0.25) * depth
        x0 = 0.0 if rng.random() < 0.5 else width - nw
        y0 = 0.0 if rng.random() < 0.5 else depth - nh
        keep &= ~_in_rect(grid, x0, y0, nw, nh)

    n_cols = len(xs)
    strokes = []
    for r in range(len(ys)):
        for c in range(n_cols - 1):
            i, j = r * n_cols + c, r * n_cols + c + 1
            if keep[i] and keep[j]:
                strokes.append((np.append(grid[i], TABLE_Z), np.append(grid[j], TABLE_Z)))

    pts = grid[keep]
    pts = pts[rng.permutation(len(pts))]
    pos = np.column_stack([pts, np.full(len(pts), TABLE_Z)])
    nrm = np.tile([0.0, 0.0, 1.0], (len(pts), 1))
    return pos, nrm, strokes
