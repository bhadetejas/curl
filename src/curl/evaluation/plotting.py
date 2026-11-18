
from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def draw_path(ax, positions, order, title: str = "", unreachable=None) -> float:
    pts = np.asarray(positions) * 1000.0
    if unreachable is not None and len(unreachable):
        bad = pts[list(unreachable)]
        ax.scatter(bad[:, 0], bad[:, 1], marker="x", s=40, color="#c44", zorder=4,
                   label="unreachable")
    seq = pts[np.asarray(order, dtype=int)]
    ax.plot(seq[:, 0], seq[:, 1], "-", color="#888", lw=1, zorder=1)
    ax.scatter(seq[:, 0], seq[:, 1], c=np.arange(len(seq)), cmap="viridis", s=18, zorder=2)
    ax.scatter(seq[0, 0], seq[0, 1], marker="o", s=120, facecolor="none",
               edgecolor="red", lw=2, zorder=3, label="start")
    ax.scatter(seq[-1, 0], seq[-1, 1], marker="s", s=120, facecolor="none",
               edgecolor="black", lw=2, zorder=3, label="end")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    length = float(np.linalg.norm(np.diff(seq, axis=0), axis=1).sum())
    ax.set_title(f"{title}\n{len(seq)} waypoints, length {length:.0f} mm")
    ax.legend(loc="upper right", fontsize=8)
    return length


def plot_path(positions, order, out_png, title: str = "Coverage path", unreachable=None) -> Path:
    fig = Figure(figsize=(10, 6))
    FigureCanvasAgg(fig)
    draw_path(fig.add_subplot(), positions, order, title, unreachable)
    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    return out_png
