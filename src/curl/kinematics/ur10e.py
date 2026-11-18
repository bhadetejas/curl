"""UR10e forward/inverse kinematics, reachability and a simple collision check.

DH parameters are the published UR10e values. IK is numerical (damped least squares
via scipy with random restarts) rather than the analytic 8-branch solver. Each restart
that converges gives one solution; solutions are de-duplicated.

The base frame is aligned with the world frame, with its origin at `base_pos` (metres).
The tool is a flange plus `tool_len` along the flange z axis. The base pose is an
assumption set in the config file: at floor height (z = 0) the link-clearance check
rejects every vanity waypoint, because the elbow dips below the 0.84 m tabletop.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

Q_LIMIT = 2 * np.pi
D = np.array([0.1807, 0.0, 0.0, 0.17415, 0.11985, 0.11655])
A = np.array([0.0, -0.6127, -0.57155, 0.0, 0.0, 0.0])
ALPHA = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0])
HOME = np.array([0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0])


def wrap(q: np.ndarray) -> np.ndarray:
    return (q + np.pi) % (2 * np.pi) - np.pi


def _dh_transforms(q: np.ndarray) -> np.ndarray:
    ct, st = np.cos(q), np.sin(q)
    ca, sa = np.cos(ALPHA), np.sin(ALPHA)
    T = np.zeros((6, 4, 4))
    T[:, 0, 0] = ct
    T[:, 0, 1] = -st * ca
    T[:, 0, 2] = st * sa
    T[:, 0, 3] = A * ct
    T[:, 1, 0] = st
    T[:, 1, 1] = ct * ca
    T[:, 1, 2] = -ct * sa
    T[:, 1, 3] = A * st
    T[:, 2, 1] = sa
    T[:, 2, 2] = ca
    T[:, 2, 3] = D
    T[:, 3, 3] = 1.0
    return T


def fk(q: np.ndarray, tool_len: float = 0.15):
    T = np.eye(4)
    frames = []
    for Ti in _dh_transforms(np.asarray(q, dtype=float)):
        T = T @ Ti
        frames.append(T.copy())
    tcp = frames[-1].copy()
    tcp[:3, 3] = frames[-1][:3, 3] + tool_len * frames[-1][:3, 2]
    return frames, tcp


def link_clearance(q: np.ndarray, table_z_base: float, tool_len: float = 0.15) -> float:
    """Minimum height margin of the elbow/wrist joint origins above the table, in metres.

    A negative value means a collision with the table plane.
    """
    frames, _ = fk(q, tool_len)
    zs = [f[2, 3] for f in frames[1:5]]
    return float(min(zs) - table_z_base)


def ik_solutions(target_pos, target_z, base_pos, restarts: int, tool_len: float = 0.15,
                 seed: int = 0, tol: float = 1e-4) -> np.ndarray:
    """Joint solutions placing the tool tip at `target_pos` with tool z = `target_z`.

    `target_pos` and `base_pos` are world coordinates (metres); `target_z` is the unit
    tool-z direction in the world frame. Returns an array of shape (k, 6), possibly empty.
    """
    rng = np.random.default_rng(seed)
    p = np.asarray(target_pos, dtype=float) - np.asarray(base_pos, dtype=float)
    z = np.asarray(target_z, dtype=float)

    def residual(q):
        _, tcp = fk(q, tool_len)
        return np.concatenate([tcp[:3, 3] - p, tcp[:3, 2] - z])

    sols: list[np.ndarray] = []
    for _ in range(restarts):
        q0 = rng.uniform(-np.pi, np.pi, 6)
        res = least_squares(residual, q0, bounds=(-Q_LIMIT, Q_LIMIT), method="trf",
                            xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=300)
        if np.linalg.norm(res.fun) > tol:
            continue
        q = wrap(res.x)
        if all(np.abs(wrap(q - s)).max() > 1e-3 for s in sols):
            sols.append(q)
    return np.array(sols, dtype=float).reshape(-1, 6)


def feasible_solutions(target_pos, target_z, table_z, base_pos, restarts: int,
                       tool_len: float = 0.15, margin: float = 0.01, seed: int = 0) -> np.ndarray:
    """IK solutions that also pass the link-clearance check."""
    sols = ik_solutions(target_pos, target_z, base_pos, restarts, tool_len, seed=seed)
    table_z_base = table_z - base_pos[2]
    keep = [q for q in sols if link_clearance(q, table_z_base, tool_len) >= margin]
    return np.array(keep, dtype=float).reshape(-1, 6)


def waypoint_solutions(positions, normals, base_pos, table_z, restarts: int):
    """Feasible joint solutions for every waypoint, seeded by waypoint index."""
    return [feasible_solutions(p, -z, table_z, base_pos, restarts, seed=i)
            for i, (p, z) in enumerate(zip(positions, normals))]
