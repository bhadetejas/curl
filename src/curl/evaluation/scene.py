"""Scene env setup with IK reachability."""
from __future__ import annotations

import time
from typing import Callable

import numpy as np

from curl.config import Config
from curl.env.factory import make_env, stroke_segments
from curl.env.sweep_env import SweepCoverageEnv
from curl.kinematics.ur10e import waypoint_solutions

Logger = Callable[[str], None]


def build_scene_env(positions, normals, cfg: Config, *, table_z: float, use_ik: bool = True,
                    full_visit: bool = False, forbid_off_surface: bool = False,
                    log: Logger | None = None) -> SweepCoverageEnv:
    """Env for a fixed scene. IK reachability uses the base pose and restarts from `cfg.ik`."""
    ik = None
    if use_ik:
        t0 = time.perf_counter()
        ik = waypoint_solutions(positions, normals, np.asarray(cfg.ik.base_pos_m), table_z,
                                cfg.ik.restarts)
        if log:
            n_ok = sum(len(s) > 0 for s in ik)
            log(f"IK: {n_ok}/{len(ik)} waypoints reachable and collision-free "
                f"({time.perf_counter() - t0:.1f}s)")
    env = make_env(positions, normals, cfg.env,
                   strokes=stroke_segments(positions, cfg.env.stroke_max_len_m),
                   ik_solutions=ik, full_visit=full_visit,
                   forbid_off_surface=forbid_off_surface)
    if log:
        log(f"grid {env.shape}, target cells T={env.T}")
    return env
