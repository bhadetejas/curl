"""Greedy validation on generated scenes and on the vanity top."""
from __future__ import annotations

import numpy as np
import torch

from curl.config import Config, EnvConfig
from curl.data.vanity import load_vanity
from curl.env.factory import make_full_visit_env, stroke_segments
from curl.models.pointer import PointerPolicy
from curl.training.rollout import rollout_batch


def load_vanity_scene(cfg: Config):
    """The vanity CSV as a validation scene: (positions, normals, stroke legs)."""
    positions, normals, _ = load_vanity(cfg.paths.input_csv)
    return positions, normals, stroke_segments(positions, cfg.env.stroke_max_len_m)


def validate(policy: PointerPolicy, scenes, env_cfg: EnvConfig, device) -> dict:
    """Mean greedy coverage (%), overlap (%), travel (m) and reward over fixed scenes."""
    rows = []
    with torch.no_grad():
        for pos, nrm, strokes in scenes:
            env = make_full_visit_env(pos, nrm, env_cfg, strokes)
            result = rollout_batch(policy, [env], device, greedy=True)
            rows.append((env.coverage_fraction(), env.stats["overlap"] / max(env.T, 1),
                         env.stats["travel"], result.returns[0]))
    cov, ov, trav, rew = np.mean(rows, axis=0)
    return dict(coverage=100 * cov, overlap=100 * ov, travel=trav, reward=rew)


def validate_vanity(policy: PointerPolicy, scene, env_cfg: EnvConfig, device) -> dict:
    """Greedy full-visit rollout on the vanity scene: coverage, steps, stuck and reward."""
    positions, normals, strokes = scene
    env = make_full_visit_env(positions, normals, env_cfg, strokes)
    with torch.no_grad():
        result = rollout_batch(policy, [env], device, greedy=True)
    return dict(coverage=100 * env.coverage_fraction(), steps=env.steps,
                stuck=bool(env.stats["stuck"]), reward=result.returns[0])
