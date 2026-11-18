import numpy as np

from curl.env.scoring import evaluate_order
from curl.env.sweep_env import SweepCoverageEnv
from curl.kinematics.ur10e import fk, ik_solutions, wrap

BASE = np.array([0.72, 0.40, 0.40])


def test_fk_ik_round_trip():
    rng = np.random.default_rng(3)
    q = rng.uniform(-1.0, 1.0, 6) + np.array([0, -1.2, 1.0, -1.5, -1.5, 0])
    _, tcp = fk(q)
    target = tcp[:3, 3] + BASE
    sols = ik_solutions(target, tcp[:3, 2], BASE, restarts=40, seed=1)
    assert len(sols) >= 1
    for s in sols:
        _, t2 = fk(s)
        assert np.linalg.norm(t2[:3, 3] - tcp[:3, 3]) < 1e-4
        assert np.linalg.norm(t2[:3, 2] - tcp[:3, 2]) < 1e-4


def _line_env(**kw):
    pos = np.array([[0, 0, 0], [0.2, 0, 0], [0.2, 0.2, 0], [0, 0.2, 0]], dtype=float)
    nrm = np.tile([0, 0, 1.0], (4, 1))
    return SweepCoverageEnv(pos, nrm, tool_radius=0.05, cell=0.005, **kw), pos


def test_off_surface_accounting():
    env, _ = _line_env()
    env.reset()
    env.step(0)
    env.step(1)
    assert 0.05 < env.stats["off"] < 0.15


def test_overlap_accounting():
    env, _ = _line_env()
    env.reset()
    env.step(1)
    env.step(0)
    assert env.stats["overlap"] > 0


def test_action_mask_blocks_revisits():
    env, _ = _line_env()
    env.reset()
    env.step(2)
    assert not env.action_mask()[2]
    try:
        env.step(2)
        raise AssertionError("revisit should raise")
    except ValueError:
        pass


def test_evaluate_order_scores_file_order():
    env, _ = _line_env()
    res = evaluate_order(env, list(range(4)))
    assert res["coverage_pct"] > 0
    assert res["steps"] == 4
    assert res["skipped"] == 0


def test_wrap_range():
    w = wrap(np.array([3 * np.pi, -3 * np.pi, 0.5]))
    assert np.all(w >= -np.pi - 1e-12) and np.all(w < np.pi + 1e-12)
    assert abs(w[2] - 0.5) < 1e-12
