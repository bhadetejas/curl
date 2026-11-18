import numpy as np

from curl.data.generate import generate_scene
from curl.env.sweep_env import SweepCoverageEnv
from curl.models.pointer import NODE_FEATURE_DIM, node_features


def _square_env():
    pos = np.array([[0, 0, 0], [0.2, 0, 0], [0.2, 0.2, 0], [0, 0.2, 0]], dtype=float)
    nrm = np.tile([0, 0, 1.0], (4, 1))
    return SweepCoverageEnv(pos, nrm, tool_radius=0.05, cell=0.005)


def test_uncovered_drops_after_visit():
    env = _square_env()
    obs, _ = env.reset()
    assert np.allclose(obs["uncovered"], 1.0)
    obs, *_ = env.step(0)
    assert obs["uncovered"][0] < 1.0
    assert 0.0 <= obs["uncovered"].min() and obs["uncovered"].max() <= 1.0


def test_node_features_have_12_columns_and_no_nan():
    rng = np.random.default_rng(0)
    pos, nrm, strokes = generate_scene(rng)
    env = SweepCoverageEnv(pos, nrm, target_segments=strokes)
    obs, _ = env.reset()
    centre = pos[:, :2].mean(0)
    scale = max(float(np.ptp(pos[:, :2], axis=0).max()), 1e-6)
    f = node_features(obs, scale, centre)
    assert f.shape == (len(pos), NODE_FEATURE_DIM) == (len(pos), 12)
    assert not np.isnan(f).any()
