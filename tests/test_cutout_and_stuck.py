import numpy as np

from curl.env.sweep_env import SweepCoverageEnv


def _two_nodes_with_gap():
    pos = np.array([[0, 0, 0], [0.2, 0, 0]], dtype=float)
    nrm = np.tile([0, 0, 1.0], (2, 1))
    return pos, nrm


def test_forbidden_move_across_gap_is_masked():
    pos, nrm = _two_nodes_with_gap()
    env = SweepCoverageEnv(pos, nrm, tool_radius=0.05, cell=0.005, forbid_off_surface=True)
    env.reset()
    env.step(0)
    assert not env.action_mask()[1]


def test_shared_pair_off_matches_fresh_computation():
    pos, nrm = _two_nodes_with_gap()
    a = SweepCoverageEnv(pos, nrm, tool_radius=0.05, cell=0.005, forbid_off_surface=True)
    b = SweepCoverageEnv(pos, nrm, tool_radius=0.05, cell=0.005, forbid_off_surface=True,
                         pair_off=a.pair_off)
    assert np.allclose(a.pair_off, b.pair_off)
    assert a.pair_off[0, 1] == a.pair_off[1, 0] > 0


def test_stuck_ends_episode_with_penalty():
    pos = np.array([[0, 0, 0], [0.4, 0, 0], [0.8, 0, 0]], dtype=float)
    nrm = np.tile([0, 0, 1.0], (3, 1))
    env = SweepCoverageEnv(pos, nrm, tool_radius=0.05, cell=0.005,
                           forbid_off_surface=True, coverage_target=1.01, w_stuck=2.0)
    env.reset()
    _, reward, terminated, _, info = env.step(0)
    assert info["stuck"] is True
    assert terminated is True
    gain = 4.0 * env.stats["new"] / env.T
    assert np.isclose(reward, gain - 2.0 * 2 / 3)
    assert env.stats["stuck"] is True
