import pytest
import yaml

from curl.config import (DEFAULT_CONFIG_PATH, ROOT, ConfigError, load_config, load_run_settings,
                         override, save_config)


def test_default_config_matches_the_v1_settings():
    cfg = load_config()
    assert cfg.env.tool_radius_m == 0.05
    assert cfg.env.cell_m == 0.01
    assert cfg.env.stroke_max_len_m == 0.25
    assert cfg.env.coverage_target == 0.98
    assert cfg.env.full_visit_target == 1.01
    assert cfg.ik.restarts == 40
    assert cfg.ik.base_pos_m == (0.72, 0.40, 0.40)
    assert cfg.training.updates == 1200
    assert cfg.training.scenes == 12
    assert cfg.training.rollouts == 8
    assert cfg.training.lr == 3e-4
    assert cfg.training.lr_min == 3e-5
    assert cfg.training.ent == 0.01
    assert cfg.training.val_every == 100
    assert cfg.training.val_scenes == 10
    assert cfg.training.seed == 0
    assert cfg.training.val_seed == 555_001


def test_relative_paths_resolve_under_repo_root():
    cfg = load_config()
    assert cfg.paths.input_csv == ROOT / "input" / "Vanity_Large_Top_Points.csv"
    assert cfg.paths.checkpoint == ROOT / "results" / "pointer_policy_v1.pt"


def _write(tmp_path, text):
    p = tmp_path / "cfg.yaml"
    p.write_text(text)
    return p


BASE_YAML = """
paths: {input_csv: input/x.csv, results_dir: results, checkpoint: results/c.pt}
env: {tool_radius_m: 0.05, cell_m: 0.01, stroke_max_len_m: 0.25, coverage_target: 0.98, full_visit_target: 1.01}
ik: {restarts: 40, base_pos_m: [0.72, 0.40, 0.40]}
training: {updates: 1, scenes: 1, rollouts: 1, lr: 0.0003, lr_min: 0.00003, ent: 0.01, val_every: 1, val_scenes: 1, seed: 0, val_seed: 1}
"""


def test_valid_custom_file_loads(tmp_path):
    assert load_config(_write(tmp_path, BASE_YAML)).training.updates == 1


def test_missing_key_is_named(tmp_path):
    text = BASE_YAML.replace("cell_m: 0.01, ", "")
    with pytest.raises(ConfigError, match="missing key.*cell_m"):
        load_config(_write(tmp_path, text))


def test_unknown_key_is_named(tmp_path):
    text = BASE_YAML.replace("restarts: 40", "restarts: 40, bogus: 1")
    with pytest.raises(ConfigError, match="unknown key.*bogus"):
        load_config(_write(tmp_path, text))


def test_string_number_is_rejected_with_hint(tmp_path):
    text = BASE_YAML.replace("lr: 0.0003", 'lr: "3e-4"')
    with pytest.raises(ConfigError, match="training.lr.*YAML reads 3e-4 as text"):
        load_config(_write(tmp_path, text))


def test_base_pose_needs_three_numbers(tmp_path):
    text = BASE_YAML.replace("[0.72, 0.40, 0.40]", "[0.72, 0.40]")
    with pytest.raises(ConfigError, match="ik.base_pos_m"):
        load_config(_write(tmp_path, text))


def test_missing_section_is_named(tmp_path):
    text = "\n".join(line for line in BASE_YAML.splitlines() if not line.startswith("env:"))
    with pytest.raises(ConfigError, match="missing section.*env"):
        load_config(_write(tmp_path, text))


def test_override_ignores_none_and_applies_values():
    cfg = load_config()
    new = override(cfg, "training", updates=5, lr=None)
    assert new.training.updates == 5
    assert new.training.lr == cfg.training.lr
    assert cfg.training.updates == 1200
    moved = override(cfg, "ik", base_pos_m=[1.0, 2.0, 3.0])
    assert moved.ik.base_pos_m == (1.0, 2.0, 3.0)


def test_override_rejects_unknown_key():
    with pytest.raises(ConfigError, match="unknown key.*nope"):
        override(load_config(), "training", nope=1)


def test_save_then_load_round_trips(tmp_path):
    cfg = load_config()
    saved = save_config(cfg, tmp_path / "snap.yaml")
    assert load_config(saved) == cfg


def test_saved_paths_inside_repo_are_relative(tmp_path):
    saved = save_config(load_config(), tmp_path / "snap.yaml")
    raw = yaml.safe_load(saved.read_text())
    assert raw["paths"]["input_csv"] == "input/Vanity_Large_Top_Points.csv"
    assert raw["ik"]["base_pos_m"] == [0.72, 0.4, 0.4]


def test_run_section_is_recorded_and_ignored_on_load(tmp_path):
    saved = save_config(load_config(), tmp_path / "snap.yaml",
                        run={"visit_all": True, "no_ik": False})
    assert yaml.safe_load(saved.read_text())["run"] == {"visit_all": True, "no_ik": False}
    assert load_config(saved) == load_config()


def test_load_run_settings_reads_switches(tmp_path):
    saved = save_config(load_config(), tmp_path / "snap.yaml",
                        run={"visit_all": True, "no_ik": False, "allow_off_surface": False})
    assert load_run_settings(saved) == {"visit_all": True, "no_ik": False,
                                        "allow_off_surface": False}
    assert load_run_settings(DEFAULT_CONFIG_PATH) == {}


def test_unknown_section_still_rejected_next_to_run(tmp_path):
    text = BASE_YAML + "extra: {a: 1}\n"
    with pytest.raises(ConfigError, match="unknown section.*extra"):
        load_config(_write(tmp_path, text))


def test_overrides_appear_in_snapshot(tmp_path):
    cfg = override(load_config(), "training", updates=5)
    saved = save_config(cfg, tmp_path / "snap.yaml")
    assert load_config(saved).training.updates == 5
