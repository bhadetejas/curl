"""Settings loaded from YAML (default: configs/default.yaml).

Every value is validated on load: missing, unknown or wrongly typed keys raise
ConfigError naming the key. Relative paths in the YAML are relative to the repo root;
relative paths given on the command line are relative to the working directory.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import get_type_hints

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "default.yaml"


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class PathsConfig:
    input_csv: Path
    results_dir: Path
    checkpoint: Path


@dataclass(frozen=True)
class EnvConfig:
    tool_radius_m: float
    cell_m: float
    stroke_max_len_m: float
    coverage_target: float
    full_visit_target: float


@dataclass(frozen=True)
class IKConfig:
    restarts: int
    base_pos_m: tuple[float, float, float]


@dataclass(frozen=True)
class TrainingConfig:
    updates: int
    scenes: int
    rollouts: int
    lr: float
    lr_min: float
    ent: float
    val_every: int
    val_scenes: int
    seed: int
    val_seed: int


@dataclass(frozen=True)
class Config:
    paths: PathsConfig
    env: EnvConfig
    ik: IKConfig
    training: TrainingConfig


_SECTIONS = {"paths": PathsConfig, "env": EnvConfig, "ik": IKConfig, "training": TrainingConfig}


def _parses_as_float(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def _coerce(value, hint, where: str, base: Path):
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            hint_text = ""
            if isinstance(value, str) and _parses_as_float(value):
                hint_text = " (YAML reads 3e-4 as text; write 3.0e-4 or leave out the quotes)"
            raise ConfigError(f"{where}: expected a number, got {value!r}{hint_text}")
        return float(value)
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where}: expected an integer, got {value!r}")
        return value
    if hint is Path:
        if not isinstance(value, (str, Path)):
            raise ConfigError(f"{where}: expected a path string, got {value!r}")
        p = Path(value).expanduser()
        return p if p.is_absolute() else base / p
    if getattr(hint, "__origin__", None) is tuple:
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ConfigError(f"{where}: expected three numbers, got {value!r}")
        return tuple(_coerce(v, float, f"{where}[{i}]", base) for i, v in enumerate(value))
    raise TypeError(f"unsupported config type {hint!r} at {where}")


def _build(cls, data, section: str, base: Path):
    if not isinstance(data, dict):
        raise ConfigError(f"{section}: expected a mapping of settings")
    names = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - names)
    if unknown:
        raise ConfigError(f"{section}: unknown key(s) {unknown}")
    missing = sorted(names - set(data))
    if missing:
        raise ConfigError(f"{section}: missing key(s) {missing}")
    hints = get_type_hints(cls)
    return cls(**{k: _coerce(data[k], hints[k], f"{section}.{k}", base) for k in names})


def load_config(path: Path | str | None = None) -> Config:
    """Load and validate settings from `path` (default: configs/default.yaml)."""
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    try:
        raw = yaml.safe_load(path.read_text())
    except OSError as e:
        raise ConfigError(f"cannot read {path}: {e}") from e
    except yaml.YAMLError as e:
        raise ConfigError(f"invalid YAML in {path}: {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping with sections {sorted(_SECTIONS)}")
    unknown = sorted(set(raw) - set(_SECTIONS) - {"run"})
    if unknown:
        raise ConfigError(f"{path}: unknown section(s) {unknown}")
    missing = sorted(set(_SECTIONS) - set(raw))
    if missing:
        raise ConfigError(f"{path}: missing section(s) {missing}")
    return Config(**{name: _build(cls, raw[name], name, ROOT) for name, cls in _SECTIONS.items()})


def load_run_settings(path: Path | str | None = None) -> dict:
    """The `run:` switches recorded in a settings file, or {} if it has none."""
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    raw = yaml.safe_load(path.read_text())
    run = raw.get("run") if isinstance(raw, dict) else None
    return dict(run) if run else {}


def _to_yaml_value(value):
    if isinstance(value, Path):
        try:
            return str(value.relative_to(ROOT))
        except ValueError:
            return str(value)
    if isinstance(value, tuple):
        return [_to_yaml_value(v) for v in value]
    return value


def config_to_dict(cfg: Config) -> dict:
    return {name: {f.name: _to_yaml_value(getattr(getattr(cfg, name), f.name))
                   for f in fields(getattr(cfg, name))}
            for name in _SECTIONS}


def save_config(cfg: Config, path: Path | str, run: dict | None = None) -> Path:
    """Write the resolved settings as YAML, so a run can be reproduced later.

    Paths inside the repo are stored relative to it. `run` records switches that affect
    the result but are not config keys; it is written under a top-level `run:` key.
    """
    data = config_to_dict(cfg)
    if run is not None:
        data["run"] = dict(run)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def override(cfg: Config, section: str, **values) -> Config:
    """Return `cfg` with `values` replacing keys of one section. None values are ignored."""
    if section not in _SECTIONS:
        raise ConfigError(f"unknown section {section!r}")
    current = dataclasses.asdict(getattr(cfg, section))
    current.update({k: v for k, v in values.items() if v is not None})
    new = _build(_SECTIONS[section], current, section, Path.cwd())
    return replace(cfg, **{section: new})
