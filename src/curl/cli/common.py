"""Shared command-line options for the curl CLIs."""
from __future__ import annotations

import argparse

from curl.config import DEFAULT_CONFIG_PATH, ROOT


def add_config_arg(ap: argparse.ArgumentParser) -> None:
    default = DEFAULT_CONFIG_PATH.relative_to(ROOT)
    ap.add_argument("--config", default=None,
                    help=f"settings file (default: {default}); command-line flags override it")


def make_parser(description: str) -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description=description,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
