"""Shared pytest configuration.

Adds the repo root to `sys.path` so `pilot` imports without an install step, and
gates the model-downloading tests behind `--run-slow`.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False,
                     help="run tests that download and load a real model")


def pytest_configure(config):
    config.addinivalue_line("markers",
                            "slow: needs a real model (use --run-slow)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip = pytest.mark.skip(reason="needs --run-slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
