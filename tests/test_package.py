from __future__ import annotations

import importlib.metadata

import glasz as m


def test_version():
    assert importlib.metadata.version("glasz") == m.__version__
