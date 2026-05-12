#!/usr/bin/env python3
"""Compatibility wrapper for refactored script location.

Keep the historical CLI path stable while the implementation lives under
`legacy_or_archive.retired_entrypoints.run_pseudo_refinement_v2`.
"""
from __future__ import annotations

import importlib
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_TARGET = "legacy_or_archive.retired_entrypoints.run_pseudo_refinement_v2"
_MODULE = None


def _load_module():
    global _MODULE
    if _MODULE is None:
        _MODULE = importlib.import_module(_TARGET)
    return _MODULE


def __getattr__(name: str):
    return getattr(_load_module(), name)


def main():
    module = _load_module()
    if hasattr(module, "main"):
        return module.main()
    return runpy.run_module(_TARGET, run_name="__main__")


if __name__ == "__main__":
    main()
