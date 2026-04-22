#!/usr/bin/env python3
"""External CLI compatibility wrapper.

Keep this old top-level path stable for manual legacy invocations, but route all
internal callers through `scripts/compat/run_pseudo_refinement.py` instead.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_COMPAT = Path(__file__).parent / 'compat' / 'run_pseudo_refinement.py'


def _load_compat_module():
    spec = importlib.util.spec_from_file_location('run_pseudo_refinement_compat', _COMPAT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main():
    module = _load_compat_module()
    module.main()


if __name__ == '__main__':
    main()
