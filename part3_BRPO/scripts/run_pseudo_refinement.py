#!/usr/bin/env python3
"""Compatibility shim for the archived legacy refine entrypoint."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_ARCHIVED = Path(__file__).parent / 'archive_experiments' / 'legacy_entry' / 'run_pseudo_refinement.py'


def _load_archived_module():
    spec = importlib.util.spec_from_file_location('run_pseudo_refinement_legacy_archived', _ARCHIVED)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main():
    module = _load_archived_module()
    module.main()


if __name__ == '__main__':
    main()
