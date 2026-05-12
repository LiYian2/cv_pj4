"""Compatibility wrapper for refactored module location.

Keep the historical import path stable while the implementation lives under
`core_shared.records.backend_pseudo_bundle`.
"""
from core_shared.records.backend_pseudo_bundle import *  # noqa: F401,F403
