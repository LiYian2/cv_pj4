"""Compatibility wrapper for refactored module location.

Keep the historical import path stable while the implementation lives under
`core_shared.records.backend_pseudo_view_loader`.
"""
from core_shared.records.backend_pseudo_view_loader import *  # noqa: F401,F403
