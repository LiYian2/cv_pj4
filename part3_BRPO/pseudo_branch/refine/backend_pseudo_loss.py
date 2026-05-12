"""Compatibility wrapper for refactored module location.

Keep the historical import path stable while the implementation lives under
`core_shared.losses.backend_pseudo_loss`.
"""
from core_shared.losses.backend_pseudo_loss import *  # noqa: F401,F403
