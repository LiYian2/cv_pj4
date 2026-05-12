"""Compatibility wrapper for refactored module location.

Keep the historical import path stable while the implementation lives under
`core_shared.verification.brpo_reprojection_verify`.
"""
from core_shared.verification.brpo_reprojection_verify import *  # noqa: F401,F403
