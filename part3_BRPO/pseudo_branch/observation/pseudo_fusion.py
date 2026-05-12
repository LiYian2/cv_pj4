"""Compatibility wrapper for refactored module location.

Keep the historical import path stable while the implementation lives under
`core_shared.fusion.pseudo_fusion`.
"""
from core_shared.fusion.pseudo_fusion import *  # noqa: F401,F403
