"""Compatibility wrapper for refactored module location.

Keep the historical import path stable while the implementation lives under
`core_shared.pose.pseudo_camera_state`.
"""
from core_shared.pose.pseudo_camera_state import *  # noqa: F401,F403
