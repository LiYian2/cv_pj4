"""Compatibility wrapper for refactored module location.

Keep the historical import path stable while the implementation lives under
`core_shared.pose.pose_gauss_newton`.
"""
from core_shared.pose.pose_gauss_newton import *  # noqa: F401,F403
