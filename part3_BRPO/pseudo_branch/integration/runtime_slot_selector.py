"""Compatibility wrapper for refactored module location.

Keep the historical import path stable while the implementation lives under
`online_mapping.runtime.slot_selector`.
"""
from online_mapping.runtime.slot_selector import *  # noqa: F401,F403
