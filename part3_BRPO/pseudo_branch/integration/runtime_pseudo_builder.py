"""Compatibility wrapper for refactored module location.

Keep the historical import path stable while the implementation lives under
`online_mapping.records.runtime_record_builder`.
"""
from online_mapping.records.runtime_record_builder import *  # noqa: F401,F403
