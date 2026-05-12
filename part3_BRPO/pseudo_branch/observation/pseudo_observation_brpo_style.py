"""Online-only facade for exact-upstream observation semantics.

Historical standalone observation builders were moved to
standalone_mask_signal.legacy_signal_observation.
"""

from core_shared.targets.exact_upstream_observation import (
    build_exact_brpo_upstream_target_observation,
    write_exact_brpo_upstream_target_observation_outputs,
)

__all__ = [
    "build_exact_brpo_upstream_target_observation",
    "write_exact_brpo_upstream_target_observation_outputs",
]
