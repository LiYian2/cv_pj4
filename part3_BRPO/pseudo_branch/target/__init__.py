"""Compatibility target package.

The shared online/standalone target authority now lives under
`core_shared.targets`. Standalone BRPO depth-target utilities should be imported
directly from `standalone_mask_signal`.
"""

from .depth_supervision_v2 import *  # noqa: F401,F403
