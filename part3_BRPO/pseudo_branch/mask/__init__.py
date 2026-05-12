"""Compatibility mask package.

The live online-mapping mask authority now lives under `online_mapping.mask`.
Standalone BRPO signal helpers live under `standalone_mask_signal` and should
be imported directly by new code.
"""

from .rgb_mask_inference import *  # noqa: F401,F403
from .dense_match_densify import *  # noqa: F401,F403
from .cm_local_expansion import *  # noqa: F401,F403
from .joint_confidence import *  # noqa: F401,F403
