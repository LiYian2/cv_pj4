"""BRPO v2 signal path utilities.

This package intentionally lives beside the legacy mask/depth path and does not
modify or overwrite the old seed_support -> train_mask -> target_depth chain.
"""

from .depth_supervision_v2 import build_depth_supervision_v2, write_depth_supervision_outputs
from .joint_confidence import build_joint_confidence_from_rgb_and_depth, build_joint_depth_target, write_joint_signal_outputs
from .joint_observation import build_joint_observation_from_candidates, write_joint_observation_outputs
from .pseudo_observation_brpo_style import build_brpo_style_observation, write_brpo_style_observation_outputs
from .pseudo_observation_verifier import build_pseudo_observation_verifier, write_pseudo_observation_verifier_outputs
from .rgb_mask_inference import build_rgb_mask_from_correspondences, write_rgb_mask_outputs
from .support_expand import build_support_expand_from_a1, write_support_expand_outputs

__all__ = [
    'build_rgb_mask_from_correspondences',
    'write_rgb_mask_outputs',
    'build_depth_supervision_v2',
    'write_depth_supervision_outputs',
    'build_joint_confidence_from_rgb_and_depth',
    'build_joint_depth_target',
    'write_joint_signal_outputs',
    'build_joint_observation_from_candidates',
    'write_joint_observation_outputs',
    'build_brpo_style_observation',
    'write_brpo_style_observation_outputs',
    'build_pseudo_observation_verifier',
    'write_pseudo_observation_verifier_outputs',
    'build_support_expand_from_a1',
    'write_support_expand_outputs',
]
