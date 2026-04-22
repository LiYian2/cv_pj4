"""Stage runtime state, loss assembly, scheduler, and refine-side orchestration."""

from .pseudo_camera_state import (
    ExportedPseudoCameraState,
    make_viewpoint_trainable,
    viewpoint_optimizer_groups,
    current_w2c,
    current_c2w,
    refresh_viewpoint_transforms_,
    apply_pose_residual_,
    load_exported_view_states,
    apply_loaded_view_state_,
    summarize_true_pose_deltas,
    export_view_state,
)
from .pseudo_loss_v2 import (
    build_stageA_loss,
    build_stageA_loss_source_aware,
    build_stageA_loss_exact_shared_cm,
)
from .pseudo_refine_scheduler import (
    StageAConfig,
    StageA5Config,
    build_stageA_optimizer,
    build_stageA5_optimizers,
)

__all__ = [
    'ExportedPseudoCameraState',
    'make_viewpoint_trainable',
    'viewpoint_optimizer_groups',
    'current_w2c',
    'current_c2w',
    'refresh_viewpoint_transforms_',
    'apply_pose_residual_',
    'load_exported_view_states',
    'apply_loaded_view_state_',
    'summarize_true_pose_deltas',
    'export_view_state',
    'build_stageA_loss',
    'build_stageA_loss_source_aware',
    'build_stageA_loss_exact_shared_cm',
    'StageAConfig',
    'StageA5Config',
    'build_stageA_optimizer',
    'build_stageA5_optimizers',
]
