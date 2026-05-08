"""Stage runtime state, loss assembly, scheduler, and refine-side orchestration."""

from .pseudo_camera_state import (
    ExportedPseudoCameraState,
    make_viewpoint_trainable,
    viewpoint_optimizer_groups,
    current_w2c,
    current_c2w,
    refresh_viewpoint_transforms_,
    apply_pose_delta_before_render_,  # CRITICAL: Apply pose delta before render for gradient flow
    restore_base_pose_after_render_,
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
    scale_reg_loss,  # Scale regularization loss
)
from .pseudo_refine_scheduler import (
    StageAConfig,
    StageA5Config,
    build_stageA_optimizer,
    build_stageA5_optimizers,
)
from .backend_pseudo_bundle import (
    LoadedPseudoBundleSample,
    PseudoBundleSample,
    PseudoBundleBatch,
    load_pseudo_bundle_from_stageA_history,
)
from .backend_pseudo_view_loader import (
    BackendPseudoViewRecord,
    build_record_from_loaded_sample,
    build_records_from_pseudo_bundle,
    normalize_stageA_pseudo_views,
)
from .backend_pseudo_loss import (
    BackendPseudoLossConfig,
    compute_backend_pseudo_exact_loss,
)
from .pose_gauss_newton import (
    compute_pose_jacobian_fd,
    gauss_newton_pose_update,
    gauss_newton_batch_update,
    GaussNewtonPoseOptimizer,
)

__all__ = [
    'ExportedPseudoCameraState',
    'make_viewpoint_trainable',
    'viewpoint_optimizer_groups',
    'current_w2c',
    'current_c2w',
    'refresh_viewpoint_transforms_',
    'apply_pose_delta_before_render_',  # CRITICAL: Apply pose delta before render
    'restore_base_pose_after_render_',
    'apply_pose_residual_',
    'load_exported_view_states',
    'apply_loaded_view_state_',
    'summarize_true_pose_deltas',
    'export_view_state',
    'build_stageA_loss',
    'build_stageA_loss_source_aware',
    'build_stageA_loss_exact_shared_cm',
    'scale_reg_loss',  # Scale regularization
    'StageAConfig',
    'StageA5Config',
    'build_stageA_optimizer',
    'build_stageA5_optimizers',
    'LoadedPseudoBundleSample',
    'PseudoBundleSample',
    'PseudoBundleBatch',
    'load_pseudo_bundle_from_stageA_history',
    'BackendPseudoViewRecord',
    'build_record_from_loaded_sample',
    'build_records_from_pseudo_bundle',
    'normalize_stageA_pseudo_views',
    'BackendPseudoLossConfig',
    'compute_backend_pseudo_exact_loss',
    # Gauss-Newton pose optimization
    'compute_pose_jacobian_fd',
    'gauss_newton_pose_update',
    'gauss_newton_batch_update',
    'GaussNewtonPoseOptimizer',
]
