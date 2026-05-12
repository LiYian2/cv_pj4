"""Online runtime refine facade consumed by the S3PO bridge."""

from .backend_pseudo_bundle import load_pseudo_bundle_from_stageA_history
from .backend_pseudo_view_loader import BackendPseudoViewRecord, build_records_from_pseudo_bundle
from .backend_pseudo_loss import BackendPseudoLossConfig, compute_backend_pseudo_exact_loss
from .pseudo_camera_state import current_w2c, refresh_viewpoint_transforms_, apply_pose_delta_before_render_, viewpoint_optimizer_groups
from .pose_gauss_newton import GaussNewtonPoseOptimizer
from .pseudo_loss_v2 import scale_reg_loss

__all__ = [
    "load_pseudo_bundle_from_stageA_history",
    "BackendPseudoViewRecord",
    "build_records_from_pseudo_bundle",
    "BackendPseudoLossConfig",
    "compute_backend_pseudo_exact_loss",
    "current_w2c",
    "refresh_viewpoint_transforms_",
    "apply_pose_delta_before_render_",
    "viewpoint_optimizer_groups",
    "GaussNewtonPoseOptimizer",
    "scale_reg_loss",
]
