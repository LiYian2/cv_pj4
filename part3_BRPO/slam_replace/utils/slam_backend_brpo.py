from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gaussian_splatting.gaussian_renderer import render
from utils.logging_utils import Log
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_mapping

from pseudo_branch.refine import (
    BackendPseudoLossConfig,
    BackendPseudoViewRecord,
    build_records_from_pseudo_bundle,
    compute_backend_pseudo_exact_loss,
    current_w2c,
    load_pseudo_bundle_from_stageA_history,
    refresh_viewpoint_transforms_,
    apply_pose_delta_before_render_,  # CRITICAL FIX: Apply pose delta before render
    viewpoint_optimizer_groups,
    scale_reg_loss,  # Scale regularization loss
    GaussNewtonPoseOptimizer,  # Gauss-Newton pose optimization
)


@dataclass
class BRPOContinuationConfig:
    stageA_history_json: str
    num_iterations: int = 40
    num_pseudo_views: int = 4
    extra_real_views: int = 2
    lambda_real: float = 1.0
    lambda_pseudo: float = 1.0
    lambda_depth: float = 1.0
    match_real_loss_weights: bool = False
    beta_rgb: float | None = None
    lambda_pose: float | None = None
    lambda_exp: float | None = None
    trans_weight: float | None = None
    lambda_abs_pose: float | None = None
    lambda_abs_t: float | None = None
    lambda_abs_r: float | None = None
    abs_pose_robust: str | None = None
    update_real_pose: bool = True
    update_pseudo_pose: bool = True
    use_depth: bool = True
    split_pseudo_authority: bool = False
    pseudo_scene_mask_mode: str = "all_valid"
    enable_densify: bool = False
    enable_prune: bool = False
    enable_opacity_reset: bool = False
    isotropic_weight: float = 10.0
    output_dir: str | None = None
    seed: int = 0


@dataclass
class BRPOMappingConfig:
    num_iterations: int = 20
    num_pseudo_views_per_step: int = 1
    lambda_real: float = 1.0
    lambda_pseudo: float = 1.0
    lambda_depth: float = 1.0
    match_real_loss_weights: bool = False
    beta_rgb: float | None = 0.7
    lambda_pose: float | None = 0.01
    lambda_exp: float | None = 0.001  # Exposure regularization weight
    lambda_scale: float | None = 0.01  # Scale regularization weight (NEW)
    trans_weight: float | None = 1.0
    lambda_abs_pose: float | None = 0.0
    lambda_abs_t: float | None = 3.0
    lambda_abs_r: float | None = 0.1
    abs_pose_robust: str | None = "charbonnier"
    update_real_pose: bool = False
    update_real_exposure: bool | None = None
    update_pseudo_pose: bool = True
    use_depth: bool = True
    split_pseudo_authority: bool = True
    pseudo_scene_mask_mode: str = "both_only"
    topology_mode: str = "side_branch"
    pseudo_window_equivalence: bool = False
    extra_real_views: int = 0
    propagate_pseudo_delta_to_neighbors: bool = True
    gaussian_maintenance_source: str = "all_views"
    enable_densify: bool = False
    enable_prune: bool = False
    enable_opacity_reset: bool = False
    isotropic_weight: float = 10.0
    max_scale: float | None = None  # Maximum allowed scale (NEW)
    # Gauss-Newton pose optimization (NEW)
    use_gauss_newton: bool = False  # Enable GN instead of Adam for pose
    gn_max_iters: int = 5  # GN iterations per step
    gn_damping: float = 0.01  # Levenberg-Marquardt damping
    gn_every_n_steps: int = 1  # Apply GN every N steps (1 = every step)
    output_dir: str | None = None
    seed: int = 0
    depth_loss_mode: str = "exact_shared_cm_v1"
    tau_rel_depth: float = 0.15


class BRPOBackEndContinuation:
    def __init__(
        self,
        *,
        config: dict[str, Any],
        gaussians,
        pipeline_params,
        opt_params,
        background: torch.Tensor,
        cameras: dict[int, Any],
        kf_indices: list[int],
        current_window: list[int] | None = None,
        iteration_start: int = 0,
        gaussian_update_every: int = 100,
        gaussian_update_offset: int = 0,
        gaussian_th: float = 0.7,
        gaussian_extent: float = 6.0,
        gaussian_reset: int = 3000,
        size_threshold: float | None = None,
    ):
        self.config = config
        self.gaussians = gaussians
        self.pipeline_params = pipeline_params
        self.opt_params = opt_params
        self.background = background
        self.cameras = cameras
        self.kf_indices = list(kf_indices)
        raw_current_window = list(current_window) if current_window else list(kf_indices[-config["Training"]["window_size"]:])
        self.current_window = self._normalize_current_window_order(raw_current_window)
        if raw_current_window != self.current_window:
            Log(
                f"[BRPOContinuation] normalized current_window order from {raw_current_window} to {self.current_window}"
            )
        self.iteration_count = int(iteration_start)
        self.gaussian_update_every = int(gaussian_update_every)
        self.gaussian_update_offset = int(gaussian_update_offset)
        self.gaussian_th = float(gaussian_th)
        self.gaussian_extent = float(gaussian_extent)
        self.gaussian_reset = int(gaussian_reset)
        self.size_threshold = size_threshold

    def _propagate_pseudo_pose_to_neighbors_(
        self,
        record: "BackendPseudoViewRecord",
        alpha: float = 0.5,
    ) -> None:
        """Propagate pseudo pose delta to neighboring keyframes.
        
        Phase 2 fix: Allow pseudo pose optimization to influence real keyframes.
        The pseudo view is interpolated between left and right reference frames.
        If pseudo pose moved, we propagate a portion of the delta to neighbors.
        
        Args:
            record: BackendPseudoViewRecord with viewpoint and reference frame IDs
            alpha: Weight for left frame (1-alpha goes to right frame)
        """
        # Get reference frame IDs from record (Phase 2 fix)
        left_ref_id = getattr(record, "left_ref_frame_id", None)
        right_ref_id = getattr(record, "right_ref_frame_id", None)
        
        if left_ref_id is None or right_ref_id is None:
            return  # No reference frame info, skip propagation
        
        pseudo_vp = record.viewpoint
        
        # Get pseudo pose delta (before fold zeros it)
        if not hasattr(pseudo_vp, "cam_rot_delta") or not hasattr(pseudo_vp, "cam_trans_delta"):
            return
        
        pseudo_rot_delta = pseudo_vp.cam_rot_delta.detach().clone()
        pseudo_trans_delta = pseudo_vp.cam_trans_delta.detach().clone()
        
        # Propagate to left keyframe
        if left_ref_id in self.cameras:
            left_vp = self.cameras[left_ref_id]
            if hasattr(left_vp, "cam_rot_delta") and hasattr(left_vp, "cam_trans_delta"):
                with torch.no_grad():
                    left_vp.cam_rot_delta.add_(alpha * pseudo_rot_delta)
                    left_vp.cam_trans_delta.add_(alpha * pseudo_trans_delta)
        
        # Propagate to right keyframe
        if right_ref_id in self.cameras:
            right_vp = self.cameras[right_ref_id]
            if hasattr(right_vp, "cam_rot_delta") and hasattr(right_vp, "cam_trans_delta"):
                with torch.no_grad():
                    right_vp.cam_rot_delta.add_((1.0 - alpha) * pseudo_rot_delta)
                    right_vp.cam_trans_delta.add_((1.0 - alpha) * pseudo_trans_delta)

    @staticmethod
    def _fold_pseudo_pose_residual_(viewpoint) -> None:
        new_w2c = current_w2c(viewpoint)
        if hasattr(viewpoint, "update_RT"):
            viewpoint.update_RT(new_w2c[:3, :3].detach().clone(), new_w2c[:3, 3].detach().clone())
        else:
            viewpoint.R = new_w2c[:3, :3].detach().clone()
            viewpoint.T = new_w2c[:3, 3].detach().clone()
            refresh_viewpoint_transforms_(viewpoint)
        with torch.no_grad():
            viewpoint.cam_rot_delta.zero_()
            viewpoint.cam_trans_delta.zero_()

    def _normalize_current_window_order(self, window: list[int]) -> list[int]:
        if not window:
            return []
        order = {int(kf_idx): pos for pos, kf_idx in enumerate(self.kf_indices)}
        normalized = sorted((int(x) for x in window), key=lambda x: order.get(int(x), -1), reverse=True)
        return normalized

    @staticmethod
    def _first_present(args: dict[str, Any], keys: tuple[str, ...]):
        for key in keys:
            if key in args and args[key] is not None:
                return args[key]
        return None

    def _resolve_history_backed_value(
        self,
        explicit_value,
        history_args: dict[str, Any],
        *,
        history_keys: tuple[str, ...],
        fallback,
        cast,
    ):
        if explicit_value is not None:
            return cast(explicit_value)
        history_value = self._first_present(history_args, history_keys)
        if history_value is not None:
            return cast(history_value)
        return cast(fallback)

    def _resolve_pseudo_loss_cfg(
        self,
        cfg: BRPOContinuationConfig,
        history_args: dict[str, Any],
    ) -> tuple[BackendPseudoLossConfig, dict[str, Any]]:
        match_real_loss_weights = bool(getattr(cfg, "match_real_loss_weights", False))
        training_alpha = float(self.config.get("Training", {}).get("alpha", 0.95))
        resolved_lambda_depth = float(cfg.lambda_depth)
        if match_real_loss_weights:
            resolved_beta_rgb = training_alpha
            resolved_lambda_depth = max(0.0, 1.0 - training_alpha) if bool(cfg.use_depth) else 0.0
        else:
            resolved_beta_rgb = self._resolve_history_backed_value(
                cfg.beta_rgb,
                history_args,
                history_keys=("stageA_beta_rgb", "beta_rgb"),
                fallback=0.7,
                cast=float,
            )
        # Apply use_depth flag regardless of match_real_loss_weights
        if not bool(cfg.use_depth):
            resolved_lambda_depth = 0.0
        resolved = {
            "beta_rgb": float(resolved_beta_rgb),
            "lambda_depth": float(resolved_lambda_depth),
            "match_real_loss_weights": match_real_loss_weights,
            "real_mapping_alpha": training_alpha,
            "lambda_pose": self._resolve_history_backed_value(
                cfg.lambda_pose,
                history_args,
                history_keys=("stageA_lambda_pose", "lambda_pose"),
                fallback=0.0,
                cast=float,
            ),
            "lambda_exp": self._resolve_history_backed_value(
                cfg.lambda_exp,
                history_args,
                history_keys=("stageA_lambda_exp", "lambda_exp"),
                fallback=0.0,
                cast=float,
            ),
            "trans_weight": self._resolve_history_backed_value(
                cfg.trans_weight,
                history_args,
                history_keys=("stageA_trans_reg_weight", "trans_reg_weight", "trans_weight"),
                fallback=1.0,
                cast=float,
            ),
            "lambda_abs_pose": self._resolve_history_backed_value(
                cfg.lambda_abs_pose,
                history_args,
                history_keys=("stageA_lambda_abs_pose", "lambda_abs_pose"),
                fallback=0.0,
                cast=float,
            ),
            "lambda_abs_t": self._resolve_history_backed_value(
                cfg.lambda_abs_t,
                history_args,
                history_keys=("stageA_lambda_abs_t", "lambda_abs_t"),
                fallback=0.0,
                cast=float,
            ),
            "lambda_abs_r": self._resolve_history_backed_value(
                cfg.lambda_abs_r,
                history_args,
                history_keys=("stageA_lambda_abs_r", "lambda_abs_r"),
                fallback=0.0,
                cast=float,
            ),
            "abs_pose_robust": self._resolve_history_backed_value(
                cfg.abs_pose_robust,
                history_args,
                history_keys=("stageA_abs_pose_robust", "abs_pose_robust"),
                fallback="charbonnier",
                cast=str,
            ),
        }
        return (
            BackendPseudoLossConfig(
                beta_rgb=float(resolved["beta_rgb"]),
                lambda_pose=float(resolved["lambda_pose"]),
                lambda_exp=float(resolved["lambda_exp"]),
                trans_weight=float(resolved["trans_weight"]),
                lambda_depth=float(resolved["lambda_depth"]),
                use_depth=bool(cfg.use_depth),
                lambda_abs_pose=float(resolved["lambda_abs_pose"]),
                lambda_abs_t=float(resolved["lambda_abs_t"]),
                lambda_abs_r=float(resolved["lambda_abs_r"]),
                abs_pose_robust=str(resolved["abs_pose_robust"]),
                depth_loss_mode=str(getattr(cfg, "depth_loss_mode", "exact_shared_cm_v1")),
            ),
            resolved,
        )

    def _valid_real_indices(self) -> list[int]:
        valid = []
        for idx in self.kf_indices:
            vp = self.cameras.get(idx)
            if vp is None:
                continue
            if getattr(vp, "original_image", None) is None:
                continue
            if getattr(vp, "mono_depth", None) is None:
                continue
            valid.append(int(idx))
        return valid

    def _build_joint_pose_optimizers(
        self,
        pseudo_records: list[BackendPseudoViewRecord],
        current_window: list[int],
        *,
        include_real_pose: bool = True,
        include_real_exposure: bool = True,
        include_pseudo_pose: bool = True,
        include_pseudo_exposure: bool = True,
    ) -> tuple[torch.optim.Optimizer | None, torch.optim.Optimizer | None, torch.optim.Optimizer | None]:
        real_opt_params = []
        pseudo_pose_opt_params = []
        pseudo_exposure_opt_params = []
        frames_to_optimize = int(self.config["Training"].get("pose_window", len(current_window)))
        lr_cfg = self.config["Training"]["lr"]
        lr_rot = float(lr_cfg["cam_rot_delta"]) * 0.5
        lr_trans = float(lr_cfg["cam_trans_delta"]) * 0.5
        lr_exp = 0.01

        for cam_pos, cam_idx in enumerate(current_window):
            viewpoint = self.cameras[cam_idx]
            if include_real_pose and cam_pos < frames_to_optimize:
                real_opt_params.append({"params": [viewpoint.cam_rot_delta], "lr": lr_rot, "name": f"real_rot_{viewpoint.uid}"})
                real_opt_params.append({"params": [viewpoint.cam_trans_delta], "lr": lr_trans, "name": f"real_trans_{viewpoint.uid}"})
            if include_real_exposure:
                real_opt_params.append({"params": [viewpoint.exposure_a], "lr": lr_exp, "name": f"real_exposure_a_{viewpoint.uid}"})
                real_opt_params.append({"params": [viewpoint.exposure_b], "lr": lr_exp, "name": f"real_exposure_b_{viewpoint.uid}"})

        for record in pseudo_records:
            uid_prefix = f"pseudo_{record.sample_id}"
            if include_pseudo_pose:
                pseudo_pose_opt_params.append({"params": [record.viewpoint.cam_rot_delta], "lr": lr_rot, "name": f"{uid_prefix}_cam_rot_delta"})
                pseudo_pose_opt_params.append({"params": [record.viewpoint.cam_trans_delta], "lr": lr_trans, "name": f"{uid_prefix}_cam_trans_delta"})
            if include_pseudo_exposure:
                pseudo_exposure_opt_params.append({"params": [record.viewpoint.exposure_a], "lr": lr_exp, "name": f"{uid_prefix}_exposure_a"})
                pseudo_exposure_opt_params.append({"params": [record.viewpoint.exposure_b], "lr": lr_exp, "name": f"{uid_prefix}_exposure_b"})

        def _make_optimizer(param_groups):
            return torch.optim.Adam(param_groups) if param_groups else None

        return (
            _make_optimizer(real_opt_params),
            _make_optimizer(pseudo_pose_opt_params),
            _make_optimizer(pseudo_exposure_opt_params),
        )

    def _sample_pseudo_records(
        self,
        rng: random.Random,
        pseudo_records: list[BackendPseudoViewRecord],
        num_pseudo_views: int,
    ) -> list[BackendPseudoViewRecord]:
        if not pseudo_records:
            return []
        n = min(int(num_pseudo_views), len(pseudo_records))
        if n >= len(pseudo_records):
            return list(pseudo_records)
        idxs = rng.sample(range(len(pseudo_records)), n)
        return [pseudo_records[i] for i in idxs]

    @staticmethod
    def _zero_optimizer_grads_(optimizer: torch.optim.Optimizer) -> None:
        for group in optimizer.param_groups:
            for param in group.get("params", []):
                if param is not None and getattr(param, "grad", None) is not None:
                    param.grad = None

    @staticmethod
    def _zero_viewpoint_grads_(viewpoint) -> None:
        for attr in ("cam_rot_delta", "cam_trans_delta", "exposure_a", "exposure_b"):
            tensor = getattr(viewpoint, attr, None)
            if tensor is not None and getattr(tensor, "grad", None) is not None:
                tensor.grad = None

    def _zero_pseudo_viewpoint_grads_(self, pseudo_records: list[BackendPseudoViewRecord]) -> None:
        for record in pseudo_records:
            self._zero_viewpoint_grads_(record.viewpoint)

    @staticmethod
    def _scene_record_for_mode(
        record: BackendPseudoViewRecord,
        scene_mask_mode: str,
    ) -> BackendPseudoViewRecord | None:
        mode = str(scene_mask_mode or "all_valid")
        if mode == "none":
            return None
        if mode == "all_valid":
            return record
        if mode == "both_only":
            if record.support_both_mask is not None:
                both_mask = (np.asarray(record.support_both_mask, dtype=np.float32) > 0.5).astype(np.float32)
            else:
                both_mask = (np.asarray(record.confidence_mask, dtype=np.float32) >= 0.75).astype(np.float32)
            scene_confidence = np.asarray(record.confidence_mask, dtype=np.float32) * both_mask
            if not np.any(scene_confidence > 0):
                return None
            return replace(record, confidence_mask=scene_confidence)
        raise ValueError(f"Unsupported pseudo_scene_mask_mode={scene_mask_mode}")

    def _run_joint_pseudo_engine(
        self,
        *,
        cfg,
        pseudo_records: list[BackendPseudoViewRecord],
        history_args: dict[str, Any],
        current_window: list[int],
        extra_real_candidates: list[int],
        source_info: dict[str, Any],
        runner_tag: str,
        log_prefix: str,
    ) -> dict[str, Any]:
        rng = random.Random(int(cfg.seed))
        torch.manual_seed(int(cfg.seed))
        np.random.seed(int(cfg.seed))

        if not pseudo_records:
            raise RuntimeError(f"{log_prefix} no pseudo records were provided")

        loss_cfg, resolved_pseudo_loss_cfg = self._resolve_pseudo_loss_cfg(cfg, history_args)
        valid_real_indices = self._valid_real_indices()
        current_window = self._normalize_current_window_order([idx for idx in current_window if idx in valid_real_indices])
        if not current_window:
            raise RuntimeError(f"{log_prefix} no valid real keyframes with RGB+mono_depth remained")
        current_window_set = set(current_window)
        extra_real_candidates = [int(idx) for idx in extra_real_candidates if int(idx) in valid_real_indices and int(idx) not in current_window_set]
        update_real_exposure = getattr(cfg, "update_real_exposure", None)
        if update_real_exposure is None:
            update_real_exposure = bool(getattr(cfg, "update_real_pose", False))
        real_pose_optimizer, pseudo_pose_optimizer, pseudo_exposure_optimizer = self._build_joint_pose_optimizers(
            pseudo_records,
            current_window,
            include_real_pose=bool(getattr(cfg, "update_real_pose", False)),
            include_real_exposure=bool(update_real_exposure),
            include_pseudo_pose=bool(getattr(cfg, "update_pseudo_pose", True)),
            include_pseudo_exposure=bool(getattr(cfg, "update_pseudo_pose", True)),
        )
        num_pseudo_views = int(
            getattr(
                cfg,
                "num_pseudo_views",
                getattr(cfg, "num_pseudo_views_per_step", len(pseudo_records)),
            )
        )

        Log(
            f"{log_prefix} resolved pseudo loss cfg: "
            f"match_real_loss_weights={resolved_pseudo_loss_cfg['match_real_loss_weights']}, "
            f"real_mapping_alpha={resolved_pseudo_loss_cfg['real_mapping_alpha']}, "
            f"beta_rgb={resolved_pseudo_loss_cfg['beta_rgb']}, "
            f"lambda_depth={resolved_pseudo_loss_cfg['lambda_depth']}, "
            f"lambda_pose={resolved_pseudo_loss_cfg['lambda_pose']}, "
            f"lambda_exp={resolved_pseudo_loss_cfg['lambda_exp']}, "
            f"trans_weight={resolved_pseudo_loss_cfg['trans_weight']}, "
            f"lambda_abs_pose={resolved_pseudo_loss_cfg['lambda_abs_pose']}, "
            f"lambda_abs_t={resolved_pseudo_loss_cfg['lambda_abs_t']}, "
            f"lambda_abs_r={resolved_pseudo_loss_cfg['lambda_abs_r']}, "
            f"abs_pose_robust={resolved_pseudo_loss_cfg['abs_pose_robust']}"
        )

        history_rows: list[dict[str, Any]] = []
        topology_mode = str(getattr(cfg, "topology_mode", "side_branch"))
        pseudo_window_equivalence = bool(getattr(cfg, "pseudo_window_equivalence", False))
        maintenance_source = str(getattr(cfg, "gaussian_maintenance_source", "all_views") or "all_views")
        propagate_pseudo_delta = bool(getattr(cfg, "propagate_pseudo_delta_to_neighbors", True))
        real_pose_enabled = bool(getattr(cfg, "update_real_pose", False))
        real_exposure_enabled = bool(update_real_exposure)
        pseudo_pose_enabled = bool(getattr(cfg, "update_pseudo_pose", True))
        pseudo_exposure_enabled = bool(getattr(cfg, "update_pseudo_pose", True))
        Log(
            f"{log_prefix} start exact joint loop: real_window={len(current_window)} extra_real={len(extra_real_candidates)} pseudo={len(pseudo_records)} pseudo_per_step={num_pseudo_views} topology={topology_mode} pseudo_window_equivalence={pseudo_window_equivalence}"
        )

        for step_idx in range(1, int(cfg.num_iterations) + 1):
            self.iteration_count += 1
            total_loss = torch.zeros((), device="cuda", dtype=torch.float32)
            real_loss_sum = torch.zeros_like(total_loss)
            pseudo_loss_sum = torch.zeros_like(total_loss)
            pseudo_pose_loss_sum = torch.zeros_like(total_loss)
            pseudo_scene_loss_sum = torch.zeros_like(total_loss)
            real_views_for_pose = [self.cameras[idx] for idx in current_window]
            sampled_extra_indices = []
            sampled_pseudo = self._sample_pseudo_records(rng, pseudo_records, num_pseudo_views)
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []
            view_source_tags: list[str] = []
            sampled_pseudo_ids = [int(rec.sample_id) for rec in sampled_pseudo]
            split_authority = bool(cfg.split_pseudo_authority) and not pseudo_window_equivalence
            scene_mask_mode = "none" if pseudo_window_equivalence else str(cfg.pseudo_scene_mask_mode or "all_valid")
            scene_active_count = 0
            extra_real_views = int(getattr(cfg, "extra_real_views", 0))

            self.gaussians.optimizer.zero_grad(set_to_none=True)
            for optimizer in (real_pose_optimizer, pseudo_pose_optimizer, pseudo_exposure_optimizer):
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)

            for cam_idx in current_window:
                viewpoint = self.cameras[cam_idx]
                # CRITICAL FIX: Apply pose delta before render so gradient flows to theta/rho
                if bool(getattr(cfg, "update_real_pose", True)):
                    apply_pose_delta_before_render_(viewpoint)
                render_pkg = render(viewpoint, self.gaussians, self.pipeline_params, self.background)
                real_loss = get_loss_mapping(self.config, render_pkg["render"], viewpoint, depth=render_pkg["depth"], monodepth=True)
                real_loss_sum = real_loss_sum + real_loss
                total_loss = total_loss + float(cfg.lambda_real) * real_loss
                viewspace_point_tensor_acm.append(render_pkg["viewspace_points"])
                visibility_filter_acm.append(render_pkg["visibility_filter"])
                radii_acm.append(render_pkg["radii"])
                n_touched_acm.append(render_pkg["n_touched"])
                view_source_tags.append("real_window")

            if extra_real_candidates and extra_real_views > 0:
                sample_n = min(extra_real_views, len(extra_real_candidates))
                sampled_extra_indices = rng.sample(extra_real_candidates, sample_n)
                for cam_idx in sampled_extra_indices:
                    viewpoint = self.cameras[cam_idx]
                    # CRITICAL FIX: Apply pose delta before render so gradient flows to theta/rho
                    if bool(getattr(cfg, "update_real_pose", True)):
                        apply_pose_delta_before_render_(viewpoint)
                    render_pkg = render(viewpoint, self.gaussians, self.pipeline_params, self.background)
                    real_loss = get_loss_mapping(self.config, render_pkg["render"], viewpoint, depth=render_pkg["depth"], monodepth=True)
                    real_loss_sum = real_loss_sum + real_loss
                    total_loss = total_loss + float(cfg.lambda_real) * real_loss
                    viewspace_point_tensor_acm.append(render_pkg["viewspace_points"])
                    visibility_filter_acm.append(render_pkg["visibility_filter"])
                    radii_acm.append(render_pkg["radii"])
                    n_touched_acm.append(render_pkg["n_touched"])
                    view_source_tags.append("real_extra")

            last_pseudo_stats = None
            last_pseudo_pose_stats = None
            last_pseudo_scene_stats = None
            for record in sampled_pseudo:
                # CRITICAL FIX: Apply pose delta before render so gradient flows to theta/rho
                apply_pose_delta_before_render_(record.viewpoint)
                render_pkg = render(record.viewpoint, self.gaussians, self.pipeline_params, self.background)
                if not split_authority:
                    pseudo_loss, pseudo_stats = compute_backend_pseudo_exact_loss(
                        render_rgb=render_pkg["render"],
                        render_depth=render_pkg["depth"],
                        record=record,
                        cfg=loss_cfg,
                    )
                    pseudo_loss_sum = pseudo_loss_sum + pseudo_loss
                    total_loss = total_loss + float(cfg.lambda_pseudo) * pseudo_loss
                    last_pseudo_stats = pseudo_stats
                    viewspace_point_tensor_acm.append(render_pkg["viewspace_points"])
                    visibility_filter_acm.append(render_pkg["visibility_filter"])
                    radii_acm.append(render_pkg["radii"])
                    n_touched_acm.append(render_pkg["n_touched"])
                    view_source_tags.append("pseudo")
                    continue

                pseudo_pose_loss, pseudo_pose_stats = compute_backend_pseudo_exact_loss(
                    render_rgb=render_pkg["render"],
                    render_depth=render_pkg["depth"],
                    record=record,
                    cfg=loss_cfg,
                )
                pseudo_pose_loss_sum = pseudo_pose_loss_sum + pseudo_pose_loss
                pseudo_loss_sum = pseudo_loss_sum + pseudo_pose_loss
                last_pseudo_pose_stats = pseudo_pose_stats

                scene_record = self._scene_record_for_mode(record, scene_mask_mode)
                if scene_record is not None:
                    pseudo_scene_loss, pseudo_scene_stats = compute_backend_pseudo_exact_loss(
                        render_rgb=render_pkg["render"],
                        render_depth=render_pkg["depth"],
                        record=scene_record,
                        cfg=loss_cfg,
                    )
                    pseudo_scene_loss_sum = pseudo_scene_loss_sum + pseudo_scene_loss
                    pseudo_loss_sum = pseudo_loss_sum + pseudo_scene_loss
                    last_pseudo_scene_stats = pseudo_scene_stats
                    last_pseudo_stats = pseudo_scene_stats
                    scene_active_count += 1
                    viewspace_point_tensor_acm.append(render_pkg["viewspace_points"])
                    visibility_filter_acm.append(render_pkg["visibility_filter"])
                    radii_acm.append(render_pkg["radii"])
                    n_touched_acm.append(render_pkg["n_touched"])
                    view_source_tags.append("pseudo")
                else:
                    last_pseudo_stats = pseudo_pose_stats

            scaling = self.gaussians.get_scaling
            # Use scale_reg_loss function for proper scale regularization
            max_scale_val = float(getattr(cfg, 'max_scale', 0.0) or 0.0)
            scale_loss = scale_reg_loss(self.gaussians, max_scale=max_scale_val if max_scale_val > 0 else None)
            # Keep isotropic_loss for backward compatibility with history format
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1)).mean()
            # Use scale_loss in total_loss if lambda_scale is set, otherwise use isotropic_weight
            lambda_scale_val = float(getattr(cfg, 'lambda_scale', 0.0) or 0.0)
            scale_weight = lambda_scale_val if lambda_scale_val > 0 else float(cfg.isotropic_weight)
            pseudo_objective = pseudo_pose_loss_sum + pseudo_scene_loss_sum if split_authority else pseudo_loss_sum

            # FIX: Simplified unified backward - no longer zero Gaussians/pseudo grads
            # Previous split_authority logic was dropping pseudo gradients incorrectly
            total_loss = (
                float(cfg.lambda_real) * real_loss_sum
                + float(cfg.lambda_pseudo) * pseudo_objective
                + scale_weight * scale_loss
            )
            total_loss.backward()

            # Gauss-Newton pose optimization (before no_grad, needs gradient)
            use_gn = bool(getattr(cfg, 'use_gauss_newton', False))
            gn_every = int(getattr(cfg, 'gn_every_n_steps', 1))
            gn_applied_this_step = False

            if use_gn and sampled_pseudo and (step_idx % gn_every == 0):
                gn_max_iters = int(getattr(cfg, 'gn_max_iters', 5))
                gn_damping = float(getattr(cfg, 'gn_damping', 0.01))

                for record in sampled_pseudo:
                    # GN loss function: render + compute loss
                    def gn_loss_fn(vp):
                        apply_pose_delta_before_render_(vp)
                        render_pkg = render(vp, self.gaussians, self.pipeline_params, self.background)
                        loss, _ = compute_backend_pseudo_exact_loss(
                            render_rgb=render_pkg["render"],
                            render_depth=render_pkg["depth"],
                            record=record,
                            cfg=loss_cfg,
                            viewpoint=vp,
                        )
                        return loss

                    gn_optimizer = GaussNewtonPoseOptimizer(
                        max_iters=gn_max_iters,
                        damping=gn_damping,
                    )
                    converged, gn_stats = gn_optimizer.optimize(record.viewpoint, gn_loss_fn, verbose=False)
                    # GN directly updates theta/rho in-place
                gn_applied_this_step = True

            with torch.no_grad():
                need_gaussian_maintenance = bool(cfg.enable_densify or cfg.enable_prune or cfg.enable_opacity_reset)
                maintenance_indices = list(range(len(viewspace_point_tensor_acm)))
                if maintenance_source == "real_only":
                    maintenance_indices = [
                        idx for idx, tag in enumerate(view_source_tags)
                        if tag in {"real_window", "real_extra"}
                    ]
                if need_gaussian_maintenance:
                    for idx in maintenance_indices:
                        visibility_filter = visibility_filter_acm[idx]
                        self.gaussians.max_radii2D[visibility_filter] = torch.max(
                            self.gaussians.max_radii2D[visibility_filter],
                            radii_acm[idx][visibility_filter],
                        )
                        self.gaussians.add_densification_stats(
                            viewspace_point_tensor_acm[idx], visibility_filter
                        )

                if cfg.enable_densify and maintenance_indices and (
                    self.iteration_count % self.gaussian_update_every == self.gaussian_update_offset
                ):
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )
                if cfg.enable_opacity_reset and maintenance_indices and (self.iteration_count % self.gaussian_reset) == 0:
                    reset_visibility = [visibility_filter_acm[idx] for idx in maintenance_indices]
                    self.gaussians.reset_opacity_nonvisible(reset_visibility)
                if cfg.enable_prune and len(current_window) == self.config["Training"]["window_size"]:
                    self.gaussians.n_obs.fill_(0)
                    for touched in n_touched_acm[: len(current_window)]:
                        self.gaussians.n_obs += (touched > 0).cpu()

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)

                # Keep real-KF Adam independent from pseudo GN.
                # update_real_pose governs whether the pseudo loop may touch real KF pose/exposure;
                # it must not disable the original S3PO real mapping loop outside this block.
                if real_pose_optimizer is not None:
                    real_pose_optimizer.step()
                    real_pose_optimizer.zero_grad(set_to_none=True)

                # Pseudo exposure uses Adam even when pseudo pose uses GN.
                if pseudo_exposure_optimizer is not None:
                    pseudo_exposure_optimizer.step()
                    pseudo_exposure_optimizer.zero_grad(set_to_none=True)

                # Pseudo pose: Adam only when GN is not applied this step.
                if not (use_gn and gn_applied_this_step):
                    if pseudo_pose_optimizer is not None:
                        pseudo_pose_optimizer.step()
                        pseudo_pose_optimizer.zero_grad(set_to_none=True)
                elif pseudo_pose_optimizer is not None:
                    pseudo_pose_optimizer.zero_grad(set_to_none=True)

                # Fold pose delta into R/T
                if cfg.update_real_pose:
                    frames_to_optimize = int(self.config["Training"].get("pose_window", len(real_views_for_pose)))
                    for viewpoint in real_views_for_pose[:frames_to_optimize]:
                        # KF0 skip removed: DL3DV poses from COLMAP reconstruction, not true GT
                        # if getattr(viewpoint, "uid", None) == 0:
                        #     continue
                        update_pose(viewpoint)

                if cfg.update_pseudo_pose:
                    for record in sampled_pseudo:
                        if cfg.update_real_pose and propagate_pseudo_delta:
                            self._propagate_pseudo_pose_to_neighbors_(record)
                        self._fold_pseudo_pose_residual_(record.viewpoint)

            row = {
                "iteration": int(step_idx),
                "loss_total": float(total_loss.detach().item()),
                "loss_real": float(real_loss_sum.detach().item()),
                "loss_pseudo": float(pseudo_loss_sum.detach().item()),
                "loss_pseudo_member": float(pseudo_loss_sum.detach().item()) if not split_authority else None,
                "loss_pseudo_pose": float(pseudo_pose_loss_sum.detach().item()) if split_authority else float(pseudo_loss_sum.detach().item()),
                "loss_pseudo_scene": float(pseudo_scene_loss_sum.detach().item()) if split_authority else 0.0,
                "loss_isotropic": float(isotropic_loss.detach().item()),
                "match_real_loss_weights": bool(resolved_pseudo_loss_cfg.get("match_real_loss_weights", False)),
                "resolved_beta_rgb": float(resolved_pseudo_loss_cfg.get("beta_rgb", 0.0)),
                "resolved_lambda_depth": float(resolved_pseudo_loss_cfg.get("lambda_depth", 0.0)),
                "real_mapping_alpha": float(resolved_pseudo_loss_cfg.get("real_mapping_alpha", self.config.get("Training", {}).get("alpha", 0.95))),
                "sampled_pseudo_ids": sampled_pseudo_ids,
                "sampled_extra_real_indices": sampled_extra_indices,
                "split_pseudo_authority": split_authority,
                "pseudo_scene_mask_mode": scene_mask_mode,
                "pseudo_scene_active_count": int(scene_active_count),
                "use_gauss_newton": use_gn,
                "gn_applied_this_step": gn_applied_this_step,
                "topology_mode": topology_mode,
                "pseudo_window_equivalence": pseudo_window_equivalence,
                "gaussian_maintenance_source": maintenance_source,
                "num_real_window_members": len(current_window),
                "num_extra_real_members": len(sampled_extra_indices),
                "num_pseudo_members": len(sampled_pseudo),
                "num_real_pose_optimized": min(int(self.config["Training"].get("pose_window", len(real_views_for_pose))), len(real_views_for_pose)) if real_pose_enabled else 0,
                "num_real_exposure_optimized": len(real_views_for_pose) if real_exposure_enabled else 0,
                "num_pseudo_pose_optimized": len(sampled_pseudo) if pseudo_pose_enabled else 0,
                "num_pseudo_exposure_optimized": len(sampled_pseudo) if pseudo_exposure_enabled else 0,
                "neighbor_pose_propagation_enabled": bool(propagate_pseudo_delta),
                "neighbor_pose_propagation_applied": bool(propagate_pseudo_delta and real_pose_enabled and pseudo_pose_enabled and sampled_pseudo),
            }
            if last_pseudo_stats is not None:
                for key in ["loss_rgb", "loss_depth", "effective_mask_nonzero_ratio", "effective_mask_mean"]:
                    if key in last_pseudo_stats:
                        row[f"pseudo_{key}"] = last_pseudo_stats[key]
            if last_pseudo_pose_stats is not None:
                for key in ["loss_rgb", "loss_depth", "effective_mask_nonzero_ratio", "effective_mask_mean"]:
                    if key in last_pseudo_pose_stats:
                        row[f"pseudo_pose_{key}"] = last_pseudo_pose_stats[key]
            if last_pseudo_scene_stats is not None:
                for key in ["loss_rgb", "loss_depth", "effective_mask_nonzero_ratio", "effective_mask_mean"]:
                    if key in last_pseudo_scene_stats:
                        row[f"pseudo_scene_{key}"] = last_pseudo_scene_stats[key]
            history_rows.append(row)

        result = {
            "cfg": asdict(cfg),
            "resolved_pseudo_loss_cfg": resolved_pseudo_loss_cfg,
            "num_pseudo_records": len(pseudo_records),
            "num_real_valid": len(valid_real_indices),
            "num_extra_real_candidates": len(extra_real_candidates),
            "current_window": current_window,
            "topology_mode": topology_mode,
            "pseudo_window_equivalence": pseudo_window_equivalence,
            "gaussian_maintenance_source": maintenance_source,
            "match_real_loss_weights": bool(resolved_pseudo_loss_cfg.get("match_real_loss_weights", False)),
            "history": history_rows,
            "iteration_count_end": int(self.iteration_count),
            "runner_tag": runner_tag,
            **source_info,
        }
        if cfg.output_dir:
            outdir = Path(cfg.output_dir)
            outdir.mkdir(parents=True, exist_ok=True)
            with open(outdir / "brpo_pseudo_history.json", "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
        Log(f"{log_prefix} done exact joint loop")
        return result

    def run_exact_pseudo_continuation(
        self,
        cfg: BRPOContinuationConfig,
    ) -> dict[str, Any]:
        bundle = load_pseudo_bundle_from_stageA_history(cfg.stageA_history_json, require_exact_upstream=True)
        pseudo_records = build_records_from_pseudo_bundle(bundle)
        if not pseudo_records:
            raise RuntimeError(f"No pseudo records resolved from {cfg.stageA_history_json}")

        history_args = dict(bundle.args or {})
        valid_real_indices = self._valid_real_indices()
        current_window = self._normalize_current_window_order([idx for idx in self.current_window if idx in valid_real_indices])
        current_window_set = set(current_window)
        extra_real_candidates = [idx for idx in valid_real_indices if idx not in current_window_set]
        return self._run_joint_pseudo_engine(
            cfg=cfg,
            pseudo_records=pseudo_records,
            history_args=history_args,
            current_window=current_window,
            extra_real_candidates=extra_real_candidates,
            source_info={
                "stageA_history_json": cfg.stageA_history_json,
                "bundle_source_history_json": str(bundle.source_history_json),
            },
            runner_tag="continuation",
            log_prefix="[BRPOContinuation]",
        )

    def run_runtime_pseudo_mapping(
        self,
        cfg: BRPOMappingConfig,
        runtime_pseudo_records: list[BackendPseudoViewRecord] | dict[int, BackendPseudoViewRecord],
    ) -> dict[str, Any]:
        if isinstance(runtime_pseudo_records, dict):
            pseudo_records = list(runtime_pseudo_records.values())
        else:
            pseudo_records = list(runtime_pseudo_records)
        current_window = list(self.current_window)
        valid_real_indices = self._valid_real_indices()
        current_window_set = set(current_window)
        extra_real_candidates = [idx for idx in valid_real_indices if idx not in current_window_set]
        return self._run_joint_pseudo_engine(
            cfg=cfg,
            pseudo_records=pseudo_records,
            history_args={},
            current_window=current_window,
            extra_real_candidates=extra_real_candidates,
            source_info={
                "runtime_pseudo_frame_ids": [int(record.frame_id) for record in pseudo_records],
            },
            runner_tag="online_mapping",
            log_prefix="[BRPOMapping]",
        )


def run_brpo_pseudo_continuation(
    *,
    config: dict[str, Any],
    gaussians,
    pipeline_params,
    opt_params,
    background: torch.Tensor,
    cameras: dict[int, Any],
    kf_indices: list[int],
    current_window: list[int] | None,
    continuation_cfg: BRPOContinuationConfig,
    iteration_start: int = 0,
    gaussian_update_every: int = 100,
    gaussian_update_offset: int = 0,
    gaussian_th: float = 0.7,
    gaussian_extent: float = 6.0,
    gaussian_reset: int = 3000,
    size_threshold: float | None = None,
) -> dict[str, Any]:
    runner = BRPOBackEndContinuation(
        config=config,
        gaussians=gaussians,
        pipeline_params=pipeline_params,
        opt_params=opt_params,
        background=background,
        cameras=cameras,
        kf_indices=kf_indices,
        current_window=current_window,
        iteration_start=iteration_start,
        gaussian_update_every=gaussian_update_every,
        gaussian_update_offset=gaussian_update_offset,
        gaussian_th=gaussian_th,
        gaussian_extent=gaussian_extent,
        gaussian_reset=gaussian_reset,
        size_threshold=size_threshold,
    )
    return runner.run_exact_pseudo_continuation(continuation_cfg)


def run_brpo_pseudo_mapping(
    *,
    config: dict[str, Any],
    gaussians,
    pipeline_params,
    opt_params,
    background: torch.Tensor,
    cameras: dict[int, Any],
    kf_indices: list[int],
    current_window: list[int] | None,
    mapping_cfg: BRPOMappingConfig,
    runtime_pseudo_records: list[BackendPseudoViewRecord] | dict[int, BackendPseudoViewRecord],
    iteration_start: int = 0,
    gaussian_update_every: int = 100,
    gaussian_update_offset: int = 0,
    gaussian_th: float = 0.7,
    gaussian_extent: float = 6.0,
    gaussian_reset: int = 3000,
    size_threshold: float | None = None,
) -> dict[str, Any]:
    runner = BRPOBackEndContinuation(
        config=config,
        gaussians=gaussians,
        pipeline_params=pipeline_params,
        opt_params=opt_params,
        background=background,
        cameras=cameras,
        kf_indices=kf_indices,
        current_window=current_window,
        iteration_start=iteration_start,
        gaussian_update_every=gaussian_update_every,
        gaussian_update_offset=gaussian_update_offset,
        gaussian_th=gaussian_th,
        gaussian_extent=gaussian_extent,
        gaussian_reset=gaussian_reset,
        size_threshold=size_threshold,
    )
    return runner.run_runtime_pseudo_mapping(mapping_cfg, runtime_pseudo_records)
