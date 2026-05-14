import random
import time
import json

import torch
import torch.multiprocessing as mp
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path
from typing import Any

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim, masked_l1_loss, masked_ssim
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_mapping
from utils.init_pose import save_depth_comparison
from utils.slam_backend_brpo import BRPOMappingConfig, run_brpo_pseudo_mapping

from pseudo_branch.integration import (
    RuntimeExactBackendConfig,
    RuntimeSlotSelectorConfig,
    build_runtime_exact_backend_bundle,
    build_runtime_exact_signal_bundle,
    build_runtime_pseudo_record_bundle,
    select_runtime_pseudo_slots,
)


class BackEnd(mp.Process):
    def __init__(self, config, save_dir=None):
        super().__init__()
        self.config = config
        self.gaussians = None
        self.pipeline_params = None
        self.opt_params = None
        self.background = None
        # CPU-only background payload from the parent process.  The spawned child
        # rebuilds the CUDA tensor in run() after selecting the mapped CUDA device.
        self.background_cpu = None
        self._target_cuda_device = 0
        self.cameras_extent = None
        self.frontend_queue = None
        self.backend_queue = None
        self.live_mode = False
        self.save_dir = save_dir

        self.pause = False
        self.device = "cuda"
        self.dtype = torch.float32
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.last_sent = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None
        self.theta = 0
        self.brpo_online_mapping_cfg = None
        self.brpo_online_slot_selector_cfg = None
        self.brpo_runtime_camera_states = {}
        self.brpo_runtime_seen_gap_keys = set()
        self.brpo_runtime_pseudo_records = {}
        self.brpo_runtime_matcher = None
        self.brpo_difix_model = None  # Difix model for RGB restoration

    def set_hyperparams(self):
        self.save_results = self.config["Results"]["save_results"]

        self.init_itr_num = self.config["Training"]["init_itr_num"]
        self.init_gaussian_update = self.config["Training"]["init_gaussian_update"]
        self.init_gaussian_reset = self.config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th = self.config["Training"]["init_gaussian_th"]
        self.init_gaussian_extent = (
            self.cameras_extent * self.config["Training"]["init_gaussian_extent"]
        )
        self.mapping_itr_num = self.config["Training"]["mapping_itr_num"]
        self.global_BA_itr_num = self.config["Training"]["global_BA_itr_num"]
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"]
        self.gaussian_th = self.config["Training"]["gaussian_th"]
        self.gaussian_extent = (
            self.cameras_extent * self.config["Training"]["gaussian_extent"]
        )
        self.gaussian_reset = self.config["Training"]["gaussian_reset"]
        self.size_threshold = self.config["Training"]["size_threshold"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = (
            self.config["Dataset"]["single_thread"]
            if "single_thread" in self.config["Dataset"]
            else False
        )
        self.brpo_online_mapping_cfg = self._resolve_brpo_online_mapping_cfg()
        if self.brpo_online_mapping_cfg is not None:
            self.brpo_online_slot_selector_cfg = RuntimeSlotSelectorConfig(
                placement_mode=str(self.brpo_online_mapping_cfg.get("placement_mode", "midpoint_only")),
                max_pseudo_per_gap=int(self.brpo_online_mapping_cfg.get("max_pseudo_per_gap", 1)),
            )
        else:
                        self.brpo_online_slot_selector_cfg = None

        # NOTE: do NOT load Difix here in the parent process.
        # slam.py calls self.backend.set_hyperparams() before spawning backend_process,
        # so attaching a live DiffusionPipeline to self would force multiprocessing
        # spawn to pickle/unpickle the custom VAE object graph. Difix custom VAE code
        # monkey-patches encoder/decoder forward methods, which is not spawn-safe here.
        # Load lazily inside the backend child process right before the first exact bundle
        # that actually needs restoration.
    # Insert new Gaussians into the Gaussian scene based on the new keyframe's viewpoint and geometry
    def add_next_kf(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None):
        self.gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map
        )
        
    def reset(self):
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None
        self.brpo_runtime_camera_states = {}
        self.brpo_runtime_seen_gap_keys = set()
        self.brpo_runtime_pseudo_records = {}

        # remove all gaussians
        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

    def _resolve_brpo_online_mapping_cfg(self):
        result_cfg = self.config.get("Results", {}).get("brpo_online_mapping", {}) or {}
        if not bool(result_cfg.get("enabled", False)):
            return None
        debug_root = result_cfg.get("debug_export_root")
        if not debug_root:
            base_dir = Path(self.save_dir) if self.save_dir else Path(".")
            debug_root = str(base_dir / "brpo_online_mapping_debug")
        topology_mode = str(result_cfg.get("topology_mode", "side_branch"))
        default_pseudo_window_equivalence = topology_mode == "joint_primary"
        default_extra_real_views = 2 if topology_mode == "joint_primary" else 0
        default_update_real_exposure = bool(result_cfg.get("update_real_pose", False))
        if topology_mode == "joint_primary":
            default_update_real_exposure = True
        default_maintenance_source = "real_only" if topology_mode == "joint_primary" else "all_views"
        default_propagate_pseudo_delta = False if default_pseudo_window_equivalence else True
        default_split_pseudo_authority = False if default_pseudo_window_equivalence else True
        match_real_loss_weights = bool(result_cfg.get("match_real_loss_weights", default_pseudo_window_equivalence))
        resolved_use_depth = bool(result_cfg.get("use_depth", True))
        training_alpha = float(self.config.get("Training", {}).get("alpha", 0.95))
        resolved_beta_rgb = float(result_cfg.get("beta_rgb", 0.7))
        resolved_lambda_depth = float(result_cfg.get("lambda_depth", 1.0))
        if match_real_loss_weights:
            resolved_beta_rgb = training_alpha
            resolved_lambda_depth = max(0.0, 1.0 - training_alpha) if resolved_use_depth else 0.0
        return {
            "trigger": str(result_cfg.get("trigger", "keyframe")),
            "topology_mode": topology_mode,
            "placement_mode": str(result_cfg.get("placement_mode", "midpoint_only")),
            "max_pseudo_per_gap": int(result_cfg.get("max_pseudo_per_gap", 1)),
            "pseudo_map_iters": int(result_cfg.get("pseudo_map_iters", 0)),
            "num_pseudo_views_per_step": int(result_cfg.get("num_pseudo_views_per_step", 1)),
            "enable_pseudo_gradient": bool(result_cfg.get("enable_pseudo_gradient", False)),
            "extra_real_views": int(result_cfg.get("extra_real_views", default_extra_real_views)),
            "pseudo_window_equivalence": bool(result_cfg.get("pseudo_window_equivalence", default_pseudo_window_equivalence)),
            "propagate_pseudo_delta_to_neighbors": bool(result_cfg.get("propagate_pseudo_delta_to_neighbors", default_propagate_pseudo_delta)),
            "update_real_exposure": bool(result_cfg.get("update_real_exposure", default_update_real_exposure)),
            "gaussian_maintenance_source": str(result_cfg.get("gaussian_maintenance_source", default_maintenance_source)),
            "joint_primary_run_legacy_prune": bool(result_cfg.get("joint_primary_run_legacy_prune", True)),
            "joint_primary_real_densify_fallback": bool(result_cfg.get("joint_primary_real_densify_fallback", False)),
            "lambda_real": float(result_cfg.get("lambda_real", 1.0)),
            "lambda_pseudo": float(result_cfg.get("lambda_pseudo", 1.0)),
            "lambda_depth": resolved_lambda_depth,
            "match_real_loss_weights": match_real_loss_weights,
            "beta_rgb": resolved_beta_rgb,
            "lambda_pose": float(result_cfg.get("lambda_pose", 0.01)),
            "lambda_exp": float(result_cfg.get("lambda_exp", 0.001)),
            "trans_weight": float(result_cfg.get("trans_weight", 1.0)),
            "lambda_abs_pose": float(result_cfg.get("lambda_abs_pose", 0.0)),
            "lambda_abs_t": float(result_cfg.get("lambda_abs_t", 3.0)),
            "lambda_abs_r": float(result_cfg.get("lambda_abs_r", 0.1)),
            "abs_pose_robust": str(result_cfg.get("abs_pose_robust", "charbonnier")),
            "update_real_pose": bool(result_cfg.get("update_real_pose", False)),
            "update_pseudo_pose": bool(result_cfg.get("update_pseudo_pose", True)),
            "use_depth": resolved_use_depth,
            "split_pseudo_authority": bool(result_cfg.get("split_pseudo_authority", default_split_pseudo_authority)),
            "pseudo_scene_mask_mode": str(result_cfg.get("pseudo_scene_mask_mode", "both_only")),
            "isotropic_weight": float(result_cfg.get("isotropic_weight", 10.0)),
            "seed": int(result_cfg.get("seed", 0)),
            "debug_export_root": str(debug_root),
            "matcher_mode": str(result_cfg.get("matcher_mode", "sparse_desc_2d")),
            "matcher_model_name": str(result_cfg.get("matcher_model_name", "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric")),
            "matcher_device": str(result_cfg.get("matcher_device", "cuda")),
            "dense3d_conf_quantile": float(result_cfg.get("dense3d_conf_quantile", 0.15)),
            "tau_reproj_px": float(result_cfg.get("tau_reproj_px", 20.0)),
            "tau_rel_depth": float(result_cfg.get("tau_rel_depth", 1.0)),
            "depth_generation_mode": str(result_cfg.get("depth_generation_mode", "projected")),
            "pseudo_rgb_source": str(result_cfg.get("pseudo_rgb_source", "render")),
            "depth_loss_mode": str(result_cfg.get("depth_loss_mode", "exact_shared_cm_v1")),
            "lambda_scale": float(result_cfg.get("lambda_scale", 0.01)),
            "max_scale": float(result_cfg.get("max_scale", 0.0)) if result_cfg.get("max_scale") is not None else None,
            "use_gauss_newton": bool(result_cfg.get("use_gauss_newton", False)),
            "gn_max_iters": int(result_cfg.get("gn_max_iters", 5)),
            "gn_damping": float(result_cfg.get("gn_damping", 0.01)),
            "gn_every_n_steps": int(result_cfg.get("gn_every_n_steps", 1)),

            "enable_densify": bool(result_cfg.get("enable_densify", True)),
            "enable_opacity_reset": bool(result_cfg.get("enable_opacity_reset", True)),
            # Difix restoration parameters
            "use_difix_restoration": bool(result_cfg.get("use_difix_restoration", False)),
            "difix_model_name": str(result_cfg.get("difix_model_name", "nvidia/difix_ref")),
            "difix_model_path": result_cfg.get("difix_model_path"),
            "difix_timestep": int(result_cfg.get("difix_timestep", 100)),
            "difix_prompt": str(result_cfg.get("difix_prompt", "")),
            "difix_height": int(result_cfg.get("difix_height", 512)),
            "difix_width": int(result_cfg.get("difix_width", 512)),
            "difix_fusion_mode": str(result_cfg.get("difix_fusion_mode", "brpo_overlap_confidence")),
            "depth_consistency_tau": float(result_cfg.get("depth_consistency_tau", 0.15)),
            "translation_scale_tau": float(result_cfg.get("translation_scale_tau", 1.0)),
            # Paper route: RGB-only C_m (decoupled from depth verification)
            "rgb_only_verification": bool(result_cfg.get("rgb_only_verification", False)),
            "rgb_only_support_mode": str(result_cfg.get("rgb_only_support_mode", "reciprocal_seed")),
            "cm_dense_point_radius": int(result_cfg.get("cm_dense_point_radius", 2)),
            "cm_dense_blur_sigma": float(result_cfg.get("cm_dense_blur_sigma", 2.0)),
            "cm_dense_blur_kernel": int(result_cfg.get("cm_dense_blur_kernel", 0)),
            "cm_dense_corr_threshold": float(result_cfg.get("cm_dense_corr_threshold", 0.15)),
            "cm_dense_seed_mode": str(result_cfg.get("cm_dense_seed_mode", "binary")),
            "cm_dense_normalize_mode": str(result_cfg.get("cm_dense_normalize_mode", "max")),
            # C_m local expansion parameters
            "cm_expansion_mode": str(result_cfg.get("cm_expansion_mode", "none")),
            "cm_expansion_radius": int(result_cfg.get("cm_expansion_radius", 1)),
            "cm_expansion_weight": float(result_cfg.get("cm_expansion_weight", 0.5)),
            "cm_expansion_tau_rgb_l1": float(result_cfg.get("cm_expansion_tau_rgb_l1", 0.08)),
            "cm_expansion_tau_depth_rel": float(result_cfg.get("cm_expansion_tau_depth_rel", 0.05)),
            "cm_expansion_min_seed_conf": float(result_cfg.get("cm_expansion_min_seed_conf", 0.0)),
            "cm_expansion_min_expanded_conf": float(result_cfg.get("cm_expansion_min_expanded_conf", 0.05)),
            "cm_expanded_both_weight": float(result_cfg.get("cm_expanded_both_weight", 0.6)),
            "cm_raw_exp_agree_weight": float(result_cfg.get("cm_raw_exp_agree_weight", 0.5)),
            "cm_expanded_single_weight": float(result_cfg.get("cm_expanded_single_weight", 0.25)),
            "cm_expansion_apply_to_depth_scope": bool(result_cfg.get("cm_expansion_apply_to_depth_scope", False)),
            # ABLATION: Disable confidence mask (A2)
            "disable_confidence_mask": bool(result_cfg.get("disable_confidence_mask", False)),
            # Color refinement with masked pseudo
            "color_refinement_use_pseudo": bool(result_cfg.get("color_refinement_use_pseudo", False)),
            "color_refinement_pseudo_ratio": float(result_cfg.get("color_refinement_pseudo_ratio", 0.5)),
            "color_refinement_pseudo_weight": float(result_cfg.get("color_refinement_pseudo_weight", 1.0)),
            "color_refinement_pseudo_mask_source": str(result_cfg.get("color_refinement_pseudo_mask_source", "confidence_mask")),
            "color_refinement_log_every": int(result_cfg.get("color_refinement_log_every", 200)),
        }

    def _update_brpo_event_summary(self, cur_frame_idx, patch: dict[str, Any]):
        cfg = self.brpo_online_mapping_cfg
        if cfg is None:
            return
        event_root = Path(cfg["debug_export_root"]) / f"event_kf_{int(cur_frame_idx):04d}"
        event_root.mkdir(parents=True, exist_ok=True)
        event_summary_path = event_root / "event_summary.json"
        if event_summary_path.exists():
            payload = json.loads(event_summary_path.read_text(encoding="utf-8"))
        else:
            payload = {
                "trigger_keyframe": int(cur_frame_idx),
                "current_window": [int(x) for x in self.current_window],
                "num_slots": 0,
                "slots": [],
            }
        payload.update(patch)
        with open(event_summary_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _ensure_brpo_runtime_matcher(self, exact_cfg: RuntimeExactBackendConfig):
        if self.brpo_runtime_matcher is None:
            from pseudo_branch.common import build_pair_matcher

            self.brpo_runtime_matcher = build_pair_matcher(
                matcher_mode=exact_cfg.matcher_mode,
                model_name=exact_cfg.matcher_model_name,
                device=exact_cfg.matcher_device,
                dense3d_conf_quantile=float(exact_cfg.dense3d_conf_quantile),
            )
        return self.brpo_runtime_matcher


    def _ensure_brpo_difix_model_loaded(self):
        cfg = self.brpo_online_mapping_cfg
        if cfg is None or not bool(cfg.get("use_difix_restoration", False)):
            return None
        if self.brpo_difix_model is None:
            from online_mapping.difix_loader import load_difix_model

            target_device = None
            if bool(cfg.get("difix_enforce_backend_device", True)) and torch.cuda.is_available():
                target_device = torch.device(f"cuda:{int(getattr(self, '_target_cuda_device', 0))}")

            self.brpo_difix_model = load_difix_model(
                model_name=str(cfg.get("difix_model_name", "nvidia/difix_ref")),
                model_path=cfg.get("difix_model_path"),
                timestep=int(cfg.get("difix_timestep", 100)),
                target_device=target_device,
            )
            device_msg = self.brpo_difix_model.get("device", target_device) if isinstance(self.brpo_difix_model, dict) else target_device
            Log(f"[BRPOOnlineMapping] Difix model loaded in backend child on {device_msg}")
        return self.brpo_difix_model

    def _maybe_prepare_brpo_runtime_slots(self, cur_frame_idx):
        cfg = self.brpo_online_mapping_cfg
        if cfg is None or str(cfg.get("trigger", "keyframe")) != "keyframe":
            return None
        available_ids = [
            int(fid)
            for fid, state in self.brpo_runtime_camera_states.items()
            if not bool(state.get("is_keyframe", False))
        ]
        slots = select_runtime_pseudo_slots(
            current_window=list(self.current_window),
            trigger_keyframe=int(cur_frame_idx),
            seen_gap_keys=set(self.brpo_runtime_seen_gap_keys),
            placement_mode=self.brpo_online_slot_selector_cfg.placement_mode,
            max_pseudo_per_gap=self.brpo_online_slot_selector_cfg.max_pseudo_per_gap,
            available_frame_ids=available_ids,
        )
        event_root = Path(cfg["debug_export_root"]) / f"event_kf_{int(cur_frame_idx):04d}"
        event_root.mkdir(parents=True, exist_ok=True)
        if not slots:
            payload = {
                "trigger_keyframe": int(cur_frame_idx),
                "current_window": [int(x) for x in self.current_window],
                "topology_mode": str(cfg.get("topology_mode", "side_branch")),
                "num_slots": 0,
                "slots": [],
            }
            with open(event_root / "event_summary.json", "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            Log(f"[BRPOOnlineMapping] keyframe={int(cur_frame_idx)} no newly-closed midpoint slot")
            return payload

        raw_result_cfg = self.config.get("Results", {}).get("brpo_online_mapping", {}) or {}
        exact_cfg = RuntimeExactBackendConfig(
            matcher_mode=cfg["matcher_mode"],
            matcher_model_name=cfg["matcher_model_name"],
            matcher_device=cfg["matcher_device"],
            dense3d_conf_quantile=float(cfg["dense3d_conf_quantile"]),
            tau_reproj_px=float(cfg["tau_reproj_px"]),
            tau_rel_depth=float(cfg["tau_rel_depth"]),
            depth_generation_mode=str(cfg.get("depth_generation_mode", "projected")),
            pseudo_rgb_source=str(cfg.get("pseudo_rgb_source", "render")),
            # Difix restoration parameters
            difix_prompt=str(cfg.get("difix_prompt", "")),
            difix_height=int(cfg.get("difix_height", 512)),
            difix_width=int(cfg.get("difix_width", 512)),
            # Fusion parameters
            difix_fusion_mode=str(cfg.get("difix_fusion_mode", "brpo_overlap_confidence")),
            depth_consistency_tau=float(cfg.get("depth_consistency_tau", 0.15)),
            translation_scale_tau=float(cfg.get("translation_scale_tau", 1.0)),
            # Paper route: RGB-only C_m
            rgb_only_verification=bool(raw_result_cfg.get("rgb_only_verification", cfg.get("rgb_only_verification", False))),
            rgb_only_support_mode=str(raw_result_cfg.get("rgb_only_support_mode", cfg.get("rgb_only_support_mode", "reciprocal_seed"))),
            cm_dense_point_radius=int(raw_result_cfg.get("cm_dense_point_radius", cfg.get("cm_dense_point_radius", 2))),
            cm_dense_blur_sigma=float(raw_result_cfg.get("cm_dense_blur_sigma", cfg.get("cm_dense_blur_sigma", 2.0))),
            cm_dense_blur_kernel=int(raw_result_cfg.get("cm_dense_blur_kernel", cfg.get("cm_dense_blur_kernel", 0))),
            cm_dense_corr_threshold=float(raw_result_cfg.get("cm_dense_corr_threshold", cfg.get("cm_dense_corr_threshold", 0.15))),
            cm_dense_seed_mode=str(raw_result_cfg.get("cm_dense_seed_mode", cfg.get("cm_dense_seed_mode", "binary"))),
            cm_dense_normalize_mode=str(raw_result_cfg.get("cm_dense_normalize_mode", cfg.get("cm_dense_normalize_mode", "max"))),
            # C_m local expansion parameters
            cm_expansion_mode=str(raw_result_cfg.get("cm_expansion_mode", cfg.get("cm_expansion_mode", "none"))),
            cm_expansion_radius=int(raw_result_cfg.get("cm_expansion_radius", cfg.get("cm_expansion_radius", 1))),
            cm_expansion_weight=float(raw_result_cfg.get("cm_expansion_weight", cfg.get("cm_expansion_weight", 0.5))),
            cm_expansion_tau_rgb_l1=float(raw_result_cfg.get("cm_expansion_tau_rgb_l1", cfg.get("cm_expansion_tau_rgb_l1", 0.08))),
            cm_expansion_tau_depth_rel=float(raw_result_cfg.get("cm_expansion_tau_depth_rel", cfg.get("cm_expansion_tau_depth_rel", 0.05))),
            cm_expansion_min_seed_conf=float(raw_result_cfg.get("cm_expansion_min_seed_conf", cfg.get("cm_expansion_min_seed_conf", 0.0))),
            cm_expansion_min_expanded_conf=float(raw_result_cfg.get("cm_expansion_min_expanded_conf", cfg.get("cm_expansion_min_expanded_conf", 0.05))),
            cm_expanded_both_weight=float(raw_result_cfg.get("cm_expanded_both_weight", cfg.get("cm_expanded_both_weight", 0.6))),
            cm_raw_exp_agree_weight=float(raw_result_cfg.get("cm_raw_exp_agree_weight", cfg.get("cm_raw_exp_agree_weight", 0.5))),
            cm_expanded_single_weight=float(raw_result_cfg.get("cm_expanded_single_weight", cfg.get("cm_expanded_single_weight", 0.25))),
            cm_expansion_apply_to_depth_scope=bool(raw_result_cfg.get("cm_expansion_apply_to_depth_scope", cfg.get("cm_expansion_apply_to_depth_scope", False))),
            # ABLATION: Disable confidence mask (A2)
            disable_confidence_mask=bool(raw_result_cfg.get("disable_confidence_mask", cfg.get("disable_confidence_mask", False))),
        )
        matcher = self._ensure_brpo_runtime_matcher(exact_cfg)
        if bool(cfg.get("use_difix_restoration", False)):
            self._ensure_brpo_difix_model_loaded()
        event_slots = []
        for slot in slots:
            pseudo_state = self.brpo_runtime_camera_states.get(int(slot.frame_id))
            left_state = self.brpo_runtime_camera_states.get(int(slot.left_ref_frame_id))
            right_state = self.brpo_runtime_camera_states.get(int(slot.right_ref_frame_id))
            if pseudo_state is None or left_state is None or right_state is None:
                Log(f"[BRPOOnlineMapping] skip frame={int(slot.frame_id)} missing cached runtime states")
                continue
            if not pseudo_state.get("image_path") or not left_state.get("image_path") or not right_state.get("image_path"):
                Log(f"[BRPOOnlineMapping] skip frame={int(slot.frame_id)} missing runtime image_path")
                continue
            frame_root = event_root / f"frame_{int(slot.frame_id):04d}"
            frame_root.mkdir(parents=True, exist_ok=True)
            expected_support_mode = str((self.config.get("Results", {}).get("brpo_online_mapping", {}) or {}).get("rgb_only_support_mode", "reciprocal_seed"))
            expected_point_radius = int((self.config.get("Results", {}).get("brpo_online_mapping", {}) or {}).get("cm_dense_point_radius", 2))
            with open(frame_root / "exact_cfg_debug.json", "w", encoding="utf-8") as f:
                json.dump({
                    "expected_support_mode": expected_support_mode,
                    "expected_point_radius": expected_point_radius,
                    "resolved_support_mode": str(exact_cfg.rgb_only_support_mode),
                    "resolved_point_radius": int(exact_cfg.cm_dense_point_radius),
                    "rgb_only_verification": bool(exact_cfg.rgb_only_verification),
                    "raw_result_cfg": raw_result_cfg,
                    "resolved_cfg": cfg,
                }, f, indent=2)
            Log(f"[BRPOOnlineMapping][cfg] frame={int(slot.frame_id)} expected_support_mode={expected_support_mode} resolved_support_mode={exact_cfg.rgb_only_support_mode} expected_point_radius={expected_point_radius} resolved_point_radius={exact_cfg.cm_dense_point_radius}")
            if expected_support_mode != str(exact_cfg.rgb_only_support_mode):
                raise RuntimeError(f"E9 support-mode mismatch: expected {expected_support_mode}, got {exact_cfg.rgb_only_support_mode}")
            exact_bundle = build_runtime_exact_backend_bundle(
                slot=slot,
                states_by_id=self.brpo_runtime_camera_states,
                gaussians=self.gaussians,
                pipe=self.pipeline_params,
                background=self.background,
                frame_root=frame_root,
                cfg=exact_cfg,
                matcher=matcher,
                difix_model=self.brpo_difix_model,  # Pass Difix model
            )
            signal_bundle = build_runtime_exact_signal_bundle(
                slot=slot,
                frame_root=frame_root,
                exact_bundle=exact_bundle,
            )
            record_bundle = build_runtime_pseudo_record_bundle(
                slot=slot,
                frame_root=frame_root,
                pseudo_state=pseudo_state,
                pseudo_render_rgb=exact_bundle.pseudo_render_rgb,
                signal_bundle=signal_bundle,
                stageA_scene_scale=float(self.cameras_extent) if self.cameras_extent is not None else None,
            )
            self.brpo_runtime_pseudo_records[int(slot.frame_id)] = record_bundle.record
            self.brpo_runtime_seen_gap_keys.add(slot.gap_key)
            event_slots.append(
                {
                    **slot.as_dict(),
                    "exact_backend_bundle_path": str(exact_bundle.exact_frame_out),
                    "signal_bundle_path": str(signal_bundle.signal_frame_out),
                    "runtime_record_path": str(record_bundle.record_frame_out),
                    "exact_summary": {
                        "left_support_ratio": float((exact_bundle.left_result["support_mask"] > 0.5).mean()),
                        "right_support_ratio": float((exact_bundle.right_result["support_mask"] > 0.5).mean()),
                    },
                    "signal_summary": signal_bundle.result["summary"],
                }
            )
            Log(
                f"[BRPOOnlineMapping] keyframe={int(cur_frame_idx)} slot frame={int(slot.frame_id)} refs=({int(slot.left_ref_frame_id)},{int(slot.right_ref_frame_id)})"
            )
        payload = {
            "trigger_keyframe": int(cur_frame_idx),
            "current_window": [int(x) for x in self.current_window],
            "topology_mode": str(cfg.get("topology_mode", "side_branch")),
            "num_slots": len(event_slots),
            "slots": event_slots,
            "pseudo_gradient_enabled": bool(cfg.get("enable_pseudo_gradient", False)),
            "pseudo_map_iters": int(cfg.get("pseudo_map_iters", 0)),
        }
        with open(event_root / "event_summary.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        Log(
            f"[BRPOOnlineMapping] keyframe={int(cur_frame_idx)} prepared {len(event_slots)} runtime slot(s); pseudo_gradient={bool(cfg.get('enable_pseudo_gradient', False))} pseudo_map_iters={int(cfg.get('pseudo_map_iters', 0))}"
        )
        return payload

    def _run_brpo_runtime_pseudo_mapping(self, cur_frame_idx, prepare_payload):
        cfg = self.brpo_online_mapping_cfg
        if cfg is None:
            return None
        topology_mode = str(cfg.get("topology_mode", "side_branch"))
        if not bool(cfg.get("enable_pseudo_gradient", False)):
            if topology_mode == "joint_primary":
                self._update_brpo_event_summary(cur_frame_idx, {
                    "joint_primary_status": "fallback_real_only",
                    "joint_primary_fallback_reason": "pseudo_gradient_disabled",
                })
            return None
        pseudo_map_iters = int(cfg.get("pseudo_map_iters", 0))
        if pseudo_map_iters <= 0:
            if topology_mode == "joint_primary":
                self._update_brpo_event_summary(cur_frame_idx, {
                    "joint_primary_status": "fallback_real_only",
                    "joint_primary_fallback_reason": "pseudo_map_iters<=0",
                })
            return None
        if prepare_payload is None:
            if topology_mode == "joint_primary":
                self._update_brpo_event_summary(cur_frame_idx, {
                    "joint_primary_status": "fallback_real_only",
                    "joint_primary_fallback_reason": "no_prepare_payload",
                })
            return None

        slot_frame_ids = [int(slot["frame_id"]) for slot in prepare_payload.get("slots", [])]
        runtime_records = [
            self.brpo_runtime_pseudo_records[int(frame_id)]
            for frame_id in slot_frame_ids
            if int(frame_id) in self.brpo_runtime_pseudo_records
        ]
        if not runtime_records:
            skip_reason = "no_runtime_records"
            if not slot_frame_ids:
                skip_reason = "no_runtime_slots"
            self._update_brpo_event_summary(cur_frame_idx, {
                "pseudo_mapping": {
                    "topology_mode": topology_mode,
                    "status": "skipped",
                    "reason": skip_reason,
                }
            })
            Log(f"[BRPOOnlineMapping] keyframe={int(cur_frame_idx)} pseudo mapping skipped: {skip_reason}")
            return None

        event_root = Path(cfg["debug_export_root"]) / f"event_kf_{int(cur_frame_idx):04d}"
        mapping_dir_name = "joint_primary_mapping" if topology_mode == "joint_primary" else "pseudo_mapping"
        mapping_out = event_root / mapping_dir_name
        joint_primary_mode = topology_mode == "joint_primary"
        mapping_cfg = BRPOMappingConfig(
            num_iterations=pseudo_map_iters,
            num_pseudo_views_per_step=int(cfg.get("num_pseudo_views_per_step", 1)),
            lambda_real=float(cfg.get("lambda_real", 1.0)),
            lambda_pseudo=float(cfg.get("lambda_pseudo", 1.0)),
            lambda_depth=float(cfg.get("lambda_depth", 1.0)),
            match_real_loss_weights=bool(cfg.get("match_real_loss_weights", False)),
            beta_rgb=float(cfg.get("beta_rgb", 0.7)),
            lambda_pose=float(cfg.get("lambda_pose", 0.01)),
            lambda_exp=float(cfg.get("lambda_exp", 0.001)),
            trans_weight=float(cfg.get("trans_weight", 1.0)),
            lambda_abs_pose=float(cfg.get("lambda_abs_pose", 0.0)),
            lambda_abs_t=float(cfg.get("lambda_abs_t", 3.0)),
            lambda_abs_r=float(cfg.get("lambda_abs_r", 0.1)),
            abs_pose_robust=str(cfg.get("abs_pose_robust", "charbonnier")),
            update_real_pose=bool(cfg.get("update_real_pose", False)),
            update_real_exposure=bool(cfg.get("update_real_exposure", cfg.get("update_real_pose", False))),
            update_pseudo_pose=bool(cfg.get("update_pseudo_pose", True)),
            use_depth=bool(cfg.get("use_depth", True)),
            split_pseudo_authority=bool(cfg.get("split_pseudo_authority", True)),
            pseudo_scene_mask_mode=str(cfg.get("pseudo_scene_mask_mode", "both_only")),
            topology_mode=topology_mode,
            pseudo_window_equivalence=bool(cfg.get("pseudo_window_equivalence", False)),
            extra_real_views=int(cfg.get("extra_real_views", 0)),
            propagate_pseudo_delta_to_neighbors=bool(cfg.get("propagate_pseudo_delta_to_neighbors", True)),
            gaussian_maintenance_source=str(cfg.get("gaussian_maintenance_source", "all_views")),
            enable_densify=bool(cfg.get("enable_densify", joint_primary_mode)),
            enable_prune=bool(cfg.get("enable_prune", False)),
            enable_opacity_reset=bool(cfg.get("enable_opacity_reset", joint_primary_mode)),
            isotropic_weight=float(cfg.get("isotropic_weight", 10.0)),
            output_dir=str(mapping_out),
            seed=int(cfg.get("seed", 0)),
            depth_loss_mode=str(cfg.get("depth_loss_mode", "exact_shared_cm_v1")),
            tau_rel_depth=float(cfg.get("tau_rel_depth", 0.15)),
            # NEW parameters for scale regularization and Gauss-Newton
            lambda_scale=float(cfg.get("lambda_scale", 0.01)),
            max_scale=float(cfg.get("max_scale", 0.0)) if cfg.get("max_scale") is not None else None,
            use_gauss_newton=bool(cfg.get("use_gauss_newton", False)),
            gn_max_iters=int(cfg.get("gn_max_iters", 5)),
            gn_damping=float(cfg.get("gn_damping", 0.01)),
            gn_every_n_steps=int(cfg.get("gn_every_n_steps", 1)),
        )

        xyz_before = self.gaussians.get_xyz.detach().clone()
        result = run_brpo_pseudo_mapping(
            config=self.config,
            gaussians=self.gaussians,
            pipeline_params=self.pipeline_params,
            opt_params=self.opt_params,
            background=self.background,
            cameras=self.viewpoints,
            kf_indices=list(self.viewpoints.keys()),
            current_window=list(self.current_window),
            mapping_cfg=mapping_cfg,
            runtime_pseudo_records=runtime_records,
            iteration_start=int(self.iteration_count),
            gaussian_update_every=int(self.gaussian_update_every),
            gaussian_update_offset=int(self.gaussian_update_offset),
            gaussian_th=float(self.gaussian_th),
            gaussian_extent=float(self.gaussian_extent),
            gaussian_reset=int(self.gaussian_reset),
            size_threshold=self.size_threshold,
        )
        xyz_after = self.gaussians.get_xyz.detach().clone()
        if xyz_before.shape == xyz_after.shape:
            gaussian_xyz_max_abs_delta = float((xyz_after - xyz_before).abs().max().item())
        else:
            gaussian_xyz_max_abs_delta = -1.0  # signal count changed
        self.iteration_count = int(result.get("iteration_count_end", self.iteration_count))

        summary = {
            "trigger_keyframe": int(cur_frame_idx),
            "topology_mode": topology_mode,
            "runtime_pseudo_frame_ids": [int(record.frame_id) for record in runtime_records],
            "num_runtime_pseudo_records": len(runtime_records),
            "num_extra_real_candidates": int(result.get("num_extra_real_candidates", 0)),
            "pseudo_map_iters": pseudo_map_iters,
            "history_path": str(mapping_out / "brpo_pseudo_history.json"),
            "gaussian_xyz_max_abs_delta": gaussian_xyz_max_abs_delta,
            "match_real_loss_weights": bool(cfg.get("match_real_loss_weights", False)),
            "beta_rgb": float(cfg.get("beta_rgb", 0.7)),
            "lambda_depth": float(cfg.get("lambda_depth", 1.0)),
            "pseudo_window_equivalence": bool(cfg.get("pseudo_window_equivalence", False)),
            "extra_real_views": int(cfg.get("extra_real_views", 0)),
            "gaussian_maintenance_source": str(cfg.get("gaussian_maintenance_source", "all_views")),
            "final_loss_total": float(result["history"][-1]["loss_total"]) if result.get("history") else None,
            "final_loss_real": float(result["history"][-1]["loss_real"]) if result.get("history") else None,
            "final_loss_pseudo": float(result["history"][-1]["loss_pseudo"]) if result.get("history") else None,
            "joint_primary_used": bool(joint_primary_mode),
        }
        with open(event_root / "pseudo_mapping_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        event_summary_path = event_root / "event_summary.json"
        if event_summary_path.exists():
            payload = json.loads(event_summary_path.read_text(encoding="utf-8"))
        else:
            payload = {
                "trigger_keyframe": int(cur_frame_idx),
                "current_window": [int(x) for x in self.current_window],
                "num_slots": len(slot_frame_ids),
                "slots": prepare_payload.get("slots", []),
            }
        payload["pseudo_mapping"] = summary
        with open(event_summary_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        Log(
            f"[BRPOOnlineMapping] keyframe={int(cur_frame_idx)} ran pseudo mapping on {len(runtime_records)} slot(s); iters={pseudo_map_iters} xyz_max_abs_delta={gaussian_xyz_max_abs_delta:.6f}"
        )
        return result
    # Initialize the SLAM map by optimizing Gaussians through multiple iterations
    def initialize_map(self, cur_frame_idx, viewpoint):
        for mapping_iteration in range(self.init_itr_num):
            self.iteration_count += 1
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )
            loss_init = get_loss_mapping(
                self.config, image, viewpoint, depth=depth,initialization=True
            )
            loss_init.backward()

            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(  
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.add_densification_stats(                 
                    viewspace_point_tensor, visibility_filter
                )
                if mapping_iteration % self.init_gaussian_update == 0:  
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )

                if self.iteration_count == self.init_gaussian_reset or (
                    self.iteration_count == self.opt_params.densify_from_iter
                ):
                    self.gaussians.reset_opacity()

                self.gaussians.optimizer.step()                         
                self.gaussians.optimizer.zero_grad(set_to_none=True)    

        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()   
        Log("Initialized map")
        return render_pkg

    def _run_joint_primary_real_gaussian_fallback(self, iter_per_kf):
        """Real-only Gaussian maintenance fallback for joint_primary mode.

        When joint_primary succeeds but enable_densify=false, this method
        runs an extra real-only Gaussian optimization pass with densify
        and opacity_reset enabled (controlled by map()'s default behavior).

        Key features:
        - Temporarily sets keyframe_optimizers=None to skip pose updates
        - Runs map() with up_pose=False (both conditions block pose changes)
        - Allows densify/opacity_reset to run via map()'s iteration logic
        """
        prev_optimizer = self.keyframe_optimizers
        try:
            self.keyframe_optimizers = None
            self.map(self.current_window, iters=iter_per_kf, up_pose=False)
        finally:
            self.keyframe_optimizers = prev_optimizer

    def _run_legacy_real_keyframe_mapping(self, iter_per_kf, frames_to_optimize):
        opt_params = []
        for cam_idx in range(len(self.current_window)):
            if self.current_window[cam_idx] == 0:
                continue
            viewpoint = self.viewpoints[self.current_window[cam_idx]]
            if cam_idx < frames_to_optimize:
                opt_params.append(
                    {
                        "params": [viewpoint.cam_rot_delta],
                        "lr": self.config["Training"]["lr"]["cam_rot_delta"] * 0.5,
                        "name": "rot_{}".format(viewpoint.uid),
                    }
                )
                opt_params.append(
                    {
                        "params": [viewpoint.cam_trans_delta],
                        "lr": self.config["Training"]["lr"]["cam_trans_delta"] * 0.5,
                        "name": "trans_{}".format(viewpoint.uid),
                    }
                )
            opt_params.append(
                {
                    "params": [viewpoint.exposure_a],
                    "lr": 0.01,
                    "name": "exposure_a_{}".format(viewpoint.uid),
                }
            )
            opt_params.append(
                {
                    "params": [viewpoint.exposure_b],
                    "lr": 0.01,
                    "name": "exposure_b_{}".format(viewpoint.uid),
                }
            )
        self.keyframe_optimizers = torch.optim.Adam(opt_params)
        self.map(self.current_window, iters=iter_per_kf, up_pose=True)
        self.map(self.current_window, prune=True)

    # Optimize keyframe poses and Gaussians scene
    def map(self, current_window, prune=False, iters=1, up_pose = True):
        if len(current_window) == 0:
            return

        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        random_viewpoint_stack = []
        frames_to_optimize = self.config["Training"]["pose_window"]

        current_window_set = set(current_window)            
        for cam_idx, viewpoint in self.viewpoints.items():  # Add viewpoints outside the current window to the random_viewpoint_stack
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)        
            
        for _ in range(iters):
            self.iteration_count += 1
            self.last_sent += 1

            loss_mapping = 0
            viewspace_point_tensor_acm = []                 
            visibility_filter_acm = []                      
            radii_acm = []                                  
            n_touched_acm = []                            

            keyframes_opt = []          

            for cam_idx in range(len(current_window)):      # For each keyframe in the current window, perform rendering and compute loss
                viewpoint = viewpoint_stack[cam_idx]
                keyframes_opt.append(viewpoint)
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (                                          
                    image,
                    viewspace_point_tensor,                 
                    visibility_filter,                     
                    radii,                                  
                    depth,                                 
                    opacity,                                
                    n_touched,                              
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )
                loss_mapping += get_loss_mapping(self.config, image, viewpoint, depth=depth, monodepth=True)
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)     
                
            # In each iteration, randomly select two non-window keyframes for optimization
            for cam_idx in torch.randperm(len(random_viewpoint_stack))[:2]:     
                viewpoint = random_viewpoint_stack[cam_idx]
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )
                loss_mapping += get_loss_mapping(self.config, image, viewpoint, depth=depth, monodepth=True)
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                
            # isotropic regularization
            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()
            loss_mapping.backward()
            gaussian_split = False
            
            # Deinsifying / Pruning Gaussians
            with torch.no_grad():
                self.occ_aware_visibility = {}            
                for idx in range((len(current_window))):
                    kf_idx = current_window[idx]
                    n_touched = n_touched_acm[idx]
                    self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()

                # Only prune on the last iteration and when we have full window
                if prune:     
                    if len(current_window) == self.config["Training"]["window_size"]:
                        prune_mode = self.config["Training"]["prune_mode"]
                        prune_coviz = self.config["Training"]["prune_num"]  # prune parameter
                        self.gaussians.n_obs.fill_(0)
                        for window_idx, visibility in self.occ_aware_visibility.items():
                            self.gaussians.n_obs += visibility.cpu()
                        to_prune = None
                        if prune_mode == "odometry":
                            to_prune = self.gaussians.n_obs < 3
                            # make sure we don't split the gaussians, break here.
                        if prune_mode == "slam":
                            # only prune keyframes which are relatively new
                            sorted_window = sorted(current_window, reverse=True)
                            mask = self.gaussians.unique_kfIDs >= sorted_window[2]
                            if not self.initialized:
                                mask = self.gaussians.unique_kfIDs >= 0
                            to_prune = torch.logical_and(
                                self.gaussians.n_obs <= prune_coviz, mask
                            )
                        if to_prune is not None and self.monocular:       
                            self.gaussians.prune_points(to_prune.cuda())
                            for idx in range((len(current_window))):
                                current_idx = current_window[idx]
                                self.occ_aware_visibility[current_idx] = (                
                                    self.occ_aware_visibility[current_idx][~to_prune]
                                )
                        if not self.initialized:
                            self.initialized = True
                            Log("Initialized SLAM")
                    return False

                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )

                update_gaussian = (
                    self.iteration_count % self.gaussian_update_every
                    == self.gaussian_update_offset
                )
                if update_gaussian:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )
                    gaussian_split = True

                if (self.iteration_count % self.gaussian_reset) == 0 and (
                    not update_gaussian) :
                    Log("Resetting the opacity of non-visible Gaussians")
                    self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                    gaussian_split = True

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)

                # Joint-primary online mapping may intentionally skip the legacy
                # real-keyframe optimizer builder. In that mode this map() tick is
                # only a Gaussian/visibility maintenance pass; do not crash or
                # apply pose deltas without an optimizer step.
                has_keyframe_optimizer = self.keyframe_optimizers is not None
                if has_keyframe_optimizer:
                    self.keyframe_optimizers.step()
                    self.keyframe_optimizers.zero_grad(set_to_none=True)

                # Pose update
                if up_pose and has_keyframe_optimizer:
                    for cam_idx in range(min(frames_to_optimize, len(current_window))):
                        viewpoint = viewpoint_stack[cam_idx]
                        update_pose(viewpoint)
        return gaussian_split
                
    # Run color refinement as a post-processing step after SLAM
    def color_refinement(self):
        """Color refinement with optional masked pseudo support.

        When color_refinement_use_pseudo=True:
        - Real views use full-image L1+SSIM (same as original)
        - Pseudo views use masked L1+SSIM with confidence_mask (C_m)
        """
        Log("Starting color refinement")

        # Get config
        results_cfg = self.config.get("Results", {})
        brpo_cfg = self.brpo_online_mapping_cfg or {}
        use_pseudo = bool(brpo_cfg.get("color_refinement_use_pseudo", False))
        pseudo_ratio = float(brpo_cfg.get("color_refinement_pseudo_ratio", 0.5))
        pseudo_weight = float(brpo_cfg.get("color_refinement_pseudo_weight", 1.0))
        mask_source = str(brpo_cfg.get("color_refinement_pseudo_mask_source", "confidence_mask"))
        log_every = int(brpo_cfg.get("color_refinement_log_every", 200))

        # Get pseudo records pool
        pseudo_pool = list(self.brpo_runtime_pseudo_records.values()) if use_pseudo else []
        pseudo_pool_size = len(pseudo_pool)

        iteration_total = int(os.environ.get("S3PO_COLOR_REFINEMENT_ITERS",
            results_cfg.get("color_refinement_iters", 26000)))

        # Statistics for summary
        num_real_steps = 0
        num_pseudo_steps = 0
        real_loss_sum = 0.0
        pseudo_loss_sum = 0.0
        pseudo_mask_nonzero_sum = 0.0

        lambda_dssim = self.opt_params.lambda_dssim

        for iteration in tqdm(range(1, iteration_total + 1)):
            self.gaussians.optimizer.zero_grad(set_to_none=True)

            # Decide: real or pseudo
            use_pseudo_this_step = use_pseudo and pseudo_pool_size > 0 and random.random() < pseudo_ratio

            if use_pseudo_this_step:
                # Sample a pseudo record
                record = random.choice(pseudo_pool)

                # Get mask based on mask_source
                if mask_source == "confidence_mask":
                    mask = record.confidence_mask
                elif mask_source == "valid_mask":
                    mask = record.valid_mask
                elif mask_source == "support_both_mask":
                    mask = record.support_both_mask if hasattr(record, 'support_both_mask') else record.confidence_mask
                else:
                    mask = record.confidence_mask  # default fallback

                # Render pseudo view
                render_pkg = render(record.viewpoint, self.gaussians, self.pipeline_params, self.background)
                image = render_pkg["render"]

                # Target RGB (already tensor or numpy)
                target_rgb = record.target_rgb
                if not isinstance(target_rgb, torch.Tensor):
                    target_rgb = torch.from_numpy(target_rgb).cuda()
                if target_rgb.dim() == 3 and target_rgb.shape[-1] == 3:
                    target_rgb = target_rgb.permute(2, 0, 1)

                # Mask to tensor
                if not isinstance(mask, torch.Tensor):
                    mask = torch.from_numpy(mask).cuda()
                if mask.dim() == 3 and mask.shape[0] != 1:
                    mask = mask.squeeze(0)

                # Masked L1 + SSIM
                Ll1 = masked_l1_loss(image, target_rgb, mask)
                ssim_val = masked_ssim(image, target_rgb, mask)
                loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim_val)
                loss = pseudo_weight * loss

                # Update stats
                num_pseudo_steps += 1
                pseudo_loss_sum += float(loss.detach().item())
                pseudo_mask_nonzero_sum += float((mask > 0).float().mean().item())

            else:
                # Original real view path
                viewpoint_idx_stack = list(self.viewpoints.keys())
                viewpoint_cam_idx = viewpoint_idx_stack.pop(
                    random.randint(0, len(viewpoint_idx_stack) - 1)
                )
                viewpoint_cam = self.viewpoints[viewpoint_cam_idx]
                render_pkg = render(
                    viewpoint_cam, self.gaussians, self.pipeline_params, self.background
                )
                image, visibility_filter, radii = (
                    render_pkg["render"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                )

                gt_image = viewpoint_cam.original_image.cuda()
                Ll1 = l1_loss(image, gt_image)
                loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim(image, gt_image))

                # Update stats
                num_real_steps += 1
                real_loss_sum += float(loss.detach().item())

                # Update radii for real views only
                with torch.no_grad():
                    self.gaussians.max_radii2D[visibility_filter] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter],
                        radii[visibility_filter],
                    )

            loss.backward()
            self.gaussians.optimizer.step()
            self.gaussians.update_learning_rate(26000)

            # Periodic logging
            if iteration % log_every == 0:
                avg_loss = (real_loss_sum + pseudo_loss_sum) / (num_real_steps + num_pseudo_steps + 1e-8)
                Log(f"Color refinement iter {iteration}: loss={avg_loss:.4f}, "
                    f"real_steps={num_real_steps}, pseudo_steps={num_pseudo_steps}")

        # Save summary
        summary = {
            "use_pseudo": use_pseudo,
            "pseudo_pool_size": pseudo_pool_size,
            "pseudo_mask_source": mask_source,
            "pseudo_ratio": pseudo_ratio,
            "pseudo_weight": pseudo_weight,
            "num_real_steps": num_real_steps,
            "num_pseudo_steps": num_pseudo_steps,
            "mean_real_loss": real_loss_sum / max(num_real_steps, 1),
            "mean_pseudo_loss": pseudo_loss_sum / max(num_pseudo_steps, 1),
            "mean_pseudo_mask_nonzero_ratio": pseudo_mask_nonzero_sum / max(num_pseudo_steps, 1),
            "color_refinement_updates_pose": False,
            "total_iterations": iteration_total,
        }

        summary_path = Path(self.save_dir) / "color_refinement_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        Log(f"Color refinement summary saved to {summary_path}")

        Log("Map refinement done")



    # ========================================================================
    # Part3 Stage1: Pseudo view mapping-only refinement
    # Added: 2026-04-05
    # Purpose: Refine Gaussian scene using pseudo views without affecting tracking
    # ========================================================================
    def pseudo_refinement(self, pseudo_cache_path, num_iterations=100, use_left=True):
        import json
        import numpy as np
        from PIL import Image
        from tqdm import tqdm
        from utils.slam_utils import get_loss_pseudo
        
        Log(f"[Part3 Stage1] Starting pseudo refinement from {pseudo_cache_path}")
        
        pseudo_cache_path = Path(pseudo_cache_path)
        manifest_path = pseudo_cache_path / "manifest.json"
        
        if not manifest_path.exists():
            Log(f"[Part3 Stage1] Warning: manifest.json not found at {manifest_path}")
            return
        
        manifest = json.load(open(manifest_path))
        sample_ids = manifest.get("sample_ids", [])
        
        if len(sample_ids) == 0:
            Log("[Part3 Stage1] Warning: no pseudo samples found")
            return
        
        Log(f"[Part3 Stage1] Found {len(sample_ids)} pseudo samples")
        
        # Load pseudo viewpoints
        pseudo_viewpoints = []
        
        for sample_id in sample_ids:
            sample_dir = pseudo_cache_path / "samples" / str(sample_id)
            
            # Load camera
            camera_path = sample_dir / "camera.json"
            if not camera_path.exists():
                continue
            
            camera = json.load(open(camera_path))
            
            # Load target RGB
            target_rgb_name = "target_rgb_left.png" if use_left else "target_rgb_right.png"
            target_rgb_path = sample_dir / target_rgb_name
            if not target_rgb_path.exists():
                continue
            
            target_rgb = np.array(Image.open(target_rgb_path)).astype(np.float32) / 255.0
            
            # Load target depth
            target_depth_path = sample_dir / "target_depth.npy"
            if not target_depth_path.exists():
                continue
            
            target_depth = np.load(target_depth_path)
            
            # Load confidence mask
            conf_path = sample_dir / "confidence_mask.npy"
            if conf_path.exists():
                confidence = np.load(conf_path)
            else:
                confidence = (target_depth > 0).astype(np.float32)
            
            # Create viewpoint from camera data
            viewpoint = self._create_pseudo_viewpoint(camera, target_rgb, target_depth, confidence)
            if viewpoint is not None:
                pseudo_viewpoints.append({
                    "viewpoint": viewpoint,
                    "target_rgb": target_rgb,
                    "target_depth": target_depth,
                    "confidence": confidence,
                    "sample_id": sample_id,
                })
        
        if len(pseudo_viewpoints) == 0:
            Log("[Part3 Stage1] Warning: no valid pseudo viewpoints loaded")
            return
        
        Log(f"[Part3 Stage1] Loaded {len(pseudo_viewpoints)} pseudo viewpoints for refinement")
        
        # Optimization loop
        for iteration in tqdm(range(1, num_iterations + 1), desc="Pseudo refinement"):
            self.iteration_count += 1
            
            loss_pseudo = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            
            # Randomly sample pseudo viewpoints
            num_to_sample = min(4, len(pseudo_viewpoints))
            indices = np.random.choice(len(pseudo_viewpoints), num_to_sample, replace=False)
            
            for idx in indices:
                pv = pseudo_viewpoints[idx]
                viewpoint = pv["viewpoint"]
                
                # Render from Gaussian scene
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                
                image = render_pkg["render"]
                depth = render_pkg["depth"]
                viewspace_point_tensor = render_pkg["viewspace_points"]
                visibility_filter = render_pkg["visibility_filter"]
                radii = render_pkg["radii"]
                
                # Compute pseudo loss
                loss = get_loss_pseudo(
                    self.config, image, depth,
                    pv["target_rgb"], pv["target_depth"], pv["confidence"]
                )
                loss_pseudo += loss
                
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
            
            # Isotropic regularization (same as map())
            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_pseudo += 10 * isotropic_loss.mean()
            
            loss_pseudo.backward()
            
            # Update Gaussians with densify/prune (A2 fix)
            with torch.no_grad():
                # 1. Collect densification statistics from all sampled views
                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx],
                        visibility_filter_acm[idx]
                    )
                    # Update max radii for pruning
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]]
                    )
                
                # 2. Densify/prune periodically (every 100 iterations after warmup)
                densify_interval = 100
                densify_from_iter = 500
                densify_until_iter = min(num_iterations, 15000)
                
                if iteration > densify_from_iter and iteration < densify_until_iter and iteration % densify_interval == 0:
                    size_threshold = 20 if iteration > 3000 else None
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        size_threshold
                    )
                
                # 3. Optimizer step
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
        
        Log(f"[Part3 Stage1] Pseudo refinement done after {num_iterations} iterations, Gaussians: {len(self.gaussians.get_xyz)}")
    
    def _create_pseudo_viewpoint(self, camera, target_rgb, target_depth, confidence):
        import torch
        
        # Extract camera parameters
        pose_c2w = np.array(camera["pose_c2w"])
        intrinsics = camera.get("intrinsics_px", {})
        image_size = camera.get("image_size", {"width": 512, "height": 512})
        
        # c2w -> w2c
        pose_w2c = np.linalg.inv(pose_c2w)
        
        R = pose_w2c[:3, :3].T  # Transpose for OpenGL convention
        T = pose_w2c[:3, 3]
        
        # Compute FoV
        fx = intrinsics.get("fx", 500)
        fy = intrinsics.get("fy", fx)
        W = image_size.get("width", 512)
        H = image_size.get("height", 512)
        
        FoVx = 2 * np.arctan(W / (2 * fx))
        FoVy = 2 * np.arctan(H / (2 * fy))
        
        # Create viewpoint object
        class PseudoViewpoint:
            def __init__(self):
                self.R = torch.from_numpy(R).float()
                self.T = torch.from_numpy(T).float()
                self.FoVx = FoVx
                self.FoVy = FoVy
                self.image_height = H
                self.image_width = W
                self.original_image = torch.zeros(3, H, W)
                self.mono_depth = target_depth
                self.exposure_a = torch.tensor(0.0)
                self.exposure_b = torch.tensor(0.0)
        
        return PseudoViewpoint()

    def _refresh_current_window_visibility(self, reason=""):
        """Recompute visibility masks after joint-primary online mapping.

        The frontend receives a cloned GaussianModel plus occ_aware_visibility.
        If online pseudo mapping changes the Gaussian count but we skip the
        legacy real map(), stale masks cause frontend logical_and length errors.
        This refresh is read-only: it renders current real keyframes and updates
        only the per-keyframe visibility tensors before push_to_frontend().
        """
        if len(self.current_window) == 0:
            return
        refreshed = {}
        with torch.no_grad():
            for kf_idx in self.current_window:
                viewpoint = self.viewpoints.get(kf_idx)
                if viewpoint is None:
                    continue
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                n_touched = render_pkg.get("n_touched")
                if n_touched is None:
                    continue
                refreshed[kf_idx] = (n_touched > 0).long()
        if refreshed:
            self.occ_aware_visibility.update(refreshed)
            expected = int(self.gaussians.get_xyz.shape[0])
            bad = {
                int(k): int(v.numel())
                for k, v in refreshed.items()
                if int(v.numel()) != expected
            }
            if bad:
                Log(f"[BRPOOnlineMapping] visibility refresh length mismatch after {reason}: expected={expected}, got={bad}")
            else:
                Log(f"[BRPOOnlineMapping] refreshed current-window visibility after {reason}; gaussians={expected}, keyframes={list(refreshed.keys())}")

    def push_to_frontend(self, tag=None):
        self.last_sent = 0
        keyframes = []
        for kf_idx in self.current_window:
            kf = self.viewpoints[kf_idx]
            keyframes.append((kf_idx, kf.R.clone(), kf.T.clone()))
        if tag is None:
            tag = "sync_backend"
            
        msg = [tag, clone_obj(self.gaussians), self.occ_aware_visibility, keyframes]
        self.frontend_queue.put(msg)
    # Main execution loop: 
    # process backend messages, perform initialization, optimize keyframe map, color refinement,
    # synchronize data, and push updates to the frontend
    def run(self):
        # Ensure the spawned child uses the same logical GPU selected by the launcher.
        # With CUDA_VISIBLE_DEVICES=1, logical cuda:0 maps to physical GPU 1.
        import os
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        self._target_cuda_device = 0
        if torch.cuda.is_available():
            if cuda_visible is not None:
                Log(f"[BackEnd.run] CUDA_VISIBLE_DEVICES={cuda_visible}, using logical cuda:0 for backend child")
            torch.cuda.set_device(self._target_cuda_device)
            # Force context creation on the selected logical device before lazy Difix loads.
            _ = torch.empty(1, device=f"cuda:{self._target_cuda_device}")
            self.device = f"cuda:{self._target_cuda_device}"

        # Rebuild the background tensor inside the child, instead of unpickling a
        # CUDA tensor from the parent process. This keeps normal SLAM behavior but
        # avoids premature CUDA-context initialization during spawn unpickle.
        if self.background_cpu is not None and torch.cuda.is_available():
            self.background = torch.tensor(
                self.background_cpu,
                dtype=torch.float32,
                device=f"cuda:{self._target_cuda_device}",
            )
        elif self.background is not None and isinstance(self.background, torch.Tensor) and torch.cuda.is_available():
            if self.background.device.type != "cuda" or self.background.device.index != self._target_cuda_device:
                bg_data = self.background.detach().cpu().numpy()
                self.background = torch.tensor(
                    bg_data,
                    dtype=torch.float32,
                    device=f"cuda:{self._target_cuda_device}",
                )
        while True:
            if self.backend_queue.empty():
                if self.pause:
                    time.sleep(0.01)
                    continue
                if len(self.current_window) == 0:
                    time.sleep(0.01)
                    continue

                if self.single_thread:
                    time.sleep(0.01)
                    continue
                if self.keyframe_optimizers is None:
                    # Joint-primary BRPO can complete a pseudo/real joint update
                    # without running the legacy real-keyframe optimizer builder
                    # (e.g. joint_primary_run_legacy_prune=false). In that case
                    # there is no idle real-window optimizer to step; skip this
                    # background map tick instead of crashing. Legacy/normal SLAM
                    # behavior is unchanged because keyframe_optimizers is set
                    # before idle mapping there.
                    time.sleep(0.01)
                    continue
                self.map(self.current_window)
                if self.last_sent >= 10:       
                    self.map(self.current_window, prune=True, iters=10)
                    self.push_to_frontend()
            else:
                data = self.backend_queue.get()
                if data[0] == "stop":
                    break
                elif data[0] == "pause":
                    self.pause = True
                elif data[0] == "unpause":
                    self.pause = False
                elif data[0] == "color_refinement":
                    self.color_refinement()
                    self.push_to_frontend()

                elif data[0] == "init":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    depth_map = data[3]
                    Log("Resetting the system")
                    self.reset()

                    self.viewpoints[cur_frame_idx] = viewpoint
                    T_np = np.linalg.inv(getWorld2View2(viewpoint.R,viewpoint.T).cpu().numpy())
                    T = torch.from_numpy(T_np).to(self.device)
                    self.add_next_kf(
                        cur_frame_idx, viewpoint, depth_map=depth_map, init=True
                    )
                    self.initialize_map(cur_frame_idx, viewpoint)
                    self.push_to_frontend("init")

                elif data[0] == "keyframe":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    current_window = data[3]
                    depth_map = data[4]
                    self.theta = data[5]
                    runtime_state_payload = data[6] if len(data) > 6 else None
                    if isinstance(runtime_state_payload, dict):
                        self.brpo_runtime_camera_states = runtime_state_payload
                    theta_value = self.theta.item()
                    print("current keyframe ",cur_frame_idx,'window is ',current_window)

                    T_np = np.linalg.inv(getWorld2View2(viewpoint.R,viewpoint.T).cpu().numpy())
                    T = torch.from_numpy(T_np).to(self.device)
                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.current_window = current_window
                    self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map)

                    frames_to_optimize = self.config["Training"]["pose_window"]
                    iter_nosingle = self.config["Training"]["mapping_itr_nosingle"]
                    iter_per_kf = self.mapping_itr_num if self.single_thread else iter_nosingle
                    if not self.initialized:
                        if (
                            len(self.current_window)
                            == self.config["Training"]["window_size"]
                        ):
                            frames_to_optimize = (
                                self.config["Training"]["window_size"] - 1
                            )
                            iter_per_kf = 50 if self.live_mode else 300
                            Log("Performing initial BA for initialization")
                        else:
                            iter_per_kf = self.mapping_itr_num

                    topology_mode = "side_branch"
                    if self.brpo_online_mapping_cfg is not None:
                        topology_mode = str(self.brpo_online_mapping_cfg.get("topology_mode", "side_branch"))

                    if topology_mode == "joint_primary":
                        prepare_payload = self._maybe_prepare_brpo_runtime_slots(cur_frame_idx)
                        joint_result = self._run_brpo_runtime_pseudo_mapping(cur_frame_idx, prepare_payload)
                        if joint_result is None:
                            self._update_brpo_event_summary(
                                cur_frame_idx,
                                {
                                    "joint_primary_status": "fallback_real_only",
                                    "joint_primary_fallback_reason": "no_runtime_pseudo_members",
                                    "joint_primary_fallback_iter_per_kf": int(iter_per_kf),
                                },
                            )
                            self._run_legacy_real_keyframe_mapping(iter_per_kf, frames_to_optimize)
                        else:
                            # Joint-primary success branch: check for real densify fallback
                            enable_densify = bool(self.brpo_online_mapping_cfg.get("enable_densify", True))
                            fallback_enabled = bool(self.brpo_online_mapping_cfg.get("joint_primary_real_densify_fallback", False))

                            if fallback_enabled and not enable_densify:
                                self._run_joint_primary_real_gaussian_fallback(iter_per_kf)
                                self._update_brpo_event_summary(
                                    cur_frame_idx,
                                    {
                                        "joint_primary_status": "completed",
                                        "joint_primary_real_densify_fallback_applied": True,
                                        "joint_primary_real_densify_fallback_iter_per_kf": int(iter_per_kf),
                                    },
                                )

                            run_legacy_prune = bool(self.brpo_online_mapping_cfg.get("joint_primary_run_legacy_prune", True))
                            full_window = len(self.current_window) == self.config["Training"]["window_size"]
                            if run_legacy_prune and full_window:
                                self.map(self.current_window, prune=True)
                                self._update_brpo_event_summary(
                                    cur_frame_idx,
                                    {
                                        "joint_primary_status": "completed",
                                        "joint_primary_legacy_real_prune": True,
                                    },
                                )
                            else:
                                self._update_brpo_event_summary(
                                    cur_frame_idx,
                                    {
                                        "joint_primary_status": "completed",
                                        "joint_primary_legacy_real_prune": False,
                                        "joint_primary_legacy_real_prune_reason": "disabled" if not run_legacy_prune else "window_not_full",
                                    },
                                )
                            self._refresh_current_window_visibility(reason=f"joint_primary_kf_{cur_frame_idx}")
                    else:
                        self._run_legacy_real_keyframe_mapping(iter_per_kf, frames_to_optimize)
                        prepare_payload = self._maybe_prepare_brpo_runtime_slots(cur_frame_idx)
                        self._run_brpo_runtime_pseudo_mapping(cur_frame_idx, prepare_payload)
                    self.push_to_frontend("keyframe")
                else:
                    raise Exception("Unprocessed data", data)
        while not self.backend_queue.empty():
            self.backend_queue.get()
        while not self.frontend_queue.empty():
            self.frontend_queue.get()
        return
