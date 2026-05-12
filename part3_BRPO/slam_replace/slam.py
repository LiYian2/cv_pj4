import os
import glob
import sys
import time
import numpy as np
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import pandas as pd

import torch
import torch.multiprocessing as mp
import yaml
from munch import munchify

import wandb
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.system_utils import mkdir_p
from gui import gui_utils, slam_gui
from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.eval_utils import eval_ate, eval_rendering, save_gaussians
from utils.internal_eval_utils import export_internal_eval_cache
from utils.logging_utils import Log
from utils.multiprocessing_utils import FakeQueue
from utils.slam_backend import BackEnd
from utils.slam_backend_brpo import BRPOContinuationConfig, run_brpo_pseudo_continuation
from utils.slam_frontend import FrontEnd

from mast3r.model import AsymmetricMASt3R


class SLAM:
    def __init__(self, config, mast3r_model, save_dir=None):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        self.config = config
        self.save_dir = save_dir
        model_params = munchify(config["model_params"])    
        opt_params = munchify(config["opt_params"])
        pipeline_params = munchify(config["pipeline_params"])
        self.model_params, self.opt_params, self.pipeline_params = (
            model_params,
            opt_params,
            pipeline_params,
        )

        self.live_mode = self.config["Dataset"]["type"] == "realsense"
        self.monocular = self.config["Dataset"]["sensor_type"] == "monocular"
        self.use_spherical_harmonics = self.config["Training"]["spherical_harmonics"]
        self.use_gui = self.config["Results"]["use_gui"]
        if self.live_mode:
            self.use_gui = True
        self.eval_rendering = self.config["Results"]["eval_rendering"]      
        self.color_refinement = self.config["Results"]["color_refinement"]   
        self.global_BA = self.config["Results"]["global_BA"]

        model_params.sh_degree = 3 if self.use_spherical_harmonics else 0

        self.gaussians = GaussianModel(model_params.sh_degree, config=self.config)
        self.gaussians.init_lr(self.config["opt_params"]["init_lr"])        
        self.dataset = load_dataset(
            model_params, model_params.source_path, config=config
        )

        self.gaussians.training_setup(opt_params)
        bg_color = [0, 0, 0]
        # Keep the frontend/main-process background on CUDA as before, but do not
        # pass this CUDA tensor into the spawned BackEnd process. Spawn unpickles
        # Process attributes before BackEnd.run() can set the CUDA device, so CUDA
        # tensors here can initialize the child CUDA context on the wrong GPU.
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.background_cpu = list(bg_color)

        frontend_queue = mp.Queue()                                         
        backend_queue = mp.Queue()

        q_main2vis = mp.Queue() if self.use_gui else FakeQueue()            
        q_vis2main = mp.Queue() if self.use_gui else FakeQueue()            

        self.config["Results"]["save_dir"] = save_dir
        self.config["Training"]["monocular"] = self.monocular

        # Initialize frontend and backend queues
        self.frontend = FrontEnd(self.config, mast3r_model, self.save_dir)
        self.backend = BackEnd(self.config, self.save_dir)
        self.frontend.dataset = self.dataset
        self.frontend.background = self.background
        self.frontend.pipeline_params = self.pipeline_params
        self.frontend.frontend_queue = frontend_queue
        self.frontend.backend_queue = backend_queue
        self.frontend.q_main2vis = q_main2vis
        self.frontend.q_vis2main = q_vis2main
        self.frontend.set_hyperparams()

        self.backend.gaussians = self.gaussians
        self.backend.background = None
        self.backend.background_cpu = self.background_cpu
        self.backend.cameras_extent = 6.0           
        self.backend.pipeline_params = self.pipeline_params
        self.backend.opt_params = self.opt_params
        self.backend.frontend_queue = frontend_queue
        self.backend.backend_queue = backend_queue
        self.backend.live_mode = self.live_mode

        self.backend.set_hyperparams()

        self.params_gui = gui_utils.ParamsGUI(
            pipe=self.pipeline_params,
            background=self.background,
            gaussians=self.gaussians,
            q_main2vis=q_main2vis,
            q_vis2main=q_vis2main,
        )

        backend_process = mp.Process(target=self.backend.run)   
        if self.use_gui:
            gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui,))
            gui_process.start()
            time.sleep(5)

        backend_process.start()        
        self.frontend.run()
        backend_queue.put(["pause"])    

        end.record()
        torch.cuda.synchronize()
        # empty the frontend queue
        N_frames = len(self.frontend.cameras)
        FPS = N_frames / (start.elapsed_time(end) * 0.001)
        Log("Total time", start.elapsed_time(end) * 0.001, tag="Eval")
        Log("Total FPS", N_frames / (start.elapsed_time(end) * 0.001), tag="Eval")

        if self.eval_rendering:                         # Evaluation and rendering
            self.gaussians = self.frontend.gaussians
            kf_indices = self.frontend.kf_indices       # Get keyframe list indices from frontend for evaluation and rendering
            ATE = eval_ate(
                self.frontend.cameras,
                self.frontend.kf_indices,
                self.save_dir,
                0,
                final=True,
                monocular=self.monocular,
            )

            rendering_result = eval_rendering(
                self.frontend.cameras,
                self.gaussians,
                self.dataset,
                self.save_dir,
                self.pipeline_params,
                self.background,
                datatype=self.config["Dataset"]["type"],
                kf_indices=kf_indices,
                iteration="before_opt",
            )
            if self.config["Results"].get("export_internal_cache", False) or os.environ.get("S3PO_EXPORT_INTERNAL_CACHE", "0") == "1":
                export_internal_eval_cache(
                    self.frontend.cameras,
                    self.gaussians,
                    self.dataset,
                    self.save_dir,
                    self.pipeline_params,
                    self.background,
                    kf_indices,
                    stage_tag="before_opt",
                    metrics=rendering_result,
                    export_camera_states=True,
                )
            columns = ["tag", "psnr", "ssim", "lpips", "RMSE ATE", "FPS"]
            metrics_table = wandb.Table(columns=columns)
            metrics_table.add_data(
                "Before",
                rendering_result["mean_psnr"],
                rendering_result["mean_ssim"],
                rendering_result["mean_lpips"],
                ATE,
                FPS,
            )
            
            
            if self.color_refinement:
                # re-used the frontend queue to retrive the gaussians from the backend.
                while not frontend_queue.empty():       
                    frontend_queue.get()
                backend_queue.put(["color_refinement"])
                while True:
                    if frontend_queue.empty():
                        time.sleep(0.01)
                        continue
                    data = frontend_queue.get()             
                    if data[0] == "sync_backend" and frontend_queue.empty():
                        gaussians = data[1]
                        self.gaussians = gaussians
                        break

                rendering_result = eval_rendering(
                    self.frontend.cameras,
                    self.gaussians,
                    self.dataset,
                    self.save_dir,
                    self.pipeline_params,
                    self.background,
                    datatype=self.config["Dataset"]["type"],
                    kf_indices=kf_indices,
                    iteration="after_opt",
            )
                if self.config["Results"].get("export_internal_cache", False) or os.environ.get("S3PO_EXPORT_INTERNAL_CACHE", "0") == "1":
                    export_internal_eval_cache(
                        self.frontend.cameras,
                        self.gaussians,
                        self.dataset,
                        self.save_dir,
                        self.pipeline_params,
                        self.background,
                        kf_indices,
                        stage_tag="after_opt",
                        metrics=rendering_result,
                        export_camera_states=False,
                    )
                metrics_table.add_data(
                    "After_opt",
                    rendering_result["mean_psnr"],
                    rendering_result["mean_ssim"],
                    rendering_result["mean_lpips"],
                    ATE,
                    FPS,
            )
            save_gaussians(self.gaussians, self.save_dir, "final_after_opt", final=True)

            brpo_continuation_result = self._maybe_run_brpo_pseudo_continuation(kf_indices)
            if brpo_continuation_result is not None:
                if self.eval_rendering:
                    rendering_result = eval_rendering(
                        self.frontend.cameras,
                        self.gaussians,
                        self.dataset,
                        self.save_dir,
                        self.pipeline_params,
                        self.background,
                        datatype=self.config["Dataset"]["type"],
                        kf_indices=kf_indices,
                        iteration="after_pseudo_opt",
                    )
                    if self.config["Results"].get("export_internal_cache", False) or os.environ.get("S3PO_EXPORT_INTERNAL_CACHE", "0") == "1":
                        export_internal_eval_cache(
                            self.frontend.cameras,
                            self.gaussians,
                            self.dataset,
                            self.save_dir,
                            self.pipeline_params,
                            self.background,
                            kf_indices,
                            stage_tag="after_pseudo_opt",
                            metrics=rendering_result,
                            export_camera_states=False,
                        )
                    metrics_table.add_data(
                        "After_pseudo_opt",
                        rendering_result["mean_psnr"],
                        rendering_result["mean_ssim"],
                        rendering_result["mean_lpips"],
                        ATE,
                        FPS,
                    )
                save_gaussians(self.gaussians, self.save_dir, "final_after_pseudo_opt", final=True)

            wandb.log({"Metrics": metrics_table})

        backend_queue.put(["stop"])
        backend_process.join()          
        Log("Backend stopped and joined the main thread")
        if self.use_gui:               
            q_main2vis.put(gui_utils.GaussianPacket(finish=True))
            gui_process.join()
            Log("GUI Stopped and joined the main thread")

    def _resolve_brpo_continuation_cfg(self):
        result_cfg = self.config.get("Results", {}).get("brpo_pseudo_continuation", {}) or {}
        env_history = os.environ.get("S3PO_BRPO_CONTINUATION_STAGEA_HISTORY", "").strip()
        enabled = bool(result_cfg.get("enabled", False) or env_history)
        if not enabled:
            return None

        history_json = env_history or str(result_cfg.get("stageA_history_json", "")).strip()
        if not history_json:
            raise ValueError("BRPO continuation enabled but no stageA_history_json provided")

        def _env_or(name, current, cast):
            raw = os.environ.get(name)
            if raw is None or raw == "":
                return current
            return cast(raw)

        def _optional_result_value(name, cast=float):
            if name not in result_cfg:
                return None
            value = result_cfg.get(name)
            if value is None or value == "":
                return None
            return cast(value)

        output_dir = str(
            result_cfg.get("output_dir")
            or (Path(self.save_dir) / "after_pseudo_opt")
        )
        return BRPOContinuationConfig(
            stageA_history_json=history_json,
            num_iterations=_env_or("S3PO_BRPO_CONTINUATION_ITERS", int(result_cfg.get("num_iterations", 40)), int),
            num_pseudo_views=_env_or("S3PO_BRPO_CONTINUATION_NUM_PSEUDO", int(result_cfg.get("num_pseudo_views", 4)), int),
            extra_real_views=_env_or("S3PO_BRPO_CONTINUATION_EXTRA_REAL", int(result_cfg.get("extra_real_views", 2)), int),
            lambda_real=_env_or("S3PO_BRPO_CONTINUATION_LAMBDA_REAL", float(result_cfg.get("lambda_real", 1.0)), float),
            lambda_pseudo=_env_or("S3PO_BRPO_CONTINUATION_LAMBDA_PSEUDO", float(result_cfg.get("lambda_pseudo", 1.0)), float),
            lambda_depth=_env_or("S3PO_BRPO_CONTINUATION_LAMBDA_DEPTH", float(result_cfg.get("lambda_depth", 1.0)), float),
            beta_rgb=_optional_result_value("beta_rgb", float),
            lambda_pose=_optional_result_value("lambda_pose", float),
            lambda_exp=_optional_result_value("lambda_exp", float),
            trans_weight=_optional_result_value("trans_weight", float),
            lambda_abs_pose=_optional_result_value("lambda_abs_pose", float),
            lambda_abs_t=_optional_result_value("lambda_abs_t", float),
            lambda_abs_r=_optional_result_value("lambda_abs_r", float),
            abs_pose_robust=_optional_result_value("abs_pose_robust", str),
            update_real_pose=bool(result_cfg.get("update_real_pose", True)),
            update_pseudo_pose=bool(result_cfg.get("update_pseudo_pose", True)),
            use_depth=bool(result_cfg.get("use_depth", True)),
            split_pseudo_authority=bool(result_cfg.get("split_pseudo_authority", False)),
            pseudo_scene_mask_mode=str(result_cfg.get("pseudo_scene_mask_mode", "all_valid")),
            enable_densify=bool(result_cfg.get("enable_densify", False)),
            enable_prune=bool(result_cfg.get("enable_prune", False)),
            enable_opacity_reset=bool(result_cfg.get("enable_opacity_reset", False)),
            isotropic_weight=float(result_cfg.get("isotropic_weight", 10.0)),
            output_dir=output_dir,
            seed=int(result_cfg.get("seed", 0)),
        )

    def _maybe_run_brpo_pseudo_continuation(self, kf_indices):
        continuation_cfg = self._resolve_brpo_continuation_cfg()
        if continuation_cfg is None:
            return None
        Log(f"[BRPOContinuation] enabling post-after_opt continuation from {continuation_cfg.stageA_history_json}")
        result = run_brpo_pseudo_continuation(
            config=self.config,
            gaussians=self.gaussians,
            pipeline_params=self.pipeline_params,
            opt_params=self.opt_params,
            background=self.background,
            cameras=self.frontend.cameras,
            kf_indices=kf_indices,
            current_window=list(self.frontend.current_window),
            continuation_cfg=continuation_cfg,
            iteration_start=0,
            gaussian_update_every=self.config["Training"]["gaussian_update_every"],
            gaussian_update_offset=self.config["Training"]["gaussian_update_offset"],
            gaussian_th=self.config["Training"]["gaussian_th"],
            gaussian_extent=6.0 * self.config["Training"]["gaussian_extent"],
            gaussian_reset=self.config["Training"]["gaussian_reset"],
            size_threshold=self.config["Training"].get("size_threshold"),
        )
        Log("[BRPOContinuation] finished post-after_opt continuation")
        return result

    def run(self):
        pass


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")

    # Required arguments
    parser.add_argument("--config", type=str)
    parser.add_argument("--begin", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)

    # Optional arguments
    parser.add_argument("--color", type=float, default=None, help="color refinement")
    parser.add_argument("--alpha", type=float, default=None, help="Loss parameter")
    parser.add_argument("--pr", type=float, default=None, help="pixel_ratio")
    parser.add_argument("--iter", type=int, default=None, help="iteration count of pose optimization")
    parser.add_argument("--windowsize", type=int, default=None, help="window size of local BA")
    parser.add_argument("--patch_size", type=int, default=None, help="patch size")
    parser.add_argument('--ns', type=int, default=None, help='mapping_itr_nosingle override')
    parser.add_argument('--sh', type=int, default=None, help='spherical harmonics degree override')
    

    args = parser.parse_args(sys.argv[1:])

    mp.set_start_method("spawn")          

    with open(args.config, "r") as yml:
        config = yaml.safe_load(yml)

    config = load_config(args.config)
    save_dir = None

    if args.alpha is not None:
        config["Training"]["alpha"] = args.alpha
        
    if args.ns is not None:
        config["Training"]["mapping_itr_nosingle"] = args.ns
        
    if args.pr is not None:
        config["depth"]["min_accurate_pixels_ratio"] = args.pr

    if args.iter is not None:
        config["Training"]["tracking_itr_num"] = args.iter

    if args.windowsize is not None:
        config["Training"]["window_size"] = args.windowsize

    if args.begin is not None:
        config["Dataset"]["begin"] = args.begin

    if args.end is not None:
        config["Dataset"]["end"] = args.end

    if args.sh is not None:
        config["model_params"]["sh_degree"] = args.sh

    if args.patch_size is not None:
        config["depth"]["patch_size"] = args.patch_size
        
    if args.color:
        config["Results"]["color_refinement"] = True

    if config["Results"]["save_results"]:
        mkdir_p(config["Results"]["save_dir"])
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")   
        path = config["Dataset"]["dataset_path"].split("/")
        save_dir = os.path.join(
            config["Results"]["save_dir"], path[-3] + "_" + path[-2], current_datetime
        )
        tmp = args.config
        tmp = tmp.split(".")[0]
        config["Results"]["save_dir"] = save_dir
        mkdir_p(save_dir)
        with open(os.path.join(save_dir, "config.yml"), "w") as file:      
            documents = yaml.dump(config, file)
        Log("saving results in " + save_dir)
        run = wandb.init(
            project="S3PO-GS",
            name=f"{tmp}_{current_datetime}",
            config=config,
            mode=None if config["Results"]["use_wandb"] else "disabled",
        )
        wandb.define_metric("frame_idx")
        wandb.define_metric("ate*", step_metric="frame_idx")
        
    model_name = "/home/bzhang512/CV_Project/third_party/S3PO-GS/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"  # Use local checkpoint
    mast3r_model = AsymmetricMASt3R.from_pretrained(model_name).to("cuda")
    
    slam = SLAM(config, save_dir=save_dir,mast3r_model=mast3r_model)

    slam.run()
    device = "cuda"
    wandb.finish()

    # All done
    Log("Done.")
