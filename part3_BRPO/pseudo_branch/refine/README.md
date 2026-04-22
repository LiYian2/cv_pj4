# refine

Stage runtime state, loss assembly, scheduler, and refine-side orchestration.

Phase 2 direct migration has landed: `pseudo_camera_state.py`, `pseudo_loss_v2.py`, and `pseudo_refine_scheduler.py` now live under this directory.
Callers should import from `pseudo_branch.refine.*` rather than the retired top-level R~ paths.
