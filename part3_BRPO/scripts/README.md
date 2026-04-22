# scripts

Top-level `scripts/` keeps only live entrypoints plus one explicitly retained external CLI wrapper for legacy path stability.
Internal non-live helpers have been pushed into dedicated subdirectories so the top level stays aligned with the active pipeline.

Current layout:
- top-level live core: `run_pseudo_refinement_v2.py`, `replay_internal_eval.py`, `prepare_stage1_difix_dataset_s3po_internal.py`, `build_brpo_v2_signal_from_internal_cache.py`, `brpo_build_mask_from_internal_cache.py`, `brpo_verify_single_branch.py`, `select_signal_aware_pseudos.py`, `materialize_m5_depth_targets.py`
- top-level external CLI wrapper only: `run_pseudo_refinement.py`
- internal compatibility layer: `scripts/compat/`
- diagnostics / summary helpers: `scripts/diagnostics/`
- legacy prepare / historical utilities: `scripts/legacy_prepare/`
- archived compare runners: `scripts/archive_experiments/`

Cleanup rule for future additions:
- top level only for current live pipeline entrypoints, or a deliberately retained external CLI wrapper
- if a script is a tiny compatibility boundary used by internal callers, place it under `scripts/compat/`
- if a script is a non-live diagnosis / summary helper, place it under `scripts/diagnostics/`
- if a script is an older prepare path retained only for history / fallback, place it under `scripts/legacy_prepare/`
- if a script is a one-off launcher or historical compare wrapper, archive it under `scripts/archive_experiments/` after the result is landed in docs
