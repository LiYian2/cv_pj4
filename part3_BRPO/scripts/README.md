# scripts

Top-level `scripts/` now keeps only live entrypoints, active builders, and still-consumed utilities.
Historical compare launchers and one-off runners have been moved under `scripts/archive_experiments/` to keep the live workspace small and readable.

Current top-level categories:
- live refine entry: `run_pseudo_refinement_v2.py`
- legacy-but-still-referenced utilities: `run_pseudo_refinement.py`, `diagnose_stageA_gradients.py`
- signal / target builders: `brpo_build_mask_from_internal_cache.py`, `build_brpo_v2_signal_from_internal_cache.py`, `prepare_stage1_difix_dataset_s3po_internal.py`, etc.
- analysis / replay helpers: `replay_internal_eval.py`, `summarize_stageA_compare.py`, `select_signal_aware_pseudos.py`

Archive layout:
- `archive_experiments/a1/`: old A1 / target-side compare runners
- `archive_experiments/g/`: old G-BRPO compare / smoke runners
- `archive_experiments/stageA/`: old StageA / abs-prior launchers
- `archive_experiments/topology/`: old topology compare runners
- `archive_experiments/legacy_entry/`: superseded training entrypoints

Cleanup rule for future additions:
- if a script is a one-off launcher or historical compare wrapper, archive it after the result is landed in docs
- if a script is a live builder / consumer / diagnosis entry that current docs still point to, keep it at top level
