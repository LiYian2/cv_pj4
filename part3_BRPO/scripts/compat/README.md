# scripts/compat

Narrow compatibility layer kept only for internal or legacy path bridging.

Current member:
- `run_pseudo_refinement.py`: loads the archived legacy refine entrypoint under `scripts/archive_experiments/legacy_entry/`

Rule: new non-live code should not accumulate here. This directory exists only when a tiny compatibility boundary is cheaper than keeping old implementation paths at top level.
