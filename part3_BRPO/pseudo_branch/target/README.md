# target

Live target-side modules still needed by the current Part3 tree.

Current online-mapping target entry kept in `pseudo_branch/target/`:
- `depth_supervision_v2.py` — exact-upstream target construction consumed by the live online route

Temporarily retained bridge / utility helpers:
- `depth_target_builder.py` — shared depth load / reprojection utilities, still re-exported from `pseudo_branch`
- `support_expand.py` — legacy/bridge support-expansion helper, not part of the live online-mapping runtime route

Archived out of the live target package on 2026-05-10:
- `brpo_depth_target.py`
- `brpo_depth_densify.py`

Archived files now live under:
- `legacy_or_archive/pseudo_branch_legacy/target/`
