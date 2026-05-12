# mask

Live online mask / confidence support modules for the current Part3 route.

Current online-mapping mask entries kept in `pseudo_branch/mask/`:
- `rgb_mask_inference.py` — reciprocal-seed RGB-only support / confidence construction
- `dense_match_densify.py` — optional `dense_match_v1` disk+blur+normalize support branch
- `cm_local_expansion.py` — optional local soft `C_m` expansion branch

Temporarily retained bridge helper:
- `joint_confidence.py` — still consumed by legacy/bridge signal builders, not by the live online-mapping runtime route

Archived out of the live mask package on 2026-05-10:
- `brpo_confidence_mask.py`
- `brpo_train_mask.py`
- `confidence_builder.py`

Archived files now live under:
- `legacy_or_archive/pseudo_branch_legacy/mask/`
