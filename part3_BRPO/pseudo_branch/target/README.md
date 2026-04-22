# target

Depth-target builders, densification helpers, reprojection helpers, supervision assembly, and target-side support expansion.

Current live target-side entries are all under this directory:
- `brpo_depth_target.py`
- `brpo_depth_densify.py`
- `depth_target_builder.py`
- `depth_supervision_v2.py`
- `support_expand.py`

Callers should import from `pseudo_branch.target.*`.
The old flat target-side paths `pseudo_branch/brpo_depth_target.py`, `pseudo_branch/brpo_depth_densify.py`, and `pseudo_branch/depth_target_builder.py` are retired.
