# observation

Joint M~/T~ observation builders, verifier glue, and fusion-side assembly.

Phase 3 direct migration has landed the current observation entrypoints: `pseudo_fusion.py`, `brpo_reprojection_verify.py`, `pseudo_observation_brpo_style.py`, `pseudo_observation_verifier.py`, and `joint_observation.py` now live under this directory.
Callers should import from `pseudo_branch.observation.*` rather than the retired top-level observation paths.
