# gaussian_management

G~ packages and Gaussian-side control helpers.

Phase 1 direct migration has landed: `local_gating/`, `spgm/`, and `gaussian_param_groups.py` now live under this directory.
Callers should import from `pseudo_branch.gaussian_management.*` rather than the retired top-level G~ paths.
