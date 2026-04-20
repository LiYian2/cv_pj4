# A1 Strict BRPO Alignment Checklist

## Goal
Freeze a non-hybrid checklist for A1 so future changes stop drifting between “exact BRPO replication” and “BRPO-inspired engineering variants”.

## Scope split
1. Layer A: BRPO C_m / joint weighting semantics.
2. Layer B: full BRPO pseudo-frame generation pipeline.
3. Current work only claims Layer A alignment unless explicitly stated otherwise.

## Current agreed checklist
1. Pseudo-frame source
- Current executable proxy pseudo-frame is `fusion/samples/<frame_id>/target_rgb_fused.png`.
- This is a proxy for BRPO `I_t^{fix}`, not a claim that the full upstream BRPO restoration/fusion stack has been replicated.

2. Exact BRPO C_m domain
- `C_m` must be built from fused pseudo-frame correspondence support sets against left/right real references.
- Exact `C_m` means: both -> 1.0, xor -> 0.5, none -> 0.0.
- Exact `C_m` must not introduce unlabeled extra gates from overlap confidence thresholds, projected-depth validity filters, or other geometry-side heuristics.

3. Hybrid label rule
- If a mode additionally gates `C_m` by overlap masks, overlap confidence, projected-depth availability, or any other geometry-side condition, it is not exact BRPO.
- Such a mode must be labeled `hybrid_*`, not `exact_*`.

4. Shared RGB/depth mask contract
- For any BRPO-style A1 mode, RGB and depth must consume the same `C_m` mask.
- Any mode using different RGB/depth masks is outside strict BRPO A1 semantics.

5. Depth-target contract separation
- `C_m` exactness and depth-target exactness are different questions.
- A mode may be `exact_brpo_cm_*` while reusing a compatibility target contract.
- If the depth target is not a strict BRPO replication, that limitation must be stated in the mode name or metadata.

6. Fallback / stabilization contract
- Reusing `target_depth_for_refine_v2_brpo`, stable-target blending, or hybrid direct projected-depth composition are all target-contract choices.
- They are not part of exact `C_m` unless explicitly justified and documented.

## Mode taxonomy after this split
- `old A1`: old support-filter baseline.
- `brpo_joint_v1`: current candidate-built joint observation rewrite.
- `brpo_style_v1/v2`: BRPO-inspired transitional builders.
- `hybrid_brpo_cm_geo_v1` / historical artifact name `brpo_direct_v1`: hybrid mode that mixes BRPO-like support sets with geometry-side gating.
- `exact_brpo_cm_old_target_v1`: exact `C_m` + old target contract.
- `exact_brpo_cm_hybrid_target_v1`: exact `C_m` + current hybrid direct target contract.
- `exact_brpo_cm_stable_target_v1`: exact `C_m` + stable-target blend contract.

## Required compare logic
1. `old A1` vs `exact_brpo_cm_old_target_v1` isolates C_m change under old target contract.
2. `hybrid_brpo_cm_geo_v1` vs `exact_brpo_cm_hybrid_target_v1` isolates geometry-gated hybrid C_m vs exact C_m under the same target contract.
3. `exact_brpo_cm_hybrid_target_v1` vs `exact_brpo_cm_stable_target_v1` isolates target / fallback / stabilization contract differences under the same exact C_m.

## Non-negotiable documentation rule
Every future A1 doc/run summary must state explicitly whether it is:
- exact BRPO C_m,
- hybrid BRPO-inspired C_m,
- or only a target-contract ablation under exact C_m.
