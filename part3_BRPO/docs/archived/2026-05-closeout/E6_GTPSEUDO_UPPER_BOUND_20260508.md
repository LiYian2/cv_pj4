# E6 GT-Pseudo Upper-Bound Experiment Plan

## Purpose

E6 tests the upper bound of the E5 online-mapping method by removing pseudo-render RGB quality as a bottleneck.

Compared with E5:
- Difix disabled.
- Pseudo RGB source is the ground-truth dataset image.
- Pseudo RGB is used for matching, RGB-only C_m/mask generation, and pseudo RGB supervision.
- Depth stays the same as E5: projected bidirectional exact backend (`depth_generation_mode: projected`).
- A/B split is preserved:
  - E6a: `update_real_pose: false`, `update_pseudo_pose: true`.
  - E6b: `update_real_pose: true`, `update_pseudo_pose: true`.

Important scope note: this does not remove rendered depth. E5 projected-depth supervision still needs current-map pseudo/ref depth for geometric verification and bidirectional projected depth target construction. What is removed is rendered pseudo RGB as target/matching input.

## Code switch

New config key:

```yaml
Results:
  brpo_online_mapping:
    pseudo_rgb_source: gt  # default: render
```

Runtime behavior:
- default `render`: unchanged existing behavior.
- `gt`: load `pseudo_state.image_path` as pseudo RGB target/matching input, while retaining the projected depth route.

## Files

Configs:
- `configs/e6a_jointprimary_maskedcolor_rgbonly_cm_gtpseudo_nodifix.yaml`
- `configs/e6b_jointprimary_maskedcolor_rgbonly_cm_gtpseudo_nodifix_realpose.yaml`

Launchers:
- `scripts/run_e6a_jointprimary_maskedcolor_rgbonly_cm_gtpseudo_nodifix.sh`
- `scripts/run_e6b_jointprimary_maskedcolor_rgbonly_cm_gtpseudo_nodifix_realpose.sh`
- `scripts/run_e6_gtpseudo_pair.sh`

Output roots:
- `/data3/bzhang512/part3_online_mapping_experiments/E6a_jointprimary_maskedcolor_rgbonly_cm_gtpseudo_nodifix`
- `/data3/bzhang512/part3_online_mapping_experiments/E6b_jointprimary_maskedcolor_rgbonly_cm_gtpseudo_nodifix_realpose`

## Verification already done

- Python syntax check passed for changed runtime modules.
- Shell syntax check passed for E6 launchers.
- Resolver check passed:
  - `pseudo_rgb_source: gt`
  - `use_difix_restoration: false`
  - `depth_generation_mode: projected`
  - `rgb_only_verification: true`
  - E6a `update_real_pose: false`
  - E6b `update_real_pose: true`

## Queue state

E6 was not automatically started when created, to avoid racing the existing E5 wait process. Recommended options:
1. run E6 immediately if enough memory is available, since no-Difix should be much lighter;
2. append E6 after the existing E5 pair;
3. pause/cancel E5 wait and run E6 first if the upper-bound check is now higher priority.
