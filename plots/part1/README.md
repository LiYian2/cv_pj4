# Part 1 Plots

This folder stores plotting artifacts for Part 1.

## Main figures

- `part1_combined_psnr_l1.png`
  - Left: PSNR curves
  - Right: L1 curves
  - Colors indicate methods (`PlanA-279`, `PlanA-96`, `PlanB-96`)
  - Solid lines indicate test curves
  - Dashed lines indicate train curves

- `part1_test_psnr_curve_clean.png`
  - Clean test-only PSNR convergence plot

- `part1_test_l1_curve_clean.png`
  - Clean test-only L1 convergence plot

## CSV files

- `convergence_long.csv`: long-format table for plotting / pandas / seaborn
- `convergence_wide.csv`: wide-format table for Excel / quick inspection
- `final_summary.csv`: final 40k summary values

## Recommended usage in report

For the main paper figure, prefer:

- `part1_combined_psnr_l1.png`

For cleaner single-metric plots or appendix:

- `part1_test_psnr_curve_clean.png`
- `part1_test_l1_curve_clean.png`
