# Part 2 Build Order

Current agreed order:

1. create directory scaffold;
2. write adapter spec;
3. build the first preprocessing notebook;
4. after review, continue with config generation and batch execution.

## Important decisions

- Keep Part 2 file management aligned with Part 1.
- Preserve the `dataset / output / results / plots` separation.
- Preprocessing is notebook-first.
- Derived data should prefer symlink over copy.
- RegGS scene directories should preserve the full sequence and rely on `sample_rate` / `n_views` for split behavior.

## Path convention

Canonical prepared-scene path: `dataset/<scene>/part2/`

This matches the existing Part 1 organization (`dataset/<scene>/part1/`) and is preferred over a single shared `dataset/part2/` bucket.
