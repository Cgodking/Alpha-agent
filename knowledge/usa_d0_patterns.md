# USA D0 Pattern Notes

For `region=USA` and `delay=0`, prefer structures that look structurally different from generic USA D1 alphas:

- Model or other predictive fields with `ts_backfill`, `winsorize`, `/ cap`, and `ts_rank`.
- Industry or subindustry `group_rank` for neutralization-compatible shape.
- Sector-aware scaling such as `group_scale(..., densify(...))` when a sector field is available.
- Reversal legs using short price change, for example `group_rank(-ts_delta(close, 1), industry)`, only as one component of a larger signal.

Use these as motifs, not copied formulas. If a referenced field is unavailable, substitute an analogous model, estimate, sentiment, or fundamental field that exists in the active dataset surface.
