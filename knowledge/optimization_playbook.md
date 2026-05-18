# Alpha Optimization Playbook

Use this playbook when the experiment plan is `optimize_best` or `setting_sweep`.

## When To Optimize

- Optimize only when the best candidate is near the official scope thresholds or has a high readiness score.
- Do not optimize candidates blocked by hard structural failures such as concentrated weight, bad turnover, self-correlation, production correlation, data diversity failure, or invalid syntax.
- If Sharpe and Fitness are far below scope thresholds, switch field family instead of making local parameter edits.
- Treat official scope thresholds as primary. USA D0, CHN, IND, EUR, GLB, HKG, KOR, ASI, and MEA can have different requirements.

## Optimization Order

- First preserve the economic mechanism and fix settings when the expression is already close: neutralization, decay, truncation, and pasteurization/max-trade settings.
- Then make controlled expression edits: windows, winsorize strength, rank placement, normalization denominator, and one confirmation leg.
- Change only one or two dimensions per candidate so simulation feedback can be attributed.
- Avoid changing field family, sign, window, normalization, and grouping all at once; that becomes new exploration, not optimization.

## Candidate Design Rules

- Keep field identifiers valid and copied from the active datafield context.
- Prefer robust preprocessing: `ts_backfill`, `winsorize`, `ts_rank`, and group normalization.
- For noisy model, analyst, text, or fundamental fields, use windows in the 33/66/120 range before trying very short windows.
- Use `group_rank(x, industry)` or `group_rank(x, subindustry)` to reduce concentration risk.
- Add a confirmation leg only when it has a distinct mechanism, such as forecast quality plus value, sentiment persistence plus quality, or model score plus low error.
- Avoid price-only rescue legs unless they are secondary and explicitly justified.

## Failure Response

- `LOW_SHARPE` plus `LOW_FITNESS`: do not only tune decay; revise the signal mechanism or field family.
- `CONCENTRATED_WEIGHT`: add grouping, winsorization, rank normalization, or divide by cap/volatility where economically valid.
- `IS_LADDER_SHARPE` or `LOW_2Y_SHARPE`: prefer longer windows, persistence, and smoother fields; avoid short reversal-only formulas.
- `LOW_SUB_UNIVERSE_SHARPE`: reduce universe-specific behavior with group normalization and less concentrated fields.
- `SELF_CORRELATION` or `PROD_CORRELATION`: keep the hypothesis but change field family or mechanism enough to be structurally orthogonal.
- `DATA_DIVERSITY`: use a different dataset family, not a renamed field in the same family.

## Lineage And Stop Rules

- Preserve `optimization_anchor_id` lineage for controlled variants.
- After three optimization or setting-sweep rounds without at least 20 percent improvement, abandon the family and explore a structurally different one.
- Do not re-optimize an abandoned anchor unless new simulation evidence shows a materially better variant.
- Record why a candidate failed so the next controller plan can assign different guidance to each generator profile.

## Multi-Model Use

- The controller should give directions, not candidate counts.
- G-1 and G-2 should optimize different dimensions or mechanisms, not duplicate the same local edits.
- The critic should challenge repeated field families, invalid fields, overfitting, and weak threshold assumptions.
- The validator should remove invalid syntax and true local duplicates, but should not reject a robust template reused on a genuinely different field family.
