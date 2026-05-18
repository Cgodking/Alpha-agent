# Research-Grade Generation Patterns

Prefer candidates that combine at least two meaningful ideas:

- Cross-sectional ranking plus time-series behavior, for example `group_rank(ts_rank(...), industry)`.
- Robust preprocessing for noisy fundamentals or model fields, for example `winsorize(ts_backfill(field, 120), std=3)`.
- Scale-aware normalization such as `/ cap` when using size-sensitive raw fields.
- Reversal or confirmation legs such as `group_rank(field_signal, industry) * group_rank(-ts_delta(close, 1), industry)`.
- Use windows such as `5`, `22`, `33`, `63`, `66`, `120`, `252`, and `504`.

Avoid low-information candidates:

- `rank(close)`, `rank(volume)`, `rank(returns)`.
- One-line price-only momentum with no grouping, normalization, or secondary signal.
- Tiny parameter edits around a recently failed expression.

## Multi-Model Cooperation

During explore or explore_new_family cycles, the service allocation is fixed externally. Do not ask the controller to decide how many candidates each model should generate.

- `gemini` should search one structurally coherent new family, emphasizing fresh field families, simple field-native mechanisms, and low syntax risk.
- `minimax` should search a different family, emphasizing alternative economic mechanisms, orthogonal fields, and structures that avoid the same failed anchors.
- The controller should provide per-profile research directions only: objective, field family, signal mechanism, formula structure, and avoid list.
- Avoid rationale such as "allocate all capacity to one model"; exploration capacity is balanced by the service.
