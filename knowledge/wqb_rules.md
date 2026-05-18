# WQB Submission Rules For Generation

- Generate FASTEXPR only. The worker must simulate and pass guard checks before any submit path.
- Treat `PENDING`, `FAIL`, and `ERROR` submission checks as blockers.
- Treat self-correlation or production correlation above `0.7` as a blocker.
- Respect the service cap of at most 4 final submits per round.
- Do not copy prior submitted alphas. Use prior results as structure and risk evidence only.
- Favor formulas whose hypothesis can survive sub-universe, ladder, weight concentration, and correlation checks.
