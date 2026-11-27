# scripts

Helpers that are not part of the `dac` package. Run them from the repository root with
`.venv/bin/python <script>`; they read and write `results/`.

- `compare.py BASE OTHER...`: paired AUC differences (OTHER − BASE) per seed, with a 95% bootstrap
  interval. `a+b` averages two result files seed by seed (e.g. two B seeds).
- `probes/hybrid.py HIGH NAME`: hand rule, lr HIGH from block 6 until the run first scores ≥ 400,
  otherwise lr_gate. Validation seeds 30–59 only. Headroom probe from 2026-11-22.
- `probes/period_rule.py SHORT MODE NAME`: hand rule, lr by lr_gate, A's target-copy period SHORT
  early and 100 later (MODE `sticky`, `gate` or `q<T`). Validation seeds 30–59 only.
- `probes/late_period.py LONG MODE NAME`: hand rule, lr by lr_gate, A's target-copy period LONG once
  the score is high (MODE `gate` or `sticky`), else 100. Validation seeds 30–59 only.

The probes are not yet controllers in `dac/`; if one becomes a reported baseline it should be made a
proper controller with tests.
- `e5_nstep1.sh`: trains and validates the best B with 1-step targets (experiment E5).
- `e6_periods.sh`: trains B seeds 0 and 1 with lr × target-period actions and validates them (experiment E6).
