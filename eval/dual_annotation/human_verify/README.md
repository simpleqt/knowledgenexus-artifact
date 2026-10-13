# Human verification of the dual-annotated reference (2026-10-13)

Verification of `../adjudicated.jsonl` by a single author on a seeded,
stratified sample: 20 units (KB1 frozen chunks + CS/biomedicine/materials),
197 adjudicated terms, per-row verdicts ok/wrong/unsure against the unit's
own text.

- `human_verify_result.json` — stats + all 197 per-row verdicts
  (confirm 0.934, Wilson 95% [0.890, 0.961]; by scope 0.833/0.967/0.925/
  0.950; consensus-origin 0.949 vs arbitrated 0.853-0.960).
- `verification_verdicts.md` — verdicts-only copy of the filled sheet,
  row order identical to the original.

The self-contained verification sheet (with inline unit texts) is NOT
distributed: KB1 units are licensed journal text (same policy as
`units.json`, see `../units_manifest.json`). Scope limits: verification
bounds precision of the model-annotated reference only; it does not
measure omissions and is single-author.
