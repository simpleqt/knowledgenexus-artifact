# Review round 2 (2026-10-13): unified-typing baselines + translation control

Fair-comparison controls requested by the second major-revision review:

1. **Unified typing** (`unified_typing.json`, `judge_*.jsonl`, `arbit_*.jsonl`):
   the equal-size frequency/C-value arms' entity types were randomly assigned
   (and types enter the blind-judge prompt). Term sets frozen; types re-assigned
   by the same LLM classifier the three-source arm uses (classifier-rejected
   terms stay `unclassified`). Random typing disagreed with the classifier on
   91-98% of classifiable terms; unified typing shifts validity by -3 to +13
   points (freq 0.405/0.362/0.450, cvalue 0.400/0.400/0.333 in CS/bio/mat) -
   nowhere near three-source (0.838/0.850/0.912). Dominant deficit is candidate
   selection; the classifier accepts only 6-19% of frequency-arm terms.
   KB1 budget sweep re-typed (`kb1_budget_curve_llmtypes.json`): R identical
   at every budget - typing-invariant.
   Judge/arbit records are ctx-stripped (ctx_len kept) per corpus policy.

2. **Translation control** (`cs_translation_control.json`,
   `zh_translation.json`): the CS schema's non-ASCII aliases are exactly 16
   type-name translations; translating them to English adds 2 net-new
   surfaces, 0 corpus hits; every metric identical to deletion-based
   alignment. No Chinese terminology to lose - the collapse is semantic.

Driver: `lexicon_unified_typing.py` (parts xdomain | kb1 | translate).
Note: an intermediate run wrote judge files without domain prefixes causing
bio/mat to reuse CS records; fixed and re-run - only the per-domain files
here are authoritative.
