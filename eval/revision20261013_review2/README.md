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

3. **From-scratch human enumeration, recall side** (`enumeration/`): the
   author enumerated domain terms from zero on 30 held-out abstracts
   (10/domain, frozen test splits, no candidates shown; 424 raw -> 418
   unique surfaces). Matched against the five cross-domain arms by exact
   case-folded surface: batch schema 0/418 (0 even among the 83 terms
   recurring in >10 construction-corpus documents); size-aligned arms cover
   16.9-22.9% of that core stratum; LLM-direct 47.0% with a 4.6x larger
   unvalidated lexicon. Half the terms (209) never occur in the 3,000-doc
   construction corpus - paper coinages no bounded static lexicon can carry
   (mention-layer scope). Per-1,000-entry recall efficiency: cvalue 16.6 >
   three-source 14.8 > freq 12.2 > llmdirect 10.5. The enumeration sheet
   itself contains public-corpus abstract text and is not redistributed;
   per-term surfaces, doc-frequency counts, and strata are in the two JSONs.
   Drivers: `enumeration_recall.py`, `enumeration_attestation.py`.
