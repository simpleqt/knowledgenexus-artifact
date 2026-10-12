# KnowledgeNexus Evaluation Artifact

Artifact release for:

1. **KnowledgeNexus: A Traceable Graph–Vector Knowledge Infrastructure and
   Adaptive Multi-Channel Retrieval for Nuclear Science Literature**
2. **The Lexicon Is the Load-Bearing Wall** (companion lexicon-quality paper)

LaTeX sources and compiled preprints of both papers are under `papers/latex/`.

## Layout

| Path | Contents |
|---|---|
| `eval/harness/` | Question sets (association 48 / real queries 39, hard93, csh cross-domain sets, paraphrase holdouts) and the corpus-first question generator `gen_assoc_set.py` |
| `eval/baselines/scripts/` | All evaluation drivers: scoped in-process runner (`run_knc_scoped.py`), channel ablation (`run_knc_scoped_ablate_channels.py`), link-precision audit (`audit_link_precision.py`), external baselines (`run_hipporag.py`, LlamaIndex, BM25), paired-bootstrap scoring (`score.py`) |
| `eval/baselines/data/` | Slice document lists, question variants, and the three English multi-hop benchmark pools (400-doc pools, HotpotQA/2Wiki/MuSiQue) |
| `eval/baselines/runs/` | Per-query prediction files backing the paper tables: channel-ablation arms, scoped KNC arms, genrag capability-pipeline arms, and the same-embedder English benchmark arms |
| `eval/baselines/results/` | Result archives: `20261009_chab` (single-variable entity-channel ablation) and `20261009_linkaudit` (100-link precision audit with per-link adjudications) |
| `corpus/` | Three-source corpus-attested lexicon generator (`gen_term_lexicons.py`) and full-chain entity-layer rebuild driver (`rebuild_graph_enrich.py`) |

## Headline numbers reproduced here

- **Entity-channel ablation** (paper 1 §7.2, paper 2 §6.3): with every
  entity-dependent channel disabled, both question families are bitwise
  identical to the full configuration (cov@10/R@10/MRR; nea cov@20 −1.0pt,
  CI [−3.1, 0.0]) while p50 latency halves 1.46 s → 0.52 s. Note for
  re-runners: the co-occurrence channel ignores the `channels` request
  parameter and must be killed via `KNC_COOC_OFF=1` (see the ablation
  script header).
- **Link precision audit** (paper 2 §6.4): n=100 seeded sample from the
  rebuilt KB, span integrity 100/100, precision 0.850 (Wilson 95% CI
  [0.767, 0.907]); domain-term stratum 25/25 = 1.000, generic stratum
  60/75 = 0.800.
- **Same-embedder English benchmark control** (paper 1 §7.1): all arms on
  Qwen3-Embedding-8B, 400-doc pools — HippoRAG 2 (reproduced) beats plain
  dense on 2 of 3 benchmarks; the cross-embedder gap in the published
  comparison belongs to the embedding model generation.
- **Capability pipeline vs assets** (paper 1 §7.6): query-time LLM
  (rewrite→fuse→rerank) wins association cov@10 +10.1pt at ~60× latency
  and zero asset accumulation.

## Reproduction prerequisites

The drivers run **in-process against the deployed KnowledgeNexus stack**
(MySQL/OpenSearch/vLLM endpoints injected via environment; see
`eval/baselines/scripts/knc_env.sh` for the variable contract — no
credentials are stored in this repository). Scoring is fully offline:

```bash
python3 eval/baselines/scripts/score.py \
  eval/baselines/runs/chab-all_nea_L.json \
  eval/baselines/runs/chab-noentity_nea_L.json \
  --ref eval/baselines/runs/chab-all_nea_L.json
```

The journal corpora are third-party copyrighted and not redistributable;
the benchmark pools under `eval/baselines/data/bench/` contain the
HippoRAG 2 evaluation subsets used in the paper.

## License / access

Evaluation code and question sets: released for review and replication.
Third-party components (benchmark data, model names) remain under their
respective licenses.

## Cross-domain replay package (2026-10-12)

`eval/cross_domain/` — three public domains (arXiv cs.CL / PubMed /
cond-mat.mtrl-sci; 10k abstracts each; construct 3000 / frozen test 100 /
10×~690 disjoint replay, seeded splits) × five lexicon arms (batch
schema / frequency / C-value / LLM-direct / three-source adapted).
Contains the drivers, per-domain arm lexicons (components/), silver +
blind-judge + cross-family arbitration records (ctx excerpts stripped;
doc-id manifests under corpus_manifest/ re-derive context), structural
scan aggregates, per-domain tables.json, the CS source ablation, and
the silver instrument-sensitivity rerun (silver_stability.json).
`eval/scanner_parity/` — replica-vs-production scanner parity on 200
frozen KB1 chunks (span-level 913/913 coverage; divergences attributed).
Source corpora are not redistributed.

## Revision-round evidence (2026-10-12, second wave)

`eval/revision20261012/` — recall funnel (R = P(C)·P(K|C)·P(D|K,C)·P(L)),
monitor leave-one-library-out validation, scanner parity, silver
instrument stability, short-symbol ambiguity four-policy comparison
(mentions + double-judge records), and the risk-constrained selection
study (wide pools, construct-side dev silver, learned/coverage/cover+risk
arms with budget curves). Cross-domain package updated with the new arms'
lexicons, judge/arbitration records (ctx stripped), and tables.

## Dual-annotation reference (2026-10-12, third wave)

`eval/dual_annotation/` — 200 complete units (80 KB1 frozen chunks + 40
test docs per public domain) cross-annotated in full text by two model
annotators (deepseek-flash + deepseek-v4-pro; family overlap disclosed;
a gateway thinking-mode model is forensically unusable for full-document
annotation), disagreements arbitrated by the cross-family Qwen judge.
Agreement: domain-set Jaccard 0.59-0.66, termness agreement 94.5-98.8%.
Eight-arm recall under the adjudicated reference incl. the first KB1
lexicon-recall estimate (v2.0 1.0% -> v2.1 4.3%). Unit texts are not
redistributed (manifest with ids/lengths only).

## Major-revision P0/P1 controls (2026-10-13)

`eval/revision20261013_p0/` — language-aligned schema control (CS
collapse is semantic, not orthographic), chunk-to-document sampling
stats, four-leg composite monitor (zero misses/false alarms on the
frozen degradation archive), overlap-suppression sensitivity (arm
orderings unchanged), KB1 fixed-budget sweep (v2.1-prefix vs
frequency-top vs C-value-top at K=200..3733 with dual-annotated
recall), entity-cluster bootstrap CIs, expansion-module adjudication,
and third-model verification of the dual-annotation consensus.
