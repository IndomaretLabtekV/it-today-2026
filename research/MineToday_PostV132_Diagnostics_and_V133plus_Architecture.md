# MineToday Post-V132 Diagnostics and V133+ Architecture

**Decision date:** 2026-09-08  
**Protected public result:** V132 = **0.66219**, rank **#2** (user-supplied); #3 = 0.66210 and #1 = 0.96609.  
**Scope:** post-V132 diagnosis and the next research architecture only. This document does not implement V133 and does not authorize a submission by itself.

Evidence labels used below are deliberately strict:

- **DIRECT** means reproduced from repository artifacts or fresh diagnostics against them.
- **RECONSTRUCTED** means computed from saved out-of-fold (OOF) features or ledgers, but not rerun through every upstream nesting level.
- **USER-SUPPLIED** means leaderboard state supplied in the task.
- **LITERATURE** means supported by the linked primary/model source.
- **INFERENCE** is the narrowest explanation consistent with the measurements; it is not a causal proof.

## 1. Executive Decision

V132 transferred positively: its submitted diagnostic candidate scored **0.66219**, an improvement of **+0.00046 over the protected V130 anchor** and +0.00049 over the immediately preceding V131 submission. The submitted bytes are recoverable and verified as [`candidate_v132_DIAGNOSTIC.csv`](../results/v132/v132/candidate_v132_DIAGNOSTIC.csv), SHA-256 **`d89714241c85dbaf93a5574550a5d9272c37b9b432c69fd5c5a2ff07fe690d84`**. Those bytes, rather than the notebook's fail-closed primary output, are now the protected base.

The most defensible explanation is not that V132 discovered a new semantic representation. V132 changed the **action policy**: a globally standardized Ridge utility score for ranks 2–4 was corrected by a shrunken pair-by-rank residual estimate. Relative to V131 it materially reordered opportunities, concentrated three of five test actions in a pair with substantial rank-2 training support, reduced exact-pair novelty from 0.80 to 0.20, and produced 15 benefits against 5 harms in OOF evaluation. This is evidence that support-aware residual routing improved the selected tail. It does **not** reveal whether any particular test user benefited.

The immediate research decision is:

> **BUILD, but do not yet submit, V133 — Exact-V132 Support-Gated Three-Expert Utility Consensus.**

V133 should retain exact V132 rankings and admit at most three Top-1 swaps. It should average three independently standardized, fully nested Ridge expected-utility heads—sparse text, BM25, and state/readiness—then require a positive mean, agreement from at least two heads, and pair-by-rank training support of at least 10. The architecture is selected for falsification because an exploratory reconstruction yielded +0.0002228 at a 12-action OOF prefix; the support-gated form retained +0.0001694 and 4/5 positive folds. These figures are **RECONSTRUCTED**, not promotion-grade, because the three branches have not yet been regenerated with full third-level nesting.

Four paths are closed for the immediate cycle: extending the V132 head farther down its ranking, adding another hierarchical residual layer, adding another pooled embedding, and fitting a learned/nonlinear stack over the same action labels. A hard pair cap is also rejected: it removed supported signal rather than controlling a demonstrated failure. The 0.96609 leader remains an unexplained outlier; visible organizer-only structure gives no credible route to reproduce it. With one slot remaining today, the correct action is to hold the slot.

## 2. Evidence Update: V128 → V132

The chain below separates the model evaluated locally from the bytes actually promoted. Local deltas are incremental NDCG@5 against each version's declared fold-local anchor.

| Version | Selected local evidence | OOF / test actions | Public score | Post-hoc reading |
|---|---:|---:|---:|---|
| V128 | +0.0002163; uncertainty and lock/world/geometry/final gates failed | candidate not promoted | 0.66028 protected primary | Correct fail-close; small mean alone was insufficient. |
| V129 | +0.0011302; 5/5 positive folds | 200 / 50 | 0.66075 | Large, directionally stable local evidence transferred +0.00047. |
| V130 | +0.0003943; fold deltas +0.000249, +0.000755, +0.000574, -0.000037, +0.000430 | 80 / 20 | 0.66173 | A sparse rank-aware package transferred +0.00098. |
| V131 | +0.0001709; 3/5 positive folds | 20 / 5 | 0.66170 | Small gain plus concentrated/shifted test geometry did not transfer (-0.00003). |
| V132 | +0.0001931; fold deltas +0.000385, +0.000343, +0.000292, -0.000438, +0.000383 | 20 / 5 | **0.66219** | One bad fold did not prevent positive transfer: +0.00046 over protected V130. |

Public scores and ranks are **USER-SUPPLIED**. Artifact hashes and local measurements are **DIRECT**.

The exact lineage matters:

| Artifact | SHA-256 | Meaning |
|---|---|---|
| V130 candidate | `28dd2121c72e910be8a01b31cd43d022c7fdf080d8e628bd8f30abc3a24bc93e` | Protected anchor used by V131 and V132. |
| V131 candidate | `837a3441de888dc0cc0761028b9afb6028e8ddedd67bf9e3c3815b9e5118c1b4` | Five-action V131 candidate. |
| V131 primary | `28dd2121c72e910be8a01b31cd43d022c7fdf080d8e628bd8f30abc3a24bc93e` | Fail-close remained exact V130. |
| V132 diagnostic candidate | `d89714241c85dbaf93a5574550a5d9272c37b9b432c69fd5c5a2ff07fe690d84` | Actual submitted V132 bytes and new protected base. |
| V132 notebook primary | `28dd2121c72e910be8a01b31cd43d022c7fdf080d8e628bd8f30abc3a24bc93e` | Formal gate stayed borderline and failed closed to V130. |

The V132 notebook therefore behaved correctly as a conservative research pipeline even though the user separately chose to submit its diagnostic candidate. Future notebooks must not confuse those two artifacts: all V133 comparisons and fail-close behavior must anchor to exact V132 diagnostic bytes.

Two older diagnostic facts remain useful. First, a reconstructed V130 rank-2-only file (SHA-256 `c1ffd6ba10c68c9da43d4125fbbc021a7986a81f49f1693358d47cce3a78a557`) scored 0.66098. Its 16 rank-2 actions accounted for only about +0.00023 over V129; the four lower-rank actions accounted for roughly +0.00075 of the full V130 package gap. This establishes that lower-rank authority once mattered, but it does not imply that it remains available after V132. Second, V132 deployed five rank-2 actions only. Its success is therefore evidence for better opportunity selection, not evidence that ranks 3–4 were required in that public package.

## 3. Why V132 Transferred Positively

### What actually changed

V131 and V132 shared 8,715 eligible user-action rows, yet their action scores had Pearson/Spearman correlation of only **0.873/0.873**. Only 4 of 20 OOF-selected actions and 1 of 5 test users overlapped. V132 was not a threshold adjustment over the same ordering; its pair-by-rank residual correction created a meaningfully different selected tail.

V132's 20 OOF actions contained **15 benefits, 5 harms, and no ties**. The selected benefits had mean realized utility +0.08218 and the harms -0.09203, so this remained a high-variance mixture. Within this tiny tail, predicted score did not separate benefit from harm: mean prediction was 0.04166 for benefits and 0.04287 for harms. Champion margin was larger for benefits (0.03498 versus 0.02025), but there is no exact fully nested no-margin causal ablation. It would therefore be incorrect to attribute the public gain solely to champion margin.

### Support and geometry

The label-free test package had five rank-2 swaps and only three distinct module pairs:

| Test action pair | Count | Relevant real-train rank-2 evidence | Interpretation |
|---|---:|---|---|
| M017 → M009 | 3 | n=36; 20 benefits / 16 harms; all-row mean Δ=-0.00069; among rows with positive final V132 score, benefit rate=0.609 | Strongest support for the residual correction; the global Ridge mean was negative, while the pair correction lifted a selected positive slice. |
| M006 → M009 | 1 | n=37; 15 benefits / 21 harms / 1 tie; mean Δ=-0.00555; positive-score benefit rate=0.400 | Mixed support; not independently persuasive. |
| M014 → M015 | 1 | n=90; 25 benefits / 63 harms / 2 ties; mean Δ=-0.04276; the selected positive slice is tiny | Weakest pair-level justification; selection depends on conditional features, not the pair mean. |

The training pair-by-rank support counts attached to the five test actions were 47, 59, 103, 47, and 47, with adjustments +0.01014, +0.00880, -0.00514, +0.01014, and +0.01014. Exact-pair unseen rate fell from 0.80 in V131 to **0.20 in V132**. V132 still had pair total-variation distance 0.80 and predicted-utility SMD -1.099, so it was not a clean in-distribution package; it was simply much better supported on the dimension its correction explicitly used.

The lone negative V132 fold was structurally the least like the test package among selected OOF folds. Standardized core-feature centroid distance from the five test actions was 0.658, 0.489, 0.758, **0.817**, and 0.608 for folds 0–4, respectively. Its selected pairs also did not match the test-selected pairs, and it contained two large harms. This supports—but cannot prove—the inference that the bad fold was diluted in public evaluation because it represented a less relevant action regime.

### Comparison with V130 and V131

| Package | OOF/test actions | Distinct test pairs | Max test-pair share | Exact-pair unseen rate | Predicted-utility SMD | OOF benefit/harm/tie |
|---|---:|---:|---:|---:|---:|---:|
| V130 | 80 / 20 | 15 | 0.20 | 0.75 | about -0.418 | 49 / 28 / 3 |
| V131 | 20 / 5 | 2 | 0.80 | 0.80 | about -1.374 | 13 / 5 / 2 |
| V132 | 20 / 5 | 3 | 0.60 | 0.20 | about -1.067 | 15 / 5 / 0 |

V131 combined several warnings: only 3/5 positive folds, extreme pair concentration, 80% unseen exact pairs, and larger utility shift. V132 retained concentration but replaced novelty with training-supported pair/rank evidence and improved fold direction. **INFERENCE:** the transferable unit was a better-supported selected action slice, not a universally better regression model.

No statement above infers the correctness, relevance, or hidden NDCG contribution of an individual V132 test row. A positive package score identifies only aggregate transfer.

## 4. Validation Calibration After V132

The four observed public moves show that local magnitude is poorly calibrated as a public point forecast:

| Transition | Local selected ΔNDCG@5 | Positive folds | Public movement |
|---|---:|---:|---:|
| V128 → V129 | +0.0011302 | 5/5 | +0.00047 |
| V129 → V130 | +0.0003943 | 4/5 | +0.00098 |
| V130 → V131 | +0.0001709 | 3/5 | -0.00003 |
| protected V130 → V132 | +0.0001931 | 4/5 | +0.00046 |

Fold direction and package construction have tracked public sign better than mean magnitude. With only five outer clusters, user and near-duplicate-cluster bootstrap intervals are appropriately wide. V132's intervals crossed zero—user CI [-0.000025, +0.000424], cluster CI [-0.000057, +0.000450]—yet the package transferred. A zero-crossing interval and one bad fold should remain reported uncertainty, not automatic vetoes.

A calibrated general promotion gate should now require:

1. Correct fold-local reconstruction, immutable anchors, unique users, score/order validity, and exact hash checks.
2. Incremental OOF mean of at least **+0.00015** against the actual protected base.
3. At least **4/5 positive outer folds**.
4. At least 60% beneficial selected actions, or a clearly larger benefit count than harm count, with no forced zero/negative actions.
5. A positive intended prefix and positive nearby-prefix sensitivity.
6. Conditional geometry review rather than a universal exact-pair novelty cutoff. A concentrated package must have direct pair-by-rank support; a diverse package may tolerate unseen exact pairs if module and feature support are stable.
7. Explicit test-to-OOF shift reporting, with core-feature and predicted-utility SMDs preferably no larger than 1.5.

This rule retrospectively accepts V130 and V132 and rejects V131. It also avoids learning the wrong lesson from V132: “ignore the bad fold” is not valid. The valid lesson is “do not reduce a multivariate uncertainty decision to one bad fold or one interval.”

Public feedback must remain a terminal validation signal. V132's score may calibrate the policy family, but it must not be used to relabel its five test users, whitelist its three pairs, or tune a V133 threshold.

## 5. Pseudo-GT v4 Diagnostic Findings

Pseudo-GT v4 is internally consistent with the submitted V132 package and useful only as a provenance/structure ledger.

**DIRECT findings:**

- It contains 1,000 test users. Exactly five rank-1 rows and five corresponding rank-2 rows changed from v3; ranks 3–6 did not change.
- Evidence sources are 995 `V130_BASE_RETAINED` and 5 `V132_5_ACTION_PACKAGE_PUBLIC_POSITIVE`; tiers are 995 `BASELINE_V130` and 5 `HIGH_PACKAGE_EVIDENCE`.
- The five changed orderings exactly match the V132 candidate file. All rows retain K=5.
- The hard K=5 representation has five nonzero relevance positions. The posterior representation has six nonzero values per row with the fixed ladder 0.0846975, 0.2637375, 0.5490725, 0.7000150, 0.8491975, and 1.0.
- Its v3 lineage was itself 980 `EXACT_V129_BASE` rows plus 20 rows changed by the V130 public-positive package.

The `V130_BASE_RETAINED` label compresses older v3 lineage for 995 rows, but it does not contradict the rank ledger. A v5 rewrite is not warranted. The public score supports the five-action package in aggregate only; it does not make the promoted Top-1 item correct for each of those users. The fixed relevance ladder is synthetic, not an observed target.

Consequently, pseudo-GT v4 was excluded from every residual, representation, and architecture probe in this report. It may be used to verify candidate ordering, K, lineage, and package overlap. It must not be used as a training label, validation target, pair prior, per-user correctness claim, or tuning source for V133.

## 6. Targeted Fresh EDA After V132

### Organizer-only structure

The available data contain 4,000 train users, 1,000 test users, 17 modules, 14,573 chat rows, and 251 invalid assessment cells. Target K is 4 for 1,362 train users, 5 for 1,285, and 6 for 1,353. Rank-score geometry is nearly deterministic: rank 1 is exactly 1.0, while ranks 2–4 average about 0.8492, 0.7000, and 0.5491 with only 11 jitter values. That makes the loss impact of a swap easy to compute; it does not make the module order easy to predict.

There are 3,942 distinct Top-5 sequences among 4,000 users. The most common sequence covers only 0.075% of users, and only 115 users belong to any repeated Top-5 sequence. Exact memorization is therefore structurally weak.

| Organizer-only grouping | Test groups seen in train | Test groups with support ≥2 | Repeated-group Top-1 purity | Leave-one-out NDCG@5, all / repeated |
|---|---:|---:|---:|---:|
| Full profile skeleton | 28.3% | 22.5% | 0.495 | 0.433 / 0.518 |
| Exact latest state | 55.6% | 35.9% | 0.505 | 0.442 / 0.469 |
| Exact assessment state | 38.6% | 29.9% | 0.383 | 0.416 / 0.447 |
| Latest + assessment | 1.8% seen | negligible | not useful | 0.399 overall |
| Character 3–5-gram component at 0.90 | 60.6% | 53.8% | 0.333 | 0.459 / 0.489 |

User-ID modulo probes also stayed near the global baseline: mean NDCG@5 was 0.38055 globally, 0.38349 for modulo 17, 0.37522 for modulo 50, and 0.37187 for modulo 100. These checks weaken deterministic-template, exact-persona, and ID-encoding explanations for the leader outlier.

### Residual opportunity and support

The V132 hierarchy spans 523 observed pair-by-rank cells. Support is highly uneven: min 1, 10th percentile 2, 25th percentile 7, median 17, 75th percentile 32, 90th percentile 53.8, and max 103. Residual standard deviation remains roughly 0.098–0.114 across support bins, while median absolute cell mean falls from 0.0356 for support 1–9 to 0.0071 for support 80+. This is the expected pattern for noisy cell estimates and supports shrinkage or a minimum-support gate.

The next five label-free V132 test actions would all be rank-2 swaps across five distinct pairs. Their mean predicted utility is 0.0180 versus 0.0295 for the submitted five, mean pair correction is only 0.00255, and one action has support 7 (supports: 59, 46, 39, 7, 73). Even before labels, the mechanical next tranche is weaker and less uniformly supported.

These EDA results point toward a policy-level improvement: demand agreement across existing evidence families and remove low-support opportunities. They do not point toward a new user key, a pair whitelist, or a broad representation rebuild.

## 7. Residual Headroom and Action-Utility Audit

### The V132 head is exhausted

The saved V132 selected mask was reproduced exactly. Its prefix curve is:

| V132-ranked prefix | Incremental NDCG@5 | Benefit / harm / tie | Reading |
|---:|---:|---:|---|
| 12 | +0.0001982 | 9 / 3 / 0 | Strongest compact region. |
| 20 | +0.0001931 | 15 / 5 / 0 | Submitted-policy validation prefix. |
| 40 | +0.0002112 | 26 / 14 / 0 | Positive in aggregate, but almost all gain was already captured. |
| 60 | +0.0002217 | — | Only +0.0000105 beyond 40. |
| 80 | +0.0001732 | — | Actions 61–80 lose about 0.0000485. |

The marginal actions 13–20 contribute **-0.0000050** despite six benefits and two harms; harm magnitude cancels the count advantage. Starting from exact V132, the next 20 same-head actions add only **+0.0000181** with 11 benefits and 9 harms. A Ridge-only next 20 lose **-0.0001485** with 8 benefits and 12 harms. There is no submission-sized same-head tranche left.

Alternative temporal windows did not expose a robust hidden reserve. At 20 actions, exact-history, latest, near-history, and contiguous variants gave +0.0001311, +0.0002767, +0.0002304, and +0.0001627; only the contiguous variant was 5/5 positive. Those are model-selection alternatives evaluated on the same small action tail, not additive gains after V132. Selecting among them now would be post-selection threshold tuning.

### What the hierarchy did—and did not do

Across all eligible actions, pair-by-rank correction slightly worsened global error: MAE/MSE were 0.09056/0.012636 versus 0.09038/0.012550 for Ridge. At the selected top 20, however, Ridge produced +0.0000445 with 12/8 benefits/harms, whereas the hierarchy produced +0.0001931 with 15/5 and only 11 shared users. The hierarchy is a **tail reranker**, not a global calibration improvement.

Its conclusion is stable to reasonable fixed shrinkage: shrink 10, 20, and 40 yield +0.0001858, +0.0001931, and +0.0001770 at 20 actions, with 19/20 and 17/20 overlap against shrink 20. There is no justification to tune shrinkage further.

At rank 2, the hierarchical score gives +0.0001931 at 20 actions and AUC 0.651. Rank 3 has exploratory positive evidence (+0.0000828, 11/9, AUC 0.670), while rank 4 has only two positive-score actions and -0.0000333. This is too thin to restore broad lower-rank authority.

Fixed residual alternatives evaluated on saved OOF residuals all fail **after exact V132**: pair-by-rank -0.000118, pair -0.000269, candidate-by-rank -0.000231, incumbent-by-rank -0.000129, candidate -0.000243, incumbent -0.000053, additive incumbent/candidate/rank -0.000120, and rank-2 pair SVD -0.000156. These are **EXPLORATORY** because they reuse saved residual geometry, but their uniformly negative direction is sufficient to close another hierarchical or latent-pair layer for V133.

Score quintiles also show poor bulk calibration: the highest max-action quintile predicted +0.002924 per user but realized -0.008549, with benefit rate 0.435. Positive action scores are useful only as a relative extreme-tail ranking. They must not be interpreted as calibrated probabilities or used to force coverage.

## 8. Representation / Embedding Audit Through V132

Cross-fold action-error correlations show near-redundancy among the current representation groups:

| Error pair | Correlation |
|---|---:|
| E5 / Nomic | 0.9985 |
| word / character TF-IDF | 0.9986 |
| sparse / dense | 0.9963 |
| sparse / BM25 | 0.9965 |
| dense / BM25 | 0.9959 |

This does not mean embeddings have no value in the base recommender. It means their **remaining V132 action errors are almost identical**, so a third pooled embedding is unlikely to create the diversity needed for another safe swap package.

Fixed-head exploratory next-20 probes after V132 support that reading:

| Head | Incremental NDCG@5 | Benefit / harm / tie |
|---|---:|---:|
| word TF-IDF | +0.0001753 | 12 / 6 / 2 |
| sparse word+character | +0.0001515 | 12 / 7 / 1 |
| full BM25 | +0.0001513 | 11 / 7 / 2 |
| state/readiness | +0.0001623 | 12 / 7 / 1 |
| E5 | +0.0000201 | 8 / 10 / 2 |
| Nomic | -0.0000011 | — |
| combined dense | -0.0000607 | — |

These are **RECONSTRUCTED** branch diagnostics, not fully nested candidates. Their purpose is decomposition: sparse, lexical-retrieval, and state heads retain different top-tail rankings, while the dense branches do not.

Accordingly, V133 should add no new pooled embedding. E5 and Nomic remain part of the protected V132 base and upstream representation; removing them would change the anchor and require a different experiment. They simply should not become new action-policy experts.

The only representation fallback worth a bounded falsifier is a **candidate-only cross-encoder**, because token-level candidate-conditioned interaction is mechanistically different from pooled similarity. `BAAI/bge-reranker-v2-m3` is a multilingual XLM-R sequence classifier with an Apache-2.0 model card and fp16 usage; the checked Hub revision is `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` ([model card](https://huggingface.co/BAAI/bge-reranker-v2-m3)). BGE-M3 and late-interaction retrieval provide the broader representation context ([BGE-M3 paper](https://arxiv.org/abs/2402.03216), [ColBERT paper](https://arxiv.org/abs/2004.12832)).

There is a strong negative prior: V34 already used this reranker family in a candidate-only Head 12, obtained cross-encoder benefit AUC around 0.566–0.580, and approved zero actions. V20's broader BGE-M3 dense/sparse/ColBERT blend and V84's MiniLM fine-tune also failed their respective gates. Those experiments do not exactly test an exact-V132-anchored residual formulation, so they do not logically close it; they do make it a fallback rather than the immediate V133 architecture.

## 9. Reassessment of “Embedding + TF-IDF + BM25 + LGBM”

The phrase describes useful ingredients but not a sufficiently specific next architecture. Through V132, most of it has already been exercised:

| Ingredient | Status through V132 | Remaining implication |
|---|---|---|
| Pooled embeddings | Present upstream; E5/Nomic residual errors correlate 0.9985 | Keep the protected base; do not add another pooled encoder. |
| TF-IDF | Word/character heads retain some top-tail utility | Preserve as an independent sparse expert. |
| BM25 | Full BM25 helped V131/V132 more than temporal/no-BM25 ablations | Preserve as an independent retrieval expert. |
| Tabular/state features | Readiness/state head retains top-tail utility | Preserve as a third expert rather than burying it in one concatenation. |
| LGBM/nonlinear stack | V132 Stage-1 append gave only +0.0000801; learned recombination has not shown additive post-V132 gain | Do not make it the immediate action selector. |

The limiting problem is no longer “can one model consume all feature families?” It is “can the selected few actions survive representation disagreement, support noise, and magnitude-heavy harms?” A single LGBM over concatenated features can fit interactions, but at this sample size it also gets one shared opportunity to exploit the same noisy action labels. It provides less diagnostic separation than three frozen experts and introduces another tuning surface.

The literal stack is therefore **rejected as V133 primary**. Its strongest components are retained in the smaller consensus architecture. A nonlinear stack should be reconsidered only if a fully nested branch audit establishes complementary residuals and enough action-level samples to validate complexity; neither condition holds now.

## 10. Assessment of the 0.96609 Outlier

The leader is **0.30390 NDCG** above V132, far outside the scale of every validated incremental change in this repository. No visible-data probe provides a credible bridge to that result:

- Top-5 sequences are almost unique (3,942/4,000), defeating simple target-sequence memorization.
- Exact profile, latest-state, and assessment groupings have limited repeated support and leave-one-out NDCG far below 0.96609.
- Near-duplicate character components improve coverage but have repeated-group Top-1 purity only 0.333 and leave-one-out NDCG 0.459 overall.
- User-ID modulo structure is indistinguishable from weak baseline variation.
- Deterministic rank-score magnitudes simplify NDCG arithmetic but contain no module-order label.

Possible explanations include unavailable information, a materially different problem formulation, a scoring/entry special case, or chance associated with an unknown evaluation regime. None can be distinguished from the organizer-visible files. There is no evidence here for target leakage, misconduct, or a recoverable deterministic mapping, and the report makes no such allegation.

The operational conclusion is to treat 0.96609 as **non-actionable**. Chasing it would encourage leakage-prone key searches, pseudo-label hardening, or architecture resets unsupported by validation. The rational objective is to defend and improve the reproducible 0.66219 line, not force a story for an outlier.

## 11. Candidate V133+ Architectures

Five serious directions were tested or bounded against exact V132:

| Candidate | Fresh evidence | Decision |
|---|---|---|
| A. Extend V132's ranked action list | Next 20 add only +0.0000181; later tranches flatten then turn negative | **Close.** No useful same-head reserve. |
| B. Deeper hierarchical pooling or latent pair factorization | Eight residual structures are negative after V132; global error also worsens under the current correction | **Close for V133.** More pair machinery is unsupported. |
| C. Support-gated sparse/BM25/state utility consensus | Reconstructed raw prefix-12 +0.0002228, 9/2/1 and 5/5 folds; support-gated +0.0001694, 8/4 and 4/5 | **Select for fully nested V133 build.** |
| D. Learned Ridge/LGBM stack over existing action heads | Reconstructed Ridge stack +0.0000295 at 12 and only 3/5 folds; V132 Stage-1 append +0.0000801 | **Reject as immediate primary.** Complexity lacks gain. |
| E. Exact-V132 candidate-only cross-encoder residual | Mechanistically novel, but no aligned result; V34 same-family candidate head had weak AUC and zero approvals | **Conditional fallback only.** Run only a cheap falsifier. |

Candidate C has a narrow but coherent signal. The equal raw mean of sparse, full-BM25, and state/readiness heads produced:

| OOF prefix | Incremental NDCG@5 | Benefit / harm / tie | Positive folds |
|---:|---:|---:|---:|
| 12 | +0.0002228 | 9 / 2 / 1 | 5/5 |
| 20 | +0.0001863 | 11 / 7 / 2 | 5/5 |
| 40 | +0.0000668 | 17 / 19 / 4 | 4/5 |

The sharp decay argues for 0.3% coverage, not a larger package. Adding dense heads reduced prefix-12 gain to +0.0001275. A 50/50 blend with the V132 score produced +0.0000885 and 3/5 folds; a learned Ridge stack produced +0.0000295 and 3/5. Equal expert averaging is both simpler and empirically stronger.

The raw prefix selected seven distinct pairs with a maximum pair count of five. That repeated pair, M015 → M001, had 15 eligible examples, 9 benefits/6 harms, mean realized utility +0.03743, and selected examples with 4 benefits/1 harm. A hard cap of two reduced total gain to +0.0001018; combining the cap with support ≥10 reduced it to +0.0000459 and 2/5 folds. Pair concentration is not itself the defect. Low-support extrapolation is the defensible target: requiring pair-by-rank support ≥10 retained +0.0001694, 8/4 actions, 4/5 positive folds, six distinct pairs, minimum support 13, and no forced fill.

All Candidate C numbers are architecture-screening reconstructions. The selected architecture earns a build, not a submission, because saved OOF branches do not reproduce the full upstream V132 analogue separately inside every outer fold.

## 12. Selected Immediate V133 Architecture

### Name and objective

**V133 — Exact-V132 Support-Gated Three-Expert Utility Consensus** asks one narrow question: after preserving exact V132, can independent existing feature families agree on a tiny set of additional Top-1 swaps whose expected NDCG@5 gain is positive and whose pair-by-rank residual regime is adequately supported?

### Immutable anchor and labels

- Submission anchor: exact [`candidate_v132_DIAGNOSTIC.csv`](../results/v132/v132/candidate_v132_DIAGNOSTIC.csv), SHA-256 `d89714241c85dbaf93a5574550a5d9272c37b9b432c69fd5c5a2ff07fe690d84`.
- Training target: exact realized ΔNDCG@5 of replacing the V132 Top-1 item with one candidate at current ranks 2–4.
- One action at most per user; no reorder below Top-1; K and all score values remain unchanged.
- Pseudo-GT v4 is forbidden as a target, feature, prior, or selection input.

### Fully nested construction

For each existing outer fold, reconstruct the V132-aligned base and candidate geometry without using that fold's labels. Inside the outer-training partition, regenerate the required V129/V130/V132 analogue through the fixed inner folds, then generate action labels and all preprocessing statistics for the three branch heads. Fit each branch only on outer-training actions and score only the held-out outer fold. Any target encoding, pair support, vocabulary, standardization, residual statistic, or calibration value must be fold-local.

This nesting is mandatory. Reusing the saved full-OOF V132 scores to train a new selector would let validation labels influence its anchor and would overstate the incremental result.

### Three fixed experts

Each expert is a standardized Ridge regressor with **alpha=10**, inherited tokenization/feature definitions, and no hyperparameter search:

1. **Sparse expert:** full/latest/recency word and character TF-IDF action differences, module identities, action rank, and V132 champion margin.
2. **Retrieval expert:** full unigram/bigram BM25 action differences, module identities, action rank, and champion margin.
3. **State expert:** assessment, readiness/latest-state, validity and chat-state flags, module identities, action rank, and champion margin.

E5 and Nomic remain upstream in the V132 anchor but are excluded from these new experts. LGBM, pair factorization, cross-encoder scores, and learned stack weights are excluded from V133.

For each eligible action, compute the arithmetic mean of the three raw expected-utility predictions. Admit the action only when:

- the mean is strictly positive;
- at least two of three expert predictions are strictly positive; and
- its outer-training pair-by-rank support is at least 10.

Rank admitted actions by the mean. Select a deterministic 12-action OOF policy (0.3% of 4,000), allocating fold quotas by largest remainder with fold index as the fixed tie-breaker. For test, select **at most three** positive admitted actions (0.3% of 1,000); do not force fill. There is no pair whitelist, pair bonus, public-pair feature, pair cap, score threshold search, or prefix search.

### Fixed engineering contract

- Reuse V132 folds, seeds, parsers, cached upstream features, BM25 settings, and score-writing routines.
- Assert exact fold disjointness, one OOF prediction per user/action, one selected action per user, unchanged K, unchanged score multiset per user, valid module IDs, and exact V132 fail-close bytes.
- Record per-expert raw score, consensus vote count, pair-by-rank support, rank, incumbent/candidate modules, fold, realized utility (OOF only), and final selection reason.
- Emit OOF diagnostics for prefixes 12, 20, and 40, but keep 12 fixed as primary. Prefixes 20/40 are sensitivity checks, not alternate candidates.
- Emit per-fold and clustered uncertainty, representation/world slices, test-to-OOF geometry, V131/V132 overlap, pair concentration, and exact candidate/fail-close hashes.
- If no test action passes, the candidate must be byte-identical to V132.

The expected runtime is approximately the V132 nested run plus three cheap CPU Ridge fits per split; cached sparse/BM25/state features should keep it within roughly **2–2.5 hours on the existing T4×2 environment**, with no new model download. Runtime is an estimate, not a gate.

## 13. Validation and Submission Decision Policy

V133 receives a submission recommendation only after a fresh, fully nested run. The fixed primary policy is the 12-action OOF / at-most-3-action test configuration above.

### Hard correctness gates

All must pass:

- exact V132 anchor hash before mutation and exact fail-close hash after generation;
- outer/inner group disjointness and fold-local fitting of vocabularies, scalers, supports, and heads;
- no pseudo-GT, test labels, public pair whitelist, or leaderboard-derived feature;
- unique user actions, legal modules, unchanged K and per-user score multiset;
- deterministic rerun hashes and complete action/provenance ledgers.

### Promotion gates

All must pass for **SUBMIT**:

1. Primary nested incremental OOF NDCG@5 ≥ **+0.00015** against the fold-local V132 analogue.
2. At least **4/5** outer folds positive.
3. Among 12 primary actions, at least **8 benefits** and at most **4 harms**; ties are reported and not counted as benefits.
4. Prefix 20 remains positive. Decay by prefix 40 is acceptable and should block expansion, not the 12-action candidate.
5. Every selected test action has positive consensus, at least 2/3 positive experts, and pair-by-rank support ≥10.
6. Test selection contains 1–3 actions; zero actions means exact V132 fail-close.
7. Core-feature and predicted-utility test-versus-selected-OOF absolute SMD are each ≤1.5.
8. A three-action test package does not overlap V131 on all three users; complete 3/3 overlap would indicate regression toward a publicly non-improving policy.
9. No unexamined world/representation slice contains a systematic contradiction large enough to account for the total gain.

User and near-duplicate-cluster confidence intervals, worst-fold delta, pair total variation, and exact-pair novelty remain mandatory diagnostics but not independent vetoes. Their interpretation must accompany the decision.

Decision labels are fixed:

- **SUBMIT:** every correctness and promotion gate passes.
- **BORDERLINE / HOLD:** correctness passes and mean gain is positive, but any promotion gate fails or test geometry is materially ambiguous. Keep exact V132.
- **DO NOT SUBMIT:** any correctness failure, nonpositive primary gain, fewer than 3/5 positive folds, forced nonpositive actions, unsupported test action, or irreproducible bytes. Keep exact V132.

No threshold, support floor, head weight, coverage, or prefix may be changed after seeing these OOF results. A changed policy is a future version with a new predeclared test, not a rescue of V133.

## 14. Expected Gain, Failure Modes, and Public Scenarios

The selected architecture has reconstructed evidence of +0.0001694 under its support gate, but a nested result may shrink materially. The appropriate expectation is therefore conditional, not a point promise:

| Public scenario if all V133 gates pass | Plausible movement from 0.66219 | Meaning |
|---|---:|---|
| Conservative | -0.00015 to 0 | Public subset or harm magnitude defeats the tiny package. |
| Central | 0 to +0.00020 | Consensus removes weak actions but adds only marginal new signal. |
| Upside | +0.00020 to +0.00050 | Independent experts identify a supported tail that transfers like V132. |

These ranges are judgmental scenarios, not calibrated intervals. If the gates do not pass, the expected action is no submission and therefore zero change from protected V132.

Primary failure modes and their containment are:

| Failure mode | Observable symptom | Required response |
|---|---|---|
| Hidden reuse of validation labels through the V132 anchor | Nested score collapses relative to reconstruction; lineage audit fails | Reject run, rebuild fold-local anchor. |
| Experts are correlated despite different feature blocks | Same users/scores across heads; 2/3 vote adds no selectivity | Hold; do not replace equal weights with learned weights. |
| Low-support residual optimism | Test actions sit in sparse pair/rank cells | Support ≥10 gate; no forced fill. |
| One supported pair dominates | High concentration but stable evidence | Report it; do not cap unless prespecified evidence shows concentration itself is harmful. |
| Harm magnitude overwhelms benefit count | Positive count with flat/negative NDCG | Enforce mean-gain and harm-count gates together. |
| Distribution shift | SMD >1.5 or severe unseen representation worlds | Hold exact V132. |
| Multiple-comparison optimism | Only one of several post-hoc prefixes looks positive | Primary prefix remains 12; 20/40 are sensitivity only. |
| Public overreaction | Desire to tune pairs after score | Archive result; no per-user inference or same-family feedback tuning. |

The tiny #2-to-#3 gap of 0.00009 means a weak exploratory submission can lose position. Conversely, the #1 gap cannot plausibly be closed by a three-action package. The decision policy appropriately prioritizes private/generalization robustness over leaderboard churn.

## 15. Final-Day Submission Strategy and Conditional Roadmap

There is one slot remaining today and approximately three tomorrow (**USER-SUPPLIED**). Slots are an option, not a quota.

### Today

**Hold the remaining slot.** No fully nested V133 artifact exists, and the V132 next tranche is already falsified. Do not submit a threshold variation, an unsupported pair cap, pseudo-GT v4 ranking, or another pooled-embedding blend.

### Tomorrow

1. Run the single frozen V133 architecture and its verification suite. If and only if it receives **SUBMIT**, use the first available slot for its at-most-three-action file.
2. If V133 is BORDERLINE or DO NOT SUBMIT, retain V132. Do not relax gates to spend a slot.
3. A candidate-only `bge-reranker-v2-m3` residual may consume research time only if a cheap fold-safe falsifier achieves action-residual correlation below 0.95 versus V132 and nested incremental gain of at least +0.00015. It must then pass the same promotion policy as a separately frozen version. The historical negative prior makes this unlikely; it is not an automatic second submission.
4. Reserve the remaining slot(s) for a genuinely mechanism-diverse, fully validated artifact. If none exists, leave them unused.

After any public result, record only package-level calibration: score, exact hash, action count, and predeclared hypothesis. Do not infer which users were right, update pair labels, or resubmit a nearby prefix based on the public movement.

The conditional roadmap is intentionally short:

| Observation | Next action |
|---|---|
| V133 passes every gate | Submit once; protect exact bytes; stop same-family tuning. |
| V133 has positive mean but 3/5 folds, unsupported test rows, or excessive shift | Hold V132; close consensus for the final day. |
| V133 nested signal disappears | Treat the reconstruction as leakage/selection optimism; hold V132. |
| Cross-encoder falsifier is decorrelated and clears +0.00015 | Freeze a separate candidate and run the full gate; otherwise stop. |
| No candidate clears its frozen gate | Finish with V132 at 0.66219. |

## 16. Final Recommendation

Protect exact V132 diagnostic bytes at SHA-256 **`d89714241c85dbaf93a5574550a5d9272c37b9b432c69fd5c5a2ff07fe690d84`**. Its positive transfer is most consistent with support-aware action reranking and better test support, not a new embedding breakthrough or proof about individual test users.

The only immediate V133 build should be **Exact-V132 Support-Gated Three-Expert Utility Consensus**: three fixed Ridge heads, equal expected-utility averaging, 2/3 positive agreement, pair-by-rank support ≥10, 12 OOF actions, and at most three test swaps. Run it with complete nesting and fail closed to exact V132. It is currently **BUILD / NOT YET SUBMIT**.

Do not extend the V132 head, add another hierarchical layer, impose a pair cap, train on pseudo-GT v4, add another pooled embedding, or pursue the 0.96609 outlier. Hold today's slot. Tomorrow, submit only a fully verified candidate that clears every predeclared gate; otherwise keep V132 and leave unused slots unused.
