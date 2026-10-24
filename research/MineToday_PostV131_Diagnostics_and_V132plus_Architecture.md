# MineToday — Post-V131 Public Diagnostics, Targeted EDA, and V132+ Architecture

## 1. Executive Decision

**FACT — protected state.** The competition baseline remains the exact public-scoring V130 candidate at **0.66173**, SHA-256 `28dd2121c72e910be8a01b31cd43d022c7fdf080d8e628bd8f30abc3a24bc93e`. The user-reported standing is approximately **#3**, with #2 at `0.66193`; the immediate target is therefore about **+0.00020**. V131 scored `0.66170` and does not replace V130. Best-score retention makes exploration affordable, but it does not justify mutating the proven 70 V130 Top1 edits.

Two public diagnostics materially change the scientific state. First, preserving only V130's 16 rank-2 promotions scored `0.66098`, only `+0.00023` over V129, while the four removed rank-3/4 promotions account for the remaining approximately `+0.00075` package difference. Second, V131 added five BM25-enhanced rank-2 swaps, was locally positive by `+0.0001709`, yet was public-neutral/slightly negative by about `−0.00003`. The valid joint conclusion is not “prefer lower ranks.” It is that candidate rank is context rather than authority, and that another average-value rank-2 retrieval layer is unlikely to find the few remaining valuable errors.

**Decision.** Build V132 next as an **exact-V130-anchored, ranks-2–4, hierarchical expected-utility micro-extension**. Retain the global Ridge objective, add a fold-safe incumbent×challenger×rank residual correction shrunk toward zero, restore the champion/policy margin that V131 omitted, use full-history BM25 only as one feature family, and let ranks 2–4 compete with no rank quota. Freeze all V130 rows and cap deployment at **at most five** new Top1 swaps. The operational fallback is exact V130, not another submission.

This is selected for implementation, not pre-approved for submission. A target-blind cross-fitted surrogate gained `+0.0002993` at the 20-action OOF equivalent, was positive in 5/5 folds, and produced 15 benefit / 4 harm / 1 tie, but it selected only one rank-3 action and its provisional five-action test geometry still showed concentration and dense-feature shift. V132 must reproduce the mechanism with full nesting and pass the decision policy in Section 12.

## 2. Causal Evidence Update: V128 → V131

The concise causal chain is:

| Version | Direct local evidence | Public/package evidence | Current lesson |
|---|---:|---:|---|
| V128 | selected no-BM25 branch `+0.0002163`; CI crossed zero; final gates failed | no public V128 test; primary was exact `0.66028` champion | Broad hybrid features had signal, but branch/purity gates and imperfect second-level nesting selected the wrong tiny region. |
| V129 | fixed 5% policy `+0.0011302`, 5/5 random folds; seen/unseen both positive | `0.66075`, about `+0.00047` | Lean TF-IDF + E5/Nomic + assessments + per-module LightGBM + fully nested utility transferred positively despite deployment-anchor mismatch. |
| V130 | V129-aligned 2% extension `+0.0003943`, 4/5; 20 actions | `0.66173`, about `+0.00098` over V129 | Exact champion/policy alignment and bounded Top1 authority mattered more than a large offline mean. |
| V130 rank2-only | same 16 rank-2 actions, four lower-rank actions reverted | `0.66098` | Sparse selected OOF support is not a valid lower-rank veto; rare lower-rank corrections can dominate a package. |
| V131 | rank-2-only 0.5% extension `+0.0001709`, 3/5; CI crossed zero | `0.66170`, about `−0.00003` versus V130 | BM25 is complementary offline, but the specific pooled rank-2 micro-package did not improve the champion. |

These values come from the direct [V128 gate report](../results/v128/v128/v128_gate_report.json), [V129 decision report](../results/v129/v129/v129_decision_report.json), [V130 decision report](../results/v130/v130/v130_decision_report.json), and [V131 decision report](../results/v131/v131/v131_decision_report.json), with public scores treated as user-supplied package facts.

**Strengthened:** exact action utility, immutable anchoring, sparse+dense representation, and conservative Top1 score swaps remain the successful lineage. **Weakened:** offline mean and grouped-world agreement are not reliable magnitude predictors; V131's near-world result was `+0.000256`, 5/5, yet public transfer was flat. **Overturned:** rank 3/4 cannot be rejected merely because V130 selected no rank-4 OOF actions. **Still unresolved:** which visible feature mechanism distinguishes a genuinely beneficial lower-rank correction from the much larger harmful lower-rank population. No public result identifies any individual test row.

## 3. What the V130 Rank2-Only Diagnostic Actually Proves

**FACT — exact byte reconstruction.** The rank2-only CSV is no longer present, but it was reconstructed in memory without writing a submission: start with exact V130 bytes and replace precisely the four V130 lower-rank user lines with their exact V129 lines. The resulting SHA-256 is `c1ffd6ba10c68c9da43d4125fbbc021a7986a81f49f1693358d47cce3a78a557`, exactly the previously recorded hash. It differs from full V130 in four raw data lines and from V129 in sixteen. Relative to V129 it makes 16 Top1 changes, zero Top5-membership changes, zero rank-5 changes, and zero score-multiset changes. Full V130 is independently verified at `28dd…93e`; V129 is `34f334429c2113ba5b96283a7974fad472068ea08047ba3275fc44057e3a95df`.

The public comparison is therefore a clean action-family ablation at package level:

- V129 → V130 rank2-only: `0.66075 → 0.66098`, approximately `+0.00023` for the 16 rank-2 package.
- V130 rank2-only → V130 full: `0.66098 → 0.66173`, approximately `+0.00075` for retaining one rank-3 and three rank-4 actions.
- V129 → V130 full: approximately `+0.00098`; about 76.5% of that rounded package uplift is associated with the four-action lower-rank family.

**SUPPORTED INFERENCE.** At least some useful mechanism exists in the lower-rank family, and its aggregate value can exceed that of a larger rank-2 package. Candidate depth also creates larger possible Top1 changes: on real OOF labels the maximum positive exact delta rises from `0.1782` at rank 2 to `0.2413` at rank 3 and `0.2755` at rank 4.

The ablation does **not** establish that all four rows helped, which row helped, or that lower ranks have positive average value. It cannot license a pair or user whitelist. Public values are rounded and may describe a leaderboard subset. The only valid use is to reopen a structural family, define label-free features, and validate those structures on organizer train labels. That is exactly how Section 6 uses it.

## 4. Why V131 Did Not Improve the Champion

**FACT — implementation.** The [V131 notebook](../notebooks/v131.ipynb) reconstructs clean nested V129 and V130 analogue authorities inside each outer fold, trains the new action head on ranks 2–4, and computes exact counterfactual ΔNDCG. Its primary 115-feature global Ridge includes candidate/incumbent module identity, rank, word/character TF-IDF full/latest/recency, E5, Nomic, five BM25 views, readiness, assessments, chat state, negation/comparison, contradiction, and explicit topic mentions. It deliberately omits pair one-hot, policy-anchor margin, and Stage-1 fusion. Deployment then reads only the rank-2 prediction column and selects four actions per random fold, five on test.

The local result was weak-positive, not decisive: `+0.0001709`, 13 benefit / 5 harm / 2 ties, 3/5 folds, user CI `[-0.0000206,+0.0003973]`, cluster CI `[-0.0000216,+0.0003577]`. Exact/latest/near/contiguous worlds were about `+0.000090/+0.000062/+0.000256/+0.000094`; the grouped worlds did not predict the public package.

BM25 itself was not empty. No-BM25 gained `+0.0000744`; full-history BM25 gained `+0.0001779`; temporal BM25 gained only `+0.0000463`; all registered streams gained `+0.0001709`. Adding V130 Stage-1 fusion to the same raw head fell to `+0.0000332`. **SUPPORTED INFERENCE:** full-history lexical saturation/length normalization supplied complementary offline information, but temporal BM25 and a simple fusion append did not. The correct update is “do not repeat this BM25 rank-2 selector,” not “BM25 can never help.”

The deployment geometry was visibly fragile. OOF selected predicted utility averaged `0.04717`, test averaged `0.03285` (SMD `−1.414`). Test-vs-OOF selected SMDs were `−0.344` for word-TFIDF difference, `+0.946` E5, `+0.519` Nomic, and `−0.672` full unigram BM25; pair TV was `0.90`, and 80% of test selected pairs were absent among OOF-selected pairs. Four of five test actions shared `M_001→M_015`, and three were no-chat cases. Allowing the saved V131 scores to choose ranks 2–4 does not repair this: at the same fold-local 20-action cap it yields `+0.0001482`, 4/5, with only two rank-3 choices; on test the same five rank-2 actions still win. Thus hard rank restriction was a mistake in principle but was not the sole cause of V131's result. Pooling, omitted geometry, test concentration, and modest expected value jointly explain the miss as far as artifacts permit. Individual correctness remains **NOT ESTABLISHED**.

## 5. Validation Calibration After V130/V131 Diagnostics

The three public-positive/diagnostic packages reject a one-number transfer rule. V129 had the strongest local mean and 5/5 folds but transferred less than half its local estimate. V130 had a smaller mean, a slightly negative fold, and CIs touching zero, yet transferred by `+0.00098`. V131 had a modest positive mean and strong near-group world but transferred approximately zero. Fitting a “public transfer model” to three adaptively developed packages would be pseudoprecision.

**SUPPORTED INFERENCE.** Random-world fold direction is useful for package sign—5/5, 4/5, and 3/5 track positive, strongly positive, and neutral in this tiny history—but it does not predict magnitude. Action-family geometry is at least as important. V130's rare lower-rank family was nearly absent from selected OOF support yet publicly consequential; V131's rank-2 family had ordinary train support but severe selected-policy shift. Grouped worlds remain stress tests, not gates that overrule the deployment-like random world.

Absence and negative evidence must be separated:

- V130 selecting zero rank-4 OOF actions is **absence under that selector**, not proof that rank 4 is harmful everywhere.
- Rank-4 opportunities have a negative population mean (`−0.1256` on residual-eligible users) and 80.4% harm rate: that is genuine negative base-rate evidence.
- The public four-action ablation is positive family evidence, while matched `M_001→{M_006,M_010}` rank-4 train support is favorable but only five eligible cases: useful, highly uncertain evidence.
- V130's four selected rank-3 OOF analogues averaged `−0.0642` (one benefit, three harms): negative evidence about that selected OOF slice, not about the distinct test pair structure.

Repeated reuse of 4,000 labels makes architecture selection overfit a first-class risk. Every new objective tried here was fixed before evaluation; no hyperparameter grid was used. V132 should register one primary, log diagnostic ablations without selecting among them after results, and evaluate **top-tail utility concentration**—benefit/harm counts, realized selected mean, rank/pair distribution, and fold direction—alongside average ΔNDCG. CIs remain descriptive. A CI barely crossing zero must not automatically block another V130-like rare-action candidate, but neither should a positive mean excuse V131-like test shift.

Calibration must also stay fold-local. Pooling the globally highest OOF scores can favor a fold whose utility scale is simply less shrunk; V131's executed four-per-fold selection is the relevant simulation of a full-data top-five deployment. The fresh comparisons therefore allocate each fixed cap deterministically across folds. Prefix behavior matters: a candidate whose first 12 actions are clean but whose next eight erase the gain should be reduced before the notebook contract is frozen, not after test rows are inspected. Conversely, lower-rank support should be evaluated through held-out expected utility and shrinkage, not by demanding an arbitrary minimum count. This policy acknowledges that a rare family may be real while preventing one favorable five-example slice from becoming an unregistered exception.

## 6. Targeted Fresh EDA for Rare High-Magnitude Top1 Corrections

### Three recent test packages

The comparison below is label-free. Feature values were reconstructed with V131's exact text normalization, TF-IDF/BM25 definitions, cached E5/Nomic arrays, fold-safe assessment parsing, and module semantics. Pair support refers to eligible train opportunities, not hidden test truth.

| Property | A: V130 rank 2 | B: V130 ranks 3/4 | C: V131 rank 2 |
|---|---:|---:|---:|
| Actions / ranks | 16; all rank 2 | 4; one rank 3, three rank 4 | 5; all rank 2 |
| Distinct structure | 14 pairs | 2 pairs; 3/4 `M_001→M_006`, 1/4 `M_001→M_010` | 2 pairs; 4/5 `M_001→M_015` |
| Base score margin, mean | `0.0399` | `0.0810` | `0.0742` |
| Head predicted utility, mean | `0.0274` | `0.0444` | `0.03285` |
| word / char full diff | `+0.0021 / −0.0117` | `+0.0031 / +0.0086` | `−0.0037 / −0.0109` |
| BM25 unigram full / latest diff | `+0.870 / +0.678` | `+0.588 / +0.014` | `+1.256 / +0.901` |
| E5 / Nomic diff | `−0.0078 / −0.0379` | `+0.0007 / −0.0136` | `+0.0218 / +0.0595` |
| readiness diff | `+0.044` | `−0.225` | `−0.020` |
| no-chat / negation | 25% / 62.5% | 25% / 75% | 60% / 40% |
| candidate / incumbent explicit | 50% / 56% | 25% / 0% | 20% / 20% |
| mean eligible pair support / benefit rate | `42.3 / 27.1%` | `4.25 / 57.5%` | `47.2 / 43.5%` |
| mean max near-template similarity | `0.926` | `0.989` | `0.949` |

Group B is not “stronger retrieval.” It is a concentrated, low-assessment, negative-readiness, often negated, near-template regime with almost no latest-message BM25 advantage and tiny but favorable pair support. Group C has the largest BM25 and dense candidate advantages, yet it is highly concentrated, frequently no-chat, sparse-cosine negative, and public-flat. This separates representation score from action utility: large agreement among BM25/E5/Nomic can still be redundant or shifted, whereas a counterintuitive lower-rank correction can be valuable when pair/rank context changes the cost-benefit geometry.

The public-important structural family was translated to train labels without using pseudo-GT. Among residual-eligible OOF rows, `M_001→M_006` has rank-3 support 2 with mean `+0.0370` (one benefit, one harm) and rank-4 support 3 with mean `+0.1213` (two benefit, one harm). `M_001→M_010` rank-4 has support 2 with mean `+0.1044` (one benefit, zero harm, one tie). Combined rank-4 support is only five, but three benefit / one harm / one tie and mean about `+0.1146` is positive structural evidence. It justifies hierarchical pooling; it does not justify hard-coding those pairs.

### Exact utility distribution and residual headroom

Current residual eligibility contains 2,905 OOF users per rank:

| Candidate rank | Benefit / harm / tie | Mean ΔNDCG | Positive mean / max | 95th / 99th percentile |
|---:|---:|---:|---:|---:|
| 2 | 797 / 2,013 / 95 | `−0.0454` | `0.0706 / 0.1782` | `0.1337 / 0.1738` |
| 3 | 513 / 2,237 / 155 | `−0.0958` | `0.1001 / 0.2413` | `0.1338 / 0.2231` |
| 4 | 406 / 2,337 / 162 | `−0.1256` | `0.1233 / 0.2737` | `0.1459 / 0.2523` |

This is not a power-law “heavy tail”: conditional-positive kurtosis is below 3 and the largest 10% of positives contributes roughly 21–24% of positive mass. The operational difficulty is a **bounded, sign-imbalanced mixture**. Benefit becomes rarer with depth, while its possible magnitude grows. Across all action rows, the pooled positive 90th-percentile threshold is `0.1891`; all 246 actions above it are rank 3 or 4 (118 and 128). Ridge underpredicts that slice by `0.3163` on average, while rank-specific benefit AUC is only `0.622–0.643`. Expected regression is behaving rationally under negative base rates, but it cannot identify the rare favorable component sharply.

Oracle headroom remains large but inaccessible: at 20 residual OOF users, the best true rank-2 actions would add `+0.000882`; allowing ranks 2–4 raises that ceiling to `+0.001338`. This is an upper bound, never a policy. The executed V130 scorer applied to its next untouched 20 users is only `+0.000024` rank2-only and `−0.000034` across all ranks, so mechanical extension is exhausted. V131's saved all-rank score is `+0.000148` at 20 and `+0.000242` at 40, but it surfaces only two rank-3 actions and no rank 4. The remaining gap is identification, not absence of label-side opportunity.

Stage-1 fusion differences and champion-versus-fusion disagreement for Groups A/B/C were not persisted in the action ledgers and were not expensively reconstructed; they are **NOT ESTABLISHED**. V132 must persist these columns so the next audit does not repeat this observability gap.

## 7. Utility Model Audit

The fresh model audit used the saved V131 random folds, exact V130-aligned OOF candidate geometry, actual residual eligibility, and fold-local quotas totaling 12, 20, or 40 actions (3, 5, or 10 test-equivalent). Reconstructed target-blind features reproduce saved V131 action scores with Pearson `0.9616`, Spearman `0.9648`, and MAE `0.00988`. Because training rows combine fold-specific target-blind vocabularies rather than recreating every V131 inner training table, new results are **exploratory**, not promotion-grade.

| Fixed model, 20 OOF actions | Rank policy | Gain | Benefit / harm / tie | Positive folds | Selected ranks |
|---|---|---:|---:|---:|---|
| saved V131 Ridge | rank 2 | `+0.0001709` | 13 / 5 / 2 | 3/5 | 20 / 0 / 0 |
| saved V131 Ridge | ranks 2–4 | `+0.0001482` | 12 / 6 / 2 | 4/5 | 18 / 2 / 0 |
| reconstructed global Ridge | ranks 2–4 | `+0.0002406` | 12 / 6 / 2 | 5/5 | 19 / 1 / 0 |
| Ridge + unshrunk pair one-hot | ranks 2–4 | `+0.0001961` | 12 / 6 / 2 | 5/5 | 17 / 1 / 2 |
| Ridge + pair×rank residual, shrink 20 | ranks 2–4 | **`+0.0002993`** | **15 / 4 / 1** | **5/5** | 19 / 1 / 0 |
| shallow LightGBM expected Δ | ranks 2–4 | `0` actions | — | 0/5 | all scores non-positive |
| two-stage benefit × positive magnitude | ranks 2–4 | `−0.0001293` | 9 / 10 / 1 | 2/5 | 0 / 2 / 18 |
| LightGBM q90 utility | ranks 2–4 | `−0.0001498` | 8 / 9 / 3 | 2/5 | 10 / 7 / 3 |

The small LightGBM used 80 trees, learning rate `0.03`, seven leaves, depth three, minimum child 100, feature fraction `0.8`, L1 `1`, and L2 `10`; the two-stage and q90 variants reused exactly that configuration. No sweep followed the results. Expected-utility LightGBM collapsed toward the globally negative mean. The two-stage model treated large possible magnitude as authority and selected 18 rank-4 actions, while q90 also over-selected lower ranks; both were harmful. **INVALID / RETRACTED:** “nonlinear action-level LGBM or an upper-tail objective is the obvious missing mechanism.” The participant clue does not rescue a model that fails the aligned falsifier.

Unshrunk pair one-hot does surface two rank-4 actions and remains positive, showing that pair context carries information, but it is weaker and vulnerable to rare-category scaling. A safer empirical-Bayes correction adds the training-fold residual sum for each incumbent×challenger×rank cell divided by `n+20` to the global Ridge prediction. It improves top-tail purity without a pair whitelist. At 12 actions it gains `+0.000238` with 10 benefit / 1 harm / 1 tie; at 20, `+0.000299`; at 40 it decays to `+0.000129`. This supports a **micro** policy and fixed shrinkage, not broad deployment.

Crucially, the best model does not manufacture rank-4 authority. Its top 20 include one rank 3 and zero rank 4, and its provisional top five test opportunities are all rank 2. Low-rank opportunities remain allowed, but evidence must overcome their base rate. V132 should include rank as learned context and hierarchical residuals, persist rank-conditional calibration, and never impose either a rank-2 ban or a lower-rank quota.

## 8. Representation and Embedding Audit Through V131

**TF-IDF.** Word and character full/latest/recency views remain the durable lexical backbone of both public-positive V129/V130 packages. The Group B/C contrast shows why word and character views should remain separate: Group B is positive on both full differences, while V131 Group C is negative on both despite high BM25/dense scores.

**BM25.** Okapi `k1=1.2`, `b=0.75` contributes saturation and document-length normalization distinct from cosine TF-IDF. V131 establishes full-history offline complement (`+0.0001779`) but not public incremental value. Keep unigram and unigram+bigram full-history differences as features in a different utility policy; drop temporal BM25 from the V132 primary because its aligned ablation was only `+0.0000463`. BM25 is evidence, not authority.

**E5 and Nomic.** Preserve the cached, pinned representations. E5 is `intfloat/multilingual-e5-base` revision `d128750597153bb5987e10b1c3493a34e5a4502a`, MIT-licensed, XLM-R-based, 12 layers, 768 dimensions, 512-token truncation, with required query/passage prefixes ([official model card](https://huggingface.co/intfloat/multilingual-e5-base)). Nomic is `nomic-ai/nomic-embed-text-v2-moe` revision `1066b6599d099fbb93dfcb64f9c37a7c9e503e85` plus code revision `7710840340a098cfb869c4f65e87cf2b1b70caca`, Apache-2.0, 475M total/305M active parameters, 768 dimensions, 512 tokens, explicit Indonesian support, and required search prefixes/remote code ([official model card](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe)). Both are already T4×2-feasible and reproducible.

Their action differences correlate `0.7348`, but prior sparse+E5 versus sparse+Nomic residual errors correlated `0.9966`. They are useful level features without proving residual independence. Do not remove a public-validated encoder, and do not add another pooled sentence encoder.

**Temporal/interactions.** A fixed temporal-only TF-IDF/BM25/state proxy gained `+0.000184` at 20 actions and was 5/5, but its errors correlate `0.9906` with the full-history proxy, all selected actions are rank 2, and 40% share one pair. It therefore fails to open expensive temporal dense encoding. This proxy does not prove message-level E5/Nomic is useless; it proves current evidence is too redundant to spend a submission cycle on it.

A candidate-only cross-encoder or late interaction remains genuinely novel. `BAAI/bge-reranker-v2-m3` is an Apache-2.0 multilingual cross-encoder based on BGE-M3, about 568M parameters/2.27 GB weights, supports up to 8,192 tokens but recommends 1,024, and is plausible on T4×2 when scoring only champion ranks 2–4 ([official model card](https://huggingface.co/BAAI/bge-reranker-v2-m3)). ColBERT-style MaxSim preserves token-level interactions while precomputing module tokens ([Khattab & Zaharia, SIGIR 2020](https://arxiv.org/abs/2004.12832)). Neither has MineToday residual evidence. If reopened, resolve and pin a full immutable revision, cache module tokens, score only three candidates per user, and require error correlation below `0.95` plus at least `+0.00015` at fixed nested micro-coverage. Until then: **no new embedding for V132**.

## 9. Reassessment of “Embedding + TF-IDF + BM25 + LGBM” Clue

The unverified clue is best interpreted as a heterogeneous feature stack, not a recipe.

- **Embedding:** directly supported as E5/Nomic level information in public-positive V129/V130; not supported as a third pooled checkpoint.
- **TF-IDF:** directly supported and durable.
- **BM25:** directly complementary offline in V131, especially full history, but the rank-2 BM25 package is publicly neutral. It should be a feature, not the selection thesis.
- **LGBM:** directly supported in Stage-1 relevance fusion across V129/V130, but the V131 fusion append gained only `+0.0000332`, and fresh shallow action-level expected/two-stage/q90 LightGBMs failed. “Use LGBM” is layer-specific, not universal.
- **“dll”:** unknown and irrelevant to a valid reconstruction.

**Most plausible interpretation:** nonlinear supervised fusion is valuable for compressing heterogeneous module relevance, while the current bottleneck is a second-stage decision problem with negative base rates, rank-dependent magnitude, and sparse pair support. V130 succeeded because aligned utility authority occasionally recognized a large lower-rank mistake. V131 added genuine lexical signal but used a pooled, margin-free, pair-free rank-2 policy whose selected test slice shifted. BM25 may help inside V132, but rare-action calibration and deployment geometry now dominate retrieval novelty.

## 10. Candidate V132+ Architectures

### Candidate A — Hierarchical all-rank expected utility (selected)

**Hypothesis/novelty:** V131's representation is adequate, but the utility head needs champion margin and partial pooling over pair×rank residuals; rank must be learned rather than vetoed. Use existing sparse/dense/state features plus full BM25, global Ridge, and fixed shrinkage 20. Fully nest policy anchors and exclude V129/V130 analogue actions. Exact V130 remains frozen; deploy 1–5 Top1 swaps from ranks 2–4.

**Evidence for:** exploratory `+0.0002993`, 5/5, 15/4/1 at 20; positive 12-action prefix; matched lower-rank pair support; only one of five provisional test choices overlaps V131 and selected-pair unseen rate drops to 20%. **Against:** surrogate rather than promotion-grade; provisional test choices remain all rank 2, three share one pair, and Nomic/E5 selected-feature SMDs are about `−1.91/−1.04`. **Cost:** reuse caches, about the V131 run (roughly 1.5–2 hours on T4×2). **Cheapest falsifier:** full nested reproduction plus the Section 12 support gates. **Public scenarios:** conservative `−0.00015–0`; realistic `0–+0.00030`; upside `+0.00030–+0.00080` if one rare high-value correction transfers.

### Candidate B — Nonlinear or upper-tail action utility (rejected now)

**Hypothesis/novelty:** shallow trees learn interactions among rank, sparse/dense/BM25 disagreement, readiness, and modules, or explicitly optimize rare positive magnitude. Same anchor, ranks, and geometry as A; exact V130 frozen; 1–5 test swaps.

**For:** mechanistically suited to heterogeneous features and the positive-magnitude gradient. **Against:** expected LightGBM produced no positive actions; two-stage and q90 all-rank policies were `−0.000129` and `−0.000150`, both 2/5, and over-selected lower ranks. **Cost:** cheap relative to Stage 1, low reproducibility risk, but current falsifier already fails. **Reopen only if:** a new decorrelated interaction feature first improves Ridge and the fixed tree learns positive expected utility without rank collapse. **Public scenarios now:** conservative substantial loss, realistic non-positive, upside not evidence-backed.

### Candidate C — Candidate-only multilingual cross-encoder or MaxSim (conditional)

**Hypothesis/novelty:** token/message interaction captures negation, comparison, and late intent missed by pooled E5/Nomic. Score only the exact V130 incumbent and ranks 2–4; feed scores to nested expected utility, never directly to submission. V130 remains frozen; at most five swaps.

**For:** truly decorrelated architecture in principle; multilingual open models and two T4s make candidate-only inference feasible. **Against:** no aligned MineToday gain, historical broad dense/cross-encoder attempts were weak or mismatched, the cheap temporal proxy has `0.991` error correlation, and dependencies/revision caches add risk. **Falsifier:** one pinned candidate-only OOF score, correlation `<0.95`, gain `≥+0.00015`, and nontrivial action disagreement. **Public scenarios:** conservative `−0.00020`; realistic unknown/near zero before a pass; upside potentially above A only after the branch opens.

### Candidate D — Lower-order boundary correction (rejected)

**Hypothesis/novelty:** residual value has moved from Top1 to B23/B34/B45. Preserve all V130 Top1 choices and make adjacent internal swaps. **For:** distinct bounded action family. **Against:** V130-aligned B23 was only about `+0.0000457`, B34 was negative, and B45 remains not established because required OOF identity/state was not persisted. It does not explain the rank2-only public ablation, which concerns Top1 promotions. **Cost:** cheap; **falsifier:** existing B23/B34 results already fail the relevance threshold. **Public scenarios:** conservative negative, realistic near zero, upside below the current #2 gap.

The immediate fallback is not Candidate B–D. It is a byte-identical V130 primary and no new leaderboard submission if Candidate A fails.

## 11. Selected Immediate V132 Architecture

Name the future notebook **V132 — Hierarchical Champion-Aligned Ranks-2–4 Utility Micro-Extension**.

### Immutable base and action contract

1. Load exact `candidate_v130_DIAGNOSTIC.csv`; assert SHA-256 `28dd…93e`, schema, module order, test IDs, and finite values. This is the only deployment base.
2. Reconstruct fold-local V129 and V130 analogue actions exactly as V131 did. Their union is ineligible. On test, freeze the exact 70 V130-changed users plus all previously protected/non-S0 users.
3. Generate only three opportunities per eligible user: swap current V130-analogue Top1 with current ranks 2, 3, or 4. Candidate rank is a feature. There is no rank quota, blacklist, or preference.
4. Apply at most five positive-score actions. Every change is a value swap, preserving Top5 membership, rank 5, tail order, and row score multiset. Any unrelated byte/row mutation fails closed to exact V130.

### Features and representation

Reuse V131 normalization, assessment masks, topic/state flags, fold-fit word/character TF-IDF full/latest/recency, pinned E5 and Nomic caches, and readiness. Retain candidate/incumbent identity and rank. Include V130's policy-anchor score margin because it encodes how much authority is required to overturn the champion. Add full-history unigram and unigram+bigram BM25 candidate/incumbent scores, differences, and ranks. Temporal BM25 remains an executed diagnostic, not a primary feature. Persist the Stage-1 LightGBM fusion difference for audit, but do not feed it to the primary head because V131's aligned append was negative; log a frozen diagnostic ablation only.

No pseudo-GT enters features, labels, folds, thresholds, or row selection. The v2/v3 files remain hypothesis notes only. No new embedding or external competition data is used.

### Nested model and objective

For every outer fold, build policy candidates with labels excluded. Within the outer-training set, produce target-safe candidate geometry for the action head as in V131. Fit standardized Ridge with alpha `10` on all ranks 2–4 and the exact counterfactual ΔNDCG target. Then compute training residuals and the fixed hierarchical correction

`pair_rank_adjustment = sum(y − ridge_prediction) / (pair_rank_support + 20)`.

At validation/test, add the relevant correction to Ridge expected utility. Unseen cells receive zero. This is partial pooling, not pair memorization. Compare the three corrected action scores per user and keep the maximum only if positive. Do not train the failed LightGBM/two-stage/q90 heads in the primary pipeline.

Register 0.5% as the primary outer-fold cap: four actions per 800-row fold, equivalent to five test users. Log deterministic 0.3% and 1.0% sensitivity prefixes, but do not choose coverage after seeing results. The 0.3% prefix must support concentration; the 1.0% curve must not be used to enlarge deployment when its marginal tail is weak.

### Runtime and artifacts

Run sparse/BM25 CPU work while the two pinned dense caches are verified. No encoder inference should occur when hashes/keys match. Parallelize independent Stage-1 module fits across the two T4s only as already established; action heads are CPU-cheap. Expected wall time is approximately 1.5–2 hours, dominated by the five nested worlds, not the hierarchical correction.

Persist OOF feature matrices, candidate anchors/margins, fusion diagnostics, pair×rank supports/adjustments, fold-local selected ledgers, test opportunity ledger, model/cache revisions, geometry audit, candidate hash, and fail-close primary hash. V131 omitted enough intermediate geometry to leave Stage-1 group comparisons not established; V132 should close that observability gap.

## 12. Validation and Submission Decision Policy

### Hard correctness gates

Any target leakage, broken nested OOF, policy-anchor leakage, pseudo-GT modeling use, train/test feature mismatch, wrong exact NDCG, unstable ties, non-finite value, unpinned model/cache, mutation of a protected V130 row, unexpected Top5/rank5/tail/multiset change, schema drift, or irreproducible bytes means **DO NOT SUBMIT** and write exact V130 as primary. These gates are absolute.

### SUBMIT

All hard gates pass, and the registered primary meets all of:

- nested V130-aligned incremental mean at least `+0.00020` at 20 OOF actions;
- at least 4/5 random folds positive, with no fold below `−0.00020`;
- 12-action deterministic prefix positive; 20-action selected slice at least 13 benefit, at most 5 harm;
- all-rank policy is not materially worse than rank2-only, and every lower-rank selection receives a fold-training shrunken estimate rather than a quota;
- test maximum pair count at most 2/5, selected-pair unseen rate at most 40%, no core numeric selected SMD above `|1.5|`, and no four-of-five overlap with V131;
- one to five actual positive-score actions; expected utility does not force filling the cap.

The `+0.00020` threshold reflects the current #2 gap, not an assumed public-transfer ratio. CI lower bounds and grouped worlds are reported but are not automatic vetoes.

### BORDERLINE

Correctness is perfect, but mean is `+0.00010–+0.00020`, only 3/5 folds are positive, a small rare-action family has favorable real-label support but wide uncertainty, or one test-support condition fails. A one-to-three-action candidate may use a daily slot only if it is structurally unlike V131, has bounded downside, and the slot budget permits. The current exploratory V132 surrogate would be **BORDERLINE**, not SUBMIT, because its provisional five test actions concentrate three in one pair and Nomic shift exceeds `|1.5|` despite good OOF purity.

### DO NOT SUBMIT

Any hard gate fails; nested gain is below `+0.00010` or non-positive; two or fewer folds are positive; benefit does not exceed harm; the policy selects lower ranks mainly because of q90/magnitude rather than expected utility; or test geometry repeats V131 (four-of-five overlap, four-of-five one pair, or unsupported shift). Never spend a slot merely because the exact V130 base remains safe on Kaggle.

## 13. Expected Gain, Failure Modes, and Public Scenarios

The selected surrogate's `+0.000299` is not a forecast. V129/V130/V131 show that local magnitude does not map monotonically to public magnitude. Conditional on a full nested pass, reasonable package scenarios are:

- **Conservative:** `−0.00015` to `0`. Pair support or public/private sampling differs, and a few wrong swaps erase expected gain.
- **Realistic:** `0` to `+0.00030`. A less shifted, non-V131 micro-package transfers partially and may cross `0.66193` near the upper half.
- **Upside:** `+0.00030` to `+0.00080`. One or more rare, high-magnitude corrections transfer as the V130 lower-rank family did. This is possible, not probable or guaranteed.

Principal failures are repeated-label policy overfit, pair×rank support too sparse, shrunken correction still dominated by a test pair, champion margin changing calibration, and selected dense/BM25 disagreement outside OOF support. A lower-rank action can have larger upside and larger downside; its presence is neither a success condition nor a warning by itself. Private robustness favors a few well-supported actions over a broad attempt to reproduce a public anomaly.

## 14. V132+ Conditional Roadmap

1. **Immediate V132:** implement Candidate A exactly, fully nested, with one registered primary and fail-close V130. Do not submit until Section 12 is applied.
2. **If utility passes but test support fails:** stop; preserve V130. Do not tune pair lists, ranks, or coverage to test users.
3. **One conditional research fallback:** only if Candidate A shows remaining errors tied to token/temporal disagreement, run a candidate-only pinned BGE reranker or MaxSim falsifier. Open it only at error correlation `<0.95` and nested `≥+0.00015`; otherwise close it.
4. Keep nonlinear expected/two-stage/q90 utility and lower-order boundaries closed until genuinely new evidence appears. Do not create pseudo-GT v4.

## 15. Final Recommendation

Build one V132 notebook around exact V130, global expected-utility Ridge, ranks 2–4, champion margin, full-history BM25, and a fixed shrink-20 pair×rank residual. Freeze all 70 V130 actions and allow at most five new Top1 swaps. Do not add an embedding, tail objective, rank quota, or action-level LightGBM now. The next notebook session should first reproduce the full nested `+0.00020`-class signal and test-support gates; if it cannot, emit byte-identical V130 and do not submit.
