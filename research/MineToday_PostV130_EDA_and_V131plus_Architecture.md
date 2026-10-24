# MineToday — Post-V130 Public Breakthrough, Targeted EDA, and V131+ Architecture

## 1. Executive Decision

**Decision.** Build V131 as a **globally pooled, champion-aligned BM25 residual utility micro-extension**. Preserve the exact public-scoring V130 package, freeze all 70 existing Top1 actions, learn exact counterfactual utility on the remaining users, and permit at most five new **rank-2-only Top1 score swaps**. The new head should reuse the existing sparse, E5, Nomic, readiness, and champion-state views, add the five already-defined V128 BM25 streams, remove incumbent→challenger pair one-hot features, and retain fixed Ridge regularization `alpha=10`. This is the only fresh candidate with direct evidence at the scale of the approximately `+0.00020` gap to #2.

The protected baseline requires an artifact correction. `results/v130/v130/submission_v130_primary.csv` is byte-identical to V129 because the executed V130 gate returned `BORDERLINE`; its SHA-256 is `34f334429c2113ba5b96283a7974fad472068ea08047ba3275fc44057e3a95df`. The file actually submitted and reported by the user at public NDCG@5 `0.66173` is `results/v130/v130/candidate_v130_DIAGNOSTIC.csv`, SHA-256 `28dd2121c72e910be8a01b31cd43d022c7fdf080d8e628bd8f30abc3a24bc93e`. Despite its filename, that exact file is now the competition champion. Every incremental V131 implementation must start from and verify those bytes.

The current state is:

- **FACT — public progression:** `0.66028 → 0.66075 → 0.66173`; V129 added about `+0.00047`, V130 added about `+0.00098`, and the branch added about `+0.00145` cumulatively.
- **FACT — strategic position:** user-supplied rank is approximately #3, the last reported #2 score is `0.66193`, and the immediate gap is about `+0.00020`.
- **FACT — V130 local evidence:** fully nested incremental gain `+0.000394293`, 4/5 random folds positive, with user bootstrap interval `[-0.00000445, +0.00086545]`; nevertheless its public increment was about 2.49 times the OOF estimate.
- **SUPPORTED INFERENCE:** policy/champion alignment and conservative residual authority materially improved deployment fidelity. They did not make OOF and test identical, and they do not identify any correct test action.
- **FACT — remaining same-head headroom:** the next 0.5% selected by the existing V130 scorer was `−0.000157639` with 0/5 folds positive. Another mechanical V130 layer is rejected.
- **EXPLORATORY EVIDENCE — reopened BM25:** a V130-aligned, globally pooled raw-feature Ridge with fixed V128 BM25 streams produced `+0.000252960` at the next 0.5%, 5/5 positive folds, 12 beneficial / 6 harmful / 2 tied actions, worst fold `+0.00009349`, user CI `[+0.00004675, +0.00047773]`, and near-template-cluster CI `[+0.00004987, +0.00049745]`. This probe was target-safe at the action-fold level but was not a full rerun of V130 Stage 1; it justifies V131, not immediate submission.

V130 changes the scientific conclusion in two ways. First, uncertainty intervals and grouped stresses must be advisory evidence rather than automatic vetoes: a rule requiring a positive CI lower bound or every stress world positive would have withheld the largest genuine improvement in this lineage. Second, success did not imply that representation is finished. Once the residual problem was aligned to the protected champion, BM25 became complementary in a simpler global utility head even though it was unhelpful in V128's mismatched, high-variance stack.

No new embedding, lower-order action family, replacement policy, or wider coverage is recommended now. Kaggle retains the best score, so one correct, pre-registered V131 package is worth a slot if it clears the calibrated policy in Section 11. The `0.96609` outlier remains only an unverified methodological clue; nothing here supports hidden-label reconstruction or chasing its magnitude.

## 2. Evidence Update: V128 → V129 → V130

The direct artifacts establish a coherent progression from broad representation search to policy-aligned selective action.

| Version | Direct local result | Deployment | Public evidence | Causal lesson retained |
|---|---:|---|---:|---|
| V128 | High-coverage DESIGN gains around `+0.00115` to `+0.00159` at 5–6%; imperfect second-level nesting | Gate handed off to a tiny ~0.5% `no_bm25` branch; primary SHA `443785…` was the old champion | Not submitted | Broad feature strength existed, but branch selection and purity gating discarded useful coverage; BM25/KNN added variance in that context |
| V129 | `+0.001130193`, 5/5 random folds; exact/latest/near/contiguous aggregates all positive | 50 Top1 swaps at fixed 5%; primary SHA `34f334…` | `0.66075`, about `+0.00047` | Lean heterogeneous representation plus fully nested exact-utility learning works publicly |
| V130 | `+0.000394293`, 4/5 random folds; mixed grouped worlds | 20 new Top1 swaps over immutable V129; actual submitted SHA `28dd212…` | `0.66173`, about `+0.00098` | Correct deployment geometry and selective residual authority can matter more than impressive aggregate OOF magnitude |

### V128: useful representation evidence, invalid decision machinery

The V128 notebook and `results/v128/v128/` confirm word/character TF-IDF, E5, Nomic, assessments, BM25, neighbor/template statistics, per-module LightGBM, and direct Top1 utility. At 5% and 6%, `full` gained `+0.00119786` and `+0.00108452`; `no_bm25` gained `+0.00125732` and `+0.00136204`; `no_neighbor` gained `+0.00129041` and `+0.00159139`; the dense branch gained `+0.00115390` and `+0.00127214`. Thus sparse+dense+assessment fusion survived, while neither BM25 nor target-derived neighbors showed incremental value in that design.

The failure was operational. A benefit≥2×harm purity rule and branch hand-off moved the final LOCK policy to a tiny region rather than preserving the stronger 5–6% design. The action learner was also not a clean second-level nest. The final primary was exact old champion bytes and V128 was not publicly tested. Therefore **INVALID / RETRACTED** is any claim that V128 publicly disproved BM25, or that its tiny final policy disproved broader Top1 authority. It only showed that BM25 and neighbors were not helpful inside that particular noisy comparison.

### V129: first public confirmation, with a geometry flaw

V129 removed BM25 and target-derived neighbors, kept TF-IDF, E5, Nomic, assessments, champion geometry, and 17 per-module LightGBMs, and fully nested Stage 1 into an exact-ΔNDCG Ridge action head. `results/v129/v129/v129_decision_report.json` records `+0.001130193`, 5/5 positive random folds, user CI `[+0.00036428, +0.00188396]`, and 50 fixed-coverage test actions. Stress gains were exact `+0.000645`, latest `+0.000820`, near `+0.001158`, and contiguous `+0.001024`; fold directions were respectively 3/5, 5/5, 5/5, and 4/5. Public `+0.00047` confirmed the branch's sign but transferred only about 42% of primary OOF magnitude.

The prior report's geometry warning is confirmed by direct code and artifacts. OOF candidates came from a sparse Ridge anchor, whereas deployment candidates came from the historical champion. Full-fit sparse versus champion agreement was only 78.3% at Top1, mean Top-4 Jaccard was 0.7591, and rank correlation was 0.8626. The sparse anchor itself scored about `0.653924` OOF versus Stage-1 fusion around `0.661244`. V129 succeeded despite a validation/deployment action-space mismatch; it did not validate that mismatch.

### V130: alignment succeeds, but the automatic primary did not contain it

Direct notebook inspection confirms full outer/inner nesting for the V42 policy anchor and Stage 1, action labels relative to the fold-local policy state, exclusion of fold-local V129-selected users, and exact deployment over untouched users. `results/v130/v130/v130_decision_report.json` records the final gate, hashes, uncertainty, and geometry checks. V129's 50 actions are unchanged. The 20 new actions preserve each row's score multiset, Top-5 membership, rank 5, tail, and protected eligibility. The main folds were `[+0.00024852, +0.00075523, +0.00057430, −0.00003651, +0.00042993]`.

Grouped results were less attractive: exact-history `−0.00000275` (3/5), latest-template `+0.00057331` (4/5), near-template `−0.00000246` (2/5), and contiguous-ID `−0.00012683` (1/5). The executed decision therefore marked `BORDERLINE`, kept exact V129 as `submission_v130_primary.csv`, and preserved the new package only as `candidate_v130_DIAGNOSTIC.csv`. Manual use of the diagnostic candidate produced the breakthrough. This discrepancy is scientifically important: the old automatic gate and filename semantics are now superseded by observed package evidence.

### Conclusions changed by the sequence

- **Strengthened:** TF-IDF + E5/Nomic + assessment + nonlinear per-module fusion is a durable information base; direct exact-action utility and Top1 score swaps are effective; fully nested validation and exact geometry audits are essential.
- **Strengthened:** selective extension is safer than replacing a public-positive package. V130 froze V129 and gained again.
- **Weakened:** CI-lower-zero, exact/near/contiguous fold positivity, and a large absolute OOF threshold are not reliable universal submission vetoes.
- **Overturned:** “BM25 is closed.” The correct conclusion is “BM25 was unhelpful in V128's context”; a new champion-aligned residual probe is positive.
- **Not established:** which V129 or V130 test users were improved, whether all 70 should remain in a private-score-optimal package, and whether OOF is generally pessimistic. Public package deltas cannot answer those questions.

## 3. Why V130 Outperformed Its Offline Estimate

The observed fact is package-level: V130's exact submitted bytes improved public NDCG@5 by about `+0.00098`, versus `+0.000394293` OOF. The evidence supports several contributing explanations, but no decomposition is identifiable without hidden labels.

### 3.1 Policy alignment is the strongest causal change

V130 fixed a concrete V129 defect. It reconstructed the champion/policy anchor inside each fold, generated candidates from that anchor, excluded the fold-local analogue of V129-selected users, trained exact counterfactual utility on those opportunities, and deployed on the same residual eligibility definition. This made the estimated action problem correspond to the actual test operation. V129's stronger OOF score partly evaluated a different candidate surface; V130's smaller OOF score evaluated a closer one.

That does not mean distribution shift disappeared. Selected OOF versus selected test pair total variation was `0.825`; 75% of selected test incumbent→challenger pairs were absent among the 80 selected OOF pairs. Test utility was lower by SMD `−0.425`, chat count by `−0.706`, E5 difference by `−0.722`, and Nomic difference by `−0.908`. Candidate-rank TV was only `0.15`, however, and all feature semantics and eligibility rules matched. Alignment repaired the *definition* of an action, not the empirical category mix.

### 3.2 V130 exercised different residual authority than V129

V129's 50 test actions had mean predicted utility `0.06559`; V130's 20 had `0.03082` (SMD `−1.92`). V130 acted on fewer-message users (mean chat count `1.65` versus `3.56`), weaker raw retrieval support (E5 challenger−incumbent `−0.0061` versus `+0.0033`; Nomic `−0.0330` versus `+0.0190`), less explicit challenger mention (45% versus 84%), and less dual-dense agreement (20% versus 50%). This is exactly the kind of residual opportunity a nonlinear, policy-aware utility learner should discover after obvious retrieval-supported actions are consumed.

Real-label OOF slices show that this behavior was not automatically reckless. Among V130's 80 selected OOF actions, chat-count≤2 contributed `+0.00022585` with 4/5 positive folds; the seven no-chat actions contributed `+0.00003700` with 4/5. Rank 2 carried the signal: 76 actions contributed `+0.00045845` with 5/5 folds, whereas four rank-3 actions contributed `−0.00006416` and rank 4 had no selected OOF support. The test package included 16 rank-2, one rank-3, and three rank-4 promotions. The public result is compatible with rank-2 strength, but it cannot validate the individual lower-supported actions.

### 3.3 Template support was more favorable on the selected test package

V130-selected test users had 65% latest-skeleton support, 45% full-skeleton support, and 65% near-template support. V129's corresponding values were 62%, 24%, and 56%; V130's fold-local selected OOF values were only 47.5%, 23.75%, and 47.5%. This label-free geometry makes a favorable public composition plausible. It also explains why an aggregate grouped fold can be pessimistic: grouped worlds intentionally remove the very template support that happened to be more common in deployed selections.

This is a **SUPPORTED INFERENCE**, not proof that templates caused the gain. Test support is measured without labels, and the public subset composition is unknown.

### 3.4 Public-subset variance can plausibly amplify a sparse policy

A bootstrap sensitivity exercise sampled V130's 4,000 OOF user deltas at hypothetical public sizes. For 200, 300, 500, and 1,000 users, the simulated 95% ranges were respectively `[-0.001516, +0.002349]`, `[-0.001136, +0.001999]`, `[-0.000802, +0.001617]`, and `[-0.000440, +0.001263]`. The probabilities of a sampled mean at least `+0.00098` were 25.3%, 21.8%, 16.3%, and 8.7%. These are illustrative because Kaggle's public size and sampling process are not known, but they show that a 20-action package can have a substantially amplified public mean without any contradiction.

Leaderboard rounding is not an explanation: rounding two NDCG values to five decimals changes their apparent difference on the order of `10^-5`, not `9.8×10^-4`.

### 3.5 What remains unidentifiable

It is **NOT ESTABLISHED** whether full-fit training made V130 systematically stronger than nested OOF, which of the 20 actions helped, whether one high-impact user dominated the public delta, or whether the private split will preserve the gain. V130's saved OOF NPZ does not include policy margins or Stage-1 feature matrices, so some post-hoc calibration questions cannot be reconstructed exactly. The correct posterior is narrower: aligned residual policies can transfer much better than conservative aggregate diagnostics imply, and package-level public evidence deserves weight alongside honest OOF—not in place of it.

## 4. Validation Calibration Audit

Only three submitted packages have sufficiently interpretable local→public pairs. That is enough to reject rigid magnitude gates, not enough to fit a transfer model.

| Package | Local protocol and delta | Action / coverage | Public delta | Sign | Magnitude ratio | Calibration use |
|---|---:|---|---:|---|---:|---|
| V30 | Three fresh salts, chosen `ltr_base` mean `+0.00073154`; 3/3 salts positive; CI `[+0.00033685,+0.00114474]` | Mixed LTR, 77 changed users | `+0.00044` over `0.65919` | Correct | `0.60×` | Comparable submitted package, but broader action family |
| V31 | `+0.00015262` over V30, 3/3 salts positive; CI crosses zero | Failed development gate; primary reconstructed V30 | No new candidate submitted | N/A | N/A | Exclude from transfer calibration |
| V49 | Development around `+0.0001993`, 4/5; failed nested floor and fail-closed | 0 deployed changes | No new candidate submitted | N/A | N/A | Context only; not a package pair |
| V127 | Corrected pooled `+0.00070954`, CI positive, worst fold `+0.00009467`; adopt=false | Local-only; 57 diagnostic actions | No public candidate | N/A | N/A | Shows local evidence alone is insufficient; leaky v1 is invalid |
| V128 | High-coverage positive DESIGN; final exact anchor | 0 deployed changes | No public candidate | N/A | N/A | Exclude from transfer calibration |
| V129 | Fully nested random `+0.00113019`, 5/5; all grouped aggregates positive | 50 Top1 swaps | `+0.00047` | Correct | `0.42×` | Strongest pre-V130 comparator, but OOF/deployment candidates mismatched |
| V130 | Fully nested aligned incremental `+0.00039429`, 4/5; mixed grouped worlds | 20 additional Top1 swaps | `+0.00098` | Correct | `2.49×` | Most relevant comparator and proof old automatic gate was too strict |

The three submitted candidates V30, V129, and V130 transferred sign correctly (3/3). Their magnitude ratios—about `0.60`, `0.42`, and `2.49`—are too variable for regression or a “public multiplier.” There is no empirical basis to say OOF is always pessimistic: V30 and V129 shrank, while V130 expanded.

Random-fold aggregate and direction are the most consistently useful positive evidence. Grouped worlds are useful as catastrophe detectors but poor quality rankers. V129 was positive in all major worlds but transferred modestly. V130 was approximately zero in exact and near worlds and negative in contiguous, yet transferred most strongly. V30 similarly had a strong sealed-template result (`+0.000969`) and an almost flat chronological result (`+0.000042`) while public gain was positive. This pattern argues against requiring every adversarial partition to be positive.

Champion alignment improved public transfer far more than it improved the nominal OOF result, but the causal quantity is confounded with residual coverage and test composition. Coverage and rank offer more actionable calibration: public-positive V129 used 5% authority, V130 added 2%, and V130 OOF signal was almost entirely rank 2. CI and fold stability still measure uncertainty; they do not decide alone.

For V131, validation should answer three questions in order: is the computation correct and leakage-free; is the primary random-world incremental signal positive and sufficiently stable; and does deployed geometry stay within feature/category support? Grouped tests should veto only a material, mechanism-relevant collapse. Repeatedly optimizing thresholds on the same 4,000 labels is itself selection overfitting; Cawley and Talbot show that finite-sample model-selection criteria can be overfit at magnitudes comparable to algorithm differences ([JMLR, 2010](https://www.jmlr.org/beta/papers/v11/cawley10a.html)). Therefore V131 must pre-register one primary representation, regularization, rank family, and coverage before full evaluation.

## 5. Pseudo-GT v2 → v3 Diagnostic Findings

The provenance files in `research/MineToday_pseudo_GT_v2_after_V129/` and `research/MineToday_pseudo_GT_v3_after_V130/` are explicit. V2 combined the earlier synthetic reconstruction with V129-selected promotions while leaving pseudo K and jitter rules unchanged. V3 applies the exact submitted V130 diagnostic package after V129. It is endogenous posterior bookkeeping, not recovered relevance.

The v2→v3 audit contains 28 changed rows: 20 V130 incremental actions and eight corrections where v2 did not match exact V129 Top1. All 20 V130 users changed pseudo Top1. Before that forced update, all 20 were tagged `BASELINE` in v2 and all had pseudo `K=5`. V3's promoted candidate ranks are 16 at rank 2, one at rank 3, and three at rank 4. None of those fields independently validates V130.

The safe diagnostic translation is:

| Pseudo-GT observation | Label-free hypothesis | Real-train check | Decision |
|---|---|---|---|
| V130 changed only v2 `BASELINE`, pseudo-K5 users | V130 may target uncertain middle-shape users | True-K slices among selected OOF: K4 `n=24`, `+0.000093`; K5 `n=27`, `+0.000085`; K6 `n=29`, `+0.000216` | **Rejected as specific K5 rule**; value spans all true K and is largest for K6 |
| Promotions included lower-readiness challengers | Readiness difference may define a harmful/beneficial regime | Lower `n=30`, `+0.000154`, 4/5; neutral `n=28`, `+0.000123`, 3/5; higher `n=22`, `+0.000117`, 3/5 | **No readiness hard rule**; the earlier “lower is harmful” concern does not persist |
| Most promoted candidates were rank 2 | Rank 2 may carry cleaner exact-action utility | Rank2 `n=76`, `+0.000458`, 5/5; rank3 `n=4`, `−0.000064`, 1/5 | **Train-validated confirmation**; restrict V131 deployment to rank 2 |
| V130 users were changed by the new policy despite weak raw support | Utility fusion may rescue retrieval contradictions | Both dense favor `n=47`, `+0.000309`; neither dense favors `n=19`, only `+0.000026` | **Partial confirmation**; contradictions are possible, but joint dense support is still much safer OOF |

The v3 pack must not enter V131 features, labels, eligibility, test-user selection, or validation. It cannot be used to “freeze good” or “correct bad” named users. The only architecture-relevant outcomes above came from label-free slice definitions tested against organizer train labels. Everything else is unsupported speculation.

## 6. Targeted Fresh EDA After V130

The audited dataset remains 4,000 train users, 1,000 test users, 17 modules, 14,573 chat rows, and 5,000 assessment rows. Train relevance cardinality is K4 for 1,362 users, K5 for 1,285, and K6 for 1,353; 251 assessment cells require the established invalid-value handling. The official ordered split is `U_0001–U_4000` versus `U_4001–U_5000`. Fresh work therefore focused on action geometry rather than repeating schema EDA.

### 6.1 V129 versus V130 actions

| Selected test geometry | V129 (n=50) | V130 incremental (n=20) |
|---|---:|---:|
| Candidate ranks 2 / 3 / 4 | 42 / 5 / 3 | 16 / 1 / 3 |
| Mean utility score | `0.06559` | `0.03082` |
| Mean champion margin | `0.06954` | `0.04809` |
| Mean chat count | `3.56` | `1.65` |
| Mean readiness challenger−incumbent | `−0.285` | `−0.010` |
| Mean E5 difference | `+0.00334` | `−0.00607` |
| Mean Nomic difference | `+0.01903` | `−0.03302` |
| Candidate explicitly mentioned | 84% | 45% |
| Both dense views favor challenger | 50% | 20% |
| Latest/full lexical contradiction | 58% | 45% |

The incumbent→challenger pair TV between the two packages is `0.94`; utility SMD is `−1.92`; margin SMD is `−0.41`; chat-count SMD is `−0.86`. V130 is not simply “the next 20 V129-like actions.” It is a lower-confidence, lower-chat residual policy operating after the obvious 50 opportunities were removed. That helps explain why its OOF magnitude was smaller and why reusing the same head for a third layer is unlikely to work.

### 6.2 Residual Top1 headroom after the V130 analogue

Using the saved fully nested V130 scores and excluding V130-selected OOF users gives the following genuinely incremental curve:

| Additional coverage | Actions | ΔNDCG | Positive folds | Benefit / harm / tie |
|---:|---:|---:|---:|---:|
| 0.25% | 10 | `+0.00003703` | 3/5 | 6 / 2 / 2 |
| 0.50% | 20 | `−0.00015764` | 0/5 | 8 / 10 / 2 |
| 1.00% | 40 | `−0.00026332` | 0/5 | 14 / 24 / 2 |
| 1.50% | 60 | `−0.00023276` | 1/5 | 26 / 32 / 2 |
| 2.00% | 80 | `−0.00003236` | 2/5 | 39 / 38 / 3 |
| 3.00% | 120 | `+0.00000726` | 3/5 | 60 / 56 / 4 |
| 4.00% | 160 | `+0.00001645` | 3/5 | 80 / 76 / 4 |

The later half-percent block from 1.5–2.0% happens to contribute `+0.00020040`, but the score-ranked blocks before it are negative. Selecting that block after seeing labels would be a post-hoc nonmonotonic policy, not deployable evidence. **FACT:** V130 consumed the reliable head of its own score ordering. A third unchanged Top1 layer is closed unless a new representation reorders residual opportunities.

### 6.3 Extension versus replacement

The aligned V130 extension produced `+0.00039429` from 80 OOF actions. A fresh aligned 5% replacement policy produced only `+0.00032878`, with 4/5 folds and CI crossing zero, while exposing 200 actions. On the 200 frozen V129-analogue users, only 20% of V130-aligned opportunities had positive predicted value; mean best realized utility was `−0.02697`. The replacement overlapped 18 such users and their aggregate contribution was `−0.00006880`.

This is policy-level evidence for freezing public-positive actions. It is not proof that every one is correct. Replacement has lower measured return, wider authority, and a direct risk of undoing a `+0.00145` public lineage; extension is the rational default.

### 6.4 Lower-order and relevance-shape probes

A V130-aligned exploratory B23 utility probe gained only `+0.00004572` at 0.5%, 3/5 folds, with CI crossing zero; 1% was effectively zero and 2% negative. B34 was negative at 0.5%, 1%, and 2%. B45 is **NOT ESTABLISHED** because V130 did not persist rank-5 identity/policy-anchor OOF state. No lower-order family has enough evidence to compete with BM25 Top1.

Nested prediction of relevance shape also failed its falsifier. True K prediction had RMSE `0.829` and rounded accuracy `0.3208`; predicted IDCG correlation was `−0.0067`. Adding nested shape/readiness interactions to the raw action proxy changed the next 0.5% from `−0.000103` to `−0.000055`, still negative. K, IDCG, and readiness should remain diagnostics, not a V131 architecture branch.

## 7. Representation and Embedding Audit Through V130

### 7.1 What has durable evidence

Sparse word/character TF-IDF is the backbone across the public-positive lineage. E5 and Nomic add semantic views and were present in both V129 and V130. Assessment/readiness and module identity provide structured context. The 17 per-module LightGBMs concentrate heterogeneous nonlinear interactions before exact-action learning; the original LightGBM paper's design targets efficient tree learning in high-dimensional data ([NeurIPS 2017](https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)), matching this small mixed-feature problem better than a new deep end-to-end trainer.

The preserved caches are reproducible: E5 is `intfloat/multilingual-e5-base` at revision `d128750597153bb5987e10b1c3493a34e5a4502a`, 768 dimensions; Nomic is `nomic-ai/nomic-embed-text-v2-moe` at revision `1066b6599d099fbb93dfcb64f9c37a7c9e503e85` with code revision `7710840340a098cfb869c4f65e87cf2b1b70caca`, also 768 dimensions. E5's official card describes a 12-layer multilingual model initialized from XLM-R and requires the trained query/passage prefixes ([model card](https://huggingface.co/intfloat/multilingual-e5-base)). Nomic v2 is Apache-2.0, supports about 100 languages, uses 305M active parameters and 768-dimensional output with a 512-token limit ([model card](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe)). Both are open-source and already cached, so they are competition-compatible and T4×2-feasible when revisions and remote code are pinned.

### 7.2 Residual diversity is insufficient for another embedding now

The fresh V130-aligned raw-feature probe was deliberately cheap: fixed action folds, exact V130 candidates and targets, target-blind full-fit representation features, but no full Stage-1 rerun. E5 and Nomic candidate−incumbent differences correlated `0.741` and agreed in sign 82.0%. Sparse+E5 and sparse+Nomic action errors correlated `0.99657`; their 2% selections overlapped on 50 of 80 users. This is slightly more diverse than the prior ~`0.998` estimate but still far too redundant to justify checkpoint shopping.

Individual next-0.5% proxies were weak: sparse-only `+0.0000506`; sparse+E5 `−0.0000249`; sparse+Nomic `−0.0000762`; sparse+both+state `−0.0001032`. These are exploratory, but they provide no case to drop either public-validated encoder or add a third single-vector model. Larger E5, BGE-M3, GTE, Qwen-family, IndoBERT, XLM-R, cross-encoder, and broad dense experiments already have weak or deployment-mismatched historical results; no new evidence overcomes that prior.

### 7.3 Temporal and late-interaction models remain conditional research, not V131

Message-level temporal pooling and token-level late interaction are genuinely different from concatenated single-vector embeddings. ColBERT's MaxSim retains token-level interactions while allowing document representations to be precomputed ([Khattab & Zaharia, 2020](https://arxiv.org/abs/2004.12832)). That gives a plausible mechanism for exact module mentions, negation, and late preference changes. It also adds token caches, custom scoring, dependency risk, and another selection surface. There is no MineToday champion-aligned residual gain yet.

The cheapest future falsifier is not full training: compute candidate-only Top-4 temporal or MaxSim features using one pinned multilingual encoder, cross-fit only the utility head, and require residual error correlation below `0.95` against V131/V130 plus at least `+0.00015` at fixed 0.5% coverage. Until both conditions hold, **no new embedding** is the evidence-based recommendation.

## 8. Reassessment of “Embedding + TF-IDF + BM25 + LGBM” Clue

The participant statement is unverified and generic, but our direct evidence now gives it a precise interpretation.

- **TF-IDF — confirmed durable.** Sparse word/character views survived every successful version and encode explicit course vocabulary efficiently.
- **Embedding ensemble — confirmed useful, not necessarily diverse.** E5 and Nomic were present in V129 and V130, though their residual errors are highly correlated.
- **LGBM — strongly supported as information concentration.** Per-module nonlinear fusion is a central shared component of the two public-positive packages. Public evidence is package-level, so no single-feature causal share is identifiable.
- **BM25 — reopened in the correct residual context.** It was directionally unnecessary in V128, but that comparison used imperfect stacking and a different action policy. The new question is whether lexical saturation/length normalization reorders opportunities after exact V130.
- **“dll” — unknown.** Guessing private features is neither scientific nor needed.

The V3 BM25 probe reused the exact simple V128 definition: Okapi `k1=1.2`, `b=0.75`; module documents indexed as unigram and unigram+bigram; user views from full history, latest message, and exponentially recency-weighted messages. BM25's probabilistic relevance framework explicitly models term saturation and document-length normalization, which are not identical to cosine TF-IDF ([Robertson & Zaragoza, 2009](https://dblp.org/rec/journals/ftir/RobertsonZ09)). That difference is the mechanism under test.

At the next 0.5% beyond V130, a global raw head without BM25 gained `+0.00007675`; adding full-history BM25 yielded `+0.00019766`; temporal BM25 alone yielded `+0.00004858`; all five fixed streams yielded `+0.00025296`, 5/5. A pair-specific version with all BM25 streams gained `+0.00017628`. Unigram-only and bigram-only pair-specific additions were negative, while full-history views were positive. The signal therefore appears to be pooled full-history lexical complementarity, not “BM25 everywhere.” Using all five already-preregistered V128 streams avoids choosing one view after observing this pilot; full-versus-temporal remains an explanatory ablation.

Regularization sensitivity did not reveal a knife edge: alpha 3, 10, 30, and 100 gave `+0.000202`, `+0.000253`, `+0.000253`, and `+0.000276`; alpha 10 remains fixed because it is inherited, not because it maximized the probe. A calibrator wrapped around saved V130 utility scores did not work (`+0.0000348` or negative at the next 0.5%) and was not fully second-level nested. BM25 should therefore be integrated into a newly cross-fit global utility head, not stacked post-hoc onto saved V130 scores.

The best-supported interpretation is a combination of clues 1, 2, and 4 from the prompt: heterogeneous supervised fusion is the main lineage mechanism; V128 did not fairly close champion-aligned BM25 complementarity; and policy alignment/selective authority enabled the public breakthrough. BM25 is a narrow residual hypothesis, not the secret behind `0.96609`.

## 9. Candidate V131+ Architectures

### Candidate A — Global BM25 rank-2 micro-extension (selected)

**Hypothesis and novelty:** BM25's saturation and length normalization provide residual lexical information after V130, while removing 289 sparse pair indicators improves pooling across shifted incumbent→challenger pairs. Train one global exact-utility Ridge on all rank2–4 action rows; deploy only rank-2 actions. V130's 70 changes remain immutable.

**Evidence for:** exploratory next-0.5% gain `+0.00025296`, 5/5 positive, positive user and cluster CIs, robustness across alpha 3–100, all selected OOF actions rank 2, and scale matching the #2 gap. **Evidence against:** this is not promotion-grade Stage-1 nesting; provisional test selections have utility SMD `−0.914`, sparse/E5/Nomic SMDs `−1.07/−1.34/−1.72`, pair TV `0.95`, and 60% unseen pairs. Removing pair identity prevents direct pair-category extrapolation but does not remove numeric covariate shift.

**Views/model/geometry:** existing TF-IDF, E5, Nomic, readiness/state, candidate/incumbent module identity and rank, plus fixed five BM25 streams; standardized Ridge alpha 10; exact ΔNDCG target; champion-aligned candidates; maximum five test rank-2 Top1 score swaps. **Cost/risk/falsifier:** approximately the V130 run plus seconds for BM25, within roughly 1.5–2 hours on Kaggle T4×2 using caches. Main risks are repeated-label selection and train/test shift. Full nested reproduction at fixed 0.5% is the falsifier. **Public scenario:** roughly `0` to `+0.0006`, with `+0.0001–+0.00035` the defensible central range, not a promise.

### Candidate B — Another unchanged V130 utility layer

**Hypothesis/novelty:** none beyond applying the proven scorer to more untouched users. It retains all V130 features, nesting, and geometry and adds 5–20 Top1 actions. **For:** lowest engineering risk and public-positive parent. **Against:** direct residual curve is negative at 0.5% (`−0.00015764`, 0/5) and 1% (`−0.00026332`, 0/5). The reliable head is exhausted. **Falsifier:** already failed. **Decision/public scenario:** reject; expected upside is below Candidate A and label-driven block skipping would be invalid.

### Candidate C — Candidate-only temporal late interaction

**Hypothesis/novelty:** token/message-level interactions capture negation, comparisons, and changing intent missed by concatenated E5/Nomic. Keep exact V130 and add at most five rank-2 Top1 swaps. **For:** genuine representational novelty and literature-supported retrieval mechanism. **Against:** no MineToday incremental result, old broad cross-encoder/dense attempts were weak or mismatched, dependencies and GPU/cache costs rise, and only 4,000 labels are repeatedly reused. **Cheapest falsifier:** frozen candidate-only Top-4 MaxSim/temporal features, fixed folds, error correlation `<0.95`, and incremental `≥+0.00015`. **T4×2:** feasible for 5,000×17 candidate scoring with cached module tokens, but materially more fragile than BM25. **Public scenario:** higher variance, perhaps larger upside if it passes; later branch only.

### Candidate D — Champion-aligned lower-order utility (B23/B34)

**Hypothesis/novelty:** gain remains below Top1 after 70 Top1 edits. It would preserve V130 and make adjacent score swaps without changing membership. **For:** different action family and bounded geometry. **Against:** B23 only `+0.00004572` at 0.5% with CI crossing zero; B34 is negative; B45 is not reconstructible. **Falsifier:** current probes already fail the `+0.00015` relevance threshold. **Cost/public scenario:** cheap but low expected value; reject.

### Candidate E — Refit/replace the full public-positive action set

**Hypothesis/novelty:** aligned modeling may correct earlier V129/V130 mistakes. **For:** a single coherent policy could remove suboptimal legacy actions in principle. **Against:** aligned 5% replacement OOF (`+0.00032878`) is below V130 extension, replacement overlap on frozen V129 analogues contributed negatively, and deployment would risk a proven `+0.00145` package. **Falsifier:** require a fully nested policy-level advantage comfortably beyond extension across random and relevant stresses; current evidence is opposite. **Cost/public scenario:** similar runtime but unacceptable downside; reject.

## 10. Selected Immediate Architecture

Name the next notebook **V131 — Global Champion-Aligned BM25 Rank-2 Micro-Extension**.

### 10.1 Immutable base and data contract

1. Load organizer data only. Preserve the established cleaning, `user_id` entity split, assessment invalid masks, module order, exact NDCG@5, and stable tie-breaking.
2. Load and hash `candidate_v130_DIAGNOSTIC.csv`; require SHA-256 `28dd2121c72e910be8a01b31cd43d022c7fdf080d8e628bd8f30abc3a24bc93e`. Do not use `submission_v130_primary.csv` as the base.
3. Reconstruct the exact V42/V30/V31+B34/B45 policy/champion anchor in every outer fold using only permitted training labels, as V130 did. Reconstruct clean fold-local V129 and V130 selected-user analogues; exclude their union from V131 eligibility. On test, exclude the exact 70 changed V130 users and all historical protected/non-S0 users.
4. V131 is extension-only. If any pre-existing V130 row changes before applying the new policy, fail closed to exact V130.

### 10.2 Representation and features

Reuse the exact V130 preprocessing and pinned E5/Nomic caches. For each user and each policy-anchor candidate at ranks 2, 3, and 4, create a global action row containing:

- candidate and incumbent module one-hot features, plus candidate-rank one-hot;
- candidate value, incumbent value, difference, and normalized ranks for word and character TF-IDF across full, latest, and recency views;
- the equivalent E5 and Nomic value/difference/rank features;
- readiness candidate/incumbent/difference, assessment summary/invalid state, log chat count/no-chat, negation, comparison, latest-vs-full contradiction, and candidate/incumbent topic-presence indicators already available without targets;
- five BM25 view blocks: unigram full/latest/recency and unigram+bigram full/latest, each with candidate value, incumbent value, difference, and normalized ranks.

Do **not** add incumbent→challenger pair one-hot features, target-neighbor/template statistics, pseudo-GT fields, predicted K/IDCG, a new encoder, or an extra score calibrator. The pair removal is deliberate pooling: provisional selected-pair TV is high, while candidate and incumbent marginal identities retain module effects. Keep V130's per-module LightGBM only where needed to reconstruct its policy state and as a pre-registered diagnostic feature ablation; the V131 primary head should match the positive raw global probe rather than silently change its comparator.

### 10.3 Learning and nesting

Use the exact counterfactual change in user NDCG@5 from swapping the anchor Top1 score with the candidate score as the action target. Cross-fit a `StandardScaler → Ridge(alpha=10, solver="lsqr")` global action model. All target-derived anchor, selection, and optional LightGBM values must be generated without the evaluated outer fold's labels. Label-free BM25 and pinned embedding computation must have identical train/test code paths. Train on all rank2–4 rows because that is the tested pooling design; at selection time consider the predicted utility of rank 2 only.

Pre-register one primary random five-fold world and the same exact/latest/near/contiguous diagnostic worlds used by V130. Do not tune alpha, BM25 parameters, included BM25 streams, rank family, or coverage from those outcomes. Report a full-history-only and temporal-only BM25 ablation to explain mechanism, but neither can replace the registered primary after results are seen.

### 10.4 Action policy and artifacts for the later notebook task

Within each validation fold, among residual-eligible users with positive predicted rank-2 utility, select at most four users (0.5% of an 800-user fold). Aggregate exact user deltas. For test, select at most five of the 819 currently residual-eligible users. Apply only Top1↔rank2 score-value swaps; preserve row score multiset, Top-5 membership, rank 5, tail scores, schema, and row order.

The notebook should emit the same compact families V130 used: decision report, fold/world metrics, action ledgers, feature names, package/model versions, cache metadata, and primary/candidate hashes. The report gate, not a filename assumption, must identify whether a submission candidate was created.

### 10.5 Resource plan

The executed V130 five worlds consumed about 5,385 fold-seconds: roughly 18 minutes per five-fold world and 90 minutes total, excluding light setup overhead. BM25 over 17 module documents is negligible beside nested LightGBM and embeddings. With existing 5,000×768 caches, V131 should fit comfortably in T4×2 memory and finish in approximately 1.5–2 hours. No new package beyond the existing scientific stack is required.

## 11. Validation and Submission Decision Policy

### Hard correctness gates

Any failure means **DO NOT SUBMIT** and exact V130 fail-close:

- exact protected-base SHA and schema mismatch;
- target leakage or an outer validation label entering the policy anchor, model, eligibility, target-derived transform, tuning, or action selection;
- pseudo-GT use anywhere in modeling or named-user decisions;
- train/test feature-order or preprocessing mismatch, non-finite features/scores, unpinned model/cache provenance, or nondeterministic ties;
- any mutation of the 70 protected V130 rows before extension, any noneligible edit, more than five new test users, or an action other than Top1↔rank2;
- changed Top-5 membership, score multiset, rank 5, tail, row order, or output bytes that cannot be reproduced.

### Calibrated performance policy

**SUBMIT** one daily slot when the fully nested registered primary has incremental random-world gain at least `+0.00015`, at least 4/5 folds positive, no registered mechanism-relevant stress aggregate below `−0.00050`, and no categorical-support or deterministic parity failure in selected test actions. A bootstrap lower bound may cross zero; report it but do not veto. This rule would have submitted V130 (`+0.000394`, 4/5) rather than repeating the old `+0.00045` threshold mistake.

**BORDERLINE but worth a slot** means gain `+0.00005` to `<+0.00015` with at least 3/5 positive folds, or `≥+0.00015` with only 3/5, provided correctness is perfect, no stress collapse occurs, selected actions remain within marginal module support, and a daily slot remains. At rank #3, best-score retention makes one such package-level measurement rational. It must still be the preregistered package, not the best of many post-hoc variants.

**DO NOT SUBMIT** when gain is below `+0.00005`, at most 2/5 folds are positive, any serious correctness gate fails, a relevant stress is below `−0.00050`, or selected test features show unsupported extrapolation that the global model cannot explain. Keep exact `0.66173`.

Selected-feature shift is evidence, not an automatic scalar veto. Report standardized mean differences, empirical quantile support, candidate/incumbent marginal counts, pair TV for description, utility distribution, chat/no-chat rate, and template support. Because the primary does not use pair identity, unseen pair rate alone is not disqualifying. Conversely, five low-utility test actions with simultaneous extreme sparse/dense/BM25 covariates should trigger fail-close even if OOF is positive.

This policy balances three facts: only about `+0.00020` is needed; V130 proves conservative gates can miss real gains; and repeated reuse of 4,000 labels makes broad adaptive search dangerous. It authorizes one narrow test, not lower standards.

## 12. Expected Gain, Failure Modes, and Public Scenarios

No calibrated probability or exact leaderboard forecast is defensible from three comparable submitted packages. The exploratory `+0.00025296` OOF signal is at the right scale, but V129 and V130 public/OOF ratios span `0.42×` to `2.49×`. Applying those ratios mechanically would produce about `+0.00011` to `+0.00063`; this is a scenario envelope, not a transfer model.

| Scenario | Incremental public outcome | Consequence | Plausible mechanism |
|---|---:|---|---|
| Conservative | `−0.00020` to `+0.00010` | Does not beat #2; Kaggle retains `0.66173` | Test numeric shift, BM25 pilot selection noise, or residual headroom exhausted |
| Realistic | `+0.00010` to `+0.00035` | Upper half can pass `0.66193` | Full-history lexical complement transfers partially, similar to V30/V129-style shrinkage |
| Upside | `+0.00035` to `+0.00065` | Clear #2 improvement | Public subset overrepresents the five rank-2 residual regimes, as V130's subset amplification suggests |

The leading failure mode is **probe optimism**. Many historical questions have been asked of the same labels; the BM25 pilot was selected after observing alternatives. Full V131 must therefore be a fixed confirmatory rerun, not a new search. Second is **test covariate shift**: provisional global-BM25 test actions have lower predicted utility and much weaker dense support than OOF selections. Third is **representation cancellation**: all-stream BM25 could win by unstable coefficient interactions even though temporal-only views were weak. The full-history/temporal ablations diagnose this without changing the registered candidate.

Other risks are incorrect base-file selection due the V130 filename discrepancy, accidental reuse of saved V130 scores in a non-nested calibrator, pair-specific overfit, mutation of public-positive rows, tie instability, cache revision drift, and fitting test-user choices from v3 pseudo-GT. The blueprint directly gates each.

The downside of a correct but weaker Kaggle submission is one daily slot, not loss of the retained best public score. The downside of a broad replacement is much larger scientific and private-score uncertainty, which is why V131 stays at five score swaps. The package is optimized for the probability of a modest `+0.0002–+0.0005`, not an attempt to explain or reach `0.96609`.

## 13. V131+ Conditional Roadmap

1. **Immediate V131:** implement exactly the global BM25 rank-2 micro-extension in Section 10, run the fixed validation package, audit bytes/geometry, and apply Section 11. Do not create parallel checkpoint, alpha, coverage, or rank-family variants.
2. **Immediate fallback:** none qualifies. If V131 fails, preserve V130 and save the submission slot. The unchanged-head, B23/B34, shape, and replacement probes are already too weak.
3. **If V131 is locally positive but public-flat/negative:** close BM25 for this residual layer. Run one candidate-only temporal/late-interaction falsifier using a pinned multilingual model and existing Top-4 opportunities. Advance only with residual correlation `<0.95`, fixed-coverage gain `≥+0.00015`, and support-safe test geometry.
4. **If V131 is public-positive:** freeze its new actions and repeat the residual-curve audit before any extension. Do not assume another 0.5% remains. The next branch should be whichever genuinely reorders untouched actions—likely temporal late interaction—not a fourth mechanical layer.
5. **If new organizer/private evidence changes the protected champion:** update only the exact base and recompute fold-local analogue policies; never infer named action correctness from score differences.

No new embedding family, supervised metric-learning run, rank5/6 membership edit, or full-list reranker should be opened without a cheap, V130/V131-aligned residual falsifier. This keeps the roadmap conditional rather than accumulating speculative features.

## 14. Final Recommendation

V130's `0.66173` result is real package-level evidence that champion-aligned selective utility was undervalued by the old gate. It does not reveal hidden test labels, make every V130 action correct, or license ignoring validation. It changes the decision rule: correctness remains absolute; random-world signal, fold direction, action support, and package history jointly determine whether a small candidate deserves one leaderboard slot; grouped stresses and CI bounds are context rather than universal vetoes.

The exact next action for the notebook-building session is:

> Build **V131 Global Champion-Aligned BM25 Rank-2 Micro-Extension** from `candidate_v130_DIAGNOSTIC.csv` SHA `28dd2121…`; freeze its 70 public-positive Top1 edits; add the five fixed V128 BM25 streams to a fully cross-fit global raw-feature exact-ΔNDCG Ridge with no pair one-hot; train on rank2–4 opportunities but deploy at most five positive rank-2 swaps; then submit only under Section 11.

This recommendation is narrow because the evidence is narrow. The existing V130 scorer, relevance-shape features, lower-order actions, and replacement policy failed their cheapest falsifiers. New embeddings lack residual diversity evidence. BM25 is the one reopened mechanism with exploratory gain `+0.00025296`, five positive folds, positive uncertainty bounds, stable regularization direction, negligible runtime cost, and the correct scale to challenge `0.66193` while leaving `0.66173` intact.
