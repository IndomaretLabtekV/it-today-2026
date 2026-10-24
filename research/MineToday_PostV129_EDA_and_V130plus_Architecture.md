# MineToday — Post-V129 Public Update, Targeted EDA, and V130+ Architecture

## 1. Executive Decision

The scientific state has changed materially. **FACT:** the exact V129 candidate with SHA256 `34f334429c2113ba5b96283a7974fad472068ea08047ba3275fc44057e3a95df` scored the user-reported public NDCG@5 of **0.66075**, improving the protected `0.66028` package by **+0.00047**. This is the first direct public evidence that the lean sparse × dense × assessment, selective-Top1 branch transfers. It does not prove that every V129 action helped, that its OOF estimate was unbiased, or that 42% is a reusable transfer coefficient.

At the user-supplied approximate rank **#8**, and assuming the last reported Top-5 cutoff remains `0.66093`, the remaining gap is only **+0.00018**. The next objective should therefore be a controlled incremental improvement over the exact V129 bytes, not a broad replacement seeking an impressive but fragile offline delta.

**Selected immediate direction — V130: Champion-Aligned Incremental Utility Extension.** Freeze the 50 public-positive V129 Top1 changes and all historical protected/non-S0 rows. Reuse V129's lean TF-IDF, E5, Nomic, assessments, per-module LightGBM, and direct exact-ΔNDCG Ridge. Correct the important train/deployment mismatch by generating action candidates from a fully nested, policy-matched `0.66028` champion analogue in OOF, while deployment uses the exact current V129 ranking. Evaluate only users not selected by the clean V129 OOF policy, and add at most **2% of all test users (20 users)** beyond V129. Existing V129 actions remain bit-exact.

This choice follows three findings:

- **FACT:** V129's second-level fitting is genuinely nested, but OOF action candidates come from its sparse Ridge anchor whereas test candidates come from the historical champion. A fresh reconstruction finds only `78.3%` Top1 agreement and `0.759` mean Top-4 set Jaccard between the full-fit sparse test anchor and the deployed champion.
- **FACT:** V129 OOF selects from all 4,000 users, while test selection is restricted to 889 eligible rows; 50 actions therefore cover `5.62%` of eligible test rows. Its configured `0.816` seen-template mixture weight is also stale for the implemented skeleton: direct reconstruction against the saved train manifest gives `55.6%` test-seen coverage.
- **SUPPORTED INFERENCE:** these parity defects, plus a large selected pair shift, are better explanations for partial transfer than insufficient encoder capacity. A cheap matched probe found that E5 and Nomic jointly add some action information, but their action-error correlation is `0.998`; adding another encoder before fixing policy geometry is poor experimental ordering.

V129 establishes the branch, not the accuracy of its absolute `+0.001130` deployment forecast. V130's purpose is to obtain the small additional `+0.0002–0.0005` public-scale opportunity needed for Top 5 while preserving the only newly public-validated package. No V130 notebook or submission is built in this research task.

## 2. Evidence Update Since the Previous Research

V128 directly implemented the earlier report's hybrid proposal: word/character TF-IDF, true BM25, E5/Nomic similarities, assessments/readiness, target-derived neighbor/template statistics, per-module LightGBM, and a Ridge action utility head. Its exact output report selected `no_bm25` at nominal `0.5%` coverage, failed every downstream promotion gate, and copied the old champion bytes. **FACT:** V128 was not a new submission.

The high-coverage DESIGN artifact is nonetheless informative:

| V128 variant | 5% gain and action behavior | 6% gain and action behavior |
|---|---:|---:|
| full | `+0.001198`; 163 actions, 94 benefit / 63 harm / 6 ties; user CI `[+0.000334,+0.002091]` | `+0.001085`; 195 actions, 106 / 80 / 9 |
| no BM25 | `+0.001257`; 163 actions, 94 / 63 / 6; user CI `[+0.000397,+0.002128]` | `+0.001362`; 195 actions, 106 / 79 / 10; user CI `[+0.000448,+0.002300]` |
| no target-neighbor | `+0.001290`; 163 actions, 92 / 63 / 8; user CI `[+0.000390,+0.002178]` | `+0.001591`; 195 actions, 111 / 72 / 12; user CI `[+0.000625,+0.002517]` |
| dense-only control | `+0.001154`; 163 actions, 88 / 68 / 7 | `+0.001272`; 195 actions, 107 / 78 / 10; user CI `[+0.000379,+0.002159]` |

These numbers come from `v128_risk_coverage_design.csv`. High-coverage fold vectors were not persisted for every row, so they cannot be reconstructed exactly without rerunning; the chosen full-5% and dense-6% policies did have positive worst folds (`+0.000204` and `+0.000372`). Cluster CIs show the same direction as user CIs.

Three V128 issues must not be conflated. Requiring `benefit >= 2 × harm` was a **methodological design choice**, but a poor one because NDCG is magnitude-weighted. The branch hand-off was an **implementation/decision bug**: BM25 was rejected by comparing full and no-BM25 at full's 5% coverage, after which `SELECTED=no_bm25` silently inherited its separately chosen `0.5%` policy. Finally, `action_oof()` was not truly second-level nested: meta-training rows used Stage-1 OOF features from models that could include the current meta-validation fold's labels. That is a **methodological implementation defect**. Consequently, V128 LOCK validly rejects its tiny four-action `0.5%` lock policy (`-0.000114`), not the high-coverage lean core.

V129 intentionally removed BM25 and target-neighbor features, fixed coverage at 5%, and rebuilt each outer fold's action-training rows through inner Stage-1 fits. Direct source confirms that outer validation labels remain untouched until both Stage-1 and action predictions exist. The feature contract contains 64 raw candidate features and 394 action features; 17 module-specific LightGBMs average three fixed seeds, and the action learner is standardized Ridge with `alpha=10`.

The executed V129 evidence is strong internally:

| World | fixed-5% gain | positive folds | worst fold |
|---|---:|---:|---:|
| random primary | `+0.00113019` | 5/5 | `+0.0000256` |
| exact full-history grouped | `+0.00064544` | 3/5 | `-0.0004539` |
| latest-skeleton grouped | `+0.00082029` | 5/5 | `+0.0000378` |
| near-template grouped | `+0.00115776` | 5/5 | `+0.0003072` |
| contiguous ID | `+0.00102368` | 4/5 | `-0.0001628` |

The random-world user CI is `[+0.000364,+0.001884]`, near-cluster CI `[+0.000316,+0.001990]`, with 112 beneficial and 78 harmful actions among 200. The test package changes exactly 50 Top1 positions, preserves Top-5 membership, rank 5, and each row's score multiset, and is byte-distinct from `443785...57d`. Those are all **FACTS** from direct artifacts.

However, the claimed `+0.001130` is an honest estimate for the **sparse-anchor action problem**, not a fully faithful estimate for the deployed champion-relative action problem. This distinction weakens the absolute OOF forecast but does not retract the public result. Prior conclusions strengthened by V129 are: candidate-specific nonlinear fusion works; sparse and dense evidence can transfer jointly; direct Top1 utility is viable; BM25 and target-neighbor statistics are not necessary. Weakened conclusions are: strict prospective locks and repeated uncertainty subtraction are always decision-useful; 5% action policies are too aggressive; and every grouped-fold reversal should veto. Overturned is the practical belief that another fail-close at `0.66028` was safer than one well-formed competition-aligned submission.

## 3. Why V129 Transferred Only Partially

The public gain is `0.00047 / 0.00113019 = 41.59%` of primary OOF. **FACT:** this is one package-level observation. Hidden test labels are unavailable, so action-level public attribution, a universal transfer coefficient, and the claim that particular pairs helped are **NOT ESTABLISHED**.

The most concrete explanation is policy-anchor mismatch. OOF action rows use `stage1_anchor` for incumbent, ranks 2–4, margin, and exact swap labels. At test time, `predict_action()` instead receives `CHAMPION_SCORES` as that anchor, while the raw Stage-1 features still contain sparse-anchor geometry. Thus one feature contract has two meanings at deployment. The OOF sparse anchor scores `0.653924`; Stage-1 fusion scores `0.661244`, showing that the nonlinear representation is useful but also that the sparse comparator is materially weaker than the production lineage.

A fresh label-free reconstruction fitted V129's sparse anchor on all 4,000 users and compared it to the exact deployed champion on test. Top1 agrees for only `783/1000`; mean Top-4 set Jaccard is `0.759`, flat rank correlation `0.863`. Marginal Top1 module frequencies are similar (total variation `0.054`), so this is not a simple global module-prior shift: it is row-level candidate-set and pair identity drift.

That drift appears directly in selected actions. OOF and test candidate ranks are close—OOF `176/20/4` at ranks 2/3/4, test `42/5/3`, rank-distribution TV `0.04`—but incumbent TV is `0.30`, challenger TV `0.36`, and incumbent→challenger pair TV is `0.715`. The 200 OOF actions span 83 pairs; the 50 test actions span 34, and their leading pairs are different. Some of this TV is expected sampling noise in sparse categories, but its scale plus the anchor substitution establishes material geometry shift.

Utility calibration also shifts. Among OOF's selected 200, mean predicted utility is `0.04972`, while mean realized exact utility conditional on selection is `0.00113019 / 0.05 = 0.02260`. In the highest all-user utility decile, predicted mean is `0.03511` and realized mean `0.01535`. The selector orders actions usefully—best-score versus realized-delta Spearman is `0.332`—but overstates magnitude. Test-selected mean predicted utility rises to `0.06559`; versus OOF-selected values, standardized mean difference is `+0.716` and KS distance `0.56`. **SUPPORTED INFERENCE:** the test head is more confident partly because champion-relative feature values are off its training support; this can produce positive ranking transfer with weaker magnitude transfer.

Eligibility is another mismatch. OOF applies the 5% cap to every user. Deployment permits 889 S0/unprotected users and chooses 50, or `5.62%` of the eligible population. This changes the accepted tail even if score calibration were identical. V130 must create fold-specific S0 eligibility and apply the same total-population cap.

The configured benchmark mixture is not faithful to V129's implemented skeleton. `seen_weight=0.816` was copied from the previous report. Reconstructing V129's exact normalizer/skeleton and asserting equality to `v129_template_manifest.csv` gives only `55.6%` of test latest skeletons seen in train. Test exact-latest coverage is `48.2%`, exact-full `27.4%`, exact full-skeleton `28.3%`, and full-skeleton nearest-train cosine at least 0.90 is `61.9%`. Selected test actions are somewhat more latest-skeleton-seen (`62%`) than all test, versus `39%` fold-local seen among selected OOF actions. The reported mix remains positive because seen/unseen gains are similar, but calling it benchmark-faithful is **INVALID / RETRACTED**.

Assessment state supplies a plausible secondary mechanism. OOF-selected challengers have mean readiness minus incumbent readiness `+0.139`; test-selected actions have `-0.285` (SMD `-0.435`). Lower-readiness challenger actions are 24.5% of OOF selection but 42% of test selection. On real OOF labels, the 49 lower-readiness actions contribute `-0.000091` overall, while 71 higher-readiness actions contribute `+0.000632`. The lower-readiness sign is only negative in two folds, so a hard filter would be post-hoc overreach; an honestly learned interaction is justified.

What remains unidentified is substantial: public correctness of the 50 actions, private-score transfer, whether readiness or pair drift caused the missing gain, and whether public rounding masks a small additional change. The defensible conclusion is that V129 transferred despite measurable deployment mismatch. That points first to alignment, not to a larger model.

## 4. Pseudo-GT Diagnostic Findings

The prompt expected an extracted directory, but the repository contains only `research/MineToday_pseudo_GT_v2_after_V129.zip`. It was inspected under `/tmp`; no historical files were modified. Its README is explicit: v2 combines organizer target geometry, earlier V49-derived pseudo ordering, and V129's selected Top1 actions. It preserves lower-order v1 ordering, promotes each V129 challenger to pseudo rank 1, and deliberately leaves K/jitter unchanged.

The pack is therefore posterior bookkeeping, not hidden truth. Of 50 selected V129 rows, 45 overwrite pseudo-v1 Top1 and five already agreed. Evidence tiers are 33 `MEDIUM_HIGH_RELATIVE`, 12 `HIGH_RELATIVE`, and five `VERY_HIGH_RELATIVE`; these tiers are derived from V129 score percentiles. All selected rows have pseudo K=5 by construction. Candidate ranks under the real champion are 42 rank-2, five rank-3, and three rank-4. These are **pseudo-GT observations**, but none independently validates an action because V129 itself generated the update.

The only safe hypothesis generated was: V129 acts in ambiguous regimes where relevance-shape, readiness, and anchor state may determine whether a seemingly strong promotion is worthwhile. That hypothesis was returned to real train labels:

| Train-validated selected slice | actions | contribution to mean NDCG | interpretation |
|---|---:|---:|---|
| true K=4 | 62 | `+0.000659` | strong magnitude, 39 benefit / 19 harm |
| true K=5 | 70 | `+0.000049` | nearly flat, 32 / 35 |
| true K=6 | 68 | `+0.000422` | positive, 41 / 24 |
| lower half anchor margin | 100 | `+0.000386` | positive but weaker |
| upper half anchor margin | 100 | `+0.000744` | larger corrections are not confined to close ties |
| readiness challenger lower by >0.25 | 49 | `-0.000091` | concerning but fold-unstable |

Thus K/relevance shape and readiness interact with utility on true labels. Test K is unobserved, so neither can become a test rule. A future action model may use cross-fitted predicted IDCG/K-shape and readiness interaction features, but only if they improve real nested OOF. The pseudo pack supplied no independent confidence label and cannot justify selecting its top 30, excluding its bottom 20, estimating hidden NDCG, or changing any named user. Those uses remain prohibited.

## 5. Targeted Fresh EDA After V129

Coverage is surprisingly flat across the useful range. Random-world gains at 3/4/5/6% are `+0.001110/+0.001172/+0.001130/+0.001262`. Averaged descriptively across all five worlds, they are approximately `+0.000694/+0.000920/+0.000956/+0.000949`. Five percent is therefore a reasonable fixed baseline; six percent is not a robust winner, and selecting among many thresholds would replay V128's optimism. For V130, the existing 5% package is frozen and the new layer receives a single preregistered **2% incremental cap**, chosen from the current small leaderboard gap and the need to limit risk, not from a new coverage sweep.

The direct V129 utility ranking is useful but imperfect. At 5%, 112 actions help, 78 harm, and ten tie; the positive mean comes from magnitudes, demonstrating why V128's 2:1 count rule was wrong. The top predicted-utility decile is the only decile with positive mean realized utility. A fixed count is safer than a raw score threshold because deployment score scale is shifted.

Pair heterogeneity is real but too sparse for independent pair models. The 200 selections occupy 83 pairs; the five most common pairs contain only `20.5%`, and only 11 pairs have at least five selected actions. Examples show both useful and unstable effects: M012→M007 has five benefits and no harms (`+0.000134` overall), while M003→M001 has eight selections but contributes only `+0.000004`. Challenger M007 contributes `+0.000369` across 15 actions and five positive folds; challengers M017 and M002 contribute `-0.000212` and `-0.000143`, but supports are 16 and 13. Test challengers concentrate on M009 (14), M017 (7), and M012 (7). This supports global pooling with module interactions, not post-hoc pair blacklists.

Template/state signal remains residual rather than deterministic. Fold-local latest-seen and unseen selected actions both contribute positively (`+0.000574` from 78 seen; `+0.000556` from 122 unseen). Seventy-seven actions where latest and full lexical Top1 disagree contribute `+0.000532` and are positive in all five folds. Comparison-language rows and negation/cancellation rows are also positive, but the latter covers 165/200 and is too broad to define authority. These findings support retaining latest/full/recency views; they do not support target-template lookup.

Dense agreement is not a sufficient selector. Both E5 and Nomic favor the challenger in 112 OOF selections, contributing `+0.000602`, but the slice is positive in only three folds. Forty-seven dense-disagreement selections are still positive overall. Test's both-dense-favor rate is 50% versus 56% OOF. A unanimity rule would discard real gain.

Two exploratory falsification probes were run with fixed V129 folds and fixed 5% coverage. They are **not promotion-grade** because the complete second-level Stage-1 stack was not rerun within each comparison. A Ridge using pair+sparse raw features gains `+0.000391`; adding E5 alone gives `+0.000375`, Nomic alone `+0.000332`, both `+0.000558`, and adding the existing LightGBM fusion raises it to `+0.001033`. This reproduces the direction of V129 cheaply. A second calibrator fitted around V129's utility score makes results worse: score-only `+0.001084`, global context `+0.000609`, pair context `+0.000413`, pair+assessment/state `+0.000519`. This falsifies an immediate extra calibration layer and warns against sparse pair coefficients.

The main fresh EDA decision is therefore simple: keep the coverage-ranking mechanism and representation; align the action problem and restrict the next package to untouched users. Treat readiness, predicted relevance shape, and pair support as logged diagnostics until a fully nested incremental ablation earns their use.

## 6. Representation and Embedding Audit Through V129

The evidence now separates raw representation from action information:

| Component | Direct or fresh evidence | Decision |
|---|---|---|
| sparse word/char, latest/full/recency | durable historical backbone; exploratory exact-action gain `+0.000391` over negative geometry-only probes | keep |
| E5 alone after sparse | exploratory `+0.000375`, slightly below sparse-only | not independently proven |
| Nomic alone after sparse | exploratory `+0.000332` | not independently proven |
| E5 + Nomic after sparse | `+0.000558`, about `+0.000168` over sparse in the cheap matched probe | keep pair for controlled continuity |
| assessments/readiness | strong relevance value historically, but naive direct addition reduces the cheap action probe; readiness slices remain heterogeneous | keep upstream in LightGBM; do not add a hard rule |
| champion/sparse anchor geometry | necessary for action definition; geometry-only probes are negative | use as context, never sole authority |
| per-module LightGBM fusion | raises cheap action probe from `+0.000092` with raw state to `+0.001033`; full V129 Stage-1 beats sparse anchor by `+0.007320` OOF | main information concentrator |

E5 and Nomic are complementary enough to retain but not diverse enough to justify checkpoint expansion now. Their raw candidate-minus-incumbent signals correlate `0.737` and agree in sign `81.7%`. More importantly, sparse+E5 and sparse+Nomic action-probe residual errors correlate `0.998`. That is evidence of nearly identical remaining errors under this use, even though the pair's combined ranking is modestly better.

The official E5 card describes a 12-layer, 768-dimensional, roughly 0.3B-parameter retrieval model spanning 94 languages, MIT licensed, with 512-token truncation; it is a natural Indonesian-capable query/document encoder. The official Nomic v2 card describes a 475M-total/305M-active multilingual MoE, 768 dimensions, 512-token maximum, Indonesian among roughly 100 languages, Apache-2.0 licensed. V129 explicitly truncates Nomic to 256 tokens and uses remote model code. See the [multilingual-E5 model card](https://huggingface.co/intfloat/multilingual-e5-base) and [Nomic v2 model card](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe).

V129 caches are practical but not immutable: metadata records text/model/prefix keys and shapes, but no model revision, tokenizer/config hashes, or remote-code hash. V130 should pin exact revisions and validate a small re-encode cosine probe; this is a reproducibility correction, not a representation change. Both cached matrices are 5,000×768 and fit easily on T4×2; no new encoding should be necessary if provenance is accepted.

Prior versions already tested BGE-M3, GTE, Qwen, IndoBERT, XLM-R, cross-encoders, and standalone dense replacements, generally weakly. Those tests do not close conditional late interaction, but they make a new checkpoint a low-priority bet. A new family should be reopened only after a cheap champion-aligned action probe shows residual-error correlation below `0.95` and paired incremental gain at least about `+0.00015` at matched coverage. Until then, **recommend no new embedding**. The bottleneck is action-policy parity and marginal authority, not missing semantic breadth.

## 7. Reassessment of the 0.96609 Methodological Clue

V128 is the first direct test of the broad “embedding + TF-IDF + BM25 + LightGBM/ensemble” clue. It confirms that such components can form a positive action representation, but it does not identify BM25 as the useful component. At 5%, no-BM25 beats full by `+0.000059`; at 6%, by `+0.000278`. Removing target-derived neighbors beats full by `+0.000093` and `+0.000507`. Dense-only remains within about `0.00010` of no-BM25 at both coverages. Because V128's second-level evaluation is imperfect, these are directional ablations, not promotion-grade causal effects.

V129 provides the stronger test: a corrected nested lean system without BM25 or target neighbors is positive in five validation worlds and on the public leaderboard. **SUPPORTED INFERENCE:** the transferable mechanism is the common core—sparse lexical views, dual dense similarities, readiness, module identity, anchor geometry, nonlinear per-module fusion, and selective direct utility—rather than BM25 or neighbor lookup.

What remains scientifically unresolved is narrow: BM25 might add conditional value under a truly champion-aligned nested action problem, or a different short-query tokenization might help a rare lexical slice. That is not worth the next submission slot because its best direct ablation is null-to-negative and the current parity defect is larger. BM25's term saturation/length normalization remains methodologically distinct from TF-IDF, as summarized by [Robertson and Zaragoza](https://www.nowpublishers.com/article/DownloadEBook/INR-019), but distinct does not mean incrementally useful here.

The `0.96609` score remains an outlier and a generic clue only. Visible template entropy and weak raw retrieval prohibit inferring near-deterministic hidden labels. No evidence supports leaderboard inversion, misconduct claims, private implementation assumptions, or optimizing toward that score.

## 8. Candidate V130+ Architectures

| Candidate | Architecture and hypothesis | Evidence, risk, falsifier, and scenario upside |
|---|---|---|
| **A. Champion-Aligned Incremental Utility Extension — selected** | Freeze exact V129 bytes and its 50 actions. On clean nested OOF, reconstruct policy-matched `0.66028` anchor/S0 state, retain V129 lean features and models, exclude V129-OOF-selected users, and rank champion-relative Top1 swaps for the remaining users. Add a fixed 2% cap (20 test users). | Strongest support: public-positive V129 representation plus measured anchor/pair/eligibility mismatch. Against: OOF baseline-selected identities cannot exactly reveal which public V129 actions helped. Cheapest falsifier: one fully nested random world; reject if incremental gain is nonpositive or selected pair shift remains worse than V129. T4×2 feasible with cached embeddings. Scenario public upside `+0.0001–+0.0005`; exact 20-user Top1-only geometry. |
| **B. Aligned 5% replacement policy** | Recreate candidates from the policy reconstruction and replace V129's 50-action set with a newly aligned 5% set from the same Ridge. It directly tests whether mismatch explains missing transfer. | Potentially larger correction, but it gives up aggregate-positive public actions without action labels. Run as a diagnostic output only. Falsified unless aligned OOF beats the surrogate policy materially, is stable, and overlaps strongly with existing test actions for label-free reasons. Expected 30–70 changed action identities versus V129; public range is broad and includes losing `0.00047`. Not a first submission. |
| **C. Hierarchical module/readiness utility model** | Global Ridge plus cross-fitted, empirical-Bayes-shrunk challenger/incumbent/readiness/IDCG residuals; no independent per-pair estimators. Intended to correct M017/M002 and lower-readiness magnitude errors without blacklists. | Support: module/readiness heterogeneity and sparse pair support. Against: cheap V129-score calibration probes fall to `+0.000413–+0.000519`, far below direct V129. Cheapest falsifier is a fully nested matched A/B on Candidate A's aligned rows; require positive incremental gain in at least four folds. Feasible CPU-only. Scenario upside `0–+0.0003`; currently deferred. |
| **D. Temporal/late-interaction representation** | Keep the same checkpoints but represent each message separately; add candidate-specific latest/full disagreement, max/recency pooling, and learned temporal aggregation to LightGBM. A more expensive variant would add token-level or cross-encoder scores only for champion Top-4. | Support: latest/full contradiction is positive in all five OOF folds and history cannot be reduced to the last message. Against: V129 already uses latest/full/recency sparse views, dense errors are highly correlated, and historical cross-encoder/standalone semantic attempts were weak. Cheapest falsifier: cached message-level cosine features in one fixed nested world, requiring `+0.00015` over Candidate A. T4×2 feasible but dependency/runtime risk rises. Scenario upside `0–+0.0004`; not immediate. |

Candidate A has the highest decision value because it changes one known defect while protecting the public-positive package. Candidate B is scientifically useful but strategically asymmetric. Candidate C is contradicted by the cheap probe. Candidate D is legitimate research, but not before alignment. There is **no qualified immediate fallback submission**; subsequent work is conditional on V130 evidence.

## 9. Selected Immediate Architecture

V130 should start from exact `submission_v129_primary.csv`, not the old champion. Its deployment contract is “V129 plus at most 20 new Top1 swaps on previously unchanged, eligible users.” The 50 V129-changed rows, all historical non-S0 rows, and V25-protected users are immutable.

Maintain two explicitly named anchors. `retrieval_anchor` is V129's sparse word+char+assessment Ridge and is used only as a Stage-1 feature. `policy_anchor` defines incumbents, ranks 2–4, margins, exact action labels, and output swaps. In validation it is reconstructed with the existing V42 `build_policy_anchor_split` lineage and B34+B45 state machine; at deployment, remaining eligible/unchanged V129 rows are identical to the old champion and therefore their candidates come directly from current V129 scores. Never pass these anchors through the same unnamed argument.

Use V129's data views unchanged: normalized latest text, chronological full history, message-level recency; 10 parsed assessment values and invalid masks; 17 organizer-only module documents. Retain word TF-IDF `(1,2)` and char-wb `(3,5)` latest/full/recency scores, E5/Nomic module similarities, readiness, chat/no-chat, retrieval-anchor geometry, and the 17 fixed LightGBMs. Do not reintroduce BM25, target-neighbor statistics, a new encoder, or a hyperparameter search.

The validation nesting is:

1. In each outer fold, reconstruct the policy-matched historical champion and S0 state using outer-training labels only.
2. Inside that outer-training set, use three inner folds to generate both policy-anchor candidates and Stage-1 fusion features for action-head training rows. Each inner validation row receives a reconstructed policy anchor from inner-training only.
3. Fit the same standardized Ridge (`alpha=10`) to exact per-user ΔNDCG for policy-anchor Top1 swaps with ranks 2–4.
4. Predict outer validation without its labels. Exclude non-S0 rows and users selected by the existing clean V129 OOF policy. Among remaining positive scores, take a fixed `ceil(0.02 × n_outer_population)` cap. Evaluate incremental delta; existing baseline actions occur on disjoint users and therefore do not interact.
5. Repeat random, exact-history, latest-skeleton, near-template, and contiguous diagnostics. Only the random world is the primary estimator; others diagnose transfer.
6. For final fit, train the action model on clean outer-OOF action features/labels. Score test relative to current V129 ranks, exclude its 50 changed rows and every protected row, and add at most 20 swaps.

Store the full action opportunity ledger, not only selected rows: user, folds, eligibility, baseline-selected flag, incumbent/challenger/rank, both anchors and margins, fusion scores, predicted/realized utility, readiness difference, template regime, and selection. Log action-pair overlap/support and OOF/test selected-feature shift. This makes the hypothesis auditable without pseudo labels.

Output geometry is exact: start from V129 bytes; for each of at most 20 rows swap the existing Top1 score value with one current rank-2/3/4 value; preserve Top-5 membership, rank 5, tail order, and row score multiset. The current 50 V129 rows must be bit-identical. If no action has positive utility, output V129 unchanged.

Resource use remains modest. Reuse the 5,000×768 caches after verifying ordered-text hash, prefix, pooling, dtype, tokenizer/config/model revision, and probe cosine. Sparse matrices and 17 LightGBMs run on CPU; E5/Nomic need no fresh encoding. Reconstructing the V42 anchor is the main extra cost, but its code and invariants already exist. This reuse is safer than inventing a new anchor surrogate.

## 10. Validation and Submission Decision Policy

Hard correctness gates are non-negotiable:

- exact organizer data hashes, 4,000/1,000 users, 17 modules, and sample schema/order;
- exact NDCG@5 implementation with stable ties and counterfactual unit tests;
- complete outer/inner isolation for policy anchor, Stage-1, action labels, transformations, and any target-derived state;
- identical OOF/test feature definitions, explicit retrieval-versus-policy anchor roles, and fold-specific S0 eligibility;
- no self-neighbor path, pseudo-GT input, external labels, or test-user tuning;
- immutable E5/Nomic revisions and cache provenance; deterministic seeds and finite features;
- exact V129 base SHA, current 50 actions and protected rows bit-identical;
- at most 20 new rows, Top1-only rank-2/3/4 swaps, unchanged membership/rank5/tail, preserved score multisets;
- clean offline rerun reproducing output bytes and logged provenance.

Performance evidence is competition-aligned. The primary quantity is fully nested, champion-aligned **incremental** random-world ΔNDCG at the fixed 2% cap. Report five fold deltas, user and near-cluster bootstrap intervals, exact/latest/near/contiguous pooled results, selected utility calibration, module/pair supports, and label-free OOF/test shift. Confidence intervals are context, not automatic vetoes. One mildly negative fold or stress world is allowed; benefit:harm counts never override magnitude.

| Decision | Evidence policy |
|---|---|
| **SUBMIT** | All correctness gates pass; primary incremental gain is at least `+0.00045`; at least 4/5 random folds are positive; no relevant stress aggregate is catastrophic (`< -0.001`); and test-selected actions do not show a new unexplained support failure materially worse than V129. This threshold maps the one observed 41.6% transfer to roughly the `+0.00018` current gap, but the ratio is used only for scenario planning. |
| **BORDERLINE** | Correctness passes and gain is `+0.00025–+0.00045`, or only 3/5 folds are positive, while stress losses remain mild. A single daily slot may be rational near deadline if the Top-5 cutoff is still close and no stronger candidate exists; record explicitly that expected gain may not clear the gap. |
| **DO NOT SUBMIT** | Any correctness/parity failure; nonpositive primary gain; two or fewer positive random folds; catastrophic relevant stress; or an unmodeled selected-action shift that recreates V129's anchor/pair extrapolation. Copy exact V129 bytes. |

The `+0.00045` threshold is not a paper-significance claim or old `+0.002` floor. It encodes current decision utility: false rejection may miss Top 5, while a false positive costs one of three daily slots but leaves the proven V129 package available. Do not subtract CI width and selection optimism repeatedly into `G_safe`. Do not require every module block to be positive; unsupported blocks should shrink and be reported.

Selective prediction literature frames coverage as a risk trade-off rather than a confidence-count purity rule; [SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html) is a general reference, not a proposed deep model here. Because repeated model selection can bias even cross-validation, the architecture, 2% cap, and decision bands must be frozen before the new OOF run; see [Cawley and Talbot](https://www.jmlr.org/papers/volume11/cawley10a.html). One fixed comparison is preferable to another threshold hunt.

## 11. Expected Gain, Failure Modes, and Public-Transfer Scenarios

These are scenarios, not forecasts:

| Scenario | Fully nested incremental evidence | Possible public outcome from 0.66075 | Interpretation |
|---|---|---|---|
| fail-close | `<=0` or correctness failure | no new submission; remains `0.66075` | Alignment finds no residual authority; preserve rank #8 package. |
| conservative | `+0.00025–+0.00050` | roughly `-0.00005–+0.00020` if submitted | May tie or narrowly clear a static cutoff, but transfer noise dominates. |
| realistic success | `+0.00050–+0.00100`, stable direction | approximately `+0.00018–+0.00050`, yielding `0.66093–0.66125` | Consistent with correcting a material mismatch while touching only 20 new users. |
| upside | `+0.00100–+0.00150` with supported test geometry | about `+0.00040–+0.00080`, yielding `0.66115–0.66155` | Alignment exposes high-value untouched actions; still far from explaining `0.96609`. |

Primary failure modes are: the V42 champion analogue no longer matches the current residual role; the 20 marginal actions are weaker than V129's top 50; baseline-selected OOF exclusions differ structurally from actual V129 test exclusions; pair/readiness shift persists; fixed historical labels make model-selection optimism larger than bootstrap uncertainty; and cached remote code drifts. The main strategic risk is false confidence from a high OOF utility scale—the test score distribution already shifted by `+0.716` SMD in V129.

There is also a ceiling. Freezing V129's 50 actions prevents correcting any harmful member of that set, but public aggregate evidence makes that deliberate conservatism rational for the next slot. If V130 cannot find residual value, a future diagnostic replacement may be scientifically warranted; it should not be smuggled into V130.

## 12. V130+ Conditional Roadmap

| Step | Condition | Single change | Stop condition |
|---|---|---|---|
| **V130 now** | unconditional research priority | Build the champion-aligned, fixed-2% incremental extension over immutable V129 exactly as specified. | Stop/copy V129 on any correctness failure or nonpositive primary gain. |
| **V131 conditional fallback** | only if V130 is positive but readiness/IDCG calibration remains a reproducible error in at least four folds | Add one globally shrunk readiness × predicted-relevance-shape residual to the same aligned action head; no pair blacklists and no new embeddings. | Stop if matched incremental gain over V130 is `<=+0.00015`, fold direction is unstable, or the cheap calibration failure repeats. |
| later representation branch | only if V130 is parity-clean but flat and an existing cached representation has action-error correlation `<0.95` versus E5/Nomic | One temporal message-level or late-interaction feature family, tested at identical coverage. | Stop after one fair negative probe; do not checkpoint-shop. |
| replacement/lower-order branch | only if aligned 5% diagnostic materially dominates and residual extension has no headroom | Audit replacement separately; B23/B34 or broad fusion requires new matched evidence. | Do not submit if it must unprotect V129 without a large, stable advantage. |

V131 is the only named fallback, and it is not yet qualified. BM25, target-neighbor lookup, a new dense ensemble, lower-order edits, and broad whole-list residuals remain deferred. The roadmap branches on new evidence instead of assigning versions to speculative ideas.

## 13. Final Recommendation

**FACT:** V129's `+0.00047` public gain establishes a transferable lean sparse × dense selective-Top1 branch. **FACT:** its absolute OOF estimate is not deployment-faithful because action candidates, eligibility, and seen-mixture weighting differ between OOF and test. **SUPPORTED INFERENCE:** correcting those mismatches is more likely to yield the small remaining Top-5 gain than adding BM25, neighbors, another encoder, pair-specific rules, or a broad residual.

The next notebook-building session should implement **V130 Champion-Aligned Incremental Utility Extension**: exact V129 bytes as base; its 50 actions immutable; V42 policy-matched anchor and S0 reconstruction inside every inner/outer fold; unchanged V129 representations and models; fixed 2% additional coverage; exact Top1 score swaps on at most 20 previously unchanged users; and the Section 10 decision policy. Build the diagnostic aligned-replacement surface for explanation, but do not make it a submission candidate. Do not use pseudo-GT anywhere in training, selection, validation, or user-level decisions.
