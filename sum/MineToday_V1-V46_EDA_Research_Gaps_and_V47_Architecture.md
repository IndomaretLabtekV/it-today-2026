# MineToday V1-V46 Research Reset

## Executive Summary

This document is the source-of-truth research handoff after V46. It is based on direct inspection of all six organizer data files, every versioned notebook through V46, every versioned results directory through V46, and all seven historical handoff ZIPs in `sum/`. Direct artifacts take precedence over handoff prose. Where a quantity is absent or sources cannot be reconciled, it is marked `NOT ESTABLISHED FROM AVAILABLE ARTIFACTS`. Questions requiring new computation are marked `UNMEASURED EDA QUESTION` and are not answered by inference.

The central diagnosis is not that MineToday needs another stronger whole-list model. The repository already contains sparse lexical systems, multiple learning-to-rank objectives, multilingual bi-encoders and cross-encoders, Qwen semantic branches, BGE-M3 dense/sparse/ColBERT features, F2LLM, IndoBERT, XLM-R, trajectory features, curriculum priors, generator-state features, and a high-capacity TabM/listwise model. Broad alternatives repeatedly find offline improvements that fail on fresh blocks, LOCK, or the public leaderboard. At the same time, public causal probes show that a small number of interventions carry disproportionate value: removing four V30 S2 Top1 actions from the current champion lowers public NDCG from `0.66028` to `0.66000`, while removing four S3 membership actions or three legacy V25 `M002 > M003` B45 actions produces no displayed movement. The strongest remaining bottleneck is therefore **selective Top1 authority**: plausible challengers exist, but the project has not learned a trustworthy rule for when a new representation should override a strong, publicly adapted incumbent.

V46 does not refute its dual-semantic hypothesis. Its tokenizer/dependency path failed before multilingual E5 or Nomic embeddings were produced; DESIGN was not meaningfully evaluated, LOCK was not opened, and the output failed closed with zero actions. V46 is `NOT EXECUTED`, not a model failure (`results/v46/v46/v46_summary.json`, fields `embeddings_ready`, `design_gate`, `lock_opened`, and test action count; `results/v46/v46/v46_reproduction_audit.json`).

The recommended V47 is a deliberately smaller version of the unexecuted idea: freeze the exact `0.66028` champion and every already protected action, restrict eligibility to S0 users, permit only one score-preserving Top1 promotion from incumbent ranks 2-4, and train two independent, strongly regularized action-utility heads over frozen multilingual E5 and Nomic-v2-MoE representations. Deploy only consensus actions that pass a preregistered risk-coverage gate on a fresh DESIGN split and then a one-shot LOCK. Existing structural features are controls, not an authority source; generator-state vetoes, membership edits, global reranking, and lower-boundary mining are excluded. This tests one scientific question: **do two not-yet-executed representation families contain transferable information about exact Top1 action utility that the champion lacks?**

## Current Competition State

| Item | Current evidence | Interpretation |
|---|---:|---|
| Public champion | `0.66028` | Exact current anchor supplied with the task and consistent with the latest handoff lineage. |
| Recent Top-5 region | about `0.66076` | Gap from champion is about `0.00048`; this is contextual public information, not a stable target. |
| Recent leaderboard lead | about `0.66142` | Gap from champion is about `0.00114`. |
| Candidate modules | 17 | Full-list scoring is easy computationally; reliable personalized ordering is not. |
| Metric | user-level NDCG@5 | Top1 and rank-boundary actions have sharply different value. |
| Public development history | many submissions and action probes | The public leaderboard is now contaminated development evidence. |

The exact current champion is the action-composed lineage ending at the `0.66028` public result, not a single independently trained model. The latest archival state reconstructs Hybrid34, S3 pruning, and B23/B34/B45 modifications; the task additionally supplies three newer causal removals. The public score establishes the package on the public subset. It does not establish private-leaderboard performance, individual-action causality, or generalization of the policy that found the actions.

## Source-of-Truth and Evidence Hierarchy

Evidence was reconciled in this order:

1. Organizer files in `data/` and direct result artifacts.
2. Executed notebook code and cell output.
3. Exact public submission results recorded in artifacts or handoffs.
4. The newest handoff synthesis.
5. Older handoffs.
6. Explicit inference.

The main provenance caveats are:

- `notebooks/v25.ipynb` contains a V24 title and V24 probe workflow. V25 itself is supported by `results/v25/v25_summary.json` and `results/v25/v25_change_manifest.csv`, not by that notebook. This is a genuine notebook/version mismatch.
- `results/v41/` is absent. V41 values come from executed output in `notebooks/v41.ipynb`, while V41.1 and V41.2 have direct result artifacts.
- V3 has no completed semantic result directory. `notebooks/v3.ipynb` terminates with `ValueError: Attempting to unscale FP16 gradients`; only the preceding sparse preparation score is evidence.
- Several public scores exist only in dated handoffs rather than result JSON. Those are labeled as public-history evidence, not direct model metrics.
- Old split names such as CAL, LOCK, fresh salts, pseudo-public, chronology, and audit are not assumed independent merely because the names differ. Reuse and overlap are audited below.

## Dataset EDA: What We Actually Know

### Users

**FACT.** `data/train_relevance.csv` contains 4,000 labeled users, `U_0001` through `U_4000`. `data/test.csv` contains 1,000 test users, `U_4001` through `U_5000`. IDs are disjoint and form contiguous blocks. `data/user_assessments.csv` contains all 5,000 competition users. `data/sample_submission.csv` contains 1,000 rows and the 17 module columns expected for test.

**FACT.** `data/chat_history.csv` contains 14,573 rows for 4,940 unique users. Of the 5,000 target users, 4,850 have at least one chat row and 150 do not: 117 train users and 33 test users. The chat file also includes 90 out-of-scope users, `U_5001` through `U_5090`, accounting for 189 rows. These rows have no competition target and must never enter supervised target statistics.

**INFERENCE.** The contiguous user-ID boundary creates a credible chronology or generator-batch risk even though IDs are not features in the strongest systems. Random folds alone cannot establish transfer across that boundary. Conversely, a contiguous ID split is a stress test, not proof that production is chronological.

### Chat Histories

**FACT.** Direct file inspection establishes multiple messages per user with timestamps, substantial exact/template repetition, and administrative content. Existing audits report 535 administrative/noise messages, about 51.5% duplicated chat rows, normalized exact train-message overlap for 69.9% of test messages, and at least one train-repeated template for 95.8% of test users who have chat. The latest handoff records 1,095 latest-message template groups and 46 train plus 46 test intent signatures, with zero unseen test signatures (`sum/MineToday_Full_Chat_Handoff_2026-09-04_CURRENT.zip::MineToday_Full_Chat_Handoff_2026-09-04_CURRENT/04_DATA_AND_EDA/DATASET_CURRENT_AUDIT.md`). These counts are handoff-derived summaries of repository analyses, not newly recomputed statistics.

**FACT.** A V4 notebook audit of latest-message topics found the largest groups to be CV 431, GenAI 408, EDA 395, Python 392, SQL 375, ML evaluation 368, statistics 354, ML 344, MLOps 327, business 221, no recognized topic 209, and dashboard 176 (`notebooks/v4.ipynb`, executed EDA output).

**INFERENCE.** Template overlap explains why word/character TF-IDF remains unusually competitive and why ordinary random folds can look stable. It also means row-level random splitting would be invalid. User-level folds avoid direct same-user leakage but still allow near-identical template families to straddle folds.

**INFERENCE.** Histories can encode state transitions rather than static intent. A user may ask an exploratory question, disclose readiness, then narrow to a goal. Most historical representations concatenate or pool the history; V39/V39.1 add hand-built trajectory summaries, but no learned sequence model has been validated. The existence and predictive value of transition families remain unmeasured.

### Assessments

**FACT.** Each assessment row stores ten skill responses. Across 50,000 cells, 251 values are invalid or noncanonical, including Indonesian number words and out-of-range values such as `99`, `12`, `-1`, and `-3`. Historical pipelines convert valid values, add invalid/missing flags, and use robust row statistics; this behavior appears from V2 onward (`notebooks/v2.ipynb`; later reproduction audits).

**FACT.** Cleaned train/test means recorded by the V2 pipeline are close but generally lower in test: Python `1.484/1.451`, SQL `1.3485/1.287`, statistics `1.3275/1.265`, EDA `1.47625/1.441`, ML `1.2295/1.165`, ML evaluation `1.06425/0.995`, deep learning `0.66975/0.619`, GenAI `0.9055/0.883`, business `1.30425/1.323`, and research `1.167/1.113` (`notebooks/v2.ipynb`, assessment summary output). Business is the only listed dimension with a higher test mean.

**INFERENCE.** Assessments are better treated as readiness conditioned on a candidate module than as a global user vector. V2's raw assessment model is weak, while later readiness interactions help only narrowly. This is compatible with prerequisite-conditioned utility and inconsistent with assessments being a sufficient recommendation signal.

### Module Catalog

**FACT.** `data/modules_catalog.csv` defines 17 modules:

| ID | Short identity | Catalog structure |
|---|---|---|
| M001 | Excel | Beginner |
| M002 | Python | Beginner |
| M003 | SQL | Beginner |
| M004 | Web scraping | Requires Python |
| M005 | Version control | Intermediate |
| M006 | Statistics | Intermediate |
| M007 | Exploratory data analysis | Requires Python and statistics |
| M008 | Dashboarding | Requires SQL or Excel |
| M009 | Machine learning | Requires EDA and statistics |
| M010 | Computer vision | Requires ML or deep learning |
| M011 | Natural language processing | Requires ML |
| M012 | Generative AI | Requires NLP |
| M013 | Prompt engineering | Intermediate |
| M014 | AI automation | Intermediate |
| M015 | No-code AI | Beginner |
| M016 | MLOps | Requires ML, Python, and version control |
| M017 | Career development | All levels |

**INFERENCE.** The catalog exposes a small prerequisite graph, but relevance is not reducible to prerequisite distance. V29 reports that handcrafted prerequisite features capture only about 35.8% of missed mass and co-occurrence features about 0.4%; V29.5 reports about 31.36% for a related transitive-union definition (`results/v29/v29_summary.json`, fields `prereq_ratio`, `cooc_captured_mass`, and `missed_relevance_mass`; `results/v29_5/v295_summary.json`, fields `transitive_union_captured_mass` and `missed_relevance_mass`). Direct curriculum additions in V30 are small and inconsistent. The graph is useful as a veto or prior, not yet as autonomous ranking authority.

**INFERENCE.** The catalog text contains semantically adjacent families that are difficult for generic similarity to order: foundations M002/M003/M006/M007/M009, advanced AI M010/M011/M012, and applied GenAI M013/M014/M015. A chat about “AI” or “data” can make several modules jointly relevant while providing little evidence for their exact Top1 order. This is a concrete mechanism by which semantic similarity can improve membership yet misorder NDCG-critical ranks (`data/modules_catalog.csv`; module-level target frequencies above).

**FACT.** Recurrent high-leverage and high-confusion pairs include `M002/M003`, `M003/M014`, `M012/M014`, and curriculum-adjacent families involving M006/M007/M009. The exact action lineage is in `results/v25/v25_change_manifest.csv`, `results/v30/v30_test_change_ledger.csv`, `results/v31/v31_test_change_ledger.csv`, and the latest causal ledger in `sum/MineToday_Full_Chat_Handoff_2026-09-04_CURRENT.zip::MineToday_Full_Chat_Handoff_2026-09-04_CURRENT/03_PUBLIC_LB_AND_CAUSAL_PROBES/CAUSAL_COMPONENT_LEDGER.md`.

### Target Geometry

**FACT.** Each train user has four, five, or six positive modules: 1,362 users have four, 1,285 have five, and 1,353 have six. Every row has exactly one relevance value equal to `1.0`. There are about 57 distinct numeric target values and 3,960 distinct ordered target sequences, but only 3,146 distinct positive sets (`data/train_relevance.csv`; executed EDA in `notebooks/v1.ipynb`).

**FACT.** The mean relevance after sorting each row descending is approximately `[1.0000, 0.8492, 0.7000, 0.5491, 0.2637, 0.0847]`, where the last two means include structural zeros for four- and five-positive users. Conditional positive-band means are approximately `0.40` and `0.25` at ranks 5 and 6 (`notebooks/v1.ipynb`; latest dataset audit handoff). These numbers are compatible rather than contradictory.

**FACT.** Module target prevalence is strongly nonuniform:

| Module | Top1 users | Positive users | Mean relevance |
|---|---:|---:|---:|
| M003 | 495 | 1,915 | 0.345700 |
| M002 | 419 | 1,701 | 0.306335 |
| M006 | 401 | 1,621 | 0.296135 |
| M012 | 376 | 1,331 | 0.244945 |
| M009 | 354 | 1,267 | 0.231707 |
| M017 | 285 | 1,358 | 0.236697 |
| M015 | 255 | 1,244 | 0.217440 |
| M016 | 252 | 1,037 | 0.184418 |
| M007 | 224 | 1,034 | 0.179815 |
| M014 | 222 | 1,353 | 0.225635 |
| M001 | 192 | 1,317 | 0.220907 |
| M010 | 156 | 906 | 0.150133 |
| M008 | 137 | 1,061 | 0.172050 |
| M005 | 95 | 768 | 0.121598 |
| M013 | 69 | 945 | 0.146060 |
| M011 | 40 | 639 | 0.094365 |
| M004 | 28 | 494 | 0.072780 |

Source: `data/train_relevance.csv` and executed frequency output in `notebooks/v1.ipynb`. The rare Top1 labels for M004 and M011 explain why semantically plausible models can still be poorly calibrated for the actual label generator.

**INFERENCE.** The values resemble a partial ranking with a tied zero tail more than independent relevance labels. The positive bands also resemble rounded draws around rank bands, but the generating distribution is `NOT ESTABLISHED FROM AVAILABLE ARTIFACTS`. The scientific implication is nevertheless clear: membership and order are distinct tasks, and exact score regression spends capacity on within-band noise that NDCG may not reward.

### Train-Test Structure

**FACT.** V20's broad adversarial train-versus-test classifier achieved AUC `0.509918`, close to chance (`results/v20/v20_summary.json`). Historical block audits report median maximum module-prevalence deviation of about 1.2 percentage points. These observations argue against a large marginal covariate shift.

**INFERENCE.** Near-chance broad adversarial AUC does not rule out conditional shift in `P(action utility | template, module pair, margin)`. The repeated pattern of ordinary-fold gains reversing on chronology, template-held-out, LOCK, or public LB supports conditional validation mismatch rather than a gross train-test distribution break.

**FACT.** Test target/module-frequency shift cannot be measured because test relevance labels are unavailable. Chat intent coverage appears high, including zero unseen intent signatures in the latest audit, and the assessment means show only modest marginal shifts. Whether `P(module rank | observed intent, readiness)` shifts is `NOT ESTABLISHED FROM AVAILABLE ARTIFACTS`.

**FACT.** A simple predictor for the number of positives is reported near the majority baseline, around `0.336` versus `0.341`, but versions of the handoff differ slightly. The exact result is `NOT ESTABLISHED FROM AVAILABLE ARTIFACTS`; K should not be treated as accurately predictable without a reproducible artifact.

### NDCG and Boundary Economics

**FACT.** V14's adjacent pairwise AUCs decline sharply down the list: B12 `0.668`, B23 `0.614`, B34 `0.563`, B45 `0.530`, and B56 `0.521` (`results/v14/v14_fold_diagnostics.csv`; `notebooks/v14.ipynb`). The ordering boundary becomes nearly unlearnable exactly where many later projects spent effort.

**FACT.** Existing exact-swap analyses place approximate per-user NDCG leverage around `0.032-0.036` for B23, `0.017-0.020` for B34, and `0.010-0.012` for B45; a membership B56 correction can be around `0.095-0.106` when it truly replaces a zero with a relevant item. These are action-conditional magnitudes from historical ledgers, not expected gains from a classifier.

**INFERENCE.** Large membership leverage is offset by very poor learnability and high harm when the inferred K or boundary is wrong. Top1 promotions combine substantial leverage with more training support than rare membership transitions. The remaining economic target is high precision at very low coverage, not more total changed users.

## EDA Questions Still Unmeasured

The following require new computation and were not executed for this audit.

| Status | Question and why it matters | Exact future statistic or plot | Interpretation of outcomes |
|---|---|---|---|
| UNMEASURED EDA QUESTION | How concentrated are histories and recency patterns per user? Pooling may erase a final intent transition. | Distribution of message count, inter-message time, history span, and fraction of target information attributable to the final 1, 2, and 3 messages, stratified by template family. | Strong recency concentration favors a state-transition encoder; flat value favors simple pooling. |
| UNMEASURED EDA QUESTION | How much random-fold optimism comes from template families? | Cluster normalized/paraphrase templates without targets, then compare user-random, exact-template-grouped, paraphrase-clustered, and contiguous-ID CV for the frozen champion and one simple residual. | A large grouped-CV drop confirms template leakage-like dependence; a small drop weakens H4. |
| UNMEASURED EDA QUESTION | Do explicit module mentions predict membership, Top1, or only topic? | Precision/recall and calibrated conditional target distributions for explicit and alias mentions, by negation, comparison, and recency. | Top1 precision supports lexical action rules; membership-only precision argues against Top1 promotion. |
| UNMEASURED EDA QUESTION | Are chat histories genuine state transitions? | Transition matrix of intent/readiness states over time, mutual information with Top1 after conditioning on latest intent, and examples by high/low transition surprisal. | Incremental information supports sequential modeling; none closes the branch. |
| UNMEASURED EDA QUESTION | Are assessment dimensions redundant or nonmonotonic? | Spearman matrix, invalid-pattern audit, module-conditional partial dependence from cross-fitted simple models, and monotonicity tests by prerequisite relation. | Stable monotonic effects justify curriculum conditioning; nonlinear or family-specific effects require interactions; no effect closes assessment expansion. |
| UNMEASURED EDA QUESTION | Which module pairs have stable global precedence and which are personalized? | Pairwise win matrix with Wilson or cluster-bootstrap intervals across template, ID, and assessment strata. | Stable precedence supports a global curriculum prior; heterogeneous reversals support conditional experts. |
| UNMEASURED EDA QUESTION | Is there a latent synthetic generator fingerprint beyond V42's hand features? | Template-family target entropy, conditional target-set duplication, lexical slot substitutions, and train/test frequency drift, all without using test labels. | Low entropy and reusable mappings support retrieval/template priors; high entropy implies irreducible or hidden state. |
| UNMEASURED EDA QUESTION | Where does action utility transfer fail? | Reliability and risk-coverage curves for exact counterfactual delta, sliced by action type, state, pair, margin, template family, and split world. | Stable curves enable selective authority; crossing curves imply the proposed gate is not transportable. |
| UNMEASURED EDA QUESTION | How coarse is public-score resolution for small action packages? | Infer only after private/full labels or organizer disclosure; compare exact package deltas on full test labels. | It would distinguish truly neutral actions from rounded or public-subset-invisible effects. Public probing cannot answer this safely. |

## Full V1-V46 Experiment Lineage

The matrix below records scientific versions, including meaningful subversions. `Offline` values are local NDCG deltas or levels under that version's own split and are not cross-version comparable unless explicitly stated. Public values are exact only where recorded.

| Version | Core hypothesis and architecture | Validation and result | Behavior, failure, and status |
|---|---|---|---|
| V1 | Sparse TF-IDF/SVD/Ridge/template features plus Qwen3 embedding/reranker and LambdaRank. | OOF: ranker `.613058`, Ridge `.635414`, semantic `.426644`, reranker `.483830`, memory `.504565`, blend `.640542`; public `.63629` (`notebooks/v1.ipynb`; `results/v1/submission_ultimate_baseline.csv`; 2026-08-23 handoff scoreboard). | Sparse signal dominates; rare M004/M011 collapse. **KEEP sparse baseline; CLOSED broad early semantic blend.** |
| V2 | Supervised multilingual GTE with LoRA/multihead, sparse, memory, assessment, LambdaRank. | OOF GTE `.635888`, sparse `.648442`, ranker `.617271`, memory `.501800`, assessment `.504263`, ensemble `.655324`; public `.64413` (`notebooks/v2.ipynb`; `results/v2/submission_v2_supervised.csv`). | Better offline representation, severe public transfer gap. **CLOSED as broad architecture.** |
| V3 | GTE reranker/cross-encoder over sparse candidates. | Sparse preparation `.655273`; training aborts with FP16 gradient-unscale error (`notebooks/v3.ipynb`). | No completed semantic evaluation. **NOT EXECUTED**, not a refuted reranker hypothesis. |
| V3.1 | Word/character TF-IDF multihead, structured features, per-module LightGBM, calibrated blend. | Sparse `.658254`; CAL035 `.662358`; CAL045 `.662646`; public sparse `.65036`, CAL035 `.65540`, CAL045 `.65639` (`notebooks/v3_1.ipynb`; `results/v3_1/submission_v31_primary.csv`; public-history handoff). | First durable jump; calibration matters. **KEEP lineage.** |
| V4 | Multi-expert LambdaRank over strong sparse anchor, message maxima, co-occurrence. | Anchor `.662608`, LambdaRank `.664117`, message-max `.615001`, co-occurrence `.629753` (`notebooks/v4.ipynb`). Public standalone result `NOT ESTABLISHED FROM AVAILABLE ARTIFACTS`. | Small offline ranking gain; broad weak experts add little. **PARTLY KEEP as evidence.** |
| V5 | Generator-aware CatBoost YetiRank. | Anchor `.661854`, YetiRank `.664240`, 5/5 fold direction (`notebooks/v5.ipynb`). Public standalone result `NOT ESTABLISHED FROM AVAILABLE ARTIFACTS`. | Promising local objective, no durable causal evidence. **UNRESOLVED historically, superseded by later objective tests.** |
| V6 | Nested ordinal/YetiRank formulation. | Anchor `.662909`, ranker `.664486`, primary `.664455`, 3/5 folds (`notebooks/v6.ipynb`). | Nesting improves validity but gain is uneven. **KEEP methodological correction; CLOSED as standalone route.** |
| V3.1/V4/V6 Borda | Rank aggregation across correlated sparse/ranking models. | OOF reported near `.6676`; public `.65489` (2026-08-23 handoff). | Offline ensemble dependence badly overstates transfer. **CLOSED arbitrary rank blending.** |
| V7 | Pairwise surgical correction of the calibrated anchor. | Strict anchor `.663534`; surgical `.663150`; delta `-.000384`, bootstrap CI `[-.000994,.000212]`, 2/5 folds; tail `-.001163` (`notebooks/v7.ipynb`). | Surgery adds harm, especially tail. **CLOSED.** |
| V8 | Repeated-CV calibration and membership/Top1 interventions. | OOF `.664735`, delta `+.002089`, CI `[.000617,.003605]`, 4/5 folds; test changes about 12.7% membership and 1.9% Top1. Public full-fit `.65577`, fold-bag `.65140` (`results/v8/v8_summary.json`; public history). | Strongest early validation-to-public reversal; calibration geometry and dependence. **CLOSED; warning case.** |
| V9 | Nested distribution-preserving ordinal correction. | Delta `+.000854`, CI `[.000014,.001710]`, 5/5; about 6.9% test Top5-set and 0.8% Top1 changes; gate false (`results/v9/v9_summary.json`). | More honest but insufficiently stable. **CLOSED.** |
| V9.1 | Adaptive shrinkage on V9. | Delta `+.001475`, CI `[.000444,.002508]`, 5/5; about 9.6% set and 1.0% Top1 changes; public `.65432` (`results/v9_1/v91_summary.json`). | Catastrophic transfer despite clean-looking OOF. **CLOSED; validation counterexample.** |
| V10 | Top5 probability, utility, and ordinal corrections. | Best C delta `+.000985`, 5/5 but CI crosses zero; about 11.2% set and 2.5% Top1 changes (`results/v10/v10_summary.json`). | Too broad and not uncertainty-safe. **CLOSED.** |
| V10.1 | Only rank-5/rank-6 swaps. | Delta `-.000242`, 1/5; mean true-swap delta `-.004589`; about 2.9% test changes (`results/v10_1/v10_1_summary.json`). | Lower boundary is both weakly learnable and harmful. **CLOSED.** |
| V11 | Temporal rolling validation and sequential correction. | Weighted delta `+.000658`, only 1/4 windows; sequential result near zero (`results/v11/v11_summary.json`). | No temporal stability. **CLOSED.** |
| V12 | Module-profile specialist on sparse leader. | Leader6 `.664058`, delta `+.001412`, 5/5; public `.65792` (`results/v12/v12_summary.json`). | First later durable public gain. **KEEP evidence and profile features.** |
| V12.1 | Four-action micro extension. | Incremental `+.000249`, 4/5, CI crosses zero (`results/v12_1/v12_1_summary.json`). Public exact result `NOT ESTABLISHED FROM AVAILABLE ARTIFACTS`. | Too small for attribution. **UNRESOLVED package.** |
| V12.2 | Consensus3 module-specific calibration weights. | Ordinary `.664368`, strict `.664966`; public `.65856` (`results/v12_2/v12_2_summary.json`). | Strong durable calibrated anchor. **KEEP champion ancestry.** |
| V13 | Router between Consensus3 and V9.1. | Delta `+.000316`, 5/5, CI crosses; 35 test routes (`results/v13/v13_summary.json`). | Candidate diversity does not solve authority. **CLOSED.** |
| V13.1 | Dual router among V9.1/V10C alternatives. | Local delta `+.000640`, CI positive, 5/5; 42 routes; public `.65754` (`results/v13_1/v13_1_summary.json`, `v13_1_second_order.csv`). | Second-level features were not fully nested; public regression confirms optimism. **INVALID for efficacy; CLOSED.** |
| V14 | Fully nested top-six pairwise ranking. | Delta `-.000398`, 3/5; only four test order changes (`results/v14/v14_summary.json`). | Adjacent AUC decays toward the tail. **CLOSED broad pairwise surgeon.** |
| V15 | Fully nested utility router with abstention. | Zero approved actions (`results/v15/v15_summary.json`). | Policy executed and abstained. This does not refute existence of useful actions, but this feature/policy has no coverage. **CLOSED implementation; hypothesis unresolved then.** |
| V16 | Frozen Qwen8B global semantic adjustment. | At beta `.025`, ordinary delta `-.000708`, audit `+.000724`, chronology `+.001237`; larger weights increase ordinary harm (`results/v16/v16_summary.json`). | Split-sign reversal. **CLOSED global semantic authority.** |
| V17 | Qwen top-five-aware semantic branch. | Ordinary/audit effects are tiny; chronology becomes mixed or negative under global locks (`results/v17/v17_summary.json`). | Top5 awareness does not stabilize it. **CLOSED.** |
| V18 | Multi-intent/message prototypes and diagonal semantic adapter. | Ordinary near zero or negative, audit positive; public attack `.65829` versus `.65856` anchor (`results/v18/v18_summary.json`). | Semantic changes hurt public. **CLOSED broad adapter.** |
| V19 | Stability-selected semantic correction. | Stable050 ordinary `+.000884` 5/5, audit `+.000939`, chronology `+.000479`; test about 9.6% set, 26% order, 1.8% Top1; public `.65791` (`results/v19/v19_summary.json`). | Multi-world offline stability still fails public. **CLOSED and important validation warning.** |
| V20 | Qwen8B plus BGE-M3 dense, sparse, and ColBERT late-interaction representations. | Adversarial AUC `.509918`; selected dev `+.000096` 2/4, novel `-.000496`, LOCK `+.000078` with wide CI (`results/v20/v20_summary.json`). | Late interaction was genuinely tested; no stable utility. **CLOSED this broad blend.** |
| V21 | Qwen8B QLoRA specialist. | CAL `+.000201` 2/4, seen `+.001322`, novel `-.000283`; LOCK `+.001374` but CAL gate false; public `.65786` (`results/v21/v21_summary.json`). | LOCK success cannot rescue failed selection; anchor transport mismatch. **CLOSED.** |
| V21.1 | Bagged low-capacity semantic calibrators, module trust on M003/M009/M014/M016. | Selected stable050 CAL `+.000717` 2/4, LOCK `-.000531`; an older stable035 policy had CAL `+.000439` 3/4 and public `.65869` (`results/v21_1/v21_1_summary.json`, `v21_1_module_trust.csv`). | Broad policy fails, but some actions seed later V22/V25 gains. **KEEP only action ancestry.** |
| V21.2 | Anchor LambdaRank inside a trust region. | CAL `+.000361`, 2/4; LOCK not opened; public `.65776` (`results/v21_2/v21_2_summary.json`). | Trust region alone is insufficient. **CLOSED.** |
| V22/V22.1 | M003/M014 residual and first surgical public package. | CAL/LOCK variants weak or negative; public `.65846` (`results/v22/v22_summary.json`, `v22_change_audit.csv`; handoff public history). | Offline model is weak; isolated public actions may help. **CLOSED model, KEEP action lineage.** |
| V22.2 | Four specific public actions. | U4047, U4498, U4876, U4953 package scores `.65878` (2026-08-26 handoff `V22_PUBLIC_DECOMPOSITION.md`). | Package improves, individual attribution not established. **KEEP package evidence.** |
| V22.3 | Retain U4047 and U4953. | Public `.65912` (same source). | Smaller package improves further. **KEEP ancestry, attribution still incomplete.** |
| V23 | F2LLM-4B structured/hybrid action-utility model. | Conservative delta `+.00000121` with negative worst block; balanced/confirmed variants negative; 3/6/1 candidate actions but primary fails (`results/v23/v23_summary.json`). | New LLM family does not supply reliable utility. **CLOSED.** |
| V24 | Causal submission-probe materialization. | PB plus U4498 is recorded flat at `.65912`; other probes `NOT ESTABLISHED FROM AVAILABLE ARTIFACTS` (`results/v24/v24_summary.json`, `v24_probe_manifest.csv`). | Establishes only tested package equality at display precision. **KEEP provenance.** |
| V25 | Legacy six-action core plus three relaxed actions. | Core6 raises public `.65912` to `.65919`; relaxed3 leave display `.65919` (`results/v25/v25_summary.json`, `v25_change_manifest.csv`; 2026-08-27 handoff). | Sparse actions can beat broad models. Individual benefit is not identified. **KEEP champion ancestry.** |
| V26 | Attribution attempt pruning U4491. | Public remains `.65919`; reconstruction hash mismatch is documented (`results/v26/v26_summary.json`, `v26_exact_diff_ledger.csv`). | Flat display and hash mismatch prevent exact attribution. **CLOSED further inference from this probe.** |
| V27 | Dedicated Top1/Top2 heads. | Gates fail and zero actions deploy (`results/v27/v27_summary.json`, `v27_validation_summary.csv`). | No validated competence. **CLOSED.** |
| V27.1 | Head12 group/ordinary action selection. | Development selection looks strong, fresh LOCK has one harmful action, chronology zero; only four test candidates with theoretical maximum about `.000717` (`results/v27_1/v271_summary.json`, `v271_validation_summary.csv`). | Selection optimism. **CLOSED.** |
| V28 | Broad hierarchical/multitask curriculum model. | LOCK `-.002248`, chronology `+.001229`, 251 changed users (`results/v28/v28_summary.json`, `v28_validation_summary.csv`). | Massive split reversal and excess authority. **CLOSED.** |
| V29 | Curriculum residual learning-to-rank. | Selected full strict `+.001116`, ordinary `+.0000026`, LOCK `+.000806`, chronology `+.000122`; pseudo-public tail fails; 78 test changes (`results/v29/v29_summary.json`, `v29_validation_summary.csv`, `v29_pseudo_public_lock.csv`). | Some curriculum residual signal, poor transport. **UNRESOLVED as prior, CLOSED broad policy.** |
| V29.5 | Uncertainty, gap, co-occurrence, curriculum-distance selection. | Uncertainty correlations near zero; fresh LOCK variants negative, best gap+cooc `-.000628`; chronology can be `+.000777` (`results/v29_5/v295_summary.json`, `v295_uncertainty_summary.csv`). | Scores do not rank action risk across worlds. **CLOSED these uncertainty proxies.** |
| V30 | Fixed-alpha V29 residual across multiple salts, with direct curriculum ablation. | LTR base C `+.000890`, D `+.000449`, E `+.000855`, 13/15 fold directions, pooled `+.000732` CI `[.000337,.001145]`; sealed F `+.000969`, chronology `+.000042`; 77 test changes, 8 Top1, 7 membership; public `.65963` (`results/v30/v30_summary.json`, `v30_three_salt_summary.csv`, `v30_test_change_ledger.csv`). Direct curriculum incremental mean is only `+.0000835`, 2/3 worlds. | Best honest broad residual and durable public jump. **KEEP frozen backbone and actions.** |
| V31 | Fold-safe intent-signature by module by readiness target encoding residual. | Incremental `+.000153`, CI crosses, 11/15; 60 changes; gate false. Whole public `.65949`; new-only 26 `.65906`; overlap34 hybrid `.66006` (`results/v31/v31_summary.json`, `v31_three_salt_summary.csv`, `v31_test_change_ledger.csv`; 2026-08-28 handoff). | New-only changes are toxic; overlap with V30 is high precision. **KEEP consensus geometry, CLOSED broad TE residual.** |
| V32 | Objective diversity using XE, YetiRank, median, and unanimity. | Fresh salts inconsistent or negative (`results/v32/v32_summary.json`, `v32_three_salt_summary.csv`). | Objective diversity is not error diversity. **CLOSED.** |
| V33 | Retrieval-memory residual. | Worlds N/O/P mixed; pooled broad delta `+.0000275`, worst block `-.000235`; protected policy about zero; zero deployment (`results/v33/v33_summary.json`). | Template retrieval adds no stable residual utility. **CLOSED.** |
| V34 | Frozen BGE reranker Head12 utility and competence gate. | Benefit AUC about `.566-.580`; stage-2 metrics around `.655-.663`, but zero approved actions (`results/v34/v34_summary.json`). | Some discrimination, no deployable confidence. **CLOSED policy; representation not universally refuted.** |
| V35 | IndoBERT TAPT plus pair-utility model. | Worlds U/V/W ATTACK pooled `+.000225`, CI lower `+.0000056`, 8 beneficial/2 harmful, 10 actions (`results/v35/v35_summary.json`, `v35_tapt_summary.json`). | Initial TAPT saw held-out text and did not preserve fold-pure isolation. **QUESTIONABLE, not causal evidence.** |
| V35.1 | Fold-pure IndoBERT confirmation. | Fresh X `+.000270` 4/5; sealed template T `-.000476`, with 1 benefit/5 harm/1 tie; chronology has zero actions (`results/v35_1/v35_1_summary.json`, split summaries). | Corrected validation reverses the claim. **CLOSED.** |
| V36 | Fold-matched XE competence router. | Hundreds of proposals but zero stable proposals under exact-identity matching; zero actions (`results/v36/v36_reproduction_audit.json`, `v36_intentstate_support_audit.csv`). | Exact identity is too strict; this implementation has no coverage. **CLOSED implementation, not all deferral.** |
| V36.1 | Pressure-stability action policy. | Corridor E `-.000577`, F `-.000311`, G `+.000376`; attack policies negative; broad oracle supply `.02-.03`; primary stays anchor (`results/v36_1/v36_1_pressure_action_audit.csv`). | Worlds overlap, so apparent replication is not independent. **CLOSED policy.** |
| V37 | Disjoint H/I/J competence test for S0 crossing actions. | About 127-138 eligible users per world, oracle `+.0032` to `+.0061`; selector zero or negative; test geometry AUC `.529`; no test actions (`results/v37/v37_summary.json`, `v37_test_pressure_audit.csv`). | Confirms action supply but selector failure. **KEEP bottleneck evidence; CLOSED selector.** |
| V38 | XLM-R direct residual pair model. | About 1,300 eligible cases per world, weak NLL, zero actions (`results/v38/v38_summary.json`, `v38_calibration_policy.csv`). | Broad eligibility does not create competence. **CLOSED.** |
| V38.1 | XLM-R curriculum relevance/pair model. | AA zero; AB SAFE `+.000248`, 9 actions with 5/2/2 benefit/harm/tie and positive CI lower; AC SAFE `+.000251`, 8 actions with 5/2/1 but CI crosses; AB membership `-.000340`; development gate fails (`results/v38_1/v38_1_summary.json`). | Small signal, no robust gate. **UNRESOLVED representation, CLOSED deployment policy.** |
| V39 | Direct trajectory Top1/membership features in S0. | One test Top1 proposal; no action type approved (`results/v39/v39_summary.json`, `v39_fresh_world_summary.csv`). | Handcrafted trajectory adds no validated authority. **CLOSED.** |
| V39.1 | Pairwise boundary trajectory extension. | Six raw validation proposals and one test Top1 proposal; none approved (`results/v39_1/v39_1_summary.json`, `v39_1_fresh_world_summary.csv`). | Same conclusion. **CLOSED.** |
| V39.5 | Public boundary decomposition of S3. | From `.66020`, B23 removal/variant yields `.65998`, B34 `.66026`, and B34 plus B45 `.66028` (`notebooks/v39_5.ipynb`; latest causal handoff). | B23 package is harmful; B34 and conditional B45 add small displayed value. **KEEP current package.** |
| V40 | B34 prerequisite membership model. | Fresh selected action mean is strongly negative, with six harmful selected actions; gate fails (`notebooks/v40.ipynb`; latest handoff timeline). | Curriculum membership extrapolation is unsafe. **CLOSED.** |
| V41 | Qwen8B plus XLM-R stacked residual logistic models. | CAL Q `+.000824` 4/4 and QX `+.000499` 3/4; LOCK Q `-.000136`, 29 changes including 12 membership (`notebooks/v41.ipynb`). | Second-level dependency is not fully nested and LOCK reverses. **INVALID for efficacy; CLOSED.** |
| V41.1 | Clean Qwen hierarchical cumulative ordinal residual. | CAL delta `-.0000285`, CI `[-.000996,.000923]`, 2/5 folds and blocks; Top1 delta `-.002` (`results/v41_1/v41_1_cal_metrics.csv`, `v41_1_cal_fold_metrics.csv`, `v41_1_cal_block_metrics.csv`). | Clean rerun removes apparent gain. **CLOSED.** |
| V41.2 | Qwen Bradley-Terry/logistic Top1 challenger. | Top1-within-top4 oracle `+.026932`; 275 of 441 wrong-Top1 cases recoverable among 865 eligible. Policy selects zero CAL actions, LOCK not opened (`results/v41_2/v41_2_summary.json`, `v41_2_cal_metrics.csv`, action ledgers). | Large oracle supply, no learnable confidence. **KEEP bottleneck evidence; CLOSED policy.** |
| V42 | Generator-state YetiRank residual. | DESIGN anchor `.659612`, candidate `.659733`, delta `+.000121`; representation increment `+.0000935`, 4/5 folds; conceptual geometry and template stress fail, LOCK not opened (`results/v42/v42_summary.json`, audits). | Generator signal exists weakly but does not pass stress tests. **CLOSED as authority.** |
| V42.1 | Corrected geometry with empirical-Bayes shrinkage. | DESIGN delta `+.00006898`, 3/5 folds, 4/5 blocks, Top1 delta `-.000294`, 21 changes; no LOCK/test (`results/v42_1/v42_1_summary.json`, `v42_1_design_metrics.csv`). | Fixes do not produce useful Top1 value. **CLOSED.** |
| V43 | Joint TabM/MLP partial Plackett-Luce style model. | TabM delta `-.003718`, 0/5, 1,475 changes and 420 Top1; MLP `-.01142` (`results/v43/v43_summary.json`, `v43_design_metrics.csv`). | High-capacity joint replacement overwhelms the anchor. **CLOSED broad joint model.** |
| V43.1 | Nested trust-region version of V43. | Best epoch 8, beta `1/32`, delta `-.00004747`, 1/5 folds/blocks, 17 changes and 3 Top1 (`results/v43_1/v43_1_summary.json`, `v43_1_strength_audit.csv`). | Even tiny authority is not positive. **CLOSED.** |
| V44 | Offline causal pruning of V30 S2 action classes. | No class passes. Top1 prune `+.0000611` but only 2/5 folds, 3/5 blocks and Top1 component `-.000882`; membership `-.0000436`; B23 `+.0000307`; B34 `-.0000334`; B45 `-.0000051`; complex `+.0000154` (`results/v44/v44_summary.json`, class metrics). | Offline class pruning is inconclusive. New public evidence shows removing four V30 S2 Top1 actions loses `.00028`. **KEEP S2 Top1; CLOSED coarse pruning.** |
| V45 | Audit/prune three repeated V25 `M002 > M003` B45 actions. | DESIGN weighted prune `-.000380`, local `+.00667`, both CIs crossing zero; LOCK weighted `+.000113`, CI crossing; design gate false (`results/v45/v45_summary.json`, `v45_design_family_audit.csv`, `v45_lock_family_audit.csv`). | Mixed local/global evidence. New public removal is flat at display precision. **LOW LEVERAGE; stop probing.** |
| V46 | Frozen champion plus multilingual E5 and Nomic embeddings, per-expert exact Top1 utility heads, consensus, and generator veto. | Encoder stage fails before embeddings due tokenizer/dependency backend. `embeddings_ready=false`; DESIGN values are fail-closed placeholders, LOCK not evaluated, zero test actions (`results/v46/v46/v46_summary.json`, `v46_reproduction_audit.json`, `notebooks/v46.ipynb`). | **NOT EXECUTED. HYPOTHESIS UNRESOLVED.** |

## Public Leaderboard and Causal Submission Evidence

### Score lineage

Public results are useful for reconstructing the incumbent but should no longer be treated as an independent validation set.

| Lineage point | Public NDCG@5 | Evidence and qualification |
|---|---:|---|
| V1 | `.63629` | Historical submission ledger. |
| V2 | `.64413` | Historical submission ledger. |
| V3.1 sparse | `.65036` | Historical submission ledger. |
| V3.1 CAL035 | `.65540` | Historical submission ledger. |
| V3.1/V4/V6 Borda | `.65489` | Broad blend regresses. |
| V3.1 CAL045 | `.65639` | Strong calibrated sparse anchor. |
| V8 full-fit / fold-bag | `.65577 / .65140` | Strong OOF claim fails publicly. |
| V9.1 | `.65432` | Another strong OOF claim fails publicly. |
| V12 | `.65792` | Profile specialist gives durable improvement. |
| V12.2 Consensus3 | `.65856` | Durable calibrated anchor. |
| V13.1 | `.65754` | Invalid/optimistic router regresses. |
| V18 / V19 | `.65829 / .65791` | Semantic attacks regress. |
| V21 / V21.1 / V21.2 | `.65786 / .65869 / .65776` | Only the narrower V21.1 package helps. |
| V22.1 / V22.2 / V22.3 | `.65846 / .65878 / .65912` | Successively narrower surgical packages. |
| V24 PB+U4498 | `.65912` | Flat at displayed precision. |
| V25 / V26 | `.65919 / .65919` | Small package improvement, then flat pruning probe. |
| V30 | `.65963` | Broadest later validated public gain. |
| V31 whole / new-only26 | `.65949 / .65906` | New-only actions are harmful. |
| V30/V31 overlap34 Hybrid | `.66006` | Consensus overlap is highly leveraged. |
| Hybrid34 plus two extras | `.65976` | Extra actions hurt. |
| Remove S3 Top1 three | `.65950` | Large displayed loss, about `-.00056`. |
| Remove S3 membership four, old `.66006` anchor | `.66006` | No displayed change in that package context. |
| Remove S3 order 27 | `.66020` | Pruning many lower-order changes helps `+.00014`. |
| B23 package | `.65998` | Loses `-.00022` from `.66020`. |
| B34 package | `.66026` | Adds `+.00006` from `.66020`. |
| B34 plus B45 | `.66028` | Adds another displayed `+.00002`, conditional on B34. |

Sources: dated public scoreboards in `sum/IT_Today_MineToday_Full_Handoff_2026-08-23.zip`, `sum/MineToday_Full_Chat_Handoff_2026-08-26.zip`, `sum/MineToday_Full_Chat_Handoff_2026-08-27.zip`, `sum/MineToday_Full_Chat_Handoff_2026-08-28_V31_READY.zip`, and `sum/MineToday_Full_Chat_Handoff_2026-09-04_CURRENT.zip`; direct action manifests cited in the lineage table. Where result directories contain only submissions and no score, the handoff score is the evidentiary source.

### New causal removals after the latest handoff

| Submission | Result | Permitted conclusion |
|---|---:|---|
| `slot1_today_066028_minus_S3_membership4.csv` | `.66028` | The four S3 membership actions have low visible public leverage in the current package. Display equality is not proof of exact zero full-test effect. |
| `slot2_today_066028_minus_V30_S2_top1_4.csv` | `.66000` | Removing the four V30 S2 Top1 interventions costs about `.00028`. This is strong package-level evidence that they should remain protected. It does not identify each action's sign. |
| `slot3_today_066028_minus_V25_repeat_M002_M003_B45_3.csv` | `.66028` | The three repeated legacy B45 actions have low visible public leverage. Again, equality at five decimals is not exact causal zero. |

These probes update, rather than replace, earlier evidence. They strengthen the asymmetry between Top1 and low-boundary actions. They do not justify further test-user-by-test-user public ablation: the public leaderboard has already become an adaptive optimization surface.

### Action-state geometry

The latest handoff partitions the 1,000 test users as S0 `897`, S1 `26`, S2 `43`, and S3 `34`. S3 consists of 3 Top1, 4 membership, and 27 order-only changes; those 27 divide into B23 `9`, B34 `10`, and B45 `8` (`sum/MineToday_Full_Chat_Handoff_2026-09-04_CURRENT.zip::MineToday_Full_Chat_Handoff_2026-09-04_CURRENT/03_PUBLIC_LB_AND_CAUSAL_PROBES/CAUSAL_COMPONENT_LEDGER.md`). The exact champion Top5 SHA256 is `7c0d29a7309b7283a043b7d37ca5556ba7c0ca2270b0137a6e1e0878a69a5fe7` (`results/v43/v43_summary.json` through `results/v46/v46/v46_summary.json`). The archive abbreviates the Hybrid34 and `.66020` hashes in some prose; the next executable notebook must recover and record their full values from the underlying submissions rather than treating abbreviations as verification.

## Validation / Leakage / Methodology Audit

### Classification of historical risks

| Risk | Versions/examples | Assessment |
|---|---|---|
| Same-user leakage | No evidence that target rows were deliberately split at message level in the core OOF pipelines. | **No direct evidence of genuine target leakage**, but every future split must remain user-level. |
| Template-family dependence | Sparse models V3.1 onward; exact/paraphrase templates can cross random user folds. | **Validation concern.** This can make folds less independent without leaking labels directly. Grouped template stress tests are required. |
| In-sample stacking | Early V4/V5 mixtures and especially V13.1/V41 second-level models. | **Concern to invalid**, depending on version. V13.1 and V41 efficacy claims should not guide V47. V6 and V41.1 are later corrections. |
| OOF nesting | V13.1 uses second-order features without complete outer nesting; V41 reuses learned predictions before a clean rerun. | **Invalid for causal efficacy.** The direct public/LOCK reversals support the diagnosis. |
| Preprocessing across validation | Label-free TF-IDF vocabulary, tokenizer fitting, or frozen embedding over train/test text. | **Usually harmless transductive preprocessing** if competition rules allow and no labels inform it. It must be declared and separated from label-derived transforms. |
| Target encoding | V31 and related state features. | V31 is described as fold-safe. Any smoothing, category pruning, or feature selection chosen after outcomes still consumes the validation world. |
| TAPT on held-out text | V35. | **Questionable fold isolation**, not target leakage. V35.1 is the fold-pure corrective evidence and fails on template-held-out data. |
| Feature/module selection after validation | Module trust sets in V21.1/V22, action families in V31/V44/V45. | **Adaptive validation contamination.** Later tests of the same family are not independent confirmations unless splits and policy were frozen before inspection. |
| CAL reuse | Repeated from V16 through V42 families. | CAL has become training/development evidence. It is not a final holdout. |
| LOCK reuse | Multiple LOCK variants and repeated opening after related design choices. | Each opened LOCK is burned for descendants influenced by it. “LOCK” is a file role, not a permanent epistemic status. |
| Fresh-salt dependence | V29-V36.1. | Multiple salts improve diagnosis, but V36.1 worlds overlap and are not independent replications. |
| Public LB feedback | V22 onward, especially V24-V26 and V39.5-V45. | **Severe adaptive contamination.** Exact test action discovery is not a reusable scientific policy. |
| User-ID/chronology | Contiguous train/test IDs; chronology worlds often disagree with ordinary folds. | **Supported validation mismatch risk**, not proven leakage. ID must never be a predictive shortcut. |
| Target-derived features | No direct artifact proves use of test targets. Pair/curriculum/TE features use train targets and require nesting. | **No established test-label leakage.** Audit future manifests mechanically. |
| False ensemble independence | Borda, correlated objective variants, same anchor residuals. | **Confirmed dependence.** Agreement is informative only when representation, supervision, or inductive bias truly differs. |

### Trustworthiness of historical evidence

**High enough to guide V47:**

- Exact organizer-file facts and direct result ledgers.
- The public score lineage as package performance on the public subset, with contamination caveats.
- V30's multi-salt OOF evidence plus public gain, because its improvement survives both an offline suite and one public deployment (`results/v30/v30_three_salt_summary.csv`).
- V12/V12.2 as durable champion ancestry.
- The latest negative removal of V30 S2 Top1 actions, as package-level evidence.
- V37 and V41.2 oracle-versus-selector gaps as evidence that candidate supply is not the same as deployable competence.
- V41.1, V42.1, V43.1, and V35.1 negative clean reruns as stronger evidence than their optimistic predecessors.

**Diagnostic only, not sufficient for model selection:**

- V8, V9.1, V19, V21, V28, and V29 gains on particular validation worlds.
- Any bootstrap CI whose action set or threshold was chosen after seeing the same data.
- Flat public scores for a few actions, because score rounding and public subset coverage are unknown.

**Do not use for efficacy claims:**

- V3 semantic stage, because it did not run.
- V13.1 and V41 headline CAL gains, because second-level nesting/dependence is invalid or materially questionable.
- V35's initial TAPT result without the V35.1 corrective evidence.
- V46 DESIGN/LOCK placeholder zeros, because embeddings were unavailable.

### Required validity reset

V47 must treat every historical split, including old LOCKs, as development data. Before looking at new model outputs, freeze:

1. a versioned champion prediction hash and action-protection ledger;
2. a user-level ACTION_TRAIN/DESIGN/LOCK partition;
3. a template-family grouping algorithm that uses no targets;
4. the eligible state, action type, coverage range, thresholds, and tie-breaking rules;
5. the exact feature families and encoder revisions;
6. one primary metric and a short list of diagnostics;
7. a rule that LOCK opens once and public submission occurs only after all gates pass.

All target-derived preprocessing, action labels, calibration, and selection must be fit inside each outer fold. Frozen encoders may be cached across folds because they do not use target labels. No public score may alter V47 thresholds or users.

## Prediction-Behavior Audit

Aggregate NDCG hides the structure that matters. The historical action record is more informative:

| Evidence | Changed behavior | What it says |
|---|---|---|
| V8/V9.1/V19 | Roughly 7-13% Top5 membership, up to 26% ordering, and 1-2.5% Top1 changes | Broad “stable” alternatives can look strong offline and still fail public transfer. Coverage is not authority. |
| V30 | 77 users, including 8 Top1 and 7 membership | A moderate residual can transfer when restrained, but later class-level evidence says some incumbent actions are much more valuable than others. |
| V31 | 60 changes; the 26 new-only changes score poorly, while 34 overlapping V30/V31 changes produce the large gain to `.66006` | Cross-method agreement is useful when it filters actions, not when predictions are blended wholesale. |
| S3 pruning | Removing 27 order-only actions improves `.00014`; removing three Top1 actions loses `.00056` | Action type dominates raw action count. |
| B23/B34/B45 | B23 loses `.00022`; B34 adds `.00006`; conditional B45 adds displayed `.00002` | Adjacent boundaries have distinct signs and cannot be pooled. B45 evidence is conditional on B34. |
| V30 S2 Top1 removal | Four removals lose `.00028` | Incumbent Top1 actions are high leverage and must be protected. |
| S3 membership and legacy V25 B45 removals | Flat at `.66028` display precision | These packages are low priority, not proven exactly neutral. |
| V37/V41.2 oracle | Large oracle gain among eligible challengers, zero deployable learned actions | Headroom resides in selection accuracy, not candidate generation alone. |

### Where disagreements helped

- V30/V31 overlap actions helped as a package. Their agreement filtered out V31's harmful new-only changes.
- V21.1/V22/V25 action ancestry suggests a weak model can surface valuable examples even when its broad deployment fails.
- B34 and B45 became useful only in a protected, surgically defined package.

### Where disagreements hurt

- Broad semantic attacks V16-V21 changed many rankings with split-dependent signs.
- V31 new-only actions, Hybrid34 extras, and S3 order-only actions reduced public performance.
- Membership models repeatedly selected high-leverage but low-precision interventions.
- Confidence, margin, uncertainty, retrieval similarity, and repeated proposal identity did not reliably distinguish beneficial from harmful actions in V29.5-V38.

### Where remaining headroom plausibly lives

**SUPPORTED:** a very small number of S0 Top1 promotions among modules already in the champion's top four, provided a genuinely complementary representation can predict exact action utility. V41.2 and V46 DESIGN geometry show oracle headroom around `.027` on their local partitions, but that is an upper bound with labels, not expected attainable gain.

**SUPPORTED:** protecting existing V30 S2 and S3 Top1 actions. The most decisive recent public removal is negative.

**PLAUSIBLE:** narrow curriculum/readiness information as a veto or regularizer for Top1 action safety. It is not supported as a positive whole-list authority.

**LOW LEVERAGE:** S3 membership and repeated V25 B45 pruning, based on flat displayed removals.

**EMPIRICALLY EXHAUSTED:** generic membership expansion, lower-boundary micro-mining, arbitrary score blending, and broad alternate reranking without a trustworthy defer rule.

## What Consistently Worked

1. **Sparse lexical and template signal.** Word/character TF-IDF and calibrated per-module heads dominate early semantic models and form the durable anchor from V3.1 onward.
2. **Calibration and module specialization.** V12 and V12.2 improve the public score without wholesale representation replacement.
3. **Residual rather than replacement authority.** V30 adds a restrained residual to the anchor and transfers publicly.
4. **Independent agreement used as a filter.** The V30/V31 overlap34 package succeeds even though V31 alone fails.
5. **Sparse causal editing.** V22-V25 and V39.5 show that a handful of actions can matter more than hundreds of broad changes.
6. **Fail-close engineering.** Later versions increasingly preserve exact anchor hashes when gates fail. V46 correctly emits zero actions after a runtime failure, even though the scientific experiment remains unexecuted.

## What Consistently Failed

1. Broad semantic authority, whether frozen, fine-tuned, prototype-based, or late-interaction, has not transferred reliably.
2. Whole-list learning-to-rank objective changes produce small local gains but do not solve split or deployment mismatch.
3. Tail and membership corrections offer large oracle leverage but poor predictive precision.
4. Confidence proxies do not calibrate action utility across validation worlds.
5. More capacity is harmful at this sample size: V43's TabM/MLP partial-list model changes too many users and loses sharply.
6. Generator-state, curriculum, trajectory, and retrieval features have not become reliable positive authority. Their best role remains diagnostic or veto-like.
7. Repeated post-hoc threshold, module, and action-family selection burns validation evidence faster than it creates generalizable knowledge.

## Why the Project Is Stuck

The repository is optimizing an adaptive decision system, not merely a recommender model. The incumbent already captures most stable lexical/template, module-prior, and calibrated ranking signal. Alternative models can rank some missed cases correctly, as large oracle deltas demonstrate, but their confidence is not aligned with **incremental NDCG relative to the champion**. Standard relevance metrics, pair AUC, model margin, and agreement on candidate rank are inadequate surrogates for that incremental action utility.

The second bottleneck is epistemic. More than 46 versions have repeatedly inspected overlapping folds, CALs, LOCKs, action families, and public packages. A delta of `0.0001` on a reused split is no longer evidence of a new hypothesis. The leaderboard target is only `0.00048` above the champion, so noise, split dependence, and display rounding are the same order as the required gain. The project cannot escape by running more variants against the same worlds.

## Plateau Hypothesis Ranking

| Rank | Hypothesis | Status | Evidence-based judgment |
|---:|---|---|---|
| 1 | H3: selection/authority bottleneck | **CONFIRMED** | V37 and V41.2 expose substantial oracle action supply while learned selectors abstain or harm; public gains come from sparse, protected actions. |
| 2 | H4: validation mismatch | **CONFIRMED** | V8, V9.1, V13.1, V19, V28, V35/V35.1, and V41 show ordinary/CAL gains reversing on public, chronology, template, or LOCK evidence. |
| 3 | H2: objective mismatch | **SUPPORTED** | Pairwise learnability drops down-rank; relevance/AUC do not predict exact NDCG action value; Top1 packages have high public leverage. Several ranking objectives were tried, so objective change alone is insufficient. |
| 4 | H8: leaderboard resolution | **CONFIRMED as an interpretation limit** | Multiple small package removals are display-flat. This prevents exact causal attribution but is not the main source of the plateau. |
| 5 | H5: generator/template structure | **SUPPORTED, low remaining leverage** | Extreme template overlap and lexical strength support a latent generator mechanism. V42/V42.1 show only tiny, stress-unstable residual value. |
| 6 | H6: curriculum structure | **REFUTED as the dominant explanation; SUPPORTED as a weak prior** | Catalog prerequisites and V29 oracle audits show structure, but direct curriculum increments are small and membership extrapolation fails in V40. |
| 7 | H7: hidden test shift | **REFUTED as a broad marginal shift; conditional shift remains PLAUSIBLE** | Conditional action utility clearly changes across validation worlds, but adversarial AUC `.509918` and modest marginal block deviations argue against a large observable covariate shift. |
| 8 | H1: hard representation ceiling | **WEAK but not ruled out at the residual scale** | Historical consensus gains show that the observables once contained missed signal, but oracle gains use labels and do not prove remaining learnable information. Repeated failures across diverse representations, template ambiguity, weak K predictability, and noisy lower boundaries keep a residual ceiling plausible. |

## Branches We Should Stop Spending Time On

| Branch | Strongest evidence and versions | Why closure is structural | Evidence required to reopen |
|---|---|---|---|
| Generic whole-list semantic blending | V2, V16-V21, V34-V35.1, V38/V38.1 | Many encoders and adaptation modes share the same failure: useful similarity does not calibrate incremental action utility. | A preregistered fresh-split risk-coverage curve showing positive exact action utility, not just relevance/AUC. |
| Another broad LambdaRank/XE/YetiRank stage | V4-V6, V21.2, V28-V32, V42 | Objective diversity changes scores but not transfer; candidate errors remain correlated with the anchor. | A formulation with different supervision or authority, plus positive grouped/contiguous/LOCK evidence. |
| Generic membership/K prediction | V8-V10.1, V14, V28, V38.1, V40/V41 | High action leverage is dominated by poor boundary learnability and harmful false positives. | Stable B56 or K calibration across untouched template and ID blocks with a positive lower CI. |
| Lower-boundary micro-mining | V10.1, V14, V39.5, V44/V45 | B45 AUC is near chance; many order actions were profitably removed; latest legacy B45 removal is display-flat. | New, non-public labels or a mechanism tied to stable curriculum evidence. |
| Arbitrary embedding or objective ensembles | Borda, V20, V32 | Shared anchor/data makes apparent diversity highly dependent; blending broad predictions dilutes the incumbent. | Demonstrated low action-error correlation and positive consensus utility on an untouched split. |
| Confidence/margin/retrieval stability as a generic selector | V15, V29.5, V33-V38 | The proxies do not transport or collapse to zero coverage. | Properly cross-fitted action-utility calibration with a stable risk-coverage frontier. |
| Generator-state as positive ranking authority | V42/V42.1 | After geometry fixes, gains remain tiny and Top1 contribution is negative. | A directly measured generator-family target entropy that predicts transferable Top1 utility. |
| High-capacity residual TabM/MLP | V43/V43.1 | 4,000 users and 17 items do not support wholesale joint replacement; even a trust region is nonpositive. | Substantially more labeled users or a much narrower target with external pretraining. |
| Coarse pruning of V30 S2 Top1 | V44 plus latest public removal | The public package loses `.00028` when removed. | Private/full-test labels demonstrating that loss was public-subset noise. |
| More public micro-ablation of S3 membership or legacy V25 B45 | V45 plus two latest flat removals | Expected visible leverage is low and the public set is already contaminated. | Full test labels or organizer-provided causal evaluation, not more adaptive submissions. |

## What Still Looks Alive

| Rank | Signal family | Why it remains alive | Expected leverage |
|---:|---|---|---|
| 1 | Protect incumbent Top1 actions | Removing four V30 S2 Top1 actions costs about `.00028`; removing three S3 Top1 actions previously cost about `.00056`. | High per action; defensive rather than additive. |
| 2 | New, S0-only Top1 action competence | Oracle headroom persists and V46 never tested E5/Nomic. Top1 has more leverage and supervision than lower boundaries. | Medium to high if precision exceeds the incumbent; likely only a few actions. |
| 3 | V30/V31-style independent consensus | Overlap34 is the strongest later package gain, while new-only actions fail. | Medium, but only with genuinely different errors. |
| 4 | Frozen sparse/calibrated/V30 champion | It is the only evidence-backed base and includes publicly protected actions. | High as foundation; do not retrain casually. |
| 5 | Curriculum/readiness as a safety veto | Handcrafted structure explains a minority of missed mass and can reject implausible promotions. | Low to medium; not supported as positive authority. |
| 6 | Validation redesign | It does not directly score points, but without it all deltas near `.0005` are uninterpretable. | Highest information value. |

## Representation and Embedding Coverage Audit

### Already tested or materially represented

| Family | Where tested | Research conclusion |
|---|---|---|
| Word and character TF-IDF, SVD, Ridge, per-module boosted models | V1, V3.1 onward | Core durable signal; part of the champion ancestry. |
| Template memory, message maxima, topic/intent lexicons | V1, V4, V18, V33 | Useful inside the sparse anchor; no stable residual authority as a separate expert. |
| Multilingual GTE bi-encoder | V2 | Supervised/adapted representation improves OOF but not public transfer. |
| GTE cross-encoder/reranker | V3 | Nominally attempted but **not executed** due FP16 training failure. |
| Qwen embedding/reranking and Qwen8B semantic features | V1, V16-V21, V41/V41.1/V41.2 | Many formulations tested, including frozen, QLoRA, ordinal, and Bradley-Terry heads; no stable broad authority. |
| F2LLM-4B | V23 | Exact action-utility formulation tested; essentially zero stable gain. |
| BGE-M3 dense, sparse, and ColBERT-style late interaction | V20 | Late interaction is not an untested family. This implementation failed novelty/LOCK gates. |
| BGE cross-encoder reranker | V34 | Weak benefit AUC and zero approved actions. |
| IndoBERT with TAPT | V35/V35.1 | Initial positive result disappears on fold-pure/template-held-out confirmation. |
| XLM-R | V38/V38.1/V41 | Direct pair, curriculum, and stacked residual formulations tested; no robust authority. |
| Generator-state representations | V42/V42.1 | Explicitly tested and corrected; only tiny unstable residual value. |
| Handcrafted curriculum, prerequisite, co-occurrence, readiness | V29-V30, V38.1, V40, V42 | Materially represented. Direct incremental value is small; unsafe for membership authority. |
| Handcrafted trajectory/state summaries | V39/V39.1 | Represented, but not as a learned sequence encoder. |
| TabM/MLP partial-list model | V43/V43.1 | High-capacity tabular/listwise replacement tested and closed. |

### Actually untested or not successfully executed

| Rank by novelty x complementarity x feasibility x information gain | Family | Exact status | Why its expected errors might differ |
|---:|---|---|---|
| 1 | Multilingual E5 plus Nomic-v2-MoE as independent Top1 action experts | Both are named and instantiated in V46, but no embeddings were produced. **Empirically untested.** | Their contrastive pretraining and pooling differ from the champion's sparse/V30 residual. Nomic's sparse mixture-of-experts routing may differ from E5, although both inherit XLM-R-like multilingual foundations, so independence must be measured rather than assumed. |
| 2 | Representation disagreement measured as action competence rather than blended score | V31 uses model agreement and V46 planned dual consensus, but V46 did not execute. **Partially attempted, exact test unrun.** | It asks whether independent encoders agree on the same counterfactual improvement, not whether an average embedding improves relevance. |
| 3 | Learned sequence representation of chat state transitions | V39/V39.1 use deterministic trajectory features only. **Structurally untested.** | A causal/time-aware encoder could distinguish exploration from final commitment, unlike concatenation or pooled prototypes. The absence of outcome sequences limits feasibility. |
| 4 | Learned module/prerequisite graph representation | Handcrafted graph distance and priors were tested; no graph neural or probabilistic latent curriculum model is evidenced. **Untested implementation, partially tested hypothesis.** | It could share statistical strength across rare modules, but only 17 nodes and weak direct-curriculum increments limit likely gain. |
| 5 | Narrow partitioned-preference learner with frozen-anchor authority | V43 uses a broad partial Plackett-Luce-like joint model and V41.2 uses a Top1 Bradley-Terry head. **Narrow residual factorization untested.** | It can learn Top1-versus-rest and positive-versus-zero partitions without regressing noisy values, while touching only candidate actions. |
| 6 | Supervised Top1 metric learning | V20/V21 adapt semantics broadly and V41.2 trains a preference head, so the concept is **substantially overlapping, not wholly new**. | Any remaining novelty would have to be a contrastive loss tied directly to incumbent challenger pairs, not a different checkpoint. High implementation ambiguity makes it lower priority. |

Multilingual E5 and Nomic are not recommended because they are newer checkpoints. They are recommended only for one controlled competence test because the repository contains no completed predictions from either and because V46's proposed comparison was never run. If their exact-action error correlation is high or their risk-coverage curve is nonpositive, the entire branch should close.

## Literature Review

The literature was filtered against MineToday's actual bottleneck. Methods that require logged propensities, online interventions, long course-consumption sequences, or large item graphs were not retained merely because they are popular recommender methods.

### Partial and top-k preference learning

[Ma et al., “Learning-to-Rank with Partitioned Preference,” AISTATS 2021](https://proceedings.mlr.press/v130/ma21a.html) derives efficient Plackett-Luce learning when supervision specifies ordered partitions rather than a complete permutation. MineToday naturally supplies a unique Top1, graded positive bands, and a tied zero tail. The closest historical test is V43's broad partial Plackett-Luce-style TabM/MLP, which failed because it replaced too much of the champion. The non-overlapping opportunity is a low-capacity residual head trained only on incumbent-versus-challenger actions. It is feasible on Kaggle T4x2 and rule-compatible because it uses organizer labels and ordinary supervised training.

[Xia et al., “Listwise Approach to Learning to Rank: Theory and Algorithm,” ICML 2008](https://mlanthology.org/icml/2008/xia2008icml-listwise/) introduced ListMLE over permutations. It supports the distinction between list-level order and pointwise relevance, but MineToday's incomplete/tied tail makes unmodified ListMLE a poor fit. V4-V6, V28-V32, V42, and V43 already cover much of the generic listwise hypothesis. Literature therefore argues for a narrower partition-aware objective, not another whole-list run.

[Wang et al., “The LambdaLoss Framework for Ranking Metric Optimization,” CIKM 2018](https://research.google/pubs/the-lambdaloss-framework-for-ranking-metric-optimization/) unifies metric-driven LambdaRank-style losses. MineToday has already used LambdaRank/XENDCG/YetiRank repeatedly. The paper validates metric weighting conceptually but does not create a research gap here. A new LambdaLoss implementation without new supervision or authority would be checkpoint-equivalent experimentation.

### Selective prediction and deferral

[Geifman and El-Yaniv, “SelectiveNet: A Deep Neural Network with an Integrated Reject Option,” ICML 2019](https://proceedings.mlr.press/v97/geifman19a.html) formalizes risk-coverage tradeoffs rather than forcing a prediction on every case. [Madras, Pitassi, and Zemel, “Predict Responsibly: Improving Fairness and Accuracy by Learning to Defer,” NeurIPS 2018](https://papers.nips.cc/paper_files/paper/2018/hash/09d37c08f7b129e96277388757530c72-Abstract.html) models a system choosing between decision makers. MineToday's analogue is explicit: the new expert should defer to the champion unless expected incremental NDCG is positive at a preregistered coverage.

The closest MineToday experiments are V13/V13.1 routers, V15, V29.5, V34, and V36-V37. Thus “learning to defer” itself is not new. The retained novelty is narrower: fit and evaluate **counterfactual action utility**, report the complete risk-coverage frontier on grouped out-of-fold predictions, and allow only S0 Top1 actions from a representation that the repository has not successfully executed. T4x2 feasibility is high because the selector is linear; rule compatibility is straightforward.

### Curriculum and prerequisite structure

[Pan et al., “Prerequisite Relation Learning for Concepts in MOOCs,” ACL 2017](https://aclanthology.org/P17-1133/) learns directed prerequisite relations from content and course structure. [Jia et al., “Learning Concept Prerequisite Relations from Educational Data via Multi-Head Attention Variational Graph Auto-Encoders,” NAACL 2021](https://aclanthology.org/2021.naacl-main.164/) uses heterogeneous graph signals. These mechanisms map to MineToday's catalog graph and could regularize rare modules such as M004/M011.

The closest MineToday work is V29/V29.5/V30, V38.1, and V40. The non-overlapping novelty would be learned uncertainty over edges and personalized deviations rather than fixed distance/co-occurrence features. Evidence against it is substantial: direct curriculum increments are small and V40's membership policy is harmful. A 17-node graph is trivial on T4x2 and rule-compatible, but this is a lower-priority diagnostic/veto direction, not the primary architecture.

### Sequential user state

[Piech et al., “Deep Knowledge Tracing,” NeurIPS 2015](https://proceedings.neurips.cc/paper_files/paper/2015/hash/bac9162b47c56fc8a4d2a519803d51b3-Abstract.html), [Kang and McAuley, “Self-Attentive Sequential Recommendation,” ICDM 2018](https://arxiv.org/abs/1808.09781), and [Sun et al., “BERT4Rec,” CIKM 2019](https://arxiv.org/abs/1904.06690) show how ordered interactions can define evolving user state. MineToday, however, provides chat utterances and one final relevance vector, not repeated course interactions, correctness events, or exposure logs. V39/V39.1 are the closest historical trajectory tests.

A learned temporal encoder is therefore scientifically distinct from chat concatenation, but the supervision is weak and the analogy to knowledge tracing is incomplete. It is feasible on T4x2 only as a small frozen text encoder plus lightweight sequence head. An organizer-data-only implementation with pinned open weights is compatible with the stated rules, but direct LLM generation of test rankings remains excluded. Directly importing DKT, SASRec, or BERT4Rec would be method-name shopping and is not recommended until the transition EDA shows incremental information over the latest message.

### Multilingual and late-interaction representations

[Wang et al., “Multilingual E5 Text Embeddings: A Technical Report,” 2024](https://arxiv.org/abs/2402.05672) describes multilingual contrastive training; the official [`multilingual-e5-large-instruct` model card](https://huggingface.co/intfloat/multilingual-e5-large-instruct) specifies a roughly 0.6B-parameter, 1024-dimensional, 512-token model with instruction-formatted queries. [Nussbaum and Duderstadt, “Training Sparse Mixture Of Experts Text Embedding Models,” 2025](https://arxiv.org/abs/2502.07972) and the [official Nomic v2 model card](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe) describe a multilingual mixture-of-experts embedder with 475M total and 305M active parameters, 768-dimensional output, 512-token context, query/document prefixes, and Matryoshka truncation support.

Both are feasible as frozen sequential inference on T4x2. E5 is MIT-licensed and Nomic v2 is Apache-2.0 according to their model cards; exact revisions, dependencies, and `trust_remote_code` must be pinned and archived. Frozen embedding extraction plus organizer-label supervised utility heads is compatible with the stated competition rule allowing open-source pretrained models and does not constitute prohibited direct LLM generation of test answers.

[Chen et al., “BGE M3-Embedding,” 2024](https://arxiv.org/abs/2402.03216) combines dense, sparse, and multi-vector retrieval, while [Khattab and Zaharia, “ColBERT,” SIGIR 2020](https://arxiv.org/abs/2004.12832) supplies token-level late interaction. V20 already evaluates BGE-M3 dense/sparse/ColBERT-style signals. Late interaction must therefore be listed as tested, not offered as a missing idea.

### Literature conclusion

The literature supports three changes in research practice: model partial-order supervision explicitly, evaluate a defer system by risk-coverage rather than aggregate relevance metrics, and distinguish temporal intent state from generic sequence recommendation. It does not support another broad reranker, an off-policy/counterfactual recommender without logged exposure propensities, or a graph model solely because the catalog has prerequisite text.

## Research Gaps Not Yet Tested

| Gap | Equivalent attempt? | Failure addressed and non-overlapping novelty | Why errors could differ | Plausible leverage | Falsifier |
|---|---|---|---|---|---|
| Dual E5/Nomic exact Top1 action competence | V46 planned it but failed before embeddings; no equivalent completed result. V31 consensus uses different experts. | Addresses V16-V21 broad semantic harm and V37 selector failure by supervising exact delta against a frozen champion, S0 only, Top1 only. | Contrastive multilingual encoders differ from sparse/V30 features; MoE routing may differ from E5. Their mutual correlation is unknown. | Medium to high at low coverage; enough to target `.0005` if a few actions are precise. | Either encoder has nonpositive DESIGN risk-coverage, consensus does not outperform singles, or errors are nearly identical. |
| Prospective action-level defer evaluation | Routers/selectors exist in V13-V15 and V29.5-V37. | Novelty is the preregistered action target, full risk-coverage curve, template-grouped evaluation, one-shot LOCK, and prohibition on post-hoc module/action mining. | It changes the decision rule and epistemic process, not representation. | Indirect but essential; can prevent another false positive. | A preregistered policy still reverses across grouped, ID-block, and LOCK worlds or has zero useful coverage. |
| Narrow partitioned-preference residual | V43 broad PL-like joint model and V41.2 Bradley-Terry are closest. | Keeps the champion fixed and learns only Top1 challenger utility from ordered partitions, avoiding noisy exact relevance and wholesale reranking. | Label factorization differs from pointwise residuals while authority stays sparse. | Medium if objective mismatch is causal; otherwise low. | No gain over ordinary delta regression at matched features/coverage on all fresh worlds. |
| Learned curriculum uncertainty as a veto | Handcrafted curriculum was tested in V29/V30/V38.1/V40; positive graph authority failed. | Learns uncertainty and personalized deviations, then only rejects unsafe promotions. | It can catch prerequisite violations without trying to infer preference from the graph alone. | Low to medium, mainly harm reduction. | Veto removes as many beneficial as harmful actions or edges are not stable across template/ID strata. |
| Learned chat-state transitions | V39/V39.1 use aggregate trajectory features; no learned temporal representation is evidenced. | Tests whether ordered conversational change adds Top1 utility beyond latest/full-history pooling. | Errors could differ on users whose early exploration conflicts with final intent. | Medium in theory, low prior until EDA. | Conditional mutual information and prospective delta over latest-message controls are negligible. |
| Hierarchical partial pooling of action utility | Module trust and subgroup mining exist, but no single Bayesian/mixed-effects utility model is evidenced. | Shares strength across module pairs/template families while shrinking rare strata and exposing posterior uncertainty. | Could reduce the variance that made local subgroup gains reverse globally. | Low to medium; structurally simple. | Posterior lower-bound selection does not improve risk-coverage over regularized Ridge or collapses to zero coverage. |

## Primary V47 Architecture

**Name:** Frozen-Champion Dual-Encoder Top1 Utility Gate.

This is not a continuation of V46 by default. It is selected independently from the full audit because: (a) Top1 is the only action family with repeated high public leverage; (b) V37/V41.2 show candidate oracle supply but no trustworthy selector; (c) E5 and Nomic are the only high-ranked representation families named in the repository but never successfully evaluated; and (d) a frozen low-authority experiment has a single interpretable failure mode. V47 removes V46's generator veto and prevents its structural head from voting, because V42/V42.1 do not support generator authority and adding three gates would confound the representation test.

### Research Hypothesis

**HYPOTHESIS.** Among S0 users, multilingual E5 and Nomic-v2-MoE contain complementary semantic evidence that predicts the exact NDCG gain of promoting a champion rank-2, rank-3, or rank-4 module to rank 1. When both encoders independently choose the same challenger and each predicts positive utility, a very-low-coverage policy will improve the frozen champion on template-grouped DESIGN and one-shot LOCK data.

The falsifier is explicit: if either encoder lacks a positive action-level risk-coverage frontier, their selected-action errors are highly correlated without positive consensus lift, the policy has no coverage, or DESIGN/LOCK signs disagree, then the dual-encoder Top1 branch closes. A tokenizer failure is an implementation failure and must be repaired before this scientific test is classified.

### Architecture

1. **Frozen incumbent.** Load the exact `0.66028` champion prediction artifact and verify its full-file and Top5 hashes against the current ledger. Freeze all scores, state assignments, and public-proven actions. No champion retraining or recalibration is part of V47.
2. **Eligibility.** Only S0 users are eligible. S1, S2, S3, V25 legacy actions, V30/V31 consensus actions, and every publicly probed user are protected. Each eligible user's candidate set is the champion's ranks 2-4.
3. **Permitted action.** At most one candidate may be promoted to rank 1. The other modules retain their relative order and exact score multiset. Membership is unchanged. No B23/B34/B45 action is permitted.
4. **Common deterministic text view.** Construct one pinned user text from the same organizer chat/assessment fields and one pinned module document from catalog text. Use the same normalization, administrative-message handling, recency order, and 512-token truncation policy for both encoders. This prevents text construction from becoming a second hypothesis.
5. **Frozen semantic experts.** Extract E5 and Nomic embeddings separately. For each challenger, derive candidate-minus-incumbent similarity, absolute similarities, user/module interaction features, and the champion margin. Do not average the embeddings or scores.
6. **Action heads.** Fit one strongly regularized Ridge utility head per encoder inside the outer folds, using a small preregistered alpha grid selected only inside ACTION_TRAIN. The structural features allowed in each head are limited to the same frozen champion margin, candidate/incumbent identities, and readiness difference. Existing Qwen/structural predictions are evaluated as controls but cannot authorize an action.
7. **Consensus.** Each expert independently selects its highest-utility candidate. The gate acts only if both select the same module, both calibrated utility lower estimates are positive, the action lies inside the preregistered coverage budget, and no simple catalog prerequisite impossibility is triggered. There is no learned meta-router.
8. **Audit output.** Record every eligible proposal, both expert utilities/ranks, encoder agreement, incumbent margin, module pair, state, template group, assessment readiness, exact realized OOF delta, and final decision. This makes zero-action and negative outcomes scientifically useful.

### Supervision

For labeled user (u), frozen champion ranking (r_u), and challenger (c) currently at rank 2-4, define:

\[
\Delta_{u,c}=\operatorname{NDCG@5}(y_u,\operatorname{promote}_{1}(r_u,c))-\operatorname{NDCG@5}(y_u,r_u).
\]

The primary target is this exact counterfactual delta, not raw relevance, Top1 correctness, pair AUC, or the challenger's semantic similarity. Both positive and harmful actions are retained. Sample weighting may reflect the absolute NDCG consequence, but its formula must be fixed before DESIGN. Because the candidate operation is deterministic and labels are fully observed for train users, this is ordinary supervised counterfactual replay, not causal off-policy learning.

The sign of the realized delta supplies benefit/tie/harm diagnostics without fitting a second model. Alternative classification or partitioned-preference objectives are deferred to V48 so V47 changes one main factor: representation competence.

### Validation

All historical CAL and opened LOCK splits are development data. V47 uses a prospective, target-blind split manifest created and hashed before encoder inference, for example 2,600 ACTION_TRAIN, 800 DESIGN, and 600 LOCK users. The allocation must be user-level, approximately stratified by observed K and Top1 module, grouped by target-blind latest-template family where feasible, and audited against contiguous ID blocks. Because all 4,000 labels have influenced prior human research, this is not a pristine external holdout; it is a prospective policy test under an explicit contamination limitation.

Within ACTION_TRAIN:

- use nested user-level cross-fitting for each utility head;
- fit every target-derived transform inside the outer training fold;
- report the risk-coverage curve at predeclared user coverages of 0.25%, 0.5%, 1.0%, and 1.5%;
- compare E5, Nomic, consensus, and the frozen structural/Qwen control on the identical action pool;
- compute cluster bootstrap intervals by template family, plus per-fold, contiguous-ID block, module-pair, and action-harm diagnostics;
- measure selected-action overlap and realized-delta correlation between encoders. Agreement is not assumed independent.

After model form, regularization, utility threshold, and tie-breaking are frozen, open DESIGN. A defensible DESIGN gate is:

- total NDCG delta at least `+0.0008` versus the frozen champion;
- cluster-bootstrap lower bound above zero, or above a preregistered negligible tolerance if discreteness prevents a strict bound;
- nonnegative direction in at least 4/5 fixed folds and no major template/ID block below `-0.0002`;
- at least twice as many beneficial as harmful selected actions;
- consensus improves utility at matched coverage over each single expert and the structural control;
- projected test coverage between 3 and 15 users.

The `+0.0008` offline gate is intentionally above the approximate `.00048` current public gap. V30 transferred roughly `.00044` public from `.000732` pooled offline, so an offline effect merely equal to the leaderboard gap is not adequate. This is a heuristic safety margin, not a guaranteed transfer equation.

LOCK opens once, only after the code, environment, hashes, and thresholds are frozen. Its required sign is positive, its pooled DESIGN+LOCK cluster-bootstrap lower bound must exceed zero, and no new action-family/module exception may be introduced after viewing it. If LOCK fails, V47 stops without test inference or submission.

### Deployment Policy

- Refit only the two predeclared linear heads on ACTION_TRAIN plus DESIGN after LOCK passes; encoder weights remain frozen.
- Apply the identical consensus gate to test S0 users.
- If more than 15 actions qualify, retain the 15 highest minimum-of-two predicted utilities. If fewer than 3 qualify, do not submit because public resolution and selection uncertainty dominate.
- Never inspect or hand-edit test users. Produce one immutable action ledger and one candidate submission only in the future implementation phase.
- One Kaggle submission is allowed if and only if every offline gate passes. Its result validates the package, not individual actions.

### Failure Modes

| Failure | Meaning | Response |
|---|---|---|
| Tokenizer/model load fails | Implementation/runtime failure | Fix environment and rerun the unchanged preflight. Do not score the hypothesis. |
| Champion/hash reproduction fails | Provenance failure | Abort and repair reconstruction; no semantic work. |
| E5 or Nomic has no positive risk-coverage | Representation competence failure | Close that expert and the required dual-consensus hypothesis. Do not substitute another checkpoint post hoc. |
| Experts agree often but consensus utility is not better | Correlated-error failure | Close arbitrary dual-encoder consensus. |
| Experts are individually useful but rarely agree | Overly strict deployment policy or complementary candidates | Record the result; a different defer rule requires a new preregistered experiment, not an in-run adjustment. |
| DESIGN positive, LOCK negative | Validation transport failure | Stop without test. Do not tune on LOCK. |
| Zero selected actions | Coverage failure | Valid negative result for this gate, unlike V46's runtime-generated zero. |
| Public score flat after all gates | Package effect below display/public-subset resolution or offline transfer failure | Record once; do not start user-level public probing. |

### T4x2 Feasibility

The architecture is feasible. E5 is roughly 0.6B parameters and Nomic v2 MoE has 475M total with 305M active parameters, both with 512-token limits. Run frozen inference sequentially on one GPU or data-parallel across two T4s, cache only organizer-data embeddings, and train small CPU heads. No encoder fine-tuning, generative decoding, graph training, or broad hyperparameter sweep is required. The expected class is medium, plausibly within one Kaggle session if models and dependencies are cached. Pin model revisions, `sentence-transformers`, `transformers`, `sentencepiece`, `tiktoken`, and Nomic remote-code revision; test both tokenizers before any expensive stage.

### Decisive Experiment

The minimal scientific experiment is not a full submission. After an environment-only smoke test, produce cross-fitted ACTION_TRAIN and blind DESIGN action tables for E5, Nomic, and their exact-candidate consensus on the same frozen S0 rank-2-to-rank-4 pool. Compare realized exact-delta risk-coverage and error correlation. Do not open LOCK or generate test embeddings unless the DESIGN gate passes. This single result answers whether the missing encoders add competent, complementary Top1 information. A negative result closes the most plausible unexecuted representation branch at modest cost.

## Alternative Research Directions

| Priority | Direction | Hypothesis | Novelty vs V1-V46 | Evidence for | Evidence against | Upside | Information gain | Complexity | T4x2 |
|---:|---|---|---|---|---|---|---|---|---|
| 1 | Hierarchically shrunk action-utility gate over existing features | Partial pooling by module pair/template can identify safe S0 Top1 actions without a new encoder. | V21.1 module trust and later subgroup mining are post-hoc; a single preregistered mixed-effects model with shrinkage is not evidenced. | Sparse actions and overlap consensus work; local effects recur by pair. | V29.5/V36/V37 selectors fail and local gains reverse globally. | Low-Medium | High | **Low** | Yes |
| 2 | Narrow partitioned-preference residual | Modeling Top1, ordered positive partitions, and zero tail will improve challenger ordering at fixed authority. | V43 broadly replaces the list; this version only scores S0 Top1 incumbent/challenger actions. | Target geometry is partial-order-like; literature supports partitioned preference. | V41.2 BT selects zero and V43/43.1 fail. | Medium | High | Medium | Yes |
| 3 | Learned curriculum uncertainty veto | A probabilistic prerequisite model can remove unsafe semantic promotions without choosing recommendations itself. | Direct curriculum features exist; learned edge uncertainty used only as veto does not. | Catalog graph and readiness have real structure; rare modules may benefit from sharing. | Direct V30 increment is tiny, V40 is harmful, only 17 nodes. | Low | Medium | Medium | Yes |
| 4 | Temporal chat-state encoder | Ordered intent changes predict Top1 deviations beyond latest/full-history pooling. | V39/V39.1 use handcrafted aggregates, not learned temporal state. | Multiple timestamped messages and possible exploration-to-commitment transitions. | One final label per user, no learning-event sequence, trajectory attempts produce no authority. | Medium | Medium-High if EDA positive | High | Maybe |

The simple priority-1 alternative is preferable to a large new model if V47 shows that representation competence exists but consensus calibration is the failure. It should not be run in parallel with V47 because doing so would spend the same prospective validation world on two policy searches.

## Prioritized V47+ Experiment Roadmap

### Experiment 0: V47 provenance, split, and dependency preflight

| Field | Specification |
|---|---|
| Research question | Can the exact champion, protected-state ledger, prospective split, and both encoder environments be reproduced before any scientific evaluation? |
| Hypothesis | The V46 failure is an isolated dependency problem, not an unavailable model or irreproducible anchor. |
| Baseline | Exact `0.66028` champion and V46 reproduction audit. |
| Single main change | Repair and pin only the tokenizer/model-loading environment. |
| Frozen components | Champion scores/order, state/action ledger, text formatter, model revisions, split manifest. |
| Required artifacts | Full SHA256 hashes, package lock/version log, split manifest, tokenizer smoke log, one fixed dummy-text embedding hash per encoder. |
| Validation protocol | Deterministic rerun of hash and shape checks only; no labels or metric evaluation. |
| Leakage safeguards | Do not inspect DESIGN/LOCK labels or run test inference; smoke text contains no competition user. |
| Primary metric | Exact hash equality and 100% successful finite embeddings on the smoke set. |
| Secondary diagnostics | Token lengths, truncation flags, dtype, device, repeatability. |
| Behavior to log | Every mismatch and fail-close decision. |
| Success threshold | All hashes exact, both models load, repeat embeddings agree within pinned numerical tolerance. |
| Failure threshold | Any unresolved anchor mismatch, missing model revision, tokenizer exception, or nondeterministic output. |
| Information if positive | V46 was an implementation non-execution and V47 can test the hypothesis. |
| Information if negative | Scientific testing is blocked by provenance/runtime, not model quality. |
| Expected runtime | Low, minutes on T4x2. |
| Kaggle submission | No. |

### Experiment 1: V47 independent encoder competence on ACTION_TRAIN and DESIGN

| Field | Specification |
|---|---|
| Research question | Does either missing encoder rank exact beneficial S0 Top1 actions better than existing controls? |
| Hypothesis | E5 and Nomic each have a positive low-coverage exact-delta frontier, and at least one exceeds the structural/Qwen control. |
| Baseline | Frozen champion plus no action; controls use identical linear heads over existing structural/Qwen features. |
| Single main change | Add frozen E5 and Nomic representations, separately. |
| Frozen components | Champion, eligible users/candidates, text view, action definition, head family, regularization grid, folds, coverage points. |
| Required artifacts | Per-action OOF table, per-encoder feature manifest, fold models, risk-coverage table, correlation/overlap matrix, DESIGN ledger. |
| Validation protocol | Nested user-level ACTION_TRAIN cross-fitting, then one blind DESIGN evaluation; template clusters and contiguous ID blocks are diagnostics. |
| Leakage safeguards | All delta labels and scaling fit inside outer folds; no encoder fine-tuning; no threshold/module selection after DESIGN. |
| Primary metric | Exact NDCG delta at preregistered low coverage versus frozen champion. |
| Secondary diagnostics | Benefit/harm precision, mean regret, calibration, Top1 component, per-pair/block deltas, selected error correlation. |
| Behavior to log | Candidate rank, chosen module, incumbent margin, each utility, action overlap, realized delta, template/state/pair slices. |
| Success threshold | At least one encoder has DESIGN delta `>= +0.0008`, positive clustered lower bound or preregistered tolerance, at least 4/5 fold directions, and benefit:harm `>=2:1`. |
| Failure threshold | Both encoder frontiers are nonpositive, unstable by major block, or indistinguishable from control at matched coverage. |
| Information if positive | A genuinely new representation contains incremental Top1 action signal. |
| Information if negative | Close E5/Nomic and further checkpoint shopping; representation is not the missing lever. |
| Expected runtime | Medium, frozen inference plus low-cost heads within one T4x2 session. |
| Kaggle submission | No. |

### Experiment 2: V47 consensus and one-shot LOCK

| Field | Specification |
|---|---|
| Research question | Does exact-candidate agreement convert individual encoder competence into transportable authority? |
| Hypothesis | Minimum-of-two utility plus same-challenger agreement reduces harmful actions enough to preserve positive delta on LOCK. |
| Baseline | Better single encoder at the exact same coverage and the frozen champion. |
| Single main change | Apply the predeclared dual-consensus defer rule. |
| Frozen components | All Experiment 1 features/heads, candidate pool, thresholds, coverage cap, catalog veto, tie-breaking. |
| Required artifacts | Frozen policy JSON, code/environment hash, DESIGN and LOCK ledgers, fold/block/bootstrap reports, no-action reproduction hash. |
| Validation protocol | Freeze after DESIGN, open LOCK exactly once, report without threshold adjustment. |
| Leakage safeguards | LOCK cannot choose module pairs, thresholds, model weights, or coverage; no public submission yet. |
| Primary metric | LOCK exact NDCG delta versus champion and pooled prospective DESIGN+LOCK delta. |
| Secondary diagnostics | Consensus lift over single experts, harm count/severity, coverage, pair/state concentration, block signs. |
| Behavior to log | All agreed and rejected actions, disagreement reason, minimum utility, realized delta. |
| Success threshold | LOCK delta positive, pooled delta `>= +0.0008` with clustered lower bound above zero, no major block below `-0.0002`, projected test actions 3-15. |
| Failure threshold | Negative LOCK, pooled lower bound nonpositive, consensus no better than a single expert, or zero/excessive coverage. |
| Information if positive | Independent semantic agreement is a viable authority mechanism. |
| Information if negative | Close this consensus rule; do not add a meta-router on the same LOCK. |
| Expected runtime | Low after embeddings; linear refits and audits. |
| Kaggle submission | Only Experiment 3 may submit after this gate passes. |

### Experiment 3: V47 single immutable public deployment

| Field | Specification |
|---|---|
| Research question | Does the prospectively validated Top1 consensus package transfer to the public test subset? |
| Hypothesis | A 3-15 action package will improve the exact `.66028` champion. |
| Baseline | Byte-verified champion submission. |
| Single main change | Apply only the locked S0 Top1 actions from Experiment 2. |
| Frozen components | Everything, including action count cap and ordering implementation. |
| Required artifacts | Candidate submission, exact diff ledger, score-multiset audit, protected-user audit, SHA256 hashes. |
| Validation protocol | Pre-submit offline reproduction only; one public submission and no follow-up user ablations. |
| Leakage safeguards | No manual test inspection or editing; public result cannot alter V47. |
| Primary metric | Public NDCG delta versus `.66028`. |
| Secondary diagnostics | Exact changed-user/action count and preservation of all nonselected rows. |
| Behavior to log | Every action and invariant, never inferred per-action public utility. |
| Success threshold | Displayed score `> .66028`; practical target `>= .66076`. |
| Failure threshold | Score `<= .66028`, hash/invariant failure, or fewer than 3 qualifying actions before submission. |
| Information if positive | The prospectively gated package transfers, but only as a package. |
| Information if negative | Offline-to-public mismatch remains; stop public decomposition and redesign validation. |
| Expected runtime | Low; inference already complete. |
| Kaggle submission | Yes, exactly one if all prior gates pass. |

### Experiment 4: V48 matched partitioned-preference ablation

| Field | Specification |
|---|---|
| Research question | Is exact-delta regression the wrong supervision even when representation and authority are fixed? |
| Hypothesis | A low-capacity partitioned-preference/Bradley-Terry residual improves S0 Top1 action ordering at matched coverage. |
| Baseline | Better V47 single expert and consensus features/policy. |
| Single main change | Replace the action head objective only; embeddings and authority stay fixed. |
| Frozen components | Champion, candidates, encoders, text, splits, feature set, coverage, deployment rule. |
| Required artifacts | Matched OOF predictions, loss definition, action ledgers, risk-coverage difference table. |
| Validation protocol | New prospective split or nested cross-validation decided before viewing V47 LOCK; do not reuse V47 LOCK for tuning. |
| Leakage safeguards | Partition construction and all parameters fold-internal; no exact-value threshold mining. |
| Primary metric | Paired change in exact NDCG delta at matched action coverage. |
| Secondary diagnostics | Pairwise calibration, Top1 recovery, harm-weighted precision, block consistency. |
| Behavior to log | Which actions change solely because of the objective and their realized utility. |
| Success threshold | Paired pooled improvement `>= +0.0003`, positive lower bound, no loss in worst block. |
| Failure threshold | No positive paired improvement or changed actions merely increase coverage/harm. |
| Information if positive | Objective mismatch matters after representation competence is established. |
| Information if negative | Close narrow preference-loss work; selector/data, not loss, is limiting. |
| Expected runtime | Low-Medium; no new embedding inference. |
| Kaggle submission | Only if it independently passes a new locked gate and changes at least 3 test actions. |

### Experiment 5: V49 conditional structure branch

| Field | Specification |
|---|---|
| Research question | Does curriculum uncertainty or temporal state add residual safety after V47/V48? |
| Hypothesis | Exactly one EDA-supported structural signal improves action harm detection without becoming positive authority. |
| Baseline | Best prospectively valid V47/V48 gate. |
| Single main change | Add either a curriculum veto **or** a temporal-state feature set, selected before modeling from the unmeasured EDA; never both. |
| Frozen components | Champion, semantic features, objective, positive authority, coverage range. |
| Required artifacts | Preregistered EDA decision, feature manifest, matched accepted/rejected action ledger, block report. |
| Validation protocol | New grouped outer folds and a new one-shot holdout; matched comparison at common proposal set. |
| Leakage safeguards | The EDA may inspect features and train labels descriptively but cannot select thresholds on the holdout; no public feedback. |
| Primary metric | Harm avoided minus benefit rejected, expressed as exact NDCG delta. |
| Secondary diagnostics | Veto rate, action pair/state slices, calibration, stability. |
| Behavior to log | Every veto with reason and counterfactual utility. |
| Success threshold | Positive net delta `>= +0.0003`, at least two harms prevented per lost benefit, stable across template/ID blocks. |
| Failure threshold | Net nonpositive, zero meaningful coverage, or high block heterogeneity. |
| Information if positive | The chosen structure is a useful safety signal. |
| Information if negative | Close that structure family and retain the simpler gate. |
| Expected runtime | Curriculum: low; temporal encoder: medium-high on T4x2. |
| Kaggle submission | Maybe, only after an untouched locked gate; never from EDA alone. |

## Recommended Decision Tree

1. If anchor, hash, or environment preflight fails, repair it without changing the scientific design.
2. If E5 and Nomic both lack positive exact-action competence, stop the semantic checkpoint branch and run the simple hierarchical-shrinkage alternative on a new prospective protocol.
3. If one encoder is useful and the other is not, do not force “dual” consensus. Record V47 as falsified and preregister a single-expert defer experiment on a new holdout.
4. If both are useful but agreement adds no lift, close consensus and investigate objective factorization in V48, without opening the old LOCK again.
5. If consensus passes DESIGN but fails LOCK, stop. The lesson is transport failure, not a request for threshold tuning.
6. If consensus passes LOCK and emits 3-15 test actions, make exactly one immutable public submission.
7. If the public result is flat or negative, preserve the result as package evidence and end test-user probing. Shift effort to validation/data-generating-process questions.
8. Only open curriculum or temporal branches if the corresponding unmeasured EDA shows incremental information. Otherwise keep the simpler champion.

## Final Recommendation

Freeze the `0.66028` champion and stop optimizing broad rankers or low-boundary action families. Execute one clean V47 competence test of frozen multilingual E5 and Nomic-v2-MoE on exact S0 Top1 counterfactual utility, with the same candidate pool, a prospective template-aware validation registry, independent expert reporting, and consensus-only authority. Remove generator-state voting and all membership/order edits from the hypothesis. This direction has the best joint expected value because it targets the only high-leverage action class, uses representations that truly never produced results, and yields a decisive negative result if competence or independence is absent.

The largest unresolved limitation is that no new external labeled data exist. Every train label and many public test outcomes have influenced prior research, so even a new split is prospectively clean only with respect to V47, not historically pristine. The second gap is raw behavioral EDA: template-cluster target entropy, explicit-mention precision, assessment redundancy, module-pair precedence stability, and incremental value of temporal transitions remain unmeasured without new analysis. These gaps should constrain confidence, not be filled with narrative.

## Appendix A: Per-Version Evidence Matrix

The detailed `Full V1-V46 Experiment Lineage` table is the normative per-version matrix. This compact index makes status and evidence gaps scannable.

| Status | Versions |
|---|---|
| KEEP as champion ancestry or durable evidence | V3.1, V6 methodological nesting, V12, V12.2, V21.1 action ancestry only, V22.2/V22.3 package ancestry, V24 provenance, V25 actions, V30, V31 overlap consensus, V37/V41.2 oracle-selector diagnosis, V39.5, V44 public-protected Top1 evidence |
| CLOSED after valid negative or nontransfer evidence | V2 broad architecture, V7-V11, V13, V14-V21 broad policies, V21.2, V22 model, V23, V27-V28, V29 broad policy, V29.5, V31 new-only, V32-V34 policies, V35.1, V36-V40 policies, V41.1, V42/V42.1, V43/V43.1, V44 coarse pruning, V45 further probing |
| INVALID or materially questionable for efficacy | V13.1, V35 initial TAPT, V41 |
| NOT EXECUTED | V3 semantic cross-encoder, V46 dual-semantic experiment |
| UNRESOLVED/partial | V4/V5 standalone public transfer, V12.1 package, V15's then-general action-supply hypothesis, V26 individual attribution, V29 curriculum as prior, V38.1 representation, V46 hypothesis |

No status should be read without the exact row in the main matrix. “Closed” applies to the tested formulation and authority, not to every imaginable method sharing a model name.

## Appendix B: Source Artifact Index

### Organizer data inspected

- `data/train_relevance.csv`
- `data/test.csv`
- `data/chat_history.csv`
- `data/user_assessments.csv`
- `data/modules_catalog.csv`
- `data/sample_submission.csv`

### Notebooks inspected

Every versioned notebook was inspected, including execution outputs and errors where present:

`notebooks/v1.ipynb`, `notebooks/v2.ipynb`, `notebooks/v3.ipynb`, `notebooks/v3_1.ipynb`, `notebooks/v4.ipynb`, `notebooks/v5.ipynb`, `notebooks/v6.ipynb`, `notebooks/v7.ipynb`, `notebooks/v8.ipynb`, `notebooks/v9.ipynb`, `notebooks/v9_1.ipynb`, `notebooks/v10.ipynb`, `notebooks/v10_1.ipynb`, `notebooks/v11.ipynb`, `notebooks/v12.ipynb`, `notebooks/v12_1.ipynb`, `notebooks/v12_2.ipynb`, `notebooks/v13.ipynb`, `notebooks/v13_1.ipynb`, `notebooks/v14.ipynb`, `notebooks/v15.ipynb`, `notebooks/v16.ipynb`, `notebooks/v17.ipynb`, `notebooks/v18.ipynb`, `notebooks/v19.ipynb`, `notebooks/v20.ipynb`, `notebooks/v21.ipynb`, `notebooks/v21_1.ipynb`, `notebooks/v21_2.ipynb`, `notebooks/v22.ipynb`, `notebooks/v23.ipynb`, `notebooks/v24.ipynb`, `notebooks/v25.ipynb`, `notebooks/v26.ipynb`, `notebooks/v27.ipynb`, `notebooks/v27_1.ipynb`, `notebooks/v28.ipynb`, `notebooks/v29.ipynb`, `notebooks/v29_5.ipynb`, `notebooks/v30.ipynb`, `notebooks/v31.ipynb`, `notebooks/v32.ipynb`, `notebooks/v33.ipynb`, `notebooks/v34.ipynb`, `notebooks/v35.ipynb`, `notebooks/v35_1.ipynb`, `notebooks/v36.ipynb`, `notebooks/v36_1.ipynb`, `notebooks/v37.ipynb`, `notebooks/v38.ipynb`, `notebooks/v38_1.ipynb`, `notebooks/v39.ipynb`, `notebooks/v39_1.ipynb`, `notebooks/v39_5.ipynb`, `notebooks/v40.ipynb`, `notebooks/v41.ipynb`, `notebooks/v41_1.ipynb`, `notebooks/v41_2.ipynb`, `notebooks/v42.ipynb`, `notebooks/v42_1.ipynb`, `notebooks/v43.ipynb`, `notebooks/v43_1.ipynb`, `notebooks/v44.ipynb`, `notebooks/v45.ipynb`, and `notebooks/v46.ipynb`.

### Direct result families inspected

- Early anchors and routers: `results/v1/`, `v2/`, `v3_1/`, `v4/` through `v21_2/`; the normative machine-readable files where present are `v8_summary.json` through `v21_2_summary.json` and associated diagnostics.
- Surgical lineage: `results/v22/v22_summary.json`, `v22_change_audit.csv`; `results/v23/v23_summary.json`; `results/v24/v24_summary.json`, `v24_probe_manifest.csv`; `results/v25/v25_summary.json`, `v25_change_manifest.csv`; `results/v26/v26_summary.json`, `v26_exact_diff_ledger.csv`.
- Head/curriculum lineage: `results/v27/`, `v27_1/`, `v28/`, `v29/`, `v29_5/`, especially their `*_summary.json`, validation, pseudo-public, ablation, uncertainty, and change ledgers.
- V30/V31 champion lineage: `results/v30/v30_summary.json`, `v30_three_salt_summary.csv`, `v30_test_change_ledger.csv`, `v30_incremental_blocks.csv`, `v30_direct_curriculum_pair_audit.csv`; `results/v31/v31_summary.json`, `v31_three_salt_summary.csv`, `v31_test_change_ledger.csv`, `v31_v30_reconstruction_audit.json`.
- Objective/retrieval/semantic audits: `results/v32/` through `results/v40/`, including reproduction audits, fresh-world summaries, tokenizer/TAPT audits, intent-state support, calibration policies, and action ledgers.
- Final lineage: V41 direct output in `notebooks/v41.ipynb`; `results/v41_1/`, `v41_2/`, `v42/`, `v42_1/`, `v43/`, `v43_1/`, `v44/`, `v45/`, and `results/v46/v46/`, with summary, split, fold/block, action, and reproduction artifacts.
- Binary model/cache blobs were not treated as primary research evidence; only metadata needed to identify model families was inspected.

### Historical handoff ZIPs inspected

- `sum/IT_Today_MineToday_Full_Handoff_2026-08-23.zip`
- `sum/MineToday_Full_Chat_Handoff_2026-08-24.zip`
- `sum/MineToday_Full_Chat_Handoff_2026-08-26.zip`
- `sum/MineToday_Full_Chat_Handoff_2026-08-27.zip`
- `sum/MineToday_Full_Chat_Handoff_2026-08-28_V31_READY.zip`
- `sum/MineToday_Full_Chat_Handoff_2026-08-31_CURRENT.zip`
- `sum/MineToday_Full_Chat_Handoff_2026-09-04_CURRENT.zip`

Key internal sources include `03_EXPERIMENT_HISTORY/experiment_timeline.md` and `04_LEADERBOARD_AND_SUBMISSIONS/submission_history.md` in the 2026-08-23 archive; `02_EXPERIMENT_HISTORY/EXPERIMENT_MASTER_TIMELINE.md` in 2026-08-24; `03_EXPERIMENT_HISTORY/PUBLIC_SCOREBOARD.md`, `V21_FAMILY.md`, `V22_PUBLIC_DECOMPOSITION.md`, and `05_VALIDATION_AND_LEAKAGE/VALIDATION_FAILURES.md` in 2026-08-26; `02_EXPERIMENT_HISTORY/V25_DESIGN_AND_PUBLIC_RESULT.md` and `05_CURRENT_STATE/EXACT_ACTION_LEDGER.md` in 2026-08-27; `02_EXPERIMENT_HISTORY/V26_TO_V30_DETAILED.md` and `07_V30_READY_PACKAGE/V30_DESIGN_AND_RESULTS.md` in 2026-08-28; `03_EXPERIMENT_HISTORY/MASTER_TIMELINE_V1_V38_1.md`, `04_PUBLIC_LB_AND_SUBMISSIONS/CURRENT_066020_ANALYSIS.md`, and `06_VALIDATION_AND_FAILURES/VALIDATION_FAILURE_MODES.md` in 2026-08-31; and the current-truth, V39-V44 timeline, causal ledger, dataset audit, validation philosophy, closed-branch, root-cause, and implementation-bug files in 2026-09-04.

## Appendix C: Unresolved Contradictions

| Issue | Direct evidence | Resolution |
|---|---|---|
| `notebooks/v25.ipynb` is titled/implemented as V24 | Notebook content versus `results/v25/v25_summary.json` and manifest | Use result artifacts for V25. Notebook provenance is incorrect and must not be silently repaired. |
| V3 is sometimes summarized as a model stage | Notebook ends in FP16 gradient-unscale error | Semantic cross-encoder is `NOT EXECUTED`; sparse prep only is valid. |
| V46 zero delta/zero actions can look like a negative result | Summary also says embeddings not ready and LOCK unopened | Treat metrics as fail-closed placeholders; hypothesis remains unresolved. |
| V41 strong CAL versus negative LOCK | Direct notebook output; no result directory | CAL optimism plus nonnested/dependent stack makes efficacy invalid. Use V41.1 corrective evidence. |
| V26 flat score but altered-file hash mismatch | `v26_summary.json` and exact diff ledger | Individual-action attribution is not established. |
| V44 Top1 pruning looks slightly positive offline; latest public removal is negative | V44 class metrics versus new `.66000` result | Preserve V30 S2 Top1 package. Public result is package-level, not individual proof. |
| Old and new S3 membership removals are both display-flat under different anchors | `.66006` historical context and `.66028` current context | Consistent evidence of low displayed leverage, not mathematical zero. |
| Rank-5/rank-6 means differ across summaries | Unconditional means include structural zeros; conditional means do not | Both are valid under different denominators. |
| K-predictability values differ slightly between handoffs | Around `.336` versus `.341` baseline | Exact statistic is `NOT ESTABLISHED FROM AVAILABLE ARTIFACTS`; rerun only as future EDA. |
| Latest handoff leaderboard context is older than the task | Archive top region differs from user-supplied current `.66076/.66142` | Use the task's current context, while retaining archives for experiment lineage. |
| OOF anchor levels differ across versions | Different folds, cohorts, strict/ordinary metrics, and reconstructed anchors | Do not compare raw levels across version protocols; compare paired deltas within a protocol. |

## Appendix D: Literature References

1. Ma, Jiaqi, et al. 2021. [Learning-to-Rank with Partitioned Preference](https://proceedings.mlr.press/v130/ma21a.html). AISTATS.
2. Xia, Fen, et al. 2008. [Listwise Approach to Learning to Rank: Theory and Algorithm](https://mlanthology.org/icml/2008/xia2008icml-listwise/). ICML. DOI: `10.1145/1390156.1390306`.
3. Wang, Xuanhui, et al. 2018. [The LambdaLoss Framework for Ranking Metric Optimization](https://research.google/pubs/the-lambdaloss-framework-for-ranking-metric-optimization/). CIKM.
4. Geifman, Yonatan, and Ran El-Yaniv. 2019. [SelectiveNet: A Deep Neural Network with an Integrated Reject Option](https://proceedings.mlr.press/v97/geifman19a.html). ICML.
5. Madras, David, Toni Pitassi, and Richard Zemel. 2018. [Predict Responsibly: Improving Fairness and Accuracy by Learning to Defer](https://papers.nips.cc/paper_files/paper/2018/hash/09d37c08f7b129e96277388757530c72-Abstract.html). NeurIPS.
6. Pan, Liangming, et al. 2017. [Prerequisite Relation Learning for Concepts in MOOCs](https://aclanthology.org/P17-1133/). ACL. DOI: `10.18653/v1/P17-1133`.
7. Jia, Chen, et al. 2021. [Learning Concept Prerequisite Relations from Educational Data via Multi-Head Attention Variational Graph Auto-Encoders](https://aclanthology.org/2021.naacl-main.164/). NAACL.
8. Piech, Chris, et al. 2015. [Deep Knowledge Tracing](https://proceedings.neurips.cc/paper_files/paper/2015/hash/bac9162b47c56fc8a4d2a519803d51b3-Abstract.html). NeurIPS.
9. Kang, Wang-Cheng, and Julian McAuley. 2018. [Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781). ICDM.
10. Sun, Fei, et al. 2019. [BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer](https://arxiv.org/abs/1904.06690). CIKM.
11. Wang, Liang, et al. 2024. [Multilingual E5 Text Embeddings: A Technical Report](https://arxiv.org/abs/2402.05672); [official model card](https://huggingface.co/intfloat/multilingual-e5-large-instruct).
12. Nussbaum, Zach, and Brandon Duderstadt. 2025. [Training Sparse Mixture Of Experts Text Embedding Models](https://arxiv.org/abs/2502.07972); [official Nomic v2 model card](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe).
13. Chen, Jianlyu, et al. 2024. [BGE M3-Embedding](https://arxiv.org/abs/2402.03216).
14. Khattab, Omar, and Matei Zaharia. 2020. [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/abs/2004.12832). SIGIR.
