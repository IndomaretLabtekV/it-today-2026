# MineToday: Cold-Start Course Recommendation

Solution for the [**MineToday** data mining competition](https://www.kaggle.com/competitions/mine-today-data-mining-competition-it-today-2026) at **IT Today 2026**, hosted on Kaggle. The task is to rank 17 learning modules for new users using only their Indonesian chat history and a skill self-assessment. Submissions are scored by mean user-level **NDCG@5**.

| [Leaderboard](https://www.kaggle.com/competitions/mine-today-data-mining-competition-it-today-2026/leaderboard) | Score | Rank |
|---|---:|---:|
| Public | **0.66219** | **#4** |
| Private | 0.65971 | #20 |

## Problem

- **Train:** 4,000 users with graded relevance (0–1) for each of the 17 modules.
- **Test:** 1,000 unseen users. This is a pure cold-start setting with no user-module interaction history, so collaborative filtering does not apply.
- **Signals:**
  - Free-text chat in casual Indonesian (14.5k messages, often switching goals over time).
  - A 10-skill self-assessment (Python, SQL, statistics, ML, GenAI, …).
  - The module catalog.
- **Data quirks:**
  - The data is synthetic and template-driven.
  - Some users have no chat at all.
  - 90 chat-only orphan IDs appear.
  - Assessment fields are corrupted, for example numbers written as words.
  - Identical observable inputs sometimes carry conflicting labels, which puts a hard ceiling on achievable accuracy.

## Approach

The solution has two stages. A strong base ranker comes first, then a conservative **Top-1 action layer** that edits its output.

### 1. Base ranker (V42 lineage, public 0.66028)

```
chat + assessment cleaning, topic/style features
  → Ridge on word + char TF-IDF (relevance and top-1 targets)
  → per-module LightGBM probability calibration
  → Consensus blend with frozen per-module weights
  → two LightGBM LambdaRank models (with and without intent-state features)
  → rule-based hybrid with protected-user and boundary rules
```

### 2. Top-1 action layer (V127 → V132)

Each user can take one possible edit: swap the current Top-1 module with a candidate at rank 2–4. The layer works as follows:

- **Label:** the exact counterfactual ΔNDCG@5 of that swap, computed against the actual deployed ranking.
- **Features:**
  - Sparse retrieval: TF-IDF word/char views over the full history, the latest message, and recency, plus BM25 (k1=1.2, b=0.75).
  - Dense retrieval: [`multilingual-e5-base`](https://huggingface.co/intfloat/multilingual-e5-base) and [`nomic-embed-text-v2-moe`](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe), pinned to fixed revisions.
  - Assessment vector, module identity, rank, and the margin between incumbent and challenger.
- **Model:** a global Ridge regressor on ΔNDCG, plus an incumbent × challenger × rank residual that is shrunk toward zero (partial pooling).
- **Deployment:** only the highest-utility positive swaps are applied, capped at 5 users. Top-5 membership, the rank-5 module, and the score multiset are preserved by construction.
- **Fail-closed gate:** if any gate fails, the pipeline writes the exact bytes of the previous champion.

### Score progression (public LB)

| Version | Change | Public |
|---|---|---:|
| V42 | Base ranker | 0.66028 |
| V129 | Sparse + dense retrieval with a nested utility selector | 0.66075 |
| V130 | Selector aligned to the champion, ranks 2–4 | 0.66173 |
| **V132** | Hierarchical ranks 2–4 utility with BM25 and pair×rank shrinkage | **0.66219** |

## Validation methodology

- **Fully nested 5-fold OOF:** every supervised component, including the anchor chain, action labels, and heads, is refit inside each outer fold.
- **Uncertainty:** paired user-level and cluster-level bootstrap confidence intervals, plus counts of beneficial vs. harmful edits.
- **Stress tests:** grouped "worlds" by exact chat template, latest template, near-duplicate, and contiguous ID blocks.
- **Pre-registration:** promotion gates are fixed before each run, and leaderboard scores are never used as features, labels, or thresholds.
- **Unit tests:** the leakage and correctness contracts are tested in [`tests/test_v127_pipeline.py`](tests/test_v127_pipeline.py). The tests cover exact swap targets, isolation of excluded labels, the gate, and byte-exact fallback.

## Lessons learned

- **The dense/sparse ceiling:** errors from TF-IDF, BM25, E5, and Nomic correlate at ≥ 0.995. Once the base ranker was strong, more representations added almost no new information.
- **Gains became too small to measure:** after V42, gains came from a handful of Top-1 swaps (+0.0002 to +0.001). That is below what 1,000 test users can resolve. The drop from public #4 to private #20 (−0.0025) is consistent with these increments sitting inside leaderboard noise.
- **What to do differently:** fewer public-validated micro-edits, plus more robust base-model improvements such as ensembles and better text features.
- **Generator analysis:** a forensic analysis of the synthetic data generator is in [`research/MineToday_reverse_engineering_FINAL`](research/MineToday_reverse_engineering_FINAL/01_FINAL_REPORT.md). It shows which generator stages can be recovered and proves that the exact ranking function cannot be identified from the released columns.

## Repository structure

```
notebooks/          Experiment notebooks v1 … v137 (each one self-contained, Kaggle-ready)
nb_code/            v127_pipeline.py: tested, side-effect-free Top-1 action pipeline
tests/              Leakage and correctness tests for the pipeline
results/vNNN/       Per-version outputs: decision/gate reports, fold metrics, primary submissions
research/           Forensic analysis of the synthetic data generator
```

The final submission comes from [`notebooks/v132.ipynb`](notebooks/v132.ipynb). The submitted file is [`results/v132/v132/candidate_v132_DIAGNOSTIC.csv`](results/v132/v132/candidate_v132_DIAGNOSTIC.csv).

## Reproducing

The notebooks were built for **Kaggle** (Python 3.12, GPU recommended for the embeddings).

1. Attach the competition data so that the organizer CSVs are available under `/kaggle/input/`. The raw data is not redistributed here.
2. Run `notebooks/v132.ipynb` from start to finish. It verifies the hashes of the embedded base submission and of the pinned model revisions, then writes `candidate_v132_DIAGNOSTIC.csv`.

To run the tests locally:

```bash
pip install -r requirements.txt
pytest tests/
```
