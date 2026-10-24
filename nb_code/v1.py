# Converted from notebooks/v1.ipynb: markdown cells become comments, %/! magics neutralized.

# ---- cell 0 (markdown) ----
# # MineToday IT Today 2026 — Ultimate Strong Baseline
#
# **Task:** rank the 17 Intelligo learning modules for each test user and maximize **NDCG@5**.
#
# This notebook is deliberately built as a **hybrid supervised learning-to-rank system** rather than a plain semantic recommender:
#
# 1. robust assessment parsing + dirty-value flags,
# 2. chronological chat profiles with recency emphasis,
# 3. **Qwen3-Embedding-0.6B** multilingual semantic representations,
# 4. **Qwen3-Reranker-0.6B** zero-shot user↔module cross-encoder scores,
# 5. word + character TF-IDF lexical signals,
# 6. exact/normalized repeated-chat template memory with leakage-safe leave-one-user-out features,
# 7. **LightGBM LambdaRank** trained directly on ordinal ranking labels,
# 8. a linear Ridge model for diversity,
# 9. OOF-only blend-weight search, then final top-5 ranking scores.
#
# ### Important observations exploited by the model
# - Every train user has only **4–6 positive modules**.
# - Exactly **one module has relevance 1.0** per user.
# - Relevance values form rank bands (roughly 1.00, 0.80–0.90, 0.65–0.75, 0.50–0.60, ...), so the model learns **ordinal ranking labels** instead of noisy raw values.
# - Chat text is heavily templated/repeated, making leakage-safe template memory a legitimate high-value feature.
# - Assessments exist for all train and test users and contain a small amount of intentionally/accidentally dirty values.
#
# ### Kaggle setup
# - Accelerator: **GPU T4 x2**
# - Internet: **ON** (needed to download public Hugging Face pretrained weights unless you attach them as a Kaggle Dataset)
# - The notebook only uses the provided competition CSVs as task-specific training data.
#
# > Before submitting, verify the competition rulebook allows public pretrained model weights. I could not verify that exact rule publicly.

# ---- cell 1 (code) ----
# Install only the versions needed for Qwen3 embedding/reranking support.
# Qwen3 requires transformers >= 4.51.0.
# MAGIC: %pip install -q -U "transformers>=4.51.0,<5" "sentence-transformers>=5.4.0,<6" "accelerate>=1.0" "lightgbm>=4.5.0"

# ---- cell 2 (code) ----
import os
import gc
import re
import json
import math
import random
import warnings
import unicodedata
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from scipy.sparse import vstack as sp_vstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, normalize

import lightgbm as lgb
import torch

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

SEED = 2026
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}:", torch.cuda.get_device_name(i))

# ---- cell 3 (code) ----
# =========================
# Configuration
# =========================
N_FOLDS = 5
N_MODULES_EXPECTED = 17

EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
RERANK_MODEL = "Qwen/Qwen3-Reranker-0.6B"
EMBED_DIM = 256
EMBED_MAX_TOKENS = 1024
EMBED_BATCH_SIZE = 24

USE_RERANKER = True
RERANK_MAX_TOKENS = 768
RERANK_BATCH_SIZE = 32

WORD_MAX_FEATURES = 25_000
CHAR_MAX_FEATURES = 35_000
LEXICAL_SVD_DIM = 96

MEMORY_SHRINK = 8.0

CACHE_DIR = Path("/kaggle/working/minetoday_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_FILES = {
    "chat_history.csv",
    "modules_catalog.csv",
    "user_assessments.csv",
    "train_relevance.csv",
    "sample_submission.csv",
    "test.csv",
}

# ---- cell 4 (code) ----
def find_data_dir():
    roots = [Path("/kaggle/input"), Path(".")]
    candidates = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("train_relevance.csv"):
            parent = p.parent
            try:
                names = {x.name for x in parent.iterdir() if x.is_file()}
            except Exception:
                continue
            if EXPECTED_FILES.issubset(names):
                candidates.append(parent)
    if not candidates:
        raise FileNotFoundError(
            "Could not find the MineToday dataset directory. Attach the competition dataset to this Kaggle notebook."
        )
    # Prefer a Kaggle input directory if available.
    candidates = sorted(candidates, key=lambda p: ("/kaggle/input" not in str(p), len(str(p))))
    return candidates[0]

DATA_DIR = find_data_dir()
print("DATA_DIR =", DATA_DIR)

chat = pd.read_csv(DATA_DIR / "chat_history.csv")
modules = pd.read_csv(DATA_DIR / "modules_catalog.csv")
assess = pd.read_csv(DATA_DIR / "user_assessments.csv")
train = pd.read_csv(DATA_DIR / "train_relevance.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
sample = pd.read_csv(DATA_DIR / "sample_submission.csv")

module_cols = [c for c in train.columns if c != "user_id"]
assert len(module_cols) == N_MODULES_EXPECTED
assert list(sample.columns[1:]) == module_cols
assert list(test["user_id"]) == list(sample["user_id"])
assert train["user_id"].is_unique and test["user_id"].is_unique
assert assess["user_id"].is_unique

modules = modules.set_index("module_id").loc[module_cols].reset_index()

print("chat       :", chat.shape)
print("modules    :", modules.shape)
print("assessment :", assess.shape)
print("train      :", train.shape)
print("test       :", test.shape)
display(modules)

# ---- cell 5 (code) ----
# =========================
# Fast competition-style NDCG@5
# Uses linear gains, matching the natural interpretation of the provided graded relevance.
# =========================
def fast_ndcg_at_k(y_true, y_score, k=5):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    order = np.argsort(-y_score, axis=1)[:, :k]
    rel = np.take_along_axis(y_true, order, axis=1)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (rel * discounts).sum(axis=1)

    ideal = np.sort(y_true, axis=1)[:, ::-1][:, :k]
    idcg = (ideal * discounts).sum(axis=1)
    return float(np.mean(np.divide(dcg, idcg, out=np.zeros_like(dcg), where=idcg > 0)))

Y_rel = train[module_cols].to_numpy(np.float32)

positive_counts = (Y_rel > 0).sum(axis=1)
ones_counts = (Y_rel == 1.0).sum(axis=1)
ranked = np.sort(Y_rel, axis=1)[:, ::-1]

print("Positive modules/user distribution:", pd.Series(positive_counts).value_counts().sort_index().to_dict())
print("Exactly-one-1.0 check:", pd.Series(ones_counts).value_counts().sort_index().to_dict())
print("Mean relevance by rank 1..6:", np.round(ranked[:, :6].mean(axis=0), 4))

# Convert noisy relevance bands to stable ordinal labels:
# top1=6, top2=5, ..., top6=1, irrelevant=0.
def relevance_to_ordinal(y):
    out = np.zeros_like(y, dtype=np.int32)
    for i, row in enumerate(y):
        pos = np.flatnonzero(row > 0)
        pos = pos[np.argsort(-row[pos])]
        labels = np.array([6, 5, 4, 3, 2, 1], dtype=np.int32)[: len(pos)]
        out[i, pos] = labels
    return out

Y_ord = relevance_to_ordinal(Y_rel)
top_class = Y_ord.argmax(axis=1)

print("Top-module frequency:")
display(pd.DataFrame({
    "module_id": module_cols,
    "module_name": modules["module_name"],
    "top1_count": np.bincount(top_class, minlength=len(module_cols)),
    "mean_relevance": Y_rel.mean(axis=0),
    "positive_rate": (Y_rel > 0).mean(axis=0),
}).sort_values("top1_count", ascending=False))

# ---- cell 6 (code) ----
# =========================
# Robust assessment parsing
# =========================
WORD_TO_NUM = {
    "nol": 0.0,
    "satu": 1.0,
    "dua": 2.0,
    "tiga": 3.0,
    "empat": 4.0,
    "lima": 5.0,
}

def parse_assessment_value(x):
    s = str(x).strip().lower()
    if s in WORD_TO_NUM:
        return WORD_TO_NUM[s]
    try:
        v = float(s)
    except Exception:
        return np.nan
    return v if 0.0 <= v <= 5.0 else np.nan

parsed_json = assess["assessment_result"].map(json.loads)
question_names = list(parsed_json.iloc[0].keys())
raw_q = pd.DataFrame(parsed_json.tolist(), columns=question_names)
clean_q = raw_q.apply(lambda col: col.map(parse_assessment_value)).astype(np.float32)
invalid_q = clean_q.isna().astype(np.float32)

train_user_set = set(train["user_id"])
ass_train_mask = assess["user_id"].isin(train_user_set).to_numpy()
train_medians = clean_q.loc[ass_train_mask].median(axis=0)
clean_q = clean_q.fillna(train_medians)

short_skill_names = [
    "Python", "SQL", "Statistics", "EDA", "Machine Learning",
    "ML Evaluation", "Deep Learning", "Generative AI / LLM",
    "Business-to-DS Translation", "Independent Research / Troubleshooting",
]
assert len(question_names) == len(short_skill_names) == 10

ass_features = pd.DataFrame({"user_id": assess["user_id"]})
for j in range(10):
    ass_features[f"assess_{j:02d}"] = clean_q.iloc[:, j].to_numpy()
    ass_features[f"assess_invalid_{j:02d}"] = invalid_q.iloc[:, j].to_numpy()

ass_values = clean_q.to_numpy(np.float32)
ass_features["assess_mean"] = ass_values.mean(axis=1)
ass_features["assess_std"] = ass_values.std(axis=1)
ass_features["assess_min"] = ass_values.min(axis=1)
ass_features["assess_max"] = ass_values.max(axis=1)
ass_features["assess_novice_count"] = (ass_values <= 1).sum(axis=1).astype(np.float32)
ass_features["assess_intermediate_plus_count"] = (ass_values >= 3).sum(axis=1).astype(np.float32)
ass_features["assess_advanced_count"] = (ass_values >= 4).sum(axis=1).astype(np.float32)
ass_features["assess_invalid_count"] = invalid_q.sum(axis=1).to_numpy(np.float32)

print("Assessment invalid cells:", int(invalid_q.to_numpy().sum()), "/", invalid_q.size)
print("Example raw dirty values:")
dirty_values = []
for col in raw_q.columns:
    for x in raw_q[col].astype(str).unique():
        if np.isnan(parse_assessment_value(x)):
            dirty_values.append(x)
print(sorted(set(dirty_values))[:30])

# ---- cell 7 (code) ----
# =========================
# Chat aggregation + normalization
# =========================
def normalize_chat_text(s):
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())

chat["timestamp"] = pd.to_datetime(chat["timestamp"], errors="coerce")
chat["norm_text"] = chat["user_chat_text"].map(normalize_chat_text)
chat = chat.sort_values(["user_id", "timestamp", "chat_id"]).reset_index(drop=True)

all_user_ids = pd.concat([train["user_id"], test["user_id"]], ignore_index=True)
n_train = len(train)
n_test = len(test)
n_all = len(all_user_ids)
all_user_set = set(all_user_ids)

chat_known = chat[chat["user_id"].isin(all_user_set)].copy()

def join_all(s):
    return " [SEP] ".join(map(str, s))

def join_last2(s):
    vals = list(map(str, s))
    return " [SEP] ".join(vals[-2:])

chat_agg = chat_known.groupby("user_id").agg(
    chat_count=("user_chat_text", "size"),
    first_ts=("timestamp", "min"),
    last_ts=("timestamp", "max"),
    all_chat=("user_chat_text", join_all),
    last_chat=("user_chat_text", "last"),
    last2_chat=("user_chat_text", join_last2),
    unique_norm_count=("norm_text", "nunique"),
)
chat_agg = chat_agg.reindex(all_user_ids).reset_index().rename(columns={"index": "user_id"})

for c in ["all_chat", "last_chat", "last2_chat"]:
    chat_agg[c] = chat_agg[c].fillna("")
chat_agg["chat_count"] = chat_agg["chat_count"].fillna(0).astype(np.float32)
chat_agg["unique_norm_count"] = chat_agg["unique_norm_count"].fillna(0).astype(np.float32)
chat_agg["chat_span_days"] = (
    (chat_agg["last_ts"] - chat_agg["first_ts"]).dt.total_seconds() / 86400.0
).fillna(0).astype(np.float32)
chat_agg["chat_char_count"] = chat_agg["all_chat"].str.len().astype(np.float32)
chat_agg["chat_word_count"] = chat_agg["all_chat"].str.split().map(len).astype(np.float32)
chat_agg["last_chat_char_count"] = chat_agg["last_chat"].str.len().astype(np.float32)

# Unique normalized messages per user for the leakage-safe template memory model.
user_message_dict = (
    chat_known.groupby("user_id")["norm_text"]
    .apply(lambda s: list(dict.fromkeys(x for x in s if x)))
    .to_dict()
)
user_messages = [user_message_dict.get(u, []) for u in all_user_ids]

# Quantify train -> test repeated-template coverage.
train_chat_texts = set(chat_known.loc[chat_known["user_id"].isin(train_user_set), "norm_text"])
test_chat_rows = chat_known[chat_known["user_id"].isin(set(test["user_id"]))].copy()
if len(test_chat_rows):
    test_chat_rows["seen_in_train"] = test_chat_rows["norm_text"].isin(train_chat_texts)
    msg_overlap = test_chat_rows["seen_in_train"].mean()
    user_overlap = test_chat_rows.groupby("user_id")["seen_in_train"].max().mean()
else:
    msg_overlap = user_overlap = 0.0

print(f"Normalized exact test-message overlap with train: {msg_overlap:.3f}")
print(f"Test users with chat and >=1 repeated train template: {user_overlap:.3f}")
print("Missing chat users - train:", int((chat_agg.loc[:n_train-1, "chat_count"] == 0).sum()))
print("Missing chat users - test :", int((chat_agg.loc[n_train:, "chat_count"] == 0).sum()))

# ---- cell 8 (code) ----
# =========================
# Assemble text profiles and numeric user features
# =========================
ass_indexed = ass_features.set_index("user_id").loc[all_user_ids]
ass_clean_indexed = pd.DataFrame(clean_q.to_numpy(), index=assess["user_id"], columns=short_skill_names).loc[all_user_ids]

profile_texts = []
latest_profile_texts = []
for i, user_id in enumerate(all_user_ids):
    scores = ass_clean_indexed.iloc[i].to_numpy()
    assessment_text = "; ".join(
        f"{name}={score:.0f}/5" for name, score in zip(short_skill_names, scores)
    )
    history = chat_agg.at[i, "all_chat"] or "No chat history available."
    latest = chat_agg.at[i, "last_chat"] or "No chat history available."

    profile_texts.append(
        "Learner profile. "
        f"Assessment: {assessment_text}. "
        f"Chronological chat history: {history}. "
        f"Most recent learner message: {latest}."
    )
    latest_profile_texts.append(
        "Learner profile with recency emphasis. "
        f"Assessment: {assessment_text}. "
        f"Most recent learner message: {latest}."
    )

module_docs = [
    f"Course module: {row.module_name}. "
    f"Syllabus: {row.description_and_syllabus}. "
    f"Prerequisite level: {row.prerequisite_level}."
    for row in modules.itertuples(index=False)
]

numeric_user_cols = [c for c in ass_indexed.columns if c != "user_id"]
assessment_matrix = ass_indexed[numeric_user_cols].to_numpy(np.float32)
chat_stat_cols = [
    "chat_count", "unique_norm_count", "chat_span_days", "chat_char_count",
    "chat_word_count", "last_chat_char_count",
]
chat_stats_matrix = chat_agg[chat_stat_cols].to_numpy(np.float32)

print(profile_texts[1][:1000])
print("\nExample module document:\n", module_docs[0])

# ---- cell 9 (markdown) ----
# ## Semantic backbone
#
# The default practical SOTA backbone here is **Qwen3-Embedding-0.6B**. It supports 100+ languages, long context, instruction-aware retrieval, and Matryoshka-style truncation. The companion **Qwen3-Reranker-0.6B** is used as an additional cross-encoder signal.
#
# References:
# - https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
# - https://huggingface.co/Qwen/Qwen3-Reranker-0.6B
#
# The 4B variants benchmark higher, but the 0.6B pair is a much safer first competition baseline on T4 while leaving enough runtime for supervised ranking and CV.

# ---- cell 10 (code) ----
# =========================
# Qwen3 multilingual embeddings
# =========================
from sentence_transformers import SentenceTransformer

EMBED_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
QUERY_PROMPT = (
    "Instruct: Given a learner profile, retrieve learning modules that are the best next step "
    "based on the learner's current skill gaps, interests, recent stated goals, career direction, "
    "and module prerequisites.\nQuery: "
)

hist_cache = CACHE_DIR / f"qwen3_hist_{EMBED_DIM}.npy"
latest_cache = CACHE_DIR / f"qwen3_latest_{EMBED_DIM}.npy"
module_cache = CACHE_DIR / f"qwen3_modules_{EMBED_DIM}.npy"

cache_ok = (
    hist_cache.exists() and latest_cache.exists() and module_cache.exists()
    and np.load(hist_cache, mmap_mode="r").shape == (n_all, EMBED_DIM)
    and np.load(latest_cache, mmap_mode="r").shape == (n_all, EMBED_DIM)
    and np.load(module_cache, mmap_mode="r").shape == (len(module_cols), EMBED_DIM)
)

if cache_ok:
    hist_emb = np.load(hist_cache).astype(np.float32)
    latest_emb = np.load(latest_cache).astype(np.float32)
    module_emb = np.load(module_cache).astype(np.float32)
    print("Loaded Qwen3 embeddings from cache.")
else:
    model_kwargs = {"torch_dtype": torch.float16} if torch.cuda.is_available() else {}
    embedder = SentenceTransformer(
        EMBED_MODEL,
        device=EMBED_DEVICE,
        model_kwargs=model_kwargs,
        tokenizer_kwargs={"padding_side": "left"},
        truncate_dim=EMBED_DIM,
    )
    embedder.max_seq_length = EMBED_MAX_TOKENS

    hist_emb = embedder.encode(
        profile_texts,
        prompt=QUERY_PROMPT,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    latest_emb = embedder.encode(
        latest_profile_texts,
        prompt=QUERY_PROMPT,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    module_emb = embedder.encode(
        module_docs,
        batch_size=17,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    np.save(hist_cache, hist_emb)
    np.save(latest_cache, latest_emb)
    np.save(module_cache, module_emb)

    del embedder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

sem_hist = hist_emb @ module_emb.T
sem_latest = latest_emb @ module_emb.T
semantic_score = 0.65 * sem_hist + 0.35 * sem_latest

print("Embedding shapes:", hist_emb.shape, latest_emb.shape, module_emb.shape)
print("Zero-shot semantic train NDCG@5:", fast_ndcg_at_k(Y_rel, semantic_score[:n_train], 5))

# ---- cell 11 (code) ----
# =========================
# Qwen3 cross-encoder reranker scores
# =========================
rerank_cache = CACHE_DIR / "qwen3_reranker_scores.npy"

if USE_RERANKER:
    if rerank_cache.exists() and np.load(rerank_cache, mmap_mode="r").shape == (n_all, len(module_cols)):
        rerank_scores = np.load(rerank_cache).astype(np.float32)
        print("Loaded reranker scores from cache.")
    else:
        from sentence_transformers import CrossEncoder

        RERANK_DEVICE = (
            "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1
            else ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        reranker_kwargs = {"torch_dtype": torch.float16} if torch.cuda.is_available() else {}
        reranker = CrossEncoder(
            RERANK_MODEL,
            device=RERANK_DEVICE,
            max_length=RERANK_MAX_TOKENS,
            model_kwargs=reranker_kwargs,
            prompts={
                "course_match": (
                    "Score how well this learning module is the learner's next recommendation, "
                    "considering skill gaps, recent goals, career direction, and prerequisites."
                )
            },
            default_prompt_name="course_match",
        )

        pairs = [
            (profile_texts[i], module_docs[j])
            for i in range(n_all)
            for j in range(len(module_cols))
        ]
        try:
            raw_scores = reranker.predict(
                pairs,
                batch_size=RERANK_BATCH_SIZE,
                show_progress_bar=True,
            )
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            print("CUDA OOM at the default reranker batch size; retrying with batch_size=8.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raw_scores = reranker.predict(
                pairs,
                batch_size=8,
                show_progress_bar=True,
            )

        rerank_scores = np.asarray(raw_scores, dtype=np.float32).reshape(n_all, len(module_cols))
        np.save(rerank_cache, rerank_scores)
        del pairs, raw_scores, reranker
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("Zero-shot reranker train NDCG@5:", fast_ndcg_at_k(Y_rel, rerank_scores[:n_train], 5))
else:
    rerank_scores = semantic_score.astype(np.float32).copy()
    print("Reranker disabled; using semantic score as fallback.")

# ---- cell 12 (code) ----
# =========================
# Lexical features: word + character TF-IDF
# Character n-grams are especially useful for slang, typos, mixed Indonesian/English, and repeated templates.
# =========================
user_chat_docs = chat_agg["all_chat"].fillna("").tolist()
lexical_corpus = user_chat_docs + module_docs

word_vec = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.995,
    max_features=WORD_MAX_FEATURES,
    sublinear_tf=True,
    dtype=np.float32,
)
word_all = word_vec.fit_transform(lexical_corpus)
word_user = word_all[:n_all]
word_module = word_all[n_all:]
word_cos = (word_user @ word_module.T).toarray().astype(np.float32)

word_dim = min(LEXICAL_SVD_DIM, max(2, word_all.shape[1] - 1))
word_svd = TruncatedSVD(n_components=word_dim, n_iter=7, random_state=SEED)
word_dense_all = normalize(word_svd.fit_transform(word_all)).astype(np.float32)
word_user_svd = word_dense_all[:n_all]

char_vec = TfidfVectorizer(
    lowercase=True,
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=2,
    max_features=CHAR_MAX_FEATURES,
    sublinear_tf=True,
    dtype=np.float32,
)
char_all = char_vec.fit_transform(lexical_corpus)
char_user = char_all[:n_all]
char_module = char_all[n_all:]
char_cos = (char_user @ char_module.T).toarray().astype(np.float32)

char_dim = min(LEXICAL_SVD_DIM, max(2, char_all.shape[1] - 1))
char_svd = TruncatedSVD(n_components=char_dim, n_iter=7, random_state=SEED + 1)
char_dense_all = normalize(char_svd.fit_transform(char_all)).astype(np.float32)
char_user_svd = char_dense_all[:n_all]

print("word vocab:", len(word_vec.vocabulary_), "char vocab:", len(char_vec.vocabulary_))
print("word SVD explained variance:", float(word_svd.explained_variance_ratio_.sum()))
print("char SVD explained variance:", float(char_svd.explained_variance_ratio_.sum()))

# ---- cell 13 (code) ----
# =========================
# Static user representation used by supervised models
# =========================
user_static = np.concatenate([
    assessment_matrix,
    chat_stats_matrix,
    hist_emb,
    latest_emb,
    word_user_svd,
    char_user_svd,
], axis=1).astype(np.float32)

print("user_static:", user_static.shape, f"~{user_static.nbytes / 1024**2:.1f} MB")

# ---- cell 14 (code) ----
# =========================
# Leakage-safe normalized-template memory
# =========================
def build_template_stats(pool_idx, labels):
    pool_idx = np.asarray(pool_idx, dtype=np.int64)
    global_prior = labels[pool_idx].mean(axis=0).astype(np.float32)
    sums = {}
    counts = {}

    for i in pool_idx:
        for msg in user_messages[i]:  # unique normalized messages per user
            if msg in sums:
                sums[msg] += labels[i]
                counts[msg] += 1
            else:
                sums[msg] = labels[i].astype(np.float32).copy()
                counts[msg] = 1
    return global_prior, sums, counts


def template_memory_for_indices(target_idx, labels, stats, leave_one_out=False):
    global_prior, sums, counts = stats
    target_idx = np.asarray(target_idx, dtype=np.int64)
    out = np.zeros((len(target_idx), len(module_cols)), dtype=np.float32)
    coverage = np.zeros(len(target_idx), dtype=np.float32)
    confidence = np.zeros(len(target_idx), dtype=np.float32)

    for r, i in enumerate(target_idx):
        msgs = user_messages[i]
        preds = []
        weights = []
        matched = 0

        for msg in msgs:
            if msg not in sums:
                continue
            n = counts[msg]
            s = sums[msg]

            if leave_one_out:
                n = n - 1
                if n <= 0:
                    continue
                s = s - labels[i]

            mu = s / n
            shrink = n / (n + MEMORY_SHRINK)
            p = shrink * mu + (1.0 - shrink) * global_prior

            # Generic service/support templates have small deviation from the global prior,
            # so they are automatically downweighted.
            information = float(np.linalg.norm(p - global_prior))
            w = (information + 1e-3) / math.sqrt(max(n, 1))
            preds.append(p)
            weights.append(w)
            matched += 1

        if preds:
            out[r] = np.average(np.stack(preds), axis=0, weights=np.asarray(weights))
            coverage[r] = matched / max(len(msgs), 1)
            confidence[r] = float(np.mean(weights))
        else:
            out[r] = global_prior
            coverage[r] = 0.0
            confidence[r] = 0.0

    return out, coverage, confidence

# ---- cell 15 (code) ----
# =========================
# Pair feature builder
# Each user contributes exactly 17 contiguous rows, one per module.
# =========================
MODULE_EYE = np.eye(len(module_cols), dtype=np.float32)

def build_pair_features(user_idx, mem_score, mem_coverage, mem_confidence, module_priors):
    user_idx = np.asarray(user_idx, dtype=np.int64)
    n = len(user_idx)

    repeated_user = np.repeat(user_static[user_idx], len(module_cols), axis=0)
    module_onehot = np.tile(MODULE_EYE, (n, 1))

    pair_specific = np.column_stack([
        sem_hist[user_idx].reshape(-1),
        sem_latest[user_idx].reshape(-1),
        semantic_score[user_idx].reshape(-1),
        rerank_scores[user_idx].reshape(-1),
        word_cos[user_idx].reshape(-1),
        char_cos[user_idx].reshape(-1),
        mem_score.reshape(-1),
        np.repeat(mem_coverage, len(module_cols)),
        np.repeat(mem_confidence, len(module_cols)),
    ]).astype(np.float32)

    # module_priors shape: [17, 3] = mean ordinal relevance, positive rate, top1 rate
    prior_block = np.tile(module_priors.astype(np.float32), (n, 1))

    return np.concatenate([
        repeated_user,
        module_onehot,
        pair_specific,
        prior_block,
    ], axis=1).astype(np.float32)

# ---- cell 16 (code) ----
# =========================
# 5-fold supervised learning-to-rank
# =========================
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
test_global_idx = np.arange(n_train, n_all)

oof_ranker = np.zeros((n_train, len(module_cols)), dtype=np.float32)
oof_ridge = np.zeros_like(oof_ranker)
oof_memory = np.zeros_like(oof_ranker)

test_ranker_folds = []
test_ridge_folds = []
test_memory_folds = []

fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(cv.split(np.arange(n_train), top_class), start=1):
    print(f"\n========== Fold {fold}/{N_FOLDS} ==========")

    stats = build_template_stats(tr_idx, Y_ord.astype(np.float32))
    mem_tr, cov_tr, conf_tr = template_memory_for_indices(
        tr_idx, Y_ord.astype(np.float32), stats, leave_one_out=True
    )
    mem_va, cov_va, conf_va = template_memory_for_indices(
        va_idx, Y_ord.astype(np.float32), stats, leave_one_out=False
    )
    mem_te, cov_te, conf_te = template_memory_for_indices(
        test_global_idx, Y_ord.astype(np.float32), stats, leave_one_out=False
    )

    oof_memory[va_idx] = mem_va
    test_memory_folds.append(mem_te)

    module_priors = np.column_stack([
        Y_ord[tr_idx].mean(axis=0),
        (Y_ord[tr_idx] > 0).mean(axis=0),
        np.bincount(top_class[tr_idx], minlength=len(module_cols)) / len(tr_idx),
    ]).astype(np.float32)

    X_tr = build_pair_features(tr_idx, mem_tr, cov_tr, conf_tr, module_priors)
    X_va = build_pair_features(va_idx, mem_va, cov_va, conf_va, module_priors)
    X_te = build_pair_features(test_global_idx, mem_te, cov_te, conf_te, module_priors)

    y_tr = Y_ord[tr_idx].reshape(-1)
    y_va = Y_ord[va_idx].reshape(-1)
    group_tr = [len(module_cols)] * len(tr_idx)
    group_va = [len(module_cols)] * len(va_idx)

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=3000,
        learning_rate=0.02,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=35,
        subsample=0.90,
        colsample_bytree=0.78,
        reg_alpha=0.20,
        reg_lambda=3.0,
        max_bin=255,
        lambdarank_truncation_level=8,
        label_gain=[0, 25, 40, 55, 70, 85, 100],
        random_state=SEED + fold,
        n_jobs=-1,
        verbosity=-1,
        force_col_wise=True,
    )

    ranker.fit(
        X_tr,
        y_tr,
        group=group_tr,
        eval_set=[(X_va, y_va)],
        eval_group=[group_va],
        eval_at=[5],
        callbacks=[
            lgb.early_stopping(180, verbose=False),
            lgb.log_evaluation(0),
        ],
    )

    pred_va = ranker.predict(X_va, num_iteration=ranker.best_iteration_).reshape(len(va_idx), len(module_cols))
    pred_te = ranker.predict(X_te, num_iteration=ranker.best_iteration_).reshape(n_test, len(module_cols))
    oof_ranker[va_idx] = pred_va.astype(np.float32)
    test_ranker_folds.append(pred_te.astype(np.float32))

    # Diverse user-level linear model. It sees exactly the same fold split and no validation labels.
    scaler = StandardScaler()
    U_tr = scaler.fit_transform(user_static[tr_idx])
    U_va = scaler.transform(user_static[va_idx])
    U_te = scaler.transform(user_static[test_global_idx])

    ridge = Ridge(alpha=35.0, solver="lsqr")
    ridge.fit(U_tr, Y_ord[tr_idx].astype(np.float32))
    oof_ridge[va_idx] = ridge.predict(U_va).astype(np.float32)
    test_ridge_folds.append(ridge.predict(U_te).astype(np.float32))

    fold_ndcg = fast_ndcg_at_k(Y_rel[va_idx], pred_va, 5)
    fold_scores.append(fold_ndcg)
    print("best_iteration:", ranker.best_iteration_)
    print(f"fold NDCG@5: {fold_ndcg:.6f}")

    del X_tr, X_va, X_te, ranker, scaler, ridge
    gc.collect()

print("\nFold scores:", [round(x, 6) for x in fold_scores])
print("Mean fold score:", float(np.mean(fold_scores)))

# ---- cell 17 (code) ----
# =========================
# OOF diagnostics + blend optimization
# =========================
test_ranker = np.mean(test_ranker_folds, axis=0)
test_ridge = np.mean(test_ridge_folds, axis=0)
test_memory = np.mean(test_memory_folds, axis=0)

train_semantic = semantic_score[:n_train]
test_semantic = semantic_score[n_train:]
train_reranker = rerank_scores[:n_train]
test_reranker = rerank_scores[n_train:]

components_oof = {
    "ranker": oof_ranker,
    "ridge": oof_ridge,
    "semantic": train_semantic,
    "reranker": train_reranker,
    "memory": oof_memory,
}
components_test = {
    "ranker": test_ranker,
    "ridge": test_ridge,
    "semantic": test_semantic,
    "reranker": test_reranker,
    "memory": test_memory,
}

print("Individual OOF NDCG@5")
for name, pred in components_oof.items():
    print(f"{name:10s}: {fast_ndcg_at_k(Y_rel, pred, 5):.6f}")

# Row-wise z-normalization keeps blend scales comparable while preserving each component's ranking.
def row_zscore(x):
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    return ((x - mu) / np.maximum(sd, 1e-8)).astype(np.float32)

names = list(components_oof)
Zoof = [row_zscore(components_oof[n]) for n in names]
Ztest = [row_zscore(components_test[n]) for n in names]

rng = np.random.default_rng(SEED)
weight_candidates = []

# Deterministic anchors.
for i in range(len(names)):
    w = np.zeros(len(names), dtype=np.float64)
    w[i] = 1.0
    weight_candidates.append(w)
weight_candidates += [
    np.array([0.60, 0.10, 0.05, 0.10, 0.15]),
    np.array([0.70, 0.10, 0.03, 0.07, 0.10]),
    np.array([0.55, 0.15, 0.05, 0.10, 0.15]),
]

# OOF-only constrained random search. The Dirichlet prior favors the supervised ranker.
alpha = np.array([7.0, 2.0, 1.2, 1.8, 2.5])
weight_candidates.extend(rng.dirichlet(alpha, size=700))

best_score = -1.0
best_w = None
for w in weight_candidates:
    w = np.asarray(w, dtype=np.float64)
    w = w / w.sum()
    blend = sum(float(wi) * z for wi, z in zip(w, Zoof))
    score = fast_ndcg_at_k(Y_rel, blend, 5)
    if score > best_score:
        best_score = score
        best_w = w.copy()

print("\nBest OOF blend NDCG@5:", round(best_score, 6))
print("Blend weights:")
for n, w in zip(names, best_w):
    print(f"  {n:10s}: {w:.4f}")

final_test_scores = sum(float(wi) * z for wi, z in zip(best_w, Ztest)).astype(np.float32)

# ---- cell 18 (code) ----
# =========================
# Submission
# =========================
submission = sample.copy()
submission[module_cols] = final_test_scores

out_path = Path("/kaggle/working/submission_ultimate_baseline.csv")
submission.to_csv(out_path, index=False)

# Save useful diagnostics for future iterations.
np.save(CACHE_DIR / "oof_ranker.npy", oof_ranker)
np.save(CACHE_DIR / "oof_final_blend.npy", sum(float(wi) * z for wi, z in zip(best_w, Zoof)).astype(np.float32))
with open(CACHE_DIR / "blend_weights.json", "w") as f:
    json.dump({n: float(w) for n, w in zip(names, best_w)}, f, indent=2)

print("Saved:", out_path)
print("shape:", submission.shape)
assert submission.shape == sample.shape
assert submission["user_id"].equals(sample["user_id"])
assert np.isfinite(submission[module_cols].to_numpy()).all()

display(submission.head())

# Sanity-check top-5 recommendation distribution.
top5_idx = np.argsort(-final_test_scores, axis=1)[:, :5]
top5_counts = np.bincount(top5_idx.reshape(-1), minlength=len(module_cols))
display(pd.DataFrame({
    "module_id": module_cols,
    "module_name": modules["module_name"],
    "top5_frequency": top5_counts,
}).sort_values("top5_frequency", ascending=False))

# ---- cell 19 (markdown) ----
# ## What to try after this baseline
#
# Do **not** change everything at once. The first leaderboard experiments should isolate one source of improvement:
#
# 1. Switch `EMBED_MODEL` to `Qwen/Qwen3-Embedding-4B` if runtime/model-download budget is comfortable.
# 2. Try the 4B reranker only after measuring whether the 0.6B reranker adds OOF lift.
# 3. Tune the template-memory shrinkage and recency weighting.
# 4. Add a second ranker (CatBoost YetiRank or XGBoost rank:ndcg) and blend only if OOF genuinely improves.
# 5. Inspect per-module OOF errors, especially rare top modules such as Web Scrapping and NLP.
# 6. Keep the exact same CV split for all ablations so leaderboard decisions are not driven by validation noise.
#
# The key baseline principle is **ranking first, semantics second, task-specific supervision third**: pretrained models create useful representations, but the final decision is learned directly against the competition's user-level ranking objective.
