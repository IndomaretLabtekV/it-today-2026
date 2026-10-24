# Converted from notebooks/v3.ipynb: markdown cells become comments, %/! magics neutralized.

# ---- cell 0 (markdown) ----
# # MineToday IT Today 2026 — V3 Listwise Cross-Encoder Ranker
#
# **Goal:** rank 17 Intelligo modules for each user and maximize **NDCG@5**.
#
# V3 is intentionally a different architecture from V1/V2. The core model is now a **module-aware supervised cross-encoder**: every learner profile is scored jointly with every module description, so the model can learn *why this specific course is the next step for this specific learner* instead of mapping a user into 17 fixed class logits.
#
# ### What changed after the first two leaderboard probes
#
# - V1 OOF `0.6405` → public LB `0.63629`.
# - V2 OOF `0.6553` → public LB `0.64413`.
# - V2's strongest classical path was raw sparse supervised text, while the user-only GTE classifier did not transfer as strongly as its OOF suggested.
# - The target is a **17-candidate grouped ranking** problem: exactly one module has relevance `1.0`, and each user has only 4–6 relevant modules.
# - Repeated/synthetic language means lexical models remain valuable; at the same time, course-specific text interaction is missing from V2.
# - Administrative chat (passwords, invoices, certificates, LMS issues, etc.) is filtered from the *learning-intent* view but retained as numeric activity metadata.
#
# ### V3 architecture
#
# 1. **Noise-filtered sparse specialist**
#    - word + character TF-IDF,
#    - all-study-history and recent-study-history views,
#    - alpha selected by fixed 5-fold OOF,
#    - multiple targets: raw relevance, ordinal ranking, top-1, binary relevance,
#    - full-data refit for test instead of relying only on 80%-train fold bags.
#
# 2. **`Alibaba-NLP/gte-multilingual-reranker-base` supervised cross-encoder**
#    - learner profile = current intent + recent learning messages + older history + assessment,
#    - candidate = module name + syllabus + prerequisite,
#    - LoRA fine-tuning of the pretrained multilingual reranker,
#    - numeric branch for assessment/readiness/prerequisite features,
#    - grouped 17-item training.
#
# 3. **NDCG-oriented multi-objective loss**
#    - LambdaRank-style pairwise loss weighted by approximate ΔNDCG@5,
#    - ListNet listwise loss,
#    - explicit top-1 cross entropy,
#    - small ordinal regression regularizer.
#
# 4. **Low-degree, fold-stable blending**
#    - sparse and cross-encoder scores are normalized per user,
#    - blend weight for each validation fold is selected using the *other* folds,
#    - final test weight is the median fold-selected weight,
#    - optional per-module affine calibration is only used if cross-fitted OOF proves a stable gain.
#
# ### Research basis
#
# - **mGTE / GTE multilingual reranker (Zhang et al., 2024)**: multilingual encoder-only reranking, 306M parameters, >70 languages, up to 8192 tokens.
# - **LambdaLoss (Wang et al., CIKM 2018)**: metric-driven ranking losses connected directly to NDCG/LambdaRank.
# - **NeuralNDCG (Pobrotyn & Białobrzeski, 2021)**: motivates reducing train/evaluation mismatch by optimizing ranking-aware surrogates.
# - **MIND (Li et al., CIKM 2019)**: motivates preserving multiple/current interests instead of compressing an entire interaction history into one undifferentiated representation.
#
# ### Kaggle setup
#
# - Accelerator: **GPU T4 x2**
# - Internet: **ON**
# - Run all cells top-to-bottom.
# - Main output: `/kaggle/working/submission_v3_primary.csv`
# - Two controlled alternatives are also written for the daily 3-submission budget:
#   - `/kaggle/working/submission_v3_uncalibrated.csv`
#   - `/kaggle/working/submission_v3_sparse_safe.csv`
#
# The competition rules allow open-source pretrained models; no external task-specific dataset is used.

# ---- cell 1 (code) ----
# Kaggle currently ships an old optional torchao build that can conflict with PEFT.
# V3 does not use TorchAO quantization, so remove it and pin the stack used by the notebook.
# MAGIC: %pip uninstall -q -y torchao
# MAGIC: %pip install -q -U "transformers==4.57.6" "peft==0.19.1" "accelerate==1.14.0" "lightgbm>=4.5.0,<5" "sentencepiece>=0.2.0"

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
from contextlib import nullcontext

import numpy as np
import pandas as pd

from scipy.sparse import hstack as sp_hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONHASHSEED"] = "2026"

SEED = 2026

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()

print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}:", torch.cuda.get_device_name(i))

# -------------------------
# Core configuration
# -------------------------
N_FOLDS = 5
N_MODULES_EXPECTED = 17

# Sparse specialist
ALPHA_GRID = [0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
WORD_MAX_FEATURES = 80_000
CHAR_MAX_FEATURES = 100_000
RECENT_WORD_MAX_FEATURES = 55_000
RECENT_CHAR_MAX_FEATURES = 70_000

# Cross encoder
CE_MODEL = "Alibaba-NLP/gte-multilingual-reranker-base"
CE_MAX_LENGTH = 256
CE_EPOCHS = 3
CE_USER_BATCH = 2               # 2 users = 34 user-module pairs, split over 2 T4s
CE_EVAL_USER_BATCH = 4
CE_GRAD_ACCUM = 8
CE_LR = 1.6e-4
CE_WEIGHT_DECAY = 0.01
CE_WARMUP_RATIO = 0.08
CE_LORA_R = 16
CE_LORA_ALPHA = 32
CE_LORA_DROPOUT = 0.05
CE_MAX_GRAD_NORM = 1.0
CE_PATIENCE = 1
RUN_CROSS_ENCODER = True

# Loss weights
W_LAMBDA = 0.55
W_LISTNET = 0.20
W_TOP1 = 0.20
W_ORDINAL = 0.05

CACHE_VERSION = "v3_20260820_listwise_ce_r1"
CACHE_DIR = Path("/kaggle/working/minetoday_v3_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---- cell 3 (code) ----
# =========================
# Locate and load competition data
# =========================
REQUIRED_FILES = {
    "chat_history.csv",
    "modules_catalog.csv",
    "user_assessments.csv",
    "train_relevance.csv",
    "sample_submission.csv",
    "test.csv",
}

def find_data_dir():
    candidates = []
    for root in [Path("/kaggle/input"), Path("/mnt/data"), Path(".")]:
        if not root.exists():
            continue
        for p in root.rglob("train_relevance.csv"):
            parent = p.parent
            if all((parent / f).exists() for f in REQUIRED_FILES):
                candidates.append(parent)
    if not candidates:
        raise FileNotFoundError(
            "Could not find all six MineToday CSV files. Attach the competition dataset to this notebook."
        )
    # Prefer Kaggle input paths, then shortest path.
    candidates = sorted(set(candidates), key=lambda p: (0 if str(p).startswith("/kaggle/input") else 1, len(str(p))))
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
assert module_cols == [c for c in sample.columns if c != "user_id"]
assert set(modules["module_id"]) == set(module_cols)
modules = modules.set_index("module_id").loc[module_cols].reset_index()

train_user_ids = train["user_id"].astype(str).tolist()
test_user_ids = test["user_id"].astype(str).tolist()
all_user_ids = train_user_ids + test_user_ids
user_to_global = {u: i for i, u in enumerate(all_user_ids)}
n_train, n_test, n_all, n_modules = len(train), len(test), len(all_user_ids), len(module_cols)
test_global_idx = np.arange(n_train, n_all)

print("chat      :", chat.shape)
print("modules   :", modules.shape)
print("assessment:", assess.shape)
print("train     :", train.shape)
print("test      :", test.shape)

assert n_train == 4000 and n_test == 1000 and n_modules == 17
assert sample["user_id"].astype(str).tolist() == test_user_ids

# ---- cell 4 (code) ----
# =========================
# Metric, targets, and fixed CV
# =========================
def ndcg_at_k(y_true, y_score, k=5):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    order = np.argsort(-y_score, axis=1)[:, :k]
    rel = np.take_along_axis(y_true, order, axis=1)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (rel * discounts).sum(axis=1)
    ideal = np.sort(y_true, axis=1)[:, ::-1][:, :k]
    idcg = (ideal * discounts).sum(axis=1)
    return float(np.mean(np.divide(dcg, idcg, out=np.zeros_like(dcg), where=idcg > 0)))

def relevance_to_ordinal(y):
    out = np.zeros_like(y, dtype=np.float32)
    for i, row in enumerate(y):
        pos = np.flatnonzero(row > 0)
        pos = pos[np.argsort(-row[pos])]
        vals = np.arange(6, 0, -1, dtype=np.float32)[:len(pos)] / 6.0
        out[i, pos] = vals
    return out

def row_zscore(x):
    x = np.asarray(x, dtype=np.float32)
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    return ((x - mu) / np.maximum(sd, 1e-6)).astype(np.float32)

Y_rel = train[module_cols].to_numpy(np.float32)
Y_ord = relevance_to_ordinal(Y_rel)
Y_bin = (Y_rel > 0).astype(np.float32)
top_class = Y_rel.argmax(axis=1).astype(np.int64)
Y_top = np.eye(n_modules, dtype=np.float32)[top_class]

positive_counts = Y_bin.sum(axis=1).astype(int)
assert set(np.unique(positive_counts)).issubset({4, 5, 6})
assert np.all((Y_rel == 1.0).sum(axis=1) == 1)

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
folds = list(cv.split(np.arange(n_train), top_class))
fold_id = np.full(n_train, -1, dtype=np.int8)
for f, (_, va_idx) in enumerate(folds):
    fold_id[va_idx] = f
assert np.all(fold_id >= 0)

print("Positive modules/user:", pd.Series(positive_counts).value_counts().sort_index().to_dict())
print("Fold sizes:", np.bincount(fold_id).tolist())

display(pd.DataFrame({
    "module_id": module_cols,
    "module_name": modules["module_name"],
    "top1_count": np.bincount(top_class, minlength=n_modules),
    "positive_count": Y_bin.sum(axis=0).astype(int),
    "mean_relevance": Y_rel.mean(axis=0),
}).sort_values("top1_count", ascending=False))

# ---- cell 5 (code) ----
# =========================
# Assessment cleaning + module-specific readiness/gap features
# =========================
WORD_TO_NUM = {
    "nol": 0.0, "satu": 1.0, "dua": 2.0, "tiga": 3.0, "empat": 4.0, "lima": 5.0,
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

assess = assess.copy()
parsed_json = assess["assessment_result"].map(json.loads)
question_names = list(parsed_json.iloc[0].keys())
raw_q = pd.DataFrame(parsed_json.tolist(), columns=question_names)

parsed_q = raw_q.applymap(parse_assessment_value)
invalid_q = parsed_q.isna().astype(np.float32)
train_ass_mask = assess["user_id"].astype(str).isin(set(train_user_ids)).to_numpy()
train_medians = parsed_q.loc[train_ass_mask].median(axis=0)
clean_q = parsed_q.fillna(train_medians)

skill_names = [
    "Python", "SQL", "Statistics", "EDA", "Machine Learning",
    "ML Evaluation", "Deep Learning", "Generative AI / LLM",
    "Business-to-DS Translation", "Independent Research / Troubleshooting",
]
assert len(skill_names) == len(question_names) == 10

clean_df = pd.DataFrame(clean_q.to_numpy(np.float32), index=assess["user_id"].astype(str), columns=skill_names)
invalid_df = pd.DataFrame(invalid_q.to_numpy(np.float32), index=assess["user_id"].astype(str), columns=skill_names)
clean_df = clean_df.loc[all_user_ids]
invalid_df = invalid_df.loc[all_user_ids]
assessment_values = clean_df.to_numpy(np.float32)
assessment_invalid = invalid_df.to_numpy(np.float32)

assessment_stats = np.column_stack([
    assessment_values.mean(axis=1),
    assessment_values.std(axis=1),
    assessment_values.min(axis=1),
    assessment_values.max(axis=1),
    (assessment_values <= 1).sum(axis=1),
    (assessment_values >= 3).sum(axis=1),
    (assessment_values >= 4).sum(axis=1),
    assessment_invalid.sum(axis=1),
]).astype(np.float32)

# Skill mappings are grounded in the supplied module names/syllabi/prerequisites.
MODULE_DIRECT_SKILLS = {
    "M_001": [8], "M_002": [0], "M_003": [1], "M_004": [0], "M_005": [9],
    "M_006": [2], "M_007": [3], "M_008": [8], "M_009": [4], "M_010": [6],
    "M_011": [4], "M_012": [7], "M_013": [7], "M_014": [7, 8],
    "M_015": [7], "M_016": [4, 9], "M_017": [8, 9],
}
MODULE_PREREQ_SKILLS = {
    "M_001": [], "M_002": [], "M_003": [], "M_004": [0], "M_005": [],
    "M_006": [], "M_007": [0, 2], "M_008": [1], "M_009": [2, 3],
    "M_010": [4, 6], "M_011": [4], "M_012": [4], "M_013": [],
    "M_014": [], "M_015": [], "M_016": [0, 4, 9], "M_017": [],
}
LEVEL_MAP = {"pemula": 1.0, "menengah": 2.5, "lanjutan": 3.5, "ahli": 4.25, "semua level": 2.5}

module_level = []
for row in modules.itertuples(index=False):
    s = str(row.prerequisite_level).lower()
    matched = [v for k, v in LEVEL_MAP.items() if k in s]
    module_level.append(matched[0] if matched else 2.5)
module_level = np.asarray(module_level, dtype=np.float32)

def rule_features_for_users(user_indices):
    user_indices = np.asarray(user_indices, dtype=np.int64)
    out = np.zeros((len(user_indices), n_modules, 12), dtype=np.float32)
    for r, ui in enumerate(user_indices):
        a = assessment_values[ui]
        global_mean = float(a.mean())
        for m, module_id in enumerate(module_cols):
            direct = MODULE_DIRECT_SKILLS[module_id]
            prereq = MODULE_PREREQ_SKILLS[module_id]
            direct_vals = a[direct] if direct else np.asarray([global_mean], dtype=np.float32)
            prereq_vals = a[prereq] if prereq else np.asarray([global_mean], dtype=np.float32)
            lvl = float(module_level[m])
            out[r, m] = [
                direct_vals.mean(), direct_vals.min(), direct_vals.max(),
                prereq_vals.mean(), prereq_vals.min(), prereq_vals.max(),
                max(0.0, lvl - direct_vals.mean()),
                max(0.0, lvl - prereq_vals.mean()),
                direct_vals.mean() - lvl,
                prereq_vals.mean() - lvl,
                global_mean - lvl,
                lvl,
            ]
    return out

rule_all = rule_features_for_users(np.arange(n_all))
assert rule_all.shape == (n_all, n_modules, 12)

print("Assessment invalid cells:", int(assessment_invalid.sum()), "/", assessment_invalid.size)

# ---- cell 6 (code) ----
# =========================
# Chat chronology, administrative-noise filtering, and multi-interest views
# =========================
ADMIN_PATTERNS = [
    r"sertifikat", r"certificate", r"pembayaran", r"cicilan", r"invoice", r"reimburse",
    r"reset password", r"lupa password", r"ganti password", r"email verifikasi",
    r"verifikasi email", r"login", r"website.*down", r"lms.*down", r"error.*lms",
    r"akun.*error", r"akses kelas", r"jadwal pembayaran", r"refund",
]
ADMIN_RE = re.compile("|".join(f"(?:{p})" for p in ADMIN_PATTERNS), flags=re.I)

def normalize_chat_text(s):
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = re.sub(r"\d+", " <num> ", s)
    s = re.sub(r"[^a-z0-9<>]+", " ", s)
    return " ".join(s.split())

chat = chat.copy()
chat["user_id"] = chat["user_id"].astype(str)
chat = chat[chat["user_id"].isin(set(all_user_ids))].copy()
chat["timestamp"] = pd.to_datetime(chat["timestamp"], errors="coerce")
chat["text"] = chat["user_chat_text"].fillna("").astype(str)
chat["norm_text"] = chat["text"].map(normalize_chat_text)
chat["is_admin"] = chat["text"].map(lambda s: bool(ADMIN_RE.search(s)))
chat = chat.sort_values(["user_id", "timestamp", "chat_id"]).reset_index(drop=True)

message_lists = defaultdict(list)
study_lists = defaultdict(list)
timestamp_lists = defaultdict(list)
for row in chat.itertuples(index=False):
    message_lists[row.user_id].append(row.text)
    timestamp_lists[row.user_id].append(row.timestamp)
    if not row.is_admin:
        study_lists[row.user_id].append(row.text)

# If every message of a user is administrative/noisy, fall back to raw history.
for u in all_user_ids:
    if not study_lists[u] and message_lists[u]:
        study_lists[u] = list(message_lists[u])

chat_stats = np.zeros((n_all, 10), dtype=np.float32)
sparse_texts, recent_texts, ce_queries = [], [], []

for i, u in enumerate(all_user_ids):
    raw = message_lists[u]
    study = study_lists[u]
    ts = [x for x in timestamp_lists[u] if pd.notna(x)]
    admin_count = max(len(raw) - len(study), 0)
    uniq = len(set(normalize_chat_text(x) for x in raw if x))
    span_days = float((max(ts) - min(ts)).days) if len(ts) >= 2 else 0.0

    current = study[-1] if study else "No learning chat available."
    recent = study[-3:] if study else []
    older = study[:-3] if len(study) > 3 else []

    assessment_tokens = " ".join(
        f"skill_{j}_{int(round(v))}" for j, v in enumerate(assessment_values[i])
    )
    assessment_natural = "; ".join(
        f"{name}={int(round(v))}/5" for name, v in zip(skill_names, assessment_values[i])
    )

    all_study = " [MSG] ".join(study) if study else "no learning chat"
    recent_study = " [MSG] ".join(recent) if recent else "no recent learning chat"

    # Sparse view keeps synthetic phrasing intact and adds discrete assessment tokens.
    sparse_texts.append(f"{all_study} [ASSESS] {assessment_tokens}")
    recent_texts.append(f"{recent_study} [ASSESS] {assessment_tokens}")

    # Cross-encoder sees current intent first, then multiple historical interests.
    ce_queries.append(
        "[CURRENT LEARNING INTENT] " + current +
        "\n[RECENT LEARNING CONTEXT] " + (" [SEP] ".join(recent[:-1]) if len(recent) > 1 else "none") +
        "\n[ASSESSMENT] " + assessment_natural +
        "\n[OLDER LEARNING HISTORY] " + (" [SEP] ".join(older[-6:]) if older else "none")
    )

    chars = sum(len(x) for x in raw)
    words = sum(len(x.split()) for x in raw)
    study_chars = sum(len(x) for x in study)
    chat_stats[i] = [
        len(raw), len(study), admin_count, uniq, span_days,
        chars, words, study_chars,
        len(current), len(current.split()),
    ]

module_docs = [
    (
        f"Course: {r.module_name}. "
        f"Syllabus: {r.description_and_syllabus}. "
        f"Prerequisite: {r.prerequisite_level}."
    )
    for r in modules.itertuples(index=False)
]

# Numeric user + user-module features for the cross-encoder additive branch.
numeric_user = np.concatenate([
    assessment_values / 5.0,
    assessment_invalid,
    assessment_stats / np.asarray([5, 2.5, 5, 5, 10, 10, 10, 10], dtype=np.float32)[None, :],
    np.log1p(chat_stats),
], axis=1).astype(np.float32)

module_onehot = np.eye(n_modules, dtype=np.float32)
pair_numeric_all = []
for i in range(n_all):
    repeated = np.repeat(numeric_user[i][None, :], n_modules, axis=0)
    pair_numeric_all.append(np.concatenate([repeated, rule_all[i] / 5.0, module_onehot], axis=1))
pair_numeric_all = np.stack(pair_numeric_all).astype(np.float32)

print("Administrative/noise rows:", int(chat["is_admin"].sum()), "/", len(chat))
print("Cross-encoder query example:\n", ce_queries[1])
print("pair_numeric_all:", pair_numeric_all.shape)

# ---- cell 7 (markdown) ----
# ## Component A — noise-filtered sparse specialist
#
# V2 showed that templated language is unusually predictive. V3 therefore treats sparse NLP as a first-class model rather than a fallback:
#
# - word 1–3 grams + character 3–5 grams,
# - full learning-history view + last-three-learning-message view,
# - ridge regularization selected by the exact fixed CV,
# - multiple supervision heads,
# - full 4,000-user refit for test predictions.
#
# This component is intentionally independent of the transformer and gives a low-variance anchor for leaderboard transfer.

# ---- cell 8 (code) ----
# =========================
# Sparse matrices
# =========================
word_vec = TfidfVectorizer(
    lowercase=True, strip_accents="unicode", ngram_range=(1, 3),
    min_df=2, max_df=0.997, max_features=WORD_MAX_FEATURES,
    sublinear_tf=True, dtype=np.float32,
)
char_vec = TfidfVectorizer(
    lowercase=True, analyzer="char_wb", ngram_range=(3, 5),
    min_df=2, max_features=CHAR_MAX_FEATURES,
    sublinear_tf=True, dtype=np.float32,
)
recent_word_vec = TfidfVectorizer(
    lowercase=True, strip_accents="unicode", ngram_range=(1, 3),
    min_df=2, max_df=0.997, max_features=RECENT_WORD_MAX_FEATURES,
    sublinear_tf=True, dtype=np.float32,
)
recent_char_vec = TfidfVectorizer(
    lowercase=True, analyzer="char_wb", ngram_range=(3, 5),
    min_df=2, max_features=RECENT_CHAR_MAX_FEATURES,
    sublinear_tf=True, dtype=np.float32,
)

# Transductive vocabulary/IDF uses only supplied competition features, never test labels.
X_word = word_vec.fit_transform(sparse_texts)
X_char = char_vec.fit_transform(sparse_texts)
X_sparse = sp_hstack([X_word, X_char], format="csr")

X_rword = recent_word_vec.fit_transform(recent_texts)
X_rchar = recent_char_vec.fit_transform(recent_texts)
X_recent = sp_hstack([X_rword, X_rchar], format="csr")

print("X_sparse:", X_sparse.shape, "nnz:", X_sparse.nnz)
print("X_recent:", X_recent.shape, "nnz:", X_recent.nnz)

# -------------------------
# Select Ridge alpha using ordinal target
# -------------------------
alpha_results = {}
alpha_oof_cache = {}

for alpha in ALPHA_GRID:
    oof = np.zeros((n_train, n_modules), dtype=np.float32)
    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(folds):
        model = Ridge(alpha=alpha, solver="sparse_cg", tol=1e-3)
        model.fit(X_sparse[tr_idx], Y_ord[tr_idx])
        pv = model.predict(X_sparse[va_idx]).astype(np.float32)
        oof[va_idx] = pv
        fold_scores.append(ndcg_at_k(Y_rel[va_idx], pv, 5))
        del model
    alpha_results[alpha] = (float(np.mean(fold_scores)), float(np.std(fold_scores)))
    alpha_oof_cache[alpha] = oof
    print(f"alpha={alpha:<5} mean={np.mean(fold_scores):.6f} std={np.std(fold_scores):.6f}")

best_alpha = max(alpha_results, key=lambda a: (alpha_results[a][0], -alpha_results[a][1]))
print("Selected sparse alpha:", best_alpha, alpha_results[best_alpha])

# -------------------------
# Multi-head OOF
# -------------------------
sparse_targets = {
    "raw": Y_rel,
    "ordinal": Y_ord,
    "top1": Y_top,
    "binary": Y_bin,
}
head_names = ["raw", "ordinal", "top1", "binary", "recent"]
sparse_oof_heads = {n: np.zeros((n_train, n_modules), dtype=np.float32) for n in head_names}

# Reuse selected ordinal OOF from the alpha search.
sparse_oof_heads["ordinal"][:] = alpha_oof_cache[best_alpha]

for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
    print(f"Sparse multi-head fold {fold}/{N_FOLDS}")
    for name in ["raw", "top1", "binary"]:
        model = Ridge(alpha=best_alpha, solver="sparse_cg", tol=1e-3)
        model.fit(X_sparse[tr_idx], sparse_targets[name][tr_idx])
        sparse_oof_heads[name][va_idx] = model.predict(X_sparse[va_idx]).astype(np.float32)
        del model

    recent_model = Ridge(alpha=best_alpha, solver="sparse_cg", tol=1e-3)
    recent_model.fit(X_recent[tr_idx], Y_ord[tr_idx])
    sparse_oof_heads["recent"][va_idx] = recent_model.predict(X_recent[va_idx]).astype(np.float32)
    del recent_model
    gc.collect()

print("\nSparse head OOF NDCG@5")
for n in head_names:
    print(f"{n:10s}: {ndcg_at_k(Y_rel, sparse_oof_heads[n], 5):.6f}")

# -------------------------
# Low-dimensional, fold-stable head blend
# -------------------------
rng = np.random.default_rng(SEED + 101)
candidate_weights = [
    np.eye(len(head_names))[i] for i in range(len(head_names))
]
candidate_weights += [
    np.array([0.15, 0.35, 0.20, 0.10, 0.20]),
    np.array([0.10, 0.40, 0.20, 0.10, 0.20]),
    np.array([0.20, 0.35, 0.15, 0.10, 0.20]),
]
candidate_weights += list(rng.dirichlet([2.5, 5.0, 3.0, 1.5, 3.0], size=600))

Zoof_heads = [row_zscore(sparse_oof_heads[n]) for n in head_names]
fold_head_weights = []
sparse_oof = np.zeros((n_train, n_modules), dtype=np.float32)

for f in range(N_FOLDS):
    tune_idx = np.flatnonzero(fold_id != f)
    val_idx = np.flatnonzero(fold_id == f)
    best_s, best_w = -1.0, None
    for w in candidate_weights:
        w = np.asarray(w, dtype=np.float64)
        w = w / w.sum()
        p = sum(float(wi) * z[tune_idx] for wi, z in zip(w, Zoof_heads))
        s = ndcg_at_k(Y_rel[tune_idx], p, 5)
        if s > best_s:
            best_s, best_w = s, w
    fold_head_weights.append(best_w)
    sparse_oof[val_idx] = sum(float(wi) * z[val_idx] for wi, z in zip(best_w, Zoof_heads))

mean_head_w = np.mean(np.stack(fold_head_weights), axis=0)
mean_head_w /= mean_head_w.sum()

print("Fold-stable sparse weights:", {n: round(float(w), 4) for n, w in zip(head_names, mean_head_w)})
print("Sparse cross-fitted OOF NDCG@5:", round(ndcg_at_k(Y_rel, sparse_oof, 5), 6))

# -------------------------
# Full-data refit for test
# -------------------------
sparse_test_heads_full = {}

for name in ["raw", "ordinal", "top1", "binary"]:
    model = Ridge(alpha=best_alpha, solver="sparse_cg", tol=1e-3)
    model.fit(X_sparse[:n_train], sparse_targets[name])
    sparse_test_heads_full[name] = model.predict(X_sparse[n_train:]).astype(np.float32)
    del model

recent_model = Ridge(alpha=best_alpha, solver="sparse_cg", tol=1e-3)
recent_model.fit(X_recent[:n_train], Y_ord)
sparse_test_heads_full["recent"] = recent_model.predict(X_recent[n_train:]).astype(np.float32)
del recent_model
gc.collect()

Ztest_heads = [row_zscore(sparse_test_heads_full[n]) for n in head_names]
sparse_test = sum(float(wi) * z for wi, z in zip(mean_head_w, Ztest_heads)).astype(np.float32)

assert sparse_oof.shape == (n_train, n_modules)
assert sparse_test.shape == (n_test, n_modules)
assert np.isfinite(sparse_oof).all() and np.isfinite(sparse_test).all()

# ---- cell 9 (markdown) ----
# ## Component B — supervised listwise multilingual cross-encoder
#
# The reranker consumes **learner + candidate module together**. This is the central correction from V2: module semantics are inside the supervised interaction, which gives rare modules a learnable lexical/semantic route even when they have few top-1 examples.
#
# Each training example is one **user group of 17 candidates**. The loss directly rewards the ordering inside that group.

# ---- cell 10 (code) ----
# =========================
# Cross-encoder dataset, model, and NDCG-oriented losses
# =========================
class UserIndexDataset(Dataset):
    def __init__(self, indices):
        self.indices = np.asarray(indices, dtype=np.int64)
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, i):
        return int(self.indices[i])

tokenizer = AutoTokenizer.from_pretrained(CE_MODEL, trust_remote_code=True)

def make_collate(include_targets):
    def collate(user_indices):
        idx = np.asarray(user_indices, dtype=np.int64)
        queries = []
        passages = []
        for ui in idx:
            queries.extend([ce_queries[ui]] * n_modules)
            passages.extend(module_docs)
        tokens = tokenizer(
            queries,
            passages,
            padding=True,
            truncation=True,
            max_length=CE_MAX_LENGTH,
            return_tensors="pt",
        )
        numeric = torch.from_numpy(pair_numeric_all[idx].reshape(-1, pair_numeric_all.shape[-1]))
        out = {
            "indices": torch.from_numpy(idx),
            "tokens": tokens,
            "numeric": numeric,
        }
        if include_targets:
            assert np.all(idx < n_train)
            out["rel"] = torch.from_numpy(Y_rel[idx])
            out["ord"] = torch.from_numpy(Y_ord[idx])
            out["top1"] = torch.from_numpy(top_class[idx])
        return out
    return collate

def make_loader(indices, training=False, include_targets=True):
    return DataLoader(
        UserIndexDataset(indices),
        batch_size=CE_USER_BATCH if training else CE_EVAL_USER_BATCH,
        shuffle=training,
        num_workers=0,
        pin_memory=True,
        collate_fn=make_collate(include_targets),
    )

def dcg_idcg_torch(rel, k=5):
    k = min(k, rel.shape[1])
    discounts = 1.0 / torch.log2(torch.arange(2, k + 2, device=rel.device, dtype=rel.dtype))
    ideal = torch.sort(rel, dim=1, descending=True).values[:, :k]
    return (ideal * discounts).sum(dim=1).clamp_min(1e-6)

def lambda_ndcg_loss(scores, rel, k=5):
    # LambdaRank-style pairwise logistic loss weighted by |ΔNDCG@5|.
    # Predicted positions are detached; gradients flow through score differences.
    B, M = scores.shape
    order = torch.argsort(scores.detach(), dim=1, descending=True)
    ranks = torch.empty_like(order)
    positions = torch.arange(M, device=scores.device).expand(B, M)
    ranks.scatter_(1, order, positions)

    discounts = torch.zeros((B, M), device=scores.device, dtype=scores.dtype)
    in_topk = ranks < k
    safe_rank = ranks.to(scores.dtype)
    discounts[in_topk] = 1.0 / torch.log2(safe_rank[in_topk] + 2.0)

    rel_i = rel.unsqueeze(2)
    rel_j = rel.unsqueeze(1)
    diff_rel = rel_i - rel_j
    pref = diff_rel > 1e-8

    sdiff = scores.unsqueeze(2) - scores.unsqueeze(1)
    ddiff = torch.abs(discounts.unsqueeze(2) - discounts.unsqueeze(1))
    idcg = dcg_idcg_torch(rel, k=k).view(B, 1, 1)
    weight = torch.abs(diff_rel) * ddiff / idcg

    pair_loss = F.softplus(-sdiff)
    weighted = pair_loss * weight * pref.to(scores.dtype)
    denom = (weight * pref.to(scores.dtype)).sum().clamp_min(1e-6)
    return weighted.sum() / denom

def listnet_loss(scores, rel):
    target = F.softmax(rel / 0.20, dim=1)
    logp = F.log_softmax(scores, dim=1)
    return -(target * logp).sum(dim=1).mean()

def ordinal_reg_loss(scores, ord_target):
    s = (scores - scores.mean(dim=1, keepdim=True)) / scores.std(dim=1, keepdim=True).clamp_min(1e-4)
    t = (ord_target - ord_target.mean(dim=1, keepdim=True)) / ord_target.std(dim=1, keepdim=True).clamp_min(1e-4)
    return F.smooth_l1_loss(s, t)

def total_rank_loss(scores, rel, ord_target, top1):
    l_lambda = lambda_ndcg_loss(scores, rel, k=5)
    l_list = listnet_loss(scores, rel)
    l_top = F.cross_entropy(scores, top1)
    l_ord = ordinal_reg_loss(scores, ord_target)
    total = W_LAMBDA*l_lambda + W_LISTNET*l_list + W_TOP1*l_top + W_ORDINAL*l_ord
    return total, {
        "lambda": float(l_lambda.detach()),
        "list": float(l_list.detach()),
        "top1": float(l_top.detach()),
        "ordinal": float(l_ord.detach()),
    }

class CourseCrossEncoder(nn.Module):
    def __init__(self, reranker, numeric_dim):
        super().__init__()
        self.reranker = reranker
        self.numeric_head = nn.Sequential(
            nn.LayerNorm(numeric_dim),
            nn.Linear(numeric_dim, 64),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(64, 1),
        )
        # Start from the pretrained reranker score; let numeric correction grow gradually.
        nn.init.zeros_(self.numeric_head[-1].weight)
        nn.init.zeros_(self.numeric_head[-1].bias)

    def forward(self, numeric, **tokens):
        out = self.reranker(**tokens)
        text_score = out.logits.reshape(-1).float()
        numeric_score = self.numeric_head(numeric.float()).reshape(-1)
        return text_score + numeric_score

def build_cross_encoder():
    base = AutoModelForSequenceClassification.from_pretrained(
        CE_MODEL,
        trust_remote_code=True,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    if hasattr(base, "gradient_checkpointing_enable"):
        try:
            base.gradient_checkpointing_enable()
        except Exception:
            pass
    if hasattr(base.config, "use_cache"):
        base.config.use_cache = False

    module_names = [n for n, _ in base.named_modules()]
    if any(n.endswith("qkv_proj") for n in module_names) and any(n.endswith("o_proj") for n in module_names):
        targets = ["qkv_proj", "o_proj"]
    else:
        # Fail loudly rather than silently training no adapters if upstream custom code changes.
        tail = [n for n in module_names if any(x in n.lower() for x in ["query", "key", "value", "q_proj", "k_proj", "v_proj"])][:30]
        raise RuntimeError(f"Expected GTE attention projections qkv_proj/o_proj not found. Candidate names: {tail}")

    classifier_exists = any("classifier" in n for n, _ in base.named_parameters())
    assert classifier_exists, "Reranker classifier head not found"

    lora_cfg = LoraConfig(
        r=CE_LORA_R,
        lora_alpha=CE_LORA_ALPHA,
        target_modules=targets,
        lora_dropout=CE_LORA_DROPOUT,
        bias="none",
        modules_to_save=["classifier"],
    )
    peft_model = get_peft_model(base, lora_cfg)
    trainable_lora = [n for n, p in peft_model.named_parameters() if p.requires_grad and "lora_" in n]
    assert trainable_lora, "LoRA adapter injection failed"

    model = CourseCrossEncoder(peft_model, pair_numeric_all.shape[-1])
    return model

# Loss regression guard before expensive model training.
_test_scores = torch.randn(3, n_modules, requires_grad=True)
_test_rel = torch.from_numpy(Y_rel[:3])
_test_ord = torch.from_numpy(Y_ord[:3])
_test_top = torch.from_numpy(top_class[:3])
_test_loss, _ = total_rank_loss(_test_scores, _test_rel, _test_ord, _test_top)
assert torch.isfinite(_test_loss)
_test_loss.backward()
assert torch.isfinite(_test_scores.grad).all()
print("Listwise/Lambda loss self-check OK:", float(_test_loss))

# ---- cell 11 (code) ----
# =========================
# Cross-encoder training helpers
# =========================
def unwrap(model):
    return model.module if isinstance(model, nn.DataParallel) else model

def capture_trainable_state(model):
    core = unwrap(model)
    return {n: p.detach().cpu().clone() for n, p in core.named_parameters() if p.requires_grad}

def restore_trainable_state(model, state):
    core = unwrap(model)
    named = dict(core.named_parameters())
    with torch.no_grad():
        for n, v in state.items():
            assert n in named
            named[n].copy_(v.to(named[n].device, dtype=named[n].dtype))

@torch.no_grad()
def predict_user_loader(model, loader):
    model.eval()
    out_idx, out_scores = [], []
    for batch in loader:
        idx = batch["indices"].numpy()
        tokens = {k: v.cuda(non_blocking=True) for k, v in batch["tokens"].items()}
        numeric = batch["numeric"].cuda(non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
            scores = model(numeric=numeric, **tokens).float().reshape(len(idx), n_modules)
        out_idx.append(idx)
        out_scores.append(scores.cpu().numpy().astype(np.float32))
    idx = np.concatenate(out_idx)
    pred = np.concatenate(out_scores)
    order = np.argsort(idx)
    return idx[order], pred[order]

def fit_fold_cross_encoder(fold, tr_idx, va_idx):
    seed_everything(SEED + 700 + fold)
    print(f"\n========== Cross-encoder fold {fold}/{N_FOLDS} ==========")

    model = build_cross_encoder().cuda()
    core = unwrap(model)
    trainable = sum(p.numel() for p in core.parameters() if p.requires_grad)
    total = sum(p.numel() for p in core.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print("Using DataParallel on", torch.cuda.device_count(), "GPUs")

    train_loader = make_loader(tr_idx, training=True, include_targets=True)
    val_loader = make_loader(va_idx, training=False, include_targets=True)
    test_loader = make_loader(test_global_idx, training=False, include_targets=False)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=CE_LR, weight_decay=CE_WEIGHT_DECAY)
    steps_per_epoch = math.ceil(len(train_loader) / CE_GRAD_ACCUM)
    total_steps = max(1, steps_per_epoch * CE_EPOCHS)
    warmup = int(total_steps * CE_WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)

    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    best_score = -1.0
    best_state = None
    best_epoch = 1
    stale = 0

    for epoch in range(1, CE_EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0

        for step, batch in enumerate(train_loader, start=1):
            idx = batch["indices"].numpy()
            tokens = {k: v.cuda(non_blocking=True) for k, v in batch["tokens"].items()}
            numeric = batch["numeric"].cuda(non_blocking=True)
            rel = batch["rel"].cuda(non_blocking=True).float()
            ord_target = batch["ord"].cuda(non_blocking=True).float()
            top1 = batch["top1"].cuda(non_blocking=True).long()

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                scores = model(numeric=numeric, **tokens).float().reshape(len(idx), n_modules)
                loss, parts = total_rank_loss(scores, rel, ord_target, top1)
                loss = loss / CE_GRAD_ACCUM

            scaler.scale(loss).backward()
            running += float(loss.detach()) * CE_GRAD_ACCUM

            if step % CE_GRAD_ACCUM == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, CE_MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        _, pv = predict_user_loader(model, val_loader)
        score = ndcg_at_k(Y_rel[va_idx], pv, 5)
        top1_acc = float((pv.argmax(axis=1) == top_class[va_idx]).mean())
        print(f"epoch {epoch}: loss={running/max(len(train_loader),1):.5f} val_NDCG@5={score:.6f} top1={top1_acc:.4f}")

        if score > best_score + 1e-6:
            best_score = score
            best_state = capture_trainable_state(model)
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale > CE_PATIENCE:
                break

    assert best_state is not None
    restore_trainable_state(model, best_state)
    _, pv = predict_user_loader(model, val_loader)
    _, pt = predict_user_loader(model, test_loader)
    final_score = ndcg_at_k(Y_rel[va_idx], pv, 5)
    print("best_epoch:", best_epoch, "restored NDCG@5:", round(final_score, 6))

    del train_loader, val_loader, test_loader, optimizer, scheduler, scaler, model, core, best_state
    gc.collect()
    torch.cuda.empty_cache()

    return pv.astype(np.float32), pt.astype(np.float32), int(best_epoch), float(final_score)

def fit_full_cross_encoder(num_epochs):
    seed_everything(SEED + 1900)
    print(f"\n========== Full cross-encoder refit: {num_epochs} epoch(s) ==========")
    model = build_cross_encoder().cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    train_loader = make_loader(np.arange(n_train), training=True, include_targets=True)
    test_loader = make_loader(test_global_idx, training=False, include_targets=False)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=CE_LR, weight_decay=CE_WEIGHT_DECAY)
    steps_per_epoch = math.ceil(len(train_loader) / CE_GRAD_ACCUM)
    total_steps = max(1, steps_per_epoch * num_epochs)
    warmup = int(total_steps * CE_WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for step, batch in enumerate(train_loader, start=1):
            idx = batch["indices"].numpy()
            tokens = {k: v.cuda(non_blocking=True) for k, v in batch["tokens"].items()}
            numeric = batch["numeric"].cuda(non_blocking=True)
            rel = batch["rel"].cuda(non_blocking=True).float()
            ord_target = batch["ord"].cuda(non_blocking=True).float()
            top1 = batch["top1"].cuda(non_blocking=True).long()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                scores = model(numeric=numeric, **tokens).float().reshape(len(idx), n_modules)
                loss, _ = total_rank_loss(scores, rel, ord_target, top1)
                loss = loss / CE_GRAD_ACCUM
            scaler.scale(loss).backward()
            running += float(loss.detach()) * CE_GRAD_ACCUM
            if step % CE_GRAD_ACCUM == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, CE_MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
        print(f"full epoch {epoch}: loss={running/max(len(train_loader),1):.5f}")

    _, pt = predict_user_loader(model, test_loader)
    del model, train_loader, test_loader, optimizer, scheduler, scaler
    gc.collect()
    torch.cuda.empty_cache()
    return pt.astype(np.float32)

# ---- cell 12 (code) ----
# =========================
# 5-fold cross-encoder OOF + fold-bag test
# =========================
CE_CACHE = CACHE_DIR / f"{CACHE_VERSION}_cross_encoder.npz"

def valid_ce_cache(path):
    if not path.exists():
        return False
    try:
        z = np.load(path)
        return (
            z["oof"].shape == (n_train, n_modules)
            and z["test_foldbag"].shape == (n_test, n_modules)
            and z["best_epochs"].shape == (N_FOLDS,)
            and np.isfinite(z["oof"]).all()
            and np.isfinite(z["test_foldbag"]).all()
        )
    except Exception:
        return False

if RUN_CROSS_ENCODER and valid_ce_cache(CE_CACHE):
    z = np.load(CE_CACHE)
    ce_oof = z["oof"].astype(np.float32)
    ce_test_foldbag = z["test_foldbag"].astype(np.float32)
    ce_best_epochs = z["best_epochs"].astype(int)
    ce_fold_scores = z["fold_scores"].astype(float)
    print("Loaded cross-encoder cache:", CE_CACHE)
elif RUN_CROSS_ENCODER:
    ce_oof = np.zeros((n_train, n_modules), dtype=np.float32)
    ce_test_folds = []
    ce_best_epochs = []
    ce_fold_scores = []

    # One model smoke-check before committing to all folds.
    smoke = build_cross_encoder()
    lora_names = [n for n, p in smoke.named_parameters() if p.requires_grad and "lora_" in n]
    print("LoRA injection OK; trainable adapter tensors:", len(lora_names))
    del smoke
    gc.collect()

    for fold0, (tr_idx, va_idx) in enumerate(folds):
        fold = fold0 + 1
        fold_cache = CACHE_DIR / f"{CACHE_VERSION}_ce_fold{fold}.npz"
        if fold_cache.exists():
            try:
                q = np.load(fold_cache)
                if q["val"].shape == (len(va_idx), n_modules) and q["test"].shape == (n_test, n_modules):
                    pv = q["val"].astype(np.float32)
                    pt = q["test"].astype(np.float32)
                    ep = int(q["best_epoch"])
                    sc = float(q["score"])
                    print(f"Loaded CE fold {fold} cache: NDCG={sc:.6f}")
                else:
                    raise ValueError("shape mismatch")
            except Exception:
                pv, pt, ep, sc = fit_fold_cross_encoder(fold, tr_idx, va_idx)
                np.savez_compressed(fold_cache, val=pv, test=pt, best_epoch=ep, score=sc)
        else:
            pv, pt, ep, sc = fit_fold_cross_encoder(fold, tr_idx, va_idx)
            np.savez_compressed(fold_cache, val=pv, test=pt, best_epoch=ep, score=sc)

        ce_oof[va_idx] = pv
        ce_test_folds.append(pt)
        ce_best_epochs.append(ep)
        ce_fold_scores.append(sc)

    ce_test_foldbag = np.mean(ce_test_folds, axis=0).astype(np.float32)
    ce_best_epochs = np.asarray(ce_best_epochs, dtype=np.int32)
    ce_fold_scores = np.asarray(ce_fold_scores, dtype=np.float32)
    np.savez_compressed(
        CE_CACHE,
        oof=ce_oof,
        test_foldbag=ce_test_foldbag,
        best_epochs=ce_best_epochs,
        fold_scores=ce_fold_scores,
    )
    print("Saved cross-encoder cache:", CE_CACHE)
else:
    ce_oof = np.zeros_like(sparse_oof)
    ce_test_foldbag = np.zeros_like(sparse_test)
    ce_best_epochs = np.ones(N_FOLDS, dtype=int)
    ce_fold_scores = np.zeros(N_FOLDS, dtype=float)

if RUN_CROSS_ENCODER:
    print("Cross-encoder fold scores:", [round(float(x), 6) for x in ce_fold_scores])
    print("Cross-encoder OOF NDCG@5:", round(ndcg_at_k(Y_rel, ce_oof, 5), 6))
    print("Cross-encoder top1 accuracy:", round(float((ce_oof.argmax(axis=1) == top_class).mean()), 4))

# ---- cell 13 (code) ----
# =========================
# Full-data cross-encoder refit
# =========================
FULL_CE_CACHE = CACHE_DIR / f"{CACHE_VERSION}_cross_full_test.npy"

if RUN_CROSS_ENCODER:
    full_epochs = int(np.clip(np.rint(np.median(ce_best_epochs)), 1, CE_EPOCHS))
    if FULL_CE_CACHE.exists():
        q = np.load(FULL_CE_CACHE)
        if q.shape == (n_test, n_modules) and np.isfinite(q).all():
            ce_test_full = q.astype(np.float32)
            print("Loaded full CE refit cache.")
        else:
            FULL_CE_CACHE.unlink(missing_ok=True)
            ce_test_full = fit_full_cross_encoder(full_epochs)
            np.save(FULL_CE_CACHE, ce_test_full)
    else:
        ce_test_full = fit_full_cross_encoder(full_epochs)
        np.save(FULL_CE_CACHE, ce_test_full)

    # Fold-bag is robust; full refit uses 25% more labels per model. Blend both.
    ce_test = (0.60 * ce_test_full + 0.40 * ce_test_foldbag).astype(np.float32)
else:
    ce_test = np.zeros_like(sparse_test)

assert ce_test.shape == (n_test, n_modules)
assert np.isfinite(ce_test).all()

# ---- cell 14 (markdown) ----
# ## Robust model combination
#
# The V2 OOF-to-LB gap suggests avoiding a large random blend search. V3 uses only one meaningful degree of freedom: **cross-encoder weight vs sparse weight**.
#
# For each fold, the weight is tuned on the other four folds, then applied to the held-out fold. The final test weight is the median of those five independently selected values.

# ---- cell 15 (code) ----
# =========================
# Fold-stable sparse + cross-encoder blend
# =========================
zs = row_zscore(sparse_oof)
zt_s = row_zscore(sparse_test)

if RUN_CROSS_ENCODER:
    zc = row_zscore(ce_oof)
    zt_c = row_zscore(ce_test)
else:
    zc = np.zeros_like(zs)
    zt_c = np.zeros_like(zt_s)

weight_grid = np.linspace(0.0, 0.85, 18) if RUN_CROSS_ENCODER else np.array([0.0])
fold_ce_weights = []
blend_oof = np.zeros_like(zs)

for f in range(N_FOLDS):
    tune_idx = np.flatnonzero(fold_id != f)
    val_idx = np.flatnonzero(fold_id == f)
    best_s, best_w = -1.0, 0.0
    for w_ce in weight_grid:
        p = (1.0 - w_ce) * zs[tune_idx] + w_ce * zc[tune_idx]
        s = ndcg_at_k(Y_rel[tune_idx], p, 5)
        if s > best_s:
            best_s, best_w = s, float(w_ce)
    fold_ce_weights.append(best_w)
    blend_oof[val_idx] = (1.0 - best_w) * zs[val_idx] + best_w * zc[val_idx]

final_ce_weight = float(np.median(fold_ce_weights))
uncal_test = ((1.0 - final_ce_weight) * zt_s + final_ce_weight * zt_c).astype(np.float32)

print("Fold-selected CE weights:", fold_ce_weights)
print("Median CE test weight:", final_ce_weight)
print("Sparse OOF NDCG@5:", round(ndcg_at_k(Y_rel, sparse_oof, 5), 6))
if RUN_CROSS_ENCODER:
    print("Cross-encoder OOF NDCG@5:", round(ndcg_at_k(Y_rel, ce_oof, 5), 6))
print("Cross-fitted blend OOF NDCG@5:", round(ndcg_at_k(Y_rel, blend_oof, 5), 6))

# ---- cell 16 (code) ----
# =========================
# Optional cross-fitted per-module affine calibration
# =========================
def crossfit_affine_calibration(score_oof, score_test, alpha=20.0):
    cal_oof = np.zeros_like(score_oof, dtype=np.float32)
    test_models = []
    fold_gains = []

    for f in range(N_FOLDS):
        tr = np.flatnonzero(fold_id != f)
        va = np.flatnonzero(fold_id == f)
        pred_va = np.zeros((len(va), n_modules), dtype=np.float32)

        for m in range(n_modules):
            reg = Ridge(alpha=alpha)
            reg.fit(score_oof[tr, m:m+1], Y_rel[tr, m])
            pred_va[:, m] = reg.predict(score_oof[va, m:m+1]).reshape(-1)

        before = ndcg_at_k(Y_rel[va], score_oof[va], 5)
        after = ndcg_at_k(Y_rel[va], pred_va, 5)
        fold_gains.append(after - before)
        cal_oof[va] = pred_va

    pred_test = np.zeros((n_test, n_modules), dtype=np.float32)
    for m in range(n_modules):
        reg = Ridge(alpha=alpha)
        reg.fit(score_oof[:, m:m+1], Y_rel[:, m])
        pred_test[:, m] = reg.predict(score_test[:, m:m+1]).reshape(-1)

    return cal_oof, pred_test, np.asarray(fold_gains)

cal_candidates = []
for a in [5.0, 10.0, 20.0, 40.0, 80.0]:
    co, ct, gains = crossfit_affine_calibration(blend_oof, uncal_test, alpha=a)
    sc = ndcg_at_k(Y_rel, co, 5)
    cal_candidates.append((sc, a, co, ct, gains))
    print(f"cal alpha={a:<5}: OOF={sc:.6f}, fold gains={[round(float(x),5) for x in gains]}")

base_blend_score = ndcg_at_k(Y_rel, blend_oof, 5)
best_cal = max(cal_candidates, key=lambda x: x[0])
best_cal_score, best_cal_alpha, best_cal_oof, best_cal_test, best_cal_gains = best_cal

# Apply only if aggregate lift is real and calibration is not depending on one lucky fold.
stable = (
    best_cal_score >= base_blend_score + 0.001
    and (best_cal_gains >= -0.0025).sum() >= 4
    and np.median(best_cal_gains) > 0
)

if stable:
    final_oof = best_cal_oof.astype(np.float32)
    final_test = best_cal_test.astype(np.float32)
    calibration_used = True
    print(f"Using affine calibration alpha={best_cal_alpha}: {base_blend_score:.6f} -> {best_cal_score:.6f}")
else:
    final_oof = blend_oof.astype(np.float32)
    final_test = uncal_test.astype(np.float32)
    calibration_used = False
    print("Calibration rejected by stability gate.")

print("FINAL OOF NDCG@5:", round(ndcg_at_k(Y_rel, final_oof, 5), 6))
print("FINAL top1 accuracy:", round(float((final_oof.argmax(axis=1) == top_class).mean()), 4))
true_top1_in5 = np.mean([
    top_class[i] in np.argsort(-final_oof[i])[:5] for i in range(n_train)
])
print("True top1 contained in predicted top5:", round(float(true_top1_in5), 4))

# ---- cell 17 (code) ----
# =========================
# Diagnostics + three reproducible submission candidates
# =========================
def module_recall_at5(y_true, scores):
    top5 = np.argsort(-scores, axis=1)[:, :5]
    rows = []
    for m in range(n_modules):
        pos = np.flatnonzero(y_true[:, m] > 0)
        top1_pos = np.flatnonzero(y_true.argmax(axis=1) == m)
        rec = float(np.mean([m in top5[i] for i in pos])) if len(pos) else np.nan
        top1_rec = float(np.mean([m in top5[i] for i in top1_pos])) if len(top1_pos) else np.nan
        rows.append((module_cols[m], modules.loc[m, "module_name"], len(pos), len(top1_pos), rec, top1_rec))
    return pd.DataFrame(rows, columns=[
        "module_id", "module_name", "positive_n", "top1_n", "relevant_recall@5", "top1_in_top5"
    ])

diag = module_recall_at5(Y_rel, final_oof)
display(diag.sort_values("relevant_recall@5"))

def write_submission(scores, filename):
    assert scores.shape == (n_test, n_modules)
    assert np.isfinite(scores).all()
    out = sample.copy()
    out[module_cols] = scores.astype(np.float64)
    assert out.shape == sample.shape
    assert out["user_id"].astype(str).tolist() == test_user_ids
    path = Path("/kaggle/working") / filename
    out.to_csv(path, index=False)
    print("Saved:", path)
    return out

# Primary: stability-gated calibrated model.
sub_primary = write_submission(final_test, "submission_v3_primary.csv")

# Variant 2: same robust model before any module-wise calibration.
sub_uncal = write_submission(uncal_test, "submission_v3_uncalibrated.csv")

# Variant 3: full-data sparse specialist only. Low variance and useful as a leaderboard diagnostic.
sub_sparse = write_submission(zt_s, "submission_v3_sparse_safe.csv")

# Top-5 frequency sanity check for the primary.
test_top5 = np.argsort(-final_test, axis=1)[:, :5]
top5_counts = np.bincount(test_top5.reshape(-1), minlength=n_modules)
display(pd.DataFrame({
    "module_id": module_cols,
    "module_name": modules["module_name"],
    "top5_frequency": top5_counts,
}).sort_values("top5_frequency", ascending=False))

# Persist everything needed to reproduce/inspect this run.
np.save(CACHE_DIR / f"{CACHE_VERSION}_sparse_oof.npy", sparse_oof)
np.save(CACHE_DIR / f"{CACHE_VERSION}_ce_oof.npy", ce_oof)
np.save(CACHE_DIR / f"{CACHE_VERSION}_final_oof.npy", final_oof)
np.save(CACHE_DIR / f"{CACHE_VERSION}_final_test.npy", final_test)

run_summary = {
    "seed": SEED,
    "best_sparse_alpha": float(best_alpha),
    "sparse_head_weights": {n: float(w) for n, w in zip(head_names, mean_head_w)},
    "sparse_oof_ndcg5": float(ndcg_at_k(Y_rel, sparse_oof, 5)),
    "cross_encoder_oof_ndcg5": float(ndcg_at_k(Y_rel, ce_oof, 5)) if RUN_CROSS_ENCODER else None,
    "cross_encoder_fold_scores": [float(x) for x in ce_fold_scores],
    "cross_encoder_best_epochs": [int(x) for x in ce_best_epochs],
    "fold_cross_encoder_weights": [float(x) for x in fold_ce_weights],
    "median_cross_encoder_weight": float(final_ce_weight),
    "precal_blend_oof_ndcg5": float(base_blend_score),
    "calibration_used": bool(calibration_used),
    "calibration_alpha": float(best_cal_alpha) if calibration_used else None,
    "final_oof_ndcg5": float(ndcg_at_k(Y_rel, final_oof, 5)),
    "top1_accuracy": float((final_oof.argmax(axis=1) == top_class).mean()),
    "true_top1_in_top5": float(true_top1_in5),
}
with open(CACHE_DIR / f"{CACHE_VERSION}_summary.json", "w") as f:
    json.dump(run_summary, f, indent=2)

print(json.dumps(run_summary, indent=2))
display(sub_primary.head())

# ---- cell 18 (markdown) ----
# ## Submission choice
#
# Use the notebook's own OOF diagnostics rather than automatically spending all three daily submissions.
#
# - **`submission_v3_primary.csv`**: preferred if calibration passes the stability gate.
# - **`submission_v3_uncalibrated.csv`**: preferred if you want the cleanest cross-encoder + sparse comparison against V2.
# - **`submission_v3_sparse_safe.csv`**: deliberately low-variance diagnostic; useful because V2's leaderboard gap suggested the transformer branch may overfit.
#
# The most important numbers to record are:
#
# 1. selected sparse `alpha`,
# 2. sparse OOF NDCG@5,
# 3. cross-encoder fold NDCG@5 values,
# 4. cross-encoder aggregate OOF,
# 5. fold-selected cross-encoder blend weights,
# 6. final OOF,
# 7. rare-module recall@5 and test top-5 frequencies,
# 8. public leaderboard score.
#
# Do **not** assume that the submission with the highest OOF will always transfer best; the fold stability and rare-module diagnostics matter here.
