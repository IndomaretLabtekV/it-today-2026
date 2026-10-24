# Converted from notebooks/v2.ipynb: markdown cells become comments, %/! magics neutralized.

# ---- cell 0 (markdown) ----
# # MineToday IT Today 2026 — V2 Supervised Multimodal Ranker
#
# **Goal:** rank all 17 Intelligo modules for every test user and maximize **NDCG@5**.
#
# This is a **new V2 architecture**, not a parameter tweak of V1. The key change is that the semantic backbone is no longer zero-shot retrieval. An open-source multilingual encoder is **fine-tuned on the competition labels** and trained jointly to learn:
#
# 1. the single true **top-1 module**,
# 2. the **relevant module set**,
# 3. the **ordinal order** among relevant modules,
# 4. pairwise ranking consistency.
#
# It is blended with four deliberately different, honest OOF components:
#
# - raw **word + character TF-IDF multi-head linear models** (all-history + recent-history views),
# - leakage-safe **repeated-template memory**,
# - nonlinear **assessment-only ExtraTrees**,
# - a direct **LightGBM LambdaRank** model with assessment/prerequisite, lexical, recency, and module-identity features.
#
# The final blend is selected **only from out-of-fold predictions**. No base learner is stacked on its own in-sample predictions.
#
# ### Runtime fix in this revision
# Kaggle ships an old optional `torchao` package that conflicts with current PEFT during LoRA injection. This revision removes that unused optional backend before importing PEFT, pins the PEFT/Transformers versions, and adds an explicit LoRA adapter self-check.
#
# ### Kaggle setup
# - Accelerator: **GPU T4 x2**
# - Internet: **ON**
# - Run all cells from top to bottom.
# - Expected final file: `/kaggle/working/submission_v2_supervised.csv`
#
# ### Competition compliance
# The technical-meeting rules explicitly allow **open-source pretrained models**, while prohibiting directly asking an LLM/Generative AI to output test answers. This notebook uses `Alibaba-NLP/gte-multilingual-base` (Apache-2.0) as a trainable encoder and uses only the supplied competition files as task-specific data.
#
# ### Why GTE multilingual base?
# `Alibaba-NLP/gte-multilingual-base` is an encoder-only multilingual model (~305M parameters, 768-d embeddings, up to 8192 tokens, 75 languages). V2 uses **LoRA** on its attention projections (`qkv_proj`, `o_proj`) so 5-fold fine-tuning is practical on dual T4 without optimizing the full 305M parameters.
#
# References:
# - https://huggingface.co/Alibaba-NLP/gte-multilingual-base
# - https://huggingface.co/Alibaba-NLP/new-impl/blob/main/modeling.py

# ---- cell 1 (code) ----
# Reproducible Kaggle dependencies for the GTE + LoRA branch.
# Kaggle's current image contains torchao 0.10.0, while PEFT 0.19.x refuses
# any installed torchao <0.16 during LoRA dispatch. This notebook does NOT use
# TorchAO quantization, so removing that optional backend is safer than changing
# Kaggle's CUDA/PyTorch stack.
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
import importlib.util
import importlib.metadata as importlib_metadata
import unicodedata
from pathlib import Path
from collections import defaultdict
from contextlib import nullcontext

import numpy as np
import pandas as pd

from scipy.sparse import hstack as sp_hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, normalize

import lightgbm as lgb
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

# Fail fast on environment drift. torchao must be absent here because PEFT's
# optional TorchAO dispatcher rejects Kaggle's bundled 0.10.0 before LoRA can
# reach our ordinary nn.Linear projections.
from packaging.version import Version
_torchao_spec = importlib.util.find_spec("torchao")
_peft_v = Version(importlib_metadata.version("peft"))
_transformers_v = Version(importlib_metadata.version("transformers"))
assert _torchao_spec is None, "torchao is still installed; rerun the dependency cell before continuing"
assert _peft_v == Version("0.19.1"), f"Expected peft 0.19.1, found {_peft_v}; rerun the dependency cell"
assert _transformers_v == Version("4.57.6"), f"Expected transformers 4.57.6, found {_transformers_v}; rerun the dependency cell"
print(f"transformers={_transformers_v} | peft={_peft_v} | torchao=absent (intentional)")
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

# ---- cell 3 (code) ----
# =========================
# Configuration
# =========================
N_FOLDS = 5
N_MODULES_EXPECTED = 17

# Sparse supervised NLP
SPARSE_ALPHA = 10.0
WORD_MAX_FEATURES = 55_000
CHAR_MAX_FEATURES = 70_000
RECENT_WORD_MAX_FEATURES = 35_000
RECENT_CHAR_MAX_FEATURES = 45_000

# Direct ranker lexical compression
RANKER_WORD_MAX_FEATURES = 25_000
RANKER_CHAR_MAX_FEATURES = 35_000
RANKER_SVD_DIM = 64
MEMORY_SHRINK = 8.0

# Supervised multilingual encoder
GTE_MODEL = "Alibaba-NLP/gte-multilingual-base"
GTE_MAX_LENGTH = 384
GTE_EPOCHS = 3
GTE_BATCH_SIZE = 8          # global batch before DataParallel split
GTE_EVAL_BATCH_SIZE = 16
GTE_GRAD_ACCUM = 2
GTE_LR = 2.0e-4
GTE_WEIGHT_DECAY = 0.01
GTE_WARMUP_RATIO = 0.08
GTE_LORA_R = 16
GTE_LORA_ALPHA = 32
GTE_LORA_DROPOUT = 0.05
GTE_MAX_GRAD_NORM = 1.0
RUN_GTE = True

CACHE_VERSION = "v2_20260819_r4_peft_fix"
CACHE_DIR = Path("/kaggle/working/minetoday_v2_cache")
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
# =========================
# Data discovery + loading
# =========================
def find_data_dir():
    roots = [Path("/kaggle/input"), Path("."), Path("/mnt/data")]
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
            "MineToday CSV directory not found. Attach the competition dataset to this Kaggle notebook."
        )
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
assert list(sample["user_id"]) == list(test["user_id"])
assert train["user_id"].is_unique and test["user_id"].is_unique
assert assess["user_id"].is_unique
assert set(train["user_id"]).isdisjoint(set(test["user_id"]))

modules = modules.set_index("module_id").loc[module_cols].reset_index()

n_train = len(train)
n_test = len(test)
n_modules = len(module_cols)
all_user_ids = pd.concat([train["user_id"], test["user_id"]], ignore_index=True)
n_all = len(all_user_ids)
all_user_index = {u: i for i, u in enumerate(all_user_ids)}
test_global_idx = np.arange(n_train, n_all)

print("chat       :", chat.shape)
print("modules    :", modules.shape)
print("assessment :", assess.shape)
print("train      :", train.shape)
print("test       :", test.shape)
display(modules)

# ---- cell 5 (code) ----
# =========================
# Competition metric + structured targets
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


def relevance_to_ordinal(y):
    out = np.zeros_like(y, dtype=np.float32)
    for i, row in enumerate(y):
        pos = np.flatnonzero(row > 0)
        pos = pos[np.argsort(-row[pos])]
        labels = np.arange(6, 0, -1, dtype=np.float32)[:len(pos)]
        out[i, pos] = labels / 6.0
    return out

Y_rel = train[module_cols].to_numpy(np.float32)
Y_bin = (Y_rel > 0).astype(np.float32)
Y_ord = relevance_to_ordinal(Y_rel)
top_class = Y_rel.argmax(axis=1).astype(np.int64)
Y_top = np.eye(n_modules, dtype=np.float32)[top_class]

positive_counts = Y_bin.sum(axis=1).astype(int)
ones_counts = (Y_rel == 1.0).sum(axis=1)
assert set(np.unique(positive_counts)).issubset({4, 5, 6})
assert np.all(ones_counts == 1)
assert np.all(Y_ord.max(axis=1) == 1.0)

print("Positive modules/user:", pd.Series(positive_counts).value_counts().sort_index().to_dict())
print("Top-1 module counts:")
display(pd.DataFrame({
    "module_id": module_cols,
    "module_name": modules["module_name"],
    "top1_count": np.bincount(top_class, minlength=n_modules),
    "positive_count": Y_bin.sum(axis=0).astype(int),
    "mean_relevance": Y_rel.mean(axis=0),
}).sort_values("top1_count", ascending=False))

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
folds = list(cv.split(np.arange(n_train), top_class))
fold_id = np.full(n_train, -1, dtype=np.int8)
for f, (_, va_idx) in enumerate(folds):
    fold_id[va_idx] = f
assert np.all(fold_id >= 0)
print("Fold sizes:", np.bincount(fold_id).tolist())

# ---- cell 6 (code) ----
# =========================
# Assessment cleaning + nonlinear numeric features
# =========================
WORD_TO_NUM = {
    "nol": 0.0, "satu": 1.0, "dua": 2.0,
    "tiga": 3.0, "empat": 4.0, "lima": 5.0,
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
train_ass_mask = assess["user_id"].isin(train_user_set).to_numpy()
train_medians = clean_q.loc[train_ass_mask].median(axis=0)
clean_q = clean_q.fillna(train_medians)

skill_names = [
    "Python", "SQL", "Statistics", "EDA", "Machine Learning",
    "ML Evaluation", "Deep Learning", "Generative AI / LLM",
    "Business-to-DS Translation", "Independent Research / Troubleshooting",
]
assert len(question_names) == len(skill_names) == 10

clean_indexed = pd.DataFrame(
    clean_q.to_numpy(np.float32), index=assess["user_id"], columns=skill_names
).loc[all_user_ids]
invalid_indexed = pd.DataFrame(
    invalid_q.to_numpy(np.float32), index=assess["user_id"], columns=skill_names
).loc[all_user_ids]
assessment_values = clean_indexed.to_numpy(np.float32)
assessment_invalid = invalid_indexed.to_numpy(np.float32)

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

print("Assessment invalid cells:", int(assessment_invalid.sum()), "/", assessment_invalid.size)
print("Train/test skill means:")
display(pd.DataFrame({
    "skill": skill_names,
    "train_mean": assessment_values[:n_train].mean(axis=0),
    "test_mean": assessment_values[n_train:].mean(axis=0),
}))

# ---- cell 7 (code) ----
# =========================
# Chat aggregation: chronological + explicit recency views
# =========================
def normalize_chat_text(s):
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = re.sub(r"\d+", " <num> ", s)
    s = re.sub(r"[^a-z0-9<>]+", " ", s)
    return " ".join(s.split())

chat["timestamp"] = pd.to_datetime(chat["timestamp"], errors="coerce")
chat = chat[chat["user_id"].isin(set(all_user_ids))].copy()
chat["norm_text"] = chat["user_chat_text"].map(normalize_chat_text)
chat = chat.sort_values(["user_id", "timestamp", "chat_id"]).reset_index(drop=True)

message_lists = chat.groupby("user_id")["user_chat_text"].apply(list).to_dict()
norm_message_lists = chat.groupby("user_id")["norm_text"].apply(list).to_dict()
ts_lists = chat.groupby("user_id")["timestamp"].apply(list).to_dict()

chat_rows = []
profile_texts = []
sparse_texts = []
recent_sparse_texts = []
user_messages_unique = []
user_message_ages = []

skill_token_names = [
    "python", "sql", "statistics", "eda", "machine_learning", "ml_evaluation",
    "deep_learning", "generative_ai", "business_translation", "research",
]

for i, user_id in enumerate(all_user_ids):
    msgs = [str(x) for x in message_lists.get(user_id, [])]
    norm_msgs = [x for x in norm_message_lists.get(user_id, []) if x]
    times = ts_lists.get(user_id, [])

    latest = msgs[-1] if msgs else "No chat history available."
    recent = list(reversed(msgs[-3:]))
    older = msgs[:-3]

    assess_text = "; ".join(
        f"{name}={score:.0f}/5" for name, score in zip(skill_names, assessment_values[i])
    )
    structured_tokens = " ".join(
        f"skill_{name}_{int(score)}" for name, score in zip(skill_token_names, assessment_values[i])
    )

    profile_texts.append(
        "[CURRENT INTENT] " + latest + "\n"
        + "[RECENT CONTEXT] " + " [SEP] ".join(recent[1:]) + "\n"
        + "[ASSESSMENT] " + assess_text + "\n"
        + "[OLDER HISTORY] " + " [SEP] ".join(older)
    )
    sparse_texts.append(" [SEP] ".join(msgs) + " " + structured_tokens)
    recent_sparse_texts.append(" [SEP] ".join(msgs[-3:]) + " " + structured_tokens)

    # Keep the newest occurrence of a normalized template and attach age-from-latest.
    newest_age = {}
    for age, m in enumerate(reversed(norm_msgs)):
        newest_age.setdefault(m, age)
    ordered_unique = sorted(newest_age, key=lambda m: newest_age[m])
    user_messages_unique.append(ordered_unique)
    user_message_ages.append([newest_age[m] for m in ordered_unique])

    if times:
        valid_times = [t for t in times if pd.notna(t)]
        span = ((max(valid_times) - min(valid_times)).total_seconds() / 86400.0) if valid_times else 0.0
    else:
        span = 0.0
    chat_rows.append([
        len(msgs), len(set(norm_msgs)), span,
        sum(len(x) for x in msgs),
        sum(len(x.split()) for x in msgs),
        len(latest) if msgs else 0,
        len(latest.split()) if msgs else 0,
        float(len(msgs) == 0),
    ])

chat_stats = np.asarray(chat_rows, dtype=np.float32)

train_template_set = set(chat.loc[chat["user_id"].isin(train_user_set), "norm_text"])
test_chat = chat[chat["user_id"].isin(set(test["user_id"]))].copy()
if len(test_chat):
    seen = test_chat["norm_text"].isin(train_template_set)
    msg_overlap = float(seen.mean())
    user_overlap = float(test_chat.assign(seen=seen).groupby("user_id")["seen"].max().mean())
else:
    msg_overlap = user_overlap = 0.0

print(f"Exact normalized test-message overlap with train: {msg_overlap:.3f}")
print(f"Test users with chat and >=1 train template: {user_overlap:.3f}")
print("Missing chat users train/test:", int(chat_stats[:n_train, -1].sum()), int(chat_stats[n_train:, -1].sum()))
print("Example transformer profile:\n", profile_texts[1][:1200])

# ---- cell 8 (code) ----
# =========================
# Module documents + prerequisite features derived ONLY from supplied catalog
# =========================
module_docs = [
    f"Course: {r.module_name}. Syllabus: {r.description_and_syllabus}. Prerequisite: {r.prerequisite_level}."
    for r in modules.itertuples(index=False)
]

# Skill indices correspond to the 10 supplied assessment questions.
# These mappings are direct interpretations of modules_catalog.csv, not external data.
MODULE_DIRECT_SKILLS = {
    "M_001": [], "M_002": [0], "M_003": [1], "M_004": [], "M_005": [],
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

level_map = {"Pemula": 1.0, "Menengah": 2.5, "Lanjutan": 3.5, "Ahli": 4.25, "Semua Level": 2.5}
module_level = []
for r in modules.itertuples(index=False):
    s = str(r.prerequisite_level)
    level = next((v for k, v in level_map.items() if k.lower() in s.lower()), 2.5)
    module_level.append(level)
module_level = np.asarray(module_level, dtype=np.float32)


def rule_features_for_users(user_idx):
    user_idx = np.asarray(user_idx, dtype=np.int64)
    out = np.zeros((len(user_idx), n_modules, 9), dtype=np.float32)
    for r, ui in enumerate(user_idx):
        a = assessment_values[ui]
        global_mean = float(a.mean())
        for m, module_id in enumerate(module_cols):
            direct = MODULE_DIRECT_SKILLS[module_id]
            prereq = MODULE_PREREQ_SKILLS[module_id]
            direct_vals = a[direct] if direct else np.asarray([global_mean], dtype=np.float32)
            prereq_vals = a[prereq] if prereq else np.asarray([global_mean], dtype=np.float32)
            lvl = module_level[m]
            out[r, m] = [
                direct_vals.mean(), direct_vals.min(),
                prereq_vals.mean(), prereq_vals.min(),
                max(0.0, lvl - direct_vals.mean()),
                max(0.0, lvl - prereq_vals.mean()),
                prereq_vals.mean() - lvl,
                global_mean - lvl,
                lvl,
            ]
    return out

rule_all = rule_features_for_users(np.arange(n_all))
assert rule_all.shape == (n_all, n_modules, 9)

# ---- cell 9 (markdown) ----
# ## Component A — Raw sparse supervised NLP
#
# The V1 system compressed lexical TF-IDF into SVD features before the final ranker. V2 keeps a much stronger **raw sparse path** and fits four complementary supervised targets directly on word+character n-grams:
#
# - graded relevance,
# - ordinal rank,
# - one-hot top-1 module,
# - relevant/not-relevant.
#
# A separate last-3-message model adds recency diversity. All five heads are combined using OOF-only weight search.

# ---- cell 10 (code) ----
# =========================
# Sparse word + char TF-IDF matrices
# Fitting IDF/vocabulary on supplied train+test text is target-free transductive preprocessing.
# =========================
word_vec = TfidfVectorizer(
    lowercase=True, strip_accents="unicode", ngram_range=(1, 2),
    min_df=2, max_df=0.995, max_features=WORD_MAX_FEATURES,
    sublinear_tf=True, dtype=np.float32,
)
char_vec = TfidfVectorizer(
    lowercase=True, analyzer="char_wb", ngram_range=(3, 5),
    min_df=2, max_features=CHAR_MAX_FEATURES,
    sublinear_tf=True, dtype=np.float32,
)
X_word = word_vec.fit_transform(sparse_texts)
X_char = char_vec.fit_transform(sparse_texts)
X_sparse = sp_hstack([X_word, X_char], format="csr")

recent_word_vec = TfidfVectorizer(
    lowercase=True, strip_accents="unicode", ngram_range=(1, 2),
    min_df=2, max_df=0.995, max_features=RECENT_WORD_MAX_FEATURES,
    sublinear_tf=True, dtype=np.float32,
)
recent_char_vec = TfidfVectorizer(
    lowercase=True, analyzer="char_wb", ngram_range=(3, 5),
    min_df=2, max_features=RECENT_CHAR_MAX_FEATURES,
    sublinear_tf=True, dtype=np.float32,
)
X_recent = sp_hstack([
    recent_word_vec.fit_transform(recent_sparse_texts),
    recent_char_vec.fit_transform(recent_sparse_texts),
], format="csr")

print("X_sparse:", X_sparse.shape, "nnz:", X_sparse.nnz)
print("X_recent:", X_recent.shape, "nnz:", X_recent.nnz)

# ---- cell 11 (code) ----
# =========================
# 5-fold sparse multi-head OOF
# =========================
def row_zscore(x):
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    return ((x - mu) / np.maximum(sd, 1e-8)).astype(np.float32)

sparse_head_names = ["raw_rel", "ordinal", "top1", "binary", "recent_ordinal"]
sparse_oof_heads = {n: np.zeros((n_train, n_modules), dtype=np.float32) for n in sparse_head_names}
sparse_test_heads = {n: [] for n in sparse_head_names}

for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
    print(f"Sparse fold {fold}/{N_FOLDS}")
    target_map = {
        "raw_rel": Y_rel,
        "ordinal": Y_ord,
        "top1": Y_top,
        "binary": Y_bin,
    }
    for name, target in target_map.items():
        model = Ridge(alpha=SPARSE_ALPHA, solver="sparse_cg", tol=1e-3)
        model.fit(X_sparse[tr_idx], target[tr_idx])
        sparse_oof_heads[name][va_idx] = model.predict(X_sparse[va_idx]).astype(np.float32)
        sparse_test_heads[name].append(model.predict(X_sparse[test_global_idx]).astype(np.float32))
        del model

    recent_model = Ridge(alpha=SPARSE_ALPHA, solver="sparse_cg", tol=1e-3)
    recent_model.fit(X_recent[tr_idx], Y_ord[tr_idx])
    sparse_oof_heads["recent_ordinal"][va_idx] = recent_model.predict(X_recent[va_idx]).astype(np.float32)
    sparse_test_heads["recent_ordinal"].append(recent_model.predict(X_recent[test_global_idx]).astype(np.float32))
    del recent_model
    gc.collect()

sparse_test_heads = {k: np.mean(v, axis=0).astype(np.float32) for k, v in sparse_test_heads.items()}

print("Sparse head OOF NDCG@5")
for name in sparse_head_names:
    print(f"{name:15s}: {fast_ndcg_at_k(Y_rel, sparse_oof_heads[name], 5):.6f}")

# OOF-only blend among the five sparse heads.
rng = np.random.default_rng(SEED + 101)
Z_oof = [row_zscore(sparse_oof_heads[n]) for n in sparse_head_names]
Z_test = [row_zscore(sparse_test_heads[n]) for n in sparse_head_names]
weight_candidates = [np.eye(len(sparse_head_names))[i] for i in range(len(sparse_head_names))]
weight_candidates += list(rng.dirichlet(np.array([3.0, 3.5, 4.5, 2.5, 1.5]), size=1800))

best_sparse_score, best_sparse_w = -1.0, None
for w in weight_candidates:
    pred = sum(float(wi) * z for wi, z in zip(w, Z_oof))
    score = fast_ndcg_at_k(Y_rel, pred, 5)
    if score > best_sparse_score:
        best_sparse_score, best_sparse_w = score, np.asarray(w, dtype=np.float64)

sparse_oof = sum(float(wi) * z for wi, z in zip(best_sparse_w, Z_oof)).astype(np.float32)
sparse_test = sum(float(wi) * z for wi, z in zip(best_sparse_w, Z_test)).astype(np.float32)
print("Best sparse OOF NDCG@5:", round(best_sparse_score, 6))
print("Sparse weights:", {n: round(float(w), 4) for n, w in zip(sparse_head_names, best_sparse_w)})

# ---- cell 12 (markdown) ----
# ## Component B — Leakage-safe repeated-template memory
#
# Chat messages are heavily templated. This component memorizes target distributions for normalized messages **only from the training side of each fold**. A training user's own label is removed from its memory feature, and newer messages receive more weight.

# ---- cell 13 (code) ----
# =========================
# Leakage-safe recency-weighted template memory
# =========================
def build_template_stats(pool_idx, labels):
    pool_idx = np.asarray(pool_idx, dtype=np.int64)
    global_prior = labels[pool_idx].mean(axis=0).astype(np.float32)
    sums, counts = {}, {}
    for i in pool_idx:
        for msg in user_messages_unique[i]:
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
    out = np.zeros((len(target_idx), n_modules), dtype=np.float32)
    coverage = np.zeros(len(target_idx), dtype=np.float32)
    confidence = np.zeros(len(target_idx), dtype=np.float32)

    for r, i in enumerate(target_idx):
        msgs = user_messages_unique[i]
        ages = user_message_ages[i]
        preds, weights = [], []
        matched = 0
        for msg, age in zip(msgs, ages):
            if msg not in sums:
                continue
            n = counts[msg]
            s = sums[msg]
            if leave_one_out:
                n -= 1
                if n <= 0:
                    continue
                s = s - labels[i]
            mu = s / n
            shrink = n / (n + MEMORY_SHRINK)
            p = shrink * mu + (1.0 - shrink) * global_prior
            information = float(np.linalg.norm(p - global_prior))
            recency = 0.80 ** float(age)
            w = recency * (information + 1e-3) / math.sqrt(max(n, 1))
            preds.append(p)
            weights.append(w)
            matched += 1
        if preds:
            out[r] = np.average(np.stack(preds), axis=0, weights=np.asarray(weights))
            coverage[r] = matched / max(len(msgs), 1)
            confidence[r] = float(np.mean(weights))
        else:
            out[r] = global_prior
    return out, coverage, confidence

oof_memory = np.zeros((n_train, n_modules), dtype=np.float32)
test_memory_folds = []
fold_memory_payload = []

for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
    stats = build_template_stats(tr_idx, Y_ord)
    mem_tr, cov_tr, conf_tr = template_memory_for_indices(tr_idx, Y_ord, stats, leave_one_out=True)
    mem_va, cov_va, conf_va = template_memory_for_indices(va_idx, Y_ord, stats, leave_one_out=False)
    mem_te, cov_te, conf_te = template_memory_for_indices(test_global_idx, Y_ord, stats, leave_one_out=False)
    oof_memory[va_idx] = mem_va
    test_memory_folds.append(mem_te)
    fold_memory_payload.append((mem_tr, cov_tr, conf_tr, mem_va, cov_va, conf_va, mem_te, cov_te, conf_te))

test_memory = np.mean(test_memory_folds, axis=0).astype(np.float32)
print("Template memory OOF NDCG@5:", round(fast_ndcg_at_k(Y_rel, oof_memory, 5), 6))

# ---- cell 14 (markdown) ----
# ## Component C — Nonlinear assessment model
#
# The assessment table contains strong nonlinear readiness/gap patterns. `ExtraTreesRegressor` is intentionally used as a separate component so prerequisite interactions can be learned without text and without imposing linearity.

# ---- cell 15 (code) ----
# =========================
# Assessment-only ExtraTrees OOF
# =========================
numeric_base = np.concatenate([
    assessment_values,
    assessment_invalid,
    assessment_stats,
    chat_stats,
], axis=1).astype(np.float32)

# Add squared scores and pairwise readiness summaries; trees can exploit them cheaply.
numeric_user = np.concatenate([
    numeric_base,
    (assessment_values / 5.0) ** 2,
    np.minimum(assessment_values[:, [0]], assessment_values[:, [2]]),  # Python & Statistics readiness
    np.minimum(assessment_values[:, [2]], assessment_values[:, [3]]),  # Statistics & EDA readiness
    np.minimum(assessment_values[:, [0]], assessment_values[:, [4]]),  # Python & ML readiness
    np.minimum(assessment_values[:, [4]], assessment_values[:, [6]]),  # ML & DL readiness
], axis=1).astype(np.float32)

oof_assessment = np.zeros((n_train, n_modules), dtype=np.float32)
test_assessment_folds = []

for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
    et = ExtraTreesRegressor(
        n_estimators=500,
        min_samples_leaf=6,
        max_features=0.90,
        n_jobs=-1,
        random_state=SEED + 300 + fold,
    )
    et.fit(numeric_user[tr_idx], Y_ord[tr_idx])
    oof_assessment[va_idx] = et.predict(numeric_user[va_idx]).astype(np.float32)
    test_assessment_folds.append(et.predict(numeric_user[test_global_idx]).astype(np.float32))
    del et
    gc.collect()

test_assessment = np.mean(test_assessment_folds, axis=0).astype(np.float32)
print("Assessment ExtraTrees OOF NDCG@5:", round(fast_ndcg_at_k(Y_rel, oof_assessment, 5), 6))

# ---- cell 16 (markdown) ----
# ## Component D — Direct structured LambdaRank
#
# This ranker never consumes predictions from the neural or sparse supervised components, so it remains an independent base learner. It sees:
#
# - assessment values and invalid-value flags,
# - chat activity/recency statistics,
# - transductive target-free lexical SVD user features,
# - user↔module word/char lexical similarity,
# - module identity and fold-only priors,
# - module-specific skill-gap/prerequisite features,
# - leakage-safe template memory.

# ---- cell 17 (code) ----
# =========================
# Target-free lexical features for the direct ranker
# =========================
ranker_corpus = [" [SEP] ".join(message_lists.get(u, [])) for u in all_user_ids] + module_docs

rank_word_vec = TfidfVectorizer(
    lowercase=True, strip_accents="unicode", ngram_range=(1, 2), min_df=2,
    max_df=0.995, max_features=RANKER_WORD_MAX_FEATURES, sublinear_tf=True, dtype=np.float32,
)
rank_word_all = rank_word_vec.fit_transform(ranker_corpus)
rank_word_user = rank_word_all[:n_all]
rank_word_module = rank_word_all[n_all:]
word_cos = (rank_word_user @ rank_word_module.T).toarray().astype(np.float32)

word_dim = min(RANKER_SVD_DIM, max(2, rank_word_all.shape[1] - 1))
word_svd = TruncatedSVD(n_components=word_dim, n_iter=7, random_state=SEED)
word_user_svd = normalize(word_svd.fit_transform(rank_word_all)[:n_all]).astype(np.float32)

rank_char_vec = TfidfVectorizer(
    lowercase=True, analyzer="char_wb", ngram_range=(3, 5), min_df=2,
    max_features=RANKER_CHAR_MAX_FEATURES, sublinear_tf=True, dtype=np.float32,
)
rank_char_all = rank_char_vec.fit_transform(ranker_corpus)
rank_char_user = rank_char_all[:n_all]
rank_char_module = rank_char_all[n_all:]
char_cos = (rank_char_user @ rank_char_module.T).toarray().astype(np.float32)

char_dim = min(RANKER_SVD_DIM, max(2, rank_char_all.shape[1] - 1))
char_svd = TruncatedSVD(n_components=char_dim, n_iter=7, random_state=SEED + 1)
char_user_svd = normalize(char_svd.fit_transform(rank_char_all)[:n_all]).astype(np.float32)

user_static = np.concatenate([numeric_user, word_user_svd, char_user_svd], axis=1).astype(np.float32)
MODULE_EYE = np.eye(n_modules, dtype=np.float32)

print("Direct ranker user_static:", user_static.shape)
print("word/char cosine:", word_cos.shape, char_cos.shape)

# ---- cell 18 (code) ----
# =========================
# Pair feature builder + 5-fold LambdaRank
# =========================
def build_pair_features(user_idx, mem_score, mem_coverage, mem_confidence, module_priors):
    user_idx = np.asarray(user_idx, dtype=np.int64)
    n = len(user_idx)
    repeated_user = np.repeat(user_static[user_idx], n_modules, axis=0)
    module_onehot = np.tile(MODULE_EYE, (n, 1))
    rules = rule_all[user_idx].reshape(n * n_modules, -1)

    pair_specific = np.column_stack([
        word_cos[user_idx].reshape(-1),
        char_cos[user_idx].reshape(-1),
        mem_score.reshape(-1),
        np.repeat(mem_coverage, n_modules),
        np.repeat(mem_confidence, n_modules),
    ]).astype(np.float32)
    prior_block = np.tile(module_priors.astype(np.float32), (n, 1))

    return np.concatenate([
        repeated_user, module_onehot, rules, pair_specific, prior_block,
    ], axis=1).astype(np.float32)

oof_ranker = np.zeros((n_train, n_modules), dtype=np.float32)
test_ranker_folds = []
ranker_fold_scores = []

for fold0, (tr_idx, va_idx) in enumerate(folds):
    fold = fold0 + 1
    print(f"\nDirect LambdaRank fold {fold}/{N_FOLDS}")
    mem_tr, cov_tr, conf_tr, mem_va, cov_va, conf_va, mem_te, cov_te, conf_te = fold_memory_payload[fold0]

    module_priors = np.column_stack([
        Y_ord[tr_idx].mean(axis=0),
        Y_bin[tr_idx].mean(axis=0),
        np.bincount(top_class[tr_idx], minlength=n_modules) / len(tr_idx),
    ]).astype(np.float32)

    X_tr = build_pair_features(tr_idx, mem_tr, cov_tr, conf_tr, module_priors)
    X_va = build_pair_features(va_idx, mem_va, cov_va, conf_va, module_priors)
    X_te = build_pair_features(test_global_idx, mem_te, cov_te, conf_te, module_priors)

    y_tr = (Y_ord[tr_idx] * 6).round().astype(np.int32).reshape(-1)
    y_va = (Y_ord[va_idx] * 6).round().astype(np.int32).reshape(-1)
    group_tr = [n_modules] * len(tr_idx)
    group_va = [n_modules] * len(va_idx)

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=3500,
        learning_rate=0.018,
        num_leaves=39,
        max_depth=-1,
        min_child_samples=28,
        subsample=0.92,
        colsample_bytree=0.82,
        reg_alpha=0.15,
        reg_lambda=3.5,
        max_bin=255,
        lambdarank_truncation_level=8,
        label_gain=[0, 20, 36, 52, 68, 84, 100],
        random_state=SEED + 500 + fold,
        n_jobs=-1,
        verbosity=-1,
        force_col_wise=True,
    )
    ranker.fit(
        X_tr, y_tr,
        group=group_tr,
        eval_set=[(X_va, y_va)],
        eval_group=[group_va],
        eval_at=[5],
        callbacks=[lgb.early_stopping(180, verbose=False), lgb.log_evaluation(0)],
    )

    pred_va = ranker.predict(X_va, num_iteration=ranker.best_iteration_).reshape(len(va_idx), n_modules)
    pred_te = ranker.predict(X_te, num_iteration=ranker.best_iteration_).reshape(n_test, n_modules)
    oof_ranker[va_idx] = pred_va.astype(np.float32)
    test_ranker_folds.append(pred_te.astype(np.float32))
    score = fast_ndcg_at_k(Y_rel[va_idx], pred_va, 5)
    ranker_fold_scores.append(score)
    print("best_iteration:", ranker.best_iteration_, "NDCG@5:", round(score, 6))

    del X_tr, X_va, X_te, ranker
    gc.collect()

test_ranker = np.mean(test_ranker_folds, axis=0).astype(np.float32)
print("Direct ranker fold scores:", [round(x, 6) for x in ranker_fold_scores])
print("Direct ranker OOF NDCG@5:", round(fast_ndcg_at_k(Y_rel, oof_ranker, 5), 6))

# ---- cell 19 (markdown) ----
# ## Component E — LoRA fine-tuned GTE multilingual multi-task scorer
#
# This is the main V2 modeling jump. Instead of embedding a learner and comparing cosine similarity to a module, the encoder is fine-tuned to output **17 task-specific scores**. Three heads learn different structures:
#
# - `top1_logits`: weighted 17-class cross entropy,
# - `relevant_logits`: weighted multi-label BCE,
# - `rank_logits`: pairwise ranking + ordinal regression.
#
# Only LoRA adapters on the official GTE attention projections plus the small task heads are optimized. Fold validation predictions are always generated from a model that never trained on those users.

# ---- cell 20 (code) ----
# =========================
# Numeric features for neural branch
# =========================
neural_scaler = StandardScaler()
neural_scaler.fit(numeric_user[:n_train])  # target-free scaling
neural_numeric = neural_scaler.transform(numeric_user).astype(np.float32)

GTE_CACHE = CACHE_DIR / f"{CACHE_VERSION}_gte_multitask.npz"
print("Neural numeric shape:", neural_numeric.shape)

# ---- cell 21 (code) ----
# =========================
# GTE dataset, model and multi-task losses
# =========================
def pairwise_rank_loss(scores, targets):
    # scores/targets: [batch, 17]. Only strict target orderings contribute.
    sdiff = scores.unsqueeze(2) - scores.unsqueeze(1)
    ydiff = targets.unsqueeze(2) - targets.unsqueeze(1)
    mask = ydiff > 1e-6
    if not mask.any():
        return scores.sum() * 0.0
    weights = 0.5 + ydiff.clamp(min=0.0)
    losses = torch.nn.functional.softplus(-sdiff) * weights
    return losses[mask].mean()


class UserDataset(Dataset):
    def __init__(self, indices):
        self.indices = np.asarray(indices, dtype=np.int64)
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, i):
        return int(self.indices[i])


class BatchCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def __call__(self, batch_indices):
        idx = np.asarray(batch_indices, dtype=np.int64)
        toks = self.tokenizer(
            [profile_texts[i] for i in idx],
            padding=True,
            truncation=True,
            max_length=GTE_MAX_LENGTH,
            return_tensors="pt",
        )
        out = {
            "input_ids": toks["input_ids"],
            "attention_mask": toks["attention_mask"],
            "numeric": torch.from_numpy(neural_numeric[idx]),
            "indices": torch.from_numpy(idx),
        }
        if np.all(idx < n_train):
            out["top1"] = torch.from_numpy(top_class[idx])
            out["binary"] = torch.from_numpy(Y_bin[idx])
            out["ordinal"] = torch.from_numpy(Y_ord[idx])
        return out


class GTECourseAdvisor(nn.Module):
    def __init__(self, encoder, hidden_size, numeric_dim, n_modules):
        super().__init__()
        self.encoder = encoder
        self.numeric_proj = nn.Sequential(
            nn.LayerNorm(numeric_dim),
            nn.Linear(numeric_dim, 64),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        fusion_dim = hidden_size + 64
        self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.dropout = nn.Dropout(0.15)
        self.top1_head = nn.Linear(fusion_dim, n_modules)
        self.relevant_head = nn.Linear(fusion_dim, n_modules)
        self.rank_head = nn.Linear(fusion_dim, n_modules)

    def forward(self, input_ids, attention_mask, numeric):
        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = enc.last_hidden_state[:, 0].float()
        num = self.numeric_proj(numeric.float())
        z = self.dropout(self.fusion_norm(torch.cat([cls, num], dim=1)))
        return self.top1_head(z), self.relevant_head(z), self.rank_head(z)


def neural_score_from_heads(top1_logits, relevant_logits, rank_logits):
    return (
        0.45 * row_zscore(top1_logits)
        + 0.20 * row_zscore(relevant_logits)
        + 0.35 * row_zscore(rank_logits)
    ).astype(np.float32)

# ---- cell 22 (code) ----
# =========================
# GTE LoRA 5-fold training + honest OOF/test predictions
# =========================
def valid_gte_cache(path):
    if not path.exists():
        return False
    try:
        z = np.load(path)
        required = ["oof_top1", "oof_rel", "oof_rank", "test_top1", "test_rel", "test_rank"]
        return all(k in z for k in required) and all(
            z[k].shape == ((n_train, n_modules) if k.startswith("oof") else (n_test, n_modules))
            for k in required
        )
    except Exception:
        return False

if RUN_GTE and valid_gte_cache(GTE_CACHE):
    z = np.load(GTE_CACHE)
    gte_oof_top1 = z["oof_top1"].astype(np.float32)
    gte_oof_rel = z["oof_rel"].astype(np.float32)
    gte_oof_rank = z["oof_rank"].astype(np.float32)
    gte_test_top1 = z["test_top1"].astype(np.float32)
    gte_test_rel = z["test_rel"].astype(np.float32)
    gte_test_rank = z["test_rank"].astype(np.float32)
    print("Loaded valid GTE OOF/test cache:", GTE_CACHE)
elif RUN_GTE:
    if not torch.cuda.is_available():
        raise RuntimeError("RUN_GTE=True requires a GPU. On Kaggle select GPU T4 x2.")

    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
    from peft import LoraConfig, get_peft_model

    tokenizer = AutoTokenizer.from_pretrained(GTE_MODEL)

    gte_oof_top1 = np.zeros((n_train, n_modules), dtype=np.float32)
    gte_oof_rel = np.zeros_like(gte_oof_top1)
    gte_oof_rank = np.zeros_like(gte_oof_top1)
    gte_test_top1_folds, gte_test_rel_folds, gte_test_rank_folds = [], [], []

    for fold0, (tr_idx, va_idx) in enumerate(folds):
        fold = fold0 + 1
        print(f"\n========== GTE fold {fold}/{N_FOLDS} ==========")
        seed_everything(SEED + 700 + fold)

        base = AutoModel.from_pretrained(GTE_MODEL, trust_remote_code=True, dtype=torch.float16)
        # Validate official attention projection names before attaching LoRA.
        module_names = [name for name, _ in base.named_modules()]
        assert any(name.endswith("qkv_proj") for name in module_names), "GTE qkv_proj not found"
        assert any(name.endswith("o_proj") for name in module_names), "GTE o_proj not found"

        lora_cfg = LoraConfig(
            r=GTE_LORA_R,
            lora_alpha=GTE_LORA_ALPHA,
            target_modules=["qkv_proj", "o_proj"],
            lora_dropout=GTE_LORA_DROPOUT,
            bias="none",
        )
        encoder = get_peft_model(base, lora_cfg)
        # Runtime regression guard: ensure PEFT really attached trainable adapters.
        lora_param_names = [n for n, p in encoder.named_parameters() if "lora_" in n and p.requires_grad]
        assert lora_param_names, "PEFT returned a model but no trainable LoRA parameters were attached"
        if fold == 1:
            print("LoRA injection OK; trainable LoRA tensors:", len(lora_param_names))
        hidden_size = int(base.config.hidden_size)
        model = GTECourseAdvisor(encoder, hidden_size, neural_numeric.shape[1], n_modules)
        model = model.cuda()

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            print("Using DataParallel on", torch.cuda.device_count(), "GPUs")

        # Fold-local class balancing: validation labels never influence training weights.
        top_counts = np.bincount(top_class[tr_idx], minlength=n_modules).astype(np.float32)
        class_weights_np = np.sqrt(top_counts.mean() / np.maximum(top_counts, 1.0))
        class_weights_np = np.clip(class_weights_np, 0.65, 2.75)
        pos_counts = Y_bin[tr_idx].sum(axis=0)
        pos_weight_np = np.sqrt((len(tr_idx) - pos_counts) / np.maximum(pos_counts, 1.0))
        pos_weight_np = np.clip(pos_weight_np, 1.0, 3.0)

        collator = BatchCollator(tokenizer)
        train_loader = DataLoader(
            UserDataset(tr_idx), batch_size=GTE_BATCH_SIZE, shuffle=True,
            num_workers=0, pin_memory=True, collate_fn=collator,
        )
        val_loader = DataLoader(
            UserDataset(va_idx), batch_size=GTE_EVAL_BATCH_SIZE, shuffle=False,
            num_workers=0, pin_memory=True, collate_fn=collator,
        )
        test_loader = DataLoader(
            UserDataset(test_global_idx), batch_size=GTE_EVAL_BATCH_SIZE, shuffle=False,
            num_workers=0, pin_memory=True, collate_fn=collator,
        )

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=GTE_LR, weight_decay=GTE_WEIGHT_DECAY,
        )
        updates_per_epoch = math.ceil(len(train_loader) / GTE_GRAD_ACCUM)
        total_steps = updates_per_epoch * GTE_EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(1, int(total_steps * GTE_WARMUP_RATIO)),
            num_training_steps=total_steps,
        )

        try:
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        except TypeError:
            scaler = torch.cuda.amp.GradScaler(enabled=True)

        class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device="cuda")
        pos_weight = torch.tensor(pos_weight_np, dtype=torch.float32, device="cuda")
        ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        def predict_loader(loader):
            model.eval()
            all_top, all_rel, all_rank, all_idx = [], [], [], []
            with torch.no_grad():
                for batch in loader:
                    input_ids = batch["input_ids"].cuda(non_blocking=True)
                    attention_mask = batch["attention_mask"].cuda(non_blocking=True)
                    numeric = batch["numeric"].cuda(non_blocking=True)
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        top_logits, rel_logits, rank_logits = model(input_ids, attention_mask, numeric)
                    all_top.append(top_logits.float().cpu().numpy())
                    all_rel.append(rel_logits.float().cpu().numpy())
                    all_rank.append(rank_logits.float().cpu().numpy())
                    all_idx.append(batch["indices"].numpy())
            idx = np.concatenate(all_idx)
            return idx, np.concatenate(all_top), np.concatenate(all_rel), np.concatenate(all_rank)

        best_score = -1.0
        best_val_payload = None
        best_test_payload = None
        optimizer.zero_grad(set_to_none=True)

        for epoch in range(1, GTE_EPOCHS + 1):
            model.train()
            running = 0.0
            for step, batch in enumerate(train_loader, start=1):
                input_ids = batch["input_ids"].cuda(non_blocking=True)
                attention_mask = batch["attention_mask"].cuda(non_blocking=True)
                numeric = batch["numeric"].cuda(non_blocking=True)
                y_top = batch["top1"].cuda(non_blocking=True)
                y_bin_b = batch["binary"].cuda(non_blocking=True)
                y_ord_b = batch["ordinal"].cuda(non_blocking=True)

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    top_logits, rel_logits, rank_logits = model(input_ids, attention_mask, numeric)
                    loss_top = ce_loss_fn(top_logits.float(), y_top)
                    loss_rel = bce_loss_fn(rel_logits.float(), y_bin_b.float())
                    loss_pair = pairwise_rank_loss(rank_logits.float(), y_ord_b.float())
                    loss_ord = torch.nn.functional.smooth_l1_loss(
                        torch.sigmoid(rank_logits.float()), y_ord_b.float()
                    )
                    loss = 0.45*loss_top + 0.25*loss_rel + 0.25*loss_pair + 0.05*loss_ord
                    loss = loss / GTE_GRAD_ACCUM

                scaler.scale(loss).backward()
                running += float(loss.item()) * GTE_GRAD_ACCUM

                if step % GTE_GRAD_ACCUM == 0 or step == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GTE_MAX_GRAD_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

            idx_v, pv_top, pv_rel, pv_rank = predict_loader(val_loader)
            order_v = np.argsort(idx_v)
            pv_top, pv_rel, pv_rank = pv_top[order_v], pv_rel[order_v], pv_rank[order_v]
            val_score_matrix = neural_score_from_heads(pv_top, pv_rel, pv_rank)
            val_ndcg = fast_ndcg_at_k(Y_rel[va_idx], val_score_matrix, 5)
            print(f"epoch {epoch}: loss={running/len(train_loader):.5f} val_NDCG@5={val_ndcg:.6f}")

            if val_ndcg > best_score:
                best_score = val_ndcg
                best_val_payload = (pv_top.astype(np.float32), pv_rel.astype(np.float32), pv_rank.astype(np.float32))
                idx_t, pt_top, pt_rel, pt_rank = predict_loader(test_loader)
                order_t = np.argsort(idx_t)
                best_test_payload = (
                    pt_top[order_t].astype(np.float32),
                    pt_rel[order_t].astype(np.float32),
                    pt_rank[order_t].astype(np.float32),
                )

        assert best_val_payload is not None and best_test_payload is not None
        gte_oof_top1[va_idx], gte_oof_rel[va_idx], gte_oof_rank[va_idx] = best_val_payload
        gte_test_top1_folds.append(best_test_payload[0])
        gte_test_rel_folds.append(best_test_payload[1])
        gte_test_rank_folds.append(best_test_payload[2])
        print("best fold NDCG@5:", round(best_score, 6))

        del model, encoder, base, optimizer, scheduler, train_loader, val_loader, test_loader
        gc.collect()
        torch.cuda.empty_cache()

    gte_test_top1 = np.mean(gte_test_top1_folds, axis=0).astype(np.float32)
    gte_test_rel = np.mean(gte_test_rel_folds, axis=0).astype(np.float32)
    gte_test_rank = np.mean(gte_test_rank_folds, axis=0).astype(np.float32)

    np.savez_compressed(
        GTE_CACHE,
        oof_top1=gte_oof_top1, oof_rel=gte_oof_rel, oof_rank=gte_oof_rank,
        test_top1=gte_test_top1, test_rel=gte_test_rel, test_rank=gte_test_rank,
    )
    print("Saved GTE cache:", GTE_CACHE)
else:
    # Debug-only fallback. Keep RUN_GTE=True for the real V2 submission.
    gte_oof_top1 = sparse_oof_heads["top1"].copy()
    gte_oof_rel = sparse_oof_heads["binary"].copy()
    gte_oof_rank = sparse_oof_heads["ordinal"].copy()
    gte_test_top1 = sparse_test_heads["top1"].copy()
    gte_test_rel = sparse_test_heads["binary"].copy()
    gte_test_rank = sparse_test_heads["ordinal"].copy()
    print("WARNING: RUN_GTE=False; using sparse fallback. Do not submit this as the intended V2 neural model.")

# ---- cell 23 (code) ----
# =========================
# Optimize GTE head blend on honest OOF predictions
# =========================
gte_head_names = ["top1", "relevant", "rank"]
gte_oof_heads = [gte_oof_top1, gte_oof_rel, gte_oof_rank]
gte_test_heads = [gte_test_top1, gte_test_rel, gte_test_rank]

print("GTE head OOF NDCG@5")
for n, p in zip(gte_head_names, gte_oof_heads):
    print(f"{n:10s}: {fast_ndcg_at_k(Y_rel, p, 5):.6f}")

ZG_oof = [row_zscore(x) for x in gte_oof_heads]
ZG_test = [row_zscore(x) for x in gte_test_heads]
rng = np.random.default_rng(SEED + 811)
gte_candidates = [np.eye(3)[i] for i in range(3)]
gte_candidates += [np.array([0.45, 0.20, 0.35]), np.array([0.55, 0.15, 0.30])]
gte_candidates += list(rng.dirichlet([4.5, 2.5, 3.5], size=1200))

best_gte_score, best_gte_w = -1.0, None
for w in gte_candidates:
    p = sum(float(wi) * z for wi, z in zip(w, ZG_oof))
    sc = fast_ndcg_at_k(Y_rel, p, 5)
    if sc > best_gte_score:
        best_gte_score, best_gte_w = sc, np.asarray(w, dtype=np.float64)

gte_oof = sum(float(wi) * z for wi, z in zip(best_gte_w, ZG_oof)).astype(np.float32)
gte_test = sum(float(wi) * z for wi, z in zip(best_gte_w, ZG_test)).astype(np.float32)
print("Best GTE OOF NDCG@5:", round(best_gte_score, 6))
print("GTE head weights:", {n: round(float(w), 4) for n, w in zip(gte_head_names, best_gte_w)})

# ---- cell 24 (markdown) ----
# ## Final honest OOF ensemble
#
# All components below are genuine OOF predictions. The blend search is deliberately constrained and low-dimensional. A small optional **module-prior calibration** is applied only when it produces a non-trivial OOF gain, targeting the rare-module collapse seen in V1.

# ---- cell 25 (code) ----
# =========================
# Final component diagnostics + OOF-only blend search
# =========================
components_oof = {
    "gte_supervised": gte_oof,
    "sparse_multihead": sparse_oof,
    "direct_ranker": oof_ranker,
    "template_memory": oof_memory,
    "assessment_trees": oof_assessment,
}
components_test = {
    "gte_supervised": gte_test,
    "sparse_multihead": sparse_test,
    "direct_ranker": test_ranker,
    "template_memory": test_memory,
    "assessment_trees": test_assessment,
}

print("Individual honest OOF NDCG@5")
for n, p in components_oof.items():
    assert p.shape == (n_train, n_modules) and np.isfinite(p).all()
    assert components_test[n].shape == (n_test, n_modules) and np.isfinite(components_test[n]).all()
    print(f"{n:18s}: {fast_ndcg_at_k(Y_rel, p, 5):.6f}")

names = list(components_oof)
Zoof = [row_zscore(components_oof[n]) for n in names]
Ztest = [row_zscore(components_test[n]) for n in names]

rng = np.random.default_rng(SEED + 999)
weight_candidates = [np.eye(len(names))[i] for i in range(len(names))]
weight_candidates += [
    np.array([0.45, 0.35, 0.15, 0.03, 0.02]),
    np.array([0.55, 0.30, 0.10, 0.03, 0.02]),
    np.array([0.35, 0.45, 0.15, 0.03, 0.02]),
]
# Prior favors the two strongest supervised text paths, but anchors guarantee no component is forced in.
weight_candidates += list(rng.dirichlet([6.0, 6.0, 3.0, 1.0, 1.0], size=3500))

best_score, best_w = -1.0, None
for w in weight_candidates:
    w = np.asarray(w, dtype=np.float64)
    w = w / w.sum()
    blend = sum(float(wi) * z for wi, z in zip(w, Zoof))
    sc = fast_ndcg_at_k(Y_rel, blend, 5)
    if sc > best_score:
        best_score, best_w = sc, w.copy()

final_oof_scores = sum(float(wi) * z for wi, z in zip(best_w, Zoof)).astype(np.float32)
final_test_scores = sum(float(wi) * z for wi, z in zip(best_w, Ztest)).astype(np.float32)

print("\nBest pre-calibration OOF NDCG@5:", round(best_score, 6))
print("Final component weights:")
for n, w in zip(names, best_w):
    print(f"  {n:18s}: {w:.4f}")

# ---- cell 26 (code) ----
# =========================
# Conservative rare-module prior calibration
# Only keep it when OOF gain is meaningful.
# =========================
true_prior = (np.bincount(top_class, minlength=n_modules) + 8.0)
true_prior = true_prior / true_prior.sum()
pred_top = final_oof_scores.argmax(axis=1)
pred_prior = (np.bincount(pred_top, minlength=n_modules) + 8.0)
pred_prior = pred_prior / pred_prior.sum()
module_bias = np.log(true_prior / pred_prior).astype(np.float32)

base_score = fast_ndcg_at_k(Y_rel, final_oof_scores, 5)
best_cal_score, best_gamma = base_score, 0.0
for gamma in np.linspace(0.0, 0.50, 26):
    p = final_oof_scores + float(gamma) * module_bias[None, :]
    sc = fast_ndcg_at_k(Y_rel, p, 5)
    if sc > best_cal_score:
        best_cal_score, best_gamma = sc, float(gamma)

# Require >=0.001 OOF lift before changing test scores.
if best_cal_score >= base_score + 0.001:
    final_oof_scores = final_oof_scores + best_gamma * module_bias[None, :]
    final_test_scores = final_test_scores + best_gamma * module_bias[None, :]
    print(f"Applied prior calibration gamma={best_gamma:.3f}; OOF {base_score:.6f} -> {best_cal_score:.6f}")
else:
    best_gamma = 0.0
    print(f"Prior calibration rejected; best gain < 0.001 (base={base_score:.6f}, best={best_cal_score:.6f})")

assert np.isfinite(final_oof_scores).all()
assert np.isfinite(final_test_scores).all()
print("FINAL honest OOF NDCG@5:", round(fast_ndcg_at_k(Y_rel, final_oof_scores, 5), 6))

# ---- cell 27 (code) ----
# =========================
# Diagnostics targeted at V1 failure modes
# =========================
top1_acc = float((final_oof_scores.argmax(axis=1) == top_class).mean())
top5 = np.argsort(-final_oof_scores, axis=1)[:, :5]
top1_in_top5 = float(np.mean([top_class[i] in top5[i] for i in range(n_train)]))

per_module = []
for m, module_id in enumerate(module_cols):
    relevant_users = np.flatnonzero(Y_rel[:, m] > 0)
    top_users = np.flatnonzero(top_class == m)
    rel_recall = float(np.mean([m in top5[i] for i in relevant_users])) if len(relevant_users) else np.nan
    top_recall = float(np.mean([m in top5[i] for i in top_users])) if len(top_users) else np.nan
    per_module.append([
        module_id, modules.loc[m, "module_name"], len(top_users), len(relevant_users), rel_recall, top_recall,
    ])

diag = pd.DataFrame(per_module, columns=[
    "module_id", "module_name", "top1_count", "positive_count", "relevant_recall@5", "top1_recall@5"
])
print("Top-1 accuracy:", round(top1_acc, 4))
print("True top-1 contained in predicted top-5:", round(top1_in_top5, 4))
display(diag.sort_values("relevant_recall@5"))

# ---- cell 28 (code) ----
# =========================
# Submission + reproducibility artifacts
# =========================
submission = sample.copy()
submission[module_cols] = final_test_scores

out_path = Path("/kaggle/working/submission_v2_supervised.csv")
submission.to_csv(out_path, index=False)

np.save(CACHE_DIR / f"{CACHE_VERSION}_final_oof.npy", final_oof_scores.astype(np.float32))
np.save(CACHE_DIR / f"{CACHE_VERSION}_final_test.npy", final_test_scores.astype(np.float32))
with open(CACHE_DIR / f"{CACHE_VERSION}_blend.json", "w") as f:
    json.dump({
        "seed": SEED,
        "oof_ndcg5": float(fast_ndcg_at_k(Y_rel, final_oof_scores, 5)),
        "component_weights": {n: float(w) for n, w in zip(names, best_w)},
        "sparse_head_weights": {n: float(w) for n, w in zip(sparse_head_names, best_sparse_w)},
        "gte_head_weights": {n: float(w) for n, w in zip(gte_head_names, best_gte_w)},
        "prior_calibration_gamma": float(best_gamma),
    }, f, indent=2)

assert submission.shape == sample.shape
assert submission["user_id"].equals(sample["user_id"])
assert np.isfinite(final_test_scores).all()
assert np.isfinite(submission[module_cols].to_numpy()).all()

print("Saved:", out_path)
print("shape:", submission.shape)
display(submission.head())

# Test top-5 distribution. V1 collapsed several rare modules; inspect this before submitting V2.
test_top5 = np.argsort(-final_test_scores, axis=1)[:, :5]
top5_counts = np.bincount(test_top5.reshape(-1), minlength=n_modules)
display(pd.DataFrame({
    "module_id": module_cols,
    "module_name": modules["module_name"],
    "top5_frequency": top5_counts,
}).sort_values("top5_frequency", ascending=False))

# ---- cell 29 (markdown) ----
# ## What to record after the run
#
# Before using a leaderboard submission, save these outputs:
#
# 1. each individual OOF NDCG@5,
# 2. GTE head weights and GTE OOF score,
# 3. final blend weights,
# 4. final honest OOF NDCG@5,
# 5. per-module recall@5 table,
# 6. test top-5 frequency table,
# 7. Kaggle public LB score.
#
# The most important comparison against V1 is not only the aggregate OOF score. V2 should also materially improve **top-1 accuracy** and recover rare modules such as Web Scrapping, NLP, Prompt Engineering, and Version Control instead of letting Python/SQL dominate almost every top-5.
