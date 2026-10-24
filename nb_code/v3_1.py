# Converted from notebooks/v3_1.ipynb: markdown cells become comments, %/! magics neutralized.

# ---- cell 0 (markdown) ----
# # MineToday IT Today 2026 — V3.1 Structured Sparse Stack
#
# **Goal:** a fast, reproducible ranking pipeline built around the strongest signal discovered in the dataset: synthetic/repeated chat templates, explicit current intent, assessment readiness, and a lightweight nonlinear calibration layer.
#
# This notebook intentionally **does not fine-tune a transformer**. The previous cross-encoder experiment was both slower and weaker than sparse supervised models. V3.1 instead uses:
#
# 1. template/topic-aware word + character TF-IDF,
# 2. five supervised sparse heads,
# 3. explicit user structure features (assessment, topic recency, style, noise),
# 4. per-module LightGBM relevance calibrators,
# 5. a conservative final blend optimized for NDCG@5.
#
# Outputs:
# - `submission_v31_primary.csv` — recommended candidate,
# - `submission_v31_conservative.csv` — lower calibration weight,
# - `submission_v31_sparse_safe.csv` — sparse-only anchor.
#
# Expected runtime is in minutes, not hours. GPU is not required, so T4x2 can stay enabled without affecting correctness.

# ---- cell 1 (code) ----
import os, re, json, ast, html, hashlib, random, warnings, gc, time, sys, subprocess
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    import lightgbm as lgb
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "lightgbm==4.6.0"])
    import lightgbm as lgb

SEED = 42
N_FOLDS = 5
RIDGE_ALPHA = 3.0
CAL_BLEND_PRIMARY = 0.50
CAL_BLEND_CONSERVATIVE = 0.35

random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

print("numpy:", np.__version__)
print("pandas:", pd.__version__)
print("lightgbm:", lgb.__version__)

# ---- cell 2 (code) ----
def locate_data_dir():
    required = {
        "chat_history.csv", "modules_catalog.csv", "user_assessments.csv",
        "train_relevance.csv", "test.csv", "sample_submission.csv"
    }
    roots = [Path("/kaggle/input"), Path("/mnt/data/comp_v3"), Path(".")]
    for root in roots:
        if not root.exists():
            continue
        candidates = [root]
        if root.name == "input":
            candidates += [p for p in root.rglob("*") if p.is_dir()]
        for d in candidates:
            try:
                names = {p.name for p in d.iterdir() if p.is_file()}
            except Exception:
                continue
            if required.issubset(names):
                return d
    raise FileNotFoundError("Could not locate MineToday dataset directory.")

DATA_DIR = locate_data_dir()
print("DATA_DIR =", DATA_DIR)

chat = pd.read_csv(DATA_DIR / "chat_history.csv")
modules = pd.read_csv(DATA_DIR / "modules_catalog.csv")
assessments = pd.read_csv(DATA_DIR / "user_assessments.csv")
train = pd.read_csv(DATA_DIR / "train_relevance.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
sample = pd.read_csv(DATA_DIR / "sample_submission.csv")

MODULES = list(train.columns[1:])
assert MODULES == list(sample.columns[1:])
assert len(MODULES) == 17
assert len(train) == 4000 and len(test) == 1000

all_users = train["user_id"].tolist() + test["user_id"].tolist()
uid2idx = {u:i for i,u in enumerate(all_users)}
y = train[MODULES].to_numpy(np.float32)

print("chat      :", chat.shape)
print("modules   :", modules.shape)
print("assessment:", assessments.shape)
print("train     :", train.shape)
print("test      :", test.shape)
print("positive modules/user:", pd.Series((y > 0).sum(1)).value_counts().sort_index().to_dict())

# ---- cell 3 (code) ----
# ---------- Competition metric + structured targets ----------
DISCOUNT = 1.0 / np.log2(np.arange(2, 7))


def ndcg5(y_true, scores):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    idx = np.argpartition(-scores, 4, axis=1)[:, :5]
    local_scores = np.take_along_axis(scores, idx, axis=1)
    idx = np.take_along_axis(idx, np.argsort(-local_scores, axis=1), axis=1)
    dcg = (np.take_along_axis(y_true, idx, axis=1) * DISCOUNT).sum(1)
    ideal = np.sort(y_true, axis=1)[:, -5:][:, ::-1]
    idcg = (ideal * DISCOUNT).sum(1)
    return float(np.mean(dcg / np.maximum(idcg, 1e-12)))


def make_ordinal_targets(rel):
    out = np.zeros_like(rel, dtype=np.float32)
    for i, row in enumerate(rel):
        pos = np.where(row > 0)[0]
        order = pos[np.argsort(-row[pos])]
        for rank, j in enumerate(order):
            out[i, j] = len(order) - rank
    return out

Y_ORD = make_ordinal_targets(y)
Y_BIN = (y > 0).astype(np.float32)
Y_TOP = np.zeros_like(y, dtype=np.float32)
Y_TOP[np.arange(len(y)), np.argmax(y, axis=1)] = 1.0

# Preserve the top-1 module distribution in every fold.
strata = np.array([MODULES[j] for j in np.argmax(y, axis=1)])
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
folds = list(skf.split(np.zeros(len(train)), strata))
print("Fold sizes:", [len(v) for _, v in folds])

# ---- cell 4 (code) ----
# ---------- Robust assessment parsing ----------
WORD_NUM = {"nol":0, "satu":1, "dua":2, "tiga":3, "empat":4, "lima":5}
SKILL_NAMES = [
    "python", "sql", "statistics", "eda", "ml", "ml_eval", "dl",
    "genai", "business", "research"
]

A = np.zeros((len(all_users), 10), np.float32)
A_INVALID = np.zeros_like(A)
assessment_questions = None

for row in assessments.itertuples(index=False):
    try:
        d = json.loads(row.assessment_result)
    except Exception:
        d = ast.literal_eval(row.assessment_result)
    if assessment_questions is None:
        assessment_questions = list(d.keys())
    vals, invalid = [], []
    for v in d.values():
        sv = str(v).strip().lower()
        if sv in WORD_NUM:
            value, bad = float(WORD_NUM[sv]), 0
        else:
            try:
                value = float(v)
            except Exception:
                value = np.nan
            bad = int((not np.isfinite(value)) or value < 0 or value > 5)
            if bad:
                value = np.nan
        vals.append(value)
        invalid.append(bad)
    vals = np.asarray(vals, float)
    med = float(np.nanmedian(vals)) if np.isfinite(vals).any() else 2.0
    vals = np.where(np.isfinite(vals), vals, med)
    i = uid2idx[row.user_id]
    A[i] = vals
    A_INVALID[i] = invalid

print("Invalid assessment cells:", int(A_INVALID.sum()), "/", A_INVALID.size)
print("Train/test assessment means:")
print(pd.DataFrame({
    "skill": SKILL_NAMES,
    "train": A[:len(train)].mean(0),
    "test": A[len(train):].mean(0),
}).round(3).to_string(index=False))

# ---- cell 5 (code) ----
# ---------- Intent/topic parser ----------
# These dictionaries are intentionally small and interpretable. They encode phrases
# that are visibly repeated in the provided dataset, not external knowledge/data.
TOPIC_PATTERNS = {
    "excel": r"\b(?:ms\.?\s*excel|excel|pivot(?: table)?|vlookup|hlookup|spreadsheet)\b",
    "python": r"\b(?:python|pyton|pithon|pandas(?:\s*&\s*| dan |\s+)?numpy|pandas|numpy)\b",
    "sql": r"\b(?:sql|escuel|query database|join tabel|database relasional|window function)\b",
    "statistics": r"\b(?:statistik(?:\s*&\s*probabilitas)?|probabilitas|a\s*/?\s*b testing|uji hipotesis|distribusi data)\b",
    "eda": r"\b(?:eda|exploratory data analysis|data cleaning|missing values?|matplotlib|seaborn)\b",
    "ml_eval": r"\b(?:evaluasi model(?: ml)?|metrik model|metric model|evaluasi performa model)\b",
    "mlops": r"\b(?:mlops|deployment model|deploy(?:ment)? model|ci\s*/?\s*cd|monitoring model|serving model|fastapi)\b",
    "cv": r"\b(?:computer vision|cnn|deep learning|deteksi objek|klasifikasi gambar|image classification)\b",
    "nlp": r"\b(?:natural language processing|nlp|analisis sentimen|sentiment analysis|word2vec|tf\s*-?\s*idf|transformer)\b",
    "genai": r"\b(?:generative ai|genai|large language model|llm|rag|chatgpt|chatbot|fine[- ]?tuning model bahasa)\b",
    "ml": r"\b(?:machine learning|bikin model ml|model machine learning|regresi\s*(?:&|dan)\s*klasifikasi|clustering|supervised|unsupervised|hyperparameter)\b",
    "business": r"\b(?:translate masalah bisnis ke data|menerjemahkan masalah bisnis|business(?:-| )to(?:-| )ds|permasalahan bisnis)\b",
    "dashboard": r"\b(?:dashboard|looker studio|tableau|visualisasi interaktif|storytelling data)\b",
    "webscrape": r"\b(?:web scrap(?:e|ing)|scraping|beautifulsoup|selenium|ekstraksi data.*website)\b",
    "git": r"\b(?:version control|git(?:hub|lab)?|repository|repo kode)\b",
    "prompt": r"\b(?:prompt engineering|prompting|merancang prompt|optimasi prompt)\b",
    "automation": r"\b(?:ai automation|otomasi|otomatis|automation|workflow|n8n)\b",
    "nocode": r"\b(?:no[ -]?code|zero coding|tanpa perlu (?:belajar )?(?:coding|koding)|gausah (?:coding|koding)|gak perlu (?:coding|koding)|tanpa (?:coding|koding))\b",
    "career": r"\b(?:portofolio|portfolio|cv|test interview|interview|karir|career|prospek kerja|negosiasi)\b",
}
TOPIC_RX = {k: re.compile(v, re.I) for k,v in TOPIC_PATTERNS.items()}
TOPICS = list(TOPIC_PATTERNS)

NOISE_RX = re.compile(
    r"reset password|email verifikasi|sertifikat kelulusan|pembayaran|cicilan|invoice|"
    r"akun.*(?:error|masalah)|lms.*(?:error|masalah)", re.I
)
STYLE_PATTERNS = {
    "beginner": r"pemula|dari nol|gaptek|belum pernah (?:ngoding|coding|koding)|background(?:nya)? bukan (?:it|data)|banting setir",
    "advanced": r"udah lumayan paham|sudah lumayan paham|stuck|prasyarat|syarat|\breq\b|basic apa|lanjutin",
    "urgent": r"urgent|secepatnya|sefastnya|intensif|seminggu|minggu depan",
    "budget": r"budget|diskon|worth it|cekak|fomo|kompetitor",
    "online": r"online|durasi|kapan buka|mulai minggu|jadwal",
    "nocode_style": r"tanpa perlu.*(?:coding|koding)|gausah.*(?:coding|koding)|gak perlu.*(?:coding|koding)|otomatis.*ai|solusi otomatis",
    "career_style": r"portofolio|portfolio|test interview|prospek kerja|karir|career",
}
STYLE_RX = {k: re.compile(v, re.I) for k,v in STYLE_PATTERNS.items()}
STYLES = list(STYLE_PATTERNS)


def clean_text(text):
    text = html.unescape(str(text)).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " <url> ", text)
    text = re.sub(r"[^\w\s<>/&+-]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def extract_topics(text):
    return [k for k, rx in TOPIC_RX.items() if rx.search(str(text))]


def make_skeleton(text):
    s = clean_text(text)
    for _, rx in TOPIC_RX.items():
        s = rx.sub(" <topic> ", s)
    s = re.sub(r"\b\d+\b", " <num> ", s)
    return re.sub(r"\s+", " ", s).strip()


def template_token(skeleton):
    # Stable short token; avoids huge raw template strings in the vocabulary.
    return "tpl_" + hashlib.md5(skeleton.encode("utf-8")).hexdigest()[:10]

chat["timestamp"] = pd.to_datetime(chat["timestamp"], errors="coerce")
chat["clean"] = chat["user_chat_text"].fillna("").map(clean_text)
chat["is_noise"] = chat["user_chat_text"].fillna("").map(lambda x: bool(NOISE_RX.search(str(x))))
chat["topics"] = chat["user_chat_text"].fillna("").map(extract_topics)
chat["skeleton"] = chat["user_chat_text"].fillna("").map(make_skeleton)
for name, rx in STYLE_RX.items():
    chat["sty_" + name] = chat["user_chat_text"].fillna("").map(lambda x, r=rx: int(bool(r.search(str(x)))))

chat = chat.sort_values(["user_id", "timestamp", "chat_id"]).reset_index(drop=True)
user_chat = {u: g.copy() for u,g in chat.groupby("user_id", sort=False)}
print("Administrative/noise rows:", int(chat["is_noise"].sum()), "/", len(chat))

# ---- cell 6 (code) ----
# ---------- Build template-aware text + structured numeric user features ----------
# Text branch memorizes repeated wording/template patterns.
# Numeric branch exposes nonlinear assessment/readiness and recency interactions.

USER_FEATURES = np.zeros((len(all_users), 10 + 10 + 5 + len(STYLES)*2 + len(TOPICS)*5), np.float32)


def build_user(u, recent_only=False):
    ui = uid2idx[u]
    g = user_chat.get(u)
    parts = []
    if g is None:
        learn = pd.DataFrame(columns=chat.columns)
        g = learn
    else:
        learn = g[~g["is_noise"]].copy()
        if len(learn) == 0:
            learn = g.copy()
    if recent_only:
        learn_for_text = learn.tail(3)
    else:
        learn_for_text = learn

    n = len(learn_for_text)
    for j, row in enumerate(learn_for_text.itertuples(index=False)):
        distance = n - 1 - j
        tag = "current" if distance == 0 else ("recent" if distance <= 2 else "older")
        parts.append(f"[{tag}] {row.clean}")
        parts.append(template_token(row.skeleton))
        for tp in row.topics:
            parts.append("topic_" + tp)
            if distance == 0:
                parts.extend(["latest_topic_" + tp] * 3)
            elif distance <= 2:
                parts.extend(["recent_topic_" + tp] * 2)
        for sty in STYLES:
            if getattr(row, "sty_" + sty):
                parts.append("style_" + sty)

    # Assessment as discrete tokens for the sparse branch.
    for j, skill in enumerate(SKILL_NAMES):
        v = int(round(float(A[ui, j])))
        band = "low" if v <= 1 else ("mid" if v <= 3 else "high")
        parts += [f"skill_{skill}_{v}", f"skill_{skill}_bin_{band}"]

    # Structured numeric branch uses the complete learning history.
    feats = list(A[ui]) + list(A_INVALID[ui])
    total = len(g)
    learning_count = len(learn)
    noise_count = int(g["is_noise"].sum()) if total else 0
    span = 0.0
    if total >= 2 and g["timestamp"].notna().sum() >= 2:
        span = float((g["timestamp"].max() - g["timestamp"].min()).total_seconds() / 86400.0)
    feats += [total, learning_count, noise_count, span, min(total, 6)]

    for sty in STYLES:
        feats.append(float(learn["sty_" + sty].sum()) if learning_count else 0.0)
    for sty in STYLES:
        feats.append(float(learn.iloc[-1]["sty_" + sty]) if learning_count else 0.0)

    rows = list(learn.itertuples(index=False))
    nfull = len(rows)
    for tp in TOPICS:
        count = rec_count = last_pos = 0.0
        latest = 0.0
        for j, row in enumerate(rows):
            if tp in row.topics:
                count += 1.0
                rec_count += float(np.exp(-0.7 * (nfull - 1 - j)))
                last_pos = max(last_pos, (j + 1) / max(nfull, 1))
        if rows and tp in rows[-1].topics:
            latest = 1.0
        feats += [count, rec_count, latest, last_pos, float(count > 0)]

    USER_FEATURES[ui] = np.asarray(feats, np.float32)
    return " ".join(parts)

texts = np.asarray([build_user(u, recent_only=False) for u in all_users], dtype=object)
recent_texts = np.asarray([build_user(u, recent_only=True) for u in all_users], dtype=object)

print("Structured user feature shape:", USER_FEATURES.shape)
print("Example augmented text:\n", texts[1][:1200])

# ---- cell 7 (code) ----
# ---------- Word + character TF-IDF ----------
t0 = time.time()
word_vec = TfidfVectorizer(
    ngram_range=(1,2), min_df=2, max_df=0.995, sublinear_tf=True,
    strip_accents="unicode", max_features=50_000, dtype=np.float32
)
char_vec = TfidfVectorizer(
    analyzer="char_wb", ngram_range=(3,5), min_df=2, sublinear_tf=True,
    max_features=60_000, dtype=np.float32
)
recent_word_vec = TfidfVectorizer(
    ngram_range=(1,2), min_df=2, max_df=0.995, sublinear_tf=True,
    strip_accents="unicode", max_features=35_000, dtype=np.float32
)
recent_char_vec = TfidfVectorizer(
    analyzer="char_wb", ngram_range=(3,5), min_df=2, sublinear_tf=True,
    max_features=40_000, dtype=np.float32
)

# Fitting vocabulary on train+test is unsupervised/transductive and uses only competition data.
X = hstack([word_vec.fit_transform(texts), char_vec.fit_transform(texts)], format="csr")
X_recent = hstack([
    recent_word_vec.fit_transform(recent_texts),
    recent_char_vec.fit_transform(recent_texts)
], format="csr")

print("X       :", X.shape, "nnz=", X.nnz)
print("X_recent:", X_recent.shape, "nnz=", X_recent.nnz)
print("Vectorization seconds:", round(time.time()-t0, 1))

# ---- cell 8 (code) ----
# ---------- Five supervised sparse heads ----------
# Each test prediction is averaged over fold models trained on 80% of train.
# This deliberately matches the OOF score distribution used by the calibration layer.
heads = {
    "raw": (y, X),
    "ordinal": (Y_ORD, X),
    "top1": (Y_TOP, X),
    "binary": (Y_BIN, X),
    "recent": (Y_ORD, X_recent),
}

oof_heads = {}
test_heads = {}

for name, (target, matrix) in heads.items():
    print(f"\nSparse head: {name}")
    oof = np.zeros_like(y, dtype=np.float32)
    pred_test = np.zeros((len(test), len(MODULES)), np.float32)
    for fold, (tr_idx, va_idx) in enumerate(folds):
        model = Ridge(alpha=RIDGE_ALPHA, solver="lsqr", tol=2e-3, max_iter=1000)
        model.fit(matrix[tr_idx], target[tr_idx])
        oof[va_idx] = model.predict(matrix[va_idx]).astype(np.float32)
        pred_test += model.predict(matrix[len(train):]).astype(np.float32) / N_FOLDS
    oof_heads[name] = oof
    test_heads[name] = pred_test
    print(f"  OOF NDCG@5 = {ndcg5(y, oof):.6f}")

# Pre-registered blend from offline ablation: raw relevance dominates; top-1 head adds diversity.
base_oof = 0.90 * oof_heads["raw"] + 0.10 * oof_heads["top1"]
base_test = 0.90 * test_heads["raw"] + 0.10 * test_heads["top1"]
print("\nSparse backbone OOF NDCG@5:", round(ndcg5(y, base_oof), 6))

# ---- cell 9 (code) ----
# ---------- Nonlinear per-module relevance calibration ----------
# A separate small model per module is intentional: the same assessment score can mean
# very different things for Python, MLOps, Prompt Engineering, etc.
#
# Early stopping is performed strictly inside each validation fold. The selected
# iteration counts are then reused for the full-data test fit, avoiding a fixed
# over-trained tree count.

meta_oof = np.zeros_like(y, dtype=np.float32)
best_iters = {mid: [] for mid in MODULES}

for m, module_id in enumerate(MODULES):
    Xm = np.column_stack([base_oof[:, m], USER_FEATURES[:len(train)]])
    target = Y_BIN[:, m].astype(np.int32)
    for fold, (tr_idx, va_idx) in enumerate(folds):
        clf = lgb.LGBMClassifier(
            n_estimators=1200,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=50,
            reg_alpha=0.5,
            reg_lambda=5.0,
            colsample_bytree=0.70,
            random_state=SEED + fold + 31*m,
            n_jobs=-1,
            verbosity=-1,
        )
        clf.fit(
            Xm[tr_idx], target[tr_idx],
            eval_set=[(Xm[va_idx], target[va_idx])],
            callbacks=[lgb.early_stopping(80, verbose=False)],
        )
        best_iters[module_id].append(int(clf.best_iteration_ or 1200))
        meta_oof[va_idx, m] = clf.predict_proba(
            Xm[va_idx], num_iteration=clf.best_iteration_
        )[:, 1].astype(np.float32)

print("Calibration-only OOF NDCG@5:", round(ndcg5(y, meta_oof), 6))
print("Median calibration iterations:", {k:int(np.median(v)) for k,v in best_iters.items()})

for w in [0.00, 0.20, 0.35, 0.50, 0.60]:
    sc = ndcg5(y, (1-w)*base_oof + w*meta_oof)
    print(f"blend calibration={w:0.2f}: {sc:.6f}")

primary_oof = (1-CAL_BLEND_PRIMARY)*base_oof + CAL_BLEND_PRIMARY*meta_oof
conservative_oof = (1-CAL_BLEND_CONSERVATIVE)*base_oof + CAL_BLEND_CONSERVATIVE*meta_oof
print("\nPRIMARY OOF NDCG@5:", round(ndcg5(y, primary_oof), 6))
print("CONSERVATIVE OOF NDCG@5:", round(ndcg5(y, conservative_oof), 6))

# ---- cell 10 (code) ----
# ---------- Fit calibration layer on all train OOF scores; predict test ----------
# Test uses fold-averaged sparse predictions, which are produced by models trained on
# the same number of users as each OOF model (3200). Calibration tree counts come from
# the median early-stopped iteration across the five validation folds.
meta_test = np.zeros((len(test), len(MODULES)), np.float32)

for m, module_id in enumerate(MODULES):
    Xtr_meta = np.column_stack([base_oof[:, m], USER_FEATURES[:len(train)]])
    Xte_meta = np.column_stack([base_test[:, m], USER_FEATURES[len(train):]])
    target = Y_BIN[:, m].astype(np.int32)
    final_iters = max(20, int(np.median(best_iters[module_id])))
    clf = lgb.LGBMClassifier(
        n_estimators=final_iters,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=50,
        reg_alpha=0.5,
        reg_lambda=5.0,
        colsample_bytree=0.70,
        random_state=SEED + 997 + 31*m,
        n_jobs=-1,
        verbosity=-1,
    )
    clf.fit(Xtr_meta, target)
    meta_test[:, m] = clf.predict_proba(Xte_meta)[:, 1].astype(np.float32)

primary_test = (1-CAL_BLEND_PRIMARY)*base_test + CAL_BLEND_PRIMARY*meta_test
conservative_test = (1-CAL_BLEND_CONSERVATIVE)*base_test + CAL_BLEND_CONSERVATIVE*meta_test

# ---- cell 11 (code) ----
# ---------- Diagnostics ----------
def topk_recall_by_module(y_true, scores, k=5):
    top = np.argsort(-scores, axis=1)[:, :k]
    rows = []
    for j, mid in enumerate(MODULES):
        relevant = y_true[:, j] > 0
        if not relevant.any():
            recall = np.nan
        else:
            recall = np.mean([j in top[i] for i in np.where(relevant)[0]])
        rows.append((mid, modules.loc[modules.module_id.eq(mid), "module_name"].iloc[0], relevant.sum(), recall))
    return pd.DataFrame(rows, columns=["module_id","module_name","n_relevant","recall_at_5"])

base_diag = topk_recall_by_module(y, base_oof).rename(columns={"recall_at_5":"base_recall5"})
pri_diag = topk_recall_by_module(y, primary_oof).rename(columns={"recall_at_5":"primary_recall5"})
diag = base_diag.merge(pri_diag[["module_id","primary_recall5"]], on="module_id")
diag["delta"] = diag["primary_recall5"] - diag["base_recall5"]
print(diag.sort_values("primary_recall5").round(4).to_string(index=False))

# Test top-5 frequency. Extreme zero/near-100% frequencies are a warning sign.
top5_test = np.argsort(-primary_test, axis=1)[:, :5]
freq = np.bincount(top5_test.ravel(), minlength=len(MODULES))
print("\nTest Top-5 frequencies:")
print(pd.DataFrame({"module_id":MODULES, "count":freq}).to_string(index=False))

# ---- cell 12 (code) ----
# ---------- Submission writer + hard sanity checks ----------
def write_submission(scores, filename):
    assert scores.shape == (len(test), len(MODULES))
    assert np.isfinite(scores).all()
    out = sample.copy()
    assert out["user_id"].tolist() == test["user_id"].tolist()
    out.loc[:, MODULES] = scores
    path = Path("/kaggle/working") / filename if Path("/kaggle/working").exists() else Path("/mnt/data") / filename
    out.to_csv(path, index=False)
    check = pd.read_csv(path)
    assert check.shape == sample.shape
    assert check["user_id"].is_unique
    assert check[MODULES].notna().all().all()
    print("Saved:", path)
    return path

primary_path = write_submission(primary_test, "submission_v31_primary.csv")
conservative_path = write_submission(conservative_test, "submission_v31_conservative.csv")
sparse_path = write_submission(base_test, "submission_v31_sparse_safe.csv")

np.savez_compressed(
    "/kaggle/working/v31_oof_artifacts.npz" if Path("/kaggle/working").exists() else "/mnt/data/v31_oof_artifacts.npz",
    y=y, base_oof=base_oof, meta_oof=meta_oof, primary_oof=primary_oof,
    base_test=base_test, meta_test=meta_test, primary_test=primary_test,
)

print("\nFinal summary")
print("Sparse backbone :", round(ndcg5(y, base_oof), 6))
print("Primary V3.1    :", round(ndcg5(y, primary_oof), 6))
print("Conservative    :", round(ndcg5(y, conservative_oof), 6))
print("Recommended first submission:", primary_path.name)
