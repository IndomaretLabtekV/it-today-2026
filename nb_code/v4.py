# Converted from notebooks/v4.ipynb: markdown cells become comments, %/! magics neutralized.

# ---- cell 0 (markdown) ----
# # MineToday IT Today 2026 — V4 Multi-Expert Structured LambdaRank
#
# V4 is a **dataset-native learning-to-rank system** built from the conclusions of V1–V3.1:
#
# - repeated/synthetic lexical patterns are stronger than expensive cross-encoders,
# - assessment behaves like a nonlinear readiness / skill-gap gate,
# - current intent and long-history signals are complementary,
# - binary relevance calibration improves Top-5 coverage but does not fully optimize ordering,
# - therefore the final layer must be rank-aware and optimized around NDCG@5.
#
# Pipeline:
# 1. deterministic multilabel-balanced 5-fold CV,
# 2. template/topic-aware full, recent, and latest TF-IDF views,
# 3. six supervised Ridge ranking heads,
# 4. fold-stable sparse blending,
# 5. structured per-module relevance calibration,
# 6. message-level max expert for multi-interest / long-history users,
# 7. module co-occurrence / learning-path expert,
# 8. candidate-level LightGBM LambdaRank reranker,
# 9. conservative topic/history gated mixture.
#
# There is **no transformer fine-tuning**. The notebook is intended to finish in minutes on Kaggle; T4x2 may remain enabled but GPU is not required.
#
# Outputs:
# - `submission_v4_primary.csv`
# - `submission_v4_ranker.csv`
# - `submission_v4_v31_anchor.csv`
# - `v4_oof_artifacts.npz`

# ---- cell 1 (code) ----
import os, re, json, ast, html, hashlib, random, warnings, gc, time, sys, subprocess
from pathlib import Path
from collections import defaultdict

# Keep classical CPU models deterministic and avoid thread oversubscription.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ["PYTHONHASHSEED"] = "42"

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")
try:
    import lightgbm as lgb
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "lightgbm==4.6.0"])
    import lightgbm as lgb

SEED = 42
N_FOLDS = 5
RIDGE_ALPHA = 3.0
REL_WEIGHT = 0.35          # public-validated V3.1 conservative calibration weight
COOC_TEMP = 0.20
RANK_GATE_WEIGHT = 0.75
LONG_HISTORY_MIN = 5
LONG_HISTORY_MSG_WEIGHT = 0.15

random.seed(SEED)
np.random.seed(SEED)
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
    raise FileNotFoundError("MineToday dataset directory not found.")

DATA_DIR = locate_data_dir()
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
assert train["user_id"].is_unique and test["user_id"].is_unique

all_users = train["user_id"].tolist() + test["user_id"].tolist()
uid2idx = {u:i for i,u in enumerate(all_users)}
y = train[MODULES].to_numpy(np.float32)
N_TRAIN, N_MODULES = y.shape

print("DATA_DIR   :", DATA_DIR)
print("chat       :", chat.shape)
print("modules    :", modules.shape)
print("assessment :", assessments.shape)
print("train/test :", train.shape, test.shape)
print("positive modules/user:", pd.Series((y > 0).sum(1)).value_counts().sort_index().to_dict())

# ---- cell 3 (code) ----
# ---------- Metric, targets, and multilabel-balanced folds ----------
DISCOUNT = 1.0 / np.log2(np.arange(2, 7))

def ndcg5(y_true, scores):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    idx = np.argsort(-scores, axis=1)[:, :5]
    dcg = (np.take_along_axis(y_true, idx, axis=1) * DISCOUNT).sum(1)
    ideal = np.sort(y_true, axis=1)[:, -5:][:, ::-1]
    idcg = (ideal * DISCOUNT).sum(1)
    return float(np.mean(dcg / np.maximum(idcg, 1e-12)))

def per_user_ndcg5(y_true, scores):
    idx = np.argsort(-scores, axis=1)[:, :5]
    dcg = (np.take_along_axis(y_true, idx, axis=1) * DISCOUNT).sum(1)
    ideal = np.sort(y_true, axis=1)[:, -5:][:, ::-1]
    return dcg / np.maximum((ideal * DISCOUNT).sum(1), 1e-12)

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
Y_TOP = np.zeros_like(y, np.float32)
TOP1 = np.argmax(y, axis=1)
Y_TOP[np.arange(N_TRAIN), TOP1] = 1.0


def make_multilabel_balanced_folds(y_bin, top1, n_splits=5, seed=42):
    """Greedy deterministic iterative-style stratification.

    Balances both all relevant-module indicators and the top-1 module while
    enforcing exactly equal fold sizes for this 4000-user dataset.
    """
    rng = np.random.default_rng(seed)
    eye = np.eye(y_bin.shape[1], dtype=np.int8)[top1]
    labels = np.concatenate([y_bin.astype(np.int8), eye], axis=1)
    n, L = labels.shape
    assert n % n_splits == 0
    target_size = n // n_splits
    desired = labels.sum(0).astype(float) / n_splits
    current = np.zeros((n_splits, L), float)
    sizes = np.zeros(n_splits, int)
    freq = np.maximum(labels.sum(0), 1)
    rarity = (labels / freq).sum(1)
    order = np.argsort(-(rarity + rng.random(n) * 1e-9))
    fold_id = np.full(n, -1, int)

    for i in order:
        active = np.flatnonzero(labels[i])
        score = np.full(n_splits, -1e12, float)
        for f in range(n_splits):
            if sizes[f] >= target_size:
                continue
            label_need = ((desired[active] - current[f, active]) / np.maximum(desired[active], 1)).sum()
            size_need = (target_size - sizes[f]) / target_size
            score[f] = label_need + 0.50 * size_need
        best = np.flatnonzero(score == score.max())
        f = int(rng.choice(best))
        fold_id[i] = f
        sizes[f] += 1
        current[f] += labels[i]

    assert np.all(fold_id >= 0)
    assert sizes.tolist() == [target_size] * n_splits
    folds = [(np.where(fold_id != f)[0], np.where(fold_id == f)[0]) for f in range(n_splits)]
    return folds, fold_id

folds, fold_id = make_multilabel_balanced_folds(Y_BIN, TOP1, N_FOLDS, SEED)
rel_rate = Y_BIN.mean(0)
top_rate = np.eye(N_MODULES)[TOP1].mean(0)
max_rel_dev = max(np.abs(Y_BIN[fold_id == f].mean(0) - rel_rate).max() for f in range(N_FOLDS))
max_top_dev = max(np.abs(np.eye(N_MODULES)[TOP1[fold_id == f]].mean(0) - top_rate).max() for f in range(N_FOLDS))
print("Fold sizes:", [len(v) for _, v in folds])
print("max relevant-rate deviation:", round(float(max_rel_dev), 5))
print("max top1-rate deviation    :", round(float(max_top_dev), 5))
assert max_rel_dev < 0.01 and max_top_dev < 0.01

# ---- cell 4 (code) ----
# ---------- Robust assessment parsing ----------
WORD_NUM = {"nol":0, "satu":1, "dua":2, "tiga":3, "empat":4, "lima":5}
SKILL_NAMES = ["python", "sql", "statistics", "eda", "ml", "ml_eval", "dl", "genai", "business", "research"]
SKILL_INDEX = {s:i for i,s in enumerate(SKILL_NAMES)}

A = np.zeros((len(all_users), len(SKILL_NAMES)), np.float32)
A_INVALID = np.zeros_like(A)
for row in assessments.itertuples(index=False):
    try:
        d = json.loads(row.assessment_result)
    except Exception:
        d = ast.literal_eval(row.assessment_result)
    vals, bads = [], []
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
        vals.append(value); bads.append(bad)
    vals = np.asarray(vals, float)
    med = float(np.nanmedian(vals)) if np.isfinite(vals).any() else 2.0
    vals = np.where(np.isfinite(vals), vals, med)
    i = uid2idx.get(row.user_id)
    if i is not None:
        A[i] = vals
        A_INVALID[i] = bads

print("Invalid assessment cells:", int(A_INVALID.sum()), "/", A_INVALID.size)
assert np.isfinite(A).all()

# ---- cell 5 (code) ----
# ---------- Dataset-native topic/template parser ----------
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
TOPICS = list(TOPIC_PATTERNS)
TOPIC_INDEX = {t:i for i,t in enumerate(TOPICS)}
TOPIC_RX = {k:re.compile(v, re.I) for k,v in TOPIC_PATTERNS.items()}
NOISE_RX = re.compile(r"reset password|email verifikasi|sertifikat kelulusan|pembayaran|cicilan|invoice|akun.*(?:error|masalah)|lms.*(?:error|masalah)", re.I)
STYLE_PATTERNS = {
    "beginner": r"pemula|dari nol|gaptek|belum pernah (?:ngoding|coding|koding)|background(?:nya)? bukan (?:it|data)|banting setir",
    "advanced": r"udah lumayan paham|sudah lumayan paham|stuck|prasyarat|syarat|\breq\b|basic apa|lanjutin",
    "urgent": r"urgent|secepatnya|sefastnya|intensif|seminggu|minggu depan",
    "budget": r"budget|diskon|worth it|cekak|fomo|kompetitor",
    "online": r"online|durasi|kapan buka|mulai minggu|jadwal",
    "nocode_style": r"tanpa perlu.*(?:coding|koding)|gausah.*(?:coding|koding)|gak perlu.*(?:coding|koding)|otomatis.*ai|solusi otomatis",
    "career_style": r"portofolio|portfolio|test interview|prospek kerja|karir|career",
}
STYLES = list(STYLE_PATTERNS)
STYLE_RX = {k:re.compile(v, re.I) for k,v in STYLE_PATTERNS.items()}

def clean_text(text):
    text = html.unescape(str(text)).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " <url> ", text)
    text = re.sub(r"[^\w\s<>/&+-]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()

def extract_topics(text):
    return [k for k,rx in TOPIC_RX.items() if rx.search(str(text))]

def make_skeleton(text):
    s = clean_text(text)
    for rx in TOPIC_RX.values():
        s = rx.sub(" <topic> ", s)
    s = re.sub(r"\b\d+\b", " <num> ", s)
    return re.sub(r"\s+", " ", s).strip()

def template_token(s):
    return "tpl_" + hashlib.md5(s.encode("utf-8")).hexdigest()[:10]

chat["timestamp"] = pd.to_datetime(chat["timestamp"], errors="coerce")
chat["clean"] = chat["user_chat_text"].fillna("").map(clean_text)
chat["is_noise"] = chat["user_chat_text"].fillna("").map(lambda x: bool(NOISE_RX.search(str(x))))
chat["topics"] = chat["user_chat_text"].fillna("").map(extract_topics)
chat["skeleton"] = chat["user_chat_text"].fillna("").map(make_skeleton)
for name,rx in STYLE_RX.items():
    chat["sty_"+name] = chat["user_chat_text"].fillna("").map(lambda x,r=rx: int(bool(r.search(str(x)))))
chat = chat.sort_values(["user_id","timestamp","chat_id"]).reset_index(drop=True)
user_chat = {u:g.copy() for u,g in chat.groupby("user_id", sort=False)}
print("Administrative/noise rows:", int(chat.is_noise.sum()), "/", len(chat))

# ---- cell 6 (code) ----
# ---------- User views + named structured features ----------
N_ALL = len(all_users); T = len(TOPICS); S = len(STYLES)
topic_count = np.zeros((N_ALL,T), np.float32)
topic_rec = np.zeros_like(topic_count)
topic_latest = np.zeros_like(topic_count)
topic_lastpos = np.zeros_like(topic_count)
topic_present = np.zeros_like(topic_count)
style_count = np.zeros((N_ALL,S), np.float32)
style_latest = np.zeros_like(style_count)
general = np.zeros((N_ALL,5), np.float32)  # total, learning, noise, timespan_days, clipped_count
latest_primary_topic = np.asarray(["none"] * N_ALL, dtype=object)

full_texts, recent_texts, latest_texts = [], [], []
learning_rows = []

for u in all_users:
    ui = uid2idx[u]
    g = user_chat.get(u)
    if g is None:
        g = pd.DataFrame(columns=chat.columns)
    learn = g[~g["is_noise"]].copy() if len(g) else g.copy()
    if len(g) and len(learn) == 0:
        learn = g.copy()
    learn = learn.sort_values(["timestamp","chat_id"]) if len(learn) else learn
    rows = list(learn.itertuples(index=False))
    n = len(rows)

    def render(selected_rows):
        parts=[]; nn=len(selected_rows)
        for j,row in enumerate(selected_rows):
            dist=nn-1-j
            tag="current" if dist==0 else ("recent" if dist<=2 else "older")
            parts += [f"[{tag}] {row.clean}", template_token(row.skeleton)]
            for tp in row.topics:
                parts.append("topic_"+tp)
                if dist==0: parts += ["latest_topic_"+tp] * 3
                elif dist<=2: parts += ["recent_topic_"+tp] * 2
            for si,sty in enumerate(STYLES):
                if getattr(row,"sty_"+sty): parts.append("style_"+sty)
        for si,skill in enumerate(SKILL_NAMES):
            v=int(round(float(A[ui,si]))); band="low" if v<=1 else ("mid" if v<=3 else "high")
            parts += [f"skill_{skill}_{v}", f"skill_{skill}_bin_{band}"]
        return " ".join(parts)

    full_texts.append(render(rows))
    recent_texts.append(render(rows[-3:]))
    latest_texts.append(render(rows[-1:]))

    total=len(g); noise_count=int(g["is_noise"].sum()) if total else 0; span=0.0
    if total>=2 and g["timestamp"].notna().sum()>=2:
        span=float((g["timestamp"].max()-g["timestamp"].min()).total_seconds()/86400.0)
    general[ui]=[total,n,noise_count,span,min(total,6)]

    for si,sty in enumerate(STYLES):
        style_count[ui,si]=float(learn["sty_"+sty].sum()) if n else 0.0
        style_latest[ui,si]=float(getattr(rows[-1],"sty_"+sty)) if n else 0.0

    if n and rows[-1].topics:
        latest_primary_topic[ui]=rows[-1].topics[0]
    for j,row in enumerate(rows):
        for tp in row.topics:
            ti=TOPIC_INDEX[tp]
            topic_count[ui,ti]+=1
            topic_rec[ui,ti]+=float(np.exp(-0.7*(n-1-j)))
            topic_lastpos[ui,ti]=max(topic_lastpos[ui,ti], (j+1)/max(n,1))
            topic_present[ui,ti]=1
            if j==n-1: topic_latest[ui,ti]=1
        learning_rows.append((ui,row.clean,row.skeleton,row.topics))

USER_FEATURES = np.column_stack([
    A, A_INVALID, general, style_count, style_latest,
    topic_count, topic_rec, topic_latest, topic_lastpos, topic_present,
]).astype(np.float32)
full_texts=np.asarray(full_texts,dtype=object);recent_texts=np.asarray(recent_texts,dtype=object);latest_texts=np.asarray(latest_texts,dtype=object)
print("Structured user features:", USER_FEATURES.shape)
print("learning messages:", len(learning_rows))
print("latest topic distribution (train):", pd.Series(latest_primary_topic[:N_TRAIN]).value_counts().to_dict())
assert USER_FEATURES.shape[0] == 5000 and np.isfinite(USER_FEATURES).all()

# ---- cell 7 (code) ----
# ---------- Three TF-IDF views ----------
t0=time.time()
def make_view(texts, word_features, char_features):
    w=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_df=0.995,sublinear_tf=True,strip_accents="unicode",max_features=word_features,dtype=np.float32)
    c=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=2,sublinear_tf=True,max_features=char_features,dtype=np.float32)
    return hstack([w.fit_transform(texts),c.fit_transform(texts)],format="csr"), (w,c)

X_full, vec_full = make_view(full_texts, 50_000, 60_000)
X_recent, vec_recent = make_view(recent_texts, 35_000, 40_000)
X_latest, vec_latest = make_view(latest_texts, 30_000, 35_000)
print("X_full  :", X_full.shape, "nnz", X_full.nnz)
print("X_recent:", X_recent.shape, "nnz", X_recent.nnz)
print("X_latest:", X_latest.shape, "nnz", X_latest.nnz)
print("vectorization seconds:", round(time.time()-t0,1))

# ---- cell 8 (code) ----
# ---------- Six supervised sparse experts ----------
head_specs = {
    "raw_all": (y, X_full),
    "ord_all": (Y_ORD, X_full),
    "top1_all": (Y_TOP, X_full),
    "bin_all": (Y_BIN, X_full),
    "ord_recent": (Y_ORD, X_recent),
    "raw_latest": (y, X_latest),
}
oof_heads={}; test_heads={}
for name,(target,matrix) in head_specs.items():
    oof=np.zeros_like(y,np.float32);pred=np.zeros((len(test),N_MODULES),np.float32)
    for f,(tr_idx,va_idx) in enumerate(folds):
        model=Ridge(alpha=RIDGE_ALPHA,solver="lsqr",tol=2e-3,max_iter=1000)
        model.fit(matrix[tr_idx],target[tr_idx])
        oof[va_idx]=model.predict(matrix[va_idx]).astype(np.float32)
        pred += model.predict(matrix[N_TRAIN:]).astype(np.float32)/N_FOLDS
    oof_heads[name]=oof;test_heads[name]=pred
    print(f"{name:12s}: OOF NDCG@5={ndcg5(y,oof):.6f}")

# V3.1 public-validated sparse anchor for comparison.
v31_sparse_oof=.90*oof_heads["raw_all"]+.10*oof_heads["top1_all"]
v31_sparse_test=.90*test_heads["raw_all"]+.10*test_heads["top1_all"]
print("V3.1-style sparse anchor:", round(ndcg5(y,v31_sparse_oof),6))

# ---- cell 9 (code) ----
# ---------- Fold-stable low-dimensional sparse blend ----------
HEAD_NAMES=list(head_specs)
H_oof=np.stack([oof_heads[k] for k in HEAD_NAMES])
H_test=np.stack([test_heads[k] for k in HEAD_NAMES])
rng=np.random.default_rng(SEED)
candidates=[
    np.array([.90,0,.10,0,0,0],float),
    np.array([.60,.10,.15,.05,.05,.05],float),
    np.array([.55,.15,.15,.05,.07,.03],float),
]
for _ in range(450):
    w=rng.dirichlet([9,2,2,1.5,2,1.5])
    if w[0]>=.40 and w[2]<=.25 and w[5]<=.20:
        candidates.append(w)

selected=[];sparse_oof=np.zeros_like(y)
for held in range(N_FOLDS):
    train_folds=[f for f in range(N_FOLDS) if f!=held]
    best=(-1e9,None,None)
    for w in candidates:
        fold_scores=[]
        for f in train_folds:
            ix=(fold_id==f); pred=np.tensordot(w,H_oof[:,ix,:],axes=(0,0)); fold_scores.append(ndcg5(y[ix],pred))
        objective=float(np.mean(fold_scores)-0.15*np.std(fold_scores))
        if objective>best[0]: best=(objective,w,fold_scores)
    selected.append(best[1])
    ix=(fold_id==held); sparse_oof[ix]=np.tensordot(best[1],H_oof[:,ix,:],axes=(0,0))

stable_w=np.mean(selected,axis=0);stable_w/=stable_w.sum()
sparse_test=np.tensordot(stable_w,H_test,axes=(0,0)).astype(np.float32)
print("Fold-stable weights:", {k:round(float(v),4) for k,v in zip(HEAD_NAMES,stable_w)})
print("Sparse multi-expert OOF:", round(ndcg5(y,sparse_oof),6))

# ---- cell 10 (code) ----
# ---------- Structured per-module relevance calibration ----------
# This preserves the V3.1 mechanism that transferred strongly to public LB,
# but exposes all sparse-head candidate scores instead of only one score.
rel_oof=np.zeros_like(y,np.float32); rel_test=np.zeros((len(test),N_MODULES),np.float32)
best_iters={m:[] for m in MODULES}
head_oof_cols=np.stack([oof_heads[k] for k in HEAD_NAMES],axis=2)
head_test_cols=np.stack([test_heads[k] for k in HEAD_NAMES],axis=2)

for m,mid in enumerate(MODULES):
    Xm=np.column_stack([sparse_oof[:,m],head_oof_cols[:,m,:],USER_FEATURES[:N_TRAIN]])
    Xt=np.column_stack([sparse_test[:,m],head_test_cols[:,m,:],USER_FEATURES[N_TRAIN:]])
    target=Y_BIN[:,m].astype(np.int32)
    for f,(tr_idx,va_idx) in enumerate(folds):
        clf=lgb.LGBMClassifier(n_estimators=1000,learning_rate=.03,num_leaves=15,min_child_samples=55,reg_alpha=.6,reg_lambda=6.0,colsample_bytree=.72,random_state=SEED+31*m+f,n_jobs=-1,verbosity=-1,force_col_wise=True)
        clf.fit(Xm[tr_idx],target[tr_idx],eval_set=[(Xm[va_idx],target[va_idx])],callbacks=[lgb.early_stopping(70,verbose=False)])
        it=int(clf.best_iteration_ or 1000);best_iters[mid].append(it)
        rel_oof[va_idx,m]=clf.predict_proba(Xm[va_idx],num_iteration=it)[:,1]
    final_it=max(20,int(np.median(best_iters[mid])))
    clf=lgb.LGBMClassifier(n_estimators=final_it,learning_rate=.03,num_leaves=15,min_child_samples=55,reg_alpha=.6,reg_lambda=6.0,colsample_bytree=.72,random_state=SEED+997+31*m,n_jobs=-1,verbosity=-1,force_col_wise=True)
    clf.fit(Xm,target)
    rel_test[:,m]=clf.predict_proba(Xt)[:,1]

v31_anchor_oof=(1-REL_WEIGHT)*sparse_oof+REL_WEIGHT*rel_oof
v31_anchor_test=(1-REL_WEIGHT)*sparse_test+REL_WEIGHT*rel_test
print("Relevance-only OOF:", round(ndcg5(y,rel_oof),6))
print("V3.1-structured anchor OOF:", round(ndcg5(y,v31_anchor_oof),6))

# ---- cell 11 (code) ----
# ---------- message-level max expert ----------
# One message is treated as one interest view; max aggregation prevents older but
# still-relevant interests from being erased by the current message.
msg_df=pd.DataFrame(learning_rows,columns=["ui","clean","skeleton","topics"])
def augment_message(r):
    parts=[r.clean,template_token(r.skeleton)]
    parts += ["topic_"+t for t in r.topics]
    return " ".join(parts)
msg_text=msg_df.apply(augment_message,axis=1).to_numpy(dtype=object)
msg_w=TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True,max_features=45_000,dtype=np.float32)
msg_c=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=2,sublinear_tf=True,max_features=55_000,dtype=np.float32)
X_msg=hstack([msg_w.fit_transform(msg_text),msg_c.fit_transform(msg_text)],format="csr")
msg_ui=msg_df.ui.to_numpy(int)

msg_oof=np.zeros_like(y,np.float32);msg_test=np.zeros((len(test),N_MODULES),np.float32)
for f,(tr_idx,va_idx) in enumerate(folds):
    tr_mask=np.isin(msg_ui,tr_idx); va_mask=np.isin(msg_ui,va_idx); te_mask=msg_ui>=N_TRAIN
    tr_rows=np.where(tr_mask)[0]
    yt=y[msg_ui[tr_rows]]
    model=Ridge(alpha=5.0,solver="lsqr",tol=2e-3,max_iter=1000).fit(X_msg[tr_rows],yt)
    prior=y[tr_idx].mean(0)
    # validation aggregation
    msg_oof[va_idx]=prior
    vr=np.where(va_mask)[0]
    if len(vr):
        P=model.predict(X_msg[vr]).astype(np.float32)
        for ui in va_idx:
            loc=np.where(msg_ui[vr]==ui)[0]
            if len(loc): msg_oof[ui]=P[loc].max(0)
    # test aggregation per fold
    fold_test=np.tile(prior,(len(test),1)).astype(np.float32)
    rr=np.where(te_mask)[0]
    if len(rr):
        P=model.predict(X_msg[rr]).astype(np.float32)
        for j in range(len(test)):
            ui=N_TRAIN+j;loc=np.where(msg_ui[rr]==ui)[0]
            if len(loc): fold_test[j]=P[loc].max(0)
    msg_test += fold_test/N_FOLDS
print("message-level max OOF:", round(ndcg5(y,msg_oof),6))

# ---- cell 12 (code) ----
# ---------- Module co-occurrence / learning-path expert ----------
def softmax_rows(x,temp):
    z=x/temp;z=z-z.max(1,keepdims=True);e=np.exp(z);return e/e.sum(1,keepdims=True)

def transition_matrix(indices,alpha=8.0):
    prior=y[indices].mean(0);top=np.argmax(y[indices],axis=1);T=np.zeros((N_MODULES,N_MODULES),np.float32)
    for m in range(N_MODULES):
        ix=indices[top==m]
        T[m]=(y[ix].sum(0)+alpha*prior)/(len(ix)+alpha) if len(ix) else prior
    return T

cooc_oof=np.zeros_like(y,np.float32)
for tr_idx,va_idx in folds:
    Tm=transition_matrix(tr_idx)
    cooc_oof[va_idx]=softmax_rows(v31_anchor_oof[va_idx],COOC_TEMP)@Tm
cooc_test=softmax_rows(v31_anchor_test,COOC_TEMP)@transition_matrix(np.arange(N_TRAIN))
print("co-occurrence expert OOF:", round(ndcg5(y,cooc_oof),6))

# ---- cell 13 (code) ----
# ---------- Candidate feature matrix for rank-aware LambdaMART ----------
MODULE_TOPIC_MAP={
    "M_001":["excel"],"M_002":["python"],"M_003":["sql"],"M_004":["webscrape"],
    "M_005":["git"],"M_006":["statistics"],"M_007":["eda"],"M_008":["dashboard"],
    "M_009":["ml"],"M_010":["cv"],"M_011":["nlp"],"M_012":["genai"],
    "M_013":["prompt","genai"],"M_014":["automation"],"M_015":["nocode"],
    "M_016":["mlops"],"M_017":["career"],
}
PREREQ_SKILLS={
    "M_004":["python"],"M_007":["python","statistics"],"M_008":["sql"],
    "M_009":["eda","statistics"],"M_010":["ml","dl"],"M_011":["ml"],
    "M_012":["ml","genai"],"M_016":["ml","python","research"],
}
COURSE_SKILL={"M_002":"python","M_003":"sql","M_006":"statistics","M_007":"eda","M_009":"ml","M_010":"dl","M_012":"genai","M_016":"ml"}

expert_oof={"sparse":sparse_oof,"rel":rel_oof,"anchor":v31_anchor_oof,"msg":msg_oof,"cooc":cooc_oof,**oof_heads}
expert_test={"sparse":sparse_test,"rel":rel_test,"anchor":v31_anchor_test,"msg":msg_test,"cooc":cooc_test,**test_heads}
EXPERT_NAMES=list(expert_oof)

def row_ranks(a): return np.argsort(np.argsort(-a,axis=1),axis=1).astype(np.float32)
def zscore_rows(a): return ((a-a.mean(1,keepdims=True))/(a.std(1,keepdims=True)+1e-6)).astype(np.float32)

def explicit_pair_features(user_global_idx,m):
    mid=MODULES[m];topics=MODULE_TOPIC_MAP[mid];tis=[TOPIC_INDEX[t] for t in topics if t in TOPIC_INDEX]
    if tis:
        latest=float(topic_latest[user_global_idx,tis].max());present=float(topic_present[user_global_idx,tis].max())
        rec=float(topic_rec[user_global_idx,tis].max());cnt=float(topic_count[user_global_idx,tis].max());last=float(topic_lastpos[user_global_idx,tis].max())
    else: latest=present=rec=cnt=last=0.0
    ps=PREREQ_SKILLS.get(mid,[])
    pv=[A[user_global_idx,SKILL_INDEX[s]] for s in ps]
    pmin=float(min(pv)) if pv else 0.;pmean=float(np.mean(pv)) if pv else 0.;pready=float(bool(pv) and min(pv)>=3)
    cs=COURSE_SKILL.get(mid);cur=float(A[user_global_idx,SKILL_INDEX[cs]]) if cs else float(A[user_global_idx].mean());gap=5.0-cur
    return np.array([latest,present,rec,cnt,last,pmin,pmean,pready,cur,gap],np.float32)

def build_candidate_features(experts,user_offset=0):
    names=list(experts);arrays=[experts[k] for k in names];n=arrays[0].shape[0]
    ranks=[row_ranks(a) for a in arrays];zs=[zscore_rows(a) for a in arrays]
    core_names=["sparse","rel","anchor","msg","cooc"]
    full_arrays=[experts[k] for k in core_names]
    I=np.eye(N_MODULES,dtype=np.float32)
    nfeat=len(names)*4+USER_FEATURES.shape[1]+N_MODULES+N_MODULES*len(core_names)+10
    F=np.empty((n,N_MODULES,nfeat),np.float32)
    for i in range(n):
        gi=user_offset+i;full=np.concatenate([a[i] for a in full_arrays])
        for m in range(N_MODULES):
            own=[a[i,m] for a in arrays];rk=[r[i,m] for r in ranks];zz=[v[i,m] for v in zs];margin=[a[i,m]-a[i].max() for a in arrays]
            F[i,m]=np.r_[own,rk,zz,margin,USER_FEATURES[gi],I[m],full,explicit_pair_features(gi,m)]
    return F

F_train=build_candidate_features(expert_oof,0)
F_test=build_candidate_features(expert_test,N_TRAIN)
print("Candidate feature matrix:",F_train.shape,F_test.shape)
assert np.isfinite(F_train).all() and np.isfinite(F_test).all()

# ---- cell 14 (code) ----
# ---------- NDCG@5-aware candidate LambdaRank ----------
rank_oof=np.zeros_like(y,np.float32);rank_best_iters=[];nf=F_train.shape[-1]
label_gain=[0,1,3,7,15,31,63]
for f,(tr_idx,va_idx) in enumerate(folds):
    ranker=lgb.LGBMRanker(
        objective="lambdarank",metric="ndcg",eval_at=[5],lambdarank_truncation_level=8,
        label_gain=label_gain,n_estimators=1200,learning_rate=.02,num_leaves=15,
        min_child_samples=120,reg_alpha=1.2,reg_lambda=12.0,colsample_bytree=.75,
        random_state=SEED+700+f,n_jobs=-1,verbosity=-1,force_col_wise=True,
    )
    ranker.fit(
        F_train[tr_idx].reshape(-1,nf),Y_ORD[tr_idx].reshape(-1),group=[N_MODULES]*len(tr_idx),
        eval_set=[(F_train[va_idx].reshape(-1,nf),Y_ORD[va_idx].reshape(-1))],eval_group=[[N_MODULES]*len(va_idx)],
        callbacks=[lgb.early_stopping(90,verbose=False)],
    )
    it=int(ranker.best_iteration_ or 1200);rank_best_iters.append(it)
    rank_oof[va_idx]=ranker.predict(F_train[va_idx].reshape(-1,nf),num_iteration=it).reshape(len(va_idx),N_MODULES)
    print(f"rank fold {f+1}: best_iter={it}  NDCG@5={ndcg5(y[va_idx],rank_oof[va_idx]):.6f}")

final_iters=max(20,int(np.median(rank_best_iters)))
ranker=lgb.LGBMRanker(
    objective="lambdarank",metric="ndcg",eval_at=[5],lambdarank_truncation_level=8,
    label_gain=label_gain,n_estimators=final_iters,learning_rate=.02,num_leaves=15,
    min_child_samples=120,reg_alpha=1.2,reg_lambda=12.0,colsample_bytree=.75,
    random_state=SEED+1700,n_jobs=-1,verbosity=-1,force_col_wise=True,
)
ranker.fit(F_train.reshape(-1,nf),Y_ORD.reshape(-1),group=[N_MODULES]*N_TRAIN)
rank_test=ranker.predict(F_test.reshape(-1,nf)).reshape(len(test),N_MODULES).astype(np.float32)
print("LambdaRank OOF:",round(ndcg5(y,rank_oof),6),"final iterations:",final_iters)

# ---- cell 15 (code) ----
# ---------- Stability-gated multi-expert final ----------
# Topic gates are enabled only when the ranker beats the V3.1-like anchor in at
# least 3/5 OOF folds and the topic has enough support. This avoids global forcing.
anchor_pu=per_user_ndcg5(y,v31_anchor_oof);rank_pu=per_user_ndcg5(y,rank_oof)
train_topics=latest_primary_topic[:N_TRAIN]
stable_rank_topics=[]
for tp in sorted(set(train_topics)):
    ix=(train_topics==tp);n=int(ix.sum())
    if n<100: continue
    wins=[]
    for f in range(N_FOLDS):
        j=ix&(fold_id==f)
        if j.sum()>=10: wins.append(float((rank_pu[j]-anchor_pu[j]).mean()))
    overall=float((rank_pu[ix]-anchor_pu[ix]).mean())
    if overall>0 and sum(v>0 for v in wins)>=3:
        stable_rank_topics.append(tp)
    print(f"topic={tp:12s} n={n:4d} rank-anchor={overall:+.5f} positive_folds={sum(v>0 for v in wins)}/{len(wins)}")
print("Stable ranker topics:",stable_rank_topics)

def gated_scores(anchor,rank,msg,topics_global,learning_counts):
    s=anchor.copy()
    ix=np.isin(topics_global,stable_rank_topics)
    s[ix]=(1-RANK_GATE_WEIGHT)*anchor[ix]+RANK_GATE_WEIGHT*rank[ix]
    long=learning_counts>=LONG_HISTORY_MIN
    s[long]=(1-LONG_HISTORY_MSG_WEIGHT)*s[long]+LONG_HISTORY_MSG_WEIGHT*msg[long]
    return s

gated_oof=gated_scores(v31_anchor_oof,rank_oof,msg_oof,train_topics,general[:N_TRAIN,1])
gated_test=gated_scores(v31_anchor_test,rank_test,msg_test,latest_primary_topic[N_TRAIN:],general[N_TRAIN:,1])

# A second, pre-registered smooth mixture from the V1–V3.1 error analysis.
robust_mix_oof=.15*v31_anchor_oof+.55*rank_oof+.10*msg_oof+.20*cooc_oof
robust_mix_test=.15*v31_anchor_test+.55*rank_test+.10*msg_test+.20*cooc_test

candidates={
    "v31_anchor":(v31_anchor_oof,v31_anchor_test),
    "ranker":(rank_oof,rank_test),
    "gated":(gated_oof,gated_test),
    "robust_mix":(robust_mix_oof,robust_mix_test),
}
print(); print("V4 candidate OOF scores")
for k,(oo,_) in candidates.items():
    fold_sc=[ndcg5(y[fold_id==f],oo[fold_id==f]) for f in range(N_FOLDS)]
    print(f"{k:12s}: mean={ndcg5(y,oo):.6f} fold_std={np.std(fold_sc):.6f} folds={[round(x,6) for x in fold_sc]}")

# Prefer the highest mean only if its fold variability is not pathological.
quality={k:ndcg5(y,oo)-0.10*np.std([ndcg5(y[fold_id==f],oo[fold_id==f]) for f in range(N_FOLDS)]) for k,(oo,_) in candidates.items()}
primary_name=max(quality,key=quality.get)
primary_oof,primary_test=candidates[primary_name]
print("PRIMARY RECIPE:",primary_name,"OOF=",round(ndcg5(y,primary_oof),6))

# ---- cell 16 (code) ----
# ---------- Diagnostics: top1, true-top1 coverage, per-module recall ----------
def true_top1_in_top5(scores):
    top5=np.argsort(-scores,axis=1)[:,:5]
    return float(np.mean([TOP1[i] in top5[i] for i in range(N_TRAIN)]))

def module_recall5(scores):
    top5=np.argsort(-scores,axis=1)[:,:5];rows=[]
    for m,mid in enumerate(MODULES):
        ix=y[:,m]>0
        rec=float(np.mean([m in top5[i] for i in np.where(ix)[0]])) if ix.any() else np.nan
        rows.append((mid,int(ix.sum()),rec))
    return pd.DataFrame(rows,columns=["module_id","n_relevant","recall5"])

for name,(oo,_) in candidates.items():
    print(f"{name:12s}: top1_acc={(oo.argmax(1)==TOP1).mean():.4f} true_top1@5={true_top1_in_top5(oo):.4f}")
print(); print("Primary per-module recall@5")
print(module_recall5(primary_oof).round(4).to_string(index=False))

top5=np.argsort(-primary_test,axis=1)[:,:5]
freq=np.bincount(top5.ravel(),minlength=N_MODULES)
print(); print("Primary test top5 frequency")
print(pd.DataFrame({"module_id":MODULES,"count":freq}).to_string(index=False))

# ---- cell 17 (code) ----
# ---------- Submission writer + hard validation ----------
def write_submission(scores,filename):
    assert scores.shape==(len(test),N_MODULES)
    assert np.isfinite(scores).all()
    out=sample.copy();assert out["user_id"].tolist()==test["user_id"].tolist()
    out.loc[:,MODULES]=scores
    root=Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("/mnt/data")
    path=root/filename;out.to_csv(path,index=False)
    chk=pd.read_csv(path)
    assert chk.shape==sample.shape and chk["user_id"].is_unique and chk[MODULES].notna().all().all()
    print("Saved:",path)
    return path

primary_path=write_submission(primary_test,"submission_v4_primary.csv")
rank_path=write_submission(rank_test,"submission_v4_ranker.csv")
anchor_path=write_submission(v31_anchor_test,"submission_v4_v31_anchor.csv")
artifact_path=(Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("/mnt/data"))/"v4_oof_artifacts.npz"
np.savez_compressed(artifact_path,y=y,sparse_oof=sparse_oof,rel_oof=rel_oof,v31_anchor_oof=v31_anchor_oof,msg_oof=msg_oof,cooc_oof=cooc_oof,rank_oof=rank_oof,gated_oof=gated_oof,robust_mix_oof=robust_mix_oof,primary_oof=primary_oof,primary_test=primary_test,rank_test=rank_test,v31_anchor_test=v31_anchor_test)

print(); print("FINAL SUMMARY")
print("Sparse multi-expert :",round(ndcg5(y,sparse_oof),6))
print("V3.1-like anchor    :",round(ndcg5(y,v31_anchor_oof),6))
print("Message max         :",round(ndcg5(y,msg_oof),6))
print("Co-occurrence       :",round(ndcg5(y,cooc_oof),6))
print("LambdaRank          :",round(ndcg5(y,rank_oof),6))
print("Gated mixture       :",round(ndcg5(y,gated_oof),6))
print("Robust mixture      :",round(ndcg5(y,robust_mix_oof),6))
print("Primary             :",primary_name,round(ndcg5(y,primary_oof),6))
print("Recommended file to inspect first:",primary_path.name)
