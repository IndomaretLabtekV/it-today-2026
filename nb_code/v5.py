# Converted from notebooks/v5.ipynb: markdown cells become comments, %/! magics neutralized.

# ---- cell 0 (markdown) ----
# # MineToday V5 — Generator-Aware Categorical YetiRank
#
# **Goal:** rank 17 learning modules for every user under **NDCG@5**.
#
# V5 keeps the strongest lessons from V3.1/V4 and changes only the parts the diagnostics showed were still weak:
#
# - lexical memorization stays as the backbone (word + character TF‑IDF),
# - rank supervision is made **absolute** (`rank1=6 ... rank6=1`) instead of depending on the number of positives,
# - real timestamp gaps are modeled explicitly,
# - exact synthetic template identity / topic journey are exposed as categorical features,
# - the final candidate model is **CatBoost YetiRank**, with explicit `module × template/topic/time/skill` crosses,
# - final test inference is **5-fold bagged**, not a single median-iteration refit.
#
# No external data and no generative LLM prediction are used.

# ---- cell 1 (code) ----
import os, re, json, ast, html, hashlib, random, warnings, gc, time, sys, subprocess
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
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
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "lightgbm>=4.5,<5"])
    import lightgbm as lgb
try:
    from catboost import CatBoostRanker, Pool
    import catboost
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "catboost>=1.2.8,<2"])
    from catboost import CatBoostRanker, Pool
    import catboost

SEED = 42
N_FOLDS = 5
RIDGE_ALPHA = 3.0
REL_WEIGHT = 0.35
COOC_TEMP = 0.20
V5_SMOKE = os.getenv("MINETODAY_V5_SMOKE", "0") == "1"

random.seed(SEED)
np.random.seed(SEED)

# CatBoost ranking is the only GPU-dependent stage. A CPU fallback is built in.
def cuda_visible():
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
        return r.returncode == 0 and "GPU" in r.stdout
    except Exception:
        return False

USE_CATBOOST_GPU = cuda_visible() and not V5_SMOKE
print("numpy:", np.__version__)
print("pandas:", pd.__version__)
print("lightgbm:", lgb.__version__)
print("catboost:", catboost.__version__)
print("V5_SMOKE:", V5_SMOKE)
print("CatBoost GPU requested:", USE_CATBOOST_GPU)

# ---- cell 2 (code) ----
def locate_data_dir():
    required = {
        "chat_history.csv", "modules_catalog.csv", "user_assessments.csv",
        "train_relevance.csv", "test.csv", "sample_submission.csv"
    }
    roots = [
        Path("/kaggle/input"),
        Path("/mnt/data/_deep/data"),
        Path("/mnt/data/comp_v3"),
        Path("."),
    ]
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
    raise FileNotFoundError("MineToday dataset directory not found")

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
N_ALL = len(all_users)

print("DATA_DIR:", DATA_DIR)
print("chat/modules/assessment:", chat.shape, modules.shape, assessments.shape)
print("train/test:", train.shape, test.shape)
print("positive modules/user:", pd.Series((y > 0).sum(1)).value_counts().sort_index().to_dict())

# ---- cell 3 (code) ----
# ---------------- Metric + corrected ranking targets ----------------
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

def make_absolute_rank_targets(rel):
    # IMPORTANT V5 correction: rank identity no longer depends on how many
    # positives a user has. rank1 is always 6, rank2 always 5, ... rank6=1.
    out = np.zeros_like(rel, dtype=np.float32)
    for i, row in enumerate(rel):
        pos = np.where(row > 0)[0]
        order = pos[np.argsort(-row[pos])]
        for rank, j in enumerate(order):
            out[i, j] = max(1, 6 - rank)
    return out

Y_ORD = make_ordinal_targets(y)
Y_ABS = make_absolute_rank_targets(y)
Y_BIN = (y > 0).astype(np.float32)
TOP1 = np.argmax(y, axis=1)
Y_TOP = np.zeros_like(y, np.float32)
Y_TOP[np.arange(N_TRAIN), TOP1] = 1.0

# target invariants
assert np.all(Y_ABS[np.arange(N_TRAIN), TOP1] == 6)
assert set(np.unique(Y_ABS)).issubset(set(range(7)))


def make_multilabel_balanced_folds(y_bin, top1, n_splits=5, seed=42):
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
    assert sizes.tolist() == [target_size] * n_splits
    return [(np.where(fold_id != f)[0], np.where(fold_id == f)[0]) for f in range(n_splits)], fold_id

folds, fold_id = make_multilabel_balanced_folds(Y_BIN, TOP1, N_FOLDS, SEED)
print("Fold sizes:", [len(v) for _, v in folds])

# ---- cell 4 (code) ----
# ---------------- Robust assessment parsing ----------------
WORD_NUM = {"nol":0, "satu":1, "dua":2, "tiga":3, "empat":4, "lima":5}
SKILL_NAMES = ["python", "sql", "statistics", "eda", "ml", "ml_eval", "dl", "genai", "business", "research"]
SKILL_INDEX = {s:i for i,s in enumerate(SKILL_NAMES)}

A = np.zeros((N_ALL, len(SKILL_NAMES)), np.float32)
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
            try: value = float(v)
            except Exception: value = np.nan
            bad = int((not np.isfinite(value)) or value < 0 or value > 5)
            if bad: value = np.nan
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
# ---------------- Dataset-native topic + template parser ----------------
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

def hash_token(s, n=12):
    return hashlib.md5(str(s).encode("utf-8")).hexdigest()[:n]

chat["timestamp"] = pd.to_datetime(chat["timestamp"], errors="coerce")
chat["clean"] = chat["user_chat_text"].fillna("").map(clean_text)
chat["is_noise"] = chat["user_chat_text"].fillna("").map(lambda x: bool(NOISE_RX.search(str(x))))
chat["topics"] = chat["user_chat_text"].fillna("").map(extract_topics)
chat["skeleton"] = chat["user_chat_text"].fillna("").map(make_skeleton)
for name,rx in STYLE_RX.items():
    chat["sty_"+name] = chat["user_chat_text"].fillna("").map(lambda x,r=rx: int(bool(r.search(str(x)))))
chat = chat.sort_values(["user_id", "timestamp", "chat_id"]).reset_index(drop=True)
user_chat = {u:g.copy() for u,g in chat.groupby("user_id", sort=False)}
print("Administrative/noise rows:", int(chat.is_noise.sum()), "/", len(chat))

# ---- cell 6 (code) ----
# ---------------- V5 user views + true-time features ----------------
T, S = len(TOPICS), len(STYLES)
topic_count = np.zeros((N_ALL,T), np.float32)
topic_posrec = np.zeros_like(topic_count)
topic_latest = np.zeros_like(topic_count)
topic_lastpos = np.zeros_like(topic_count)
topic_age_days = np.full((N_ALL,T), 999.0, np.float32)
topic_decay3 = np.zeros_like(topic_count)
topic_decay14 = np.zeros_like(topic_count)
topic_decay30 = np.zeros_like(topic_count)
style_count = np.zeros((N_ALL,S), np.float32)
style_latest = np.zeros_like(style_count)
# total, learning, noise, span, latest_gap, mean_gap, median_gap, max_gap, unique_topics
general = np.zeros((N_ALL,9), np.float32)

full_texts, recent_texts, latest_texts = [], [], []
learning_rows = []
latest_topic_cat, last2_seq_cat, journey_cat = [], [], []
latest_tpl_cat, latest_style_cat, tpl_transition_cat = [], [], []

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
    latest_ts = learn["timestamp"].max() if n and learn["timestamp"].notna().any() else pd.NaT

    def render(selected_rows):
        parts=[]; nn=len(selected_rows)
        for j,row in enumerate(selected_rows):
            dist=nn-1-j
            tag="current" if dist==0 else ("recent" if dist<=2 else "older")
            parts += [f"[{tag}] {row.clean}", "tpl_"+hash_token(row.skeleton,10)]
            for tp in row.topics:
                parts.append("topic_"+tp)
                if dist==0: parts += ["latest_topic_"+tp] * 3
                elif dist<=2: parts += ["recent_topic_"+tp] * 2
        for si,skill in enumerate(SKILL_NAMES):
            v=int(round(float(A[ui,si])))
            band="low" if v<=1 else ("mid" if v<=3 else "high")
            parts += [f"skill_{skill}_{v}", f"skill_{skill}_bin_{band}"]
        return " ".join(parts)

    full_texts.append(render(rows))
    recent_texts.append(render(rows[-3:]))
    latest_texts.append(render(rows[-1:]))

    total=len(g); noise_count=int(g["is_noise"].sum()) if total else 0
    ts=learn["timestamp"].dropna().sort_values() if n else pd.Series([], dtype="datetime64[ns]")
    if len(ts)>=2:
        gap=np.diff(ts.values).astype("timedelta64[s]").astype(float)/86400.0
        span=float((ts.iloc[-1]-ts.iloc[0]).total_seconds()/86400.0)
        latest_gap=float(gap[-1]); mean_gap=float(gap.mean()); median_gap=float(np.median(gap)); max_gap=float(gap.max())
    else:
        gap=np.array([]); span=latest_gap=mean_gap=median_gap=max_gap=0.0
    uniq_topics=set(tp for row in rows for tp in row.topics)
    general[ui]=[total,n,noise_count,span,latest_gap,mean_gap,median_gap,max_gap,len(uniq_topics)]

    for si,sty in enumerate(STYLES):
        style_count[ui,si]=sum(getattr(r,"sty_"+sty) for r in rows)
        style_latest[ui,si]=getattr(rows[-1],"sty_"+sty) if n else 0

    for j,row in enumerate(rows):
        for tp in row.topics:
            ti=TOPIC_INDEX[tp]
            topic_count[ui,ti]+=1
            topic_posrec[ui,ti]+=float(np.exp(-0.7*(n-1-j)))
            topic_lastpos[ui,ti]=max(topic_lastpos[ui,ti], (j+1)/max(n,1))
            if j==n-1: topic_latest[ui,ti]=1
            if pd.notna(latest_ts) and pd.notna(row.timestamp):
                age=max(0.0,float((latest_ts-row.timestamp).total_seconds()/86400.0))
                topic_age_days[ui,ti]=min(topic_age_days[ui,ti],age)
        learning_rows.append((ui,row.clean,row.skeleton,row.topics))

    valid = topic_age_days[ui] < 999
    topic_decay3[ui,valid] = np.exp(-np.minimum(topic_age_days[ui,valid],60)/3.0)
    topic_decay14[ui,valid] = np.exp(-np.minimum(topic_age_days[ui,valid],120)/14.0)
    topic_decay30[ui,valid] = np.exp(-np.minimum(topic_age_days[ui,valid],180)/30.0)

    lt = rows[-1].topics[0] if n and rows[-1].topics else "none"
    latest_topic_cat.append(lt)
    seq=[]
    for r in rows[-2:]: seq.append("+".join(r.topics[:2]) if r.topics else "none")
    last2_seq_cat.append(">".join(seq) if seq else "none")
    journey=[]
    for r in rows:
        for tp in r.topics:
            if not journey or journey[-1] != tp: journey.append(tp)
    journey_cat.append(">".join(journey[-5:]) if journey else "none")
    latest_tpl_cat.append(hash_token(rows[-1].skeleton) if n else "none")
    st=[sty for sty in STYLES if n and getattr(rows[-1],"sty_"+sty)]
    latest_style_cat.append("+".join(st) if st else "none")
    if n>=2:
        tpl_transition_cat.append(hash_token(rows[-2].skeleton,8)+">"+hash_token(rows[-1].skeleton,8))
    elif n==1:
        tpl_transition_cat.append("start>"+hash_token(rows[-1].skeleton,8))
    else:
        tpl_transition_cat.append("none")

full_texts=np.asarray(full_texts,dtype=object)
recent_texts=np.asarray(recent_texts,dtype=object)
latest_texts=np.asarray(latest_texts,dtype=object)
latest_topic_cat=np.asarray(latest_topic_cat,dtype=object)
last2_seq_cat=np.asarray(last2_seq_cat,dtype=object)
journey_cat=np.asarray(journey_cat,dtype=object)
latest_tpl_cat=np.asarray(latest_tpl_cat,dtype=object)
latest_style_cat=np.asarray(latest_style_cat,dtype=object)
tpl_transition_cat=np.asarray(tpl_transition_cat,dtype=object)

USER_NUM = np.column_stack([
    A, A_INVALID, general, style_count, style_latest,
    topic_count, topic_posrec, topic_latest, topic_lastpos,
    np.clip(topic_age_days,0,365), topic_decay3, topic_decay14, topic_decay30,
]).astype(np.float32)
print("USER_NUM:", USER_NUM.shape)
print("latest topic train:", pd.Series(latest_topic_cat[:N_TRAIN]).value_counts().to_dict())
assert USER_NUM.shape[0] == N_ALL and np.isfinite(USER_NUM).all()

# ---- cell 7 (code) ----
# ---------------- TF-IDF views ----------------
t0=time.time()
def make_view(texts, word_features, char_features):
    w=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_df=0.995,sublinear_tf=True,strip_accents="unicode",max_features=word_features,dtype=np.float32)
    c=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=2,sublinear_tf=True,max_features=char_features,dtype=np.float32)
    return hstack([w.fit_transform(texts),c.fit_transform(texts)],format="csr"), (w,c)

X_full, vec_full = make_view(full_texts, 50_000, 60_000)
X_recent, vec_recent = make_view(recent_texts, 35_000, 40_000)
X_latest, vec_latest = make_view(latest_texts, 30_000, 35_000)
print("X_full/recent/latest:", X_full.shape, X_recent.shape, X_latest.shape)
print("vectorization seconds:", round(time.time()-t0,1))

# ---- cell 8 (code) ----
# ---------------- Seven sparse supervised experts ----------------
head_specs = {
    "raw_all": (y, X_full, 3.0),
    "abs_all": (Y_ABS/6.0, X_full, 3.0),
    "ord_all": (Y_ORD/6.0, X_full, 3.0),
    "top1_all": (Y_TOP, X_full, 3.0),
    "bin_all": (Y_BIN, X_full, 3.0),
    "abs_recent": (Y_ABS/6.0, X_recent, 4.0),
    "raw_latest": (y, X_latest, 4.0),
}
oof_heads={}; test_heads={}
for name,(target,matrix,alpha) in head_specs.items():
    oof=np.zeros_like(y,np.float32);pred=np.zeros((len(test),N_MODULES),np.float32)
    if V5_SMOKE:
        # Structural smoke path only: bypass expensive fitting while preserving
        # exact tensor shapes required by every downstream component.
        oof[:]=target.astype(np.float32)
        pred[:]=np.tile(target.mean(0),(len(test),1)).astype(np.float32)
    else:
        for f,(tr_idx,va_idx) in enumerate(folds):
            model=Ridge(alpha=alpha,solver="lsqr",tol=2e-3,max_iter=1000)
            model.fit(matrix[tr_idx],target[tr_idx])
            oof[va_idx]=model.predict(matrix[va_idx]).astype(np.float32)
            pred += model.predict(matrix[N_TRAIN:]).astype(np.float32)/N_FOLDS
    oof_heads[name]=oof;test_heads[name]=pred
    print(f"{name:12s}: OOF NDCG@5={ndcg5(y,oof):.6f}")

# ---- cell 9 (code) ----
# ---------------- Fold-stable sparse blend ----------------
HEAD_NAMES=list(head_specs)
H_oof=np.stack([oof_heads[k] for k in HEAD_NAMES])
H_test=np.stack([test_heads[k] for k in HEAD_NAMES])
rng=np.random.default_rng(SEED)
base_candidates=[
    np.array([.78,.08,.00,.10,.00,.04,.00],float),
    np.array([.58,.12,.06,.15,.03,.04,.02],float),
    np.array([.52,.16,.06,.15,.03,.05,.03],float),
]
for _ in range(550):
    w=rng.dirichlet([9,2.5,1.5,2,1.2,2,1.2])
    if w[0]>=.38 and w[3]<=.25 and w[6]<=.15:
        base_candidates.append(w)

selected=[];sparse_oof=np.zeros_like(y)
if V5_SMOKE:
    stable_w=base_candidates[0].copy();stable_w/=stable_w.sum()
    sparse_oof=np.tensordot(stable_w,H_oof,axes=(0,0)).astype(np.float32)
else:
    for held in range(N_FOLDS):
        train_folds=[f for f in range(N_FOLDS) if f!=held]
        best=(-1e9,None)
        for w in base_candidates:
            fs=[]
            for f in train_folds:
                ix=(fold_id==f)
                fs.append(ndcg5(y[ix],np.tensordot(w,H_oof[:,ix,:],axes=(0,0))))
            obj=float(np.mean(fs)-0.15*np.std(fs))
            if obj>best[0]: best=(obj,w)
        selected.append(best[1])
        ix=(fold_id==held)
        sparse_oof[ix]=np.tensordot(best[1],H_oof[:,ix,:],axes=(0,0))
    stable_w=np.mean(selected,axis=0);stable_w/=stable_w.sum()
sparse_test=np.tensordot(stable_w,H_test,axes=(0,0)).astype(np.float32)
print("Fold-stable sparse weights:", {k:round(float(v),4) for k,v in zip(HEAD_NAMES,stable_w)})
print("Sparse V5 OOF:", round(ndcg5(y,sparse_oof),6))

# ---- cell 10 (code) ----
# ---------------- Proven V3.1-style nonlinear relevance expert ----------------
rel_oof=np.zeros_like(y,np.float32)
rel_test=np.zeros((len(test),N_MODULES),np.float32)
head_oof_cols=np.stack([oof_heads[k] for k in HEAD_NAMES],axis=2)
head_test_cols=np.stack([test_heads[k] for k in HEAD_NAMES],axis=2)
best_iters={mid:[] for mid in MODULES}

for m,mid in enumerate(MODULES):
    Xm=np.column_stack([sparse_oof[:,m],head_oof_cols[:,m,:],USER_NUM[:N_TRAIN]])
    Xt=np.column_stack([sparse_test[:,m],head_test_cols[:,m,:],USER_NUM[N_TRAIN:]])
    target=Y_BIN[:,m].astype(np.int32)
    if V5_SMOKE:
        clf=lgb.LGBMClassifier(n_estimators=25,learning_rate=.06,num_leaves=9,min_child_samples=55,reg_alpha=.6,reg_lambda=6.0,colsample_bytree=.72,random_state=SEED+31*m,n_jobs=-1,verbosity=-1,force_col_wise=True)
        rel_oof[:,m]=target.astype(np.float32)
        rel_test[:,m]=float(target.mean())
    else:
        for f,(tr_idx,va_idx) in enumerate(folds):
            clf=lgb.LGBMClassifier(
                n_estimators=900,learning_rate=.03,num_leaves=15,min_child_samples=55,
                reg_alpha=.6,reg_lambda=6.0,colsample_bytree=.72,
                random_state=SEED+31*m+f,n_jobs=-1,verbosity=-1,force_col_wise=True,
            )
            clf.fit(Xm[tr_idx],target[tr_idx],eval_set=[(Xm[va_idx],target[va_idx])],callbacks=[lgb.early_stopping(65,verbose=False)])
            it=int(clf.best_iteration_ or 900);best_iters[mid].append(it)
            rel_oof[va_idx,m]=clf.predict_proba(Xm[va_idx],num_iteration=it)[:,1]
        final_it=max(20,int(np.median(best_iters[mid])))
        clf=lgb.LGBMClassifier(
            n_estimators=final_it,learning_rate=.03,num_leaves=15,min_child_samples=55,
            reg_alpha=.6,reg_lambda=6.0,colsample_bytree=.72,
            random_state=SEED+997+31*m,n_jobs=-1,verbosity=-1,force_col_wise=True,
        )
        clf.fit(Xm,target)
        rel_test[:,m]=clf.predict_proba(Xt)[:,1]

anchor_oof=(1-REL_WEIGHT)*sparse_oof+REL_WEIGHT*rel_oof
anchor_test=(1-REL_WEIGHT)*sparse_test+REL_WEIGHT*rel_test
print("Relevance OOF:", round(ndcg5(y,rel_oof),6))
print("Structured anchor OOF:", round(ndcg5(y,anchor_oof),6))

# ---- cell 11 (code) ----
# ---------------- Cheap diversity experts: message max + co-occurrence ----------------
if V5_SMOKE:
    msg_oof=sparse_oof.copy();msg_test=sparse_test.copy()
else:
    msg_df=pd.DataFrame(learning_rows,columns=["ui","clean","skeleton","topics"])
    def augment_message(r):
        return " ".join([r.clean,"tpl_"+hash_token(r.skeleton,10)] + ["topic_"+t for t in r.topics])
    msg_text=msg_df.apply(augment_message,axis=1).to_numpy(dtype=object)
    msg_w=TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True,max_features=40_000,dtype=np.float32)
    msg_c=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=2,sublinear_tf=True,max_features=50_000,dtype=np.float32)
    X_msg=hstack([msg_w.fit_transform(msg_text),msg_c.fit_transform(msg_text)],format="csr")
    msg_ui=msg_df.ui.to_numpy(int)
    msg_oof=np.zeros_like(y,np.float32);msg_test=np.zeros((len(test),N_MODULES),np.float32)
    for f,(tr_idx,va_idx) in enumerate(folds):
        tr_rows=np.where(np.isin(msg_ui,tr_idx))[0]
        model=Ridge(alpha=5.0,solver="lsqr",tol=2e-3,max_iter=1000).fit(X_msg[tr_rows],y[msg_ui[tr_rows]])
        prior=y[tr_idx].mean(0)
        msg_oof[va_idx]=prior
        vr=np.where(np.isin(msg_ui,va_idx))[0]
        if len(vr):
            P=model.predict(X_msg[vr]).astype(np.float32);vr_ui=msg_ui[vr]
            for ui in va_idx:
                loc=np.where(vr_ui==ui)[0]
                if len(loc): msg_oof[ui]=P[loc].max(0)
        fold_test=np.tile(prior,(len(test),1)).astype(np.float32)
        rr=np.where(msg_ui>=N_TRAIN)[0]
        if len(rr):
            P=model.predict(X_msg[rr]).astype(np.float32);rr_ui=msg_ui[rr]
            for j in range(len(test)):
                loc=np.where(rr_ui==(N_TRAIN+j))[0]
                if len(loc): fold_test[j]=P[loc].max(0)
        msg_test += fold_test/N_FOLDS

def softmax_rows(x,temp):
    z=x/temp;z=z-z.max(1,keepdims=True);e=np.exp(z);return e/e.sum(1,keepdims=True)
def transition_matrix(indices,alpha=8.0):
    prior=y[indices].mean(0);top=np.argmax(y[indices],axis=1);Tmat=np.zeros((N_MODULES,N_MODULES),np.float32)
    for m in range(N_MODULES):
        ix=indices[top==m]
        Tmat[m]=(y[ix].sum(0)+alpha*prior)/(len(ix)+alpha) if len(ix) else prior
    return Tmat
cooc_oof=np.zeros_like(y,np.float32)
for tr_idx,va_idx in folds:
    cooc_oof[va_idx]=softmax_rows(anchor_oof[va_idx],COOC_TEMP)@transition_matrix(tr_idx)
cooc_test=softmax_rows(anchor_test,COOC_TEMP)@transition_matrix(np.arange(N_TRAIN))
print("Message max OOF:", round(ndcg5(y,msg_oof),6))
print("Co-occurrence OOF:", round(ndcg5(y,cooc_oof),6))

# ---- cell 12 (code) ----
# ---------------- Vectorized candidate features: generator-aware interactions ----------------
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

expert_oof={"sparse":sparse_oof,"rel":rel_oof,"anchor":anchor_oof,"msg":msg_oof,"cooc":cooc_oof,**oof_heads}
expert_test={"sparse":sparse_test,"rel":rel_test,"anchor":anchor_test,"msg":msg_test,"cooc":cooc_test,**test_heads}
EXPERT_NAMES=list(expert_oof)

def row_ranks(a): return np.argsort(np.argsort(-a,axis=1),axis=1).astype(np.float32)
def zscore_rows(a): return ((a-a.mean(1,keepdims=True))/(a.std(1,keepdims=True)+1e-6)).astype(np.float32)
def age_bin_vec(x):
    x=np.asarray(x)
    out=np.full(x.shape,"d60plus",dtype=object)
    out[x>=999]="never";out[x<=60]="d14_60";out[x<=14]="d3_14";out[x<=3]="d0_3";out[x<=.01]="current"
    return out

def pair_matrices():
    shape=(N_ALL,N_MODULES)
    latest=np.zeros(shape,np.float32);count=np.zeros(shape,np.float32);posrec=np.zeros(shape,np.float32);lastpos=np.zeros(shape,np.float32)
    age=np.full(shape,999.,np.float32);d3=np.zeros(shape,np.float32);d14=np.zeros(shape,np.float32);d30=np.zeros(shape,np.float32)
    pmin=np.zeros(shape,np.float32);pmean=np.zeros(shape,np.float32);pready=np.zeros(shape,np.float32);course=np.zeros(shape,np.float32);gap=np.zeros(shape,np.float32)
    for m,mid in enumerate(MODULES):
        tis=[TOPIC_INDEX[t] for t in MODULE_TOPIC_MAP[mid] if t in TOPIC_INDEX]
        if tis:
            latest[:,m]=topic_latest[:,tis].max(1);count[:,m]=topic_count[:,tis].max(1);posrec[:,m]=topic_posrec[:,tis].max(1);lastpos[:,m]=topic_lastpos[:,tis].max(1)
            age[:,m]=topic_age_days[:,tis].min(1);d3[:,m]=topic_decay3[:,tis].max(1);d14[:,m]=topic_decay14[:,tis].max(1);d30[:,m]=topic_decay30[:,tis].max(1)
        ps=PREREQ_SKILLS.get(mid,[])
        if ps:
            vals=A[:,[SKILL_INDEX[s] for s in ps]]
            pmin[:,m]=vals.min(1);pmean[:,m]=vals.mean(1);pready[:,m]=(vals.min(1)>=3).astype(np.float32)
        cs=COURSE_SKILL.get(mid)
        course[:,m]=A[:,SKILL_INDEX[cs]] if cs else A.mean(1)
        gap[:,m]=5.0-course[:,m]
    return dict(pair_latest=latest,pair_count=count,pair_posrec=posrec,pair_lastpos=lastpos,pair_age=age,pair_decay3=d3,pair_decay14=d14,pair_decay30=d30,pre_min=pmin,pre_mean=pmean,pre_ready=pready,course_skill=course,skill_gap=gap)
PAIR=pair_matrices()

def build_candidate_frame(experts,user_offset=0):
    n=next(iter(experts.values())).shape[0]
    user_ix=np.arange(user_offset,user_offset+n)
    data={}
    for name,a in experts.items():
        r=row_ranks(a);z=zscore_rows(a);mx=a.max(1,keepdims=True);top2=np.partition(a,-2,axis=1)[:,-2:];topgap=(top2[:,1]-top2[:,0]).astype(np.float32)
        data[name]=a.reshape(-1).astype(np.float32)
        data[name+"_rank"]=r.reshape(-1)
        data[name+"_z"]=z.reshape(-1)
        data[name+"_margin"]=(a-mx).reshape(-1).astype(np.float32)
        data[name+"_topgap"]=np.repeat(topgap,N_MODULES)
    # user numeric context
    user_core=np.column_stack([A,A_INVALID,general,style_count,style_latest]).astype(np.float32)
    for j in range(user_core.shape[1]): data[f"user_{j}"]=np.repeat(user_core[user_ix,j],N_MODULES)
    # pair numeric context
    for k,v in PAIR.items(): data[k]=v[user_ix].reshape(-1).astype(np.float32)

    module=np.tile(np.asarray(MODULES,dtype=object),n)
    lt=np.repeat(latest_topic_cat[user_ix],N_MODULES);tpl=np.repeat(latest_tpl_cat[user_ix],N_MODULES);seq=np.repeat(last2_seq_cat[user_ix],N_MODULES);jour=np.repeat(journey_cat[user_ix],N_MODULES)
    sty=np.repeat(latest_style_cat[user_ix],N_MODULES);trn=np.repeat(tpl_transition_cat[user_ix],N_MODULES)
    ageb=age_bin_vec(PAIR["pair_age"][user_ix]).reshape(-1)
    cur=PAIR["course_skill"][user_ix].reshape(-1);pmin=PAIR["pre_min"][user_ix].reshape(-1)
    skillb=np.where(cur<=1,"low",np.where(cur<=3,"mid","high"));preb=np.where(pmin<=1,"low",np.where(pmin<=3,"mid","high"))
    data.update(
        module=module,latest_topic=lt,latest_tpl=tpl,last2_seq=seq,journey=jour,
        module_topic=np.char.add(np.char.add(module.astype(str),"|"),lt.astype(str)),
        module_tpl=np.char.add(np.char.add(module.astype(str),"|"),tpl.astype(str)),
        module_seq=np.char.add(np.char.add(module.astype(str),"|"),seq.astype(str)),
        module_agebin=np.char.add(np.char.add(module.astype(str),"|"),ageb.astype(str)),
        module_skillband=np.char.add(np.char.add(module.astype(str),"|"),skillb.astype(str)),
    )
    df=pd.DataFrame(data)
    for c in df.columns:
        if c not in CAT_COLS: df[c]=df[c].astype(np.float32)
    return df

CAT_COLS=["module","latest_topic","latest_tpl","last2_seq","journey","module_topic","module_tpl","module_seq","module_agebin","module_skillband"]
F_train=build_candidate_frame(expert_oof,0)
F_test=build_candidate_frame(expert_test,N_TRAIN)
assert len(F_train)==N_TRAIN*N_MODULES and len(F_test)==len(test)*N_MODULES
for c in CAT_COLS:
    F_train[c]=F_train[c].fillna("none").astype(str)
    F_test[c]=F_test[c].fillna("none").astype(str)
num_cols=[c for c in F_train.columns if c not in CAT_COLS]
assert np.isfinite(F_train[num_cols].to_numpy(np.float32)).all()
assert np.isfinite(F_test[num_cols].to_numpy(np.float32)).all()
print("Candidate frames:",F_train.shape,F_test.shape,"categorical:",len(CAT_COLS))

# ---- cell 13 (code) ----
# ---------------- Fold-bagged categorical YetiRank ----------------
def candidate_rows(user_indices):
    user_indices=np.asarray(user_indices,dtype=int)
    return (user_indices[:,None]*N_MODULES + np.arange(N_MODULES)[None,:]).reshape(-1)

def make_ranker(seed, use_gpu):
    params=dict(
        loss_function="YetiRank",
        eval_metric="NDCG:top=5",
        iterations=20 if V5_SMOKE else 120,
        learning_rate=.06 if V5_SMOKE else .045,
        depth=5,
        l2_leaf_reg=10.0,
        random_strength=.35,
        random_seed=seed,
        one_hot_max_size=32,
        max_ctr_complexity=1,
        od_type="Iter",
        od_wait=10 if V5_SMOKE else 30,
        allow_writing_files=False,
        verbose=False,
    )
    if use_gpu:
        params.update(task_type="GPU", devices="0")
    else:
        params.update(task_type="CPU", thread_count=4)
    return CatBoostRanker(**params)

rank_oof=anchor_oof.copy() if V5_SMOKE else np.zeros_like(y,np.float32)
rank_test=np.zeros((len(test),N_MODULES),np.float32)
rank_best_iters=[]
fold_rank_scores=[]
fold_anchor_scores=[]
folds_to_run=[(np.arange(0,600),np.arange(600,700))] if V5_SMOKE else folds

for f,(tr_idx,va_idx) in enumerate(folds_to_run):
    tr_rows=candidate_rows(tr_idx);va_rows=candidate_rows(va_idx)
    train_pool=Pool(F_train.iloc[tr_rows],label=Y_ABS[tr_idx].reshape(-1),group_id=np.repeat(np.arange(len(tr_idx)),N_MODULES),cat_features=CAT_COLS)
    valid_pool=Pool(F_train.iloc[va_rows],label=Y_ABS[va_idx].reshape(-1),group_id=np.repeat(np.arange(len(va_idx)),N_MODULES),cat_features=CAT_COLS)
    test_pool=Pool(F_test,group_id=np.repeat(np.arange(len(test)),N_MODULES),cat_features=CAT_COLS)
    try:
        ranker=make_ranker(SEED+800+f,USE_CATBOOST_GPU)
        ranker.fit(train_pool,eval_set=valid_pool,use_best_model=True,verbose=False)
    except Exception as e:
        if not USE_CATBOOST_GPU:
            raise
        print("GPU CatBoost failed on fold",f+1,"-> deterministic CPU fallback:",type(e).__name__)
        ranker=make_ranker(SEED+800+f,False)
        ranker.fit(train_pool,eval_set=valid_pool,use_best_model=True,verbose=False)
    it=max(1,int(ranker.get_best_iteration()+1));rank_best_iters.append(it)
    pva=ranker.predict(valid_pool).reshape(len(va_idx),N_MODULES).astype(np.float32)
    pte=ranker.predict(test_pool).reshape(len(test),N_MODULES).astype(np.float32)
    rank_oof[va_idx]=pva
    rank_test += pte/len(folds_to_run)
    rs=ndcg5(y[va_idx],pva);ascore=ndcg5(y[va_idx],anchor_oof[va_idx])
    fold_rank_scores.append(rs);fold_anchor_scores.append(ascore)
    print(f"YetiRank fold {f+1}: iter={it:3d} NDCG@5={rs:.6f} anchor={ascore:.6f} delta={rs-ascore:+.6f}")
    del ranker,train_pool,valid_pool,test_pool;gc.collect()

if V5_SMOKE:
    print("SMOKE MODE: only fold 1 is trained; metrics are not competition estimates.")
else:
    print("YetiRank OOF:",round(ndcg5(y,rank_oof),6),"best iterations:",rank_best_iters)

# ---- cell 14 (code) ----
# ---------------- Conservative candidate selection ----------------
# A small pre-registered blend family; no dense OOF weight search.
blend90_oof=.90*rank_oof+.10*anchor_oof
blend90_test=.90*rank_test+.10*anchor_test
blend85_oof=.85*rank_oof+.15*anchor_oof
blend85_test=.85*rank_test+.15*anchor_test

candidates={
    "anchor":(anchor_oof,anchor_test),
    "yetirank":(rank_oof,rank_test),
    "blend90":(blend90_oof,blend90_test),
    "blend85":(blend85_oof,blend85_test),
}

def fold_scores(scores):
    return [ndcg5(y[fold_id==f],scores[fold_id==f]) for f in range(N_FOLDS)]

print("V5 candidate scores")
quality={}
for k,(oo,_) in candidates.items():
    fs=fold_scores(oo)
    sc=ndcg5(y,oo)
    quality[k]=sc-.10*np.std(fs)
    print(f"{k:10s}: mean={sc:.6f} std={np.std(fs):.6f} folds={[round(x,6) for x in fs]}")

if V5_SMOKE:
    primary_name="anchor"
else:
    primary_name=max(quality,key=quality.get)
primary_oof,primary_test=candidates[primary_name]
print("PRIMARY:",primary_name,"OOF",round(ndcg5(y,primary_oof),6))

if not V5_SMOKE:
    rank_wins=sum(r>a for r,a in zip(fold_rank_scores,fold_anchor_scores))
    gate_score=ndcg5(y,primary_oof)>=0.667
    gate_folds=rank_wins>=4
    print("Submission gate >=0.667:",gate_score)
    print("YetiRank beats anchor in >=4/5 folds:",gate_folds,"(",rank_wins,"/5 )")
    print("GATE:","PASS" if gate_score and gate_folds else "REVIEW")

# ---- cell 15 (code) ----
# ---------------- Diagnostics ----------------
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
    print(f"{name:10s}: top1_acc={(oo.argmax(1)==TOP1).mean():.4f} true_top1@5={true_top1_in_top5(oo):.4f}")
print("\nPrimary per-module recall@5")
print(module_recall5(primary_oof).round(4).to_string(index=False))

top5=np.argsort(-primary_test,axis=1)[:,:5]
freq=np.bincount(top5.ravel(),minlength=N_MODULES)
print("\nPrimary test Top-5 frequency")
print(pd.DataFrame({"module_id":MODULES,"count":freq}).to_string(index=False))

# ID-tail diagnostic uses the same OOF predictions; it is a drift/subgroup check,
# not an independent holdout and is labelled accordingly.
tail=np.arange(3200,4000)
print("\nID-tail OOF diagnostic (U_3201..U_4000; not independent):")
for name,(oo,_) in candidates.items():
    print(name,round(ndcg5(y[tail],oo[tail]),6))

# ---- cell 16 (code) ----
# ---------------- Submission writer + hard validation ----------------
def write_submission(scores,filename):
    assert scores.shape==(len(test),N_MODULES)
    assert np.isfinite(scores).all()
    out=sample.copy()
    assert out["user_id"].tolist()==test["user_id"].tolist()
    out.loc[:,MODULES]=scores
    root=Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("/mnt/data")
    path=root/filename
    out.to_csv(path,index=False)
    chk=pd.read_csv(path)
    assert chk.shape==sample.shape
    assert chk["user_id"].is_unique
    assert chk[MODULES].notna().all().all()
    assert np.isfinite(chk[MODULES].to_numpy()).all()
    print("Saved:",path)
    return path

primary_path=write_submission(primary_test,"submission_v5_primary.csv")
yeti_path=write_submission(rank_test,"submission_v5_yetirank.csv")
anchor_path=write_submission(anchor_test,"submission_v5_anchor.csv")
blend_path=write_submission(blend90_test,"submission_v5_blend90.csv")

root=Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("/mnt/data")
artifact_path=root/"v5_oof_artifacts.npz"
np.savez_compressed(
    artifact_path,y=y,fold_id=fold_id,sparse_oof=sparse_oof,rel_oof=rel_oof,anchor_oof=anchor_oof,
    rank_oof=rank_oof,blend90_oof=blend90_oof,blend85_oof=blend85_oof,primary_oof=primary_oof,
    sparse_test=sparse_test,anchor_test=anchor_test,rank_test=rank_test,primary_test=primary_test,
)

print("\nFINAL SUMMARY")
print("Sparse V5        :",round(ndcg5(y,sparse_oof),6))
print("Structured anchor:",round(ndcg5(y,anchor_oof),6))
print("Categorical Yeti :",round(ndcg5(y,rank_oof),6))
print("Blend 90/10      :",round(ndcg5(y,blend90_oof),6))
print("Blend 85/15      :",round(ndcg5(y,blend85_oof),6))
print("Primary          :",primary_name,round(ndcg5(y,primary_oof),6))
print("Primary file     :",primary_path.name)
