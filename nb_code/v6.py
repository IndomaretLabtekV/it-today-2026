# Converted from notebooks/v6.ipynb: markdown cells become comments, %/! magics neutralized.

# ---- cell 0 (markdown) ----
# # MineToday V6 — Nested Ordinal Multi-Expert NDCG Stack
#
# **Competition target:** NDCG@5 over 17 fixed course candidates.
#
# V6 is deliberately built around the strongest findings from V1–V5:
#
# - preserve the high-performing lexical sparse backbone;
# - model current intent, full history, real timestamp decay, assessment readiness, and template identity separately;
# - use fold-safe target-memory features for synthetic template structure;
# - add dedicated **relevance**, **top-1**, and **absolute-rank distribution** experts;
# - train the final ranker with a metric-aligned CatBoost YetiRank NDCG@5 objective;
# - evaluate with **outer 5-fold nested stacking**, so every outer validation row is untouched by all supervised feature-generation steps;
# - average the five outer-fold test predictions (no median-iteration refit mismatch).
#
# The notebook writes several candidates, but **do not submit automatically**. Use the printed final gate and diagnostics first.

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
N_OUTER = 5
N_INNER = 4
V6_SMOKE = os.getenv("MINETODAY_V6_SMOKE", "0") == "1"

random.seed(SEED)
np.random.seed(SEED)

print("numpy:", np.__version__)
print("pandas:", pd.__version__)
print("lightgbm:", lgb.__version__)
print("catboost:", catboost.__version__)
print("V6_SMOKE:", V6_SMOKE)

# ---- cell 2 (code) ----
def locate_data_dir():
    required = {
        "chat_history.csv", "modules_catalog.csv", "user_assessments.csv",
        "train_relevance.csv", "test.csv", "sample_submission.csv"
    }
    roots = [Path("/kaggle/input"), Path("/mnt/data/comp_v3"), Path("/mnt/data"), Path(".")]
    for root in roots:
        if not root.exists():
            continue
        candidates = [root]
        if root.name in {"input", "data"}:
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
N_TEST = len(test)
N_ALL = len(all_users)

print("DATA_DIR:", DATA_DIR)
print("chat/modules/assessment:", chat.shape, modules.shape, assessments.shape)
print("train/test:", train.shape, test.shape)
print("positive modules/user:", pd.Series((y > 0).sum(1)).value_counts().sort_index().to_dict())

# ---- cell 3 (code) ----
# ---------------- Metric, targets, and deterministic multilabel folds ----------------
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

def make_absolute_rank_targets(rel):
    out = np.zeros_like(rel, dtype=np.int8)
    for i, row in enumerate(rel):
        pos = np.where(row > 0)[0]
        order = pos[np.argsort(-row[pos])]
        for rank, j in enumerate(order):
            out[i, j] = max(1, 6-rank)
    return out

Y_ABS = make_absolute_rank_targets(y)
Y_BIN = (y > 0).astype(np.int8)
TOP1 = np.argmax(y, axis=1)
Y_TOP = np.zeros_like(Y_BIN, np.int8)
Y_TOP[np.arange(N_TRAIN), TOP1] = 1
# Fixed rank gains approximate the competition relevance bands while removing within-band jitter.
GAIN_BY_CLASS = np.asarray([0.00, 0.25, 0.40, 0.55, 0.70, 0.85, 1.00], np.float32)
Y_GAIN = GAIN_BY_CLASS[Y_ABS]
assert np.all(Y_ABS[np.arange(N_TRAIN), TOP1] == 6)

def balanced_fold_ids(indices, n_splits, seed):
    indices = np.asarray(indices, dtype=int)
    labels = np.concatenate([
        Y_BIN[indices].astype(np.int8),
        np.eye(N_MODULES, dtype=np.int8)[TOP1[indices]]
    ], axis=1)
    n = len(indices)
    target_sizes = np.full(n_splits, n // n_splits, int)
    target_sizes[:n % n_splits] += 1
    desired = labels.sum(0).astype(float) / n_splits
    current = np.zeros((n_splits, labels.shape[1]), float)
    sizes = np.zeros(n_splits, int)
    freq = np.maximum(labels.sum(0), 1)
    rng = np.random.default_rng(seed)
    rarity = (labels / freq).sum(1)
    order = np.argsort(-(rarity + rng.random(n) * 1e-9))
    fid = np.full(n, -1, int)
    for loc in order:
        active = np.flatnonzero(labels[loc])
        score = np.full(n_splits, -1e12, float)
        for f in range(n_splits):
            if sizes[f] >= target_sizes[f]:
                continue
            label_need = ((desired[active] - current[f, active]) / np.maximum(desired[active], 1)).sum()
            size_need = (target_sizes[f] - sizes[f]) / max(target_sizes[f], 1)
            score[f] = label_need + 0.45 * size_need
        best = np.flatnonzero(score == score.max())
        f = int(rng.choice(best))
        fid[loc] = f
        sizes[f] += 1
        current[f] += labels[loc]
    assert np.all(fid >= 0)
    return fid

outer_indices = np.arange(N_TRAIN)
outer_fid = balanced_fold_ids(outer_indices, N_OUTER, SEED)
outer_folds = [(np.where(outer_fid != f)[0], np.where(outer_fid == f)[0]) for f in range(N_OUTER)]
print("Outer fold sizes:", [len(v) for _,v in outer_folds])

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

assert np.isfinite(A).all()
print("Invalid assessment cells:", int(A_INVALID.sum()), "/", A_INVALID.size)

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
# ---------------- User views + true calendar-time recency ----------------
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
# total chat, learning chat, noise, span, last gap, mean gap, median gap, max gap, unique topics
general = np.zeros((N_ALL,9), np.float32)

full_texts, recent_texts, latest_texts = [], [], []
latest_topic_cat, latest_tpl_cat, last2_seq_cat, journey_cat = [], [], [], []
latest_style_cat, tpl_transition_cat = [], []

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
                if dist==0:
                    parts += ["latest_topic_"+tp] * 3
                elif dist<=2:
                    parts += ["recent_topic_"+tp] * 2
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
        gaps=np.diff(ts.values).astype("timedelta64[s]").astype(float)/86400.0
        span=float((ts.iloc[-1]-ts.iloc[0]).total_seconds()/86400.0)
        latest_gap=float(gaps[-1]); mean_gap=float(gaps.mean()); median_gap=float(np.median(gaps)); max_gap=float(gaps.max())
    else:
        span=latest_gap=mean_gap=median_gap=max_gap=0.0
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
            if j==n-1:
                topic_latest[ui,ti]=1
            if pd.notna(latest_ts) and pd.notna(row.timestamp):
                age=max(0.0,float((latest_ts-row.timestamp).total_seconds()/86400.0))
                topic_age_days[ui,ti]=min(topic_age_days[ui,ti],age)

    valid = topic_age_days[ui] < 999
    topic_decay3[ui,valid] = np.exp(-np.minimum(topic_age_days[ui,valid],60)/3.0)
    topic_decay14[ui,valid] = np.exp(-np.minimum(topic_age_days[ui,valid],120)/14.0)
    topic_decay30[ui,valid] = np.exp(-np.minimum(topic_age_days[ui,valid],180)/30.0)

    lt = rows[-1].topics[0] if n and rows[-1].topics else "none"
    latest_topic_cat.append(lt)
    seq=[]
    for r in rows[-2:]:
        seq.append("+".join(r.topics[:2]) if r.topics else "none")
    last2_seq_cat.append(">".join(seq) if seq else "none")
    journey=[]
    for r in rows:
        for tp in r.topics:
            if not journey or journey[-1] != tp:
                journey.append(tp)
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
latest_tpl_cat=np.asarray(latest_tpl_cat,dtype=object)
last2_seq_cat=np.asarray(last2_seq_cat,dtype=object)
journey_cat=np.asarray(journey_cat,dtype=object)
latest_style_cat=np.asarray(latest_style_cat,dtype=object)
tpl_transition_cat=np.asarray(tpl_transition_cat,dtype=object)
hist_bucket_cat=np.where(general[:,1]>=8,"h8p",np.where(general[:,1]>=5,"h5_7",np.where(general[:,1]>=3,"h3_4",np.where(general[:,1]>=1,"h1_2","h0")))).astype(object)

USER_CORE = np.column_stack([A,A_INVALID,general,style_count,style_latest]).astype(np.float32)
assert np.isfinite(USER_CORE).all()
print("USER_CORE:", USER_CORE.shape)
print("latest topic train:", pd.Series(latest_topic_cat[:N_TRAIN]).value_counts().to_dict())

# ---- cell 7 (code) ----
# ---------------- TF-IDF views (unsupervised vocabulary fit on organizer-provided train+test text) ----------------
t0=time.time()
def make_view(texts, word_features, char_features):
    w=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_df=0.995,sublinear_tf=True,strip_accents="unicode",max_features=word_features,dtype=np.float32)
    c=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=2,sublinear_tf=True,max_features=char_features,dtype=np.float32)
    return hstack([w.fit_transform(texts),c.fit_transform(texts)],format="csr")

X_full = make_view(full_texts, 50_000, 60_000)
X_recent = make_view(recent_texts, 35_000, 40_000)
X_latest = make_view(latest_texts, 30_000, 35_000)
print("X_full/recent/latest:", X_full.shape, X_recent.shape, X_latest.shape)
print("vectorization seconds:", round(time.time()-t0,1))

# ---- cell 8 (code) ----
# ---------------- Candidate-native assessment / intent interactions ----------------
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

PAIR={}
shape=(N_ALL,N_MODULES)
for k in ["latest","count","posrec","lastpos","decay3","decay14","decay30"]:
    PAIR[k]=np.zeros(shape,np.float32)
PAIR["age"]=np.full(shape,999.,np.float32)
for k in ["pre_min","pre_mean","pre_ready","course_skill","skill_gap"]:
    PAIR[k]=np.zeros(shape,np.float32)

for m,mid in enumerate(MODULES):
    tis=[TOPIC_INDEX[t] for t in MODULE_TOPIC_MAP[mid] if t in TOPIC_INDEX]
    if tis:
        PAIR["latest"][:,m]=topic_latest[:,tis].max(1)
        PAIR["count"][:,m]=topic_count[:,tis].max(1)
        PAIR["posrec"][:,m]=topic_posrec[:,tis].max(1)
        PAIR["lastpos"][:,m]=topic_lastpos[:,tis].max(1)
        PAIR["age"][:,m]=topic_age_days[:,tis].min(1)
        PAIR["decay3"][:,m]=topic_decay3[:,tis].max(1)
        PAIR["decay14"][:,m]=topic_decay14[:,tis].max(1)
        PAIR["decay30"][:,m]=topic_decay30[:,tis].max(1)
    ps=PREREQ_SKILLS.get(mid,[])
    if ps:
        vals=A[:,[SKILL_INDEX[s] for s in ps]]
        PAIR["pre_min"][:,m]=vals.min(1)
        PAIR["pre_mean"][:,m]=vals.mean(1)
        PAIR["pre_ready"][:,m]=(vals.min(1)>=3).astype(np.float32)
    cs=COURSE_SKILL.get(mid)
    PAIR["course_skill"][:,m]=A[:,SKILL_INDEX[cs]] if cs else A.mean(1)
    PAIR["skill_gap"][:,m]=5.0-PAIR["course_skill"][:,m]

assert all(np.isfinite(v).all() for v in PAIR.values())

# ---- cell 9 (code) ----
# ---------------- Fold-safe target-memory encoder ----------------
MEMORY_KEYS = {
    "topic": latest_topic_cat,
    "template": latest_tpl_cat,
    "template_topic": np.asarray([f"{a}|{b}" for a,b in zip(latest_tpl_cat,latest_topic_cat)],object),
    "last2": last2_seq_cat,
    "journey": journey_cat,
    "transition": tpl_transition_cat,
    "topic_history": np.asarray([f"{a}|{b}" for a,b in zip(latest_topic_cat,hist_bucket_cat)],object),
}
MEMORY_ALPHA = {"topic":16.0,"template":8.0,"template_topic":6.0,"last2":8.0,"journey":6.0,"transition":8.0,"topic_history":12.0}

def memory_lookup(train_idx, query_idx, key_arr, target, alpha):
    train_idx=np.asarray(train_idx,int); query_idx=np.asarray(query_idx,int)
    prior=target[train_idx].mean(0).astype(np.float64)
    sums={}; counts={}
    for i in train_idx:
        k=str(key_arr[i])
        if k in sums:
            sums[k]+=target[i]; counts[k]+=1
        else:
            sums[k]=target[i].astype(np.float64).copy(); counts[k]=1
    out=np.empty((len(query_idx),N_MODULES),np.float32)
    for q,i in enumerate(query_idx):
        k=str(key_arr[i]); c=counts.get(k,0)
        if c:
            out[q]=((sums[k]+alpha*prior)/(c+alpha)).astype(np.float32)
        else:
            out[q]=prior
    return out

def build_memory_features(outer_tr, inner_fid, va_idx):
    # Returns module-aligned memory arrays for outer-train OOF, outer-val, and test.
    targets={"raw":y, "abs":Y_GAIN, "top":Y_TOP.astype(np.float32)}
    tr_mem={}; va_mem={}; te_mem={}
    test_idx=np.arange(N_TRAIN,N_ALL)
    for key_name,key_arr in MEMORY_KEYS.items():
        alpha=MEMORY_ALPHA[key_name]
        for target_name,target in targets.items():
            name=f"mem_{key_name}_{target_name}"
            o=np.zeros((len(outer_tr),N_MODULES),np.float32)
            for k in range(N_INNER):
                loc_tr=np.where(inner_fid!=k)[0]; loc_va=np.where(inner_fid==k)[0]
                o[loc_va]=memory_lookup(outer_tr[loc_tr],outer_tr[loc_va],key_arr,target,alpha)
            tr_mem[name]=o
            va_mem[name]=memory_lookup(outer_tr,va_idx,key_arr,target,alpha)
            te_mem[name]=memory_lookup(outer_tr,test_idx,key_arr,target,alpha)
    return tr_mem,va_mem,te_mem

# ---- cell 10 (code) ----
# ---------------- Nested base experts ----------------
HEAD_SPECS = {
    "raw_full": (y, X_full, 2.6),
    "abs_full": (Y_GAIN, X_full, 3.0),
    "top_full": (Y_TOP.astype(np.float32), X_full, 3.2),
    "bin_full": (Y_BIN.astype(np.float32), X_full, 3.2),
    "raw_recent": (y, X_recent, 3.6),
    "raw_latest": (y, X_latest, 4.0),
}
HEAD_NAMES=list(HEAD_SPECS)

def ridge_fit_predict(X, target, fit_idx, pred_idx, alpha):
    model=Ridge(alpha=alpha,solver="lsqr",tol=2e-3,max_iter=1000)
    model.fit(X[fit_idx],target[fit_idx])
    return model.predict(X[pred_idx]).astype(np.float32)

def fit_nested_heads(outer_tr, va_idx, inner_fid):
    tr_heads={};va_heads={};te_heads={}
    test_idx=np.arange(N_TRAIN,N_ALL)
    for name,(target,X,alpha) in HEAD_SPECS.items():
        o=np.zeros((len(outer_tr),N_MODULES),np.float32)
        for k in range(N_INNER):
            loc_tr=np.where(inner_fid!=k)[0];loc_va=np.where(inner_fid==k)[0]
            o[loc_va]=ridge_fit_predict(X,target,outer_tr[loc_tr],outer_tr[loc_va],alpha)
        tr_heads[name]=o
        va_heads[name]=ridge_fit_predict(X,target,outer_tr,va_idx,alpha)
        te_heads[name]=ridge_fit_predict(X,target,outer_tr,test_idx,alpha)
    return tr_heads,va_heads,te_heads

def choose_sparse_weights(tr_heads, outer_tr, inner_fid):
    H=np.stack([tr_heads[k] for k in HEAD_NAMES])
    candidates=[]
    # Pure heads are mandatory candidates: the ensemble can never be forced away from the best head.
    for i in range(len(HEAD_NAMES)):
        w=np.zeros(len(HEAD_NAMES));w[i]=1.0;candidates.append(w)
    candidates += [
        np.array([.78,.06,.10,.02,.03,.01]),
        np.array([.62,.12,.12,.04,.07,.03]),
        np.array([.70,.10,.08,.04,.06,.02]),
        np.array([.55,.15,.12,.05,.09,.04]),
    ]
    rng=np.random.default_rng(SEED+991)
    for _ in range(180):
        w=rng.dirichlet([10,2.5,2.5,1.3,2.0,1.2])
        candidates.append(w)
    best=(-1e9,None,None)
    for w in candidates:
        fs=[]
        pred=np.tensordot(w,H,axes=(0,0))
        for k in range(N_INNER):
            loc=np.where(inner_fid==k)[0]
            fs.append(ndcg5(y[outer_tr[loc]],pred[loc]))
        obj=float(np.mean(fs)-0.12*np.std(fs))
        if obj>best[0]:best=(obj,w,fs)
    return best[1]/best[1].sum(), best[2]

# ---- cell 11 (code) ----
# ---------------- Fast nonlinear experts ----------------
def row_rank01(a):
    r=np.argsort(np.argsort(-a,axis=1),axis=1).astype(np.float32)
    return 1.0-r/max(N_MODULES-1,1)

def row_z(a):
    return ((a-a.mean(1,keepdims=True))/(a.std(1,keepdims=True)+1e-6)).astype(np.float32)

def basic_candidate_numeric(user_global_idx, heads, sparse, rel=None, memories=None):
    n=len(user_global_idx)
    cols=[]; names=[]
    for name,a in heads.items():
        cols += [a, row_rank01(a), row_z(a)]
        names += [name,name+"_rank",name+"_z"]
    cols += [sparse,row_rank01(sparse),row_z(sparse)]
    names += ["sparse","sparse_rank","sparse_z"]
    if rel is not None:
        cols += [rel,row_rank01(rel),row_z(rel)]
        names += ["rel","rel_rank","rel_z"]
    if memories:
        for name,a in memories.items():
            cols.append(a); names.append(name)
    # pair features aligned with candidate module
    for name,a in PAIR.items():
        cols.append(a[user_global_idx]);names.append("pair_"+name)
    # repeat compact user-level numerics across 17 candidates
    uc=USER_CORE[user_global_idx]
    for j in range(uc.shape[1]):
        cols.append(np.repeat(uc[:,j:j+1],N_MODULES,axis=1));names.append(f"user_{j}")
    # module one-hot helps global LightGBM specialists learn module-specific rules.
    for m in range(N_MODULES):
        one=np.zeros((n,N_MODULES),np.float32);one[:,m]=1.0
        cols.append(one);names.append(f"is_module_{m}")
    X=np.stack(cols,axis=2).astype(np.float32)
    assert X.shape[:2]==(n,N_MODULES) and np.isfinite(X).all()
    return X,names

def fit_relevance_expert(outer_tr,va_idx,inner_fid,tr_heads,va_heads,te_heads,sparse_tr,sparse_va,sparse_te,tr_mem,va_mem,te_mem):
    rel_tr=np.zeros((len(outer_tr),N_MODULES),np.float32)
    rel_va=np.zeros((len(va_idx),N_MODULES),np.float32)
    rel_te=np.zeros((N_TEST,N_MODULES),np.float32)
    # Per-module models proved especially transferable in V3.1.
    for m in range(N_MODULES):
        def matrix(heads,sparse,mem,global_idx):
            parts=[sparse[:,m:m+1]]+[heads[k][:,m:m+1] for k in HEAD_NAMES]
            parts += [mem[k][:,m:m+1] for k in sorted(mem)]
            parts += [USER_CORE[global_idx]]
            parts += [np.column_stack([PAIR[k][global_idx,m] for k in PAIR])]
            return np.column_stack(parts).astype(np.float32)
        Xtr=matrix(tr_heads,sparse_tr,tr_mem,outer_tr)
        Xva=matrix(va_heads,sparse_va,va_mem,va_idx)
        Xte=matrix(te_heads,sparse_te,te_mem,np.arange(N_TRAIN,N_ALL))
        target=Y_BIN[outer_tr,m]
        if target.min()==target.max():
            rel_tr[:,m]=target[0];rel_va[:,m]=target[0];rel_te[:,m]=target[0];continue
        for k in range(N_INNER):
            a=np.where(inner_fid!=k)[0];b=np.where(inner_fid==k)[0]
            clf=lgb.LGBMClassifier(n_estimators=220 if not V6_SMOKE else 30,learning_rate=.035,num_leaves=13,min_child_samples=55,
                reg_alpha=.7,reg_lambda=7.0,colsample_bytree=.75,random_state=SEED+101*m+k,n_jobs=-1,verbosity=-1,force_col_wise=True)
            clf.fit(Xtr[a],target[a])
            rel_tr[b,m]=clf.predict_proba(Xtr[b])[:,1]
        clf=lgb.LGBMClassifier(n_estimators=220 if not V6_SMOKE else 30,learning_rate=.035,num_leaves=13,min_child_samples=55,
            reg_alpha=.7,reg_lambda=7.0,colsample_bytree=.75,random_state=SEED+7001+101*m,n_jobs=-1,verbosity=-1,force_col_wise=True)
        clf.fit(Xtr,target)
        rel_va[:,m]=clf.predict_proba(Xva)[:,1]
        rel_te[:,m]=clf.predict_proba(Xte)[:,1]
    return rel_tr,rel_va,rel_te

def fit_global_specialists(outer_tr,va_idx,inner_fid,tr_num,va_num,te_num):
    # Flatten candidate rows. Every inner split is by USER, never by candidate row.
    Xtr=tr_num.reshape(-1,tr_num.shape[2]);Xva=va_num.reshape(-1,va_num.shape[2]);Xte=te_num.reshape(-1,te_num.shape[2])
    top_tr=np.zeros((len(outer_tr),N_MODULES),np.float32)
    ord_tr=np.zeros_like(top_tr)
    top_va=np.zeros((len(va_idx),N_MODULES),np.float32);top_te=np.zeros((N_TEST,N_MODULES),np.float32)
    ord_va=np.zeros_like(top_va);ord_te=np.zeros_like(top_te)
    ytop=Y_TOP[outer_tr].reshape(-1)
    yord=Y_ABS[outer_tr].reshape(-1)
    for k in range(N_INNER):
        a=np.where(inner_fid!=k)[0];b=np.where(inner_fid==k)[0]
        ar=(a[:,None]*N_MODULES+np.arange(N_MODULES)).reshape(-1)
        br=(b[:,None]*N_MODULES+np.arange(N_MODULES)).reshape(-1)
        top=lgb.LGBMClassifier(objective="binary",n_estimators=260 if not V6_SMOKE else 35,learning_rate=.035,num_leaves=21,min_child_samples=70,
            reg_alpha=.8,reg_lambda=8.0,colsample_bytree=.8,scale_pos_weight=10.0,random_state=SEED+300+k,n_jobs=-1,verbosity=-1,force_col_wise=True)
        top.fit(Xtr[ar],ytop[ar]);top_tr[b]=top.predict_proba(Xtr[br])[:,1].reshape(len(b),N_MODULES)
        ordm=lgb.LGBMClassifier(objective="multiclass",num_class=7,n_estimators=300 if not V6_SMOKE else 40,learning_rate=.035,num_leaves=25,min_child_samples=65,
            reg_alpha=.6,reg_lambda=8.0,colsample_bytree=.8,random_state=SEED+500+k,n_jobs=-1,verbosity=-1,force_col_wise=True,
            class_weight={0:1.0,1:1.6,2:1.8,3:2.0,4:2.3,5:2.7,6:3.2})
        ordm.fit(Xtr[ar],yord[ar]);p=ordm.predict_proba(Xtr[br]);ord_tr[b]=(p@GAIN_BY_CLASS).reshape(len(b),N_MODULES)
    all_rows=np.arange(len(outer_tr)*N_MODULES)
    top=lgb.LGBMClassifier(objective="binary",n_estimators=260 if not V6_SMOKE else 35,learning_rate=.035,num_leaves=21,min_child_samples=70,
        reg_alpha=.8,reg_lambda=8.0,colsample_bytree=.8,scale_pos_weight=10.0,random_state=SEED+3300,n_jobs=-1,verbosity=-1,force_col_wise=True)
    top.fit(Xtr,ytop);top_va[:]=top.predict_proba(Xva)[:,1].reshape(len(va_idx),N_MODULES);top_te[:]=top.predict_proba(Xte)[:,1].reshape(N_TEST,N_MODULES)
    ordm=lgb.LGBMClassifier(objective="multiclass",num_class=7,n_estimators=300 if not V6_SMOKE else 40,learning_rate=.035,num_leaves=25,min_child_samples=65,
        reg_alpha=.6,reg_lambda=8.0,colsample_bytree=.8,random_state=SEED+3500,n_jobs=-1,verbosity=-1,force_col_wise=True,
        class_weight={0:1.0,1:1.6,2:1.8,3:2.0,4:2.3,5:2.7,6:3.2})
    ordm.fit(Xtr,yord);ord_va[:]=(ordm.predict_proba(Xva)@GAIN_BY_CLASS).reshape(len(va_idx),N_MODULES);ord_te[:]=(ordm.predict_proba(Xte)@GAIN_BY_CLASS).reshape(N_TEST,N_MODULES)
    return top_tr,top_va,top_te,ord_tr,ord_va,ord_te

# ---- cell 12 (code) ----
# ---------------- Final categorical candidate frame ----------------
def age_bin(x):
    x=np.asarray(x)
    out=np.full(x.shape,"d60plus",dtype=object)
    out[x>=999]="never";out[x<=60]="d14_60";out[x<=14]="d3_14";out[x<=3]="d0_3";out[x<=.01]="current"
    return out

CAT_COLS=["module","latest_topic","latest_tpl","last2_seq","journey","latest_style","hist_bucket",
          "module_topic","module_tpl","module_seq","module_agebin","module_skillband"]

def build_rank_frame(global_idx,heads,sparse,rel,anchor,top,ordinal,memories):
    global_idx=np.asarray(global_idx,int);n=len(global_idx)
    data={}
    experts={**heads,"sparse":sparse,"rel":rel,"anchor":anchor,"top1expert":top,"ordinalexpert":ordinal}
    for name,a in experts.items():
        data[name]=a.reshape(-1).astype(np.float32)
        data[name+"_rank"]=row_rank01(a).reshape(-1)
        data[name+"_z"]=row_z(a).reshape(-1)
        mx=a.max(1,keepdims=True)
        data[name+"_margin"]=(a-mx).reshape(-1).astype(np.float32)
    for name,a in memories.items():
        data[name]=a.reshape(-1).astype(np.float32)
    for name,a in PAIR.items():
        data["pair_"+name]=a[global_idx].reshape(-1).astype(np.float32)
    for j in range(USER_CORE.shape[1]):
        data[f"user_{j}"]=np.repeat(USER_CORE[global_idx,j],N_MODULES).astype(np.float32)

    module=np.tile(np.asarray(MODULES,dtype=object),n)
    lt=np.repeat(latest_topic_cat[global_idx],N_MODULES)
    tpl=np.repeat(latest_tpl_cat[global_idx],N_MODULES)
    seq=np.repeat(last2_seq_cat[global_idx],N_MODULES)
    jour=np.repeat(journey_cat[global_idx],N_MODULES)
    sty=np.repeat(latest_style_cat[global_idx],N_MODULES)
    hb=np.repeat(hist_bucket_cat[global_idx],N_MODULES)
    ageb=age_bin(PAIR["age"][global_idx]).reshape(-1)
    cur=PAIR["course_skill"][global_idx].reshape(-1)
    skillb=np.where(cur<=1,"low",np.where(cur<=3,"mid","high"))
    data.update(
        module=module,latest_topic=lt,latest_tpl=tpl,last2_seq=seq,journey=jour,latest_style=sty,hist_bucket=hb,
        module_topic=np.asarray([f"{a}|{b}" for a,b in zip(module,lt)],object),
        module_tpl=np.asarray([f"{a}|{b}" for a,b in zip(module,tpl)],object),
        module_seq=np.asarray([f"{a}|{b}" for a,b in zip(module,seq)],object),
        module_agebin=np.asarray([f"{a}|{b}" for a,b in zip(module,ageb)],object),
        module_skillband=np.asarray([f"{a}|{b}" for a,b in zip(module,skillb)],object),
    )
    df=pd.DataFrame(data)
    for c in CAT_COLS:
        df[c]=df[c].fillna("none").astype(str)
    nums=[c for c in df if c not in CAT_COLS]
    df[nums]=df[nums].astype(np.float32)
    assert np.isfinite(df[nums].to_numpy(np.float32)).all()
    return df

def user_rows(local_user_indices):
    local_user_indices=np.asarray(local_user_indices,int)
    return (local_user_indices[:,None]*N_MODULES+np.arange(N_MODULES)[None,:]).reshape(-1)

def make_ndcg_ranker(seed):
    # Non-classic YetiRank NDCG mode is CPU-only in CatBoost.
    return CatBoostRanker(
        loss_function="YetiRank:mode=NDCG;top=5;dcg_type=Base;dcg_denominator=LogPosition",
        eval_metric="NDCG:top=5;type=Base;denominator=LogPosition",
        iterations=45 if V6_SMOKE else 180,
        learning_rate=.07 if V6_SMOKE else .035,
        depth=5,l2_leaf_reg=11.0,random_strength=.30,random_seed=seed,
        one_hot_max_size=32,max_ctr_complexity=1,od_type="Iter",od_wait=12 if V6_SMOKE else 35,
        task_type="CPU",thread_count=4,allow_writing_files=False,verbose=False,
    )

# ---- cell 13 (code) ----
# ---------------- Outer nested CV: every supervised meta-feature is trained without the outer validation fold ----------------
rank_oof=np.zeros_like(y,np.float32)
anchor_oof=np.zeros_like(y,np.float32)
ordinal_oof=np.zeros_like(y,np.float32)
top_oof=np.zeros_like(y,np.float32)
rank_test_folds=[];anchor_test_folds=[];ordinal_test_folds=[];top_test_folds=[]
outer_rank_scores=[];outer_anchor_scores=[];best_iters=[]
fold_debug=[]

outer_to_run = outer_folds[:1] if V6_SMOKE else outer_folds
for f,(outer_tr,va_idx) in enumerate(outer_to_run):
    print(f"\n========== V6 OUTER FOLD {f+1}/{len(outer_to_run)} ==========")
    inner_fid=balanced_fold_ids(outer_tr,N_INNER,SEED+1000+f)
    print("inner sizes:",[int((inner_fid==k).sum()) for k in range(N_INNER)])

    tr_heads,va_heads,te_heads=fit_nested_heads(outer_tr,va_idx,inner_fid)
    w,inner_scores=choose_sparse_weights(tr_heads,outer_tr,inner_fid)
    Htr=np.stack([tr_heads[k] for k in HEAD_NAMES]);Hva=np.stack([va_heads[k] for k in HEAD_NAMES]);Hte=np.stack([te_heads[k] for k in HEAD_NAMES])
    sparse_tr=np.tensordot(w,Htr,axes=(0,0)).astype(np.float32)
    sparse_va=np.tensordot(w,Hva,axes=(0,0)).astype(np.float32)
    sparse_te=np.tensordot(w,Hte,axes=(0,0)).astype(np.float32)
    print("sparse weights:",{k:round(float(v),3) for k,v in zip(HEAD_NAMES,w) if v>.001},"inner",[round(x,5) for x in inner_scores])

    tr_mem,va_mem,te_mem=build_memory_features(outer_tr,inner_fid,va_idx)
    rel_tr,rel_va,rel_te=fit_relevance_expert(outer_tr,va_idx,inner_fid,tr_heads,va_heads,te_heads,sparse_tr,sparse_va,sparse_te,tr_mem,va_mem,te_mem)
    # V3.1 transfer lesson: relevance calibration is valuable, but keep a majority sparse backbone.
    anchor_tr=.64*sparse_tr+.36*rel_tr
    anchor_va=.64*sparse_va+.36*rel_va
    anchor_te=.64*sparse_te+.36*rel_te

    tr_num,_=basic_candidate_numeric(outer_tr,tr_heads,sparse_tr,rel_tr,tr_mem)
    va_num,_=basic_candidate_numeric(va_idx,va_heads,sparse_va,rel_va,va_mem)
    te_num,_=basic_candidate_numeric(np.arange(N_TRAIN,N_ALL),te_heads,sparse_te,rel_te,te_mem)
    top_tr,top_va,top_te,ord_tr,ord_va,ord_te=fit_global_specialists(outer_tr,va_idx,inner_fid,tr_num,va_num,te_num)

    Ftr=build_rank_frame(outer_tr,tr_heads,sparse_tr,rel_tr,anchor_tr,top_tr,ord_tr,tr_mem)
    Fva=build_rank_frame(va_idx,va_heads,sparse_va,rel_va,anchor_va,top_va,ord_va,va_mem)
    Fte=build_rank_frame(np.arange(N_TRAIN,N_ALL),te_heads,sparse_te,rel_te,anchor_te,top_te,ord_te,te_mem)

    train_pool=Pool(Ftr,label=Y_GAIN[outer_tr].reshape(-1),group_id=np.repeat(np.arange(len(outer_tr)),N_MODULES),cat_features=CAT_COLS)
    valid_pool=Pool(Fva,label=Y_GAIN[va_idx].reshape(-1),group_id=np.repeat(np.arange(len(va_idx)),N_MODULES),cat_features=CAT_COLS)
    test_pool=Pool(Fte,group_id=np.repeat(np.arange(N_TEST),N_MODULES),cat_features=CAT_COLS)
    ranker=make_ndcg_ranker(SEED+8000+f)
    ranker.fit(train_pool,eval_set=valid_pool,use_best_model=True,verbose=False)
    it=max(1,int(ranker.get_best_iteration()+1));best_iters.append(it)
    pva=ranker.predict(valid_pool).reshape(len(va_idx),N_MODULES).astype(np.float32)
    pte=ranker.predict(test_pool).reshape(N_TEST,N_MODULES).astype(np.float32)

    rank_oof[va_idx]=pva;anchor_oof[va_idx]=anchor_va;ordinal_oof[va_idx]=ord_va;top_oof[va_idx]=top_va
    rank_test_folds.append(pte);anchor_test_folds.append(anchor_te);ordinal_test_folds.append(ord_te);top_test_folds.append(top_te)
    rs=ndcg5(y[va_idx],pva);ascore=ndcg5(y[va_idx],anchor_va)
    outer_rank_scores.append(rs);outer_anchor_scores.append(ascore)
    fold_debug.append(dict(fold=f+1,rank=rs,anchor=ascore,iter=it))
    print(f"outer fold {f+1}: rank={rs:.6f} anchor={ascore:.6f} delta={rs-ascore:+.6f} iter={it}")
    del Ftr,Fva,Fte,train_pool,valid_pool,test_pool,ranker,tr_num,va_num,te_num
    gc.collect()

if V6_SMOKE:
    print("SMOKE MODE completed one outer fold; metrics are not competition estimates.")
else:
    rank_test=np.mean(rank_test_folds,axis=0).astype(np.float32)
    anchor_test=np.mean(anchor_test_folds,axis=0).astype(np.float32)
    ordinal_test=np.mean(ordinal_test_folds,axis=0).astype(np.float32)
    top_test=np.mean(top_test_folds,axis=0).astype(np.float32)
    print("\nNested ranker OOF:",round(ndcg5(y,rank_oof),6))
    print("Nested anchor OOF:",round(ndcg5(y,anchor_oof),6))
    print("Best iterations:",best_iters)

# ---- cell 14 (code) ----
# ---------------- Pre-registered robust final candidates ----------------
if not V6_SMOKE:
    # Only a tiny, pre-declared family: avoids dense OOF weight mining.
    candidates={
        "ranker":(rank_oof,rank_test),
        "rank90_anchor10":(.90*rank_oof+.10*anchor_oof,.90*rank_test+.10*anchor_test),
        "rank90_ord10":(.90*rank_oof+.10*ordinal_oof,.90*rank_test+.10*ordinal_test),
        "rank85_anchor10_top05":(.85*rank_oof+.10*anchor_oof+.05*top_oof,.85*rank_test+.10*anchor_test+.05*top_test),
    }
    def fold_scores(scores):
        return [ndcg5(y[outer_fid==f],scores[outer_fid==f]) for f in range(N_OUTER)]
    quality={}
    print("\nV6 CANDIDATES")
    for name,(oo,_) in candidates.items():
        fs=fold_scores(oo);mean=ndcg5(y,oo);std=float(np.std(fs))
        quality[name]=mean-.12*std
        print(f"{name:24s} mean={mean:.6f} std={std:.6f} folds={[round(x,6) for x in fs]}")
    primary_name=max(quality,key=quality.get)
    primary_oof,primary_test=candidates[primary_name]
    print("PRIMARY:",primary_name,round(ndcg5(y,primary_oof),6))

# ---- cell 15 (code) ----
# ---------------- Diagnostics and submission gate ----------------
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

if not V6_SMOKE:
    print("\nFINAL DIAGNOSTICS")
    for name,(oo,_) in candidates.items():
        print(f"{name:24s} top1_acc={(oo.argmax(1)==TOP1).mean():.4f} true_top1@5={true_top1_in_top5(oo):.4f}")
    print("\nPrimary per-module recall@5")
    print(module_recall5(primary_oof).round(4).to_string(index=False))
    top5=np.argsort(-primary_test,axis=1)[:,:5]
    freq=np.bincount(top5.ravel(),minlength=N_MODULES)
    print("\nPrimary test Top-5 frequency")
    print(pd.DataFrame({"module_id":MODULES,"count":freq}).to_string(index=False))

    rank_wins=sum(r>a for r,a in zip(outer_rank_scores,outer_anchor_scores))
    primary_score=ndcg5(y,primary_oof)
    gate_score=primary_score>=0.667
    gate_folds=rank_wins>=4
    # Tail is explicitly a diagnostic, not an independent holdout, because folds span the whole train set.
    tail=np.arange(3200,4000)
    tail_score=ndcg5(y[tail],primary_oof[tail])
    print("\nSubmission gate")
    print("Nested OOF >= 0.667:",gate_score,"score=",round(primary_score,6))
    print("Ranker beats nested anchor in >=4/5 outer folds:",gate_folds,f"({rank_wins}/5)")
    print("ID-tail OOF diagnostic U_3201..U_4000:",round(tail_score,6))
    print("GATE:","PASS" if gate_score and gate_folds else "REVIEW — do not spend the last submission yet")

# ---- cell 16 (code) ----
# ---------------- Hard-validated outputs ----------------
def write_submission(scores,filename):
    assert scores.shape==(N_TEST,N_MODULES)
    assert np.isfinite(scores).all()
    out=sample.copy()
    assert out["user_id"].tolist()==test["user_id"].tolist()
    out.loc[:,MODULES]=scores
    root=Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("/mnt/data")
    path=root/filename
    out.to_csv(path,index=False)
    chk=pd.read_csv(path)
    assert chk.shape==sample.shape and chk["user_id"].is_unique
    assert chk[MODULES].notna().all().all() and np.isfinite(chk[MODULES].to_numpy()).all()
    print("Saved:",path)
    return path

if not V6_SMOKE:
    primary_path=write_submission(primary_test,"submission_v6_primary.csv")
    ranker_path=write_submission(rank_test,"submission_v6_ranker.csv")
    anchor_path=write_submission(anchor_test,"submission_v6_anchor.csv")
    root=Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("/mnt/data")
    np.savez_compressed(root/"v6_oof_artifacts.npz",
        y=y,outer_fid=outer_fid,rank_oof=rank_oof,anchor_oof=anchor_oof,ordinal_oof=ordinal_oof,top_oof=top_oof,
        primary_oof=primary_oof,rank_test=rank_test,anchor_test=anchor_test,ordinal_test=ordinal_test,top_test=top_test,primary_test=primary_test)
    print("\nFINAL SUMMARY")
    print("Nested structured anchor:",round(ndcg5(y,anchor_oof),6))
    print("Nested NDCG ranker     :",round(ndcg5(y,rank_oof),6))
    print("Primary               :",primary_name,round(ndcg5(y,primary_oof),6))
    print("Primary top1 accuracy :",round(float((primary_oof.argmax(1)==TOP1).mean()),4))
    print("Primary true top1@5   :",round(true_top1_in_top5(primary_oof),4))
    print("Primary file          :",primary_path.name)
