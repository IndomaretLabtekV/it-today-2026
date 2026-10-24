# Converted from notebooks/v36.ipynb: markdown cells become comments, %/! magics neutralized.

# ---- cell 0 (markdown) ----
# # MineToday V36 — Hybrid34-Locked State-Conditional Fold-Matched XE Proposal Competence Router
#
# **Objective:** improve the current public-best **0.66006 Hybrid34** without replacing its ranking geometry globally.
#
# V36 treats LightGBM XE-NDCG as a **proposal generator**, not a standalone ranker. A low-capacity competence router estimates whether an XE Top-5 proposal is likely to improve exact NDCG@5. The public-proven Hybrid34 region is immutable; the public-toxic V31-only state is excluded.
#
# Core protocol:
# - reconstruct V25 → V30 → V31 and deterministically rebuild **Hybrid34**;
# - lock the S3 disagreement region that produced public 0.66006;
# - exclude S1 (`V30=V25-like`, `V31!=V30`) from V36 actions;
# - train XE proposal competence on **honest fold-matched shadow predictions**;
# - use exact signed `ΔNDCG@5` as the meta target, with separate benefit/harm/regression heads;
# - compare **CORRIDOR** (S2 only) and **ATTACK** (S2 + S0) under three untouched development worlds;
# - sealed group-disjoint D and chronology are rejection-only;
# - if any promotion gate fails, `submission_v36_primary.csv` is the exact reconstructed **Hybrid34 fail-close anchor**.
#
# No external dataset is used. All learned components use organizer-provided data only. The notebook is CPU-first for deterministic LightGBM ranking and is compatible with Kaggle T4x2 runtimes.

# ---- cell 1 (code) ----
# Runtime + preregistered V36 configuration
import os, sys, re, json, ast, html, hashlib, random, warnings, time, subprocess, gc, math
from pathlib import Path
from collections import Counter, defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ["PYTHONHASHSEED"] = "42"

V36_SKIP_INSTALL = bool(int(os.getenv("V36_SKIP_INSTALL", "0")))
try:
    import lightgbm as lgb
except Exception:
    if V36_SKIP_INSTALL:
        raise
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "lightgbm==4.6.0"])
    import lightgbm as lgb

try:
    from catboost import CatBoostRanker, Pool
    import catboost
except Exception:
    if V36_SKIP_INSTALL:
        raise
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "catboost==1.2.10"])
    from catboost import CatBoostRanker, Pool
    import catboost

import numpy as np
import pandas as pd
import scipy
import sklearn
from IPython.display import display
from scipy.sparse import hstack
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, StratifiedShuffleSplit, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")

SEED = 42
V36_SMOKE = bool(int(os.getenv("V36_SMOKE", "0")))
V36_STRICT_GEOMETRY = bool(int(os.getenv("V36_STRICT_GEOMETRY", "1")))

# Compatibility aliases consumed by the frozen V3.1/V25/V30/V31 cells.
V31_SMOKE = V36_SMOKE
V30_SMOKE = V36_SMOKE
V30_STRICT_GEOMETRY = V36_STRICT_GEOMETRY
V29_SMOKE = V36_SMOKE
V29_STRICT_GEOMETRY = V36_STRICT_GEOMETRY
V27_SMOKE = V36_SMOKE
V27_STRICT_GEOMETRY = V36_STRICT_GEOMETRY

N_FOLDS = 2 if V36_SMOKE else 5
ANCHOR_INNER_FOLDS = 2 if V36_SMOKE else 4
INTENT_INNER_FOLDS = 2 if V36_SMOKE else 4

RIDGE_ALPHA = 3.0
REFERENCE_W = 0.45

PUBLIC_CAL045_LB = 0.65639
PUBLIC_CONSENSUS3_LB = 0.65856
PUBLIC_V25_LB = 0.65919
PUBLIC_V30_LB = 0.65963
PUBLIC_V31_LB = 0.65949
PUBLIC_HYBRID34_LB = 0.66006
PUBLIC_LEADER_LB = 0.66142
PUBLIC_LEADER_GAP = PUBLIC_LEADER_LB - PUBLIC_HYBRID34_LB

LGB_ESTIMATORS = 12 if V36_SMOKE else 1200
LGB_EARLY_STOP = 3 if V36_SMOKE else 80
RANKER_ESTIMATORS = 16 if V36_SMOKE else 360
RANKER_LEARNING_RATE = 0.04 if V36_SMOKE else 0.025
YETI_ITERATIONS = 16 if V36_SMOKE else 180
YETI_LEARNING_RATE = 0.07 if V36_SMOKE else 0.035

# Exact public-proven V30 residual strength and frozen V31 Intent-State hierarchy.
ALPHA = 0.16
INTENT_PARENT_STRENGTH = 40.0
INTENT_CHILD_STRENGTH = 30.0

# V36 fold-matching: every XE shadow used by the router is trained on the same number of users.
DEV_VAL_SIZE = 80 if V36_SMOKE else 400
DEV_SHADOW_FOLDS = 2 if V36_SMOKE else 4
EXPERT_TRAIN_SIZE = 300 if V36_SMOKE else 2700
FINAL_SHADOW_FOLDS = 2 if V36_SMOKE else 4
META_CV_FOLDS = 2 if V36_SMOKE else 4

# Benefit/harm economics are frozen from the prior XE audit, not fit on fresh A/B/C.
PROB_GAIN_REF = 0.050
PROB_HARM_REF = 0.055

# Inner-only policy families. Fresh A/B/C never tune these grids.
CORRIDOR_COVERAGE_GRID = (0.005, 0.010, 0.015) if not V36_SMOKE else (0.05, 0.10)
ATTACK_COVERAGE_GRID = (0.005, 0.010, 0.015, 0.020) if not V36_SMOKE else (0.05, 0.10)
CORRIDOR_MIN_PRECISION = 0.62
CORRIDOR_MAX_HARM = 0.35
ATTACK_MIN_PRECISION = 0.65
ATTACK_MAX_HARM = 0.30

# Shadow consensus. Development uses four 2700-user shadows; final deployment also uses four.
MIN_SHADOW_VOTES = 1 if V36_SMOKE else 3

# Development promotion gates versus Hybrid34.
CORRIDOR_MIN_POOLED = -1.0 if V36_SMOKE else 0.00035
CORRIDOR_MIN_WORST = -1.0 if V36_SMOKE else 0.0
CORRIDOR_CI_FLOOR = -1.0 if V36_SMOKE else -0.00005
ATTACK_MIN_POOLED = -1.0 if V36_SMOKE else 0.00150
ATTACK_MIN_WORST = -1.0 if V36_SMOKE else 0.00055
ATTACK_CI_FLOOR = -1.0 if V36_SMOKE else 0.00010
ORACLE_MIN_WORST = -1.0 if V36_SMOKE else 0.00120

BOOTSTRAPS = 20 if V36_SMOKE else 1200
GEOMETRY_ATOL = 2.5e-4

# Completely fresh partitions for V36. Historical G/H/I/K/L/M/N/O/P/Q/R/S are burned.
def _seed_from_text(s):
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:8], 16) % (2**31-1)

FRESH_WORLDS = {
    "A": _seed_from_text("MINETODAY-V36-FRESH-A-20260830-XE-COMPETENCE"),
    "B": _seed_from_text("MINETODAY-V36-FRESH-B-20260830-XE-COMPETENCE"),
    "C": _seed_from_text("MINETODAY-V36-FRESH-C-20260830-XE-COMPETENCE"),
}
SEALED_D_SALT = "MINETODAY-V36-SEALED-D-20260830-XE-COMPETENCE"

PROTECTED_TEST_USERS = {
    "U_4047", "U_4953", "U_4074", "U_4176", "U_4491", "U_4567", "U_4836", "U_4929"
}

# Exact V30 rank-geometry invariant from the executed public artifact.
EXPECTED_V30_CHANGED_USERS = {
    'U_4005','U_4015','U_4017','U_4043','U_4044','U_4058','U_4069','U_4070','U_4075','U_4087',
    'U_4089','U_4118','U_4121','U_4122','U_4130','U_4131','U_4137','U_4138','U_4169','U_4178',
    'U_4186','U_4214','U_4232','U_4233','U_4237','U_4265','U_4266','U_4269','U_4274','U_4285',
    'U_4300','U_4320','U_4335','U_4345','U_4354','U_4363','U_4379','U_4389','U_4394','U_4402',
    'U_4407','U_4430','U_4436','U_4438','U_4442','U_4515','U_4545','U_4555','U_4569','U_4596',
    'U_4633','U_4639','U_4661','U_4663','U_4665','U_4672','U_4687','U_4717','U_4718','U_4725',
    'U_4751','U_4759','U_4827','U_4828','U_4838','U_4886','U_4893','U_4897','U_4905','U_4919',
    'U_4933','U_4956','U_4960','U_4965','U_4978','U_4988','U_4989'
}
EXPECTED_V30_TOP1_CHANGES = 8
EXPECTED_V30_MEMBERSHIP_CHANGES = 7
EXPECTED_V31_CHANGED_VS_V30 = 60
EXPECTED_HYBRID_OVERLAP = 34
EXPECTED_STATE_COUNTS = {"S0":897, "S1":26, "S2":43, "S3":34}
EXPECTED_HYBRID_TOP5_SHA256 = "47f7db52ce8f782b8b393fd120140f1b6f51aeac452a9fabb5794cc9d8ba5e6e"

random.seed(SEED)
np.random.seed(SEED)

print("numpy", np.__version__)
print("pandas", pd.__version__)
print("scipy", scipy.__version__)
print("sklearn", sklearn.__version__)
print("lightgbm", lgb.__version__)
print("catboost", catboost.__version__)
print("V36_SMOKE", V36_SMOKE)
print("current public Hybrid34", PUBLIC_HYBRID34_LB, "leader gap", round(PUBLIC_LEADER_GAP,5))
print("fresh worlds", FRESH_WORLDS)
print("fold-matched XE train size", EXPERT_TRAIN_SIZE)
try:
    print("GPU runtime:\n" + subprocess.check_output(["nvidia-smi", "-L"], text=True).strip())
except Exception:
    print("nvidia-smi unavailable; V36 remains runnable. LightGBM is deliberately CPU-first for deterministic ranking.")

# ---- cell 2 (code) ----
def locate_data_dir():
    required = {
        "chat_history.csv", "modules_catalog.csv", "user_assessments.csv",
        "train_relevance.csv", "test.csv", "sample_submission.csv"
    }
    roots = [Path("/kaggle/input"), Path("/mnt/data/mine_data"), Path("/mnt/data/comp_v3"), Path("/mnt/data/comp_full"), Path(".")]
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
    raise FileNotFoundError("Could not locate MineToday competition dataset directory.")

DATA_DIR = locate_data_dir()
print("DATA_DIR =", DATA_DIR)

chat = pd.read_csv(DATA_DIR / "chat_history.csv")
modules = pd.read_csv(DATA_DIR / "modules_catalog.csv")
assessments = pd.read_csv(DATA_DIR / "user_assessments.csv")
train = pd.read_csv(DATA_DIR / "train_relevance.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
sample = pd.read_csv(DATA_DIR / "sample_submission.csv")

MODULES = list(train.columns[1:])
N_MODULES = len(MODULES)
N_TRAIN = len(train)
N_TEST = len(test)

assert MODULES == list(sample.columns[1:])
assert N_MODULES == 17
assert N_TRAIN == 4000 and N_TEST == 1000
assert train["user_id"].is_unique and test["user_id"].is_unique

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
# Competition metric + ordered targets
DISCOUNT = 1.0 / np.log2(np.arange(2, 7))

def per_user_ndcg(y_true, scores):
    y_true=np.asarray(y_true); scores=np.asarray(scores)
    idx=np.argpartition(-scores,4,axis=1)[:,:5]
    local=np.take_along_axis(scores,idx,axis=1)
    idx=np.take_along_axis(idx,np.argsort(-local,axis=1),axis=1)
    dcg=(np.take_along_axis(y_true,idx,axis=1)*DISCOUNT).sum(1)
    ideal=np.sort(y_true,axis=1)[:,-5:][:,::-1]
    idcg=(ideal*DISCOUNT).sum(1)
    return dcg/np.maximum(idcg,1e-12)

def ndcg5(y_true,scores): return float(per_user_ndcg(y_true,scores).mean())

def top5_indices(scores):
    idx=np.argpartition(-scores,4,axis=1)[:,:5]
    local=np.take_along_axis(scores,idx,axis=1)
    return np.take_along_axis(idx,np.argsort(-local,axis=1),axis=1)

def top5_set_change_rate(a,b):
    ta,tb=top5_indices(a),top5_indices(b)
    return float(np.mean([set(x)!=set(z) for x,z in zip(ta,tb)]))

def top1_change_rate(a,b): return float(np.mean(np.argmax(a,axis=1)!=np.argmax(b,axis=1)))

def boot_ci(delta):
    delta=np.asarray(delta,float); rng=np.random.default_rng(20260823); n=len(delta); z=np.empty(BOOTSTRAPS)
    for i in range(BOOTSTRAPS): z[i]=delta[rng.integers(0,n,n)].mean()
    return tuple(float(x) for x in np.quantile(z,[.025,.975]))

Y_BIN=(y>0).astype(np.float32)
Y_TOP=np.zeros_like(y,np.float32)
Y_TOP[np.arange(N_TRAIN),np.argmax(y,axis=1)]=1.0
RANK_POS=np.full_like(y,99,dtype=np.int16)
for i,row in enumerate(y):
    p=np.flatnonzero(row>0); order=p[np.argsort(-row[p])]
    for r,m in enumerate(order,1): RANK_POS[i,m]=r
Y_TOP5=(RANK_POS<=5).astype(np.float32)

# ID order is part of the validation hypothesis, so enforce it explicitly.
def uid_num(u):
    m=re.search(r"(\d+)$",str(u))
    if m is None: raise ValueError(f"Cannot parse ordered user id: {u}")
    return int(m.group(1))
TRAIN_ID_NUM=np.asarray([uid_num(u) for u in train.user_id],np.int32)
TEST_ID_NUM=np.asarray([uid_num(u) for u in test.user_id],np.int32)
assert np.all(np.diff(TRAIN_ID_NUM)>0), "train.csv must be ordered by increasing user_id for V12"
assert np.all(np.diff(TEST_ID_NUM)>0), "test.csv must be ordered by increasing user_id for V12"
assert TEST_ID_NUM.min()>TRAIN_ID_NUM.max(), "V12 diagnostic expects test to be the next ID block"

strata=np.asarray([MODULES[j] for j in np.argmax(y,axis=1)])
ref_skf=StratifiedKFold(N_FOLDS,shuffle=True,random_state=SEED)
ref_folds=list(ref_skf.split(np.zeros(N_TRAIN),strata))
print("train id range",TRAIN_ID_NUM.min(),TRAIN_ID_NUM.max(),"test",TEST_ID_NUM.min(),TEST_ID_NUM.max())
print("positive modules/user",pd.Series((y>0).sum(1)).value_counts().sort_index().to_dict())

# ---- cell 4 (code) ----
# Robust assessment parsing — exact V3.1 mechanism
WORD_NUM = {"nol":0, "satu":1, "dua":2, "tiga":3, "empat":4, "lima":5}
SKILL_NAMES = ["python","sql","statistics","eda","ml","ml_eval","dl","genai","business","research"]

A = np.zeros((len(all_users), 10), np.float32)
A_INVALID = np.zeros_like(A)

for row in assessments.itertuples(index=False):
    try:
        d = json.loads(row.assessment_result)
    except Exception:
        d = ast.literal_eval(row.assessment_result)

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

print("invalid assessment cells:", int(A_INVALID.sum()), "/", A_INVALID.size)

# ---- cell 5 (code) ----
# Exact V3.1 dataset-derived topic/style parser
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
    return [k for k,rx in TOPIC_RX.items() if rx.search(str(text))]

def make_skeleton(text):
    s = clean_text(text)
    for rx in TOPIC_RX.values():
        s = rx.sub(" <topic> ", s)
    s = re.sub(r"\b\d+\b", " <num> ", s)
    return re.sub(r"\s+", " ", s).strip()

def template_token(skeleton):
    return "tpl_" + hashlib.md5(skeleton.encode("utf-8")).hexdigest()[:10]

chat["timestamp"] = pd.to_datetime(chat["timestamp"], errors="coerce")
chat["clean"] = chat["user_chat_text"].fillna("").map(clean_text)
chat["is_noise"] = chat["user_chat_text"].fillna("").map(lambda x: bool(NOISE_RX.search(str(x))))
chat["topics"] = chat["user_chat_text"].fillna("").map(extract_topics)
chat["skeleton"] = chat["user_chat_text"].fillna("").map(make_skeleton)
for name,rx in STYLE_RX.items():
    chat["sty_"+name] = chat["user_chat_text"].fillna("").map(lambda x,r=rx: int(bool(r.search(str(x)))))

chat = chat.sort_values(["user_id","timestamp","chat_id"]).reset_index(drop=True)
user_chat = {u:g.copy() for u,g in chat.groupby("user_id", sort=False)}
print("administrative/noise rows:", int(chat["is_noise"].sum()), "/", len(chat))

# ---- cell 6 (code) ----
# V3.1 template-aware sparse representation + structured user features
USER_FEATURES = np.zeros((len(all_users), 10 + 10 + 5 + len(STYLES)*2 + len(TOPICS)*5), np.float32)

def build_user(u):
    ui = uid2idx[u]
    g = user_chat.get(u)
    if g is None:
        learn = pd.DataFrame(columns=chat.columns)
        g = learn
    else:
        learn = g[~g["is_noise"]].copy()
        if len(learn) == 0:
            learn = g.copy()

    parts = []
    n = len(learn)
    for j,row in enumerate(learn.itertuples(index=False)):
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
            if getattr(row, "sty_"+sty):
                parts.append("style_" + sty)

    for j,skill in enumerate(SKILL_NAMES):
        v = int(round(float(A[ui,j])))
        band = "low" if v <= 1 else ("mid" if v <= 3 else "high")
        parts += [f"skill_{skill}_{v}", f"skill_{skill}_bin_{band}"]

    feats = list(A[ui]) + list(A_INVALID[ui])
    total = len(g)
    learning_count = len(learn)
    noise_count = int(g["is_noise"].sum()) if total else 0
    span = 0.0
    if total >= 2 and g["timestamp"].notna().sum() >= 2:
        span = float((g["timestamp"].max() - g["timestamp"].min()).total_seconds() / 86400.0)
    feats += [total, learning_count, noise_count, span, min(total,6)]

    for sty in STYLES:
        feats.append(float(learn["sty_"+sty].sum()) if learning_count else 0.0)
    for sty in STYLES:
        feats.append(float(learn.iloc[-1]["sty_"+sty]) if learning_count else 0.0)

    rows = list(learn.itertuples(index=False))
    nfull = len(rows)
    for tp in TOPICS:
        count = rec_count = last_pos = 0.0
        latest = 0.0
        for j,row in enumerate(rows):
            if tp in row.topics:
                count += 1.0
                rec_count += float(np.exp(-0.7 * (nfull - 1 - j)))
                last_pos = max(last_pos, (j+1) / max(nfull,1))
        if rows and tp in rows[-1].topics:
            latest = 1.0
        feats += [count, rec_count, latest, last_pos, float(count > 0)]

    USER_FEATURES[ui] = np.asarray(feats, np.float32)
    return " ".join(parts)

texts = np.asarray([build_user(u) for u in all_users], dtype=object)
assert USER_FEATURES.shape == (5000, 134)
assert np.isfinite(USER_FEATURES).all()
print("USER_FEATURES:", USER_FEATURES.shape)
print("example augmented text:", texts[1][:500])

# ---- cell 7 (markdown) ----
# ## Frozen public-proven backbone
#
# The V3.1 sparse backbone, V3.1/V7 calibrator, Consensus3 profile, and V25 public-tested actions are reconstructed exactly before V31 adds any new information.

# ---- cell 8 (code) ----
# Exact public-validated V3.1 sparse backbone.
t0=time.time()
word_vec=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_df=.995,sublinear_tf=True,strip_accents="unicode",max_features=(2000 if V27_SMOKE else 50000),dtype=np.float32)
char_vec=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=2,sublinear_tf=True,max_features=(3000 if V27_SMOKE else 60000),dtype=np.float32)
X_WORD=word_vec.fit_transform(texts)
X_CHAR=char_vec.fit_transform(texts)
X=hstack([X_WORD,X_CHAR],format="csr")
def sparse_ref(target):
    oo=np.zeros_like(y,np.float32); te=np.zeros((N_TEST,N_MODULES),np.float32)
    for tr,va in ref_folds:
        m=Ridge(alpha=RIDGE_ALPHA,solver="lsqr",tol=2e-3,max_iter=1000); m.fit(X[tr],target[tr]); oo[va]=m.predict(X[va]).astype(np.float32); te+=m.predict(X[N_TRAIN:]).astype(np.float32)/N_FOLDS
    return oo,te
raw_oof,raw_test=sparse_ref(y); top_oof,top_test=sparse_ref(Y_TOP)
base_oof=(.9*raw_oof+.1*top_oof).astype(np.float32); base_test=(.9*raw_test+.1*top_test).astype(np.float32)
print("X",X.shape,"base OOF",round(ndcg5(y,base_oof),6),"sec",round(time.time()-t0,1))

# ---- cell 9 (code) ----
# Exact public-winning V3.1/V7 relevance calibration path.
def make_cal(seed,n_estimators=None):
    return lgb.LGBMClassifier(
        n_estimators=int(LGB_ESTIMATORS if n_estimators is None else n_estimators),
        learning_rate=.03,num_leaves=15,min_child_samples=50,
        reg_alpha=.5,reg_lambda=5.,colsample_bytree=.7,
        random_state=int(seed),n_jobs=4,verbosity=-1
    )

def fit_bin(seed,folds,Xtr,Xout,targets):
    oo=np.zeros((len(targets),N_MODULES),np.float32)
    out=np.zeros((Xout[0].shape[0],N_MODULES),np.float32)
    its={mid:[] for mid in MODULES}
    for m,mid in enumerate(MODULES):
        target=targets[:,m].astype(np.int32)
        for f,(tr,va) in enumerate(folds):
            clf=make_cal(seed+f+31*m)
            clf.fit(
                Xtr[m][tr],target[tr],
                eval_set=[(Xtr[m][va],target[va])],
                callbacks=[lgb.early_stopping(LGB_EARLY_STOP,verbose=False)]
            )
            it=int(clf.best_iteration_ or LGB_ESTIMATORS)
            its[mid].append(it)
            oo[va,m]=clf.predict_proba(Xtr[m][va],num_iteration=it)[:,1].astype(np.float32)
        final_iters=max(20,int(np.median(its[mid])))
        full=make_cal(seed+997+31*m,final_iters)
        full.fit(Xtr[m],target)
        out[:,m]=full.predict_proba(Xout[m])[:,1].astype(np.float32)
    return oo,out,its

META_TR=[np.column_stack([base_oof[:,m],USER_FEATURES[:N_TRAIN]]).astype(np.float32) for m in range(N_MODULES)]
META_TE=[np.column_stack([base_test[:,m],USER_FEATURES[N_TRAIN:]]).astype(np.float32) for m in range(N_MODULES)]
meta_oof,meta_test,best_iters=fit_bin(SEED,ref_folds,META_TR,META_TE,Y_BIN)
cal045_oof=(.55*base_oof+.45*meta_oof).astype(np.float32)
cal045_test=(.55*base_test+.45*meta_test).astype(np.float32)

print("sparse OOF ",round(ndcg5(y,base_oof),6))
print("meta OOF   ",round(ndcg5(y,meta_oof),6))
print("cal045 OOF ",round(ndcg5(y,cal045_oof),6),"public",PUBLIC_CAL045_LB)
print("median calibrator iterations:",{k:int(np.median(v)) for k,v in best_iters.items()})

# ---- cell 10 (code) ----
# Exact V12.2 Consensus3 profile. No weight search in V24.
CONSENSUS3_OVERRIDES = {
    "M_002": 0.44,
    "M_006": 0.55,
    "M_007": 0.60,
    "M_010": 0.40,
    "M_011": 0.55,
    "M_012": 0.55,
    "M_015": 0.44,
    "M_017": 0.53,
}

def consensus3_vector():
    w = np.full(N_MODULES, REFERENCE_W, np.float32)
    for mid, val in CONSENSUS3_OVERRIDES.items():
        if mid not in MODULES:
            raise KeyError(mid)
        w[MODULES.index(mid)] = float(val)
    return w

CONS_W = consensus3_vector()
consensus3_oof = ((1.0 - CONS_W) * base_oof + CONS_W * meta_oof).astype(np.float32)
consensus3_test = ((1.0 - CONS_W) * base_test + CONS_W * meta_test).astype(np.float32)

ref_fold_id = np.full(N_TRAIN, -1, np.int16)
for _f, (_, _va) in enumerate(ref_folds):
    ref_fold_id[_va] = _f
assert (ref_fold_id >= 0).all()

print("Consensus3 OOF:", round(ndcg5(y, consensus3_oof), 6))
print("Consensus3 frozen weights:", {m: float(CONS_W[i]) for i,m in enumerate(MODULES)})

# ---- cell 11 (markdown) ----
# ## Reconstruct the V25 protected-action package
#
# These rows remain protected during both V30-control and V31 deployment. They are not used as supervised training labels.

# ---- cell 12 (code) ----
TEST_UID_TO_ROW = {u:i for i,u in enumerate(test["user_id"].tolist())}
MID_TO_COL = {m:i for i,m in enumerate(MODULES)}

KNOWN_PB_SWAPS = [
    ("U_4047", "M_009", "M_007"),  # promote EDA over ML
    ("U_4953", "M_016", "M_005"),  # promote Version Control over MLOps
]


def swap_score_values(scores, row_idx, higher_mid, lower_mid, require_order=True):
    """Swap two numeric score values so only those two ranking positions exchange."""
    out = scores
    hi = MID_TO_COL[higher_mid]
    lo = MID_TO_COL[lower_mid]
    if require_order and not (float(out[row_idx, hi]) > float(out[row_idx, lo])):
        raise AssertionError(
            f"Expected {higher_mid}>{lower_mid} before swap, got "
            f"{out[row_idx, hi]:.8f} <= {out[row_idx, lo]:.8f}"
        )
    out[row_idx, hi], out[row_idx, lo] = out[row_idx, lo], out[row_idx, hi]
    return out

pb_test = consensus3_test.copy()
known_rows = []
for uid, higher_mid, lower_mid in KNOWN_PB_SWAPS:
    r = TEST_UID_TO_ROW[uid]
    hi, lo = MID_TO_COL[higher_mid], MID_TO_COL[lower_mid]
    margin = float(pb_test[r,hi] - pb_test[r,lo])
    known_rows.append({
        "user_id": uid, "higher_before": higher_mid, "promote": lower_mid,
        "consensus_margin": margin,
    })
    if not V30_SMOKE:
        assert 0 < margin < 0.02, f"PB reconstruction drift for {uid}: margin={margin}"
        swap_score_values(pb_test, r, higher_mid, lower_mid)
    else:
        if margin > 0:
            swap_score_values(pb_test, r, higher_mid, lower_mid, require_order=False)

known_pb_diag = pd.DataFrame(known_rows)
display(known_pb_diag)

# Exactly two users must change Top-5 ordering versus Consensus3.
cons_top5 = top5_indices(consensus3_test)
pb_top5 = top5_indices(pb_test)
changed_rows = np.flatnonzero(np.any(cons_top5 != pb_top5, axis=1))
changed_uids = test.iloc[changed_rows]["user_id"].tolist()
if not V30_SMOKE:
    assert set(changed_uids) == {"U_4047", "U_4953"}, changed_uids
print("PB reference changes:", changed_uids)

# ---- cell 13 (code) ----
# Canonical V25 package. Notation: higher_before was above promoted before V25.
V25_CORE_ACTIONS = [
    {"user_id":"U_4074", "higher_before":"M_014", "promoted":"M_003", "expected_margin":0.000682443},
    {"user_id":"U_4176", "higher_before":"M_003", "promoted":"M_017", "expected_margin":0.000203669},
    {"user_id":"U_4491", "higher_before":"M_003", "promoted":"M_002", "expected_margin":0.001700282},
    {"user_id":"U_4567", "higher_before":"M_014", "promoted":"M_012", "expected_margin":0.000054896},
    {"user_id":"U_4836", "higher_before":"M_003", "promoted":"M_002", "expected_margin":0.000934988},
    {"user_id":"U_4929", "higher_before":"M_003", "promoted":"M_002", "expected_margin":0.000661612},
]
V25_RELAXED_ACTIONS = [
    {"user_id":"U_4030", "higher_before":"M_007", "promoted":"M_003", "expected_margin":0.002789915},
    {"user_id":"U_4650", "higher_before":"M_003", "promoted":"M_012", "expected_margin":0.002036035},
    {"user_id":"U_4731", "higher_before":"M_014", "promoted":"M_006", "expected_margin":0.001383066},
]
EXPECTED_PRE_SWAP_RANKS = {
    "U_4074": (5,6),
    "U_4176": (2,3),
    "U_4491": (4,5),
    "U_4567": (3,4),
    "U_4836": (4,5),
    "U_4929": (4,5),
}


def full_rank_matrix(scores):
    order = np.argsort(-np.asarray(scores), axis=1, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.int16)
    rows = np.arange(len(order))[:,None]
    ranks[rows, order] = np.arange(1, order.shape[1]+1, dtype=np.int16)[None,:]
    return ranks


def apply_promotion(scores, action, strict_geometry=True):
    out = scores.copy()
    uid = action["user_id"]
    row = TEST_UID_TO_ROW[uid]
    hi = MID_TO_COL[action["higher_before"]]
    pr = MID_TO_COL[action["promoted"]]
    margin = float(out[row, hi] - out[row, pr])
    if strict_geometry:
        if not margin > 0:
            raise AssertionError(
                f"V25 geometry drift for {uid}: expected {action['higher_before']}>{action['promoted']} "
                f"before promotion, margin={margin:.9f}"
            )
        if abs(margin - float(action["expected_margin"])) > GEOMETRY_ATOL:
            raise AssertionError(
                f"V25 frozen-margin drift for {uid}: got {margin:.9f}, "
                f"expected {action['expected_margin']:.9f}, atol={GEOMETRY_ATOL}"
            )
    # Idempotent safety if strict geometry is disabled for debugging only.
    if margin > 0:
        out[row, hi], out[row, pr] = out[row, pr], out[row, hi]
    return out, margin

pb_ranks_before_core = full_rank_matrix(pb_test)
core_geometry_rows = []
v25_anchor_test = pb_test.copy()
for action in V25_CORE_ACTIONS:
    uid = action["user_id"]
    r = TEST_UID_TO_ROW[uid]
    hi = MID_TO_COL[action["higher_before"]]
    pr = MID_TO_COL[action["promoted"]]
    rank_pair = (int(pb_ranks_before_core[r,hi]), int(pb_ranks_before_core[r,pr]))
    expected_pair = EXPECTED_PRE_SWAP_RANKS[uid]
    if V27_STRICT_GEOMETRY:
        assert rank_pair == expected_pair, (uid, rank_pair, expected_pair)
    v25_anchor_test, margin = apply_promotion(v25_anchor_test, action, strict_geometry=V27_STRICT_GEOMETRY)
    core_geometry_rows.append({
        **action,
        "rank_higher_before": rank_pair[0],
        "rank_promoted_before": rank_pair[1],
        "current_margin": margin,
        "margin_abs_error": abs(margin-float(action["expected_margin"])),
    })

core_geometry = pd.DataFrame(core_geometry_rows)
display(core_geometry)

# V25 primary must differ from Consensus3 only on the 2 PB + 6 core users.
cons_top = top5_indices(consensus3_test)
v25_top = top5_indices(v25_anchor_test)
changed_rows_v25 = np.flatnonzero(np.any(cons_top != v25_top, axis=1))
changed_users_v25 = test.iloc[changed_rows_v25]["user_id"].tolist()
expected_v25_users = {"U_4047","U_4953", *[a["user_id"] for a in V25_CORE_ACTIONS]}
if not V30_SMOKE:
    assert set(changed_users_v25) == expected_v25_users, changed_users_v25
    assert len(changed_users_v25) == 8
print("V25 anchor reconstructed; changed users vs Consensus3:", changed_users_v25)

# ---- cell 14 (markdown) ----
# ## V31 supervised state representation
#
# The new block is intentionally low-dimensional and empirical. The latest learning-message intent is represented as a **topic-set signature** rather than an exact text/template identity. Candidate readiness is computed from organizer-provided assessments using the same module-to-skill map as V30, then discretized into fixed low/mid/high bands on the 0–5 scale.
#
# Target statistics are always learned inside the relevant training partition. For outer-training rows they are generated by a second group-disjoint cross-fit, which is the critical leakage barrier for V31.

# ---- cell 15 (code) ----
# Module-native lexical / intent / readiness surfaces inherited from V30 Base.
MODULE_TEXT = (
    modules.set_index("module_id").loc[MODULES, "module_name"].astype(str) + " " +
    modules.set_index("module_id").loc[MODULES, "description_and_syllabus"].astype(str) + " " +
    modules.set_index("module_id").loc[MODULES, "prerequisite_level"].astype(str)
).tolist()
MODULE_WORD = word_vec.transform(MODULE_TEXT)
MODULE_CHAR = char_vec.transform(MODULE_TEXT)
LEX_WORD = (X_WORD @ MODULE_WORD.T).toarray().astype(np.float32)
LEX_CHAR = (X_CHAR @ MODULE_CHAR.T).toarray().astype(np.float32)

MODULE_TOPIC_MAP = {
    "M_001":["excel"], "M_002":["python"], "M_003":["sql"],
    "M_004":["webscrape","python"], "M_005":["git"], "M_006":["statistics"],
    "M_007":["eda","python","statistics"], "M_008":["dashboard","sql","excel"],
    "M_009":["ml","ml_eval","eda","statistics"], "M_010":["cv","ml"],
    "M_011":["nlp","ml"], "M_012":["genai","nlp"], "M_013":["prompt","genai"],
    "M_014":["automation","genai"], "M_015":["nocode","genai"],
    "M_016":["mlops","ml","python","git"], "M_017":["career"],
}
MODULE_SKILL_MAP = {
    "M_001":[], "M_002":[0], "M_003":[1], "M_004":[0], "M_005":[9],
    "M_006":[2], "M_007":[0,2,3], "M_008":[1], "M_009":[2,3,4,5],
    "M_010":[4,6], "M_011":[4], "M_012":[4,7], "M_013":[7],
    "M_014":[7,8], "M_015":[7], "M_016":[0,4,5,9], "M_017":[8,9],
}

MODULE_COUNT = np.zeros((len(all_users),N_MODULES),np.float32)
MODULE_RECENCY = np.zeros_like(MODULE_COUNT)
MODULE_LATEST = np.zeros_like(MODULE_COUNT)
LATEST_GROUP = np.empty(len(all_users),dtype=object); LATEST_GROUP[:] = "no_chat"
LATEST_INTENT_KEY = np.empty(len(all_users),dtype=object); LATEST_INTENT_KEY[:] = "intent:no_chat"

for u in all_users:
    ui=uid2idx[u]
    g=user_chat.get(u)
    if g is None or len(g)==0:
        continue
    learn=g[~g["is_noise"]].copy()
    if len(learn)==0:
        learn=g.copy()
    rows=list(learn.itertuples(index=False))
    if rows:
        LATEST_GROUP[ui]=rows[-1].skeleton or "empty"
        tps=sorted(set(rows[-1].topics))
        LATEST_INTENT_KEY[ui] = "intent:" + ("+".join(tps) if tps else "none")
    n=len(rows)
    for j,row in enumerate(rows):
        dist=n-1-j; w=float(np.exp(-0.7*dist)); row_topics=set(row.topics)
        for m,mid in enumerate(MODULES):
            hits=sum(tp in row_topics for tp in MODULE_TOPIC_MAP[mid])
            if hits:
                MODULE_COUNT[ui,m]+=hits
                MODULE_RECENCY[ui,m]+=w*hits
                if dist==0:
                    MODULE_LATEST[ui,m]=1.0

MODULE_READINESS=np.zeros((len(all_users),N_MODULES),np.float32)
MODULE_INVALID=np.zeros_like(MODULE_READINESS)
for m,mid in enumerate(MODULES):
    idx=MODULE_SKILL_MAP[mid]
    if idx:
        MODULE_READINESS[:,m]=A[:,idx].mean(1)
        MODULE_INVALID[:,m]=A_INVALID[:,idx].mean(1)
    else:
        MODULE_READINESS[:,m]=A.mean(1)
        MODULE_INVALID[:,m]=A_INVALID.mean(1)

# Fixed readiness discretization consistent with the V3.1 low/mid/high assessment bins.
_ready_round=np.floor(MODULE_READINESS+0.5)
MODULE_READY_BAND=np.where(_ready_round<=1,0,np.where(_ready_round<=3,1,2)).astype(np.int8)

TRAIN_GROUP=np.asarray(LATEST_GROUP[:N_TRAIN],dtype=object)
TRAIN_INTENT=np.asarray(LATEST_INTENT_KEY[:N_TRAIN],dtype=object)
TEST_INTENT=np.asarray(LATEST_INTENT_KEY[N_TRAIN:],dtype=object)

prtxt=modules.set_index("module_id")["prerequisite_level"].astype(str).to_dict()
def module_level(mid):
    s=prtxt[mid].lower()
    if "pemula" in s: return 0
    if "menengah" in s: return 1
    if "lanjutan" in s: return 2
    if "ahli" in s: return 3
    return 4
MODULE_LEVEL=np.asarray([module_level(m) for m in MODULES],np.int16)
MODULE_LEVEL_ONEHOT=np.eye(5,dtype=np.float32)[MODULE_LEVEL]

# Ordinal LambdaRank labels: 0=irrelevant, 1=#6 ... 6=#1.
ORDINAL_GRADES=np.zeros_like(y,dtype=np.int16)
for i,row in enumerate(y):
    pos=np.flatnonzero(row>0)
    order=pos[np.argsort(-row[pos],kind="mergesort")]
    for rank,m in enumerate(order,1):
        ORDINAL_GRADES[i,m]=max(1,7-rank)
LABEL_GAIN=[0.0,0.25,0.40,0.55,0.70,0.85,1.00]

def user_strata_from_y(yy):
    yy=np.asarray(yy)
    k=(yy>0).sum(1).astype(int)
    top=np.argmax(yy,axis=1).astype(int)
    return (top*4+np.clip(k-3,0,3)).astype(int)

def make_group_folds_local(yy,groups,n_splits,seed):
    yy=np.asarray(yy); groups=np.asarray(groups,dtype=object)
    ns=min(int(n_splits),len(np.unique(groups)))
    if ns<2:
        sk=StratifiedKFold(n_splits=2,shuffle=True,random_state=int(seed))
        return list(sk.split(np.zeros(len(yy)),user_strata_from_y(yy)))
    sg=StratifiedGroupKFold(n_splits=ns,shuffle=True,random_state=int(seed))
    folds=list(sg.split(np.zeros(len(yy)),user_strata_from_y(yy),groups=groups))
    for tr,va in folds:
        assert len(set(groups[tr]).intersection(set(groups[va])))==0
    return folds

def make_group_folds_global(rows,n_splits,seed):
    rows=np.asarray(rows,int)
    local=make_group_folds_local(y[rows],TRAIN_GROUP[rows],n_splits,seed)
    return [(rows[tr],rows[va]) for tr,va in local]

print("catalog lexical matrices:",LEX_WORD.shape,LEX_CHAR.shape)
print("latest template groups:",len(np.unique(TRAIN_GROUP)))
print("latest intent signatures train/test:",len(np.unique(TRAIN_INTENT)),len(np.unique(TEST_INTENT)))
print("test intent signatures unseen in train:",len(set(TEST_INTENT)-set(TRAIN_INTENT)))
print("readiness band counts:",dict(zip(*np.unique(MODULE_READY_BAND[:N_TRAIN],return_counts=True))))
print("ordinal grade counts:",dict(zip(*np.unique(ORDINAL_GRADES,return_counts=True))))

# ---- cell 16 (code) ----
# Strict fold-safe hierarchical Intent-State target encoding.
INTENTSTATE_KEYS = [
    "IS_PARENT_SEEN","IS_PARENT_SUPPORT_LOG","IS_PARENT_CONF",
    "IS_PARENT_POS","IS_PARENT_ORD","IS_PARENT_RAW",
    "IS_CHILD_SUPPORT_LOG","IS_CHILD_CONF",
    "IS_CHILD_POS","IS_CHILD_ORD","IS_CHILD_RAW","IS_CHILD_TOP5",
    "IS_POS_LOG_LIFT_GLOBAL","IS_ORD_DELTA_GLOBAL","IS_ORD_DELTA_PARENT","IS_RAW_DELTA_PARENT",
]

def _intent_targets(y_fit,grades_fit):
    y_fit=np.asarray(y_fit,np.float32); grades_fit=np.asarray(grades_fit,np.int16)
    return {
        "pos":(y_fit>0).astype(np.float32),
        "ord":(grades_fit.astype(np.float32)/6.0),
        "raw":y_fit.astype(np.float32),
        "top5":(grades_fit>=2).astype(np.float32),
    }

def fit_intentstate_bundle(y_fit,grades_fit,intent_fit,ready_band_fit):
    y_fit=np.asarray(y_fit,np.float32)
    grades_fit=np.asarray(grades_fit,np.int16)
    intent_fit=np.asarray(intent_fit,dtype=object)
    ready_band_fit=np.asarray(ready_band_fit,np.int8)
    n=len(y_fit)
    assert grades_fit.shape==y_fit.shape==ready_band_fit.shape
    T=_intent_targets(y_fit,grades_fit)
    glob={k:v.mean(0).astype(np.float32) for k,v in T.items()}

    parent={}
    child={}
    for key in np.unique(intent_fit):
        idx=np.flatnonzero(intent_fit==key)
        ns=int(len(idx))
        vals={}
        for name,M in T.items():
            sm=(M[idx].sum(0)+INTENT_PARENT_STRENGTH*glob[name])/(ns+INTENT_PARENT_STRENGTH)
            vals[name]=sm.astype(np.float32)
        parent[str(key)]={"n":ns,**vals}

        for m in range(N_MODULES):
            rb=ready_band_fit[idx,m]
            for b in (0,1,2):
                local=idx[rb==b]
                nc=int(len(local))
                if nc==0:
                    continue
                cv={}
                for name,M in T.items():
                    prior=float(vals[name][m])
                    cv[name]=float((M[local,m].sum()+INTENT_CHILD_STRENGTH*prior)/(nc+INTENT_CHILD_STRENGTH))
                child[(str(key),m,int(b))]={"n":nc,**cv}

    return {"n_fit":n,"global":glob,"parent":parent,"child":child}

def apply_intentstate_bundle(bundle,intent_keys,ready_band):
    intent_keys=np.asarray(intent_keys,dtype=object)
    ready_band=np.asarray(ready_band,np.int8)
    n=len(intent_keys)
    assert ready_band.shape==(n,N_MODULES)
    out={k:np.zeros((n,N_MODULES),np.float32) for k in INTENTSTATE_KEYS}
    glob=bundle["global"]
    logden=np.log1p(max(1,int(bundle["n_fit"])))

    for i,key0 in enumerate(intent_keys):
        key=str(key0)
        p=bundle["parent"].get(key)
        parent_seen=float(p is not None)
        if p is None:
            pn=0
            pvals={name:glob[name] for name in ("pos","ord","raw","top5")}
        else:
            pn=int(p["n"])
            pvals={name:p[name] for name in ("pos","ord","raw","top5")}
        pconf=pn/(pn+INTENT_PARENT_STRENGTH) if pn else 0.0
        pslog=np.log1p(pn)/logden if logden>0 else 0.0

        for m in range(N_MODULES):
            c=bundle["child"].get((key,m,int(ready_band[i,m])))
            if c is None:
                cn=0
                cpos=float(pvals["pos"][m]); cord=float(pvals["ord"][m])
                craw=float(pvals["raw"][m]); ctop5=float(pvals["top5"][m])
            else:
                cn=int(c["n"]); cpos=float(c["pos"]); cord=float(c["ord"])
                craw=float(c["raw"]); ctop5=float(c["top5"])
            cconf=cn/(cn+INTENT_CHILD_STRENGTH) if cn else 0.0
            cslog=np.log1p(cn)/logden if logden>0 else 0.0
            gpos=float(glob["pos"][m]); gord=float(glob["ord"][m]); graw=float(glob["raw"][m])
            pord=float(pvals["ord"][m]); praw=float(pvals["raw"][m])

            out["IS_PARENT_SEEN"][i,m]=parent_seen
            out["IS_PARENT_SUPPORT_LOG"][i,m]=pslog
            out["IS_PARENT_CONF"][i,m]=pconf
            out["IS_PARENT_POS"][i,m]=float(pvals["pos"][m])
            out["IS_PARENT_ORD"][i,m]=pord
            out["IS_PARENT_RAW"][i,m]=praw
            out["IS_CHILD_SUPPORT_LOG"][i,m]=cslog
            out["IS_CHILD_CONF"][i,m]=cconf
            out["IS_CHILD_POS"][i,m]=cpos
            out["IS_CHILD_ORD"][i,m]=cord
            out["IS_CHILD_RAW"][i,m]=craw
            out["IS_CHILD_TOP5"][i,m]=ctop5
            out["IS_POS_LOG_LIFT_GLOBAL"][i,m]=float(np.clip(np.log((cpos+1e-4)/(gpos+1e-4)),-2.0,2.0))
            out["IS_ORD_DELTA_GLOBAL"][i,m]=cord-gord
            out["IS_ORD_DELTA_PARENT"][i,m]=cord-pord
            out["IS_RAW_DELTA_PARENT"][i,m]=craw-praw

    for k,v in out.items():
        if not np.isfinite(v).all():
            raise ValueError(f"non-finite Intent-State feature {k}")
    return out

def crossfit_intentstate_features(y_local,grades_local,intent_local,ready_band_local,groups_local,seed):
    y_local=np.asarray(y_local,np.float32)
    grades_local=np.asarray(grades_local,np.int16)
    intent_local=np.asarray(intent_local,dtype=object)
    ready_band_local=np.asarray(ready_band_local,np.int8)
    groups_local=np.asarray(groups_local,dtype=object)
    out={k:np.zeros_like(y_local,np.float32) for k in INTENTSTATE_KEYS}
    folds=make_group_folds_local(y_local,groups_local,INTENT_INNER_FOLDS,seed)
    seen=np.zeros(len(y_local),bool)
    for tr,va in folds:
        assert len(set(groups_local[tr]).intersection(set(groups_local[va])))==0
        bundle=fit_intentstate_bundle(y_local[tr],grades_local[tr],intent_local[tr],ready_band_local[tr])
        vv=apply_intentstate_bundle(bundle,intent_local[va],ready_band_local[va])
        for k in INTENTSTATE_KEYS:
            out[k][va]=vv[k]
        seen[va]=True
    assert seen.all(), "Intent-State crossfit did not cover every training row"
    return out

def intentstate_support_audit(bundle,intent_keys,ready_band):
    intent_keys=np.asarray(intent_keys,dtype=object); ready_band=np.asarray(ready_band,np.int8)
    supports=[]; parent_support=[]; parent_seen=[]
    for i,key0 in enumerate(intent_keys):
        key=str(key0); p=bundle["parent"].get(key)
        pn=int(p["n"]) if p is not None else 0
        for m in range(N_MODULES):
            c=bundle["child"].get((key,m,int(ready_band[i,m])))
            supports.append(int(c["n"]) if c is not None else 0)
            parent_support.append(pn); parent_seen.append(int(p is not None))
    s=np.asarray(supports,int); ps=np.asarray(parent_support,int); pseen=np.asarray(parent_seen,int)
    rows=[]
    for thr in [1,5,10,20,30,50]:
        rows.append({"level":"child","threshold":thr,"coverage":float(np.mean(s>=thr))})
    rows.append({"level":"child","threshold":-1,"coverage":1.0,"median_support":float(np.median(s)),"p10_support":float(np.quantile(s,.10)),"p25_support":float(np.quantile(s,.25))})
    rows.append({"level":"parent","threshold":1,"coverage":float(np.mean(pseen>0)),"median_support":float(np.median(ps))})
    return pd.DataFrame(rows)

print("strict nested Intent-State TE helpers ready; features:",len(INTENTSTATE_KEYS))

# ---- cell 17 (markdown) ----
# ## Exact V30 Base ranker + V31 feature branch
#
# The Base portion below preserves the V30 long-format feature order exactly. Both control and candidate use the same LambdaRank hyperparameters, same outer folds, same anchor, same `alpha`, and the same ranker seed inside each fold. The only experimental difference is whether the Intent-State columns are present.

# ---- cell 18 (markdown) ----
# ## V36 proposal surfaces and Hybrid34 state machine
#
# The public result establishes four states from the path `pre-V30 anchor → V30 → V31`:
# - **S0:** V30 stable and V31 stable;
# - **S1:** V30 stable but V31 changes it — public-toxic region, never routed by V36;
# - **S2:** V30 changed the prior anchor but V31 abstained — unresolved corridor;
# - **S3:** both V30 and V31 disagree — this is the 34-user public-proven Hybrid34 region and is immutable.
#
# XE-NDCG only proposes an alternative Top-5. It never directly replaces Hybrid34.

# ---- cell 19 (code) ----
def rank_matrix(scores):
    order=np.argsort(-np.asarray(scores),axis=1,kind="mergesort")
    ranks=np.empty_like(order,dtype=np.int16)
    ranks[np.arange(len(order))[:,None],order]=np.arange(1,N_MODULES+1,dtype=np.int16)[None,:]
    return ranks


def top5_order_change_rate(a,b):
    return float(np.mean(np.any(top5_indices(a)!=top5_indices(b),axis=1)))


def build_long_matrix_v31(rows_global,anchor,base,meta,intentstate=None):
    rows_global=np.asarray(rows_global,int); anchor=np.asarray(anchor,np.float32)
    base=np.asarray(base,np.float32); meta=np.asarray(meta,np.float32); n=len(rows_global)

    ranks=rank_matrix(anchor).astype(np.float32)
    top1=anchor.max(1,keepdims=True)
    sorted_scores=np.sort(anchor,axis=1)[:,::-1]
    cut5=sorted_scores[:,4:5]
    mu=anchor.mean(1,keepdims=True); sd=np.maximum(anchor.std(1,keepdims=True),1e-4)
    z=(anchor-mu)/sd
    order=np.argsort(-anchor,axis=1,kind="mergesort")
    prev_gap=np.zeros_like(anchor); next_gap=np.zeros_like(anchor)
    for i in range(n):
        oo=order[i]; ss=anchor[i,oo]
        pg=np.r_[0.0,ss[:-1]-ss[1:]]
        ng=np.r_[ss[:-1]-ss[1:],0.0]
        prev_gap[i,oo]=pg; next_gap[i,oo]=ng

    base_mats=[
        anchor, ranks/float(N_MODULES), (ranks<=5).astype(np.float32), top1-anchor, anchor-cut5,
        prev_gap,next_gap,z,base,meta,meta-base,
        LEX_WORD[rows_global],LEX_CHAR[rows_global],MODULE_COUNT[rows_global],MODULE_RECENCY[rows_global],
        MODULE_LATEST[rows_global],MODULE_READINESS[rows_global],MODULE_INVALID[rows_global],
    ]
    base_names=[
        "anchor_score","anchor_rank_norm","anchor_top5","top1_minus_score","score_minus_cut5",
        "prev_gap","next_gap","anchor_z","base_score","meta_score","meta_minus_base",
        "lex_word","lex_char","topic_count","topic_recency","topic_latest","readiness","invalid_readiness",
    ]
    pieces=[np.column_stack([m.reshape(-1) for m in base_mats]).astype(np.float32)]
    names=list(base_names)

    pieces.append(np.repeat(USER_FEATURES[rows_global],N_MODULES,axis=0).astype(np.float32))
    names += [f"user_f{i:03d}" for i in range(USER_FEATURES.shape[1])]
    pieces.append(np.tile(np.eye(N_MODULES,dtype=np.float32),(n,1)))
    names += [f"module_{m}" for m in MODULES]
    pieces.append(np.tile(MODULE_LEVEL_ONEHOT,(n,1)).astype(np.float32))
    names += [f"module_level_{i}" for i in range(5)]
    bounds={"base":sum(p.shape[1] for p in pieces)}

    if intentstate is not None:
        mats=[intentstate[k] for k in INTENTSTATE_KEYS]
        pieces.append(np.column_stack([m.reshape(-1) for m in mats]).astype(np.float32))
        names += [k.lower() for k in INTENTSTATE_KEYS]
        bounds["intentstate"]=sum(p.shape[1] for p in pieces)

    XX=np.hstack(pieces).astype(np.float32,copy=False)
    if XX.shape[0] != n*N_MODULES:
        raise AssertionError(XX.shape)
    if len(names)!=XX.shape[1]:
        raise AssertionError((len(names),XX.shape))
    if not np.isfinite(XX).all():
        raise ValueError("non-finite V36 long features")
    return XX,names,bounds


def make_lambdarank(seed):
    return lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg",
        n_estimators=RANKER_ESTIMATORS, learning_rate=RANKER_LEARNING_RATE,
        num_leaves=15, min_child_samples=80,
        reg_alpha=1.0, reg_lambda=10.0,
        colsample_bytree=0.78, subsample=0.90, subsample_freq=1,
        max_bin=127, lambdarank_truncation_level=8,
        label_gain=LABEL_GAIN, random_state=int(seed), n_jobs=4,
        verbosity=-1, force_col_wise=True,
    )


def make_xendcg(seed):
    return lgb.LGBMRanker(
        objective="rank_xendcg", metric="ndcg",
        n_estimators=RANKER_ESTIMATORS, learning_rate=RANKER_LEARNING_RATE,
        num_leaves=15, min_child_samples=80,
        reg_alpha=1.0, reg_lambda=10.0,
        colsample_bytree=0.78, subsample=0.90, subsample_freq=1,
        max_bin=127, label_gain=LABEL_GAIN,
        random_state=int(seed), n_jobs=4, verbosity=-1, force_col_wise=True,
    )


def make_yetirank(seed):
    return CatBoostRanker(
        loss_function="YetiRank:mode=NDCG;top=5;dcg_type=Base;dcg_denominator=LogPosition",
        eval_metric="NDCG:top=5;type=Base;denominator=LogPosition",
        iterations=YETI_ITERATIONS, learning_rate=YETI_LEARNING_RATE,
        depth=5, l2_leaf_reg=11.0, random_strength=0.30,
        random_seed=int(seed), task_type="CPU", thread_count=4,
        allow_writing_files=False, verbose=False,
    )


def fit_yeti_predict(Xtr,Xva,grades_tr,n_users,seed):
    gains=np.asarray(LABEL_GAIN,np.float32)[np.asarray(grades_tr,np.int16)]
    pool=Pool(
        Xtr,label=gains.reshape(-1).astype(np.float32),
        group_id=np.repeat(np.arange(int(n_users),dtype=np.int32),N_MODULES),
    )
    model=make_yetirank(seed); model.fit(pool,verbose=False)
    pred=model.predict(Xva).reshape(-1,N_MODULES).astype(np.float32)
    return model,pred


def fit_ranker_predict(kind,Xtr,Xva,grades_tr,n_users,seed):
    model = make_lambdarank(seed) if kind=="lambda" else make_xendcg(seed)
    group=np.full(int(n_users),N_MODULES,dtype=np.int32)
    model.fit(Xtr,grades_tr.reshape(-1).astype(np.int32),group=group)
    pred=model.predict(Xva).reshape(-1,N_MODULES).astype(np.float32)
    return model,pred


def align_ranker_to_anchor(raw_ranker,anchor):
    raw=np.asarray(raw_ranker,np.float32); anchor=np.asarray(anchor,np.float32)
    rmu=raw.mean(1,keepdims=True); rsd=np.maximum(raw.std(1,keepdims=True),1e-5)
    amu=anchor.mean(1,keepdims=True); asd=np.maximum(anchor.std(1,keepdims=True),1e-5)
    return (amu+(raw-rmu)*(asd/rsd)).astype(np.float32)


def blend_residual(anchor,raw_ranker,alpha=ALPHA):
    aligned=align_ranker_to_anchor(raw_ranker,anchor)
    return (anchor+float(alpha)*(aligned-anchor)).astype(np.float32)


def state_codes(pre_anchor,v30,v31):
    pre_top=top5_indices(pre_anchor); v30_top=top5_indices(v30); v31_top=top5_indices(v31)
    a=np.any(pre_top!=v30_top,axis=1)
    b=np.any(v30_top!=v31_top,axis=1)
    # 0=S0, 1=S1, 2=S2, 3=S3
    return (2*a.astype(np.int8)+b.astype(np.int8)).astype(np.int8)

STATE_NAMES=np.asarray(["S0","S1","S2","S3"],dtype=object)


def hybrid34_surface(pre_anchor,v30,v31):
    st=state_codes(pre_anchor,v30,v31)
    out=np.asarray(v30,np.float32).copy()
    out[st==3]=np.asarray(v31,np.float32)[st==3]
    return out,st


def top5_key_row(scores_row):
    return tuple(np.argsort(-np.asarray(scores_row),kind="mergesort")[:5].astype(int).tolist())


def apply_target_top5(anchor_row,target_top5):
    """Permute only existing score levels over the union needed to realize target Top-5."""
    row=np.asarray(anchor_row,np.float32)
    target=tuple(int(x) for x in target_top5)
    if len(target)!=5 or len(set(target))!=5:
        raise ValueError("target_top5 must contain five unique module indices")
    cur=tuple(np.argsort(-row,kind="mergesort")[:5].astype(int).tolist())
    if cur==target:
        return row.copy()
    union=[]
    for m in cur+target:
        if m not in union:
            union.append(m)
    vals=sorted([float(row[m]) for m in union],reverse=True)
    out=row.copy()
    for m,v in zip(target,vals[:5]):
        out[m]=v
    leftovers=[m for m in union if m not in set(target)]
    leftovers=sorted(leftovers,key=lambda m:(-float(row[m]),m))
    for m,v in zip(leftovers,vals[5:]):
        out[m]=v
    got=tuple(np.argsort(-out,kind="mergesort")[:5].astype(int).tolist())
    if got!=target:
        raise AssertionError((cur,target,got))
    # Exact score multiset is preserved.
    if not np.allclose(np.sort(out),np.sort(row),atol=0,rtol=0):
        raise AssertionError("score multiset changed")
    return out


def cluster_boot_ci(values,groups,seed,B=BOOTSTRAPS):
    values=np.asarray(values,float); groups=np.asarray(groups,dtype=object)
    ug=np.unique(groups); by={g:np.flatnonzero(groups==g) for g in ug}
    rng=np.random.default_rng(int(seed)); z=np.empty(int(B),float)
    for b in range(int(B)):
        sampled=rng.choice(ug,size=len(ug),replace=True)
        idx=np.concatenate([by[g] for g in sampled])
        z[b]=values[idx].mean()
    return tuple(float(x) for x in np.quantile(z,[.025,.975]))

print("V36 V30/V31/XE + Hybrid34 state helpers ready")

# ---- cell 20 (markdown) ----
# ## Fold-matched XE proposal generator
#
# For development, each world holds out 400 users. The remaining 3600 users are split into four shadow folds. Each shadow expert trains on exactly **2700 users**, predicts its 900-user meta fold, and simultaneously predicts the untouched 400-user outer validation set. This makes meta-training proposals and outer-validation proposals come from the same expert training size.
#
# At final deployment, four 2700-user shadows predict Kaggle test. Their consensus is used only to decide whether an XE proposal is stable enough to consider. The final score levels always come from exact full-data Hybrid34.

# ---- cell 21 (code) ----
def _make_cal_local(seed,n_estimators=None):
    return lgb.LGBMClassifier(
        n_estimators=int(LGB_ESTIMATORS if n_estimators is None else n_estimators),
        learning_rate=.03,num_leaves=15,min_child_samples=50,
        reg_alpha=.5,reg_lambda=5.,colsample_bytree=.7,
        random_state=int(seed),n_jobs=4,verbosity=-1
    )

# build_anchor_split_full is deliberately identical to V31/V32.
def build_anchor_split_full(train_rows,val_rows,seed):
    train_rows=np.asarray(train_rows,int); val_rows=np.asarray(val_rows,int)
    local_y=y[train_rows]; local_top=Y_TOP[train_rows]; local_bin=Y_BIN[train_rows]
    folds=make_group_folds_local(local_y,TRAIN_GROUP[train_rows],ANCHOR_INNER_FOLDS,seed)
    n=len(train_rows)

    raw=np.zeros((n,N_MODULES),np.float32); top=np.zeros_like(raw)
    for lt,lv in folds:
        mr=Ridge(alpha=RIDGE_ALPHA,solver="lsqr",tol=2e-3,max_iter=1000)
        mt=Ridge(alpha=RIDGE_ALPHA,solver="lsqr",tol=2e-3,max_iter=1000)
        mr.fit(X[train_rows[lt]],local_y[lt]); mt.fit(X[train_rows[lt]],local_top[lt])
        raw[lv]=mr.predict(X[train_rows[lv]]).astype(np.float32)
        top[lv]=mt.predict(X[train_rows[lv]]).astype(np.float32)
    mr=Ridge(alpha=RIDGE_ALPHA,solver="lsqr",tol=2e-3,max_iter=1000)
    mt=Ridge(alpha=RIDGE_ALPHA,solver="lsqr",tol=2e-3,max_iter=1000)
    mr.fit(X[train_rows],local_y); mt.fit(X[train_rows],local_top)
    base_o=(.9*raw+.1*top).astype(np.float32)
    base_v=(.9*mr.predict(X[val_rows])+.1*mt.predict(X[val_rows])).astype(np.float32)

    meta_o=np.zeros_like(base_o); meta_v=np.zeros((len(val_rows),N_MODULES),np.float32)
    for m,mid in enumerate(MODULES):
        xt=np.column_stack([base_o[:,m],USER_FEATURES[train_rows]]).astype(np.float32)
        xv=np.column_stack([base_v[:,m],USER_FEATURES[val_rows]]).astype(np.float32)
        yy=local_bin[:,m].astype(np.int32); its=[]
        for f,(lt,lv) in enumerate(folds):
            clf=_make_cal_local(seed+31*m+f)
            clf.fit(xt[lt],yy[lt],eval_set=[(xt[lv],yy[lv])],callbacks=[lgb.early_stopping(LGB_EARLY_STOP,verbose=False)])
            it=int(clf.best_iteration_ or LGB_ESTIMATORS); its.append(it)
            meta_o[lv,m]=clf.predict_proba(xt[lv],num_iteration=it)[:,1].astype(np.float32)
        final_it=max(20,int(np.median(its)))
        full=_make_cal_local(seed+997+31*m,final_it); full.fit(xt,yy)
        meta_v[:,m]=full.predict_proba(xv)[:,1].astype(np.float32)

    cons_o=((1.0-CONS_W)*base_o+CONS_W*meta_o).astype(np.float32)
    cons_v=((1.0-CONS_W)*base_v+CONS_W*meta_v).astype(np.float32)
    return {"base_train":base_o,"meta_train":meta_o,"cons_train":cons_o,
            "base_val":base_v,"meta_val":meta_v,"cons_val":cons_v}


def deterministic_train_subset(rows,size,seed):
    rows=np.asarray(rows,int)
    if len(rows)<=int(size):
        return rows.copy()
    strata=np.sum(Y_BIN[rows],axis=1).astype(int)
    splitter=StratifiedShuffleSplit(n_splits=1,train_size=int(size),random_state=int(seed))
    take,_=next(splitter.split(np.zeros(len(rows)),strata))
    return np.sort(rows[take])


def make_shadow_folds(rows,n_splits,seed):
    rows=np.asarray(rows,int)
    strata=np.sum(Y_BIN[rows],axis=1).astype(int)
    sk=StratifiedKFold(n_splits=int(n_splits),shuffle=True,random_state=int(seed))
    return [(rows[tr],rows[va]) for tr,va in sk.split(rows,strata)]


def fit_proposal_bundle(train_rows,pred_rows,seed):
    """Train frozen V30/V31 plus XE on train_rows and predict pred_rows."""
    train_rows=np.asarray(train_rows,int); pred_rows=np.asarray(pred_rows,int)
    anc=build_anchor_split_full(train_rows,pred_rows,seed+101)

    intent_tr=crossfit_intentstate_features(
        y[train_rows],ORDINAL_GRADES[train_rows],TRAIN_INTENT[train_rows],MODULE_READY_BAND[train_rows],
        TRAIN_GROUP[train_rows],seed+211
    )
    intent_bundle=fit_intentstate_bundle(
        y[train_rows],ORDINAL_GRADES[train_rows],TRAIN_INTENT[train_rows],MODULE_READY_BAND[train_rows]
    )
    intent_pr=apply_intentstate_bundle(intent_bundle,TRAIN_INTENT[pred_rows] if pred_rows.max()<N_TRAIN else LATEST_INTENT_KEY[pred_rows],MODULE_READY_BAND[pred_rows])

    Xtr,_,bounds=build_long_matrix_v31(
        train_rows,anc["cons_train"],anc["base_train"],anc["meta_train"],intent_tr
    )
    Xpr,_,bounds2=build_long_matrix_v31(
        pred_rows,anc["cons_val"],anc["base_val"],anc["meta_val"],intent_pr
    )
    assert bounds==bounds2

    rank_seed=seed+401
    m0,r0=fit_ranker_predict("lambda",Xtr[:,:bounds["base"]],Xpr[:,:bounds["base"]],ORDINAL_GRADES[train_rows],len(train_rows),rank_seed)
    m1,r1=fit_ranker_predict("lambda",Xtr[:,:bounds["intentstate"]],Xpr[:,:bounds["intentstate"]],ORDINAL_GRADES[train_rows],len(train_rows),rank_seed)
    mx,rx=fit_ranker_predict("xe",Xtr[:,:bounds["intentstate"]],Xpr[:,:bounds["intentstate"]],ORDINAL_GRADES[train_rows],len(train_rows),rank_seed)
    my,ry=fit_yeti_predict(Xtr[:,:bounds["intentstate"]],Xpr[:,:bounds["intentstate"]],ORDINAL_GRADES[train_rows],len(train_rows),rank_seed)

    v30=blend_residual(anc["cons_val"],r0,ALPHA)
    v31=blend_residual(anc["cons_val"],r1,ALPHA)
    xe=blend_residual(anc["cons_val"],rx,ALPHA)
    yeti=blend_residual(anc["cons_val"],ry,ALPHA)
    objective_median=np.median(np.stack([v30,xe,yeti],axis=0),axis=0).astype(np.float32)
    hybrid,st=hybrid34_surface(anc["cons_val"],v30,v31)
    out={
        "pre":anc["cons_val"].astype(np.float32),
        "v30":v30.astype(np.float32),
        "v31":v31.astype(np.float32),
        "xe":xe.astype(np.float32),
        "yeti":yeti.astype(np.float32),
        "objective_median":objective_median,
        "hybrid":hybrid.astype(np.float32),
        "state":st.astype(np.int8),
    }
    del Xtr,Xpr,intent_tr,intent_pr,intent_bundle,m0,m1,mx,my,anc
    gc.collect()
    return out


def fit_reference_hybrid(train_rows,val_rows,seed):
    """Reference Hybrid34 analogue: full outer-train V30/V31, never used as XE proposal expert."""
    train_rows=np.asarray(train_rows,int); val_rows=np.asarray(val_rows,int)
    anc=build_anchor_split_full(train_rows,val_rows,seed+701)
    intent_tr=crossfit_intentstate_features(
        y[train_rows],ORDINAL_GRADES[train_rows],TRAIN_INTENT[train_rows],MODULE_READY_BAND[train_rows],
        TRAIN_GROUP[train_rows],seed+809
    )
    bundle=fit_intentstate_bundle(y[train_rows],ORDINAL_GRADES[train_rows],TRAIN_INTENT[train_rows],MODULE_READY_BAND[train_rows])
    intent_va=apply_intentstate_bundle(bundle,TRAIN_INTENT[val_rows],MODULE_READY_BAND[val_rows])
    Xtr,_,bounds=build_long_matrix_v31(train_rows,anc["cons_train"],anc["base_train"],anc["meta_train"],intent_tr)
    Xva,_,_=build_long_matrix_v31(val_rows,anc["cons_val"],anc["base_val"],anc["meta_val"],intent_va)
    rank_seed=seed+907
    m0,r0=fit_ranker_predict("lambda",Xtr[:,:bounds["base"]],Xva[:,:bounds["base"]],ORDINAL_GRADES[train_rows],len(train_rows),rank_seed)
    m1,r1=fit_ranker_predict("lambda",Xtr[:,:bounds["intentstate"]],Xva[:,:bounds["intentstate"]],ORDINAL_GRADES[train_rows],len(train_rows),rank_seed)
    v30=blend_residual(anc["cons_val"],r0,ALPHA)
    v31=blend_residual(anc["cons_val"],r1,ALPHA)
    hybrid,st=hybrid34_surface(anc["cons_val"],v30,v31)
    out={"pre":anc["cons_val"],"v30":v30,"v31":v31,"hybrid":hybrid,"state":st}
    del Xtr,Xva,intent_tr,intent_va,bundle,m0,m1,anc
    gc.collect()
    return out

print("Fold-matched proposal builders ready")

# ---- cell 22 (markdown) ----
# ## Proposal-context competence features and signed-utility router
#
# The router does **not** learn general relevance. One row means: “XE proposes this alternative Top-5 for this user.” Features describe model state, rank edits, margins, score geometry, module/user readiness, and decision-profile differences. Targets are exact `ΔNDCG@5` of accepting the XE proposal versus the corresponding Hybrid34 shadow.
#
# The meta learner estimates `P(benefit)`, `P(harm)`, and signed expected `ΔNDCG`. Policy thresholds are selected exclusively from cross-fitted meta-training predictions.
#
# A frozen YetiRank-NDCG sibling and the V32-style objective median are computed **only as competence metadata** (agreement/output-profile features). They can never become the submitted ranking expert in V36.

# ---- cell 23 (code) ----
def _safe_corr_row(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    sa=float(np.std(a)); sb=float(np.std(b))
    if sa<1e-8 or sb<1e-8:
        return 0.0
    return float(np.corrcoef(a,b)[0,1])


def proposal_feature_matrix(rows_global,bundle):
    rows_global=np.asarray(rows_global,int)
    pre=np.asarray(bundle["pre"],np.float32); v30=np.asarray(bundle["v30"],np.float32)
    v31=np.asarray(bundle["v31"],np.float32); xe=np.asarray(bundle["xe"],np.float32)
    yeti=np.asarray(bundle["yeti"],np.float32); med=np.asarray(bundle["objective_median"],np.float32)
    hyb=np.asarray(bundle["hybrid"],np.float32); st=np.asarray(bundle["state"],np.int8)
    n=len(rows_global)
    assert pre.shape==v30.shape==v31.shape==xe.shape==yeti.shape==med.shape==hyb.shape==(n,N_MODULES)

    rh=rank_matrix(hyb); rx=rank_matrix(xe); ry=rank_matrix(yeti); rm=rank_matrix(med)
    rv30=rank_matrix(v30); rp=rank_matrix(pre); rv31=rank_matrix(v31)
    th=top5_indices(hyb); tx=top5_indices(xe); ty=top5_indices(yeti); tm=top5_indices(med)

    names=[]; cols=[]
    def add(name,v):
        names.append(name); cols.append(np.asarray(v,np.float32).reshape(n))

    for s in range(4): add(f"state_{s}",(st==s).astype(np.float32))
    top1chg=(th[:,0]!=tx[:,0]).astype(np.float32)
    memb=np.asarray([set(a)!=set(b) for a,b in zip(th,tx)],np.float32)
    changed_pos=np.sum(th!=tx,axis=1).astype(np.float32)
    common=np.asarray([len(set(a).intersection(set(b))) for a,b in zip(th,tx)],np.float32)
    add("top1_changed",top1chg); add("membership_changed",memb)
    add("changed_top5_positions",changed_pos/5.0); add("top5_common",common/5.0)
    add("rank_l1",np.abs(rx-rh).sum(1)/(N_MODULES*N_MODULES))
    add("rank_max",np.abs(rx-rh).max(1)/N_MODULES)
    add("v30_pre_rank_l1",np.abs(rv30-rp).sum(1)/(N_MODULES*N_MODULES))
    add("v31_v30_rank_l1",np.abs(rv31-rv30).sum(1)/(N_MODULES*N_MODULES))
    # Objective-diversity context is metadata only: XE remains the sole proposal expert.
    add("xe_yeti_rank_l1",np.abs(rx-ry).sum(1)/(N_MODULES*N_MODULES))
    add("xe_median_rank_l1",np.abs(rx-rm).sum(1)/(N_MODULES*N_MODULES))
    add("xe_yeti_exact_top5",np.asarray([tuple(a)==tuple(b) for a,b in zip(tx,ty)],np.float32))
    add("xe_median_exact_top5",np.asarray([tuple(a)==tuple(b) for a,b in zip(tx,tm)],np.float32))
    add("xe_yeti_top1_agree",(tx[:,0]==ty[:,0]).astype(np.float32))
    add("xe_median_top1_agree",(tx[:,0]==tm[:,0]).astype(np.float32))
    add("xe_yeti_membership_overlap",np.asarray([len(set(a).intersection(set(b)))/5.0 for a,b in zip(tx,ty)],np.float32))
    add("xe_median_membership_overlap",np.asarray([len(set(a).intersection(set(b)))/5.0 for a,b in zip(tx,tm)],np.float32))

    # Score geometry, normalized per user so absolute score scaling cannot dominate.
    def zmat(s):
        mu=s.mean(1,keepdims=True); sd=np.maximum(s.std(1,keepdims=True),1e-5)
        return (s-mu)/sd
    zh=zmat(hyb); zx=zmat(xe); zv30=zmat(v30); zv31=zmat(v31); zp=zmat(pre)
    dz=zx-zh
    add("score_delta_mean_abs",np.mean(np.abs(dz),axis=1))
    add("score_delta_max_abs",np.max(np.abs(dz),axis=1))
    add("v30_pre_score_shift",np.mean(np.abs(zv30-zp),axis=1))
    add("v31_v30_score_shift",np.mean(np.abs(zv31-zv30),axis=1))
    add("hybrid_score_sd",hyb.std(1)); add("xe_score_sd",xe.std(1))
    add("score_corr",np.asarray([_safe_corr_row(zh[i],zx[i]) for i in range(n)],np.float32))
    zy=zmat(yeti); zm=zmat(med)
    add("xe_yeti_score_corr",np.asarray([_safe_corr_row(zx[i],zy[i]) for i in range(n)],np.float32))
    add("xe_median_score_corr",np.asarray([_safe_corr_row(zx[i],zm[i]) for i in range(n)],np.float32))

    sh=np.sort(hyb,axis=1)[:,::-1]; sx=np.sort(xe,axis=1)[:,::-1]
    for k in range(5):
        add(f"hyb_gap_{k+1}_{k+2}",sh[:,k]-sh[:,k+1])
        add(f"xe_gap_{k+1}_{k+2}",sx[:,k]-sx[:,k+1])

    # Decision-output profile itself: module rank movement + normalized score movement.
    for m,mid in enumerate(MODULES):
        add(f"rank_delta_{mid}",(rh[:,m]-rx[:,m])/N_MODULES)
    for m,mid in enumerate(MODULES):
        add(f"score_delta_z_{mid}",dz[:,m])
    for m,mid in enumerate(MODULES):
        add(f"hyb_top5_{mid}",(rh[:,m]<=5).astype(np.float32))
    for m,mid in enumerate(MODULES):
        add(f"xe_top5_{mid}",(rx[:,m]<=5).astype(np.float32))

    # Compact user state: assessments, invalid flags, chat-count/span meta only.
    user_compact=USER_FEATURES[rows_global,:25].astype(np.float32)
    for j in range(user_compact.shape[1]):
        add(f"user_compact_{j:02d}",user_compact[:,j])

    # Module-native evidence for the current and proposed Top-1 module.
    cur1=th[:,0].astype(int); tar1=tx[:,0].astype(int)
    for base_name,M in [
        ("lex_word",LEX_WORD),("lex_char",LEX_CHAR),("topic_count",MODULE_COUNT),
        ("topic_recency",MODULE_RECENCY),("topic_latest",MODULE_LATEST),
        ("readiness",MODULE_READINESS),("invalid_readiness",MODULE_INVALID),
    ]:
        a=M[rows_global,cur1]; b=M[rows_global,tar1]
        add(base_name+"_current_top1",a); add(base_name+"_target_top1",b); add(base_name+"_target_minus_current",b-a)

    Xp=np.column_stack(cols).astype(np.float32)
    if not np.isfinite(Xp).all():
        raise ValueError("non-finite proposal features")
    proposal_mask=np.any(th!=tx,axis=1)
    info={
        "state":st,
        "hybrid_top5":th,
        "xe_top5":tx,
        "proposal_mask":proposal_mask,
        "top1_changed":top1chg.astype(bool),
        "membership_changed":memb.astype(bool),
    }
    return Xp,names,info


def make_meta_classifier(seed):
    return lgb.LGBMClassifier(
        n_estimators=80 if V36_SMOKE else 240, learning_rate=.03,
        num_leaves=7, max_depth=3, min_child_samples=18,
        reg_alpha=1.0, reg_lambda=10.0, colsample_bytree=.78,
        subsample=.90, subsample_freq=1,
        random_state=int(seed), n_jobs=4, verbosity=-1, force_col_wise=True,
    )


def make_meta_regressor(seed):
    return lgb.LGBMRegressor(
        objective="regression_l1", n_estimators=80 if V36_SMOKE else 260, learning_rate=.025,
        num_leaves=7, max_depth=3, min_child_samples=18,
        reg_alpha=1.0, reg_lambda=12.0, colsample_bytree=.78,
        subsample=.90, subsample_freq=1,
        random_state=int(seed), n_jobs=4, verbosity=-1, force_col_wise=True,
    )


def _fit_binary_or_constant(X,target,weights,seed):
    target=np.asarray(target,np.int8)
    if len(np.unique(target))<2:
        return {"model":None,"constant":float(target.mean())}
    m=make_meta_classifier(seed); m.fit(X,target,sample_weight=weights)
    return {"model":m,"constant":None}


def _predict_binary(obj,X):
    if obj["model"] is None:
        return np.full(len(X),float(obj["constant"]),np.float32)
    return obj["model"].predict_proba(X)[:,1].astype(np.float32)


def fit_meta_router(Xp,delta,seed):
    Xp=np.asarray(Xp,np.float32); delta=np.asarray(delta,np.float32)
    w=(0.5+np.minimum(np.abs(delta)/0.05,3.0)).astype(np.float32)
    benefit=(delta>1e-12).astype(np.int8); harm=(delta<-1e-12).astype(np.int8)
    mb=_fit_binary_or_constant(Xp,benefit,w,seed+11)
    mh=_fit_binary_or_constant(Xp,harm,w,seed+23)
    mr=make_meta_regressor(seed+37); mr.fit(Xp,delta,sample_weight=w)
    return {"benefit":mb,"harm":mh,"delta":mr}


def predict_meta_router(router,Xp):
    pb=_predict_binary(router["benefit"],Xp)
    ph=_predict_binary(router["harm"],Xp)
    pdlt=router["delta"].predict(Xp).astype(np.float32)
    peu=(PROB_GAIN_REF*pb-PROB_HARM_REF*ph).astype(np.float32)
    score=np.minimum(pdlt,peu).astype(np.float32)
    return {"p_benefit":pb,"p_harm":ph,"pred_delta":pdlt,"prob_eu":peu,"score":score}


def crossfit_meta_predictions(Xp,delta,seed):
    Xp=np.asarray(Xp,np.float32); delta=np.asarray(delta,np.float32)
    n=len(delta)
    if n<8:
        router=fit_meta_router(Xp,delta,seed)
        pr=predict_meta_router(router,Xp)
        return pr
    sign=np.where(delta>1e-12,2,np.where(delta<-1e-12,0,1)).astype(int)
    counts=np.bincount(sign,minlength=3)
    k=min(META_CV_FOLDS,max(2,int(counts[counts>0].min()))) if np.any(counts>0) else 2
    if len(np.unique(sign))>=2 and counts[counts>0].min()>=2:
        splitter=StratifiedKFold(n_splits=k,shuffle=True,random_state=int(seed))
        splits=splitter.split(Xp,sign)
    else:
        splitter=KFold(n_splits=k,shuffle=True,random_state=int(seed))
        splits=splitter.split(Xp)
    out={k:np.zeros(n,np.float32) for k in ("p_benefit","p_harm","pred_delta","prob_eu","score")}
    for f,(tr,va) in enumerate(splits):
        r=fit_meta_router(Xp[tr],delta[tr],seed+101*f)
        p=predict_meta_router(r,Xp[va])
        for key in out: out[key][va]=p[key]
    return out


def scope_mask(states,scope):
    states=np.asarray(states,np.int8)
    if scope=="CORRIDOR":
        return states==2
    if scope=="ATTACK":
        return np.isin(states,[0,2])
    raise ValueError(scope)


def select_policy_from_meta(delta,states,preds,n_users,scope):
    delta=np.asarray(delta,np.float32); states=np.asarray(states,np.int8)
    score=np.asarray(preds["score"],np.float32)
    pb=np.asarray(preds["p_benefit"],np.float32); ph=np.asarray(preds["p_harm"],np.float32)
    pdlt=np.asarray(preds["pred_delta"],np.float32); peu=np.asarray(preds["prob_eu"],np.float32)
    base=(scope_mask(states,scope) & (score>0) & (pdlt>0) & (peu>0) & (pb>ph))
    idx=np.flatnonzero(base)
    grid=CORRIDOR_COVERAGE_GRID if scope=="CORRIDOR" else ATTACK_COVERAGE_GRID
    min_prec=CORRIDOR_MIN_PRECISION if scope=="CORRIDOR" else ATTACK_MIN_PRECISION
    max_harm=CORRIDOR_MAX_HARM if scope=="CORRIDOR" else ATTACK_MAX_HARM
    rows=[]
    for cov in grid:
        k=max(1,int(math.ceil(float(cov)*int(n_users))))
        if len(idx)==0: continue
        order=idx[np.argsort(-score[idx],kind="mergesort")]
        sel=order[:min(k,len(order))]
        if len(sel)<2: continue
        inc=float(delta[sel].sum()/max(1,int(n_users)))
        prec=float(np.mean(delta[sel]>0)); harm=float(np.mean(delta[sel]<0))
        rows.append({
            "scope":scope,"coverage_cap":float(cov),"k":int(len(sel)),"threshold":float(score[sel[-1]]),
            "train_increment":inc,"precision":prec,"harm_rate":harm,
            "mean_action_delta":float(delta[sel].mean()),
            "eligible":bool(prec>=min_prec and harm<=max_harm and inc>0),
        })
    tab=pd.DataFrame(rows)
    if len(tab)==0 or not tab["eligible"].any():
        return {"scope":scope,"threshold":float("inf"),"coverage_cap":0.0,"selected_train":0,"train_increment":0.0,"precision":0.0,"harm_rate":1.0},tab
    good=tab[tab.eligible].sort_values(["train_increment","precision","k"],ascending=[False,False,True],kind="mergesort")
    r=good.iloc[0]
    policy={
        "scope":scope,"threshold":float(r.threshold),"coverage_cap":float(r.coverage_cap),
        "selected_train":int(r.k),"train_increment":float(r.train_increment),
        "precision":float(r.precision),"harm_rate":float(r.harm_rate),
    }
    return policy,tab

print("V36 proposal competence meta-router ready")

# ---- cell 24 (markdown) ----
# ## Strict world runner: honest meta-training, stable shadow voting, and oracle gate
#
# For each world:
# 1. fit a full-reference Hybrid34 analogue on the outer train pool;
# 2. create four 2700-user shadow experts;
# 3. use each shadow's held-out train fold to build honest meta-training proposals;
# 4. choose CORRIDOR/ATTACK thresholds only from cross-fitted meta predictions;
# 5. evaluate the fixed policy on the untouched outer validation users by requiring at least 3/4 shadow votes for the same XE Top-5 proposal.
#
# The oracle audit uses the same stable proposal pool. It answers whether deployable XE proposals contain enough remaining utility before judging the router.

# ---- cell 25 (code) ----
def make_fresh_holdout(seed,val_size=DEV_VAL_SIZE):
    rows=np.arange(N_TRAIN,dtype=int)
    strata=np.sum(Y_BIN,axis=1).astype(int)
    sp=StratifiedShuffleSplit(n_splits=1,test_size=int(val_size),random_state=int(seed))
    tr,va=next(sp.split(rows,strata))
    return rows[tr],rows[va]


def build_meta_oof_and_outer_shadows(train_pool,outer_val,seed):
    train_pool=np.asarray(train_pool,int); outer_val=np.asarray(outer_val,int)
    folds=make_shadow_folds(train_pool,DEV_SHADOW_FOLDS,seed+101)
    pos={u:i for i,u in enumerate(train_pool.tolist())}
    meta={k:np.zeros((len(train_pool),N_MODULES),np.float32) for k in ("pre","v30","v31","xe","yeti","objective_median","hybrid")}
    meta_state=np.zeros(len(train_pool),np.int8)
    outer_shadows=[]
    for f,(tr0,va0) in enumerate(folds):
        fit_rows=deterministic_train_subset(tr0,EXPERT_TRAIN_SIZE,seed+1009+f*131)
        pred_rows=np.concatenate([va0,outer_val]).astype(int)
        b=fit_proposal_bundle(fit_rows,pred_rows,seed+2003+f*173)
        nva=len(va0)
        for key in meta:
            meta[key][[pos[int(u)] for u in va0]]=b[key][:nva]
        meta_state[[pos[int(u)] for u in va0]]=b["state"][:nva]
        outer_shadows.append({key:(b[key][nva:] if key!="state" else b[key][nva:]) for key in b})
        del b
        gc.collect()
    meta["state"]=meta_state
    for key in ("pre","v30","v31","xe","hybrid"):
        if not np.isfinite(meta[key]).all(): raise ValueError((key,"nonfinite meta OOF"))
    return meta,outer_shadows


def proposal_rows_dataset(rows_global,bundle,with_labels=True):
    Xall,names,info=proposal_feature_matrix(rows_global,bundle)
    mask=np.asarray(info["proposal_mask"],bool)
    idx=np.flatnonzero(mask)
    out={
        "X":Xall[idx],"feature_names":names,"rows_global":np.asarray(rows_global,int)[idx],
        "state":np.asarray(info["state"],np.int8)[idx],
        "target_top5":np.asarray(info["xe_top5"],np.int16)[idx],
        "hybrid_top5":np.asarray(info["hybrid_top5"],np.int16)[idx],
        "top1_changed":np.asarray(info["top1_changed"],bool)[idx],
        "membership_changed":np.asarray(info["membership_changed"],bool)[idx],
    }
    if with_labels:
        if np.any(np.asarray(rows_global)[idx]>=N_TRAIN):
            raise ValueError("labels requested for test rows")
        u0=per_user_ndcg(y[np.asarray(rows_global,int)[idx]],np.asarray(bundle["hybrid"])[idx])
        ux=per_user_ndcg(y[np.asarray(rows_global,int)[idx]],np.asarray(bundle["xe"])[idx])
        out["delta"]=(ux-u0).astype(np.float32)
    return out


def apply_shadow_policy(reference,rows_global,shadow_bundles,router,policy,scope,min_votes=MIN_SHADOW_VOTES,with_labels=True):
    rows_global=np.asarray(rows_global,int)
    ref_h=np.asarray(reference["hybrid"],np.float32); ref_state=np.asarray(reference["state"],np.int8)
    n=len(rows_global)
    votes=[defaultdict(list) for _ in range(n)]
    raw_proposal_votes=[Counter() for _ in range(n)]

    for sidx,b in enumerate(shadow_bundles):
        ds=proposal_rows_dataset(rows_global,b,with_labels=False)
        if len(ds["X"])==0: continue
        pr=predict_meta_router(router,ds["X"])
        for j,rg in enumerate(ds["rows_global"]):
            local=int(np.flatnonzero(rows_global==rg)[0])
            target=tuple(int(x) for x in ds["target_top5"][j])
            raw_proposal_votes[local][target]+=1
            shadow_state=int(ds["state"][j])
            eligible_scope=bool(scope_mask(np.asarray([shadow_state]),scope)[0])
            accepted=bool(
                eligible_scope and pr["score"][j]>=float(policy["threshold"]) and
                pr["pred_delta"][j]>0 and pr["prob_eu"][j]>0 and pr["p_benefit"][j]>pr["p_harm"][j]
            )
            if accepted:
                votes[local][target].append({
                    "shadow":sidx,"score":float(pr["score"][j]),"p_benefit":float(pr["p_benefit"][j]),
                    "p_harm":float(pr["p_harm"][j]),"pred_delta":float(pr["pred_delta"][j]),"state":shadow_state,
                })

    out=ref_h.copy(); ledger=[]
    for i in range(n):
        # Reference state is authoritative: S3 is immutable and S1 is excluded by design.
        if not bool(scope_mask(np.asarray([ref_state[i]]),scope)[0]):
            continue
        if not votes[i]: continue
        ranked=[]
        for target,recs in votes[i].items():
            ranked.append((len(recs),float(np.median([r["score"] for r in recs])),target,recs))
        ranked.sort(key=lambda z:(-z[0],-z[1],z[2]))
        nv,medscore,target,recs=ranked[0]
        if nv<int(min_votes):
            continue
        out[i]=apply_target_top5(ref_h[i],target)
        ledger.append({
            "row_local":i,"row_global":int(rows_global[i]),
            "user_id":str(train.iloc[rows_global[i]].user_id) if rows_global[i]<N_TRAIN else str(test.iloc[rows_global[i]-N_TRAIN].user_id),
            "scope":scope,"reference_state":str(STATE_NAMES[ref_state[i]]),
            "shadow_votes":int(nv),"median_router_score":float(medscore),
            "reference_top5":" ".join(MODULES[j] for j in top5_indices(ref_h[i:i+1])[0]),
            "target_top5":" ".join(MODULES[j] for j in target),
            "top1_changed":bool(top5_indices(ref_h[i:i+1])[0,0]!=target[0]),
            "membership_changed":bool(set(top5_indices(ref_h[i:i+1])[0])!=set(target)),
            "median_p_benefit":float(np.median([r["p_benefit"] for r in recs])),
            "median_p_harm":float(np.median([r["p_harm"] for r in recs])),
            "median_pred_delta":float(np.median([r["pred_delta"] for r in recs])),
            "raw_same_target_votes":int(raw_proposal_votes[i][target]),
        })
    led=pd.DataFrame(ledger)
    return out,led


def stable_oracle_candidate(reference,rows_global,shadow_bundles,scope,min_votes=MIN_SHADOW_VOTES):
    rows_global=np.asarray(rows_global,int); ref_h=np.asarray(reference["hybrid"],np.float32)
    ref_state=np.asarray(reference["state"],np.int8); n=len(rows_global)
    counters=[Counter() for _ in range(n)]
    for b in shadow_bundles:
        tx=top5_indices(b["xe"]); th=top5_indices(b["hybrid"])
        for i in range(n):
            if np.any(tx[i]!=th[i]):
                counters[i][tuple(int(x) for x in tx[i])]+=1
    proposed=ref_h.copy(); has=np.zeros(n,bool)
    for i,c in enumerate(counters):
        if not bool(scope_mask(np.asarray([ref_state[i]]),scope)[0]) or not c: continue
        target,nv=sorted(c.items(),key=lambda kv:(-kv[1],kv[0]))[0]
        if nv>=int(min_votes):
            proposed[i]=apply_target_top5(ref_h[i],target); has[i]=True
    if np.any(rows_global>=N_TRAIN):
        return proposed,has,np.full(n,np.nan,np.float32)
    base=per_user_ndcg(y[rows_global],ref_h); cand=per_user_ndcg(y[rows_global],proposed)
    d=(cand-base).astype(np.float32)
    return proposed,has,d


def run_v36_world(train_pool,val_rows,seed,world):
    train_pool=np.asarray(train_pool,int); val_rows=np.asarray(val_rows,int)
    print(f"\n=== V36 {world}: train_pool={len(train_pool)} val={len(val_rows)} ===")
    reference=fit_reference_hybrid(train_pool,val_rows,seed+17)
    meta_bundle,outer_shadows=build_meta_oof_and_outer_shadows(train_pool,val_rows,seed+31)

    meta_ds=proposal_rows_dataset(train_pool,meta_bundle,with_labels=True)
    if len(meta_ds["X"])<8:
        raise RuntimeError(f"Too few XE proposal rows in {world}: {len(meta_ds['X'])}")
    meta_oof_pred=crossfit_meta_predictions(meta_ds["X"],meta_ds["delta"],seed+53)
    router=fit_meta_router(meta_ds["X"],meta_ds["delta"],seed+79)

    policies={}; policy_tabs=[]
    for scope in ("CORRIDOR","ATTACK"):
        pol,tab=select_policy_from_meta(meta_ds["delta"],meta_ds["state"],meta_oof_pred,len(train_pool),scope)
        policies[scope]=pol
        if len(tab):
            tab=tab.copy(); tab["world"]=world; policy_tabs.append(tab)

    uref=per_user_ndcg(y[val_rows],reference["hybrid"])
    world_rows=[]; candidate_scores={}; ledgers={}; oracle_rows=[]
    for scope in ("CORRIDOR","ATTACK"):
        cand,ledger=apply_shadow_policy(reference,val_rows,outer_shadows,router,policies[scope],scope,MIN_SHADOW_VOTES,True)
        uc=per_user_ndcg(y[val_rows],cand); inc=uc-uref
        candidate_scores[scope]=cand; ledgers[scope]=ledger
        ci=cluster_boot_ci(inc,TRAIN_GROUP[val_rows],seed+101+(0 if scope=="CORRIDOR" else 1))

        oracle_cand,oracle_has,oracle_delta=stable_oracle_candidate(reference,val_rows,outer_shadows,scope,MIN_SHADOW_VOTES)
        pos=np.where(oracle_delta>0,oracle_delta,0.0)
        k=max(1,int(math.ceil(0.01*len(val_rows))))
        top1pct=float(np.sort(pos)[::-1][:k].sum()/len(val_rows))
        oracle_gain=float(pos.sum()/len(val_rows))
        oracle_rows.append({"world":world,"scope":scope,"stable_proposals":int(oracle_has.sum()),"oracle_gain":oracle_gain,"oracle_top1pct_gain":top1pct})

        if len(ledger):
            local_idx=ledger["row_local"].to_numpy(int)
            realized=inc[local_idx]
            benefit=int(np.sum(realized>0)); harm=int(np.sum(realized<0)); tie=int(np.sum(np.isclose(realized,0)))
        else:
            benefit=harm=tie=0
        world_rows.append({
            "world":world,"scope":scope,"n_val":len(val_rows),"meta_proposals":len(meta_ds["X"]),
            "reference_hybrid_ndcg":float(uref.mean()),"candidate_ndcg":float(uc.mean()),
            "increment_vs_hybrid34":float(inc.mean()),"ci_low":ci[0],"ci_high":ci[1],
            "actions":int(len(ledger)),"beneficial_actions":benefit,"harmful_actions":harm,"tie_actions":tie,
            "precision":float(benefit/max(1,benefit+harm)),
            "top1_changes":int(ledger["top1_changed"].sum()) if len(ledger) else 0,
            "membership_changes":int(ledger["membership_changed"].sum()) if len(ledger) else 0,
            "policy_threshold":float(policies[scope]["threshold"]),"policy_train_increment":float(policies[scope]["train_increment"]),
            "policy_train_precision":float(policies[scope]["precision"]),"policy_train_harm":float(policies[scope]["harm_rate"]),
        })

    feature_imp=pd.DataFrame({
        "feature":meta_ds["feature_names"],
        "benefit_importance":router["benefit"]["model"].feature_importances_ if router["benefit"]["model"] is not None else np.zeros(len(meta_ds["feature_names"])),
        "harm_importance":router["harm"]["model"].feature_importances_ if router["harm"]["model"] is not None else np.zeros(len(meta_ds["feature_names"])),
        "delta_importance":router["delta"].feature_importances_,
    })
    del reference,meta_bundle,outer_shadows,router,meta_ds,meta_oof_pred
    gc.collect()
    return pd.DataFrame(world_rows),pd.DataFrame(oracle_rows),pd.concat(policy_tabs,ignore_index=True) if policy_tabs else pd.DataFrame(),ledgers,feature_imp

print("V36 strict world runner ready")

# ---- cell 26 (markdown) ----
# ## Fresh A/B/C development worlds
#
# These partitions are new for V36. They are used only to **measure** the fixed architecture. Each world's routing threshold is learned from that world's training-side cross-fitted meta predictions; the 400 outer validation labels never select thresholds.

# ---- cell 27 (code) ----
DEV_WORLD_ROWS=[]; DEV_ORACLE_ROWS=[]; DEV_POLICY_ROWS=[]; DEV_LEDGERS=[]; DEV_IMPORTANCES=[]
for w,seed in FRESH_WORLDS.items():
    tr,va=make_fresh_holdout(seed,DEV_VAL_SIZE)
    wr,orow,ptab,leds,fimp=run_v36_world(tr,va,seed,w)
    DEV_WORLD_ROWS.append(wr); DEV_ORACLE_ROWS.append(orow)
    if len(ptab): DEV_POLICY_ROWS.append(ptab)
    for scope,ld in leds.items():
        if len(ld):
            x=ld.copy(); x["world"]=w; DEV_LEDGERS.append(x)
    fimp=fimp.copy(); fimp["world"]=w; DEV_IMPORTANCES.append(fimp)

V36_DEV=pd.concat(DEV_WORLD_ROWS,ignore_index=True)
V36_ORACLE=pd.concat(DEV_ORACLE_ROWS,ignore_index=True)
V36_POLICY_SEARCH=pd.concat(DEV_POLICY_ROWS,ignore_index=True) if DEV_POLICY_ROWS else pd.DataFrame()
V36_DEV_LEDGER=pd.concat(DEV_LEDGERS,ignore_index=True) if DEV_LEDGERS else pd.DataFrame()
V36_FEATURE_IMPORTANCE=pd.concat(DEV_IMPORTANCES,ignore_index=True)

display(V36_DEV)
display(V36_ORACLE)

# ---- cell 28 (markdown) ----
# ## Development gate and max-min candidate choice
#
# ATTACK is eligible only if it shows leader-scale fresh improvement. CORRIDOR is a smaller incremental candidate. Both require all three worlds non-negative and sufficient stable XE oracle headroom. No public leaderboard information beyond the already-frozen Hybrid34 anchor participates in this gate.

# ---- cell 29 (code) ----
pooled_rows=[]
for scope in ("CORRIDOR","ATTACK"):
    d=V36_DEV[V36_DEV.scope==scope].copy()
    # Pooled observations are world-level user means; world overlap is handled by requiring all three worlds positive.
    pooled=float(d["increment_vs_hybrid34"].mean())
    worst=float(d["increment_vs_hybrid34"].min())
    all_pos=bool((d["increment_vs_hybrid34"]>0).all())
    oracle_worst=float(V36_ORACLE[V36_ORACLE.scope==scope]["oracle_gain"].min())
    # Conservative world-bootstrap CI over the three preregistered worlds.
    vals=d["increment_vs_hybrid34"].to_numpy(float)
    rng=np.random.default_rng(_seed_from_text("V36-DEV-WORLD-CI-"+scope))
    if V36_SMOKE:
        ci=(float(vals.min()),float(vals.max()))
    else:
        boots=np.asarray([rng.choice(vals,size=len(vals),replace=True).mean() for _ in range(5000)])
        ci=tuple(float(x) for x in np.quantile(boots,[.025,.975]))
    if scope=="ATTACK":
        passed=bool(all_pos and pooled>=ATTACK_MIN_POOLED and worst>=ATTACK_MIN_WORST and ci[0]>=ATTACK_CI_FLOOR and oracle_worst>=ORACLE_MIN_WORST)
    else:
        passed=bool(all_pos and pooled>=CORRIDOR_MIN_POOLED and worst>=CORRIDOR_MIN_WORST and ci[0]>=CORRIDOR_CI_FLOOR and oracle_worst>=ORACLE_MIN_WORST)
    pooled_rows.append({
        "scope":scope,"pooled_increment":pooled,"worst_world":worst,"all_worlds_positive":all_pos,
        "ci_low":ci[0],"ci_high":ci[1],"oracle_worst":oracle_worst,
        "total_actions":int(d.actions.sum()),"passed":passed,
    })
V36_GATE=pd.DataFrame(pooled_rows)
display(V36_GATE)

# Precommitted priority: leader-scale ATTACK first, then smaller CORRIDOR, else fail-close.
if bool(V36_GATE.loc[V36_GATE.scope=="ATTACK","passed"].iloc[0]):
    DEV_WINNER="ATTACK"
elif bool(V36_GATE.loc[V36_GATE.scope=="CORRIDOR","passed"].iloc[0]):
    DEV_WINNER="CORRIDOR"
else:
    DEV_WINNER=None
DEV_GATE=bool(DEV_WINNER is not None)
print("DEV_WINNER =",DEV_WINNER,"DEV_GATE =",DEV_GATE)

# ---- cell 30 (markdown) ----
# ## Sealed D + chronology: rejection-only
#
# Only if A/B/C promote a candidate do we open:
# - **sealed D:** template-group-disjoint hashed holdout;
# - **chronology:** U0001–U3600 → U3601–U4000.
#
# The same algorithm chooses its training-side threshold; validation outcomes can only reject the already-chosen scope.

# ---- cell 31 (code) ----
def sealed_d_bucket(group):
    h=hashlib.sha1(f"{SEALED_D_SALT}::{group}".encode("utf-8")).hexdigest()
    return int(h[:12],16)%10

SEALED_D_MASK=np.asarray([sealed_d_bucket(g)==0 for g in TRAIN_GROUP],bool)
SEALED_D_TR=np.flatnonzero(~SEALED_D_MASK)
SEALED_D_VA=np.flatnonzero(SEALED_D_MASK)
assert len(set(TRAIN_GROUP[SEALED_D_TR]).intersection(set(TRAIN_GROUP[SEALED_D_VA])))==0

CHRONO_TR=np.arange(0,3600,dtype=int)
CHRONO_VA=np.arange(3600,4000,dtype=int)

SEALED_ROWS=[]; SEALED_ORACLE=[]; SEALED_LEDGERS=[]
if DEV_GATE:
    for name,tr,va,seed in [
        ("sealed_template_D",SEALED_D_TR,SEALED_D_VA,_seed_from_text("V36-SEALED-D-FIT")),
        ("chronology_U3601_U4000",CHRONO_TR,CHRONO_VA,_seed_from_text("V36-CHRONO-FIT")),
    ]:
        wr,orow,ptab,leds,fimp=run_v36_world(tr,va,seed,name)
        row=wr[wr.scope==DEV_WINNER].copy(); SEALED_ROWS.append(row)
        SEALED_ORACLE.append(orow[orow.scope==DEV_WINNER].copy())
        ld=leds[DEV_WINNER]
        if len(ld):
            x=ld.copy(); x["world"]=name; SEALED_LEDGERS.append(x)
    V36_SEALED=pd.concat(SEALED_ROWS,ignore_index=True)
    V36_SEALED_ORACLE=pd.concat(SEALED_ORACLE,ignore_index=True)
    V36_SEALED_LEDGER=pd.concat(SEALED_LEDGERS,ignore_index=True) if SEALED_LEDGERS else pd.DataFrame()
    sealed_inc=float(V36_SEALED.loc[V36_SEALED.world=="sealed_template_D","increment_vs_hybrid34"].iloc[0])
    chrono_inc=float(V36_SEALED.loc[V36_SEALED.world=="chronology_U3601_U4000","increment_vs_hybrid34"].iloc[0])
    SEALED_GATE=bool(sealed_inc>=-0.00005 and chrono_inc>=0.0)
else:
    V36_SEALED=pd.DataFrame([{"world":"skipped_dev_gate_failed","scope":"NONE","increment_vs_hybrid34":np.nan,"actions":0}])
    V36_SEALED_ORACLE=pd.DataFrame()
    V36_SEALED_LEDGER=pd.DataFrame()
    SEALED_GATE=False

display(V36_SEALED)
print("SEALED_GATE =",SEALED_GATE)

# ---- cell 32 (markdown) ----
# ## Full-data reconstruction of public Hybrid34 and fold-matched test shadows
#
# This cell first recreates V30 and V31 from organizer data, then deterministically builds the public 0.66006 rule:
#
# `Hybrid34 = V31 only where (V30 != V25) AND (V31 != V30); otherwise V30.`
#
# No list of 34 user IDs is used for routing. Geometry is asserted by counts and a Top-5 signature hash. V36 test XE proposals come from four 2700-user shadows; the final score levels remain Hybrid34.

# ---- cell 33 (code) ----
OUT_DIR=Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("/mnt/data")
OUT_DIR.mkdir(parents=True,exist_ok=True)

# Full strict Intent-State representation for exact V30/V31 reconstruction.
full_intent_tr=crossfit_intentstate_features(
    y,ORDINAL_GRADES,TRAIN_INTENT,MODULE_READY_BAND[:N_TRAIN],TRAIN_GROUP,
    _seed_from_text("V31-FULL-INTENTSTATE-CROSSFIT")
)
FULL_INTENT_BUNDLE=fit_intentstate_bundle(y,ORDINAL_GRADES,TRAIN_INTENT,MODULE_READY_BAND[:N_TRAIN])
full_intent_te=apply_intentstate_bundle(FULL_INTENT_BUNDLE,TEST_INTENT,MODULE_READY_BAND[N_TRAIN:])
V36_SUPPORT_AUDIT=intentstate_support_audit(FULL_INTENT_BUNDLE,TEST_INTENT,MODULE_READY_BAND[N_TRAIN:])

Xtr_full,names_full,bounds_full=build_long_matrix_v31(
    np.arange(N_TRAIN),consensus3_oof,base_oof,meta_oof,full_intent_tr
)
Xte_full,_,bounds_test=build_long_matrix_v31(
    np.arange(N_TRAIN,N_TRAIN+N_TEST),consensus3_test,base_test,meta_test,full_intent_te
)
assert bounds_full==bounds_test
full_rank_seed=_seed_from_text("V30-FULL-RANKER")
model_v30,raw_v30=fit_ranker_predict("lambda",Xtr_full[:,:bounds_full["base"]],Xte_full[:,:bounds_full["base"]],ORDINAL_GRADES,N_TRAIN,full_rank_seed)
model_v31,raw_v31=fit_ranker_predict("lambda",Xtr_full[:,:bounds_full["intentstate"]],Xte_full[:,:bounds_full["intentstate"]],ORDINAL_GRADES,N_TRAIN,full_rank_seed)

v30_test=blend_residual(consensus3_test,raw_v30,ALPHA)
v31_test=blend_residual(consensus3_test,raw_v31,ALPHA)
protected_rows=[TEST_UID_TO_ROW[u] for u in sorted(PROTECTED_TEST_USERS)]
v30_test[protected_rows]=v25_anchor_test[protected_rows]
v31_test[protected_rows]=v25_anchor_test[protected_rows]

full_state=state_codes(v25_anchor_test,v30_test,v31_test)
hybrid34_test=v30_test.copy(); hybrid34_test[full_state==3]=v31_test[full_state==3]

# Reproduction audit for V30, V31 disagreement geometry, and public Hybrid34 state machine.
v25_top=top5_indices(v25_anchor_test); v30_top=top5_indices(v30_test); v31_top=top5_indices(v31_test)
v30_changed_rows=np.flatnonzero(np.any(v25_top!=v30_top,axis=1))
v31_changed_rows=np.flatnonzero(np.any(v30_top!=v31_top,axis=1))
v30_changed_users=set(test.iloc[v30_changed_rows]["user_id"].astype(str))
state_count={s:int(np.sum(full_state==i)) for i,s in enumerate(STATE_NAMES)}
hybrid_top=top5_indices(hybrid34_test).astype(np.int16)
hybrid_hash=hashlib.sha256(hybrid_top.tobytes()).hexdigest()
V30_REPRO_OK=bool(
    v30_changed_users==EXPECTED_V30_CHANGED_USERS and
    int(np.sum(v25_top[:,0]!=v30_top[:,0]))==EXPECTED_V30_TOP1_CHANGES and
    int(np.sum([set(a)!=set(b) for a,b in zip(v25_top,v30_top)]))==EXPECTED_V30_MEMBERSHIP_CHANGES
)
HYBRID_REPRO_OK=bool(
    len(v31_changed_rows)==EXPECTED_V31_CHANGED_VS_V30 and
    int(np.sum(full_state==3))==EXPECTED_HYBRID_OVERLAP and
    state_count==EXPECTED_STATE_COUNTS and
    hybrid_hash==EXPECTED_HYBRID_TOP5_SHA256
)
V36_REPRO_AUDIT={
    "v30_reproduction_ok":V30_REPRO_OK,"hybrid34_reproduction_ok":HYBRID_REPRO_OK,
    "v30_changed_users":len(v30_changed_rows),"v31_changed_vs_v30":len(v31_changed_rows),
    "state_counts":state_count,"hybrid_top5_sha256":hybrid_hash,
    "expected_hybrid_top5_sha256":EXPECTED_HYBRID_TOP5_SHA256,
}
print(json.dumps(V36_REPRO_AUDIT,indent=2))
if not V36_SMOKE:
    assert V30_REPRO_OK, "Exact V30 geometry did not reproduce."
    assert HYBRID_REPRO_OK, "Public Hybrid34 geometry did not reproduce; fail before any V36 submission."

# Full-train meta OOF + four 2700-user test shadows. Each train user gets an honest shadow proposal.
all_train=np.arange(N_TRAIN,dtype=int); test_global=np.arange(N_TRAIN,N_TRAIN+N_TEST,dtype=int)
final_folds=make_shadow_folds(all_train,FINAL_SHADOW_FOLDS,_seed_from_text("V36-FINAL-SHADOW-FOLDS"))
pos={u:i for i,u in enumerate(all_train.tolist())}
meta_full={k:np.zeros((N_TRAIN,N_MODULES),np.float32) for k in ("pre","v30","v31","xe","yeti","objective_median","hybrid")}
meta_full_state=np.zeros(N_TRAIN,np.int8); test_shadows=[]
for f,(tr0,va0) in enumerate(final_folds):
    fit_rows=deterministic_train_subset(tr0,EXPERT_TRAIN_SIZE,_seed_from_text(f"V36-FINAL-SHADOW-TRAIN-{f}"))
    pred_rows=np.concatenate([va0,test_global]).astype(int)
    b=fit_proposal_bundle(fit_rows,pred_rows,_seed_from_text(f"V36-FINAL-SHADOW-MODEL-{f}"))
    nva=len(va0)
    for key in meta_full:
        meta_full[key][va0]=b[key][:nva]
    meta_full_state[va0]=b["state"][:nva]
    tb={key:(b[key][nva:] if key!="state" else b[key][nva:]) for key in b}
    # Protected V25 rows are immutable in every test-side shadow, matching the public lineage.
    for key in ("v30","v31","xe","yeti","objective_median","hybrid"):
        tb[key][protected_rows]=v25_anchor_test[protected_rows]
    tb["pre"][protected_rows]=v25_anchor_test[protected_rows]
    tb["objective_median"]=np.median(np.stack([tb["v30"],tb["xe"],tb["yeti"]],axis=0),axis=0).astype(np.float32)
    tb["objective_median"][protected_rows]=v25_anchor_test[protected_rows]
    tb["state"]=state_codes(tb["pre"],tb["v30"],tb["v31"])
    tb["hybrid"],_=hybrid34_surface(tb["pre"],tb["v30"],tb["v31"])
    test_shadows.append(tb)
    del b
    gc.collect()
meta_full["state"]=meta_full_state

full_meta_ds=proposal_rows_dataset(all_train,meta_full,with_labels=True)
full_meta_oof=crossfit_meta_predictions(full_meta_ds["X"],full_meta_ds["delta"],_seed_from_text("V36-FINAL-META-CROSSFIT"))
full_router=fit_meta_router(full_meta_ds["X"],full_meta_ds["delta"],_seed_from_text("V36-FINAL-META-FIT"))

full_policies={}; full_policy_tables=[]
for scope in ("CORRIDOR","ATTACK"):
    pol,tab=select_policy_from_meta(full_meta_ds["delta"],full_meta_ds["state"],full_meta_oof,N_TRAIN,scope)
    full_policies[scope]=pol
    if len(tab):
        tab=tab.copy(); tab["world"]="FULL_TRAIN"; full_policy_tables.append(tab)
V36_FULL_POLICY_SEARCH=pd.concat(full_policy_tables,ignore_index=True) if full_policy_tables else pd.DataFrame()

# Reference object for final action application: exact full-data Hybrid34.
final_reference={"hybrid":hybrid34_test,"state":full_state}
TEST_CANDIDATES={}; TEST_LEDGERS={}
for scope in ("CORRIDOR","ATTACK"):
    cand,led=apply_shadow_policy(final_reference,test_global,test_shadows,full_router,full_policies[scope],scope,MIN_SHADOW_VOTES,False)
    # Hard safety: public-proven S3 and public-toxic S1 are never changed.
    ch=np.any(top5_indices(cand)!=top5_indices(hybrid34_test),axis=1)
    assert not np.any(ch & np.isin(full_state,[1,3]))
    TEST_CANDIDATES[scope]=cand; TEST_LEDGERS[scope]=led

FINAL_GATE=bool(DEV_GATE and SEALED_GATE and V30_REPRO_OK and HYBRID_REPRO_OK)
selected_test=TEST_CANDIDATES[DEV_WINNER].copy() if DEV_WINNER is not None else hybrid34_test.copy()
primary_test=selected_test.copy() if FINAL_GATE else hybrid34_test.copy()

assert sample["user_id"].astype(str).tolist()==test["user_id"].astype(str).tolist()
def submission_from_scores(scores):
    out=sample.copy(); out.loc[:,MODULES]=np.asarray(scores,np.float32)
    assert np.isfinite(out[MODULES].to_numpy()).all(); return out

SUBMISSIONS={
    "anchor_hybrid34":submission_from_scores(hybrid34_test),
    "corridor":submission_from_scores(TEST_CANDIDATES["CORRIDOR"]),
    "attack":submission_from_scores(TEST_CANDIDATES["ATTACK"]),
    "primary":submission_from_scores(primary_test),
}
SUBMISSIONS["anchor_hybrid34"].to_csv(OUT_DIR/"submission_v36_anchor_hybrid34_reconstructed.csv",index=False)
SUBMISSIONS["corridor"].to_csv(OUT_DIR/"submission_v36_DIAGNOSTIC_corridor.csv",index=False)
SUBMISSIONS["attack"].to_csv(OUT_DIR/"submission_v36_DIAGNOSTIC_attack.csv",index=False)
SUBMISSIONS["primary"].to_csv(OUT_DIR/"submission_v36_primary.csv",index=False)
if not FINAL_GATE:
    assert np.array_equal(SUBMISSIONS["primary"][MODULES].to_numpy(),SUBMISSIONS["anchor_hybrid34"][MODULES].to_numpy())

V36_TEST_GEOMETRY=pd.DataFrame([
    {
        "scope":scope,
        "changed_users_vs_hybrid34":int(np.sum(np.any(top5_indices(hybrid34_test)!=top5_indices(TEST_CANDIDATES[scope]),axis=1))),
        "top1_changes":int(np.sum(top5_indices(hybrid34_test)[:,0]!=top5_indices(TEST_CANDIDATES[scope])[:,0])),
        "membership_changes":int(np.sum([set(a)!=set(b) for a,b in zip(top5_indices(hybrid34_test),top5_indices(TEST_CANDIDATES[scope]))])),
        "full_policy_threshold":float(full_policies[scope]["threshold"]),
        "full_policy_train_increment":float(full_policies[scope]["train_increment"]),
        "full_policy_train_precision":float(full_policies[scope]["precision"]),
    } for scope in ("CORRIDOR","ATTACK")
])
display(V36_TEST_GEOMETRY)
print("FINAL_GATE =",FINAL_GATE,"DEV_WINNER =",DEV_WINNER)

# Release largest matrices before serializing artifacts.
del Xtr_full,Xte_full,full_intent_tr,full_intent_te,model_v30,model_v31,full_router

gc.collect()

# ---- cell 34 (markdown) ----
# ## Artifacts and submission contract
#
# `submission_v36_primary.csv` is submission-eligible **only** when A/B/C, sealed D, chronology, V30 reconstruction, and Hybrid34 reconstruction all pass. Otherwise it is byte-equivalent in values to the reconstructed 0.66006 Hybrid34 fail-close anchor.
#
# Diagnostic candidate files are exported for research only; do not bypass the gate.

# ---- cell 35 (code) ----
V36_DEV.to_csv(OUT_DIR/"v36_development_worlds.csv",index=False)
V36_ORACLE.to_csv(OUT_DIR/"v36_oracle_headroom.csv",index=False)
V36_GATE.to_csv(OUT_DIR/"v36_development_gate.csv",index=False)
V36_POLICY_SEARCH.to_csv(OUT_DIR/"v36_policy_search.csv",index=False)
V36_DEV_LEDGER.to_csv(OUT_DIR/"v36_development_interventions.csv",index=False)
V36_FEATURE_IMPORTANCE.to_csv(OUT_DIR/"v36_meta_feature_importance.csv",index=False)
V36_SEALED.to_csv(OUT_DIR/"v36_sealed_validation.csv",index=False)
V36_SEALED_ORACLE.to_csv(OUT_DIR/"v36_sealed_oracle.csv",index=False)
V36_SEALED_LEDGER.to_csv(OUT_DIR/"v36_sealed_interventions.csv",index=False)
V36_FULL_POLICY_SEARCH.to_csv(OUT_DIR/"v36_full_policy_search.csv",index=False)
V36_TEST_GEOMETRY.to_csv(OUT_DIR/"v36_test_geometry.csv",index=False)
TEST_LEDGERS["CORRIDOR"].to_csv(OUT_DIR/"v36_test_interventions_corridor.csv",index=False)
TEST_LEDGERS["ATTACK"].to_csv(OUT_DIR/"v36_test_interventions_attack.csv",index=False)
V36_SUPPORT_AUDIT.to_csv(OUT_DIR/"v36_intentstate_support_audit.csv",index=False)
with open(OUT_DIR/"v36_reproduction_audit.json","w") as f: json.dump(V36_REPRO_AUDIT,f,indent=2)
with open(OUT_DIR/"v36_meta_feature_names.json","w") as f: json.dump(full_meta_ds["feature_names"],f,indent=2)

slot_plan=pd.DataFrame([
    {
        "file":"submission_v36_primary.csv","role":"PRIMARY_ONLY_IF_ALL_GATES_PASS","eligible_to_submit":bool(FINAL_GATE),
        "note":f"{DEV_WINNER} only if fresh A/B/C + sealed D + chronology + exact Hybrid34 reconstruction pass; otherwise exact Hybrid34 0.66006 fail-close.",
    },
    {
        "file":"submission_v36_DIAGNOSTIC_corridor.csv","role":"DIAGNOSTIC_ONLY","eligible_to_submit":False,
        "note":"S2 unresolved-instability XE competence candidate. Never submit by bypassing the gate.",
    },
    {
        "file":"submission_v36_DIAGNOSTIC_attack.csv","role":"DIAGNOSTIC_ONLY","eligible_to_submit":False,
        "note":"S2+S0 XE competence candidate. S1 excluded and S3 public-proven Hybrid34 rows immutable.",
    },
    {
        "file":"submission_v36_anchor_hybrid34_reconstructed.csv","role":"CURRENT_PUBLIC_ANCHOR_REFERENCE","eligible_to_submit":False,
        "note":"Deterministic reconstruction of current public 0.66006 Hybrid34.",
    },
])
slot_plan.to_csv(OUT_DIR/"v36_slot_plan.csv",index=False)

summary={
    "version":"V36 Hybrid34-Locked State-Conditional Fold-Matched XE Proposal Competence Router",
    "public_anchor_hybrid34":PUBLIC_HYBRID34_LB,"leader_reference":PUBLIC_LEADER_LB,"leader_gap":PUBLIC_LEADER_GAP,
    "expert_train_size":EXPERT_TRAIN_SIZE,"shadow_folds":DEV_SHADOW_FOLDS,
    "fresh_worlds":FRESH_WORLDS,"development_winner":DEV_WINNER,"development_gate":bool(DEV_GATE),
    "sealed_gate":bool(SEALED_GATE),"v30_reproduction_ok":bool(V30_REPRO_OK),"hybrid34_reproduction_ok":bool(HYBRID_REPRO_OK),
    "final_gate":bool(FINAL_GATE),"development":V36_DEV.to_dict("records"),"oracle":V36_ORACLE.to_dict("records"),
    "test_geometry":V36_TEST_GEOMETRY.to_dict("records"),"full_policies":full_policies,
    "submission_instruction":(
        f"PRIMARY IS ELIGIBLE: {DEV_WINNER} passed all V36 gates. Audit the intervention ledger before spending a Kaggle slot."
        if FINAL_GATE else
        "NO NEW V36 SUBMISSION: primary is the exact reconstructed Hybrid34 0.66006 fail-close anchor."
    ),
}
with open(OUT_DIR/"v36_summary.json","w") as f: json.dump(summary,f,indent=2)

assert SUBMISSIONS["primary"].shape==sample.shape==SUBMISSIONS["anchor_hybrid34"].shape
assert HYBRID_REPRO_OK or V36_SMOKE
if DEV_WINNER is not None:
    assert DEV_WINNER in {"CORRIDOR","ATTACK"}
print(json.dumps(summary,indent=2))
print("\nV36 outputs written to",OUT_DIR)
print(slot_plan.to_string(index=False))

# ---- cell 36 (markdown) ----
# ### Interpretation contract
#
# - **Hybrid34 is the only fail-close anchor.** V36 never falls back to V30.
# - **S3 is immutable.** These are the 34 public-proven V31 overrides inside the V30-instability region.
# - **S1 is excluded.** The prior public ablation showed V31 changes in the V30-stable region are strongly harmful.
# - **XE is a proposal generator, not a replacement ranker. Yeti/median are metadata-only competence context.** Only stable Top-5 proposals that the competence router accepts can change Hybrid34.
# - **Transition-family aggregate heuristics are not action rules.** The failed two-row public probe is treated as negative evidence.
# - **Fresh A/B/C are measurement-only.** Policy thresholds come from training-side cross-fitted meta predictions.
# - **Oracle headroom is a prerequisite diagnostic.** If stable XE proposals do not contain sufficient utility, a selector cannot manufacture it.
# - **Sealed D and chronology can only reject.** No threshold or feature is changed after development.
# - **Reproducibility is explicit.** The notebook reconstructs V30, V31, state counts, the 34-user overlap rule, and the public Hybrid34 Top-5 signature from organizer data before producing any V36 primary file.
