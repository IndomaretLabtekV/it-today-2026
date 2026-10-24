# Converted from notebooks/v33.ipynb: markdown cells become comments, %/! magics neutralized.

# ---- cell 0 (markdown) ----
# # MineToday V33 — V30-Locked Retrieval-Augmented Residual Action Learning
#
# **Research contract.** V33 does not create a sibling ranking surface. The exact public-proven V30 geometry is Stage-1 and stays immutable. Stage-2 may only make a small adjacent Top-6 correction when train-only local residual memory and a low-capacity action-utility learner agree that V30 is likely wrong.
#
# Key constraints:
#
# - Exact V30 (`alpha = 0.16`, 174-feature residual LambdaRank) is the only Stage-1 control and fail-close submission.
# - The broad V31 Intent-State block is **not** used in V33.
# - Stage-2 training never receives in-sample V30 predictions. Every train action is built from a strict V30 OOF prediction produced by a model that excluded that user.
# - Retrieval memory uses organizer-provided training data only and self-excludes the query training user.
# - Fresh deployment-like worlds **N/O/P** are user-disjoint but intentionally permit repeated-template overlap, matching the observed train/test setting.
# - Historical template-disjoint **J** is preserved as a cold-template catastrophic-harm rejection test. Chronology is also rejection-only.
# - At most one adjacent Top-6 swap is applied per user. Weak/uncertain retrieval evidence abstains to exact V30.
# - No test labels, external data, LLM predictions, leaderboard-conditioned threshold search, or post-hoc salt generation are used.
#
# Default run is the full preregistered experiment. Set environment variable `V33_SMOKE=1` only for a cheap plumbing check; smoke output is never submission-eligible.

# ---- cell 1 (code) ----
# Runtime + preregistered V33 configuration
import os, sys, re, json, ast, html, hashlib, random, warnings, time, subprocess, gc, math
from pathlib import Path
from collections import defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ["PYTHONHASHSEED"] = "42"

V33_SKIP_INSTALL = bool(int(os.getenv("V33_SKIP_INSTALL", "0")))
try:
    import lightgbm as lgb
except Exception:
    if V33_SKIP_INSTALL:
        raise
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "lightgbm==4.6.0"])
    import lightgbm as lgb

import numpy as np
import pandas as pd
import scipy
import sklearn
from IPython.display import display
from scipy.sparse import hstack
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")

SEED = 42
V33_SMOKE = bool(int(os.getenv("V33_SMOKE", "0")))
V33_STRICT_GEOMETRY = bool(int(os.getenv("V33_STRICT_GEOMETRY", "1")))

# Compatibility aliases consumed by frozen V3.1/V25 reconstruction cells.
V30_SMOKE = V33_SMOKE
V30_STRICT_GEOMETRY = V33_STRICT_GEOMETRY
V29_SMOKE = V33_SMOKE
V29_STRICT_GEOMETRY = V33_STRICT_GEOMETRY
V27_SMOKE = V33_SMOKE
V27_STRICT_GEOMETRY = V33_STRICT_GEOMETRY

N_FOLDS = 2 if V33_SMOKE else 5
ANCHOR_INNER_FOLDS = 2 if V33_SMOKE else 4
STAGE1_OOF_FOLDS = 2 if V33_SMOKE else 3
FULL_STAGE1_OOF_FOLDS = 2 if V33_SMOKE else 4

RIDGE_ALPHA = 3.0
REFERENCE_W = 0.45
PUBLIC_CAL045_LB = 0.65639
PUBLIC_CONSENSUS3_LB = 0.65856
PUBLIC_V25_LB = 0.65919
PUBLIC_V30_LB = 0.65963
PUBLIC_V31_DIAGNOSTIC_LB = 0.65949
PUBLIC_RANK4_LB = 0.65974
PUBLIC_LEADER_LB = 0.66142

LGB_ESTIMATORS = 12 if V33_SMOKE else 1200
LGB_EARLY_STOP = 3 if V33_SMOKE else 80
RANKER_ESTIMATORS = 16 if V33_SMOKE else 360
RANKER_LEARNING_RATE = 0.04 if V33_SMOKE else 0.025
ACTION_ESTIMATORS = 30 if V33_SMOKE else 420
ACTION_LR = 0.05 if V33_SMOKE else 0.025
ALPHA = 0.16  # exact public-proven V30 residual strength

# Retrieval is fixed, not tuned after N/O/P.
MAX_NEIGHBORS = 24 if V33_SMOKE else 96
BROAD_MIN_SUPPORT = 1 if V33_SMOKE else 3
BROAD_MIN_EFF_N = 1.0 if V33_SMOKE else 2.0
BROAD_MIN_PROB = 0.50 if V33_SMOKE else 0.55
PROTECTED_MIN_SUPPORT = 1 if V33_SMOKE else 5
PROTECTED_MIN_EFF_N = 1.0 if V33_SMOKE else 3.0
PROTECTED_MIN_PROB = 0.50 if V33_SMOKE else 0.62

# Development gates. Broad is the leader-scale attack; protected is the safer progress candidate.
BROAD_DEV_MIN_POOLED = -1.0 if V33_SMOKE else 0.00150
BROAD_DEV_MIN_WORST = -1.0 if V33_SMOKE else 0.00050
BROAD_DEV_CI_FLOOR = -1.0 if V33_SMOKE else 0.00020
PROTECTED_DEV_MIN_POOLED = -1.0 if V33_SMOKE else 0.00035
PROTECTED_DEV_MIN_WORST = -1.0 if V33_SMOKE else 0.00005
PROTECTED_DEV_CI_FLOOR = -1.0 if V33_SMOKE else -0.00002
COLD_J_HARM_FLOOR = -1.0 if V33_SMOKE else -0.00020
CHRONO_HARM_FLOOR = -1.0 if V33_SMOKE else 0.0
BOOTSTRAPS = 30 if V33_SMOKE else 1200
GEOMETRY_ATOL = 2.5e-4

# Fresh deployment-like worlds. These are deliberately NOT template-group-disjoint.
def _seed_from_text(s):
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:8], 16) % (2**31-1)

FRESH_WORLDS = {
    "N": _seed_from_text("MINETODAY-V33-DEPLOY-N-20260829-RESIDUALMEMORY"),
    "O": _seed_from_text("MINETODAY-V33-DEPLOY-O-20260829-RESIDUALMEMORY"),
    "P": _seed_from_text("MINETODAY-V33-DEPLOY-P-20260829-RESIDUALMEMORY"),
}
SEALED_LOCK_SALT = "MINETODAY-V31-SEALED-J-20260828-INTENTSTATE-REJECTION-ONLY"

PROTECTED_TEST_USERS = {
    "U_4047", "U_4953", "U_4074", "U_4176", "U_4491", "U_4567", "U_4836", "U_4929"
}
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

random.seed(SEED)
np.random.seed(SEED)
print("numpy", np.__version__, "pandas", pd.__version__, "sklearn", sklearn.__version__, "lightgbm", lgb.__version__)
print("V33_SMOKE", V33_SMOKE)
print("fixed V30 alpha", ALPHA)
print("fresh deployment-like worlds", FRESH_WORLDS)
print("public V30", PUBLIC_V30_LB, "V31 diagnostic", PUBLIC_V31_DIAGNOSTIC_LB, "leader gap", round(PUBLIC_LEADER_LB-PUBLIC_V30_LB, 5))
try:
    print("GPU runtime:\n" + subprocess.check_output(["nvidia-smi", "-L"], text=True).strip())
except Exception:
    print("nvidia-smi unavailable. V33 is CPU-first and remains runnable.")

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
# The V3.1 sparse backbone, V3.1/V7 calibrator, Consensus3 profile, and V25 public-tested actions are reconstructed exactly before the V30-locked Stage-2 experiment adds any new information.

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
# These eight public-tested rows remain protected in the exact V30 deployment geometry. They are never used as supervised training labels for V33.

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
# ## Frozen V30 module-native surfaces
#
# The following module-level lexical, intent, readiness, and ordinal surfaces are inherited unchanged from V30. V31 Intent-State target encoding is intentionally absent.

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

# ---- cell 16 (markdown) ----
# ## Exact V30 Stage-1 helpers
#
# Every Stage-2 training row must be based on a V30 prediction from a pipeline that excluded that user. The expensive nested construction is intentional.

# ---- cell 17 (code) ----
def rank_matrix(scores):
    order=np.argsort(-np.asarray(scores),axis=1,kind="mergesort")
    ranks=np.empty_like(order,dtype=np.int16)
    ranks[np.arange(len(order))[:,None],order]=np.arange(1,N_MODULES+1,dtype=np.int16)[None,:]
    return ranks

def top5_order_change_rate(a,b):
    return float(np.mean(np.any(top5_indices(a)!=top5_indices(b),axis=1)))

def build_long_matrix_v30(rows_global,anchor,base,meta):
    # Exact 174-feature V30 Base surface from the public-winning notebook.
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
    mats=[
        anchor, ranks/float(N_MODULES), (ranks<=5).astype(np.float32), top1-anchor, anchor-cut5,
        prev_gap,next_gap,z,base,meta,meta-base,
        LEX_WORD[rows_global],LEX_CHAR[rows_global],MODULE_COUNT[rows_global],MODULE_RECENCY[rows_global],
        MODULE_LATEST[rows_global],MODULE_READINESS[rows_global],MODULE_INVALID[rows_global],
    ]
    names=[
        "anchor_score","anchor_rank_norm","anchor_top5","top1_minus_score","score_minus_cut5",
        "prev_gap","next_gap","anchor_z","base_score","meta_score","meta_minus_base",
        "lex_word","lex_char","topic_count","topic_recency","topic_latest","readiness","invalid_readiness",
    ]
    pieces=[np.column_stack([m.reshape(-1) for m in mats]).astype(np.float32)]
    pieces.append(np.repeat(USER_FEATURES[rows_global],N_MODULES,axis=0).astype(np.float32))
    names += [f"user_f{i:03d}" for i in range(USER_FEATURES.shape[1])]
    pieces.append(np.tile(np.eye(N_MODULES,dtype=np.float32),(n,1)))
    names += [f"module_{m}" for m in MODULES]
    pieces.append(np.tile(MODULE_LEVEL_ONEHOT,(n,1)).astype(np.float32))
    names += [f"module_level_{i}" for i in range(5)]
    XX=np.hstack(pieces).astype(np.float32,copy=False)
    assert XX.shape==(n*N_MODULES,174), XX.shape
    assert len(names)==174 and np.isfinite(XX).all()
    return XX,names

def make_lambdarank(seed):
    return lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg",
        n_estimators=RANKER_ESTIMATORS, learning_rate=RANKER_LEARNING_RATE,
        num_leaves=15, min_child_samples=80,
        reg_alpha=1.0, reg_lambda=10.0,
        colsample_bytree=0.78, subsample=0.90, subsample_freq=1,
        max_bin=127, lambdarank_truncation_level=8,
        label_gain=LABEL_GAIN,
        random_state=int(seed), n_jobs=4, verbosity=-1, force_col_wise=True,
    )

def align_ranker_to_anchor(raw_ranker,anchor):
    raw=np.asarray(raw_ranker,np.float32); anchor=np.asarray(anchor,np.float32)
    rmu=raw.mean(1,keepdims=True); rsd=np.maximum(raw.std(1,keepdims=True),1e-5)
    amu=anchor.mean(1,keepdims=True); asd=np.maximum(anchor.std(1,keepdims=True),1e-5)
    return (amu+(raw-rmu)*(asd/rsd)).astype(np.float32)

def blend_residual(anchor,raw_ranker,alpha=ALPHA):
    aligned=align_ranker_to_anchor(raw_ranker,anchor)
    return (anchor+float(alpha)*(aligned-anchor)).astype(np.float32)

def _make_cal_local(seed,n_estimators=None):
    return lgb.LGBMClassifier(
        n_estimators=int(LGB_ESTIMATORS if n_estimators is None else n_estimators),
        learning_rate=.03,num_leaves=15,min_child_samples=50,
        reg_alpha=.5,reg_lambda=5.,colsample_bytree=.7,
        random_state=int(seed),n_jobs=4,verbosity=-1
    )

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

def build_v30_split(train_rows,val_rows,seed):
    """Train the exact V30 local pipeline on train_rows and predict val_rows only."""
    train_rows=np.asarray(train_rows,int); val_rows=np.asarray(val_rows,int)
    anc=build_anchor_split_full(train_rows,val_rows,seed)
    Xtr,_=build_long_matrix_v30(train_rows,anc["cons_train"],anc["base_train"],anc["meta_train"])
    Xva,_=build_long_matrix_v30(val_rows,anc["cons_val"],anc["base_val"],anc["meta_val"])
    model=make_lambdarank(seed+313)
    group=np.full(len(train_rows),N_MODULES,dtype=np.int32)
    model.fit(Xtr,ORDINAL_GRADES[train_rows].reshape(-1),group=group)
    raw=model.predict(Xva).reshape(-1,N_MODULES).astype(np.float32)
    v30=blend_residual(anc["cons_val"],raw,ALPHA)
    out={"v30_val":v30,"anchor_val":anc["cons_val"]}
    del Xtr,Xva,model,anc,raw
    gc.collect()
    return out

def crossfit_v30_rows(rows,seed,n_splits=STAGE1_OOF_FOLDS):
    """Strict V30 OOF bank: each returned user prediction is from a pipeline excluding that user."""
    rows=np.asarray(rows,int)
    strata=user_strata_from_y(y[rows])
    ns=min(int(n_splits),max(2,int(pd.Series(strata).value_counts().min())))
    sk=StratifiedKFold(n_splits=ns,shuffle=True,random_state=int(seed))
    pred=np.zeros((len(rows),N_MODULES),np.float32)
    anchor=np.zeros_like(pred)
    fold_id=np.full(len(rows),-1,np.int16)
    for f,(lt,lv) in enumerate(sk.split(np.zeros(len(rows)),strata)):
        tr=rows[lt]; va=rows[lv]
        print(f"  strict Stage-1 OOF fold {f+1}/{ns}: train={len(tr)} val={len(va)}")
        res=build_v30_split(tr,va,seed+1009+f*211)
        pred[lv]=res["v30_val"]; anchor[lv]=res["anchor_val"]; fold_id[lv]=f
    assert (fold_id>=0).all() and np.isfinite(pred).all() and np.isfinite(anchor).all()
    return pred,anchor,fold_id

def world_bootstrap_ci(world_delta_dict,seed,B=BOOTSTRAPS):
    rng=np.random.default_rng(int(seed)); keys=list(world_delta_dict); z=[]
    for _ in range(int(B)):
        sampled_worlds=rng.choice(keys,size=len(keys),replace=True)
        means=[]
        for w in sampled_worlds:
            d=np.asarray(world_delta_dict[w],float)
            means.append(float(d[rng.integers(0,len(d),len(d))].mean()))
        z.append(float(np.mean(means)))
    return tuple(float(x) for x in np.quantile(z,[.025,.975]))

print("Exact V30 nested Stage-1 helpers ready")

# ---- cell 18 (markdown) ----
# ## V30-conditional retrieval memory
#
# Local memory is not a recommender. It estimates whether **V30's current ordered module pair** has historically been wrong among users sharing repeated learning messages. The query user is always excluded from its own training memory.

# ---- cell 19 (code) ----
# User retrieval profiles from organizer chat only.
USER_MSG_WEIGHTS=[]
USER_LATEST_CLEAN=[]
USER_LATEST_SKEL=[]
USER_LATEST_TOPICS=[]
for u in all_users:
    g=user_chat.get(u)
    if g is None or len(g)==0:
        learn=None
    else:
        learn=g[~g["is_noise"]].copy()
        if len(learn)==0: learn=g.copy()
    mw={}; latest_clean=""; latest_skel=""; latest_topics=frozenset()
    if learn is not None and len(learn):
        rows=list(learn.itertuples(index=False)); n=len(rows)
        for j,row in enumerate(rows):
            key=str(row.clean).strip()
            if key:
                w=float(np.exp(-0.7*(n-1-j)))
                mw[key]=max(mw.get(key,0.0),w)
        latest_clean=str(rows[-1].clean).strip()
        latest_skel=str(rows[-1].skeleton).strip()
        latest_topics=frozenset(rows[-1].topics)
    USER_MSG_WEIGHTS.append(mw)
    USER_LATEST_CLEAN.append(latest_clean)
    USER_LATEST_SKEL.append(latest_skel)
    USER_LATEST_TOPICS.append(latest_topics)


def _topic_jaccard(a,b):
    a=set(a); b=set(b)
    if not a and not b: return 1.0
    return len(a&b)/max(1,len(a|b))


def adjacent_action_table(rows,v30_scores):
    rows=np.asarray(rows,int); scores=np.asarray(v30_scores,np.float32)
    order=np.argsort(-scores,axis=1,kind="mergesort")[:,:6]
    rank=rank_matrix(scores)
    records=[]
    for i,grow in enumerate(rows):
        rel=y[grow]
        ideal=np.sort(rel)[-5:][::-1]
        idcg=float(np.sum(ideal*DISCOUNT)); idcg=max(idcg,1e-12)
        for p in range(5):
            hi=int(order[i,p]); lo=int(order[i,p+1])
            if p<4:
                utility=float((rel[lo]-rel[hi])*(DISCOUNT[p]-DISCOUNT[p+1])/idcg)
            else:
                utility=float((rel[lo]-rel[hi])*DISCOUNT[4]/idcg)
            records.append((i,int(grow),p,hi,lo,utility))
    return records


def fit_residual_memory(fit_rows,v30_scores):
    fit_rows=np.asarray(fit_rows,int); v30_scores=np.asarray(v30_scores,np.float32)
    local_of_global={int(g):i for i,g in enumerate(fit_rows)}
    exact_index=defaultdict(list); skel_index=defaultdict(list)
    for li,g in enumerate(fit_rows):
        for key in USER_MSG_WEIGHTS[g]: exact_index[key].append(li)
        sk=USER_LATEST_SKEL[g]
        if sk: skel_index[sk].append(li)

    ranks=rank_matrix(v30_scores)
    # Pair-global V30-conditional preference prior. Fit labels only.
    pair_support=np.zeros((N_MODULES,N_MODULES),np.int32)
    pair_sum=np.zeros((N_MODULES,N_MODULES),np.float64)
    pair_pos=np.zeros((N_MODULES,N_MODULES),np.float64)
    for li,g in enumerate(fit_rows):
        rel=y[g]
        for hi in range(N_MODULES):
            for lo in range(N_MODULES):
                if hi==lo or ranks[li,hi]>=ranks[li,lo]: continue
                d=float(rel[lo]-rel[hi])
                pair_support[hi,lo]+=1; pair_sum[hi,lo]+=d; pair_pos[hi,lo]+=float(d>0)

    # Sparse exact adjacent-utility lookup for the strongest memory statistic.
    adj_lookup={}
    for li,g,p,hi,lo,utility in adjacent_action_table(fit_rows,v30_scores):
        adj_lookup[(li,hi,lo)]=utility

    return {
        "fit_rows":fit_rows,"scores":v30_scores,"ranks":ranks,"local_of_global":local_of_global,
        "exact_index":dict(exact_index),"skel_index":dict(skel_index),
        "pair_support":pair_support,"pair_sum":pair_sum,"pair_pos":pair_pos,"adj_lookup":adj_lookup,
    }


def retrieval_stats(query_global,hi,lo,memory,exclude_global=None):
    q=int(query_global); qmw=USER_MSG_WEIGHTS[q]
    cand=set()
    for key in qmw:
        cand.update(memory["exact_index"].get(key,()))
    sk=USER_LATEST_SKEL[q]
    if sk: cand.update(memory["skel_index"].get(sk,()))
    ex_local=memory["local_of_global"].get(int(exclude_global),None) if exclude_global is not None else None
    if ex_local is not None: cand.discard(ex_local)

    scored=[]
    for li in cand:
        ng=int(memory["fit_rows"][li]); nmw=USER_MSG_WEIGHTS[ng]
        shared=sum(min(qmw[k],nmw[k]) for k in (set(qmw)&set(nmw)))
        latest_exact=float(bool(USER_LATEST_CLEAN[q]) and USER_LATEST_CLEAN[q]==USER_LATEST_CLEAN[ng])
        same_skel=float(bool(sk) and sk==USER_LATEST_SKEL[ng])
        tj=_topic_jaccard(USER_LATEST_TOPICS[q],USER_LATEST_TOPICS[ng])
        readiness_sim=float(np.exp(-np.mean(np.abs(A[q]-A[ng]))/1.5))
        w=(shared+1.5*latest_exact+0.35*same_skel+0.25*tj)*(0.4+0.6*readiness_sim)
        if w>0: scored.append((w,li,latest_exact,same_skel))
    scored.sort(reverse=True,key=lambda z:(z[0],-z[1]))
    if len(scored)>MAX_NEIGHBORS: scored=scored[:MAX_NEIGHBORS]

    vals=[]; weights=[]; adj_vals=[]; adj_w=[]; latest_n=0.; skel_n=0.
    for w,li,le,ss in scored:
        # Residual condition: neighbor V30 must make the same ordered preference hi > lo.
        if memory["ranks"][li,hi] >= memory["ranks"][li,lo]:
            continue
        ng=int(memory["fit_rows"][li])
        d=float(y[ng,lo]-y[ng,hi])
        vals.append(d); weights.append(w); latest_n+=le; skel_n+=ss
        au=memory["adj_lookup"].get((li,hi,lo))
        if au is not None:
            adj_vals.append(float(au)); adj_w.append(w)

    # Self-exclude the query from pair-global supervised priors as well.
    ps=int(memory["pair_support"][hi,lo])
    psum=float(memory["pair_sum"][hi,lo]); ppos=float(memory["pair_pos"][hi,lo])
    if ex_local is not None and memory["ranks"][ex_local,hi] < memory["ranks"][ex_local,lo]:
        self_d=float(y[int(exclude_global),lo]-y[int(exclude_global),hi])
        ps-=1; psum-=self_d; ppos-=float(self_d>0)
    pair_mean=float(psum/ps) if ps>0 else 0.0
    pair_pos=float(ppos/ps) if ps>0 else 0.5

    if weights:
        w=np.asarray(weights,float); v=np.asarray(vals,float); sw=float(w.sum())
        mu=float(np.sum(w*v)/sw); pos=float(np.sum(w*(v>0))/sw); harm=float(np.sum(w*(v<0))/sw)
        var=float(np.sum(w*(v-mu)**2)/sw); eff=float(sw*sw/max(np.sum(w*w),1e-12))
        se=float(math.sqrt(max(var,0.0)/max(eff,1.0))); lcb=mu-se
        maxw=float(w.max()/sw)
    else:
        mu=0.0; pos=0.5; harm=0.5; var=0.0; eff=0.0; se=0.0; lcb=0.0; maxw=0.0

    if adj_w:
        aw=np.asarray(adj_w,float); av=np.asarray(adj_vals,float)
        adj_mu=float(np.sum(aw*av)/max(aw.sum(),1e-12)); adj_support=len(adj_w)
    else:
        adj_mu=0.0; adj_support=0

    return np.asarray([
        len(weights),eff,mu,pos,harm,var,se,lcb,maxw,
        latest_n,skel_n,adj_support,adj_mu,ps,pair_mean,pair_pos,
    ],np.float32)

RETRIEVAL_FEATURE_NAMES=[
    "local_support","local_effective_n","local_pref_mean","local_pref_positive","local_pref_harm",
    "local_pref_var","local_pref_se","local_pref_lcb","local_max_weight_share",
    "local_latest_exact_n","local_latest_skeleton_n","local_adj_support","local_adj_utility_mean",
    "pair_global_support","pair_global_pref_mean","pair_global_pref_positive",
]
print("retrieval profiles ready; users with >=1 learning message:",sum(bool(x) for x in USER_MSG_WEIGHTS))

# ---- cell 20 (markdown) ----
# ## Stage-2 action-utility learner
#
# The target is the exact user-level ΔNDCG@5 from swapping one adjacent V30 Top-6 pair. Two predeclared surfaces are evaluated: a broad leader-scale correction and a V30-protected conservative correction.

# ---- cell 21 (code) ----
def action_features(rows,v30_scores,anchor_scores,memory,training=False):
    rows=np.asarray(rows,int); v30_scores=np.asarray(v30_scores,np.float32); anchor_scores=np.asarray(anchor_scores,np.float32)
    order=np.argsort(-v30_scores,axis=1,kind="mergesort")[:,:6]
    sd=np.maximum(v30_scores.std(1),1e-5)
    feats=[]; targets=[]; meta=[]
    eye=np.eye(N_MODULES,dtype=np.float32)
    for i,g in enumerate(rows):
        rel=y[g] if g<N_TRAIN else None
        if training:
            ideal=np.sort(rel)[-5:][::-1]; idcg=max(float(np.sum(ideal*DISCOUNT)),1e-12)
        for p in range(5):
            hi=int(order[i,p]); lo=int(order[i,p+1])
            mem=retrieval_stats(g,hi,lo,memory,exclude_global=(g if training else None))
            base=np.asarray([
                p/4.0,float(p==4),v30_scores[i,hi],v30_scores[i,lo],v30_scores[i,hi]-v30_scores[i,lo],
                (v30_scores[i,hi]-v30_scores[i,lo])/sd[i],anchor_scores[i,hi],anchor_scores[i,lo],
                anchor_scores[i,hi]-anchor_scores[i,lo],
                (v30_scores[i,hi]-anchor_scores[i,hi]),(v30_scores[i,lo]-anchor_scores[i,lo]),
                LEX_WORD[g,lo]-LEX_WORD[g,hi],LEX_CHAR[g,lo]-LEX_CHAR[g,hi],
                MODULE_RECENCY[g,lo]-MODULE_RECENCY[g,hi],MODULE_LATEST[g,lo]-MODULE_LATEST[g,hi],
                MODULE_READINESS[g,lo]-MODULE_READINESS[g,hi],MODULE_INVALID[g,lo]-MODULE_INVALID[g,hi],
                MODULE_LEVEL[hi],MODULE_LEVEL[lo],A[g].mean(),A_INVALID[g].mean(),
            ],np.float32)
            f=np.concatenate([base,mem,eye[hi],eye[lo]]).astype(np.float32)
            feats.append(f)
            if training:
                if p<4:
                    util=float((rel[lo]-rel[hi])*(DISCOUNT[p]-DISCOUNT[p+1])/idcg)
                else:
                    util=float((rel[lo]-rel[hi])*DISCOUNT[4]/idcg)
                targets.append(util)
            meta.append((i,int(g),p,hi,lo))
    Xf=np.vstack(feats).astype(np.float32)
    yf=np.asarray(targets,np.float32) if training else None
    return Xf,yf,meta

ACTION_BASE_NAMES=[
    "position_norm","boundary_5_6","v30_high","v30_low","v30_margin","v30_margin_z",
    "anchor_high","anchor_low","anchor_margin","residual_high","residual_low",
    "lex_word_delta","lex_char_delta","recency_delta","latest_delta","readiness_delta","invalid_delta",
    "high_level","low_level","assessment_mean","assessment_invalid_mean",
]
ACTION_FEATURE_NAMES=ACTION_BASE_NAMES+RETRIEVAL_FEATURE_NAMES+[f"high_{m}" for m in MODULES]+[f"low_{m}" for m in MODULES]


def make_action_regressor(seed):
    return lgb.LGBMRegressor(
        objective="huber",n_estimators=ACTION_ESTIMATORS,learning_rate=ACTION_LR,
        num_leaves=7,max_depth=4,min_child_samples=120,reg_alpha=2.0,reg_lambda=20.0,
        colsample_bytree=0.80,subsample=0.90,subsample_freq=1,
        random_state=int(seed),n_jobs=4,verbosity=-1,force_col_wise=True,
    )

def make_action_classifier(seed):
    return lgb.LGBMClassifier(
        objective="binary",n_estimators=ACTION_ESTIMATORS,learning_rate=ACTION_LR,
        num_leaves=7,max_depth=4,min_child_samples=120,reg_alpha=2.0,reg_lambda=20.0,
        colsample_bytree=0.80,subsample=0.90,subsample_freq=1,
        random_state=int(seed),n_jobs=4,verbosity=-1,force_col_wise=True,
    )

def fit_action_models(Xa,utility,seed):
    utility=np.asarray(utility,np.float32)
    # Small nonzero utilities matter most; zero-utility actions remain useful abstention negatives.
    sw=(0.75+np.minimum(2.0,20.0*np.abs(utility))).astype(np.float32)
    reg=make_action_regressor(seed); clf=make_action_classifier(seed+17)
    reg.fit(Xa,utility,sample_weight=sw)
    clf.fit(Xa,(utility>0).astype(np.int8),sample_weight=sw)
    return reg,clf

def apply_action_policy(v30_scores,anchor_scores,Xa,meta,reg,clf,mode,global_rows,hard_locked_rows=None):
    v30_scores=np.asarray(v30_scores,np.float32); out=v30_scores.copy()
    pred=reg.predict(Xa).astype(float); prob=clf.predict_proba(Xa)[:,1].astype(float)
    hard_locked_rows=set() if hard_locked_rows is None else set(int(x) for x in hard_locked_rows)
    by_user=defaultdict(list)
    # Retrieval feature offsets in the action vector.
    roff=len(ACTION_BASE_NAMES)
    i_support=roff+RETRIEVAL_FEATURE_NAMES.index("local_support")
    i_eff=roff+RETRIEVAL_FEATURE_NAMES.index("local_effective_n")
    i_mean=roff+RETRIEVAL_FEATURE_NAMES.index("local_pref_mean")
    i_lcb=roff+RETRIEVAL_FEATURE_NAMES.index("local_pref_lcb")
    i_pos=roff+RETRIEVAL_FEATURE_NAMES.index("local_pref_positive")
    for k,(li,g,p,hi,lo) in enumerate(meta):
        if li in hard_locked_rows: continue
        support=float(Xa[k,i_support]); eff=float(Xa[k,i_eff]); mu=float(Xa[k,i_mean]); lcb=float(Xa[k,i_lcb]); mpos=float(Xa[k,i_pos])
        if mode=="broad":
            eligible=(support>=BROAD_MIN_SUPPORT and eff>=BROAD_MIN_EFF_N and mu>0 and pred[k]>0 and prob[k]>=BROAD_MIN_PROB and mpos>0.5)
        elif mode=="protected":
            eligible=(support>=PROTECTED_MIN_SUPPORT and eff>=PROTECTED_MIN_EFF_N and lcb>0 and pred[k]>0 and prob[k]>=PROTECTED_MIN_PROB and mpos>0.5)
        else:
            raise ValueError(mode)
        if not eligible: continue
        confidence=min(1.0,eff/5.0)*max(0.0,2.0*abs(mpos-0.5))
        score=float(pred[k])*float(2*prob[k]-1.0)*confidence
        if score>0:
            by_user[li].append((score,k,p,hi,lo,support,eff,mu,lcb,prob[k]))
    ledger=[]
    for li,arr in by_user.items():
        score,k,p,hi,lo,support,eff,mu,lcb,pk=max(arr,key=lambda z:(z[0],-z[2]))
        out[li,hi],out[li,lo]=out[li,lo],out[li,hi]
        ledger.append({
            "local_row":li,"user_id":str(global_rows[li]),"mode":mode,"position_pair":f"{p+1}<->{p+2}",
            "higher_before":MODULES[hi],"promoted":MODULES[lo],"policy_score":score,
            "predicted_utility":float(pred[k]),"positive_probability":float(pk),"local_support":support,
            "local_effective_n":eff,"local_pref_mean":mu,"local_pref_lcb":lcb,
        })
    return out,pd.DataFrame(ledger)

print("Stage-2 action learner ready; features",len(ACTION_FEATURE_NAMES))

# ---- cell 22 (markdown) ----
# ## Fresh deployment-like worlds N/O/P
#
# Each world holds out 20% of users. Template overlap is allowed by design, while labels remain strictly user-disjoint. For every world, Stage-2 training actions use an inner strict V30 OOF bank.

# ---- cell 23 (code) ----
def deployment_split(seed):
    strata=user_strata_from_y(y)
    sss=StratifiedShuffleSplit(n_splits=1,test_size=(0.10 if V33_SMOKE else 0.20),random_state=int(seed))
    tr,va=next(sss.split(np.zeros(N_TRAIN),strata))
    assert len(set(tr)&set(va))==0
    return np.asarray(tr,int),np.asarray(va,int)

def locked_rows_from_anchor(v30_scores,anchor_scores):
    # OOF analogue of the public V30 protection principle: if V30 already changed Top-5 vs its anchor, protected policy abstains.
    return set(np.flatnonzero(np.any(top5_indices(v30_scores)!=top5_indices(anchor_scores),axis=1)).tolist())

def evaluate_world(train_rows,val_rows,seed,world,selected_mode=None):
    train_rows=np.asarray(train_rows,int); val_rows=np.asarray(val_rows,int)
    print(f"\n=== V33 {world}: train={len(train_rows)} val={len(val_rows)} ===")
    # Honest Stage-1 predictions for every Stage-2 training user.
    tr_v30,tr_anchor,_=crossfit_v30_rows(train_rows,seed+100,STAGE1_OOF_FOLDS)
    # Outer validation Stage-1 never sees validation users.
    va_stage1=build_v30_split(train_rows,val_rows,seed+200)
    va_v30=va_stage1["v30_val"]; va_anchor=va_stage1["anchor_val"]

    memory=fit_residual_memory(train_rows,tr_v30)
    Xa_tr,util_tr,_=action_features(train_rows,tr_v30,tr_anchor,memory,training=True)
    Xa_va,_,meta_va=action_features(val_rows,va_v30,va_anchor,memory,training=False)
    assert Xa_tr.shape[1]==Xa_va.shape[1]==len(ACTION_FEATURE_NAMES)
    reg,clf=fit_action_models(Xa_tr,util_tr,seed+300)

    modes=[selected_mode] if selected_mode is not None else ["broad","protected"]
    results={}; rows=[]; ledgers={}
    u0=per_user_ndcg(y[val_rows],va_v30)
    protect_lock=locked_rows_from_anchor(va_v30,va_anchor)
    for mode in modes:
        lock=protect_lock if mode=="protected" else set()
        cand,ledger=apply_action_policy(va_v30,va_anchor,Xa_va,meta_va,reg,clf,mode,val_rows,lock)
        uc=per_user_ndcg(y[val_rows],cand); inc=uc-u0
        results[mode]=(inc,cand)
        ledgers[mode]=ledger
        rows.append({
            "world":world,"mode":mode,"n":len(val_rows),"v30_ndcg":float(u0.mean()),"candidate_ndcg":float(uc.mean()),
            "increment_vs_v30":float(inc.mean()),"positive_rate":float(np.mean(inc>0)),"harm_rate":float(np.mean(inc<0)),
            "interventions":int(len(ledger)),"intervention_rate":float(len(ledger)/len(val_rows)),
            "top5_order_change":top5_order_change_rate(va_v30,cand),"top5_membership_change":top5_set_change_rate(va_v30,cand),
            "top1_change":top1_change_rate(va_v30,cand),"protected_lock_rate":float(len(protect_lock)/len(val_rows)),
            "retrieval_train_action_rows":int(len(util_tr)),"retrieval_positive_action_rate":float(np.mean(util_tr>0)),
        })
    del tr_v30,tr_anchor,va_stage1,va_v30,va_anchor,memory,Xa_tr,Xa_va,util_tr,reg,clf
    gc.collect()
    return pd.DataFrame(rows),results,ledgers

DEV_ROWS=[]; DEV_DELTA={"broad":{},"protected":{}}; DEV_LEDGERS=[]
for world,seed in FRESH_WORLDS.items():
    tr,va=deployment_split(seed)
    df,res,led=evaluate_world(tr,va,seed,world)
    DEV_ROWS.append(df)
    for mode in ("broad","protected"):
        DEV_DELTA[mode][world]=res[mode][0]
        if len(led[mode]):
            x=led[mode].copy(); x["world"]=world; DEV_LEDGERS.append(x)
V33_DEVELOPMENT=pd.concat(DEV_ROWS,ignore_index=True)
V33_DEV_LEDGER=pd.concat(DEV_LEDGERS,ignore_index=True) if DEV_LEDGERS else pd.DataFrame()
display(V33_DEVELOPMENT)

pooled=[]; gates=[]
for mode in ("broad","protected"):
    world_means={w:float(np.mean(d)) for w,d in DEV_DELTA[mode].items()}
    mean=float(np.mean(list(world_means.values()))); worst=float(np.min(list(world_means.values())))
    ci=world_bootstrap_ci(DEV_DELTA[mode],_seed_from_text(f"V33-{mode}-BOOT"))
    if mode=="broad":
        passed=(all(v>0 for v in world_means.values()) and mean>=BROAD_DEV_MIN_POOLED and worst>=BROAD_DEV_MIN_WORST and ci[0]>=BROAD_DEV_CI_FLOOR)
    else:
        passed=(all(v>0 for v in world_means.values()) and mean>=PROTECTED_DEV_MIN_POOLED and worst>=PROTECTED_DEV_MIN_WORST and ci[0]>=PROTECTED_DEV_CI_FLOOR)
    pooled.append({"mode":mode,"pooled_increment":mean,"worst_world":worst,"ci_low":ci[0],"ci_high":ci[1],**{f"world_{k}":v for k,v in world_means.items()}})
    gates.append({"mode":mode,"development_gate":bool(passed)})
V33_POOLED=pd.DataFrame(pooled); V33_GATES=pd.DataFrame(gates)
display(V33_POOLED); display(V33_GATES)

# Precommitted selection ladder: leader-scale broad first, then safer protected.
if bool(V33_GATES.loc[V33_GATES["mode"]=="broad","development_gate"].iloc[0]):
    DEV_WINNER="broad"
elif bool(V33_GATES.loc[V33_GATES["mode"]=="protected","development_gate"].iloc[0]):
    DEV_WINNER="protected"
else:
    DEV_WINNER=None
DEV_GATE=DEV_WINNER is not None
print("DEV_WINNER",DEV_WINNER,"DEV_GATE",DEV_GATE)

# ---- cell 24 (markdown) ----
# ## Sealed J + chronology rejection tests
#
# J remains a cold-template stress test. Retrieval is expected to weaken there, so J only rejects catastrophic harm. Chronology remains a stricter transfer check. No tuning is allowed after these are opened.

# ---- cell 25 (code) ----
def sealed_lock_bucket(group):
    h=hashlib.sha1(f"{SEALED_LOCK_SALT}::{group}".encode("utf-8")).hexdigest()
    return int(h[:12],16)%5

SEALED_LOCK_MASK=np.asarray([sealed_lock_bucket(g)==0 for g in TRAIN_GROUP],bool)
SEALED_DEV_IDX=np.flatnonzero(~SEALED_LOCK_MASK); SEALED_LOCK_IDX=np.flatnonzero(SEALED_LOCK_MASK)
assert len(set(TRAIN_GROUP[SEALED_DEV_IDX]).intersection(set(TRAIN_GROUP[SEALED_LOCK_IDX])))==0
CHRONO_TR=np.arange(0,3000,dtype=int); CHRONO_VA=np.arange(3000,4000,dtype=int)

if DEV_GATE:
    sealed=[]
    for tr,va,seed,world in [
        (SEALED_DEV_IDX,SEALED_LOCK_IDX,_seed_from_text("V33-SEALED-J-FIT"),"sealed_template_lock_J"),
        (CHRONO_TR,CHRONO_VA,_seed_from_text("V33-CHRONO-FIT"),"chronological_U3001_U4000"),
    ]:
        df,res,led=evaluate_world(tr,va,seed,world,selected_mode=DEV_WINNER)
        sealed.append(df)
    V33_SEALED=pd.concat(sealed,ignore_index=True)
    j=float(V33_SEALED.loc[V33_SEALED.world=="sealed_template_lock_J","increment_vs_v30"].iloc[0])
    ch=float(V33_SEALED.loc[V33_SEALED.world=="chronological_U3001_U4000","increment_vs_v30"].iloc[0])
    SEALED_GATE=bool(j>=COLD_J_HARM_FLOOR and ch>=CHRONO_HARM_FLOOR)
else:
    V33_SEALED=pd.DataFrame([{"world":"skipped_development_gate_failed","mode":None,"increment_vs_v30":np.nan}])
    SEALED_GATE=False

display(V33_SEALED)
print("SEALED_GATE",SEALED_GATE)

# ---- cell 26 (markdown) ----
# ## Full-train deployment
#
# The final test surface starts from an exact reconstruction of public V30. Stage-2 is trained on a strict full-train V30 OOF bank. Protected mode additionally locks every test user whose V30 Top-5 already differs from the V25 anchor, preventing the exact-reversion failure observed in V31/V32.

# ---- cell 27 (code) ----
OUT_DIR=Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("/mnt/data")
OUT_DIR.mkdir(parents=True,exist_ok=True)

# Exact public V30 full-test reconstruction from the frozen 174-feature surface.
Xtr_full,_=build_long_matrix_v30(np.arange(N_TRAIN),consensus3_oof,base_oof,meta_oof)
Xte_full,_=build_long_matrix_v30(np.arange(N_TRAIN,N_TRAIN+N_TEST),consensus3_test,base_test,meta_test)
full_rank_seed=_seed_from_text("V30-FULL-RANKER")
model_v30=make_lambdarank(full_rank_seed)
model_v30.fit(Xtr_full,ORDINAL_GRADES.reshape(-1),group=np.full(N_TRAIN,N_MODULES,dtype=np.int32))
raw_v30_test=model_v30.predict(Xte_full).reshape(N_TEST,N_MODULES).astype(np.float32)
v30_test=blend_residual(consensus3_test,raw_v30_test,ALPHA)

# Preserve the eight public-tested V25 rows exactly, matching V30 deployment contract.
protected_v25_rows=[TEST_UID_TO_ROW[u] for u in sorted(PROTECTED_TEST_USERS)]
v30_test[protected_v25_rows]=v25_anchor_test[protected_v25_rows]

# Exact public V30 geometry audit.
v25_top=top5_indices(v25_anchor_test); v30_top=top5_indices(v30_test)
v30_changed_rows=np.flatnonzero(np.any(v25_top!=v30_top,axis=1))
v30_changed_users=set(test.iloc[v30_changed_rows]["user_id"].astype(str))
v30_top1_changes=int(np.sum(v25_top[:,0]!=v30_top[:,0]))
v30_membership_changes=int(np.sum([set(a)!=set(b) for a,b in zip(v25_top,v30_top)]))
V30_REPRO_OK=bool(v30_changed_users==EXPECTED_V30_CHANGED_USERS and v30_top1_changes==EXPECTED_V30_TOP1_CHANGES and v30_membership_changes==EXPECTED_V30_MEMBERSHIP_CHANGES)
V30_REPRO_AUDIT={
    "expected_changed_users":len(EXPECTED_V30_CHANGED_USERS),"actual_changed_users":len(v30_changed_users),
    "expected_top1_changes":EXPECTED_V30_TOP1_CHANGES,"actual_top1_changes":v30_top1_changes,
    "expected_membership_changes":EXPECTED_V30_MEMBERSHIP_CHANGES,"actual_membership_changes":v30_membership_changes,
    "exact_changed_user_set_match":bool(v30_changed_users==EXPECTED_V30_CHANGED_USERS),"reproduction_ok":V30_REPRO_OK,
}
print(json.dumps(V30_REPRO_AUDIT,indent=2))
if not V33_SMOKE:
    assert V30_REPRO_OK, "Exact V30 rank geometry failed; no V33 submission may be produced from this run."

# Strict full-train OOF V30 bank for Stage-2. No train action sees an in-sample V30 prediction.
full_rows=np.arange(N_TRAIN,dtype=int)
v30_train_oof,anchor_train_oof,full_stage1_fold=crossfit_v30_rows(full_rows,_seed_from_text("V33-FULL-STAGE1-OOF"),FULL_STAGE1_OOF_FOLDS)
assert np.isfinite(v30_train_oof).all()

memory_full=fit_residual_memory(full_rows,v30_train_oof)
Xa_train,util_train,_=action_features(full_rows,v30_train_oof,anchor_train_oof,memory_full,training=True)
test_global_rows=np.arange(N_TRAIN,N_TRAIN+N_TEST,dtype=int)
Xa_test,_,meta_test_actions=action_features(test_global_rows,v30_test,consensus3_test,memory_full,training=False)
reg_full,clf_full=fit_action_models(Xa_train,util_train,_seed_from_text("V33-FULL-ACTION"))

# Broad locks only the eight explicit public-tested V25 rows. Protected additionally locks all 77 public V30 intervention rows.
broad_hard_lock=set(protected_v25_rows)
protected_hard_lock=set(v30_changed_rows.tolist())|set(protected_v25_rows)
BROAD_TEST,BROAD_LEDGER=apply_action_policy(v30_test,consensus3_test,Xa_test,meta_test_actions,reg_full,clf_full,"broad",test["user_id"].astype(str).tolist(),broad_hard_lock)
PROTECTED_TEST,PROTECTED_LEDGER=apply_action_policy(v30_test,consensus3_test,Xa_test,meta_test_actions,reg_full,clf_full,"protected",test["user_id"].astype(str).tolist(),protected_hard_lock)

# Lock invariants are exact score invariants, not merely ranking invariants.
for r in protected_v25_rows:
    assert np.array_equal(BROAD_TEST[r],v30_test[r]) and np.array_equal(PROTECTED_TEST[r],v30_test[r])
for r in v30_changed_rows:
    assert np.array_equal(PROTECTED_TEST[r],v30_test[r])

FINAL_GATE=bool(DEV_GATE and SEALED_GATE and V30_REPRO_OK and not V33_SMOKE)
if FINAL_GATE:
    selected_test=BROAD_TEST if DEV_WINNER=="broad" else PROTECTED_TEST
else:
    selected_test=v30_test.copy()

assert sample["user_id"].astype(str).tolist()==test["user_id"].astype(str).tolist()
def submission_from_scores(scores):
    out=sample.copy(); out.loc[:,MODULES]=np.asarray(scores,np.float32)
    assert np.isfinite(out[MODULES].to_numpy()).all(); return out

SUBMISSIONS={
    "anchor_v30":submission_from_scores(v30_test),
    "broad":submission_from_scores(BROAD_TEST),
    "protected":submission_from_scores(PROTECTED_TEST),
    "primary":submission_from_scores(selected_test),
}
SUBMISSIONS["anchor_v30"].to_csv(OUT_DIR/"submission_v33_anchor_v30_reconstructed.csv",index=False)
SUBMISSIONS["broad"].to_csv(OUT_DIR/"submission_v33_DIAGNOSTIC_broad_residual.csv",index=False)
SUBMISSIONS["protected"].to_csv(OUT_DIR/"submission_v33_DIAGNOSTIC_protected_residual.csv",index=False)
SUBMISSIONS["primary"].to_csv(OUT_DIR/"submission_v33_primary.csv",index=False)
if not FINAL_GATE:
    assert np.array_equal(SUBMISSIONS["primary"][MODULES].to_numpy(),SUBMISSIONS["anchor_v30"][MODULES].to_numpy())

# Test geometry + ledgers.
def test_geom(name,s):
    return {"candidate":name,"top5_order_changed_users":int(np.sum(np.any(top5_indices(v30_test)!=top5_indices(s),axis=1))),
            "top5_order_change_rate":top5_order_change_rate(v30_test,s),"top5_membership_change_rate":top5_set_change_rate(v30_test,s),
            "top1_change_rate":top1_change_rate(v30_test,s)}
V33_TEST_GEOMETRY=pd.DataFrame([test_geom("broad",BROAD_TEST),test_geom("protected",PROTECTED_TEST)])
display(V33_TEST_GEOMETRY)
print("full Stage-2 training actions",len(util_train),"positive rate",float(np.mean(util_train>0)))
print("test interventions broad/protected",len(BROAD_LEDGER),len(PROTECTED_LEDGER),"FINAL_GATE",FINAL_GATE,"DEV_WINNER",DEV_WINNER)

# Free largest matrices only after all deployment surfaces are constructed.
del Xtr_full,Xte_full,model_v30,raw_v30_test,Xa_train,Xa_test,reg_full,clf_full
gc.collect()

# ---- cell 28 (markdown) ----
# ## Artifacts and submission contract
#
# `submission_v33_primary.csv` is submission-eligible **only** when the preregistered development and rejection gates pass. Otherwise it is an exact V30 fail-close copy. Diagnostic files are always written for analysis and must not bypass the gate.

# ---- cell 29 (code) ----
V33_DEVELOPMENT.to_csv(OUT_DIR/"v33_development_worlds.csv",index=False)
V33_POOLED.to_csv(OUT_DIR/"v33_pooled_incremental.csv",index=False)
V33_GATES.to_csv(OUT_DIR/"v33_development_gate.csv",index=False)
V33_SEALED.to_csv(OUT_DIR/"v33_sealed_validation.csv",index=False)
V33_TEST_GEOMETRY.to_csv(OUT_DIR/"v33_test_geometry.csv",index=False)
if len(V33_DEV_LEDGER): V33_DEV_LEDGER.to_csv(OUT_DIR/"v33_development_interventions.csv",index=False)
BROAD_LEDGER.to_csv(OUT_DIR/"v33_test_interventions_broad.csv",index=False)
PROTECTED_LEDGER.to_csv(OUT_DIR/"v33_test_interventions_protected.csv",index=False)
pd.DataFrame({"feature":ACTION_FEATURE_NAMES}).to_csv(OUT_DIR/"v33_action_feature_names.csv",index=False)
with open(OUT_DIR/"v33_v30_reconstruction_audit.json","w") as f: json.dump(V30_REPRO_AUDIT,f,indent=2)
np.savez_compressed(OUT_DIR/"v33_stage1_oof_artifacts.npz",v30_train_oof=v30_train_oof,anchor_train_oof=anchor_train_oof,stage1_fold=full_stage1_fold)

slot_plan=pd.DataFrame([
    {"file":"submission_v33_primary.csv","role":"PRIMARY_ONLY_IF_ALL_GATES_PASS","eligible_to_submit":bool(FINAL_GATE),
     "note":f"Selected mode={DEV_WINNER}; otherwise exact V30 fail-close."},
    {"file":"submission_v33_DIAGNOSTIC_broad_residual.csv","role":"DIAGNOSTIC_ONLY","eligible_to_submit":False,
     "note":"Leader-scale V30-locked residual-memory attack; never bypass the gate."},
    {"file":"submission_v33_DIAGNOSTIC_protected_residual.csv","role":"DIAGNOSTIC_ONLY","eligible_to_submit":False,
     "note":"Conservative residual-memory surface; locks all 77 public V30 intervention rows."},
    {"file":"submission_v33_anchor_v30_reconstructed.csv","role":"CURRENT_PUBLIC_ANCHOR_REFERENCE","eligible_to_submit":False,
     "note":"Exact V30 reference, public 0.65963."},
])
slot_plan.to_csv(OUT_DIR/"v33_slot_plan.csv",index=False)

summary={
    "version":"V33 V30-Locked Retrieval-Augmented Residual Action Learning",
    "public_v30":PUBLIC_V30_LB,"public_v31_diagnostic":PUBLIC_V31_DIAGNOSTIC_LB,"leader_reference":PUBLIC_LEADER_LB,
    "leader_gap":PUBLIC_LEADER_LB-PUBLIC_V30_LB,"fixed_alpha":ALPHA,"fresh_worlds":FRESH_WORLDS,
    "stage1_oof_folds":STAGE1_OOF_FOLDS,"full_stage1_oof_folds":FULL_STAGE1_OOF_FOLDS,
    "development":V33_DEVELOPMENT.to_dict("records"),"pooled":V33_POOLED.to_dict("records"),
    "development_winner":DEV_WINNER,"development_gate_passed":bool(DEV_GATE),"sealed_gate_passed":bool(SEALED_GATE),
    "v30_reproduction_ok":bool(V30_REPRO_OK),"final_gate_passed":bool(FINAL_GATE),
    "test_geometry":V33_TEST_GEOMETRY.to_dict("records"),"broad_test_interventions":int(len(BROAD_LEDGER)),
    "protected_test_interventions":int(len(PROTECTED_LEDGER)),
    "submission_instruction":("PRIMARY IS SUBMISSION-ELIGIBLE after full V33 protocol." if FINAL_GATE else "NO NEW V33 SUBMISSION: primary is exact V30 fail-close."),
}
with open(OUT_DIR/"v33_summary.json","w") as f: json.dump(summary,f,indent=2)

assert set(V33_DEVELOPMENT["world"])==set(FRESH_WORLDS)
assert set(V33_DEVELOPMENT["mode"])=={"broad","protected"}
assert len(V33_DEVELOPMENT)==6
assert SUBMISSIONS["primary"].shape==sample.shape
print(json.dumps(summary,indent=2))
print("\nV33 outputs written to",OUT_DIR)
print(slot_plan.to_string(index=False))

# ---- cell 30 (markdown) ----
# ### Interpretation contract
#
# - **Do not submit a diagnostic file merely because one world is strong.** N/O/P were predeclared as a package.
# - If `final_gate_passed` is false, `submission_v33_primary.csv` is deliberately identical to reconstructed V30.
# - A positive deployment-like result with cold-J near zero is acceptable because retrieval is designed to abstain under cold templates; meaningful cold-J harm is not.
# - V33 is considered a useful scientific result even when it fail-closes: it directly tests whether repeated-chat neighborhoods contain **errors-after-V30** signal rather than raw relevance signal.
