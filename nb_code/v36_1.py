# Converted from notebooks/v36_1.ipynb: markdown cells become comments, %/! magics neutralized.

# ---- cell 0 (markdown) ----
# # MineToday V36.1 — Hybrid34-Locked Cross-Shadow XE Residual Pressure Stability
#
# **Objective:** test the last justified XE hypothesis after V36: although discrete XE Top-5 and atomic rank-flip identities are unstable across 2700-user shadows, the *continuous direction of residual pressure* may still be stable before a rank crossing occurs.
#
# V36.1 therefore never asks four shadows to emit the same full ranking. For each authoritative Hybrid34 Top-6 adjacent pair, each shadow measures the standardized change in the pair margin from its own Hybrid34 analogue to XE-NDCG. A candidate correction exists only when the lower-ranked module is consistently pushed upward across shadows.
#
# The public **0.66006 Hybrid34** is immutable. S3 is locked, S1 is excluded, all score multisets are preserved, and at most one adjacent correction is allowed per user. Fresh E/F/G are measurement-only. Sealed D and chronology are rejection-only. A new submission is emitted as `primary` only if a precommitted leader-scale pressure policy survives every gate; otherwise `primary` is exact Hybrid34 fail-close.
#
# This notebook is intended for Kaggle with a T4x2 runtime. The ranking models remain CPU-first for deterministic LightGBM behavior; the two GPUs may be present but are deliberately not required.

# ---- cell 1 (code) ----
# Runtime + preregistered V36.1 configuration
import os, sys, re, json, ast, html, hashlib, random, warnings, time, subprocess, gc, math
from pathlib import Path
from collections import Counter, defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ["PYTHONHASHSEED"] = "42"

V361_SKIP_INSTALL = bool(int(os.getenv("V361_SKIP_INSTALL", "0")))
try:
    import lightgbm as lgb
except Exception:
    if V361_SKIP_INSTALL:
        raise
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "lightgbm==4.6.0"])
    import lightgbm as lgb

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
V361_SMOKE = bool(int(os.getenv("V361_SMOKE", "0")))
V361_STRICT_GEOMETRY = bool(int(os.getenv("V361_STRICT_GEOMETRY", "1")))

# Compatibility aliases consumed by the frozen V3.1/V25/V30/V31/V36 helper cells.
V36_SMOKE = V361_SMOKE
V36_STRICT_GEOMETRY = V361_STRICT_GEOMETRY
V31_SMOKE = V361_SMOKE
V30_SMOKE = V361_SMOKE
V30_STRICT_GEOMETRY = V361_STRICT_GEOMETRY
V29_SMOKE = V361_SMOKE
V29_STRICT_GEOMETRY = V361_STRICT_GEOMETRY
V27_SMOKE = V361_SMOKE
V27_STRICT_GEOMETRY = V361_STRICT_GEOMETRY

N_FOLDS = 2 if V361_SMOKE else 5
ANCHOR_INNER_FOLDS = 2 if V361_SMOKE else 4
INTENT_INNER_FOLDS = 2 if V361_SMOKE else 4

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

LGB_ESTIMATORS = 12 if V361_SMOKE else 1200
LGB_EARLY_STOP = 3 if V361_SMOKE else 80
RANKER_ESTIMATORS = 16 if V361_SMOKE else 360
RANKER_LEARNING_RATE = 0.04 if V361_SMOKE else 0.025
# Kept only because frozen helper definitions contain Yeti symbols; V36.1 never calls them.
YETI_ITERATIONS = 16 if V361_SMOKE else 180
YETI_LEARNING_RATE = 0.07 if V361_SMOKE else 0.035

ALPHA = 0.16
INTENT_PARENT_STRENGTH = 40.0
INTENT_CHILD_STRENGTH = 30.0

# Fold-matched pressure shadows.
DEV_VAL_SIZE = 80 if V361_SMOKE else 400
DEV_SHADOW_FOLDS = 2 if V361_SMOKE else 4
EXPERT_TRAIN_SIZE = 300 if V361_SMOKE else 2700
FINAL_SHADOW_FOLDS = 2 if V361_SMOKE else 4
MIN_PRESSURE_VOTES = 1 if V361_SMOKE else 3

# Label-free fixed action budgets. These are fixed before E/F/G labels are observed.
CORRIDOR_COVERAGE = 0.05 if V361_SMOKE else 0.010  # S2 only
ATTACK_COVERAGE = 0.10 if V361_SMOKE else 0.020    # S0 + S2

# Promotion gates. ATTACK is the only leader-scale path; CORRIDOR is a smaller fallback improvement path.
CORRIDOR_MIN_POOLED = -1.0 if V361_SMOKE else 0.00035
CORRIDOR_MIN_WORST = -1.0 if V361_SMOKE else 0.00000
CORRIDOR_MIN_PRECISION = 0.50 if V361_SMOKE else 0.67
CORRIDOR_ORACLE_MIN_WORST = -1.0 if V361_SMOKE else 0.00070

ATTACK_MIN_POOLED = -1.0 if V361_SMOKE else 0.00150
ATTACK_MIN_WORST = -1.0 if V361_SMOKE else 0.00065
ATTACK_MIN_PRECISION = 0.50 if V361_SMOKE else 0.72
ATTACK_ORACLE_MIN_WORST = -1.0 if V361_SMOKE else 0.00160
ATTACK_MIN_ACTIONS_PER_WORLD = 1 if V361_SMOKE else 4

BOOTSTRAPS = 20 if V361_SMOKE else 1200
GEOMETRY_ATOL = 2.5e-4


def _seed_from_text(s):
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:8], 16) % (2**31-1)

# A/B/C were burned by V36. E/F/G are new and are never used to tune a threshold/grid.
FRESH_WORLDS = {
    "E": _seed_from_text("MINETODAY-V361-FRESH-E-20260830-XE-PRESSURE"),
    "F": _seed_from_text("MINETODAY-V361-FRESH-F-20260830-XE-PRESSURE"),
    "G": _seed_from_text("MINETODAY-V361-FRESH-G-20260830-XE-PRESSURE"),
}
# V36 never opened sealed D because its development gate failed. Keep the exact same D salt as a rejection-only surface.
SEALED_D_SALT = "MINETODAY-V36-SEALED-D-20260830-XE-COMPETENCE"

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
EXPECTED_V31_CHANGED_VS_V30 = 60
EXPECTED_HYBRID_OVERLAP = 34
EXPECTED_STATE_COUNTS = {"S0":897, "S1":26, "S2":43, "S3":34}
EXPECTED_HYBRID_TOP5_SHA256 = "47f7db52ce8f782b8b393fd120140f1b6f51aeac452a9fabb5794cc9d8ba5e6e"

random.seed(SEED)
np.random.seed(SEED)
print("numpy", np.__version__, "pandas", pd.__version__, "sklearn", sklearn.__version__, "lightgbm", lgb.__version__)
print("V361_SMOKE", V361_SMOKE, "Hybrid34", PUBLIC_HYBRID34_LB, "leader gap", round(PUBLIC_LEADER_GAP,5))
print("fresh worlds", FRESH_WORLDS)
try:
    print("GPU runtime:\n" + subprocess.check_output(["nvidia-smi", "-L"], text=True).strip())
except Exception:
    print("nvidia-smi unavailable; V36.1 remains runnable. Ranking is deliberately CPU-first for reproducibility.")

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
# ## Hybrid34 state machine and frozen ranking surfaces
#
# The public result defines four states from `V25 → V30 → V31`: S0 stable/stable, S1 stable then V31-change (public-toxic), S2 V30-change then V31-abstain, and S3 V30-change plus V31-change (the 34-row public-positive region). V36.1 uses the same exact state machine. S1 is never touched and S3 is immutable.

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
# ## Fold-matched XE surfaces
#
# V36.1 keeps the V36 fold-matching principle: each shadow XE expert is trained on exactly 2700 users. Unlike V36, shadows are **not** required to emit the same Top-5 tuple. Their continuous standardized residual pressure on the same authoritative Hybrid34 adjacent pair is aggregated instead.

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
# ## Continuous residual pressure instead of discrete XE votes
#
# For authoritative Hybrid34 Top-6 pair `(higher, lower)`, each shadow produces
#
# `pressure = [(z_XE(lower)-z_XE(higher)) - (z_HybridShadow(lower)-z_HybridShadow(higher))]`.
#
# Positive pressure means XE pushes the lower-ranked module upward relative to the higher-ranked module, even if the shadow does not cross the ranking boundary. Only adjacent Hybrid34 Top-6 boundaries are considered, which includes the Top-5 membership boundary `#5↔#6`.
#
# The selection score is label-free and preregistered: 3/4 directional consistency, median pressure, dispersion, authoritative margin, and the exact NDCG positional discount gap. E/F/G labels only measure this fixed rule.

# ---- cell 23 (code) ----
def fit_pressure_bundle(train_rows,pred_rows,seed):
    """V30/V31/Hybrid34 + XE only. No Yeti/median side experts are trained in V36.1."""
    train_rows=np.asarray(train_rows,int); pred_rows=np.asarray(pred_rows,int)
    anc=build_anchor_split_full(train_rows,pred_rows,seed+101)
    intent_tr=crossfit_intentstate_features(
        y[train_rows],ORDINAL_GRADES[train_rows],TRAIN_INTENT[train_rows],MODULE_READY_BAND[train_rows],
        TRAIN_GROUP[train_rows],seed+211
    )
    ib=fit_intentstate_bundle(y[train_rows],ORDINAL_GRADES[train_rows],TRAIN_INTENT[train_rows],MODULE_READY_BAND[train_rows])
    intent_pr=apply_intentstate_bundle(ib,LATEST_INTENT_KEY[pred_rows],MODULE_READY_BAND[pred_rows])
    Xtr,_,bounds=build_long_matrix_v31(train_rows,anc["cons_train"],anc["base_train"],anc["meta_train"],intent_tr)
    Xpr,_,_=build_long_matrix_v31(pred_rows,anc["cons_val"],anc["base_val"],anc["meta_val"],intent_pr)
    rank_seed=seed+401
    m0,r0=fit_ranker_predict("lambda",Xtr[:,:bounds["base"]],Xpr[:,:bounds["base"]],ORDINAL_GRADES[train_rows],len(train_rows),rank_seed)
    m1,r1=fit_ranker_predict("lambda",Xtr[:,:bounds["intentstate"]],Xpr[:,:bounds["intentstate"]],ORDINAL_GRADES[train_rows],len(train_rows),rank_seed)
    mx,rx=fit_ranker_predict("xe",Xtr[:,:bounds["intentstate"]],Xpr[:,:bounds["intentstate"]],ORDINAL_GRADES[train_rows],len(train_rows),rank_seed)
    v30=blend_residual(anc["cons_val"],r0,ALPHA)
    v31=blend_residual(anc["cons_val"],r1,ALPHA)
    xe=blend_residual(anc["cons_val"],rx,ALPHA)
    hybrid,st=hybrid34_surface(anc["cons_val"],v30,v31)
    out={"pre":anc["cons_val"].astype(np.float32),"v30":v30,"v31":v31,"hybrid":hybrid,"xe":xe,"state":st}
    del Xtr,Xpr,intent_tr,intent_pr,ib,m0,m1,mx,anc
    gc.collect()
    return out


def build_outer_pressure_shadows(train_pool,outer_val,seed):
    train_pool=np.asarray(train_pool,int); outer_val=np.asarray(outer_val,int)
    folds=make_shadow_folds(train_pool,DEV_SHADOW_FOLDS,seed+101)
    shadows=[]
    for f,(tr0,_) in enumerate(folds):
        fit_rows=deterministic_train_subset(tr0,EXPERT_TRAIN_SIZE,seed+1009+f*131)
        b=fit_pressure_bundle(fit_rows,outer_val,seed+2003+f*173)
        shadows.append(b)
    return shadows


def zrows(a):
    a=np.asarray(a,np.float32)
    mu=a.mean(1,keepdims=True); sd=np.maximum(a.std(1,keepdims=True),1e-5)
    return ((a-mu)/sd).astype(np.float32)

# Exact metric leverage of adjacent positions. Rank #6 contributes zero to NDCG@5.
_POSITION_LEVERAGE=np.asarray([
    DISCOUNT[0]-DISCOUNT[1], DISCOUNT[1]-DISCOUNT[2], DISCOUNT[2]-DISCOUNT[3],
    DISCOUNT[3]-DISCOUNT[4], DISCOUNT[4]
],np.float32)
_POSITION_LEVERAGE/=float(_POSITION_LEVERAGE.max())


def swap_adjacent_score_levels(row,higher,lower):
    out=np.asarray(row,np.float32).copy()
    out[int(higher)],out[int(lower)]=out[int(lower)],out[int(higher)]
    if not np.array_equal(np.sort(out),np.sort(np.asarray(row,np.float32))):
        raise AssertionError("adjacent swap changed score multiset")
    return out


def pressure_action_table(reference,rows_global,shadows,with_labels=True):
    rows_global=np.asarray(rows_global,int)
    ref_h=np.asarray(reference["hybrid"],np.float32); ref_state=np.asarray(reference["state"],np.int8)
    if len(shadows)!=DEV_SHADOW_FOLDS and not V361_SMOKE:
        raise AssertionError((len(shadows),DEV_SHADOW_FOLDS))
    ref_z=zrows(ref_h)
    sh_h=[zrows(b["hybrid"]) for b in shadows]
    sh_x=[zrows(b["xe"]) for b in shadows]
    top6=np.argsort(-ref_h,axis=1,kind="mergesort")[:,:6]
    records=[]
    for i,rg in enumerate(rows_global):
        for p in range(5):
            hi=int(top6[i,p]); lo=int(top6[i,p+1])
            pressures=[]; crosses=[]
            for hz,xz,b in zip(sh_h,sh_x,shadows):
                pr=float((xz[i,lo]-xz[i,hi])-(hz[i,lo]-hz[i,hi]))
                pressures.append(pr)
                crosses.append(bool(float(b["xe"][i,lo])>float(b["xe"][i,hi])))
            arr=np.asarray(pressures,float)
            votes=int(np.sum(arr>0)); med=float(np.median(arr)); mad=float(np.median(np.abs(arr-med)))
            ref_margin=float(ref_z[i,hi]-ref_z[i,lo])
            snr=float(max(med,0.0)/(mad+0.05))
            relative=float(max(med,0.0)/(max(ref_margin,0.0)+0.05))
            vote_frac=float(votes/max(1,len(arr)))
            stability_score=float((vote_frac**2)*snr*relative*float(_POSITION_LEVERAGE[p]))
            rec={
                "row_local":i,"row_global":int(rg),"state":int(ref_state[i]),"state_name":str(STATE_NAMES[ref_state[i]]),
                "position":int(p+1),"higher_idx":hi,"lower_idx":lo,"higher_mid":MODULES[hi],"lower_mid":MODULES[lo],
                "pressure_votes":votes,"pressure_vote_fraction":vote_frac,"median_pressure":med,"pressure_mad":mad,
                "pressure_snr":snr,"reference_margin_z":ref_margin,"pressure_to_margin":relative,
                "xe_cross_votes":int(np.sum(crosses)),"position_leverage":float(_POSITION_LEVERAGE[p]),
                "stability_score":stability_score,
            }
            if with_labels:
                if rg>=N_TRAIN: raise ValueError("labels requested for test row")
                base=float(per_user_ndcg(y[[rg]],ref_h[[i]])[0])
                sw=swap_adjacent_score_levels(ref_h[i],hi,lo)
                alt=float(per_user_ndcg(y[[rg]],sw[None,:])[0])
                rec["delta_ndcg"]=float(alt-base)
            records.append(rec)
    return pd.DataFrame(records)


def scope_state_ok(state,scope):
    state=int(state)
    return (state==2) if scope=="CORRIDOR" else (state in (0,2))


def fixed_coverage(scope):
    return CORRIDOR_COVERAGE if scope=="CORRIDOR" else ATTACK_COVERAGE


def select_pressure_policy(table,n_users,scope,blocked_rows=None):
    d=table.copy()
    d=d[d["state"].map(lambda x: scope_state_ok(x,scope))]
    d=d[(d.pressure_votes>=MIN_PRESSURE_VOTES)&(d.median_pressure>0)&np.isfinite(d.stability_score)]
    if blocked_rows is not None:
        d=d[~d.row_local.isin(set(int(x) for x in blocked_rows))]
    # One canonical pressure action per user: keep the strongest label-free stability score.
    d=d.sort_values(["stability_score","pressure_votes","position"],ascending=[False,False,True])
    d=d.drop_duplicates("row_local",keep="first")
    k=max(1,int(math.ceil(float(fixed_coverage(scope))*int(n_users))))
    return d.head(k).copy()


def apply_pressure_selection(reference,selected):
    out=np.asarray(reference["hybrid"],np.float32).copy()
    for r in selected.itertuples(index=False):
        out[int(r.row_local)]=swap_adjacent_score_levels(out[int(r.row_local)],int(r.higher_idx),int(r.lower_idx))
    return out


def pressure_oracle_summary(table,n_users,scope,min_votes):
    d=table.copy()
    d=d[d["state"].map(lambda x: scope_state_ok(x,scope))]
    d=d[(d.pressure_votes>=int(min_votes))&(d.median_pressure>0)]
    if len(d)==0:
        return {"stable_actions":0,"stable_users":0,"oracle_gain":0.0,"oracle_top1pct_gain":0.0,"positive_actions":0}
    # Oracle is allowed at most one positive action per user.
    pos=d[d.delta_ndcg>0].sort_values("delta_ndcg",ascending=False).drop_duplicates("row_local")
    gain=float(pos.delta_ndcg.sum()/int(n_users)) if len(pos) else 0.0
    k=max(1,int(math.ceil(0.01*int(n_users))))
    top=float(pos.delta_ndcg.head(k).sum()/int(n_users)) if len(pos) else 0.0
    return {"stable_actions":int(len(d)),"stable_users":int(d.row_local.nunique()),"oracle_gain":gain,"oracle_top1pct_gain":top,"positive_actions":int(len(pos))}

print("V36.1 continuous pressure helpers ready; positional leverage",_POSITION_LEVERAGE.tolist())

# ---- cell 24 (markdown) ----
# ## Fresh E/F/G: supply audit first, fixed policy second
#
# Each world independently reconstructs a Hybrid34 analogue from its outer-train pool and obtains four 2700-user XE shadows. The notebook first measures whether **3/4 continuous pressure** retains enough positive oracle supply. Only then is the fixed, label-free CORRIDOR/ATTACK selection evaluated. No E/F/G threshold or coverage search is performed.

# ---- cell 25 (code) ----
def make_fresh_holdout_v361(seed,val_size=DEV_VAL_SIZE):
    rows=np.arange(N_TRAIN,dtype=int)
    strata=np.sum(Y_BIN,axis=1).astype(int)
    sp=StratifiedShuffleSplit(n_splits=1,test_size=int(val_size),random_state=int(seed))
    tr,va=next(sp.split(rows,strata))
    return rows[tr],rows[va]


def evaluate_pressure_world(train_pool,val_rows,seed,world):
    train_pool=np.asarray(train_pool,int); val_rows=np.asarray(val_rows,int)
    print(f"\n=== V36.1 {world}: train={len(train_pool)} val={len(val_rows)} ===")
    reference=fit_reference_hybrid(train_pool,val_rows,seed+17)
    shadows=build_outer_pressure_shadows(train_pool,val_rows,seed+31)
    tab=pressure_action_table(reference,val_rows,shadows,with_labels=True)
    base=per_user_ndcg(y[val_rows],reference["hybrid"])
    rows=[]; ledgers={}; oracle_rows=[]; user_deltas={}
    for scope in ("CORRIDOR","ATTACK"):
        for mv in (2,3,4):
            o=pressure_oracle_summary(tab,len(val_rows),scope,mv)
            oracle_rows.append({"world":world,"scope":scope,"min_votes":mv,**o})
        sel=select_pressure_policy(tab,len(val_rows),scope)
        cand=apply_pressure_selection(reference,sel)
        uc=per_user_ndcg(y[val_rows],cand); inc=uc-base
        user_deltas[scope]=inc
        if len(sel):
            realized=inc[sel.row_local.to_numpy(int)]
            sel=sel.copy(); sel["realized_delta_ndcg"]=realized
            ben=int(np.sum(realized>0)); harm=int(np.sum(realized<0)); tie=int(np.sum(np.isclose(realized,0)))
        else:
            ben=harm=tie=0
        ledgers[scope]=sel
        ci=cluster_boot_ci(inc,TRAIN_GROUP[val_rows],seed+101+(0 if scope=="CORRIDOR" else 1))
        rows.append({
            "world":world,"scope":scope,"n_val":len(val_rows),"reference_ndcg":float(base.mean()),
            "candidate_ndcg":float(uc.mean()),"increment_vs_hybrid34":float(inc.mean()),
            "ci_low":ci[0],"ci_high":ci[1],"actions":int(len(sel)),"beneficial_actions":ben,"harmful_actions":harm,"tie_actions":tie,
            "precision":float(ben/max(1,ben+harm)),"mean_selected_pressure":float(sel.median_pressure.mean()) if len(sel) else 0.0,
            "mean_selected_score":float(sel.stability_score.mean()) if len(sel) else 0.0,
            "top1_changes":int(np.sum(sel.position==1)) if len(sel) else 0,
            "membership_changes":int(np.sum(sel.position==5)) if len(sel) else 0,
        })
    all_led=[]
    for scope,ld in ledgers.items():
        if len(ld):
            x=ld.copy(); x["scope"]=scope; x["world"]=world; all_led.append(x)
    del reference,shadows
    gc.collect()
    return pd.DataFrame(rows),pd.DataFrame(oracle_rows),pd.concat(all_led,ignore_index=True) if all_led else pd.DataFrame(),tab

DEV_ROWS=[]; ORACLE_ROWS=[]; LEDGERS=[]; ACTION_TABLES=[]
for w,seed in FRESH_WORLDS.items():
    tr,va=make_fresh_holdout_v361(seed,DEV_VAL_SIZE)
    wr,orr,led,tab=evaluate_pressure_world(tr,va,seed,w)
    DEV_ROWS.append(wr); ORACLE_ROWS.append(orr)
    if len(led): LEDGERS.append(led)
    tab=tab.copy(); tab["world"]=w; ACTION_TABLES.append(tab)

V361_DEV=pd.concat(DEV_ROWS,ignore_index=True)
V361_ORACLE=pd.concat(ORACLE_ROWS,ignore_index=True)
V361_DEV_LEDGER=pd.concat(LEDGERS,ignore_index=True) if LEDGERS else pd.DataFrame()
V361_ACTION_AUDIT=pd.concat(ACTION_TABLES,ignore_index=True)
display(V361_DEV)
display(V361_ORACLE)

# ---- cell 26 (markdown) ----
# ## Development gate
#
# The **supply gate** asks whether 3/4 directional pressure still contains enough oracle utility after moving from discrete ranking votes to continuous margins. The **policy gate** is stricter: the fixed label-free action budget must actually capture that supply consistently.
#
# ATTACK is leader-scale only. CORRIDOR may become a smaller first-submission candidate if it is robust, but it is not described as a leader attack unless the ATTACK gate passes.

# ---- cell 27 (code) ----
gate_rows=[]
for scope in ("CORRIDOR","ATTACK"):
    d=V361_DEV[V361_DEV.scope==scope].copy()
    o=V361_ORACLE[(V361_ORACLE.scope==scope)&(V361_ORACLE.min_votes==3)].copy()
    pooled=float(d.increment_vs_hybrid34.mean()); worst=float(d.increment_vs_hybrid34.min())
    all_pos=bool((d.increment_vs_hybrid34>0).all())
    min_prec=float(d.precision.min()); min_actions=int(d.actions.min())
    oracle_worst=float(o.oracle_gain.min())
    if scope=="ATTACK":
        passed=bool(
            all_pos and pooled>=ATTACK_MIN_POOLED and worst>=ATTACK_MIN_WORST and
            min_prec>=ATTACK_MIN_PRECISION and min_actions>=ATTACK_MIN_ACTIONS_PER_WORLD and
            oracle_worst>=ATTACK_ORACLE_MIN_WORST
        )
    else:
        passed=bool(
            all_pos and pooled>=CORRIDOR_MIN_POOLED and worst>=CORRIDOR_MIN_WORST and
            min_prec>=CORRIDOR_MIN_PRECISION and oracle_worst>=CORRIDOR_ORACLE_MIN_WORST
        )
    gate_rows.append({"scope":scope,"pooled_increment":pooled,"worst_world":worst,"all_worlds_positive":all_pos,
                      "min_precision":min_prec,"min_actions":min_actions,"oracle_3of4_worst":oracle_worst,"development_pass":passed})
V361_GATE=pd.DataFrame(gate_rows)
display(V361_GATE)

ATTACK_PASS=bool(V361_GATE.loc[V361_GATE.scope=="ATTACK","development_pass"].iloc[0])
CORRIDOR_PASS=bool(V361_GATE.loc[V361_GATE.scope=="CORRIDOR","development_pass"].iloc[0])
DEV_WINNER="ATTACK" if ATTACK_PASS else ("CORRIDOR" if CORRIDOR_PASS else None)
DEV_GATE=DEV_WINNER is not None
print("DEV_WINNER =",DEV_WINNER,"DEV_GATE =",DEV_GATE)
if not DEV_GATE:
    print("Pressure hypothesis did not earn a deployment attempt. Primary must remain exact Hybrid34.")

# ---- cell 28 (markdown) ----
# ## Sealed D + chronology: rejection-only
#
# These surfaces are opened only after E/F/G promote a fixed scope. The same fixed coverage, 3/4 pressure rule, state scope and label-free stability score are used unchanged. Sealed D remains template-group-disjoint; chronology uses U0001–U3600 → U3601–U4000.

# ---- cell 29 (code) ----
def sealed_d_bucket_v361(group):
    h=hashlib.sha1(f"{SEALED_D_SALT}::{group}".encode("utf-8")).hexdigest()
    return int(h[:12],16)%10

SEALED_D_MASK=np.asarray([sealed_d_bucket_v361(g)==0 for g in TRAIN_GROUP],bool)
SEALED_D_TR=np.flatnonzero(~SEALED_D_MASK); SEALED_D_VA=np.flatnonzero(SEALED_D_MASK)
assert len(set(TRAIN_GROUP[SEALED_D_TR]).intersection(set(TRAIN_GROUP[SEALED_D_VA])))==0
CHRONO_TR=np.arange(0,3600,dtype=int); CHRONO_VA=np.arange(3600,4000,dtype=int)

SEALED_ROWS=[]; SEALED_LED=[]; SEALED_ORACLE=[]
if DEV_GATE:
    for name,tr,va,seed in [
        ("SEALED_D",SEALED_D_TR,SEALED_D_VA,_seed_from_text("V361-SEALED-D-RUN")),
        ("CHRONO",CHRONO_TR,CHRONO_VA,_seed_from_text("V361-CHRONO-RUN")),
    ]:
        reference=fit_reference_hybrid(tr,va,seed+17)
        shadows=build_outer_pressure_shadows(tr,va,seed+31)
        tab=pressure_action_table(reference,va,shadows,with_labels=True)
        o=pressure_oracle_summary(tab,len(va),DEV_WINNER,3)
        sel=select_pressure_policy(tab,len(va),DEV_WINNER)
        base=per_user_ndcg(y[va],reference["hybrid"])
        cand=apply_pressure_selection(reference,sel); inc=per_user_ndcg(y[va],cand)-base
        if len(sel):
            realized=inc[sel.row_local.to_numpy(int)]; sel=sel.copy(); sel["realized_delta_ndcg"]=realized
            ben=int(np.sum(realized>0)); harm=int(np.sum(realized<0))
        else: ben=harm=0
        SEALED_ROWS.append({"surface":name,"scope":DEV_WINNER,"n_val":len(va),"increment_vs_hybrid34":float(inc.mean()),
                            "actions":int(len(sel)),"beneficial_actions":ben,"harmful_actions":harm,
                            "precision":float(ben/max(1,ben+harm)),"oracle_3of4_gain":float(o["oracle_gain"])})
        if len(sel):
            x=sel.copy(); x["surface"]=name; SEALED_LED.append(x)
        SEALED_ORACLE.append({"surface":name,"scope":DEV_WINNER,**o})
        del reference,shadows
        gc.collect()

V361_SEALED=pd.DataFrame(SEALED_ROWS)
V361_SEALED_LEDGER=pd.concat(SEALED_LED,ignore_index=True) if SEALED_LED else pd.DataFrame()
V361_SEALED_ORACLE=pd.DataFrame(SEALED_ORACLE)
if DEV_GATE:
    display(V361_SEALED)
    # Rejection only. ATTACK must remain positive on both; CORRIDOR may be near-flat on cold-template D but never materially harmful.
    if DEV_WINNER=="ATTACK":
        SEALED_GATE=bool(len(V361_SEALED)==2 and (V361_SEALED.increment_vs_hybrid34>0).all() and (V361_SEALED.oracle_3of4_gain>=0.0005).all())
    else:
        SEALED_GATE=bool(len(V361_SEALED)==2 and (V361_SEALED.increment_vs_hybrid34>=-0.00005).all())
else:
    SEALED_GATE=False
print("SEALED_GATE =",SEALED_GATE)

# ---- cell 30 (markdown) ----
# ## Exact Hybrid34 reconstruction and fold-matched test pressure
#
# The current public 0.66006 anchor is regenerated from organizer data. V36.1 then trains four 2700-user XE pressure shadows on the full 4,000-user training universe. The final action policy uses the exact same label-free rule as development. S1/S3 and the eight public-protected rows are immutable.

# ---- cell 31 (code) ----
OUT_DIR=Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("/mnt/data")
OUT_DIR.mkdir(parents=True,exist_ok=True)

# Exact full V30/V31/Hybrid34 reconstruction.
full_intent_tr=crossfit_intentstate_features(
    y,ORDINAL_GRADES,TRAIN_INTENT,MODULE_READY_BAND[:N_TRAIN],TRAIN_GROUP,_seed_from_text("V31-FULL-INTENTSTATE-CROSSFIT")
)
FULL_INTENT_BUNDLE=fit_intentstate_bundle(y,ORDINAL_GRADES,TRAIN_INTENT,MODULE_READY_BAND[:N_TRAIN])
full_intent_te=apply_intentstate_bundle(FULL_INTENT_BUNDLE,TEST_INTENT,MODULE_READY_BAND[N_TRAIN:])
V361_SUPPORT_AUDIT=intentstate_support_audit(FULL_INTENT_BUNDLE,TEST_INTENT,MODULE_READY_BAND[N_TRAIN:])
Xtr_full,_,bounds_full=build_long_matrix_v31(np.arange(N_TRAIN),consensus3_oof,base_oof,meta_oof,full_intent_tr)
Xte_full,_,bounds_test=build_long_matrix_v31(np.arange(N_TRAIN,N_TRAIN+N_TEST),consensus3_test,base_test,meta_test,full_intent_te)
assert bounds_full==bounds_test
full_rank_seed=_seed_from_text("V30-FULL-RANKER")
model_v30,raw_v30=fit_ranker_predict("lambda",Xtr_full[:,:bounds_full["base"]],Xte_full[:,:bounds_full["base"]],ORDINAL_GRADES,N_TRAIN,full_rank_seed)
model_v31,raw_v31=fit_ranker_predict("lambda",Xtr_full[:,:bounds_full["intentstate"]],Xte_full[:,:bounds_full["intentstate"]],ORDINAL_GRADES,N_TRAIN,full_rank_seed)
v30_test=blend_residual(consensus3_test,raw_v30,ALPHA); v31_test=blend_residual(consensus3_test,raw_v31,ALPHA)
protected_rows=[TEST_UID_TO_ROW[u] for u in sorted(PROTECTED_TEST_USERS)]
v30_test[protected_rows]=v25_anchor_test[protected_rows]; v31_test[protected_rows]=v25_anchor_test[protected_rows]
full_state=state_codes(v25_anchor_test,v30_test,v31_test)
hybrid34_test=v30_test.copy(); hybrid34_test[full_state==3]=v31_test[full_state==3]

v25_top=top5_indices(v25_anchor_test); v30_top=top5_indices(v30_test); v31_top=top5_indices(v31_test)
v30_changed_rows=np.flatnonzero(np.any(v25_top!=v30_top,axis=1)); v31_changed_rows=np.flatnonzero(np.any(v30_top!=v31_top,axis=1))
v30_changed_users=set(test.iloc[v30_changed_rows]["user_id"].astype(str))
state_count={s:int(np.sum(full_state==i)) for i,s in enumerate(STATE_NAMES)}
hybrid_hash=hashlib.sha256(top5_indices(hybrid34_test).astype(np.int16).tobytes()).hexdigest()
V30_REPRO_OK=bool(v30_changed_users==EXPECTED_V30_CHANGED_USERS and int(np.sum(v25_top[:,0]!=v30_top[:,0]))==EXPECTED_V30_TOP1_CHANGES and int(np.sum([set(a)!=set(b) for a,b in zip(v25_top,v30_top)]))==EXPECTED_V30_MEMBERSHIP_CHANGES)
HYBRID_REPRO_OK=bool(len(v31_changed_rows)==EXPECTED_V31_CHANGED_VS_V30 and int(np.sum(full_state==3))==EXPECTED_HYBRID_OVERLAP and state_count==EXPECTED_STATE_COUNTS and hybrid_hash==EXPECTED_HYBRID_TOP5_SHA256)
V361_REPRO_AUDIT={"v30_reproduction_ok":V30_REPRO_OK,"hybrid34_reproduction_ok":HYBRID_REPRO_OK,"v30_changed_users":len(v30_changed_rows),"v31_changed_vs_v30":len(v31_changed_rows),"state_counts":state_count,"hybrid_top5_sha256":hybrid_hash}
print(json.dumps(V361_REPRO_AUDIT,indent=2))
if not V361_SMOKE:
    assert V30_REPRO_OK and HYBRID_REPRO_OK

# Four fold-matched 2700-user pressure shadows predict Kaggle test.
all_train=np.arange(N_TRAIN,dtype=int); test_global=np.arange(N_TRAIN,N_TRAIN+N_TEST,dtype=int)
final_folds=make_shadow_folds(all_train,FINAL_SHADOW_FOLDS,_seed_from_text("V361-FINAL-SHADOW-FOLDS"))
test_shadows=[]
for f,(tr0,_) in enumerate(final_folds):
    fit_rows=deterministic_train_subset(tr0,EXPERT_TRAIN_SIZE,_seed_from_text(f"V361-FINAL-SHADOW-TRAIN-{f}"))
    b=fit_pressure_bundle(fit_rows,test_global,_seed_from_text(f"V361-FINAL-SHADOW-MODEL-{f}"))
    # Keep historical protected rows inert inside shadow pressure as well.
    for key in ("pre","v30","v31","hybrid","xe"):
        b[key][protected_rows]=v25_anchor_test[protected_rows]
    b["state"]=state_codes(b["pre"],b["v30"],b["v31"])
    b["hybrid"],_=hybrid34_surface(b["pre"],b["v30"],b["v31"])
    test_shadows.append(b)

final_reference={"hybrid":hybrid34_test,"state":full_state}
V361_TEST_PRESSURE=pressure_action_table(final_reference,test_global,test_shadows,with_labels=False)
TEST_CANDIDATES={}; TEST_LEDGERS={}
for scope in ("CORRIDOR","ATTACK"):
    sel=select_pressure_policy(V361_TEST_PRESSURE,N_TEST,scope,blocked_rows=protected_rows)
    cand=apply_pressure_selection(final_reference,sel)
    changed=np.any(top5_indices(cand)!=top5_indices(hybrid34_test),axis=1)
    assert not np.any(changed & np.isin(full_state,[1,3]))
    assert not np.any(changed[protected_rows])
    TEST_CANDIDATES[scope]=cand; TEST_LEDGERS[scope]=sel

FINAL_GATE=bool(DEV_GATE and SEALED_GATE and V30_REPRO_OK and HYBRID_REPRO_OK)
selected_test=TEST_CANDIDATES[DEV_WINNER].copy() if DEV_WINNER is not None else hybrid34_test.copy()
primary_test=selected_test.copy() if FINAL_GATE else hybrid34_test.copy()

def submission_from_scores(scores):
    out=sample.copy(); out.loc[:,MODULES]=np.asarray(scores,np.float32)
    assert np.isfinite(out[MODULES].to_numpy()).all(); return out

SUBMISSIONS={
    "anchor":submission_from_scores(hybrid34_test),
    "corridor":submission_from_scores(TEST_CANDIDATES["CORRIDOR"]),
    "attack":submission_from_scores(TEST_CANDIDATES["ATTACK"]),
    "primary":submission_from_scores(primary_test),
}
SUBMISSIONS["anchor"].to_csv(OUT_DIR/"submission_v36_1_anchor_hybrid34_reconstructed.csv",index=False)
SUBMISSIONS["corridor"].to_csv(OUT_DIR/"submission_v36_1_DIAGNOSTIC_corridor_pressure.csv",index=False)
SUBMISSIONS["attack"].to_csv(OUT_DIR/"submission_v36_1_DIAGNOSTIC_attack_pressure.csv",index=False)
SUBMISSIONS["primary"].to_csv(OUT_DIR/"submission_v36_1_primary.csv",index=False)
if not FINAL_GATE:
    assert np.array_equal(SUBMISSIONS["primary"][MODULES].to_numpy(),SUBMISSIONS["anchor"][MODULES].to_numpy())

V361_TEST_GEOMETRY=pd.DataFrame([{
    "scope":scope,
    "changed_users_vs_hybrid34":int(np.sum(np.any(top5_indices(TEST_CANDIDATES[scope])!=top5_indices(hybrid34_test),axis=1))),
    "top1_changes":int(np.sum(top5_indices(TEST_CANDIDATES[scope])[:,0]!=top5_indices(hybrid34_test)[:,0])),
    "membership_changes":int(np.sum([set(a)!=set(b) for a,b in zip(top5_indices(TEST_CANDIDATES[scope]),top5_indices(hybrid34_test))])),
    "selected_actions":int(len(TEST_LEDGERS[scope])),
} for scope in ("CORRIDOR","ATTACK")])
display(V361_TEST_GEOMETRY)
print("FINAL_GATE =",FINAL_GATE,"DEV_WINNER =",DEV_WINNER)

del Xtr_full,Xte_full,full_intent_tr,full_intent_te,model_v30,model_v31
gc.collect()

# ---- cell 32 (markdown) ----
# ## Artifacts and submission contract
#
# `submission_v36_1_primary.csv` is eligible only when fresh E/F/G, 3/4 pressure oracle supply, fixed-policy performance, sealed D, chronology, and exact Hybrid34 reconstruction all pass. If not, it is exactly the 0.66006 Hybrid34 fail-close anchor. Diagnostic files are exported for analysis and must not be submitted by bypassing the gate.

# ---- cell 33 (code) ----
V361_DEV.to_csv(OUT_DIR/"v36_1_pressure_worlds.csv",index=False)
V361_ORACLE.to_csv(OUT_DIR/"v36_1_pressure_oracle.csv",index=False)
V361_GATE.to_csv(OUT_DIR/"v36_1_development_gate.csv",index=False)
V361_DEV_LEDGER.to_csv(OUT_DIR/"v36_1_development_interventions.csv",index=False)
V361_ACTION_AUDIT.to_csv(OUT_DIR/"v36_1_pressure_action_audit.csv",index=False)
V361_SEALED.to_csv(OUT_DIR/"v36_1_sealed_validation.csv",index=False)
V361_SEALED_ORACLE.to_csv(OUT_DIR/"v36_1_sealed_oracle.csv",index=False)
V361_SEALED_LEDGER.to_csv(OUT_DIR/"v36_1_sealed_interventions.csv",index=False)
V361_TEST_PRESSURE.to_csv(OUT_DIR/"v36_1_test_pressure_audit.csv",index=False)
V361_TEST_GEOMETRY.to_csv(OUT_DIR/"v36_1_test_geometry.csv",index=False)
TEST_LEDGERS["CORRIDOR"].to_csv(OUT_DIR/"v36_1_test_interventions_corridor.csv",index=False)
TEST_LEDGERS["ATTACK"].to_csv(OUT_DIR/"v36_1_test_interventions_attack.csv",index=False)
V361_SUPPORT_AUDIT.to_csv(OUT_DIR/"v36_1_intentstate_support_audit.csv",index=False)
with open(OUT_DIR/"v36_1_reproduction_audit.json","w") as f: json.dump(V361_REPRO_AUDIT,f,indent=2)

slot_plan=pd.DataFrame([
    {"file":"submission_v36_1_primary.csv","role":"PRIMARY_ONLY_IF_ALL_GATES_PASS","eligible_to_submit":bool(FINAL_GATE),"note":f"{DEV_WINNER} continuous-pressure candidate if every gate passes; otherwise exact Hybrid34 0.66006."},
    {"file":"submission_v36_1_DIAGNOSTIC_attack_pressure.csv","role":"DIAGNOSTIC_ONLY","eligible_to_submit":False,"note":"S0+S2 3/4 continuous-pressure policy. Never bypass the gate."},
    {"file":"submission_v36_1_DIAGNOSTIC_corridor_pressure.csv","role":"DIAGNOSTIC_ONLY","eligible_to_submit":False,"note":"S2-only 3/4 continuous-pressure policy."},
    {"file":"submission_v36_1_anchor_hybrid34_reconstructed.csv","role":"CURRENT_PUBLIC_ANCHOR_REFERENCE","eligible_to_submit":False,"note":"Exact deterministic 0.66006 Hybrid34."},
])
slot_plan.to_csv(OUT_DIR/"v36_1_slot_plan.csv",index=False)
summary={
    "version":"V36.1 Hybrid34-Locked Cross-Shadow XE Residual Pressure Stability",
    "public_anchor":PUBLIC_HYBRID34_LB,"leader":PUBLIC_LEADER_LB,"leader_gap":PUBLIC_LEADER_GAP,
    "fresh_worlds":FRESH_WORLDS,"min_pressure_votes":MIN_PRESSURE_VOTES,
    "corridor_coverage":CORRIDOR_COVERAGE,"attack_coverage":ATTACK_COVERAGE,
    "development_winner":DEV_WINNER,"development_gate":bool(DEV_GATE),"sealed_gate":bool(SEALED_GATE),"final_gate":bool(FINAL_GATE),
    "development":V361_DEV.to_dict("records"),"oracle":V361_ORACLE.to_dict("records"),"test_geometry":V361_TEST_GEOMETRY.to_dict("records"),
    "submission_instruction":("PRIMARY IS A NEW PRESSURE CANDIDATE; inspect its ledger before spending a slot." if FINAL_GATE else "NO NEW SUBMISSION: primary is exact Hybrid34 0.66006 fail-close."),
}
with open(OUT_DIR/"v36_1_summary.json","w") as f: json.dump(summary,f,indent=2)
print(json.dumps(summary,indent=2))
print(slot_plan.to_string(index=False))

# ---- cell 34 (markdown) ----
# ### Interpretation contract
#
# - Hybrid34 0.66006 is the only fail-close anchor.
# - S3 is immutable and S1 is excluded.
# - V36.1 audits **continuous pair-margin pressure**, not full-ranking or atomic-flip identity votes.
# - Fresh E/F/G do not tune coverage or thresholds.
# - A new Kaggle submission is justified only if stable 3/4 pressure has leader-scale oracle supply *and* the fixed policy captures it robustly.
# - If the 3/4 pressure oracle itself collapses, the XE branch is considered closed rather than relaxed to 2/4 post hoc.
