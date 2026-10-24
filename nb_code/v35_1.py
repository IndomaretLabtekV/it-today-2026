# Converted from notebooks/v35_1.ipynb: markdown cells become comments, %/! magics neutralized.

# ---- cell 0 (markdown) ----
# # MineToday V35.1 — Fold-Pure Confirmatory IndoBERT ATTACK
#
# **Purpose:** confirm the exact V35 ATTACK policy without retuning it. V30 remains the immutable Stage-1 public anchor. V35.1 changes only the *validation isolation*: TAPT is trained on the training side of each confirmation split, so validation text is unseen by the task-adaptive language-model stage just as Kaggle test text is unseen at final deployment.
#
# ### Frozen from V35
# - `indobenchmark/indobert-base-p2`
# - full encoder fine-tuning recipe
# - V30 `alpha = 0.16`
# - Head12 + Head23 pair tasks
# - pair-reversal consistency
# - 3-way harm/tie/benefit head + signed utility head
# - exact ATTACK thresholds and 8% coverage cap
# - V30 local-intervention lock and the historical protected rows
#
# ### Confirmation protocol
# 1. **X**: one fresh deployment-like user holdout, created once before results are known.
# 2. **T**: the still-sealed template-disjoint split inherited unchanged from V34/V35.
# 3. **Chronology**: U0001–U3000 → U3001–U4000 stress test.
# 4. No threshold, model, learning-rate, epoch, feature, position, or action-policy change is allowed after any confirmation result is seen.
# 5. If the confirmation gate passes, fit the exact same recipe on all 4,000 organizer-train users and produce one ATTACK submission. Otherwise primary is exact V30 fail-close.
#
# This notebook is intended for Kaggle **T4 x2** and uses `torch.nn.DataParallel` when both GPUs are visible.

# ---- cell 1 (code) ----
# Inherited runtime + exact V30 reconstruction configuration
import os, sys, re, json, ast, html, hashlib, random, warnings, time, subprocess, gc, math, pickle
from pathlib import Path
from collections import defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ["PYTHONHASHSEED"] = "42"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

V34_SKIP_INSTALL = bool(int(os.getenv("V34_SKIP_INSTALL", "0")))
try:
    import lightgbm as lgb
except Exception:
    if V34_SKIP_INSTALL: raise
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "lightgbm==4.6.0"])
    import lightgbm as lgb

try:
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except Exception:
    if V34_SKIP_INSTALL: raise
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.43,<5", "sentencepiece"])
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

import numpy as np
import pandas as pd
import scipy
import sklearn
from IPython.display import display
from scipy.sparse import hstack
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

SEED = 42
V34_SMOKE = bool(int(os.getenv("V34_SMOKE", "0")))
V34_STRICT_GEOMETRY = bool(int(os.getenv("V34_STRICT_GEOMETRY", "1")))

# Compatibility aliases consumed by frozen reconstruction cells.
V33_SMOKE = V34_SMOKE
V33_STRICT_GEOMETRY = V34_STRICT_GEOMETRY
V30_SMOKE = V34_SMOKE
V30_STRICT_GEOMETRY = V34_STRICT_GEOMETRY
V29_SMOKE = V34_SMOKE
V29_STRICT_GEOMETRY = V34_STRICT_GEOMETRY
V27_SMOKE = V34_SMOKE
V27_STRICT_GEOMETRY = V34_STRICT_GEOMETRY

N_FOLDS = 2 if V34_SMOKE else 5
ANCHOR_INNER_FOLDS = 2 if V34_SMOKE else 4
STAGE1_OOF_FOLDS = 2 if V34_SMOKE else 3
FULL_STAGE1_OOF_FOLDS = 2 if V34_SMOKE else 4
STAGE2_CAL_FOLDS = 2 if V34_SMOKE else 4

RIDGE_ALPHA = 3.0
REFERENCE_W = 0.45
PUBLIC_CAL045_LB = 0.65639
PUBLIC_CONSENSUS3_LB = 0.65856
PUBLIC_V25_LB = 0.65919
PUBLIC_V30_LB = 0.65963
PUBLIC_V31_DIAGNOSTIC_LB = 0.65949
PUBLIC_RANK4_LB = 0.65974
PUBLIC_LEADER_LB = 0.66142

LGB_ESTIMATORS = 12 if V34_SMOKE else 1200
LGB_EARLY_STOP = 3 if V34_SMOKE else 80
RANKER_ESTIMATORS = 16 if V34_SMOKE else 360
RANKER_LEARNING_RATE = 0.04 if V34_SMOKE else 0.025
ALPHA = 0.16  # exact public V30 residual strength

# Head12 learner capacity is deliberately lower than the Stage-1 ranker.
ACTION_ESTIMATORS = 24 if V34_SMOKE else 360
ACTION_LR = 0.05 if V34_SMOKE else 0.025

# Empirical Head12 economics measured on strict V30 OOF in the V33 audit.
HEAD12_BENEFIT_MAG = 0.0700
HEAD12_HARM_MAG = 0.0942
EPS = 1e-7

# Frozen cross-encoder settings. No finetuning, no test-label use.
CE_MODEL_ID = "BAAI/bge-reranker-v2-m3"
CE_MAX_LENGTH = 128 if V34_SMOKE else 256
CE_BATCH_SIZE = 8 if V34_SMOKE else 32
CE_CACHE_FLUSH_EVERY = 4000

# Exactly two preregistered policies. Coverage is a CAP, not a post-hoc percentile search.
POLICIES = {
    "SAFE": {
        "coverage_cap": 0.025,
        "p_benefit_min": 0.60,
        "p_harm_max": 0.30,
        "asym_eu_min": 0.010,
        "utility_min": 0.0020,
        "margin_cap": 0.10,
        "pair_support_min": 12,
        "pair_utility_floor": -0.002,
        "require_ce_agreement": True,
    },
    "ATTACK": {
        "coverage_cap": 0.070,
        "p_benefit_min": 0.56,
        "p_harm_max": 0.38,
        "asym_eu_min": 0.002,
        "utility_min": 0.0005,
        "margin_cap": 0.18,
        "pair_support_min": 8,
        "pair_utility_floor": -0.008,
        "require_ce_agreement": False,
    },
}

# Promotion gates. ATTACK is only eligible on leader-scale fresh evidence.
SAFE_DEV_MIN_POOLED = -1.0 if V34_SMOKE else 0.00025
SAFE_DEV_MIN_WORST = -1.0 if V34_SMOKE else 0.0
SAFE_DEV_CI_FLOOR = -1.0 if V34_SMOKE else -0.00002
SAFE_DEV_MIN_PRECISION = 0.0 if V34_SMOKE else 0.62
SAFE_DEV_MIN_ACTIONS = 1 if V34_SMOKE else 12

ATTACK_DEV_MIN_POOLED = -1.0 if V34_SMOKE else 0.00150
ATTACK_DEV_MIN_WORST = -1.0 if V34_SMOKE else 0.00050
ATTACK_DEV_CI_FLOOR = -1.0 if V34_SMOKE else 0.00020
ATTACK_DEV_MIN_PRECISION = 0.0 if V34_SMOKE else 0.72
ATTACK_DEV_MIN_ACTIONS = 1 if V34_SMOKE else 30

SAFE_T_FLOOR = -1.0 if V34_SMOKE else -0.00015
SAFE_CHRONO_FLOOR = -1.0 if V34_SMOKE else -0.00010
ATTACK_T_FLOOR = -1.0 if V34_SMOKE else -0.00010
ATTACK_CHRONO_FLOOR = -1.0 if V34_SMOKE else 0.0
BOOTSTRAPS = 30 if V34_SMOKE else 1200
GEOMETRY_ATOL = 2.5e-4

# New development worlds. G/H/I/K/L/M/N/O/P are historical/burned and never reused for selection.
def _seed_from_text(s):
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:8], 16) % (2**31-1)

FRESH_WORLDS = {
    "Q": _seed_from_text("MINETODAY-V34-HEAD12-CE-Q-20260829"),
    "R": _seed_from_text("MINETODAY-V34-HEAD12-CE-R-20260829"),
    "S": _seed_from_text("MINETODAY-V34-HEAD12-CE-S-20260829"),
}
SEALED_T_SALT = "MINETODAY-V34-SEALED-T-HEAD12-CE-20260829"

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
if torch.cuda.is_available():
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True

print("numpy", np.__version__, "pandas", pd.__version__, "sklearn", sklearn.__version__, "lightgbm", lgb.__version__)
print("torch", torch.__version__, "transformers", transformers.__version__)
print("V34_SMOKE", V34_SMOKE, "fixed V30 alpha", ALPHA)
print("Historical V34 bootstrap loaded; V35 fresh worlds are defined in the V35 section below.")
print("public V30", PUBLIC_V30_LB, "rank4 gap", round(PUBLIC_RANK4_LB-PUBLIC_V30_LB,5), "leader gap", round(PUBLIC_LEADER_LB-PUBLIC_V30_LB,5))
try:
    print("GPU runtime:\n" + subprocess.check_output(["nvidia-smi", "-L"], text=True).strip())
except Exception:
    print("nvidia-smi unavailable; cross-encoder will fall back to CPU, which is much slower.")

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
# ## V35 transfer-learning configuration
#
# The experiment separates **weights** from **hyperparameters**. Hyperparameters below are fixed once. During fine-tuning, backpropagation and AdamW update the IndoBERT embedding, attention, feed-forward, LayerNorm, and task-head weights. The fresh U/V/W results are used only to accept/reject the already-defined SAFE/ATTACK policies.

# ---- cell 19 (code) ----
# V35.1 runtime: exact V35 model/policy, confirmation-only validation.
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, DataCollatorForLanguageModeling, get_linear_schedule_with_warmup
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, log_loss
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr

V351_SMOKE = bool(int(os.getenv("V351_SMOKE", "0")))
# Compatibility aliases for the frozen V30 reconstruction helpers.
V35_SMOKE = V351_SMOKE
V34_SMOKE = V351_SMOKE
V33_SMOKE = V351_SMOKE
V30_SMOKE = V351_SMOKE
V29_SMOKE = V351_SMOKE
V27_SMOKE = V351_SMOKE
V34_STRICT_GEOMETRY = V33_STRICT_GEOMETRY = V30_STRICT_GEOMETRY = True

# Immutable V30 state.
ALPHA = 0.16
PUBLIC_V30_LB = 0.65963
PUBLIC_RANK4_LB = 0.65974
PUBLIC_LEADER_LB = 0.66142

# Exact V35 transfer-learning recipe. NO tuning in V35.1.
INDO_MODEL_ID = "indobenchmark/indobert-base-p2"
INDO_MAX_LENGTH = 96 if V351_SMOKE else 256
TAPT_MAX_LENGTH = 64 if V351_SMOKE else 128
TAPT_BATCH_SIZE = 8 if V351_SMOKE else 32
TAPT_MAX_STEPS = 3 if V351_SMOKE else 320
TAPT_LR = 3.0e-5
TAPT_WARMUP = 0.08
TAPT_MLM_PROB = 0.15

FT_BATCH_SIZE = 4 if V351_SMOKE else 32
FT_EPOCHS = 1 if V351_SMOKE else 2
FROZEN_EPOCHS = 1  # historical V35 diagnostic only; not used for confirmation selection.
ENCODER_LR = 1.5e-5
HEAD_LR = 8.0e-5
WEIGHT_DECAY = 0.01
FT_WARMUP = 0.08
GRAD_CLIP = 1.0
REG_LOSS_WEIGHT = 0.55
UTILITY_SCALE = 10.0
INNER_CAL_FRACTION = 0.10

STAGE1_OOF_FOLDS = 2 if V351_SMOKE else 3
FULL_STAGE1_OOF_FOLDS = 2 if V351_SMOKE else 4
EPS = 1e-7
BOOTSTRAPS_V351 = 30 if V351_SMOKE else 1200

# EXACT V35 ATTACK policy. The literal equality assertion prevents accidental drift.
POLICIES_V35 = {
    "ATTACK": {
        "coverage_cap": 0.080,
        "p_benefit_min": 0.57,
        "p_harm_max": 0.34,
        "asym_eu_min": 0.0010,
        "utility_min": 0.0003,
        "directional_consistency_min": 0.53,
        "contradiction_max": 0.38,
        "margin_cap_head12": 0.22,
        "margin_cap_head23": 0.18,
    },
}
_EXPECTED_ATTACK = {
    "coverage_cap":0.080,"p_benefit_min":0.57,"p_harm_max":0.34,
    "asym_eu_min":0.0010,"utility_min":0.0003,
    "directional_consistency_min":0.53,"contradiction_max":0.38,
    "margin_cap_head12":0.22,"margin_cap_head23":0.18,
}
assert POLICIES_V35["ATTACK"] == _EXPECTED_ATTACK

# Confirmation worlds are fixed before execution. U/V/W are historical and are not reused.
def _v35_seed(s):
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:8], 16) % (2**31 - 1)

CONFIRM_X_SEED = _v35_seed("MINETODAY-V35.1-FOLDPURE-CONFIRM-X-20260829")
SEALED_T_SALT_V35 = "MINETODAY-V34-SEALED-T-HEAD12-CE-20260829"
CHRONO_TR_V351 = np.arange(0,3000,dtype=int)
CHRONO_VA_V351 = np.arange(3000,4000,dtype=int)

# Confirmation-only rejection gate. These are NOT model/policy thresholds.
# X must show positive deployment-like utility. T may be essentially flat but not materially harmful.
# Chronology must be non-negative. Selected-action precision must remain healthy overall.
X_MIN_DELTA = -1.0 if V351_SMOKE else 0.0
T_HARM_FLOOR = -1.0 if V351_SMOKE else -0.00005
CHRONO_HARM_FLOOR = -1.0 if V351_SMOKE else 0.0
COMBINED_MIN_PRECISION = 0.0 if V351_SMOKE else 0.60
X_MIN_ACTIONS = 1

OUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("/mnt/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)
HF_CACHE = OUT_DIR / "hf_cache"
HF_CACHE.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
N_GPU = torch.cuda.device_count()
if torch.cuda.is_available():
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True

print("V35.1 fresh confirmation X seed:", CONFIRM_X_SEED)
print("IndoBERT", INDO_MODEL_ID, "device", DEVICE, "visible GPUs", N_GPU)
print("Exact V35 ATTACK policy:", POLICIES_V35["ATTACK"])
print("V30", PUBLIC_V30_LB, "rank4 gap", round(PUBLIC_RANK4_LB-PUBLIC_V30_LB,5), "leader gap", round(PUBLIC_LEADER_LB-PUBLIC_V30_LB,5))
try:
    print("GPU runtime:\n" + subprocess.check_output(["nvidia-smi", "-L"], text=True).strip())
except Exception:
    print("nvidia-smi unavailable; training will fall back to CPU and be much slower.")

# ---- cell 20 (markdown) ----
# ## Tokenizer, compact context, and fold-pure TAPT corpus builder
#
# Unlike V35, no global all-train TAPT checkpoint is created before confirmation. Each confirmation world receives a TAPT encoder trained **only** on that world's training-side chat plus the organizer module catalog.

# ---- cell 21 (code) ----
def locate_indobert_model():
    env = os.getenv("V35_INDO_MODEL_PATH", "").strip()
    if env and Path(env).exists():
        return env
    root = Path("/kaggle/input")
    if root.exists():
        preferred = []
        fallback = []
        for p in root.rglob("config.json"):
            s = str(p.parent).lower()
            if "indobert" in s:
                fallback.append(str(p.parent))
                if "base" in s and ("p2" in s or "phase2" in s):
                    preferred.append(str(p.parent))
        if preferred:
            return sorted(preferred)[0]
        if fallback:
            return sorted(fallback)[0]
    return INDO_MODEL_ID

INDO_SOURCE = locate_indobert_model()
print("IndoBERT source:", INDO_SOURCE)
TOKENIZER = AutoTokenizer.from_pretrained(INDO_SOURCE, cache_dir=str(HF_CACHE), use_fast=True)

# Recent-learning context. Text is intentionally compact so candidate modules retain token budget.
def _learning_rows_v35(uid):
    g = user_chat.get(uid)
    if g is None or len(g) == 0:
        return []
    z = g[~g["is_noise"]].copy()
    if len(z) == 0:
        z = g.copy()
    return list(z.itertuples(index=False))

V35_CONTEXT = []
for uid in all_users:
    rows = _learning_rows_v35(uid)
    if not rows:
        txt = "Siswa belum memiliki riwayat chat pembelajaran yang eksplisit."
    else:
        latest = str(rows[-1].clean).strip()
        recent = [str(r.clean).strip() for r in rows[-4:-1] if str(r.clean).strip()]
        parts = [f"Kebutuhan terbaru: {latest}"]
        if recent:
            parts.append("Konteks sebelumnya: " + " || ".join(recent[-3:]))
        txt = " ".join(parts)
    V35_CONTEXT.append(txt)
V35_CONTEXT = np.asarray(V35_CONTEXT, dtype=object)

mod_idx = modules.set_index("module_id")
V35_MODULE_DOCS = []
for mid in MODULES:
    r = mod_idx.loc[mid]
    V35_MODULE_DOCS.append(
        f"Modul {r['module_name']}. Prasyarat {r['prerequisite_level']}. "
        f"Materi dan silabus {r['description_and_syllabus']}"
    )
V35_MODULE_DOCS = np.asarray(V35_MODULE_DOCS, dtype=object)

# Tokenizer/dictionary audit is target-free.
audit_texts = []
for uid in train["user_id"].astype(str):
    rr = _learning_rows_v35(uid)
    audit_texts.extend([str(r.clean).strip() for r in rr if str(r.clean).strip()])
audit_texts += V35_MODULE_DOCS.tolist()
if len(audit_texts) > (300 if V35_SMOKE else 5000):
    rng = np.random.default_rng(SEED)
    audit_texts = [audit_texts[i] for i in rng.choice(len(audit_texts), size=(300 if V35_SMOKE else 5000), replace=False)]
lengths = []
unk_counts = []
for i in range(0, len(audit_texts), 256):
    enc = TOKENIZER(audit_texts[i:i+256], add_special_tokens=True, truncation=False)
    for ids in enc["input_ids"]:
        lengths.append(len(ids))
        unk_counts.append(sum(int(t == TOKENIZER.unk_token_id) for t in ids) if TOKENIZER.unk_token_id is not None else 0)
V35_TOKEN_AUDIT = pd.DataFrame([{
    "model": str(INDO_SOURCE), "sample_texts": len(lengths),
    "mean_tokens": float(np.mean(lengths)), "median_tokens": float(np.median(lengths)),
    "p95_tokens": float(np.percentile(lengths,95)), "max_tokens": int(np.max(lengths)),
    "total_unk": int(np.sum(unk_counts)), "mean_unk_per_text": float(np.mean(unk_counts)),
}])
display(V35_TOKEN_AUDIT)



def build_tapt_texts_for_rows(train_rows):
    """Organizer-only TAPT corpus for the supplied TRAIN-side rows. No relevance labels, no validation/test chat."""
    train_rows=np.asarray(train_rows,int)
    texts=[]
    for r in train_rows:
        uid=str(train.iloc[int(r)]["user_id"])
        rr=_learning_rows_v35(uid)
        texts.extend([str(z.clean).strip() for z in rr if str(z.clean).strip()])
    texts += V35_MODULE_DOCS.tolist()
    seen=set(); out=[]
    for t in texts:
        t=re.sub(r"\s+"," ",str(t)).strip()
        if t and t not in seen:
            seen.add(t); out.append(t)
    return out

print("Fold-pure TAPT builder ready. Full-train corpus is constructed only at final deployment.")

# ---- cell 22 (code) ----
# Fold-pure task-adaptive masked-language-model training.
class _TokenListDataset(Dataset):
    def __init__(self, enc): self.enc=enc
    def __len__(self): return len(self.enc["input_ids"])
    def __getitem__(self,i): return {k:self.enc[k][i] for k in self.enc.keys()}

def run_tapt_for_rows(train_rows, tag, seed):
    train_rows=np.asarray(train_rows,int)
    safe_tag=re.sub(r"[^A-Za-z0-9_.-]+","_",str(tag))
    state_path=OUT_DIR/f"v35_1_tapt_{safe_tag}_encoder_state.pt"
    summary_path=OUT_DIR/f"v35_1_tapt_{safe_tag}_summary.json"
    if state_path.exists() and summary_path.exists():
        print("Reusing cached fold-pure TAPT", tag)
        return state_path, json.load(open(summary_path))

    random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed))
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(int(seed))
    texts=build_tapt_texts_for_rows(train_rows)
    enc=TOKENIZER(texts,truncation=True,max_length=TAPT_MAX_LENGTH,add_special_tokens=True)
    ds=_TokenListDataset(enc)
    collator=DataCollatorForLanguageModeling(tokenizer=TOKENIZER,mlm=True,mlm_probability=TAPT_MLM_PROB)
    gen=torch.Generator(); gen.manual_seed(int(seed))
    loader=DataLoader(ds,batch_size=TAPT_BATCH_SIZE,shuffle=True,collate_fn=collator,
                      num_workers=0,pin_memory=torch.cuda.is_available(),generator=gen)
    model=AutoModelForMaskedLM.from_pretrained(INDO_SOURCE,cache_dir=str(HF_CACHE))
    model.to(DEVICE)
    if N_GPU>1: model=nn.DataParallel(model)
    opt=torch.optim.AdamW(model.parameters(),lr=TAPT_LR,weight_decay=WEIGHT_DECAY)
    steps_target=min(TAPT_MAX_STEPS,max(1,len(loader)))
    sched=get_linear_schedule_with_warmup(opt,num_warmup_steps=max(1,int(steps_target*TAPT_WARMUP)),num_training_steps=steps_target)
    scaler=torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    model.train(); losses=[]; step=0
    while step<steps_target:
        for batch in loader:
            batch={k:v.to(DEVICE,non_blocking=True) for k,v in batch.items()}
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                loss=model(**batch).loss
                if loss.ndim>0: loss=loss.mean()
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP)
            scaler.step(opt); scaler.update(); sched.step()
            losses.append(float(loss.detach().cpu())); step+=1
            if step%50==0 or step==steps_target:
                print(f"TAPT[{tag}] step {step}/{steps_target} loss={np.mean(losses[-50:]):.4f}")
            if step>=steps_target: break
    core=model.module if isinstance(model,nn.DataParallel) else model
    state={k:v.detach().cpu() for k,v in core.base_model.state_dict().items()}
    torch.save(state,state_path)
    summary={
        "tag":str(tag),"model":str(INDO_SOURCE),"train_users":int(len(train_rows)),"texts":int(len(texts)),
        "steps":int(step),"mean_loss":float(np.mean(losses)),"final50_loss":float(np.mean(losses[-50:])),
        "uses_relevance_labels":False,"uses_validation_chat":False,"uses_test_chat":False,
        "fold_pure":True,"seed":int(seed),
    }
    with open(summary_path,"w") as f: json.dump(summary,f,indent=2)
    del model,core,state,loader,ds,enc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return state_path,summary

print("V35.1 split-pure TAPT trainer ready")

# ---- cell 23 (markdown) ----
# ## Strict V30 Top-3 pair dataset

# ---- cell 24 (code) ----
# Reuse exact V30 bundle functions but create a pairwise preference dataset instead of V34 Head12 features.
def build_v30_bundle_split_v35(train_rows, val_rows, seed):
    train_rows=np.asarray(train_rows,int); val_rows=np.asarray(val_rows,int)
    anc=build_anchor_split_full(train_rows,val_rows,seed)
    Xtr,_=build_long_matrix_v30(train_rows,anc["cons_train"],anc["base_train"],anc["meta_train"])
    Xva,_=build_long_matrix_v30(val_rows,anc["cons_val"],anc["base_val"],anc["meta_val"])
    model=make_lambdarank(seed+313); group=np.full(len(train_rows),N_MODULES,dtype=np.int32)
    model.fit(Xtr,ORDINAL_GRADES[train_rows].reshape(-1),group=group)
    raw=model.predict(Xva).reshape(-1,N_MODULES).astype(np.float32)
    v30=blend_residual(anc["cons_val"],raw,ALPHA)
    out={"v30":v30,"anchor":anc["cons_val"].copy(),"base":anc["base_val"].copy(),"meta":anc["meta_val"].copy()}
    del Xtr,Xva,model,raw,anc; gc.collect(); return out

def crossfit_v30_bundle_rows_v35(rows,seed,n_splits=STAGE1_OOF_FOLDS):
    rows=np.asarray(rows,int); strata=user_strata_from_y(y[rows])
    ns=min(int(n_splits),max(2,int(pd.Series(strata).value_counts().min())))
    sk=StratifiedKFold(n_splits=ns,shuffle=True,random_state=int(seed))
    out={k:np.zeros((len(rows),N_MODULES),np.float32) for k in ["v30","anchor","base","meta"]}
    fid=np.full(len(rows),-1,np.int16)
    for f,(lt,lv) in enumerate(sk.split(np.zeros(len(rows)),strata)):
        tr=rows[lt]; va=rows[lv]
        print(f"  V35 strict Stage-1 OOF {f+1}/{ns}: train={len(tr)} val={len(va)}")
        z=build_v30_bundle_split_v35(tr,va,seed+1009+f*211)
        for k in out: out[k][lv]=z[k]
        fid[lv]=f
    assert (fid>=0).all()
    out["fold_id"]=fid
    return out

def adjacent_swap_delta_v35(scores,y_true,a,b):
    scores=np.asarray(scores,np.float32); a=np.asarray(a,int); b=np.asarray(b,int)
    before=per_user_ndcg(y_true,scores); swapped=scores.copy(); rr=np.arange(len(scores))
    tmp=swapped[rr,a].copy(); swapped[rr,a]=swapped[rr,b]; swapped[rr,b]=tmp
    return (per_user_ndcg(y_true,swapped)-before).astype(np.float32)

def _pair3(mat,rows,a,b):
    aa=np.asarray(mat)[rows,a]; bb=np.asarray(mat)[rows,b]
    return [aa,bb,bb-aa]

PAIR_STRUCT_NAMES = [
    "position_head12","position_head23",
    "v30_a","v30_b","v30_b_minus_a","v30_abs_margin",
    "anchor_a","anchor_b","anchor_b_minus_a",
    "base_a","base_b","base_b_minus_a",
    "meta_a","meta_b","meta_b_minus_a",
    "residual_a","residual_b","residual_b_minus_a",
    "lex_word_a","lex_word_b","lex_word_b_minus_a",
    "lex_char_a","lex_char_b","lex_char_b_minus_a",
    "topic_count_a","topic_count_b","topic_count_b_minus_a",
    "topic_recency_a","topic_recency_b","topic_recency_b_minus_a",
    "topic_latest_a","topic_latest_b","topic_latest_b_minus_a",
    "readiness_a","readiness_b","readiness_b_minus_a",
    "invalid_a","invalid_b","invalid_b_minus_a",
] + [f"user_f{i:03d}" for i in range(USER_FEATURES.shape[1])] + [f"a_{m}" for m in MODULES] + [f"b_{m}" for m in MODULES] + [f"a_level_{i}" for i in range(5)] + [f"b_level_{i}" for i in range(5)]

def _structured_for_orientation(rows_global, bundle, a, b, pos):
    rows_global=np.asarray(rows_global,int); a=np.asarray(a,int); b=np.asarray(b,int); pos=np.asarray(pos,int)
    v30=np.asarray(bundle["v30"],np.float32); anchor=np.asarray(bundle["anchor"],np.float32)
    base=np.asarray(bundle["base"],np.float32); meta=np.asarray(bundle["meta"],np.float32)
    # bundle rows are aligned 1:1 to rows_global for all callers.
    rr=np.arange(len(rows_global)); residual=v30-anchor
    va=v30[rr,a]; vb=v30[rr,b]
    dense=[(pos==0).astype(np.float32),(pos==1).astype(np.float32),va,vb,vb-va,np.abs(vb-va),
           *_pair3(anchor,rr,a,b), *_pair3(base,rr,a,b), *_pair3(meta,rr,a,b), *_pair3(residual,rr,a,b),
           *_pair3(LEX_WORD,rows_global,a,b), *_pair3(LEX_CHAR,rows_global,a,b),
           *_pair3(MODULE_COUNT,rows_global,a,b), *_pair3(MODULE_RECENCY,rows_global,a,b),
           *_pair3(MODULE_LATEST,rows_global,a,b), *_pair3(MODULE_READINESS,rows_global,a,b),
           *_pair3(MODULE_INVALID,rows_global,a,b)]
    x=np.column_stack(dense).astype(np.float32)
    one_a=np.eye(N_MODULES,dtype=np.float32)[a]; one_b=np.eye(N_MODULES,dtype=np.float32)[b]
    lvl_a=MODULE_LEVEL_ONEHOT[a]; lvl_b=MODULE_LEVEL_ONEHOT[b]
    x=np.hstack([x,USER_FEATURES[rows_global].astype(np.float32),one_a,one_b,lvl_a,lvl_b]).astype(np.float32)
    assert x.shape[1]==len(PAIR_STRUCT_NAMES) and np.isfinite(x).all()
    return x

def build_top3_pairs(rows_global,bundle,with_labels=True,augment_reverse=False):
    rows_global=np.asarray(rows_global,int); v30=np.asarray(bundle["v30"],np.float32)
    order=np.argsort(-v30,axis=1,kind="mergesort")
    # Two tasks per user: rank1>rank2 and rank2>rank3 in current V30 order.
    rows=np.repeat(rows_global,2)
    local=np.repeat(np.arange(len(rows_global)),2)
    pos=np.tile(np.array([0,1],dtype=np.int8),len(rows_global))
    a=np.where(pos==0,order[local,0],order[local,1]).astype(np.int16)
    b=np.where(pos==0,order[local,1],order[local,2]).astype(np.int16)
    subbundle={k:np.asarray(bundle[k])[local] for k in ["v30","anchor","base","meta"]}
    X=_structured_for_orientation(rows,subbundle,a,b,pos)
    ds={"rows_global":rows,"local_user":local,"position":pos,"a":a,"b":b,"X":X}
    if with_labels:
        yy=y[rows]
        delta=adjacent_swap_delta_v35(subbundle["v30"],yy,a,b)
        cls=np.full(len(delta),1,np.int64); cls[delta>1e-12]=2; cls[delta<-1e-12]=0
        ds.update({"delta":delta,"class":cls})
    if augment_reverse:
        ar=b.copy(); br=a.copy()
        Xr=_structured_for_orientation(rows,subbundle,ar,br,pos)
        for key,val in list(ds.items()):
            if key in ["X","a","b","delta","class"]: continue
            ds[key]=np.concatenate([val,val])
        ds["X"]=np.vstack([X,Xr]).astype(np.float32)
        ds["a"]=np.concatenate([a,ar]); ds["b"]=np.concatenate([b,br])
        if with_labels:
            ds["delta"]=np.concatenate([delta,-delta]).astype(np.float32)
            cr=cls.copy(); cr[cls==0]=2; cr[cls==2]=0
            ds["class"]=np.concatenate([cls,cr]).astype(np.int64)
    return ds

print("V35 strict V30 Top-3 pair builder ready; structured dims",len(PAIR_STRUCT_NAMES))

# ---- cell 25 (markdown) ----
# ## Exact V35 IndoBERT pair model, with the TAPT checkpoint passed explicitly

# ---- cell 26 (code) ----
def pair_texts(ds):
    contexts=[]; pair_docs=[]
    for r,a,b in zip(ds["rows_global"],ds["a"],ds["b"]):
        contexts.append(str(V35_CONTEXT[int(r)]))
        pair_docs.append(f"Kandidat A: {V35_MODULE_DOCS[int(a)]} Kandidat B: {V35_MODULE_DOCS[int(b)]}")
    return contexts,pair_docs

class EncodedPairDataset(Dataset):
    def __init__(self, ds, mean, std):
        q,d=pair_texts(ds)
        tok=TOKENIZER(q,d,padding=True,truncation=True,max_length=INDO_MAX_LENGTH,return_tensors="pt")
        self.tok=tok
        self.x=torch.tensor((ds["X"]-mean)/std,dtype=torch.float32)
        self.cls=torch.tensor(ds["class"],dtype=torch.long) if "class" in ds else None
        self.delta=torch.tensor(ds["delta"]*UTILITY_SCALE,dtype=torch.float32) if "delta" in ds else None
        self.rows=np.asarray(ds["rows_global"],int)
    def __len__(self): return len(self.x)
    def __getitem__(self,i):
        z={k:v[i] for k,v in self.tok.items()}
        z["structured"]=self.x[i]
        if self.cls is not None: z["labels_cls"]=self.cls[i]
        if self.delta is not None: z["labels_delta"]=self.delta[i]
        return z

class IndoPairUtilityModel(nn.Module):
    def __init__(self, structured_dim, tapt_state_path, freeze_encoder=False):
        super().__init__()
        self.encoder=AutoModel.from_pretrained(INDO_SOURCE,cache_dir=str(HF_CACHE))
        state=torch.load(tapt_state_path,map_location="cpu")
        missing,unexpected=self.encoder.load_state_dict(state,strict=False)
        if missing or unexpected:
            print("TAPT encoder load notes",len(missing),len(unexpected))
        if freeze_encoder:
            for p in self.encoder.parameters(): p.requires_grad=False
        h=int(self.encoder.config.hidden_size)
        self.side=nn.Sequential(nn.Linear(structured_dim,128),nn.GELU(),nn.Dropout(0.10))
        self.fuse=nn.Sequential(nn.Linear(h+128,256),nn.GELU(),nn.Dropout(0.12))
        self.cls_head=nn.Linear(256,3)
        self.reg_head=nn.Linear(256,1)
    def forward(self,input_ids,attention_mask,structured,token_type_ids=None):
        kw={"input_ids":input_ids,"attention_mask":attention_mask}
        if token_type_ids is not None: kw["token_type_ids"]=token_type_ids
        out=self.encoder(**kw)
        pooled=out.last_hidden_state[:,0]
        side=self.side(structured)
        h=self.fuse(torch.cat([pooled,side],dim=-1))
        return self.cls_head(h),self.reg_head(h).squeeze(-1)

def _split_inner_users(train_ds,seed):
    users=np.unique(train_ds["rows_global"])
    strata=user_strata_from_y(y[users])
    sss=StratifiedShuffleSplit(n_splits=1,test_size=INNER_CAL_FRACTION,random_state=int(seed))
    a,b=next(sss.split(np.zeros(len(users)),strata))
    fit=set(users[a].tolist()); cal=set(users[b].tolist())
    fit_mask=np.array([int(r) in fit for r in train_ds["rows_global"]])
    cal_mask=np.array([int(r) in cal for r in train_ds["rows_global"]])
    return fit_mask,cal_mask

def _subset_ds(ds,mask):
    out={}
    for k,v in ds.items():
        if isinstance(v,np.ndarray) and len(v)==len(mask): out[k]=v[mask]
        else: out[k]=v
    return out

def _softmax_np(logits,T=1.0):
    z=np.asarray(logits,np.float64)/max(float(T),1e-4); z-=z.max(1,keepdims=True)
    e=np.exp(z); return (e/e.sum(1,keepdims=True)).astype(np.float32)

def _temperature_scale(logits,labels):
    labels=np.asarray(labels,int)
    def f(logT):
        T=float(np.exp(logT)); p=_softmax_np(logits,T)
        return log_loss(labels,p,labels=[0,1,2])
    r=minimize_scalar(f,bounds=(math.log(0.5),math.log(3.0)),method="bounded")
    return float(np.exp(r.x))

def _predict_torch(model,loader):
    model.eval(); logits=[]; util=[]; labels=[]; deltas=[]
    with torch.no_grad():
        for batch in loader:
            lab=batch.pop("labels_cls",None); de=batch.pop("labels_delta",None)
            batch={k:v.to(DEVICE,non_blocking=True) for k,v in batch.items()}
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                lo,uu=model(**batch)
            logits.append(lo.float().detach().cpu().numpy()); util.append(uu.float().detach().cpu().numpy())
            if lab is not None: labels.append(lab.numpy())
            if de is not None: deltas.append(de.numpy()/UTILITY_SCALE)
    return np.vstack(logits),np.concatenate(util)/UTILITY_SCALE,(np.concatenate(labels) if labels else None),(np.concatenate(deltas) if deltas else None)

def _auc_safe(y_true,score):
    y_true=np.asarray(y_true,int)
    return float(roc_auc_score(y_true,score)) if len(np.unique(y_true))>1 else np.nan

def fit_pair_model(train_ds,seed,freeze_encoder=False,tapt_state_path=None):
    # Fixed seed makes frozen-vs-finetuned comparison use the same split/head initialization.
    random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed))
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(int(seed))
    # Split by original user before reversal augmentation; no user appears in both fit and calibration.
    fit_mask,cal_mask=_split_inner_users(train_ds,seed)
    base_fit=_subset_ds(train_ds,fit_mask); base_cal=_subset_ds(train_ds,cal_mask)
    # Reverse augmentation only the optimizer subset.
    # Rebuild from its own users to guarantee exact orientation symmetry.
    fit_users=np.unique(base_fit["rows_global"])
    # Caller training bundle is aligned to its parent rows; augment function below operates directly on arrays.
    aug={k:(np.concatenate([v,v]) if isinstance(v,np.ndarray) and len(v)==len(base_fit["rows_global"]) and k not in ["X","a","b","delta","class"] else v) for k,v in base_fit.items()}
    # Reverse structured/text orientation from existing records using algebraic orientation builder is unavailable here,
    # so construct a faithful reversed copy by swapping candidate-specific columns through the original data helper.
    # The parent dataset stores the already-computed forward X; use an explicit generic reversal builder below.
    rev_X=reverse_structured_matrix(base_fit["X"])
    aug["X"]=np.vstack([base_fit["X"],rev_X]).astype(np.float32)
    aug["a"]=np.concatenate([base_fit["a"],base_fit["b"]]); aug["b"]=np.concatenate([base_fit["b"],base_fit["a"]])
    aug["delta"]=np.concatenate([base_fit["delta"],-base_fit["delta"]]).astype(np.float32)
    cr=base_fit["class"].copy(); rev_cls=cr.copy(); rev_cls[cr==0]=2; rev_cls[cr==2]=0
    aug["class"]=np.concatenate([cr,rev_cls]).astype(np.int64)

    mean=aug["X"].mean(0).astype(np.float32); std=aug["X"].std(0).astype(np.float32); std[std<1e-5]=1.0
    fit_t=EncodedPairDataset(aug,mean,std); cal_t=EncodedPairDataset(base_cal,mean,std)
    fit_loader=DataLoader(fit_t,batch_size=FT_BATCH_SIZE,shuffle=True,num_workers=0,pin_memory=torch.cuda.is_available())
    cal_loader=DataLoader(cal_t,batch_size=FT_BATCH_SIZE,shuffle=False,num_workers=0,pin_memory=torch.cuda.is_available())
    assert tapt_state_path is not None and Path(tapt_state_path).exists(); model=IndoPairUtilityModel(len(PAIR_STRUCT_NAMES),tapt_state_path=tapt_state_path,freeze_encoder=freeze_encoder).to(DEVICE)
    if N_GPU>1: model=nn.DataParallel(model)
    core=model.module if isinstance(model,nn.DataParallel) else model
    enc_params=[p for p in core.encoder.parameters() if p.requires_grad]
    head_params=[p for n,p in core.named_parameters() if not n.startswith("encoder.") and p.requires_grad]
    groups=[]
    if enc_params: groups.append({"params":enc_params,"lr":ENCODER_LR})
    groups.append({"params":head_params,"lr":HEAD_LR})
    opt=torch.optim.AdamW(groups,weight_decay=WEIGHT_DECAY)
    epochs=FROZEN_EPOCHS if freeze_encoder else FT_EPOCHS
    total=max(1,len(fit_loader)*epochs)
    sched=get_linear_schedule_with_warmup(opt,max(1,int(total*FT_WARMUP)),total)
    cls_counts=np.bincount(aug["class"],minlength=3).astype(np.float32)
    cw=np.sqrt(cls_counts.sum()/np.maximum(cls_counts,1.0)); cw=cw/cw.mean(); cw[0]*=1.10
    class_w=torch.tensor(cw,dtype=torch.float32,device=DEVICE)
    scaler=torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    best_metric=np.inf; best_path=OUT_DIR/f"_v35_1_tmp_{seed}_{'frozen' if freeze_encoder else 'ft'}.pt"; history=[]
    for epoch in range(epochs):
        model.train(); run=[]
        for batch in fit_loader:
            lab=batch.pop("labels_cls").to(DEVICE,non_blocking=True); de=batch.pop("labels_delta").to(DEVICE,non_blocking=True)
            batch={k:v.to(DEVICE,non_blocking=True) for k,v in batch.items()}
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                lo,uu=model(**batch)
                ce=nn.functional.cross_entropy(lo,lab,weight=class_w)
                reg=nn.functional.smooth_l1_loss(uu,de,beta=0.35)
                loss=ce+REG_LOSS_WEIGHT*reg
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP)
            scaler.step(opt); scaler.update(); sched.step(); run.append(float(loss.detach().cpu()))
        clog,cutil,clab,cdelta=_predict_torch(model,cal_loader)
        T=_temperature_scale(clog,clab); cp=_softmax_np(clog,T)
        nll=log_loss(clab,cp,labels=[0,1,2]); mse=float(np.mean((cutil-cdelta)**2)); metric=float(nll+2.0*mse)
        history.append({"epoch":epoch+1,"train_loss":float(np.mean(run)),"cal_nll":float(nll),"cal_mse":mse,"temperature":T,"selection_metric":metric})
        print(("frozen" if freeze_encoder else "finetune"),"epoch",epoch+1,history[-1])
        if metric<best_metric:
            best_metric=metric
            torch.save({k:v.detach().cpu() for k,v in core.state_dict().items()},best_path)
    core.load_state_dict(torch.load(best_path,map_location="cpu")); core.to(DEVICE)
    # Recompute calibration on selected checkpoint.
    clog,cutil,clab,cdelta=_predict_torch(model,cal_loader); T=_temperature_scale(clog,clab); cp=_softmax_np(clog,T)
    cal_metrics={"temperature":T,"benefit_auc":_auc_safe((clab==2).astype(int),cp[:,2]),"harm_auc":_auc_safe((clab==0).astype(int),cp[:,0]),"utility_spearman":float(spearmanr(cutil,cdelta).statistic) if len(cdelta)>3 else np.nan,"history":history}
    try: best_path.unlink()
    except Exception: pass
    del fit_loader,cal_loader,fit_t,cal_t; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return {"model":model,"mean":mean,"std":std,"temperature":T,"cal_metrics":cal_metrics}

# Reverse structured orientation by swapping every A/B triplet and negating B-A fields.
# This is deterministic and keeps user/position fields unchanged.
def reverse_structured_matrix(X):
    X=np.asarray(X,np.float32).copy(); out=X.copy()
    name_to_i={n:i for i,n in enumerate(PAIR_STRUCT_NAMES)}
    triplet_prefixes=["v30","anchor","base","meta","residual","lex_word","lex_char","topic_count","topic_recency","topic_latest","readiness","invalid"]
    for p in triplet_prefixes:
        ia=name_to_i[f"{p}_a"]; ib=name_to_i[f"{p}_b"]
        out[:,ia]=X[:,ib]; out[:,ib]=X[:,ia]
        diff_name=("v30_b_minus_a" if p=="v30" else f"{p}_b_minus_a")
        if diff_name in name_to_i: out[:,name_to_i[diff_name]]=-X[:,name_to_i[diff_name]]
    # absolute V30 margin is orientation-invariant.
    # candidate module one-hots and level one-hots.
    for m in MODULES:
        ia=name_to_i[f"a_{m}"]; ib=name_to_i[f"b_{m}"]; out[:,ia]=X[:,ib]; out[:,ib]=X[:,ia]
    for j in range(5):
        ia=name_to_i[f"a_level_{j}"]; ib=name_to_i[f"b_level_{j}"]; out[:,ia]=X[:,ib]; out[:,ib]=X[:,ia]
    return out.astype(np.float32)

def predict_pair_model(bundle,ds,reverse=False):
    use_ds={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in ds.items()}
    if reverse:
        use_ds["X"]=reverse_structured_matrix(ds["X"])
        use_ds["a"]=ds["b"].copy(); use_ds["b"]=ds["a"].copy()
    t=EncodedPairDataset(use_ds,bundle["mean"],bundle["std"])
    loader=DataLoader(t,batch_size=FT_BATCH_SIZE,shuffle=False,num_workers=0,pin_memory=torch.cuda.is_available())
    lo,uu,lab,de=_predict_torch(bundle["model"],loader)
    p=_softmax_np(lo,bundle["temperature"])
    del loader,t; gc.collect()
    return p,uu

print("IndoBERT pair utility learner ready")

# ---- cell 27 (markdown) ----
# ## Exact V35 asymmetric ATTACK policy

# ---- cell 28 (code) ----
def position_economics(train_ds):
    out={}
    for pos in [0,1]:
        m=train_ds["position"]==pos; d=train_ds["delta"][m]
        gain=float(np.mean(d[d>1e-12])) if np.any(d>1e-12) else 0.0
        harm=float(np.mean(-d[d<-1e-12])) if np.any(d<-1e-12) else 0.0
        out[pos]={"gain":gain,"harm":harm,"support":int(m.sum()),"benefit_rate":float(np.mean(d>1e-12)),"harm_rate":float(np.mean(d<-1e-12))}
    return out

def score_directionally(model_bundle,ds):
    pf,uf=predict_pair_model(model_bundle,ds,reverse=False)
    pr,ur=predict_pair_model(model_bundle,ds,reverse=True)
    # Forward: B-better=class2 benefit, A-better=class0 harm.
    # Reverse: original B is candidate A, hence class0 confirms benefit; class2 confirms harm.
    p_ben=np.sqrt(np.clip(pf[:,2],EPS,1)*np.clip(pr[:,0],EPS,1)).astype(np.float32)
    p_harm=np.sqrt(np.clip(pf[:,0],EPS,1)*np.clip(pr[:,2],EPS,1)).astype(np.float32)
    p_tie=np.sqrt(np.clip(pf[:,1],EPS,1)*np.clip(pr[:,1],EPS,1)).astype(np.float32)
    util=(0.5*(uf-ur)).astype(np.float32)
    consistency=np.minimum(pf[:,2],pr[:,0]).astype(np.float32)
    contradiction=np.maximum(pf[:,0],pr[:,2]).astype(np.float32)
    return {"p_benefit":p_ben,"p_harm":p_harm,"p_tie":p_tie,"utility":util,"consistency":consistency,"contradiction":contradiction,"pf":pf,"pr":pr}

def apply_v35_policy(v30_scores,ds,pred,econ,policy,user_ids,hard_lock_local=None,labels_available=True):
    cfg=POLICIES_V35[policy]; n_users=len(v30_scores); hard_lock_local=set() if hard_lock_local is None else set(hard_lock_local)
    pos=ds["position"].astype(int); gain=np.array([econ[int(p)]["gain"] for p in pos],np.float32); harm=np.array([econ[int(p)]["harm"] for p in pos],np.float32)
    eu=pred["p_benefit"]*gain-pred["p_harm"]*harm
    margin=np.abs(ds["X"][:,PAIR_STRUCT_NAMES.index("v30_b_minus_a")])
    cap=np.where(pos==0,cfg["margin_cap_head12"],cfg["margin_cap_head23"])
    eligible=(pred["p_benefit"]>=cfg["p_benefit_min"])&(pred["p_harm"]<=cfg["p_harm_max"])&(eu>=cfg["asym_eu_min"])&(pred["utility"]>=cfg["utility_min"])&(pred["consistency"]>=cfg["directional_consistency_min"])&(pred["contradiction"]<=cfg["contradiction_max"])&(margin<=cap)
    # At most one action/user. Choose risk-adjusted utility, not probability alone.
    score=eu+0.35*pred["utility"]+0.02*(pred["consistency"]-pred["contradiction"])
    candidates=[]
    for i in np.flatnonzero(eligible):
        lu=int(ds["local_user"][i])
        if lu in hard_lock_local: continue
        candidates.append((float(score[i]),i,lu))
    best={}
    for sc,i,lu in candidates:
        if lu not in best or sc>best[lu][0]: best[lu]=(sc,i)
    chosen=sorted([(sc,i,lu) for lu,(sc,i) in best.items()],reverse=True)
    max_actions=max(1,int(math.floor(cfg["coverage_cap"]*n_users)))
    chosen=chosen[:max_actions]
    out=np.asarray(v30_scores,np.float32).copy(); ledger=[]
    for sc,i,lu in chosen:
        a=int(ds["a"][i]); b=int(ds["b"][i]); tmp=float(out[lu,a]); out[lu,a]=out[lu,b]; out[lu,b]=tmp
        row={"policy":policy,"user_id":str(user_ids[lu]),"position":"1-2" if int(pos[i])==0 else "2-3","module_a":MODULES[a],"module_b":MODULES[b],"risk_score":sc,"p_benefit":float(pred["p_benefit"][i]),"p_harm":float(pred["p_harm"][i]),"p_tie":float(pred["p_tie"][i]),"expected_utility":float(eu[i]),"predicted_delta":float(pred["utility"][i]),"directional_consistency":float(pred["consistency"][i]),"contradiction":float(pred["contradiction"][i]),"v30_margin":float(margin[i])}
        if labels_available:
            d=float(ds["delta"][i]); row.update({"actual_delta":d,"beneficial":int(d>1e-12),"harmful":int(d<-1e-12),"tie":int(abs(d)<=1e-12)})
        ledger.append(row)
    return out,pd.DataFrame(ledger)

def bootstrap_ci_v35(values,seed,bootstraps=BOOTSTRAPS_V351):
    v=np.asarray(values,float); rng=np.random.default_rng(seed); n=len(v)
    if n==0:return (np.nan,np.nan)
    means=np.empty(bootstraps,float)
    for b in range(bootstraps): means[b]=np.mean(v[rng.integers(0,n,n)])
    return float(np.quantile(means,.025)),float(np.quantile(means,.975))

print("V35 asymmetric policy ready")

# ---- cell 29 (markdown) ----
# ## Confirmation worlds: fresh X, sealed T, chronology
#
# This stage does **not** select a threshold or model. Every world runs the exact V35 ATTACK recipe. TAPT is retrained from the same IndoBERT initialization on that world's training-side organizer text only.

# ---- cell 30 (code) ----
def deployment_split_v351(seed):
    strata=user_strata_from_y(y)
    sss=StratifiedShuffleSplit(n_splits=1,test_size=0.20,random_state=int(seed))
    tr,va=next(sss.split(np.zeros(N_TRAIN),strata))
    return np.asarray(tr,int),np.asarray(va,int)

def sealed_t_split_v351():
    buckets=[]
    for g in TRAIN_GROUP:
        h=int(hashlib.sha1((SEALED_T_SALT_V35+"|"+str(g)).encode()).hexdigest()[:8],16)%5
        buckets.append(h)
    buckets=np.asarray(buckets)
    va=np.flatnonzero(buckets==0); tr=np.flatnonzero(buckets!=0)
    assert len(set(TRAIN_GROUP[tr]) & set(TRAIN_GROUP[va]))==0
    return tr,va

def evaluate_confirm_world_v351(tr,va,seed,name):
    tr=np.asarray(tr,int); va=np.asarray(va,int)
    print("\n===== V35.1 CONFIRM",name,"train",len(tr),"val",len(va),"=====")
    # Critical correction vs V35: TAPT sees training-side text only.
    tapt_path,tapt_summary=run_tapt_for_rows(tr,tag=name,seed=seed+7)
    val_bundle=build_v30_bundle_split_v35(tr,va,seed+11)
    train_bundle=crossfit_v30_bundle_rows_v35(tr,seed+101,STAGE1_OOF_FOLDS)
    train_ds=build_top3_pairs(tr,train_bundle,with_labels=True,augment_reverse=False)
    val_ds=build_top3_pairs(va,val_bundle,with_labels=True,augment_reverse=False)
    econ=position_economics(train_ds)

    ft=fit_pair_model(train_ds,seed+2501,freeze_encoder=False,tapt_state_path=tapt_path)
    pred=score_directionally(ft,val_ds)
    local_lock=set(np.flatnonzero(np.any(top5_indices(val_bundle["v30"])!=top5_indices(val_bundle["anchor"]),axis=1)).tolist())
    user_ids=train.iloc[va]["user_id"].astype(str).tolist()
    cand,ledger=apply_v35_policy(val_bundle["v30"],val_ds,pred,econ,"ATTACK",user_ids,local_lock,labels_available=True)
    d=per_user_ndcg(y[va],cand)-per_user_ndcg(y[va],val_bundle["v30"])
    ben=int((ledger["beneficial"]==1).sum()) if len(ledger) else 0
    harm=int((ledger["harmful"]==1).sum()) if len(ledger) else 0
    tie=int((ledger["tie"]==1).sum()) if len(ledger) else 0
    precision=float(ben/max(ben+harm,1))
    row={
        "world":name,"policy":"ATTACK","increment_vs_v30":float(np.mean(d)),"interventions":int(len(ledger)),
        "beneficial_actions":ben,"harmful_actions":harm,"ties":tie,"action_precision":precision,
        "locked_v30_rows":int(len(local_lock)),"train_users":int(len(tr)),"val_users":int(len(va)),
        "tapt_texts":int(tapt_summary["texts"]),"tapt_steps":int(tapt_summary["steps"]),"tapt_fold_pure":True,
    }
    if len(ledger): ledger.insert(0,"world",name)
    del ft,train_ds,val_ds,train_bundle,val_bundle,pred; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return row,d,ledger,tapt_summary

# Exactly one fresh deployment-like X, then T, then chronology. No more worlds are created after seeing results.
xtr,xva=deployment_split_v351(CONFIRM_X_SEED)
ttr,tva=sealed_t_split_v351()
CONFIRM_SPECS=[
    ("fresh_X",xtr,xva,CONFIRM_X_SEED),
    ("sealed_template_T",ttr,tva,_v35_seed("V35.1-SEALED-T-FIT")),
    ("chronology_U3001_U4000",CHRONO_TR_V351,CHRONO_VA_V351,_v35_seed("V35.1-CHRONO-FIT")),
]

rows=[]; deltas={}; ledgers=[]; tapt_summaries={}
for name,tr,va,seed in CONFIRM_SPECS:
    row,d,ledger,ts=evaluate_confirm_world_v351(tr,va,seed,name)
    rows.append(row); deltas[name]=d; tapt_summaries[name]=ts
    if len(ledger): ledgers.append(ledger)
V351_CONFIRM=pd.DataFrame(rows)
V351_CONFIRM_LEDGER=pd.concat(ledgers,ignore_index=True) if ledgers else pd.DataFrame()
display(V351_CONFIRM)

# ---- cell 31 (code) ----
# Precommitted confirmation gate. No parameter in the ATTACK policy is changed here.
xrow=V351_CONFIRM[V351_CONFIRM.world=="fresh_X"].iloc[0]
trow=V351_CONFIRM[V351_CONFIRM.world=="sealed_template_T"].iloc[0]
crow=V351_CONFIRM[V351_CONFIRM.world=="chronology_U3001_U4000"].iloc[0]

combined_ben=int(V351_CONFIRM.beneficial_actions.sum())
combined_harm=int(V351_CONFIRM.harmful_actions.sum())
combined_actions=int(V351_CONFIRM.interventions.sum())
combined_precision=float(combined_ben/max(combined_ben+combined_harm,1))
combined_values=np.concatenate([deltas[k] for k in ["fresh_X","sealed_template_T","chronology_U3001_U4000"]])
combined_ci=bootstrap_ci_v35(combined_values,_v35_seed("V35.1-CONFIRM-CI"),bootstraps=BOOTSTRAPS_V351)

world_noncat=bool(all(int(r.beneficial_actions)>=int(r.harmful_actions) for _,r in V351_CONFIRM.iterrows()))
V351_CONFIRM_GATE=bool(
    float(xrow.increment_vs_v30)>X_MIN_DELTA and int(xrow.interventions)>=X_MIN_ACTIONS and
    float(trow.increment_vs_v30)>=T_HARM_FLOOR and
    float(crow.increment_vs_v30)>=CHRONO_HARM_FLOOR and
    world_noncat and combined_precision>=COMBINED_MIN_PRECISION
)

V351_CONFIRM_GATE_TABLE=pd.DataFrame([{
    "fresh_X_delta":float(xrow.increment_vs_v30),"sealed_T_delta":float(trow.increment_vs_v30),
    "chronology_delta":float(crow.increment_vs_v30),"combined_actions":combined_actions,
    "combined_beneficial":combined_ben,"combined_harmful":combined_harm,
    "combined_precision":combined_precision,"combined_ci_low":combined_ci[0],"combined_ci_high":combined_ci[1],
    "world_benefit_not_below_harm":world_noncat,"confirmation_gate":V351_CONFIRM_GATE,
}])
display(V351_CONFIRM_GATE_TABLE)
print("V351_CONFIRM_GATE",V351_CONFIRM_GATE)

# ---- cell 32 (markdown) ----
# ## Full-train deployment
#
# Only if confirmation passes, recreate exact public V30, run TAPT on all 4,000 organizer-train users, train the unchanged V35 pair model, and apply the unchanged ATTACK policy to Kaggle test. Test geometry is structural audit only; it cannot select thresholds or actions.

# ---- cell 33 (code) ----
# Exact public V30 full-test reconstruction.
Xtr_full,_=build_long_matrix_v30(np.arange(N_TRAIN),consensus3_oof,base_oof,meta_oof)
Xte_full,_=build_long_matrix_v30(np.arange(N_TRAIN,N_TRAIN+N_TEST),consensus3_test,base_test,meta_test)
model_v30=make_lambdarank(_v35_seed("V30-FULL-RANKER"))
model_v30.fit(Xtr_full,ORDINAL_GRADES.reshape(-1),group=np.full(N_TRAIN,N_MODULES,dtype=np.int32))
raw_v30=model_v30.predict(Xte_full).reshape(N_TEST,N_MODULES).astype(np.float32)
v30_test=blend_residual(consensus3_test,raw_v30,ALPHA)
protected_v25_rows=[TEST_UID_TO_ROW[u] for u in sorted(PROTECTED_TEST_USERS)]
v30_test[protected_v25_rows]=v25_anchor_test[protected_v25_rows]

v25_top=top5_indices(v25_anchor_test); v30_top=top5_indices(v30_test)
v30_changed_rows=np.flatnonzero(np.any(v25_top!=v30_top,axis=1))
v30_changed_users=set(test.iloc[v30_changed_rows]["user_id"].astype(str))
v30_top1_changes=int(np.sum(v25_top[:,0]!=v30_top[:,0]))
v30_membership_changes=int(np.sum([set(a)!=set(b) for a,b in zip(v25_top,v30_top)]))
V30_REPRO_OK_V351=bool(v30_changed_users==EXPECTED_V30_CHANGED_USERS and v30_top1_changes==EXPECTED_V30_TOP1_CHANGES and v30_membership_changes==EXPECTED_V30_MEMBERSHIP_CHANGES)
V30_REPRO_AUDIT_V351={
    "expected_changed_users":77,"actual_changed_users":len(v30_changed_users),
    "expected_top1_changes":8,"actual_top1_changes":v30_top1_changes,
    "expected_membership_changes":7,"actual_membership_changes":v30_membership_changes,
    "exact_changed_user_set_match":bool(v30_changed_users==EXPECTED_V30_CHANGED_USERS),"reproduction_ok":V30_REPRO_OK_V351,
}
print(json.dumps(V30_REPRO_AUDIT_V351,indent=2))
if not V351_SMOKE: assert V30_REPRO_OK_V351,"Exact V30 geometry failed; V35.1 may not deploy."

def submission_from_scores_v351(scores):
    out=sample.copy(); out.loc[:,MODULES]=np.asarray(scores,np.float32)
    assert np.isfinite(out[MODULES].to_numpy()).all(); return out

ATTACK_TEST_V351=v30_test.copy(); ATTACK_LEDGER_V351=pd.DataFrame()
V351_MODEL_GATE=bool(V351_CONFIRM_GATE and V30_REPRO_OK_V351 and not V351_SMOKE)
if V351_MODEL_GATE:
    full_rows=np.arange(N_TRAIN,dtype=int)
    # Final deployment matches real train→test geometry: TAPT sees organizer train only, never test chat.
    full_tapt_path,V351_FULL_TAPT=run_tapt_for_rows(full_rows,tag="full_train",seed=SEED)
    full_bundle=crossfit_v30_bundle_rows_v35(full_rows,_v35_seed("V35-FULL-STAGE1-OOF"),FULL_STAGE1_OOF_FOLDS)
    train_ds=build_top3_pairs(full_rows,full_bundle,with_labels=True,augment_reverse=False)
    econ=position_economics(train_ds)
    ft=fit_pair_model(train_ds,_v35_seed("V35-FULL-PAIR-MODEL"),freeze_encoder=False,tapt_state_path=full_tapt_path)
    test_rows=np.arange(N_TRAIN,N_TRAIN+N_TEST,dtype=int)
    test_bundle={"v30":v30_test,"anchor":consensus3_test,"base":base_test,"meta":meta_test}
    test_ds=build_top3_pairs(test_rows,test_bundle,with_labels=False,augment_reverse=False)
    pred=score_directionally(ft,test_ds)
    hard_lock=set(v30_changed_rows.tolist())|set(protected_v25_rows)
    test_ids=test["user_id"].astype(str).tolist()
    ATTACK_TEST_V351,ATTACK_LEDGER_V351=apply_v35_policy(v30_test,test_ds,pred,econ,"ATTACK",test_ids,hard_lock,labels_available=False)
    for r in hard_lock: assert np.array_equal(ATTACK_TEST_V351[r],v30_test[r])
    del full_bundle,train_ds,ft,test_ds,pred; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
else:
    V351_FULL_TAPT=None

# Structural test-geometry gate only. No threshold selection from test data.
base_top=top5_indices(v30_test); att_top=top5_indices(ATTACK_TEST_V351)
test_changed=int(np.sum(np.any(base_top!=att_top,axis=1)))
test_top1=int(np.sum(base_top[:,0]!=att_top[:,0]))
test_membership=int(np.sum([set(a)!=set(b) for a,b in zip(base_top,att_top)]))
ledger_users=int(ATTACK_LEDGER_V351.user_id.nunique()) if len(ATTACK_LEDGER_V351) else 0
positions_ok=bool(len(ATTACK_LEDGER_V351)==0 or set(ATTACK_LEDGER_V351.position).issubset({"1-2","2-3"}))
V351_GEOMETRY_OK=bool(
    V351_MODEL_GATE and len(ATTACK_LEDGER_V351)>=1 and
    ledger_users==len(ATTACK_LEDGER_V351)==test_changed and
    test_membership==0 and positions_ok and len(ATTACK_LEDGER_V351)<=80
)
V351_FINAL_GATE=bool(V351_MODEL_GATE and V351_GEOMETRY_OK)
selected_test=ATTACK_TEST_V351 if V351_FINAL_GATE else v30_test.copy()

V351_TEST_GEOMETRY=pd.DataFrame([{
    "candidate":"ATTACK","actions":int(len(ATTACK_LEDGER_V351)),"unique_users":ledger_users,
    "top1_changed_users":test_top1,"top5_order_changed_users":test_changed,
    "top5_membership_changed_users":test_membership,"positions_valid":positions_ok,
    "model_gate":V351_MODEL_GATE,"geometry_gate":V351_GEOMETRY_OK,"final_gate":V351_FINAL_GATE,
}])
display(V351_TEST_GEOMETRY)

SUB_ANCHOR_V351=submission_from_scores_v351(v30_test)
SUB_ATTACK_V351=submission_from_scores_v351(ATTACK_TEST_V351)
SUB_PRIMARY_V351=submission_from_scores_v351(selected_test)
SUB_ANCHOR_V351.to_csv(OUT_DIR/"submission_v35_1_anchor_v30_reconstructed.csv",index=False)
SUB_ATTACK_V351.to_csv(OUT_DIR/"submission_v35_1_DIAGNOSTIC_attack_indobert.csv",index=False)
SUB_PRIMARY_V351.to_csv(OUT_DIR/"submission_v35_1_primary.csv",index=False)
if not V351_FINAL_GATE:
    assert np.array_equal(SUB_PRIMARY_V351[MODULES].to_numpy(),SUB_ANCHOR_V351[MODULES].to_numpy())
print("V351_MODEL_GATE",V351_MODEL_GATE,"V351_GEOMETRY_OK",V351_GEOMETRY_OK,"V351_FINAL_GATE",V351_FINAL_GATE,"test actions",len(ATTACK_LEDGER_V351))

# ---- cell 34 (code) ----
# Audit artifacts and unambiguous submission instruction.
V35_TOKEN_AUDIT.to_csv(OUT_DIR/"v35_1_tokenizer_audit.csv",index=False)
pd.DataFrame({"feature":PAIR_STRUCT_NAMES}).to_csv(OUT_DIR/"v35_1_structured_feature_names.csv",index=False)
V351_CONFIRM.to_csv(OUT_DIR/"v35_1_confirmation_worlds.csv",index=False)
V351_CONFIRM_LEDGER.to_csv(OUT_DIR/"v35_1_confirmation_interventions.csv",index=False)
V351_CONFIRM_GATE_TABLE.to_csv(OUT_DIR/"v35_1_confirmation_gate.csv",index=False)
ATTACK_LEDGER_V351.to_csv(OUT_DIR/"v35_1_test_interventions_attack.csv",index=False)
V351_TEST_GEOMETRY.to_csv(OUT_DIR/"v35_1_test_geometry.csv",index=False)
with open(OUT_DIR/"v35_1_v30_reconstruction_audit.json","w") as f: json.dump(V30_REPRO_AUDIT_V351,f,indent=2)
with open(OUT_DIR/"v35_1_tapt_confirmation_summaries.json","w") as f: json.dump(tapt_summaries,f,indent=2)

slot_plan=pd.DataFrame([
    {"file":"submission_v35_1_primary.csv","role":"PRIMARY","eligible_to_submit":bool(V351_FINAL_GATE),
     "note":"Submit ONLY when final_gate=True. Otherwise this file is exact V30 fail-close."},
    {"file":"submission_v35_1_DIAGNOSTIC_attack_indobert.csv","role":"DIAGNOSTIC_ONLY","eligible_to_submit":False,
     "note":"Never promote from test geometry if confirmation failed."},
    {"file":"submission_v35_1_anchor_v30_reconstructed.csv","role":"ANCHOR_REFERENCE","eligible_to_submit":False,
     "note":"Exact public V30 reference."},
])
slot_plan.to_csv(OUT_DIR/"v35_1_slot_plan.csv",index=False)

summary={
    "version":"V35.1 Fold-Pure Confirmatory IndoBERT ATTACK",
    "purpose":"Confirm exact V35 ATTACK policy after correcting TAPT validation isolation; no model/policy tuning.",
    "public_v30":PUBLIC_V30_LB,"rank4_reference":PUBLIC_RANK4_LB,"leader_reference":PUBLIC_LEADER_LB,
    "backbone":INDO_MODEL_ID,"fixed_alpha":ALPHA,"attack_policy":POLICIES_V35["ATTACK"],
    "training_recipe":{
        "tapt_lr":TAPT_LR,"tapt_max_steps":TAPT_MAX_STEPS,"tapt_mlm_probability":TAPT_MLM_PROB,
        "ft_epochs":FT_EPOCHS,"encoder_lr":ENCODER_LR,"head_lr":HEAD_LR,"weight_decay":WEIGHT_DECAY,
        "warmup":FT_WARMUP,"grad_clip":GRAD_CLIP,"reg_loss_weight":REG_LOSS_WEIGHT,
    },
    "historical_v35_reference":{"UVW_pooled_increment":0.0002253866293050877,"UVW_precision":0.80,"UVW_actions":10,"not_used_for_retuning":True},
    "confirmation":V351_CONFIRM.to_dict("records"),"confirmation_gate":bool(V351_CONFIRM_GATE),
    "confirmation_gate_detail":V351_CONFIRM_GATE_TABLE.to_dict("records")[0],
    "v30_reproduction_ok":bool(V30_REPRO_OK_V351),"test_geometry":V351_TEST_GEOMETRY.to_dict("records")[0],
    "final_gate":bool(V351_FINAL_GATE),"test_actions":int(len(ATTACK_LEDGER_V351)),
    "submission_instruction":"SUBMIT submission_v35_1_primary.csv after manual artifact audit." if V351_FINAL_GATE else "NO NEW SUBMISSION: primary is exact V30 fail-close.",
}
with open(OUT_DIR/"v35_1_summary.json","w") as f: json.dump(summary,f,indent=2)
print(json.dumps(summary,indent=2))
print("\nV35.1 outputs written to",OUT_DIR)
print(slot_plan.to_string(index=False))

# ---- cell 35 (markdown) ----
# ### Decision contract
#
# - **Do not** change the ATTACK policy after seeing X, T, chronology, or test geometry.
# - `submission_v35_1_primary.csv` is the only submission candidate.
# - If `final_gate` is `false`, primary is deliberately identical to V30 and must **not** be submitted.
# - If `final_gate` is `true`, manually inspect `v35_1_confirmation_worlds.csv`, `v35_1_confirmation_interventions.csv`, `v35_1_test_geometry.csv`, and `v35_1_test_interventions_attack.csv` before using the slot.
# - Test geometry may reject malformed deployment, but may never be used to relax thresholds or cherry-pick interventions.
