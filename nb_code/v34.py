# Converted from notebooks/v34.ipynb: markdown cells become comments, %/! magics neutralized.

# ---- cell 0 (markdown) ----
# # MineToday V34 — V30-Locked Risk-Controlled Head12 Cross-Encoder Counterfactual
#
# **Stage-1 is the exact public V30 geometry. V34 never rebuilds the 17-module ordering from scratch.**
#
# V34 tests one narrow question only: when exact V30 ranks module **B at #2 behind A at #1**, is there sufficiently strong, independent evidence that swapping B above A has positive NDCG@5 utility?
#
# The new information source is a frozen multilingual cross-encoder (`BAAI/bge-reranker-v2-m3`) applied only to the current Head12 candidates. The model is not used to generate submission scores directly. Its frozen logits become supervised features for organizer-label action models. The decision layer separately estimates benefit risk, harm risk, and signed counterfactual utility, then abstains unless asymmetric expected utility is positive.
#
# Precommitted experiment discipline:
# - exact V30 `alpha=0.16`, exact 174-feature Stage-1, exact V25/V30 public protections;
# - Head12 only for deployment; Head23 is not part of the candidate surface;
# - two frozen policy tiers only: **SAFE** and **ATTACK**;
# - fresh deployment-like worlds **Q/R/S** for development;
# - a fresh template-disjoint **T** and chronology are rejection-only after a tier is frozen;
# - no threshold, coverage, model, or cross-encoder tuning after seeing Q/R/S;
# - if the gates fail, `submission_v34_primary.csv` is the exact reconstructed V30 surface.
#
# Kaggle: Python + GPU. T4x2 is supported; the cross-encoder runs inference on CUDA and the tabular stages remain deterministic CPU LightGBM. If Hugging Face internet access is disabled, attach `bge-reranker-v2-m3` as a Kaggle dataset or set `V34_CE_MODEL_PATH` to its local directory.

# ---- cell 1 (code) ----
# Runtime + preregistered V34 configuration
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
print("fresh worlds", FRESH_WORLDS)
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
# ## Frozen cross-encoder evidence
#
# The cross-encoder is a **feature extractor only**. It never sees relevance labels and its logits never become submission scores directly. For each current V30 Head12 pair, two target-free views are scored:
# 1. latest learning chat → candidate module;
# 2. recent learning trajectory (up to the last three learning chats) → candidate module.
#
# Only A/B score differences, agreement, and strength enter Stage-2. The cache is keyed by `(user, module, view)`, so repeated Q/R/S requests do not rerun the same inference.

# ---- cell 19 (code) ----
# Cross-encoder text views + lazy GPU scorer.
def _learning_chat_rows(u):
    g=user_chat.get(u)
    if g is None or len(g)==0: return []
    learn=g[~g["is_noise"]].copy()
    if len(learn)==0: learn=g.copy()
    return list(learn.itertuples(index=False))

CE_LATEST_TEXT=[]; CE_RECENT_TEXT=[]
for u in all_users:
    rows=_learning_chat_rows(u)
    if not rows:
        latest="Saya belum memiliki riwayat chat pembelajaran yang eksplisit."
        recent=latest
    else:
        latest=str(rows[-1].clean).strip() or "Kebutuhan pembelajaran terbaru tidak eksplisit."
        tail=rows[-3:]
        recent=" || ".join([f"pesan_{len(tail)-k}: {str(r.clean).strip()}" for k,r in enumerate(tail)])
    CE_LATEST_TEXT.append("Kebutuhan terbaru siswa: " + latest)
    CE_RECENT_TEXT.append("Riwayat belajar terbaru siswa: " + recent)
CE_LATEST_TEXT=np.asarray(CE_LATEST_TEXT,dtype=object)
CE_RECENT_TEXT=np.asarray(CE_RECENT_TEXT,dtype=object)

MODULE_CE_DOCS=[]
mod_idx=modules.set_index("module_id")
for mid in MODULES:
    r=mod_idx.loc[mid]
    MODULE_CE_DOCS.append(
        f"Modul {r['module_name']}. Prasyarat: {r['prerequisite_level']}. "
        f"Materi dan silabus: {r['description_and_syllabus']}"
    )
MODULE_CE_DOCS=np.asarray(MODULE_CE_DOCS,dtype=object)

OUT_DIR=Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("/mnt/data")
OUT_DIR.mkdir(parents=True,exist_ok=True)
CE_CACHE_PATH=OUT_DIR/"v34_cross_encoder_cache.pkl"


def locate_ce_model():
    env=os.getenv("V34_CE_MODEL_PATH","").strip()
    if env and Path(env).exists(): return env
    root=Path("/kaggle/input")
    if root.exists():
        for p in root.rglob("config.json"):
            s=str(p.parent).lower()
            if "bge" in s and "reranker" in s and "v2" in s and "m3" in s:
                return str(p.parent)
    return CE_MODEL_ID

class FrozenCrossEncoder:
    def __init__(self):
        self.model=None; self.tokenizer=None
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size=int(CE_BATCH_SIZE)
        self.cache={}
        if CE_CACHE_PATH.exists():
            try:
                with open(CE_CACHE_PATH,"rb") as f: self.cache=pickle.load(f)
                print("loaded CE cache entries",len(self.cache))
            except Exception:
                self.cache={}
    def _load(self):
        if self.model is not None: return
        src=locate_ce_model(); print("loading frozen cross-encoder from",src,"on",self.device)
        self.tokenizer=AutoTokenizer.from_pretrained(src,cache_dir=str(OUT_DIR/"hf_cache"))
        dtype=torch.float16 if self.device.type=="cuda" else torch.float32
        self.model=AutoModelForSequenceClassification.from_pretrained(src,cache_dir=str(OUT_DIR/"hf_cache"),torch_dtype=dtype)
        self.model.to(self.device); self.model.eval()
    def _flush(self):
        try:
            with open(CE_CACHE_PATH,"wb") as f: pickle.dump(self.cache,f,pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print("CE cache flush warning:",repr(e))
    def _query(self,row,view):
        return CE_LATEST_TEXT[int(row)] if view=="latest" else CE_RECENT_TEXT[int(row)]
    def score(self,rows,mods,view):
        rows=np.asarray(rows,int); mods=np.asarray(mods,int)
        assert rows.shape==mods.shape
        out=np.empty(len(rows),np.float32); missing=[]
        for i,(r,m) in enumerate(zip(rows,mods)):
            key=(int(r),int(m),str(view))
            if key in self.cache: out[i]=self.cache[key]
            else: missing.append((i,key))
        if missing:
            self._load(); start=0; bs=self.batch_size; added=0
            while start<len(missing):
                chunk=missing[start:start+bs]
                q=[self._query(key[0],view) for _,key in chunk]
                d=[str(MODULE_CE_DOCS[key[1]]) for _,key in chunk]
                try:
                    enc=self.tokenizer(q,d,padding=True,truncation=True,max_length=CE_MAX_LENGTH,return_tensors="pt")
                    enc={k:v.to(self.device,non_blocking=True) for k,v in enc.items()}
                    with torch.inference_mode():
                        logits=self.model(**enc).logits
                        scores=logits.float().reshape(len(chunk),-1)[:,0].detach().cpu().numpy().astype(np.float32)
                    for (ii,key),s in zip(chunk,scores):
                        self.cache[key]=float(s); out[ii]=s; added+=1
                    start+=len(chunk)
                    if added and added%CE_CACHE_FLUSH_EVERY < len(chunk): self._flush()
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and self.device.type=="cuda" and bs>2:
                        torch.cuda.empty_cache(); bs=max(2,bs//2); self.batch_size=bs
                        print("CUDA OOM: reducing CE batch size to",bs); continue
                    raise
            self._flush()
        return out
    def head_features(self,rows,a,b):
        rows=np.asarray(rows,int); a=np.asarray(a,int); b=np.asarray(b,int)
        al=self.score(rows,a,"latest"); bl=self.score(rows,b,"latest")
        ar=self.score(rows,a,"recent"); br=self.score(rows,b,"recent")
        dl=bl-al; dr=br-ar; dm=.5*(dl+dr)
        agree=((dl>0)&(dr>0)).astype(np.float32)
        disagree=((dl*dr)<0).astype(np.float32)
        strength=np.minimum(np.abs(dl),np.abs(dr)).astype(np.float32)
        return np.column_stack([al,bl,dl,ar,br,dr,dm,agree,disagree,strength]).astype(np.float32)

CE_SCORER=FrozenCrossEncoder()
CE_FEATURE_NAMES=["ce_a_latest","ce_b_latest","ce_b_minus_a_latest","ce_a_recent","ce_b_recent","ce_b_minus_a_recent","ce_delta_mean","ce_both_vote_b","ce_views_disagree","ce_min_abs_strength"]
print("cross-encoder feature surface ready; model loads lazily")

# ---- cell 20 (markdown) ----
# ## Strict V30 Head12 counterfactual dataset
#
# Every Stage-2 example is one user and one action: swap the current V30 #1 and #2. Stage-2 training receives **strict V30 OOF predictions**. Outer validation receives a V30 model trained without those validation users.
#
# The action target has three states: beneficial, tie, harmful. Benefit and harm are modeled separately; ties are not silently treated as harm. A separate regressor predicts signed exact ΔNDCG@5.

# ---- cell 21 (code) ----
# V30 bundle variants retain base/meta surfaces for Stage-2 without changing Stage-1 scores.
def build_v30_bundle_split(train_rows,val_rows,seed):
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


def crossfit_v30_bundle_rows(rows,seed,n_splits=STAGE1_OOF_FOLDS):
    rows=np.asarray(rows,int); strata=user_strata_from_y(y[rows])
    ns=min(int(n_splits),max(2,int(pd.Series(strata).value_counts().min())))
    sk=StratifiedKFold(n_splits=ns,shuffle=True,random_state=int(seed))
    out={k:np.zeros((len(rows),N_MODULES),np.float32) for k in ["v30","anchor","base","meta"]}
    fid=np.full(len(rows),-1,np.int16)
    for f,(lt,lv) in enumerate(sk.split(np.zeros(len(rows)),strata)):
        tr=rows[lt]; va=rows[lv]
        print(f"  strict Stage-1 OOF fold {f+1}/{ns}: train={len(tr)} val={len(va)}")
        z=build_v30_bundle_split(tr,va,seed+1009+f*211)
        for k in out: out[k][lv]=z[k]
        fid[lv]=f
    assert (fid>=0).all()
    for k in out: assert np.isfinite(out[k]).all()
    out["fold_id"]=fid; return out


def adjacent_swap_delta(scores,y_true,a,b):
    scores=np.asarray(scores,np.float32); a=np.asarray(a,int); b=np.asarray(b,int)
    before=per_user_ndcg(y_true,scores); swapped=scores.copy(); rr=np.arange(len(scores))
    tmp=swapped[rr,a].copy(); swapped[rr,a]=swapped[rr,b]; swapped[rr,b]=tmp
    return (per_user_ndcg(y_true,swapped)-before).astype(np.float32)


def _pair_cols(mat,rows,a,b):
    aa=np.asarray(mat)[rows,a]; bb=np.asarray(mat)[rows,b]
    return [aa,bb,bb-aa]

HEAD_BASE_FEATURE_NAMES = [
    "v30_a","v30_b","v30_margin",
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
    *CE_FEATURE_NAMES,
]
V34_ACTION_FEATURE_NAMES=HEAD_BASE_FEATURE_NAMES+[f"user_f{i:03d}" for i in range(USER_FEATURES.shape[1])]+[f"a_{m}" for m in MODULES]+[f"b_{m}" for m in MODULES]+[f"a_level_{i}" for i in range(5)]+[f"b_level_{i}" for i in range(5)]


def build_head12(rows_global,bundle,with_labels=True):
    rows_global=np.asarray(rows_global,int); v30=np.asarray(bundle["v30"],np.float32)
    order=np.argsort(-v30,axis=1,kind="mergesort"); a=order[:,0].astype(np.int16); b=order[:,1].astype(np.int16)
    rr=np.arange(len(rows_global)); anchor=np.asarray(bundle["anchor"],np.float32)
    base=np.asarray(bundle["base"],np.float32); meta=np.asarray(bundle["meta"],np.float32)
    residual=v30-anchor
    dense=[
        v30[rr,a],v30[rr,b],v30[rr,a]-v30[rr,b],
        anchor[rr,a],anchor[rr,b],anchor[rr,b]-anchor[rr,a],
        base[rr,a],base[rr,b],base[rr,b]-base[rr,a],
        meta[rr,a],meta[rr,b],meta[rr,b]-meta[rr,a],
        residual[rr,a],residual[rr,b],residual[rr,b]-residual[rr,a],
        *_pair_cols(LEX_WORD,rows_global,a,b), *_pair_cols(LEX_CHAR,rows_global,a,b),
        *_pair_cols(MODULE_COUNT,rows_global,a,b), *_pair_cols(MODULE_RECENCY,rows_global,a,b),
        *_pair_cols(MODULE_LATEST,rows_global,a,b), *_pair_cols(MODULE_READINESS,rows_global,a,b),
        *_pair_cols(MODULE_INVALID,rows_global,a,b),
    ]
    numeric=np.column_stack(dense).astype(np.float32)
    ce=CE_SCORER.head_features(rows_global,a,b)
    one_a=np.eye(N_MODULES,dtype=np.float32)[a]; one_b=np.eye(N_MODULES,dtype=np.float32)[b]
    lvl_a=MODULE_LEVEL_ONEHOT[a]; lvl_b=MODULE_LEVEL_ONEHOT[b]
    Xh=np.hstack([numeric,ce,USER_FEATURES[rows_global].astype(np.float32),one_a,one_b,lvl_a,lvl_b]).astype(np.float32)
    names=V34_ACTION_FEATURE_NAMES
    assert Xh.shape[1]==len(names) and np.isfinite(Xh).all()
    ds={"X":Xh,"feature_names":names,"a":a,"b":b,"rows_global":rows_global,"margin":(v30[rr,a]-v30[rr,b]).astype(np.float32)}
    ds["ce_delta_latest"]=ce[:,2]; ds["ce_delta_recent"]=ce[:,5]; ds["ce_delta_mean"]=ce[:,6]; ds["ce_agree"]=ce[:,7]
    ds["pair_id"]=(a.astype(np.int32)*N_MODULES+b.astype(np.int32))
    if with_labels:
        yy=y[rows_global]
        delta=adjacent_swap_delta(v30,yy,a,b)
        ds["delta"]=delta; ds["benefit"]=(delta>1e-12).astype(np.int8); ds["harm"]=(delta<-1e-12).astype(np.int8)
    return ds

print("Head12 Stage-2 feature builder ready")

# ---- cell 22 (markdown) ----
# ## Three-output action learner + asymmetric policy layer
#
# The learner deliberately omits exact pair-ID one-hot features to reduce shortcut collapse. Ordered-pair history is used only as a smoothed **risk prior/veto** fitted from Stage-2 training labels. Raw learner predictions are cross-fitted before Platt/Ridge calibration.

# ---- cell 23 (code) ----
def make_action_classifier(seed):
    return lgb.LGBMClassifier(
        objective="binary",n_estimators=ACTION_ESTIMATORS,learning_rate=ACTION_LR,
        num_leaves=7,max_depth=4,min_child_samples=70,
        reg_alpha=2.0,reg_lambda=15.0,colsample_bytree=.72,
        subsample=.90,subsample_freq=1,max_bin=127,
        random_state=int(seed),n_jobs=4,verbosity=-1,force_col_wise=True,
    )

def make_action_regressor(seed):
    return lgb.LGBMRegressor(
        objective="huber",alpha=.85,n_estimators=ACTION_ESTIMATORS,learning_rate=ACTION_LR,
        num_leaves=7,max_depth=4,min_child_samples=70,
        reg_alpha=2.0,reg_lambda=18.0,colsample_bytree=.72,
        subsample=.90,subsample_freq=1,max_bin=127,
        random_state=int(seed),n_jobs=4,verbosity=-1,force_col_wise=True,
    )

def _logit(p):
    p=np.clip(np.asarray(p,float),1e-5,1-1e-5); return np.log(p/(1-p))


def stage2_raw_crossfit(ds,seed):
    X=ds["X"]; ben=ds["benefit"]; harm=ds["harm"]; delta=ds["delta"]
    state=np.where(ben>0,1,np.where(harm>0,2,0)).astype(int)
    ns=min(STAGE2_CAL_FOLDS,max(2,int(pd.Series(state).value_counts().min())))
    sk=StratifiedKFold(n_splits=ns,shuffle=True,random_state=int(seed))
    pb=np.zeros(len(X),np.float32); ph=np.zeros(len(X),np.float32); ur=np.zeros(len(X),np.float32); fid=np.full(len(X),-1,np.int16)
    for f,(tr,va) in enumerate(sk.split(np.zeros(len(X)),state)):
        mb=make_action_classifier(seed+101+f); mh=make_action_classifier(seed+201+f); mr=make_action_regressor(seed+301+f)
        mb.fit(X[tr],ben[tr]); mh.fit(X[tr],harm[tr]); mr.fit(X[tr],delta[tr])
        pb[va]=mb.predict_proba(X[va])[:,1]; ph[va]=mh.predict_proba(X[va])[:,1]; ur[va]=mr.predict(X[va]); fid[va]=f
    assert (fid>=0).all() and np.isfinite(pb).all() and np.isfinite(ph).all() and np.isfinite(ur).all()
    return pb,ph,ur,fid


def fit_stage2_bundle(ds,seed):
    pb_oof,ph_oof,ur_oof,fid=stage2_raw_crossfit(ds,seed+11)
    # Platt scaling from strictly OOF raw probabilities only.
    cal_b=LogisticRegression(C=.35,max_iter=2500,random_state=int(seed+41)).fit(_logit(pb_oof)[:,None],ds["benefit"])
    cal_h=LogisticRegression(C=.35,max_iter=2500,random_state=int(seed+42)).fit(_logit(ph_oof)[:,None],ds["harm"])
    pb_cal=cal_b.predict_proba(_logit(pb_oof)[:,None])[:,1]
    ph_cal=cal_h.predict_proba(_logit(ph_oof)[:,None])[:,1]
    # Direct signed-utility calibration uses OOF raw regression plus calibrated benefit/harm risk.
    util_cal=Ridge(alpha=24.0,solver="lsqr",tol=1e-5,max_iter=4000)
    util_cal.fit(np.column_stack([ur_oof,pb_cal,ph_cal]),ds["delta"])
    mb=make_action_classifier(seed+501); mh=make_action_classifier(seed+601); mr=make_action_regressor(seed+701)
    mb.fit(ds["X"],ds["benefit"]); mh.fit(ds["X"],ds["harm"]); mr.fit(ds["X"],ds["delta"])
    return {"benefit_model":mb,"harm_model":mh,"utility_model":mr,"benefit_cal":cal_b,"harm_cal":cal_h,"utility_cal":util_cal,
            "oof_pb":pb_cal.astype(np.float32),"oof_ph":ph_cal.astype(np.float32),"oof_util":util_cal.predict(np.column_stack([ur_oof,pb_cal,ph_cal])).astype(np.float32),"oof_fold":fid}


def predict_stage2(bundle,ds):
    X=ds["X"]
    rb=bundle["benefit_model"].predict_proba(X)[:,1]; rh=bundle["harm_model"].predict_proba(X)[:,1]; ru=bundle["utility_model"].predict(X)
    pb=bundle["benefit_cal"].predict_proba(_logit(rb)[:,None])[:,1]
    ph=bundle["harm_cal"].predict_proba(_logit(rh)[:,None])[:,1]
    util=bundle["utility_cal"].predict(np.column_stack([ru,pb,ph]))
    return pb.astype(np.float32),ph.astype(np.float32),util.astype(np.float32)

PAIR_ALPHA=20.0; PAIR_UTILITY_ALPHA=28.0

def fit_pair_risk_prior(ds):
    pair=ds["pair_id"].astype(int); nkey=N_MODULES*N_MODULES
    cnt=np.bincount(pair,minlength=nkey).astype(np.float64)
    sb=np.bincount(pair,weights=ds["benefit"],minlength=nkey).astype(np.float64)
    sh=np.bincount(pair,weights=ds["harm"],minlength=nkey).astype(np.float64)
    sd=np.bincount(pair,weights=ds["delta"],minlength=nkey).astype(np.float64)
    gb=float(ds["benefit"].mean()); gh=float(ds["harm"].mean()); gd=float(ds["delta"].mean())
    return {"cnt":cnt,"sb":sb,"sh":sh,"sd":sd,"gb":gb,"gh":gh,"gd":gd}

def pair_risk_apply(ds,state):
    p=ds["pair_id"].astype(int); c=state["cnt"][p]
    pb=(state["sb"][p]+PAIR_ALPHA*state["gb"])/(c+PAIR_ALPHA)
    ph=(state["sh"][p]+PAIR_ALPHA*state["gh"])/(c+PAIR_ALPHA)
    du=(state["sd"][p]+PAIR_UTILITY_ALPHA*state["gd"])/(c+PAIR_UTILITY_ALPHA)
    return c.astype(np.float32),pb.astype(np.float32),ph.astype(np.float32),du.astype(np.float32)


def apply_head12_policy(v30_scores,ds,pb,ph,util,pair_state,policy_name,user_ids,hard_lock=None):
    policy=POLICIES[str(policy_name)]; hard_lock=set() if hard_lock is None else set(hard_lock)
    n=len(ds["X"]); pair_support,pair_pb,pair_ph,pair_util=pair_risk_apply(ds,pair_state)
    asym=pb*HEAD12_BENEFIT_MAG-ph*HEAD12_HARM_MAG
    ce_mean=ds["ce_delta_mean"]; ce_agree=ds["ce_agree"]>0.5
    eligible=(
        (pb>=policy["p_benefit_min"]) & (ph<=policy["p_harm_max"]) &
        (asym>=policy["asym_eu_min"]) & (util>=policy["utility_min"]) &
        (ds["margin"]<=policy["margin_cap"]) &
        (pair_support>=policy["pair_support_min"]) & (pair_util>=policy["pair_utility_floor"]) &
        (ce_mean>0)
    )
    if policy["require_ce_agreement"]: eligible &= ce_agree
    if hard_lock:
        lock=np.zeros(n,bool); lock[list(sorted(hard_lock))]=True; eligible &= ~lock
    # Risk score demands agreement between asymmetric probability economics and direct signed utility.
    risk_score=np.sqrt(np.maximum(asym,0)*np.maximum(util,0))
    k=max(1,int(np.ceil(policy["coverage_cap"]*n)))
    cand=np.flatnonzero(eligible)
    if len(cand)>k: cand=cand[np.argsort(-risk_score[cand],kind="mergesort")[:k]]
    selected=np.zeros(n,bool); selected[cand]=True
    out=np.asarray(v30_scores,np.float32).copy(); rr=np.arange(n)
    for i in cand:
        a=int(ds["a"][i]); b=int(ds["b"][i]); tmp=float(out[i,a]); out[i,a]=out[i,b]; out[i,b]=tmp
    ledger=[]
    for i in cand:
        rec={"row":int(i),"user_id":str(user_ids[i]),"policy":str(policy_name),"a_module":MODULES[int(ds['a'][i])],"b_module":MODULES[int(ds['b'][i])],
             "v30_margin":float(ds['margin'][i]),"p_benefit":float(pb[i]),"p_harm":float(ph[i]),"asym_expected_utility":float(asym[i]),"pred_signed_utility":float(util[i]),
             "ce_delta_latest":float(ds['ce_delta_latest'][i]),"ce_delta_recent":float(ds['ce_delta_recent'][i]),"ce_delta_mean":float(ce_mean[i]),"ce_agreement":bool(ce_agree[i]),
             "pair_support":float(pair_support[i]),"pair_benefit_prior":float(pair_pb[i]),"pair_harm_prior":float(pair_ph[i]),"pair_utility_prior":float(pair_util[i]),"risk_score":float(risk_score[i])}
        if "delta" in ds:
            rec.update({"actual_delta":float(ds['delta'][i]),"actual_benefit":bool(ds['benefit'][i]),"actual_harm":bool(ds['harm'][i])})
        ledger.append(rec)
    return out,pd.DataFrame(ledger),selected,asym,risk_score

print("Three-output risk-controlled Head12 learner ready")

# ---- cell 24 (markdown) ----
# ## Fresh deployment-like worlds Q/R/S
#
# Q/R/S are user-disjoint stratified worlds. Repeated chat templates may appear on both sides because the real train→test deployment has heavy template overlap. Labels never cross the user boundary. The cross-encoder is frozen and target-free.

# ---- cell 25 (code) ----
def deployment_split(seed):
    strata=user_strata_from_y(y)
    sss=StratifiedShuffleSplit(n_splits=1,test_size=(0.10 if V34_SMOKE else 0.20),random_state=int(seed))
    tr,va=next(sss.split(np.zeros(N_TRAIN),strata)); return np.asarray(tr,int),np.asarray(va,int)


def local_v30_lock(v30,anchor):
    return set(np.flatnonzero(np.any(top5_indices(v30)!=top5_indices(anchor),axis=1)).tolist())


def evaluate_world(train_rows,val_rows,seed,world,selected_policy=None):
    train_rows=np.asarray(train_rows,int); val_rows=np.asarray(val_rows,int)
    print(f"\n=== V34 {world}: train={len(train_rows)} val={len(val_rows)} ===")
    tr_bundle=crossfit_v30_bundle_rows(train_rows,seed+100,STAGE1_OOF_FOLDS)
    va_bundle=build_v30_bundle_split(train_rows,val_rows,seed+200)
    tr_ds=build_head12(train_rows,tr_bundle,with_labels=True); va_ds=build_head12(val_rows,va_bundle,with_labels=True)
    stage2=fit_stage2_bundle(tr_ds,seed+300); pb,ph,util=predict_stage2(stage2,va_ds); pair_state=fit_pair_risk_prior(tr_ds)
    train_b_auc=float(roc_auc_score(tr_ds["benefit"],stage2["oof_pb"])) if len(np.unique(tr_ds["benefit"]))>1 else np.nan
    train_h_auc=float(roc_auc_score(tr_ds["harm"],stage2["oof_ph"])) if len(np.unique(tr_ds["harm"]))>1 else np.nan
    train_util_corr=float(np.corrcoef(tr_ds["delta"],stage2["oof_util"])[0,1]) if np.std(stage2["oof_util"])>0 else np.nan
    val_ce_b_auc=float(roc_auc_score(va_ds["benefit"],va_ds["ce_delta_mean"])) if len(np.unique(va_ds["benefit"]))>1 else np.nan
    policies=[selected_policy] if selected_policy is not None else ["SAFE","ATTACK"]
    base_u=per_user_ndcg(y[val_rows],va_bundle["v30"]); lock=local_v30_lock(va_bundle["v30"],va_bundle["anchor"])
    rows=[]; deltas={}; ledgers={}
    for pol in policies:
        val_user_ids=train.iloc[val_rows]["user_id"].astype(str).tolist()
        cand,ledger,sel,asym,risk=apply_head12_policy(va_bundle["v30"],va_ds,pb,ph,util,pair_state,pol,val_user_ids,hard_lock=lock)
        cu=per_user_ndcg(y[val_rows],cand); inc=cu-base_u; deltas[pol]=inc; ledgers[pol]=ledger
        if len(ledger):
            pos=int(ledger["actual_benefit"].sum()); neg=int(ledger["actual_harm"].sum()); ties=int(len(ledger)-pos-neg); precision=pos/max(pos+neg,1)
        else:
            pos=neg=ties=0; precision=np.nan
        rows.append({"world":world,"policy":pol,"n":len(val_rows),"v30_ndcg":float(base_u.mean()),"candidate_ndcg":float(cu.mean()),"increment_vs_v30":float(inc.mean()),
                     "positive_user_rate":float(np.mean(inc>0)),"harm_user_rate":float(np.mean(inc<0)),"interventions":int(sel.sum()),"intervention_rate":float(sel.mean()),
                     "beneficial_actions":pos,"harmful_actions":neg,"ties":ties,"action_precision":float(precision),"local_v30_locked":int(len(lock)),
                     "stage2_oof_benefit_auc":train_b_auc,"stage2_oof_harm_auc":train_h_auc,"stage2_oof_utility_corr":train_util_corr,"cross_encoder_val_benefit_auc":val_ce_b_auc,
                     "top1_change_rate":top1_change_rate(va_bundle["v30"],cand),"top5_order_change_rate":top5_order_change_rate(va_bundle["v30"],cand)})
    del tr_bundle,va_bundle,tr_ds,va_ds,stage2; gc.collect()
    return pd.DataFrame(rows),deltas,ledgers

DEV_ROWS=[]; DEV_DELTA={"SAFE":{},"ATTACK":{}}; DEV_LEDGERS=[]
for world,seed in FRESH_WORLDS.items():
    tr,va=deployment_split(seed); df,dd,ll=evaluate_world(tr,va,seed,world)
    DEV_ROWS.append(df)
    for pol in ["SAFE","ATTACK"]:
        DEV_DELTA[pol][world]=dd[pol]
        if len(ll[pol]):
            x=ll[pol].copy(); x["world"]=world; DEV_LEDGERS.append(x)
V34_DEVELOPMENT=pd.concat(DEV_ROWS,ignore_index=True)
V34_DEV_LEDGER=pd.concat(DEV_LEDGERS,ignore_index=True) if DEV_LEDGERS else pd.DataFrame()
display(V34_DEVELOPMENT)

pooled=[]; gate_rows=[]
for pol in ["SAFE","ATTACK"]:
    means={w:float(np.mean(d)) for w,d in DEV_DELTA[pol].items()}; pooled_mean=float(np.mean(list(means.values()))); worst=float(np.min(list(means.values())))
    ci=world_bootstrap_ci(DEV_DELTA[pol],_seed_from_text(f"V34-{pol}-BOOT"))
    dpol=V34_DEVELOPMENT[V34_DEVELOPMENT.policy==pol]
    pos=int(dpol.beneficial_actions.sum()); neg=int(dpol.harmful_actions.sum()); total=int(dpol.interventions.sum()); precision=pos/max(pos+neg,1)
    if pol=="SAFE":
        passed=bool(all(v>=0 for v in means.values()) and pooled_mean>=SAFE_DEV_MIN_POOLED and worst>=SAFE_DEV_MIN_WORST and ci[0]>=SAFE_DEV_CI_FLOOR and precision>=SAFE_DEV_MIN_PRECISION and total>=SAFE_DEV_MIN_ACTIONS)
    else:
        passed=bool(all(v>0 for v in means.values()) and pooled_mean>=ATTACK_DEV_MIN_POOLED and worst>=ATTACK_DEV_MIN_WORST and ci[0]>=ATTACK_DEV_CI_FLOOR and precision>=ATTACK_DEV_MIN_PRECISION and total>=ATTACK_DEV_MIN_ACTIONS)
    pooled.append({"policy":pol,"pooled_increment":pooled_mean,"worst_world":worst,"ci_low":ci[0],"ci_high":ci[1],"beneficial_actions":pos,"harmful_actions":neg,"total_actions":total,"action_precision":precision,**{f"world_{k}":v for k,v in means.items()}})
    gate_rows.append({"policy":pol,"development_gate":passed})
V34_POOLED=pd.DataFrame(pooled); V34_GATES=pd.DataFrame(gate_rows)
display(V34_POOLED); display(V34_GATES)

# Precommitted ladder: ATTACK only on leader-scale evidence; otherwise SAFE.
if bool(V34_GATES.loc[V34_GATES.policy=="ATTACK","development_gate"].iloc[0]): DEV_WINNER="ATTACK"
elif bool(V34_GATES.loc[V34_GATES.policy=="SAFE","development_gate"].iloc[0]): DEV_WINNER="SAFE"
else: DEV_WINNER=None
DEV_GATE=DEV_WINNER is not None
print("DEV_WINNER",DEV_WINNER,"DEV_GATE",DEV_GATE)

# ---- cell 26 (markdown) ----
# ## Sealed template-disjoint T + chronology
#
# Only the frozen Q/R/S winner reaches this cell. T and chronology are rejection-only; they never retune the policy. Because the cross-encoder may have less useful evidence under cold templates, the criterion is catastrophic-harm control rather than a demand for large positive lift.

# ---- cell 27 (code) ----
def sealed_t_split():
    buckets=[]
    for g in TRAIN_GROUP:
        h=int(hashlib.sha1((SEALED_T_SALT+"|"+str(g)).encode()).hexdigest()[:8],16)%5
        buckets.append(h)
    buckets=np.asarray(buckets); va=np.flatnonzero(buckets==0); tr=np.flatnonzero(buckets!=0)
    assert len(set(TRAIN_GROUP[tr])&set(TRAIN_GROUP[va]))==0
    return tr,va

CHRONO_TR=np.arange(0,3000,dtype=int); CHRONO_VA=np.arange(3000,4000,dtype=int)
SEALED_ROWS=[]; SEALED_DELTAS={}
if DEV_WINNER is not None:
    ttr,tva=sealed_t_split(); df,dd,ll=evaluate_world(ttr,tva,_seed_from_text("V34-SEALED-T-FIT"),"sealed_template_T",DEV_WINNER)
    SEALED_ROWS.append(df); SEALED_DELTAS["T"]=dd[DEV_WINNER]
    df,dd,ll=evaluate_world(CHRONO_TR,CHRONO_VA,_seed_from_text("V34-CHRONO-FIT"),"chronology_U3001_U4000",DEV_WINNER)
    SEALED_ROWS.append(df); SEALED_DELTAS["chronology"]=dd[DEV_WINNER]
    V34_SEALED=pd.concat(SEALED_ROWS,ignore_index=True)
    tdelta=float(np.mean(SEALED_DELTAS["T"])); cdelta=float(np.mean(SEALED_DELTAS["chronology"]))
    if DEV_WINNER=="ATTACK": SEALED_GATE=bool(tdelta>=ATTACK_T_FLOOR and cdelta>=ATTACK_CHRONO_FLOOR)
    else: SEALED_GATE=bool(tdelta>=SAFE_T_FLOOR and cdelta>=SAFE_CHRONO_FLOOR)
else:
    V34_SEALED=pd.DataFrame([{"world":"skipped_development_gate_failed","policy":None,"increment_vs_v30":np.nan,"interventions":0}])
    SEALED_GATE=False

display(V34_SEALED); print("SEALED_GATE",SEALED_GATE)

# ---- cell 28 (markdown) ----
# ## Full-train deployment
#
# The full-test Stage-1 is reconstructed exactly as V30. A geometry audit must recover the known 77 V30-changed users, 8 top-1 changes, and 7 membership changes before V34 is allowed to modify anything.
#
# If the development or sealed gate fails, expensive full Stage-2 cross-encoder inference is skipped and the primary file is exact V30.

# ---- cell 29 (code) ----
OUT_DIR=Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("/mnt/data")
OUT_DIR.mkdir(parents=True,exist_ok=True)

# Exact public V30 full-test reconstruction.
Xtr_full,_=build_long_matrix_v30(np.arange(N_TRAIN),consensus3_oof,base_oof,meta_oof)
Xte_full,_=build_long_matrix_v30(np.arange(N_TRAIN,N_TRAIN+N_TEST),consensus3_test,base_test,meta_test)
model_v30=make_lambdarank(_seed_from_text("V30-FULL-RANKER"))
model_v30.fit(Xtr_full,ORDINAL_GRADES.reshape(-1),group=np.full(N_TRAIN,N_MODULES,dtype=np.int32))
raw_v30=model_v30.predict(Xte_full).reshape(N_TEST,N_MODULES).astype(np.float32)
v30_test=blend_residual(consensus3_test,raw_v30,ALPHA)
protected_v25_rows=[TEST_UID_TO_ROW[u] for u in sorted(PROTECTED_TEST_USERS)]
v30_test[protected_v25_rows]=v25_anchor_test[protected_v25_rows]

v25_top=top5_indices(v25_anchor_test); v30_top=top5_indices(v30_test)
v30_changed_rows=np.flatnonzero(np.any(v25_top!=v30_top,axis=1)); v30_changed_users=set(test.iloc[v30_changed_rows]["user_id"].astype(str))
v30_top1_changes=int(np.sum(v25_top[:,0]!=v30_top[:,0])); v30_membership_changes=int(np.sum([set(a)!=set(b) for a,b in zip(v25_top,v30_top)]))
V30_REPRO_OK=bool(v30_changed_users==EXPECTED_V30_CHANGED_USERS and v30_top1_changes==EXPECTED_V30_TOP1_CHANGES and v30_membership_changes==EXPECTED_V30_MEMBERSHIP_CHANGES)
V30_REPRO_AUDIT={"expected_changed_users":77,"actual_changed_users":len(v30_changed_users),"expected_top1_changes":8,"actual_top1_changes":v30_top1_changes,"expected_membership_changes":7,"actual_membership_changes":v30_membership_changes,"exact_changed_user_set_match":bool(v30_changed_users==EXPECTED_V30_CHANGED_USERS),"reproduction_ok":V30_REPRO_OK}
print(json.dumps(V30_REPRO_AUDIT,indent=2))
if not V34_SMOKE: assert V30_REPRO_OK, "Exact V30 geometry failed; V34 is not allowed to deploy."

def submission_from_scores(scores):
    out=sample.copy(); out.loc[:,MODULES]=np.asarray(scores,np.float32); assert np.isfinite(out[MODULES].to_numpy()).all(); return out

SAFE_TEST=v30_test.copy(); ATTACK_TEST=v30_test.copy(); SAFE_LEDGER=pd.DataFrame(); ATTACK_LEDGER=pd.DataFrame()
FINAL_GATE=bool(DEV_GATE and SEALED_GATE and V30_REPRO_OK and not V34_SMOKE)

if FINAL_GATE:
    full_rows=np.arange(N_TRAIN,dtype=int)
    full_bundle=crossfit_v30_bundle_rows(full_rows,_seed_from_text("V34-FULL-STAGE1-OOF"),FULL_STAGE1_OOF_FOLDS)
    train_ds=build_head12(full_rows,full_bundle,with_labels=True)
    stage2=fit_stage2_bundle(train_ds,_seed_from_text("V34-FULL-STAGE2")); pair_state=fit_pair_risk_prior(train_ds)
    test_rows=np.arange(N_TRAIN,N_TRAIN+N_TEST,dtype=int)
    test_bundle={"v30":v30_test,"anchor":consensus3_test,"base":base_test,"meta":meta_test}
    test_ds=build_head12(test_rows,test_bundle,with_labels=False)
    pb,ph,util=predict_stage2(stage2,test_ds)
    hard_lock=set(v30_changed_rows.tolist())|set(protected_v25_rows)
    SAFE_TEST,SAFE_LEDGER,_,_,_=apply_head12_policy(v30_test,test_ds,pb,ph,util,pair_state,"SAFE",test["user_id"].astype(str).tolist(),hard_lock)
    ATTACK_TEST,ATTACK_LEDGER,_,_,_=apply_head12_policy(v30_test,test_ds,pb,ph,util,pair_state,"ATTACK",test["user_id"].astype(str).tolist(),hard_lock)
    selected_test=ATTACK_TEST if DEV_WINNER=="ATTACK" else SAFE_TEST
    # Immutable-lock invariants.
    for r in hard_lock:
        assert np.array_equal(SAFE_TEST[r],v30_test[r]) and np.array_equal(ATTACK_TEST[r],v30_test[r])
    del full_bundle,train_ds,stage2,test_ds
else:
    selected_test=v30_test.copy()

SUB_ANCHOR=submission_from_scores(v30_test); SUB_SAFE=submission_from_scores(SAFE_TEST); SUB_ATTACK=submission_from_scores(ATTACK_TEST); SUB_PRIMARY=submission_from_scores(selected_test)
SUB_ANCHOR.to_csv(OUT_DIR/"submission_v34_anchor_v30_reconstructed.csv",index=False)
SUB_SAFE.to_csv(OUT_DIR/"submission_v34_DIAGNOSTIC_safe_head12.csv",index=False)
SUB_ATTACK.to_csv(OUT_DIR/"submission_v34_DIAGNOSTIC_attack_head12.csv",index=False)
SUB_PRIMARY.to_csv(OUT_DIR/"submission_v34_primary.csv",index=False)
if not FINAL_GATE: assert np.array_equal(SUB_PRIMARY[MODULES].to_numpy(),SUB_ANCHOR[MODULES].to_numpy())

print("FINAL_GATE",FINAL_GATE,"DEV_WINNER",DEV_WINNER,"SAFE test actions",len(SAFE_LEDGER),"ATTACK test actions",len(ATTACK_LEDGER))
del Xtr_full,Xte_full,model_v30,raw_v30; gc.collect()

# ---- cell 30 (code) ----
# Artifacts + slot discipline.
pd.DataFrame({"feature":V34_ACTION_FEATURE_NAMES}).to_csv(OUT_DIR/"v34_action_feature_names.csv",index=False)
V34_DEVELOPMENT.to_csv(OUT_DIR/"v34_development_worlds.csv",index=False)
V34_DEV_LEDGER.to_csv(OUT_DIR/"v34_development_interventions.csv",index=False)
V34_POOLED.to_csv(OUT_DIR/"v34_pooled_incremental.csv",index=False)
V34_GATES.to_csv(OUT_DIR/"v34_development_gate.csv",index=False)
V34_SEALED.to_csv(OUT_DIR/"v34_sealed_validation.csv",index=False)
SAFE_LEDGER.to_csv(OUT_DIR/"v34_test_interventions_safe.csv",index=False)
ATTACK_LEDGER.to_csv(OUT_DIR/"v34_test_interventions_attack.csv",index=False)
with open(OUT_DIR/"v34_v30_reconstruction_audit.json","w") as f: json.dump(V30_REPRO_AUDIT,f,indent=2)


def geom(name,s):
    return {"candidate":name,"top1_changed_users":int(np.sum(top5_indices(v30_test)[:,0]!=top5_indices(s)[:,0])),"top5_order_changed_users":int(np.sum(np.any(top5_indices(v30_test)!=top5_indices(s),axis=1))),"top5_membership_change_rate":top5_set_change_rate(v30_test,s)}
V34_TEST_GEOMETRY=pd.DataFrame([geom("SAFE",SAFE_TEST),geom("ATTACK",ATTACK_TEST)])
V34_TEST_GEOMETRY.to_csv(OUT_DIR/"v34_test_geometry.csv",index=False)

slot_plan=pd.DataFrame([
    {"file":"submission_v34_primary.csv","role":"PRIMARY_ONLY_IF_ALL_GATES_PASS","eligible_to_submit":bool(FINAL_GATE),"note":f"Frozen {DEV_WINNER} policy if all gates pass; otherwise exact V30."},
    {"file":"submission_v34_DIAGNOSTIC_safe_head12.csv","role":"DIAGNOSTIC_ONLY","eligible_to_submit":False,"note":"Never promote based on test geometry alone."},
    {"file":"submission_v34_DIAGNOSTIC_attack_head12.csv","role":"DIAGNOSTIC_ONLY","eligible_to_submit":False,"note":"Leader-scale attack surface only if preregistered ATTACK gate won."},
    {"file":"submission_v34_anchor_v30_reconstructed.csv","role":"ANCHOR_REFERENCE","eligible_to_submit":False,"note":"Exact public V30 reference."},
])
slot_plan.to_csv(OUT_DIR/"v34_slot_plan.csv",index=False)

summary={
    "version":"V34 V30-Locked Risk-Controlled Head12 Cross-Encoder Counterfactual",
    "public_v30":PUBLIC_V30_LB,"rank4_reference":PUBLIC_RANK4_LB,"leader_reference":PUBLIC_LEADER_LB,
    "cross_encoder":CE_MODEL_ID,"fixed_alpha":ALPHA,"fresh_worlds":FRESH_WORLDS,"policies":POLICIES,
    "development_winner":DEV_WINNER,"development_gate_passed":bool(DEV_GATE),"sealed_gate_passed":bool(SEALED_GATE),"v30_reproduction_ok":bool(V30_REPRO_OK),"final_gate_passed":bool(FINAL_GATE),
    "development":V34_DEVELOPMENT.to_dict("records"),"pooled":V34_POOLED.to_dict("records"),"sealed":V34_SEALED.to_dict("records"),
    "safe_test_actions":int(len(SAFE_LEDGER)),"attack_test_actions":int(len(ATTACK_LEDGER)),
    "submission_instruction":"PRIMARY IS ELIGIBLE after manual artifact audit." if FINAL_GATE else "NO NEW SUBMISSION: primary is exact V30 fail-close anchor.",
}
with open(OUT_DIR/"v34_summary.json","w") as f: json.dump(summary,f,indent=2)

assert set(V34_DEVELOPMENT.world)==set(FRESH_WORLDS)
assert set(V34_DEVELOPMENT.policy)=={"SAFE","ATTACK"}
assert len(V34_DEVELOPMENT)==6
assert SUB_PRIMARY.shape==sample.shape and SUB_SAFE.shape==sample.shape and SUB_ATTACK.shape==sample.shape
print(json.dumps(summary,indent=2))
print("\nV34 outputs written to",OUT_DIR)
print(slot_plan.to_string(index=False))

# ---- cell 31 (markdown) ----
# ### Interpretation contract
#
# - **Do not submit `submission_v34_primary.csv` if `final_gate_passed` is false.** In that case it is intentionally exact V30.
# - If **ATTACK** wins Q/R/S and survives T + chronology, it is the only leader-scale submission candidate.
# - If ATTACK fails but **SAFE** passes, SAFE is the only promoted candidate and is aimed primarily at moving above the current ~0.65974 boundary.
# - Diagnostic SAFE/ATTACK files are exported for audit, not for post-hoc leaderboard selection.
# - Do not tune policy thresholds, coverage caps, cross-encoder views, LightGBM capacity, or risk constants after seeing Q/R/S/T.
# - A positive cross-encoder logit difference is only evidence. The organizer-supervised Stage-2 and asymmetric harm model decide whether it is actionable.
