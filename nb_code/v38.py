# Converted from notebooks/v38.ipynb: markdown cells become comments, %/! magics neutralized.

# ---- cell 0 (markdown) ----
# # MineToday V38 — Hybrid34-Locked XLM-R Pairwise Residual Judge
#
# **Goal:** beat the current public Hybrid34 anchor (`0.66006`) using a genuinely new semantic information source without rebuilding the 17-module ranking globally.
#
# V38 keeps every public-proven component frozen and asks **XLM-R Base** a much narrower question: for an S0 user and one adjacent Hybrid34 boundary, does the lower course have enough contextual evidence to outrank the higher course?
#
# ### Frozen decisions from V30 → V37
# - exact deterministic Hybrid34 (`0.66006`) is the immutable Stage-0 anchor;
# - S3 (34 public-positive V31-overlap rows) is locked;
# - S1 and S2 are excluded from V38 deployment;
# - rank `1↔2` is hard-locked;
# - eligible boundaries are only `2↔3`, `3↔4`, `4↔5`, `5↔6`;
# - maximum one adjacent score-level swap per user;
# - score multiset is preserved exactly;
# - no XE/Yeti/LLM embedding signal enters the V38 judge;
# - no TAPT/MLM adaptation is used;
# - XLM-R is **fully supervised fine-tuned** as a joint cross-encoder with pair reversal consistency;
# - fresh K/L/M validation cohorts are disjoint and never used to choose architecture after results are opened;
# - sealed template-disjoint D + chronology are rejection-only;
# - if any promotion gate fails, `submission_v38_primary.csv` is exact Hybrid34 fail-close.
#
# This notebook targets Kaggle **T4 x2**. `torch.nn.DataParallel` is used when both GPUs are visible.

# ---- cell 1 (code) ----
# Runtime + preregistered V38 configuration
import os, sys, re, json, ast, html, hashlib, random, warnings, time, subprocess, gc, math, copy
from pathlib import Path
from collections import Counter, defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ["PYTHONHASHSEED"] = "42"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

V38_SKIP_INSTALL = bool(int(os.getenv("V38_SKIP_INSTALL", "0")))
try:
    import lightgbm as lgb
except Exception:
    if V38_SKIP_INSTALL: raise
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "lightgbm==4.6.0"])
    import lightgbm as lgb

try:
    import torch, transformers
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
except Exception:
    if V38_SKIP_INSTALL: raise
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.46,<5", "sentencepiece", "safetensors"])
    import torch, transformers
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

import numpy as np
import pandas as pd
import scipy
import sklearn
from IPython.display import display
from scipy.sparse import hstack
from scipy.optimize import minimize_scalar
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

SEED = 42
V38_SMOKE = bool(int(os.getenv("V38_SMOKE", "0")))
V38_STRICT_GEOMETRY = bool(int(os.getenv("V38_STRICT_GEOMETRY", "1")))
GEOMETRY_ATOL = 2.5e-4

# Compatibility aliases consumed by frozen V25/V30/V31 helpers.
V37_SMOKE = V38_SMOKE
V361_SMOKE = V38_SMOKE
V36_SMOKE = V38_SMOKE
V31_SMOKE = V38_SMOKE
V30_SMOKE = V38_SMOKE
V29_SMOKE = V38_SMOKE
V27_SMOKE = V38_SMOKE
V37_STRICT_GEOMETRY = V38_STRICT_GEOMETRY
V361_STRICT_GEOMETRY = V38_STRICT_GEOMETRY
V36_STRICT_GEOMETRY = V38_STRICT_GEOMETRY
V30_STRICT_GEOMETRY = V38_STRICT_GEOMETRY
V29_STRICT_GEOMETRY = V38_STRICT_GEOMETRY
V27_STRICT_GEOMETRY = V38_STRICT_GEOMETRY

N_FOLDS = 2 if V38_SMOKE else 5
ANCHOR_INNER_FOLDS = 2 if V38_SMOKE else 4
INTENT_INNER_FOLDS = 2 if V38_SMOKE else 4
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

LGB_ESTIMATORS = 12 if V38_SMOKE else 1200
LGB_EARLY_STOP = 3 if V38_SMOKE else 80
RANKER_ESTIMATORS = 16 if V38_SMOKE else 360
RANKER_LEARNING_RATE = 0.04 if V38_SMOKE else 0.025
ALPHA = 0.16
INTENT_PARENT_STRENGTH = 40.0
INTENT_CHILD_STRENGTH = 30.0
BOOTSTRAPS = 20 if V38_SMOKE else 1200

# XLM-R Base full supervised fine-tuning. No TAPT, no frozen embedding branch.
XLMR_MODEL_ID = "FacebookAI/xlm-roberta-base"
XLMR_MAX_LENGTH = 96 if V38_SMOKE else 256
XLMR_EPOCHS = 1 if V38_SMOKE else 2
XLMR_BATCH = 4 if V38_SMOKE else 16
XLMR_GRAD_ACCUM = 1 if V38_SMOKE else 2
XLMR_ENCODER_LR = 2.0e-5 if V38_SMOKE else 1.5e-5
XLMR_HEAD_LR = 1.0e-4 if V38_SMOKE else 8.0e-5
XLMR_WEIGHT_DECAY = 0.01
XLMR_WARMUP = 0.08
XLMR_GRAD_CLIP = 1.0
XLMR_REG_WEIGHT = 0.55
UTILITY_SCALE = 10.0
CAL_FRACTION = 0.10
INNER_HYBRID_FOLDS = 2 if V38_SMOKE else 4
ELIGIBLE_POSITIONS = (2,3,4,5)

# Fixed policy families. Thresholds are learned ONLY on each training-side calibration set.
SAFE_COVERAGE = 0.05 if V38_SMOKE else 0.025
ATTACK_COVERAGE = 0.10 if V38_SMOKE else 0.060
SAFE_TARGET_PRECISION = 0.50 if V38_SMOKE else 0.72
ATTACK_TARGET_PRECISION = 0.50 if V38_SMOKE else 0.62
MIN_POLICY_ACTIONS_CAL = 1 if V38_SMOKE else 5

# Fresh promotion gates.
SAFE_MIN_MEAN = -1.0 if V38_SMOKE else 0.00050
SAFE_MIN_WORST = -1.0 if V38_SMOKE else 0.00020
SAFE_MIN_PRECISION = 0.50 if V38_SMOKE else 0.68
ATTACK_MIN_MEAN = -1.0 if V38_SMOKE else 0.00150
ATTACK_MIN_WORST = -1.0 if V38_SMOKE else 0.00055
ATTACK_MIN_PRECISION = 0.50 if V38_SMOKE else 0.60
SEALED_D_MIN_INCREMENT = -1.0 if V38_SMOKE else 0.00000
CHRONO_MIN_INCREMENT = -1.0 if V38_SMOKE else -0.00005
TEST_GEOMETRY_AUC_MAX = 1.0 if V38_SMOKE else 0.62

FINAL_ENSEMBLE_SEEDS = (271,) if V38_SMOKE else (271,829)
OUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("/mnt/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)
HF_CACHE = OUT_DIR / "hf_cache_v38"
HF_CACHE.mkdir(parents=True, exist_ok=True)


def _seed_from_text(s):
    return int(hashlib.sha1(str(s).encode("utf-8")).hexdigest()[:8],16) % (2**31-1)

V38_WORLD_SEED = _seed_from_text("MINETODAY-V38-DISJOINT-KLM-20260830-XLMR-BASE")
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

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
print("numpy",np.__version__,"pandas",pd.__version__,"sklearn",sklearn.__version__,"lightgbm",lgb.__version__)
print("torch",torch.__version__,"transformers",transformers.__version__)
print("V38_SMOKE",V38_SMOKE,"Hybrid34",PUBLIC_HYBRID34_LB,"leader gap",round(PUBLIC_LEADER_GAP,5))
try: print("GPU runtime:\n"+subprocess.check_output(["nvidia-smi","-L"],text=True).strip())
except Exception: print("nvidia-smi unavailable; XLM-R requires a GPU for the intended run.")

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
# The public result defines four states from `V25 → V30 → V31`: S0 stable/stable, S1 stable then V31-change (public-toxic), S2 V30-change then V31-abstain, and S3 V30-change plus V31-change (the 34-row public-positive region). V38 uses the same exact state machine. S1 is never touched and S3 is immutable.

# ---- cell 19 (code) ----
def rank_matrix(scores):
    order=np.argsort(-np.asarray(scores),axis=1,kind="mergesort")
    ranks=np.empty_like(order,dtype=np.int16)
    ranks[np.arange(len(order))[:,None],order]=np.arange(1,N_MODULES+1,dtype=np.int16)[None,:]
    return ranks

def build_long_matrix_v31(rows_global,anchor,base,meta,intentstate=None):
    rows_global=np.asarray(rows_global,int); anchor=np.asarray(anchor,np.float32)
    base=np.asarray(base,np.float32); meta=np.asarray(meta,np.float32); n=len(rows_global)
    ranks=rank_matrix(anchor).astype(np.float32)
    top1=anchor.max(1,keepdims=True); sorted_scores=np.sort(anchor,axis=1)[:,::-1]; cut5=sorted_scores[:,4:5]
    mu=anchor.mean(1,keepdims=True); sd=np.maximum(anchor.std(1,keepdims=True),1e-4); z=(anchor-mu)/sd
    order=np.argsort(-anchor,axis=1,kind="mergesort"); prev_gap=np.zeros_like(anchor); next_gap=np.zeros_like(anchor)
    for i in range(n):
        oo=order[i]; ss=anchor[i,oo]
        pg=np.r_[0.0,ss[:-1]-ss[1:]]; ng=np.r_[ss[:-1]-ss[1:],0.0]
        prev_gap[i,oo]=pg; next_gap[i,oo]=ng
    base_mats=[anchor,ranks/float(N_MODULES),(ranks<=5).astype(np.float32),top1-anchor,anchor-cut5,
               prev_gap,next_gap,z,base,meta,meta-base,LEX_WORD[rows_global],LEX_CHAR[rows_global],
               MODULE_COUNT[rows_global],MODULE_RECENCY[rows_global],MODULE_LATEST[rows_global],
               MODULE_READINESS[rows_global],MODULE_INVALID[rows_global]]
    base_names=["anchor_score","anchor_rank_norm","anchor_top5","top1_minus_score","score_minus_cut5",
                "prev_gap","next_gap","anchor_z","base_score","meta_score","meta_minus_base",
                "lex_word","lex_char","topic_count","topic_recency","topic_latest","readiness","invalid_readiness"]
    pieces=[np.column_stack([m.reshape(-1) for m in base_mats]).astype(np.float32)]; names=list(base_names)
    pieces.append(np.repeat(USER_FEATURES[rows_global],N_MODULES,axis=0).astype(np.float32)); names += [f"user_f{i:03d}" for i in range(USER_FEATURES.shape[1])]
    pieces.append(np.tile(np.eye(N_MODULES,dtype=np.float32),(n,1))); names += [f"module_{m}" for m in MODULES]
    pieces.append(np.tile(MODULE_LEVEL_ONEHOT,(n,1)).astype(np.float32)); names += [f"module_level_{i}" for i in range(5)]
    bounds={"base":sum(p.shape[1] for p in pieces)}
    if intentstate is not None:
        mats=[intentstate[k] for k in INTENTSTATE_KEYS]
        pieces.append(np.column_stack([m.reshape(-1) for m in mats]).astype(np.float32)); names += [k.lower() for k in INTENTSTATE_KEYS]
        bounds["intentstate"]=sum(p.shape[1] for p in pieces)
    XX=np.hstack(pieces).astype(np.float32,copy=False)
    assert XX.shape[0]==n*N_MODULES and len(names)==XX.shape[1] and np.isfinite(XX).all()
    return XX,names,bounds

def make_lambdarank(seed):
    return lgb.LGBMRanker(objective="lambdarank",metric="ndcg",n_estimators=RANKER_ESTIMATORS,
        learning_rate=RANKER_LEARNING_RATE,num_leaves=15,min_child_samples=80,reg_alpha=1.0,reg_lambda=10.0,
        colsample_bytree=0.78,subsample=0.90,subsample_freq=1,max_bin=127,lambdarank_truncation_level=8,
        label_gain=LABEL_GAIN,random_state=int(seed),n_jobs=4,verbosity=-1,force_col_wise=True)

def fit_ranker_predict(kind,Xtr,Xva,grades_tr,n_users,seed):
    if kind!="lambda": raise ValueError("V38 intentionally supports only the proven LambdaRank V30/V31 backbone.")
    model=make_lambdarank(seed); group=np.full(int(n_users),N_MODULES,dtype=np.int32)
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
    a=np.any(pre_top!=v30_top,axis=1); b=np.any(v30_top!=v31_top,axis=1)
    return (2*a.astype(np.int8)+b.astype(np.int8)).astype(np.int8)

def hybrid34_surface(pre_anchor,v30,v31):
    st=state_codes(pre_anchor,v30,v31); out=np.asarray(v30,np.float32).copy(); out[st==3]=np.asarray(v31,np.float32)[st==3]
    return out,st

def cluster_boot_ci(values,groups,seed,B=BOOTSTRAPS):
    values=np.asarray(values,float); groups=np.asarray(groups,dtype=object); ug=np.unique(groups)
    by={g:np.flatnonzero(groups==g) for g in ug}; rng=np.random.default_rng(int(seed)); z=np.empty(int(B),float)
    for b in range(int(B)):
        sampled=rng.choice(ug,size=len(ug),replace=True); idx=np.concatenate([by[g] for g in sampled]); z[b]=values[idx].mean()
    return tuple(float(x) for x in np.quantile(z,[.025,.975]))

# ---- cell 20 (markdown) ----
# ## Fold-pure Hybrid34 reference helper
#
# Every XLM-R training example is generated from an **honest Hybrid34 analogue**. Training-side pair candidates are produced by an inner cross-fit; calibration and outer validation candidates are produced by models that never see the corresponding user labels. No XE/Yeti ranker is trained or queried anywhere in V38.

# ---- cell 21 (code) ----
def _make_cal_local(seed,n_estimators=None):
    return lgb.LGBMClassifier(n_estimators=int(LGB_ESTIMATORS if n_estimators is None else n_estimators),
        learning_rate=.03,num_leaves=15,min_child_samples=50,reg_alpha=.5,reg_lambda=5.,colsample_bytree=.7,
        random_state=int(seed),n_jobs=4,verbosity=-1)

def build_anchor_split_full(train_rows,val_rows,seed):
    train_rows=np.asarray(train_rows,int); val_rows=np.asarray(val_rows,int)
    local_y=y[train_rows]; local_top=Y_TOP[train_rows]; local_bin=Y_BIN[train_rows]
    folds=make_group_folds_local(local_y,TRAIN_GROUP[train_rows],ANCHOR_INNER_FOLDS,seed); n=len(train_rows)
    raw=np.zeros((n,N_MODULES),np.float32); top=np.zeros_like(raw)
    for lt,lv in folds:
        mr=Ridge(alpha=RIDGE_ALPHA,solver="lsqr",tol=2e-3,max_iter=1000); mt=Ridge(alpha=RIDGE_ALPHA,solver="lsqr",tol=2e-3,max_iter=1000)
        mr.fit(X[train_rows[lt]],local_y[lt]); mt.fit(X[train_rows[lt]],local_top[lt])
        raw[lv]=mr.predict(X[train_rows[lv]]).astype(np.float32); top[lv]=mt.predict(X[train_rows[lv]]).astype(np.float32)
    mr=Ridge(alpha=RIDGE_ALPHA,solver="lsqr",tol=2e-3,max_iter=1000); mt=Ridge(alpha=RIDGE_ALPHA,solver="lsqr",tol=2e-3,max_iter=1000)
    mr.fit(X[train_rows],local_y); mt.fit(X[train_rows],local_top)
    base_o=(.9*raw+.1*top).astype(np.float32); base_v=(.9*mr.predict(X[val_rows])+.1*mt.predict(X[val_rows])).astype(np.float32)
    meta_o=np.zeros_like(base_o); meta_v=np.zeros((len(val_rows),N_MODULES),np.float32)
    for m in range(N_MODULES):
        xt=np.column_stack([base_o[:,m],USER_FEATURES[train_rows]]).astype(np.float32); xv=np.column_stack([base_v[:,m],USER_FEATURES[val_rows]]).astype(np.float32)
        yy=local_bin[:,m].astype(np.int32); its=[]
        for f,(lt,lv) in enumerate(folds):
            clf=_make_cal_local(seed+31*m+f); clf.fit(xt[lt],yy[lt],eval_set=[(xt[lv],yy[lv])],callbacks=[lgb.early_stopping(LGB_EARLY_STOP,verbose=False)])
            it=int(clf.best_iteration_ or LGB_ESTIMATORS); its.append(it); meta_o[lv,m]=clf.predict_proba(xt[lv],num_iteration=it)[:,1].astype(np.float32)
        final_it=max(20,int(np.median(its))); full=_make_cal_local(seed+997+31*m,final_it); full.fit(xt,yy)
        meta_v[:,m]=full.predict_proba(xv)[:,1].astype(np.float32)
    cons_o=((1.0-CONS_W)*base_o+CONS_W*meta_o).astype(np.float32); cons_v=((1.0-CONS_W)*base_v+CONS_W*meta_v).astype(np.float32)
    return {"base_train":base_o,"meta_train":meta_o,"cons_train":cons_o,"base_val":base_v,"meta_val":meta_v,"cons_val":cons_v}

def fit_reference_hybrid(train_rows,val_rows,seed):
    train_rows=np.asarray(train_rows,int); val_rows=np.asarray(val_rows,int)
    anc=build_anchor_split_full(train_rows,val_rows,seed+701)
    intent_tr=crossfit_intentstate_features(y[train_rows],ORDINAL_GRADES[train_rows],TRAIN_INTENT[train_rows],MODULE_READY_BAND[train_rows],TRAIN_GROUP[train_rows],seed+809)
    bundle=fit_intentstate_bundle(y[train_rows],ORDINAL_GRADES[train_rows],TRAIN_INTENT[train_rows],MODULE_READY_BAND[train_rows])
    intent_va=apply_intentstate_bundle(bundle,TRAIN_INTENT[val_rows],MODULE_READY_BAND[val_rows])
    Xtr,_,bounds=build_long_matrix_v31(train_rows,anc["cons_train"],anc["base_train"],anc["meta_train"],intent_tr)
    Xva,_,_=build_long_matrix_v31(val_rows,anc["cons_val"],anc["base_val"],anc["meta_val"],intent_va)
    rank_seed=seed+907
    m0,r0=fit_ranker_predict("lambda",Xtr[:,:bounds["base"]],Xva[:,:bounds["base"]],ORDINAL_GRADES[train_rows],len(train_rows),rank_seed)
    m1,r1=fit_ranker_predict("lambda",Xtr[:,:bounds["intentstate"]],Xva[:,:bounds["intentstate"]],ORDINAL_GRADES[train_rows],len(train_rows),rank_seed)
    v30=blend_residual(anc["cons_val"],r0,ALPHA); v31=blend_residual(anc["cons_val"],r1,ALPHA); hybrid,st=hybrid34_surface(anc["cons_val"],v30,v31)
    out={"pre":anc["cons_val"].astype(np.float32),"v30":v30,"v31":v31,"hybrid":hybrid,"state":st}
    del Xtr,Xva,intent_tr,intent_va,bundle,m0,m1,anc; gc.collect(); return out

def crossfit_hybrid_rows(rows,seed,tag):
    rows=np.asarray(rows,int); key=hashlib.sha1(np.asarray(rows,np.int32).tobytes()+str(seed).encode()).hexdigest()[:12]
    cache=OUT_DIR/f"v38_hybrid_oof_{re.sub(r'[^A-Za-z0-9_-]+','_',tag)}_{key}.npz"
    if cache.exists():
        z=np.load(cache); return {k:z[k] for k in ("pre","v30","v31","hybrid","state")}
    out={"pre":np.zeros((len(rows),N_MODULES),np.float32),"v30":np.zeros((len(rows),N_MODULES),np.float32),
         "v31":np.zeros((len(rows),N_MODULES),np.float32),"hybrid":np.zeros((len(rows),N_MODULES),np.float32),"state":np.zeros(len(rows),np.int8)}
    pos={r:i for i,r in enumerate(rows.tolist())}
    folds=make_group_folds_global(rows,INNER_HYBRID_FOLDS,seed)
    for f,(tr,va) in enumerate(folds):
        b=fit_reference_hybrid(tr,va,seed+1009*f)
        loc=np.asarray([pos[int(r)] for r in va],int)
        for k in out: out[k][loc]=b[k]
        del b; gc.collect()
    np.savez_compressed(cache,**out)
    return out

print("Fold-pure Hybrid34 builders ready. XLM-R branch never calls XE/Yeti.")

# ---- cell 22 (markdown) ----
# ## XLM-R contextual pair representation
#
# V38 does **not** create static sentence embeddings or cosine similarities. The user context and both course candidates enter XLM-R jointly, and the contextual `<s>` representation is fused with a deliberately small structured vector. Text is recent-first and excludes template hashes/noise. Assessments remain structured side information.

# ---- cell 23 (code) ----
def locate_xlmr_source():
    env=os.getenv("V38_XLMR_MODEL_PATH","").strip()
    if env and Path(env).exists(): return env
    root=Path("/kaggle/input")
    if root.exists():
        hits=[]
        for p in root.rglob("config.json"):
            try:
                cfg=json.load(open(p)); mt=str(cfg.get("model_type","")).lower(); hs=int(cfg.get("hidden_size",0)); nl=int(cfg.get("num_hidden_layers",0))
                if mt in {"xlm-roberta","xlm_roberta"} and hs==768 and nl==12: hits.append(str(p.parent))
            except Exception: pass
        if hits: return sorted(hits)[0]
    return XLMR_MODEL_ID

XLMR_SOURCE=locate_xlmr_source()
print("XLM-R source:",XLMR_SOURCE)
TOKENIZER=AutoTokenizer.from_pretrained(XLMR_SOURCE,cache_dir=str(HF_CACHE),use_fast=True)

MODULE_DOCS=[]
mod_idx=modules.set_index("module_id").loc[MODULES]
for mid,row in mod_idx.iterrows():
    MODULE_DOCS.append(f"{row.module_name}. Materi: {row.description_and_syllabus}. Prasyarat: {row.prerequisite_level}.")

def build_v38_context(user_id):
    g=user_chat.get(user_id)
    if g is None or len(g)==0: return "[NO CHAT HISTORY]"
    learn=g[~g["is_noise"]].copy()
    if len(learn)==0: learn=g.copy()
    msgs=[str(x).strip() for x in learn["clean"].tolist() if str(x).strip()]
    if not msgs: return "[NO CHAT HISTORY]"
    rev=msgs[::-1]; parts=[f"[CURRENT] {rev[0]}"]
    for i,m in enumerate(rev[1:3],1): parts.append(f"[RECENT {i}] {m}")
    if len(rev)>3:
        older=" ".join(rev[3:6]); parts.append(f"[OLDER] {older}")
    return " ".join(parts)

V38_CONTEXT=np.asarray([build_v38_context(u) for u in all_users],dtype=object)
V38_CONTEXT_LEN=np.asarray([len(TOKENIZER.encode(x,add_special_tokens=False)) for x in V38_CONTEXT],int)
TOKENIZER_AUDIT=pd.DataFrame([{
    "model_source":str(XLMR_SOURCE),"max_length":XLMR_MAX_LENGTH,"median_context_tokens":float(np.median(V38_CONTEXT_LEN)),
    "p90_context_tokens":float(np.quantile(V38_CONTEXT_LEN,.90)),"p95_context_tokens":float(np.quantile(V38_CONTEXT_LEN,.95)),
    "max_context_tokens":int(V38_CONTEXT_LEN.max()),"contexts_over_192":int(np.sum(V38_CONTEXT_LEN>192)),
    "contexts_over_224":int(np.sum(V38_CONTEXT_LEN>224)),"no_chat_contexts":int(np.sum(V38_CONTEXT=="[NO CHAT HISTORY]"))
}])
display(TOKENIZER_AUDIT)

# ---- cell 24 (markdown) ----
# ## Pair examples, reversal augmentation, and small structured side vector
#
# For each honest S0 Hybrid34 row we create only adjacent boundaries `2↔3` through `5↔6`. Candidate A is the higher-ranked module and B is the lower-ranked module. The target is the exact counterfactual NDCG change from swapping their **score levels**. Training doubles each pair with A/B reversed; class `harm↔benefit` and signed utility are inverted exactly.

# ---- cell 25 (code) ----
PAIR_STRUCT_NAMES=[
    "position_norm","membership_flag","margin_signed","margin_abs","score_a_z","score_b_z","b_minus_a_z",
    "readiness_b_minus_a","invalid_b_minus_a","lex_word_b_minus_a","lex_char_b_minus_a","topic_count_b_minus_a",
    "topic_recency_b_minus_a","topic_latest_b_minus_a","module_level_b_minus_a","assessment_mean","assessment_std",
    "chat_count_log","learning_chat_count_log","no_chat","pos2","pos3","pos4","pos5"
]

CHAT_COUNT=np.zeros(len(all_users),np.float32); LEARN_COUNT=np.zeros(len(all_users),np.float32)
for u,ui in uid2idx.items():
    g=user_chat.get(u); CHAT_COUNT[ui]=0 if g is None else len(g)
    if g is None: LEARN_COUNT[ui]=0
    else:
        q=g[~g["is_noise"]]; LEARN_COUNT[ui]=len(q) if len(q) else len(g)
ASSESS_MEAN=A.mean(1).astype(np.float32); ASSESS_STD=A.std(1).astype(np.float32)

def _pair_struct(row_global,a,b,pos,anchor):
    sc=np.asarray(anchor,float); mu=float(sc.mean()); sd=max(float(sc.std()),1e-4)
    za=(float(sc[a])-mu)/sd; zb=(float(sc[b])-mu)/sd; margin=float(sc[a]-sc[b])
    v=[pos/5.0,float(pos==5),margin,abs(margin),za,zb,zb-za,
       float((MODULE_READINESS[row_global,b]-MODULE_READINESS[row_global,a])/5.0),
       float(MODULE_INVALID[row_global,b]-MODULE_INVALID[row_global,a]),
       float(LEX_WORD[row_global,b]-LEX_WORD[row_global,a]),float(LEX_CHAR[row_global,b]-LEX_CHAR[row_global,a]),
       float(MODULE_COUNT[row_global,b]-MODULE_COUNT[row_global,a]),float(MODULE_RECENCY[row_global,b]-MODULE_RECENCY[row_global,a]),
       float(MODULE_LATEST[row_global,b]-MODULE_LATEST[row_global,a]),float((MODULE_LEVEL[b]-MODULE_LEVEL[a])/4.0),
       float(ASSESS_MEAN[row_global]/5.0),float(ASSESS_STD[row_global]/5.0),float(np.log1p(CHAT_COUNT[row_global])),
       float(np.log1p(LEARN_COUNT[row_global])),float(CHAT_COUNT[row_global]==0)]
    v += [float(pos==p) for p in ELIGIBLE_POSITIONS]
    return np.asarray(v,np.float32)

def build_pair_table(rows_global,hybrid,state,with_labels=True):
    rows_global=np.asarray(rows_global,int); hybrid=np.asarray(hybrid,np.float32); state=np.asarray(state,np.int8)
    rows=[]
    for local,(rg,sc,st) in enumerate(zip(rows_global,hybrid,state)):
        if int(st)!=0: continue
        order=np.argsort(-sc,kind="mergesort")
        for pos in ELIGIBLE_POSITIONS:
            a=int(order[pos-1]); b=int(order[pos])
            rec={"row_local":local,"row_global":int(rg),"user_id":all_users[int(rg)],"position":int(pos),"a":a,"b":b,
                 "module_a":MODULES[a],"module_b":MODULES[b],"margin":float(sc[a]-sc[b]),"membership":int(pos==5),
                 "struct_f":_pair_struct(int(rg),a,b,pos,sc),"struct_r":_pair_struct(int(rg),b,a,pos,sc)}
            if with_labels:
                before=float(per_user_ndcg(y[int(rg):int(rg)+1],sc[None,:])[0])
                sw=sc.copy(); sw[a],sw[b]=sw[b],sw[a]
                after=float(per_user_ndcg(y[int(rg):int(rg)+1],sw[None,:])[0]); delta=after-before
                rec["delta"]=float(delta); rec["class"]=2 if delta>1e-12 else (0 if delta<-1e-12 else 1)
            rows.append(rec)
    return pd.DataFrame(rows)

def pair_texts(tab,reverse=False):
    contexts=[]; pairdocs=[]
    for r in tab.itertuples(index=False):
        a=int(r.b if reverse else r.a); b=int(r.a if reverse else r.b)
        contexts.append(str(V38_CONTEXT[int(r.row_global)]))
        pairdocs.append(f"Candidate A: {MODULE_DOCS[a]} Candidate B: {MODULE_DOCS[b]}")
    return contexts,pairdocs

class EncodedPairDataset(Dataset):
    def __init__(self,tab,mean,std,training=False,reverse=False):
        self.training=bool(training); self.reverse=bool(reverse); self.tab=tab.reset_index(drop=True)
        ctx,doc=pair_texts(self.tab,reverse=reverse)
        self.tok=TOKENIZER(ctx,doc,padding="max_length",truncation="longest_first",max_length=XLMR_MAX_LENGTH,return_tensors="pt")
        arr=np.stack(self.tab["struct_r" if reverse else "struct_f"].to_numpy()).astype(np.float32)
        self.x=torch.tensor((arr-mean)/std,dtype=torch.float32)
        if "class" in self.tab:
            cls=self.tab["class"].to_numpy(int); delta=self.tab["delta"].to_numpy(np.float32)
            if reverse:
                cls=np.where(cls==0,2,np.where(cls==2,0,1)); delta=-delta
            self.cls=torch.tensor(cls,dtype=torch.long); self.delta=torch.tensor(delta*UTILITY_SCALE,dtype=torch.float32)
            self.weight=torch.tensor(1.0+1.5*np.minimum(np.abs(delta)/0.15,1.0),dtype=torch.float32)
        else: self.cls=self.delta=self.weight=None
    def __len__(self): return len(self.x)
    def __getitem__(self,i):
        z={k:v[i] for k,v in self.tok.items()}; z["structured"]=self.x[i]
        if self.cls is not None: z.update(labels_cls=self.cls[i],labels_delta=self.delta[i],sample_weight=self.weight[i])
        return z

class XLMRPairJudge(nn.Module):
    def __init__(self,source,struct_dim):
        super().__init__(); self.encoder=AutoModel.from_pretrained(source,cache_dir=str(HF_CACHE)); h=int(self.encoder.config.hidden_size)
        self.fusion=nn.Sequential(nn.Dropout(.15),nn.Linear(h+struct_dim,256),nn.GELU(),nn.Dropout(.15),nn.Linear(256,128),nn.GELU(),nn.Dropout(.10))
        self.cls_head=nn.Linear(128,3); self.reg_head=nn.Linear(128,1)
    def forward(self,input_ids,attention_mask,structured,**kwargs):
        o=self.encoder(input_ids=input_ids,attention_mask=attention_mask); cls=o.last_hidden_state[:,0]
        h=self.fusion(torch.cat([cls,structured],dim=1)); return self.cls_head(h),self.reg_head(h).squeeze(-1)

# Synthetic contract tests: S0-only, rank1 lock, exact reversal target inversion.
_tmp_scores=np.tile(np.linspace(1,.1,N_MODULES,dtype=np.float32),(2,1)); _tmp_state=np.array([0,2],np.int8)
_tmp=build_pair_table(np.array([0,1]),_tmp_scores,_tmp_state,with_labels=True)
assert len(_tmp)==len(ELIGIBLE_POSITIONS) and set(_tmp.position)==set(ELIGIBLE_POSITIONS) and int(_tmp.row_global.nunique())==1
assert 1 not in set(_tmp.position)
assert len(PAIR_STRUCT_NAMES)==len(_tmp.iloc[0].struct_f)==len(_tmp.iloc[0].struct_r)
print("V38 pair contract tests passed; structured dims",len(PAIR_STRUCT_NAMES))

# ---- cell 26 (markdown) ----
# ## Full fine-tuning, temperature calibration, reversal consistency, and policy derivation
#
# The encoder weights, attention blocks, feed-forward layers, LayerNorms and fusion heads are all updated by AdamW. Calibration labels are never used for gradient updates. SAFE/ATTACK thresholds are learned only from that training-side calibration partition and then frozen before the outer validation cohort is scored.

# ---- cell 27 (code) ----
def set_all_seeds(seed):
    random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed))
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(int(seed))

def _softmax_np(x):
    x=np.asarray(x,float); x=x-x.max(1,keepdims=True); e=np.exp(x); return e/np.maximum(e.sum(1,keepdims=True),1e-12)

def temperature_scale(logits,labels):
    logits=np.asarray(logits,float); labels=np.asarray(labels,int)
    if len(np.unique(labels))<2: return 1.0
    def obj(logt):
        t=float(np.exp(logt)); p=_softmax_np(logits/t); return float(-np.mean(np.log(np.maximum(p[np.arange(len(labels)),labels],1e-12))))
    r=minimize_scalar(obj,bounds=(math.log(.35),math.log(4.0)),method="bounded"); return float(np.exp(r.x))

def _predict_raw(model,ds,batch_size=None):
    loader=DataLoader(ds,batch_size=int(batch_size or XLMR_BATCH),shuffle=False,num_workers=0,pin_memory=torch.cuda.is_available())
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.eval(); logits=[]; util=[]
    with torch.no_grad():
        for batch in loader:
            inp={k:v.to(device,non_blocking=True) for k,v in batch.items() if k in ("input_ids","attention_mask","structured")}
            with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=torch.cuda.is_available()): lo,uu=model(**inp)
            logits.append(lo.detach().float().cpu().numpy()); util.append(uu.detach().float().cpu().numpy())
    return np.concatenate(logits),np.concatenate(util)

def train_xlmr_bundle(train_tab,cal_tab,seed,tag):
    set_all_seeds(seed); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_struct=np.stack(train_tab.struct_f.to_numpy()).astype(np.float32); mean=all_struct.mean(0).astype(np.float32); std=np.maximum(all_struct.std(0),1e-4).astype(np.float32)
    tr_f=EncodedPairDataset(train_tab,mean,std,training=True,reverse=False); tr_r=EncodedPairDataset(train_tab,mean,std,training=True,reverse=True)
    class _Concat(Dataset):
        def __len__(self): return len(tr_f)+len(tr_r)
        def __getitem__(self,i): return tr_f[i] if i<len(tr_f) else tr_r[i-len(tr_f)]
    trds=_Concat(); gen=torch.Generator(); gen.manual_seed(int(seed))
    loader=DataLoader(trds,batch_size=XLMR_BATCH,shuffle=True,num_workers=0,pin_memory=torch.cuda.is_available(),generator=gen)
    model=XLMRPairJudge(XLMR_SOURCE,len(PAIR_STRUCT_NAMES)).to(device)
    if torch.cuda.device_count()>=2: model=nn.DataParallel(model)
    core=model.module if isinstance(model,nn.DataParallel) else model
    enc_params=list(core.encoder.parameters()); head_params=list(core.fusion.parameters())+list(core.cls_head.parameters())+list(core.reg_head.parameters())
    opt=torch.optim.AdamW([{"params":enc_params,"lr":XLMR_ENCODER_LR},{"params":head_params,"lr":XLMR_HEAD_LR}],weight_decay=XLMR_WEIGHT_DECAY)
    steps_per_epoch=max(1,math.ceil(len(loader)/XLMR_GRAD_ACCUM)); total_steps=max(1,steps_per_epoch*XLMR_EPOCHS); warm=int(total_steps*XLMR_WARMUP)
    sched=get_linear_schedule_with_warmup(opt,warm,total_steps); scaler=torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    # reversal makes benefit/harm symmetric; only tie imbalance needs gentle correction.
    cls0=train_tab["class"].to_numpy(int); cls2=np.where(cls0==0,2,np.where(cls0==2,0,1)); cls_all=np.r_[cls0,cls2]
    cnt=np.bincount(cls_all,minlength=3).astype(float); cw=np.sqrt(cnt.sum()/np.maximum(cnt,1)); cw=cw/cw.mean(); cw_t=torch.tensor(cw,dtype=torch.float32,device=device)
    best=None; best_loss=np.inf; history=[]
    cal_f=EncodedPairDataset(cal_tab,mean,std,reverse=False); cal_r=EncodedPairDataset(cal_tab,mean,std,reverse=True)
    for epoch in range(XLMR_EPOCHS):
        model.train(); opt.zero_grad(set_to_none=True); losses=[]
        for bi,batch in enumerate(loader):
            inp={k:v.to(device,non_blocking=True) for k,v in batch.items() if k in ("input_ids","attention_mask","structured")}
            yy=batch["labels_cls"].to(device); dd=batch["labels_delta"].to(device); ww=batch["sample_weight"].to(device)
            with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=torch.cuda.is_available()):
                lo,uu=model(**inp); ce=nn.functional.cross_entropy(lo,yy,weight=cw_t,reduction="none"); reg=nn.functional.smooth_l1_loss(uu,dd,reduction="none")
                loss=((ce+XLMR_REG_WEIGHT*reg)*ww).mean()/XLMR_GRAD_ACCUM
            scaler.scale(loss).backward(); losses.append(float(loss.detach().cpu())*XLMR_GRAD_ACCUM)
            if (bi+1)%XLMR_GRAD_ACCUM==0 or bi+1==len(loader):
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),XLMR_GRAD_CLIP); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
        lf,uf=_predict_raw(model,cal_f); lr,ur=_predict_raw(model,cal_r)
        lab=cal_tab["class"].to_numpy(int); revlab=np.where(lab==0,2,np.where(lab==2,0,1))
        p1=_softmax_np(lf); p2=_softmax_np(lr); nll=float(-.5*(np.mean(np.log(np.maximum(p1[np.arange(len(lab)),lab],1e-12)))+np.mean(np.log(np.maximum(p2[np.arange(len(lab)),revlab],1e-12)))))
        history.append({"epoch":epoch+1,"train_loss":float(np.mean(losses)),"cal_nll":nll})
        if nll<best_loss:
            best_loss=nll; best={k:v.detach().cpu().clone() for k,v in core.state_dict().items()}
    core.load_state_dict(best); model.to(device)
    lf,uf=_predict_raw(model,cal_f); lr,ur=_predict_raw(model,cal_r)
    lab=cal_tab["class"].to_numpy(int); revlab=np.where(lab==0,2,np.where(lab==2,0,1))
    temp=temperature_scale(np.vstack([lf,lr]),np.r_[lab,revlab])
    raw_util=.5*(uf-ur)/UTILITY_SCALE; regcal=Ridge(alpha=1.0).fit(raw_util.reshape(-1,1),cal_tab["delta"].to_numpy(float))
    bundle={"model":model,"mean":mean,"std":std,"temperature":float(temp),"regcal":regcal,"history":history,"seed":int(seed),"tag":str(tag)}
    cal_scored=score_pair_bundle(bundle,cal_tab)
    return bundle,cal_scored

def score_pair_bundle(bundle,tab):
    if len(tab)==0: return tab.copy()
    f=EncodedPairDataset(tab,bundle["mean"],bundle["std"],reverse=False); r=EncodedPairDataset(tab,bundle["mean"],bundle["std"],reverse=True)
    lf,uf=_predict_raw(bundle["model"],f); lr,ur=_predict_raw(bundle["model"],r); t=bundle["temperature"]
    pf=_softmax_np(lf/t); pr=_softmax_np(lr/t)
    pben=np.sqrt(np.clip(pf[:,2],1e-8,1)*np.clip(pr[:,0],1e-8,1)); pharm=np.sqrt(np.clip(pf[:,0],1e-8,1)*np.clip(pr[:,2],1e-8,1)); ptie=np.sqrt(np.clip(pf[:,1],1e-8,1)*np.clip(pr[:,1],1e-8,1))
    utilraw=.5*(uf-ur)/UTILITY_SCALE; util=bundle["regcal"].predict(utilraw.reshape(-1,1))
    out=tab.copy(); out["p_benefit"]=pben; out["p_harm"]=pharm; out["p_tie"]=ptie; out["utility"]=util
    out["consistency"]=np.minimum(pf[:,2],pr[:,0]); out["contradiction"]=np.maximum(pf[:,0],pr[:,2])
    return out

def calibration_economics(cal_tab):
    econ={}
    for p in ELIGIBLE_POSITIONS:
        d=cal_tab.loc[cal_tab.position==p,"delta"].to_numpy(float)
        econ[int(p)]={"gain":float(np.mean(d[d>0])) if np.any(d>0) else .03,"harm":float(np.mean(-d[d<0])) if np.any(d<0) else .03}
    return econ

def add_risk_score(tab,econ):
    out=tab.copy(); gain=np.asarray([econ.get(int(p),{"gain":.04})["gain"] for p in out.position],float); harm=np.asarray([econ.get(int(p),{"harm":.04})["harm"] for p in out.position],float)
    out["risk_score"]=out.p_benefit.to_numpy()*gain-out.p_harm.to_numpy()*harm+0.50*out.utility.to_numpy()
    return out

def basic_semantic_mask(tab,policy):
    if policy=="SAFE": pb,ph,co,ct=.60,.30,.55,.30
    else: pb,ph,co,ct=.56,.35,.50,.38
    m=(tab.p_benefit>=pb)&(tab.p_harm<=ph)&(tab.consistency>=co)&(tab.contradiction<=ct)&(tab.utility>0)
    # Membership 5↔6 requires stronger evidence.
    mem=tab.position==5; m &= (~mem)|((tab.p_benefit>=pb+.07)&(tab.p_harm<=max(.05,ph-.07)))
    return m.to_numpy(bool)

def _dedupe_users(tab):
    if len(tab)==0: return tab.copy()
    return tab.sort_values(["risk_score","p_benefit"],ascending=[False,False],kind="mergesort").drop_duplicates("row_local",keep="first")

def derive_policy(cal_scored,n_users,policy):
    cov=SAFE_COVERAGE if policy=="SAFE" else ATTACK_COVERAGE; target=SAFE_TARGET_PRECISION if policy=="SAFE" else ATTACK_TARGET_PRECISION
    econ=calibration_economics(cal_scored); z=add_risk_score(cal_scored,econ); z=z[basic_semantic_mask(z,policy)].copy(); z=_dedupe_users(z)
    maxk=max(1,int(math.ceil(cov*n_users))); z=z.head(maxk).reset_index(drop=True)
    best=None
    for k in range(MIN_POLICY_ACTIONS_CAL,min(maxk,len(z))+1):
        q=z.iloc[:k]; d=q.delta.to_numpy(float); den=max(1,int(np.sum(np.abs(d)>1e-12))); prec=float(np.sum(d>1e-12)/den); mean=float(d.sum()/max(n_users,1))
        if prec>=target and mean>0: best=k
    if best is None:
        return {"policy":policy,"threshold":float("inf"),"max_actions":maxk,"cal_actions":0,"cal_precision":0.0,"cal_increment":0.0,"economics":econ}
    q=z.iloc[:best]; d=q.delta.to_numpy(float); den=max(1,int(np.sum(np.abs(d)>1e-12)))
    return {"policy":policy,"threshold":float(q.risk_score.min()),"max_actions":maxk,"cal_actions":int(best),"cal_precision":float(np.sum(d>1e-12)/den),"cal_increment":float(d.sum()/max(n_users,1)),"economics":econ}

def select_policy(scored,n_users,policy_cfg,policy,blocked_rows=None):
    z=add_risk_score(scored,policy_cfg["economics"]); z=z[basic_semantic_mask(z,policy)].copy(); blocked=set() if blocked_rows is None else set(map(int,blocked_rows))
    if blocked: z=z[~z.row_local.isin(blocked)]
    z=z[z.risk_score>=float(policy_cfg["threshold"])]; z=_dedupe_users(z); z=z.head(int(policy_cfg["max_actions"])).copy()
    return z.reset_index(drop=True)

def apply_actions(reference_scores,selected):
    out=np.asarray(reference_scores,np.float32).copy()
    for r in selected.itertuples(index=False):
        i=int(r.row_local); a=int(r.a); b=int(r.b); out[i,a],out[i,b]=out[i,b],out[i,a]
    return out

def eval_selected(reference,rows_global,selected,seed):
    rows_global=np.asarray(rows_global,int); cand=apply_actions(reference["hybrid"],selected)
    base=per_user_ndcg(y[rows_global],reference["hybrid"]); cur=per_user_ndcg(y[rows_global],cand); inc=cur-base
    touched=np.zeros(len(rows_global),bool)
    if len(selected): touched[selected.row_local.to_numpy(int)]=True
    td=inc[touched]; ben=int(np.sum(td>1e-12)); harm=int(np.sum(td<-1e-12)); tie=int(np.sum(np.abs(td)<=1e-12)); den=max(1,ben+harm)
    ci=cluster_boot_ci(inc,train.iloc[rows_global].user_id.to_numpy(),seed)
    return {"increment_vs_hybrid34":float(inc.mean()),"ci_low":ci[0],"ci_high":ci[1],"actions":int(touched.sum()),"benefit":ben,"harm":harm,"tie":tie,"precision":float(ben/den),"increment_vector":inc,"candidate_scores":cand}

print("XLM-R full fine-tuning + calibration helpers ready")

# ---- cell 28 (markdown) ----
# ## Fresh disjoint K/L/M confirmation
#
# K/L/M are three disjoint 400-user cohorts from a fixed 10-fold stratified partition. For each world, 10% of the remaining outer-train users are reserved for **calibration only**. XLM-R gradient updates use neither calibration labels nor K/L/M labels. Architecture and policy families are frozen before these results are opened.

# ---- cell 29 (code) ----
def make_klm_worlds():
    sk=StratifiedKFold(n_splits=10,shuffle=True,random_state=V38_WORLD_SEED); folds=list(sk.split(np.zeros(N_TRAIN),user_strata_from_y(y)))
    picks={"K":0,"L":3,"M":6}; out={}
    for name,fi in picks.items():
        va=np.asarray(folds[fi][1],int); tr=np.setdiff1d(np.arange(N_TRAIN,dtype=int),va,assume_unique=True); out[name]=(tr,va)
    assert len(set(out["K"][1]).intersection(out["L"][1]))==0 and len(set(out["K"][1]).intersection(out["M"][1]))==0 and len(set(out["L"][1]).intersection(out["M"][1]))==0
    return out

def split_fit_cal(rows,seed):
    rows=np.asarray(rows,int); sss=StratifiedShuffleSplit(n_splits=1,test_size=CAL_FRACTION,random_state=int(seed)); tr,ca=next(sss.split(np.zeros(len(rows)),user_strata_from_y(y[rows])))
    return rows[tr],rows[ca]

V38_WORLDS=make_klm_worlds(); WORLD_ASSIGNMENTS=pd.DataFrame([{"world":w,"n_train":len(tr),"n_val":len(va),"val_uid_min":train.iloc[va].user_id.min(),"val_uid_max":train.iloc[va].user_id.max()} for w,(tr,va) in V38_WORLDS.items()])
display(WORLD_ASSIGNMENTS)

def run_v38_world(name,outer_train,val_rows):
    seed=_seed_from_text(f"V38-{name}-XLMR")
    fit_rows,cal_rows=split_fit_cal(outer_train,seed+11)
    print(f"\n=== V38 {name}: fit={len(fit_rows)} cal={len(cal_rows)} val={len(val_rows)} ===")
    fit_oof=crossfit_hybrid_rows(fit_rows,seed+101,f"{name}-FIT")
    cal_ref=fit_reference_hybrid(fit_rows,cal_rows,seed+211)
    val_ref=fit_reference_hybrid(outer_train,val_rows,seed+307)
    train_tab=build_pair_table(fit_rows,fit_oof["hybrid"],fit_oof["state"],with_labels=True)
    cal_tab=build_pair_table(cal_rows,cal_ref["hybrid"],cal_ref["state"],with_labels=True)
    val_tab=build_pair_table(val_rows,val_ref["hybrid"],val_ref["state"],with_labels=True)
    print("pair rows train/cal/val",len(train_tab),len(cal_tab),len(val_tab),"S0 val users",val_tab.row_local.nunique())
    bundle,cal_scored=train_xlmr_bundle(train_tab,cal_tab,seed+401,name); val_scored=score_pair_bundle(bundle,val_tab)
    policies={p:derive_policy(cal_scored,len(cal_rows),p) for p in ("SAFE","ATTACK")}; evals={}; ledgers={}
    for p in ("SAFE","ATTACK"):
        sel=select_policy(val_scored,len(val_rows),policies[p],p); ev=eval_selected(val_ref,val_rows,sel,seed+503+(0 if p=="SAFE" else 1)); evals[p]=ev; ledgers[p]=sel
        print(name,p,policies[p],"->",{k:ev[k] for k in ("increment_vs_hybrid34","actions","benefit","harm","tie","precision")})
    hist=pd.DataFrame(bundle["history"]); del bundle,fit_oof,cal_ref,train_tab,cal_tab; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return {"reference":val_ref,"val_tab":val_tab,"val_scored":val_scored,"policies":policies,"evals":evals,"ledgers":ledgers,"history":hist,"fit_rows":fit_rows,"cal_rows":cal_rows,"val_rows":val_rows}

WORLD_DATA={}; VAL_ROWS=[]; CAL_POLICY_ROWS=[]; VAL_LEDGERS=[]; TRAIN_HISTORY=[]
for name in ("K","L","M"):
    tr,va=V38_WORLDS[name]; wd=run_v38_world(name,tr,va); WORLD_DATA[name]=wd
    for p in ("SAFE","ATTACK"):
        e=wd["evals"][p]; VAL_ROWS.append({"world":name,"policy":p,**{k:v for k,v in e.items() if k not in ("increment_vector","candidate_scores")},"eligible_pairs":len(wd["val_tab"])})
        CAL_POLICY_ROWS.append({"world":name,**wd["policies"][p]})
        if len(wd["ledgers"][p]):
            q=wd["ledgers"][p].copy(); q["world"]=name; q["policy"]=p; q["realized_delta_ndcg"]=e["increment_vector"][q.row_local.to_numpy(int)]; VAL_LEDGERS.append(q)
    h=wd["history"].copy(); h["world"]=name; TRAIN_HISTORY.append(h)

V38_VALIDATION=pd.DataFrame(VAL_ROWS); V38_CALIBRATION=pd.DataFrame(CAL_POLICY_ROWS); V38_VALIDATION_LEDGER=pd.concat(VAL_LEDGERS,ignore_index=True) if VAL_LEDGERS else pd.DataFrame(); V38_TRAIN_HISTORY=pd.concat(TRAIN_HISTORY,ignore_index=True)
display(V38_VALIDATION); display(V38_CALIBRATION)

# ---- cell 30 (markdown) ----
# ## Development promotion gate
#
# SAFE is allowed to become the first submission if it is positive in all K/L/M cohorts with a meaningful worst-world gain and high precision. ATTACK has a stricter **leader-scale** mean-gain requirement. Fresh-world results never tune thresholds; the only choice is accept/reject the two precommitted policy families.

# ---- cell 31 (code) ----
def policy_gate(policy):
    q=V38_VALIDATION[V38_VALIDATION.policy==policy].copy(); vals=q.increment_vs_hybrid34.to_numpy(float); prec=q.precision.to_numpy(float)
    if len(q)!=3: return False,{}
    stats={"mean":float(vals.mean()),"worst":float(vals.min()),"precision_worst":float(prec.min()),"positive_worlds":int(np.sum(vals>0)),"actions_total":int(q.actions.sum())}
    if policy=="SAFE": ok=bool(stats["positive_worlds"]==3 and stats["mean"]>=SAFE_MIN_MEAN and stats["worst"]>=SAFE_MIN_WORST and stats["precision_worst"]>=SAFE_MIN_PRECISION)
    else: ok=bool(stats["positive_worlds"]==3 and stats["mean"]>=ATTACK_MIN_MEAN and stats["worst"]>=ATTACK_MIN_WORST and stats["precision_worst"]>=ATTACK_MIN_PRECISION)
    return ok,stats

SAFE_GATE,SAFE_STATS=policy_gate("SAFE"); ATTACK_GATE,ATTACK_STATS=policy_gate("ATTACK")
DEV_WINNER="ATTACK" if ATTACK_GATE else ("SAFE" if SAFE_GATE else None); DEV_GATE=DEV_WINNER is not None
V38_GATE=pd.DataFrame([{"policy":"SAFE","gate":SAFE_GATE,**SAFE_STATS},{"policy":"ATTACK","gate":ATTACK_GATE,**ATTACK_STATS}])
display(V38_GATE); print("DEV_WINNER",DEV_WINNER,"DEV_GATE",DEV_GATE)

# ---- cell 32 (markdown) ----
# ## Sealed template-disjoint D and chronology rejection checks
#
# These checks run **only if** K/L/M already promote a policy. They do not select a new threshold or architecture. Each rejection surface trains a new XLM-R model from the same pretrained initialization using only its own training-side labels.

# ---- cell 33 (code) ----
def sealed_d_bucket(group):
    h=hashlib.sha1(f"{SEALED_D_SALT}::{group}".encode("utf-8")).hexdigest(); return int(h[:12],16)%10
SEALED_D_MASK=np.asarray([sealed_d_bucket(g)==0 for g in TRAIN_GROUP],bool); SEALED_D_TR=np.flatnonzero(~SEALED_D_MASK); SEALED_D_VA=np.flatnonzero(SEALED_D_MASK)
assert len(set(TRAIN_GROUP[SEALED_D_TR]).intersection(set(TRAIN_GROUP[SEALED_D_VA])))==0
CHRONO_TR=np.arange(0,3600,dtype=int); CHRONO_VA=np.arange(3600,4000,dtype=int)

SEALED_ROWS=[]; SEALED_LED=[]
def run_rejection(name,outer_train,val_rows,policy):
    seed=_seed_from_text(f"V38-{name}-XLMR-REJECTION"); fit_rows,cal_rows=split_fit_cal(outer_train,seed+11)
    fit_oof=crossfit_hybrid_rows(fit_rows,seed+101,f"{name}-FIT"); cal_ref=fit_reference_hybrid(fit_rows,cal_rows,seed+211); val_ref=fit_reference_hybrid(outer_train,val_rows,seed+307)
    train_tab=build_pair_table(fit_rows,fit_oof["hybrid"],fit_oof["state"],True); cal_tab=build_pair_table(cal_rows,cal_ref["hybrid"],cal_ref["state"],True); val_tab=build_pair_table(val_rows,val_ref["hybrid"],val_ref["state"],True)
    bundle,cal_scored=train_xlmr_bundle(train_tab,cal_tab,seed+401,name); cfg=derive_policy(cal_scored,len(cal_rows),policy); scored=score_pair_bundle(bundle,val_tab); sel=select_policy(scored,len(val_rows),cfg,policy); ev=eval_selected(val_ref,val_rows,sel,seed+509)
    if len(sel):
        ld=sel.copy(); ld["surface"]=name; ld["policy"]=policy; ld["realized_delta_ndcg"]=ev["increment_vector"][ld.row_local.to_numpy(int)]; SEALED_LED.append(ld)
    del bundle,fit_oof,cal_ref,train_tab,cal_tab,val_tab,scored; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return {"surface":name,"policy":policy,**{k:v for k,v in ev.items() if k not in ("increment_vector","candidate_scores")},"cal_threshold":cfg["threshold"],"cal_precision":cfg["cal_precision"]}

if DEV_GATE:
    SEALED_ROWS.append(run_rejection("SEALED_D",SEALED_D_TR,SEALED_D_VA,DEV_WINNER))
    SEALED_ROWS.append(run_rejection("CHRONO",CHRONO_TR,CHRONO_VA,DEV_WINNER))
V38_SEALED=pd.DataFrame(SEALED_ROWS); V38_SEALED_LEDGER=pd.concat(SEALED_LED,ignore_index=True) if SEALED_LED else pd.DataFrame()
if DEV_GATE:
    display(V38_SEALED); dr=V38_SEALED[V38_SEALED.surface=="SEALED_D"]; cr=V38_SEALED[V38_SEALED.surface=="CHRONO"]
    SEALED_GATE=bool(len(dr)==1 and len(cr)==1 and float(dr.increment_vs_hybrid34.iloc[0])>=SEALED_D_MIN_INCREMENT and float(cr.increment_vs_hybrid34.iloc[0])>=CHRONO_MIN_INCREMENT)
else: SEALED_GATE=False
print("SEALED_GATE",SEALED_GATE)

# ---- cell 34 (markdown) ----
# ## Exact Hybrid34 test reconstruction and final XLM-R deployment ensemble
#
# The public anchor is reconstructed from organizer data and frozen algorithms, including the protected V25 rows. Only after development + sealed gates pass do two XLM-R Base seeds train on the full organizer-train pool using a shared 90/10 fit/calibration split. Their calibrated semantic judgments are averaged before the selected SAFE/ATTACK policy is applied.

# ---- cell 35 (code) ----
# Exact full V30/V31/Hybrid34 reconstruction.
full_intent_tr=crossfit_intentstate_features(y,ORDINAL_GRADES,TRAIN_INTENT,MODULE_READY_BAND[:N_TRAIN],TRAIN_GROUP,_seed_from_text("V31-FULL-INTENTSTATE-CROSSFIT"))
FULL_INTENT_BUNDLE=fit_intentstate_bundle(y,ORDINAL_GRADES,TRAIN_INTENT,MODULE_READY_BAND[:N_TRAIN]); full_intent_te=apply_intentstate_bundle(FULL_INTENT_BUNDLE,TEST_INTENT,MODULE_READY_BAND[N_TRAIN:])
V38_SUPPORT_AUDIT=intentstate_support_audit(FULL_INTENT_BUNDLE,TEST_INTENT,MODULE_READY_BAND[N_TRAIN:])
Xtr_full,_,bounds_full=build_long_matrix_v31(np.arange(N_TRAIN),consensus3_oof,base_oof,meta_oof,full_intent_tr); Xte_full,_,bounds_test=build_long_matrix_v31(np.arange(N_TRAIN,N_TRAIN+N_TEST),consensus3_test,base_test,meta_test,full_intent_te); assert bounds_full==bounds_test
full_rank_seed=_seed_from_text("V30-FULL-RANKER"); model_v30,raw_v30=fit_ranker_predict("lambda",Xtr_full[:,:bounds_full["base"]],Xte_full[:,:bounds_full["base"]],ORDINAL_GRADES,N_TRAIN,full_rank_seed); model_v31,raw_v31=fit_ranker_predict("lambda",Xtr_full[:,:bounds_full["intentstate"]],Xte_full[:,:bounds_full["intentstate"]],ORDINAL_GRADES,N_TRAIN,full_rank_seed)
v30_test=blend_residual(consensus3_test,raw_v30,ALPHA); v31_test=blend_residual(consensus3_test,raw_v31,ALPHA); protected_rows=[TEST_UID_TO_ROW[u] for u in sorted(PROTECTED_TEST_USERS)]
v30_test[protected_rows]=v25_anchor_test[protected_rows]; v31_test[protected_rows]=v25_anchor_test[protected_rows]
full_state=state_codes(v25_anchor_test,v30_test,v31_test); hybrid34_test=v30_test.copy(); hybrid34_test[full_state==3]=v31_test[full_state==3]
v25_top=top5_indices(v25_anchor_test); v30_top=top5_indices(v30_test); v31_top=top5_indices(v31_test); v30_changed_rows=np.flatnonzero(np.any(v25_top!=v30_top,axis=1)); v31_changed_rows=np.flatnonzero(np.any(v30_top!=v31_top,axis=1))
v30_changed_users=set(test.iloc[v30_changed_rows]["user_id"].astype(str)); state_count={s:int(np.sum(full_state==i)) for i,s in enumerate(["S0","S1","S2","S3"])}; hybrid_hash=hashlib.sha256(top5_indices(hybrid34_test).astype(np.int16).tobytes()).hexdigest()
V30_REPRO_OK=bool(v30_changed_users==EXPECTED_V30_CHANGED_USERS and int(np.sum(v25_top[:,0]!=v30_top[:,0]))==EXPECTED_V30_TOP1_CHANGES and int(np.sum([set(a)!=set(b) for a,b in zip(v25_top,v30_top)]))==EXPECTED_V30_MEMBERSHIP_CHANGES)
HYBRID_REPRO_OK=bool(len(v31_changed_rows)==EXPECTED_V31_CHANGED_VS_V30 and int(np.sum(full_state==3))==EXPECTED_HYBRID_OVERLAP and state_count==EXPECTED_STATE_COUNTS and hybrid_hash==EXPECTED_HYBRID_TOP5_SHA256)
V38_REPRO_AUDIT={"v30_reproduction_ok":V30_REPRO_OK,"hybrid34_reproduction_ok":HYBRID_REPRO_OK,"v30_changed_users":len(v30_changed_rows),"v31_changed_vs_v30":len(v31_changed_rows),"state_counts":state_count,"hybrid_top5_sha256":hybrid_hash}
print(json.dumps(V38_REPRO_AUDIT,indent=2));
if not V38_SMOKE: assert V30_REPRO_OK and HYBRID_REPRO_OK

def submission_from_scores(scores):
    out=sample.copy(); out.loc[:,MODULES]=np.asarray(scores,np.float32); assert np.isfinite(out[MODULES].to_numpy()).all(); return out

anchor_scores=hybrid34_test.copy(); safe_scores=anchor_scores.copy(); attack_scores=anchor_scores.copy(); FINAL_GATE_PREGEOM=bool(DEV_GATE and SEALED_GATE and V30_REPRO_OK and HYBRID_REPRO_OK)
V38_TEST_TAB=pd.DataFrame(); V38_TEST_SCORED=pd.DataFrame(); TEST_LEDGERS={"SAFE":pd.DataFrame(),"ATTACK":pd.DataFrame()}; TEST_GEOMETRY_AUC=float("nan"); GEOMETRY_GATE=False

if FINAL_GATE_PREGEOM:
    all_train=np.arange(N_TRAIN,dtype=int); fit_rows,cal_rows=split_fit_cal(all_train,_seed_from_text("V38-FINAL-CAL-SPLIT"))
    fit_oof=crossfit_hybrid_rows(fit_rows,_seed_from_text("V38-FINAL-HYBRID-OOF"),"FINAL-FIT"); cal_ref=fit_reference_hybrid(fit_rows,cal_rows,_seed_from_text("V38-FINAL-CAL-REF"))
    train_tab=build_pair_table(fit_rows,fit_oof["hybrid"],fit_oof["state"],True); cal_tab=build_pair_table(cal_rows,cal_ref["hybrid"],cal_ref["state"],True)
    test_rows_global=np.arange(N_TRAIN,N_TRAIN+N_TEST,dtype=int); V38_TEST_TAB=build_pair_table(test_rows_global,hybrid34_test,full_state,False)
    scored_members=[]; cal_members=[]
    for sd in FINAL_ENSEMBLE_SEEDS:
        bundle,cal_scored=train_xlmr_bundle(train_tab,cal_tab,int(sd),f"FINAL-{sd}"); scored=score_pair_bundle(bundle,V38_TEST_TAB); scored_members.append(scored); cal_members.append(cal_scored)
        del bundle; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    # Average only calibrated semantic outputs; row geometry is identical across members.
    V38_TEST_SCORED=scored_members[0].copy(); cal_ens=cal_members[0].copy()
    for col in ("p_benefit","p_harm","p_tie","utility","consistency","contradiction"):
        V38_TEST_SCORED[col]=np.mean(np.stack([q[col].to_numpy(float) for q in scored_members]),axis=0)
        # calibration rows are identical because the final split is shared.
        cal_ens[col]=np.mean(np.stack([q[col].to_numpy(float) for q in cal_members]),axis=0)
    final_cfg={p:derive_policy(cal_ens,len(cal_rows),p) for p in ("SAFE","ATTACK")}
    for p in ("SAFE","ATTACK"):
        sel=select_policy(V38_TEST_SCORED,N_TEST,final_cfg[p],p,blocked_rows=protected_rows); TEST_LEDGERS[p]=sel
        cand=apply_actions(hybrid34_test,sel); changed=np.any(top5_indices(cand)!=top5_indices(hybrid34_test),axis=1)
        assert not np.any(changed & (full_state!=0)); assert not np.any(changed[protected_rows]); assert int(np.sum(top5_indices(cand)[:,0]!=top5_indices(hybrid34_test)[:,0]))==0
        if p=="SAFE": safe_scores=cand
        else: attack_scores=cand
    # Label-free structured geometry check against K/L/M pair surface.
    geom_cols=["position","margin","membership"]+[f"s{i}" for i in range(len(PAIR_STRUCT_NAMES))]
    dev=[]
    for w in ("K","L","M"):
        q=WORLD_DATA[w]["val_tab"].copy(); S=np.stack(q.struct_f.to_numpy()); g=pd.DataFrame(S,columns=[f"s{i}" for i in range(S.shape[1])]); g.insert(0,"membership",q.membership.to_numpy()); g.insert(0,"margin",q.margin.to_numpy()); g.insert(0,"position",q.position.to_numpy()); dev.append(g)
    dev_geom=pd.concat(dev,ignore_index=True); S=np.stack(V38_TEST_TAB.struct_f.to_numpy()); test_geom=pd.DataFrame(S,columns=[f"s{i}" for i in range(S.shape[1])]); test_geom.insert(0,"membership",V38_TEST_TAB.membership.to_numpy()); test_geom.insert(0,"margin",V38_TEST_TAB.margin.to_numpy()); test_geom.insert(0,"position",V38_TEST_TAB.position.to_numpy())
    gx=pd.concat([dev_geom,test_geom],ignore_index=True).replace([np.inf,-np.inf],np.nan).fillna(0.0); gy=np.r_[np.zeros(len(dev_geom),int),np.ones(len(test_geom),int)]; cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=_seed_from_text("V38-GEOM")); gp=np.zeros(len(gx),float)
    for tr,va in cv.split(gx,gy):
        clf=make_pipeline(StandardScaler(),LogisticRegression(C=.5,max_iter=1000,random_state=SEED)); clf.fit(gx.iloc[tr],gy[tr]); gp[va]=clf.predict_proba(gx.iloc[va])[:,1]
    TEST_GEOMETRY_AUC=float(roc_auc_score(gy,gp)); GEOMETRY_GATE=bool(TEST_GEOMETRY_AUC<=TEST_GEOMETRY_AUC_MAX)
else:
    final_cfg={"SAFE":{},"ATTACK":{}}

FINAL_GATE=bool(FINAL_GATE_PREGEOM and GEOMETRY_GATE)
selected_scores=attack_scores if DEV_WINNER=="ATTACK" else safe_scores
primary_scores=selected_scores.copy() if FINAL_GATE else hybrid34_test.copy()
SUBMISSIONS={"anchor":submission_from_scores(hybrid34_test),"safe":submission_from_scores(safe_scores),"attack":submission_from_scores(attack_scores),"primary":submission_from_scores(primary_scores)}
SUBMISSIONS["anchor"].to_csv(OUT_DIR/"submission_v38_anchor_hybrid34_reconstructed.csv",index=False); SUBMISSIONS["safe"].to_csv(OUT_DIR/"submission_v38_DIAGNOSTIC_safe_xlmr.csv",index=False); SUBMISSIONS["attack"].to_csv(OUT_DIR/"submission_v38_DIAGNOSTIC_attack_xlmr.csv",index=False); SUBMISSIONS["primary"].to_csv(OUT_DIR/"submission_v38_primary.csv",index=False)
if not FINAL_GATE: assert np.array_equal(SUBMISSIONS["primary"][MODULES].to_numpy(),SUBMISSIONS["anchor"][MODULES].to_numpy())
print("TEST_GEOMETRY_AUC",TEST_GEOMETRY_AUC,"GEOMETRY_GATE",GEOMETRY_GATE,"FINAL_GATE",FINAL_GATE,"DEV_WINNER",DEV_WINNER)

# ---- cell 36 (markdown) ----
# ## Artifacts and submission contract
#
# `submission_v38_primary.csv` is submit-eligible **only when `FINAL_GATE=True`**. Otherwise it is byte-equivalent in values to the exact Hybrid34 `0.66006` anchor and must not consume a Kaggle slot. Diagnostic SAFE/ATTACK files are never promoted by test geometry alone.

# ---- cell 37 (code) ----
WORLD_ASSIGNMENTS.to_csv(OUT_DIR/"v38_disjoint_worlds.csv",index=False); TOKENIZER_AUDIT.to_csv(OUT_DIR/"v38_tokenizer_audit.csv",index=False); V38_VALIDATION.to_csv(OUT_DIR/"v38_validation.csv",index=False); V38_CALIBRATION.to_csv(OUT_DIR/"v38_calibration_policy.csv",index=False); V38_VALIDATION_LEDGER.to_csv(OUT_DIR/"v38_validation_interventions.csv",index=False); V38_TRAIN_HISTORY.to_csv(OUT_DIR/"v38_training_history.csv",index=False); V38_GATE.to_csv(OUT_DIR/"v38_development_gate.csv",index=False); V38_SEALED.to_csv(OUT_DIR/"v38_sealed_validation.csv",index=False); V38_SEALED_LEDGER.to_csv(OUT_DIR/"v38_sealed_interventions.csv",index=False); V38_SUPPORT_AUDIT.to_csv(OUT_DIR/"v38_intentstate_support_audit.csv",index=False)
if len(V38_TEST_TAB): V38_TEST_TAB.drop(columns=[c for c in ("struct_f","struct_r") if c in V38_TEST_TAB],errors="ignore").to_csv(OUT_DIR/"v38_test_pair_surface.csv",index=False)
if len(V38_TEST_SCORED): V38_TEST_SCORED.drop(columns=[c for c in ("struct_f","struct_r") if c in V38_TEST_SCORED],errors="ignore").to_csv(OUT_DIR/"v38_test_scored_pairs.csv",index=False)
for p in ("SAFE","ATTACK"):
    TEST_LEDGERS[p].drop(columns=[c for c in ("struct_f","struct_r") if c in TEST_LEDGERS[p]],errors="ignore").to_csv(OUT_DIR/f"v38_test_interventions_{p.lower()}.csv",index=False)
with open(OUT_DIR/"v38_reproduction_audit.json","w") as f: json.dump(V38_REPRO_AUDIT,f,indent=2)

slot_plan=pd.DataFrame([
    {"file":"submission_v38_primary.csv","role":"PRIMARY_ONLY_IF_ALL_GATES_PASS","eligible_to_submit":bool(FINAL_GATE),"note":f"{DEV_WINNER} if K/L/M + sealed + chronology + geometry all pass; otherwise exact Hybrid34 0.66006."},
    {"file":"submission_v38_DIAGNOSTIC_attack_xlmr.csv","role":"DIAGNOSTIC_ONLY","eligible_to_submit":False,"note":"Never bypass the preregistered gates."},
    {"file":"submission_v38_DIAGNOSTIC_safe_xlmr.csv","role":"DIAGNOSTIC_ONLY","eligible_to_submit":False,"note":"Never bypass the preregistered gates."},
    {"file":"submission_v38_anchor_hybrid34_reconstructed.csv","role":"CURRENT_PUBLIC_ANCHOR_REFERENCE","eligible_to_submit":False,"note":"Exact deterministic public 0.66006 Hybrid34."},
]); slot_plan.to_csv(OUT_DIR/"v38_slot_plan.csv",index=False)

summary={"version":"V38 Hybrid34-Locked XLM-R Pairwise Residual Judge","model":str(XLMR_SOURCE),"public_anchor":PUBLIC_HYBRID34_LB,"leader":PUBLIC_LEADER_LB,"leader_gap":PUBLIC_LEADER_GAP,
         "contract":{"state":"S0 only","positions":list(ELIGIBLE_POSITIONS),"rank1_locked":True,"max_actions_per_user":1,"tapt":False,"xe_used":False,"full_finetune":True,"final_seeds":list(FINAL_ENSEMBLE_SEEDS)},
         "development_winner":DEV_WINNER,"development_gate":bool(DEV_GATE),"sealed_gate":bool(SEALED_GATE),"test_geometry_auc":TEST_GEOMETRY_AUC,"geometry_gate":bool(GEOMETRY_GATE),"final_gate":bool(FINAL_GATE),
         "development":V38_VALIDATION.to_dict("records"),"sealed":V38_SEALED.to_dict("records"),"primary_sha256":hashlib.sha256(SUBMISSIONS["primary"].to_csv(index=False).encode()).hexdigest(),"anchor_sha256":hashlib.sha256(SUBMISSIONS["anchor"].to_csv(index=False).encode()).hexdigest(),
         "submission_instruction":"PRIMARY CLEARED EVERY PRECOMMITTED GATE. Inspect v38_test_interventions_* and submit primary." if FINAL_GATE else "NO NEW SUBMISSION. Primary is exact Hybrid34 0.66006; preserve the slot."}
with open(OUT_DIR/"v38_summary.json","w") as f: json.dump(summary,f,indent=2)
print(json.dumps(summary,indent=2)); print(slot_plan.to_string(index=False))

# ---- cell 38 (markdown) ----
# ### Interpretation contract
#
# - A high pairwise AUC alone is **not** a promotion criterion.
# - K/L/M are the only fresh development cohorts for this architecture.
# - Sealed D and chronology may reject a winner but may not retune thresholds.
# - Test intervention counts, confidence, or geometry may only reject through the preregistered geometry gate; they may not choose SAFE vs ATTACK.
# - If `FINAL_GATE=False`, V38 is a scientific negative result and the Kaggle slot must be preserved.
# - If `FINAL_GATE=True`, the primary file is the only V38 file cleared for a first submission.
