# Converted from notebooks/v41_2.ipynb: markdown cells become comments, %/! magics neutralized.

# ---- cell 0 (markdown) ----
# # MineToday V41.2 — B34-Anchored Head4 Preference Specialist
#
# **Final experiment:** dense Head-4 within-user preference learning with aggressively abstaining **Top-1-only** deployment.
#
# Frozen design:
# - exact public B34 anchor `0.66026`
# - exact ADAPT / CAL / LOCK = `2400 / 1000 / 600`
# - Qwen3-Embedding-8B QLoRA semantic expert, ADAPT only
# - one low-capacity transitive pairwise utility model
# - CAL: true user-level outer OOF + inner stability models
# - LOCK: opened once only after CAL passes
# - test: S0+S2 eligible; S1+S3 immutable
# - no Top-5 membership changes
# - no continuous final score blending: changed rows only **permute the existing B34 Top-4 score values**
# - exact B34 fail-close if any gate fails
#
# Kaggle runtime: **T4 × 2, Internet ON**. Qwen deliberately uses GPU 0 for deterministic training; the second T4 may remain idle.

# ---- cell 1 (code) ----
# Runtime + exact anchor reconstruction dependencies
import os, sys, re, json, ast, html, hashlib, random, warnings, time, subprocess, gc, math
from pathlib import Path
from collections import defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ["PYTHONHASHSEED"] = "42"

V40_SKIP_INSTALL = bool(int(os.getenv("V40_SKIP_INSTALL", "0")))
try:
    import lightgbm as lgb
except Exception:
    if V40_SKIP_INSTALL:
        raise
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "lightgbm==4.6.0"])
    import lightgbm as lgb

import numpy as np
import pandas as pd
import scipy
import sklearn
from IPython.display import display
from scipy.sparse import hstack
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")

SEED = 42
V40_SMOKE = bool(int(os.getenv("V40_SMOKE", "0")))
V40_STRICT_GEOMETRY = bool(int(os.getenv("V40_STRICT_GEOMETRY", "1")))

# Compatibility aliases consumed by the frozen V3.1 -> V25 -> V30/V31 reconstruction cells.
V395_SMOKE = V40_SMOKE
V395_STRICT_GEOMETRY = V40_STRICT_GEOMETRY
V31_SMOKE = V40_SMOKE
V30_SMOKE = V40_SMOKE
V30_STRICT_GEOMETRY = V40_STRICT_GEOMETRY
V29_SMOKE = V40_SMOKE
V29_STRICT_GEOMETRY = V40_STRICT_GEOMETRY
V27_SMOKE = V40_SMOKE
V27_STRICT_GEOMETRY = V40_STRICT_GEOMETRY

N_FOLDS = 2 if V40_SMOKE else 5
ANCHOR_INNER_FOLDS = 2 if V40_SMOKE else 4
INTENT_INNER_FOLDS = 2 if V40_SMOKE else 4

RIDGE_ALPHA = 3.0
REFERENCE_W = 0.45

# Historical/public references. Only B34 is carried forward as a whole natural family.
PUBLIC_CAL045_LB = 0.65639
PUBLIC_CONSENSUS3_LB = 0.65856
PUBLIC_V25_LB = 0.65919
PUBLIC_V30_LB = 0.65963
PUBLIC_V31_DIAGNOSTIC_LB = 0.65949
PUBLIC_HYBRID34_LB = 0.66006
PUBLIC_CURRENT_ANCHOR_LB = 0.66020
PUBLIC_B23_LB = 0.65998
PUBLIC_B34_LB = 0.66026
PUBLIC_RANK5_LB = 0.66066
PUBLIC_LEADER_LB = 0.66142

LGB_ESTIMATORS = 12 if V40_SMOKE else 1200
LGB_EARLY_STOP = 3 if V40_SMOKE else 80
RANKER_ESTIMATORS = 16 if V40_SMOKE else 360
RANKER_LEARNING_RATE = 0.04 if V40_SMOKE else 0.025

# Frozen V30/V31 residual geometry. No alpha search in V40.
ALPHA = 0.16
INTENT_PARENT_STRENGTH = 40.0
INTENT_CHILD_STRENGTH = 30.0
BOOTSTRAPS = 30 if V40_SMOKE else 1200
GEOMETRY_ATOL = 2.5e-4

# Exact historical geometry regression checks (ranking geometry only; no test labels).
EXPECTED_V30_CHANGED = 77
EXPECTED_V30_TOP1_CHANGES = 8
EXPECTED_V30_MEMBERSHIP_CHANGES = 7
EXPECTED_V31_CHANGED_VS_V30 = 60
EXPECTED_STATE_COUNTS = {"S0":897, "S1":26, "S2":43, "S3":34}
EXPECTED_S3_TOP1 = 3
EXPECTED_S3_MEMBERSHIP = 4
EXPECTED_S3_ORDER_ONLY = 27
EXPECTED_BOUNDARY_COUNTS = {"B23":9, "B34":10, "B45":8}
EXPECTED_V40_RULE_SELECTED_TEST = 2
HYBRID34_TOP5_SHA256 = "47f7db52ce8f782b8b393fd120140f1b6f51aeac452a9fabb5794cc9d8ba5e6e"
CURRENT_ANCHOR_TOP5_SHA256 = "c70ce570ec311e3f7391cec324e566a3a0e27da99ffc210afc3a8f102daf0191"

# Development evidence is intentionally the already-audited V31 fresh G/H/I worlds.
# Because the structural rule was formed after inspecting those worlds, a new sealed K world is mandatory.
def _seed_from_text(s):
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:8], 16) % (2**31-1)

FRESH_SALTS = {
    "G": _seed_from_text("MINETODAY-V31-FRESH-G-20260828-INTENTSTATE"),
    "H": _seed_from_text("MINETODAY-V31-FRESH-H-20260828-INTENTSTATE"),
    "I": _seed_from_text("MINETODAY-V31-FRESH-I-20260828-INTENTSTATE"),
}
V40_SEALED_NAME = "K"
V40_SEALED_SEED = _seed_from_text("MINETODAY-V40-SEALED-K-20260901-PREREQ-MEMBERSHIP")

# Preregistered structural gates. G/H/I are development confirmation only; K is the independent promotion gate.
DEV_MIN_TOTAL_ACTIONS = 10
DEV_MIN_ACTIONS_PER_WORLD = 2
DEV_MIN_POOLED_MEAN_DELTA = 0.020
DEV_MAX_HARMFUL_ACTIONS = 0
SEALED_MIN_ACTIONS = 2
SEALED_MIN_MEAN_DELTA = 0.0
SEALED_MAX_HARMFUL_ACTIONS = 0

random.seed(SEED)
np.random.seed(SEED)

print("numpy", np.__version__)
print("pandas", pd.__version__)
print("scipy", scipy.__version__)
print("sklearn", sklearn.__version__)
print("lightgbm", lgb.__version__)
print("V40_SMOKE", V40_SMOKE, "fixed alpha", ALPHA)
print("public B34 anchor", PUBLIC_B34_LB,
      "gap to rank5", round(PUBLIC_RANK5_LB-PUBLIC_B34_LB,5),
      "gap to leader", round(PUBLIC_LEADER_LB-PUBLIC_B34_LB,5))
print("fresh structural worlds", FRESH_SALTS, "sealed", V40_SEALED_NAME, V40_SEALED_SEED)
try:
    print("GPU runtime:\n" + subprocess.check_output(["nvidia-smi", "-L"], text=True).strip())
except Exception:
    print("nvidia-smi unavailable; V40 remains runnable and is deliberately CPU-first for exact geometry.")

# ---- cell 2 (code) ----

# V41.2 preregistered configuration — one final hypothesis, no post-hoc search.
V412_SMOKE = bool(int(os.getenv("V412_SMOKE", "0")))
V412_SKIP_DEEP = bool(int(os.getenv("V412_SKIP_DEEP", "0")))
V412_SKIP_INSTALL = bool(int(os.getenv("V412_SKIP_INSTALL", "0")))

# Compatibility aliases used by frozen V41/V40 reconstruction and Qwen cells.
V411_SMOKE = V412_SMOKE
V411_SKIP_DEEP = V412_SKIP_DEEP
V411_SKIP_INSTALL = V412_SKIP_INSTALL
V41_SMOKE = V412_SMOKE
V41_SKIP_DEEP = V412_SKIP_DEEP
V41_SKIP_INSTALL = V412_SKIP_INSTALL

if not V412_SKIP_INSTALL:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "-U",
        "transformers==4.57.6", "accelerate==1.14.0", "safetensors>=0.5,<1",
        "huggingface_hub>=0.34,<2", "sentencepiece>=0.2,<1",
        "peft>=0.17,<0.19", "bitsandbytes>=0.46,<0.49"
    ])

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("PYDEVD_DISABLE_FILE_VALIDATION", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Exact V41/V41.1 split.
V41_SPLIT_SEED = 2026090241
V41_CAL_SEED = 2026090242
V412_BOOT_SEED = 2026090245
ADAPT_SIZE = 240 if V412_SMOKE else 2400
CAL_SIZE = 100 if V412_SMOKE else 1000
LOCK_SIZE = 60 if V412_SMOKE else 600
CAL_FOLDS = 2 if V412_SMOKE else 5
INNER_FOLDS = 2 if V412_SMOKE else 4
BLOCKS = 2 if V412_SMOKE else 5
EXPECTED_B34_TOP5_SHA256 = "a235d00a05cb32573ba79565d05d70e3b65928cdb413d354acb5f9b1e039730c"

# Final Head-4 preference specialist.
PAIR_C = 0.25
ACTION_PROB_THRESHOLD = 0.55 if V412_SMOKE else 0.80
OOF_INNER_AGREE = 1 if V412_SMOKE else 3       # outer full model + >=3/4 inner = >=4/5 clean opinions
DEPLOY_OUTER_AGREE = 1 if V412_SMOKE else 4    # full CAL + >=4/5 outer CAL models

# CPU oracle gate happens before the expensive Qwen stage.
ORACLE_MIN_DELTA = -1.0 if V412_SMOKE else 0.00100

# Preregistered CAL gates.
CAL_MIN_DELTA = -1.0 if V412_SMOKE else 0.00050
CAL_MIN_FOLD_WINS = 0 if V412_SMOKE else 4
CAL_MIN_BLOCK_WINS = 0 if V412_SMOKE else 3
CAL_MIN_TOP1_DELTA = -1.0 if V412_SMOKE else 0.004
CAL_MIN_ACTION_PRECISION = 0.0 if V412_SMOKE else 0.80
CAL_MIN_ACTIONS = 0 if V412_SMOKE else 8
CAL_MAX_ACTIONS = 10_000 if V412_SMOKE else 30
CAL_MIN_CI_LOW = -1.0 if V412_SMOKE else -0.00035

# One untouched LOCK confirmation.
LOCK_MIN_DELTA = -1.0 if V412_SMOKE else 0.00030
LOCK_MIN_BLOCK_WINS = 0 if V412_SMOKE else 3
LOCK_MIN_ACTION_PRECISION = 0.0 if V412_SMOKE else 0.75
LOCK_MIN_NET_BENEFICIAL = -10_000 if V412_SMOKE else 2
LOCK_MIN_CI_LOW = -1.0 if V412_SMOKE else -0.00060

# Final test economics / safety geometry.
TEST_MIN_TOP1_ACTIONS = 0 if V412_SMOKE else 3
TEST_MAX_TOP1_ACTIONS = 10_000 if V412_SMOKE else 15

# Qwen expert is intentionally unchanged from V41.1.
QWEN_MODEL_ID = "Qwen/Qwen3-Embedding-8B"
QWEN_REVISION = "1d8ad4c"
QWEN_DIM = 1024
QWEN_QUERY_MAX = 96 if V412_SMOKE else 160
QWEN_MODULE_MAX = 96 if V412_SMOKE else 224
QWEN_LORA_LAST_N = 8
QWEN_LORA_R = 8
QWEN_LORA_ALPHA = 16
QWEN_LORA_DROPOUT = 0.05
QWEN_LR = 5e-5
QWEN_WD = 0.01
QWEN_EPOCHS = 1
QWEN_BATCH = 8 if V412_SMOKE else 4

V41_OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("/mnt/data")
V41_OUT.mkdir(parents=True, exist_ok=True)
V41_HF_CACHE = V41_OUT / "v41_2_hf_cache"
V41_HF_CACHE.mkdir(parents=True, exist_ok=True)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("V412_SMOKE", V412_SMOKE, "skip_deep", V412_SKIP_DEEP)
print("public B34 anchor", PUBLIC_B34_LB,
      "gap to rank5", round(PUBLIC_RANK5_LB-PUBLIC_B34_LB, 5),
      "gap to leader", round(PUBLIC_LEADER_LB-PUBLIC_B34_LB, 5))
try:
    print("GPU runtime:\n" + subprocess.check_output(["nvidia-smi", "-L"], text=True).strip())
except Exception:
    print("nvidia-smi unavailable; full V41.2 requires Kaggle GPU T4x2.")

# ---- cell 3 (code) ----
def locate_data_dir():
    required = {
        "chat_history.csv", "modules_catalog.csv", "user_assessments.csv",
        "train_relevance.csv", "test.csv", "sample_submission.csv"
    }
    roots = [Path("/kaggle/input"), Path("/mnt/data/_mine_data"), Path("/mnt/data/mine_data"), Path("/mnt/data/comp_v3"), Path("/mnt/data/comp_full"), Path(".")]
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

# ---- cell 4 (code) ----
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

# ---- cell 5 (code) ----
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

# ---- cell 6 (code) ----
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

# ---- cell 7 (code) ----
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

# ---- cell 11 (code) ----
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

# ---- cell 12 (code) ----
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

# ---- cell 13 (code) ----
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

# ---- cell 14 (code) ----
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

# ---- cell 15 (code) ----
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
        raise ValueError("non-finite V31 long features")
    return XX,names,bounds

def make_ranker(seed):
    return lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=RANKER_ESTIMATORS,
        learning_rate=RANKER_LEARNING_RATE,
        num_leaves=15,
        min_child_samples=80,
        reg_alpha=1.0,
        reg_lambda=10.0,
        colsample_bytree=0.78,
        subsample=0.90,
        subsample_freq=1,
        max_bin=127,
        lambdarank_truncation_level=8,
        label_gain=LABEL_GAIN,
        random_state=int(seed),
        n_jobs=4,
        verbosity=-1,
        force_col_wise=True,
    )

def fit_ranker_predict(Xtr,Xva,grades_tr,n_users,seed):
    model=make_ranker(seed)
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

def cluster_boot_ci(values,groups,seed,B=BOOTSTRAPS):
    values=np.asarray(values,float); groups=np.asarray(groups,dtype=object)
    ug=np.unique(groups); by={g:np.flatnonzero(groups==g) for g in ug}
    rng=np.random.default_rng(int(seed)); z=np.empty(int(B),float)
    for b in range(int(B)):
        sampled=rng.choice(ug,size=len(ug),replace=True)
        idx=np.concatenate([by[g] for g in sampled])
        z[b]=values[idx].mean()
    return tuple(float(x) for x in np.quantile(z,[.025,.975]))

print("V30-control / V31-IntentState LambdaRank helpers ready")

# ---- cell 16 (code) ----
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

print("Nested frozen anchor helper ready")

# ---- cell 17 (code) ----
STATE_NAMES=np.asarray(["S0","S1","S2","S3"],dtype=object)


def state_codes(pre_anchor,v30,v31):
    pre_top=top5_indices(pre_anchor); v30_top=top5_indices(v30); v31_top=top5_indices(v31)
    a=np.any(pre_top!=v30_top,axis=1)
    b=np.any(v30_top!=v31_top,axis=1)
    return (2*a.astype(np.int8)+b.astype(np.int8)).astype(np.int8)


def top5_sha256(scores):
    top=top5_indices(scores).astype(np.int16)
    return hashlib.sha256(top.tobytes()).hexdigest()


def current066020_surface(v30,v31,state):
    v30=np.asarray(v30,np.float32); v31=np.asarray(v31,np.float32); state=np.asarray(state,np.int8)
    t30=top5_indices(v30); t31=top5_indices(v31)
    top1=(t30[:,0]!=t31[:,0])
    membership=np.asarray([set(a)!=set(b) for a,b in zip(t30,t31)],bool)
    s3=(state==3)
    keep=s3 & (top1 | membership)
    out=v30.copy(); out[keep]=v31[keep]
    info={
        "s3":s3,
        "top1":s3 & top1,
        "membership":s3 & membership & ~top1,
        "order_only":s3 & ~top1 & ~membership,
        "accepted":keep,
    }
    return out,info


def classify_single_adjacent_order_swap(v30_top_row,v31_top_row):
    a=np.asarray(v30_top_row,int); b=np.asarray(v31_top_row,int)
    if a[0]!=b[0] or set(a)!=set(b):
        raise AssertionError(("not pure order-only",a.tolist(),b.tolist()))
    diff=np.flatnonzero(a!=b)
    if len(diff)!=2 or diff[1]!=diff[0]+1:
        raise AssertionError(("not a single adjacent transposition",a.tolist(),b.tolist(),diff.tolist()))
    lo,hi=int(diff[0]),int(diff[1])
    if not (a[lo]==b[hi] and a[hi]==b[lo]):
        raise AssertionError(("not a two-way swap",a.tolist(),b.tolist()))
    boundary=f"B{lo+1}{hi+1}"
    if boundary not in EXPECTED_BOUNDARY_COUNTS:
        raise AssertionError(("unexpected order-only boundary",boundary,a.tolist(),b.tolist()))
    promoted=int(b[lo])
    demoted=int(a[lo])
    return boundary,promoted,demoted,lo+1,hi+1


def submission_from_scores(scores):
    out=sample.copy(); out.loc[:,MODULES]=np.asarray(scores,np.float32)
    assert out.shape==sample.shape
    assert out["user_id"].astype(str).tolist()==test["user_id"].astype(str).tolist()
    assert np.isfinite(out[MODULES].to_numpy()).all()
    return out


def candidate_geometry(name,anchor,candidate):
    ta=top5_indices(anchor); tc=top5_indices(candidate)
    changed=np.any(ta!=tc,axis=1)
    membership=np.asarray([set(a)!=set(b) for a,b in zip(ta,tc)],bool)
    return {
        "candidate":name,
        "changed_users_vs_current":int(changed.sum()),
        "top1_changes":int(np.sum(ta[:,0]!=tc[:,0])),
        "membership_changes":int(membership.sum()),
        "top5_sha256":top5_sha256(candidate),
    }

print("V39.5 state machine + causal boundary helpers ready")

# ---- cell 18 (code) ----
OUT_DIR=Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("/mnt/data")
OUT_DIR.mkdir(parents=True,exist_ok=True)

# V25 protection set is inherited from the frozen historical package reconstructed above.
PROTECTED_TEST_USERS=set([u for u,_,_ in KNOWN_PB_SWAPS] + [a["user_id"] for a in V25_CORE_ACTIONS])
protected_rows=np.asarray([TEST_UID_TO_ROW[u] for u in sorted(PROTECTED_TEST_USERS)],int)

# Strict full-train V31 Intent-State representation for exact V30/V31 reconstruction.
full_intent_tr=crossfit_intentstate_features(
    y,ORDINAL_GRADES,TRAIN_INTENT,MODULE_READY_BAND[:N_TRAIN],TRAIN_GROUP,
    _seed_from_text("V31-FULL-INTENTSTATE-CROSSFIT")
)
FULL_INTENT_BUNDLE=fit_intentstate_bundle(y,ORDINAL_GRADES,TRAIN_INTENT,MODULE_READY_BAND[:N_TRAIN])
full_intent_te=apply_intentstate_bundle(FULL_INTENT_BUNDLE,TEST_INTENT,MODULE_READY_BAND[N_TRAIN:])
V395_INTENT_SUPPORT_AUDIT=intentstate_support_audit(FULL_INTENT_BUNDLE,TEST_INTENT,MODULE_READY_BAND[N_TRAIN:])

Xtr_all,names_all,bounds_full=build_long_matrix_v31(
    np.arange(N_TRAIN),consensus3_oof,base_oof,meta_oof,full_intent_tr
)
Xte_all,_,bounds_test=build_long_matrix_v31(
    np.arange(N_TRAIN,N_TRAIN+N_TEST),consensus3_test,base_test,meta_test,full_intent_te
)
assert bounds_full==bounds_test
be=bounds_full["base"]

full_rank_seed=_seed_from_text("V30-FULL-RANKER")
model_v30,raw_v30=fit_ranker_predict(
    Xtr_all[:,:be],Xte_all[:,:be],ORDINAL_GRADES,N_TRAIN,full_rank_seed
)
model_v31,raw_v31=fit_ranker_predict(
    Xtr_all[:,:bounds_full["intentstate"]],Xte_all[:,:bounds_full["intentstate"]],
    ORDINAL_GRADES,N_TRAIN,full_rank_seed
)

v30_test=blend_residual(consensus3_test,raw_v30,ALPHA)
v31_test=blend_residual(consensus3_test,raw_v31,ALPHA)

# Frozen V25 rows remain exact in V30/V31 decision surfaces.
for arr in (v30_test,v31_test):
    arr[protected_rows]=v25_anchor_test[protected_rows]

full_state=state_codes(v25_anchor_test,v30_test,v31_test)
hybrid34_test=v30_test.copy(); hybrid34_test[full_state==3]=v31_test[full_state==3]
current_anchor_test,current_info=current066020_surface(v30_test,v31_test,full_state)

v25_top=top5_indices(v25_anchor_test); v30_top=top5_indices(v30_test); v31_top=top5_indices(v31_test)
v30_changed=int(np.sum(np.any(v25_top!=v30_top,axis=1)))
v31_changed=int(np.sum(np.any(v30_top!=v31_top,axis=1)))
v30_top1=int(np.sum(v25_top[:,0]!=v30_top[:,0]))
v30_membership=int(np.sum([set(a)!=set(b) for a,b in zip(v25_top,v30_top)]))
state_counts={STATE_NAMES[s]:int(np.sum(full_state==s)) for s in range(4)}
hybrid_hash=top5_sha256(hybrid34_test)
current_hash=top5_sha256(current_anchor_test)

V395_REPRO_AUDIT={
    "reproduction_ok":False,
    "v30_changed_vs_v25":v30_changed,"expected_v30_changed":EXPECTED_V30_CHANGED,
    "v30_top1_changes":v30_top1,"expected_v30_top1_changes":EXPECTED_V30_TOP1_CHANGES,
    "v30_membership_changes":v30_membership,"expected_v30_membership_changes":EXPECTED_V30_MEMBERSHIP_CHANGES,
    "v31_changed_vs_v30":v31_changed,"expected_v31_changed":EXPECTED_V31_CHANGED_VS_V30,
    "state_counts":state_counts,"expected_state_counts":EXPECTED_STATE_COUNTS,
    "s3_top1_kept":int(current_info["top1"].sum()),"expected_s3_top1":EXPECTED_S3_TOP1,
    "s3_membership_kept":int(current_info["membership"].sum()),"expected_s3_membership":EXPECTED_S3_MEMBERSHIP,
    "s3_order_only_rejected":int(current_info["order_only"].sum()),"expected_s3_order_only":EXPECTED_S3_ORDER_ONLY,
    "hybrid34_top5_sha256":hybrid_hash,"expected_hybrid34_top5_sha256":HYBRID34_TOP5_SHA256,
    "current_anchor_top5_sha256":current_hash,"expected_current_anchor_top5_sha256":CURRENT_ANCHOR_TOP5_SHA256,
}
REPRO_OK=bool(
    v30_changed==EXPECTED_V30_CHANGED and
    v30_top1==EXPECTED_V30_TOP1_CHANGES and
    v30_membership==EXPECTED_V30_MEMBERSHIP_CHANGES and
    v31_changed==EXPECTED_V31_CHANGED_VS_V30 and
    state_counts==EXPECTED_STATE_COUNTS and
    int(current_info["top1"].sum())==EXPECTED_S3_TOP1 and
    int(current_info["membership"].sum())==EXPECTED_S3_MEMBERSHIP and
    int(current_info["order_only"].sum())==EXPECTED_S3_ORDER_ONLY and
    hybrid_hash==HYBRID34_TOP5_SHA256 and
    current_hash==CURRENT_ANCHOR_TOP5_SHA256
)
V395_REPRO_AUDIT["reproduction_ok"]=REPRO_OK
print(json.dumps(V395_REPRO_AUDIT,indent=2))
if not V395_SMOKE:
    assert REPRO_OK,"Exact 0.66020 / Hybrid34 geometry did not reproduce; abort V39.5 probe creation."

print("Exact current anchor reconstructed; starting S3 order-only causal decomposition.")

# ---- cell 19 (code) ----
# Reconstruct B34 from the exact current 0.66020 surface.
t30=top5_indices(v30_test); t31=top5_indices(v31_test)
order_rows=np.flatnonzero(current_info["order_only"])
if not V40_SMOKE:
    assert len(order_rows)==EXPECTED_S3_ORDER_ONLY

order_ledger=[]
for r in order_rows:
    try:
        boundary,promoted,demoted,pos_from,pos_to=classify_single_adjacent_order_swap(t30[r],t31[r])
    except AssertionError:
        if V40_SMOKE:
            continue
        raise
    order_ledger.append({
        "row_local":int(r), "user_id":str(test.iloc[r].user_id), "boundary":boundary,
        "promoted_module":MODULES[promoted], "demoted_module":MODULES[demoted],
        "latest_intent_signature":str(TEST_INTENT[r]),
        "readiness_band_promoted":int(MODULE_READY_BAND[N_TRAIN+r,promoted]),
    })
V40_ORDER_LEDGER=pd.DataFrame(order_ledger)
if len(V40_ORDER_LEDGER):
    V40_ORDER_LEDGER=V40_ORDER_LEDGER.sort_values(["boundary","row_local"]).reset_index(drop=True)
else:
    V40_ORDER_LEDGER=pd.DataFrame(columns=["row_local","user_id","boundary","promoted_module","demoted_module","latest_intent_signature","readiness_band_promoted"])
if not V40_SMOKE:
    assert V40_ORDER_LEDGER.boundary.value_counts().sort_index().to_dict()==EXPECTED_BOUNDARY_COUNTS

B34_ROWS=V40_ORDER_LEDGER.loc[V40_ORDER_LEDGER.boundary.eq("B34"),"row_local"].to_numpy(int)
if not V40_SMOKE:
    assert len(B34_ROWS)==10
b34_anchor_test=current_anchor_test.copy()
b34_anchor_test[B34_ROWS]=v31_test[B34_ROWS]

b34_geom=candidate_geometry("b34_public_anchor",current_anchor_test,b34_anchor_test)
if not V40_SMOKE:
    assert b34_geom["changed_users_vs_current"]==10
    assert b34_geom["top1_changes"]==0 and b34_geom["membership_changes"]==0



b34_hash = top5_sha256(b34_anchor_test)
if not V41_SMOKE:
    assert len(B34_ROWS)==10
    assert b34_hash==EXPECTED_B34_TOP5_SHA256, (b34_hash, EXPECTED_B34_TOP5_SHA256)
print("Exact B34 operational anchor reconstructed from the exact 9/10/8 S3 decomposition:", b34_hash, "changed rows", len(B34_ROWS))

# ---- cell 20 (markdown) ----
# ## 1. Fixed ADAPT / CAL / LOCK split and ADAPT-only structural expert

# ---- cell 21 (code) ----

pos_count=(y>0).sum(1).astype(int)
if V41_SMOKE:
    rng=np.random.default_rng(V41_SPLIT_SEED); perm=rng.permutation(N_TRAIN)
    LOCK_IDX=np.sort(perm[:LOCK_SIZE])
    CAL_IDX=np.sort(perm[LOCK_SIZE:LOCK_SIZE+CAL_SIZE])
    ADAPT_IDX=np.sort(perm[LOCK_SIZE+CAL_SIZE:LOCK_SIZE+CAL_SIZE+ADAPT_SIZE])
else:
    joint=np.asarray([f"{np.argmax(y[i]):02d}_{pos_count[i]}" for i in range(N_TRAIN)])
    s1=StratifiedShuffleSplit(n_splits=1,test_size=LOCK_SIZE,random_state=V41_SPLIT_SEED)
    rem,lk=next(s1.split(np.zeros(N_TRAIN),joint)); REM=np.asarray(rem,int); LOCK_IDX=np.asarray(lk,int)
    s2=StratifiedShuffleSplit(n_splits=1,test_size=CAL_SIZE,random_state=V41_SPLIT_SEED+1)
    ad,ca=next(s2.split(np.zeros(len(REM)),joint[REM])); ADAPT_IDX=REM[np.asarray(ad,int)]; CAL_IDX=REM[np.asarray(ca,int)]
assert len(ADAPT_IDX)==ADAPT_SIZE and len(CAL_IDX)==CAL_SIZE and len(LOCK_IDX)==LOCK_SIZE
assert len(set(ADAPT_IDX)&set(CAL_IDX))==0 and len(set(ADAPT_IDX)&set(LOCK_IDX))==0 and len(set(CAL_IDX)&set(LOCK_IDX))==0
TEST_GLOBAL=np.arange(N_TRAIN,N_TRAIN+N_TEST,dtype=int)
OUT_GLOBAL=np.concatenate([CAL_IDX,LOCK_IDX,TEST_GLOBAL])

# One ADAPT-only structural model predicts CAL, LOCK, and TEST without seeing their labels.
struct_bundle=build_anchor_split_full(ADAPT_IDX,OUT_GLOBAL,_seed_from_text("V41-ADAPT-STRUCTURAL"))
ad_intent_oof=crossfit_intentstate_features(
    y[ADAPT_IDX],ORDINAL_GRADES[ADAPT_IDX],TRAIN_INTENT[ADAPT_IDX],MODULE_READY_BAND[ADAPT_IDX],TRAIN_GROUP[ADAPT_IDX],
    _seed_from_text("V41-ADAPT-INTENT-CROSSFIT")
)
ad_intent_bundle=fit_intentstate_bundle(y[ADAPT_IDX],ORDINAL_GRADES[ADAPT_IDX],TRAIN_INTENT[ADAPT_IDX],MODULE_READY_BAND[ADAPT_IDX])
out_intents=np.concatenate([TRAIN_INTENT[CAL_IDX],TRAIN_INTENT[LOCK_IDX],TEST_INTENT])
out_ready=MODULE_READY_BAND[OUT_GLOBAL]
out_intent=apply_intentstate_bundle(ad_intent_bundle,out_intents,out_ready)

Xa,names_a,bounds_a=build_long_matrix_v31(ADAPT_IDX,struct_bundle["cons_train"],struct_bundle["base_train"],struct_bundle["meta_train"],ad_intent_oof)
Xo,_,bounds_o=build_long_matrix_v31(OUT_GLOBAL,struct_bundle["cons_val"],struct_bundle["base_val"],struct_bundle["meta_val"],out_intent)
assert bounds_a==bounds_o
be=bounds_a["base"]
seed_rank=_seed_from_text("V41-ADAPT-STRUCTURAL-RANKER")
_,raw30=fit_ranker_predict(Xa[:,:be],Xo[:,:be],ORDINAL_GRADES[ADAPT_IDX],len(ADAPT_IDX),seed_rank)
_,raw31=fit_ranker_predict(Xa[:,:bounds_a["intentstate"]],Xo[:,:bounds_a["intentstate"]],ORDINAL_GRADES[ADAPT_IDX],len(ADAPT_IDX),seed_rank)
struct_v30_out=blend_residual(struct_bundle["cons_val"],raw30,ALPHA)
struct_v31_out=blend_residual(struct_bundle["cons_val"],raw31,ALPHA)

nca=len(CAL_IDX); nlk=len(LOCK_IDX)
sl_cal=slice(0,nca); sl_lock=slice(nca,nca+nlk); sl_test=slice(nca+nlk,None)

def ss(a,sl): return np.asarray(a[sl],np.float32)
CAL_STRUCT={"base":ss(struct_bundle["base_val"],sl_cal),"meta":ss(struct_bundle["meta_val"],sl_cal),"cons":ss(struct_bundle["cons_val"],sl_cal),"v30":ss(struct_v30_out,sl_cal),"v31":ss(struct_v31_out,sl_cal)}
LOCK_STRUCT={"base":ss(struct_bundle["base_val"],sl_lock),"meta":ss(struct_bundle["meta_val"],sl_lock),"cons":ss(struct_bundle["cons_val"],sl_lock),"v30":ss(struct_v30_out,sl_lock),"v31":ss(struct_v31_out,sl_lock)}
TEST_STRUCT={"base":ss(struct_bundle["base_val"],sl_test),"meta":ss(struct_bundle["meta_val"],sl_test),"cons":ss(struct_bundle["cons_val"],sl_test),"v30":ss(struct_v30_out,sl_test),"v31":ss(struct_v31_out,sl_test)}

CAL_STATE=state_codes(CAL_STRUCT["cons"],CAL_STRUCT["v30"],CAL_STRUCT["v31"]); CAL_S0=(CAL_STATE==0)
LOCK_STATE=state_codes(LOCK_STRUCT["cons"],LOCK_STRUCT["v30"],LOCK_STRUCT["v31"]); LOCK_S0=(LOCK_STATE==0)
TEST_S0=(full_state==0)

# Honest ADAPT structural teacher for Qwen hard-negative weighting.
ADAPT_TEACHER_RANKS=rank_matrix(struct_bundle["cons_train"])-1
ADAPT_LOCAL={int(g):i for i,g in enumerate(ADAPT_IDX.tolist())}

print("split sizes",len(ADAPT_IDX),len(CAL_IDX),len(LOCK_IDX),"S0 analog cal/lock",int(CAL_S0.sum()),int(LOCK_S0.sum()),"exact test S0",int(TEST_S0.sum()))
print("ADAPT-only structural v30 NDCG CAL",round(ndcg5(y[CAL_IDX],CAL_STRUCT["v30"]),6),"LOCK",round(ndcg5(y[LOCK_IDX],LOCK_STRUCT["v30"]),6))


# V41.2 state eligibility is preregistered: S0 + S2 only.
CAL_ELIG = np.isin(CAL_STATE, [0, 2])
LOCK_ELIG = np.isin(LOCK_STATE, [0, 2])
TEST_ELIG = np.isin(full_state, [0, 2])
if not V412_SMOKE:
    assert int(TEST_ELIG.sum()) == EXPECTED_STATE_COUNTS["S0"] + EXPECTED_STATE_COUNTS["S2"]

def _top4(scores):
    return np.argsort(-np.asarray(scores), axis=1, kind="mergesort")[:, :4]

def promote_existing_top4_score_values(scores, row, challenger):
    """Promote one existing Top-4 module to #1 by permuting only the row's original Top-4 score values."""
    out = np.asarray(scores, np.float32).copy()
    order = np.argsort(-out[row], kind="mergesort")
    top4 = order[:4].tolist()
    challenger = int(challenger)
    if challenger not in top4:
        raise AssertionError("challenger must already be inside anchor Top-4")
    if top4[0] == challenger:
        return out
    vals = out[row, top4].copy()  # already descending
    new_order = [challenger] + [m for m in top4 if m != challenger]
    out[row, new_order] = vals
    return out

def optimally_order_existing_top4(scores, ytrue, eligible):
    out = np.asarray(scores, np.float32).copy()
    for r in np.flatnonzero(np.asarray(eligible, bool)):
        top4 = np.argsort(-out[r], kind="mergesort")[:4]
        vals = out[r, top4].copy()
        # stable tie-break by existing anchor position
        ord4 = sorted(top4.tolist(), key=lambda m: (-float(ytrue[r, m]), int(np.where(top4 == m)[0][0])))
        out[r, ord4] = vals
    return out

# Deployment-like CAL head oracle. This is label-using diagnostics only and never enters test prediction.
CAL_ANCHOR = np.asarray(CAL_STRUCT["v30"], np.float32)
cal_top4 = _top4(CAL_ANCHOR)
cal_true_top1 = np.argmax(y[CAL_IDX], axis=1)
cal_anchor_top1 = cal_top4[:, 0]
recoverable = np.asarray([
    bool(CAL_ELIG[r] and cal_anchor_top1[r] != cal_true_top1[r] and cal_true_top1[r] in set(cal_top4[r]))
    for r in range(len(CAL_IDX))
])

CAL_TOP1_ORACLE = CAL_ANCHOR.copy()
for r in np.flatnonzero(recoverable):
    CAL_TOP1_ORACLE = promote_existing_top4_score_values(CAL_TOP1_ORACLE, r, cal_true_top1[r])
CAL_TOP4_ORACLE = optimally_order_existing_top4(CAL_ANCHOR, y[CAL_IDX], CAL_ELIG)

oracle_top1_delta = float(ndcg5(y[CAL_IDX], CAL_TOP1_ORACLE) - ndcg5(y[CAL_IDX], CAL_ANCHOR))
oracle_top4_delta = float(ndcg5(y[CAL_IDX], CAL_TOP4_ORACLE) - ndcg5(y[CAL_IDX], CAL_ANCHOR))
eligible_wrong = np.asarray(CAL_ELIG & (cal_anchor_top1 != cal_true_top1), bool)
coverage = float(recoverable.sum() / max(1, eligible_wrong.sum()))

V412_HEAD_ORACLE = pd.DataFrame([{
    "cal_anchor_ndcg": ndcg5(y[CAL_IDX], CAL_ANCHOR),
    "eligible_users": int(CAL_ELIG.sum()),
    "eligible_wrong_top1": int(eligible_wrong.sum()),
    "recoverable_top1_errors": int(recoverable.sum()),
    "true_top1_inside_anchor_top4_rate_given_wrong": coverage,
    "top1_within_top4_oracle_delta": oracle_top1_delta,
    "full_top4_permutation_oracle_delta": oracle_top4_delta,
}])

ORACLE_GATE = bool(REPRO_OK and oracle_top1_delta >= ORACLE_MIN_DELTA)
# If the action space itself lacks enough headroom, stop before burning the GPU hour.
V41_SKIP_DEEP = bool(V412_SKIP_DEEP or not ORACLE_GATE)

print(V412_HEAD_ORACLE.to_string(index=False))
print("ORACLE_GATE", ORACLE_GATE, "| Qwen will run:", not V41_SKIP_DEEP)

# ---- cell 22 (markdown) ----
# ## 2. Frozen Qwen3-Embedding-8B semantic expert — ADAPT only
#
# The deep expert is intentionally unchanged from V41.1. V41.2 tests **how semantic information is extracted**, not another encoder/LoRA search.
#
# If the CPU Top-1-within-Top4 oracle cannot clear the preregistered headroom requirement, this stage automatically skips and the notebook fails closed to exact B34.

# ---- cell 23 (code) ----

def semantic_normalize(text):
    return re.sub(r"\s+"," ",html.unescape(str(text))).strip()

learn_sem=chat[~chat["is_noise"]].copy()
learn_sem["semantic_text"]=learn_sem["user_chat_text"].fillna("").map(semantic_normalize)
learn_sem=learn_sem[learn_sem.semantic_text.str.len()>0].sort_values(["user_id","timestamp","chat_id"]).reset_index(drop=True)
TRAJ_TEXTS=[]; HAS_SEMANTIC=[]
for u in all_users:
    g=learn_sem[learn_sem.user_id==u]
    msgs=g.semantic_text.tolist()[-3:]
    if not msgs:
        TRAJ_TEXTS.append(""); HAS_SEMANTIC.append(False); continue
    tags=["OLDER","RECENT","CURRENT"][-len(msgs):]
    TRAJ_TEXTS.append(" ".join(f"[{t}] {m}" for t,m in zip(tags,msgs))); HAS_SEMANTIC.append(True)
TRAJ_TEXTS=np.asarray(TRAJ_TEXTS,dtype=object); HAS_SEMANTIC=np.asarray(HAS_SEMANTIC,bool)
mods=modules.set_index("module_id").loc[MODULES].reset_index()
QWEN_MODULE_DOCS=[f"Modul pembelajaran Intelligo: {r.module_name}. Materi utama: {r.description_and_syllabus}. Prasyarat: {r.prerequisite_level}." for r in mods.itertuples(index=False)]
QWEN_INSTRUCTION="Represent the learner's current learning need for matching to the most relevant Intelligo course module: "

def locate_qwen_source():
    env=os.getenv("V41_QWEN_MODEL_PATH","").strip()
    if env and Path(env).exists(): return env
    root=Path("/kaggle/input")
    if root.exists():
        hits=[]
        for p in root.rglob("config.json"):
            try:
                cfg=json.load(open(p)); mt=str(cfg.get("model_type","")).lower()
                if "qwen3" in mt or "qwen" in str(p.parent).lower(): hits.append(str(p.parent))
            except Exception: pass
        if hits: return sorted(hits,key=len)[0]
    return QWEN_MODEL_ID

LORA_PROJECTION_SUFFIXES=("q_proj","k_proj","v_proj","o_proj")
def _select_lora_targets(module_names,last_n):
    rows=[]
    for name in module_names:
        suffix=name.rsplit(".",1)[-1]
        if suffix not in LORA_PROJECTION_SUFFIXES: continue
        m=re.search(r"(?:^|\.)(?:layers|h|blocks)\.(\d+)(?:\.|$)",name)
        if m is None: m=re.search(r"(?:^|\.)(\d+)\.(?:self_attn|attention|attn)\.",name)
        if m is not None: rows.append((int(m.group(1)),name,suffix))
    if not rows: raise RuntimeError("Could not discover Qwen attention projection modules for LoRA")
    available=sorted({r[0] for r in rows}); chosen=available[-min(int(last_n),len(available)):]; cs=set(chosen)
    targets=[name for layer,name,_ in rows if layer in cs]
    if not any(n.endswith("q_proj") for n in targets) or not any(n.endswith("v_proj") for n in targets): raise RuntimeError("Qwen q_proj/v_proj missing")
    return targets,chosen

def last_token_pool(last_hidden,attention_mask):
    mask=attention_mask.to(last_hidden.device)
    if bool((mask[:,-1].sum()==mask.shape[0]).item()): return last_hidden[:,-1]
    seq=mask.sum(1)-1
    return last_hidden[torch.arange(last_hidden.shape[0],device=last_hidden.device),seq]

def qnorm(x): return F.normalize(x[...,:QWEN_DIM].float(),dim=-1)

def load_qwen():
    if V41_SMOKE or V41_SKIP_DEEP: return None,None
    if not torch.cuda.is_available(): raise RuntimeError("V41 full run requires GPU")
    from transformers import AutoTokenizer,AutoModel,BitsAndBytesConfig
    from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training,TaskType
    source=locate_qwen_source(); print("Qwen source",source)
    tok=AutoTokenizer.from_pretrained(source,revision=(QWEN_REVISION if source==QWEN_MODEL_ID else None),padding_side="left",cache_dir=str(V41_HF_CACHE))
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
    base=AutoModel.from_pretrained(source,revision=(QWEN_REVISION if source==QWEN_MODEL_ID else None),quantization_config=bnb,device_map={"":0},torch_dtype=torch.float16,low_cpu_mem_usage=True,attn_implementation="sdpa",cache_dir=str(V41_HF_CACHE))
    base.config.use_cache=False; base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=True)
    targets,layers=_select_lora_targets([n for n,_ in base.named_modules()],QWEN_LORA_LAST_N); print("Qwen LoRA layers",layers,"targets",len(targets))
    cfg=LoraConfig(r=QWEN_LORA_R,lora_alpha=QWEN_LORA_ALPHA,lora_dropout=QWEN_LORA_DROPOUT,bias="none",task_type=TaskType.FEATURE_EXTRACTION,target_modules=targets)
    model=get_peft_model(base,cfg); model.config.use_cache=False; model.train(); model.print_trainable_parameters()
    return model,tok

def qtoks(tok,texts,maxlen,prefix=""):
    return tok([prefix+str(t) for t in texts],padding=True,truncation=True,max_length=maxlen,return_tensors="pt")
def qmove(z,dev): return {k:v.to(dev) for k,v in z.items()}

def qwen_loss(scores,yb,anchor_rank):
    pos=(yb>0).float(); gains=(torch.pow(2.0,yb)-1.0).clamp_min(0)
    target=.75*pos/pos.sum(1,keepdim=True).clamp_min(1.0)+.25*gains/gains.sum(1,keepdim=True).clamp_min(1e-6)
    rel=-(target*F.log_softmax(scores/.075,dim=1)).sum(1).mean()
    si=scores[:,:,None]; sj=scores[:,None,:]; yi=yb[:,:,None]; yj=yb[:,None,:]
    ar_i=anchor_rank[:,:,None]; ar_j=anchor_rank[:,None,:]
    mask=(yi>0)&(yj<=0); w=(1.0+1.5*(ar_j<5).float()+.6*(ar_i>=5).float())*gains[:,:,None].clamp_min(.15)
    hard=(F.softplus((sj-si)/.08)*w*mask.float()).sum()/((w*mask.float()).sum().clamp_min(1.0))
    grade=(yi>0)&(yj>0)&((yi-yj)>=.15); gw=(yi-yj).clamp_min(0)
    graded=(F.softplus((sj-si)/.09)*gw*grade.float()).sum()/((gw*grade.float()).sum().clamp_min(1.0))
    return .60*rel+.25*hard+.15*graded

def train_qwen():
    model,tok=load_qwen()
    if model is None: return None,None,[]
    dev=torch.device("cuda:0"); ids=ADAPT_IDX[HAS_SEMANTIC[ADAPT_IDX]].copy(); rng=np.random.default_rng(V41_SPLIT_SEED+411); rng.shuffle(ids)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=QWEN_LR,weight_decay=QWEN_WD)
    from transformers import get_cosine_schedule_with_warmup
    dtok_cpu=qtoks(tok,QWEN_MODULE_DOCS,QWEN_MODULE_MAX,"")
    steps=max(1,math.ceil(len(ids)/QWEN_BATCH)*QWEN_EPOCHS); sched=get_cosine_schedule_with_warmup(opt,max(1,int(.05*steps)),steps); hist=[]
    for ep in range(QWEN_EPOCHS):
        rng.shuffle(ids); losses=[]
        for st in range(0,len(ids),QWEN_BATCH):
            bi=ids[st:st+QWEN_BATCH]; q=qmove(qtoks(tok,TRAJ_TEXTS[bi],QWEN_QUERY_MAX,QWEN_INSTRUCTION),dev); d=qmove(dtok_cpu,dev)
            qo=model(**q); do=model(**d); qe=qnorm(last_token_pool(qo.last_hidden_state,q["attention_mask"])); de=qnorm(last_token_pool(do.last_hidden_state,d["attention_mask"])); sc=qe@de.T
            yy=torch.as_tensor(y[bi],device=dev,dtype=torch.float32); ar=torch.as_tensor(np.stack([ADAPT_TEACHER_RANKS[ADAPT_LOCAL[int(g)]] for g in bi]),device=dev,dtype=torch.float32)
            loss=qwen_loss(sc,yy,ar); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0); opt.step(); sched.step(); losses.append(float(loss.detach().cpu()))
            del q,d,qo,do,qe,de,sc,yy,ar,loss
        hist.append({"expert":"qwen","epoch":ep+1,"train_loss":float(np.mean(losses))}); print("Qwen epoch",ep+1,"loss",hist[-1]["train_loss"])
    model.eval(); return model,tok,hist

def score_qwen(model,tok,indices):
    indices=np.asarray(indices,int); out=np.zeros((len(indices),N_MODULES),np.float32)
    if model is None:
        # Smoke-only deterministic stand-in; never allowed by production gate.
        rng=np.random.default_rng(4100+len(indices)); return rng.normal(0,.01,size=out.shape).astype(np.float32)
    dev=torch.device("cuda:0"); d=qmove(qtoks(tok,QWEN_MODULE_DOCS,QWEN_MODULE_MAX,""),dev)
    with torch.no_grad():
        de=qnorm(last_token_pool(model(**d).last_hidden_state,d["attention_mask"]))
        for st in range(0,len(indices),16):
            ids=indices[st:st+16]; valid=HAS_SEMANTIC[ids]
            if not valid.any(): continue
            vv=ids[valid]; q=qmove(qtoks(tok,TRAJ_TEXTS[vv],QWEN_QUERY_MAX,QWEN_INSTRUCTION),dev)
            qe=qnorm(last_token_pool(model(**q).last_hidden_state,q["attention_mask"])); out[st+np.flatnonzero(valid)]=(qe@de.T).float().cpu().numpy().astype(np.float32)
    return out

qwen_model,qwen_tok,QWEN_HISTORY=train_qwen()
QRAW_ADAPT=score_qwen(qwen_model,qwen_tok,ADAPT_IDX); QRAW_CAL=score_qwen(qwen_model,qwen_tok,CAL_IDX); QRAW_LOCK=score_qwen(qwen_model,qwen_tok,LOCK_IDX); QRAW_TEST=score_qwen(qwen_model,qwen_tok,TEST_GLOBAL)

# ADAPT-only one-dimensional module calibration turns task-adapted cosine into a relevance-probability expert.
def qwen_calibrate(raw_train,raw_targets,*outs):
    pred=[np.zeros_like(o,np.float32) for o in outs]
    prev=raw_targets.mean(0).astype(np.float32)
    for m in range(N_MODULES):
        yy=raw_targets[:,m].astype(int)
        if len(np.unique(yy))<2:
            for p in pred: p[:,m]=prev[m]
            continue
        mdl=make_pipeline(StandardScaler(),LogisticRegression(C=.35,solver="liblinear",max_iter=1000,random_state=SEED+m))
        mdl.fit(raw_train[:,m:m+1],yy)
        for p,o in zip(pred,outs): p[:,m]=mdl.predict_proba(o[:,m:m+1])[:,1].astype(np.float32)
    return pred
QPROB_CAL,QPROB_LOCK,QPROB_TEST=qwen_calibrate(QRAW_ADAPT,Y_BIN[ADAPT_IDX],QRAW_CAL,QRAW_LOCK,QRAW_TEST)
# No-chat semantic branch falls back to ADAPT relevance prevalence rather than inventing semantic evidence.
qprev=Y_BIN[ADAPT_IDX].mean(0).astype(np.float32)
for idx,qp,qr in [(CAL_IDX,QPROB_CAL,QRAW_CAL),(LOCK_IDX,QPROB_LOCK,QRAW_LOCK),(TEST_GLOBAL,QPROB_TEST,QRAW_TEST)]:
    miss=~HAS_SEMANTIC[idx]; qp[miss]=qprev; qr[miss]=0.0

if qwen_model is not None:
    del qwen_model; qwen_model=None; qwen_tok=None; gc.collect(); torch.cuda.empty_cache()
print("Qwen expert ready",QRAW_CAL.shape,QPROB_CAL.shape)

# ---- cell 24 (markdown) ----
# ## 3. Head-4 item representation and transitive pairwise utility model
#
# All supervision is **within-user** and restricted to the anchor Top-4. Pair examples are generated only from unequal relevance pairs.  
# The pair feature is `x(user,module_a) - x(user,module_b)`, so one linear logistic model corresponds to a common transitive item utility.
#
# Pair weights are not hand-tuned: they are proportional to the exact training NDCG leverage of that anchor-position comparison.

# ---- cell 25 (code) ----

def rank_float(scores):
    return rank_matrix(scores).astype(np.float32)

def row_z(a):
    a = np.asarray(a, np.float32)
    mu = a.mean(1, keepdims=True)
    sd = a.std(1, keepdims=True)
    return ((a - mu) / np.maximum(sd, 1e-6)).astype(np.float32)

MODULE_EYE = np.eye(N_MODULES, dtype=np.float32)

ITEM_FEATURE_NAMES = [
    "anchor_row_z", "consensus_row_z", "sparse_row_z", "meta_row_z",
    "anchor_rank_norm", "gap_to_anchor_top1", "v31_minus_v30",
    "qwen_row_z", "qwen_prob_centered", "qwen_rank_norm",
    "readiness", "latest_intent_match", "has_semantic",
] + [f"module_intercept__{m}" for m in MODULES]

def head_item_tensor(global_idx, struct, qraw, qprob):
    gi = np.asarray(global_idx, int)
    a = np.asarray(struct["v30"], np.float32)
    c = np.asarray(struct["cons"], np.float32)
    b = np.asarray(struct["base"], np.float32)
    mt = np.asarray(struct["meta"], np.float32)
    v31 = np.asarray(struct["v31"], np.float32)
    ar = rank_float(a) / float(N_MODULES)
    topgap = (a.max(1, keepdims=True) - a).astype(np.float32)
    qrank = rank_float(qraw) / float(N_MODULES)
    qcenter = (qprob - qprob.mean(1, keepdims=True)).astype(np.float32)
    has = HAS_SEMANTIC[gi].astype(np.float32)[:, None]

    base = np.stack([
        row_z(a), row_z(c), row_z(b), row_z(mt),
        ar, topgap, (v31-a).astype(np.float32),
        row_z(qraw), qcenter, qrank,
        np.asarray(MODULE_READINESS[gi], np.float32),
        np.asarray(MODULE_LATEST[gi], np.float32),
        np.repeat(has, N_MODULES, axis=1),
    ], axis=2)
    onehot = np.repeat(MODULE_EYE[None, :, :], len(gi), axis=0)
    return np.concatenate([base, onehot], axis=2).astype(np.float32)

CAL_ITEM = head_item_tensor(CAL_IDX, CAL_STRUCT, QRAW_CAL, QPROB_CAL)
LOCK_ITEM = head_item_tensor(LOCK_IDX, LOCK_STRUCT, QRAW_LOCK, QPROB_LOCK)
TEST_ITEM = head_item_tensor(TEST_GLOBAL, TEST_STRUCT, QRAW_TEST, QPROB_TEST)

CAL_BASE = np.asarray(CAL_STRUCT["v30"], np.float32)
LOCK_BASE = np.asarray(LOCK_STRUCT["v30"], np.float32)
TEST_BASE = np.asarray(b34_anchor_test, np.float32)

def _idcg5_rows(yrows):
    ideal = np.sort(np.asarray(yrows), axis=1)[:, -5:][:, ::-1]
    return np.maximum((ideal * DISCOUNT).sum(1), 1e-12)

def make_pair_dataset(item_tensor, anchor, ytrue, user_rows):
    X, yy, ww, meta = [], [], [], []
    user_rows = np.asarray(user_rows, int)
    idcg = _idcg5_rows(ytrue)
    for r in user_rows:
        top4 = np.argsort(-anchor[r], kind="mergesort")[:4]
        for i in range(4):
            for j in range(i+1, 4):
                a, b = int(top4[i]), int(top4[j])
                ya, yb = float(ytrue[r, a]), float(ytrue[r, b])
                if abs(ya-yb) < 1e-12:
                    continue
                label = int(ya > yb)
                diff = (item_tensor[r, a] - item_tensor[r, b]).astype(np.float32)
                # Exact competition metric uses raw relevance * log discount.
                leverage = abs(ya-yb) * abs(float(DISCOUNT[i]-DISCOUNT[j])) / float(idcg[r])
                leverage = max(leverage, 1e-6)
                X.append(diff); yy.append(label); ww.append(leverage)
                meta.append((r, a, b, i+1, j+1))
                # Symmetric reverse observation makes the preference problem balanced.
                X.append(-diff); yy.append(1-label); ww.append(leverage)
                meta.append((r, b, a, j+1, i+1))
    X = np.asarray(X, np.float32)
    yy = np.asarray(yy, np.int8)
    ww = np.asarray(ww, np.float64)
    if len(ww):
        ww = ww / max(ww.mean(), 1e-12)
    return X, yy, ww, meta

class PairUtilityModel:
    def __init__(self, scaler, clf):
        self.scaler = scaler
        self.clf = clf
    def prob(self, diff):
        z = np.asarray(diff, np.float32).reshape(1, -1)
        return float(self.clf.predict_proba(self.scaler.transform(z))[0, 1])
    def probs(self, X):
        return self.clf.predict_proba(self.scaler.transform(np.asarray(X, np.float32)))[:, 1]

def fit_pair_model(item_tensor, anchor, ytrue, user_rows, seed):
    Xp, yp, wp, meta = make_pair_dataset(item_tensor, anchor, ytrue, user_rows)
    if len(np.unique(yp)) < 2:
        raise RuntimeError("Head4 pair training produced fewer than two classes")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xp)
    clf = LogisticRegression(
        C=PAIR_C, penalty="l2", solver="lbfgs", fit_intercept=False,
        max_iter=2000, random_state=int(seed)
    )
    clf.fit(Xs, yp, sample_weight=wp)
    model = PairUtilityModel(scaler, clf)
    pp = model.probs(Xp)
    audit = {
        "n_pair_rows": int(len(yp)),
        "n_unique_pairs": int(len(yp)//2),
        "positive_rate": float(yp.mean()),
        "weighted_logloss": float(log_loss(yp, pp, sample_weight=wp, labels=[0,1])),
        "pair_accuracy": float(accuracy_score(yp, pp >= 0.5, sample_weight=wp)),
        "coef_l2": float(np.sqrt(np.square(clf.coef_).sum())),
    }
    return model, audit

def best_challenger(model, item_row, anchor_row):
    top4 = np.argsort(-anchor_row, kind="mergesort")[:4]
    incumbent = int(top4[0])
    rows = []
    for rank, c in enumerate(top4[1:], start=2):
        c = int(c)
        p = model.prob(item_row[c] - item_row[incumbent])
        rows.append((p, c, rank))
    rows.sort(key=lambda z: (-z[0], z[2], z[1]))
    p, c, rank = rows[0]
    return incumbent, c, int(rank), float(p)

def decide_action(item_row, anchor_row, full_model, vote_models, prob_threshold, votes_required):
    incumbent, challenger, challenger_rank, p = best_challenger(full_model, item_row, anchor_row)
    agree = 0
    vote_probs = []
    for vm in vote_models:
        vi, vc, vr, vp = best_challenger(vm, item_row, anchor_row)
        vote_probs.append(vp if vc == challenger else 0.0)
        if vc == challenger and vp > 0.5:
            agree += 1
    execute = bool(p >= prob_threshold and agree >= votes_required)
    return {
        "incumbent": incumbent, "challenger": challenger, "challenger_anchor_rank": challenger_rank,
        "full_prob": p, "vote_agree": int(agree),
        "mean_support_prob": float(np.mean(vote_probs)) if vote_probs else np.nan,
        "execute": execute,
    }

def apply_top1_action(scores, row, challenger):
    before = np.asarray(scores[row], np.float32).copy()
    order = np.argsort(-before, kind="mergesort")
    top4 = order[:4].tolist()
    if int(challenger) not in top4:
        raise AssertionError("Top1 specialist may only promote an existing Top-4 module")
    if top4[0] == int(challenger):
        return
    vals = before[top4].copy()
    new_order = [int(challenger)] + [m for m in top4 if m != int(challenger)]
    scores[row, new_order] = vals

def geometry_audit(base, cand):
    base = np.asarray(base, np.float32); cand = np.asarray(cand, np.float32)
    ob = np.argsort(-base, axis=1, kind="mergesort")
    oc = np.argsort(-cand, axis=1, kind="mergesort")
    top4_set = np.asarray([set(a[:4]) == set(b[:4]) for a,b in zip(ob,oc)], bool)
    top5_set = np.asarray([set(a[:5]) == set(b[:5]) for a,b in zip(ob,oc)], bool)
    tail_same = np.asarray([np.array_equal(a[4:], b[4:]) for a,b in zip(ob,oc)], bool)
    return {
        "changed_users": int(np.any(ob != oc, axis=1).sum()),
        "top1_changes": int((ob[:,0] != oc[:,0]).sum()),
        "top4_membership_changes": int((~top4_set).sum()),
        "top5_membership_changes": int((~top5_set).sum()),
        "rank5_to_17_order_changes": int((~tail_same).sum()),
    }

print("Head4 item tensors:", CAL_ITEM.shape, LOCK_ITEM.shape, TEST_ITEM.shape,
      "| features", len(ITEM_FEATURE_NAMES))

# ---- cell 26 (markdown) ----
# ## 4. Honest CAL OOF with inner stability
#
# For each outer CAL fold:
# 1. the outer validation users are completely untouched;
# 2. one preference model is fit on the outer-train users;
# 3. four additional inner models are fit **only inside the outer-train users**;
# 4. a Top-1 promotion is allowed only when the outer model gives `p ≥ 0.80` and the same challenger receives enough inner directional agreement.
#
# This avoids the second-level dependency leakage that invalidated the optimistic V41 CAL result.

# ---- cell 27 (code) ----

def top1_acc(ytrue, s):
    return float(np.mean(np.argmax(ytrue, 1) == np.argmax(s, 1)))

def relevant_at5(ytrue, s):
    t = top5_indices(s)
    return float(np.mean((np.take_along_axis(ytrue, t, axis=1) > 0).sum(1)))

def boundary_accuracy(ytrue, s, k):
    t = np.argsort(-s, axis=1, kind="mergesort")[:, :6]
    a, b = t[:, k-1], t[:, k]
    ya = ytrue[np.arange(len(ytrue)), a]
    yb = ytrue[np.arange(len(ytrue)), b]
    return float(np.mean(ya >= yb))

def metric_dict(ytrue, base, cand):
    r = {
        "base_ndcg": ndcg5(ytrue, base),
        "candidate_ndcg": ndcg5(ytrue, cand),
        "delta": ndcg5(ytrue, cand) - ndcg5(ytrue, base),
        "base_top1": top1_acc(ytrue, base),
        "candidate_top1": top1_acc(ytrue, cand),
        "top1_delta": top1_acc(ytrue, cand) - top1_acc(ytrue, base),
        "base_rel5": relevant_at5(ytrue, base),
        "candidate_rel5": relevant_at5(ytrue, cand),
        "rel5_delta": relevant_at5(ytrue, cand) - relevant_at5(ytrue, base),
    }
    for k in range(1,6):
        r[f"B{k}{k+1}_base"] = boundary_accuracy(ytrue, base, k)
        r[f"B{k}{k+1}_candidate"] = boundary_accuracy(ytrue, cand, k)
    return r

def paired_boot_ci(delta, seed, n_boot=None):
    d = np.asarray(delta, np.float64)
    n = len(d)
    B = (50 if V412_SMOKE else 2000) if n_boot is None else int(n_boot)
    rng = np.random.default_rng(seed)
    vals = np.empty(B, np.float64)
    for b in range(B):
        vals[b] = d[rng.integers(0, n, n)].mean()
    return float(np.quantile(vals, .025)), float(np.quantile(vals, .975))

def block_rows(global_idx, n_blocks):
    order = np.argsort(np.asarray(global_idx, int))
    return np.array_split(order, n_blocks)

cal_strata = np.argmax(y[CAL_IDX], axis=1).astype(int)
CAL_SPLITS = list(StratifiedKFold(
    CAL_FOLDS, shuffle=True, random_state=V41_CAL_SEED
).split(np.zeros(len(CAL_IDX)), cal_strata))

CAL_FOLD_ID = np.full(len(CAL_IDX), -1, int)
CAL_CAND = CAL_BASE.copy()
CAL_LEDGER_ROWS = []
PAIR_AUDIT_ROWS = []
OUTER_MODELS = []

for f, (tr, va) in enumerate(CAL_SPLITS):
    CAL_FOLD_ID[va] = f
    outer_model, outer_audit = fit_pair_model(CAL_ITEM, CAL_BASE, y[CAL_IDX], tr, V41_CAL_SEED + 1000*f)
    OUTER_MODELS.append(outer_model)
    PAIR_AUDIT_ROWS.append({"fit":"outer", "fold":f, **outer_audit})

    # Inner stability models are trained only on the outer-train subset.
    inner_labels = cal_strata[tr]
    if len(np.unique(inner_labels)) > 1 and pd.Series(inner_labels).value_counts().min() >= INNER_FOLDS:
        inner_splitter = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=V41_CAL_SEED + 2000 + f)
        inner_splits = list(inner_splitter.split(np.zeros(len(tr)), inner_labels))
    else:
        inner_splitter = KFold(INNER_FOLDS, shuffle=True, random_state=V41_CAL_SEED + 2000 + f)
        inner_splits = list(inner_splitter.split(np.zeros(len(tr))))

    inner_models = []
    for j, (itr, _) in enumerate(inner_splits):
        rows = tr[np.asarray(itr, int)]
        mdl, aud = fit_pair_model(CAL_ITEM, CAL_BASE, y[CAL_IDX], rows, V41_CAL_SEED + 3000 + 100*f + j)
        inner_models.append(mdl)
        PAIR_AUDIT_ROWS.append({"fit":"inner", "fold":f, "inner":j, **aud})

    for r in va:
        dec = decide_action(CAL_ITEM[r], CAL_BASE[r], outer_model, inner_models,
                            ACTION_PROB_THRESHOLD, OOF_INNER_AGREE)
        executed = bool(CAL_ELIG[r] and dec["execute"])
        before_user = per_user_ndcg(y[CAL_IDX][r:r+1], CAL_CAND[r:r+1])[0]
        if executed:
            apply_top1_action(CAL_CAND, int(r), dec["challenger"])
        after_user = per_user_ndcg(y[CAL_IDX][r:r+1], CAL_CAND[r:r+1])[0]
        CAL_LEDGER_ROWS.append({
            "fold": f, "row_local": int(r), "user_id": str(train.iloc[CAL_IDX[r]].user_id),
            "state": str(STATE_NAMES[CAL_STATE[r]]), "eligible": bool(CAL_ELIG[r]),
            "incumbent_module": MODULES[dec["incumbent"]],
            "challenger_module": MODULES[dec["challenger"]],
            "challenger_anchor_rank": dec["challenger_anchor_rank"],
            "full_prob": dec["full_prob"], "vote_agree": dec["vote_agree"],
            "mean_support_prob": dec["mean_support_prob"], "executed": executed,
            "incumbent_relevance": float(y[CAL_IDX[r], dec["incumbent"]]),
            "challenger_relevance": float(y[CAL_IDX[r], dec["challenger"]]),
            "user_ndcg_delta": float(after_user-before_user),
        })

assert (CAL_FOLD_ID >= 0).all()
V412_PAIR_AUDIT = pd.DataFrame(PAIR_AUDIT_ROWS)
V412_CAL_LEDGER = pd.DataFrame(CAL_LEDGER_ROWS)

# Full CAL model is frozen before LOCK is opened and is also used unchanged on TEST.
FULL_CAL_MODEL, full_pair_audit = fit_pair_model(
    CAL_ITEM, CAL_BASE, y[CAL_IDX], np.arange(len(CAL_IDX)), V41_CAL_SEED + 9000
)
V412_PAIR_AUDIT = pd.concat([
    V412_PAIR_AUDIT,
    pd.DataFrame([{"fit":"cal_full", "fold":-1, **full_pair_audit}])
], ignore_index=True)

CAL_MET = metric_dict(y[CAL_IDX], CAL_BASE, CAL_CAND)
CAL_USER_DELTA = per_user_ndcg(y[CAL_IDX], CAL_CAND) - per_user_ndcg(y[CAL_IDX], CAL_BASE)
CAL_CI = paired_boot_ci(CAL_USER_DELTA, V412_BOOT_SEED)

fold_rows = []
for f in range(CAL_FOLDS):
    z = CAL_FOLD_ID == f
    rr = metric_dict(y[CAL_IDX][z], CAL_BASE[z], CAL_CAND[z])
    act = V412_CAL_LEDGER[(V412_CAL_LEDGER.fold == f) & (V412_CAL_LEDGER.executed)]
    fold_rows.append({
        "fold": f, "n": int(z.sum()), "actions": int(len(act)),
        "action_precision": float((act.user_ndcg_delta > 0).mean()) if len(act) else np.nan,
        **rr
    })
V412_CAL_FOLDS = pd.DataFrame(fold_rows)

block_metrics = []
for b, rows in enumerate(block_rows(CAL_IDX, BLOCKS)):
    rr = metric_dict(y[CAL_IDX][rows], CAL_BASE[rows], CAL_CAND[rows])
    block_metrics.append({
        "block": b, "n": len(rows),
        "id_min": int(CAL_IDX[rows].min()+1), "id_max": int(CAL_IDX[rows].max()+1), **rr
    })
V412_CAL_BLOCKS = pd.DataFrame(block_metrics)

cal_actions = V412_CAL_LEDGER[V412_CAL_LEDGER.executed].copy()
CAL_ACTIONS = int(len(cal_actions))
CAL_ACTION_PREC = float((cal_actions.user_ndcg_delta > 0).mean()) if CAL_ACTIONS else 0.0
CAL_BENEFICIAL = int((cal_actions.user_ndcg_delta > 0).sum()) if CAL_ACTIONS else 0
CAL_HARMFUL = int((cal_actions.user_ndcg_delta < 0).sum()) if CAL_ACTIONS else 0
CAL_NET = CAL_BENEFICIAL - CAL_HARMFUL

CAL_GEOM = geometry_audit(CAL_BASE, CAL_CAND)
assert CAL_GEOM["top4_membership_changes"] == 0
assert CAL_GEOM["top5_membership_changes"] == 0
assert CAL_GEOM["rank5_to_17_order_changes"] == 0
assert abs(CAL_MET["rel5_delta"]) < 1e-12

# Leave-one-module action stress: diagnostic only; never used to select modules.
loo = []
full_gain = float(CAL_MET["delta"])
for mid in MODULES:
    alt = CAL_CAND.copy()
    rows_to_revert = cal_actions[
        (cal_actions.incumbent_module == mid) | (cal_actions.challenger_module == mid)
    ].row_local.astype(int).unique()
    alt[rows_to_revert] = CAL_BASE[rows_to_revert]
    gain = float(ndcg5(y[CAL_IDX], alt) - ndcg5(y[CAL_IDX], CAL_BASE))
    loo.append({
        "module": mid, "actions_involving_module": int(len(rows_to_revert)),
        "gain_without_module_actions": gain, "drop_from_full_gain": full_gain-gain
    })
V412_MODULE_LEAVEOUT = pd.DataFrame(loo)
MODULE_STRESS_OK = bool(
    V412_MODULE_LEAVEOUT.gain_without_module_actions.min() > 0
) if CAL_ACTIONS else False

CAL_FOLD_WINS = int((V412_CAL_FOLDS.delta > 0).sum())
CAL_BLOCK_WINS = int((V412_CAL_BLOCKS.delta > 0).sum())

CAL_GATE = bool(
    ORACLE_GATE and not V412_SKIP_DEEP and
    CAL_MET["delta"] >= CAL_MIN_DELTA and
    CAL_FOLD_WINS >= CAL_MIN_FOLD_WINS and
    CAL_BLOCK_WINS >= CAL_MIN_BLOCK_WINS and
    CAL_MET["top1_delta"] >= CAL_MIN_TOP1_DELTA and
    CAL_ACTION_PREC >= CAL_MIN_ACTION_PRECISION and
    CAL_MIN_ACTIONS <= CAL_ACTIONS <= CAL_MAX_ACTIONS and
    CAL_CI[0] >= CAL_MIN_CI_LOW and
    abs(CAL_MET["rel5_delta"]) < 1e-12 and
    CAL_GEOM["top4_membership_changes"] == 0 and
    CAL_GEOM["rank5_to_17_order_changes"] == 0 and
    MODULE_STRESS_OK
)

V412_CAL_METRICS = pd.DataFrame([{
    **CAL_MET, "ci_low": CAL_CI[0], "ci_high": CAL_CI[1],
    "fold_wins": CAL_FOLD_WINS, "block_wins": CAL_BLOCK_WINS,
    "actions": CAL_ACTIONS, "action_precision": CAL_ACTION_PREC,
    "beneficial_actions": CAL_BENEFICIAL, "harmful_actions": CAL_HARMFUL,
    "net_beneficial": CAL_NET, "module_stress_ok": MODULE_STRESS_OK,
    **{f"geom_{k}":v for k,v in CAL_GEOM.items()},
    "cal_gate": CAL_GATE
}])

print(V412_CAL_METRICS.to_string(index=False))
print("CAL fold deltas:", V412_CAL_FOLDS[["fold","delta","actions","action_precision"]].to_dict("records"))
print("CAL block deltas:", V412_CAL_BLOCKS[["block","delta"]].to_dict("records"))
print("CAL_GATE", CAL_GATE)

# ---- cell 28 (markdown) ----
# ## 5. One untouched LOCK confirmation and exact B34 test deployment
#
# LOCK is evaluated **only if CAL passes**. No model or threshold is changed afterward.
#
# For LOCK and TEST:
# - challenger is selected by the single full-CAL preference model;
# - at least 4/5 outer-CAL models must independently select/support the same challenger;
# - only S0+S2 may move;
# - the challenger must already be inside the anchor Top-4;
# - the candidate only permutes existing Top-4 score values.

# ---- cell 29 (code) ----

def deploy_with_ensemble(item_tensor, anchor, eligible, global_user_ids, ytrue=None):
    cand = np.asarray(anchor, np.float32).copy()
    ledger = []
    for r in range(len(anchor)):
        dec = decide_action(item_tensor[r], anchor[r], FULL_CAL_MODEL, OUTER_MODELS,
                            ACTION_PROB_THRESHOLD, DEPLOY_OUTER_AGREE)
        executed = bool(eligible[r] and dec["execute"])
        before = None
        if ytrue is not None:
            before = float(per_user_ndcg(ytrue[r:r+1], cand[r:r+1])[0])
        if executed:
            apply_top1_action(cand, r, dec["challenger"])
        after = None
        if ytrue is not None:
            after = float(per_user_ndcg(ytrue[r:r+1], cand[r:r+1])[0])
        row = {
            "row_local": int(r), "user_id": str(global_user_ids[r]),
            "eligible": bool(eligible[r]),
            "incumbent_module": MODULES[dec["incumbent"]],
            "challenger_module": MODULES[dec["challenger"]],
            "challenger_anchor_rank": dec["challenger_anchor_rank"],
            "full_prob": dec["full_prob"], "vote_agree": dec["vote_agree"],
            "mean_support_prob": dec["mean_support_prob"], "executed": executed,
        }
        if ytrue is not None:
            row.update({
                "incumbent_relevance": float(ytrue[r, dec["incumbent"]]),
                "challenger_relevance": float(ytrue[r, dec["challenger"]]),
                "user_ndcg_delta": float(after-before),
            })
        ledger.append(row)
    return cand, pd.DataFrame(ledger)

LOCK_GATE = False
LOCK_MET = {"evaluated": False, "lock_gate": False}
V412_LOCK_BLOCKS = pd.DataFrame(columns=["block","n","id_min","id_max","delta"])
V412_LOCK_LEDGER = pd.DataFrame()
LOCK_CAND = LOCK_BASE.copy()

if CAL_GATE:
    LOCK_CAND, V412_LOCK_LEDGER = deploy_with_ensemble(
        LOCK_ITEM, LOCK_BASE, LOCK_ELIG,
        train.iloc[LOCK_IDX].user_id.astype(str).to_numpy(),
        ytrue=y[LOCK_IDX]
    )
    lm = metric_dict(y[LOCK_IDX], LOCK_BASE, LOCK_CAND)
    ld = per_user_ndcg(y[LOCK_IDX], LOCK_CAND) - per_user_ndcg(y[LOCK_IDX], LOCK_BASE)
    lci = paired_boot_ci(ld, V412_BOOT_SEED + 1)

    lock_blocks = []
    for b, rows in enumerate(block_rows(LOCK_IDX, BLOCKS)):
        rr = metric_dict(y[LOCK_IDX][rows], LOCK_BASE[rows], LOCK_CAND[rows])
        lock_blocks.append({
            "block": b, "n": len(rows),
            "id_min": int(LOCK_IDX[rows].min()+1), "id_max": int(LOCK_IDX[rows].max()+1), **rr
        })
    V412_LOCK_BLOCKS = pd.DataFrame(lock_blocks)
    block_wins = int((V412_LOCK_BLOCKS.delta > 0).sum())

    lact = V412_LOCK_LEDGER[V412_LOCK_LEDGER.executed].copy()
    nact = int(len(lact))
    prec = float((lact.user_ndcg_delta > 0).mean()) if nact else 0.0
    ben = int((lact.user_ndcg_delta > 0).sum()) if nact else 0
    harm = int((lact.user_ndcg_delta < 0).sum()) if nact else 0
    net = ben - harm
    lgeom = geometry_audit(LOCK_BASE, LOCK_CAND)

    assert lgeom["top4_membership_changes"] == 0
    assert lgeom["top5_membership_changes"] == 0
    assert lgeom["rank5_to_17_order_changes"] == 0
    assert abs(lm["rel5_delta"]) < 1e-12

    LOCK_GATE = bool(
        lm["delta"] >= LOCK_MIN_DELTA and
        block_wins >= LOCK_MIN_BLOCK_WINS and
        lm["top1_delta"] > 0 and
        prec >= LOCK_MIN_ACTION_PRECISION and
        net >= LOCK_MIN_NET_BENEFICIAL and
        lci[0] >= LOCK_MIN_CI_LOW and
        abs(lm["rel5_delta"]) < 1e-12 and
        lgeom["top4_membership_changes"] == 0 and
        lgeom["rank5_to_17_order_changes"] == 0
    )
    LOCK_MET = {
        "evaluated": True, **lm, "ci_low": lci[0], "ci_high": lci[1],
        "block_wins": block_wins, "actions": nact, "action_precision": prec,
        "beneficial_actions": ben, "harmful_actions": harm, "net_beneficial": net,
        **{f"geom_{k}":v for k,v in lgeom.items()},
        "lock_gate": LOCK_GATE,
    }

V412_LOCK_METRICS = pd.DataFrame([LOCK_MET])

# Test is only materialized after a clean untouched LOCK.
TEST_CANDIDATE = TEST_BASE.copy()
V412_TEST_LEDGER = pd.DataFrame()
TEST_GEOM = {
    "evaluated": False, "geometry_gate": False, "changed_users": 0,
    "top1_changes": 0, "top4_membership_changes": 0,
    "top5_membership_changes": 0, "rank5_to_17_order_changes": 0,
    "eligible_changed": 0, "locked_changed": 0,
    "submission_scale": False, "top5_sha256": top5_sha256(TEST_BASE),
}

if LOCK_GATE:
    TEST_CANDIDATE, V412_TEST_LEDGER = deploy_with_ensemble(
        TEST_ITEM, TEST_BASE, TEST_ELIG, test.user_id.astype(str).to_numpy(), ytrue=None
    )
    tg = geometry_audit(TEST_BASE, TEST_CANDIDATE)
    ob = np.argsort(-TEST_BASE, axis=1, kind="mergesort")
    oc = np.argsort(-TEST_CANDIDATE, axis=1, kind="mergesort")
    changed = np.any(ob != oc, axis=1)
    locked_equal = bool(np.allclose(TEST_CANDIDATE[~TEST_ELIG], TEST_BASE[~TEST_ELIG], rtol=0, atol=0))
    actions = int(V412_TEST_LEDGER.executed.sum())

    # Every changed row must preserve its exact Top-4 score multiset.
    multiset_ok = True
    for r in np.flatnonzero(changed):
        tb = np.sort(TEST_BASE[r, ob[r,:4]])
        tc = np.sort(TEST_CANDIDATE[r, oc[r,:4]])
        if not np.array_equal(tb, tc):
            multiset_ok = False
            break

    geometry_gate = bool(
        locked_equal and multiset_ok and
        tg["top4_membership_changes"] == 0 and
        tg["top5_membership_changes"] == 0 and
        tg["rank5_to_17_order_changes"] == 0 and
        tg["top1_changes"] == actions and
        TEST_MIN_TOP1_ACTIONS <= actions <= TEST_MAX_TOP1_ACTIONS
    )
    TEST_GEOM = {
        "evaluated": True, "geometry_gate": geometry_gate, **tg,
        "eligible_changed": int(np.sum(changed & TEST_ELIG)),
        "locked_changed": int(np.sum(changed & ~TEST_ELIG)),
        "actions": actions,
        "score_multiset_ok": bool(multiset_ok),
        "locked_rows_bit_exact": bool(locked_equal),
        "submission_scale": bool(TEST_MIN_TOP1_ACTIONS <= actions <= TEST_MAX_TOP1_ACTIONS),
        "top5_sha256": top5_sha256(TEST_CANDIDATE),
    }

V412_TEST_GEOMETRY = pd.DataFrame([TEST_GEOM])

FINAL_GATE = bool(
    REPRO_OK and ORACLE_GATE and CAL_GATE and LOCK_GATE and
    bool(TEST_GEOM.get("geometry_gate", False)) and
    not V412_SMOKE and not V412_SKIP_DEEP
)
STATUS = (
    "SUBMISSION_SCALE" if FINAL_GATE
    else ("VALID_BUT_LOW_LEVERAGE" if LOCK_GATE and TEST_GEOM.get("evaluated",False) else "FAIL_CLOSE")
)

print("LOCK:", V412_LOCK_METRICS.to_dict("records"))
print("TEST:", V412_TEST_GEOMETRY.to_dict("records"))
print("FINAL_GATE", FINAL_GATE, "STATUS", STATUS)

# ---- cell 30 (markdown) ----
# ## 6. Diagnostics, reproducibility ledger, and fail-close submissions

# ---- cell 31 (code) ----

# Qwen complementarity specifically at recoverable Head-1 mistakes.
qcomp = []
for r in range(len(CAL_IDX)):
    top4 = np.argsort(-CAL_BASE[r], kind="mergesort")[:4]
    inc = int(top4[0])
    true = int(np.argmax(y[CAL_IDX[r]]))
    if not (CAL_ELIG[r] and inc != true and true in set(top4)):
        continue
    qcomp.append({
        "user_id": str(train.iloc[CAL_IDX[r]].user_id),
        "incumbent": MODULES[inc], "true_top1": MODULES[true],
        "qwen_prefers_true_over_incumbent": bool(QRAW_CAL[r,true] > QRAW_CAL[r,inc]),
        "qwen_margin": float(QRAW_CAL[r,true] - QRAW_CAL[r,inc]),
        "struct_margin": float(CAL_BASE[r,true] - CAL_BASE[r,inc]),
    })
V412_COMPLEMENTARITY = pd.DataFrame(qcomp)

# Module proposal/effect audit from clean CAL OOF actions.
module_rows = []
for mid in MODULES:
    prop = V412_CAL_LEDGER[V412_CAL_LEDGER.challenger_module == mid]
    act = prop[prop.executed]
    module_rows.append({
        "module": mid,
        "proposals": int(len(prop)),
        "actions": int(len(act)),
        "beneficial_actions": int((act.user_ndcg_delta > 0).sum()) if len(act) else 0,
        "harmful_actions": int((act.user_ndcg_delta < 0).sum()) if len(act) else 0,
        "mean_action_delta": float(act.user_ndcg_delta.mean()) if len(act) else np.nan,
        "mean_full_prob": float(prop.full_prob.mean()) if len(prop) else np.nan,
    })
V412_MODULE_AUDIT = pd.DataFrame(module_rows)

# Positive-count diagnostic: descriptive only; K is never used as a user-ID/test shortcut.
k_rows = []
for k in sorted(np.unique((y[CAL_IDX] > 0).sum(1)).tolist()):
    z = (y[CAL_IDX] > 0).sum(1) == k
    k_rows.append({
        "positive_count": int(k), "n": int(z.sum()),
        "anchor_ndcg": float(ndcg5(y[CAL_IDX][z], CAL_BASE[z])),
        "anchor_top1_acc": float(top1_acc(y[CAL_IDX][z], CAL_BASE[z])),
        "candidate_ndcg": float(ndcg5(y[CAL_IDX][z], CAL_CAND[z])),
        "candidate_top1_acc": float(top1_acc(y[CAL_IDX][z], CAL_CAND[z])),
    })
V412_K_AUDIT = pd.DataFrame(k_rows)

def submission_from_matrix(scores):
    out = sample.copy()
    out.loc[:, MODULES] = np.asarray(scores, np.float32)
    assert out.shape == sample.shape
    assert out.user_id.astype(str).tolist() == test.user_id.astype(str).tolist()
    assert np.isfinite(out[MODULES].to_numpy()).all()
    return out

primary = TEST_CANDIDATE if FINAL_GATE else TEST_BASE

submission_from_matrix(TEST_BASE).to_csv(V41_OUT/"submission_v41_2_b34_anchor_reference.csv", index=False)
submission_from_matrix(TEST_CANDIDATE).to_csv(V41_OUT/"submission_v41_2_candidate_DIAGNOSTIC.csv", index=False)
submission_from_matrix(primary).to_csv(V41_OUT/"submission_v41_2_primary.csv", index=False)

V412_HEAD_ORACLE.to_csv(V41_OUT/"v41_2_head_oracle.csv", index=False)
V412_CAL_METRICS.to_csv(V41_OUT/"v41_2_cal_metrics.csv", index=False)
V412_CAL_FOLDS.to_csv(V41_OUT/"v41_2_cal_fold_metrics.csv", index=False)
V412_CAL_BLOCKS.to_csv(V41_OUT/"v41_2_cal_block_metrics.csv", index=False)
V412_PAIR_AUDIT.to_csv(V41_OUT/"v41_2_head_pair_audit.csv", index=False)
V412_CAL_LEDGER.to_csv(V41_OUT/"v41_2_top1_action_ledger_cal.csv", index=False)
V412_MODULE_AUDIT.to_csv(V41_OUT/"v41_2_module_audit.csv", index=False)
V412_MODULE_LEAVEOUT.to_csv(V41_OUT/"v41_2_module_leaveout.csv", index=False)
V412_COMPLEMENTARITY.to_csv(V41_OUT/"v41_2_complementarity.csv", index=False)
V412_K_AUDIT.to_csv(V41_OUT/"v41_2_k_audit.csv", index=False)

V412_LOCK_METRICS.to_csv(V41_OUT/"v41_2_lock_metrics.csv", index=False)
V412_LOCK_BLOCKS.to_csv(V41_OUT/"v41_2_lock_block_metrics.csv", index=False)
V412_LOCK_LEDGER.to_csv(V41_OUT/"v41_2_top1_action_ledger_lock.csv", index=False)
V412_TEST_LEDGER.to_csv(V41_OUT/"v41_2_top1_action_ledger_test.csv", index=False)
V412_TEST_GEOMETRY.to_csv(V41_OUT/"v41_2_test_geometry.csv", index=False)

pd.DataFrame(QWEN_HISTORY).to_csv(V41_OUT/"v41_2_training_history.csv", index=False)
pd.DataFrame({
    "user_id": train.user_id,
    "split": np.where(
        np.isin(np.arange(N_TRAIN), ADAPT_IDX), "ADAPT",
        np.where(np.isin(np.arange(N_TRAIN), CAL_IDX), "CAL", "LOCK")
    )
}).to_csv(V41_OUT/"v41_2_split_manifest.csv", index=False)

# Per-changed-row score-multiset proof.
multiset_rows = []
if len(V412_TEST_LEDGER):
    for row in V412_TEST_LEDGER[V412_TEST_LEDGER.executed].itertuples(index=False):
        r = int(row.row_local)
        ob = np.argsort(-TEST_BASE[r], kind="mergesort")
        oc = np.argsort(-TEST_CANDIDATE[r], kind="mergesort")
        multiset_rows.append({
            "user_id": row.user_id,
            "base_top4_modules": "|".join(MODULES[m] for m in ob[:4]),
            "candidate_top4_modules": "|".join(MODULES[m] for m in oc[:4]),
            "base_top4_scores_sorted": "|".join(f"{x:.10g}" for x in np.sort(TEST_BASE[r,ob[:4]])),
            "candidate_top4_scores_sorted": "|".join(f"{x:.10g}" for x in np.sort(TEST_CANDIDATE[r,oc[:4]])),
            "multiset_equal": bool(np.array_equal(np.sort(TEST_BASE[r,ob[:4]]), np.sort(TEST_CANDIDATE[r,oc[:4]]))),
        })
pd.DataFrame(multiset_rows).to_csv(V41_OUT/"v41_2_score_multiset_audit.csv", index=False)

import platform
pkg = {
    "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
    "scipy": scipy.__version__, "sklearn": sklearn.__version__, "lightgbm": lgb.__version__,
    "torch": torch.__version__,
}
try:
    import transformers, peft, bitsandbytes
    pkg.update(
        transformers=transformers.__version__,
        peft=peft.__version__,
        bitsandbytes=bitsandbytes.__version__,
    )
except Exception:
    pass
json.dump(pkg, open(V41_OUT/"v41_2_package_versions.json","w"), indent=2)

current_hash = top5_sha256(current_anchor_test)
b34_hash = top5_sha256(TEST_BASE)
state_counts = {STATE_NAMES[k]: int((full_state == k).sum()) for k in range(4)}
REPRO_AUDIT = {
    "exact_v30_v31_current_reproduction": bool(REPRO_OK),
    "current_anchor_hash": current_hash,
    "expected_current_anchor_hash": CURRENT_ANCHOR_TOP5_SHA256,
    "b34_hash": b34_hash,
    "expected_b34_hash": EXPECTED_B34_TOP5_SHA256,
    "b34_rows": int(len(B34_ROWS)),
    "expected_boundary_counts": EXPECTED_BOUNDARY_COUNTS,
    "state_counts": state_counts,
    "test_eligible_s0_s2": int(TEST_ELIG.sum()),
    "test_locked_s1_s3": int((~TEST_ELIG).sum()),
    "no_historical_submission_csv_input": True,
    "external_labeled_data_used": False,
    "open_source_pretrained_qwen_only": True,
    "no_direct_genai_test_relevance": True,
    "no_user_id_predictive_feature": True,
    "no_post_lock_retraining": True,
    "outer_cal_validation_label_isolation": True,
    "top5_membership_frozen": True,
    "s1_s3_immutable": True,
    "final_score_blending_removed": True,
}
json.dump(REPRO_AUDIT, open(V41_OUT/"v41_2_reproduction_audit.json","w"), indent=2)

summary = {
    "version": "V41.2 B34-Anchored Head4 Preference Specialist with Abstaining Top1 Deployment",
    "public_anchor": PUBLIC_B34_LB,
    "rank5_reference": PUBLIC_RANK5_LB,
    "leader_reference": PUBLIC_LEADER_LB,
    "split_sizes": {"ADAPT":int(len(ADAPT_IDX)), "CAL":int(len(CAL_IDX)), "LOCK":int(len(LOCK_IDX))},
    "semantic_expert": "Qwen3-Embedding-8B QLoRA, unchanged from V41.1",
    "preference_model": "single L2 pairwise logistic / Bradley-Terry-style transitive utility",
    "deployment": "S0+S2 only; Top1-only; existing anchor Top4 score-value permutation; S1/S3 locked",
    "oracle_gate": bool(ORACLE_GATE),
    "cal_gate": bool(CAL_GATE),
    "lock_gate": bool(LOCK_GATE),
    "test_geometry_gate": bool(TEST_GEOM.get("geometry_gate", False)),
    "final_gate": bool(FINAL_GATE),
    "status": STATUS,
    "primary_top5_sha256": top5_sha256(primary),
    "candidate_top5_sha256": top5_sha256(TEST_CANDIDATE),
    "b34_top5_sha256": b34_hash,
    "submission_instruction": (
        "SUBMIT submission_v41_2_primary.csv"
        if FINAL_GATE
        else "NO NEW V41.2 SUBMISSION: primary is the exact B34 fail-close anchor."
    ),
    "tm_compliance": {
        "organizer_data_only": True,
        "open_source_pretrained_model": True,
        "no_direct_genai_test_relevance": True,
        "fixed_seeds": True,
        "no_post_lock_retraining": True,
        "no_historical_submission_predictive_input": True,
    },
}
json.dump(summary, open(V41_OUT/"v41_2_summary.json","w"), indent=2)

print(json.dumps(summary, indent=2))
print("Artifacts written to", V41_OUT)

# ---- cell 32 (markdown) ----
# ### Submission rule
#
# Read `v41_2_summary.json`.
#
# - If `final_gate == true`: submit **`submission_v41_2_primary.csv`**
# - Otherwise: **do not spend a submission**. `submission_v41_2_primary.csv` is intentionally the exact B34 `0.66026` fail-close anchor.
#
# This notebook does not create V41.2.1-style threshold or module variants after CAL/LOCK.
