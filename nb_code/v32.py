# Converted from notebooks/v32.ipynb: markdown cells become comments, %/! magics neutralized.

# ---- cell 0 (markdown) ----
# # MineToday V32 — Cross-Objective Residual Consensus LTR
#
# **Last-shot thesis:** V30 public `0.65963` remains the immutable control/fallback. V31 established that the frozen hierarchical Intent-State block is valid and directionally useful but only weakly incremental under the same LambdaRank learner. V32 therefore keeps the representation frozen and changes the **ranking objective/model family**, while preserving V30 as the decision prior.
#
# The only new experts are:
#
# - **CatBoost YetiRank-NDCG**, fixed to `mode=NDCG`, `top=5`, `dcg_type=Base`, `LogPosition`, using the historical V6 regularization regime without hyperparameter search.
# - **LightGBM XE_NDCG_MART** via `rank_xendcg`, using the same capacity and the same frozen Intent-State feature surface.
#
# Both experts are aligned back to the frozen anchor geometry and use the exact public-proven residual strength `alpha = 0.16`.
#
# Two predeclared deployment candidates are evaluated:
#
# 1. **ObjectiveMedian** — elementwise median of the anchor-scale V30, YetiRank-residual, and XENDCG-residual scores. This is the broad candidate and is eligible only under a V30-class robustness gate.
# 2. **UnanimousTop6** — starts from exact V30 and performs at most one adjacent Top-6 swap per user when YetiRank and XENDCG both disagree with V30 in the same direction. For the NDCG-critical `rank 5 ↔ 6` boundary, the frozen V31 Intent-State LambdaRank must also agree. If several swaps are eligible, only the smallest V30-margin pair is changed. Otherwise it abstains to V30.
#
# ### Precommitted experiment contract
#
# - Exact V30 Base LambdaRank is the immutable control and exact fail-close submission.
# - V30 base features and all 16 V31 Intent-State features are frozen. No new feature engineering.
# - Intent-State smoothing remains exactly `parent=40`, `child=30`; target encoding remains strict nested OOF.
# - Residual strength remains exactly `alpha=0.16`; no alpha search.
# - Fresh development worlds are **K/L/M only**. G/H/I and C/D/E/F are historical and forbidden selectors.
# - The previously untouched **sealed J** partition is preserved exactly and opened only after a development winner is frozen. Chronology is rejection-only.
# - Broad candidates are `YetiResidual`, `XEResidual`, and `ObjectiveMedian`. Broad selection is max-min robust: highest worst-salt among candidates that pass the precommitted broad gate.
# - If no broad candidate passes, `UnanimousTop6` may be selected only through its separate conservative gate.
# - No learned router, no semantic model, no curriculum/co-occurrence branch, no transition veto, no smoothing search, no extra salts.
# - The eight previously public-tested V25 rows remain protected exactly in every deployment surface.
# - If any final gate fails, `submission_v32_primary.csv` is an exact V30 fail-close.

# ---- cell 1 (code) ----
# Runtime + preregistered V32 configuration
import os, sys, re, json, ast, html, hashlib, random, warnings, time, subprocess, gc
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ["PYTHONHASHSEED"] = "42"

V32_SKIP_INSTALL = bool(int(os.getenv("V32_SKIP_INSTALL", "0")))
try:
    import lightgbm as lgb
except Exception:
    if V32_SKIP_INSTALL:
        raise
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "lightgbm==4.6.0"])
    import lightgbm as lgb

try:
    from catboost import CatBoostRanker, Pool
    import catboost
except Exception:
    if V32_SKIP_INSTALL:
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
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")

SEED = 42
V32_SMOKE = bool(int(os.getenv("V32_SMOKE", "0")))
V32_STRICT_GEOMETRY = bool(int(os.getenv("V32_STRICT_GEOMETRY", "1")))

# Compatibility aliases consumed by the frozen V3.1/V25 reconstruction cells.
V31_SMOKE = V32_SMOKE
V30_SMOKE = V32_SMOKE
V30_STRICT_GEOMETRY = V32_STRICT_GEOMETRY
V29_SMOKE = V32_SMOKE
V29_STRICT_GEOMETRY = V32_STRICT_GEOMETRY
V27_SMOKE = V32_SMOKE
V27_STRICT_GEOMETRY = V32_STRICT_GEOMETRY

N_FOLDS = 2 if V32_SMOKE else 5
ANCHOR_INNER_FOLDS = 2 if V32_SMOKE else 4
INTENT_INNER_FOLDS = 2 if V32_SMOKE else 4

RIDGE_ALPHA = 3.0
REFERENCE_W = 0.45

PUBLIC_CAL045_LB = 0.65639
PUBLIC_CONSENSUS3_LB = 0.65856
PUBLIC_V25_LB = 0.65919
PUBLIC_V30_LB = 0.65963
PUBLIC_RANK4_LB = 0.65974
PUBLIC_LEADER_LB = 0.66141

LGB_ESTIMATORS = 12 if V32_SMOKE else 1200
LGB_EARLY_STOP = 3 if V32_SMOKE else 80
RANKER_ESTIMATORS = 16 if V32_SMOKE else 360
RANKER_LEARNING_RATE = 0.04 if V32_SMOKE else 0.025
YETI_ITERATIONS = 16 if V32_SMOKE else 180
YETI_LEARNING_RATE = 0.07 if V32_SMOKE else 0.035

# Exact public-proven V30 residual strength. No search in V32.
ALPHA = 0.16

# Exact frozen V31 Intent-State hierarchy. No smoothing search in V32.
INTENT_PARENT_STRENGTH = 40.0
INTENT_CHILD_STRENGTH = 30.0

# Three entirely fresh development partitions. Historical G/H/I are burned.
def _seed_from_text(s):
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:8], 16) % (2**31-1)

FRESH_SALTS = {
    "K": _seed_from_text("MINETODAY-V32-FRESH-K-20260828-CROSSOBJECTIVE"),
    "L": _seed_from_text("MINETODAY-V32-FRESH-L-20260828-CROSSOBJECTIVE"),
    "M": _seed_from_text("MINETODAY-V32-FRESH-M-20260828-CROSSOBJECTIVE"),
}

# Preserve the exact J partition that V31 intentionally never opened.
SEALED_LOCK_SALT = "MINETODAY-V31-SEALED-J-20260828-INTENTSTATE-REJECTION-ONLY"

BOOTSTRAPS = 20 if V32_SMOKE else 1200
GEOMETRY_ATOL = 2.5e-4
assert np.isfinite(GEOMETRY_ATOL) and GEOMETRY_ATOL > 0

# Broad gate: only V30-class, unusually strong objective-diverse gains are eligible.
BROAD_MIN_POSITIVE_SALTS = 2 if V32_SMOKE else 3
BROAD_MIN_FOLD_WINS = 3 if V32_SMOKE else 12
BROAD_MIN_POOLED_INCREMENT = -1.0 if V32_SMOKE else 0.00065
BROAD_MIN_WORST_SALT = -1.0 if V32_SMOKE else 0.00030
BROAD_CI_FLOOR = -1.0 if V32_SMOKE else 0.00010

# Conservative gate: sized to make a +0.00011 public rank-4 move plausible under V30-like transfer.
CONS_MIN_POSITIVE_SALTS = 2 if V32_SMOKE else 3
CONS_MIN_FOLD_WINS = 3 if V32_SMOKE else 11
CONS_MIN_POOLED_INCREMENT = -1.0 if V32_SMOKE else 0.00020
CONS_MIN_WORST_SALT = -1.0 if V32_SMOKE else 0.0
CONS_CI_FLOOR = -1.0 if V32_SMOKE else -0.00005

BROAD_CANDIDATES = ("yeti_residual", "xe_residual", "objective_median")
CONSERVATIVE_CANDIDATE = "unanimous_top6"
ALL_CANDIDATES = BROAD_CANDIDATES + (CONSERVATIVE_CANDIDATE,)

PROTECTED_TEST_USERS = {
    "U_4047", "U_4953", "U_4074", "U_4176", "U_4491", "U_4567", "U_4836", "U_4929"
}

# Rank-geometry invariant from the executed public V30 artifact. No test labels are encoded here.
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

print("numpy", np.__version__)
print("pandas", pd.__version__)
print("scipy", scipy.__version__)
print("sklearn", sklearn.__version__)
print("lightgbm", lgb.__version__)
print("catboost", catboost.__version__)
print("V32_SMOKE", V32_SMOKE)
print("fixed alpha", ALPHA)
print("frozen Intent-State smoothing", INTENT_PARENT_STRENGTH, INTENT_CHILD_STRENGTH)
print("fresh salts", FRESH_SALTS)
print("public V30", PUBLIC_V30_LB, "rank4 gap", round(PUBLIC_RANK4_LB-PUBLIC_V30_LB, 5), "leader gap", round(PUBLIC_LEADER_LB-PUBLIC_V30_LB, 5))
try:
    print("GPU runtime:\n" + subprocess.check_output(["nvidia-smi", "-L"], text=True).strip())
except Exception:
    print("nvidia-smi unavailable. V32 remains runnable; metric-aligned YetiRank-NDCG is intentionally CPU-first.")

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
# ## Frozen V31 Intent-State representation
#
# The 16 hierarchical Intent-State features are reused exactly as executed in V31. Their hierarchy, smoothing, target definitions, and nested cross-fitting are frozen. V32 does not tune or add representation features.

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
# ## Frozen feature surface + cross-objective residual experts
#
# V30 LambdaRank on the 174-feature base surface remains the exact control. V31 LambdaRank on the 190-feature Intent-State surface is retained only as a diagnostic and as the additional confirmation required for a `rank 5 ↔ 6` UnanimousTop6 intervention. The two new experts use the same 190-feature surface.

# ---- cell 18 (code) ----
def rank_matrix(scores):
    order=np.argsort(-np.asarray(scores),axis=1,kind="mergesort")
    ranks=np.empty_like(order,dtype=np.int16)
    ranks[np.arange(len(order))[:,None],order]=np.arange(1,N_MODULES+1,dtype=np.int16)[None,:]
    return ranks

def top5_order_change_rate(a,b):
    return float(np.mean(np.any(top5_indices(a)!=top5_indices(b),axis=1)))

def build_long_matrix_v31(rows_global,anchor,base,meta,intentstate=None):
    # Frozen V31 feature builder. Do not modify feature definitions in V32.
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
        raise ValueError("non-finite frozen V31 long features")
    return XX,names,bounds

def make_lambdarank(seed):
    # Exact V30 ranker.
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

def make_xendcg(seed):
    # Same capacity as V30; objective is the controlled intervention.
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
    # Historical V6 metric-aligned regime, frozen without search.
    # This NDCG mode is deliberately CPU-first in CatBoost.
    return CatBoostRanker(
        loss_function="YetiRank:mode=NDCG;top=5;dcg_type=Base;dcg_denominator=LogPosition",
        eval_metric="NDCG:top=5;type=Base;denominator=LogPosition",
        iterations=YETI_ITERATIONS, learning_rate=YETI_LEARNING_RATE,
        depth=5, l2_leaf_reg=11.0, random_strength=0.30,
        random_seed=int(seed), task_type="CPU", thread_count=4,
        allow_writing_files=False, verbose=False,
    )

def fit_lgb_ranker_predict(kind,Xtr,Xva,grades_tr,n_users,seed):
    model = make_lambdarank(seed) if kind=="lambda" else make_xendcg(seed)
    group=np.full(int(n_users),N_MODULES,dtype=np.int32)
    model.fit(Xtr,grades_tr.reshape(-1).astype(np.int32),group=group)
    pred=model.predict(Xva).reshape(-1,N_MODULES).astype(np.float32)
    return model,pred

def fit_yeti_predict(Xtr,Xva,grades_tr,n_users,seed):
    # CatBoost Base-DCG receives the same fixed ordinal gain map that LightGBM gets via label_gain.
    gains=np.asarray(LABEL_GAIN,np.float32)[np.asarray(grades_tr,np.int16)]
    train_pool=Pool(
        Xtr,
        label=gains.reshape(-1).astype(np.float32),
        group_id=np.repeat(np.arange(int(n_users),dtype=np.int32),N_MODULES),
    )
    model=make_yetirank(seed)
    model.fit(train_pool,verbose=False)
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

def objective_median(v30,yeti,xe):
    return np.median(np.stack([v30,yeti,xe],axis=0),axis=0).astype(np.float32)

def unanimous_top6(v30,yeti,xe,v31_confirm,return_ledger=False,user_ids=None):
    """At most one adjacent Top-6 correction per user; otherwise exact V30 abstention."""
    v30=np.asarray(v30,np.float32); yeti=np.asarray(yeti,np.float32)
    xe=np.asarray(xe,np.float32); v31_confirm=np.asarray(v31_confirm,np.float32)
    out=v30.copy(); order=np.argsort(-v30,axis=1,kind="mergesort")[:,:6]
    ledger=[]
    for i in range(len(v30)):
        eligible=[]
        for p in range(5):
            high=int(order[i,p]); low=int(order[i,p+1])
            agree=bool(yeti[i,low]>yeti[i,high] and xe[i,low]>xe[i,high])
            if p==4:
                agree=bool(agree and v31_confirm[i,low]>v31_confirm[i,high])
            if agree:
                margin=float(v30[i,high]-v30[i,low])
                eligible.append((margin,p,high,low))
        if not eligible:
            continue
        margin,p,high,low=min(eligible,key=lambda z:(z[0],z[1],z[2],z[3]))
        out[i,high],out[i,low]=out[i,low],out[i,high]
        if return_ledger:
            ledger.append({
                "row":i,
                "user_id":str(user_ids[i]) if user_ids is not None else str(i),
                "position_pair":f"{p+1}<->{p+2}",
                "higher_before":MODULES[high],
                "promoted":MODULES[low],
                "v30_margin":margin,
                "v31_confirmation_required":bool(p==4),
            })
    if return_ledger:
        return out,pd.DataFrame(ledger)
    return out

def build_candidate_surfaces(anchor,raw_v30,raw_v31,raw_xe,raw_yeti):
    v30=blend_residual(anchor,raw_v30,ALPHA)
    v31=blend_residual(anchor,raw_v31,ALPHA)
    xe=blend_residual(anchor,raw_xe,ALPHA)
    yeti=blend_residual(anchor,raw_yeti,ALPHA)
    med=objective_median(v30,yeti,xe)
    uni=unanimous_top6(v30,yeti,xe,v31)
    return {
        "v30_control":v30,
        "v31_intentstate":v31,
        "yeti_residual":yeti,
        "xe_residual":xe,
        "objective_median":med,
        "unanimous_top6":uni,
    }

print("V32 cross-objective helpers ready")

# ---- cell 19 (code) ----
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

# ---- cell 20 (markdown) ----
# ## Fresh K/L/M: cross-objective development
#
# K/L/M are the only development worlds for V32. Every candidate is evaluated incrementally against exact V30. G/H/I are historical and are not read by the selection logic.

# ---- cell 21 (code) ----
assert callable(build_anchor_split_full)
assert abs(ALPHA-0.16)<1e-12
assert INTENT_PARENT_STRENGTH==40.0 and INTENT_CHILD_STRENGTH==30.0
assert set(FRESH_SALTS)=={"K","L","M"}
assert set(ALL_CANDIDATES)=={"yeti_residual","xe_residual","objective_median","unanimous_top6"}

SALT_SCORES={}
SALT_INCREMENT={c:{} for c in ALL_CANDIDATES}
SALT_FOLD_ID={}
salt_rows=[]
fold_rows=[]
diversity_rows=[]
user_delta_rows=[]

for salt_name,salt_seed in FRESH_SALTS.items():
    print(f"\n=== V32 fresh salt {salt_name} seed={salt_seed} ===")
    outer=make_group_folds_global(np.arange(N_TRAIN),N_FOLDS,salt_seed)
    anchor_oof=np.zeros_like(y,np.float32)
    raw_v30=np.zeros_like(y,np.float32)
    raw_v31=np.zeros_like(y,np.float32)
    raw_xe=np.zeros_like(y,np.float32)
    raw_yeti=np.zeros_like(y,np.float32)
    fold_id=np.full(N_TRAIN,-1,np.int16)

    for f,(tr,va) in enumerate(outer):
        print(f"salt {salt_name} fold {f+1}/{len(outer)} train={len(tr)} val={len(va)}")
        anc=build_anchor_split_full(tr,va,salt_seed+1000+f*211)
        anchor_oof[va]=anc["cons_val"]
        fold_id[va]=f

        intent_tr=crossfit_intentstate_features(
            y[tr],ORDINAL_GRADES[tr],TRAIN_INTENT[tr],MODULE_READY_BAND[tr],TRAIN_GROUP[tr],
            salt_seed+2000+f*223
        )
        intent_bundle=fit_intentstate_bundle(
            y[tr],ORDINAL_GRADES[tr],TRAIN_INTENT[tr],MODULE_READY_BAND[tr]
        )
        intent_va=apply_intentstate_bundle(intent_bundle,TRAIN_INTENT[va],MODULE_READY_BAND[va])

        Xtr,names,bounds=build_long_matrix_v31(
            tr,anc["cons_train"],anc["base_train"],anc["meta_train"],intent_tr
        )
        Xva,_,bounds_va=build_long_matrix_v31(
            va,anc["cons_val"],anc["base_val"],anc["meta_val"],intent_va
        )
        assert bounds==bounds_va
        assert bounds["intentstate"]>bounds["base"]

        rank_seed=salt_seed+4000+f*239
        m0,r0=fit_lgb_ranker_predict(
            "lambda",Xtr[:,:bounds["base"]],Xva[:,:bounds["base"]],ORDINAL_GRADES[tr],len(tr),rank_seed
        )
        m1,r1=fit_lgb_ranker_predict(
            "lambda",Xtr[:,:bounds["intentstate"]],Xva[:,:bounds["intentstate"]],ORDINAL_GRADES[tr],len(tr),rank_seed
        )
        mx,rx=fit_lgb_ranker_predict(
            "xe",Xtr[:,:bounds["intentstate"]],Xva[:,:bounds["intentstate"]],ORDINAL_GRADES[tr],len(tr),rank_seed
        )
        my,ry=fit_yeti_predict(
            Xtr[:,:bounds["intentstate"]],Xva[:,:bounds["intentstate"]],ORDINAL_GRADES[tr],len(tr),rank_seed
        )
        raw_v30[va]=r0; raw_v31[va]=r1; raw_xe[va]=rx; raw_yeti[va]=ry

        del Xtr,Xva,intent_tr,intent_va,intent_bundle,m0,m1,mx,my,anc
        gc.collect()

    assert (fold_id>=0).all()
    for arr in (anchor_oof,raw_v30,raw_v31,raw_xe,raw_yeti):
        assert np.isfinite(arr).all()

    scores=build_candidate_surfaces(anchor_oof,raw_v30,raw_v31,raw_xe,raw_yeti)
    SALT_SCORES[salt_name]=scores
    SALT_FOLD_ID[salt_name]=fold_id
    u30=per_user_ndcg(y,scores["v30_control"])

    # Candidate-level metrics, all strictly incremental to V30.
    for cand in ALL_CANDIDATES:
        uc=per_user_ndcg(y,scores[cand]); inc=uc-u30
        SALT_INCREMENT[cand][salt_name]=inc
        fold_means=[]
        for f in range(len(outer)):
            fm=float(inc[fold_id==f].mean()); fold_means.append(fm)
            fold_rows.append({"salt":salt_name,"candidate":cand,"fold":f,"increment_vs_v30":fm})
        ci=cluster_boot_ci(inc,TRAIN_GROUP,salt_seed+7000+101*ALL_CANDIDATES.index(cand))
        salt_rows.append({
            "salt":salt_name,"candidate":cand,"alpha":ALPHA,
            "v30_control_ndcg":float(u30.mean()),
            "candidate_ndcg":float(uc.mean()),
            "increment_vs_v30":float(inc.mean()),
            "positive_rate":float(np.mean(inc>0)),
            "harm_rate":float(np.mean(inc<0)),
            "fold_wins":int(np.sum(np.asarray(fold_means)>0)),
            "min_fold":float(np.min(fold_means)),
            "max_fold":float(np.max(fold_means)),
            "ci_low":ci[0],"ci_high":ci[1],
            "top5_membership_change":top5_set_change_rate(scores["v30_control"],scores[cand]),
            "top5_order_change":top5_order_change_rate(scores["v30_control"],scores[cand]),
            "top1_change":top1_change_rate(scores["v30_control"],scores[cand]),
        })
        for i,d in enumerate(inc):
            user_delta_rows.append({"salt":salt_name,"user_id":train.iloc[i].user_id,"candidate":cand,"increment_vs_v30":float(d)})

    # Diversity is measured on realized per-user NDCG deltas, not raw prediction correlation.
    dv31=per_user_ndcg(y,scores["v31_intentstate"])-u30
    dy=per_user_ndcg(y,scores["yeti_residual"])-u30
    dx=per_user_ndcg(y,scores["xe_residual"])-u30
    def _safe_corr(a,b,method="pearson"):
        a=np.asarray(a,float); b=np.asarray(b,float)
        if np.std(a)<1e-12 or np.std(b)<1e-12:
            return 0.0
        if method=="pearson":
            return float(np.corrcoef(a,b)[0,1])
        return float(spearmanr(a,b).statistic)
    diversity_rows.append({
        "salt":salt_name,
        "yeti_xe_pearson_delta":_safe_corr(dy,dx,"pearson"),
        "yeti_xe_spearman_delta":_safe_corr(dy,dx,"spearman"),
        "yeti_v31_pearson_delta":_safe_corr(dy,dv31,"pearson"),
        "xe_v31_pearson_delta":_safe_corr(dx,dv31,"pearson"),
        "yeti_xe_both_positive":float(np.mean((dy>0)&(dx>0))),
        "yeti_xe_both_harm":float(np.mean((dy<0)&(dx<0))),
        "yeti_xe_opposite_sign":float(np.mean(((dy>0)&(dx<0))|((dy<0)&(dx>0)))),
        "unanimous_top6_intervention_rate":top5_order_change_rate(scores["v30_control"],scores["unanimous_top6"]),
    })

    del raw_v30,raw_v31,raw_xe,raw_yeti,anchor_oof
    gc.collect()

V32_THREE_SALT=pd.DataFrame(salt_rows)
V32_FOLD_DELTAS=pd.DataFrame(fold_rows)
V32_DIVERSITY=pd.DataFrame(diversity_rows)
V32_USER_DELTAS=pd.DataFrame(user_delta_rows)
display(V32_THREE_SALT)
display(V32_DIVERSITY)

# ---- cell 22 (code) ----
# Precommitted candidate gates and max-min selection ladder.
pooled_rows=[]
for cand in ALL_CANDIDATES:
    salt_incs=np.asarray([float(V32_THREE_SALT.query("candidate==@cand and salt==@s")["increment_vs_v30"].iloc[0]) for s in FRESH_SALTS],float)
    fold_wins=int(V32_THREE_SALT.query("candidate==@cand")["fold_wins"].sum())
    avg_user_inc=np.mean(np.stack([SALT_INCREMENT[cand][s] for s in FRESH_SALTS],axis=0),axis=0)
    ci=cluster_boot_ci(avg_user_inc,TRAIN_GROUP,_seed_from_text(f"V32-POOLED-{cand}"))
    pooled_rows.append({
        "candidate":cand,
        "salt_K_increment":float(salt_incs[0]),
        "salt_L_increment":float(salt_incs[1]),
        "salt_M_increment":float(salt_incs[2]),
        "positive_salts":int(np.sum(salt_incs>0)),
        "worst_salt_increment":float(np.min(salt_incs)),
        "mean_salt_increment":float(np.mean(salt_incs)),
        "fold_wins_total":fold_wins,
        "pooled_user_increment":float(avg_user_inc.mean()),
        "pooled_ci_low":ci[0],
        "pooled_ci_high":ci[1],
    })

V32_POOLED=pd.DataFrame(pooled_rows)

def passes_broad(r):
    return bool(
        int(r.positive_salts)>=BROAD_MIN_POSITIVE_SALTS and
        int(r.fold_wins_total)>=BROAD_MIN_FOLD_WINS and
        float(r.pooled_user_increment)>=BROAD_MIN_POOLED_INCREMENT and
        float(r.worst_salt_increment)>=BROAD_MIN_WORST_SALT and
        float(r.pooled_ci_low)>=BROAD_CI_FLOOR
    )

def passes_conservative(r):
    return bool(
        int(r.positive_salts)>=CONS_MIN_POSITIVE_SALTS and
        int(r.fold_wins_total)>=CONS_MIN_FOLD_WINS and
        float(r.pooled_user_increment)>=CONS_MIN_POOLED_INCREMENT and
        float(r.worst_salt_increment)>CONS_MIN_WORST_SALT and
        float(r.pooled_ci_low)>=CONS_CI_FLOOR
    )

V32_POOLED["broad_gate_passed"]=[passes_broad(r) if r.candidate in BROAD_CANDIDATES else False for _,r in V32_POOLED.iterrows()]
V32_POOLED["conservative_gate_passed"]=[passes_conservative(r) if r.candidate==CONSERVATIVE_CANDIDATE else False for _,r in V32_POOLED.iterrows()]

broad_passers=V32_POOLED[(V32_POOLED.candidate.isin(BROAD_CANDIDATES)) & (V32_POOLED.broad_gate_passed)].copy()
if len(broad_passers):
    # V30 lesson: robustness first, then mean. No score mining beyond this predeclared order.
    broad_passers=broad_passers.sort_values(
        ["worst_salt_increment","pooled_user_increment","fold_wins_total","candidate"],
        ascending=[False,False,False,True],kind="mergesort"
    )
    DEV_WINNER=str(broad_passers.iloc[0].candidate)
    DEV_MODE="BROAD"
else:
    cr=V32_POOLED[V32_POOLED.candidate==CONSERVATIVE_CANDIDATE].iloc[0]
    if passes_conservative(cr):
        DEV_WINNER=CONSERVATIVE_CANDIDATE
        DEV_MODE="CONSERVATIVE"
    else:
        DEV_WINNER=None
        DEV_MODE=None

DEV_GATE=bool(DEV_WINNER is not None)
V32_GATE=pd.DataFrame([{
    "development_gate_passed":DEV_GATE,
    "development_winner":DEV_WINNER,
    "development_mode":DEV_MODE,
    "broad_min_positive_salts":BROAD_MIN_POSITIVE_SALTS,
    "broad_min_fold_wins":BROAD_MIN_FOLD_WINS,
    "broad_min_pooled_increment":BROAD_MIN_POOLED_INCREMENT,
    "broad_min_worst_salt":BROAD_MIN_WORST_SALT,
    "broad_ci_floor":BROAD_CI_FLOOR,
    "cons_min_positive_salts":CONS_MIN_POSITIVE_SALTS,
    "cons_min_fold_wins":CONS_MIN_FOLD_WINS,
    "cons_min_pooled_increment":CONS_MIN_POOLED_INCREMENT,
    "cons_min_worst_salt":CONS_MIN_WORST_SALT,
    "cons_ci_floor":CONS_CI_FLOOR,
}])

display(V32_POOLED)
display(V32_GATE)
print("FROZEN DEVELOPMENT DECISION =",DEV_WINNER,"mode=",DEV_MODE)

# ---- cell 23 (markdown) ----
# ## Sealed J + chronology: rejection-only
#
# The J partition is exactly the untouched V31 sealed partition. It is evaluated only after K/L/M have frozen one development winner. Chronology is also rejection-only. Neither surface can trigger retuning or a different candidate choice.

# ---- cell 24 (code) ----
def sealed_lock_bucket(group):
    h=hashlib.sha1(f"{SEALED_LOCK_SALT}::{group}".encode("utf-8")).hexdigest()
    return int(h[:12],16)%5

SEALED_LOCK_MASK=np.asarray([sealed_lock_bucket(g)==0 for g in TRAIN_GROUP],bool)
SEALED_DEV_IDX=np.flatnonzero(~SEALED_LOCK_MASK)
SEALED_LOCK_IDX=np.flatnonzero(SEALED_LOCK_MASK)
assert len(set(TRAIN_GROUP[SEALED_DEV_IDX]).intersection(set(TRAIN_GROUP[SEALED_LOCK_IDX])))==0

CHRONO_TR=np.arange(0,3000,dtype=int)
CHRONO_VA=np.arange(3000,4000,dtype=int)

def evaluate_v32_candidate(train_rows,val_rows,seed,world,candidate_name):
    train_rows=np.asarray(train_rows,int); val_rows=np.asarray(val_rows,int)
    anc=build_anchor_split_full(train_rows,val_rows,seed)

    intent_tr=crossfit_intentstate_features(
        y[train_rows],ORDINAL_GRADES[train_rows],TRAIN_INTENT[train_rows],MODULE_READY_BAND[train_rows],
        TRAIN_GROUP[train_rows],seed+101
    )
    bundle=fit_intentstate_bundle(
        y[train_rows],ORDINAL_GRADES[train_rows],TRAIN_INTENT[train_rows],MODULE_READY_BAND[train_rows]
    )
    intent_va=apply_intentstate_bundle(bundle,TRAIN_INTENT[val_rows],MODULE_READY_BAND[val_rows])

    Xtr,_,bounds=build_long_matrix_v31(
        train_rows,anc["cons_train"],anc["base_train"],anc["meta_train"],intent_tr
    )
    Xva,_,_=build_long_matrix_v31(
        val_rows,anc["cons_val"],anc["base_val"],anc["meta_val"],intent_va
    )
    rank_seed=seed+303
    m0,r0=fit_lgb_ranker_predict("lambda",Xtr[:,:bounds["base"]],Xva[:,:bounds["base"]],ORDINAL_GRADES[train_rows],len(train_rows),rank_seed)
    m1,r1=fit_lgb_ranker_predict("lambda",Xtr[:,:bounds["intentstate"]],Xva[:,:bounds["intentstate"]],ORDINAL_GRADES[train_rows],len(train_rows),rank_seed)
    mx,rx=fit_lgb_ranker_predict("xe",Xtr[:,:bounds["intentstate"]],Xva[:,:bounds["intentstate"]],ORDINAL_GRADES[train_rows],len(train_rows),rank_seed)
    my,ry=fit_yeti_predict(Xtr[:,:bounds["intentstate"]],Xva[:,:bounds["intentstate"]],ORDINAL_GRADES[train_rows],len(train_rows),rank_seed)
    scores=build_candidate_surfaces(anc["cons_val"],r0,r1,rx,ry)
    s0=scores["v30_control"]; sc=scores[candidate_name]
    u0=per_user_ndcg(y[val_rows],s0); uc=per_user_ndcg(y[val_rows],sc)
    inc=uc-u0
    ci=cluster_boot_ci(inc,TRAIN_GROUP[val_rows],seed+404)
    row={
        "world":world,"candidate":candidate_name,"n":len(val_rows),"alpha":ALPHA,
        "v30_control_ndcg":float(u0.mean()),"candidate_ndcg":float(uc.mean()),
        "increment_vs_v30":float(inc.mean()),
        "positive_rate":float(np.mean(inc>0)),"harm_rate":float(np.mean(inc<0)),
        "ci_low":ci[0],"ci_high":ci[1],
        "top5_membership_change":top5_set_change_rate(s0,sc),
        "top5_order_change":top5_order_change_rate(s0,sc),
        "top1_change":top1_change_rate(s0,sc),
    }
    del Xtr,Xva,intent_tr,intent_va,bundle,m0,m1,mx,my,anc,scores
    gc.collect()
    return row

if DEV_GATE:
    sealed_rows=[
        evaluate_v32_candidate(SEALED_DEV_IDX,SEALED_LOCK_IDX,_seed_from_text("V32-SEALED-J-FIT"),"sealed_template_lock_J",DEV_WINNER),
        evaluate_v32_candidate(CHRONO_TR,CHRONO_VA,_seed_from_text("V32-CHRONO-FIT"),"chronological_U3001_U4000",DEV_WINNER),
    ]
    V32_SEALED=pd.DataFrame(sealed_rows)
    j_inc=float(V32_SEALED.loc[V32_SEALED.world=="sealed_template_lock_J","increment_vs_v30"].iloc[0])
    chrono_inc=float(V32_SEALED.loc[V32_SEALED.world=="chronological_U3001_U4000","increment_vs_v30"].iloc[0])
    SEALED_GATE=bool(j_inc>0.0 and chrono_inc>=0.0)
else:
    V32_SEALED=pd.DataFrame([{
        "world":"skipped_development_gate_failed","candidate":None,"n":0,"alpha":ALPHA,
        "v30_control_ndcg":np.nan,"candidate_ndcg":np.nan,"increment_vs_v30":np.nan,
        "positive_rate":np.nan,"harm_rate":np.nan,"ci_low":np.nan,"ci_high":np.nan,
        "top5_membership_change":np.nan,"top5_order_change":np.nan,"top1_change":np.nan,
    }])
    SEALED_GATE=False

display(V32_SEALED)
print("SEALED_GATE =",SEALED_GATE)

# ---- cell 25 (markdown) ----
# ## Full-train deployment and exact V30 fail-close
#
# Full deployment refits the same four frozen expert surfaces on all organizer-provided training users. The exact V30 reconstruction invariant is checked before any V32 submission file can become eligible.

# ---- cell 26 (code) ----
OUT_DIR=Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("/mnt/data")
OUT_DIR.mkdir(parents=True,exist_ok=True)

# Full strict cross-fit for train; test mapping uses train-only fitted statistics.
full_intent_tr=crossfit_intentstate_features(
    y,ORDINAL_GRADES,TRAIN_INTENT,MODULE_READY_BAND[:N_TRAIN],TRAIN_GROUP,
    _seed_from_text("V31-FULL-INTENTSTATE-CROSSFIT")
)
FULL_INTENT_BUNDLE=fit_intentstate_bundle(
    y,ORDINAL_GRADES,TRAIN_INTENT,MODULE_READY_BAND[:N_TRAIN]
)
full_intent_te=apply_intentstate_bundle(
    FULL_INTENT_BUNDLE,TEST_INTENT,MODULE_READY_BAND[N_TRAIN:]
)
V32_SUPPORT_AUDIT=intentstate_support_audit(
    FULL_INTENT_BUNDLE,TEST_INTENT,MODULE_READY_BAND[N_TRAIN:]
)
display(V32_SUPPORT_AUDIT)

Xtr_full,names_full,bounds_full=build_long_matrix_v31(
    np.arange(N_TRAIN),consensus3_oof,base_oof,meta_oof,full_intent_tr
)
Xte_full,_,bounds_test=build_long_matrix_v31(
    np.arange(N_TRAIN,N_TRAIN+N_TEST),consensus3_test,base_test,meta_test,full_intent_te
)
assert bounds_full==bounds_test
assert int(bounds_full["intentstate"]-bounds_full["base"])==len(INTENTSTATE_KEYS)==16

# Exact V30 full-ranker seed preserves the public control; all objectives share it to reduce seed confounding.
full_rank_seed=_seed_from_text("V30-FULL-RANKER")
model_v30,raw_v30=fit_lgb_ranker_predict(
    "lambda",Xtr_full[:,:bounds_full["base"]],Xte_full[:,:bounds_full["base"]],ORDINAL_GRADES,N_TRAIN,full_rank_seed
)
model_v31,raw_v31=fit_lgb_ranker_predict(
    "lambda",Xtr_full[:,:bounds_full["intentstate"]],Xte_full[:,:bounds_full["intentstate"]],ORDINAL_GRADES,N_TRAIN,full_rank_seed
)
model_xe,raw_xe=fit_lgb_ranker_predict(
    "xe",Xtr_full[:,:bounds_full["intentstate"]],Xte_full[:,:bounds_full["intentstate"]],ORDINAL_GRADES,N_TRAIN,full_rank_seed
)
model_yeti,raw_yeti=fit_yeti_predict(
    Xtr_full[:,:bounds_full["intentstate"]],Xte_full[:,:bounds_full["intentstate"]],ORDINAL_GRADES,N_TRAIN,full_rank_seed
)

SURFACES=build_candidate_surfaces(consensus3_test,raw_v30,raw_v31,raw_xe,raw_yeti)

# Preserve all eight previously public-tested V25 rows exactly in every deployment surface.
protected_rows=[TEST_UID_TO_ROW[u] for u in sorted(PROTECTED_TEST_USERS)]
for key in list(SURFACES):
    SURFACES[key][protected_rows]=v25_anchor_test[protected_rows]
    for r in protected_rows:
        assert np.array_equal(SURFACES[key][r],v25_anchor_test[r])

# Recompute composites after expert protection so the invariant is explicit.
SURFACES["objective_median"]=objective_median(SURFACES["v30_control"],SURFACES["yeti_residual"],SURFACES["xe_residual"])
SURFACES["unanimous_top6"],V32_UNANIMOUS_LEDGER=unanimous_top6(
    SURFACES["v30_control"],SURFACES["yeti_residual"],SURFACES["xe_residual"],SURFACES["v31_intentstate"],
    return_ledger=True,user_ids=test["user_id"].astype(str).tolist()
)
for key in ("objective_median","unanimous_top6"):
    SURFACES[key][protected_rows]=v25_anchor_test[protected_rows]

# Exact rank-geometry reconstruction audit for current public V30.
v25_top=top5_indices(v25_anchor_test)
v30_top=top5_indices(SURFACES["v30_control"])
v30_changed_rows=np.flatnonzero(np.any(v25_top!=v30_top,axis=1))
v30_changed_users=set(test.iloc[v30_changed_rows]["user_id"].astype(str))
v30_top1_changes=int(np.sum(v25_top[:,0]!=v30_top[:,0]))
v30_membership_changes=int(np.sum([set(a)!=set(b) for a,b in zip(v25_top,v30_top)]))
V30_REPRO_OK=bool(
    v30_changed_users==EXPECTED_V30_CHANGED_USERS and
    v30_top1_changes==EXPECTED_V30_TOP1_CHANGES and
    v30_membership_changes==EXPECTED_V30_MEMBERSHIP_CHANGES
)
V30_REPRO_AUDIT={
    "expected_changed_users":len(EXPECTED_V30_CHANGED_USERS),
    "actual_changed_users":len(v30_changed_users),
    "expected_top1_changes":EXPECTED_V30_TOP1_CHANGES,
    "actual_top1_changes":v30_top1_changes,
    "expected_membership_changes":EXPECTED_V30_MEMBERSHIP_CHANGES,
    "actual_membership_changes":v30_membership_changes,
    "exact_changed_user_set_match":bool(v30_changed_users==EXPECTED_V30_CHANGED_USERS),
    "reproduction_ok":V30_REPRO_OK,
}
print(json.dumps(V30_REPRO_AUDIT,indent=2))
if not V32_SMOKE:
    assert V30_REPRO_OK, "V30 control rank geometry did not reproduce exactly; do not create/submit V32 from this run."

FINAL_GATE=bool(DEV_GATE and SEALED_GATE and V30_REPRO_OK)
selected_test=SURFACES[DEV_WINNER].copy() if DEV_WINNER is not None else SURFACES["v30_control"].copy()
primary_test=selected_test.copy() if FINAL_GATE else SURFACES["v30_control"].copy()

assert sample["user_id"].astype(str).tolist()==test["user_id"].astype(str).tolist()
def submission_from_scores(scores):
    out=sample.copy()
    out.loc[:,MODULES]=np.asarray(scores,np.float32)
    assert np.isfinite(out[MODULES].to_numpy()).all()
    return out

SUBMISSIONS={
    "anchor_v30":submission_from_scores(SURFACES["v30_control"]),
    "v31_intentstate":submission_from_scores(SURFACES["v31_intentstate"]),
    "yeti_residual":submission_from_scores(SURFACES["yeti_residual"]),
    "xe_residual":submission_from_scores(SURFACES["xe_residual"]),
    "objective_median":submission_from_scores(SURFACES["objective_median"]),
    "unanimous_top6":submission_from_scores(SURFACES["unanimous_top6"]),
    "primary":submission_from_scores(primary_test),
}

SUBMISSIONS["anchor_v30"].to_csv(OUT_DIR/"submission_v32_anchor_v30_reconstructed.csv",index=False)
SUBMISSIONS["v31_intentstate"].to_csv(OUT_DIR/"submission_v32_DIAGNOSTIC_v31_intentstate.csv",index=False)
SUBMISSIONS["yeti_residual"].to_csv(OUT_DIR/"submission_v32_DIAGNOSTIC_yeti_residual.csv",index=False)
SUBMISSIONS["xe_residual"].to_csv(OUT_DIR/"submission_v32_DIAGNOSTIC_xe_residual.csv",index=False)
SUBMISSIONS["objective_median"].to_csv(OUT_DIR/"submission_v32_DIAGNOSTIC_objective_median.csv",index=False)
SUBMISSIONS["unanimous_top6"].to_csv(OUT_DIR/"submission_v32_DIAGNOSTIC_unanimous_top6.csv",index=False)
SUBMISSIONS["primary"].to_csv(OUT_DIR/"submission_v32_primary.csv",index=False)

if not FINAL_GATE:
    assert np.array_equal(SUBMISSIONS["primary"][MODULES].to_numpy(),SUBMISSIONS["anchor_v30"][MODULES].to_numpy())

# Test geometry for all predeclared candidates against V30.
test_geometry=[]
a=SURFACES["v30_control"]
for cand in ("v31_intentstate",)+ALL_CANDIDATES:
    c=SURFACES[cand]
    test_geometry.append({
        "candidate":cand,
        "top5_order_changed_users":int(np.sum(np.any(top5_indices(a)!=top5_indices(c),axis=1))),
        "top5_order_change_rate":top5_order_change_rate(a,c),
        "top5_membership_change_rate":top5_set_change_rate(a,c),
        "top1_change_rate":top1_change_rate(a,c),
    })
V32_TEST_GEOMETRY=pd.DataFrame(test_geometry)
display(V32_TEST_GEOMETRY)

# Selected-candidate change ledger, if development chose one.
if DEV_WINNER is not None:
    a_top=top5_indices(a); c_top=top5_indices(SURFACES[DEV_WINNER])
    changed=np.flatnonzero(np.any(a_top!=c_top,axis=1))
    rows=[]
    for r in changed:
        rows.append({
            "user_id":test.iloc[r].user_id,
            "selected_candidate":DEV_WINNER,
            "v30_top5":" ".join(MODULES[j] for j in a_top[r]),
            "v32_top5":" ".join(MODULES[j] for j in c_top[r]),
            "top1_changed":bool(a_top[r,0]!=c_top[r,0]),
            "membership_changed":bool(set(a_top[r])!=set(c_top[r])),
            "protected_v25_user":bool(test.iloc[r].user_id in PROTECTED_TEST_USERS),
            "latest_intent":str(TEST_INTENT[r]),
        })
    V32_TEST_LEDGER=pd.DataFrame(rows)
else:
    V32_TEST_LEDGER=pd.DataFrame(columns=["user_id","selected_candidate","v30_top5","v32_top5","top1_changed","membership_changed","protected_v25_user","latest_intent"])

print("DEV_WINNER =",DEV_WINNER,"FINAL_GATE =",FINAL_GATE,"selected test changes =",len(V32_TEST_LEDGER))

# Release model matrices before artifact serialization.
del Xtr_full,Xte_full,full_intent_tr,full_intent_te,model_v30,model_v31,model_xe,model_yeti

gc.collect()

# ---- cell 27 (code) ----
# Research + submission artifacts.
V32_THREE_SALT.to_csv(OUT_DIR/"v32_three_salt_summary.csv",index=False)
V32_FOLD_DELTAS.to_csv(OUT_DIR/"v32_fold_deltas.csv",index=False)
V32_DIVERSITY.to_csv(OUT_DIR/"v32_objective_diversity.csv",index=False)
V32_USER_DELTAS.to_csv(OUT_DIR/"v32_validation_user_deltas.csv",index=False)
V32_POOLED.to_csv(OUT_DIR/"v32_pooled_incremental.csv",index=False)
V32_GATE.to_csv(OUT_DIR/"v32_development_gate.csv",index=False)
V32_SEALED.to_csv(OUT_DIR/"v32_sealed_validation.csv",index=False)
V32_SUPPORT_AUDIT.to_csv(OUT_DIR/"v32_intentstate_support_audit.csv",index=False)
V32_TEST_GEOMETRY.to_csv(OUT_DIR/"v32_test_geometry.csv",index=False)
V32_TEST_LEDGER.to_csv(OUT_DIR/"v32_test_change_ledger.csv",index=False)
V32_UNANIMOUS_LEDGER.to_csv(OUT_DIR/"v32_unanimous_top6_test_interventions.csv",index=False)

with open(OUT_DIR/"v32_v30_reconstruction_audit.json","w") as f:
    json.dump(V30_REPRO_AUDIT,f,indent=2)
with open(OUT_DIR/"v32_feature_names.json","w") as f:
    json.dump({
        "base_end":int(bounds_full["base"]),
        "intentstate_end":int(bounds_full["intentstate"]),
        "intentstate_feature_count":len(INTENTSTATE_KEYS),
        "feature_names":names_full,
    },f,indent=2)

# Compact score artifact makes post-run auditing possible without re-running the notebook.
np.savez_compressed(
    OUT_DIR/"v32_test_score_artifacts.npz",
    v30=SURFACES["v30_control"],
    v31_intentstate=SURFACES["v31_intentstate"],
    yeti=SURFACES["yeti_residual"],
    xe=SURFACES["xe_residual"],
    objective_median=SURFACES["objective_median"],
    unanimous_top6=SURFACES["unanimous_top6"],
    primary=primary_test,
)

slot_plan=pd.DataFrame([
    {
        "file":"submission_v32_primary.csv",
        "role":"PRIMARY_ONLY_IF_ALL_GATES_PASS",
        "eligible_to_submit":bool(FINAL_GATE),
        "note":f"Selected {DEV_WINNER} only if K/L/M + sealed J + chronology + V30 reconstruction all pass; otherwise exact V30 fail-close.",
    },
    {
        "file":"submission_v32_DIAGNOSTIC_yeti_residual.csv",
        "role":"DIAGNOSTIC_ONLY",
        "eligible_to_submit":False,
        "note":"Frozen CatBoost YetiRank-NDCG residual expert. Do not bypass the gate.",
    },
    {
        "file":"submission_v32_DIAGNOSTIC_xe_residual.csv",
        "role":"DIAGNOSTIC_ONLY",
        "eligible_to_submit":False,
        "note":"Frozen LightGBM rank_xendcg residual expert. Do not bypass the gate.",
    },
    {
        "file":"submission_v32_DIAGNOSTIC_objective_median.csv",
        "role":"DIAGNOSTIC_ONLY",
        "eligible_to_submit":False,
        "note":"Broad cross-objective median candidate. Submission eligibility is represented only by primary.",
    },
    {
        "file":"submission_v32_DIAGNOSTIC_unanimous_top6.csv",
        "role":"DIAGNOSTIC_ONLY",
        "eligible_to_submit":False,
        "note":"Deterministic abstaining consensus candidate. Submission eligibility is represented only by primary.",
    },
    {
        "file":"submission_v32_anchor_v30_reconstructed.csv",
        "role":"CURRENT_PUBLIC_ANCHOR_REFERENCE",
        "eligible_to_submit":False,
        "note":"Reconstructed V30 public anchor; current public reference 0.65963.",
    },
])
slot_plan.to_csv(OUT_DIR/"v32_slot_plan.csv",index=False)

summary={
    "version":"V32 Cross-Objective Residual Consensus LTR",
    "public_anchor_v30":PUBLIC_V30_LB,
    "rank4_reference":PUBLIC_RANK4_LB,
    "rank4_gap":PUBLIC_RANK4_LB-PUBLIC_V30_LB,
    "leader_reference":PUBLIC_LEADER_LB,
    "leader_gap":PUBLIC_LEADER_LB-PUBLIC_V30_LB,
    "fixed_alpha":ALPHA,
    "frozen_intent_parent_strength":INTENT_PARENT_STRENGTH,
    "frozen_intent_child_strength":INTENT_CHILD_STRENGTH,
    "fresh_salts":FRESH_SALTS,
    "sealed_j_salt":SEALED_LOCK_SALT,
    "development_gate_passed":bool(DEV_GATE),
    "development_winner":DEV_WINNER,
    "development_mode":DEV_MODE,
    "sealed_gate_passed":bool(SEALED_GATE),
    "v30_reproduction_ok":bool(V30_REPRO_OK),
    "final_gate_passed":bool(FINAL_GATE),
    "pooled_incremental":V32_POOLED.to_dict("records"),
    "sealed_validation":V32_SEALED.to_dict("records"),
    "objective_diversity":V32_DIVERSITY.to_dict("records"),
    "test_geometry":V32_TEST_GEOMETRY.to_dict("records"),
    "selected_test_changed_users_vs_v30":int(len(V32_TEST_LEDGER)),
    "unanimous_test_interventions":int(len(V32_UNANIMOUS_LEDGER)),
    "submission_instruction":(
        f"PRIMARY IS SUBMISSION-ELIGIBLE: selected candidate {DEV_WINNER} passed the full precommitted V32 protocol. Audit artifacts before spending the final Kaggle slot."
        if FINAL_GATE else
        "NO NEW V32 SUBMISSION: primary is the reconstructed V30 fail-close anchor."
    ),
}
with open(OUT_DIR/"v32_summary.json","w") as f:
    json.dump(summary,f,indent=2)

assert set(V32_THREE_SALT.salt)==set(FRESH_SALTS)
assert set(V32_THREE_SALT.candidate)==set(ALL_CANDIDATES)
assert len(V32_THREE_SALT)==3*len(ALL_CANDIDATES)
assert np.isfinite(V32_THREE_SALT["increment_vs_v30"].to_numpy()).all()
assert SUBMISSIONS["primary"].shape==sample.shape==SUBMISSIONS["anchor_v30"].shape
if len(V32_TEST_LEDGER):
    assert not V32_TEST_LEDGER["protected_v25_user"].any()

print(json.dumps(summary,indent=2))
print("\nV32 outputs written to",OUT_DIR)
print(slot_plan.to_string(index=False))

# ---- cell 28 (markdown) ----
# ### Interpretation contract
#
# - V32 tests **objective/model-family diversity on a frozen representation surface**. It does not claim that all future feature engineering is impossible.
# - K/L/M are the only V32 development worlds. Do not create additional salts because the result is inconvenient.
# - Every promotion statistic is strictly `candidate − exact V30`.
# - The frozen V31 Intent-State block is not retuned. Parent strength `40`, child strength `30`, alpha `0.16`, feature definitions, and hierarchy are immutable.
# - YetiRank and XENDCG are trained on the same 190-feature surface. V30 remains the exact 174-feature public-proven control.
# - `ObjectiveMedian` is a broad candidate and requires the stronger V30-class gate. `UnanimousTop6` is the conservative abstaining candidate and is considered only if no broad candidate passes.
# - Broad winner selection is max-min robust: among broad passers, maximize worst-salt increment before mean increment.
# - `UnanimousTop6` never makes more than one adjacent Top-6 intervention per user. The `5↔6` boundary additionally requires agreement from frozen V31 Intent-State LambdaRank.
# - Sealed J is the exact V31 partition that was never opened because V31 failed development. J and chronology are rejection-only and cannot trigger candidate switching or retuning.
# - The CatBoost objective uses `YetiRank:mode=NDCG;top=5;dcg_type=Base;dcg_denominator=LogPosition`; the label values use the same fixed ordinal gain map used by V30 LightGBM through `label_gain`.
# - If `final_gate_passed == false`, do not spend the final Kaggle slot on a diagnostic candidate. `submission_v32_primary.csv` is then exact V30 fail-close.
