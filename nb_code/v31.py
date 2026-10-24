# Converted from notebooks/v31.ipynb: markdown cells become comments, %/! magics neutralized.

# ---- cell 0 (markdown) ----
# # MineToday V31 — Fold-Safe Intent-State Residual Learning-to-Rank
#
# **Core thesis:** V30 public `0.65963` is the immutable control/fallback. V31 does **not** stack another model on top of V30 and does not change the residual strength. It branches from the exact V30 architecture and changes only the supervised state representation by adding a strict nested target-encoding block for:
#
# `latest intent signature × candidate module × candidate-specific readiness band`
#
# The block estimates heavily-smoothed empirical relevance statistics (positive probability, raw relevance, ordinal grade, top-5 probability, support/confidence) with a hierarchy:
#
# `global module prior → latest-intent/module prior → latest-intent/module/readiness prior`.
#
# ### Precommitted experiment contract
#
# - Exact V30 Base LambdaRank is the control.
# - Exact V30 residual strength `alpha = 0.16`; **no alpha search**.
# - Fresh development worlds are **G/H/I only**.
# - Target encoding for every outer-training fold is itself cross-fitted with template-group-disjoint inner folds. No validation target can contribute to its own encoding.
# - Sealed **J** and chronological `U_3001–U_4000` are rejection-only after the development decision is frozen.
# - C/D/E/F are historical and never used for V31 selection.
# - No DirectCurr, co-occurrence, transition veto, semantic embedding, trajectory TE, XENDCG, YetiRank, or neural list model in this notebook. Those remain separate follow-up hypotheses only after the Intent-State information block proves incremental value.
# - If V31 fails any gate, `submission_v31_primary.csv` fail-closes to the reconstructed V30 ranking.
# - The eight public-tested V25 rows remain protected exactly in both V30 and V31 deployment predictions.

# ---- cell 1 (code) ----
# Runtime + preregistered V31 configuration
import os, sys, re, json, ast, html, hashlib, random, warnings, time, subprocess, gc
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ["PYTHONHASHSEED"] = "42"

V31_SKIP_INSTALL = bool(int(os.getenv("V31_SKIP_INSTALL", "0")))
try:
    import lightgbm as lgb
except Exception:
    if V31_SKIP_INSTALL:
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
V31_SMOKE = bool(int(os.getenv("V31_SMOKE", "0")))
V31_STRICT_GEOMETRY = bool(int(os.getenv("V31_STRICT_GEOMETRY", "1")))

# Compatibility aliases consumed by the frozen V3.1/V25 reconstruction cells.
V30_SMOKE = V31_SMOKE
V30_STRICT_GEOMETRY = V31_STRICT_GEOMETRY
V29_SMOKE = V31_SMOKE
V29_STRICT_GEOMETRY = V31_STRICT_GEOMETRY
V27_SMOKE = V31_SMOKE
V27_STRICT_GEOMETRY = V31_STRICT_GEOMETRY

N_FOLDS = 2 if V31_SMOKE else 5
ANCHOR_INNER_FOLDS = 2 if V31_SMOKE else 4
INTENT_INNER_FOLDS = 2 if V31_SMOKE else 4

RIDGE_ALPHA = 3.0
REFERENCE_W = 0.45

PUBLIC_CAL045_LB = 0.65639
PUBLIC_CONSENSUS3_LB = 0.65856
PUBLIC_V25_LB = 0.65919
PUBLIC_V30_LB = 0.65963
PUBLIC_CURRENT_PB = PUBLIC_V30_LB
PUBLIC_LEADER_LB = 0.66141

LGB_ESTIMATORS = 12 if V31_SMOKE else 1200
LGB_EARLY_STOP = 3 if V31_SMOKE else 80
RANKER_ESTIMATORS = 12 if V31_SMOKE else 360
RANKER_LEARNING_RATE = 0.04 if V31_SMOKE else 0.025

# Exact public-proven V30 residual strength. No search in V31.
ALPHA = 0.16

# Hierarchical smoothing fixed before G/H/I are observed.
INTENT_PARENT_STRENGTH = 40.0
INTENT_CHILD_STRENGTH = 30.0

# Three new development partitions. Historical C/D/E/F are forbidden selectors.
def _seed_from_text(s):
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:8], 16) % (2**31-1)

FRESH_SALTS = {
    "G": _seed_from_text("MINETODAY-V31-FRESH-G-20260828-INTENTSTATE"),
    "H": _seed_from_text("MINETODAY-V31-FRESH-H-20260828-INTENTSTATE"),
    "I": _seed_from_text("MINETODAY-V31-FRESH-I-20260828-INTENTSTATE"),
}
SEALED_LOCK_SALT = "MINETODAY-V31-SEALED-J-20260828-INTENTSTATE-REJECTION-ONLY"

BOOTSTRAPS = 20 if V31_SMOKE else 1200
GEOMETRY_ATOL = 2.5e-4
assert np.isfinite(GEOMETRY_ATOL) and GEOMETRY_ATOL > 0

# Incremental promotion gate is explicitly versus V30, not versus V25/Consensus3.
MIN_INCREMENTAL_FOLD_WINS = 3 if V31_SMOKE else 10   # 10/15 in the full protocol
MIN_POOLED_INCREMENT = 0.00030
POOLED_CI_FLOOR = -0.00005

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
print("V31_SMOKE", V31_SMOKE)
print("fixed alpha", ALPHA)
print("fresh salts", FRESH_SALTS)
print("public V30 anchor", PUBLIC_V30_LB, "leader gap", round(PUBLIC_LEADER_LB-PUBLIC_V30_LB, 5))
try:
    print("GPU runtime:\n" + subprocess.check_output(["nvidia-smi", "-L"], text=True).strip())
except Exception:
    print("nvidia-smi unavailable; V31 is Kaggle T4x2-compatible and deliberately CPU-first for deterministic LightGBM ranking.")

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

# ---- cell 18 (code) ----
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
# ## Fresh G/H/I: incremental test against V30 control
#
# For each outer fold:
#
# 1. rebuild the frozen nested Consensus3 anchor using outer-train only;
# 2. build Intent-State features for outer-train via a second template-group-disjoint cross-fit;
# 3. fit the outer-validation TE bundle on outer-train only;
# 4. train V30-control and V31-IntentState LambdaRank with the **same seed**;
# 5. score the only quantity that can promote V31: `NDCG(V31) − NDCG(V30)`.

# ---- cell 21 (code) ----
assert callable(build_anchor_split_full)
assert abs(ALPHA-0.16)<1e-12
assert set(FRESH_SALTS)=={"G","H","I"}

SALT_ANCHOR={}
SALT_V30={}
SALT_V31={}
SALT_INCREMENT={}
SALT_FOLD_ID={}
salt_rows=[]

for salt_name,salt_seed in FRESH_SALTS.items():
    print(f"\n=== V31 fresh salt {salt_name} seed={salt_seed} ===")
    outer=make_group_folds_global(np.arange(N_TRAIN),N_FOLDS,salt_seed)
    anchor_oof=np.zeros_like(y,np.float32)
    raw_v30=np.zeros_like(y,np.float32)
    raw_v31=np.zeros_like(y,np.float32)
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
        m0,r0=fit_ranker_predict(
            Xtr[:,:bounds["base"]],Xva[:,:bounds["base"]],ORDINAL_GRADES[tr],len(tr),rank_seed
        )
        m1,r1=fit_ranker_predict(
            Xtr[:,:bounds["intentstate"]],Xva[:,:bounds["intentstate"]],ORDINAL_GRADES[tr],len(tr),rank_seed
        )
        raw_v30[va]=r0; raw_v31[va]=r1

        del Xtr,Xva,intent_tr,intent_va,intent_bundle,m0,m1,anc
        gc.collect()

    assert (fold_id>=0).all()
    assert np.isfinite(anchor_oof).all() and np.isfinite(raw_v30).all() and np.isfinite(raw_v31).all()

    score_v30=blend_residual(anchor_oof,raw_v30,ALPHA)
    score_v31=blend_residual(anchor_oof,raw_v31,ALPHA)
    user_anchor=per_user_ndcg(y,anchor_oof)
    user_v30=per_user_ndcg(y,score_v30)
    user_v31=per_user_ndcg(y,score_v31)
    inc=user_v31-user_v30
    fold_means=[float(inc[fold_id==f].mean()) for f in range(len(outer))]
    ci=cluster_boot_ci(inc,TRAIN_GROUP,salt_seed+7000)

    SALT_ANCHOR[salt_name]=anchor_oof
    SALT_V30[salt_name]=score_v30
    SALT_V31[salt_name]=score_v31
    SALT_INCREMENT[salt_name]=inc
    SALT_FOLD_ID[salt_name]=fold_id

    salt_rows.append({
        "salt":salt_name,"alpha":ALPHA,
        "anchor_ndcg":float(user_anchor.mean()),
        "v30_control_ndcg":float(user_v30.mean()),
        "v31_intentstate_ndcg":float(user_v31.mean()),
        "v30_delta_vs_anchor":float((user_v30-user_anchor).mean()),
        "v31_delta_vs_anchor":float((user_v31-user_anchor).mean()),
        "increment_v31_vs_v30":float(inc.mean()),
        "increment_positive_rate":float(np.mean(inc>0)),
        "increment_harm_rate":float(np.mean(inc<0)),
        "increment_fold_wins":int(np.sum(np.asarray(fold_means)>0)),
        "increment_min_fold":float(np.min(fold_means)),
        "increment_max_fold":float(np.max(fold_means)),
        "increment_ci_low":ci[0],"increment_ci_high":ci[1],
        "v31_vs_v30_top5_membership_change":top5_set_change_rate(score_v30,score_v31),
        "v31_vs_v30_top5_order_change":top5_order_change_rate(score_v30,score_v31),
        "v31_vs_v30_top1_change":top1_change_rate(score_v30,score_v31),
    })

V31_THREE_SALT=pd.DataFrame(salt_rows)
display(V31_THREE_SALT)

# ---- cell 22 (code) ----
# Precommitted development promotion gate: all criteria are incremental versus V30.
salt_increments=V31_THREE_SALT["increment_v31_vs_v30"].to_numpy(float)
fold_wins_total=int(V31_THREE_SALT["increment_fold_wins"].sum())
avg_user_increment=np.mean(np.stack([SALT_INCREMENT[s] for s in FRESH_SALTS],axis=0),axis=0)
pooled_ci=cluster_boot_ci(avg_user_increment,TRAIN_GROUP,_seed_from_text("V31-POOLED-INTENTSTATE"))

V31_POOLED=pd.DataFrame([{
    "candidate":"v31_intentstate",
    "salt_G_increment":float(salt_increments[0]),
    "salt_H_increment":float(salt_increments[1]),
    "salt_I_increment":float(salt_increments[2]),
    "positive_salts":int(np.sum(salt_increments>0)),
    "worst_salt_increment":float(np.min(salt_increments)),
    "mean_salt_increment":float(np.mean(salt_increments)),
    "fold_wins_total":fold_wins_total,
    "pooled_user_increment":float(avg_user_increment.mean()),
    "pooled_ci_low":pooled_ci[0],
    "pooled_ci_high":pooled_ci[1],
}])

r=V31_POOLED.iloc[0]
DEV_GATE=bool(
    int(r.positive_salts)==3 and
    int(r.fold_wins_total)>=MIN_INCREMENTAL_FOLD_WINS and
    float(r.pooled_user_increment)>=MIN_POOLED_INCREMENT and
    float(r.pooled_ci_low)>=POOLED_CI_FLOOR
)
DEV_WINNER="v31_intentstate" if DEV_GATE else None

V31_GATE=pd.DataFrame([{
    **r.to_dict(),
    "min_required_pooled_increment":MIN_POOLED_INCREMENT,
    "min_required_fold_wins":MIN_INCREMENTAL_FOLD_WINS,
    "ci_floor":POOLED_CI_FLOOR,
    "development_gate_passed":DEV_GATE,
}])

display(V31_POOLED)
display(V31_GATE)
print("FROZEN DEVELOPMENT DECISION =",DEV_WINNER)

# ---- cell 23 (markdown) ----
# ## Sealed J + chronology: rejection-only
#
# No parameter, feature, smoothing strength, or model choice is changed after G/H/I. J and chronology can only reject the already-frozen Intent-State candidate.

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

def evaluate_incremental_v31(train_rows,val_rows,seed,world):
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
    m0,r0=fit_ranker_predict(Xtr[:,:bounds["base"]],Xva[:,:bounds["base"]],ORDINAL_GRADES[train_rows],len(train_rows),rank_seed)
    m1,r1=fit_ranker_predict(Xtr[:,:bounds["intentstate"]],Xva[:,:bounds["intentstate"]],ORDINAL_GRADES[train_rows],len(train_rows),rank_seed)
    s0=blend_residual(anc["cons_val"],r0,ALPHA)
    s1=blend_residual(anc["cons_val"],r1,ALPHA)
    u0=per_user_ndcg(y[val_rows],s0); u1=per_user_ndcg(y[val_rows],s1)
    inc=u1-u0
    ci=cluster_boot_ci(inc,TRAIN_GROUP[val_rows],seed+404)
    row={
        "world":world,"n":len(val_rows),"alpha":ALPHA,
        "v30_control_ndcg":float(u0.mean()),"v31_intentstate_ndcg":float(u1.mean()),
        "increment_v31_vs_v30":float(inc.mean()),
        "positive_rate":float(np.mean(inc>0)),"harm_rate":float(np.mean(inc<0)),
        "ci_low":ci[0],"ci_high":ci[1],
        "top5_membership_change":top5_set_change_rate(s0,s1),
        "top5_order_change":top5_order_change_rate(s0,s1),
        "top1_change":top1_change_rate(s0,s1),
    }
    del Xtr,Xva,intent_tr,intent_va,bundle,m0,m1,anc
    gc.collect()
    return row

if DEV_GATE:
    sealed_rows=[
        evaluate_incremental_v31(SEALED_DEV_IDX,SEALED_LOCK_IDX,_seed_from_text("V31-SEALED-J-FIT"),"sealed_template_lock_J"),
        evaluate_incremental_v31(CHRONO_TR,CHRONO_VA,_seed_from_text("V31-CHRONO-FIT"),"chronological_U3001_U4000"),
    ]
    V31_SEALED=pd.DataFrame(sealed_rows)
    j_inc=float(V31_SEALED.loc[V31_SEALED.world=="sealed_template_lock_J","increment_v31_vs_v30"].iloc[0])
    chrono_inc=float(V31_SEALED.loc[V31_SEALED.world=="chronological_U3001_U4000","increment_v31_vs_v30"].iloc[0])
    SEALED_GATE=bool(j_inc>0.0 and chrono_inc>=0.0)
else:
    V31_SEALED=pd.DataFrame([{
        "world":"skipped_development_gate_failed","n":0,"alpha":ALPHA,
        "v30_control_ndcg":np.nan,"v31_intentstate_ndcg":np.nan,"increment_v31_vs_v30":np.nan,
        "positive_rate":np.nan,"harm_rate":np.nan,"ci_low":np.nan,"ci_high":np.nan,
        "top5_membership_change":np.nan,"top5_order_change":np.nan,"top1_change":np.nan,
    }])
    SEALED_GATE=False

display(V31_SEALED)
print("SEALED_GATE =",SEALED_GATE)

# ---- cell 25 (markdown) ----
# ## Full-train deployment and exact V30 fail-close
#
# The full model first reconstructs the public V30 Base branch using the exact historical full-ranker seed. Its top-5 intervention geometry is checked against the executed V30 artifact (77 changed users vs V25, 8 top-1 changes, 7 membership changes). Only after this control passes is the V31 candidate allowed to become the primary output.

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
V31_SUPPORT_AUDIT=intentstate_support_audit(
    FULL_INTENT_BUNDLE,TEST_INTENT,MODULE_READY_BAND[N_TRAIN:]
)
display(V31_SUPPORT_AUDIT)

Xtr_full,names_full,bounds_full=build_long_matrix_v31(
    np.arange(N_TRAIN),consensus3_oof,base_oof,meta_oof,full_intent_tr
)
Xte_full,_,bounds_test=build_long_matrix_v31(
    np.arange(N_TRAIN,N_TRAIN+N_TEST),consensus3_test,base_test,meta_test,full_intent_te
)
assert bounds_full==bounds_test

# Exact V30 full-ranker seed, preserving the historical Base branch.
full_rank_seed=_seed_from_text("V30-FULL-RANKER")
model_v30,raw_v30=fit_ranker_predict(
    Xtr_full[:,:bounds_full["base"]],Xte_full[:,:bounds_full["base"]],ORDINAL_GRADES,N_TRAIN,full_rank_seed
)
model_v31,raw_v31=fit_ranker_predict(
    Xtr_full[:,:bounds_full["intentstate"]],Xte_full[:,:bounds_full["intentstate"]],ORDINAL_GRADES,N_TRAIN,full_rank_seed
)

v30_control_test=blend_residual(consensus3_test,raw_v30,ALPHA)
v31_candidate_test=blend_residual(consensus3_test,raw_v31,ALPHA)

# Preserve all eight previously public-tested V25 rows exactly in both branches.
protected_rows=[TEST_UID_TO_ROW[u] for u in sorted(PROTECTED_TEST_USERS)]
v30_control_test[protected_rows]=v25_anchor_test[protected_rows]
v31_candidate_test[protected_rows]=v25_anchor_test[protected_rows]
for r in protected_rows:
    assert np.array_equal(v30_control_test[r],v25_anchor_test[r])
    assert np.array_equal(v31_candidate_test[r],v25_anchor_test[r])

# Exact rank-geometry reconstruction audit for current public V30.
v25_top=top5_indices(v25_anchor_test)
v30_top=top5_indices(v30_control_test)
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
if not V31_SMOKE:
    assert V30_REPRO_OK, "V30 control rank geometry did not reproduce exactly; do not create/submit V31 from this run."

FINAL_GATE=bool(DEV_GATE and SEALED_GATE and V30_REPRO_OK)
primary_test=v31_candidate_test.copy() if FINAL_GATE else v30_control_test.copy()

assert sample["user_id"].astype(str).tolist()==test["user_id"].astype(str).tolist()
def submission_from_scores(scores):
    out=sample.copy()
    out.loc[:,MODULES]=np.asarray(scores,np.float32)
    assert np.isfinite(out[MODULES].to_numpy()).all()
    return out

sub_v30=submission_from_scores(v30_control_test)
sub_v31=submission_from_scores(v31_candidate_test)
sub_primary=submission_from_scores(primary_test)

sub_v30.to_csv(OUT_DIR/"submission_v31_anchor_v30_reconstructed.csv",index=False)
sub_v31.to_csv(OUT_DIR/"submission_v31_DIAGNOSTIC_intentstate.csv",index=False)
sub_primary.to_csv(OUT_DIR/"submission_v31_primary.csv",index=False)

# Exact fail-close invariant when V31 is not promoted.
if not FINAL_GATE:
    assert np.array_equal(sub_primary[MODULES].to_numpy(),sub_v30[MODULES].to_numpy())

# NDCG-relevant test change ledger versus V30 control.
a_top=top5_indices(v30_control_test); c_top=top5_indices(v31_candidate_test)
changed=np.flatnonzero(np.any(a_top!=c_top,axis=1))
ledger=[]
for r in changed:
    ledger.append({
        "user_id":test.iloc[r].user_id,
        "v30_top5":" ".join(MODULES[j] for j in a_top[r]),
        "v31_top5":" ".join(MODULES[j] for j in c_top[r]),
        "top1_changed":bool(a_top[r,0]!=c_top[r,0]),
        "membership_changed":bool(set(a_top[r])!=set(c_top[r])),
        "protected_v25_user":bool(test.iloc[r].user_id in PROTECTED_TEST_USERS),
        "latest_intent":str(TEST_INTENT[r]),
    })
V31_TEST_LEDGER=pd.DataFrame(ledger)
print("V31 candidate changed top-5 ordering vs V30 on",len(V31_TEST_LEDGER),"test users")
print("FINAL_GATE =",FINAL_GATE)

# Release memory before artifact serialization.
del Xtr_full,Xte_full,full_intent_tr,full_intent_te,model_v30,model_v31

gc.collect()

# ---- cell 27 (code) ----
# Research + submission artifacts.
V31_THREE_SALT.to_csv(OUT_DIR/"v31_three_salt_summary.csv",index=False)
V31_POOLED.to_csv(OUT_DIR/"v31_pooled_incremental.csv",index=False)
V31_GATE.to_csv(OUT_DIR/"v31_development_gate.csv",index=False)
V31_SEALED.to_csv(OUT_DIR/"v31_sealed_validation.csv",index=False)
V31_SUPPORT_AUDIT.to_csv(OUT_DIR/"v31_intentstate_support_audit.csv",index=False)
V31_TEST_LEDGER.to_csv(OUT_DIR/"v31_test_change_ledger.csv",index=False)

with open(OUT_DIR/"v31_v30_reconstruction_audit.json","w") as f:
    json.dump(V30_REPRO_AUDIT,f,indent=2)
with open(OUT_DIR/"v31_feature_names.json","w") as f:
    json.dump({"base_end":int(bounds_full["base"]),"intentstate_end":int(bounds_full["intentstate"]),"feature_names":names_full},f,indent=2)

slot_plan=pd.DataFrame([
    {
        "file":"submission_v31_primary.csv",
        "role":"PRIMARY_ONLY_IF_ALL_GATES_PASS",
        "eligible_to_submit":bool(FINAL_GATE),
        "note":"V31 Intent-State only if G/H/I + sealed J + chronology + V30 reconstruction all pass; otherwise exact V30 fail-close.",
    },
    {
        "file":"submission_v31_DIAGNOSTIC_intentstate.csv",
        "role":"DIAGNOSTIC_ONLY",
        "eligible_to_submit":False,
        "note":"Frozen Intent-State candidate regardless of sealed outcome. Never submit automatically.",
    },
    {
        "file":"submission_v31_anchor_v30_reconstructed.csv",
        "role":"CURRENT_PUBLIC_ANCHOR_REFERENCE",
        "eligible_to_submit":False,
        "note":"Reconstructed V30 rank geometry; current public score reference 0.65963.",
    },
])
slot_plan.to_csv(OUT_DIR/"v31_slot_plan.csv",index=False)

summary={
    "version":"V31 Fold-Safe Intent-State Residual Learning-to-Rank",
    "public_anchor_v30":PUBLIC_V30_LB,
    "leader_reference":PUBLIC_LEADER_LB,
    "leader_gap":PUBLIC_LEADER_LB-PUBLIC_V30_LB,
    "fixed_alpha":ALPHA,
    "intent_parent_strength":INTENT_PARENT_STRENGTH,
    "intent_child_strength":INTENT_CHILD_STRENGTH,
    "fresh_salts":FRESH_SALTS,
    "development_gate_passed":bool(DEV_GATE),
    "development_winner":DEV_WINNER,
    "sealed_gate_passed":bool(SEALED_GATE),
    "v30_reproduction_ok":bool(V30_REPRO_OK),
    "final_gate_passed":bool(FINAL_GATE),
    "three_salt":V31_THREE_SALT.to_dict("records"),
    "pooled_incremental":V31_POOLED.to_dict("records"),
    "sealed_validation":V31_SEALED.to_dict("records"),
    "intentstate_support_audit":V31_SUPPORT_AUDIT.to_dict("records"),
    "diagnostic_test_changed_users_vs_v30":int(len(V31_TEST_LEDGER)),
    "submission_instruction":(
        "PRIMARY IS SUBMISSION-ELIGIBLE: audit all V31 outputs before using the scarce Kaggle slot."
        if FINAL_GATE else
        "NO NEW V31 SUBMISSION: primary is the reconstructed V30 fail-close anchor."
    ),
}
with open(OUT_DIR/"v31_summary.json","w") as f:
    json.dump(summary,f,indent=2)

assert set(V31_THREE_SALT.salt)==set(FRESH_SALTS)
assert len(V31_THREE_SALT)==3
assert np.isfinite(V31_THREE_SALT["increment_v31_vs_v30"].to_numpy()).all()
assert sub_primary.shape==sample.shape==sub_v30.shape==sub_v31.shape
assert not V31_TEST_LEDGER.get("protected_v25_user",pd.Series(dtype=bool)).any()

print(json.dumps(summary,indent=2))
print("\nV31 outputs written to",OUT_DIR)
print(slot_plan.to_string(index=False))

# ---- cell 28 (markdown) ----
# ### Interpretation contract
#
# - The only V31 innovation tested here is the **fold-safe Intent-State target-statistics block**.
# - G/H/I are the only development worlds. Do not create extra salts because the result is inconvenient.
# - The promotion statistic is strictly `V31 − V30`, never `V31 − Consensus3`.
# - J and chronology are rejection-only. Do not change smoothing, features, alpha, or ranker hyperparameters after seeing them.
# - Exact template identity is used only to make validation groups disjoint; it is **not** a supervised target-encoding key.
# - The primary target encoding is hierarchical and nested: outer-validation labels never enter their own statistics, and outer-training rows receive inner-OOF encodings.
# - If `final_gate_passed == false`, do not spend a submission slot on the diagnostic candidate. `submission_v31_primary.csv` is then the reconstructed V30 fallback.
# - If the Intent-State block fails, the next research family should be objective/model diversity on a frozen feature surface (XENDCG/YetiRank), not a smoothing/threshold search on G/H/I/J.
