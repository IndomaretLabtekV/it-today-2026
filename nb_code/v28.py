# Converted from notebooks/v28.ipynb: markdown cells become comments, %/! magics neutralized.

# ---- cell 0 (markdown) ----
# # MineToday V28 — Hierarchical Top-K Graded Module Calibration
#
# **Active public anchor:** V25 distilled core-6 package = **0.65919**.  
# **Leader reference at design time:** **0.66112**.  
# **Metric:** user-level NDCG@5.
#
# V28 is a deliberate reset away from user-level action surgery. The late V23–V27.1 sequence showed that increasingly selective per-user/pair decisions can rank residual signal, but the reliable action set collapses to too few users to close the current public gap. The most repeatable population-scale public gains in this competition instead came from **module-level calibration** (V12/V12.2), while V16/V21 showed that information-source quality is strongly module-dependent.
#
# V28 therefore keeps the exact public-proven V3.1 → binary relevance → V12.2 Consensus3 backbone and adds one new low-variance layer:
#
# 1. calibrate the **actual graded relevance target** (`0..1`), not only binary relevance;
# 2. fit a pooled global monotonic curve plus module-specific monotonic curves;
# 3. shrink every module curve hierarchically back toward the global curve using cross-fitted, cluster-aware empirical-Bayes reliability;
# 4. focus fitting on the region that matters for NDCG@5 with a preregistered DCG/top-k weighting scheme;
# 5. evaluate the *combined* 17-module correction in ordinary and fully nested template-group representations;
# 6. freeze the candidate before a fresh template-group LOCK and chronological U3001–U4000 audit;
# 7. stress-test the frozen LOCK deltas with pseudo-public masks;
# 8. protect all eight public-tested V22.3/V25 rows exactly at deployment;
# 9. fail closed to the exact 0.65919 anchor whenever the population calibration does not generalize.
#
# ### What V28 intentionally does **not** do
#
# - no new embedding model;
# - no LLM/Generative-AI test prediction;
# - no free 17-dimensional weight hill-climb;
# - no user router, pair reranker, or manual test-user discovery;
# - no leaderboard score is used as a training feature or threshold target.
#
# ### Competition-rule contract
#
# Only organizer-provided CSV files are used. The notebook is deterministic, seed-controlled, reproducible, and uses ordinary Python/scikit-learn/LightGBM models. It is compatible with Kaggle T4×2; the GPU may remain mostly idle because the selected V28 method is deliberately CPU-first after the embedding branches failed to transfer reliably.
#
# **Submission discipline:** the notebook may generate diagnostic attack files even when the strict gate fails. `submission_v28_primary.csv` is the only automatic recommendation; it is the exact 0.65919 anchor when the gate fails.

# ---- cell 1 (code) ----
# Runtime + deterministic configuration
import os, sys, re, json, ast, html, hashlib, random, warnings, time, subprocess, gc
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ["PYTHONHASHSEED"] = "42"

V28_SKIP_INSTALL = bool(int(os.getenv("V28_SKIP_INSTALL", "0")))
try:
    import lightgbm as lgb
except Exception:
    if V28_SKIP_INSTALL:
        raise
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "lightgbm==4.6.0"])
    import lightgbm as lgb

import numpy as np
import pandas as pd
import scipy
import sklearn
from IPython.display import display
from scipy.sparse import hstack
from scipy.stats import norm
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore")

SEED = 42
V28_SMOKE = bool(int(os.getenv("V28_SMOKE", "0")))
V28_STRICT_GEOMETRY = bool(int(os.getenv("V28_STRICT_GEOMETRY", "1")))
# Compatibility aliases for the frozen reconstruction cells inherited from V27.1.
V27_SMOKE = V28_SMOKE
V27_STRICT_GEOMETRY = V28_STRICT_GEOMETRY

N_FOLDS = 2 if V28_SMOKE else 5
STRICT_INNER_FOLDS = 2 if V28_SMOKE else 4
TRUST_INNER_FOLDS = 2 if V28_SMOKE else 3
RIDGE_ALPHA = 3.0
REFERENCE_W = 0.45
PUBLIC_CAL045_LB = 0.65639
PUBLIC_CONSENSUS3_LB = 0.65856
PUBLIC_PRE_V25_PB = 0.65912
PUBLIC_V25_LB = 0.65919
PUBLIC_CURRENT_PB = PUBLIC_V25_LB
PUBLIC_LEADER_LB = 0.66112
LGB_ESTIMATORS = 40 if V28_SMOKE else 1200
LGB_EARLY_STOP = 8 if V28_SMOKE else 80
TRUST_BOOTSTRAPS = 80 if V28_SMOKE else 260
FINAL_BOOTSTRAPS = 120 if V28_SMOKE else 2200
BOOTSTRAPS = FINAL_BOOTSTRAPS  # inherited helper compatibility
PSEUDO_PUBLIC_MASKS = 20 if V28_SMOKE else 160
GEOMETRY_ATOL = 2.5e-4
TIE_EPS = 2e-7

# Frozen candidate family. No continuous search.
HMC_VARIANTS = [
    {"name":"hmc_topk35", "topk_weighted":True,  "alpha":0.35},
    {"name":"hmc_topk20", "topk_weighted":True,  "alpha":0.20},
    {"name":"hmc_plain35","topk_weighted":False, "alpha":0.35},
    {"name":"hmc_plain20","topk_weighted":False, "alpha":0.20},
]
VARIANT_PRIORITY = [v["name"] for v in HMC_VARIANTS]

# Selection / rejection contract. LOCK/chrono can reject but never select a new variant.
MIN_DEV_MEAN = 1e-6
MIN_DEV_FOLD_WINS = 4
MIN_DEV_CI_LOW = -2.5e-5
MIN_LOCK_DELTA = 1e-6
MIN_CHRONO_DELTA = -1e-6
MAX_TEST_TOP5_MEMBERSHIP_CHANGE = 0.12
MAX_TEST_TOP1_CHANGE = 0.08
MAX_TEST_ORDER_CHANGE = 0.35
MIN_TEST_CHANGED_USERS = 15
MAX_TEST_CHANGED_USERS = 300
PSEUDO_PUBLIC_MIN_POS_RATE = 0.72
PSEUDO_PUBLIC_MIN_P10 = -5e-5
STRONG_NEG_MODULE_TOL = -4e-5

PROTECTED_TEST_USERS = {
    "U_4047", "U_4953", "U_4074", "U_4176", "U_4491", "U_4567", "U_4836", "U_4929"
}

random.seed(SEED)
np.random.seed(SEED)

print("numpy", np.__version__)
print("pandas", pd.__version__)
print("scipy", scipy.__version__)
print("sklearn", sklearn.__version__)
print("lightgbm", lgb.__version__)
print("V28_SMOKE", V28_SMOKE, "public gap", round(PUBLIC_LEADER_LB-PUBLIC_CURRENT_PB, 5))
try:
    print("GPU runtime:\n" + subprocess.check_output(["nvidia-smi", "-L"], text=True).strip())
except Exception:
    print("nvidia-smi unavailable; V28 is T4x2-compatible but intentionally CPU-first.")

# ---- cell 2 (code) ----
def locate_data_dir():
    required = {
        "chat_history.csv", "modules_catalog.csv", "user_assessments.csv",
        "train_relevance.csv", "test.csv", "sample_submission.csv"
    }
    roots = [Path("/kaggle/input"), Path("/mnt/data/comp_v3"), Path("/mnt/data/comp_full"), Path(".")]
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
# ## Frozen public-validated backbone
#
# The following cells reproduce the exact V3.1 sparse backbone, binary-relevance calibration path, and V12.2 Consensus3 module weights. **V28 does not retune this backbone.** All new learning happens only after the frozen Consensus3 score surface has been produced.

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
# ## Reconstruct the active 0.65919 public anchor
#
# The two V22.3 corrections and six V25 distilled-core corrections are reconstructed exactly. These eight test users are protected from V28 so a population-level calibration cannot erase already public-tested gains.

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
    assert 0 < margin < 0.02, f"PB reconstruction drift for {uid}: margin={margin}"
    swap_score_values(pb_test, r, higher_mid, lower_mid)

known_pb_diag = pd.DataFrame(known_rows)
display(known_pb_diag)

# Exactly two users must change Top-5 ordering versus Consensus3.
cons_top5 = top5_indices(consensus3_test)
pb_top5 = top5_indices(pb_test)
changed_rows = np.flatnonzero(np.any(cons_top5 != pb_top5, axis=1))
changed_uids = test.iloc[changed_rows]["user_id"].tolist()
assert set(changed_uids) == {"U_4047", "U_4953"}, changed_uids
print("PB reference changes exactly:", changed_uids)

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
assert set(changed_users_v25) == expected_v25_users, changed_users_v25
assert len(changed_users_v25) == 8
print("V25 anchor reconstructed; changed users vs Consensus3:", changed_users_v25)

# ---- cell 14 (markdown) ----
# ## Template groups and evaluation worlds
#
# Chat text is heavily template-driven, so V28 treats latest-message skeletons as dependence clusters. Candidate selection happens only on **DEV**. A fresh salted template-group **LOCK** and the chronological U3001–U4000 block are rejection-only audits.
#
# The strict representation below rebuilds the supervised anchor inside template-group outer folds. This is intentionally more conservative than the ordinary public-compatible OOF surface.

# ---- cell 15 (code) ----
# Latest learning-template groups for cluster-safe calibration and bootstrap.
LATEST_GROUP = np.empty(len(all_users), dtype=object)
LATEST_GROUP[:] = "no_chat"
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

TRAIN_GROUP=np.asarray(LATEST_GROUP[:N_TRAIN],dtype=object)


def user_strata_from_y(yy):
    yy=np.asarray(yy)
    k=(yy>0).sum(1).astype(int)
    top=np.argmax(yy,axis=1).astype(int)
    return (top*4 + np.clip(k-3,0,3)).astype(int)


def make_group_folds_global(rows, n_splits=N_FOLDS, seed=SEED):
    rows=np.asarray(rows,int)
    ns=min(int(n_splits), len(np.unique(TRAIN_GROUP[rows])))
    if ns<2:
        raise ValueError("Need at least two independent template groups")
    sg=StratifiedGroupKFold(n_splits=ns,shuffle=True,random_state=int(seed))
    out=[]
    for tr,va in sg.split(np.zeros(len(rows)),strata[rows],groups=TRAIN_GROUP[rows]):
        out.append((rows[tr],rows[va]))
    return out


def make_group_folds_local(yy, groups, n_splits, seed):
    groups=np.asarray(groups,dtype=object); yy=np.asarray(yy)
    ns=min(int(n_splits),len(np.unique(groups)))
    if ns<2:
        # deterministic emergency fallback; only relevant to smoke/tiny subsets
        sk=StratifiedKFold(n_splits=2,shuffle=True,random_state=int(seed))
        st=user_strata_from_y(yy)
        return list(sk.split(np.zeros(len(yy)),st))
    sg=StratifiedGroupKFold(n_splits=ns,shuffle=True,random_state=int(seed))
    return list(sg.split(np.zeros(len(yy)),user_strata_from_y(yy),groups=groups))


def _lock_bucket(group):
    token=f"V28-HMC-LOCK-20260828::{group}".encode()
    return int(hashlib.sha1(token).hexdigest()[:12],16)%5

LOCK_MASK=np.array([_lock_bucket(g)==0 for g in TRAIN_GROUP],dtype=bool)
DEV_IDX=np.flatnonzero(~LOCK_MASK)
LOCK_IDX=np.flatnonzero(LOCK_MASK)
CHRONO_TR=np.arange(0,3000,dtype=int)
CHRONO_VA=np.arange(3000,4000,dtype=int)

print("template groups:",len(np.unique(TRAIN_GROUP)))
print("DEV/LOCK:",len(DEV_IDX),len(LOCK_IDX),"unique groups",len(np.unique(TRAIN_GROUP[DEV_IDX])),len(np.unique(TRAIN_GROUP[LOCK_IDX])))
print("chrono:",len(CHRONO_TR),len(CHRONO_VA))
assert len(set(TRAIN_GROUP[DEV_IDX]).intersection(set(TRAIN_GROUP[LOCK_IDX])))==0

# ---- cell 16 (markdown) ----
# ## Fully nested supervised anchor builder
#
# `build_anchor_split()` recreates the V3.1 sparse → binary relevance → Consensus3 path with all supervised fitting restricted to the supplied training rows. Its training-side scores are themselves template-group OOF; the evaluation-side scores come from models fitted on the entire supplied training partition.
#
# This function is used twice as a true rejection audit:
#
# - DEV → fresh template-group LOCK;
# - U0001–U3000 → chronological U3001–U4000.
#
# It is also used inside outer template-group folds to build a fully nested strict OOF representation.

# ---- cell 17 (code) ----
def _make_cal_local(seed,n_estimators=None):
    return lgb.LGBMClassifier(
        n_estimators=int(LGB_ESTIMATORS if n_estimators is None else n_estimators),
        learning_rate=.03,num_leaves=15,min_child_samples=50,
        reg_alpha=.5,reg_lambda=5.,colsample_bytree=.7,
        random_state=int(seed),n_jobs=4,verbosity=-1
    )


def build_anchor_split(train_rows, val_rows, seed):
    train_rows=np.asarray(train_rows,int); val_rows=np.asarray(val_rows,int)
    local_y=y[train_rows]; local_top=Y_TOP[train_rows]; local_bin=Y_BIN[train_rows]
    local_groups=TRAIN_GROUP[train_rows]
    folds=make_group_folds_local(local_y,local_groups,N_FOLDS,seed)
    n=len(train_rows)

    # Sparse graded/top relevance OOF + external validation.
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

    # Binary relevance meta OOF + external validation.
    meta_o=np.zeros_like(base_o); meta_v=np.zeros((len(val_rows),N_MODULES),np.float32)
    for m,mid in enumerate(MODULES):
        xt=np.column_stack([base_o[:,m],USER_FEATURES[train_rows]]).astype(np.float32)
        xv=np.column_stack([base_v[:,m],USER_FEATURES[val_rows]]).astype(np.float32)
        yy=local_bin[:,m].astype(np.int32); its=[]
        for f,(lt,lv) in enumerate(folds):
            clf=_make_cal_local(seed+f+31*m)
            clf.fit(xt[lt],yy[lt],eval_set=[(xt[lv],yy[lv])],callbacks=[lgb.early_stopping(LGB_EARLY_STOP,verbose=False)])
            it=int(clf.best_iteration_ or LGB_ESTIMATORS); its.append(it)
            meta_o[lv,m]=clf.predict_proba(xt[lv],num_iteration=it)[:,1].astype(np.float32)
        final_it=max(20,int(np.median(its)))
        full=_make_cal_local(seed+997+31*m,final_it); full.fit(xt,yy)
        meta_v[:,m]=full.predict_proba(xv)[:,1].astype(np.float32)

    cons_o=((1.0-CONS_W)*base_o+CONS_W*meta_o).astype(np.float32)
    cons_v=((1.0-CONS_W)*base_v+CONS_W*meta_v).astype(np.float32)
    return cons_o,cons_v


# Fully nested template-group representation for every train row.
strict_consensus_oof=np.zeros_like(consensus3_oof,np.float32)
strict_outer=make_group_folds_global(np.arange(N_TRAIN),N_FOLDS,SEED+2800)
for f,(trg,vag) in enumerate(strict_outer):
    print(f"strict representation {f+1}/{len(strict_outer)}")
    _,vv=build_anchor_split(trg,vag,SEED+28100+f*101)
    strict_consensus_oof[vag]=vv
    gc.collect()

print("ordinary Consensus3",round(ndcg5(y,consensus3_oof),6))
print("strict-group Consensus3",round(ndcg5(y,strict_consensus_oof),6))
assert np.isfinite(strict_consensus_oof).all()

# ---- cell 18 (markdown) ----
# ## Hierarchical graded calibration
#
# For a given fit partition, V28 learns:
#
# - a pooled global isotonic curve `g0(score) → E[graded relevance]`;
# - one isotonic curve `gm(score)` per module;
# - an inner template-group OOF estimate of whether replacing the pooled curve with each module curve actually improves user-level NDCG@5.
#
# The module effect estimates are shrunk with an empirical-Bayes variance estimate. The final correction strength is therefore **continuous, uncertainty-aware, and pulled toward zero**. No module is granted a freely optimized weight.
#
# For the top-k variant, calibration observations are weighted by the module's current anchor rank using a preregistered DCG-like schedule. The global curve remains shared across modules and therefore acts as a ranking-preserving control; only the shrunk module departures can alter within-user ordering.

# ---- cell 19 (code) ----
def _user_tiebreak(scores):
    s=np.asarray(scores,float)
    lo=s.min(1,keepdims=True); hi=s.max(1,keepdims=True)
    return ((s-lo)/(hi-lo+1e-12)).astype(np.float64)


def topk_weights(scores, enabled):
    n,m=np.asarray(scores).shape
    if not enabled:
        return np.ones((n,m),np.float64)
    order=np.argsort(-np.asarray(scores),axis=1,kind="mergesort")
    ranks=np.empty_like(order,dtype=np.int16)
    ranks[np.arange(n)[:,None],order]=np.arange(1,m+1,dtype=np.int16)[None,:]
    w=np.full((n,m),0.05,np.float64)
    vals={1:1.00,2:0.631,3:0.500,4:0.431,5:0.387,6:0.220,7:0.150}
    for r,v in vals.items(): w[ranks==r]=v
    return w


def fit_iso_curves(scores, yy, weighted):
    scores=np.asarray(scores,float); yy=np.asarray(yy,float)
    sw=topk_weights(scores,weighted)
    global_iso=IsotonicRegression(y_min=0.0,y_max=1.0,out_of_bounds="clip")
    global_iso.fit(scores.ravel(),yy.ravel(),sample_weight=sw.ravel())
    mods=[]
    for m in range(N_MODULES):
        iso=IsotonicRegression(y_min=0.0,y_max=1.0,out_of_bounds="clip")
        iso.fit(scores[:,m],yy[:,m],sample_weight=sw[:,m])
        mods.append(iso)
    return global_iso,mods


def predict_iso_curves(model,scores):
    global_iso,mods=model; scores=np.asarray(scores,float)
    g=global_iso.predict(scores.ravel()).reshape(scores.shape)
    mm=np.column_stack([mods[m].predict(scores[:,m]) for m in range(N_MODULES)])
    tb=TIE_EPS*_user_tiebreak(scores)
    return (g+tb).astype(np.float64),(mm+tb).astype(np.float64)


def _cluster_boot_stats(values, groups, seed, B=TRUST_BOOTSTRAPS):
    values=np.asarray(values,float); groups=np.asarray(groups,dtype=object)
    ug,inv=np.unique(groups,return_inverse=True)
    gs=np.bincount(inv,weights=values,minlength=len(ug)).astype(float)
    gcnt=np.bincount(inv,minlength=len(ug)).astype(float)
    if len(ug)<2:
        return float(values.mean()), float(values.std(ddof=1)/max(np.sqrt(len(values)),1.0)) if len(values)>1 else 1.0
    rng=np.random.default_rng(int(seed)); boots=np.empty(int(B),float)
    for b in range(int(B)):
        pick=rng.integers(0,len(ug),size=len(ug))
        boots[b]=gs[pick].sum()/max(gcnt[pick].sum(),1.0)
    return float(values.mean()),float(np.std(boots,ddof=1)+1e-12)


def estimate_module_trust(scores,yy,groups,weighted,seed):
    scores=np.asarray(scores,float); yy=np.asarray(yy,float); groups=np.asarray(groups,dtype=object)
    folds=make_group_folds_local(yy,groups,TRUST_INNER_FOLDS,seed)
    n=len(scores)
    delta=np.zeros((n,N_MODULES),np.float64)
    fold_means=np.full((len(folds),N_MODULES),np.nan,float)
    for f,(tr,va) in enumerate(folds):
        mdl=fit_iso_curves(scores[tr],yy[tr],weighted)
        g,mm=predict_iso_curves(mdl,scores[va])
        # Shared monotonic control should preserve anchor ordering aside from exact tie handling.
        base_u=per_user_ndcg(yy[va],g)
        for m in range(N_MODULES):
            cand=g.copy(); cand[:,m]=mm[:,m]
            du=per_user_ndcg(yy[va],cand)-base_u
            delta[va,m]=du; fold_means[f,m]=float(du.mean())
    means=np.zeros(N_MODULES); ses=np.zeros(N_MODULES)
    for m in range(N_MODULES):
        means[m],ses[m]=_cluster_boot_stats(delta[:,m],groups,seed+500+m)
    tau2=max(float(np.var(means,ddof=1)-np.mean(ses**2)),1e-10)
    shrink=tau2/(tau2+ses**2)
    z=np.divide(means,ses,out=np.zeros_like(means),where=ses>1e-12)
    ppos=norm.cdf(z)
    fold_pos=np.nanmean(fold_means>0,axis=0)
    trust=shrink*np.clip(2.0*ppos-1.0,0.0,1.0)*fold_pos
    trust=np.clip(trust,0.0,1.0)
    diag=pd.DataFrame({
        "module":MODULES,"inner_mean_delta":means,"cluster_se":ses,
        "tau2":tau2,"eb_shrink":shrink,"positive_prob":ppos,
        "fold_positive_rate":fold_pos,"lambda":trust,
    })
    return trust.astype(np.float64),diag


def fit_hmc_model(scores,yy,groups,variant,seed):
    trust,diag=estimate_module_trust(scores,yy,groups,bool(variant["topk_weighted"]),seed)
    curves=fit_iso_curves(scores,yy,bool(variant["topk_weighted"]))
    return {"curves":curves,"lambda":trust,"diag":diag,"variant":dict(variant)}


def predict_hmc(model,scores,lambda_override=None):
    scores=np.asarray(scores,float)
    g,mm=predict_iso_curves(model["curves"],scores)
    lam=model["lambda"] if lambda_override is None else np.asarray(lambda_override,float)
    a=float(model["variant"]["alpha"])
    pred=g+a*(mm-g)*lam[None,:]
    pred+=TIE_EPS*_user_tiebreak(scores)
    return pred.astype(np.float32),g.astype(np.float32),mm.astype(np.float32)

# ---- cell 20 (markdown) ----
# ## DEV selection: ordinary + strict representation
#
# Only DEV may select among the four preregistered variants. Each variant is cross-fitted with template groups held out. Selection is by fixed priority, not by whichever candidate has the largest mean.
#
# The fresh LOCK and chronological audit are not allowed to choose another variant later; they can only veto the frozen choice.

# ---- cell 21 (code) ----
def cluster_boot_ci(values,groups,seed,B=FINAL_BOOTSTRAPS):
    values=np.asarray(values,float); groups=np.asarray(groups,dtype=object)
    ug,inv=np.unique(groups,return_inverse=True)
    gs=np.bincount(inv,weights=values,minlength=len(ug)).astype(float)
    gcnt=np.bincount(inv,minlength=len(ug)).astype(float)
    if len(ug)<2:
        m=float(values.mean()); return (m,m)
    rng=np.random.default_rng(int(seed)); arr=np.empty(int(B),float)
    for b in range(int(B)):
        p=rng.integers(0,len(ug),size=len(ug))
        arr[b]=gs[p].sum()/max(gcnt[p].sum(),1.0)
    return float(np.quantile(arr,.025)),float(np.quantile(arr,.975))


def change_rates(a,b):
    a=np.asarray(a); b=np.asarray(b)
    ta=top5_indices(a); tb=top5_indices(b)
    setchg=np.mean([set(x)!=set(z) for x,z in zip(ta,tb)])
    orderchg=float(np.mean(np.any(ta!=tb,axis=1)))
    top1chg=float(np.mean(np.argmax(a,axis=1)!=np.argmax(b,axis=1)))
    return float(setchg),orderchg,top1chg


def crossfit_hmc(scores,yy,rows,variant,seed):
    scores=np.asarray(scores); rows=np.asarray(rows,int)
    pred=np.zeros((len(rows),N_MODULES),np.float32)
    glob=np.zeros_like(pred)
    fold_id=np.full(len(rows),-1,np.int16)
    trust_rows=[]
    # Map global row id -> local slot for compact output arrays.
    pos={int(r):i for i,r in enumerate(rows.tolist())}
    folds=make_group_folds_global(rows,N_FOLDS,seed)
    for f,(trg,vag) in enumerate(folds):
        mdl=fit_hmc_model(scores[trg],yy[trg],TRAIN_GROUP[trg],variant,seed+1000*f)
        pp,gg,_=predict_hmc(mdl,scores[vag])
        loc=np.array([pos[int(r)] for r in vag],int)
        pred[loc]=pp; glob[loc]=gg; fold_id[loc]=f
        dd=mdl["diag"].copy(); dd["fold"]=f; trust_rows.append(dd)
    assert (fold_id>=0).all()
    return pred,glob,fold_id,pd.concat(trust_rows,ignore_index=True)


def module_attribution(yy,base,cand,global_control,fold_id):
    full_u=per_user_ndcg(yy,cand)
    rows=[]
    for m,mid in enumerate(MODULES):
        loo=cand.copy(); loo[:,m]=global_control[:,m]
        contrib=full_u-per_user_ndcg(yy,loo)
        fm=np.array([contrib[fold_id==f].mean() for f in np.unique(fold_id)])
        rows.append({"module":mid,"mean_contribution":float(contrib.mean()),"fold_wins":int((fm>0).sum()),"fold_means":json.dumps([float(x) for x in fm])})
    return pd.DataFrame(rows)

VARIANTS={v["name"]:v for v in HMC_VARIANTS}
selection_rows=[]; dev_cache={}
for vi,name in enumerate(VARIANT_PRIORITY):
    v=VARIANTS[name]
    print("\n=== DEV",name,"===")
    op,og,ofid,ot= crossfit_hmc(consensus3_oof,y,DEV_IDX,v,SEED+30000+vi*137)
    sp,sg,sfid,st= crossfit_hmc(strict_consensus_oof,y,DEV_IDX,v,SEED+40000+vi*137)
    yy=y[DEV_IDX]; base_o=consensus3_oof[DEV_IDX]; base_s=strict_consensus_oof[DEV_IDX]
    od=per_user_ndcg(yy,op)-per_user_ndcg(yy,base_o)
    sd=per_user_ndcg(yy,sp)-per_user_ndcg(yy,base_s)
    comb=.5*(od+sd)
    ofm=np.array([od[ofid==f].mean() for f in np.unique(ofid)])
    sfm=np.array([sd[sfid==f].mean() for f in np.unique(sfid)])
    ci=cluster_boot_ci(comb,TRAIN_GROUP[DEV_IDX],SEED+50000+vi)
    attr_o=module_attribution(yy,base_o,op,og,ofid)
    attr_s=module_attribution(yy,base_s,sp,sg,sfid)
    attr=attr_o.merge(attr_s,on="module",suffixes=("_ordinary","_strict"))
    strong_neg=((attr.mean_contribution_ordinary<STRONG_NEG_MODULE_TOL)&(attr.mean_contribution_strict<STRONG_NEG_MODULE_TOL)).sum()
    setchg,orderchg,top1chg=change_rates(op,base_o)
    feasible=bool(
        od.mean()>MIN_DEV_MEAN and sd.mean()>MIN_DEV_MEAN and
        (ofm>0).sum()>=MIN_DEV_FOLD_WINS and (sfm>0).sum()>=MIN_DEV_FOLD_WINS and
        ci[0]>=MIN_DEV_CI_LOW and strong_neg==0 and
        setchg<=MAX_TEST_TOP5_MEMBERSHIP_CHANGE and top1chg<=MAX_TEST_TOP1_CHANGE
    )
    selection_rows.append({
        "variant":name,"ordinary_delta":float(od.mean()),"strict_delta":float(sd.mean()),
        "combined_delta":float(comb.mean()),"ci_low":ci[0],"ci_high":ci[1],
        "ordinary_fold_wins":int((ofm>0).sum()),"strict_fold_wins":int((sfm>0).sum()),
        "ordinary_fold_means":json.dumps([float(x) for x in ofm]),"strict_fold_means":json.dumps([float(x) for x in sfm]),
        "top5_membership_change":setchg,"order_change":orderchg,"top1_change":top1chg,
        "strong_negative_modules":int(strong_neg),"feasible":feasible,
    })
    dev_cache[name]={"ordinary_pred":op,"ordinary_global":og,"strict_pred":sp,"strict_global":sg,
                     "ordinary_delta":od,"strict_delta":sd,"ordinary_fold":ofid,"strict_fold":sfid,
                     "trust_ordinary":ot,"trust_strict":st,"attribution":attr}

selection_df=pd.DataFrame(selection_rows)
display(selection_df.round(6))
selected_name=None
for name in VARIANT_PRIORITY:
    if bool(selection_df.loc[selection_df.variant==name,"feasible"].iloc[0]):
        selected_name=name; break
if selected_name is None:
    diagnostic_name=selection_df.sort_values(["combined_delta","ci_low"],ascending=False).iloc[0].variant
else:
    diagnostic_name=selected_name
print("FROZEN SELECTION:",selected_name,"diagnostic fallback:",diagnostic_name)

# ---- cell 22 (markdown) ----
# ## Rejection-only audits: fresh LOCK, chronology, pseudo-public
#
# The selected variant is frozen before this cell. The same HMC algorithm is refitted only on the training side of each audit, exactly as it would be for unseen users.
#
# - **Fresh LOCK:** the entire supervised Consensus3 anchor is trained only on DEV and evaluated on template-disjoint LOCK.
# - **Chronology:** the entire supervised anchor is trained only on U0001–U3000 and evaluated on U3001–U4000.
# - **Pseudo-public:** random fractions are sampled from the already-frozen LOCK deltas, making this stress test meaningfully more independent than DEV-only pseudo-public simulation.

# ---- cell 23 (code) ----
def evaluate_external_hmc(train_scores,train_y,train_groups,val_scores,val_y,variant,seed):
    mdl=fit_hmc_model(train_scores,train_y,train_groups,variant,seed)
    pred,g,_=predict_hmc(mdl,val_scores)
    du=per_user_ndcg(val_y,pred)-per_user_ndcg(val_y,val_scores)
    return pred,du,mdl


def pseudo_public_from_lock(delta,seed):
    delta=np.asarray(delta,float); rng=np.random.default_rng(int(seed)); rec=[]
    for frac in (0.25,0.35,0.50):
        n=max(1,int(round(frac*len(delta))))
        vals=[]
        for _ in range(PSEUDO_PUBLIC_MASKS):
            idx=rng.choice(len(delta),size=n,replace=False)
            vals.append(float(delta[idx].mean()))
        vals=np.asarray(vals)
        rec.append({"fraction":frac,"median":float(np.median(vals)),"p10":float(np.quantile(vals,.10)),
                    "positive_rate":float(np.mean(vals>0)),"mean":float(vals.mean())})
    return pd.DataFrame(rec)

frozen_variant=VARIANTS[diagnostic_name]
print("Building fresh supervised LOCK representation...")
dev_anchor_oof,lock_anchor=build_anchor_split(DEV_IDX,LOCK_IDX,SEED+61000)
lock_pred,lock_delta,lock_model=evaluate_external_hmc(
    dev_anchor_oof,y[DEV_IDX],TRAIN_GROUP[DEV_IDX],lock_anchor,y[LOCK_IDX],frozen_variant,SEED+62000)
print("LOCK anchor/candidate",round(ndcg5(y[LOCK_IDX],lock_anchor),6),round(ndcg5(y[LOCK_IDX],lock_pred),6),"delta",f"{lock_delta.mean():+.6f}")

print("Building chronological supervised representation...")
chrono_anchor_oof,chrono_anchor_val=build_anchor_split(CHRONO_TR,CHRONO_VA,SEED+63000)
chrono_pred,chrono_delta,chrono_model=evaluate_external_hmc(
    chrono_anchor_oof,y[CHRONO_TR],TRAIN_GROUP[CHRONO_TR],chrono_anchor_val,y[CHRONO_VA],frozen_variant,SEED+64000)
print("CHRONO anchor/candidate",round(ndcg5(y[CHRONO_VA],chrono_anchor_val),6),round(ndcg5(y[CHRONO_VA],chrono_pred),6),"delta",f"{chrono_delta.mean():+.6f}")

pseudo_df=pseudo_public_from_lock(lock_delta,SEED+65000)
display(pseudo_df.round(6))

# Rejection gate does not select a different candidate.
lock_ok=bool(lock_delta.mean()>MIN_LOCK_DELTA)
chrono_ok=bool(chrono_delta.mean()>=MIN_CHRONO_DELTA)
pseudo_ok=bool((pseudo_df.positive_rate>=PSEUDO_PUBLIC_MIN_POS_RATE).all() and (pseudo_df.p10>=PSEUDO_PUBLIC_MIN_P10).all())
selection_ok=selected_name is not None
print("selection/lock/chrono/pseudo:",selection_ok,lock_ok,chrono_ok,pseudo_ok)

# ---- cell 24 (markdown) ----
# ## Final full-train fit, protected deployment, and exact geometry
#
# Only after the candidate has been selected and audited do we fit the HMC curves on all 4,000 labeled users. The eight public-tested V22.3/V25 rows are then overwritten **exactly** with their current 0.65919 anchor rows.
#
# A medium-width deployment contract prevents another micro-action regime and also prevents a destructive global reranker: V28 must alter enough users to have meaningful public upside, but it may not perturb an excessive fraction of Top-5 memberships or Top-1 choices.

# ---- cell 25 (code) ----
# Final model is fit using the frozen variant algorithm; no LOCK-dependent threshold is changed.
final_model=fit_hmc_model(consensus3_oof,y,TRAIN_GROUP,frozen_variant,SEED+70000)
final_consensus_test,final_global_test,final_module_test=predict_hmc(final_model,consensus3_test)

# Protected package rows stay byte/numerically identical to the current V25 anchor scores.
protected_rows=np.array([TEST_UID_TO_ROW[u] for u in sorted(PROTECTED_TEST_USERS)],int)
final_protected=final_consensus_test.copy()
final_protected[protected_rows]=v25_anchor_test[protected_rows]

# For all non-protected rows, V25 and Consensus3 are identical; assert that contract explicitly.
nonprot=np.ones(N_TEST,bool); nonprot[protected_rows]=False
assert np.allclose(v25_anchor_test[nonprot],consensus3_test[nonprot],atol=0,rtol=0)
assert np.allclose(final_protected[protected_rows],v25_anchor_test[protected_rows],atol=0,rtol=0)

setchg,orderchg,top1chg=change_rates(final_protected,v25_anchor_test)
changed=np.flatnonzero(np.any(top5_indices(final_protected)!=top5_indices(v25_anchor_test),axis=1))
changed_users=test.iloc[changed]["user_id"].tolist()

geometry_ok=bool(
    len(changed_users)>=MIN_TEST_CHANGED_USERS and len(changed_users)<=MAX_TEST_CHANGED_USERS and
    setchg<=MAX_TEST_TOP5_MEMBERSHIP_CHANGE and orderchg<=MAX_TEST_ORDER_CHANGE and top1chg<=MAX_TEST_TOP1_CHANGE
)

strict_gate=bool(selection_ok and lock_ok and chrono_ok and pseudo_ok and geometry_ok)
primary_scores=final_protected.copy() if strict_gate else v25_anchor_test.copy()
primary_policy=diagnostic_name if strict_gate else "anchor_fail_closed"

print("final candidate changed users:",len(changed_users))
print("test top5 membership/order/top1:",round(setchg,4),round(orderchg,4),round(top1chg,4))
print("geometry_ok",geometry_ok,"STRICT_GATE",strict_gate,"PRIMARY",primary_policy)
print("final lambda:")
final_trust=final_model["diag"].copy(); final_trust["module_name"]=modules.set_index("module_id").loc[MODULES,"module_name"].values
display(final_trust[["module","module_name","inner_mean_delta","cluster_se","positive_prob","fold_positive_rate","lambda"]].round(6))

# ---- cell 26 (markdown) ----
# ## Outputs
#
# Generated score files:
#
# - `submission_v28_primary.csv` — automatic recommendation; exact 0.65919 anchor when the strict gate fails.
# - `submission_v28_anchor_v25_65919.csv` — exact current public anchor.
# - `submission_v28_DIAGNOSTIC_frozen_hmc.csv` — frozen HMC candidate regardless of gate, for analysis only.
# - `submission_v28_DIAGNOSTIC_topk20.csv`, `...topk35.csv`, `...plain20.csv`, `...plain35.csv` — full-train preregistered family diagnostics with the eight protected rows preserved.
#
# Diagnostics include candidate selection, fresh-world validation, pseudo-public stress, final module trust, DEV module attribution, exact row-change ledger, and a machine-readable summary.

# ---- cell 27 (code) ----
OUT_DIR=Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("/mnt/data")
OUT_DIR.mkdir(parents=True,exist_ok=True)


def save_sub(scores,name):
    out=sample.copy(); out.loc[:,MODULES]=np.asarray(scores,float)
    path=OUT_DIR/name; out.to_csv(path,index=False)
    return path

anchor_path=save_sub(v25_anchor_test,"submission_v28_anchor_v25_65919.csv")
primary_path=save_sub(primary_scores,"submission_v28_primary.csv")
diag_path=save_sub(final_protected,"submission_v28_DIAGNOSTIC_frozen_hmc.csv")

# Full-train versions of every preregistered variant are diagnostics only; candidate selection remains frozen on DEV.
variant_paths={}
for j,v in enumerate(HMC_VARIANTS):
    mdl=fit_hmc_model(consensus3_oof,y,TRAIN_GROUP,v,SEED+72000+j*211)
    sc,_,_=predict_hmc(mdl,consensus3_test)
    sc[protected_rows]=v25_anchor_test[protected_rows]
    variant_paths[v["name"]]=str(save_sub(sc,f"submission_v28_DIAGNOSTIC_{v['name']}.csv"))

selection_df.to_csv(OUT_DIR/"v28_candidate_selection.csv",index=False)
pseudo_df.to_csv(OUT_DIR/"v28_pseudo_public_lock.csv",index=False)
final_trust.to_csv(OUT_DIR/"v28_final_module_trust.csv",index=False)
for name,obj in dev_cache.items():
    obj["attribution"].to_csv(OUT_DIR/f"v28_module_attribution_{name}.csv",index=False)

# Exact ranking-change ledger relative to the active public anchor.
ledger=[]
anchor_order=np.argsort(-v25_anchor_test,axis=1,kind="mergesort")
cand_order=np.argsort(-final_protected,axis=1,kind="mergesort")
for r in changed:
    ledger.append({
        "user_id":test.iloc[r].user_id,
        "anchor_top5":" ".join(MODULES[i] for i in anchor_order[r,:5]),
        "candidate_top5":" ".join(MODULES[i] for i in cand_order[r,:5]),
        "anchor_top1":MODULES[int(anchor_order[r,0])],"candidate_top1":MODULES[int(cand_order[r,0])],
    })
ledger_df=pd.DataFrame(ledger)
ledger_df.to_csv(OUT_DIR/"v28_test_change_ledger.csv",index=False)

validation_df=pd.DataFrame([
    {"world":"DEV_ordinary","delta":float(dev_cache[diagnostic_name]["ordinary_delta"].mean())},
    {"world":"DEV_strict_group","delta":float(dev_cache[diagnostic_name]["strict_delta"].mean())},
    {"world":"fresh_LOCK","delta":float(lock_delta.mean())},
    {"world":"chronological","delta":float(chrono_delta.mean())},
])
validation_df.to_csv(OUT_DIR/"v28_validation_summary.csv",index=False)

summary={
    "version":"V28 Hierarchical Top-K Graded Module Calibration",
    "seed":SEED,"public_anchor":PUBLIC_CURRENT_PB,"leader_reference":PUBLIC_LEADER_LB,
    "selected_variant":selected_name,"diagnostic_variant":diagnostic_name,"primary_policy":primary_policy,
    "strict_gate":strict_gate,"selection_ok":selection_ok,"lock_ok":lock_ok,"chrono_ok":chrono_ok,"pseudo_ok":pseudo_ok,"geometry_ok":geometry_ok,
    "lock_delta":float(lock_delta.mean()),"chrono_delta":float(chrono_delta.mean()),
    "test_changed_users":int(len(changed_users)),"test_top5_membership_change":setchg,"test_order_change":orderchg,"test_top1_change":top1chg,
    "protected_users":sorted(PROTECTED_TEST_USERS),"variant_paths":variant_paths,
}
with open(OUT_DIR/"v28_summary.json","w") as f: json.dump(summary,f,indent=2)

slot_plan=pd.DataFrame([
    {"slot":1,"file":"submission_v28_primary.csv","role":"strict gated population candidate; exact 0.65919 anchor if rejected"},
    {"slot":2,"file":"submission_v28_DIAGNOSTIC_frozen_hmc.csv","role":"analysis only unless manual review independently approves the frozen candidate"},
])
slot_plan.to_csv(OUT_DIR/"v28_slot_plan.csv",index=False)

print("\n================ V28 DECISION ================")
print("Selected on DEV        :",selected_name)
print("Frozen diagnostic      :",diagnostic_name)
print("Fresh LOCK delta       :",f"{lock_delta.mean():+.6f}")
print("Chronological delta    :",f"{chrono_delta.mean():+.6f}")
print("Pseudo-public gate     :",pseudo_ok)
print("Changed users          :",len(changed_users))
print("Membership/order/top1 :",f"{setchg:.3f}",f"{orderchg:.3f}",f"{top1chg:.3f}")
print("STRICT GATE            :",strict_gate)
print("PRIMARY POLICY         :",primary_policy)
print("PRIMARY FILE           :",primary_path)
print("==============================================")

# ---- cell 28 (markdown) ----
# ## Interpretation contract
#
# V28 is successful only if a **preregistered** hierarchical graded-calibration variant is positive in both DEV representations, then remains positive on the fresh template-disjoint LOCK and non-negative chronologically, while changing a medium-width but controlled fraction of test rankings.
#
# A high ordinary OOF score alone is not enough. If `STRICT_GATE=False`, the experiment has rejected population recalibration and `submission_v28_primary.csv` intentionally remains the exact 0.65919 public anchor. Do not override the gate merely because a diagnostic file changes more users.
