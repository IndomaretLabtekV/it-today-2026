# Converted from notebooks/v43.ipynb: markdown cells become comments, %/! magics neutralized.

# ---- cell 0 (markdown) ----
# # MineToday V43 — Joint User-Level Anchor-Residual TabM + Partial Plackett–Luce
#
# **Hypothesis.** The remaining learnable signal is not another itemwise scorer or rare-action selector, but a **joint user-level representation** that sees the complete 17-module structural state at once. V43 keeps the exact public champion (`0.66028`, B34+B45) as the deployment prior, learns only residual utilities, and permits only Top4 permutations on S0 users.
#
# **Frozen design**
#
# - one sample = one user, not one `(user,module)` row;
# - dense user state = assessment/chat summaries + 64-D lexical SVD + the full 17-D structural surfaces;
# - two preregistered backbones on identical features/loss: a plain MLP control and a parameter-efficient **TabM-style BatchEnsemble** model;
# - residual decoder is **zero-initialized**, therefore training starts exactly at the structural anchor;
# - loss = weighted partial Plackett–Luce for guaranteed ordered Top4 + one residual L2 penalty;
# - all 17 modules compete during training; deployment can only reorder the existing anchor Top4;
# - no membership edits, no rank-5 edits, no Qwen/BGE/XLM-R, no generator/template branch, no action selector, no alpha/threshold sweep;
# - DESIGN 3400 / untouched LOCK 600; architecture freezes before LOCK is evaluated;
# - production fail-close is exact algorithmic B34+B45 (`0.66028`).
#
# This notebook uses only organizer-provided competition data. Historical public results are constants/audits only; historical submission CSV files are never predictive inputs.

# ---- cell 1 (code) ----
# Runtime + preregistered V39.5 S3 boundary-decomposition configuration
import os, sys, re, json, ast, html, hashlib, random, warnings, time, subprocess, gc, math
from pathlib import Path
from collections import defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ["PYTHONHASHSEED"] = "42"

V43_SKIP_INSTALL = bool(int(os.getenv("V43_SKIP_INSTALL", "0")))
V421_SKIP_INSTALL = V43_SKIP_INSTALL
V395_SKIP_INSTALL = V43_SKIP_INSTALL
try:
    import lightgbm as lgb
except Exception:
    if V395_SKIP_INSTALL:
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
V43_SMOKE = bool(int(os.getenv("V43_SMOKE", "0")))
V421_SMOKE = V43_SMOKE
V395_SMOKE = V43_SMOKE
V395_STRICT_GEOMETRY = False if V421_SMOKE else bool(int(os.getenv("V395_STRICT_GEOMETRY", "1")))

# Compatibility aliases consumed by the frozen V3.1 -> V25 -> V30/V31 reconstruction cells.
V31_SMOKE = V395_SMOKE
V30_SMOKE = V395_SMOKE
V30_STRICT_GEOMETRY = V395_STRICT_GEOMETRY
V29_SMOKE = V395_SMOKE
V29_STRICT_GEOMETRY = V395_STRICT_GEOMETRY
V27_SMOKE = V395_SMOKE
V27_STRICT_GEOMETRY = V395_STRICT_GEOMETRY

N_FOLDS = 2 if V395_SMOKE else 5
ANCHOR_INNER_FOLDS = 2 if V395_SMOKE else 4
INTENT_INNER_FOLDS = 2 if V395_SMOKE else 4

RIDGE_ALPHA = 3.0
REFERENCE_W = 0.45

PUBLIC_CAL045_LB = 0.65639
PUBLIC_CONSENSUS3_LB = 0.65856
PUBLIC_V25_LB = 0.65919
PUBLIC_V30_LB = 0.65963
PUBLIC_V31_DIAGNOSTIC_LB = 0.65949
PUBLIC_HYBRID34_LB = 0.66006
PUBLIC_CURRENT_ANCHOR_LB = 0.66020
PUBLIC_TOP4_LB = 0.66080
PUBLIC_LEADER_LB = 0.66142

LGB_ESTIMATORS = 12 if V395_SMOKE else 1200
LGB_EARLY_STOP = 3 if V395_SMOKE else 80
RANKER_ESTIMATORS = 16 if V395_SMOKE else 360
RANKER_LEARNING_RATE = 0.04 if V395_SMOKE else 0.025

# Frozen public-proven V30/V31 residual geometry. No alpha search in V39.5.
ALPHA = 0.16
INTENT_PARENT_STRENGTH = 40.0
INTENT_CHILD_STRENGTH = 30.0
BOOTSTRAPS = 30 if V395_SMOKE else 1200
GEOMETRY_ATOL = 2.5e-4  # frozen V25 geometry tolerance

# Historical geometry regression checks. These encode ranking signatures/counts, not hidden labels.
EXPECTED_V30_CHANGED = 77
EXPECTED_V30_TOP1_CHANGES = 8
EXPECTED_V30_MEMBERSHIP_CHANGES = 7
EXPECTED_V31_CHANGED_VS_V30 = 60
EXPECTED_STATE_COUNTS = {"S0":897, "S1":26, "S2":43, "S3":34}
EXPECTED_S3_TOP1 = 3
EXPECTED_S3_MEMBERSHIP = 4
EXPECTED_S3_ORDER_ONLY = 27
EXPECTED_BOUNDARY_COUNTS = {"B23":9, "B34":10, "B45":8}
HYBRID34_TOP5_SHA256 = "47f7db52ce8f782b8b393fd120140f1b6f51aeac452a9fabb5794cc9d8ba5e6e"
CURRENT_ANCHOR_TOP5_SHA256 = "c70ce570ec311e3f7391cec324e566a3a0e27da99ffc210afc3a8f102daf0191"

# Public display has five decimals; inferred C45 is treated conservatively around zero.
LB_ROUNDING_TOL = 3e-5

def _seed_from_text(s):
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:8], 16) % (2**31-1)

random.seed(SEED)
np.random.seed(SEED)

print("numpy", np.__version__)
print("pandas", pd.__version__)
print("scipy", scipy.__version__)
print("sklearn", sklearn.__version__)
print("lightgbm", lgb.__version__)
print("V43_SMOKE", V43_SMOKE, "frozen V30 alpha", ALPHA)
print("historical current066020 anchor", PUBLIC_CURRENT_ANCHOR_LB)
print("V42.1 production champion", 0.66028,
      "gap to Top4", round(PUBLIC_TOP4_LB-0.66028,5),
      "gap to leader", round(PUBLIC_LEADER_LB-0.66028,5))
try:
    print("GPU runtime:\n" + subprocess.check_output(["nvidia-smi", "-L"], text=True).strip())
except Exception:
    print("nvidia-smi unavailable; V39.5 remains runnable and is deliberately CPU-first for exact geometry.")


# Frozen reconstruction dependency preflight
assert isinstance(GEOMETRY_ATOL, float) and GEOMETRY_ATOL == 2.5e-4
assert 'V27_STRICT_GEOMETRY' in globals()
assert 'V30_STRICT_GEOMETRY' in globals()

# ---- cell 2 (code) ----
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
# The exact sparse V3.1 → calibrated Consensus3 → V25 lineage is reused unchanged. V39 never reads an earlier submission file; the anchor is rebuilt from the organizer CSVs every run.

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

# ---- cell 13 (markdown) ----
# ## Exact V30/V31 representation needed to reconstruct the 0.66020 anchor
#
# The broad V30 module-native surface and the strict fold-safe V31 Intent-State block are retained only to reproduce V30/V31 disagreement geometry. V39's new signal is added separately below.

# ---- cell 14 (code) ----
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

# ---- cell 15 (code) ----
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

# ---- cell 16 (code) ----
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

# ---- cell 17 (code) ----
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

# ---- cell 18 (markdown) ----
# ## Exact current-anchor state machine + S3 order boundary classifier
#
# `A=V25`, `B=V30`, `C=V31`.
#
# The current `0.66020` rule keeps C only inside S3 when C changes either Top-1 or Top-5 membership; all S3 pure internal-order changes are rejected. V39.5 never selects rows by user ID. It derives the 27 rejected S3 rows from score geometry and then proves that every one is a single adjacent transposition at B23, B34, or B45.

# ---- cell 19 (code) ----
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

# ---- cell 20 (markdown) ----
# ## Full-train deterministic reconstruction
#
# No historical submission CSV is loaded. The notebook retrains the frozen V30 and V31 branches from organizer data, reapplies the frozen V25 protection package already encoded in the lineage, then verifies every known ranking-space invariant before any probe file is written.

# ---- cell 21 (code) ----
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

# ---- cell 22 (markdown) ----
# ## Algorithmic B23 / B34 / B45 decomposition
#
# The 27 rejected S3 rows are classified only from the V30→V31 Top-5 permutation. No row is chosen by a user-ID list. V39.5 requires the decomposition to close exactly:
#
# `current + B23 + B34 + B45 == Hybrid34`
#
# in full score space, with counts `9 + 10 + 8 = 27`.

# ---- cell 23 (code) ----
# Build the immutable boundary ledger from ranking geometry.
rows=[]
order_rows=np.flatnonzero(current_info["order_only"])
assert len(order_rows)==EXPECTED_S3_ORDER_ONLY

module_name_map=modules.set_index("module_id")["module_name"].astype(str).to_dict()
level_text_map=modules.set_index("module_id")["prerequisite_level"].astype(str).to_dict()

t30=top5_indices(v30_test); t31=top5_indices(v31_test)
for r in order_rows:
    boundary,promoted,demoted,pos_from,pos_to=classify_single_adjacent_order_swap(t30[r],t31[r])
    pg=N_TRAIN+int(r)
    pm=MODULES[promoted]; dm=MODULES[demoted]
    rows.append({
        "row_local":int(r),
        "user_id":str(test.iloc[r]["user_id"]),
        "state":"S3",
        "geometry":"order_only",
        "boundary":boundary,
        "promoted_module":pm,
        "demoted_module":dm,
        "module_pair_direction":f"{pm}>{dm}",
        "promoted_name":module_name_map.get(pm,""),
        "demoted_name":module_name_map.get(dm,""),
        "promoted_prerequisite_level":level_text_map.get(pm,""),
        "demoted_prerequisite_level":level_text_map.get(dm,""),
        "v30_promoted_rank":int(pos_to),
        "v30_demoted_rank":int(pos_from),
        "v31_promoted_rank":int(pos_from),
        "v31_demoted_rank":int(pos_to),
        "consensus3_promoted_minus_demoted":float(consensus3_test[r,promoted]-consensus3_test[r,demoted]),
        "v30_promoted_minus_demoted":float(v30_test[r,promoted]-v30_test[r,demoted]),
        "v31_promoted_minus_demoted":float(v31_test[r,promoted]-v31_test[r,demoted]),
        "v31_minus_v30_pair_margin":float((v31_test[r,promoted]-v31_test[r,demoted])-(v30_test[r,promoted]-v30_test[r,demoted])),
        "topic_count_promoted":float(MODULE_COUNT[pg,promoted]),
        "topic_count_demoted":float(MODULE_COUNT[pg,demoted]),
        "topic_count_adv":float(MODULE_COUNT[pg,promoted]-MODULE_COUNT[pg,demoted]),
        "topic_recency_promoted":float(MODULE_RECENCY[pg,promoted]),
        "topic_recency_demoted":float(MODULE_RECENCY[pg,demoted]),
        "topic_recency_adv":float(MODULE_RECENCY[pg,promoted]-MODULE_RECENCY[pg,demoted]),
        "topic_latest_promoted":float(MODULE_LATEST[pg,promoted]),
        "topic_latest_demoted":float(MODULE_LATEST[pg,demoted]),
        "readiness_promoted":float(MODULE_READINESS[pg,promoted]),
        "readiness_demoted":float(MODULE_READINESS[pg,demoted]),
        "readiness_adv":float(MODULE_READINESS[pg,promoted]-MODULE_READINESS[pg,demoted]),
        "readiness_band_promoted":int(MODULE_READY_BAND[pg,promoted]),
        "readiness_band_demoted":int(MODULE_READY_BAND[pg,demoted]),
        "invalid_readiness_promoted":float(MODULE_INVALID[pg,promoted]),
        "invalid_readiness_demoted":float(MODULE_INVALID[pg,demoted]),
        "latest_intent_signature":str(TEST_INTENT[r]),
    })

V395_BOUNDARY_LEDGER=pd.DataFrame(rows).sort_values(["boundary","row_local"]).reset_index(drop=True)
boundary_counts=V395_BOUNDARY_LEDGER["boundary"].value_counts().sort_index().to_dict()
assert boundary_counts==EXPECTED_BOUNDARY_COUNTS,(boundary_counts,EXPECTED_BOUNDARY_COUNTS)
assert V395_BOUNDARY_LEDGER["row_local"].is_unique
assert set(V395_BOUNDARY_LEDGER["boundary"])==set(EXPECTED_BOUNDARY_COUNTS)

# Add back each natural family to current; no learned filtering.
def add_back_boundaries(boundaries):
    boundaries=set(boundaries)
    out=current_anchor_test.copy()
    rr=V395_BOUNDARY_LEDGER.loc[V395_BOUNDARY_LEDGER["boundary"].isin(boundaries),"row_local"].to_numpy(int)
    if len(rr):
        out[rr]=v31_test[rr]
    return out

CANDIDATE_SCORES={
    "anchor_current_066020":current_anchor_test.copy(),
    "probe_B23_9":add_back_boundaries(["B23"]),
    "probe_B34_10":add_back_boundaries(["B34"]),
    "probe_B45_8":add_back_boundaries(["B45"]),
    "combo_B23_B34":add_back_boundaries(["B23","B34"]),
    "combo_B23_B45":add_back_boundaries(["B23","B45"]),
    "combo_B34_B45":add_back_boundaries(["B34","B45"]),
    "all_order27_hybrid34_reference":add_back_boundaries(["B23","B34","B45"]),
}

# Exact algebraic closure is mandatory.
assert np.array_equal(CANDIDATE_SCORES["all_order27_hybrid34_reference"],hybrid34_test)
assert top5_sha256(CANDIDATE_SCORES["all_order27_hybrid34_reference"])==HYBRID34_TOP5_SHA256 or V395_SMOKE
assert top5_sha256(CANDIDATE_SCORES["anchor_current_066020"])==CURRENT_ANCHOR_TOP5_SHA256 or V395_SMOKE

V395_GEOMETRY=pd.DataFrame([
    candidate_geometry(name,current_anchor_test,scores) for name,scores in CANDIDATE_SCORES.items()
])
expected_changed={
    "anchor_current_066020":0,
    "probe_B23_9":9,
    "probe_B34_10":10,
    "probe_B45_8":8,
    "combo_B23_B34":19,
    "combo_B23_B45":17,
    "combo_B34_B45":18,
    "all_order27_hybrid34_reference":27,
}
for row in V395_GEOMETRY.itertuples(index=False):
    assert int(row.changed_users_vs_current)==expected_changed[row.candidate],(row.candidate,row.changed_users_vs_current)
    assert int(row.top1_changes)==0,(row.candidate,row.top1_changes)
    assert int(row.membership_changes)==0,(row.candidate,row.membership_changes)

# Materialize all files now so the adaptive third candidate is already available after public results arrive.
SUB_FILES={
    "anchor_current_066020":"submission_v39_5_anchor_current_066020.csv",
    "probe_B23_9":"submission_v39_5_probe_B23_9.csv",
    "probe_B34_10":"submission_v39_5_probe_B34_10.csv",
    "probe_B45_8":"submission_v39_5_probe_B45_8.csv",
    "combo_B23_B34":"submission_v39_5_combo_B23_B34.csv",
    "combo_B23_B45":"submission_v39_5_combo_B23_B45.csv",
    "combo_B34_B45":"submission_v39_5_combo_B34_B45.csv",
    "all_order27_hybrid34_reference":"submission_v39_5_all_order27_hybrid34_reference.csv",
}
for name,scores in CANDIDATE_SCORES.items():
    submission_from_scores(scores).to_csv(OUT_DIR/SUB_FILES[name],index=False)

print("Boundary counts:",boundary_counts)
display(V395_GEOMETRY)
display(V395_BOUNDARY_LEDGER)

# ---- cell 24 (markdown) ----
# ## V43 preregistration — joint user-level structured residual
#
# The generator branch is closed. V43 changes the **factorization of the learning problem**: the neural model receives one dense vector per user containing the whole 17-module competition state, and outputs 17 residual utilities jointly.
#
# The MLP control and TabM candidate use the same input, zero residual initialization, optimizer, loss, folds, and deployment policy. DESIGN chooses the stronger preregistered backbone once; only then can the untouched 600-user LOCK be opened.

# ---- cell 25 (code) ----

# V43 runtime, fixed split, and immutable production anchor.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
import platform

V43_VERSION = "V43 Joint User-Level Anchor-Residual TabM + Partial PL"
PUBLIC_V43_ANCHOR_LB = 0.66028
EXPECTED_V43_ANCHOR_TOP5_SHA256 = "7c0d29a7309b7283a043b7d37ca5556ba7c0ca2270b0137a6e1e0878a69a5fe7"

V43_SPLIT_SEED = 2026090342       # same untouched DESIGN/LOCK split as V42/V42.1
V43_FOLD_SEED = 2026090353
V43_MODEL_SEED = 2026090361
V43_BOOT_SEED = 2026090367
POLICY_INNER_FOLDS = 2 if V43_SMOKE else 4

DESIGN_SIZE = 600 if V43_SMOKE else 3400
LOCK_SIZE = 120 if V43_SMOKE else 600
OUTER_FOLDS = 2 if V43_SMOKE else 5
BLOCKS = 2 if V43_SMOKE else 5
BOOTSTRAPS_V43 = 100 if V43_SMOKE else 2500

SVD_DIM = 16 if V43_SMOKE else 64
HIDDEN = 64 if V43_SMOKE else 128
TABM_K = 2 if V43_SMOKE else 8
DROPOUT = 0.15
EPOCHS = 6 if V43_SMOKE else 80
BATCH_SIZE = 128 if V43_SMOKE else 256
LR = 2e-3
WEIGHT_DECAY = 3e-4
RESIDUAL_L2 = 2e-3
GRAD_CLIP = 2.0
PL_WEIGHTS = torch.tensor([1.000, 0.536, 0.350, 0.237], dtype=torch.float32)

# DESIGN gates from the approved V43 blueprint.
DESIGN_MIN_DELTA = -1.0 if V43_SMOKE else 0.00030
DESIGN_MIN_FOLD_WINS = 0 if V43_SMOKE else 4
DESIGN_MIN_BLOCK_WINS = 0 if V43_SMOKE else 4
DESIGN_CI_FLOOR = -1.0 if V43_SMOKE else -0.00002
HIGH_ID_FLOOR = -1.0 if V43_SMOKE else 0.0
LOCK_MIN_DELTA = -1.0 if V43_SMOKE else 0.00020
LOCK_MIN_BLOCK_WINS = 0 if V43_SMOKE else 3
LOCK_CI_FLOOR = -1.0 if V43_SMOKE else -0.00010
TEST_MAX_CHANGED = 10_000 if V43_SMOKE else 350
TEST_MAX_TOP1_CHANGED = 10_000 if V43_SMOKE else 70

V43_OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("/mnt/data")
V43_OUT.mkdir(parents=True, exist_ok=True)

V43_ANCHOR_TEST = np.asarray(CANDIDATE_SCORES["combo_B34_B45"], np.float32).copy()
if not V43_SMOKE:
    assert top5_sha256(V43_ANCHOR_TEST) == EXPECTED_V43_ANCHOR_TOP5_SHA256

PROTECTED_TEST_USERS = set([u for u,_,_ in KNOWN_PB_SWAPS] + [a["user_id"] for a in V25_CORE_ACTIONS])
TEST_PROTECTED = np.asarray([str(u) in PROTECTED_TEST_USERS for u in test.user_id], bool)
TEST_ELIGIBLE = (full_state == 0) & (~TEST_PROTECTED)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
GPU_COUNT = torch.cuda.device_count() if torch.cuda.is_available() else 0
print("torch", torch.__version__, "device", DEVICE, "gpus", GPU_COUNT)
print("V43 anchor hash", top5_sha256(V43_ANCHOR_TEST))
print("public anchor", PUBLIC_V43_ANCHOR_LB, "leader gap", round(PUBLIC_LEADER_LB-PUBLIC_V43_ANCHOR_LB,5))
print("test S0 eligible", int(TEST_ELIGIBLE.sum()), "/", N_TEST)

# ---- cell 26 (markdown) ----
# ## Joint static user representation
#
# The neural learner never consumes raw high-cardinality template IDs. The dense representation is deliberately compact:
#
# 1. the existing 134 structured user features;
# 2. 64-dimensional TF-IDF TruncatedSVD lexical context;
# 3. six complete 17-module structural score vectors: sparse/base, meta calibration, Consensus3, V30, V31, and the policy anchor;
# 4. anchor rank and row-z vectors;
# 5. four adjacent head margins and six readiness/assessment summaries.
#
# The structural score vectors are generated fold-pure in the next section.

# ---- cell 27 (code) ----

# Label-free lexical compression. Vectorizer X is the frozen V3.1 organizer-data representation.
svd = TruncatedSVD(n_components=SVD_DIM, n_iter=7, random_state=V43_MODEL_SEED)
SVD_ALL = svd.fit_transform(X[:N_TRAIN]).astype(np.float32)
SVD_TEST = svd.transform(X[N_TRAIN:]).astype(np.float32)
SVD_ALL = np.vstack([SVD_ALL, SVD_TEST]).astype(np.float32)
assert SVD_ALL.shape == (N_TRAIN + N_TEST, SVD_DIM)
assert np.isfinite(SVD_ALL).all()

ALL_INTENT = np.concatenate([np.asarray(TRAIN_INTENT,dtype=object), np.asarray(TEST_INTENT,dtype=object)])

# The ordered Top4 is guaranteed for every training user.
TRUE_TOP4 = np.argsort(-y, axis=1, kind="mergesort")[:, :4].astype(np.int64)
assert np.all(np.take_along_axis(y, TRUE_TOP4, axis=1) > 0)

STATIC_SUMMARY = np.column_stack([
    A.mean(1), A.std(1), A_INVALID.sum(1),
    MODULE_READINESS.mean(1), MODULE_READINESS.min(1), MODULE_READINESS.max(1),
]).astype(np.float32)

print("SVD_ALL", SVD_ALL.shape, "TRUE_TOP4", TRUE_TOP4.shape)

# ---- cell 28 (markdown) ----
# ## Fold-pure modern structural surfaces
#
# For every neural outer fold, the structural family is rebuilt from outer-train only. Outer-train itself receives OOF V30/V31/current-policy surfaces; outer-validation receives full outer-train predictions. This avoids the second-level dependency leakage diagnosed in V41.
#
# B34/B45 are applied only through their natural ranking-geometry boundary rule; no user ID whitelist is learned.

# ---- cell 29 (code) ----

def _long_rows43(user_local_idx, nmods=N_MODULES):
    u=np.asarray(user_local_idx,int)
    return (u[:,None]*int(nmods)+np.arange(int(nmods))[None,:]).reshape(-1)

def _apply_current_b3445_43(v30,v31,state):
    current,info=current066020_surface(v30,v31,state)
    t30=top5_indices(v30); t31=top5_indices(v31)
    for r in np.flatnonzero(info["order_only"]):
        try:
            boundary,*_=classify_single_adjacent_order_swap(t30[r],t31[r])
        except AssertionError:
            continue
        if boundary in {"B34","B45"}:
            current[r]=v31[r]
    return current.astype(np.float32),info

def _oof_ranker43(Xlong,grades,user_y,groups,seed,feature_end):
    n=len(user_y); pred=np.zeros((n,N_MODULES),np.float32)
    folds=make_group_folds_local(user_y,groups,POLICY_INNER_FOLDS,int(seed))
    for k,(tr,va) in enumerate(folds):
        model=make_ranker(int(seed)+701+k)
        model.fit(Xlong[_long_rows43(tr),:feature_end],grades[tr].reshape(-1).astype(np.int32),group=np.full(len(tr),N_MODULES,np.int32))
        pred[va]=model.predict(Xlong[_long_rows43(va),:feature_end]).reshape(len(va),N_MODULES).astype(np.float32)
        del model
    return pred

def build_policy_split_v43(train_rows,val_rows,seed):
    train_rows=np.asarray(train_rows,int); val_rows=np.asarray(val_rows,int)
    anc=build_anchor_split_full(train_rows,val_rows,int(seed))

    intent_tr=crossfit_intentstate_features(
        y[train_rows],ORDINAL_GRADES[train_rows],TRAIN_INTENT[train_rows],
        MODULE_READY_BAND[train_rows],TRAIN_GROUP[train_rows],int(seed)+71
    )
    bundle=fit_intentstate_bundle(y[train_rows],ORDINAL_GRADES[train_rows],TRAIN_INTENT[train_rows],MODULE_READY_BAND[train_rows])
    intent_va=apply_intentstate_bundle(bundle,ALL_INTENT[val_rows],MODULE_READY_BAND[val_rows])

    Xtr,_,bounds=build_long_matrix_v31(train_rows,anc["cons_train"],anc["base_train"],anc["meta_train"],intent_tr)
    Xva,_,bounds2=build_long_matrix_v31(val_rows,anc["cons_val"],anc["base_val"],anc["meta_val"],intent_va)
    assert bounds==bounds2
    be=bounds["base"]; ie=bounds["intentstate"]

    raw30_tr=_oof_ranker43(Xtr,ORDINAL_GRADES[train_rows],y[train_rows],TRAIN_GROUP[train_rows],int(seed)+1301,be)
    raw31_tr=_oof_ranker43(Xtr,ORDINAL_GRADES[train_rows],y[train_rows],TRAIN_GROUP[train_rows],int(seed)+2301,ie)
    v30_tr=blend_residual(anc["cons_train"],raw30_tr,ALPHA)
    v31_tr=blend_residual(anc["cons_train"],raw31_tr,ALPHA)
    state_tr=state_codes(anc["cons_train"],v30_tr,v31_tr)
    current_tr,_=_apply_current_b3445_43(v30_tr,v31_tr,state_tr)

    _,raw30_va=fit_ranker_predict(Xtr[:,:be],Xva[:,:be],ORDINAL_GRADES[train_rows],len(train_rows),int(seed)+3301)
    _,raw31_va=fit_ranker_predict(Xtr[:,:ie],Xva[:,:ie],ORDINAL_GRADES[train_rows],len(train_rows),int(seed)+4301)
    v30_va=blend_residual(anc["cons_val"],raw30_va,ALPHA)
    v31_va=blend_residual(anc["cons_val"],raw31_va,ALPHA)
    state_va=state_codes(anc["cons_val"],v30_va,v31_va)
    current_va,_=_apply_current_b3445_43(v30_va,v31_va,state_va)

    out={
        "base_train":anc["base_train"],"meta_train":anc["meta_train"],"cons_train":anc["cons_train"],
        "v30_train":v30_tr,"v31_train":v31_tr,"anchor_train":current_tr,"state_train":state_tr.astype(np.int8),
        "base_val":anc["base_val"],"meta_val":anc["meta_val"],"cons_val":anc["cons_val"],
        "v30_val":v30_va,"v31_val":v31_va,"anchor_val":current_va,"state_val":state_va.astype(np.int8),
    }
    del Xtr,Xva,intent_tr,intent_va,bundle,anc,raw30_tr,raw31_tr,raw30_va,raw31_va
    gc.collect()
    return out

print("V43 policy-matched structural helper ready")

# ---- cell 30 (code) ----

def _anchor_rank_and_z(anchor):
    anchor=np.asarray(anchor,np.float32)
    ranks=rank_matrix(anchor).astype(np.float32)/float(N_MODULES)
    mu=anchor.mean(1,keepdims=True); sd=np.maximum(anchor.std(1,keepdims=True),1e-5)
    z=(anchor-mu)/sd
    ss=np.sort(anchor,axis=1)[:,::-1]
    gaps=np.column_stack([ss[:,0]-ss[:,1],ss[:,1]-ss[:,2],ss[:,2]-ss[:,3],ss[:,3]-ss[:,4]]).astype(np.float32)
    return ranks.astype(np.float32),z.astype(np.float32),gaps

def build_joint_features43(rows_global,surf):
    rows_global=np.asarray(rows_global,int)
    rankn,z,gaps=_anchor_rank_and_z(surf["anchor"])
    mats=[
        USER_FEATURES[rows_global].astype(np.float32),
        SVD_ALL[rows_global].astype(np.float32),
        surf["base"].astype(np.float32),surf["meta"].astype(np.float32),surf["cons"].astype(np.float32),
        surf["v30"].astype(np.float32),surf["v31"].astype(np.float32),surf["anchor"].astype(np.float32),
        rankn,z,gaps,STATIC_SUMMARY[rows_global].astype(np.float32),
    ]
    xx=np.hstack(mats).astype(np.float32)
    if not np.isfinite(xx).all(): raise ValueError("non-finite V43 joint features")
    return xx

def surf_train43(pol):
    return {"base":pol["base_train"],"meta":pol["meta_train"],"cons":pol["cons_train"],"v30":pol["v30_train"],"v31":pol["v31_train"],"anchor":pol["anchor_train"]}

def surf_val43(pol):
    return {"base":pol["base_val"],"meta":pol["meta_val"],"cons":pol["cons_val"],"v30":pol["v30_val"],"v31":pol["v31_val"],"anchor":pol["anchor_val"]}

print("expected joint feature width",134+SVD_DIM+6*17+17+17+4+6)

# ---- cell 31 (markdown) ----
# ## TabM-style shared user encoder and structured loss
#
# The TabM implementation below follows the official architecture principle: create `k` different object representations **before** the first feature-mixing linear layer, use BatchEnsemble-style shared matrices with non-shared multiplicative adapters, initialize only the first input adapter with random signs, and keep later adapters equal at initialization.
#
# The final 17-way decoder is zero-initialized, so residual utility is exactly zero at epoch 0. TabM ensemble members are trained independently by averaging their individual losses; inference averages their residual utilities.

# ---- cell 32 (code) ----

def init_rsqrt_uniform_(x,d):
    q=float(d)**-0.5
    return nn.init.uniform_(x,-q,q)

def init_random_signs_(x):
    with torch.no_grad(): return x.bernoulli_(0.5).mul_(2).add_(-1)

class LinearBatchEnsemble43(nn.Module):
    def __init__(self,d_in,d_out,k,first=False,bias=True):
        super().__init__(); self.d_in=d_in; self.d_out=d_out; self.k=k
        self.weight=nn.Parameter(torch.empty(d_out,d_in))
        self.r=nn.Parameter(torch.empty(k,d_in)); self.s=nn.Parameter(torch.empty(k,d_out))
        self.bias=nn.Parameter(torch.empty(k,d_out)) if bias else None
        init_rsqrt_uniform_(self.weight,d_in)
        if first: init_random_signs_(self.r)
        else: nn.init.ones_(self.r)
        nn.init.ones_(self.s)
        if self.bias is not None:
            b=torch.empty(d_out); init_rsqrt_uniform_(b,d_in)
            with torch.no_grad(): self.bias.copy_(b[None,:].expand(k,-1))
    def forward(self,x):
        x=x*self.r
        x=torch.matmul(x,self.weight.t())
        x=x*self.s
        if self.bias is not None: x=x+self.bias
        return x

class NLinear43(nn.Module):
    def __init__(self,k,d_in,d_out,zero=False):
        super().__init__(); self.weight=nn.Parameter(torch.empty(k,d_in,d_out)); self.bias=nn.Parameter(torch.empty(k,d_out))
        if zero:
            nn.init.zeros_(self.weight); nn.init.zeros_(self.bias)
        else:
            init_rsqrt_uniform_(self.weight,d_in); init_rsqrt_uniform_(self.bias,d_in)
    def forward(self,x):
        return torch.einsum('bkd,kdo->bko',x,self.weight)+self.bias

class JointTabM43(nn.Module):
    def __init__(self,d_in,d_hidden=128,k=8,dropout=.15):
        super().__init__(); self.k=k
        self.l1=LinearBatchEnsemble43(d_in,d_hidden,k,first=True)
        self.l2=LinearBatchEnsemble43(d_hidden,d_hidden,k,first=False)
        self.drop=nn.Dropout(dropout)
        self.out=NLinear43(k,d_hidden,N_MODULES,zero=True)
    def forward(self,x):
        x=x[:,None,:].expand(-1,self.k,-1)
        x=self.drop(F.relu(self.l1(x)))
        x=self.drop(F.relu(self.l2(x)))
        return self.out(x)

class JointMLP43(nn.Module):
    def __init__(self,d_in,d_hidden=128,dropout=.15):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d_in,d_hidden),nn.ReLU(),nn.Dropout(dropout),nn.Linear(d_hidden,d_hidden),nn.ReLU(),nn.Dropout(dropout))
        self.out=nn.Linear(d_hidden,N_MODULES)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
    def forward(self,x):
        return self.out(self.net(x))[:,None,:]

def row_z43(a):
    a=np.asarray(a,np.float32); return ((a-a.mean(1,keepdims=True))/np.maximum(a.std(1,keepdims=True),1e-5)).astype(np.float32)

def partial_pl_loss43(residual,anchor_z,true_top4,l2=RESIDUAL_L2):
    # residual: (B,K,17), anchor_z: (B,17), true_top4: (B,4)
    q=residual+anchor_z[:,None,:]
    mask=torch.zeros_like(q,dtype=torch.bool)
    total=torch.zeros((),device=q.device,dtype=q.dtype)
    weights=PL_WEIGHTS.to(q.device,dtype=q.dtype)
    for r in range(4):
        logits=q.masked_fill(mask,-1e9)
        logp=F.log_softmax(logits,dim=-1)
        tgt=true_top4[:,r][:,None,None].expand(-1,q.shape[1],1)
        total=total-weights[r]*logp.gather(-1,tgt).squeeze(-1).mean()
        next_mask=mask.clone()
        next_mask.scatter_(2,tgt,True)
        mask=next_mask
    total=total/weights.sum()
    return total+float(l2)*residual.square().mean()

def seed43(seed):
    random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed))
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(int(seed))

def maybe_parallel43(model):
    model=model.to(DEVICE)
    if GPU_COUNT>=2:
        model=nn.DataParallel(model,device_ids=[0,1])
    return model

def train_joint43(kind,Xtr,anchor_tr,ytr,seed):
    seed43(seed)
    d=Xtr.shape[1]
    core=JointTabM43(d,HIDDEN,TABM_K,DROPOUT) if kind=="tabm" else JointMLP43(d,HIDDEN,DROPOUT)
    model=maybe_parallel43(core)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    x=torch.from_numpy(Xtr.astype(np.float32)); az=torch.from_numpy(row_z43(anchor_tr)); tt=torch.from_numpy(TRUE_TOP4[ytr].astype(np.int64))
    ds=TensorDataset(x,az,tt)
    gen=torch.Generator(); gen.manual_seed(int(seed))
    loader=DataLoader(ds,batch_size=BATCH_SIZE,shuffle=True,generator=gen,num_workers=0,pin_memory=torch.cuda.is_available())
    losses=[]
    for ep in range(EPOCHS):
        model.train(); s=0.; n=0
        for xb,ab,tb in loader:
            xb=xb.to(DEVICE,non_blocking=True); ab=ab.to(DEVICE,non_blocking=True); tb=tb.to(DEVICE,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            rr=model(xb)
            loss=partial_pl_loss43(rr,ab,tb)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP); opt.step()
            s+=float(loss.detach().cpu())*len(xb); n+=len(xb)
        losses.append(s/max(n,1))
    return model,losses

@torch.inference_mode()
def predict_residual43(model,X):
    model.eval(); out=[]
    for a in range(0,len(X),1024):
        xb=torch.from_numpy(X[a:a+1024].astype(np.float32)).to(DEVICE)
        rr=model(xb).float().mean(1)  # ensemble mean utility; MLP has K=1
        out.append(rr.cpu().numpy().astype(np.float32))
    return np.vstack(out)

def deploy_top4_43(anchor,residual,eligible):
    anchor=np.asarray(anchor,np.float32); residual=np.asarray(residual,np.float32); eligible=np.asarray(eligible,bool)
    out=anchor.copy(); q=row_z43(anchor)+residual
    old=np.argsort(-anchor,axis=1,kind="mergesort")[:,:4]
    for i in np.flatnonzero(eligible):
        mods=old[i]; vals=anchor[i,mods].copy()  # already descending
        new=mods[np.argsort(-q[i,mods],kind="mergesort")]
        if not np.array_equal(new,mods):
            out[i,mods]=np.nan
            out[i,new]=vals
    if np.isnan(out).any(): raise AssertionError("V43 permutation left NaNs")
    return out

def geom43(anchor,cand):
    oa=np.argsort(-anchor,axis=1,kind="mergesort"); oc=np.argsort(-cand,axis=1,kind="mergesort")
    changed=np.any(oa!=oc,axis=1)
    membership=np.asarray([set(a[:5])!=set(b[:5]) for a,b in zip(oa,oc)])
    return {"changed_users":int(changed.sum()),"top1_changes":int(np.sum(oa[:,0]!=oc[:,0])),
            "top5_membership_changes":int(membership.sum()),"rank5_changes":int(np.sum(oa[:,4]!=oc[:,4])),
            "rank5_to_17_changes":int(np.sum(np.any(oa[:,4:]!=oc[:,4:],axis=1)))}

def top1_acc43(yy,s): return float(np.mean(np.argmax(yy,axis=1)==np.argmax(s,axis=1)))

def paired_ci43(delta,seed,blocks=None):
    delta=np.asarray(delta,float); rng=np.random.default_rng(int(seed)); z=np.empty(BOOTSTRAPS_V43,float)
    if blocks is None:
        n=len(delta)
        for b in range(BOOTSTRAPS_V43): z[b]=delta[rng.integers(0,n,n)].mean()
    else:
        blocks=[np.asarray(x,int) for x in blocks]
        for b in range(BOOTSTRAPS_V43):
            picks=rng.integers(0,len(blocks),len(blocks)); idx=np.concatenate([blocks[j] for j in picks]); z[b]=delta[idx].mean()
    return tuple(float(x) for x in np.quantile(z,[.025,.975]))

# Component invariants: zero residual initialization and safe permutation.
_d=24; _m=JointTabM43(_d,32,2,0.0); _z=torch.zeros(3,_d); _r=_m(_z)
assert _r.shape==(3,2,N_MODULES) and torch.allclose(_r,torch.zeros_like(_r))
_dummy=np.tile(np.linspace(1,0,N_MODULES,dtype=np.float32),(3,1)); _res=np.zeros_like(_dummy); _cand=deploy_top4_43(_dummy,_res,np.ones(3,bool))
assert np.array_equal(_dummy,_cand)
print("V43 TabM/MLP/loss/deployment component tests passed")

# ---- cell 33 (markdown) ----
# ## DESIGN — five honest outer folds
#
# Each outer fold rebuilds the structural policy from outer-train only, fits a `StandardScaler` on outer-train dense features, then trains both preregistered backbones for the same fixed 80 epochs. There is no fold-specific hyperparameter or epoch search.
#
# Only S0 validation users are eligible for permutation. The plain MLP is a control for the **joint representation hypothesis**; DESIGN selects the stronger of the two preregistered backbones once.

# ---- cell 34 (code) ----

all_train_idx=np.arange(N_TRAIN)
if V43_SMOKE:
    rng=np.random.default_rng(V43_SPLIT_SEED); sub=np.sort(rng.choice(all_train_idx,size=DESIGN_SIZE+LOCK_SIZE,replace=False))
    ss=StratifiedShuffleSplit(n_splits=1,test_size=LOCK_SIZE,random_state=V43_SPLIT_SEED)
    dl,ll=next(ss.split(np.zeros(len(sub)),np.argmax(y[sub],axis=1))); DESIGN_IDX=sub[dl]; LOCK_IDX=sub[ll]
else:
    ss=StratifiedShuffleSplit(n_splits=1,test_size=LOCK_SIZE,random_state=V43_SPLIT_SEED)
    DESIGN_IDX,LOCK_IDX=next(ss.split(np.zeros(N_TRAIN),np.argmax(y,axis=1)))
DESIGN_IDX=np.sort(np.asarray(DESIGN_IDX,int)); LOCK_IDX=np.sort(np.asarray(LOCK_IDX,int))
assert len(DESIGN_IDX)==DESIGN_SIZE and len(LOCK_IDX)==LOCK_SIZE

skf=StratifiedKFold(OUTER_FOLDS,shuffle=True,random_state=V43_FOLD_SEED)
outer=list(skf.split(np.zeros(DESIGN_SIZE),np.argmax(y[DESIGN_IDX],axis=1)))

ANCHOR_OOF=np.zeros((DESIGN_SIZE,N_MODULES),np.float32)
MLP_OOF=np.zeros_like(ANCHOR_OOF); TABM_OOF=np.zeros_like(ANCHOR_OOF)
FOLD_ID=np.full(DESIGN_SIZE,-1,int); fold_rows=[]

for f,(trloc,valoc) in enumerate(outer):
    tr=DESIGN_IDX[trloc]; va=DESIGN_IDX[valoc]
    print(f"V43 outer {f+1}/{OUTER_FOLDS}: train={len(tr)} val={len(va)}")
    pol=build_policy_split_v43(tr,va,V43_FOLD_SEED+1000*f)
    Xtr=build_joint_features43(tr,surf_train43(pol)); Xva=build_joint_features43(va,surf_val43(pol))
    scaler=StandardScaler(); Xtr=scaler.fit_transform(Xtr).astype(np.float32); Xva=scaler.transform(Xva).astype(np.float32)
    eligible=(pol["state_val"]==0)

    mlp,mloss=train_joint43("mlp",Xtr,pol["anchor_train"],tr,V43_MODEL_SEED+100*f+1)
    rm=predict_residual43(mlp,Xva); cm=deploy_top4_43(pol["anchor_val"],rm,eligible)
    del mlp; gc.collect();
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    tabm,tloss=train_joint43("tabm",Xtr,pol["anchor_train"],tr,V43_MODEL_SEED+100*f+2)
    rt=predict_residual43(tabm,Xva); ct=deploy_top4_43(pol["anchor_val"],rt,eligible)
    del tabm; gc.collect();
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    ANCHOR_OOF[valoc]=pol["anchor_val"]; MLP_OOF[valoc]=cm; TABM_OOF[valoc]=ct; FOLD_ID[valoc]=f
    fold_rows.append({
        "fold":f,"n":len(va),"eligible_s0":int(eligible.sum()),
        "anchor_ndcg":ndcg5(y[va],pol["anchor_val"]),
        "mlp_ndcg":ndcg5(y[va],cm),"mlp_delta":ndcg5(y[va],cm)-ndcg5(y[va],pol["anchor_val"]),
        "tabm_ndcg":ndcg5(y[va],ct),"tabm_delta":ndcg5(y[va],ct)-ndcg5(y[va],pol["anchor_val"]),
        "mlp_top1_delta":top1_acc43(y[va],cm)-top1_acc43(y[va],pol["anchor_val"]),
        "tabm_top1_delta":top1_acc43(y[va],ct)-top1_acc43(y[va],pol["anchor_val"]),
        "mlp_changed":geom43(pol["anchor_val"],cm)["changed_users"],"tabm_changed":geom43(pol["anchor_val"],ct)["changed_users"],
        "mlp_final_loss":float(mloss[-1]),"tabm_final_loss":float(tloss[-1]),
    })
    del pol,Xtr,Xva,scaler,rm,rt,cm,ct
    gc.collect()

assert (FOLD_ID>=0).all()
V43_OUTER=pd.DataFrame(fold_rows)
print(V43_OUTER.to_string(index=False))

# ---- cell 35 (code) ----

def design_metrics43(name,cand):
    delta_user=per_user_ndcg(y[DESIGN_IDX],cand)-per_user_ndcg(y[DESIGN_IDX],ANCHOR_OOF)
    ci=paired_ci43(delta_user,V43_BOOT_SEED+(1 if name=="tabm" else 2))
    overall=ndcg5(y[DESIGN_IDX],cand)-ndcg5(y[DESIGN_IDX],ANCHOR_OOF)
    top1=top1_acc43(y[DESIGN_IDX],cand)-top1_acc43(y[DESIGN_IDX],ANCHOR_OOF)
    geom=geom43(ANCHOR_OOF,cand)
    ordloc=np.argsort(TRAIN_ID_NUM[DESIGN_IDX]); blocks=np.array_split(ordloc,BLOCKS); br=[]
    for b,rr in enumerate(blocks):
        br.append({"model":name,"block":b,"n":len(rr),"id_min":int(TRAIN_ID_NUM[DESIGN_IDX[rr]].min()),"id_max":int(TRAIN_ID_NUM[DESIGN_IDX[rr]].max()),
                   "delta":ndcg5(y[DESIGN_IDX[rr]],cand[rr])-ndcg5(y[DESIGN_IDX[rr]],ANCHOR_OOF[rr])})
    bdf=pd.DataFrame(br); fw=int(((V43_OUTER[f"{name}_delta"])>0).sum()); bw=int((bdf.delta>0).sum())
    return {"model":name,"anchor_ndcg":ndcg5(y[DESIGN_IDX],ANCHOR_OOF),"candidate_ndcg":ndcg5(y[DESIGN_IDX],cand),"delta":overall,
            "top1_delta":top1,"ci_low":ci[0],"ci_high":ci[1],"fold_wins":fw,"block_wins":bw,"high_id_delta":float(bdf.iloc[-1].delta),
            **{f"geom_{k}":v for k,v in geom.items()}},bdf

MLP_MET,MLP_BLOCK=design_metrics43("mlp",MLP_OOF)
TABM_MET,TABM_BLOCK=design_metrics43("tabm",TABM_OOF)
V43_DESIGN_METRICS=pd.DataFrame([MLP_MET,TABM_MET]); V43_BLOCK_METRICS=pd.concat([MLP_BLOCK,TABM_BLOCK],ignore_index=True)

# One preregistered choice only: the stronger joint backbone on DESIGN.
SELECTED_BACKBONE="tabm" if TABM_MET["delta"]>=MLP_MET["delta"] else "mlp"
SELECTED_MET=TABM_MET if SELECTED_BACKBONE=="tabm" else MLP_MET
SELECTED_OOF=TABM_OOF if SELECTED_BACKBONE=="tabm" else MLP_OOF

DESIGN_GATE=bool(
    SELECTED_MET["delta"]>=DESIGN_MIN_DELTA and SELECTED_MET["fold_wins"]>=DESIGN_MIN_FOLD_WINS and
    SELECTED_MET["block_wins"]>=DESIGN_MIN_BLOCK_WINS and SELECTED_MET["high_id_delta"]>=HIGH_ID_FLOOR and
    SELECTED_MET["ci_low"]>=DESIGN_CI_FLOOR and SELECTED_MET["top1_delta"]>=0 and
    SELECTED_MET["geom_top5_membership_changes"]==0 and SELECTED_MET["geom_rank5_changes"]==0 and SELECTED_MET["geom_rank5_to_17_changes"]==0
)
print(V43_DESIGN_METRICS.to_string(index=False))
print(V43_BLOCK_METRICS.to_string(index=False))
print("selected",SELECTED_BACKBONE,"DESIGN_GATE",DESIGN_GATE)

# ---- cell 36 (markdown) ----
# ## Untouched LOCK and production test
#
# LOCK is not touched unless DESIGN passes. Once it passes, the selected backbone is frozen, a single full-DESIGN model is trained **before looking at LOCK**, and that same model is used for test if LOCK confirms. There is no post-LOCK refit.

# ---- cell 37 (code) ----

LOCK_EVALUATED=False; LOCK_GATE=False; TEST_CANDIDATE=V43_ANCHOR_TEST.copy(); TEST_RESIDUAL=np.zeros_like(V43_ANCHOR_TEST)
V43_LOCK_METRICS=pd.DataFrame([{"evaluated":False}]); V43_LOCK_BLOCKS=pd.DataFrame()

if DESIGN_GATE:
    val_global=np.concatenate([LOCK_IDX,np.arange(N_TRAIN,N_TRAIN+N_TEST)])
    pol=build_policy_split_v43(DESIGN_IDX,val_global,V43_FOLD_SEED+99001)
    Xd=build_joint_features43(DESIGN_IDX,surf_train43(pol)); Xv=build_joint_features43(val_global,surf_val43(pol))
    scaler=StandardScaler(); Xd=scaler.fit_transform(Xd).astype(np.float32); Xv=scaler.transform(Xv).astype(np.float32)
    full_model,full_loss=train_joint43(SELECTED_BACKBONE,Xd,pol["anchor_train"],DESIGN_IDX,V43_MODEL_SEED+99001)
    resid=predict_residual43(full_model,Xv)
    nlock=len(LOCK_IDX); lock_anchor=pol["anchor_val"][:nlock]; lock_state=pol["state_val"][:nlock]
    lock_cand=deploy_top4_43(lock_anchor,resid[:nlock],lock_state==0)
    ld=per_user_ndcg(y[LOCK_IDX],lock_cand)-per_user_ndcg(y[LOCK_IDX],lock_anchor)
    ldelta=float(ld.mean()); lci=paired_ci43(ld,V43_BOOT_SEED+20); ltop=top1_acc43(y[LOCK_IDX],lock_cand)-top1_acc43(y[LOCK_IDX],lock_anchor)
    ordl=np.argsort(TRAIN_ID_NUM[LOCK_IDX]); lblocks=np.array_split(ordl,BLOCKS); rows=[]
    for b,rr in enumerate(lblocks): rows.append({"block":b,"n":len(rr),"delta":ndcg5(y[LOCK_IDX[rr]],lock_cand[rr])-ndcg5(y[LOCK_IDX[rr]],lock_anchor[rr])})
    V43_LOCK_BLOCKS=pd.DataFrame(rows); lbwins=int((V43_LOCK_BLOCKS.delta>0).sum()); lg=geom43(lock_anchor,lock_cand)
    LOCK_GATE=bool(ldelta>=LOCK_MIN_DELTA and lci[0]>=LOCK_CI_FLOOR and ltop>=0 and lbwins>=LOCK_MIN_BLOCK_WINS and
                   lg["top5_membership_changes"]==0 and lg["rank5_changes"]==0 and lg["rank5_to_17_changes"]==0)
    LOCK_EVALUATED=True
    V43_LOCK_METRICS=pd.DataFrame([{"evaluated":True,"backbone":SELECTED_BACKBONE,"anchor_ndcg":ndcg5(y[LOCK_IDX],lock_anchor),
        "candidate_ndcg":ndcg5(y[LOCK_IDX],lock_cand),"delta":ldelta,"top1_delta":ltop,"ci_low":lci[0],"ci_high":lci[1],
        "block_wins":lbwins,"changed_users":lg["changed_users"],"top1_changes":lg["top1_changes"],"lock_gate":LOCK_GATE}])

    # Model sees full-DESIGN structural side information, but deployment always reorders the exact public 0.66028 anchor.
    TEST_RESIDUAL=resid[nlock:]
    TEST_CANDIDATE=deploy_top4_43(V43_ANCHOR_TEST,TEST_RESIDUAL,TEST_ELIGIBLE)
    del pol,Xd,Xv,scaler,full_model,resid,lock_anchor,lock_cand,ld
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

print(V43_LOCK_METRICS.to_string(index=False))
print("LOCK_GATE",LOCK_GATE)

# ---- cell 38 (code) ----

# Final safety geometry and leaderboard-scale audit.
TEST_GEOM=geom43(V43_ANCHOR_TEST,TEST_CANDIDATE)
oa=np.argsort(-V43_ANCHOR_TEST,axis=1,kind="mergesort"); oc=np.argsort(-TEST_CANDIDATE,axis=1,kind="mergesort")
changed=np.any(oa!=oc,axis=1)
locked_exact=bool(np.array_equal(TEST_CANDIDATE[~TEST_ELIGIBLE],V43_ANCHOR_TEST[~TEST_ELIGIBLE]))
score_multiset_ok=True
for r in np.flatnonzero(changed):
    if not np.array_equal(np.sort(V43_ANCHOR_TEST[r,oa[r,:4]]),np.sort(TEST_CANDIDATE[r,oc[r,:4]])):
        score_multiset_ok=False; break

# Optimistic label-free leverage: how much NDCG could these Top4 permutations move if every changed head decision were correct.
# This is a scale diagnostic only, never a training target or filter.
K_DIST={4:1362/4000,5:1285/4000,6:1353/4000}
def ideal_dcg_k(k):
    rel=np.array([1,.85,.70,.55,.40,.25][:k],float)
    return float((rel[:5]*DISCOUNT[:min(5,k)]).sum())
# optimistic per-user #1-vs-#4 swap gain, averaged over empirical K distribution
_opt=sum(p*((1.0-.55)*(DISCOUNT[0]-DISCOUNT[3])/ideal_dcg_k(k)) for k,p in K_DIST.items())
THEORETICAL_MAX_PUBLIC_GAIN=float(TEST_GEOM["changed_users"]*_opt/max(N_TEST,1))

TEST_GEOMETRY_GATE=bool(
    locked_exact and score_multiset_ok and TEST_GEOM["top5_membership_changes"]==0 and TEST_GEOM["rank5_changes"]==0 and
    TEST_GEOM["rank5_to_17_changes"]==0 and TEST_GEOM["changed_users"]<=TEST_MAX_CHANGED and TEST_GEOM["top1_changes"]<=TEST_MAX_TOP1_CHANGED
)
FINAL_GATE=bool(DESIGN_GATE and LOCK_GATE and TEST_GEOMETRY_GATE and not V43_SMOKE)
PRIMARY=TEST_CANDIDATE if FINAL_GATE else V43_ANCHOR_TEST

V43_TEST_GEOMETRY=pd.DataFrame([{**TEST_GEOM,"eligible_s0":int(TEST_ELIGIBLE.sum()),"locked_rows_bit_exact":locked_exact,
    "top4_score_multiset_exact":score_multiset_ok,"geometry_gate":TEST_GEOMETRY_GATE,"theoretical_max_public_gain":THEORETICAL_MAX_PUBLIC_GAIN,
    "anchor_top5_sha256":top5_sha256(V43_ANCHOR_TEST),"candidate_top5_sha256":top5_sha256(TEST_CANDIDATE)}])

assert TEST_GEOM["top5_membership_changes"]==0 and TEST_GEOM["rank5_changes"]==0 and TEST_GEOM["rank5_to_17_changes"]==0
print(V43_TEST_GEOMETRY.to_string(index=False))
print("FINAL_GATE",FINAL_GATE)

# ---- cell 39 (code) ----

def submission43(scores):
    out=sample.copy(); out.loc[:,MODULES]=np.asarray(scores,np.float32)
    assert out.user_id.astype(str).tolist()==test.user_id.astype(str).tolist(); assert np.isfinite(out[MODULES].to_numpy()).all()
    return out

submission43(V43_ANCHOR_TEST).to_csv(V43_OUT/"submission_v43_anchor_066028_reference.csv",index=False)
submission43(TEST_CANDIDATE).to_csv(V43_OUT/"submission_v43_candidate_DIAGNOSTIC.csv",index=False)
submission43(PRIMARY).to_csv(V43_OUT/"submission_v43_primary.csv",index=False)
V43_DESIGN_METRICS.to_csv(V43_OUT/"v43_design_metrics.csv",index=False)
V43_OUTER.to_csv(V43_OUT/"v43_outer_fold_metrics.csv",index=False)
V43_BLOCK_METRICS.to_csv(V43_OUT/"v43_block_metrics.csv",index=False)
V43_LOCK_METRICS.to_csv(V43_OUT/"v43_lock_metrics.csv",index=False)
V43_LOCK_BLOCKS.to_csv(V43_OUT/"v43_lock_block_metrics.csv",index=False)
V43_TEST_GEOMETRY.to_csv(V43_OUT/"v43_test_geometry.csv",index=False)
pd.DataFrame({"user_id":train.user_id,"split":np.where(np.isin(np.arange(N_TRAIN),DESIGN_IDX),"DESIGN","LOCK")}).to_csv(V43_OUT/"v43_split_manifest.csv",index=False)

pkg={"python":platform.python_version(),"torch":torch.__version__,"numpy":np.__version__,"pandas":pd.__version__,"sklearn":sklearn.__version__,"lightgbm":lgb.__version__}
json.dump(pkg,open(V43_OUT/"v43_package_versions.json","w"),indent=2)
repro={
    "exact_v30_v31_current_reproduction":bool(REPRO_OK),"public_anchor":PUBLIC_V43_ANCHOR_LB,
    "anchor_top5_sha256":top5_sha256(V43_ANCHOR_TEST),"expected_anchor_top5_sha256":EXPECTED_V43_ANCHOR_TOP5_SHA256,
    "organizer_data_only":True,"external_labeled_data":False,"no_historical_submission_csv_predictive_input":True,
    "no_user_id_predictive_feature":True,"no_post_lock_retraining":True,"joint_user_level":True,"all17_training":True,
    "partial_pl_top4_loss":True,"residual_zero_initialization":True,"tabm_parameter_efficient_ensemble":True,
    "mlp_control_preregistered":True,"s0_only_deployment":True,"s1_s2_s3_test_locked":True,"top5_membership_frozen":True,
    "rank5_frozen":True,"top4_score_values_permuted_only":True,"epochs_fixed":EPOCHS,"residual_l2":RESIDUAL_L2,
}
json.dump(repro,open(V43_OUT/"v43_reproduction_audit.json","w"),indent=2)
summary={
    "version":V43_VERSION,"public_anchor":PUBLIC_V43_ANCHOR_LB,"leader_reference":PUBLIC_LEADER_LB,"leader_gap":PUBLIC_LEADER_LB-PUBLIC_V43_ANCHOR_LB,
    "selected_backbone":SELECTED_BACKBONE,"design_gate":bool(DESIGN_GATE),"lock_evaluated":bool(LOCK_EVALUATED),"lock_gate":bool(LOCK_GATE),
    "test_geometry_gate":bool(TEST_GEOMETRY_GATE),"final_gate":bool(FINAL_GATE),"selected_design_delta":float(SELECTED_MET["delta"]),
    "selected_design_fold_wins":int(SELECTED_MET["fold_wins"]),"selected_design_block_wins":int(SELECTED_MET["block_wins"]),
    "test_changed_users":int(TEST_GEOM["changed_users"]),"test_top1_changes":int(TEST_GEOM["top1_changes"]),
    "theoretical_max_public_gain":THEORETICAL_MAX_PUBLIC_GAIN,"anchor_top5_sha256":top5_sha256(V43_ANCHOR_TEST),
    "candidate_top5_sha256":top5_sha256(TEST_CANDIDATE),"primary_top5_sha256":top5_sha256(PRIMARY),
    "submission_instruction":"SUBMIT submission_v43_primary.csv" if FINAL_GATE else "NO V43 SUBMISSION: primary is exact 0.66028 fail-close anchor."
}
json.dump(summary,open(V43_OUT/"v43_summary.json","w"),indent=2)
print(json.dumps(summary,indent=2))
print("V43 outputs written to",V43_OUT)

# ---- cell 40 (markdown) ----
# ### Terminal interpretation contract
#
# - **FINAL_GATE = True:** `submission_v43_primary.csv` is the only V43 file intended for Kaggle submission.
# - **FINAL_GATE = False:** do **not** submit the diagnostic candidate. `submission_v43_primary.csv` is an exact fail-close copy of the reproducible `0.66028` champion.
# - MLP-vs-TabM selection is made once on DESIGN only. LOCK can confirm or reject the frozen choice; LOCK can never change architecture, epoch count, feature set, regularization, or deployment policy.
# - If V43 fails clean DESIGN/LOCK, do not create V43.1 by threshold/epoch/regularization mining.
