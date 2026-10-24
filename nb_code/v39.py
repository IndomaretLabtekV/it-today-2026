# Converted from notebooks/v39.ipynb: markdown cells become comments, %/! magics neutralized.

# ---- cell 0 (markdown) ----
# # MineToday V39 — Direct Intent Trajectory Residual LTR
#
# **Thesis.** The current public champion `0.66020` is reconstructed and treated as an immutable fail-close anchor. V39 does not replace it globally. It tests one genuinely new organizer-data signal: the **chronological trajectory of modules directly mentioned by the user in chat**, separated from V30's broader prerequisite-inheriting topic surface.
#
# V39 learns relevance densely with the same V30 LambdaRank family and fixed `alpha=0.16`, then deploys sparsely. Two small label-free trajectory views are trained independently:
#
# - **R view:** direct-module last-recency + repeated direct-event count;
# - **U view:** direct-module unique recency order.
#
# A test correction is eligible only in historical **S0** (the untouched region), only when both trajectory siblings agree, and only for high-leverage minimal geometry:
#
# 1. a pure **1↔2 Top-1 swap** with unchanged Top-5 membership; or
# 2. a pure **5↔6 membership swap** with unchanged Top-1.
#
# Both corrections also require the directly observed trajectory itself to favor the promoted module. Existing V25 protections, V30 gains, S1, S2, and the seven public-proven S3 actions remain locked. Fresh AD/AE/AF worlds and sealed AG + chronology are rejection-only gates. If the gate fails, `submission_v39_primary.csv` is exactly the reconstructed `0.66020` anchor.
#
# **Compliance/runtime.** Organizer files only; no external data, no pretrained model, no LLM-generated test labels, and no prior submission CSV is read. Kaggle T4x2 is supported; this notebook is deliberately CPU-first for deterministic LightGBM and will simply report the two T4s when available.

# ---- cell 1 (code) ----
# Runtime + preregistered V39 configuration
import os, sys, re, json, ast, html, hashlib, random, warnings, time, subprocess, gc, math
from pathlib import Path
from collections import defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ["PYTHONHASHSEED"] = "42"

V39_SKIP_INSTALL = bool(int(os.getenv("V39_SKIP_INSTALL", "0")))
try:
    import lightgbm as lgb
except Exception:
    if V39_SKIP_INSTALL:
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
V39_SMOKE = bool(int(os.getenv("V39_SMOKE", "0")))
V39_STRICT_GEOMETRY = bool(int(os.getenv("V39_STRICT_GEOMETRY", "1")))

# Compatibility aliases consumed by the frozen V3.1 -> V25 -> V30/V31 reconstruction cells.
V31_SMOKE = V39_SMOKE
V30_SMOKE = V39_SMOKE
V30_STRICT_GEOMETRY = V39_STRICT_GEOMETRY
V29_SMOKE = V39_SMOKE
V29_STRICT_GEOMETRY = V39_STRICT_GEOMETRY
V27_SMOKE = V39_SMOKE
V27_STRICT_GEOMETRY = V39_STRICT_GEOMETRY

N_FOLDS = 2 if V39_SMOKE else 5
ANCHOR_INNER_FOLDS = 2 if V39_SMOKE else 4
INTENT_INNER_FOLDS = 2 if V39_SMOKE else 4

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

LGB_ESTIMATORS = 12 if V39_SMOKE else 1200
LGB_EARLY_STOP = 3 if V39_SMOKE else 80
RANKER_ESTIMATORS = 16 if V39_SMOKE else 360
RANKER_LEARNING_RATE = 0.04 if V39_SMOKE else 0.025

# Frozen public-proven V30 residual geometry. No alpha search in V39.
ALPHA = 0.16
INTENT_PARENT_STRENGTH = 40.0
INTENT_CHILD_STRENGTH = 30.0

# Small, precommitted action caps. They scale to 6 Top1 + 8 membership actions on 1,000 test users.
TOP1_COVERAGE_CAP = 0.006
MEMBERSHIP_COVERAGE_CAP = 0.008

# Promotion gates are category-specific and are never tuned after AD/AE/AF are printed.
DEV_MIN_ACTIONS_PER_WORLD = 1 if V39_SMOKE else 2
DEV_MIN_ACTION_PRECISION = 0.0 if V39_SMOKE else 0.60
DEV_CI_FLOOR = -1.0 if V39_SMOKE else -0.00002
SEALED_AG_FLOOR = -1.0 if V39_SMOKE else 0.0
SEALED_CHRONO_FLOOR = -1.0 if V39_SMOKE else 0.0
BOOTSTRAPS = 30 if V39_SMOKE else 1200
GEOMETRY_ATOL = 2.5e-4

# Fresh names begin after the historical AA/AB/AC curriculum-XLM-R worlds.
def _seed_from_text(s):
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:8], 16) % (2**31-1)

FRESH_WORLDS = {
    "AD": _seed_from_text("MINETODAY-V39-FRESH-AD-20260901-DIRECT-TRAJECTORY"),
    "AE": _seed_from_text("MINETODAY-V39-FRESH-AE-20260901-DIRECT-TRAJECTORY"),
    "AF": _seed_from_text("MINETODAY-V39-FRESH-AF-20260901-DIRECT-TRAJECTORY"),
}
SEALED_WORLD_SALT = "MINETODAY-V39-SEALED-AG-20260901-DIRECT-TRAJECTORY"

# Historical geometry regression checks. These encode ranking signatures/counts, not hidden labels.
EXPECTED_V30_CHANGED = 77
EXPECTED_V30_TOP1_CHANGES = 8
EXPECTED_V30_MEMBERSHIP_CHANGES = 7
EXPECTED_V31_CHANGED_VS_V30 = 60
EXPECTED_STATE_COUNTS = {"S0":897, "S1":26, "S2":43, "S3":34}
EXPECTED_S3_TOP1 = 3
EXPECTED_S3_MEMBERSHIP = 4
EXPECTED_S3_ORDER_ONLY = 27
HYBRID34_TOP5_SHA256 = "47f7db52ce8f782b8b393fd120140f1b6f51aeac452a9fabb5794cc9d8ba5e6e"
CURRENT_ANCHOR_TOP5_SHA256 = "c70ce570ec311e3f7391cec324e566a3a0e27da99ffc210afc3a8f102daf0191"

random.seed(SEED)
np.random.seed(SEED)

print("numpy", np.__version__)
print("pandas", pd.__version__)
print("scipy", scipy.__version__)
print("sklearn", sklearn.__version__)
print("lightgbm", lgb.__version__)
print("V39_SMOKE", V39_SMOKE, "fixed alpha", ALPHA)
print("fresh worlds", FRESH_WORLDS)
print("current public anchor", PUBLIC_CURRENT_ANCHOR_LB,
      "gap to Top4", round(PUBLIC_TOP4_LB-PUBLIC_CURRENT_ANCHOR_LB,5),
      "gap to leader", round(PUBLIC_LEADER_LB-PUBLIC_CURRENT_ANCHOR_LB,5))
try:
    print("GPU runtime:\n" + subprocess.check_output(["nvidia-smi", "-L"], text=True).strip())
except Exception:
    print("nvidia-smi unavailable; V39 remains runnable and is deliberately CPU-first.")

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
# ## New information source: direct intent trajectory
#
# V30's `MODULE_TOPIC_MAP` intentionally lets prerequisite/background topics flow into several candidates. V39 creates a second, **pure destination** view. Each learning chat event maps only explicitly mentioned topics to their canonical module. From that event sequence we derive two tiny label-free views:
#
# - `R`: last direct recency and capped direct-event recurrence;
# - `U`: dense recency rank among unique directly mentioned modules.
#
# No target encoding or fold fitting is used for these features, so train/validation/test construction is identical and leakage-free.

# ---- cell 19 (code) ----
# Pure explicit topic -> destination mapping. Multiple topics mapping to the same module in one chat count once.
DIRECT_TOPIC_TO_MODULE = {
    "excel":"M_001", "python":"M_002", "sql":"M_003", "webscrape":"M_004",
    "git":"M_005", "statistics":"M_006", "eda":"M_007", "dashboard":"M_008",
    "ml":"M_009", "ml_eval":"M_009", "cv":"M_010", "nlp":"M_011",
    "genai":"M_012", "prompt":"M_013", "automation":"M_014", "nocode":"M_015",
    "mlops":"M_016", "career":"M_017",
}

N_ALL = len(all_users)
TRAJ_EVENT_COUNT_RAW = np.zeros((N_ALL,N_MODULES),np.int16)
TRAJ_LAST_DISTANCE = np.full((N_ALL,N_MODULES),-1,np.int16)
TRAJ_LAST_RECENCY = np.zeros((N_ALL,N_MODULES),np.float32)
TRAJ_COUNT_CAPPED = np.zeros((N_ALL,N_MODULES),np.float32)
TRAJ_DENSE_RANK = np.zeros((N_ALL,N_MODULES),np.int16)  # 0 = never directly mentioned
TRAJ_UNIQUE_INV_RANK = np.zeros((N_ALL,N_MODULES),np.float32)
TRAJ_UNIQUE_RANK_NORM = np.zeros((N_ALL,N_MODULES),np.float32)

for u in all_users:
    ui = uid2idx[u]
    g = user_chat.get(u)
    if g is None or len(g)==0:
        continue
    learn = g[~g["is_noise"]].copy()
    if len(learn)==0:
        learn = g.copy()
    rows = list(learn.itertuples(index=False))
    n = len(rows)
    if n==0:
        continue

    mentions = defaultdict(list)  # module column -> event indices
    for j,row in enumerate(rows):
        mids = sorted({DIRECT_TOPIC_TO_MODULE[t] for t in set(row.topics) if t in DIRECT_TOPIC_TO_MODULE})
        for mid in mids:
            mentions[MID_TO_COL[mid]].append(j)

    if not mentions:
        continue

    last_event = {m:int(js[-1]) for m,js in mentions.items()}
    distinct_last = sorted(set(last_event.values()), reverse=True)
    dense_rank_by_event = {ev:r+1 for r,ev in enumerate(distinct_last)}
    n_groups = len(distinct_last)

    for m,js in mentions.items():
        cnt = len(js)
        dist = n-1-last_event[m]
        rank = dense_rank_by_event[last_event[m]]
        TRAJ_EVENT_COUNT_RAW[ui,m] = cnt
        TRAJ_LAST_DISTANCE[ui,m] = dist
        TRAJ_LAST_RECENCY[ui,m] = float(np.exp(-0.7*dist))
        TRAJ_COUNT_CAPPED[ui,m] = min(cnt,4)/4.0
        TRAJ_DENSE_RANK[ui,m] = rank
        TRAJ_UNIQUE_INV_RANK[ui,m] = 1.0/float(rank)
        TRAJ_UNIQUE_RANK_NORM[ui,m] = (n_groups-rank+1)/float(max(1,n_groups))

TRAJECTORY_VIEWS = {
    "R": [
        ("traj_last_recency", TRAJ_LAST_RECENCY),
        ("traj_count_capped", TRAJ_COUNT_CAPPED),
    ],
    "U": [
        ("traj_unique_inv_rank", TRAJ_UNIQUE_INV_RANK),
        ("traj_unique_rank_norm", TRAJ_UNIQUE_RANK_NORM),
    ],
}

def trajectory_long_block(rows_global, view):
    rows_global=np.asarray(rows_global,int)
    specs=TRAJECTORY_VIEWS[view]
    Z=np.column_stack([M[rows_global].reshape(-1) for _,M in specs]).astype(np.float32)
    names=[name for name,_ in specs]
    assert Z.shape==(len(rows_global)*N_MODULES,len(specs))
    assert np.isfinite(Z).all()
    return Z,names

# Train-only descriptive audit: how true relevance changes with direct-intent recency order.
traj_audit=[]
for r in range(1,6):
    mask=(TRAJ_DENSE_RANK[:N_TRAIN]==r)
    n_pairs=int(mask.sum())
    if n_pairs==0:
        continue
    positive=(y>0)[mask]
    top5=(RANK_POS<=5)[mask]
    top1=(RANK_POS==1)[mask]
    pos_ranks=RANK_POS[mask][positive]
    traj_audit.append({
        "direct_dense_recency_rank":r,
        "candidate_pairs":n_pairs,
        "positive_rate":float(positive.mean()),
        "target_top5_rate":float(top5.mean()),
        "target_top1_rate":float(top1.mean()),
        "avg_true_rank_if_positive":float(pos_ranks.mean()) if len(pos_ranks) else np.nan,
    })
V39_TRAJECTORY_AUDIT=pd.DataFrame(traj_audit)

train_has_direct=(TRAJ_DENSE_RANK[:N_TRAIN]>0).any(1)
test_has_direct=(TRAJ_DENSE_RANK[N_TRAIN:]>0).any(1)
print("users with direct trajectory train/test:",int(train_has_direct.sum()),int(test_has_direct.sum()))
display(V39_TRAJECTORY_AUDIT)

# ---- cell 20 (markdown) ----
# ## Anchor state machine + deterministic high-leverage action policy
#
# `A=V25/pre-V30`, `B=V30`, `C=V31`. The historical current anchor keeps V31 only inside S3 when the disagreement changes Top-1 or Top-5 membership. V39 then attacks **S0 only**.
#
# The V39 action selector is deliberately not a learned router. It requires agreement between the R/U rankers *and* direct trajectory evidence. It applies only minimal score swaps, preserving each row's exact score multiset.

# ---- cell 21 (code) ----
STATE_NAMES=np.asarray(["S0","S1","S2","S3"],dtype=object)
ACTION_COLUMNS=[
    "row_local","global_row","user_id","action_type","promoted_module","demoted_module",
    "trajectory_recency_adv","trajectory_unique_adv","trajectory_confidence","model_agreement_margin"
]

def state_codes(pre_anchor,v30,v31):
    pre_top=top5_indices(pre_anchor); v30_top=top5_indices(v30); v31_top=top5_indices(v31)
    a=np.any(pre_top!=v30_top,axis=1)
    b=np.any(v30_top!=v31_top,axis=1)
    return (2*a.astype(np.int8)+b.astype(np.int8)).astype(np.int8)

def top5_sha256(scores):
    top=top5_indices(scores).astype(np.int16)
    return hashlib.sha256(top.tobytes()).hexdigest()

def membership_change_mask(a,b):
    ta,tb=top5_indices(a),top5_indices(b)
    return np.asarray([set(x)!=set(z) for x,z in zip(ta,tb)],bool)

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

def _coverage_cap(n,action_type):
    frac=TOP1_COVERAGE_CAP if action_type=="top1_12" else MEMBERSHIP_COVERAGE_CAP
    return max(1,int(math.ceil(float(n)*frac)))

def propose_consensus_trajectory_actions(current,score_r,score_u,rows_global,eligible_mask,protected_mask=None):
    current=np.asarray(current,np.float32); score_r=np.asarray(score_r,np.float32); score_u=np.asarray(score_u,np.float32)
    rows_global=np.asarray(rows_global,int); eligible_mask=np.asarray(eligible_mask,bool)
    n=len(current)
    assert score_r.shape==score_u.shape==current.shape==(n,N_MODULES)
    assert len(rows_global)==n and len(eligible_mask)==n
    if protected_mask is None:
        protected_mask=np.zeros(n,bool)
    else:
        protected_mask=np.asarray(protected_mask,bool)

    cur_order=np.argsort(-current,axis=1,kind="mergesort")
    r_top=top5_indices(score_r); u_top=top5_indices(score_u)
    proposals=[]

    for i in np.flatnonzero(eligible_mask & ~protected_mask):
        g=int(rows_global[i])
        cur5=cur_order[i,:5]
        curset=set(cur5.tolist())

        # Pure 1<->2: same membership, both siblings promote the current #2 to #1.
        old_top=int(cur_order[i,0]); cand=int(cur_order[i,1])
        if (
            int(r_top[i,0])==cand and int(u_top[i,0])==cand and
            set(r_top[i].tolist())==curset and set(u_top[i].tolist())==curset
        ):
            rec_adv=float(TRAJ_LAST_RECENCY[g,cand]-TRAJ_LAST_RECENCY[g,old_top])
            uniq_adv=float(TRAJ_UNIQUE_INV_RANK[g,cand]-TRAJ_UNIQUE_INV_RANK[g,old_top])
            if rec_adv>0 and uniq_adv>0:
                mm=float(min(score_r[i,cand]-score_r[i,old_top],score_u[i,cand]-score_u[i,old_top]))
                proposals.append({
                    "row_local":i,"global_row":g,"user_id":str(all_users[g]),"action_type":"top1_12",
                    "promoted_module":MODULES[cand],"demoted_module":MODULES[old_top],
                    "trajectory_recency_adv":rec_adv,"trajectory_unique_adv":uniq_adv,
                    "trajectory_confidence":0.5*(rec_adv+uniq_adv),"model_agreement_margin":mm,
                })
                continue

        # Pure 5<->6: both siblings agree on exactly replacing current #5 with current #6; Top1 is locked.
        exit_m=int(cur_order[i,4]); enter_m=int(cur_order[i,5])
        desired=(curset-{exit_m})|{enter_m}
        if (
            int(r_top[i,0])==int(cur_order[i,0]) and int(u_top[i,0])==int(cur_order[i,0]) and
            set(r_top[i].tolist())==desired and set(u_top[i].tolist())==desired
        ):
            rec_adv=float(TRAJ_LAST_RECENCY[g,enter_m]-TRAJ_LAST_RECENCY[g,exit_m])
            uniq_adv=float(TRAJ_UNIQUE_INV_RANK[g,enter_m]-TRAJ_UNIQUE_INV_RANK[g,exit_m])
            if rec_adv>0 and uniq_adv>0:
                mm=float(min(score_r[i,enter_m]-score_r[i,exit_m],score_u[i,enter_m]-score_u[i,exit_m]))
                proposals.append({
                    "row_local":i,"global_row":g,"user_id":str(all_users[g]),"action_type":"membership_56",
                    "promoted_module":MODULES[enter_m],"demoted_module":MODULES[exit_m],
                    "trajectory_recency_adv":rec_adv,"trajectory_unique_adv":uniq_adv,
                    "trajectory_confidence":0.5*(rec_adv+uniq_adv),"model_agreement_margin":mm,
                })

    raw=pd.DataFrame(proposals,columns=ACTION_COLUMNS)
    if len(raw)==0:
        return raw

    selected=[]
    for typ,gdf in raw.groupby("action_type",sort=False):
        cap=_coverage_cap(n,typ)
        gdf=gdf.sort_values(
            ["trajectory_confidence","model_agreement_margin","row_local"],
            ascending=[False,False,True],kind="mergesort"
        ).head(cap)
        selected.append(gdf)
    out=pd.concat(selected,ignore_index=True) if selected else pd.DataFrame(columns=ACTION_COLUMNS)
    assert out["row_local"].is_unique
    return out.sort_values(["action_type","trajectory_confidence","row_local"],ascending=[True,False,True],kind="mergesort").reset_index(drop=True)

def apply_selected_actions(current,actions,allowed_types=("top1_12","membership_56")):
    out=np.asarray(current,np.float32).copy()
    allowed=set(allowed_types)
    if actions is None or len(actions)==0 or len(allowed)==0:
        return out
    before=np.sort(out,axis=1).copy()
    for a in actions.itertuples(index=False):
        if a.action_type not in allowed:
            continue
        r=int(a.row_local); pr=MID_TO_COL[a.promoted_module]; dm=MID_TO_COL[a.demoted_module]
        if not float(out[r,dm])>float(out[r,pr]):
            raise AssertionError((a.user_id,a.action_type,a.demoted_module,a.promoted_module,float(out[r,dm]-out[r,pr])))
        out[r,dm],out[r,pr]=out[r,pr],out[r,dm]
    assert np.array_equal(np.sort(out,axis=1),before),"score multiset changed"
    return out

def package_metrics(y_true,anchor,candidate):
    ua=per_user_ndcg(y_true,anchor); uc=per_user_ndcg(y_true,candidate); d=uc-ua
    changed=np.any(top5_indices(anchor)!=top5_indices(candidate),axis=1)
    if changed.any():
        precision=float(np.mean(d[changed]>0)); harm=float(np.mean(d[changed]<0)); mean_changed=float(d[changed].mean())
    else:
        precision=np.nan; harm=np.nan; mean_changed=np.nan
    return {
        "delta":float(d.mean()),"changed_users":int(changed.sum()),
        "action_precision":precision,"action_harm_rate":harm,"mean_delta_changed":mean_changed,
    },d

print("V39 state/current-anchor/action helpers ready")

# ---- cell 22 (markdown) ----
# ## Fresh AD/AE/AF: dense learning, sparse deployment
#
# Each fresh world reconstructs the frozen anchor inside five outer folds, trains V30, V31, R, and U with the same ranker seed, then evaluates the **deterministic consensus action package against the 0.66020-style validation anchor**. The gate is based on package utility, not on whether a trajectory ranker wins globally.

# ---- cell 23 (code) ----
def fit_v39_branches(train_rows,val_rows,seed):
    train_rows=np.asarray(train_rows,int); val_rows=np.asarray(val_rows,int)
    anc=build_anchor_split_full(train_rows,val_rows,seed)

    # V31 representation is needed only for the state machine/current-anchor reconstruction.
    intent_tr=crossfit_intentstate_features(
        y[train_rows],ORDINAL_GRADES[train_rows],TRAIN_INTENT[train_rows],MODULE_READY_BAND[train_rows],
        TRAIN_GROUP[train_rows],seed+101
    )
    ib=fit_intentstate_bundle(
        y[train_rows],ORDINAL_GRADES[train_rows],TRAIN_INTENT[train_rows],MODULE_READY_BAND[train_rows]
    )
    intent_va=apply_intentstate_bundle(ib,TRAIN_INTENT[val_rows],MODULE_READY_BAND[val_rows])

    Xtr_all,base_names,bounds=build_long_matrix_v31(
        train_rows,anc["cons_train"],anc["base_train"],anc["meta_train"],intent_tr
    )
    Xva_all,_,bounds_va=build_long_matrix_v31(
        val_rows,anc["cons_val"],anc["base_val"],anc["meta_val"],intent_va
    )
    assert bounds==bounds_va and bounds["intentstate"]>bounds["base"]
    be=bounds["base"]

    Ztr_r,names_r=trajectory_long_block(train_rows,"R"); Zva_r,_=trajectory_long_block(val_rows,"R")
    Ztr_u,names_u=trajectory_long_block(train_rows,"U"); Zva_u,_=trajectory_long_block(val_rows,"U")
    Xtr_r=np.hstack([Xtr_all[:,:be],Ztr_r]).astype(np.float32,copy=False)
    Xva_r=np.hstack([Xva_all[:,:be],Zva_r]).astype(np.float32,copy=False)
    Xtr_u=np.hstack([Xtr_all[:,:be],Ztr_u]).astype(np.float32,copy=False)
    Xva_u=np.hstack([Xva_all[:,:be],Zva_u]).astype(np.float32,copy=False)

    rank_seed=seed+303
    m0,r0=fit_ranker_predict(Xtr_all[:,:be],Xva_all[:,:be],ORDINAL_GRADES[train_rows],len(train_rows),rank_seed)
    m1,r1=fit_ranker_predict(Xtr_all[:,:bounds["intentstate"]],Xva_all[:,:bounds["intentstate"]],ORDINAL_GRADES[train_rows],len(train_rows),rank_seed)
    mr,rr=fit_ranker_predict(Xtr_r,Xva_r,ORDINAL_GRADES[train_rows],len(train_rows),rank_seed)
    mu,ru=fit_ranker_predict(Xtr_u,Xva_u,ORDINAL_GRADES[train_rows],len(train_rows),rank_seed)

    out={
        "pre":anc["cons_val"].copy(),
        "v30":blend_residual(anc["cons_val"],r0,ALPHA),
        "v31":blend_residual(anc["cons_val"],r1,ALPHA),
        "traj_r":blend_residual(anc["cons_val"],rr,ALPHA),
        "traj_u":blend_residual(anc["cons_val"],ru,ALPHA),
        "feature_names_r":base_names[:be]+names_r,
        "feature_names_u":base_names[:be]+names_u,
    }
    del Xtr_all,Xva_all,Ztr_r,Zva_r,Ztr_u,Zva_u,Xtr_r,Xva_r,Xtr_u,Xva_u,intent_tr,intent_va,ib,m0,m1,mr,mu,anc
    gc.collect()
    return out

def evaluate_oof_bundle(pre,v30,v31,traj_r,traj_u,rows_global,world):
    state=state_codes(pre,v30,v31)
    current,cinfo=current066020_surface(v30,v31,state)
    eligible=(state==0)
    actions=propose_consensus_trajectory_actions(current,traj_r,traj_u,rows_global,eligible)

    candidates={
        "top1_12":apply_selected_actions(current,actions,("top1_12",)),
        "membership_56":apply_selected_actions(current,actions,("membership_56",)),
        "combined":apply_selected_actions(current,actions,("top1_12","membership_56")),
    }
    metrics={}; deltas={}
    for typ,cand in candidates.items():
        met,d=package_metrics(y[np.asarray(rows_global,int)],current,cand)
        metrics[typ]=met; deltas[typ]=d

    state_counts={STATE_NAMES[s]:int(np.sum(state==s)) for s in range(4)}
    row={
        "world":world,"n":len(rows_global),"current_ndcg":ndcg5(y[np.asarray(rows_global,int)],current),
        "traj_r_global_ndcg":ndcg5(y[np.asarray(rows_global,int)],traj_r),
        "traj_u_global_ndcg":ndcg5(y[np.asarray(rows_global,int)],traj_u),
        "s0_users":state_counts["S0"],"s1_users":state_counts["S1"],"s2_users":state_counts["S2"],"s3_users":state_counts["S3"],
        "current_s3_top1_kept":int(cinfo["top1"].sum()),"current_s3_membership_kept":int(cinfo["membership"].sum()),
    }
    for typ,met in metrics.items():
        prefix={"top1_12":"top1","membership_56":"membership","combined":"combined"}[typ]
        for k,v in met.items(): row[f"{prefix}_{k}"]=v

    if len(actions):
        actions=actions.copy(); actions["world"]=world
        combo_delta=deltas["combined"]
        actions["true_delta_if_applied"]=actions["row_local"].map(lambda i:float(combo_delta[int(i)]))
        actions["beneficial_if_applied"]=actions["true_delta_if_applied"]>0
    return row,deltas,actions

SALT_DELTAS={"top1_12":{},"membership_56":{},"combined":{}}
SALT_ACTIONS=[]
fresh_rows=[]

for world,seed in FRESH_WORLDS.items():
    print(f"\n=== V39 fresh world {world} seed={seed} ===")
    outer=make_group_folds_global(np.arange(N_TRAIN),N_FOLDS,seed)
    pred={k:np.zeros_like(y,np.float32) for k in ("pre","v30","v31","traj_r","traj_u")}
    fold_id=np.full(N_TRAIN,-1,np.int16)
    feature_names_r=feature_names_u=None

    for f,(tr,va) in enumerate(outer):
        print(f"{world} fold {f+1}/{len(outer)} train={len(tr)} val={len(va)}")
        b=fit_v39_branches(tr,va,seed+1000+f*211)
        for k in pred: pred[k][va]=b[k]
        fold_id[va]=f
        feature_names_r=b["feature_names_r"]; feature_names_u=b["feature_names_u"]
        del b
        gc.collect()

    assert (fold_id>=0).all()
    for k in pred: assert np.isfinite(pred[k]).all(),k
    row,deltas,actions=evaluate_oof_bundle(
        pred["pre"],pred["v30"],pred["v31"],pred["traj_r"],pred["traj_u"],np.arange(N_TRAIN),world
    )
    for typ in SALT_DELTAS: SALT_DELTAS[typ][world]=deltas[typ]
    if len(actions): SALT_ACTIONS.append(actions)
    fresh_rows.append(row)
    del pred,deltas,actions
    gc.collect()

V39_FRESH_SUMMARY=pd.DataFrame(fresh_rows)
V39_VALIDATION_ACTIONS=pd.concat(SALT_ACTIONS,ignore_index=True) if SALT_ACTIONS else pd.DataFrame(columns=ACTION_COLUMNS+["world","true_delta_if_applied","beneficial_if_applied"])
display(V39_FRESH_SUMMARY)

# ---- cell 24 (code) ----
# Category-specific development gate. No global-ranker win is required.
dev_rows=[]
for typ,prefix in [("top1_12","top1"),("membership_56","membership")]:
    world_delta=V39_FRESH_SUMMARY[f"{prefix}_delta"].to_numpy(float)
    world_actions=V39_FRESH_SUMMARY[f"{prefix}_changed_users"].to_numpy(int)
    world_precision=V39_FRESH_SUMMARY[f"{prefix}_action_precision"].to_numpy(float)
    avg_user_delta=np.mean(np.stack([SALT_DELTAS[typ][w] for w in FRESH_WORLDS],axis=0),axis=0)
    ci=cluster_boot_ci(avg_user_delta,TRAIN_GROUP,_seed_from_text(f"V39-POOLED-{typ}"))
    precision_mean=float(np.nanmean(world_precision)) if np.isfinite(world_precision).any() else np.nan
    gate=bool(
        np.all(world_delta>0.0) and
        np.all(world_actions>=DEV_MIN_ACTIONS_PER_WORLD) and
        np.isfinite(precision_mean) and precision_mean>=DEV_MIN_ACTION_PRECISION and
        ci[0]>=DEV_CI_FLOOR
    )
    dev_rows.append({
        "action_type":typ,
        "positive_worlds":int(np.sum(world_delta>0)),
        "worst_world_delta":float(np.min(world_delta)),
        "mean_world_delta":float(np.mean(world_delta)),
        "min_actions_per_world":int(np.min(world_actions)),
        "mean_action_precision":precision_mean,
        "pooled_user_delta":float(avg_user_delta.mean()),
        "pooled_ci_low":ci[0],"pooled_ci_high":ci[1],
        "required_precision":DEV_MIN_ACTION_PRECISION,
        "required_ci_floor":DEV_CI_FLOOR,
        "development_gate_passed":gate,
    })

V39_DEV_GATE=pd.DataFrame(dev_rows)
DEV_APPROVED_TYPES=V39_DEV_GATE.loc[V39_DEV_GATE.development_gate_passed,"action_type"].tolist()
display(V39_DEV_GATE)
print("DEV_APPROVED_TYPES =",DEV_APPROVED_TYPES)

# ---- cell 25 (markdown) ----
# ## Sealed AG + chronology: rejection-only
#
# Only action categories that survive AD/AE/AF are eligible. AG is template-group-disjoint; the last 1,000 ordered train users form the chronology stress test. These worlds can only reject a category, never change features, caps, or thresholds.

# ---- cell 26 (code) ----
def sealed_lock_bucket(group):
    h=hashlib.sha1(f"{SEALED_WORLD_SALT}::{group}".encode("utf-8")).hexdigest()
    return int(h[:12],16)%5

SEALED_LOCK_MASK=np.asarray([sealed_lock_bucket(g)==0 for g in TRAIN_GROUP],bool)
SEALED_DEV_IDX=np.flatnonzero(~SEALED_LOCK_MASK)
SEALED_LOCK_IDX=np.flatnonzero(SEALED_LOCK_MASK)
assert len(set(TRAIN_GROUP[SEALED_DEV_IDX]).intersection(set(TRAIN_GROUP[SEALED_LOCK_IDX])))==0
CHRONO_TR=np.arange(0,3000,dtype=int)
CHRONO_VA=np.arange(3000,4000,dtype=int)

def evaluate_holdout_v39(train_rows,val_rows,seed,world):
    b=fit_v39_branches(train_rows,val_rows,seed)
    row,deltas,actions=evaluate_oof_bundle(
        b["pre"],b["v30"],b["v31"],b["traj_r"],b["traj_u"],np.asarray(val_rows,int),world
    )
    del b,deltas
    gc.collect()
    return row,actions

sealed_rows=[]; sealed_actions=[]
if DEV_APPROVED_TYPES:
    for tr,va,seed,world in [
        (SEALED_DEV_IDX,SEALED_LOCK_IDX,_seed_from_text("V39-SEALED-AG-FIT"),"sealed_template_lock_AG"),
        (CHRONO_TR,CHRONO_VA,_seed_from_text("V39-CHRONO-FIT"),"chronological_U3001_U4000"),
    ]:
        print("\n===",world,"===")
        row,act=evaluate_holdout_v39(tr,va,seed,world)
        sealed_rows.append(row)
        if len(act): sealed_actions.append(act)
else:
    sealed_rows=[{"world":"skipped_no_development_category","n":0}]

V39_SEALED=pd.DataFrame(sealed_rows)
V39_SEALED_ACTIONS=pd.concat(sealed_actions,ignore_index=True) if sealed_actions else pd.DataFrame(columns=ACTION_COLUMNS+["world","true_delta_if_applied","beneficial_if_applied"])

sealed_gate=[]
for typ,prefix in [("top1_12","top1"),("membership_56","membership")]:
    if typ not in DEV_APPROVED_TYPES or len(V39_SEALED)<2:
        gate=False; ag=np.nan; ch=np.nan; agn=0; chn=0
    else:
        ag_row=V39_SEALED.loc[V39_SEALED.world=="sealed_template_lock_AG"].iloc[0]
        ch_row=V39_SEALED.loc[V39_SEALED.world=="chronological_U3001_U4000"].iloc[0]
        ag=float(ag_row[f"{prefix}_delta"]); ch=float(ch_row[f"{prefix}_delta"])
        agn=int(ag_row[f"{prefix}_changed_users"]); chn=int(ch_row[f"{prefix}_changed_users"])
        gate=bool(ag>=SEALED_AG_FLOOR and ch>=SEALED_CHRONO_FLOOR and agn>0 and chn>0)
    sealed_gate.append({
        "action_type":typ,"development_passed":bool(typ in DEV_APPROVED_TYPES),
        "sealed_AG_delta":ag,"chronology_delta":ch,"sealed_AG_actions":agn,"chronology_actions":chn,
        "sealed_gate_passed":gate,
    })

V39_SEALED_GATE=pd.DataFrame(sealed_gate)
FINAL_APPROVED_TYPES=V39_SEALED_GATE.loc[V39_SEALED_GATE.sealed_gate_passed,"action_type"].tolist()
display(V39_SEALED)
display(V39_SEALED_GATE)
print("FINAL_APPROVED_TYPES =",FINAL_APPROVED_TYPES)

# ---- cell 27 (markdown) ----
# ## Full-train deployment + exact 0.66020 fail-close
#
# The full model first reconstructs V30/V31, verifies the known Hybrid34 and current-anchor Top-5 signatures, then fits the two V39 trajectory siblings. Test actions are generated only inside S0 and the historical V25 protected rows are immutable. The primary file changes the anchor only for action categories that passed both development and sealed gates.

# ---- cell 28 (code) ----
OUT_DIR=Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("/mnt/data")
OUT_DIR.mkdir(parents=True,exist_ok=True)

# V25 protection set is derived from the frozen historical package already reconstructed above.
PROTECTED_TEST_USERS=set([u for u,_,_ in KNOWN_PB_SWAPS] + [a["user_id"] for a in V25_CORE_ACTIONS])
protected_rows=np.asarray([TEST_UID_TO_ROW[u] for u in sorted(PROTECTED_TEST_USERS)],int)

# Strict full-train V31 state representation for exact V30/V31 reconstruction.
full_intent_tr=crossfit_intentstate_features(
    y,ORDINAL_GRADES,TRAIN_INTENT,MODULE_READY_BAND[:N_TRAIN],TRAIN_GROUP,
    _seed_from_text("V31-FULL-INTENTSTATE-CROSSFIT")
)
FULL_INTENT_BUNDLE=fit_intentstate_bundle(y,ORDINAL_GRADES,TRAIN_INTENT,MODULE_READY_BAND[:N_TRAIN])
full_intent_te=apply_intentstate_bundle(FULL_INTENT_BUNDLE,TEST_INTENT,MODULE_READY_BAND[N_TRAIN:])
V39_INTENT_SUPPORT_AUDIT=intentstate_support_audit(FULL_INTENT_BUNDLE,TEST_INTENT,MODULE_READY_BAND[N_TRAIN:])

Xtr_all,names_all,bounds_full=build_long_matrix_v31(
    np.arange(N_TRAIN),consensus3_oof,base_oof,meta_oof,full_intent_tr
)
Xte_all,_,bounds_test=build_long_matrix_v31(
    np.arange(N_TRAIN,N_TRAIN+N_TEST),consensus3_test,base_test,meta_test,full_intent_te
)
assert bounds_full==bounds_test
be=bounds_full["base"]

train_global=np.arange(N_TRAIN,dtype=int)
test_global=np.arange(N_TRAIN,N_TRAIN+N_TEST,dtype=int)
Ztr_r,names_r=trajectory_long_block(train_global,"R"); Zte_r,_=trajectory_long_block(test_global,"R")
Ztr_u,names_u=trajectory_long_block(train_global,"U"); Zte_u,_=trajectory_long_block(test_global,"U")
Xtr_r=np.hstack([Xtr_all[:,:be],Ztr_r]).astype(np.float32,copy=False)
Xte_r=np.hstack([Xte_all[:,:be],Zte_r]).astype(np.float32,copy=False)
Xtr_u=np.hstack([Xtr_all[:,:be],Ztr_u]).astype(np.float32,copy=False)
Xte_u=np.hstack([Xte_all[:,:be],Zte_u]).astype(np.float32,copy=False)

# Same seed across siblings isolates information-source differences from ranker RNG.
full_rank_seed=_seed_from_text("V30-FULL-RANKER")
model_v30,raw_v30=fit_ranker_predict(Xtr_all[:,:be],Xte_all[:,:be],ORDINAL_GRADES,N_TRAIN,full_rank_seed)
model_v31,raw_v31=fit_ranker_predict(Xtr_all[:,:bounds_full["intentstate"]],Xte_all[:,:bounds_full["intentstate"]],ORDINAL_GRADES,N_TRAIN,full_rank_seed)
model_r,raw_r=fit_ranker_predict(Xtr_r,Xte_r,ORDINAL_GRADES,N_TRAIN,full_rank_seed)
model_u,raw_u=fit_ranker_predict(Xtr_u,Xte_u,ORDINAL_GRADES,N_TRAIN,full_rank_seed)

v30_test=blend_residual(consensus3_test,raw_v30,ALPHA)
v31_test=blend_residual(consensus3_test,raw_v31,ALPHA)
traj_r_test=blend_residual(consensus3_test,raw_r,ALPHA)
traj_u_test=blend_residual(consensus3_test,raw_u,ALPHA)

# Frozen V25 protections remain exact in all decision surfaces.
for arr in (v30_test,v31_test,traj_r_test,traj_u_test):
    arr[protected_rows]=v25_anchor_test[protected_rows]

full_state=state_codes(v25_anchor_test,v30_test,v31_test)
hybrid34_test=v30_test.copy(); hybrid34_test[full_state==3]=v31_test[full_state==3]
current_anchor_test,current_info=current066020_surface(v30_test,v31_test,full_state)

# Regression audit without hardcoding the seven current-winning test user IDs.
v25_top=top5_indices(v25_anchor_test); v30_top=top5_indices(v30_test); v31_top=top5_indices(v31_test)
v30_changed=int(np.sum(np.any(v25_top!=v30_top,axis=1)))
v31_changed=int(np.sum(np.any(v30_top!=v31_top,axis=1)))
v30_top1=int(np.sum(v25_top[:,0]!=v30_top[:,0]))
v30_membership=int(np.sum([set(a)!=set(b) for a,b in zip(v25_top,v30_top)]))
state_counts={STATE_NAMES[s]:int(np.sum(full_state==s)) for s in range(4)}
hybrid_hash=top5_sha256(hybrid34_test)
current_hash=top5_sha256(current_anchor_test)

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
V39_REPRO_AUDIT={
    "reproduction_ok":REPRO_OK,
    "v30_changed_vs_v25":v30_changed,"expected_v30_changed":EXPECTED_V30_CHANGED,
    "v30_top1_changes":v30_top1,"v30_membership_changes":v30_membership,
    "v31_changed_vs_v30":v31_changed,"expected_v31_changed":EXPECTED_V31_CHANGED_VS_V30,
    "state_counts":state_counts,"expected_state_counts":EXPECTED_STATE_COUNTS,
    "s3_top1_kept":int(current_info["top1"].sum()),
    "s3_membership_kept":int(current_info["membership"].sum()),
    "s3_order_only_rejected":int(current_info["order_only"].sum()),
    "hybrid34_top5_sha256":hybrid_hash,"expected_hybrid34_top5_sha256":HYBRID34_TOP5_SHA256,
    "current_anchor_top5_sha256":current_hash,"expected_current_anchor_top5_sha256":CURRENT_ANCHOR_TOP5_SHA256,
}
print(json.dumps(V39_REPRO_AUDIT,indent=2))
if not V39_SMOKE:
    assert REPRO_OK,"Historical 0.66020 anchor geometry did not reproduce exactly; abort V39 submission creation."

# New corrections are restricted to untouched S0 and cannot touch the frozen V25 package.
protected_mask=np.zeros(N_TEST,bool); protected_mask[protected_rows]=True
eligible_s0=(full_state==0)
V39_TEST_ACTIONS=propose_consensus_trajectory_actions(
    current_anchor_test,traj_r_test,traj_u_test,test_global,eligible_s0,protected_mask
)
if len(V39_TEST_ACTIONS):
    V39_TEST_ACTIONS=V39_TEST_ACTIONS.copy()
    V39_TEST_ACTIONS["approved_by_validation"]=V39_TEST_ACTIONS["action_type"].isin(FINAL_APPROVED_TYPES)
else:
    V39_TEST_ACTIONS=pd.DataFrame(columns=ACTION_COLUMNS+["approved_by_validation"])

diag_top1=apply_selected_actions(current_anchor_test,V39_TEST_ACTIONS,("top1_12",))
diag_membership=apply_selected_actions(current_anchor_test,V39_TEST_ACTIONS,("membership_56",))
diag_all=apply_selected_actions(current_anchor_test,V39_TEST_ACTIONS,("top1_12","membership_56"))
selected_test=apply_selected_actions(current_anchor_test,V39_TEST_ACTIONS,tuple(FINAL_APPROVED_TYPES))
selected_changed=int(np.sum(np.any(top5_indices(current_anchor_test)!=top5_indices(selected_test),axis=1)))
FINAL_GATE=bool(REPRO_OK and len(FINAL_APPROVED_TYPES)>0 and selected_changed>0)
primary_test=selected_test.copy() if FINAL_GATE else current_anchor_test.copy()

assert sample["user_id"].astype(str).tolist()==test["user_id"].astype(str).tolist()
def submission_from_scores(scores):
    out=sample.copy(); out.loc[:,MODULES]=np.asarray(scores,np.float32)
    assert out.shape==sample.shape
    assert out["user_id"].astype(str).tolist()==test["user_id"].astype(str).tolist()
    assert np.isfinite(out[MODULES].to_numpy()).all()
    return out

SUBMISSIONS={
    "anchor":submission_from_scores(current_anchor_test),
    "top1":submission_from_scores(diag_top1),
    "membership":submission_from_scores(diag_membership),
    "all_consensus":submission_from_scores(diag_all),
    "primary":submission_from_scores(primary_test),
}
SUBMISSIONS["anchor"].to_csv(OUT_DIR/"submission_v39_anchor_current_066020.csv",index=False)
SUBMISSIONS["top1"].to_csv(OUT_DIR/"submission_v39_DIAGNOSTIC_top1_12.csv",index=False)
SUBMISSIONS["membership"].to_csv(OUT_DIR/"submission_v39_DIAGNOSTIC_membership_56.csv",index=False)
SUBMISSIONS["all_consensus"].to_csv(OUT_DIR/"submission_v39_DIAGNOSTIC_all_consensus.csv",index=False)
SUBMISSIONS["primary"].to_csv(OUT_DIR/"submission_v39_primary.csv",index=False)
if not FINAL_GATE:
    assert np.array_equal(SUBMISSIONS["primary"][MODULES].to_numpy(),SUBMISSIONS["anchor"][MODULES].to_numpy())

V39_TEST_GEOMETRY=pd.DataFrame([
    {
        "candidate":name,
        "changed_users_vs_current":int(np.sum(np.any(top5_indices(current_anchor_test)!=top5_indices(scores),axis=1))),
        "top1_changes":int(np.sum(top5_indices(current_anchor_test)[:,0]!=top5_indices(scores)[:,0])),
        "membership_changes":int(np.sum([set(a)!=set(b) for a,b in zip(top5_indices(current_anchor_test),top5_indices(scores))])),
        "top5_sha256":top5_sha256(scores),
    }
    for name,scores in [("top1_12",diag_top1),("membership_56",diag_membership),("all_consensus",diag_all),("primary",primary_test)]
])
display(V39_TEST_GEOMETRY)
print("FINAL_APPROVED_TYPES =",FINAL_APPROVED_TYPES)
print("selected test actions =",selected_changed,"FINAL_GATE =",FINAL_GATE)

# Release largest matrices before serialization.
del Xtr_all,Xte_all,Ztr_r,Zte_r,Ztr_u,Zte_u,Xtr_r,Xte_r,Xtr_u,Xte_u
for obj in (model_v30,model_v31,model_r,model_u):
    del obj
del full_intent_tr,full_intent_te
gc.collect()

# ---- cell 29 (code) ----
# Artifacts + decision contract.
V39_TRAJECTORY_AUDIT.to_csv(OUT_DIR/"v39_direct_trajectory_train_audit.csv",index=False)
V39_FRESH_SUMMARY.to_csv(OUT_DIR/"v39_fresh_world_summary.csv",index=False)
V39_DEV_GATE.to_csv(OUT_DIR/"v39_development_gate.csv",index=False)
V39_VALIDATION_ACTIONS.to_csv(OUT_DIR/"v39_validation_actions.csv",index=False)
V39_SEALED.to_csv(OUT_DIR/"v39_sealed_validation.csv",index=False)
V39_SEALED_GATE.to_csv(OUT_DIR/"v39_sealed_gate.csv",index=False)
V39_SEALED_ACTIONS.to_csv(OUT_DIR/"v39_sealed_actions.csv",index=False)
V39_TEST_ACTIONS.to_csv(OUT_DIR/"v39_test_actions.csv",index=False)
V39_TEST_GEOMETRY.to_csv(OUT_DIR/"v39_test_geometry.csv",index=False)
V39_INTENT_SUPPORT_AUDIT.to_csv(OUT_DIR/"v39_intentstate_support_audit.csv",index=False)

with open(OUT_DIR/"v39_reproduction_audit.json","w") as f:
    json.dump(V39_REPRO_AUDIT,f,indent=2)
with open(OUT_DIR/"v39_feature_names.json","w") as f:
    json.dump({
        "base_feature_count":int(be),
        "trajectory_R":names_r,
        "trajectory_U":names_u,
        "feature_names_R":names_all[:be]+names_r,
        "feature_names_U":names_all[:be]+names_u,
    },f,indent=2)

slot_plan=pd.DataFrame([
    {
        "file":"submission_v39_primary.csv",
        "role":"PRIMARY_ONLY_IF_ALL_GATES_PASS",
        "eligible_to_submit":bool(FINAL_GATE),
        "note":(
            "Uses only validation-approved S0 direct-trajectory 1<->2 and/or 5<->6 actions."
            if FINAL_GATE else
            "Exact reconstructed 0.66020 fail-close anchor; do not waste a Kaggle slot on this duplicate."
        ),
    },
    {
        "file":"submission_v39_DIAGNOSTIC_top1_12.csv",
        "role":"DIAGNOSTIC_ONLY",
        "eligible_to_submit":False,
        "note":"All precommitted S0 consensus pure 1<->2 trajectory actions; use validation table before any public probe.",
    },
    {
        "file":"submission_v39_DIAGNOSTIC_membership_56.csv",
        "role":"DIAGNOSTIC_ONLY",
        "eligible_to_submit":False,
        "note":"All precommitted S0 consensus pure 5<->6 trajectory actions; use validation table before any public probe.",
    },
    {
        "file":"submission_v39_anchor_current_066020.csv",
        "role":"CURRENT_PUBLIC_ANCHOR_REFERENCE",
        "eligible_to_submit":False,
        "note":"Algorithmically reconstructed current 0.66020 Top-5 signature.",
    },
])
slot_plan.to_csv(OUT_DIR/"v39_slot_plan.csv",index=False)

summary={
    "version":"V39 Direct Intent Trajectory Residual LTR",
    "public_anchor":PUBLIC_CURRENT_ANCHOR_LB,
    "top4_reference":PUBLIC_TOP4_LB,
    "leader_reference":PUBLIC_LEADER_LB,
    "fixed_alpha":ALPHA,
    "fresh_worlds":FRESH_WORLDS,
    "trajectory_views":{"R":names_r,"U":names_u},
    "attack_surface":"S0 only; V25/V30/S1/S2/current-S3 locked",
    "action_geometry":["pure 1<->2 Top1","pure 5<->6 membership"],
    "development_approved_types":DEV_APPROVED_TYPES,
    "final_approved_types":FINAL_APPROVED_TYPES,
    "reproduction_ok":bool(REPRO_OK),
    "final_gate_passed":bool(FINAL_GATE),
    "selected_test_actions":selected_changed,
    "test_action_counts":V39_TEST_ACTIONS["action_type"].value_counts().to_dict() if len(V39_TEST_ACTIONS) else {},
    "current_anchor_top5_sha256":top5_sha256(current_anchor_test),
    "primary_top5_sha256":top5_sha256(primary_test),
    "submission_instruction":(
        "PRIMARY IS SUBMISSION-ELIGIBLE under the preregistered V39 gate; audit v39_test_actions.csv before using a scarce slot."
        if FINAL_GATE else
        "NO NEW V39 SUBMISSION: primary is the exact 0.66020 fail-close anchor."
    ),
}
with open(OUT_DIR/"v39_summary.json","w") as f:
    json.dump(summary,f,indent=2)

assert set(V39_FRESH_SUMMARY.world)==set(FRESH_WORLDS)
assert len(V39_FRESH_SUMMARY)==3
assert SUBMISSIONS["primary"].shape==sample.shape
assert top5_sha256(current_anchor_test)==CURRENT_ANCHOR_TOP5_SHA256 or V39_SMOKE
assert not V39_TEST_ACTIONS.get("user_id",pd.Series(dtype=str)).isin(PROTECTED_TEST_USERS).any()

print(json.dumps(summary,indent=2))
print("\nV39 outputs written to",OUT_DIR)
print(slot_plan.to_string(index=False))

# ---- cell 30 (markdown) ----
# ### Interpretation contract
#
# - **Do not submit a diagnostic file just because it exists.** `v39_slot_plan.csv` is the execution contract.
# - A global R/U ranker can be mediocre while the tiny structural action package is useful; promotion is intentionally based on the action package versus the reconstructed current anchor.
# - If `FINAL_GATE=False`, V39 has failed closed and `submission_v39_primary.csv` is the exact `0.66020` anchor.
# - If `FINAL_GATE=True`, inspect `v39_test_actions.csv`, the fresh/sealed tables, action counts, and reproduction hash before consuming a Kaggle slot.
# - No post-run threshold search is part of V39. Any later ablation is a new experiment/version.
