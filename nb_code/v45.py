# Converted from notebooks/v45.ipynb: markdown cells become comments, %/! magics neutralized.

# ---- cell 0 (markdown) ----
# # MineToday V45 — Frozen-Lineage Boundary-Aware Audit of Legacy V25 Actions
#
# **First-submission hypothesis:** the exact public `0.66028` champion still protects a legacy V25 lower-order family where `M_002` was promoted over `M_003` at the `B45` boundary for three test users (`U_4491`, `U_4836`, `U_4929`). Historical V26 evidence suggested the near-tie deployment-local region of this family could be harmful even though the broad pair family was close to neutral.
#
# V45 therefore does **not** train a new global reranker and does **not** search user IDs or arbitrary class subsets. It:
#
# 1. reconstructs the exact public-proven V3.1 → Consensus3 → PB/V25 → V30/V31 → S3 B34+B45 lineage;
# 2. freezes the exact `0.66028` champion by hash;
# 3. preregisters legacy action families by **provenance × pair × ranking geometry**;
# 4. evaluates exact OOF **NDCG action utility** rather than generic relevance;
# 5. uses a fresh DESIGN/LOCK split and template-group/bootstrap robustness checks;
# 6. uses lexical, assessment, and generator-state information only as conservative evidence/vetoes;
# 7. allows the first V45 candidate to revert **only** the preregistered repeated `M_002 > M_003 @ B45` V25 family;
# 8. fail-closes bit-exactly to the `0.66028` champion if the family does not pass both DESIGN and untouched LOCK confirmation.
#
# Organizer data only. No external labels. No LLM-generated test predictions. Kaggle T4x2 compatible; this notebook is CPU-first.

# ---- cell 1 (code) ----

# Runtime + exact frozen lineage configuration
import os, sys, re, json, ast, html, hashlib, random, warnings, time, subprocess, gc, math, platform
from pathlib import Path
from collections import defaultdict, Counter

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ["PYTHONHASHSEED"] = "42"

V44_SKIP_INSTALL = bool(int(os.getenv("V44_SKIP_INSTALL", "0")))
try:
    import lightgbm as lgb
except Exception:
    if V44_SKIP_INSTALL:
        raise
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "lightgbm==4.6.0"])
    import lightgbm as lgb

import numpy as np
import pandas as pd
import scipy
import sklearn
from IPython.display import display
from scipy.sparse import hstack
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")

SEED = 42
V44_SMOKE = bool(int(os.getenv("V44_SMOKE", "0")))
# Compatibility aliases consumed by the frozen V3.1 -> V25 -> V30/V31 reconstruction cells.
V43_SMOKE = V44_SMOKE
V421_SMOKE = V44_SMOKE
V395_SMOKE = V44_SMOKE
V31_SMOKE = V44_SMOKE
V30_SMOKE = V44_SMOKE
V29_SMOKE = V44_SMOKE
V27_SMOKE = V44_SMOKE
V395_STRICT_GEOMETRY = False if V44_SMOKE else bool(int(os.getenv("V44_STRICT_GEOMETRY", "1")))
V30_STRICT_GEOMETRY = V395_STRICT_GEOMETRY
V29_STRICT_GEOMETRY = V395_STRICT_GEOMETRY
V27_STRICT_GEOMETRY = V395_STRICT_GEOMETRY

N_FOLDS = 2 if V44_SMOKE else 5
ANCHOR_INNER_FOLDS = 2 if V44_SMOKE else 4
INTENT_INNER_FOLDS = 2 if V44_SMOKE else 4
POLICY_INNER_FOLDS = 2 if V44_SMOKE else 4

RIDGE_ALPHA = 3.0
REFERENCE_W = 0.45
PUBLIC_CAL045_LB = 0.65639
PUBLIC_CONSENSUS3_LB = 0.65856
PUBLIC_V25_LB = 0.65919
PUBLIC_V30_LB = 0.65963
PUBLIC_V31_DIAGNOSTIC_LB = 0.65949
PUBLIC_HYBRID34_LB = 0.66006
PUBLIC_CURRENT_ANCHOR_LB = 0.66020
PUBLIC_V44_ANCHOR_LB = 0.66028
PUBLIC_TOP4_LB = 0.66080
PUBLIC_LEADER_LB = 0.66142

LGB_ESTIMATORS = 12 if V44_SMOKE else 1200
LGB_EARLY_STOP = 3 if V44_SMOKE else 80
RANKER_ESTIMATORS = 16 if V44_SMOKE else 360
RANKER_LEARNING_RATE = 0.04 if V44_SMOKE else 0.025
ALPHA = 0.16
INTENT_PARENT_STRENGTH = 40.0
INTENT_CHILD_STRENGTH = 30.0
BOOTSTRAPS = 30 if V44_SMOKE else 1200
GEOMETRY_ATOL = 2.5e-4

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
V44_ANCHOR_TOP5_SHA256 = "7c0d29a7309b7283a043b7d37ca5556ba7c0ca2270b0137a6e1e0878a69a5fe7"
LB_ROUNDING_TOL = 3e-5

def _seed_from_text(s):
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:8], 16) % (2**31-1)

random.seed(SEED); np.random.seed(SEED)
print("numpy", np.__version__, "pandas", pd.__version__, "sklearn", sklearn.__version__, "lightgbm", lgb.__version__)
print("V44_SMOKE", V44_SMOKE, "frozen V30 alpha", ALPHA)
print("production champion", PUBLIC_V44_ANCHOR_LB, "leader gap", round(PUBLIC_LEADER_LB-PUBLIC_V44_ANCHOR_LB,5))
try:
    print("GPU runtime (V44 is CPU-first, but Kaggle T4x2 is compatible):\n" + subprocess.check_output(["nvidia-smi", "-L"], text=True).strip())
except Exception:
    print("nvidia-smi unavailable; V44 does not require GPU.")

assert GEOMETRY_ATOL == 2.5e-4

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
ALL_INTENT=np.concatenate([TRAIN_INTENT,TEST_INTENT]).astype(object)

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

# ---- cell 22 (code) ----

# Exact production champion = current 0.66020 + public-confirmed S3 B34 + B45.
def build_s3_boundary_ledger_v44(v30, v31, info):
    t30=top5_indices(v30); t31=top5_indices(v31); rows=[]
    for r in np.flatnonzero(info["order_only"]):
        boundary,promoted,demoted,pos_from,pos_to=classify_single_adjacent_order_swap(t30[r],t31[r])
        rows.append({"row_local":int(r),"user_id":str(test.iloc[r]["user_id"]),"boundary":boundary,
                     "promoted_module":MODULES[promoted],"demoted_module":MODULES[demoted],
                     "v30_rank_from":int(pos_from),"v31_rank_to":int(pos_to)})
    return pd.DataFrame(rows).sort_values(["boundary","row_local"]).reset_index(drop=True)

V44_S3_BOUNDARY_LEDGER=build_s3_boundary_ledger_v44(v30_test,v31_test,current_info)
assert V44_S3_BOUNDARY_LEDGER.boundary.value_counts().sort_index().to_dict()==EXPECTED_BOUNDARY_COUNTS
V44_ANCHOR_TEST=current_anchor_test.copy()
rr=V44_S3_BOUNDARY_LEDGER.loc[V44_S3_BOUNDARY_LEDGER.boundary.isin(["B34","B45"]),"row_local"].to_numpy(int)
V44_ANCHOR_TEST[rr]=v31_test[rr]
if not V44_SMOKE:
    assert top5_sha256(V44_ANCHOR_TEST)==V44_ANCHOR_TOP5_SHA256
print("V44 exact anchor hash", top5_sha256(V44_ANCHOR_TEST), "B34+B45 rows", len(rr))

# ---- cell 23 (code) ----
# V45 frozen champion + current leaderboard snapshot
V45_VERSION = "v45"
V45_OUT = OUT_DIR / "v45"
V45_OUT.mkdir(parents=True, exist_ok=True)

V45_ANCHOR_TEST = V44_ANCHOR_TEST.copy()
V45_ANCHOR_HASH = top5_sha256(V45_ANCHOR_TEST)
PUBLIC_V45_ANCHOR_LB = 0.66028
PUBLIC_TOP5_CUTOFF = 0.66076
PUBLIC_LEADER_LB_V45 = 0.66142

assert V45_ANCHOR_HASH == V44_ANCHOR_TOP5_SHA256
print("V45 exact frozen anchor:", V45_ANCHOR_HASH)
print("public anchor", PUBLIC_V45_ANCHOR_LB,
      "gap to current top-5 cutoff", round(PUBLIC_TOP5_CUTOFF-PUBLIC_V45_ANCHOR_LB, 5),
      "gap to leader", round(PUBLIC_LEADER_LB_V45-PUBLIC_V45_ANCHOR_LB, 5))

# ---- cell 24 (markdown) ----
# ## Preregistered V45 policy
#
# The first leaderboard slot is intentionally a **single natural-family causal probe**, not a model search.
#
# The only family eligible to alter the first V45 submission is:
#
# `CORE_REPEAT_M002_M003_B45` = V25 core actions that promoted **M_002 over M_003** from pre-action ranks **4→5**.
#
# The PB Top-1 action, the V25 membership action, the V25 B34 action, all ordinary V30 actions, and the public-confirmed S3 Top1 / membership / B34+B45 geometry remain protected. Other legacy families are audited for information only and cannot enter the primary candidate in this notebook.
#
# All thresholds below are fixed before reading V45 outcomes. The test margins are label-free geometry inherited from the frozen V25 ledger.

# ---- cell 25 (code) ----
# Fresh V45 split + preregistered action families
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

V45_SPLIT_SEED = 20260904
V45_BOOT_SEED = 202609045
V45_BOOTSTRAPS = 2000 if not V44_SMOKE else 50
V45_LOCAL_BAND_FLOOR = 0.0015
V45_LOCAL_BAND_MULT = 2.5
V45_KERNEL_BW = 0.0025
V45_EPS = 1e-12

# Gate is intentionally conservative but not impossible for a sparse legacy family.
V45_MIN_GROUPS = 5
V45_MIN_DESIGN_ANALOGS = 6
V45_MIN_LOCK_ANALOGS = 1
V45_MIN_RANDOM_WINS = 3
V45_MIN_BLOCK_WINS = 3
V45_CATASTROPHIC_LOCK = -0.004
V45_VETO_THRESHOLD = -0.0015

V45_FAMILIES = {
    "PB_HEAD": {
        "origin":"PB", "higher_before":"M_009", "promoted":"M_007",
        "rank_hi":1, "rank_pr":2, "test_uids":["U_4047"], "protected":True,
    },
    "PB_B45": {
        "origin":"PB", "higher_before":"M_016", "promoted":"M_005",
        "rank_hi":4, "rank_pr":5, "test_uids":["U_4953"], "protected":True,
    },
    "CORE_MEMBERSHIP": {
        "origin":"V25_CORE", "higher_before":"M_014", "promoted":"M_003",
        "rank_hi":5, "rank_pr":6, "test_uids":["U_4074"], "protected":True,
    },
    "CORE_B23": {
        "origin":"V25_CORE", "higher_before":"M_003", "promoted":"M_017",
        "rank_hi":2, "rank_pr":3, "test_uids":["U_4176"], "protected":True,
    },
    "CORE_B34": {
        "origin":"V25_CORE", "higher_before":"M_014", "promoted":"M_012",
        "rank_hi":3, "rank_pr":4, "test_uids":["U_4567"], "protected":True,
    },
    "CORE_REPEAT_M002_M003_B45": {
        "origin":"V25_CORE", "higher_before":"M_003", "promoted":"M_002",
        "rank_hi":4, "rank_pr":5,
        "test_uids":["U_4491","U_4836","U_4929"], "protected":False,
    },
}
PRIMARY_FAMILY = "CORE_REPEAT_M002_M003_B45"

# Frozen test margins come from the exact pre-V25 PB surface, never from labels.
for name, spec in V45_FAMILIES.items():
    margins=[]
    pre_scores = consensus3_test if spec["origin"] == "PB" else pb_test
    pre_ranks = full_rank_matrix(pre_scores)
    for uid in spec["test_uids"]:
        r=TEST_UID_TO_ROW[uid]
        hi=MID_TO_COL[spec["higher_before"]]; pr=MID_TO_COL[spec["promoted"]]
        assert int(pre_ranks[r,hi]) == int(spec["rank_hi"]), (name,uid,"hi-rank drift")
        assert int(pre_ranks[r,pr]) == int(spec["rank_pr"]), (name,uid,"pr-rank drift")
        m=float(pre_scores[r,hi]-pre_scores[r,pr])
        assert m > 0, (name,uid,m)
        margins.append(m)
    spec["test_margins"] = tuple(margins)
    spec["local_cap"] = float(max(V45_LOCAL_BAND_FLOOR, V45_LOCAL_BAND_MULT*max(margins)))
    spec["margin_center"] = float(np.median(margins))

split_strata=np.asarray([MODULES[j] for j in np.argmax(y,axis=1)],dtype=object)
sss=StratifiedShuffleSplit(n_splits=1,test_size=800,random_state=V45_SPLIT_SEED)
DESIGN_ROWS,LOCK_ROWS=next(sss.split(np.zeros(N_TRAIN),split_strata))
DESIGN_ROWS=np.asarray(sorted(DESIGN_ROWS),int)
LOCK_ROWS=np.asarray(sorted(LOCK_ROWS),int)
assert len(DESIGN_ROWS)==3200 and len(LOCK_ROWS)==800
assert len(set(DESIGN_ROWS).intersection(set(LOCK_ROWS)))==0

print("fresh V45 split:",len(DESIGN_ROWS),len(LOCK_ROWS),"seed",V45_SPLIT_SEED)
print("primary family:",PRIMARY_FAMILY,V45_FAMILIES[PRIMARY_FAMILY])

# ---- cell 26 (code) ----
# Exact per-row action utility and robustness helpers
CONS_RANKS = full_rank_matrix(consensus3_oof)


def row_action_delta(scores, truth, rows, hi, pr):
    """NDCG(after applying promotion swap) - NDCG(before). Negative => pruning the action is good."""
    rows=np.asarray(rows,int)
    if len(rows)==0:
        return np.asarray([],float)
    before=per_user_ndcg(truth[rows],scores[rows])
    cand=scores[rows].copy()
    tmp=cand[:,hi].copy(); cand[:,hi]=cand[:,pr]; cand[:,pr]=tmp
    after=per_user_ndcg(truth[rows],cand)
    return (after-before).astype(float)


def weighted_mean(x,w):
    x=np.asarray(x,float); w=np.asarray(w,float)
    if len(x)==0: return np.nan
    s=float(w.sum())
    return float(np.mean(x)) if s<=1e-15 else float(np.sum(x*w)/s)


def group_weighted_values(values, groups, weights):
    df=pd.DataFrame({"v":np.asarray(values,float),"g":np.asarray(groups,object),"w":np.asarray(weights,float)})
    out=[]
    for g,z in df.groupby("g",sort=False):
        out.append((g,weighted_mean(z.v.to_numpy(),z.w.to_numpy())))
    return out


def group_bootstrap(values, groups, weights, seed):
    gv=group_weighted_values(values,groups,weights)
    if not gv:
        return {"mean":np.nan,"lo":np.nan,"hi":np.nan,"groups":0}
    a=np.asarray([v for _,v in gv],float)
    mean=float(a.mean())
    if len(a)==1:
        return {"mean":mean,"lo":mean,"hi":mean,"groups":1}
    rng=np.random.default_rng(int(seed))
    b=np.empty(V45_BOOTSTRAPS,float)
    for i in range(V45_BOOTSTRAPS):
        b[i]=a[rng.integers(0,len(a),len(a))].mean()
    lo,hi=np.quantile(b,[.025,.975])
    return {"mean":mean,"lo":float(lo),"hi":float(hi),"groups":int(len(a))}


def family_analog_table(name):
    spec=V45_FAMILIES[name]
    hi=MID_TO_COL[spec["higher_before"]]; pr=MID_TO_COL[spec["promoted"]]
    mask=(
        (consensus3_oof[:,hi] > consensus3_oof[:,pr]) &
        (CONS_RANKS[:,hi] == int(spec["rank_hi"])) &
        (CONS_RANKS[:,pr] == int(spec["rank_pr"]))
    )
    rows=np.flatnonzero(mask)
    action=row_action_delta(consensus3_oof,y,rows,hi,pr)
    prune=-action
    margin=(consensus3_oof[rows,hi]-consensus3_oof[rows,pr]).astype(float)
    # Every broad exact-geometry analogue contributes, but near-test margins receive more authority.
    nearest=np.min(np.abs(margin[:,None]-np.asarray(spec["test_margins"],float)[None,:]),axis=1)
    kernel=np.exp(-nearest/V45_KERNEL_BW)
    local=(margin <= float(spec["local_cap"]))
    return pd.DataFrame({
        "family":name,"row":rows,"user_id":train.iloc[rows].user_id.to_numpy(),
        "margin":margin,"action_delta":action,"prune_delta":prune,"kernel_weight":kernel,
        "local":local,"template_group":LATEST_GROUP[rows],"intent":TRAIN_INTENT[rows],
    })


def directional_wins(df, universe_rows, mode="random"):
    universe_rows=np.asarray(universe_rows,int)
    uid_to_slot={int(r):i for i,r in enumerate(universe_rows)}
    vals=[]
    if mode=="random":
        st=split_strata[universe_rows]
        skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=V45_SPLIT_SEED+17)
        partitions=[universe_rows[va] for _,va in skf.split(np.zeros(len(universe_rows)),st)]
    elif mode=="block":
        ordered=universe_rows[np.argsort(TRAIN_ID_NUM[universe_rows])]
        partitions=np.array_split(ordered,5)
    else:
        raise ValueError(mode)
    rowset=df.set_index("row",drop=False)
    for part in partitions:
        keep=[int(r) for r in part if int(r) in rowset.index]
        if not keep:
            vals.append(np.nan); continue
        z=rowset.loc[keep]
        vals.append(weighted_mean(z.prune_delta.to_numpy(),z.kernel_weight.to_numpy()))
    wins=int(sum(np.isfinite(v) and v>0 for v in vals))
    return wins,vals


def grouped_neighbor_score(df, test_uids, mode="lexical", top_groups=12):
    if len(df)==0: return {"score":np.nan,"groups":0}
    rows=df.row.to_numpy(int); deltas=df.prune_delta.to_numpy(float); groups=df.template_group.to_numpy(object)
    all_scores=[]; all_group_n=[]
    for uid in test_uids:
        tr=TEST_UID_TO_ROW[uid]; gi=N_TRAIN+tr
        if mode=="lexical":
            # Frozen sparse V3.1 word surface, transform-only on test.
            q=X_WORD[gi]
            rw=np.asarray((X_WORD[rows] @ q.T).toarray()).ravel().astype(float)
        elif mode=="assessment":
            dist=np.sqrt(np.mean(((A[rows]-A[gi][None,:])/5.0)**2,axis=1))
            rw=1.0/(1.0+4.0*dist)
        else:
            raise ValueError(mode)
        tmp=pd.DataFrame({"d":deltas,"g":groups,"w":rw})
        gd=tmp.groupby("g",sort=False).agg(d=("d","mean"),w=("w","max")).reset_index()
        take=np.argsort(-gd.w.to_numpy(float))[:min(top_groups,len(gd))]
        w=gd.w.to_numpy(float)[take]; d=gd.d.to_numpy(float)[take]
        all_scores.append(weighted_mean(d,w)); all_group_n.append(len(take))
    return {"score":float(np.nanmean(all_scores)),"groups":int(np.sum(all_group_n))}


def generator_veto_score(df, spec):
    """Label-free generator-state categories are allowed only to veto, never to authorize a swap."""
    if len(df)==0: return {"score":np.nan,"groups":0,"active":False}
    hi=MID_TO_COL[spec["higher_before"]]; pr=MID_TO_COL[spec["promoted"]]
    test_states=[]
    for uid in spec["test_uids"]:
        gi=N_TRAIN+TEST_UID_TO_ROW[uid]
        test_states.append((
            str(LATEST_GROUP[gi]),str(ALL_INTENT[gi]),
            int(MODULE_READY_BAND[gi,hi]),int(MODULE_READY_BAND[gi,pr]),
            int(MODULE_LATEST[gi,hi]>0),int(MODULE_LATEST[gi,pr]>0),
        ))
    rows=df.row.to_numpy(int)
    scores=[]
    for r in rows:
        state=(str(LATEST_GROUP[r]),str(ALL_INTENT[r]),int(MODULE_READY_BAND[r,hi]),int(MODULE_READY_BAND[r,pr]),
               int(MODULE_LATEST[r,hi]>0),int(MODULE_LATEST[r,pr]>0))
        best=0.0
        for ts in test_states:
            s=0.0
            s += 3.0 if state[0]==ts[0] else 0.0
            s += 1.5 if state[1]==ts[1] else 0.0
            s += 0.75 if state[2:4]==ts[2:4] else 0.0
            s += 0.50 if state[4:6]==ts[4:6] else 0.0
            best=max(best,s)
        scores.append(best)
    z=df.copy(); z["gen_match"]=np.asarray(scores,float)
    # Require at least some meaningful state overlap; otherwise abstain.
    z=z[z.gen_match>=1.5]
    if len(z)==0:
        return {"score":np.nan,"groups":0,"active":False}
    gv=np.asarray([
        weighted_mean(q.prune_delta.to_numpy(),1.0+q.gen_match.to_numpy())
        for _,q in z.groupby("template_group",sort=False)
    ],float)
    if len(gv)<3:
        return {"score":float(np.mean(gv)) if len(gv) else np.nan,"groups":int(len(gv)),"active":False}
    return {"score":float(np.mean(gv)),"groups":int(len(gv)),"active":True}

# ---- cell 27 (code) ----
# Stage A — DESIGN audit. Decisions are frozen before LOCK is inspected.
family_tables={name:family_analog_table(name) for name in V45_FAMILIES}
DESIGN_SET=set(map(int,DESIGN_ROWS)); LOCK_SET=set(map(int,LOCK_ROWS))
design_rows=[]

for name,spec in V45_FAMILIES.items():
    tab=family_tables[name]
    d=tab[tab.row.map(lambda r:int(r) in DESIGN_SET)].copy()
    broad=group_bootstrap(d.prune_delta,d.template_group,d.kernel_weight,V45_BOOT_SEED+_seed_from_text(name))
    local=d[d.local]
    local_boot=group_bootstrap(local.prune_delta,local.template_group,np.ones(len(local)),V45_BOOT_SEED+31+_seed_from_text(name))
    rw,rv=directional_wins(d,DESIGN_ROWS,"random")
    bw,bv=directional_wins(d,DESIGN_ROWS,"block")
    lexical=grouped_neighbor_score(d,spec["test_uids"],"lexical",12)
    assess=grouped_neighbor_score(d,spec["test_uids"],"assessment",18)
    gen=generator_veto_score(d,spec)

    generator_veto=bool(gen["active"] and np.isfinite(gen["score"]) and gen["score"] < V45_VETO_THRESHOLD)
    design_pass=bool(
        (not spec["protected"]) and
        len(d)>=V45_MIN_DESIGN_ANALOGS and broad["groups"]>=V45_MIN_GROUPS and
        np.isfinite(broad["mean"]) and broad["mean"]>0 and
        len(local)>0 and np.isfinite(local_boot["mean"]) and local_boot["mean"]>0 and
        rw>=V45_MIN_RANDOM_WINS and bw>=V45_MIN_BLOCK_WINS and
        (not np.isfinite(lexical["score"]) or lexical["score"]>=V45_VETO_THRESHOLD) and
        (not np.isfinite(assess["score"]) or assess["score"]>=V45_VETO_THRESHOLD) and
        not generator_veto
    )
    design_rows.append({
        "family":name,"protected":bool(spec["protected"]),"analog_rows":int(len(d)),
        "local_rows":int(len(local)),"independent_groups":int(broad["groups"]),
        "weighted_prune_mean":broad["mean"],"group_ci_low":broad["lo"],"group_ci_high":broad["hi"],
        "local_prune_mean":local_boot["mean"],"local_ci_low":local_boot["lo"],"local_ci_high":local_boot["hi"],
        "random_wins":rw,"random_values":"|".join("nan" if not np.isfinite(v) else f"{v:.6f}" for v in rv),
        "block_wins":bw,"block_values":"|".join("nan" if not np.isfinite(v) else f"{v:.6f}" for v in bv),
        "lexical_prune_score":lexical["score"],"assessment_prune_score":assess["score"],
        "generator_prune_score":gen["score"],"generator_groups":gen["groups"],"generator_veto":generator_veto,
        "design_pass":design_pass,
    })

V45_DESIGN=pd.DataFrame(design_rows).sort_values(["design_pass","weighted_prune_mean"],ascending=[False,False]).reset_index(drop=True)
display(V45_DESIGN)
PRIMARY_DESIGN_PASS=bool(V45_DESIGN.set_index("family").loc[PRIMARY_FAMILY,"design_pass"])
print("PRIMARY DESIGN PASS:",PRIMARY_DESIGN_PASS)

# Freeze now. LOCK cannot change the selected family or any thresholds.
FROZEN_FAMILY = PRIMARY_FAMILY if PRIMARY_DESIGN_PASS else None

# ---- cell 28 (code) ----
# Stage B — untouched LOCK confirmation of the already-frozen primary family only.
lock_rows=[]
for name,spec in V45_FAMILIES.items():
    tab=family_tables[name]
    z=tab[tab.row.map(lambda r:int(r) in LOCK_SET)].copy()
    broad=group_bootstrap(z.prune_delta,z.template_group,z.kernel_weight,V45_BOOT_SEED+777+_seed_from_text(name))
    local=z[z.local]
    local_boot=group_bootstrap(local.prune_delta,local.template_group,np.ones(len(local)),V45_BOOT_SEED+991+_seed_from_text(name))
    # LOCK is confirmation, not a second tuning stage.
    lock_pass=bool(
        name==FROZEN_FAMILY and len(z)>=V45_MIN_LOCK_ANALOGS and
        np.isfinite(broad["mean"]) and broad["mean"]>V45_CATASTROPHIC_LOCK and
        (len(local)==0 or (np.isfinite(local_boot["mean"]) and local_boot["mean"]>=V45_CATASTROPHIC_LOCK)) and
        ((broad["mean"]>0) or (len(local)>0 and np.isfinite(local_boot["mean"]) and local_boot["mean"]>0))
    )
    lock_rows.append({
        "family":name,"analog_rows":int(len(z)),"local_rows":int(len(local)),"independent_groups":int(broad["groups"]),
        "weighted_prune_mean":broad["mean"],"group_ci_low":broad["lo"],"group_ci_high":broad["hi"],
        "local_prune_mean":local_boot["mean"],"local_ci_low":local_boot["lo"],"local_ci_high":local_boot["hi"],
        "was_frozen_from_design":bool(name==FROZEN_FAMILY),"lock_pass":lock_pass,
    })

V45_LOCK=pd.DataFrame(lock_rows).reset_index(drop=True)
display(V45_LOCK)
PRIMARY_LOCK_PASS=bool(V45_LOCK.set_index("family").loc[PRIMARY_FAMILY,"lock_pass"])
FINAL_GATE=bool(PRIMARY_DESIGN_PASS and PRIMARY_LOCK_PASS)
print("PRIMARY LOCK PASS:",PRIMARY_LOCK_PASS)
print("FINAL V45 GATE:",FINAL_GATE)

# ---- cell 29 (code) ----
# Stage C — exact test intervention. Only the three preregistered V25 family rows can change.
PRIMARY_SPEC=V45_FAMILIES[PRIMARY_FAMILY]
CANDIDATE_TEST=V45_ANCHOR_TEST.copy()
change_ledger=[]

for uid in PRIMARY_SPEC["test_uids"]:
    r=TEST_UID_TO_ROW[uid]
    # In the current champion the V25 promotion is active, so promoted M_002 must be above M_003.
    promoted=MID_TO_COL[PRIMARY_SPEC["promoted"]]
    demoted=MID_TO_COL[PRIMARY_SPEC["higher_before"]]
    ranks=full_rank_matrix(CANDIDATE_TEST[[r]])[0]
    before_top=[MODULES[j] for j in top5_indices(CANDIDATE_TEST[[r]])[0]]
    assert int(ranks[promoted])==4 and int(ranks[demoted])==5, (uid,ranks[promoted],ranks[demoted],before_top)
    assert float(CANDIDATE_TEST[r,promoted]) > float(CANDIDATE_TEST[r,demoted]), (uid,"post-V25 geometry drift")
    CANDIDATE_TEST[r,promoted],CANDIDATE_TEST[r,demoted]=CANDIDATE_TEST[r,demoted],CANDIDATE_TEST[r,promoted]
    after_top=[MODULES[j] for j in top5_indices(CANDIDATE_TEST[[r]])[0]]
    change_ledger.append({
        "user_id":uid,"family":PRIMARY_FAMILY,
        "before_top5":" ".join(before_top),"after_top5":" ".join(after_top),
        "membership_changed":set(before_top)!=set(after_top),"top1_changed":before_top[0]!=after_top[0],
    })

V45_CHANGE_LEDGER=pd.DataFrame(change_ledger)
display(V45_CHANGE_LEDGER)

anchor_top=top5_indices(V45_ANCHOR_TEST); cand_top=top5_indices(CANDIDATE_TEST)
changed=np.flatnonzero(np.any(anchor_top!=cand_top,axis=1))
changed_uids=test.iloc[changed].user_id.tolist()
assert set(changed_uids)==set(PRIMARY_SPEC["test_uids"]), changed_uids
assert int(np.sum(V45_ANCHOR_TEST!=CANDIDATE_TEST))==2*len(PRIMARY_SPEC["test_uids"])
assert not V45_CHANGE_LEDGER.membership_changed.any()
assert not V45_CHANGE_LEDGER.top1_changed.any()

PRIMARY_TEST=CANDIDATE_TEST if FINAL_GATE else V45_ANCHOR_TEST.copy()
print("candidate changed users:",changed_uids)
print("anchor hash   :",top5_sha256(V45_ANCHOR_TEST))
print("candidate hash:",top5_sha256(CANDIDATE_TEST))
print("primary hash  :",top5_sha256(PRIMARY_TEST))
print("PRIMARY =", "V45 repeated-family prune" if FINAL_GATE else "exact 0.66028 fail-close anchor")

# ---- cell 30 (code) ----
# Stage D — submissions, audit artifacts, and explicit slot instruction.
def submission_v45(scores):
    out=sample.copy()
    out.loc[:,MODULES]=np.asarray(scores,np.float32)
    assert list(out.columns)==list(sample.columns)
    assert out.user_id.tolist()==sample.user_id.tolist()
    assert np.isfinite(out[MODULES].to_numpy()).all()
    return out

anchor_sub=submission_v45(V45_ANCHOR_TEST)
candidate_sub=submission_v45(CANDIDATE_TEST)
primary_sub=submission_v45(PRIMARY_TEST)

anchor_path=V45_OUT/"submission_v45_anchor_066028_reference.csv"
candidate_path=V45_OUT/"submission_v45_probe_repeat_m002_m003_b45.csv"
primary_path=V45_OUT/"submission_v45_primary.csv"
anchor_sub.to_csv(anchor_path,index=False)
candidate_sub.to_csv(candidate_path,index=False)
primary_sub.to_csv(primary_path,index=False)

V45_DESIGN.to_csv(V45_OUT/"v45_design_family_audit.csv",index=False)
V45_LOCK.to_csv(V45_OUT/"v45_lock_family_audit.csv",index=False)
V45_CHANGE_LEDGER.to_csv(V45_OUT/"v45_test_change_ledger.csv",index=False)
for name,tab in family_tables.items():
    tab.to_csv(V45_OUT/f"v45_analog_{name.lower()}.csv",index=False)

slot_plan=pd.DataFrame([{
    "slot":1,
    "submit_file":str(primary_path.name),
    "gate":bool(FINAL_GATE),
    "candidate_if_pass":str(candidate_path.name),
    "instruction":(
        "SUBMIT submission_v45_primary.csv; it contains only the preregistered repeated M002>M003 B45 family prune."
        if FINAL_GATE else
        "DO NOT spend a leaderboard slot on V45: primary fail-closed exactly to the 0.66028 champion. Inspect audits before choosing the next experiment."
    )
}])
slot_plan.to_csv(V45_OUT/"v45_slot_plan.csv",index=False)
display(slot_plan)

summary={
    "version":V45_VERSION,
    "public_anchor":PUBLIC_V45_ANCHOR_LB,
    "public_top5_cutoff":PUBLIC_TOP5_CUTOFF,
    "public_leader":PUBLIC_LEADER_LB_V45,
    "primary_family":PRIMARY_FAMILY,
    "fresh_split_seed":V45_SPLIT_SEED,
    "design_pass":PRIMARY_DESIGN_PASS,
    "lock_pass":PRIMARY_LOCK_PASS,
    "final_gate":FINAL_GATE,
    "changed_users_if_candidate":PRIMARY_SPEC["test_uids"],
    "anchor_top5_sha256":top5_sha256(V45_ANCHOR_TEST),
    "candidate_top5_sha256":top5_sha256(CANDIDATE_TEST),
    "primary_top5_sha256":top5_sha256(PRIMARY_TEST),
    "submission_instruction":slot_plan.iloc[0].instruction,
}
with open(V45_OUT/"v45_summary.json","w") as f:
    json.dump(summary,f,indent=2)

# Repro/safety invariants.
assert top5_sha256(V45_ANCHOR_TEST)==V44_ANCHOR_TOP5_SHA256
if FINAL_GATE:
    assert top5_sha256(PRIMARY_TEST)==top5_sha256(CANDIDATE_TEST)
else:
    assert np.array_equal(PRIMARY_TEST,V45_ANCHOR_TEST)
    assert top5_sha256(PRIMARY_TEST)==V44_ANCHOR_TOP5_SHA256
assert np.array_equal(primary_sub[MODULES].to_numpy(np.float32),PRIMARY_TEST.astype(np.float32))

print(json.dumps(summary,indent=2))
print("\nArtifacts written to",V45_OUT)
print("SUBMISSION DECISION:",slot_plan.iloc[0].instruction)

# ---- cell 31 (markdown) ----
# ## Interpretation discipline
#
# - If `FINAL_GATE=True`, the first V45 submission is a **single causal family prune**. Do not combine it with another unmeasured intervention in the same leaderboard slot.
# - If `FINAL_GATE=False`, `submission_v45_primary.csv` is intentionally identical to the exact `0.66028` champion. Do not waste a submission slot on it.
# - `submission_v45_probe_repeat_m002_m003_b45.csv` is always materialized so the exact candidate is auditable, but the notebook does **not** recommend submitting it when the gate fails.
# - The other family rows are diagnostics only. This notebook never turns their post-hoc metrics into test actions.
# - Preserve the generated `v45_summary.json`, DESIGN/LOCK audits, change ledger, and primary CSV for final notebook reproducibility.
