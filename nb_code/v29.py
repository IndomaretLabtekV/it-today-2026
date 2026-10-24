# Converted from notebooks/v29.ipynb: markdown cells become comments, %/! magics neutralized.

# ---- cell 0 (markdown) ----
# # MineToday V29 — Curriculum-Aware Residual Learning-to-Rank
#
# **Active public anchor:** V25 core package = **0.65919**  
# **Leader reference at design time:** **0.66112**  
# **Metric:** user-level NDCG@5
#
# V29 is an architectural reset after V27/V27.1 showed that reliable per-user actions become too sparse, while V28 showed that broad graded-score recalibration has enough coverage but the correction direction does not generalize. The new hypothesis is that the remaining headroom lives in **latent curriculum / label structure** that Consensus3 does not explicitly model.
#
# V29 keeps the exact public-proven V3.1 sparse → binary-relevance → V12.2 Consensus3 backbone and the exact V25 public package. It adds a conservative **LambdaRank residual expert**, not a replacement recommender.
#
# ### New information sources
#
# 1. **User × module curriculum/skill-gap interactions** derived only from `user_assessments.csv` and the prerequisite text in `modules_catalog.csv`.
# 2. **Fold-safe module co-occurrence graph** estimated only from the training side of each evaluation fold.
# 3. **Fold-safe residual graph** derived from `Y - Consensus3`, including conditional residual response and residual-correlation propagation.
# 4. **NDCG-aware LambdaRank** on 68k long-format user-module rows with ordinal relevance grades 0..6.
#
# ### Safety / generalization contract
#
# - Consensus3/V25 is always the anchor feature and deployment baseline.
# - The LTR output is aligned back to the anchor's within-user scale and blended only through a small preregistered residual coefficient.
# - Candidate selection uses only fully nested template-group OOF.
# - A fresh template-disjoint LOCK and chronological U3001–U4000 block are rejection-only audits.
# - Label/residual graph features are cross-fitted for ranker-training rows and never use validation/test labels.
# - The eight public-tested V22.3/V25 rows are protected exactly at deployment.
# - No new embedding/LLM branch is introduced in V29.
# - `submission_v29_primary.csv` fails closed to the exact 0.65919 anchor whenever the frozen policy does not generalize.
#
# The notebook also computes an **OOF oracle-headroom diagnostic** on the 100 worst strict-anchor users and exports a human-readable error-audit table so the failure modes can be inspected manually after the run.

# ---- cell 1 (code) ----
# Runtime + deterministic V29 configuration
import os, sys, re, json, ast, html, hashlib, random, warnings, time, subprocess, gc
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ["PYTHONHASHSEED"] = "42"

V29_SKIP_INSTALL = bool(int(os.getenv("V29_SKIP_INSTALL", "0")))
try:
    import lightgbm as lgb
except Exception:
    if V29_SKIP_INSTALL:
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
V29_SMOKE = bool(int(os.getenv("V29_SMOKE", "0")))
V29_STRICT_GEOMETRY = bool(int(os.getenv("V29_STRICT_GEOMETRY", "1")))
# Compatibility aliases used by frozen reconstruction cells inherited from earlier versions.
V27_SMOKE = V29_SMOKE
V27_STRICT_GEOMETRY = V29_STRICT_GEOMETRY

N_FOLDS = 2 if V29_SMOKE else 5
ANCHOR_INNER_FOLDS = 2 if V29_SMOKE else 4
GRAPH_INNER_FOLDS = 2 if V29_SMOKE else 3
RIDGE_ALPHA = 3.0
REFERENCE_W = 0.45
PUBLIC_CAL045_LB = 0.65639
PUBLIC_CONSENSUS3_LB = 0.65856
PUBLIC_PRE_V25_PB = 0.65912
PUBLIC_V25_LB = 0.65919
PUBLIC_CURRENT_PB = PUBLIC_V25_LB
PUBLIC_LEADER_LB = 0.66112

LGB_ESTIMATORS = 40 if V29_SMOKE else 1200
LGB_EARLY_STOP = 8 if V29_SMOKE else 80
RANKER_ESTIMATORS = 35 if V29_SMOKE else 360
RANKER_LEARNING_RATE = 0.04 if V29_SMOKE else 0.025
BOOTSTRAPS = 80 if V29_SMOKE else 1400
PSEUDO_PUBLIC_MASKS = 20 if V29_SMOKE else 180
GEOMETRY_ATOL = 2.5e-4

# Small preregistered residual blend family. Ranker scores are first aligned to the anchor scale.
RESIDUAL_ALPHAS = [0.06, 0.10] if V29_SMOKE else [0.06, 0.10, 0.16]
ABLATION_ORDER = ["ltr_base", "ltr_gap", "ltr_gap_cooc", "ltr_gap_cooc_resid"]

# Strict selection and rejection-only audit gates.
MIN_STRICT_MEAN = 1e-6
MIN_STRICT_FOLD_WINS = 4 if not V29_SMOKE else 1
MIN_STRICT_CI_LOW = -5e-5
MIN_ORDINARY_AUDIT = -3e-5
MIN_LOCK_DELTA = 1e-6
MIN_CHRONO_DELTA = -1e-6
PSEUDO_PUBLIC_MIN_POS_RATE = 0.72
PSEUDO_PUBLIC_MIN_P10 = -5e-5
MIN_TEST_CHANGED_USERS = 15 if not V29_SMOKE else 1
MAX_TEST_CHANGED_USERS = 320
MAX_TEST_TOP5_MEMBERSHIP_CHANGE = 0.12
MAX_TEST_TOP1_CHANGE = 0.08
MAX_TEST_ORDER_CHANGE = 0.35

# Oracle diagnostic thresholds are fixed before looking at results.
ORACLE_WORST_N = 100
ORACLE_PAIR_SUPPORT = 3 if V29_SMOKE else 15
ORACLE_MIN_COND = 0.15
ORACLE_MIN_LIFT = 1.20

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
print("V29_SMOKE", V29_SMOKE, "leader gap", round(PUBLIC_LEADER_LB-PUBLIC_CURRENT_PB, 5))
try:
    print("GPU runtime:\n" + subprocess.check_output(["nvidia-smi", "-L"], text=True).strip())
except Exception:
    print("nvidia-smi unavailable; V29 is T4x2-compatible but deliberately CPU-first.")

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
# ## Frozen public-proven anchor backbone
#
# The next cells reproduce the exact V3.1 sparse model, binary relevance calibration, V12.2 Consensus3 weights, and V25 public package. **V29 does not retune any of these pieces.** They are the immutable baseline and the strongest features supplied to the new residual ranker.

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
# The two V22.3 corrections and six V25 core corrections are reconstructed exactly. All eight users are protected from V29 deployment so the new population model cannot erase already public-tested gains.

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
# ## Domain-native user × module features
#
# V29 now adds information that the anchor did not explicitly encode as a joint user-item representation. Everything in this section is derived from organizer files only.
#
# - module-text lexical affinity from the already-fitted V3.1 word/char vectorizers;
# - explicit current/recent module intent;
# - assessment readiness;
# - catalog difficulty and direct prerequisite graph;
# - numerical prerequisite deficits and curriculum adjacency;
# - latest-message template groups used only for dependence-safe validation.

# ---- cell 15 (code) ----
# Module-native lexical / intent / readiness features.
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
    n=len(rows)
    for j,row in enumerate(rows):
        dist=n-1-j; w=float(np.exp(-0.7*dist)); row_topics=set(row.topics)
        for m,mid in enumerate(MODULES):
            hits=sum(tp in row_topics for tp in MODULE_TOPIC_MAP[mid])
            if hits:
                MODULE_COUNT[ui,m]+=hits
                MODULE_RECENCY[ui,m]+=w*hits
                if dist==0: MODULE_LATEST[ui,m]=1.0

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

TRAIN_GROUP=np.asarray(LATEST_GROUP[:N_TRAIN],dtype=object)

# Direct prerequisite edges grounded in modules_catalog.csv.
# Edge (p, m) means module p is a prerequisite of module m.
PREREQ_EDGES = [
    ("M_002","M_004"),
    ("M_002","M_007"), ("M_006","M_007"),
    ("M_003","M_008"), ("M_001","M_008"),
    ("M_007","M_009"), ("M_006","M_009"),
    ("M_009","M_010"),
    ("M_009","M_011"),
    ("M_011","M_012"),
    ("M_009","M_016"), ("M_002","M_016"), ("M_005","M_016"),
]
PREREQ_MAT=np.zeros((N_MODULES,N_MODULES),np.float32)
for p,m in PREREQ_EDGES:
    PREREQ_MAT[MODULES.index(p),MODULES.index(m)]=1.0

# Sanity-check the hand-parsed graph against the actual organizer prerequisite strings.
prtxt=modules.set_index("module_id")["prerequisite_level"].astype(str).to_dict()
assert "Python" in prtxt["M_004"]
assert "Python" in prtxt["M_007"] and "Stat" in prtxt["M_007"]
assert "SQL" in prtxt["M_008"] and "Excel" in prtxt["M_008"]
assert "EDA" in prtxt["M_009"] and "Stat" in prtxt["M_009"]
assert "ML" in prtxt["M_010"] and "ML" in prtxt["M_011"]
assert "NLP" in prtxt["M_012"] and "ML" in prtxt["M_016"]

# Explicit assessed prerequisite skills. We do not invent proxies for prerequisites that are absent from assessments.
PREREQ_SKILL_INDEX = {
    "M_004":[0],               # Python
    "M_007":[0,2],             # Python, Statistics
    "M_008":[1],               # SQL; Excel is not assessed
    "M_009":[2,3],             # Statistics, EDA
    "M_010":[4,6],             # ML, Deep Learning
    "M_011":[4],               # ML
    "M_012":[],                # NLP is not assessed directly
    "M_016":[0,4],             # Python, ML; Version Control is not assessed directly
}
UNOBSERVED_PREREQ_COUNT = {"M_008":1, "M_012":1, "M_016":1}

def module_level(mid):
    s=prtxt[mid].lower()
    if "pemula" in s: return 0
    if "menengah" in s: return 1
    if "lanjutan" in s: return 2
    if "ahli" in s: return 3
    return 4  # semua level

MODULE_LEVEL=np.asarray([module_level(m) for m in MODULES],np.int16)
LEVEL_TARGET=np.asarray([1.0,2.4,3.4,4.1,2.5],np.float32)
MODULE_LEVEL_ONEHOT=np.eye(5,dtype=np.float32)[MODULE_LEVEL]

PREREQ_GAP_MEAN=np.zeros((len(all_users),N_MODULES),np.float32)
PREREQ_GAP_MAX=np.zeros_like(PREREQ_GAP_MEAN)
PREREQ_UNMET_FRAC=np.zeros_like(PREREQ_GAP_MEAN)
PREREQ_OBS_COUNT=np.zeros(N_MODULES,np.float32)
PREREQ_UNOBS_COUNT=np.asarray([UNOBSERVED_PREREQ_COUNT.get(m,0) for m in MODULES],np.float32)
for m,mid in enumerate(MODULES):
    idx=PREREQ_SKILL_INDEX.get(mid,[])
    PREREQ_OBS_COUNT[m]=len(idx)
    if not idx: continue
    req=float(LEVEL_TARGET[MODULE_LEVEL[m]])
    gaps=np.maximum(0.0, req-A[:,idx])
    PREREQ_GAP_MEAN[:,m]=gaps.mean(1)
    PREREQ_GAP_MAX[:,m]=gaps.max(1)
    PREREQ_UNMET_FRAC[:,m]=(gaps>0.25).mean(1)

OVERALL_SKILL=A.mean(1).astype(np.float32)
LEVEL_GAP_SIGNED=(LEVEL_TARGET[MODULE_LEVEL][None,:]-OVERALL_SKILL[:,None]).astype(np.float32)
LEVEL_GAP_ABS=np.abs(LEVEL_GAP_SIGNED).astype(np.float32)
BEGINNER_FIT=np.mean(A<=1.0,axis=1).astype(np.float32)

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
    return list(sg.split(np.zeros(len(yy)),user_strata_from_y(yy),groups=groups))


def make_group_folds_global(rows,n_splits,seed):
    rows=np.asarray(rows,int)
    local=make_group_folds_local(y[rows],TRAIN_GROUP[rows],n_splits,seed)
    return [(rows[tr],rows[va]) for tr,va in local]

print("catalog lexical matrices:",LEX_WORD.shape,LEX_CHAR.shape)
print("latest template groups:",len(np.unique(TRAIN_GROUP)))
print("curriculum edges:",len(PREREQ_EDGES))
print("ordinal grade counts:",dict(zip(*np.unique(ORDINAL_GRADES,return_counts=True))))

# ---- cell 16 (markdown) ----
# ## Fold-safe label and residual graphs
#
# The graph expert is where V29 gets genuinely new information. A validation/test label is never used to construct its graph features.
#
# **Label graph:** smoothed `P(module m relevant | seed module j relevant)` and lift, with support shrinkage.  
# **Residual graph:** systematic `Y - anchor` response conditional on anchor Top-5 membership, plus residual-correlation propagation.
#
# For ranker-training users these statistics are generated by an inner template-group cross-fit. External validation/test rows use a graph fit only on the corresponding training partition.

# ---- cell 17 (code) ----
RANK_WEIGHTS=np.asarray([1.00,0.80,0.60,0.45,0.35],np.float32)


def topk_seed_weights(anchor,k=5):
    anchor=np.asarray(anchor,np.float32); n=len(anchor)
    order=np.argsort(-anchor,axis=1,kind="mergesort")[:,:k]
    W=np.zeros_like(anchor,np.float32)
    rr=np.arange(n)[:,None]
    W[rr,order]=RANK_WEIGHTS[:k][None,:]
    W/=np.maximum(W.sum(1,keepdims=True),1e-8)
    return W


def fit_graph_bundle(y_fit,anchor_fit):
    y_fit=np.asarray(y_fit,np.float32); anchor_fit=np.asarray(anchor_fit,np.float32)
    n=len(y_fit); b=(y_fit>0).astype(np.float32)
    counts=b.sum(0)
    pair=b.T@b
    base=(counts+2.0)/(n+4.0)
    cond=(pair+2.0)/(counts[:,None]+4.0)
    lift=cond/np.maximum(base[None,:],1e-6)
    shrink=np.minimum(1.0,counts[:,None]/50.0)*np.minimum(1.0,pair/20.0)
    log_lift=np.log(np.clip(lift,0.25,4.0))*shrink
    cond_delta=(cond-base[None,:])*shrink
    support_norm=np.minimum(1.0,pair/25.0)
    for M in (log_lift,cond_delta,support_norm):
        np.fill_diagonal(M,0.0)

    residual=(y_fit-anchor_fit).astype(np.float32)
    global_resid=residual.mean(0).astype(np.float32)
    topmask=np.zeros_like(anchor_fit,np.float32)
    order=np.argsort(-anchor_fit,axis=1,kind="mergesort")[:,:5]
    topmask[np.arange(n)[:,None],order]=1.0
    seed_support=topmask.sum(0)
    means=(topmask.T@residual)/np.maximum(seed_support[:,None],1.0)
    rshrink=(seed_support/(seed_support+50.0))[:,None]
    resid_response=((means-global_resid[None,:])*rshrink).astype(np.float32)
    np.fill_diagonal(resid_response,0.0)

    if n>=3:
        corr=np.corrcoef(residual,rowvar=False)
        corr=np.nan_to_num(corr,nan=0.0,posinf=0.0,neginf=0.0).astype(np.float32)
    else:
        corr=np.zeros((N_MODULES,N_MODULES),np.float32)
    np.fill_diagonal(corr,0.0)
    anchor_mean=anchor_fit.mean(0).astype(np.float32)
    anchor_std=np.maximum(anchor_fit.std(0),1e-4).astype(np.float32)

    return {
        "log_lift":log_lift.astype(np.float32),
        "cond_delta":cond_delta.astype(np.float32),
        "support_norm":support_norm.astype(np.float32),
        "cond":cond.astype(np.float32),
        "lift":lift.astype(np.float32),
        "pair_counts":pair.astype(np.float32),
        "resid_response":resid_response,
        "global_resid":global_resid,
        "resid_corr":corr,
        "anchor_mean":anchor_mean,
        "anchor_std":anchor_std,
    }


def apply_graph_bundle(bundle,anchor):
    anchor=np.asarray(anchor,np.float32)
    W=topk_seed_weights(anchor)
    z=(anchor-bundle["anchor_mean"][None,:])/bundle["anchor_std"][None,:]
    return {
        "COOC_LOG_LIFT":(W@bundle["log_lift"]).astype(np.float32),
        "COOC_COND_DELTA":(W@bundle["cond_delta"]).astype(np.float32),
        "COOC_SUPPORT":(W@bundle["support_norm"]).astype(np.float32),
        "RESID_RESPONSE":(bundle["global_resid"][None,:]+W@bundle["resid_response"]).astype(np.float32),
        "RESID_COV":((z@bundle["resid_corr"])/max(N_MODULES-1,1)).astype(np.float32),
    }


def crossfit_graph_features(y_local,anchor_local,groups_local,seed):
    y_local=np.asarray(y_local,np.float32); anchor_local=np.asarray(anchor_local,np.float32)
    groups_local=np.asarray(groups_local,dtype=object)
    keys=["COOC_LOG_LIFT","COOC_COND_DELTA","COOC_SUPPORT","RESID_RESPONSE","RESID_COV"]
    out={k:np.zeros_like(anchor_local,np.float32) for k in keys}
    folds=make_group_folds_local(y_local,groups_local,GRAPH_INNER_FOLDS,seed)
    for f,(tr,va) in enumerate(folds):
        bundle=fit_graph_bundle(y_local[tr],anchor_local[tr])
        vv=apply_graph_bundle(bundle,anchor_local[va])
        for k in keys: out[k][va]=vv[k]
    for k in keys:
        if not np.isfinite(out[k]).all(): raise ValueError(f"non-finite graph feature {k}")
    return out


def oracle_cooc_support(bundle,anchor_val,y_val):
    anchor_val=np.asarray(anchor_val); y_val=np.asarray(y_val)
    criteria=(bundle["pair_counts"]>=ORACLE_PAIR_SUPPORT)&(bundle["cond"]>=ORACLE_MIN_COND)&(bundle["lift"]>=ORACLE_MIN_LIFT)
    np.fill_diagonal(criteria,False)
    top=top5_indices(anchor_val)
    support=np.zeros_like(y_val,dtype=bool)
    for i in range(len(anchor_val)):
        seeds=[int(j) for j in top[i] if y_val[i,int(j)]>0]
        if seeds:
            support[i]=np.any(criteria[seeds],axis=0)
    return support

print("graph helpers ready")

# ---- cell 18 (markdown) ----
# ## Long-format curriculum-aware LambdaRank
#
# Each user becomes exactly 17 contiguous rows. The anchor score/rank/margins remain the strongest features. The four preregistered ablations isolate the source of any improvement:
#
# 1. `ltr_base`: existing anchor + lexical/assessment/module features;
# 2. `ltr_gap`: + explicit skill-gap and curriculum adjacency;
# 3. `ltr_gap_cooc`: + fold-safe label co-occurrence;
# 4. `ltr_gap_cooc_resid`: + residual response/correlation graph.
#
# The ranker uses ordinal grades `0..6` and a gain vector approximating the competition's relevance bands. Its raw scores are never deployed directly; they are affine-aligned to the anchor within each user and used only as a small residual correction.

# ---- cell 19 (code) ----
def curriculum_support(rows_global,anchor):
    rows_global=np.asarray(rows_global,int); anchor=np.asarray(anchor,np.float32)
    W=topk_seed_weights(anchor)
    anchor_prereq=W@PREREQ_MAT.T      # candidate is prerequisite of anchor seed
    anchor_next=W@PREREQ_MAT          # candidate is next step after anchor seed
    iraw=(MODULE_RECENCY[rows_global]+2.0*MODULE_LATEST[rows_global]).astype(np.float32)
    isum=iraw.sum(1,keepdims=True)
    IW=np.divide(iraw,np.maximum(isum,1e-8),out=np.zeros_like(iraw),where=isum>0)
    intent_prereq=IW@PREREQ_MAT.T
    intent_next=IW@PREREQ_MAT
    return anchor_prereq.astype(np.float32),anchor_next.astype(np.float32),intent_prereq.astype(np.float32),intent_next.astype(np.float32)


def rank_matrix(scores):
    order=np.argsort(-np.asarray(scores),axis=1,kind="mergesort")
    ranks=np.empty_like(order,dtype=np.int16)
    ranks[np.arange(len(order))[:,None],order]=np.arange(1,N_MODULES+1,dtype=np.int16)[None,:]
    return ranks


def top5_order_change_rate(a,b):
    return float(np.mean(np.any(top5_indices(a)!=top5_indices(b),axis=1)))


def build_long_matrix(rows_global,anchor,base,meta,graph):
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

    # Base user-item block.
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

    # User structure repeated for each item.
    pieces.append(np.repeat(USER_FEATURES[rows_global],N_MODULES,axis=0).astype(np.float32))
    names += [f"user_f{i:03d}" for i in range(USER_FEATURES.shape[1])]
    pieces.append(np.tile(np.eye(N_MODULES,dtype=np.float32),(n,1)))
    names += [f"module_{m}" for m in MODULES]
    pieces.append(np.tile(MODULE_LEVEL_ONEHOT,(n,1)).astype(np.float32))
    names += [f"module_level_{i}" for i in range(5)]
    bounds={"base":sum(p.shape[1] for p in pieces)}

    # Explicit prerequisite deficit + curriculum adjacency block.
    ap,an,ip,inn=curriculum_support(rows_global,anchor)
    obs=np.tile(PREREQ_OBS_COUNT[None,:],(n,1))
    unobs=np.tile(PREREQ_UNOBS_COUNT[None,:],(n,1))
    bfit=np.tile(BEGINNER_FIT[rows_global,None],(1,N_MODULES))
    gap_mats=[
        PREREQ_GAP_MEAN[rows_global],PREREQ_GAP_MAX[rows_global],PREREQ_UNMET_FRAC[rows_global],
        LEVEL_GAP_SIGNED[rows_global],LEVEL_GAP_ABS[rows_global],obs,unobs,bfit,ap,an,ip,inn,
    ]
    gap_names=[
        "prereq_gap_mean","prereq_gap_max","prereq_unmet_frac","level_gap_signed","level_gap_abs",
        "observed_prereq_count","unobserved_prereq_count","beginner_fit",
        "anchor_prereq_support","anchor_nextstep_support","intent_prereq_support","intent_nextstep_support",
    ]
    pieces.append(np.column_stack([m.reshape(-1) for m in gap_mats]).astype(np.float32)); names+=gap_names
    bounds["gap"]=sum(p.shape[1] for p in pieces)

    cooc_mats=[graph["COOC_LOG_LIFT"],graph["COOC_COND_DELTA"],graph["COOC_SUPPORT"]]
    pieces.append(np.column_stack([m.reshape(-1) for m in cooc_mats]).astype(np.float32))
    names += ["cooc_log_lift","cooc_cond_delta","cooc_support"]
    bounds["cooc"]=sum(p.shape[1] for p in pieces)

    resid_mats=[graph["RESID_RESPONSE"],graph["RESID_COV"]]
    pieces.append(np.column_stack([m.reshape(-1) for m in resid_mats]).astype(np.float32))
    names += ["resid_response","resid_cov"]
    bounds["resid"]=sum(p.shape[1] for p in pieces)

    XX=np.hstack(pieces).astype(np.float32,copy=False)
    if XX.shape[0] != n*N_MODULES: raise AssertionError(XX.shape)
    if len(names)!=XX.shape[1]: raise AssertionError((len(names),XX.shape))
    if not np.isfinite(XX).all(): raise ValueError("non-finite long features")
    return XX,names,bounds

ABLATION_BOUND={
    "ltr_base":"base",
    "ltr_gap":"gap",
    "ltr_gap_cooc":"cooc",
    "ltr_gap_cooc_resid":"resid",
}


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


def blend_residual(anchor,raw_ranker,alpha):
    aligned=align_ranker_to_anchor(raw_ranker,anchor)
    return (anchor+float(alpha)*(aligned-anchor)).astype(np.float32)

print("long-format + LambdaRank helpers ready")

# ---- cell 20 (markdown) ----
# ## Fully nested template-group OOF: ablation selection + oracle headroom
#
# This is the only world allowed to **select** the V29 feature ablation and residual alpha. The outer validation users are template-group disjoint from their ranker/graph/anchor training partition. The supervised anchor is rebuilt inside every outer fold.
#
# The same folds also create a fold-safe co-occurrence support mask used only for the worst-100 oracle-headroom diagnostic.

# ---- cell 21 (code) ----
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


def cluster_boot_ci(values,groups,seed,B=BOOTSTRAPS):
    values=np.asarray(values,float); groups=np.asarray(groups,dtype=object)
    ug=np.unique(groups); by={g:np.flatnonzero(groups==g) for g in ug}
    rng=np.random.default_rng(int(seed)); z=np.empty(int(B),float)
    for b in range(int(B)):
        sampled=rng.choice(ug,size=len(ug),replace=True)
        idx=np.concatenate([by[g] for g in sampled])
        z[b]=values[idx].mean()
    return tuple(float(x) for x in np.quantile(z,[.025,.975]))


def order_change_rate(a,b):
    return float(np.mean(np.any(top5_indices(a)!=top5_indices(b),axis=1)))

STRICT_OUTER=make_group_folds_global(np.arange(N_TRAIN),N_FOLDS,SEED+29001)
STRICT_ANCHOR=np.zeros_like(y,np.float32)
STRICT_RAW={a:np.zeros_like(y,np.float32) for a in ABLATION_ORDER}
STRICT_FOLD_ID=np.full(N_TRAIN,-1,np.int16)
STRICT_ORACLE_COOC=np.zeros_like(y,dtype=bool)

for f,(tr,va) in enumerate(STRICT_OUTER):
    print(f"strict fold {f+1}/{len(STRICT_OUTER)} train={len(tr)} val={len(va)}")
    anc=build_anchor_split_full(tr,va,SEED+29100+f*101)
    STRICT_ANCHOR[va]=anc["cons_val"]
    STRICT_FOLD_ID[va]=f

    graph_tr=crossfit_graph_features(y[tr],anc["cons_train"],TRAIN_GROUP[tr],SEED+29200+f*107)
    bundle=fit_graph_bundle(y[tr],anc["cons_train"])
    graph_va=apply_graph_bundle(bundle,anc["cons_val"])
    STRICT_ORACLE_COOC[va]=oracle_cooc_support(bundle,anc["cons_val"],y[va])

    Xtr,names,bounds=build_long_matrix(tr,anc["cons_train"],anc["base_train"],anc["meta_train"],graph_tr)
    Xva,_,bounds_va=build_long_matrix(va,anc["cons_val"],anc["base_val"],anc["meta_val"],graph_va)
    assert bounds==bounds_va
    grades=ORDINAL_GRADES[tr]
    for ai,abl in enumerate(ABLATION_ORDER):
        end=bounds[ABLATION_BOUND[abl]]
        model,pred=fit_ranker_predict(Xtr[:,:end],Xva[:,:end],grades,len(tr),SEED+29300+f*131+ai*17)
        STRICT_RAW[abl][va]=pred
        del model
    del Xtr,Xva,graph_tr,graph_va,bundle,anc
    gc.collect()

assert (STRICT_FOLD_ID>=0).all() and np.isfinite(STRICT_ANCHOR).all()
print("strict anchor NDCG",round(ndcg5(y,STRICT_ANCHOR),6))

ab_rows=[]
for abl in ABLATION_ORDER:
    for alpha in RESIDUAL_ALPHAS:
        ss=blend_residual(STRICT_ANCHOR,STRICT_RAW[abl],alpha)
        delta=per_user_ndcg(y,ss)-per_user_ndcg(y,STRICT_ANCHOR)
        fold_means=[float(delta[STRICT_FOLD_ID==f].mean()) for f in range(len(STRICT_OUTER))]
        ci=cluster_boot_ci(delta,TRAIN_GROUP,SEED+29400+int(alpha*1000)+ABLATION_ORDER.index(abl)*41)
        row={
            "ablation":abl,"alpha":float(alpha),"mean_delta":float(delta.mean()),
            "fold_wins":int(np.sum(np.asarray(fold_means)>0)),"min_fold":float(np.min(fold_means)),
            "median_fold":float(np.median(fold_means)),"ci_low":ci[0],"ci_high":ci[1],
            "top5_membership_change":top5_set_change_rate(STRICT_ANCHOR,ss),
            "top1_change":top1_change_rate(STRICT_ANCHOR,ss),
            "order_change":order_change_rate(STRICT_ANCHOR,ss),
        }
        row["feasible"]=bool(
            row["mean_delta"]>MIN_STRICT_MEAN and row["fold_wins"]>=MIN_STRICT_FOLD_WINS and
            row["ci_low"]>=MIN_STRICT_CI_LOW and row["top5_membership_change"]<=MAX_TEST_TOP5_MEMBERSHIP_CHANGE and
            row["top1_change"]<=MAX_TEST_TOP1_CHANGE and row["order_change"]<=MAX_TEST_ORDER_CHANGE
        )
        row["robust_score"]=float(row["mean_delta"]+0.35*min(row["min_fold"],0.0)+0.10*min(row["ci_low"],0.0))
        ab_rows.append(row)
V29_ABLATION=pd.DataFrame(ab_rows)
display(V29_ABLATION.sort_values(["feasible","robust_score"],ascending=[False,False]))

feas=V29_ABLATION[V29_ABLATION.feasible].copy()
if len(feas):
    SELECTED_ROW=feas.sort_values(["robust_score","mean_delta"],ascending=False).iloc[0]
    SELECTION_FEASIBLE=True
else:
    SELECTED_ROW=V29_ABLATION.sort_values(["robust_score","mean_delta"],ascending=False).iloc[0]
    SELECTION_FEASIBLE=False
SELECTED_ABLATION=str(SELECTED_ROW.ablation); SELECTED_ALPHA=float(SELECTED_ROW.alpha)
print("SELECTED",SELECTED_ABLATION,"alpha",SELECTED_ALPHA,"strict feasible",SELECTION_FEASIBLE)

# -------- Oracle headroom on the 100 worst strict-anchor users --------
def curriculum_oracle_support(rows_global,anchor):
    rows_global=np.asarray(rows_global,int); anchor=np.asarray(anchor)
    out=np.zeros((len(rows_global),N_MODULES),dtype=bool)
    top1=np.argmax(anchor,axis=1)
    for ii,grow in enumerate(rows_global):
        intent=np.flatnonzero(MODULE_RECENCY[grow]>0)
        seeds=intent.tolist() if len(intent) else [int(top1[ii])]
        for s in seeds:
            out[ii] |= (PREREQ_MAT[:,s]>0) | (PREREQ_MAT[s,:]>0)
    return out

STRICT_CURR_ORACLE=curriculum_oracle_support(np.arange(N_TRAIN),STRICT_ANCHOR)
anchor_user_ndcg=per_user_ndcg(y,STRICT_ANCHOR)
worst_n=min(ORACLE_WORST_N,N_TRAIN)
WORST_IDX=np.argsort(anchor_user_ndcg)[:worst_n]
pred_top=top5_indices(STRICT_ANCHOR)
pred_mask=np.zeros_like(y,dtype=bool); pred_mask[np.arange(N_TRAIN)[:,None],pred_top]=True
missed=(y>0)&(~pred_mask)
cooc_cap=missed&STRICT_ORACLE_COOC
curr_cap=missed&STRICT_CURR_ORACLE
union_cap=cooc_cap|curr_cap

def mass(mask,rows): return float((y[rows]*mask[rows]).sum())
miss_mass=mass(missed,WORST_IDX)
cooc_mass=mass(cooc_cap,WORST_IDX); curr_mass=mass(curr_cap,WORST_IDX); union_mass=mass(union_cap,WORST_IDX)
ORACLE_HEADROOM={
    "worst_users":int(worst_n),"missed_relevance_mass":miss_mass,
    "cooc_captured_mass":cooc_mass,"prereq_captured_mass":curr_mass,"union_captured_mass":union_mass,
    "cooc_ratio":cooc_mass/max(miss_mass,1e-12),"prereq_ratio":curr_mass/max(miss_mass,1e-12),
    "union_ratio":union_mass/max(miss_mass,1e-12),
}
ORACLE_HEADROOM["strength"]="strong" if ORACLE_HEADROOM["union_ratio"]>=0.50 else ("weak" if ORACLE_HEADROOM["union_ratio"]<0.15 else "moderate")
print("ORACLE HEADROOM",ORACLE_HEADROOM)

module_name=modules.set_index("module_id")["module_name"].to_dict()
def mids_from_mask(maskrow): return [MODULES[j] for j in np.flatnonzero(maskrow)]
def mids_top(row_scores,k=5): return [MODULES[j] for j in np.argsort(-row_scores,kind="mergesort")[:k]]
def truth_top(row_y,k=6):
    pos=np.flatnonzero(row_y>0); pos=pos[np.argsort(-row_y[pos],kind="mergesort")]
    return [MODULES[j] for j in pos[:k]]
def latest_learning_text(uid):
    g=user_chat.get(uid)
    if g is None or len(g)==0: return ""
    learn=g[~g["is_noise"]]
    if len(learn)==0: learn=g
    return str(learn.iloc[-1]["user_chat_text"]) if len(learn) else ""

audit_rows=[]
for i in WORST_IDX:
    uid=train.iloc[i].user_id
    audit_rows.append({
        "user_id":uid,"strict_anchor_ndcg":float(anchor_user_ndcg[i]),
        "pred_top5":" > ".join(mids_top(STRICT_ANCHOR[i],5)),
        "truth_positive_order":" > ".join(truth_top(y[i],6)),
        "missed_positive":"|".join(mids_from_mask(missed[i])),
        "cooc_supported_missed":"|".join(mids_from_mask(cooc_cap[i])),
        "prereq_supported_missed":"|".join(mids_from_mask(curr_cap[i])),
        "latest_learning_chat":latest_learning_text(uid),
        "has_learning_chat":int(bool(latest_learning_text(uid))),
        "assessment_mean":float(A[i].mean()),
        "assessment_values":"|".join(f"{SKILL_NAMES[j]}={A[i,j]:.1f}" for j in range(10)),
        "manual_error_type":"",
    })
V29_WORST100=pd.DataFrame(audit_rows)
display(V29_WORST100.head(12))

# ---- cell 22 (markdown) ----
# ## Frozen-policy rejection audits
#
# After strict OOF chooses one ablation and one residual coefficient, the policy is frozen. The following worlds may **reject** it but may not choose a different model:
#
# - ordinary/public-compatible OOF audit;
# - fresh salted template-disjoint LOCK;
# - chronological U0001–3000 → U3001–4000;
# - pseudo-public subsamples of the LOCK deltas.
#
# This prevents V29 from turning the fresh audits into another hyperparameter search surface.

# ---- cell 23 (code) ----
def feature_end(bounds,abl): return bounds[ABLATION_BOUND[abl]]


def evaluate_reference_ordinary(abl,alpha):
    raw=np.zeros_like(y,np.float32)
    for f,(tr,va) in enumerate(ref_folds):
        graph_tr=crossfit_graph_features(y[tr],consensus3_oof[tr],TRAIN_GROUP[tr],SEED+29500+f*101)
        bundle=fit_graph_bundle(y[tr],consensus3_oof[tr]); graph_va=apply_graph_bundle(bundle,consensus3_oof[va])
        Xtr,names,bounds=build_long_matrix(tr,consensus3_oof[tr],base_oof[tr],meta_oof[tr],graph_tr)
        Xva,_,_=build_long_matrix(va,consensus3_oof[va],base_oof[va],meta_oof[va],graph_va)
        end=feature_end(bounds,abl)
        model,pred=fit_ranker_predict(Xtr[:,:end],Xva[:,:end],ORDINAL_GRADES[tr],len(tr),SEED+29600+f*113)
        raw[va]=pred
        del model,Xtr,Xva,graph_tr,graph_va,bundle; gc.collect()
    score=blend_residual(consensus3_oof,raw,alpha)
    delta=per_user_ndcg(y,score)-per_user_ndcg(y,consensus3_oof)
    return score,delta


def evaluate_external_split(train_rows,val_rows,abl,alpha,seed):
    train_rows=np.asarray(train_rows,int); val_rows=np.asarray(val_rows,int)
    anc=build_anchor_split_full(train_rows,val_rows,seed)
    graph_tr=crossfit_graph_features(y[train_rows],anc["cons_train"],TRAIN_GROUP[train_rows],seed+101)
    bundle=fit_graph_bundle(y[train_rows],anc["cons_train"]); graph_va=apply_graph_bundle(bundle,anc["cons_val"])
    Xtr,names,bounds=build_long_matrix(train_rows,anc["cons_train"],anc["base_train"],anc["meta_train"],graph_tr)
    Xva,_,_=build_long_matrix(val_rows,anc["cons_val"],anc["base_val"],anc["meta_val"],graph_va)
    end=feature_end(bounds,abl)
    model,pred=fit_ranker_predict(Xtr[:,:end],Xva[:,:end],ORDINAL_GRADES[train_rows],len(train_rows),seed+202)
    score=blend_residual(anc["cons_val"],pred,alpha)
    delta=per_user_ndcg(y[val_rows],score)-per_user_ndcg(y[val_rows],anc["cons_val"])
    out={"anchor":anc["cons_val"],"score":score,"delta":delta,"model":model,"feature_names":names[:end]}
    del Xtr,Xva,graph_tr,graph_va,bundle,anc; gc.collect()
    return out

# Ordinary reference audit (diagnostic only; selection never reads it).
ORD_SCORE,ORD_DELTA=evaluate_reference_ordinary(SELECTED_ABLATION,SELECTED_ALPHA)

# Fresh template-group LOCK.
def lock_bucket(g):
    h=hashlib.sha1(f"V29-CURRICULUM-LTR-LOCK-20260828::{g}".encode()).hexdigest()
    return int(h[:12],16)%5
LOCK_MASK=np.asarray([lock_bucket(g)==0 for g in TRAIN_GROUP],bool)
DEV_IDX=np.flatnonzero(~LOCK_MASK); LOCK_IDX=np.flatnonzero(LOCK_MASK)
assert len(set(TRAIN_GROUP[DEV_IDX]).intersection(set(TRAIN_GROUP[LOCK_IDX])))==0
LOCK_RES=evaluate_external_split(DEV_IDX,LOCK_IDX,SELECTED_ABLATION,SELECTED_ALPHA,SEED+29700)

# Chronological audit.
CHRONO_TR=np.arange(0,3000,dtype=int); CHRONO_VA=np.arange(3000,4000,dtype=int)
CHRONO_RES=evaluate_external_split(CHRONO_TR,CHRONO_VA,SELECTED_ABLATION,SELECTED_ALPHA,SEED+29800)

STRICT_SCORE=blend_residual(STRICT_ANCHOR,STRICT_RAW[SELECTED_ABLATION],SELECTED_ALPHA)
STRICT_DELTA=per_user_ndcg(y,STRICT_SCORE)-per_user_ndcg(y,STRICT_ANCHOR)
STRICT_FOLD_MEANS=[float(STRICT_DELTA[STRICT_FOLD_ID==f].mean()) for f in range(len(STRICT_OUTER))]
STRICT_CI=cluster_boot_ci(STRICT_DELTA,TRAIN_GROUP,SEED+29850)

validation_rows=[
    {"world":"strict_template_group","n":N_TRAIN,"anchor_ndcg":ndcg5(y,STRICT_ANCHOR),"candidate_ndcg":ndcg5(y,STRICT_SCORE),"delta":float(STRICT_DELTA.mean()),"fold_wins":int(np.sum(np.asarray(STRICT_FOLD_MEANS)>0)),"ci_low":STRICT_CI[0],"ci_high":STRICT_CI[1]},
    {"world":"ordinary_reference","n":N_TRAIN,"anchor_ndcg":ndcg5(y,consensus3_oof),"candidate_ndcg":ndcg5(y,ORD_SCORE),"delta":float(ORD_DELTA.mean()),"fold_wins":np.nan,"ci_low":np.nan,"ci_high":np.nan},
    {"world":"fresh_lock","n":len(LOCK_IDX),"anchor_ndcg":ndcg5(y[LOCK_IDX],LOCK_RES["anchor"]),"candidate_ndcg":ndcg5(y[LOCK_IDX],LOCK_RES["score"]),"delta":float(LOCK_RES["delta"].mean()),"fold_wins":np.nan,"ci_low":np.nan,"ci_high":np.nan},
    {"world":"chronological","n":len(CHRONO_VA),"anchor_ndcg":ndcg5(y[CHRONO_VA],CHRONO_RES["anchor"]),"candidate_ndcg":ndcg5(y[CHRONO_VA],CHRONO_RES["score"]),"delta":float(CHRONO_RES["delta"].mean()),"fold_wins":np.nan,"ci_low":np.nan,"ci_high":np.nan},
]
V29_VALIDATION=pd.DataFrame(validation_rows)
display(V29_VALIDATION)

# Pseudo-public masks are a stress test of the frozen LOCK deltas, not another selector.
pseudo=[]; rng=np.random.default_rng(SEED+29900); ld=np.asarray(LOCK_RES["delta"],float)
for frac in [0.25,0.35,0.50]:
    vals=[]; k=max(1,int(round(frac*len(ld))))
    for b in range(PSEUDO_PUBLIC_MASKS):
        idx=rng.choice(len(ld),size=k,replace=False); vals.append(float(ld[idx].mean()))
    vals=np.asarray(vals)
    pseudo.append({"fraction":frac,"masks":len(vals),"positive_rate":float(np.mean(vals>0)),"mean":float(vals.mean()),"p10":float(np.quantile(vals,.10)),"median":float(np.median(vals)),"p90":float(np.quantile(vals,.90))})
V29_PSEUDO=pd.DataFrame(pseudo)
display(V29_PSEUDO)

p35=V29_PSEUDO.iloc[np.argmin(np.abs(V29_PSEUDO.fraction.to_numpy()-0.35))]
PREDEPLOY_GATE=bool(
    SELECTION_FEASIBLE and
    float(STRICT_DELTA.mean())>MIN_STRICT_MEAN and int(np.sum(np.asarray(STRICT_FOLD_MEANS)>0))>=MIN_STRICT_FOLD_WINS and STRICT_CI[0]>=MIN_STRICT_CI_LOW and
    float(ORD_DELTA.mean())>=MIN_ORDINARY_AUDIT and
    float(LOCK_RES["delta"].mean())>MIN_LOCK_DELTA and
    float(CHRONO_RES["delta"].mean())>=MIN_CHRONO_DELTA and
    float(p35.positive_rate)>=PSEUDO_PUBLIC_MIN_POS_RATE and float(p35.p10)>=PSEUDO_PUBLIC_MIN_P10
)
print("PREDEPLOY_GATE",PREDEPLOY_GATE)

# ---- cell 24 (markdown) ----
# ## Final full-train fit and protected deployment
#
# The frozen ablation/alpha is now fit on all 4,000 labeled users. Training graph features remain inner-cross-fitted; test graph features come from one graph fit on all 4,000 train labels and OOF anchor residuals.
#
# The final residual scores are applied only to non-protected test users. The eight V25 public-tested rows are overwritten with their exact 0.65919 anchor rows. Deployment geometry is checked before the strict gate can promote the diagnostic candidate to `submission_v29_primary.csv`.

# ---- cell 25 (code) ----
# Full-train graph features: OOF for ranker training, all-train graph for test.
FULL_GRAPH_TR=crossfit_graph_features(y,consensus3_oof,TRAIN_GROUP,SEED+30000)
FULL_BUNDLE=fit_graph_bundle(y,consensus3_oof)
FULL_GRAPH_TE=apply_graph_bundle(FULL_BUNDLE,consensus3_test)
XTR_FULL,FULL_NAMES,FULL_BOUNDS=build_long_matrix(np.arange(N_TRAIN),consensus3_oof,base_oof,meta_oof,FULL_GRAPH_TR)
XTE_FULL,_,_=build_long_matrix(np.arange(N_TRAIN,N_TRAIN+N_TEST),consensus3_test,base_test,meta_test,FULL_GRAPH_TE)
FULL_END=feature_end(FULL_BOUNDS,SELECTED_ABLATION)
FINAL_RANKER,TEST_RAW=fit_ranker_predict(XTR_FULL[:,:FULL_END],XTE_FULL[:,:FULL_END],ORDINAL_GRADES,N_TRAIN,SEED+30100)
TEST_BLEND=blend_residual(consensus3_test,TEST_RAW,SELECTED_ALPHA)

# Start from the active V25 public anchor and modify only non-protected rows.
DIAGNOSTIC_TEST=v25_anchor_test.copy()
protected_mask=test.user_id.isin(PROTECTED_TEST_USERS).to_numpy()
DIAGNOSTIC_TEST[~protected_mask]=TEST_BLEND[~protected_mask]
DIAGNOSTIC_TEST[protected_mask]=v25_anchor_test[protected_mask]
assert np.array_equal(DIAGNOSTIC_TEST[protected_mask],v25_anchor_test[protected_mask])

anchor_top=top5_indices(v25_anchor_test); cand_top=top5_indices(DIAGNOSTIC_TEST)
changed=np.any(anchor_top!=cand_top,axis=1)
changed_idx=np.flatnonzero(changed)
changed_users=test.iloc[changed_idx].user_id.tolist()
geom={
    "changed_users":int(changed.sum()),
    "top5_membership_change":top5_set_change_rate(v25_anchor_test,DIAGNOSTIC_TEST),
    "top1_change":top1_change_rate(v25_anchor_test,DIAGNOSTIC_TEST),
    "order_change":order_change_rate(v25_anchor_test,DIAGNOSTIC_TEST),
}
GEOMETRY_GATE=bool(
    MIN_TEST_CHANGED_USERS<=geom["changed_users"]<=MAX_TEST_CHANGED_USERS and
    geom["top5_membership_change"]<=MAX_TEST_TOP5_MEMBERSHIP_CHANGE and
    geom["top1_change"]<=MAX_TEST_TOP1_CHANGE and geom["order_change"]<=MAX_TEST_ORDER_CHANGE
)
STRICT_GATE=bool(PREDEPLOY_GATE and GEOMETRY_GATE)
PRIMARY_POLICY="selected_residual_ltr" if STRICT_GATE else "anchor_fail_closed"
PRIMARY_TEST=DIAGNOSTIC_TEST.copy() if STRICT_GATE else v25_anchor_test.copy()
print("deployment geometry",geom)
print("GEOMETRY_GATE",GEOMETRY_GATE,"STRICT_GATE",STRICT_GATE,"PRIMARY_POLICY",PRIMARY_POLICY)

# Change ledger.
ledger=[]
for i in changed_idx:
    ledger.append({
        "user_id":test.iloc[i].user_id,
        "anchor_top5":" > ".join(MODULES[j] for j in anchor_top[i]),
        "candidate_top5":" > ".join(MODULES[j] for j in cand_top[i]),
        "anchor_top1":MODULES[int(anchor_top[i,0])],"candidate_top1":MODULES[int(cand_top[i,0])],
        "top1_changed":int(anchor_top[i,0]!=cand_top[i,0]),
        "protected":int(protected_mask[i]),
    })
V29_TEST_LEDGER=pd.DataFrame(ledger)

# Module-level entry/exit attribution.
module_changes=[]
for m,mid in enumerate(MODULES):
    ain=np.any(anchor_top==m,axis=1); cin=np.any(cand_top==m,axis=1)
    module_changes.append({"module_id":mid,"entries":int(np.sum((~ain)&cin)),"exits":int(np.sum(ain&(~cin))),"net":int(np.sum(cin)-np.sum(ain))})
V29_MODULE_CHANGES=pd.DataFrame(module_changes)

# Final selected-ranker feature importance for auditability.
imp=FINAL_RANKER.booster_.feature_importance(importance_type="gain")
V29_FEATURE_IMPORTANCE=pd.DataFrame({"feature":FULL_NAMES[:FULL_END],"gain":imp}).sort_values("gain",ascending=False)
display(V29_FEATURE_IMPORTANCE.head(30))
display(V29_TEST_LEDGER.head(30))

del XTR_FULL,XTE_FULL,FULL_GRAPH_TR,FULL_GRAPH_TE; gc.collect()

# ---- cell 26 (markdown) ----
# ## Outputs and submission discipline
#
# Diagnostic files are always written so the run can be audited. **Only `submission_v29_primary.csv` is an automatic submission recommendation.** If `PRIMARY_POLICY == "anchor_fail_closed"`, the primary file is intentionally identical to the 0.65919 V25 anchor and should not consume a Kaggle slot.

# ---- cell 27 (code) ----
OUT_DIR=Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("/mnt/data")
OUT_DIR.mkdir(parents=True,exist_ok=True)


def write_submission(name,scores):
    out=sample.copy(); out.loc[:,MODULES]=np.asarray(scores,np.float32)
    assert out.shape==sample.shape and out.user_id.tolist()==test.user_id.tolist()
    assert np.isfinite(out[MODULES].to_numpy()).all()
    path=OUT_DIR/name; out.to_csv(path,index=False); return path

anchor_path=write_submission("submission_v29_anchor_v25_65919.csv",v25_anchor_test)
diag_path=write_submission("submission_v29_DIAGNOSTIC_selected.csv",DIAGNOSTIC_TEST)
primary_path=write_submission("submission_v29_primary.csv",PRIMARY_TEST)

# Core diagnostics.
V29_ABLATION.to_csv(OUT_DIR/"v29_ablation_summary.csv",index=False)
V29_VALIDATION.to_csv(OUT_DIR/"v29_validation_summary.csv",index=False)
V29_PSEUDO.to_csv(OUT_DIR/"v29_pseudo_public_lock.csv",index=False)
V29_WORST100.to_csv(OUT_DIR/"v29_worst100_oof_audit.csv",index=False)
V29_TEST_LEDGER.to_csv(OUT_DIR/"v29_test_change_ledger.csv",index=False)
V29_MODULE_CHANGES.to_csv(OUT_DIR/"v29_module_change_summary.csv",index=False)
V29_FEATURE_IMPORTANCE.to_csv(OUT_DIR/"v29_feature_importance.csv",index=False)
pd.DataFrame([ORACLE_HEADROOM]).to_csv(OUT_DIR/"v29_oracle_headroom.csv",index=False)

slot_plan=pd.DataFrame([
    {"priority":1,"file":"submission_v29_primary.csv","submit":bool(STRICT_GATE),"reason":"strict cross-world gate passed" if STRICT_GATE else "FAIL CLOSED: identical anchor; do not burn a slot"},
    {"priority":2,"file":"submission_v29_DIAGNOSTIC_selected.csv","submit":False,"reason":"diagnostic only; inspect before any manual override"},
])
slot_plan.to_csv(OUT_DIR/"v29_slot_plan.csv",index=False)

summary={
    "version":"V29 Curriculum-Aware Residual Learning-to-Rank",
    "public_anchor":PUBLIC_CURRENT_PB,"leader_reference":PUBLIC_LEADER_LB,
    "selected_ablation":SELECTED_ABLATION,"selected_alpha":SELECTED_ALPHA,
    "selection_feasible":bool(SELECTION_FEASIBLE),"oracle_headroom":ORACLE_HEADROOM,
    "validation":{r["world"]:float(r["delta"]) for r in validation_rows},
    "strict_fold_means":[float(x) for x in STRICT_FOLD_MEANS],"strict_ci":[float(x) for x in STRICT_CI],
    "predeploy_gate":bool(PREDEPLOY_GATE),"geometry_gate":bool(GEOMETRY_GATE),"strict_gate":bool(STRICT_GATE),
    "primary_policy":PRIMARY_POLICY,"test_geometry":geom,
    "protected_test_users":sorted(PROTECTED_TEST_USERS),
    "outputs":{"anchor":str(anchor_path),"diagnostic":str(diag_path),"primary":str(primary_path)},
}
with open(OUT_DIR/"v29_summary.json","w") as f: json.dump(summary,f,indent=2)

# Hard invariants.
assert set(PROTECTED_TEST_USERS).issubset(set(test.user_id))
assert np.array_equal(PRIMARY_TEST[protected_mask],v25_anchor_test[protected_mask])
if not STRICT_GATE:
    assert np.array_equal(PRIMARY_TEST,v25_anchor_test)

print(json.dumps(summary,indent=2))
print("\nFiles written to",OUT_DIR)
for p in [anchor_path,diag_path,primary_path]: print(" -",p)
if PRIMARY_POLICY=="anchor_fail_closed":
    print("\nWARNING: V29 PRIMARY FAILED CLOSED TO 0.65919 ANCHOR — DO NOT BURN A KAGGLE SLOT.")
else:
    print("\nV29 strict gate passed. Inspect v29_test_change_ledger.csv before submission.")

# ---- cell 28 (markdown) ----
# ## Interpretation contract
#
# After a full Kaggle run, inspect the evidence in this order:
#
# 1. **Oracle headroom** — `union_ratio >= 0.50` means curriculum/co-occurrence can explain a large share of missed relevance mass on the worst anchor users; `< 0.15` is a strong warning that this hypothesis has little coverage.
# 2. **Ablation progression** — the important question is whether skill-gap, co-occurrence, and residual graph add stable incremental value beyond `ltr_base`, not whether LambdaRank alone has a high score.
# 3. **Strict template-group OOF** — this is the only model-selection surface.
# 4. **Fresh LOCK + chronology** — either may veto the frozen policy. Do not retune thresholds after seeing them.
# 5. **Pseudo-public LOCK stress test** — check the 35% positive rate and p10.
# 6. **Deployment geometry** — V29 is intended to make medium-width, small residual changes, not hundreds of aggressive Top-1 reversals.
# 7. **Worst-100 audit** — manually fill `manual_error_type` in `v29_worst100_oof_audit.csv` if the branch needs further diagnosis.
#
# A failed gate is a research result, not a reason to submit the diagnostic file. The notebook deliberately preserves the 0.65919 anchor whenever the new information source does not generalize.
