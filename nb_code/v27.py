# Converted from notebooks/v27.ipynb: markdown cells become comments, %/! magics neutralized.

# ---- cell 0 (markdown) ----
# # MineToday V27 — Selective Upper-Rank Counterfactual Reranker
#
# **Active public anchor:** V25 core-6 package = **0.65919**.  
# **Current leader reference:** **0.66112**.  
# **Metric:** user-level NDCG@5.
#
# V27 is a deliberate break from the V24–V26 one-user micro-probe regime. The public gap is now too large for isolated edits, but the earlier global rerankers were too unstable. This notebook therefore targets the **middle width**: only the two most learnable adjacent ranking decisions, **rank #1↔#2** and **rank #2↔#3**, with one action at most per user.
#
# The model never predicts a new Top-5 set. It only asks a counterfactual question on the current anchor:
#
# > “Would swapping these two adjacent upper-rank modules improve NDCG@5?”
#
# The selection path is intentionally strict:
#
# 1. exact public-validated V3.1 → relevance calibration → V12.2 Consensus3;
# 2. reconstruct the current 0.65919 V25 anchor exactly;
# 3. train **position-specific** counterfactual heads for #1/#2 and #2/#3;
# 4. use template-cluster OOF as the only threshold-selection world;
# 5. audit the frozen coverage on ordinary CV, chronological validation, a deterministic fresh hash LOCK, and pseudo-public masks;
# 6. fail closed to the 0.65919 anchor when generalization evidence is weak.
#
# ### Competition-rule contract
#
# This notebook uses **only the organizer-provided CSV files**. It uses no external dataset, no paid model, no direct LLM/Generative-AI prediction of test labels, and no external ground truth. It is deterministic and records all seeds. The ML models are ordinary Python/scikit-learn/LightGBM models, which is within the Technical Meeting rules allowing multiple machine-learning models and Python solutions.
#
# > **Revision fix:** diagnostic AUC alignment now slices full-length cross-fit predictions to DEV rows before scoring, preventing intentional out-of-fold `NaN` placeholders from reaching `roc_auc_score`.

# ---- cell 1 (code) ----
# Runtime + deterministic configuration
import os, sys, re, json, ast, html, hashlib, random, warnings, time, subprocess
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ["PYTHONHASHSEED"] = "42"

V27_SKIP_INSTALL = bool(int(os.getenv("V27_SKIP_INSTALL", "0")))
try:
    import lightgbm as lgb
except Exception:
    if V27_SKIP_INSTALL:
        raise
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "lightgbm==4.6.0"])
    import lightgbm as lgb

if getattr(lgb, "__version__", "") != "4.6.0" and not V27_SKIP_INSTALL:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "lightgbm==4.6.0"])
    import importlib
    lgb = importlib.reload(lgb)

import numpy as np
import pandas as pd
import scipy
import sklearn
from IPython.display import display
from scipy.sparse import hstack
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

SEED = 42
V27_SMOKE = bool(int(os.getenv("V27_SMOKE", "0")))
V27_STRICT_GEOMETRY = bool(int(os.getenv("V27_STRICT_GEOMETRY", "1")))
N_FOLDS = 2 if V27_SMOKE else 5
RIDGE_ALPHA = 3.0
REFERENCE_W = 0.45
PUBLIC_CONSENSUS3_LB = 0.65856
PUBLIC_PRE_V25_PB = 0.65912
PUBLIC_V25_LB = 0.65919
PUBLIC_CURRENT_PB = PUBLIC_V25_LB
PUBLIC_LEADER_LB = 0.66112
PUBLIC_CAL045_LB = 0.65639
LGB_ESTIMATORS = 40 if V27_SMOKE else 1200
LGB_EARLY_STOP = 8 if V27_SMOKE else 80
ACTION_TREES = 80 if V27_SMOKE else 500
BOOTSTRAPS = 250 if V27_SMOKE else 2500
PSEUDO_PUBLIC_MASKS = 20 if V27_SMOKE else 100
PSEUDO_PUBLIC_FRAC = 0.35
GEOMETRY_ATOL = 2.5e-4

# Medium-width search space. Coverage is chosen ONLY on cluster-safe development OOF.
COVERAGE_GRID = [0.01, 0.02] if V27_SMOKE else [0.005, 0.010, 0.015, 0.020, 0.030, 0.040, 0.050]
MIN_DEV_ACTIONS = 4 if V27_SMOKE else 12
MAX_TEST_HEAD12 = 30
MAX_TEST_HEAD23 = 30
MAX_TEST_COMBINED = 45

# Existing public-validated package is immutable during V27 discovery.
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
print("V27_SMOKE", V27_SMOKE, "leader gap", round(PUBLIC_LEADER_LB-PUBLIC_CURRENT_PB, 5))
try:
    print("GPU runtime:\n" + subprocess.check_output(["nvidia-smi", "-L"], text=True).strip())
except Exception:
    print("nvidia-smi unavailable; V27 is T4x2-compatible but intentionally CPU-first.")

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
# ## Exact public-validated anchor family
#
# The next cells intentionally preserve the V3.1/V7/V12.2 public-winning mechanics. V27 does **not** retune the sparse recommender or the module calibration weights before learning upper-rank corrections. This keeps the residual problem deployment-aligned.

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
# ## Reconstruct the current 0.65919 anchor
#
# V27 inherits the two-action V22.3 package and the six-action V25 core package exactly. The eight users are **protected** from new V27 actions so the experiment measures genuinely new upper-rank corrections instead of rewriting already public-tested decisions.

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
# ## Dataset-native upper-rank evidence
#
# No new external representation is introduced. V27 derives every feature from the organizer data:
#
# - frozen Consensus3/V3.1/relevance-calibration geometry;
# - current/recent/full lexical intent through the existing sparse representation;
# - label-free TF-IDF similarity to the organizer module catalog;
# - explicit topic recency;
# - organizer assessment readiness and prerequisite structure;
# - fold-safe empirical-Bayes pair and intent-conditioned pair priors.
#
# The pair priors are **leave-one-out on model-training rows** and use only the outer-training partition for validation rows. They therefore cannot copy a row's target into its own features.

# ---- cell 15 (code) ----
# Module-native features from modules_catalog.csv and chat/assessment data only.
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
    "M_001":["excel"],
    "M_002":["python"],
    "M_003":["sql"],
    "M_004":["webscrape","python"],
    "M_005":["git"],
    "M_006":["statistics"],
    "M_007":["eda","python","statistics"],
    "M_008":["dashboard","sql","excel"],
    "M_009":["ml","ml_eval","eda","statistics"],
    "M_010":["cv","ml"],
    "M_011":["nlp","ml"],
    "M_012":["genai","nlp"],
    "M_013":["prompt","genai"],
    "M_014":["automation","genai"],
    "M_015":["nocode","genai"],
    "M_016":["mlops","ml","python","git"],
    "M_017":["career"],
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
LATEST_GROUP = np.empty(len(all_users), dtype=object)
LATEST_GROUP[:] = "no_chat"

for u in all_users:
    ui = uid2idx[u]
    g = user_chat.get(u)
    if g is None or len(g)==0:
        continue
    learn = g[~g["is_noise"]].copy()
    if len(learn)==0:
        learn = g.copy()
    rows = list(learn.itertuples(index=False))
    if rows:
        LATEST_GROUP[ui] = rows[-1].skeleton or "empty"
    n=len(rows)
    for j,row in enumerate(rows):
        dist=n-1-j
        w=float(np.exp(-0.7*dist))
        row_topics=set(row.topics)
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

DOM_INTENT=np.full(len(all_users),N_MODULES,np.int16)
for i in range(len(all_users)):
    if float(MODULE_RECENCY[i].max())>0:
        DOM_INTENT[i]=int(np.argmax(MODULE_RECENCY[i]))

print("catalog lexical matrices:",LEX_WORD.shape,LEX_CHAR.shape)
print("latest template groups:",len(np.unique(LATEST_GROUP[:N_TRAIN])))

# ---- cell 16 (code) ----
# Exact counterfactual target: swap the two current adjacent score values and recompute NDCG@5.
def adjacent_swap_delta(scores, y_true, higher_idx, lower_idx):
    scores=np.asarray(scores,np.float32)
    higher_idx=np.asarray(higher_idx,np.int64)
    lower_idx=np.asarray(lower_idx,np.int64)
    n=len(scores)
    before=per_user_ndcg(y_true,scores)
    swapped=scores.copy()
    rr=np.arange(n)
    tmp=swapped[rr,higher_idx].copy()
    swapped[rr,higher_idx]=swapped[rr,lower_idx]
    swapped[rr,lower_idx]=tmp
    after=per_user_ndcg(y_true,swapped)
    return (after-before).astype(np.float32)


def gather(mat, rows, cols):
    return np.asarray(mat)[np.asarray(rows,int),np.asarray(cols,int)]


def build_head_dataset(anchor_scores, base_scores, rel_scores, row_offset, head):
    """One example/user. head=12 means current #1/#2; head=23 means current #2/#3."""
    scores=np.asarray(anchor_scores,np.float32)
    n=len(scores)
    order=np.argsort(-scores,axis=1,kind="mergesort")
    pos=0 if head==12 else 1
    a=order[:,pos].astype(np.int16)      # currently higher
    b=order[:,pos+1].astype(np.int16)    # candidate to promote
    top1=order[:,0].astype(np.int16)
    rows_global=np.arange(row_offset,row_offset+n)
    rr=np.arange(n)

    top_scores=np.take_along_axis(scores,order[:,:5],axis=1)
    gap12=top_scores[:,0]-top_scores[:,1]
    gap23=top_scores[:,1]-top_scores[:,2]
    gap34=top_scores[:,2]-top_scores[:,3]
    gap45=top_scores[:,3]-top_scores[:,4]

    def pair_cols(mat):
        aa=gather(mat, rows_global, a); bb=gather(mat, rows_global, b)
        return [aa,bb,aa-bb]

    dense=[
        scores[rr,a],scores[rr,b],scores[rr,a]-scores[rr,b],
        gap12,gap23,gap34,gap45,
        *pair_cols(np.vstack([base_oof,base_test])),
        *pair_cols(np.vstack([meta_oof,meta_test])),
        *pair_cols(LEX_WORD),
        *pair_cols(LEX_CHAR),
        *pair_cols(MODULE_COUNT),
        *pair_cols(MODULE_RECENCY),
        *pair_cols(MODULE_LATEST),
        *pair_cols(MODULE_READINESS),
        *pair_cols(MODULE_INVALID),
    ]
    numeric=np.column_stack(dense).astype(np.float32)
    structured=USER_FEATURES[rows_global].astype(np.float32)

    one_a=np.eye(N_MODULES,dtype=np.float32)[a]
    one_b=np.eye(N_MODULES,dtype=np.float32)[b]
    one_top=np.eye(N_MODULES,dtype=np.float32)[top1]
    pair_id=(a.astype(np.int32)*N_MODULES+b.astype(np.int32))
    pair_one=np.eye(N_MODULES*N_MODULES,dtype=np.float32)[pair_id]

    base=np.column_stack([numeric,structured,one_a,one_b,one_top,pair_one]).astype(np.float32)
    return {
        "head":int(head), "base":base, "a":a, "b":b, "top1":top1,
        "pair_id":pair_id, "ctx":DOM_INTENT[rows_global].astype(np.int16),
        "rows_global":rows_global.astype(np.int32),
        "margin":(scores[rr,a]-scores[rr,b]).astype(np.float32),
    }

TRAIN12=build_head_dataset(consensus3_oof,base_oof,meta_oof,0,12)
TRAIN23=build_head_dataset(consensus3_oof,base_oof,meta_oof,0,23)
TEST12=build_head_dataset(v25_anchor_test,base_test,meta_test,N_TRAIN,12)
TEST23=build_head_dataset(v25_anchor_test,base_test,meta_test,N_TRAIN,23)

TRAIN12["delta"]=adjacent_swap_delta(consensus3_oof,y,TRAIN12["a"],TRAIN12["b"])
TRAIN23["delta"]=adjacent_swap_delta(consensus3_oof,y,TRAIN23["a"],TRAIN23["b"])
for ds in [TRAIN12,TRAIN23]:
    ds["label"]=(ds["delta"]>1e-12).astype(np.int8)

for name,ds in [("#1/#2",TRAIN12),("#2/#3",TRAIN23)]:
    d=ds["delta"]
    print(name,"positive",round(float(np.mean(d>1e-12)),4),"harm",round(float(np.mean(d<-1e-12)),4),
          "tie",round(float(np.mean(np.abs(d)<=1e-12)),4),"mean forced",round(float(d.mean()),6))

# ---- cell 17 (markdown) ----
# ## Fold-safe empirical-Bayes priors and action models
#
# Each position gets its own head. A head blends a conservative nonlinear model (LightGBM) with a low-capacity linear model. Classification predicts whether the swap is positive; regression predicts direct ΔNDCG. The two signals must agree before an action is eligible.
#
# Importantly, the empirical pair/context priors are rebuilt inside every split. Training rows receive leave-one-out priors; validation/test rows see only aggregate statistics from the fit partition.

# ---- cell 18 (code) ----
PAIR_ALPHA=12.0
CTX_ALPHA=24.0
DELTA_ALPHA=16.0


def _aggregate(keys, values, nkeys):
    cnt=np.bincount(keys,minlength=nkeys).astype(np.float64)
    sm=np.bincount(keys,weights=values,minlength=nkeys).astype(np.float64)
    return cnt,sm


def build_prior_state(ds, fit_idx):
    fit_idx=np.asarray(fit_idx,int)
    ycls=ds["label"][fit_idx].astype(float)
    ydel=ds["delta"][fit_idx].astype(float)
    pair=ds["pair_id"][fit_idx].astype(int)
    # ctx has N_MODULES + 1 values; keep ordered pair in the key.
    ctxkey=(ds["ctx"][fit_idx].astype(int)*(N_MODULES*N_MODULES)+pair).astype(int)
    npair=N_MODULES*N_MODULES
    nctx=(N_MODULES+1)*npair
    pc,ps=_aggregate(pair,ycls,npair)
    _,pd=_aggregate(pair,ydel,npair)
    cc,cs=_aggregate(ctxkey,ycls,nctx)
    _,cd=_aggregate(ctxkey,ydel,nctx)
    return {
        "fit_idx":fit_idx,"pair_cnt":pc,"pair_sum":ps,"pair_delta":pd,
        "ctx_cnt":cc,"ctx_sum":cs,"ctx_delta":cd,
        "global_prob":float(ycls.mean()),"global_delta":float(ydel.mean()),
    }


def prior_features_external(ds, idx, state):
    idx=np.asarray(idx,int)
    pair=ds["pair_id"][idx].astype(int)
    ctxkey=(ds["ctx"][idx].astype(int)*(N_MODULES*N_MODULES)+pair).astype(int)
    pc=state["pair_cnt"][pair]; ps=state["pair_sum"][pair]; pd=state["pair_delta"][pair]
    pair_prob=(ps+PAIR_ALPHA*state["global_prob"])/(pc+PAIR_ALPHA)
    pair_delta=(pd+DELTA_ALPHA*state["global_delta"])/(pc+DELTA_ALPHA)
    cc=state["ctx_cnt"][ctxkey]; cs=state["ctx_sum"][ctxkey]; cd=state["ctx_delta"][ctxkey]
    ctx_prob=(cs+CTX_ALPHA*pair_prob)/(cc+CTX_ALPHA)
    ctx_delta=(cd+CTX_ALPHA*pair_delta)/(cc+CTX_ALPHA)
    return np.column_stack([pair_prob,np.log1p(pc),pair_delta,ctx_prob,np.log1p(cc),ctx_delta]).astype(np.float32)


def prior_features_loo(ds, state):
    idx=state["fit_idx"]
    pair=ds["pair_id"][idx].astype(int)
    ctxkey=(ds["ctx"][idx].astype(int)*(N_MODULES*N_MODULES)+pair).astype(int)
    ycls=ds["label"][idx].astype(float); ydel=ds["delta"][idx].astype(float)
    pc=np.maximum(0.0,state["pair_cnt"][pair]-1.0)
    ps=state["pair_sum"][pair]-ycls
    pd=state["pair_delta"][pair]-ydel
    global_prob=(ycls.sum()-ycls+PAIR_ALPHA*state["global_prob"])/(max(len(idx)-1,1)+PAIR_ALPHA)
    global_delta=(ydel.sum()-ydel+DELTA_ALPHA*state["global_delta"])/(max(len(idx)-1,1)+DELTA_ALPHA)
    pair_prob=(ps+PAIR_ALPHA*global_prob)/(pc+PAIR_ALPHA)
    pair_delta=(pd+DELTA_ALPHA*global_delta)/(pc+DELTA_ALPHA)
    cc=np.maximum(0.0,state["ctx_cnt"][ctxkey]-1.0)
    cs=state["ctx_sum"][ctxkey]-ycls
    cd=state["ctx_delta"][ctxkey]-ydel
    ctx_prob=(cs+CTX_ALPHA*pair_prob)/(cc+CTX_ALPHA)
    ctx_delta=(cd+CTX_ALPHA*pair_delta)/(cc+CTX_ALPHA)
    return np.column_stack([pair_prob,np.log1p(pc),pair_delta,ctx_prob,np.log1p(cc),ctx_delta]).astype(np.float32)


def make_split_features(train_ds, fit_idx, eval_ds, eval_idx):
    state=build_prior_state(train_ds,fit_idx)
    xfit=np.column_stack([train_ds["base"][fit_idx],prior_features_loo(train_ds,state)]).astype(np.float32)
    xeval=np.column_stack([eval_ds["base"][eval_idx],prior_features_external(eval_ds,eval_idx,state)]).astype(np.float32)
    return xfit,xeval,state


def fit_action_models(xfit,yfit_cls,yfit_delta,xeval,seed):
    # Fixed iteration counts: the validation partition is never used for early stopping.
    clf=lgb.LGBMClassifier(
        n_estimators=ACTION_TREES,learning_rate=.025,num_leaves=15,min_child_samples=70,
        reg_alpha=1.0,reg_lambda=10.0,colsample_bytree=.72,subsample=.85,
        random_state=int(seed),n_jobs=4,verbosity=-1,class_weight="balanced"
    )
    reg=lgb.LGBMRegressor(
        objective="regression_l1",n_estimators=ACTION_TREES,learning_rate=.025,
        num_leaves=15,min_child_samples=70,reg_alpha=1.0,reg_lambda=10.0,
        colsample_bytree=.72,subsample=.85,random_state=int(seed+1000),n_jobs=4,verbosity=-1
    )
    clf.fit(xfit,yfit_cls)
    reg.fit(xfit,yfit_delta)
    p_tree=clf.predict_proba(xeval)[:,1]
    r_tree=reg.predict(xeval)

    scaler=StandardScaler()
    xs=scaler.fit_transform(xfit)
    xe=scaler.transform(xeval)
    lin_clf=LogisticRegression(C=.08,class_weight="balanced",max_iter=1800,random_state=int(seed))
    lin_reg=Ridge(alpha=25.0,solver="lsqr",tol=1e-3,max_iter=2500)
    lin_clf.fit(xs,yfit_cls)
    lin_reg.fit(xs,yfit_delta)
    p_lin=lin_clf.predict_proba(xe)[:,1]
    r_lin=lin_reg.predict(xe)

    return (0.70*p_tree+0.30*p_lin).astype(np.float32),(0.70*r_tree+0.30*r_lin).astype(np.float32)


def run_crossfit(ds, rows, folds, seed_offset=0):
    rows=np.asarray(rows,int)
    p=np.full(len(ds["delta"]),np.nan,np.float32)
    r=np.full(len(ds["delta"]),np.nan,np.float32)
    fid=np.full(len(ds["delta"]),-1,np.int16)
    for f,(tr_pos,va_pos) in enumerate(folds):
        fit_idx=rows[np.asarray(tr_pos,int)]
        val_idx=rows[np.asarray(va_pos,int)]
        xf,xv,_=make_split_features(ds,fit_idx,ds,val_idx)
        pp,rr=fit_action_models(xf,ds["label"][fit_idx],ds["delta"][fit_idx],xv,SEED+seed_offset+97*f)
        p[val_idx]=pp; r[val_idx]=rr; fid[val_idx]=f
    assert np.isfinite(p[rows]).all() and np.isfinite(r[rows]).all()
    return p,r,fid


def run_single_split(ds,fit_idx,val_idx,seed_offset=0):
    fit_idx=np.asarray(fit_idx,int); val_idx=np.asarray(val_idx,int)
    xf,xv,_=make_split_features(ds,fit_idx,ds,val_idx)
    p,r=fit_action_models(xf,ds["label"][fit_idx],ds["delta"][fit_idx],xv,SEED+seed_offset)
    return p,r


def run_final_test(train_ds,test_ds,seed_offset=0):
    fit_idx=np.arange(N_TRAIN)
    test_idx=np.arange(N_TEST)
    xf,xt,_=make_split_features(train_ds,fit_idx,test_ds,test_idx)
    p,r=fit_action_models(xf,train_ds["label"],train_ds["delta"],xt,SEED+seed_offset)
    return p,r

# ---- cell 19 (markdown) ----
# ## Four validation worlds
#
# - **Selection world:** deterministic hash DEV users, evaluated by latest-template `GroupKFold`. Only this world may choose coverage.
# - **Ordinary audit:** shuffled stratified CV on the same DEV rows.
# - **Fresh LOCK:** deterministic hash holdout never used for coverage selection.
# - **Chronological audit:** users U0001–U3000 train, U3001–U4000 validation.
#
# The LOCK and chronological results are read only after the coverage is frozen by cluster-safe DEV OOF.

# ---- cell 20 (code) ----
def is_lock_uid(u):
    h=int(hashlib.md5(str(u).encode("utf-8")).hexdigest()[:8],16)
    return (h % 5)==0

LOCK_MASK=np.asarray([is_lock_uid(u) for u in train.user_id],bool)
DEV_IDX=np.flatnonzero(~LOCK_MASK)
LOCK_IDX=np.flatnonzero(LOCK_MASK)
CHRONO_TRAIN=np.flatnonzero(TRAIN_ID_NUM<=3000)
CHRONO_VAL=np.flatnonzero(TRAIN_ID_NUM>3000)

print("DEV/LOCK",len(DEV_IDX),len(LOCK_IDX),"chrono",len(CHRONO_TRAIN),len(CHRONO_VAL))


def make_group_folds(ds):
    groups=LATEST_GROUP[DEV_IDX]
    gkf=GroupKFold(n_splits=N_FOLDS)
    return list(gkf.split(np.zeros(len(DEV_IDX)),ds["label"][DEV_IDX],groups))


def make_ordinary_folds(ds):
    skf=StratifiedKFold(N_FOLDS,shuffle=True,random_state=SEED+ds["head"])
    return list(skf.split(np.zeros(len(DEV_IDX)),ds["label"][DEV_IDX]))

worlds={}
for head,ds in [(12,TRAIN12),(23,TRAIN23)]:
    print("\n=== HEAD",head,"===")
    gp,gr,gfid=run_crossfit(ds,DEV_IDX,make_group_folds(ds),seed_offset=10000+head)
    op,orr,ofid=run_crossfit(ds,DEV_IDX,make_ordinary_folds(ds),seed_offset=20000+head)
    lp,lr=run_single_split(ds,DEV_IDX,LOCK_IDX,seed_offset=30000+head)
    cp,cr=run_single_split(ds,CHRONO_TRAIN,CHRONO_VAL,seed_offset=40000+head)
    worlds[head]={
        "group_p":gp,"group_r":gr,"group_fid":gfid,
        "ordinary_p":op,"ordinary_r":orr,"ordinary_fid":ofid,
        "lock_p":lp,"lock_r":lr,"chrono_p":cp,"chrono_r":cr,
    }
    def _eval_pred_for_rows(pred, rows, total_n):
        """Return predictions aligned to `rows` for diagnostics.

        Cross-fit predictions are stored in full-length arrays with NaN outside
        the evaluated subset, while LOCK/chrono predictions are already compact.
        Slice only the former so roc_auc_score never sees the intentional NaNs.
        """
        pred=np.asarray(pred)
        rows=np.asarray(rows,int)
        if pred.shape[0] == int(total_n):
            pred=pred[rows]
        if pred.shape[0] != rows.shape[0]:
            raise AssertionError(
                f"Diagnostic prediction length mismatch: pred={pred.shape[0]} rows={rows.shape[0]}"
            )
        if not np.isfinite(pred).all():
            bad=int(np.sum(~np.isfinite(pred)))
            raise AssertionError(f"Diagnostic predictions contain {bad} non-finite values after alignment")
        return pred

    for nm,rows,pred in [
        ("group",DEV_IDX,gp),("ordinary",DEV_IDX,op),("lock",LOCK_IDX,lp),("chrono",CHRONO_VAL,cp)
    ]:
        yy=ds["label"][rows]
        eval_pred=_eval_pred_for_rows(pred,rows,len(ds["label"]))
        auc=float(roc_auc_score(yy,eval_pred)) if len(np.unique(yy))>1 else np.nan
        print(nm,"AUC",round(auc,4))

# ---- cell 21 (code) ----
# Coverage selection and robustness metrics.
def gain_loss_break_even(delta, rows):
    d=np.asarray(delta)[np.asarray(rows,int)]
    pos=d[d>1e-12]; neg=d[d<-1e-12]
    gain=float(pos.mean()) if len(pos) else 0.0
    loss=float(-neg.mean()) if len(neg) else 0.0
    be=loss/max(gain+loss,1e-12)
    return gain,loss,be


def conservative_joint(p,r,gain,loss):
    prob_eu=p*gain-(1.0-p)*loss
    return np.minimum(r,prob_eu).astype(np.float32)


def group_boot_ci_policy(policy_delta, rows, seed):
    rows=np.asarray(rows,int)
    groups=LATEST_GROUP[rows]
    df=pd.DataFrame({"g":groups,"d":np.asarray(policy_delta,float)})
    gd=df.groupby("g",sort=False)["d"].agg(["sum","count"])
    keys=np.arange(len(gd)); rng=np.random.default_rng(seed)
    vals=np.empty(BOOTSTRAPS,float)
    sums=gd["sum"].to_numpy(float); cnt=gd["count"].to_numpy(float)
    for i in range(BOOTSTRAPS):
        z=rng.choice(keys,size=len(keys),replace=True)
        vals[i]=sums[z].sum()/max(cnt[z].sum(),1.0)
    return float(np.quantile(vals,.025)),float(np.quantile(vals,.975))


def select_by_coverage(ds, rows, p, r, coverage, fold_id=None, min_prob=None):
    rows=np.asarray(rows,int); p=np.asarray(p,float); r=np.asarray(r,float)
    gain,loss,be=gain_loss_break_even(ds["delta"],DEV_IDX)
    if min_prob is None:
        min_prob=max(be+0.03,0.58 if ds["head"]==12 else 0.55)
    joint=conservative_joint(p,r,gain,loss)
    eligible=(p>=min_prob)&(r>0)&(joint>0)
    k=max(1,int(np.ceil(float(coverage)*len(rows))))
    cand=np.flatnonzero(eligible)
    if len(cand)>k:
        cand=cand[np.argsort(-joint[cand],kind="mergesort")[:k]]
    selected=np.zeros(len(rows),bool); selected[cand]=True
    d=ds["delta"][rows]
    policy=np.where(selected,d,0.0)
    pos=int(np.sum(d[selected]>1e-12)); neg=int(np.sum(d[selected]<-1e-12)); ties=int(np.sum(np.abs(d[selected])<=1e-12))
    precision=pos/max(pos+neg,1)
    ci_lo,ci_hi=group_boot_ci_policy(policy,rows,SEED+ds["head"]+int(coverage*10000))
    uniq_groups=len(np.unique(LATEST_GROUP[rows][selected])) if selected.any() else 0
    fold_wins=fold_count=0; worst=np.nan
    if fold_id is not None:
        fv=[]
        ff=np.asarray(fold_id)[rows]
        for f in sorted(set(ff.tolist())):
            if f<0: continue
            z=policy[ff==f]
            if len(z): fv.append(float(z.mean()))
        fold_count=len(fv); fold_wins=sum(v>0 for v in fv); worst=min(fv) if fv else np.nan
    return {
        "coverage":float(coverage),"selected":int(selected.sum()),"positive":pos,"harmful":neg,"ties":ties,
        "precision":float(precision),"mean_delta":float(policy.mean()),"ci_low":ci_lo,"ci_high":ci_hi,
        "independent_groups":int(uniq_groups),"fold_wins":int(fold_wins),"fold_count":int(fold_count),
        "worst_fold":float(worst) if np.isfinite(worst) else np.nan,"break_even":float(be),"min_prob":float(min_prob),
        "joint":joint,"selected_mask":selected,"policy_delta":policy,
    }


def choose_coverage(ds, p, r, fold_id):
    rows=DEV_IDX
    rec=[]
    for cov in COVERAGE_GRID:
        m=select_by_coverage(ds,rows,p[rows],r[rows],cov,fold_id=fold_id)
        min_prec=.65 if ds["head"]==12 else .62
        feasible=bool(
            m["selected"]>=MIN_DEV_ACTIONS and m["precision"]>=min_prec and m["mean_delta"]>0 and
            m["fold_wins"]>=max(1,N_FOLDS-1) and m["independent_groups"]>=max(6,int(.60*m["selected"])) and
            (not np.isfinite(m["worst_fold"]) or m["worst_fold"]>=-1.5e-4) and m["ci_low"]>=-5e-5
        )
        robust=m["mean_delta"]+0.50*min(0.0,m["ci_low"])
        rec.append({k:v for k,v in m.items() if k not in {"joint","selected_mask","policy_delta"}} | {"feasible":feasible,"robust_score":robust})
    df=pd.DataFrame(rec)
    ok=df[df["feasible"]].copy()
    if len(ok):
        best=ok.sort_values(["robust_score","precision"],ascending=False).iloc[0]
    else:
        best=df.sort_values(["robust_score","precision"],ascending=False).iloc[0]
    return df,float(best["coverage"]),bool(best["feasible"])


def pseudo_public_stress(policy_delta, seed):
    d=np.asarray(policy_delta,float); n=len(d); rng=np.random.default_rng(seed)
    k=max(1,int(PSEUDO_PUBLIC_FRAC*n)); vals=[]
    for _ in range(PSEUDO_PUBLIC_MASKS):
        idx=rng.choice(n,size=k,replace=False)
        vals.append(float(d[idx].mean()))
    vals=np.asarray(vals,float)
    return {
        "pseudo_positive_rate":float(np.mean(vals>0)),
        "pseudo_nonnegative_rate":float(np.mean(vals>=0)),
        "pseudo_median":float(np.median(vals)),
        "pseudo_p10":float(np.quantile(vals,.10)),
        "pseudo_p90":float(np.quantile(vals,.90)),
    }

selection_tables={}; selected_coverage={}; selection_feasible={}
for head,ds in [(12,TRAIN12),(23,TRAIN23)]:
    w=worlds[head]
    table,cov,feasible=choose_coverage(ds,w["group_p"],w["group_r"],w["group_fid"])
    selection_tables[head]=table; selected_coverage[head]=cov; selection_feasible[head]=feasible
    print("\nHEAD",head,"selected coverage",cov,"cluster-feasible",feasible)
    display(table)

# ---- cell 22 (markdown) ----
# ## Frozen-coverage audits
#
# Coverage is now frozen. The following audits are **not allowed to retune it**. A strict head must remain positive in ordinary CV, chronological validation, and the fresh hash LOCK, and it must survive random pseudo-public masks. A relaxed gate is reported separately for analysis, but `submission_v27_primary.csv` uses only strict evidence.

# ---- cell 23 (code) ----
def audit_one(head,ds):
    cov=selected_coverage[head]; w=worlds[head]
    group=select_by_coverage(ds,DEV_IDX,w["group_p"][DEV_IDX],w["group_r"][DEV_IDX],cov,fold_id=w["group_fid"])
    ordinary=select_by_coverage(ds,DEV_IDX,w["ordinary_p"][DEV_IDX],w["ordinary_r"][DEV_IDX],cov,fold_id=w["ordinary_fid"])
    lock=select_by_coverage(ds,LOCK_IDX,w["lock_p"],w["lock_r"],cov,fold_id=None)
    chrono=select_by_coverage(ds,CHRONO_VAL,w["chrono_p"],w["chrono_r"],cov,fold_id=None)
    pseudo=pseudo_public_stress(group["policy_delta"],SEED+50000+head)

    strict=bool(
        selection_feasible[head] and ordinary["mean_delta"]>0 and lock["mean_delta"]>0 and chrono["mean_delta"]>0 and
        pseudo["pseudo_positive_rate"]>=0.70 and pseudo["pseudo_p10"]>=-5e-5
    )
    relaxed=bool(
        selection_feasible[head] and ordinary["mean_delta"]>=-1e-5 and lock["mean_delta"]>=-5e-5 and chrono["mean_delta"]>=-5e-5 and
        sum(x["mean_delta"]>0 for x in [ordinary,lock,chrono])>=2 and pseudo["pseudo_positive_rate"]>=0.60
    )
    rows=[]
    for nm,m in [("group_selection",group),("ordinary",ordinary),("lock",lock),("chrono",chrono)]:
        rows.append({"head":head,"world":nm,**{k:v for k,v in m.items() if k not in {"joint","selected_mask","policy_delta"}}})
    return pd.DataFrame(rows),pseudo,strict,relaxed

validation_parts=[]; pseudo_rows=[]; strict_gate={}; relaxed_gate={}
for head,ds in [(12,TRAIN12),(23,TRAIN23)]:
    tab,pseudo,strict,relaxed=audit_one(head,ds)
    validation_parts.append(tab); pseudo_rows.append({"head":head,**pseudo})
    strict_gate[head]=strict; relaxed_gate[head]=relaxed
    print("HEAD",head,"STRICT",strict,"RELAXED",relaxed,pseudo)
    display(tab)

validation_summary=pd.concat(validation_parts,ignore_index=True)
pseudo_summary=pd.DataFrame(pseudo_rows)
display(pseudo_summary)

# ---- cell 24 (markdown) ----
# ## Final test inference and medium-width policies
#
# The final models are fit once on all 4,000 labeled users with the frozen hyperparameters and frozen coverage. Existing V22.3/V25 package users are protected. No user may receive more than one new action.
#
# Generated policies:
#
# - **upper12_strict:** only #1↔#2, enabled only if Head-12 strict validation passes;
# - **upper23_strict:** only #2↔#3, diagnostic/alternative;
# - **upper123_balanced:** union of heads that pass the relaxed gate, with Head-12 priority on conflicts;
# - **primary:** the safest strict candidate; otherwise exact 0.65919 anchor.

# ---- cell 25 (code) ----
TEST_UID=np.asarray(test.user_id.astype(str))
PROTECTED_MASK=np.asarray([u in PROTECTED_TEST_USERS for u in TEST_UID],bool)


def test_actions_for_head(head,train_ds,test_ds,max_actions):
    p,r=run_final_test(train_ds,test_ds,seed_offset=60000+head)
    gain,loss,be=gain_loss_break_even(train_ds["delta"],DEV_IDX)
    min_prob=max(be+0.03,0.58 if head==12 else 0.55)
    joint=conservative_joint(p,r,gain,loss)
    eligible=(p>=min_prob)&(r>0)&(joint>0)&(~PROTECTED_MASK)
    k=max(1,int(np.ceil(selected_coverage[head]*N_TEST)))
    k=min(k,max_actions)
    cand=np.flatnonzero(eligible)
    if len(cand)>k:
        cand=cand[np.argsort(-joint[cand],kind="mergesort")[:k]]
    mask=np.zeros(N_TEST,bool); mask[cand]=True
    return p,r,joint,mask

TP12,TR12,TJ12,TM12=test_actions_for_head(12,TRAIN12,TEST12,MAX_TEST_HEAD12)
TP23,TR23,TJ23,TM23=test_actions_for_head(23,TRAIN23,TEST23,MAX_TEST_HEAD23)

# Strict candidates fail closed to no new action when their head is not strict.
STRICT12=TM12 & bool(strict_gate[12])
STRICT23=TM23 & bool(strict_gate[23])
RELAX12=TM12 & bool(relaxed_gate[12])
RELAX23=TM23 & bool(relaxed_gate[23])

# Balanced: relaxed heads, one action/user, upper head wins a collision because NDCG impact is larger.
BAL12=RELAX12.copy()
BAL23=RELAX23 & (~BAL12)
if int(BAL12.sum()+BAL23.sum())>MAX_TEST_COMBINED:
    rows=[]
    for i in np.flatnonzero(BAL12): rows.append((float(TJ12[i]),12,i))
    for i in np.flatnonzero(BAL23): rows.append((float(TJ23[i]),23,i))
    keep=sorted(rows,reverse=True)[:MAX_TEST_COMBINED]
    BAL12[:]=False; BAL23[:]=False
    for _,h,i in keep:
        (BAL12 if h==12 else BAL23)[i]=True

print("test raw head12",int(TM12.sum()),"strict",int(STRICT12.sum()),"relaxed",int(RELAX12.sum()))
print("test raw head23",int(TM23.sum()),"strict",int(STRICT23.sum()),"relaxed",int(RELAX23.sum()))
print("balanced",int(BAL12.sum()+BAL23.sum()))

# ---- cell 26 (code) ----
def apply_variable_swaps(anchor, ds, mask):
    out=np.asarray(anchor,np.float32).copy()
    rows=np.flatnonzero(mask)
    for i in rows:
        a=int(ds["a"][i]); b=int(ds["b"][i])
        if not float(out[i,a])>float(out[i,b]):
            raise AssertionError(f"Current ordering drift at {test.iloc[i].user_id}: {MODULES[a]} !> {MODULES[b]}")
        out[i,a],out[i,b]=out[i,b],out[i,a]
    return out

anchor_scores=v25_anchor_test.copy()
# Raw policies are diagnostics only: they expose the frozen-coverage action set even if a final gate abstains.
RAWBAL12=TM12.copy()
RAWBAL23=TM23 & (~RAWBAL12)
if int(RAWBAL12.sum()+RAWBAL23.sum())>MAX_TEST_COMBINED:
    _rows=[]
    for i in np.flatnonzero(RAWBAL12): _rows.append((float(TJ12[i]),12,i))
    for i in np.flatnonzero(RAWBAL23): _rows.append((float(TJ23[i]),23,i))
    _keep=sorted(_rows,reverse=True)[:MAX_TEST_COMBINED]
    RAWBAL12[:]=False; RAWBAL23[:]=False
    for _,h,i in _keep:
        (RAWBAL12 if h==12 else RAWBAL23)[i]=True

raw12_scores=apply_variable_swaps(anchor_scores,TEST12,TM12)
raw23_scores=apply_variable_swaps(anchor_scores,TEST23,TM23)
rawbalanced_scores=apply_variable_swaps(anchor_scores,TEST12,RAWBAL12)
rawbalanced_scores=apply_variable_swaps(rawbalanced_scores,TEST23,RAWBAL23)
upper12_scores=apply_variable_swaps(anchor_scores,TEST12,STRICT12)
upper23_scores=apply_variable_swaps(anchor_scores,TEST23,STRICT23)
balanced_scores=apply_variable_swaps(anchor_scores,TEST12,BAL12)
balanced_scores=apply_variable_swaps(balanced_scores,TEST23,BAL23)

# First-shot primary: safest strict head. Head-12 is preferred because that is the high-leverage hypothesis.
if strict_gate[12] and int(STRICT12.sum())>0:
    primary_policy="upper12_strict"
    primary_scores=upper12_scores
elif strict_gate[23] and int(STRICT23.sum())>0:
    primary_policy="upper23_strict"
    primary_scores=upper23_scores
else:
    primary_policy="anchor_fail_closed"
    primary_scores=anchor_scores.copy()

# Collect action diagnostics.
action_rows=[]
for head,ds,p,r,j,raw,strict,bal in [
    (12,TEST12,TP12,TR12,TJ12,TM12,STRICT12,BAL12),
    (23,TEST23,TP23,TR23,TJ23,TM23,STRICT23,BAL23),
]:
    for i in np.flatnonzero(raw):
        action_rows.append({
            "user_id":TEST_UID[i],"head":head,"current_higher":MODULES[int(ds['a'][i])],
            "promote":MODULES[int(ds['b'][i])],"anchor_margin":float(ds['margin'][i]),
            "prob_positive":float(p[i]),"reg_delta":float(r[i]),"joint_utility":float(j[i]),
            "strict_selected":bool(strict[i]),"balanced_selected":bool(bal[i]),
            "protected":bool(PROTECTED_MASK[i]),
        })
test_actions=pd.DataFrame(action_rows).sort_values(["head","joint_utility"],ascending=[True,False]) if action_rows else pd.DataFrame()
display(test_actions.head(60))
print("PRIMARY POLICY:",primary_policy)

# ---- cell 27 (markdown) ----
# ## Hard invariants, outputs, and submission discipline
#
# Every V27 candidate must preserve Top-5 membership for every user because it only swaps ranks inside the current Top-3. Existing eight public-tested package users are untouched. The notebook writes all diagnostics so the first remaining Kaggle slot can be decided from actual run evidence rather than from the design-time hypothesis.

# ---- cell 28 (code) ----
OUT_DIR=Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("/mnt/data")
OUT_DIR.mkdir(parents=True,exist_ok=True)


def save_submission(scores,filename):
    out=sample.copy(); out.loc[:,MODULES]=np.asarray(scores,np.float32)
    assert list(out.columns)==list(sample.columns)
    assert out["user_id"].tolist()==sample["user_id"].tolist()
    assert np.isfinite(out[MODULES].to_numpy()).all()
    path=OUT_DIR/filename; out.to_csv(path,index=False)
    return out,path

anchor_sub,anchor_path=save_submission(anchor_scores,"submission_v27_anchor_v25_65919.csv")
primary_sub,primary_path=save_submission(primary_scores,"submission_v27_primary.csv")
upper12_sub,upper12_path=save_submission(upper12_scores,"submission_v27_upper12_strict.csv")
upper23_sub,upper23_path=save_submission(upper23_scores,"submission_v27_upper23_strict.csv")
balanced_sub,balanced_path=save_submission(balanced_scores,"submission_v27_upper123_balanced.csv")
raw12_sub,raw12_path=save_submission(raw12_scores,"submission_v27_DIAGNOSTIC_raw_upper12.csv")
raw23_sub,raw23_path=save_submission(raw23_scores,"submission_v27_DIAGNOSTIC_raw_upper23.csv")
rawbal_sub,rawbal_path=save_submission(rawbalanced_scores,"submission_v27_DIAGNOSTIC_raw_upper123.csv")

# Invariants: no membership change and no protected-user edit.
def changed_rows(a,b):
    return np.flatnonzero(np.any(np.asarray(a)!=np.asarray(b),axis=1))

for name,scores in [("primary",primary_scores),("upper12",upper12_scores),("upper23",upper23_scores),("balanced",balanced_scores),("raw12",raw12_scores),("raw23",raw23_scores),("rawbalanced",rawbalanced_scores)]:
    rows=changed_rows(anchor_scores,scores)
    assert not any(PROTECTED_MASK[rows]), f"{name}: protected user changed"
    assert top5_set_change_rate(anchor_scores,scores)==0.0, f"{name}: Top5 membership changed"
    # One swap = two numeric cells per changed row.
    assert int(np.sum(np.asarray(anchor_scores)!=np.asarray(scores)))==2*len(rows)
    print(name,"changed users",len(rows),"top1 change rate",round(top1_change_rate(anchor_scores,scores),5))

validation_summary.to_csv(OUT_DIR/"v27_validation_summary.csv",index=False)
pseudo_summary.to_csv(OUT_DIR/"v27_pseudo_public_summary.csv",index=False)
selection_tables[12].to_csv(OUT_DIR/"v27_head12_coverage_search.csv",index=False)
selection_tables[23].to_csv(OUT_DIR/"v27_head23_coverage_search.csv",index=False)
test_actions.to_csv(OUT_DIR/"v27_test_actions.csv",index=False)

slot_plan=pd.DataFrame([
    {
        "slot":1,
        "candidate":"submission_v27_primary.csv",
        "rule":"Submit only when primary_policy is not anchor_fail_closed.",
        "purpose":"Safest strict upper-rank population correction; Head-12 preferred.",
    },
    {
        "slot":2,
        "candidate":"submission_v27_upper123_balanced.csv",
        "rule":"Consider only if first strict candidate improves public LB and balanced has nonzero additional actions.",
        "purpose":"Medium-width expansion across independently validated upper-rank heads.",
    },
])
slot_plan.to_csv(OUT_DIR/"v27_slot_plan.csv",index=False)
display(slot_plan)

summary={
    "version":"V27",
    "strategy":"selective upper-rank counterfactual reranking",
    "public_anchor":PUBLIC_V25_LB,
    "leader_reference":PUBLIC_LEADER_LB,
    "primary_policy":primary_policy,
    "selected_coverage":{"12":selected_coverage[12],"23":selected_coverage[23]},
    "strict_gate":{"12":strict_gate[12],"23":strict_gate[23]},
    "relaxed_gate":{"12":relaxed_gate[12],"23":relaxed_gate[23]},
    "test_counts":{
        "head12_raw":int(TM12.sum()),"head12_strict":int(STRICT12.sum()),
        "head23_raw":int(TM23.sum()),"head23_strict":int(STRICT23.sum()),
        "balanced":int(BAL12.sum()+BAL23.sum()),
    },
    "protected_users":sorted(PROTECTED_TEST_USERS),
    "rules_contract":"Organizer CSVs only; no external data; no direct LLM/GenAI test prediction; deterministic seeds.",
}
with open(OUT_DIR/"v27_summary.json","w") as f: json.dump(summary,f,indent=2)

print("\nSaved V27 outputs:")
for p in sorted(OUT_DIR.glob("*v27*")):
    print(" -",p.name)
print("\nPRIMARY POLICY:",primary_policy)
print("If primary_policy == anchor_fail_closed, DO NOT burn a Kaggle slot on submission_v27_primary.csv.")

# ---- cell 29 (markdown) ----
# ## Interpretation after the run
#
# The notebook is deliberately allowed to abstain. A non-anchor `submission_v27_primary.csv` means that the #1/#2 or #2/#3 policy survived **cluster-safe selection plus ordinary, chronological, LOCK, and pseudo-public audits**. That is the standard required before using one of the remaining public submissions.
#
# If `primary_policy == anchor_fail_closed`, do not submit the primary file just because it exists; inspect `v27_validation_summary.csv`, `v27_pseudo_public_summary.csv`, and `v27_test_actions.csv` first. The balanced file is an expansion candidate, not an automatic second submission.
