# Converted from notebooks/v27_1.ipynb: markdown cells become comments, %/! magics neutralized.

# ---- cell 0 (markdown) ----
# # MineToday V27.1 — Calibrated Support-Aware Head12 Counterfactual Reranker
#
# **Active public anchor:** V25 core-6 package = **0.65919**.  
# **Leader reference at design time:** **0.66112**.  
# **Metric:** user-level NDCG@5.
#
# V27 established one useful residual fact: the **#1↔#2 counterfactual head ranks signal consistently** across template-safe CV, ordinary CV, fresh holdout, and chronology, while #2↔#3 is close to noise. V27 failed at deployment because its balanced-class scores were treated as calibrated probabilities, the action policy could override strong anchor margins, pair/template counterevidence was only a feature rather than an abstention authority, and validation coverage did not exactly match test coverage.
#
# V27.1 keeps the useful Head12 learner and replaces the decision layer:
#
# 1. **Head12 only** — no #2↔#3 branch.
# 2. Raw classifier/regressor predictions are generated out-of-fold.
# 3. A second, **cross-fitted unweighted calibrator** learns true swap probability and direct expected ΔNDCG from OOF predictions plus organizer-data support evidence.
# 4. The champion anchor receives a learned **trust region** on its #1/#2 score margin.
# 5. Pair-family, exact-template, and assessment-neighbour evidence are computed strictly from the fit partition; strong independent contradictions force abstention.
# 6. A small preregistered policy family is selected only on template-cluster DEV OOF by worst-case stability, then frozen.
# 7. The frozen policy is audited on ordinary CV, a **new V27.1 salted hash LOCK**, chronological validation, and multi-fraction pseudo-public masks.
# 8. Validation and test use the **exact same policy and coverage fraction**; there is no arbitrary test action cap.
# 9. Existing public-tested V22.3/V25 users are protected. The notebook fails closed to 0.65919 when the strict gate is not passed.
#
# ### Competition-rule contract
#
# Only the organizer-provided CSV files are used. No external dataset or external ground truth is used. No LLM/Generative-AI model is asked to predict test labels. The solution uses deterministic Python/scikit-learn/LightGBM models and records all seeds. This remains within the Technical Meeting constraints.
#
# **Important:** this notebook is designed to maximize *generalizable* public upside; it does not guarantee a leaderboard score above 0.66112. A Kaggle submission should be made only after inspecting the generated validation diagnostics.

# ---- cell 1 (code) ----
# Runtime + deterministic configuration
import os, sys, re, json, ast, html, hashlib, random, warnings, time, subprocess
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ["PYTHONHASHSEED"] = "42"

V271_SKIP_INSTALL = bool(int(os.getenv("V271_SKIP_INSTALL", "0")))
try:
    import lightgbm as lgb
except Exception:
    if V271_SKIP_INSTALL:
        raise
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "lightgbm==4.6.0"])
    import lightgbm as lgb

if getattr(lgb, "__version__", "") != "4.6.0" and not V271_SKIP_INSTALL:
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
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

warnings.filterwarnings("ignore")

SEED = 42
V271_SMOKE = bool(int(os.getenv("V271_SMOKE", "0")))
V271_STRICT_GEOMETRY = bool(int(os.getenv("V271_STRICT_GEOMETRY", "1")))
# Compatibility aliases used only by the frozen V3.1/V25 reconstruction cells below.
V27_SMOKE = V271_SMOKE
V27_STRICT_GEOMETRY = V271_STRICT_GEOMETRY

N_FOLDS = 2 if V271_SMOKE else 5
INNER_CAL_FOLDS = 2 if V271_SMOKE else 5
RIDGE_ALPHA = 3.0
REFERENCE_W = 0.45
PUBLIC_CONSENSUS3_LB = 0.65856
PUBLIC_PRE_V25_PB = 0.65912
PUBLIC_V25_LB = 0.65919
PUBLIC_CURRENT_PB = PUBLIC_V25_LB
PUBLIC_LEADER_LB = 0.66112
PUBLIC_CAL045_LB = 0.65639
LGB_ESTIMATORS = 40 if V271_SMOKE else 1200
LGB_EARLY_STOP = 8 if V271_SMOKE else 80
ACTION_TREES = 80 if V271_SMOKE else 420
BOOTSTRAPS = 200 if V271_SMOKE else 2200
PSEUDO_PUBLIC_MASKS = 20 if V271_SMOKE else 120
GEOMETRY_ATOL = 2.5e-4

# Existing public-tested package remains immutable during V27.1 discovery.
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
print("V271_SMOKE", V271_SMOKE, "leader gap", round(PUBLIC_LEADER_LB-PUBLIC_CURRENT_PB, 5))
try:
    print("GPU runtime:\n" + subprocess.check_output(["nvidia-smi", "-L"], text=True).strip())
except Exception:
    print("nvidia-smi unavailable; V27.1 is T4x2-compatible but intentionally CPU-first.")

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
# ## V27.1 decision layer: calibrated, support-aware, and abstention-first
#
# The Head12 base learner is retained because V27 showed consistent ranking AUC across multiple worlds. The upgrade is entirely in **decision theory**:
#
# - balanced-class raw probabilities are treated only as scores;
# - unweighted cross-fitted calibration estimates true `P(ΔNDCG > 0)`;
# - a low-capacity Ridge calibrator estimates direct expected ΔNDCG;
# - ordered-pair support, exact latest-template support, and assessment-neighbour evidence are built only from the fit partition;
# - strong contradictions from two or more independent organizer-data sources force abstention;
# - pair directions with little/no train support cannot trigger Top-1 reversals.
#
# No test label or public leaderboard feedback enters these features.

# ---- cell 18 (code) ----
# Head12-only support evidence.
PAIR_ALPHA = 14.0
CTX_ALPHA = 28.0
DELTA_ALPHA = 18.0
TEMPLATE_ALPHA = 10.0
KNN_K = 15
EPS = 1e-6

# Stable integer ids for exact latest-template clusters across train+test.
_group_values = pd.Series(LATEST_GROUP.astype(str)).fillna("no_chat")
_group_cat = pd.Categorical(_group_values)
TEMPLATE_ID = _group_cat.codes.astype(np.int32)
N_TEMPLATE = int(TEMPLATE_ID.max()) + 1

# Assessment evidence uses only organizer self-assessment values + invalid flags.
ASSESS_EVIDENCE = np.column_stack([A / 5.0, A_INVALID.astype(np.float32)]).astype(np.float32)

EVIDENCE_COLS = [
    "pair_prob", "pair_count", "pair_delta",
    "ctx_prob", "ctx_count", "ctx_delta",
    "template_prob", "template_count", "template_delta",
    "knn_prob", "knn_count", "knn_delta",
    "lex_word_direction", "lex_char_direction",
    "latest_direction", "recency_direction", "readiness_direction",
    "negative_votes", "positive_votes", "unseen_pair",
]
EVIDENCE_INDEX = {c:i for i,c in enumerate(EVIDENCE_COLS)}


def _aggregate(keys, values, nkeys):
    cnt=np.bincount(keys,minlength=nkeys).astype(np.float64)
    sm=np.bincount(keys,weights=values,minlength=nkeys).astype(np.float64)
    return cnt,sm


def build_prior_state(ds, fit_idx):
    fit_idx=np.asarray(fit_idx,int)
    ycls=ds["label"][fit_idx].astype(float)
    ydel=ds["delta"][fit_idx].astype(float)
    pair=ds["pair_id"][fit_idx].astype(int)
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
    gp=(ycls.sum()-ycls+PAIR_ALPHA*state["global_prob"])/(max(len(idx)-1,1)+PAIR_ALPHA)
    gd=(ydel.sum()-ydel+DELTA_ALPHA*state["global_delta"])/(max(len(idx)-1,1)+DELTA_ALPHA)
    pair_prob=(ps+PAIR_ALPHA*gp)/(pc+PAIR_ALPHA)
    pair_delta=(pd+DELTA_ALPHA*gd)/(pc+DELTA_ALPHA)
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
    # Balanced models are deliberately treated as *ranking models*, not calibrated probabilities.
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
    clf.fit(xfit,yfit_cls); reg.fit(xfit,yfit_delta)
    p_tree=clf.predict_proba(xeval)[:,1]; r_tree=reg.predict(xeval)

    scaler=StandardScaler(); xs=scaler.fit_transform(xfit); xe=scaler.transform(xeval)
    lin_clf=LogisticRegression(C=.08,class_weight="balanced",max_iter=1800,random_state=int(seed))
    lin_reg=Ridge(alpha=25.0,solver="lsqr",tol=1e-3,max_iter=2500)
    lin_clf.fit(xs,yfit_cls); lin_reg.fit(xs,yfit_delta)
    p_lin=lin_clf.predict_proba(xe)[:,1]; r_lin=lin_reg.predict(xe)
    return (0.70*p_tree+0.30*p_lin).astype(np.float32),(0.70*r_tree+0.30*r_lin).astype(np.float32)


def build_evidence_state(ds, fit_idx):
    fit_idx=np.asarray(fit_idx,int)
    prior=build_prior_state(ds,fit_idx)
    # Exact-template × ordered-pair aggregates.
    tdict={}
    for i in fit_idx:
        key=(int(TEMPLATE_ID[int(ds["rows_global"][i])]), int(ds["pair_id"][i]))
        z=tdict.get(key)
        if z is None: tdict[key]=[1.0,float(ds["label"][i]),float(ds["delta"][i])]
        else:
            z[0]+=1.0; z[1]+=float(ds["label"][i]); z[2]+=float(ds["delta"][i])
    # Pair-specific row pools for assessment-neighbour evidence.
    pair_rows={}
    for pid in np.unique(ds["pair_id"][fit_idx]):
        pair_rows[int(pid)] = fit_idx[ds["pair_id"][fit_idx] == pid]
    return {
        "prior":prior,"template":tdict,"pair_rows":pair_rows,
        "source_rows_global":ds["rows_global"].copy(),
        "source_label":ds["label"].copy(),
        "source_delta":ds["delta"].copy(),
    }


def support_features_external(ds, idx, state):
    idx=np.asarray(idx,int)
    prior_raw=prior_features_external(ds,idx,state["prior"])
    pair_prob=prior_raw[:,0].astype(float)
    pair_count=np.expm1(prior_raw[:,1]).astype(float)
    pair_delta=prior_raw[:,2].astype(float)
    ctx_prob=prior_raw[:,3].astype(float)
    ctx_count=np.expm1(prior_raw[:,4]).astype(float)
    ctx_delta=prior_raw[:,5].astype(float)

    n=len(idx)
    tprob=np.empty(n,float); tcount=np.zeros(n,float); tdelta=np.empty(n,float)
    kprob=np.empty(n,float); kcount=np.zeros(n,float); kdelta=np.empty(n,float)

    for z,i in enumerate(idx):
        pid=int(ds["pair_id"][i]); gid=int(TEMPLATE_ID[int(ds["rows_global"][i])])
        st=state["template"].get((gid,pid))
        if st is None:
            cnt=pos=sm=0.0
        else:
            cnt,pos,sm=st
        tcount[z]=cnt
        tprob[z]=(pos+TEMPLATE_ALPHA*pair_prob[z])/(cnt+TEMPLATE_ALPHA)
        tdelta[z]=(sm+TEMPLATE_ALPHA*pair_delta[z])/(cnt+TEMPLATE_ALPHA)

        cand=state["pair_rows"].get(pid,np.empty(0,dtype=int))
        if len(cand)==0:
            kprob[z]=pair_prob[z]; kdelta[z]=pair_delta[z]; kcount[z]=0.0
        else:
            eg=int(ds["rows_global"][i])
            fg=state["source_rows_global"][cand].astype(int)
            diff=ASSESS_EVIDENCE[fg]-ASSESS_EVIDENCE[eg]
            dist=np.sqrt(np.sum(diff*diff,axis=1))
            kk=min(KNN_K,len(cand))
            take=np.argpartition(dist,kk-1)[:kk] if kk<len(cand) else np.arange(len(cand))
            dd=dist[take]
            w=1.0/(0.20+dd)
            rr=cand[take]
            kcount[z]=float(kk)
            kprob[z]=float(np.sum(w*state["source_label"][rr])/np.sum(w))
            kdelta[z]=float(np.sum(w*state["source_delta"][rr])/np.sum(w))

    gr=ds["rows_global"][idx].astype(int)
    a=ds["a"][idx].astype(int); b=ds["b"][idx].astype(int)
    lexw=LEX_WORD[gr,b]-LEX_WORD[gr,a]
    lexc=LEX_CHAR[gr,b]-LEX_CHAR[gr,a]
    latest=MODULE_LATEST[gr,b]-MODULE_LATEST[gr,a]
    recency=MODULE_RECENCY[gr,b]-MODULE_RECENCY[gr,a]
    readiness=MODULE_READINESS[gr,b]-MODULE_READINESS[gr,a]

    pair_neg=(pair_count>=10)&(pair_prob<0.40)&(pair_delta<-0.005)
    temp_neg=(tcount>=5)&(tprob<0.40)&(tdelta<-0.008)
    knn_neg=(kcount>=5)&(kprob<0.40)&(kdelta<-0.008)
    pair_pos=(pair_count>=10)&(pair_prob>0.55)&(pair_delta>0.005)
    temp_pos=(tcount>=5)&(tprob>0.55)&(tdelta>0.008)
    knn_pos=(kcount>=5)&(kprob>0.55)&(kdelta>0.008)
    neg_votes=pair_neg.astype(int)+temp_neg.astype(int)+knn_neg.astype(int)
    pos_votes=pair_pos.astype(int)+temp_pos.astype(int)+knn_pos.astype(int)
    unseen=(pair_count<3).astype(int)

    ev=np.column_stack([
        pair_prob,pair_count,pair_delta,ctx_prob,ctx_count,ctx_delta,
        tprob,tcount,tdelta,kprob,kcount,kdelta,
        lexw,lexc,latest,recency,readiness,
        neg_votes,pos_votes,unseen,
    ]).astype(np.float32)
    assert np.isfinite(ev).all()
    return ev

print("Head12 evidence columns:", EVIDENCE_COLS)

# ---- cell 19 (markdown) ----
# ## Leakage-safe raw Head12 worlds
#
# Every raw prediction used by the calibration layer is itself out-of-fold. Support evidence for an evaluation row is generated only from that split's fit rows. The calibration layer therefore never sees a row's target through either the base learner or its pair/template/assessment evidence.

# ---- cell 20 (code) ----
def run_crossfit_support(ds, rows, folds, seed_offset=0):
    rows=np.asarray(rows,int)
    n=len(ds["delta"])
    p=np.full(n,np.nan,np.float32); r=np.full(n,np.nan,np.float32)
    ev=np.full((n,len(EVIDENCE_COLS)),np.nan,np.float32)
    fid=np.full(n,-1,np.int16)
    for f,(tr_pos,va_pos) in enumerate(folds):
        fit_idx=rows[np.asarray(tr_pos,int)]; val_idx=rows[np.asarray(va_pos,int)]
        xf,xv,_=make_split_features(ds,fit_idx,ds,val_idx)
        pp,rr=fit_action_models(xf,ds["label"][fit_idx],ds["delta"][fit_idx],xv,SEED+seed_offset+97*f)
        est=build_evidence_state(ds,fit_idx)
        p[val_idx]=pp; r[val_idx]=rr; ev[val_idx]=support_features_external(ds,val_idx,est); fid[val_idx]=f
    assert np.isfinite(p[rows]).all() and np.isfinite(r[rows]).all() and np.isfinite(ev[rows]).all()
    return p,r,ev,fid


def run_single_split_support(train_ds,fit_idx,eval_ds,eval_idx,seed_offset=0):
    fit_idx=np.asarray(fit_idx,int); eval_idx=np.asarray(eval_idx,int)
    xf,xv,_=make_split_features(train_ds,fit_idx,eval_ds,eval_idx)
    p,r=fit_action_models(xf,train_ds["label"][fit_idx],train_ds["delta"][fit_idx],xv,SEED+seed_offset)
    est=build_evidence_state(train_ds,fit_idx)
    ev=support_features_external(eval_ds,eval_idx,est)
    return p,r,ev


def make_group_folds_for_rows(ds, rows, n_splits=N_FOLDS):
    rows=np.asarray(rows,int)
    groups=LATEST_GROUP[ds["rows_global"][rows]]
    n_unique=len(np.unique(groups))
    k=max(2,min(int(n_splits),n_unique))
    gkf=GroupKFold(n_splits=k)
    return list(gkf.split(np.zeros(len(rows)),ds["label"][rows],groups))


def make_strat_folds_for_rows(ds, rows, n_splits=N_FOLDS, seed=SEED):
    rows=np.asarray(rows,int)
    k=max(2,min(int(n_splits),int(np.bincount(ds["label"][rows]).min())))
    skf=StratifiedKFold(k,shuffle=True,random_state=int(seed))
    return list(skf.split(np.zeros(len(rows)),ds["label"][rows]))

# New protocol LOCK: different salt from V27, so V27.1 policy design has not seen this split.
def is_v271_lock_uid(u):
    h=int(hashlib.sha1(("v271-fresh-lock::"+str(u)).encode("utf-8")).hexdigest()[:10],16)
    return (h % 5)==0

V271_LOCK_MASK=np.asarray([is_v271_lock_uid(u) for u in train.user_id],bool)
DEV_IDX=np.flatnonzero(~V271_LOCK_MASK)
LOCK_IDX=np.flatnonzero(V271_LOCK_MASK)
CHRONO_TRAIN=np.flatnonzero(TRAIN_ID_NUM<=3000)
CHRONO_VAL=np.flatnonzero(TRAIN_ID_NUM>3000)

print("V27.1 DEV/LOCK",len(DEV_IDX),len(LOCK_IDX),"chrono",len(CHRONO_TRAIN),len(CHRONO_VAL))

# Selection world: template-safe DEV OOF.
gp,gr,gev,gfid=run_crossfit_support(TRAIN12,DEV_IDX,make_group_folds_for_rows(TRAIN12,DEV_IDX),seed_offset=11000)
# Ordinary audit world: same DEV rows, ordinary stratified OOF.
op,orr,oev,ofid=run_crossfit_support(TRAIN12,DEV_IDX,make_strat_folds_for_rows(TRAIN12,DEV_IDX,seed=SEED+12),seed_offset=21000)
# Fresh LOCK gets a model fit only on DEV.
lp,lr,lev=run_single_split_support(TRAIN12,DEV_IDX,TRAIN12,LOCK_IDX,seed_offset=31000)
# Chronological validation gets a model fit only on U0001-U3000.
cp,cr,cev=run_single_split_support(TRAIN12,CHRONO_TRAIN,TRAIN12,CHRONO_VAL,seed_offset=41000)

for nm,rows,pred in [("group",DEV_IDX,gp),("ordinary",DEV_IDX,op),("lock",LOCK_IDX,lp),("chrono",CHRONO_VAL,cp)]:
    z=pred[rows] if len(pred)==N_TRAIN else pred
    print(nm,"raw AUC",round(float(roc_auc_score(TRAIN12["label"][rows],z)),4))

# ---- cell 21 (markdown) ----
# ## Cross-fitted calibration of probability and direct utility
#
# The balanced-class Head12 output is now treated only as a ranking score. A tiny **unweighted** logistic calibrator estimates the natural-prevalence probability of a positive swap, while Ridge estimates direct expected ΔNDCG. Both are trained only on OOF raw predictions and fold-safe support evidence.
#
# This directly fixes the main V27 error: using class-balanced `predict_proba()` as if it were an economic posterior probability.

# ---- cell 22 (code) ----
CAL_FEATURE_NAMES = [
    "raw_logit","raw_reg","anchor_margin",
    *EVIDENCE_COLS,
]


def calibration_matrix(ds, rows, raw_p, raw_r, ev):
    rows=np.asarray(rows,int)
    pp=np.clip(np.asarray(raw_p,float),1e-5,1-1e-5)
    logit=np.log(pp/(1.0-pp))
    return np.column_stack([logit,np.asarray(raw_r,float),ds["margin"][rows],np.asarray(ev,float)]).astype(np.float32)


def fit_calibrators(X, y_cls, y_delta):
    sc=StandardScaler(); xs=sc.fit_transform(X)
    # Unweighted logistic regression restores the natural class prevalence.
    pc=LogisticRegression(C=.30,max_iter=2500,random_state=SEED)
    dr=Ridge(alpha=18.0,solver="lsqr",tol=1e-4,max_iter=4000)
    pc.fit(xs,y_cls); dr.fit(xs,y_delta)
    return sc,pc,dr


def apply_calibrators(models, X):
    sc,pc,dr=models; xs=sc.transform(X)
    return pc.predict_proba(xs)[:,1].astype(np.float32),dr.predict(xs).astype(np.float32)


def crossfit_calibration(ds, rows, raw_p_full, raw_r_full, ev_full, fid_full):
    rows=np.asarray(rows,int); fid=np.asarray(fid_full)[rows]
    cp=np.full(len(rows),np.nan,np.float32); cd=np.full(len(rows),np.nan,np.float32)
    for f in sorted(set(fid.tolist())):
        va_pos=np.flatnonzero(fid==f); tr_pos=np.flatnonzero(fid!=f)
        tr=rows[tr_pos]; va=rows[va_pos]
        Xtr=calibration_matrix(ds,tr,raw_p_full[tr],raw_r_full[tr],ev_full[tr])
        Xva=calibration_matrix(ds,va,raw_p_full[va],raw_r_full[va],ev_full[va])
        models=fit_calibrators(Xtr,ds["label"][tr],ds["delta"][tr])
        pp,dd=apply_calibrators(models,Xva); cp[va_pos]=pp; cd[va_pos]=dd
    assert np.isfinite(cp).all() and np.isfinite(cd).all()
    return cp,cd


def fit_calibrator_from_oof(ds, rows, raw_p_full, raw_r_full, ev_full):
    rows=np.asarray(rows,int)
    X=calibration_matrix(ds,rows,raw_p_full[rows],raw_r_full[rows],ev_full[rows])
    return fit_calibrators(X,ds["label"][rows],ds["delta"][rows])

# Cross-fitted calibration inside the two DEV worlds.
GCP,GCD=crossfit_calibration(TRAIN12,DEV_IDX,gp,gr,gev,gfid)
OCP,OCD=crossfit_calibration(TRAIN12,DEV_IDX,op,orr,oev,ofid)

# LOCK calibrator: trained only on template-safe DEV OOF predictions.
gcal=fit_calibrator_from_oof(TRAIN12,DEV_IDX,gp,gr,gev)
LX=calibration_matrix(TRAIN12,LOCK_IDX,lp,lr,lev)
LCP,LCD=apply_calibrators(gcal,LX)

# Chronological calibrator: build OOF raw evidence *inside U0001-U3000 only*.
ctp,ctr,ctev,ctfid=run_crossfit_support(
    TRAIN12,CHRONO_TRAIN,make_group_folds_for_rows(TRAIN12,CHRONO_TRAIN),seed_offset=51000
)
ccal=fit_calibrator_from_oof(TRAIN12,CHRONO_TRAIN,ctp,ctr,ctev)
CX=calibration_matrix(TRAIN12,CHRONO_VAL,cp,cr,cev)
CCP,CCD=apply_calibrators(ccal,CX)


def print_calibration(name, rows, p, d):
    yy=TRAIN12["label"][rows]; dd=TRAIN12["delta"][rows]
    print(name,
          "AUC",round(float(roc_auc_score(yy,p)),4),
          "Brier",round(float(brier_score_loss(yy,p)),5),
          "LogLoss",round(float(log_loss(yy,np.clip(p,1e-6,1-1e-6))),5),
          "corr_delta",round(float(np.corrcoef(dd,d)[0,1]),4))

print_calibration("group calibrated",DEV_IDX,GCP,GCD)
print_calibration("ordinary calibrated",DEV_IDX,OCP,OCD)
print_calibration("LOCK calibrated",LOCK_IDX,LCP,LCD)
print_calibration("chrono calibrated",CHRONO_VAL,CCP,CCD)

# ---- cell 23 (markdown) ----
# ## Preregistered support-aware trust policies
#
# V27.1 intentionally does **not** optimize dozens of free thresholds. It evaluates a small preregistered family that encodes the conclusions from V27:
#
# - Top-1 reversals need a small anchor margin unless support is very strong;
# - an ordered pair must have train support;
# - calibrated probability and calibrated expected ΔNDCG must both be positive;
# - two independent strong contradictions force abstention;
# - the selection criterion prioritizes **worst-fold stability and lower confidence bound**, not the largest mean OOF gain.
#
# Only template-safe DEV OOF may choose the policy. LOCK/chronology cannot retune it.

# ---- cell 24 (code) ----
POLICY_FAMILY = [
    {"name":"ultra005",    "coverage":0.005,"margin_cap":0.010,"pair_support":10,"pmin":0.60,"max_neg":0},
    {"name":"ultra010",    "coverage":0.010,"margin_cap":0.020,"pair_support":10,"pmin":0.60,"max_neg":0},
    {"name":"trust010",    "coverage":0.010,"margin_cap":0.040,"pair_support":15,"pmin":0.60,"max_neg":0},
    {"name":"trust015",    "coverage":0.015,"margin_cap":0.040,"pair_support":10,"pmin":0.60,"max_neg":0},
    {"name":"trust020",    "coverage":0.020,"margin_cap":0.040,"pair_support":10,"pmin":0.58,"max_neg":0},
    {"name":"support020",  "coverage":0.020,"margin_cap":0.020,"pair_support":5, "pmin":0.60,"max_neg":0},
    {"name":"support030",  "coverage":0.030,"margin_cap":0.040,"pair_support":10,"pmin":0.60,"max_neg":1},
    {"name":"wide020",     "coverage":0.020,"margin_cap":0.080,"pair_support":20,"pmin":0.65,"max_neg":0},
]


def group_boot_ci_policy(policy_delta, rows, seed):
    rows=np.asarray(rows,int); groups=LATEST_GROUP[TRAIN12["rows_global"][rows]]
    df=pd.DataFrame({"g":groups,"d":np.asarray(policy_delta,float)})
    gd=df.groupby("g",sort=False)["d"].agg(["sum","count"])
    rng=np.random.default_rng(seed); vals=np.empty(BOOTSTRAPS,float)
    sums=gd["sum"].to_numpy(float); cnt=gd["count"].to_numpy(float); keys=np.arange(len(gd))
    for i in range(BOOTSTRAPS):
        z=rng.choice(keys,size=len(keys),replace=True)
        vals[i]=sums[z].sum()/max(cnt[z].sum(),1.0)
    return float(np.quantile(vals,.025)),float(np.quantile(vals,.975))


def evaluate_policy(ds, rows, cal_p, cal_d, ev, policy, fold_id=None, protected=None):
    rows=np.asarray(rows,int); cal_p=np.asarray(cal_p,float); cal_d=np.asarray(cal_d,float); ev=np.asarray(ev,float)
    if len(cal_p)!=len(rows) or len(cal_d)!=len(rows) or len(ev)!=len(rows):
        raise AssertionError("Policy inputs must be compact and aligned to rows")
    pair_count=ev[:,EVIDENCE_INDEX["pair_count"]]
    neg_votes=ev[:,EVIDENCE_INDEX["negative_votes"]]
    unseen=ev[:,EVIDENCE_INDEX["unseen_pair"]]>0.5
    margin=ds["margin"][rows]

    # Two independent contradictions automatically abstain. Strict policies require zero contradictions.
    eligible=(
        (cal_p>=float(policy["pmin"])) & (cal_d>0) &
        (margin<=float(policy["margin_cap"])) &
        (pair_count>=float(policy["pair_support"])) &
        (neg_votes<=float(policy["max_neg"])) & (~unseen)
    )
    if protected is not None:
        eligible &= ~np.asarray(protected,bool)

    # Exact same fraction-based policy is used in every world and on test; no test-only hard cap.
    k=max(1,int(np.ceil(float(policy["coverage"])*len(rows))))
    cand=np.flatnonzero(eligible)
    # Conservative ranking uses calibrated expected ΔNDCG, probability only as a guard.
    score=cal_d*np.clip((cal_p-0.50)/0.50,0.0,1.0)
    if len(cand)>k:
        cand=cand[np.argsort(-score[cand],kind="mergesort")[:k]]
    selected=np.zeros(len(rows),bool); selected[cand]=True

    has_labels = "delta" in ds
    if has_labels:
        d=ds["delta"][rows]
        pd=np.where(selected,d,0.0)
        pos=int(np.sum(d[selected]>1e-12)); neg=int(np.sum(d[selected]<-1e-12)); ties=int(np.sum(np.abs(d[selected])<=1e-12))
        precision=pos/max(pos+neg,1)
        ci_lo,ci_hi=group_boot_ci_policy(pd,rows,SEED+70000+int(policy["coverage"]*10000)+int(policy["margin_cap"]*1000))
    else:
        d=np.full(len(rows),np.nan,float); pd=np.zeros(len(rows),float)
        pos=neg=ties=0; precision=np.nan; ci_lo=ci_hi=np.nan
    uniq_groups=len(np.unique(LATEST_GROUP[ds["rows_global"][rows[selected]]])) if selected.any() else 0

    fold_wins=fold_count=0; worst=np.nan
    if has_labels and fold_id is not None:
        ff=np.asarray(fold_id)[rows]; fv=[]
        for f in sorted(set(ff.tolist())):
            if f<0: continue
            z=pd[ff==f]
            if len(z): fv.append(float(z.mean()))
        fold_count=len(fv); fold_wins=sum(v>0 for v in fv); worst=min(fv) if fv else np.nan

    return {
        "policy":policy["name"],"coverage":float(policy["coverage"]),"margin_cap":float(policy["margin_cap"]),
        "pair_support":int(policy["pair_support"]),"pmin":float(policy["pmin"]),"max_neg":int(policy["max_neg"]),
        "selected":int(selected.sum()),"eligible":int(eligible.sum()),"positive":pos,"harmful":neg,"ties":ties,
        "precision":float(precision),"mean_delta":float(pd.mean()),"ci_low":ci_lo,"ci_high":ci_hi,
        "independent_groups":int(uniq_groups),"fold_wins":int(fold_wins),"fold_count":int(fold_count),
        "worst_fold":float(worst) if np.isfinite(worst) else np.nan,
        "selected_mask":selected,"policy_delta":pd,"ranking_score":score,"eligible_mask":eligible,
    }


def group_selection_table():
    rec=[]
    for p in POLICY_FAMILY:
        m=evaluate_policy(TRAIN12,DEV_IDX,GCP,GCD,gev[DEV_IDX],p,fold_id=gfid)
        min_actions=4 if V271_SMOKE else 10
        feasible=bool(
            m["selected"]>=min_actions and m["precision"]>=0.58 and m["mean_delta"]>0 and
            m["fold_wins"]>=max(1,N_FOLDS-1) and
            (not np.isfinite(m["worst_fold"]) or m["worst_fold"]>=0.0) and
            m["ci_low"]>=-5e-5 and m["independent_groups"]>=max(6,int(.60*m["selected"]))
        )
        # Generalization-first objective: worst fold, then CI lower bound, then mean.
        rec.append({k:v for k,v in m.items() if k not in {"selected_mask","policy_delta","ranking_score","eligible_mask"}} | {"feasible":feasible})
    df=pd.DataFrame(rec)
    ok=df[df.feasible].copy()
    if len(ok):
        best=ok.sort_values(["worst_fold","ci_low","mean_delta","precision"],ascending=False).iloc[0]
        feasible=True
    else:
        best=df.sort_values(["worst_fold","ci_low","mean_delta","precision"],ascending=False).iloc[0]
        feasible=False
    return df,str(best["policy"]),feasible

selection_table,SELECTED_POLICY_NAME,SELECTION_FEASIBLE=group_selection_table()
POLICY_BY_NAME={p["name"]:p for p in POLICY_FAMILY}
SELECTED_POLICY=POLICY_BY_NAME[SELECTED_POLICY_NAME]
print("Selected policy:",SELECTED_POLICY_NAME,"group-feasible:",SELECTION_FEASIBLE)
display(selection_table.sort_values(["feasible","worst_fold","ci_low","mean_delta"],ascending=False))

# ---- cell 25 (markdown) ----
# ## Frozen audits: ordinary, new LOCK, chronology, and pseudo-public stability
#
# The policy is now frozen. These worlds may **reject** it but may not change its thresholds or coverage. Pseudo-public simulation uses several public-fraction assumptions so the candidate is not validated only under one arbitrary split size.

# ---- cell 26 (code) ----
def pseudo_public_multi(policy_delta, rows, seed):
    rows=np.asarray(rows,int); d=np.asarray(policy_delta,float); rng=np.random.default_rng(seed)
    rec=[]
    for frac in ([0.35] if V271_SMOKE else [0.25,0.35,0.50]):
        k=max(1,int(frac*len(d))); vals=[]
        for _ in range(PSEUDO_PUBLIC_MASKS):
            ix=rng.choice(len(d),size=k,replace=False); vals.append(float(d[ix].mean()))
        vals=np.asarray(vals,float)
        rec.append({
            "fraction":frac,"positive_rate":float(np.mean(vals>0)),"nonnegative_rate":float(np.mean(vals>=0)),
            "median":float(np.median(vals)),"p10":float(np.quantile(vals,.10)),"p90":float(np.quantile(vals,.90)),
        })
    return pd.DataFrame(rec)

P=SELECTED_POLICY
GROUP_AUDIT=evaluate_policy(TRAIN12,DEV_IDX,GCP,GCD,gev[DEV_IDX],P,fold_id=gfid)
ORD_AUDIT=evaluate_policy(TRAIN12,DEV_IDX,OCP,OCD,oev[DEV_IDX],P,fold_id=ofid)
LOCK_AUDIT=evaluate_policy(TRAIN12,LOCK_IDX,LCP,LCD,lev,P,fold_id=None)
CHRONO_AUDIT=evaluate_policy(TRAIN12,CHRONO_VAL,CCP,CCD,cev,P,fold_id=None)
PSEUDO=pseudo_public_multi(GROUP_AUDIT["policy_delta"],DEV_IDX,SEED+88000)

validation_rows=[]
for world,m in [("group_selection",GROUP_AUDIT),("ordinary",ORD_AUDIT),("fresh_lock",LOCK_AUDIT),("chronological",CHRONO_AUDIT)]:
    validation_rows.append({"world":world,**{k:v for k,v in m.items() if k not in {"selected_mask","policy_delta","ranking_score","eligible_mask"}}})
VALIDATION_SUMMARY=pd.DataFrame(validation_rows)
display(VALIDATION_SUMMARY)
display(PSEUDO)

pseudo_ok=bool((PSEUDO.positive_rate>=0.75).all() and (PSEUDO.p10>=-5e-5).all())
STRICT_GATE=bool(
    SELECTION_FEASIBLE and
    GROUP_AUDIT["mean_delta"]>0 and ORD_AUDIT["mean_delta"]>0 and
    LOCK_AUDIT["mean_delta"]>0 and CHRONO_AUDIT["mean_delta"]>0 and
    pseudo_ok
)
# A diagnostic medium gate is intentionally looser, but primary never uses it.
MEDIUM_GATE=bool(
    SELECTION_FEASIBLE and
    min(ORD_AUDIT["mean_delta"],LOCK_AUDIT["mean_delta"],CHRONO_AUDIT["mean_delta"])>=-3e-5 and
    sum(x["mean_delta"]>0 for x in [ORD_AUDIT,LOCK_AUDIT,CHRONO_AUDIT])>=2 and
    float(PSEUDO.positive_rate.min())>=0.65
)
print("STRICT_GATE",STRICT_GATE,"MEDIUM_GATE",MEDIUM_GATE,"pseudo_ok",pseudo_ok)

# ---- cell 27 (markdown) ----
# ## Full-train calibration and exact-geometry test deployment
#
# For test inference, the calibrator is trained on **full-train template-group OOF raw predictions**, not on in-sample model outputs. The base Head12 model is then fit on all 4,000 labels once and applied to test. The frozen policy uses the same margin/support/probability/coverage rule as validation.
#
# Existing public-tested V25 package users remain protected.

# ---- cell 28 (code) ----
# Full-train OOF is used solely to train the final calibration layer.
ALL_IDX=np.arange(N_TRAIN)
atp,atr,atev,atfid=run_crossfit_support(TRAIN12,ALL_IDX,make_group_folds_for_rows(TRAIN12,ALL_IDX),seed_offset=61000)
FINAL_CAL=fit_calibrator_from_oof(TRAIN12,ALL_IDX,atp,atr,atev)

# Final raw Head12 model fit on all train -> test.
TEST_IDX=np.arange(N_TEST)
tp,tr,tev=run_single_split_support(TRAIN12,ALL_IDX,TEST12,TEST_IDX,seed_offset=71000)
TX=calibration_matrix(TEST12,TEST_IDX,tp,tr,tev)
TCP,TCD=apply_calibrators(FINAL_CAL,TX)

TEST_UID=np.asarray(test.user_id.astype(str))
PROTECTED_MASK=np.asarray([u in PROTECTED_TEST_USERS for u in TEST_UID],bool)
TEST_POLICY=evaluate_policy(TEST12,TEST_IDX,TCP,TCD,tev,SELECTED_POLICY,fold_id=None,protected=PROTECTED_MASK)
TEST_MASK=TEST_POLICY["selected_mask"]

print("Selected test actions:",int(TEST_MASK.sum()),"of target",int(np.ceil(SELECTED_POLICY["coverage"]*N_TEST)))
print("Protected eligible suppressed:",int(np.sum(PROTECTED_MASK & TEST_POLICY["eligible_mask"])))


def apply_variable_swaps(anchor, ds, mask):
    out=np.asarray(anchor,np.float32).copy(); rows=np.flatnonzero(mask)
    for i in rows:
        a=int(ds["a"][i]); b=int(ds["b"][i])
        if not float(out[i,a])>float(out[i,b]):
            raise AssertionError(f"Anchor ordering drift at {test.iloc[i].user_id}: {MODULES[a]} !> {MODULES[b]}")
        out[i,a],out[i,b]=out[i,b],out[i,a]
    return out

anchor_scores=v25_anchor_test.copy()
frozen_policy_scores=apply_variable_swaps(anchor_scores,TEST12,TEST_MASK)
primary_scores=frozen_policy_scores if STRICT_GATE else anchor_scores.copy()
medium_scores=frozen_policy_scores if MEDIUM_GATE else anchor_scores.copy()
PRIMARY_POLICY=SELECTED_POLICY_NAME if STRICT_GATE else "anchor_fail_closed"

# Per-action diagnostics include all independent evidence channels.
rows=[]
for i in np.flatnonzero(TEST_MASK):
    ev=tev[i]
    rows.append({
        "user_id":TEST_UID[i],"current_top1":MODULES[int(TEST12['a'][i])],"promote":MODULES[int(TEST12['b'][i])],
        "anchor_margin":float(TEST12["margin"][i]),"raw_score":float(tp[i]),"raw_reg_delta":float(tr[i]),
        "calibrated_prob":float(TCP[i]),"calibrated_expected_delta":float(TCD[i]),
        **{c:float(ev[EVIDENCE_INDEX[c]]) for c in EVIDENCE_COLS},
        "protected":bool(PROTECTED_MASK[i]),
    })
TEST_ACTIONS=pd.DataFrame(rows)
if len(TEST_ACTIONS):
    TEST_ACTIONS=TEST_ACTIONS.sort_values(["calibrated_expected_delta","calibrated_prob"],ascending=False)
    display(TEST_ACTIONS)
print("PRIMARY_POLICY:",PRIMARY_POLICY)

# ---- cell 29 (markdown) ----
# ## Outputs and submission discipline
#
# V27.1 writes three score files:
#
# - `submission_v271_primary.csv` — strict gated candidate; exact 0.65919 anchor if the gate fails.
# - `submission_v271_frozen_policy_DIAGNOSTIC.csv` — the frozen policy regardless of gate, for analysis only.
# - `submission_v271_medium_DIAGNOSTIC.csv` — same frozen action set under the looser medium gate; also analysis-only unless reviewed.
#
# Do **not** submit a file just because it exists. The strict candidate is intended to be the only automatic submission candidate.

# ---- cell 30 (code) ----
OUT_DIR=Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("/mnt/data")
OUT_DIR.mkdir(parents=True,exist_ok=True)


def save_submission(scores,filename):
    out=sample.copy(); out.loc[:,MODULES]=np.asarray(scores,np.float32)
    assert list(out.columns)==list(sample.columns)
    assert out["user_id"].tolist()==sample["user_id"].tolist()
    assert np.isfinite(out[MODULES].to_numpy()).all()
    path=OUT_DIR/filename; out.to_csv(path,index=False)
    return out,path

anchor_sub,anchor_path=save_submission(anchor_scores,"submission_v271_anchor_v25_65919.csv")
primary_sub,primary_path=save_submission(primary_scores,"submission_v271_primary.csv")
frozen_sub,frozen_path=save_submission(frozen_policy_scores,"submission_v271_frozen_policy_DIAGNOSTIC.csv")
medium_sub,medium_path=save_submission(medium_scores,"submission_v271_medium_DIAGNOSTIC.csv")


def changed_rows(a,b):
    return np.flatnonzero(np.any(np.asarray(a)!=np.asarray(b),axis=1))

for name,scores in [("primary",primary_scores),("frozen",frozen_policy_scores),("medium",medium_scores)]:
    rr=changed_rows(anchor_scores,scores)
    assert not any(PROTECTED_MASK[rr]), f"{name}: protected user changed"
    assert top5_set_change_rate(anchor_scores,scores)==0.0, f"{name}: Top5 membership changed"
    assert int(np.sum(np.asarray(anchor_scores)!=np.asarray(scores)))==2*len(rr)
    print(name,"changed users",len(rr),"top1 change rate",round(top1_change_rate(anchor_scores,scores),5))

selection_table.to_csv(OUT_DIR/"v271_policy_selection.csv",index=False)
VALIDATION_SUMMARY.to_csv(OUT_DIR/"v271_validation_summary.csv",index=False)
PSEUDO.to_csv(OUT_DIR/"v271_pseudo_public_summary.csv",index=False)
TEST_ACTIONS.to_csv(OUT_DIR/"v271_test_actions.csv",index=False)

summary={
    "version":"V27.1",
    "strategy":"calibrated support-aware Head12 counterfactual reranking",
    "public_anchor":PUBLIC_CURRENT_PB,
    "leader_reference":PUBLIC_LEADER_LB,
    "selected_policy":SELECTED_POLICY,
    "selection_feasible":bool(SELECTION_FEASIBLE),
    "strict_gate":bool(STRICT_GATE),"medium_gate":bool(MEDIUM_GATE),
    "primary_policy":PRIMARY_POLICY,
    "test_selected":int(TEST_MASK.sum()),
    "validation":VALIDATION_SUMMARY.to_dict(orient="records"),
    "pseudo_public":PSEUDO.to_dict(orient="records"),
    "rules_contract":"Organizer CSVs only; no external data; no direct LLM/GenAI test prediction; deterministic seeds.",
}
with open(OUT_DIR/"v271_summary.json","w") as f: json.dump(summary,f,indent=2)

slot_plan=pd.DataFrame([
    {"slot":2,"candidate":"submission_v271_primary.csv","rule":"Submit only if STRICT_GATE=True and primary differs from anchor.","purpose":"Generalization-gated Head12 population correction."},
    {"slot":3,"candidate":"conditional after public result","rule":"Do not pre-commit slot 3; inspect the public response and diagnostics first.","purpose":"Avoid another blind expansion cycle."},
])
slot_plan.to_csv(OUT_DIR/"v271_slot_plan.csv",index=False)
display(slot_plan)

print("\nSaved V27.1 outputs:")
for p in sorted(OUT_DIR.glob("*v271*")): print(" -",p.name)
print("\nPRIMARY_POLICY:",PRIMARY_POLICY)
if PRIMARY_POLICY=="anchor_fail_closed":
    print("STRICT GATE FAILED: submission_v271_primary.csv is the exact 0.65919 anchor. Do not burn a Kaggle slot on it.")
else:
    print("STRICT GATE PASSED: inspect v271_validation_summary.csv and v271_test_actions.csv before submission.")

# ---- cell 31 (markdown) ----
# ## How to interpret the run
#
# A successful V27.1 is **not** defined by a high AUC alone. The branch is considered submission-ready only when the frozen Head12 policy:
#
# - was selected by template-safe DEV OOF without using LOCK/chronology;
# - has non-negative worst-fold behavior and a stable lower confidence bound;
# - remains positive in ordinary CV, the new salted V27.1 LOCK, and chronology;
# - remains robust across several pseudo-public fractions;
# - produces a non-trivial but medium-width test action set under the exact same coverage rule;
# - contains no protected V22.3/V25 users and never changes Top-5 membership.
#
# If the strict gate fails again, the notebook intentionally closes the Head12 branch rather than silently tuning another threshold. The next research direction should then move away from upper-rank action modeling toward a population-level calibration family.
