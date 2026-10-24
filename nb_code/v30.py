# Converted from notebooks/v30.ipynb: markdown cells become comments, %/! magics neutralized.

# ---- cell 0 (markdown) ----
# # MineToday V30 — 3-Fresh-Salt Split-Robust Pair-Conditional Curriculum Residual LTR
#
# **Purpose:** final research-to-submission candidate built from the V29/V29.5 audit.
#
# V29 and V29.5 are historical design evidence only. Their partitions are **not** reused for V30 candidate selection.
#
# V30 precommits to exactly three candidate families at a single residual strength:
#
# 1. `ltr_base` — anchor-conditioned long-format LambdaRank.
# 2. `ltr_direct` — Base + fold-safe **pair-conditional direct prerequisite** priors, activated only by a pure explicit latest intent.
# 3. `ltr_direct_cooc` — DirectCurr + the existing fold-safe label co-occurrence block.
#
# The development surface is three entirely fresh repeated nested template-group partitions (`C`, `D`, `E`). Candidate selection optimizes **worst-salt robustness**, not best mean score. A fourth sealed template-disjoint LOCK (`F`) and the chronological U_3001–U_4000 world are rejection-only after the winner is frozen.
#
# Hard exclusions from prior evidence: no generic uncertainty router, no residual covariance graph, no transitive distance-2/3 curriculum, no anchor-Top1 curriculum seed, no new semantic embedding, no post-hoc alpha search.

# ---- cell 1 (code) ----
# Runtime + preregistered V30 configuration
import os, sys, re, json, ast, html, hashlib, random, warnings, time, subprocess, gc
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ["PYTHONHASHSEED"] = "42"

V30_SKIP_INSTALL = bool(int(os.getenv("V30_SKIP_INSTALL", "0")))
try:
    import lightgbm as lgb
except Exception:
    if V30_SKIP_INSTALL:
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
V30_SMOKE = bool(int(os.getenv("V30_SMOKE", "0")))
V30_STRICT_GEOMETRY = bool(int(os.getenv("V30_STRICT_GEOMETRY", "1")))

# Compatibility aliases consumed by the frozen anchor reconstruction cells.
V29_SMOKE = V30_SMOKE
V29_STRICT_GEOMETRY = V30_STRICT_GEOMETRY
V27_SMOKE = V30_SMOKE
V27_STRICT_GEOMETRY = V30_STRICT_GEOMETRY

N_FOLDS = 2 if V30_SMOKE else 5
ANCHOR_INNER_FOLDS = 2 if V30_SMOKE else 4
DIRECT_INNER_FOLDS = 2 if V30_SMOKE else 3
GRAPH_INNER_FOLDS = 2 if V30_SMOKE else 3

RIDGE_ALPHA = 3.0
REFERENCE_W = 0.45

PUBLIC_CAL045_LB = 0.65639
PUBLIC_CONSENSUS3_LB = 0.65856
PUBLIC_V25_LB = 0.65919
PUBLIC_CURRENT_PB = PUBLIC_V25_LB
PUBLIC_LEADER_LB = 0.66112

LGB_ESTIMATORS = 12 if V30_SMOKE else 1200
LGB_EARLY_STOP = 3 if V30_SMOKE else 80
RANKER_ESTIMATORS = 12 if V30_SMOKE else 360
RANKER_LEARNING_RATE = 0.04 if V30_SMOKE else 0.025

# One and only residual strength. No post-run alpha selection.
ALPHA = 0.16

# Three candidates only.
CANDIDATES = ["ltr_base", "ltr_direct", "ltr_direct_cooc"]
CANDIDATE_COMPLEXITY = {"ltr_base":0, "ltr_direct":1, "ltr_direct_cooc":2}

# Pair-prior shrinkage is fixed before seeing V30 salts.
PAIR_PRIOR_STRENGTH = 50.0
PAIR_SUPPORT_STRENGTH = 50.0

# Three entirely fresh development partitions.
def _seed_from_text(s):
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:8], 16) % (2**31-1)

FRESH_SALTS = {
    "C": _seed_from_text("MINETODAY-V30-FRESH-C-20260828"),
    "D": _seed_from_text("MINETODAY-V30-FRESH-D-20260828"),
    "E": _seed_from_text("MINETODAY-V30-FRESH-E-20260828"),
}
SEALED_LOCK_SALT = "MINETODAY-V30-SEALED-F-20260828-REJECTION-ONLY"

BOOTSTRAPS = 20 if V30_SMOKE else 1200
# Frozen V25 geometry tolerance inherited from the public-proven V26/V27/V29 anchor reconstruction.
GEOMETRY_ATOL = 2.5e-4

# Fail-fast contract for the frozen V25 reconstruction reused below.
assert np.isfinite(GEOMETRY_ATOL) and GEOMETRY_ATOL > 0
POOLED_CI_FLOOR = -0.00005
MIN_FOLD_WINS = 4 if V30_SMOKE else 11   # 4/6 in smoke, 11/15 in full
MIN_INCREMENTAL_SALT_WINS = 2

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
print("V30_SMOKE", V30_SMOKE)
print("fixed alpha", ALPHA)
print("fresh salts", FRESH_SALTS)
print("leader gap", round(PUBLIC_LEADER_LB-PUBLIC_CURRENT_PB, 5))
try:
    print("GPU runtime:\n" + subprocess.check_output(["nvidia-smi", "-L"], text=True).strip())
except Exception:
    print("nvidia-smi unavailable; V30 is Kaggle T4x2-compatible and deliberately CPU-first.")

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
# ## Frozen public-proven anchor
#
# The next cells reproduce the exact V3.1 sparse backbone, binary relevance calibration, Consensus3 profile, and V25 public-tested package. V30 never tunes these components.

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
# ## Reconstruct the active V25 public anchor
#
# The eight public-tested rows are protected at deployment regardless of the residual-LTR result.

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
# ## V30 information blocks
#
# `DirectCurr` is deliberately narrower than V29's generic gap block. It uses only catalog-confirmed **direct** prerequisite edges and only pure explicit latest destination intents. The historical organizer preference for each destination→prerequisite edge is estimated fold-safely and shrunk toward the candidate module's baseline prevalence.
#
# `Cooc` retains the three V29 label co-occurrence features as an optional third candidate. The residual covariance graph is absent.

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

# Historical broad module-topic surfaces retained inside the Base expert.
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

# Pure explicit destination mapping for V30 DirectCurr.
# Unlike MODULE_TOPIC_MAP, these seeds do not inherit prerequisite topics.
EXPLICIT_TOPIC_TO_MODULE = {
    "excel":"M_001", "python":"M_002", "sql":"M_003", "webscrape":"M_004",
    "git":"M_005", "statistics":"M_006", "eda":"M_007", "dashboard":"M_008",
    "ml":"M_009", "ml_eval":"M_009", "cv":"M_010", "nlp":"M_011",
    "genai":"M_012", "prompt":"M_013", "automation":"M_014", "nocode":"M_015",
    "mlops":"M_016", "career":"M_017",
}
EXPLICIT_LATEST_INTENT = np.zeros((len(all_users),N_MODULES),np.float32)

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
        for tp in set(rows[-1].topics):
            mid=EXPLICIT_TOPIC_TO_MODULE.get(tp)
            if mid is not None:
                EXPLICIT_LATEST_INTENT[ui, MODULES.index(mid)] = 1.0
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
# Edge (p, m) means candidate p is a direct prerequisite of destination m.
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
# seed(destination) x candidate(prerequisite)
DIRECT_EDGE = PREREQ_MAT.T.astype(np.float32)

# Sanity-check the hand-parsed graph against organizer prerequisite strings.
prtxt=modules.set_index("module_id")["prerequisite_level"].astype(str).to_dict()
assert "Python" in prtxt["M_004"]
assert "Python" in prtxt["M_007"] and "Stat" in prtxt["M_007"]
assert "SQL" in prtxt["M_008"] and "Excel" in prtxt["M_008"]
assert "EDA" in prtxt["M_009"] and "Stat" in prtxt["M_009"]
assert "ML" in prtxt["M_010"] and "ML" in prtxt["M_011"]
assert "NLP" in prtxt["M_012"] and "ML" in prtxt["M_016"]

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
    return list(sg.split(np.zeros(len(yy)),user_strata_from_y(yy),groups=groups))

def make_group_folds_global(rows,n_splits,seed):
    rows=np.asarray(rows,int)
    local=make_group_folds_local(y[rows],TRAIN_GROUP[rows],n_splits,seed)
    return [(rows[tr],rows[va]) for tr,va in local]

print("catalog lexical matrices:",LEX_WORD.shape,LEX_CHAR.shape)
print("latest template groups:",len(np.unique(TRAIN_GROUP)))
print("direct curriculum edges:",len(PREREQ_EDGES))
print("users with pure explicit latest intent:",int(np.sum(EXPLICIT_LATEST_INTENT[:N_TRAIN].sum(1)>0)))
print("ordinal grade counts:",dict(zip(*np.unique(ORDINAL_GRADES,return_counts=True))))

# ---- cell 16 (code) ----
RANK_WEIGHTS=np.asarray([1.00,0.80,0.60,0.45,0.35],np.float32)

def topk_seed_weights(anchor,k=5):
    anchor=np.asarray(anchor,np.float32); n=len(anchor)
    order=np.argsort(-anchor,axis=1,kind="mergesort")[:,:k]
    W=np.zeros_like(anchor,np.float32)
    W[np.arange(n)[:,None],order]=RANK_WEIGHTS[:k][None,:]
    W/=np.maximum(W.sum(1,keepdims=True),1e-8)
    return W

def fit_cooc_bundle(y_fit):
    y_fit=np.asarray(y_fit,np.float32)
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
    return {
        "log_lift":log_lift.astype(np.float32),
        "cond_delta":cond_delta.astype(np.float32),
        "support_norm":support_norm.astype(np.float32),
    }

def apply_cooc_bundle(bundle,anchor):
    W=topk_seed_weights(anchor)
    return {
        "COOC_LOG_LIFT":(W@bundle["log_lift"]).astype(np.float32),
        "COOC_COND_DELTA":(W@bundle["cond_delta"]).astype(np.float32),
        "COOC_SUPPORT":(W@bundle["support_norm"]).astype(np.float32),
    }

def crossfit_cooc_features(y_local,anchor_local,groups_local,seed):
    y_local=np.asarray(y_local,np.float32); anchor_local=np.asarray(anchor_local,np.float32)
    groups_local=np.asarray(groups_local,dtype=object)
    keys=["COOC_LOG_LIFT","COOC_COND_DELTA","COOC_SUPPORT"]
    out={k:np.zeros_like(anchor_local,np.float32) for k in keys}
    folds=make_group_folds_local(y_local,groups_local,GRAPH_INNER_FOLDS,seed)
    for tr,va in folds:
        bundle=fit_cooc_bundle(y_local[tr])
        vv=apply_cooc_bundle(bundle,anchor_local[va])
        for k in keys:
            out[k][va]=vv[k]
    for k in keys:
        if not np.isfinite(out[k]).all():
            raise ValueError(f"non-finite cooccurrence feature {k}")
    return out

print("fold-safe co-occurrence helpers ready")

# ---- cell 17 (code) ----
DIRECT_KEYS = [
    "DIRECT_FLAG","DIRECT_SEED_COUNT","DIRECT_SUPPORT","DIRECT_PPOS",
    "DIRECT_LOG_LIFT","DIRECT_MEAN_REL","DIRECT_BASE_PREV","DIRECT_NEED"
]

def fit_direct_curr_bundle(y_fit, explicit_latest_fit):
    y_fit=np.asarray(y_fit,np.float32)
    latest=np.asarray(explicit_latest_fit,np.float32)
    n=len(y_fit)
    biny=(y_fit>0).astype(np.float32)
    base_prev=biny.mean(0).astype(np.float32)
    base_rel=y_fit.mean(0).astype(np.float32)

    support=np.zeros((N_MODULES,N_MODULES),np.float32)  # seed,candidate
    ppos=np.zeros_like(support)
    mean_rel=np.zeros_like(support)
    log_lift=np.zeros_like(support)

    for seed in range(N_MODULES):
        seed_mask=latest[:,seed]>0
        ns=int(seed_mask.sum())
        if ns==0:
            continue
        for cand in np.flatnonzero(DIRECT_EDGE[seed]>0):
            pos=float(biny[seed_mask,cand].sum())
            rel=float(y_fit[seed_mask,cand].sum())
            p=(pos + PAIR_PRIOR_STRENGTH*float(base_prev[cand]))/(ns+PAIR_PRIOR_STRENGTH)
            mr=(rel + PAIR_PRIOR_STRENGTH*float(base_rel[cand]))/(ns+PAIR_PRIOR_STRENGTH)
            support[seed,cand]=ns
            ppos[seed,cand]=p
            mean_rel[seed,cand]=mr
            log_lift[seed,cand]=np.log((p+1e-6)/(float(base_prev[cand])+1e-6))

    support_norm=support/(support+PAIR_SUPPORT_STRENGTH)
    return {
        "support":support,
        "support_norm":support_norm.astype(np.float32),
        "ppos":ppos.astype(np.float32),
        "mean_rel":mean_rel.astype(np.float32),
        "log_lift":log_lift.astype(np.float32),
        "base_prev":base_prev,
        "base_rel":base_rel,
    }

def apply_direct_curr_bundle(bundle, explicit_latest, readiness):
    latest=np.asarray(explicit_latest,np.float32)
    readiness=np.asarray(readiness,np.float32)
    edge=DIRECT_EDGE
    den=(latest@edge).astype(np.float32)
    flag=(den>0).astype(np.float32)
    safe=np.maximum(den,1.0)

    def agg(mat):
        z=(latest@(edge*mat))/safe
        return (z*flag).astype(np.float32)

    ppos=agg(bundle["ppos"])
    log_lift=agg(bundle["log_lift"])
    mean_rel=agg(bundle["mean_rel"])
    support=agg(bundle["support_norm"])
    seed_count=np.minimum(den,3.0)/3.0
    base_prev=(np.tile(bundle["base_prev"][None,:],(len(latest),1))*flag).astype(np.float32)
    need=(flag*np.clip((5.0-readiness)/5.0,0.0,1.0)).astype(np.float32)

    return {
        "DIRECT_FLAG":flag,
        "DIRECT_SEED_COUNT":seed_count.astype(np.float32),
        "DIRECT_SUPPORT":support,
        "DIRECT_PPOS":ppos,
        "DIRECT_LOG_LIFT":log_lift,
        "DIRECT_MEAN_REL":mean_rel,
        "DIRECT_BASE_PREV":base_prev,
        "DIRECT_NEED":need,
    }

def crossfit_direct_curr_features(y_local,latest_local,readiness_local,groups_local,seed):
    y_local=np.asarray(y_local,np.float32)
    latest_local=np.asarray(latest_local,np.float32)
    readiness_local=np.asarray(readiness_local,np.float32)
    groups_local=np.asarray(groups_local,dtype=object)
    out={k:np.zeros_like(y_local,np.float32) for k in DIRECT_KEYS}
    folds=make_group_folds_local(y_local,groups_local,DIRECT_INNER_FOLDS,seed)
    for tr,va in folds:
        bundle=fit_direct_curr_bundle(y_local[tr],latest_local[tr])
        vv=apply_direct_curr_bundle(bundle,latest_local[va],readiness_local[va])
        for k in DIRECT_KEYS:
            out[k][va]=vv[k]
    for k in DIRECT_KEYS:
        if not np.isfinite(out[k]).all():
            raise ValueError(f"non-finite DirectCurr feature {k}")
    return out

def direct_pair_audit(bundle, latest_fit, y_fit):
    rows=[]
    latest_fit=np.asarray(latest_fit,np.float32)
    y_fit=np.asarray(y_fit,np.float32)
    for seed in range(N_MODULES):
        mask=latest_fit[:,seed]>0
        for cand in np.flatnonzero(DIRECT_EDGE[seed]>0):
            ns=int(mask.sum())
            raw_pos=float((y_fit[mask,cand]>0).mean()) if ns else np.nan
            raw_rel=float(y_fit[mask,cand].mean()) if ns else np.nan
            rows.append({
                "seed_module":MODULES[seed],
                "candidate_prerequisite":MODULES[cand],
                "support":ns,
                "raw_positive_rate":raw_pos,
                "raw_mean_relevance":raw_rel,
                "candidate_base_prevalence":float(bundle["base_prev"][cand]),
                "smoothed_positive_rate":float(bundle["ppos"][seed,cand]),
                "smoothed_log_lift":float(bundle["log_lift"][seed,cand]),
                "smoothed_mean_relevance":float(bundle["mean_rel"][seed,cand]),
            })
    return pd.DataFrame(rows)

print("fold-safe pair-conditional DirectCurr helpers ready")

# ---- cell 18 (markdown) ----
# ## Anchor-conditioned residual LambdaRank
#
# All three candidates use the same ordinal LambdaRank target and the same fixed residual coefficient `α = 0.16`. Candidate differences are only information blocks; there is no hyperparameter selection after seeing fresh-salt results.

# ---- cell 19 (code) ----
def rank_matrix(scores):
    order=np.argsort(-np.asarray(scores),axis=1,kind="mergesort")
    ranks=np.empty_like(order,dtype=np.int16)
    ranks[np.arange(len(order))[:,None],order]=np.arange(1,N_MODULES+1,dtype=np.int16)[None,:]
    return ranks

def top5_order_change_rate(a,b):
    return float(np.mean(np.any(top5_indices(a)!=top5_indices(b),axis=1)))

def build_long_matrix_v30(rows_global,anchor,base,meta,direct,cooc):
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

    direct_mats=[direct[k] for k in DIRECT_KEYS]
    pieces.append(np.column_stack([m.reshape(-1) for m in direct_mats]).astype(np.float32))
    names += [
        "direct_flag","direct_seed_count","direct_support","direct_ppos",
        "direct_log_lift","direct_mean_rel","direct_base_prev","direct_need"
    ]
    bounds["direct"]=sum(p.shape[1] for p in pieces)

    cooc_mats=[cooc["COOC_LOG_LIFT"],cooc["COOC_COND_DELTA"],cooc["COOC_SUPPORT"]]
    pieces.append(np.column_stack([m.reshape(-1) for m in cooc_mats]).astype(np.float32))
    names += ["cooc_log_lift","cooc_cond_delta","cooc_support"]
    bounds["cooc"]=sum(p.shape[1] for p in pieces)

    XX=np.hstack(pieces).astype(np.float32,copy=False)
    if XX.shape[0] != n*N_MODULES:
        raise AssertionError(XX.shape)
    if len(names)!=XX.shape[1]:
        raise AssertionError((len(names),XX.shape))
    if not np.isfinite(XX).all():
        raise ValueError("non-finite V30 long features")
    return XX,names,bounds

CANDIDATE_BOUND={"ltr_base":"base","ltr_direct":"direct","ltr_direct_cooc":"cooc"}

def candidate_end(bounds,candidate):
    return bounds[CANDIDATE_BOUND[candidate]]

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

print("V30 long-format LambdaRank helpers ready")

# ---- cell 20 (code) ----
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

print("V30 nested anchor helper ready")

# ---- cell 21 (markdown) ----
# ## Three completely fresh repeated nested OOF salts
#
# `C`, `D`, and `E` are the **only** V30 development selection surface. Each salt produces a full 5-fold template-group OOF prediction, so every train user is evaluated out-of-fold three separate times under three different group partitions.
#
# The experiment is preregistered:
# - exactly 3 candidates,
# - exactly one `α = 0.16`,
# - no use of V29/V29.5 partitions in selection,
# - no sealed LOCK or chronology result can alter the frozen winner.

# ---- cell 22 (code) ----
assert callable(build_anchor_split_full)
assert set(CANDIDATES)=={"ltr_base","ltr_direct","ltr_direct_cooc"}
assert abs(ALPHA-0.16)<1e-12

SALT_ANCHOR={}
SALT_SCORES={s:{} for s in FRESH_SALTS}
SALT_DELTAS={s:{} for s in FRESH_SALTS}
SALT_FOLD_ID={}
salt_rows=[]

for salt_name,salt_seed in FRESH_SALTS.items():
    print(f"\n=== V30 fresh salt {salt_name} seed={salt_seed} ===")
    outer=make_group_folds_global(np.arange(N_TRAIN),N_FOLDS,salt_seed)
    anchor_oof=np.zeros_like(y,np.float32)
    raw_oof={c:np.zeros_like(y,np.float32) for c in CANDIDATES}
    fold_id=np.full(N_TRAIN,-1,np.int16)

    for f,(tr,va) in enumerate(outer):
        print(f"salt {salt_name} fold {f+1}/{len(outer)} train={len(tr)} val={len(va)}")
        anc=build_anchor_split_full(tr,va,salt_seed+1000+f*211)
        anchor_oof[va]=anc["cons_val"]
        fold_id[va]=f

        direct_tr=crossfit_direct_curr_features(
            y[tr],EXPLICIT_LATEST_INTENT[tr],MODULE_READINESS[tr],TRAIN_GROUP[tr],
            salt_seed+2000+f*223
        )
        direct_bundle=fit_direct_curr_bundle(y[tr],EXPLICIT_LATEST_INTENT[tr])
        direct_va=apply_direct_curr_bundle(
            direct_bundle,EXPLICIT_LATEST_INTENT[va],MODULE_READINESS[va]
        )

        cooc_tr=crossfit_cooc_features(
            y[tr],anc["cons_train"],TRAIN_GROUP[tr],salt_seed+3000+f*227
        )
        cooc_bundle=fit_cooc_bundle(y[tr])
        cooc_va=apply_cooc_bundle(cooc_bundle,anc["cons_val"])

        Xtr,names,bounds=build_long_matrix_v30(
            tr,anc["cons_train"],anc["base_train"],anc["meta_train"],direct_tr,cooc_tr
        )
        Xva,_,bounds_va=build_long_matrix_v30(
            va,anc["cons_val"],anc["base_val"],anc["meta_val"],direct_va,cooc_va
        )
        assert bounds==bounds_va

        for ci,cand in enumerate(CANDIDATES):
            end=candidate_end(bounds,cand)
            model,raw=fit_ranker_predict(
                Xtr[:,:end],Xva[:,:end],ORDINAL_GRADES[tr],len(tr),
                salt_seed+4000+f*239+ci*31
            )
            raw_oof[cand][va]=raw
            del model

        del Xtr,Xva,direct_tr,direct_va,direct_bundle,cooc_tr,cooc_va,cooc_bundle,anc
        gc.collect()

    assert (fold_id>=0).all()
    assert np.isfinite(anchor_oof).all()
    SALT_ANCHOR[salt_name]=anchor_oof
    SALT_FOLD_ID[salt_name]=fold_id

    print("salt anchor NDCG",salt_name,round(ndcg5(y,anchor_oof),6))
    for cand in CANDIDATES:
        score=blend_residual(anchor_oof,raw_oof[cand],ALPHA)
        delta=per_user_ndcg(y,score)-per_user_ndcg(y,anchor_oof)
        fold_means=[float(delta[fold_id==f].mean()) for f in range(len(outer))]
        ci=cluster_boot_ci(delta,TRAIN_GROUP,salt_seed+7000+CANDIDATES.index(cand)*101)
        SALT_SCORES[salt_name][cand]=score
        SALT_DELTAS[salt_name][cand]=delta
        salt_rows.append({
            "salt":salt_name,"candidate":cand,"alpha":ALPHA,
            "anchor_ndcg":ndcg5(y,anchor_oof),
            "candidate_ndcg":ndcg5(y,score),
            "mean_delta":float(delta.mean()),
            "fold_wins":int(np.sum(np.asarray(fold_means)>0)),
            "min_fold_delta":float(np.min(fold_means)),
            "max_fold_delta":float(np.max(fold_means)),
            "ci_low":ci[0],"ci_high":ci[1],
            "top5_membership_change":top5_set_change_rate(anchor_oof,score),
            "top5_order_change":top5_order_change_rate(anchor_oof,score),
            "top1_change":top1_change_rate(anchor_oof,score),
        })

V30_THREE_SALT=pd.DataFrame(salt_rows)
display(V30_THREE_SALT)

# ---- cell 23 (markdown) ----
# ## Precommitted candidate selection
#
# A candidate must be positive on **3/3 fresh salt means**, win at least **11/15 outer folds** in full mode, and have a template-cluster bootstrap lower bound at least `-0.00005` on the per-user delta averaged across the three salts.
#
# `DirectCurr` must additionally improve Base on at least 2/3 salts with positive pooled incremental mean. `Cooc` must likewise improve DirectCurr on at least 2/3 salts. Among eligible candidates, the frozen winner maximizes **worst-salt delta**.

# ---- cell 24 (code) ----
# Preregistered max-min candidate gate.
pooled_rows=[]
for cand in CANDIDATES:
    salt_means=[]
    fold_wins_total=0
    for salt_name in FRESH_SALTS:
        z=V30_THREE_SALT[(V30_THREE_SALT.salt==salt_name)&(V30_THREE_SALT.candidate==cand)].iloc[0]
        salt_means.append(float(z.mean_delta))
        fold_wins_total+=int(z.fold_wins)

    avg_user_delta=np.mean(np.stack([SALT_DELTAS[s][cand] for s in FRESH_SALTS],axis=0),axis=0)
    ci=cluster_boot_ci(avg_user_delta,TRAIN_GROUP,_seed_from_text(f"V30-POOLED-{cand}"))

    pooled_rows.append({
        "candidate":cand,
        "salt_C_delta":salt_means[0],
        "salt_D_delta":salt_means[1],
        "salt_E_delta":salt_means[2],
        "salt_positive_count":int(np.sum(np.asarray(salt_means)>0)),
        "worst_salt_delta":float(np.min(salt_means)),
        "mean_salt_delta":float(np.mean(salt_means)),
        "fold_wins_total":int(fold_wins_total),
        "pooled_user_delta":float(avg_user_delta.mean()),
        "pooled_ci_low":ci[0],
        "pooled_ci_high":ci[1],
    })

V30_POOLED=pd.DataFrame(pooled_rows)

mean_by={(r.salt,r.candidate):float(r.mean_delta) for r in V30_THREE_SALT.itertuples(index=False)}
direct_inc=[mean_by[(s,"ltr_direct")]-mean_by[(s,"ltr_base")] for s in FRESH_SALTS]
cooc_inc=[mean_by[(s,"ltr_direct_cooc")]-mean_by[(s,"ltr_direct")] for s in FRESH_SALTS]

V30_INCREMENTAL=pd.DataFrame([
    {
        "block":"DirectCurr_vs_Base",
        "salt_C_increment":direct_inc[0],"salt_D_increment":direct_inc[1],"salt_E_increment":direct_inc[2],
        "positive_salts":int(np.sum(np.asarray(direct_inc)>0)),
        "mean_increment":float(np.mean(direct_inc)),
        "increment_gate":bool(np.sum(np.asarray(direct_inc)>0)>=MIN_INCREMENTAL_SALT_WINS and np.mean(direct_inc)>0),
    },
    {
        "block":"Cooc_vs_DirectCurr",
        "salt_C_increment":cooc_inc[0],"salt_D_increment":cooc_inc[1],"salt_E_increment":cooc_inc[2],
        "positive_salts":int(np.sum(np.asarray(cooc_inc)>0)),
        "mean_increment":float(np.mean(cooc_inc)),
        "increment_gate":bool(np.sum(np.asarray(cooc_inc)>0)>=MIN_INCREMENTAL_SALT_WINS and np.mean(cooc_inc)>0),
    }
])

direct_gate=bool(V30_INCREMENTAL.loc[V30_INCREMENTAL.block=="DirectCurr_vs_Base","increment_gate"].iloc[0])
cooc_gate=bool(V30_INCREMENTAL.loc[V30_INCREMENTAL.block=="Cooc_vs_DirectCurr","increment_gate"].iloc[0])

gate_rows=[]
for r in V30_POOLED.itertuples(index=False):
    absolute_gate=bool(
        r.salt_positive_count==3 and
        r.fold_wins_total>=MIN_FOLD_WINS and
        r.pooled_ci_low>=POOLED_CI_FLOOR
    )
    if r.candidate=="ltr_base":
        incremental_gate=True
    elif r.candidate=="ltr_direct":
        incremental_gate=direct_gate
    else:
        incremental_gate=direct_gate and cooc_gate
    eligible=bool(absolute_gate and incremental_gate)
    gate_rows.append({
        **r._asdict(),
        "absolute_gate":absolute_gate,
        "incremental_gate":bool(incremental_gate),
        "eligible":eligible,
    })

V30_CANDIDATE_GATE=pd.DataFrame(gate_rows)
eligible=V30_CANDIDATE_GATE[V30_CANDIDATE_GATE.eligible].copy()
if len(eligible):
    eligible["complexity"]=eligible.candidate.map(CANDIDATE_COMPLEXITY)
    eligible=eligible.sort_values(
        ["worst_salt_delta","mean_salt_delta","complexity"],
        ascending=[False,False,True],kind="mergesort"
    )
    DEV_WINNER=str(eligible.iloc[0].candidate)
else:
    DEV_WINNER=None

print("\nIncremental gates")
display(V30_INCREMENTAL)
print("\nCandidate gates")
display(V30_CANDIDATE_GATE)
print("FROZEN DEVELOPMENT WINNER =",DEV_WINNER)

# ---- cell 25 (markdown) ----
# ## Sealed rejection-only validation
#
# Only after the development winner is frozen do we open:
# - a fourth unseen template-disjoint Salt F LOCK, and
# - the chronological U_3001–U_4000 world.
#
# The frozen winner survives only if Salt F is **strictly positive** and chronology is **non-negative**. These worlds never choose between candidates.

# ---- cell 26 (code) ----
# Sealed Salt F + chronology are rejection-only. They are evaluated only after DEV_WINNER is frozen.
def sealed_lock_bucket(group):
    h=hashlib.sha1(f"{SEALED_LOCK_SALT}::{group}".encode("utf-8")).hexdigest()
    return int(h[:12],16)%5

SEALED_LOCK_MASK=np.asarray([sealed_lock_bucket(g)==0 for g in TRAIN_GROUP],bool)
SEALED_DEV_IDX=np.flatnonzero(~SEALED_LOCK_MASK)
SEALED_LOCK_IDX=np.flatnonzero(SEALED_LOCK_MASK)
assert len(set(TRAIN_GROUP[SEALED_DEV_IDX]).intersection(set(TRAIN_GROUP[SEALED_LOCK_IDX])))==0

CHRONO_TR=np.arange(0,3000,dtype=int)
CHRONO_VA=np.arange(3000,4000,dtype=int)

def evaluate_frozen_candidate(candidate,train_rows,val_rows,seed,world):
    train_rows=np.asarray(train_rows,int); val_rows=np.asarray(val_rows,int)
    anc=build_anchor_split_full(train_rows,val_rows,seed)

    direct_tr=crossfit_direct_curr_features(
        y[train_rows],EXPLICIT_LATEST_INTENT[train_rows],MODULE_READINESS[train_rows],
        TRAIN_GROUP[train_rows],seed+101
    )
    direct_bundle=fit_direct_curr_bundle(y[train_rows],EXPLICIT_LATEST_INTENT[train_rows])
    direct_va=apply_direct_curr_bundle(
        direct_bundle,EXPLICIT_LATEST_INTENT[val_rows],MODULE_READINESS[val_rows]
    )

    cooc_tr=crossfit_cooc_features(
        y[train_rows],anc["cons_train"],TRAIN_GROUP[train_rows],seed+202
    )
    cooc_bundle=fit_cooc_bundle(y[train_rows])
    cooc_va=apply_cooc_bundle(cooc_bundle,anc["cons_val"])

    Xtr,names,bounds=build_long_matrix_v30(
        train_rows,anc["cons_train"],anc["base_train"],anc["meta_train"],direct_tr,cooc_tr
    )
    Xva,_,_=build_long_matrix_v30(
        val_rows,anc["cons_val"],anc["base_val"],anc["meta_val"],direct_va,cooc_va
    )
    end=candidate_end(bounds,candidate)
    model,raw=fit_ranker_predict(
        Xtr[:,:end],Xva[:,:end],ORDINAL_GRADES[train_rows],len(train_rows),seed+303
    )
    score=blend_residual(anc["cons_val"],raw,ALPHA)
    delta=per_user_ndcg(y[val_rows],score)-per_user_ndcg(y[val_rows],anc["cons_val"])
    ci=cluster_boot_ci(delta,TRAIN_GROUP[val_rows],seed+404)

    row={
        "world":world,"candidate":candidate,"alpha":ALPHA,"n":len(val_rows),
        "anchor_ndcg":ndcg5(y[val_rows],anc["cons_val"]),
        "candidate_ndcg":ndcg5(y[val_rows],score),
        "mean_delta":float(delta.mean()),
        "positive_rate":float(np.mean(delta>0)),
        "harm_rate":float(np.mean(delta<0)),
        "ci_low":ci[0],"ci_high":ci[1],
        "top5_membership_change":top5_set_change_rate(anc["cons_val"],score),
        "top5_order_change":top5_order_change_rate(anc["cons_val"],score),
        "top1_change":top1_change_rate(anc["cons_val"],score),
    }
    del Xtr,Xva,direct_tr,direct_va,direct_bundle,cooc_tr,cooc_va,cooc_bundle,model,anc
    gc.collect()
    return row

sealed_rows=[]
if DEV_WINNER is not None:
    sealed_rows.append(evaluate_frozen_candidate(
        DEV_WINNER,SEALED_DEV_IDX,SEALED_LOCK_IDX,_seed_from_text("V30-SEALED-F-FIT"),"sealed_template_lock_F"
    ))
    sealed_rows.append(evaluate_frozen_candidate(
        DEV_WINNER,CHRONO_TR,CHRONO_VA,_seed_from_text("V30-CHRONO-FIT"),"chronological_U3001_U4000"
    ))
    V30_SEALED=pd.DataFrame(sealed_rows)
    lock_delta=float(V30_SEALED.loc[V30_SEALED.world=="sealed_template_lock_F","mean_delta"].iloc[0])
    chrono_delta=float(V30_SEALED.loc[V30_SEALED.world=="chronological_U3001_U4000","mean_delta"].iloc[0])
    SEALED_GATE=bool(lock_delta>0.0 and chrono_delta>=0.0)
else:
    V30_SEALED=pd.DataFrame([{
        "world":"skipped_no_development_winner","candidate":None,"alpha":ALPHA,"n":0,
        "anchor_ndcg":np.nan,"candidate_ndcg":np.nan,"mean_delta":np.nan,
        "positive_rate":np.nan,"harm_rate":np.nan,"ci_low":np.nan,"ci_high":np.nan,
        "top5_membership_change":np.nan,"top5_order_change":np.nan,"top1_change":np.nan,
    }])
    SEALED_GATE=False

display(V30_SEALED)
print("SEALED_GATE =",SEALED_GATE)

# ---- cell 27 (code) ----
# Transition-family audit is diagnostic only and never changes DEV_WINNER.
def promoted_demoted_signature(anchor,score):
    ra=rank_matrix(anchor); rb=rank_matrix(score)
    out=[]
    for i in range(len(anchor)):
        improve=ra[i]-rb[i]   # positive = promoted
        worsen=rb[i]-ra[i]    # positive = demoted
        if np.max(improve)<=0 or np.max(worsen)<=0:
            out.append("no_change")
            continue
        p=int(np.argmax(improve)); d=int(np.argmax(worsen))
        out.append(f"{MODULES[p]}>{MODULES[d]}")
    return np.asarray(out,dtype=object)

transition_rows=[]
for salt_name in FRESH_SALTS:
    for cand in CANDIDATES:
        sig=promoted_demoted_signature(SALT_ANCHOR[salt_name],SALT_SCORES[salt_name][cand])
        delta=SALT_DELTAS[salt_name][cand]
        for s in np.unique(sig):
            if s=="no_change":
                continue
            mask=sig==s
            transition_rows.append({
                "salt":salt_name,"candidate":cand,"transition":s,"support":int(mask.sum()),
                "mean_delta":float(delta[mask].mean()),
                "positive_rate":float(np.mean(delta[mask]>0)),
                "harm_rate":float(np.mean(delta[mask]<0)),
            })

V30_TRANSITIONS=pd.DataFrame(transition_rows)
if len(V30_TRANSITIONS):
    V30_TRANSITION_AGG=(
        V30_TRANSITIONS.groupby(["candidate","transition"],as_index=False)
        .agg(support=("support","sum"),mean_delta=("mean_delta","mean"),
             positive_rate=("positive_rate","mean"),harm_rate=("harm_rate","mean"),
             salts_seen=("salt","nunique"))
        .sort_values(["candidate","support"],ascending=[True,False])
    )
else:
    V30_TRANSITION_AGG=pd.DataFrame(columns=["candidate","transition","support","mean_delta","positive_rate","harm_rate","salts_seen"])

display(V30_TRANSITION_AGG.head(40))

# ---- cell 28 (markdown) ----
# ## Full-train deployment and fail-closed submission
#
# The selected candidate is fit once on all 4,000 labeled users. Its test correction is always exported as a diagnostic frozen file if a development winner exists.
#
# `submission_v30_primary.csv` uses that correction **only** when both the development gate and sealed rejection gate pass. Otherwise it is exactly the V25 0.65919 anchor.
#
# The eight V25 public-tested rows are overwritten with their exact anchor rows in every residual-LTR test candidate.

# ---- cell 29 (code) ----
OUT_DIR=Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("/mnt/data")
OUT_DIR.mkdir(parents=True,exist_ok=True)

# Full-train pair audit: descriptive only. Validation used cross-fitted priors.
FULL_DIRECT_BUNDLE=fit_direct_curr_bundle(y,EXPLICIT_LATEST_INTENT[:N_TRAIN])
V30_PAIR_AUDIT=direct_pair_audit(FULL_DIRECT_BUNDLE,EXPLICIT_LATEST_INTENT[:N_TRAIN],y)
display(V30_PAIR_AUDIT.sort_values(["smoothed_log_lift","support"],ascending=[False,False]))

if DEV_WINNER is not None:
    full_direct_tr=crossfit_direct_curr_features(
        y,EXPLICIT_LATEST_INTENT[:N_TRAIN],MODULE_READINESS[:N_TRAIN],TRAIN_GROUP,
        _seed_from_text("V30-FULL-DIRECT-CROSSFIT")
    )
    full_direct_te=apply_direct_curr_bundle(
        FULL_DIRECT_BUNDLE,EXPLICIT_LATEST_INTENT[N_TRAIN:],MODULE_READINESS[N_TRAIN:]
    )

    full_cooc_tr=crossfit_cooc_features(
        y,consensus3_oof,TRAIN_GROUP,_seed_from_text("V30-FULL-COOC-CROSSFIT")
    )
    FULL_COOC_BUNDLE=fit_cooc_bundle(y)
    full_cooc_te=apply_cooc_bundle(FULL_COOC_BUNDLE,consensus3_test)

    Xtr_full,names_full,bounds_full=build_long_matrix_v30(
        np.arange(N_TRAIN),consensus3_oof,base_oof,meta_oof,full_direct_tr,full_cooc_tr
    )
    Xte_full,_,_=build_long_matrix_v30(
        np.arange(N_TRAIN,N_TRAIN+N_TEST),consensus3_test,base_test,meta_test,full_direct_te,full_cooc_te
    )

    end=candidate_end(bounds_full,DEV_WINNER)
    full_model,raw_test=fit_ranker_predict(
        Xtr_full[:,:end],Xte_full[:,:end],ORDINAL_GRADES,N_TRAIN,_seed_from_text("V30-FULL-RANKER")
    )
    diagnostic_test=blend_residual(consensus3_test,raw_test,ALPHA)

    # Preserve all eight public-tested V25 rows exactly.
    protected_rows=[TEST_UID_TO_ROW[u] for u in sorted(PROTECTED_TEST_USERS)]
    diagnostic_test[protected_rows]=v25_anchor_test[protected_rows]
    for r in protected_rows:
        assert np.array_equal(diagnostic_test[r],v25_anchor_test[r])

    del Xtr_full,Xte_full,full_direct_tr,full_direct_te,full_cooc_tr,full_cooc_te,FULL_COOC_BUNDLE,full_model
    gc.collect()
else:
    diagnostic_test=v25_anchor_test.copy()

FINAL_GATE=bool(DEV_WINNER is not None and SEALED_GATE)
primary_test=diagnostic_test.copy() if FINAL_GATE else v25_anchor_test.copy()

assert sample["user_id"].astype(str).tolist()==test["user_id"].astype(str).tolist()
def submission_from_scores(scores):
    out=sample.copy()
    out.loc[:,MODULES]=np.asarray(scores,np.float32)
    assert np.isfinite(out[MODULES].to_numpy()).all()
    return out

sub_anchor=submission_from_scores(v25_anchor_test)
sub_diag=submission_from_scores(diagnostic_test)
sub_primary=submission_from_scores(primary_test)

sub_anchor.to_csv(OUT_DIR/"submission_v30_anchor_v25_65919.csv",index=False)
sub_diag.to_csv(OUT_DIR/"submission_v30_DIAGNOSTIC_frozen.csv",index=False)
sub_primary.to_csv(OUT_DIR/"submission_v30_primary.csv",index=False)

# Exact fail-close invariant.
if not FINAL_GATE:
    assert np.array_equal(sub_primary[MODULES].to_numpy(),sub_anchor[MODULES].to_numpy())

# Test change ledger relative to active V25 anchor.
a_top=top5_indices(v25_anchor_test); d_top=top5_indices(diagnostic_test)
changed=np.flatnonzero(np.any(a_top!=d_top,axis=1))
ledger=[]
for r in changed:
    ledger.append({
        "user_id":test.iloc[r].user_id,
        "anchor_top5":" ".join(MODULES[j] for j in a_top[r]),
        "diagnostic_top5":" ".join(MODULES[j] for j in d_top[r]),
        "top1_changed":bool(a_top[r,0]!=d_top[r,0]),
        "membership_changed":bool(set(a_top[r])!=set(d_top[r])),
        "protected_v25_user":bool(test.iloc[r].user_id in PROTECTED_TEST_USERS),
    })
V30_TEST_LEDGER=pd.DataFrame(ledger)
print("diagnostic changed users vs V25:",len(V30_TEST_LEDGER))
print("FINAL_GATE =",FINAL_GATE)

# ---- cell 30 (code) ----
# Research + submission artifacts.
V30_THREE_SALT.to_csv(OUT_DIR/"v30_three_salt_summary.csv",index=False)
V30_POOLED.to_csv(OUT_DIR/"v30_pooled_robustness.csv",index=False)
V30_INCREMENTAL.to_csv(OUT_DIR/"v30_incremental_blocks.csv",index=False)
V30_CANDIDATE_GATE.to_csv(OUT_DIR/"v30_candidate_gate.csv",index=False)
V30_SEALED.to_csv(OUT_DIR/"v30_sealed_validation.csv",index=False)
V30_TRANSITIONS.to_csv(OUT_DIR/"v30_transition_observations.csv",index=False)
V30_TRANSITION_AGG.to_csv(OUT_DIR/"v30_transition_family_audit.csv",index=False)
V30_PAIR_AUDIT.to_csv(OUT_DIR/"v30_direct_curriculum_pair_audit.csv",index=False)
V30_TEST_LEDGER.to_csv(OUT_DIR/"v30_test_change_ledger.csv",index=False)

slot_plan=pd.DataFrame([
    {
        "file":"submission_v30_primary.csv",
        "role":"PRIMARY_ONLY_IF_ALL_GATES_PASS",
        "eligible_to_submit":bool(FINAL_GATE),
        "note":"Candidate if development + sealed gates pass; otherwise exact V25 anchor.",
    },
    {
        "file":"submission_v30_DIAGNOSTIC_frozen.csv",
        "role":"DIAGNOSTIC_ONLY",
        "eligible_to_submit":False,
        "note":"Frozen development winner on test; never submit automatically if sealed gate fails.",
    },
    {
        "file":"submission_v30_anchor_v25_65919.csv",
        "role":"ANCHOR_REFERENCE",
        "eligible_to_submit":False,
        "note":"Exact current public anchor.",
    },
])
slot_plan.to_csv(OUT_DIR/"v30_slot_plan.csv",index=False)

summary={
    "version":"V30 3-Fresh-Salt Split-Robust Pair-Conditional Curriculum Residual LTR",
    "public_anchor":PUBLIC_CURRENT_PB,
    "leader_reference":PUBLIC_LEADER_LB,
    "fixed_alpha":ALPHA,
    "fresh_salts":FRESH_SALTS,
    "candidates":CANDIDATES,
    "development_winner":DEV_WINNER,
    "development_gate_passed":bool(DEV_WINNER is not None),
    "sealed_gate_passed":bool(SEALED_GATE),
    "final_gate_passed":bool(FINAL_GATE),
    "candidate_gate":V30_CANDIDATE_GATE.to_dict("records"),
    "incremental_blocks":V30_INCREMENTAL.to_dict("records"),
    "sealed_validation":V30_SEALED.to_dict("records"),
    "diagnostic_test_changed_users":int(len(V30_TEST_LEDGER)),
    "submission_instruction":(
        "PRIMARY IS SUBMISSION-ELIGIBLE: still audit the run before Kaggle submission."
        if FINAL_GATE else
        "NO NEW SUBMISSION: primary is exact V25 fail-close anchor."
    ),
}
with open(OUT_DIR/"v30_summary.json","w") as f:
    json.dump(summary,f,indent=2)

# Final invariants.
assert set(V30_THREE_SALT.salt)==set(FRESH_SALTS)
assert set(V30_THREE_SALT.candidate)==set(CANDIDATES)
assert len(V30_THREE_SALT)==len(FRESH_SALTS)*len(CANDIDATES)
assert np.isfinite(V30_THREE_SALT.mean_delta.to_numpy()).all()
assert V30_CANDIDATE_GATE.candidate.nunique()==3
assert sub_primary.shape==sample.shape
assert sub_diag.shape==sample.shape
assert sub_anchor.shape==sample.shape

print(json.dumps(summary,indent=2))
print("\nV30 outputs written to",OUT_DIR)
print(slot_plan.to_string(index=False))

# ---- cell 31 (markdown) ----
# ### Interpretation contract
#
# - V29/V29.5 scores are historical context only and must not be merged into the V30 pass/fail gate.
# - `C/D/E` are development replication worlds; do not add new salts if the result is inconvenient.
# - Salt F and chronology are rejection-only; never switch candidates after seeing them.
# - `submission_v30_primary.csv` is worth considering only if `final_gate_passed == true`.
# - If V30 fails, do **not** create a V30.1 threshold search on the same surfaces. Treat the DirectCurr/LTR escalation as rejected and return to a different hypothesis family.
