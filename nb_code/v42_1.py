# Converted from notebooks/v42_1.ipynb: markdown cells become comments, %/! magics neutralized.

# ---- cell 0 (markdown) ----
# # MineToday V42.1 — Champion-Anchored Support-Shrunk Generator-Exclusive Head Ranker
#
# **Purpose.** Test the one remaining generator-state hypothesis exposed by V42 without repeating its architecture mismatches: train and deploy on the same Top4 policy surface, isolate only the incremental high-cardinality generator signal (`GENCAT − CONTROL`), and shrink that residual by fold-safe empirical-Bayes support reliability.
#
# **Frozen production geometry.** Exact public champion = **0.66028** (`current066020 + B34 + B45`). Test deployment is S0-only, keeps S1/S2/S3 and protected rows bit-exact, freezes Top5 membership and rank #5, and may only permute the four score values already present in the anchor Top4.
#
# **No post-result tuning.** CatBoost hyperparameters, alpha `0.16`, split, gates, support-shrinkage estimator, and deployment rule are fixed in this notebook. If DESIGN or LOCK fails, `submission_v42_1_primary.csv` is the exact 0.66028 fail-close anchor.
#
# **Runtime target:** Kaggle GPU **T4 x2**. CatBoost uses both GPUs when visible; the frozen LightGBM reconstruction remains CPU-first for deterministic geometry.

# ---- cell 1 (code) ----
# Runtime + preregistered V39.5 S3 boundary-decomposition configuration
import os, sys, re, json, ast, html, hashlib, random, warnings, time, subprocess, gc, math
from pathlib import Path
from collections import defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ["PYTHONHASHSEED"] = "42"

V421_SKIP_INSTALL = bool(int(os.getenv("V421_SKIP_INSTALL", "0")))
V395_SKIP_INSTALL = V421_SKIP_INSTALL
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
V421_SMOKE = bool(int(os.getenv("V421_SMOKE", "0")))
V395_SMOKE = V421_SMOKE
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
print("V421_SMOKE", V421_SMOKE, "fixed alpha", ALPHA)
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
# ## V42.1 frozen experiment configuration
#
# This is a **representation/deployment correction**, not a hyperparameter sweep. DESIGN/LOCK stays identical to V42 so the 600-user LOCK remains untouched until the new architecture clears DESIGN.

# ---- cell 25 (code) ----
try:
    from catboost import CatBoostRanker, Pool
    import catboost
except Exception:
    if V421_SKIP_INSTALL:
        raise
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "catboost==1.2.8"])
    from catboost import CatBoostRanker, Pool
    import catboost

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
import shutil

V421_VERSION = "V42.1 Champion-Anchored Support-Shrunk Generator-Exclusive Head Ranker"
PUBLIC_V421_ANCHOR_LB = 0.66028
EXPECTED_V421_ANCHOR_TOP5_SHA256 = "7c0d29a7309b7283a043b7d37ca5556ba7c0ca2270b0137a6e1e0878a69a5fe7"

# Preserve V42's split: LOCK has never been evaluated.
V421_SPLIT_SEED = 2026090342
V421_FOLD_SEED = 2026090343
V421_BOOT_SEED = 20260903451
V421_ALPHA = 0.16
POLICY_INNER_FOLDS = 2 if V421_SMOKE else 4

DESIGN_SIZE = 600 if V421_SMOKE else 3400
LOCK_SIZE = 120 if V421_SMOKE else 600
OUTER_FOLDS = 2 if V421_SMOKE else 5
BLOCKS = 2 if V421_SMOKE else 5
BOOTSTRAPS_V421 = 80 if V421_SMOKE else 2500

# Identical CONTROL / GENCAT model capacity. Only the generator-specific block differs.
CAT_ITERS = 25 if V421_SMOKE else 180
CAT_LR = 0.08 if V421_SMOKE else 0.035
CAT_DEPTH = 5
CAT_L2 = 11.0
CAT_RANDOM_STRENGTH = 0.30

# Pre-registered gates from the V42 diagnosis.
DESIGN_MIN_DELTA = -1.0 if V421_SMOKE else 0.00030
DESIGN_MIN_FOLD_WINS = 0 if V421_SMOKE else 4
DESIGN_MIN_BLOCK_WINS = 0 if V421_SMOKE else 4
DESIGN_BOOT_CI_FLOOR = -1.0 if V421_SMOKE else -0.00005
DESIGN_WORST_BLOCK_FLOOR = -1.0 if V421_SMOKE else -0.00010
COLD_TEMPLATE_FLOOR = -1.0 if V421_SMOKE else -0.00010
LOCK_MIN_DELTA = -1.0 if V421_SMOKE else 0.00020
LOCK_MIN_BLOCK_WINS = 0 if V421_SMOKE else 3
LOCK_BOOT_CI_FLOOR = -1.0 if V421_SMOKE else -0.00020
TEST_MAX_CHANGED = 10_000 if V421_SMOKE else 260
TEST_MAX_TOP1_CHANGED = 10_000 if V421_SMOKE else 45

V421_OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("/mnt/data")
V421_OUT.mkdir(parents=True, exist_ok=True)

V421_ANCHOR_TEST = np.asarray(CANDIDATE_SCORES["combo_B34_B45"], np.float32).copy()
if not V421_SMOKE:
    assert top5_sha256(V421_ANCHOR_TEST) == EXPECTED_V421_ANCHOR_TOP5_SHA256

PROTECTED_TEST_USERS = set([u for u,_,_ in KNOWN_PB_SWAPS] + [a["user_id"] for a in V25_CORE_ACTIONS])
TEST_PROTECTED = np.asarray([str(u) in PROTECTED_TEST_USERS for u in test.user_id], bool)
TEST_ELIGIBLE = (full_state == 0) & (~TEST_PROTECTED)

print("catboost", catboost.__version__)
print("V42.1 anchor hash", top5_sha256(V421_ANCHOR_TEST))
print("public anchor", PUBLIC_V421_ANCHOR_LB, "leader gap", round(PUBLIC_LEADER_LB-PUBLIC_V421_ANCHOR_LB,5))
print("test S0 eligible after frozen protection:", int(TEST_ELIGIBLE.sum()), "/", N_TEST)
try:
    print("GPU runtime:\n" + subprocess.check_output(["nvidia-smi", "-L"], text=True).strip())
except Exception:
    print("GPU runtime not visible. Notebook remains runnable on Kaggle T4x2.")

# ---- cell 26 (markdown) ----
# ## Generator state and support hierarchy
#
# The raw generator state is label-free. Target-dependent empirical-Bayes statistics are created later **inside each outer training partition** with leave-one-out features for its training rows and frozen mappings for validation/test.

# ---- cell 27 (code) ----
N_ALL = N_TRAIN + N_TEST
ALL_INTENT = np.concatenate([np.asarray(TRAIN_INTENT, dtype=object), np.asarray(TEST_INTENT, dtype=object)])

latest_tpl_cat = np.full(N_ALL, "__NO_CHAT__", dtype=object)
prev_tpl_cat = np.full(N_ALL, "__NO_PREV__", dtype=object)
tpl_transition_cat = np.full(N_ALL, "__NO_PREV__>>__NO_CHAT__", dtype=object)
latest_topic_cat = np.full(N_ALL, "none", dtype=object)
prev_topic_cat = np.full(N_ALL, "none", dtype=object)
topic_journey_cat = np.full(N_ALL, "none", dtype=object)
latest_style_cat = np.full(N_ALL, "none", dtype=object)
hist_bucket_cat = np.full(N_ALL, "n0", dtype=object)
chat_count_all = np.zeros(N_ALL, np.float32)
learning_count_all = np.zeros(N_ALL, np.float32)
noise_count_all = np.zeros(N_ALL, np.float32)
span_days_all = np.zeros(N_ALL, np.float32)
latest_gap_days_all = np.full(N_ALL, 999.0, np.float32)
MODULE_AGE_DAYS = np.full((N_ALL, N_MODULES), 999.0, np.float32)

def topic_sig(topics):
    vals = sorted(set(map(str, topics)))
    return "|".join(vals) if vals else "none"

def history_bucket(n):
    if n <= 0: return "n0"
    if n == 1: return "n1"
    if n == 2: return "n2"
    if n <= 4: return "n3_4"
    return "n5plus"

def styles_for_row(row):
    z = [sty for sty in STYLES if int(getattr(row, "sty_"+sty, 0))]
    return "|".join(z) if z else "none"

for u in all_users:
    gi = uid2idx[u]
    g = user_chat.get(u)
    if g is None or len(g) == 0:
        continue
    learn = g[~g["is_noise"]].copy()
    if len(learn) == 0:
        learn = g.copy()
    rows = list(learn.itertuples(index=False))
    n = len(rows)
    chat_count_all[gi] = float(len(g))
    learning_count_all[gi] = float(n)
    noise_count_all[gi] = float(g["is_noise"].sum())
    hist_bucket_cat[gi] = history_bucket(n)
    if len(g) >= 2 and g["timestamp"].notna().sum() >= 2:
        span_days_all[gi] = float((g["timestamp"].max()-g["timestamp"].min()).total_seconds()/86400.0)
    if n:
        last = rows[-1]
        latest_tpl_cat[gi] = template_token(str(last.skeleton))
        latest_topic_cat[gi] = topic_sig(last.topics)
        latest_style_cat[gi] = styles_for_row(last)
        if n >= 2:
            prev = rows[-2]
            prev_tpl_cat[gi] = template_token(str(prev.skeleton))
            prev_topic_cat[gi] = topic_sig(prev.topics)
            tpl_transition_cat[gi] = prev_tpl_cat[gi] + ">>" + latest_tpl_cat[gi]
            if pd.notna(last.timestamp) and pd.notna(prev.timestamp):
                latest_gap_days_all[gi] = max(0.0, float((last.timestamp-prev.timestamp).total_seconds()/86400.0))
        else:
            tpl_transition_cat[gi] = "__NO_PREV__>>" + latest_tpl_cat[gi]
        topic_journey_cat[gi] = ">>".join(topic_sig(r.topics) for r in rows[-3:])

        t_last = last.timestamp if pd.notna(last.timestamp) else None
        for m,mid in enumerate(MODULES):
            topic_names = set(MODULE_TOPIC_MAP.get(mid, []))
            hit_time = None
            for r in reversed(rows):
                if topic_names.intersection(set(r.topics)):
                    hit_time = r.timestamp if pd.notna(r.timestamp) else None
                    break
            if hit_time is not None and t_last is not None:
                MODULE_AGE_DAYS[gi,m] = max(0.0, float((t_last-hit_time).total_seconds()/86400.0))

ASSESS_MEAN = A.mean(1).astype(np.float32)
ASSESS_STD = A.std(1).astype(np.float32)
ASSESS_INVALID_N = A_INVALID.sum(1).astype(np.float32)
MODULE_LEVEL_CAT = modules.set_index("module_id")["prerequisite_level"].astype(str).reindex(MODULES).fillna("unknown").to_numpy(object)

# Label-free full-train support audit only; model features are fitted per fold below.
full_support = pd.DataFrame({
    "latest_tpl": latest_tpl_cat[:N_TRAIN],
    "journey": topic_journey_cat[:N_TRAIN],
    "transition": tpl_transition_cat[:N_TRAIN],
})
V421_TEST_SUPPORT_AUDIT = pd.DataFrame({
    "user_id": test.user_id.astype(str),
    "latest_tpl_support_fulltrain": pd.Series(latest_tpl_cat[N_TRAIN:]).map(pd.Series(latest_tpl_cat[:N_TRAIN]).value_counts()).fillna(0).astype(int),
    "journey_support_fulltrain": pd.Series(topic_journey_cat[N_TRAIN:]).map(pd.Series(topic_journey_cat[:N_TRAIN]).value_counts()).fillna(0).astype(int),
    "transition_support_fulltrain": pd.Series(tpl_transition_cat[N_TRAIN:]).map(pd.Series(tpl_transition_cat[:N_TRAIN]).value_counts()).fillna(0).astype(int),
})
print("test latest-template seen/full-train:", round(float((V421_TEST_SUPPORT_AUDIT.latest_tpl_support_fulltrain>0).mean()),4))
print("test latest-template support>=5:", round(float((V421_TEST_SUPPORT_AUDIT.latest_tpl_support_fulltrain>=5).mean()),4))

# ---- cell 28 (markdown) ----
# ## Policy-matched structural anchor for both train and validation
#
# V42's key mismatch is removed here: outer-train rows receive an **OOF V30/V31/current+B34+B45 analogue**, not Consensus3. Validation receives the same policy family trained on the outer-train partition. No validation label enters either surface.

# ---- cell 29 (code) ----
def _long_rows(user_local_idx, nmods=N_MODULES):
    u = np.asarray(user_local_idx, int)
    return (u[:,None]*int(nmods) + np.arange(int(nmods))[None,:]).reshape(-1)

def _apply_current_b3445(v30, v31, state):
    current, info = current066020_surface(v30, v31, state)
    t30 = top5_indices(v30); t31 = top5_indices(v31)
    for r in np.flatnonzero(info["order_only"]):
        try:
            boundary, *_ = classify_single_adjacent_order_swap(t30[r], t31[r])
        except AssertionError:
            continue
        if boundary in {"B34", "B45"}:
            current[r] = v31[r]
    return current.astype(np.float32), info

def _oof_ranker_predictions(Xlong, grades, user_y, groups, base_seed, feature_end):
    n = len(user_y)
    pred = np.zeros((n, N_MODULES), np.float32)
    folds = make_group_folds_local(user_y, groups, POLICY_INNER_FOLDS, int(base_seed))
    for k,(tr,va) in enumerate(folds):
        model = make_ranker(int(base_seed)+701+k)
        model.fit(
            Xlong[_long_rows(tr), :feature_end],
            grades[tr].reshape(-1).astype(np.int32),
            group=np.full(len(tr), N_MODULES, dtype=np.int32),
        )
        pred[va] = model.predict(Xlong[_long_rows(va), :feature_end]).reshape(len(va), N_MODULES).astype(np.float32)
        del model
    return pred

def build_policy_matched_split_v421(train_rows, val_rows, seed):
    train_rows = np.asarray(train_rows, int)
    val_rows = np.asarray(val_rows, int)
    anc = build_anchor_split_full(train_rows, val_rows, int(seed))

    intent_tr = crossfit_intentstate_features(
        y[train_rows], ORDINAL_GRADES[train_rows], TRAIN_INTENT[train_rows],
        MODULE_READY_BAND[train_rows], TRAIN_GROUP[train_rows], int(seed)+71
    )
    bundle = fit_intentstate_bundle(
        y[train_rows], ORDINAL_GRADES[train_rows], TRAIN_INTENT[train_rows], MODULE_READY_BAND[train_rows]
    )
    intent_va = apply_intentstate_bundle(bundle, ALL_INTENT[val_rows], MODULE_READY_BAND[val_rows])

    Xtr, _, bounds = build_long_matrix_v31(train_rows, anc["cons_train"], anc["base_train"], anc["meta_train"], intent_tr)
    Xva, _, bounds2 = build_long_matrix_v31(val_rows, anc["cons_val"], anc["base_val"], anc["meta_val"], intent_va)
    assert bounds == bounds2
    be = bounds["base"]; ie = bounds["intentstate"]

    # OOF structural policy for training rows: same family as deployment.
    raw30_tr = _oof_ranker_predictions(
        Xtr, ORDINAL_GRADES[train_rows], y[train_rows], TRAIN_GROUP[train_rows], int(seed)+1301, be
    )
    raw31_tr = _oof_ranker_predictions(
        Xtr, ORDINAL_GRADES[train_rows], y[train_rows], TRAIN_GROUP[train_rows], int(seed)+2301, ie
    )
    v30_tr = blend_residual(anc["cons_train"], raw30_tr, ALPHA)
    v31_tr = blend_residual(anc["cons_train"], raw31_tr, ALPHA)
    state_tr = state_codes(anc["cons_train"], v30_tr, v31_tr)
    current_tr, _ = _apply_current_b3445(v30_tr, v31_tr, state_tr)

    # Full outer-train structural policy for validation/test rows.
    _, raw30_va = fit_ranker_predict(Xtr[:,:be], Xva[:,:be], ORDINAL_GRADES[train_rows], len(train_rows), int(seed)+3301)
    _, raw31_va = fit_ranker_predict(Xtr[:,:ie], Xva[:,:ie], ORDINAL_GRADES[train_rows], len(train_rows), int(seed)+4301)
    v30_va = blend_residual(anc["cons_val"], raw30_va, ALPHA)
    v31_va = blend_residual(anc["cons_val"], raw31_va, ALPHA)
    state_va = state_codes(anc["cons_val"], v30_va, v31_va)
    current_va, _ = _apply_current_b3445(v30_va, v31_va, state_va)

    out = {
        "train_anchor": current_tr,
        "val_anchor": current_va,
        "train_state": state_tr.astype(np.int8),
        "val_state": state_va.astype(np.int8),
        "base_train": anc["base_train"], "meta_train": anc["meta_train"],
        "base_val": anc["base_val"], "meta_val": anc["meta_val"],
    }
    del Xtr, Xva, intent_tr, intent_va, bundle, anc, raw30_tr, raw31_tr, raw30_va, raw31_va
    gc.collect()
    return out

print("V42.1 policy-matched OOF train/validation anchor helper ready")

# ---- cell 30 (markdown) ----
# ## Hierarchical empirical-Bayes generator reliability
#
# Parent = `latest_topic × module`. Children = exact latest-template, topic-journey, and template-transition × module. Shrinkage strength is estimated by method-of-moments from the **training partition only**. Training features are leave-one-out; validation/test use frozen training aggregates. No support threshold is tuned.

# ---- cell 31 (code) ----
GAIN_BY_CLASS = np.asarray([0.00,0.25,0.40,0.55,0.70,0.85,1.00], np.float32)
Y_GAIN = GAIN_BY_CLASS[np.asarray(ORDINAL_GRADES, dtype=np.int16)]
GEN_FAMILIES = ("tpl", "journey", "transition")

def _head_top4(anchor):
    return np.argsort(-np.asarray(anchor), axis=1, kind="mergesort")[:,:4]

def _head_gather(mat, top4):
    mat = np.asarray(mat)
    return np.take_along_axis(mat, np.asarray(top4,int), axis=1)

def _head_keys(global_rows, top4):
    global_rows = np.asarray(global_rows, int); top4=np.asarray(top4,int)
    module = np.asarray(MODULES,object)[top4].reshape(-1)
    ltopic = np.repeat(latest_topic_cat[global_rows], 4)
    tpl = np.repeat(latest_tpl_cat[global_rows], 4)
    journey = np.repeat(topic_journey_cat[global_rows], 4)
    transition = np.repeat(tpl_transition_cat[global_rows], 4)
    return {
        "module":module,
        "parent":np.asarray([f"{t}|{m}" for t,m in zip(ltopic,module)],object),
        "tpl":np.asarray([f"{g}|{m}" for g,m in zip(tpl,module)],object),
        "journey":np.asarray([f"{g}|{m}" for g,m in zip(journey,module)],object),
        "transition":np.asarray([f"{g}|{m}" for g,m in zip(transition,module)],object),
        "tpl_group":tpl, "journey_group":journey, "transition_group":transition,
    }

def _agg(keys, values):
    d = pd.DataFrame({"k":np.asarray(keys,dtype=object),"y":np.asarray(values,float)})
    g = d.groupby("k", sort=False)["y"].agg(["count","sum","mean"])
    return g

def _estimate_kappa(child_keys, parent_keys, yv, child_agg, parent_agg):
    child_keys=np.asarray(child_keys,dtype=object); parent_keys=np.asarray(parent_keys,dtype=object); yv=np.asarray(yv,float)
    cmean = pd.Series(child_keys).map(child_agg["mean"]).to_numpy(float)
    resid = yv - cmean
    sigma2 = float(np.mean(resid*resid))
    # one representative parent per child key
    rep = pd.DataFrame({"child":child_keys,"parent":parent_keys}).drop_duplicates("child").set_index("child")["parent"]
    tab = child_agg.copy()
    parent_for_child = rep.reindex(tab.index).astype(str)
    pmean = parent_for_child.map(parent_agg["mean"]).to_numpy(float)
    delta2 = (tab["mean"].to_numpy(float)-pmean)**2
    noise = sigma2/np.maximum(tab["count"].to_numpy(float),1.0)
    w = tab["count"].to_numpy(float)
    tau2 = float(np.average(np.maximum(delta2-noise,0.0),weights=np.maximum(w,1.0))) if len(tab) else 0.0
    kappa = sigma2/max(tau2,1e-6)
    return float(np.clip(kappa,1.0,200.0)), sigma2, tau2

def fit_generator_eb_bundle(train_rows, train_anchor):
    train_rows=np.asarray(train_rows,int); top4=_head_top4(train_anchor)
    keys=_head_keys(train_rows,top4)
    ylong=_head_gather(Y_GAIN[train_rows],top4).reshape(-1).astype(float)

    module_agg=_agg(keys["module"],ylong)
    parent_agg=_agg(keys["parent"],ylong)
    child_aggs={fam:_agg(keys[fam],ylong) for fam in GEN_FAMILIES}
    kappas={}; moments={}
    for fam in GEN_FAMILIES:
        k,s2,t2=_estimate_kappa(keys[fam],keys["parent"],ylong,child_aggs[fam],parent_agg)
        kappas[fam]=k; moments[fam]={"sigma2":s2,"tau2":t2}

    group_counts={fam:pd.Series(keys[fam+"_group"]).value_counts().to_dict() for fam in GEN_FAMILIES}
    bundle={"module_agg":module_agg,"parent_agg":parent_agg,"child_aggs":child_aggs,
            "kappas":kappas,"moments":moments,"group_counts":group_counts}

    # Leave-one-out training features. No row sees its own target in the statistic.
    n=len(ylong); out={}
    global_mean=float(ylong.mean())
    module_count=pd.Series(keys["module"]).map(module_agg["count"]).to_numpy(float)-1.0
    module_sum=pd.Series(keys["module"]).map(module_agg["sum"]).to_numpy(float)-ylong
    module_mean=np.where(module_count>0,module_sum/np.maximum(module_count,1.0),global_mean)
    parent_count=pd.Series(keys["parent"]).map(parent_agg["count"]).to_numpy(float)-1.0
    parent_sum=pd.Series(keys["parent"]).map(parent_agg["sum"]).to_numpy(float)-ylong
    parent_mean=np.where(parent_count>0,parent_sum/np.maximum(parent_count,1.0),module_mean)

    for fam in GEN_FAMILIES:
        agg=child_aggs[fam]; kappa=kappas[fam]
        cnt=pd.Series(keys[fam]).map(agg["count"]).to_numpy(float)-1.0
        sm=pd.Series(keys[fam]).map(agg["sum"]).to_numpy(float)-ylong
        raw=np.where(cnt>0,sm/np.maximum(cnt,1.0),parent_mean)
        w=np.maximum(cnt,0.0)/(np.maximum(cnt,0.0)+kappa)
        eb=parent_mean+w*(raw-parent_mean)
        gsupport=np.asarray([max(0,int(group_counts[fam].get(str(g),0))-1) for g in keys[fam+"_group"]],float)
        out[f"{fam}_support"]=np.maximum(cnt,0.0).astype(np.float32)
        out[f"{fam}_support_log"]=np.log1p(np.maximum(cnt,0.0)).astype(np.float32)
        out[f"{fam}_group_support"]=gsupport.astype(np.float32)
        out[f"{fam}_weight"]=w.astype(np.float32)
        out[f"{fam}_eb"]=eb.astype(np.float32)
        out[f"{fam}_eb_delta"]=(eb-parent_mean).astype(np.float32)
    train_df=pd.DataFrame(out)
    return bundle, train_df

def apply_generator_eb_bundle(bundle, global_rows, anchor):
    global_rows=np.asarray(global_rows,int); top4=_head_top4(anchor); keys=_head_keys(global_rows,top4)
    module_agg=bundle["module_agg"]; parent_agg=bundle["parent_agg"]
    global_mean=float(module_agg["sum"].sum()/max(module_agg["count"].sum(),1))
    module_mean=pd.Series(keys["module"]).map(module_agg["mean"]).fillna(global_mean).to_numpy(float)
    parent_mean=pd.Series(keys["parent"]).map(parent_agg["mean"]).to_numpy(float)
    parent_mean=np.where(np.isfinite(parent_mean),parent_mean,module_mean)
    out={}
    for fam in GEN_FAMILIES:
        agg=bundle["child_aggs"][fam]; kappa=bundle["kappas"][fam]
        cnt=pd.Series(keys[fam]).map(agg["count"]).fillna(0).to_numpy(float)
        raw=pd.Series(keys[fam]).map(agg["mean"]).to_numpy(float)
        raw=np.where(np.isfinite(raw),raw,parent_mean)
        w=cnt/(cnt+kappa)
        eb=parent_mean+w*(raw-parent_mean)
        gcnt=bundle["group_counts"][fam]
        gsupport=np.asarray([int(gcnt.get(str(g),0)) for g in keys[fam+"_group"]],float)
        out[f"{fam}_support"]=cnt.astype(np.float32)
        out[f"{fam}_support_log"]=np.log1p(cnt).astype(np.float32)
        out[f"{fam}_group_support"]=gsupport.astype(np.float32)
        out[f"{fam}_weight"]=w.astype(np.float32)
        out[f"{fam}_eb"]=eb.astype(np.float32)
        out[f"{fam}_eb_delta"]=(eb-parent_mean).astype(np.float32)
    return pd.DataFrame(out)

print("V42.1 empirical-Bayes support shrinkage ready")

# ---- cell 32 (markdown) ----
# ## Top4-only CONTROL and GENCAT rankers
#
# Both models see the same policy-matched Top4 candidates and the same capacity. CONTROL contains structural/broad context. GENCAT adds only high-cardinality generator identity/crosses plus fold-safe support-shrunk EB features. Training groups contain exactly four candidates because that is the only surface deployment may reorder.

# ---- cell 33 (code) ----
def age_bin421(x):
    x=np.asarray(x,float)
    out=np.full(x.shape,"d60plus",dtype=object)
    out[x>=998]="never"; out[x<=60]="d15_60"; out[x<=14]="d4_14"; out[x<=3]="d0_3"; out[x<=.01]="current"
    return out

CONTROL_CAT = [
    "module","module_level","latest_topic","previous_topic","latest_style","hist_bucket","age_bin","readiness_band",
    "module_topic","module_readiness","module_agebin",
]
GEN_EXTRA_CAT = ["latest_tpl","topic_journey","tpl_transition","module_tpl","module_journey","module_transition"]
GEN_CAT = CONTROL_CAT + GEN_EXTRA_CAT

def build_v421_top4_frame(global_rows, anchor, base, meta, eb_df, generator):
    global_rows=np.asarray(global_rows,int); anchor=np.asarray(anchor,np.float32); base=np.asarray(base,np.float32); meta=np.asarray(meta,np.float32)
    n=len(global_rows); assert anchor.shape==base.shape==meta.shape==(n,N_MODULES)
    top4=_head_top4(anchor); hs=_head_gather(anchor,top4).astype(np.float32)
    hb=_head_gather(base,top4).astype(np.float32); hm=_head_gather(meta,top4).astype(np.float32)
    cand=top4.reshape(-1); module=np.asarray(MODULES,object)[cand]; mlevel=MODULE_LEVEL_CAT[cand]
    rank=np.tile(np.arange(4,dtype=np.float32),n)
    z=((hs-hs.mean(1,keepdims=True))/np.maximum(hs.std(1,keepdims=True),1e-5)).reshape(-1).astype(np.float32)
    gap_top1=(hs[:,0:1]-hs).reshape(-1).astype(np.float32)
    prev=np.zeros_like(hs); nxt=np.zeros_like(hs); gaps=hs[:,:-1]-hs[:,1:]
    prev[:,1:]=gaps; nxt[:,:-1]=gaps

    gr=global_rows[:,None]
    ltopic=np.repeat(latest_topic_cat[global_rows],4); ptopic=np.repeat(prev_topic_cat[global_rows],4)
    sty=np.repeat(latest_style_cat[global_rows],4); hbcat=np.repeat(hist_bucket_cat[global_rows],4)
    age=_head_gather(MODULE_AGE_DAYS[global_rows],top4).reshape(-1); ageb=age_bin421(age)
    rb=_head_gather(MODULE_READY_BAND[global_rows],top4).reshape(-1).astype(str)

    data={
        "anchor_score":hs.reshape(-1),"anchor_rank":rank/3.0,"anchor_z":z,"top1_gap":gap_top1,
        "prev_gap":prev.reshape(-1),"next_gap":nxt.reshape(-1),
        "base_score":hb.reshape(-1),"meta_score":hm.reshape(-1),"meta_minus_base":(hm-hb).reshape(-1),
        "lex_word":_head_gather(LEX_WORD[global_rows],top4).reshape(-1),
        "lex_char":_head_gather(LEX_CHAR[global_rows],top4).reshape(-1),
        "topic_count":_head_gather(MODULE_COUNT[global_rows],top4).reshape(-1),
        "topic_recency":_head_gather(MODULE_RECENCY[global_rows],top4).reshape(-1),
        "topic_latest":_head_gather(MODULE_LATEST[global_rows],top4).reshape(-1),
        "readiness":_head_gather(MODULE_READINESS[global_rows],top4).reshape(-1),
        "invalid_readiness":_head_gather(MODULE_INVALID[global_rows],top4).reshape(-1),
        "assess_mean":np.repeat(ASSESS_MEAN[global_rows],4),"assess_std":np.repeat(ASSESS_STD[global_rows],4),
        "assess_invalid_n":np.repeat(ASSESS_INVALID_N[global_rows],4),"chat_count":np.repeat(chat_count_all[global_rows],4),
        "learning_count":np.repeat(learning_count_all[global_rows],4),"noise_count":np.repeat(noise_count_all[global_rows],4),
        "chat_span_days":np.repeat(span_days_all[global_rows],4),"latest_gap_days":np.repeat(latest_gap_days_all[global_rows],4),
        "module":module,"module_level":mlevel,"latest_topic":ltopic,"previous_topic":ptopic,"latest_style":sty,"hist_bucket":hbcat,
        "age_bin":ageb,"readiness_band":rb,
        "module_topic":np.asarray([f"{m}|{x}" for m,x in zip(module,ltopic)],object),
        "module_readiness":np.asarray([f"{m}|{x}" for m,x in zip(module,rb)],object),
        "module_agebin":np.asarray([f"{m}|{x}" for m,x in zip(module,ageb)],object),
    }
    if generator:
        lt=np.repeat(latest_tpl_cat[global_rows],4); journey=np.repeat(topic_journey_cat[global_rows],4); trans=np.repeat(tpl_transition_cat[global_rows],4)
        data.update(
            latest_tpl=lt,topic_journey=journey,tpl_transition=trans,
            module_tpl=np.asarray([f"{m}|{x}" for m,x in zip(module,lt)],object),
            module_journey=np.asarray([f"{m}|{x}" for m,x in zip(module,journey)],object),
            module_transition=np.asarray([f"{m}|{x}" for m,x in zip(module,trans)],object),
        )
        assert len(eb_df)==4*n
        for c in eb_df.columns:
            data[c]=eb_df[c].to_numpy(np.float32)
    df=pd.DataFrame(data)
    cats=GEN_CAT if generator else CONTROL_CAT
    for c in cats: df[c]=df[c].fillna("none").astype(str)
    nums=[c for c in df.columns if c not in cats]
    df[nums]=df[nums].astype(np.float32)
    assert np.isfinite(df[nums].to_numpy(np.float32)).all()
    return df,cats,top4

def make_v421_ranker(seed):
    params=dict(
        loss_function="YetiRank",eval_metric="NDCG:top=4",iterations=CAT_ITERS,learning_rate=CAT_LR,depth=CAT_DEPTH,
        l2_leaf_reg=CAT_L2,random_strength=CAT_RANDOM_STRENGTH,random_seed=int(seed),one_hot_max_size=32,max_ctr_complexity=1,
        allow_writing_files=False,verbose=False,
    )
    if (not V421_SMOKE) and shutil.which("nvidia-smi") is not None:
        try:
            gpu_lines = subprocess.check_output(["nvidia-smi", "-L"], text=True).strip().splitlines()
            devices = "0:1" if len(gpu_lines) >= 2 else "0"
        except Exception:
            devices = "0"
        params.update(task_type="GPU", devices=devices)
    else:
        params.update(task_type="CPU",thread_count=min(8,os.cpu_count() or 4))
    return CatBoostRanker(**params)

def fit_v421_pair(train_rows,val_rows,train_anchor,val_anchor,base_tr,meta_tr,base_va,meta_va,seed):
    eb_bundle,eb_tr=fit_generator_eb_bundle(train_rows,train_anchor)
    eb_va=apply_generator_eb_bundle(eb_bundle,val_rows,val_anchor)
    Fct,ccat,top4_tr=build_v421_top4_frame(train_rows,train_anchor,base_tr,meta_tr,eb_tr,False)
    Fgt,gcat,top4_tr2=build_v421_top4_frame(train_rows,train_anchor,base_tr,meta_tr,eb_tr,True)
    Fcv,_,top4_va=build_v421_top4_frame(val_rows,val_anchor,base_va,meta_va,eb_va,False)
    Fgv,_,top4_va2=build_v421_top4_frame(val_rows,val_anchor,base_va,meta_va,eb_va,True)
    assert np.array_equal(top4_tr,top4_tr2) and np.array_equal(top4_va,top4_va2)
    labels=_head_gather(Y_GAIN[np.asarray(train_rows,int)],top4_tr).reshape(-1).astype(np.float32)
    grp=np.repeat(np.arange(len(train_rows)),4)
    mc=make_v421_ranker(seed); mg=make_v421_ranker(seed)
    mc.fit(Pool(Fct,label=labels,group_id=grp,cat_features=ccat),verbose=False)
    mg.fit(Pool(Fgt,label=labels,group_id=grp,cat_features=gcat),verbose=False)
    gva=np.repeat(np.arange(len(val_rows)),4)
    rc=mc.predict(Pool(Fcv,group_id=gva,cat_features=ccat)).reshape(len(val_rows),4).astype(np.float32)
    rg=mg.predict(Pool(Fgv,group_id=gva,cat_features=gcat)).reshape(len(val_rows),4).astype(np.float32)
    return mc,mg,rc,rg,top4_va,eb_va,eb_bundle,Fgt.columns.tolist()

print("V42.1 Top4 CONTROL/GENCAT frames ready")

# ---- cell 34 (markdown) ----
# ## Generator-exclusive residual deployment
#
# The model is **not** allowed to use GENCAT as a replacement ranker. Each raw head is aligned to the existing anchor Top4 scale, the common CONTROL component is subtracted, and only `GENCAT − CONTROL` survives. Empirical-Bayes support weights shrink that residual continuously. The original four anchor score values are then permuted; every other score remains bit-exact.

# ---- cell 35 (code) ----
def align_head421(raw,anchor_head):
    raw=np.asarray(raw,np.float32); anchor_head=np.asarray(anchor_head,np.float32)
    return (anchor_head.mean(1,keepdims=True)+(raw-raw.mean(1,keepdims=True))*
            (np.maximum(anchor_head.std(1,keepdims=True),1e-5)/np.maximum(raw.std(1,keepdims=True),1e-5))).astype(np.float32)

def generator_exclusive_candidate(anchor,rc,rg,top4,eb_val,eligible,alpha=V421_ALPHA):
    anchor=np.asarray(anchor,np.float32); top4=np.asarray(top4,int); eligible=np.asarray(eligible,bool)
    hs=_head_gather(anchor,top4).astype(np.float32)
    ac=align_head421(rc,hs); ag=align_head421(rg,hs)
    dgen=(ag-ac).astype(np.float32)
    wt=eb_val["tpl_weight"].to_numpy(np.float32).reshape(len(anchor),4)
    wj=eb_val["journey_weight"].to_numpy(np.float32).reshape(len(anchor),4)
    wx=eb_val["transition_weight"].to_numpy(np.float32).reshape(len(anchor),4)
    reliability=(1.0-(1.0-wt)*(1.0-wj)*(1.0-wx)).astype(np.float32)
    star=(hs+float(alpha)*reliability*dgen).astype(np.float32)
    out=anchor.copy()
    for r in np.flatnonzero(eligible):
        old4=top4[r]
        new4=old4[np.argsort(-star[r],kind="mergesort")]
        if np.array_equal(old4,new4):
            continue
        vals=hs[r].copy()  # exact anchor descending score multiset
        out[r,new4]=vals
    return out,reliability,dgen,star

def geom421(base,cand):
    ob=np.argsort(-base,axis=1,kind="mergesort"); oc=np.argsort(-cand,axis=1,kind="mergesort")
    changed=np.any(ob!=oc,axis=1)
    top4mem=np.asarray([set(a[:4])!=set(b[:4]) for a,b in zip(ob,oc)])
    top5mem=np.asarray([set(a[:5])!=set(b[:5]) for a,b in zip(ob,oc)])
    rank5=np.asarray([a[4]!=b[4] for a,b in zip(ob,oc)])
    tail=np.asarray([not np.array_equal(a[4:],b[4:]) for a,b in zip(ob,oc)])
    return dict(changed_users=int(changed.sum()),top1_changes=int((ob[:,0]!=oc[:,0]).sum()),
                top4_membership_changes=int(top4mem.sum()),top5_membership_changes=int(top5mem.sum()),
                rank5_changes=int(rank5.sum()),rank5_to_17_changes=int(tail.sum()))

def top1_acc421(yt,s): return float(np.mean(np.argmax(yt,axis=1)==np.argmax(s,axis=1)))

def paired_ci421(delta,seed,blocks=None):
    d=np.asarray(delta,float); rng=np.random.default_rng(int(seed)); vals=np.empty(BOOTSTRAPS_V421,float)
    if blocks is None:
        for b in range(BOOTSTRAPS_V421): vals[b]=d[rng.integers(0,len(d),len(d))].mean()
    else:
        blocks=[np.asarray(x,int) for x in blocks if len(x)]
        for b in range(BOOTSTRAPS_V421):
            pick=rng.integers(0,len(blocks),len(blocks)); idx=np.concatenate([blocks[j] for j in pick]); vals[b]=d[idx].mean()
    return float(np.quantile(vals,.025)),float(np.quantile(vals,.975))

print("V42.1 fixed generator-exclusive support-shrunk deployment ready")

# ---- cell 36 (markdown) ----
# ## DESIGN — honest outer OOF experiment
#
# Same 3400/600 split as V42. Every outer model receives policy-matched OOF training geometry. The candidate is evaluated only on S0 analogue rows. Cold-template behavior is measured directly from each fold's training support, so no second template-stress model is needed.

# ---- cell 37 (code) ----
all_train_idx=np.arange(N_TRAIN)
if V421_SMOKE:
    rng=np.random.default_rng(V421_SPLIT_SEED)
    sub=np.sort(rng.choice(all_train_idx,size=DESIGN_SIZE+LOCK_SIZE,replace=False))
    strata_sub=np.argmax(y[sub],axis=1)
    sss=StratifiedShuffleSplit(n_splits=1,test_size=LOCK_SIZE,random_state=V421_SPLIT_SEED)
    dloc,lloc=next(sss.split(np.zeros(len(sub)),strata_sub)); DESIGN_IDX=sub[dloc]; LOCK_IDX=sub[lloc]
else:
    strata=np.argmax(y,axis=1)
    sss=StratifiedShuffleSplit(n_splits=1,test_size=LOCK_SIZE,random_state=V421_SPLIT_SEED)
    DESIGN_IDX,LOCK_IDX=next(sss.split(np.zeros(N_TRAIN),strata))
DESIGN_IDX=np.sort(np.asarray(DESIGN_IDX,int)); LOCK_IDX=np.sort(np.asarray(LOCK_IDX,int))
assert len(DESIGN_IDX)==DESIGN_SIZE and len(LOCK_IDX)==LOCK_SIZE

skf=StratifiedKFold(OUTER_FOLDS,shuffle=True,random_state=V421_FOLD_SEED)
design_top=np.argmax(y[DESIGN_IDX],axis=1)
outer=list(skf.split(np.zeros(len(DESIGN_IDX)),design_top))

ANCHOR_OOF=np.zeros((len(DESIGN_IDX),N_MODULES),np.float32)
CAND_OOF=np.zeros_like(ANCHOR_OOF)
FOLD_ID=np.full(len(DESIGN_IDX),-1,int)
RELIABILITY_OOF=np.zeros((len(DESIGN_IDX),4),np.float32)
TPL_SUPPORT_OOF=np.zeros(len(DESIGN_IDX),np.float32)
JOURNEY_SUPPORT_OOF=np.zeros(len(DESIGN_IDX),np.float32)
TRANSITION_SUPPORT_OOF=np.zeros(len(DESIGN_IDX),np.float32)
fold_rows=[]; kappa_rows=[]

for f,(trloc,valoc) in enumerate(outer):
    tr=DESIGN_IDX[trloc]; va=DESIGN_IDX[valoc]
    print(f"V42.1 outer {f+1}/{OUTER_FOLDS}: train={len(tr)} val={len(va)}")
    pol=build_policy_matched_split_v421(tr,va,V421_FOLD_SEED+1000*f)
    _,_,rc,rg,top4_va,eb_va,eb_bundle,_=fit_v421_pair(
        tr,va,pol["train_anchor"],pol["val_anchor"],pol["base_train"],pol["meta_train"],pol["base_val"],pol["meta_val"],V421_FOLD_SEED+5000+f
    )
    eligible=(pol["val_state"]==0)
    cand,rel,dgen,_=generator_exclusive_candidate(pol["val_anchor"],rc,rg,top4_va,eb_va,eligible)
    ANCHOR_OOF[valoc]=pol["val_anchor"]; CAND_OOF[valoc]=cand; FOLD_ID[valoc]=f; RELIABILITY_OOF[valoc]=rel
    # category-level support is identical across the four candidate rows; take first.
    TPL_SUPPORT_OOF[valoc]=eb_va["tpl_group_support"].to_numpy(np.float32).reshape(len(va),4)[:,0]
    JOURNEY_SUPPORT_OOF[valoc]=eb_va["journey_group_support"].to_numpy(np.float32).reshape(len(va),4)[:,0]
    TRANSITION_SUPPORT_OOF[valoc]=eb_va["transition_group_support"].to_numpy(np.float32).reshape(len(va),4)[:,0]
    delta=ndcg5(y[va],cand)-ndcg5(y[va],pol["val_anchor"])
    fm={
        "fold":f,"n":len(va),"eligible_s0":int(eligible.sum()),"anchor_ndcg":ndcg5(y[va],pol["val_anchor"]),
        "candidate_ndcg":ndcg5(y[va],cand),"delta":delta,
        "anchor_top1":top1_acc421(y[va],pol["val_anchor"]),"candidate_top1":top1_acc421(y[va],cand),
        "mean_reliability":float(rel.mean()),"mean_abs_dgen":float(np.mean(np.abs(dgen))),
        **{f"geom_{k}":v for k,v in geom421(pol["val_anchor"],cand).items()},
    }
    fold_rows.append(fm)
    for fam in GEN_FAMILIES:
        kappa_rows.append({"fold":f,"family":fam,"kappa":eb_bundle["kappas"][fam],**eb_bundle["moments"][fam]})
    del pol,rc,rg,cand,rel,dgen,eb_va,eb_bundle
    gc.collect()

assert (FOLD_ID>=0).all()
V421_OUTER=pd.DataFrame(fold_rows); V421_KAPPA=pd.DataFrame(kappa_rows)
DESIGN_ANCHOR_NDCG=ndcg5(y[DESIGN_IDX],ANCHOR_OOF)
DESIGN_CAND_NDCG=ndcg5(y[DESIGN_IDX],CAND_OOF)
DESIGN_DELTA=DESIGN_CAND_NDCG-DESIGN_ANCHOR_NDCG
DESIGN_USER_DELTA=per_user_ndcg(y[DESIGN_IDX],CAND_OOF)-per_user_ndcg(y[DESIGN_IDX],ANCHOR_OOF)
DESIGN_CI=paired_ci421(DESIGN_USER_DELTA,V421_BOOT_SEED)
DESIGN_TOP1_DELTA=top1_acc421(y[DESIGN_IDX],CAND_OOF)-top1_acc421(y[DESIGN_IDX],ANCHOR_OOF)
DESIGN_GEOM=geom421(ANCHOR_OOF,CAND_OOF)
FOLD_WINS=int((V421_OUTER.delta>0).sum())

ordloc=np.argsort(TRAIN_ID_NUM[DESIGN_IDX]); contiguous_blocks=np.array_split(ordloc,BLOCKS); block_rows=[]
for b,rr in enumerate(contiguous_blocks):
    block_rows.append({"block":b,"n":len(rr),"id_min":int(TRAIN_ID_NUM[DESIGN_IDX[rr]].min()),"id_max":int(TRAIN_ID_NUM[DESIGN_IDX[rr]].max()),
                       "anchor_ndcg":ndcg5(y[DESIGN_IDX[rr]],ANCHOR_OOF[rr]),"candidate_ndcg":ndcg5(y[DESIGN_IDX[rr]],CAND_OOF[rr]),
                       "delta":ndcg5(y[DESIGN_IDX[rr]],CAND_OOF[rr])-ndcg5(y[DESIGN_IDX[rr]],ANCHOR_OOF[rr]),
                       "tpl_seen_rate":float(np.mean(TPL_SUPPORT_OOF[rr]>0)),"tpl_support_ge5":float(np.mean(TPL_SUPPORT_OOF[rr]>=5)),
                       "journey_seen_rate":float(np.mean(JOURNEY_SUPPORT_OOF[rr]>0)),"transition_seen_rate":float(np.mean(TRANSITION_SUPPORT_OOF[rr]>0))})
V421_BLOCKS=pd.DataFrame(block_rows); BLOCK_WINS=int((V421_BLOCKS.delta>0).sum()); DESIGN_BLOCK_CI=paired_ci421(DESIGN_USER_DELTA,V421_BOOT_SEED+1,blocks=contiguous_blocks)

cold=TPL_SUPPORT_OOF<=0
COLD_TEMPLATE_DELTA=float(DESIGN_USER_DELTA[cold].mean()) if cold.any() else 0.0
COLD_TEMPLATE_N=int(cold.sum())
low=(TPL_SUPPORT_OOF<2)&(JOURNEY_SUPPORT_OOF<2)&(TRANSITION_SUPPORT_OOF<2)
LOW_SUPPORT_DELTA=float(DESIGN_USER_DELTA[low].mean()) if low.any() else 0.0
LOW_SUPPORT_N=int(low.sum())

V421_DESIGN_METRICS=pd.DataFrame([{
    "anchor_ndcg":DESIGN_ANCHOR_NDCG,"candidate_ndcg":DESIGN_CAND_NDCG,"delta":DESIGN_DELTA,"top1_delta":DESIGN_TOP1_DELTA,
    "ci_low":DESIGN_CI[0],"ci_high":DESIGN_CI[1],"block_ci_low":DESIGN_BLOCK_CI[0],"block_ci_high":DESIGN_BLOCK_CI[1],
    "fold_wins":FOLD_WINS,"block_wins":BLOCK_WINS,"worst_block_delta":float(V421_BLOCKS.delta.min()),
    "cold_template_n":COLD_TEMPLATE_N,"cold_template_delta":COLD_TEMPLATE_DELTA,"low_support_n":LOW_SUPPORT_N,"low_support_delta":LOW_SUPPORT_DELTA,
    **{f"geom_{k}":v for k,v in DESIGN_GEOM.items()},
}])

print(V421_OUTER.to_string(index=False))
print(V421_BLOCKS.to_string(index=False))
print("DESIGN anchor/candidate:",DESIGN_ANCHOR_NDCG,DESIGN_CAND_NDCG,"delta",DESIGN_DELTA,"CI",DESIGN_CI)
print("cold template",COLD_TEMPLATE_N,COLD_TEMPLATE_DELTA,"low support",LOW_SUPPORT_N,LOW_SUPPORT_DELTA)
print(V421_KAPPA.groupby("family")["kappa"].agg(["mean","min","max"]).to_string())

# ---- cell 38 (markdown) ----
# ## DESIGN gate → untouched LOCK
#
# LOCK opens only if the support-shrunk generator-exclusive policy reaches leaderboard-relevant scale on DESIGN with consistent folds/blocks and no Top1 or geometry regression. The full model is trained on DESIGN only; LOCK never changes hyperparameters or shrinkage rules.

# ---- cell 39 (code) ----
DESIGN_GATE=bool(
    DESIGN_DELTA>=DESIGN_MIN_DELTA and FOLD_WINS>=DESIGN_MIN_FOLD_WINS and BLOCK_WINS>=DESIGN_MIN_BLOCK_WINS and
    DESIGN_CI[0]>=DESIGN_BOOT_CI_FLOOR and float(V421_BLOCKS.delta.min())>=DESIGN_WORST_BLOCK_FLOOR and
    COLD_TEMPLATE_DELTA>=COLD_TEMPLATE_FLOOR and DESIGN_TOP1_DELTA>=0 and
    DESIGN_GEOM["top5_membership_changes"]==0 and DESIGN_GEOM["rank5_changes"]==0 and DESIGN_GEOM["rank5_to_17_changes"]==0
)
print("DESIGN_GATE",DESIGN_GATE)

LOCK_EVALUATED=False; LOCK_GATE=False
V421_LOCK_METRICS=pd.DataFrame([{"evaluated":False,"lock_gate":False}]); V421_LOCK_BLOCKS=pd.DataFrame()
FULL_CONTROL_MODEL=None; FULL_GENCAT_MODEL=None; FULL_FEATURE_NAMES=[]
TEST_CANDIDATE=V421_ANCHOR_TEST.copy(); TEST_RELIABILITY=np.zeros((N_TEST,4),np.float32)

if DESIGN_GATE:
    combined=np.concatenate([LOCK_IDX,np.arange(N_TRAIN,N_ALL)])
    pol=build_policy_matched_split_v421(DESIGN_IDX,combined,V421_FOLD_SEED+90000)
    nlock=len(LOCK_IDX)
    eb_bundle,eb_tr=fit_generator_eb_bundle(DESIGN_IDX,pol["train_anchor"])

    Fct,ccat,top4_tr=build_v421_top4_frame(DESIGN_IDX,pol["train_anchor"],pol["base_train"],pol["meta_train"],eb_tr,False)
    Fgt,gcat,top4_tr2=build_v421_top4_frame(DESIGN_IDX,pol["train_anchor"],pol["base_train"],pol["meta_train"],eb_tr,True)
    assert np.array_equal(top4_tr,top4_tr2)
    labels=_head_gather(Y_GAIN[DESIGN_IDX],top4_tr).reshape(-1).astype(np.float32); grp=np.repeat(np.arange(len(DESIGN_IDX)),4)
    FULL_CONTROL_MODEL=make_v421_ranker(V421_FOLD_SEED+99001); FULL_GENCAT_MODEL=make_v421_ranker(V421_FOLD_SEED+99001)
    FULL_CONTROL_MODEL.fit(Pool(Fct,label=labels,group_id=grp,cat_features=ccat),verbose=False)
    FULL_GENCAT_MODEL.fit(Pool(Fgt,label=labels,group_id=grp,cat_features=gcat),verbose=False)
    FULL_FEATURE_NAMES=Fgt.columns.tolist()

    # LOCK: exact DESIGN-trained policy-matched structural surface.
    lock_anchor=pol["val_anchor"][:nlock]; eb_lock=apply_generator_eb_bundle(eb_bundle,LOCK_IDX,lock_anchor)
    Flc,_,top4_lock=build_v421_top4_frame(LOCK_IDX,lock_anchor,pol["base_val"][:nlock],pol["meta_val"][:nlock],eb_lock,False)
    Flg,_,top4_lock2=build_v421_top4_frame(LOCK_IDX,lock_anchor,pol["base_val"][:nlock],pol["meta_val"][:nlock],eb_lock,True)
    assert np.array_equal(top4_lock,top4_lock2)
    gl=np.repeat(np.arange(nlock),4)
    rc=FULL_CONTROL_MODEL.predict(Pool(Flc,group_id=gl,cat_features=ccat)).reshape(nlock,4).astype(np.float32)
    rg=FULL_GENCAT_MODEL.predict(Pool(Flg,group_id=gl,cat_features=gcat)).reshape(nlock,4).astype(np.float32)
    lock_eligible=(pol["val_state"][:nlock]==0)
    LOCK_CAND,lock_rel,_,_=generator_exclusive_candidate(lock_anchor,rc,rg,top4_lock,eb_lock,lock_eligible)
    ldelta=ndcg5(y[LOCK_IDX],LOCK_CAND)-ndcg5(y[LOCK_IDX],lock_anchor)
    ltop=top1_acc421(y[LOCK_IDX],LOCK_CAND)-top1_acc421(y[LOCK_IDX],lock_anchor)
    ld=per_user_ndcg(y[LOCK_IDX],LOCK_CAND)-per_user_ndcg(y[LOCK_IDX],lock_anchor); lci=paired_ci421(ld,V421_BOOT_SEED+10)
    lgeom=geom421(lock_anchor,LOCK_CAND)
    ol=np.argsort(TRAIN_ID_NUM[LOCK_IDX]); lb=[]
    for b,rr in enumerate(np.array_split(ol,BLOCKS)):
        lb.append({"block":b,"n":len(rr),"delta":ndcg5(y[LOCK_IDX[rr]],LOCK_CAND[rr])-ndcg5(y[LOCK_IDX[rr]],lock_anchor[rr])})
    V421_LOCK_BLOCKS=pd.DataFrame(lb); lbwins=int((V421_LOCK_BLOCKS.delta>0).sum())
    LOCK_GATE=bool(ldelta>=LOCK_MIN_DELTA and ltop>=0 and lci[0]>=LOCK_BOOT_CI_FLOOR and lbwins>=LOCK_MIN_BLOCK_WINS and
                   lgeom["top5_membership_changes"]==0 and lgeom["rank5_changes"]==0 and lgeom["rank5_to_17_changes"]==0)
    LOCK_EVALUATED=True
    V421_LOCK_METRICS=pd.DataFrame([{"evaluated":True,"anchor_ndcg":ndcg5(y[LOCK_IDX],lock_anchor),"candidate_ndcg":ndcg5(y[LOCK_IDX],LOCK_CAND),
                                    "delta":ldelta,"top1_delta":ltop,"ci_low":lci[0],"ci_high":lci[1],"block_wins":lbwins,
                                    "mean_reliability":float(lock_rel.mean()),**{f"geom_{k}":v for k,v in lgeom.items()},"lock_gate":LOCK_GATE}])

    # TEST uses the exact public 0.66028 anchor; side-features are DESIGN-only.
    test_base=pol["base_val"][nlock:]; test_meta=pol["meta_val"][nlock:]
    eb_test=apply_generator_eb_bundle(eb_bundle,np.arange(N_TRAIN,N_ALL),V421_ANCHOR_TEST)
    Ftc,_,top4_test=build_v421_top4_frame(np.arange(N_TRAIN,N_ALL),V421_ANCHOR_TEST,test_base,test_meta,eb_test,False)
    Ftg,_,top4_test2=build_v421_top4_frame(np.arange(N_TRAIN,N_ALL),V421_ANCHOR_TEST,test_base,test_meta,eb_test,True)
    assert np.array_equal(top4_test,top4_test2)
    gt=np.repeat(np.arange(N_TEST),4)
    rtc=FULL_CONTROL_MODEL.predict(Pool(Ftc,group_id=gt,cat_features=ccat)).reshape(N_TEST,4).astype(np.float32)
    rtg=FULL_GENCAT_MODEL.predict(Pool(Ftg,group_id=gt,cat_features=gcat)).reshape(N_TEST,4).astype(np.float32)
    TEST_CANDIDATE,TEST_RELIABILITY,_,_=generator_exclusive_candidate(V421_ANCHOR_TEST,rtc,rtg,top4_test,eb_test,TEST_ELIGIBLE)
    del pol,Fct,Fgt,Flc,Flg,Ftc,Ftg,rc,rg,rtc,rtg,eb_tr,eb_lock,eb_test,eb_bundle
    gc.collect()

print(V421_LOCK_METRICS.to_string(index=False))
print("LOCK_GATE",LOCK_GATE)

# ---- cell 40 (markdown) ----
# ## Test geometry, fail-close primary, and reproducibility artifacts
#
# A passing model still cannot submit unless test geometry respects the frozen policy. The primary CSV is the candidate only when DESIGN + LOCK + geometry all pass; otherwise it is the exact 0.66028 anchor.

# ---- cell 41 (code) ----
TEST_GEOM=geom421(V421_ANCHOR_TEST,TEST_CANDIDATE)
ob=np.argsort(-V421_ANCHOR_TEST,axis=1,kind="mergesort"); oc=np.argsort(-TEST_CANDIDATE,axis=1,kind="mergesort")
changed=np.any(ob!=oc,axis=1)
locked_exact=bool(np.array_equal(TEST_CANDIDATE[~TEST_ELIGIBLE],V421_ANCHOR_TEST[~TEST_ELIGIBLE]))
score_multiset_ok=True
for r in np.flatnonzero(changed):
    if not np.array_equal(np.sort(V421_ANCHOR_TEST[r,ob[r,:4]]),np.sort(TEST_CANDIDATE[r,oc[r,:4]])):
        score_multiset_ok=False; break
TEST_GEOMETRY_GATE=bool(
    locked_exact and score_multiset_ok and TEST_GEOM["top5_membership_changes"]==0 and TEST_GEOM["rank5_changes"]==0 and
    TEST_GEOM["rank5_to_17_changes"]==0 and TEST_GEOM["changed_users"]<=TEST_MAX_CHANGED and TEST_GEOM["top1_changes"]<=TEST_MAX_TOP1_CHANGED
)
FINAL_GATE=bool(DESIGN_GATE and LOCK_GATE and TEST_GEOMETRY_GATE and not V421_SMOKE)
PRIMARY=TEST_CANDIDATE if FINAL_GATE else V421_ANCHOR_TEST

V421_TEST_GEOMETRY=pd.DataFrame([{**TEST_GEOM,"eligible_s0":int(TEST_ELIGIBLE.sum()),"locked_users":int((~TEST_ELIGIBLE).sum()),
    "locked_rows_bit_exact":locked_exact,"top4_score_multiset_exact":score_multiset_ok,"geometry_gate":TEST_GEOMETRY_GATE,
    "mean_reliability":float(TEST_RELIABILITY.mean()),"candidate_top5_sha256":top5_sha256(TEST_CANDIDATE),"anchor_top5_sha256":top5_sha256(V421_ANCHOR_TEST)}])

# DESIGN module audit remains diagnostic only; no module whitelist is permitted.
module_a=[]; oa=np.argsort(-ANCHOR_OOF,axis=1,kind="mergesort"); og=np.argsort(-CAND_OOF,axis=1,kind="mergesort")
for m,mid in enumerate(MODULES):
    involved=np.asarray([(m in a[:4]) or (m in b[:4]) for a,b in zip(oa,og)])
    ch=np.any(oa!=og,axis=1)&involved; d=DESIGN_USER_DELTA[ch]
    module_a.append({"module":mid,"changed_users_involving_module":int(ch.sum()),"mean_user_delta":float(d.mean()) if len(d) else np.nan,
                     "positive_rate":float(np.mean(d>0)) if len(d) else np.nan})
V421_MODULE_AUDIT=pd.DataFrame(module_a)

if FULL_GENCAT_MODEL is not None:
    imp=FULL_GENCAT_MODEL.get_feature_importance()
    V421_FEATURE_IMPORTANCE=pd.DataFrame({"feature":FULL_FEATURE_NAMES,"importance":imp}).sort_values("importance",ascending=False)
else:
    V421_FEATURE_IMPORTANCE=pd.DataFrame(columns=["feature","importance"])

def submission421(scores):
    out=sample.copy(); out.loc[:,MODULES]=np.asarray(scores,np.float32)
    assert out.user_id.astype(str).tolist()==test.user_id.astype(str).tolist(); assert np.isfinite(out[MODULES].to_numpy()).all()
    return out

submission421(V421_ANCHOR_TEST).to_csv(V421_OUT/"submission_v42_1_anchor_066028_reference.csv",index=False)
submission421(TEST_CANDIDATE).to_csv(V421_OUT/"submission_v42_1_candidate_DIAGNOSTIC.csv",index=False)
submission421(PRIMARY).to_csv(V421_OUT/"submission_v42_1_primary.csv",index=False)

V421_DESIGN_METRICS.to_csv(V421_OUT/"v42_1_design_metrics.csv",index=False)
V421_OUTER.to_csv(V421_OUT/"v42_1_outer_fold_metrics.csv",index=False)
V421_BLOCKS.to_csv(V421_OUT/"v42_1_block_metrics.csv",index=False)
V421_KAPPA.to_csv(V421_OUT/"v42_1_eb_kappa.csv",index=False)
V421_LOCK_METRICS.to_csv(V421_OUT/"v42_1_lock_metrics.csv",index=False)
V421_LOCK_BLOCKS.to_csv(V421_OUT/"v42_1_lock_block_metrics.csv",index=False)
V421_TEST_GEOMETRY.to_csv(V421_OUT/"v42_1_test_geometry.csv",index=False)
V421_TEST_SUPPORT_AUDIT.to_csv(V421_OUT/"v42_1_test_support_audit.csv",index=False)
V421_MODULE_AUDIT.to_csv(V421_OUT/"v42_1_module_audit.csv",index=False)
V421_FEATURE_IMPORTANCE.to_csv(V421_OUT/"v42_1_feature_importance.csv",index=False)
pd.DataFrame({"user_id":train.user_id,"split":np.where(np.isin(np.arange(N_TRAIN),DESIGN_IDX),"DESIGN","LOCK")}).to_csv(V421_OUT/"v42_1_split_manifest.csv",index=False)

import platform
pkg={"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__,
     "lightgbm":lgb.__version__,"catboost":catboost.__version__}
json.dump(pkg,open(V421_OUT/"v42_1_package_versions.json","w"),indent=2)

REPRO={
    "exact_v30_v31_current_reproduction":bool(REPRO_OK),"anchor_public_score":PUBLIC_V421_ANCHOR_LB,
    "anchor_top5_sha256":top5_sha256(V421_ANCHOR_TEST),"expected_anchor_top5_sha256":EXPECTED_V421_ANCHOR_TOP5_SHA256,
    "state_counts":{STATE_NAMES[k]:int((full_state==k).sum()) for k in range(4)},"expected_state_counts":EXPECTED_STATE_COUNTS,
    "boundary_counts":boundary_counts,"expected_boundary_counts":EXPECTED_BOUNDARY_COUNTS,
    "anchor_family":"current066020 + B34 + B45, algorithmic ranking-geometry rule",
    "organizer_data_only":True,"external_labeled_data":False,"no_historical_submission_csv_predictive_input":True,"no_user_id_predictive_feature":True,
    "no_post_lock_retraining":True,"s1_s2_s3_test_locked":True,"s0_only_deployment":True,"top5_membership_frozen":True,"rank5_frozen":True,
    "top4_score_values_permuted_only":True,"fixed_alpha":V421_ALPHA,"control_vs_gencat_same_hyperparameters":True,
    "generator_authority":"aligned GENCAT minus aligned CONTROL only","policy_matched_oof_training_anchor":True,"head_training_group_size":4,
    "support_shrinkage":"training-only method-of-moments empirical Bayes; LOO train features; frozen validation/test mappings",
}
json.dump(REPRO,open(V421_OUT/"v42_1_reproduction_audit.json","w"),indent=2)

summary={
    "version":V421_VERSION,"public_anchor":PUBLIC_V421_ANCHOR_LB,"leader_reference":PUBLIC_LEADER_LB,"leader_gap":PUBLIC_LEADER_LB-PUBLIC_V421_ANCHOR_LB,
    "design_gate":bool(DESIGN_GATE),"lock_evaluated":bool(LOCK_EVALUATED),"lock_gate":bool(LOCK_GATE),"test_geometry_gate":bool(TEST_GEOMETRY_GATE),
    "final_gate":bool(FINAL_GATE),"design_delta":float(DESIGN_DELTA),"design_fold_wins":int(FOLD_WINS),"design_block_wins":int(BLOCK_WINS),
    "cold_template_delta":float(COLD_TEMPLATE_DELTA),"test_changed_users":int(TEST_GEOM["changed_users"]),"test_top1_changes":int(TEST_GEOM["top1_changes"]),
    "anchor_top5_sha256":top5_sha256(V421_ANCHOR_TEST),"candidate_top5_sha256":top5_sha256(TEST_CANDIDATE),"primary_top5_sha256":top5_sha256(PRIMARY),
    "submission_instruction":"SUBMIT submission_v42_1_primary.csv" if FINAL_GATE else "NO NEW V42.1 SUBMISSION: primary is exact 0.66028 fail-close anchor.",
}
json.dump(summary,open(V421_OUT/"v42_1_summary.json","w"),indent=2)
print(json.dumps(summary,indent=2))
print("V42.1 outputs written to",V421_OUT)

# ---- cell 42 (markdown) ----
# ### Terminal interpretation contract
#
# - **FINAL_GATE = True:** V42.1 has passed DESIGN, untouched LOCK, and exact test geometry. `submission_v42_1_primary.csv` is the only intended submission.
# - **FINAL_GATE = False:** do **not** submit the diagnostic candidate. `submission_v42_1_primary.csv` is the exact 0.66028 champion fail-close.
# - No alpha/support/module/threshold tuning is authorized after seeing DESIGN or LOCK. If this architecture fails, the generator branch is closed and the next family is joint user-level modeling (e.g. TabM), not V42.1.x mining.
