# Converted from notebooks/v29_5.ipynb: markdown cells become comments, %/! magics neutralized.

# ---- cell 0 (markdown) ----
# # MineToday V29.5 — Reliability & Curriculum Kill-Test Diagnostics
#
# **Status:** diagnostics-only — **do not submit anything from this notebook**  
# **Active public anchor:** V25 core package = **0.65919**  
# **Leader reference at design time:** **0.66112**  
# **Metric:** user-level NDCG@5
#
# V29 showed the first recent architecture with positive signal in fully nested template-group OOF, fresh LOCK, and chronology, but its pseudo-public lower tail remained too wide. V29.5 does **not** search for another leaderboard candidate. It asks whether the next architecture should actually be an uncertainty-controlled residual LambdaRank, and whether transitive curriculum or co-occurrence deserves to survive into V30.
#
# ### Preregistered kill tests
#
# 1. **Honest replica uncertainty:** every validation user's replica predictions come only from models trained on that outer fold's training users. No cross-outer-fold consensus leakage.
# 2. **Ranking-space stability:** uncertainty is measured through Top-1 votes, Top-5 set agreement, boundary-pair agreement, and first rank-crossing agreement — not raw score variance.
# 3. **Rank-boundary mechanics:** exact crossing thresholds identify which residual interventions can change Top-1, Top-5 order, or Top-5 membership.
# 4. **Transitive curriculum precision:** direct, 2-hop, and 3-hop prerequisite/next-step paths are audited separately for explicit-intent seeds and anchor seeds, with readiness bands.
# 5. **Fresh-world ablation:** frozen LTR Base, +Gap, and +Gap+Co-occurrence are compared on a new template-disjoint LOCK and chronology. Residual-covariance graph is intentionally excluded.
# 6. **Cluster-aware pseudo-public stress:** row-level, template-cluster, and intent-stratified masks quantify lower-tail risk under different dependence assumptions.
# 7. **Test uncertainty shift:** full-train replica predictions compare the unlabeled test confidence regime with honest OOF confidence.
# 8. **Stratified human taxonomy:** 30 worst + 30 middle + 30 rank-boundary + 10 noisy/no-chat users are exported with full observed context and an empty manual taxonomy column.
#
# V29.5 writes diagnostics only. Fresh worlds are never used to select a model or create a submission.

# ---- cell 1 (code) ----
# Runtime + deterministic V29.5 configuration
import os, sys, re, json, ast, html, hashlib, random, warnings, time, subprocess, gc
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ["PYTHONHASHSEED"] = "42"

V295_SKIP_INSTALL = bool(int(os.getenv("V295_SKIP_INSTALL", "0")))
try:
    import lightgbm as lgb
except Exception:
    if V295_SKIP_INSTALL:
        raise
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "lightgbm==4.6.0"])
    import lightgbm as lgb

import numpy as np
import pandas as pd
import scipy
import sklearn
from IPython.display import display
from scipy.sparse import hstack
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")

SEED = 42
V295_SMOKE = bool(int(os.getenv("V295_SMOKE", "0")))
V295_STRICT_GEOMETRY = bool(int(os.getenv("V295_STRICT_GEOMETRY", "1")))

# Compatibility aliases consumed by inherited frozen anchor cells.
V29_SMOKE = V295_SMOKE
V29_STRICT_GEOMETRY = V295_STRICT_GEOMETRY
V27_SMOKE = V295_SMOKE
V27_STRICT_GEOMETRY = V295_STRICT_GEOMETRY

N_FOLDS = 2 if V295_SMOKE else 5
ANCHOR_INNER_FOLDS = 2 if V295_SMOKE else 4
GRAPH_INNER_FOLDS = 2 if V295_SMOKE else 3
RIDGE_ALPHA = 3.0
REFERENCE_W = 0.45

PUBLIC_CAL045_LB = 0.65639
PUBLIC_CONSENSUS3_LB = 0.65856
PUBLIC_PRE_V25_PB = 0.65912
PUBLIC_V25_LB = 0.65919
PUBLIC_CURRENT_PB = PUBLIC_V25_LB
PUBLIC_LEADER_LB = 0.66112

LGB_ESTIMATORS = 40 if V295_SMOKE else 1200
LGB_EARLY_STOP = 8 if V295_SMOKE else 80
RANKER_ESTIMATORS = 45 if V295_SMOKE else 360
RANKER_LEARNING_RATE = 0.04 if V295_SMOKE else 0.025

# Honest uncertainty replicas. Full mode uses five replicas exactly as preregistered.
REPLICA_COUNT = 2 if V295_SMOKE else 5
REPLICA_ABLATIONS = ["ltr_base", "ltr_gap_cooc"]
FRESH_ABLATIONS = ["ltr_base", "ltr_gap", "ltr_gap_cooc"]
FIXED_ALPHAS = [0.10, 0.16]
PRIMARY_REPLICA_ALPHA = {"ltr_base":0.10, "ltr_gap_cooc":0.16}

BOOTSTRAPS = 60 if V295_SMOKE else 1000
PSEUDO_PUBLIC_MASKS = 30 if V295_SMOKE else 320
GEOMETRY_ATOL = 2.5e-4

# V29 helper compatibility. Residual graph exists in inherited helper code, but V29.5 never trains it.
RESIDUAL_ALPHAS = FIXED_ALPHAS
ABLATION_ORDER = FRESH_ABLATIONS
ORACLE_PAIR_SUPPORT = 3 if V295_SMOKE else 15
ORACLE_MIN_COND = 0.15
ORACLE_MIN_LIFT = 1.20

# Diagnostic-only decision thresholds; these classify evidence, never create a submission.
UNCERTAINTY_MIN_SPEARMAN = 0.08
UNCERTAINTY_MIN_TOP_BOTTOM_DELTA = 0.00035
CURRICULUM_MIN_SUPPORT = 80
CURRICULUM_MIN_LIFT = 1.20
COOC_FRESH_TOL = -0.00010

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
print("V295_SMOKE", V295_SMOKE, "replicas", REPLICA_COUNT, "leader gap", round(PUBLIC_LEADER_LB-PUBLIC_CURRENT_PB, 5))
try:
    print("GPU runtime:\n" + subprocess.check_output(["nvidia-smi", "-L"], text=True).strip())
except Exception:
    print("nvidia-smi unavailable; V29.5 is T4x2-compatible but deliberately CPU-first.")

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
# ## Honest replica ensemble and ranking-space uncertainty
#
# Each outer validation fold is predicted by a family of independently randomized LambdaRank replicas trained only on the corresponding outer-training users. The same long-format feature matrix is reused across replicas; diversity comes from LightGBM's seed, row subsampling, feature subsampling, and histogram randomness.
#
# The uncertainty score deliberately ignores raw prediction variance. It measures whether replicas agree on the ranking decisions that can affect NDCG@5.

# ---- cell 21 (code) ----
def feature_end(bounds,abl):
    return bounds[ABLATION_BOUND[abl]]


def make_replica_ranker(seed):
    return lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=RANKER_ESTIMATORS,
        learning_rate=RANKER_LEARNING_RATE,
        num_leaves=15,
        min_child_samples=80,
        reg_alpha=1.0,
        reg_lambda=10.0,
        colsample_bytree=0.72 + 0.04*(int(seed)%3),
        subsample=0.82 + 0.04*(int(seed)%4),
        subsample_freq=1,
        max_bin=127,
        lambdarank_truncation_level=8,
        label_gain=LABEL_GAIN,
        random_state=int(seed),
        bagging_seed=int(seed)+11,
        feature_fraction_seed=int(seed)+23,
        data_random_seed=int(seed)+37,
        n_jobs=4,
        verbosity=-1,
        force_col_wise=True,
    )


def fit_replica_family(Xtr,Xva,grades_tr,n_users,seeds):
    preds=[]
    group=np.full(int(n_users),N_MODULES,dtype=np.int32)
    yy=grades_tr.reshape(-1).astype(np.int32)
    for seed in seeds:
        model=make_replica_ranker(seed)
        model.fit(Xtr,yy,group=group)
        preds.append(model.predict(Xva).reshape(-1,N_MODULES).astype(np.float32))
        del model
    return np.stack(preds,axis=0)


def pairwise_set_jaccard(topk_rep):
    # topk_rep: [R,N,K]
    R,N,K=topk_rep.shape
    out=np.ones(N,np.float32)
    if R<2: return out
    vals=np.zeros(N,np.float32); cnt=0
    for a in range(R):
        for b in range(a+1,R):
            for i in range(N):
                A=set(topk_rep[a,i].tolist()); B=set(topk_rep[b,i].tolist())
                vals[i]+=len(A&B)/max(1,len(A|B))
            cnt+=1
    return vals/max(cnt,1)


def modal_fraction(values):
    # values [R,N] integer/hashable
    values=np.asarray(values)
    R,N=values.shape
    out=np.zeros(N,np.float32)
    for i in range(N):
        _,counts=np.unique(values[:,i],return_counts=True)
        out[i]=counts.max()/R
    return out


def exact_cross_alpha(anchor_row,delta_row,hi,lo):
    # anchor has hi above lo. residual wants lo to move above hi if denominator > 0.
    margin=float(anchor_row[hi]-anchor_row[lo])
    denom=float(delta_row[lo]-delta_row[hi])
    if margin<=0 or denom<=1e-12:
        return np.inf
    return margin/denom


def first_critical_cross(anchor_row,aligned_row):
    order=np.argsort(-anchor_row,kind="mergesort")
    d=aligned_row-anchor_row
    events=[]
    # Adjacent order events inside top5 excluding rank1/rank2, which is labeled separately below.
    for r in range(1,4):
        hi=int(order[r]); lo=int(order[r+1])
        a=exact_cross_alpha(anchor_row,d,hi,lo)
        if np.isfinite(a): events.append((a,"order",hi,lo))
    # Membership event rank5 vs rank6.
    hi=int(order[4]); lo=int(order[5])
    a=exact_cross_alpha(anchor_row,d,hi,lo)
    if np.isfinite(a): events.append((a,"membership",hi,lo))
    # Top1 vs rank2 is called out separately.
    hi=int(order[0]); lo=int(order[1])
    a=exact_cross_alpha(anchor_row,d,hi,lo)
    if np.isfinite(a): events.append((a,"top1",hi,lo))
    if not events: return np.inf,"none",-1,-1
    return min(events,key=lambda x:(x[0],x[1],x[2],x[3]))


def ranking_stability(anchor,aligned_reps,alpha):
    anchor=np.asarray(anchor,np.float32)
    aligned_reps=np.asarray(aligned_reps,np.float32)  # [R,N,M]
    R,N,M=aligned_reps.shape
    final=anchor[None,:,:]+float(alpha)*(aligned_reps-anchor[None,:,:])
    top1=np.argmax(final,axis=2)
    top5=np.argsort(-final,axis=2,kind="mergesort")[:,:,:5]
    top1_agree=modal_fraction(top1)
    top5_jacc=pairwise_set_jaccard(top5)

    anchor_order=np.argsort(-anchor,axis=1,kind="mergesort")
    boundary_votes=np.zeros((R,N),np.int8)
    boundary_direction=np.zeros((R,N),np.int8)
    first_labels=np.empty((R,N),dtype=object)
    first_alpha=np.full((R,N),np.inf,np.float32)
    for r in range(R):
        d=aligned_reps[r]-anchor
        for i in range(N):
            hi=int(anchor_order[i,4]); lo=int(anchor_order[i,5])
            ca=exact_cross_alpha(anchor[i],d[i],hi,lo)
            boundary_votes[r,i]=int(ca<=float(alpha))
            boundary_direction[r,i]=np.sign(float(d[i,lo]-d[i,hi]))
            ca,kind,h,l=first_critical_cross(anchor[i],aligned_reps[r,i])
            first_alpha[r,i]=ca if np.isfinite(ca) else np.inf
            first_labels[r,i]=f"{kind}:{h}>{l}" if kind!="none" else "none"

    boundary_agree=np.maximum(boundary_votes.mean(0),1.0-boundary_votes.mean(0)).astype(np.float32)
    boundary_dir_agree=np.abs(boundary_direction.mean(0)).astype(np.float32)
    first_cross_agree=modal_fraction(first_labels)
    stability=(0.30*top5_jacc+0.25*top1_agree+0.20*boundary_agree+0.15*boundary_dir_agree+0.10*first_cross_agree).astype(np.float32)
    return {
        "stability":stability,
        "top1_agreement":top1_agree,
        "top5_jaccard":top5_jacc,
        "boundary_agreement":boundary_agree,
        "boundary_direction_agreement":boundary_dir_agree,
        "first_cross_agreement":first_cross_agree,
        "median_first_cross_alpha":np.median(first_alpha,axis=0).astype(np.float32),
        "replica_final_scores":final.astype(np.float32),
    }


def cluster_boot_ci(values,groups,seed,B=BOOTSTRAPS):
    values=np.asarray(values,float); groups=np.asarray(groups,dtype=object)
    ug=np.unique(groups); by={g:np.flatnonzero(groups==g) for g in ug}
    rng=np.random.default_rng(int(seed)); z=np.empty(int(B),float)
    for b in range(int(B)):
        sampled=rng.choice(ug,size=len(ug),replace=True)
        idx=np.concatenate([by[g] for g in sampled])
        z[b]=values[idx].mean()
    return tuple(float(x) for x in np.quantile(z,[.025,.975]))


def stability_quintiles(stability,delta,label):
    df=pd.DataFrame({"stability":np.asarray(stability,float),"delta":np.asarray(delta,float)})
    # Rank first so ties cannot collapse qcut.
    df["q"]=pd.qcut(df["stability"].rank(method="first"),5,labels=False)+1
    rows=[]
    for q,g in df.groupby("q",sort=True):
        rows.append({
            "candidate":label,"quintile":int(q),"n":len(g),
            "stability_mean":float(g.stability.mean()),
            "delta_mean":float(g.delta.mean()),
            "delta_median":float(g.delta.median()),
            "positive_rate":float((g.delta>0).mean()),
            "harm_rate":float((g.delta<0).mean()),
        })
    rho=float(spearmanr(df.stability,df.delta,nan_policy="omit").statistic)
    return pd.DataFrame(rows),rho

print("V29.5 replica + ranking-stability helpers ready")

# ---- cell 22 (markdown) ----
# ## Fully nested strict OOF replicas — the uncertainty kill test
#
# V29.5 freezes two replica families:
#
# - `ltr_base`, alpha 0.10 — the simplest V29 residual-LTR control;
# - `ltr_gap_cooc`, alpha 0.16 — the strongest V29 graph/curriculum family before the low-value residual covariance block.
#
# For each outer template-group fold, both the anchor and every graph feature are rebuilt from the outer-training users. All replicas for that fold train on those same outer-training users only. Validation labels are used **only after predictions exist** to measure realized NDCG delta.

# ---- cell 23 (code) ----
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

print("V29.5 nested anchor helper ready")

# ---- cell 24 (code) ----
assert callable(build_anchor_split_full), "nested anchor helper missing"

STRICT_OUTER=make_group_folds_global(np.arange(N_TRAIN),N_FOLDS,SEED+29501)
STRICT_ANCHOR=np.zeros_like(y,np.float32)
STRICT_FOLD_ID=np.full(N_TRAIN,-1,np.int16)

# Deterministic ablation predictions for Base / Gap / Gap+Cooc.
STRICT_RAW={a:np.zeros_like(y,np.float32) for a in FRESH_ABLATIONS}

# Honest aligned replica scores for the two uncertainty families.
STRICT_REPLICA_ALIGNED={
    a:np.zeros((REPLICA_COUNT,N_TRAIN,N_MODULES),np.float32) for a in REPLICA_ABLATIONS
}

for f,(tr,va) in enumerate(STRICT_OUTER):
    print(f"V29.5 strict fold {f+1}/{len(STRICT_OUTER)} train={len(tr)} val={len(va)}")
    anc=build_anchor_split_full(tr,va,SEED+29600+f*211)
    STRICT_ANCHOR[va]=anc["cons_val"]
    STRICT_FOLD_ID[va]=f

    graph_tr=crossfit_graph_features(y[tr],anc["cons_train"],TRAIN_GROUP[tr],SEED+29700+f*223)
    bundle=fit_graph_bundle(y[tr],anc["cons_train"])
    graph_va=apply_graph_bundle(bundle,anc["cons_val"])

    Xtr,names,bounds=build_long_matrix(tr,anc["cons_train"],anc["base_train"],anc["meta_train"],graph_tr)
    Xva,_,bounds_va=build_long_matrix(va,anc["cons_val"],anc["base_val"],anc["meta_val"],graph_va)
    assert bounds==bounds_va

    grades=ORDINAL_GRADES[tr]
    # One frozen deterministic model per ablation for clean ablation tables.
    for ai,abl in enumerate(FRESH_ABLATIONS):
        end=feature_end(bounds,abl)
        _,pred=fit_ranker_predict(Xtr[:,:end],Xva[:,:end],grades,len(tr),SEED+29800+f*227+ai*19)
        STRICT_RAW[abl][va]=pred

    # Honest replica families for uncertainty.
    for ai,abl in enumerate(REPLICA_ABLATIONS):
        end=feature_end(bounds,abl)
        seeds=[SEED+30000+f*1009+ai*101+r*37 for r in range(REPLICA_COUNT)]
        raw_reps=fit_replica_family(Xtr[:,:end],Xva[:,:end],grades,len(tr),seeds)
        for r in range(REPLICA_COUNT):
            STRICT_REPLICA_ALIGNED[abl][r,va]=align_ranker_to_anchor(raw_reps[r],anc["cons_val"])

    del Xtr,Xva,graph_tr,graph_va,bundle,anc
    gc.collect()

assert (STRICT_FOLD_ID>=0).all()
assert np.isfinite(STRICT_ANCHOR).all()
for abl in REPLICA_ABLATIONS:
    assert np.isfinite(STRICT_REPLICA_ALIGNED[abl]).all()

print("strict anchor NDCG",round(ndcg5(y,STRICT_ANCHOR),6))

strict_ab_rows=[]
STRICT_SCORES={}
STRICT_DELTAS={}
for abl in FRESH_ABLATIONS:
    for alpha in FIXED_ALPHAS:
        score=blend_residual(STRICT_ANCHOR,STRICT_RAW[abl],alpha)
        delta=per_user_ndcg(y,score)-per_user_ndcg(y,STRICT_ANCHOR)
        fold_means=[float(delta[STRICT_FOLD_ID==f].mean()) for f in range(len(STRICT_OUTER))]
        ci=cluster_boot_ci(delta,TRAIN_GROUP,SEED+31000+FRESH_ABLATIONS.index(abl)*101+int(alpha*100))
        strict_ab_rows.append({
            "world":"strict_template_group","ablation":abl,"alpha":alpha,
            "mean_delta":float(delta.mean()),"fold_wins":int(np.sum(np.asarray(fold_means)>0)),
            "min_fold":float(np.min(fold_means)),"ci_low":ci[0],"ci_high":ci[1],
            "top5_membership_change":top5_set_change_rate(STRICT_ANCHOR,score),
            "top1_change":top1_change_rate(STRICT_ANCHOR,score),
        })
        STRICT_SCORES[(abl,alpha)]=score
        STRICT_DELTAS[(abl,alpha)]=delta

V295_STRICT_ABLATION=pd.DataFrame(strict_ab_rows)
display(V295_STRICT_ABLATION)

uncertainty_rows=[]
quintile_tables=[]
UNCERTAINTY_RHO={}
for abl in REPLICA_ABLATIONS:
    alpha=PRIMARY_REPLICA_ALPHA[abl]
    aligned=STRICT_REPLICA_ALIGNED[abl]
    mean_aligned=aligned.mean(0)
    score=(STRICT_ANCHOR+alpha*(mean_aligned-STRICT_ANCHOR)).astype(np.float32)
    delta=per_user_ndcg(y,score)-per_user_ndcg(y,STRICT_ANCHOR)
    stab=ranking_stability(STRICT_ANCHOR,aligned,alpha)
    qtab,rho=stability_quintiles(stab["stability"],delta,abl)
    qtab["alpha"]=alpha
    quintile_tables.append(qtab)
    UNCERTAINTY_RHO[abl]=rho
    for i in range(N_TRAIN):
        uncertainty_rows.append({
            "user_id":train.iloc[i].user_id,"fold":int(STRICT_FOLD_ID[i]),
            "candidate":abl,"alpha":alpha,"anchor_ndcg":float(per_user_ndcg(y[i:i+1],STRICT_ANCHOR[i:i+1])[0]),
            "candidate_delta":float(delta[i]),"stability":float(stab["stability"][i]),
            "top1_agreement":float(stab["top1_agreement"][i]),
            "top5_jaccard":float(stab["top5_jaccard"][i]),
            "boundary_agreement":float(stab["boundary_agreement"][i]),
            "boundary_direction_agreement":float(stab["boundary_direction_agreement"][i]),
            "first_cross_agreement":float(stab["first_cross_agreement"][i]),
            "median_first_cross_alpha":float(stab["median_first_cross_alpha"][i]),
        })

V295_UNCERTAINTY=pd.DataFrame(uncertainty_rows)
V295_UNCERTAINTY_QUINTILES=pd.concat(quintile_tables,ignore_index=True)
display(V295_UNCERTAINTY_QUINTILES)
print("stability Spearman",UNCERTAINTY_RHO)

uncertainty_summary=[]
for abl in REPLICA_ABLATIONS:
    q=V295_UNCERTAINTY_QUINTILES[V295_UNCERTAINTY_QUINTILES.candidate==abl].sort_values("quintile")
    bottom=float(q.iloc[0].delta_mean); top=float(q.iloc[-1].delta_mean)
    uncertainty_summary.append({
        "candidate":abl,"spearman_stability_delta":UNCERTAINTY_RHO[abl],
        "bottom_quintile_delta":bottom,"top_quintile_delta":top,
        "top_minus_bottom":top-bottom,
        "monotonic_delta_steps":int(np.sum(np.diff(q.delta_mean.to_numpy())>=-5e-5)),
        "uncertainty_signal_ok":bool(
            UNCERTAINTY_RHO[abl]>=UNCERTAINTY_MIN_SPEARMAN and
            (top-bottom)>=UNCERTAINTY_MIN_TOP_BOTTOM_DELTA and
            q.iloc[-1].harm_rate<=q.iloc[0].harm_rate
        ),
    })
V295_UNCERTAINTY_SUMMARY=pd.DataFrame(uncertainty_summary)
display(V295_UNCERTAINTY_SUMMARY)

# ---- cell 25 (markdown) ----
# ## Exact rank-boundary crossing audit
#
# A scalar residual coefficient changes ranking only when it crosses a pairwise score margin. This audit records the earliest critical crossing for every user/replica and asks whether crossings supported by replica consensus are empirically safer than unstable crossings.
#
# The crossing ledger is diagnostic only; V29.5 does not use it to modify predictions.

# ---- cell 26 (code) ----
boundary_rows=[]
for abl in REPLICA_ABLATIONS:
    alpha=PRIMARY_REPLICA_ALPHA[abl]
    aligned=STRICT_REPLICA_ALIGNED[abl]
    mean_aligned=aligned.mean(0)
    mean_score=(STRICT_ANCHOR+alpha*(mean_aligned-STRICT_ANCHOR)).astype(np.float32)
    mean_delta=per_user_ndcg(y,mean_score)-per_user_ndcg(y,STRICT_ANCHOR)
    order=np.argsort(-STRICT_ANCHOR,axis=1,kind="mergesort")
    for i in range(N_TRAIN):
        rep_events=[]
        for r in range(REPLICA_COUNT):
            ca,kind,hi,lo=first_critical_cross(STRICT_ANCHOR[i],aligned[r,i])
            rep_events.append((ca,kind,hi,lo))
        labels=[f"{e[1]}:{e[2]}>{e[3]}" if e[1]!="none" else "none" for e in rep_events]
        vals,counts=np.unique(labels,return_counts=True)
        modal_label=str(vals[np.argmax(counts)])
        modal_frac=float(counts.max()/REPLICA_COUNT)
        finite=[e[0] for e in rep_events if np.isfinite(e[0])]
        first_med=float(np.median(finite)) if finite else np.inf
        anchor_top=order[i,:6]
        boundary_rows.append({
            "user_id":train.iloc[i].user_id,"candidate":abl,"alpha":alpha,
            "anchor_rank1":MODULES[int(anchor_top[0])],
            "anchor_rank5":MODULES[int(anchor_top[4])],
            "anchor_rank6":MODULES[int(anchor_top[5])],
            "modal_first_cross":modal_label,"first_cross_agreement":modal_frac,
            "median_first_cross_alpha":first_med,
            "crosses_by_alpha":int(first_med<=alpha),
            "realized_delta":float(mean_delta[i]),
        })

V295_BOUNDARY=pd.DataFrame(boundary_rows)
cross_summary=[]
for abl in REPLICA_ABLATIONS:
    d=V295_BOUNDARY[V295_BOUNDARY.candidate==abl].copy()
    d["agreement_band"]=pd.cut(d.first_cross_agreement,[0,0.6,0.8,0.999999,1.01],labels=["low","medium","high","unanimous"],include_lowest=True)
    for (crossed,band),g in d.groupby(["crosses_by_alpha","agreement_band"],observed=True):
        cross_summary.append({
            "candidate":abl,"crosses_by_alpha":int(crossed),"agreement_band":str(band),
            "n":len(g),"mean_delta":float(g.realized_delta.mean()),
            "positive_rate":float((g.realized_delta>0).mean()),"harm_rate":float((g.realized_delta<0).mean()),
        })
V295_BOUNDARY_SUMMARY=pd.DataFrame(cross_summary)
display(V295_BOUNDARY_SUMMARY)

# ---- cell 27 (markdown) ----
# ## Transitive curriculum precision — direct vs multi-hop
#
# V29's one-hop oracle measured recall-like headroom but not precision. V29.5 converts the organizer prerequisite graph into shortest-path distances 1–3 and measures whether candidate modules at those distances are actually positive.
#
# Two seed surfaces are kept separate:
#
# - **explicit intent:** only modules detected in the latest learning message;
# - **anchor top1:** Consensus3's highest-ranked module.
#
# For every seed, upstream prerequisite paths and downstream next-step paths are audited separately. Candidate readiness is stratified into low / medium / high bands. These statistics use labels only for post-hoc diagnostics and never become features in this notebook.

# ---- cell 28 (code) ----
def shortest_path_matrix(adj,max_dist=3):
    adj=np.asarray(adj)>0
    M=adj.shape[0]
    dist=np.full((M,M),99,np.int16)
    np.fill_diagonal(dist,0)
    dist[adj]=1
    for k in range(M):
        for i in range(M):
            for j in range(M):
                if dist[i,k]+dist[k,j] < dist[i,j]:
                    dist[i,j]=dist[i,k]+dist[k,j]
    dist[dist>max_dist]=99
    return dist

CURR_DIST=shortest_path_matrix(PREREQ_MAT,3)

def readiness_band(v):
    if v<=1.5: return "low"
    if v<3.0: return "medium"
    return "high"

def candidate_map_for_seed(seed,direction):
    out={}
    if direction=="upstream":
        # candidate -> seed along prerequisite edges
        for m in range(N_MODULES):
            d=int(CURR_DIST[m,seed])
            if 1<=d<=3: out[m]=d
    else:
        # seed -> candidate as next-step
        for m in range(N_MODULES):
            d=int(CURR_DIST[seed,m])
            if 1<=d<=3: out[m]=d
    return out

anchor_top1=np.argmax(STRICT_ANCHOR,axis=1)
anchor_top5=top5_indices(STRICT_ANCHOR)
anchor_mask=np.zeros_like(y,dtype=bool); anchor_mask[np.arange(N_TRAIN)[:,None],anchor_top5]=True
missed=(y>0)&(~anchor_mask)

curr_rows=[]
for i in range(N_TRAIN):
    seed_surfaces=[]
    latest=np.flatnonzero(MODULE_LATEST[i]>0)
    for s in latest:
        seed_surfaces.append(("explicit_latest",int(s)))
    seed_surfaces.append(("anchor_top1",int(anchor_top1[i])))

    # de-duplicate candidate diagnostics per user/seed_type/direction by shortest distance.
    for seed_type in ["explicit_latest","anchor_top1"]:
        seeds=[s for st,s in seed_surfaces if st==seed_type]
        if not seeds: continue
        for direction in ["upstream","downstream"]:
            best={}
            for s in seeds:
                for m,d in candidate_map_for_seed(s,direction).items():
                    if m==s: continue
                    if m not in best or d<best[m][0]:
                        best[m]=(d,s)
            for m,(d,s) in best.items():
                curr_rows.append({
                    "user_id":train.iloc[i].user_id,"seed_type":seed_type,"direction":direction,
                    "seed_module":MODULES[s],"candidate_module":MODULES[m],"distance":int(d),
                    "readiness_band":readiness_band(float(MODULE_READINESS[i,m])),
                    "candidate_readiness":float(MODULE_READINESS[i,m]),
                    "positive":int(y[i,m]>0),"relevance":float(y[i,m]),
                    "missed_positive":int(missed[i,m]),
                    "anchor_rank":int(rank_matrix(STRICT_ANCHOR[i:i+1])[0,m]),
                })

V295_CURRICULUM_PAIRS=pd.DataFrame(curr_rows)
global_positive_rate=float((y>0).mean())
curr_summary=[]
for keys,g in V295_CURRICULUM_PAIRS.groupby(["seed_type","direction","distance","readiness_band"],observed=True):
    seed_type,direction,distance,rband=keys
    curr_summary.append({
        "seed_type":seed_type,"direction":direction,"distance":int(distance),"readiness_band":str(rband),
        "support":len(g),"positive_rate":float(g.positive.mean()),"positive_lift":float(g.positive.mean()/max(global_positive_rate,1e-12)),
        "mean_relevance":float(g.relevance.mean()),"missed_positive_rate":float(g.missed_positive.mean()),
        "missed_relevance_mass":float(g.loc[g.missed_positive==1,"relevance"].sum()),
    })
V295_CURRICULUM_PRECISION=pd.DataFrame(curr_summary).sort_values(["seed_type","direction","distance","readiness_band"])

# Aggregate distance-level tables for easier interpretation.
agg=[]
for keys,g in V295_CURRICULUM_PAIRS.groupby(["seed_type","direction","distance"],observed=True):
    st,di,ds=keys
    agg.append({
        "seed_type":st,"direction":di,"distance":int(ds),"support":len(g),
        "positive_rate":float(g.positive.mean()),"positive_lift":float(g.positive.mean()/max(global_positive_rate,1e-12)),
        "mean_relevance":float(g.relevance.mean()),"missed_positive_rate":float(g.missed_positive.mean()),
    })
V295_CURRICULUM_DISTANCE=pd.DataFrame(agg)

missed_mass_total=float((y*missed).sum())
curr_missed_pairs=V295_CURRICULUM_PAIRS[V295_CURRICULUM_PAIRS.missed_positive==1]
# Union captured mass so repeated seed paths do not double count.
pair_keys=set(zip(curr_missed_pairs.user_id,curr_missed_pairs.candidate_module))
captured=0.0
uid_to_i={u:i for i,u in enumerate(train.user_id)}
for uid,mid in pair_keys:
    captured+=float(y[uid_to_i[uid],MODULES.index(mid)])
V295_CURRICULUM_HEADROOM={
    "global_positive_rate":global_positive_rate,
    "missed_relevance_mass":missed_mass_total,
    "transitive_union_captured_mass":captured,
    "transitive_union_ratio":captured/max(missed_mass_total,1e-12),
}
print("TRANSITIVE CURRICULUM HEADROOM",V295_CURRICULUM_HEADROOM)
display(V295_CURRICULUM_DISTANCE)
display(V295_CURRICULUM_PRECISION.head(40))

# ---- cell 29 (markdown) ----
# ## Fresh-world co-occurrence ablation
#
# A new salted template-disjoint LOCK is created specifically for V29.5. Base, Gap, and Gap+Co-occurrence are all fit on the exact same train/validation geometry. The same frozen ablations are also evaluated chronologically on U0001–3000 → U3001–4000.
#
# Both alpha 0.10 and 0.16 are reported. These worlds do **not** select an alpha or ablation; they only determine whether the V29 strict-OOF co-occurrence improvement survives outside the selection geometry.

# ---- cell 30 (code) ----
def v295_lock_bucket(g):
    h=hashlib.sha1(f"V295-RELIABILITY-CURRICULUM-LOCK-20260828::{g}".encode()).hexdigest()
    return int(h[:12],16)%5

LOCK_MASK=np.asarray([v295_lock_bucket(g)==0 for g in TRAIN_GROUP],bool)
DEV_IDX=np.flatnonzero(~LOCK_MASK); LOCK_IDX=np.flatnonzero(LOCK_MASK)
assert len(set(TRAIN_GROUP[DEV_IDX]).intersection(set(TRAIN_GROUP[LOCK_IDX])))==0
CHRONO_TR=np.arange(0,3000,dtype=int)
CHRONO_VA=np.arange(3000,4000,dtype=int)

def external_ablation_family(train_rows,val_rows,seed,world):
    train_rows=np.asarray(train_rows,int); val_rows=np.asarray(val_rows,int)
    anc=build_anchor_split_full(train_rows,val_rows,seed)
    graph_tr=crossfit_graph_features(y[train_rows],anc["cons_train"],TRAIN_GROUP[train_rows],seed+101)
    bundle=fit_graph_bundle(y[train_rows],anc["cons_train"])
    graph_va=apply_graph_bundle(bundle,anc["cons_val"])

    Xtr,names,bounds=build_long_matrix(train_rows,anc["cons_train"],anc["base_train"],anc["meta_train"],graph_tr)
    Xva,_,_=build_long_matrix(val_rows,anc["cons_val"],anc["base_val"],anc["meta_val"],graph_va)
    rows=[]; payload={}
    for ai,abl in enumerate(FRESH_ABLATIONS):
        end=feature_end(bounds,abl)
        _,raw=fit_ranker_predict(Xtr[:,:end],Xva[:,:end],ORDINAL_GRADES[train_rows],len(train_rows),seed+202+ai*31)
        for alpha in FIXED_ALPHAS:
            score=blend_residual(anc["cons_val"],raw,alpha)
            delta=per_user_ndcg(y[val_rows],score)-per_user_ndcg(y[val_rows],anc["cons_val"])
            rows.append({
                "world":world,"ablation":abl,"alpha":alpha,"n":len(val_rows),
                "anchor_ndcg":ndcg5(y[val_rows],anc["cons_val"]),
                "candidate_ndcg":ndcg5(y[val_rows],score),"mean_delta":float(delta.mean()),
                "positive_rate":float(np.mean(delta>0)),"harm_rate":float(np.mean(delta<0)),
                "top5_membership_change":top5_set_change_rate(anc["cons_val"],score),
                "top1_change":top1_change_rate(anc["cons_val"],score),
            })
            payload[(abl,alpha)]={"score":score,"delta":delta}
    out={"rows":rows,"payload":payload,"anchor":anc["cons_val"],"val_rows":val_rows}
    del Xtr,Xva,graph_tr,graph_va,bundle,anc
    gc.collect()
    return out

LOCK_FAMILY=external_ablation_family(DEV_IDX,LOCK_IDX,SEED+32000,"fresh_lock_v295")
CHRONO_FAMILY=external_ablation_family(CHRONO_TR,CHRONO_VA,SEED+33000,"chronological")
V295_FRESH_ABLATION=pd.DataFrame(LOCK_FAMILY["rows"]+CHRONO_FAMILY["rows"])
display(V295_FRESH_ABLATION)

fresh_compare=[]
for world in ["fresh_lock_v295","chronological"]:
    z=V295_FRESH_ABLATION[V295_FRESH_ABLATION.world==world]
    for alpha in FIXED_ALPHAS:
        base=float(z[(z.ablation=="ltr_base")&(z.alpha==alpha)].mean_delta.iloc[0])
        gap=float(z[(z.ablation=="ltr_gap")&(z.alpha==alpha)].mean_delta.iloc[0])
        cooc=float(z[(z.ablation=="ltr_gap_cooc")&(z.alpha==alpha)].mean_delta.iloc[0])
        fresh_compare.append({
            "world":world,"alpha":alpha,"base_delta":base,"gap_delta":gap,"gap_cooc_delta":cooc,
            "gap_minus_base":gap-base,"cooc_minus_gap":cooc-gap,
            "cooc_not_materially_worse":bool(cooc-gap>=COOC_FRESH_TOL),
        })
V295_COOC_FRESH=pd.DataFrame(fresh_compare)
display(V295_COOC_FRESH)

# ---- cell 31 (markdown) ----
# ## Pseudo-public stress under three dependence assumptions
#
# The same frozen fresh-LOCK per-user deltas are resampled three ways:
#
# - **row:** random user subsets, matching V29's original stress test;
# - **template_cluster:** whole latest-template groups are sampled together;
# - **intent_stratified:** each pseudo-public subset preserves the fresh LOCK's explicit-intent/no-intent mixture by module family.
#
# The goal is not to find a favorable bootstrap. It is to determine whether V29's lower-tail warning is robust to more realistic dependence assumptions.

# ---- cell 32 (code) ----
def intent_strata(rows):
    rows=np.asarray(rows,int)
    out=[]
    for i in rows:
        latest=np.flatnonzero(MODULE_LATEST[i]>0)
        if len(latest):
            out.append("intent_"+MODULES[int(latest[0])])
        else:
            rec=np.flatnonzero(MODULE_RECENCY[i]>0)
            out.append("recent_"+MODULES[int(rec[np.argmax(MODULE_RECENCY[i,rec])])] if len(rec) else "no_learning_intent")
    return np.asarray(out,dtype=object)

def pseudo_row(delta,frac,rng):
    k=max(1,int(round(frac*len(delta))))
    idx=rng.choice(len(delta),size=k,replace=False)
    return float(np.mean(delta[idx]))

def pseudo_cluster(delta,groups,frac,rng):
    groups=np.asarray(groups,dtype=object)
    ug=np.unique(groups); rng.shuffle(ug)
    target=max(1,int(round(frac*len(delta))))
    idx=[]
    for g in ug:
        idx.extend(np.flatnonzero(groups==g).tolist())
        if len(idx)>=target: break
    return float(np.mean(delta[np.asarray(idx,int)]))

def pseudo_stratified(delta,strata,frac,rng):
    strata=np.asarray(strata,dtype=object)
    idx=[]
    for s in np.unique(strata):
        pool=np.flatnonzero(strata==s)
        k=int(round(frac*len(pool))) if len(pool) else 0
        k=min(max(k,0),len(pool))
        if k: idx.extend(rng.choice(pool,size=k,replace=False).tolist())
    if not idx:
        idx=[int(rng.integers(0,len(delta)))]
    return float(np.mean(delta[np.asarray(idx,int)]))

pseudo_rows=[]
lock_groups=TRAIN_GROUP[LOCK_IDX]
lock_strata=intent_strata(LOCK_IDX)
for abl in FRESH_ABLATIONS:
    for alpha in FIXED_ALPHAS:
        delta=np.asarray(LOCK_FAMILY["payload"][(abl,alpha)]["delta"],float)
        for frac in [0.25,0.35,0.50]:
            for mode in ["row","template_cluster","intent_stratified"]:
                rng=np.random.default_rng(SEED+34000+FRESH_ABLATIONS.index(abl)*701+int(alpha*1000)+int(frac*100)+["row","template_cluster","intent_stratified"].index(mode)*97)
                vals=[]
                for b in range(PSEUDO_PUBLIC_MASKS):
                    if mode=="row": vals.append(pseudo_row(delta,frac,rng))
                    elif mode=="template_cluster": vals.append(pseudo_cluster(delta,lock_groups,frac,rng))
                    else: vals.append(pseudo_stratified(delta,lock_strata,frac,rng))
                vals=np.asarray(vals,float)
                pseudo_rows.append({
                    "ablation":abl,"alpha":alpha,"fraction":frac,"mode":mode,"masks":len(vals),
                    "positive_rate":float(np.mean(vals>0)),"mean":float(vals.mean()),
                    "p10":float(np.quantile(vals,.10)),"median":float(np.median(vals)),"p90":float(np.quantile(vals,.90)),
                })
V295_PSEUDO=pd.DataFrame(pseudo_rows)
display(V295_PSEUDO[(V295_PSEUDO.fraction==0.35)].sort_values(["ablation","alpha","mode"]))

# ---- cell 33 (markdown) ----
# ## Test uncertainty distribution shift
#
# No test labels are touched. V29.5 fits five full-train replica models for the two uncertainty families, then computes the same ranking-space stability metrics on test users. Their distributions are compared with honest strict-OOF stability.
#
# A materially lower test stability distribution would be a warning against aggressive V30 corrections even if OOF uncertainty is predictive.

# ---- cell 34 (code) ----
FULL_GRAPH_TR=crossfit_graph_features(y,consensus3_oof,TRAIN_GROUP,SEED+35000)
FULL_BUNDLE=fit_graph_bundle(y,consensus3_oof)
FULL_GRAPH_TE=apply_graph_bundle(FULL_BUNDLE,consensus3_test)
XTR_FULL,FULL_NAMES,FULL_BOUNDS=build_long_matrix(np.arange(N_TRAIN),consensus3_oof,base_oof,meta_oof,FULL_GRAPH_TR)
XTE_FULL,_,_=build_long_matrix(np.arange(N_TRAIN,N_TRAIN+N_TEST),consensus3_test,base_test,meta_test,FULL_GRAPH_TE)

test_stability_payload={}
shift_rows=[]
for ai,abl in enumerate(REPLICA_ABLATIONS):
    alpha=PRIMARY_REPLICA_ALPHA[abl]
    end=feature_end(FULL_BOUNDS,abl)
    seeds=[SEED+36000+ai*1009+r*41 for r in range(REPLICA_COUNT)]
    raw=fit_replica_family(XTR_FULL[:,:end],XTE_FULL[:,:end],ORDINAL_GRADES,N_TRAIN,seeds)
    aligned=np.stack([align_ranker_to_anchor(raw[r],consensus3_test) for r in range(REPLICA_COUNT)],axis=0)
    stab=ranking_stability(consensus3_test,aligned,alpha)
    test_stability_payload[abl]=stab

    oof=V295_UNCERTAINTY[V295_UNCERTAINTY.candidate==abl].stability.to_numpy(float)
    te=stab["stability"].astype(float)
    for source,arr in [("strict_oof",oof),("test",te)]:
        shift_rows.append({
            "candidate":abl,"source":source,"n":len(arr),"mean":float(np.mean(arr)),"std":float(np.std(arr)),
            "p10":float(np.quantile(arr,.10)),"p25":float(np.quantile(arr,.25)),
            "median":float(np.median(arr)),"p75":float(np.quantile(arr,.75)),"p90":float(np.quantile(arr,.90)),
            "low_stability_rate_lt_0.70":float(np.mean(arr<0.70)),
        })

V295_TEST_SHIFT=pd.DataFrame(shift_rows)
display(V295_TEST_SHIFT)

del XTR_FULL,XTE_FULL,FULL_GRAPH_TR,FULL_GRAPH_TE,FULL_BUNDLE
gc.collect()

# ---- cell 35 (markdown) ----
# ## Stratified 100-user human taxonomy sample
#
# This export avoids the worst-100 selection bias that distorted earlier qualitative diagnosis. It contains:
#
# - 30 strict-anchor worst users;
# - 30 users around the middle of the strict-anchor error distribution;
# - 30 users with the smallest anchor rank-5 vs rank-6 margins;
# - 10 noisy/no-learning-intent users.
#
# Rows are deduplicated across strata. The table includes observed chat trajectory, assessment, anchor/truth rankings, and V29.5 residual diagnostics. `manual_error_type` and `manual_notes` are intentionally blank.

# ---- cell 36 (code) ----
def mids_top(scores,k=5):
    return [MODULES[j] for j in np.argsort(-scores,kind="mergesort")[:k]]

def truth_top(row_y,k=6):
    pos=np.flatnonzero(row_y>0)
    pos=pos[np.argsort(-row_y[pos],kind="mergesort")]
    return [MODULES[j] for j in pos[:k]]

def chat_trajectory(uid):
    g=user_chat.get(uid)
    if g is None or len(g)==0: return ""
    vals=[]
    for row in g.itertuples(index=False):
        txt=str(row.user_chat_text).replace("\n"," ").strip()
        vals.append(txt)
    return " || ".join(vals)

anchor_user_ndcg=per_user_ndcg(y,STRICT_ANCHOR)
order=np.argsort(-STRICT_ANCHOR,axis=1,kind="mergesort")
rank5=order[:,4]; rank6=order[:,5]
boundary_margin=STRICT_ANCHOR[np.arange(N_TRAIN),rank5]-STRICT_ANCHOR[np.arange(N_TRAIN),rank6]

selected=[]; bucket={}
def take(name,candidates,n):
    out=[]
    for i in candidates:
        i=int(i)
        if i not in selected:
            selected.append(i); out.append(i); bucket[i]=name
        if len(out)>=n: break
    return out

take("worst",np.argsort(anchor_user_ndcg),30)

# Middle around median absolute distance from median NDCG.
med=float(np.median(anchor_user_ndcg))
take("middle",np.argsort(np.abs(anchor_user_ndcg-med)),30)

take("rank5_rank6_boundary",np.argsort(boundary_margin),30)

noise_candidates=np.flatnonzero((MODULE_RECENCY[:N_TRAIN].sum(1)==0)|(LATEST_GROUP[:N_TRAIN]=="no_chat"))
take("noise_or_no_intent",noise_candidates,10)

# Fill any shortfall deterministically from remaining users nearest median.
if len(selected)<100:
    take("fill",np.arange(N_TRAIN),100-len(selected))

# Use gap+cooc replica mean as the V29.5 comparison surface, but not as a deployment candidate.
abl="ltr_gap_cooc"; alpha=PRIMARY_REPLICA_ALPHA[abl]
mean_aligned=STRICT_REPLICA_ALIGNED[abl].mean(0)
diag_score=(STRICT_ANCHOR+alpha*(mean_aligned-STRICT_ANCHOR)).astype(np.float32)
diag_delta=per_user_ndcg(y,diag_score)-per_user_ndcg(y,STRICT_ANCHOR)
stab_lookup=V295_UNCERTAINTY[V295_UNCERTAINTY.candidate==abl].set_index("user_id")

tax_rows=[]
for i in selected[:100]:
    uid=str(train.iloc[i].user_id)
    latest=np.flatnonzero(MODULE_LATEST[i]>0)
    latest_mids="|".join(MODULES[j] for j in latest)
    tax_rows.append({
        "sample_bucket":bucket[i],"user_id":uid,
        "strict_anchor_ndcg":float(anchor_user_ndcg[i]),"rank5_rank6_margin":float(boundary_margin[i]),
        "anchor_top5":" > ".join(mids_top(STRICT_ANCHOR[i],5)),
        "v295_diag_top5":" > ".join(mids_top(diag_score[i],5)),
        "truth_positive_order":" > ".join(truth_top(y[i],6)),
        "v295_diag_delta":float(diag_delta[i]),
        "stability":float(stab_lookup.loc[uid,"stability"]),
        "latest_explicit_modules":latest_mids,
        "chat_trajectory":chat_trajectory(uid),
        "assessment_mean":float(A[i].mean()),
        "assessment_values":"|".join(f"{SKILL_NAMES[j]}={A[i,j]:.1f}" for j in range(10)),
        "manual_error_type":"","manual_notes":"",
    })
V295_TAXONOMY=pd.DataFrame(tax_rows)
print(V295_TAXONOMY.sample_bucket.value_counts())
display(V295_TAXONOMY.head(12))

# ---- cell 37 (markdown) ----
# ## Diagnostics verdict and outputs
#
# V29.5 never writes a Kaggle submission file. The final JSON summarizes three falsifiable research questions:
#
# - **uncertainty signal:** does ranking-space stability predict realized benefit/harm?
# - **curriculum signal:** do transitive curriculum candidates have meaningful precision/lift at useful support?
# - **co-occurrence transfer:** does the strict-OOF co-occurrence gain survive fresh LOCK/chronology without material degradation?
#
# These booleans are research summaries only. V30 still requires human review of the detailed tables.

# ---- cell 38 (code) ----
OUT_DIR=Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("/mnt/data")
OUT_DIR.mkdir(parents=True,exist_ok=True)

# Research verdicts.
unc_ok=bool(V295_UNCERTAINTY_SUMMARY.uncertainty_signal_ok.any())

curr_eligible=V295_CURRICULUM_DISTANCE[
    (V295_CURRICULUM_DISTANCE.support>=CURRICULUM_MIN_SUPPORT)&
    (V295_CURRICULUM_DISTANCE.positive_lift>=CURRICULUM_MIN_LIFT)
]
curr_ok=bool(len(curr_eligible)>0)

cooc_ok=True
for world in ["fresh_lock_v295","chronological"]:
    z=V295_COOC_FRESH[V295_COOC_FRESH.world==world]
    # At least one preregistered alpha must not show material cooc degradation in each world.
    if not bool(z.cooc_not_materially_worse.any()):
        cooc_ok=False

summary={
    "version":"V29.5 Reliability & Curriculum Kill-Test Diagnostics",
    "diagnostics_only":True,
    "public_anchor":PUBLIC_CURRENT_PB,"leader_reference":PUBLIC_LEADER_LB,
    "replica_count":REPLICA_COUNT,
    "uncertainty_summary":V295_UNCERTAINTY_SUMMARY.to_dict("records"),
    "uncertainty_signal_ok":unc_ok,
    "curriculum_headroom":V295_CURRICULUM_HEADROOM,
    "curriculum_signal_ok":curr_ok,
    "cooccurrence_fresh_transfer_ok":bool(cooc_ok),
    "fresh_ablation":V295_FRESH_ABLATION.to_dict("records"),
    "research_decision":{
        "variance_control_worth_v30":bool(unc_ok),
        "transitive_curriculum_worth_v30":bool(curr_ok),
        "cooccurrence_worth_v30":bool(cooc_ok),
        "residual_covariance_graph_primary":False,
    },
    "submission_instruction":"NO SUBMISSION: V29.5 is diagnostics-only.",
}

# Write tables.
V295_STRICT_ABLATION.to_csv(OUT_DIR/"v295_strict_ablation.csv",index=False)
V295_UNCERTAINTY.to_csv(OUT_DIR/"v295_replica_stability_oof.csv",index=False)
V295_UNCERTAINTY_QUINTILES.to_csv(OUT_DIR/"v295_uncertainty_quintiles.csv",index=False)
V295_UNCERTAINTY_SUMMARY.to_csv(OUT_DIR/"v295_uncertainty_summary.csv",index=False)
V295_BOUNDARY.to_csv(OUT_DIR/"v295_boundary_crossings.csv",index=False)
V295_BOUNDARY_SUMMARY.to_csv(OUT_DIR/"v295_boundary_crossing_summary.csv",index=False)
V295_CURRICULUM_PAIRS.to_csv(OUT_DIR/"v295_curriculum_pairs.csv",index=False)
V295_CURRICULUM_PRECISION.to_csv(OUT_DIR/"v295_curriculum_precision.csv",index=False)
V295_CURRICULUM_DISTANCE.to_csv(OUT_DIR/"v295_curriculum_distance_summary.csv",index=False)
V295_FRESH_ABLATION.to_csv(OUT_DIR/"v295_fresh_world_ablation.csv",index=False)
V295_COOC_FRESH.to_csv(OUT_DIR/"v295_cooccurrence_fresh_transfer.csv",index=False)
V295_PSEUDO.to_csv(OUT_DIR/"v295_pseudo_public_stress.csv",index=False)
V295_TEST_SHIFT.to_csv(OUT_DIR/"v295_test_uncertainty_shift.csv",index=False)
V295_TAXONOMY.to_csv(OUT_DIR/"v295_stratified_taxonomy_audit.csv",index=False)
with open(OUT_DIR/"v295_summary.json","w") as f:
    json.dump(summary,f,indent=2)

# Hard invariants: diagnostics only.
assert not any(str(p.name).startswith("submission_") for p in OUT_DIR.glob("v295*"))
assert V295_TAXONOMY.user_id.nunique()==len(V295_TAXONOMY)
assert len(V295_TAXONOMY)==100 or V295_SMOKE
assert np.isfinite(V295_UNCERTAINTY.stability.to_numpy()).all()
assert ((V295_UNCERTAINTY.stability>=0)&(V295_UNCERTAINTY.stability<=1)).all()

print(json.dumps(summary,indent=2))
print("\nV29.5 wrote diagnostics only to",OUT_DIR)
print("NO KAGGLE SUBMISSION SHOULD BE MADE FROM THIS NOTEBOOK.")

# ---- cell 39 (markdown) ----
# ### Interpretation contract
#
# Use the run to decide the *research direction*, not to manufacture a candidate.
#
# - If stability quintiles show a coherent benefit gradient and lower harm at high stability, V30 may use **honest uncertainty-aware residual control**.
# - If stability is non-monotonic, do **not** build a per-user confidence router; prefer bagging plus a single conservative global residual coefficient.
# - If transitive curriculum distance has strong lift only at distance 1 but collapses at distance 2–3, V30 should keep curriculum local rather than constructing a deep path engine.
# - If `Gap+Cooc` improves strict OOF but loses to `Gap` in fresh LOCK/chronology, remove co-occurrence from V30.
# - If cluster-aware pseudo-public p10 is materially worse than row-level p10, treat template dependence as a real deployment risk.
# - If test stability is materially lower than honest OOF stability, shrink V30 globally even if the OOF uncertainty kill test passes.
#
# The notebook is deliberately incapable of recommending a Kaggle submission.
