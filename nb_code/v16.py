# Converted from notebooks/v16.ipynb: markdown cells become comments, %/! magics neutralized.

# ---- cell 0 (markdown) ----
# # MineToday V16 — Qwen3-Embedding-8B Semantic Calibration
#
# **Goal:** add a genuinely new semantic first-stage signal to the exact public-winning V12.2 **Consensus3** family without repeating the failure modes of V13–V15 or the early Qwen baseline.
#
# ### What is intentionally different from V1
# V1 used a small 0.6B Qwen embedding backbone truncated to 256 dimensions, simple all-history/latest cosine, a separate 0.6B reranker, raw embedding concatenation, LambdaRank, and OOF random blend search. V16 does **none** of those things. It uses a frozen **Qwen3-Embedding-8B** at its native 4096-dimensional representation, three recency-specialized learner views, organizer-supervised semantic linear heads and leakage-safe semantic KNN priors, a regularized **binary relevance** calibrator, fixed smooth fusion into Consensus3, and a new fold-matched nested audit.
#
# ### Technical Meeting compliance
# The MineToday Technical Meeting allows **open-source pretrained** models and multiple ML models, forbids external task data, and forbids using an LLM/Generative AI to directly output test values. V16 uses only the organizer CSV files as task-specific data. `Qwen/Qwen3-Embedding-8B` is used only as a frozen Apache-2.0 feature extractor: it produces vectors, never generated answers and never directly outputs test relevance labels/scores. All task-specific supervision comes only from `train_relevance.csv`.
#
# ### Kaggle setup
# - Accelerator: **GPU T4 x2**
# - Internet: **ON** (to download the open-source pretrained weights)
# - Run all cells top-to-bottom.
# - Full mode uses FP16 model sharding across both T4s, then frees the model before LightGBM validation.
# - `V16_SMOKE=1` is only a local pipeline test and must stay **0** for the real Kaggle run.
#
# ### Safety principle
# V12.2 Consensus3 is immutable. V16 evaluates only four predeclared smooth semantic strengths. The primary submission is **fail-closed** to exact Consensus3 unless a semantic candidate clears ordinary OOF, a new fold-matched nested audit, chronological validation, uncertainty, and deployment-shift gates.

# ---- cell 1 (code) ----
# Reproducible dependencies. In a normal Kaggle run this installs only runtime libraries,
# never any external task dataset.
import os, sys, subprocess
V16_SKIP_INSTALL=bool(int(os.getenv("V16_SKIP_INSTALL","0")))
if not V16_SKIP_INSTALL:
    subprocess.check_call([
        sys.executable,"-m","pip","install","-q","-U",
        "transformers==4.57.6","accelerate==1.14.0","safetensors>=0.5,<1",
        "huggingface_hub>=0.34,<2","lightgbm>=4.5,<5","sentencepiece>=0.2,<1"
    ])
print("dependency setup complete; skip_install=",V16_SKIP_INSTALL)

# ---- cell 2 (code) ----
import os,re,json,ast,html,hashlib,random,warnings,gc,time,sys,subprocess,math
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS","4")
os.environ.setdefault("OPENBLAS_NUM_THREADS","4")
os.environ.setdefault("MKL_NUM_THREADS","4")
os.environ["PYTHONHASHSEED"]="42"
os.environ["TOKENIZERS_PARALLELISM"]="false"

import numpy as np, pandas as pd
from IPython.display import display
warnings.filterwarnings("ignore")
from scipy.sparse import hstack
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
import lightgbm as lgb

SEED=42
AUDIT_SEED=20260824
V16_SMOKE=bool(int(os.getenv("V16_SMOKE","0")))
V16_SKIP_QWEN=bool(int(os.getenv("V16_SKIP_QWEN","0")))
V91_SMOKE=V16_SMOKE  # compatibility with the exact V3.1 sparse cell
N_FOLDS=2 if V16_SMOKE else 5
N_INNER=2 if V16_SMOKE else 3
RIDGE_ALPHA=3.0
REFERENCE_W=0.45
LGB_ESTIMATORS=8 if V16_SMOKE else 1200
LGB_EARLY_STOP=2 if V16_SMOKE else 80
SEM_LGB_ESTIMATORS=8 if V16_SMOKE else 900
SEM_LGB_EARLY_STOP=2 if V16_SMOKE else 70
BOOTSTRAPS=300 if V16_SMOKE else 5000

PUBLIC_CAL045_LB=0.65639
PUBLIC_CONSENSUS3_LB=0.65856
PUBLIC_V13_1_LB=0.65754

# Qwen3-Embedding-8B native / Matryoshka dimensions.
EMBED_MODEL="Qwen/Qwen3-Embedding-8B"
EMBED_REVISION="1d8ad4c"
EMBED_DIM_FULL=4096
EMBED_DIM_SUP=1024
EMBED_DIM_KNN=256
RUNTIME_DIM_FULL=64 if V16_SMOKE else EMBED_DIM_FULL
RUNTIME_DIM_SUP=min(32 if V16_SMOKE else EMBED_DIM_SUP,RUNTIME_DIM_FULL)
RUNTIME_DIM_KNN=min(16 if V16_SMOKE else EMBED_DIM_KNN,RUNTIME_DIM_SUP)

CURRENT_MAX_TOKENS=192
RECENT_MAX_TOKENS=384
PROFILE_MAX_TOKENS=640
MODULE_MAX_TOKENS=384
SEMANTIC_BETAS=(0.25,0.50,0.75,1.00)
KNN_KS=(8,16) if V16_SMOKE else (32,96)
KNN_TEMP=0.08
SEM_RIDGE_ALPHA_BIN=18.0
SEM_RIDGE_ALPHA_TOP=30.0

# Promotion gates are incremental versus exact Consensus3, not versus older baselines.
MIN_ORD_INCREMENT=-0.01 if V16_SMOKE else 0.00010
MIN_AUDIT_INCREMENT=-0.01 if V16_SMOKE else 0.00030
MIN_CHRONO_INCREMENT=-0.01 if V16_SMOKE else 0.00010
MIN_ORD_FOLD_WINS=1 if V16_SMOKE else 3
MIN_AUDIT_FOLD_WINS=1 if V16_SMOKE else 4
MIN_AUDIT_CI_LOW=-1.0 if V16_SMOKE else 0.0
MAX_TEST_TOP5_CHANGE=1.0 if V16_SMOKE else 0.12
MAX_TEST_TOP1_CHANGE=1.0 if V16_SMOKE else 0.05

CACHE_ROOT=(Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('/mnt/data'))/'v16_cache'
CACHE_ROOT.mkdir(parents=True,exist_ok=True)

random.seed(SEED); np.random.seed(SEED)
print("V16_SMOKE",V16_SMOKE,"V16_SKIP_QWEN",V16_SKIP_QWEN)
print("Qwen model",EMBED_MODEL,"revision",EMBED_REVISION)
print("semantic dims",RUNTIME_DIM_FULL,RUNTIME_DIM_SUP,RUNTIME_DIM_KNN)
print("public Consensus3",PUBLIC_CONSENSUS3_LB,"V13.1",PUBLIC_V13_1_LB)

# ---- cell 3 (code) ----
def locate_data_dir():
    required = {
        "chat_history.csv", "modules_catalog.csv", "user_assessments.csv",
        "train_relevance.csv", "test.csv", "sample_submission.csv"
    }
    roots = [Path("/kaggle/input"), Path("/mnt/data/v16_data"), Path("/mnt/data/comp_v3"), Path("/mnt/data/comp_full"), Path(".")]
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

if V16_SMOKE:
    train=train.iloc[:400].copy(); test=test.iloc[:100].copy()
    sample=sample.set_index("user_id").loc[test.user_id].reset_index()
    _keep=set(train.user_id)|set(test.user_id)
    assessments=assessments[assessments.user_id.isin(_keep)].copy()
    chat=chat[chat.user_id.isin(_keep)].copy()
MODULES = list(train.columns[1:])
N_MODULES = len(MODULES)
N_TRAIN = len(train)
N_TEST = len(test)

assert MODULES == list(sample.columns[1:])
assert N_MODULES == 17
assert (N_TRAIN,N_TEST)==((400,100) if V16_SMOKE else (4000,1000))
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

# ---- cell 4 (code) ----
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

# ---- cell 5 (code) ----
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

# ---- cell 6 (code) ----
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

# ---- cell 7 (code) ----
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
assert USER_FEATURES.shape == (len(all_users), 134)
assert np.isfinite(USER_FEATURES).all()
print("USER_FEATURES:", USER_FEATURES.shape)
print("example augmented text:", texts[1][:500])

# ---- cell 8 (code) ----
# Exact public-validated V3.1 sparse backbone.
t0=time.time()
word_vec=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_df=.995,sublinear_tf=True,strip_accents="unicode",max_features=(600 if V16_SMOKE else 50000),dtype=np.float32)
char_vec=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=2,sublinear_tf=True,max_features=(900 if V16_SMOKE else 60000),dtype=np.float32)
X=hstack([word_vec.fit_transform(texts),char_vec.fit_transform(texts)],format="csr")
def sparse_ref(target):
    oo=np.zeros_like(y,np.float32); te=np.zeros((N_TEST,N_MODULES),np.float32)
    for tr,va in ref_folds:
        m=Ridge(alpha=RIDGE_ALPHA,solver="lsqr",tol=(2e-2 if V16_SMOKE else 2e-3),max_iter=(80 if V16_SMOKE else 1000)); m.fit(X[tr],target[tr]); oo[va]=m.predict(X[va]).astype(np.float32); te+=m.predict(X[N_TRAIN:]).astype(np.float32)/N_FOLDS
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

# ---- cell 10 (markdown) ----
# ## Immutable public champion
# The next cell reproduces the exact V12.2 Consensus3 calibration profile. Semantic experiments are allowed to modify only the *binary relevance meta signal* smoothly; the sparse backbone and this module-wise trust vector never change.

# ---- cell 11 (code) ----
CONSENSUS3_OVERRIDES={
    "M_002":0.44,"M_006":0.55,"M_007":0.60,"M_010":0.40,"M_011":0.55,
    "M_012":0.55,"M_015":0.44,"M_017":0.53,
}
def consensus_weights():
    w=np.full(N_MODULES,REFERENCE_W,np.float32)
    for mid,val in CONSENSUS3_OVERRIDES.items():
        w[MODULES.index(mid)]=float(val)
    return w
CONS_W=consensus_weights()
def apply_consensus(base,meta):
    return ((1.0-CONS_W)*base+CONS_W*meta).astype(np.float32)

consensus3_oof=apply_consensus(base_oof,meta_oof)
consensus3_test=apply_consensus(base_test,meta_test)
ref_fold_id=np.full(N_TRAIN,-1,int)
for f,(_,va) in enumerate(ref_folds): ref_fold_id[va]=f
assert (ref_fold_id>=0).all()
print("exact Consensus3 ordinary OOF",round(ndcg5(y,consensus3_oof),6),"public",PUBLIC_CONSENSUS3_LB)

# ---- cell 12 (markdown) ----
# ## Recency-specialized semantic views
# V16 deliberately avoids V1's long-history embedding. Older chat is compressed into organizer-derived topic/style summaries so stale intent cannot dominate the 8B representation. The three encoded learner views have distinct roles: **current intent**, **recent trajectory**, and **compact profile/readiness**.

# ---- cell 13 (code) ----
# Build compact semantic texts only from organizer-provided chat/assessment fields.
current_texts=[]; recent_texts=[]; profile_texts=[]
for u in all_users:
    ui=uid2idx[u]
    g=user_chat.get(u)
    if g is None or len(g)==0:
        learn=pd.DataFrame(columns=chat.columns)
    else:
        learn=g[~g["is_noise"]].copy()
        if len(learn)==0: learn=g.copy()
    rows=list(learn.itertuples(index=False))
    current=rows[-1].clean if rows else "No learning chat available."
    recent_rows=rows[-3:]
    labels=(['earlier','recent','current'][-len(recent_rows):] if recent_rows else [])
    recent=" ".join(f"[{lab}] {row.clean}" for lab,row in zip(labels,recent_rows)) or current

    skill_text="; ".join(f"{name} skill {float(A[ui,j]):.0f}/5" for j,name in enumerate(SKILL_NAMES))
    topic_scores=[]
    n=len(rows)
    for tp in TOPICS:
        count=0.0; rec=0.0
        for j,row in enumerate(rows):
            if tp in row.topics:
                count+=1.0; rec+=float(np.exp(-0.7*(n-1-j)))
        if count>0: topic_scores.append((rec,count,tp))
    topic_scores.sort(reverse=True)
    hist_summary=", ".join(f"{tp} ({int(cnt)} mentions)" for _,cnt,tp in topic_scores[:6]) or "none"
    style_names=[]
    for sty in STYLES:
        if len(learn) and float(learn['sty_'+sty].sum())>0: style_names.append(sty)
    style_summary=", ".join(style_names) or "none"

    current_texts.append(f"Current learner request: {current}")
    recent_texts.append(f"Recent learning conversation in chronological order: {recent}")
    profile_texts.append(
        f"Current request: {current}. Recent trajectory: {recent}. "
        f"Assessment: {skill_text}. Earlier topic summary: {hist_summary}. Learner style signals: {style_summary}."
    )

module_content_docs=[]; module_full_docs=[]; module_prereq_docs=[]
mods=modules.set_index('module_id').loc[MODULES].reset_index()
for row in mods.itertuples(index=False):
    module_content_docs.append(f"Course module: {row.module_name}. Learning content: {row.description_and_syllabus}.")
    module_full_docs.append(f"Course module: {row.module_name}. Learning content: {row.description_and_syllabus}. Prerequisite: {row.prerequisite_level}.")
    module_prereq_docs.append(f"Course module: {row.module_name}. Entry prerequisite or learner level: {row.prerequisite_level}.")

CURRENT_INSTRUCTION=(
    "Instruct: Retrieve the Intelligo course module that best matches the learner's CURRENT explicitly stated learning intent. "
    "Prefer the newest goal when it conflicts with older goals.\nQuery: "
)
RECENT_INSTRUCTION=(
    "Instruct: Retrieve the best NEXT Intelligo course module from the learner's recent learning trajectory. "
    "Give more importance to newer messages.\nQuery: "
)
PROFILE_INSTRUCTION=(
    "Instruct: Retrieve the best NEXT Intelligo course module considering current intent, recent trajectory, skill gaps, learner readiness, and course prerequisites.\nQuery: "
)

assert len(current_texts)==len(recent_texts)==len(profile_texts)==len(all_users)
assert len(module_full_docs)==N_MODULES
print(current_texts[1][:400])
print(profile_texts[1][:900])
print(module_full_docs[0])

# ---- cell 14 (markdown) ----
# ## Frozen Qwen3-Embedding-8B encoder
# Full mode uses the official 8B embedding checkpoint as a frozen feature extractor. It is loaded with `AutoModel`, `torch.float16`, `device_map="balanced"`, and explicit per-GPU memory caps so a 15.2GB FP16 checkpoint is sharded across both T4s rather than accidentally filling one GPU. No `.generate()` call exists anywhere in V16.

# ---- cell 15 (code) ----
def _normalize_rows(x):
    x=np.asarray(x,np.float32)
    return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8)

def _fake_embeddings(texts,dim,salt):
    out=np.empty((len(texts),dim),np.float32)
    for i,t in enumerate(texts):
        seed=int(hashlib.sha256((salt+'|'+str(t)).encode()).hexdigest()[:16],16)%(2**32)
        out[i]=np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return _normalize_rows(out)

def _text_hash(texts):
    h=hashlib.sha256()
    for t in texts:
        h.update(str(t).encode('utf-8')); h.update(b'\0')
    return h.hexdigest()[:16]

def _last_token_pool(last_hidden,attention_mask):
    # Qwen3-Embedding uses last-token pooling. Move mask to the output device when the model is sharded.
    mask=attention_mask.to(last_hidden.device)
    left_padding=bool((mask[:,-1].sum()==mask.shape[0]).item())
    if left_padding: return last_hidden[:,-1]
    seq=mask.sum(dim=1)-1
    return last_hidden[__import__('torch').arange(last_hidden.shape[0],device=last_hidden.device),seq]

qwen_model=None; qwen_tokenizer=None; qwen_input_device=None

def load_qwen():
    global qwen_model,qwen_tokenizer,qwen_input_device
    if qwen_model is not None: return
    if V16_SMOKE or V16_SKIP_QWEN: return
    import torch
    from transformers import AutoTokenizer,AutoModel
    if not torch.cuda.is_available() or torch.cuda.device_count()<2:
        raise RuntimeError("Full V16 requires Kaggle GPU T4 x2. Select Accelerator: GPU T4 x2.")
    for i in range(torch.cuda.device_count()):
        prop=torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {prop.name}, {prop.total_memory/1024**3:.1f} GiB")
    qwen_tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL,revision=EMBED_REVISION,padding_side='left')
    max_memory={0:'13GiB',1:'13GiB','cpu':'24GiB'}
    qwen_model=AutoModel.from_pretrained(
        EMBED_MODEL,revision=EMBED_REVISION,torch_dtype=torch.float16,
        device_map='balanced',max_memory=max_memory,low_cpu_mem_usage=True,
        attn_implementation='sdpa'
    )
    qwen_model.eval(); qwen_model.config.use_cache=False
    qwen_input_device=qwen_model.get_input_embeddings().weight.device
    print("hf_device_map:",qwen_model.hf_device_map)
    print("input device:",qwen_input_device)
    probe=qwen_tokenizer("probe",add_special_tokens=True)['input_ids']
    print("tokenizer eos appended:",bool(probe and probe[-1]==qwen_tokenizer.eos_token_id))

def encode_cached(name,texts,instruction,max_length,start_batch):
    dim=RUNTIME_DIM_FULL
    if V16_SMOKE or V16_SKIP_QWEN:
        return _fake_embeddings([instruction+t for t in texts],dim,'v16-'+name)
    import torch
    import torch.nn.functional as F
    load_qwen()
    key=f"{name}_{_text_hash(texts)}_{max_length}_{EMBED_REVISION}_{dim}"
    path=CACHE_ROOT/f"{key}.npy"
    if path.exists():
        arr=np.load(path)
        if arr.shape==(len(texts),dim) and np.isfinite(arr).all():
            print("loaded cache",path.name,arr.shape); return arr.astype(np.float32)
    prepared=[instruction+str(t) for t in texts]
    result=np.empty((len(prepared),dim),np.float32)
    bs=int(start_batch); pos=0
    while pos<len(prepared):
        take=min(bs,len(prepared)-pos)
        batch_text=prepared[pos:pos+take]
        try:
            toks=qwen_tokenizer(batch_text,padding=True,truncation=True,max_length=max_length,return_tensors='pt')
            toks={k:v.to(qwen_input_device) for k,v in toks.items()}
            with torch.inference_mode():
                out=qwen_model(**toks,use_cache=False,return_dict=True)
                emb=_last_token_pool(out.last_hidden_state,toks['attention_mask'])
                emb=F.normalize(emb,p=2,dim=1)
            result[pos:pos+take]=emb.detach().float().cpu().numpy()[:,:dim]
            pos+=take
            if pos%max(100,bs*20)==0 or pos==len(prepared): print(name,pos,"/",len(prepared),"batch",bs)
            del toks,out,emb
        except RuntimeError as e:
            if 'out of memory' not in str(e).lower() or bs<=1: raise
            bs=max(1,bs//2); print("OOM -> retry batch",bs)
            torch.cuda.empty_cache(); gc.collect()
    result=_normalize_rows(result)
    np.save(path,result.astype(np.float16))
    print("saved cache",path.name,result.shape)
    return result

# Three user views; modules are tiny and encoded in three document views.
current_emb=encode_cached('current',current_texts,CURRENT_INSTRUCTION,CURRENT_MAX_TOKENS,8)
recent_emb=encode_cached('recent',recent_texts,RECENT_INSTRUCTION,RECENT_MAX_TOKENS,6)
profile_emb=encode_cached('profile',profile_texts,PROFILE_INSTRUCTION,PROFILE_MAX_TOKENS,4)
content_emb=encode_cached('module_content',module_content_docs,'',MODULE_MAX_TOKENS,17)
full_emb=encode_cached('module_full',module_full_docs,'',MODULE_MAX_TOKENS,17)
prereq_emb=encode_cached('module_prereq',module_prereq_docs,'',MODULE_MAX_TOKENS,17)

if qwen_model is not None:
    del qwen_model,qwen_tokenizer
    qwen_model=qwen_tokenizer=None
    import torch
    torch.cuda.empty_cache(); gc.collect()

for name,arr in [('current',current_emb),('recent',recent_emb),('profile',profile_emb),('content',content_emb),('full',full_emb),('prereq',prereq_emb)]:
    assert arr.shape[1]==RUNTIME_DIM_FULL and np.isfinite(arr).all(),(name,arr.shape)
print("Qwen embedding shapes",current_emb.shape,profile_emb.shape,full_emb.shape)

# ---- cell 16 (code) ----
# Direct semantic geometry. Every matrix is label-independent.
def row_z(x):
    x=np.asarray(x,np.float32); mu=x.mean(1,keepdims=True); sd=x.std(1,keepdims=True)
    return ((x-mu)/np.maximum(sd,1e-6)).astype(np.float32)
def row_rank_score(x):
    order=np.argsort(-x,axis=1); ranks=np.empty_like(order)
    ranks[np.arange(len(x))[:,None],order]=np.arange(N_MODULES)[None,:]
    return (1.0-ranks.astype(np.float32)/(N_MODULES-1)).astype(np.float32)
def trunc_norm(x,d): return _normalize_rows(np.asarray(x[:,:d],np.float32))

SIM_MATRICES={
    'current_content':current_emb@content_emb.T,
    'current_full':current_emb@full_emb.T,
    'recent_content':recent_emb@content_emb.T,
    'recent_full':recent_emb@full_emb.T,
    'profile_content':profile_emb@content_emb.T,
    'profile_full':profile_emb@full_emb.T,
    'profile_prereq':profile_emb@prereq_emb.T,
}
SEM_PAIR_BLOCKS=[]; SEM_PAIR_NAMES=[]
for name,mat in SIM_MATRICES.items():
    mat=np.asarray(mat,np.float32)
    SEM_PAIR_BLOCKS += [mat[:,:,None],row_z(mat)[:,:,None],row_rank_score(mat)[:,:,None]]
    SEM_PAIR_NAMES += [name,name+'_z',name+'_rank']
SEM_PAIR_BASE=np.concatenate(SEM_PAIR_BLOCKS,axis=2).astype(np.float32)

SEM_VIEWS_SUP={
    'current':trunc_norm(current_emb,RUNTIME_DIM_SUP),
    'recent':trunc_norm(recent_emb,RUNTIME_DIM_SUP),
    'profile':trunc_norm(profile_emb,RUNTIME_DIM_SUP),
}
SEM_KNN_VEC=trunc_norm(0.40*current_emb+0.60*profile_emb,RUNTIME_DIM_KNN)

sem_zero=(.30*SIM_MATRICES['current_full']+.20*SIM_MATRICES['recent_full']+.40*SIM_MATRICES['profile_full']+.10*SIM_MATRICES['profile_prereq']).astype(np.float32)
print("semantic pair features",SEM_PAIR_BASE.shape,SEM_PAIR_NAMES)
print("Qwen zero-shot train NDCG@5",round(ndcg5(y,sem_zero[:N_TRAIN]),6))

# ---- cell 17 (markdown) ----
# ## Organizer-supervised semantic heads
# The 8B model stays frozen. Task adaptation happens only in lightweight organizer-trained components: multi-output Ridge heads on Matryoshka-truncated embeddings and semantic-neighbor priors. For any validation user, its supervised semantic features are produced only from that fold's training users.

# ---- cell 18 (code) ----
def make_local_folds(global_train,n_splits,seed):
    labs=strata[np.asarray(global_train)]
    return list(StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=int(seed)).split(np.zeros(len(global_train)),labs))

def semantic_heads_matched(global_train,global_out,folds_local):
    global_train=np.asarray(global_train,int); global_out=np.asarray(global_out,int)
    n=len(global_train); no=len(global_out)
    train_feat=np.zeros((n,N_MODULES,6),np.float32); out_feat=np.zeros((no,N_MODULES,6),np.float32)
    for vi,(vname,V) in enumerate(SEM_VIEWS_SUP.items()):
        out_bin=np.zeros((no,N_MODULES),np.float32); out_top=np.zeros_like(out_bin)
        for lf,(lt,lv) in enumerate(folds_local):
            gtr=global_train[lt]; gva=global_train[lv]
            mb=Ridge(alpha=SEM_RIDGE_ALPHA_BIN,solver='lsqr',tol=2e-3,max_iter=1200)
            mt=Ridge(alpha=SEM_RIDGE_ALPHA_TOP,solver='lsqr',tol=2e-3,max_iter=1200)
            mb.fit(V[gtr],Y_BIN[gtr]); mt.fit(V[gtr],Y_TOP[gtr])
            train_feat[lv,:,2*vi]=mb.predict(V[gva]).astype(np.float32)
            train_feat[lv,:,2*vi+1]=mt.predict(V[gva]).astype(np.float32)
            out_bin+=mb.predict(V[global_out]).astype(np.float32)/len(folds_local)
            out_top+=mt.predict(V[global_out]).astype(np.float32)/len(folds_local)
        out_feat[:,:,2*vi]=out_bin; out_feat[:,:,2*vi+1]=out_top
    return train_feat,out_feat

def _weighted_knn_multi(query,ref,ybin,ytop,ks=KNN_KS,temp=KNN_TEMP):
    # Compute cosine geometry once and reuse it for every predeclared k.
    sim=np.asarray(query@ref.T,np.float32)
    maxk=min(int(max(ks)),ref.shape[0])
    ix=np.argpartition(-sim,maxk-1,axis=1)[:,:maxk]
    s=np.take_along_axis(sim,ix,axis=1)
    order=np.argsort(-s,axis=1)
    ix=np.take_along_axis(ix,order,axis=1); s=np.take_along_axis(s,order,axis=1)
    out=np.zeros((len(query),N_MODULES,2*len(ks)),np.float32)
    for ki,k in enumerate(ks):
        kk=min(int(k),maxk)
        ss=s[:,:kk]-s[:,:kk].max(axis=1,keepdims=True)
        w=np.exp(np.clip(ss/max(temp,1e-4),-30,0)).astype(np.float32)
        w/=np.maximum(w.sum(axis=1,keepdims=True),1e-8)
        nbr=ix[:,:kk]
        out[:,:,2*ki]=np.einsum('nk,nkj->nj',w,ybin[nbr]).astype(np.float32)
        out[:,:,2*ki+1]=np.einsum('nk,nkj->nj',w,ytop[nbr]).astype(np.float32)
    return out

def semantic_knn_matched(global_train,global_out,folds_local):
    global_train=np.asarray(global_train,int); global_out=np.asarray(global_out,int)
    n=len(global_train); no=len(global_out); nf=2*len(KNN_KS)
    train_feat=np.zeros((n,N_MODULES,nf),np.float32); out_feat=np.zeros((no,N_MODULES,nf),np.float32)
    V=SEM_KNN_VEC
    for lf,(lt,lv) in enumerate(folds_local):
        gtr=global_train[lt]; gva=global_train[lv]
        train_feat[lv]=_weighted_knn_multi(V[gva],V[gtr],Y_BIN[gtr],Y_TOP[gtr])
        out_feat+=_weighted_knn_multi(V[global_out],V[gtr],Y_BIN[gtr],Y_TOP[gtr])/len(folds_local)
    return train_feat,out_feat

def make_sem_cal(seed,n_estimators=None):
    return lgb.LGBMClassifier(
        n_estimators=int(SEM_LGB_ESTIMATORS if n_estimators is None else n_estimators),
        learning_rate=.025,num_leaves=12,max_depth=4,min_child_samples=65,
        reg_alpha=1.0,reg_lambda=8.0,colsample_bytree=.78,
        random_state=int(seed),n_jobs=4,verbosity=-1
    )

def build_sem_X(global_idx,base_scores,head_feat,knn_feat,m):
    global_idx=np.asarray(global_idx,int)
    return np.column_stack([
        base_scores[:,m],USER_FEATURES[global_idx],SEM_PAIR_BASE[global_idx,m,:],head_feat[:,m,:],knn_feat[:,m,:]
    ]).astype(np.float32)

def fit_sem_meta_oof(global_train,global_out,folds_local,base_train,base_out,head_train,head_out,knn_train,knn_out):
    global_train=np.asarray(global_train,int); global_out=np.asarray(global_out,int)
    oo=np.zeros((len(global_train),N_MODULES),np.float32); out=np.zeros((len(global_out),N_MODULES),np.float32)
    for m,mid in enumerate(MODULES):
        Xt=build_sem_X(global_train,base_train,head_train,knn_train,m)
        Xo=build_sem_X(global_out,base_out,head_out,knn_out,m)
        yy=Y_BIN[global_train,m].astype(np.int32); its=[]
        for f,(lt,lv) in enumerate(folds_local):
            clf=make_sem_cal(SEED+10000+131*m+f)
            clf.fit(Xt[lt],yy[lt],eval_set=[(Xt[lv],yy[lv])],callbacks=[lgb.early_stopping(SEM_LGB_EARLY_STOP,verbose=False)])
            it=int(clf.best_iteration_ or SEM_LGB_ESTIMATORS); its.append(it)
            oo[lv,m]=clf.predict_proba(Xt[lv],num_iteration=it)[:,1].astype(np.float32)
            # Keep deployment geometry matched to OOF: test/out is the average of the
            # exact fold models, never a separately refit full-data second-stage model.
            out[:,m]+=clf.predict_proba(Xo,num_iteration=it)[:,1].astype(np.float32)/len(folds_local)
    return oo,out

# ---- cell 19 (code) ----
# Ordinary reference-fold semantic pipeline. This is comparable with historical OOF,
# but the new matched nested audit below is the promotion authority.
TRAIN_GLOBAL=np.arange(N_TRAIN); TEST_GLOBAL=np.arange(N_TRAIN,N_TRAIN+N_TEST)
ref_local=ref_folds
sem_head_oof,sem_head_test=semantic_heads_matched(TRAIN_GLOBAL,TEST_GLOBAL,ref_local)
sem_knn_oof,sem_knn_test=semantic_knn_matched(TRAIN_GLOBAL,TEST_GLOBAL,ref_local)
sem_meta_oof,sem_meta_test=fit_sem_meta_oof(
    TRAIN_GLOBAL,TEST_GLOBAL,ref_local,base_oof,base_test,sem_head_oof,sem_head_test,sem_knn_oof,sem_knn_test
)

print("semantic head shapes",sem_head_oof.shape,sem_knn_oof.shape)
print("old meta OOF",round(ndcg5(y,meta_oof),6),"semantic meta OOF",round(ndcg5(y,sem_meta_oof),6))
print("semantic Ridge-bin mean-view NDCG",round(ndcg5(y,sem_head_oof[:,:,[0,2,4]].mean(2)),6))
print("semantic KNN-bin mean-k NDCG",round(ndcg5(y,sem_knn_oof[:,:,[0,2]].mean(2)),6))

# ---- cell 20 (markdown) ----
# ## New fold-matched nested audit
# This is the key V16 safeguard. The outer folds use a **new seed (`20260824`)**, not the repeatedly inspected V12 folds. Inside each outer-training set, base predictions, semantic Ridge heads, and KNN priors are inner-OOF. Outer-validation first-stage predictions are the average of those same inner models, so meta-feature geometry is matched rather than switching from 80%-trained OOF models to a full-fit representation.

# ---- cell 21 (code) ----
def matched_base(global_train,global_out,inner):
    global_train=np.asarray(global_train,int); global_out=np.asarray(global_out,int)
    raw=np.zeros((len(global_train),N_MODULES),np.float32); top=np.zeros_like(raw)
    out_raw=np.zeros((len(global_out),N_MODULES),np.float32); out_top=np.zeros_like(out_raw)
    for lt,lv in inner:
        gtr=global_train[lt]; gva=global_train[lv]
        mr=Ridge(alpha=RIDGE_ALPHA,solver='lsqr',tol=(2e-2 if V16_SMOKE else 2e-3),max_iter=(80 if V16_SMOKE else 1000))
        mt=Ridge(alpha=RIDGE_ALPHA,solver='lsqr',tol=(2e-2 if V16_SMOKE else 2e-3),max_iter=(80 if V16_SMOKE else 1000))
        mr.fit(X[gtr],y[gtr]); mt.fit(X[gtr],Y_TOP[gtr])
        raw[lv]=mr.predict(X[gva]).astype(np.float32); top[lv]=mt.predict(X[gva]).astype(np.float32)
        out_raw+=mr.predict(X[global_out]).astype(np.float32)/len(inner)
        out_top+=mt.predict(X[global_out]).astype(np.float32)/len(inner)
    return (.9*raw+.1*top).astype(np.float32),(.9*out_raw+.1*out_top).astype(np.float32)

def matched_old_meta(global_train,global_out,inner,btr,bout):
    global_train=np.asarray(global_train,int); global_out=np.asarray(global_out,int)
    ao=np.zeros((len(global_train),N_MODULES),np.float32); av=np.zeros((len(global_out),N_MODULES),np.float32)
    for m,mid in enumerate(MODULES):
        Xt=np.column_stack([btr[:,m],USER_FEATURES[global_train]]).astype(np.float32)
        Xv=np.column_stack([bout[:,m],USER_FEATURES[global_out]]).astype(np.float32)
        yy=Y_BIN[global_train,m].astype(np.int32); its=[]
        for f,(lt,lv) in enumerate(inner):
            clf=make_cal(SEED+30000+31*m+f)
            clf.fit(Xt[lt],yy[lt],eval_set=[(Xt[lv],yy[lv])],callbacks=[lgb.early_stopping(LGB_EARLY_STOP,verbose=False)])
            it=int(clf.best_iteration_ or LGB_ESTIMATORS); its.append(it)
            ao[lv,m]=clf.predict_proba(Xt[lv],num_iteration=it)[:,1].astype(np.float32)
        final_iters=max(20,int(np.median(its)))
        full=make_cal(SEED+40000+31*m,final_iters); full.fit(Xt,yy)
        av[:,m]=full.predict_proba(Xv)[:,1].astype(np.float32)
    return ao,av

def matched_sem_meta(global_train,global_out,inner,btr,bout,htr,hout,ktr,kout):
    global_train=np.asarray(global_train,int); global_out=np.asarray(global_out,int)
    ao=np.zeros((len(global_train),N_MODULES),np.float32); av=np.zeros((len(global_out),N_MODULES),np.float32)
    for m,mid in enumerate(MODULES):
        Xt=build_sem_X(global_train,btr,htr,ktr,m); Xv=build_sem_X(global_out,bout,hout,kout,m)
        yy=Y_BIN[global_train,m].astype(np.int32); its=[]
        for f,(lt,lv) in enumerate(inner):
            clf=make_sem_cal(SEED+50000+131*m+f)
            clf.fit(Xt[lt],yy[lt],eval_set=[(Xt[lv],yy[lv])],callbacks=[lgb.early_stopping(SEM_LGB_EARLY_STOP,verbose=False)])
            it=int(clf.best_iteration_ or SEM_LGB_ESTIMATORS); its.append(it)
            ao[lv,m]=clf.predict_proba(Xt[lv],num_iteration=it)[:,1].astype(np.float32)
        final_iters=max(20,int(np.median(its)))
        full=make_sem_cal(SEED+60000+131*m,final_iters); full.fit(Xt,yy)
        av[:,m]=full.predict_proba(Xv)[:,1].astype(np.float32)
    return ao,av

def eval_outer_block(global_train,global_out,seed):
    inner=make_local_folds(global_train,N_INNER,seed)
    btr,bout=matched_base(global_train,global_out,inner)
    htr,hout=semantic_heads_matched(global_train,global_out,inner)
    ktr,kout=semantic_knn_matched(global_train,global_out,inner)
    _,old_out=matched_old_meta(global_train,global_out,inner,btr,bout)
    _,sem_out=matched_sem_meta(global_train,global_out,inner,btr,bout,htr,hout,ktr,kout)
    return bout,old_out,sem_out

# New outer folds unseen during V12–V15 development.
if V16_SMOKE:
    # Execute one small nested probe to validate the difficult code path, then reuse ordinary arrays
    # so smoke mode remains a minutes-level structural test rather than a full experiment.
    probe_tr=np.arange(200); probe_va=np.arange(200,300)
    pb,po,ps=eval_outer_block(probe_tr,probe_va,AUDIT_SEED)
    assert pb.shape==po.shape==ps.shape==(100,N_MODULES)
    print("matched nested smoke probe: PASS")
    audit_base=base_oof.copy(); audit_old=meta_oof.copy(); audit_sem=sem_meta_oof.copy(); audit_fold_id=ref_fold_id.copy()
else:
    audit_skf=StratifiedKFold(n_splits=N_FOLDS,shuffle=True,random_state=AUDIT_SEED)
    audit_folds=list(audit_skf.split(np.zeros(N_TRAIN),strata))
    audit_base=np.zeros_like(y); audit_old=np.zeros_like(y); audit_sem=np.zeros_like(y); audit_fold_id=np.full(N_TRAIN,-1,int)
    for f,(tr,va) in enumerate(audit_folds):
        print(f"matched audit outer {f+1}/{N_FOLDS}")
        b,o,s=eval_outer_block(tr,va,AUDIT_SEED+100*f)
        audit_base[va]=b; audit_old[va]=o; audit_sem[va]=s; audit_fold_id[va]=f
        gc.collect()
    assert (audit_fold_id>=0).all()
audit_anchor=apply_consensus(audit_base,audit_old)
print("matched audit Consensus3",round(ndcg5(y,audit_anchor),6))

# ---- cell 22 (markdown) ----
# ## Chronological future-block stress test
# The test IDs follow the train IDs, so V16 also trains only on U0001–U3000 and evaluates U3001–U4000 with the same fold-matched inner representation. This is not used to tune a flexible model; it is only a hard promotion gate for the four frozen semantic strengths.

# ---- cell 23 (code) ----
if V16_SMOKE:
    cut=N_TRAIN//2; chrono_tr=np.arange(cut); chrono_va=np.arange(cut,N_TRAIN)
    chrono_base=base_oof[chrono_va].copy(); chrono_old=meta_oof[chrono_va].copy(); chrono_sem=sem_meta_oof[chrono_va].copy()
else:
    cut=3000; chrono_tr=np.arange(cut); chrono_va=np.arange(cut,N_TRAIN)
    chrono_base,chrono_old,chrono_sem=eval_outer_block(chrono_tr,chrono_va,AUDIT_SEED+999)
chrono_anchor=apply_consensus(chrono_base,chrono_old)
print("chronological anchor",round(ndcg5(y[chrono_va],chrono_anchor),6),"n",len(chrono_va))

# ---- cell 24 (code) ----
# Fixed smooth fusion only. No random search and no per-user policy.
def candidate_from(base,old_meta,new_meta,beta):
    mixed=((1.0-beta)*old_meta+beta*new_meta).astype(np.float32)
    return apply_consensus(base,mixed)

def fold_means(delta,fold_id):
    return np.asarray([np.asarray(delta)[fold_id==f].mean() for f in range(N_FOLDS)],float)
def paired_boot(delta,seed=20260824):
    delta=np.asarray(delta,float); rng=np.random.default_rng(seed); n=len(delta); vals=np.empty(BOOTSTRAPS,float)
    for i in range(BOOTSTRAPS): vals[i]=delta[rng.integers(0,n,n)].mean()
    return tuple(float(v) for v in np.quantile(vals,[.025,.975]))
def top5_set_change_rate(a,b):
    ta,tb=top5_indices(a),top5_indices(b)
    return float(np.mean([set(x)!=set(z) for x,z in zip(ta,tb)]))
def top5_order_change_rate(a,b): return float(np.mean(np.any(top5_indices(a)!=top5_indices(b),axis=1)))

audit_anchor_u=per_user_ndcg(y,audit_anchor)
ord_anchor_u=per_user_ndcg(y,consensus3_oof)
chrono_anchor_u=per_user_ndcg(y[chrono_va],chrono_anchor)
rows=[]; candidate_test={}; candidate_oof={}; candidate_audit={}
for beta in SEMANTIC_BETAS:
    name=f"sem{int(beta*100):03d}"
    po=candidate_from(base_oof,meta_oof,sem_meta_oof,beta)
    pa=candidate_from(audit_base,audit_old,audit_sem,beta)
    pc=candidate_from(chrono_base,chrono_old,chrono_sem,beta)
    pt=candidate_from(base_test,meta_test,sem_meta_test,beta)
    candidate_oof[name]=po; candidate_audit[name]=pa; candidate_test[name]=pt
    du_o=per_user_ndcg(y,po)-ord_anchor_u
    du_a=per_user_ndcg(y,pa)-audit_anchor_u
    du_c=per_user_ndcg(y[chrono_va],pc)-chrono_anchor_u
    ord_f=fold_means(du_o,ref_fold_id); aud_f=fold_means(du_a,audit_fold_id)
    ci=paired_boot(du_a,seed=AUDIT_SEED+int(beta*1000))
    top5=top5_set_change_rate(pt,consensus3_test); order=top5_order_change_rate(pt,consensus3_test); top1=top1_change_rate(pt,consensus3_test)
    passed=bool(
        du_o.mean()>=MIN_ORD_INCREMENT and du_a.mean()>=MIN_AUDIT_INCREMENT and du_c.mean()>=MIN_CHRONO_INCREMENT and
        int((ord_f>0).sum())>=MIN_ORD_FOLD_WINS and int((aud_f>0).sum())>=MIN_AUDIT_FOLD_WINS and
        ci[0]>=MIN_AUDIT_CI_LOW and top5<=MAX_TEST_TOP5_CHANGE and top1<=MAX_TEST_TOP1_CHANGE
    )
    rows.append({
        'candidate':name,'beta':beta,'ordinary_score':ndcg5(y,po),'ordinary_increment':float(du_o.mean()),
        'ordinary_fold_wins':int((ord_f>0).sum()),'ordinary_fold_deltas':json.dumps(ord_f.tolist()),
        'audit_score':ndcg5(y,pa),'audit_increment':float(du_a.mean()),'audit_fold_wins':int((aud_f>0).sum()),
        'audit_fold_deltas':json.dumps(aud_f.tolist()),'audit_ci_low':ci[0],'audit_ci_high':ci[1],
        'chrono_score':ndcg5(y[chrono_va],pc),'chrono_increment':float(du_c.mean()),
        'test_top5_set_change':top5,'test_top5_order_change':order,'test_top1_change':top1,'promotion_gate':passed,
    })

diagnostics=pd.DataFrame(rows)
display(diagnostics.round(6))

# Conservative selection: first (smallest) beta that passes every predeclared gate.
selected_name=None
for beta in SEMANTIC_BETAS:
    name=f"sem{int(beta*100):03d}"
    if bool(diagnostics.loc[diagnostics.candidate==name,'promotion_gate'].iloc[0]):
        selected_name=name; break
promotion_gate=selected_name is not None
if promotion_gate:
    primary_test=candidate_test[selected_name].copy(); primary_label=selected_name
else:
    primary_test=consensus3_test.copy(); primary_label='consensus3_fallback'

print("\n================ V16 DECISION ================")
print("ordinary Consensus3",round(ndcg5(y,consensus3_oof),6))
print("matched audit anchor",round(ndcg5(y,audit_anchor),6))
print("chronological anchor",round(ndcg5(y[chrono_va],chrono_anchor),6))
print("promotion_gate",promotion_gate,"primary",primary_label)
print("If no semantic candidate passes, do NOT spend a Kaggle slot: primary is exact public Consensus3.")
print("================================================")

# ---- cell 25 (code) ----
def save_submission(scores,name):
    out=sample.copy(); out.loc[:,MODULES]=scores
    root=Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('/mnt/data')
    path=root/name; out.to_csv(path,index=False)
    chk=pd.read_csv(path)
    assert chk.shape==sample.shape and chk.user_id.tolist()==sample.user_id.tolist() and chk.user_id.is_unique
    assert chk[MODULES].notna().all().all() and np.isfinite(chk[MODULES].to_numpy()).all()
    print("saved",path); return path

root=Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('/mnt/data')
paths={}
paths['consensus3']=save_submission(consensus3_test,'submission_v16_consensus3_reference.csv')
for beta in SEMANTIC_BETAS:
    name=f"sem{int(beta*100):03d}"
    paths[name]=save_submission(candidate_test[name],f"submission_v16_{name}.csv")
paths['primary']=save_submission(primary_test,'submission_v16_primary.csv')
diagnostics.to_csv(root/'v16_diagnostics.csv',index=False)

np.savez_compressed(
    root/'v16_oof_artifacts.npz',y=y,base_oof=base_oof,meta_oof=meta_oof,consensus3_oof=consensus3_oof,
    sem_meta_oof=sem_meta_oof,audit_base=audit_base,audit_old=audit_old,audit_sem=audit_sem,audit_anchor=audit_anchor,
    base_test=base_test,meta_test=meta_test,sem_meta_test=sem_meta_test,consensus3_test=consensus3_test,
    ref_fold_id=ref_fold_id,audit_fold_id=audit_fold_id
)
summary={
    'model':EMBED_MODEL,'model_revision':EMBED_REVISION,'full_embedding_dim':EMBED_DIM_FULL,
    'supervised_matryoshka_dim':EMBED_DIM_SUP,'knn_matryoshka_dim':EMBED_DIM_KNN,
    'public_consensus3_lb':PUBLIC_CONSENSUS3_LB,'ordinary_consensus3':float(ndcg5(y,consensus3_oof)),
    'matched_audit_consensus3':float(ndcg5(y,audit_anchor)),'chronological_anchor':float(ndcg5(y[chrono_va],chrono_anchor)),
    'diagnostics':diagnostics.to_dict(orient='records'),'promotion_gate':bool(promotion_gate),
    'primary_label':primary_label,'primary_file':paths['primary'].name,
    'compliance':{
        'task_specific_data':'organizer files only','pretrained_usage':'frozen embedding feature extractor only',
        'generation_used':False,'direct_test_label_request':False,'open_source_pretrained':True
    }
}
with open(root/'v16_summary.json','w') as f: json.dump(summary,f,indent=2)
print("saved diagnostics/artifacts/summary")

# ---- cell 26 (markdown) ----
# ## Hard invariants
# These assertions intentionally make the notebook fail rather than silently produce a non-reproducible or non-compliant candidate. In full mode the historical ordinary Consensus3 OOF score is also checked within a tight tolerance to confirm that the public anchor was reproduced before semantic fusion.

# ---- cell 27 (code) ----
assert len(MODULES)==17 and y.shape==(N_TRAIN,17) and N_TEST==(100 if V16_SMOKE else 1000)
assert CONSENSUS3_OVERRIDES=={
    "M_002":0.44,"M_006":0.55,"M_007":0.60,"M_010":0.40,"M_011":0.55,
    "M_012":0.55,"M_015":0.44,"M_017":0.53,
}
assert np.allclose(consensus3_oof,apply_consensus(base_oof,meta_oof),atol=0,rtol=0)
assert np.allclose(consensus3_test,apply_consensus(base_test,meta_test),atol=0,rtol=0)
for arr in [base_oof,meta_oof,consensus3_oof,sem_meta_oof,base_test,meta_test,sem_meta_test,consensus3_test,audit_base,audit_old,audit_sem,audit_anchor]:
    assert np.isfinite(arr).all()
assert SEM_PAIR_BASE.shape[:2]==(len(all_users),17)
assert sem_head_oof.shape[:2]==sem_knn_oof.shape[:2]==(N_TRAIN,17)
assert diagnostics.candidate.tolist()==[f"sem{int(b*100):03d}" for b in SEMANTIC_BETAS]
if not V16_SMOKE:
    assert abs(ndcg5(y,consensus3_oof)-0.6643682395561207)<0.0012,ndcg5(y,consensus3_oof)
if promotion_gate:
    assert primary_label in candidate_test and np.allclose(primary_test,candidate_test[primary_label],atol=0,rtol=0)
else:
    assert primary_label=='consensus3_fallback' and np.allclose(primary_test,consensus3_test,atol=0,rtol=0)
print("V16 hard invariants: PASS")
