# Converted from notebooks/v23.ipynb: markdown cells become comments, %/! magics neutralized.

# ---- cell 0 (markdown) ----
# # MineToday V23 — Consensus3 Local-Action Utility Corrector + F2LLM-v2-4B
#
# **Goal:** preserve the public-best **0.65912** ranking as an immutable anchor, then add only a tiny number of novel local score swaps when two independent action models agree that the swap has positive expected **NDCG@5 utility**.
#
# This notebook is deliberately **not** another global reranker. It implements the post-V22 diagnosis:
#
# 1. reproduce the exact V3.1 sparse backbone and V12.2 `Consensus3` calibration;
# 2. reconstruct the current PB by applying only the two already public-validated ordering corrections:
#    - `U_4047`: `M_007 (EDA)` above `M_009 (ML)`;
#    - `U_4953`: `M_005 (Version Control)` above `M_016 (MLOps)`;
# 3. generate only low-margin local Top-7 actions, lock all **novel Top-1** changes, and allow at most one novel action per user;
# 4. train **direct swap-utility** models rather than module-relevance models;
# 5. require agreement between a structured branch and a semantic branch using the open-source `codefuse-ai/F2LLM-v2-4B` embedding model;
# 6. discover pair-family priors only inside each outer-training fold, so candidate-family selection never sees its outer-validation labels;
# 7. export three capped attacks: `<=3`, `<=6`, and strict semantic-confirmed `<=8` novel users, plus an exact PB reference and a fail-closed primary.
#
# ### Kaggle settings
#
# - Accelerator: **GPU T4 x2**
# - Internet: **ON** on first run so the open-source F2LLM checkpoint can be downloaded
# - No external task dataset is used. F2LLM is used only as a frozen feature extractor.
# - This notebook never calls `.generate()` and never asks an LLM to infer test labels directly.
#
# ### Research basis
#
# - F2LLM-v2: *Inclusive, Performant, and Efficient Embeddings for a Multilingual World* (arXiv:2603.19223). The 4B checkpoint is Apache-2.0 and supports query/document embedding prompts.
# - NV-Embed / ICLR 2025: the useful transferable idea here is **hard-negative emphasis**, not replacing the competition ranker with a new embedding ranker.
# - Late-interaction motivation from ColBERT-style retrieval: current request, trajectory, and readiness are kept as separate semantic views instead of one long-history vector.
#
# The immutable decision rule is: **when evidence is weak, keep the 0.65912 PB ranking unchanged.**

# ---- cell 1 (code) ----
# Runtime + deterministic configuration
import os, sys, re, json, ast, html, hashlib, random, warnings, gc, time, math, subprocess
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ["PYTHONHASHSEED"] = "42"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("HF_HOME", "/kaggle/working/hf_cache")

V23_SKIP_INSTALL = bool(int(os.getenv("V23_SKIP_INSTALL", "0")))
if not V23_SKIP_INSTALL:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "lightgbm>=4.5,<5",
        "sentence-transformers>=5.1,<6",
        "transformers>=4.56,<5",
        "accelerate>=1.10,<2",
        "huggingface-hub>=0.34",
    ])

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
import lightgbm as lgb

warnings.filterwarnings("ignore")

SEED = 42
V23_SMOKE = bool(int(os.getenv("V23_SMOKE", "0")))
N_FOLDS = 2 if V23_SMOKE else 5
RIDGE_ALPHA = 3.0
REFERENCE_W = 0.45
PUBLIC_CAL045_LB = 0.65639  # historical public reference used only for printed diagnostics
LGB_ESTIMATORS = 40 if V23_SMOKE else 1200
LGB_EARLY_STOP = 8 if V23_SMOKE else 80

# V23 action contract
ACTION_MARGIN_QUANTILE = 0.10       # winners U4047/U4953 were both extreme near-ties
ACTION_TOPK = 7
ACTION_MAX_RANK_DISTANCE = 2
LOCK_NOVEL_TOP1 = True
PAIR_MIN_SUPPORT = 4 if V23_SMOKE else 8
PAIR_MIN_POS_RATE = 0.18
PAIR_MIN_MEAN_UTILITY = 0.0
HARD_NEGATIVE_MULT = 1.75
ACTION_MODEL_TREES = 120 if V23_SMOKE else 520
ACTION_MODEL_LR = 0.025
POLICY_RATES = {
    "conservative": 0.003,  # ~3 / 1000 test users
    "balanced": 0.006,      # ~6 / 1000
    "confirmed": 0.008,     # <=8 / 1000, but stricter pair+semantic gate
}

# Fail-closed OOF promotion gate. Attack CSVs are always exported for controlled probing;
# submission_v23_primary.csv uses a novel policy only when this passes.
MIN_POLICY_OOF_DELTA = -1.0 if V23_SMOKE else 0.00010
MIN_POLICY_FOLD_WINS = 1 if V23_SMOKE else 4
MAX_POLICY_WORST_FOLD = -1.0 if V23_SMOKE else -0.00030
MIN_SELECTED_PRECISION = 0.0 if V23_SMOKE else 0.45

# Frozen semantic model: open-source pretrained feature extractor only.
F2LLM_MODEL = os.getenv("V23_F2LLM_MODEL", "codefuse-ai/F2LLM-v2-4B")
F2LLM_BATCH = int(os.getenv("V23_F2LLM_BATCH", "4" if V23_SMOKE else "8"))
F2LLM_MAX_SEQ = int(os.getenv("V23_F2LLM_MAX_SEQ", "384"))
F2LLM_DUAL_GPU = bool(int(os.getenv("V23_F2LLM_DUAL_GPU", "1")))

random.seed(SEED)
np.random.seed(SEED)
print("numpy", np.__version__, "pandas", pd.__version__, "lightgbm", lgb.__version__)
print("V23_SMOKE", V23_SMOKE, "F2LLM_MODEL", F2LLM_MODEL)

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
BOOTSTRAPS = 200 if V23_SMOKE else 2000
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
# ## Exact public-validated anchor
#
# The next two cells are intentionally inherited from the public-validated V3.1/V7/V12.2 family. V23 does **not** retune the sparse representation or binary relevance calibration.

# ---- cell 8 (code) ----
# Exact public-validated V3.1 sparse backbone.
t0=time.time()
word_vec=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_df=.995,sublinear_tf=True,strip_accents="unicode",max_features=(2000 if V23_SMOKE else 50000),dtype=np.float32)
char_vec=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=2,sublinear_tf=True,max_features=(3000 if V23_SMOKE else 60000),dtype=np.float32)
X=hstack([word_vec.fit_transform(texts),char_vec.fit_transform(texts)],format="csr")
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
# Exact V12.2 Consensus3 profile. No weight search in V23.
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
# ## Immutable current-PB reference: 0.65912 ranking
#
# The public-best ranking from the previous day was **Consensus3 everywhere except two deterministic local ordering corrections**. V23 rebuilds that ranking first and treats it as immutable. Any novel action is applied *on top of* this PB reference, never instead of it.

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

# ---- cell 13 (markdown) ----
# ## Organizer-derived prerequisite graph and learner views
#
# The prerequisite graph below is **not external knowledge**. It is a normalized representation of the organizer's `modules_catalog.csv` prerequisite text. It is used only as a feature/gate, never as a universal rule, because the OOF audit showed that broad prerequisite-first swapping is usually harmful.

# ---- cell 14 (code) ----
# Organizer-derived prerequisite edges from modules_catalog.csv.
PREREQ = {
    "M_004": ["M_002"],
    "M_007": ["M_002", "M_006"],
    "M_008": ["M_003", "M_001"],
    "M_009": ["M_007", "M_006"],
    "M_010": ["M_009"],
    "M_011": ["M_009"],
    "M_012": ["M_011"],
    "M_016": ["M_009", "M_002", "M_005"],
}

catalog = modules.set_index("module_id").loc[MODULES]
# Drift guards against silently using a different organizer catalog.
assert "Butuh EDA & Stat" in str(catalog.loc["M_009", "prerequisite_level"])
assert "Version Control" in str(catalog.loc["M_016", "prerequisite_level"])
assert "Butuh NLP" in str(catalog.loc["M_012", "prerequisite_level"])

PREREQ_MATRIX = np.zeros((N_MODULES, N_MODULES), np.float32)  # [prereq, destination]
for dest, reqs in PREREQ.items():
    for req in reqs:
        PREREQ_MATRIX[MID_TO_COL[req], MID_TO_COL[dest]] = 1.0

TOPIC_TO_MODULE = {
    "excel":"M_001", "python":"M_002", "sql":"M_003", "webscrape":"M_004",
    "git":"M_005", "statistics":"M_006", "eda":"M_007", "dashboard":"M_008",
    "ml":"M_009", "ml_eval":"M_009", "cv":"M_010", "nlp":"M_011",
    "genai":"M_012", "prompt":"M_013", "automation":"M_014", "nocode":"M_015",
    "mlops":"M_016", "career":"M_017", "business":"M_017",
}
PREREQ_ASK_RX = re.compile(
    r"\b(?:prasyarat|syarat|prereq|prasyaratnya|requirement|requirements|basic apa|butuh apa|"
    r"harus (?:paham|kuasai|belajar)|sebelum (?:belajar|ambil|masuk))\b", re.I
)

LATEST_MENTION = np.zeros((len(all_users), N_MODULES), np.float32)
RECENT_MENTION = np.zeros_like(LATEST_MENTION)
ANY_MENTION = np.zeros_like(LATEST_MENTION)
PREREQ_ASK = np.zeros(len(all_users), np.float32)

latest_view, recent_view, profile_view = [], [], []

for u in all_users:
    ui = uid2idx[u]
    g = user_chat.get(u)
    if g is None or len(g) == 0:
        learn = pd.DataFrame(columns=chat.columns)
    else:
        learn = g[~g["is_noise"]].copy()
        if len(learn) == 0:
            learn = g.copy()

    rows = list(learn.itertuples(index=False))
    latest_raw = str(rows[-1].user_chat_text) if rows else "Tidak ada chat pembelajaran."
    recent_raw = [str(r.user_chat_text) for r in rows[-3:]] if rows else ["Tidak ada chat pembelajaran."]

    if rows:
        PREREQ_ASK[ui] = float(bool(PREREQ_ASK_RX.search(latest_raw)))
        for j, row in enumerate(rows):
            for tp in row.topics:
                mid = TOPIC_TO_MODULE.get(tp)
                if mid is None:
                    continue
                mi = MID_TO_COL[mid]
                ANY_MENTION[ui, mi] = 1.0
                if j >= max(0, len(rows)-3):
                    RECENT_MENTION[ui, mi] = 1.0
                if j == len(rows)-1:
                    LATEST_MENTION[ui, mi] = 1.0

    skill_text = ", ".join(
        f"{SKILL_NAMES[j]}={A[ui,j]:.0f}/5" + (" (invalid->imputed)" if A_INVALID[ui,j] else "")
        for j in range(len(SKILL_NAMES))
    )
    latest_view.append("Permintaan terbaru learner: " + latest_raw)
    recent_view.append(
        "Trajektori chat terbaru, urut lama ke baru:\n" +
        "\n".join(f"- {t}" for t in recent_raw)
    )
    profile_view.append(
        "Konteks learner. Chat terbaru: " + latest_raw +
        "\nSelf-assessment saat ini: " + skill_text
    )

latest_view = np.asarray(latest_view, dtype=object)
recent_view = np.asarray(recent_view, dtype=object)
profile_view = np.asarray(profile_view, dtype=object)

module_docs = []
for mid in MODULES:
    r = catalog.loc[mid]
    module_docs.append(
        f"Course {mid}: {r['module_name']}. "
        f"Isi: {r['description_and_syllabus']} "
        f"Level/prasyarat organizer: {r['prerequisite_level']}."
    )

ACTION_PAIRS = []
action_docs = []
for lo in range(N_MODULES):
    for hi in range(N_MODULES):
        if lo == hi:
            continue
        lo_mid, hi_mid = MODULES[lo], MODULES[hi]
        relation = (
            f"Menurut katalog organizer, {lo_mid} adalah prasyarat langsung untuk {hi_mid}."
            if PREREQ_MATRIX[lo, hi] > 0 else
            "Tidak ada edge prasyarat langsung yang eksplisit antara dua course ini di katalog."
        )
        ACTION_PAIRS.append((lo, hi))
        action_docs.append(
            f"Rekomendasi next-step: prioritaskan {lo_mid} ({catalog.loc[lo_mid,'module_name']}) "
            f"sebelum {hi_mid} ({catalog.loc[hi_mid,'module_name']}). {relation} "
            f"Course yang diprioritaskan: {catalog.loc[lo_mid,'description_and_syllabus']} "
            f"Course yang ditunda: {catalog.loc[hi_mid,'description_and_syllabus']} "
            f"Prasyarat course yang ditunda: {catalog.loc[hi_mid,'prerequisite_level']}."
        )
ACTION_PAIR_TO_IDX = {p:i for i,p in enumerate(ACTION_PAIRS)}
print("user semantic views", len(latest_view), "action docs", len(action_docs))

# ---- cell 15 (markdown) ----
# ## Frozen F2LLM-v2-4B semantic evidence
#
# F2LLM is **not fine-tuned on test data** and never outputs labels. It only encodes organizer-derived text. Three learner views remain separate:
#
# - latest request;
# - last-three-message trajectory;
# - latest request + assessment readiness.
#
# This avoids the long-history semantic averaging failure seen in earlier versions. Cached embeddings are keyed by model + prompt + exact text content.

# ---- cell 16 (code) ----
import torch
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer

assert torch.cuda.is_available(), "V23 full mode requires a Kaggle GPU."
if not V23_SMOKE:
    assert torch.cuda.device_count() >= 2, (
        f"Select Kaggle T4 x2 accelerator for the intended V23 run; detected {torch.cuda.device_count()} GPU(s)."
    )
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

V23_QUERY_PROMPT = (
    "Instruct: Given a learner's current request, recent learning trajectory, and skill readiness, "
    "retrieve the course or local learning-step recommendation that is most useful to learn now. "
    "Distinguish an immediate next step from a longer-term destination.\nQuery: "
)
CACHE_DIR = Path("/kaggle/working/v23_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def text_digest(texts, is_query):
    h = hashlib.sha1()
    h.update(F2LLM_MODEL.encode())
    h.update(str(F2LLM_MAX_SEQ).encode())
    h.update((V23_QUERY_PROMPT if is_query else "DOCUMENT").encode())
    for t in texts:
        h.update(str(t).encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return h.hexdigest()[:18]


def load_f2(device):
    print("Loading", F2LLM_MODEL, "on", device)
    model = SentenceTransformer(
        F2LLM_MODEL,
        device=device,
        model_kwargs={"torch_dtype": torch.float16, "low_cpu_mem_usage": True},
    )
    model.max_seq_length = F2LLM_MAX_SEQ
    return model

# One 4B copy fits comfortably on a 16 GB T4 in FP16 for short embedding batches.
# With T4x2 we use two independent copies to split inference deterministically.
f2_models = [load_f2("cuda:0")]
if F2LLM_DUAL_GPU and torch.cuda.device_count() >= 2:
    try:
        f2_models.append(load_f2("cuda:1"))
    except Exception as e:
        print("Second-GPU encoder load failed; deterministic single-GPU fallback:", repr(e))
        gc.collect(); torch.cuda.empty_cache()


def _encode_one(model, texts, is_query):
    prompt = V23_QUERY_PROMPT if is_query else None
    return model.encode(
        list(texts),
        batch_size=F2LLM_BATCH,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
        prompt=prompt,
    ).astype(np.float32)


def encode_cached(texts, name, is_query):
    texts = list(map(str, texts))
    key = text_digest(texts, is_query)
    path = CACHE_DIR / f"{name}_{key}.npy"
    if path.exists():
        arr = np.load(path)
        print("cache hit", path.name, arr.shape)
        return arr.astype(np.float32)

    print("encoding", name, "n=", len(texts), "using", len(f2_models), "GPU model(s)")
    if len(f2_models) == 1 or len(texts) < 4:
        arr = _encode_one(f2_models[0], texts, is_query)
    else:
        cut = (len(texts) + 1) // 2
        chunks = [texts[:cut], texts[cut:]]
        with ThreadPoolExecutor(max_workers=2) as ex:
            futures = [
                ex.submit(_encode_one, f2_models[i], chunks[i], is_query)
                for i in range(2)
            ]
            parts = [f.result() for f in futures]
        arr = np.vstack(parts).astype(np.float32)
    np.save(path, arr.astype(np.float16))
    print("saved", path.name, arr.shape)
    return arr

E_LATEST = encode_cached(latest_view, "latest", True)
E_RECENT = encode_cached(recent_view, "recent", True)
E_PROFILE = encode_cached(profile_view, "profile", True)
E_MODULE = encode_cached(module_docs, "module_docs", False)
E_ACTION = encode_cached(action_docs, "action_docs", False)

assert E_LATEST.shape[0] == len(all_users)
assert E_MODULE.shape[0] == N_MODULES
assert E_ACTION.shape[0] == len(ACTION_PAIRS)

# Only 17 module documents, so these dense similarity matrices are cheap.
SIM_LATEST = E_LATEST @ E_MODULE.T
SIM_RECENT = E_RECENT @ E_MODULE.T
SIM_PROFILE = E_PROFILE @ E_MODULE.T
print("semantic matrices", SIM_LATEST.shape, SIM_RECENT.shape, SIM_PROFILE.shape)

# ---- cell 17 (markdown) ----
# ## Direct local-swap utility dataset
#
# For every candidate, the action is always **promote the currently lower-ranked module over the currently higher-ranked module by swapping their score values**. The target is the exact resulting user-level change in NDCG@5.
#
# Novel Top-1 changes are not part of the learnable action space. The already public-validated `U_4047` Top-1 correction remains only inside the immutable PB anchor.

# ---- cell 18 (code) ----
# Exact analytical swap delta for a score-position exchange.
IDEAL_DCG = (np.sort(y, axis=1)[:, -5:][:, ::-1] * DISCOUNT).sum(1)


def build_candidates(scores, global_user_indices, split, with_target):
    rows = []
    scores = np.asarray(scores, np.float32)
    for local_i, ui in enumerate(global_user_indices):
        order = np.argsort(-scores[local_i])[:ACTION_TOPK]
        # r_hi is 0-based rank of the currently higher-scored module.
        start_rank = 1 if LOCK_NOVEL_TOP1 else 0
        for r_hi in range(start_rank, min(5, ACTION_TOPK-1)):
            for dist in range(1, ACTION_MAX_RANK_DISTANCE+1):
                r_lo = r_hi + dist
                if r_lo >= ACTION_TOPK:
                    continue
                hi = int(order[r_hi])
                lo = int(order[r_lo])
                margin = float(scores[local_i,hi] - scores[local_i,lo])
                if margin <= 0:
                    continue
                row = {
                    "split": split,
                    "user_idx": int(ui),
                    "local_idx": int(local_i),
                    "user_id": all_users[int(ui)],
                    "hi": hi,
                    "lo": lo,
                    "hi_mid": MODULES[hi],
                    "lo_mid": MODULES[lo],
                    "pair_key": int(lo*N_MODULES + hi),
                    "rank_hi": int(r_hi+1),
                    "rank_lo": int(r_lo+1),
                    "rank_distance": int(dist),
                    "hi_score": float(scores[local_i,hi]),
                    "lo_score": float(scores[local_i,lo]),
                    "margin": margin,
                    "cross_top5": int(r_hi < 5 <= r_lo),
                    "both_top5": int(r_lo < 5),
                    "lo_is_prereq_hi": float(PREREQ_MATRIX[lo,hi]),
                    "hi_is_prereq_lo": float(PREREQ_MATRIX[hi,lo]),
                    "prereq_question": float(PREREQ_ASK[ui]),
                    "latest_lo": float(LATEST_MENTION[ui,lo]),
                    "latest_hi": float(LATEST_MENTION[ui,hi]),
                    "recent_lo": float(RECENT_MENTION[ui,lo]),
                    "recent_hi": float(RECENT_MENTION[ui,hi]),
                    "any_lo": float(ANY_MENTION[ui,lo]),
                    "any_hi": float(ANY_MENTION[ui,hi]),
                }
                if with_target:
                    # Because scores are swapped between exact rank positions, the NDCG delta is analytical.
                    idcg = max(float(IDEAL_DCG[ui]), 1e-12)
                    y_hi = float(y[ui,hi]); y_lo = float(y[ui,lo])
                    if r_lo < 5:
                        d = (y_lo-y_hi) * float(DISCOUNT[r_hi] - DISCOUNT[r_lo]) / idcg
                    else:
                        d = (y_lo-y_hi) * float(DISCOUNT[r_hi]) / idcg
                    row["delta"] = float(d)
                    row["positive"] = int(d > 1e-12)
                    row["fold"] = int(ref_fold_id[ui])
                rows.append(row)
    return pd.DataFrame(rows)

train_candidates = build_candidates(
    consensus3_oof,
    np.arange(N_TRAIN, dtype=int),
    split="train",
    with_target=True,
)
test_candidates = build_candidates(
    consensus3_test,
    np.arange(N_TRAIN, N_TRAIN+N_TEST, dtype=int),
    split="test",
    with_target=False,
)

# Exclude the two already-public-validated PB users from all novel actions.
test_candidates = test_candidates[~test_candidates["user_id"].isin({"U_4047","U_4953"})].copy()

print("all local candidates", train_candidates.shape, test_candidates.shape)
print("positive action fraction before margin gate", round(train_candidates["positive"].mean(), 4))

# ---- cell 19 (code) ----
# Label-independent semantic scalars attached to candidate actions.
def attach_semantic(df):
    df = df.copy()
    ui = df["user_idx"].to_numpy(int)
    lo = df["lo"].to_numpy(int)
    hi = df["hi"].to_numpy(int)

    df["sem_latest_lo"] = SIM_LATEST[ui, lo]
    df["sem_latest_hi"] = SIM_LATEST[ui, hi]
    df["sem_recent_lo"] = SIM_RECENT[ui, lo]
    df["sem_recent_hi"] = SIM_RECENT[ui, hi]
    df["sem_profile_lo"] = SIM_PROFILE[ui, lo]
    df["sem_profile_hi"] = SIM_PROFILE[ui, hi]
    df["sem_delta_latest"] = df["sem_latest_lo"] - df["sem_latest_hi"]
    df["sem_delta_recent"] = df["sem_recent_lo"] - df["sem_recent_hi"]
    df["sem_delta_profile"] = df["sem_profile_lo"] - df["sem_profile_hi"]

    action_idx = np.fromiter(
        (ACTION_PAIR_TO_IDX[(int(a),int(b))] for a,b in zip(lo,hi)),
        dtype=np.int32,
        count=len(df),
    )
    # Candidate-wise late interaction: no 5000 x 272 dense action matrix is materialized.
    df["sem_action_latest"] = np.einsum("ij,ij->i", E_LATEST[ui], E_ACTION[action_idx])
    df["sem_action_recent"] = np.einsum("ij,ij->i", E_RECENT[ui], E_ACTION[action_idx])
    df["sem_action_profile"] = np.einsum("ij,ij->i", E_PROFILE[ui], E_ACTION[action_idx])
    df["sem_action_mean"] = df[["sem_action_latest","sem_action_recent","sem_action_profile"]].mean(axis=1)
    df["sem_action_min"] = df[["sem_action_latest","sem_action_recent","sem_action_profile"]].min(axis=1)
    return df

train_candidates = attach_semantic(train_candidates)
test_candidates = attach_semantic(test_candidates)

print(train_candidates[[
    "margin","delta","sem_delta_latest","sem_delta_recent","sem_action_mean"
]].describe().round(4))

# ---- cell 20 (markdown) ----
# ## Structured branch vs semantic-confirmed branch
#
# V23 trains two independent views of the same action:
#
# - **Structured:** Consensus3 geometry, organizer chat/assessment features, module identities, mention flags, prerequisite flags.
# - **Hybrid:** the same structured features plus F2LLM module/action similarities.
#
# A novel action is deployable only when both branches predict positive expected utility. High-semantic-similarity actions that are actually harmful are upweighted as **hard negatives** in the hybrid branch.

# ---- cell 21 (code) ----
MODULE_EYE = np.eye(N_MODULES, dtype=np.float32)

BASE_ACTION_COLS = [
    "rank_hi","rank_lo","rank_distance","hi_score","lo_score","margin",
    "cross_top5","both_top5","lo_is_prereq_hi","hi_is_prereq_lo",
    "prereq_question","latest_lo","latest_hi","recent_lo","recent_hi","any_lo","any_hi",
]
SEM_ACTION_COLS = [
    "sem_latest_lo","sem_latest_hi","sem_recent_lo","sem_recent_hi",
    "sem_profile_lo","sem_profile_hi",
    "sem_delta_latest","sem_delta_recent","sem_delta_profile",
    "sem_action_latest","sem_action_recent","sem_action_profile",
    "sem_action_mean","sem_action_min",
]


def feature_matrix(df, include_semantic):
    ui = df["user_idx"].to_numpy(int)
    hi = df["hi"].to_numpy(int)
    lo = df["lo"].to_numpy(int)
    pieces = [
        df[BASE_ACTION_COLS].to_numpy(np.float32),
        USER_FEATURES[ui].astype(np.float32),
        MODULE_EYE[hi],
        MODULE_EYE[lo],
    ]
    if include_semantic:
        pieces.append(df[SEM_ACTION_COLS].to_numpy(np.float32))
    x = np.column_stack(pieces).astype(np.float32)
    assert np.isfinite(x).all()
    return x


def make_action_clf(seed):
    return lgb.LGBMClassifier(
        n_estimators=ACTION_MODEL_TREES,
        learning_rate=ACTION_MODEL_LR,
        num_leaves=7,
        max_depth=3,
        min_child_samples=60 if not V23_SMOKE else 15,
        reg_alpha=2.0,
        reg_lambda=12.0,
        colsample_bytree=0.75,
        subsample=0.85,
        subsample_freq=1,
        random_state=int(seed),
        n_jobs=4,
        verbosity=-1,
    )


def make_action_reg(seed):
    return lgb.LGBMRegressor(
        objective="huber",
        alpha=0.85,
        n_estimators=ACTION_MODEL_TREES,
        learning_rate=ACTION_MODEL_LR,
        num_leaves=7,
        max_depth=3,
        min_child_samples=60 if not V23_SMOKE else 15,
        reg_alpha=2.0,
        reg_lambda=12.0,
        colsample_bytree=0.75,
        subsample=0.85,
        subsample_freq=1,
        random_state=int(seed),
        n_jobs=4,
        verbosity=-1,
    )


def fit_action_branch(df, include_semantic, seed):
    x = feature_matrix(df, include_semantic)
    y_cls = df["positive"].to_numpy(np.int32)
    y_reg = df["delta"].to_numpy(np.float32)
    base_w = (1.0 + 2.0*np.minimum(np.abs(y_reg), 0.25)).astype(np.float32)
    if include_semantic:
        sem_q = float(df["sem_action_mean"].quantile(0.75))
        hard_neg = (y_cls == 0) & (df["sem_action_mean"].to_numpy(float) >= sem_q)
        base_w = base_w * np.where(hard_neg, HARD_NEGATIVE_MULT, 1.0).astype(np.float32)

    clf = make_action_clf(seed)
    reg = make_action_reg(seed+10000)
    clf.fit(x, y_cls, sample_weight=base_w)
    reg.fit(x, y_reg, sample_weight=base_w)
    return clf, reg


def predict_action_branch(models, df, include_semantic):
    clf, reg = models
    x = feature_matrix(df, include_semantic)
    p = clf.predict_proba(x)[:,1].astype(np.float32)
    mu = reg.predict(x).astype(np.float32)
    return p, mu

# ---- cell 22 (code) ----
# Pair-family discovery is label-using, therefore it is kept OUT of model features and used only
# as an outer-train deployment gate. This avoids target leakage from per-pair utility priors.
def discover_pair_gate(df, available_folds, confirmed=False):
    stats = []
    available_folds = sorted(set(map(int, available_folds)))
    for pair_key, g in df.groupby("pair_key", sort=False):
        fold_means = g.groupby("fold")["delta"].mean()
        nfolds = int(len(fold_means))
        foldwins = int((fold_means > 0).sum())
        need_wins = max(1, math.ceil(0.60*max(nfolds,1)))
        if confirmed:
            need_wins = max(need_wins, nfolds-0)  # unanimous among observed outer-train folds
        mean_u = float(g["delta"].mean())
        pos_rate = float(g["positive"].mean())
        allowed = (
            len(g) >= PAIR_MIN_SUPPORT and
            nfolds >= min(3, len(available_folds)) and
            mean_u > PAIR_MIN_MEAN_UTILITY and
            pos_rate >= PAIR_MIN_POS_RATE and
            foldwins >= need_wins
        )
        lo = int(g.iloc[0]["lo"]); hi = int(g.iloc[0]["hi"])
        stats.append({
            "pair_key": int(pair_key),
            "pair": f"{MODULES[lo]}>{MODULES[hi]}",
            "support": int(len(g)),
            "mean_delta": mean_u,
            "positive_rate": pos_rate,
            "nfolds": nfolds,
            "foldwins": foldwins,
            "worst_fold_mean": float(fold_means.min()) if nfolds else np.nan,
            "allowed": bool(allowed),
        })
    out = pd.DataFrame(stats)
    allowed_keys = set(out.loc[out["allowed"], "pair_key"].astype(int).tolist()) if len(out) else set()
    return allowed_keys, out


def confidence_columns(df):
    df = df.copy()
    pos_mu_s = np.clip(df["mu_struct"].to_numpy(float), 0, None)
    pos_mu_h = np.clip(df["mu_hybrid"].to_numpy(float), 0, None)
    p_s = np.clip(df["p_struct"].to_numpy(float), 1e-6, 1)
    p_h = np.clip(df["p_hybrid"].to_numpy(float), 1e-6, 1)
    # Geometric agreement: one pessimistic branch collapses the score.
    df["action_confidence"] = np.sqrt(pos_mu_s*pos_mu_h) * np.sqrt(p_s*p_h)
    df["branch_min_mu"] = np.minimum(pos_mu_s, pos_mu_h)
    df["branch_min_p"] = np.minimum(p_s, p_h)
    return df

# ---- cell 23 (markdown) ----
# ## Fully outer-fold-isolated action validation
#
# For each reference fold:
#
# 1. the low-margin threshold is learned only from the other folds;
# 2. pair-family support and fold consistency are discovered only from the other folds;
# 3. structured and hybrid action models fit only the other folds;
# 4. the held-out fold receives predictions once;
# 5. deployment policies are evaluated on the actual NDCG effect of the chosen swaps.
#
# The historical Consensus3 OOF surface itself stays frozen. V23 never uses the held-out action labels to pick its candidate families.

# ---- cell 24 (code) ----
# OOF action predictions
for col in ["p_struct","mu_struct","p_hybrid","mu_hybrid","pair_allowed","pair_confirmed","sem_confirm_thr","margin_thr"]:
    train_candidates[col] = np.nan

fold_pair_tables = []
margin_thresholds = {}
semantic_thresholds = {}

for f in range(N_FOLDS):
    train_user_mask = ref_fold_id != f
    val_user_mask = ref_fold_id == f
    train_user_ids = set(np.flatnonzero(train_user_mask).tolist())
    val_user_ids = set(np.flatnonzero(val_user_mask).tolist())

    base_fit = train_candidates[train_candidates["user_idx"].isin(train_user_ids)].copy()
    margin_thr = float(base_fit["margin"].quantile(ACTION_MARGIN_QUANTILE))
    fit_df = base_fit[base_fit["margin"] <= margin_thr].copy()
    val_idx = train_candidates.index[
        train_candidates["user_idx"].isin(val_user_ids) &
        (train_candidates["margin"] <= margin_thr)
    ]
    val_df = train_candidates.loc[val_idx].copy()

    # Pair discovery uses only the outer-training action labels.
    available = sorted(set(fit_df["fold"].astype(int).unique().tolist()))
    allowed, pair_tab = discover_pair_gate(fit_df, available, confirmed=False)
    confirmed, pair_tab_strict = discover_pair_gate(fit_df, available, confirmed=True)
    pair_tab["outer_fold"] = f
    pair_tab["strict_allowed"] = pair_tab["pair_key"].isin(confirmed)
    fold_pair_tables.append(pair_tab)

    struct_models = fit_action_branch(fit_df, include_semantic=False, seed=SEED+100*f)
    hybrid_models = fit_action_branch(fit_df, include_semantic=True, seed=SEED+100*f+17)
    ps, mus = predict_action_branch(struct_models, val_df, include_semantic=False)
    ph, muh = predict_action_branch(hybrid_models, val_df, include_semantic=True)

    sem_thr = float(fit_df["sem_action_mean"].quantile(0.60))
    train_candidates.loc[val_idx, "p_struct"] = ps
    train_candidates.loc[val_idx, "mu_struct"] = mus
    train_candidates.loc[val_idx, "p_hybrid"] = ph
    train_candidates.loc[val_idx, "mu_hybrid"] = muh
    train_candidates.loc[val_idx, "pair_allowed"] = val_df["pair_key"].isin(allowed).astype(float).to_numpy()
    train_candidates.loc[val_idx, "pair_confirmed"] = val_df["pair_key"].isin(confirmed).astype(float).to_numpy()
    train_candidates.loc[val_idx, "sem_confirm_thr"] = sem_thr
    train_candidates.loc[val_idx, "margin_thr"] = margin_thr
    margin_thresholds[f] = margin_thr
    semantic_thresholds[f] = sem_thr

    print(
        f"fold {f}: margin<={margin_thr:.5f} fit={len(fit_df)} val={len(val_df)} "
        f"pair_gate={len(allowed)} strict_pairs={len(confirmed)} sem60={sem_thr:.4f}"
    )
    del struct_models, hybrid_models
    gc.collect()

oof_actions = train_candidates[
    train_candidates["p_struct"].notna() &
    train_candidates["p_hybrid"].notna()
].copy()
oof_actions = confidence_columns(oof_actions)
assert oof_actions["fold"].nunique() == N_FOLDS
print("OOF scored candidate actions", oof_actions.shape)

# ---- cell 25 (code) ----
# Policy selection and exact OOF NDCG evaluation.
def policy_eligible(df, policy):
    base = (
        (df["pair_allowed"].fillna(0).to_numpy(float) > 0.5) &
        (df["branch_min_mu"].to_numpy(float) > 0) &
        (df["action_confidence"].to_numpy(float) > 0)
    )
    if policy == "confirmed":
        base &= (df["pair_confirmed"].fillna(0).to_numpy(float) > 0.5)
        base &= (df["sem_action_mean"].to_numpy(float) >= df["sem_confirm_thr"].to_numpy(float))
    return base


def choose_one_per_user(df, k, policy):
    if len(df) == 0 or k <= 0:
        return df.iloc[:0].copy()
    e = df.loc[policy_eligible(df, policy)].copy()
    if len(e) == 0:
        return e
    e = e.sort_values(
        ["action_confidence","branch_min_mu","branch_min_p","margin"],
        ascending=[False,False,False,True],
    )
    e = e.drop_duplicates("user_id", keep="first")
    return e.head(int(k)).copy()


def apply_action_rows(scores, action_rows, score_user_offset=0):
    out = np.asarray(scores, np.float32).copy()
    for r in action_rows.itertuples(index=False):
        local_i = int(r.user_idx) - int(score_user_offset)
        hi, lo = int(r.hi), int(r.lo)
        assert out[local_i,hi] > out[local_i,lo], (
            r.user_id, MODULES[hi], MODULES[lo], out[local_i,hi], out[local_i,lo]
        )
        out[local_i,hi], out[local_i,lo] = out[local_i,lo], out[local_i,hi]
    return out

policy_rows = []
policy_selected = {}
base_user_ndcg = per_user_ndcg(y, consensus3_oof)

for policy, rate in POLICY_RATES.items():
    selected_parts = []
    fold_deltas = []
    for f in range(N_FOLDS):
        g = oof_actions[oof_actions["fold"] == f].copy()
        n_fold_users = int((ref_fold_id == f).sum())
        k = max(1, int(round(rate*n_fold_users)))
        chosen = choose_one_per_user(g, k, policy)
        selected_parts.append(chosen)
        if len(chosen):
            cand_scores = apply_action_rows(consensus3_oof, chosen, score_user_offset=0)
            fold_mask = ref_fold_id == f
            fold_delta = float(
                per_user_ndcg(y[fold_mask], cand_scores[fold_mask]).mean() -
                per_user_ndcg(y[fold_mask], consensus3_oof[fold_mask]).mean()
            )
        else:
            fold_delta = 0.0
        fold_deltas.append(fold_delta)

    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else oof_actions.iloc[:0].copy()
    policy_selected[policy] = selected
    candidate_oof = apply_action_rows(consensus3_oof, selected, score_user_offset=0)
    overall_delta = float(ndcg5(y, candidate_oof) - ndcg5(y, consensus3_oof))
    precision = float(selected["positive"].mean()) if len(selected) else 0.0
    selected_mean_delta = float(selected["delta"].mean()) if len(selected) else 0.0
    fold_wins = int(np.sum(np.asarray(fold_deltas) > 0))
    worst_fold = float(np.min(fold_deltas)) if fold_deltas else 0.0
    gate = (
        overall_delta >= MIN_POLICY_OOF_DELTA and
        fold_wins >= MIN_POLICY_FOLD_WINS and
        worst_fold >= MAX_POLICY_WORST_FOLD and
        precision >= MIN_SELECTED_PRECISION
    )
    policy_rows.append({
        "policy":policy,
        "rate":rate,
        "selected":int(len(selected)),
        "oof_delta":overall_delta,
        "selected_precision":precision,
        "selected_mean_delta":selected_mean_delta,
        "fold_wins":fold_wins,
        "worst_fold":worst_fold,
        "fold_deltas":fold_deltas,
        "promotion_gate":bool(gate),
    })

policy_diag = pd.DataFrame(policy_rows)
display(policy_diag)

# ---- cell 26 (markdown) ----
# ## Full-fit deployment on test
#
# The full action model is trained only after the complete OOF audit has been generated. Test pair-family gates use all five OOF folds, while the two PB users are explicitly excluded from novel actions so the validated 0.65912 corrections cannot be overwritten.

# ---- cell 27 (code) ----
# Full training candidate geometry from train only.
global_margin_thr = float(train_candidates["margin"].quantile(ACTION_MARGIN_QUANTILE))
full_fit = train_candidates[train_candidates["margin"] <= global_margin_thr].copy()
full_test = test_candidates[test_candidates["margin"] <= global_margin_thr].copy()

all_folds = sorted(set(full_fit["fold"].astype(int).unique().tolist()))
allowed_full, pair_full = discover_pair_gate(full_fit, all_folds, confirmed=False)
confirmed_full, _ = discover_pair_gate(full_fit, all_folds, confirmed=True)
pair_full["strict_allowed"] = pair_full["pair_key"].isin(confirmed_full)
full_sem_thr = float(full_fit["sem_action_mean"].quantile(0.60))

print(
    "full margin threshold", round(global_margin_thr,6),
    "fit", len(full_fit), "test", len(full_test),
    "allowed pairs", len(allowed_full), "strict", len(confirmed_full),
)

action_struct = fit_action_branch(full_fit, include_semantic=False, seed=SEED+2300)
action_hybrid = fit_action_branch(full_fit, include_semantic=True, seed=SEED+2317)
ps, mus = predict_action_branch(action_struct, full_test, include_semantic=False)
ph, muh = predict_action_branch(action_hybrid, full_test, include_semantic=True)

full_test["p_struct"] = ps
full_test["mu_struct"] = mus
full_test["p_hybrid"] = ph
full_test["mu_hybrid"] = muh
full_test["pair_allowed"] = full_test["pair_key"].isin(allowed_full).astype(float)
full_test["pair_confirmed"] = full_test["pair_key"].isin(confirmed_full).astype(float)
full_test["sem_confirm_thr"] = full_sem_thr
full_test["margin_thr"] = global_margin_thr
full_test = confidence_columns(full_test)

# Geometry report: a large shift is a warning, not something hidden by calibration.
print("OOF confidence quantiles", np.quantile(oof_actions["action_confidence"], [0.5,0.9,0.95,0.99]).round(6))
print("TEST confidence quantiles", np.quantile(full_test["action_confidence"], [0.5,0.9,0.95,0.99]).round(6))

# ---- cell 28 (code) ----
# Test action budgets are exact caps, not forced counts: strict gates can return fewer actions.
def choose_test_actions(df, policy, cap):
    e = df.loc[policy_eligible(df, policy)].copy()
    if len(e) == 0:
        return e
    e = e.sort_values(
        ["action_confidence","branch_min_mu","branch_min_p","margin"],
        ascending=[False,False,False,True],
    ).drop_duplicates("user_id", keep="first")
    return e.head(int(cap)).copy()

TEST_CAPS = {"conservative":3, "balanced":6, "confirmed":8}
test_selected = {
    p: choose_test_actions(full_test, p, TEST_CAPS[p])
    for p in TEST_CAPS
}

attacks = {}
for policy, selected in test_selected.items():
    attacks[policy] = apply_action_rows(pb_test, selected, score_user_offset=N_TRAIN)
    print("\n", policy, "selected", len(selected))
    if len(selected):
        display(selected[[
            "user_id","lo_mid","hi_mid","rank_hi","rank_lo","margin",
            "lo_is_prereq_hi","prereq_question","p_struct","mu_struct",
            "p_hybrid","mu_hybrid","sem_action_mean","action_confidence"
        ]])

# ---- cell 29 (markdown) ----
# ## Fail-closed exports
#
# `primary` never silently becomes the most aggressive candidate. It uses the conservative `<=3` policy only if its fully OOF gate passes. If not, it falls back to the exact **0.65912 PB reference**. The other attack files remain available for deliberate Kaggle probing.

# ---- cell 30 (code) ----
OUT_DIR = Path("/kaggle/working")


def save_submission(scores, filename):
    out = sample.copy()
    out.loc[:, MODULES] = np.asarray(scores, np.float32)
    assert out.shape == sample.shape
    assert out["user_id"].tolist() == sample["user_id"].tolist()
    assert np.isfinite(out[MODULES].to_numpy()).all()
    path = OUT_DIR / filename
    out.to_csv(path, index=False)
    print("saved", path)
    return out

pb_submission = save_submission(pb_test, "submission_v23_pb_reference.csv")
sub_conservative = save_submission(attacks["conservative"], "submission_v23_attack1_conservative.csv")
sub_balanced = save_submission(attacks["balanced"], "submission_v23_attack2_balanced.csv")
sub_confirmed = save_submission(attacks["confirmed"], "submission_v23_attack3_confirmed.csv")

conservative_gate = bool(policy_diag.set_index("policy").loc["conservative", "promotion_gate"])
if conservative_gate and len(test_selected["conservative"]) > 0:
    primary_scores = attacks["conservative"]
    primary_name = "conservative"
else:
    primary_scores = pb_test
    primary_name = "pb_reference"
primary_submission = save_submission(primary_scores, "submission_v23_primary.csv")
print("PRIMARY POLICY:", primary_name)

# ---- cell 31 (code) ----
# Diagnostics and reproducibility artifacts.
pair_full = pair_full.sort_values(["allowed","foldwins","mean_delta","support"], ascending=[False,False,False,False])
pair_full.to_csv(OUT_DIR / "v23_pair_family_diagnostics.csv", index=False)
policy_diag.to_csv(OUT_DIR / "v23_policy_diagnostics.csv", index=False)

oof_export_cols = [
    "user_id","fold","lo_mid","hi_mid","rank_hi","rank_lo","margin","delta","positive",
    "lo_is_prereq_hi","prereq_question","sem_action_mean",
    "p_struct","mu_struct","p_hybrid","mu_hybrid","pair_allowed","pair_confirmed","action_confidence"
]
oof_actions[oof_export_cols].to_csv(OUT_DIR / "v23_oof_action_predictions.csv", index=False)

all_test_action_rows = []
for policy, dfp in test_selected.items():
    if len(dfp):
        z = dfp.copy(); z["policy"] = policy; all_test_action_rows.append(z)
if all_test_action_rows:
    test_actions_export = pd.concat(all_test_action_rows, ignore_index=True)
else:
    test_actions_export = full_test.iloc[:0].copy(); test_actions_export["policy"] = []
test_actions_export.to_csv(OUT_DIR / "v23_selected_test_actions.csv", index=False)

summary = {
    "version": "V23",
    "seed": SEED,
    "anchor": "V12.2 Consensus3 + public-validated U4047/U4953 score swaps",
    "public_anchor_score_recorded": 0.65912,
    "f2llm_model": F2LLM_MODEL,
    "action_margin_quantile": ACTION_MARGIN_QUANTILE,
    "global_margin_threshold": global_margin_thr,
    "novel_top1_locked": LOCK_NOVEL_TOP1,
    "pair_gate_allowed": int(len(allowed_full)),
    "pair_gate_confirmed": int(len(confirmed_full)),
    "policy_diagnostics": policy_diag.to_dict(orient="records"),
    "test_actions": {k:int(len(v)) for k,v in test_selected.items()},
    "primary_policy": primary_name,
}
with open(OUT_DIR / "v23_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=float)

np.savez_compressed(
    OUT_DIR / "v23_artifacts.npz",
    y=y,
    consensus3_oof=consensus3_oof,
    consensus3_test=consensus3_test,
    pb_test=pb_test,
    ref_fold_id=ref_fold_id,
    consensus_weights=CONS_W,
)
print(json.dumps(summary, indent=2, default=float))

# ---- cell 32 (markdown) ----
# ## Hard invariants
#
# These assertions intentionally fail rather than emit a silent ranking drift. They verify the competition schema, exact PB reconstruction, intervention caps, one-action-per-user policy, no overwrite of the two public-validated users, and no novel Top-1 changes relative to the PB anchor.

# ---- cell 33 (code) ----
assert len(MODULES) == 17 and y.shape == (4000,17) and N_TEST == 1000
assert CONSENSUS3_OVERRIDES == {
    "M_002":0.44,"M_006":0.55,"M_007":0.60,"M_010":0.40,
    "M_011":0.55,"M_012":0.55,"M_015":0.44,"M_017":0.53,
}
assert set(changed_uids) == {"U_4047","U_4953"}

for policy, cap in TEST_CAPS.items():
    sel = test_selected[policy]
    assert len(sel) <= cap
    assert sel["user_id"].is_unique
    assert not set(sel["user_id"]).intersection({"U_4047","U_4953"})
    # Novel action generator locks rank #1, so attack vs PB must preserve Top-1 for every user.
    assert np.array_equal(np.argmax(attacks[policy], axis=1), np.argmax(pb_test, axis=1)), policy
    changed = np.flatnonzero(np.any(top5_indices(attacks[policy]) != top5_indices(pb_test), axis=1))
    assert len(changed) <= cap, (policy, len(changed), cap)

# Every submission retains exactly the organizer schema.
for df in [pb_submission, sub_conservative, sub_balanced, sub_confirmed, primary_submission]:
    assert list(df.columns) == list(sample.columns)
    assert df["user_id"].tolist() == sample["user_id"].tolist()
    assert np.isfinite(df[MODULES].to_numpy()).all()

print("ALL V23 INVARIANTS PASSED")
print("Recommended workflow: inspect v23_policy_diagnostics.csv and v23_selected_test_actions.csv; "
      "submit attack1 first only when its OOF evidence is acceptable. PB reference remains available unchanged.")
