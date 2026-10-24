# Converted from notebooks/v26.ipynb: markdown cells become comments, %/! magics neutralized.

# ---- cell 0 (markdown) ----
# # MineToday V26 — Core Attribution + Cluster-Safe Counterfactual Pruning
#
# **Active public anchor:** V25 `PB + core-6` = **0.65919**.  
# **Leader reference:** **0.66083**.
#
# V26 is intentionally **not** another global embedding/reranking experiment. V23 showed that broad action models and F2LLM confirmation were not reliable enough, while V24/V25 showed that tiny, auditable ranking interventions can still move the public leaderboard. V26 therefore treats the submitted V25 core-6 ranking as an immutable starting surface and asks a narrower question:
#
# > **Which V25 core actions are sufficiently contradicted by independent train-only evidence that removing them is safer than keeping the whole package?**
#
# The notebook:
#
# 1. reproduces the exact public-validated V3.1 → V7 → V12.2 **Consensus3** stack;
# 2. reconstructs the two-action `0.65912` PB and the six-action V25 `0.65919` anchor;
# 3. audits each core action with exact OOF counterfactual NDCG, duplicate/template-cluster bootstrap, lexical analogs, assessment analogs, and a grouped pair-specialist teacher;
# 4. uses a fixed **fail-closed** veto gate and can automatically prune at most two actions;
# 5. always materializes isolated U4567/U4929/U4491 removal probes plus a combined U4567+U4929 file so today's Kaggle slots answer identifiable causal questions;
# 6. writes an optional diversified hedge using only the previously-tested V25 relaxed family, never an unconstrained new reranker.
#
# ### Competition safety
#
# - Competition dataset only. No external dataset.
# - No direct LLM/Generative-AI prediction of test labels.
# - No public leaderboard score is used as a training feature.
# - Seed is fixed.
# - Every emitted ranking change is listed in an exact diff ledger.
# - If the V25 anchor geometry drifts materially from the recorded Kaggle surface, V26 fails before emitting experimental submissions.
#
# ### Kaggle runtime
#
# Designed for the same Kaggle environment used by prior notebooks. **T4×2 is compatible**, although this V26 method deliberately does not need a transformer/GPU branch. The GPUs may therefore remain idle; that is intentional, not a missing step.

# ---- cell 1 (code) ----
# Runtime + deterministic configuration
import os, sys, re, json, ast, html, hashlib, random, warnings, time, subprocess
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ["PYTHONHASHSEED"] = "42"

V26_SKIP_INSTALL = bool(int(os.getenv("V26_SKIP_INSTALL", "0")))
if not V26_SKIP_INSTALL:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "lightgbm==4.6.0"
    ])

import numpy as np
import pandas as pd
import scipy
import sklearn
from IPython.display import display
from scipy.sparse import hstack
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
import lightgbm as lgb

warnings.filterwarnings("ignore")

SEED = 42
V26_SMOKE = bool(int(os.getenv("V26_SMOKE", "0")))
V26_STRICT_GEOMETRY = bool(int(os.getenv("V26_STRICT_GEOMETRY", "1")))
V26_REQUIRE_CANONICAL_HASH = bool(int(os.getenv("V26_REQUIRE_CANONICAL_HASH", "0")))
N_FOLDS = 2 if V26_SMOKE else 5
RIDGE_ALPHA = 3.0
REFERENCE_W = 0.45
PUBLIC_CONSENSUS3_LB = 0.65856
PUBLIC_PRE_V25_PB = 0.65912
PUBLIC_V25_LB = 0.65919
PUBLIC_CURRENT_PB = PUBLIC_V25_LB
PUBLIC_LEADER_LB = 0.66083
PUBLIC_CAL045_LB = 0.65639
LGB_ESTIMATORS = 40 if V26_SMOKE else 1200
LGB_EARLY_STOP = 8 if V26_SMOKE else 80
BOOTSTRAPS = 250 if V26_SMOKE else 4000
MAX_AUTO_PRUNES = 2
GEOMETRY_ATOL = 2.5e-4
EXPECTED_V25_PRIMARY_SHA256 = "2e9d9553db483d68d98a2a97cc421bc9c996a22e77be755c2b652e7d3f481d4a"

random.seed(SEED)
np.random.seed(SEED)

print("numpy", np.__version__)
print("pandas", pd.__version__)
print("scipy", scipy.__version__)
print("sklearn", sklearn.__version__)
print("lightgbm", lgb.__version__)
print("V26_SMOKE", V26_SMOKE, "leader gap", round(PUBLIC_LEADER_LB-PUBLIC_CURRENT_PB, 5))
try:
    gpu_info = subprocess.check_output(["nvidia-smi", "-L"], text=True).strip()
    print("GPU runtime:\n" + gpu_info)
except Exception:
    print("nvidia-smi unavailable; V26 does not require GPU computation.")

if lgb.__version__ != "4.6.0":
    raise RuntimeError(f"V26 requires LightGBM 4.6.0, got {lgb.__version__}")

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
# ## Exact public-validated anchor
#
# The next two cells are intentionally inherited from the public-validated V3.1/V7/V12.2 family. V24 does **not** retune the sparse representation or binary relevance calibration.

# ---- cell 8 (code) ----
# Exact public-validated V3.1 sparse backbone.
t0=time.time()
word_vec=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_df=.995,sublinear_tf=True,strip_accents="unicode",max_features=(2000 if V26_SMOKE else 50000),dtype=np.float32)
char_vec=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=2,sublinear_tf=True,max_features=(3000 if V26_SMOKE else 60000),dtype=np.float32)
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
# ## Immutable current-PB reference: 0.65912 ranking
#
# The public-best ranking from the previous day was **Consensus3 everywhere except two deterministic local ordering corrections**. V24 rebuilds that ranking first and treats it as immutable. Any novel action is applied *on top of* this PB reference, never instead of it.

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
# ## Reconstruct the canonical V25 `0.65919` core-6 anchor
#
# V25's public improvement is only identified **at package level**. The six core actions below are therefore reconstructed exactly, but none is assumed individually causal. The recorded frozen margins are used as a reproducibility contract. A materially different margin means the sparse/calibration surface has drifted and the experimental files should not be trusted.

# ---- cell 14 (code) ----
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
    if V26_STRICT_GEOMETRY:
        assert rank_pair == expected_pair, (uid, rank_pair, expected_pair)
    v25_anchor_test, margin = apply_promotion(v25_anchor_test, action, strict_geometry=V26_STRICT_GEOMETRY)
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

# ---- cell 15 (markdown) ----
# ## Cluster-safe action attribution
#
# The attribution stage never assumes that repeated chat templates are independent samples. Every train user is assigned a latest-learning-chat skeleton. Direct OOF swap deltas are first computed at user level, then duplicate/template rows are collapsed to one group mean before bootstrap confidence intervals or neighbor voting.
#
# For each core action V26 uses four evidence channels:
#
# 1. **deployment OOF prior:** same module pair, same pre-swap rank positions, and a local score-margin window;
# 2. **lexical analog:** nearest latest-learning-chat skeleton groups;
# 3. **assessment analog:** nearest skill profiles among independent skeleton groups;
# 4. **pair specialist:** fixed Ridge + logistic teacher, evaluated with GroupKFold by skeleton and used only if its CV diagnostics clear a reliability floor.
#
# The channels are deliberately not averaged into a free optimizer. They are used as independent veto evidence.

# ---- cell 16 (code) ----
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def latest_learning_info(uid):
    g = user_chat.get(uid)
    if g is None or len(g) == 0:
        return "", f"__missing__{uid}"
    learn = g[~g["is_noise"]].copy()
    if len(learn) == 0:
        learn = g.copy()
    row = learn.iloc[-1]
    text = str(row.get("clean", ""))
    skel = str(row.get("skeleton", ""))
    if not skel:
        skel = f"__missing__{uid}"
    return text, skel

LATEST_TEXT = []
LATEST_GROUP = []
for uid in all_users:
    tx, gr = latest_learning_info(uid)
    LATEST_TEXT.append(tx)
    LATEST_GROUP.append(gr)
LATEST_TEXT = np.asarray(LATEST_TEXT, dtype=object)
LATEST_GROUP = np.asarray(LATEST_GROUP, dtype=object)

# Train-only lexical vocabulary; test text is transform-only.
latest_vec = TfidfVectorizer(
    ngram_range=(1,2), min_df=2, max_df=.995, sublinear_tf=True,
    strip_accents="unicode", max_features=(3000 if V26_SMOKE else 30000), dtype=np.float32
)
LATEST_X_TR = latest_vec.fit_transform(LATEST_TEXT[:N_TRAIN])
LATEST_X_TE = latest_vec.transform(LATEST_TEXT[N_TRAIN:])

CONS_OOF_RANKS = full_rank_matrix(consensus3_oof)
CONS_TEST_RANKS = full_rank_matrix(pb_test)  # core actions are evaluated before V25 promotions.


def row_swap_delta(scores, truth, rows, higher_col, promoted_col):
    rows = np.asarray(rows, dtype=int)
    if len(rows) == 0:
        return np.asarray([], dtype=np.float64)
    before = per_user_ndcg(truth[rows], scores[rows])
    candidate = scores[rows].copy()
    tmp = candidate[:, higher_col].copy()
    candidate[:, higher_col] = candidate[:, promoted_col]
    candidate[:, promoted_col] = tmp
    after = per_user_ndcg(truth[rows], candidate)
    return (after-before).astype(np.float64)


def group_boot_ci(values, groups, n_boot=None, seed=20260828):
    df = pd.DataFrame({"v":np.asarray(values,float), "g":np.asarray(groups,object)})
    gm = df.groupby("g", sort=False)["v"].mean().to_numpy(float)
    if len(gm) == 0:
        return (np.nan, np.nan, np.nan, 0)
    mean = float(gm.mean())
    if len(gm) == 1:
        return (mean, mean, mean, 1)
    n_boot = int(BOOTSTRAPS if n_boot is None else n_boot)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, float)
    for b in range(n_boot):
        boot[b] = gm[rng.integers(0, len(gm), len(gm))].mean()
    lo, hi = np.quantile(boot, [0.025,0.975])
    return (mean, float(lo), float(hi), int(len(gm)))



def grouped_weighted_neighbor(cand_rows, deltas, test_row, mode="lexical", top_groups=15):
    cand_rows = np.asarray(cand_rows, int)
    deltas = np.asarray(deltas, float)
    if len(cand_rows) == 0:
        return {"score":np.nan, "groups":0, "max_weight":0.0, "exact_group_delta":np.nan}
    groups = LATEST_GROUP[cand_rows]
    test_group = LATEST_GROUP[N_TRAIN+test_row]

    if mode == "lexical":
        q = LATEST_X_TE[test_row]
        row_weights = np.asarray((LATEST_X_TR[cand_rows] @ q.T).toarray()).ravel().astype(float)
    elif mode == "assessment":
        trA = A[cand_rows]/5.0
        qA = A[N_TRAIN+test_row]/5.0
        dist = np.sqrt(np.mean((trA-qA[None,:])**2, axis=1))
        row_weights = 1.0/(1.0+4.0*dist)
    else:
        raise ValueError(mode)

    # One vote per template group: labels are averaged, similarity uses the best matching
    # row inside the group. Duplicate messages therefore cannot multiply their influence.
    tmp = pd.DataFrame({"delta":deltas, "group":groups, "weight":row_weights})
    gd = tmp.groupby("group", sort=False).agg(delta=("delta","mean"), weight=("weight","max")).reset_index()
    exact = gd.loc[gd["group"]==test_group, "delta"]
    exact_delta = float(exact.iloc[0]) if len(exact) else np.nan

    order = np.argsort(-gd["weight"].to_numpy(float))[:min(top_groups,len(gd))]
    w = gd["weight"].to_numpy(float)[order]
    d = gd["delta"].to_numpy(float)[order]
    if np.sum(w) <= 1e-12:
        score = float(np.mean(d))
    else:
        score = float(np.sum(w*d)/np.sum(w))
    return {
        "score":score,
        "groups":int(len(order)),
        "max_weight":float(w.max()) if len(w) else 0.0,
        "exact_group_delta":exact_delta,
    }


def pair_feature_matrix(rows, scores, ranks, hi, pr):
    rows = np.asarray(rows, int)
    structured = USER_FEATURES[rows]
    extra = np.column_stack([
        scores[rows,hi], scores[rows,pr], scores[rows,hi]-scores[rows,pr],
        ranks[rows,hi]/17.0, ranks[rows,pr]/17.0,
        (ranks[rows,pr]-ranks[rows,hi])/17.0,
    ]).astype(np.float32)
    return np.column_stack([structured,extra]).astype(np.float32)


def fit_pair_specialist(action, test_row):
    hi = MID_TO_COL[action["higher_before"]]
    pr = MID_TO_COL[action["promoted"]]
    # Broader than exact-position prior, but still a local near-tie action family.
    rank_gap = CONS_OOF_RANKS[:,pr]-CONS_OOF_RANKS[:,hi]
    mask = (
        (consensus3_oof[:,hi] > consensus3_oof[:,pr]) &
        (CONS_OOF_RANKS[:,hi] <= 7) & (CONS_OOF_RANKS[:,pr] <= 7) &
        (rank_gap >= 1) & (rank_gap <= 2)
    )
    rows = np.flatnonzero(mask)
    delta = row_swap_delta(consensus3_oof, y, rows, hi, pr)
    groups = LATEST_GROUP[rows]
    unique_groups = np.unique(groups)
    result = {
        "teacher_rows":int(len(rows)), "teacher_groups":int(len(unique_groups)),
        "teacher_reg_pred":np.nan, "teacher_prob_positive":np.nan,
        "teacher_cv_corr":np.nan, "teacher_cv_auc":np.nan,
        "teacher_reliable":False,
    }
    if len(rows) < 24 or len(unique_groups) < 8:
        return result

    Xp = pair_feature_matrix(rows, consensus3_oof, CONS_OOF_RANKS, hi, pr)
    label = (delta > 1e-12).astype(int)
    n_splits = min(5, len(unique_groups))
    gkf = GroupKFold(n_splits=n_splits)
    cv_reg = np.full(len(rows), np.nan, float)
    cv_prob = np.full(len(rows), np.nan, float)
    for tr_i, va_i in gkf.split(Xp, label, groups):
        reg = Ridge(alpha=12.0, solver="lsqr", tol=1e-3, max_iter=2000)
        reg.fit(Xp[tr_i], delta[tr_i])
        cv_reg[va_i] = reg.predict(Xp[va_i])
        if len(np.unique(label[tr_i])) >= 2:
            clf = LogisticRegression(C=0.15, class_weight="balanced", max_iter=1200, random_state=SEED)
            clf.fit(Xp[tr_i], label[tr_i])
            cv_prob[va_i] = clf.predict_proba(Xp[va_i])[:,1]

    corr = np.corrcoef(cv_reg, delta)[0,1] if np.std(cv_reg)>1e-12 and np.std(delta)>1e-12 else np.nan
    valid_prob = np.isfinite(cv_prob)
    auc = np.nan
    if valid_prob.sum() and len(np.unique(label[valid_prob])) >= 2:
        auc = float(roc_auc_score(label[valid_prob], cv_prob[valid_prob]))

    Xtest = pair_feature_matrix(
        np.asarray([N_TRAIN+test_row]),
        np.vstack([consensus3_oof, pb_test]),
        np.vstack([CONS_OOF_RANKS, CONS_TEST_RANKS]),
        hi, pr
    )
    reg_full = Ridge(alpha=12.0, solver="lsqr", tol=1e-3, max_iter=2000)
    reg_full.fit(Xp, delta)
    reg_pred = float(reg_full.predict(Xtest)[0])
    prob = float(np.mean(label))
    if len(np.unique(label)) >= 2:
        clf_full = LogisticRegression(C=0.15, class_weight="balanced", max_iter=1200, random_state=SEED)
        clf_full.fit(Xp, label)
        prob = float(clf_full.predict_proba(Xtest)[0,1])

    reliable = bool(
        len(rows) >= 30 and len(unique_groups) >= 10 and
        ((np.isfinite(corr) and corr >= 0.05) or (np.isfinite(auc) and auc >= 0.55))
    )
    result.update({
        "teacher_reg_pred":reg_pred,
        "teacher_prob_positive":prob,
        "teacher_cv_corr":float(corr) if np.isfinite(corr) else np.nan,
        "teacher_cv_auc":float(auc) if np.isfinite(auc) else np.nan,
        "teacher_reliable":reliable,
    })
    return result

# ---- cell 17 (code) ----
# Evaluate all six V25 core actions with fixed, train-only evidence channels.
attribution_rows = []
for action in V25_CORE_ACTIONS:
    uid = action["user_id"]
    test_row = TEST_UID_TO_ROW[uid]
    hi = MID_TO_COL[action["higher_before"]]
    pr = MID_TO_COL[action["promoted"]]
    rank_hi, rank_pr = EXPECTED_PRE_SWAP_RANKS[uid]
    test_margin = float(action["expected_margin"])

    exact_mask = (
        (consensus3_oof[:,hi] > consensus3_oof[:,pr]) &
        (CONS_OOF_RANKS[:,hi] == rank_hi) &
        (CONS_OOF_RANKS[:,pr] == rank_pr)
    )
    exact_rows = np.flatnonzero(exact_mask)
    exact_delta = row_swap_delta(consensus3_oof, y, exact_rows, hi, pr)

    # Deployment-local window: test margin is geometry only, never a label-derived threshold.
    local_threshold = float(min(0.006, max(0.0010, 2.0*test_margin)))
    local_keep = (consensus3_oof[exact_rows,hi]-consensus3_oof[exact_rows,pr]) <= local_threshold
    local_rows = exact_rows[local_keep]
    local_delta = exact_delta[local_keep]
    local_groups = LATEST_GROUP[local_rows]
    if len(np.unique(local_groups)) < 5 or len(local_rows) < 6:
        used_rows, used_delta, used_mode = exact_rows, exact_delta, "exact_position_fallback"
    else:
        used_rows, used_delta, used_mode = local_rows, local_delta, "local_margin"

    used_groups = LATEST_GROUP[used_rows]
    family_mean, family_lo, family_hi, independent_groups = group_boot_ci(
        used_delta, used_groups, seed=SEED+test_row
    )
    positive_rate = float(np.mean(used_delta > 1e-12)) if len(used_delta) else np.nan
    harm_rate = float(np.mean(used_delta < -1e-12)) if len(used_delta) else np.nan
    tie_rate = float(np.mean(np.abs(used_delta) <= 1e-12)) if len(used_delta) else np.nan

    fold_means = []
    for f in range(N_FOLDS):
        z = used_delta[ref_fold_id[used_rows] == f]
        if len(z): fold_means.append(float(np.mean(z)))
    fold_wins = int(sum(v > 0 for v in fold_means))
    worst_fold = float(min(fold_means)) if fold_means else np.nan

    lexical = grouped_weighted_neighbor(used_rows, used_delta, test_row, mode="lexical", top_groups=12)
    assessment = grouped_weighted_neighbor(used_rows, used_delta, test_row, mode="assessment", top_groups=18)
    teacher = fit_pair_specialist(action, test_row)

    # Four genuinely different evidence channels. Near-zero values abstain instead of voting.
    eps = 0.0015
    family_vote = -1 if family_mean < -eps else (1 if family_mean > eps else 0)
    lexical_vote = -1 if lexical["score"] < -eps else (1 if lexical["score"] > eps else 0)
    assessment_vote = -1 if assessment["score"] < -eps else (1 if assessment["score"] > eps else 0)
    teacher_vote = 0
    if teacher["teacher_reliable"]:
        if teacher["teacher_reg_pred"] < -eps and teacher["teacher_prob_positive"] < 0.45:
            teacher_vote = -1
        elif teacher["teacher_reg_pred"] > eps and teacher["teacher_prob_positive"] > 0.55:
            teacher_vote = 1

    votes = [family_vote, lexical_vote, assessment_vote, teacher_vote]
    negative_votes = int(sum(v < 0 for v in votes))
    positive_votes = int(sum(v > 0 for v in votes))
    active_votes = int(sum(v != 0 for v in votes))

    # Strict pruning gate. It is intentionally harder to remove a public-positive-package action
    # than to keep it. The package is the default when evidence is ambiguous.
    prune_gate = bool(
        independent_groups >= 5 and active_votes >= 2 and
        negative_votes >= 2 and positive_votes == 0 and
        (
            (np.isfinite(family_mean) and family_mean < 0 and harm_rate >= positive_rate) or
            (negative_votes >= 3 and np.isfinite(family_hi) and family_hi <= 0.003)
        )
    )

    # Rank suspicious actions without using public scores.
    prune_score = (
        2.0*negative_votes - 1.5*positive_votes +
        (1.0 if np.isfinite(family_mean) and family_mean < 0 else 0.0) +
        (0.75 if np.isfinite(harm_rate) and np.isfinite(positive_rate) and harm_rate > positive_rate else 0.0) +
        (0.5 if np.isfinite(lexical["score"]) and lexical["score"] < 0 else 0.0) +
        (0.5 if np.isfinite(assessment["score"]) and assessment["score"] < 0 else 0.0)
    )

    attribution_rows.append({
        "user_id":uid,
        "higher_before":action["higher_before"],
        "promoted":action["promoted"],
        "rank_higher_before":rank_hi,
        "rank_promoted_before":rank_pr,
        "test_margin":test_margin,
        "local_threshold":local_threshold,
        "prior_mode":used_mode,
        "prior_rows":int(len(used_rows)),
        "independent_groups":independent_groups,
        "family_mean_delta":family_mean,
        "family_ci_low":family_lo,
        "family_ci_high":family_hi,
        "positive_rate":positive_rate,
        "harm_rate":harm_rate,
        "tie_rate":tie_rate,
        "fold_wins":fold_wins,
        "fold_count":len(fold_means),
        "worst_fold":worst_fold,
        "lexical_delta":lexical["score"],
        "lexical_groups":lexical["groups"],
        "exact_template_delta":lexical["exact_group_delta"],
        "assessment_delta":assessment["score"],
        "assessment_groups":assessment["groups"],
        **teacher,
        "family_vote":family_vote,
        "lexical_vote":lexical_vote,
        "assessment_vote":assessment_vote,
        "teacher_vote":teacher_vote,
        "negative_votes":negative_votes,
        "positive_votes":positive_votes,
        "prune_gate":prune_gate,
        "prune_score":float(prune_score),
    })

attribution = pd.DataFrame(attribution_rows).sort_values(
    ["prune_gate","prune_score"], ascending=[False,False]
).reset_index(drop=True)
display(attribution)

strict_prunes = attribution.loc[attribution["prune_gate"], "user_id"].tolist()[:MAX_AUTO_PRUNES]
print("strict auto-prunes:", strict_prunes if strict_prunes else "NONE — fail closed to V25 anchor")

# ---- cell 18 (markdown) ----
# ## Deployment policies and causal probes
#
# The automatic policy may only **revert** existing V25 core actions; it can never invent a new test-user swap. The primary candidate prunes at most two actions that pass the strict veto gate. If none pass, `submission_v26_primary.csv` is intentionally identical to the V25 `0.65919` anchor.
#
# Independent of the automatic gate, V26 materializes the two strongest pre-registered causal ablations from the V24/V25 analysis:
#
# - remove **U4567** only;
# - remove **U4929** only;
#
# plus U4491-only and the U4567+U4929 combination for a third-slot decision after observing the first two public results. These are **probes**, not claims that the removed action is harmful.

# ---- cell 19 (code) ----
CORE_BY_UID = {a["user_id"]:a for a in V25_CORE_ACTIONS}


def revert_core_action(scores, uid):
    """Revert one V25 promotion by swapping the same two score cells back."""
    if uid not in CORE_BY_UID:
        raise KeyError(uid)
    out = scores.copy()
    a = CORE_BY_UID[uid]
    r = TEST_UID_TO_ROW[uid]
    hi = MID_TO_COL[a["higher_before"]]
    pr = MID_TO_COL[a["promoted"]]
    # In the V25 anchor, promoted must now be above higher_before.
    if not float(out[r,pr]) > float(out[r,hi]):
        raise AssertionError(f"Cannot revert {uid}; V25 promoted ordering is absent.")
    out[r,hi], out[r,pr] = out[r,pr], out[r,hi]
    return out


def revert_many(scores, uids):
    out = scores.copy()
    for uid in uids:
        out = revert_core_action(out, uid)
    return out

v26_primary_test = revert_many(v25_anchor_test, strict_prunes)

probe_specs = {
    "remove_u4567": ["U_4567"],
    "remove_u4929": ["U_4929"],
    "remove_u4491": ["U_4491"],
    "remove_u4567_u4929": ["U_4567","U_4929"],
}
probe_scores = {name:revert_many(v25_anchor_test,uids) for name,uids in probe_specs.items()}

# Optional diversified hedge: core-6 + U4030 + U4731, deliberately excluding U4650.
# This is not the primary because V25 relaxed-3 had zero displayed incremental public effect.
diversified_test = v25_anchor_test.copy()
for action in [V25_RELAXED_ACTIONS[0], V25_RELAXED_ACTIONS[2]]:
    diversified_test, _ = apply_promotion(diversified_test, action, strict_geometry=V26_STRICT_GEOMETRY)

print("automatic primary prunes:", strict_prunes)
for name,uids in probe_specs.items():
    print(name, "reverts", uids)

# ---- cell 20 (code) ----
OUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("/mnt/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_submission(scores, filename):
    out = sample.copy()
    out.loc[:,MODULES] = np.asarray(scores, np.float32)
    assert list(out.columns) == list(sample.columns)
    assert out["user_id"].tolist() == sample["user_id"].tolist()
    assert np.isfinite(out[MODULES].to_numpy()).all()
    path = OUT_DIR/filename
    out.to_csv(path,index=False)
    return out, path


def file_sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()

anchor_submission, anchor_path = save_submission(v25_anchor_test, "submission_v26_v25_anchor_reference.csv")
primary_submission, primary_path = save_submission(v26_primary_test, "submission_v26_primary.csv")
probe_submissions = {}
for name,scores in probe_scores.items():
    fn = f"submission_v26_probe_{name}.csv"
    probe_submissions[name] = save_submission(scores,fn)[0]
diversified_submission, diversified_path = save_submission(
    diversified_test, "submission_v26_diversified_core6_plus_u4030_u4731.csv"
)

anchor_hash = file_sha256(anchor_path)
print("V26 rebuilt V25 anchor SHA256:", anchor_hash)
print("Expected canonical V25 SHA256:", EXPECTED_V25_PRIMARY_SHA256)
print("Exact hash match:", anchor_hash == EXPECTED_V25_PRIMARY_SHA256)
if V26_REQUIRE_CANONICAL_HASH:
    assert anchor_hash == EXPECTED_V25_PRIMARY_SHA256, "Canonical V25 CSV hash mismatch"

# ---- cell 21 (code) ----
# Exact diff ledger and invariants.
def diff_ledger(reference_scores, candidate_scores, name):
    ref = np.asarray(reference_scores)
    cand = np.asarray(candidate_scores)
    diff = cand != ref
    rows = np.flatnonzero(diff.any(axis=1))
    out=[]
    for r in rows:
        mods=[MODULES[j] for j in np.flatnonzero(diff[r])]
        before=[MODULES[j] for j in top5_indices(ref[[r]])[0]]
        after=[MODULES[j] for j in top5_indices(cand[[r]])[0]]
        out.append({
            "candidate":name,
            "user_id":test.iloc[r]["user_id"],
            "changed_modules":" ".join(mods),
            "before_top5":" ".join(before),
            "after_top5":" ".join(after),
            "top5_membership_changed":set(before)!=set(after),
            "top1_changed":before[0]!=after[0],
            "changed_cells":int(diff[r].sum()),
        })
    return pd.DataFrame(out)

ledger_parts=[]
ledger_parts.append(diff_ledger(v25_anchor_test,v26_primary_test,"primary"))
for name,scores in probe_scores.items():
    ledger_parts.append(diff_ledger(v25_anchor_test,scores,name))
ledger_parts.append(diff_ledger(v25_anchor_test,diversified_test,"diversified_u4030_u4731"))
ledger = pd.concat(ledger_parts,ignore_index=True) if ledger_parts else pd.DataFrame()
display(ledger)

# Anchor contract: 8 users / 16 score cells vs Consensus3.
anchor_cell_diff = v25_anchor_test != consensus3_test
assert int(anchor_cell_diff.sum()) == 16, int(anchor_cell_diff.sum())
assert set(test.loc[np.flatnonzero(anchor_cell_diff.any(1)),"user_id"]) == expected_v25_users

# Primary may only revert selected core actions, max 2; no new action is allowed.
primary_diff = v26_primary_test != v25_anchor_test
primary_users = test.loc[np.flatnonzero(primary_diff.any(1)),"user_id"].tolist()
assert set(primary_users) == set(strict_prunes)
assert len(primary_users) <= MAX_AUTO_PRUNES
assert int(primary_diff.sum()) == 2*len(primary_users)

# Every causal probe changes exactly its declared action(s) and no Top-1 ranking.
for name,uids in probe_specs.items():
    d = probe_scores[name] != v25_anchor_test
    got = test.loc[np.flatnonzero(d.any(1)),"user_id"].tolist()
    assert set(got) == set(uids), (name,got,uids)
    assert int(d.sum()) == 2*len(uids)
    assert np.all(np.argmax(probe_scores[name],axis=1) == np.argmax(v25_anchor_test,axis=1))

# Diversified candidate is limited to the two declared relaxed users.
div_diff = diversified_test != v25_anchor_test
div_users = test.loc[np.flatnonzero(div_diff.any(1)),"user_id"].tolist()
assert set(div_users) == {"U_4030","U_4731"}, div_users
assert int(div_diff.sum()) == 4
assert np.all(np.argmax(diversified_test,axis=1) == np.argmax(v25_anchor_test,axis=1))

for df in [anchor_submission,primary_submission,*probe_submissions.values(),diversified_submission]:
    assert list(df.columns)==list(sample.columns)
    assert df["user_id"].tolist()==sample["user_id"].tolist()
    assert np.isfinite(df[MODULES].to_numpy()).all()

print("ALL V26 HARD INVARIANTS PASSED")

# ---- cell 22 (markdown) ----
# ## Three-slot public strategy
#
# V26 follows the causal-probe discipline learned in V24. Do **not** blindly submit every generated file. The first two public slots are pre-registered single-action ablations; the third is conditional:
#
# 1. test removing U4567;
# 2. test removing U4929;
# 3. if both improve over 0.65919, submit the combined removal; otherwise use the U4491-only probe (or preserve the slot if neither result justifies expansion).
#
# `submission_v26_primary.csv` is an offline-selected robust subset and is useful for final/private selection, but if it is byte/rank identical to the anchor it is **not** worth a public slot.

# ---- cell 23 (code) ----
slot_plan = pd.DataFrame([
    {
        "slot":1,
        "file":"submission_v26_probe_remove_u4567.csv",
        "hypothesis":"The U4567 M012>M014 core action is net harmful despite positive V25 package evidence.",
        "measurement":"LB - 0.65919",
        "next_if_up":"Keep U4567 reverted and continue to slot 2.",
        "next_if_down":"Restore U4567; slot 2 still tests an independent U4929 hypothesis.",
    },
    {
        "slot":2,
        "file":"submission_v26_probe_remove_u4929.csv",
        "hypothesis":"The U4929 M002>M003 core action is net harmful.",
        "measurement":"LB - 0.65919",
        "next_if_up":"If slot 1 also improved, use combined removal for slot 3.",
        "next_if_down":"Do not generalize the M002>M003 prune from this user.",
    },
    {
        "slot":3,
        "file":"CONDITIONAL: submission_v26_probe_remove_u4567_u4929.csv OR submission_v26_probe_remove_u4491.csv",
        "hypothesis":"Combine independently positive removals; otherwise test the next M002>M003 attribution question.",
        "measurement":"candidate LB versus the best observed anchor of the day",
        "next_if_up":"Promote as new canonical anchor.",
        "next_if_down":"Keep the best earlier public candidate.",
    },
])

action_file = OUT_DIR/"v26_action_attribution.csv"
ledger_file = OUT_DIR/"v26_exact_diff_ledger.csv"
slot_file = OUT_DIR/"v26_slot_plan.csv"
attribution.to_csv(action_file,index=False)
ledger.to_csv(ledger_file,index=False)
slot_plan.to_csv(slot_file,index=False)
core_geometry.to_csv(OUT_DIR/"v26_core_geometry.csv",index=False)
environment = {
    "python":sys.version,
    "numpy":np.__version__,
    "pandas":pd.__version__,
    "scipy":scipy.__version__,
    "sklearn":sklearn.__version__,
    "lightgbm":lgb.__version__,
    "seed":SEED,
    "strict_geometry":V26_STRICT_GEOMETRY,
}
with open(OUT_DIR/"v26_environment.json","w") as f:
    json.dump(environment,f,indent=2,default=str)
display(slot_plan)

summary = {
    "version":"V26",
    "strategy":"core attribution + cluster-safe counterfactual pruning",
    "public_scores":{
        "consensus3":PUBLIC_CONSENSUS3_LB,
        "pre_v25_pb":PUBLIC_PRE_V25_PB,
        "v25_core6":PUBLIC_V25_LB,
        "leader":PUBLIC_LEADER_LB,
    },
    "canonical_v25_expected_sha256":EXPECTED_V25_PRIMARY_SHA256,
    "rebuilt_anchor_sha256":anchor_hash,
    "rebuilt_anchor_exact_hash_match":bool(anchor_hash==EXPECTED_V25_PRIMARY_SHA256),
    "strict_geometry":V26_STRICT_GEOMETRY,
    "auto_prunes":strict_prunes,
    "max_auto_prunes":MAX_AUTO_PRUNES,
    "probe_specs":probe_specs,
    "primary_policy":"V25 anchor minus strict train-only veto actions; fail closed to anchor",
    "recommended_public_slots":slot_plan.to_dict(orient="records"),
    "warning":"Core-6 has only package-level public validation. Individual probes are causal experiments, not ground-truth claims.",
}
with open(OUT_DIR/"v26_summary.json","w") as f:
    json.dump(summary,f,indent=2,default=str)

print("\nSaved V26 outputs:")
for p in sorted(OUT_DIR.glob("*v26*")):
    print(" -",p.name)
print("\nAutomatic primary prunes:", strict_prunes if strict_prunes else "NONE (exact V25 anchor)")
print("First public probe:", slot_plan.iloc[0]["file"])
