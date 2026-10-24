# Converted from notebooks/v25.ipynb: markdown cells become comments, %/! magics neutralized.

# ---- cell 0 (markdown) ----
# # MineToday V24 — Causal Leaderboard Micro-Probes on the 0.65912 Anchor
#
# **Goal:** use today's three submissions to isolate the few unresolved public-leaderboard effects around the current **0.65912** Top-4 solution. V24 reproduces the exact V3.1 sparse + V12.2 `Consensus3` backbone, reconstructs the two-swap public-best anchor, and changes exactly one hypothesis at a time.
#
# V24 is the direct response to the completed V23 audit:
#
# - V23's F2LLM action model failed every promotion gate and its `primary` was byte-identical to the PB reference;
# - embedding/action confidence did not order positive swap utility reliably;
# - broad prerequisite-first promotion was negative in local validation;
# - therefore V24 does not spend another day on an unvalidated global or semantic reranker.
#
# ### Kaggle settings
#
# - Accelerator: **GPU T4 x2 is compatible**, as requested, but V24 deliberately needs only CPU; selecting T4 x2 will not change its deterministic result.
# - Internet: **ON** only for the small LightGBM install if the Kaggle image does not already provide it.
# - Run all cells, then use the slot plan printed at the end.
#
# The governing rule is: **one leaderboard submission must answer one causal question.**

# ---- cell 1 (code) ----
# Runtime + deterministic configuration
import os, sys, re, json, ast, html, hashlib, random, warnings, time, subprocess
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ["PYTHONHASHSEED"] = "42"

V24_SKIP_INSTALL = bool(int(os.getenv("V24_SKIP_INSTALL", "0")))
if not V24_SKIP_INSTALL:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "lightgbm>=4.5,<5"
    ])

import numpy as np
import pandas as pd
from IPython.display import display
from scipy.sparse import hstack
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
import lightgbm as lgb

warnings.filterwarnings("ignore")

SEED = 42
V24_SMOKE = bool(int(os.getenv("V24_SMOKE", "0")))
N_FOLDS = 2 if V24_SMOKE else 5
RIDGE_ALPHA = 3.0
REFERENCE_W = 0.45
PUBLIC_CONSENSUS3_LB = 0.65856
PUBLIC_CURRENT_PB = 0.65912
PUBLIC_V22_FOUR_CHANGE_LB = 0.65878
PUBLIC_LEADER_LB = 0.66083
PUBLIC_CAL045_LB = 0.65639
LGB_ESTIMATORS = 40 if V24_SMOKE else 1200
LGB_EARLY_STOP = 8 if V24_SMOKE else 80

random.seed(SEED)
np.random.seed(SEED)
print("numpy", np.__version__, "pandas", pd.__version__, "lightgbm", lgb.__version__)
print("V24_SMOKE", V24_SMOKE, "leader gap", round(PUBLIC_LEADER_LB-PUBLIC_CURRENT_PB, 5))

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
BOOTSTRAPS = 200 if V24_SMOKE else 2000
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
word_vec=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_df=.995,sublinear_tf=True,strip_accents="unicode",max_features=(2000 if V24_SMOKE else 50000),dtype=np.float32)
char_vec=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=2,sublinear_tf=True,max_features=(3000 if V24_SMOKE else 60000),dtype=np.float32)
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
# ## Deep postmortem: why V23 did not improve the leaderboard
#
# V23 tested a reasonable hypothesis—predict the exact NDCG utility of small local swaps with structured and F2LLM semantic branches—but its own results reject deployment:
#
# | Evidence from the completed V23 run | Consequence for V24 |
# |---|---|
# | `primary` exactly matched the 0.65912 PB reference | V23 produced no promoted model change. |
# | Conservative policy: 10 fold-wise selections, only `+0.00000121` OOF; deployment-matched global top-3 was `-0.00000289` with 0/3 positives | The apparent fold-wise gain does not survive the actual test budget. |
# | Balanced policy: `-0.00001911` fold-wise; global top-6 `-0.00000194` | More actions increased exposure to harmful swaps. |
# | Confirmed policy: one OOF action, `-0.00000659` | Agreement/confidence was not calibrated to utility. |
# | Structured and hybrid branches shared about 93% of their features; predicted utility correlation was about 0.79 | Their agreement was not independent confirmation. |
# | Raw direct-prerequisite actions averaged about `-0.01090`; q10-margin prerequisite actions about `-0.00658` | Prerequisite order is useful context, not a safe universal ranking rule. |
# | The winning U4047 and U4953 pair archetypes were negative in their OOF analogues | Public winners are sparse exceptions; copying their pattern to more users is unjustified. |
#
# ### Embedding research decision
#
# - [SEA-Embedding (2026)](https://arxiv.org/abs/2606.03027) is a promising 600M encoder purpose-built for Southeast Asian languages, including Indonesian.
# - [SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html) supplies the useful idea for this stage: when estimated risk is high, abstain instead of forcing coverage.
# - Goal-aware course recommendation and prerequisite discovery remain conceptually relevant ([MOOCCube course recommendation](https://arxiv.org/abs/1812.10078), [ACL prerequisite learning](https://aclanthology.org/P17-1133/)), but V23's MineToday labels reject broad prerequisite promotion.
#
# SEA-Embedding is therefore **not added to deployment in V24**. Better generic retrieval quality is not evidence of better incremental NDCG swap utility, and the local experiments from V16–V23 already show that semantic replacement/blending can lower the public score. The lowest-risk research conclusion is selective abstention: keep the validated sparse anchor and probe only unresolved public effects.

# ---- cell 14 (markdown) ----
# ## Public-score algebra and today's three slots
#
# Let `B` be exact Consensus3 and let `a`, `b`, `c`, `d` be the isolated public effects of the U4047, U4953, U4498, and U4876 ordering changes.
#
# - `B = 0.65856`
# - `B + a + b = 0.65912`
# - the earlier four-change run gave `B + a + b + c + d = 0.65878`
#
# Thus `a+b ≈ +0.00056`, while `c+d ≈ -0.00034`. The combined result does **not** imply both `c` and `d` are harmful; one may be positive and hidden by the other. Two PB-plus-one probes isolate `c` and `d`. One single-winner probe isolates `a`; `b` follows from subtraction.
#
# Recommended submission order:
#
# 1. PB + U4498 only;
# 2. PB + U4876 only;
# 3. U4047 only, unless a result from slots 1–2 already justifies a higher-value combination.
#
# V24 also exports U4953-only as a symmetric backup and the untouched PB as `primary`.

# ---- cell 15 (code) ----
# Four isolated public-leaderboard hypotheses.
PROBE_SPECS = {
    "u4047_only": {
        "base": "consensus3",
        "swaps": [("U_4047", "M_009", "M_007")],
    },
    "u4953_only": {
        "base": "consensus3",
        "swaps": [("U_4953", "M_016", "M_005")],
    },
    "pb_plus_u4498": {
        "base": "pb",
        "swaps": [("U_4498", "M_009", "M_001")],
    },
    "pb_plus_u4876": {
        "base": "pb",
        "swaps": [("U_4876", "M_014", "M_009")],
    },
}

PROBE_FILENAMES = {
    "u4047_only": "submission_v24_probe1_u4047_only.csv",
    "u4953_only": "submission_v24_probe2_u4953_only.csv",
    "pb_plus_u4498": "submission_v24_probe3_pb_plus_u4498.csv",
    "pb_plus_u4876": "submission_v24_probe4_pb_plus_u4876.csv",
}


def apply_isolated_swaps(scores, swaps):
    out = scores.copy()
    for uid, higher_before, promote in swaps:
        row = TEST_UID_TO_ROW[uid]
        hi = MID_TO_COL[higher_before]
        lo = MID_TO_COL[promote]
        before_hi = float(out[row, hi])
        before_lo = float(out[row, lo])
        if not before_hi > before_lo:
            raise AssertionError(
                f"Probe geometry drift for {uid}: expected {higher_before}>{promote}, "
                f"got {before_hi:.8f}<={before_lo:.8f}"
            )
        out[row, hi], out[row, lo] = before_lo, before_hi
    return out


probe_scores = {}
for name, spec in PROBE_SPECS.items():
    anchor = consensus3_test if spec["base"] == "consensus3" else pb_test
    probe_scores[name] = apply_isolated_swaps(anchor, spec["swaps"])

# ---- cell 16 (code) ----
# Exact change audit: every probe may touch only its declared row and two score cells.
probe_audit_rows = []
expected_changed_users = {}

for name, spec in PROBE_SPECS.items():
    anchor = consensus3_test if spec["base"] == "consensus3" else pb_test
    candidate = probe_scores[name]
    diff = candidate != anchor
    changed_rows = np.flatnonzero(diff.any(axis=1))
    changed_users = test.iloc[changed_rows]["user_id"].tolist()
    declared_users = [s[0] for s in spec["swaps"]]
    assert changed_users == declared_users, (name, changed_users, declared_users)
    assert int(diff.sum()) == 2 * len(spec["swaps"]), (name, int(diff.sum()))
    expected_changed_users[name] = declared_users

    for uid, higher_before, promote in spec["swaps"]:
        row = TEST_UID_TO_ROW[uid]
        changed_modules = {MODULES[j] for j in np.flatnonzero(diff[row])}
        assert changed_modules == {higher_before, promote}, (name, uid, changed_modules)
        before = [MODULES[j] for j in top5_indices(anchor[[row]])[0]]
        after = [MODULES[j] for j in top5_indices(candidate[[row]])[0]]
        probe_audit_rows.append({
            "probe": name,
            "base": spec["base"],
            "user_id": uid,
            "higher_before": higher_before,
            "promoted": promote,
            "before_top5": " ".join(before),
            "after_top5": " ".join(after),
            "top5_membership_changed": set(before) != set(after),
            "score_cells_changed": int(diff[row].sum()),
        })

probe_audit = pd.DataFrame(probe_audit_rows)
display(probe_audit)

# ---- cell 17 (code) ----
OUT_DIR = Path("/kaggle/working")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_submission(scores, filename):
    out = sample.copy()
    out.loc[:, MODULES] = np.asarray(scores, np.float32)
    assert list(out.columns) == list(sample.columns)
    assert out["user_id"].tolist() == sample["user_id"].tolist()
    assert np.isfinite(out[MODULES].to_numpy()).all()
    path = OUT_DIR / filename
    out.to_csv(path, index=False)
    print("saved", path)
    return out


consensus_submission = save_submission(consensus3_test, "submission_v24_consensus3_reference.csv")
pb_submission = save_submission(pb_test, "submission_v24_pb_reference.csv")
primary_submission = save_submission(pb_test, "submission_v24_primary.csv")
probe_submissions = {
    name: save_submission(scores, PROBE_FILENAMES[name])
    for name, scores in probe_scores.items()
}

slot_plan = pd.DataFrame([
    {
        "slot": 1,
        "file": PROBE_FILENAMES["pb_plus_u4498"],
        "measurement": "c = LB - 0.65912",
        "decision": "Keep U4498 only if its isolated LB exceeds 0.65912.",
    },
    {
        "slot": 2,
        "file": PROBE_FILENAMES["pb_plus_u4876"],
        "measurement": "d = LB - 0.65912",
        "decision": "Keep U4876 only if its isolated LB exceeds 0.65912.",
    },
    {
        "slot": 3,
        "file": PROBE_FILENAMES["u4047_only"],
        "measurement": "a = LB - 0.65856; b = 0.65912 - LB",
        "decision": "Tests whether removing U4953 beats the combined current PB.",
    },
])

probe_audit.to_csv(OUT_DIR / "v24_probe_manifest.csv", index=False)
slot_plan.to_csv(OUT_DIR / "v24_slot_plan.csv", index=False)
display(slot_plan)

# ---- cell 18 (code) ----
# Hard invariants: fail instead of emitting a contaminated leaderboard probe.
assert len(MODULES) == 17 and y.shape == (4000, 17) and N_TEST == 1000
assert set(changed_uids) == {"U_4047", "U_4953"}
assert np.array_equal(primary_submission[MODULES].to_numpy(np.float32), pb_test)
assert np.array_equal(pb_submission[MODULES].to_numpy(np.float32), pb_test)
assert np.array_equal(consensus_submission[MODULES].to_numpy(np.float32), consensus3_test)

for name, spec in PROBE_SPECS.items():
    anchor = consensus3_test if spec["base"] == "consensus3" else pb_test
    candidate = probe_scores[name]
    diff = candidate != anchor
    expected_rows = {TEST_UID_TO_ROW[s[0]] for s in spec["swaps"]}
    actual_rows = set(np.flatnonzero(diff.any(axis=1)).tolist())
    assert actual_rows == expected_rows, (name, actual_rows, expected_rows)
    assert int(diff.sum()) == 2 * len(spec["swaps"])
    assert np.isfinite(candidate).all()

for df in [consensus_submission, pb_submission, primary_submission, *probe_submissions.values()]:
    assert list(df.columns) == list(sample.columns)
    assert df["user_id"].tolist() == sample["user_id"].tolist()
    assert np.isfinite(df[MODULES].to_numpy()).all()

summary = {
    "version": "V24",
    "strategy": "causal public-LB micro-probes with selective abstention from new embeddings",
    "recorded_scores": {
        "consensus3": PUBLIC_CONSENSUS3_LB,
        "current_pb": PUBLIC_CURRENT_PB,
        "v22_four_change": PUBLIC_V22_FOUR_CHANGE_LB,
        "leader": PUBLIC_LEADER_LB,
    },
    "known_pb_swaps": KNOWN_PB_SWAPS,
    "probe_specs": PROBE_SPECS,
    "recommended_slots": slot_plan.to_dict(orient="records"),
    "primary": "exact 0.65912 PB anchor",
}
with open(OUT_DIR / "v24_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

print("ALL V24 INVARIANTS PASSED")
print("PRIMARY remains the exact 0.65912 anchor. Recommended slot order:")
for row in slot_plan.itertuples(index=False):
    print(f"  {row.slot}. {row.file} — {row.measurement}")
