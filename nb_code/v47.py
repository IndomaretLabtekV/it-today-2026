# Converted from notebooks/v47.ipynb: markdown cells become comments, %/! magics neutralized.

# ---- cell 0 (markdown) ----
# # MineToday V47 — Frozen-Champion Dual-Encoder Top1 Utility Gate
#
# **Scientific hypothesis:** among target-blind S0 users, frozen `multilingual-e5-large-instruct` and `nomic-embed-text-v2-moe` contain complementary information about the exact NDCG@5 utility of promoting an existing champion rank-2/3/4 module to rank 1.
#
# This notebook preserves the exact public `0.66028` champion and permits **Top1 score-value permutations only**. It does not modify Top4/Top5 membership, does not rerank lower boundaries, does not use generator-state voting, and does not allow structural controls to authorize actions.
#
# The V47 evaluation authority is prospective: template-grouped `ACTION_TRAIN / DESIGN / LOCK`, nested action-utility cross-fitting inside ACTION_TRAIN, a frozen risk-coverage policy before DESIGN, and a one-shot LOCK. If any gate fails, the notebook fails closed exactly to the `0.66028` champion.
#
# **Runtime target:** Kaggle **T4x2**, Internet ON unless both pinned encoder snapshots are attached as Kaggle inputs.

# ---- cell 1 (code) ----

# Runtime + exact frozen lineage configuration
import os, sys, re, json, ast, html, hashlib, random, warnings, time, subprocess, gc, math, platform
from pathlib import Path
from collections import defaultdict, Counter

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ["PYTHONHASHSEED"] = "42"

V44_SKIP_INSTALL = bool(int(os.getenv("V44_SKIP_INSTALL", "0")))
try:
    import lightgbm as lgb
except Exception:
    if V44_SKIP_INSTALL:
        raise
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "lightgbm==4.6.0"])
    import lightgbm as lgb

import numpy as np
import pandas as pd
import scipy
import sklearn
from IPython.display import display
from scipy.sparse import hstack
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")

SEED = 42
V44_SMOKE = bool(int(os.getenv("V44_SMOKE", "0")))
# Compatibility aliases consumed by the frozen V3.1 -> V25 -> V30/V31 reconstruction cells.
V43_SMOKE = V44_SMOKE
V421_SMOKE = V44_SMOKE
V395_SMOKE = V44_SMOKE
V31_SMOKE = V44_SMOKE
V30_SMOKE = V44_SMOKE
V29_SMOKE = V44_SMOKE
V27_SMOKE = V44_SMOKE
V395_STRICT_GEOMETRY = False if V44_SMOKE else bool(int(os.getenv("V44_STRICT_GEOMETRY", "1")))
V30_STRICT_GEOMETRY = V395_STRICT_GEOMETRY
V29_STRICT_GEOMETRY = V395_STRICT_GEOMETRY
V27_STRICT_GEOMETRY = V395_STRICT_GEOMETRY

N_FOLDS = 2 if V44_SMOKE else 5
ANCHOR_INNER_FOLDS = 2 if V44_SMOKE else 4
INTENT_INNER_FOLDS = 2 if V44_SMOKE else 4
POLICY_INNER_FOLDS = 2 if V44_SMOKE else 4

RIDGE_ALPHA = 3.0
REFERENCE_W = 0.45
PUBLIC_CAL045_LB = 0.65639
PUBLIC_CONSENSUS3_LB = 0.65856
PUBLIC_V25_LB = 0.65919
PUBLIC_V30_LB = 0.65963
PUBLIC_V31_DIAGNOSTIC_LB = 0.65949
PUBLIC_HYBRID34_LB = 0.66006
PUBLIC_CURRENT_ANCHOR_LB = 0.66020
PUBLIC_V44_ANCHOR_LB = 0.66028
PUBLIC_TOP4_LB = 0.66080
PUBLIC_LEADER_LB = 0.66142

LGB_ESTIMATORS = 12 if V44_SMOKE else 1200
LGB_EARLY_STOP = 3 if V44_SMOKE else 80
RANKER_ESTIMATORS = 16 if V44_SMOKE else 360
RANKER_LEARNING_RATE = 0.04 if V44_SMOKE else 0.025
ALPHA = 0.16
INTENT_PARENT_STRENGTH = 40.0
INTENT_CHILD_STRENGTH = 30.0
BOOTSTRAPS = 30 if V44_SMOKE else 1200
GEOMETRY_ATOL = 2.5e-4

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
V44_ANCHOR_TOP5_SHA256 = "7c0d29a7309b7283a043b7d37ca5556ba7c0ca2270b0137a6e1e0878a69a5fe7"
LB_ROUNDING_TOL = 3e-5

def _seed_from_text(s):
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:8], 16) % (2**31-1)

random.seed(SEED); np.random.seed(SEED)
print("numpy", np.__version__, "pandas", pd.__version__, "sklearn", sklearn.__version__, "lightgbm", lgb.__version__)
print("V44_SMOKE", V44_SMOKE, "frozen V30 alpha", ALPHA)
print("production champion", PUBLIC_V44_ANCHOR_LB, "leader gap", round(PUBLIC_LEADER_LB-PUBLIC_V44_ANCHOR_LB,5))
try:
    print("GPU runtime (V44 is CPU-first, but Kaggle T4x2 is compatible):\n" + subprocess.check_output(["nvidia-smi", "-L"], text=True).strip())
except Exception:
    print("nvidia-smi unavailable; V44 does not require GPU.")

assert GEOMETRY_ATOL == 2.5e-4

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
ALL_INTENT=np.concatenate([TRAIN_INTENT,TEST_INTENT]).astype(object)

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

# ---- cell 22 (code) ----

# Exact production champion = current 0.66020 + public-confirmed S3 B34 + B45.
def build_s3_boundary_ledger_v44(v30, v31, info):
    t30=top5_indices(v30); t31=top5_indices(v31); rows=[]
    for r in np.flatnonzero(info["order_only"]):
        boundary,promoted,demoted,pos_from,pos_to=classify_single_adjacent_order_swap(t30[r],t31[r])
        rows.append({"row_local":int(r),"user_id":str(test.iloc[r]["user_id"]),"boundary":boundary,
                     "promoted_module":MODULES[promoted],"demoted_module":MODULES[demoted],
                     "v30_rank_from":int(pos_from),"v31_rank_to":int(pos_to)})
    return pd.DataFrame(rows).sort_values(["boundary","row_local"]).reset_index(drop=True)

V44_S3_BOUNDARY_LEDGER=build_s3_boundary_ledger_v44(v30_test,v31_test,current_info)
assert V44_S3_BOUNDARY_LEDGER.boundary.value_counts().sort_index().to_dict()==EXPECTED_BOUNDARY_COUNTS
V44_ANCHOR_TEST=current_anchor_test.copy()
rr=V44_S3_BOUNDARY_LEDGER.loc[V44_S3_BOUNDARY_LEDGER.boundary.isin(["B34","B45"]),"row_local"].to_numpy(int)
V44_ANCHOR_TEST[rr]=v31_test[rr]
if not V44_SMOKE:
    assert top5_sha256(V44_ANCHOR_TEST)==V44_ANCHOR_TOP5_SHA256
print("V44 exact anchor hash", top5_sha256(V44_ANCHOR_TEST), "B34+B45 rows", len(rr))

# ---- cell 23 (markdown) ----
# ## V47 preregistration
#
# Everything below is fixed before V47 model outcomes are inspected. Historical CAL/LOCK/public results are treated as development history only.
#
# V47 changes one scientific factor: **representation competence for exact S0 Top1 action utility**. E5 and Nomic are independent experts. Existing structural features are evaluated only as a diagnostic control. No learned router, generator veto, membership edit, B23/B34/B45 edit, or post-DESIGN subgroup rule is allowed.

# ---- cell 24 (code) ----
# V47 fixed configuration and prospective evaluation authority.
import importlib, importlib.metadata
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

V47_VERSION = "V47 Frozen-Champion Dual-Encoder Top1 Utility Gate"
V47_SMOKE = bool(int(os.getenv("V47_SMOKE", "0")))
V47_SKIP_DEEP = bool(int(os.getenv("V47_SKIP_DEEP", "0")))
V47_OUT = OUT_DIR / "v47"
V47_OUT.mkdir(parents=True, exist_ok=True)

# Prospective split. Exact sizes are targets because latest-template groups are indivisible.
V47_SPLIT_SEED = 2026090470
V47_STRUCT_SEED = 2026090471
V47_HEAD_SEED = 2026090472
V47_BOOT_SEED = 2026090473
TARGET_ACTION_TRAIN = 600 if V47_SMOKE else 2600
TARGET_DESIGN = 200 if V47_SMOKE else 800
TARGET_LOCK = 200 if V47_SMOKE else 600
SPLIT_SEARCH_TRIALS = 16 if V47_SMOKE else 256
HEAD_OUTER_FOLDS = 2 if V47_SMOKE else 5
STRUCT_TRAIN_CROSSFIT_FOLDS = 2
HEAD_INNER_FOLDS = 2 if V47_SMOKE else 4
EVAL_FOLDS = 2 if V47_SMOKE else 5
EVAL_BLOCKS = 2 if V47_SMOKE else 5
BOOTSTRAPS_V47 = 100 if V47_SMOKE else 2000

# Low-capacity action heads. Alpha is chosen only inside ACTION_TRAIN.
RIDGE_ALPHA_GRID = [3.0, 10.0, 30.0]
COVERAGE_GRID = [0.01] if V47_SMOKE else [0.0025, 0.0050, 0.0100, 0.0150]
WEIGHT_CLIP_LOW, WEIGHT_CLIP_HIGH = 0.25, 4.0
LOWER_ESTIMATE = "min_across_fold_models"

# Preregistered scientific gates from the research reset.
ORACLE_MIN_DELTA = -1.0 if V47_SMOKE else 0.00150
DESIGN_MIN_DELTA = -1.0 if V47_SMOKE else 0.00080
DESIGN_MIN_FOLD_WINS = 0 if V47_SMOKE else 4
BLOCK_CATASTROPHIC_FLOOR = -1.0 if V47_SMOKE else -0.00020
BENEFIT_HARM_RATIO = 0.0 if V47_SMOKE else 2.0
POOLED_DESIGN_LOCK_MIN_DELTA = -1.0 if V47_SMOKE else 0.00080
TEST_MIN_ACTIONS = 0 if V47_SMOKE else 3
TEST_MAX_ACTIONS = 999 if V47_SMOKE else 15

PUBLIC_V47_ANCHOR_LB = 0.66028
PUBLIC_V47_TOP5_CUTOFF = 0.66076
PUBLIC_V47_LEADER = 0.66142
EXPECTED_V47_ANCHOR_HASH = "7c0d29a7309b7283a043b7d37ca5556ba7c0ca2270b0137a6e1e0878a69a5fe7"

# Explicit historical public-probe rows are protected even if they happen to fall in S0.
PUBLIC_PROBE_TEST_USERS = {
    "U_4030", "U_4047", "U_4074", "U_4144", "U_4176", "U_4491", "U_4498",
    "U_4567", "U_4650", "U_4731", "U_4836", "U_4876", "U_4929", "U_4953",
}

# Frozen open-weight encoders and revisions.
E5_MODEL_ID = "intfloat/multilingual-e5-large-instruct"
E5_REVISION = "b447574844fbdae893c9a720e376d77ff5cb123a"
NOMIC_MODEL_ID = "nomic-ai/nomic-embed-text-v2-moe"
NOMIC_REVISION = "f0bcc8894f384250588bd1ba946c080e4339dedf"
NOMIC_CODE_REVISION = "7710840340a098cfb869c4f65e87cf2b1b70caca"
ENCODER_MAX_LEN = 512
ENCODER_BATCH = 4 if V47_SMOKE else 16
E5_TASK = (
    "Given a learner's recent learning conversation and assessment profile, "
    "retrieve the Intelligo course module that should be learned next."
)

assert top5_sha256(V44_ANCHOR_TEST) == EXPECTED_V47_ANCHOR_HASH
print("V47 frozen champion hash:", top5_sha256(V44_ANCHOR_TEST))
print("public anchor", PUBLIC_V47_ANCHOR_LB,
      "gap to Top-5", round(PUBLIC_V47_TOP5_CUTOFF-PUBLIC_V47_ANCHOR_LB,5),
      "gap to leader", round(PUBLIC_V47_LEADER-PUBLIC_V47_ANCHOR_LB,5))
print("V47_SMOKE", V47_SMOKE, "V47_SKIP_DEEP", V47_SKIP_DEEP)

# ---- cell 25 (code) ----
# Target-blind latest-template grouped three-way split with target stratification used only to choose a balanced manifest.
pos_count = (y > 0).sum(1).astype(int)
joint = np.asarray([f"{np.argmax(y[i]):02d}_{pos_count[i]}" for i in range(N_TRAIN)], dtype=object)
all_idx = np.arange(N_TRAIN, dtype=int)
FULL_STRATA = pd.Series(joint).value_counts(normalize=True)


def _strata_l1(rows):
    q = pd.Series(joint[np.asarray(rows,int)]).value_counts(normalize=True)
    keys = FULL_STRATA.index.union(q.index)
    return float(np.abs(FULL_STRATA.reindex(keys,fill_value=0)-q.reindex(keys,fill_value=0)).sum())


def _group_holdout(base_idx, target_size, seed, trials):
    base_idx=np.asarray(base_idx,int)
    groups=np.asarray(TRAIN_GROUP[base_idx],dtype=object)
    base_frac=float(target_size/max(1,len(base_idx)))
    factors=[0.90,0.95,1.00,1.05,1.10]
    best=None
    for t in range(int(trials)):
        frac=float(np.clip(base_frac*factors[t%len(factors)],0.03,0.70))
        gss=GroupShuffleSplit(n_splits=1,test_size=frac,random_state=int(seed+104729*t))
        trloc,holoc=next(gss.split(np.zeros(len(base_idx)),groups=groups))
        ho=base_idx[np.asarray(holoc,int)]; tr=base_idx[np.asarray(trloc,int)]
        size_pen=abs(len(ho)-target_size)/max(1,target_size)
        strata_pen=_strata_l1(ho)
        score=4.0*size_pen+strata_pen
        cand=(score,size_pen,strata_pen,tr,ho)
        if best is None or cand[0] < best[0]: best=cand
    return best

lock_best=_group_holdout(all_idx,TARGET_LOCK,V47_SPLIT_SEED,SPLIT_SEARCH_TRIALS)
REM=np.sort(lock_best[3]); LOCK_IDX=np.sort(lock_best[4])
design_best=_group_holdout(REM,TARGET_DESIGN,V47_SPLIT_SEED+1,SPLIT_SEARCH_TRIALS)
ACTION_TRAIN_IDX=np.sort(design_best[3]); DESIGN_IDX=np.sort(design_best[4])

assert len(set(ACTION_TRAIN_IDX)&set(DESIGN_IDX))==0
assert len(set(ACTION_TRAIN_IDX)&set(LOCK_IDX))==0
assert len(set(DESIGN_IDX)&set(LOCK_IDX))==0
assert len(ACTION_TRAIN_IDX)+len(DESIGN_IDX)+len(LOCK_IDX)==N_TRAIN

# Exact-template groups are prospectively isolated across all three partitions.
ga=set(map(str,TRAIN_GROUP[ACTION_TRAIN_IDX])); gd=set(map(str,TRAIN_GROUP[DESIGN_IDX])); gl=set(map(str,TRAIN_GROUP[LOCK_IDX]))
assert not (ga&gd) and not (ga&gl) and not (gd&gl)

# Fail if indivisible template groups make the target sizes pathologically different.
for name,rows,target in [("ACTION_TRAIN",ACTION_TRAIN_IDX,TARGET_ACTION_TRAIN),("DESIGN",DESIGN_IDX,TARGET_DESIGN),("LOCK",LOCK_IDX,TARGET_LOCK)]:
    if not V47_SMOKE:
        assert abs(len(rows)-target) <= max(200,int(0.25*target)), f"{name} group split too far from target size"

V47_SPLIT_MANIFEST=pd.DataFrame({
    "user_id":train.user_id.astype(str),
    "row":all_idx,
    "template_group":np.asarray(TRAIN_GROUP,dtype=object),
    "stratum":joint,
    "split":np.where(np.isin(all_idx,ACTION_TRAIN_IDX),"ACTION_TRAIN",np.where(np.isin(all_idx,DESIGN_IDX),"DESIGN","LOCK")),
})
V47_SPLIT_AUDIT=pd.DataFrame([
    {"split":"ACTION_TRAIN","n":len(ACTION_TRAIN_IDX),"template_groups":len(ga),"strata_l1":_strata_l1(ACTION_TRAIN_IDX)},
    {"split":"DESIGN","n":len(DESIGN_IDX),"template_groups":len(gd),"strata_l1":_strata_l1(DESIGN_IDX)},
    {"split":"LOCK","n":len(LOCK_IDX),"template_groups":len(gl),"strata_l1":_strata_l1(LOCK_IDX)},
])
print(V47_SPLIT_AUDIT.to_string(index=False))

# ---- cell 26 (markdown) ----
# ## Prospective frozen-champion analogue and oracle
#
# The exact test champion remains immutable. For labeled evaluation users, the V3.1 → Consensus3 → V30/V31 state machine is rebuilt using only the designated training side. The S3 B34+B45 rule is applied algorithmically. **Only state S0 is eligible for V47.**
#
# DESIGN and LOCK are both predicted from an ACTION_TRAIN-only structural fit. Their labels are not used by this stage. Oracle analysis only asks whether the true Top1 is already inside the immutable Top4; it does not train or select the V47 policy.

# ---- cell 27 (code) ----
# Deployment-like V47 validation surface inherited exactly from the V44 champion construction.
def build_v47_val_surface(train_rows,val_rows,seed):
    train_rows=np.asarray(train_rows,int); val_rows=np.asarray(val_rows,int)
    anc=build_anchor_split_full(train_rows,val_rows,int(seed))
    intent_tr=crossfit_intentstate_features(
        y[train_rows],ORDINAL_GRADES[train_rows],TRAIN_INTENT[train_rows],
        MODULE_READY_BAND[train_rows],TRAIN_GROUP[train_rows],int(seed)+71)
    bundle=fit_intentstate_bundle(
        y[train_rows],ORDINAL_GRADES[train_rows],TRAIN_INTENT[train_rows],MODULE_READY_BAND[train_rows])
    intent_va=apply_intentstate_bundle(bundle,ALL_INTENT[val_rows],MODULE_READY_BAND[val_rows])
    Xtr,_,bounds=build_long_matrix_v31(train_rows,anc["cons_train"],anc["base_train"],anc["meta_train"],intent_tr)
    Xva,_,bounds2=build_long_matrix_v31(val_rows,anc["cons_val"],anc["base_val"],anc["meta_val"],intent_va)
    assert bounds==bounds2
    _,raw30=fit_ranker_predict(Xtr[:,:bounds["base"]],Xva[:,:bounds["base"]],ORDINAL_GRADES[train_rows],len(train_rows),int(seed)+3301)
    _,raw31=fit_ranker_predict(Xtr[:,:bounds["intentstate"]],Xva[:,:bounds["intentstate"]],ORDINAL_GRADES[train_rows],len(train_rows),int(seed)+4301)
    pre=anc["cons_val"].astype(np.float32)
    v30=blend_residual(pre,raw30,ALPHA); v31=blend_residual(pre,raw31,ALPHA)
    state=state_codes(pre,v30,v31)
    current,info=current066020_surface(v30,v31,state)
    t30=top5_indices(v30); t31=top5_indices(v31)
    for r in np.flatnonzero(info["order_only"]):
        try: boundary,*_=classify_single_adjacent_order_swap(t30[r],t31[r])
        except AssertionError: continue
        if boundary in {"B34","B45"}: current[r]=v31[r]
    out={"pre":pre,"v30":v30.astype(np.float32),"v31":v31.astype(np.float32),
         "state":state.astype(np.int8),"anchor":current.astype(np.float32)}
    del Xtr,Xva,intent_tr,intent_va,bundle,anc,raw30,raw31
    gc.collect()
    return out


def crossfit_v47_surface(rows,n_splits,seed):
    rows=np.asarray(rows,int); n=len(rows)
    out={"pre":np.zeros((n,N_MODULES),np.float32),"v30":np.zeros((n,N_MODULES),np.float32),
         "v31":np.zeros((n,N_MODULES),np.float32),"anchor":np.zeros((n,N_MODULES),np.float32),
         "state":np.full(n,-1,np.int8)}
    pos={int(g):i for i,g in enumerate(rows)}
    folds=make_group_folds_global(rows,n_splits,int(seed))
    for f,(tr,va) in enumerate(folds):
        print(f"V47 nested structural crossfit {f+1}/{len(folds)}: train={len(tr)} val={len(va)}")
        s=build_v47_val_surface(tr,va,int(seed)+10007*f)
        loc=np.asarray([pos[int(g)] for g in va],int)
        for k in out: out[k][loc]=s[k]
        del s; gc.collect()
    assert (out["state"]>=0).all()
    return out


def apply_top1_permutation_v47(scores,row,challenger):
    before=np.asarray(scores[row],np.float32).copy()
    order=np.argsort(-before,kind="mergesort"); top4=order[:4].tolist(); challenger=int(challenger)
    if challenger not in top4: raise AssertionError("V47 may promote only an existing anchor Top4 module")
    if top4[0]==challenger: return
    vals=before[top4].copy(); new_order=[challenger]+[m for m in top4 if m!=challenger]
    scores[row,new_order]=vals


def geometry_audit_v47(base,cand):
    base=np.asarray(base,np.float32); cand=np.asarray(cand,np.float32)
    ob=np.argsort(-base,axis=1,kind="mergesort"); oc=np.argsort(-cand,axis=1,kind="mergesort")
    changed=np.any(ob!=oc,axis=1)
    top4_same=np.asarray([set(a[:4])==set(b[:4]) for a,b in zip(ob,oc)],bool)
    top5_same=np.asarray([set(a[:5])==set(b[:5]) for a,b in zip(ob,oc)],bool)
    tail_same=np.asarray([np.array_equal(a[4:],b[4:]) for a,b in zip(ob,oc)],bool)
    multiset_ok=True
    for r in np.flatnonzero(changed):
        if not np.array_equal(np.sort(base[r,ob[r,:4]]),np.sort(cand[r,oc[r,:4]])):
            multiset_ok=False; break
    return {"changed_users":int(changed.sum()),"top1_changes":int((ob[:,0]!=oc[:,0]).sum()),
            "top4_membership_changes":int((~top4_same).sum()),"top5_membership_changes":int((~top5_same).sum()),
            "rank5_to_17_order_changes":int((~tail_same).sum()),"top4_score_multiset_ok":bool(multiset_ok)}


def top1_acc_v47(ytrue,s): return float(np.mean(np.argmax(ytrue,axis=1)==np.argmax(s,axis=1)))
def relevant_at5_v47(ytrue,s):
    t=top5_indices(s); return float(np.mean((np.take_along_axis(ytrue,t,axis=1)>0).sum(1)))
def metric_v47(ytrue,base,cand):
    return {"base_ndcg":ndcg5(ytrue,base),"candidate_ndcg":ndcg5(ytrue,cand),"delta":ndcg5(ytrue,cand)-ndcg5(ytrue,base),
            "base_top1":top1_acc_v47(ytrue,base),"candidate_top1":top1_acc_v47(ytrue,cand),
            "top1_delta":top1_acc_v47(ytrue,cand)-top1_acc_v47(ytrue,base),
            "base_rel5":relevant_at5_v47(ytrue,base),"candidate_rel5":relevant_at5_v47(ytrue,cand),
            "rel5_delta":relevant_at5_v47(ytrue,cand)-relevant_at5_v47(ytrue,base)}


def cluster_bootstrap_mean(delta,groups,seed,B=BOOTSTRAPS_V47):
    d=np.asarray(delta,float); g=np.asarray(groups,dtype=object); ug=np.unique(g)
    by={x:np.flatnonzero(g==x) for x in ug}; rng=np.random.default_rng(int(seed)); vals=np.empty(int(B),float)
    for b in range(int(B)):
        draw=rng.choice(ug,size=len(ug),replace=True); idx=np.concatenate([by[x] for x in draw])
        vals[b]=float(d[idx].mean()) if len(idx) else 0.0
    return float(np.quantile(vals,.025)),float(np.quantile(vals,.975))

# ACTION_TRAIN-only structural fit predicts DESIGN and LOCK.
eval_global=np.concatenate([DESIGN_IDX,LOCK_IDX])
print("Building prospective DESIGN/LOCK champion analogue from ACTION_TRAIN only...")
eval_surf=build_v47_val_surface(ACTION_TRAIN_IDX,eval_global,V47_STRUCT_SEED+90001)
nd=len(DESIGN_IDX)
DESIGN_SURF={k:np.asarray(v[:nd]).copy() for k,v in eval_surf.items()}
LOCK_SURF={k:np.asarray(v[nd:]).copy() for k,v in eval_surf.items()}
del eval_surf; gc.collect()
DESIGN_ELIG=(DESIGN_SURF["state"]==0); LOCK_ELIG=(LOCK_SURF["state"]==0)

DESIGN_BASE=DESIGN_SURF["anchor"].copy()
true_top1=np.argmax(y[DESIGN_IDX],axis=1); base_top4=np.argsort(-DESIGN_BASE,axis=1,kind="mergesort")[:,:4]; base_top1=base_top4[:,0]
recoverable=np.asarray([bool(DESIGN_ELIG[r] and base_top1[r]!=true_top1[r] and true_top1[r] in set(base_top4[r])) for r in range(len(DESIGN_IDX))],bool)
ORACLE=DESIGN_BASE.copy()
for r in np.flatnonzero(recoverable): apply_top1_permutation_v47(ORACLE,int(r),int(true_top1[r]))
oracle_delta=float(ndcg5(y[DESIGN_IDX],ORACLE)-ndcg5(y[DESIGN_IDX],DESIGN_BASE))
wrong=DESIGN_ELIG & (base_top1!=true_top1)
V47_ORACLE=pd.DataFrame([{"design_anchor_ndcg":ndcg5(y[DESIGN_IDX],DESIGN_BASE),"eligible_s0_users":int(DESIGN_ELIG.sum()),
    "eligible_wrong_top1":int(wrong.sum()),"recoverable_top1_errors":int(recoverable.sum()),
    "true_top1_in_top4_given_wrong":float(recoverable.sum()/max(1,wrong.sum())),"top1_within_top4_oracle_delta":oracle_delta}])
ORACLE_GATE=bool(top5_sha256(V44_ANCHOR_TEST)==EXPECTED_V47_ANCHOR_HASH and oracle_delta>=ORACLE_MIN_DELTA)
print(V47_ORACLE.to_string(index=False)); print("ORACLE_GATE",ORACLE_GATE)

# ---- cell 28 (markdown) ----
# ## Frozen multilingual semantic experts
#
# V47 uses exactly one deterministic user text view for both encoders: the last three non-administrative learning messages in chronological order plus the normalized ten-skill assessment profile. Module documents contain only organizer-provided catalog name, syllabus, and prerequisite text.
#
# The encoders are frozen. E5 follows its official instruction format and mean pooling. Nomic follows its official `search_query:` / `search_document:` prefixes and mean pooling. Both use 512-token truncation. Test embeddings are **not generated unless DESIGN and LOCK pass**.

# ---- cell 29 (code) ----
# Pin the V47 deep-inference stack only when the oracle says the action space is worth testing.
DEEP_ENV_READY=False; DEEP_ENV_ERROR=""
if ORACLE_GATE and not V47_SKIP_DEEP:
    try:
        required={
            "transformers":"4.44.2",
            "sentence-transformers":"3.3.0",
            "sentencepiece":"0.2.0",
            "tiktoken":"0.8.0",
            "einops":"0.8.0",
        }
        need=[]
        for pkg,ver in required.items():
            try: cur=importlib.metadata.version(pkg)
            except importlib.metadata.PackageNotFoundError: cur=None
            if cur!=ver: need.append(f"{pkg}=={ver}")
        if need:
            print("Installing pinned V47 deep dependencies:",need)
            subprocess.check_call([sys.executable,"-m","pip","install","-q","--upgrade"]+need)
            importlib.invalidate_caches()
        import torch, transformers, einops, sentencepiece, tiktoken
        import torch.nn.functional as F
        from transformers import AutoTokenizer, AutoModel, AutoConfig
        assert transformers.__version__==required["transformers"], transformers.__version__
        if not torch.cuda.is_available(): raise RuntimeError("V47 full run requires Kaggle GPU runtime")
        if torch.cuda.device_count()<2 and not V47_SMOKE: raise RuntimeError(f"V47 expects T4x2; detected {torch.cuda.device_count()} GPU(s)")
        os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
        os.environ.setdefault("HF_HOME","/kaggle/working/v47_hf_cache" if Path("/kaggle/working").exists() else "/tmp/v47_hf_cache")
        DEEP_ENV_READY=True
        print("torch",torch.__version__,"transformers",transformers.__version__,"CUDA GPUs",torch.cuda.device_count())
    except Exception as e:
        DEEP_ENV_ERROR=f"{type(e).__name__}: {e}"; print("V47 deep environment failed closed:",DEEP_ENV_ERROR)
else:
    print("Deep dependency stage skipped because oracle failed or V47_SKIP_DEEP=1")


def semantic_text_norm_v47(x): return re.sub(r"\s+"," ",html.unescape(str(x))).strip()

BASE_USER_TEXTS=[]
for u in all_users:
    ui=uid2idx[u]; g=user_chat.get(u)
    recent=[]
    if g is not None and len(g):
        learn=g[~g["is_noise"]].copy()
        if len(learn):
            rows=list(learn.sort_values(["timestamp","chat_id"]).itertuples(index=False))[-3:]
            tags=["OLDER","RECENT","CURRENT"][-len(rows):]
            recent=[f"[{t}] {semantic_text_norm_v47(r.user_chat_text)}" for t,r in zip(tags,rows)]
    history=" ".join(recent) if recent else "[NO_LEARNING_CHAT]"
    assess="; ".join(f"{SKILL_NAMES[j]}={float(A[ui,j]):.0f}/5" for j in range(len(SKILL_NAMES)))
    BASE_USER_TEXTS.append(f"Riwayat percakapan belajar: {history}. Profil asesmen: {assess}.")
BASE_USER_TEXTS=np.asarray(BASE_USER_TEXTS,dtype=object)
mods=modules.set_index("module_id").loc[MODULES].reset_index()
V47_MODULE_DOCS=[f"Modul Intelligo: {r.module_name}. Materi: {r.description_and_syllabus}. Prasyarat: {r.prerequisite_level}." for r in mods.itertuples(index=False)]
print("V47 deterministic text views ready",len(BASE_USER_TEXTS),len(V47_MODULE_DOCS))

# ---- cell 30 (code) ----
# Robust pinned model resolution and frozen train-user semantic extraction.
EMBEDDINGS_READY=False; EMBED_ERROR=""; V47_EMBED_AUDIT={}
E5_TRAIN_SCORES=np.zeros((N_TRAIN,N_MODULES),np.float32); NOMIC_TRAIN_SCORES=np.zeros_like(E5_TRAIN_SCORES)


def resolve_hf_source_v47(model_id,env_key):
    env=os.getenv(env_key,"").strip()
    if env and Path(env).exists(): return env
    short=model_id.split("/")[-1].lower(); root=Path("/kaggle/input")
    if root.exists():
        hits=[]
        for p in root.rglob("config.json"):
            if short in str(p.parent).lower(): hits.append(str(p.parent))
        if hits: return sorted(hits,key=len)[0]
    return model_id


def average_pool_v47(last_hidden,mask):
    z=last_hidden.masked_fill(~mask[...,None].bool(),0.0)
    return z.sum(dim=1)/mask.sum(dim=1)[...,None].clamp_min(1)


def encode_batches_v47(tok,mdl,texts,device,batch=ENCODER_BATCH):
    texts=[str(x) for x in texts]; cur=int(batch)
    while True:
        try:
            chunks=[]
            with torch.inference_mode():
                for st in range(0,len(texts),cur):
                    z=tok(texts[st:st+cur],padding=True,truncation=True,max_length=ENCODER_MAX_LEN,return_tensors="pt")
                    z={k:v.to(device,non_blocking=True) for k,v in z.items()}
                    o=mdl(**z); h=o.last_hidden_state if hasattr(o,"last_hidden_state") else o[0]
                    e=average_pool_v47(h,z["attention_mask"]); e=F.normalize(e.float(),p=2,dim=1).cpu().numpy().astype(np.float32)
                    chunks.append(e); del z,o,h,e
            return np.concatenate(chunks,axis=0)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or cur<=2: raise
            cur=max(2,cur//2); print(device,"OOM; retry batch",cur); gc.collect(); torch.cuda.empty_cache()


def load_encoder_v47(kind,device):
    if kind=="e5":
        model_id,revision,env_key=E5_MODEL_ID,E5_REVISION,"V47_E5_MODEL_PATH"; trust=False
    else:
        model_id,revision,env_key=NOMIC_MODEL_ID,NOMIC_REVISION,"V47_NOMIC_MODEL_PATH"; trust=True
    source=resolve_hf_source_v47(model_id,env_key); rev=revision if source==model_id else None
    print("loading",kind,model_id,"from",source,"on",device,"revision",rev or "local")
    tok=AutoTokenizer.from_pretrained(source,revision=rev,trust_remote_code=trust)
    if kind=="nomic":
        cfg=AutoConfig.from_pretrained(source,revision=rev,trust_remote_code=True,code_revision=NOMIC_CODE_REVISION if source==model_id else None)
        for attr in ["use_flash_attn","fused_dropout_add_ln","fused_bias_fc","use_xentropy"]:
            if hasattr(cfg,attr): setattr(cfg,attr,False)
        kwargs=dict(config=cfg,revision=rev,trust_remote_code=True)
        if source==model_id: kwargs["code_revision"]=NOMIC_CODE_REVISION
        mdl=AutoModel.from_pretrained(source,**kwargs).to(device)
    else:
        mdl=AutoModel.from_pretrained(source,revision=rev,torch_dtype=torch.float16).to(device)
    mdl.eval()
    for p in mdl.parameters(): p.requires_grad_(False)
    return tok,mdl,source,rev


def encoder_inputs_v47(kind,user_texts,module_docs):
    if kind=="e5":
        qs=[f"Instruct: {E5_TASK}\nQuery: {x}" for x in user_texts]; ds=list(map(str,module_docs))
    else:
        qs=["search_query: "+str(x) for x in user_texts]; ds=["search_document: "+str(x) for x in module_docs]
    return qs,ds


def extract_semantic_scores_v47(kind,user_texts,device):
    tok=mdl=None
    try:
        tok,mdl,source,rev=load_encoder_v47(kind,device)
        q,docs=encoder_inputs_v47(kind,user_texts,V47_MODULE_DOCS)
        # Environment-only repeatability smoke before the full extraction.
        dq,dd=encoder_inputs_v47(kind,["Saya ingin menentukan materi belajar berikutnya dari profil saya."],[V47_MODULE_DOCS[0]])
        e1=encode_batches_v47(tok,mdl,dq+dd,device,batch=2); e2=encode_batches_v47(tok,mdl,dq+dd,device,batch=2)
        drift=float(np.max(np.abs(e1-e2)))
        if not np.isfinite(e1).all() or drift>5e-4: raise RuntimeError(f"{kind} dummy repeatability failed, drift={drift}")
        de=encode_batches_v47(tok,mdl,docs,device)
        ue=encode_batches_v47(tok,mdl,q,device)
        scores=(ue@de.T).astype(np.float32)
        audit={"kind":kind,"model_id":E5_MODEL_ID if kind=="e5" else NOMIC_MODEL_ID,
               "revision":E5_REVISION if kind=="e5" else NOMIC_REVISION,"resolved_source":source,"resolved_revision":rev,
               "device":device,"embedding_dim":int(de.shape[1]),"users":int(len(user_texts)),"max_length":ENCODER_MAX_LEN,
               "dummy_repeat_max_abs_diff":drift,"finite":bool(np.isfinite(scores).all()),
               "nomic_code_revision":NOMIC_CODE_REVISION if kind=="nomic" else None}
        return scores,audit
    finally:
        if mdl is not None: del mdl
        if tok is not None: del tok
        gc.collect()
        if 'torch' in globals() and torch.cuda.is_available(): torch.cuda.empty_cache()

if ORACLE_GATE and DEEP_ENV_READY and not V47_SKIP_DEEP:
    try:
        # Independent GPUs; fallback to sequential extraction if concurrent HF loading is not stable in the runtime.
        devices={"e5":"cuda:0","nomic":"cuda:1" if torch.cuda.device_count()>1 else "cuda:0"}
        results={}
        try:
            with ThreadPoolExecutor(max_workers=2) as ex:
                fut={ex.submit(extract_semantic_scores_v47,k,BASE_USER_TEXTS[:N_TRAIN],devices[k]):k for k in ["e5","nomic"]}
                for f in as_completed(fut): results[fut[f]]=f.result()
        except Exception as parallel_error:
            print("Parallel encoder extraction failed; retrying sequentially:",type(parallel_error).__name__,str(parallel_error)[:240])
            gc.collect(); torch.cuda.empty_cache(); results={}
            for k in ["e5","nomic"]: results[k]=extract_semantic_scores_v47(k,BASE_USER_TEXTS[:N_TRAIN],devices[k])
        E5_TRAIN_SCORES,V47_EMBED_AUDIT["e5"]=results["e5"]
        NOMIC_TRAIN_SCORES,V47_EMBED_AUDIT["nomic"]=results["nomic"]
        assert E5_TRAIN_SCORES.shape==(N_TRAIN,N_MODULES) and NOMIC_TRAIN_SCORES.shape==(N_TRAIN,N_MODULES)
        assert np.isfinite(E5_TRAIN_SCORES).all() and np.isfinite(NOMIC_TRAIN_SCORES).all()
        EMBEDDINGS_READY=True
        np.savez_compressed(V47_OUT/"v47_train_semantic_scores.npz",e5=E5_TRAIN_SCORES,nomic=NOMIC_TRAIN_SCORES)
        json.dump(V47_EMBED_AUDIT,open(V47_OUT/"v47_embedding_audit.json","w"),indent=2)
        print("V47 train semantic score matrices",E5_TRAIN_SCORES.shape,NOMIC_TRAIN_SCORES.shape)
    except Exception as e:
        EMBED_ERROR=f"{type(e).__name__}: {e}"; print("V47 embedding stage failed closed:",EMBED_ERROR); EMBEDDINGS_READY=False
else:
    print("Semantic extraction skipped because preflight/oracle failed or V47_SKIP_DEEP=1")

# ---- cell 31 (markdown) ----
# ## Exact Top1 action-utility heads
#
# Each eligible S0 user contributes exactly three candidate actions: promote the current champion rank 2, 3, or 4 module to rank 1 while preserving the Top4 score multiset and every lower rank.
#
# The supervised target is the exact per-user `ΔNDCG@5` of that action. Each semantic head receives only its own frozen similarity signal plus the shared champion margin, challenger rank, readiness difference, and incumbent/challenger identities. A structural head is trained on historical structured features solely as a matched diagnostic control.
#
# Ridge regularization is selected from the preregistered grid only inside ACTION_TRAIN. All ACTION_TRAIN OOF evaluation uses a nested structural construction so the held-out outer fold never contributes labels to the utility-head training surfaces.

# ---- cell 32 (code) ----
SEM_NUM_NAMES=["challenger_sem","incumbent_sem","sem_diff","abs_sem_diff","anchor_gap","anchor_gap_z","readiness_diff","rank2","rank3","rank4"]
STRUCT_NUM_NAMES=["anchor_gap","anchor_gap_z","rank2","rank3","rank4","pre_diff","v31_minus_v30_diff",
                  "lex_word_diff","lex_char_diff","topic_count_diff","topic_recency_diff","topic_latest_diff","readiness_diff","invalid_readiness_diff"]


def action_numeric_v47(gi,surf,r,inc,c,rank,sem_scores=None,control=False):
    a=surf["anchor"][r]; sd=max(float(np.std(a)),1e-6); gap=float(a[c]-a[inc]); gapz=gap/sd
    if control:
        return np.asarray([gap,gapz,float(rank==2),float(rank==3),float(rank==4),
            float(surf["pre"][r,c]-surf["pre"][r,inc]),
            float((surf["v31"][r,c]-surf["v30"][r,c])-(surf["v31"][r,inc]-surf["v30"][r,inc])),
            float(LEX_WORD[gi,c]-LEX_WORD[gi,inc]),float(LEX_CHAR[gi,c]-LEX_CHAR[gi,inc]),
            float(MODULE_COUNT[gi,c]-MODULE_COUNT[gi,inc]),float(MODULE_RECENCY[gi,c]-MODULE_RECENCY[gi,inc]),
            float(MODULE_LATEST[gi,c]-MODULE_LATEST[gi,inc]),float(MODULE_READINESS[gi,c]-MODULE_READINESS[gi,inc]),
            float(MODULE_INVALID[gi,c]-MODULE_INVALID[gi,inc])],np.float32)
    chal=float(sem_scores[r,c]); incs=float(sem_scores[r,inc]); diff=chal-incs
    return np.asarray([chal,incs,diff,abs(diff),gap,gapz,float(MODULE_READINESS[gi,c]-MODULE_READINESS[gi,inc]),
                       float(rank==2),float(rank==3),float(rank==4)],np.float32)


def make_action_dataset_v47(global_idx,surf,eligible,e5_rows,nomic_rows,ytrue=None):
    global_idx=np.asarray(global_idx,int); eligible=np.asarray(eligible,bool)
    e5_rows=np.asarray(e5_rows,np.float32); nomic_rows=np.asarray(nomic_rows,np.float32)
    E=[]; N=[]; S=[]; incs=[]; chals=[]; target=[]; meta=[]
    for r,gi in enumerate(global_idx):
        if not eligible[r]: continue
        anchor=surf["anchor"][r]; top4=np.argsort(-anchor,kind="mergesort")[:4]; inc=int(top4[0])
        before=float(per_user_ndcg(ytrue[r:r+1],anchor[None,:])[0]) if ytrue is not None else np.nan
        for rank,c in enumerate(top4[1:],start=2):
            c=int(c); delta=np.nan
            if ytrue is not None:
                tmp=anchor[None,:].copy(); apply_top1_permutation_v47(tmp,0,c)
                delta=float(per_user_ndcg(ytrue[r:r+1],tmp)[0]-before)
            E.append(action_numeric_v47(int(gi),surf,r,inc,c,rank,e5_rows,False))
            N.append(action_numeric_v47(int(gi),surf,r,inc,c,rank,nomic_rows,False))
            S.append(action_numeric_v47(int(gi),surf,r,inc,c,rank,None,True))
            incs.append(inc); chals.append(c); target.append(delta)
            meta.append({"global_row":int(gi),"user_id":str(all_users[int(gi)]),"template_group":str(LATEST_GROUP[int(gi)]),
                         "incumbent":inc,"challenger":c,"incumbent_module":MODULES[inc],"challenger_module":MODULES[c],
                         "challenger_rank":int(rank),"target_delta":float(delta) if np.isfinite(delta) else np.nan})
    return {"e5":np.asarray(E,np.float32).reshape(-1,len(SEM_NUM_NAMES)),
            "nomic":np.asarray(N,np.float32).reshape(-1,len(SEM_NUM_NAMES)),
            "struct":np.asarray(S,np.float32).reshape(-1,len(STRUCT_NUM_NAMES)),
            "inc":np.asarray(incs,np.int16),"chal":np.asarray(chals,np.int16),"target":np.asarray(target,float),"meta":pd.DataFrame(meta),
            "global_idx":global_idx,"surface":surf,"eligible":eligible}


def concat_action_data_v47(parts):
    parts=[p for p in parts if p is not None and len(p["meta"])]
    if not parts: return None
    return {"e5":np.concatenate([p["e5"] for p in parts]),"nomic":np.concatenate([p["nomic"] for p in parts]),
            "struct":np.concatenate([p["struct"] for p in parts]),"inc":np.concatenate([p["inc"] for p in parts]),
            "chal":np.concatenate([p["chal"] for p in parts]),"target":np.concatenate([p["target"] for p in parts]),
            "meta":pd.concat([p["meta"] for p in parts],ignore_index=True)}


def action_weights_v47(delta):
    d=np.asarray(delta,float); scale=max(float(np.mean(np.abs(d))),1e-6)
    return np.clip(np.abs(d)/scale,WEIGHT_CLIP_LOW,WEIGHT_CLIP_HIGH)


def fixed_onehot_v47(idx): return np.eye(N_MODULES,dtype=np.float32)[np.asarray(idx,int)]

class RidgeUtilityV47:
    def __init__(self,scaler,model,alpha,head): self.scaler=scaler; self.model=model; self.alpha=float(alpha); self.head=head
    def matrix(self,num,inc,chal):
        z=self.scaler.transform(np.asarray(num,np.float32))
        return np.hstack([z,fixed_onehot_v47(inc),fixed_onehot_v47(chal)]).astype(np.float32)
    def predict(self,num,inc,chal): return self.model.predict(self.matrix(num,inc,chal))


def fit_ridge_v47(data,head,alpha):
    num=np.asarray(data[head],np.float32); sc=StandardScaler(); z=sc.fit_transform(num)
    Xd=np.hstack([z,fixed_onehot_v47(data["inc"]),fixed_onehot_v47(data["chal"])]).astype(np.float32)
    w=action_weights_v47(data["target"]); mdl=Ridge(alpha=float(alpha),fit_intercept=True,solver="lsqr")
    mdl.fit(Xd,data["target"],sample_weight=w); pred=mdl.predict(Xd)
    return RidgeUtilityV47(sc,mdl,alpha,head),{"n":int(len(pred)),"weighted_rmse":float(np.sqrt(np.average((data['target']-pred)**2,weights=w))),
        "sign_accuracy":float(np.average((pred>0)==(data['target']>0),weights=w)),"coef_l2":float(np.sqrt(np.square(mdl.coef_).sum()))}


def subset_action_data_v47(data,mask):
    mask=np.asarray(mask,bool)
    return {"e5":data["e5"][mask],"nomic":data["nomic"][mask],"struct":data["struct"][mask],
            "inc":data["inc"][mask],"chal":data["chal"][mask],"target":data["target"][mask],"meta":data["meta"].loc[mask].reset_index(drop=True)}


def choose_alpha_v47(data,head,seed):
    users=np.sort(data["meta"].global_row.unique().astype(int))
    folds=make_group_folds_global(users,min(HEAD_INNER_FOLDS,len(users)),int(seed))
    rows_global=data["meta"].global_row.to_numpy(int); audit=[]
    for alpha in RIDGE_ALPHA_GRID:
        ss=0.0; ww=0.0
        for f,(tr,va) in enumerate(folds):
            mt=np.isin(rows_global,tr); mv=np.isin(rows_global,va)
            if not mt.any() or not mv.any(): continue
            model,_=fit_ridge_v47(subset_action_data_v47(data,mt),head,alpha)
            pred=model.predict(data[head][mv],data["inc"][mv],data["chal"][mv]); w=action_weights_v47(data["target"][mv])
            ss+=float(np.sum(w*(data["target"][mv]-pred)**2)); ww+=float(w.sum())
        rmse=float(np.sqrt(ss/max(ww,1e-12))); audit.append({"head":head,"alpha":float(alpha),"cv_weighted_rmse":rmse})
    ad=pd.DataFrame(audit).sort_values(["cv_weighted_rmse","alpha"]).reset_index(drop=True)
    return float(ad.iloc[0].alpha),ad

print("V47 action utilities configured:",len(SEM_NUM_NAMES),"semantic numeric + 34 fixed identity features")

# ---- cell 33 (code) ----
# Fully prospective ACTION_TRAIN outer evaluation. Each utility outer fold excludes its validation users from all nested training surfaces.
MODELS_READY=False; V47_ALPHA_AUDIT=[]; V47_HEAD_FIT_AUDIT=[]; V47_OUTER_FOLD_MODELS={"e5":[],"nomic":[],"struct":[]}
V47_OOF_PARTS=[]; V47_OOF_ACTIONS=pd.DataFrame()

if EMBEDDINGS_READY:
    OUTER_FOLDS=make_group_folds_global(ACTION_TRAIN_IDX,HEAD_OUTER_FOLDS,V47_HEAD_SEED)
    for f,(tr,va) in enumerate(OUTER_FOLDS):
        print(f"\nV47 ACTION_TRAIN outer fold {f+1}/{len(OUTER_FOLDS)}: train={len(tr)} val={len(va)}")
        # Cross-fit the structural champion analogue on the utility-head training side only.
        tr_surf=crossfit_v47_surface(tr,STRUCT_TRAIN_CROSSFIT_FOLDS,V47_STRUCT_SEED+50000+1000*f)
        va_surf=build_v47_val_surface(tr,va,V47_STRUCT_SEED+60000+1000*f)
        tr_data=make_action_dataset_v47(tr,tr_surf,tr_surf["state"]==0,E5_TRAIN_SCORES[tr],NOMIC_TRAIN_SCORES[tr],y[tr])
        va_data=make_action_dataset_v47(va,va_surf,va_surf["state"]==0,E5_TRAIN_SCORES[va],NOMIC_TRAIN_SCORES[va],y[va])
        va_data["meta"]["outer_fold"]=f
        for head in ["e5","nomic","struct"]:
            alpha,aa=choose_alpha_v47(tr_data,head,V47_HEAD_SEED+100000*f+_seed_from_text(head))
            aa["outer_fold"]=f; V47_ALPHA_AUDIT.append(aa)
            mdl,fit_aud=fit_ridge_v47(tr_data,head,alpha); V47_OUTER_FOLD_MODELS[head].append(mdl)
            pred=mdl.predict(va_data[head],va_data["inc"],va_data["chal"]); va_data["meta"][head+"_oof_pred"]=pred
            V47_HEAD_FIT_AUDIT.append({"head":head,"fit":"outer_fold","fold":f,"alpha":alpha,**fit_aud})
        V47_OOF_PARTS.append({"e5":va_data["e5"],"nomic":va_data["nomic"],"struct":va_data["struct"],
                              "inc":va_data["inc"],"chal":va_data["chal"],"target":va_data["target"],"meta":va_data["meta"]})
        del tr_surf,va_surf,tr_data,va_data; gc.collect()

    OOF_ACTION_DATA=concat_action_data_v47(V47_OOF_PARTS)
    V47_OOF_ACTIONS=OOF_ACTION_DATA["meta"].copy(); V47_OOF_ACTIONS["target_delta"]=OOF_ACTION_DATA["target"]
    assert set(V47_OOF_ACTIONS.global_row.unique()) <= set(ACTION_TRAIN_IDX.tolist())

    # Full ACTION_TRAIN heads train only on the nested OOF feature table. Hyperparameters never see DESIGN.
    V47_FULL_MODELS={}; V47_GLOBAL_ALPHA={}
    for head in ["e5","nomic","struct"]:
        alpha,aa=choose_alpha_v47(OOF_ACTION_DATA,head,V47_HEAD_SEED+900000+_seed_from_text(head))
        aa["outer_fold"]=-1; V47_ALPHA_AUDIT.append(aa); V47_GLOBAL_ALPHA[head]=alpha
        mdl,fit_aud=fit_ridge_v47(OOF_ACTION_DATA,head,alpha); V47_FULL_MODELS[head]=mdl
        V47_HEAD_FIT_AUDIT.append({"head":head,"fit":"full_oof_features","fold":-1,"alpha":alpha,**fit_aud})
    V47_ALPHA_AUDIT=pd.concat(V47_ALPHA_AUDIT,ignore_index=True); V47_HEAD_FIT_AUDIT=pd.DataFrame(V47_HEAD_FIT_AUDIT)
    MODELS_READY=True
    print("Global alphas",V47_GLOBAL_ALPHA)
else:
    OOF_ACTION_DATA=None; V47_FULL_MODELS={}; V47_GLOBAL_ALPHA={}; V47_ALPHA_AUDIT=pd.DataFrame(); V47_HEAD_FIT_AUDIT=pd.DataFrame()
    print("V47 action heads not fit because embeddings are unavailable")

# ---- cell 34 (markdown) ----
# ## ACTION_TRAIN risk-coverage policy freeze
#
# The policy is chosen only from nested ACTION_TRAIN OOF predictions. Coverage is measured among eligible S0 users and is limited to the preregistered grid `0.25% / 0.5% / 1.0% / 1.5%` in the full run.
#
# Each expert independently picks one challenger. Consensus exists only when E5 and Nomic pick the same challenger and both predict positive utility. At each coverage, the highest minimum-of-two OOF utilities are selected. The selected coverage is frozen before DESIGN.

# ---- cell 35 (code) ----
def choices_from_action_table_v47(meta,pred_col,score_col=None):
    rows=[]; score_col=pred_col if score_col is None else score_col
    for gi,z in meta.groupby("global_row",sort=False):
        idx=z.index.to_numpy(int); vals=meta.loc[idx,pred_col].to_numpy(float); j=int(idx[np.argmax(vals)])
        r=meta.loc[j]
        rows.append({"global_row":int(gi),"user_id":r.user_id,"action_index":j,"challenger":int(r.challenger),
                     "challenger_module":r.challenger_module,"utility":float(meta.loc[j,pred_col]),"score":float(meta.loc[j,score_col]),
                     "valid":bool(float(meta.loc[j,score_col])>0),"target_delta":float(r.target_delta),"outer_fold":int(r.get("outer_fold",-1)),
                     "template_group":str(r.template_group)})
    return pd.DataFrame(rows)


def consensus_choices_v47(e5,nomic):
    a=e5.set_index("global_row"); b=nomic.set_index("global_row"); rows=[]
    for gi in sorted(set(a.index)&set(b.index)):
        x=a.loc[gi]; z=b.loc[gi]; same=int(x.challenger)==int(z.challenger)
        rows.append({"global_row":int(gi),"user_id":x.user_id,"challenger":int(x.challenger),"challenger_module":x.challenger_module,
                     "score":float(min(x.score,z.score)),"e5_utility":float(x.utility),"nomic_utility":float(z.utility),
                     "valid":bool(same and x.valid and z.valid),"target_delta":float(x.target_delta) if same else np.nan,
                     "outer_fold":int(x.outer_fold),"template_group":str(x.template_group)})
    return pd.DataFrame(rows)


def select_top_coverage_v47(choice_df,coverage,n_eligible):
    z=choice_df[choice_df.valid==True].sort_values(["score","global_row"],ascending=[False,True]).copy()
    n=max(1,int(math.ceil(float(coverage)*max(1,int(n_eligible)))))
    z["selected"]=False
    if len(z): z.loc[z.index[:min(n,len(z))],"selected"]=True
    return z,n


def policy_stats_from_choices_v47(choice_df,coverage,population_rows,seed):
    population_rows=np.asarray(population_rows,int); selected,target_n=select_top_coverage_v47(choice_df,coverage,len(population_rows))
    chosen=selected[selected.selected==True].copy(); delta=np.zeros(len(population_rows),float); pos={int(g):i for i,g in enumerate(population_rows)}
    for r in chosen.itertuples(index=False):
        if int(r.global_row) in pos and np.isfinite(r.target_delta): delta[pos[int(r.global_row)]]=float(r.target_delta)
    ci=cluster_bootstrap_mean(delta,TRAIN_GROUP[population_rows],seed)
    folds=[]
    for f in range(HEAD_OUTER_FOLDS):
        users=set(choice_df.loc[choice_df.outer_fold==f,"global_row"].astype(int).tolist()); ix=np.asarray([i for i,g in enumerate(population_rows) if int(g) in users],int)
        if len(ix): folds.append(float(delta[ix].mean()))
    ordix=np.argsort(population_rows); blocks=np.array_split(ordix,EVAL_BLOCKS); block_delta=[float(delta[ix].mean()) for ix in blocks if len(ix)]
    benef=int((chosen.target_delta>1e-12).sum()) if len(chosen) else 0; harm=int((chosen.target_delta<-1e-12).sum()) if len(chosen) else 0; ties=int(len(chosen)-benef-harm)
    return {"coverage":float(coverage),"target_actions":int(target_n),"actions":int(len(chosen)),"delta":float(delta.mean()),
            "ci_low":ci[0],"ci_high":ci[1],"beneficial":benef,"harmful":harm,"ties":ties,
            "benefit_harm_ratio":float(benef/max(1,harm)),"fold_wins":int(sum(x>0 for x in folds)),
            "min_id_block_delta":float(min(block_delta)) if block_delta else 0.0,"actual_coverage":float(len(chosen)/max(1,len(population_rows)))},chosen,delta

POLICY_READY=False; SELECTED_COVERAGE=None; V47_ACTIONTRAIN_RISK=pd.DataFrame(); V47_ACTIONTRAIN_SELECTED=pd.DataFrame()
if MODELS_READY:
    e5=choices_from_action_table_v47(V47_OOF_ACTIONS,"e5_oof_pred")
    nom=choices_from_action_table_v47(V47_OOF_ACTIONS,"nomic_oof_pred")
    st=choices_from_action_table_v47(V47_OOF_ACTIONS,"struct_oof_pred")
    con=consensus_choices_v47(e5,nom)
    rec=[]; selected_by_cov={}
    eligible_users=np.sort(V47_OOF_ACTIONS.global_row.unique().astype(int))
    for ci,cov in enumerate(COVERAGE_GRID):
        cr,cz,du=policy_stats_from_choices_v47(con,cov,eligible_users,V47_BOOT_SEED+100*ci)
        er,_,_=policy_stats_from_choices_v47(e5,cov,eligible_users,V47_BOOT_SEED+100*ci+1)
        nr,_,_=policy_stats_from_choices_v47(nom,cov,eligible_users,V47_BOOT_SEED+100*ci+2)
        sr,_,_=policy_stats_from_choices_v47(st,cov,eligible_users,V47_BOOT_SEED+100*ci+3)
        cr.update({"e5_delta_matched":er["delta"],"nomic_delta_matched":nr["delta"],"struct_delta_matched":sr["delta"]})
        cr["consensus_lift_matched"]=bool(cr["delta"]>max(er["delta"],nr["delta"],sr["delta"]))
        rec.append(cr); selected_by_cov[float(cov)]=cz
    V47_ACTIONTRAIN_RISK=pd.DataFrame(rec)
    eligible=V47_ACTIONTRAIN_RISK[
        (V47_ACTIONTRAIN_RISK.delta>0) & (V47_ACTIONTRAIN_RISK.ci_low>=-1e-12) &
        (V47_ACTIONTRAIN_RISK.fold_wins>=DESIGN_MIN_FOLD_WINS) &
        (V47_ACTIONTRAIN_RISK.min_id_block_delta>=BLOCK_CATASTROPHIC_FLOOR) &
        (V47_ACTIONTRAIN_RISK.benefit_harm_ratio>=BENEFIT_HARM_RATIO)
    ].copy()
    if len(eligible):
        best=eligible.sort_values(["delta","coverage"],ascending=[False,True]).iloc[0]
        SELECTED_COVERAGE=float(best.coverage); POLICY_READY=True; V47_ACTIONTRAIN_SELECTED=selected_by_cov[SELECTED_COVERAGE]
    print(V47_ACTIONTRAIN_RISK.to_string(index=False)); print("POLICY_READY",POLICY_READY,"SELECTED_COVERAGE",SELECTED_COVERAGE)
else:
    print("ACTION_TRAIN risk-coverage unavailable")

# ---- cell 36 (markdown) ----
# ## Blind DESIGN gate
#
# No feature family, alpha, coverage, candidate rule, or tie-break is changed after the ACTION_TRAIN policy is frozen. DESIGN evaluates E5-only, Nomic-only, the structural control, and dual consensus at matched coverage.
#
# For deployment-side uncertainty, each selected action must have a positive full-model utility and a positive **minimum prediction across all five ACTION_TRAIN outer-fold models** for that expert. Dual consensus must choose the same challenger in both encoders.

# ---- cell 37 (code) ----
def predict_head_bundle_v47(data,head,full_model,fold_models):
    full=full_model.predict(data[head],data["inc"],data["chal"])
    folds=np.vstack([m.predict(data[head],data["inc"],data["chal"]) for m in fold_models])
    lower=np.min(folds,axis=0)
    return full,folds,lower


def choices_with_lower_v47(data,head,full_model,fold_models):
    full,folds,lower=predict_head_bundle_v47(data,head,full_model,fold_models)
    meta=data["meta"].copy(); meta[head+"_full"]=full; meta[head+"_lower"]=lower
    rows=[]
    for gi,z in meta.groupby("global_row",sort=False):
        idx=z.index.to_numpy(int); j=int(idx[np.argmax(meta.loc[idx,head+"_full"].to_numpy(float))]); r=meta.loc[j]
        rows.append({"global_row":int(gi),"user_id":r.user_id,"action_index":j,"challenger":int(r.challenger),
                     "challenger_module":r.challenger_module,"utility":float(meta.loc[j,head+"_full"]),"score":float(meta.loc[j,head+"_lower"]),
                     "valid":bool(float(meta.loc[j,head+"_full"])>0 and float(meta.loc[j,head+"_lower"])>0),
                     "target_delta":float(r.target_delta) if np.isfinite(r.target_delta) else np.nan,"template_group":str(r.template_group)})
    return pd.DataFrame(rows),meta


def deploy_choice_v47(data,choice_df,coverage):
    base=np.asarray(data["surface"]["anchor"],np.float32); cand=base.copy(); selected,target_n=select_top_coverage_v47(choice_df,coverage,int(np.sum(data["eligible"])))
    chosen=selected[selected.selected==True].copy(); local_by_global={int(g):i for i,g in enumerate(data["global_idx"])}
    for r in chosen.itertuples(index=False): apply_top1_permutation_v47(cand,local_by_global[int(r.global_row)],int(r.challenger))
    ledger=choice_df.copy(); picked=set(chosen.global_row.astype(int).tolist()); ledger["executed"]=ledger.global_row.astype(int).isin(picked)
    return cand,ledger,target_n


def eval_partition_v47(global_idx,ytrue,base,cand,ledger,seed):
    global_idx=np.asarray(global_idx,int); du=per_user_ndcg(ytrue,cand)-per_user_ndcg(ytrue,base); ci=cluster_bootstrap_mean(du,LATEST_GROUP[global_idx],seed)
    acts=ledger[ledger.executed==True].copy(); benef=int((acts.target_delta>1e-12).sum()) if len(acts) else 0; harm=int((acts.target_delta<-1e-12).sum()) if len(acts) else 0
    # Fixed grouped evaluation folds and contiguous ID blocks.
    fr=[]
    efolds=make_group_folds_global(global_idx,min(EVAL_FOLDS,len(global_idx)),seed+11)
    for f,(_,va) in enumerate(efolds):
        loc=np.asarray([np.where(global_idx==g)[0][0] for g in va],int); mm=metric_v47(ytrue[loc],base[loc],cand[loc]); fr.append({"fold":f,"n":len(loc),**mm})
    br=[]; order=np.argsort(global_idx)
    for b,ix in enumerate(np.array_split(order,EVAL_BLOCKS)):
        if len(ix): br.append({"block":b,"n":len(ix),"id_min":int(global_idx[ix].min()+1),"id_max":int(global_idx[ix].max()+1),**metric_v47(ytrue[ix],base[ix],cand[ix])})
    mm=metric_v47(ytrue,base,cand); geom=geometry_audit_v47(base,cand)
    summary={**mm,"ci_low":ci[0],"ci_high":ci[1],"actions":int(len(acts)),"beneficial":benef,"harmful":harm,
             "benefit_harm_ratio":float(benef/max(1,harm)),"fold_wins":int(sum(x["delta"]>0 for x in fr)),
             "min_id_block_delta":float(min(x["delta"] for x in br)) if br else 0.0,**{f"geom_{k}":v for k,v in geom.items()}}
    return summary,pd.DataFrame(fr),pd.DataFrame(br),du

DESIGN_GATE=False; DESIGN_EVALUATED=False
V47_DESIGN_METRICS=pd.DataFrame([{"evaluated":False,"design_gate":False}]); V47_DESIGN_SINGLE=pd.DataFrame(); V47_DESIGN_LEDGER=pd.DataFrame()
V47_DESIGN_FOLDS=pd.DataFrame(); V47_DESIGN_BLOCKS=pd.DataFrame(); DESIGN_CAND=DESIGN_BASE.copy(); DESIGN_USER_DELTA=np.zeros(len(DESIGN_IDX),float)
DESIGN_DATA=None

if POLICY_READY and MODELS_READY:
    DESIGN_DATA=make_action_dataset_v47(DESIGN_IDX,DESIGN_SURF,DESIGN_ELIG,E5_TRAIN_SCORES[DESIGN_IDX],NOMIC_TRAIN_SCORES[DESIGN_IDX],y[DESIGN_IDX])
    e5,e5meta=choices_with_lower_v47(DESIGN_DATA,"e5",V47_FULL_MODELS["e5"],V47_OUTER_FOLD_MODELS["e5"])
    nom,nmeta=choices_with_lower_v47(DESIGN_DATA,"nomic",V47_FULL_MODELS["nomic"],V47_OUTER_FOLD_MODELS["nomic"])
    st,smeta=choices_with_lower_v47(DESIGN_DATA,"struct",V47_FULL_MODELS["struct"],V47_OUTER_FOLD_MODELS["struct"])
    con=consensus_choices_v47(e5,nom)
    # target_delta is already carried from the selected action; consensus keeps it only when both experts choose the same challenger.
    policies={}
    for name,ch in [("E5_ONLY",e5),("NOMIC_ONLY",nom),("STRUCT_CONTROL",st),("DUAL",con)]:
        c,l,_=deploy_choice_v47(DESIGN_DATA,ch,SELECTED_COVERAGE); policies[name]=(c,l)
    DESIGN_CAND,V47_DESIGN_LEDGER=policies["DUAL"]
    rec=[]
    for name,(c,l) in policies.items():
        sm,_,_,_=eval_partition_v47(DESIGN_IDX,y[DESIGN_IDX],DESIGN_BASE,c,l,V47_BOOT_SEED+_seed_from_text(name))
        rec.append({"policy":name,**sm})
    V47_DESIGN_SINGLE=pd.DataFrame(rec)
    dsm,V47_DESIGN_FOLDS,V47_DESIGN_BLOCKS,DESIGN_USER_DELTA=eval_partition_v47(DESIGN_IDX,y[DESIGN_IDX],DESIGN_BASE,DESIGN_CAND,V47_DESIGN_LEDGER,V47_BOOT_SEED+700)
    si=V47_DESIGN_SINGLE.set_index("policy")
    consensus_lift=bool(dsm["delta"]>max(float(si.loc["E5_ONLY","delta"]),float(si.loc["NOMIC_ONLY","delta"]),float(si.loc["STRUCT_CONTROL","delta"])))
    protected_probe_rows=np.asarray([TEST_UID_TO_ROW[u] for u in PUBLIC_PROBE_TEST_USERS if u in TEST_UID_TO_ROW],int)
    TEST_ELIGIBLE_PREVIEW=((full_state==0) & ~np.isin(np.arange(N_TEST),protected_probe_rows)); projected=int(math.ceil(SELECTED_COVERAGE*int(TEST_ELIGIBLE_PREVIEW.sum())))
    DESIGN_GATE=bool(
        ORACLE_GATE and dsm["delta"]>=DESIGN_MIN_DELTA and dsm["ci_low"]>0 and dsm["fold_wins"]>=DESIGN_MIN_FOLD_WINS and
        dsm["min_id_block_delta"]>=BLOCK_CATASTROPHIC_FLOOR and dsm["benefit_harm_ratio"]>=BENEFIT_HARM_RATIO and
        consensus_lift and dsm["top1_delta"]>0 and abs(dsm["rel5_delta"])<1e-12 and
        dsm["geom_top4_membership_changes"]==0 and dsm["geom_top5_membership_changes"]==0 and
        dsm["geom_rank5_to_17_order_changes"]==0 and bool(dsm["geom_top4_score_multiset_ok"]) and
        TEST_MIN_ACTIONS<=projected<=TEST_MAX_ACTIONS
    )
    V47_DESIGN_METRICS=pd.DataFrame([{"evaluated":True,**dsm,"consensus_lift_matched":consensus_lift,
                                      "selected_coverage":SELECTED_COVERAGE,"projected_test_actions":projected,"design_gate":DESIGN_GATE}])
    DESIGN_EVALUATED=True
    print(V47_DESIGN_SINGLE.to_string(index=False)); print(V47_DESIGN_METRICS.to_string(index=False)); print("DESIGN_GATE",DESIGN_GATE)
else:
    print("DESIGN not opened because ACTION_TRAIN policy or models did not pass")

# ---- cell 38 (markdown) ----
# ## One-shot LOCK
#
# LOCK is evaluated only if DESIGN passes exactly as preregistered. No alpha, coverage, feature, module pair, subgroup, or threshold may change after DESIGN. The LOCK sign must be positive and the pooled prospective DESIGN+LOCK effect must remain at least `+0.0008` with a positive template-cluster bootstrap lower bound.

# ---- cell 39 (code) ----
LOCK_GATE=False; LOCK_EVALUATED=False; LOCK_CAND=LOCK_SURF["anchor"].copy(); LOCK_USER_DELTA=np.zeros(len(LOCK_IDX),float)
V47_LOCK_METRICS=pd.DataFrame([{"evaluated":False,"lock_gate":False}]); V47_LOCK_SINGLE=pd.DataFrame(); V47_LOCK_LEDGER=pd.DataFrame(); V47_LOCK_FOLDS=pd.DataFrame(); V47_LOCK_BLOCKS=pd.DataFrame()
LOCK_DATA=None; POOLED_CI=(np.nan,np.nan); POOLED_DELTA=np.nan

if DESIGN_GATE:
    LOCK_DATA=make_action_dataset_v47(LOCK_IDX,LOCK_SURF,LOCK_ELIG,E5_TRAIN_SCORES[LOCK_IDX],NOMIC_TRAIN_SCORES[LOCK_IDX],y[LOCK_IDX])
    e5,_=choices_with_lower_v47(LOCK_DATA,"e5",V47_FULL_MODELS["e5"],V47_OUTER_FOLD_MODELS["e5"])
    nom,_=choices_with_lower_v47(LOCK_DATA,"nomic",V47_FULL_MODELS["nomic"],V47_OUTER_FOLD_MODELS["nomic"])
    st,_=choices_with_lower_v47(LOCK_DATA,"struct",V47_FULL_MODELS["struct"],V47_OUTER_FOLD_MODELS["struct"])
    con=consensus_choices_v47(e5,nom)
    policies={}
    for name,ch in [("E5_ONLY",e5),("NOMIC_ONLY",nom),("STRUCT_CONTROL",st),("DUAL",con)]: policies[name]=deploy_choice_v47(LOCK_DATA,ch,SELECTED_COVERAGE)[:2]
    LOCK_CAND,V47_LOCK_LEDGER=policies["DUAL"]
    rec=[]
    for name,(c,l) in policies.items():
        sm,_,_,_=eval_partition_v47(LOCK_IDX,y[LOCK_IDX],LOCK_SURF["anchor"],c,l,V47_BOOT_SEED+2000+_seed_from_text(name)); rec.append({"policy":name,**sm})
    V47_LOCK_SINGLE=pd.DataFrame(rec)
    lsm,V47_LOCK_FOLDS,V47_LOCK_BLOCKS,LOCK_USER_DELTA=eval_partition_v47(LOCK_IDX,y[LOCK_IDX],LOCK_SURF["anchor"],LOCK_CAND,V47_LOCK_LEDGER,V47_BOOT_SEED+2700)
    pooled_delta_vec=np.concatenate([DESIGN_USER_DELTA,LOCK_USER_DELTA]); pooled_groups=np.concatenate([LATEST_GROUP[DESIGN_IDX],LATEST_GROUP[LOCK_IDX]])
    POOLED_DELTA=float(pooled_delta_vec.mean()); POOLED_CI=cluster_bootstrap_mean(pooled_delta_vec,pooled_groups,V47_BOOT_SEED+2800)
    si=V47_LOCK_SINGLE.set_index("policy"); lock_consensus_lift=bool(lsm["delta"]>max(float(si.loc["E5_ONLY","delta"]),float(si.loc["NOMIC_ONLY","delta"]),float(si.loc["STRUCT_CONTROL","delta"])))
    LOCK_GATE=bool(lsm["delta"]>0 and POOLED_DELTA>=POOLED_DESIGN_LOCK_MIN_DELTA and POOLED_CI[0]>0 and
                   lsm["min_id_block_delta"]>=BLOCK_CATASTROPHIC_FLOOR and lsm["benefit_harm_ratio"]>=BENEFIT_HARM_RATIO and
                   lock_consensus_lift and lsm["top1_delta"]>=0 and abs(lsm["rel5_delta"])<1e-12 and
                   lsm["geom_top4_membership_changes"]==0 and lsm["geom_top5_membership_changes"]==0 and
                   lsm["geom_rank5_to_17_order_changes"]==0 and bool(lsm["geom_top4_score_multiset_ok"]))
    V47_LOCK_METRICS=pd.DataFrame([{"evaluated":True,**lsm,"pooled_design_lock_delta":POOLED_DELTA,"pooled_ci_low":POOLED_CI[0],"pooled_ci_high":POOLED_CI[1],
                                    "consensus_lift_matched":lock_consensus_lift,"selected_coverage":SELECTED_COVERAGE,"lock_gate":LOCK_GATE}])
    LOCK_EVALUATED=True
    print(V47_LOCK_SINGLE.to_string(index=False)); print(V47_LOCK_METRICS.to_string(index=False)); print("LOCK_GATE",LOCK_GATE)
else:
    print("LOCK remains sealed because DESIGN did not pass")

# ---- cell 40 (markdown) ----
# ## Final refit and test deployment
#
# Only after LOCK passes are the two semantic heads refit on the already-honest ACTION_TRAIN OOF feature table plus the blind DESIGN feature table, using the **same fixed alphas and coverage**. LOCK labels are never used for fitting.
#
# Test embeddings are generated only at this point. Eligibility is exact full-data S0 and excludes every explicitly public-probed historical test user. The final action remains a Top1-only score-value permutation. If fewer than 3 or more than 15 actions survive, or any geometry invariant fails, V47 fails closed to the exact champion.

# ---- cell 41 (code) ----
FINAL_GATE=False; TEST_EMBEDDINGS_READY=False; TEST_EMBED_ERROR=""; TEST_CANDIDATE=V44_ANCHOR_TEST.copy(); V47_TEST_LEDGER=pd.DataFrame()
V47_TEST_GEOMETRY={"evaluated":False,"actions":0,"geometry_gate":False,"anchor_top5_sha256":top5_sha256(V44_ANCHOR_TEST),"candidate_top5_sha256":top5_sha256(V44_ANCHOR_TEST)}
E5_TEST_SCORES=np.zeros((N_TEST,N_MODULES),np.float32); NOMIC_TEST_SCORES=np.zeros_like(E5_TEST_SCORES)
FINAL_MODELS={}; FINAL_FOLD_MODELS={"e5":[],"nomic":[]}

if LOCK_GATE and DESIGN_DATA is not None:
    # Final training uses honest ACTION_TRAIN OOF features + DESIGN features predicted from ACTION_TRAIN only. LOCK remains excluded.
    FINAL_TRAIN_DATA=concat_action_data_v47([OOF_ACTION_DATA,{"e5":DESIGN_DATA["e5"],"nomic":DESIGN_DATA["nomic"],"struct":DESIGN_DATA["struct"],
        "inc":DESIGN_DATA["inc"],"chal":DESIGN_DATA["chal"],"target":DESIGN_DATA["target"],"meta":DESIGN_DATA["meta"]}])
    for head in ["e5","nomic"]:
        FINAL_MODELS[head],aud=fit_ridge_v47(FINAL_TRAIN_DATA,head,V47_GLOBAL_ALPHA[head])
        users=np.sort(FINAL_TRAIN_DATA["meta"].global_row.unique().astype(int)); flds=make_group_folds_global(users,HEAD_OUTER_FOLDS,V47_HEAD_SEED+990000+_seed_from_text(head))
        rg=FINAL_TRAIN_DATA["meta"].global_row.to_numpy(int)
        for f,(tr,va) in enumerate(flds):
            keep=np.isin(rg,tr); mdl,_=fit_ridge_v47(subset_action_data_v47(FINAL_TRAIN_DATA,keep),head,V47_GLOBAL_ALPHA[head]); FINAL_FOLD_MODELS[head].append(mdl)

    try:
        devices={"e5":"cuda:0","nomic":"cuda:1" if torch.cuda.device_count()>1 else "cuda:0"}; results={}
        try:
            with ThreadPoolExecutor(max_workers=2) as ex:
                fut={ex.submit(extract_semantic_scores_v47,k,BASE_USER_TEXTS[N_TRAIN:],devices[k]):k for k in ["e5","nomic"]}
                for f in as_completed(fut): results[fut[f]]=f.result()
        except Exception as parallel_error:
            print("Parallel test encoding failed; retrying sequentially:",type(parallel_error).__name__,str(parallel_error)[:240]); results={}; gc.collect(); torch.cuda.empty_cache()
            for k in ["e5","nomic"]: results[k]=extract_semantic_scores_v47(k,BASE_USER_TEXTS[N_TRAIN:],devices[k])
        E5_TEST_SCORES,test_e5_audit=results["e5"]; NOMIC_TEST_SCORES,test_nomic_audit=results["nomic"]
        TEST_EMBEDDINGS_READY=True; json.dump({"e5":test_e5_audit,"nomic":test_nomic_audit},open(V47_OUT/"v47_test_embedding_audit.json","w"),indent=2)
        np.savez_compressed(V47_OUT/"v47_test_semantic_scores.npz",e5=E5_TEST_SCORES,nomic=NOMIC_TEST_SCORES)
    except Exception as e:
        TEST_EMBED_ERROR=f"{type(e).__name__}: {e}"; print("V47 test embedding stage failed closed:",TEST_EMBED_ERROR); TEST_EMBEDDINGS_READY=False

    if TEST_EMBEDDINGS_READY:
        TEST_GLOBAL=np.arange(N_TRAIN,N_TRAIN+N_TEST,dtype=int)
        TEST_SURF={"pre":np.asarray(consensus3_test,np.float32),"v30":np.asarray(v30_test,np.float32),"v31":np.asarray(v31_test,np.float32),
                   "state":np.asarray(full_state,np.int8),"anchor":np.asarray(V44_ANCHOR_TEST,np.float32)}
        public_probe_rows=np.asarray([TEST_UID_TO_ROW[u] for u in PUBLIC_PROBE_TEST_USERS if u in TEST_UID_TO_ROW],int)
        TEST_ELIG=(full_state==0) & ~np.isin(np.arange(N_TEST),public_probe_rows)
        TEST_DATA=make_action_dataset_v47(TEST_GLOBAL,TEST_SURF,TEST_ELIG,E5_TEST_SCORES,NOMIC_TEST_SCORES,ytrue=None)
        e5,_=choices_with_lower_v47(TEST_DATA,"e5",FINAL_MODELS["e5"],FINAL_FOLD_MODELS["e5"])
        nom,_=choices_with_lower_v47(TEST_DATA,"nomic",FINAL_MODELS["nomic"],FINAL_FOLD_MODELS["nomic"])
        con=consensus_choices_v47(e5,nom)
        TEST_CANDIDATE,V47_TEST_LEDGER,_=deploy_choice_v47(TEST_DATA,con,SELECTED_COVERAGE)
        geom=geometry_audit_v47(V44_ANCHOR_TEST,TEST_CANDIDATE); acts=int(V47_TEST_LEDGER.executed.sum()) if len(V47_TEST_LEDGER) else 0
        locked_equal=bool(np.array_equal(TEST_CANDIDATE[~TEST_ELIG],V44_ANCHOR_TEST[~TEST_ELIG]))
        geometry_gate=bool(locked_equal and geom["changed_users"]==acts and geom["top1_changes"]==acts and
            geom["top4_membership_changes"]==0 and geom["top5_membership_changes"]==0 and geom["rank5_to_17_order_changes"]==0 and
            geom["top4_score_multiset_ok"] and TEST_MIN_ACTIONS<=acts<=TEST_MAX_ACTIONS)
        V47_TEST_GEOMETRY={"evaluated":True,**geom,"actions":acts,"locked_rows_bit_exact":locked_equal,"geometry_gate":geometry_gate,
            "anchor_top5_sha256":top5_sha256(V44_ANCHOR_TEST),"candidate_top5_sha256":top5_sha256(TEST_CANDIDATE)}
        FINAL_GATE=bool(LOCK_GATE and geometry_gate and top5_sha256(V44_ANCHOR_TEST)==EXPECTED_V47_ANCHOR_HASH and not V47_SMOKE)

PRIMARY=TEST_CANDIDATE if FINAL_GATE else V44_ANCHOR_TEST.copy()
print(pd.DataFrame([V47_TEST_GEOMETRY]).to_string(index=False)); print("FINAL_GATE",FINAL_GATE)

# ---- cell 42 (markdown) ----
# ## Artifacts and submission decision
#
# `submission_v47_primary.csv` is leaderboard-eligible **only if `final_gate == true`**. Otherwise it is intentionally bit-identical to the exact `0.66028` champion and should not consume a Kaggle slot.
#
# The notebook exports split provenance, nested ACTION_TRAIN OOF ledgers, alpha selection, risk-coverage, DESIGN/LOCK single-expert comparisons, block/fold audits, embedding audits, exact test geometry, and a machine-readable reproduction summary.

# ---- cell 43 (code) ----
def submission_v47(scores):
    out=sample.copy(); out.loc[:,MODULES]=np.asarray(scores,np.float32)
    assert out.user_id.astype(str).tolist()==test.user_id.astype(str).tolist(); assert np.isfinite(out[MODULES].to_numpy()).all()
    return out

anchor_sub=submission_v47(V44_ANCHOR_TEST); candidate_sub=submission_v47(TEST_CANDIDATE); primary_sub=submission_v47(PRIMARY)
anchor_sub.to_csv(V47_OUT/"submission_v47_anchor_066028_reference.csv",index=False)
candidate_sub.to_csv(V47_OUT/"submission_v47_candidate_dual_encoder.csv",index=False)
primary_sub.to_csv(V47_OUT/"submission_v47_primary.csv",index=False)
primary_sub.to_csv(OUT_DIR/"submission_v47_primary.csv",index=False)

V47_SPLIT_MANIFEST.to_csv(V47_OUT/"v47_split_manifest.csv",index=False); V47_SPLIT_AUDIT.to_csv(V47_OUT/"v47_split_audit.csv",index=False)
V47_ORACLE.to_csv(V47_OUT/"v47_oracle.csv",index=False)
V47_ALPHA_AUDIT.to_csv(V47_OUT/"v47_alpha_selection.csv",index=False); V47_HEAD_FIT_AUDIT.to_csv(V47_OUT/"v47_head_fit_audit.csv",index=False)
V47_OOF_ACTIONS.to_csv(V47_OUT/"v47_action_train_oof_actions.csv",index=False); V47_ACTIONTRAIN_RISK.to_csv(V47_OUT/"v47_action_train_risk_coverage.csv",index=False)
V47_ACTIONTRAIN_SELECTED.to_csv(V47_OUT/"v47_action_train_selected.csv",index=False)
V47_DESIGN_SINGLE.to_csv(V47_OUT/"v47_design_single_expert_comparison.csv",index=False); V47_DESIGN_METRICS.to_csv(V47_OUT/"v47_design_metrics.csv",index=False)
V47_DESIGN_LEDGER.to_csv(V47_OUT/"v47_design_action_ledger.csv",index=False); V47_DESIGN_FOLDS.to_csv(V47_OUT/"v47_design_fold_metrics.csv",index=False); V47_DESIGN_BLOCKS.to_csv(V47_OUT/"v47_design_id_blocks.csv",index=False)
V47_LOCK_SINGLE.to_csv(V47_OUT/"v47_lock_single_expert_comparison.csv",index=False); V47_LOCK_METRICS.to_csv(V47_OUT/"v47_lock_metrics.csv",index=False)
V47_LOCK_LEDGER.to_csv(V47_OUT/"v47_lock_action_ledger.csv",index=False); V47_LOCK_FOLDS.to_csv(V47_OUT/"v47_lock_fold_metrics.csv",index=False); V47_LOCK_BLOCKS.to_csv(V47_OUT/"v47_lock_id_blocks.csv",index=False)
V47_TEST_LEDGER.to_csv(V47_OUT/"v47_test_action_ledger.csv",index=False); pd.DataFrame([V47_TEST_GEOMETRY]).to_csv(V47_OUT/"v47_test_geometry.csv",index=False)

versions={"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__,"lightgbm":lgb.__version__}
try:
    versions.update({"torch":torch.__version__,"transformers":transformers.__version__,"cuda":torch.version.cuda,"gpu_count":torch.cuda.device_count(),
                     "sentence_transformers":importlib.metadata.version("sentence-transformers"),"sentencepiece":importlib.metadata.version("sentencepiece"),"tiktoken":importlib.metadata.version("tiktoken")})
except Exception: pass
json.dump(versions,open(V47_OUT/"v47_package_versions.json","w"),indent=2)

repro={"version":V47_VERSION,"organizer_data_only":True,"external_labeled_data":False,"open_weight_pretrained_encoders_only":True,
       "no_direct_genai_test_relevance":True,"champion_frozen":True,"s0_only":True,"top1_only":True,"top4_membership_frozen":True,
       "top5_membership_frozen":True,"ranks5_to17_frozen":True,"generator_veto_removed":True,"structural_control_has_no_authority":True,
       "public_probe_users_protected":sorted(PUBLIC_PROBE_TEST_USERS),"action_target":"exact counterfactual Top1 delta NDCG@5",
       "split_seed":V47_SPLIT_SEED,"struct_seed":V47_STRUCT_SEED,"head_seed":V47_HEAD_SEED,"coverage_grid":COVERAGE_GRID,
       "selected_coverage":SELECTED_COVERAGE,"ridge_alpha_grid":RIDGE_ALPHA_GRID,"selected_alphas":V47_GLOBAL_ALPHA,
       "e5_model_id":E5_MODEL_ID,"e5_revision":E5_REVISION,"nomic_model_id":NOMIC_MODEL_ID,"nomic_revision":NOMIC_REVISION,"nomic_code_revision":NOMIC_CODE_REVISION,
       "anchor_top5_sha256":top5_sha256(V44_ANCHOR_TEST),"expected_anchor_top5_sha256":EXPECTED_V47_ANCHOR_HASH,
       "oracle_gate":bool(ORACLE_GATE),"deep_env_ready":bool(DEEP_ENV_READY),"deep_env_error":DEEP_ENV_ERROR,
       "embeddings_ready":bool(EMBEDDINGS_READY),"embedding_error":EMBED_ERROR,"policy_ready":bool(POLICY_READY),
       "design_evaluated":bool(DESIGN_EVALUATED),"design_gate":bool(DESIGN_GATE),"lock_evaluated":bool(LOCK_EVALUATED),"lock_gate":bool(LOCK_GATE),
       "test_embeddings_ready":bool(TEST_EMBEDDINGS_READY),"test_embedding_error":TEST_EMBED_ERROR,"final_gate":bool(FINAL_GATE)}
json.dump(repro,open(V47_OUT/"v47_reproduction_audit.json","w"),indent=2)

instruction=("SUBMIT /kaggle/working/submission_v47_primary.csv. V47 passed prospective ACTION_TRAIN, blind DESIGN, one-shot LOCK, and exact test geometry gates."
             if FINAL_GATE else
             "DO NOT spend a leaderboard slot on V47. submission_v47_primary.csv failed closed exactly to the 0.66028 champion; inspect v47_summary.json and the first failed gate.")
summary={"version":V47_VERSION,"public_anchor":PUBLIC_V47_ANCHOR_LB,"public_top5_cutoff":PUBLIC_V47_TOP5_CUTOFF,"public_leader":PUBLIC_V47_LEADER,
         "oracle_delta":float(V47_ORACLE.iloc[0].top1_within_top4_oracle_delta),"oracle_gate":bool(ORACLE_GATE),"embeddings_ready":bool(EMBEDDINGS_READY),
         "policy_ready":bool(POLICY_READY),"selected_coverage":SELECTED_COVERAGE,
         "design_evaluated":bool(DESIGN_EVALUATED),"design_delta":float(V47_DESIGN_METRICS.iloc[0].get("delta",0.0)),"design_gate":bool(DESIGN_GATE),
         "lock_evaluated":bool(LOCK_EVALUATED),"lock_delta":float(V47_LOCK_METRICS.iloc[0].get("delta",0.0)) if LOCK_EVALUATED else None,
         "pooled_design_lock_delta":float(POOLED_DELTA) if np.isfinite(POOLED_DELTA) else None,"lock_gate":bool(LOCK_GATE),
         "test_actions":int(V47_TEST_GEOMETRY.get("actions",0)),"test_geometry_gate":bool(V47_TEST_GEOMETRY.get("geometry_gate",False)),"final_gate":bool(FINAL_GATE),
         "anchor_top5_sha256":top5_sha256(V44_ANCHOR_TEST),"candidate_top5_sha256":top5_sha256(TEST_CANDIDATE),"primary_top5_sha256":top5_sha256(PRIMARY),
         "submission_instruction":instruction}
json.dump(summary,open(V47_OUT/"v47_summary.json","w"),indent=2)

assert top5_sha256(V44_ANCHOR_TEST)==EXPECTED_V47_ANCHOR_HASH
if FINAL_GATE: assert top5_sha256(PRIMARY)==top5_sha256(TEST_CANDIDATE)
else:
    assert np.array_equal(PRIMARY,V44_ANCHOR_TEST); assert top5_sha256(PRIMARY)==EXPECTED_V47_ANCHOR_HASH
assert np.array_equal(primary_sub[MODULES].to_numpy(np.float32),PRIMARY.astype(np.float32))
print(json.dumps(summary,indent=2)); print("Artifacts:",V47_OUT); print("SUBMISSION DECISION:",instruction)

# ---- cell 44 (markdown) ----
# ### Terminal interpretation contract
#
# - If `final_gate == true`, use **only** `/kaggle/working/submission_v47_primary.csv` for the V47 leaderboard slot.
# - If `final_gate == false`, do **not** submit V47. The primary file is deliberately the exact `0.66028` champion.
# - A tokenizer/model/runtime failure is an implementation failure, not evidence against the E5/Nomic hypothesis.
# - A negative ACTION_TRAIN competence frontier closes the representation branch before DESIGN.
# - A positive DESIGN followed by negative LOCK is a validation-transport failure. Do not retune on LOCK.
# - If V47 passes LOCK but the public score is flat/negative, record the package once and do not perform test-user micro-ablation.
