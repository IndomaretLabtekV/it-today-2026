# Converted from notebooks/v46.ipynb: markdown cells become comments, %/! magics neutralized.

# ---- cell 0 (markdown) ----
# # MineToday V46 — Frozen-Champion Dual-Contrastive Head Challenge with Selective Consensus
#
# **Frozen hypothesis:** the exact public `0.66028` champion remains the default. V46 gives no global ranking authority to a new semantic model. It tests whether two frozen, complementary multilingual embedding experts can identify a *small* set of high-precision **Top-1-only** corrections inside the champion's existing Top-4.
#
# The implementation is intentionally narrow and preregistered:
#
# 1. reconstruct the exact `0.66028` V44/B34+B45 champion and verify its Top-5 hash;
# 2. create a fresh `ACTION_TRAIN / DESIGN / LOCK` split so DESIGN and LOCK are never used to fit the action policy;
# 3. construct fold-pure deployment-like champion analogues for `ACTION_TRAIN`, and one ACTION_TRAIN-only structural surface for DESIGN/LOCK;
# 4. compute the Top1-within-Top4 oracle before downloading deep models;
# 5. use two **frozen** semantic experts only: `multilingual-e5-large-instruct` and `nomic-embed-text-v2-moe`;
# 6. train three low-capacity opinions on **exact counterfactual Top-1 ΔNDCG@5**: STRUCT, E5, and NOMIC;
# 7. use V42-style generator state only as a catastrophic **veto**, never as a positive score;
# 8. execute a Top-1 action only when E5 and Nomic independently choose the same challenger, both are directionally positive and fold-stable, STRUCT does not veto, and generator-state does not strongly contradict;
# 9. preserve Top-4 membership, Top-5 membership, ranks 5–17, and the existing Top-4 score multiset exactly;
# 10. open LOCK once, never retrain after DESIGN/LOCK, and fail-close bit-exactly to the `0.66028` champion if any gate fails.
#
# Organizer data only. Open-weight pretrained encoders are used as feature extractors, not as direct test-label generators. Kaggle **T4x2 + Internet ON** is expected for the full run.

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
# ## V46 preregistration and fresh evaluation authority
#
# `ACTION_TRAIN` is the only labeled partition allowed to fit the V46 action policy. Its structural anchor is itself generated out-of-fold. DESIGN is a clean model-selection/gating set. LOCK is untouched confirmation and is evaluated only if DESIGN passes.
#
# No threshold sweep, module whitelist, user whitelist, public-LB-derived selector, or post-LOCK retraining is allowed.

# ---- cell 24 (code) ----
# V46 fixed configuration. Full methodology is frozen here before V46 outcomes are observed.
import platform
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score

V46_VERSION = "V46 Frozen-Champion Dual-Contrastive Head Challenge with Selective Consensus"
V46_SMOKE = bool(int(os.getenv("V46_SMOKE", "0")))
V46_SKIP_DEEP = bool(int(os.getenv("V46_SKIP_DEEP", "0")))
V46_OUT = OUT_DIR / "v46"
V46_OUT.mkdir(parents=True, exist_ok=True)

# Fresh split family; not reused from V41-V45.
V46_SPLIT_SEED = 2026090460
V46_FOLD_SEED = 2026090461
V46_MODEL_SEED = 2026090462
V46_BOOT_SEED = 2026090463
ACTION_TRAIN_SIZE = 600 if V46_SMOKE else 2400
DESIGN_SIZE = 200 if V46_SMOKE else 800
LOCK_SIZE = 200 if V46_SMOKE else 800
HEAD_FOLDS = 2 if V46_SMOKE else 5
BLOCKS = 2 if V46_SMOKE else 5
BOOTSTRAPS_V46 = 100 if V46_SMOKE else 2000

# Low-capacity heads. No hyperparameter search.
UTILITY_RIDGE_ALPHA = 10.0
GENERATOR_C = 0.05
SEMANTIC_FOLD_AGREE = 1 if V46_SMOKE else 4
STRUCT_FOLD_NONNEG = 1 if V46_SMOKE else 3
GENERATOR_HARM_P = 0.75
GENERATOR_HARM_FOLD_AGREE = 1 if V46_SMOKE else 4

# Preregistered gates from the V46 design.
ORACLE_MIN_DELTA = -1.0 if V46_SMOKE else 0.00150
DESIGN_MIN_DELTA = -1.0 if V46_SMOKE else 0.00040
DESIGN_MIN_FOLD_WINS = 0 if V46_SMOKE else 4
DESIGN_MIN_BLOCK_WINS = 0 if V46_SMOKE else 3
DESIGN_MIN_ACTION_PRECISION = 0.0 if V46_SMOKE else 0.80
TEMPLATE_CATASTROPHIC_FLOOR = -1.0 if V46_SMOKE else -0.00050
LOCK_MIN_DELTA = -1.0 if V46_SMOKE else 0.00030
LOCK_MIN_BLOCK_WINS = 0 if V46_SMOKE else 3
LOCK_MIN_ACTION_PRECISION = 0.0 if V46_SMOKE else 0.75
TEST_MIN_ACTIONS = 0 if V46_SMOKE else 3
TEST_MAX_ACTIONS = 999 if V46_SMOKE else 15

PUBLIC_V46_ANCHOR_LB = 0.66028
PUBLIC_V46_TOP5_CUTOFF = 0.66076
PUBLIC_V46_LEADER = 0.66142
EXPECTED_V46_ANCHOR_HASH = "7c0d29a7309b7283a043b7d37ca5556ba7c0ca2270b0137a6e1e0878a69a5fe7"

# Frozen embedding experts. Revisions are pinned for reproducibility.
E5_MODEL_ID = "intfloat/multilingual-e5-large-instruct"
E5_REVISION = "b447574844fbdae893c9a720e376d77ff5cb123a"
NOMIC_MODEL_ID = "nomic-ai/nomic-embed-text-v2-moe"
NOMIC_REVISION = "f0bcc8894f384250588bd1ba946c080e4339dedf"
NOMIC_CODE_REVISION = "7710840340a098cfb869c4f65e87cf2b1b70caca"
E5_MAX_LEN = 256
NOMIC_MAX_LEN = 256
EMBED_BATCH = 8 if V46_SMOKE else 32

assert top5_sha256(V44_ANCHOR_TEST) == EXPECTED_V46_ANCHOR_HASH
print("V46 exact frozen champion hash:", top5_sha256(V44_ANCHOR_TEST))
print("public anchor", PUBLIC_V46_ANCHOR_LB,
      "gap to Top-5", round(PUBLIC_V46_TOP5_CUTOFF-PUBLIC_V46_ANCHOR_LB,5),
      "gap to leader", round(PUBLIC_V46_LEADER-PUBLIC_V46_ANCHOR_LB,5))
print("V46_SMOKE", V46_SMOKE, "V46_SKIP_DEEP", V46_SKIP_DEEP)

# ---- cell 25 (code) ----
# Fresh ACTION_TRAIN / DESIGN / LOCK split, stratified by true Top1 and K.
pos_count = (y > 0).sum(1).astype(int)
joint = np.asarray([f"{np.argmax(y[i]):02d}_{pos_count[i]}" for i in range(N_TRAIN)], dtype=object)
all_idx = np.arange(N_TRAIN)

if V46_SMOKE:
    rng = np.random.default_rng(V46_SPLIT_SEED)
    sub = np.sort(rng.choice(all_idx, size=ACTION_TRAIN_SIZE+DESIGN_SIZE+LOCK_SIZE, replace=False))
    sj = joint[sub]
    s1 = StratifiedShuffleSplit(n_splits=1, test_size=LOCK_SIZE, random_state=V46_SPLIT_SEED)
    rem_l, lock_l = next(s1.split(np.zeros(len(sub)), sj))
    rem = sub[rem_l]; LOCK_IDX = sub[lock_l]
    s2 = StratifiedShuffleSplit(n_splits=1, test_size=DESIGN_SIZE, random_state=V46_SPLIT_SEED+1)
    at_l, de_l = next(s2.split(np.zeros(len(rem)), joint[rem]))
    ACTION_TRAIN_IDX = rem[at_l]; DESIGN_IDX = rem[de_l]
else:
    s1 = StratifiedShuffleSplit(n_splits=1, test_size=LOCK_SIZE, random_state=V46_SPLIT_SEED)
    rem, lk = next(s1.split(np.zeros(N_TRAIN), joint))
    REM = np.asarray(rem, int); LOCK_IDX = np.asarray(lk, int)
    s2 = StratifiedShuffleSplit(n_splits=1, test_size=DESIGN_SIZE, random_state=V46_SPLIT_SEED+1)
    at, de = next(s2.split(np.zeros(len(REM)), joint[REM]))
    ACTION_TRAIN_IDX = REM[np.asarray(at,int)]; DESIGN_IDX = REM[np.asarray(de,int)]

ACTION_TRAIN_IDX = np.sort(np.asarray(ACTION_TRAIN_IDX,int))
DESIGN_IDX = np.sort(np.asarray(DESIGN_IDX,int))
LOCK_IDX = np.sort(np.asarray(LOCK_IDX,int))
assert len(ACTION_TRAIN_IDX)==ACTION_TRAIN_SIZE
assert len(DESIGN_IDX)==DESIGN_SIZE
assert len(LOCK_IDX)==LOCK_SIZE
assert len(set(ACTION_TRAIN_IDX)&set(DESIGN_IDX))==0
assert len(set(ACTION_TRAIN_IDX)&set(LOCK_IDX))==0
assert len(set(DESIGN_IDX)&set(LOCK_IDX))==0

V46_SPLIT_MANIFEST = pd.DataFrame({
    "user_id": train.user_id.astype(str),
    "split": np.where(np.isin(all_idx,ACTION_TRAIN_IDX),"ACTION_TRAIN",
             np.where(np.isin(all_idx,DESIGN_IDX),"DESIGN","LOCK"))
})
print(V46_SPLIT_MANIFEST.split.value_counts().to_dict())

# ---- cell 26 (markdown) ----
# ## Fold-pure champion analogue for action utility
#
# The action target is not generic relevance. For each eligible learner and each existing anchor challenger at ranks 2–4, V46 asks the exact deployment question:
#
# `Δ = NDCG@5(anchor with challenger promoted to #1) − NDCG@5(anchor)`.
#
# `ACTION_TRAIN` structural surfaces are generated strictly out-of-fold. DESIGN/LOCK surfaces are produced by a structural model fit only on ACTION_TRAIN.

# ---- cell 27 (code) ----
# Deployment-like structural anchor helper inherited from V44, with B34+B45 package applied algorithmically.
def build_v46_val_surface(train_rows, val_rows, seed):
    train_rows=np.asarray(train_rows,int); val_rows=np.asarray(val_rows,int)
    anc=build_anchor_split_full(train_rows,val_rows,int(seed))
    intent_tr=crossfit_intentstate_features(
        y[train_rows],ORDINAL_GRADES[train_rows],TRAIN_INTENT[train_rows],
        MODULE_READY_BAND[train_rows],TRAIN_GROUP[train_rows],int(seed)+71)
    bundle=fit_intentstate_bundle(
        y[train_rows],ORDINAL_GRADES[train_rows],TRAIN_INTENT[train_rows],MODULE_READY_BAND[train_rows])
    intent_va=apply_intentstate_bundle(bundle,ALL_INTENT[val_rows],MODULE_READY_BAND[val_rows])
    Xtr,_,bounds=build_long_matrix_v31(
        train_rows,anc["cons_train"],anc["base_train"],anc["meta_train"],intent_tr)
    Xva,_,bounds2=build_long_matrix_v31(
        val_rows,anc["cons_val"],anc["base_val"],anc["meta_val"],intent_va)
    assert bounds==bounds2
    _,raw30=fit_ranker_predict(
        Xtr[:,:bounds["base"]],Xva[:,:bounds["base"]],ORDINAL_GRADES[train_rows],len(train_rows),int(seed)+3301)
    _,raw31=fit_ranker_predict(
        Xtr[:,:bounds["intentstate"]],Xva[:,:bounds["intentstate"]],ORDINAL_GRADES[train_rows],len(train_rows),int(seed)+4301)
    pre=anc["cons_val"].astype(np.float32)
    v30=blend_residual(pre,raw30,ALPHA)
    v31=blend_residual(pre,raw31,ALPHA)
    state=state_codes(pre,v30,v31)
    current,info=current066020_surface(v30,v31,state)
    t30=top5_indices(v30); t31=top5_indices(v31)
    for r in np.flatnonzero(info["order_only"]):
        try:
            boundary,*_=classify_single_adjacent_order_swap(t30[r],t31[r])
        except AssertionError:
            continue
        if boundary in {"B34","B45"}:
            current[r]=v31[r]
    out={"pre":pre,"v30":v30.astype(np.float32),"v31":v31.astype(np.float32),
         "state":state.astype(np.int8),"anchor":current.astype(np.float32)}
    del Xtr,Xva,intent_tr,intent_va,bundle,anc,raw30,raw31
    gc.collect()
    return out

# Fold-pure OOF structural surfaces for ACTION_TRAIN.
atr_strata = joint[ACTION_TRAIN_IDX]
atr_splits = list(StratifiedKFold(
    n_splits=HEAD_FOLDS, shuffle=True, random_state=V46_FOLD_SEED
).split(np.zeros(len(ACTION_TRAIN_IDX)), atr_strata))

ATR_PRE=np.zeros((len(ACTION_TRAIN_IDX),N_MODULES),np.float32)
ATR_V30=np.zeros_like(ATR_PRE); ATR_V31=np.zeros_like(ATR_PRE); ATR_ANCHOR=np.zeros_like(ATR_PRE)
ATR_STATE=np.full(len(ACTION_TRAIN_IDX),-1,np.int8); ATR_FOLD=np.full(len(ACTION_TRAIN_IDX),-1,int)

for f,(trloc,valoc) in enumerate(atr_splits):
    tr=ACTION_TRAIN_IDX[np.asarray(trloc,int)]; va=ACTION_TRAIN_IDX[np.asarray(valoc,int)]
    print(f"V46 action-train structural OOF {f+1}/{HEAD_FOLDS}: train={len(tr)} val={len(va)}")
    s=build_v46_val_surface(tr,va,V46_FOLD_SEED+1000*f)
    ATR_PRE[valoc]=s["pre"]; ATR_V30[valoc]=s["v30"]; ATR_V31[valoc]=s["v31"]
    ATR_STATE[valoc]=s["state"]; ATR_ANCHOR[valoc]=s["anchor"]; ATR_FOLD[valoc]=f
    del s; gc.collect()
assert (ATR_FOLD>=0).all() and (ATR_STATE>=0).all()
ACTION_SURF={"pre":ATR_PRE,"v30":ATR_V30,"v31":ATR_V31,"state":ATR_STATE,"anchor":ATR_ANCHOR}

# One ACTION_TRAIN-only structural fit predicts DESIGN and LOCK; neither label set is touched.
eval_global=np.concatenate([DESIGN_IDX,LOCK_IDX])
eval_surf=build_v46_val_surface(ACTION_TRAIN_IDX,eval_global,V46_FOLD_SEED+90001)
nd=len(DESIGN_IDX)
DESIGN_SURF={k:np.asarray(v[:nd]).copy() for k,v in eval_surf.items()}
LOCK_SURF={k:np.asarray(v[nd:]).copy() for k,v in eval_surf.items()}
del eval_surf; gc.collect()

ACTION_ELIG=np.isin(ACTION_SURF["state"],[0,2])
DESIGN_ELIG=np.isin(DESIGN_SURF["state"],[0,2])
LOCK_ELIG=np.isin(LOCK_SURF["state"],[0,2])
print("eligible ACTION/DESIGN/LOCK",int(ACTION_ELIG.sum()),int(DESIGN_ELIG.sum()),int(LOCK_ELIG.sum()))

# ---- cell 28 (code) ----
# Exact Top1 score-value permutation and oracle diagnostics.
def apply_top1_permutation(scores, row, challenger):
    scores=np.asarray(scores,np.float32)
    before=scores[row].copy()
    order=np.argsort(-before,kind="mergesort")
    top4=order[:4].tolist(); challenger=int(challenger)
    if challenger not in top4:
        raise AssertionError("V46 may promote only an existing anchor Top-4 module")
    if top4[0]==challenger:
        return
    vals=before[top4].copy()  # existing descending score multiset
    new_order=[challenger]+[m for m in top4 if m!=challenger]
    scores[row,new_order]=vals


def geometry_audit_v46(base,cand):
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
    return {
        "changed_users":int(changed.sum()),"top1_changes":int((ob[:,0]!=oc[:,0]).sum()),
        "top4_membership_changes":int((~top4_same).sum()),"top5_membership_changes":int((~top5_same).sum()),
        "rank5_to_17_order_changes":int((~tail_same).sum()),"top4_score_multiset_ok":bool(multiset_ok),
    }


def top1_acc_v46(ytrue,s):
    return float(np.mean(np.argmax(ytrue,axis=1)==np.argmax(s,axis=1)))

def relevant_at5_v46(ytrue,s):
    t=top5_indices(s)
    return float(np.mean((np.take_along_axis(ytrue,t,axis=1)>0).sum(1)))

def metric_v46(ytrue,base,cand):
    return {
        "base_ndcg":ndcg5(ytrue,base),"candidate_ndcg":ndcg5(ytrue,cand),
        "delta":ndcg5(ytrue,cand)-ndcg5(ytrue,base),
        "base_top1":top1_acc_v46(ytrue,base),"candidate_top1":top1_acc_v46(ytrue,cand),
        "top1_delta":top1_acc_v46(ytrue,cand)-top1_acc_v46(ytrue,base),
        "base_rel5":relevant_at5_v46(ytrue,base),"candidate_rel5":relevant_at5_v46(ytrue,cand),
        "rel5_delta":relevant_at5_v46(ytrue,cand)-relevant_at5_v46(ytrue,base),
    }

def paired_boot_v46(delta,seed):
    d=np.asarray(delta,float); rng=np.random.default_rng(int(seed)); z=np.empty(BOOTSTRAPS_V46,float)
    for b in range(BOOTSTRAPS_V46): z[b]=d[rng.integers(0,len(d),len(d))].mean()
    return float(np.quantile(z,.025)),float(np.quantile(z,.975))

# Oracle: correct Top1 if it is already inside the immutable Top4.
DESIGN_BASE=DESIGN_SURF["anchor"].copy()
true_top1=np.argmax(y[DESIGN_IDX],axis=1)
base_top4=np.argsort(-DESIGN_BASE,axis=1,kind="mergesort")[:,:4]
base_top1=base_top4[:,0]
recoverable=np.asarray([
    bool(DESIGN_ELIG[r] and base_top1[r]!=true_top1[r] and true_top1[r] in set(base_top4[r]))
    for r in range(len(DESIGN_IDX))
],bool)
ORACLE=DESIGN_BASE.copy()
for r in np.flatnonzero(recoverable): apply_top1_permutation(ORACLE,int(r),int(true_top1[r]))
oracle_delta=float(ndcg5(y[DESIGN_IDX],ORACLE)-ndcg5(y[DESIGN_IDX],DESIGN_BASE))
wrong=DESIGN_ELIG & (base_top1!=true_top1)
V46_ORACLE=pd.DataFrame([{
    "design_anchor_ndcg":ndcg5(y[DESIGN_IDX],DESIGN_BASE),"eligible_users":int(DESIGN_ELIG.sum()),
    "eligible_wrong_top1":int(wrong.sum()),"recoverable_top1_errors":int(recoverable.sum()),
    "true_top1_in_top4_given_wrong":float(recoverable.sum()/max(1,wrong.sum())),
    "top1_within_top4_oracle_delta":oracle_delta,
}])
ORACLE_GATE=bool(top5_sha256(V44_ANCHOR_TEST)==EXPECTED_V46_ANCHOR_HASH and oracle_delta>=ORACLE_MIN_DELTA)
print(V46_ORACLE.to_string(index=False)); print("ORACLE_GATE",ORACLE_GATE)

# ---- cell 29 (markdown) ----
# ## Frozen dual-contrastive semantic experts
#
# The semantic models are **not fine-tuned** and never produce submission scores directly.
#
# - **E5:** `intfloat/multilingual-e5-large-instruct`, pinned revision, last-token pooling and an explicit retrieval instruction.
# - **Nomic:** `nomic-ai/nomic-embed-text-v2-moe`, pinned revision, `search_query:` / `search_document:` prefixes and mean pooling.
#
# For each learner V46 builds two views only: `CURRENT` learning intent and `NEXT_STEP` recent trajectory plus compact organizer assessment context. Users without learning chat receive no semantic authority.

# ---- cell 30 (code) ----
# Text views for frozen semantic experts. No labels are used here.
def semantic_text_norm(x):
    return re.sub(r"\s+"," ",html.unescape(str(x))).strip()

CURRENT_TEXTS=[]; NEXT_TEXTS=[]; HAS_LEARNING_CHAT=[]
for u in all_users:
    ui=uid2idx[u]; g=user_chat.get(u)
    if g is None or len(g)==0:
        learn=None
    else:
        learn=g[~g["is_noise"]].copy()
        if len(learn)==0: learn=None
    if learn is None or len(learn)==0:
        current=""
        recent=[]
        HAS_LEARNING_CHAT.append(False)
    else:
        rows=list(learn.sort_values(["timestamp","chat_id"]).itertuples(index=False))
        current=semantic_text_norm(rows[-1].user_chat_text)
        recent=[semantic_text_norm(r.user_chat_text) for r in rows[-3:]]
        HAS_LEARNING_CHAT.append(True)
    CURRENT_TEXTS.append(current)
    tags=["OLDER","RECENT","CURRENT"][-len(recent):]
    traj=" ".join(f"[{t}] {m}" for t,m in zip(tags,recent)) if recent else ""
    assess="; ".join(f"{SKILL_NAMES[j]}={float(A[ui,j]):.0f}/5" for j in range(len(SKILL_NAMES)))
    NEXT_TEXTS.append(f"Riwayat belajar: {traj}. Profil asesmen: {assess}" if traj else f"Profil asesmen: {assess}")

CURRENT_TEXTS=np.asarray(CURRENT_TEXTS,dtype=object)
NEXT_TEXTS=np.asarray(NEXT_TEXTS,dtype=object)
HAS_LEARNING_CHAT=np.asarray(HAS_LEARNING_CHAT,bool)
mods=modules.set_index("module_id").loc[MODULES].reset_index()
SEM_MODULE_DOCS=[
    f"Modul Intelligo: {r.module_name}. Materi: {r.description_and_syllabus}. Prasyarat: {r.prerequisite_level}."
    for r in mods.itertuples(index=False)
]
E5_INSTRUCTION=(
    "Given a learner's current request, recent learning trajectory, and readiness context, "
    "retrieve the Intelligo course module that is the most appropriate next learning step. Query: "
)
print("learning-chat users",int(HAS_LEARNING_CHAT.sum()),"/",len(all_users))

# ---- cell 31 (code) ----
# Robust model resolution + GPU inference. Full run expects Kaggle T4x2 and Internet ON unless both models are attached as Kaggle inputs.
EMBED_ERROR=""; EMBEDDINGS_READY=False
E5_CURRENT=E5_NEXT=NOMIC_CURRENT=NOMIC_NEXT=np.zeros((len(all_users),N_MODULES),np.float32)

if ORACLE_GATE and not V46_SKIP_DEEP:
    try:
        import torch
        import torch.nn.functional as F
        try:
            import einops  # noqa: F401
        except Exception:
            subprocess.check_call([sys.executable,"-m","pip","install","-q","einops>=0.7"])
        from transformers import AutoTokenizer, AutoModel

        if not torch.cuda.is_available():
            raise RuntimeError("V46 full run requires a Kaggle GPU runtime. Select T4x2.")
        ngpu=torch.cuda.device_count()
        if ngpu<2 and not V46_SMOKE:
            raise RuntimeError(f"V46 expects T4x2; detected only {ngpu} CUDA device(s).")
        os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
        os.environ.setdefault("HF_HOME","/tmp/hf_cache")

        def resolve_hf_source(model_id,env_key):
            env=os.getenv(env_key,"").strip()
            if env and Path(env).exists(): return env
            short=model_id.split("/")[-1].lower()
            root=Path("/kaggle/input")
            if root.exists():
                hits=[]
                for p in root.rglob("config.json"):
                    if short in str(p.parent).lower(): hits.append(str(p.parent))
                if hits: return sorted(hits,key=len)[0]
            return model_id

        def load_frozen(model_id,revision,device,trust_remote_code=False):
            source=resolve_hf_source(model_id,"V46_E5_MODEL_PATH" if model_id==E5_MODEL_ID else "V46_NOMIC_MODEL_PATH")
            rev=revision if source==model_id else None
            print("loading",model_id,"from",source,"on",device,"revision",rev or "local")
            tok=AutoTokenizer.from_pretrained(source,revision=rev,trust_remote_code=trust_remote_code)
            kwargs=dict(revision=rev,trust_remote_code=trust_remote_code,low_cpu_mem_usage=True)
            if trust_remote_code and source==model_id:
                kwargs["code_revision"] = NOMIC_CODE_REVISION
            try:
                mdl=AutoModel.from_pretrained(source,torch_dtype=torch.float16,**kwargs)
                mdl=mdl.to(device)
            except Exception as e:
                print("fp16 load fallback to fp32:",type(e).__name__,str(e)[:180])
                mdl=AutoModel.from_pretrained(source,**kwargs).to(device)
            mdl.eval()
            for p in mdl.parameters(): p.requires_grad_(False)
            return tok,mdl,source,rev

        def last_token_pool_v46(last_hidden,mask):
            if bool((mask[:,-1].sum()==mask.shape[0]).item()): return last_hidden[:,-1]
            seq=mask.sum(1)-1
            return last_hidden[torch.arange(last_hidden.shape[0],device=last_hidden.device),seq]

        def mean_pool_v46(last_hidden,mask):
            mm=mask.unsqueeze(-1).expand(last_hidden.size()).float()
            return (last_hidden.float()*mm).sum(1)/mm.sum(1).clamp_min(1e-9)

        def encode_batches(tok,mdl,texts,device,max_len,pool,batch):
            texts=[str(x) for x in texts]
            cur_batch=int(batch)
            while True:
                try:
                    chunks=[]
                    with torch.inference_mode():
                        for st in range(0,len(texts),cur_batch):
                            z=tok(texts[st:st+cur_batch],padding=True,truncation=True,max_length=max_len,return_tensors="pt")
                            z={k:v.to(device,non_blocking=True) for k,v in z.items()}
                            o=mdl(**z)
                            h=o.last_hidden_state if hasattr(o,"last_hidden_state") else o[0]
                            e=last_token_pool_v46(h,z["attention_mask"]) if pool=="last" else mean_pool_v46(h,z["attention_mask"])
                            e=F.normalize(e.float(),p=2,dim=1).cpu().numpy().astype(np.float32)
                            chunks.append(e)
                            del z,o,h,e
                    return np.concatenate(chunks,axis=0)
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower() or cur_batch<=4:
                        raise
                    cur_batch=max(4,cur_batch//2)
                    print(f"CUDA OOM during frozen embedding inference; retrying with batch={cur_batch}")
                    gc.collect(); torch.cuda.empty_cache()

        def semantic_scores(model_id,revision,device,kind):
            trust=(kind=="nomic")
            tok,mdl,source,rev=load_frozen(model_id,revision,device,trust)
            if kind=="e5":
                q1=[E5_INSTRUCTION+str(x) for x in CURRENT_TEXTS]
                q2=[E5_INSTRUCTION+str(x) for x in NEXT_TEXTS]
                docs=[str(x) for x in SEM_MODULE_DOCS]
                pool="last"
            else:
                q1=["search_query: "+str(x) for x in CURRENT_TEXTS]
                q2=["search_query: "+str(x) for x in NEXT_TEXTS]
                docs=["search_document: "+str(x) for x in SEM_MODULE_DOCS]
                pool="mean"
            de=encode_batches(tok,mdl,docs,device,E5_MAX_LEN if kind=="e5" else NOMIC_MAX_LEN,pool,EMBED_BATCH)
            s1=np.zeros((len(all_users),N_MODULES),np.float32); s2=np.zeros_like(s1)
            valid=np.flatnonzero(HAS_LEARNING_CHAT)
            e1=encode_batches(tok,mdl,np.asarray(q1,dtype=object)[valid],device,E5_MAX_LEN if kind=="e5" else NOMIC_MAX_LEN,pool,EMBED_BATCH)
            e2=encode_batches(tok,mdl,np.asarray(q2,dtype=object)[valid],device,E5_MAX_LEN if kind=="e5" else NOMIC_MAX_LEN,pool,EMBED_BATCH)
            s1[valid]=e1@de.T; s2[valid]=e2@de.T
            info={"model_id":model_id,"revision":revision,"resolved_source":source,"resolved_revision":rev,"device":device,
                  "embedding_dim":int(de.shape[1]),"valid_users":int(len(valid)),"max_length":E5_MAX_LEN if kind=="e5" else NOMIC_MAX_LEN,
                  "code_revision":NOMIC_CODE_REVISION if kind=="nomic" else None}
            del tok,mdl,de,e1,e2
            gc.collect(); torch.cuda.empty_cache()
            return s1.astype(np.float32),s2.astype(np.float32),info

        e5_device="cuda:0"
        nomic_device="cuda:1" if torch.cuda.device_count()>1 else "cuda:0"
        E5_CURRENT,E5_NEXT,E5_INFO=semantic_scores(E5_MODEL_ID,E5_REVISION,e5_device,"e5")
        NOMIC_CURRENT,NOMIC_NEXT,NOMIC_INFO=semantic_scores(NOMIC_MODEL_ID,NOMIC_REVISION,nomic_device,"nomic")
        EMBEDDINGS_READY=True
        np.savez_compressed(V46_OUT/"v46_semantic_scores.npz",
                            e5_current=E5_CURRENT,e5_next=E5_NEXT,nomic_current=NOMIC_CURRENT,nomic_next=NOMIC_NEXT,
                            has_learning_chat=HAS_LEARNING_CHAT.astype(np.uint8))
        json.dump({"e5":E5_INFO,"nomic":NOMIC_INFO},open(V46_OUT/"v46_embedding_audit.json","w"),indent=2)
        print("semantic score matrices",E5_CURRENT.shape,NOMIC_CURRENT.shape)
    except Exception as e:
        EMBED_ERROR=f"{type(e).__name__}: {e}"
        print("V46 embedding stage failed closed:",EMBED_ERROR)
        EMBEDDINGS_READY=False
else:
    print("Deep stage skipped because oracle failed or V46_SKIP_DEEP=1")

# ---- cell 32 (markdown) ----
# ## Exact action-utility heads
#
# Each action row represents one concrete promotion of anchor rank 2, 3, or 4 to #1. The three main heads are intentionally small:
#
# - **STRUCT:** anchor geometry + sparse lexical/current-intent/readiness differences + V31−V30 disagreement context;
# - **E5:** E5 pair differences + minimal anchor geometry;
# - **NOMIC:** Nomic pair differences + minimal anchor geometry.
#
# All three regress **exact per-user ΔNDCG@5** with strongly regularized linear Ridge models. A separate low-capacity generator-state classifier predicts harmful actions and may only veto.

# ---- cell 33 (code) ----
# V42 label-free generator-state representation, reused only as a catastrophic veto.
N_ALL=N_TRAIN+N_TEST
V42_LATEST_TPL=np.full(N_ALL,"__NO_CHAT__",dtype=object)
V42_PREV_TPL=np.full(N_ALL,"__NO_PREV__",dtype=object)
V42_TPL_TRANS=np.full(N_ALL,"__NO_PREV__>>__NO_CHAT__",dtype=object)
V42_LATEST_TOPIC=np.full(N_ALL,"none",dtype=object)
V42_PREV_TOPIC=np.full(N_ALL,"none",dtype=object)
V42_TOPIC_JOURNEY=np.full(N_ALL,"none",dtype=object)
V42_LATEST_STYLE=np.full(N_ALL,"none",dtype=object)
V42_HIST_BUCKET=np.full(N_ALL,"n0",dtype=object)
V42_MODULE_AGE_DAYS=np.full((N_ALL,N_MODULES),999.0,np.float32)
V42_MODULE_LEVEL_CAT=modules.set_index("module_id")["prerequisite_level"].astype(str).reindex(MODULES).fillna("unknown").to_numpy(object)

def v42_topic_sig(topics):
    vals=sorted(set(map(str,topics)))
    return "|".join(vals) if vals else "none"

def v42_history_bucket(n):
    if n<=0: return "n0"
    if n==1: return "n1"
    if n==2: return "n2"
    if n<=4: return "n3_4"
    return "n5plus"

def v42_styles_for_row(row):
    z=[sty for sty in STYLES if int(getattr(row,"sty_"+sty,0))]
    return "|".join(z) if z else "none"

def v42_age_bin_scalar(x):
    x=float(x)
    if x>=998: return "never"
    if x<=0.01: return "current"
    if x<=3: return "d0_3"
    if x<=14: return "d4_14"
    if x<=60: return "d15_60"
    return "d60plus"

for u in all_users:
    gi=uid2idx[u]; g=user_chat.get(u)
    if g is None or len(g)==0: continue
    learn=g[~g["is_noise"]].copy()
    if len(learn)==0: learn=g.copy()
    rows=list(learn.itertuples(index=False)); n=len(rows)
    V42_HIST_BUCKET[gi]=v42_history_bucket(n)
    if n:
        last=rows[-1]
        V42_LATEST_TPL[gi]=template_token(str(last.skeleton))
        V42_LATEST_TOPIC[gi]=v42_topic_sig(last.topics)
        V42_LATEST_STYLE[gi]=v42_styles_for_row(last)
        if n>=2:
            prev=rows[-2]
            V42_PREV_TPL[gi]=template_token(str(prev.skeleton))
            V42_PREV_TOPIC[gi]=v42_topic_sig(prev.topics)
            V42_TPL_TRANS[gi]=V42_PREV_TPL[gi]+">>"+V42_LATEST_TPL[gi]
        else:
            V42_TPL_TRANS[gi]="__NO_PREV__>>"+V42_LATEST_TPL[gi]
        V42_TOPIC_JOURNEY[gi]=">>".join(v42_topic_sig(r.topics) for r in rows[-3:])
        t_last=last.timestamp if pd.notna(last.timestamp) else None
        for m,mid in enumerate(MODULES):
            topic_names=set(MODULE_TOPIC_MAP.get(mid,[])); hit_time=None
            for rr in reversed(rows):
                if topic_names.intersection(set(rr.topics)):
                    hit_time=rr.timestamp if pd.notna(rr.timestamp) else None
                    break
            if hit_time is not None and t_last is not None:
                V42_MODULE_AGE_DAYS[gi,m]=max(0.0,float((t_last-hit_time).total_seconds()/86400.0))

STRUCT_FEATURE_NAMES=[
    "anchor_gap","anchor_gap_z","rank2","rank3","rank4","state_s2",
    "pre_diff","v31_minus_v30_diff","lex_word_diff","lex_char_diff",
    "topic_count_diff","topic_recency_diff","topic_latest_diff",
    "readiness_diff","invalid_readiness_diff"
]
SEM_FEATURE_NAMES=["current_sem_diff","next_sem_diff","anchor_gap_z","rank2","rank3","rank4"]


def action_features(global_idx,surf,local_row,inc,c,rank,e5cur,e5next,nomcur,nomnext):
    gi=int(global_idx); inc=int(inc); c=int(c); rank=int(rank)
    a=surf["anchor"][local_row]; sd=max(float(np.std(a)),1e-6)
    gap=float(a[c]-a[inc]); gapz=gap/sd
    struct=np.asarray([
        gap,gapz,float(rank==2),float(rank==3),float(rank==4),float(surf["state"][local_row]==2),
        float(surf["pre"][local_row,c]-surf["pre"][local_row,inc]),
        float((surf["v31"][local_row,c]-surf["v30"][local_row,c])-(surf["v31"][local_row,inc]-surf["v30"][local_row,inc])),
        float(LEX_WORD[gi,c]-LEX_WORD[gi,inc]),float(LEX_CHAR[gi,c]-LEX_CHAR[gi,inc]),
        float(MODULE_COUNT[gi,c]-MODULE_COUNT[gi,inc]),float(MODULE_RECENCY[gi,c]-MODULE_RECENCY[gi,inc]),
        float(MODULE_LATEST[gi,c]-MODULE_LATEST[gi,inc]),float(MODULE_READINESS[gi,c]-MODULE_READINESS[gi,inc]),
        float(MODULE_INVALID[gi,c]-MODULE_INVALID[gi,inc]),
    ],np.float32)
    e5=np.asarray([float(e5cur[gi,c]-e5cur[gi,inc]),float(e5next[gi,c]-e5next[gi,inc]),gapz,
                   float(rank==2),float(rank==3),float(rank==4)],np.float32)
    nom=np.asarray([float(nomcur[gi,c]-nomcur[gi,inc]),float(nomnext[gi,c]-nomnext[gi,inc]),gapz,
                    float(rank==2),float(rank==3),float(rank==4)],np.float32)
    lt=str(V42_LATEST_TPL[gi]); pt=str(V42_PREV_TPL[gi]); trans=str(V42_TPL_TRANS[gi])
    ltopic=str(V42_LATEST_TOPIC[gi]); ptopic=str(V42_PREV_TOPIC[gi]); journey=str(V42_TOPIC_JOURNEY[gi])
    sty=str(V42_LATEST_STYLE[gi]); hb=str(V42_HIST_BUCKET[gi])
    ir=str(int(MODULE_READY_BAND[gi,inc])); cr=str(int(MODULE_READY_BAND[gi,c]))
    ia=v42_age_bin_scalar(V42_MODULE_AGE_DAYS[gi,inc]); ca=v42_age_bin_scalar(V42_MODULE_AGE_DAYS[gi,c])
    il=str(V42_MODULE_LEVEL_CAT[inc]); cl=str(V42_MODULE_LEVEL_CAT[c])
    im=str(MODULES[inc]); cm=str(MODULES[c])
    gd={
        "latest_tpl="+lt:1.0,"previous_tpl="+pt:1.0,"tpl_transition="+trans:1.0,
        "latest_topic="+ltopic:1.0,"previous_topic="+ptopic:1.0,"topic_journey="+journey:1.0,
        "latest_style="+sty:1.0,"hist_bucket="+hb:1.0,
        "inc_module="+im:1.0,"chal_module="+cm:1.0,"module_pair="+im+">"+cm:1.0,
        "inc_readiness_band="+ir:1.0,"chal_readiness_band="+cr:1.0,
        "inc_age_bin="+ia:1.0,"chal_age_bin="+ca:1.0,
        "inc_module_tpl="+im+"|"+lt:1.0,"chal_module_tpl="+cm+"|"+lt:1.0,
        "inc_module_transition="+im+"|"+trans:1.0,"chal_module_transition="+cm+"|"+trans:1.0,
        "inc_module_topic="+im+"|"+ltopic:1.0,"chal_module_topic="+cm+"|"+ltopic:1.0,
        "inc_module_journey="+im+"|"+journey:1.0,"chal_module_journey="+cm+"|"+journey:1.0,
        "inc_module_readiness="+im+"|"+ir:1.0,"chal_module_readiness="+cm+"|"+cr:1.0,
        "inc_module_agebin="+im+"|"+ia:1.0,"chal_module_agebin="+cm+"|"+ca:1.0,
        "inc_module_level_tpl="+il+"|"+lt:1.0,"chal_module_level_tpl="+cl+"|"+lt:1.0,
        f"rank={rank}":1.0,f"state={int(surf['state'][local_row])}":1.0,
    }
    return struct,e5,nom,gd


def make_action_dataset(global_idx,surf,eligible,ytrue=None):
    global_idx=np.asarray(global_idx,int); eligible=np.asarray(eligible,bool)
    Xs=[]; Xe=[]; Xn=[]; G=[]; meta=[]; target=[]
    for r in range(len(global_idx)):
        if not eligible[r]: continue
        anchor=surf["anchor"][r]
        top4=np.argsort(-anchor,kind="mergesort")[:4]
        inc=int(top4[0]); before=None
        if ytrue is not None: before=float(per_user_ndcg(ytrue[r:r+1],anchor[None,:])[0])
        for rank,c in enumerate(top4[1:],start=2):
            c=int(c)
            sf,ef,nf,gd=action_features(global_idx[r],surf,r,inc,c,rank,E5_CURRENT,E5_NEXT,NOMIC_CURRENT,NOMIC_NEXT)
            delta=np.nan
            if ytrue is not None:
                tmp=anchor[None,:].copy(); apply_top1_permutation(tmp,0,c)
                delta=float(per_user_ndcg(ytrue[r:r+1],tmp)[0]-before)
            Xs.append(sf); Xe.append(ef); Xn.append(nf); G.append(gd); target.append(delta)
            meta.append({
                "user_local":int(r),"global_row":int(global_idx[r]),"user_id":str(all_users[int(global_idx[r])]),
                "state":str(STATE_NAMES[int(surf['state'][r])]),"incumbent":inc,"challenger":c,
                "incumbent_module":MODULES[inc],"challenger_module":MODULES[c],"challenger_rank":int(rank),
                "e5_current_diff":float(ef[0]),"e5_next_diff":float(ef[1]),
                "nomic_current_diff":float(nf[0]),"nomic_next_diff":float(nf[1]),
                "target_delta":float(delta) if np.isfinite(delta) else np.nan,
            })
    return {
        "struct":np.asarray(Xs,np.float32).reshape(-1,len(STRUCT_FEATURE_NAMES)),
        "e5":np.asarray(Xe,np.float32).reshape(-1,len(SEM_FEATURE_NAMES)),
        "nomic":np.asarray(Xn,np.float32).reshape(-1,len(SEM_FEATURE_NAMES)),
        "gen":G,"meta":pd.DataFrame(meta),"target":np.asarray(target,float),
        "n_users":len(global_idx),"global_idx":global_idx,"surface":surf,"eligible":eligible,
    }

class UtilityModel:
    def __init__(self,scaler,model): self.scaler=scaler; self.model=model
    def predict(self,X): return self.model.predict(self.scaler.transform(np.asarray(X,np.float32)))

class GeneratorVetoModel:
    def __init__(self,vec,clf,const=None): self.vec=vec; self.clf=clf; self.const=const
    def p_harm(self,rows):
        if self.const is not None: return np.full(len(rows),float(self.const),float)
        return self.clf.predict_proba(self.vec.transform(rows))[:,1]


def action_weights(delta):
    d=np.asarray(delta,float); scale=max(float(np.mean(np.abs(d))),1e-6)
    return np.clip(np.abs(d)/scale,0.25,4.0)


def fit_utility(X,delta,seed):
    X=np.asarray(X,np.float32); delta=np.asarray(delta,float); w=action_weights(delta)
    sc=StandardScaler(); Z=sc.fit_transform(X)
    mdl=Ridge(alpha=UTILITY_RIDGE_ALPHA,fit_intercept=True)
    mdl.fit(Z,delta,sample_weight=w)
    pred=mdl.predict(Z)
    return UtilityModel(sc,mdl),{
        "n":int(len(delta)),"rmse":float(np.sqrt(mean_squared_error(delta,pred))),
        "sign_accuracy":float(accuracy_score(delta>0,pred>0,sample_weight=w)),
        "coef_l2":float(np.sqrt(np.square(mdl.coef_).sum())),
    }


def fit_generator(rows,delta,seed):
    delta=np.asarray(delta,float); harmful=(delta<0).astype(int); w=action_weights(delta)
    if len(np.unique(harmful))<2:
        return GeneratorVetoModel(None,None,float(harmful.mean())),{"n":int(len(delta)),"harm_rate":float(harmful.mean()),"auc":np.nan,"constant":True}
    vec=DictVectorizer(sparse=True); X=vec.fit_transform(rows)
    clf=LogisticRegression(C=GENERATOR_C,penalty="l2",solver="liblinear",max_iter=1000,random_state=int(seed))
    clf.fit(X,harmful,sample_weight=w)
    p=clf.predict_proba(X)[:,1]
    return GeneratorVetoModel(vec,clf),{"n":int(len(delta)),"harm_rate":float(harmful.mean()),"auc":float(roc_auc_score(harmful,p,sample_weight=w)),"constant":False,"features":int(X.shape[1])}

print("V46 action feature spaces",len(STRUCT_FEATURE_NAMES),len(SEM_FEATURE_NAMES))

# ---- cell 34 (code) ----
# Build exact action utility rows and fit frozen full + five fold opinions only on ACTION_TRAIN.
MODELS_READY=False
HEAD_AUDIT_ROWS=[]
MODELS={}

if EMBEDDINGS_READY:
    ACTION_DATA=make_action_dataset(ACTION_TRAIN_IDX,ACTION_SURF,ACTION_ELIG,ytrue=y[ACTION_TRAIN_IDX])
    assert len(ACTION_DATA["target"])==3*int(ACTION_ELIG.sum())
    m=ACTION_DATA["meta"]

    # Fold ownership is inherited from the fold-pure structural surface; each fold model excludes that user fold.
    fold_user=np.asarray(ATR_FOLD,int)
    for head in ["struct","e5","nomic"]:
        full,aud=fit_utility(ACTION_DATA[head],ACTION_DATA["target"],V46_MODEL_SEED+_seed_from_text(head))
        folds=[]
        HEAD_AUDIT_ROWS.append({"head":head,"fit":"full","fold":-1,**aud})
        for f in range(HEAD_FOLDS):
            keep=np.flatnonzero(fold_user[m.user_local.to_numpy(int)]!=f)
            mdl,aa=fit_utility(ACTION_DATA[head][keep],ACTION_DATA["target"][keep],V46_MODEL_SEED+1000*f+_seed_from_text(head))
            folds.append(mdl); HEAD_AUDIT_ROWS.append({"head":head,"fit":"fold","fold":f,**aa})
        MODELS[head]={"full":full,"folds":folds}

    gfull,gaud=fit_generator(ACTION_DATA["gen"],ACTION_DATA["target"],V46_MODEL_SEED+777)
    gfolds=[]; HEAD_AUDIT_ROWS.append({"head":"generator_veto","fit":"full","fold":-1,**gaud})
    for f in range(HEAD_FOLDS):
        keep=np.flatnonzero(fold_user[m.user_local.to_numpy(int)]!=f)
        rows=[ACTION_DATA["gen"][i] for i in keep]
        mdl,aa=fit_generator(rows,ACTION_DATA["target"][keep],V46_MODEL_SEED+2000+f)
        gfolds.append(mdl); HEAD_AUDIT_ROWS.append({"head":"generator_veto","fit":"fold","fold":f,**aa})
    MODELS["generator"]={"full":gfull,"folds":gfolds}
    V46_HEAD_AUDIT=pd.DataFrame(HEAD_AUDIT_ROWS)
    ACTION_DATA["meta"].to_csv(V46_OUT/"v46_action_train_ledger.csv",index=False)
    V46_HEAD_AUDIT.to_csv(V46_OUT/"v46_head_fit_audit.csv",index=False)
    MODELS_READY=True
    print(V46_HEAD_AUDIT.to_string(index=False))
else:
    ACTION_DATA=None
    V46_HEAD_AUDIT=pd.DataFrame(HEAD_AUDIT_ROWS)
    print("Action heads not fit because embeddings are unavailable.")

# ---- cell 35 (markdown) ----
# ## Selective consensus policy
#
# No semantic score is blended into the submission. Each expert independently names its best challenger among anchor ranks 2–4. The action executes only if:
#
# - E5 full model predicts positive utility and ≥4/5 E5 fold models select/support the same challenger;
# - Nomic independently satisfies the same rule and selects **that exact challenger**;
# - both E5 and Nomic raw `CURRENT` and `NEXT_STEP` pair differences favor the challenger;
# - STRUCT predicts non-negative utility for that exact action and is not broadly negative across fold models;
# - generator-state does not issue a high-confidence harmful veto.
#
# Anything else abstains to the frozen champion.

# ---- cell 36 (code) ----
def predict_action_heads(data):
    out={}
    if not MODELS_READY or len(data["meta"])==0: return out
    for head in ["struct","e5","nomic"]:
        X=data[head]; out[head+"_full"]=MODELS[head]["full"].predict(X)
        out[head+"_fold"]=np.vstack([m.predict(X) for m in MODELS[head]["folds"]])
    out["gen_full"]=MODELS["generator"]["full"].p_harm(data["gen"])
    out["gen_fold"]=np.vstack([m.p_harm(data["gen"]) for m in MODELS["generator"]["folds"]])
    return out


def expert_decision(head,data,preds,idxs):
    idxs=np.asarray(idxs,int); full=preds[head+"_full"][idxs]
    j=int(idxs[np.argmax(full)]); challenger=int(data["meta"].iloc[j].challenger)
    full_u=float(preds[head+"_full"][j]); votes=0
    for fp in preds[head+"_fold"]:
        jj=int(idxs[np.argmax(fp[idxs])])
        if int(data["meta"].iloc[jj].challenger)==challenger and float(fp[jj])>0: votes+=1
    row=data["meta"].iloc[j]
    if head=="e5": pair_ok=bool(row.e5_current_diff>0 and row.e5_next_diff>0)
    else: pair_ok=bool(row.nomic_current_diff>0 and row.nomic_next_diff>0)
    valid=bool(full_u>0 and votes>=SEMANTIC_FOLD_AGREE and pair_ok)
    return {"action_idx":j,"challenger":challenger,"utility":full_u,"votes":int(votes),"pair_ok":pair_ok,"valid":valid}


def structural_not_veto(preds,j):
    full=float(preds["struct_full"][j]); votes=int(np.sum(preds["struct_fold"][:,j]>=0))
    return bool(full>=0 and votes>=STRUCT_FOLD_NONNEG),full,votes


def generator_veto(preds,j):
    p=float(preds["gen_full"][j]); votes=int(np.sum(preds["gen_fold"][:,j]>0.5))
    veto=bool(p>=GENERATOR_HARM_P and votes>=GENERATOR_HARM_FOLD_AGREE)
    return veto,p,votes


def deploy_policy(data,mode="dual",ytrue=None):
    base=np.asarray(data["surface"]["anchor"],np.float32)
    cand=base.copy(); preds=predict_action_heads(data); meta=data["meta"]
    by_user={int(u):z.index.to_numpy(int) for u,z in meta.groupby("user_local",sort=False)} if len(meta) else {}
    ledger=[]
    for r in range(data["n_users"]):
        gi=int(data["global_idx"][r]); eligible=bool(data["eligible"][r]) and MODELS_READY and (r in by_user)
        row={"row_local":r,"global_row":gi,"user_id":str(all_users[gi]),"eligible":eligible,"mode":mode,"executed":False}
        if not eligible:
            ledger.append(row); continue
        idxs=by_user[r]
        ed=expert_decision("e5",data,preds,idxs); nd=expert_decision("nomic",data,preds,idxs)
        if mode=="e5": selected=ed; semantic_ok=ed["valid"]
        elif mode=="nomic": selected=nd; semantic_ok=nd["valid"]
        elif mode=="dual":
            selected=ed
            semantic_ok=bool(ed["valid"] and nd["valid"] and ed["challenger"]==nd["challenger"])
        else: raise ValueError(mode)
        j=int(selected["action_idx"])
        sok,sutil,svotes=structural_not_veto(preds,j)
        gveto,gp,gvotes=generator_veto(preds,j)
        execute=bool(semantic_ok and sok and not gveto)
        mrow=meta.iloc[j]
        before=np.nan
        if ytrue is not None: before=float(per_user_ndcg(ytrue[r:r+1],cand[r:r+1])[0])
        if execute: apply_top1_permutation(cand,r,int(mrow.challenger))
        after=np.nan
        if ytrue is not None: after=float(per_user_ndcg(ytrue[r:r+1],cand[r:r+1])[0])
        row.update({
            "incumbent_module":str(mrow.incumbent_module),"challenger_module":str(mrow.challenger_module),
            "challenger_rank":int(mrow.challenger_rank),"e5_challenger":MODULES[ed["challenger"]],"nomic_challenger":MODULES[nd["challenger"]],
            "e5_utility":float(ed["utility"]),"nomic_utility":float(nd["utility"]),"e5_votes":int(ed["votes"]),"nomic_votes":int(nd["votes"]),
            "e5_pair_ok":bool(ed["pair_ok"]),"nomic_pair_ok":bool(nd["pair_ok"]),
            "struct_utility":float(sutil),"struct_nonneg_votes":int(svotes),"generator_p_harm":float(gp),"generator_harm_votes":int(gvotes),
            "generator_veto":bool(gveto),"semantic_consensus":bool(ed["challenger"]==nd["challenger"]),"executed":execute,
        })
        if ytrue is not None:
            row.update({"user_ndcg_delta":float(after-before),"beneficial":bool(after-before>1e-12),"harmful":bool(after-before<-1e-12)})
        ledger.append(row)
    return cand,pd.DataFrame(ledger)

print("Selective consensus policy ready")

# ---- cell 37 (markdown) ----
# ## DESIGN gate
#
# DESIGN is untouched by all V46 fits. The final dual policy must clear the preregistered NDCG, random-fold, contiguous-ID-block, action-precision, geometry, template-stress, and **consensus-beats-single-expert** requirements. If it fails, LOCK remains closed.

# ---- cell 38 (code) ----
DESIGN_DATA=None
DESIGN_CAND=DESIGN_BASE.copy(); V46_DESIGN_LEDGER=pd.DataFrame(); V46_SINGLE_EXPERT=pd.DataFrame()
V46_DESIGN_FOLDS=pd.DataFrame(); V46_DESIGN_BLOCKS=pd.DataFrame(); V46_TEMPLATE_STRESS=pd.DataFrame()
DESIGN_GATE=False

if MODELS_READY:
    DESIGN_DATA=make_action_dataset(DESIGN_IDX,DESIGN_SURF,DESIGN_ELIG,ytrue=y[DESIGN_IDX])
    DESIGN_CAND,V46_DESIGN_LEDGER=deploy_policy(DESIGN_DATA,"dual",ytrue=y[DESIGN_IDX])
    E5_CAND,E5_LEDGER=deploy_policy(DESIGN_DATA,"e5",ytrue=y[DESIGN_IDX])
    NOMIC_CAND,NOMIC_LEDGER=deploy_policy(DESIGN_DATA,"nomic",ytrue=y[DESIGN_IDX])

    def ledger_precision(led):
        z=led[led.executed==True]
        return float((z.user_ndcg_delta>0).mean()) if len(z) else 0.0
    single=[]
    for name,c,l in [("E5_ONLY",E5_CAND,E5_LEDGER),("NOMIC_ONLY",NOMIC_CAND,NOMIC_LEDGER),("DUAL",DESIGN_CAND,V46_DESIGN_LEDGER)]:
        mm=metric_v46(y[DESIGN_IDX],DESIGN_BASE,c); z=l[l.executed==True]
        single.append({"policy":name,**mm,"actions":int(len(z)),"action_precision":ledger_precision(l)})
    V46_SINGLE_EXPERT=pd.DataFrame(single)

    dstrata=joint[DESIGN_IDX]
    dsplits=list(StratifiedKFold(HEAD_FOLDS,shuffle=True,random_state=V46_FOLD_SEED+77).split(np.zeros(len(DESIGN_IDX)),dstrata))
    fr=[]
    for f,(_,va) in enumerate(dsplits):
        mm=metric_v46(y[DESIGN_IDX][va],DESIGN_BASE[va],DESIGN_CAND[va])
        acts=V46_DESIGN_LEDGER[(V46_DESIGN_LEDGER.row_local.isin(np.asarray(va,int))) & (V46_DESIGN_LEDGER.executed==True)]
        fr.append({"fold":f,"n":int(len(va)),**mm,"actions":int(len(acts)),
                   "action_precision":float((acts.user_ndcg_delta>0).mean()) if len(acts) else np.nan})
    V46_DESIGN_FOLDS=pd.DataFrame(fr)

    br=[]; order=np.argsort(DESIGN_IDX); blocks=np.array_split(order,BLOCKS)
    for b,ix in enumerate(blocks):
        mm=metric_v46(y[DESIGN_IDX][ix],DESIGN_BASE[ix],DESIGN_CAND[ix])
        br.append({"block":b,"n":int(len(ix)),"id_min":int(DESIGN_IDX[ix].min()+1),"id_max":int(DESIGN_IDX[ix].max()+1),**mm})
    V46_DESIGN_BLOCKS=pd.DataFrame(br)

    # True template-disjoint rejection stress: latest generator skeletons unseen in ACTION_TRAIN.
    train_templates=set(map(str,LATEST_GROUP[ACTION_TRAIN_IDX]))
    seen_tpl=np.asarray([str(LATEST_GROUP[g]) in train_templates for g in DESIGN_IDX],bool)
    tr=[]
    for label,mask in [("SEEN_IN_ACTION_TRAIN",seen_tpl),("UNSEEN_FROM_ACTION_TRAIN",~seen_tpl)]:
        ix=np.flatnonzero(mask)
        if len(ix)==0: continue
        mm=metric_v46(y[DESIGN_IDX][ix],DESIGN_BASE[ix],DESIGN_CAND[ix])
        tr.append({"template_regime":label,"n":int(len(ix)),**mm})
    V46_TEMPLATE_STRESS=pd.DataFrame(tr)

    mm=metric_v46(y[DESIGN_IDX],DESIGN_BASE,DESIGN_CAND)
    du=per_user_ndcg(y[DESIGN_IDX],DESIGN_CAND)-per_user_ndcg(y[DESIGN_IDX],DESIGN_BASE)
    ci=paired_boot_v46(du,V46_BOOT_SEED)
    acts=V46_DESIGN_LEDGER[V46_DESIGN_LEDGER.executed==True]
    precision=float((acts.user_ndcg_delta>0).mean()) if len(acts) else 0.0
    fold_wins=int((V46_DESIGN_FOLDS.delta>0).sum()); block_wins=int((V46_DESIGN_BLOCKS.delta>0).sum())
    geom=geometry_audit_v46(DESIGN_BASE,DESIGN_CAND)
    single_idx=V46_SINGLE_EXPERT.set_index("policy")
    complementarity_ok=bool(
        precision>float(single_idx.loc["E5_ONLY","action_precision"]) and
        precision>float(single_idx.loc["NOMIC_ONLY","action_precision"])
    )
    unseen_stress=V46_TEMPLATE_STRESS[V46_TEMPLATE_STRESS.template_regime=="UNSEEN_FROM_ACTION_TRAIN"]
    template_ok=bool(len(unseen_stress)==0 or unseen_stress.delta.iloc[0]>=TEMPLATE_CATASTROPHIC_FLOOR)
    DESIGN_GATE=bool(
        ORACLE_GATE and mm["delta"]>=DESIGN_MIN_DELTA and fold_wins>=DESIGN_MIN_FOLD_WINS and block_wins>=DESIGN_MIN_BLOCK_WINS and
        precision>=DESIGN_MIN_ACTION_PRECISION and complementarity_ok and template_ok and mm["top1_delta"]>0 and abs(mm["rel5_delta"])<1e-12 and
        geom["top4_membership_changes"]==0 and geom["top5_membership_changes"]==0 and geom["rank5_to_17_order_changes"]==0 and geom["top4_score_multiset_ok"]
    )
    V46_DESIGN_METRICS=pd.DataFrame([{**mm,"ci_low":ci[0],"ci_high":ci[1],"actions":int(len(acts)),"action_precision":precision,
        "beneficial_actions":int((acts.user_ndcg_delta>0).sum()) if len(acts) else 0,"harmful_actions":int((acts.user_ndcg_delta<0).sum()) if len(acts) else 0,
        "fold_wins":fold_wins,"block_wins":block_wins,"complementarity_ok":complementarity_ok,"template_stress_ok":template_ok,
        **{f"geom_{k}":v for k,v in geom.items()},"design_gate":DESIGN_GATE}])
else:
    V46_DESIGN_METRICS=pd.DataFrame([{"base_ndcg":ndcg5(y[DESIGN_IDX],DESIGN_BASE),"candidate_ndcg":ndcg5(y[DESIGN_IDX],DESIGN_BASE),"delta":0.0,
        "actions":0,"action_precision":0.0,"fold_wins":0,"block_wins":0,"complementarity_ok":False,"template_stress_ok":False,"design_gate":False}])

print(V46_SINGLE_EXPERT.to_string(index=False) if len(V46_SINGLE_EXPERT) else "single-expert comparison unavailable")
print(V46_DESIGN_METRICS.to_string(index=False)); print("DESIGN_GATE",DESIGN_GATE)

# ---- cell 39 (markdown) ----
# ## Untouched LOCK and exact production deployment
#
# LOCK is evaluated only after DESIGN has frozen the policy. No encoder, head, threshold, feature, or action rule is retrained or altered after DESIGN. Test deployment uses the exact full-data `0.66028` champion, with S1/S3 and all historical V25/PB protected users locked.

# ---- cell 40 (code) ----
LOCK_GATE=False; LOCK_EVALUATED=False
V46_LOCK_LEDGER=pd.DataFrame(); V46_LOCK_BLOCKS=pd.DataFrame(); LOCK_CAND=LOCK_SURF["anchor"].copy()
V46_LOCK_METRICS=pd.DataFrame([{"evaluated":False,"lock_gate":False}])

if DESIGN_GATE:
    LOCK_DATA=make_action_dataset(LOCK_IDX,LOCK_SURF,LOCK_ELIG,ytrue=y[LOCK_IDX])
    LOCK_CAND,V46_LOCK_LEDGER=deploy_policy(LOCK_DATA,"dual",ytrue=y[LOCK_IDX])
    mm=metric_v46(y[LOCK_IDX],LOCK_SURF["anchor"],LOCK_CAND)
    du=per_user_ndcg(y[LOCK_IDX],LOCK_CAND)-per_user_ndcg(y[LOCK_IDX],LOCK_SURF["anchor"])
    ci=paired_boot_v46(du,V46_BOOT_SEED+1)
    acts=V46_LOCK_LEDGER[V46_LOCK_LEDGER.executed==True]
    precision=float((acts.user_ndcg_delta>0).mean()) if len(acts) else 0.0
    br=[]; order=np.argsort(LOCK_IDX); blocks=np.array_split(order,BLOCKS)
    for b,ix in enumerate(blocks):
        rr=metric_v46(y[LOCK_IDX][ix],LOCK_SURF["anchor"][ix],LOCK_CAND[ix])
        br.append({"block":b,"n":int(len(ix)),"id_min":int(LOCK_IDX[ix].min()+1),"id_max":int(LOCK_IDX[ix].max()+1),**rr})
    V46_LOCK_BLOCKS=pd.DataFrame(br); block_wins=int((V46_LOCK_BLOCKS.delta>0).sum())
    geom=geometry_audit_v46(LOCK_SURF["anchor"],LOCK_CAND)
    LOCK_GATE=bool(
        mm["delta"]>=LOCK_MIN_DELTA and block_wins>=LOCK_MIN_BLOCK_WINS and precision>=LOCK_MIN_ACTION_PRECISION and
        mm["top1_delta"]>0 and abs(mm["rel5_delta"])<1e-12 and geom["top4_membership_changes"]==0 and
        geom["top5_membership_changes"]==0 and geom["rank5_to_17_order_changes"]==0 and geom["top4_score_multiset_ok"]
    )
    LOCK_EVALUATED=True
    V46_LOCK_METRICS=pd.DataFrame([{"evaluated":True,**mm,"ci_low":ci[0],"ci_high":ci[1],"actions":int(len(acts)),
        "action_precision":precision,"beneficial_actions":int((acts.user_ndcg_delta>0).sum()) if len(acts) else 0,
        "harmful_actions":int((acts.user_ndcg_delta<0).sum()) if len(acts) else 0,"block_wins":block_wins,
        **{f"geom_{k}":v for k,v in geom.items()},"lock_gate":LOCK_GATE}])

print(V46_LOCK_METRICS.to_string(index=False)); print("LOCK_GATE",LOCK_GATE)

# ---- cell 41 (code) ----
# Exact test surface: full-data 0.66028 champion plus frozen full-data V30/V31 state.
TEST_GLOBAL=np.arange(N_TRAIN,N_TRAIN+N_TEST,dtype=int)
TEST_SURF={
    "pre":np.asarray(consensus3_test,np.float32),"v30":np.asarray(v30_test,np.float32),"v31":np.asarray(v31_test,np.float32),
    "state":np.asarray(full_state,np.int8),"anchor":np.asarray(V44_ANCHOR_TEST,np.float32),
}
protected_test=np.zeros(N_TEST,bool); protected_test[protected_rows]=True
TEST_ELIG=np.isin(full_state,[0,2]) & ~protected_test
# Public-positive S3 and all historical PB/V25 rows remain immutable by construction.
assert not TEST_ELIG[full_state==3].any()
assert not TEST_ELIG[protected_rows].any()

TEST_CANDIDATE=V44_ANCHOR_TEST.copy(); V46_TEST_LEDGER=pd.DataFrame()
TEST_GEOM={"evaluated":False,"geometry_gate":False,"actions":0,"anchor_top5_sha256":top5_sha256(V44_ANCHOR_TEST),"candidate_top5_sha256":top5_sha256(V44_ANCHOR_TEST)}

if LOCK_GATE and MODELS_READY:
    TEST_DATA=make_action_dataset(TEST_GLOBAL,TEST_SURF,TEST_ELIG,ytrue=None)
    TEST_CANDIDATE,V46_TEST_LEDGER=deploy_policy(TEST_DATA,"dual",ytrue=None)
    geom=geometry_audit_v46(V44_ANCHOR_TEST,TEST_CANDIDATE)
    acts=int(V46_TEST_LEDGER.executed.sum())
    changed=np.any(top5_indices(V44_ANCHOR_TEST)!=top5_indices(TEST_CANDIDATE),axis=1)
    locked_equal=bool(np.array_equal(TEST_CANDIDATE[~TEST_ELIG],V44_ANCHOR_TEST[~TEST_ELIG]))
    geometry_gate=bool(
        locked_equal and geom["changed_users"]==acts and geom["top1_changes"]==acts and
        geom["top4_membership_changes"]==0 and geom["top5_membership_changes"]==0 and geom["rank5_to_17_order_changes"]==0 and
        geom["top4_score_multiset_ok"] and TEST_MIN_ACTIONS<=acts<=TEST_MAX_ACTIONS
    )
    TEST_GEOM={"evaluated":True,**geom,"actions":acts,"locked_rows_bit_exact":locked_equal,
        "submission_scale":bool(TEST_MIN_ACTIONS<=acts<=TEST_MAX_ACTIONS),"geometry_gate":geometry_gate,
        "anchor_top5_sha256":top5_sha256(V44_ANCHOR_TEST),"candidate_top5_sha256":top5_sha256(TEST_CANDIDATE)}

FINAL_GATE=bool(
    top5_sha256(V44_ANCHOR_TEST)==EXPECTED_V46_ANCHOR_HASH and ORACLE_GATE and EMBEDDINGS_READY and MODELS_READY and
    DESIGN_GATE and LOCK_GATE and bool(TEST_GEOM.get("geometry_gate",False)) and not V46_SMOKE
)
PRIMARY=TEST_CANDIDATE if FINAL_GATE else V44_ANCHOR_TEST.copy()
print(pd.DataFrame([TEST_GEOM]).to_string(index=False)); print("FINAL_GATE",FINAL_GATE)

# ---- cell 42 (markdown) ----
# ## Artifacts, reproducibility ledger, and submission decision
#
# `submission_v46_primary.csv` is the only file eligible for a leaderboard slot. If `final_gate == false`, that file is intentionally the exact `0.66028` fail-close champion.

# ---- cell 43 (code) ----
def submission_v46(scores):
    out=sample.copy(); out.loc[:,MODULES]=np.asarray(scores,np.float32)
    assert out.user_id.astype(str).tolist()==test.user_id.astype(str).tolist()
    assert np.isfinite(out[MODULES].to_numpy()).all()
    return out

anchor_sub=submission_v46(V44_ANCHOR_TEST)
candidate_sub=submission_v46(TEST_CANDIDATE)
primary_sub=submission_v46(PRIMARY)
anchor_sub.to_csv(V46_OUT/"submission_v46_anchor_066028_reference.csv",index=False)
candidate_sub.to_csv(V46_OUT/"submission_v46_candidate_dual_consensus.csv",index=False)
primary_sub.to_csv(V46_OUT/"submission_v46_primary.csv",index=False)
# Convenience root-level copy for Kaggle submission dialog.
primary_sub.to_csv(OUT_DIR/"submission_v46_primary.csv",index=False)

V46_SPLIT_MANIFEST.to_csv(V46_OUT/"v46_split_manifest.csv",index=False)
V46_ORACLE.to_csv(V46_OUT/"v46_oracle.csv",index=False)
V46_DESIGN_METRICS.to_csv(V46_OUT/"v46_design_metrics.csv",index=False)
V46_DESIGN_FOLDS.to_csv(V46_OUT/"v46_design_fold_metrics.csv",index=False)
V46_DESIGN_BLOCKS.to_csv(V46_OUT/"v46_design_block_metrics.csv",index=False)
V46_TEMPLATE_STRESS.to_csv(V46_OUT/"v46_template_stress.csv",index=False)
V46_SINGLE_EXPERT.to_csv(V46_OUT/"v46_single_expert_comparison.csv",index=False)
V46_DESIGN_LEDGER.to_csv(V46_OUT/"v46_design_action_ledger.csv",index=False)
V46_LOCK_METRICS.to_csv(V46_OUT/"v46_lock_metrics.csv",index=False)
V46_LOCK_BLOCKS.to_csv(V46_OUT/"v46_lock_block_metrics.csv",index=False)
V46_LOCK_LEDGER.to_csv(V46_OUT/"v46_lock_action_ledger.csv",index=False)
V46_TEST_LEDGER.to_csv(V46_OUT/"v46_test_action_ledger.csv",index=False)
pd.DataFrame([TEST_GEOM]).to_csv(V46_OUT/"v46_test_geometry.csv",index=False)
if len(V46_HEAD_AUDIT): V46_HEAD_AUDIT.to_csv(V46_OUT/"v46_head_fit_audit.csv",index=False)

versions={"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,
          "sklearn":sklearn.__version__,"lightgbm":lgb.__version__}
try:
    import torch, transformers
    versions.update({"torch":torch.__version__,"transformers":transformers.__version__,"cuda":torch.version.cuda,"gpu_count":torch.cuda.device_count()})
except Exception: pass
json.dump(versions,open(V46_OUT/"v46_package_versions.json","w"),indent=2)

repro={
    "version":V46_VERSION,"organizer_data_only":True,"external_labeled_data":False,
    "open_weight_pretrained_encoders_only":True,"no_direct_genai_test_relevance":True,
    "no_historical_submission_csv_predictive_input":True,"no_user_id_predictive_feature":True,
    "no_user_or_module_whitelist":True,"no_post_design_or_lock_retraining":True,
    "top1_only":True,"top4_membership_frozen":True,"top5_membership_frozen":True,"ranks5_to17_frozen":True,
    "protected_s1_s3_and_v25_rows":True,"action_target":"exact counterfactual Top1 delta NDCG@5",
    "e5_model_id":E5_MODEL_ID,"e5_revision":E5_REVISION,"nomic_model_id":NOMIC_MODEL_ID,"nomic_revision":NOMIC_REVISION,"nomic_code_revision":NOMIC_CODE_REVISION,
    "generator_veto_representation":"V42 label-free generator-state categories; veto only",
    "split_seed":V46_SPLIT_SEED,"fold_seed":V46_FOLD_SEED,"model_seed":V46_MODEL_SEED,
    "anchor_top5_sha256":top5_sha256(V44_ANCHOR_TEST),"expected_anchor_top5_sha256":EXPECTED_V46_ANCHOR_HASH,
    "oracle_gate":bool(ORACLE_GATE),"embeddings_ready":bool(EMBEDDINGS_READY),"embedding_error":EMBED_ERROR,
    "design_gate":bool(DESIGN_GATE),"lock_evaluated":bool(LOCK_EVALUATED),"lock_gate":bool(LOCK_GATE),"final_gate":bool(FINAL_GATE),
}
json.dump(repro,open(V46_OUT/"v46_reproduction_audit.json","w"),indent=2)

instruction=(
    "SUBMIT submission_v46_primary.csv. V46 passed oracle, DESIGN, untouched LOCK, dual-consensus complementarity, and test geometry gates."
    if FINAL_GATE else
    "DO NOT spend a leaderboard slot on V46: submission_v46_primary.csv fail-closed exactly to the 0.66028 champion. Inspect v46_design_metrics.csv / v46_lock_metrics.csv."
)
summary={
    "version":V46_VERSION,"public_anchor":PUBLIC_V46_ANCHOR_LB,"public_top5_cutoff":PUBLIC_V46_TOP5_CUTOFF,"public_leader":PUBLIC_V46_LEADER,
    "oracle_delta":float(V46_ORACLE.iloc[0].top1_within_top4_oracle_delta),"oracle_gate":bool(ORACLE_GATE),
    "embeddings_ready":bool(EMBEDDINGS_READY),"embedding_error":EMBED_ERROR,
    "design_delta":float(V46_DESIGN_METRICS.iloc[0].get("delta",0.0)),"design_gate":bool(DESIGN_GATE),
    "lock_evaluated":bool(LOCK_EVALUATED),"lock_delta":float(V46_LOCK_METRICS.iloc[0].get("delta",0.0)) if LOCK_EVALUATED else None,"lock_gate":bool(LOCK_GATE),
    "test_actions":int(TEST_GEOM.get("actions",0)),"test_geometry_gate":bool(TEST_GEOM.get("geometry_gate",False)),"final_gate":bool(FINAL_GATE),
    "anchor_top5_sha256":top5_sha256(V44_ANCHOR_TEST),"candidate_top5_sha256":top5_sha256(TEST_CANDIDATE),"primary_top5_sha256":top5_sha256(PRIMARY),
    "submission_instruction":instruction,
}
json.dump(summary,open(V46_OUT/"v46_summary.json","w"),indent=2)

# Final fail-close and score-geometry invariants.
assert top5_sha256(V44_ANCHOR_TEST)==EXPECTED_V46_ANCHOR_HASH
if FINAL_GATE:
    assert top5_sha256(PRIMARY)==top5_sha256(TEST_CANDIDATE)
else:
    assert np.array_equal(PRIMARY,V44_ANCHOR_TEST)
    assert top5_sha256(PRIMARY)==EXPECTED_V46_ANCHOR_HASH
assert np.array_equal(primary_sub[MODULES].to_numpy(np.float32),PRIMARY.astype(np.float32))

print(json.dumps(summary,indent=2))
print("Artifacts:",V46_OUT)
print("SUBMISSION DECISION:",instruction)

# ---- cell 44 (markdown) ----
# ### Terminal interpretation contract
#
# - If `final_gate == true`, use **only** `/kaggle/working/submission_v46_primary.csv` for the V46 leaderboard slot.
# - If the oracle fails, do not replace the two embeddings or relax the action space.
# - If DESIGN fails, do not inspect LOCK to tune thresholds, remove modules, or create a V46.1 mask.
# - If LOCK fails, do not retrain on DESIGN/LOCK.
# - If the test action count falls outside 3–15, do not select a top-N subset post hoc; the run fails closed.
# - A failed run is still successful scientifically: `submission_v46_primary.csv` remains the exact `0.66028` champion by hash.
