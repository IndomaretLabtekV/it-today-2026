# Converted from notebooks/v13_1.ipynb: markdown cells become comments, %/! magics neutralized.

# ---- cell 0 (markdown) ----
# # MineToday V13.1 — Dual-Expert Utility Router
#
# **Immutable public champion:** V12.2 `Consensus3`, public NDCG@5 **0.65856**.
#
# ## Diagnosis carried forward
#
# V13 established that selective expert routing is directionally valid, but routing only between Consensus3 and V9.1 leaves most of the existing expert diversity unused. The strict post-V13 diagnosis showed that the strongest next step is **not** a free multi-model selector and **not** more module-weight hill climbing. It is a tightly regularized, cross-fitted **dual-expert utility router**:
#
# - Consensus3 remains the exact default for every user.
# - V9.1 remains the first complementary expert, with a frozen **12.5% ML nomination budget**.
# - V10 `C_boundary_utility_v91` becomes the second complementary expert, with a frozen **5% ML nomination budget**.
# - Each expert has its own low-capacity score-geometry gate and train-only Top-5 transition-evidence table.
# - A user can move away from Consensus3 only when both the ML nomination and transition-evidence gate agree.
# - If both experts are eligible, the router uses **predicted utility × historical transition evidence** to choose one action.
# - Every outer validation user is untouched by sparse fitting, relevance calibration, V9.1 construction, V10-C construction, router fitting, and transition statistics.
#
# The route budgets are **frozen before this notebook run**. A second-order diagnostic evaluates a tiny pre-registered budget set by choosing on four outer folds and measuring once on the fifth, but it never changes the deployed V13.1 policy.
#
# ## Exact expert family
#
# 1. **Consensus3 champion**: exact V3.1 word+char TF-IDF + Ridge, seed-42 LightGBM relevance calibration, and exact V12.2 module-aware calibration weights.
# 2. **V9.1 expert**: exact distribution-preserving multi-seed relevance family + cumulative ordinal expected-relevance + fold-safe empirical-Bayes residual control.
# 3. **V10-C expert**: exact cal045 anchor + `P(rank<=5)` head + NDCG-normalized utility head + rank-local V9.1 correction.
# 4. **Dual router**: two tiny LightGBM gates using only score/rank disagreement geometry. No raw `user_id`, no external data, no LLM/GenAI test prediction.
#
# ## Frozen deployed routing contract
#
# - `V91_ROUTE_FRACTION=0.125`
# - `V10C_ROUTE_FRACTION=0.05`
# - transition support >= 8 train examples
# - positive transition mean in >= 2 independent folds
# - positive train transition mean
# - low-capacity gate: 7 leaves, strong L1/L2, `min_child_samples=150`
# - Consensus3 wins every unresolved case
#
# ## Kaggle
#
# Ready for **Kaggle T4 ×2**. The pipeline is CPU-oriented TF-IDF/Ridge/LightGBM, so low GPU utilization is expected and does not indicate a problem.
#
# ## Outputs
#
# - `submission_v13_1_dual_router.csv`
# - `submission_v13_1_primary.csv` — dual router only if all frozen promotion gates pass, otherwise exact Consensus3
# - `submission_v13_1_consensus3_reference.csv`
# - `submission_v13_1_v91_expert.csv`
# - `submission_v13_1_v10c_expert.csv`
# - `v13_1_summary.json`
# - `v13_1_router_diagnostics.csv`
# - `v13_1_second_order.csv`
# - `v13_1_routed_test_users.csv`
# - `v13_1_oof_artifacts.npz`
#
# **Run all cells. Do not submit anything before reading `V13.1 DECISION`.**

# ---- cell 1 (code) ----
import os,re,json,ast,html,hashlib,random,warnings,gc,time,sys,subprocess
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS","4")
os.environ.setdefault("OPENBLAS_NUM_THREADS","4")
os.environ.setdefault("MKL_NUM_THREADS","4")
os.environ["PYTHONHASHSEED"]="42"
os.environ["TOKENIZERS_PARALLELISM"]="false"
os.environ.setdefault("PYDEVD_DISABLE_FILE_VALIDATION","1")

import numpy as np, pandas as pd
from scipy.sparse import hstack
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
warnings.filterwarnings("ignore")
try:
    import lightgbm as lgb
except Exception:
    subprocess.check_call([sys.executable,"-m","pip","install","-q","lightgbm>=4.5,<5"])
    import lightgbm as lgb

SEED=42
V131_SMOKE=bool(int(os.getenv("V131_SMOKE","0")))
V13_SMOKE=V131_SMOKE
V91_SMOKE=V131_SMOKE
N_FOLDS=2 if V131_SMOKE else 5
N_INNER=2 if V131_SMOKE else 3
RIDGE_ALPHA=3.0
REFERENCE_W=0.45
LGB_ESTIMATORS=40 if V131_SMOKE else 1200
LGB_EARLY_STOP=5 if V131_SMOKE else 80
ORD_ESTIMATORS=60 if V131_SMOKE else 300
ORD_EARLY_STOP=8 if V131_SMOKE else 40
DIVERSITY_SEEDS=[137] if V131_SMOKE else [137,2026,7,31415]
ORDINAL_WEIGHT=0.10
META_SCALE_GRID=np.asarray([1.0,1.45] if V131_SMOKE else [1.35,1.45,1.55],np.float32)
BOOTSTRAPS=300 if V131_SMOKE else 5000

PUBLIC_CAL045_LB=0.65639
PUBLIC_V91_LB=0.65432
PUBLIC_CHAMPION_LB=0.65856

# Exact V10-C configuration.
V10C_P5_WEIGHT=0.08
V10C_UTILITY_WEIGHT=0.15
V91_RANK_WEIGHTS=np.asarray([1.00,0.00,0.25,0.75,1.00,1.00,0.50,0.25]+[0.00]*9,np.float32)
UTILITY_ESTIMATORS=40 if V131_SMOKE else 1000
UTILITY_EARLY_STOP=5 if V131_SMOKE else 70

# Frozen dual-router policy from the post-V13 diagnosis.
V91_ROUTE_FRACTION=0.125
V10C_ROUTE_FRACTION=0.05
TRANSITION_MIN_SUPPORT=8
TRANSITION_MIN_POS_FOLDS=2
GATE_ESTIMATORS=50 if V131_SMOKE else 180
GATE_LEAVES=7
GATE_MIN_CHILD=20 if V131_SMOKE else 150
GATE_REG_L2=12.0
GATE_REG_L1=2.0

# Tiny pre-registered budget family for second-order diagnostics only.
SECOND_ORDER_BUDGETS=[(0.10,0.03),(0.125,0.05),(0.15,0.05)]

# Frozen promotion requirements versus exact Consensus3.
MIN_ROUTER_INCREMENT=-0.01 if V131_SMOKE else 0.00035
MIN_OUTER_WINS=1 if V131_SMOKE else 5
MAX_WORST_FOLD=-0.01 if V131_SMOKE else 0.0
MIN_QUARTER_WINS=1 if V131_SMOKE else 4
MIN_TAIL_INCREMENT=-0.01 if V131_SMOKE else 0.0
MIN_BOOT_CI_LOW=-1.0 if V131_SMOKE else 0.0
MAX_TEST_ROUTE_RATE=0.12 if V131_SMOKE else 0.06
MAX_TEST_TOP5_CHANGE=0.15 if V131_SMOKE else 0.05
MAX_TEST_TOP1_CHANGE=0.08 if V131_SMOKE else 0.01

random.seed(SEED); np.random.seed(SEED)
print("versions",np.__version__,pd.__version__,lgb.__version__,"V131_SMOKE",V131_SMOKE)
print("public champion",PUBLIC_CHAMPION_LB,
      "budgets",{"v91":V91_ROUTE_FRACTION,"v10c":V10C_ROUTE_FRACTION},
      "transition",TRANSITION_MIN_SUPPORT,TRANSITION_MIN_POS_FOLDS,
      "gate min child",GATE_MIN_CHILD)

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

# Compatibility alias used by the V13 router diagnostics.
def paired_boot(delta):
    return boot_ci(delta)

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

# ---- cell 7 (code) ----
# Exact public-validated V3.1 sparse backbone.
t0=time.time()
word_vec=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_df=.995,sublinear_tf=True,strip_accents="unicode",max_features=(2000 if V91_SMOKE else 50000),dtype=np.float32)
char_vec=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=2,sublinear_tf=True,max_features=(3000 if V91_SMOKE else 60000),dtype=np.float32)
X=hstack([word_vec.fit_transform(texts),char_vec.fit_transform(texts)],format="csr")
def sparse_ref(target):
    oo=np.zeros_like(y,np.float32); te=np.zeros((N_TEST,N_MODULES),np.float32)
    for tr,va in ref_folds:
        m=Ridge(alpha=RIDGE_ALPHA,solver="lsqr",tol=2e-3,max_iter=1000); m.fit(X[tr],target[tr]); oo[va]=m.predict(X[va]).astype(np.float32); te+=m.predict(X[N_TRAIN:]).astype(np.float32)/N_FOLDS
    return oo,te
raw_oof,raw_test=sparse_ref(y); top_oof,top_test=sparse_ref(Y_TOP)
base_oof=(.9*raw_oof+.1*top_oof).astype(np.float32); base_test=(.9*raw_test+.1*top_test).astype(np.float32)
print("X",X.shape,"base OOF",round(ndcg5(y,base_oof),6),"sec",round(time.time()-t0,1))

# ---- cell 8 (code) ----
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

# ---- cell 9 (markdown) ----
# ## Immutable Consensus3 champion
#
# This is the exact V12.2 public champion. V13.1 never re-optimizes its module weights.

# ---- cell 10 (code) ----
PROFILE_OVERRIDES={
    "consensus3": {
        "M_002":0.44,"M_006":0.55,"M_007":0.60,"M_010":0.40,"M_011":0.55,
        "M_012":0.55,"M_015":0.44,"M_017":0.53,
    },
}

def profile_vector(name="consensus3"):
    w=np.full(N_MODULES,REFERENCE_W,np.float32)
    for mid,val in PROFILE_OVERRIDES[name].items():
        w[MODULES.index(mid)]=float(val)
    return w

def apply_profile(base,meta,name="consensus3"):
    w=profile_vector(name)
    return ((1.0-w)*base+w*meta).astype(np.float32)

consensus3_oof=apply_profile(base_oof,meta_oof)
consensus3_test=apply_profile(base_test,meta_test)
print("Consensus3 ordinary OOF",round(ndcg5(y,consensus3_oof),6),"public",PUBLIC_CHAMPION_LB)

# ---- cell 11 (markdown) ----
# ## Exact V9.1 complementary expert
#
# The expert is reproduced from organizer data. It is never globally blended into the champion.

# ---- cell 12 (code) ----
def qalign(src_train,tgt_train,vals):
    s=np.sort(np.asarray(src_train,float)); t=np.sort(np.asarray(tgt_train,float)); v=np.asarray(vals,float)
    n=len(s); q=(np.arange(n)+.5)/n
    u,first,cnt=np.unique(s,return_index=True,return_counts=True)
    qu=(first+(cnt-1)/2+.5)/n
    return np.interp(np.interp(v,u,qu,left=qu[0],right=qu[-1]),q,t,left=t[0],right=t[-1]).astype(np.float32)

def align_mat(src_oof,tgt_oof,out):
    z=np.zeros_like(out,np.float32)
    for m in range(N_MODULES): z[:,m]=qalign(src_oof[:,m],tgt_oof[:,m],out[:,m])
    return z

def make_ord_cal(seed,n_estimators=None):
    return lgb.LGBMClassifier(
        n_estimators=int(ORD_ESTIMATORS if n_estimators is None else n_estimators),
        learning_rate=.025,num_leaves=11,min_child_samples=60,
        reg_alpha=1.,reg_lambda=8.,colsample_bytree=.8,
        random_state=int(seed),n_jobs=4,verbosity=-1
    )

def fit_seed_fixed(seed):
    oo=np.zeros_like(meta_oof); te=np.zeros_like(meta_test)
    for m,mid in enumerate(MODULES):
        target=Y_BIN[:,m].astype(np.int32)
        for f,(tr,va) in enumerate(ref_folds):
            it=int(best_iters[mid][f])
            clf=make_cal(seed+f+31*m,it); clf.fit(META_TR[m][tr],target[tr])
            oo[va,m]=clf.predict_proba(META_TR[m][va])[:,1].astype(np.float32)
        fi=max(20,int(np.median(best_iters[mid])))
        full=make_cal(seed+997+31*m,fi); full.fit(META_TR[m],target)
        te[:,m]=full.predict_proba(META_TE[m])[:,1].astype(np.float32)
    return oo,te

aligned_oof=[]; aligned_test=[]
for s in DIVERSITY_SEEDS:
    print("V9.1 diversity seed",s)
    oo,te=fit_seed_fixed(s)
    aligned_oof.append(align_mat(oo,meta_oof,oo))
    aligned_test.append(align_mat(oo,meta_oof,te))
dist_meta_oof=np.mean(np.stack(aligned_oof),0).astype(np.float32)
dist_meta_test=np.mean(np.stack(aligned_test),0).astype(np.float32)
dist_oof=(.55*base_oof+.45*dist_meta_oof).astype(np.float32)
dist_test=(.55*base_test+.45*dist_meta_test).astype(np.float32)

def fit_ord_full(targets,tag):
    oo=np.zeros_like(y,np.float32); te=np.zeros((N_TEST,N_MODULES),np.float32)
    for m in range(N_MODULES):
        target=targets[:,m].astype(np.int32); its=[]
        for f,(tr,va) in enumerate(ref_folds):
            clf=make_ord_cal(SEED+5000+101*tag+f+31*m)
            clf.fit(META_TR[m][tr],target[tr],eval_set=[(META_TR[m][va],target[va])],
                    callbacks=[lgb.early_stopping(ORD_EARLY_STOP,verbose=False)])
            it=int(clf.best_iteration_ or ORD_ESTIMATORS); its.append(it)
            oo[va,m]=clf.predict_proba(META_TR[m][va],num_iteration=it)[:,1].astype(np.float32)
        fi=max(20,int(np.median(its)))
        full=make_ord_cal(SEED+8000+101*tag+31*m,fi); full.fit(META_TR[m],target)
        te[:,m]=full.predict_proba(META_TE[m])[:,1].astype(np.float32)
    return oo,te

ord_oof={}; ord_test={}
for t in range(1,6):
    print("ordinal threshold",t)
    ord_oof[t],ord_test[t]=fit_ord_full((RANK_POS<=t).astype(np.int32),t)

def exp_rel(prel,P):
    p5=np.minimum(P[5],prel); p4=np.minimum(P[4],p5); p3=np.minimum(P[3],p4)
    p2=np.minimum(P[2],p3); p1=np.minimum(P[1],p2)
    return (.25*prel+.15*(p5+p4+p3+p2+p1)).astype(np.float32)

ordexp_oof=exp_rel(meta_oof,ord_oof); ordexp_test=exp_rel(meta_test,ord_test)
v9_oof=((1-ORDINAL_WEIGHT)*dist_oof+ORDINAL_WEIGHT*ordexp_oof).astype(np.float32)
v9_test=((1-ORDINAL_WEIGHT)*dist_test+ORDINAL_WEIGHT*ordexp_test).astype(np.float32)
print("V9 ordinary diagnostic",round(ndcg5(y,v9_oof),6))

# ---- cell 13 (code) ----
def fold_ids_from_splits(n,splits):
    z=np.full(n,-1,np.int16)
    for f,(_,va) in enumerate(splits): z[va]=f
    assert np.all(z>=0)
    return z

def module_ablation_fold_deltas(ytrue,anchor,candidate,fold_id,n_folds):
    residual=(candidate-anchor).astype(np.float32)
    out=np.zeros((n_folds,N_MODULES),np.float64)
    for f in range(n_folds):
        mask=(fold_id==f); basef=ndcg5(ytrue[mask],anchor[mask])
        for m in range(N_MODULES):
            s=anchor[mask].copy(); s[:,m]+=residual[mask,m]
            out[f,m]=ndcg5(ytrue[mask],s)-basef
    return out

def module_support(fold_deltas):
    pos=(np.asarray(fold_deltas)>0).sum(axis=0)
    return (pos>=1).astype(np.float32),pos.astype(np.int16)

def select_shrinkage(ytrue,anchor,candidate,support):
    residual=(candidate-anchor).astype(np.float32); vals=[]
    for scale in META_SCALE_GRID:
        vals.append(ndcg5(ytrue,anchor+float(scale)*residual*support.reshape(1,-1)))
    j=int(np.argmax(vals))
    return float(META_SCALE_GRID[j]),np.asarray(vals,np.float64)

def apply_shrinkage(anchor,candidate,support,scale):
    return (anchor+float(scale)*(candidate-anchor)*support.reshape(1,-1)).astype(np.float32)

# ---- cell 14 (markdown) ----
# ## Exact V10-C Top-5 / utility expert
#
# V10-C is deliberately retained as a complementary expert even though it was not a public champion. Its value here is **conditional diversity**: it can repair some users where Consensus3 is wrong.

# ---- cell 15 (code) ----
# Metric-aligned V10 targets.
IDEAL_TOP5=np.sort(y,axis=1)[:,-5:][:,::-1]
IDCG5=(IDEAL_TOP5*DISCOUNT).sum(1).astype(np.float32)
Y_UTILITY=(y/np.maximum(IDCG5[:,None],1e-12)).astype(np.float32)

def make_utility_reg(seed,n_estimators=None):
    return lgb.LGBMRegressor(
        objective="regression_l2",
        n_estimators=int(UTILITY_ESTIMATORS if n_estimators is None else n_estimators),
        learning_rate=.025,num_leaves=11,min_child_samples=60,
        reg_alpha=1.0,reg_lambda=8.0,colsample_bytree=.8,
        random_state=int(seed),n_jobs=4,verbosity=-1,
    )

def fit_v10_heads_global():
    p5_oof=np.zeros_like(y,np.float32); p5_test=np.zeros((N_TEST,N_MODULES),np.float32)
    util_oof=np.zeros_like(y,np.float32); util_test=np.zeros((N_TEST,N_MODULES),np.float32)
    for m,mid in enumerate(MODULES):
        yp=Y_TOP5[:,m].astype(np.int32); yu=Y_UTILITY[:,m].astype(np.float32)
        p_iters=[]; u_iters=[]
        for f,(tr,va) in enumerate(ref_folds):
            clf=make_cal(SEED+15000+f+31*m)
            clf.fit(META_TR[m][tr],yp[tr],eval_set=[(META_TR[m][va],yp[va])],
                    callbacks=[lgb.early_stopping(LGB_EARLY_STOP,verbose=False)])
            pit=int(clf.best_iteration_ or LGB_ESTIMATORS); p_iters.append(pit)
            p5_oof[va,m]=clf.predict_proba(META_TR[m][va],num_iteration=pit)[:,1].astype(np.float32)

            reg=make_utility_reg(SEED+17000+f+31*m)
            reg.fit(META_TR[m][tr],yu[tr],eval_set=[(META_TR[m][va],yu[va])],
                    callbacks=[lgb.early_stopping(UTILITY_EARLY_STOP,verbose=False)])
            uit=int(reg.best_iteration_ or UTILITY_ESTIMATORS); u_iters.append(uit)
            util_oof[va,m]=reg.predict(META_TR[m][va],num_iteration=uit).astype(np.float32)

        pfull=make_cal(SEED+16000+31*m,max(20,int(np.median(p_iters)))); pfull.fit(META_TR[m],yp)
        p5_test[:,m]=pfull.predict_proba(META_TE[m])[:,1].astype(np.float32)
        ufull=make_utility_reg(SEED+18000+31*m,max(20,int(np.median(u_iters)))); ufull.fit(META_TR[m],yu)
        util_test[:,m]=ufull.predict(META_TE[m]).astype(np.float32)
    return p5_oof,p5_test,util_oof,util_test

def rank_locality_mask(anchor):
    order=np.argsort(-anchor,axis=1); ranks=np.empty_like(order)
    ranks[np.arange(len(anchor))[:,None],order]=np.arange(N_MODULES)[None,:]
    return V91_RANK_WEIGHTS[ranks]

def make_v10c(anchor,p5,utility,v91):
    score=anchor.astype(np.float32).copy()
    score += V10C_P5_WEIGHT*p5.astype(np.float32)
    score += V10C_UTILITY_WEIGHT*utility.astype(np.float32)
    score += rank_locality_mask(anchor)*(v91-anchor)
    return score.astype(np.float32)

p5_oof,p5_test,utility_oof,utility_test=fit_v10_heads_global()
print("V10 P5 ordinary diagnostic",round(ndcg5(y,p5_oof),6))
print("V10 utility ordinary diagnostic",round(ndcg5(y,utility_oof),6))

# ---- cell 16 (markdown) ----
# ## Strict joint reconstruction of champion + both experts
#
# Every outer validation user is untouched by all model fitting and expert construction. V10-C reuses the same outer-training sparse / seed-42 relevance representation as the champion and adds only its own fold-safe P5 and utility heads.

# ---- cell 17 (code) ----
def inner_folds(trg):
    return list(StratifiedKFold(n_splits=N_INNER,shuffle=True,random_state=SEED).split(
        np.zeros(len(trg)),strata[trg]))

def nested_base(trg,vag,inn):
    n=len(trg); raw=np.zeros((n,N_MODULES),np.float32); top=np.zeros_like(raw)
    for lt,lv in inn:
        gtr=trg[lt]; gva=trg[lv]
        mr=Ridge(alpha=RIDGE_ALPHA,solver="lsqr",tol=2e-3,max_iter=1000)
        mt=Ridge(alpha=RIDGE_ALPHA,solver="lsqr",tol=2e-3,max_iter=1000)
        mr.fit(X[gtr],y[gtr]); mt.fit(X[gtr],Y_TOP[gtr])
        raw[lv]=mr.predict(X[gva]).astype(np.float32); top[lv]=mt.predict(X[gva]).astype(np.float32)
    mr=Ridge(alpha=RIDGE_ALPHA,solver="lsqr",tol=2e-3,max_iter=1000)
    mt=Ridge(alpha=RIDGE_ALPHA,solver="lsqr",tol=2e-3,max_iter=1000)
    mr.fit(X[trg],y[trg]); mt.fit(X[trg],Y_TOP[trg])
    val=(.9*mr.predict(X[vag])+.1*mt.predict(X[vag])).astype(np.float32)
    return (.9*raw+.1*top).astype(np.float32),val

def nested_anchor_relevance(trg,vag,btr,bva,inn):
    Xt=[np.column_stack([btr[:,m],USER_FEATURES[trg]]).astype(np.float32) for m in range(N_MODULES)]
    Xv=[np.column_stack([bva[:,m],USER_FEATURES[vag]]).astype(np.float32) for m in range(N_MODULES)]
    ao=np.zeros((len(trg),N_MODULES),np.float32); av=np.zeros((len(vag),N_MODULES),np.float32)
    its={mid:[] for mid in MODULES}
    for m,mid in enumerate(MODULES):
        yy=Y_BIN[trg,m].astype(np.int32)
        for f,(lt,lv) in enumerate(inn):
            clf=make_cal(SEED+f+31*m)
            clf.fit(Xt[m][lt],yy[lt],eval_set=[(Xt[m][lv],yy[lv])],
                    callbacks=[lgb.early_stopping(LGB_EARLY_STOP,verbose=False)])
            it=int(clf.best_iteration_ or LGB_ESTIMATORS); its[mid].append(it)
            ao[lv,m]=clf.predict_proba(Xt[m][lv],num_iteration=it)[:,1].astype(np.float32)
        fi=max(20,int(np.median(its[mid])))
        full=make_cal(SEED+997+31*m,fi); full.fit(Xt[m],yy)
        av[:,m]=full.predict_proba(Xv[m])[:,1].astype(np.float32)
    return ao,av,its,Xt,Xv

def nested_v91_extension(trg,vag,btr,bva,ao,av,its,Xt,Xv,inn):
    Do=[]; Dv=[]
    for s in DIVERSITY_SEEDS:
        oo=np.zeros_like(ao); vv=np.zeros_like(av)
        for m,mid in enumerate(MODULES):
            yy=Y_BIN[trg,m].astype(np.int32)
            for f,(lt,lv) in enumerate(inn):
                it=int(its[mid][f]); clf=make_cal(s+f+31*m,it); clf.fit(Xt[m][lt],yy[lt])
                oo[lv,m]=clf.predict_proba(Xt[m][lv])[:,1].astype(np.float32)
            fi=max(20,int(np.median(its[mid])))
            full=make_cal(s+997+31*m,fi); full.fit(Xt[m],yy)
            vv[:,m]=full.predict_proba(Xv[m])[:,1].astype(np.float32)
        Do.append(align_mat(oo,ao,oo)); Dv.append(align_mat(oo,ao,vv))
    do=np.mean(np.stack(Do),0).astype(np.float32); dv=np.mean(np.stack(Dv),0).astype(np.float32)

    rp=RANK_POS[trg]; probs_o={}; probs_v={}
    for tt in range(1,6):
        po=np.zeros_like(ao); pv=np.zeros_like(av); yyall=(rp<=tt).astype(np.int32)
        for m in range(N_MODULES):
            oits=[]
            for f,(lt,lv) in enumerate(inn):
                clf=make_ord_cal(SEED+9000+101*tt+f+31*m)
                clf.fit(Xt[m][lt],yyall[lt,m],eval_set=[(Xt[m][lv],yyall[lv,m])],
                        callbacks=[lgb.early_stopping(ORD_EARLY_STOP,verbose=False)])
                it=int(clf.best_iteration_ or ORD_ESTIMATORS); oits.append(it)
                po[lv,m]=clf.predict_proba(Xt[m][lv],num_iteration=it)[:,1].astype(np.float32)
            fi=max(20,int(np.median(oits)))
            full=make_ord_cal(SEED+12000+101*tt+31*m,fi); full.fit(Xt[m],yyall[:,m])
            pv[:,m]=full.predict_proba(Xv[m])[:,1].astype(np.float32)
        probs_o[tt]=po; probs_v[tt]=pv
    oo=exp_rel(ao,probs_o); ov=exp_rel(av,probs_v)

    atr=(.55*btr+.45*ao).astype(np.float32); ava=(.55*bva+.45*av).astype(np.float32)
    dtr=(.55*btr+.45*do).astype(np.float32); dva=(.55*bva+.45*dv).astype(np.float32)
    v9tr=((1-ORDINAL_WEIGHT)*dtr+ORDINAL_WEIGHT*oo).astype(np.float32)
    v9va=((1-ORDINAL_WEIGHT)*dva+ORDINAL_WEIGHT*ov).astype(np.float32)
    inner_id=fold_ids_from_splits(len(trg),inn)
    evid=module_ablation_fold_deltas(y[trg],atr,v9tr,inner_id,N_INNER)
    support,pos=module_support(evid); scale,_=select_shrinkage(y[trg],atr,v9tr,support)
    return v9tr,v9va,apply_shrinkage(ava,v9va,support,scale),support,pos,scale

def nested_v10_heads(trg,vag,Xt,Xv,inn):
    p5_o=np.zeros((len(trg),N_MODULES),np.float32); p5_v=np.zeros((len(vag),N_MODULES),np.float32)
    u_o=np.zeros_like(p5_o); u_v=np.zeros_like(p5_v)
    for m in range(N_MODULES):
        yp=Y_TOP5[trg,m].astype(np.int32); yu=Y_UTILITY[trg,m].astype(np.float32)
        p_iters=[]; u_iters=[]
        for f,(lt,lv) in enumerate(inn):
            clf=make_cal(SEED+21000+f+31*m)
            clf.fit(Xt[m][lt],yp[lt],eval_set=[(Xt[m][lv],yp[lv])],
                    callbacks=[lgb.early_stopping(LGB_EARLY_STOP,verbose=False)])
            pit=int(clf.best_iteration_ or LGB_ESTIMATORS); p_iters.append(pit)
            p5_o[lv,m]=clf.predict_proba(Xt[m][lv],num_iteration=pit)[:,1].astype(np.float32)

            reg=make_utility_reg(SEED+23000+f+31*m)
            reg.fit(Xt[m][lt],yu[lt],eval_set=[(Xt[m][lv],yu[lv])],
                    callbacks=[lgb.early_stopping(UTILITY_EARLY_STOP,verbose=False)])
            uit=int(reg.best_iteration_ or UTILITY_ESTIMATORS); u_iters.append(uit)
            u_o[lv,m]=reg.predict(Xt[m][lv],num_iteration=uit).astype(np.float32)

        pfull=make_cal(SEED+22000+31*m,max(20,int(np.median(p_iters)))); pfull.fit(Xt[m],yp)
        p5_v[:,m]=pfull.predict_proba(Xv[m])[:,1].astype(np.float32)
        ufull=make_utility_reg(SEED+24000+31*m,max(20,int(np.median(u_iters)))); ufull.fit(Xt[m],yu)
        u_v[:,m]=ufull.predict(Xv[m]).astype(np.float32)
    return p5_o,p5_v,u_o,u_v

ref_fold_id=np.full(N_TRAIN,-1,np.int16)
for f,(_,va) in enumerate(ref_folds): ref_fold_id[va]=f

strict_base=np.zeros_like(y,np.float32); strict_meta=np.zeros_like(y,np.float32)
strict_consensus3=np.zeros_like(y,np.float32); strict_cal045=np.zeros_like(y,np.float32)
strict_v9=np.zeros_like(y,np.float32); strict_v91=np.zeros_like(y,np.float32)
strict_v10c=np.zeros_like(y,np.float32)
v91_outer=[]; v10c_outer=[]

for f,(trg,vag) in enumerate(ref_folds):
    print(f"strict dual outer fold {f+1}/{N_FOLDS}")
    inn=inner_folds(trg)
    btr,bva=nested_base(trg,vag,inn)
    ao,av,its,Xt,Xv=nested_anchor_relevance(trg,vag,btr,bva,inn)
    _,v9va,v91va,support,pos,scale=nested_v91_extension(trg,vag,btr,bva,ao,av,its,Xt,Xv,inn)
    _,p5v,_,uv=nested_v10_heads(trg,vag,Xt,Xv,inn)

    cva=apply_profile(bva,av)
    ava=(.55*bva+.45*av).astype(np.float32)
    v10cva=make_v10c(ava,p5v,uv,v91va)

    strict_base[vag]=bva; strict_meta[vag]=av; strict_consensus3[vag]=cva
    strict_cal045[vag]=ava; strict_v9[vag]=v9va; strict_v91[vag]=v91va; strict_v10c[vag]=v10cva
    v91_outer.append(ndcg5(y[vag],v91va)-ndcg5(y[vag],cva))
    v10c_outer.append(ndcg5(y[vag],v10cva)-ndcg5(y[vag],cva))
    print(" champion",round(ndcg5(y[vag],cva),6),
          "v91",round(ndcg5(y[vag],v91va),6),"d",round(v91_outer[-1],6),
          "v10c",round(ndcg5(y[vag],v10cva),6),"d",round(v10c_outer[-1],6),
          "v91scale",scale)
    gc.collect()

# Final V9.1 test control from strict outer evidence only.
outer_evid=module_ablation_fold_deltas(y,strict_cal045,strict_v9,ref_fold_id,N_FOLDS)
final_support,final_pos=module_support(outer_evid)
final_scale,_=select_shrinkage(y,strict_cal045,strict_v9,final_support)
v91_test=apply_shrinkage(cal045_test,v9_test,final_support,final_scale)
v10c_test=make_v10c(cal045_test,p5_test,utility_test,v91_test)

print("strict Consensus3",round(ndcg5(y,strict_consensus3),6),"public",PUBLIC_CHAMPION_LB)
print("strict V9.1",round(ndcg5(y,strict_v91),6),"historical public",PUBLIC_V91_LB)
print("strict V10-C",round(ndcg5(y,strict_v10c),6))

# ---- cell 18 (markdown) ----
# ## Dual-expert score-geometry router
#
# The router does not see raw chat text, assessment answers, `user_id`, or test labels. It sees only how an expert's 17-module scores/ranks disagree with the champion. Transition evidence is learned from training folds only.

# ---- cell 19 (code) ----
def top5_order(scores):
    idx=np.argpartition(-scores,4,axis=1)[:,:5]; local=np.take_along_axis(scores,idx,axis=1)
    return np.take_along_axis(idx,np.argsort(-local,axis=1),axis=1)

def score_ranks(scores):
    order=np.argsort(-scores,axis=1); ranks=np.empty_like(order)
    ranks[np.arange(len(scores))[:,None],order]=np.arange(N_MODULES)[None,:]
    return ranks

def geometry_features(champion,expert):
    champion=np.asarray(champion,np.float32); expert=np.asarray(expert,np.float32); eps=1e-6
    cz=(champion-champion.mean(1,keepdims=True))/(champion.std(1,keepdims=True)+eps)
    ez=(expert-expert.mean(1,keepdims=True))/(expert.std(1,keepdims=True)+eps)
    cr=score_ranks(champion); er=score_ranks(expert)
    ctop=(cr<5).astype(np.float32); etop=(er<5).astype(np.float32)
    co=np.argsort(-champion,axis=1); eo=np.argsort(-expert,axis=1)
    cs=np.take_along_axis(champion,co,axis=1); es=np.take_along_axis(expert,eo,axis=1)
    scal=np.zeros((len(champion),5),np.float32)
    for i in range(len(champion)):
        scal[i]=[float(co[i,0]==eo[i,0]),len(set(co[i,:3])&set(eo[i,:3]))/3.0,
                 len(set(co[i,:5])&set(eo[i,:5]))/5.0,float(np.mean(np.abs(cr[i]-er[i]))),
                 float(np.max(np.abs(cr[i]-er[i])))]
    cmar=np.column_stack([cs[:,j]-cs[:,j+1] for j in range(5)])
    emar=np.column_stack([es[:,j]-es[:,j+1] for j in range(5)])
    extra=np.column_stack([champion.std(1),expert.std(1),champion.max(1)-champion.min(1),
                           expert.max(1)-expert.min(1),np.linalg.norm(cz-ez,axis=1)])
    Xg=np.column_stack([scal,cmar,emar,emar-cmar,extra,ctop,etop,
                        cr/(N_MODULES-1),er/(N_MODULES-1),(er-cr)/(N_MODULES-1),cz,ez,ez-cz]).astype(np.float32)
    assert np.isfinite(Xg).all(); return Xg

def transition_keys(champion,expert):
    tc=top5_order(champion); te=top5_order(expert); keys=[]
    for i in range(len(champion)):
        out=tuple(sorted(set(tc[i])-set(te[i]))); inn=tuple(sorted(set(te[i])-set(tc[i])))
        if out or inn: keys.append(("set",out,inn))
        elif np.any(tc[i]!=te[i]): keys.append(("order_top1",int(tc[i,0]),int(te[i,0])))
        else: keys.append(("same",))
    return keys

def fit_router_gate(Xg,delta,train_idx,seed):
    train_idx=np.asarray(train_idx,int); beneficial=(delta>0).astype(np.int32)
    weights=np.where(np.abs(delta[train_idx])<1e-12,0.20,
                     1.0+12.0*np.minimum(np.abs(delta[train_idx]),0.12)).astype(np.float32)
    model=lgb.LGBMClassifier(
        n_estimators=GATE_ESTIMATORS,learning_rate=.025,num_leaves=GATE_LEAVES,
        min_child_samples=GATE_MIN_CHILD,reg_lambda=GATE_REG_L2,reg_alpha=GATE_REG_L1,
        subsample=.85,colsample_bytree=.75,verbosity=-1,random_state=int(seed),n_jobs=4)
    model.fit(Xg[train_idx],beneficial[train_idx],sample_weight=weights)
    return model

def transition_stats(keys,delta,fold_ids,train_idx):
    from collections import defaultdict
    buckets=defaultdict(lambda:defaultdict(list))
    for i in np.asarray(train_idx,int):
        k=keys[i]
        if k==("same",): continue
        buckets[k][int(fold_ids[i])].append(float(delta[i]))
    stats={}
    for k,byfold in buckets.items():
        means=[float(np.mean(v)) for v in byfold.values()]
        n=sum(len(v) for v in byfold.values()); total=sum(sum(v) for v in byfold.values())
        stats[k]={"support":int(n),"mean":float(total/max(n,1)),
                  "positive_folds":int(sum(v>0 for v in means))}
    return stats

def transition_query(keys_query,stats):
    accept=np.zeros(len(keys_query),bool); mean=np.zeros(len(keys_query),np.float32); support=np.zeros(len(keys_query),np.int32)
    for i,k in enumerate(keys_query):
        s=stats.get(k)
        if s is not None:
            mean[i]=float(s["mean"]); support[i]=int(s["support"])
            accept[i]=(s["support"]>=TRANSITION_MIN_SUPPORT and
                       s["positive_folds"]>=TRANSITION_MIN_POS_FOLDS and s["mean"]>0)
    return accept,mean,support

def top_fraction_mask(prob,fraction):
    prob=np.asarray(prob,float); k=max(1,int(round(float(fraction)*len(prob))))
    idx=np.argpartition(-prob,k-1)[:k]; out=np.zeros(len(prob),bool); out[idx]=True; return out

def route_from_fold_predictions(probs,accepts,means,fold_ids,budget_v91,budget_v10c):
    route=np.zeros(len(fold_ids),np.int8)  # 0 champ, 1 v91, 2 v10c
    for f in range(N_FOLDS):
        va=np.where(fold_ids==f)[0]
        c1=top_fraction_mask(probs["v91"][va],budget_v91) & accepts["v91"][va]
        c2=top_fraction_mask(probs["v10c"][va],budget_v10c) & accepts["v10c"][va]
        u1=probs["v91"][va]*np.maximum(means["v91"][va],0)
        u2=probs["v10c"][va]*np.maximum(means["v10c"][va],0)
        route[va[c1 & ~c2]]=1; route[va[c2 & ~c1]]=2
        both=c1 & c2
        if both.any():
            loc=np.where(both)[0]; choose2=u2[loc]>u1[loc]
            route[va[loc]]=1; route[va[loc[choose2]]]=2
    return route

def apply_route(champion,v91,v10c,route):
    out=np.asarray(champion,np.float32).copy(); out[route==1]=v91[route==1]; out[route==2]=v10c[route==2]
    return out

# ---- cell 20 (markdown) ----
# ## Strict cross-fitted dual routing
#
# Each expert gate is fit on four outer folds and predicts only the untouched fifth fold. Transition evidence is also built only from those four folds. The fixed 12.5% / 5% route budgets are then applied independently inside the untouched fold.

# ---- cell 21 (code) ----
champ_u=per_user_ndcg(y,strict_consensus3)
expert_scores={"v91":strict_v91,"v10c":strict_v10c}
expert_u={k:per_user_ndcg(y,v) for k,v in expert_scores.items()}
expert_delta={k:(expert_u[k]-champ_u).astype(np.float64) for k in expert_scores}
router_X={k:geometry_features(strict_consensus3,v) for k,v in expert_scores.items()}
router_keys={k:transition_keys(strict_consensus3,v) for k,v in expert_scores.items()}

cv_prob={k:np.zeros(N_TRAIN,np.float32) for k in expert_scores}
cv_accept={k:np.zeros(N_TRAIN,bool) for k in expert_scores}
cv_mean={k:np.zeros(N_TRAIN,np.float32) for k in expert_scores}
cv_support={k:np.zeros(N_TRAIN,np.int32) for k in expert_scores}

for f in range(N_FOLDS):
    tr=np.where(ref_fold_id!=f)[0]; va=np.where(ref_fold_id==f)[0]
    for j,name in enumerate(["v91","v10c"]):
        gate=fit_router_gate(router_X[name],expert_delta[name],tr,SEED+13000+100*j+f)
        cv_prob[name][va]=gate.predict_proba(router_X[name][va])[:,1].astype(np.float32)
        st=transition_stats(router_keys[name],expert_delta[name],ref_fold_id,tr)
        acc,mu,sup=transition_query([router_keys[name][i] for i in va],st)
        cv_accept[name][va]=acc; cv_mean[name][va]=mu; cv_support[name][va]=sup

route=route_from_fold_predictions(cv_prob,cv_accept,cv_mean,ref_fold_id,V91_ROUTE_FRACTION,V10C_ROUTE_FRACTION)
dual_router=apply_route(strict_consensus3,strict_v91,strict_v10c,route)
dual_u=per_user_ndcg(y,dual_router); dual_user_delta=dual_u-champ_u
dual_increment=float(dual_user_delta.mean()); dual_ci=boot_ci(dual_user_delta)

fold_deltas=[]; fold_records=[]
for f in range(N_FOLDS):
    va=(ref_fold_id==f); d=float(dual_user_delta[va].mean()); fold_deltas.append(d)
    fold_records.append({"fold":f,"increment":d,"route_rate":float(np.mean(route[va]!=0)),
                         "v91_rate":float(np.mean(route[va]==1)),"v10c_rate":float(np.mean(route[va]==2))})
fold_wins=int(sum(v>0 for v in fold_deltas)); worst_fold=float(min(fold_deltas))

bounds=np.linspace(0,N_TRAIN,3,dtype=int) if V131_SMOKE else np.asarray([0,1000,2000,3000,4000],int)
quarter_deltas=[float(dual_user_delta[bounds[i]:bounds[i+1]].mean()) for i in range(len(bounds)-1)]
quarter_wins=int(sum(v>0 for v in quarter_deltas)); tail_increment=float(dual_user_delta[-(N_TRAIN//2 if V131_SMOKE else 1000):].mean())

print("STRICT champion",round(float(champ_u.mean()),6),"dual router",round(float(dual_u.mean()),6),
      "increment",round(dual_increment,6),"CI",tuple(round(v,6) for v in dual_ci))
print("folds",[round(v,6) for v in fold_deltas],"wins",fold_wins,"worst",round(worst_fold,6))
print("quarters",[round(v,6) for v in quarter_deltas],"tail",round(tail_increment,6))
print("route counts",{"champ":int((route==0).sum()),"v91":int((route==1).sum()),"v10c":int((route==2).sum())})

# ---- cell 22 (markdown) ----
# ## Second-order budget diagnostic
#
# This block does **not** tune deployment. For each held-out outer fold it chooses one budget pair using only the other four folds' already cross-fitted router outcomes, then evaluates that frozen choice once on the untouched fold. It answers whether the diagnosed `(12.5%, 5%)` neighborhood is stable rather than a single full-OOF optimum.

# ---- cell 23 (code) ----
def evaluate_budget_pair(b1,b2):
    r=route_from_fold_predictions(cv_prob,cv_accept,cv_mean,ref_fold_id,b1,b2)
    s=apply_route(strict_consensus3,strict_v91,strict_v10c,r)
    return r,per_user_ndcg(y,s)-champ_u

budget_cache={}
for b1,b2 in SECOND_ORDER_BUDGETS:
    budget_cache[(b1,b2)]=evaluate_budget_pair(b1,b2)

second_order=[]; second_order_delta=np.zeros(N_TRAIN,np.float64); selected_budgets=[]
for f in range(N_FOLDS):
    tr=(ref_fold_id!=f); va=(ref_fold_id==f)
    chosen=max(SECOND_ORDER_BUDGETS,key=lambda b:float(budget_cache[b][1][tr].mean()))
    selected_budgets.append(chosen); second_order_delta[va]=budget_cache[chosen][1][va]
    second_order.append({"heldout_fold":f,"selected_v91_budget":chosen[0],"selected_v10c_budget":chosen[1],
                         "train_increment":float(budget_cache[chosen][1][tr].mean()),
                         "heldout_increment":float(budget_cache[chosen][1][va].mean())})

second_order_mean=float(second_order_delta.mean()); second_order_ci=boot_ci(second_order_delta)
second_order_wins=int(sum(r["heldout_increment"]>0 for r in second_order))
print("second_order",round(second_order_mean,6),"CI",tuple(round(v,6) for v in second_order_ci),
      "wins",second_order_wins,"selected",selected_budgets)
print(pd.DataFrame(second_order).to_string(index=False))

# ---- cell 24 (markdown) ----
# ## Full-data dual-router inference
#
# Both final gates and both transition tables are fit only on strict cross-fitted train evidence. Test users are routed with the same frozen budgets and conflict-resolution rule.

# ---- cell 25 (code) ----
all_idx=np.arange(N_TRAIN)
full_gate={}; full_stats={}; test_prob={}; test_accept={}; test_mean={}; test_support={}
expert_test={"v91":v91_test,"v10c":v10c_test}
for j,name in enumerate(["v91","v10c"]):
    full_gate[name]=fit_router_gate(router_X[name],expert_delta[name],all_idx,SEED+17777+100*j)
    Xte=geometry_features(consensus3_test,expert_test[name])
    test_prob[name]=full_gate[name].predict_proba(Xte)[:,1].astype(np.float32)
    full_stats[name]=transition_stats(router_keys[name],expert_delta[name],ref_fold_id,all_idx)
    tkeys=transition_keys(consensus3_test,expert_test[name])
    acc,mu,sup=transition_query(tkeys,full_stats[name]); test_accept[name]=acc; test_mean[name]=mu; test_support[name]=sup

# One synthetic fold so the same frozen route builder can be reused without changing policy.
test_fold=np.zeros(N_TEST,np.int16)
old_nf=N_FOLDS
N_FOLDS=1
test_route=route_from_fold_predictions(test_prob,test_accept,test_mean,test_fold,V91_ROUTE_FRACTION,V10C_ROUTE_FRACTION)
N_FOLDS=old_nf
router_test=apply_route(consensus3_test,v91_test,v10c_test,test_route)

test_route_rate=float(np.mean(test_route!=0))
test_top5_change=top5_set_change_rate(router_test,consensus3_test)
test_top5_order_change=float(np.mean(np.any(top5_order(router_test)!=top5_order(consensus3_test),axis=1)))
test_top1_change=float(np.mean(np.argmax(router_test,axis=1)!=np.argmax(consensus3_test,axis=1)))
print("test routes",{"champ":int((test_route==0).sum()),"v91":int((test_route==1).sum()),"v10c":int((test_route==2).sum())})
print("test changes top5-set",round(test_top5_change,4),"top5-order",round(test_top5_order_change,4),"top1",round(test_top1_change,4))

# ---- cell 26 (markdown) ----
# ## Promotion gate, exports, and audit
#
# The primary file is fail-closed: if the dual router does not beat the exact champion under the frozen strict criteria, it becomes an exact Consensus3 fallback.

# ---- cell 27 (code) ----
promotion_gate=(
    dual_increment>=MIN_ROUTER_INCREMENT and fold_wins>=MIN_OUTER_WINS and
    worst_fold>=MAX_WORST_FOLD and dual_ci[0]>MIN_BOOT_CI_LOW and
    quarter_wins>=MIN_QUARTER_WINS and tail_increment>MIN_TAIL_INCREMENT and
    test_route_rate<=MAX_TEST_ROUTE_RATE and test_top5_change<=MAX_TEST_TOP5_CHANGE and
    test_top1_change<=MAX_TEST_TOP1_CHANGE
)

print("V13.1 DECISION",{
    "strict_champion":round(float(champ_u.mean()),6),"strict_dual":round(float(dual_u.mean()),6),
    "increment":round(dual_increment,6),"ci":[round(v,6) for v in dual_ci],
    "fold_wins":fold_wins,"quarter_wins":quarter_wins,"tail":round(tail_increment,6),
    "second_order":round(second_order_mean,6),"second_order_wins":second_order_wins,
    "test_route":round(test_route_rate,4),"test_top5_change":round(test_top5_change,4),
    "gate":bool(promotion_gate)
})

def save_submission(scores,name):
    root=Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('/mnt/data')
    out=sample.copy(); out.loc[:,MODULES]=np.asarray(scores,np.float32)
    path=root/name; out.to_csv(path,index=False); chk=pd.read_csv(path)
    assert chk.shape==sample.shape and chk.user_id.is_unique and chk[MODULES].notna().all().all()
    assert chk.user_id.tolist()==sample.user_id.tolist(); print("saved",path); return path

root=Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('/mnt/data')
p_champ=save_submission(consensus3_test,'submission_v13_1_consensus3_reference.csv')
p_v91=save_submission(v91_test,'submission_v13_1_v91_expert.csv')
p_v10=save_submission(v10c_test,'submission_v13_1_v10c_expert.csv')
p_router=save_submission(router_test,'submission_v13_1_dual_router.csv')
p_primary=save_submission(router_test if promotion_gate else consensus3_test,'submission_v13_1_primary.csv')

# Test routing audit contains no target information.
audit=[]
for i in np.where(test_route!=0)[0]:
    name='v91' if test_route[i]==1 else 'v10c'
    audit.append({"user_id":test.user_id.iloc[i],"expert":name,"gate_probability":float(test_prob[name][i]),
                  "transition_mean":float(test_mean[name][i]),"transition_support":int(test_support[name][i]),
                  "utility_score":float(test_prob[name][i]*max(float(test_mean[name][i]),0.0))})
pd.DataFrame(audit).to_csv(root/'v13_1_routed_test_users.csv',index=False)

pd.DataFrame(fold_records+[{"fold":f"quarter_{i+1}","increment":v} for i,v in enumerate(quarter_deltas)]).to_csv(
    root/'v13_1_router_diagnostics.csv',index=False)
pd.DataFrame(second_order).to_csv(root/'v13_1_second_order.csv',index=False)

summary={
    "public_champion_lb":PUBLIC_CHAMPION_LB,
    "config":{"v91_route_fraction":V91_ROUTE_FRACTION,"v10c_route_fraction":V10C_ROUTE_FRACTION,
              "transition_min_support":TRANSITION_MIN_SUPPORT,"transition_min_positive_folds":TRANSITION_MIN_POS_FOLDS,
              "gate_min_child":GATE_MIN_CHILD,"gate_leaves":GATE_LEAVES},
    "strict_consensus3":float(champ_u.mean()),"strict_v91":float(expert_u['v91'].mean()),
    "strict_v10c":float(expert_u['v10c'].mean()),"strict_dual_router":float(dual_u.mean()),
    "router_increment":dual_increment,"bootstrap_ci":[float(v) for v in dual_ci],
    "fold_deltas":[float(v) for v in fold_deltas],"fold_wins":fold_wins,"worst_fold":worst_fold,
    "quarter_deltas":[float(v) for v in quarter_deltas],"quarter_wins":quarter_wins,"tail_increment":tail_increment,
    "oof_route_rate":float(np.mean(route!=0)),"oof_v91_users":int(np.sum(route==1)),"oof_v10c_users":int(np.sum(route==2)),
    "second_order_mean":second_order_mean,"second_order_ci":[float(v) for v in second_order_ci],
    "second_order_wins":second_order_wins,"second_order_selected":[list(x) for x in selected_budgets],
    "test_route_rate":test_route_rate,"test_v91_users":int(np.sum(test_route==1)),"test_v10c_users":int(np.sum(test_route==2)),
    "test_top5_change":test_top5_change,"test_top5_order_change":test_top5_order_change,"test_top1_change":test_top1_change,
    "promotion_gate":bool(promotion_gate),"primary_label":"dual_router" if promotion_gate else "consensus3_fallback",
    "primary_file":p_primary.name,"router_file":p_router.name,
}
json.dump(summary,open(root/'v13_1_summary.json','w'),indent=2)
np.savez_compressed(root/'v13_1_oof_artifacts.npz',y=y,ref_fold_id=ref_fold_id,
                    strict_consensus3=strict_consensus3,strict_v91=strict_v91,strict_v10c=strict_v10c,
                    dual_router=dual_router,route=route,dual_user_delta=dual_user_delta,
                    consensus3_test=consensus3_test,v91_test=v91_test,v10c_test=v10c_test,
                    router_test=router_test,test_route=test_route)

# ---- cell 28 (markdown) ----
# ## Hard invariants
#
# These checks protect the public champion fallback, file shape/order, routing action space, and absence of accidental NaNs.

# ---- cell 29 (code) ----
assert len(MODULES)==17 and y.shape==(4000,17) and consensus3_test.shape==(1000,17)
assert strict_consensus3.shape==strict_v91.shape==strict_v10c.shape==dual_router.shape==y.shape
assert np.isfinite(strict_consensus3).all() and np.isfinite(strict_v91).all() and np.isfinite(strict_v10c).all()
assert np.isfinite(consensus3_test).all() and np.isfinite(v91_test).all() and np.isfinite(v10c_test).all() and np.isfinite(router_test).all()
assert set(np.unique(route)).issubset({0,1,2}) and set(np.unique(test_route)).issubset({0,1,2})
# Unrouted users must be exact champion predictions.
assert np.allclose(dual_router[route==0],strict_consensus3[route==0],atol=0,rtol=0)
assert np.allclose(router_test[test_route==0],consensus3_test[test_route==0],atol=0,rtol=0)
# Primary fallback semantics.
expected_primary=router_test if promotion_gate else consensus3_test
primary_df=pd.read_csv((Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('/mnt/data'))/'submission_v13_1_primary.csv')
assert np.allclose(primary_df[MODULES].to_numpy(np.float32),expected_primary,atol=2e-7,rtol=0)
print("V13.1 hard invariants PASS")
