# Converted from notebooks/v11.ipynb: markdown cells become comments, %/! magics neutralized.

# ---- cell 0 (markdown) ----
# # MineToday V11 — Temporal-Robust Cal045
#
# **Goal:** recover the public strength of `cal045` (LB **0.65639**) while adapting only to evidence that survives **rolling next-block validation**.
#
# V9.1 was the strongest random nested experiment offline but collapsed on public LB (`0.65432`). That failure is treated as evidence that the residual family was selected under the wrong validation regime. Train users are ordered `U_0001..U_4000`, while test is the next contiguous block `U_4001..U_5000`; V11 therefore promotes corrections using **rolling-origin future-block validation**, not shuffled OOF improvement.
#
# ### Design
#
# - exact public-validated **word + char TF-IDF + Ridge** backbone;
# - exact **seed-42 per-module relevance calibration, w=0.45** as immutable deployment anchor;
# - a **global relevance calibrator** and **global Top-5 calibrator** define the historical regime;
# - recent experts are fit only on the most recent `{1500, 2000, 3000}` labeled users available at each rolling origin;
# - V11 uses only the **recent-minus-global calibration residual**, never raw recent probabilities;
# - the correction is restricted to anchor ranks **3–8**, and anchor ranks **1–2 are mathematically preserved**;
# - every block uses a deterministic **5% Top-5 change budget**, with automatic alpha backoff;
# - candidate hyperparameters are selected by recency-weighted rolling performance plus stability, then checked by **sequential one-step-ahead policy validation** and a fixed-policy shuffled-CV sanity test;
# - final inference uses the exact full-data cal045 anchor plus a single recent expert trained on the selected tail window. No fold-bag residual, no ordinal V9/V9.1 residual, no semantic encoder, and no pairwise rank-5/6 classifier.
#
# ### Competition compliance
#
# Only organizer-provided files are used. No external dataset, direct LLM test prediction, pretrained model, or paid model is used.
#
# ### Kaggle runtime
#
# Ready for Kaggle **T4 ×2**. This pipeline is intentionally CPU-based because the sparse supervised family is the only family with repeated positive public evidence. GPUs may remain idle.
#
# ### Outputs
#
# - `submission_v11_candidate.csv` — actual temporal candidate;
# - `submission_v11_primary.csv` — candidate only if the pre-registered temporal gate passes, otherwise exact cal045 fallback;
# - `submission_v11_cal045_reference.csv` — immutable public anchor;
# - `v11_summary.json` — rolling, sequential, shuffled sanity, deployment, and gate diagnostics;
# - `v11_rolling_candidates.csv` — all pre-registered policy results;
# - `v11_artifacts.npz` — compact prediction diagnostics.
#
# **Read the final `V11 DECISION` block before submitting anything.**

# ---- cell 1 (code) ----
import os,re,json,ast,html,hashlib,random,warnings,gc,time,sys,subprocess
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS","4")
os.environ.setdefault("OPENBLAS_NUM_THREADS","4")
os.environ.setdefault("MKL_NUM_THREADS","4")
os.environ["PYTHONHASHSEED"]="42"
os.environ["TOKENIZERS_PARALLELISM"]="false"

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
V11_SMOKE=bool(int(os.getenv("V11_SMOKE","0")))
V91_SMOKE=V11_SMOKE  # compatibility with the exact V3.1 sparse cell
N_FOLDS=2 if V11_SMOKE else 5
N_INNER=2 if V11_SMOKE else 3
RIDGE_ALPHA=3.0
REFERENCE_W=0.45
LGB_ESTIMATORS=40 if V11_SMOKE else 1200
LGB_EARLY_STOP=5 if V11_SMOKE else 80
BOOTSTRAPS=300 if V11_SMOKE else 5000

PUBLIC_ANCHOR_LB=0.65639
LEADER_LB=0.65805
REQUIRED_PUBLIC_DELTA=LEADER_LB-PUBLIC_ANCHOR_LB

RECENT_WINDOWS=[500,800] if V11_SMOKE else [1500,2000,3000]
ALPHA_GRID=[0.25,0.50] if V11_SMOKE else [0.20,0.35,0.50,0.75]
TOP5_GAMMA_GRID=[0.0,0.25] if V11_SMOKE else [0.0,0.25,0.50]
ALPHA_BACKOFF=[1.0,0.5,0.0] if V11_SMOKE else [1.0,0.8,0.6,0.4,0.2,0.0]
MAX_TOP5_CHANGE=0.05
ROLL_WEIGHTS=np.asarray([0.10,0.15,0.25,0.50],np.float64)
ROLL_WEIGHTS/=ROLL_WEIGHTS.sum()

# Promotion thresholds are intentionally tied to the public gap and future-block stability.
MIN_ROLL_WEIGHTED_DELTA=REQUIRED_PUBLIC_DELTA
MIN_ROLL_WINS=3
MIN_SEQ_WINS=2
MIN_SEQ_WEIGHTED_DELTA=0.00020
MAX_RANDOM_SANITY_DROP=0.00030
MAX_LAST2_NEG=-0.00010
MAX_TEST_TOP5_CHANGE=0.05

random.seed(SEED); np.random.seed(SEED)
print("versions",np.__version__,pd.__version__,lgb.__version__,"V11_SMOKE",V11_SMOKE)
print("public anchor",PUBLIC_ANCHOR_LB,"leader",LEADER_LB,"required delta",round(REQUIRED_PUBLIC_DELTA,6))

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
assert np.all(np.diff(TRAIN_ID_NUM)>0), "train.csv must be ordered by increasing user_id for V11"
assert np.all(np.diff(TEST_ID_NUM)>0), "test.csv must be ordered by increasing user_id for V11"
assert TEST_ID_NUM.min()>TRAIN_ID_NUM.max(), "V11 assumes test is the next ID block"

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
# Exact V3.1 LightGBM calibration helpers.
def make_cal(seed,n_estimators=None):
    return lgb.LGBMClassifier(
        n_estimators=int(LGB_ESTIMATORS if n_estimators is None else n_estimators),
        learning_rate=.03,num_leaves=15,min_child_samples=50,
        reg_alpha=.5,reg_lambda=5.,colsample_bytree=.7,
        random_state=int(seed),n_jobs=4,verbosity=-1
    )

def fit_bin(seed,folds,Xtr,Xout,targets,fixed_iters=None):
    oo=np.zeros((len(targets),N_MODULES),np.float32)
    out=np.zeros((Xout[0].shape[0],N_MODULES),np.float32)
    its={mid:[] for mid in MODULES}
    for m,mid in enumerate(MODULES):
        target=targets[:,m].astype(np.int32)
        for f,(tr,va) in enumerate(folds):
            if fixed_iters is None:
                clf=make_cal(seed+f+31*m)
                clf.fit(Xtr[m][tr],target[tr],eval_set=[(Xtr[m][va],target[va])],callbacks=[lgb.early_stopping(LGB_EARLY_STOP,verbose=False)])
                it=int(clf.best_iteration_ or LGB_ESTIMATORS)
            else:
                it=int(fixed_iters[mid][f]); clf=make_cal(seed+f+31*m,it); clf.fit(Xtr[m][tr],target[tr])
            its[mid].append(it); oo[va,m]=clf.predict_proba(Xtr[m][va])[:,1].astype(np.float32)
        fi=max(20,int(np.median(its[mid])))
        full=make_cal(seed+997+31*m,fi); full.fit(Xtr[m],target)
        out[:,m]=full.predict_proba(Xout[m])[:,1].astype(np.float32)
    return oo,out,its

def median_iters(iters): return {mid:max(20,int(np.median(iters[mid]))) for mid in MODULES}

META_TR=[np.column_stack([base_oof[:,m],USER_FEATURES[:N_TRAIN]]).astype(np.float32) for m in range(N_MODULES)]
META_TE=[np.column_stack([base_test[:,m],USER_FEATURES[N_TRAIN:]]).astype(np.float32) for m in range(N_MODULES)]

# Immutable public anchor.
anchor_meta_oof,anchor_meta_test,anchor_iters=fit_bin(42,ref_folds,META_TR,META_TE,Y_BIN)
cal045_oof=(.55*base_oof+.45*anchor_meta_oof).astype(np.float32)
cal045_test=(.55*base_test+.45*anchor_meta_test).astype(np.float32)

# Top-5 head is NOT added to cal045. It is used only in recent-minus-global temporal evidence.
global_top5_oof,global_top5_test,top5_iters=fit_bin(4200,ref_folds,META_TR,META_TE,Y_TOP5)
print("cal045 diagnostic OOF",round(ndcg5(y,cal045_oof),6))
print("global top5 diagnostic",round(ndcg5(y,global_top5_oof),6))

# ---- cell 9 (markdown) ----
# ## Rolling-origin model assessment
#
# The four validation blocks imitate the actual competition geometry: train only on earlier IDs and evaluate on the immediately following unseen block. Hyperparameters never use Kaggle labels or public scores beyond treating `0.65639` as the immutable anchor reference.
#
# The candidate residual is
#
# `0.45 × [(1-γ)(recent relevance - global relevance) + γ(recent top5 - global top5)]`.
#
# It is applied only to the anchor's ranks 3–8. Ranks 1–2 remain unchanged, and alpha is deterministically backed off until Top-5 membership changes are at most 5%.

# ---- cell 10 (code) ----
def make_inner_splits(global_idx,seed):
    n=len(global_idx)
    return list(StratifiedKFold(N_INNER,shuffle=True,random_state=int(seed)).split(np.zeros(n),strata[global_idx]))

def fit_base_for_split(trg,vag,inner):
    n=len(trg)
    raw=np.zeros((n,N_MODULES),np.float32); top=np.zeros_like(raw)
    for lt,lv in inner:
        gtr=trg[lt]; glv=trg[lv]
        mr=Ridge(alpha=RIDGE_ALPHA,solver="lsqr",tol=2e-3,max_iter=1000)
        mt=Ridge(alpha=RIDGE_ALPHA,solver="lsqr",tol=2e-3,max_iter=1000)
        mr.fit(X[gtr],y[gtr]); mt.fit(X[gtr],Y_TOP[gtr])
        raw[lv]=mr.predict(X[glv]).astype(np.float32); top[lv]=mt.predict(X[glv]).astype(np.float32)
    mr=Ridge(alpha=RIDGE_ALPHA,solver="lsqr",tol=2e-3,max_iter=1000)
    mt=Ridge(alpha=RIDGE_ALPHA,solver="lsqr",tol=2e-3,max_iter=1000)
    mr.fit(X[trg],y[trg]); mt.fit(X[trg],Y_TOP[trg])
    val=.9*mr.predict(X[vag])+.1*mt.predict(X[vag])
    return (.9*raw+.1*top).astype(np.float32),val.astype(np.float32)

def fit_global_head_local(Xt,Xv,target_local,inner,seed):
    pred=np.zeros((len(Xv[0]),N_MODULES),np.float32); iters={}
    for m,mid in enumerate(MODULES):
        yy=target_local[:,m].astype(np.int32); its=[]
        for f,(lt,lv) in enumerate(inner):
            clf=make_cal(seed+f+31*m)
            clf.fit(Xt[m][lt],yy[lt],eval_set=[(Xt[m][lv],yy[lv])],callbacks=[lgb.early_stopping(LGB_EARLY_STOP,verbose=False)])
            its.append(int(clf.best_iteration_ or LGB_ESTIMATORS))
        fi=max(20,int(np.median(its))); iters[mid]=fi
        full=make_cal(seed+997+31*m,fi); full.fit(Xt[m],yy)
        pred[:,m]=full.predict_proba(Xv[m])[:,1].astype(np.float32)
    return pred,iters

def fit_recent_head_local(Xt,Xv,target_local,recent_local,iters,seed):
    pred=np.zeros((len(Xv[0]),N_MODULES),np.float32)
    for m,mid in enumerate(MODULES):
        yy=target_local[:,m].astype(np.int32)
        # Defensive fallback for an extremely rare class in a tiny smoke window.
        uniq=np.unique(yy[recent_local])
        if len(uniq)<2:
            pred[:,m]=float(uniq[0]) if len(uniq) else float(yy.mean())
            continue
        clf=make_cal(seed+31*m,iters[mid]); clf.fit(Xt[m][recent_local],yy[recent_local])
        pred[:,m]=clf.predict_proba(Xv[m])[:,1].astype(np.float32)
    return pred

def temporal_shift(global_rel,recent_rel,global_top,recent_top,gamma):
    return (.45*((1.0-float(gamma))*(recent_rel-global_rel)+float(gamma)*(recent_top-global_top))).astype(np.float32)

def apply_temporal_once(anchor,shift,alpha):
    anchor=np.asarray(anchor,np.float32); shift=np.asarray(shift,np.float32)
    order=np.argsort(-anchor,axis=1)
    cand=anchor.copy()
    # Only original anchor ranks 3..8 can receive temporal corrections.
    rows=np.arange(len(anchor))[:,None]; editable=order[:,2:8]
    cand[rows,editable]+=float(alpha)*shift[rows,editable]
    # Preserve original rank-1/rank-2 modules and their ordering exactly.
    rank2=order[:,1]
    cap=anchor[np.arange(len(anchor)),rank2]-1e-6*np.maximum(1.0,np.abs(anchor[np.arange(len(anchor)),rank2]))
    for r in range(len(anchor)):
        cand[r,editable[r]]=np.minimum(cand[r,editable[r]],cap[r])
    return cand.astype(np.float32)

def apply_temporal_budget(anchor,shift,alpha,max_change=MAX_TOP5_CHANGE):
    for mult in ALPHA_BACKOFF:
        eff=float(alpha)*float(mult)
        cand=apply_temporal_once(anchor,shift,eff)
        change=top5_set_change_rate(cand,anchor)
        if change<=float(max_change)+1e-12:
            return cand,eff,change
    return anchor.copy(),0.0,0.0

def build_split_predictions(trg,vag,windows,seed):
    inner=make_inner_splits(trg,seed)
    btr,bva=fit_base_for_split(trg,vag,inner)
    Xt=[np.column_stack([btr[:,m],USER_FEATURES[trg]]).astype(np.float32) for m in range(N_MODULES)]
    Xv=[np.column_stack([bva[:,m],USER_FEATURES[vag]]).astype(np.float32) for m in range(N_MODULES)]
    rel_global,rel_iters=fit_global_head_local(Xt,Xv,Y_BIN[trg],inner,seed+10000)
    top_global,top_iters_local=fit_global_head_local(Xt,Xv,Y_TOP5[trg],inner,seed+20000)
    anchor=(.55*bva+.45*rel_global).astype(np.float32)
    recent={}
    # trg is sorted for rolling folds; for random sanity use ID order explicitly.
    local_order=np.argsort(TRAIN_ID_NUM[trg])
    for w in windows:
        k=min(int(w),len(trg))
        if k>=len(trg):
            # A recent window equal to the entire history is the global regime, not a random-seed perturbation.
            rr=rel_global.copy(); rt=top_global.copy()
        else:
            recent_local=np.sort(local_order[-k:])
            rr=fit_recent_head_local(Xt,Xv,Y_BIN[trg],recent_local,rel_iters,seed+30000+int(w))
            rt=fit_recent_head_local(Xt,Xv,Y_TOP5[trg],recent_local,top_iters_local,seed+40000+int(w))
        recent[int(w)]=(rr,rt)
    return anchor,rel_global,top_global,recent

# ---- cell 11 (code) ----
# Pre-registered temporal policy grid.
POLICIES=[]
for w in RECENT_WINDOWS:
    for gamma in TOP5_GAMMA_GRID:
        for alpha in ALPHA_GRID:
            POLICIES.append({"window":int(w),"gamma":float(gamma),"alpha":float(alpha)})

def policy_key(p): return f"w{p['window']}_g{p['gamma']:.2f}_a{p['alpha']:.2f}"

def recency_weights(k):
    if k==4: return ROLL_WEIGHTS.copy()
    # Fixed geometric preference for the most recent available historical block.
    w=np.power(1.7,np.arange(k,dtype=np.float64)); return w/w.sum()

def summarize_history(rows,keys):
    d=np.asarray([rows[k]["delta"] for k in keys],float)
    c=np.asarray([rows[k]["top5_change"] for k in keys],float)
    wt=recency_weights(len(keys)); weighted=float(np.dot(wt,d)); std=float(np.std(d))
    wins=int((d>0).sum()); worst=float(d.min()); latest=float(d[-1])
    # Stability penalty is intentionally low-capacity and fixed before evaluation.
    objective=weighted-0.25*std-0.004*float(c.mean())
    return {"weighted_delta":weighted,"std":std,"wins":wins,"worst":worst,"latest":latest,
            "mean_change":float(c.mean()),"objective":float(objective)}

def choose_policy(results_by_policy,history_blocks):
    ranked=[]
    for p in POLICIES:
        k=policy_key(p); s=summarize_history(results_by_policy[k],history_blocks)
        n=len(history_blocks)
        need_wins=max(1,int(np.ceil(0.60*n)))
        eligible=(s["wins"]>=need_wins and s["latest"]>0 and s["worst"]>=-0.00050 and s["mean_change"]<=MAX_TOP5_CHANGE+1e-12)
        ranked.append((eligible,s["objective"],s["weighted_delta"],-s["std"],k,p,s))
    ranked.sort(reverse=True,key=lambda x:(x[0],x[1],x[2],x[3]))
    return ranked[0],ranked

print("policy count",len(POLICIES),"windows",RECENT_WINDOWS,"alphas",ALPHA_GRID,"gammas",TOP5_GAMMA_GRID)

# ---- cell 12 (markdown) ----
# ## Four future-block evaluations
#
# For each block, all supervised fitting uses only earlier IDs. Every policy is evaluated on the same untouched future block, but final deployment selection is additionally audited by a sequential selector that only sees older blocks when choosing the policy for the next block.

# ---- cell 13 (code) ----
if V11_SMOKE:
    ROLL_SPECS=[(0,1200,1200,1600),(0,1600,1600,2000),(0,2000,2000,2400),(0,2400,2400,2800)]
else:
    ROLL_SPECS=[(0,2000,2000,2500),(0,2500,2500,3000),(0,3000,3000,3500),(0,3500,3500,4000)]

roll_results={policy_key(p):{} for p in POLICIES}
roll_anchors=[]; roll_anchor_scores=[]
for b,(tr0,tr1,va0,va1) in enumerate(ROLL_SPECS):
    trg=np.arange(tr0,tr1,dtype=int); vag=np.arange(va0,va1,dtype=int)
    print(f"\n=== ROLL {b+1}: train {TRAIN_ID_NUM[trg[0]]}-{TRAIN_ID_NUM[trg[-1]]} -> val {TRAIN_ID_NUM[vag[0]]}-{TRAIN_ID_NUM[vag[-1]]} ===")
    anchor,gr,gt,recent=build_split_predictions(trg,vag,RECENT_WINDOWS,SEED+1000*b)
    roll_anchors.append(anchor); a_score=ndcg5(y[vag],anchor); roll_anchor_scores.append(a_score)
    print("anchor",round(a_score,6))
    for p in POLICIES:
        rr,rt=recent[p["window"]]
        shift=temporal_shift(gr,rr,gt,rt,p["gamma"])
        cand,eff_alpha,change=apply_temporal_budget(anchor,shift,p["alpha"])
        score=ndcg5(y[vag],cand); delta=score-a_score
        roll_results[policy_key(p)][b]={
            "score":float(score),"delta":float(delta),"top5_change":float(change),
            "top1_change":float(top1_change_rate(cand,anchor)),"effective_alpha":float(eff_alpha)
        }
    best=max(POLICIES,key=lambda p:roll_results[policy_key(p)][b]["delta"])
    br=roll_results[policy_key(best)][b]
    print("block oracle diagnostic",policy_key(best),"delta",round(br["delta"],6),"change",round(br["top5_change"],4))
    gc.collect()

# ---- cell 14 (code) ----
# Sequential one-step-ahead policy assessment: policy for block t sees only blocks < t.
seq=[]
for target_block in range(1,4):
    history=list(range(target_block))
    chosen,ranked=choose_policy(roll_results,history)
    _,_,_,_,key,p,s=chosen
    future=roll_results[key][target_block]
    seq.append({"target_block":target_block+1,"policy":key,"history":s,"future":future})
    print("sequential -> block",target_block+1,"selected",key,
          "history weighted",round(s["weighted_delta"],6),"future delta",round(future["delta"],6))

seq_deltas=np.asarray([x["future"]["delta"] for x in seq],float)
seq_weights=np.asarray([0.20,0.30,0.50],float); seq_weights/=seq_weights.sum()
seq_weighted=float(np.dot(seq_weights,seq_deltas)); seq_wins=int((seq_deltas>0).sum())

# Final test policy sees all four historical future blocks.
chosen,ranking=choose_policy(roll_results,[0,1,2,3])
elig,obj,wdelta,nstd,final_key,final_policy,final_hist=chosen
final_deltas=np.asarray([roll_results[final_key][b]["delta"] for b in range(4)],float)
final_changes=np.asarray([roll_results[final_key][b]["top5_change"] for b in range(4)],float)
print("\nFINAL ROLL POLICY",final_key)
print("deltas",[round(x,6) for x in final_deltas],"wins",int((final_deltas>0).sum()))
print("weighted",round(final_hist["weighted_delta"],6),"std",round(final_hist["std"],6),"worst",round(final_hist["worst"],6))
print("mean/max top5 change",round(final_changes.mean(),4),round(final_changes.max(),4))
print("sequential deltas",[round(x,6) for x in seq_deltas],"weighted",round(seq_weighted,6),"wins",seq_wins)

# Persist the complete pre-registered grid for transparent post-run analysis.
rows=[]
for p in POLICIES:
    k=policy_key(p); s=summarize_history(roll_results[k],[0,1,2,3])
    row={"policy":k,**p,**s}
    for b in range(4):
        row[f"delta_b{b+1}"]=roll_results[k][b]["delta"]
        row[f"change_b{b+1}"]=roll_results[k][b]["top5_change"]
        row[f"eff_alpha_b{b+1}"]=roll_results[k][b]["effective_alpha"]
    rows.append(row)
rolling_df=pd.DataFrame(rows).sort_values(["objective","weighted_delta"],ascending=False).reset_index(drop=True)
display(rolling_df.head(12))

# ---- cell 15 (markdown) ----
# ## Shuffled-CV sanity check
#
# This is **not** used to tune V11. The already-selected rolling policy is evaluated under the old shuffled regime only to reject a catastrophic family mismatch. The recent subset inside each shuffled training fold is always chosen by largest user IDs, never by validation labels.

# ---- cell 16 (code) ----
random_deltas=[]; random_changes=[]
for f,(trg,vag) in enumerate(ref_folds):
    print("random sanity fold",f+1)
    anchor,gr,gt,recent=build_split_predictions(np.asarray(trg),np.asarray(vag),[final_policy["window"]],SEED+6000+f)
    rr,rt=recent[final_policy["window"]]
    shift=temporal_shift(gr,rr,gt,rt,final_policy["gamma"])
    cand,eff,chg=apply_temporal_budget(anchor,shift,final_policy["alpha"])
    d=ndcg5(y[vag],cand)-ndcg5(y[vag],anchor)
    random_deltas.append(float(d)); random_changes.append(float(chg))
    print(" delta",round(d,6),"change",round(chg,4),"effective alpha",round(eff,4))
    gc.collect()
random_delta=float(np.average(random_deltas,weights=[len(v) for _,v in ref_folds]))
print("random sanity overall",round(random_delta,6),"folds",[round(x,6) for x in random_deltas])

# ---- cell 17 (markdown) ----
# ## Full-data deployment
#
# The public anchor remains the exact full-data cal045 path. Only the selected recent tail expert is newly fitted. Its per-module tree counts are inherited from the corresponding full-data global head, so the recent expert changes the training regime, not model capacity.

# ---- cell 18 (code) ----
def fit_recent_full(Xtr,Xte,targets,iters_dict,recent_n,seed):
    pred=np.zeros((N_TEST,N_MODULES),np.float32)
    order=np.argsort(TRAIN_ID_NUM); k=min(int(recent_n),N_TRAIN); ridx=np.sort(order[-k:])
    final_it=median_iters(iters_dict)
    for m,mid in enumerate(MODULES):
        yy=targets[:,m].astype(np.int32)
        uniq=np.unique(yy[ridx])
        if len(uniq)<2:
            pred[:,m]=float(uniq[0]) if len(uniq) else float(yy.mean()); continue
        clf=make_cal(seed+31*m,final_it[mid]); clf.fit(Xtr[m][ridx],yy[ridx])
        pred[:,m]=clf.predict_proba(Xte[m])[:,1].astype(np.float32)
    return pred

recent_rel_test=fit_recent_full(META_TR,META_TE,Y_BIN,anchor_iters,final_policy["window"],SEED+80000)
recent_top_test=fit_recent_full(META_TR,META_TE,Y_TOP5,top5_iters,final_policy["window"],SEED+90000)
shift_test=temporal_shift(anchor_meta_test,recent_rel_test,global_top5_test,recent_top_test,final_policy["gamma"])
candidate_test,effective_alpha_test,test_top5_change=apply_temporal_budget(cal045_test,shift_test,final_policy["alpha"])
test_top1_change=top1_change_rate(candidate_test,cal045_test)

# Invariants by design: original top-2 anchor modules never move.
def top2_preserved(a,b):
    oa=np.argsort(-a,axis=1)[:,:2]; ob=np.argsort(-b,axis=1)[:,:2]
    return float(np.mean(np.all(oa==ob,axis=1)))

test_top2_preserved=top2_preserved(cal045_test,candidate_test)
expected_lb=PUBLIC_ANCHOR_LB+final_hist["weighted_delta"]

roll_wins=int((final_deltas>0).sum()); last2_ok=bool(np.min(final_deltas[-2:])>=MAX_LAST2_NEG)
promotion_gate=bool(
    final_hist["weighted_delta"]>=MIN_ROLL_WEIGHTED_DELTA and
    expected_lb>=LEADER_LB and
    roll_wins>=MIN_ROLL_WINS and
    last2_ok and
    seq_wins>=MIN_SEQ_WINS and
    seq_weighted>=MIN_SEQ_WEIGHTED_DELTA and
    random_delta>=-MAX_RANDOM_SANITY_DROP and
    test_top5_change<=MAX_TEST_TOP5_CHANGE+1e-12 and
    test_top1_change==0.0 and
    test_top2_preserved==1.0
)
primary_test=candidate_test if promotion_gate else cal045_test

print("\n================ V11 DECISION ================")
print("selected policy        :",final_key)
print("rolling deltas         :",[round(float(x),6) for x in final_deltas])
print("rolling weighted delta :",round(final_hist["weighted_delta"],6),"wins",roll_wins,"last2 ok",last2_ok)
print("1:1 expected public LB :",round(expected_lb,6),"leader",LEADER_LB)
print("sequential future delta:",[round(float(x),6) for x in seq_deltas],"weighted",round(seq_weighted,6),"wins",seq_wins)
print("random sanity delta    :",round(random_delta,6))
print("test effective alpha   :",round(effective_alpha_test,4))
print("test top5/top1 change  :",round(test_top5_change,4),round(test_top1_change,4))
print("test top2 preserved    :",test_top2_preserved)
print("promotion gate         :",promotion_gate)
print("ACTION                 :", "REVIEW submission_v11_primary.csv AS FIRST V11 CANDIDATE" if promotion_gate else "DO NOT SUBMIT V11 PRIMARY; it is exact cal045 fallback. Review candidate diagnostics first.")
print("===============================================")

# ---- cell 19 (code) ----
def save_submission(scores,name):
    out=sample.copy(); out.loc[:,MODULES]=scores
    root=Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('/mnt/data')
    path=root/name; out.to_csv(path,index=False)
    chk=pd.read_csv(path)
    assert chk.shape==sample.shape and chk.user_id.is_unique
    assert chk[MODULES].notna().all().all() and np.isfinite(chk[MODULES].to_numpy()).all()
    print("saved",path)
    return path

root=Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('/mnt/data')
if not V11_SMOKE:
    p_ref=save_submission(cal045_test,'submission_v11_cal045_reference.csv')
    p_cand=save_submission(candidate_test,'submission_v11_candidate.csv')
    p_primary=save_submission(primary_test,'submission_v11_primary.csv')
    rolling_df.to_csv(root/'v11_rolling_candidates.csv',index=False)
    np.savez_compressed(
        root/'v11_artifacts.npz',
        cal045_oof=cal045_oof,cal045_test=cal045_test,candidate_test=candidate_test,
        anchor_meta_test=anchor_meta_test,global_top5_test=global_top5_test,
        recent_rel_test=recent_rel_test,recent_top_test=recent_top_test,shift_test=shift_test,
        final_deltas=final_deltas,seq_deltas=seq_deltas,random_deltas=np.asarray(random_deltas,np.float32)
    )
    payload={
        "public_anchor_lb":PUBLIC_ANCHOR_LB,
        "leader_lb_at_design_time":LEADER_LB,
        "required_public_delta":REQUIRED_PUBLIC_DELTA,
        "selected_policy":final_policy,
        "selected_policy_key":final_key,
        "rolling_anchor_scores":[float(x) for x in roll_anchor_scores],
        "rolling_deltas":[float(x) for x in final_deltas],
        "rolling_weighted_delta":float(final_hist["weighted_delta"]),
        "rolling_std":float(final_hist["std"]),
        "rolling_wins":roll_wins,
        "rolling_worst":float(final_hist["worst"]),
        "rolling_mean_top5_change":float(final_changes.mean()),
        "sequential":[{"target_block":int(x["target_block"]),"policy":x["policy"],"future_delta":float(x["future"]["delta"])} for x in seq],
        "sequential_weighted_delta":float(seq_weighted),
        "sequential_wins":seq_wins,
        "random_sanity_deltas":[float(x) for x in random_deltas],
        "random_sanity_delta":float(random_delta),
        "expected_public_lb_if_1to1":float(expected_lb),
        "effective_alpha_test":float(effective_alpha_test),
        "test_top5_change":float(test_top5_change),
        "test_top1_change":float(test_top1_change),
        "test_top2_preserved":float(test_top2_preserved),
        "promotion_gate":promotion_gate,
        "primary":p_primary.name,
        "candidate_file":p_cand.name,
        "reference_file":p_ref.name
    }
    with open(root/'v11_summary.json','w') as f: json.dump(payload,f,indent=2)
    print("saved",root/'v11_summary.json')

# ---- cell 20 (markdown) ----
# ## Hard invariants
#
# These checks enforce the V11 contract. The immutable public anchor remains finite; temporal corrections cannot alter the anchor's top-2 ordering; candidate Top-5 membership changes are capped at 5%; and a failed promotion gate makes `primary` exactly equal to cal045.

# ---- cell 21 (code) ----
assert len(MODULES)==17 and y.shape==(4000,17) and N_TEST==1000
for arr in [base_oof,base_test,anchor_meta_oof,anchor_meta_test,cal045_oof,cal045_test,
            global_top5_oof,global_top5_test,recent_rel_test,recent_top_test,shift_test,candidate_test,primary_test]:
    assert np.isfinite(arr).all()
assert test_top5_change<=MAX_TEST_TOP5_CHANGE+1e-12
assert test_top1_change==0.0
assert test_top2_preserved==1.0
if not V11_SMOKE:
    # Broad reproduction tolerance guards accidental drift without pretending OOF is public LB.
    assert abs(ndcg5(y,cal045_oof)-0.662646)<0.003
    if promotion_gate:
        assert not np.allclose(candidate_test,cal045_test)
    else:
        assert np.allclose(primary_test,cal045_test)
print("All V11 hard invariants passed.")
