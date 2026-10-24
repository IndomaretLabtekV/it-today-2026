# Converted from notebooks/v12_1.ipynb: markdown cells become comments, %/! magics neutralized.

# ---- cell 0 (markdown) ----
# # MineToday V12.1 — Micro Trust-Region Calibration Refinement
#
# **Goal:** improve the validated V12 Leader6 public score **0.65792** enough to clear the current leader **0.65805**, while changing as little as possible from the only post-cal045 model family that has actually transferred positively to the public leaderboard.
#
# ## Public evidence entering V12.1
#
# - exact V3.1/V7 `cal045`: **0.65639** public;
# - V12 `Leader6`: **0.65792** public;
# - current leader: **0.65805**;
# - remaining public gap from V12: only **0.00013**.
#
# V12.1 deliberately does **not** add a new learner, residual model, ranker, semantic encoder, temporal expert, pairwise swap, or listwise objective. The exact V12 sparse + seed-42 LightGBM calibration pipeline is preserved. The only experimental action is a frozen, tiny trust-region movement around the already-public-validated Leader6 module weights.
#
# ## Frozen Micro4 profile
#
# Leader6 is reproduced exactly as the public reference:
#
# - `M006=.55, M007=.60, M010=.40, M011=.55, M012=.55, M017=.50`, all others `.45`.
#
# V12.1 **Micro4** changes only four of those weights:
#
# - `M006: .55 → .56`
# - `M007: .60 → .59`
# - `M011: .55 → .54`
# - `M017: .50 → .53`
#
# while keeping `M010=.40`, `M012=.55`, and all unspecified modules at `.45`.
#
# The resulting frozen profile is therefore:
#
# `M006=.56, M007=.59, M010=.40, M011=.54, M012=.55, M017=.53`, others `.45`.
#
# These values are constants fixed **before this notebook run**. There is no grid search over module weights in this notebook.
#
# ## Why this is a trust-region refinement
#
# The target is not to discover another high-variance OOF model. V12 already demonstrated positive public transfer. V12.1 only asks whether a tiny movement around that public-validated point improves the same exact model family. Validation therefore compares Micro4 against **both** cal045 and Leader6, including:
#
# 1. exact-path 5-fold OOF NDCG@5;
# 2. paired bootstrap intervals;
# 3. fold deltas versus both references;
# 4. four contiguous 1,000-user quarter deltas;
# 5. tail U_3001–U_4000 behavior;
# 6. rank-position contribution decomposition;
# 7. revert-one-of-four sensitivity;
# 8. test Top-5 / Top-1 geometry versus Leader6 and cal045;
# 9. an observed same-family transfer diagnostic derived from the already-known V12 public result.
#
# The public transfer diagnostic is **never used to fit scores**. It is only a final deployment diagnostic.
#
# ## Decision policy
#
# `submission_v12_1_primary.csv` becomes Micro4 only if all stability and trust-region gates pass. Otherwise it falls back **exactly to V12 Leader6**, the current team public champion. The raw Micro4 candidate is always saved separately for inspection.
#
# ## Competition compliance
#
# Only organizer-provided dataset files are used to fit predictions. No external dataset, direct LLM test prediction, paid model, or external pretrained model is required.
#
# ## Kaggle runtime
#
# Compatible with Kaggle **T4 ×2**. The exact sparse + LightGBM pipeline is CPU-oriented; GPUs may remain idle without affecting correctness.
#
# ## Outputs
#
# - `submission_v12_1_micro4.csv` — actual frozen V12.1 candidate;
# - `submission_v12_1_primary.csv` — Micro4 only if the gate passes, otherwise exact Leader6 fallback;
# - `submission_v12_1_leader6_reference.csv` — exact V12 public-champion reference;
# - `submission_v12_1_cal045_reference.csv` — original cal045 reference;
# - `v12_1_summary.json` — full diagnostics and decision;
# - `v12_1_diagnostics.csv` — compact comparison table;
# - `v12_1_oof_artifacts.npz` — OOF/test predictions and fold IDs for audit.
#
# **Run all cells, then inspect the final `V12.1 DECISION` block before using the last submission slot.**

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
V121_SMOKE=bool(int(os.getenv("V121_SMOKE","0")))
V91_SMOKE=V121_SMOKE  # compatibility with exact sparse-backbone cell
N_FOLDS=2 if V121_SMOKE else 5
RIDGE_ALPHA=3.0
REFERENCE_W=0.45
LGB_ESTIMATORS=40 if V121_SMOKE else 1200
LGB_EARLY_STOP=5 if V121_SMOKE else 80
BOOTSTRAPS=300 if V121_SMOKE else 5000

PUBLIC_CAL045_LB=0.65639
PUBLIC_V12_LB=0.65792
LEADER_LB=0.65805
REQUIRED_PUBLIC_INCREMENT=LEADER_LB-PUBLIC_V12_LB

# Frozen deployment gates. They are intentionally framed around the already-public-validated
# Leader6 point, not around a new search over the 17-dimensional weight space.
MIN_TOTAL_DELTA=0.0005 if V121_SMOKE else 0.00160
MIN_INCREMENTAL_DELTA=-0.01 if V121_SMOKE else 0.00012
MIN_TOTAL_FOLD_WINS=1 if V121_SMOKE else 5
MIN_INCREMENTAL_FOLD_WINS=1 if V121_SMOKE else 4
MIN_TOTAL_QUARTER_WINS=1 if V121_SMOKE else 4
MIN_INCREMENTAL_QUARTER_WINS=1 if V121_SMOKE else 4
MIN_TOTAL_TAIL_DELTA=-0.01 if V121_SMOKE else 0.00050
MIN_INCREMENTAL_TAIL_DELTA=-0.01 if V121_SMOKE else 0.00020
MIN_REVERT_TOTAL_DELTA=-0.01 if V121_SMOKE else 0.00120
MAX_TOP5_CHANGE_VS_LEADER=0.10 if V121_SMOKE else 0.020
MAX_TOP1_CHANGE_VS_LEADER=0.10 if V121_SMOKE else 0.006
MAX_TOP5_CHANGE_VS_CAL045=0.15 if V121_SMOKE else 0.070
MAX_TOP1_CHANGE_VS_CAL045=0.10 if V121_SMOKE else 0.022

random.seed(SEED); np.random.seed(SEED)
print("versions",np.__version__,pd.__version__,lgb.__version__,"V121_SMOKE",V121_SMOKE)
print("cal045 public",PUBLIC_CAL045_LB,"V12 public",PUBLIC_V12_LB,"leader",LEADER_LB,
      "remaining public gap",round(REQUIRED_PUBLIC_INCREMENT,6))

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
# ## Frozen Leader6 reference and V12.1 Micro4
#
# No weight is searched inside this notebook. `leader6` reproduces the already-submitted V12 point. `micro4` is the single pre-registered trust-region refinement under evaluation.
#
# Because the public gap is only `0.00013`, V12.1 prioritizes **small score geometry changes** over higher-capacity modeling. A successful candidate should improve the same-family OOF signal while changing only a tiny fraction of Leader6 test rankings.

# ---- cell 10 (code) ----
PROFILE_OVERRIDES={
    "leader6": {
        "M_006":0.55,"M_007":0.60,"M_010":0.40,"M_011":0.55,"M_012":0.55,"M_017":0.50,
    },
    "micro4": {
        "M_006":0.56,"M_007":0.59,"M_010":0.40,"M_011":0.54,"M_012":0.55,"M_017":0.53,
    },
}
MICRO_CHANGED_FROM_LEADER=("M_006","M_007","M_011","M_017")


def profile_vector(name):
    w=np.full(N_MODULES,REFERENCE_W,np.float32)
    for mid,val in PROFILE_OVERRIDES[name].items():
        if mid not in MODULES: raise KeyError(mid)
        w[MODULES.index(mid)]=float(val)
    return w


def apply_profile(base,meta,name):
    w=profile_vector(name)
    return ((1.0-w)*base+w*meta).astype(np.float32)

leader6_oof=apply_profile(base_oof,meta_oof,"leader6")
leader6_test=apply_profile(base_test,meta_test,"leader6")
micro4_oof=apply_profile(base_oof,meta_oof,"micro4")
micro4_test=apply_profile(base_test,meta_test,"micro4")

print("Leader6 frozen weights",PROFILE_OVERRIDES["leader6"])
print("Micro4 frozen weights ",PROFILE_OVERRIDES["micro4"])
print("Changed from Leader6  ",MICRO_CHANGED_FROM_LEADER)

# ---- cell 11 (code) ----
# Validation helpers.
def top5_set_change_rate(a,b):
    ta,tb=top5_indices(a),top5_indices(b)
    return float(np.mean([set(x)!=set(z) for x,z in zip(ta,tb)]))

def top5_order_change_rate(a,b):
    return float(np.mean(np.any(top5_indices(a)!=top5_indices(b),axis=1)))

def top1_change_rate(a,b):
    return float(np.mean(np.argmax(a,axis=1)!=np.argmax(b,axis=1)))

def paired_boot(delta):
    delta=np.asarray(delta,float); n=len(delta); rng=np.random.default_rng(20260823)
    z=np.empty(BOOTSTRAPS,np.float64)
    for i in range(BOOTSTRAPS):
        ix=rng.integers(0,n,n); z[i]=delta[ix].mean()
    return tuple(float(v) for v in np.quantile(z,[.025,.975]))

def rank_contributions(anchor,cand):
    oa=np.argsort(-anchor,axis=1)[:,:5]
    oc=np.argsort(-cand,axis=1)[:,:5]
    ideal=np.sort(y,axis=1)[:,-5:][:,::-1]
    idcg=(ideal*DISCOUNT).sum(1)
    out=[]
    for r in range(5):
        z=(y[np.arange(N_TRAIN),oc[:,r]]-y[np.arange(N_TRAIN),oa[:,r]])*DISCOUNT[r]/np.maximum(idcg,1e-12)
        out.append(float(z.mean()))
    return out

# Reconstruct reference fold IDs once, from the same exact split as the backbone.
ref_fold_id=np.full(N_TRAIN,-1,int)
for f,(_,va) in enumerate(ref_folds): ref_fold_id[va]=f
assert (ref_fold_id>=0).all()

# Stress masks use only organizer-provided train data.
CHAT_USERS=set(chat.user_id.astype(str))
NOCHAT_MASK=np.asarray([str(u) not in CHAT_USERS for u in train.user_id])
K6_MASK=(Y_BIN.sum(1)==6)
INVALID_MASK=(A_INVALID[:N_TRAIN].sum(1)>0)
TAIL_MASK=np.zeros(N_TRAIN,bool); TAIL_MASK[-1000:]=True

# ---- cell 12 (code) ----
# V12.1 diagnostics: Micro4 is evaluated against both cal045 and public-validated Leader6.
cal_pu=per_user_ndcg(y,cal045_oof)
leader_pu=per_user_ndcg(y,leader6_oof)
micro_pu=per_user_ndcg(y,micro4_oof)

total_u=micro_pu-cal_pu
incremental_u=micro_pu-leader_pu
leader_u=leader_pu-cal_pu

if V121_SMOKE:
    quarter_bounds=np.linspace(0,N_TRAIN,3,dtype=int)
else:
    quarter_bounds=np.asarray([0,1000,2000,3000,4000],int)


def fold_means(delta_u):
    return np.asarray([delta_u[ref_fold_id==f].mean() for f in range(N_FOLDS)],float)


def quarter_means(delta_u):
    return np.asarray([delta_u[quarter_bounds[i]:quarter_bounds[i+1]].mean()
                       for i in range(len(quarter_bounds)-1)],float)

leader_delta=float(ndcg5(y,leader6_oof)-ndcg5(y,cal045_oof))
total_delta=float(ndcg5(y,micro4_oof)-ndcg5(y,cal045_oof))
incremental_delta=float(ndcg5(y,micro4_oof)-ndcg5(y,leader6_oof))

leader_fold=fold_means(leader_u)
total_fold=fold_means(total_u)
incremental_fold=fold_means(incremental_u)
leader_quarter=quarter_means(leader_u)
total_quarter=quarter_means(total_u)
incremental_quarter=quarter_means(incremental_u)

leader_ci=paired_boot(leader_u)
total_ci=paired_boot(total_u)
incremental_ci=paired_boot(incremental_u)

total_rank=rank_contributions(cal045_oof,micro4_oof)
incremental_rank=rank_contributions(leader6_oof,micro4_oof)

top5_vs_leader=top5_set_change_rate(micro4_test,leader6_test)
order_vs_leader=top5_order_change_rate(micro4_test,leader6_test)
top1_vs_leader=top1_change_rate(micro4_test,leader6_test)
top5_vs_cal=top5_set_change_rate(micro4_test,cal045_test)
order_vs_cal=top5_order_change_rate(micro4_test,cal045_test)
top1_vs_cal=top1_change_rate(micro4_test,cal045_test)

leader_tail=float(leader_u[TAIL_MASK].mean())
total_tail=float(total_u[TAIL_MASK].mean())
incremental_tail=float(incremental_u[TAIL_MASK].mean())

# Revert-one sensitivity: each of the four micro changes is independently returned to
# its Leader6 value. This checks that the candidate is not entirely dependent on one tiny tweak.
revert_one={}
for mid in MICRO_CHANGED_FROM_LEADER:
    w=profile_vector("micro4").copy()
    j=MODULES.index(mid)
    w[j]=profile_vector("leader6")[j]
    pred=((1-w)*base_oof+w*meta_oof).astype(np.float32)
    revert_one[mid]={
        "total_delta_vs_cal045":float(ndcg5(y,pred)-ndcg5(y,cal045_oof)),
        "incremental_delta_vs_leader6":float(ndcg5(y,pred)-ndcg5(y,leader6_oof)),
    }
min_revert_total=float(min(v["total_delta_vs_cal045"] for v in revert_one.values()))

# Same-family public transfer is a diagnostic only. The numerator is actual public gain
# of V12 Leader6 over cal045; the denominator is Leader6 OOF gain in this exact rerun.
if leader_delta>1e-6:
    observed_v12_transfer_ratio=float((PUBLIC_V12_LB-PUBLIC_CAL045_LB)/leader_delta)
    projected_public_observed_transfer=float(PUBLIC_V12_LB+observed_v12_transfer_ratio*incremental_delta)
else:
    observed_v12_transfer_ratio=float("nan")
    projected_public_observed_transfer=float("-inf")
projected_public_1to1=float(PUBLIC_V12_LB+incremental_delta)

incremental_fold_wins=int((incremental_fold>0).sum())
incremental_quarter_wins=int((incremental_quarter>0).sum())
total_fold_wins=int((total_fold>0).sum())
total_quarter_wins=int((total_quarter>0).sum())

micro_gate=bool(
    total_delta>=MIN_TOTAL_DELTA and
    incremental_delta>=MIN_INCREMENTAL_DELTA and
    total_fold_wins>=MIN_TOTAL_FOLD_WINS and
    incremental_fold_wins>=MIN_INCREMENTAL_FOLD_WINS and
    total_quarter_wins>=MIN_TOTAL_QUARTER_WINS and
    incremental_quarter_wins>=MIN_INCREMENTAL_QUARTER_WINS and
    total_ci[0]>0 and
    total_tail>=MIN_TOTAL_TAIL_DELTA and
    incremental_tail>=MIN_INCREMENTAL_TAIL_DELTA and
    min_revert_total>=MIN_REVERT_TOTAL_DELTA and
    top5_vs_leader<=MAX_TOP5_CHANGE_VS_LEADER+1e-12 and
    top1_vs_leader<=MAX_TOP1_CHANGE_VS_LEADER+1e-12 and
    top5_vs_cal<=MAX_TOP5_CHANGE_VS_CAL045+1e-12 and
    top1_vs_cal<=MAX_TOP1_CHANGE_VS_CAL045+1e-12 and
    projected_public_observed_transfer>=LEADER_LB
)

diagnostics={
    "cal045_oof":float(ndcg5(y,cal045_oof)),
    "leader6_oof":float(ndcg5(y,leader6_oof)),
    "micro4_oof":float(ndcg5(y,micro4_oof)),
    "leader6_delta_vs_cal045":leader_delta,
    "total_delta":total_delta,
    "incremental_delta":incremental_delta,
    "leader6_bootstrap_ci_vs_cal045":leader_ci,
    "total_bootstrap_ci_vs_cal045":total_ci,
    "incremental_bootstrap_ci_vs_leader6":incremental_ci,
    "leader6_fold_deltas":leader_fold.tolist(),
    "total_fold_deltas":total_fold.tolist(),
    "incremental_fold_deltas":incremental_fold.tolist(),
    "total_fold_wins":total_fold_wins,
    "incremental_fold_wins":incremental_fold_wins,
    "leader6_quarter_deltas":leader_quarter.tolist(),
    "total_quarter_deltas":total_quarter.tolist(),
    "incremental_quarter_deltas":incremental_quarter.tolist(),
    "total_quarter_wins":total_quarter_wins,
    "incremental_quarter_wins":incremental_quarter_wins,
    "leader6_tail_delta":leader_tail,
    "total_tail_delta":total_tail,
    "incremental_tail_delta":incremental_tail,
    "rank_contributions_vs_cal045":total_rank,
    "rank_contributions_vs_leader6":incremental_rank,
    "revert_one":revert_one,
    "min_revert_total_delta":min_revert_total,
    "test_top5_change_vs_leader6":top5_vs_leader,
    "test_top5_order_change_vs_leader6":order_vs_leader,
    "test_top1_change_vs_leader6":top1_vs_leader,
    "test_top5_change_vs_cal045":top5_vs_cal,
    "test_top5_order_change_vs_cal045":order_vs_cal,
    "test_top1_change_vs_cal045":top1_vs_cal,
    "observed_v12_transfer_ratio":observed_v12_transfer_ratio,
    "projected_public_1to1_incremental":projected_public_1to1,
    "projected_public_observed_transfer":projected_public_observed_transfer,
    "gate":micro_gate,
}

rows=[
    {"reference":"leader6","oof":ndcg5(y,leader6_oof),"delta_vs_cal045":leader_delta,
     "delta_vs_leader6":0.0,"fold_wins_vs_cal045":int((leader_fold>0).sum()),
     "quarter_wins_vs_cal045":int((leader_quarter>0).sum()),"tail_delta_vs_cal045":leader_tail,
     "top5_change_vs_leader6":0.0,"top1_change_vs_leader6":0.0},
    {"reference":"micro4","oof":ndcg5(y,micro4_oof),"delta_vs_cal045":total_delta,
     "delta_vs_leader6":incremental_delta,"fold_wins_vs_cal045":total_fold_wins,
     "quarter_wins_vs_cal045":total_quarter_wins,"tail_delta_vs_cal045":total_tail,
     "top5_change_vs_leader6":top5_vs_leader,"top1_change_vs_leader6":top1_vs_leader},
]
diagnostic_df=pd.DataFrame(rows)
display(diagnostic_df.round(6))
print("incremental folds   ",np.round(incremental_fold,6).tolist())
print("incremental quarters",np.round(incremental_quarter,6).tolist())
print("revert-one",revert_one)

# ---- cell 13 (markdown) ----
# ## Decision policy
#
# The public leader is only `0.00013` above V12 Leader6, but the final submission slot is scarce. V12.1 therefore requires two things simultaneously:
#
# 1. **total robustness:** Micro4 must remain a strong improvement over cal045 across folds, chronological quarters, tail users, and revert-one checks;
# 2. **incremental robustness:** the tiny refinement must improve Leader6 in enough independent folds and all chronological quarters while changing very few Leader6 test rankings.
#
# The observed V12 public/OOF transfer ratio is reported because it is now real same-family evidence. It is used only as a final deployment gate, never to train or alter predictions.
#
# If any gate fails, `submission_v12_1_primary.csv` is exactly the already-public-validated Leader6 scores.

# ---- cell 14 (code) ----
if diagnostics["gate"]:
    primary_test=micro4_test.copy()
    primary_label="micro4"
else:
    primary_test=leader6_test.copy()
    primary_label="leader6_fallback"

print("\n================ V12.1 DECISION ================")
print("cal045 OOF              :",round(diagnostics["cal045_oof"],6),"public",PUBLIC_CAL045_LB)
print("Leader6 OOF             :",round(diagnostics["leader6_oof"],6),
      "delta",f"{diagnostics['leader6_delta_vs_cal045']:+.6f}","public",PUBLIC_V12_LB)
print("Micro4 OOF              :",round(diagnostics["micro4_oof"],6))
print("total delta vs cal045   :",f"{diagnostics['total_delta']:+.6f}",
      "CI",tuple(round(x,6) for x in diagnostics["total_bootstrap_ci_vs_cal045"]))
print("increment vs Leader6    :",f"{diagnostics['incremental_delta']:+.6f}",
      "CI",tuple(round(x,6) for x in diagnostics["incremental_bootstrap_ci_vs_leader6"]))
print("fold wins total/incr    :",diagnostics["total_fold_wins"],"/",diagnostics["incremental_fold_wins"])
print("quarter wins total/incr :",diagnostics["total_quarter_wins"],"/",diagnostics["incremental_quarter_wins"])
print("tail total/incremental  :",f"{diagnostics['total_tail_delta']:+.6f}",f"{diagnostics['incremental_tail_delta']:+.6f}")
print("test change vs Leader6  : top5",round(diagnostics["test_top5_change_vs_leader6"],4),
      "order",round(diagnostics["test_top5_order_change_vs_leader6"],4),
      "top1",round(diagnostics["test_top1_change_vs_leader6"],4))
print("observed V12 transfer   :",round(diagnostics["observed_v12_transfer_ratio"],4))
print("public projection 1:1   :",round(diagnostics["projected_public_1to1_incremental"],6))
print("public projection obs   :",round(diagnostics["projected_public_observed_transfer"],6),"leader",LEADER_LB)
print("GATE                     :",diagnostics["gate"])
print("PRIMARY                  :",primary_label)
print("ACTION                   :",("Micro4 cleared the offline gate. Inspect this block and result artifacts before deciding on the last Kaggle submission." if diagnostics["gate"] else "Micro4 did not clear the gate. Primary is exact Leader6; do not spend the last slot on the fallback."))
print("==================================================")

# ---- cell 15 (code) ----
def save_submission(scores,name):
    out=sample.copy(); out.loc[:,MODULES]=scores
    root=Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('/mnt/data')
    path=root/name; out.to_csv(path,index=False)
    chk=pd.read_csv(path)
    assert chk.shape==sample.shape
    assert chk.user_id.tolist()==sample.user_id.tolist()
    assert chk.user_id.is_unique
    assert chk[MODULES].notna().all().all()
    assert np.isfinite(chk[MODULES].to_numpy()).all()
    print("saved",path)
    return path

root=Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('/mnt/data')
if not V121_SMOKE:
    paths={}
    paths['cal045']=save_submission(cal045_test,'submission_v12_1_cal045_reference.csv')
    paths['leader6']=save_submission(leader6_test,'submission_v12_1_leader6_reference.csv')
    paths['micro4']=save_submission(micro4_test,'submission_v12_1_micro4.csv')
    paths['primary']=save_submission(primary_test,'submission_v12_1_primary.csv')
    diagnostic_df.to_csv(root/'v12_1_diagnostics.csv',index=False)
    np.savez_compressed(
        root/'v12_1_oof_artifacts.npz',
        y=y,base_oof=base_oof,meta_oof=meta_oof,cal045_oof=cal045_oof,
        leader6_oof=leader6_oof,micro4_oof=micro4_oof,
        base_test=base_test,meta_test=meta_test,cal045_test=cal045_test,
        leader6_test=leader6_test,micro4_test=micro4_test,
        ref_fold_id=ref_fold_id,
    )
    payload={
        "public_cal045_lb":PUBLIC_CAL045_LB,
        "public_v12_leader6_lb":PUBLIC_V12_LB,
        "leader_lb":LEADER_LB,
        "required_public_increment":REQUIRED_PUBLIC_INCREMENT,
        "profiles":PROFILE_OVERRIDES,
        "micro_changed_from_leader6":list(MICRO_CHANGED_FROM_LEADER),
        "diagnostics":diagnostics,
        "primary_label":primary_label,
        "primary_file":paths['primary'].name,
        "candidate_file":paths['micro4'].name,
        "leader6_reference_file":paths['leader6'].name,
        "cal045_reference_file":paths['cal045'].name,
    }
    with open(root/'v12_1_summary.json','w') as f: json.dump(payload,f,indent=2)
    print("saved",root/'v12_1_summary.json')
    print("saved",root/'v12_1_diagnostics.csv')
    print("saved",root/'v12_1_oof_artifacts.npz')

# ---- cell 16 (markdown) ----
# ## Hard invariants
#
# These checks prevent accidental drift away from the V12.1 contract:
#
# - exact 17-module schema and 4,000/1,000 split;
# - cal045 remains exactly `0.55*sparse + 0.45*relevance`;
# - Leader6 exactly reproduces the frozen V12 public profile;
# - Micro4 changes only `M006`, `M007`, `M011`, and `M017` relative to Leader6;
# - all unspecified module weights remain exactly `0.45`;
# - all predictions are finite;
# - the Micro4 primary may pass only inside the configured tiny test-change budget;
# - a failed V12.1 gate makes `primary` exactly equal to Leader6, never to a new experimental score.

# ---- cell 17 (code) ----
assert len(MODULES)==17 and y.shape==(4000,17) and N_TEST==1000
assert np.allclose(cal045_oof,.55*base_oof+.45*meta_oof,atol=1e-7)
assert np.allclose(cal045_test,.55*base_test+.45*meta_test,atol=1e-7)

expected_leader={"M_006":0.55,"M_007":0.60,"M_010":0.40,"M_011":0.55,"M_012":0.55,"M_017":0.50}
expected_micro={"M_006":0.56,"M_007":0.59,"M_010":0.40,"M_011":0.54,"M_012":0.55,"M_017":0.53}
assert PROFILE_OVERRIDES["leader6"]==expected_leader
assert PROFILE_OVERRIDES["micro4"]==expected_micro

wl=profile_vector("leader6"); wm=profile_vector("micro4")
for j,mid in enumerate(MODULES):
    if mid not in expected_leader:
        assert abs(float(wl[j])-REFERENCE_W)<1e-7
    if mid not in expected_micro:
        assert abs(float(wm[j])-REFERENCE_W)<1e-7
    if mid not in MICRO_CHANGED_FROM_LEADER:
        assert abs(float(wm[j])-float(wl[j]))<1e-7

assert np.isfinite(leader6_oof).all() and np.isfinite(leader6_test).all()
assert np.isfinite(micro4_oof).all() and np.isfinite(micro4_test).all()

if diagnostics['gate']:
    assert diagnostics['test_top5_change_vs_leader6']<=MAX_TOP5_CHANGE_VS_LEADER+1e-12
    assert diagnostics['test_top1_change_vs_leader6']<=MAX_TOP1_CHANGE_VS_LEADER+1e-12
    assert diagnostics['test_top5_change_vs_cal045']<=MAX_TOP5_CHANGE_VS_CAL045+1e-12
    assert diagnostics['test_top1_change_vs_cal045']<=MAX_TOP1_CHANGE_VS_CAL045+1e-12
    assert np.allclose(primary_test,micro4_test,atol=0,rtol=0)
else:
    assert np.allclose(primary_test,leader6_test,atol=0,rtol=0)

# Broad exact-path reproduction sanity guard. This catches accidental parser/model drift
# without pretending OOF is the public leaderboard.
if not V121_SMOKE:
    assert abs(ndcg5(y,cal045_oof)-0.662646)<0.003

print("V12.1 hard invariants: PASS")
