# Converted from notebooks/v12.ipynb: markdown cells become comments, %/! magics neutralized.

# ---- cell 0 (markdown) ----
# # MineToday V12 — Module-Aware Champion Calibration
#
# **Goal:** attack the current public leader **0.65805** without abandoning the only model family that has repeatedly transferred positively to the public leaderboard.
#
# Public evidence entering V12:
#
# - V3.1/V7 sparse-only: **0.65036**;
# - V3.1/V7 `cal035`: **0.65540**;
# - V3.1/V7 `cal045`: **0.65639**, current team champion;
# - V9.1 strict nested looked strong offline but collapsed to **0.65432** public;
# - V10/V10.1/V11 did not produce a credible replacement.
#
# V12 therefore makes **no new residual learner, no global ranker, no pairwise swap model, no temporal expert, and no semantic encoder**. It changes only the calibration proportion of a small pre-registered subset of modules while preserving the exact V3.1 sparse + seed-42 LightGBM inference path.
#
# ## Core hypothesis
#
# A single global `0.45` relevance-calibration weight is unnecessarily restrictive because module learnability differs. The six-module profiles below were chosen before this notebook run from two independent offline representations, fold stability, chronological-quarter stability, and deployment-change constraints.
#
# ### Pre-registered profiles
#
# - **Safe3**: `M007=.60, M012=.55, M015=.50`.
# - **Power4**: `M006=.55, M007=.60, M012=.55, M015=.50`.
# - **Balanced6**: `M006=.55, M007=.60, M010=.35, M011=.55, M012=.55, M017=.50`.
# - **Leader6**: `M006=.55, M007=.60, M010=.40, M011=.55, M012=.55, M017=.50`.
#
# All unspecified modules remain **exactly 0.45**. `Leader6` is the intended leaderboard attack; `Balanced6` is the robustness backup; `Safe3` is the conservative control.
#
# ## Validation philosophy
#
# V12 does not optimize 17 free weights. It evaluates only the four frozen profiles above and reports:
#
# 1. exact-path 5-fold OOF NDCG@5 versus cal045;
# 2. paired bootstrap confidence intervals;
# 3. all five fold deltas;
# 4. four contiguous 1,000-user chronological-quarter deltas;
# 5. tail, no-chat, K=6, and invalid-assessment stress slices;
# 6. rank-position contribution decomposition;
# 7. leave-one-changed-module sensitivity;
# 8. test Top-5 / Top-1 change budgets;
# 9. historical same-family transfer projections, shown as diagnostics only.
#
# The **primary file is never silently an experimental candidate**. `submission_v12_primary.csv` uses the first profile in the fixed priority `Leader6 → Balanced6 → Safe3` that clears the hard gate; otherwise it is exact cal045.
#
# ## Competition compliance
#
# Only organizer-provided data are used. No external dataset, no direct LLM test prediction, no paid model, and no external pretrained model are required.
#
# ## Kaggle runtime
#
# Compatible with Kaggle **T4 ×2**. The pipeline is intentionally CPU-based because the sparse + LightGBM calibration family is the only family with repeated positive public evidence. GPU availability does not affect correctness.
#
# ## Outputs
#
# - `submission_v12_leader6.csv` — highest-upside pre-registered attack;
# - `submission_v12_balanced6.csv` — robustness-focused six-module profile;
# - `submission_v12_power4.csv` — earlier four-module attack;
# - `submission_v12_safe3.csv` — conservative three-module profile;
# - `submission_v12_primary.csv` — first gated profile in fixed priority order, else cal045;
# - `submission_v12_cal045_reference.csv` — immutable public champion;
# - `v12_summary.json` — all validation and deployment diagnostics;
# - `v12_profile_diagnostics.csv` — compact comparison table;
# - `v12_oof_artifacts.npz` — predictions and fold IDs for audit.
#
# **Read the final `V12 DECISION` block before submitting anything.**

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
V12_SMOKE=bool(int(os.getenv("V12_SMOKE","0")))
V91_SMOKE=V12_SMOKE  # compatibility with exact sparse-backbone cell
N_FOLDS=2 if V12_SMOKE else 5
RIDGE_ALPHA=3.0
REFERENCE_W=0.45
LGB_ESTIMATORS=40 if V12_SMOKE else 1200
LGB_EARLY_STOP=5 if V12_SMOKE else 80
BOOTSTRAPS=300 if V12_SMOKE else 5000

PUBLIC_ANCHOR_LB=0.65639
LEADER_LB=0.65805
REQUIRED_PUBLIC_DELTA=LEADER_LB-PUBLIC_ANCHOR_LB
# Historical same-family public/OOF transfer diagnostics. Never used to fit predictions.
HIST_TOTAL_CAL_TRANSFER=1.37
HIST_CONSERVATIVE_TRANSFER=1.20

# Hard deployment/stability gates. The attack is allowed to be stronger than Safe3,
# but still much smaller than V9.1's failed 9.6% Top-5 reshuffle.
MIN_OOF_DELTA=0.00075 if V12_SMOKE else 0.00100
MIN_FOLD_WINS=1 if V12_SMOKE else 5
MIN_QUARTER_WINS=1 if V12_SMOKE else 4
MAX_TEST_TOP5_CHANGE=0.08 if V12_SMOKE else 0.065
MAX_TEST_TOP1_CHANGE=0.05 if V12_SMOKE else 0.020
MIN_TAIL_DELTA=-0.005 if V12_SMOKE else 0.00010
MIN_NOCHAT_DELTA=-0.01 if V12_SMOKE else -0.00050
MIN_DROPONE_DELTA=-0.01 if V12_SMOKE else 0.00050

random.seed(SEED); np.random.seed(SEED)
print("versions",np.__version__,pd.__version__,lgb.__version__,"V12_SMOKE",V12_SMOKE)
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
print("cal045 OOF ",round(ndcg5(y,cal045_oof),6),"public",PUBLIC_ANCHOR_LB)
print("median calibrator iterations:",{k:int(np.median(v)) for k,v in best_iters.items()})

# ---- cell 9 (markdown) ----
# ## Frozen module-aware profiles
#
# These are **not searched inside this notebook**. Their weights are constants fixed before execution. Every profile mixes the exact same `base` and exact same seed-42 relevance probabilities as cal045:
#
# `score[:,m] = (1-w_m) * sparse[:,m] + w_m * relevance[:,m]`.
#
# This keeps inference geometry inside the one model family that has already transferred positively to the leaderboard.

# ---- cell 10 (code) ----
PROFILE_OVERRIDES={
    "safe3": {
        "M_007":0.60,"M_012":0.55,"M_015":0.50,
    },
    "power4": {
        "M_006":0.55,"M_007":0.60,"M_012":0.55,"M_015":0.50,
    },
    "balanced6": {
        "M_006":0.55,"M_007":0.60,"M_010":0.35,"M_011":0.55,"M_012":0.55,"M_017":0.50,
    },
    "leader6": {
        "M_006":0.55,"M_007":0.60,"M_010":0.40,"M_011":0.55,"M_012":0.55,"M_017":0.50,
    },
}
PROFILE_PRIORITY=["leader6","balanced6","safe3"]

def profile_vector(name):
    w=np.full(N_MODULES,REFERENCE_W,np.float32)
    for mid,val in PROFILE_OVERRIDES[name].items():
        if mid not in MODULES: raise KeyError(mid)
        w[MODULES.index(mid)]=float(val)
    return w

def apply_profile(base,meta,name):
    w=profile_vector(name)
    return ((1.0-w)*base+w*meta).astype(np.float32)

profile_oof={name:apply_profile(base_oof,meta_oof,name) for name in PROFILE_OVERRIDES}
profile_test={name:apply_profile(base_test,meta_test,name) for name in PROFILE_OVERRIDES}

for name in PROFILE_OVERRIDES:
    print(name, {m:PROFILE_OVERRIDES[name][m] for m in PROFILE_OVERRIDES[name]})

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
# Full diagnostics for each frozen profile.
anchor_pu=per_user_ndcg(y,cal045_oof)
rows=[]; diagnostics={}
for name in PROFILE_OVERRIDES:
    pred=profile_oof[name]; ptest=profile_test[name]
    pu=per_user_ndcg(y,pred); delta_u=pu-anchor_pu
    fold_deltas=np.asarray([delta_u[ref_fold_id==f].mean() for f in range(N_FOLDS)],float)
    if V12_SMOKE:
        quarter_bounds=np.linspace(0,N_TRAIN,3,dtype=int)
    else:
        quarter_bounds=np.asarray([0,1000,2000,3000,4000],int)
    q_deltas=np.asarray([delta_u[quarter_bounds[i]:quarter_bounds[i+1]].mean() for i in range(len(quarter_bounds)-1)],float)
    ci=paired_boot(delta_u)
    rank_delta=rank_contributions(cal045_oof,pred)
    top5chg=top5_set_change_rate(ptest,cal045_test)
    orderchg=top5_order_change_rate(ptest,cal045_test)
    top1chg=top1_change_rate(ptest,cal045_test)
    no_chat=float(delta_u[NOCHAT_MASK].mean()) if NOCHAT_MASK.any() else np.nan
    k6=float(delta_u[K6_MASK].mean()) if K6_MASK.any() else np.nan
    invalid=float(delta_u[INVALID_MASK].mean()) if INVALID_MASK.any() else np.nan
    tail=float(delta_u[TAIL_MASK].mean())

    # Drop-one-changed-module stress: every module change is individually removable.
    drop_one={}
    for mid in PROFILE_OVERRIDES[name]:
        w=profile_vector(name).copy(); w[MODULES.index(mid)]=REFERENCE_W
        pp=((1-w)*base_oof+w*meta_oof).astype(np.float32)
        drop_one[mid]=float(ndcg5(y,pp)-ndcg5(y,cal045_oof))
    min_drop=float(min(drop_one.values())) if drop_one else 0.0

    score=float(ndcg5(y,pred)); delta=float(score-ndcg5(y,cal045_oof))
    hist_cons=PUBLIC_ANCHOR_LB+HIST_CONSERVATIVE_TRANSFER*delta
    hist_total=PUBLIC_ANCHOR_LB+HIST_TOTAL_CAL_TRANSFER*delta

    gate=bool(
        delta>=MIN_OOF_DELTA and
        int((fold_deltas>0).sum())>=MIN_FOLD_WINS and
        int((q_deltas>0).sum())>=MIN_QUARTER_WINS and
        ci[0]>0 and
        tail>=MIN_TAIL_DELTA and
        (np.isnan(no_chat) or no_chat>=MIN_NOCHAT_DELTA) and
        min_drop>=MIN_DROPONE_DELTA and
        top5chg<=MAX_TEST_TOP5_CHANGE+1e-12 and
        top1chg<=MAX_TEST_TOP1_CHANGE+1e-12
    )
    diagnostics[name]={
        "oof":score,"delta":delta,"bootstrap_ci":ci,
        "fold_deltas":fold_deltas.tolist(),"fold_wins":int((fold_deltas>0).sum()),
        "quarter_deltas":q_deltas.tolist(),"quarter_wins":int((q_deltas>0).sum()),
        "rank_contributions":rank_delta,
        "nochat_delta":no_chat,"k6_delta":k6,"invalid_assessment_delta":invalid,"tail_delta":tail,
        "drop_one_deltas":drop_one,"min_drop_one_delta":min_drop,
        "test_top5_change":top5chg,"test_top5_order_change":orderchg,"test_top1_change":top1chg,
        "historical_conservative_projection":hist_cons,
        "historical_total_calibration_projection":hist_total,
        "gate":gate,
    }
    rows.append({
        "profile":name,"oof":score,"delta":delta,"ci_low":ci[0],"ci_high":ci[1],
        "fold_wins":int((fold_deltas>0).sum()),"quarter_wins":int((q_deltas>0).sum()),
        "worst_fold":float(fold_deltas.min()),"worst_quarter":float(q_deltas.min()),
        "tail_delta":tail,"nochat_delta":no_chat,"k6_delta":k6,"invalid_delta":invalid,
        "min_drop_one":min_drop,"top5_change":top5chg,"top1_change":top1chg,
        "proj_1p20":hist_cons,"proj_1p37":hist_total,"gate":gate,
    })

profile_df=pd.DataFrame(rows).sort_values("delta",ascending=False).reset_index(drop=True)
display(profile_df.round(6))

# ---- cell 13 (markdown) ----
# ## Decision policy
#
# The leaderboard is too noisy to treat an OOF delta as a guaranteed public delta. V12 therefore separates **candidate strength** from **deployment admissibility**.
#
# The fixed priority is:
#
# 1. `leader6` if it clears every gate;
# 2. `balanced6` if `leader6` fails but `balanced6` clears every gate;
# 3. `safe3` if both six-module attacks fail but the conservative profile clears every gate;
# 4. exact `cal045` otherwise.
#
# The historical `1.20×` and `1.37×` same-family projections are diagnostics, **not fitting criteria and not guarantees**.

# ---- cell 14 (code) ----
selected_name=None
for name in PROFILE_PRIORITY:
    if diagnostics[name]["gate"]:
        selected_name=name
        break

if selected_name is None:
    primary_test=cal045_test.copy()
    primary_label="cal045_fallback"
else:
    primary_test=profile_test[selected_name].copy()
    primary_label=selected_name

print("\n================ V12 DECISION ================")
print("cal045 diagnostic OOF :",round(ndcg5(y,cal045_oof),6),"public",PUBLIC_ANCHOR_LB)
for name in ["safe3","power4","balanced6","leader6"]:
    d=diagnostics[name]
    print(f"{name:10s}: delta={d['delta']:+.6f} CI=({d['bootstrap_ci'][0]:+.6f},{d['bootstrap_ci'][1]:+.6f}) ",
          f"folds={d['fold_wins']}/{N_FOLDS} quarters={d['quarter_wins']}/{2 if V12_SMOKE else 4} ",
          f"tail={d['tail_delta']:+.6f} top5={d['test_top5_change']:.3f} top1={d['test_top1_change']:.3f} ",
          f"proj1.37={d['historical_total_calibration_projection']:.6f} gate={d['gate']}")
print("PRIMARY                 :",primary_label)
print("ACTION                  :",("Review submission_v12_primary.csv and the explicit selected profile before submission." if selected_name else "V12 gate failed; primary is exact cal045. Do not burn a slot on primary."))
print("================================================")

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
if not V12_SMOKE:
    paths={}
    paths['cal045']=save_submission(cal045_test,'submission_v12_cal045_reference.csv')
    for name in PROFILE_OVERRIDES:
        paths[name]=save_submission(profile_test[name],f'submission_v12_{name}.csv')
    paths['primary']=save_submission(primary_test,'submission_v12_primary.csv')
    profile_df.to_csv(root/'v12_profile_diagnostics.csv',index=False)
    np.savez_compressed(
        root/'v12_oof_artifacts.npz',
        y=y,base_oof=base_oof,meta_oof=meta_oof,cal045_oof=cal045_oof,
        base_test=base_test,meta_test=meta_test,cal045_test=cal045_test,
        safe3_oof=profile_oof['safe3'],power4_oof=profile_oof['power4'],
        balanced6_oof=profile_oof['balanced6'],leader6_oof=profile_oof['leader6'],
        ref_fold_id=ref_fold_id,
    )
    payload={
        "public_anchor_lb":PUBLIC_ANCHOR_LB,
        "leader_lb_at_design_time":LEADER_LB,
        "required_public_delta":REQUIRED_PUBLIC_DELTA,
        "cal045_oof":float(ndcg5(y,cal045_oof)),
        "profiles":diagnostics,
        "profile_overrides":PROFILE_OVERRIDES,
        "priority":PROFILE_PRIORITY,
        "selected_profile":selected_name,
        "primary_label":primary_label,
        "primary_file":paths['primary'].name,
        "reference_file":paths['cal045'].name,
    }
    with open(root/'v12_summary.json','w') as f: json.dump(payload,f,indent=2)
    print("saved",root/'v12_summary.json')
    print("saved",root/'v12_profile_diagnostics.csv')
    print("saved",root/'v12_oof_artifacts.npz')

# ---- cell 16 (markdown) ----
# ## Hard invariants
#
# These checks prevent accidental drift away from the V12 contract:
#
# - exact 17-module schema and 4,000/1,000 split;
# - cal045 remains exactly `0.55*sparse + 0.45*relevance`;
# - unspecified module weights remain 0.45;
# - all predictions finite;
# - no candidate may silently exceed the configured deployment-change budget and still pass its gate;
# - a failed global decision makes `primary` exactly equal to cal045.

# ---- cell 17 (code) ----
assert len(MODULES)==17 and y.shape==(4000,17) and N_TEST==1000
assert np.allclose(cal045_oof,.55*base_oof+.45*meta_oof,atol=1e-7)
assert np.allclose(cal045_test,.55*base_test+.45*meta_test,atol=1e-7)
for name in PROFILE_OVERRIDES:
    w=profile_vector(name)
    changed=set(PROFILE_OVERRIDES[name])
    for j,mid in enumerate(MODULES):
        if mid not in changed: assert abs(float(w[j])-REFERENCE_W)<1e-7
    assert np.isfinite(profile_oof[name]).all() and np.isfinite(profile_test[name]).all()
    if diagnostics[name]['gate']:
        assert diagnostics[name]['test_top5_change']<=MAX_TEST_TOP5_CHANGE+1e-12
        assert diagnostics[name]['test_top1_change']<=MAX_TEST_TOP1_CHANGE+1e-12
if selected_name is None:
    assert np.allclose(primary_test,cal045_test,atol=0,rtol=0)
else:
    assert np.allclose(primary_test,profile_test[selected_name],atol=0,rtol=0)

# Broad reproduction sanity guard; this catches accidental feature/parser drift without
# pretending OOF equals the public leaderboard.
if not V12_SMOKE:
    assert abs(ndcg5(y,cal045_oof)-0.662646)<0.003

print("V12 hard invariants: PASS")
