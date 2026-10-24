# Converted from notebooks/v12_2.ipynb: markdown cells become comments, %/! magics neutralized.

# ---- cell 0 (markdown) ----
# # MineToday V12.2 — Cross-Representation Consensus Calibration
#
# **Goal:** improve the public-validated V12 `Leader6` score **0.65792** enough to clear the current leader **0.65805**, while staying inside the same sparse + per-module relevance-calibration family that has already transferred positively to the public leaderboard.
#
# ## Why V12.2 exists
#
# V12 established that module-aware calibration is the correct local family: exact `cal045` improved from **0.65639** to **0.65792** public with the frozen `Leader6` profile. V12.1 `Micro4` increased ordinary OOF, but most of that incremental signal weakened under the stricter nested representation. V12.2 therefore refuses to optimize another 17-dimensional weight vector and instead evaluates only three **frozen, pre-registered** trust-region profiles against the exact public Leader6 reference in **two independent validation representations**.
#
# ### Frozen public reference: Leader6
#
# - `M006=.55, M007=.60, M010=.40, M011=.55, M012=.55, M017=.50`; all other modules `.45`.
#
# ### Frozen attacks
#
# - **Consensus3**: Leader6 plus `M002=.44`, `M015=.44`, `M017=.53`.
# - **Stable3**: Leader6 plus `M002=.44`, `M015=.44`, `M016=.44`.
# - **M15-only control**: Leader6 plus only `M015=.44`.
#
# No profile search, grid search, optimizer, leaderboard fitting, residual learner, pairwise swap, temporal expert, ranker, transformer, or external data is used.
#
# ## Double-validation contract
#
# 1. **Ordinary exact-path OOF** reproduces the V12/V3.1 inference family.
# 2. **Strict nested representation** rebuilds both sparse and relevance components inside untouched outer folds, so every strict validation user is excluded from the sparse backbone and calibrator used to score that user.
# 3. Every candidate is compared **incrementally against Leader6**, not merely against cal045.
# 4. A paired **combined bootstrap** averages each user's ordinary and strict incremental NDCG evidence.
# 5. A **second-order leave-one-fold-out selection diagnostic** chooses among the three frozen profiles using only the other four folds, then evaluates the selected profile on the held-out fold. It is diagnostic protection against local profile-selection inflation; it never changes the frozen profile weights.
# 6. Deployment perturbation is capped relative to the actual public Leader6 test scores.
#
# ## Decision policy
#
# `Consensus3` is the intended attack. `Stable3` is the conservative backup. `M15-only` is a control and is never preferred over a passing attack. If neither attack clears the double-validation gate, `submission_v12_2_primary.csv` is an **exact Leader6 fallback**.
#
# ## Competition compliance and runtime
#
# Only organizer-provided files are used. There is no direct LLM/Generative-AI prediction of test labels and no external dataset. The notebook is compatible with Kaggle **T4 ×2**; the proven pipeline itself is CPU-oriented TF-IDF + Ridge + LightGBM, so the GPUs may remain mostly idle without affecting correctness.
#
# ## Outputs
#
# - `submission_v12_2_consensus3.csv`
# - `submission_v12_2_stable3.csv`
# - `submission_v12_2_m15_only.csv`
# - `submission_v12_2_leader6_reference.csv`
# - `submission_v12_2_cal045_reference.csv`
# - `submission_v12_2_primary.csv`
# - `v12_2_summary.json`
# - `v12_2_diagnostics.csv`
# - `v12_2_second_order.csv`
# - `v12_2_oof_artifacts.npz`
#
# **Run all cells and read the final `V12.2 DECISION` block before using the last Kaggle submission.**

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
V122_SMOKE=bool(int(os.getenv("V122_SMOKE","0")))
V91_SMOKE=V122_SMOKE  # compatibility with the exact sparse-backbone cell
N_FOLDS=2 if V122_SMOKE else 5
N_INNER=2 if V122_SMOKE else 3
RIDGE_ALPHA=3.0
REFERENCE_W=0.45
LGB_ESTIMATORS=30 if V122_SMOKE else 1200
LGB_EARLY_STOP=5 if V122_SMOKE else 80
BOOTSTRAPS=300 if V122_SMOKE else 5000

PUBLIC_CAL045_LB=0.65639
PUBLIC_V12_LB=0.65792
LEADER_LB=0.65805
REQUIRED_PUBLIC_INCREMENT=LEADER_LB-PUBLIC_V12_LB

# Frozen incremental gates around the actual public Leader6 point.
MIN_ORD_INCREMENT=-0.01 if V122_SMOKE else 0.00020
MIN_STRICT_INCREMENT=-0.01 if V122_SMOKE else 0.00020
MIN_ORD_FOLD_WINS=1 if V122_SMOKE else 4
MIN_STRICT_FOLD_WINS=1 if V122_SMOKE else 4
MIN_TAIL_INCREMENT=-0.01 if V122_SMOKE else 0.0
MIN_COMBINED_CI_LOW=-1.0 if V122_SMOKE else 0.0
MAX_TOP5_CHANGE_VS_LEADER=0.015
MAX_TOP1_CHANGE_VS_LEADER=0.007
CONSERVATIVE_TRANSFER_FRACTION=0.50

# Second-order profile-selection diagnostic gates. These validate the selection process,
# not a tuned weight vector.
MIN_SECOND_ORDER_ORD=-0.01 if V122_SMOKE else 0.00010
MIN_SECOND_ORDER_STRICT=-0.01 if V122_SMOKE else 0.00010
MIN_SECOND_ORDER_COMBINED_WINS=1 if V122_SMOKE else 4
MIN_SECOND_ORDER_CI_LOW=-1.0 if V122_SMOKE else 0.0

random.seed(SEED); np.random.seed(SEED)
print("versions",np.__version__,pd.__version__,lgb.__version__,"V122_SMOKE",V122_SMOKE)
print("cal045 public",PUBLIC_CAL045_LB,"V12 Leader6 public",PUBLIC_V12_LB,
      "leader",LEADER_LB,"remaining gap",round(REQUIRED_PUBLIC_INCREMENT,6))

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
# ## Frozen V12.2 profiles
#
# The three V12.2 profiles are constants fixed before this run. All are tiny movements around the public-validated `Leader6`; unspecified module weights remain exactly `0.45`. `Consensus3` and `Stable3` are attacks, while `M15-only` is a low-dimensional control.

# ---- cell 10 (code) ----
PROFILE_OVERRIDES={
    "leader6": {
        "M_006":0.55,"M_007":0.60,"M_010":0.40,"M_011":0.55,"M_012":0.55,"M_017":0.50,
    },
    "consensus3": {
        "M_002":0.44,"M_006":0.55,"M_007":0.60,"M_010":0.40,"M_011":0.55,
        "M_012":0.55,"M_015":0.44,"M_017":0.53,
    },
    "stable3": {
        "M_002":0.44,"M_006":0.55,"M_007":0.60,"M_010":0.40,"M_011":0.55,
        "M_012":0.55,"M_015":0.44,"M_016":0.44,"M_017":0.50,
    },
    "m15_only": {
        "M_006":0.55,"M_007":0.60,"M_010":0.40,"M_011":0.55,"M_012":0.55,
        "M_015":0.44,"M_017":0.50,
    },
}
PROFILE_PRIORITY=("consensus3","stable3")
CONTROL_PROFILE="m15_only"


def profile_vector(name):
    w=np.full(N_MODULES,REFERENCE_W,np.float32)
    for mid,val in PROFILE_OVERRIDES[name].items():
        if mid not in MODULES: raise KeyError(mid)
        w[MODULES.index(mid)]=float(val)
    return w


def apply_profile(base,meta,name):
    w=profile_vector(name)
    return ((1.0-w)*base+w*meta).astype(np.float32)

ordinary_oof={name:apply_profile(base_oof,meta_oof,name) for name in PROFILE_OVERRIDES}
profile_test={name:apply_profile(base_test,meta_test,name) for name in PROFILE_OVERRIDES}
leader6_oof=ordinary_oof["leader6"]
leader6_test=profile_test["leader6"]

print("Frozen profiles:")
for name in PROFILE_OVERRIDES: print(name,PROFILE_OVERRIDES[name])

# ---- cell 11 (code) ----
# Validation helpers.
def top5_set_change_rate(a,b):
    ta,tb=top5_indices(a),top5_indices(b)
    return float(np.mean([set(x)!=set(z) for x,z in zip(ta,tb)]))

def top5_order_change_rate(a,b):
    return float(np.mean(np.any(top5_indices(a)!=top5_indices(b),axis=1)))

def top1_change_rate(a,b):
    return float(np.mean(np.argmax(a,axis=1)!=np.argmax(b,axis=1)))

def paired_boot(delta,seed=20260823):
    delta=np.asarray(delta,float); n=len(delta); rng=np.random.default_rng(seed)
    z=np.empty(BOOTSTRAPS,np.float64)
    for i in range(BOOTSTRAPS):
        ix=rng.integers(0,n,n); z[i]=delta[ix].mean()
    return tuple(float(v) for v in np.quantile(z,[.025,.975]))

def rank_contributions(anchor,cand):
    oa=np.argsort(-anchor,axis=1)[:,:5]; oc=np.argsort(-cand,axis=1)[:,:5]
    ideal=np.sort(y,axis=1)[:,-5:][:,::-1]; idcg=(ideal*DISCOUNT).sum(1)
    out=[]
    for r in range(5):
        z=(y[np.arange(N_TRAIN),oc[:,r]]-y[np.arange(N_TRAIN),oa[:,r]])*DISCOUNT[r]/np.maximum(idcg,1e-12)
        out.append(float(z.mean()))
    return out

def mean_by_fold(delta,fold_ids):
    return np.asarray([np.asarray(delta)[fold_ids==f].mean() for f in range(N_FOLDS)],float)

def quarter_means(delta):
    delta=np.asarray(delta,float)
    if V122_SMOKE:
        bounds=np.linspace(0,N_TRAIN,3,dtype=int)
    else:
        bounds=np.asarray([0,1000,2000,3000,4000],int)
    return np.asarray([delta[bounds[i]:bounds[i+1]].mean() for i in range(len(bounds)-1)],float)

ref_fold_id=np.full(N_TRAIN,-1,int)
for f,(_,va) in enumerate(ref_folds): ref_fold_id[va]=f
assert (ref_fold_id>=0).all()
TAIL_MASK=np.zeros(N_TRAIN,bool); TAIL_MASK[-(N_TRAIN//2 if V122_SMOKE else 1000):]=True

# ---- cell 12 (markdown) ----
# ## Strict nested representation
#
# This is the second validation representation required by the V12.2 contract. For each outer fold, the sparse backbone is rebuilt only from the outer-training users. The relevance calibrator then receives only inner-OOF sparse predictions for outer-training users and a sparse model trained on the full outer-training partition for the untouched outer-validation users. No outer-validation label can influence either component used to score that fold.

# ---- cell 13 (code) ----
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
        raw[lv]=mr.predict(X[gva]).astype(np.float32)
        top[lv]=mt.predict(X[gva]).astype(np.float32)
    mr=Ridge(alpha=RIDGE_ALPHA,solver="lsqr",tol=2e-3,max_iter=1000)
    mt=Ridge(alpha=RIDGE_ALPHA,solver="lsqr",tol=2e-3,max_iter=1000)
    mr.fit(X[trg],y[trg]); mt.fit(X[trg],Y_TOP[trg])
    val=(.9*mr.predict(X[vag])+.1*mt.predict(X[vag])).astype(np.float32)
    return (.9*raw+.1*top).astype(np.float32),val


def nested_relevance(trg,vag,btr,bva,inn):
    Xt=[np.column_stack([btr[:,m],USER_FEATURES[trg]]).astype(np.float32) for m in range(N_MODULES)]
    Xv=[np.column_stack([bva[:,m],USER_FEATURES[vag]]).astype(np.float32) for m in range(N_MODULES)]
    targets=Y_BIN[trg]
    ao=np.zeros((len(trg),N_MODULES),np.float32); av=np.zeros((len(vag),N_MODULES),np.float32)
    for m,mid in enumerate(MODULES):
        yy=targets[:,m].astype(np.int32); its=[]
        for f,(lt,lv) in enumerate(inn):
            clf=make_cal(SEED+f+31*m)
            clf.fit(Xt[m][lt],yy[lt],eval_set=[(Xt[m][lv],yy[lv])],
                    callbacks=[lgb.early_stopping(LGB_EARLY_STOP,verbose=False)])
            it=int(clf.best_iteration_ or LGB_ESTIMATORS); its.append(it)
            ao[lv,m]=clf.predict_proba(Xt[m][lv],num_iteration=it)[:,1].astype(np.float32)
        final_iters=max(20,int(np.median(its)))
        full=make_cal(SEED+997+31*m,final_iters); full.fit(Xt[m],yy)
        av[:,m]=full.predict_proba(Xv[m])[:,1].astype(np.float32)
    return ao,av

strict_base=np.zeros_like(y,np.float32)
strict_meta=np.zeros_like(y,np.float32)
for f,(trg,vag) in enumerate(ref_folds):
    print(f"strict outer fold {f+1}/{N_FOLDS}")
    inn=inner_folds(trg)
    btr,bva=nested_base(trg,vag,inn)
    _,av=nested_relevance(trg,vag,btr,bva,inn)
    strict_base[vag]=bva
    strict_meta[vag]=av
    gc.collect()

strict_oof={name:apply_profile(strict_base,strict_meta,name) for name in PROFILE_OVERRIDES}
strict_cal045=(.55*strict_base+.45*strict_meta).astype(np.float32)
print("ordinary cal045",round(ndcg5(y,cal045_oof),6),"strict cal045",round(ndcg5(y,strict_cal045),6))
print("ordinary Leader6",round(ndcg5(y,ordinary_oof['leader6']),6),
      "strict Leader6",round(ndcg5(y,strict_oof['leader6']),6))

# ---- cell 14 (markdown) ----
# ## Double-validation diagnostics and second-order selection test
#
# The attack is only credible if it improves the actual public reference `Leader6` in both representations. The second-order diagnostic additionally asks whether a profile selected without one fold still improves that unseen fold. Candidate weights remain frozen throughout; only the identity of the already-frozen profile is selected in this diagnostic.

# ---- cell 15 (code) ----
ord_leader_u=per_user_ndcg(y,ordinary_oof["leader6"])
strict_leader_u=per_user_ndcg(y,strict_oof["leader6"])
ord_cal_u=per_user_ndcg(y,cal045_oof)
strict_cal_u=per_user_ndcg(y,strict_cal045)

leader_ord_total=float(ndcg5(y,ordinary_oof["leader6"])-ndcg5(y,cal045_oof))
leader_strict_total=float(ndcg5(y,strict_oof["leader6"])-ndcg5(y,strict_cal045))
observed_v12_transfer_ratio=float((PUBLIC_V12_LB-PUBLIC_CAL045_LB)/leader_ord_total)

profile_deltas={}; diagnostics={}; rows=[]
for name in ("consensus3","stable3","m15_only"):
    ord_u=per_user_ndcg(y,ordinary_oof[name])-ord_leader_u
    strict_u=per_user_ndcg(y,strict_oof[name])-strict_leader_u
    combined_u=.5*(ord_u+strict_u)
    ord_inc=float(ord_u.mean()); strict_inc=float(strict_u.mean())
    combined_bootstrap_ci=paired_boot(combined_u,seed=20260823+len(name))
    ord_fold=mean_by_fold(ord_u,ref_fold_id); strict_fold=mean_by_fold(strict_u,ref_fold_id)
    ord_q=quarter_means(ord_u); strict_q=quarter_means(strict_u)
    ord_tail=float(ord_u[TAIL_MASK].mean()); strict_tail=float(strict_u[TAIL_MASK].mean())
    top5chg=top5_set_change_rate(profile_test[name],leader6_test)
    orderchg=top5_order_change_rate(profile_test[name],leader6_test)
    top1chg=top1_change_rate(profile_test[name],leader6_test)
    min_rep=min(ord_inc,strict_inc)
    public_projection_half=float(PUBLIC_V12_LB+CONSERVATIVE_TRANSFER_FRACTION*min_rep)
    public_projection_1to1=float(PUBLIC_V12_LB+ord_inc)
    public_projection_observed=float(PUBLIC_V12_LB+observed_v12_transfer_ratio*ord_inc)
    profile_deltas[name]={"ordinary":ord_u,"strict":strict_u,"combined":combined_u}
    diagnostics[name]={
        "ordinary_oof":float(ndcg5(y,ordinary_oof[name])),
        "strict_oof":float(ndcg5(y,strict_oof[name])),
        "ordinary_total_vs_cal045":float(ndcg5(y,ordinary_oof[name])-ndcg5(y,cal045_oof)),
        "strict_total_vs_cal045":float(ndcg5(y,strict_oof[name])-ndcg5(y,strict_cal045)),
        "ordinary_increment_vs_leader6":ord_inc,
        "strict_increment_vs_leader6":strict_inc,
        "combined_increment":float(combined_u.mean()),
        "combined_bootstrap_ci":combined_bootstrap_ci,
        "ordinary_fold_deltas":ord_fold.tolist(),"strict_fold_deltas":strict_fold.tolist(),
        "ordinary_fold_wins":int((ord_fold>0).sum()),"strict_fold_wins":int((strict_fold>0).sum()),
        "ordinary_quarter_deltas":ord_q.tolist(),"strict_quarter_deltas":strict_q.tolist(),
        "ordinary_tail_increment":ord_tail,"strict_tail_increment":strict_tail,
        "ordinary_rank_contributions":rank_contributions(ordinary_oof["leader6"],ordinary_oof[name]),
        "strict_rank_contributions":rank_contributions(strict_oof["leader6"],strict_oof[name]),
        "test_top5_change_vs_leader6":top5chg,
        "test_top5_order_change_vs_leader6":orderchg,
        "test_top1_change_vs_leader6":top1chg,
        "public_projection_half_min_rep":public_projection_half,
        "public_projection_1to1_ordinary":public_projection_1to1,
        "public_projection_observed_v12_transfer":public_projection_observed,
    }
    rows.append({
        "profile":name,"ord_inc":ord_inc,"strict_inc":strict_inc,"combined":float(combined_u.mean()),
        "ci_low":combined_bootstrap_ci[0],"ci_high":combined_bootstrap_ci[1],
        "ord_wins":int((ord_fold>0).sum()),"strict_wins":int((strict_fold>0).sum()),
        "ord_tail":ord_tail,"strict_tail":strict_tail,"top5_change":top5chg,"top1_change":top1chg,
        "proj_half":public_projection_half,"proj_1to1":public_projection_1to1,
    })

diagnostic_df=pd.DataFrame(rows).sort_values("combined",ascending=False).reset_index(drop=True)
display(diagnostic_df.round(6))

# Second-order leave-one-fold-out profile selection. Selection uses only the other folds
# and a conservative min(ordinary, strict) training-fold mean. No weights are tuned.
def second_order_selection():
    ord_selected=np.zeros(N_TRAIN,np.float64); strict_selected=np.zeros(N_TRAIN,np.float64); records=[]
    candidates=("consensus3","stable3","m15_only")
    for hold in range(N_FOLDS):
        train_mask=ref_fold_id!=hold; hold_mask=ref_fold_id==hold
        trials=[]
        for name in candidates:
            ou=profile_deltas[name]["ordinary"]; su=profile_deltas[name]["strict"]
            mo=float(ou[train_mask].mean()); ms=float(su[train_mask].mean())
            robust=min(mo,ms)
            trials.append((robust,name,mo,ms))
        trials.sort(reverse=True); robust,name,mo,ms=trials[0]
        ou=profile_deltas[name]["ordinary"]; su=profile_deltas[name]["strict"]
        ord_selected[hold_mask]=ou[hold_mask]; strict_selected[hold_mask]=su[hold_mask]
        records.append({
            "heldout_fold":hold,"selected_profile":name,"train_ordinary":mo,"train_strict":ms,
            "train_robust_score":robust,"heldout_ordinary":float(ou[hold_mask].mean()),
            "heldout_strict":float(su[hold_mask].mean()),
            "heldout_combined":float(.5*(ou[hold_mask].mean()+su[hold_mask].mean())),
        })
    combined=.5*(ord_selected+strict_selected)
    return ord_selected,strict_selected,combined,pd.DataFrame(records)

so_ord_u,so_strict_u,so_comb_u,second_order_df=second_order_selection()
second_order={
    "ordinary_mean":float(so_ord_u.mean()),"strict_mean":float(so_strict_u.mean()),
    "combined_mean":float(so_comb_u.mean()),"combined_bootstrap_ci":paired_boot(so_comb_u,seed=20260824),
    "ordinary_fold_wins":int(sum(x>0 for x in second_order_df.heldout_ordinary)),
    "strict_fold_wins":int(sum(x>0 for x in second_order_df.heldout_strict)),
    "combined_fold_wins":int(sum(x>0 for x in second_order_df.heldout_combined)),
    "selected_profiles":second_order_df.selected_profile.tolist(),
}
display(second_order_df.round(6))
print("second-order",second_order)

selection_gate=bool(
    second_order["ordinary_mean"]>=MIN_SECOND_ORDER_ORD and
    second_order["strict_mean"]>=MIN_SECOND_ORDER_STRICT and
    second_order["combined_fold_wins"]>=MIN_SECOND_ORDER_COMBINED_WINS and
    second_order["combined_bootstrap_ci"][0]>=MIN_SECOND_ORDER_CI_LOW
)

for name in diagnostics:
    d=diagnostics[name]
    attack_gate=bool(
        name!=CONTROL_PROFILE and
        d["ordinary_increment_vs_leader6"]>=MIN_ORD_INCREMENT and
        d["strict_increment_vs_leader6"]>=MIN_STRICT_INCREMENT and
        d["ordinary_fold_wins"]>=MIN_ORD_FOLD_WINS and
        d["strict_fold_wins"]>=MIN_STRICT_FOLD_WINS and
        d["combined_bootstrap_ci"][0]>=MIN_COMBINED_CI_LOW and
        d["ordinary_tail_increment"]>=MIN_TAIL_INCREMENT and
        d["strict_tail_increment"]>=MIN_TAIL_INCREMENT and
        d["test_top5_change_vs_leader6"]<=MAX_TOP5_CHANGE_VS_LEADER+1e-12 and
        d["test_top1_change_vs_leader6"]<=MAX_TOP1_CHANGE_VS_LEADER+1e-12 and
        d["public_projection_half_min_rep"]>LEADER_LB and
        selection_gate
    )
    d["attack_gate"]=attack_gate

diagnostic_df["gate"]=[diagnostics[n]["attack_gate"] for n in diagnostic_df.profile]
display(diagnostic_df.round(6))

# ---- cell 16 (markdown) ----
# ## Decision policy
#
# The last submission is too valuable for an automatic OOF hill-climb. V12.2 only promotes a frozen attack when both validation representations, the combined bootstrap, deployment-change budget, tail evidence, conservative 50%-transfer projection, and second-order profile-selection diagnostic agree. The fixed priority is `Consensus3 → Stable3 → exact Leader6 fallback`.

# ---- cell 17 (code) ----
selected_name=None
for name in PROFILE_PRIORITY:
    if diagnostics[name]["attack_gate"]:
        selected_name=name; break

if selected_name is None:
    primary_test=leader6_test.copy(); primary_label="leader6_fallback"
else:
    primary_test=profile_test[selected_name].copy(); primary_label=selected_name

print("\n================ V12.2 DECISION ================")
print("public cal045 / Leader6 / leader:",PUBLIC_CAL045_LB,PUBLIC_V12_LB,LEADER_LB)
print("remaining public gap             :",round(REQUIRED_PUBLIC_INCREMENT,6))
print("ordinary/strict Leader6 total    :",f"{leader_ord_total:+.6f}",f"{leader_strict_total:+.6f}")
print("observed V12 transfer ratio      :",round(observed_v12_transfer_ratio,4))
for name in ("consensus3","stable3","m15_only"):
    d=diagnostics[name]
    print(f"{name:10s}: ord={d['ordinary_increment_vs_leader6']:+.6f} strict={d['strict_increment_vs_leader6']:+.6f} ",
          f"CI=({d['combined_bootstrap_ci'][0]:+.6f},{d['combined_bootstrap_ci'][1]:+.6f}) ",
          f"folds={d['ordinary_fold_wins']}/{d['strict_fold_wins']} tail={d['ordinary_tail_increment']:+.6f}/{d['strict_tail_increment']:+.6f} ",
          f"top5={d['test_top5_change_vs_leader6']:.3f} top1={d['test_top1_change_vs_leader6']:.3f} ",
          f"proj50={d['public_projection_half_min_rep']:.6f} gate={d['attack_gate']}")
print("second-order means ord/strict    :",f"{second_order['ordinary_mean']:+.6f}",f"{second_order['strict_mean']:+.6f}")
print("second-order combined CI/wins    :",tuple(round(x,6) for x in second_order['combined_bootstrap_ci']),second_order['combined_fold_wins'])
print("second-order selected profiles   :",second_order['selected_profiles'])
print("selection gate                   :",selection_gate)
print("PRIMARY                          :",primary_label)
print("ACTION                           :",("Candidate cleared V12.2 double-validation. Inspect artifacts before using the final Kaggle slot." if selected_name else "No V12.2 attack cleared the double-validation gate. Primary is exact Leader6; do not waste the final slot on the fallback."))
print("==================================================")

# ---- cell 18 (code) ----
def save_submission(scores,name):
    out=sample.copy(); out.loc[:,MODULES]=scores
    root=Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('/mnt/data')
    path=root/name; out.to_csv(path,index=False)
    chk=pd.read_csv(path)
    assert chk.shape==sample.shape and chk.user_id.tolist()==sample.user_id.tolist() and chk.user_id.is_unique
    assert chk[MODULES].notna().all().all() and np.isfinite(chk[MODULES].to_numpy()).all()
    print("saved",path); return path

root=Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('/mnt/data')
if not V122_SMOKE:
    paths={}
    paths['cal045']=save_submission(cal045_test,'submission_v12_2_cal045_reference.csv')
    paths['leader6']=save_submission(leader6_test,'submission_v12_2_leader6_reference.csv')
    paths['consensus3']=save_submission(profile_test['consensus3'],'submission_v12_2_consensus3.csv')
    paths['stable3']=save_submission(profile_test['stable3'],'submission_v12_2_stable3.csv')
    paths['m15_only']=save_submission(profile_test['m15_only'],'submission_v12_2_m15_only.csv')
    paths['primary']=save_submission(primary_test,'submission_v12_2_primary.csv')
    diagnostic_df.to_csv(root/'v12_2_diagnostics.csv',index=False)
    second_order_df.to_csv(root/'v12_2_second_order.csv',index=False)
    np.savez_compressed(
        root/'v12_2_oof_artifacts.npz',
        y=y,base_oof=base_oof,meta_oof=meta_oof,cal045_oof=cal045_oof,
        leader6_oof=ordinary_oof['leader6'],consensus3_oof=ordinary_oof['consensus3'],
        stable3_oof=ordinary_oof['stable3'],m15_only_oof=ordinary_oof['m15_only'],
        strict_base=strict_base,strict_meta=strict_meta,strict_cal045=strict_cal045,
        strict_leader6=strict_oof['leader6'],strict_consensus3=strict_oof['consensus3'],
        strict_stable3=strict_oof['stable3'],strict_m15_only=strict_oof['m15_only'],
        base_test=base_test,meta_test=meta_test,leader6_test=leader6_test,
        consensus3_test=profile_test['consensus3'],stable3_test=profile_test['stable3'],
        m15_only_test=profile_test['m15_only'],ref_fold_id=ref_fold_id,
        second_order_ordinary=so_ord_u,second_order_strict=so_strict_u,
    )
    payload={
        "public_cal045_lb":PUBLIC_CAL045_LB,"public_v12_leader6_lb":PUBLIC_V12_LB,
        "leader_lb":LEADER_LB,"required_public_increment":REQUIRED_PUBLIC_INCREMENT,
        "profiles":PROFILE_OVERRIDES,"profile_priority":list(PROFILE_PRIORITY),
        "leader6_ordinary_total_vs_cal045":leader_ord_total,
        "leader6_strict_total_vs_cal045":leader_strict_total,
        "observed_v12_transfer_ratio":observed_v12_transfer_ratio,
        "diagnostics":diagnostics,"second_order":second_order,
        "second_order_records":second_order_df.to_dict(orient='records'),
        "selection_gate":selection_gate,"primary_label":primary_label,
        "primary_file":paths['primary'].name,
        "consensus3_file":paths['consensus3'].name,"stable3_file":paths['stable3'].name,
        "control_file":paths['m15_only'].name,"leader6_reference_file":paths['leader6'].name,
    }
    with open(root/'v12_2_summary.json','w') as f: json.dump(payload,f,indent=2)
    print("saved",root/'v12_2_summary.json')
    print("saved",root/'v12_2_diagnostics.csv')
    print("saved",root/'v12_2_second_order.csv')
    print("saved",root/'v12_2_oof_artifacts.npz')

# ---- cell 19 (markdown) ----
# ## Hard invariants
#
# These checks protect the V12.2 contract: exact 17-module schema, immutable cal045 and Leader6 definitions, only the intended trust-region module changes, finite ordinary/strict/test predictions, capped deployment change, frozen candidate profiles, and exact Leader6 fallback whenever the double-validation gate rejects both attacks.

# ---- cell 20 (code) ----
assert len(MODULES)==17 and y.shape==(4000,17) and N_TEST==1000
assert np.allclose(cal045_oof,.55*base_oof+.45*meta_oof,atol=1e-7)
assert np.allclose(cal045_test,.55*base_test+.45*meta_test,atol=1e-7)

expected_leader={"M_006":0.55,"M_007":0.60,"M_010":0.40,"M_011":0.55,"M_012":0.55,"M_017":0.50}
expected_consensus={"M_002":0.44,"M_006":0.55,"M_007":0.60,"M_010":0.40,"M_011":0.55,"M_012":0.55,"M_015":0.44,"M_017":0.53}
expected_stable={"M_002":0.44,"M_006":0.55,"M_007":0.60,"M_010":0.40,"M_011":0.55,"M_012":0.55,"M_015":0.44,"M_016":0.44,"M_017":0.50}
expected_m15={"M_006":0.55,"M_007":0.60,"M_010":0.40,"M_011":0.55,"M_012":0.55,"M_015":0.44,"M_017":0.50}
assert PROFILE_OVERRIDES["leader6"]==expected_leader
assert PROFILE_OVERRIDES["consensus3"]==expected_consensus
assert PROFILE_OVERRIDES["stable3"]==expected_stable
assert PROFILE_OVERRIDES["m15_only"]==expected_m15

for arr in [base_oof,meta_oof,cal045_oof,base_test,meta_test,cal045_test,strict_base,strict_meta,strict_cal045]:
    assert np.isfinite(arr).all()
for name in PROFILE_OVERRIDES:
    assert np.isfinite(ordinary_oof[name]).all() and np.isfinite(strict_oof[name]).all() and np.isfinite(profile_test[name]).all()

# Profiles may differ from Leader6 only by their declared tiny trust-region modules.
allowed={"consensus3":{"M_002","M_015","M_017"},"stable3":{"M_002","M_015","M_016"},"m15_only":{"M_015"}}
wl=profile_vector("leader6")
for name,changed in allowed.items():
    w=profile_vector(name)
    actual={MODULES[j] for j in range(N_MODULES) if abs(float(w[j]-wl[j]))>1e-8}
    assert actual==changed,(name,actual,changed)

assert len(second_order_df)==N_FOLDS
if selected_name is not None:
    d=diagnostics[selected_name]
    assert d['attack_gate'] and selection_gate
    assert d['test_top5_change_vs_leader6']<=MAX_TOP5_CHANGE_VS_LEADER+1e-12
    assert d['test_top1_change_vs_leader6']<=MAX_TOP1_CHANGE_VS_LEADER+1e-12
    assert np.allclose(primary_test,profile_test[selected_name],atol=0,rtol=0)
else:
    assert np.allclose(primary_test,leader6_test,atol=0,rtol=0)

if not V122_SMOKE:
    assert abs(ndcg5(y,cal045_oof)-0.662646)<0.003
    assert abs(ndcg5(y,leader6_oof)-0.664058)<0.003
    assert 0.660 < ndcg5(y,strict_cal045) < 0.668

print("V12.2 hard invariants: PASS")
