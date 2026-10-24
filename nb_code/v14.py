# Converted from notebooks/v14.ipynb: markdown cells become comments, %/! magics neutralized.

# ---- cell 0 (markdown) ----
# # MineToday V14 — Fully Nested Top-6 Pairwise Reranker
#
# **Immutable public champion:** V12.2 `Consensus3`, public NDCG@5 **0.65856**.
#
# ## Diagnosis carried forward
#
# V13/V13.1 proved that selective expert routing can add small, real gains, but the larger remaining opportunity is concentrated inside the champion's first six modules. The post-V13.1 audit showed three critical facts:
#
# 1. A whole-user oracle over existing experts is high but difficult to route reliably.
# 2. A naive second-stage pairwise model can report misleading `0.72x` OOF if its training scores are not genuinely cross-fitted.
# 3. When the leakage is removed, a **Top-6 pairwise reranker** still improves genuinely untouched future blocks, with the strongest improvement on the latest `U3001–U4000` block.
#
# V14 therefore does **one thing only**: learn the relative ordering among the six modules already ranked highest by exact Consensus3.
#
# ## Non-negotiable validation contract
#
# - Consensus3 is unchanged and remains the anchor.
# - Candidate pool is exactly **Top-6 Consensus3**.
# - Each outer fold creates **inner-OOF** sparse, relevance-calibration, and Consensus3 scores for the outer-training users.
# - The pairwise model is fit only on those inner-OOF training features.
# - The outer validation fold is untouched by sparse fitting, relevance calibration, pairwise fitting, early stopping, and blend selection.
# - A separate chronological audit trains only on `U0001–U3000` and evaluates once on `U3001–U4000`.
# - Exact target magnitude jitter is not regressed. Pair labels are only `relevance(A) > relevance(B)` and are weighted by their approximate NDCG swap importance.
# - The deployed blend is frozen at **65% pairwise / 35% champion prior** before this notebook run.
# - The reranker only permutes the champion's original Top-6 score levels. Every module outside Top-6 keeps its exact Consensus3 score.
#
# ## Why Top-6
#
# The clean diagnosis found that Top-5-only reranking was approximately neutral, Top-7 diluted the signal, while Top-6 consistently exposed the important #5/#6 boundary without opening the candidate set too widely.
#
# ## Kaggle
#
# Ready for **Kaggle T4 ×2**. The workload is CPU-oriented TF-IDF/Ridge/LightGBM; low GPU utilization is expected.
#
# ## Outputs
#
# - `submission_v14_top6_pairwise.csv`
# - `submission_v14_primary.csv` — pairwise candidate only if the frozen promotion gate passes, otherwise exact Consensus3
# - `submission_v14_consensus3_reference.csv`
# - `v14_summary.json`
# - `v14_fold_diagnostics.csv`
# - `v14_future_block.json`
# - `v14_oof_artifacts.npz`
#
# **Run all cells. Do not submit before inspecting the final diagnostics.**

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
V14_SMOKE=bool(int(os.getenv("V14_SMOKE","0")))
N_FOLDS=2 if V14_SMOKE else 5
N_INNER=2 if V14_SMOKE else 3
RIDGE_ALPHA=3.0
REFERENCE_W=0.45
LGB_ESTIMATORS=40 if V14_SMOKE else 1200
LGB_EARLY_STOP=5 if V14_SMOKE else 80
BOOTSTRAPS=250 if V14_SMOKE else 5000
PUBLIC_CHAMPION_LB=0.65856
PUBLIC_CAL045_LB=0.65639

PAIRWISE_TOPK=6
PAIRWISE_BLEND=0.65
PAIRWISE_ESTIMATORS=80 if V14_SMOKE else 900
PAIRWISE_EARLY_STOP=8 if V14_SMOKE else 80
PAIRWISE_LEAVES=9
PAIRWISE_MIN_CHILD=20 if V14_SMOKE else 100
PAIRWISE_REG_L1=2.0
PAIRWISE_REG_L2=15.0
PAIRWISE_MIN_WEIGHT=0.0025

# Frozen fail-closed promotion requirements.
MIN_STRICT_INCREMENT=-1.0 if V14_SMOKE else 0.0010
MIN_FOLD_WINS=1 if V14_SMOKE else 4
MAX_WORST_FOLD=-1.0 if V14_SMOKE else -0.0005
MIN_BOOT_CI_LOW=-1.0 if V14_SMOKE else 0.0
MIN_QUARTER_WINS=1 if V14_SMOKE else 3
MIN_TAIL_INCREMENT=-1.0 if V14_SMOKE else 0.0010
MIN_FUTURE_BLOCK_INCREMENT=-1.0 if V14_SMOKE else 0.0010
MAX_TEST_TOP5_SET_CHANGE=1.0 if V14_SMOKE else 0.35
MAX_TEST_TOP1_CHANGE=1.0 if V14_SMOKE else 0.30

random.seed(SEED); np.random.seed(SEED)
print("versions",np.__version__,pd.__version__,lgb.__version__,"V14_SMOKE",V14_SMOKE)
print("public champion",PUBLIC_CHAMPION_LB,"topk",PAIRWISE_TOPK,"blend",PAIRWISE_BLEND)

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
word_vec=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_df=.995,sublinear_tf=True,strip_accents="unicode",max_features=(2000 if V14_SMOKE else 50000),dtype=np.float32)
char_vec=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=2,sublinear_tf=True,max_features=(3000 if V14_SMOKE else 60000),dtype=np.float32)
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
# Exact V12.2 public champion. V14 never tunes or modifies its module calibration weights.

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
# ## Fully nested Top-6 pairwise machinery
#
# For each user, the champion supplies six candidate modules. The model sees organizer-derived user features plus the anchor/base/meta scores for a pair of candidates and predicts which member of the pair should rank higher. Pair weights approximate the absolute NDCG@5 effect of swapping their current champion positions.

# ---- cell 12 (code) ----
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
    ao=np.zeros((len(trg),N_MODULES),np.float32); av=np.zeros((len(vag),N_MODULES),np.float32)
    for m in range(N_MODULES):
        yy=Y_BIN[trg,m].astype(np.int32); its=[]
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


def topk_order(scores,k=PAIRWISE_TOPK):
    return np.argsort(-np.asarray(scores),axis=1)[:,:k]


def score_ranks(scores):
    order=np.argsort(-scores,axis=1); ranks=np.empty_like(order)
    ranks[np.arange(len(scores))[:,None],order]=np.arange(scores.shape[1])[None,:]
    return ranks

MODULE_EYE=np.eye(N_MODULES,dtype=np.float32)
PAIR_INDEX=[(a,b) for a in range(PAIRWISE_TOPK) for b in range(a+1,PAIRWISE_TOPK)]
PAIR_DISCOUNT=np.r_[DISCOUNT,0.0].astype(np.float32)


def _pair_feature(uf,anchor,base,meta,mods,pa,pb):
    ma=int(mods[pa]); mb=int(mods[pb])
    a=float(anchor[ma]); b=float(anchor[mb]); ba=float(base[ma]); bb=float(base[mb]);
    ra=float(meta[ma]); rb=float(meta[mb]);
    top_scores=anchor[mods]
    # Score geometry is deliberately low-dimensional and anchored to the current Top-6.
    scalar=np.asarray([
        pa/5.0,pb/5.0,(pb-pa)/5.0,
        a,b,a-b,ba,bb,ba-bb,ra,rb,ra-rb,
        ra-ba,rb-bb,(ra-ba)-(rb-bb),
        float(top_scores[0]-top_scores[-1]),float(top_scores[4]-top_scores[5]),
        float(np.std(top_scores)),float(np.std(anchor)),
    ],np.float32)
    return np.concatenate([uf.astype(np.float32),MODULE_EYE[ma],MODULE_EYE[mb],scalar])


def build_pairwise_rows(user_global,anchor,base,meta,target=None):
    user_global=np.asarray(user_global,int)
    mods_all=topk_order(anchor,PAIRWISE_TOPK)
    Xrows=[]; labels=[]; weights=[]; owner=[]; pair_meta=[]
    for li,g in enumerate(user_global):
        mods=mods_all[li]
        if target is not None:
            ideal=np.sort(target[li])[-5:][::-1]; idcg=float(np.dot(ideal,DISCOUNT)); idcg=max(idcg,1e-12)
        for pa,pb in PAIR_INDEX:
            ma=int(mods[pa]); mb=int(mods[pb])
            Xrows.append(_pair_feature(USER_FEATURES[g],anchor[li],base[li],meta[li],mods,pa,pb))
            owner.append(li); pair_meta.append((li,pa,pb,ma,mb))
            if target is not None:
                ya=float(target[li,ma]); yb=float(target[li,mb])
                if abs(ya-yb)<1e-12:
                    labels.append(-1); weights.append(0.0)
                else:
                    labels.append(int(ya>yb))
                    impact=abs((ya-yb)*(float(PAIR_DISCOUNT[pa])-float(PAIR_DISCOUNT[pb]))/idcg)
                    weights.append(max(PAIRWISE_MIN_WEIGHT,impact));
    Xrows=np.asarray(Xrows,np.float32); owner=np.asarray(owner,np.int32)
    pair_meta=np.asarray(pair_meta,np.int32)
    if target is None: return Xrows,owner,pair_meta,mods_all
    labels=np.asarray(labels,np.int8); weights=np.asarray(weights,np.float32)
    keep=labels>=0
    return Xrows[keep],labels[keep],weights[keep],owner[keep],pair_meta[keep],mods_all


def make_pairwise_clf(seed,n_estimators):
    return lgb.LGBMClassifier(
        objective="binary",n_estimators=int(n_estimators),learning_rate=.025,
        num_leaves=PAIRWISE_LEAVES,min_child_samples=PAIRWISE_MIN_CHILD,
        reg_alpha=PAIRWISE_REG_L1,reg_lambda=PAIRWISE_REG_L2,
        subsample=.85,colsample_bytree=.80,random_state=int(seed),n_jobs=4,verbosity=-1)


def fit_pairwise_model(user_global,anchor,base,meta,target,es_user_local,seed):
    Xp,yp,wp,owner,_,_=build_pairwise_rows(user_global,anchor,base,meta,target)
    es_user_local=np.asarray(es_user_local,int)
    es_mask=np.isin(owner,es_user_local); tr_mask=~es_mask
    # If a smoke split becomes tiny, fall back to deterministic row split without touching outer validation.
    if tr_mask.sum()<50 or es_mask.sum()<25:
        cut=max(1,int(.8*len(Xp))); tr_mask=np.arange(len(Xp))<cut; es_mask=~tr_mask
    probe=make_pairwise_clf(seed,PAIRWISE_ESTIMATORS)
    probe.fit(Xp[tr_mask],yp[tr_mask],sample_weight=wp[tr_mask],
              eval_set=[(Xp[es_mask],yp[es_mask])],eval_sample_weight=[wp[es_mask]],
              callbacks=[lgb.early_stopping(PAIRWISE_EARLY_STOP,verbose=False)])
    best=max(20,int(probe.best_iteration_ or PAIRWISE_ESTIMATORS))
    final=make_pairwise_clf(seed+997,best)
    final.fit(Xp,yp,sample_weight=wp)
    return final,best,{"rows":int(len(Xp)),"positive_rate":float(np.average(yp,weights=wp)),"weight_mean":float(wp.mean())}


def rerank_top6(model,user_global,anchor,base,meta,blend=PAIRWISE_BLEND):
    Xp,owner,pmeta,mods_all=build_pairwise_rows(user_global,anchor,base,meta,target=None)
    prob=model.predict_proba(Xp)[:,1].astype(np.float32)
    wins=np.zeros((len(anchor),PAIRWISE_TOPK),np.float32)
    cnt=np.zeros_like(wins)
    for p,(li,pa,pb,ma,mb) in enumerate(pmeta):
        q=float(prob[p]); wins[li,pa]+=q; wins[li,pb]+=1.0-q; cnt[li,pa]+=1; cnt[li,pb]+=1
    wins/=np.maximum(cnt,1)
    out=np.asarray(anchor,np.float32).copy()
    new_orders=np.empty_like(mods_all)
    for i,mods in enumerate(mods_all):
        levels=np.sort(anchor[i,mods])[::-1].astype(np.float32)
        av=anchor[i,mods].astype(np.float32)
        az=(av-av.mean())/(av.std()+1e-6)
        wv=wins[i]; wz=(wv-wv.mean())/(wv.std()+1e-6)
        key=float(blend)*wz+(1.0-float(blend))*az
        # Anchor position is a deterministic final tie breaker.
        key=key+1e-7*np.arange(PAIRWISE_TOPK,0,-1)
        local=np.argsort(-key); order=mods[local]; new_orders[i]=order
        out[i,order]=levels
    return out,new_orders,wins

# ---- cell 13 (markdown) ----
# ## Strict outer validation
#
# The second-stage learner never sees in-fold champion predictions. Each outer-training user receives inner-OOF anchor/base/meta values before pairwise training. The outer fold receives predictions from models fit only on outer-training users.

# ---- cell 14 (code) ----
ref_fold_id=np.full(N_TRAIN,-1,np.int16)
for f,(_,va) in enumerate(ref_folds): ref_fold_id[va]=f

strict_consensus3=np.zeros_like(y,np.float32)
strict_base=np.zeros_like(y,np.float32)
strict_meta=np.zeros_like(y,np.float32)
strict_pairwise=np.zeros_like(y,np.float32)
strict_pairwise_raw=np.zeros_like(y,np.float32)
fold_records=[]

for f,(trg,vag) in enumerate(ref_folds):
    print(f"V14 strict outer fold {f+1}/{N_FOLDS}")
    inn=inner_folds(trg)
    btr,bva=nested_base(trg,vag,inn)
    mtr,mva=nested_relevance(trg,vag,btr,bva,inn)
    ctr=apply_profile(btr,mtr)
    cva=apply_profile(bva,mva)
    # Early stopping uses only one inner user fold of outer-training data.
    es_users=inn[-1][1]
    pw,best,pinfo=fit_pairwise_model(trg,ctr,btr,mtr,y[trg],es_users,SEED+30000+f)
    rva,_,_=rerank_top6(pw,vag,cva,bva,mva,PAIRWISE_BLEND)
    rraw,_,_=rerank_top6(pw,vag,cva,bva,mva,1.0)
    strict_base[vag]=bva; strict_meta[vag]=mva; strict_consensus3[vag]=cva
    strict_pairwise[vag]=rva; strict_pairwise_raw[vag]=rraw
    d=ndcg5(y[vag],rva)-ndcg5(y[vag],cva)
    draw=ndcg5(y[vag],rraw)-ndcg5(y[vag],cva)
    fold_records.append({"fold":f,"champion":ndcg5(y[vag],cva),"pairwise":ndcg5(y[vag],rva),
                         "increment":d,"raw_increment":draw,"best_iteration":best,**pinfo})
    print(" champion",round(ndcg5(y[vag],cva),6),"pairwise",round(ndcg5(y[vag],rva),6),
          "delta",round(d,6),"raw-delta",round(draw,6),"best_iter",best)
    gc.collect()

champ_u=per_user_ndcg(y,strict_consensus3)
pair_u=per_user_ndcg(y,strict_pairwise)
user_delta=pair_u-champ_u
strict_increment=float(user_delta.mean()); strict_ci=boot_ci(user_delta)
fold_deltas=[float(r["increment"]) for r in fold_records]
fold_wins=int(sum(v>0 for v in fold_deltas)); worst_fold=float(min(fold_deltas))

bounds=np.linspace(0,N_TRAIN,3,dtype=int) if V14_SMOKE else np.asarray([0,1000,2000,3000,4000],int)
quarter_deltas=[float(user_delta[bounds[i]:bounds[i+1]].mean()) for i in range(len(bounds)-1)]
quarter_wins=int(sum(v>0 for v in quarter_deltas))
tail_increment=float(user_delta[-(N_TRAIN//2 if V14_SMOKE else 1000):].mean())

print("STRICT champion",round(float(champ_u.mean()),6),"V14",round(float(pair_u.mean()),6),
      "increment",round(strict_increment,6),"CI",tuple(round(x,6) for x in strict_ci))
print("folds",[round(x,6) for x in fold_deltas],"wins",fold_wins,"worst",round(worst_fold,6))
print("quarters",[round(x,6) for x in quarter_deltas],"tail",round(tail_increment,6))

# ---- cell 15 (markdown) ----
# ## Clean chronological future-block audit
#
# This is deliberately separate from random outer CV. The pairwise model and its anchor features are trained only from the first 3,000 labeled users, then evaluated once on `U3001–U4000`. This is the closest available supervised analogue to the competition's `U4001–U5000` test block.

# ---- cell 16 (code) ----
if V14_SMOKE:
    cut=max(20,N_TRAIN//2); future_tr=np.arange(cut); future_va=np.arange(cut,N_TRAIN)
else:
    future_tr=np.arange(0,3000); future_va=np.arange(3000,4000)

finn=list(StratifiedKFold(n_splits=N_INNER,shuffle=True,random_state=SEED+77).split(
    np.zeros(len(future_tr)),strata[future_tr]))
fbtr,fbva=nested_base(future_tr,future_va,finn)
fmtr,fmva=nested_relevance(future_tr,future_va,fbtr,fbva,finn)
fctr=apply_profile(fbtr,fmtr); fcva=apply_profile(fbva,fmva)
fpw,fbest,finfo=fit_pairwise_model(future_tr,fctr,fbtr,fmtr,y[future_tr],finn[-1][1],SEED+41000)
frva,_,_=rerank_top6(fpw,future_va,fcva,fbva,fmva,PAIRWISE_BLEND)
future_champion=float(ndcg5(y[future_va],fcva)); future_pairwise=float(ndcg5(y[future_va],frva))
future_block_delta=float(future_pairwise-future_champion)
future_user_delta=per_user_ndcg(y[future_va],frva)-per_user_ndcg(y[future_va],fcva)
future_ci=boot_ci(future_user_delta)
print("FUTURE BLOCK champion",round(future_champion,6),"pairwise",round(future_pairwise,6),
      "delta",round(future_block_delta,6),"CI",tuple(round(x,6) for x in future_ci),"best_iter",fbest)

# ---- cell 17 (markdown) ----
# ## Full-data inference
#
# The final pairwise learner is trained on exact ordinary OOF Consensus3/base/meta features for all 4,000 labeled users. Test features use the exact full-fit champion path. No test label or external information is used.

# ---- cell 18 (code) ----
all_train=np.arange(N_TRAIN); all_test=np.arange(N_TRAIN,N_TRAIN+N_TEST)
# Early stopping uses one reference fold only; all pairwise feature values are already OOF.
es_users=np.where(ref_fold_id==ref_fold_id.max())[0]
final_pw,final_best,final_info=fit_pairwise_model(all_train,consensus3_oof,base_oof,meta_oof,y,es_users,SEED+50000)
pairwise_test,test_order,test_wins=rerank_top6(final_pw,all_test,consensus3_test,base_test,meta_test,PAIRWISE_BLEND)

test_top5_set_change=top5_set_change_rate(pairwise_test,consensus3_test)
test_top5_order_change=float(np.mean(np.any(top5_indices(pairwise_test)!=top5_indices(consensus3_test),axis=1)))
test_top1_change=top1_change_rate(pairwise_test,consensus3_test)
test_top6_order_change=float(np.mean(np.any(topk_order(pairwise_test,6)!=topk_order(consensus3_test,6),axis=1)))
print("TEST geometry top5-set",round(test_top5_set_change,4),"top5-order",round(test_top5_order_change,4),
      "top1",round(test_top1_change,4),"top6-order",round(test_top6_order_change,4),"best_iter",final_best)

# ---- cell 19 (markdown) ----
# ## Promotion gate and exports
#
# `submission_v14_top6_pairwise.csv` always contains the experiment. `submission_v14_primary.csv` is fail-closed and contains the pairwise candidate only if both strict cross-fitting and the clean future-block audit pass the frozen requirements.

# ---- cell 20 (code) ----
promotion_gate=(
    strict_increment>=MIN_STRICT_INCREMENT and fold_wins>=MIN_FOLD_WINS and
    worst_fold>=MAX_WORST_FOLD and strict_ci[0]>MIN_BOOT_CI_LOW and
    quarter_wins>=MIN_QUARTER_WINS and tail_increment>=MIN_TAIL_INCREMENT and
    future_block_delta>=MIN_FUTURE_BLOCK_INCREMENT and
    test_top5_set_change<=MAX_TEST_TOP5_SET_CHANGE and test_top1_change<=MAX_TEST_TOP1_CHANGE
)

print("V14 DECISION",{
    "strict_champion":round(float(champ_u.mean()),6),"strict_pairwise":round(float(pair_u.mean()),6),
    "increment":round(strict_increment,6),"ci":[round(x,6) for x in strict_ci],
    "fold_wins":fold_wins,"worst_fold":round(worst_fold,6),"quarter_wins":quarter_wins,
    "tail":round(tail_increment,6),"future_block":round(future_block_delta,6),
    "test_top5_set_change":round(test_top5_set_change,4),"test_top1_change":round(test_top1_change,4),
    "gate":bool(promotion_gate)
})

def save_submission(scores,name):
    root=Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('/mnt/data')
    out=sample.copy(); out.loc[:,MODULES]=np.asarray(scores,np.float32)
    path=root/name; out.to_csv(path,index=False); chk=pd.read_csv(path)
    assert chk.shape==sample.shape and chk.user_id.is_unique and chk[MODULES].notna().all().all()
    assert chk.user_id.tolist()==sample.user_id.tolist(); print("saved",path); return path

root=Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('/mnt/data')
p_champ=save_submission(consensus3_test,'submission_v14_consensus3_reference.csv')
p_pair=save_submission(pairwise_test,'submission_v14_top6_pairwise.csv')
p_primary=save_submission(pairwise_test if promotion_gate else consensus3_test,'submission_v14_primary.csv')

pd.DataFrame(fold_records).to_csv(root/'v14_fold_diagnostics.csv',index=False)
future_summary={"train_users":int(len(future_tr)),"validation_users":int(len(future_va)),
                "champion":future_champion,"pairwise":future_pairwise,"future_block_delta":future_block_delta,
                "bootstrap_ci":[float(x) for x in future_ci],"best_iteration":int(fbest),**finfo}
json.dump(future_summary,open(root/'v14_future_block.json','w'),indent=2)

summary={
    "public_champion_lb":PUBLIC_CHAMPION_LB,"pairwise_topk":PAIRWISE_TOPK,"pairwise_blend":PAIRWISE_BLEND,
    "strict_consensus3":float(champ_u.mean()),"strict_pairwise":float(pair_u.mean()),
    "strict_increment":strict_increment,"bootstrap_ci":[float(x) for x in strict_ci],
    "fold_deltas":fold_deltas,"fold_wins":fold_wins,"worst_fold":worst_fold,
    "quarter_deltas":quarter_deltas,"quarter_wins":quarter_wins,"tail_increment":tail_increment,
    "future_block_delta":future_block_delta,"future_block_ci":[float(x) for x in future_ci],
    "test_top5_set_change":test_top5_set_change,"test_top5_order_change":test_top5_order_change,
    "test_top1_change":test_top1_change,"test_top6_order_change":test_top6_order_change,
    "final_pairwise_best_iteration":int(final_best),"final_pairwise_info":final_info,
    "promotion_gate":bool(promotion_gate),"primary_label":"top6_pairwise" if promotion_gate else "consensus3_fallback",
    "primary_file":p_primary.name,"candidate_file":p_pair.name,
}
json.dump(summary,open(root/'v14_summary.json','w'),indent=2)
np.savez_compressed(root/'v14_oof_artifacts.npz',y=y,ref_fold_id=ref_fold_id,
                    strict_consensus3=strict_consensus3,strict_pairwise=strict_pairwise,
                    strict_pairwise_raw=strict_pairwise_raw,user_delta=user_delta,
                    consensus3_oof=consensus3_oof,base_oof=base_oof,meta_oof=meta_oof,
                    consensus3_test=consensus3_test,pairwise_test=pairwise_test,test_order=test_order,test_wins=test_wins)

# ---- cell 21 (markdown) ----
# ## Hard invariants
#
# These checks verify that V14 is a true Top-6 permutation layer rather than an accidental new global scorer.

# ---- cell 22 (code) ----
assert len(MODULES)==17 and y.shape==(4000,17) and consensus3_test.shape==(1000,17)
assert strict_consensus3.shape==strict_pairwise.shape==y.shape
assert np.isfinite(strict_pairwise).all() and np.isfinite(pairwise_test).all()
# Outside the original champion Top-6, scores must be bitwise unchanged.
orig6=topk_order(consensus3_test,6)
mask=np.ones_like(consensus3_test,dtype=bool)
mask[np.arange(N_TEST)[:,None],orig6]=False
assert np.allclose(pairwise_test[mask],consensus3_test[mask],atol=0,rtol=0)
# The Top-6 candidate set itself must be identical; only its order may change.
new6=topk_order(pairwise_test,6)
assert all(set(a)==set(b) for a,b in zip(orig6,new6))
# Score levels inside Top-6 are permuted, not synthesized.
for i in range(N_TEST):
    assert np.allclose(np.sort(pairwise_test[i,orig6[i]]),np.sort(consensus3_test[i,orig6[i]]),atol=2e-7,rtol=0)
# Primary fallback semantics.
root=Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('/mnt/data')
primary=pd.read_csv(root/'submission_v14_primary.csv')[MODULES].to_numpy(np.float32)
expected=pairwise_test if promotion_gate else consensus3_test
assert np.allclose(primary,expected,atol=2e-7,rtol=0)
print("V14 hard invariants PASS")
