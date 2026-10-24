# Converted from notebooks/v15.ipynb: markdown cells become comments, %/! magics neutralized.

# ---- cell 0 (markdown) ----
# # MineToday V15 — Fully Nested Surgical Utility Router
#
# **Purpose:** repair the V13.1 failure mode without touching the proven V12.2 Consensus3 calibration family.
#
# This notebook implements the post-V13.1 diagnosis directly:
#
# - exact V12.2 Consensus3 remains the immutable public champion anchor;
# - V9.1 and V10-C are **proposal generators only**, never whole-user replacements;
# - the second-stage utility model is trained **inside each outer fold** from inner-OOF expert geometry;
# - only **single Top-5 membership swaps** are allowed; order-only routing is forbidden;
# - only champion ranks 3–5 may be dropped, and additions must come from champion ranks 6–9;
# - the router predicts **expected ΔNDCG@5 directly**, not probability of “beneficial”;
# - routing uses an **absolute utility threshold with abstention**, never a fixed top-fraction quota;
# - test actions come from fold-matched 80%-train models and require cross-fold action consensus;
# - final accepted actions are applied surgically to the exact full-fit Consensus3 submission;
# - every promotion path is fail-closed to the exact `0.65856` Consensus3 champion.
#
# No external dataset and no test-label inference are used.

# ---- cell 1 (code) ----
import os,re,json,ast,html,hashlib,random,warnings,gc,time,sys,subprocess
from pathlib import Path
from collections import Counter,defaultdict
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
V15_SMOKE=bool(int(os.getenv("V15_SMOKE","0")))
N_FOLDS=2 if V15_SMOKE else 5
N_INNER=2 if V15_SMOKE else 3
RIDGE_ALPHA=3.0
REFERENCE_W=0.45
LGB_ESTIMATORS=45 if V15_SMOKE else 1200
LGB_EARLY_STOP=6 if V15_SMOKE else 80
ORD_ESTIMATORS=60 if V15_SMOKE else 300
ORD_EARLY_STOP=8 if V15_SMOKE else 40
DIVERSITY_SEEDS=[137] if V15_SMOKE else [137,2026,7,31415]
ORDINAL_WEIGHT=0.10
META_SCALE_GRID=np.asarray([1.0,1.45] if V15_SMOKE else [1.35,1.45,1.55],np.float32)
UTILITY_ESTIMATORS=45 if V15_SMOKE else 1000
UTILITY_EARLY_STOP=6 if V15_SMOKE else 70
BOOTSTRAPS=250 if V15_SMOKE else 5000

PUBLIC_CHAMPION_LB=0.65856
PUBLIC_V131_LB=0.65754
PUBLIC_CAL045_LB=0.65639
PUBLIC_V91_LB=0.65432

# Exact V10-C expert definition retained only as a proposal source.
V10C_P5_WEIGHT=0.08
V10C_UTILITY_WEIGHT=0.15
V91_RANK_WEIGHTS=np.asarray([1.00,0.00,0.25,0.75,1.00,1.00,0.50,0.25]+[0.00]*9,np.float32)

# Surgical action policy from the V13.1/V14 diagnosis.
MIN_DROP_POS=2              # 0-based: only ranks 3,4,5 may be removed
MAX_ADD_RANK=9              # 1-based champion rank: additions only from ranks 6..9
MIN_ACTION_SUPPORT=2 if V15_SMOKE else 8
UTILITY_EST=60 if V15_SMOKE else 240
UTILITY_LEAVES=5
UTILITY_MIN_CHILD=8 if V15_SMOKE else 70
UTILITY_REG_L1=4.0
UTILITY_REG_L2=24.0
UTILITY_THRESHOLDS=np.asarray([0.0,0.003] if V15_SMOKE else [0.003,0.005,0.0075,0.010,0.015,0.020,0.030],np.float32)
MAX_INNER_ROUTE_RATE=0.20 if V15_SMOKE else 0.035
MIN_INNER_ROUTE_USERS=1 if V15_SMOKE else 12
MIN_INNER_POS_FOLDS=1 if V15_SMOKE else 2

# Fold-matched test consensus. Primary is deliberately conservative.
VOTE_MIN_PRIMARY=4 if not V15_SMOKE else 1
VOTE_MIN_ALT=3 if not V15_SMOKE else 1
MIN_TEST_MEAN_UTILITY=0.0 if V15_SMOKE else 0.005
MAX_TEST_ROUTE_RATE=1.0 if V15_SMOKE else 0.025

# Fail-closed promotion requirements against exact Consensus3.
MIN_STRICT_INCREMENT=-1.0 if V15_SMOKE else 0.00020
MIN_FOLD_WINS=1 if V15_SMOKE else 4
MAX_WORST_FOLD=-1.0 if V15_SMOKE else -0.00020
MIN_BOOT_CI_LOW=-1.0 if V15_SMOKE else 0.0
MIN_QUARTER_WINS=1 if V15_SMOKE else 3
MIN_TAIL_INCREMENT=-1.0 if V15_SMOKE else 0.0
MIN_FUTURE_INCREMENT=-1.0 if V15_SMOKE else 0.0

random.seed(SEED); np.random.seed(SEED)
print("versions",np.__version__,pd.__version__,lgb.__version__,"V15_SMOKE",V15_SMOKE)
print("public",{"champion":PUBLIC_CHAMPION_LB,"v13.1":PUBLIC_V131_LB},
      "action",{"drop_pos>=":MIN_DROP_POS+1,"add_rank<=":MAX_ADD_RANK},
      "votes",{"primary":VOTE_MIN_PRIMARY,"alt":VOTE_MIN_ALT})

# ---- cell 2 (code) ----
def locate_data_dir():
    required = {
        "chat_history.csv", "modules_catalog.csv", "user_assessments.csv",
        "train_relevance.csv", "test.csv", "sample_submission.csv"
    }
    roots = [Path("/kaggle/input"), Path("/mnt/data/_data_comp"), Path("/mnt/data/comp_v3"), Path("/mnt/data/comp_full"), Path("/mnt/data"), Path(".")]
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

if V15_SMOKE:
    train=train.iloc[:320].copy()
    test=test.iloc[:120].copy()
    keep=set(train.user_id)|set(test.user_id)
    chat=chat[chat.user_id.isin(keep)].copy()
    assessments=assessments[assessments.user_id.isin(keep)].copy()
    sample=sample[sample.user_id.isin(set(test.user_id))].copy().reset_index(drop=True)

MODULES = list(train.columns[1:])
N_MODULES = len(MODULES)
N_TRAIN = len(train)
N_TEST = len(test)

assert MODULES == list(sample.columns[1:])
assert N_MODULES == 17
assert (N_TRAIN == 4000 and N_TEST == 1000) if not V15_SMOKE else (N_TRAIN == 320 and N_TEST == 120)
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
assert USER_FEATURES.shape == (len(all_users), 134)
assert np.isfinite(USER_FEATURES).all()
print("USER_FEATURES:", USER_FEATURES.shape)
print("example augmented text:", texts[1][:500])

# ---- cell 7 (code) ----
# Exact public-validated V3.1 sparse backbone.
t0=time.time()
word_vec=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_df=.995,sublinear_tf=True,strip_accents="unicode",max_features=(2000 if V15_SMOKE else 50000),dtype=np.float32)
char_vec=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=2,sublinear_tf=True,max_features=(3000 if V15_SMOKE else 60000),dtype=np.float32)
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
# ## Immutable public champion
#
# The only deployment anchor is the exact V12.2 Consensus3 calibration profile that scored **0.65856** publicly. The router may only exchange two existing score levels for a single approved membership repair.

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
# ## Proposal experts and fully nested first stage
#
# V9.1 and V10-C are retained because they contain complementary candidate-set information, but their complete rankings are never deployed. For every outer fold, inner-OOF versions are created for router training and outer-train fits are used for the untouched outer validation plus test block.

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
        random_state=int(seed),n_jobs=4,verbosity=-1)

def exp_rel(prel,P):
    p5=np.minimum(P[5],prel); p4=np.minimum(P[4],p5); p3=np.minimum(P[3],p4)
    p2=np.minimum(P[2],p3); p1=np.minimum(P[1],p2)
    return (.25*prel+.15*(p5+p4+p3+p2+p1)).astype(np.float32)

IDEAL_TOP5=np.sort(y,axis=1)[:,-5:][:,::-1]
IDCG5=(IDEAL_TOP5*DISCOUNT).sum(1).astype(np.float32)
Y_UTILITY=(y/np.maximum(IDCG5[:,None],1e-12)).astype(np.float32)

def make_utility_reg(seed,n_estimators=None):
    return lgb.LGBMRegressor(
        objective="regression_l2",n_estimators=int(UTILITY_ESTIMATORS if n_estimators is None else n_estimators),
        learning_rate=.025,num_leaves=11,min_child_samples=60,
        reg_alpha=1.0,reg_lambda=8.0,colsample_bytree=.8,
        random_state=int(seed),n_jobs=4,verbosity=-1)

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

def fold_ids_from_splits(n,splits):
    z=np.full(n,-1,np.int16)
    for f,(_,va) in enumerate(splits): z[va]=f
    assert np.all(z>=0); return z

def module_ablation_fold_deltas(ytrue,anchor,candidate,fold_id,n_folds):
    residual=(candidate-anchor).astype(np.float32); out=np.zeros((n_folds,N_MODULES),np.float64)
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
    j=int(np.argmax(vals)); return float(META_SCALE_GRID[j]),np.asarray(vals,np.float64)

def apply_shrinkage(anchor,candidate,support,scale):
    return (anchor+float(scale)*(candidate-anchor)*support.reshape(1,-1)).astype(np.float32)

# ---- cell 13 (code) ----
def inner_folds(trg,seed_offset=0):
    return list(StratifiedKFold(n_splits=N_INNER,shuffle=True,random_state=SEED+int(seed_offset)).split(
        np.zeros(len(trg)),strata[trg]))

def nested_base(trg,evalg,inn):
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
    ev=(.9*mr.predict(X[evalg])+.1*mt.predict(X[evalg])).astype(np.float32)
    return (.9*raw+.1*top).astype(np.float32),ev

def nested_anchor_relevance(trg,evalg,btr,bev,inn,seed_offset=0):
    Xt=[np.column_stack([btr[:,m],USER_FEATURES[trg]]).astype(np.float32) for m in range(N_MODULES)]
    Xe=[np.column_stack([bev[:,m],USER_FEATURES[evalg]]).astype(np.float32) for m in range(N_MODULES)]
    ao=np.zeros((len(trg),N_MODULES),np.float32); ae=np.zeros((len(evalg),N_MODULES),np.float32)
    its={mid:[] for mid in MODULES}
    for m,mid in enumerate(MODULES):
        yy=Y_BIN[trg,m].astype(np.int32)
        for f,(lt,lv) in enumerate(inn):
            clf=make_cal(SEED+int(seed_offset)+f+31*m)
            clf.fit(Xt[m][lt],yy[lt],eval_set=[(Xt[m][lv],yy[lv])],
                    callbacks=[lgb.early_stopping(LGB_EARLY_STOP,verbose=False)])
            it=int(clf.best_iteration_ or LGB_ESTIMATORS); its[mid].append(it)
            ao[lv,m]=clf.predict_proba(Xt[m][lv],num_iteration=it)[:,1].astype(np.float32)
        fi=max(20,int(np.median(its[mid])))
        full=make_cal(SEED+int(seed_offset)+997+31*m,fi); full.fit(Xt[m],yy)
        ae[:,m]=full.predict_proba(Xe[m])[:,1].astype(np.float32)
    return ao,ae,its,Xt,Xe

def nested_v91_extension(trg,evalg,btr,bev,ao,ae,its,Xt,Xe,inn,seed_offset=0):
    Do=[]; De=[]
    for s in DIVERSITY_SEEDS:
        oo=np.zeros_like(ao); ee=np.zeros_like(ae)
        for m,mid in enumerate(MODULES):
            yy=Y_BIN[trg,m].astype(np.int32)
            for f,(lt,lv) in enumerate(inn):
                it=int(its[mid][f]); clf=make_cal(s+int(seed_offset)+f+31*m,it); clf.fit(Xt[m][lt],yy[lt])
                oo[lv,m]=clf.predict_proba(Xt[m][lv])[:,1].astype(np.float32)
            fi=max(20,int(np.median(its[mid])))
            full=make_cal(s+int(seed_offset)+997+31*m,fi); full.fit(Xt[m],yy)
            ee[:,m]=full.predict_proba(Xe[m])[:,1].astype(np.float32)
        Do.append(align_mat(oo,ao,oo)); De.append(align_mat(oo,ao,ee))
    do=np.mean(np.stack(Do),0).astype(np.float32); de=np.mean(np.stack(De),0).astype(np.float32)

    rp=RANK_POS[trg]; probs_o={}; probs_e={}
    for tt in range(1,6):
        po=np.zeros_like(ao); pe=np.zeros_like(ae); yyall=(rp<=tt).astype(np.int32)
        for m in range(N_MODULES):
            oits=[]
            for f,(lt,lv) in enumerate(inn):
                clf=make_ord_cal(SEED+int(seed_offset)+9000+101*tt+f+31*m)
                clf.fit(Xt[m][lt],yyall[lt,m],eval_set=[(Xt[m][lv],yyall[lv,m])],
                        callbacks=[lgb.early_stopping(ORD_EARLY_STOP,verbose=False)])
                it=int(clf.best_iteration_ or ORD_ESTIMATORS); oits.append(it)
                po[lv,m]=clf.predict_proba(Xt[m][lv],num_iteration=it)[:,1].astype(np.float32)
            fi=max(20,int(np.median(oits)))
            full=make_ord_cal(SEED+int(seed_offset)+12000+101*tt+31*m,fi); full.fit(Xt[m],yyall[:,m])
            pe[:,m]=full.predict_proba(Xe[m])[:,1].astype(np.float32)
        probs_o[tt]=po; probs_e[tt]=pe
    oo=exp_rel(ao,probs_o); oe=exp_rel(ae,probs_e)

    atr=(.55*btr+.45*ao).astype(np.float32); aev=(.55*bev+.45*ae).astype(np.float32)
    dtr=(.55*btr+.45*do).astype(np.float32); dev=(.55*bev+.45*de).astype(np.float32)
    v9tr=((1-ORDINAL_WEIGHT)*dtr+ORDINAL_WEIGHT*oo).astype(np.float32)
    v9ev=((1-ORDINAL_WEIGHT)*dev+ORDINAL_WEIGHT*oe).astype(np.float32)
    inner_id=fold_ids_from_splits(len(trg),inn)
    evid=module_ablation_fold_deltas(y[trg],atr,v9tr,inner_id,N_INNER)
    support,pos=module_support(evid); scale,_=select_shrinkage(y[trg],atr,v9tr,support)
    v91tr=apply_shrinkage(atr,v9tr,support,scale)
    v91ev=apply_shrinkage(aev,v9ev,support,scale)
    return v91tr,v91ev,support,pos,scale

def nested_v10_heads(trg,evalg,Xt,Xe,inn,seed_offset=0):
    p5_o=np.zeros((len(trg),N_MODULES),np.float32); p5_e=np.zeros((len(evalg),N_MODULES),np.float32)
    u_o=np.zeros_like(p5_o); u_e=np.zeros_like(p5_e)
    for m in range(N_MODULES):
        yp=Y_TOP5[trg,m].astype(np.int32); yu=Y_UTILITY[trg,m].astype(np.float32)
        p_iters=[]; u_iters=[]
        for f,(lt,lv) in enumerate(inn):
            clf=make_cal(SEED+int(seed_offset)+21000+f+31*m)
            clf.fit(Xt[m][lt],yp[lt],eval_set=[(Xt[m][lv],yp[lv])],
                    callbacks=[lgb.early_stopping(LGB_EARLY_STOP,verbose=False)])
            pit=int(clf.best_iteration_ or LGB_ESTIMATORS); p_iters.append(pit)
            p5_o[lv,m]=clf.predict_proba(Xt[m][lv],num_iteration=pit)[:,1].astype(np.float32)
            reg=make_utility_reg(SEED+int(seed_offset)+23000+f+31*m)
            reg.fit(Xt[m][lt],yu[lt],eval_set=[(Xt[m][lv],yu[lv])],
                    callbacks=[lgb.early_stopping(UTILITY_EARLY_STOP,verbose=False)])
            uit=int(reg.best_iteration_ or UTILITY_ESTIMATORS); u_iters.append(uit)
            u_o[lv,m]=reg.predict(Xt[m][lv],num_iteration=uit).astype(np.float32)
        pfull=make_cal(SEED+int(seed_offset)+22000+31*m,max(20,int(np.median(p_iters)))); pfull.fit(Xt[m],yp)
        p5_e[:,m]=pfull.predict_proba(Xe[m])[:,1].astype(np.float32)
        ufull=make_utility_reg(SEED+int(seed_offset)+24000+31*m,max(20,int(np.median(u_iters)))); ufull.fit(Xt[m],yu)
        u_e[:,m]=ufull.predict(Xe[m]).astype(np.float32)
    return p5_o,p5_e,u_o,u_e

def nested_outer_bundle(trg,evalg,seed_offset=0):
    inn=inner_folds(trg,seed_offset)
    btr,bev=nested_base(trg,evalg,inn)
    mtr,mev,its,Xt,Xe=nested_anchor_relevance(trg,evalg,btr,bev,inn,seed_offset)
    ctr=apply_profile(btr,mtr); cev=apply_profile(bev,mev)
    if V15_SMOKE:
        # Fast deterministic proposal surrogates exercise the entire routing/export path.
        # Full Kaggle runs never enter this branch.
        rtr=(mtr-btr).astype(np.float32); rev=(mev-bev).astype(np.float32)
        v91tr=(ctr+0.10*rtr).astype(np.float32); v91ev=(cev+0.10*rev).astype(np.float32)
        v10tr=(ctr-0.08*rtr).astype(np.float32); v10ev=(cev-0.08*rev).astype(np.float32)
        return {"inn":inn,"base_tr":btr,"base_ev":bev,"meta_tr":mtr,"meta_ev":mev,
                "champ_tr":ctr,"champ_ev":cev,"v91_tr":v91tr,"v91_ev":v91ev,
                "v10c_tr":v10tr,"v10c_ev":v10ev,"v91_scale":0.0,"v91_support":np.ones(N_MODULES,np.float32)}
    v91tr,v91ev,support,pos,scale=nested_v91_extension(trg,evalg,btr,bev,mtr,mev,its,Xt,Xe,inn,seed_offset)
    p5tr,p5ev,utr,uev=nested_v10_heads(trg,evalg,Xt,Xe,inn,seed_offset)
    a45tr=(.55*btr+.45*mtr).astype(np.float32); a45ev=(.55*bev+.45*mev).astype(np.float32)
    v10tr=make_v10c(a45tr,p5tr,utr,v91tr); v10ev=make_v10c(a45ev,p5ev,uev,v91ev)
    return {"inn":inn,"base_tr":btr,"base_ev":bev,"meta_tr":mtr,"meta_ev":mev,
            "champ_tr":ctr,"champ_ev":cev,"v91_tr":v91tr,"v91_ev":v91ev,
            "v10c_tr":v10tr,"v10c_ev":v10ev,"v91_scale":scale,"v91_support":support}

# ---- cell 14 (markdown) ----
# ## Surgical expected-utility action model
#
# The V13.1 whole-user router is removed. Each expert may only propose one set repair. The deployed action swaps the champion score level of one rank-3/4/5 module with one rank-6/7/8/9 module. No other score changes are permitted.

# ---- cell 15 (code) ----
MODULE_EYE=np.eye(N_MODULES,dtype=np.float32)
EXPERT_NAMES=("v91","v10c")

def top5_order(scores):
    idx=np.argpartition(-scores,4,axis=1)[:,:5]; local=np.take_along_axis(scores,idx,axis=1)
    return np.take_along_axis(idx,np.argsort(-local,axis=1),axis=1)

def score_ranks(scores):
    order=np.argsort(-scores,axis=1); ranks=np.empty_like(order)
    ranks[np.arange(len(scores))[:,None],order]=np.arange(N_MODULES)[None,:]
    return ranks

def apply_surgical_action(row,drop,add):
    out=np.asarray(row,np.float32).copy(); d=float(out[int(drop)]); a=float(out[int(add)])
    out[int(drop)]=a; out[int(add)]=d
    return out

def _candidate_feature(global_user,champ,base,meta,expert,expert_id,drop,add):
    cr=np.argsort(np.argsort(-champ)); er=np.argsort(np.argsort(-expert))
    co=np.argsort(-champ); cs=champ[co]
    drop_pos=int(cr[drop]); add_rank=int(cr[add]); edrop=int(er[drop]); eadd=int(er[add])
    scalar=np.asarray([
        drop_pos/4.0,add_rank/(N_MODULES-1),edrop/(N_MODULES-1),eadd/(N_MODULES-1),
        champ[drop],champ[add],champ[drop]-champ[add],
        base[drop],base[add],base[drop]-base[add],
        meta[drop],meta[add],meta[drop]-meta[add],
        expert[drop],expert[add],expert[drop]-expert[add],
        expert[drop]-champ[drop],expert[add]-champ[add],
        meta[drop]-base[drop],meta[add]-base[add],
        cs[3]-cs[4],cs[4]-cs[5],float(np.std(cs[:6])),float(np.std(champ)),
    ],np.float32)
    ex=np.zeros(2,np.float32); ex[int(expert_id)]=1.0
    return np.concatenate([USER_FEATURES[int(global_user)],ex,MODULE_EYE[int(drop)],MODULE_EYE[int(add)],scalar])

def build_swap_candidates(user_global,champ,base,meta,expert_dict,target=None):
    user_global=np.asarray(user_global,int); champ=np.asarray(champ,np.float32)
    ctop=top5_order(champ); cr=score_ranks(champ)
    Xrows=[]; owners=[]; experts=[]; drops=[]; adds=[]; drop_pos=[]; deltas=[]
    for i,g in enumerate(user_global):
        cset=set(ctop[i])
        for eid,name in enumerate(EXPERT_NAMES):
            expert=np.asarray(expert_dict[name],np.float32)
            etop=top5_order(expert[i:i+1])[0]; eset=set(etop)
            out=list(cset-eset); inn=list(eset-cset)
            if len(out)!=1 or len(inn)!=1: continue
            d=int(out[0]); a=int(inn[0]); dp=int(cr[i,d]); ar=int(cr[i,a])+1
            if dp<MIN_DROP_POS or ar<6 or ar>MAX_ADD_RANK: continue
            Xrows.append(_candidate_feature(g,champ[i],base[i],meta[i],expert[i],eid,d,a))
            owners.append(i); experts.append(eid); drops.append(d); adds.append(a); drop_pos.append(dp)
            if target is not None:
                swapped=apply_surgical_action(champ[i],d,a)
                deltas.append(float(per_user_ndcg(target[i:i+1],swapped[None,:])[0]-per_user_ndcg(target[i:i+1],champ[i:i+1])[0]))
    dim=134+2+17+17+24
    Xo=np.asarray(Xrows,np.float32) if Xrows else np.zeros((0,dim),np.float32)
    out={"X":Xo,"owner":np.asarray(owners,np.int32),"expert":np.asarray(experts,np.int8),
         "drop":np.asarray(drops,np.int8),"add":np.asarray(adds,np.int8),"drop_pos":np.asarray(drop_pos,np.int8)}
    if target is not None: out["delta"]=np.asarray(deltas,np.float32)
    return out

def signature_counts(cand):
    cnt=Counter()
    for e,d,a,p in zip(cand["expert"],cand["drop"],cand["add"],cand["drop_pos"]):
        cnt[(int(e),int(d),int(a),int(p))]+=1
    return cnt

def candidate_support(cand,cnt):
    return np.asarray([cnt[(int(e),int(d),int(a),int(p))] for e,d,a,p in
                       zip(cand["expert"],cand["drop"],cand["add"],cand["drop_pos"])],np.int16)

def fit_utility_model(Xc,delta,mask,seed):
    mask=np.asarray(mask,bool)
    if mask.sum()<20: return None
    model=lgb.LGBMRegressor(
        objective="huber",alpha=.85,n_estimators=UTILITY_EST,learning_rate=.025,
        num_leaves=UTILITY_LEAVES,max_depth=3,min_child_samples=UTILITY_MIN_CHILD,
        reg_alpha=UTILITY_REG_L1,reg_lambda=UTILITY_REG_L2,
        subsample=.85,colsample_bytree=.75,random_state=int(seed),n_jobs=4,verbosity=-1)
    w=(0.35+8.0*np.minimum(np.abs(delta),0.10)).astype(np.float32)
    model.fit(Xc[mask],delta[mask],sample_weight=w[mask])
    return model

def crossfit_utility_predictions(cand,inn,eligible,seed):
    pred=np.full(len(cand["X"]),np.nan,np.float32)
    for k,(ut,uv) in enumerate(inn):
        tr=np.isin(cand["owner"],ut) & eligible; va=np.isin(cand["owner"],uv) & eligible
        model=fit_utility_model(cand["X"],cand["delta"],tr,seed+101*k)
        if model is not None and va.any(): pred[va]=model.predict(cand["X"][va]).astype(np.float32)
    return pred

def user_action_delta_from_candidates(cand,pred,threshold,n_users,eligible=None):
    if eligible is None: eligible=np.ones(len(pred),bool)
    chosen=np.full(n_users,-1,np.int32); best=np.full(n_users,-np.inf,np.float32)
    for r,(u,p) in enumerate(zip(cand["owner"],pred)):
        if not eligible[r] or not np.isfinite(p) or p<=threshold: continue
        if p>best[u]: best[u]=p; chosen[u]=r
    ud=np.zeros(n_users,np.float32); routed=chosen>=0
    if routed.any(): ud[routed]=cand["delta"][chosen[routed]]
    return chosen,ud,best

def choose_absolute_threshold(cand,cv_pred,eligible,inn,n_users):
    rows=[]; inner_id=fold_ids_from_splits(n_users,inn)
    for th in UTILITY_THRESHOLDS:
        chosen,ud,_=user_action_delta_from_candidates(cand,cv_pred,float(th),n_users,eligible)
        rate=float(np.mean(chosen>=0)); nroute=int(np.sum(chosen>=0)); mean=float(ud.mean())
        fdelta=[float(ud[inner_id==k].mean()) for k in range(N_INNER)]
        wins=sum(v>0 for v in fdelta); routed_mean=float(ud[chosen>=0].mean()) if nroute else 0.0
        rows.append({"threshold":float(th),"increment":mean,"route_rate":rate,"routes":nroute,
                     "positive_folds":int(wins),"routed_mean":routed_mean,"fold_deltas":fdelta})
    valid=[r for r in rows if r["route_rate"]<=MAX_INNER_ROUTE_RATE and r["routes"]>=MIN_INNER_ROUTE_USERS and
           r["increment"]>0 and r["routed_mean"]>0 and r["positive_folds"]>=MIN_INNER_POS_FOLDS]
    if not valid: return np.inf,rows
    best=max(valid,key=lambda r:(r["increment"],r["threshold"]))
    return float(best["threshold"]),rows

def select_eval_actions(cand,pred,threshold,support,n_users):
    eligible=support>=MIN_ACTION_SUPPORT
    chosen=np.full(n_users,-1,np.int32); best=np.full(n_users,-np.inf,np.float32)
    for r,(u,p) in enumerate(zip(cand["owner"],pred)):
        if not eligible[r] or not np.isfinite(p) or p<=threshold: continue
        if p>best[u]: best[u]=p; chosen[u]=r
    return chosen,best

def apply_selected_actions(champ,cand,chosen):
    out=np.asarray(champ,np.float32).copy()
    for u,r in enumerate(chosen):
        if r>=0: out[u]=apply_surgical_action(out[u],cand["drop"][r],cand["add"][r])
    return out

# ---- cell 16 (markdown) ----
# ## Fully nested outer validation + fold-matched test proposals
#
# The utility model for an outer fold sees only inner-OOF proposals from that outer fold's training users. The untouched outer validation is predicted once. The same outer-trained first-stage models and utility model also predict test actions, so deployment geometry matches the validation geometry.

# ---- cell 17 (code) ----
TEST_GLOBAL=np.arange(N_TRAIN,N_TRAIN+N_TEST)
ref_fold_id=np.full(N_TRAIN,-1,np.int16)
for f,(_,va) in enumerate(ref_folds): ref_fold_id[va]=f

strict_champion=np.zeros_like(y,np.float32)
strict_surgical=np.zeros_like(y,np.float32)
strict_route=np.zeros(N_TRAIN,bool)
strict_expert=np.full(N_TRAIN,-1,np.int8)
strict_drop=np.full(N_TRAIN,-1,np.int8)
strict_add=np.full(N_TRAIN,-1,np.int8)
strict_pred_utility=np.zeros(N_TRAIN,np.float32)
fold_records=[]; threshold_records=[]; fold_test_actions=[]

for f,(trg,vag) in enumerate(ref_folds):
    print(f"V15 fully nested outer fold {f+1}/{N_FOLDS}")
    evalg=np.concatenate([vag,TEST_GLOBAL])
    bun=nested_outer_bundle(trg,evalg,seed_offset=1000*f)
    nv=len(vag)
    ctr=bun["champ_tr"]; cva=bun["champ_ev"][:nv]; cte=bun["champ_ev"][nv:]
    btr=bun["base_tr"]; bva=bun["base_ev"][:nv]; bte=bun["base_ev"][nv:]
    mtr=bun["meta_tr"]; mva=bun["meta_ev"][:nv]; mte=bun["meta_ev"][nv:]
    etr={"v91":bun["v91_tr"],"v10c":bun["v10c_tr"]}
    eva={"v91":bun["v91_ev"][:nv],"v10c":bun["v10c_ev"][:nv]}
    ete={"v91":bun["v91_ev"][nv:],"v10c":bun["v10c_ev"][nv:]}

    train_cand=build_swap_candidates(trg,ctr,btr,mtr,etr,target=y[trg])
    counts=signature_counts(train_cand); train_sup=candidate_support(train_cand,counts)
    eligible=train_sup>=MIN_ACTION_SUPPORT
    cv_pred=crossfit_utility_predictions(train_cand,bun["inn"],eligible,SEED+30000+f*1000)
    threshold,th_rows=choose_absolute_threshold(train_cand,cv_pred,eligible,bun["inn"],len(trg))
    for r in th_rows: threshold_records.append({"outer_fold":f,**r})

    final_model=fit_utility_model(train_cand["X"],train_cand["delta"],eligible,SEED+40000+f)
    val_cand=build_swap_candidates(vag,cva,bva,mva,eva,target=None)
    test_cand=build_swap_candidates(TEST_GLOBAL,cte,bte,mte,ete,target=None)
    val_sup=candidate_support(val_cand,counts); test_sup=candidate_support(test_cand,counts)
    if final_model is None or not np.isfinite(threshold):
        val_pred=np.full(len(val_cand["X"]),-np.inf,np.float32); test_pred=np.full(len(test_cand["X"]),-np.inf,np.float32)
    else:
        val_pred=final_model.predict(val_cand["X"]).astype(np.float32) if len(val_cand["X"]) else np.zeros(0,np.float32)
        test_pred=final_model.predict(test_cand["X"]).astype(np.float32) if len(test_cand["X"]) else np.zeros(0,np.float32)
    val_chosen,val_best=select_eval_actions(val_cand,val_pred,threshold,val_sup,len(vag))
    test_chosen,test_best=select_eval_actions(test_cand,test_pred,threshold,test_sup,N_TEST)
    rval=apply_selected_actions(cva,val_cand,val_chosen)

    strict_champion[vag]=cva; strict_surgical[vag]=rval
    rr=val_chosen>=0; strict_route[vag]=rr
    for li,r in enumerate(val_chosen):
        if r>=0:
            g=vag[li]; strict_expert[g]=val_cand["expert"][r]; strict_drop[g]=val_cand["drop"][r]; strict_add[g]=val_cand["add"][r]; strict_pred_utility[g]=val_best[li]

    test_rows=[]
    for u,r in enumerate(test_chosen):
        if r>=0:
            test_rows.append({"test_index":u,"drop":int(test_cand["drop"][r]),"add":int(test_cand["add"][r]),
                              "expert":int(test_cand["expert"][r]),"pred_utility":float(test_best[u]),
                              "threshold":float(threshold),"support":int(test_sup[r])})
    fold_test_actions.append(test_rows)

    vu=per_user_ndcg(y[vag],rval)-per_user_ndcg(y[vag],cva)
    nroute=int(rr.sum()); beneficial=int(np.sum(vu>1e-12)); harmful=int(np.sum(vu<-1e-12))
    fold_records.append({"fold":f,"champion":ndcg5(y[vag],cva),"surgical":ndcg5(y[vag],rval),
                         "increment":float(vu.mean()),"threshold":float(threshold) if np.isfinite(threshold) else None,
                         "route_rate":float(rr.mean()),"routes":nroute,"beneficial":beneficial,"harmful":harmful,
                         "train_candidates":int(len(train_cand["X"])),"eligible_train_candidates":int(eligible.sum()),
                         "val_candidates":int(len(val_cand["X"])),"test_actions":int(len(test_rows)),
                         "v91_scale":float(bun["v91_scale"])})
    print(" champion",round(ndcg5(y[vag],cva),6),"surgical",round(ndcg5(y[vag],rval),6),
          "delta",round(float(vu.mean()),6),"threshold",threshold,"routes",nroute,
          "benefit/harm",beneficial,harmful,"test proposals",len(test_rows))
    gc.collect()

champ_u=per_user_ndcg(y,strict_champion); surg_u=per_user_ndcg(y,strict_surgical)
user_delta=surg_u-champ_u; strict_increment=float(user_delta.mean()); strict_ci=boot_ci(user_delta)
fold_deltas=[float(r["increment"]) for r in fold_records]; fold_wins=int(sum(v>0 for v in fold_deltas)); worst_fold=float(min(fold_deltas))
bounds=np.linspace(0,N_TRAIN,3,dtype=int) if V15_SMOKE else np.asarray([0,1000,2000,3000,4000],int)
quarter_deltas=[float(user_delta[bounds[i]:bounds[i+1]].mean()) for i in range(len(bounds)-1)]
quarter_wins=int(sum(v>0 for v in quarter_deltas)); tail_increment=float(user_delta[-(N_TRAIN//2 if V15_SMOKE else 1000):].mean())
print("STRICT",{"champion":round(float(champ_u.mean()),6),"surgical":round(float(surg_u.mean()),6),
                "increment":round(strict_increment,6),"ci":tuple(round(x,6) for x in strict_ci),
                "folds":[round(x,6) for x in fold_deltas],"quarters":[round(x,6) for x in quarter_deltas],
                "route_rate":round(float(strict_route.mean()),4)})

# ---- cell 18 (markdown) ----
# ## Chronological U3001–U4000 audit
#
# This is a separate one-way future block. Threshold selection and the utility model see only U0001–U3000. U3001–U4000 is touched once.

# ---- cell 19 (code) ----
if V15_SMOKE:
    cut=max(20,N_TRAIN//2); future_tr=np.arange(cut); future_va=np.arange(cut,N_TRAIN)
else:
    future_tr=np.arange(0,3000); future_va=np.arange(3000,4000)
fb=nested_outer_bundle(future_tr,future_va,seed_offset=7777)
ftrain={"v91":fb["v91_tr"],"v10c":fb["v10c_tr"]}; feval={"v91":fb["v91_ev"],"v10c":fb["v10c_ev"]}
fcand=build_swap_candidates(future_tr,fb["champ_tr"],fb["base_tr"],fb["meta_tr"],ftrain,target=y[future_tr])
fcnt=signature_counts(fcand); fsup=candidate_support(fcand,fcnt); felig=fsup>=MIN_ACTION_SUPPORT
fcv=crossfit_utility_predictions(fcand,fb["inn"],felig,SEED+51000)
fth,frows=choose_absolute_threshold(fcand,fcv,felig,fb["inn"],len(future_tr))
fmodel=fit_utility_model(fcand["X"],fcand["delta"],felig,SEED+52000)
fvc=build_swap_candidates(future_va,fb["champ_ev"],fb["base_ev"],fb["meta_ev"],feval,target=None)
fvs=candidate_support(fvc,fcnt)
if fmodel is None or not np.isfinite(fth): fp=np.full(len(fvc["X"]),-np.inf,np.float32)
else: fp=fmodel.predict(fvc["X"]).astype(np.float32) if len(fvc["X"]) else np.zeros(0,np.float32)
fchoose,_=select_eval_actions(fvc,fp,fth,fvs,len(future_va))
fscore=apply_selected_actions(fb["champ_ev"],fvc,fchoose)
future_delta_user=per_user_ndcg(y[future_va],fscore)-per_user_ndcg(y[future_va],fb["champ_ev"])
future_increment=float(future_delta_user.mean()); future_ci=boot_ci(future_delta_user)
print("FUTURE",{"champion":round(ndcg5(y[future_va],fb["champ_ev"]),6),"surgical":round(ndcg5(y[future_va],fscore),6),
                "increment":round(future_increment,6),"ci":tuple(round(x,6) for x in future_ci),
                "threshold":fth,"routes":int(np.sum(fchoose>=0))})

# ---- cell 20 (markdown) ----
# ## Fold-consensus deployment on the exact public champion
#
# Each test action must independently survive the fully nested router in several outer folds. The final action is only a score swap on the exact full-fit Consensus3 row. Two audit candidates are exported: 3/5 and 4/5 agreement.

# ---- cell 21 (code) ----
def aggregate_test_actions(fold_test_actions,min_votes):
    by_user=defaultdict(list)
    for f,rows in enumerate(fold_test_actions):
        for r in rows:
            by_user[int(r["test_index"])].append((f,r))
    chosen=[]
    full_top=top5_order(consensus3_test); full_rank=score_ranks(consensus3_test)
    for u,items in by_user.items():
        groups=defaultdict(list)
        for f,r in items: groups[(int(r["drop"]),int(r["add"]))].append((f,r))
        ranked=[]
        for key,vals in groups.items():
            votes=len(vals); mu=float(np.mean([x[1]["pred_utility"] for x in vals])); sup=min(int(x[1]["support"]) for x in vals)
            ranked.append((votes,mu,sup,key,vals))
        ranked.sort(key=lambda z:(z[0],z[1],z[2]),reverse=True)
        if not ranked: continue
        votes,mu,sup,(d,a),vals=ranked[0]
        if votes<min_votes or mu<MIN_TEST_MEAN_UTILITY: continue
        # Revalidate against the exact full-fit public champion geometry.
        cset=set(full_top[u]); dp=int(full_rank[u,d]); ar=int(full_rank[u,a])+1
        if d not in cset or a in cset or dp<MIN_DROP_POS or ar<6 or ar>MAX_ADD_RANK: continue
        chosen.append({"test_index":u,"drop":d,"add":a,"votes":votes,"mean_utility":mu,"min_support":sup,
                       "experts":"|".join(str(int(x[1]["expert"])) for x in vals)})
    return chosen

def apply_test_action_list(actions):
    out=consensus3_test.copy()
    for r in actions: out[int(r["test_index"])]=apply_surgical_action(out[int(r["test_index"])],r["drop"],r["add"])
    return out

actions3=aggregate_test_actions(fold_test_actions,VOTE_MIN_ALT)
actions4=aggregate_test_actions(fold_test_actions,VOTE_MIN_PRIMARY)
vote3_test=apply_test_action_list(actions3); vote4_test=apply_test_action_list(actions4)

def test_geometry(scores):
    return {"route_rate":top5_set_change_rate(scores,consensus3_test),
            "top5_order":float(np.mean(np.any(top5_order(scores)!=top5_order(consensus3_test),axis=1))),
            "top1":top1_change_rate(scores,consensus3_test)}
geom3=test_geometry(vote3_test); geom4=test_geometry(vote4_test)
print("TEST vote3",len(actions3),geom3)
print("TEST vote4",len(actions4),geom4)

# ---- cell 22 (markdown) ----
# ## Fail-closed decision and exports
#
# A candidate is not promoted merely because its mean OOF rises. The fully nested outer score, CI, fold stability, chronology, untouched future block, and deployment intervention budget must all agree.

# ---- cell 23 (code) ----
promotion_gate=(
    strict_increment>=MIN_STRICT_INCREMENT and fold_wins>=MIN_FOLD_WINS and worst_fold>=MAX_WORST_FOLD and
    strict_ci[0]>MIN_BOOT_CI_LOW and quarter_wins>=MIN_QUARTER_WINS and tail_increment>=MIN_TAIL_INCREMENT and
    future_increment>=MIN_FUTURE_INCREMENT
)

# Prefer 4/5 consensus. 3/5 is only eligible if it stays inside the same conservative route budget.
if promotion_gate and geom4["route_rate"]>0 and geom4["route_rate"]<=MAX_TEST_ROUTE_RATE:
    primary_scores=vote4_test; primary_label="vote4_surgical"
elif promotion_gate and geom3["route_rate"]>0 and geom3["route_rate"]<=MAX_TEST_ROUTE_RATE:
    primary_scores=vote3_test; primary_label="vote3_surgical"
else:
    primary_scores=consensus3_test; primary_label="consensus3_fallback"

print("V15 DECISION",{"strict_champion":round(float(champ_u.mean()),6),"strict_surgical":round(float(surg_u.mean()),6),
      "increment":round(strict_increment,6),"ci":[round(x,6) for x in strict_ci],"fold_wins":fold_wins,
      "worst_fold":round(worst_fold,6),"quarter_wins":quarter_wins,"tail":round(tail_increment,6),
      "future":round(future_increment,6),"vote3_actions":len(actions3),"vote4_actions":len(actions4),
      "gate":bool(promotion_gate),"primary":primary_label})

def save_submission(scores,name):
    root=Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('/mnt/data')
    out=sample.copy(); out.loc[:,MODULES]=np.asarray(scores,np.float32)
    path=root/name; out.to_csv(path,index=False); chk=pd.read_csv(path)
    assert chk.shape==sample.shape and chk.user_id.is_unique and chk[MODULES].notna().all().all()
    assert chk.user_id.tolist()==sample.user_id.tolist(); print("saved",path); return path

root=Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('/mnt/data')
p_ref=save_submission(consensus3_test,'submission_v15_consensus3_reference.csv')
p_v3=save_submission(vote3_test,'submission_v15_vote3_surgical.csv')
p_v4=save_submission(vote4_test,'submission_v15_vote4_surgical.csv')
p_primary=save_submission(primary_scores,'submission_v15_primary.csv')

pd.DataFrame(fold_records).to_csv(root/'v15_fold_diagnostics.csv',index=False)
pd.DataFrame(threshold_records).to_csv(root/'v15_threshold_diagnostics.csv',index=False)
pd.DataFrame(actions3).to_csv(root/'v15_test_actions_vote3.csv',index=False)
pd.DataFrame(actions4).to_csv(root/'v15_test_actions_vote4.csv',index=False)

summary={
    "public_champion_lb":PUBLIC_CHAMPION_LB,"public_v13_1_lb":PUBLIC_V131_LB,
    "strict_consensus3":float(champ_u.mean()),"strict_surgical":float(surg_u.mean()),"strict_increment":strict_increment,
    "bootstrap_ci":[float(x) for x in strict_ci],"fold_deltas":fold_deltas,"fold_wins":fold_wins,"worst_fold":worst_fold,
    "quarter_deltas":quarter_deltas,"quarter_wins":quarter_wins,"tail_increment":tail_increment,
    "oof_route_rate":float(strict_route.mean()),"oof_routes":int(strict_route.sum()),
    "future_increment":future_increment,"future_ci":[float(x) for x in future_ci],
    "vote3_actions":len(actions3),"vote4_actions":len(actions4),"vote3_geometry":geom3,"vote4_geometry":geom4,
    "promotion_gate":bool(promotion_gate),"primary_label":primary_label,"primary_file":p_primary.name,
    "policy":{"min_drop_rank":MIN_DROP_POS+1,"max_add_rank":MAX_ADD_RANK,"min_action_support":MIN_ACTION_SUPPORT,
              "utility_thresholds":[float(x) for x in UTILITY_THRESHOLDS],"max_inner_route_rate":MAX_INNER_ROUTE_RATE,
              "vote_primary":VOTE_MIN_PRIMARY,"vote_alt":VOTE_MIN_ALT}
}
json.dump(summary,open(root/'v15_summary.json','w'),indent=2)
np.savez_compressed(root/'v15_oof_artifacts.npz',y=y,ref_fold_id=ref_fold_id,
                    strict_champion=strict_champion,strict_surgical=strict_surgical,user_delta=user_delta,
                    strict_route=strict_route,strict_expert=strict_expert,strict_drop=strict_drop,strict_add=strict_add,
                    strict_pred_utility=strict_pred_utility,consensus3_test=consensus3_test,
                    vote3_test=vote3_test,vote4_test=vote4_test)

# ---- cell 24 (markdown) ----
# ## Hard invariants
#
# These checks enforce the exact failure-mode fixes: no whole-user expert switch, no order-only action, no top-1/top-2 mutation, at most one score swap per routed row, and exact Consensus3 fallback semantics.

# ---- cell 25 (code) ----
assert len(MODULES)==17 and y.shape==(N_TRAIN,17) and consensus3_test.shape==(N_TEST,17)
assert strict_champion.shape==strict_surgical.shape==y.shape
assert np.isfinite(strict_champion).all() and np.isfinite(strict_surgical).all()
assert np.isfinite(vote3_test).all() and np.isfinite(vote4_test).all()
# A surgical row only permutes two existing score values; entire row score multiset is unchanged.
for i in np.where(strict_route)[0]:
    assert np.allclose(np.sort(strict_champion[i]),np.sort(strict_surgical[i]),atol=2e-7,rtol=0)
    diff=np.where(np.abs(strict_champion[i]-strict_surgical[i])>2e-7)[0]
    assert len(diff)==2
    assert strict_drop[i] in diff and strict_add[i] in diff
# Unrouted outer-validation users are bitwise exact champion.
assert np.allclose(strict_surgical[~strict_route],strict_champion[~strict_route],atol=0,rtol=0)
# High-value ranks 1-2 are immutable by construction.
assert np.all(np.argmax(strict_surgical,axis=1)==np.argmax(strict_champion,axis=1))
# Deployment outputs obey the same one-swap geometry and preserve top-1.
for cand in (vote3_test,vote4_test):
    assert np.all(np.argmax(cand,axis=1)==np.argmax(consensus3_test,axis=1))
    for i in np.where(np.any(np.abs(cand-consensus3_test)>2e-7,axis=1))[0]:
        assert np.allclose(np.sort(cand[i]),np.sort(consensus3_test[i]),atol=2e-7,rtol=0)
        assert np.sum(np.abs(cand[i]-consensus3_test[i])>2e-7)==2
# Primary semantics.
root=Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('/mnt/data')
primary=pd.read_csv(root/'submission_v15_primary.csv')[MODULES].to_numpy(np.float32)
assert np.allclose(primary,primary_scores,atol=2e-7,rtol=0)
if primary_label=="consensus3_fallback": assert np.allclose(primary,consensus3_test,atol=2e-7,rtol=0)
print("V15 hard invariants PASS")
