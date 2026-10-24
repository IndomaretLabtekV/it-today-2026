# Converted from notebooks/v9_1.ipynb: markdown cells become comments, %/! magics neutralized.

# ---- cell 0 (markdown) ----
# # MineToday V9.1 — Adaptive-Shrunk Distribution-Preserving Ordinal Calibration
#
# Public anchor: **V3.1/V7 seed-42 full-data relevance calibration, w=0.45 (LB 0.65639)**.
#
# V9 established a clean positive residual over the public anchor on all five strict outer folds, but the nested effect size (+0.000854) was still too small to spend a leaderboard submission. V9.1 therefore **does not add a new learner family**. It keeps the exact V9 distribution-preserving + cumulative-ordinal residual and adds only a low-capacity, fold-safe empirical-Bayes control layer:
#
# 1. learn module support **inside each outer-training set only** from inner-OOF ablation evidence;
# 2. shrink a module's residual to zero only when it has **no positive inner-fold evidence at all**;
# 3. choose one global residual multiplier from the pre-registered tight grid **[1.35, 1.45, 1.55]** using only outer-training inner-OOF predictions;
# 4. apply the chosen mask/scale once to the untouched outer validation fold;
# 5. promote only if strict nested paired evidence clears a materially stronger gate than V9.
#
# The final test model remains a **single full-data seed-42 calibration path** (no foldbag calibration, no cross-version ensemble). If the promotion gate fails, `submission_v91_primary.csv` automatically falls back to the exact cal045 public reference.

# ---- cell 1 (code) ----
import os,re,json,ast,html,hashlib,random,warnings,gc,time,sys,subprocess
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS","4"); os.environ.setdefault("OPENBLAS_NUM_THREADS","4"); os.environ.setdefault("MKL_NUM_THREADS","4")
os.environ["PYTHONHASHSEED"]="42"; os.environ["TOKENIZERS_PARALLELISM"]="false"
import numpy as np, pandas as pd
from scipy.sparse import hstack
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
warnings.filterwarnings("ignore")
try: import lightgbm as lgb
except Exception:
    subprocess.check_call([sys.executable,"-m","pip","install","-q","lightgbm>=4.5,<5"]); import lightgbm as lgb
SEED=42; V91_SMOKE=bool(int(os.getenv("V91_SMOKE","0")))
N_FOLDS=2 if V91_SMOKE else 5; N_INNER=2 if V91_SMOKE else 3; RIDGE_ALPHA=3.0
REFERENCE_W=0.45; DIVERSITY_SEEDS=[137] if V91_SMOKE else [137,2026,7,31415]; ORDINAL_WEIGHT=0.10
LGB_ESTIMATORS=1200; LGB_EARLY_STOP=80; ORD_ESTIMATORS=80 if V91_SMOKE else 300; ORD_EARLY_STOP=15 if V91_SMOKE else 40
META_SCALE_GRID=np.asarray([1.0,1.45] if V91_SMOKE else [1.35,1.45,1.55],dtype=np.float32)
MIN_NESTED_DELTA=0.0001 if V91_SMOKE else 0.0015; MIN_OUTER_WINS=1 if V91_SMOKE else 4; MAX_WORST_OUTER_DELTA=-0.01 if V91_SMOKE else 0.0
MAX_TEST_TOP5_CHANGE=0.08; MAX_TEST_TOP1_CHANGE=0.015; BOOTSTRAPS=300 if V91_SMOKE else 5000
random.seed(SEED); np.random.seed(SEED)
print("versions",np.__version__,pd.__version__,lgb.__version__,"smoke",V91_SMOKE,"scale_grid",META_SCALE_GRID.tolist())

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
# Competition metric + targets
DISCOUNT = 1.0 / np.log2(np.arange(2, 7))

def per_user_ndcg(y_true, scores):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    idx = np.argpartition(-scores, 4, axis=1)[:, :5]
    local_scores = np.take_along_axis(scores, idx, axis=1)
    idx = np.take_along_axis(idx, np.argsort(-local_scores, axis=1), axis=1)
    dcg = (np.take_along_axis(y_true, idx, axis=1) * DISCOUNT).sum(1)
    ideal = np.sort(y_true, axis=1)[:, -5:][:, ::-1]
    idcg = (ideal * DISCOUNT).sum(1)
    return dcg / np.maximum(idcg, 1e-12)

def ndcg5(y_true, scores):
    return float(per_user_ndcg(y_true, scores).mean())

def top5_indices(scores):
    idx = np.argpartition(-scores, 4, axis=1)[:, :5]
    local = np.take_along_axis(scores, idx, axis=1)
    return np.take_along_axis(idx, np.argsort(-local, axis=1), axis=1)

def top5_set_change_rate(a, b):
    ta, tb = top5_indices(a), top5_indices(b)
    return float(np.mean([set(x) != set(z) for x,z in zip(ta,tb)]))

def top1_change_rate(a, b):
    return float(np.mean(np.argmax(a, axis=1) != np.argmax(b, axis=1)))

Y_BIN = (y > 0).astype(np.float32)
Y_TOP = np.zeros_like(y, np.float32)
Y_TOP[np.arange(N_TRAIN), np.argmax(y, axis=1)] = 1.0

strata = np.asarray([MODULES[j] for j in np.argmax(y, axis=1)])
ref_skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
ref_folds = list(ref_skf.split(np.zeros(N_TRAIN), strata))
ref_fold_id = np.full(N_TRAIN, -1, int)
for f, (_, va) in enumerate(ref_folds):
    ref_fold_id[va] = f

print("reference fold sizes:", [len(va) for _,va in ref_folds])

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
word_vec=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_df=.995,sublinear_tf=True,strip_accents="unicode",max_features=(8000 if V91_SMOKE else 50000),dtype=np.float32)
char_vec=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=2,sublinear_tf=True,max_features=(12000 if V91_SMOKE else 60000),dtype=np.float32)
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
def make_cal(seed,n_estimators=None,ordinal=False):
    if ordinal:
        return lgb.LGBMClassifier(n_estimators=int(ORD_ESTIMATORS if n_estimators is None else n_estimators),learning_rate=.025,num_leaves=11,min_child_samples=60,reg_alpha=1.,reg_lambda=8.,colsample_bytree=.8,random_state=int(seed),n_jobs=4,verbosity=-1)
    return lgb.LGBMClassifier(n_estimators=int(LGB_ESTIMATORS if n_estimators is None else n_estimators),learning_rate=.03,num_leaves=15,min_child_samples=50,reg_alpha=.5,reg_lambda=5.,colsample_bytree=.7,random_state=int(seed),n_jobs=4,verbosity=-1)
def qalign(src_train,tgt_train,vals):
    s=np.sort(np.asarray(src_train,float)); t=np.sort(np.asarray(tgt_train,float)); v=np.asarray(vals,float); n=len(s); q=(np.arange(n)+.5)/n
    u,first,cnt=np.unique(s,return_index=True,return_counts=True); qu=(first+(cnt-1)/2+.5)/n
    return np.interp(np.interp(v,u,qu,left=qu[0],right=qu[-1]),q,t,left=t[0],right=t[-1]).astype(np.float32)
def align_mat(src_oof,tgt_oof,out):
    z=np.zeros_like(out,np.float32)
    for m in range(N_MODULES): z[:,m]=qalign(src_oof[:,m],tgt_oof[:,m],out[:,m])
    return z
def top5idx(s):
    i=np.argpartition(-s,4,axis=1)[:,:5]; q=np.take_along_axis(s,i,axis=1); return np.take_along_axis(i,np.argsort(-q,axis=1),axis=1)
def per_ndcg(yt,s):
    i=top5idx(s); dcg=(np.take_along_axis(yt,i,axis=1)*DISCOUNT).sum(1); ideal=np.sort(yt,axis=1)[:,-5:][:,::-1]; return dcg/np.maximum((ideal*DISCOUNT).sum(1),1e-12)
def top5change(a,b):
    x,y5=top5idx(a),top5idx(b); return float(np.mean([set(x[i])!=set(y5[i]) for i in range(len(x))]))
def top1change(a,b): return float(np.mean(np.argmax(a,1)!=np.argmax(b,1)))
def boot(delta):
    delta=np.asarray(delta,float); rng=np.random.default_rng(20260823); n=len(delta); z=np.empty(BOOTSTRAPS)
    for i in range(BOOTSTRAPS): z[i]=delta[rng.integers(0,n,n)].mean()
    return tuple(np.quantile(z,[.025,.975]))

# ---- cell 9 (code) ----
META_TR=[np.column_stack([base_oof[:,m],USER_FEATURES[:N_TRAIN]]).astype(np.float32) for m in range(N_MODULES)]
META_TE=[np.column_stack([base_test[:,m],USER_FEATURES[N_TRAIN:]]).astype(np.float32) for m in range(N_MODULES)]
def fit_bin(seed,folds,Xtr,Xout,targets,fixed_iters=None):
    oo=np.zeros((len(targets),N_MODULES),np.float32); out=np.zeros((Xout[0].shape[0],N_MODULES),np.float32); its={mid:[] for mid in MODULES}
    for m,mid in enumerate(MODULES):
        target=targets[:,m].astype(np.int32)
        for f,(tr,va) in enumerate(folds):
            if fixed_iters is None:
                clf=make_cal(seed+f+31*m); clf.fit(Xtr[m][tr],target[tr],eval_set=[(Xtr[m][va],target[va])],callbacks=[lgb.early_stopping(LGB_EARLY_STOP,verbose=False)]); it=int(clf.best_iteration_ or LGB_ESTIMATORS)
            else:
                it=int(fixed_iters[mid][f]); clf=make_cal(seed+f+31*m,it); clf.fit(Xtr[m][tr],target[tr])
            its[mid].append(it); oo[va,m]=clf.predict_proba(Xtr[m][va])[:,1].astype(np.float32)
        fi=max(20,int(np.median(its[mid]))); full=make_cal(seed+997+31*m,fi); full.fit(Xtr[m],target); out[:,m]=full.predict_proba(Xout[m])[:,1].astype(np.float32)
    return oo,out,its
anchor_meta_oof,anchor_meta_test,anchor_iters=fit_bin(42,ref_folds,META_TR,META_TE,Y_BIN)
cal045_oof=(.55*base_oof+.45*anchor_meta_oof).astype(np.float32); cal045_test=(.55*base_test+.45*anchor_meta_test).astype(np.float32)
AOO=[]; ATE=[]
for s in DIVERSITY_SEEDS:
    oo,te,_=fit_bin(s,ref_folds,META_TR,META_TE,Y_BIN,anchor_iters); AOO.append(align_mat(oo,anchor_meta_oof,oo)); ATE.append(align_mat(oo,anchor_meta_oof,te))
dist_meta_oof=np.mean(np.stack(AOO),0).astype(np.float32); dist_meta_test=np.mean(np.stack(ATE),0).astype(np.float32)
dist_oof=(.55*base_oof+.45*dist_meta_oof).astype(np.float32); dist_test=(.55*base_test+.45*dist_meta_test).astype(np.float32)
print("cal045 diag",round(ndcg5(y,cal045_oof),6),"dist diag",round(ndcg5(y,dist_oof),6),"top5 change",round(top5change(dist_test,cal045_test),4))

# ---- cell 10 (code) ----
RANK_POS=np.full_like(y,99,dtype=np.int16)
for i,row in enumerate(y):
    p=np.flatnonzero(row>0); order=p[np.argsort(-row[p])]
    for r,m in enumerate(order,1): RANK_POS[i,m]=r
def fit_ord(targets,tag):
    oo=np.zeros_like(y,np.float32); te=np.zeros((N_TEST,N_MODULES),np.float32)
    for m in range(N_MODULES):
        target=targets[:,m].astype(np.int32); its=[]
        for f,(tr,va) in enumerate(ref_folds):
            clf=make_cal(SEED+5000+101*tag+f+31*m,ordinal=True); clf.fit(META_TR[m][tr],target[tr],eval_set=[(META_TR[m][va],target[va])],callbacks=[lgb.early_stopping(ORD_EARLY_STOP,verbose=False)]); it=int(clf.best_iteration_ or ORD_ESTIMATORS); its.append(it); oo[va,m]=clf.predict_proba(META_TR[m][va])[:,1]
        fi=max(20,int(np.median(its))); full=make_cal(SEED+8000+101*tag+31*m,fi,ordinal=True); full.fit(META_TR[m],target); te[:,m]=full.predict_proba(META_TE[m])[:,1]
    return oo,te
ord_oof={}; ord_test={}
for t in range(1,6): ord_oof[t],ord_test[t]=fit_ord((RANK_POS<=t).astype(np.int32),t)
def exp_rel(prel,P):
    p5=np.minimum(P[5],prel); p4=np.minimum(P[4],p5); p3=np.minimum(P[3],p4); p2=np.minimum(P[2],p3); p1=np.minimum(P[1],p2)
    return (.25*prel+.15*(p5+p4+p3+p2+p1)).astype(np.float32)
ordexp_oof=exp_rel(anchor_meta_oof,ord_oof); ordexp_test=exp_rel(anchor_meta_test,ord_test)
v9_oof=((1-ORDINAL_WEIGHT)*dist_oof+ORDINAL_WEIGHT*ordexp_oof).astype(np.float32); v9_test=((1-ORDINAL_WEIGHT)*dist_test+ORDINAL_WEIGHT*ordexp_test).astype(np.float32)
print("ordinal diag",round(ndcg5(y,ordexp_oof),6),"V9 diag",round(ndcg5(y,v9_oof),6))

# ---- cell 11 (code) ----
# V9.1 low-capacity empirical-Bayes residual control.
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
    # Conservative zero-only shrinkage: suppress only modules with no positive
    # evidence in any available OOF fold. Everything else retains the shared residual.
    pos=(np.asarray(fold_deltas)>0).sum(axis=0)
    return (pos>=1).astype(np.float32),pos.astype(np.int16)

def select_shrinkage(ytrue,anchor,candidate,support,grid=META_SCALE_GRID):
    residual=(candidate-anchor).astype(np.float32)
    vals=[]
    for scale in grid:
        s=anchor+float(scale)*residual*support.reshape(1,-1)
        vals.append(ndcg5(ytrue,s))
    j=int(np.argmax(vals))
    return float(grid[j]),np.asarray(vals,np.float64)

def apply_shrinkage(anchor,candidate,support,scale):
    return (anchor+float(scale)*(candidate-anchor)*support.reshape(1,-1)).astype(np.float32)

# ---- cell 12 (markdown) ----
# ## Strict nested V9.1 promotion gate
# Every residual-support decision and multiplier choice is learned only from **inner OOF predictions inside the current outer-training set**. The untouched outer validation fold is never used for sparse fitting, calibration fitting, ordinal fitting, module support, or residual-scale selection.

# ---- cell 13 (code) ----
def inner_folds(trg):
    return list(StratifiedKFold(n_splits=N_INNER,shuffle=True,random_state=SEED).split(np.zeros(len(trg)),strata[trg]))

def nested_base(trg,vag,inn):
    n=len(trg); r=np.zeros((n,N_MODULES),np.float32); t=np.zeros_like(r)
    for lt,lv in inn:
        gtr=trg[lt]; gva=trg[lv]
        mr=Ridge(alpha=RIDGE_ALPHA,solver="lsqr",tol=2e-3,max_iter=1000); mt=Ridge(alpha=RIDGE_ALPHA,solver="lsqr",tol=2e-3,max_iter=1000)
        mr.fit(X[gtr],y[gtr]); mt.fit(X[gtr],Y_TOP[gtr]); r[lv]=mr.predict(X[gva]); t[lv]=mt.predict(X[gva])
    mr=Ridge(alpha=RIDGE_ALPHA,solver="lsqr",tol=2e-3,max_iter=1000); mt=Ridge(alpha=RIDGE_ALPHA,solver="lsqr",tol=2e-3,max_iter=1000)
    mr.fit(X[trg],y[trg]); mt.fit(X[trg],Y_TOP[trg]); val=.9*mr.predict(X[vag])+.1*mt.predict(X[vag])
    return (.9*r+.1*t).astype(np.float32),val.astype(np.float32)

def nested_family(trg,vag,btr,bva,inn):
    Xt=[np.column_stack([btr[:,m],USER_FEATURES[trg]]).astype(np.float32) for m in range(N_MODULES)]
    Xv=[np.column_stack([bva[:,m],USER_FEATURES[vag]]).astype(np.float32) for m in range(N_MODULES)]
    targ=Y_BIN[trg]
    ao=np.zeros((len(trg),N_MODULES),np.float32); av=np.zeros((len(vag),N_MODULES),np.float32); its={mid:[] for mid in MODULES}
    for m,mid in enumerate(MODULES):
        yy=targ[:,m]
        for f,(lt,lv) in enumerate(inn):
            clf=make_cal(42+f+31*m); clf.fit(Xt[m][lt],yy[lt],eval_set=[(Xt[m][lv],yy[lv])],callbacks=[lgb.early_stopping(LGB_EARLY_STOP,verbose=False)])
            it=int(clf.best_iteration_ or LGB_ESTIMATORS); its[mid].append(it); ao[lv,m]=clf.predict_proba(Xt[m][lv])[:,1]
        fi=max(20,int(np.median(its[mid]))); full=make_cal(42+997+31*m,fi); full.fit(Xt[m],yy); av[:,m]=full.predict_proba(Xv[m])[:,1]

    # Same-inner-split diversity. Both outer-training OOF and outer-validation predictions
    # are quantile-aligned back to the seed-42 anchor marginals.
    Do=[]; Dv=[]
    for s in DIVERSITY_SEEDS:
        oo=np.zeros_like(ao); vv=np.zeros_like(av)
        for m,mid in enumerate(MODULES):
            yy=targ[:,m]
            for f,(lt,lv) in enumerate(inn):
                it=int(its[mid][f]); clf=make_cal(s+f+31*m,it); clf.fit(Xt[m][lt],yy[lt]); oo[lv,m]=clf.predict_proba(Xt[m][lv])[:,1]
            fi=max(20,int(np.median(its[mid]))); full=make_cal(s+997+31*m,fi); full.fit(Xt[m],yy); vv[:,m]=full.predict_proba(Xv[m])[:,1]
        Do.append(align_mat(oo,ao,oo)); Dv.append(align_mat(oo,ao,vv))
    do=np.mean(np.stack(Do),0).astype(np.float32); dv=np.mean(np.stack(Dv),0).astype(np.float32)

    # Cumulative ordinal expected relevance, with inner OOF predictions for meta-control.
    rp=RANK_POS[trg]; probs_o={}; probs_v={}
    for tt in range(1,6):
        po=np.zeros_like(ao); pv=np.zeros_like(av); yyall=(rp<=tt).astype(np.int32)
        for m in range(N_MODULES):
            oits=[]
            for f,(lt,lv) in enumerate(inn):
                clf=make_cal(SEED+9000+101*tt+f+31*m,ordinal=True)
                clf.fit(Xt[m][lt],yyall[lt,m],eval_set=[(Xt[m][lv],yyall[lv,m])],callbacks=[lgb.early_stopping(ORD_EARLY_STOP,verbose=False)])
                it=int(clf.best_iteration_ or ORD_ESTIMATORS); oits.append(it); po[lv,m]=clf.predict_proba(Xt[m][lv])[:,1]
            fi=max(20,int(np.median(oits))); full=make_cal(SEED+12000+101*tt+31*m,fi,ordinal=True); full.fit(Xt[m],yyall[:,m]); pv[:,m]=full.predict_proba(Xv[m])[:,1]
        probs_o[tt]=po; probs_v[tt]=pv
    oo=exp_rel(ao,probs_o); ov=exp_rel(av,probs_v)
    return ao,av,do,dv,oo,ov

nest_anchor=np.zeros_like(y); nest_v9=np.zeros_like(y); nest_v91=np.zeros_like(y)
outer_v9=[]; outer_v91=[]; outer_scales=[]; outer_support=[]; outer_support_counts=[]
for f,(trg,vag) in enumerate(ref_folds):
    inn=inner_folds(trg); inner_id=fold_ids_from_splits(len(trg),inn)
    btr,bva=nested_base(trg,vag,inn)
    ao,av,do,dv,oo,ov=nested_family(trg,vag,btr,bva,inn)
    atr=(.55*btr+.45*ao).astype(np.float32); ava=(.55*bva+.45*av).astype(np.float32)
    dtr=(.55*btr+.45*do).astype(np.float32); dva=(.55*bva+.45*dv).astype(np.float32)
    ctr=((1-ORDINAL_WEIGHT)*dtr+ORDINAL_WEIGHT*oo).astype(np.float32)
    cva=((1-ORDINAL_WEIGHT)*dva+ORDINAL_WEIGHT*ov).astype(np.float32)

    evid=module_ablation_fold_deltas(y[trg],atr,ctr,inner_id,N_INNER)
    support,pos=module_support(evid)
    scale,scale_scores=select_shrinkage(y[trg],atr,ctr,support)
    s_va=apply_shrinkage(ava,cva,support,scale)

    nest_anchor[vag]=ava; nest_v9[vag]=cva; nest_v91[vag]=s_va
    da=ndcg5(y[vag],ava); d9=ndcg5(y[vag],cva); d91=ndcg5(y[vag],s_va)
    outer_v9.append(d9-da); outer_v91.append(d91-da); outer_scales.append(scale); outer_support.append(support); outer_support_counts.append(pos)
    suppressed=[MODULES[m] for m in range(N_MODULES) if support[m]==0]
    print("outer",f+1,"anchor",round(da,6),"v9",round(d9,6),"v91",round(d91,6),"delta",round(d91-da,6),"scale",scale,"suppressed",suppressed)
    gc.collect()

# ---- cell 14 (code) ----
ref=ndcg5(y,nest_anchor); cand9=ndcg5(y,nest_v9); cand=ndcg5(y,nest_v91); delta=cand-ref
ci=boot(per_ndcg(y,nest_v91)-per_ndcg(y,nest_anchor)); wins=sum(x>0 for x in outer_v91); worst=min(outer_v91)

# Final test control is learned only from strict outer OOF evidence.
outer_evid=module_ablation_fold_deltas(y,nest_anchor,nest_v9,ref_fold_id,N_FOLDS)
final_support,final_pos=module_support(outer_evid)
final_scale,final_scale_scores=select_shrinkage(y,nest_anchor,nest_v9,final_support)
v91_test=apply_shrinkage(cal045_test,v9_test,final_support,final_scale)
ch5=top5change(v91_test,cal045_test); ch1=top1change(v91_test,cal045_test)
gate=(delta>=MIN_NESTED_DELTA and wins>=MIN_OUTER_WINS and worst>=MAX_WORST_OUTER_DELTA and ci[0]>0 and ch5<=MAX_TEST_TOP5_CHANGE and ch1<=MAX_TEST_TOP1_CHANGE)
print("STRICT nested anchor",round(ref,6),"V9",round(cand9,6),"V9.1",round(cand,6),"delta",round(delta,6),"CI",tuple(round(float(x),6) for x in ci),"wins",wins,"worst",round(float(worst),6))
print("V9.1 outer deltas",[round(float(x),6) for x in outer_v91],"outer scales",outer_scales)
print("final scale",final_scale,"support counts",dict(zip(MODULES,final_pos.tolist())),"suppressed",[MODULES[m] for m in range(N_MODULES) if final_support[m]==0])
print("test changes",round(ch5,4),round(ch1,4),"GATE",gate)

# ---- cell 15 (code) ----
def save(scores,name):
    out=sample.copy(); out.loc[:,MODULES]=scores
    root=Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('/mnt/data'); p=root/name; out.to_csv(p,index=False)
    chk=pd.read_csv(p); assert chk.shape==sample.shape and chk.user_id.is_unique and chk[MODULES].notna().all().all(); print('saved',p); return p
root=Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('/mnt/data')
if not V91_SMOKE:
    p_ref=save(cal045_test,'submission_v91_cal045_reference.csv')
    p_v9=save(v9_test,'submission_v91_v9_base.csv')
    p_cand=save(v91_test,'submission_v91_adaptive.csv')
    p_primary=save(v91_test if gate else cal045_test,'submission_v91_primary.csv')
    np.savez_compressed(root/'v91_oof_artifacts.npz',y=y,base_oof=base_oof,cal045_oof=cal045_oof,v9_oof=v9_oof,nest_anchor=nest_anchor,nest_v9=nest_v9,nest_v91=nest_v91,outer_v9=np.asarray(outer_v9),outer_v91=np.asarray(outer_v91),outer_scales=np.asarray(outer_scales),final_support=final_support,final_pos=final_pos,final_scale=np.asarray([final_scale]))
    json.dump({'public_anchor_lb':.65639,'leader':.65805,'nested_ref':float(ref),'nested_v9':float(cand9),'nested_v91':float(cand),'nested_delta':float(delta),'ci':[float(x) for x in ci],'outer_v91':[float(x) for x in outer_v91],'outer_scales':[float(x) for x in outer_scales],'final_scale':float(final_scale),'final_support':{MODULES[m]:float(final_support[m]) for m in range(N_MODULES)},'test_top5_change':float(ch5),'test_top1_change':float(ch1),'gate':bool(gate),'primary':p_primary.name},open(root/'v91_summary.json','w'),indent=2)
print('FINAL',{'nested_ref':round(ref,6),'nested_v91':round(cand,6),'delta':round(delta,6),'wins':wins,'final_scale':final_scale,'test_top5_change':round(ch5,4),'gate':gate})

# ---- cell 16 (code) ----
assert len(MODULES)==17 and y.shape==(4000,17) and N_TEST==1000
for arr in [base_oof,base_test,anchor_meta_oof,anchor_meta_test,cal045_oof,cal045_test,dist_oof,dist_test,ordexp_oof,ordexp_test,v9_oof,v9_test,nest_anchor,nest_v9,nest_v91,v91_test]: assert np.isfinite(arr).all()
assert np.all(np.ptp(cal045_test,axis=1)>0) and np.all(np.ptp(v91_test,axis=1)>0)
assert final_support.shape==(17,) and np.isin(final_support,[0.,1.]).all() and final_scale in set(float(x) for x in META_SCALE_GRID)
if not V91_SMOKE: assert abs(ndcg5(y,cal045_oof)-.662646)<.003
print('All V9.1 hard invariants passed.')

# ---- cell 17 (markdown) ----
# ## Output policy
# `submission_v91_primary.csv` is deliberately conservative: it contains V9.1 only when the **strict nested promotion gate** passes; otherwise it is byte-for-byte score-equivalent to the public `cal045` anchor. Inspect `v91_summary.json` before any leaderboard submission.
