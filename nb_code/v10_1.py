# Converted from notebooks/v10_1.ipynb: markdown cells become comments, %/! magics neutralized.

# ---- cell 0 (markdown) ----
# # MineToday V10.1 — Pairwise Rank-5/6 Boundary Surgeon
#
# **Goal:** beat the public `cal045` anchor (LB **0.65639**) without repeating the V10 failure mode.
#
# V10 showed that globally adding `P(top-5)` and utility scores damages ranks that were already correct. V10.1 therefore makes **no global score correction at all**. It treats `cal045` as immutable and learns only one decision:
#
# > **Should the current cal045 rank-6 module replace the current cal045 rank-5 module?**
#
# The design is intentionally surgical:
#
# - exact **word + char TF-IDF + Ridge** representation is unchanged;
# - exact public-validated **seed-42 per-module relevance calibration, w=0.45** remains the anchor;
# - the V9 ordinal family is used only as **meta-evidence** for the boundary model, never added globally;
# - a single global pairwise classifier predicts whether rank-6 has larger true gain than rank-5;
# - a companion pairwise regressor predicts the **actual NDCG@5 delta** of that swap and acts as a veto/ranking signal;
# - the deployment action is a literal **score swap between anchor ranks 5 and 6**, so ranks 1–4 and the entire row score distribution remain unchanged;
# - a quantile-based swap budget is selected from nested OOF evidence, avoiding probability-calibration drift and hard-capping Top-5 membership changes at **8%**;
# - promotion is based only on **strict untouched outer folds**.
#
# No external data, LLM prediction, pretrained model, or paid model is used.
#
# ### Kaggle runtime
# Compatible with Kaggle **T4 ×2**. The pipeline is intentionally CPU-based because the sparse supervised family has consistently outperformed dense semantic models for this synthetic/template-heavy dataset. The GPUs may remain idle.
#
# ### Output policy
# The notebook writes:
#
# - `submission_v10_1_candidate.csv` — actual pairwise V10.1 candidate;
# - `submission_v10_1_primary.csv` — candidate only when the strict promotion gate passes, otherwise exact cal045 fallback;
# - `submission_v10_1_cal045_reference.csv` — immutable public anchor;
# - `v10_1_summary.json` — nested score, fold stability, bootstrap CI, swap-budget diagnostics, and deployment rule;
# - `v10_1_oof_artifacts.npz` — strict OOF evidence for post-run analysis.
#
# **Do not submit before reading the final `V10.1 DECISION` block.**

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

SEED=42
V101_SMOKE=bool(int(os.getenv("V101_SMOKE","0")))
V10_SMOKE=V101_SMOKE
V91_SMOKE=V101_SMOKE
N_FOLDS=2 if V101_SMOKE else 5
N_INNER=2 if V101_SMOKE else 3
RIDGE_ALPHA=3.0
REFERENCE_W=0.45
DIVERSITY_SEEDS=[137] if V101_SMOKE else [137,2026,7,31415]
ORDINAL_WEIGHT=0.10
LGB_ESTIMATORS=12 if V101_SMOKE else 1200
LGB_EARLY_STOP=3 if V101_SMOKE else 80
ORD_ESTIMATORS=8 if V101_SMOKE else 300
ORD_EARLY_STOP=3 if V101_SMOKE else 40
META_SCALE_GRID=np.asarray([1.0,1.45] if V101_SMOKE else [1.35,1.45,1.55],dtype=np.float32)
BOOTSTRAPS=300 if V101_SMOKE else 5000
random.seed(SEED); np.random.seed(SEED)
print("versions",np.__version__,pd.__version__,lgb.__version__,"V101_SMOKE",V101_SMOKE)

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
def exp_rel(prel,P):
    p5=np.minimum(P[5],prel); p4=np.minimum(P[4],p5); p3=np.minimum(P[3],p4); p2=np.minimum(P[2],p3); p1=np.minimum(P[1],p2)
    return (.25*prel+.15*(p5+p4+p3+p2+p1)).astype(np.float32)
if V10_SMOKE:
    ord_oof={t:anchor_meta_oof.copy() for t in range(1,6)}; ord_test={t:anchor_meta_test.copy() for t in range(1,6)}
    ordexp_oof=anchor_meta_oof.copy(); ordexp_test=anchor_meta_test.copy()
else:
    ord_oof={}; ord_test={}
    for t in range(1,6): ord_oof[t],ord_test[t]=fit_ord((RANK_POS<=t).astype(np.int32),t)
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
# ## V10.1 hypothesis and pre-registered policy
#
# V10's direct additive heads were informative but miscalibrated relative to the champion score scale. V10.1 converts the same idea into a **pairwise local decision**. Only the anchor modules currently sitting at ranks 5 and 6 can exchange positions.
#
# The training target is exactly the local metric consequence of that swap:
#
# `delta_56 = (relevance(rank6) - relevance(rank5)) * discount(rank5) / IDCG@5(user)`
#
# The classifier predicts the sign of this delta. The regressor predicts its magnitude. Selection is by **swap-rate quantile**, not a raw probability threshold.

# ---- cell 13 (code) ----
# Pairwise V10.1 policy. Keep the search tiny and pre-registered.
PAIR_Q_GRID = np.asarray([0.00,0.02,0.03,0.04,0.05,0.06,0.07,0.08] if not V101_SMOKE else [0.00,0.05,0.08], dtype=np.float32)
PAIR_VARIANTS = ("classifier", "consensus", "joint")
PAIR_SELECTION_SLACK = 0.00005
PAIR_CLF_TREES = 30 if V101_SMOKE else 450
PAIR_REG_TREES = 30 if V101_SMOKE else 450
PAIR_CV_FOLDS = 2 if V101_SMOKE else 3

PUBLIC_ANCHOR_LB = 0.65639
LEADER_LB = 0.65805
REQUIRED_PUBLIC_DELTA = LEADER_LB - PUBLIC_ANCHOR_LB

# Must exceed the public gap with a little buffer before spending a submission.
V101_MIN_NESTED_DELTA = 0.0001 if V101_SMOKE else 0.00180
V101_MIN_OUTER_WINS = 1 if V101_SMOKE else 4
V101_MAX_WORST_OUTER_DELTA = -0.01 if V101_SMOKE else 0.0
V101_MAX_TEST_TOP5_CHANGE = 0.08
V101_MAX_TEST_TOP1_CHANGE = 0.0

IDEAL_TOP5 = np.sort(y, axis=1)[:, -5:][:, ::-1]
IDCG5 = (IDEAL_TOP5 * DISCOUNT).sum(1).astype(np.float32)

print("required public delta", round(REQUIRED_PUBLIC_DELTA,6))
print("pair q grid", PAIR_Q_GRID.tolist())
print("variants", PAIR_VARIANTS)

# ---- cell 14 (markdown) ----
# ## Pairwise boundary features
#
# The pair model sees only information available before the final decision: cal045/base/relevance/ordinal scores, V9 residual evidence, anchor confidence margins, module identities, and the existing structured user features. It never receives true module relevance as a feature.
#
# The action itself remains distribution-preserving: the two existing anchor scores are swapped, not rescaled or shifted.

# ---- cell 15 (code) ----
def boundary_indices(anchor):
    order = np.argsort(-anchor, axis=1)
    return order[:,4].astype(np.int16), order[:,5].astype(np.int16), order

def _take(mat, idx):
    rows = np.arange(len(mat))
    return np.asarray(mat[rows, idx], np.float32)

def build_pair_features(anchor, base, relprob, v9, p5ord, user_features):
    anchor=np.asarray(anchor,np.float32); base=np.asarray(base,np.float32); relprob=np.asarray(relprob,np.float32)
    v9=np.asarray(v9,np.float32); p5ord=np.asarray(p5ord,np.float32); uf=np.asarray(user_features,np.float32)
    i5,i6,order = boundary_indices(anchor)
    rows=np.arange(len(anchor))
    blocks=[]

    # Candidate-local values and oriented rank6-rank5 differences.
    for mat in (anchor, base, relprob, v9, p5ord):
        a=_take(mat,i5); b=_take(mat,i6)
        blocks += [a[:,None], b[:,None], (b-a)[:,None], np.abs(b-a)[:,None]]

    # Explicit ordinal residual evidence. V10.1 never adds it to global scores.
    resid=(v9-anchor).astype(np.float32)
    ra=_take(resid,i5); rb=_take(resid,i6)
    blocks += [ra[:,None], rb[:,None], (rb-ra)[:,None]]

    # Champion confidence geometry around the boundary.
    s4=anchor[rows,order[:,3]]; s5=anchor[rows,order[:,4]]; s6=anchor[rows,order[:,5]]; s7=anchor[rows,order[:,6]]
    row_std=anchor.std(1); row_rng=anchor.max(1)-anchor.min(1)
    geom=np.column_stack([
        s4-s5, s5-s6, s6-s7, s4-s6, s5-s7,
        row_std, row_rng,
        (s5-s6)/np.maximum(row_std,1e-6),
    ]).astype(np.float32)
    blocks.append(geom)

    # Module identity lets one compact model learn module-specific boundary biases.
    oh5=np.eye(N_MODULES,dtype=np.float32)[i5]
    oh6=np.eye(N_MODULES,dtype=np.float32)[i6]
    blocks += [oh5,oh6,uf]

    Xp=np.concatenate(blocks,axis=1).astype(np.float32)
    assert np.isfinite(Xp).all()
    return Xp,i5,i6

def pair_metric_delta(ytrue, i5, i6):
    rows=np.arange(len(ytrue))
    ideal=np.sort(ytrue,axis=1)[:,-5:][:,::-1]
    idcg=(ideal*DISCOUNT).sum(1)
    return (((ytrue[rows,i6]-ytrue[rows,i5])*DISCOUNT[4]) / np.maximum(idcg,1e-12)).astype(np.float32)

def make_pair_clf(seed):
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=PAIR_CLF_TREES,
        learning_rate=.025,
        num_leaves=11,
        max_depth=4,
        min_child_samples=80,
        reg_alpha=2.0,
        reg_lambda=12.0,
        colsample_bytree=.80,
        subsample=.90,
        subsample_freq=1,
        random_state=int(seed),
        n_jobs=4,
        verbosity=-1,
    )

def make_pair_reg(seed):
    return lgb.LGBMRegressor(
        objective="huber",
        alpha=.90,
        n_estimators=PAIR_REG_TREES,
        learning_rate=.025,
        num_leaves=11,
        max_depth=4,
        min_child_samples=80,
        reg_alpha=2.0,
        reg_lambda=12.0,
        colsample_bytree=.80,
        subsample=.90,
        subsample_freq=1,
        random_state=int(seed),
        n_jobs=4,
        verbosity=-1,
    )

def pair_fit_predict(Xtr, delta_tr, Xout, seed):
    # Ties have zero metric consequence and are excluded from sign-classifier fitting.
    useful=np.abs(delta_tr)>1e-10
    ysign=(delta_tr>0).astype(np.int32)
    if useful.sum()<50 or np.unique(ysign[useful]).size<2:
        raise RuntimeError("Pairwise training set became degenerate.")
    w=np.abs(delta_tr[useful]).astype(np.float64)
    w=w/np.maximum(w.mean(),1e-12)
    w=np.clip(w,0.20,8.0)

    clf=make_pair_clf(seed)
    clf.fit(Xtr[useful],ysign[useful],sample_weight=w)
    prob=clf.predict_proba(Xout)[:,1].astype(np.float32)

    # Scale by 1e3 purely for regression numerics; sign is invariant.
    reg=make_pair_reg(seed+100003)
    rw=1.0+4.0*np.minimum(np.abs(delta_tr)/np.maximum(np.quantile(np.abs(delta_tr),.90),1e-8),1.0)
    reg.fit(Xtr,(delta_tr*1000.0).astype(np.float32),sample_weight=rw)
    pred=(reg.predict(Xout)/1000.0).astype(np.float32)
    return prob,pred

def pair_crossfit(Xp, delta, split_strata, seed):
    prob=np.zeros(len(Xp),np.float32); pred=np.zeros(len(Xp),np.float32)
    skf=StratifiedKFold(n_splits=PAIR_CV_FOLDS,shuffle=True,random_state=int(seed))
    for f,(tr,va) in enumerate(skf.split(np.zeros(len(Xp)),split_strata)):
        prob[va],pred[va]=pair_fit_predict(Xp[tr],delta[tr],Xp[va],seed+7919*f)
    return prob,pred

def swap_mask_from_rule(prob,pred,variant,q):
    prob=np.asarray(prob); pred=np.asarray(pred); n=len(prob)
    if variant=="classifier":
        eligible=prob>0.50; confidence=prob
    elif variant=="consensus":
        eligible=(prob>0.50)&(pred>0); confidence=prob
    elif variant=="joint":
        eligible=(prob>0.50)&(pred>0)
        scale=max(float(np.median(np.abs(pred[eligible]))) if eligible.any() else 0.0,1e-8)
        confidence=(prob-0.50)*np.clip(pred/scale,0,4)
    else:
        raise ValueError(variant)

    mask=np.zeros(n,dtype=bool)
    k=min(int(np.floor(float(q)*n+1e-9)),int(eligible.sum()))
    if k<=0: return mask
    cand=np.flatnonzero(eligible)
    take=cand[np.argpartition(-confidence[cand],k-1)[:k]]
    mask[take]=True
    return mask

def apply_boundary_swap(anchor,mask):
    out=np.asarray(anchor,np.float32).copy(); i5,i6,_=boundary_indices(anchor); rows=np.flatnonzero(mask)
    if len(rows):
        a=i5[rows]; b=i6[rows]
        tmp=out[rows,a].copy(); out[rows,a]=out[rows,b]; out[rows,b]=tmp
    return out

def select_pair_rule(ytrue,anchor,prob,pred):
    trials=[]
    for variant in PAIR_VARIANTS:
        for q in PAIR_Q_GRID:
            mask=swap_mask_from_rule(prob,pred,variant,float(q))
            score=ndcg5(ytrue,apply_boundary_swap(anchor,mask))
            trials.append((float(score),variant,float(q),float(mask.mean())))
    best=max(x[0] for x in trials)
    # Conservative one-near-best rule: among essentially tied choices, make fewer changes.
    near=[x for x in trials if x[0]>=best-PAIR_SELECTION_SLACK]
    chosen=sorted(near,key=lambda x:(x[3],-x[0],x[1]))[0]
    return {"score":chosen[0],"variant":chosen[1],"q":chosen[2],"swap_rate":chosen[3],"best_score":best,"trials":trials}

def boot_ci(delta):
    delta=np.asarray(delta,float); rng=np.random.default_rng(20260823); n=len(delta); z=np.empty(BOOTSTRAPS)
    for b in range(BOOTSTRAPS): z[b]=delta[rng.integers(0,n,n)].mean()
    return tuple(np.quantile(z,[.025,.975]))

# ---- cell 16 (markdown) ----
# ## Strict nested validation
#
# Each outer validation fold is untouched by the sparse backbone, relevance calibration, ordinal model, pairwise classifier, pairwise regressor, and swap-budget selection. The inner data chooses the V10.1 boundary rule; the outer fold measures whether that rule actually transfers.

# ---- cell 17 (code) ----
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

    if V101_SMOKE:
        return ao,av,ao.copy(),av.copy(),ao.copy(),av.copy(),ao.copy(),av.copy()

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
    return ao,av,do,dv,oo,ov,probs_o[5],probs_v[5]

# ---- cell 18 (code) ----
# Strict outer evidence.
nest_anchor=np.zeros_like(y,np.float32)
nest_base=np.zeros_like(y,np.float32)
nest_relprob=np.zeros_like(y,np.float32)
nest_v9=np.zeros_like(y,np.float32)
nest_p5ord=np.zeros_like(y,np.float32)
nest_pair_prob=np.zeros(N_TRAIN,np.float32)
nest_pair_delta_pred=np.zeros(N_TRAIN,np.float32)
nest_candidate=np.zeros_like(y,np.float32)
nest_swap_mask=np.zeros(N_TRAIN,dtype=bool)
outer_deltas=[]; outer_rules=[]; outer_swap_rates=[]

outer_run_folds=ref_folds[:1] if V101_SMOKE else ref_folds
for f,(trg,vag) in enumerate(outer_run_folds):
    print("\n--- V10.1 strict outer fold",f+1,"---")
    inn=inner_folds(trg)
    btr,bva=nested_base(trg,vag,inn)
    ao,av,do,dv,oo,ov,p5o,p5v=nested_family(trg,vag,btr,bva,inn)

    atr=(.55*btr+.45*ao).astype(np.float32)
    ava=(.55*bva+.45*av).astype(np.float32)
    dtr=(.55*btr+.45*do).astype(np.float32)
    dva=(.55*bva+.45*dv).astype(np.float32)
    v9tr=((1-ORDINAL_WEIGHT)*dtr+ORDINAL_WEIGHT*oo).astype(np.float32)
    v9va=((1-ORDINAL_WEIGHT)*dva+ORDINAL_WEIGHT*ov).astype(np.float32)

    Ftr,i5tr,i6tr=build_pair_features(atr,btr,ao,v9tr,p5o,USER_FEATURES[trg])
    Fva,i5va,i6va=build_pair_features(ava,bva,av,v9va,p5v,USER_FEATURES[vag])
    dtr_true=pair_metric_delta(y[trg],i5tr,i6tr)

    # Second-stage cross-fit only for selecting the local boundary rule.
    p_oof,d_oof=pair_crossfit(Ftr,dtr_true,strata[trg],SEED+30000+f*1000)
    rule=select_pair_rule(y[trg],atr,p_oof,d_oof)

    p_va,d_va=pair_fit_predict(Ftr,dtr_true,Fva,SEED+40000+f*1000)
    mask_va=swap_mask_from_rule(p_va,d_va,rule["variant"],rule["q"])
    cand_va=apply_boundary_swap(ava,mask_va)

    nest_anchor[vag]=ava; nest_base[vag]=bva; nest_relprob[vag]=av; nest_v9[vag]=v9va; nest_p5ord[vag]=p5v
    nest_pair_prob[vag]=p_va; nest_pair_delta_pred[vag]=d_va; nest_candidate[vag]=cand_va; nest_swap_mask[vag]=mask_va

    anchor_score=ndcg5(y[vag],ava); cand_score=ndcg5(y[vag],cand_va); delta=cand_score-anchor_score
    outer_deltas.append(float(delta)); outer_rules.append({k:v for k,v in rule.items() if k!="trials"}); outer_swap_rates.append(float(mask_va.mean()))
    print("rule",rule["variant"],"q",rule["q"],"inner swap",round(rule["swap_rate"],4))
    print("anchor",round(anchor_score,6),"candidate",round(cand_score,6),"delta",round(delta,6),"outer swap",round(mask_va.mean(),4))
    gc.collect()

# ---- cell 19 (markdown) ----
# ## Final deployment model and promotion gate
#
# The promotion score below uses the **fold-specific rules chosen inside each outer training partition**. Only after that unbiased estimate is recorded do we choose one final quantile rule from strict OOF pair predictions and fit the deployment pair model on all strict OOF meta-features.

# ---- cell 20 (code) ----
nested_anchor_score=ndcg5(y,nest_anchor)
nested_candidate_score=ndcg5(y,nest_candidate)
nested_delta=float(nested_candidate_score-nested_anchor_score)
user_delta=per_user_ndcg(y,nest_candidate)-per_user_ndcg(y,nest_anchor)
ci=boot_ci(user_delta)
wins=int(sum(d>0 for d in outer_deltas)); worst=float(min(outer_deltas))
strict_swap_rate=float(nest_swap_mask.mean())

# Diagnostics on the actual swaps made by untouched outer models.
i5n,i6n,_=boundary_indices(nest_anchor)
true_pair_delta=pair_metric_delta(y,i5n,i6n)
if nest_swap_mask.any():
    swap_precision=float(np.mean(true_pair_delta[nest_swap_mask]>0))
    harmful_swap_rate=float(np.mean(true_pair_delta[nest_swap_mask]<0))
    mean_true_swap_delta=float(true_pair_delta[nest_swap_mask].mean())
else:
    swap_precision=harmful_swap_rate=mean_true_swap_delta=0.0

# Final rule selection uses strict outer pair predictions, not optimistic ordinary OOF heads.
final_rule=select_pair_rule(y,nest_anchor,nest_pair_prob,nest_pair_delta_pred)

# Train deployment pair model on strict OOF meta-features; infer on full-data champion features.
Ffull,i5full,i6full=build_pair_features(nest_anchor,nest_base,nest_relprob,nest_v9,nest_p5ord,USER_FEATURES[:N_TRAIN])
dfull=pair_metric_delta(y,i5full,i6full)
Ftest,i5test,i6test=build_pair_features(cal045_test,base_test,anchor_meta_test,v9_test,ord_test[5],USER_FEATURES[N_TRAIN:])
final_prob,final_delta_pred=pair_fit_predict(Ffull,dfull,Ftest,SEED+77777)
final_mask=swap_mask_from_rule(final_prob,final_delta_pred,final_rule["variant"],final_rule["q"])
candidate_test=apply_boundary_swap(cal045_test,final_mask)

test_top5_change=top5_set_change_rate(candidate_test,cal045_test)
test_top1_change=top1_change_rate(candidate_test,cal045_test)
expected_lb=PUBLIC_ANCHOR_LB+nested_delta

# By construction only anchor rank5/rank6 can exchange positions.
anchor_top=top5_indices(cal045_test); cand_top=top5_indices(candidate_test)
top4_unchanged=float(np.mean(np.all(anchor_top[:,:4]==cand_top[:,:4],axis=1)))

promotion_gate=bool(
    nested_delta>=V101_MIN_NESTED_DELTA and
    expected_lb>LEADER_LB and
    wins>=V101_MIN_OUTER_WINS and
    worst>=V101_MAX_WORST_OUTER_DELTA and
    ci[0]>0 and
    test_top5_change<=V101_MAX_TEST_TOP5_CHANGE+1e-12 and
    test_top1_change<=V101_MAX_TEST_TOP1_CHANGE+1e-12 and
    top4_unchanged==1.0
)
primary_test=candidate_test if promotion_gate else cal045_test

print("\n================ V10.1 DECISION ================")
print("nested cal045     :",round(nested_anchor_score,6))
print("nested V10.1      :",round(nested_candidate_score,6))
print("nested delta      :",round(nested_delta,6))
print("expected LB 1:1   :",round(expected_lb,6),"leader",LEADER_LB)
print("bootstrap 95% CI  :",tuple(round(float(x),6) for x in ci))
print("outer deltas      :",[round(float(x),6) for x in outer_deltas],"wins",wins,"worst",round(worst,6))
print("strict swap rate  :",round(strict_swap_rate,4),"swap precision",round(swap_precision,4),"harmful",round(harmful_swap_rate,4))
print("final rule        :",final_rule["variant"],"q",final_rule["q"],"test swaps",round(final_mask.mean(),4))
print("test top5/top1 chg:",round(test_top5_change,4),round(test_top1_change,4),"top4 unchanged",top4_unchanged)
print("promotion gate    :",promotion_gate)
print("ACTION            :", "SUBMIT submission_v10_1_primary.csv" if promotion_gate else "DO NOT SUBMIT V10.1 PRIMARY; it falls back to cal045")
print("==================================================")

# ---- cell 21 (code) ----
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
if not V101_SMOKE:
    p_ref=save_submission(cal045_test,'submission_v10_1_cal045_reference.csv')
    p_cand=save_submission(candidate_test,'submission_v10_1_candidate.csv')
    p_primary=save_submission(primary_test,'submission_v10_1_primary.csv')

    np.savez_compressed(
        root/'v10_1_oof_artifacts.npz',
        y=y,
        nest_anchor=nest_anchor,
        nest_base=nest_base,
        nest_relprob=nest_relprob,
        nest_v9=nest_v9,
        nest_p5ord=nest_p5ord,
        nest_pair_prob=nest_pair_prob,
        nest_pair_delta_pred=nest_pair_delta_pred,
        nest_candidate=nest_candidate,
        nest_swap_mask=nest_swap_mask.astype(np.int8),
        true_pair_delta=true_pair_delta,
        final_prob=final_prob,
        final_delta_pred=final_delta_pred,
        final_mask=final_mask.astype(np.int8),
    )

    payload={
        "public_anchor_lb":PUBLIC_ANCHOR_LB,
        "leader_lb_at_design_time":LEADER_LB,
        "required_public_delta":REQUIRED_PUBLIC_DELTA,
        "nested_anchor":float(nested_anchor_score),
        "nested_candidate":float(nested_candidate_score),
        "nested_delta":float(nested_delta),
        "expected_lb_if_1to1_transfer":float(expected_lb),
        "bootstrap_ci":[float(ci[0]),float(ci[1])],
        "outer_deltas":[float(x) for x in outer_deltas],
        "outer_wins":wins,
        "worst_outer_delta":worst,
        "outer_rules":outer_rules,
        "outer_swap_rates":[float(x) for x in outer_swap_rates],
        "strict_swap_rate":strict_swap_rate,
        "strict_swap_precision":swap_precision,
        "strict_harmful_swap_rate":harmful_swap_rate,
        "strict_mean_true_swap_delta":mean_true_swap_delta,
        "final_rule":{k:v for k,v in final_rule.items() if k!="trials"},
        "test_swap_rate":float(final_mask.mean()),
        "test_top5_change":float(test_top5_change),
        "test_top1_change":float(test_top1_change),
        "test_top4_unchanged_rate":top4_unchanged,
        "promotion_gate":promotion_gate,
        "primary":p_primary.name,
        "candidate_file":p_cand.name,
    }
    with open(root/'v10_1_summary.json','w') as f: json.dump(payload,f,indent=2)
    print("saved",root/'v10_1_summary.json')

# ---- cell 22 (markdown) ----
# ## Hard invariants
#
# These checks enforce the core V10.1 promise: exact champion reproduction, finite outputs, unchanged score multiset per user, untouched ranks 1–4, zero Top-1 changes, and a hard 8% membership-change budget.

# ---- cell 23 (code) ----
assert len(MODULES)==17 and y.shape==(4000,17) and N_TEST==1000
for arr in [base_oof,base_test,cal045_oof,cal045_test,v9_oof,v9_test,ord_oof[5],ord_test[5],
            nest_anchor,nest_base,nest_relprob,nest_v9,nest_p5ord,nest_candidate,candidate_test,primary_test]:
    assert np.isfinite(arr).all()
assert np.isfinite(nest_pair_prob).all() and np.isfinite(nest_pair_delta_pred).all()
assert np.isfinite(final_prob).all() and np.isfinite(final_delta_pred).all()
assert np.all(np.ptp(cal045_test,axis=1)>0) and np.all(np.ptp(candidate_test,axis=1)>0)

# Literal score swaps preserve every user's multiset of champion scores.
assert np.allclose(np.sort(candidate_test,axis=1),np.sort(cal045_test,axis=1),rtol=0,atol=0)
assert top1_change_rate(candidate_test,cal045_test)==0.0
assert np.all(top5_indices(candidate_test)[:,:4]==top5_indices(cal045_test)[:,:4])
assert top5_set_change_rate(candidate_test,cal045_test)<=V101_MAX_TEST_TOP5_CHANGE+1e-12

if not V101_SMOKE:
    assert abs(ndcg5(y,cal045_oof)-.662646)<.003
    if promotion_gate:
        assert not np.allclose(candidate_test,cal045_test)
        assert nested_delta>=V101_MIN_NESTED_DELTA
        assert expected_lb>LEADER_LB
    else:
        assert np.allclose(primary_test,cal045_test)
print("All V10.1 hard invariants passed.")
