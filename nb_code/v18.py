# Converted from notebooks/v18.ipynb: markdown cells become comments, %/! magics neutralized.

# ---- cell 0 (markdown) ----
# # MineToday V18 — Multi-Intent In-Domain Semantic Prototype Calibration
#
# **Public anchor:** V12.2 `Consensus3` = **0.65856**.  Current leader = **0.66082**.  Required public improvement = **+0.00226**.
#
# V18 is a deliberate architecture change after V16/V17 showed that global user-level semantic fusion is too redundant and too destructive at the Top-5 boundary.
#
# ## Core design
#
# 1. **Consensus3 stays immutable.** The exact public-winning sparse + structured relevance path remains the anchor.
# 2. **Message-level Qwen3-Embedding-8B.** Every non-administrative learning message is embedded separately. Duplicate normalized messages are encoded once and reused.
# 3. **No-chat semantic hard-off.** Users with no usable learning message receive exactly the structured/sparse anchor; V18 never embeds a synthetic "no chat" sentence.
# 4. **Multi-scale Matryoshka views.** Formal module matching is measured at 512 / 1024 / 2048 dimensions after re-normalization.
# 5. **Fold-safe in-domain prototypes.** For each module, only outer-training users may create organizer-language prototypes. Each positive Top-5 user contributes the message most aligned with that module; repeated templates are de-duplicated before compact spherical clustering.
# 6. **Tiny diagonal metric adapter.** A strongly regularized organizer-supervised metric reweights 512 embedding dimensions. Qwen itself remains frozen. No generation and no direct test-label request is used.
# 7. **Second-level semantic calibration.** Binary relevance and Top-5 semantic heads use old anchor predictions + assessment + multi-intent semantic evidence. Their outputs are distribution-aligned back to the old relevance geometry.
# 8. **Representation-specific module trust.** Modules are never hard-coded from V16/V17. Trust is selected inside the same V18 representation with leave-one-fold-out NDCG evidence.
# 9. **Membership changes are allowed, but only through smooth trusted residuals.** This directly attacks the dominant Top-5 membership bottleneck without returning to a hard user router.
# 10. **Strict promotion contract.** Ordinary OOF, fresh-seed fully isolated audit, chronological future audit, fold consistency, bootstrap interval, and test perturbation are all reported before any submission is recommended.
#
# ## Kaggle setup
#
# - Accelerator: **GPU T4 x2**
# - Internet: **ON**
# - Run all cells from top to bottom.
# - Do not change `V18_SMOKE=0` / `V18_SKIP_QWEN=0` for the real run.
#
# The notebook writes two attack files because two submissions remain, but **hidden leaderboard performance cannot be guaranteed before Kaggle evaluates it**. Use the final `V18 DECISION` block rather than the filename alone.

# ---- cell 1 (code) ----
# Runtime dependencies. No external task dataset is downloaded.
import os,sys,subprocess
V18_SKIP_INSTALL=bool(int(os.getenv("V18_SKIP_INSTALL","0")))
if not V18_SKIP_INSTALL:
    subprocess.check_call([
        sys.executable,"-m","pip","install","-q","-U",
        "transformers==4.57.6","accelerate==1.14.0","safetensors>=0.5,<1",
        "huggingface_hub>=0.34,<2","lightgbm>=4.5,<5","sentencepiece>=0.2,<1"
    ])
print("dependency setup complete; skip_install=",V18_SKIP_INSTALL)

# ---- cell 2 (code) ----
import os,re,json,ast,html,hashlib,random,warnings,gc,time,math
from pathlib import Path
from collections import defaultdict,Counter
os.environ.setdefault("OMP_NUM_THREADS","4"); os.environ.setdefault("OPENBLAS_NUM_THREADS","4"); os.environ.setdefault("MKL_NUM_THREADS","4")
os.environ["PYTHONHASHSEED"]="42"; os.environ["TOKENIZERS_PARALLELISM"]="false"; os.environ.setdefault("PYDEVD_DISABLE_FILE_VALIDATION","1")

import numpy as np,pandas as pd
from IPython.display import display
from scipy.sparse import hstack
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge,SGDClassifier
from sklearn.cluster import MiniBatchKMeans
import lightgbm as lgb
warnings.filterwarnings("ignore")

SEED=42
AUDIT_SEED=20260824
V18_SMOKE=bool(int(os.getenv("V18_SMOKE","0")))
V18_SKIP_QWEN=bool(int(os.getenv("V18_SKIP_QWEN","0")))
N_FOLDS=2 if V18_SMOKE else 5
N_INNER=2 if V18_SMOKE else 3
N_AUDIT_OUTER=2 if V18_SMOKE else 3
RIDGE_ALPHA=3.0
REFERENCE_W=0.45
LGB_ESTIMATORS=60 if V18_SMOKE else 1200
LGB_EARLY_STOP=6 if V18_SMOKE else 80
AUDIT_LGB_ESTIMATORS=50 if V18_SMOKE else 700
AUDIT_EARLY_STOP=5 if V18_SMOKE else 50
SEM_LGB_ESTIMATORS=45 if V18_SMOKE else 700
SEM_LGB_EARLY_STOP=5 if V18_SMOKE else 55
BOOTSTRAPS=300 if V18_SMOKE else 5000

PUBLIC_CONSENSUS3_LB=0.65856
PUBLIC_LEADER_LB=0.66082
REQUIRED_PUBLIC_DELTA=PUBLIC_LEADER_LB-PUBLIC_CONSENSUS3_LB

EMBED_MODEL="Qwen/Qwen3-Embedding-8B"
EMBED_REVISION="1d8ad4c"
EMBED_DIM_FULL=4096
RUNTIME_DIM_FULL=96 if V18_SMOKE else EMBED_DIM_FULL
STATIC_DIMS=([32,64] if V18_SMOKE else [512,1024,2048])
ADAPTER_DIM=min(48 if V18_SMOKE else 512,RUNTIME_DIM_FULL)
PROTOTYPE_K=2 if V18_SMOKE else 5
MEMORY_K=4 if V18_SMOKE else 14
MEMORY_TEMP=.08
ADAPTER_MAX_POS=500 if V18_SMOKE else 9000
ADAPTER_ALPHA=0.002 if V18_SMOKE else 0.0008
ADAPTER_METRIC_STRENGTH=0.12
MESSAGE_MAX_TOKENS=192
MODULE_MAX_TOKENS=448
NO_CHAT_SEMANTIC_OFF=True

# Two low-dimensional attack families; selection of modules is nested and representation-specific.
ATTACK_CONFIG={
    "bin35":dict(kind="bin",beta=.35,broad=False),
    "dual35":dict(kind="dual",beta=.35,broad=False),
    "dual50":dict(kind="dual",beta=.50,broad=True),
}
TRUST_PROBE_BETA=.35

# Promotion is intentionally ambitious because the public gap is +0.00226.
MIN_ORD_DELTA=-0.01 if V18_SMOKE else 0.00055
MIN_AUDIT_DELTA=-0.01 if V18_SMOKE else 0.00055
MIN_CHRONO_DELTA=-0.01 if V18_SMOKE else 0.00000
MIN_AUDIT_WINS=1 if V18_SMOKE else 2
MIN_AUDIT_CI_LOW=-1.0 if V18_SMOKE else -0.00015
MIN_TEST_TOP5_CHANGE=0.0 if V18_SMOKE else 0.015
MAX_TEST_TOP5_CHANGE=1.0 if V18_SMOKE else 0.12
MAX_TEST_TOP1_CHANGE=1.0 if V18_SMOKE else 0.055

CACHE_ROOT=(Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('/mnt/data'))/'v18_cache'
CACHE_ROOT.mkdir(parents=True,exist_ok=True)
random.seed(SEED); np.random.seed(SEED)
print("V18_SMOKE",V18_SMOKE,"V18_SKIP_QWEN",V18_SKIP_QWEN)
print("public champion",PUBLIC_CONSENSUS3_LB,"leader",PUBLIC_LEADER_LB,"gap",round(REQUIRED_PUBLIC_DELTA,6))
print("Qwen",EMBED_MODEL,"revision",EMBED_REVISION,"full_dim",RUNTIME_DIM_FULL,"adapter_dim",ADAPTER_DIM)

# ---- cell 3 (code) ----
def locate_data_dir():
    required={"chat_history.csv","modules_catalog.csv","user_assessments.csv","train_relevance.csv","test.csv","sample_submission.csv"}
    roots=[Path("/kaggle/input"),Path("/mnt/data/_mine_data"),Path("/mnt/data/v18_data"),Path("/mnt/data/comp_v3"),Path(".")]
    for root in roots:
        if not root.exists(): continue
        candidates=[root]
        if root.name=="input": candidates += [p for p in root.rglob("*") if p.is_dir()]
        for d in candidates:
            try: names={p.name for p in d.iterdir() if p.is_file()}
            except Exception: continue
            if required.issubset(names): return d
    raise FileNotFoundError("MineToday dataset directory not found. Attach the competition dataset to this Kaggle notebook.")

DATA_DIR=locate_data_dir(); print("DATA_DIR =",DATA_DIR)
chat=pd.read_csv(DATA_DIR/"chat_history.csv")
modules=pd.read_csv(DATA_DIR/"modules_catalog.csv")
assessments=pd.read_csv(DATA_DIR/"user_assessments.csv")
train=pd.read_csv(DATA_DIR/"train_relevance.csv")
test=pd.read_csv(DATA_DIR/"test.csv")
sample=pd.read_csv(DATA_DIR/"sample_submission.csv")

if V18_SMOKE:
    train=train.iloc[:400].copy(); test=test.iloc[:100].copy()
    sample=sample.set_index("user_id").loc[test.user_id].reset_index()
    keep=set(train.user_id)|set(test.user_id)
    chat=chat[chat.user_id.isin(keep)].copy(); assessments=assessments[assessments.user_id.isin(keep)].copy()

MODULES=list(train.columns[1:]); N_MODULES=len(MODULES); N_TRAIN=len(train); N_TEST=len(test)
assert MODULES==list(sample.columns[1:]) and N_MODULES==17
assert (N_TRAIN,N_TEST)==((400,100) if V18_SMOKE else (4000,1000))
assert train.user_id.is_unique and test.user_id.is_unique
all_users=train.user_id.tolist()+test.user_id.tolist(); uid2idx={u:i for i,u in enumerate(all_users)}
y=train[MODULES].to_numpy(np.float32)
print("shapes",chat.shape,modules.shape,assessments.shape,train.shape,test.shape)
print("positive modules/user",pd.Series((y>0).sum(1)).value_counts().sort_index().to_dict())

# ---- cell 4 (code) ----
DISCOUNT=1.0/np.log2(np.arange(2,7))

def per_user_ndcg(y_true,scores):
    y_true=np.asarray(y_true); scores=np.asarray(scores)
    idx=np.argpartition(-scores,4,axis=1)[:,:5]
    loc=np.take_along_axis(scores,idx,axis=1)
    idx=np.take_along_axis(idx,np.argsort(-loc,axis=1),axis=1)
    dcg=(np.take_along_axis(y_true,idx,axis=1)*DISCOUNT).sum(1)
    ideal=np.sort(y_true,axis=1)[:,-5:][:,::-1]
    return dcg/np.maximum((ideal*DISCOUNT).sum(1),1e-12)

def ndcg5(y_true,scores): return float(per_user_ndcg(y_true,scores).mean())
def top5_indices(scores):
    idx=np.argpartition(-scores,4,axis=1)[:,:5]; loc=np.take_along_axis(scores,idx,axis=1)
    return np.take_along_axis(idx,np.argsort(-loc,axis=1),axis=1)
def top5_set_change_rate(a,b):
    ta,tb=top5_indices(a),top5_indices(b); return float(np.mean([set(x)!=set(z) for x,z in zip(ta,tb)]))
def top5_order_change_rate(a,b): return float(np.mean(np.any(top5_indices(a)!=top5_indices(b),axis=1)))
def top1_change_rate(a,b): return float(np.mean(np.argmax(a,axis=1)!=np.argmax(b,axis=1)))

def paired_boot(delta,seed):
    delta=np.asarray(delta,float); rng=np.random.default_rng(seed); n=len(delta); vals=np.empty(BOOTSTRAPS,float)
    for i in range(BOOTSTRAPS): vals[i]=delta[rng.integers(0,n,n)].mean()
    return tuple(float(v) for v in np.quantile(vals,[.025,.975]))

Y_BIN=(y>0).astype(np.float32)
Y_TOP=np.zeros_like(y,np.float32); Y_TOP[np.arange(N_TRAIN),np.argmax(y,axis=1)]=1.0
RANK_POS=np.full_like(y,99,dtype=np.int16)
for i,row in enumerate(y):
    p=np.flatnonzero(row>0); order=p[np.argsort(-row[p])]
    for r,m in enumerate(order,1): RANK_POS[i,m]=r
Y_TOP5=(RANK_POS<=5).astype(np.float32)

strata=np.asarray([MODULES[j] for j in np.argmax(y,axis=1)])
ref_folds=list(StratifiedKFold(N_FOLDS,shuffle=True,random_state=SEED).split(np.zeros(N_TRAIN),strata))
ref_fold_id=np.full(N_TRAIN,-1,int)
for f,(_,va) in enumerate(ref_folds): ref_fold_id[va]=f

def make_local_folds(global_train,n_splits,seed):
    labs=strata[np.asarray(global_train)]
    return list(StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=int(seed)).split(np.zeros(len(global_train)),labs))
print("fold sizes",[len(v) for _,v in ref_folds])

# ---- cell 5 (code) ----
# Exact V3.1 assessment + organizer-derived template/topic structure.
WORD_NUM={"nol":0,"satu":1,"dua":2,"tiga":3,"empat":4,"lima":5}
SKILL_NAMES=["python","sql","statistics","eda","ml","ml_eval","dl","genai","business","research"]
A=np.zeros((len(all_users),10),np.float32); A_INVALID=np.zeros_like(A)
for row in assessments.itertuples(index=False):
    try:d=json.loads(row.assessment_result)
    except Exception:d=ast.literal_eval(row.assessment_result)
    vals=[]; bads=[]
    for v in d.values():
        sv=str(v).strip().lower()
        if sv in WORD_NUM: value,bad=float(WORD_NUM[sv]),0
        else:
            try:value=float(v)
            except Exception:value=np.nan
            bad=int((not np.isfinite(value)) or value<0 or value>5)
            if bad:value=np.nan
        vals.append(value); bads.append(bad)
    vals=np.asarray(vals,float); med=float(np.nanmedian(vals)) if np.isfinite(vals).any() else 2.0
    vals=np.where(np.isfinite(vals),vals,med); i=uid2idx[row.user_id]; A[i]=vals; A_INVALID[i]=bads

TOPIC_PATTERNS={
"excel":r"\b(?:ms\.?\s*excel|excel|pivot(?: table)?|vlookup|hlookup|spreadsheet)\b",
"python":r"\b(?:python|pyton|pithon|pandas(?:\s*&\s*| dan |\s+)?numpy|pandas|numpy)\b",
"sql":r"\b(?:sql|escuel|query database|join tabel|database relasional|window function)\b",
"statistics":r"\b(?:statistik(?:\s*&\s*probabilitas)?|probabilitas|a\s*/?\s*b testing|uji hipotesis|distribusi data)\b",
"eda":r"\b(?:eda|exploratory data analysis|data cleaning|missing values?|matplotlib|seaborn)\b",
"ml_eval":r"\b(?:evaluasi model(?: ml)?|metrik model|metric model|evaluasi performa model)\b",
"mlops":r"\b(?:mlops|deployment model|deploy(?:ment)? model|ci\s*/?\s*cd|monitoring model|serving model|fastapi)\b",
"cv":r"\b(?:computer vision|cnn|deep learning|deteksi objek|klasifikasi gambar|image classification)\b",
"nlp":r"\b(?:natural language processing|nlp|analisis sentimen|sentiment analysis|word2vec|tf\s*-?\s*idf|transformer)\b",
"genai":r"\b(?:generative ai|genai|large language model|llm|rag|chatgpt|chatbot|fine[- ]?tuning model bahasa)\b",
"ml":r"\b(?:machine learning|bikin model ml|model machine learning|regresi\s*(?:&|dan)\s*klasifikasi|clustering|supervised|unsupervised|hyperparameter)\b",
"business":r"\b(?:translate masalah bisnis ke data|menerjemahkan masalah bisnis|business(?:-| )to(?:-| )ds|permasalahan bisnis)\b",
"dashboard":r"\b(?:dashboard|looker studio|tableau|visualisasi interaktif|storytelling data)\b",
"webscrape":r"\b(?:web scrap(?:e|ing)|scraping|beautifulsoup|selenium|ekstraksi data.*website)\b",
"git":r"\b(?:version control|git(?:hub|lab)?|repository|repo kode)\b",
"prompt":r"\b(?:prompt engineering|prompting|merancang prompt|optimasi prompt)\b",
"automation":r"\b(?:ai automation|otomasi|otomatis|automation|workflow|n8n)\b",
"nocode":r"\b(?:no[ -]?code|zero coding|tanpa perlu (?:belajar )?(?:coding|koding)|gausah (?:coding|koding)|gak perlu (?:coding|koding)|tanpa (?:coding|koding))\b",
"career":r"\b(?:portofolio|portfolio|cv|test interview|interview|karir|career|prospek kerja|negosiasi)\b"}
TOPIC_RX={k:re.compile(v,re.I) for k,v in TOPIC_PATTERNS.items()}; TOPICS=list(TOPIC_PATTERNS)
NOISE_RX=re.compile(r"reset password|email verifikasi|sertifikat kelulusan|pembayaran|cicilan|invoice|akun.*(?:error|masalah)|lms.*(?:error|masalah)",re.I)
STYLE_PATTERNS={
"beginner":r"pemula|dari nol|gaptek|belum pernah (?:ngoding|coding|koding)|background(?:nya)? bukan (?:it|data)|banting setir",
"advanced":r"udah lumayan paham|sudah lumayan paham|stuck|prasyarat|syarat|\breq\b|basic apa|lanjutin",
"urgent":r"urgent|secepatnya|sefastnya|intensif|seminggu|minggu depan",
"budget":r"budget|diskon|worth it|cekak|fomo|kompetitor","online":r"online|durasi|kapan buka|mulai minggu|jadwal",
"nocode_style":r"tanpa perlu.*(?:coding|koding)|gausah.*(?:coding|koding)|gak perlu.*(?:coding|koding)|otomatis.*ai|solusi otomatis",
"career_style":r"portofolio|portfolio|test interview|prospek kerja|karir|career"}
STYLE_RX={k:re.compile(v,re.I) for k,v in STYLE_PATTERNS.items()}; STYLES=list(STYLE_PATTERNS)

def clean_text(text):
    text=html.unescape(str(text)).lower(); text=re.sub(r"https?://\S+|www\.\S+"," <url> ",text)
    text=re.sub(r"[^\w\s<>/&+-]"," ",text,flags=re.UNICODE); return re.sub(r"\s+"," ",text).strip()
def extract_topics(text): return [k for k,rx in TOPIC_RX.items() if rx.search(str(text))]
def make_skeleton(text):
    s=clean_text(text)
    for rx in TOPIC_RX.values(): s=rx.sub(" <topic> ",s)
    s=re.sub(r"\b\d+\b"," <num> ",s); return re.sub(r"\s+"," ",s).strip()
def template_token(s): return "tpl_"+hashlib.md5(s.encode("utf-8")).hexdigest()[:10]

chat["timestamp"]=pd.to_datetime(chat["timestamp"],errors="coerce")
chat["clean"]=chat.user_chat_text.fillna("").map(clean_text)
chat["is_noise"]=chat.user_chat_text.fillna("").map(lambda x:bool(NOISE_RX.search(str(x))))
chat["topics"]=chat.user_chat_text.fillna("").map(extract_topics)
chat["skeleton"]=chat.user_chat_text.fillna("").map(make_skeleton)
for name,rx in STYLE_RX.items(): chat["sty_"+name]=chat.user_chat_text.fillna("").map(lambda x,r=rx:int(bool(r.search(str(x)))))
chat=chat.sort_values(["user_id","timestamp","chat_id"]).reset_index(drop=True)
user_chat={u:g.copy() for u,g in chat.groupby("user_id",sort=False)}
USER_FEATURES=np.zeros((len(all_users),10+10+5+len(STYLES)*2+len(TOPICS)*5),np.float32)

def build_user(u):
    ui=uid2idx[u]; g=user_chat.get(u); parts=[]
    if g is None: learn=pd.DataFrame(columns=chat.columns); g=learn
    else:
        learn=g[~g.is_noise].copy()
        if len(learn)==0: learn=g.copy()
    n=len(learn)
    for j,row in enumerate(learn.itertuples(index=False)):
        distance=n-1-j; tag="current" if distance==0 else ("recent" if distance<=2 else "older")
        parts += [f"[{tag}] {row.clean}",template_token(row.skeleton)]
        for tp in row.topics:
            parts.append("topic_"+tp)
            if distance==0: parts.extend(["latest_topic_"+tp]*3)
            elif distance<=2: parts.extend(["recent_topic_"+tp]*2)
        for sty in STYLES:
            if getattr(row,"sty_"+sty): parts.append("style_"+sty)
    for j,skill in enumerate(SKILL_NAMES):
        v=int(round(float(A[ui,j]))); band="low" if v<=1 else ("mid" if v<=3 else "high")
        parts += [f"skill_{skill}_{v}",f"skill_{skill}_bin_{band}"]
    feats=list(A[ui])+list(A_INVALID[ui]); total=len(g); learning_count=len(learn); noise_count=int(g.is_noise.sum()) if total else 0; span=0.0
    if total>=2 and g.timestamp.notna().sum()>=2: span=float((g.timestamp.max()-g.timestamp.min()).total_seconds()/86400.0)
    feats += [total,learning_count,noise_count,span,min(total,6)]
    for sty in STYLES: feats.append(float(learn["sty_"+sty].sum()) if learning_count else 0.0)
    for sty in STYLES: feats.append(float(learn.iloc[-1]["sty_"+sty]) if learning_count else 0.0)
    rows=list(learn.itertuples(index=False)); nfull=len(rows)
    for tp in TOPICS:
        count=rec_count=last_pos=0.0; latest=0.0
        for j,row in enumerate(rows):
            if tp in row.topics:
                count+=1.0; rec_count+=float(np.exp(-0.7*(nfull-1-j))); last_pos=max(last_pos,(j+1)/max(nfull,1))
        if rows and tp in rows[-1].topics: latest=1.0
        feats += [count,rec_count,latest,last_pos,float(count>0)]
    USER_FEATURES[ui]=np.asarray(feats,np.float32); return " ".join(parts)

texts=np.asarray([build_user(u) for u in all_users],dtype=object)
assert USER_FEATURES.shape[1]==134 and np.isfinite(USER_FEATURES).all()
print("USER_FEATURES",USER_FEATURES.shape,"invalid assessment cells",int(A_INVALID.sum()))

# ---- cell 6 (code) ----
# Exact public-validated V3.1 sparse backbone.
t0=time.time()
word_vec=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_df=.995,sublinear_tf=True,strip_accents="unicode",max_features=(3000 if V18_SMOKE else 50000),dtype=np.float32)
char_vec=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=2,sublinear_tf=True,max_features=(5000 if V18_SMOKE else 60000),dtype=np.float32)
X=hstack([word_vec.fit_transform(texts),char_vec.fit_transform(texts)],format="csr")

def sparse_ref(target):
    oo=np.zeros_like(y,np.float32); te=np.zeros((N_TEST,N_MODULES),np.float32)
    for tr,va in ref_folds:
        model=Ridge(alpha=RIDGE_ALPHA,solver="lsqr",tol=2e-3,max_iter=1000)
        model.fit(X[tr],target[tr]); oo[va]=model.predict(X[va]).astype(np.float32); te+=model.predict(X[N_TRAIN:]).astype(np.float32)/len(ref_folds)
    return oo,te
raw_oof,raw_test=sparse_ref(y); top_oof,top_test=sparse_ref(Y_TOP)
base_oof=(.9*raw_oof+.1*top_oof).astype(np.float32); base_test=(.9*raw_test+.1*top_test).astype(np.float32)
print("X",X.shape,"base OOF",round(ndcg5(y,base_oof),6),"sec",round(time.time()-t0,1))

# ---- cell 7 (code) ----
# Exact seed-42 binary relevance path + immutable Consensus3.
def make_cal(seed,n_estimators=None,audit=False):
    n=int((AUDIT_LGB_ESTIMATORS if audit else LGB_ESTIMATORS) if n_estimators is None else n_estimators)
    return lgb.LGBMClassifier(n_estimators=n,learning_rate=.03,num_leaves=15,min_child_samples=50,
        reg_alpha=.5,reg_lambda=5.,colsample_bytree=.7,random_state=int(seed),n_jobs=4,verbosity=-1,force_col_wise=True)

def fit_full_path(targets,seed):
    oo=np.zeros((N_TRAIN,N_MODULES),np.float32); te_full=np.zeros((N_TEST,N_MODULES),np.float32); te_foldbag=np.zeros((N_TEST,N_MODULES),np.float32); its={mid:[] for mid in MODULES}
    for m,mid in enumerate(MODULES):
        Xt=np.column_stack([base_oof[:,m],USER_FEATURES[:N_TRAIN]]).astype(np.float32)
        Xo=np.column_stack([base_test[:,m],USER_FEATURES[N_TRAIN:]]).astype(np.float32); yy=targets[:,m].astype(np.int32)
        for f,(tr,va) in enumerate(ref_folds):
            clf=make_cal(seed+f+31*m); clf.fit(Xt[tr],yy[tr],eval_set=[(Xt[va],yy[va])],callbacks=[lgb.early_stopping(LGB_EARLY_STOP,verbose=False)])
            it=int(clf.best_iteration_ or LGB_ESTIMATORS); its[mid].append(it); oo[va,m]=clf.predict_proba(Xt[va],num_iteration=it)[:,1].astype(np.float32); te_foldbag[:,m]+=clf.predict_proba(Xo,num_iteration=it)[:,1].astype(np.float32)/len(ref_folds)
        final_it=max(20,int(np.median(its[mid]))); clf=make_cal(seed+997+31*m,final_it); clf.fit(Xt,yy); te_full[:,m]=clf.predict_proba(Xo)[:,1].astype(np.float32)
    return oo,te_full,te_foldbag,its

meta_oof,meta_test,meta_test_foldbag,best_iters=fit_full_path(Y_BIN,SEED)
# Structured P(top5) is deliberately a feature source only, never a direct final replacement.
p5_oof,p5_test,p5_test_foldbag,p5_iters=fit_full_path(Y_TOP5,SEED+4242)

CONSENSUS3_OVERRIDES={"M_002":0.44,"M_006":0.55,"M_007":0.60,"M_010":0.40,"M_011":0.55,"M_012":0.55,"M_015":0.44,"M_017":0.53}
CONS_W=np.full(N_MODULES,REFERENCE_W,np.float32)
for mid,val in CONSENSUS3_OVERRIDES.items(): CONS_W[MODULES.index(mid)]=float(val)
def apply_consensus(base,meta): return ((1.0-CONS_W)*base+CONS_W*meta).astype(np.float32)
consensus3_oof=apply_consensus(base_oof,meta_oof); consensus3_test=apply_consensus(base_test,meta_test)
print("old meta OOF",round(ndcg5(y,meta_oof),6),"structured P(top5)",round(ndcg5(y,p5_oof),6))
print("Consensus3 ordinary OOF",round(ndcg5(y,consensus3_oof),6),"public",PUBLIC_CONSENSUS3_LB)

# ---- cell 8 (markdown) ----
# ## V18 semantic corpus — message level, de-duplicated, no-chat hard-off
#
# V18 does not concatenate a user's trajectory into one vector. Each usable message is an independent semantic instance. The same normalized message is encoded only once, which is especially valuable for this synthetic/template-heavy organizer dataset.

# ---- cell 9 (code) ----
def semantic_normalize(text):
    s=html.unescape(str(text)); s=re.sub(r"\s+"," ",s).strip(); return s

def semantic_key(text): return semantic_normalize(text).casefold()

# Use only non-administrative messages. If no usable learning message exists, semantic branch is hard-off.
learn_sem=chat[~chat.is_noise].copy()
learn_sem["semantic_text"]=learn_sem.user_chat_text.fillna("").map(semantic_normalize)
learn_sem["semantic_key"]=learn_sem.semantic_text.map(semantic_key)
learn_sem=learn_sem[learn_sem.semantic_key.str.len()>0].copy()
learn_sem=learn_sem.sort_values(["user_id","timestamp","chat_id"]).reset_index(drop=True)

key_to_mid={}; unique_message_texts=[]
for row in learn_sem.itertuples(index=False):
    k=row.semantic_key
    if k not in key_to_mid:
        key_to_mid[k]=len(unique_message_texts); unique_message_texts.append(row.semantic_text)
learn_sem["msg_uid"]=learn_sem.semantic_key.map(key_to_mid).astype(int)

user_message_ids=[[] for _ in all_users]
for u,g in learn_sem.groupby("user_id",sort=False):
    if u in uid2idx: user_message_ids[uid2idx[u]]=g.msg_uid.astype(int).tolist()
HAS_SEMANTIC=np.asarray([len(x)>0 for x in user_message_ids],bool)
assert NO_CHAT_SEMANTIC_OFF and len(unique_message_texts)>0
print("semantic message rows",len(learn_sem),"unique encoded messages",len(unique_message_texts))
print("users semantic-on",int(HAS_SEMANTIC.sum()),"semantic-off",int((~HAS_SEMANTIC).sum()))

mods=modules.set_index("module_id").loc[MODULES].reset_index()
module_content_docs=[f"Modul pembelajaran Intelligo: {r.module_name}. Materi utama: {r.description_and_syllabus}." for r in mods.itertuples(index=False)]
module_full_docs=[f"Modul pembelajaran Intelligo: {r.module_name}. Materi utama: {r.description_and_syllabus}. Prasyarat: {r.prerequisite_level}." for r in mods.itertuples(index=False)]
MESSAGE_INSTRUCTION=(
    "Instruct: Representasikan maksud BELAJAR utama pada satu pesan siswa Intelligo. "
    "Fokus pada topik, skill, level, dan tujuan belajar yang sedang diminta. "
    "Abaikan pembayaran, akun, sertifikat, jadwal administratif, dan percakapan non-belajar. "
    "Representasi harus spesifik agar dapat dicocokkan dengan modul kursus yang paling relevan.\nQuery: "
)
print("example semantic message:",unique_message_texts[0][:300])
print("example module doc:",module_full_docs[0][:400])

# ---- cell 10 (code) ----
def _normalize_rows(x):
    x=np.asarray(x,np.float32); return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8)
def _fake_embeddings(texts,dim,salt):
    out=np.empty((len(texts),dim),np.float32)
    for i,t in enumerate(texts):
        seed=int(hashlib.sha256((salt+'|'+str(t)).encode()).hexdigest()[:16],16)%(2**32)
        out[i]=np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return _normalize_rows(out)
def _text_hash(texts,instruction):
    h=hashlib.sha256(instruction.encode())
    for t in texts: h.update(str(t).encode('utf-8')); h.update(b'\0')
    return h.hexdigest()[:16]
def _last_token_pool(last_hidden,attention_mask):
    mask=attention_mask.to(last_hidden.device)
    left_padding=bool((mask[:,-1].sum()==mask.shape[0]).item())
    if left_padding: return last_hidden[:,-1]
    seq=mask.sum(dim=1)-1
    return last_hidden[__import__('torch').arange(last_hidden.shape[0],device=last_hidden.device),seq]

qwen_model=None; qwen_tokenizer=None; qwen_input_device=None
def load_qwen():
    global qwen_model,qwen_tokenizer,qwen_input_device
    if qwen_model is not None or V18_SMOKE or V18_SKIP_QWEN: return
    import torch
    from transformers import AutoTokenizer,AutoModel
    if not torch.cuda.is_available() or torch.cuda.device_count()<2:
        raise RuntimeError("Full V18 requires Kaggle GPU T4 x2. Select Accelerator: GPU T4 x2.")
    for i in range(torch.cuda.device_count()):
        p=torch.cuda.get_device_properties(i); print(f"GPU {i}: {p.name}, {p.total_memory/1024**3:.1f} GiB")
    qwen_tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL,revision=EMBED_REVISION,padding_side='left')
    qwen_model=AutoModel.from_pretrained(
        EMBED_MODEL,revision=EMBED_REVISION,torch_dtype=torch.float16,
        device_map='balanced',max_memory={0:'13GiB',1:'13GiB','cpu':'24GiB'},
        low_cpu_mem_usage=True,attn_implementation='sdpa'
    )
    qwen_model.eval(); qwen_model.config.use_cache=False
    qwen_input_device=qwen_model.get_input_embeddings().weight.device
    print("hf_device_map",qwen_model.hf_device_map,"input",qwen_input_device)

def encode_cached(name,texts,instruction,max_length,start_batch):
    key=_text_hash(texts,instruction); path=CACHE_ROOT/f"{name}_{key}_{RUNTIME_DIM_FULL}.npy"
    if path.exists():
        arr=np.load(path).astype(np.float32)
        if arr.shape==(len(texts),RUNTIME_DIM_FULL): print("loaded cache",path.name); return arr
    if V18_SMOKE or V18_SKIP_QWEN:
        arr=_fake_embeddings([instruction+t for t in texts],RUNTIME_DIM_FULL,'v18-'+name); np.save(path,arr.astype(np.float16)); return arr
    import torch
    load_qwen(); prepared=[instruction+str(t) for t in texts]; result=np.empty((len(prepared),RUNTIME_DIM_FULL),np.float32)
    pos=0; bs=start_batch
    while pos<len(prepared):
        take=min(bs,len(prepared)-pos); batch=prepared[pos:pos+take]
        try:
            toks=qwen_tokenizer(batch,padding=True,truncation=True,max_length=max_length,return_tensors='pt')
            toks={k:v.to(qwen_input_device) for k,v in toks.items()}
            with torch.inference_mode(): out=qwen_model(**toks); emb=_last_token_pool(out.last_hidden_state,toks['attention_mask'])
            emb=emb.float().cpu().numpy().astype(np.float32); result[pos:pos+take]=emb; pos+=take
            if pos%max(100,bs*20)==0 or pos==len(prepared): print(name,pos,"/",len(prepared),"batch",bs)
            del toks,out,emb
        except RuntimeError as e:
            if 'out of memory' not in str(e).lower() or bs<=1: raise
            bs=max(1,bs//2); print("OOM -> retry batch",bs); torch.cuda.empty_cache(); gc.collect()
    result=_normalize_rows(result); np.save(path,result.astype(np.float16)); return result

message_emb=encode_cached('messages',unique_message_texts,MESSAGE_INSTRUCTION,MESSAGE_MAX_TOKENS,8)
module_content_emb=encode_cached('module_content',module_content_docs,'',MODULE_MAX_TOKENS,17)
module_full_emb=encode_cached('module_full',module_full_docs,'',MODULE_MAX_TOKENS,17)
if qwen_model is not None:
    del qwen_model,qwen_tokenizer; qwen_model=qwen_tokenizer=None
    import torch; torch.cuda.empty_cache(); gc.collect()
assert message_emb.shape==(len(unique_message_texts),RUNTIME_DIM_FULL)
assert module_full_emb.shape==(N_MODULES,RUNTIME_DIM_FULL)
print("message/module embedding shapes",message_emb.shape,module_full_emb.shape)

# ---- cell 11 (markdown) ----
# ## Label-independent multi-scale late interaction
#
# These features retain separate intent messages and aggregate only **after** message-to-module similarities are computed. This is the key representational difference from V16/V17.

# ---- cell 12 (code) ----
def trunc_norm(x,d): return _normalize_rows(np.asarray(x[:,:d],np.float32))
def row_z(x): return ((x-x.mean(1,keepdims=True))/np.maximum(x.std(1,keepdims=True),1e-6)).astype(np.float32)
def row_rank_score(x):
    order=np.argsort(-x,axis=1); ranks=np.empty_like(order); ranks[np.arange(len(x))[:,None],order]=np.arange(N_MODULES)[None,:]
    return (1.0-ranks.astype(np.float32)/(N_MODULES-1)).astype(np.float32)

def aggregate_similarity(msg_d,doc_d):
    latest=np.zeros((len(all_users),N_MODULES),np.float32); mx=np.zeros_like(latest); rec=np.zeros_like(latest); top2=np.zeros_like(latest); bestpos=np.zeros_like(latest)
    for ui,ids in enumerate(user_message_ids):
        if not ids: continue
        S=msg_d[np.asarray(ids,int)]@doc_d.T; n=len(ids)
        latest[ui]=S[-1]; mx[ui]=S.max(0)
        w=np.exp(-.70*np.arange(n-1,-1,-1,dtype=np.float32)); w/=w.sum(); rec[ui]=(S*w[:,None]).sum(0)
        if n==1: top2[ui]=S[0]
        else: top2[ui]=np.partition(S,-2,axis=0)[-2:].mean(0)
        bestpos[ui]=(np.argmax(S,axis=0)+1)/float(n)
    return latest,mx,rec,top2,bestpos

static_parts=[]; key_1024={}
for d in STATIC_DIMS:
    md=trunc_norm(message_emb,d); cd=trunc_norm(module_content_emb,d); fd=trunc_norm(module_full_emb,d)
    for view,docs in [('content',cd),('full',fd)]:
        latest,mx,rec,top2,bestpos=aggregate_similarity(md,docs)
        static_parts += [latest[:,:,None],mx[:,:,None],rec[:,:,None],top2[:,:,None]]
        # Relative features only at the central semantic scale to control variance.
        central=(d==STATIC_DIMS[min(1,len(STATIC_DIMS)-1)])
        if central:
            for mat in (latest,mx,rec): static_parts += [row_z(mat)[:,:,None],row_rank_score(mat)[:,:,None]]
            static_parts.append(bestpos[:,:,None]); key_1024[(view,'max')]=mx
    del md,cd,fd; gc.collect()
STATIC_SEM=np.concatenate(static_parts,axis=2).astype(np.float32)
STATIC_SEM[~HAS_SEMANTIC,:,:]=0.0
MSG_ADAPT_BASE=trunc_norm(message_emb,ADAPTER_DIM); MOD_ADAPT_BASE=trunc_norm(module_full_emb,ADAPTER_DIM)
FORMAL_BASE_SIM=(MSG_ADAPT_BASE@MOD_ADAPT_BASE.T).astype(np.float32)
print("STATIC_SEM",STATIC_SEM.shape,"adapter bases",MSG_ADAPT_BASE.shape,MOD_ADAPT_BASE.shape)

# ---- cell 13 (markdown) ----
# ## Fold-safe organizer metric and in-domain prototype bank
#
# The adapter is intentionally tiny: it learns only a positive diagonal reweighting around the frozen Qwen space. Prototype construction and adapter fitting only see the supplied organizer labels inside the current training fold.

# ---- cell 14 (code) ----
def fit_diagonal_metric_adapter(global_train,seed):
    """Learn a tiny positive diagonal metric from organizer Top-5 positives + hard negatives."""
    rng=np.random.default_rng(int(seed)); pairs=[]
    for u in np.asarray(global_train,int):
        ids=user_message_ids[u]
        if not ids: continue
        ids=np.asarray(ids,int); user_max=FORMAL_BASE_SIM[ids].max(0)
        positives=np.flatnonzero(RANK_POS[u]<=3)
        negatives=np.flatnonzero(RANK_POS[u]>3)
        if len(negatives)==0: continue
        hard_neg=int(negatives[np.argmax(user_max[negatives])])
        for m in positives:
            best=int(ids[np.argmax(FORMAL_BASE_SIM[ids,m])]); pairs.append((best,int(m),1))
            pairs.append((best,hard_neg,0))
    if not pairs: return np.ones(ADAPTER_DIM,np.float32)
    pos_ix=np.arange(0,len(pairs),2)
    if len(pos_ix)>ADAPTER_MAX_POS:
        keep_pos=np.sort(rng.choice(pos_ix,size=ADAPTER_MAX_POS,replace=False)); keep=np.ravel(np.column_stack([keep_pos,keep_pos+1])); pairs=[pairs[i] for i in keep]
    Xp=np.empty((len(pairs),ADAPTER_DIM),np.float32); yp=np.empty(len(pairs),np.int8)
    for i,(mi,m,lab) in enumerate(pairs): Xp[i]=MSG_ADAPT_BASE[mi]*MOD_ADAPT_BASE[m]; yp[i]=lab
    clf=SGDClassifier(loss='log_loss',alpha=ADAPTER_ALPHA,max_iter=(20 if V18_SMOKE else 60),tol=1e-4,class_weight='balanced',random_state=int(seed),average=True)
    clf.fit(Xp,yp); coef=np.asarray(clf.coef_[0],float)
    med=np.median(coef); scale=np.median(np.abs(coef-med))*1.4826+1e-8; z=np.clip((coef-med)/scale,-2.5,2.5)
    metric=np.exp(ADAPTER_METRIC_STRENGTH*z).astype(np.float32)
    return metric

def _adapt_space(metric):
    sw=np.sqrt(np.maximum(metric,1e-4)).astype(np.float32)
    return _normalize_rows(MSG_ADAPT_BASE*sw),_normalize_rows(MOD_ADAPT_BASE*sw)

def build_module_banks(global_train,msg_adapt,doc_adapt,seed):
    banks=[]; global_train=np.asarray(global_train,int)
    for m in range(N_MODULES):
        chosen=[]; weights=[]
        for u in global_train:
            if RANK_POS[u,m]>3 or not user_message_ids[u]: continue
            ids=np.asarray(user_message_ids[u],int); sims=msg_adapt[ids]@doc_adapt[m]
            chosen.append(int(ids[np.argmax(sims)])); weights.append(float(max(y[u,m],.2)))
        if not chosen:
            banks.append(doc_adapt[m:m+1].copy()); continue
        counts=Counter(chosen); uniq=np.asarray(sorted(counts),int)
        Xc=msg_adapt[uniq]; sw=np.asarray([math.sqrt(counts[int(q)]) for q in uniq],np.float32)
        k=min(PROTOTYPE_K,len(uniq))
        if k==1: centers=Xc[:1].copy()
        else:
            km=MiniBatchKMeans(n_clusters=k,batch_size=min(256,max(32,len(uniq))),n_init=2,max_iter=(10 if V18_SMOKE else 35),random_state=int(seed+97*m))
            km.fit(Xc,sample_weight=sw); centers=_normalize_rows(km.cluster_centers_.astype(np.float32))
        banks.append(centers)
    return banks

def score_users_with_bank(global_users,msg_adapt,doc_adapt,banks):
    global_users=np.asarray(global_users,int); out=np.zeros((len(global_users),N_MODULES,8),np.float32)
    for ii,u in enumerate(global_users):
        ids=user_message_ids[u]
        if not ids: continue
        ids=np.asarray(ids,int); X=msg_adapt[ids]; n=len(ids); rw=np.exp(-.70*np.arange(n-1,-1,-1,dtype=np.float32)); rw/=rw.sum()
        for m in range(N_MODULES):
            P=banks[m]; S=X@P.T; best=S.max(1); formal=X@doc_adapt[m]
            out[ii,m,0]=best[-1]                         # latest -> closest organizer prototype
            out[ii,m,1]=best.max()                       # any intent -> closest prototype
            out[ii,m,2]=(best*rw).sum()                  # recency-weighted prototype evidence
            out[ii,m,3]=best[-2:].mean() if n>=2 else best[0]
            out[ii,m,4]=(np.argmax(best)+1)/float(n)     # where strongest prototype evidence occurs
            out[ii,m,5]=formal[-1]                       # adapted formal latest
            out[ii,m,6]=formal.max()                     # adapted formal max
            out[ii,m,7]=min(float(best.max()),float(formal.max())) # formal/in-domain agreement
    return out

def build_message_memory(fit_users):
    fit_users=np.asarray(fit_users,int); sums5={}; sumsrel={}; counts=Counter(); user_keys={}
    for u in fit_users:
        keys=sorted(set(user_message_ids[u])); user_keys[int(u)]=set(keys)
        for q in keys:
            if q not in sums5:
                sums5[q]=np.zeros(N_MODULES,np.float32); sumsrel[q]=np.zeros(N_MODULES,np.float32)
            sums5[q]+=Y_TOP5[u]; sumsrel[q]+=y[u]; counts[q]+=1
    bank_ids=np.asarray(sorted(counts),int)
    if len(bank_ids)==0:
        return dict(bank_ids=bank_ids,sum5=np.zeros((0,N_MODULES),np.float32),sumrel=np.zeros((0,N_MODULES),np.float32),count=np.zeros(0,np.float32),user_keys=user_keys)
    return dict(bank_ids=bank_ids,sum5=np.stack([sums5[q] for q in bank_ids]),sumrel=np.stack([sumsrel[q] for q in bank_ids]),count=np.asarray([counts[q] for q in bank_ids],np.float32),user_keys=user_keys)

def _topk_message_neighbors(msg_adapt,bank_ids,query_ids,k):
    bank_ids=np.asarray(bank_ids,int); query_ids=np.asarray(query_ids,int); k=min(int(k),len(bank_ids))
    if k<=0 or len(query_ids)==0: return {}
    B=msg_adapt[bank_ids]; out={}; used_gpu=False
    try:
        import torch
        if torch.cuda.is_available() and not V18_SMOKE:
            used_gpu=True; dev=torch.device('cuda:0'); bt=torch.from_numpy(B).to(dev)
            for st in range(0,len(query_ids),512):
                ids=query_ids[st:st+512]; qt=torch.from_numpy(msg_adapt[ids]).to(dev); vals,ix=torch.topk(qt@bt.T,k=k,dim=1)
                vals=vals.float().cpu().numpy(); ix=ix.cpu().numpy()
                for j,q in enumerate(ids): out[int(q)]=(ix[j],vals[j])
                del qt,vals,ix
            del bt; torch.cuda.empty_cache()
    except Exception as e:
        print('memory GPU retrieval fallback:',type(e).__name__)
    if not used_gpu:
        for st in range(0,len(query_ids),256):
            ids=query_ids[st:st+256]; S=msg_adapt[ids]@B.T
            ix=np.argpartition(-S,k-1,axis=1)[:,:k]; vals=np.take_along_axis(S,ix,axis=1); order=np.argsort(-vals,axis=1); ix=np.take_along_axis(ix,order,axis=1); vals=np.take_along_axis(vals,order,axis=1)
            for j,q in enumerate(ids): out[int(q)]=(ix[j],vals[j])
    return out

def score_users_with_memory(fit_users,score_users,msg_adapt,memory):
    score_users=np.asarray(score_users,int); fit_set=set(map(int,np.asarray(fit_users,int)))
    out=np.zeros((len(score_users),N_MODULES,6),np.float32); bank_ids=memory['bank_ids']
    if len(bank_ids)==0: return out
    query_ids=np.asarray(sorted({int(q) for u in score_users for q in user_message_ids[u]}),int)
    neigh=_topk_message_neighbors(msg_adapt,bank_ids,query_ids,MEMORY_K)
    for ii,u0 in enumerate(score_users):
        u=int(u0); ids=user_message_ids[u]
        if not ids: continue
        pri5=[]; prir=[]; conf=[]; exact=[]
        for q in ids:
            ni,sv=neigh[int(q)]; vals5=[]; valsr=[]; ww=[]
            for bix,sim in zip(ni,sv):
                cnt=float(memory['count'][bix]); s5=memory['sum5'][bix].copy(); sr=memory['sumrel'][bix].copy(); bid=int(bank_ids[bix])
                if u in fit_set and bid in memory['user_keys'].get(u,set()):
                    cnt-=1.0; s5-=Y_TOP5[u]; sr-=y[u]
                if cnt<=0: continue
                vals5.append(s5/cnt); valsr.append(sr/cnt); ww.append((float(sim),cnt))
            if not ww:
                pri5.append(np.zeros(N_MODULES,np.float32)); prir.append(np.zeros(N_MODULES,np.float32)); conf.append(0.0); exact.append(0.0); continue
            sims=np.asarray([a for a,_ in ww],float); cnts=np.asarray([b for _,b in ww],float)
            w=np.exp((sims-sims.max())/MEMORY_TEMP)*np.log1p(cnts); w/=max(w.sum(),1e-12)
            pri5.append(np.sum(np.asarray(vals5)*w[:,None],axis=0)); prir.append(np.sum(np.asarray(valsr)*w[:,None],axis=0)); conf.append(float(sims.max())); exact.append(float(sims.max()>.999))
        P=np.asarray(pri5,np.float32); R=np.asarray(prir,np.float32); n=len(ids); rw=np.exp(-.70*np.arange(n-1,-1,-1,dtype=np.float32)); rw/=rw.sum()
        out[ii,:,0]=P[-1]; out[ii,:,1]=(P*rw[:,None]).sum(0); out[ii,:,2]=P.max(0); out[ii,:,3]=(R*rw[:,None]).sum(0)
        out[ii,:,4]=float(np.sum(np.asarray(conf)*rw)); out[ii,:,5]=float(max(exact))
    return out

def build_prototype_features(fit_users,score_users,seed):
    """Fit adapter + organizer prototype bank + fuzzy message memory on fit_users; score users without reading their labels."""
    fit_users=np.asarray(fit_users,int); score_users=np.asarray(score_users,int)
    metric=fit_diagonal_metric_adapter(fit_users,seed+1009); msg_a,doc_a=_adapt_space(metric); banks=build_module_banks(fit_users,msg_a,doc_a,seed+2003)
    proto=score_users_with_bank(score_users,msg_a,doc_a,banks)
    memory=build_message_memory(fit_users); mem=score_users_with_memory(fit_users,score_users,msg_a,memory)
    out=np.concatenate([proto,mem],axis=2).astype(np.float32); out[~HAS_SEMANTIC[score_users],:,:]=0.0
    del msg_a,doc_a,banks,memory,proto,mem; gc.collect(); return out

print("prototype functions ready")

# ---- cell 15 (markdown) ----
# ## Semantic second stage + distribution-preserving transport
#
# The semantic heads are trained out-of-fold. The new probabilities are not allowed to replace Consensus3 directly; they are quantile-aligned back to the old module-wise relevance scale first.

# ---- cell 16 (code) ----
def make_sem_cal(seed,n_estimators=None):
    n=int(SEM_LGB_ESTIMATORS if n_estimators is None else n_estimators)
    return lgb.LGBMClassifier(n_estimators=n,learning_rate=.025,num_leaves=9,min_child_samples=90,
        reg_alpha=1.5,reg_lambda=10.0,colsample_bytree=.72,random_state=int(seed),n_jobs=4,verbosity=-1,force_col_wise=True)

def semantic_design(global_idx,proto_s):
    # Deliberately exclude old/base/p5 supervised predictions: V18 semantic expert must remain orthogonal and cannot inherit second-level validation leakage.
    global_idx=np.asarray(global_idx,int); rows=[]
    a=np.column_stack([A[global_idx],A_INVALID[global_idx]]).astype(np.float32)
    for m in range(N_MODULES): rows.append(np.column_stack([a,STATIC_SEM[global_idx,m,:],proto_s[:,m,:]]).astype(np.float32))
    return rows

def fit_semantic_heads(global_train,global_out,folds_local,seed):
    global_train=np.asarray(global_train,int); global_out=np.asarray(global_out,int)
    bin_tr=np.zeros((len(global_train),N_MODULES),np.float32); top_tr=np.zeros_like(bin_tr)
    bin_out=np.zeros((len(global_out),N_MODULES),np.float32); top_out=np.zeros_like(bin_out)
    for f,(lt,lv) in enumerate(folds_local):
        # Fold-aligned features: validation labels never enter adapter, prototypes, or semantic training features.
        fit_users=global_train[lt]; val_users=global_train[lv]
        score_users=np.concatenate([fit_users,val_users,global_out])
        pf=build_prototype_features(fit_users,score_users,seed+7001*f)
        nfit=len(fit_users); nval=len(val_users)
        pfit=pf[:nfit]; pval=pf[nfit:nfit+nval]; pout=pf[nfit+nval:]
        Xfit=semantic_design(fit_users,pfit); Xval=semantic_design(val_users,pval); Xout=semantic_design(global_out,pout)
        for m in range(N_MODULES):
            for T,oo,ot,off in [(Y_BIN,bin_tr,bin_out,0),(Y_TOP5,top_tr,top_out,50000)]:
                yy=T[fit_users,m].astype(np.int32); clf=make_sem_cal(seed+off+37*m+f)
                clf.fit(Xfit[m],yy,eval_set=[(Xval[m],T[val_users,m].astype(np.int32))],callbacks=[lgb.early_stopping(SEM_LGB_EARLY_STOP,verbose=False)])
                it=int(clf.best_iteration_ or SEM_LGB_ESTIMATORS); oo[lv,m]=clf.predict_proba(Xval[m],num_iteration=it)[:,1].astype(np.float32); ot[:,m]+=clf.predict_proba(Xout[m],num_iteration=it)[:,1].astype(np.float32)/len(folds_local)
        del pf,pfit,pval,pout,Xfit,Xval,Xout; gc.collect()
    return bin_tr,bin_out,top_tr,top_out

def quantile_align(source,target):
    source=np.asarray(source,float); target=np.asarray(target,float); order=np.argsort(source,kind='mergesort'); out=np.empty_like(source,dtype=np.float32); out[order]=np.sort(target).astype(np.float32); return out

def align_matrix(source,target,group_id=None):
    out=np.zeros_like(source,np.float32)
    if group_id is None:
        for m in range(source.shape[1]): out[:,m]=quantile_align(source[:,m],target[:,m])
    else:
        for g in np.unique(group_id):
            ix=np.where(group_id==g)[0]
            for m in range(source.shape[1]): out[ix,m]=quantile_align(source[ix,m],target[ix,m])
    return out

def local_fold_ids(n,folds_local):
    ids=np.full(n,-1,int)
    for f,(_,lv) in enumerate(folds_local): ids[lv]=f
    assert (ids>=0).all(); return ids

print("semantic head / alignment functions ready")

# ---- cell 17 (markdown) ----
# ## Nested module trust
#
# This removes the V17 mistake of inheriting "semantic-friendly" modules from an older representation. For each representation and attack family, module trust is learned only from other folds.

# ---- cell 18 (code) ----
def _single_module_score(base_s,old_s,proposal_s,m,beta,mask):
    mm=old_s.copy(); mm[:,m]=old_s[:,m]+float(beta)*mask*(proposal_s[:,m]-old_s[:,m]); return apply_consensus(base_s,mm)

def nested_module_trust(y_local,base_s,old_s,proposal_s,fold_id_local,has_sem_local,broad=False):
    F=int(fold_id_local.max()+1); gains=np.zeros((N_MODULES,F),np.float64); anchor=apply_consensus(base_s,old_s); au=per_user_ndcg(y_local,anchor)
    for m in range(N_MODULES):
        probe=_single_module_score(base_s,old_s,proposal_s,m,TRUST_PROBE_BETA,has_sem_local.astype(np.float32)); du=per_user_ndcg(y_local,probe)-au
        for f in range(F): gains[m,f]=du[fold_id_local==f].mean()
    trust_train=np.zeros((len(y_local),N_MODULES),np.float32)
    for f in range(F):
        others=[g for g in range(F) if g!=f]
        for m in range(N_MODULES):
            vals=gains[m,others]; need=max(1,int(math.ceil(len(vals)*(0.50 if broad else 0.66))))
            ok=(vals.mean()>0) and int((vals>0).sum())>=need
            trust_train[fold_id_local==f,m]=float(ok)
    trust_out=np.zeros(N_MODULES,np.float32)
    for m in range(N_MODULES):
        vals=gains[m]; need=max(1,int(math.ceil(F*(0.50 if broad else 0.66))))
        trust_out[m]=float((vals.mean()>0) and int((vals>0).sum())>=need)
    return trust_train,trust_out,gains

def apply_trusted_attack(base_s,old_s,proposal_s,trust,beta,has_sem):
    if np.ndim(trust)==1: t=np.asarray(trust,np.float32)[None,:]
    else: t=np.asarray(trust,np.float32)
    row=np.asarray(has_sem,np.float32)[:,None]
    mm=old_s+float(beta)*row*t*(proposal_s-old_s)
    return apply_consensus(base_s,mm)

def build_semantic_attacks(global_train,global_out,folds_local,base_tr,base_out,old_tr,old_out,p5_tr,p5_out,seed,old_anchor_out=None):
    global_train=np.asarray(global_train,int); global_out=np.asarray(global_out,int); fid=local_fold_ids(len(global_train),folds_local)
    old_anchor_out=old_out if old_anchor_out is None else np.asarray(old_anchor_out,np.float32)
    sb_tr,sb_out,st_tr,st_out=fit_semantic_heads(global_train,global_out,folds_local,seed+20000)
    ab_tr=align_matrix(sb_tr,old_tr,fid); ab_out=align_matrix(sb_out,old_anchor_out,None)
    at_tr=align_matrix(st_tr,old_tr,fid); at_out=align_matrix(st_out,old_anchor_out,None)
    dual_tr=align_matrix(.65*ab_tr+.35*at_tr,old_tr,fid); dual_out=align_matrix(.65*ab_out+.35*at_out,old_anchor_out,None)
    proposals_tr={'bin':ab_tr,'dual':dual_tr}; proposals_out={'bin':ab_out,'dual':dual_out}
    attacks_tr={}; attacks_out={}; trust_info={}
    for name,cfg in ATTACK_CONFIG.items():
        ptr=proposals_tr[cfg['kind']]; pout=proposals_out[cfg['kind']]
        ttr,tout,g=nested_module_trust(y[global_train],base_tr,old_tr,ptr,fid,HAS_SEMANTIC[global_train],broad=cfg['broad'])
        attacks_tr[name]=apply_trusted_attack(base_tr,old_tr,ptr,ttr,cfg['beta'],HAS_SEMANTIC[global_train])
        attacks_out[name]=apply_trusted_attack(base_out,old_anchor_out,pout,tout,cfg['beta'],HAS_SEMANTIC[global_out])
        trust_info[name]={'out_trust':tout.copy(),'gains':g.copy()}
    return attacks_tr,attacks_out,trust_info,dict(bin_tr=sb_tr,bin_out=sb_out,top_tr=st_tr,top_out=st_out)

print("nested_module_trust ready")

# ---- cell 19 (markdown) ----
# ## Ordinary reference OOF + test attacks

# ---- cell 20 (code) ----
TRAIN_GLOBAL=np.arange(N_TRAIN); TEST_GLOBAL=np.arange(N_TRAIN,N_TRAIN+N_TEST)
ordinary_attacks,test_attacks,ordinary_trust,ordinary_sem=build_semantic_attacks(
    TRAIN_GLOBAL,TEST_GLOBAL,ref_folds,base_oof,base_test,meta_oof,meta_test_foldbag,p5_oof,p5_test_foldbag,SEED+18000,old_anchor_out=meta_test
)
print("Consensus3 ordinary",round(ndcg5(y,consensus3_oof),6),"public",PUBLIC_CONSENSUS3_LB)
for name,s in ordinary_attacks.items():
    selected=[MODULES[i] for i,v in enumerate(ordinary_trust[name]['out_trust']) if v>0]
    print(name,"delta",round(ndcg5(y,s)-ndcg5(y,consensus3_oof),6),"test-selected",selected)

# ---- cell 21 (markdown) ----
# ## Fresh-seed fully isolated audit
#
# Qwen embeddings and static semantic similarities are label-independent and may be shared. Every supervised sparse model, structured calibrator, metric adapter, organizer prototype, semantic head, and module-trust decision is rebuilt without the audit validation users.

# ---- cell 22 (code) ----
def sparse_matched(global_train,global_out,folds_local):
    global_train=np.asarray(global_train,int); global_out=np.asarray(global_out,int); n=len(global_train)
    raw=np.zeros((n,N_MODULES),np.float32); top=np.zeros_like(raw); raw_out=np.zeros((len(global_out),N_MODULES),np.float32); top_out=np.zeros_like(raw_out)
    for lt,lv in folds_local:
        gtr=global_train[lt]; gva=global_train[lv]
        mr=Ridge(alpha=RIDGE_ALPHA,solver='lsqr',tol=2e-3,max_iter=1000); mt=Ridge(alpha=RIDGE_ALPHA,solver='lsqr',tol=2e-3,max_iter=1000)
        mr.fit(X[gtr],y[gtr]); mt.fit(X[gtr],Y_TOP[gtr]); raw[lv]=mr.predict(X[gva]); top[lv]=mt.predict(X[gva])
        raw_out+=mr.predict(X[global_out]).astype(np.float32)/len(folds_local); top_out+=mt.predict(X[global_out]).astype(np.float32)/len(folds_local)
    return (.9*raw+.1*top).astype(np.float32),(.9*raw_out+.1*top_out).astype(np.float32)

def structured_matched(global_train,global_out,folds_local,base_tr,base_out,targets,seed):
    global_train=np.asarray(global_train,int); global_out=np.asarray(global_out,int); oo=np.zeros((len(global_train),N_MODULES),np.float32); out=np.zeros((len(global_out),N_MODULES),np.float32)
    for m in range(N_MODULES):
        Xt=np.column_stack([base_tr[:,m],USER_FEATURES[global_train]]).astype(np.float32); Xo=np.column_stack([base_out[:,m],USER_FEATURES[global_out]]).astype(np.float32); yy=targets[global_train,m].astype(np.int32)
        for f,(lt,lv) in enumerate(folds_local):
            clf=make_cal(seed+31*m+f,audit=True); clf.fit(Xt[lt],yy[lt],eval_set=[(Xt[lv],yy[lv])],callbacks=[lgb.early_stopping(AUDIT_EARLY_STOP,verbose=False)])
            it=int(clf.best_iteration_ or AUDIT_LGB_ESTIMATORS); oo[lv,m]=clf.predict_proba(Xt[lv],num_iteration=it)[:,1].astype(np.float32); out[:,m]+=clf.predict_proba(Xo,num_iteration=it)[:,1].astype(np.float32)/len(folds_local)
    return oo,out

def build_isolated_anchor(global_train,global_out,seed):
    global_train=np.asarray(global_train,int); global_out=np.asarray(global_out,int); folds_local=make_local_folds(global_train,N_INNER,seed)
    btr,bout=sparse_matched(global_train,global_out,folds_local)
    otr,oout=structured_matched(global_train,global_out,folds_local,btr,bout,Y_BIN,seed+1000)
    ptr,pout=structured_matched(global_train,global_out,folds_local,btr,bout,Y_TOP5,seed+3000)
    return folds_local,btr,bout,otr,oout,ptr,pout

audit_skf=StratifiedKFold(N_AUDIT_OUTER,shuffle=True,random_state=AUDIT_SEED)
audit_splits=list(audit_skf.split(np.zeros(N_TRAIN),strata)); audit_fold_id=np.full(N_TRAIN,-1,int)
audit_anchor=np.zeros_like(y,np.float32); audit_attacks={name:np.zeros_like(y,np.float32) for name in ATTACK_CONFIG}
for of,(otr,ova) in enumerate(audit_splits):
    print(f"AUDIT OUTER {of+1}/{N_AUDIT_OUTER}")
    inner,btr,bva,otr_old,ova_old,ptr,pva=build_isolated_anchor(otr,ova,AUDIT_SEED+100*of)
    atr,avo,tinfo,_=build_semantic_attacks(otr,ova,inner,btr,bva,otr_old,ova_old,ptr,pva,AUDIT_SEED+5000+100*of)
    audit_anchor[ova]=apply_consensus(bva,ova_old); audit_fold_id[ova]=of
    for name in ATTACK_CONFIG: audit_attacks[name][ova]=avo[name]
    gc.collect()
assert (audit_fold_id>=0).all()
print("matched audit anchor",round(ndcg5(y,audit_anchor),6))
for name,s in audit_attacks.items(): print(name,"delta",round(ndcg5(y,s)-ndcg5(y,audit_anchor),6))

# ---- cell 23 (markdown) ----
# ## Chronological future-like audit

# ---- cell 24 (code) ----
chrono_cut=int(round(N_TRAIN*.75)); chrono_tr=np.arange(chrono_cut); chrono_va=np.arange(chrono_cut,N_TRAIN)
inner,btr,bva,otr_old,ova_old,ptr,pva=build_isolated_anchor(chrono_tr,chrono_va,AUDIT_SEED+9000)
ctr,cvo,chrono_trust,_=build_semantic_attacks(chrono_tr,chrono_va,inner,btr,bva,otr_old,ova_old,ptr,pva,AUDIT_SEED+19000)
chrono_anchor=apply_consensus(bva,ova_old); chrono_attacks=cvo
print("chrono anchor",round(ndcg5(y[chrono_va],chrono_anchor),6),"n",len(chrono_va))
for name,s in chrono_attacks.items(): print(name,"delta",round(ndcg5(y[chrono_va],s)-ndcg5(y[chrono_va],chrono_anchor),6))

# ---- cell 25 (markdown) ----
# ## Decision, two submission slots, and artifacts

# ---- cell 26 (code) ----
ord_anchor_u=per_user_ndcg(y,consensus3_oof); aud_anchor_u=per_user_ndcg(y,audit_anchor); chr_anchor_u=per_user_ndcg(y[chrono_va],chrono_anchor)
rows=[]
for i,name in enumerate(ATTACK_CONFIG):
    po=ordinary_attacks[name]; pa=audit_attacks[name]; pc=chrono_attacks[name]; pt=test_attacks[name]
    du_o=per_user_ndcg(y,po)-ord_anchor_u; du_a=per_user_ndcg(y,pa)-aud_anchor_u; du_c=per_user_ndcg(y[chrono_va],pc)-chr_anchor_u
    ord_f=np.asarray([du_o[ref_fold_id==f].mean() for f in range(N_FOLDS)])
    aud_f=np.asarray([du_a[audit_fold_id==f].mean() for f in range(N_AUDIT_OUTER)])
    ci=paired_boot(du_a,AUDIT_SEED+777+i)
    top5=top5_set_change_rate(pt,consensus3_test); order=top5_order_change_rate(pt,consensus3_test); top1=top1_change_rate(pt,consensus3_test)
    min_core=min(float(du_a.mean()),float(du_c.mean()))
    robust=float(.25*du_o.mean()+.50*du_a.mean()+.25*du_c.mean()+.40*min(0.0,min_core))
    gate=bool(du_o.mean()>=MIN_ORD_DELTA and du_a.mean()>=MIN_AUDIT_DELTA and du_c.mean()>=MIN_CHRONO_DELTA and int((aud_f>0).sum())>=MIN_AUDIT_WINS and ci[0]>=MIN_AUDIT_CI_LOW and top5>=MIN_TEST_TOP5_CHANGE and top5<=MAX_TEST_TOP5_CHANGE and top1<=MAX_TEST_TOP1_CHANGE)
    selected=[MODULES[j] for j,v in enumerate(ordinary_trust[name]['out_trust']) if v>0]
    rows.append(dict(candidate=name,ordinary_delta=float(du_o.mean()),ordinary_fold_wins=int((ord_f>0).sum()),ordinary_folds=json.dumps(ord_f.tolist()),audit_delta=float(du_a.mean()),audit_fold_wins=int((aud_f>0).sum()),audit_folds=json.dumps(aud_f.tolist()),audit_ci_low=ci[0],audit_ci_high=ci[1],chrono_delta=float(du_c.mean()),test_top5_set_change=top5,test_top5_order_change=order,test_top1_change=top1,selected_modules=','.join(selected),robust_score=robust,promotion_gate=gate))

diagnostics=pd.DataFrame(rows).sort_values(['promotion_gate','robust_score','audit_delta'],ascending=[False,False,False]).reset_index(drop=True)
display(diagnostics.round(6))
ranked=diagnostics.candidate.tolist(); attack1_name=ranked[0]
# Prefer a genuinely different family/strength for slot 2 when possible.
attack2_name=ranked[1] if len(ranked)>1 else ranked[0]
attack1_test=test_attacks[attack1_name]; attack2_test=test_attacks[attack2_name]
attack1_gate=bool(diagnostics.loc[0,'promotion_gate'])
primary_test=attack1_test.copy() if attack1_gate else consensus3_test.copy(); primary_label=attack1_name if attack1_gate else 'consensus3_fallback'
print("\n================ V18 DECISION ================")
print("ordinary Consensus3",round(ndcg5(y,consensus3_oof),6))
print("matched audit anchor",round(ndcg5(y,audit_anchor),6))
print("chronological anchor",round(ndcg5(y[chrono_va],chrono_anchor),6))
print("ATTACK 1",attack1_name,"gate",attack1_gate)
print("ATTACK 2",attack2_name)
print("PRIMARY",primary_label)
print("Required public improvement to beat leader:",round(REQUIRED_PUBLIC_DELTA,6))
print("Do not infer a guaranteed hidden-LB score from local CV. Share this decision block before spending either submission slot.")
print("================================================")

# ---- cell 27 (code) ----
def save_submission(scores,name):
    out=sample.copy(); out.loc[:,MODULES]=scores
    root=Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('/mnt/data'); path=root/name; out.to_csv(path,index=False)
    chk=pd.read_csv(path); assert chk.shape==sample.shape and chk.user_id.tolist()==sample.user_id.tolist() and chk.user_id.is_unique
    assert chk[MODULES].notna().all().all() and np.isfinite(chk[MODULES].to_numpy()).all(); print("saved",path); return path
root=Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('/mnt/data')
paths={}
paths['consensus3']=save_submission(consensus3_test,'submission_v18_consensus3_reference.csv')
paths['attack1']=save_submission(attack1_test,'submission_v18_attack1.csv')
paths['attack2']=save_submission(attack2_test,'submission_v18_attack2.csv')
paths['primary']=save_submission(primary_test,'submission_v18_primary.csv')
diagnostics.to_csv(root/'v18_diagnostics.csv',index=False)
np.savez_compressed(root/'v18_oof_artifacts.npz',y=y,base_oof=base_oof,meta_oof=meta_oof,p5_oof=p5_oof,consensus3_oof=consensus3_oof,audit_anchor=audit_anchor,base_test=base_test,meta_test=meta_test,p5_test=p5_test,consensus3_test=consensus3_test,ref_fold_id=ref_fold_id,audit_fold_id=audit_fold_id,**{f'ordinary_{k}':v for k,v in ordinary_attacks.items()},**{f'audit_{k}':v for k,v in audit_attacks.items()})
summary={
    'public_consensus3_lb':PUBLIC_CONSENSUS3_LB,'public_leader_lb':PUBLIC_LEADER_LB,'required_public_delta':REQUIRED_PUBLIC_DELTA,
    'model':EMBED_MODEL,'revision':EMBED_REVISION,'representation':'message-level multi-intent + in-domain fold-safe prototypes + diagonal metric adapter',
    'diagnostics':diagnostics.to_dict(orient='records'),'attack1':attack1_name,'attack2':attack2_name,'primary':primary_label,
    'files':{k:v.name for k,v in paths.items()},
    'compliance':{'task_specific_data':'organizer files only','external task dataset':False,'pretrained_usage':'frozen open-source embedding feature extractor only','generation_used':False,'direct_test_label_request':False}
}
with open(root/'v18_summary.json','w') as f: json.dump(summary,f,indent=2)
print("saved diagnostics, summary, OOF artifacts")

# ---- cell 28 (code) ----
# Hard invariants: fail loudly on malformed or unsafe outputs.
assert len(MODULES)==17 and y.shape==(N_TRAIN,17)
assert CONSENSUS3_OVERRIDES=={"M_002":0.44,"M_006":0.55,"M_007":0.60,"M_010":0.40,"M_011":0.55,"M_012":0.55,"M_015":0.44,"M_017":0.53}
assert np.allclose(consensus3_oof,apply_consensus(base_oof,meta_oof),atol=0,rtol=0)
assert np.allclose(consensus3_test,apply_consensus(base_test,meta_test),atol=0,rtol=0)
assert NO_CHAT_SEMANTIC_OFF and np.all(STATIC_SEM[~HAS_SEMANTIC]==0)
for name in ATTACK_CONFIG:
    assert np.isfinite(ordinary_attacks[name]).all() and np.isfinite(test_attacks[name]).all() and np.isfinite(audit_attacks[name]).all()
# Semantic hard-off must leave no-chat test rows exactly at the anchor.
nochat_test=(~HAS_SEMANTIC[TEST_GLOBAL])
for name in ATTACK_CONFIG:
    if nochat_test.any(): assert np.allclose(test_attacks[name][nochat_test],consensus3_test[nochat_test],atol=1e-7)
assert diagnostics.candidate.nunique()==len(ATTACK_CONFIG)
if not V18_SMOKE:
    assert abs(ndcg5(y,consensus3_oof)-0.6643682395561207)<0.0012,ndcg5(y,consensus3_oof)
if not attack1_gate: assert np.allclose(primary_test,consensus3_test,atol=0,rtol=0)
print("V18 hard invariants: PASS")
