# Converted from notebooks/v17.ipynb: markdown cells become comments, %/! magics neutralized.

# ---- cell 0 (markdown) ----
# # MineToday V17 — Top5-Aware Module-Specific Semantic Calibration
#
# **Immutable public champion:** V12.2 `Consensus3`, public NDCG@5 **0.65856**. Current public leader at design time: **0.66082**.
#
# V17 is built from the post-V16 diagnosis rather than from another generic ranker. It keeps the exact public-winning sparse + binary relevance + module-wise Consensus3 path, then asks a narrower question:
#
# > Can **current/recent Qwen3-Embedding-8B semantics** and **Top-5 inclusion supervision** improve the organizer-specific binary relevance signal without disturbing the score geometry that already transferred to public LB?
#
# ## Why V17 is different
#
# 1. **Exact Consensus3 is immutable.** No free 17-weight search.
# 2. **Top-5 supervision is not blended directly.** A fresh offline probe showed direct `P(rank<=5)` replacement is harmful. V17 instead uses Top-5 predictions only as nested features inside a new relevance calibrator.
# 3. **Current + recent semantics only.** V16's broad profile/readiness view was highly redundant with the structured champion. V17 removes that view.
# 4. **Frozen Qwen3-Embedding-8B.** The model is used only as a feature extractor. No `.generate()` call exists.
# 5. **Module reliability is explicit.** The public attack family gives special smooth semantic trust only to **M_006** and **M_010**, the two modules with the cleanest cross-representation semantic evidence in V16.
# 6. **Boundary damage is isolated.** Two additional pre-registered candidates keep the champion Top-5 membership fixed so semantic evidence can improve upper ordering without touching the weak #5/#6 boundary.
# 7. **Matched audit is the promotion authority.** A new-seed outer CV rebuilds sparse, old relevance, Top-5, semantic heads, and semantic calibration with the outer validation users completely excluded.
#
# ## Kaggle setup
#
# - Accelerator: **GPU T4 ×2**
# - Internet: **ON** to download `Qwen/Qwen3-Embedding-8B`
# - Run all cells top-to-bottom.
# - `V17_SMOKE=0`, `V17_SKIP_QWEN=0` for the real run.
#
# ## Output policy
#
# The notebook always writes:
#
# - `submission_v17_consensus3_reference.csv`
# - `submission_v17_attack1.csv` — strongest pre-registered candidate under the robust offline ranking
# - `submission_v17_attack2.csv` — second strongest under the offline ranking
# - `submission_v17_stronger_same_family.csv` — same-family escalation for slot #2 if slot #1 moves public upward
# - `submission_v17_diverse_backup.csv` — opposite-family hedge if slot #1 is flat/down
# - `submission_v17_primary.csv` — attack1 only if the strict promotion gate passes, otherwise exact Consensus3
# - `v17_diagnostics.csv`, `v17_summary.json`, `v17_oof_artifacts.npz`
#
# **No notebook can guarantee a public score before Kaggle evaluates hidden labels. Read the final `V17 DECISION` block before submitting.**

# ---- cell 1 (code) ----
# Runtime dependencies. No external task dataset is downloaded.
import os, sys, subprocess
V17_SKIP_INSTALL=bool(int(os.getenv("V17_SKIP_INSTALL","0")))
if not V17_SKIP_INSTALL:
    subprocess.check_call([
        sys.executable,"-m","pip","install","-q","-U",
        "transformers==4.57.6","accelerate==1.14.0","safetensors>=0.5,<1",
        "huggingface_hub>=0.34,<2","lightgbm>=4.5,<5","sentencepiece>=0.2,<1"
    ])
print("dependency setup complete; skip_install=",V17_SKIP_INSTALL)

# ---- cell 2 (code) ----
import os,re,json,ast,html,hashlib,random,warnings,gc,time,math
from pathlib import Path
from collections import defaultdict
os.environ.setdefault("OMP_NUM_THREADS","4")
os.environ.setdefault("OPENBLAS_NUM_THREADS","4")
os.environ.setdefault("MKL_NUM_THREADS","4")
os.environ["PYTHONHASHSEED"]="42"
os.environ["TOKENIZERS_PARALLELISM"]="false"
os.environ.setdefault("PYDEVD_DISABLE_FILE_VALIDATION","1")

import numpy as np, pandas as pd
from IPython.display import display
from scipy.sparse import hstack
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
import lightgbm as lgb
warnings.filterwarnings("ignore")

SEED=42
AUDIT_SEED=20260824
V17_SMOKE=bool(int(os.getenv("V17_SMOKE","0")))
V17_SKIP_QWEN=bool(int(os.getenv("V17_SKIP_QWEN","0")))
N_FOLDS=2 if V17_SMOKE else 5
N_INNER=2 if V17_SMOKE else 3
RIDGE_ALPHA=3.0
REFERENCE_W=0.45
LGB_ESTIMATORS=60 if V17_SMOKE else 1200
LGB_EARLY_STOP=6 if V17_SMOKE else 80
AUDIT_LGB_ESTIMATORS=50 if V17_SMOKE else 750
AUDIT_EARLY_STOP=5 if V17_SMOKE else 55
SEM_LGB_ESTIMATORS=50 if V17_SMOKE else 850
SEM_LGB_EARLY_STOP=5 if V17_SMOKE else 65
BOOTSTRAPS=300 if V17_SMOKE else 5000

PUBLIC_CONSENSUS3_LB=0.65856
PUBLIC_LEADER_LB=0.66082
REQUIRED_PUBLIC_DELTA=PUBLIC_LEADER_LB-PUBLIC_CONSENSUS3_LB

EMBED_MODEL="Qwen/Qwen3-Embedding-8B"
EMBED_REVISION="1d8ad4c"
EMBED_DIM_FULL=4096
EMBED_DIM_SUP=1024
RUNTIME_DIM_FULL=64 if V17_SMOKE else EMBED_DIM_FULL
RUNTIME_DIM_SUP=min(32 if V17_SMOKE else EMBED_DIM_SUP,RUNTIME_DIM_FULL)
CURRENT_MAX_TOKENS=192
RECENT_MAX_TOKENS=384
MODULE_MAX_TOKENS=384
SEM_RIDGE_ALPHA_BIN=18.0
SEM_RIDGE_ALPHA_TOP5=24.0

# Pre-registered attack strengths. These are intentionally tiny and low-dimensional.
TIER1_MODULES=("M_006","M_010")
TIER1_BETAS=(0.25,0.40)
GLOBAL_LOCK_BETA=0.25

# Promotion gates are incremental versus exact Consensus3.
MIN_ORD_DELTA=-0.01 if V17_SMOKE else -0.00015
MIN_AUDIT_DELTA=-0.01 if V17_SMOKE else 0.00015
MIN_CHRONO_DELTA=-0.01 if V17_SMOKE else 0.00010
MIN_AUDIT_WINS=1 if V17_SMOKE else 3
MIN_AUDIT_CI_LOW=-1.0 if V17_SMOKE else -0.00005
MAX_TEST_TOP5_CHANGE=1.0 if V17_SMOKE else 0.06
MAX_TEST_TOP1_CHANGE=1.0 if V17_SMOKE else 0.035

CACHE_ROOT=(Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('/mnt/data'))/'v17_cache'
CACHE_ROOT.mkdir(parents=True,exist_ok=True)
random.seed(SEED); np.random.seed(SEED)
print("V17_SMOKE",V17_SMOKE,"V17_SKIP_QWEN",V17_SKIP_QWEN)
print("public champion",PUBLIC_CONSENSUS3_LB,"leader",PUBLIC_LEADER_LB,"gap",round(REQUIRED_PUBLIC_DELTA,6))
print("Qwen",EMBED_MODEL,"revision",EMBED_REVISION,"dims",RUNTIME_DIM_FULL,RUNTIME_DIM_SUP)

# ---- cell 3 (code) ----
def locate_data_dir():
    required={"chat_history.csv","modules_catalog.csv","user_assessments.csv","train_relevance.csv","test.csv","sample_submission.csv"}
    roots=[Path("/kaggle/input"),Path("/mnt/data/_mine_data"),Path("/mnt/data/v17_data"),Path("/mnt/data/comp_v3"),Path(".")]
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

if V17_SMOKE:
    train=train.iloc[:400].copy(); test=test.iloc[:100].copy()
    sample=sample.set_index("user_id").loc[test.user_id].reset_index()
    keep=set(train.user_id)|set(test.user_id)
    chat=chat[chat.user_id.isin(keep)].copy(); assessments=assessments[assessments.user_id.isin(keep)].copy()

MODULES=list(train.columns[1:]); N_MODULES=len(MODULES); N_TRAIN=len(train); N_TEST=len(test)
assert MODULES==list(sample.columns[1:]) and N_MODULES==17
assert (N_TRAIN,N_TEST)==((400,100) if V17_SMOKE else (4000,1000))
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
word_vec=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_df=.995,sublinear_tf=True,strip_accents="unicode",max_features=(3000 if V17_SMOKE else 50000),dtype=np.float32)
char_vec=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=2,sublinear_tf=True,max_features=(5000 if V17_SMOKE else 60000),dtype=np.float32)
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
# ## Current/recent semantic representation
#
# V17 deliberately drops V16's broad `profile` embedding. Assessment/readiness remains available through the exact structured feature branch, while Qwen is reserved for information that the sparse champion is most likely to miss: **current intent and recent trajectory**.

# ---- cell 9 (code) ----
current_texts=[]; recent_texts=[]
for u in all_users:
    g=user_chat.get(u)
    if g is None or len(g)==0: learn=pd.DataFrame(columns=chat.columns)
    else:
        learn=g[~g.is_noise].copy()
        if len(learn)==0: learn=g.copy()
    rows=list(learn.itertuples(index=False)); current=rows[-1].clean if rows else "No learning chat available."
    recent_rows=rows[-3:]; labels=(['earlier','recent','current'][-len(recent_rows):] if recent_rows else [])
    recent=" ".join(f"[{lab}] {row.clean}" for lab,row in zip(labels,recent_rows)) or current
    current_texts.append(f"Current learner request: {current}")
    recent_texts.append(f"Recent learning conversation in chronological order: {recent}")

mods=modules.set_index("module_id").loc[MODULES].reset_index()
module_content_docs=[]; module_full_docs=[]
for row in mods.itertuples(index=False):
    module_content_docs.append(f"Course module: {row.module_name}. Learning content: {row.description_and_syllabus}.")
    module_full_docs.append(f"Course module: {row.module_name}. Learning content: {row.description_and_syllabus}. Prerequisite: {row.prerequisite_level}.")
CURRENT_INSTRUCTION=("Instruct: Retrieve the Intelligo course module that best matches the learner's CURRENT explicitly stated learning intent. Prefer the newest goal when it conflicts with older goals.\nQuery: ")
RECENT_INSTRUCTION=("Instruct: Retrieve the best NEXT Intelligo course module from the learner's recent learning trajectory. Give more importance to newer messages.\nQuery: ")
assert len(current_texts)==len(recent_texts)==len(all_users)
print(current_texts[1][:400]); print(recent_texts[1][:600]); print(module_full_docs[0])

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
    if qwen_model is not None or V17_SMOKE or V17_SKIP_QWEN: return
    import torch
    from transformers import AutoTokenizer,AutoModel
    if not torch.cuda.is_available() or torch.cuda.device_count()<2:
        raise RuntimeError("Full V17 requires Kaggle GPU T4 x2. Select Accelerator: GPU T4 x2.")
    for i in range(torch.cuda.device_count()):
        p=torch.cuda.get_device_properties(i); print(f"GPU {i}: {p.name}, {p.total_memory/1024**3:.1f} GiB")
    qwen_tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL,revision=EMBED_REVISION,padding_side='left')
    qwen_model=AutoModel.from_pretrained(EMBED_MODEL,revision=EMBED_REVISION,torch_dtype=torch.float16,
        device_map='balanced',max_memory={0:'13GiB',1:'13GiB','cpu':'24GiB'},low_cpu_mem_usage=True,attn_implementation='sdpa')
    qwen_model.eval(); qwen_model.config.use_cache=False; qwen_input_device=qwen_model.get_input_embeddings().weight.device
    print("hf_device_map",qwen_model.hf_device_map,"input",qwen_input_device)

def encode_cached(name,texts,instruction,max_length,start_batch):
    key=_text_hash(texts,instruction); path=CACHE_ROOT/f"{name}_{key}_{RUNTIME_DIM_FULL}.npy"
    if path.exists():
        arr=np.load(path).astype(np.float32)
        if arr.shape==(len(texts),RUNTIME_DIM_FULL): print("loaded cache",path.name); return arr
    if V17_SMOKE or V17_SKIP_QWEN:
        arr=_fake_embeddings([instruction+t for t in texts],RUNTIME_DIM_FULL,'v17-'+name); np.save(path,arr.astype(np.float16)); return arr
    import torch
    load_qwen(); prepared=[instruction+str(t) for t in texts]; result=np.empty((len(prepared),RUNTIME_DIM_FULL),np.float32); pos=0; bs=start_batch
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

current_emb=encode_cached('current',current_texts,CURRENT_INSTRUCTION,CURRENT_MAX_TOKENS,8)
recent_emb=encode_cached('recent',recent_texts,RECENT_INSTRUCTION,RECENT_MAX_TOKENS,6)
content_emb=encode_cached('module_content',module_content_docs,'',MODULE_MAX_TOKENS,17)
full_emb=encode_cached('module_full',module_full_docs,'',MODULE_MAX_TOKENS,17)
if qwen_model is not None:
    del qwen_model,qwen_tokenizer; qwen_model=qwen_tokenizer=None
    import torch; torch.cuda.empty_cache(); gc.collect()
for name,arr in [('current',current_emb),('recent',recent_emb),('content',content_emb),('full',full_emb)]:
    assert arr.shape[1]==RUNTIME_DIM_FULL and np.isfinite(arr).all(),(name,arr.shape)
print("Qwen embedding shapes",current_emb.shape,recent_emb.shape,content_emb.shape)

# ---- cell 11 (code) ----
def row_z(x):
    x=np.asarray(x,np.float32); return ((x-x.mean(1,keepdims=True))/np.maximum(x.std(1,keepdims=True),1e-6)).astype(np.float32)
def row_rank_score(x):
    order=np.argsort(-x,axis=1); ranks=np.empty_like(order); ranks[np.arange(len(x))[:,None],order]=np.arange(N_MODULES)[None,:]
    return (1.0-ranks.astype(np.float32)/(N_MODULES-1)).astype(np.float32)
def trunc_norm(x,d): return _normalize_rows(np.asarray(x[:,:d],np.float32))
SIM_MATRICES={
    'current_content':np.asarray(current_emb@content_emb.T,np.float32),
    'current_full':np.asarray(current_emb@full_emb.T,np.float32),
    'recent_content':np.asarray(recent_emb@content_emb.T,np.float32),
    'recent_full':np.asarray(recent_emb@full_emb.T,np.float32),
}
SEM_PAIR=np.concatenate([z[:,:,None] for mat in SIM_MATRICES.values() for z in (mat,row_z(mat),row_rank_score(mat))],axis=2).astype(np.float32)
CURRENT_SUP=trunc_norm(current_emb,RUNTIME_DIM_SUP); RECENT_SUP=trunc_norm(recent_emb,RUNTIME_DIM_SUP)
sem_zero=(.45*SIM_MATRICES['current_full']+.35*SIM_MATRICES['recent_full']+.10*SIM_MATRICES['current_content']+.10*SIM_MATRICES['recent_content']).astype(np.float32)
print("SEM_PAIR",SEM_PAIR.shape,"zero-shot",round(ndcg5(y,sem_zero[:N_TRAIN]),6))

# ---- cell 12 (markdown) ----
# ## Top5-aware semantic calibrator
#
# `Y_TOP5` enters V17 in two ways: a structured LightGBM `P(top5)` feature and current/recent supervised semantic Ridge heads. The final target is still binary module relevance because that is the only target family with repeated public transfer. This keeps Top-5 information as **context**, not as a direct score replacement.

# ---- cell 13 (code) ----
def semantic_ridge_features(global_train,global_out,folds_local):
    global_train=np.asarray(global_train,int); global_out=np.asarray(global_out,int); n=len(global_train); no=len(global_out)
    trf=np.zeros((n,N_MODULES,4),np.float32); outf=np.zeros((no,N_MODULES,4),np.float32)
    for vi,V in enumerate((CURRENT_SUP,RECENT_SUP)):
        ob=np.zeros((no,N_MODULES),np.float32); ot=np.zeros_like(ob)
        for lf,(lt,lv) in enumerate(folds_local):
            gtr=global_train[lt]; gva=global_train[lv]
            mb=Ridge(alpha=SEM_RIDGE_ALPHA_BIN,solver='lsqr',tol=2e-3,max_iter=1200)
            mt=Ridge(alpha=SEM_RIDGE_ALPHA_TOP5,solver='lsqr',tol=2e-3,max_iter=1200)
            mb.fit(V[gtr],Y_BIN[gtr]); mt.fit(V[gtr],Y_TOP5[gtr])
            trf[lv,:,2*vi]=mb.predict(V[gva]).astype(np.float32); trf[lv,:,2*vi+1]=mt.predict(V[gva]).astype(np.float32)
            ob+=mb.predict(V[global_out]).astype(np.float32)/len(folds_local); ot+=mt.predict(V[global_out]).astype(np.float32)/len(folds_local)
        outf[:,:,2*vi]=ob; outf[:,:,2*vi+1]=ot
    return trf,outf

def make_sem_cal(seed,n_estimators=None):
    n=int(SEM_LGB_ESTIMATORS if n_estimators is None else n_estimators)
    return lgb.LGBMClassifier(n_estimators=n,learning_rate=.025,num_leaves=11,min_child_samples=85,
        reg_alpha=1.0,reg_lambda=8.0,colsample_bytree=.72,random_state=int(seed),n_jobs=4,verbosity=-1,force_col_wise=True)

def sem_design(global_idx,base,old,p5,sem_head):
    global_idx=np.asarray(global_idx,int); rows=[]
    for m in range(N_MODULES):
        rows.append(np.column_stack([
            base[:,m],old[:,m],p5[:,m],USER_FEATURES[global_idx],sem_head[:,m,:],SEM_PAIR[global_idx,m,:]
        ]).astype(np.float32))
    return rows

def fit_sem_meta(global_train,global_out,folds_local,base_tr,base_out,old_tr,old_out,p5_tr,p5_out,sem_head_tr,sem_head_out,seed):
    Xtr=sem_design(global_train,base_tr,old_tr,p5_tr,sem_head_tr); Xo=sem_design(global_out,base_out,old_out,p5_out,sem_head_out)
    oo=np.zeros((len(global_train),N_MODULES),np.float32); out=np.zeros((len(global_out),N_MODULES),np.float32)
    for m in range(N_MODULES):
        yy=Y_BIN[np.asarray(global_train),m].astype(np.int32); its=[]
        for f,(lt,lv) in enumerate(folds_local):
            clf=make_sem_cal(seed+31*m+f); clf.fit(Xtr[m][lt],yy[lt],eval_set=[(Xtr[m][lv],yy[lv])],callbacks=[lgb.early_stopping(SEM_LGB_EARLY_STOP,verbose=False)])
            it=int(clf.best_iteration_ or SEM_LGB_ESTIMATORS); its.append(it); oo[lv,m]=clf.predict_proba(Xtr[m][lv],num_iteration=it)[:,1].astype(np.float32)
            out[:,m]+=clf.predict_proba(Xo[m],num_iteration=it)[:,1].astype(np.float32)/len(folds_local)
    return oo,out

TRAIN_GLOBAL=np.arange(N_TRAIN); TEST_GLOBAL=np.arange(N_TRAIN,N_TRAIN+N_TEST)
sem_head_oof,sem_head_test=semantic_ridge_features(TRAIN_GLOBAL,TEST_GLOBAL,ref_folds)
sem_meta_oof,sem_meta_test=fit_sem_meta(TRAIN_GLOBAL,TEST_GLOBAL,ref_folds,base_oof,base_test,meta_oof,meta_test_foldbag,p5_oof,p5_test_foldbag,sem_head_oof,sem_head_test,SEED+17000)
print("new semantic meta OOF",round(ndcg5(y,sem_meta_oof),6),"old meta",round(ndcg5(y,meta_oof),6))

# ---- cell 14 (code) ----
# Distribution-preserving semantic transport. This preserves the old meta marginal scale.
def quantile_align(source,target):
    source=np.asarray(source,float); target=np.asarray(target,float); order=np.argsort(source,kind='mergesort'); out=np.empty_like(source,dtype=np.float32)
    out[order]=np.sort(target).astype(np.float32); return out

def align_matrix(source,target,group_id=None):
    out=np.zeros_like(source,np.float32)
    if group_id is None:
        for m in range(source.shape[1]): out[:,m]=quantile_align(source[:,m],target[:,m])
    else:
        for g in np.unique(group_id):
            ix=np.where(group_id==g)[0]
            for m in range(source.shape[1]): out[ix,m]=quantile_align(source[ix,m],target[ix,m])
    return out

def module_semantic_candidate(base,old,new_aligned,beta,modules_to_change=TIER1_MODULES):
    mm=old.copy()
    for mid in modules_to_change:
        m=MODULES.index(mid); mm[:,m]=old[:,m]+float(beta)*(new_aligned[:,m]-old[:,m])
    return apply_consensus(base,mm)

def global_semantic_candidate(base,old,new_aligned,beta):
    return apply_consensus(base,old+float(beta)*(new_aligned-old))

def membership_lock(anchor,proposal,lock_rank5=False):
    # Keep exact champion Top-5 membership and exact champion score levels.
    out=anchor.copy(); at=top5_indices(anchor)
    for i in range(len(anchor)):
        members=at[i]
        if lock_rank5:
            movable=members[:4]; levels=np.sort(anchor[i,movable])[::-1]; order=movable[np.argsort(-proposal[i,movable])]
            out[i,order]=levels
        else:
            levels=np.sort(anchor[i,members])[::-1]; order=members[np.argsort(-proposal[i,members])]
            out[i,order]=levels
    return out.astype(np.float32)

aligned_sem_oof=align_matrix(sem_meta_oof,meta_oof,ref_fold_id)
aligned_sem_test=align_matrix(sem_meta_test,meta_test,None)
ordinary_candidates={}
for beta in TIER1_BETAS:
    ordinary_candidates[f"tier1_{int(beta*100):02d}"]=module_semantic_candidate(base_oof,meta_oof,aligned_sem_oof,beta)
prop=global_semantic_candidate(base_oof,meta_oof,aligned_sem_oof,GLOBAL_LOCK_BETA)
ordinary_candidates['global_membership_lock']=membership_lock(consensus3_oof,prop,False)
ordinary_candidates['global_top4_lock']=membership_lock(consensus3_oof,prop,True)
for k,v in ordinary_candidates.items(): print(k,round(ndcg5(y,v)-ndcg5(y,consensus3_oof),6),round(ndcg5(y,v),6))

# ---- cell 15 (markdown) ----
# ## New-seed fully outer-isolated matched audit
#
# The next section is the main safeguard. Every audit validation user is excluded from all supervised first-stage and second-stage fitting. The sparse vocabulary and frozen Qwen embeddings are label-independent and may be shared; all supervised predictions are rebuilt inside the audit outer-training population.

# ---- cell 16 (code) ----
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

def build_isolated_representation(global_train,global_out,seed):
    global_train=np.asarray(global_train,int); global_out=np.asarray(global_out,int); folds_local=make_local_folds(global_train,N_INNER,seed)
    btr,bout=sparse_matched(global_train,global_out,folds_local)
    otr,oout=structured_matched(global_train,global_out,folds_local,btr,bout,Y_BIN,seed+1000)
    ptr,pout=structured_matched(global_train,global_out,folds_local,btr,bout,Y_TOP5,seed+3000)
    htr,hout=semantic_ridge_features(global_train,global_out,folds_local)
    str_,sout=fit_sem_meta(global_train,global_out,folds_local,btr,bout,otr,oout,ptr,pout,htr,hout,seed+5000)
    return btr,bout,otr,oout,ptr,pout,str_,sout

audit_skf=StratifiedKFold(N_FOLDS,shuffle=True,random_state=AUDIT_SEED)
audit_folds=list(audit_skf.split(np.zeros(N_TRAIN),strata)); audit_fold_id=np.full(N_TRAIN,-1,int)
audit_anchor=np.zeros_like(y,np.float32); audit_sem=np.zeros_like(y,np.float32); audit_base=np.zeros_like(y,np.float32); audit_old=np.zeros_like(y,np.float32)
for of,(otr,ova) in enumerate(audit_folds):
    print(f"AUDIT OUTER {of+1}/{N_FOLDS}")
    btr,bva,otr_old,ova_old,ptr,pva,str_,sva=build_isolated_representation(otr,ova,AUDIT_SEED+100*of)
    audit_base[ova]=bva; audit_old[ova]=ova_old; audit_sem[ova]=sva; audit_anchor[ova]=apply_consensus(bva,ova_old); audit_fold_id[ova]=of
    gc.collect()
assert (audit_fold_id>=0).all()
audit_aligned=align_matrix(audit_sem,audit_old,audit_fold_id)
audit_candidates={}
for beta in TIER1_BETAS: audit_candidates[f"tier1_{int(beta*100):02d}"]=module_semantic_candidate(audit_base,audit_old,audit_aligned,beta)
prop=global_semantic_candidate(audit_base,audit_old,audit_aligned,GLOBAL_LOCK_BETA)
audit_candidates['global_membership_lock']=membership_lock(audit_anchor,prop,False)
audit_candidates['global_top4_lock']=membership_lock(audit_anchor,prop,True)
print("matched audit anchor",round(ndcg5(y,audit_anchor),6))
for k,v in audit_candidates.items(): print(k,round(ndcg5(y,v)-ndcg5(y,audit_anchor),6))

# ---- cell 17 (markdown) ----
# ## Future-like chronological audit
#
# Test IDs follow the train IDs. V16 semantics were strongest on later users, so V17 retains one clean future-like audit: train on the first 75% of labeled users, validate on the last 25%. This is not allowed to override a negative matched audit, but it prevents discarding a real future-generalization signal.

# ---- cell 18 (code) ----
chrono_cut=int(round(N_TRAIN*.75)); chrono_tr=np.arange(chrono_cut); chrono_va=np.arange(chrono_cut,N_TRAIN)
cbtr,cbva,cotr,cova,cptr,cpva,cstr,csva=build_isolated_representation(chrono_tr,chrono_va,AUDIT_SEED+9000)
chrono_anchor=apply_consensus(cbva,cova); chrono_aligned=align_matrix(csva,cova,None); chrono_candidates={}
for beta in TIER1_BETAS: chrono_candidates[f"tier1_{int(beta*100):02d}"]=module_semantic_candidate(cbva,cova,chrono_aligned,beta)
prop=global_semantic_candidate(cbva,cova,chrono_aligned,GLOBAL_LOCK_BETA)
chrono_candidates['global_membership_lock']=membership_lock(chrono_anchor,prop,False)
chrono_candidates['global_top4_lock']=membership_lock(chrono_anchor,prop,True)
print("chrono anchor",round(ndcg5(y[chrono_va],chrono_anchor),6),"n",len(chrono_va))
for k,v in chrono_candidates.items(): print(k,round(ndcg5(y[chrono_va],v)-ndcg5(y[chrono_va],chrono_anchor),6))

# ---- cell 19 (code) ----
# Final test candidates use the full public-winning anchor and fold-matched semantic test expert.
test_candidates={}
for beta in TIER1_BETAS: test_candidates[f"tier1_{int(beta*100):02d}"]=module_semantic_candidate(base_test,meta_test,aligned_sem_test,beta)
prop_test=global_semantic_candidate(base_test,meta_test,aligned_sem_test,GLOBAL_LOCK_BETA)
test_candidates['global_membership_lock']=membership_lock(consensus3_test,prop_test,False)
test_candidates['global_top4_lock']=membership_lock(consensus3_test,prop_test,True)

ord_anchor_u=per_user_ndcg(y,consensus3_oof); aud_anchor_u=per_user_ndcg(y,audit_anchor); chr_anchor_u=per_user_ndcg(y[chrono_va],chrono_anchor)
rows=[]
for i,name in enumerate(ordinary_candidates):
    po=ordinary_candidates[name]; pa=audit_candidates[name]; pc=chrono_candidates[name]; pt=test_candidates[name]
    du_o=per_user_ndcg(y,po)-ord_anchor_u; du_a=per_user_ndcg(y,pa)-aud_anchor_u; du_c=per_user_ndcg(y[chrono_va],pc)-chr_anchor_u
    ord_f=np.asarray([du_o[ref_fold_id==f].mean() for f in range(N_FOLDS)]); aud_f=np.asarray([du_a[audit_fold_id==f].mean() for f in range(N_FOLDS)])
    ci=paired_boot(du_a,AUDIT_SEED+777+i)
    top5=top5_set_change_rate(pt,consensus3_test); order=top5_order_change_rate(pt,consensus3_test); top1=top1_change_rate(pt,consensus3_test)
    # Robust ranking emphasizes untouched audit and future-like evidence, while punishing regime reversal.
    min_core=min(float(du_a.mean()),float(du_c.mean()))
    robust=float(.20*du_o.mean()+.50*du_a.mean()+.30*du_c.mean()+.35*min(0.0,min_core))
    gate=bool(du_o.mean()>=MIN_ORD_DELTA and du_a.mean()>=MIN_AUDIT_DELTA and du_c.mean()>=MIN_CHRONO_DELTA and int((aud_f>0).sum())>=MIN_AUDIT_WINS and ci[0]>=MIN_AUDIT_CI_LOW and top5<=MAX_TEST_TOP5_CHANGE and top1<=MAX_TEST_TOP1_CHANGE)
    rows.append(dict(candidate=name,ordinary_delta=float(du_o.mean()),ordinary_fold_wins=int((ord_f>0).sum()),ordinary_folds=json.dumps(ord_f.tolist()),audit_delta=float(du_a.mean()),audit_fold_wins=int((aud_f>0).sum()),audit_folds=json.dumps(aud_f.tolist()),audit_ci_low=ci[0],audit_ci_high=ci[1],chrono_delta=float(du_c.mean()),test_top5_set_change=top5,test_top5_order_change=order,test_top1_change=top1,robust_score=robust,promotion_gate=gate))

diagnostics=pd.DataFrame(rows).sort_values(['promotion_gate','robust_score','audit_delta'],ascending=[False,False,False]).reset_index(drop=True)
display(diagnostics.round(6))

ranked=diagnostics.candidate.tolist(); attack1_name=ranked[0]; attack2_name=ranked[1]
attack1_test=test_candidates[attack1_name]; attack2_test=test_candidates[attack2_name]
# Precompute both a stronger same-family option and a diverse-family backup so slot #2 can be chosen after seeing slot #1 public feedback without rerunning V17.
if attack1_name.startswith('tier1'):
    same_family=[n for n in ranked if n.startswith('tier1') and n!=attack1_name]
    diverse_family=[n for n in ranked if n.startswith('global_')]
else:
    same_family=[n for n in ranked if n.startswith('global_') and n!=attack1_name]
    diverse_family=[n for n in ranked if n.startswith('tier1')]
stronger_same_name=(same_family[0] if same_family else attack2_name)
diverse_name=(diverse_family[0] if diverse_family else attack2_name)
stronger_same_test=test_candidates[stronger_same_name]; diverse_test=test_candidates[diverse_name]
attack1_gate=bool(diagnostics.loc[0,'promotion_gate'])
primary_test=attack1_test.copy() if attack1_gate else consensus3_test.copy(); primary_label=attack1_name if attack1_gate else 'consensus3_fallback'
print("\n================ V17 DECISION ================")
print("ordinary Consensus3",round(ndcg5(y,consensus3_oof),6))
print("matched audit anchor",round(ndcg5(y,audit_anchor),6))
print("chronological anchor",round(ndcg5(y[chrono_va],chrono_anchor),6))
print("ATTACK 1",attack1_name,"gate",attack1_gate)
print("ATTACK 2 ranked",attack2_name)
print("STRONGER SAME FAMILY",stronger_same_name)
print("DIVERSE BACKUP",diverse_name)
print("PRIMARY",primary_label)
print("Required public improvement to beat leader:",round(REQUIRED_PUBLIC_DELTA,6))
print("Do not infer a guaranteed public score from local CV. If attack1 fails the gate, primary intentionally remains exact Consensus3.")
print("================================================")

# ---- cell 20 (code) ----
def save_submission(scores,name):
    out=sample.copy(); out.loc[:,MODULES]=scores
    root=Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('/mnt/data'); path=root/name; out.to_csv(path,index=False)
    chk=pd.read_csv(path); assert chk.shape==sample.shape and chk.user_id.tolist()==sample.user_id.tolist() and chk.user_id.is_unique
    assert chk[MODULES].notna().all().all() and np.isfinite(chk[MODULES].to_numpy()).all(); print("saved",path); return path
root=Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('/mnt/data')
paths={}
paths['consensus3']=save_submission(consensus3_test,'submission_v17_consensus3_reference.csv')
paths['attack1']=save_submission(attack1_test,'submission_v17_attack1.csv')
paths['attack2']=save_submission(attack2_test,'submission_v17_attack2.csv')
paths['stronger_same']=save_submission(stronger_same_test,'submission_v17_stronger_same_family.csv')
paths['diverse_backup']=save_submission(diverse_test,'submission_v17_diverse_backup.csv')
paths['primary']=save_submission(primary_test,'submission_v17_primary.csv')
diagnostics.to_csv(root/'v17_diagnostics.csv',index=False)
np.savez_compressed(root/'v17_oof_artifacts.npz',y=y,base_oof=base_oof,meta_oof=meta_oof,p5_oof=p5_oof,sem_meta_oof=sem_meta_oof,consensus3_oof=consensus3_oof,audit_anchor=audit_anchor,audit_old=audit_old,audit_sem=audit_sem,base_test=base_test,meta_test=meta_test,p5_test=p5_test,p5_test_foldbag=p5_test_foldbag,meta_test_foldbag=meta_test_foldbag,sem_meta_test=sem_meta_test,consensus3_test=consensus3_test,ref_fold_id=ref_fold_id,audit_fold_id=audit_fold_id)
summary={
    'public_consensus3_lb':PUBLIC_CONSENSUS3_LB,'public_leader_lb':PUBLIC_LEADER_LB,'required_public_delta':REQUIRED_PUBLIC_DELTA,
    'model':EMBED_MODEL,'revision':EMBED_REVISION,'semantic_views':['current','recent'],'tier1_modules':list(TIER1_MODULES),
    'diagnostics':diagnostics.to_dict(orient='records'),'attack1':attack1_name,'attack2':attack2_name,'stronger_same_family':stronger_same_name,'diverse_backup':diverse_name,'primary':primary_label,
    'files':{k:v.name for k,v in paths.items()},'compliance':{'task_specific_data':'organizer files only','pretrained_usage':'frozen embedding feature extractor only','generation_used':False,'direct_test_label_request':False}
}
with open(root/'v17_summary.json','w') as f: json.dump(summary,f,indent=2)
print("saved diagnostics, summary, OOF artifacts")

# ---- cell 21 (code) ----
# Hard invariants: fail loudly instead of silently producing a malformed or non-reproducible candidate.
assert len(MODULES)==17 and y.shape==(N_TRAIN,17)
assert CONSENSUS3_OVERRIDES=={"M_002":0.44,"M_006":0.55,"M_007":0.60,"M_010":0.40,"M_011":0.55,"M_012":0.55,"M_015":0.44,"M_017":0.53}
assert np.allclose(consensus3_oof,apply_consensus(base_oof,meta_oof),atol=0,rtol=0)
assert np.allclose(consensus3_test,apply_consensus(base_test,meta_test),atol=0,rtol=0)
for arr in [base_oof,meta_oof,p5_oof,sem_meta_oof,consensus3_oof,audit_anchor,audit_old,audit_sem,base_test,meta_test,p5_test,p5_test_foldbag,meta_test_foldbag,sem_meta_test,consensus3_test,attack1_test,attack2_test,stronger_same_test,diverse_test,primary_test]: assert np.isfinite(arr).all()
assert diagnostics.candidate.nunique()==4 and set(diagnostics.candidate)=={'tier1_25','tier1_40','global_membership_lock','global_top4_lock'}
assert top5_set_change_rate(test_candidates['global_membership_lock'],consensus3_test)==0.0
assert top5_set_change_rate(test_candidates['global_top4_lock'],consensus3_test)==0.0
if not V17_SMOKE:
    assert abs(ndcg5(y,consensus3_oof)-0.6643682395561207)<0.0012,ndcg5(y,consensus3_oof)
if not attack1_gate: assert np.allclose(primary_test,consensus3_test,atol=0,rtol=0)
print("V17 hard invariants: PASS")
