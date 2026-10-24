# Converted from notebooks/v20.ipynb: markdown cells become comments, %/! magics neutralized.

# ---- cell 0 (markdown) ----
# # MineToday V20 — Distribution-Matched Contrastive Late-Interaction Recommender
#
# **Goal:** improve on the public-validated Consensus3 anchor (`0.65856`) by improving the *embedding decision space*, not by globally replacing the champion.
#
# V20 implements the conclusions from V16–V19:
#
# - exact V12.2/Consensus3 remains the immutable anchor;
# - Qwen3-Embedding-8B is still the primary high-capacity semantic backbone, but V20 learns a **small residual metric adapter** on organizer labels using graded multi-positive/listwise + pairwise hard-negative objectives;
# - messages remain separate and are combined with recency-aware late interaction;
# - BGE-M3 provides an orthogonal **dense + lexical + ColBERT-style multi-vector** signal;
# - train/test covariate shift is measured by adversarial validation and may importance-weight semantic training;
# - one **untouched 20% lockbox** is opened only after the development architecture/candidate is frozen;
# - final semantic intervention is selective (4% or 6% strongest evidence), not a global blend;
# - no-chat users are always exact Consensus3;
# - `submission_v20_primary.csv` is **Consensus3 unless every promotion gate passes**.
#
# No external task data are used. The only labels are `train_relevance.csv`. Pretrained open-source models are used only as representation models.

# ---- cell 1 (code) ----
import os,sys,subprocess
V20_SKIP_INSTALL=bool(int(os.getenv("V20_SKIP_INSTALL","0")))
if not V20_SKIP_INSTALL:
    subprocess.check_call([
        sys.executable,"-m","pip","install","-q","-U",
        "transformers==4.57.6","accelerate==1.14.0","safetensors>=0.5,<1",
        "huggingface_hub>=0.34,<2","lightgbm>=4.5,<5","sentencepiece>=0.2,<1",
        "FlagEmbedding>=1.3.5,<2","sentence-transformers>=3.4,<4"
    ])
print("dependency setup complete; skip_install=",V20_SKIP_INSTALL)

# ---- cell 2 (code) ----
import os,re,json,ast,html,hashlib,random,warnings,gc,time,math
from pathlib import Path
from collections import defaultdict,Counter
os.environ.setdefault("OMP_NUM_THREADS","4"); os.environ.setdefault("OPENBLAS_NUM_THREADS","4"); os.environ.setdefault("MKL_NUM_THREADS","4")
os.environ["PYTHONHASHSEED"]="42"; os.environ["TOKENIZERS_PARALLELISM"]="false"; os.environ.setdefault("PYDEVD_DISABLE_FILE_VALIDATION","1")

import numpy as np,pandas as pd
from IPython.display import display
from scipy.sparse import hstack
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
warnings.filterwarnings("ignore")

SEED=42
LOCKBOX_SEED=20260825
ADV_SEED=2026082501
V20_SMOKE=bool(int(os.getenv("V20_SMOKE","0")))
V20_SKIP_QWEN=bool(int(os.getenv("V20_SKIP_QWEN","0")))
V20_SKIP_BGE=bool(int(os.getenv("V20_SKIP_BGE","0")))
N_FOLDS=2 if V20_SMOKE else 5
N_INNER=2 if V20_SMOKE else 4
DEV_SEM_FOLDS=2 if V20_SMOKE else 4
RIDGE_ALPHA=3.0
REFERENCE_W=0.45
LGB_ESTIMATORS=60 if V20_SMOKE else 1200
LGB_EARLY_STOP=6 if V20_SMOKE else 80
AUDIT_LGB_ESTIMATORS=45 if V20_SMOKE else 700
AUDIT_EARLY_STOP=5 if V20_SMOKE else 50
BOOTSTRAPS=300 if V20_SMOKE else 8000

PUBLIC_CONSENSUS3_LB=0.65856
PUBLIC_LEADER_LB=0.66082
PUBLIC_V18_LB=0.65829
PUBLIC_V19_LB=0.65791

QWEN_MODEL="Qwen/Qwen3-Embedding-8B"
QWEN_REVISION="1d8ad4c"
QWEN_DIM=96 if V20_SMOKE else 4096
QWEN_ADAPTER_DIM=min(48 if V20_SMOKE else 256,QWEN_DIM)
QWEN_BOTTLENECK=min(24 if V20_SMOKE else 64,QWEN_ADAPTER_DIM)
QWEN_RESIDUAL_SCALE=.20
MAX_USER_MESSAGES=3 if V20_SMOKE else 4
MESSAGE_MAX_TOKENS=160 if V20_SMOKE else 192
MODULE_MAX_TOKENS=256 if V20_SMOKE else 448
ADAPTER_EPOCHS=2 if V20_SMOKE else 28
ADAPTER_BATCH=48 if V20_SMOKE else 64
ADAPTER_LR=1.2e-3
ADAPTER_WD=.02
AGG_TEMP=.07
LIST_TEMP=.08
PAIR_TEMP=.08

BGE_MODEL="BAAI/bge-m3"
BGE_QUERY_MAXLEN=96 if V20_SMOKE else 160
BGE_DOC_MAXLEN=192 if V20_SMOKE else 320
BGE_BATCH=16
BGE_WEIGHTS=np.asarray([.40,.20,.40],np.float32)  # dense / lexical / ColBERT

# Predeclared selective intervention candidates. Development chooses between these two only.
CANDIDATES={"selective4":.04,"selective6":.06}
SEMANTIC_BETA=.38
QWEN_SHARE=.75
BGE_SHARE=.25

# Promotion gate: unlike V18/V19, fail-closed means do not submit the attack.
MIN_DEV_DELTA=-.01 if V20_SMOKE else .00045
MIN_DEV_WEIGHTED_DELTA=-.01 if V20_SMOKE else .00035
MIN_DEV_WINS=1 if V20_SMOKE else 3
MIN_LOCK_DELTA=-.01 if V20_SMOKE else .00020
MIN_LOCK_CI_LOW=-1.0 if V20_SMOKE else -.00120
MIN_REPEAT_GROUP_DELTA=-1.0 if V20_SMOKE else -.00060
MIN_NOVEL_GROUP_DELTA=-1.0 if V20_SMOKE else -.00120
MIN_TEST_SET_CHANGE=0.0 if V20_SMOKE else .010
MAX_TEST_SET_CHANGE=1.0 if V20_SMOKE else .075
MAX_TEST_ORDER_CHANGE=1.0 if V20_SMOKE else .150
MAX_TEST_TOP1_CHANGE=1.0 if V20_SMOKE else .018

CACHE_ROOT=(Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('/mnt/data'))/'v20_cache'
CACHE_ROOT.mkdir(parents=True,exist_ok=True)
random.seed(SEED); np.random.seed(SEED)
print("V20_SMOKE",V20_SMOKE,"skip_qwen",V20_SKIP_QWEN,"skip_bge",V20_SKIP_BGE)
print("public anchor",PUBLIC_CONSENSUS3_LB,"leader",PUBLIC_LEADER_LB,"gap",round(PUBLIC_LEADER_LB-PUBLIC_CONSENSUS3_LB,6))

# ---- cell 3 (code) ----
def locate_data_dir():
    required={"chat_history.csv","modules_catalog.csv","user_assessments.csv","train_relevance.csv","test.csv","sample_submission.csv"}
    roots=[Path("/kaggle/input"),Path("/mnt/data/_mine_data"),Path("/mnt/data/v19_data"),Path("/mnt/data/comp_v3"),Path(".")]
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

if V20_SMOKE:
    train=train.iloc[:400].copy(); test=test.iloc[:100].copy()
    sample=sample.set_index("user_id").loc[test.user_id].reset_index()
    keep=set(train.user_id)|set(test.user_id)
    chat=chat[chat.user_id.isin(keep)].copy(); assessments=assessments[assessments.user_id.isin(keep)].copy()

MODULES=list(train.columns[1:]); N_MODULES=len(MODULES); N_TRAIN=len(train); N_TEST=len(test)
assert MODULES==list(sample.columns[1:]) and N_MODULES==17
assert (N_TRAIN,N_TEST)==((400,100) if V20_SMOKE else (4000,1000))
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
word_vec=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_df=.995,sublinear_tf=True,strip_accents="unicode",max_features=(3000 if V20_SMOKE else 50000),dtype=np.float32)
char_vec=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=2,sublinear_tf=True,max_features=(5000 if V20_SMOKE else 60000),dtype=np.float32)
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
# ## Honest validation: adversarial train/test shift + untouched lockbox
#
# The public leaderboard feedback from V18/V19 showed that repeated random-CV research can become over-adapted. V20 therefore uses two independent safeguards:
#
# 1. **Adversarial validation** estimates how different train users are from test users using target-free structured + sparse-SVD features. If the domain AUC is meaningfully above chance, semantic training uses clipped square-root importance weights.
# 2. **Lockbox**: 20% of labeled users are removed before semantic candidate development. The chosen candidate is evaluated once on this lockbox. It cannot be promoted without a positive lockbox result.

# ---- cell 9 (code) ----
# Target-free adversarial validation. SVD is fit on organizer-provided train+test text only; no relevance labels are used.
ADV_DIM=16 if V20_SMOKE else 64
svd=TruncatedSVD(n_components=ADV_DIM,random_state=ADV_SEED)
X_svd=svd.fit_transform(X).astype(np.float32)
ADV_X=np.column_stack([USER_FEATURES,X_svd]).astype(np.float32)
domain_y=np.r_[np.zeros(N_TRAIN,np.int8),np.ones(N_TEST,np.int8)]
domain_oof=np.zeros(len(all_users),np.float32)
adv_skf=StratifiedKFold(3 if V20_SMOKE else 5,shuffle=True,random_state=ADV_SEED)
for tr,va in adv_skf.split(ADV_X,domain_y):
    mdl=make_pipeline(StandardScaler(),LogisticRegression(C=.35,max_iter=1000,solver='liblinear',random_state=ADV_SEED))
    mdl.fit(ADV_X[tr],domain_y[tr]); domain_oof[va]=mdl.predict_proba(ADV_X[va])[:,1]
ADV_AUC=float(roc_auc_score(domain_y,domain_oof))
p=np.clip(domain_oof[:N_TRAIN],.02,.98)
raw_w=(p/(1-p))*(N_TRAIN/max(N_TEST,1))
shift_w=np.sqrt(np.clip(raw_w,.25,4.0)).astype(np.float32)
shift_w/=shift_w.mean()
if ADV_AUC < .58: shift_w[:]=1.0
print("adversarial train/test AUC",round(ADV_AUC,5),"importance weighting",bool(ADV_AUC>=.58))
print("train weight quantiles",np.quantile(shift_w,[0,.1,.5,.9,1]).round(3).tolist())

# Lockbox is never used for candidate/strength selection.
lock_size=80 if V20_SMOKE else 800
if V20_SMOKE:
    _rng=np.random.default_rng(LOCKBOX_SEED); _perm=_rng.permutation(N_TRAIN)
    LOCK_IDX=np.sort(_perm[:lock_size]); DEV_IDX=np.sort(_perm[lock_size:])
else:
    lock_strata=np.asarray(strata,dtype=object).copy()
    _counts=Counter(lock_strata.tolist())
    # Full data has ample support; collapse only truly tiny strata defensively.
    rare={k for k,v in _counts.items() if v<3}
    if rare:
        lock_strata=np.asarray([s if s not in rare else "__rare__" for s in lock_strata],dtype=object)
    # If the collapsed rare bucket is itself too small, fall back to the uncollapsed top-module strata.
    _c2=Counter(lock_strata.tolist())
    if min(_c2.values())<2: lock_strata=np.asarray(strata,dtype=object)
    sss=StratifiedShuffleSplit(n_splits=1,test_size=lock_size,random_state=LOCKBOX_SEED)
    DEV_IDX,LOCK_IDX=next(sss.split(np.zeros(N_TRAIN),lock_strata))
    DEV_IDX=np.asarray(DEV_IDX,int); LOCK_IDX=np.asarray(LOCK_IDX,int)
print("dev/lockbox",len(DEV_IDX),len(LOCK_IDX))
print("lockbox top-module distribution",pd.Series(strata[LOCK_IDX]).value_counts().sort_index().to_dict())

# ---- cell 10 (code) ----
# Fully isolated Consensus3 anchor for dev OOF and untouched lockbox.
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
    global_train=np.asarray(global_train,int); global_out=np.asarray(global_out,int)
    oo=np.zeros((len(global_train),N_MODULES),np.float32); out=np.zeros((len(global_out),N_MODULES),np.float32)
    for m in range(N_MODULES):
        Xt=np.column_stack([base_tr[:,m],USER_FEATURES[global_train]]).astype(np.float32)
        Xo=np.column_stack([base_out[:,m],USER_FEATURES[global_out]]).astype(np.float32); yy=targets[global_train,m].astype(np.int32)
        for f,(lt,lv) in enumerate(folds_local):
            clf=make_cal(seed+31*m+f,audit=True)
            clf.fit(Xt[lt],yy[lt],eval_set=[(Xt[lv],yy[lv])],callbacks=[lgb.early_stopping(AUDIT_EARLY_STOP,verbose=False)])
            it=int(clf.best_iteration_ or AUDIT_LGB_ESTIMATORS)
            oo[lv,m]=clf.predict_proba(Xt[lv],num_iteration=it)[:,1].astype(np.float32)
            out[:,m]+=clf.predict_proba(Xo,num_iteration=it)[:,1].astype(np.float32)/len(folds_local)
    return oo,out

def build_isolated_anchor(global_train,global_out,seed):
    global_train=np.asarray(global_train,int); global_out=np.asarray(global_out,int)
    folds_local=make_local_folds(global_train,N_INNER,seed)
    btr,bout=sparse_matched(global_train,global_out,folds_local)
    otr,oout=structured_matched(global_train,global_out,folds_local,btr,bout,Y_BIN,seed+1000)
    return folds_local,btr,bout,otr,oout

DEV_FOLDS,dev_base_oof,lock_base,dev_meta_oof,lock_meta=build_isolated_anchor(DEV_IDX,LOCK_IDX,LOCKBOX_SEED+100)
dev_anchor=apply_consensus(dev_base_oof,dev_meta_oof)
lock_anchor=apply_consensus(lock_base,lock_meta)
print("isolated dev anchor",round(ndcg5(y[DEV_IDX],dev_anchor),6),"lockbox anchor",round(ndcg5(y[LOCK_IDX],lock_anchor),6))

# ---- cell 11 (markdown) ----
# ## Message-level Qwen3-Embedding-8B corpus
#
# Qwen remains frozen. V20 does **not** ask the model for test labels. It produces message/module vectors only. A small organizer-supervised residual adapter later changes the embedding geometry.

# ---- cell 12 (code) ----
def semantic_normalize(text):
    s=html.unescape(str(text)); return re.sub(r"\s+"," ",s).strip()
def semantic_key(text): return semantic_normalize(text).casefold()

learn_sem=chat[~chat.is_noise].copy()
learn_sem["semantic_text"]=learn_sem.user_chat_text.fillna("").map(semantic_normalize)
learn_sem["semantic_key"]=learn_sem.semantic_text.map(semantic_key)
learn_sem=learn_sem[learn_sem.semantic_key.str.len()>0].sort_values(["user_id","timestamp","chat_id"]).reset_index(drop=True)
key_to_mid={}; unique_message_texts=[]
for row in learn_sem.itertuples(index=False):
    if row.semantic_key not in key_to_mid:
        key_to_mid[row.semantic_key]=len(unique_message_texts); unique_message_texts.append(row.semantic_text)
learn_sem["msg_uid"]=learn_sem.semantic_key.map(key_to_mid).astype(int)
user_message_ids=[[] for _ in all_users]
for u,g in learn_sem.groupby("user_id",sort=False):
    if u in uid2idx: user_message_ids[uid2idx[u]]=g.msg_uid.astype(int).tolist()
HAS_SEMANTIC=np.asarray([len(v)>0 for v in user_message_ids],bool)

mods=modules.set_index("module_id").loc[MODULES].reset_index()
module_content_docs=[f"Modul pembelajaran Intelligo: {r.module_name}. Materi utama: {r.description_and_syllabus}." for r in mods.itertuples(index=False)]
module_full_docs=[f"Modul pembelajaran Intelligo: {r.module_name}. Materi utama: {r.description_and_syllabus}. Prasyarat: {r.prerequisite_level}." for r in mods.itertuples(index=False)]
CURRENT_TEXTS=[]
for ids in user_message_ids:
    CURRENT_TEXTS.append(unique_message_texts[ids[-1]] if ids else "")

MESSAGE_INSTRUCTION=(
    "Instruct: Buat representasi semantik untuk maksud BELAJAR pada satu pesan siswa Intelligo. "
    "Fokus pada skill/topik yang ingin dipelajari, level yang tersirat, dan tujuan belajar. "
    "Abaikan pembayaran, akun, sertifikat, jadwal administratif, dan percakapan non-belajar. "
    "Representasi akan dipakai untuk mencocokkan kebutuhan siswa dengan modul kursus.\nQuery: "
)
print("semantic rows",len(learn_sem),"unique messages",len(unique_message_texts),"semantic users",int(HAS_SEMANTIC.sum()))

# ---- cell 13 (code) ----
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
    for t in texts: h.update(str(t).encode()); h.update(b'\0')
    return h.hexdigest()[:16]
def _last_token_pool(last_hidden,attention_mask):
    mask=attention_mask.to(last_hidden.device)
    left_padding=bool((mask[:,-1].sum()==mask.shape[0]).item())
    if left_padding: return last_hidden[:,-1]
    seq=mask.sum(dim=1)-1
    import torch
    return last_hidden[torch.arange(last_hidden.shape[0],device=last_hidden.device),seq]

qwen_model=None; qwen_tokenizer=None; qwen_input_device=None
def load_qwen():
    global qwen_model,qwen_tokenizer,qwen_input_device
    if qwen_model is not None or V20_SMOKE or V20_SKIP_QWEN: return
    import torch
    from transformers import AutoTokenizer,AutoModel
    if not torch.cuda.is_available() or torch.cuda.device_count()<2:
        raise RuntimeError("Full V20 requires Kaggle GPU T4 x2. Select Accelerator: GPU T4 x2.")
    for i in range(torch.cuda.device_count()):
        p=torch.cuda.get_device_properties(i); print(f"GPU {i}: {p.name}, {p.total_memory/1024**3:.1f} GiB")
    qwen_tokenizer=AutoTokenizer.from_pretrained(QWEN_MODEL,revision=QWEN_REVISION,padding_side='left')
    qwen_model=AutoModel.from_pretrained(QWEN_MODEL,revision=QWEN_REVISION,torch_dtype=torch.float16,
        device_map='balanced',max_memory={0:'13GiB',1:'13GiB','cpu':'24GiB'},low_cpu_mem_usage=True,attn_implementation='sdpa')
    qwen_model.eval(); qwen_model.config.use_cache=False
    qwen_input_device=qwen_model.get_input_embeddings().weight.device
    print("Qwen device map",qwen_model.hf_device_map)

def encode_qwen_cached(name,texts,instruction,max_length,start_batch):
    key=_text_hash(texts,instruction); path=CACHE_ROOT/f"qwen_{name}_{key}_{QWEN_DIM}.npy"
    if path.exists():
        a=np.load(path).astype(np.float32)
        if a.shape==(len(texts),QWEN_DIM): print("loaded",path.name); return a
    if V20_SMOKE or V20_SKIP_QWEN:
        a=_fake_embeddings([instruction+t for t in texts],QWEN_DIM,'v20-qwen-'+name); np.save(path,a.astype(np.float16)); return a
    import torch
    load_qwen(); prepared=[instruction+str(t) for t in texts]; result=np.empty((len(prepared),QWEN_DIM),np.float32)
    pos=0; bs=start_batch
    while pos<len(prepared):
        take=min(bs,len(prepared)-pos); batch=prepared[pos:pos+take]
        try:
            toks=qwen_tokenizer(batch,padding=True,truncation=True,max_length=max_length,return_tensors='pt')
            toks={k:v.to(qwen_input_device) for k,v in toks.items()}
            with torch.inference_mode(): out=qwen_model(**toks); emb=_last_token_pool(out.last_hidden_state,toks['attention_mask'])
            result[pos:pos+take]=emb.float().cpu().numpy(); pos+=take
            if pos%max(100,bs*20)==0 or pos==len(prepared): print(name,pos,"/",len(prepared),"batch",bs)
            del toks,out,emb
        except RuntimeError as e:
            if 'out of memory' not in str(e).lower() or bs<=1: raise
            bs=max(1,bs//2); print("Qwen OOM -> batch",bs); torch.cuda.empty_cache(); gc.collect()
    result=_normalize_rows(result); np.save(path,result.astype(np.float16)); return result

message_emb=encode_qwen_cached('messages',unique_message_texts,MESSAGE_INSTRUCTION,MESSAGE_MAX_TOKENS,8)
module_content_emb=encode_qwen_cached('module_content',module_content_docs,'',MODULE_MAX_TOKENS,17)
module_full_emb=encode_qwen_cached('module_full',module_full_docs,'',MODULE_MAX_TOKENS,17)
if qwen_model is not None:
    del qwen_model,qwen_tokenizer; qwen_model=qwen_tokenizer=None
    import torch; torch.cuda.empty_cache(); gc.collect()
print("Qwen shapes",message_emb.shape,module_full_emb.shape)

# ---- cell 14 (markdown) ----
# ## BGE-M3 orthogonal late interaction
#
# BGE-M3 is used as a frozen second representation family. V20 extracts dense and lexical scores plus the model's ColBERT-style token vectors. ColBERT scores are computed from separately encoded user-current-message and module vectors, so documents are not repeatedly re-encoded.

# ---- cell 15 (code) ----
def row_z(x):
    x=np.asarray(x,np.float32); return ((x-x.mean(1,keepdims=True))/np.maximum(x.std(1,keepdims=True),1e-6)).astype(np.float32)

def fake_bge_scores():
    q=_fake_embeddings(CURRENT_TEXTS,64,'v20-bge-q'); d=_fake_embeddings(module_full_docs,64,'v20-bge-d')
    dense=q@d.T; sparse=.15*row_z(dense); colbert=.85*dense+.15*row_z(dense)
    return dense.astype(np.float32),sparse.astype(np.float32),colbert.astype(np.float32)

def lexical_matrix(qweights,dweights):
    out=np.zeros((len(qweights),len(dweights)),np.float32)
    for i,q in enumerate(qweights):
        if not q: continue
        for j,d in enumerate(dweights):
            if not d: continue
            if len(q)>len(d): q0,d0=d,q
            else: q0,d0=q,d
            s=0.0
            for k,v in q0.items():
                if k in d0: s+=float(v)*float(d0[k])
            out[i,j]=s
    return out

def colbert_matrix(qvecs,dvecs,device='cuda:0'):
    import torch
    out=np.zeros((len(qvecs),len(dvecs)),np.float32)
    with torch.inference_mode():
        for j,p_np in enumerate(dvecs):
            p=torch.as_tensor(np.asarray(p_np),device=device,dtype=torch.float16)
            for st in range(0,len(qvecs),32):
                batch=qvecs[st:st+32]; lens=[len(v) for v in batch]; maxq=max(max(lens),1); dim=p.shape[-1]
                qpad=torch.zeros((len(batch),maxq,dim),device=device,dtype=torch.float16)
                mask=torch.zeros((len(batch),maxq),device=device,dtype=torch.float16)
                for b,v in enumerate(batch):
                    vv=torch.as_tensor(np.asarray(v),device=device,dtype=torch.float16); l=min(len(vv),maxq)
                    if l: qpad[b,:l]=vv[:l]; mask[b,:l]=1
                mx=torch.einsum('bqd,pd->bqp',qpad,p).max(-1).values
                sc=(mx*mask).sum(-1)/mask.sum(-1).clamp_min(1)
                out[st:st+len(batch),j]=sc.float().cpu().numpy()
            del p
    return out

bge_cache=CACHE_ROOT/'bge_m3_scores.npz'
if bge_cache.exists():
    z=np.load(bge_cache); bge_dense=z['dense']; bge_sparse=z['sparse']; bge_colbert=z['colbert']; print("loaded BGE score cache")
elif V20_SMOKE or V20_SKIP_BGE:
    bge_dense,bge_sparse,bge_colbert=fake_bge_scores(); np.savez_compressed(bge_cache,dense=bge_dense,sparse=bge_sparse,colbert=bge_colbert)
else:
    import torch
    from FlagEmbedding import BGEM3FlagModel
    if not torch.cuda.is_available(): raise RuntimeError("BGE-M3 requires GPU in full V20")
    bge=BGEM3FlagModel(BGE_MODEL,use_fp16=True,pooling_method='cls',devices='cuda:0')
    q_text=[t if t else "tidak ada pesan belajar" for t in CURRENT_TEXTS]
    qo=bge.encode(q_text,batch_size=BGE_BATCH,max_length=BGE_QUERY_MAXLEN,return_dense=True,return_sparse=True,return_colbert_vecs=True)
    do=bge.encode(module_full_docs,batch_size=8,max_length=BGE_DOC_MAXLEN,return_dense=True,return_sparse=True,return_colbert_vecs=True)
    bge_dense=(np.asarray(qo['dense_vecs'],np.float32)@np.asarray(do['dense_vecs'],np.float32).T).astype(np.float32)
    bge_sparse=lexical_matrix(qo['lexical_weights'],do['lexical_weights'])
    bge_colbert=colbert_matrix(qo['colbert_vecs'],do['colbert_vecs'],'cuda:0')
    del bge,qo,do; torch.cuda.empty_cache(); gc.collect()
    np.savez_compressed(bge_cache,dense=bge_dense,sparse=bge_sparse,colbert=bge_colbert)
BGE_HYBRID=(BGE_WEIGHTS[0]*row_z(bge_dense)+BGE_WEIGHTS[1]*row_z(bge_sparse)+BGE_WEIGHTS[2]*row_z(bge_colbert)).astype(np.float32)
BGE_HYBRID[~HAS_SEMANTIC]=0
print("BGE score matrices",bge_dense.shape,bge_sparse.shape,bge_colbert.shape)

# ---- cell 16 (markdown) ----
# ## MineToday-specific residual metric adapter
#
# The adapter changes **embedding geometry**, not the 8B backbone. Initial geometry is Qwen's Matryoshka prefix; trainable low-rank residual projections are zero-initialized. Loss combines:
#
# - graded multi-positive listwise cross entropy;
# - all graded positive/negative pair comparisons;
# - hard negatives chosen from frozen Qwen+BGE semantic proximity.
#
# This keeps capacity low while making the representation organizer-specific.

# ---- cell 17 (code) ----
# Build fixed message packs (oldest -> latest, right aligned) and frozen hard-negative scores.
MSG_PACK=np.zeros((len(all_users),MAX_USER_MESSAGES,QWEN_DIM),np.float16)
MSG_MASK=np.zeros((len(all_users),MAX_USER_MESSAGES),np.float32)
for ui,ids in enumerate(user_message_ids):
    ids=ids[-MAX_USER_MESSAGES:]
    if not ids: continue
    start=MAX_USER_MESSAGES-len(ids)
    MSG_PACK[ui,start:]=message_emb[np.asarray(ids,int)].astype(np.float16); MSG_MASK[ui,start:]=1
RECENCY=np.asarray(([.50,.72,1.0] if MAX_USER_MESSAGES==3 else [.35,.50,.72,1.0]),np.float32)

# Frozen Qwen baseline similarity used only for hard-negative mining.
qdim=QWEN_ADAPTER_DIM
qmsg=_normalize_rows(message_emb[:,:qdim]); qdoc=_normalize_rows((module_content_emb[:,:qdim]+module_full_emb[:,:qdim])*.5)
QRAW=np.zeros((len(all_users),N_MODULES),np.float32)
for ui,ids in enumerate(user_message_ids):
    ids=ids[-MAX_USER_MESSAGES:]
    if not ids: continue
    S=qmsg[np.asarray(ids,int)]@qdoc.T
    w=RECENCY[-len(ids):]; w=w/w.sum(); QRAW[ui]=(.55*S[-1]+.45*(S*w[:,None]).sum(0)).astype(np.float32)
HARD_SCORE=(.65*row_z(QRAW)+.35*row_z(BGE_HYBRID)).astype(np.float32)
HARD_SCORE[~HAS_SEMANTIC]=0

# Exact-template diagnostics: repeated organizer phrasing vs lexically novel messages.
train_key_counts=Counter(learn_sem[learn_sem.user_id.isin(set(train.user_id))].semantic_key.tolist())
USER_REPEAT=np.zeros(len(all_users),bool)
for ui,ids in enumerate(user_message_ids):
    if ids:
        USER_REPEAT[ui]=any(train_key_counts[semantic_key(unique_message_texts[k])]>=2 for k in ids)
print("repeat users train/test",int(USER_REPEAT[:N_TRAIN].sum()),int(USER_REPEAT[N_TRAIN:].sum()))

# ---- cell 18 (code) ----
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualMetricAdapter(nn.Module):
    def __init__(self,input_dim,adapter_dim,bottleneck):
        super().__init__(); self.adapter_dim=adapter_dim
        self.q_down=nn.Linear(input_dim,bottleneck,bias=False); self.q_up=nn.Linear(bottleneck,adapter_dim,bias=False)
        self.d_down=nn.Linear(input_dim,bottleneck,bias=False); self.d_up=nn.Linear(bottleneck,adapter_dim,bias=False)
        nn.init.normal_(self.q_down.weight,std=.012); nn.init.normal_(self.d_down.weight,std=.012)
        nn.init.zeros_(self.q_up.weight); nn.init.zeros_(self.d_up.weight)
    def project_q(self,x):
        base=x[...,:self.adapter_dim]
        delta=self.q_up(F.gelu(self.q_down(x)))
        return F.normalize(base+QWEN_RESIDUAL_SCALE*delta,dim=-1)
    def project_d(self,x):
        base=x[...,:self.adapter_dim]
        delta=self.d_up(F.gelu(self.d_down(x)))
        return F.normalize(base+QWEN_RESIDUAL_SCALE*delta,dim=-1)
    def forward(self,msg,mask,doc):
        q=self.project_q(msg); d=self.project_d(doc)
        sim=torch.einsum('bkd,md->bkm',q,d)
        rw=torch.as_tensor(RECENCY,device=sim.device,dtype=sim.dtype).clamp_min(1e-5)
        bias=torch.log(rw)[None,:,None]
        valid=mask[:,:,None]>0
        z=sim/AGG_TEMP+bias
        z=torch.where(valid,z,torch.full_like(z,-1e4))
        return AGG_TEMP*torch.logsumexp(z,dim=1)

def hard_negative_indices(global_idx,k=3):
    out=np.zeros((len(global_idx),k),np.int64)
    for ii,g in enumerate(global_idx):
        neg=np.flatnonzero(y[g]<=0)
        if len(neg)==0: neg=np.arange(N_MODULES)
        order=neg[np.argsort(-HARD_SCORE[g,neg])]
        take=order[:k]
        if len(take)<k: take=np.resize(take,k)
        out[ii]=take
    return out

def train_adapter(train_idx,seed):
    train_idx=np.asarray(train_idx,int); use=train_idx[HAS_SEMANTIC[train_idx]]
    torch.manual_seed(int(seed)); np.random.seed(int(seed)); random.seed(int(seed))
    dev=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model=ResidualMetricAdapter(QWEN_DIM,QWEN_ADAPTER_DIM,QWEN_BOTTLENECK).to(dev)
    docs=torch.as_tensor(((module_content_emb+module_full_emb)*.5).astype(np.float32),device=dev)
    opt=torch.optim.AdamW(model.parameters(),lr=ADAPTER_LR,weight_decay=ADAPTER_WD)
    scaler=torch.amp.GradScaler('cuda',enabled=(dev.type=='cuda'))
    hneg=hard_negative_indices(use,3); posmap={int(g):i for i,g in enumerate(use)}
    rng=np.random.default_rng(seed)
    for ep in range(ADAPTER_EPOCHS):
        order=rng.permutation(use)
        ep_loss=[]
        model.train()
        for st in range(0,len(order),ADAPTER_BATCH):
            gi=order[st:st+ADAPTER_BATCH]
            msg=torch.as_tensor(MSG_PACK[gi].astype(np.float32),device=dev)
            mask=torch.as_tensor(MSG_MASK[gi],device=dev)
            yy=torch.as_tensor(y[gi],device=dev)
            ww=torch.as_tensor(shift_w[gi],device=dev)
            hn=torch.as_tensor(np.stack([hneg[posmap[int(g)]] for g in gi]),device=dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type,dtype=torch.float16,enabled=(dev.type=='cuda')):
                logits=model(msg,mask,docs)
                gain=torch.where(yy>0,torch.expm1(2.0*yy),torch.zeros_like(yy))
                target=gain/gain.sum(1,keepdim=True).clamp_min(1e-6)
                list_u=-(target*F.log_softmax(logits/LIST_TEMP,dim=1)).sum(1)
                dy=yy[:,:,None]-yy[:,None,:]; ds=logits[:,:,None]-logits[:,None,:]
                pmask=dy>.04
                pair_mat=F.softplus(-ds/PAIR_TEMP)*dy.clamp_min(0)
                pair_u=(pair_mat*pmask).sum((1,2))/pmask.sum((1,2)).clamp_min(1)
                top_pos=torch.argmax(yy,dim=1)
                pos_log=logits.gather(1,top_pos[:,None]).squeeze(1)
                neg_log=logits.gather(1,hn)
                hard_u=F.softplus((neg_log-pos_log[:,None])/PAIR_TEMP).mean(1)
                loss_u=.50*list_u+.35*pair_u+.15*hard_u
                loss=(loss_u*ww).sum()/ww.sum().clamp_min(1)
            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(opt); scaler.update(); ep_loss.append(float(loss.detach().cpu()))
        if ep in {0,ADAPTER_EPOCHS-1} or (ep+1)%10==0: print("adapter",seed,"epoch",ep+1,"loss",round(float(np.mean(ep_loss)),5))
    return model,docs

def predict_adapter(model,docs,global_idx,batch=128):
    global_idx=np.asarray(global_idx,int); dev=next(model.parameters()).device; out=np.zeros((len(global_idx),N_MODULES),np.float32); model.eval()
    with torch.inference_mode():
        for st in range(0,len(global_idx),batch):
            gi=global_idx[st:st+batch]
            msg=torch.as_tensor(MSG_PACK[gi].astype(np.float32),device=dev); mask=torch.as_tensor(MSG_MASK[gi],device=dev)
            with torch.autocast(device_type=dev.type,dtype=torch.float16,enabled=(dev.type=='cuda')):
                z=model(msg,mask,docs)
            out[st:st+len(gi)]=z.float().cpu().numpy()
    out[~HAS_SEMANTIC[global_idx]]=0
    return out

# ---- cell 19 (markdown) ----
# ## Development OOF semantic space
#
# The lockbox remains unopened here. Adapter folds are aligned to the isolated dev anchor folds. Only two selective intervention budgets (4% and 6%) are considered.

# ---- cell 20 (code) ----
# Adapter OOF on development users.
dev_sem_qwen=np.zeros((len(DEV_IDX),N_MODULES),np.float32)
dev_fold_id=np.full(len(DEV_IDX),-1,int)
for f,(lt,lv) in enumerate(DEV_FOLDS):
    gtr=DEV_IDX[lt]; gva=DEV_IDX[lv]
    print(f"semantic dev fold {f+1}/{len(DEV_FOLDS)} train={len(gtr)} val={len(gva)}")
    mdl,docs=train_adapter(gtr,SEED+20000+f)
    dev_sem_qwen[lv]=predict_adapter(mdl,docs,gva); dev_fold_id[lv]=f
    del mdl,docs; torch.cuda.empty_cache() if torch.cuda.is_available() else None; gc.collect()
assert (dev_fold_id>=0).all()

def combine_sem(qwen_score,global_idx):
    global_idx=np.asarray(global_idx,int)
    return (QWEN_SHARE*row_z(qwen_score)+BGE_SHARE*row_z(BGE_HYBRID[global_idx])).astype(np.float32)

dev_sem=combine_sem(dev_sem_qwen,DEV_IDX)
print("dev semantic matrices",dev_sem_qwen.shape,dev_sem.shape)

# ---- cell 21 (code) ----
def quantile_transport(src_ref,tgt_ref,src_out):
    src_ref=np.asarray(src_ref,float); tgt_ref=np.asarray(tgt_ref,float); src_out=np.asarray(src_out,float)
    out=np.zeros_like(src_out,np.float32); n=len(src_ref); grid=np.linspace(0,1,n)
    for m in range(N_MODULES):
        xs=np.sort(src_ref[:,m]); ys=np.sort(tgt_ref[:,m])
        q=np.searchsorted(xs,src_out[:,m],side='right')/max(n,1)
        q=np.clip(q,0,1); out[:,m]=np.interp(q,grid,ys).astype(np.float32)
    return out

def semantic_evidence(anchor,qscore,bgescore):
    qz=row_z(qscore); bz=row_z(bgescore); out=np.zeros(len(anchor),np.float32)
    ranks=np.argsort(-anchor,axis=1)
    for i in range(len(anchor)):
        inside=ranks[i,:5]; outside=ranks[i,5:10]
        best=-1e9
        for o in outside:
            qd=qz[i,o]-qz[i,inside]; bd=bz[i,o]-bz[i,inside]
            joint=.75*qd+.25*bd; agree=np.maximum(np.minimum(qd,bd),0)
            best=max(best,float(np.max(joint+.35*agree)))
        out[i]=best
    return out

def selective_candidate(base,oldmeta,anchor,sem_ref,meta_ref,sem_out,qscore_out,bge_out,semantic_mask,pct,threshold=None):
    mapped=quantile_transport(sem_ref,meta_ref,sem_out)
    newmeta=oldmeta+SEMANTIC_BETA*(mapped-oldmeta)
    proposal=apply_consensus(base,newmeta)
    ev=semantic_evidence(anchor,qscore_out,bge_out); valid=np.asarray(semantic_mask,bool)&np.isfinite(ev)
    if threshold is None:
        vv=ev[valid]; threshold=float(np.quantile(vv,1-pct)) if len(vv) else np.inf
    take=valid&(ev>=threshold)&(ev>0)
    out=anchor.copy(); out[take]=proposal[take]
    return out,ev,threshold,take

def weighted_ndcg(ytrue,scores,w):
    u=per_user_ndcg(ytrue,scores); w=np.asarray(w,float); return float(np.sum(u*w)/np.maximum(w.sum(),1e-12))

def subgroup_delta(ytrue,cand,anchor,mask):
    mask=np.asarray(mask,bool)
    if mask.sum()<10:return np.nan
    return float(ndcg5(ytrue[mask],cand[mask])-ndcg5(ytrue[mask],anchor[mask]))

dev_rows=[]; dev_candidates={}; dev_meta_ref=dev_meta_oof
for name,pct in CANDIDATES.items():
    cand,ev,thr,take=selective_candidate(dev_base_oof,dev_meta_oof,dev_anchor,dev_sem,dev_meta_ref,dev_sem,dev_sem_qwen,BGE_HYBRID[DEV_IDX],HAS_SEMANTIC[DEV_IDX],pct)
    dev_candidates[name]=(cand,ev,thr,take)
    d=ndcg5(y[DEV_IDX],cand)-ndcg5(y[DEV_IDX],dev_anchor)
    wd=weighted_ndcg(y[DEV_IDX],cand,shift_w[DEV_IDX])-weighted_ndcg(y[DEV_IDX],dev_anchor,shift_w[DEV_IDX])
    wins=0; fold_d=[]
    for f in range(len(DEV_FOLDS)):
        mm=dev_fold_id==f; fd=ndcg5(y[DEV_IDX][mm],cand[mm])-ndcg5(y[DEV_IDX][mm],dev_anchor[mm]); fold_d.append(fd); wins+=fd>0
    rep=subgroup_delta(y[DEV_IDX],cand,dev_anchor,USER_REPEAT[DEV_IDX])
    nov=subgroup_delta(y[DEV_IDX],cand,dev_anchor,~USER_REPEAT[DEV_IDX])
    dev_rows.append(dict(candidate=name,dev_delta=d,weighted_delta=wd,fold_wins=int(wins),fold_deltas=fold_d,repeat_delta=rep,novel_delta=nov,threshold=thr,selected_users=int(take.sum())))
    print(name,"dev",round(d,6),"weighted",round(wd,6),"wins",wins,"/",len(DEV_FOLDS),"repeat",round(rep,6) if np.isfinite(rep) else None,"novel",round(nov,6) if np.isfinite(nov) else None,"selected",int(take.sum()))

# Freeze one candidate before opening the lockbox. Selection rule is predeclared: maximize the worse of raw and test-weighted dev delta.
sel_scores={r['candidate']:min(r['dev_delta'],r['weighted_delta']) for r in dev_rows}
SELECTED=max(sel_scores,key=sel_scores.get)
SELECTED_PCT=CANDIDATES[SELECTED]
SELECTED_THR=dev_candidates[SELECTED][2]
print("FROZEN BEFORE LOCKBOX:",SELECTED,"pct",SELECTED_PCT,"threshold",round(SELECTED_THR,5),"selection_score",round(sel_scores[SELECTED],6))

# ---- cell 22 (markdown) ----
# ## Open the untouched lockbox once
#
# No candidate, strength, threshold percentile, model architecture, or loss is changed after this cell. If the lockbox does not confirm the development signal, V20 must fail closed.

# ---- cell 23 (code) ----
# Train the frozen semantic adapter on all development users; evaluate the untouched lockbox once.
full_dev_model,full_dev_docs=train_adapter(DEV_IDX,LOCKBOX_SEED+7000)
lock_qwen=predict_adapter(full_dev_model,full_dev_docs,LOCK_IDX)
lock_sem=combine_sem(lock_qwen,LOCK_IDX)
lock_candidate,lock_ev,_,lock_take=selective_candidate(lock_base,lock_meta,lock_anchor,dev_sem,dev_meta_ref,lock_sem,lock_qwen,BGE_HYBRID[LOCK_IDX],HAS_SEMANTIC[LOCK_IDX],SELECTED_PCT,threshold=SELECTED_THR)
lock_delta=ndcg5(y[LOCK_IDX],lock_candidate)-ndcg5(y[LOCK_IDX],lock_anchor)
lock_user_delta=per_user_ndcg(y[LOCK_IDX],lock_candidate)-per_user_ndcg(y[LOCK_IDX],lock_anchor)
lock_ci=paired_boot(lock_user_delta,LOCKBOX_SEED+99)
print("LOCKBOX anchor",round(ndcg5(y[LOCK_IDX],lock_anchor),6),"candidate",round(ndcg5(y[LOCK_IDX],lock_candidate),6),"delta",round(lock_delta,6),"CI",tuple(round(v,6) for v in lock_ci),"selected",int(lock_take.sum()))

dev_selected_row=next(r for r in dev_rows if r['candidate']==SELECTED)
DEV_GATE=(dev_selected_row['dev_delta']>=MIN_DEV_DELTA and dev_selected_row['weighted_delta']>=MIN_DEV_WEIGHTED_DELTA and dev_selected_row['fold_wins']>=MIN_DEV_WINS and (not np.isfinite(dev_selected_row['repeat_delta']) or dev_selected_row['repeat_delta']>=MIN_REPEAT_GROUP_DELTA) and (not np.isfinite(dev_selected_row['novel_delta']) or dev_selected_row['novel_delta']>=MIN_NOVEL_GROUP_DELTA))
LOCK_GATE=(lock_delta>=MIN_LOCK_DELTA and lock_ci[0]>=MIN_LOCK_CI_LOW)
print("DEV_GATE",DEV_GATE,"LOCK_GATE",LOCK_GATE)

# ---- cell 24 (markdown) ----
# ## Final all-train fit and test scoring
#
# The final adapter is fit on all 4,000 labeled users only after the candidate is frozen and lockbox is measured. Transport/evidence thresholds are learned from **out-of-sample train semantic scores**: dev OOF plus untouched lockbox predictions.

# ---- cell 25 (code) ----
# OOS semantic reference for all labeled users: dev is CV OOF, lockbox is prediction from dev-only adapter.
oos_qwen=np.zeros((N_TRAIN,N_MODULES),np.float32); oos_qwen[DEV_IDX]=dev_sem_qwen; oos_qwen[LOCK_IDX]=lock_qwen
oos_sem=combine_sem(oos_qwen,np.arange(N_TRAIN))
oos_ev=semantic_evidence(consensus3_oof,oos_qwen,BGE_HYBRID[:N_TRAIN])
valid=HAS_SEMANTIC[:N_TRAIN]
FINAL_EVIDENCE_THR=float(np.quantile(oos_ev[valid],1-SELECTED_PCT)) if valid.any() else np.inf

# Fit all train, predict test.
all_model,all_docs=train_adapter(np.arange(N_TRAIN),SEED+29000)
test_qwen=predict_adapter(all_model,all_docs,np.arange(N_TRAIN,N_TRAIN+N_TEST))
test_sem=combine_sem(test_qwen,np.arange(N_TRAIN,N_TRAIN+N_TEST))

# Build both predeclared attack files for diagnostics, but primary may only use the frozen selected family.
final_attacks={}; final_take={}
for name,pct in CANDIDATES.items():
    thr=float(np.quantile(oos_ev[valid],1-pct)) if valid.any() else np.inf
    cand,ev,_,take=selective_candidate(base_test,meta_test,consensus3_test,oos_sem,meta_oof,test_sem,test_qwen,BGE_HYBRID[N_TRAIN:],HAS_SEMANTIC[N_TRAIN:],pct,threshold=thr)
    final_attacks[name]=cand; final_take[name]=take
    print(name,"test selected",int(take.sum()),"top5-set",round(top5_set_change_rate(cand,consensus3_test),4),"order",round(top5_order_change_rate(cand,consensus3_test),4),"top1",round(top1_change_rate(cand,consensus3_test),4))

selected_test=final_attacks[SELECTED]
TEST_SET=top5_set_change_rate(selected_test,consensus3_test); TEST_ORDER=top5_order_change_rate(selected_test,consensus3_test); TEST_TOP1=top1_change_rate(selected_test,consensus3_test)
PERTURB_GATE=(TEST_SET>=MIN_TEST_SET_CHANGE and TEST_SET<=MAX_TEST_SET_CHANGE and TEST_ORDER<=MAX_TEST_ORDER_CHANGE and TEST_TOP1<=MAX_TEST_TOP1_CHANGE)
PROMOTE=bool(DEV_GATE and LOCK_GATE and PERTURB_GATE)
primary_test=selected_test.copy() if PROMOTE else consensus3_test.copy()
primary_label=SELECTED if PROMOTE else 'consensus3_fallback'
print("PERTURB_GATE",PERTURB_GATE,"PROMOTE",PROMOTE,"PRIMARY",primary_label)

# ---- cell 26 (markdown) ----
# ## Decision report and artifacts
#
# **Submission discipline:** submit `submission_v20_primary.csv` only. If it is the fallback, V20 did not earn a leaderboard attempt.

# ---- cell 27 (code) ----
rows=[]
for r in dev_rows:
    name=r['candidate']; cand=final_attacks[name]
    rows.append({
        'candidate':name,'selected_for_lockbox':name==SELECTED,'dev_delta':r['dev_delta'],'dev_weighted_delta':r['weighted_delta'],'dev_fold_wins':r['fold_wins'],
        'dev_repeat_delta':r['repeat_delta'],'dev_novel_delta':r['novel_delta'],'lockbox_delta':lock_delta if name==SELECTED else np.nan,
        'lockbox_ci_low':lock_ci[0] if name==SELECTED else np.nan,'lockbox_ci_high':lock_ci[1] if name==SELECTED else np.nan,
        'test_top5_set_change':top5_set_change_rate(cand,consensus3_test),'test_order_change':top5_order_change_rate(cand,consensus3_test),'test_top1_change':top1_change_rate(cand,consensus3_test),
        'promoted':bool(PROMOTE and name==SELECTED)
    })
diag=pd.DataFrame(rows); display(diag)
print("================ V20 DECISION ================")
print("adversarial AUC       :",round(ADV_AUC,6))
print("frozen candidate      :",SELECTED)
print("dev delta             :",round(dev_selected_row['dev_delta'],6))
print("test-weighted dev     :",round(dev_selected_row['weighted_delta'],6))
print("dev fold wins         :",dev_selected_row['fold_wins'],'/',len(DEV_FOLDS))
print("lockbox delta         :",round(lock_delta,6),"CI",tuple(round(v,6) for v in lock_ci))
print("test perturbation     : set",round(TEST_SET,4),"order",round(TEST_ORDER,4),"top1",round(TEST_TOP1,4))
print("gates                 : DEV",DEV_GATE,"LOCK",LOCK_GATE,"PERTURB",PERTURB_GATE)
print("PRIMARY               :",primary_label)
print("public anchor/leader  :",PUBLIC_CONSENSUS3_LB,PUBLIC_LEADER_LB)

# ---- cell 28 (code) ----
def save_submission(scores,name):
    out=sample.copy(); out.loc[:,MODULES]=np.asarray(scores,np.float32)
    path=Path('/kaggle/working')/name if Path('/kaggle/working').exists() else Path('/mnt/data')/name
    out.to_csv(path,index=False); print("saved",path); return str(path)

root=Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('/mnt/data')
paths={}
paths['attack1']=save_submission(final_attacks['selective4'],'submission_v20_attack1.csv')
paths['attack2']=save_submission(final_attacks['selective6'],'submission_v20_attack2.csv')
paths['primary']=save_submission(primary_test,'submission_v20_primary.csv')
paths['consensus3']=save_submission(consensus3_test,'submission_v20_consensus3_reference.csv')
diag.to_csv(root/'v20_diagnostics.csv',index=False)
summary={
    'public_consensus3_lb':PUBLIC_CONSENSUS3_LB,'public_leader_lb':PUBLIC_LEADER_LB,'public_v18_lb':PUBLIC_V18_LB,'public_v19_lb':PUBLIC_V19_LB,
    'adversarial_auc':ADV_AUC,'selected':SELECTED,'dev_row':dev_selected_row,'lockbox_delta':lock_delta,'lockbox_ci':lock_ci,
    'dev_gate':bool(DEV_GATE),'lock_gate':bool(LOCK_GATE),'perturb_gate':bool(PERTURB_GATE),'promote':bool(PROMOTE),'primary_label':primary_label,
    'test_set_change':TEST_SET,'test_order_change':TEST_ORDER,'test_top1_change':TEST_TOP1
}
with open(root/'v20_summary.json','w') as f: json.dump(summary,f,indent=2,default=lambda x:float(x) if isinstance(x,(np.floating,np.integer)) else str(x))
np.savez_compressed(root/'v20_oof_artifacts.npz',y=y,consensus3_oof=consensus3_oof,consensus3_test=consensus3_test,dev_idx=DEV_IDX,lock_idx=LOCK_IDX,dev_anchor=dev_anchor,lock_anchor=lock_anchor,dev_sem_qwen=dev_sem_qwen,lock_qwen=lock_qwen,oos_qwen=oos_qwen,domain_oof=domain_oof,shift_w=shift_w,**{f'dev_{k}':v[0] for k,v in dev_candidates.items()},**{f'test_{k}':v for k,v in final_attacks.items()})
print("artifacts complete")

# ---- cell 29 (code) ----
# Hard invariants. Fail loudly if anything malformed or if no-chat users were changed.
assert y.shape==(N_TRAIN,17) and len(MODULES)==17
assert np.isfinite(consensus3_oof).all() and np.isfinite(consensus3_test).all()
assert np.isfinite(dev_sem_qwen).all() and np.isfinite(lock_qwen).all() and np.isfinite(test_qwen).all()
assert np.isfinite(primary_test).all()
assert len(set(DEV_IDX)&set(LOCK_IDX))==0 and len(DEV_IDX)+len(LOCK_IDX)==N_TRAIN
nochat_test=~HAS_SEMANTIC[N_TRAIN:]
for name,cand in final_attacks.items():
    if nochat_test.any(): assert np.allclose(cand[nochat_test],consensus3_test[nochat_test],atol=1e-7)
if not PROMOTE: assert np.allclose(primary_test,consensus3_test,atol=0,rtol=0)
if not V20_SMOKE:
    assert abs(ndcg5(y,consensus3_oof)-0.6643682395561207)<0.0012,ndcg5(y,consensus3_oof)
print("V20 hard invariants: PASS")
