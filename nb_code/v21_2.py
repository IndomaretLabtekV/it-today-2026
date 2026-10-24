# Converted from notebooks/v21_2.ipynb: markdown cells become comments, %/! magics neutralized.

# ---- cell 0 (markdown) ----
# # MineToday V21.2 — Anchor-Initialized Semantic LambdaRank
#
# **Goal:** preserve the public-validated Consensus3 ranking as the immutable prior, adapt Qwen3-Embedding-8B only on ADAPT, then learn a very low-capacity *within-user* LambdaRank residual on CAL. The ranker never replaces Consensus3 directly: it is projected back through a discrete trust region (ordering-only or one boundary swap). A fresh LOCK is opened only if a CAL candidate passes a stability-first gate.

# ---- cell 1 (code) ----
import os,sys,subprocess
V21_2_SKIP_INSTALL=bool(int(os.getenv("V21_2_SKIP_INSTALL","0")))
if not V21_2_SKIP_INSTALL:
    subprocess.check_call([
        sys.executable,"-m","pip","install","-q","-U",
        "transformers==4.57.6","accelerate==1.14.0","safetensors>=0.5,<1",
        "huggingface_hub>=0.34,<2","lightgbm>=4.5,<5","sentencepiece>=0.2,<1",
        "peft>=0.17,<0.19","bitsandbytes>=0.46,<0.49"
    ])
print("dependency setup complete; skip_install=",V21_2_SKIP_INSTALL)

# ---- cell 2 (code) ----
import os,re,json,ast,html,hashlib,random,warnings,gc,time,math
from pathlib import Path
from collections import defaultdict,Counter
os.environ.setdefault("OMP_NUM_THREADS","4"); os.environ.setdefault("OPENBLAS_NUM_THREADS","4"); os.environ.setdefault("MKL_NUM_THREADS","4")
os.environ["PYTHONHASHSEED"]="42"; os.environ["TOKENIZERS_PARALLELISM"]="false"; os.environ.setdefault("PYDEVD_DISABLE_FILE_VALIDATION","1")

import numpy as np,pandas as pd
from IPython.display import display
from scipy.sparse import hstack
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit,KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
warnings.filterwarnings("ignore")

SEED=42
SPLIT_SEED=2026082527
CAL_SEED=2026082528
RANK_SEED=2026082529
BOOT_SEED=2026082530
V21_2_SMOKE=bool(int(os.getenv("V21_2_SMOKE","0")))
V21_2_SKIP_QWEN=bool(int(os.getenv("V21_2_SKIP_QWEN","0")))
V21_UNUSED_BGE=True
N_FOLDS=2 if V21_2_SMOKE else 5
RIDGE_ALPHA=3.0
REFERENCE_W=0.45
LGB_ESTIMATORS=60 if V21_2_SMOKE else 1200
LGB_EARLY_STOP=6 if V21_2_SMOKE else 80
AUDIT_LGB_ESTIMATORS=45 if V21_2_SMOKE else 700
AUDIT_EARLY_STOP=5 if V21_2_SMOKE else 50
BOOTSTRAPS=250 if V21_2_SMOKE else 12000

PUBLIC_CONSENSUS3_LB=0.65856
PUBLIC_TOP2_LB=0.65966
PUBLIC_LEADER_LB=0.66082
PUBLIC_V18_LB=0.65829
PUBLIC_V19_LB=0.65791

QWEN_MODEL="Qwen/Qwen3-Embedding-8B"
QWEN_REVISION="1d8ad4c"
QUERY_MAX_TOKENS=96 if V21_2_SMOKE else 160
MODULE_MAX_TOKENS=128 if V21_2_SMOKE else 224
MAX_TRAJECTORY_MESSAGES=3
MULTI_DIMS=(32,48,64) if V21_2_SMOKE else (512,1024,4096)
DIM_WEIGHTS=np.asarray([.15,.60,.25],np.float32)
PRIMARY_DIM=MULTI_DIMS[1]

# Freeze the QLoRA recipe that produced the stable V21 representation.
LORA_LAST_N_LAYERS=8
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=.05
QLORA_LR=5e-5
QLORA_WD=.01
QLORA_EPOCHS=1
QLORA_BATCH=16 if V21_2_SMOKE else 4
QLORA_GRAD_CLIP=1.0
LIST_TEMP=.075
PAIR_TEMP=.08
GRADE_TEMP=.09
LOSS_REL=.60
LOSS_HARD=.25
LOSS_GRADED=.15

# Fresh isolation. Qwen sees ADAPT only; relevance stacker + ranker see CAL only; LOCK is untouched until CAL gate passes.
ADAPT_SIZE=260 if V21_2_SMOKE else 2600
CAL_SIZE=80 if V21_2_SMOKE else 800
LOCK_SIZE=60 if V21_2_SMOKE else 600
assert ADAPT_SIZE+CAL_SIZE+LOCK_SIZE==(400 if V21_2_SMOKE else 4000)
CAL_FOLDS=2 if V21_2_SMOKE else 4
REL_BAG_SEEDS=(0,) if V21_2_SMOKE else (0,911,1823)
REL_LGB_EST=45 if V21_2_SMOKE else 260

# Anchor-initialized LambdaRank translator: intentionally tiny and predeclared.
RANK_BAG_SEEDS=(0,17,43,89,137)
RANK_NUM_BOOST=20 if V21_2_SMOKE else 120
RANK_LR=.02
RANK_MIN_LEAF=25 if V21_2_SMOKE else 170
RANK_TRUNCATION=8
RANK_LABEL_GAIN=[0,25,40,55,70,85,100]  # approximates organizer relevance bands for grades 0..6
SAFE_PAIR_AGREE=len(RANK_BAG_SEEDS)
BOUNDARY_PAIR_AGREE=max(1,len(RANK_BAG_SEEDS)-1)
BOUNDARY_TOP5_AGREE=max(1,len(RANK_BAG_SEEDS)-1)
ORDER_GAP_QUANTILE=.25
ORDER_TOP1_GAP_QUANTILE=.10
BOUNDARY_GAP_QUANTILE=.60

# Stability-first promotion. Do not override these gates after LOCK opens.
MIN_CAL_DELTA=-.01 if V21_2_SMOKE else .00050
MIN_CAL_WINS=1 if V21_2_SMOKE else 3
MIN_CAL_STABILITY=-1 if V21_2_SMOKE else -.00015
MIN_LOCK_DELTA=-.01 if V21_2_SMOKE else .00040
MIN_LOCK_CI_LOW=-1 if V21_2_SMOKE else -.00050
MAX_TEST_SET_CHANGE=1 if V21_2_SMOKE else .070
MAX_TEST_ORDER_CHANGE=1 if V21_2_SMOKE else .190
MAX_TEST_TOP1_CHANGE=1 if V21_2_SMOKE else .020

ROOT=Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('/mnt/data')
CACHE_ROOT=ROOT/'v21_2_cache'; CACHE_ROOT.mkdir(parents=True,exist_ok=True)
random.seed(SEED); np.random.seed(SEED)
print("V21_2_SMOKE",V21_2_SMOKE,"skip_qwen",V21_2_SKIP_QWEN)
print("public anchor",PUBLIC_CONSENSUS3_LB,"top2",PUBLIC_TOP2_LB,"leader",PUBLIC_LEADER_LB,"gap",round(PUBLIC_LEADER_LB-PUBLIC_CONSENSUS3_LB,6))

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

if V21_2_SMOKE:
    train=train.iloc[:400].copy(); test=test.iloc[:100].copy()
    sample=sample.set_index("user_id").loc[test.user_id].reset_index()
    keep=set(train.user_id)|set(test.user_id)
    chat=chat[chat.user_id.isin(keep)].copy(); assessments=assessments[assessments.user_id.isin(keep)].copy()

MODULES=list(train.columns[1:]); N_MODULES=len(MODULES); N_TRAIN=len(train); N_TEST=len(test)
assert MODULES==list(sample.columns[1:]) and N_MODULES==17
assert (N_TRAIN,N_TEST)==((400,100) if V21_2_SMOKE else (4000,1000))
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
word_vec=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_df=.995,sublinear_tf=True,strip_accents="unicode",max_features=(3000 if V21_2_SMOKE else 50000),dtype=np.float32)
char_vec=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=2,sublinear_tf=True,max_features=(5000 if V21_2_SMOKE else 60000),dtype=np.float32)
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
# ## Fresh ADAPT / CAL / LOCK split
#
# V21.2 treats the previous V21 run as development evidence only. A new split is created with a new seed:
#
# 1. **ADAPT (2400)** — labels may update Qwen through QLoRA;
# 2. **CAL (1000)** — Qwen never trains on these users; CAL trains and cross-validates the semantic relevance calibrators and selects module trust;
# 3. **LOCK (600)** — neither Qwen nor calibrators see these labels before the single final gate.
#
# The extra CAL data directly targets V21's observed bottleneck: the semantic representation was stable CAL→LOCK, while 17 independent calibrators trained on only 700 users were noisy.

# ---- cell 9 (code) ----
# Stratify jointly by top relevance module and number of relevant modules.
pos_count=(y>0).sum(1).astype(int)
if V21_2_SMOKE:
    rng=np.random.default_rng(SPLIT_SEED); perm=rng.permutation(N_TRAIN)
    LOCK_IDX=np.sort(perm[:LOCK_SIZE]); CAL_IDX=np.sort(perm[LOCK_SIZE:LOCK_SIZE+CAL_SIZE]); ADAPT_IDX=np.sort(perm[LOCK_SIZE+CAL_SIZE:])
    joint_strata=np.asarray([str(np.argmax(y[i])) for i in range(N_TRAIN)])
else:
    joint_strata=np.asarray([f"{np.argmax(y[i]):02d}_{pos_count[i]}" for i in range(N_TRAIN)])
    sss_lock=StratifiedShuffleSplit(n_splits=1,test_size=LOCK_SIZE,random_state=SPLIT_SEED)
    remain_pos,lock_pos=next(sss_lock.split(np.zeros(N_TRAIN),joint_strata))
    REMAIN=np.asarray(remain_pos,int); LOCK_IDX=np.asarray(lock_pos,int)
    sss_cal=StratifiedShuffleSplit(n_splits=1,test_size=CAL_SIZE,random_state=SPLIT_SEED+1)
    adapt_local,cal_local=next(sss_cal.split(np.zeros(len(REMAIN)),joint_strata[REMAIN]))
    ADAPT_IDX=REMAIN[np.asarray(adapt_local,int)]; CAL_IDX=REMAIN[np.asarray(cal_local,int)]
assert len(ADAPT_IDX)==ADAPT_SIZE and len(CAL_IDX)==CAL_SIZE and len(LOCK_IDX)==LOCK_SIZE
assert len(set(ADAPT_IDX)&set(CAL_IDX))==0 and len(set(ADAPT_IDX)&set(LOCK_IDX))==0 and len(set(CAL_IDX)&set(LOCK_IDX))==0
assert len(set(ADAPT_IDX)|set(CAL_IDX)|set(LOCK_IDX))==N_TRAIN
print("split sizes adapt/cal/lock",len(ADAPT_IDX),len(CAL_IDX),len(LOCK_IDX))
print("anchor NDCG adapt/cal/lock",round(ndcg5(y[ADAPT_IDX],consensus3_oof[ADAPT_IDX]),6),round(ndcg5(y[CAL_IDX],consensus3_oof[CAL_IDX]),6),round(ndcg5(y[LOCK_IDX],consensus3_oof[LOCK_IDX]),6))

# ---- cell 10 (markdown) ----
# ## ADAPT-only anchor for leakage-safe semantic supervision
#
# V21 does **not** use the global 4,000-user OOF Consensus3 predictions as a teacher or as CAL/LOCK calibration features. That would let CAL/LOCK labels influence other users' OOF features.
#
# Instead, an isolated anchor is trained from **ADAPT labels only**:
#
# - ADAPT OOF sparse scores provide hard-negative ranks for QLoRA;
# - full ADAPT sparse + relevance calibrators predict CAL, LOCK, and test;
# - CAL/LOCK labels never influence QLoRA inputs or anchor features.
#
# At deployment, the semantic relevance **residual learned relative to this ADAPT-only anchor** is transported onto the exact full-data Consensus3 meta.

# ---- cell 11 (code) ----
# Isolated ADAPT-only anchor. Only organizer labels from ADAPT are allowed here.
ADAPT_FOLDS_N=2 if V21_2_SMOKE else 4
adapt_strata=np.argmax(y[ADAPT_IDX],axis=1).astype(str)
ADAPT_FOLDS=list(StratifiedKFold(ADAPT_FOLDS_N,shuffle=True,random_state=SPLIT_SEED+77).split(np.zeros(len(ADAPT_IDX)),adapt_strata))

def ridge_adapt_path(target):
    oo=np.zeros((len(ADAPT_IDX),N_MODULES),np.float32)
    for tr,va in ADAPT_FOLDS:
        mdl=Ridge(alpha=RIDGE_ALPHA,solver='lsqr',tol=2e-3,max_iter=1000)
        mdl.fit(X[ADAPT_IDX[tr]],target[ADAPT_IDX[tr]]); oo[va]=mdl.predict(X[ADAPT_IDX[va]]).astype(np.float32)
    full=Ridge(alpha=RIDGE_ALPHA,solver='lsqr',tol=2e-3,max_iter=1000); full.fit(X[ADAPT_IDX],target[ADAPT_IDX])
    return oo,full

ad_raw_oof,ad_raw_model=ridge_adapt_path(y); ad_top_oof,ad_top_model=ridge_adapt_path(Y_TOP)
ad_base_oof=(.9*ad_raw_oof+.1*ad_top_oof).astype(np.float32)

def adapt_base_predict(global_idx):
    global_idx=np.asarray(global_idx,int)
    rr=ad_raw_model.predict(X[global_idx]).astype(np.float32); tt=ad_top_model.predict(X[global_idx]).astype(np.float32)
    return (.9*rr+.1*tt).astype(np.float32)

AD_BASE_CAL=adapt_base_predict(CAL_IDX); AD_BASE_LOCK=adapt_base_predict(LOCK_IDX); AD_BASE_TEST=adapt_base_predict(np.arange(N_TRAIN,N_TRAIN+N_TEST))

def make_adapt_meta(seed,nest=500):
    return lgb.LGBMClassifier(n_estimators=(60 if V21_2_SMOKE else nest),learning_rate=.03,num_leaves=15,min_child_samples=50,reg_alpha=.5,reg_lambda=5.,colsample_bytree=.7,random_state=int(seed),n_jobs=4,verbosity=-1,force_col_wise=True)

# Historical full-fit pattern: train relevance calibrators on ADAPT labels using ADAPT OOF sparse features, predict held-out users from full-ADAPT sparse scores.
AD_META_CAL=np.zeros((len(CAL_IDX),N_MODULES),np.float32); AD_META_LOCK=np.zeros((len(LOCK_IDX),N_MODULES),np.float32); AD_META_TEST=np.zeros((N_TEST,N_MODULES),np.float32)
AD_P5_CAL=np.zeros((len(CAL_IDX),N_MODULES),np.float32); AD_P5_LOCK=np.zeros((len(LOCK_IDX),N_MODULES),np.float32); AD_P5_TEST=np.zeros((N_TEST,N_MODULES),np.float32)
for m in range(N_MODULES):
    Xtr=np.column_stack([ad_base_oof[:,m],USER_FEATURES[ADAPT_IDX]]).astype(np.float32)
    Xcal=np.column_stack([AD_BASE_CAL[:,m],USER_FEATURES[CAL_IDX]]).astype(np.float32)
    Xlock=np.column_stack([AD_BASE_LOCK[:,m],USER_FEATURES[LOCK_IDX]]).astype(np.float32)
    Xtest=np.column_stack([AD_BASE_TEST[:,m],USER_FEATURES[N_TRAIN:]]).astype(np.float32)
    yy=Y_BIN[ADAPT_IDX,m].astype(np.int32)
    mdl=make_adapt_meta(SPLIT_SEED+3000+m*31); mdl.fit(Xtr,yy)
    AD_META_CAL[:,m]=mdl.predict_proba(Xcal)[:,1]; AD_META_LOCK[:,m]=mdl.predict_proba(Xlock)[:,1]; AD_META_TEST[:,m]=mdl.predict_proba(Xtest)[:,1]
    yy5=Y_TOP5[ADAPT_IDX,m].astype(np.int32)
    mdl5=make_adapt_meta(SPLIT_SEED+7000+m*37); mdl5.fit(Xtr,yy5)
    AD_P5_CAL[:,m]=mdl5.predict_proba(Xcal)[:,1]; AD_P5_LOCK[:,m]=mdl5.predict_proba(Xlock)[:,1]; AD_P5_TEST[:,m]=mdl5.predict_proba(Xtest)[:,1]

AD_ANCHOR_CAL=apply_consensus(AD_BASE_CAL,AD_META_CAL); AD_ANCHOR_LOCK=apply_consensus(AD_BASE_LOCK,AD_META_LOCK); AD_ANCHOR_TEST=apply_consensus(AD_BASE_TEST,AD_META_TEST)
ADAPT_TEACHER_RANKS=np.empty((len(ADAPT_IDX),N_MODULES),np.int16)
_o=np.argsort(-ad_base_oof,axis=1); ADAPT_TEACHER_RANKS[np.arange(len(ADAPT_IDX))[:,None],_o]=np.arange(N_MODULES)[None,:]
ADAPT_LOCAL={g:i for i,g in enumerate(ADAPT_IDX.tolist())}
print('ADAPT-only anchor CAL/LOCK',round(ndcg5(y[CAL_IDX],AD_ANCHOR_CAL),6),round(ndcg5(y[LOCK_IDX],AD_ANCHOR_LOCK),6))

# ---- cell 12 (markdown) ----
# ## Learner trajectory queries and module documents
#
# Qwen is responsible only for **learning intent semantics**. Assessment/readiness stays structured and enters the relevance calibrator later.
#
# - user query: up to the last 3 non-administrative learning messages, explicitly tagged by recency;
# - module content view: title + syllabus;
# - module full view: title + syllabus + prerequisite, used as a weaker auxiliary document view;
# - no-chat users have the semantic branch hard-disabled.

# ---- cell 13 (code) ----
def semantic_normalize(text):
    s=html.unescape(str(text)); return re.sub(r"\s+"," ",s).strip()
def semantic_key(text): return semantic_normalize(text).casefold()

learn_sem=chat[~chat.is_noise].copy()
learn_sem["semantic_text"]=learn_sem.user_chat_text.fillna("").map(semantic_normalize)
learn_sem["semantic_key"]=learn_sem.semantic_text.map(semantic_key)
learn_sem=learn_sem[learn_sem.semantic_key.str.len()>0].sort_values(["user_id","timestamp","chat_id"]).reset_index(drop=True)

TRAJ_TEXTS=[]; LATEST_TEXTS=[]; LATEST_KEYS=[]; HAS_SEMANTIC=[]
for u in all_users:
    g=learn_sem[learn_sem.user_id==u]
    msgs=g.semantic_text.tolist()[-MAX_TRAJECTORY_MESSAGES:]
    keys=g.semantic_key.tolist()[-MAX_TRAJECTORY_MESSAGES:]
    if not msgs:
        TRAJ_TEXTS.append(""); LATEST_TEXTS.append(""); LATEST_KEYS.append(""); HAS_SEMANTIC.append(False); continue
    tags=["OLDER","RECENT","CURRENT"][-len(msgs):]
    TRAJ_TEXTS.append(" ".join(f"[{t}] {m}" for t,m in zip(tags,msgs)))
    LATEST_TEXTS.append(msgs[-1]); LATEST_KEYS.append(keys[-1]); HAS_SEMANTIC.append(True)
HAS_SEMANTIC=np.asarray(HAS_SEMANTIC,bool); TRAJ_TEXTS=np.asarray(TRAJ_TEXTS,dtype=object); LATEST_KEYS=np.asarray(LATEST_KEYS,dtype=object)

mods=modules.set_index("module_id").loc[MODULES].reset_index()
module_content_docs=[f"Modul pembelajaran Intelligo: {r.module_name}. Materi utama: {r.description_and_syllabus}." for r in mods.itertuples(index=False)]
module_full_docs=[f"Modul pembelajaran Intelligo: {r.module_name}. Materi utama: {r.description_and_syllabus}. Prasyarat: {r.prerequisite_level}." for r in mods.itertuples(index=False)]
QUERY_INSTRUCTION=(
    "Instruct: Representasikan kebutuhan BELAJAR siswa Intelligo untuk dicocokkan dengan modul kursus yang paling relevan. "
    "Fokus pada skill, topik, tujuan belajar, dan maksud terbaru. Abaikan pembayaran, akun, sertifikat, jadwal administratif, dan percakapan non-belajar.\nQuery: "
)

adapt_key_counts=Counter(LATEST_KEYS[ADAPT_IDX][LATEST_KEYS[ADAPT_IDX] != ""].tolist())
def seen_mask(indices):
    indices=np.asarray(indices,int); return np.asarray([(LATEST_KEYS[i]!="" and adapt_key_counts[LATEST_KEYS[i]]>0) for i in indices],bool)

print("semantic users train/test",int(HAS_SEMANTIC[:N_TRAIN].sum()),int(HAS_SEMANTIC[N_TRAIN:].sum()))
print("CAL seen/novel",int(seen_mask(CAL_IDX).sum()),int((HAS_SEMANTIC[CAL_IDX]&~seen_mask(CAL_IDX)).sum()))
print("LOCK seen/novel",int(seen_mask(LOCK_IDX).sum()),int((HAS_SEMANTIC[LOCK_IDX]&~seen_mask(LOCK_IDX)).sum()))

# ---- cell 14 (markdown) ----
# ## Qwen3-Embedding-8B QLoRA recommendation adaptation
#
# V21 fine-tunes the **embedding backbone**, but conservatively:
#
# - 4-bit NF4 QLoRA;
# - LoRA rank 8 on q/k/v/o attention projections in only the last 8 transformer layers;
# - one epoch, low learning rate, gradient clipping;
# - three simultaneous Matryoshka resolutions: 512 / 1024 / 4096;
# - 1024D receives the largest loss weight.
#
# The objective hierarchy is intentional:
#
# 1. **multi-positive relevance contrastive**: all organizer-positive modules must beat irrelevant modules;
# 2. **Consensus3-mined hard boundary pairs**: high-ranked false positives and low-ranked true positives receive extra pressure;
# 3. **graded positive ordering** only when relevance differences are meaningful (`>= 0.15`).
#
# No BGE score is used for negative mining.

# ---- cell 15 (code) ----
import torch
import torch.nn.functional as F

DEVICE=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def last_token_pool(last_hidden,attention_mask):
    mask=attention_mask.to(last_hidden.device)
    left_padding=bool((mask[:,-1].sum()==mask.shape[0]).item())
    if left_padding: return last_hidden[:,-1]
    seq=mask.sum(dim=1)-1
    return last_hidden[torch.arange(last_hidden.shape[0],device=last_hidden.device),seq]

def normalize_prefix(x,dim): return F.normalize(x[...,:dim].float(),dim=-1)

qwen_tokenizer=None; qwen_model=None

LORA_PROJECTION_SUFFIXES=('q_proj','k_proj','v_proj','o_proj')

def _select_lora_targets_from_names(module_names,last_n):
    """Select attention projection modules from the last N real transformer blocks.

    We intentionally do not use PEFT's layers_to_transform/layers_pattern here.
    Some bare AutoModel / feature-extraction architectures expose valid q/k/v/o
    projections but PEFT's layer-index matcher can fail before injection.  Exact
    discovered module names are accepted by PEFT and avoid that brittle path.
    """
    rows=[]
    for name in module_names:
        suffix=name.rsplit('.',1)[-1]
        if suffix not in LORA_PROJECTION_SUFFIXES:
            continue
        # Standard HF layouts: model.layers.28.self_attn.q_proj, layers.28....,
        # transformer.h.28...., blocks.28....
        m=re.search(r'(?:^|\.)(?:layers|h|blocks)\.(\d+)(?:\.|$)',name)
        if m is None:
            # Conservative fallback for custom wrappers where the numeric block is
            # immediately before an attention component.
            m=re.search(r'(?:^|\.)(\d+)\.(?:self_attn|attention|attn)\.',name)
        if m is not None:
            rows.append((int(m.group(1)),name,suffix))
    if not rows:
        sample=[n for n in module_names if any(x in n.lower() for x in ('attn','proj'))][:40]
        raise RuntimeError(
            'Could not discover Qwen LoRA attention projections from model.named_modules(). '
            f'Sample attention/projection module names: {sample}'
        )
    available=sorted({r[0] for r in rows})
    chosen=available[-min(int(last_n),len(available)):]
    chosen_set=set(chosen)
    targets=[name for layer,name,_ in rows if layer in chosen_set]
    counts=Counter(suffix for layer,name,suffix in rows if layer in chosen_set)
    # q/v are the canonical Qwen LoRA targets; k/o are included when present.
    if counts.get('q_proj',0)==0 or counts.get('v_proj',0)==0:
        raise RuntimeError(f'Qwen q_proj/v_proj were not found in selected layers {chosen}; counts={dict(counts)}')
    return targets,chosen,dict(counts)

def discover_lora_target_modules(base,last_n):
    names=[name for name,_ in base.named_modules()]
    return _select_lora_targets_from_names(names,last_n)

# Resolver regression test. This runs even in smoke mode and guards the exact bug
# that caused the Kaggle PEFT injection failure.
_fake_names=[f'model.layers.{i}.self_attn.{s}' for i in range(4) for s in LORA_PROJECTION_SUFFIXES]
_fake_targets,_fake_layers,_fake_counts=_select_lora_targets_from_names(_fake_names,2)
assert _fake_layers==[2,3] and len(_fake_targets)==8 and _fake_counts['q_proj']==2 and _fake_counts['v_proj']==2

def load_qlora_model():
    global qwen_tokenizer,qwen_model
    if V21_2_SMOKE or V21_2_SKIP_QWEN:
        print("SMOKE/SKIP: real QLoRA model not loaded")
        return None,None
    if not torch.cuda.is_available(): raise RuntimeError("V21 full run requires Kaggle GPU T4 x2 with Internet ON.")
    print("CUDA devices",torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        p=torch.cuda.get_device_properties(i); print(f"GPU {i}: {p.name}, {p.total_memory/1024**3:.1f} GiB")
    from transformers import AutoTokenizer,AutoModel,BitsAndBytesConfig
    from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training,TaskType
    qwen_tokenizer=AutoTokenizer.from_pretrained(QWEN_MODEL,revision=QWEN_REVISION,padding_side='left')
    if qwen_tokenizer.pad_token_id is None: qwen_tokenizer.pad_token=qwen_tokenizer.eos_token
    bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
    base=AutoModel.from_pretrained(
        QWEN_MODEL,revision=QWEN_REVISION,quantization_config=bnb,device_map={'':0},
        dtype=torch.float16,low_cpu_mem_usage=True,attn_implementation='sdpa'
    )
    base.config.use_cache=False
    base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=True)

    # IMPORTANT FIX: discover exact target names from the loaded model instead of
    # relying on PEFT layers_to_transform/layers_pattern matching.
    target_names,selected_layers,target_counts=discover_lora_target_modules(base,LORA_LAST_N_LAYERS)
    print('LoRA selected transformer layers',selected_layers)
    print('LoRA target counts',target_counts,'total',len(target_names))
    print('LoRA target sample',target_names[:8])
    cfg=LoraConfig(
        r=LORA_R,lora_alpha=LORA_ALPHA,lora_dropout=LORA_DROPOUT,bias='none',
        task_type=TaskType.FEATURE_EXTRACTION,target_modules=target_names
    )
    qwen_model=get_peft_model(base,cfg); qwen_model.config.use_cache=False

    injected=list(getattr(qwen_model.base_model,'targeted_module_names',[]) or [])
    if not injected:
        raise RuntimeError('PEFT returned no targeted_module_names after LoRA injection.')
    missing=[name for name in target_names if name not in injected]
    if missing:
        raise RuntimeError(f'PEFT failed to inject {len(missing)} requested LoRA modules; first missing={missing[:8]}')
    print('PEFT injection verified:',len(injected),'modules')
    print('PEFT injected sample',injected[:8])
    qwen_model.print_trainable_parameters(); qwen_model.train()
    return qwen_model,qwen_tokenizer

def tokenize_texts(tok,texts,max_length,instruction=''):
    prepared=[instruction+str(t) for t in texts]
    return tok(prepared,padding=True,truncation=True,max_length=max_length,return_tensors='pt')

def move_tokens(toks,device): return {k:v.to(device) for k,v in toks.items()}

def multi_positive_relevance_loss(scores,yb,anchor_rank):
    # scores: [B,17]. Primary objective treats every positive as relevant while preserving a small graded preference.
    pos=(yb>0).float(); gains=torch.pow(2.0,yb)-1.0
    pos_norm=pos/pos.sum(1,keepdim=True).clamp_min(1.0)
    gain_norm=gains/gains.sum(1,keepdim=True).clamp_min(1e-6)
    target=.75*pos_norm+.25*gain_norm
    rel=-(target*F.log_softmax(scores/LIST_TEMP,dim=1)).sum(1).mean()

    # True positive vs irrelevant pairs, with extra weight on Consensus3 top-6 false positives and positives outside anchor top-5.
    si=scores[:,:,None]; sj=scores[:,None,:]
    yi=yb[:,:,None]; yj=yb[:,None,:]
    posneg=(yi>0)&(yj<=0)
    ar_i=anchor_rank[:,:,None]; ar_j=anchor_rank[:,None,:]
    gain_i=(torch.pow(2.0,yi)-1.0).clamp_min(0)
    hard_w=(1.0+1.50*(ar_j<=5).float()+.60*(ar_i>=5).float())*gain_i.clamp_min(.15)
    hval=F.softplus((sj-si)/PAIR_TEMP)*hard_w
    hard=(hval*posneg.float()).sum()/((hard_w*posneg.float()).sum().clamp_min(1.0))

    # Only meaningful positive-positive grade gaps; ignore within-band jitter.
    graded=(yi>0)&(yj>0)&((yi-yj)>=.15)
    grade_w=(yi-yj).clamp_min(0)
    gval=F.softplus((sj-si)/GRADE_TEMP)*grade_w
    grade=(gval*graded.float()).sum()/((grade_w*graded.float()).sum().clamp_min(1.0))
    return LOSS_REL*rel+LOSS_HARD*hard+LOSS_GRADED*grade, (float(rel.detach()),float(hard.detach()),float(grade.detach()))

print("QLoRA config",dict(last_layers=LORA_LAST_N_LAYERS,r=LORA_R,alpha=LORA_ALPHA,lr=QLORA_LR,epochs=QLORA_EPOCHS,batch=QLORA_BATCH,dims=MULTI_DIMS))

# ---- cell 16 (code) ----
# Train exactly once on ADAPT. CAL and LOCK labels never update Qwen.
def train_qlora():
    if V21_2_SMOKE or V21_2_SKIP_QWEN:
        print("SMOKE: skipping real 8B QLoRA training")
        return None,None
    model,tok=load_qlora_model()
    train_idx=ADAPT_IDX[HAS_SEMANTIC[ADAPT_IDX]].copy()
    rng=np.random.default_rng(SEED+21000); rng.shuffle(train_idx)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=QLORA_LR,weight_decay=QLORA_WD)
    from transformers import get_cosine_schedule_with_warmup
    doc_content_cpu=tokenize_texts(tok,module_content_docs,MODULE_MAX_TOKENS,'')
    doc_full_cpu=tokenize_texts(tok,module_full_docs,MODULE_MAX_TOKENS,'')
    total_steps=max(1,math.ceil(len(train_idx)/QLORA_BATCH)*QLORA_EPOCHS)
    warm=max(1,int(.05*total_steps)); sched=get_cosine_schedule_with_warmup(opt,num_warmup_steps=warm,num_training_steps=total_steps); step=0; losses=[]
    for ep in range(QLORA_EPOCHS):
        rng.shuffle(train_idx)
        for st in range(0,len(train_idx),QLORA_BATCH):
            ids=train_idx[st:st+QLORA_BATCH]
            qtok=move_tokens(tokenize_texts(tok,TRAJ_TEXTS[ids],QUERY_MAX_TOKENS,QUERY_INSTRUCTION),DEVICE)
            # 75% content-only view, 25% full/prerequisite view. This keeps intent semantics primary.
            use_full=(step%4==3); dtok=move_tokens(doc_full_cpu if use_full else doc_content_cpu,DEVICE)
            qout=model(**qtok); dout=model(**dtok)
            q=last_token_pool(qout.last_hidden_state,qtok['attention_mask']); d=last_token_pool(dout.last_hidden_state,dtok['attention_mask'])
            yb=torch.as_tensor(y[ids],device=DEVICE,dtype=torch.float32)
            ar=torch.as_tensor(np.stack([ADAPT_TEACHER_RANKS[ADAPT_LOCAL[int(g)]] for g in ids]),device=DEVICE,dtype=torch.long)
            loss=torch.zeros((),device=DEVICE); parts=np.zeros(3,float)
            for w,dim in zip(DIM_WEIGHTS,MULTI_DIMS):
                qs=normalize_prefix(q,dim); ds=normalize_prefix(d,dim); scores=qs@ds.T
                ld,pp=multi_positive_relevance_loss(scores,yb,ar); loss=loss+float(w)*ld; parts+=float(w)*np.asarray(pp)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],QLORA_GRAD_CLIP); opt.step(); sched.step()
            step+=1; losses.append(float(loss.detach().cpu()))
            if step%25==0 or step==1 or step==total_steps:
                print(f"QLoRA step {step}/{total_steps} loss={losses[-1]:.5f} rel={parts[0]:.4f} hard={parts[1]:.4f} grade={parts[2]:.4f}")
            del qtok,dtok,qout,dout,q,d,yb,ar,loss
            if step%50==0: torch.cuda.empty_cache(); gc.collect()
    model.eval(); print("QLoRA mean last50 loss",round(float(np.mean(losses[-50:])),6))
    adapter_dir=ROOT/'v21_2_qwen_lora_adapter'; model.save_pretrained(adapter_dir); tok.save_pretrained(adapter_dir)
    return model,tok

qwen_model,qwen_tokenizer=train_qlora()

# ---- cell 17 (markdown) ----
# ## Semantic scoring after QLoRA
#
# Inference averages two module-document views (`75%` syllabus/content, `25%` full/prerequisite) and returns three cosine matrices. The 1024D view is the primary semantic signal; 512D and 4096D are retained as auxiliary calibration evidence.

# ---- cell 18 (code) ----
def fake_semantic_scores(indices,salt):
    indices=np.asarray(indices,int); out=[]
    for k,dim in enumerate(MULTI_DIMS):
        arr=np.zeros((len(indices),N_MODULES),np.float32)
        for ii,g in enumerate(indices):
            rs=np.random.default_rng((int(g)+1)*104729+(k+1)*1009)
            # Smoke-only deterministic signal; never used in full run.
            arr[ii]=.72*consensus3_oof[g] if g<N_TRAIN else .72*consensus3_test[g-N_TRAIN]
            arr[ii]+=.12*rs.standard_normal(N_MODULES)
        out.append(arr)
    return out

def score_indices(model,tok,indices,batch=12):
    indices=np.asarray(indices,int)
    if V21_2_SMOKE or V21_2_SKIP_QWEN:
        return fake_semantic_scores(indices,'smoke')
    model.eval()
    with torch.inference_mode():
        dc=move_tokens(tokenize_texts(tok,module_content_docs,MODULE_MAX_TOKENS,''),DEVICE); df=move_tokens(tokenize_texts(tok,module_full_docs,MODULE_MAX_TOKENS,''),DEVICE)
        oc=model(**dc); of=model(**df); hc=last_token_pool(oc.last_hidden_state,dc['attention_mask']); hf=last_token_pool(of.last_hidden_state,df['attention_mask'])
        docs=[]
        for dim in MULTI_DIMS:
            docs.append((.75*normalize_prefix(hc,dim)+.25*normalize_prefix(hf,dim)))
            docs[-1]=F.normalize(docs[-1],dim=-1)
        outs=[np.zeros((len(indices),N_MODULES),np.float32) for _ in MULTI_DIMS]
        for st in range(0,len(indices),batch):
            ids=indices[st:st+batch]; valid=HAS_SEMANTIC[ids]
            if not valid.any(): continue
            vv=ids[valid]
            qt=move_tokens(tokenize_texts(tok,TRAJ_TEXTS[vv],QUERY_MAX_TOKENS,QUERY_INSTRUCTION),DEVICE); qo=model(**qt); hq=last_token_pool(qo.last_hidden_state,qt['attention_mask'])
            local=np.flatnonzero(valid)
            for k,dim in enumerate(MULTI_DIMS):
                s=(normalize_prefix(hq,dim)@docs[k].T).float().cpu().numpy().astype(np.float32)
                outs[k][st+local]=s
            del qt,qo,hq
        del dc,df,oc,of,hc,hf,docs
    torch.cuda.empty_cache(); gc.collect(); return outs

CAL_SEM=score_indices(qwen_model,qwen_tokenizer,CAL_IDX)
LOCK_SEM=score_indices(qwen_model,qwen_tokenizer,LOCK_IDX)
TEST_GLOBAL=np.arange(N_TRAIN,N_TRAIN+N_TEST)
TEST_SEM=score_indices(qwen_model,qwen_tokenizer,TEST_GLOBAL)

for label,idx,sem in [("CAL",CAL_IDX,CAL_SEM),("LOCK",LOCK_IDX,LOCK_SEM)]:
    for dim,s in zip(MULTI_DIMS,sem):
        mask=HAS_SEMANTIC[idx]
        val=ndcg5(y[idx][mask],s[mask]) if mask.sum() else np.nan
        print(label,"semantic standalone dim",dim,"NDCG",round(float(val),6) if np.isfinite(val) else None)

# ---- cell 19 (markdown) ----
# ## Cross-fitted semantic relevance probabilities — feature only
#
# The V21/V21.1 binary relevance models are retained only as *features*. Their probabilities are never blended directly into the final rank. CAL predictions are strictly out-of-fold; LOCK/test predictions come from CAL-only full fits. A prior-normalized log-odds transform gives the LambdaRank translator a common-scale relevance-evidence feature.

# ---- cell 20 (code) ----
def rank_matrix(scores):
    order=np.argsort(-scores,axis=1); ranks=np.empty_like(order)
    ranks[np.arange(len(order))[:,None],order]=np.arange(1,N_MODULES+1)[None,:]
    return ranks.astype(np.float32)

def kth_score(scores,k): return np.sort(scores,axis=1)[:,-k]
def row_z_np(x):
    x=np.asarray(x,np.float32); return ((x-x.mean(1,keepdims=True))/np.maximum(x.std(1,keepdims=True),1e-6)).astype(np.float32)
def safe_logit(p):
    p=np.clip(np.asarray(p,np.float32),1e-4,1-1e-4); return np.log(p/(1-p)).astype(np.float32)

# Features used by the relevance stacker. Still module-wise because that was stable in AUC;
# cross-module comparability is handled later by LambdaRank, not by the probability scale itself.
def relevance_features(global_idx,base,meta,p5,anchor,sem_list):
    global_idx=np.asarray(global_idx,int); s512,s1024,s4096=[np.asarray(s,np.float32) for s in sem_list]
    ar=rank_matrix(anchor); sr=rank_matrix(s1024); a5=kth_score(anchor,5); a6=kth_score(anchor,6); s5=kth_score(s1024,5)
    z1024=row_z_np(s1024)
    user_small=np.column_stack([A[global_idx],A_INVALID[global_idx].sum(1),USER_FEATURES[global_idx,20:25]]).astype(np.float32)
    per_module=[]
    for m in range(N_MODULES):
        fm=np.column_stack([
            base[:,m],meta[:,m],p5[:,m],anchor[:,m],
            ar[:,m]/17.0,anchor[:,m]-a5,anchor[:,m]-a6,(a5-a6),
            s512[:,m],s1024[:,m],s4096[:,m],z1024[:,m],sr[:,m]/17.0,s1024[:,m]-s5,
            (sr[:,m]-ar[:,m])/17.0,user_small
        ]).astype(np.float32)
        per_module.append(fm)
    return per_module

CAL_REL_FEATS=relevance_features(CAL_IDX,AD_BASE_CAL,AD_META_CAL,AD_P5_CAL,AD_ANCHOR_CAL,CAL_SEM)
LOCK_REL_FEATS=relevance_features(LOCK_IDX,AD_BASE_LOCK,AD_META_LOCK,AD_P5_LOCK,AD_ANCHOR_LOCK,LOCK_SEM)
TEST_GLOBAL=np.arange(N_TRAIN,N_TRAIN+N_TEST)
TEST_REL_FEATS=relevance_features(TEST_GLOBAL,AD_BASE_TEST,AD_META_TEST,AD_P5_TEST,AD_ANCHOR_TEST,TEST_SEM)

def make_rel_cal(seed):
    return lgb.LGBMClassifier(
        n_estimators=REL_LGB_EST,learning_rate=.02,num_leaves=7,min_child_samples=(25 if V21_2_SMOKE else 70),
        reg_alpha=1.5,reg_lambda=12.0,colsample_bytree=.80,subsample=.90,subsample_freq=1,
        random_state=int(seed),n_jobs=4,verbosity=-1,force_col_wise=True
    )

cal_strata=np.asarray([f"{np.argmax(y[g]):02d}_{int((y[g]>0).sum())}" for g in CAL_IDX])
if V21_2_SMOKE:
    cal_folds=list(KFold(CAL_FOLDS,shuffle=True,random_state=CAL_SEED).split(np.zeros(len(CAL_IDX))))
else:
    cal_folds=list(StratifiedKFold(CAL_FOLDS,shuffle=True,random_state=CAL_SEED).split(np.zeros(len(CAL_IDX)),cal_strata))
cal_fold_id=np.full(len(CAL_IDX),-1,int)
cal_rel_oof=np.zeros((len(CAL_IDX),N_MODULES),np.float32)
for f,(tr,va) in enumerate(cal_folds):
    cal_fold_id[va]=f
    for m in range(N_MODULES):
        yy=Y_BIN[CAL_IDX,m].astype(np.int32); pred=np.zeros(len(va),np.float32)
        for off in REL_BAG_SEEDS:
            mdl=make_rel_cal(CAL_SEED+f*1009+m*37+off); mdl.fit(CAL_REL_FEATS[m][tr],yy[tr])
            pred+=mdl.predict_proba(CAL_REL_FEATS[m][va])[:,1].astype(np.float32)/len(REL_BAG_SEEDS)
        cal_rel_oof[va,m]=pred
assert (cal_fold_id>=0).all()

lock_rel=np.zeros((len(LOCK_IDX),N_MODULES),np.float32); test_rel=np.zeros((N_TEST,N_MODULES),np.float32)
for m in range(N_MODULES):
    yy=Y_BIN[CAL_IDX,m].astype(np.int32)
    for off in REL_BAG_SEEDS:
        mdl=make_rel_cal(CAL_SEED+9000+m*41+off); mdl.fit(CAL_REL_FEATS[m],yy)
        lock_rel[:,m]+=mdl.predict_proba(LOCK_REL_FEATS[m])[:,1].astype(np.float32)/len(REL_BAG_SEEDS)
        test_rel[:,m]+=mdl.predict_proba(TEST_REL_FEATS[m])[:,1].astype(np.float32)/len(REL_BAG_SEEDS)

# Hard-off semantic stack where the user has no learning chat.
cal_rel_oof[~HAS_SEMANTIC[CAL_IDX]]=AD_META_CAL[~HAS_SEMANTIC[CAL_IDX]]
lock_rel[~HAS_SEMANTIC[LOCK_IDX]]=AD_META_LOCK[~HAS_SEMANTIC[LOCK_IDX]]
test_rel[~HAS_SEMANTIC[TEST_GLOBAL]]=AD_META_TEST[~HAS_SEMANTIC[TEST_GLOBAL]]
ADAPT_PRIOR=np.clip(Y_BIN[ADAPT_IDX].mean(0),1e-3,1-1e-3).astype(np.float32)
ADAPT_PRIOR_LOGIT=safe_logit(ADAPT_PRIOR)
print('cross-fitted relevance feature ready; CAL fold sizes',[int((cal_fold_id==f).sum()) for f in range(CAL_FOLDS)])

# ---- cell 21 (markdown) ----
# ## Anchor-initialized shared LambdaRank translator
#
# Each learner is one ranking group with exactly 17 modules. LambdaRank starts from the ADAPT-only Consensus3 anchor (`init_score`) and learns only a shallow residual. It receives QLoRA similarities, cross-fitted relevance evidence, anchor uncertainty, assessments, and module identity. Five tiny rankers are bagged; their *relative preferences* are later projected back onto Consensus3 instead of using raw ranker scores.

# ---- cell 22 (code) ----
# Ordinal grades exactly follow the organizer's rigid positive rank bands: #1=6,...,#6=1, irrelevant=0.
def ordinal_grades(global_idx):
    rp=RANK_POS[np.asarray(global_idx,int)]
    return np.where(rp<=6,7-rp,0).astype(np.int32)

def make_rank_features(global_idx,base,meta,p5,anchor,sem_list,rel_prob):
    global_idx=np.asarray(global_idx,int); s512,s1024,s4096=[np.asarray(s,np.float32) for s in sem_list]
    ar=rank_matrix(anchor); sr=rank_matrix(s1024); az=row_z_np(anchor); sz=row_z_np(s1024)
    a1=kth_score(anchor,1); a5=kth_score(anchor,5); a6=kth_score(anchor,6); s5=kth_score(s1024,5)
    rel_logit=safe_logit(rel_prob); evidence=(rel_logit-ADAPT_PRIOR_LOGIT[None,:]).astype(np.float32)
    rel_z=row_z_np(evidence)
    n=len(global_idx); eye=np.eye(N_MODULES,dtype=np.float32)
    rows=[]
    for m in range(N_MODULES):
        core=np.column_stack([
            base[:,m],meta[:,m],p5[:,m],anchor[:,m],az[:,m],
            ar[:,m]/17.0,anchor[:,m]-a1,anchor[:,m]-a5,anchor[:,m]-a6,(a5-a6),
            s512[:,m],s1024[:,m],s4096[:,m],sz[:,m],sr[:,m]/17.0,s1024[:,m]-s5,(sr[:,m]-ar[:,m])/17.0,
            rel_prob[:,m],rel_logit[:,m],evidence[:,m],rel_z[:,m],rel_prob[:,m]-meta[:,m],
            np.full(n,ADAPT_PRIOR[m],np.float32),
            A[global_idx],A_INVALID[global_idx].sum(1)[:,None]
        ]).astype(np.float32)
        mod=np.repeat(eye[m][None,:],n,axis=0)
        rows.append(np.column_stack([core,mod]).astype(np.float32))
    # module-major -> user-major [n,17,F]
    arr=np.stack(rows,axis=1)
    return arr,evidence

CAL_RANK_X,CAL_EVIDENCE=make_rank_features(CAL_IDX,AD_BASE_CAL,AD_META_CAL,AD_P5_CAL,AD_ANCHOR_CAL,CAL_SEM,cal_rel_oof)
LOCK_RANK_X,LOCK_EVIDENCE=make_rank_features(LOCK_IDX,AD_BASE_LOCK,AD_META_LOCK,AD_P5_LOCK,AD_ANCHOR_LOCK,LOCK_SEM,lock_rel)
TEST_RANK_X,TEST_EVIDENCE=make_rank_features(TEST_GLOBAL,AD_BASE_TEST,AD_META_TEST,AD_P5_TEST,AD_ANCHOR_TEST,TEST_SEM,test_rel)
FEATURE_COUNT=CAL_RANK_X.shape[-1]
print('rank features',CAL_RANK_X.shape,'feature_count',FEATURE_COUNT)

def flatten_users(x,idx): return np.asarray(x[idx],np.float32).reshape(-1,x.shape[-1])
def flatten_scores(x,idx): return np.asarray(x[idx],np.float32).reshape(-1)
def flatten_labels(x,idx): return np.asarray(x[idx],np.int32).reshape(-1)

def rank_params(seed):
    return {
        'objective':'lambdarank','metric':'None','learning_rate':RANK_LR,'num_leaves':3,'max_depth':2,
        'min_data_in_leaf':RANK_MIN_LEAF,'lambda_l1':1.5,'lambda_l2':24.0,'min_gain_to_split':0.0,
        'feature_fraction':.82,'bagging_fraction':.90,'bagging_freq':1,'verbosity':-1,'num_threads':4,
        'seed':int(seed),'feature_fraction_seed':int(seed)+11,'bagging_seed':int(seed)+23,
        'deterministic':True,'force_col_wise':True,'lambdarank_truncation_level':RANK_TRUNCATION,
        'lambdarank_norm':True,'label_gain':RANK_LABEL_GAIN
    }

def train_ranker(x_users,labels_users,anchor_users,train_users,seed):
    tr=np.asarray(train_users,int)
    Xf=flatten_users(x_users,tr); yf=flatten_labels(labels_users,tr)
    init=flatten_scores(row_z_np(anchor_users[tr]),np.arange(len(tr)))
    ds=lgb.Dataset(Xf,label=yf,group=np.full(len(tr),N_MODULES,int),init_score=init,free_raw_data=False)
    return lgb.train(rank_params(seed),ds,num_boost_round=RANK_NUM_BOOST)

def predict_ranker(booster,x_users,anchor_users,user_idx):
    ui=np.asarray(user_idx,int); Xf=flatten_users(x_users,ui)
    residual=booster.predict(Xf,num_iteration=booster.current_iteration()).reshape(len(ui),N_MODULES).astype(np.float32)
    return (row_z_np(anchor_users[ui])+residual).astype(np.float32)

CAL_GRADES=ordinal_grades(CAL_IDX)
# OOF ranker scores for every seed. Each CAL user is scored only by rankers that never saw its labels.
cal_rank_seed=np.zeros((len(CAL_IDX),len(RANK_BAG_SEEDS),N_MODULES),np.float32)
for f,(tr,va) in enumerate(cal_folds):
    for si,off in enumerate(RANK_BAG_SEEDS):
        bst=train_ranker(CAL_RANK_X,CAL_GRADES,AD_ANCHOR_CAL,tr,RANK_SEED+f*1000+off)
        cal_rank_seed[va,si]=predict_ranker(bst,CAL_RANK_X,AD_ANCHOR_CAL,va)

# Final CAL-only rankers for the untouched LOCK and test.
lock_rank_seed=np.zeros((len(LOCK_IDX),len(RANK_BAG_SEEDS),N_MODULES),np.float32)
test_rank_seed=np.zeros((N_TEST,len(RANK_BAG_SEEDS),N_MODULES),np.float32)
all_cal=np.arange(len(CAL_IDX))
for si,off in enumerate(RANK_BAG_SEEDS):
    bst=train_ranker(CAL_RANK_X,CAL_GRADES,AD_ANCHOR_CAL,all_cal,RANK_SEED+9000+off)
    lock_rank_seed[:,si]=predict_ranker(bst,LOCK_RANK_X,AD_ANCHOR_LOCK,np.arange(len(LOCK_IDX)))
    test_rank_seed[:,si]=predict_ranker(bst,TEST_RANK_X,AD_ANCHOR_TEST,np.arange(N_TEST))
print('anchor-initialized LambdaRank bags ready')

# ---- cell 23 (markdown) ----
# ## Trust-region projection and stability-first CAL selection
#
# `rank_safe` only reorders the five modules already selected by the anchor, and only on unanimous adjacent pair preferences across the ranker bag. `rank_boundary` starts from the same safe ordering and may replace the fifth module by anchor #6/#7 only when at least 4/5 rankers agree, the candidate is repeatedly predicted inside the top-5, QLoRA common-scale relevance evidence agrees, raw 1024D similarity agrees, and the anchor itself is uncertain. Raw LambdaRank scores are never submitted.

# ---- cell 24 (code) ----
def score_order(scores): return np.argsort(-np.asarray(scores),axis=1)

def pair_votes(seed_scores,a,b):
    # seed_scores shape [S,17]; positive vote means a should outrank b.
    return int(np.sum(seed_scores[:,a]>seed_scores[:,b]))

def project_one(anchor_row,seed_rows,evidence_row,sem1024_row,has_sem,mode,gap_threshold,order_gap_threshold,top1_gap_threshold):
    order=list(np.argsort(-anchor_row))
    if not has_sem: return order
    # Rank-safe trust region: at most ONE adjacent Top-5 swap per learner.
    # The inversion must be unanimous across the bag AND the anchor must be locally uncertain.
    top=order[:5]; az=row_z_np(anchor_row[None,:])[0]
    proposals=[]
    for j in range(4):
        a,b=top[j],top[j+1]
        votes=pair_votes(seed_rows,b,a)
        gap=float(az[a]-az[b])
        allowed_gap=top1_gap_threshold if j==0 else order_gap_threshold
        semantic_ok=bool(evidence_row[b]>evidence_row[a] and sem1024_row[b]>sem1024_row[a])
        if votes>=SAFE_PAIR_AGREE and gap<=allowed_gap and semantic_ok:
            mean_margin=float(np.mean(seed_rows[:,b]-seed_rows[:,a]))
            proposals.append((mean_margin,-gap,-j,j))
    if proposals:
        _,_,_,j=max(proposals)
        top[j],top[j+1]=top[j+1],top[j]
    rest=[m for m in order if m not in top]
    out=top+rest
    if mode!='rank_boundary': return out

    current5=out[4]
    anchor_order=np.argsort(-anchor_row)
    candidates=[int(anchor_order[5]),int(anchor_order[6])]
    best=None
    for c in candidates:
        votes=pair_votes(seed_rows,c,current5)
        top5_votes=int(np.sum([np.where(np.argsort(-seed_rows[s])==c)[0][0] < 5 for s in range(seed_rows.shape[0])]))
        gap=float(row_z_np(anchor_row[None,:])[0,current5]-row_z_np(anchor_row[None,:])[0,c])
        semantic_ok=bool(evidence_row[c]>evidence_row[current5] and sem1024_row[c]>sem1024_row[current5])
        if votes>=BOUNDARY_PAIR_AGREE and top5_votes>=BOUNDARY_TOP5_AGREE and semantic_ok and gap<=gap_threshold:
            mean_rank=float(np.mean([np.where(np.argsort(-seed_rows[s])==c)[0][0] for s in range(seed_rows.shape[0])]))
            key=(mean_rank,-votes,-float(evidence_row[c]-evidence_row[current5]))
            if best is None or key<best[0]: best=(key,c)
    if best is not None:
        c=best[1]; pos=out.index(c); out[4],out[pos]=c,current5
    return out

def permute_anchor_scores(anchor,desired_orders):
    anchor=np.asarray(anchor,np.float32); out=anchor.copy()
    for i,desired in enumerate(desired_orders):
        base_order=np.argsort(-anchor[i]); sorted_vals=anchor[i,base_order].copy()
        out[i,np.asarray(desired,int)]=sorted_vals
    return out

def build_projected(anchor,seed_scores,evidence,sem1024,has_sem,mode,gap_threshold):
    orders=[]
    for i in range(len(anchor)):
        orders.append(project_one(anchor[i],seed_scores[i],evidence[i],sem1024[i],bool(has_sem[i]),mode,gap_threshold,ORDER_GAP_THRESHOLD,ORDER_TOP1_GAP_THRESHOLD))
    return permute_anchor_scores(anchor,orders)

# Unsupervised CAL-only anchor-uncertainty threshold; no relevance labels are used to set it.
cal_anchor_order=np.argsort(-AD_ANCHOR_CAL,axis=1)
cal_az=row_z_np(AD_ANCHOR_CAL)
cal_boundary_gaps=np.asarray([cal_az[i,cal_anchor_order[i,4]]-cal_az[i,cal_anchor_order[i,5]] for i in range(len(CAL_IDX))])
cal_order_gaps=[]; cal_top1_gaps=[]
for i in range(len(CAL_IDX)):
    oo=cal_anchor_order[i]
    cal_top1_gaps.append(float(cal_az[i,oo[0]]-cal_az[i,oo[1]]))
    for j in range(1,4): cal_order_gaps.append(float(cal_az[i,oo[j]]-cal_az[i,oo[j+1]]))
ORDER_GAP_THRESHOLD=float(np.quantile(np.asarray(cal_order_gaps),ORDER_GAP_QUANTILE))
ORDER_TOP1_GAP_THRESHOLD=float(np.quantile(np.asarray(cal_top1_gaps),ORDER_TOP1_GAP_QUANTILE))
BOUNDARY_GAP_THRESHOLD=float(np.quantile(cal_boundary_gaps,BOUNDARY_GAP_QUANTILE))
print('trust-region z-gap thresholds (CAL X only): order',round(ORDER_GAP_THRESHOLD,6),'top1',round(ORDER_TOP1_GAP_THRESHOLD,6),'boundary',round(BOUNDARY_GAP_THRESHOLD,6))

CAL_HAS=HAS_SEMANTIC[CAL_IDX]
cal_candidates={
    'rank_safe':build_projected(AD_ANCHOR_CAL,cal_rank_seed,CAL_EVIDENCE,CAL_SEM[1],CAL_HAS,'rank_safe',BOUNDARY_GAP_THRESHOLD),
    'rank_boundary':build_projected(AD_ANCHOR_CAL,cal_rank_seed,CAL_EVIDENCE,CAL_SEM[1],CAL_HAS,'rank_boundary',BOUNDARY_GAP_THRESHOLD)
}

def subgroup_delta(ytrue,cand,anchor,mask):
    mask=np.asarray(mask,bool)
    if mask.sum()<10:return np.nan
    return float(ndcg5(ytrue[mask],cand[mask])-ndcg5(ytrue[mask],anchor[mask]))

CAL_SEEN=seen_mask(CAL_IDX); CAL_NOVEL=CAL_HAS&~CAL_SEEN
cal_rows=[]
for name,cand in cal_candidates.items():
    delta=ndcg5(y[CAL_IDX],cand)-ndcg5(y[CAL_IDX],AD_ANCHOR_CAL)
    fold_d=[]
    for f in range(CAL_FOLDS):
        vv=cal_fold_id==f; fold_d.append(ndcg5(y[CAL_IDX][vv],cand[vv])-ndcg5(y[CAL_IDX][vv],AD_ANCHOR_CAL[vv]))
    fold_d=np.asarray(fold_d,float); wins=int((fold_d>0).sum()); std=float(fold_d.std(ddof=0)); stability=float(delta-.50*std)
    cal_rows.append(dict(candidate=name,cal_delta=float(delta),fold_wins=wins,fold_std=std,stability_score=stability,
        fold_deltas=fold_d.tolist(),seen_delta=subgroup_delta(y[CAL_IDX],cand,AD_ANCHOR_CAL,CAL_SEEN),novel_delta=subgroup_delta(y[CAL_IDX],cand,AD_ANCHOR_CAL,CAL_NOVEL),
        set_change=top5_set_change_rate(cand,AD_ANCHOR_CAL),order_change=top5_order_change_rate(cand,AD_ANCHOR_CAL),top1_change=top1_change_rate(cand,AD_ANCHOR_CAL)))
    print(name,'CAL delta',round(delta,6),'wins',wins,'/',CAL_FOLDS,'std',round(std,6),'stability',round(stability,6),'set',round(top5_set_change_rate(cand,AD_ANCHOR_CAL),4),'order',round(top5_order_change_rate(cand,AD_ANCHOR_CAL),4))

# Stability-first selection: eligible candidates need >=3/4 fold wins; then maximize mean - 0.5*fold_std.
eligible=[r for r in cal_rows if r['fold_wins']>=MIN_CAL_WINS]
if eligible: SELECTED=max(eligible,key=lambda r:(r['stability_score'],r['cal_delta'],-r['set_change']))['candidate']
else: SELECTED=max(cal_rows,key=lambda r:(r['fold_wins'],r['stability_score'],r['cal_delta']))['candidate']
selected_cal=next(r for r in cal_rows if r['candidate']==SELECTED)
CAL_GATE=bool(selected_cal['cal_delta']>=MIN_CAL_DELTA and selected_cal['fold_wins']>=MIN_CAL_WINS and selected_cal['stability_score']>=MIN_CAL_STABILITY)
print('FROZEN BEFORE LOCK',SELECTED,'CAL_GATE',CAL_GATE,'folds',selected_cal['fold_deltas'])

# ---- cell 25 (markdown) ----
# ## Fresh LOCK opening, exact deployment projection, and promotion
#
# If the CAL stability gate fails, LOCK labels are not read at all and the primary output is Consensus3. If CAL passes, exactly one frozen candidate is evaluated on the fresh LOCK. The same discrete projection is then applied to the *full-data public-proven Consensus3* on test; this transports only relative rank preferences, never ADAPT-only score magnitudes.

# ---- cell 26 (code) ----
lock_delta=np.nan; lock_ci=(np.nan,np.nan); lock_seen=np.nan; lock_novel=np.nan; LOCK_GATE=False
lock_candidate=None
if CAL_GATE:
    LOCK_HAS=HAS_SEMANTIC[LOCK_IDX]
    lock_candidate=build_projected(AD_ANCHOR_LOCK,lock_rank_seed,LOCK_EVIDENCE,LOCK_SEM[1],LOCK_HAS,SELECTED,BOUNDARY_GAP_THRESHOLD)
    lock_delta=ndcg5(y[LOCK_IDX],lock_candidate)-ndcg5(y[LOCK_IDX],AD_ANCHOR_LOCK)
    lock_user_delta=per_user_ndcg(y[LOCK_IDX],lock_candidate)-per_user_ndcg(y[LOCK_IDX],AD_ANCHOR_LOCK)
    lock_ci=paired_boot(lock_user_delta,BOOT_SEED)
    LOCK_SEEN=seen_mask(LOCK_IDX); LOCK_NOVEL=LOCK_HAS&~LOCK_SEEN
    lock_seen=subgroup_delta(y[LOCK_IDX],lock_candidate,AD_ANCHOR_LOCK,LOCK_SEEN); lock_novel=subgroup_delta(y[LOCK_IDX],lock_candidate,AD_ANCHOR_LOCK,LOCK_NOVEL)
    LOCK_GATE=bool(lock_delta>=MIN_LOCK_DELTA and lock_ci[0]>=MIN_LOCK_CI_LOW)
    print('LOCK OPENED once: delta',round(lock_delta,6),'CI',tuple(round(float(v),6) for v in lock_ci),'seen',round(lock_seen,6) if np.isfinite(lock_seen) else None,'novel',round(lock_novel,6) if np.isfinite(lock_novel) else None,'LOCK_GATE',LOCK_GATE)
else:
    print('LOCK remains unopened because CAL gate failed.')

# Exact deployment: derive preference order from ADAPT/CAL-trained rankers, then permute full Consensus3 scores.
TEST_HAS=HAS_SEMANTIC[TEST_GLOBAL]
final_attacks={
    'rank_safe':build_projected(consensus3_test,test_rank_seed,TEST_EVIDENCE,TEST_SEM[1],TEST_HAS,'rank_safe',BOUNDARY_GAP_THRESHOLD),
    'rank_boundary':build_projected(consensus3_test,test_rank_seed,TEST_EVIDENCE,TEST_SEM[1],TEST_HAS,'rank_boundary',BOUNDARY_GAP_THRESHOLD)
}
for name,cand in final_attacks.items():
    print(name,'test set',round(top5_set_change_rate(cand,consensus3_test),4),'order',round(top5_order_change_rate(cand,consensus3_test),4),'top1',round(top1_change_rate(cand,consensus3_test),4))

selected_test=final_attacks[SELECTED]
TEST_SET=top5_set_change_rate(selected_test,consensus3_test); TEST_ORDER=top5_order_change_rate(selected_test,consensus3_test); TEST_TOP1=top1_change_rate(selected_test,consensus3_test)
PERTURB_GATE=bool(TEST_SET<=MAX_TEST_SET_CHANGE and TEST_ORDER<=MAX_TEST_ORDER_CHANGE and TEST_TOP1<=MAX_TEST_TOP1_CHANGE)
PROMOTE=bool(CAL_GATE and LOCK_GATE and PERTURB_GATE)
primary_test=selected_test.copy() if PROMOTE else consensus3_test.copy(); primary_label=SELECTED if PROMOTE else 'consensus3_fallback'
print('PERTURB_GATE',PERTURB_GATE,'PROMOTE',PROMOTE,'PRIMARY',primary_label)

# ---- cell 27 (markdown) ----
# ## Diagnostics and artifacts

# ---- cell 28 (code) ----
def rank_contrib(ytrue,cand,anchor):
    tc=top5_indices(cand); ta=top5_indices(anchor); out=[]
    ideal=np.sort(ytrue,axis=1)[:,-5:][:,::-1]; denom=np.maximum((ideal*DISCOUNT).sum(1),1e-12)
    for k in range(5):
        d=(ytrue[np.arange(len(ytrue)),tc[:,k]]-ytrue[np.arange(len(ytrue)),ta[:,k]])*DISCOUNT[k]/denom
        out.append(float(d.mean()))
    return out

diag_rows=[]
for r in cal_rows:
    cand=final_attacks[r['candidate']]
    diag_rows.append({
        'candidate':r['candidate'],'selected':r['candidate']==SELECTED,'cal_delta':r['cal_delta'],'cal_fold_wins':r['fold_wins'],'cal_fold_std':r['fold_std'],'cal_stability_score':r['stability_score'],
        'cal_seen_delta':r['seen_delta'],'cal_novel_delta':r['novel_delta'],'cal_set_change':r['set_change'],'cal_order_change':r['order_change'],'cal_top1_change':r['top1_change'],
        'lock_delta':lock_delta if r['candidate']==SELECTED and CAL_GATE else np.nan,'lock_ci_low':lock_ci[0] if r['candidate']==SELECTED and CAL_GATE else np.nan,'lock_ci_high':lock_ci[1] if r['candidate']==SELECTED and CAL_GATE else np.nan,
        'lock_seen_delta':lock_seen if r['candidate']==SELECTED and CAL_GATE else np.nan,'lock_novel_delta':lock_novel if r['candidate']==SELECTED and CAL_GATE else np.nan,
        'test_top5_set_change':top5_set_change_rate(cand,consensus3_test),'test_order_change':top5_order_change_rate(cand,consensus3_test),'test_top1_change':top1_change_rate(cand,consensus3_test),
        'promoted':bool(PROMOTE and r['candidate']==SELECTED)
    })
diag=pd.DataFrame(diag_rows); display(diag)
print('================ V21.2 DECISION ================')
print('frozen candidate      :',SELECTED)
print('CAL delta / wins / std:',round(selected_cal['cal_delta'],6),selected_cal['fold_wins'],'/',CAL_FOLDS,round(selected_cal['fold_std'],6))
print('CAL stability score   :',round(selected_cal['stability_score'],6),'folds',[round(float(v),6) for v in selected_cal['fold_deltas']])
print('CAL seen / novel diag :',round(selected_cal['seen_delta'],6) if np.isfinite(selected_cal['seen_delta']) else None,round(selected_cal['novel_delta'],6) if np.isfinite(selected_cal['novel_delta']) else None)
print('LOCK opened?          :',bool(CAL_GATE))
print('LOCK delta / CI       :',round(float(lock_delta),6) if np.isfinite(lock_delta) else None,tuple(round(float(v),6) for v in lock_ci) if np.isfinite(lock_ci[0]) else None)
print('LOCK seen / novel diag:',round(float(lock_seen),6) if np.isfinite(lock_seen) else None,round(float(lock_novel),6) if np.isfinite(lock_novel) else None)
print('test perturbation     : set',round(TEST_SET,4),'order',round(TEST_ORDER,4),'top1',round(TEST_TOP1,4))
print('gates                 : CAL',CAL_GATE,'LOCK',LOCK_GATE,'PERTURB',PERTURB_GATE)
print('PRIMARY               :',primary_label)
print('public anchor/top2/top1:',PUBLIC_CONSENSUS3_LB,PUBLIC_TOP2_LB,PUBLIC_LEADER_LB)
if CAL_GATE:
    print('LOCK rank contributions:',[round(v,6) for v in rank_contrib(y[LOCK_IDX],lock_candidate,AD_ANCHOR_LOCK)])

# ---- cell 29 (code) ----
def save_submission(scores,name):
    out=sample.copy(); out.loc[:,MODULES]=np.asarray(scores,np.float32)
    path=ROOT/name; out.to_csv(path,index=False); print('saved',path); return str(path)

paths={}
paths['attack1']=save_submission(final_attacks['rank_safe'],'submission_v21_2_attack1.csv')
paths['attack2']=save_submission(final_attacks['rank_boundary'],'submission_v21_2_attack2.csv')
paths['primary']=save_submission(primary_test,'submission_v21_2_primary.csv')
paths['consensus3']=save_submission(consensus3_test,'submission_v21_2_consensus3_reference.csv')
diag.to_csv(ROOT/'v21_2_diagnostics.csv',index=False)
summary={
    'architecture':'anchor-initialized bagged LambdaRank + discrete trust-region projection',
    'public_consensus3_lb':PUBLIC_CONSENSUS3_LB,'public_top2_lb':PUBLIC_TOP2_LB,'public_leader_lb':PUBLIC_LEADER_LB,
    'selected':SELECTED,'selected_cal':selected_cal,'cal_gate':bool(CAL_GATE),'lock_opened':bool(CAL_GATE),
    'lock_delta':float(lock_delta) if np.isfinite(lock_delta) else None,'lock_ci':[float(v) for v in lock_ci] if np.isfinite(lock_ci[0]) else None,'lock_gate':bool(LOCK_GATE),
    'perturb_gate':bool(PERTURB_GATE),'promote':bool(PROMOTE),'primary_label':primary_label,
    'test_set_change':TEST_SET,'test_order_change':TEST_ORDER,'test_top1_change':TEST_TOP1,
    'split_sizes':{'adapt':len(ADAPT_IDX),'cal':len(CAL_IDX),'lock':len(LOCK_IDX)},'dims':list(MULTI_DIMS),'rank_bag_seeds':list(RANK_BAG_SEEDS),
    'rank_num_boost':RANK_NUM_BOOST,'rank_truncation':RANK_TRUNCATION,'order_gap_threshold':ORDER_GAP_THRESHOLD,'top1_gap_threshold':ORDER_TOP1_GAP_THRESHOLD,'boundary_gap_threshold':BOUNDARY_GAP_THRESHOLD
}
with open(ROOT/'v21_2_summary.json','w') as f: json.dump(summary,f,indent=2,default=lambda x:float(x) if isinstance(x,(np.floating,np.integer)) else str(x))
np.savez_compressed(ROOT/'v21_2_oof_artifacts.npz',
    y=y,consensus3_oof=consensus3_oof,consensus3_test=consensus3_test,adapt_idx=ADAPT_IDX,cal_idx=CAL_IDX,lock_idx=LOCK_IDX,
    cal_rel_oof=cal_rel_oof,cal_rank_seed=cal_rank_seed,test_rank_seed=test_rank_seed,
    **{f'cal_sem_{d}':s for d,s in zip(MULTI_DIMS,CAL_SEM)},**{f'lock_sem_{d}':s for d,s in zip(MULTI_DIMS,LOCK_SEM)})
print('artifacts complete')

# ---- cell 30 (code) ----
# Hard invariants — fail loudly rather than silently emit a risky submission.
assert y.shape==(N_TRAIN,17) and len(MODULES)==17
assert np.isfinite(consensus3_oof).all() and np.isfinite(consensus3_test).all()
assert np.isfinite(cal_rel_oof).all() and np.isfinite(lock_rel).all() and np.isfinite(test_rel).all()
assert np.isfinite(cal_rank_seed).all() and np.isfinite(lock_rank_seed).all() and np.isfinite(test_rank_seed).all()
assert np.isfinite(primary_test).all()
assert len(set(ADAPT_IDX)&set(CAL_IDX))==0 and len(set(ADAPT_IDX)&set(LOCK_IDX))==0 and len(set(CAL_IDX)&set(LOCK_IDX))==0
assert len(ADAPT_IDX)+len(CAL_IDX)+len(LOCK_IDX)==N_TRAIN
# Rank-safe must preserve membership exactly; rank-boundary is limited to at most one entering module per user by construction.
assert top5_set_change_rate(final_attacks['rank_safe'],consensus3_test)==0.0
nochat_test=~HAS_SEMANTIC[TEST_GLOBAL]
for name,cand in final_attacks.items():
    if nochat_test.any(): assert np.allclose(cand[nochat_test],consensus3_test[nochat_test],atol=1e-7)
if not PROMOTE: assert np.allclose(primary_test,consensus3_test,atol=0,rtol=0)
if not V21_2_SMOKE:
    assert abs(ndcg5(y,consensus3_oof)-0.6643682395561207)<0.0012,ndcg5(y,consensus3_oof)
assert list(sample.columns[1:])==MODULES
print('V21.2 hard invariants: PASS')
