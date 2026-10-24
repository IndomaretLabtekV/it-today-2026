# Converted from notebooks/v22.ipynb: markdown cells become comments, %/! magics neutralized.

# ---- cell 0 (markdown) ----
# # MineToday V22 — Surgical M003/M014 Semantic Residual
#
# **Goal:** turn the only public-positive semantic result (V21.1 `stable035`, LB **0.65869**) into three tightly controlled, deployment-aligned ablations without giving the semantic branch freedom to rerank the full 17-module list.
#
# V22 keeps the exact V21.1 representation/calibration path and changes only the final decision policy:
#
# - exact V3.1 sparse + V12.2 Consensus3 remain the immutable structural anchor;
# - exact V21.1 Qwen3-Embedding-8B QLoRA recipe, split, 3-seed bagged semantic-aware calibrators, and no-chat hard-off are retained;
# - the public-positive V21.1 behavior is decomposed into the two modules responsible for almost all test rank changes: `M_003` and `M_014`;
# - three attacks are predeclared before labels are inspected: `M014@3.5%`, `M003+M014@3.5%`, and `M003+M014@5.0%`;
# - every candidate has a **hard Top-1 row lock**: if a semantic correction would change Consensus3 Top-1, the entire user row is reverted to the anchor;
# - unlike V21/V21.1, validation is performed on the **same full-Consensus3 decision surface used at test time**. The semantic residual is transported onto full OOF base/meta scores before CAL/LOCK NDCG is measured;
# - no dynamic module search, free LambdaRank, router, graph model, or global semantic blend is introduced;
# - `submission_v22_primary.csv` is the best predeclared candidate only if deployment-aligned CAL+LOCK gates pass; otherwise it falls back to the reproducible V21.1-style four-module `stable035` reference.
#
# ### Three leaderboard files
# 1. `submission_v22_attack1_m014035.csv` — safest module ablation.
# 2. `submission_v22_attack2_m003_m014035.csv` — closest surgical decomposition of the 0.65869 public winner.
# 3. `submission_v22_attack3_m003_m014050.csv` — controlled higher-dose variant for upside.
#
# ### Kaggle runtime
# Use **GPU T4 x2** with **Internet ON**. QLoRA training is pinned to GPU 0 exactly as in V21.1; downstream LightGBM runs on CPU.

# ---- cell 1 (code) ----
import os,sys,subprocess
V21_SKIP_INSTALL=bool(int(os.getenv("V21_SKIP_INSTALL","0")))
if not V21_SKIP_INSTALL:
    subprocess.check_call([
        sys.executable,"-m","pip","install","-q","-U",
        "transformers==4.57.6","accelerate==1.14.0","safetensors>=0.5,<1",
        "huggingface_hub>=0.34,<2","lightgbm>=4.5,<5","sentencepiece>=0.2,<1",
        "peft>=0.17,<0.19","bitsandbytes>=0.46,<0.49"
    ])
print("dependency setup complete; skip_install=",V21_SKIP_INSTALL)

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
SPLIT_SEED=2026082521
CAL_SEED=2026082522
BOOT_SEED=2026082523
V21_SMOKE=bool(int(os.getenv("V21_SMOKE","0")))
V21_SKIP_QWEN=bool(int(os.getenv("V21_SKIP_QWEN","0")))
V21_UNUSED_BGE=True
N_FOLDS=2 if V21_SMOKE else 5
RIDGE_ALPHA=3.0
REFERENCE_W=0.45
LGB_ESTIMATORS=60 if V21_SMOKE else 1200
LGB_EARLY_STOP=6 if V21_SMOKE else 80
AUDIT_LGB_ESTIMATORS=45 if V21_SMOKE else 700
AUDIT_EARLY_STOP=5 if V21_SMOKE else 50
BOOTSTRAPS=250 if V21_SMOKE else 10000

PUBLIC_CONSENSUS3_LB=0.65856
PUBLIC_V21_1_LB=0.65869
PUBLIC_LEADER_LB=0.66082
PUBLIC_V18_LB=0.65829
PUBLIC_V19_LB=0.65791

QWEN_MODEL="Qwen/Qwen3-Embedding-8B"
QWEN_REVISION="1d8ad4c"
QUERY_MAX_TOKENS=96 if V21_SMOKE else 160
MODULE_MAX_TOKENS=128 if V21_SMOKE else 224
MAX_TRAJECTORY_MESSAGES=3
MULTI_DIMS=(32,48,64) if V21_SMOKE else (512,1024,4096)
DIM_WEIGHTS=np.asarray([.15,.60,.25],np.float32)
PRIMARY_DIM=MULTI_DIMS[1]

# Conservative QLoRA: attention projections only in the last 8 transformer blocks.
LORA_LAST_N_LAYERS=8
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=.05
QLORA_LR=5e-5
QLORA_WD=.01
QLORA_EPOCHS=1
QLORA_BATCH=16 if V21_SMOKE else 4
QLORA_GRAD_CLIP=1.0
LIST_TEMP=.075
PAIR_TEMP=.08
GRADE_TEMP=.09
LOSS_REL=.60
LOSS_HARD=.25
LOSS_GRADED=.15

# Fixed three-way split. The exact development model is also the exact deployment model.
ADAPT_SIZE=240 if V21_SMOKE else 2400
CAL_SIZE=100 if V21_SMOKE else 1000
LOCK_SIZE=60 if V21_SMOKE else 600
assert ADAPT_SIZE+CAL_SIZE+LOCK_SIZE==(400 if V21_SMOKE else 4000)

# V22: three predeclared surgical candidates. No module/alpha search is performed after labels are inspected.
V22_CANDIDATES={
    "m014035":{"modules":("M_014",),"alpha":.035},
    "m003_m014035":{"modules":("M_003","M_014"),"alpha":.035},
    "m003_m014050":{"modules":("M_003","M_014"),"alpha":.050},
}
V21_1_REFERENCE_MODULES=("M_003","M_009","M_014","M_016")
V21_1_REFERENCE_ALPHA=.035
CAL_FOLDS=2 if V21_SMOKE else 4
CAL_LGB_EST=45 if V21_SMOKE else 300

# Promotion gates. Primary stays exact Consensus3 when any gate fails.
MIN_CAL_DELTA=-.01 if V21_SMOKE else .00010
MIN_CAL_WINS=1 if V21_SMOKE else 3
MIN_CAL_SEEN_DELTA=-1 if V21_SMOKE else -.00015
MIN_CAL_NOVEL_DELTA=-1 if V21_SMOKE else -.00015
MIN_LOCK_DELTA=-.01 if V21_SMOKE else .00010
MIN_LOCK_CI_LOW=-1 if V21_SMOKE else -.00055
MIN_LOCK_SEEN_DELTA=-1 if V21_SMOKE else -.00020
MIN_LOCK_NOVEL_DELTA=-1 if V21_SMOKE else -.00020
MAX_TEST_SET_CHANGE=1 if V21_SMOKE else .030
MAX_TEST_ORDER_CHANGE=1 if V21_SMOKE else .080
MAX_TEST_TOP1_CHANGE=0.0


# V21.1 calibration stability controls.
CAL_BAG_SEEDS=(0,911,1823) if not V21_SMOKE else (0,)
SEM_AUC_MIN=-1.0 if V21_SMOKE else 0.575
MODULE_MIN_FOLD_WINS=1 if V21_SMOKE else 2
MODULE_MIN_DELTA=-1.0 if V21_SMOKE else 0.0
MIN_TRUSTED_MODULES=1 if V21_SMOKE else 2
MAX_TRUSTED_MODULES=17 if V21_SMOKE else 10
TRUST_PROBE_ALPHA=.035

ROOT=Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('/mnt/data')
CACHE_ROOT=ROOT/'v21_1_cache'; CACHE_ROOT.mkdir(parents=True,exist_ok=True)
random.seed(SEED); np.random.seed(SEED)
print("V21_SMOKE",V21_SMOKE,"skip_qwen",V21_SKIP_QWEN)
print("public Consensus3/V21.1/leader",PUBLIC_CONSENSUS3_LB,PUBLIC_V21_1_LB,PUBLIC_LEADER_LB,"gap from V21.1",round(PUBLIC_LEADER_LB-PUBLIC_V21_1_LB,6))

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

if V21_SMOKE:
    train=train.iloc[:400].copy(); test=test.iloc[:100].copy()
    sample=sample.set_index("user_id").loc[test.user_id].reset_index()
    keep=set(train.user_id)|set(test.user_id)
    chat=chat[chat.user_id.isin(keep)].copy(); assessments=assessments[assessments.user_id.isin(keep)].copy()

MODULES=list(train.columns[1:]); N_MODULES=len(MODULES); N_TRAIN=len(train); N_TEST=len(test)
assert MODULES==list(sample.columns[1:]) and N_MODULES==17
assert (N_TRAIN,N_TEST)==((400,100) if V21_SMOKE else (4000,1000))
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
word_vec=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_df=.995,sublinear_tf=True,strip_accents="unicode",max_features=(3000 if V21_SMOKE else 50000),dtype=np.float32)
char_vec=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=2,sublinear_tf=True,max_features=(5000 if V21_SMOKE else 60000),dtype=np.float32)
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
# V21.1 treats the previous V21 run as development evidence only. A new split is created with a new seed:
#
# 1. **ADAPT (2400)** — labels may update Qwen through QLoRA;
# 2. **CAL (1000)** — Qwen never trains on these users; CAL trains and cross-validates the semantic relevance calibrators and selects module trust;
# 3. **LOCK (600)** — neither Qwen nor calibrators see these labels before the single final gate.
#
# The extra CAL data directly targets V21's observed bottleneck: the semantic representation was stable CAL→LOCK, while 17 independent calibrators trained on only 700 users were noisy.

# ---- cell 9 (code) ----
# Stratify jointly by top relevance module and number of relevant modules.
pos_count=(y>0).sum(1).astype(int)
if V21_SMOKE:
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
ADAPT_FOLDS_N=2 if V21_SMOKE else 4
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
    return lgb.LGBMClassifier(n_estimators=(60 if V21_SMOKE else nest),learning_rate=.03,num_leaves=15,min_child_samples=50,reg_alpha=.5,reg_lambda=5.,colsample_bytree=.7,random_state=int(seed),n_jobs=4,verbosity=-1,force_col_wise=True)

# Historical full-fit pattern: train relevance calibrators on ADAPT labels using ADAPT OOF sparse features, predict held-out users from full-ADAPT sparse scores.
AD_META_CAL=np.zeros((len(CAL_IDX),N_MODULES),np.float32); AD_META_LOCK=np.zeros((len(LOCK_IDX),N_MODULES),np.float32); AD_META_TEST=np.zeros((N_TEST,N_MODULES),np.float32)
for m in range(N_MODULES):
    Xtr=np.column_stack([ad_base_oof[:,m],USER_FEATURES[ADAPT_IDX]]).astype(np.float32); yy=Y_BIN[ADAPT_IDX,m].astype(np.int32)
    mdl=make_adapt_meta(SPLIT_SEED+3000+m*31); mdl.fit(Xtr,yy)
    AD_META_CAL[:,m]=mdl.predict_proba(np.column_stack([AD_BASE_CAL[:,m],USER_FEATURES[CAL_IDX]]).astype(np.float32))[:,1]
    AD_META_LOCK[:,m]=mdl.predict_proba(np.column_stack([AD_BASE_LOCK[:,m],USER_FEATURES[LOCK_IDX]]).astype(np.float32))[:,1]
    AD_META_TEST[:,m]=mdl.predict_proba(np.column_stack([AD_BASE_TEST[:,m],USER_FEATURES[N_TRAIN:]]).astype(np.float32))[:,1]

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
    if V21_SMOKE or V21_SKIP_QWEN:
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
    if V21_SMOKE or V21_SKIP_QWEN:
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
    adapter_dir=ROOT/'v21_1_qwen_lora_adapter'; model.save_pretrained(adapter_dir); tok.save_pretrained(adapter_dir)
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
    if V21_SMOKE or V21_SKIP_QWEN:
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
# ## Bagged module-wise semantic relevance calibration
#
# V21.1 keeps module-specific calibrators because QLoRA usefulness is highly module-dependent, but reduces their variance:
#
# - CAL is enlarged to 1000 users;
# - each fold/module prediction averages 3 strongly regularized LightGBM seeds;
# - model capacity is reduced relative to V21;
# - semantic usefulness is measured **before LOCK** by 1024D relevance AUC and by realized single-module NDCG contribution across CAL folds;
# - only CAL-stable modules may receive a semantic residual at deployment.
#
# This preserves the proven relevance-calibration family while preventing weak semantic modules from perturbing Consensus3.

# ---- cell 20 (code) ----
def rank_matrix(scores):
    order=np.argsort(-scores,axis=1); ranks=np.empty_like(order); ranks[np.arange(len(order))[:,None],order]=np.arange(1,N_MODULES+1)[None,:]; return ranks.astype(np.float32)
def kth_score(scores,k): return np.sort(scores,axis=1)[:,-k]
def row_z_np(x):
    x=np.asarray(x,np.float32); return ((x-x.mean(1,keepdims=True))/np.maximum(x.std(1,keepdims=True),1e-6)).astype(np.float32)

def calibration_features(global_idx,base,meta,anchor,sem_list):
    global_idx=np.asarray(global_idx,int); s512,s1024,s4096=[np.asarray(s,np.float32) for s in sem_list]
    ar=rank_matrix(anchor); sr=rank_matrix(s1024); a5=kth_score(anchor,5); a6=kth_score(anchor,6); s5=kth_score(s1024,5)
    z1024=row_z_np(s1024)
    per_module=[]
    user_small=np.column_stack([A[global_idx],A_INVALID[global_idx],USER_FEATURES[global_idx,20:25]]).astype(np.float32)
    for m in range(N_MODULES):
        fm=np.column_stack([
            base[:,m],meta[:,m],anchor[:,m],
            ar[:,m]/17.0,anchor[:,m]-a5,anchor[:,m]-a6,(a5-a6),
            s512[:,m],s1024[:,m],s4096[:,m],z1024[:,m],sr[:,m]/17.0,s1024[:,m]-s5,
            (sr[:,m]-ar[:,m])/17.0,
            user_small
        ]).astype(np.float32)
        per_module.append(fm)
    return per_module

CAL_FEATS=calibration_features(CAL_IDX,AD_BASE_CAL,AD_META_CAL,AD_ANCHOR_CAL,CAL_SEM)
LOCK_FEATS=calibration_features(LOCK_IDX,AD_BASE_LOCK,AD_META_LOCK,AD_ANCHOR_LOCK,LOCK_SEM)
TEST_FEATS=calibration_features(TEST_GLOBAL,AD_BASE_TEST,AD_META_TEST,AD_ANCHOR_TEST,TEST_SEM)

def make_sem_cal(seed):
    return lgb.LGBMClassifier(
        n_estimators=CAL_LGB_EST,learning_rate=.02,num_leaves=7,min_child_samples=(25 if V21_SMOKE else 75),
        reg_alpha=1.5,reg_lambda=10.0,colsample_bytree=.80,subsample=.90,subsample_freq=1,
        random_state=int(seed),n_jobs=4,verbosity=-1,force_col_wise=True
    )

cal_strata=np.argmax(y[CAL_IDX],axis=1).astype(str)
cal_folds=(list(KFold(CAL_FOLDS,shuffle=True,random_state=CAL_SEED).split(np.zeros(len(CAL_IDX)))) if V21_SMOKE else list(StratifiedKFold(CAL_FOLDS,shuffle=True,random_state=CAL_SEED).split(np.zeros(len(CAL_IDX)),cal_strata)))
cal_fold_id=np.full(len(CAL_IDX),-1,int)
cal_prob_oof=np.zeros((len(CAL_IDX),N_MODULES),np.float32)

# Bagged OOF calibration: same fold boundaries, independent seeds, averaged probability.
for f,(tr,va) in enumerate(cal_folds):
    cal_fold_id[va]=f
    for m in range(N_MODULES):
        yy=Y_BIN[CAL_IDX,m].astype(np.int32); pred=np.zeros(len(va),np.float32)
        for b,off in enumerate(CAL_BAG_SEEDS):
            mdl=make_sem_cal(CAL_SEED+f*1009+m*37+off); mdl.fit(CAL_FEATS[m][tr],yy[tr])
            pred+=mdl.predict_proba(CAL_FEATS[m][va])[:,1].astype(np.float32)/len(CAL_BAG_SEEDS)
        cal_prob_oof[va,m]=pred
assert (cal_fold_id>=0).all()

# Final CAL-only bagged calibrators. LOCK remains unopened.
lock_prob=np.zeros((len(LOCK_IDX),N_MODULES),np.float32); test_prob=np.zeros((N_TEST,N_MODULES),np.float32)
for m in range(N_MODULES):
    yy=Y_BIN[CAL_IDX,m].astype(np.int32)
    for off in CAL_BAG_SEEDS:
        mdl=make_sem_cal(CAL_SEED+9000+m*41+off); mdl.fit(CAL_FEATS[m],yy)
        lock_prob[:,m]+=mdl.predict_proba(LOCK_FEATS[m])[:,1].astype(np.float32)/len(CAL_BAG_SEEDS)
        test_prob[:,m]+=mdl.predict_proba(TEST_FEATS[m])[:,1].astype(np.float32)/len(CAL_BAG_SEEDS)

# Semantic branch is hard-off where there is no learning chat.
cal_prob_oof[~HAS_SEMANTIC[CAL_IDX]]=AD_META_CAL[~HAS_SEMANTIC[CAL_IDX]]
lock_prob[~HAS_SEMANTIC[LOCK_IDX]]=AD_META_LOCK[~HAS_SEMANTIC[LOCK_IDX]]
test_prob[~HAS_SEMANTIC[TEST_GLOBAL]]=AD_META_TEST[~HAS_SEMANTIC[TEST_GLOBAL]]
print('bagged semantic-aware relevance calibrator ready')

# ---- cell 21 (markdown) ----
# ## Deployment-aligned surgical ablations
#
# V22 does **not** select trusted modules from noisy single-module probes. The masks are fixed from the prior public experiment before this notebook is run.
#
# The key correction versus V21/V21.1 is validation alignment: CAL and LOCK candidates are constructed by transporting the semantic residual onto the **full OOF Consensus3 base/meta path**, exactly mirroring test deployment. Every attack also uses the same hard Top-1 row lock.

# ---- cell 22 (code) ----
def subgroup_delta(ytrue,cand,anchor,mask):
    mask=np.asarray(mask,bool)
    if mask.sum()<10:return np.nan
    return float(ndcg5(ytrue[mask],cand[mask])-ndcg5(ytrue[mask],anchor[mask]))

def mask_from_modules(module_names):
    names=set(module_names)
    return np.asarray([m in names for m in MODULES],bool)

def hard_top1_row_lock(candidate,anchor):
    """If a correction changes Top-1, revert that user's complete row to the anchor."""
    candidate=np.asarray(candidate,np.float32).copy(); anchor=np.asarray(anchor,np.float32)
    bad=np.argmax(candidate,axis=1)!=np.argmax(anchor,axis=1)
    if bad.any(): candidate[bad]=anchor[bad]
    return candidate,bad

def build_deployment_aligned(base_full,meta_full,anchor_full,ad_meta,new_prob,has_sem,alpha,module_mask,top1_lock=True):
    """Exact test-time transport, evaluated on full-anchor OOF scores for CAL/LOCK."""
    mm=np.asarray(meta_full,np.float32).copy(); ad_meta=np.asarray(ad_meta,np.float32); new_prob=np.asarray(new_prob,np.float32)
    has_sem=np.asarray(has_sem,bool); module_mask=np.asarray(module_mask,bool)
    if has_sem.any() and module_mask.any():
        rows=np.flatnonzero(has_sem); cols=np.flatnonzero(module_mask)
        mm[np.ix_(rows,cols)]=meta_full[np.ix_(rows,cols)]+float(alpha)*(new_prob[np.ix_(rows,cols)]-ad_meta[np.ix_(rows,cols)])
    cand=apply_consensus(base_full,mm)
    reverted=np.zeros(len(cand),bool)
    if top1_lock: cand,reverted=hard_top1_row_lock(cand,anchor_full)
    return cand,reverted

def paired_boot_local(user_delta,seed,nboot=None):
    user_delta=np.asarray(user_delta,float); nboot=(250 if V21_SMOKE else 10000) if nboot is None else int(nboot)
    rng=np.random.default_rng(seed); n=len(user_delta); means=np.empty(nboot,float)
    for b in range(nboot): means[b]=user_delta[rng.integers(0,n,n)].mean()
    return float(np.quantile(means,.025)),float(np.quantile(means,.975))

# Recreate the CAL fold identities used to obtain cal_prob_oof. These folds are diagnostics only.
cal_strata=np.argmax(y[CAL_IDX],axis=1).astype(str)
cal_folds=(list(KFold(CAL_FOLDS,shuffle=True,random_state=CAL_SEED).split(np.zeros(len(CAL_IDX)))) if V21_SMOKE else list(StratifiedKFold(CAL_FOLDS,shuffle=True,random_state=CAL_SEED).split(np.zeros(len(CAL_IDX)),cal_strata)))
cal_fold_id=np.full(len(CAL_IDX),-1,int)
for f,(_,va) in enumerate(cal_folds): cal_fold_id[va]=f
assert (cal_fold_id>=0).all()

FULL_BASE_CAL=base_oof[CAL_IDX]; FULL_META_CAL=meta_oof[CAL_IDX]; FULL_ANCHOR_CAL=consensus3_oof[CAL_IDX]
FULL_BASE_LOCK=base_oof[LOCK_IDX]; FULL_META_LOCK=meta_oof[LOCK_IDX]; FULL_ANCHOR_LOCK=consensus3_oof[LOCK_IDX]

# Predeclared candidate diagnostics on the exact deployment decision surface.
v22_cal={}; v22_lock={}; v22_rows=[]
for ci,(name,spec) in enumerate(V22_CANDIDATES.items()):
    mm=mask_from_modules(spec['modules']); alpha=float(spec['alpha'])
    cc,cal_revert=build_deployment_aligned(FULL_BASE_CAL,FULL_META_CAL,FULL_ANCHOR_CAL,AD_META_CAL,cal_prob_oof,HAS_SEMANTIC[CAL_IDX],alpha,mm,True)
    lc,lock_revert=build_deployment_aligned(FULL_BASE_LOCK,FULL_META_LOCK,FULL_ANCHOR_LOCK,AD_META_LOCK,lock_prob,HAS_SEMANTIC[LOCK_IDX],alpha,mm,True)
    v22_cal[name]=cc; v22_lock[name]=lc
    cal_ud=per_user_ndcg(y[CAL_IDX],cc)-per_user_ndcg(y[CAL_IDX],FULL_ANCHOR_CAL)
    lock_ud=per_user_ndcg(y[LOCK_IDX],lc)-per_user_ndcg(y[LOCK_IDX],FULL_ANCHOR_LOCK)
    cal_delta=float(cal_ud.mean()); lock_delta=float(lock_ud.mean())
    fold_d=[]
    for f in range(CAL_FOLDS):
        vv=cal_fold_id==f; fold_d.append(float((per_user_ndcg(y[CAL_IDX][vv],cc[vv])-per_user_ndcg(y[CAL_IDX][vv],FULL_ANCHOR_CAL[vv])).mean()))
    cal_ci=paired_boot_local(cal_ud,BOOT_SEED+100+ci); lock_ci=paired_boot_local(lock_ud,BOOT_SEED+200+ci)
    v22_rows.append(dict(candidate=name,modules='+'.join(spec['modules']),alpha=alpha,
        cal_delta=cal_delta,cal_fold_wins=int(np.sum(np.asarray(fold_d)>0)),cal_fold_std=float(np.std(fold_d)),cal_fold_deltas=fold_d,
        cal_ci_low=cal_ci[0],cal_ci_high=cal_ci[1],lock_delta=lock_delta,lock_ci_low=lock_ci[0],lock_ci_high=lock_ci[1],
        cal_top1_reverted=int(cal_revert.sum()),lock_top1_reverted=int(lock_revert.sum())))
    print(name,'FULL-anchor CAL',round(cal_delta,6),'wins',int(np.sum(np.asarray(fold_d)>0)),'/',CAL_FOLDS,
          'LOCK',round(lock_delta,6),'LOCK CI',tuple(round(x,6) for x in lock_ci),'top1 reverts',int(cal_revert.sum()),int(lock_revert.sum()))

# Public V21.1-style four-module 3.5% reference, now also diagnosed on the full anchor.
ref_mask=mask_from_modules(V21_1_REFERENCE_MODULES)
V21_REF_CAL,_=build_deployment_aligned(FULL_BASE_CAL,FULL_META_CAL,FULL_ANCHOR_CAL,AD_META_CAL,cal_prob_oof,HAS_SEMANTIC[CAL_IDX],V21_1_REFERENCE_ALPHA,ref_mask,False)
V21_REF_LOCK,_=build_deployment_aligned(FULL_BASE_LOCK,FULL_META_LOCK,FULL_ANCHOR_LOCK,AD_META_LOCK,lock_prob,HAS_SEMANTIC[LOCK_IDX],V21_1_REFERENCE_ALPHA,ref_mask,False)
print('V21.1-style reference FULL-anchor CAL/LOCK',round(ndcg5(y[CAL_IDX],V21_REF_CAL)-ndcg5(y[CAL_IDX],FULL_ANCHOR_CAL),6),round(ndcg5(y[LOCK_IDX],V21_REF_LOCK)-ndcg5(y[LOCK_IDX],FULL_ANCHOR_LOCK),6))

# ---- cell 23 (markdown) ----
# ## Exact test deployment and fail-closed recommendation
#
# The same transport equation used in deployment-aligned CAL/LOCK diagnostics is now applied to the full test Consensus3 path. All three attacks are emitted regardless of gates so the three daily slots can be used deliberately; `primary` remains fail-closed.

# ---- cell 24 (code) ----
# Exact V21.1 public-family reference on full test scores (four-module stable035, no extra V22 Top-1 lock).
ref_mask=mask_from_modules(V21_1_REFERENCE_MODULES)
_v21_ref_meta=meta_test.copy(); _rows=np.flatnonzero(HAS_SEMANTIC[TEST_GLOBAL]); _cols=np.flatnonzero(ref_mask)
if len(_rows) and len(_cols):
    _v21_ref_meta[np.ix_(_rows,_cols)]=meta_test[np.ix_(_rows,_cols)]+V21_1_REFERENCE_ALPHA*(test_prob[np.ix_(_rows,_cols)]-AD_META_TEST[np.ix_(_rows,_cols)])
v21_1_reference_test=apply_consensus(base_test,_v21_ref_meta)

final_attacks={}; test_reverted={}
for name,spec in V22_CANDIDATES.items():
    mm=mask_from_modules(spec['modules']); alpha=float(spec['alpha'])
    cand,reverted=build_deployment_aligned(base_test,meta_test,consensus3_test,AD_META_TEST,test_prob,HAS_SEMANTIC[TEST_GLOBAL],alpha,mm,True)
    final_attacks[name]=cand; test_reverted[name]=reverted
    print(name,'test set',round(top5_set_change_rate(cand,consensus3_test),4),'order',round(top5_order_change_rate(cand,consensus3_test),4),
          'top1',round(top1_change_rate(cand,consensus3_test),4),'top1 rows reverted',int(reverted.sum()))

# Attach deployment perturbation + gates to the predeclared validation table.
for r in v22_rows:
    cand=final_attacks[r['candidate']]
    r['test_top5_set_change']=top5_set_change_rate(cand,consensus3_test)
    r['test_order_change']=top5_order_change_rate(cand,consensus3_test)
    r['test_top1_change']=top1_change_rate(cand,consensus3_test)
    r['test_top1_reverted']=int(test_reverted[r['candidate']].sum())
    r['stability_score']=float(min(r['cal_delta'],r['lock_delta'])-.35*r['cal_fold_std'])
    r['gate']=bool(
        r['cal_delta']>=MIN_CAL_DELTA and r['cal_fold_wins']>=MIN_CAL_WINS and
        r['lock_delta']>=MIN_LOCK_DELTA and r['lock_ci_low']>=MIN_LOCK_CI_LOW and
        r['test_top5_set_change']<=MAX_TEST_SET_CHANGE and r['test_order_change']<=MAX_TEST_ORDER_CHANGE and
        r['test_top1_change']<=MAX_TEST_TOP1_CHANGE+1e-12
    )

eligible=[r for r in v22_rows if r['gate']]
if eligible:
    SELECTED_ROW=max(eligible,key=lambda r:(r['stability_score'],min(r['cal_delta'],r['lock_delta']),-r['test_order_change']))
    primary_label=SELECTED_ROW['candidate']; primary_test=final_attacks[primary_label].copy()
else:
    SELECTED_ROW=None; primary_label='v21_1_reference_fallback'; primary_test=v21_1_reference_test.copy()
print('V22 PRIMARY',primary_label,'eligible',[r['candidate'] for r in eligible])

# ---- cell 25 (markdown) ----
# ## Diagnostics, three attacks, and reproducibility artifacts
#
# The three `attack` files are the intended daily leaderboard probes. Use the final table to decide ordering; `primary` is a fail-closed recommendation, not an instruction to spend a slot blindly.

# ---- cell 26 (code) ----
diag=pd.DataFrame(v22_rows)
show_cols=['candidate','modules','alpha','cal_delta','cal_fold_wins','cal_fold_std','cal_ci_low','cal_ci_high','lock_delta','lock_ci_low','lock_ci_high','test_top5_set_change','test_order_change','test_top1_change','test_top1_reverted','stability_score','gate']
display(diag[show_cols].sort_values(['gate','stability_score'],ascending=[False,False]))
print('================ V22 DECISION ================')
print('public Consensus3 / V21.1 / leader:',PUBLIC_CONSENSUS3_LB,PUBLIC_V21_1_LB,PUBLIC_LEADER_LB)
print('primary:',primary_label)
if SELECTED_ROW is not None:
    print('selected diagnostics:',{k:SELECTED_ROW[k] for k in ['candidate','cal_delta','cal_fold_wins','lock_delta','lock_ci_low','test_top5_set_change','test_order_change','stability_score']})
else:
    print('No new V22 candidate passed deployment-aligned CAL+LOCK gates; primary falls back to V21.1-family reference.')
print('Three attack files remain available as predeclared leaderboard experiments.')

# ---- cell 27 (code) ----
def save_submission(scores,name):
    out=sample.copy(); out.loc[:,MODULES]=np.asarray(scores,np.float32)
    path=ROOT/name; out.to_csv(path,index=False); print('saved',path); return str(path)

paths={}
paths['attack1']=save_submission(final_attacks['m014035'],'submission_v22_attack1_m014035.csv')
paths['attack2']=save_submission(final_attacks['m003_m014035'],'submission_v22_attack2_m003_m014035.csv')
paths['attack3']=save_submission(final_attacks['m003_m014050'],'submission_v22_attack3_m003_m014050.csv')
paths['primary']=save_submission(primary_test,'submission_v22_primary.csv')
paths['v21_1_reference']=save_submission(v21_1_reference_test,'submission_v22_v21_1_reference.csv')
paths['consensus3']=save_submission(consensus3_test,'submission_v22_consensus3_reference.csv')
diag.to_csv(ROOT/'v22_diagnostics.csv',index=False)

# Change audit for post-LB diagnosis.
audit_rows=[]
for name,cand in final_attacks.items():
    ao=np.argsort(-consensus3_test,axis=1)[:,:7]; co=np.argsort(-cand,axis=1)[:,:7]
    for i,u in enumerate(test.user_id.tolist()):
        aset=set(ao[i,:5]); cset=set(co[i,:5]); order_changed=not np.array_equal(ao[i,:5],co[i,:5])
        if order_changed:
            incoming=[MODULES[j] for j in cset-aset]; outgoing=[MODULES[j] for j in aset-cset]
            audit_rows.append({'candidate':name,'user_id':u,'anchor_top5':' '.join(MODULES[j] for j in ao[i,:5]),
                               'candidate_top5':' '.join(MODULES[j] for j in co[i,:5]),'incoming':' '.join(incoming),'outgoing':' '.join(outgoing),
                               'membership_changed':bool(aset!=cset),'top1_reverted':bool(test_reverted[name][i])})
pd.DataFrame(audit_rows).to_csv(ROOT/'v22_change_audit.csv',index=False)

summary={
    'public_consensus3_lb':PUBLIC_CONSENSUS3_LB,'public_v21_1_lb':PUBLIC_V21_1_LB,'public_leader_lb':PUBLIC_LEADER_LB,
    'candidate_specs':V22_CANDIDATES,'diagnostics':v22_rows,'primary_label':primary_label,'paths':paths,
    'validation_axis':'full Consensus3 OOF transport; exact same equation as test deployment',
    'top1_policy':'hard row revert to Consensus3 whenever candidate would change Top-1'
}
with open(ROOT/'v22_summary.json','w') as f: json.dump(summary,f,indent=2,default=lambda x:float(x) if isinstance(x,(np.floating,np.integer)) else (bool(x) if isinstance(x,np.bool_) else str(x)))
np.savez_compressed(ROOT/'v22_oof_artifacts.npz',y=y,consensus3_oof=consensus3_oof,consensus3_test=consensus3_test,adapt_idx=ADAPT_IDX,cal_idx=CAL_IDX,lock_idx=LOCK_IDX,
                    cal_prob_oof=cal_prob_oof,lock_prob=lock_prob,**{f'cal_{k}':v for k,v in v22_cal.items()},**{f'lock_{k}':v for k,v in v22_lock.items()})
print('V22 artifacts complete')

# ---- cell 28 (code) ----
# Hard invariants: fail loudly on any drift from the V22 contract.
assert y.shape==(N_TRAIN,17) and len(MODULES)==17
assert np.isfinite(consensus3_oof).all() and np.isfinite(consensus3_test).all()
assert np.isfinite(cal_prob_oof).all() and np.isfinite(lock_prob).all() and np.isfinite(test_prob).all()
assert set(V22_CANDIDATES)=={'m014035','m003_m014035','m003_m014050'}
assert V22_CANDIDATES['m014035']['modules']==('M_014',)
assert V22_CANDIDATES['m003_m014035']['modules']==('M_003','M_014')
assert V22_CANDIDATES['m003_m014050']['modules']==('M_003','M_014')
assert abs(V22_CANDIDATES['m003_m014035']['alpha']-.035)<1e-12 and abs(V22_CANDIDATES['m003_m014050']['alpha']-.050)<1e-12
assert len(set(ADAPT_IDX)&set(CAL_IDX))==0 and len(set(ADAPT_IDX)&set(LOCK_IDX))==0 and len(set(CAL_IDX)&set(LOCK_IDX))==0
assert len(ADAPT_IDX)+len(CAL_IDX)+len(LOCK_IDX)==N_TRAIN
for name,cand in final_attacks.items():
    assert np.isfinite(cand).all()
    assert np.array_equal(np.argmax(cand,axis=1),np.argmax(consensus3_test,axis=1)),(name,'Top-1 lock violated')
    nochat=~HAS_SEMANTIC[TEST_GLOBAL]
    if nochat.any(): assert np.allclose(cand[nochat],consensus3_test[nochat],atol=1e-7),name
    # Only M003/M014 score columns may differ from Consensus3 under V22 attacks.
    diff=np.abs(cand-consensus3_test)>1e-7
    allowed=mask_from_modules(V22_CANDIDATES[name]['modules'])
    assert not diff[:,~allowed].any(),(name,'unexpected module score drift')
assert np.isfinite(primary_test).all() and np.isfinite(v21_1_reference_test).all()
if SELECTED_ROW is None: assert np.allclose(primary_test,v21_1_reference_test,atol=0,rtol=0)
if not V21_SMOKE:
    assert abs(ndcg5(y,consensus3_oof)-0.6643682395561207)<0.0012,ndcg5(y,consensus3_oof)
assert list(sample.columns[1:])==MODULES
print('V22 hard invariants: PASS')
