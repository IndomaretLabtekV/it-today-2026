# Converted from notebooks/v7.ipynb: markdown cells become comments, %/! magics neutralized.

# ---- cell 0 (markdown) ----
# # MineToday V7 — Champion-Constrained Surgical Reranker
#
# **Target:** improve NDCG@5 without abandoning the only recipe already confirmed by public leaderboard: **V3.1 conservative**.
#
# V7 is deliberately conservative. It does **not** replace the champion with another global ranker. It:
#
# 1. reproduces the V3.1 sparse + per-module relevance stack;
# 2. freezes the `65% sparse + 35% relevance` champion;
# 3. writes a clean `45% relevance` calibration probe as a separate leaderboard experiment;
# 4. adds a fold-bagged calibration inference probe to remove the V3.1 full-fit/meta inference mismatch;
# 5. learns only **local adjacent swaps at ranks 2–6** (rank #1 is protected);
# 6. permits **at most one swap per user** and enforces a hard change budget;
# 7. validates the correction with an outer 5-fold procedure where every supervised feature for the outer validation users is trained without their labels;
# 8. selects the swap threshold only inside each outer-training split;
# 9. reports paired user deltas, fold wins, bootstrap confidence interval, change rate, and an ID-tail stress test;
# 10. uses fold-bagged pairwise inference on test.
#
# The pairwise gate predicts the **actual NDCG delta of an adjacent swap**, not merely whether the lower candidate is relevant. This makes the correction directly metric-aware while staying anchored to V3.1.
#
# **No external data, no previous-model artifact, and no direct LLM prediction are used.** All outputs are reproduced from the organizer dataset in this notebook.

# ---- cell 1 (code) ----
import os, re, json, ast, html, hashlib, random, warnings, gc, time, sys, subprocess
from pathlib import Path
from collections import defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ["PYTHONHASHSEED"] = "42"

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")
try:
    import lightgbm as lgb
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "lightgbm>=4.5,<5"])
    import lightgbm as lgb

SEED = 42
V7_SMOKE = bool(int(os.getenv("V7_SMOKE", "0")))
N_FOLDS = 2 if V7_SMOKE else 5
N_INNER = 3
RIDGE_ALPHA = 3.0
CHAMPION_W = 0.35
CAL045_W = 0.45
MAX_CHANGE_RATE = 0.20
MIN_CHANGE_RATE = 0.02
MIN_GATE_DELTA = 0.0025
PROTECTED_MODULES = {"M_006", "M_009", "M_017"}
FAVORED_EVIDENCE_MODULES = {"M_001", "M_010", "M_015"}
PAIR_POSITIONS = (1, 2, 3, 4)  # rank2↔3, rank3↔4, rank4↔5, rank5↔6; top1 is frozen
PAIR_THRESHOLDS = [0.0, 0.001, 0.0025, 0.005, 0.0075, 0.010, 0.015, 0.020]

random.seed(SEED); np.random.seed(SEED)
print("numpy", np.__version__, "pandas", pd.__version__, "lightgbm", lgb.__version__)
print("SMOKE MODE:", V7_SMOKE)

# ---- cell 2 (code) ----
def locate_data_dir():
    required={"chat_history.csv","modules_catalog.csv","user_assessments.csv","train_relevance.csv","test.csv","sample_submission.csv"}
    roots=[Path("/kaggle/input"),Path("/mnt/data/comp_v3"),Path("/mnt/data/comp_full"),Path(".")]
    for root in roots:
        if not root.exists(): continue
        candidates=[root]
        if root.name=="input": candidates += [p for p in root.rglob("*") if p.is_dir()]
        for d in candidates:
            try: names={p.name for p in d.iterdir() if p.is_file()}
            except Exception: continue
            if required.issubset(names): return d
    raise FileNotFoundError("MineToday dataset directory not found")

DATA_DIR=locate_data_dir(); print("DATA_DIR =",DATA_DIR)
chat=pd.read_csv(DATA_DIR/"chat_history.csv")
modules=pd.read_csv(DATA_DIR/"modules_catalog.csv")
assessments=pd.read_csv(DATA_DIR/"user_assessments.csv")
train=pd.read_csv(DATA_DIR/"train_relevance.csv")
test=pd.read_csv(DATA_DIR/"test.csv")
sample=pd.read_csv(DATA_DIR/"sample_submission.csv")
MODULES=list(train.columns[1:]); N_MODULES=len(MODULES); N_TRAIN=len(train); N_TEST=len(test)
assert MODULES==list(sample.columns[1:]) and N_MODULES==17 and N_TRAIN==4000 and N_TEST==1000
all_users=train.user_id.tolist()+test.user_id.tolist(); uid2idx={u:i for i,u in enumerate(all_users)}
y=train[MODULES].to_numpy(np.float32)
print("shapes:",chat.shape,modules.shape,assessments.shape,train.shape,test.shape)
print("positive modules/user:",pd.Series((y>0).sum(1)).value_counts().sort_index().to_dict())

# ---- cell 3 (code) ----
DISCOUNT=1.0/np.log2(np.arange(2,7))

def per_user_ndcg(y_true,scores):
    idx=np.argpartition(-scores,4,axis=1)[:,:5]
    loc=np.take_along_axis(scores,idx,axis=1)
    idx=np.take_along_axis(idx,np.argsort(-loc,axis=1),axis=1)
    dcg=(np.take_along_axis(y_true,idx,axis=1)*DISCOUNT).sum(1)
    ideal=np.sort(y_true,axis=1)[:,-5:][:,::-1]
    idcg=(ideal*DISCOUNT).sum(1)
    return dcg/np.maximum(idcg,1e-12)

def ndcg5(y_true,scores): return float(per_user_ndcg(np.asarray(y_true),np.asarray(scores)).mean())

def make_ordinal_targets(rel):
    out=np.zeros_like(rel,np.float32)
    for i,row in enumerate(rel):
        pos=np.where(row>0)[0]; order=pos[np.argsort(-row[pos])]
        for r,j in enumerate(order): out[i,j]=len(order)-r
    return out

Y_BIN=(y>0).astype(np.float32)
Y_TOP=np.zeros_like(y,np.float32); Y_TOP[np.arange(N_TRAIN),np.argmax(y,axis=1)]=1
Y_ORD=make_ordinal_targets(y)
strata=np.asarray([MODULES[j] for j in np.argmax(y,axis=1)])
skf=StratifiedKFold(N_FOLDS,shuffle=True,random_state=SEED)
folds=list(skf.split(np.zeros(N_TRAIN),strata))
fold_id=np.full(N_TRAIN,-1,int)
for f,(_,va) in enumerate(folds): fold_id[va]=f
print("fold sizes:",[len(v) for _,v in folds])

# ---- cell 4 (code) ----
WORD_NUM={"nol":0,"satu":1,"dua":2,"tiga":3,"empat":4,"lima":5}
SKILL_NAMES=["python","sql","statistics","eda","ml","ml_eval","dl","genai","business","research"]
A=np.zeros((len(all_users),10),np.float32); A_INVALID=np.zeros_like(A)
for row in assessments.itertuples(index=False):
    try: d=json.loads(row.assessment_result)
    except Exception: d=ast.literal_eval(row.assessment_result)
    vals=[]; bads=[]
    for v in d.values():
        sv=str(v).strip().lower()
        if sv in WORD_NUM: val,bad=float(WORD_NUM[sv]),0
        else:
            try: val=float(v)
            except Exception: val=np.nan
            bad=int((not np.isfinite(val)) or val<0 or val>5)
            if bad: val=np.nan
        vals.append(val); bads.append(bad)
    vals=np.asarray(vals,float); med=float(np.nanmedian(vals)) if np.isfinite(vals).any() else 2.0
    vals=np.where(np.isfinite(vals),vals,med); i=uid2idx[row.user_id]
    A[i]=vals; A_INVALID[i]=bads
print("invalid assessment cells:",int(A_INVALID.sum()),"/",A_INVALID.size)

# ---- cell 5 (code) ----
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
"budget":r"budget|diskon|worth it|cekak|fomo|kompetitor",
"online":r"online|durasi|kapan buka|mulai minggu|jadwal",
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
def template_token(s): return "tpl_"+hashlib.md5(s.encode()).hexdigest()[:10]

chat["timestamp"]=pd.to_datetime(chat["timestamp"],errors="coerce")
chat["clean"]=chat.user_chat_text.fillna("").map(clean_text)
chat["is_noise"]=chat.user_chat_text.fillna("").map(lambda x:bool(NOISE_RX.search(str(x))))
chat["topics"]=chat.user_chat_text.fillna("").map(extract_topics)
chat["skeleton"]=chat.user_chat_text.fillna("").map(make_skeleton)
for name,rx in STYLE_RX.items(): chat["sty_"+name]=chat.user_chat_text.fillna("").map(lambda x,r=rx:int(bool(r.search(str(x)))))
chat=chat.sort_values(["user_id","timestamp","chat_id"]).reset_index(drop=True)
user_chat={u:g.copy() for u,g in chat.groupby("user_id",sort=False)}
print("noise rows:",int(chat.is_noise.sum()),"/",len(chat))

# ---- cell 6 (code) ----
USER_FEATURES=np.zeros((len(all_users),10+10+5+len(STYLES)*2+len(TOPICS)*5),np.float32)

def build_user(u):
    ui=uid2idx[u]; g=user_chat.get(u)
    if g is None: learn=pd.DataFrame(columns=chat.columns); g=learn
    else:
        learn=g[~g.is_noise].copy()
        if len(learn)==0: learn=g.copy()
    parts=[]; n=len(learn)
    for j,row in enumerate(learn.itertuples(index=False)):
        d=n-1-j; tag="current" if d==0 else ("recent" if d<=2 else "older")
        parts += [f"[{tag}] {row.clean}",template_token(row.skeleton)]
        for tp in row.topics:
            parts.append("topic_"+tp)
            if d==0: parts += ["latest_topic_"+tp]*3
            elif d<=2: parts += ["recent_topic_"+tp]*2
        for sty in STYLES:
            if getattr(row,"sty_"+sty): parts.append("style_"+sty)
    for j,skill in enumerate(SKILL_NAMES):
        v=int(round(float(A[ui,j]))); band="low" if v<=1 else ("mid" if v<=3 else "high")
        parts += [f"skill_{skill}_{v}",f"skill_{skill}_bin_{band}"]
    feats=list(A[ui])+list(A_INVALID[ui]); total=len(g); lc=len(learn); noise=int(g.is_noise.sum()) if total else 0
    span=0.0
    if total>=2 and g.timestamp.notna().sum()>=2: span=float((g.timestamp.max()-g.timestamp.min()).total_seconds()/86400)
    feats += [total,lc,noise,span,min(total,6)]
    for sty in STYLES: feats.append(float(learn["sty_"+sty].sum()) if lc else 0.0)
    for sty in STYLES: feats.append(float(learn.iloc[-1]["sty_"+sty]) if lc else 0.0)
    rows=list(learn.itertuples(index=False)); nf=len(rows)
    for tp in TOPICS:
        count=rec=last=latest=0.0
        for j,row in enumerate(rows):
            if tp in row.topics:
                count+=1; rec+=float(np.exp(-0.7*(nf-1-j))); last=max(last,(j+1)/max(nf,1))
        if rows and tp in rows[-1].topics: latest=1.0
        feats += [count,rec,latest,last,float(count>0)]
    USER_FEATURES[ui]=np.asarray(feats,np.float32)
    return " ".join(parts)

texts=np.asarray([build_user(u) for u in all_users],object)
print("USER_FEATURES",USER_FEATURES.shape,"example",texts[1][:500])

# ---- cell 7 (code) ----
t0=time.time()
word_vec=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_df=.995,sublinear_tf=True,strip_accents="unicode",max_features=(5_000 if V7_SMOKE else 50_000),dtype=np.float32)
char_vec=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=2,sublinear_tf=True,max_features=(8_000 if V7_SMOKE else 60_000),dtype=np.float32)
X=hstack([word_vec.fit_transform(texts),char_vec.fit_transform(texts)],format="csr")
print("X",X.shape,"nnz",X.nnz,"seconds",round(time.time()-t0,1))

# ---- cell 8 (code) ----
# ----- Exact V3.1-style frozen champion backbone + one extra ordinal disagreement head -----
head_targets={"raw":y,"top":Y_TOP,"ord":Y_ORD}
oof_heads={}; test_heads={}
for name,target in head_targets.items():
    oo=np.zeros_like(y,np.float32); te=np.zeros((N_TEST,N_MODULES),np.float32)
    for f,(tr,va) in enumerate(folds):
        model=Ridge(alpha=RIDGE_ALPHA,solver="lsqr",tol=2e-3,max_iter=1000)
        model.fit(X[tr],target[tr]); oo[va]=model.predict(X[va]).astype(np.float32)
        te += model.predict(X[N_TRAIN:]).astype(np.float32)/N_FOLDS
    oof_heads[name]=oo; test_heads[name]=te
    print(name,"OOF",round(ndcg5(y,oo),6))
base_oof=.90*oof_heads["raw"]+.10*oof_heads["top"]
base_test=.90*test_heads["raw"]+.10*test_heads["top"]
print("sparse backbone",round(ndcg5(y,base_oof),6))

# ---- cell 9 (code) ----
# ----- Per-module relevance calibration: full-fit test and fold-bagged test are both retained -----
meta_oof=np.zeros_like(y,np.float32)
meta_test_foldbag=np.zeros((N_TEST,N_MODULES),np.float32)
best_iters={mid:[] for mid in MODULES}
if V7_SMOKE:
    # Smoke mode validates the new V7 mechanics cheaply. The full run below uses the exact LightGBM calibration path.
    meta_oof=base_oof.copy(); meta_test_foldbag=base_test.copy(); meta_test=base_test.copy()
    best_iters={mid:[20]*N_FOLDS for mid in MODULES}
else:
    for m,mid in enumerate(MODULES):
        Xm=np.column_stack([base_oof[:,m],USER_FEATURES[:N_TRAIN]])
        Xte=np.column_stack([base_test[:,m],USER_FEATURES[N_TRAIN:]])
        target=Y_BIN[:,m].astype(np.int32)
        for f,(tr,va) in enumerate(folds):
            clf=lgb.LGBMClassifier(n_estimators=1200,learning_rate=.03,num_leaves=15,min_child_samples=50,
                reg_alpha=.5,reg_lambda=5.0,colsample_bytree=.70,random_state=SEED+f+31*m,n_jobs=-1,verbosity=-1)
            clf.fit(Xm[tr],target[tr],eval_set=[(Xm[va],target[va])],callbacks=[lgb.early_stopping(80,verbose=False)])
            it=int(clf.best_iteration_ or 1200); best_iters[mid].append(it)
            meta_oof[va,m]=clf.predict_proba(Xm[va],num_iteration=it)[:,1].astype(np.float32)
            meta_test_foldbag[:,m]+=clf.predict_proba(Xte,num_iteration=it)[:,1].astype(np.float32)/N_FOLDS
    meta_test=np.zeros((N_TEST,N_MODULES),np.float32)
    for m,mid in enumerate(MODULES):
        Xtr=np.column_stack([base_oof[:,m],USER_FEATURES[:N_TRAIN]])
        Xte=np.column_stack([base_test[:,m],USER_FEATURES[N_TRAIN:]])
        it=max(20,int(np.median(best_iters[mid])))
        clf=lgb.LGBMClassifier(n_estimators=it,learning_rate=.03,num_leaves=15,min_child_samples=50,
            reg_alpha=.5,reg_lambda=5.0,colsample_bytree=.70,random_state=SEED+997+31*m,n_jobs=-1,verbosity=-1)
        clf.fit(Xtr,Y_BIN[:,m].astype(np.int32)); meta_test[:,m]=clf.predict_proba(Xte)[:,1].astype(np.float32)

champion_oof=(1-CHAMPION_W)*base_oof+CHAMPION_W*meta_oof
champion_test=(1-CHAMPION_W)*base_test+CHAMPION_W*meta_test
cal045_oof=(1-CAL045_W)*base_oof+CAL045_W*meta_oof
cal045_test=(1-CAL045_W)*base_test+CAL045_W*meta_test
foldbag035_test=(1-CHAMPION_W)*base_test+CHAMPION_W*meta_test_foldbag
foldbag045_test=(1-CAL045_W)*base_test+CAL045_W*meta_test_foldbag
print("meta only",round(ndcg5(y,meta_oof),6))
print("V3.1 champion w=.35",round(ndcg5(y,champion_oof),6))
print("calibration probe w=.45",round(ndcg5(y,cal045_oof),6),"delta",round(ndcg5(y,cal045_oof)-ndcg5(y,champion_oof),6))

# ---- cell 10 (code) ----
# ----- Pairwise feature engineering: only local ranks 2-6; top1 is immutable -----
MODULE_TOPIC={
"M_001":"excel","M_002":"python","M_003":"sql","M_004":"webscrape","M_005":"git","M_006":"statistics",
"M_007":"eda","M_008":"dashboard","M_009":"ml","M_010":"cv","M_011":"nlp","M_012":"genai","M_013":"prompt",
"M_014":"automation","M_015":"nocode","M_016":"mlops","M_017":"career"}
MODULE_SKILL={"M_002":0,"M_003":1,"M_004":0,"M_006":2,"M_007":3,"M_008":1,"M_009":4,"M_010":6,
              "M_011":4,"M_012":7,"M_013":7,"M_014":8,"M_015":7,"M_016":4,"M_017":9}
TOPIC_START=10+10+5+len(STYLES)*2
TOPIC_OFFSET={tp:TOPIC_START+5*i for i,tp in enumerate(TOPICS)}
MID2I={m:i for i,m in enumerate(MODULES)}

def swap_delta_for_pair(row, pos, a, b):
    ideal=np.sort(row)[-5:][::-1]; idcg=float(np.dot(ideal,DISCOUNT));
    if idcg<=1e-12: return 0.0
    if pos<4: return float((row[b]-row[a])*(DISCOUNT[pos]-DISCOUNT[pos+1])/idcg)
    return float((row[b]-row[a])*DISCOUNT[4]/idcg)

def make_pair_table(user_abs, champ, base, meta, raw, top, ords, labels=None):
    feats=[]; targets=[]; info=[]
    user_abs=np.asarray(user_abs,int)
    for li,u in enumerate(user_abs):
        order=np.argsort(-champ[li])[:6]
        for pos in PAIR_POSITIONS:
            a,b=int(order[pos]),int(order[pos+1]); ma,mb=MODULES[a],MODULES[b]
            f=[float(pos),float(pos==1),float(pos==2),float(pos==3),float(pos==4),
               float(ma in PROTECTED_MODULES),float(mb in PROTECTED_MODULES),
               float(ma in FAVORED_EVIDENCE_MODULES),float(mb in FAVORED_EVIDENCE_MODULES)]
            f += [float(k==a) for k in range(N_MODULES)] + [float(k==b) for k in range(N_MODULES)]
            for mat in (champ,base,meta,raw,top,ords):
                sa=float(mat[li,a]); sb=float(mat[li,b]); f += [sa,sb,sa-sb,abs(sa-sb)]
            uf=USER_FEATURES[u]; f += uf.tolist()
            for m in (ma,mb):
                tp=MODULE_TOPIC[m]; st=TOPIC_OFFSET[tp]; f += uf[st:st+5].tolist()
                sk=MODULE_SKILL.get(m); f.append(float(uf[sk]) if sk is not None else -1.0)
            sa=TOPIC_OFFSET[MODULE_TOPIC[ma]]; sb=TOPIC_OFFSET[MODULE_TOPIC[mb]]
            f += (uf[sa:sa+5]-uf[sb:sb+5]).tolist()
            ska=MODULE_SKILL.get(ma); skb=MODULE_SKILL.get(mb)
            va=float(uf[ska]) if ska is not None else -1.0; vb=float(uf[skb]) if skb is not None else -1.0
            f += [va-vb]
            feats.append(f); info.append((li,u,pos,a,b))
            if labels is not None: targets.append(swap_delta_for_pair(labels[li],pos,a,b))
    return np.asarray(feats,np.float32), (None if labels is None else np.asarray(targets,np.float32)), np.asarray(info,int)

def make_pair_regressor(seed):
    return lgb.LGBMRegressor(objective="huber",alpha=.80,n_estimators=320 if not V7_SMOKE else 20,
        learning_rate=.025 if not V7_SMOKE else .06,num_leaves=11,min_child_samples=100,
        reg_alpha=2.0,reg_lambda=12.0,colsample_bytree=.78,subsample=.90,
        random_state=seed,n_jobs=-1,verbosity=-1)

def apply_one_swap(scores, info, pred_gain, threshold):
    out=scores.copy(); changed=np.zeros(len(scores),bool); chosen=[]
    for li in range(len(scores)):
        ix=np.where(info[:,0]==li)[0]
        if not len(ix): continue
        j=ix[int(np.argmax(pred_gain[ix]))]
        if float(pred_gain[j])>float(threshold):
            _,u,pos,a,b=info[j]; out[li,a],out[li,b]=out[li,b],out[li,a]
            changed[li]=True; chosen.append((li,u,pos,a,b,float(pred_gain[j])))
    return out,changed,chosen

def choose_threshold(labels, champ, info, pred_gain, local_fold_id):
    best=(0.0,float("inf"),0.0,None)
    for thr in PAIR_THRESHOLDS+[float("inf")]:
        pred,ch,_=apply_one_swap(champ,info,pred_gain,thr)
        rate=float(ch.mean()); deltas=[]
        for f in np.unique(local_fold_id):
            ix=local_fold_id==f
            deltas.append(ndcg5(labels[ix],pred[ix])-ndcg5(labels[ix],champ[ix]))
        mean=float(np.mean(deltas)); std=float(np.std(deltas)); obj=mean-.25*std
        eligible=(rate<=MAX_CHANGE_RATE and (rate>=MIN_CHANGE_RATE or np.isinf(thr)))
        if eligible and obj>best[0]: best=(obj,thr,rate,deltas)
    return best

# ---- cell 11 (code) ----
# ----- Strict outer validation: reconstruct the champion without outer validation labels -----
def nested_champion(outer_tr, outer_va, seed):
    outer_tr=np.asarray(outer_tr,int); outer_va=np.asarray(outer_va,int)
    inner_s=strata[outer_tr]; n_inner=2 if V7_SMOKE else N_INNER
    iskf=StratifiedKFold(n_inner,shuffle=True,random_state=seed)
    inner=list(iskf.split(np.zeros(len(outer_tr)),inner_s)); inner_fid=np.full(len(outer_tr),-1,int)
    heads_tr={k:np.zeros((len(outer_tr),N_MODULES),np.float32) for k in ("raw","top","ord")}
    heads_va={k:np.zeros((len(outer_va),N_MODULES),np.float32) for k in ("raw","top","ord")}
    targets={"raw":y,"top":Y_TOP,"ord":Y_ORD}
    for k,target in targets.items():
        for f,(itr,iva) in enumerate(inner):
            abs_tr=outer_tr[itr]; abs_va=outer_tr[iva]; inner_fid[iva]=f
            model=Ridge(alpha=RIDGE_ALPHA,solver="lsqr",tol=2e-3,max_iter=1000)
            model.fit(X[abs_tr],target[abs_tr]); heads_tr[k][iva]=model.predict(X[abs_va]).astype(np.float32)
            heads_va[k]+=model.predict(X[outer_va]).astype(np.float32)/n_inner
    base_tr=.90*heads_tr["raw"]+.10*heads_tr["top"]; base_va=.90*heads_va["raw"]+.10*heads_va["top"]
    meta_tr=np.zeros_like(base_tr); meta_va=np.zeros_like(base_va)
    if V7_SMOKE:
        meta_tr=base_tr.copy(); meta_va=base_va.copy()
    else:
        for m,mid in enumerate(MODULES):
            Xm=np.column_stack([base_tr[:,m],USER_FEATURES[outer_tr]])
            Xv=np.column_stack([base_va[:,m],USER_FEATURES[outer_va]])
            target=Y_BIN[outer_tr,m].astype(np.int32)
            for f,(itr,iva) in enumerate(inner):
                clf=lgb.LGBMClassifier(n_estimators=500,learning_rate=.035,
                    num_leaves=15,min_child_samples=45,reg_alpha=.5,reg_lambda=5.0,colsample_bytree=.70,
                    random_state=seed+31*m+f,n_jobs=-1,verbosity=-1)
                clf.fit(Xm[itr],target[itr],eval_set=[(Xm[iva],target[iva])],callbacks=[lgb.early_stopping(45,verbose=False)])
                it=int(clf.best_iteration_ or clf.n_estimators)
                meta_tr[iva,m]=clf.predict_proba(Xm[iva],num_iteration=it)[:,1].astype(np.float32)
                meta_va[:,m]+=clf.predict_proba(Xv,num_iteration=it)[:,1].astype(np.float32)/n_inner
    champ_tr=(1-CHAMPION_W)*base_tr+CHAMPION_W*meta_tr; champ_va=(1-CHAMPION_W)*base_va+CHAMPION_W*meta_va
    return dict(inner_fid=inner_fid,raw_tr=heads_tr["raw"],top_tr=heads_tr["top"],ord_tr=heads_tr["ord"],
                raw_va=heads_va["raw"],top_va=heads_va["top"],ord_va=heads_va["ord"],
                base_tr=base_tr,base_va=base_va,meta_tr=meta_tr,meta_va=meta_va,champ_tr=champ_tr,champ_va=champ_va)

outer_champ=np.zeros_like(y,np.float32); outer_corr=np.zeros_like(y,np.float32)
outer_fold_id=np.full(N_TRAIN,-1,int); fold_rows=[]; selected_thresholds=[]
outer_to_run=folds[:1] if V7_SMOKE else folds
for of,(otr,ova) in enumerate(outer_to_run):
    print(f"\n========== OUTER {of+1}/{len(outer_to_run)} ==========")
    pack=nested_champion(otr,ova,SEED+1000+of)
    Xp,yd,inf=make_pair_table(otr,pack["champ_tr"],pack["base_tr"],pack["meta_tr"],pack["raw_tr"],pack["top_tr"],pack["ord_tr"],y[otr])
    pred_gain_tr=np.zeros(len(yd),np.float32)
    for f in np.unique(pack["inner_fid"]):
        user_fid=pack["inner_fid"][inf[:,0]]; trm=user_fid!=f; vam=user_fid==f
        reg=make_pair_regressor(SEED+2000+100*of+f); reg.fit(Xp[trm],yd[trm]); pred_gain_tr[vam]=reg.predict(Xp[vam]).astype(np.float32)
    best=choose_threshold(y[otr],pack["champ_tr"],inf,pred_gain_tr,pack["inner_fid"])
    _,thr,train_rate,inner_deltas=best; selected_thresholds.append(thr)
    reg=make_pair_regressor(SEED+3000+of); reg.fit(Xp,yd)
    Xv,_,vinf=make_pair_table(ova,pack["champ_va"],pack["base_va"],pack["meta_va"],pack["raw_va"],pack["top_va"],pack["ord_va"],None)
    pg=reg.predict(Xv).astype(np.float32); corr,changed,_=apply_one_swap(pack["champ_va"],vinf,pg,thr)
    outer_champ[ova]=pack["champ_va"]; outer_corr[ova]=corr; outer_fold_id[ova]=of
    b=ndcg5(y[ova],pack["champ_va"]); s=ndcg5(y[ova],corr); d=s-b
    fold_rows.append(dict(fold=of+1,champion=b,surgical=s,delta=d,threshold=thr,train_change=train_rate,val_change=float(changed.mean())))
    print("champ",round(b,6),"surgical",round(s,6),"delta",round(d,6),"thr",thr,"val_change",round(float(changed.mean()),4))
    del pack,Xp,yd,inf,pred_gain_tr,reg,Xv,vinf,pg,corr; gc.collect()

if V7_SMOKE:
    print("SMOKE MODE finished one outer fold. Full competition metrics intentionally skipped.")
else:
    fold_df=pd.DataFrame(fold_rows); display(fold_df.round(6))
    print("strict nested champion",round(ndcg5(y,outer_champ),6))
    print("strict nested surgical",round(ndcg5(y,outer_corr),6),"delta",round(ndcg5(y,outer_corr)-ndcg5(y,outer_champ),6))

# ---- cell 12 (code) ----
# ----- Paired validation gate: evidence before submission -----
if not V7_SMOKE:
    pu_base=per_user_ndcg(y,outer_champ); pu_new=per_user_ndcg(y,outer_corr); delta_u=pu_new-pu_base
    rng=np.random.default_rng(SEED+77); boots=[]
    for _ in range(2500):
        ix=rng.integers(0,N_TRAIN,N_TRAIN); boots.append(float(delta_u[ix].mean()))
    lo,hi=np.quantile(boots,[.025,.975]); mean_delta=float(delta_u.mean())
    fold_deltas=fold_df.delta.to_numpy(float); fold_wins=int((fold_deltas>0).sum())
    changed_rate=float(np.mean(np.argsort(-outer_corr,axis=1)[:,:5].tolist()!=np.argsort(-outer_champ,axis=1)[:,:5].tolist())) if False else float(np.mean(np.any(np.argsort(-outer_corr,axis=1)[:,:5]!=np.argsort(-outer_champ,axis=1)[:,:5],axis=1)))
    tail=np.arange(3200,4000); tail_delta=ndcg5(y[tail],outer_corr[tail])-ndcg5(y[tail],outer_champ[tail])
    gate=(mean_delta>=MIN_GATE_DELTA and fold_wins>=4 and lo>0 and changed_rate<=MAX_CHANGE_RATE and tail_delta>-0.001)
    print("\nPAIRWISE EVIDENCE")
    print("paired mean delta      :",round(mean_delta,6))
    print("bootstrap 95% CI       :",(round(float(lo),6),round(float(hi),6)))
    print("outer fold wins        :",f"{fold_wins}/5",[round(x,6) for x in fold_deltas])
    print("changed users          :",round(changed_rate,4))
    print("U_3201..U_4000 stress  :",round(tail_delta,6))
    print("selected thresholds    :",selected_thresholds)
    print("PAIRWISE GATE           :", "PASS" if gate else "REVIEW / DO NOT AUTO-SUBMIT")

# ---- cell 13 (code) ----
# ----- Final fold-bagged surgical inference on the proven V3.1 champion -----
if not V7_SMOKE:
    train_pair_X,train_pair_y,train_pair_info=make_pair_table(np.arange(N_TRAIN),champion_oof,base_oof,meta_oof,
        oof_heads["raw"],oof_heads["top"],oof_heads["ord"],y)
    test_pair_X,_,test_pair_info=make_pair_table(np.arange(N_TRAIN,N_TRAIN+N_TEST),champion_test,base_test,meta_test,
        test_heads["raw"],test_heads["top"],test_heads["ord"],None)
    test_gain=np.zeros(len(test_pair_info),np.float32)
    for f,(tr,va) in enumerate(folds):
        trmask=fold_id[train_pair_info[:,1]]!=f
        reg=make_pair_regressor(SEED+5000+f); reg.fit(train_pair_X[trmask],train_pair_y[trmask])
        test_gain += reg.predict(test_pair_X).astype(np.float32)/N_FOLDS
    finite_thr=[x for x in selected_thresholds if np.isfinite(x)]
    final_thr=float(np.median(finite_thr)) if finite_thr else float("inf")
    pairwise_test,pair_changed,pair_choices=apply_one_swap(champion_test,test_pair_info,test_gain,final_thr)
    print("final pair threshold:",final_thr,"test changed:",round(float(pair_changed.mean()),4),"n=",int(pair_changed.sum()))
    assert float(pair_changed.mean())<=MAX_CHANGE_RATE+0.03, "Pairwise test change budget unexpectedly high"
    # Top-1 must be unchanged by construction.
    assert np.array_equal(np.argmax(pairwise_test,axis=1),np.argmax(champion_test,axis=1))

# ---- cell 14 (code) ----
# ----- Submission writer + experiment order -----
def write_submission(scores,name):
    assert scores.shape==(N_TEST,N_MODULES) and np.isfinite(scores).all()
    out=sample.copy(); assert out.user_id.tolist()==test.user_id.tolist(); out.loc[:,MODULES]=scores
    root=Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("/mnt/data")
    path=root/name; out.to_csv(path,index=False)
    chk=pd.read_csv(path); assert chk.shape==sample.shape and chk.user_id.is_unique and chk[MODULES].notna().all().all()
    print("saved",path); return path

if not V7_SMOKE:
    p_champ=write_submission(champion_test,"submission_v7_champion_reference.csv")
    p_cal45=write_submission(cal045_test,"submission_v7_cal045_probe.csv")
    p_fb35=write_submission(foldbag035_test,"submission_v7_foldbag_cal035.csv")
    p_fb45=write_submission(foldbag045_test,"submission_v7_foldbag_cal045.csv")
    p_pair=write_submission(pairwise_test,"submission_v7_pairwise.csv")
    primary_scores=pairwise_test if gate else champion_test
    p_primary=write_submission(primary_scores,"submission_v7_primary.csv")
    np.savez_compressed((Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("/mnt/data"))/"v7_oof_artifacts.npz",
        y=y,base_oof=base_oof,meta_oof=meta_oof,champion_oof=champion_oof,cal045_oof=cal045_oof,
        outer_champ=outer_champ,outer_corr=outer_corr,champion_test=champion_test,cal045_test=cal045_test,
        foldbag035_test=foldbag035_test,pairwise_test=pairwise_test,outer_fold_id=outer_fold_id)
    print("\nFINAL SUMMARY")
    print("V3.1 champion OOF      :",round(ndcg5(y,champion_oof),6))
    print("Cal .45 probe OOF      :",round(ndcg5(y,cal045_oof),6))
    print("Strict pair baseline   :",round(ndcg5(y,outer_champ),6))
    print("Strict pair surgical   :",round(ndcg5(y,outer_corr),6))
    print("Pairwise gate           :", "PASS" if gate else "REVIEW")
    print("\nTomorrow submission logic:")
    print("1) CLEAN PROBE: submission_v7_cal045_probe.csv")
    print("2) PAIRWISE only if gate PASS: submission_v7_pairwise.csv")
    print("3) FOLDBAG probe if needed: submission_v7_foldbag_cal035.csv")
    print("Auto-primary file:",p_primary.name)

# ---- cell 15 (code) ----
# ----- Hard invariants -----
assert len(set(MODULES))==17
assert np.isfinite(USER_FEATURES).all()
assert np.isfinite(base_oof).all() and np.isfinite(meta_oof).all()
assert abs(ndcg5(y,champion_oof)-ndcg5(y,champion_oof.copy()))<1e-12
if not V7_SMOKE:
    assert np.all(outer_fold_id>=0)
    assert np.isfinite(outer_champ).all() and np.isfinite(outer_corr).all()
    assert np.array_equal(np.argmax(pairwise_test,axis=1),np.argmax(champion_test,axis=1))
print("All hard invariants passed.")
