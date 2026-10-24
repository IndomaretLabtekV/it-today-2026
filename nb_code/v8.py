# Converted from notebooks/v8.ipynb: markdown cells become comments, %/! magics neutralized.

# ---- cell 0 (markdown) ----
# # MineToday V8 — Repeated-CV Relevance Calibration Bagging
#
# **Goal:** preserve the public-validated V3.1 sparse backbone and improve only the relevance calibration layer by reducing its fold/seed variance.
#
# This notebook intentionally avoids new global rankers, pairwise swaps, transformers, manual rare-module boosts, and cross-version ensembles. The only substantive change from the proven V3.1 mechanism is **5-seed repeated 5-fold per-module LightGBM calibration**, followed by seed averaging.
#
# Primary candidate: **repeated calibration weight 0.60**, only if the pre-registered stability gate passes.  
# Reference public champion entering V8: **V7/V3.1 calibration 0.45 = 0.65639 public LB**.

# ---- cell 1 (code) ----
import os, re, json, ast, html, hashlib, random, warnings, gc, time, sys, subprocess
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ["PYTHONHASHSEED"] = "42"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
V8_SMOKE = bool(int(os.getenv("V8_SMOKE", "0")))

N_FOLDS = 2 if V8_SMOKE else 5
RIDGE_ALPHA = 3.0
CAL_SEEDS = [42, 137] if V8_SMOKE else [42, 137, 2026, 7, 31415]
BLEND_WEIGHTS = [0.45, 0.50, 0.55, 0.60, 0.675]
REFERENCE_W = 0.45
PRIMARY_W = 0.60

LGB_ESTIMATORS = 80 if V8_SMOKE else 1200
LGB_EARLY_STOP = 10 if V8_SMOKE else 80

# Pre-registered evidence gate from the post-V7 analysis.
MIN_PRIMARY_DELTA = 0.0015
MIN_REFERENCE_FOLD_WINS = 4 if not V8_SMOKE else 1
MAX_TEST_TOP5_CHANGE = 0.30

random.seed(SEED)
np.random.seed(SEED)

print("numpy", np.__version__)
print("pandas", pd.__version__)
print("lightgbm", lgb.__version__)
print("SMOKE MODE:", V8_SMOKE)
print("calibration seeds:", CAL_SEEDS)

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
# Exact public-validated V3.1 sparse backbone: word+char TF-IDF, 90% raw + 10% top1 Ridge
t0 = time.time()
word_vec = TfidfVectorizer(
    ngram_range=(1,2), min_df=2, max_df=0.995, sublinear_tf=True,
    strip_accents="unicode", max_features=(8_000 if V8_SMOKE else 50_000), dtype=np.float32
)
char_vec = TfidfVectorizer(
    analyzer="char_wb", ngram_range=(3,5), min_df=2, sublinear_tf=True,
    max_features=(12_000 if V8_SMOKE else 60_000), dtype=np.float32
)

# Unsupervised train+test vocabulary fitting uses only organizer-provided competition data.
X = hstack([word_vec.fit_transform(texts), char_vec.fit_transform(texts)], format="csr")
print("X:", X.shape, "nnz:", X.nnz, "seconds:", round(time.time()-t0,1))

def fit_sparse_head(target):
    oof = np.zeros_like(y, np.float32)
    pred_test = np.zeros((N_TEST, N_MODULES), np.float32)
    for f,(tr,va) in enumerate(ref_folds):
        model = Ridge(alpha=RIDGE_ALPHA, solver="lsqr", tol=2e-3, max_iter=1000)
        model.fit(X[tr], target[tr])
        oof[va] = model.predict(X[va]).astype(np.float32)
        pred_test += model.predict(X[N_TRAIN:]).astype(np.float32) / N_FOLDS
    return oof, pred_test

raw_oof, raw_test = fit_sparse_head(y)
top_oof, top_test = fit_sparse_head(Y_TOP)
base_oof = (0.90*raw_oof + 0.10*top_oof).astype(np.float32)
base_test = (0.90*raw_test + 0.10*top_test).astype(np.float32)

print("raw OOF          :", round(ndcg5(y,raw_oof),6))
print("top1 OOF         :", round(ndcg5(y,top_oof),6))
print("sparse backbone  :", round(ndcg5(y,base_oof),6))

if not V8_SMOKE:
    assert abs(ndcg5(y, raw_oof) - 0.657337) < 0.002
    assert abs(ndcg5(y, base_oof) - 0.658254) < 0.002

# ---- cell 8 (code) ----
# Precompute the 17 module-specific calibration matrices once.
META_X_TRAIN = []
META_X_TEST = []
for m in range(N_MODULES):
    META_X_TRAIN.append(np.column_stack([base_oof[:,m], USER_FEATURES[:N_TRAIN]]).astype(np.float32))
    META_X_TEST.append(np.column_stack([base_test[:,m], USER_FEATURES[N_TRAIN:]]).astype(np.float32))

def make_calibrator(random_state, n_estimators=None):
    return lgb.LGBMClassifier(
        n_estimators=(LGB_ESTIMATORS if n_estimators is None else int(n_estimators)),
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=50,
        reg_alpha=0.5,
        reg_lambda=5.0,
        colsample_bytree=0.70,
        random_state=int(random_state),
        n_jobs=-1,
        verbosity=-1,
    )

def fit_one_calibration_seed(cal_seed):
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=int(cal_seed))
    folds = list(skf.split(np.zeros(N_TRAIN), strata))

    seed_oof = np.zeros_like(y, np.float32)
    seed_test_foldbag = np.zeros((N_TEST,N_MODULES), np.float32)
    seed_test_full = np.zeros((N_TEST,N_MODULES), np.float32)
    best_iters = {mid:[] for mid in MODULES}

    for m,mid in enumerate(MODULES):
        Xm = META_X_TRAIN[m]
        Xte = META_X_TEST[m]
        target = Y_BIN[:,m].astype(np.int32)

        for f,(tr,va) in enumerate(folds):
            clf = make_calibrator(cal_seed + f + 31*m)
            clf.fit(
                Xm[tr], target[tr],
                eval_set=[(Xm[va],target[va])],
                callbacks=[lgb.early_stopping(LGB_EARLY_STOP, verbose=False)],
            )
            it = int(clf.best_iteration_ or LGB_ESTIMATORS)
            best_iters[mid].append(it)
            seed_oof[va,m] = clf.predict_proba(Xm[va], num_iteration=it)[:,1].astype(np.float32)
            seed_test_foldbag[:,m] += clf.predict_proba(Xte, num_iteration=it)[:,1].astype(np.float32) / N_FOLDS

        # Preserve the public-winning V3.1 inference style: fit on all train using
        # the median early-stopped iteration count from this seed's CV.
        final_iters = max(20, int(np.median(best_iters[mid])))
        clf_full = make_calibrator(cal_seed + 997 + 31*m, final_iters)
        clf_full.fit(Xm, target)
        seed_test_full[:,m] = clf_full.predict_proba(Xte)[:,1].astype(np.float32)

    return seed_oof, seed_test_full, seed_test_foldbag, best_iters

print("calibration matrices:", META_X_TRAIN[0].shape, META_X_TEST[0].shape)

# ---- cell 9 (code) ----
# V8 core: repeated-CV calibration bagging
seed_meta_oof = []
seed_meta_test_full = []
seed_meta_test_foldbag = []
seed_best_iters = []

t0 = time.time()
for si,cal_seed in enumerate(CAL_SEEDS,1):
    print(f"\n===== calibration seed {si}/{len(CAL_SEEDS)}: {cal_seed} =====")
    oo,te_full,te_fb,best = fit_one_calibration_seed(cal_seed)
    seed_meta_oof.append(oo)
    seed_meta_test_full.append(te_full)
    seed_meta_test_foldbag.append(te_fb)
    seed_best_iters.append(best)
    print("meta-only OOF:", round(ndcg5(y,oo),6))
    print("seed w=.60  :", round(ndcg5(y,0.40*base_oof + 0.60*oo),6))

seed_meta_oof = np.stack(seed_meta_oof).astype(np.float32)
seed_meta_test_full = np.stack(seed_meta_test_full).astype(np.float32)
seed_meta_test_foldbag = np.stack(seed_meta_test_foldbag).astype(np.float32)

rep_meta_oof = seed_meta_oof.mean(axis=0).astype(np.float32)
rep_meta_test_full = seed_meta_test_full.mean(axis=0).astype(np.float32)
rep_meta_test_foldbag = seed_meta_test_foldbag.mean(axis=0).astype(np.float32)

print("\nRepeated calibration runtime seconds:", round(time.time()-t0,1))
print("Repeated meta-only OOF:", round(ndcg5(y,rep_meta_oof),6))

# ---- cell 10 (code) ----
# Reference calibration .45 must be seed-42, because that is the exact public-tested mechanism.
seed42_idx = CAL_SEEDS.index(42)
meta42_oof = seed_meta_oof[seed42_idx]
meta42_test_full = seed_meta_test_full[seed42_idx]

cal045_oof = ((1-REFERENCE_W)*base_oof + REFERENCE_W*meta42_oof).astype(np.float32)
cal045_test = ((1-REFERENCE_W)*base_test + REFERENCE_W*meta42_test_full).astype(np.float32)
reference_score = ndcg5(y,cal045_oof)

candidate_oof = {}
candidate_test = {}
candidate_test_foldbag = {}

print("\nREFERENCE cal045 OOF:", round(reference_score,6))
for w in BLEND_WEIGHTS:
    name = f"w{str(w).replace('.','p')}"
    candidate_oof[name] = ((1-w)*base_oof + w*rep_meta_oof).astype(np.float32)
    candidate_test[name] = ((1-w)*base_test + w*rep_meta_test_full).astype(np.float32)
    candidate_test_foldbag[name] = ((1-w)*base_test + w*rep_meta_test_foldbag).astype(np.float32)
    sc = ndcg5(y,candidate_oof[name])
    print(f"repeated calibration {w:0.3f}: {sc:.6f} delta_vs_cal045={sc-reference_score:+.6f}")

primary_name = "w0p6"
primary_oof = candidate_oof[primary_name]
primary_test = candidate_test[primary_name]
primary_foldbag_test = candidate_test_foldbag[primary_name]
primary_score = ndcg5(y,primary_oof)

# ---- cell 11 (code) ----
# Stability diagnostics: paired bootstrap, reference-fold wins, leave-one-seed-out.
def paired_bootstrap_ci(y_true, pred_new, pred_ref, n_boot=4000, seed=20260822):
    delta = per_user_ndcg(y_true,pred_new) - per_user_ndcg(y_true,pred_ref)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot,np.float64)
    n = len(delta)
    for b in range(n_boot):
        means[b] = delta[rng.integers(0,n,n)].mean()
    return float(delta.mean()), tuple(np.quantile(means,[0.025,0.975]))

paired_delta, boot_ci = paired_bootstrap_ci(y,primary_oof,cal045_oof)

ref_fold_deltas = []
for f in range(N_FOLDS):
    ix = ref_fold_id == f
    ref_fold_deltas.append(ndcg5(y[ix],primary_oof[ix]) - ndcg5(y[ix],cal045_oof[ix]))
fold_wins = sum(x > 0 for x in ref_fold_deltas)

loo_scores = []
loo_gains = []
for held in range(len(CAL_SEEDS)):
    keep = [j for j in range(len(CAL_SEEDS)) if j != held]
    loo_meta = seed_meta_oof[keep].mean(axis=0)
    loo_pred = (1-PRIMARY_W)*base_oof + PRIMARY_W*loo_meta
    sc = ndcg5(y,loo_pred)
    loo_scores.append(sc)
    loo_gains.append(sc-reference_score)

test_top5_change = top5_set_change_rate(primary_test,cal045_test)
test_top1_change = top1_change_rate(primary_test,cal045_test)

print("PRIMARY repeated w=.60 OOF :", round(primary_score,6))
print("paired mean delta           :", round(paired_delta,6))
print("bootstrap 95% CI            :", tuple(round(x,6) for x in boot_ci))
print("reference-fold deltas       :", [round(float(x),6) for x in ref_fold_deltas], f"wins={fold_wins}/{N_FOLDS}")
print("leave-one-seed-out gains    :", {CAL_SEEDS[i]:round(float(loo_gains[i]),6) for i in range(len(CAL_SEEDS))})
print("test Top5 set change vs .45 :", round(test_top5_change,4))
print("test top1 change vs .45     :", round(test_top1_change,4))

gate = (
    (primary_score - reference_score >= MIN_PRIMARY_DELTA)
    and (boot_ci[0] > 0)
    and (fold_wins >= MIN_REFERENCE_FOLD_WINS)
    and (min(loo_gains) > 0 if len(loo_gains) >= 3 else True)
    and (test_top5_change <= MAX_TEST_TOP5_CHANGE)
)

if V8_SMOKE:
    print("SMOKE MODE: evidence gate is diagnostic only.")
else:
    print("V8 STABILITY GATE:", "PASS" if gate else "REVIEW / DO NOT AUTO-SUBMIT")

# ---- cell 12 (code) ----
# Metric-focused diagnostics.
def top1_accuracy(scores):
    return float(np.mean(np.argmax(scores,axis=1) == np.argmax(y,axis=1)))

def true_top1_in5(scores):
    t5 = top5_indices(scores)
    truth = np.argmax(y,axis=1)
    return float(np.mean([truth[i] in t5[i] for i in range(N_TRAIN)]))

def avg_relevant_in5(scores):
    t5 = top5_indices(scores)
    return float(np.mean([(y[i,t5[i]] > 0).sum() for i in range(N_TRAIN)]))

print("cal045 top1 accuracy         :", round(top1_accuracy(cal045_oof),4))
print("V8 w60 top1 accuracy         :", round(top1_accuracy(primary_oof),4))
print("cal045 true-top1 in Top5     :", round(true_top1_in5(cal045_oof),4))
print("V8 w60 true-top1 in Top5     :", round(true_top1_in5(primary_oof),4))
print("cal045 avg relevant in Top5  :", round(avg_relevant_in5(cal045_oof),4))
print("V8 w60 avg relevant in Top5  :", round(avg_relevant_in5(primary_oof),4))

for label,scores in [("cal045",cal045_test),("v8_w60",primary_test)]:
    t5 = top5_indices(scores)
    freq = np.bincount(t5.ravel(), minlength=N_MODULES)
    print(label, "test Top5 frequency:", {MODULES[m]:int(freq[m]) for m in range(N_MODULES)})

# ---- cell 13 (code) ----
# Submission writer + reproducible artifacts.
def write_submission(scores, filename):
    assert scores.shape == (N_TEST,N_MODULES)
    assert np.isfinite(scores).all()
    out = sample.copy()
    assert out["user_id"].tolist() == test["user_id"].tolist()
    out.loc[:,MODULES] = scores

    root = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("/mnt/data")
    path = root / filename
    out.to_csv(path,index=False)

    chk = pd.read_csv(path)
    assert chk.shape == sample.shape
    assert chk["user_id"].is_unique
    assert chk[MODULES].notna().all().all()
    print("saved",path)
    return path

paths = {}
if not V8_SMOKE:
    paths["reference"] = write_submission(cal045_test, "submission_v8_cal045_reference.csv")
    for w in [0.50,0.55,0.60,0.675]:
        nm = f"w{str(w).replace('.','p')}"
        paths[nm] = write_submission(candidate_test[nm], f"submission_v8_repcal_{nm}.csv")
    paths["foldbag_w60"] = write_submission(primary_foldbag_test, "submission_v8_repcal_foldbag_w0p6.csv")

    auto_primary = primary_test if gate else cal045_test
    paths["primary"] = write_submission(auto_primary, "submission_v8_primary.csv")

    out_root = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("/mnt/data")
    np.savez_compressed(
        out_root / "v8_oof_artifacts.npz",
        y=y, base_oof=base_oof, base_test=base_test,
        seed_meta_oof=seed_meta_oof,
        seed_meta_test_full=seed_meta_test_full,
        seed_meta_test_foldbag=seed_meta_test_foldbag,
        rep_meta_oof=rep_meta_oof,
        rep_meta_test_full=rep_meta_test_full,
        rep_meta_test_foldbag=rep_meta_test_foldbag,
        cal045_oof=cal045_oof, cal045_test=cal045_test,
        primary_oof=primary_oof, primary_test=primary_test,
        ref_fold_id=ref_fold_id,
        cal_seeds=np.asarray(CAL_SEEDS,dtype=np.int64),
        loo_gains=np.asarray(loo_gains,dtype=np.float64),
        ref_fold_deltas=np.asarray(ref_fold_deltas,dtype=np.float64),
    )

    summary = {
        "reference_cal045_oof": reference_score,
        "primary_w": PRIMARY_W,
        "primary_oof": primary_score,
        "primary_delta": primary_score-reference_score,
        "bootstrap_ci": list(boot_ci),
        "reference_fold_wins": f"{fold_wins}/{N_FOLDS}",
        "reference_fold_deltas": [float(x) for x in ref_fold_deltas],
        "leave_one_seed_out_gains": {str(CAL_SEEDS[i]):float(loo_gains[i]) for i in range(len(CAL_SEEDS))},
        "test_top5_change_vs_cal045": test_top5_change,
        "test_top1_change_vs_cal045": test_top1_change,
        "gate": bool(gate),
    }
    with open(out_root/"v8_summary.json","w") as f:
        json.dump(summary,f,indent=2)

print("\nFINAL SUMMARY")
print("Public-tested reference      : calibration .45 (LB 0.65639)")
print("Reference OOF                :", round(reference_score,6))
print("Repeated-CV w=.60 OOF        :", round(primary_score,6))
print("Paired OOF gain              :", round(primary_score-reference_score,6))
print("Bootstrap 95% CI             :", tuple(round(x,6) for x in boot_ci))
print("Reference-fold wins          :", f"{fold_wins}/{N_FOLDS}")
print("LOO-seed minimum gain        :", round(float(min(loo_gains)),6))
print("Test Top5 change rate        :", round(test_top5_change,4))
print("V8 gate                      :", "PASS" if gate else "REVIEW")
if not V8_SMOKE:
    print("Auto-primary                 :", paths["primary"].name)
    print("Target leaderboard candidate : submission_v8_repcal_w0p6.csv")

# ---- cell 14 (code) ----
# Hard invariants — fail fast on corrupted outputs, not on performance fluctuations.
assert len(MODULES) == 17
assert y.shape == (4000,17)
assert np.isfinite(base_oof).all() and np.isfinite(base_test).all()
assert np.isfinite(seed_meta_oof).all()
assert np.isfinite(rep_meta_oof).all()
assert np.isfinite(primary_oof).all() and np.isfinite(primary_test).all()
assert seed_meta_oof.shape == (len(CAL_SEEDS),N_TRAIN,N_MODULES)
assert seed_meta_test_full.shape == (len(CAL_SEEDS),N_TEST,N_MODULES)
assert seed_meta_test_foldbag.shape == (len(CAL_SEEDS),N_TEST,N_MODULES)

if not V8_SMOKE:
    assert abs(reference_score - 0.662646) < 0.003, f"Unexpected reference drift: {reference_score:.6f}"

assert np.all(np.ptp(primary_test,axis=1) > 0)
print("All hard invariants passed.")
