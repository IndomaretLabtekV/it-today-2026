"""V127 corrected Top1 action selector.

Leak-free nested pipeline with a champion-structured (v42-lineage) comparator.

Design contract (from ``results/leader_audit/FINDINGS_AND_NEXT_MODEL.md``):
- every supervised component inside an outer fold trains without that fold's
  validation labels, including the anchor chain, action labels and heads;
- the comparator reconstructs the v42 lineage (sparse Ridge relevance+top1,
  per-module LightGBM calibration, frozen Consensus3 weights, LambdaRank on a
  long anchor-geometry matrix, fixed residual blend);
- action labels are exact counterfactual swap gains on the anchor being
  evaluated (verified against direct score-swap scoring);
- semantic features preserve module coordinates, not just cosines;
- at most one swap per user, consensus of two frozen-encoder heads;
- deployment only permutes existing champion score values on allowed users,
  preserving Top5 membership and rank5; gate failure copies champion bytes.

Known deviations from full v42 (documented, not hidden):
- no intent-state bundle / Hybrid34 / B34-B45 state machine on the train-side
  analogue (single LambdaRank surface instead of the V30+V31 pair);
- compact user surface (assessment vector) instead of the full 134-dim
  USER_FEATURES for calibration, plus light text cleaning instead of the full
  topic/style augmentation;
- float64 math instead of v42 float32;
- train-side OOF policy is evaluated on all users; test deployment additionally
  requires the V25 non-protected mask (S0 state reconstruction lives in v42).

Side-effect-free on import: no data loading, fitting, downloads or writes.
Heavy third-party imports (pandas/sklearn/lightgbm) happen lazily inside the
functions that need them so ``import v127_pipeline`` stays instant.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

POOLED_FLOOR = 0.002
MODULES = [f"M_{i:03d}" for i in range(1, 18)]
N_MODULES = len(MODULES)
DISCOUNT = 1.0 / np.log2(np.arange(2, 7))
ACTION_SLOTS = (1, 2, 3)

# Frozen Consensus3 weights (v42 cell 10): default 0.45 with module overrides.
CONSENSUS3_OVERRIDES = {
    "M_002": 0.44, "M_006": 0.55, "M_007": 0.60, "M_010": 0.40,
    "M_011": 0.55, "M_012": 0.55, "M_015": 0.44, "M_017": 0.53,
}
REFERENCE_W = 0.45

# V25 protected test users (v42: 2 KNOWN_PB_SWAPS + 6 V25_CORE_ACTIONS).
V25_PROTECTED_USERS = [
    "U_4047", "U_4953",
    "U_4074", "U_4176", "U_4491", "U_4567", "U_4836", "U_4929",
]

# Linear label gain for LambdaRank (v42 LABEL_GAIN): metric-compatible gain.
LABEL_GAIN = [0.0, 0.25, 0.40, 0.55, 0.70, 0.85, 1.00]


# ---------------------------------------------------------------------------
# Metric and ranking primitives
# ---------------------------------------------------------------------------

def per_user_ndcg(y_true, scores):
    """User-level NDCG@5 with linear gain (v42 convention)."""
    y_true = np.asarray(y_true, dtype=float)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, axis=1, kind="stable")[:, :5]
    rel = np.take_along_axis(y_true, order, axis=1)
    dcg = (rel * DISCOUNT).sum(axis=1)
    ideal = (np.sort(y_true, axis=1)[:, -5:][:, ::-1] * DISCOUNT).sum(axis=1)
    return dcg / np.maximum(ideal, 1e-12)


def ndcg5(y_true, scores):
    return float(per_user_ndcg(y_true, scores).mean())


def top5_indices(scores):
    scores = np.asarray(scores, dtype=float)
    idx = np.argpartition(-scores, 4, axis=1)[:, :5]
    local = np.take_along_axis(scores, idx, axis=1)
    return np.take_along_axis(idx, np.argsort(-local, axis=1), axis=1)


def stable_order(scores):
    return np.argsort(-np.asarray(scores, dtype=float), axis=1, kind="stable")


def apply_actions(scores, actions, allowed):
    """Swap the Top1 score value with the rank-``a`` value (``a`` in 1..3).

    ``actions`` holds candidate slots: 1 swaps Top1<->Top2, 2 swaps
    Top1<->Top3, 3 swaps Top1<->Top4. Values outside 1..3 (e.g. -1) abstain.
    Disallowed rows are returned bit-identical.
    """
    out = np.array(scores, dtype=float, copy=True)
    actions = np.asarray(actions, dtype=int).reshape(-1)
    allowed = np.asarray(allowed, dtype=bool).reshape(-1)
    if out.ndim != 2 or actions.shape[0] != out.shape[0] or allowed.shape[0] != out.shape[0]:
        raise AssertionError("apply_actions shape mismatch")
    order = stable_order(out)
    for i in range(out.shape[0]):
        a = int(actions[i])
        if not bool(allowed[i]) or a not in ACTION_SLOTS:
            continue
        p0, pa = int(order[i, 0]), int(order[i, a])
        out[i, p0], out[i, pa] = out[i, pa], out[i, p0]
    return out


def action_targets(y_true, scores):
    """Exact counterfactual gain of each Top1 swap, scored directly.

    Computed by applying the swap to the score matrix and re-scoring NDCG,
    so labels match deployment by construction (no closed-form drift).
    """
    y_true = np.asarray(y_true, dtype=float)
    scores = np.asarray(scores, dtype=float)
    base = per_user_ndcg(y_true, scores)
    allow = np.ones(y_true.shape[0], dtype=bool)
    out = np.zeros((y_true.shape[0], len(ACTION_SLOTS)))
    for k, slot in enumerate(ACTION_SLOTS):
        cand = apply_actions(scores, np.full(y_true.shape[0], slot), allow)
        out[:, k] = per_user_ndcg(y_true, cand) - base
    return out


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def _norm_rows(m):
    m = np.asarray(m, dtype=float)
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)


def semantic_features(user_emb, module_emb, scores):
    """Coordinate-preserving semantic features, one row per action.

    For each candidate challenger (anchor ranks 2..4) the features are the
    user.(challenger-incumbent) interaction, the challenger/incumbent dots,
    the squared difference norm and the raw difference coordinates. Pure
    cosines cannot separate candidates at equal angle to the user; the raw
    difference coordinates can.
    """
    u = _norm_rows(user_emb)
    m = _norm_rows(module_emb)
    s = np.asarray(scores, dtype=float)
    if u.ndim != 2 or m.ndim != 2 or s.ndim != 2:
        raise AssertionError("semantic_features expects 2D inputs")
    if u.shape[0] != s.shape[0] or m.shape[0] != s.shape[1]:
        raise AssertionError("semantic_features shape mismatch")
    order = stable_order(s)
    inc = m[order[:, 0]]
    blocks = []
    for k in range(len(ACTION_SLOTS)):
        ch = m[order[:, k + 1]]
        diff = ch - inc
        blocks.append(np.column_stack([
            (u * diff).sum(axis=1),
            (u * ch).sum(axis=1),
            (u * inc).sum(axis=1),
            (diff ** 2).sum(axis=1),
            diff,
        ]))
    return np.stack(blocks, axis=1)


def control_features(data, rows, anchor_scores):
    """Anchor/action features without embeddings (embedding control arm)."""
    rows = np.asarray(rows, dtype=int)
    anchor = np.asarray(anchor_scores, dtype=float)[rows]
    order = stable_order(anchor)
    margin = np.ptp(anchor[np.arange(len(rows))[:, None], order[:, :2]], axis=1)
    gap = anchor[np.arange(len(rows))[:, None], order[:, 1:4]] - anchor[np.arange(len(rows))[:, None], order[:, :1]]
    readiness = np.asarray(data.A[rows, :-1], dtype=float).mean(axis=1)
    invalid = np.asarray(data.A[rows, -1], dtype=float)
    rank = np.tile([1, 2, 3], (len(rows), 1)).astype(float)
    return np.stack([
        np.repeat(margin[:, None], 3, axis=1), rank, gap,
        np.repeat(readiness[:, None], 3, axis=1), np.repeat(invalid[:, None], 3, axis=1),
    ], axis=-1)


def action_features(data, rows, anchor_scores, user_emb, module_emb):
    """Control features plus one encoder's semantic features."""
    rows = np.asarray(rows, dtype=int)
    anchor = np.asarray(anchor_scores, dtype=float)
    ctrl = control_features(data, rows, anchor)
    sem = semantic_features(np.asarray(user_emb, dtype=float)[rows], module_emb, anchor[rows])
    return np.concatenate([ctrl, sem], axis=-1)


# ---------------------------------------------------------------------------
# Gate and deployment
# ---------------------------------------------------------------------------

def promotion_gate(pooled, ci, fold_gains, block_gains, dom_gain, violated):
    """Adopt only on a positive, certain, everywhere-non-negative increment."""
    if bool(violated):
        return False
    if not (float(pooled) >= POOLED_FLOOR):
        return False
    if not (float(ci[0]) > 0.0):
        return False
    if any(float(x) < 0.0 for x in fold_gains):
        return False
    if any(float(x) < 0.0 for x in block_gains):
        return False
    if not (float(dom_gain) >= 0.0):
        return False
    return True


def check_geometry(base_scores, cand_scores):
    """Top5 membership and rank5 must be identical on every row."""
    base = np.asarray(base_scores, dtype=float)
    cand = np.asarray(cand_scores, dtype=float)
    if base.shape != cand.shape:
        raise AssertionError("geometry shape mismatch")
    ob = stable_order(base)
    oc = stable_order(cand)
    bad = [i for i in range(base.shape[0])
           if set(ob[i, :5]) != set(oc[i, :5]) or ob[i, 4] != oc[i, 4]]
    if bad:
        raise AssertionError(f"geometry violated on rows {bad[:8]}")
    return True


def write_submission(source, out, scores, adopt, allowed):
    """Fail-close submission writer.

    Gate failure copies the champion bytes verbatim. Gate success writes the
    candidate only after locked rows verify bit-identical and every row
    preserves Top5 membership and rank5.
    """
    source, out = Path(source), Path(out)
    raw = source.read_bytes()
    if not bool(adopt):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(raw)
        return out
    with open(source, newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = list(reader)
    champ = np.array([[float(x) for x in r[1:]] for r in rows], dtype=float)
    scores = np.asarray(scores, dtype=float)
    allowed = np.asarray(allowed, dtype=bool).reshape(-1)
    if scores.shape != champ.shape or allowed.shape[0] != champ.shape[0]:
        raise AssertionError("submission shape mismatch")
    if not np.isfinite(scores).all():
        raise AssertionError("non-finite candidate scores")
    if not np.array_equal(scores[~allowed], champ[~allowed]):
        raise AssertionError("locked rows must stay bit-identical")
    check_geometry(champ, scores)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for uid, row in zip([r[0] for r in rows], scores):
            writer.writerow([uid] + [repr(float(v)) for v in row])
    return out


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class Data:
    user_ids: list
    n_train: int
    n_test: int
    y: np.ndarray
    groups: np.ndarray
    strata: np.ndarray
    A: np.ndarray
    docs: list
    modules: list = field(default_factory=lambda: list(MODULES))
    mod_texts: list = field(default_factory=list)


def load_data(data_dir):
    """Load organizer CSVs into label-free structures plus train labels."""
    import json
    import re

    import pandas as pd

    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train_relevance.csv")
    test = pd.read_csv(data_dir / "test.csv")
    chats = pd.read_csv(data_dir / "chat_history.csv")
    assess = pd.read_csv(data_dir / "user_assessments.csv")
    catalog = pd.read_csv(data_dir / "modules_catalog.csv")
    modules = list(train.columns[1:])
    if modules != list(MODULES):
        raise AssertionError("module catalogue mismatch")
    if set(train["user_id"]) & set(test["user_id"]):
        raise AssertionError("train/test overlap")
    n_train, n_test = len(train), len(test)
    y = train[MODULES].to_numpy(dtype=float)
    user_ids = list(train["user_id"]) + list(test["user_id"])

    words = {"nol": 0, "satu": 1, "dua": 2, "tiga": 3, "empat": 4,
             "lima": 5, "tidak": 0, "ya": 5}

    def parse(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(words.get(str(value).strip().lower(), np.nan))

    blobs = {r["user_id"]: json.loads(r["assessment_result"]) for _, r in assess.iterrows()}
    questions = sorted({q for b in blobs.values() for q in b})
    raw = np.array([[parse(blobs[u].get(q)) for q in questions] for u in user_ids], dtype=float)
    invalid = (~np.isfinite(raw)) | (raw < 0) | (raw > 5)
    raw = np.clip(np.where(np.isfinite(raw), raw, np.nan), 0, 5)
    col_mean = np.nanmean(raw, axis=0)
    raw = np.where(np.isnan(raw), col_mean, raw)
    a = (raw - raw.mean(axis=0)) / (raw.std(axis=0) + 1e-9)
    a = np.hstack([a, invalid.sum(axis=1, keepdims=True).astype(float)])

    chat_lists = chats.groupby("user_id")["user_chat_text"].apply(list)
    docs = [" ".join(str(t) for t in chat_lists.get(u, [])) for u in user_ids]

    def group_of(u):
        if u not in chat_lists:
            return "NO_CHAT"
        first = str(chat_lists[u][0]).lower().split()[:6]
        return re.sub(r"\d+", "#", " ".join(first)) if first else "NO_CHAT"

    groups = np.array([group_of(u) for u in train["user_id"]], dtype=object)
    strata = np.asarray(y.argmax(axis=1), dtype=int)
    mod_rows = catalog.set_index("module_id")
    mod_texts = [str(mod_rows.loc[m, "module_name"]) + " "
                 + str(mod_rows.loc[m, "description_and_syllabus"]) + " "
                 + str(mod_rows.loc[m, "prerequisite_level"]) for m in MODULES]
    return Data(user_ids=user_ids, n_train=n_train, n_test=n_test, y=y,
                groups=groups, strata=strata, A=a, docs=docs, mod_texts=mod_texts)


# ---------------------------------------------------------------------------
# Config and nested engine
# ---------------------------------------------------------------------------

@dataclass
class Config:
    seed: int = 6161
    smoke: bool = False
    outer_splits: int = 5
    inner_splits: int = 3
    calibration_trees: int = 150
    ranker_trees: int = 150
    word_features: int = 5000
    char_features: int = 8000
    threads: int = 4
    bootstraps: int = 2000
    thr_grid: tuple = (0.0, 0.0005, 0.001, 0.002)
    ridge_alpha: float = 3.0
    residual_alpha: float = 0.16


def _group_splits(strata_rows, groups_rows, n_splits, seed):
    """Group-disjoint folds, stratified when the counts allow it."""
    from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold

    strata_rows = np.asarray(strata_rows)
    groups_rows = np.asarray(groups_rows, dtype=object)
    n_groups = len(np.unique(groups_rows))
    if n_splits >= 2 and n_groups >= n_splits:
        try:
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
            return list(splitter.split(strata_rows, strata_rows, groups_rows))
        except ValueError:
            pass
    if n_groups >= 2:
        ns = max(2, min(int(n_splits), n_groups))
        try:
            return list(GroupKFold(n_splits=ns).split(strata_rows, strata_rows, groups_rows))
        except ValueError:
            pass
    ns = max(2, min(int(n_splits), len(strata_rows)))
    return list(KFold(n_splits=ns, shuffle=True, random_state=int(seed)).split(strata_rows))


def _user_splits(strata_rows, n_splits, seed):
    from sklearn.model_selection import KFold, StratifiedKFold

    strata_rows = np.asarray(strata_rows)
    try:
        splitter = StratifiedKFold(n_splits=int(n_splits), shuffle=True, random_state=int(seed))
        return list(splitter.split(strata_rows, strata_rows))
    except ValueError:
        ns = max(2, min(int(n_splits), len(strata_rows)))
        return list(KFold(n_splits=ns, shuffle=True, random_state=int(seed)).split(strata_rows))


def consensus_weights():
    w = np.full(N_MODULES, REFERENCE_W, dtype=float)
    for mid, val in CONSENSUS3_OVERRIDES.items():
        w[MODULES.index(mid)] = float(val)
    return w


def ordinal_grades(y_rows):
    y_rows = np.asarray(y_rows, dtype=float)
    grades = np.zeros_like(y_rows, dtype=np.int16)
    for i, row in enumerate(y_rows):
        pos = np.flatnonzero(row > 0)
        order = pos[np.argsort(-row[pos], kind="stable")]
        for rank, m in enumerate(order, 1):
            grades[i, m] = max(1, 7 - rank)
    return grades


def build_long_matrix(rows_all, anchor, base, meta, lex_word, lex_col, a_all):
    """V30-style long anchor-geometry matrix (one row per user-module)."""
    rows_all = np.asarray(rows_all, dtype=int)
    anchor = np.asarray(anchor, dtype=float)
    base = np.asarray(base, dtype=float)
    meta = np.asarray(meta, dtype=float)
    n = len(rows_all)
    ranks = np.empty_like(anchor, dtype=float)
    order = stable_order(anchor)
    ranks[np.arange(n)[:, None], order] = np.tile(np.arange(anchor.shape[1]), (n, 1)) / float(anchor.shape[1] - 1)
    top1 = anchor.max(axis=1, keepdims=True)
    cut5 = np.sort(anchor, axis=1)[:, 4:5]
    mu = anchor.mean(axis=1, keepdims=True)
    sd = np.maximum(anchor.std(axis=1, keepdims=True), 1e-4)
    prev_gap = np.zeros_like(anchor)
    next_gap = np.zeros_like(anchor)
    for i in range(n):
        oo = order[i]
        gaps = anchor[i, oo[:-1]] - anchor[i, oo[1:]]
        prev_gap[i, oo[1:]] = gaps
        next_gap[i, oo[:-1]] = gaps
    mods = np.tile(np.eye(anchor.shape[1]), (n, 1))
    users = np.repeat(np.asarray(a_all, dtype=float)[rows_all], anchor.shape[1], axis=0)
    parts = [anchor.reshape(-1, 1), ranks.reshape(-1, 1), (ranks <= 4.0 / (anchor.shape[1] - 1)).astype(float).reshape(-1, 1),
             (top1 - anchor).reshape(-1, 1), (anchor - cut5).reshape(-1, 1),
             prev_gap.reshape(-1, 1), next_gap.reshape(-1, 1), ((anchor - mu) / sd).reshape(-1, 1),
             base.reshape(-1, 1), meta.reshape(-1, 1), (meta - base).reshape(-1, 1),
             np.asarray(lex_word, dtype=float)[rows_all].reshape(-1, 1),
             np.asarray(lex_col, dtype=float)[rows_all].reshape(-1, 1),
             mods, users]
    mat = np.hstack(parts)
    if not np.isfinite(mat).all():
        raise AssertionError("non-finite long features")
    return mat


class Engine:
    """Fold-safe champion-structured anchor plus Top1 selector training."""

    def __init__(self, data, labels, cfg, cache):
        self.data = data
        self.labels = np.asarray(labels, dtype=float)
        self.cfg = cfg
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.fit_audit = []
        self._policy = None
        if self.labels.shape != (data.n_train, N_MODULES):
            raise AssertionError("label shape mismatch")

    # -- anchor chain ----------------------------------------------------
    def fit_anchor(self, train_rows, score_rows):
        """Fit the v42-structured chain on train_rows; score score_rows.

        Only ``labels[train_rows]`` may influence the result. Texts and
        embeddings are label-free. Every fit is recorded for audit.
        """
        from scipy.sparse import hstack as sparse_hstack
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import Ridge

        import lightgbm as lgb

        cfg = self.cfg
        data = self.data
        train_rows = np.asarray(sorted(int(i) for i in train_rows))
        score_rows = np.asarray([int(i) for i in score_rows])
        self.fit_audit.append({"stage": "anchor", "train_rows": [int(i) for i in train_rows]})
        min_df = 1 if (cfg.smoke or len(train_rows) < 200) else 2
        word_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, max_df=0.995,
                                   sublinear_tf=True, strip_accents="unicode",
                                   max_features=int(cfg.word_features))
        char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=min_df,
                                   sublinear_tf=True, max_features=int(cfg.char_features))
        train_docs = [data.docs[i] for i in train_rows]
        word_vec.fit(train_docs)
        char_vec.fit(train_docs)
        x_word_all = word_vec.transform(data.docs)
        x_char_all = char_vec.transform(data.docs)
        x_all = sparse_hstack([x_word_all, x_char_all], format="csr")
        mod_word = word_vec.transform(data.mod_texts)
        mod_char = char_vec.transform(data.mod_texts)
        lex_word = np.asarray((x_word_all @ mod_word.T).todense())
        lex_char = np.asarray((x_char_all @ mod_char.T).todense())

        def ridge_pair(fit_idx):
            rel = Ridge(alpha=3.0, solver="lsqr", tol=2e-3, max_iter=1000)
            top = Ridge(alpha=3.0, solver="lsqr", tol=2e-3, max_iter=1000)
            rel.fit(x_all[fit_idx], self.labels[fit_idx])
            top_bin = np.zeros_like(self.labels[fit_idx])
            top_bin[np.arange(len(fit_idx)), self.labels[fit_idx].argmax(axis=1)] = 1.0
            top.fit(x_all[fit_idx], top_bin)
            return rel, top

        inner = _group_splits(data.strata[train_rows], data.groups[train_rows],
                              cfg.inner_splits, cfg.seed + 11)
        base_oof = np.zeros((len(train_rows), N_MODULES))
        for itr_loc, iva_loc in inner:
            rel, top = ridge_pair(train_rows[itr_loc])
            base_oof[iva_loc] = 0.9 * rel.predict(x_all[train_rows[iva_loc]]) + 0.1 * top.predict(x_all[train_rows[iva_loc]])
        rel_full, top_full = ridge_pair(train_rows)
        base_sc = 0.9 * rel_full.predict(x_all[score_rows]) + 0.1 * top_full.predict(x_all[score_rows])

        child = 1 if cfg.smoke else 50

        def calibrator(feats, target):
            clf = lgb.LGBMClassifier(n_estimators=int(cfg.calibration_trees), learning_rate=0.03,
                                     num_leaves=15, min_child_samples=int(child),
                                     reg_alpha=0.5, reg_lambda=5.0, colsample_bytree=0.7,
                                     random_state=int(cfg.seed), n_jobs=int(cfg.threads),
                                     verbose=-1, deterministic=True)
            clf.fit(feats, target)
            return clf

        meta_oof = np.zeros_like(base_oof)
        for m in range(N_MODULES):
            target = (self.labels[train_rows, m] > 0).astype(int)
            for itr_loc, iva_loc in inner:
                clf = calibrator(np.column_stack([base_oof[itr_loc, m], data.A[train_rows[itr_loc]]]), target[itr_loc])
                meta_oof[iva_loc, m] = clf.predict_proba(np.column_stack([base_oof[iva_loc, m], data.A[train_rows[iva_loc]]]))[:, 1]
        w = consensus_weights()
        cons_oof = (1.0 - w) * base_oof + w * meta_oof
        meta_sc = np.zeros_like(base_sc)
        for m in range(N_MODULES):
            clf = calibrator(np.column_stack([base_oof[:, m], data.A[train_rows]]),
                             (self.labels[train_rows, m] > 0).astype(int))
            meta_sc[:, m] = clf.predict_proba(np.column_stack([base_sc[:, m], data.A[score_rows]]))[:, 1]
        cons_sc = (1.0 - w) * base_sc + w * meta_sc

        def ranker_fit_predict(long_tr, grades_tr, long_sc):
            ranker = lgb.LGBMRanker(objective="lambdarank", metric="ndcg",
                                    n_estimators=int(cfg.ranker_trees), learning_rate=0.025,
                                    num_leaves=15, min_child_samples=int(child),
                                    reg_alpha=1.0, reg_lambda=10.0, colsample_bytree=0.78,
                                    subsample=0.90, subsample_freq=1, max_bin=127,
                                    lambdarank_truncation_level=8, label_gain=list(LABEL_GAIN),
                                    random_state=int(cfg.seed), n_jobs=int(cfg.threads),
                                    verbose=-1, deterministic=True, force_col_wise=True)
            group = np.full(len(long_tr) // N_MODULES, N_MODULES, dtype=np.int32)
            ranker.fit(long_tr, grades_tr.reshape(-1).astype(np.int32), group=group)
            return ranker.predict(long_sc).reshape(len(long_sc) // N_MODULES, N_MODULES)

        grades_tr = ordinal_grades(self.labels[train_rows])
        rank_oof = np.zeros_like(cons_oof)
        for itr_loc, iva_loc in inner:
            long_tr = build_long_matrix(train_rows[itr_loc], cons_oof[itr_loc], base_oof[itr_loc],
                                        meta_oof[itr_loc], lex_word, lex_char, data.A)
            long_va = build_long_matrix(train_rows[iva_loc], cons_oof[iva_loc], base_oof[iva_loc],
                                        meta_oof[iva_loc], lex_word, lex_char, data.A)
            rank_oof[iva_loc] = ranker_fit_predict(long_tr, grades_tr[itr_loc], long_va)
        long_full = build_long_matrix(train_rows, cons_oof, base_oof, meta_oof, lex_word, lex_char, data.A)
        long_sc = build_long_matrix(score_rows, cons_sc, base_sc, meta_sc, lex_word, lex_char, data.A)
        rank_sc = ranker_fit_predict(long_full, grades_tr, long_sc)

        def residual(anchor, raw):
            raw = np.asarray(raw, dtype=float)
            anchor = np.asarray(anchor, dtype=float)
            aligned = (anchor.mean(axis=1, keepdims=True)
                       + (raw - raw.mean(axis=1, keepdims=True))
                       * (np.maximum(anchor.std(axis=1, keepdims=True), 1e-5)
                          / np.maximum(raw.std(axis=1, keepdims=True), 1e-5)))
            return anchor + float(cfg.residual_alpha) * (aligned - anchor)

        return {"scores": residual(cons_sc, rank_sc),
                "train_scores": residual(cons_oof, rank_oof),
                "base": base_sc, "meta": meta_sc, "cons": cons_sc}

    # -- selector --------------------------------------------------------
    def _action_rows(self, idx):
        return (np.asarray(idx, dtype=int)[:, None] * 3 + np.arange(3)).ravel()

    def policy(self, rows):
        """Fold-safe anchor scores for every user from a fit on ``rows``."""
        key = tuple(sorted(int(i) for i in np.asarray(rows, dtype=int).reshape(-1)))
        if self._policy is not None and self._policy[0] == key:
            return self._policy[1]
        full = self.fit_anchor(np.array(key, dtype=int),
                               np.arange(self.data.n_train + self.data.n_test))
        result = {"scores": full["scores"],
                  "order": stable_order(full["scores"]),
                  "challengers": stable_order(full["scores"])[:, 1:4]}
        self._policy = (key, result)
        return result


def fit_selector(engine, allowed, e5_users, e5_modules, nomic_users, nomic_modules):
    """Train control + dual-encoder utility heads on ``allowed`` rows only."""
    from sklearn.linear_model import Ridge

    data = engine.data
    cfg = engine.cfg
    allowed = np.asarray(sorted(int(i) for i in np.asarray(allowed, dtype=int).reshape(-1)))
    inner = _group_splits(data.strata[allowed], data.groups[allowed], cfg.inner_splits, cfg.seed + 101)
    anchor_oof = np.zeros((len(allowed), N_MODULES))
    for itr_loc, iva_loc in inner:
        part = engine.fit_anchor(allowed[itr_loc], allowed[iva_loc])
        anchor_oof[iva_loc] = part["scores"]
    full = engine.fit_anchor(allowed, np.arange(data.n_train + data.n_test))
    engine._policy = (tuple(int(i) for i in allowed),
                      {"scores": full["scores"], "order": stable_order(full["scores"]),
                       "challengers": stable_order(full["scores"])[:, 1:4]})
    anchor_mat = full["scores"].copy()
    anchor_mat[allowed] = anchor_oof
    targets = action_targets(engine.labels[allowed], anchor_oof)
    ctrl = control_features(data, allowed, anchor_mat)
    feat_e = np.concatenate([ctrl, semantic_features(np.asarray(e5_users)[allowed], e5_modules, anchor_mat[allowed])], axis=-1)
    feat_n = np.concatenate([ctrl, semantic_features(np.asarray(nomic_users)[allowed], nomic_modules, anchor_mat[allowed])], axis=-1)
    flat = targets.reshape(-1)
    head_c = Ridge(alpha=float(cfg.ridge_alpha)).fit(ctrl.reshape(-1, ctrl.shape[-1]), flat)
    head_e = Ridge(alpha=float(cfg.ridge_alpha)).fit(feat_e.reshape(-1, feat_e.shape[-1]), flat)
    head_n = Ridge(alpha=float(cfg.ridge_alpha)).fit(feat_n.reshape(-1, feat_n.shape[-1]), flat)

    grid = [float(t) for t in cfg.thr_grid]
    sel = _user_splits(data.strata[allowed], 2, cfg.seed + 202)
    str_u = allowed[sel[0][0]]
    stv_u = allowed[sel[0][1]]

    def _heads_for(users, anchor_block):
        ctrl_b = control_features(data, users, anchor_block)
        fe_b = np.concatenate([ctrl_b, semantic_features(np.asarray(e5_users)[users], e5_modules, anchor_block[users])], axis=-1)
        fn_b = np.concatenate([ctrl_b, semantic_features(np.asarray(nomic_users)[users], nomic_modules, anchor_block[users])], axis=-1)
        return ctrl_b, fe_b, fn_b

    sub = engine.fit_anchor(str_u, np.arange(data.n_train + data.n_test))
    sub_mat = sub["scores"].copy()
    sub_mat[str_u] = sub["train_scores"]
    ctrl_s, fe_s, fn_s = _heads_for(str_u, sub_mat)
    tgt_s = action_targets(engine.labels[str_u], sub["train_scores"]).reshape(-1)
    he_s = Ridge(alpha=float(cfg.ridge_alpha)).fit(fe_s.reshape(-1, fe_s.shape[-1]), tgt_s)
    hn_s = Ridge(alpha=float(cfg.ridge_alpha)).fit(fn_s.reshape(-1, fn_s.shape[-1]), tgt_s)
    hc_s = Ridge(alpha=float(cfg.ridge_alpha)).fit(ctrl_s.reshape(-1, ctrl_s.shape[-1]), tgt_s)
    ctrl_v, fe_v, fn_v = _heads_for(stv_u, sub["scores"])
    u1 = he_s.predict(fe_v.reshape(-1, fe_v.shape[-1])).reshape(-1, 3)
    u2 = hn_s.predict(fn_v.reshape(-1, fn_v.shape[-1])).reshape(-1, 3)
    uc = hc_s.predict(ctrl_v.reshape(-1, ctrl_v.shape[-1])).reshape(-1, 3)
    base_v = per_user_ndcg(engine.labels[stv_u], sub["scores"][stv_u])
    gains_f, gains_c = [], []
    for thr in grid:
        out = sub["scores"][stv_u].copy()
        for j in range(len(stv_u)):
            a1, a2 = int(u1[j].argmax()), int(u2[j].argmax())
            if a1 == a2 and u1[j, a1] > thr and u2[j, a2] > thr:
                out[j] = apply_actions(out[j:j + 1], [a1 + 1], [True])[0]
        gains_f.append(float(per_user_ndcg(engine.labels[stv_u], out).mean() - base_v.mean()))
        out_c = sub["scores"][stv_u].copy()
        for j in range(len(stv_u)):
            a = int(uc[j].argmax())
            if uc[j, a] > thr:
                out_c[j] = apply_actions(out_c[j:j + 1], [a + 1], [True])[0]
        gains_c.append(float(per_user_ndcg(engine.labels[stv_u], out_c).mean() - base_v.mean()))
    mean_f = np.array(gains_f)
    mean_c = np.array(gains_c)
    best_f = grid[int(np.where(mean_f == mean_f.max())[0][-1])]
    best_c = grid[int(np.where(mean_c == mean_c.max())[0][-1])]
    engine.fit_audit.append({"stage": "selector", "train_rows": [int(i) for i in allowed]})
    return {"control": head_c, "e5": head_e, "nomic": head_n,
            "threshold": float(best_f), "threshold_control": float(best_c),
            "anchor_oof": anchor_oof, "anchor_full": full["scores"]}




def verify_audit(engine, allowed):
    """Raise unless every recorded fit used only ``allowed`` rows."""
    allowed = set(int(i) for i in np.asarray(allowed).reshape(-1))
    for rec in engine.fit_audit:
        if not set(rec["train_rows"]) <= allowed:
            raise AssertionError(f"nested fit leaked outside allowed rows: {rec.get('stage')}")
    return True


# ---------------------------------------------------------------------------
# Full outer experiment
# ---------------------------------------------------------------------------

def cluster_bootstrap_ci(values, groups, seed, n_boot=2000):
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups, dtype=object)
    uniq = np.unique(groups)
    by = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(int(seed))
    draws = rng.choice(uniq, size=(int(n_boot), len(uniq)), replace=True)
    means = np.array([values[np.concatenate([by[g] for g in row])].mean() for row in draws])
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def run_experiment(data, labels, cfg, e5_users, e5_modules, nomic_users, nomic_modules, engine=None):
    """Nested outer evaluation of the agree-policy against its anchor."""
    labels = np.asarray(labels, dtype=float)
    if engine is None:
        engine = Engine(data, labels, cfg, cfg_cache_dir(cfg))
    elif not np.array_equal(engine.labels, labels):
        raise AssertionError("engine labels mismatch")
    n_train = data.n_train
    outer = _group_splits(data.strata, data.groups, cfg.outer_splits, cfg.seed)
    anchor_oof = np.zeros((n_train, N_MODULES))
    fin_c = np.zeros((n_train, N_MODULES))
    fin_f = np.zeros((n_train, N_MODULES))
    fold_rows = []
    thr_f, thr_c = [], []
    for fi, (tr_loc, va_loc) in enumerate(outer):
        tr = np.arange(n_train)[tr_loc]
        va = np.arange(n_train)[va_loc]
        mark = len(engine.fit_audit)
        fit = fit_selector(engine, tr, e5_users, e5_modules, nomic_users, nomic_modules)
        for rec in engine.fit_audit[mark:]:
            if not set(rec["train_rows"]) <= set(int(i) for i in tr):
                raise AssertionError(f"outer fold {fi}: selector fit saw validation rows")
        anchor = fit["anchor_full"][:n_train]
        anchor_oof[va] = anchor[va]
        ctrl_va = control_features(data, va, anchor)
        fe_va = np.concatenate([ctrl_va, semantic_features(np.asarray(e5_users)[va], e5_modules, anchor[va])], axis=-1)
        fn_va = np.concatenate([ctrl_va, semantic_features(np.asarray(nomic_users)[va], nomic_modules, anchor[va])], axis=-1)
        uc = fit["control"].predict(ctrl_va.reshape(-1, ctrl_va.shape[-1])).reshape(-1, 3)
        u1 = fit["e5"].predict(fe_va.reshape(-1, fe_va.shape[-1])).reshape(-1, 3)
        u2 = fit["nomic"].predict(fn_va.reshape(-1, fn_va.shape[-1])).reshape(-1, 3)
        out_c = anchor[va].copy()
        out_f = anchor[va].copy()
        n_c = n_f = 0
        for j in range(len(va)):
            a = int(uc[j].argmax())
            if uc[j, a] > fit["threshold_control"]:
                out_c[j] = apply_actions(anchor[va][j:j + 1], [a + 1], [True])[0]
                n_c += 1
            a1, a2 = int(u1[j].argmax()), int(u2[j].argmax())
            if a1 == a2 and u1[j, a1] > fit["threshold"] and u2[j, a2] > fit["threshold"]:
                out_f[j] = apply_actions(anchor[va][j:j + 1], [a1 + 1], [True])[0]
                n_f += 1
        fin_c[va] = out_c
        fin_f[va] = out_f
        thr_f.append(float(fit["threshold"]))
        thr_c.append(float(fit["threshold_control"]))
        fold_rows.append({"fold": fi, "thr_c": float(fit["threshold_control"]), "thr_f": float(fit["threshold"]),
                          "n_c": int(n_c), "n_f": int(n_f),
                          "oof_anchor": float(per_user_ndcg(labels[va], anchor[va]).mean()),
                          "oof_ctrl": float(per_user_ndcg(labels[va], out_c).mean()),
                          "oof_full": float(per_user_ndcg(labels[va], out_f).mean())})
    n_c_all = per_user_ndcg(labels, fin_c)
    n_f_all = per_user_ndcg(labels, fin_f)
    n_a_all = per_user_ndcg(labels, anchor_oof)
    d_c, d_f = n_c_all - n_a_all, n_f_all - n_a_all
    ci_f = cluster_bootstrap_ci(d_f, data.groups, cfg.seed, cfg.bootstraps)
    ci_c = cluster_bootstrap_ci(d_c, data.groups, cfg.seed + 1, cfg.bootstraps)
    fold_ids = np.zeros(n_train, dtype=int)
    for fi, (_, va_loc) in enumerate(outer):
        fold_ids[np.arange(n_train)[va_loc]] = fi
    fold_f = np.array([d_f[fold_ids == fi].mean() for fi in range(len(outer))])
    fold_c = np.array([d_c[fold_ids == fi].mean() for fi in range(len(outer))])
    top1 = stable_order(anchor_oof)[:, 0]
    blocks = {int(m): [float(d_f[top1 == m].mean()), int((top1 == m).sum())] for m in np.unique(top1)}
    dom = max(blocks, key=lambda m: blocks[m][1])
    no_chat = np.array([len(data.docs[i]) == 0 for i in range(n_train)])
    try:
        check_geometry(anchor_oof, fin_f)
        violated = False
    except AssertionError:
        violated = True
    gate = promotion_gate(float(d_f.mean()), ci_f, fold_f,
                          [v[0] for v in blocks.values()], blocks[dom][0], violated)

    full = fit_selector(engine, np.arange(n_train), e5_users, e5_modules, nomic_users, nomic_modules)
    anchor_te = full["anchor_full"][n_train:]
    ctrl_te = control_features(data, np.arange(n_train, n_train + data.n_test), full["anchor_full"])
    fe_te = np.concatenate([ctrl_te, semantic_features(np.asarray(e5_users)[n_train:], e5_modules, anchor_te)], axis=-1)
    fn_te = np.concatenate([ctrl_te, semantic_features(np.asarray(nomic_users)[n_train:], nomic_modules, anchor_te)], axis=-1)
    u1 = full["e5"].predict(fe_te.reshape(-1, fe_te.shape[-1])).reshape(-1, 3)
    u2 = full["nomic"].predict(fn_te.reshape(-1, fn_te.shape[-1])).reshape(-1, 3)
    fin_te = anchor_te.copy()
    actions = np.full(data.n_test, -1)
    for j in range(data.n_test):
        a1, a2 = int(u1[j].argmax()), int(u2[j].argmax())
        if a1 == a2 and u1[j, a1] > full["threshold"] and u2[j, a2] > full["threshold"]:
            fin_te[j] = apply_actions(anchor_te[j:j + 1], [a1 + 1], [True])[0]
            actions[j] = a1 + 1
    check_geometry(anchor_te, fin_te)
    return {"anchor_oof": anchor_oof, "fin_c": fin_c, "fin_f": fin_f,
            "fin_te": fin_te, "test_actions": actions,
            "thr_f": thr_f, "thr_c": thr_c, "thr_test": float(full["threshold"]),
            "fold_table": fold_rows, "fold_ids": fold_ids,
            "pooled_full": float(d_f.mean()), "ci_full": [float(ci_f[0]), float(ci_f[1])],
            "worst_fold_full": float(fold_f.min()),
            "pooled_ctrl": float(d_c.mean()), "ci_ctrl": [float(ci_c[0]), float(ci_c[1])],
            "worst_fold_ctrl": float(fold_c.min()),
            "fold_full": [float(x) for x in fold_f], "fold_ctrl": [float(x) for x in fold_c],
            "blocks": blocks, "dominant_block": [int(dom), float(blocks[dom][0])],
            "min_block": float(min(v[0] for v in blocks.values())),
            "no_chat_n": int(no_chat.sum()),
            "no_chat_gain": float(d_f[no_chat].mean()) if no_chat.sum() else float("nan"),
            "deployed_full": int((np.abs(fin_f - anchor_oof).sum(axis=1) > 0).sum()),
            "deployed_ctrl": int((np.abs(fin_c - anchor_oof).sum(axis=1) > 0).sum()),
            "gate": bool(gate), "violated": bool(violated),
            "fit_audit": [dict(r) for r in engine.fit_audit]}


def cfg_cache_dir(cfg):
    del cfg
    return Path("results/v127/_engine_cache")
