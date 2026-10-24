# Converted from notebooks/v49.ipynb: markdown cells become comments, %/! magics neutralized.

# ---- cell 0 (markdown) ----
# # MineToday V49 - Historical Prediction Ensemble
#
# ## Objective and evidence-based decision
#
# V49 tests whether honest historical OOF prediction surfaces contain complementary ranking signal beyond the current public `0.66028` champion. The competition metric is mean user-level NDCG@5 over 17 modules. No public leaderboard score is used as a feature, weight, threshold, or selection target.
#
# Repository evidence rules out another broad replacement model. V8, V9.1, V13.1, V19, V28, V35, and V41 show that attractive reused-OOF or CAL gains can reverse on chronology, template-held-out validation, LOCK, or public evaluation. V30 and the later consensus packages show that restrained residual authority and model agreement can transfer. V46 and V47 were not executed successfully, while V48 failed before ensemble evaluation because its Qwen runtime did not initialize.
#
# The selected architecture is therefore a low-capacity, family-balanced forward ensemble in rank space. It uses a mandatory V30 OOF proxy, adds at most four distinct historical families, gives the anchor two prior votes, and applies only low-coverage permutations of the anchor's existing Top-4 score values. Two anchor votes make adjacent changes require corroboration while still letting a sharply different expert earn measurable marginal value. The search compares two preregistered policies: preserve Top1 while reordering ranks 2-4, or reorder the full Top-4. Top-5 membership, ranks 5-17, and all rows whose Top-5 geometry differs between the champion lineage, V30, or Consensus3 remain immutable.
#
# ## Required Kaggle inputs
#
# Attach the organizer dataset containing `chat_history.csv`, `modules_catalog.csv`, `sample_submission.csv`, `test.csv`, `train_relevance.csv`, and `user_assessments.csv`. Also attach a participant-owned historical artifact dataset that preserves this repository layout under one dataset root. At minimum it must contain:
#
# - `results/v12_2/v12_2_oof_artifacts.npz`
# - `results/v32/v32_test_score_artifacts.npz`
# - `results/v33/v33_stage1_oof_artifacts.npz`
# - `results/v30/submission_v30_primary.csv`
# - `results/v48/v48/v48_split_manifest.csv`
# - one exact champion reference accepted by the loader, preferably `results/v48/v48/submission_v48_anchor_066028_reference.csv`
#
# For the intended ensemble search, include the compact prediction files referenced by `CANDIDATE_SPECS` below. The full 16 GB results tree is not required. The rule evidence recorded in `MineToday_Full_Chat_Handoff_2026-09-04_CURRENT.zip` permits multiple models, prohibits external data, and requires a reproducible final notebook. These artifacts are outputs of the user's own organizer-data experiments, not external labeled data. The dataset must be declared as a notebook input. If the organizer requires every base prediction to be regenerated from raw data in the final notebook, do not submit V49 in artifact-stage form.
#
# ## Validation and leakage contract
#
# Only prediction matrices documented as genuine cross-fitted train predictions and paired with the matching test configuration may enter learned selection. In-sample train predictions, partial CAL or LOCK matrices, ambiguous OOF-to-test mappings, public-adapted action surfaces without an honest train analogue, nonnested V13.1 and V41 stacks, contaminated V35 TAPT output, and non-executed V3, V46, V47, and V48 experts are excluded. File count is never treated as model diversity.
#
# The exact V48 split manifest is reused because direct V47 and V48 artifacts show that its 598-user LOCK was never opened. DEVELOPMENT contains 3,402 users. Five outer latest-template-grouped folds fit every expert choice, family representative, coverage, authority mode, and ensemble prefix using only the corresponding outer-train labels. Historical artifact design was adaptive, so DEVELOPMENT is explicitly model-selection data rather than a pristine holdout. LOCK is evaluated once only if the nested DEVELOPMENT gate passes. Test predictions are never used to choose experts or weights.
#
# The current champion has no exact honest OOF counterpart because some components were selected through public causal probes. V30's fully cross-fitted surface is the conservative train-side proxy. On test, V49 starts from the exact champion and protects every row where champion, V30, and Consensus3 geometry disagree. This limitation is explicit in every audit and is why deployment gates are strict.
#
# ## Selection, diagnostics, and fail-close behavior
#
# Candidates are aligned to organizer `user_id` and module order, checked for finite values, hashed for exact score and rank duplicates, and clustered for near-duplicate rank behavior on DEVELOPMENT only. Within each outer fit, weak experts are filtered, one representative per family and near-duplicate cluster is retained, and direct NDCG@5 forward selection chooses at most four families. Vote-strength ties favor the proposal with the smaller normalized disturbance to the anchor scores, reducing row-order tie breaks. The final policy must beat the anchor and the best single-expert anchored policy with stable fold, ID-block, template-cluster, bootstrap, and action-level behavior.
#
# If required provenance, alignment, DEVELOPMENT, LOCK, anchor hash, or test geometry fails, `submission_v49_primary.csv` is the exact champion reference and `v49_summary.json` labels it `exact_anchor_fail_close`. It is not presented as an improvement. A passing run writes exactly one primary submission plus explicit diagnostics under `/kaggle/working/results/v49/`. Seeds, package versions, paths, hashes, gates, selected experts, and weights are recorded. This notebook was statically prepared and contains no fabricated execution output.

# ---- cell 1 (code) ----
from pathlib import Path
from collections import Counter
import hashlib
import json
import math
import os
import platform
import random
import re
import traceback

os.environ.setdefault('PYTHONHASHSEED', '20260905')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')

import numpy as np
import pandas as pd
import sklearn
from IPython.display import display
from sklearn.model_selection import StratifiedGroupKFold

SEED = 20260905
random.seed(SEED)
np.random.seed(SEED)

SMOKE = bool(int(os.getenv('V49_SMOKE', '0')))
OUT_DIR = Path('/kaggle/working/results/v49') if Path('/kaggle/working').exists() else Path.cwd() / 'results' / 'v49'
OUT_DIR.mkdir(parents=True, exist_ok=True)

PUBLIC_ANCHOR = 0.66028
PUBLIC_TOP5_CUTOFF = 0.66076
PUBLIC_LEADER = 0.66142
EXPECTED_ANCHOR_TOP5_SHA256 = '7c0d29a7309b7283a043b7d37ca5556ba7c0ca2270b0137a6e1e0878a69a5fe7'
EXPECTED_ANCHOR_FILE_SHA256 = '443785fb534df9820ce57bfe450a2768881625bf7ae37aa849dd0d9f2afc957d'
EXPECTED_CORE_FILE_SHA256 = {
    'results/v12_2/v12_2_oof_artifacts.npz': '4a070b640abf12103f8ae5717b8d78a7ebb5afc667f3517956ec744ba09718e7',
    'results/v30/submission_v30_primary.csv': '8bb5631d9384d0b1b2147dfd683850905d1311139fe973bcd8ee2c67b0b289ad',
    'results/v32/v32_test_score_artifacts.npz': '0148c55f93719464af4dbf85783a0a915e1fa3708793e57618887a64c9f13142',
    'results/v33/v33_stage1_oof_artifacts.npz': '5b4326a21b50563c21ff0956634a025e93c8657bee4449f4a0b3e61d83be50c6',
    'results/v48/v48/v48_split_manifest.csv': '0be4b5697c3833b4d06b9887629321d93865a58eece42a787702c05fbfc43cce',
}
EXPECTED_SPLIT_COUNTS = {'ACTION_TRAIN': 2595, 'DESIGN': 807, 'LOCK': 598}

OUTER_FOLDS = 2 if SMOKE else 5
BOOTSTRAPS = 100 if SMOKE else 2500
ID_BLOCKS = 2 if SMOKE else 5
ANCHOR_PRIOR_VOTES = 2
MAX_SELECTED_FAMILIES = 2 if SMOKE else 4
COVERAGE_GRID = [0.01] if SMOKE else [0.005, 0.010, 0.015, 0.020, 0.030]
AUTHORITY_MODES = ['preserve_top1', 'head4']
QUALITY_FLOOR_BELOW_ANCHOR = 0.030
NEAR_RANK_CORR = 0.995
NEAR_TOP1_AGREEMENT = 0.990
NEAR_TOP5_JACCARD = 0.990
MIN_LOADED_FAMILIES = 3
MIN_ELIGIBLE_DEVELOPMENT = 250

DEV_MIN_DELTA = -1.0 if SMOKE else 0.00045
DEV_MIN_OUTER_WINS = 0 if SMOKE else 4
DEV_MIN_ID_BLOCK_WINS = 0 if SMOKE else 4
DEV_MIN_BENEFIT_HARM = 0.0 if SMOKE else 1.50
DEV_WORST_FOLD_FLOOR = -1.0 if SMOKE else -0.00020
DEV_MIN_EDGE_VS_SINGLE = -1.0 if SMOKE else 0.00001
DEV_MIN_FAMILY_STABILITY = 0.0 if SMOKE else 0.25
LOCK_POOLED_MIN_DELTA = -1.0 if SMOKE else 0.00040
LOCK_MIN_BENEFIT_HARM = 0.0 if SMOKE else 1.25
LOCK_BLOCK_FLOOR = -1.0 if SMOKE else -0.00025
TEST_MIN_ACTIONS = 0 if SMOKE else 3
TEST_MAX_ACTIONS = 999 if SMOKE else 50

print('V49 runtime:', platform.python_version(), 'numpy', np.__version__, 'pandas', pd.__version__, 'sklearn', sklearn.__version__)
print('CPU-first artifact ensemble; smoke mode =', SMOKE)
print('public scores are audit metadata only:', PUBLIC_ANCHOR, PUBLIC_TOP5_CUTOFF, PUBLIC_LEADER)

# ---- cell 2 (code) ----
DATA_FILES = {
    'chat_history.csv', 'modules_catalog.csv', 'sample_submission.csv',
    'test.csv', 'train_relevance.csv', 'user_assessments.csv',
}

def _existing_roots():
    roots = [Path('/kaggle/input'), Path.cwd(), Path('/mnt/data')]
    return [p for p in roots if p.exists()]

def locate_data_dir():
    override = os.getenv('V49_DATA_DIR', '').strip()
    if override:
        d = Path(override)
        if DATA_FILES.issubset({p.name for p in d.iterdir() if p.is_file()}):
            return d
        raise FileNotFoundError(f'V49_DATA_DIR does not contain all organizer files: {d}')
    candidates = []
    for root in _existing_roots():
        direct = [root]
        try:
            direct += sorted([p for p in root.iterdir() if p.is_dir()])
        except PermissionError:
            pass
        for d in direct:
            try:
                names = {p.name for p in d.iterdir() if p.is_file()}
            except (OSError, PermissionError):
                continue
            if DATA_FILES.issubset(names):
                candidates.append(d)
        if candidates:
            break
        for marker in sorted(root.rglob('train_relevance.csv')):
            d = marker.parent
            try:
                names = {p.name for p in d.iterdir() if p.is_file()}
            except OSError:
                continue
            if DATA_FILES.issubset(names):
                candidates.append(d)
    if not candidates:
        raise FileNotFoundError('Could not locate the six organizer data files.')
    return sorted(set(candidates), key=lambda p: (len(p.parts), p.as_posix()))[0]

ARTIFACT_MARKERS = [
    'results/v48/v48/submission_v48_anchor_066028_reference.csv',
    'results/v44/submission_v44_anchor_066028_reference.csv',
    'results/v12_2/v12_2_oof_artifacts.npz',
]

def _root_from_marker(path, relative):
    root = path
    for _ in Path(relative).parts:
        root = root.parent
    return root

def locate_artifact_root():
    override = os.getenv('V49_ARTIFACT_ROOT', '').strip()
    if override:
        root = Path(override)
        if any((root / rel).is_file() for rel in ARTIFACT_MARKERS):
            return root
        raise FileNotFoundError(f'V49_ARTIFACT_ROOT has no recognized marker: {root}')
    for base in _existing_roots():
        for rel in ARTIFACT_MARKERS:
            direct = base / rel
            if direct.is_file():
                return base
            try:
                children = sorted([p for p in base.iterdir() if p.is_dir()])
            except (OSError, PermissionError):
                children = []
            for child in children:
                if (child / rel).is_file():
                    return child
        for rel in ARTIFACT_MARKERS:
            matches = sorted(base.rglob(Path(rel).name))
            for path in matches:
                if path.is_file() and path.as_posix().endswith(rel):
                    return _root_from_marker(path, rel)
    raise FileNotFoundError('Could not locate the declared historical artifact dataset.')

DATA_DIR = locate_data_dir()
ARTIFACT_ROOT = locate_artifact_root()

def resolve_rel(relative, required=False):
    path = ARTIFACT_ROOT / relative
    if path.is_file():
        return path
    if required:
        raise FileNotFoundError(f'Missing required artifact: {relative}')
    return None

train = pd.read_csv(DATA_DIR / 'train_relevance.csv')
test = pd.read_csv(DATA_DIR / 'test.csv')
sample = pd.read_csv(DATA_DIR / 'sample_submission.csv')
chat = pd.read_csv(DATA_DIR / 'chat_history.csv')
assessments = pd.read_csv(DATA_DIR / 'user_assessments.csv')
modules = pd.read_csv(DATA_DIR / 'modules_catalog.csv')

MODULES = list(train.columns[1:])
TRAIN_IDS = train['user_id'].astype(str).to_numpy()
TEST_IDS = test['user_id'].astype(str).to_numpy()
Y = train[MODULES].to_numpy(np.float32)

assert len(MODULES) == 17 and MODULES == list(sample.columns[1:])
assert len(train) == 4000 and len(test) == 1000
assert train['user_id'].is_unique and test['user_id'].is_unique
assert sample['user_id'].is_unique and assessments['user_id'].is_unique
assert set(TRAIN_IDS).isdisjoint(set(TEST_IDS))
assert sample['user_id'].astype(str).tolist() == TEST_IDS.tolist()
assert modules['module_id'].is_unique and set(modules['module_id'].astype(str)) == set(MODULES)
assert set(TRAIN_IDS).issubset(set(assessments['user_id'].astype(str)))
assert set(TEST_IDS).issubset(set(assessments['user_id'].astype(str)))
assert chat['user_id'].notna().all()
CHAT_OUT_OF_SCOPE_USERS = sorted(set(chat['user_id'].astype(str)) - set(TRAIN_IDS).union(TEST_IDS))
assert np.isfinite(Y).all() and (Y >= 0).all() and (Y.max(axis=1) > 0).all()

print('DATA_DIR =', DATA_DIR)
print('ARTIFACT_ROOT =', ARTIFACT_ROOT)
print('train/test/modules =', len(train), len(test), len(MODULES))
print('unused out-of-cohort chat users =', len(CHAT_OUT_OF_SCOPE_USERS))

# ---- cell 3 (code) ----
DISCOUNT5 = 1.0 / np.log2(np.arange(2, 7, dtype=np.float64))

def stable_order(scores):
    scores = np.asarray(scores)
    if scores.ndim != 2 or scores.shape[1] != len(MODULES):
        raise ValueError(f'Expected a two-dimensional 17-module matrix, got {scores.shape}')
    return np.argsort(-scores, axis=1, kind='mergesort')

def top5_indices(scores):
    scores = np.asarray(scores)
    idx = np.argpartition(-scores, 4, axis=1)[:, :5]
    local = np.take_along_axis(scores, idx, axis=1)
    return np.take_along_axis(idx, np.argsort(-local, axis=1, kind='mergesort'), axis=1)

def per_user_ndcg(y_true, scores):
    y_true = np.asarray(y_true, np.float64)
    scores = np.asarray(scores, np.float64)
    if y_true.shape != scores.shape:
        raise ValueError(f'NDCG shape mismatch: {y_true.shape} versus {scores.shape}')
    idx = top5_indices(scores)
    dcg = (np.take_along_axis(y_true, idx, axis=1) * DISCOUNT5).sum(axis=1)
    ideal = np.sort(y_true, axis=1)[:, -5:][:, ::-1]
    idcg = (ideal * DISCOUNT5).sum(axis=1)
    return dcg / np.maximum(idcg, 1e-12)

def ndcg5(y_true, scores):
    return float(per_user_ndcg(y_true, scores).mean())

def rank_matrix(scores):
    order = stable_order(scores)
    ranks = np.empty_like(order, dtype=np.int16)
    ranks[np.arange(len(order))[:, None], order] = np.arange(order.shape[1], dtype=np.int16)
    return ranks

def matrix_sha256(scores):
    arr = np.ascontiguousarray(np.asarray(scores, np.float32))
    return hashlib.sha256(arr.tobytes()).hexdigest()

def rank_sha256(scores):
    arr = np.ascontiguousarray(stable_order(scores).astype(np.int16))
    return hashlib.sha256(arr.tobytes()).hexdigest()

def top5_sha256(scores):
    arr = np.ascontiguousarray(top5_indices(scores).astype(np.int16))
    return hashlib.sha256(arr.tobytes()).hexdigest()

def same_top_order(a, b, k=5):
    return np.all(stable_order(a)[:, :k] == stable_order(b)[:, :k], axis=1)

def mean_top5_jaccard(a, b):
    ta, tb = top5_indices(a), top5_indices(b)
    return float(np.mean([len(set(x).intersection(set(z))) / 5.0 for x, z in zip(ta, tb)]))

def prediction_geometry(base, candidate):
    base = np.asarray(base, np.float32)
    candidate = np.asarray(candidate, np.float32)
    ob, oc = stable_order(base), stable_order(candidate)
    changed = np.any(ob != oc, axis=1)
    same4 = np.all(np.sort(ob[:, :4], axis=1) == np.sort(oc[:, :4], axis=1), axis=1)
    same5 = np.all(np.sort(ob[:, :5], axis=1) == np.sort(oc[:, :5], axis=1), axis=1)
    tail_same = np.all(ob[:, 4:] == oc[:, 4:], axis=1)
    rank_shift = np.abs(rank_matrix(base) - rank_matrix(candidate))
    score_shift = np.abs(base - candidate)
    multiset_ok = True
    for row in np.flatnonzero(changed):
        old_values = np.sort(base[row, ob[row, :4]])
        new_values = np.sort(candidate[row, oc[row, :4]])
        if not np.array_equal(old_values, new_values):
            multiset_ok = False
            break
    return {
        'changed_users': int(changed.sum()),
        'top1_changes': int((ob[:, 0] != oc[:, 0]).sum()),
        'top2_changes': int((ob[:, 1] != oc[:, 1]).sum()),
        'top3_changes': int((ob[:, 2] != oc[:, 2]).sum()),
        'top4_changes': int((ob[:, 3] != oc[:, 3]).sum()),
        'top4_membership_changes': int((~same4).sum()),
        'top5_membership_changes': int((~same5).sum()),
        'rank5_to17_order_changes': int((~tail_same).sum()),
        'mean_abs_rank_shift_changed': float(rank_shift[changed].mean()) if changed.any() else 0.0,
        'max_rank_shift': int(rank_shift.max()) if len(rank_shift) else 0,
        'mean_abs_score_shift_changed': float(score_shift[changed].mean()) if changed.any() else 0.0,
        'max_abs_score_shift': float(score_shift.max()) if len(score_shift) else 0.0,
        'top4_score_multiset_ok': bool(multiset_ok),
    }

def policy_metrics(y_true, base, candidate):
    base_u = per_user_ndcg(y_true, base)
    cand_u = per_user_ndcg(y_true, candidate)
    delta = cand_u - base_u
    changed = np.any(stable_order(base) != stable_order(candidate), axis=1)
    benefit = int(np.sum(delta[changed] > 1e-12))
    harm = int(np.sum(delta[changed] < -1e-12))
    tie = int(np.sum(np.abs(delta[changed]) <= 1e-12))
    top1_base = np.argmax(base, axis=1) == np.argmax(y_true, axis=1)
    top1_cand = np.argmax(candidate, axis=1) == np.argmax(y_true, axis=1)
    return {
        'anchor_ndcg': float(base_u.mean()),
        'candidate_ndcg': float(cand_u.mean()),
        'delta': float(delta.mean()),
        'top1_delta': float(top1_cand.mean() - top1_base.mean()),
        'beneficial_actions': benefit,
        'harmful_actions': harm,
        'tied_actions': tie,
        'benefit_harm_ratio': float(benefit / max(1, harm)),
        **{f'geometry_{k}': v for k, v in prediction_geometry(base, candidate).items()},
    }, delta

def cluster_bootstrap_ci(values, groups, seed, bootstraps=BOOTSTRAPS):
    values = np.asarray(values, np.float64)
    groups = np.asarray(groups, dtype=object)
    unique = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(bootstraps), np.float64)
    for i in range(int(bootstraps)):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([members[g] for g in sampled])
        draws[i] = values[idx].mean() if len(idx) else 0.0
    return tuple(float(x) for x in np.quantile(draws, [0.025, 0.975]))

def id_block_metrics(global_rows, delta, blocks=ID_BLOCKS):
    global_rows = np.asarray(global_rows, int)
    delta = np.asarray(delta, np.float64)
    numbers = np.asarray([int(re.search(r'(\d+)$', TRAIN_IDS[r]).group(1)) for r in global_rows])
    chunks = np.array_split(np.argsort(numbers, kind='mergesort'), int(blocks))
    rows = []
    for block, idx in enumerate(chunks):
        if len(idx):
            rows.append({'block': block, 'n': len(idx), 'id_min': int(numbers[idx].min()),
                         'id_max': int(numbers[idx].max()), 'delta': float(delta[idx].mean())})
    return pd.DataFrame(rows)

_yt = np.asarray([[1.0, 0.7, 0.5] + [0.0] * 14], np.float32)
_perfect = _yt.copy()
_wrong = _yt[:, ::-1].copy()
assert abs(ndcg5(_yt, _perfect) - 1.0) < 1e-12
assert ndcg5(_yt, _wrong) < 1.0
assert prediction_geometry(_perfect, _perfect)['changed_users'] == 0
del _yt, _perfect, _wrong
print('Metric, hash, bootstrap, and geometry helpers are defined.')

# ---- cell 4 (markdown) ----
# ## Historical provenance registry
#
# The registry is deliberately conservative. A missing optional expert is excluded with a reason. A missing core anchor, V30 proxy, Consensus3 control, or prospective split prevents validation and triggers the exact-anchor fail close where possible.

# ---- cell 5 (code) ----
def candidate(name, version, family, oof_rel, oof_key, test_key, test_rel=None,
              oof_kind='npz', test_kind='npz', embedded_y=True, provenance='', warning=''):
    return {
        'name': name, 'version': version, 'family': family,
        'oof_rel': oof_rel, 'oof_key': oof_key, 'oof_kind': oof_kind,
        'test_rel': test_rel or oof_rel, 'test_key': test_key, 'test_kind': test_kind,
        'embedded_y': bool(embedded_y), 'provenance': provenance, 'warning': warning,
    }

CANDIDATE_SPECS = [
    candidate('v1_final_blend', 'v1', 'early_hybrid',
              'results/v1/minetoday_cache/oof_final_blend.npy', None, None,
              'results/v1/submission_ultimate_baseline.csv', 'npy', 'csv', False,
              'Five-fold ranker plus fixed sparse, semantic, reranker, and memory blend.',
              'Weak public transfer; retained only as a diversity candidate.'),
    candidate('v2_supervised_multimodal', 'v2', 'supervised_multimodal',
              'results/v2/minetoday_v2_cache/v2_20260819_r4_peft_fix_final_oof.npy', None, None,
              'results/v2/minetoday_v2_cache/v2_20260819_r4_peft_fix_final_test.npy', 'npy', 'npy', False,
              'Honest five-fold GTE, sparse, memory, assessment, and ranker ensemble.',
              'Large OOF-to-public transfer gap.'),
    candidate('v3_1_base', 'v3_1', 'sparse_calibrated',
              'results/v3_1/v31_oof_artifacts.npz', 'base_oof', 'base_test',
              provenance='Five-fold sparse word and character TF-IDF backbone.'),
    candidate('v3_1_meta', 'v3_1', 'sparse_calibrated',
              'results/v3_1/v31_oof_artifacts.npz', 'meta_oof', 'meta_test',
              provenance='Five-fold module-wise calibrated sparse surface.'),
    candidate('v3_1_primary', 'v3_1', 'sparse_calibrated',
              'results/v3_1/v31_oof_artifacts.npz', 'primary_oof', 'primary_test',
              provenance='Executed V3.1 structured sparse primary surface.'),
    candidate('v4_lambdarank', 'v4', 'structured_lambdarank',
              'results/v4/v4_oof_artifacts.npz', 'rank_oof', 'rank_test',
              provenance='Five-fold multi-expert LambdaRank predictions.'),
    candidate('v4_primary', 'v4', 'structured_lambdarank',
              'results/v4/v4_oof_artifacts.npz', 'primary_oof', 'primary_test',
              provenance='Executed V4 primary prediction surface.'),
    candidate('v5_yetirank', 'v5', 'generator_yetirank',
              'results/v5/v5_oof_artifacts.npz', 'rank_oof', 'rank_test',
              provenance='Five-fold generator-aware CatBoost YetiRank surface.'),
    candidate('v5_primary', 'v5', 'generator_yetirank',
              'results/v5/v5_oof_artifacts.npz', 'primary_oof', 'primary_test',
              provenance='Executed V5 primary surface.'),
    candidate('v6_anchor', 'v6', 'nested_rank_stack',
              'results/v6/v6_oof_artifacts.npz', 'anchor_oof', 'anchor_test',
              provenance='Outer-fold nested structured anchor.'),
    candidate('v6_lambdarank', 'v6', 'nested_rank_stack',
              'results/v6/v6_oof_artifacts.npz', 'rank_oof', 'rank_test',
              provenance='Outer-fold nested NDCG ranker.'),
    candidate('v6_ordinal', 'v6', 'nested_rank_stack',
              'results/v6/v6_oof_artifacts.npz', 'ordinal_oof', 'ordinal_test',
              provenance='Outer-fold nested ordinal specialist.'),
    candidate('v6_top1', 'v6', 'nested_rank_stack',
              'results/v6/v6_oof_artifacts.npz', 'top_oof', 'top_test',
              provenance='Outer-fold nested Top1 specialist.'),
    candidate('v6_primary', 'v6', 'nested_rank_stack',
              'results/v6/v6_oof_artifacts.npz', 'primary_oof', 'primary_test',
              provenance='Low-capacity fixed V6 blend of nested surfaces.'),
    candidate('v8_repeated_meta', 'v8', 'repeated_calibration',
              'results/v8/v8_oof_artifacts.npz', 'rep_meta_oof', 'rep_meta_test_full',
              provenance='Repeated-CV calibrated sparse probability surface.',
              warning='Strong OOF gain reversed publicly; selection must be marginal and nested.'),
    candidate('v8_primary', 'v8', 'repeated_calibration',
              'results/v8/v8_oof_artifacts.npz', 'primary_oof', 'primary_test',
              provenance='Executed repeated-calibration primary surface.',
              warning='Known validation-to-public reversal.'),
    candidate('v12_leader6', 'v12', 'module_calibration',
              'results/v12/v12_oof_artifacts.npz', 'leader6_oof', None,
              'results/v12/submission_v12_leader6.csv', 'npz', 'csv', True,
              provenance='Module-aware profile adjustment selected in V12.'),
    candidate('v12_1_micro4', 'v12_1', 'module_calibration',
              'results/v12_1/v12_1_oof_artifacts.npz', 'micro4_oof', 'micro4_test',
              provenance='Four-module V12.1 calibration extension.'),
    candidate('v12_2_consensus3', 'v12_2', 'module_calibration',
              'results/v12_2/v12_2_oof_artifacts.npz', 'consensus3_oof', 'consensus3_test',
              provenance='Double-validated Consensus3 surface and durable champion ancestor.'),
    candidate('v12_2_stable3', 'v12_2', 'module_calibration',
              'results/v12_2/v12_2_oof_artifacts.npz', 'stable3_oof', 'stable3_test',
              provenance='Lower-authority V12.2 stable calibration surface.'),
    candidate('v12_2_m15_only', 'v12_2', 'module_calibration',
              'results/v12_2/v12_2_oof_artifacts.npz', 'm15_only_oof', 'm15_only_test',
              provenance='Single-module V12.2 diagnostic calibration surface.'),
    candidate('v13_router', 'v13', 'utility_router',
              'results/v13/v13_oof_artifacts.npz', 'strict_router', 'router_test',
              provenance='Strict cross-fitted utility router from Consensus3 to V9.1.',
              warning='Closed after weak transfer; V13.1 is not used because it was not fully nested.'),
    candidate('v14_pairwise', 'v14', 'nested_pairwise',
              'results/v14/v14_oof_artifacts.npz', 'strict_pairwise', 'pairwise_test',
              provenance='Fully nested Top-6 pairwise surface.'),
    candidate('v16_semantic_meta', 'v16', 'qwen_semantic',
              'results/v16/v16_oof_artifacts.npz', 'sem_meta_oof', 'sem_meta_test',
              provenance='Five-fold Qwen8B semantic calibration head; encoder features are label-free.',
              warning='Global semantic authority showed split-sign reversal.'),
    candidate('v17_top5', 'v17', 'qwen_semantic',
              'results/v17/v17_oof_artifacts.npz', 'p5_oof', 'p5_test',
              provenance='Five-fold structural Top-5 probability view.'),
    candidate('v17_semantic_meta', 'v17', 'qwen_semantic',
              'results/v17/v17_oof_artifacts.npz', 'sem_meta_oof', 'sem_meta_test',
              provenance='Five-fold Top5-aware Qwen semantic calibration surface.',
              warning='Standalone semantic calibration underperformed; retained for complementarity only.'),
]

CORE_SPECS = {
    'v30_oof': ('results/v33/v33_stage1_oof_artifacts.npz', 'v30_train_oof'),
    'v30_test': ('results/v32/v32_test_score_artifacts.npz', 'v30'),
    'v30_submission': ('results/v30/submission_v30_primary.csv', None),
    'consensus_oof': ('results/v12_2/v12_2_oof_artifacts.npz', 'consensus3_oof'),
    'consensus_test': ('results/v12_2/v12_2_oof_artifacts.npz', 'consensus3_test'),
    'split_manifest': ('results/v48/v48/v48_split_manifest.csv', None),
}
ANCHOR_ALTERNATIVES = [
    'results/v48/v48/submission_v48_anchor_066028_reference.csv',
    'results/v48/v48/submission_v48_primary.csv',
    'results/v47/v47/submission_v47_primary.csv',
    'results/v46/v46/submission_v46_primary.csv',
    'results/v45/submission_v45_primary.csv',
    'results/v44/submission_v44_anchor_066028_reference.csv',
]

LINEAGE_AUDIT = pd.DataFrame([
    ('v1-v2', 'eligible paired experts', 'Executed cross-fitted OOF and matching test matrices exist.'),
    ('v3', 'not executed', 'Cross-encoder training ended with an FP16 gradient-unscale error.'),
    ('v3_1-v8', 'eligible subset', 'Direct full OOF/test pairs exist; V7 mapping is omitted and V8 carries a public-transfer warning.'),
    ('v9-v10_1', 'excluded', 'Candidate test arrays are absent or mapping from saved OOF key to CSV is ambiguous.'),
    ('v11', 'excluded', 'Candidate OOF counterpart is absent; stored anchor is redundant.'),
    ('v12-v12_2', 'eligible subset', 'Direct paired calibration surfaces exist.'),
    ('v13', 'eligible', 'Strict router OOF and matching test surface exist.'),
    ('v13_1', 'invalid for stacking', 'Second-level dependencies were not fully nested.'),
    ('v14', 'eligible', 'Fully nested OOF and matching pairwise test surface exist.'),
    ('v15', 'excluded duplicate', 'Policy selected zero actions and test surfaces collapse to Consensus3.'),
    ('v16-v17', 'eligible subset', 'Full cross-fitted structural and semantic calibration pairs exist.'),
    ('v18-v19', 'excluded candidate surfaces', 'Selected OOF attacks have no explicit matching full test matrices in their NPZ files.'),
    ('v20-v22', 'excluded partial validation', 'Candidate predictions cover DEV, CAL, or LOCK subsets rather than one full honest OOF/test pair.'),
    ('v23-v26', 'diagnostic or action-only', 'No honest full train analogue exists for public-adapted action packages.'),
    ('v27-v29_5', 'excluded', 'No saved full paired OOF/test prediction matrices for the deployed policies.'),
    ('v30', 'mandatory anchor proxy', 'V33 stores full V30 OOF and V32 stores the matching V30 test surface.'),
    ('v31-v32', 'excluded as experts', 'V31 and objective-diverse candidates have test surfaces but no matching full OOF matrix.'),
    ('v33-v34', 'diagnostic', 'V33 supplies V30 provenance; retrieval and BGE policy outputs are incomplete or zero-action.'),
    ('v35', 'invalid for efficacy', 'Initial TAPT crossed held-out text boundaries.'),
    ('v35_1-v40', 'excluded', 'Corrected branches failed gates or saved only partial/action surfaces.'),
    ('v41', 'invalid for efficacy', 'Nonnested stack reversed on untouched LOCK.'),
    ('v41_1-v45', 'excluded', 'Fail-close or action-only outputs lack honest full paired train predictions.'),
    ('v46-v47', 'not executed', 'Embedding failures prevented the scientific experiments.'),
    ('v48', 'anchor only', 'Semantic preflight failed before ensemble evaluation; LOCK remained unopened.'),
], columns=['versions', 'status', 'reason'])

print('Curated trainable prediction pairs:', len(CANDIDATE_SPECS))
display(LINEAGE_AUDIT)

# ---- cell 6 (code) ----
def scan_artifacts(root):
    rows = []
    result_root = root / 'results'
    if not result_root.exists():
        return pd.DataFrame(columns=['relative_path', 'bytes', 'suffix', 'version', 'prediction_hint'])
    skip_dirs = {'hf_cache', '__pycache__'}
    suffixes = {'.csv', '.json', '.npy', '.npz', '.parquet', '.pkl', '.pickle'}
    for current, dirs, files in os.walk(result_root):
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith('models--') and 'adapter' not in d.lower()]
        for filename in files:
            path = Path(current) / filename
            if path.suffix.lower() not in suffixes:
                continue
            rel = path.relative_to(root).as_posix()
            match = re.search(r'results/(v[^/]+)', rel)
            hint = bool(re.search(r'oof|pred|score|submission|artifact|action|ledger|manifest|metric', filename, re.I))
            rows.append({'relative_path': rel, 'bytes': int(path.stat().st_size),
                         'suffix': path.suffix.lower(), 'version': match.group(1) if match else '',
                         'prediction_hint': hint})
    return pd.DataFrame(rows).sort_values('relative_path').reset_index(drop=True)

ARTIFACT_INVENTORY = scan_artifacts(ARTIFACT_ROOT)
ARRAY_CACHE = {}

def load_np_key(relative, key=None):
    path = resolve_rel(relative, required=True)
    cache_key = (path.as_posix(), key)
    if cache_key in ARRAY_CACHE:
        return ARRAY_CACHE[cache_key]
    if path.suffix.lower() == '.npy':
        if key is not None:
            raise ValueError(f'NPY artifact cannot use key {key}: {relative}')
        arr = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == '.npz':
        if key is None:
            raise ValueError(f'NPZ artifact requires a key: {relative}')
        with np.load(path, allow_pickle=False) as archive:
            if key not in archive.files:
                raise KeyError(f'{key} missing from {relative}; keys={archive.files}')
            arr = archive[key]
    else:
        raise ValueError(f'Unsupported NumPy artifact: {relative}')
    arr = np.asarray(arr)
    ARRAY_CACHE[cache_key] = arr
    return arr

def load_csv_matrix(relative, expected_ids):
    path = resolve_rel(relative, required=True)
    frame = pd.read_csv(path)
    if 'user_id' not in frame.columns or not frame['user_id'].is_unique:
        raise ValueError(f'CSV needs one unique user_id per row: {relative}')
    missing = [m for m in MODULES if m not in frame.columns]
    if missing:
        raise ValueError(f'Module columns missing from {relative}: {missing}')
    frame['user_id'] = frame['user_id'].astype(str)
    got, expected = set(frame['user_id']), set(map(str, expected_ids))
    if got != expected:
        raise ValueError(f'user_id set mismatch in {relative}: got={len(got)} expected={len(expected)}')
    aligned = frame.set_index('user_id').loc[list(map(str, expected_ids)), MODULES]
    return aligned.to_numpy(np.float32)

def validate_matrix(arr, rows, label):
    arr = np.asarray(arr, np.float32)
    if arr.shape != (rows, len(MODULES)):
        raise ValueError(f'{label} shape {arr.shape}, expected {(rows, len(MODULES))}')
    if not np.isfinite(arr).all():
        raise ValueError(f'{label} contains NaN or Inf')
    return np.ascontiguousarray(arr)

def load_side(relative, kind, key, expected_ids, label):
    if kind == 'csv':
        arr = load_csv_matrix(relative, expected_ids)
    elif kind in {'npy', 'npz'}:
        arr = load_np_key(relative, key)
    else:
        raise ValueError(f'Unknown artifact kind {kind}')
    return validate_matrix(arr, len(expected_ids), label)

EXPERTS = {}
manifest_rows = []
for spec in CANDIDATE_SPECS:
    row = dict(spec)
    row.update({'loaded': False, 'eligible_for_meta': False, 'exclusion_reason': ''})
    try:
        oof = load_side(spec['oof_rel'], spec['oof_kind'], spec['oof_key'], TRAIN_IDS, spec['name'] + ' OOF')
        test_scores = load_side(spec['test_rel'], spec['test_kind'], spec['test_key'], TEST_IDS, spec['name'] + ' test')
        if spec['embedded_y']:
            stored_y = validate_matrix(load_np_key(spec['oof_rel'], 'y'), len(TRAIN_IDS), spec['name'] + ' stored y')
            if not np.allclose(stored_y, Y, rtol=0.0, atol=1e-7):
                raise ValueError('embedded target does not match organizer train row and module order')
        EXPERTS[spec['name']] = {'oof': oof, 'test': test_scores, 'spec': spec}
        row.update({
            'loaded': True, 'eligible_for_meta': True,
            'oof_shape': str(oof.shape), 'test_shape': str(test_scores.shape),
            'oof_sha256': matrix_sha256(oof), 'test_sha256': matrix_sha256(test_scores),
            'oof_rank_sha256': rank_sha256(oof), 'test_rank_sha256': rank_sha256(test_scores),
        })
    except Exception as exc:
        row['exclusion_reason'] = f'{type(exc).__name__}: {exc}'
    manifest_rows.append(row)

EXPERT_MANIFEST = pd.DataFrame(manifest_rows)

anchor_relative = next((rel for rel in ANCHOR_ALTERNATIVES if resolve_rel(rel) is not None), None)
if anchor_relative is None:
    raise FileNotFoundError('No exact 0.66028 champion reference is available, so V49 cannot fail close safely.')
ANCHOR_PATH = resolve_rel(anchor_relative, required=True)
ANCHOR_TEST = load_csv_matrix(anchor_relative, TEST_IDS)
ANCHOR_TEST = validate_matrix(ANCHOR_TEST, len(TEST_IDS), 'exact champion')
ANCHOR_TOP5_HASH = top5_sha256(ANCHOR_TEST)
ANCHOR_FILE_SHA256 = hashlib.sha256(ANCHOR_PATH.read_bytes()).hexdigest()
if ANCHOR_FILE_SHA256 != EXPECTED_ANCHOR_FILE_SHA256:
    raise RuntimeError(f'Champion file hash mismatch: {ANCHOR_FILE_SHA256}')
if ANCHOR_TOP5_HASH != EXPECTED_ANCHOR_TOP5_SHA256:
    raise RuntimeError(f'Champion Top-5 hash mismatch: {ANCHOR_TOP5_HASH}')

CORE_ERRORS = []
try:
    for relative, expected_hash in EXPECTED_CORE_FILE_SHA256.items():
        actual_hash = hashlib.sha256(resolve_rel(relative, required=True).read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f'Core artifact hash mismatch: {relative}')
    V30_OOF = validate_matrix(load_np_key(*CORE_SPECS['v30_oof']), len(TRAIN_IDS), 'V30 OOF proxy')
    V30_TEST = validate_matrix(load_np_key(*CORE_SPECS['v30_test']), len(TEST_IDS), 'V30 test')
    V30_TEST_CSV = load_csv_matrix(CORE_SPECS['v30_submission'][0], TEST_IDS)
    if not np.allclose(V30_TEST, V30_TEST_CSV, rtol=0.0, atol=1e-7):
        raise ValueError('V30 test matrix does not match its user-aligned submission')
    CONSENSUS_OOF = validate_matrix(load_np_key(*CORE_SPECS['consensus_oof']), len(TRAIN_IDS), 'Consensus3 OOF')
    CONSENSUS_TEST = validate_matrix(load_np_key(*CORE_SPECS['consensus_test']), len(TEST_IDS), 'Consensus3 test')
    consensus_y = validate_matrix(load_np_key('results/v12_2/v12_2_oof_artifacts.npz', 'y'), len(TRAIN_IDS), 'Consensus3 stored y')
    if not np.allclose(consensus_y, Y, rtol=0.0, atol=1e-7):
        raise ValueError('Consensus3 OOF user or module order differs from organizer train')
    stage1_fold = np.asarray(load_np_key('results/v33/v33_stage1_oof_artifacts.npz', 'stage1_fold'))
    if stage1_fold.shape != (len(TRAIN_IDS),) or np.any(stage1_fold < 0) or len(np.unique(stage1_fold)) < 3:
        raise ValueError('V30 OOF fold provenance is incomplete')
except Exception as exc:
    CORE_ERRORS.append(f'core prediction error: {type(exc).__name__}: {exc}')
    V30_OOF = V30_TEST = CONSENSUS_OOF = CONSENSUS_TEST = None

try:
    split_path = resolve_rel(CORE_SPECS['split_manifest'][0], required=True)
    split_frame = pd.read_csv(split_path)
    required_split_cols = {'user_id', 'row', 'template_group', 'split'}
    if not required_split_cols.issubset(split_frame.columns) or not split_frame['user_id'].is_unique:
        raise ValueError('V48 split manifest schema or user uniqueness is invalid')
    split_frame['user_id'] = split_frame['user_id'].astype(str)
    if set(split_frame['user_id']) != set(TRAIN_IDS):
        raise ValueError('V48 split manifest user IDs do not match organizer train')
    split_frame = split_frame.set_index('user_id').loc[TRAIN_IDS].reset_index()
    if not np.array_equal(split_frame['row'].to_numpy(int), np.arange(len(TRAIN_IDS))):
        raise ValueError('V48 split manifest row order differs from organizer train')
    observed_counts = split_frame['split'].value_counts().to_dict()
    if any(observed_counts.get(k) != v for k, v in EXPECTED_SPLIT_COUNTS.items()):
        raise ValueError(f'V48 split counts changed: {observed_counts}')
    DEV_IDX = np.flatnonzero(split_frame['split'].isin(['ACTION_TRAIN', 'DESIGN']).to_numpy())
    LOCK_IDX = np.flatnonzero((split_frame['split'] == 'LOCK').to_numpy())
    TEMPLATE_GROUP = split_frame['template_group'].fillna('no_chat').astype(str).to_numpy()
    if len(set(DEV_IDX).intersection(set(LOCK_IDX))) or len(DEV_IDX) + len(LOCK_IDX) != len(TRAIN_IDS):
        raise ValueError('DEVELOPMENT and LOCK do not form a disjoint train partition')
    if set(TEMPLATE_GROUP[DEV_IDX]).intersection(set(TEMPLATE_GROUP[LOCK_IDX])):
        raise ValueError('Template groups cross the DEVELOPMENT and LOCK boundary')
except Exception as exc:
    CORE_ERRORS.append(f'split manifest error: {type(exc).__name__}: {exc}')
    split_frame = pd.DataFrame()
    DEV_IDX = np.asarray([], int)
    LOCK_IDX = np.asarray([], int)
    TEMPLATE_GROUP = np.asarray([], dtype=object)

CORE_READY = not CORE_ERRORS
print('Artifact files inventoried:', len(ARTIFACT_INVENTORY))
print('Eligible paired experts loaded:', len(EXPERTS), 'of', len(CANDIDATE_SPECS))
print('Anchor source:', anchor_relative, 'Top-5 hash:', ANCHOR_TOP5_HASH)
print('Core ready:', CORE_READY, CORE_ERRORS)
display(EXPERT_MANIFEST[['name', 'version', 'family', 'loaded', 'warning', 'exclusion_reason']])

# ---- cell 7 (markdown) ----
# ## Deduplication and protected rank-space policy
#
# Exact duplicates are identified by canonical float32 score and ranking hashes. Near duplicates require DEVELOPMENT rank correlation of at least 0.995, Top1 agreement of at least 0.99, and mean Top-5 Jaccard of at least 0.99. Near-duplicate analysis is label-free and never uses test predictions.

# ---- cell 8 (code) ----
class UnionFind:
    def __init__(self, names):
        self.parent = {name: name for name in names}

    def find(self, name):
        while self.parent[name] != name:
            self.parent[name] = self.parent[self.parent[name]]
            name = self.parent[name]
        return name

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            keep, drop = sorted([ra, rb])
            self.parent[drop] = keep

DIVERSITY_PAIRS = pd.DataFrame()
DUPLICATE_REPORT = pd.DataFrame()
RANK_CORRELATION = pd.DataFrame()
EXPERT_METRICS = pd.DataFrame()
NEAR_CLUSTER = {}
DUPLICATE_OF_ANCHOR = {}
ANCHOR_SIMILARITY = {}
TRAIN_ELIGIBLE = np.zeros(len(TRAIN_IDS), bool)
TEST_ELIGIBLE = np.zeros(len(TEST_IDS), bool)

if CORE_READY and len(EXPERTS):
    expert_names = sorted(EXPERTS)
    uf = UnionFind(expert_names)
    anchor_ranks = rank_matrix(V30_OOF[DEV_IDX]).reshape(-1).astype(np.float32)
    anchor_top1 = np.argmax(V30_OOF[DEV_IDX], axis=1)
    dev_ranks = {name: rank_matrix(EXPERTS[name]['oof'][DEV_IDX]).reshape(-1).astype(np.float32) for name in expert_names}
    dev_top1 = {name: np.argmax(EXPERTS[name]['oof'][DEV_IDX], axis=1) for name in expert_names}
    score_hashes = {name: (matrix_sha256(EXPERTS[name]['oof']), matrix_sha256(EXPERTS[name]['test'])) for name in expert_names}
    rank_hashes = {name: (rank_sha256(EXPERTS[name]['oof']), rank_sha256(EXPERTS[name]['test'])) for name in expert_names}
    anchor_score_hashes = (matrix_sha256(V30_OOF), matrix_sha256(V30_TEST))
    anchor_rank_hashes = (rank_sha256(V30_OOF), rank_sha256(V30_TEST))
    pair_rows = []
    for i, left in enumerate(expert_names):
        for right in expert_names[i + 1:]:
            corr = float(np.corrcoef(dev_ranks[left], dev_ranks[right])[0, 1])
            top1_agreement = float(np.mean(dev_top1[left] == dev_top1[right]))
            top5_jaccard = mean_top5_jaccard(EXPERTS[left]['oof'][DEV_IDX], EXPERTS[right]['oof'][DEV_IDX])
            exact_score = score_hashes[left] == score_hashes[right]
            exact_rank = rank_hashes[left] == rank_hashes[right]
            near = bool(corr >= NEAR_RANK_CORR and top1_agreement >= NEAR_TOP1_AGREEMENT and top5_jaccard >= NEAR_TOP5_JACCARD)
            if exact_score or exact_rank or near:
                uf.union(left, right)
            kind = 'exact_score' if exact_score else ('exact_rank' if exact_rank else ('near_rank' if near else 'distinct'))
            pair_rows.append({'left': left, 'right': right, 'rank_correlation': corr,
                              'top1_agreement': top1_agreement, 'top5_jaccard': top5_jaccard,
                              'exact_score_pair': exact_score, 'exact_rank_pair': exact_rank,
                              'near_duplicate': near, 'duplicate_kind': kind})
    for name in expert_names:
        corr = float(np.corrcoef(anchor_ranks, dev_ranks[name])[0, 1])
        top1_agreement = float(np.mean(anchor_top1 == dev_top1[name]))
        top5_jaccard = mean_top5_jaccard(V30_OOF[DEV_IDX], EXPERTS[name]['oof'][DEV_IDX])
        exact_score = anchor_score_hashes == score_hashes[name]
        exact_rank = anchor_rank_hashes == rank_hashes[name]
        near = bool(corr >= NEAR_RANK_CORR and top1_agreement >= NEAR_TOP1_AGREEMENT and
                    top5_jaccard >= NEAR_TOP5_JACCARD)
        DUPLICATE_OF_ANCHOR[name] = bool(exact_score or exact_rank or near)
        ANCHOR_SIMILARITY[name] = (corr, top1_agreement, top5_jaccard)
        kind = 'anchor_exact_score' if exact_score else ('anchor_exact_rank' if exact_rank else ('anchor_near_rank' if near else 'distinct'))
        pair_rows.append({'left': 'anchor_v30', 'right': name, 'rank_correlation': corr,
                          'top1_agreement': top1_agreement, 'top5_jaccard': top5_jaccard,
                          'exact_score_pair': exact_score, 'exact_rank_pair': exact_rank,
                          'near_duplicate': near, 'duplicate_kind': kind})
    DIVERSITY_PAIRS = pd.DataFrame(pair_rows)
    DUPLICATE_REPORT = DIVERSITY_PAIRS[DIVERSITY_PAIRS['duplicate_kind'] != 'distinct'].reset_index(drop=True)
    roots = {name: uf.find(name) for name in expert_names}
    root_order = {root: i for i, root in enumerate(sorted(set(roots.values())))}
    NEAR_CLUSTER = {name: int(root_order[roots[name]]) for name in expert_names}
    corr_names = ['anchor_v30'] + expert_names
    corr_vectors = [anchor_ranks] + [dev_ranks[n] for n in expert_names]
    RANK_CORRELATION = pd.DataFrame(np.corrcoef(np.vstack(corr_vectors)), index=corr_names, columns=corr_names)

    TRAIN_ELIGIBLE = same_top_order(V30_OOF, CONSENSUS_OOF, 5)
    TEST_ELIGIBLE = same_top_order(ANCHOR_TEST, V30_TEST, 5) & same_top_order(V30_TEST, CONSENSUS_TEST, 5)

    anchor_dev_score = ndcg5(Y[DEV_IDX], V30_OOF[DEV_IDX])
    metric_rows = [{
        'expert': 'anchor_v30', 'version': 'v30', 'family': 'anchor_proxy',
        'development_ndcg': anchor_dev_score, 'delta_vs_anchor': 0.0,
        'rank_correlation_with_anchor': 1.0, 'top1_agreement_with_anchor': 1.0,
        'top5_jaccard_with_anchor': 1.0,
        'near_cluster': -1, 'duplicate_of_anchor': True,
    }]
    for name in expert_names:
        surface = EXPERTS[name]['oof'][DEV_IDX]
        score = ndcg5(Y[DEV_IDX], surface)
        spec = EXPERTS[name]['spec']
        metric_rows.append({
            'expert': name, 'version': spec['version'], 'family': spec['family'],
            'development_ndcg': score, 'delta_vs_anchor': score - anchor_dev_score,
            'rank_correlation_with_anchor': ANCHOR_SIMILARITY[name][0],
            'top1_agreement_with_anchor': ANCHOR_SIMILARITY[name][1],
            'top5_jaccard_with_anchor': ANCHOR_SIMILARITY[name][2],
            'near_cluster': NEAR_CLUSTER[name], 'duplicate_of_anchor': DUPLICATE_OF_ANCHOR[name],
        })
    EXPERT_METRICS = pd.DataFrame(metric_rows).sort_values('development_ndcg', ascending=False).reset_index(drop=True)

EXPERT_MANIFEST['near_cluster'] = EXPERT_MANIFEST['name'].map(NEAR_CLUSTER)
EXPERT_MANIFEST['anchor_rank_correlation'] = EXPERT_MANIFEST['name'].map({n: v[0] for n, v in ANCHOR_SIMILARITY.items()})
EXPERT_MANIFEST['duplicate_of_anchor'] = EXPERT_MANIFEST['name'].map(DUPLICATE_OF_ANCHOR).fillna(False)
loaded_families = sorted({v['spec']['family'] for v in EXPERTS.values() if not DUPLICATE_OF_ANCHOR.get(v['spec']['name'], False)})
VALIDATION_READY = bool(CORE_READY and len(loaded_families) >= MIN_LOADED_FAMILIES and int(TRAIN_ELIGIBLE[DEV_IDX].sum()) >= MIN_ELIGIBLE_DEVELOPMENT)

print('Near-duplicate clusters:', len(set(NEAR_CLUSTER.values())) if NEAR_CLUSTER else 0)
print('Loaded non-anchor families:', loaded_families)
print('Geometry-derived eligible rows, DEVELOPMENT/test:', int(TRAIN_ELIGIBLE[DEV_IDX].sum()) if CORE_READY else 0, int(TEST_ELIGIBLE.sum()))
print('Validation ready:', VALIDATION_READY)
display(EXPERT_METRICS)

# ---- cell 9 (code) ----
POSITION_DISCOUNT4 = 1.0 / np.log2(np.arange(2, 6, dtype=np.float32))
ANCHOR_LOCAL_VOTE = np.asarray([3.0, 2.0, 1.0, 0.0], np.float32)

def local_rank_votes(anchor, surface):
    top4 = stable_order(anchor)[:, :4]
    values = np.take_along_axis(np.asarray(surface, np.float32), top4, axis=1)
    order = np.argsort(-values, axis=1, kind='mergesort')
    ranks = np.empty_like(order, dtype=np.int8)
    ranks[np.arange(len(order))[:, None], order] = np.arange(4, dtype=np.int8)
    return top4, (3 - ranks).astype(np.float32)

def classify_head_action(local_order):
    key = tuple(map(int, local_order))
    if key[0] != 0:
        return 'TOP1'
    if key == (0, 2, 1, 3):
        return 'B23'
    if key == (0, 1, 3, 2):
        return 'B34'
    return 'COMPLEX_HEAD'

def empty_policy():
    return {'experts': [], 'families': [], 'mode': 'preserve_top1', 'coverage': 0.0,
            'anchor_prior_votes': ANCHOR_PRIOR_VOTES, 'meta_delta': 0.0, 'meta_top1_delta': 0.0}

def apply_policy(anchor, expert_surfaces, eligible, policy):
    anchor = np.asarray(anchor, np.float32)
    eligible = np.asarray(eligible, bool)
    selected_names = list(policy.get('experts', []))
    if not selected_names or float(policy.get('coverage', 0.0)) <= 0:
        return anchor.copy(), pd.DataFrame(), np.zeros(len(anchor), np.float32)
    top4 = stable_order(anchor)[:, :4]
    aggregate = np.tile(ANCHOR_PRIOR_VOTES * ANCHOR_LOCAL_VOTE, (len(anchor), 1)).astype(np.float32)
    for name in selected_names:
        if name not in expert_surfaces:
            raise KeyError(f'Selected expert missing at prediction time: {name}')
        expert_top4, vote = local_rank_votes(anchor, expert_surfaces[name])
        if not np.array_equal(expert_top4, top4):
            raise AssertionError('Anchor Top-4 alignment changed during voting')
        aggregate += vote
    if policy['mode'] == 'head4':
        local_order = np.argsort(-aggregate, axis=1, kind='mergesort')
    elif policy['mode'] == 'preserve_top1':
        rest = 1 + np.argsort(-aggregate[:, 1:], axis=1, kind='mergesort')
        local_order = np.concatenate([np.zeros((len(anchor), 1), np.int64), rest], axis=1)
    else:
        raise ValueError(f'Unknown authority mode: {policy["mode"]}')
    identity = np.tile(np.arange(4), (len(anchor), 1))
    proposal = eligible & np.any(local_order != identity, axis=1)
    best_value = (np.take_along_axis(aggregate, local_order, axis=1) * POSITION_DISCOUNT4).sum(axis=1)
    anchor_value = (aggregate * POSITION_DISCOUNT4).sum(axis=1)
    vote_gain = best_value - anchor_value
    anchor_values = np.take_along_axis(anchor, top4, axis=1)
    inverse_order = np.empty_like(local_order)
    inverse_order[np.arange(len(anchor))[:, None], local_order] = np.arange(4)
    reassigned_values = np.take_along_axis(anchor_values, inverse_order, axis=1)
    normalized_shift = np.abs(reassigned_values - anchor_values).sum(axis=1) / np.maximum(np.ptp(anchor_values, axis=1), 1e-7)
    confidence = vote_gain - 1e-4 * normalized_shift
    proposal &= np.isfinite(confidence) & (vote_gain > 1e-9)
    proposed_rows = np.flatnonzero(proposal)
    take = min(len(proposed_rows), int(math.ceil(float(policy['coverage']) * max(1, int(eligible.sum())))))
    chosen = np.asarray([], int)
    if take:
        chosen = proposed_rows[np.argsort(-confidence[proposed_rows], kind='mergesort')[:take]]
    candidate_scores = anchor.copy()
    ledger = []
    for row in chosen:
        old_modules = top4[row].copy()
        old_values = anchor[row, old_modules].copy()
        order = local_order[row].copy()
        new_modules = old_modules[order]
        candidate_scores[row, new_modules] = old_values
        ledger.append({
            'row_local': int(row), 'confidence': float(confidence[row]),
            'vote_gain': float(vote_gain[row]), 'normalized_anchor_shift': float(normalized_shift[row]),
            'action_type': classify_head_action(order),
            'old_top4': '|'.join(MODULES[m] for m in old_modules),
            'new_top4': '|'.join(MODULES[m] for m in new_modules),
            'old_top1': MODULES[int(old_modules[0])], 'new_top1': MODULES[int(new_modules[0])],
        })
    return candidate_scores, pd.DataFrame(ledger), confidence.astype(np.float32)

def _policy_key(metrics, policy):
    safe_mode = int(policy['mode'] == 'preserve_top1')
    return (float(metrics['delta']), float(metrics['top1_delta']),
            float(metrics['benefit_harm_ratio']), safe_mode, -float(policy['coverage']),
            -len(policy['experts']))

def family_representatives(global_rows):
    global_rows = np.asarray(global_rows, int)
    anchor_score = ndcg5(Y[global_rows], V30_OOF[global_rows])
    quality_rows = []
    for name, payload in EXPERTS.items():
        score = ndcg5(Y[global_rows], payload['oof'][global_rows])
        quality_rows.append({
            'expert': name, 'family': payload['spec']['family'], 'near_cluster': NEAR_CLUSTER.get(name, -1),
            'ndcg': score, 'delta_vs_anchor': score - anchor_score,
            'duplicate_of_anchor': DUPLICATE_OF_ANCHOR.get(name, False),
            'quality_pass': bool(score >= anchor_score - QUALITY_FLOOR_BELOW_ANCHOR),
        })
    quality = pd.DataFrame(quality_rows)
    if quality.empty:
        return [], quality
    eligible = quality[quality['quality_pass'] & ~quality['duplicate_of_anchor']].copy()
    representatives = []
    used_clusters = set()
    family_order = eligible.groupby('family')['ndcg'].max().sort_values(ascending=False, kind='mergesort').index
    for family in family_order:
        options = eligible[eligible['family'] == family].sort_values(['ndcg', 'expert'], ascending=[False, True])
        for row in options.itertuples(index=False):
            cluster = int(row.near_cluster)
            if cluster not in used_clusters:
                used_clusters.add(cluster)
                representatives.append(str(row.expert))
                break
    quality['family_representative'] = quality['expert'].isin(representatives)
    return representatives, quality

def evaluate_policy_fit(global_rows, experts, mode, coverage):
    global_rows = np.asarray(global_rows, int)
    policy = {
        'experts': list(experts),
        'families': [EXPERTS[n]['spec']['family'] for n in experts],
        'mode': mode, 'coverage': float(coverage), 'anchor_prior_votes': ANCHOR_PRIOR_VOTES,
    }
    surfaces = {n: EXPERTS[n]['oof'][global_rows] for n in experts}
    candidate_scores, ledger, _ = apply_policy(V30_OOF[global_rows], surfaces, TRAIN_ELIGIBLE[global_rows], policy)
    metrics, _ = policy_metrics(Y[global_rows], V30_OOF[global_rows], candidate_scores)
    policy.update({'meta_delta': metrics['delta'], 'meta_top1_delta': metrics['top1_delta']})
    return policy, metrics, ledger

def fit_policy(global_rows, context):
    representatives, quality = family_representatives(global_rows)
    traces = []
    prefixes = [(empty_policy(), {'delta': 0.0, 'top1_delta': 0.0, 'benefit_harm_ratio': 0.0})]
    best_single = empty_policy()
    best_single_metrics = {'delta': 0.0, 'top1_delta': 0.0, 'benefit_harm_ratio': 0.0}
    selected = []
    for step in range(1, min(MAX_SELECTED_FAMILIES, len(representatives)) + 1):
        best_trial = None
        for name in representatives:
            if name in selected:
                continue
            trial_members = selected + [name]
            for mode in AUTHORITY_MODES:
                for coverage in COVERAGE_GRID:
                    policy, metrics, _ = evaluate_policy_fit(global_rows, trial_members, mode, coverage)
                    record = {
                        'context': context, 'step': step, 'selected_before': '|'.join(selected),
                        'candidate_added': name, 'trial_experts': '|'.join(trial_members),
                        'trial_families': '|'.join(policy['families']), 'mode': mode, 'coverage': coverage,
                        **metrics,
                    }
                    traces.append(record)
                    key = _policy_key(metrics, policy)
                    if step == 1 and key > _policy_key(best_single_metrics, best_single):
                        best_single, best_single_metrics = policy, metrics
                    if best_trial is None or key > best_trial[0]:
                        best_trial = (key, policy, metrics)
        if best_trial is None:
            break
        _, chosen_policy, chosen_metrics = best_trial
        selected = list(chosen_policy['experts'])
        prefixes.append((chosen_policy, chosen_metrics))
    best_policy, best_metrics = max(prefixes, key=lambda item: _policy_key(item[1], item[0]))
    best_policy = dict(best_policy)
    best_policy.update({'meta_delta': float(best_metrics['delta']), 'meta_top1_delta': float(best_metrics['top1_delta'])})
    return best_policy, best_single, pd.DataFrame(traces), quality

def apply_oof_policy(global_rows, policy):
    global_rows = np.asarray(global_rows, int)
    surfaces = {name: EXPERTS[name]['oof'][global_rows] for name in policy.get('experts', [])}
    return apply_policy(V30_OOF[global_rows], surfaces, TRAIN_ELIGIBLE[global_rows], policy)

_demo_anchor = np.tile(np.arange(len(MODULES), 0, -1, dtype=np.float32), (2, 1))
_demo_expert = _demo_anchor.copy()
_demo_expert[:, [1, 2]] = _demo_expert[:, [2, 1]]
_demo_policy = {'experts': ['a', 'b', 'c'], 'mode': 'preserve_top1', 'coverage': 1.0}
_demo_scores, _demo_ledger, _ = apply_policy(
    _demo_anchor, {name: _demo_expert for name in _demo_policy['experts']}, np.ones(2, bool), _demo_policy
)
_demo_geometry = prediction_geometry(_demo_anchor, _demo_scores)
assert len(_demo_ledger) == 2 and _demo_geometry['top1_changes'] == 0
assert _demo_geometry['top5_membership_changes'] == 0 and _demo_geometry['rank5_to17_order_changes'] == 0
del _demo_anchor, _demo_expert, _demo_policy, _demo_scores, _demo_ledger, _demo_geometry

print('Rank-vote and family-balanced forward-selection functions are defined.')

# ---- cell 10 (markdown) ----
# ## Fully nested DEVELOPMENT evaluation
#
# Each outer fold freezes family representatives, ensemble members, prefix length, authority mode, and coverage before scoring its held-out template groups. The aggregate below is the honest meta-ensemble DEVELOPMENT estimate.

# ---- cell 11 (code) ----
def policy_row(policy, context):
    return {
        'context': context,
        'experts': '|'.join(policy.get('experts', [])),
        'families': '|'.join(policy.get('families', [])),
        'expert_count': len(policy.get('experts', [])),
        'mode': policy.get('mode', 'preserve_top1'),
        'coverage': float(policy.get('coverage', 0.0)),
        'anchor_prior_votes': int(policy.get('anchor_prior_votes', ANCHOR_PRIOR_VOTES)),
        'meta_delta': float(policy.get('meta_delta', 0.0)),
        'meta_top1_delta': float(policy.get('meta_top1_delta', 0.0)),
    }

DEV_ERROR = ''
DEV_GATE = False
DEV_GATE_REASONS = ['development_not_evaluated']
FINAL_POLICY = empty_policy()
FINAL_SINGLE_POLICY = empty_policy()
DEV_CANDIDATE_OOF = V30_OOF[DEV_IDX].copy() if CORE_READY else np.empty((0, len(MODULES)), np.float32)
DEV_SINGLE_OOF = DEV_CANDIDATE_OOF.copy()
DEV_DELTA_VALUES = np.asarray([], np.float64)
DEV_METRICS = pd.DataFrame()
OUTER_METRICS = pd.DataFrame()
OUTER_POLICIES = pd.DataFrame()
SELECTION_TRACE = pd.DataFrame()
QUALITY_AUDIT = pd.DataFrame()
DEV_ID_BLOCKS = pd.DataFrame()
DEV_TEMPLATE_STABILITY = pd.DataFrame()
DEV_ACTION_LEDGER = pd.DataFrame()
FINAL_SELECTED = pd.DataFrame()
LEAVE_ONE_FAMILY_OUT = pd.DataFrame()

if VALIDATION_READY:
    try:
        dev_groups = TEMPLATE_GROUP[DEV_IDX]
        dev_strata = np.argmax(Y[DEV_IDX], axis=1)
        splitter = StratifiedGroupKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED + 17)
        assigned = np.zeros(len(DEV_IDX), bool)
        outer_metric_rows, outer_policy_rows = [], []
        trace_frames, quality_frames, ledger_frames = [], [], []
        outer_family_sets = []

        for fold, (fit_pos, val_pos) in enumerate(splitter.split(np.zeros(len(DEV_IDX)), dev_strata, dev_groups)):
            fit_rows, val_rows = DEV_IDX[fit_pos], DEV_IDX[val_pos]
            if set(TEMPLATE_GROUP[fit_rows]).intersection(set(TEMPLATE_GROUP[val_rows])):
                raise AssertionError(f'Template-group leakage in outer fold {fold}')
            policy, single_policy, trace, quality = fit_policy(fit_rows, f'outer_{fold}')
            val_scores, ledger, _ = apply_oof_policy(val_rows, policy)
            single_scores, _, _ = apply_oof_policy(val_rows, single_policy)
            metrics, _ = policy_metrics(Y[val_rows], V30_OOF[val_rows], val_scores)
            single_metrics, _ = policy_metrics(Y[val_rows], V30_OOF[val_rows], single_scores)
            DEV_CANDIDATE_OOF[val_pos] = val_scores
            DEV_SINGLE_OOF[val_pos] = single_scores
            assigned[val_pos] = True

            outer_metric_rows.append({
                'outer_fold': fold, 'fit_rows': len(fit_rows), 'validation_rows': len(val_rows),
                'fit_template_groups': len(np.unique(TEMPLATE_GROUP[fit_rows])),
                'validation_template_groups': len(np.unique(TEMPLATE_GROUP[val_rows])),
                'anchor_ndcg': metrics['anchor_ndcg'], 'candidate_ndcg': metrics['candidate_ndcg'],
                'delta': metrics['delta'], 'best_single_ndcg': single_metrics['candidate_ndcg'],
                'edge_vs_best_single': metrics['candidate_ndcg'] - single_metrics['candidate_ndcg'],
                'top1_delta': metrics['top1_delta'],
                'beneficial_actions': metrics['beneficial_actions'],
                'harmful_actions': metrics['harmful_actions'],
                'changed_users': metrics['geometry_changed_users'],
            })
            outer_policy_rows.append({
                'outer_fold': fold, **policy_row(policy, f'outer_{fold}'),
                'best_single_experts': '|'.join(single_policy.get('experts', [])),
            })
            outer_family_sets.append(set(policy.get('families', [])))
            if not trace.empty:
                trace_frames.append(trace.assign(outer_fold=fold))
            if not quality.empty:
                quality_frames.append(quality.assign(context=f'outer_{fold}', outer_fold=fold))
            if not ledger.empty:
                ledger = ledger.copy()
                ledger['row_global'] = val_rows[ledger['row_local'].to_numpy(int)]
                ledger['user_id'] = TRAIN_IDS[ledger['row_global'].to_numpy(int)]
                ledger['outer_fold'] = fold
                ledger_frames.append(ledger)

        if not assigned.all():
            raise AssertionError('Outer evaluation did not assign every DEVELOPMENT row exactly once')

        OUTER_METRICS = pd.DataFrame(outer_metric_rows)
        OUTER_POLICIES = pd.DataFrame(outer_policy_rows)
        DEV_ACTION_LEDGER = pd.concat(ledger_frames, ignore_index=True) if ledger_frames else pd.DataFrame()
        FINAL_POLICY, FINAL_SINGLE_POLICY, final_trace, final_quality = fit_policy(DEV_IDX, 'final_development')
        if not final_trace.empty:
            trace_frames.append(final_trace.assign(outer_fold=-1))
        if not final_quality.empty:
            quality_frames.append(final_quality.assign(context='final_development', outer_fold=-1))
        SELECTION_TRACE = pd.concat(trace_frames, ignore_index=True) if trace_frames else pd.DataFrame()
        QUALITY_AUDIT = pd.concat(quality_frames, ignore_index=True) if quality_frames else pd.DataFrame()

        dev_metrics, DEV_DELTA_VALUES = policy_metrics(Y[DEV_IDX], V30_OOF[DEV_IDX], DEV_CANDIDATE_OOF)
        single_metrics, _ = policy_metrics(Y[DEV_IDX], V30_OOF[DEV_IDX], DEV_SINGLE_OOF)
        best_standalone = EXPERT_METRICS[EXPERT_METRICS['expert'] != 'anchor_v30'].iloc[0]
        dev_ci = cluster_bootstrap_ci(DEV_DELTA_VALUES, dev_groups, SEED + 201)
        DEV_ID_BLOCKS = id_block_metrics(DEV_IDX, DEV_DELTA_VALUES)
        DEV_TEMPLATE_STABILITY = (
            pd.DataFrame({'template_group': dev_groups, 'delta': DEV_DELTA_VALUES})
            .groupby('template_group', as_index=False)
            .agg(n=('delta', 'size'), delta=('delta', 'mean'))
            .sort_values(['n', 'template_group'], ascending=[False, True])
            .reset_index(drop=True)
        )
        DEV_TEMPLATE_STABILITY['positive'] = DEV_TEMPLATE_STABILITY['delta'] > 0

        final_fit_scores, _, _ = apply_oof_policy(DEV_IDX, FINAL_POLICY)
        final_fit_metrics, _ = policy_metrics(Y[DEV_IDX], V30_OOF[DEV_IDX], final_fit_scores)
        final_families = set(FINAL_POLICY.get('families', []))
        family_jaccards = [
            len(final_families.intersection(s)) / max(1, len(final_families.union(s)))
            for s in outer_family_sets
        ]
        family_stability = float(np.mean(family_jaccards)) if family_jaccards else 0.0

        expert_frequency = Counter(
            expert for policy in OUTER_POLICIES.get('experts', pd.Series(dtype=str))
            for expert in str(policy).split('|') if expert
        )
        denominator = ANCHOR_PRIOR_VOTES + len(FINAL_POLICY.get('experts', []))
        selected_rows = [{
            'expert': 'anchor_prior', 'family': 'champion_anchor', 'version': 'v30_oof__v48_test',
            'train_surface': 'v30_train_oof', 'test_surface': anchor_relative,
            'participation_weight': ANCHOR_PRIOR_VOTES / max(1, denominator),
            'outer_fold_selection_count': OUTER_FOLDS,
        }]
        for name in FINAL_POLICY.get('experts', []):
            selected_rows.append({
                'expert': name, 'family': EXPERTS[name]['spec']['family'],
                'version': EXPERTS[name]['spec']['version'],
                'train_surface': EXPERTS[name]['spec']['oof_rel'],
                'test_surface': EXPERTS[name]['spec']['test_rel'],
                'participation_weight': 1.0 / max(1, denominator),
                'outer_fold_selection_count': int(expert_frequency[name]),
            })
        FINAL_SELECTED = pd.DataFrame(selected_rows)

        attribution_rows = []
        for name in FINAL_POLICY.get('experts', []):
            reduced = dict(FINAL_POLICY)
            reduced['experts'] = [n for n in FINAL_POLICY['experts'] if n != name]
            reduced['families'] = [EXPERTS[n]['spec']['family'] for n in reduced['experts']]
            reduced_scores, _, _ = apply_oof_policy(DEV_IDX, reduced)
            reduced_metrics, _ = policy_metrics(Y[DEV_IDX], V30_OOF[DEV_IDX], reduced_scores)
            attribution_rows.append({
                'removed_expert': name, 'removed_family': EXPERTS[name]['spec']['family'],
                'frozen_full_fit_ndcg': final_fit_metrics['candidate_ndcg'],
                'ndcg_without': reduced_metrics['candidate_ndcg'],
                'loss_from_removal': final_fit_metrics['candidate_ndcg'] - reduced_metrics['candidate_ndcg'],
                'diagnostic_only': True,
            })
        LEAVE_ONE_FAMILY_OUT = pd.DataFrame(attribution_rows)

        dev_metrics.update({
            'bootstrap_ci_low': dev_ci[0], 'bootstrap_ci_high': dev_ci[1],
            'best_single_nested_ndcg': single_metrics['candidate_ndcg'],
            'edge_vs_best_single': dev_metrics['candidate_ndcg'] - single_metrics['candidate_ndcg'],
            'best_standalone_expert': best_standalone['expert'],
            'best_standalone_expert_ndcg': float(best_standalone['development_ndcg']),
            'delta_vs_best_standalone_expert': dev_metrics['candidate_ndcg'] - float(best_standalone['development_ndcg']),
            'outer_fold_wins': int((OUTER_METRICS['delta'] > 0).sum()),
            'outer_fold_count': len(OUTER_METRICS),
            'worst_outer_fold_delta': float(OUTER_METRICS['delta'].min()),
            'id_block_wins': int((DEV_ID_BLOCKS['delta'] > 0).sum()),
            'id_block_count': len(DEV_ID_BLOCKS),
            'family_stability_jaccard': family_stability,
            'eligible_rows': int(TRAIN_ELIGIBLE[DEV_IDX].sum()),
            'selected_experts': '|'.join(FINAL_POLICY.get('experts', [])),
            'selected_families': '|'.join(FINAL_POLICY.get('families', [])),
            'selected_mode': FINAL_POLICY.get('mode'),
            'selected_coverage': float(FINAL_POLICY.get('coverage', 0.0)),
            'full_development_fit_delta_diagnostic': final_fit_metrics['delta'],
        })
        DEV_METRICS = pd.DataFrame([dev_metrics])

        dev_checks = {
            'nested_delta_floor': dev_metrics['delta'] >= DEV_MIN_DELTA,
            'cluster_bootstrap_positive': dev_ci[0] > 0,
            'outer_fold_stability': dev_metrics['outer_fold_wins'] >= DEV_MIN_OUTER_WINS,
            'id_block_stability': dev_metrics['id_block_wins'] >= DEV_MIN_ID_BLOCK_WINS,
            'worst_fold_floor': dev_metrics['worst_outer_fold_delta'] >= DEV_WORST_FOLD_FLOOR,
            'benefit_harm_ratio': dev_metrics['benefit_harm_ratio'] >= DEV_MIN_BENEFIT_HARM,
            'top1_noninferior': dev_metrics['top1_delta'] >= -1e-12,
            'edge_vs_best_single': dev_metrics['edge_vs_best_single'] >= DEV_MIN_EDGE_VS_SINGLE,
            'selected_expert_exists': len(FINAL_POLICY.get('experts', [])) >= 1,
            'family_stability': family_stability >= DEV_MIN_FAMILY_STABILITY,
            'top4_membership_frozen': dev_metrics['geometry_top4_membership_changes'] == 0,
            'top5_membership_frozen': dev_metrics['geometry_top5_membership_changes'] == 0,
            'tail_frozen': dev_metrics['geometry_rank5_to17_order_changes'] == 0,
            'score_multiset_preserved': bool(dev_metrics['geometry_top4_score_multiset_ok']),
            'at_least_three_nested_actions': dev_metrics['geometry_changed_users'] >= 3,
        }
        DEV_GATE_REASONS = [name for name, passed in dev_checks.items() if not passed]
        DEV_GATE = not DEV_GATE_REASONS
    except Exception as exc:
        DEV_ERROR = f'{type(exc).__name__}: {exc}'
        DEV_GATE_REASONS = ['development_exception']
        traceback.print_exc()
else:
    DEV_GATE_REASONS = ['core_or_inventory_not_ready'] + CORE_ERRORS

print('DEVELOPMENT gate:', DEV_GATE, DEV_GATE_REASONS)
if DEV_ERROR:
    print('DEVELOPMENT error:', DEV_ERROR)
display(DEV_METRICS)
display(OUTER_METRICS)
display(FINAL_SELECTED)

# ---- cell 12 (markdown) ----
# ## One-shot LOCK
#
# The frozen DEVELOPMENT policy is applied once to LOCK. No expert, family, coverage, mode, threshold, or weight changes after LOCK is opened.

# ---- cell 13 (code) ----
LOCK_ERROR = ''
LOCK_OPENED = False
LOCK_GATE = False
LOCK_GATE_REASONS = ['development_gate_failed']
LOCK_CANDIDATE_OOF = V30_OOF[LOCK_IDX].copy() if CORE_READY else np.empty((0, len(MODULES)), np.float32)
LOCK_SINGLE_OOF = LOCK_CANDIDATE_OOF.copy()
LOCK_DELTA_VALUES = np.asarray([], np.float64)
LOCK_METRICS = pd.DataFrame()
POOLED_METRICS = pd.DataFrame()
LOCK_ID_BLOCKS = pd.DataFrame()
LOCK_TEMPLATE_STABILITY = pd.DataFrame()
LOCK_ACTION_LEDGER = pd.DataFrame()
LOCK_SINGLE_EXPERTS = pd.DataFrame()
FROZEN_POLICY_SHA256 = hashlib.sha256(
    json.dumps(FINAL_POLICY, sort_keys=True, separators=(',', ':')).encode()
).hexdigest()

if DEV_GATE:
    try:
        LOCK_OPENED = True
        LOCK_CANDIDATE_OOF, LOCK_ACTION_LEDGER, _ = apply_oof_policy(LOCK_IDX, FINAL_POLICY)
        LOCK_SINGLE_OOF, _, _ = apply_oof_policy(LOCK_IDX, FINAL_SINGLE_POLICY)
        lock_metrics, LOCK_DELTA_VALUES = policy_metrics(Y[LOCK_IDX], V30_OOF[LOCK_IDX], LOCK_CANDIDATE_OOF)
        lock_single_metrics, _ = policy_metrics(Y[LOCK_IDX], V30_OOF[LOCK_IDX], LOCK_SINGLE_OOF)
        lock_ci = cluster_bootstrap_ci(LOCK_DELTA_VALUES, TEMPLATE_GROUP[LOCK_IDX], SEED + 301)
        LOCK_ID_BLOCKS = id_block_metrics(LOCK_IDX, LOCK_DELTA_VALUES)
        LOCK_TEMPLATE_STABILITY = (
            pd.DataFrame({'template_group': TEMPLATE_GROUP[LOCK_IDX], 'delta': LOCK_DELTA_VALUES})
            .groupby('template_group', as_index=False)
            .agg(n=('delta', 'size'), delta=('delta', 'mean'))
            .sort_values(['n', 'template_group'], ascending=[False, True])
            .reset_index(drop=True)
        )
        LOCK_TEMPLATE_STABILITY['positive'] = LOCK_TEMPLATE_STABILITY['delta'] > 0
        if not LOCK_ACTION_LEDGER.empty:
            LOCK_ACTION_LEDGER = LOCK_ACTION_LEDGER.copy()
            LOCK_ACTION_LEDGER['row_global'] = LOCK_IDX[LOCK_ACTION_LEDGER['row_local'].to_numpy(int)]
            LOCK_ACTION_LEDGER['user_id'] = TRAIN_IDS[LOCK_ACTION_LEDGER['row_global'].to_numpy(int)]

        single_rows = []
        for name in FINAL_POLICY.get('experts', []):
            single_policy = dict(FINAL_POLICY)
            single_policy['experts'] = [name]
            single_policy['families'] = [EXPERTS[name]['spec']['family']]
            scores, _, _ = apply_oof_policy(LOCK_IDX, single_policy)
            metrics, _ = policy_metrics(Y[LOCK_IDX], V30_OOF[LOCK_IDX], scores)
            single_rows.append({
                'expert': name, 'family': EXPERTS[name]['spec']['family'],
                'lock_ndcg': metrics['candidate_ndcg'], 'delta_vs_anchor': metrics['delta'],
                'changed_users': metrics['geometry_changed_users'],
                'diagnostic_only_no_reselection': True,
            })
        LOCK_SINGLE_EXPERTS = pd.DataFrame(single_rows)

        pooled_delta = np.concatenate([DEV_DELTA_VALUES, LOCK_DELTA_VALUES])
        pooled_groups = np.concatenate([TEMPLATE_GROUP[DEV_IDX], TEMPLATE_GROUP[LOCK_IDX]])
        pooled_ci = cluster_bootstrap_ci(pooled_delta, pooled_groups, SEED + 302)
        pooled_anchor = np.concatenate([
            per_user_ndcg(Y[DEV_IDX], V30_OOF[DEV_IDX]),
            per_user_ndcg(Y[LOCK_IDX], V30_OOF[LOCK_IDX]),
        ])
        pooled_candidate = np.concatenate([
            per_user_ndcg(Y[DEV_IDX], DEV_CANDIDATE_OOF),
            per_user_ndcg(Y[LOCK_IDX], LOCK_CANDIDATE_OOF),
        ])
        pooled_metrics = {
            'rows': len(pooled_delta), 'anchor_ndcg': float(pooled_anchor.mean()),
            'candidate_ndcg': float(pooled_candidate.mean()),
            'delta': float(pooled_delta.mean()),
            'bootstrap_ci_low': pooled_ci[0], 'bootstrap_ci_high': pooled_ci[1],
            'development_component_is_nested': True, 'lock_policy_is_frozen': True,
        }
        POOLED_METRICS = pd.DataFrame([pooled_metrics])

        current_policy_hash = hashlib.sha256(
            json.dumps(FINAL_POLICY, sort_keys=True, separators=(',', ':')).encode()
        ).hexdigest()
        lock_metrics.update({
            'bootstrap_ci_low': lock_ci[0], 'bootstrap_ci_high': lock_ci[1],
            'development_best_single_policy_ndcg': lock_single_metrics['candidate_ndcg'],
            'edge_vs_development_best_single': lock_metrics['candidate_ndcg'] - lock_single_metrics['candidate_ndcg'],
            'id_block_wins': int((LOCK_ID_BLOCKS['delta'] > 0).sum()),
            'id_block_count': len(LOCK_ID_BLOCKS),
            'worst_id_block_delta': float(LOCK_ID_BLOCKS['delta'].min()),
            'selected_experts': '|'.join(FINAL_POLICY.get('experts', [])),
            'selected_families': '|'.join(FINAL_POLICY.get('families', [])),
            'selected_mode': FINAL_POLICY.get('mode'),
            'selected_coverage': float(FINAL_POLICY.get('coverage', 0.0)),
            'frozen_policy_sha256': FROZEN_POLICY_SHA256,
        })
        LOCK_METRICS = pd.DataFrame([lock_metrics])

        lock_checks = {
            'policy_not_mutated': current_policy_hash == FROZEN_POLICY_SHA256,
            'lock_delta_positive': lock_metrics['delta'] > 0,
            'pooled_delta_floor': pooled_metrics['delta'] >= LOCK_POOLED_MIN_DELTA,
            'pooled_cluster_bootstrap_positive': pooled_ci[0] > 0,
            'benefit_harm_ratio': lock_metrics['benefit_harm_ratio'] >= LOCK_MIN_BENEFIT_HARM,
            'top1_noninferior': lock_metrics['top1_delta'] >= -1e-12,
            'worst_id_block_floor': lock_metrics['worst_id_block_delta'] >= LOCK_BLOCK_FLOOR,
            'top4_membership_frozen': lock_metrics['geometry_top4_membership_changes'] == 0,
            'top5_membership_frozen': lock_metrics['geometry_top5_membership_changes'] == 0,
            'tail_frozen': lock_metrics['geometry_rank5_to17_order_changes'] == 0,
            'score_multiset_preserved': bool(lock_metrics['geometry_top4_score_multiset_ok']),
            'at_least_one_lock_action': lock_metrics['geometry_changed_users'] >= 1,
        }
        LOCK_GATE_REASONS = [name for name, passed in lock_checks.items() if not passed]
        LOCK_GATE = not LOCK_GATE_REASONS
    except Exception as exc:
        LOCK_ERROR = f'{type(exc).__name__}: {exc}'
        LOCK_GATE_REASONS = ['lock_exception']
        traceback.print_exc()

print('LOCK opened:', LOCK_OPENED)
print('LOCK gate:', LOCK_GATE, LOCK_GATE_REASONS)
if LOCK_ERROR:
    print('LOCK error:', LOCK_ERROR)
display(LOCK_METRICS)
display(POOLED_METRICS)

# ---- cell 14 (markdown) ----
# ## Frozen test transformation and submission
#
# Only a policy that passed DEVELOPMENT and one-shot LOCK is applied to test. Eligibility is derived from prediction geometry, never from a user whitelist or leaderboard response.

# ---- cell 15 (code) ----
TEST_ERROR = ''
FINAL_GATE = False
FINAL_GATE_REASONS = ['lock_gate_failed']
TEST_CANDIDATE = ANCHOR_TEST.copy()
PRIMARY_SCORES = ANCHOR_TEST.copy()
PRIMARY_KIND = 'exact_anchor_fail_close'
TEST_ACTION_LEDGER = pd.DataFrame()
TEST_BEHAVIOR = pd.DataFrame()
TEST_MODULE_PAIRS = pd.DataFrame()
TEST_ACTION_SIGNATURES = pd.DataFrame()

if LOCK_GATE:
    try:
        test_surfaces = {name: EXPERTS[name]['test'] for name in FINAL_POLICY.get('experts', [])}
        TEST_CANDIDATE, TEST_ACTION_LEDGER, _ = apply_policy(
            ANCHOR_TEST, test_surfaces, TEST_ELIGIBLE, FINAL_POLICY
        )
        if not TEST_ACTION_LEDGER.empty:
            TEST_ACTION_LEDGER = TEST_ACTION_LEDGER.copy()
            TEST_ACTION_LEDGER['row_global'] = TEST_ACTION_LEDGER['row_local'].to_numpy(int)
            TEST_ACTION_LEDGER['user_id'] = TEST_IDS[TEST_ACTION_LEDGER['row_global'].to_numpy(int)]
            TEST_ACTION_LEDGER['moved_modules'] = TEST_ACTION_LEDGER.apply(
                lambda row: '|'.join(sorted({module for old, new in zip(row.old_top4.split('|'), row.new_top4.split('|')) for module in (old, new) if old != new})),
                axis=1,
            )
            TEST_MODULE_PAIRS = (
                TEST_ACTION_LEDGER.groupby(['action_type', 'moved_modules'], as_index=False)
                .size().rename(columns={'size': 'users'}).sort_values('users', ascending=False)
            )
            TEST_ACTION_SIGNATURES = (
                TEST_ACTION_LEDGER.groupby(['action_type', 'old_top4', 'new_top4'], as_index=False)
                .size().rename(columns={'size': 'users'}).sort_values('users', ascending=False)
            )

        geometry = prediction_geometry(ANCHOR_TEST, TEST_CANDIDATE)
        policy_hash = hashlib.sha256(
            json.dumps(FINAL_POLICY, sort_keys=True, separators=(',', ':')).encode()
        ).hexdigest()
        locked_exact = bool(np.array_equal(TEST_CANDIDATE[~TEST_ELIGIBLE], ANCHOR_TEST[~TEST_ELIGIBLE]))
        final_checks = {
            'lock_gate_passed': LOCK_GATE,
            'policy_not_mutated': policy_hash == FROZEN_POLICY_SHA256,
            'all_selected_test_surfaces_present': set(FINAL_POLICY.get('experts', [])).issubset(EXPERTS),
            'test_action_lower_bound': geometry['changed_users'] >= TEST_MIN_ACTIONS,
            'test_action_upper_bound': geometry['changed_users'] <= TEST_MAX_ACTIONS,
            'historical_disagreement_rows_exactly_locked': locked_exact,
            'top4_membership_frozen': geometry['top4_membership_changes'] == 0,
            'top5_membership_frozen': geometry['top5_membership_changes'] == 0,
            'tail_frozen': geometry['rank5_to17_order_changes'] == 0,
            'score_multiset_preserved': bool(geometry['top4_score_multiset_ok']),
            'finite_scores': bool(np.isfinite(TEST_CANDIDATE).all()),
            'shape_valid': TEST_CANDIDATE.shape == ANCHOR_TEST.shape,
            'smoke_mode_disabled': not SMOKE,
        }
        FINAL_GATE_REASONS = [name for name, passed in final_checks.items() if not passed]
        FINAL_GATE = not FINAL_GATE_REASONS
        TEST_BEHAVIOR = pd.DataFrame([{
            **geometry,
            'eligible_users': int(TEST_ELIGIBLE.sum()),
            'protected_users': int((~TEST_ELIGIBLE).sum()),
            'selected_experts': '|'.join(FINAL_POLICY.get('experts', [])),
            'selected_families': '|'.join(FINAL_POLICY.get('families', [])),
            'mode': FINAL_POLICY.get('mode'),
            'coverage': float(FINAL_POLICY.get('coverage', 0.0)),
            'candidate_rank_sha256': rank_sha256(TEST_CANDIDATE),
            'candidate_top5_sha256': top5_sha256(TEST_CANDIDATE),
            'final_gate': FINAL_GATE,
        }])
    except Exception as exc:
        TEST_ERROR = f'{type(exc).__name__}: {exc}'
        FINAL_GATE_REASONS = ['test_transformation_exception']
        TEST_CANDIDATE = ANCHOR_TEST.copy()
        traceback.print_exc()

if FINAL_GATE:
    PRIMARY_SCORES = TEST_CANDIDATE.copy()
    PRIMARY_KIND = 'validated_historical_ensemble'
else:
    PRIMARY_SCORES = ANCHOR_TEST.copy()
    PRIMARY_KIND = 'exact_anchor_fail_close'
    if not np.array_equal(PRIMARY_SCORES, ANCHOR_TEST):
        raise AssertionError('Fail-close output differs from exact champion')

print('Final gate:', FINAL_GATE, FINAL_GATE_REASONS)
print('Primary kind:', PRIMARY_KIND)
if TEST_ERROR:
    print('Test transformation error:', TEST_ERROR)
display(TEST_BEHAVIOR)
display(TEST_ACTION_SIGNATURES.head(20))

# ---- cell 16 (code) ----
def submission_frame(scores):
    scores = validate_matrix(scores, len(TEST_IDS), 'submission scores')
    frame = pd.DataFrame({'user_id': TEST_IDS})
    frame[MODULES] = scores
    return frame

def validate_submission(frame, scores, label):
    if list(frame.columns) != list(sample.columns):
        raise ValueError(f'{label} columns differ from sample_submission.csv')
    if frame['user_id'].astype(str).tolist() != TEST_IDS.tolist() or not frame['user_id'].is_unique:
        raise ValueError(f'{label} user order or uniqueness is invalid')
    values = frame[MODULES].to_numpy(np.float32)
    if values.shape != (len(TEST_IDS), len(MODULES)) or not np.isfinite(values).all():
        raise ValueError(f'{label} values are invalid')
    if not np.array_equal(values, np.asarray(scores, np.float32)):
        raise ValueError(f'{label} changed score precision or ordering')

def json_default(value):
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f'Not JSON serializable: {type(value).__name__}')

def write_json(payload, filename):
    path = OUT_DIR / filename
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + '\n')
    return path

EXPERT_MANIFEST = EXPERT_MANIFEST.copy()
selected_names = set(FINAL_POLICY.get('experts', []))
EXPERT_MANIFEST['selected_by_final_development_fit'] = EXPERT_MANIFEST['name'].isin(selected_names)
EXPERT_MANIFEST['selection_status'] = np.select(
    [
        ~EXPERT_MANIFEST['loaded'],
        EXPERT_MANIFEST['duplicate_of_anchor'],
        EXPERT_MANIFEST['selected_by_final_development_fit'],
    ],
    ['excluded_load_or_schema_failure', 'excluded_duplicate_or_near_anchor', 'selected'],
    default='eligible_not_selected',
)
EXPERT_MANIFEST['deployed_in_primary'] = EXPERT_MANIFEST['name'].isin(selected_names) & FINAL_GATE

if FINAL_SELECTED.empty:
    FINAL_SELECTED = pd.DataFrame([{
        'expert': 'anchor_prior', 'family': 'champion_anchor', 'version': 'v30_oof__v48_test',
        'train_surface': CORE_SPECS['v30_oof'][0], 'test_surface': anchor_relative,
        'participation_weight': 1.0, 'outer_fold_selection_count': 0,
    }])
FINAL_SELECTED = FINAL_SELECTED.copy()
FINAL_SELECTED['deployed_in_primary'] = FINAL_GATE
if not np.isclose(FINAL_SELECTED['participation_weight'].sum(), 1.0, rtol=0.0, atol=1e-7):
    raise AssertionError('Selected participation weights do not sum to one')

PRIMARY_SUBMISSION = submission_frame(PRIMARY_SCORES)
ANCHOR_SUBMISSION = submission_frame(ANCHOR_TEST)
validate_submission(PRIMARY_SUBMISSION, PRIMARY_SCORES, 'primary submission')
validate_submission(ANCHOR_SUBMISSION, ANCHOR_TEST, 'anchor submission')
if top5_sha256(ANCHOR_TEST) != EXPECTED_ANCHOR_TOP5_SHA256:
    raise AssertionError('Anchor changed before export')
if not FINAL_GATE and not np.array_equal(PRIMARY_SCORES, ANCHOR_TEST):
    raise AssertionError('Failed validation did not close to the exact champion')

primary_order = stable_order(PRIMARY_SCORES)[:, :5]
PRIMARY_RANKING = pd.DataFrame({'user_id': TEST_IDS})
for rank in range(5):
    PRIMARY_RANKING[f'rank_{rank + 1}'] = [MODULES[i] for i in primary_order[:, rank]]

core_rows = [{
    'role': 'exact_current_champion_test_anchor', 'relative_path': anchor_relative,
    'key': '', 'shape': str(ANCHOR_TEST.shape), 'matrix_sha256': matrix_sha256(ANCHOR_TEST),
    'rank_sha256': rank_sha256(ANCHOR_TEST), 'alignment_evidence': 'user-aligned CSV plus fixed Top-5 hash',
}]
if CORE_READY:
    for role, relative, key, matrix, evidence in [
        ('v30_train_anchor_proxy', CORE_SPECS['v30_oof'][0], CORE_SPECS['v30_oof'][1], V30_OOF, 'executed V33 full four-fold cross-fit plus stage1_fold'),
        ('v30_test', CORE_SPECS['v30_test'][0], CORE_SPECS['v30_test'][1], V30_TEST, 'exact match to user-aligned V30 submission'),
        ('consensus3_oof', CORE_SPECS['consensus_oof'][0], CORE_SPECS['consensus_oof'][1], CONSENSUS_OOF, 'embedded y matches organizer row and module order'),
        ('consensus3_test', CORE_SPECS['consensus_test'][0], CORE_SPECS['consensus_test'][1], CONSENSUS_TEST, 'paired with verified Consensus3 OOF'),
    ]:
        core_rows.append({
            'role': role, 'relative_path': relative, 'key': key, 'shape': str(matrix.shape),
            'matrix_sha256': matrix_sha256(matrix), 'rank_sha256': rank_sha256(matrix),
            'alignment_evidence': evidence,
        })
CORE_ARTIFACT_AUDIT = pd.DataFrame(core_rows)

FAILURE_FLAGS = pd.DataFrame([
    {'scope': 'anchor_hash', 'passed': top5_sha256(ANCHOR_TEST) == EXPECTED_ANCHOR_TOP5_SHA256,
     'reasons': ''},
    {'scope': 'core_artifacts', 'passed': CORE_READY, 'reasons': '|'.join(CORE_ERRORS)},
    {'scope': 'validation_inventory', 'passed': VALIDATION_READY,
     'reasons': '' if VALIDATION_READY else 'insufficient_valid_families_or_eligible_rows'},
    {'scope': 'development_gate', 'passed': DEV_GATE, 'reasons': '|'.join(DEV_GATE_REASONS + ([DEV_ERROR] if DEV_ERROR else []))},
    {'scope': 'lock_opening_contract', 'passed': bool(LOCK_OPENED == DEV_GATE),
     'reasons': '' if LOCK_OPENED == DEV_GATE else 'LOCK opening disagrees with DEVELOPMENT gate'},
    {'scope': 'lock_gate', 'passed': LOCK_GATE, 'reasons': '|'.join(LOCK_GATE_REASONS + ([LOCK_ERROR] if LOCK_ERROR else []))},
    {'scope': 'final_gate', 'passed': FINAL_GATE, 'reasons': '|'.join(FINAL_GATE_REASONS + ([TEST_ERROR] if TEST_ERROR else []))},
    {'scope': 'fail_close_exactness', 'passed': bool(FINAL_GATE or np.array_equal(PRIMARY_SCORES, ANCHOR_TEST)),
     'reasons': ''},
])

CONFIG = {
    'version': 'v49',
    'seed': SEED,
    'smoke_mode': SMOKE,
    'metric': 'mean user-level NDCG@5',
    'algorithm': 'nested family-balanced greedy rank voting with champion shrinkage',
    'anchor': {
        'public_score_audit_only': PUBLIC_ANCHOR,
        'source_relative': anchor_relative,
        'file_sha256': ANCHOR_FILE_SHA256,
        'expected_file_sha256': EXPECTED_ANCHOR_FILE_SHA256,
        'top5_sha256': ANCHOR_TOP5_HASH,
        'expected_top5_sha256': EXPECTED_ANCHOR_TOP5_SHA256,
        'train_proxy': 'V30 full cross-fitted OOF from V33',
        'prior_votes': ANCHOR_PRIOR_VOTES,
    },
    'paths': {
        'data_dir': DATA_DIR,
        'artifact_root': ARTIFACT_ROOT,
        'output_dir': OUT_DIR,
    },
    'data_audit': {
        'unused_out_of_cohort_chat_users': len(CHAT_OUT_OF_SCOPE_USERS),
        'chat_features_used_by_v49': False,
    },
    'expected_core_file_sha256': EXPECTED_CORE_FILE_SHA256,
    'validation': {
        'development_rows': len(DEV_IDX),
        'lock_rows': len(LOCK_IDX),
        'outer_folds': OUTER_FOLDS,
        'group': 'v48 latest-template group',
        'stratification': 'highest-relevance module',
        'bootstrap_replicates': BOOTSTRAPS,
        'id_blocks': ID_BLOCKS,
        'lock_opened_only_after_development_gate': True,
    },
    'search_space': {
        'authority_modes': AUTHORITY_MODES,
        'coverage_grid': COVERAGE_GRID,
        'maximum_distinct_families': MAX_SELECTED_FAMILIES,
        'quality_floor_below_anchor': QUALITY_FLOOR_BELOW_ANCHOR,
        'nonnegative_rank_votes': True,
        'test_specific_weights': False,
        'public_scores_used_for_selection': False,
    },
    'gates': {
        'development_delta': DEV_MIN_DELTA,
        'development_outer_wins': DEV_MIN_OUTER_WINS,
        'development_id_block_wins': DEV_MIN_ID_BLOCK_WINS,
        'development_benefit_harm': DEV_MIN_BENEFIT_HARM,
        'development_worst_fold': DEV_WORST_FOLD_FLOOR,
        'development_edge_vs_single': DEV_MIN_EDGE_VS_SINGLE,
        'development_family_stability': DEV_MIN_FAMILY_STABILITY,
        'pooled_delta': LOCK_POOLED_MIN_DELTA,
        'lock_benefit_harm': LOCK_MIN_BENEFIT_HARM,
        'lock_block_floor': LOCK_BLOCK_FLOOR,
        'test_action_range': [TEST_MIN_ACTIONS, TEST_MAX_ACTIONS],
    },
    'final_policy': FINAL_POLICY,
    'frozen_policy_sha256': FROZEN_POLICY_SHA256,
    'gate_status': {
        'core_ready': CORE_READY, 'validation_ready': VALIDATION_READY,
        'development_gate': DEV_GATE, 'lock_opened': LOCK_OPENED,
        'lock_gate': LOCK_GATE, 'final_gate': FINAL_GATE,
    },
    'primary_kind': PRIMARY_KIND,
    'libraries': {
        'python': platform.python_version(), 'numpy': np.__version__,
        'pandas': pd.__version__, 'scikit_learn': sklearn.__version__,
    },
    'provenance_contract': {
        'organizer_data_only': True,
        'participant_owned_historical_predictions_only': True,
        'genuine_cross_fitted_oof_required_for_learned_selection': True,
        'test_predictions_never_used_for_selection': True,
        'test_user_ids_never_used_for_policy': True,
        'hard_coded_test_actions': False,
    },
}

SUMMARY = {
    'version': 'v49',
    'primary_kind': PRIMARY_KIND,
    'core_errors': CORE_ERRORS,
    'final_gate': FINAL_GATE,
    'final_gate_reasons': FINAL_GATE_REASONS,
    'development_gate': DEV_GATE,
    'development_gate_reasons': DEV_GATE_REASONS,
    'development_error': DEV_ERROR,
    'lock_opened': LOCK_OPENED,
    'lock_gate': LOCK_GATE,
    'lock_gate_reasons': LOCK_GATE_REASONS,
    'lock_error': LOCK_ERROR,
    'test_error': TEST_ERROR,
    'selected_experts': FINAL_POLICY.get('experts', []),
    'selected_families': FINAL_POLICY.get('families', []),
    'authority_mode': FINAL_POLICY.get('mode'),
    'coverage': float(FINAL_POLICY.get('coverage', 0.0)),
    'test_changed_users': int(prediction_geometry(ANCHOR_TEST, PRIMARY_SCORES)['changed_users']),
    'primary_top5_sha256': top5_sha256(PRIMARY_SCORES),
    'anchor_top5_sha256': ANCHOR_TOP5_HASH,
    'interpretation': (
        'Validated historical ensemble.' if FINAL_GATE else
        'Exact champion fail-close; this is not evidence of a V49 improvement.'
    ),
}

tables = {
    'artifact_inventory.csv': ARTIFACT_INVENTORY,
    'core_artifact_audit.csv': CORE_ARTIFACT_AUDIT,
    'lineage_provenance_audit.csv': LINEAGE_AUDIT,
    'expert_manifest.csv': EXPERT_MANIFEST,
    'duplicate_near_duplicate_report.csv': DUPLICATE_REPORT,
    'oof_expert_metrics.csv': EXPERT_METRICS,
    'pairwise_diversity.csv': DIVERSITY_PAIRS,
    'rank_correlation.csv': RANK_CORRELATION.rename_axis('expert').reset_index(),
    'ensemble_selection_trace.csv': SELECTION_TRACE,
    'family_quality_audit.csv': QUALITY_AUDIT,
    'final_selected_experts.csv': FINAL_SELECTED,
    'outer_fold_policies.csv': OUTER_POLICIES,
    'outer_fold_stability.csv': OUTER_METRICS,
    'development_metrics.csv': DEV_METRICS,
    'development_id_block_stability.csv': DEV_ID_BLOCKS,
    'development_template_stability.csv': DEV_TEMPLATE_STABILITY,
    'development_action_ledger.csv': DEV_ACTION_LEDGER,
    'lock_metrics.csv': LOCK_METRICS,
    'pooled_development_lock_metrics.csv': POOLED_METRICS,
    'lock_id_block_stability.csv': LOCK_ID_BLOCKS,
    'lock_template_stability.csv': LOCK_TEMPLATE_STABILITY,
    'lock_single_expert_attribution.csv': LOCK_SINGLE_EXPERTS,
    'lock_action_ledger.csv': LOCK_ACTION_LEDGER,
    'leave_one_family_out_attribution.csv': LEAVE_ONE_FAMILY_OUT,
    'test_behavior_audit.csv': TEST_BEHAVIOR,
    'test_module_pair_concentration.csv': TEST_MODULE_PAIRS,
    'test_action_signature_concentration.csv': TEST_ACTION_SIGNATURES,
    'test_action_ledger.csv': TEST_ACTION_LEDGER,
    'primary_top5_ranking_audit.csv': PRIMARY_RANKING,
    'failure_flags.csv': FAILURE_FLAGS,
}
for filename, frame in tables.items():
    frame.to_csv(OUT_DIR / filename, index=False)

np.save(OUT_DIR / 'v49_test_candidate_scores.npy', np.asarray(TEST_CANDIDATE, np.float32))
np.save(OUT_DIR / 'v49_test_primary_scores.npy', np.asarray(PRIMARY_SCORES, np.float32))
np.save(OUT_DIR / 'v49_anchor_reference_scores.npy', np.asarray(ANCHOR_TEST, np.float32))
PRIMARY_SUBMISSION.to_csv(OUT_DIR / 'submission_v49_primary.csv', index=False)
ANCHOR_SUBMISSION.to_csv(OUT_DIR / 'submission_v49_anchor_reference.csv', index=False)
write_json(CONFIG, 'reproducibility_config.json')
write_json(SUMMARY, 'v49_summary.json')

output_rows = []
for path in sorted(p for p in OUT_DIR.iterdir() if p.is_file() and p.name != 'runtime_output_manifest.csv'):
    output_rows.append({
        'filename': path.name, 'bytes': int(path.stat().st_size),
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    })
pd.DataFrame(output_rows).to_csv(OUT_DIR / 'runtime_output_manifest.csv', index=False)

if Path('/kaggle/working').exists():
    PRIMARY_SUBMISSION.to_csv('/kaggle/working/submission_v49_primary.csv', index=False)

print('Wrote', len(output_rows) + 1, 'auditable V49 artifacts to', OUT_DIR)
print('PRIMARY:', PRIMARY_KIND, 'final_gate =', FINAL_GATE)
print('Use only:', OUT_DIR / 'submission_v49_primary.csv')
display(pd.DataFrame([SUMMARY]))

# ---- cell 17 (markdown) ----
# ## Terminal interpretation contract
#
# Use only `submission_v49_primary.csv`, and only when `final_gate` is true in `v49_summary.json`. When the gate is false, the primary file is the exact champion reference and must not consume a leaderboard slot as a V49 claim. A positive LOCK result validates only the frozen package. Do not retune from LOCK or public feedback.
