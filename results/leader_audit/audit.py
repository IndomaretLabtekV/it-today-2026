"""Diagnostic audit, not a submission builder or evidence of beating the champion.

Fixed before execution: reproduce V98 with seed 2424, change only bootstrap
training membership to respect the same outer folds, then repeat with seed
20260905 and fold-fitted vocabulary. Ten bags, Ridge alpha=1, no tuning.
Ranker diagnostic: one V125 fold, identical preprocessing/grades/parameters,
only append module one-hot features. Do not interpret this as a promotion gate.
All organizer files and historical results are read only. Prints JSON to stdout.
"""

import hashlib
import json
import re
from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[2]
MODS = [f"M_{i:03d}" for i in range(1, 18)]
DISCOUNT = 1 / np.log2(np.arange(2, 7))
WORDS = dict(zip("nol satu dua tiga empat lima".split(), range(6)))


def ndcg(y, scores):
    # Same convention as V98/V125, so the reproduction is directly comparable.
    order = np.argsort(-scores, axis=1)[:, :5]
    dcg = (np.take_along_axis(y, order, axis=1) * DISCOUNT).sum(1)
    ideal = (np.sort(y, axis=1)[:, -5:][:, ::-1] * DISCOUNT).sum(1)
    return dcg / np.maximum(ideal, 1e-12)


def compare(y, base, candidate, fids, groups, seed):
    delta = ndcg(y, candidate) - ndcg(y, base)
    rng = np.random.RandomState(seed)
    ci = np.quantile(
        [rng.choice(delta, len(delta), replace=True).mean() for _ in range(2000)],
        [0.025, 0.975],
    )
    # Resample whole prefix groups and preserve the user-weighted estimand.
    unique, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=delta)
    sizes = np.bincount(inv)
    group_draws = rng.randint(len(unique), size=(2000, len(unique)))
    cluster_ci = np.quantile(
        sums[group_draws].sum(1) / sizes[group_draws].sum(1), [0.025, 0.975]
    )
    folds = [float(delta[fids == f].mean()) for f in np.unique(fids)]
    return {
        "baseline_ndcg": float(ndcg(y, base).mean()),
        "candidate_ndcg": float(ndcg(y, candidate).mean()),
        "delta": float(delta.mean()),
        "user_ci95": ci.tolist(),
        "prefix_cluster_ci95": cluster_ci.tolist(),
        "fold_deltas": folds,
        "worst_fold": min(folds),
        "passes_historical_002_floor": bool(ci[0] > 0.002 and min(folds) >= 0),
    }


def bagging_audit(docs, y, groups, seed, fold_fitted):
    n = len(y)
    vectorizer = lambda: TfidfVectorizer(max_features=5000, min_df=3)
    # Global train vocabulary retained only in the exact V98 counterfactual.
    full_x = None if fold_fitted else vectorizer().fit_transform(docs).toarray()
    single = np.zeros_like(y)
    nested_bags = np.zeros_like(y)
    fids = np.zeros(n, int)
    splits = list(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(
            np.zeros(n), y.argmax(1), groups
        )
    )
    for fi, (tr, va) in enumerate(splits):
        assert not (set(groups[tr]) & set(groups[va]))
        if fold_fitted:
            vec = vectorizer()
            xt = vec.fit_transform(docs[tr]).toarray()
            xv = vec.transform(docs[va]).toarray()
        else:
            xt, xv = full_x[tr], full_x[va]
        # Multi-target Ridge uses the same independent least-squares solutions.
        single[va] = Ridge(alpha=1.0).fit(xt, y[tr]).predict(xv)
        rng = np.random.RandomState(seed + fi)
        for _ in range(10):
            local = rng.choice(len(tr), len(tr), replace=True)
            assert not (set(groups[tr[local]]) & set(groups[va]))
            nested_bags[va] += Ridge(alpha=1.0).fit(xt[local], y[tr[local]]).predict(xv) / 10
        fids[va] = fi

    result = {
        "seed": seed,
        "fold_fitted_vocabulary": fold_fitted,
        "fold_sizes": [len(va) for _, va in splits],
        "same_outer_fold_comparison": compare(y, single, nested_bags, fids, groups, seed),
    }
    if not fold_fitted:
        rng = np.random.RandomState(seed)
        acc = np.zeros_like(y)
        counts = np.zeros(n, int)
        contaminated, evaluations = 0, 0
        for _ in range(10):
            idx = rng.choice(n, n, replace=True)
            oob = np.setdiff1d(np.arange(n), np.unique(idx))
            assert not (set(idx) & set(oob))
            contaminated += int(np.isin(groups[oob], np.unique(groups[idx])).sum())
            evaluations += len(oob)
            acc[oob] += Ridge(alpha=1.0).fit(full_x[idx], y[idx]).predict(full_x[oob])
            counts[oob] += 1
        oob_preds = np.where(counts[:, None] > 0, acc / np.maximum(counts[:, None], 1), single)
        result["original_v98_mismatched_comparison"] = compare(y, single, oob_preds, fids, groups, seed)
        result["zero_oob_users"] = int((counts == 0).sum())
        result["oob_evaluations_with_seen_prefix_group_fraction"] = contaminated / evaluations
        result["mean_oob_models_per_user"] = float(counts.mean())
    return result


def ranker_diagnostic(docs_all, assessment_raw, y, groups):
    # Preserve V125 preprocessing to isolate the missing module identity.
    words = TfidfVectorizer(max_features=5000, min_df=3).fit_transform(docs_all).toarray()
    a = (assessment_raw - assessment_raw.mean(0)) / (assessment_raw.std(0) + 1e-9)
    x = np.hstack([words, a]).astype(np.float32)[:len(y)]
    tr, va = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=5656).split(x, y.argmax(1), groups))
    grades = np.rint(y * 5).astype(int)
    result = {"scope": "one fold; original V125 preprocessing; not a generalization gate", "n_train": len(tr), "n_val": len(va)}
    for with_module in [False, True]:
        xt, xv = np.repeat(x[tr], 17, axis=0), np.repeat(x[va], 17, axis=0)
        if with_module:
            xt = np.hstack([xt, np.tile(np.eye(17, dtype=np.float32), (len(tr), 1))])
            xv = np.hstack([xv, np.tile(np.eye(17, dtype=np.float32), (len(va), 1))])
        within_user_feature_variation = np.any(np.ptp(xv.reshape(len(va), 17, -1), axis=1) > 0, axis=1)
        assert within_user_feature_variation.all() if with_module else not within_user_feature_variation.any()
        model = lgb.LGBMRanker(
            objective="lambdarank", metric="ndcg", n_estimators=200,
            learning_rate=0.05, num_leaves=15, min_child_samples=80,
            reg_alpha=1.0, reg_lambda=10.0, verbose=-1,
            deterministic=True, seed=5656, force_col_wise=True, n_jobs=2,
        )
        model.fit(xt, grades[tr].reshape(-1), group=np.full(len(tr), 17))
        pred = model.predict(xv).reshape(len(va), 17)
        result["module_onehot" if with_module else "original_user_only"] = {
            "varying_feature_fraction": float(within_user_feature_variation.mean()),
            "mean_within_user_prediction_std": float(pred.std(1).mean()),
            "nonconstant_user_fraction": float((np.ptp(pred, axis=1) > 1e-8).mean()),
            "ndcg": float(ndcg(y[va], pred).mean()),
        }
    assert result["original_user_only"]["nonconstant_user_fraction"] == 0
    assert result["module_onehot"]["nonconstant_user_fraction"] > 0
    return result


def utility_diagnostic(docs_all, y, groups):
    """Reproduce V124 labels, then verify the exact swap identity on every user.

    This audits counterfactual construction only, not a learned utility policy.
    """
    w = TfidfVectorizer(max_features=5000, min_df=3).fit_transform(docs_all).toarray()[:len(y)]
    incumbent = np.zeros_like(y)
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=5555)
    for tr, va in folds.split(w, y.argmax(1), groups):
        incumbent[va] = Ridge(alpha=1.0).fit(w[tr], y[tr]).predict(w[va])
    order = np.argsort(-incumbent, axis=1, kind="stable")
    base = ndcg(y, incumbent)
    ideal = (np.sort(y, axis=1)[:, -5:][:, ::-1] * DISCOUNT).sum(1)
    bad = np.zeros((len(y), 3))
    correct = np.zeros_like(bad)
    rows = np.arange(len(y))
    for slot in [1, 2, 3]:
        swapped = order.copy()
        swapped[:, [0, slot]] = swapped[:, [slot, 0]]
        sim = np.zeros_like(y)
        sim[rows[:, None], swapped] = y[rows[:, None], swapped]
        # This is exactly V124's assignment; the permutation cancels itself.
        assert np.array_equal(sim, y)
        bad[:, slot - 1] = (sim[:, :5] * DISCOUNT).sum(1) / ideal - base
        correct[:, slot - 1] = (y[rows, order[:, slot]] - y[rows, order[:, 0]]) * (DISCOUNT[0] - DISCOUNT[slot]) / ideal
        actual = incumbent.copy()
        actual[rows, order[:, 0]] = incumbent[rows, order[:, slot]]
        actual[rows, order[:, slot]] = incumbent[rows, order[:, 0]]
        np.testing.assert_allclose(ndcg(y, actual) - base, correct[:, slot - 1], atol=1e-12)
    return {
        "scope": "label identity audit against word Ridge, not champion; no selector trained",
        "baseline_ndcg": float(base.mean()),
        "old_mean_utility": float(bad.mean()),
        "old_positive_action_fraction": float((bad > 0).mean()),
        "old_users_with_different_action_labels": int((np.ptp(bad, axis=1) > 1e-12).sum()),
        "correct_mean_utility": float(correct.mean()),
        "correct_positive_action_fraction": float((correct > 0).mean()),
        "correct_users_with_different_action_labels": int((np.ptp(correct, axis=1) > 1e-12).sum()),
        "correct_best_single_swap_oracle_delta": float(np.maximum(correct.max(1), 0).mean()),
        "verified_exact_score_swap_cases": len(y) * 3,
    }


def main():
    train = pd.read_csv(ROOT / "data/train_relevance.csv")
    test = pd.read_csv(ROOT / "data/test.csv")
    chats = pd.read_csv(ROOT / "data/chat_history.csv")
    assessments = pd.read_csv(ROOT / "data/user_assessments.csv")
    y = train[MODS].to_numpy(float)
    ids = train.user_id.tolist() + test.user_id.tolist()
    assert not (set(train.user_id) & set(test.user_id))
    by = chats.groupby("user_id").user_chat_text.apply(list)
    docs = np.array([" ".join(map(str, by.get(u, []))) for u in ids], dtype=object)
    groups = np.array([
        re.sub(r"\d+", "#", " ".join(str(by[u][0]).lower().split()[:6])) if u in by else "NO_CHAT"
        for u in train.user_id
    ])
    blobs = dict(zip(assessments.user_id, assessments.assessment_result.map(json.loads)))
    questions = sorted({q for blob in blobs.values() for q in blob})
    def num(value):
        return float(WORDS[str(value).strip().lower()]) if str(value).strip().lower() in WORDS else float(value)
    raw = np.array([[num(blobs[u][q]) for q in questions] for u in ids])
    if "--utility-only" in sys.argv:
        print(json.dumps({"utility_diagnostic": utility_diagnostic(docs, y, groups)}), flush=True)
        return
    no_chat = np.array([u not in by for u in ids])
    invalid = (raw < 0) | (raw > 5) | ~np.isfinite(raw)
    champion = pd.read_csv(ROOT / "results/v42/submission_v42_primary.csv")
    assert champion.user_id.tolist() == test.user_id.tolist()
    profile = {
        "train_users": len(train), "test_users": len(test), "modules": len(MODS),
        "nonzero_label_fraction": float((y > 0).mean()),
        "no_chat_train": int(no_chat[:len(train)].sum()), "no_chat_test": int(no_chat[len(train):].sum()),
        "out_of_scope_chat_users": len(set(chats.user_id) - set(ids)),
        "invalid_assessment_cells": int(invalid.sum()),
        "invalid_assessment_users": int(invalid.any(1).sum()),
        "invalid_nochat_train_users": int((invalid.any(1) & no_chat)[:len(train)].sum()),
        "invalid_nochat_test_users": int((invalid.any(1) & no_chat)[len(train):].sum()),
        "prefix_group_count": len(np.unique(groups)),
        "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT / "data").glob("*.csv"))},
        "champion_file_sha256": hashlib.sha256((ROOT / "results/v42/submission_v42_primary.csv").read_bytes()).hexdigest(),
    }
    print(json.dumps({"profile": profile}), flush=True)
    for seed, fold_fitted in [(2424, False), (20260905, True)]:
        print(json.dumps({"bagging_audit": bagging_audit(docs[:len(train)], y, groups, seed, fold_fitted)}), flush=True)
    print(json.dumps({"ranker_diagnostic": ranker_diagnostic(docs, raw, y, groups)}), flush=True)
    print(json.dumps({"utility_diagnostic": utility_diagnostic(docs, y, groups)}), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
