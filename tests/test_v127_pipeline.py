"""Behavioral regressions for the V127 leakage and deployment review."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_pipeline():
    path = ROOT / "nb_code/v127_pipeline.py"
    assert path.exists(), "V127 needs an importable, side-effect-free pipeline for boundary checks"
    spec = importlib.util.spec_from_file_location("v127_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_swap_uses_champion_scores_and_leaves_protected_users_exact():
    pipeline = load_pipeline()
    champion = np.array([[.9, .8, .7, .6, .5, .4], [.8, .9, .7, .6, .5, .4]])
    result = pipeline.apply_actions(champion, np.array([2, 1]), np.array([True, False]))
    np.testing.assert_array_equal(result[0], [.7, .8, .9, .6, .5, .4])
    np.testing.assert_array_equal(result[1], champion[1])
    np.testing.assert_array_equal(pipeline.apply_actions(champion, [-1, -1], [True, True]), champion)


def test_swap_targets_equal_direct_ndcg_for_all_actions():
    pipeline = load_pipeline()
    y = np.array([[0, .2, .9, .6, .7, .1], [.9, 0, .2, .7, .1, .5]])
    scores = np.array([[.8, .7, .6, .5, .4, .3], [.4, .5, .6, .7, .8, .9]])
    targets = pipeline.action_targets(y, scores)
    for slot in [1, 2, 3]:
        candidate = pipeline.apply_actions(scores, [slot, slot], [True, True])
        expected = pipeline.per_user_ndcg(y, candidate) - pipeline.per_user_ndcg(y, scores)
        np.testing.assert_allclose(targets[:, slot - 1], expected, atol=1e-12)
    assert np.ptp(targets[0]) > 0


def test_semantic_features_preserve_coordinates_not_just_cosines():
    pipeline = load_pipeline()
    # These candidate modules have equal cosine to the user but different coordinates.
    users = np.array([[1., 1.]]) / np.sqrt(2)
    modules = np.array([[0., -1.], [1., 0.], [0., 1.], [-1., 0.]])
    scores = np.array([[.9, .8, .7, .6]])
    features = pipeline.semantic_features(users, modules, scores)
    assert features.shape[1] == 3
    assert features.shape[2] >= 6
    assert not np.array_equal(features[0, 0], features[0, 1])


def test_gate_rejects_uncertain_or_harmful_increment():
    pipeline = load_pipeline()
    assert not pipeline.promotion_gate(.003, [-.0001, .005], [.003] * 5, [.003] * 5, .001, False)
    assert not pipeline.promotion_gate(.003, [.0021, .005], [.003] * 5, [-.001, .004], .001, False)
    assert not pipeline.promotion_gate(.003, [.0021, .005], [.003] * 5, [.003] * 5, -.001, False)
    assert not pipeline.promotion_gate(.003, [.0021, .005], [.003] * 5, [.003] * 5, .001, True)
    assert pipeline.promotion_gate(.003, [.0021, .005], [.003] * 5, [.003] * 5, .001, False)


def test_failed_gate_copies_original_bytes_and_success_checks_geometry(tmp_path):
    pipeline = load_pipeline()
    source = tmp_path / "champion.csv"
    source.write_bytes(b"user_id,M_001,M_002,M_003,M_004,M_005,M_006\nU1,0.9,0.8,0.7,0.6,0.5,0.4\n")
    scores = np.array([[.9, .8, .7, .6, .5, .4]])
    out = tmp_path / "failed"
    path = pipeline.write_submission(source, out, scores, False, np.array([False]))
    assert path.read_bytes() == source.read_bytes()
    unsafe = scores.copy()
    unsafe[0, 5] = 1.
    with pytest.raises(AssertionError):
        pipeline.write_submission(source, tmp_path / "unsafe", unsafe, True, np.array([True]))
    changed = pipeline.apply_actions(scores, [1], [True])
    with pytest.raises(AssertionError):
        pipeline.write_submission(source, tmp_path / "protected", changed, True, np.array([False]))


def test_selector_training_does_not_depend_on_excluded_labels(tmp_path):
    pipeline = load_pipeline()
    data = pipeline.load_data(ROOT / "data")
    # Whole groups are kept together; avoid choosing a contiguous label-defined slice.
    groups = np.unique(data.groups)[:24]
    allowed = np.flatnonzero(np.isin(data.groups, groups))[:72]
    excluded = np.setdiff1d(np.arange(data.n_train), allowed)
    cfg = pipeline.Config(smoke=True, inner_splits=2, outer_splits=2,
                          calibration_trees=3, ranker_trees=3,
                          word_features=64, char_features=64, threads=1)
    rng = np.random.default_rng(19)
    user_embeddings = rng.normal(size=(len(data.user_ids), 4))
    module_embeddings = rng.normal(size=(17, 4))
    def run(labels, cache):
        engine = pipeline.Engine(data, labels, cfg, cache)
        fitted = pipeline.fit_selector(engine, allowed, user_embeddings, module_embeddings,
                                       user_embeddings, module_embeddings)
        score = engine.policy(allowed)["scores"]
        features = pipeline.action_features(data, excluded[:10], score,
                                           user_embeddings, module_embeddings)
        return fitted["e5"].predict(features.reshape(-1, features.shape[-1])), engine
    before, engine1 = run(data.y.copy(), tmp_path / "before")
    poisoned = data.y.copy()
    poisoned[excluded] = 1. - poisoned[excluded]
    after, engine2 = run(poisoned, tmp_path / "after")
    np.testing.assert_allclose(before, after, rtol=0, atol=1e-10)
    assert engine1.fit_audit and engine2.fit_audit
    for record in engine1.fit_audit + engine2.fit_audit:
        assert set(record["train_rows"]) <= set(allowed)
