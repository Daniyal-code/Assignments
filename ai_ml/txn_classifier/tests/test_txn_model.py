"""Unit tests for the model module: pipeline, imbalance handling, metrics, explainability, persistence."""
import numpy as np
import pandas as pd
import pytest

import txn_model as tm
import txn_pipeline as tp
from tests.test_txn_pipeline import make_raw


@pytest.fixture(scope="module")
def data():
    clean, _ = tp.clean_transactions(make_raw(n_accounts=60, per_account=12))
    train, test = tp.group_train_test_split(clean, seed=0)
    return clean, train, test


# ---------------- pipeline ----------------
@pytest.mark.parametrize("algo", ["logreg", "linear_svc", "complement_nb", "hgb"])
def test_every_algorithm_fits_and_predicts_known_classes(data, algo):
    _, train, test = data
    model = tm.fit_model(tm.ModelConfig(algorithm=algo), train)
    pred = model.predict(test)
    assert len(pred) == len(test) and set(pred) <= set(train["label"])
    assert (pred == test["label"]).mean() > 0.95    # templates are trivially separable here


def test_amount_and_date_features_are_optional(data):
    _, train, test = data
    cfg = tm.ModelConfig(use_amount=True, use_date=True)
    model = tm.fit_model(cfg, train)
    names = model.named_steps["features"].get_feature_names_out()
    assert any(n.startswith("amount__") for n in names) and any(n.startswith("date__") for n in names)
    assert len(model.predict(test)) == len(test)


def test_model_handles_unseen_description_without_crashing(data):
    _, train, _ = data
    model = tm.fit_model(tm.ModelConfig(), train)
    row = train.iloc[[0]].copy()
    row["desc_norm"] = "COMPLETELY NEW MERCHANT 123"
    assert model.predict(row)[0] in set(train["label"])


# ---------------- imbalance ----------------
def test_random_oversample_balances_classes_and_only_duplicates_existing_rows():
    df = pd.DataFrame({"label": ["a"] * 10 + ["b"] * 3, "x": range(13)})
    out = tm.random_oversample(df)
    assert out["label"].value_counts().to_dict() == {"a": 10, "b": 10}
    assert set(out["x"]) == set(df["x"])


def test_class_weight_balanced_is_set_only_when_requested():
    assert tm.build_pipeline(tm.ModelConfig(imbalance="class_weight")).named_steps["clf"].class_weight == "balanced"
    assert tm.build_pipeline(tm.ModelConfig(imbalance="none")).named_steps["clf"].class_weight is None


def test_oversampling_happens_inside_fit_so_validation_data_is_untouched(data):
    _, train, test = data
    before = len(test)
    tm.fit_model(tm.ModelConfig(imbalance="oversample"), train)
    assert len(test) == before and len(train) == len(train.drop_duplicates("txn_id"))


# ---------------- validation ----------------
def test_grouped_cv_never_shares_accounts_between_folds(data):
    from sklearn.model_selection import StratifiedGroupKFold
    clean, _, _ = data
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    for tr, va in sgkf.split(clean, clean["label"], groups=clean["account_number"]):
        assert not set(clean.iloc[tr]["account_number"]) & set(clean.iloc[va]["account_number"])


def test_cross_validate_returns_oof_prediction_for_every_row(data):
    _, train, _ = data
    summary, oof = tm.cross_validate_grouped(train, tm.ModelConfig(), n_splits=3)
    assert oof.notna().all() and 0.9 < summary["accuracy_mean"] <= 1.0


# ---------------- metrics ----------------
def test_per_class_report_matches_hand_computation():
    y = ["a", "a", "a", "b", "b", "c"]
    p = ["a", "a", "b", "b", "b", "a"]
    rep = tm.per_class_report(y, p)
    assert rep.loc["a", "precision"] == pytest.approx(2 / 3, abs=1e-3)   # 2 of 3 predicted a are right
    assert rep.loc["a", "recall"] == pytest.approx(2 / 3, abs=1e-3)
    assert rep.loc["b", "recall"] == 1.0 and rep.loc["c", "recall"] == 0.0
    assert rep.loc["accuracy", "f1"] == pytest.approx(4 / 6, abs=1e-3)
    assert rep.loc["weighted avg", "support"] == 6


def test_confusion_df_counts():
    cm = tm.confusion_df(["a", "a", "b"], ["a", "b", "b"])
    assert cm.loc["a", "a"] == 1 and cm.loc["a", "b"] == 1 and cm.loc["b", "b"] == 1 and cm.values.sum() == 3


def test_reference_noise_flags_use_train_statistics_only(data):
    _, train, test = data
    test = test.copy()
    i = test.index[test["label"] == "salary"][0]
    test.loc[i, "label"] = "shopping"
    flags = tm.reference_noise_flags(train, test)
    assert bool(flags.loc[i]) and flags.sum() >= 1


def test_noise_aware_metrics_splits_errors(data):
    _, train, test = data
    test = test.copy()
    i = test.index[0]
    test.loc[i, "label"] = "rent" if test.loc[i, "label"] != "rent" else "salary"
    pred = tm.fit_model(tm.ModelConfig(), train).predict(test)
    flags = tm.reference_noise_flags(train, test)
    out = tm.noise_aware_metrics(test, pred, flags)
    assert out.attrs["errors_total"] == out.attrs["errors_on_likely_noisy"] + out.attrs["errors_on_clean"]
    assert out.loc[1, "accuracy"] >= out.loc[0, "accuracy"]


def test_segment_errors_counts_errors_per_segment():
    test = pd.DataFrame({"label": ["a", "a", "b", "b"], "txn_type": ["debit", "debit", "credit", "credit"]})
    seg = tm.segment_errors(test, ["a", "b", "b", "b"], "txn_type")
    assert seg.loc["debit", "errors"] == 1 and seg.loc["credit", "errors"] == 0


def test_bootstrap_ci_contains_point_estimate(data):
    _, train, test = data
    pred = tm.fit_model(tm.ModelConfig(), train).predict(test)
    point, lo, hi = tm.bootstrap_ci(test, pred, n_boot=100)
    assert lo <= point <= hi or hi == pytest.approx(point)


# ---------------- explainability ----------------
def test_linear_shap_is_additive_and_reproduces_the_logit(data):
    """base + sum(SHAP) must equal the model's raw score (logit) for every class."""
    _, train, test = data
    model = tm.fit_model(tm.ModelConfig(C=1.0), train)
    ex = tm.LinearExplainer(model, train)
    rows = test.iloc[:15]
    sv = ex.shap_values(rows)                                 # [rows, classes, features]
    recon = ex.base[None, :] + sv.sum(axis=2)
    np.testing.assert_allclose(recon, ex.logits(rows), atol=1e-8)


def test_explain_row_returns_top_features_sorted_by_magnitude(data):
    _, train, test = data
    model = tm.fit_model(tm.ModelConfig(), train)
    ex = tm.LinearExplainer(model, train)
    row = test.iloc[[0]]
    out = ex.explain_row(row, model.predict(row)[0], top_k=5)
    assert len(out) <= 5 and (out["shap"].abs().diff().dropna() <= 1e-12).all()


def test_linear_explainer_rejects_non_linear_models(data):
    _, train, _ = data
    model = tm.fit_model(tm.ModelConfig(algorithm="hgb"), train)
    with pytest.raises(TypeError):
        tm.LinearExplainer(model, train)


# ---------------- persistence ----------------
def test_save_and_load_roundtrip_gives_identical_predictions(data, tmp_path):
    _, train, test = data
    cfg = tm.ModelConfig()
    model = tm.fit_model(cfg, train)
    path = str(tmp_path / "m.joblib")
    tm.save_model(model, path, cfg)
    loaded, blob = tm.load_model(path)
    assert list(loaded.predict(test)) == list(model.predict(test))
    assert blob["config"]["algorithm"] == "logreg"


def test_explain_row_merges_direction_columns_into_one_feature(data):
    _, train, test = data
    model = tm.fit_model(tm.ModelConfig(), train)
    ex = tm.LinearExplainer(model, train)
    row = test.iloc[[0]]
    out = ex.explain_row(row, model.predict(row)[0], top_k=50)
    assert out["feature"].str.startswith("direction=").sum() == 1
