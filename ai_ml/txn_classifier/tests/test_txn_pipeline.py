"""Unit tests for cleaning, splitting / leakage checks and label-noise flagging."""
import numpy as np
import pandas as pd
import pytest

import txn_pipeline as tp


def make_raw(n_accounts=12, per_account=10, seed=0):
    """Tiny synthetic frame with the same schema as the real data (3 clean templates)."""
    rng = np.random.RandomState(seed)
    kinds = [("SALARY CREDIT ACME CORP", "credit", "salary"),
             ("ONLINE PURCHASE GLOBEX", "debit", "shopping"),
             ("HOUSE RENT PAYMENT", "debit", "rent")]
    rows = []
    for a in range(n_accounts):
        for k in range(per_account):
            d, t, l = kinds[(a + k) % 3]
            rows.append({"txn_id": f"T{len(rows)}", "account_number": f"AC{a}", "txn_date": f"2026-01-{1 + k:02d}",
                         "description": d, "amount": float(rng.randint(100, 9000)), "txn_type": t, "label": l})
    return pd.DataFrame(rows)


# ---------------- cleaning ----------------
def test_to_template_masks_company_and_person():
    assert tp.to_template("SAL CR WAYNE ENT PAYROLL") == "SAL CR <CO> PAYROLL"
    assert tp.to_template("IMPS TRANSFER TO Diya") == "IMPS TRANSFER TO <NAME>"
    assert tp.to_template("UPI-NORTHWIND-ORDER") == "UPI-<CO>-ORDER"


def test_clean_removes_invalid_rows_and_never_changes_labels():
    raw = make_raw()
    bad = raw.iloc[:4].copy()
    bad["txn_id"] = ["X1", "X2", "X3", "X4"]
    bad.loc[bad.index[0], "amount"] = -5          # non-positive amount
    bad.loc[bad.index[1], "txn_type"] = "weird"   # invalid type
    bad.loc[bad.index[2], "txn_date"] = "not-a-date"
    bad.loc[bad.index[3], "label"] = None         # missing label
    dup = raw.iloc[[10]].copy()                   # exact duplicate txn_id
    dirty = pd.concat([raw, bad, dup], ignore_index=True)
    clean, log = tp.clean_transactions(dirty)
    assert len(clean) == len(raw)
    assert set(clean["label"]) == set(raw["label"])
    assert log["rows_removed"].sum() == len(dirty) - len(raw)


def test_clean_standardises_text_but_keeps_label_values():
    raw = make_raw()
    raw.loc[0, ["description", "txn_type", "label"]] = ["  salary   credit acme corp ", " CREDIT", " Salary "]
    clean, _ = tp.clean_transactions(raw)
    row = clean.iloc[0]
    assert row["desc_norm"] == "SALARY CREDIT ACME CORP"
    assert row["txn_type"] == "credit" and row["label"] == "salary"


def test_data_quality_report_flags_issues():
    raw = make_raw()
    raw.loc[0, "amount"] = -1
    rep = tp.data_quality_report(raw).set_index("check")
    assert rep.loc["amount <= 0", "n_rows_affected"] == 1
    assert rep.loc["duplicate txn_id", "status"] == "OK"


# ---------------- splitting / leakage ----------------
def test_group_split_has_no_account_overlap_and_covers_all_rows():
    clean, _ = tp.clean_transactions(make_raw(n_accounts=30))
    train, test = tp.group_train_test_split(clean, test_size=0.2, seed=1)
    assert not set(train["account_number"]) & set(test["account_number"])
    assert len(train) + len(test) == len(clean)
    assert len(test) > 0


def test_group_split_is_deterministic():
    clean, _ = tp.clean_transactions(make_raw(n_accounts=30))
    a = tp.group_train_test_split(clean, seed=7)[1].index
    b = tp.group_train_test_split(clean, seed=7)[1].index
    assert list(a) == list(b)


def test_assert_no_group_overlap_raises_on_leak():
    clean, _ = tp.clean_transactions(make_raw())
    with pytest.raises(AssertionError):
        tp.assert_no_group_overlap(clean.iloc[:50], clean.iloc[40:])


def test_leakage_audit_detects_shared_accounts():
    clean, _ = tp.clean_transactions(make_raw(n_accounts=20))
    tr, te = tp.group_train_test_split(clean, seed=3)
    ok = tp.leakage_audit(tr, te).set_index("check")
    assert ok.loc["accounts shared between train and test", "status"] == "OK"
    leaky = tp.leakage_audit(clean.iloc[:100], clean.iloc[50:150]).set_index("check")
    assert leaky.loc["accounts shared between train and test", "status"] == "REVIEW"


# ---------------- label-noise detection ----------------
def test_template_disagreement_flags_only_the_flipped_row():
    clean, _ = tp.clean_transactions(make_raw(n_accounts=20))
    i = clean.index[clean["label"] == "salary"][0]
    clean.loc[i, "label"] = "shopping"            # inject one wrong label
    flagged = tp.flag_template_disagreement(clean)
    assert flagged["sig_template"].sum() == 1 and bool(flagged.loc[i, "sig_template"])


def test_direction_conflict_flags_rare_direction():
    clean, _ = tp.clean_transactions(make_raw(n_accounts=30, per_account=20))
    i = clean.index[clean["label"] == "salary"][0]
    clean.loc[i, "txn_type"] = "debit"            # salary booked as debit
    flagged = tp.flag_direction_conflict(clean, min_share=0.05)
    assert bool(flagged.loc[i, "sig_direction"])
    assert flagged["sig_direction"].sum() == 1


def test_confident_learning_finds_injected_issue():
    classes = np.array(["a", "b"])
    labels = np.array(["a"] * 20 + ["b"] * 20 + ["a"])          # last row is really a 'b'
    proba = np.vstack([np.tile([0.95, 0.05], (20, 1)), np.tile([0.05, 0.95], (20, 1)), [[0.03, 0.97]]])
    issue, conf, joint, rate = tp.confident_learning(labels, proba, classes)
    assert issue.sum() == 1 and issue[-1]
    assert conf[-1] == "b" and 0 < rate < 0.05


def test_label_issue_report_marks_injected_noise_as_suspect():
    clean, _ = tp.clean_transactions(make_raw(n_accounts=40, per_account=12))
    i = clean.index[clean["label"] == "rent"][0]
    clean.loc[i, "label"] = "salary"
    rep, _, _ = tp.label_issue_report(clean)
    assert bool(rep.loc[i, "suspect"])
    assert rep["suspect"].sum() <= 3                 # and (almost) nothing else
    assert (rep["label"] == clean["label"]).all()   # labels untouched


def test_class_distribution_and_imbalance_summary():
    df = pd.DataFrame({"label": ["a"] * 8 + ["b"] * 2})
    cd = tp.class_distribution(df)
    assert cd.loc["a", "count"] == 8 and cd.loc["b", "imbalance_vs_largest"] == 4.0
    assert tp.imbalance_summary(df)["max_min_ratio"] == 4.0
