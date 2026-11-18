"""
txn_pipeline.py
================
Reusable building blocks for the bank-transaction classification assignment.

Scope of this version (steps 1-7 of the brief):
    1. EDA                      -> `data_quality_report`, `class_distribution`, `plot_*`, ...
    2. Data cleaning            -> `clean_transactions`
    3. Data analysis            -> `type_label_crosstab`, `amount_signal_tests`, `template_label_table`, ...
    4. Leakage avoidance        -> `group_train_test_split`, `assert_no_group_overlap`, `leakage_audit`
    5-7. Label quality / noise  -> `label_issue_report`, `noise_structure`, ...

Design rules
------------
* Every function is a small, pure function (DataFrame in -> DataFrame/dict out), so it can be
  called cell by cell in a notebook AND unit-tested later.
* Nothing here modifies labels. Label-noise functions only *flag* rows.
* No randomness without a `seed` argument.

Typical notebook usage
----------------------
    import txn_pipeline as tp
    raw = tp.load_data(DATA_PATH)
    df, log = tp.clean_transactions(raw)
    tp.data_quality_report(raw)
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline


# Constants

REQUIRED_COLUMNS = ["txn_id", "account_number", "txn_date", "description", "amount", "txn_type", "label"]
VALID_TXN_TYPES = ("debit", "credit")
RANDOM_SEED = 42

# Counterparty (company) names seen in the synthetic descriptions. Used ONLY to build a
# "template" of a description (e.g. "SAL CR <CO> PAYROLL") for analysis - never as a model feature.
DEFAULT_ENTITIES = (
    "ACME CORP", "GLOBEX", "INITECH", "UMBRELLA", "STARK IND",
    "WAYNE ENT", "HOOLI", "SIRIUS CYB", "VEHEMENT CAP", "NORTHWIND",
)

# Neutral palette for plots (one hue for magnitude, grey for context, one accent for "problem").
C_MAIN, C_GREY, C_ACCENT = "#2a6f97", "#b8bfc6", "#d1495b"



# 1. Loading + data-quality report

def load_data(path: str) -> pd.DataFrame:
    """Read the CSV as-is (no cleaning) and verify the required columns exist."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run a battery of raw-data checks and return one row per check:
    check | n_rows_affected | status ('OK' / 'ISSUE').
    Works on the RAW frame (before cleaning) so problems are not hidden.
    """
    d = df.copy()
    dates = pd.to_datetime(d["txn_date"], errors="coerce")
    amounts = pd.to_numeric(d["amount"], errors="coerce")
    desc = d["description"].astype("string")
    checks = {
        "null cells (any column)": int(d.isna().any(axis=1).sum()),
        "duplicate txn_id": int(d["txn_id"].duplicated().sum()),
        "duplicate rows ignoring txn_id": int(d.drop(columns="txn_id").duplicated().sum()),
        "unparseable dates": int(dates.isna().sum()),
        "non-numeric amount": int(amounts.isna().sum()),
        "amount <= 0": int((amounts <= 0).sum()),
        "txn_type not in {debit, credit}": int((~d["txn_type"].astype(str).str.strip().str.lower()
                                               .isin(VALID_TXN_TYPES)).sum()),
        "description with leading/trailing/double spaces": int(
            (desc != desc.str.strip().str.replace(r"\s+", " ", regex=True)).sum()),
        "description inconsistent casing of same text": int(
            desc.str.strip().nunique() - desc.str.strip().str.upper().nunique()),
        "label with spaces/mixed case": int(
            (d["label"].astype(str) != d["label"].astype(str).str.strip().str.lower()).sum()),
        "future-dated txns (> today)": int((dates > pd.Timestamp.today()).sum()),
    }
    out = pd.DataFrame({"check": list(checks), "n_rows_affected": list(checks.values())})
    out["status"] = np.where(out["n_rows_affected"] == 0, "OK", "ISSUE")
    return out



# 2. Cleaning

def to_template(description: str, entities: Sequence[str] = DEFAULT_ENTITIES) -> str:
    """
    Collapse the variable parts of a description into placeholders.
        'SAL CR WAYNE ENT PAYROLL'    -> 'SAL CR <CO> PAYROLL'
        'IMPS TRANSFER TO Diya'       -> 'IMPS TRANSFER TO <NAME>'
    Person names are detected as Title-case words; companies from `entities`.
    """
    s = re.sub(r"\b[A-Z][a-z]+\b", "<NAME>", description)
    for e in sorted(entities, key=len, reverse=True):
        s = re.sub(rf"\b{re.escape(e)}\b", "<CO>", s)
    return re.sub(r"\s+", " ", s).strip()


def clean_transactions(raw: pd.DataFrame, drop_duplicates: bool = True,
                       entities: Sequence[str] = DEFAULT_ENTITIES) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Standardise types/text and drop unusable rows. LABELS ARE NEVER CHANGED (only trimmed/lower-cased).

    Returns
    -------
    clean : DataFrame with extra columns
        desc_norm  - upper-cased, whitespace-collapsed description (model input)
        template   - description with company/person names masked (analysis only)
        month, dow, day_of_month - date parts
    log   : DataFrame(step, rows_before, rows_after, rows_removed) - an audit trail.
    """
    df = raw.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    steps = []

    def record(name: str, before: int, after: int) -> None:
        steps.append({"step": name, "rows_before": before, "rows_after": after, "rows_removed": before - after})

    # --- text standardisation (no row loss)
    for col in ("txn_id", "account_number"):
        df[col] = df[col].astype(str).str.strip().str.upper()
    df["description"] = df["description"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df["txn_type"] = df["txn_type"].astype(str).str.strip().str.lower()
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    # --- type conversion
    df["txn_date"] = pd.to_datetime(df["txn_date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    n = len(df)
    df = df.dropna(subset=REQUIRED_COLUMNS)
    df = df[~df["label"].isin(["", "nan", "none"])]
    record("drop null / unparseable key fields", n, len(df))

    n = len(df)
    df = df[df["amount"] > 0]
    record("drop amount <= 0", n, len(df))

    n = len(df)
    df = df[df["txn_type"].isin(VALID_TXN_TYPES)]
    record("drop invalid txn_type", n, len(df))

    if drop_duplicates:
        n = len(df)
        df = df.drop_duplicates(subset="txn_id", keep="first")
        record("drop duplicate txn_id", n, len(df))
        n = len(df)
        df = df.drop_duplicates(subset=[c for c in REQUIRED_COLUMNS if c != "txn_id"], keep="first")
        record("drop duplicate rows (ignoring txn_id)", n, len(df))

    # --- derived columns
    df["desc_norm"] = df["description"].str.upper()
    df["template"] = df["description"].map(lambda s: to_template(s, entities))
    df["month"] = df["txn_date"].dt.month
    df["dow"] = df["txn_date"].dt.dayofweek
    df["day_of_month"] = df["txn_date"].dt.day
    df = df.reset_index(drop=True)
    return df, pd.DataFrame(steps)



# 3. Analysis helpers (tables + statistical tests)

def class_distribution(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """Counts, share and imbalance ratio vs. the largest class."""
    vc = df[label_col].value_counts()
    out = pd.DataFrame({"count": vc, "share_pct": (vc / vc.sum() * 100).round(2)})
    out["imbalance_vs_largest"] = (vc.max() / vc).round(2)
    return out


def imbalance_summary(df: pd.DataFrame, label_col: str = "label") -> Dict[str, float]:
    """Headline imbalance numbers: max/min ratio and normalised entropy (1 = perfectly balanced)."""
    p = df[label_col].value_counts(normalize=True).to_numpy()
    entropy = float(-(p * np.log(p)).sum() / np.log(len(p)))
    counts = df[label_col].value_counts()
    return {"n_classes": int(len(p)), "largest": int(counts.max()), "smallest": int(counts.min()),
            "max_min_ratio": round(counts.max() / counts.min(), 2), "normalised_entropy": round(entropy, 3)}


def type_label_crosstab(df: pd.DataFrame, normalize: bool = False) -> pd.DataFrame:
    """label x txn_type counts (or row-share if normalize=True)."""
    ct = pd.crosstab(df["label"], df["txn_type"])
    return (ct.div(ct.sum(axis=1), axis=0).round(3)) if normalize else ct


def amount_by_label(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("label")["amount"].describe().round(0)


def amount_signal_tests(df: pd.DataFrame) -> Dict[str, float]:
    """
    Is `amount` informative for the label?
      * Kruskal-Wallis across classes (H0: same distribution in every class)
      * KS test of amount vs Uniform(min, max)   (H0: amount ~ uniform)
    A large p-value on Kruskal + a uniform amount => amount is (almost) pure noise for this task.
    """
    groups = [g["amount"].to_numpy() for _, g in df.groupby("label")]
    kw = stats.kruskal(*groups)
    lo, hi = df["amount"].min(), df["amount"].max()
    ks = stats.kstest(df["amount"], "uniform", args=(lo, hi - lo))
    return {"kruskal_H": float(kw.statistic), "kruskal_p": float(kw.pvalue),
            "ks_uniform_stat": float(ks.statistic), "ks_uniform_p": float(ks.pvalue)}


def date_signal_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Chi-square independence tests of label vs month / day-of-week / day-of-month-bucket."""
    rows = []
    buckets = pd.cut(df["day_of_month"], [0, 5, 10, 20, 25, 31], labels=["1-5", "6-10", "11-20", "21-25", "26-31"])
    for name, series in [("month", df["month"]), ("day_of_week", df["dow"]), ("day_of_month_bucket", buckets)]:
        chi2, p, dof, _ = stats.chi2_contingency(pd.crosstab(series, df["label"]))
        rows.append({"feature": name, "chi2": round(chi2, 1), "dof": dof, "p_value": round(p, 4)})
    return pd.DataFrame(rows)


def template_label_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every description template: rows, majority label, majority share, number of distinct labels,
    and the minority labels. The core table for seeing "same text -> different labels".
    """
    def summarise(g: pd.DataFrame) -> pd.Series:
        vc = g["label"].value_counts()
        minority = {k: int(v) for k, v in vc.iloc[1:].items()}
        return pd.Series({"n": len(g), "majority_label": vc.index[0],
                          "majority_share": round(vc.iloc[0] / len(g), 3),
                          "n_labels": len(vc), "minority_labels": minority})
    return (df.groupby("template").apply(summarise, include_groups=False)
              .sort_values("n", ascending=False))


def account_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-account rows, #labels used, #templates used. Shows how many rows share an account."""
    return df.groupby("account_number").agg(
        n_txns=("txn_id", "size"), n_labels=("label", "nunique"),
        n_templates=("template", "nunique"), first=("txn_date", "min"), last=("txn_date", "max"))


# Illustrative real-world plausibility rules (min, max) per label. Edit freely.
DOMAIN_AMOUNT_RULES = {
    "salary": (5_000, None),     # salaries below Rs 5k are implausible
    "rent": (2_000, None),
    "other": (None, 10_000),     # bank fees / service charges / misc are small
}


def domain_sanity_checks(df: pd.DataFrame, rules: Optional[Dict[str, tuple]] = None) -> pd.DataFrame:
    """
    Banking-domain plausibility: count rows whose amount violates a rule of thumb for their label,
    plus direction (credit/debit) anomalies. Many violations => amount is synthetic / not usable as a signal.
    """
    rules = rules or DOMAIN_AMOUNT_RULES
    rows = []
    for label, (lo, hi) in rules.items():
        sub = df[df["label"] == label]
        bad = pd.Series(False, index=sub.index)
        if lo is not None:
            bad |= sub["amount"] < lo
        if hi is not None:
            bad |= sub["amount"] > hi
        rows.append({"rule": f"{label}: amount outside [{lo}, {hi}]", "n_label_rows": len(sub),
                     "n_violations": int(bad.sum()), "pct": round(bad.mean() * 100, 1)})
    fee = df[df["template"].isin(["BANK FEE", "SERVICE CHARGE"])]
    rows.append({"rule": "bank fee / service charge booked as CREDIT", "n_label_rows": len(fee),
                 "n_violations": int((fee["txn_type"] == "credit").sum()),
                 "pct": round((fee["txn_type"] == "credit").mean() * 100, 1) if len(fee) else 0.0})
    return pd.DataFrame(rows)



# 4. Leakage: account-grouped split + audit

def group_train_test_split(df: pd.DataFrame, group_col: str = "account_number", label_col: str = "label",
                           test_size: float = 0.2, seed: int = RANDOM_SEED) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split so that NO account appears in both train and test, while keeping class proportions similar.
    Uses StratifiedGroupKFold (one fold = test). Returns (train_df, test_df) with original index kept.
    """
    n_splits = max(2, int(round(1 / test_size)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, test_idx = next(sgkf.split(df, df[label_col], groups=df[group_col]))
    train, test = df.iloc[train_idx], df.iloc[test_idx]
    assert_no_group_overlap(train, test, group_col)
    return train, test


def assert_no_group_overlap(train: pd.DataFrame, test: pd.DataFrame, group_col: str = "account_number") -> None:
    """Raise if any group (account) is present in both splits."""
    overlap = set(train[group_col]) & set(test[group_col])
    if overlap:
        raise AssertionError(f"{len(overlap)} {group_col} values appear in both train and test, e.g. {sorted(overlap)[:3]}")


def account_memorisation_probe(df: pd.DataFrame, group_col: str = "account_number", label_col: str = "label",
                               test_size: float = 0.2, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Does knowing the ACCOUNT alone help predict the label? ("account fingerprint" leakage)
    Predict each test row with its account's most frequent TRAIN label (global majority if unseen).
      * random row split  -> test accounts are seen in train  (leak-prone)
      * grouped split     -> test accounts are unseen         (honest)
    If random-split accuracy is clearly above the majority baseline, accounts leak label info.
    """
    rng = np.random.RandomState(seed)
    majority = df[label_col].mode()[0]

    def score(train: pd.DataFrame, test: pd.DataFrame) -> float:
        acc_mode = train.groupby(group_col)[label_col].agg(lambda s: s.mode()[0])
        pred = test[group_col].map(acc_mode).fillna(majority)
        return float((pred == test[label_col]).mean())

    perm = rng.permutation(len(df))
    cut = int(len(df) * (1 - test_size))
    rand_train, rand_test = df.iloc[perm[:cut]], df.iloc[perm[cut:]]
    g_train, g_test = group_train_test_split(df, group_col, label_col, test_size, seed)
    return pd.DataFrame([
        {"strategy": "majority-class baseline", "accuracy": float((g_test[label_col] == majority).mean())},
        {"strategy": "account-only, RANDOM split", "accuracy": score(rand_train, rand_test)},
        {"strategy": "account-only, GROUPED split", "accuracy": score(g_train, g_test)},
    ]).round(4)


def id_label_association(df: pd.DataFrame, id_cols: Iterable[str] = ("txn_id",), bins: int = 10) -> pd.DataFrame:
    """Chi-square test of label vs binned running-id columns. Small p => the ID order encodes the label (leak)."""
    rows = []
    for col in id_cols:
        num = pd.to_numeric(df[col].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
        b = pd.qcut(num.rank(method="first"), bins, labels=False)
        chi2, p, dof, _ = stats.chi2_contingency(pd.crosstab(b, df["label"]))
        rows.append({"column": col, "chi2": round(chi2, 1), "dof": dof, "p_value": round(p, 4),
                     "verdict": "possible leak" if p < 0.01 else "no association"})
    return pd.DataFrame(rows)


def leakage_audit(train: pd.DataFrame, test: pd.DataFrame, group_col: str = "account_number") -> pd.DataFrame:
    """
    Post-split checklist, one row per check (status OK / REVIEW).
    'description overlap' is *expected* (the same generic texts exist for every bank customer);
    what must be zero is overlap of accounts, txn_ids and exact duplicate transactions.
    """
    key = ["description", "amount", "txn_type"]
    exact_dups = pd.merge(train[key], test[key], how="inner").shape[0]
    rows = [
        ("accounts shared between train and test", len(set(train[group_col]) & set(test[group_col])), "must be 0"),
        ("txn_ids shared between train and test", len(set(train["txn_id"]) & set(test["txn_id"])), "must be 0"),
        ("identical (description, amount, txn_type) in both", int(exact_dups), "must be 0"),
        ("% test rows whose exact description appears in train",
         round(test["description"].isin(set(train["description"])).mean() * 100, 1), "expected high (templated text)"),
        ("% test rows whose TEMPLATE appears in train",
         round(test["template"].isin(set(train["template"])).mean() * 100, 1), "expected ~100"),
    ]
    out = pd.DataFrame(rows, columns=["check", "value", "expectation"])
    must0 = out["expectation"] == "must be 0"
    out["status"] = np.where(must0 & (out["value"] != 0), "REVIEW", "OK")
    return out


def class_balance_across_split(train: pd.DataFrame, test: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """Share of each class in train vs test - verifies the grouped split stayed stratified."""
    a = train[label_col].value_counts(normalize=True).rename("train_pct") * 100
    b = test[label_col].value_counts(normalize=True).rename("test_pct") * 100
    out = pd.concat([a, b], axis=1).fillna(0).round(2)
    out["abs_diff"] = (out["train_pct"] - out["test_pct"]).abs().round(2)
    return out



# 5. Label-noise detection (three independent signals)

def flag_template_disagreement(df: pd.DataFrame, min_support: int = 5, min_share: float = 0.8) -> pd.DataFrame:
    """
    Signal A - "same text, different label".
    For each template with >= `min_support` rows whose majority label has share >= `min_share`,
    rows carrying a different label are flagged. Adds: tmpl_majority, tmpl_majority_share, sig_template.
    """
    out = df.copy()
    grp = out.groupby("template")["label"]
    size = grp.transform("size")
    maj = grp.transform(lambda s: s.value_counts().index[0])
    share = grp.transform(lambda s: s.value_counts(normalize=True).iloc[0])
    out["tmpl_majority"], out["tmpl_majority_share"] = maj, share.round(3)
    out["sig_template"] = (size >= min_support) & (share >= min_share) & (out["label"] != maj)
    return out


def flag_direction_conflict(df: pd.DataFrame, min_share: float = 0.05) -> pd.DataFrame:
    """
    Signal B - banking-logic check on credit/debit direction.
    Per label, find the share of rows in each direction. A row whose (label, direction) combination
    covers < `min_share` of that label is flagged (e.g. 'salary' booked as a debit).
    Adds: dir_share_in_label, sig_direction.
    """
    out = df.copy()
    share = out.groupby("label")["txn_type"].transform(lambda s: s.map(s.value_counts(normalize=True)))
    out["dir_share_in_label"] = share.round(3)
    out["sig_direction"] = share < min_share
    return out


def make_text_model(C: float = 10.0) -> Pipeline:
    """TF-IDF (word 1-2gram + char 2-5gram) -> multinomial LogisticRegression. Used for noise detection only."""
    feats = FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), lowercase=True, token_pattern=r"[A-Za-z0-9<>]+")),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), lowercase=True, sublinear_tf=True)),
    ])
    return Pipeline([("tfidf", feats), ("clf", LogisticRegression(C=C, max_iter=3000))])


def model_text(df: pd.DataFrame) -> pd.Series:
    """Model input text = normalised description + a direction token."""
    return df["desc_norm"] + " DIR_" + df["txn_type"].str.upper()


def out_of_fold_proba(df: pd.DataFrame, n_splits: int = 5, seed: int = RANDOM_SEED,
                      group_col: str = "account_number") -> Tuple[np.ndarray, np.ndarray]:
    """
    Out-of-fold class probabilities with account-grouped, stratified CV, so every row is scored by a
    model that never saw it (nor its account). Returns (proba [n, K], classes [K]).
    """
    X, y = model_text(df).to_numpy(), df["label"].to_numpy()
    classes = np.array(sorted(df["label"].unique()))
    proba = np.zeros((len(df), len(classes)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in sgkf.split(X, y, groups=df[group_col]):
        m = make_text_model().fit(X[tr], y[tr])
        p = m.predict_proba(X[va])
        proba[np.ix_(va, [int(np.where(classes == c)[0][0]) for c in m.classes_])] = p
    return proba, classes


def confident_learning(labels: np.ndarray, proba: np.ndarray, classes: np.ndarray):
    """
    Compact implementation of Confident Learning (Northcutt et al.).
    * t_j = mean predicted prob of class j over rows labelled j (per-class threshold)
    * a row's 'confident class' = argmax over classes whose prob >= t_j
    * confident joint C[given, confident]; off-diagonal rows are label issues
    Returns (issue_mask, confident_class, confident_joint DataFrame, estimated_noise_rate).
    """
    idx = {c: i for i, c in enumerate(classes)}
    y = np.array([idx[l] for l in labels])
    K = len(classes)
    t = np.array([proba[y == j, j].mean() for j in range(K)])
    conf_class = np.full(len(y), -1)
    for i in range(len(y)):
        ok = np.where(proba[i] >= t)[0]
        if len(ok):
            conf_class[i] = ok[np.argmax(proba[i, ok])]
    joint = np.zeros((K, K), dtype=int)
    for g, c in zip(y, conf_class):
        if c >= 0:
            joint[g, c] += 1
    issue = (conf_class >= 0) & (conf_class != y)
    noise_rate = issue.sum() / max((conf_class >= 0).sum(), 1)
    joint_df = pd.DataFrame(joint, index=pd.Index(classes, name="given"), columns=pd.Index(classes, name="confident"))
    return issue, np.where(conf_class >= 0, classes[np.clip(conf_class, 0, None)], None), joint_df, float(noise_rate)


def label_issue_report(df: pd.DataFrame, min_support: int = 5, min_share: float = 0.8,
                       direction_share: float = 0.05, n_splits: int = 5, seed: int = RANDOM_SEED):
    """
    Combine the three signals into one per-row report. Labels are NOT modified.

    Added columns
    -------------
    sig_template, sig_direction, sig_model   booleans (A, B, C)
    n_signals                                how many of the three fire (0-3)
    p_given_label                            out-of-fold probability of the label the row carries
    model_label                              the label the model confidently prefers
    noise_score                              1 - p_given_label (higher = more suspicious)
    suspect                                  n_signals >= 2  (high-precision shortlist for human review)
    any_signal                               n_signals >= 1  (high-recall list)

    Returns (report_df, confident_joint_df, estimated_noise_rate_from_model)
    """
    out = flag_template_disagreement(df, min_support, min_share)
    out = flag_direction_conflict(out, direction_share)
    proba, classes = out_of_fold_proba(out, n_splits=n_splits, seed=seed)
    issue, conf_class, joint, rate = confident_learning(out["label"].to_numpy(), proba, classes)
    idx = {c: i for i, c in enumerate(classes)}
    out["p_given_label"] = [proba[i, idx[l]] for i, l in enumerate(out["label"])]
    out["p_given_label"] = out["p_given_label"].round(4)
    out["model_label"] = pd.Series(conf_class, index=out.index)
    out["sig_model"] = issue
    out["noise_score"] = (1 - out["p_given_label"]).round(4)
    out["n_signals"] = out[["sig_template", "sig_direction", "sig_model"]].sum(axis=1)
    out["suspect"] = out["n_signals"] >= 2
    out["any_signal"] = out["n_signals"] >= 1
    return out, joint, rate


def signal_agreement(report: pd.DataFrame) -> pd.DataFrame:
    """How many rows each signal flags, and pairwise overlap - shows whether signals corroborate each other."""
    sig = ["sig_template", "sig_direction", "sig_model"]
    rows = [{"signal": s, "n_flagged": int(report[s].sum()), "pct": round(report[s].mean() * 100, 2)} for s in sig]
    rows.append({"signal": "any signal", "n_flagged": int(report["any_signal"].sum()), "pct": round(report["any_signal"].mean() * 100, 2)})
    rows.append({"signal": ">=2 signals (suspect)", "n_flagged": int(report["suspect"].sum()), "pct": round(report["suspect"].mean() * 100, 2)})
    overlap = report[sig].astype(int).T.dot(report[sig].astype(int))
    return pd.DataFrame(rows).set_index("signal"), overlap


def noise_by_class(report: pd.DataFrame, flag_col: str = "sig_template") -> pd.DataFrame:
    """
    Per GIVEN label: how many rows are flagged and the flag rate. Small classes usually show a larger rate
    because the same absolute number of random flips is a larger fraction of them.
    """
    g = report.groupby("label")[flag_col].agg(["sum", "size"])
    g.columns = ["n_flagged", "n_rows"]
    g["flag_rate_pct"] = (g["n_flagged"] / g["n_rows"] * 100).round(1)
    return g.sort_values("flag_rate_pct", ascending=False)


def flip_matrix(report: pd.DataFrame, flag_col: str = "sig_template") -> pd.DataFrame:
    """For rows flagged by template disagreement: expected (template majority) label -> label actually given."""
    f = report[report[flag_col]]
    return pd.crosstab(f["tmpl_majority"], f["label"]).rename_axis(index="expected (template majority)", columns="given label")


def noise_structure(report: pd.DataFrame, flag_col: str = "sig_template", n_perm: int = 2000,
                    seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Is the noise RANDOM or SYSTEMATIC? Tests whether flagged rows depend on amount, direction, month, account.
    Large p-values => flags look like random flips (independent of every observed feature).
    """
    rng = np.random.RandomState(seed)
    flag = report[flag_col].astype(int).to_numpy()
    rows = []
    # amount
    u = stats.mannwhitneyu(report.loc[report[flag_col], "amount"], report.loc[~report[flag_col], "amount"])
    rows.append({"test": "amount: flagged vs clean (Mann-Whitney)", "p_value": float(u.pvalue)})
    # direction & month
    for name, col in [("direction (txn_type)", "txn_type"), ("month", "month")]:
        chi2, p, _, _ = stats.chi2_contingency(pd.crosstab(report[col], report[flag_col]))
        rows.append({"test": f"{name}: chi-square independence", "p_value": float(p)})
    # account concentration via permutation test on variance of per-account flag counts
    acc = pd.factorize(report["account_number"])[0]
    def var_counts(f):
        return np.bincount(acc, weights=f).var()
    obs = var_counts(flag)
    perm = np.array([var_counts(rng.permutation(flag)) for _ in range(n_perm)])
    rows.append({"test": "account clustering of flags (permutation)", "p_value": float((perm >= obs).mean())})
    out = pd.DataFrame(rows)
    out["reading"] = np.where(out["p_value"] < 0.01, "systematic pattern", "consistent with random")
    return out


def flip_destination_uniformity(report: pd.DataFrame, flag_col: str = "sig_template") -> Dict[str, float]:
    """
    Where do wrong labels LAND? Chi-square of the given labels of flagged rows against a uniform spread
    over the classes. A high p-value means flips are spread evenly (random noise) rather than funnelled
    into one or two classes (systematic confusion, e.g. everything mislabelled as 'other').
    """
    counts = report.loc[report[flag_col], "label"].value_counts().reindex(sorted(report["label"].unique()), fill_value=0)
    chi2, p = stats.chisquare(counts.to_numpy())
    return {"n_flagged": int(counts.sum()), "chi2_vs_uniform": float(chi2), "p_value": float(p),
            "min_class": int(counts.min()), "max_class": int(counts.max())}


def review_queue(report: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    """Highest-priority rows for manual relabelling: more signals first, then higher noise score."""
    cols = ["txn_id", "account_number", "description", "txn_type", "amount", "label",
            "tmpl_majority", "model_label", "p_given_label", "n_signals"]
    return (report[report["any_signal"]].sort_values(["n_signals", "noise_score"], ascending=False)[cols].head(top_n))


def split_suspects(report: pd.DataFrame, min_signals: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(kept, quarantined). Does NOT edit labels; just partitions rows so the choice can be evaluated later."""
    mask = report["n_signals"] >= min_signals
    return report[~mask].copy(), report[mask].copy()



# Plots (matplotlib only; each returns the Figure so notebooks / reports can save it)

def _style(ax, grid_axis: str = "y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color="#e6e9ec", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_class_distribution(df: pd.DataFrame):
    import matplotlib.pyplot as plt
    vc = df["label"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(vc.index[::-1], vc.values[::-1], color=C_MAIN, height=0.6)
    for b, v in zip(bars, vc.values[::-1]):
        ax.text(v + vc.max() * 0.01, b.get_y() + b.get_height() / 2, f"{v}  ({v / vc.sum():.1%})", va="center", fontsize=9)
    ax.set_xlim(0, vc.max() * 1.18)
    ax.set_title("Transactions per class")
    ax.set_xlabel("rows")
    _style(ax, "x")
    fig.tight_layout()
    return fig


def plot_amount_by_label(df: pd.DataFrame):
    import matplotlib.pyplot as plt
    order = df.groupby("label")["amount"].median().sort_values().index
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot([df.loc[df["label"] == l, "amount"] for l in order], vert=False, tick_labels=order,
               patch_artist=True, boxprops=dict(facecolor="#dbe7f0", edgecolor=C_MAIN),
               medianprops=dict(color=C_ACCENT), whiskerprops=dict(color=C_GREY), capprops=dict(color=C_GREY),
               flierprops=dict(marker=".", markerfacecolor=C_GREY, markeredgecolor=C_GREY))
    ax.set_title("Amount by label - distributions overlap almost completely")
    ax.set_xlabel("amount")
    _style(ax, "x")
    fig.tight_layout()
    return fig


def plot_direction_by_label(df: pd.DataFrame):
    import matplotlib.pyplot as plt
    ct = type_label_crosstab(df, normalize=True)
    ct = ct.sort_values("credit")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(ct.index, ct["credit"], color=C_MAIN, label="credit", height=0.6)
    ax.barh(ct.index, ct["debit"], left=ct["credit"], color=C_GREY, label="debit", height=0.6,
            edgecolor="white", linewidth=1.5)
    ax.set_xlim(0, 1)
    ax.set_title("Direction mix per label (share of rows)")
    ax.legend(ncol=2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.22))
    _style(ax, "x")
    fig.tight_layout()
    return fig


def plot_monthly_volume(df: pd.DataFrame):
    import matplotlib.pyplot as plt
    m = df.groupby(df["txn_date"].dt.to_period("M")).size()
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.bar(m.index.astype(str), m.values, color=C_MAIN, width=0.6)
    ax.set_title("Transactions per month")
    _style(ax)
    fig.tight_layout()
    return fig


def plot_account_sizes(df: pd.DataFrame):
    import matplotlib.pyplot as plt
    s = df.groupby("account_number").size()
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.hist(s, bins=range(s.min(), s.max() + 2), color=C_MAIN, edgecolor="white")
    ax.set_title(f"Transactions per account (220 accounts; median {int(s.median())})".replace("220", str(len(s))))
    ax.set_xlabel("transactions in account")
    _style(ax)
    fig.tight_layout()
    return fig


def plot_noise_by_class(nbc: pd.DataFrame):
    import matplotlib.pyplot as plt
    d = nbc.sort_values("flag_rate_pct")
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(d.index, d["flag_rate_pct"], color=C_ACCENT, height=0.6)
    for b, (_, r) in zip(bars, d.iterrows()):
        ax.text(r["flag_rate_pct"] + 0.2, b.get_y() + b.get_height() / 2,
                f"{r['flag_rate_pct']:.1f}%  ({int(r['n_flagged'])}/{int(r['n_rows'])})", va="center", fontsize=9)
    ax.set_xlim(0, d["flag_rate_pct"].max() * 1.3)
    ax.set_title("Share of each given label that contradicts its description")
    ax.set_xlabel("% of rows flagged")
    _style(ax, "x")
    fig.tight_layout()
    return fig


def plot_heatmap(table: pd.DataFrame, title: str, fmt: str = "d"):
    """Generic annotated heatmap (confident joint, flip matrix, ...)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(table.values, cmap="Blues")
    ax.set_xticks(range(table.shape[1]), table.columns, rotation=45, ha="right")
    ax.set_yticks(range(table.shape[0]), table.index)
    vmax = table.values.max() or 1
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = table.values[i, j]
            if v:
                ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=8,
                        color="white" if v > vmax * 0.6 else "#222")
    ax.set_title(title)
    ax.set_xlabel(table.columns.name or "")
    ax.set_ylabel(table.index.name or "")
    fig.tight_layout()
    return fig
