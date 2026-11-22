"""
txn_model.py
============
Model building, validation, evaluation, error analysis and explainability for the
bank-transaction classifier. Builds on `txn_pipeline.py` (cleaning, splitting, label-noise flags).

Main entry points
-----------------
    ModelConfig / build_pipeline / fit_model     -> define and train a model
    cross_validate_grouped                       -> account-grouped, stratified CV (no leakage)
    per_class_report / confusion_df              -> precision / recall / F1 per class, confusion matrix
    reference_noise_flags / noise_aware_metrics  -> separate "model error" from "label error" on the test set
    segment_errors / misclassified_table         -> where does the model fail?
    LinearExplainer                              -> exact SHAP values for the linear model
    save_model / load_model                      -> persistence
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

RANDOM_SEED = 42
C_MAIN, C_GREY, C_ACCENT = "#2a6f97", "#b8bfc6", "#d1495b"



# Configuration

@dataclass
class ModelConfig:
    """
    algorithm   : 'logreg' | 'linear_svc' | 'complement_nb' | 'hgb' (gradient boosting, dense word features)
    imbalance   : 'none' | 'class_weight' | 'oversample'
    use_type    : add the credit/debit flag
    use_amount  : add log(amount)
    use_date    : add month, weekday, day-of-month
    """
    algorithm: str = "logreg"
    imbalance: str = "class_weight"
    use_type: bool = True
    use_amount: bool = False
    use_date: bool = False
    C: float = 10.0
    word_ngram: Tuple[int, int] = (1, 2)
    char_ngram: Tuple[int, int] = (2, 5)
    seed: int = RANDOM_SEED

    def name(self) -> str:
        feats = "text" + ("+type" if self.use_type else "") + ("+amount" if self.use_amount else "") + ("+date" if self.use_date else "")
        return f"{self.algorithm} | {self.imbalance} | {feats} | C={self.C}"


def _log1p(x):
    return np.log1p(np.asarray(x, dtype=float))


def _to_dense(x):
    return x.toarray() if sparse.issparse(x) else x


def build_pipeline(cfg: ModelConfig) -> Pipeline:
    """Feature transformer + classifier. The pipeline takes the CLEANED DataFrame directly."""
    tok = r"[A-Za-z0-9<>]+"
    if cfg.algorithm == "hgb":     # trees: word-level TF-IDF only (dense, small vocabulary)
        text = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), lowercase=True, token_pattern=tok)
    else:
        text = FeatureUnion([
            ("word", TfidfVectorizer(analyzer="word", ngram_range=cfg.word_ngram, lowercase=True, token_pattern=tok)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=cfg.char_ngram, lowercase=True, sublinear_tf=True)),
        ])
    parts = [("text", text, "desc_norm")]
    if cfg.use_type:
        parts.append(("type", OneHotEncoder(handle_unknown="ignore"), ["txn_type"]))
    if cfg.use_amount:
        from sklearn.pipeline import make_pipeline
        parts.append(("amount", make_pipeline(FunctionTransformer(_log1p, feature_names_out="one-to-one"), StandardScaler()), ["amount"]))
    if cfg.use_date:
        parts.append(("date", StandardScaler(), ["month", "dow", "day_of_month"]))
    ct = ColumnTransformer(parts, sparse_threshold=0.3 if cfg.algorithm != "hgb" else 0.0)

    cw = "balanced" if cfg.imbalance == "class_weight" else None
    if cfg.algorithm == "logreg":
        clf = LogisticRegression(C=cfg.C, max_iter=5000, class_weight=cw)
    elif cfg.algorithm == "linear_svc":
        clf = LinearSVC(C=cfg.C / 10, class_weight=cw, max_iter=10000, random_state=cfg.seed)
    elif cfg.algorithm == "complement_nb":
        clf = ComplementNB(alpha=0.1)
    elif cfg.algorithm == "hgb":
        clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_depth=4, class_weight=cw, random_state=cfg.seed)
    else:
        raise ValueError(f"unknown algorithm {cfg.algorithm}")
    steps = [("features", ct)]
    if cfg.algorithm == "hgb":
        steps.append(("dense", FunctionTransformer(_to_dense, accept_sparse=True)))
    steps.append(("clf", clf))
    return Pipeline(steps)


def random_oversample(df: pd.DataFrame, label_col: str = "label", seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Duplicate minority-class rows (with replacement) until every class matches the largest one."""
    rng = np.random.RandomState(seed)
    target = df[label_col].value_counts().max()
    parts = [df]
    for _, g in df.groupby(label_col):
        extra = target - len(g)
        if extra > 0:
            parts.append(g.iloc[rng.randint(0, len(g), extra)])
    return pd.concat(parts, ignore_index=True)


def fit_model(cfg: ModelConfig, train: pd.DataFrame) -> Pipeline:
    """Fit on `train` only. Oversampling (if chosen) happens HERE, i.e. inside the training fold, never before the split."""
    data = random_oversample(train, seed=cfg.seed) if cfg.imbalance == "oversample" else train
    return build_pipeline(cfg).fit(data, data["label"])



# Validation

def drop_suspect_rows(train: pd.DataFrame, min_signals: int = 2, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Remove training rows that >= `min_signals` label-noise checks flag (see txn_pipeline.label_issue_report).
    Flags are computed from the TRAINING rows only, so no validation/test information is used.
    """
    # import txn_pipeline as tp
    from ai_ml.txn_classifier import txn_pipeline as tp
    report, _, _ = tp.label_issue_report(train, seed=seed)
    return train.loc[report.index[report["n_signals"] < min_signals]]


def cross_validate_grouped(df: pd.DataFrame, cfg: ModelConfig, n_splits: int = 5, seed: int = RANDOM_SEED,
                           group_col: str = "account_number", drop_suspects: bool = False):
    """
    Account-grouped, stratified K-fold. Returns (summary Series, out-of-fold predictions Series).
    drop_suspects=True removes suspected-noisy rows from each TRAINING fold (flags computed on that fold's
    training rows only); validation rows always keep their raw labels.
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = pd.Series(index=df.index, dtype=object)
    fold_f1, fold_acc = [], []
    for tr, va in sgkf.split(df, df["label"], groups=df[group_col]):
        train, valid = df.iloc[tr], df.iloc[va]
        if drop_suspects:
            train = drop_suspect_rows(train, seed=seed)
        model = fit_model(cfg, train)
        pred = model.predict(valid)
        oof.iloc[va] = pred
        fold_f1.append(f1_score(valid["label"], pred, average="macro"))
        fold_acc.append(accuracy_score(valid["label"], pred))
    summary = pd.Series({"macro_f1_mean": np.mean(fold_f1), "macro_f1_std": np.std(fold_f1),
                         "accuracy_mean": np.mean(fold_acc), "accuracy_std": np.std(fold_acc)})
    return summary, oof


def compare_configs(df: pd.DataFrame, configs: Sequence[ModelConfig], n_splits: int = 5, seed: int = RANDOM_SEED,
                    drop_suspects: bool = False) -> pd.DataFrame:
    """Run `cross_validate_grouped` for each config; returns a leaderboard sorted by macro-F1."""
    rows = []
    for cfg in configs:
        s, _ = cross_validate_grouped(df, cfg, n_splits, seed, drop_suspects=drop_suspects)
        rows.append({"config": cfg.name() + (" | drop-suspects" if drop_suspects else ""), **s.round(4).to_dict()})
    return pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False).reset_index(drop=True)



# Evaluation

def per_class_report(y_true, y_pred, labels: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Precision / recall / F1 / support per class plus macro and weighted averages and accuracy."""
    labels = list(labels) if labels is not None else sorted(set(y_true) | set(y_pred))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    rep = pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s}, index=labels)
    total = rep["support"].sum()
    macro = rep[["precision", "recall", "f1"]].mean()
    w = np.average(rep[["precision", "recall", "f1"]], axis=0, weights=rep["support"])
    rep.loc["macro avg"] = [*macro.values, total]
    rep.loc["weighted avg"] = [*w, total]
    rep.loc["accuracy"] = [accuracy_score(y_true, y_pred)] * 3 + [total]
    return rep.round(3)


def confusion_df(y_true, y_pred, labels: Optional[Sequence[str]] = None) -> pd.DataFrame:
    labels = list(labels) if labels is not None else sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=pd.Index(labels, name="true"), columns=pd.Index(labels, name="predicted"))


def bootstrap_ci(test: pd.DataFrame, pred: Sequence[str], metric: str = "macro_f1", n_boot: int = 1000,
                 group_col: str = "account_number", seed: int = RANDOM_SEED) -> Tuple[float, float, float]:
    """Cluster bootstrap (resamples whole ACCOUNTS) 95% CI for macro-F1 / accuracy. Returns (point, lo, hi)."""
    rng = np.random.RandomState(seed)
    d = pd.DataFrame({"g": test[group_col].to_numpy(), "y": test["label"].to_numpy(), "p": np.asarray(pred)})
    groups = {g: x for g, x in d.groupby("g")}
    keys = list(groups)

    def stat(x):
        return f1_score(x["y"], x["p"], average="macro") if metric == "macro_f1" else accuracy_score(x["y"], x["p"])
    vals = [stat(pd.concat([groups[k] for k in rng.choice(keys, len(keys))])) for _ in range(n_boot)]
    return stat(d), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ---- noise-aware evaluation ---------------------------------------------------------
def reference_noise_flags(train: pd.DataFrame, test: pd.DataFrame, min_support: int = 5, min_share: float = 0.8,
                          direction_share: float = 0.05) -> pd.Series:
    """
    Flag test rows whose label is almost certainly wrong, using ONLY training-set statistics (no test labels
    are used to decide the reference): template majority label and label/direction frequencies from `train`.
    A test row is 'likely noisy' if (template has >= min_support train rows with majority share >= min_share
    AND the row's label differs) OR its (label, direction) pair is rare (< direction_share) in train.
    """
    g = train.groupby("template")["label"]
    size = g.size()
    maj = g.agg(lambda s: s.value_counts().index[0])
    share = g.agg(lambda s: s.value_counts(normalize=True).iloc[0])
    ok = size[(size >= min_support) & (share >= min_share)].index
    ref = maj[ok]
    t_ref = test["template"].map(ref)
    tmpl_bad = t_ref.notna() & (t_ref != test["label"])
    dir_share = train.groupby("label")["txn_type"].value_counts(normalize=True)
    pair = pd.Series(list(zip(test["label"], test["txn_type"])), index=test.index)
    dir_bad = pair.map(lambda k: dir_share.get(k, 0.0) < direction_share)
    return (tmpl_bad | dir_bad).rename("likely_noisy")


def noise_aware_metrics(test: pd.DataFrame, pred: Sequence[str], likely_noisy: pd.Series) -> pd.DataFrame:
    """
    Metrics on (a) all test rows with the given labels and (b) the rows NOT flagged as noisy.
    Also reports how many of the model's errors fall on flagged rows.
    """
    pred = pd.Series(np.asarray(pred), index=test.index)
    wrong = pred != test["label"]
    clean = ~likely_noisy
    rows = [
        {"view": "all test rows (given labels)", "n": len(test),
         "accuracy": accuracy_score(test["label"], pred), "macro_f1": f1_score(test["label"], pred, average="macro")},
        {"view": "excluding likely-noisy labels", "n": int(clean.sum()),
         "accuracy": accuracy_score(test.loc[clean, "label"], pred[clean]),
         "macro_f1": f1_score(test.loc[clean, "label"], pred[clean], average="macro")},
    ]
    out = pd.DataFrame(rows).round(4)
    out.attrs["errors_total"] = int(wrong.sum())
    out.attrs["errors_on_likely_noisy"] = int((wrong & likely_noisy).sum())
    out.attrs["errors_on_clean"] = int((wrong & clean).sum())
    return out


# ---- error analysis -----------------------------------------------------------------
def segment_errors(test: pd.DataFrame, pred: Sequence[str], by: str, bins: Optional[Sequence[float]] = None) -> pd.DataFrame:
    """Accuracy and error count per segment of a column (or of binned numeric column)."""
    d = test.copy()
    d["pred"] = np.asarray(pred)
    d["wrong"] = d["pred"] != d["label"]
    key = pd.cut(d[by], bins) if bins is not None else d[by]
    g = d.groupby(key, observed=True)["wrong"].agg(["size", "sum"])
    g.columns = ["n", "errors"]
    g["error_rate_pct"] = (g["errors"] / g["n"] * 100).round(1)
    return g.sort_values("error_rate_pct", ascending=False)


def misclassified_table(test: pd.DataFrame, pred: Sequence[str], proba: Optional[np.ndarray] = None,
                        classes: Optional[Sequence[str]] = None, likely_noisy: Optional[pd.Series] = None) -> pd.DataFrame:
    """All misclassified rows with true/predicted label, confidence, and (optionally) the likely-noisy flag."""
    d = test[["txn_id", "account_number", "description", "txn_type", "amount", "label"]].copy()
    d["pred"] = np.asarray(pred)
    if proba is not None:
        d["pred_conf"] = proba.max(axis=1).round(3)
        idx = {c: i for i, c in enumerate(classes)}
        d["p_true_label"] = [proba[i, idx[l]].round(4) for i, l in enumerate(d["label"])]
    if likely_noisy is not None:
        d["likely_noisy_label"] = likely_noisy
    return d[d["pred"] != d["label"]].rename(columns={"label": "true_label"})



# Explainability: exact SHAP values for a linear (softmax) model

class LinearExplainer:
    """
    Exact SHAP values for the multinomial logistic-regression pipeline (same result as
    `shap.LinearExplainer` with an independent-feature background):

        phi[class c, feature j] = coef[c, j] * (x[j] - mean_background[j])
        logit[c]                = base[c] + sum_j phi[c, j]      with   base[c] = intercept[c] + coef[c] . mean_background

    Values are in log-odds units of each class's raw score (logit). Features come from the fitted
    ColumnTransformer (word / char n-grams, direction, optional amount/date).
    """

    def __init__(self, pipeline: Pipeline, background: pd.DataFrame):
        if not isinstance(pipeline.named_steps["clf"], LogisticRegression):
            raise TypeError("LinearExplainer supports the LogisticRegression pipeline only")
        self.pipe = pipeline
        self.clf: LogisticRegression = pipeline.named_steps["clf"]
        self.classes = list(self.clf.classes_)
        Xb = pipeline.named_steps["features"].transform(background)
        self.mean = np.asarray(Xb.mean(axis=0)).ravel()
        self.names = pipeline.named_steps["features"].get_feature_names_out()
        self.base = self.clf.intercept_ + self.clf.coef_ @ self.mean

    def _x(self, rows: pd.DataFrame):
        return self.pipe.named_steps["features"].transform(rows)

    def shap_values(self, rows: pd.DataFrame) -> np.ndarray:
        """Array [n_rows, n_classes, n_features]."""
        X = self._x(rows)
        Xd = X.toarray() if sparse.issparse(X) else np.asarray(X)
        return (Xd - self.mean)[:, None, :] * self.clf.coef_[None, :, :]

    def logits(self, rows: pd.DataFrame) -> np.ndarray:
        return self.pipe.decision_function(rows)

    def explain_row(self, row: pd.DataFrame, class_name: str, top_k: int = 6) -> pd.DataFrame:
        """Top features (by |SHAP|) pushing the logit of `class_name` for one row."""
        c = self.classes.index(class_name)
        X = self._x(row)
        x = (X.toarray() if sparse.issparse(X) else np.asarray(X)).ravel()
        phi = (x - self.mean) * self.clf.coef_[c]
        df = pd.DataFrame({"feature": self.names, "value": x, "shap": phi})
        df = df[df["shap"].abs() > 1e-9]
        df["feature"] = df["feature"].str.replace(r"^text__(word|char)__", r"\1:", regex=True).str.replace(r"^type__txn_type_", "direction=", regex=True)
        # The one-hot credit/debit columns carry the same information; show them as ONE bar (sum of both).
        is_dir = df["feature"].str.startswith("direction=")
        if is_dir.any():
            active = df.loc[is_dir & (df["value"] > 0), "feature"]
            merged = pd.DataFrame([{"feature": active.iloc[0] if len(active) else "direction",
                                    "value": 1.0, "shap": df.loc[is_dir, "shap"].sum()}])
            df = pd.concat([df[~is_dir], merged], ignore_index=True)
        return df.reindex(df["shap"].abs().sort_values(ascending=False).index).head(top_k).reset_index(drop=True)

    def explain_error(self, row: pd.DataFrame, true_label: str, pred_label: str, top_k: int = 5) -> Dict[str, object]:
        """Why predicted `pred_label` instead of `true_label`: top drivers for each and the logit gap."""
        lg = self.logits(row)[0]
        return {"logit_pred": float(lg[self.classes.index(pred_label)]), "logit_true": float(lg[self.classes.index(true_label)]),
                "for_pred": self.explain_row(row, pred_label, top_k), "for_true": self.explain_row(row, true_label, top_k)}

    def global_importance(self, rows: pd.DataFrame, top_k: int = 8) -> pd.DataFrame:
        """Mean |SHAP| per feature for each class (top_k features per class), a global view of what drives each class."""
        sv = np.abs(self.shap_values(rows)).mean(axis=0)  # [classes, features]
        out = []
        for ci, cname in enumerate(self.classes):
            top = np.argsort(-sv[ci])[:top_k]
            out += [{"class": cname, "feature": self.names[j], "mean_abs_shap": round(float(sv[ci, j]), 4)} for j in top]
        d = pd.DataFrame(out)
        d["feature"] = d["feature"].str.replace(r"^text__(word|char)__", r"\1:", regex=True).str.replace(r"^type__txn_type_", "direction=", regex=True)
        return d


def try_shap_package(pipeline: Pipeline, background: pd.DataFrame, rows: pd.DataFrame):
    """If the `shap` package is installed, return shap.LinearExplainer values (cross-check). Else None."""
    try:
        import shap
    except Exception:
        return None
    feats = pipeline.named_steps["features"]
    Xb, X = feats.transform(background), feats.transform(rows)
    expl = shap.LinearExplainer(pipeline.named_steps["clf"], Xb)
    return expl.shap_values(X)



# Persistence + plots

def save_model(pipeline: Pipeline, path: str, cfg: Optional[ModelConfig] = None, extra: Optional[dict] = None) -> None:
    joblib.dump({"pipeline": pipeline, "config": asdict(cfg) if cfg else None, "extra": extra or {}}, path)


def load_model(path: str) -> Tuple[Pipeline, dict]:
    blob = joblib.load(path)
    return blob["pipeline"], blob


def _style(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color="#e6e9ec", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_confusion_matrix(cm: pd.DataFrame, normalize: bool = True, title: str = "Confusion matrix"):
    """Heatmap with counts in cells; colour = row-normalised recall when normalize=True."""
    import matplotlib.pyplot as plt
    vals = cm.values.astype(float)
    shade = vals / vals.sum(axis=1, keepdims=True).clip(min=1) if normalize else vals
    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.imshow(shade, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(cm.shape[1]), cm.columns, rotation=45, ha="right")
    ax.set_yticks(range(cm.shape[0]), cm.index)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = int(vals[i, j])
            if v:
                ax.text(j, i, v, ha="center", va="center", fontsize=8,
                        color="white" if shade[i, j] > 0.55 else ("#d1495b" if i != j else "#222"))
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_per_class_f1(rep: pd.DataFrame, title: str = "F1 per class (test set)"):
    import matplotlib.pyplot as plt
    d = rep.drop(index=[i for i in ("macro avg", "weighted avg", "accuracy") if i in rep.index]).sort_values("f1")
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(d.index, d["f1"], color=C_MAIN, height=0.6)
    for b, (_, r) in zip(bars, d.iterrows()):
        ax.text(r["f1"] + 0.004, b.get_y() + b.get_height() / 2, f"F1 {r['f1']:.2f}  (P {r['precision']:.2f} / R {r['recall']:.2f}, n={int(r['support'])})", va="center", fontsize=8.5)
    ax.set_xlim(max(0, d["f1"].min() - 0.15), 1.25)
    ax.set_title(title); _style(ax, "x")
    fig.tight_layout()
    return fig


def plot_explanation(expl: Dict[str, object], title: str, true_label: str, pred_label: str):
    """Side-by-side bar chart of top SHAP drivers for the predicted class and the (given) true class."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    for ax, key, name in [(axes[0], "for_pred", f"predicted: {pred_label}"), (axes[1], "for_true", f"given label: {true_label}")]:
        d = expl[key].iloc[::-1]
        ax.barh(d["feature"], d["shap"], color=[C_MAIN if v > 0 else C_ACCENT for v in d["shap"]], height=0.6)
        ax.axvline(0, color="#888", linewidth=0.8)
        ax.set_title(name, fontsize=10); ax.set_xlabel("SHAP (logit units)"); _style(ax, "x")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig



# Robustness probes (go beyond the random hold-out)

def leave_one_template_out(df: pd.DataFrame, cfg: ModelConfig, eval_mask: Optional[pd.Series] = None,
                           min_rows: int = 5) -> pd.DataFrame:
    """
    Generalisation to NEW WORDING: for each description template, train on every other template and predict that
    template's rows. Shows which kinds of transaction text the model could not classify without having seen it.
    `eval_mask` (bool Series on df.index) restricts evaluation rows, e.g. to rows with trusted labels.
    """
    rows = []
    mask = eval_mask if eval_mask is not None else pd.Series(True, index=df.index)
    for t, g in df.groupby("template"):
        ev = g[mask.loc[g.index]]
        if len(ev) < min_rows:
            continue
        model = fit_model(cfg, df[df["template"] != t])
        pred = model.predict(ev)
        wrong = pd.Series(pred, index=ev.index) != ev["label"]
        confused = pd.Series(pred, index=ev.index)[wrong].value_counts().head(2).to_dict()
        rows.append({"template": t, "true_label": ev["label"].mode()[0], "n": len(ev),
                     "accuracy": round(1 - wrong.mean(), 3), "most_common_wrong_pred": confused})
    return pd.DataFrame(rows).sort_values("accuracy").reset_index(drop=True)


def small_data_imbalance_probe(train: pd.DataFrame, test: pd.DataFrame, cfgs: Sequence[ModelConfig],
                               fractions: Sequence[float] = (0.03, 0.1, 0.3), n_repeats: int = 5,
                               seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Train on a small random fraction of the TRAINING ACCOUNTS (where imbalance bites hardest) and score on the
    fixed test set. Returns mean macro-F1 and mean minority-class (insurance/dividend) recall per strategy.
    """
    rng = np.random.RandomState(seed)
    accts = train["account_number"].unique()
    minority = train["label"].value_counts().nsmallest(2).index.tolist()
    rows = []
    for frac in fractions:
        for cfg in cfgs:
            f1s, recs = [], []
            for r in range(n_repeats):
                sel = rng.choice(accts, max(3, int(len(accts) * frac)), replace=False)
                sub = train[train["account_number"].isin(sel)]
                if sub["label"].nunique() < train["label"].nunique():
                    continue
                pred = fit_model(cfg, sub).predict(test)
                f1s.append(f1_score(test["label"], pred, average="macro"))
                rep = precision_recall_fscore_support(test["label"], pred, labels=minority, zero_division=0)
                recs.append(rep[1].mean())
            rows.append({"train_fraction": frac, "strategy": cfg.imbalance, "macro_f1": np.mean(f1s),
                         f"recall_{'+'.join(minority)}": np.mean(recs), "runs": len(f1s)})
    return pd.DataFrame(rows).round(4)


def select_config(df: pd.DataFrame, candidates: Sequence[ModelConfig], tol: float = 0.002, n_splits: int = 5,
                  seed: int = RANDOM_SEED) -> Tuple[ModelConfig, pd.DataFrame]:
    """
    Pick a config by grouped-CV macro-F1 with a simplicity tie-break: `candidates` must be ordered from most
    preferred (simplest / most interpretable) to least; the first one within `tol` of the best score wins.
    Returns (chosen config, leaderboard in candidate order).
    """
    rows = []
    for cfg in candidates:
        s, _ = cross_validate_grouped(df, cfg, n_splits, seed)
        rows.append({"config": cfg.name(), "macro_f1_mean": round(s["macro_f1_mean"], 4),
                     "macro_f1_std": round(s["macro_f1_std"], 4), "accuracy_mean": round(s["accuracy_mean"], 4)})
    board = pd.DataFrame(rows)
    best = board["macro_f1_mean"].max()
    chosen = candidates[int(np.argmax((board["macro_f1_mean"] >= best - tol).to_numpy()))]
    board["chosen"] = [c is chosen for c in candidates]
    return chosen, board


DEFAULT_CANDIDATES = [
    ModelConfig(algorithm="logreg", imbalance="class_weight", C=0.3),
    ModelConfig(algorithm="logreg", imbalance="class_weight", C=1.0),
    ModelConfig(algorithm="logreg", imbalance="class_weight", C=10.0),
    ModelConfig(algorithm="logreg", imbalance="none", C=0.3),
    ModelConfig(algorithm="logreg", imbalance="oversample", C=10.0),
    ModelConfig(algorithm="linear_svc", imbalance="class_weight", C=1.0),
    ModelConfig(algorithm="complement_nb", imbalance="none"),
    ModelConfig(algorithm="hgb", imbalance="class_weight", use_amount=True, use_date=True),
]
