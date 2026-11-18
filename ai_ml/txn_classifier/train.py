"""
train.py - reproduce everything: load -> clean -> account-grouped split -> CV model selection -> fit -> evaluate
-> save model + reports.

    python train.py --data labeled_transactions.csv --out outputs

Outputs (in --out):  model.joblib, metrics.json, per_class_report.csv, confusion_matrix.csv, misclassified.csv,
                     cv_leaderboard.csv, figures/*.png
"""
import argparse, json, os, warnings
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import txn_pipeline as tp
import txn_model as tm


def main(data: str, out: str, seed: int = tm.RANDOM_SEED, test_size: float = 0.2) -> dict:
    warnings.filterwarnings("ignore")
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)

    df, _ = tp.clean_transactions(tp.load_data(data))
    train, test = tp.group_train_test_split(df, test_size=test_size, seed=seed)       # grouped by account
    tp.assert_no_group_overlap(train, test)

    cfg, board = tm.select_config(train, tm.DEFAULT_CANDIDATES, seed=seed)              # CV on TRAIN only
    board.to_csv(os.path.join(out, "cv_leaderboard.csv"), index=False)
    print(board.to_string(index=False)); print("chosen:", cfg.name())

    model = tm.fit_model(cfg, train)
    pred, proba = model.predict(test), model.predict_proba(test)
    rep = tm.per_class_report(test["label"], pred)
    cm = tm.confusion_df(test["label"], pred)
    noisy = tm.reference_noise_flags(train, test)
    na = tm.noise_aware_metrics(test, pred, noisy)
    mis = tm.misclassified_table(test, pred, proba, model.classes_, noisy)
    f1_pt, f1_lo, f1_hi = tm.bootstrap_ci(test, pred, "macro_f1", seed=seed)

    rep.to_csv(os.path.join(out, "per_class_report.csv")); cm.to_csv(os.path.join(out, "confusion_matrix.csv"))
    mis.to_csv(os.path.join(out, "misclassified.csv"), index=False)
    tm.plot_confusion_matrix(cm).savefig(os.path.join(out, "figures", "confusion_matrix.png"), dpi=130)
    tm.plot_per_class_f1(rep).savefig(os.path.join(out, "figures", "per_class_f1.png"), dpi=130)
    expl = tm.LinearExplainer(model, train)                                            # exact SHAP for the linear model
    for _, r in mis.drop_duplicates("true_label").head(5).iterrows():
        e = expl.explain_error(test.loc[[_]], r["true_label"], r["pred"])
        fig = tm.plot_explanation(e, f"{r['description']} -> predicted {r['pred']}, labelled {r['true_label']}", r["true_label"], r["pred"])
        fig.savefig(os.path.join(out, "figures", f"shap_{r['txn_id']}.png"), dpi=130)
    tm.save_model(model, os.path.join(out, "model.joblib"), cfg, {"classes": list(model.classes_)})

    metrics = {"config": cfg.name(), "n_train": len(train), "n_test": len(test),
               "accuracy": float(accuracy_score(test["label"], pred)),
               "macro_f1": float(f1_score(test["label"], pred, average="macro")),
               "macro_f1_ci95": [f1_lo, f1_hi], "errors": int(len(mis)),
               "errors_on_likely_noisy_labels": int(na.attrs["errors_on_likely_noisy"]),
               "accuracy_excluding_likely_noisy": float(na.loc[1, "accuracy"])}
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2)); print(rep)
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="labeled_transactions.csv")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--seed", type=int, default=tm.RANDOM_SEED)
    a = ap.parse_args()
    main(a.data, a.out, a.seed)
