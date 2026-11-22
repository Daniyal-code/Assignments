# Bank transaction classification

Classify bank transactions (`salary`, `rent`, `emi`, `shopping`, ...) from description, direction, amount and date.
Synthetic data only (`labeled_transactions.csv`, 3,200 rows, 220 accounts, 10 classes).

## Layout
| File | Purpose |
|---|---|
| `txn_pipeline.py` | load, audit, clean, **account-grouped split**, leakage audit, **label-noise detection** |
| `txn_model.py` | model, imbalance handling, grouped CV, metrics, noise-aware evaluation, error analysis, **exact SHAP**, save/load |
| `train.py` | one command that reproduces everything and writes `outputs/` |
| `tests/` | 35 unit tests (`pytest`) |
| `01_eda_cleaning_label_quality.ipynb` | EDA, cleaning, leakage, label-noise analysis (cell by cell) |
| `02_model_evaluation.ipynb` | imbalance, model selection, evaluation, error analysis, SHAP (cell by cell) |
| `REPORT.md` | short report (approach, results, per-class, error analysis, improvements) |
| `outputs/` | trained `model.joblib`, metrics, per-class report, confusion matrix, misclassified rows, figures |

## Reproduce
```bash
pip install -r requirements.txt
python train.py --data labeled_transactions.csv --out outputs   # ~1 min, deterministic (seed 42)
python -m pytest -q tests                                        # 35 tests
```
In Colab/Jupyter: put the `.py` files and CSV next to the notebooks and run them top to bottom.
(The notebooks look for `sample_data/assignment-1-classification/labeled_transactions.csv` first.)

Use the trained model:
```python
import pandas as pd, txn_model as tm
model, blob = tm.load_model("outputs/model.joblib")
new = pd.DataFrame({"description": ["SALARY CREDIT ACME CORP"], "txn_type": ["credit"], "amount": [50000]})
new["desc_norm"] = new["description"].str.upper()           # same normalisation as txn_pipeline.clean_transactions
print(model.predict(new), model.predict_proba(new).max())
```

## Design in one paragraph
TF-IDF (word 1-2-grams + char 2-5-grams of the description) plus the credit/debit flag -> multinomial logistic regression,
`class_weight='balanced'`, C=0.3. Selected by account-grouped, stratified 5-fold CV on macro-F1 (ties broken toward the simplest,
most interpretable model). `amount` and date were tested and excluded (no gain; amount is uniform noise in this synthetic data).
No labels are modified; ~4% suspected-noisy rows are flagged for review and evaluation reports noise-separated metrics.

## Data-leakage controls
Account-grouped stratified split and CV; test set used only for the final report; vectoriser, class weights, oversampling and noise flags are all
fitted inside training folds; no id/account features; automated `leakage_audit`; unit tests for each.

## Notes
* `shap` could not be installed in the sandbox where this was built, so `txn_model.LinearExplainer` implements the **exact** linear-model SHAP values
  (`coef * (x - mean)`; additivity unit-tested). If `shap` is installed, the notebook cross-checks against it.
* **AI assistance:** this solution (code, tests, notebooks, report) was developed with Claude (Anthropic) as a coding/analysis assistant but the architecural decision was mine, so basically i guided it to follow each & evry step. every
  number in the report comes from running the code in this repo (`train.py`, notebooks, tests).
