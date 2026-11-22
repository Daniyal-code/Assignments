# Report - Bank transaction classification

## 1. Summary
| | |
|---|---|
| Model | TF-IDF (word 1-2gram + char 2-5gram on description) + credit/debit flag -> logistic regression, `class_weight='balanced'`, C=0.3 |
| Test accuracy / macro-F1 | **0.966 / 0.960** (95% CI for macro-F1 0.941-0.975, bootstrap over accounts; n=645) |
| Validation | account-grouped, stratified 5-fold CV on train: macro-F1 0.946 +/- 0.011 |
| Label noise | ~4% of labels (128/3,200) contradict their description; labels **not modified**, flagged for review |
| Main risk | generalisation to bank wording not seen in training (leave-one-template-out) |

## 2. Data and key observations
* 3,200 transactions, 220 accounts (6-26 rows each), 10 classes, Jan-Aug 2026. No nulls/duplicates/invalid values.
* Descriptions come from **28 templates x counterparty**; the template almost fully determines the class.
* **Direction (credit/debit)** is informative and banking-logical (salary/business_receipt/dividend = credit; emi/rent/utility/shopping/insurance = debit; transfer/other mixed).
* **`amount` and date carry no signal** (amount ~ Uniform(100, 90k) in every class, Kruskal p=0.26; label independent of month/weekday, p>0.3) and are domain-implausible (e.g. 57% of bank fees booked as credits). Treated as a synthetic-data artefact; excluded from the model (CV: no gain, slight loss at C=10).

## 3. Approach and why
* **Primary input = description**; char n-grams give robustness to spelling/format variants; word bigrams capture `EMI AUTO DEBIT`, `SAL CR`.
* **Logistic regression** over alternatives tried (linear SVM, gradient boosting, Naive Bayes): equal or better CV score, plus calibrated probabilities (review queue / thresholds) and exact SHAP explanations - important in a banking setting.
* **Strong regularisation (C=0.3)**: with ~4% flipped labels a flexible model (C=10) memorises wrong labels (CV macro-F1 0.934 vs 0.946).
* Selection metric: **macro-F1** (all classes matter equally), simplest model within 0.002 of the best.

## 4. Train/test split and leakage
Rows of one account are correlated, so a random row split would overstate performance. I use `StratifiedGroupKFold` **by `account_number`** for the 80/20 split (train 2,555 rows / 177 accounts; test 645 / 43; class shares differ <0.7 pt) and for all CV. Audit: 0 shared accounts, 0 shared txn_ids, 0 identical transactions across the split. Other controls: vectoriser/class weights/oversampling/noise flags fitted inside training folds only; no id or account features (account-only probe never beats the majority baseline, so accounts carry no label information here); test set not used for any model choice (the small-data imbalance probe uses an inner validation split of train). Description templates overlap ~100% between splits - expected and legitimate.

## 5. Class imbalance
Largest/smallest = 8.0x (`shopping` 21% vs `insurance` 2.7%). Compared `none`, `class_weight='balanced'`, random oversampling (inside each fold); SMOTE rejected for sparse text.
* Full data, C=0.3: all tie (macro-F1 0.946). With flexible C=10, weighting/oversampling are worse (0.934 / 0.921 vs 0.942) because they up-weight noisy minority rows.
* **Small-data stress test** (train on 3-10% of accounts): `none` collapses (macro-F1 0.49 / 0.75; minority recall 0.20 / 0.22); `class_weight` holds up (0.80 / 0.955; recall 0.90 / 0.91).
* **Choice: `class_weight='balanced'`** - no cost on full data, protects minority classes when data is scarce, no duplicated rows.

## 6. Label noise
* **Evidence (three signals):** (A) same description template, different label (128 rows = 4.0%); (B) direction conflict, e.g. salary booked as debit (45 rows, all inside A); (C) out-of-fold model disagreement via Confident Learning (109 rows, estimated noise 4.2%). 115 rows (3.6%) have >=2 signals.
* **Looks random, not systematic:** flagged rows are independent of amount, direction, month and account (p>0.2) and land evenly across classes (chi-square vs uniform p=0.41). Small classes are hit hardest in relative terms: `insurance` 17.6%, `dividend` 12.8%, vs `shopping` 1.3%.
* **Handling: no label was changed.** Auto-relabelling to the template majority is a guess and would hide real exceptions. Instead: strong regularisation, suspects optionally dropped from training (helps at C=10: 0.934 -> 0.945; no difference at C=0.3), a ranked **review queue** for humans, and **noise-aware evaluation** on the test set.

## 7. Evaluation (test set, given labels)
| class | precision | recall | F1 | support |
|---|---|---|---|---|
| business_receipt | 0.983 | 0.905 | 0.942 | 63 |
| dividend | 0.955 | 0.913 | 0.933 | 23 |
| emi | 0.919 | 0.983 | 0.950 | 58 |
| insurance | 1.000 | 0.833 | 0.909 | 18 |
| other | 0.971 | 1.000 | 0.986 | 34 |
| rent | 1.000 | 0.981 | 0.991 | 54 |
| salary | 0.979 | 0.969 | 0.974 | 97 |
| shopping | 0.957 | 0.985 | 0.971 | 135 |
| transfer | 0.949 | 0.979 | 0.964 | 95 |
| utility | 0.985 | 0.971 | 0.978 | 68 |
| **macro / weighted** | 0.970 / 0.967 | 0.952 / 0.966 | **0.960 / 0.966** | 645 |

Confusion matrix: `outputs/confusion_matrix.csv`, `outputs/figures/confusion_matrix.png`. Baselines: always-`shopping` 0.209 accuracy / 0.035 macro-F1; template-majority lookup rule 0.966 / 0.960 (**identical to the model's predictions**).

## 8. Error analysis
**Weakest classes:** `insurance` (recall 0.83), `dividend` (0.91), `business_receipt` (0.91); lowest F1 for `insurance` 0.91, `dividend` 0.93, `business_receipt` 0.94; `emi` has the lowest precision (0.92).
**Why:**
1. *Label noise.* All 22 test errors are rows whose label contradicts the description's dominant label (e.g. `AUTO DEBIT EMI HOME LOAN` labelled `salary`; `UPI-HOOLI-ORDER` labelled `emi`); excluding rows flagged as likely-noisy (using only train statistics) accuracy is 100%. Small classes lose more recall per flipped row.
2. *Small support:* one error = 4-6 recall points in `insurance`/`dividend`; hence the wide CI.
3. *Caveat:* "likely noisy" is defined from the same text pattern the model learns, so the 100% is partly circular. Independent support: the direction-conflict signal, the SHAP analysis, and my reading of the 22 rows - most are clearly impossible, a few (`NEFT TRANSFER` labelled `shopping`, `IMPS` credit labelled `business_receipt`) are arguable and are what a reviewer should check.
4. Error rates are flat across direction (3-4%) and amount band (1-5%); no template has more than 4 of the 22 errors.

**Real weakness - unseen wording.** Leave-one-template-out accuracy (train without a template, predict it): 17/28 templates 100%, but `BANK FEE`, `SERVICE CHARGE`, `MISC DEBIT ENTRY`, `RENT TRANSFER TO <name>`, `UPI-<co>-ORDER` **0%**, `INVOICE SETTLEMENT` 7%, `IMPS/NEFT TRANSFER <name>` 45% / 59% (confused with `rent`), `SAL CR ... PAYROLL` 57%. TF-IDF has no semantics ("FEE" ~ "CHARGE"), so new bank wording will be misclassified; the 96.6% test accuracy does not reflect this because train and test share templates. Low-confidence examples (e.g. unseen `ATM CASH WITHDRAWAL` -> confidence 0.20) can be routed to review.

## 9. Explainability (SHAP)
Exact linear-SHAP (`coef x (x - mean)`, logit units, additivity unit-tested; `shap` package unavailable in the build sandbox, cross-checked automatically if installed). Five misclassified rows explained (`outputs/figures/shap_*.png`, notebook section 6). In every case the **predicted class is supported by words in the description** (`NEFT TRANSFER`, `SERVICE CHARGE`, `EMI LOAN`, `IMPS TRANSFER`, `CR PAYROLL`) while the **given label has no word support** - only the credit/debit flag, which in two cases actively contradicts it (credit `IMPS TRANSFER` labelled `utility`; debit `EMI` labelled `dividend`). These are label errors, not subtle model mistakes. The least clear row (`SAL CR UMBRELLA PAYROLL` labelled `business_receipt`, confidence 0.76) is a credit-vs-credit ambiguity a human should confirm.

## 10. Limitations and what I would do next
1. **Unseen wording:** sentence embeddings or a small fine-tuned transformer; collect more diverse real descriptions; keep a "template not seen / low confidence -> review" rule.
2. **Real data signals:** amount, timing and recurrence (salary monthly, EMI fixed amount and day) - add account-level recurrence features computed per training fold; out-of-time validation split.
3. **Test set size:** 645 rows / 43 accounts gives wide CIs; use repeated grouped CV or a larger holdout for final claims.
4. **Active learning / targeted review (optional):** rank rows by `noise_score` and low model confidence (`tp.review_queue`), have annotators relabel the top-N each cycle, retrain, re-estimate the noise rate; stop when flagged precision drops. Prioritise `insurance`, `dividend`, `business_receipt`.
5. Calibrate probabilities and set per-class confidence thresholds; monitor drift in description templates in production.
