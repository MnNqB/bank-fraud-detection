# 🏦 Bank Account Fraud Detection

End-to-end fraud detection pipeline on 1 million+ real-world bank account applications, built on the [Bank Account Fraud (BAF) Dataset](https://huggingface.co/datasets/featurespace/BAF) — a benchmark dataset designed specifically for fraud detection research.

> **Target role context:** This project was built to demonstrate domain expertise relevant to identity and application fraud detection, with particular focus on the kind of end-to-end ownership, feature engineering, and model explainability expected in production fraud systems.

---

## 📊 Dataset

The BAF dataset contains 1 million bank account applications with 32 features covering applicant identity signals, device behavior, velocity metrics, and account history. The Base variant was used — sampled to best represent the original real-world distribution.

- **Size:** 1,000,000 applications
- **Fraud rate:** 1.1% (highly imbalanced)
- **Features:** 32 (mix of behavioral, identity, and device signals)

**Key features include:**
- Identity signals: `name_email_similarity`, `customer_age`, `date_of_birth_distinct_emails_4w`
- Device behavior: `device_os`, `device_distinct_emails_8w`, `keep_alive_session`
- Velocity: `velocity_6h`, `velocity_24h`, `velocity_4w`
- Account signals: `bank_months_count`, `has_other_cards`, `credit_risk_score`

---

## 🔍 Exploratory Data Analysis

### Key Findings

**Categorical fraud rates** revealed several high-signal categories:

| Feature | High-Risk Category | Fraud Rate | vs Baseline (1.1%) |
|---|---|---|---|
| `housing_status` | `BA` | 3.7% | 4x |
| `employment_status` | `CC` | 2.5% | 2.3x |
| `device_os` | `Windows` | 2.5% | 2.3x |
| `source` | `TELEAPP` | 1.6% | 1.5x |

**The fabrication signature:** Legitimate applicants cluster at psychologically natural values — round income numbers, standard credit limits. Fraudulent applications show artificially smooth distributions, consistent with programmatic generation rather than genuine human behavior.

**The synthetic identity profile emerging from EDA:**
> An elderly stated age, foreign origin, thin credit file with no other cards, free email that doesn't match the name, invalid contact numbers, submitted immediately from a Windows device that has cycled through multiple email addresses in recent weeks.

---

## ⚙️ Feature Engineering

Beyond the 32 original features, the following were engineered:

| Feature | Description | Motivation |
|---|---|---|
| `housing_os_fraud_rate` | Target-encoded fraud rate for housing × OS interaction | BA + Windows = 6.8% fraud rate (6x baseline) |
| `no_valid_phone` | Binary: neither home nor mobile phone valid | Both invalid = strong synthetic identity signal |
| `income_high` | Binary: income > 0.8 | Fraudsters cluster at suspiciously high incomes |
| `is_immediate` | Binary: request submitted within 1 day | Fraudsters submit and disappear |
| `multi_email_device` | Binary: 2+ emails from same device in 8 weeks | Device cycling through identities |

**Key discovery:** `housing_os_fraud_rate` — the interaction feature — became the single most important predictor in the final model, outperforming all 51 original features according to SHAP analysis.

---

## 🔀 Train / Test Split

A **temporal split** was used rather than random splitting, as proposed in the BAF paper:

```
Months 0–5 → Train (794,989 applications)
Months 6–7 → Test  (205,011 applications)
```

This simulates real production conditions — the model is always predicting future fraud from historical patterns. Random splitting would allow the model to "see" future fraud patterns during training, leading to optimistic evaluation that wouldn't hold in production.

---

## ⚖️ Handling Class Imbalance

With a 1:89 fraud-to-legit ratio, three approaches were evaluated:

| Approach | AUPRC | Notes |
|---|---|---|
| `class_weight='balanced'` on RF | 0.11 | RF overwhelmed by imbalance |
| `scale_pos_weight=89` on XGBoost | 0.18 | Strong improvement |
| `scale_pos_weight=89` on LightGBM | 0.19 | Best baseline |
| SMOTE (10% sampling strategy) | 0.19 | No improvement — dropped |

**SMOTE was discarded:** With 800k training rows and `scale_pos_weight` already compensating for imbalance, SMOTE introduced noise rather than signal. A large dataset and well-configured loss function rendered synthetic oversampling redundant.

---

## 🤖 Model Selection

Three models were compared at their optimal classification threshold:

```
======================================================================
Metric                    Log Reg     XGBoost    LightGBM
======================================================================
AUPRC                      0.1630      0.1790      0.1865
ROC-AUC                    0.8796      0.8878      0.8886
Fraud Precision            0.2013      0.2126      0.2192
Fraud Recall               0.3085      0.3068      0.3339
Fraud F1                   0.2437      0.2511      0.2647
======================================================================
```

**LightGBM** won across every metric — selected as the final model.

**Why AUPRC over ROC-AUC?** With 1:89 class imbalance, ROC-AUC is dominated by the majority (legit) class and gives an overly optimistic picture. AUPRC focuses specifically on the minority (fraud) class and is a more honest evaluation metric for imbalanced fraud detection.

---

## 🎯 Hyperparameter Tuning

**Optuna** (Bayesian optimization) was used to tune LightGBM — 50 trials optimizing directly for AUPRC:

```python
Best params:
  n_estimators:      933
  learning_rate:     0.030
  max_depth:         3       ← surprisingly shallow
  num_leaves:        138
  min_child_samples: 14
  subsample:         0.742
  colsample_bytree:  0.526
  reg_alpha:         5.18    ← strong L1 regularization
  reg_lambda:        1.39
```

**Notable finding:** `max_depth=3` — shallow trees won. This suggests the fraud signal is relatively linear and doesn't require deeply nested feature interactions, which is consistent with the strong individual feature signals found in EDA.

**AUPRC improvement: 0.1865 → 0.1986 (+6.5%)**

---

## 📈 Final Model Performance

```
Tuned LightGBM at optimal threshold (0.91):

              precision    recall  f1-score   support
       Legit       0.99      0.98      0.99    202,133
       Fraud       0.23      0.33      0.27      2,878

AUPRC:   0.1986
ROC-AUC: 0.8949
```

---

## 🔬 SHAP Explainability

SHAP (SHapley Additive exPlanations) was used to interpret the final model.

**Top predictors (by mean absolute SHAP value):**

| Rank | Feature | Direction |
|---|---|---|
| 1 | `housing_os_fraud_rate` | High value → fraud ✅ engineered feature |
| 2 | `phone_home_valid` | Valid → legit |
| 3 | `prev_address_months_count` | Long history → legit |
| 4 | `has_other_cards` | No cards → fraud |
| 5 | `name_email_similarity` | Low similarity → fraud |
| 6 | `current_address_months_count` | Stable address → legit |
| 7 | `keep_alive_session` | Not kept alive → fraud |
| 8 | `email_is_free` | Free email → fraud |
| 9 | `income` | High income → fraud |
| 10 | `device_os_windows` | Windows → fraud |

**Key insight:** The engineered interaction feature `housing_os_fraud_rate` ranked #1, validating the hypothesis that housing situation × device OS creates a disproportionate fraud risk signal that neither feature captures independently.

**Interesting paradox:** Higher `credit_risk_score` pushes *away* from fraud — counterintuitive until you realize synthetic identities are thin-file by definition. A real person with a long credit history, even a risky one, is less likely to be fabricated.

---

## 🛠️ Stack

- **Modeling:** LightGBM, XGBoost, scikit-learn, Logistic Regression
- **Tuning:** Optuna (Bayesian optimization)
- **Explainability:** SHAP
- **Data:** pandas, NumPy
- **Visualization:** matplotlib, seaborn
- **Environment:** Google Colab

---

## 🔮 Next Steps

- [ ] Build interactive fraud risk scorer demo (FastAPI + HuggingFace Spaces)
- [ ] Stress-test on BAF Variant II (prevalence shift) to evaluate robustness
- [ ] Add calibration analysis — are the fraud probabilities well-calibrated?
- [ ] Explore graph-based features — are fraudulent applicants connected through shared devices/emails?
- [ ] Investigate threshold optimization for specific business cost assumptions (false negative vs false positive cost ratio)
