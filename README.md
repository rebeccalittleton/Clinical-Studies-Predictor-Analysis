# Clinical Study Analysis — Bell's Palsy Recovery Prediction

A machine learning classification project exploring whether 9-month recovery from Bell's Palsy can be predicted from patient demographics, baseline severity, and treatment group data from a clinical trial.

---

## Overview

Bell's Palsy is a condition causing sudden, temporary weakness or paralysis of the facial muscles. This project analyses a clinical trial dataset of 494 patients to build a binary classifier predicting whether a patient will achieve full recovery within 9 months, based on information available at the start of treatment.

The trial tested two drugs — Prednisolone (a corticosteroid) and Acyclovir (an antiviral) — against placebo across four treatment groups. A key finding from both the exploratory analysis and the model is that **Prednisolone significantly improves recovery rates, while Acyclovir shows little to no effect**.

---

## Dataset

- **Source:** Bell's Palsy Clinical Trial
- **Patients:** 494
- **Target:** Full recovery at 9 months (Yes / No)
- **Class balance:** 89.3% recovered, 10.7% did not — imbalanced

| Feature | Type | Description |
|---|---|---|
| Age | Numeric | Patient age |
| Sex | Categorical | Male / Female |
| Baseline Score | Numeric | House–Brackmann scale (1 = mild, 6 = complete paralysis) |
| Time to Treatment | Ordinal | Time between symptom onset and treatment start |
| Treatment Group | Categorical | Prednisolone–Placebo, Acyclovir–Prednisolone, Acyclovir–Placebo, Placebo–Placebo |
| Received Prednisolone | Categorical | Yes / No |
| Received Acyclovir | Categorical | Yes / No |

> **Note:** 3-month scores and recovery status were excluded to prevent data leakage — these are measured after baseline and would not be available at the point of prediction.

---

## Project Structure

```
├── Bells_Palsy_Clinical_Trial.csv        # Raw dataset
├── Clinical_Study_Analysis.ipynb         # Full analysis notebook
└── README.md
```

---

## Methodology

### 1. Exploratory Data Analysis
- Class balance check — identified 89/11 imbalance
- Age distribution by outcome — not-recovered patients averaged ~10 years older
- Recovery rate by treatment group — Prednisolone groups recovered at 93–94% vs 84–85% without
- Baseline severity analysis — higher H-B scores associated with lower recovery rates
- Feature signal assessment — Acyclovir showed no meaningful effect beyond placebo

### 2. Preprocessing
- Dropped post-baseline columns (data leakage prevention)
- Ordinal encoding for time-to-treatment
- One-hot encoding for categorical features (`Sex`, `Treatment_Group`, `Prednisolone`, `Acyclovir`)
- Standard scaling for numeric features (`Age`, `Baseline_Score`)

### 3. Modelling
Two models were trained and compared:

**Logistic Regression** (baseline)
- `class_weight='balanced'` to handle class imbalance
- ROC-AUC: 0.653

**Random Forest** (primary model)
- `class_weight='balanced'`
- Hyperparameter tuning via `RandomizedSearchCV` (50 iterations, stratified 5-fold CV)
- Best parameters: `max_depth=3`, `n_estimators=300`, `min_samples_leaf=3`, `min_samples_split=15`, `max_features=0.5`
- Cross-validated ROC-AUC: 0.694 ± 0.049

### 4. Evaluation
- Stratified 5-fold cross-validation throughout
- Metrics: ROC-AUC, F1 macro, Recall macro, Confusion matrix
- ROC curve comparison between models

---

## Key Results

| Metric | Logistic Regression | Tuned Random Forest |
|---|---|---|
| ROC-AUC (CV) | 0.653 | 0.694 |
| F1 macro | 0.551 | 0.545 |
| Recall macro | 0.594 | 0.641 |
| Not-recovered recall | 0.64 | 0.73 |

**Top features by importance:**
1. Age
2. Baseline H-B Score
3. Sex
4. Time to Treatment
5. Prednisolone
6. Acyclovir (lowest — consistent with EDA findings)

---

## Example Prediction

A fictional patient — 65-year-old male, baseline score 5, treated within 24 hours, receiving Prednisolone only:

```
Predicted outcome:        Not recovered
Probability of recovery:  46.9%
```

This aligns with the EDA: older age and high baseline severity are the two strongest negative predictors, and even Prednisolone cannot fully offset their combined effect.

---

## Limitations

- **Small minority class:** Only 53 non-recoverers in 494 patients. Most classifiers require hundreds of examples per class to learn reliably. This is the primary constraint on model performance.
- **Overall dataset size:** With fewer than 500 patients, the model has limited data to learn fine-grained patterns from.
- **Modest separability:** A cross-validated ROC-AUC of 0.694 indicates the features carry real signal but do not cleanly separate the two classes — reflecting the genuine difficulty of predicting Bell's Palsy recovery from pre-treatment information alone.

Despite these limitations, the model's predictions are consistent with the clinical literature and the EDA findings, and it correctly identifies 73% of non-recoverers in the test set.

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## Usage

1. Clone the repository
2. Place `Bells_Palsy_Clinical_Trial.csv` in the root directory
3. Open `Clinical_Study_Analysis.ipynb` in Jupyter and run all cells top to bottom

---

## Author

Rebecca  
Clinical data analysis project — machine learning portfolio
