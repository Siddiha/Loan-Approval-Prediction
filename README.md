# Loan Approval Prediction — Machine Learning Project

A complete machine learning pipeline for predicting loan approval status and estimating the maximum allowable loan amount. The project covers end-to-end data preprocessing, classification, ensemble learning, and regression using a real-world financial dataset of 58,645 loan applications.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Notebooks Summary](#notebooks-summary)
  - [Notebook 1 — Data Preprocessing](#notebook-1--data-preprocessing)
  - [Notebook 2 — Classification Models](#notebook-2--classification-models)
  - [Notebook 3 — Ensemble & Regression](#notebook-3--ensemble--regression)
- [Models & Results](#models--results)
  - [Classification Results](#classification-results)
  - [Regression Results](#regression-results)
- [Technologies Used](#technologies-used)
- [How to Run](#how-to-run)

---

## Project Overview

This project tackles two distinct machine learning problems using the same loan dataset:

| Task | Problem Type | Target Variable | Dataset Size |
|------|-------------|-----------------|-------------|
| Loan Approval Prediction | Binary Classification | `loan_approval_status` (0 = Rejected, 1 = Approved) | 58,645 rows |
| Maximum Loan Amount Prediction | Regression | `max_allowed_loan` (£) | 50,292 rows (approved loans only) |

---

## Dataset

**File:** `loan_approval_data.csv`  
**Shape:** 58,645 rows × 13 columns

| Column | Type | Description |
|--------|------|-------------|
| `id` | int | Unique applicant identifier |
| `age` | float | Applicant age |
| `income` | int | Annual income |
| `home_ownership` | object | RENT / OWN / MORTGAGE / OTHER |
| `employment_length` | int | Years employed |
| `loan_intent` | object | EDUCATION / MEDICAL / PERSONAL / HOMEIMPROVEMENT / VENTURE / DEBTCONSOLIDATION |
| `loan_amount` | int | Requested loan amount |
| `loan_interest_rate` | float | Interest rate (%) |
| `loan_income_ratio` | float | Loan-to-income ratio |
| `payment_default_on_file` | object | Prior default history (Y / N) |
| `credit_history_length` | int | Length of credit history (years) |
| `loan_approval_status` | int | Target — 0 = Rejected, 1 = Approved |
| `max_allowed_loan` | int | Maximum loan amount allowed |

**Class Distribution (Classification):**
- Rejected (0): 85.76% (50,295 samples)
- Approved (1): 14.24% (8,350 samples) — imbalanced dataset

---

## Project Structure

```
Loan-Approval-Prediction/
│
├── data/
│   └── loan_approval_data.csv              # Raw dataset
│
├── notebooks/
│   ├── Siddiha_Rimzan_Notebook1.ipynb      # Data preprocessing & cleaning
│   ├── Siddiha_Rimzan_Notebook2.ipynb      # Classification models (NB, LR, KNN)
│   └── Siddiha_Rimzan_Notebook3.ipynb      # Ensemble classifier + Decision Tree regression
│
├── outputs/
│   ├── dt1_fully_grown.png                 # Decision tree visualisation (fully grown)
│   └── dt2_pruned.png                      # Decision tree visualisation (pruned, depth=4)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Notebooks Summary

### Notebook 1 — Data Preprocessing

Prepares two separate cleaned datasets for the classification and regression tasks.

**Steps performed:**

**Feature Selection**
- Dropped `id` (no predictive value) and `max_allowed_loan` (data leakage) for classification
- Dropped `id` and `loan_approval_status`, kept only approved loans for regression

**Data Quality Checks**
- Fixed typo: `emplyment_length` → `employment_length`
- Missing values detected in `age` (6), `loan_interest_rate` (11), `payment_default_on_file` (5)
- No duplicate rows found
- Unrealistic age values (age > 100): 1 row detected and capped at 100

**Outlier Treatment** — IQR method (cap, not remove)

| Feature | Outliers Capped (Classification) |
|---------|----------------------------------|
| age | 2,446 |
| income | 2,411 |
| employment_length | 1,275 |
| loan_amount | 2,045 |
| credit_history_length | 1,993 |

**Encoding**
- Label Encoding: `payment_default_on_file` (N=0, Y=1)
- One-Hot Encoding (drop_first=True): `home_ownership`, `loan_intent`
- Final feature count after encoding: **16 features**

**Outputs:**
- `dataset1_classification_cleaned.csv` — shape: (58,645 × 17)
- `dataset2_regression_cleaned.csv` — shape: (50,292 × 17)

---

### Notebook 2 — Classification Models

Trains and evaluates three classification algorithms on the loan approval prediction task.

**Setup:**
- 80/20 stratified train-test split (`random_state=42`)
- Training set: 46,902 samples | Test set: 11,726 samples
- StandardScaler applied for Logistic Regression and KNN

**Models trained:**

| Model | Preprocessing |
|-------|--------------|
| Gaussian Naive Bayes (NB) | Raw (unscaled) features |
| Logistic Regression (LR) | StandardScaler |
| K-Nearest Neighbours (KNN) | StandardScaler |

**KNN K-value selection:**  
Elbow Method tested K = 1 to 30. Optimal K selected: **K = 7** (used K = 5 in final model).

**Evaluation outputs per model:**
- Confusion matrix (heatmap)
- Full classification report (precision, recall, F1, support)
- AUC-ROC curves (all 3 models on one plot)
- Comparative metrics summary table

**Hyperparameter Tuning:**  
GridSearchCV on KNN with 5-fold cross-validation, optimising for recall:
```
Best params: {'metric': 'euclidean', 'n_neighbors': 3, 'weights': 'distance'}
Best CV Recall: 0.5538
```

---

### Notebook 3 — Ensemble & Regression

**Part A — Voting Ensemble Classifier**

Combines Naive Bayes and Logistic Regression using soft voting (probability averaging).

- NB wrapped in a `Pipeline` with `FunctionTransformer` (no scaling)
- LR wrapped in a `Pipeline` with `StandardScaler`
- Ensemble strategy: `voting='soft'`

**Part B — Decision Tree Regression**

Predicts `max_allowed_loan` for approved loan applicants.

Two Decision Tree Regressors trained and compared:

| Model | Constraint | Tree Depth | Leaf Nodes |
|-------|-----------|-----------|------------|
| DT-1 | None (fully grown) | 30 | 21,673 |
| DT-2 | `max_depth=4` (pruned) | 4 | 16 |

Both trees visualised and saved (`dt1_fully_grown.png`, `dt2_pruned.png`).

**Feature Importance (DT-2):**

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | income | 0.9126 |
| 2 | age | 0.0874 |
| 3+ | all others | 0.0000 |

**Client Prediction Example (Client 60256):**
```
Age: 56 | Income: £57,000 | Employment: 15 yrs | Loan: £25,700
Loan Intent: MEDICAL | Home: RENT | No prior default
→ DT-2 Predicted Maximum Loan Amount: £85,093.65
```

Decision path traced through DT-2:
```
income (57000) ≤ 78756  → go LEFT
  income (57000) > 53022 → go RIGHT
    age (56) > 28.5      → go RIGHT
      income (57000) ≤ 64999.5 → go LEFT
        → LEAF: £85,093.65
```

---

## Models & Results

### Classification Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Naive Bayes (NB) | 0.8865 | 0.7493 | 0.3044 | 0.4329 | 0.8440 |
| Logistic Regression (LR) | 0.8954 | 0.7264 | 0.4248 | 0.5361 | 0.8864 |
| **KNN (K=5)** | **0.9207** | **0.8302** | **0.5566** | **0.6664** | **0.8590** |
| KNN Tuned (GridSearchCV) | 0.9128 | 0.7576 | 0.5692 | 0.6500 | 0.8398 |
| Voting Ensemble (NB+LR) | 0.8934 | 0.7622 | 0.3649 | 0.4935 | 0.8792 |

> **Best classifier:** KNN (K=5) — highest accuracy (92.07%) and F1-score (0.6664)

### Regression Results

| Model | MSE | RMSE | MAE | R² |
|-------|-----|------|-----|----|
| DT-1 (Fully Grown) | 8,464,656 | £2,909.41 | £992.28 | **0.9946** |
| DT-2 (Pruned, depth=4) | 162,871,124 | £12,762.10 | £8,549.24 | 0.8956 |

> **DT-1** fits training data almost perfectly (R² = 0.9946) but risks overfitting.  
> **DT-2** is interpretable, generalisable, and still achieves R² = 0.8956.

---

## Technologies Used

| Library | Purpose |
|---------|---------|
| `pandas` | Data manipulation and cleaning |
| `numpy` | Numerical operations |
| `plotly` | Interactive charts and visualisations |
| `matplotlib` | Decision tree visualisations |
| `scikit-learn` | ML models, preprocessing, evaluation, GridSearchCV |

**Models used:**
- `GaussianNB` — Naive Bayes classifier
- `LogisticRegression` — Logistic Regression
- `KNeighborsClassifier` — K-Nearest Neighbours
- `VotingClassifier` — Soft voting ensemble
- `DecisionTreeRegressor` — Regression trees
- `GridSearchCV` — Hyperparameter tuning
- `StandardScaler` — Feature normalisation
- `LabelEncoder` + `pd.get_dummies` — Categorical encoding

---

## How to Run

These notebooks were developed on **Google Colab** with data stored in Google Drive. To run locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Loan-Approval-Prediction.git
   cd Loan-Approval-Prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebooks **in order** from the `notebooks/` folder:
   - **Notebook 1** first — generates the cleaned CSV files into `data/`
   - **Notebook 2** — requires `data/dataset1_classification_cleaned.csv`
   - **Notebook 3** — requires both cleaned CSVs from Notebook 1
