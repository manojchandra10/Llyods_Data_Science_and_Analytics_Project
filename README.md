# Customer Retention Enhancement at SmartBank (Lloyds Banking Group)

A two-phase data science project on customer churn prediction for SmartBank, a subsidiary of Lloyds Banking Group. The project covers the full workflow from raw data gathering through to a deliverable model artefact, with deliberate attention to leakage diagnostics, subgroup fairness, calibration and stability.

**Author:** Manoj  
**Date:** May 2026

---

## Overview

SmartBank asked for help reducing churn from its retail base. The dataset covers 1,000 customers with a 20.4% churn rate. The work is split into two phases.

**Task 1, Data Gathering and Exploratory Analysis.** Five source tables (demographics, transactions, customer service, online activity, churn labels) were merged into a customer-level table of 1,000 rows with 26 predictor features. Statistical testing covered every categorical and numerical variable. Multicollinearity diagnostics flagged severe VIFs in the transaction aggregates, and temporal-leakage risks were identified for the login features.

**Task 2, Predictive Machine Learning Model.** Five algorithms were compared (dummy baseline, regularised logistic regression, shallow decision tree, random forest, gradient boosting) with proper hyperparameter tuning via GridSearchCV scored on PR-AUC. The work includes sigmoid calibration, subgroup fairness reporting, individual leakage-sensitivity diagnostics, multi-seed stability checks, and a metadata-rich model artefact for handover.

---

## Headline Results

| Metric | Selected model (constrained random forest) |
|---|---|
| Hold-out PR-AUC | 0.234 (no-info floor 0.205) |
| Hold-out ROC-AUC | 0.530 |
| Recall at top 10% | 0.122 (5 of 41 churners) |
| Brier score, uncalibrated | 0.234 |
| Brier score, sigmoid-calibrated | 0.163 (matches dummy baseline) |

The model gives a modest but real lift over random selection in the top 10% of risk scores (5 churners caught against around 2 by chance). The reports document explicitly why no model in the comparison set is statistically separable from any other at this sample size, with the choice of random forest reflecting practical alignment with the retention workflow.

---

## What This Project Demonstrates

The methodological elements that go beyond a basic ML pipeline:

- **Hyperparameter tuning** via GridSearchCV scored on PR-AUC, fitted only on training folds.
- **Class-balanced gradient boosting** via HistGradientBoostingClassifier with `class_weight="balanced"` (the older GradientBoostingClassifier does not support class weighting directly, which would have made the comparison unfair).
- **Sigmoid (Platt) calibration** to fix the uncalibrated Brier score from 0.234 down to 0.163, with the calibration curve plotted side-by-side.
- **Permutation importance** on the held-out set, with caveat colouring when error bars cross zero.
- **Subgroup fairness reporting** across gender, income band and marital status, with small subgroups marked as unstable so that no misleading number gets reported.
- **Individual leakage drops** isolating which login feature carries the leakage exposure (the answer is LoginFrequency alone, which collapses recall@10% from 0.122 to 0.049 when removed).
- **Multi-seed stability check** across 10 random 80/20 splits to verify the model rankings cluster within their own standard deviations.
- **Metadata-rich joblib artefact** containing the trained model, feature column order, hyperparameters, performance summary, leakage warnings, fairness notes and a do-not-use-for note.

---

## Repository Contents

```
.
├── Task1_EDA_Notebook.ipynb           Reproducible exploratory analysis
├── Task1_EDA_Report.docx              7-page Task 1 report
├── Cleaned_Customer_Churn_Data.xlsx   Cleaned dataset (1,000 × 28),
│                                      data dictionary, and Phase 2 modelling notes
├── Task2_Churn_Modelling.ipynb        Reproducible modelling pipeline
├── Task2_Churn_Model_Report.docx      10-page Task 2 report
├── selected_churn_model.joblib        Metadata-rich model artefact
├── task2_model_predictions.csv        Hold-out predictions with risk ranking
└── README.md                          This file
```

---

## How to Run

Place the raw workbook (`Customer_Churn_Data_Large.xlsx`) in the project directory, install the requirements below, then run the notebooks in order.

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scipy scikit-learn openpyxl joblib jupyter

# Task 1 produces Cleaned_Customer_Churn_Data.xlsx
jupyter notebook Task1_EDA_Notebook.ipynb

# Task 2 reads the cleaned workbook directly (no rebuild of the cleaning pipeline)
jupyter notebook Task2_Churn_Modelling.ipynb
```

---

## Tech Stack

- **Python 3.11**
- **pandas, numpy, matplotlib, seaborn** for data handling and visualisation
- **scikit-learn** for modelling, cross-validation, calibration, permutation importance
- **scipy** for statistical testing (chi-square, Mann-Whitney U, point-biserial correlation, Wilson confidence intervals)
- **openpyxl** for workbook input/output
- **joblib** for the model artefact

---

## Note on the Dataset

The supplied dataset shows several hallmarks of a synthetic or training sample: no missing values across 8,000+ records, uniform-looking feature distributions, transaction counts capped at 9 per customer, and service interactions capped at 2 per customer. Findings should be read as methodological practice. Both reports state this caveat explicitly.

The reports also flag three structural risks for any production use: a temporal-leakage concern around the undefined churn measurement date, missing tenure and occupation fields, and a small positive sample of 204 churners which puts a wide confidence interval on every metric.
