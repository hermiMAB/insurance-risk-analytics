# AlphaCare Insurance Solutions (ACIS) - Risk Analytics & Predictive Modeling

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![DVC](https://img.shields.io/badge/DVC-Tracked-orange.svg)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-success.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Enabled-yellow)
![XGBoost](https://img.shields.io/badge/XGBoost-Powered-red)

## 📖 Project Overview
This project delivers an end-to-end insurance risk analytics and machine learning pipeline for AlphaCare Insurance Solutions (ACIS). As ACIS expands within the South African auto-insurance market, this repository provides the statistical foundation and predictive architecture required to transition from intuition-based underwriting to a dynamic, evidence-driven pricing strategy.

Using historical claims data (Feb 2014 - Aug 2015), this project isolates key risk drivers, validates geographical and demographic hypotheses, and deploys predictive models to forecast claim likelihood and severity.

### 🎯 Business Objectives
* **Optimize Marketing:** Identify "low-risk" demographics to target with reduced, highly competitive premiums.
* **Validate Risk Drivers:** Use rigorous A/B testing to prove the impact of geographic location, gender, and vehicle specifications on claim rates.
* **Dynamic Pricing:** Develop robust machine learning pipelines to predict both *if* a customer will crash (Probability) and *how much* it will cost (Severity).
* **Interpretability:** Crack open black-box algorithms using SHAP to provide pricing actuaries with transparent, actionable business rules.

---

## 📊 Key Findings & Insights
Through rigorous Exploratory Data Analysis (EDA) and A/B Hypothesis Testing, several critical business insights were uncovered:
* **Geographical Risk:** Claim severity varies significantly by province. For example, KwaZulu-Natal exhibits significantly higher average claim severity (+33.1%) compared to Gauteng.
* **The Gender Nuance:** There is a statistically significant difference in claim *frequency* between genders, but absolutely no statistical difference in claim *severity*. (Gender affects how often crashes happen, but physics dictates how much they cost).
* **Margin Consistency:** Despite regional risk variations, the current premium logic successfully balances out localized claim risks to maintain consistent profit margins across different zip codes.

---

## 🧠 Machine Learning Architecture
To accurately price risk, the modeling pipeline is split into two distinct tracks, followed by advanced interpretability mapping.

1. **Pipeline A: Claim Severity (Regression)**
   * **Goal:** Predict the exact financial cost of a claim (`TotalClaims`).
   * **Models Evaluated:** Linear Regression, Decision Tree, Random Forest, XGBoost.
   * **Note:** Heavy-tailed outliers in insurance data mean tree-based models perform well on absolute error (MAE) but struggle with squared error metrics (MSE/R²).

2. **Pipeline B: Claim Probability (Classification)**
   * **Goal:** Predict the binary likelihood of a claim occurring (`Has_Claim`).
   * **Models Evaluated:** Logistic Regression, Random Forest Classifier, XGBoost Classifier.

3. **Model Interpretability (SHAP)**
   * We utilize SHAP (SHapley Additive exPlanations) on the XGBoost severity model to extract business rules. 
   * **Key Driver:** Financial baselines (`TotalPremium`, `SumInsured`) and `CoverType_Own Damage` are the primary drivers of extreme claim costs, while higher `VehicleAge` actively reduces predicted severity.

---

## 📂 Repository Structure

```text
insurance-risk-analytics/
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions CI/CD pipeline
├── data/                      # Tracked by DVC (Ignored by Git)
│   ├── MachineLearningRating_v3.txt
│   ├── MachineLearningRating_v3_clean.csv
│   └── preprocessed_data.csv  # Final engineered & encoded dataset
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   ├── 02_hypothesis_testing.ipynb # A/B Testing & Statistical Validation
│   └── 03_modeling.ipynb      # ML Training, Evaluation & SHAP Interpretability
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Data ingestion utilities
│   ├── visualizations.py      # Modular plotting functions
│   ├── hypothesis_tests.py    # Statistical testing wrappers
│   └── modeling.py            # Automated dual-pipeline ML training script
├── models/                    # Serialized .joblib models (Tracked by DVC)
├── reports/
│   └── final_report.md
├── README.md
├── requirements.txt
└── .dvc/                      # Data Version Control configuration

## How to Reproduce the Data Pipeline
To fetch the version-controlled datasets and trained models for this project, ensure you have DVC installed and run the following command in your terminal:
`dvc pull`

Alternatively, you can reproduce the entire data processing and modeling pipeline locally from scratch by running:
`dvc repro`

**Tracked Artifacts & Datasets:**
* `data/MachineLearningRating_v3.txt`: The original, unaltered claims dataset.
* `data/MachineLearningRating_v3_clean.csv`: The cleaned dataset after initial EDA and missing value handling.
* `data/preprocessed_data.csv`: The final engineered, scaled, and encoded dataset ready for machine learning.
* `models/`: Contains the serialized predictive models trained to estimate claim severity:
  * `linear_regression.joblib`
  * `decision_tree.joblib`
  * `random_forest.joblib`
  * `xgboost.joblib`
