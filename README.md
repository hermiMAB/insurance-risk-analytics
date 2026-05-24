# AlphaCare Insurance Solutions (ACIS) - Risk Analytics & Predictive Modeling

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![DVC](https://img.shields.io/badge/DVC-Tracked-orange.svg)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-success.svg)

## 📖 Project Overview
This project focuses on developing an end-to-end insurance risk analytics and predictive modeling pipeline for AlphaCare Insurance Solutions (ACIS). As ACIS expands in the South African auto-insurance market, this project aims to transition the company from intuition-based pricing to an evidence-driven strategy using historical claims data (Feb 2014 - Aug 2015).

### 🎯 Business Objectives
* Identify "low-risk" targets to optimize marketing strategies and reduce premiums to attract new clients.
* Statistically validate risk drivers across provinces, zip codes, and vehicle types.
* Develop predictive machine learning models to estimate claim probability and severity for dynamic, risk-based pricing.

---

## 📂 Project Structure

```text
insurance-risk-analytics/
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions CI/CD pipeline
├── data/                      # Data directory (Ignored by Git, Tracked by DVC)
│   ├── MachineLearningRating_v3.txt
│   └── MachineLearningRating_v3_clean.csv
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis & Visualizations
│   ├── 02_hypothesis_testing.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── visualizations.py      # Modular plotting and EDA functions
│   ├── hypothesis_tests.py
│   └── modeling.py
├── reports/
│   └── final_report.md
├── README.md
├── requirements.txt
└── .dvc/                      # DVC configuration