# -----------------------------------------------
# model.py
# -----------------------------------------------
# Purpose: Handles everything related to model training and evaluation.
# -----------------------------------------------

import pandas as pd
import numpy as np
import logging
import sys
import os
import joblib
import matplotlib.pyplot as plt

# Regression Imports
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Classification Imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# -----------------------------------------------
# CONFIGURE LOGGING
# -----------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# -----------------------------------------------
# FUNCTION 1: Split Data
# -----------------------------------------------
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# -----------------------------------------------
# FUNCTION 2: Train Regression Models (Severity)
# -----------------------------------------------
def train_models(X_train, y_train):
    try:
        logging.info("Training Linear Regression...")
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)

        logging.info("Training Decision Tree...")
        dt_model = DecisionTreeRegressor(random_state=42)
        dt_model.fit(X_train, y_train)

        logging.info("Training Random Forest (This may take a moment)...")
        rfr_model = RandomForestRegressor(random_state=42, n_jobs=-1)
        rfr_model.fit(X_train, y_train)

        logging.info("Training XGBoost...")
        xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
        xgb_model.fit(X_train, y_train)

        logging.info("All regression models trained successfully.")
        return lr_model, dt_model, rfr_model, xgb_model
        
    except Exception as e:
        logging.error(f"A failure occurred during regression training: {e}")
        raise

# -----------------------------------------------
# FUNCTION 3: Evaluate Regression Model
# -----------------------------------------------
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mae, mse, r2, y_pred

# -----------------------------------------------
# FUNCTION 4: Train Classification Models (Probability)
# -----------------------------------------------
def train_classifiers(X_train, y_train):
    try:
        logging.info("Training Logistic Regression...")
        lr_class = LogisticRegression(max_iter=1000)
        lr_class.fit(X_train, y_train)

        logging.info("Training Random Forest Classifier...")
        # Max depth limited to prevent freezing on large datasets
        rf_class = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=10)
        rf_class.fit(X_train, y_train)

        logging.info("Training XGBoost Classifier...")
        xgb_class = xgb.XGBClassifier(random_state=42, n_jobs=-1, max_depth=6)
        xgb_class.fit(X_train, y_train)

        logging.info("All classifiers trained successfully.")
        return lr_class, rf_class, xgb_class
    except Exception as e:
        logging.error(f"A failure occurred during classifier training: {e}")
        raise

# -----------------------------------------------
# FUNCTION 5: Evaluate Classification Model
# -----------------------------------------------
def evaluate_classifier(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] 
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    return acc, prec, rec, f1, auc

# -----------------------------------------------
# FUNCTION 6: Plot Metrics (Regression only)
# -----------------------------------------------
def plot_metrics(models, mae_scores, mse_scores, r2_scores):
    logging.info("Generating performance visualizations...")
    
    plt.figure(figsize=(8, 5))
    plt.bar(models, r2_scores, color='salmon')
    plt.xlabel('Models', fontweight='bold')
    plt.ylabel('R-squared Score', fontweight='bold')
    plt.title('Comparison of R² Scores — Higher is Better', fontweight='bold')
    plt.axhline(0, color='black', linewidth=1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# -----------------------------------------------
# FUNCTION 7: Save Trained Models
# -----------------------------------------------
def save_trained_models(lr, dt, rfr, xgb_mod, output_dir="models"):
    try:
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Exporting models to '{output_dir}/' directory...")
        
        joblib.dump(lr, os.path.join(output_dir, "linear_regression.joblib"))
        joblib.dump(dt, os.path.join(output_dir, "decision_tree.joblib"))
        joblib.dump(rfr, os.path.join(output_dir, "random_forest.joblib"))
        joblib.dump(xgb_mod, os.path.join(output_dir, "xgboost.joblib"))
        
        logging.info("All 4 regression models successfully saved!")
    except Exception as e:
        logging.error(f"Failed to save models to disk: {e}")
        raise


# -----------------------------------------------
# EXECUTION BLOCK
# -----------------------------------------------
if __name__ == "__main__":
    logging.info("PHASE 1: LOADING PREPARED DATA")
    
    try:
        df = pd.read_csv("data/preprocessed_data.csv", low_memory=False)
        logging.info(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
    except FileNotFoundError:
        logging.error("Cannot find 'data/preprocessed_data.csv'. Check your file path or run the preprocessing script first.")
        sys.exit(1)
    
    # -----------------------------------------------
    # PIPELINE A: REGRESSION (CLAIM SEVERITY)
    # -----------------------------------------------
    logging.info("--- STARTING REGRESSION PIPELINE (SEVERITY) ---")
    target_col = 'TotalClaims'
    X_reg = df.drop(columns=[target_col])
    y_reg = df[target_col]
    
    X_train_r, X_test_r, y_train_r, y_test_r = split_data(X_reg, y_reg, test_size=0.2)
    
    X_train_r.columns = X_train_r.columns.str.replace(r'[\[\]<]', '', regex=True)
    X_test_r.columns = X_test_r.columns.str.replace(r'[\[\]<]', '', regex=True)

    lr, dt, rfr, xgb_mod = train_models(X_train_r, y_train_r)

    mae_lr, mse_lr, r2_lr, _ = evaluate_model(lr, X_test_r, y_test_r)
    mae_dt, mse_dt, r2_dt, _ = evaluate_model(dt, X_test_r, y_test_r)
    mae_rfr, mse_rfr, r2_rfr, _ = evaluate_model(rfr, X_test_r, y_test_r)
    mae_xgb, mse_xgb, r2_xgb, _ = evaluate_model(xgb_mod, X_test_r, y_test_r)

    models_list = ['Linear Reg', 'Decision Tree', 'Random Forest', 'XGBoost']
    r2_scores = [r2_lr, r2_dt, r2_rfr, r2_xgb]
    mae_scores = [mae_lr, mae_dt, mae_rfr, mae_xgb]
    mse_scores = [mse_lr, mse_dt, mse_rfr, mse_xgb]

    print(f"\n{'Regression Model':<20} | {'RMSE':<15} | {'R-Squared (R²)'}")
    print("-" * 55)
    for i in range(len(models_list)):
        rmse = np.sqrt(mse_scores[i])
        print(f"{models_list[i]:<20} | ${rmse:<14,.2f} | {r2_scores[i]:.4f}")

    save_trained_models(lr, dt, rfr, xgb_mod)

    # -----------------------------------------------
    # PIPELINE B: CLASSIFICATION (CLAIM PROBABILITY)
    # -----------------------------------------------
    logging.info("--- STARTING CLASSIFICATION PIPELINE (PROBABILITY) ---")
    df_class = df.copy()
    df_class['Has_Claim'] = (df_class['TotalClaims'] > 0).astype(int)
    
    X_class = df_class.drop(columns=['TotalClaims', 'Has_Claim'])
    y_class = df_class['Has_Claim']

    X_train_c, X_test_c, y_train_c, y_test_c = split_data(X_class, y_class, test_size=0.2)
    
    X_train_c.columns = X_train_c.columns.str.replace(r'[\[\]<]', '', regex=True)
    X_test_c.columns = X_test_c.columns.str.replace(r'[\[\]<]', '', regex=True)

    lr_class, rf_class, xgb_class = train_classifiers(X_train_c, y_train_c)

    metrics_lr = evaluate_classifier(lr_class, X_test_c, y_test_c)
    metrics_rf = evaluate_classifier(rf_class, X_test_c, y_test_c)
    metrics_xgb = evaluate_classifier(xgb_class, X_test_c, y_test_c)

    print(f"\n{'Classifier Model':<20} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'ROC-AUC'}")
    print("-" * 85)
    class_models = [('Logistic Reg', metrics_lr), ('Random Forest', metrics_rf), ('XGBoost', metrics_xgb)]
    for name, metrics in class_models:
        print(f"{name:<20} | {metrics[0]:<10.4f} | {metrics[1]:<10.4f} | {metrics[2]:<10.4f} | {metrics[3]:<10.4f} | {metrics[4]:.4f}")
    print("\n")
    
    logging.info("PIPELINES COMPLETE.")