# -----------------------------------------------
# model.py
# -----------------------------------------------
# Purpose: Handles everything related to model training and evaluation.
# Five responsibilities:
#   1. Split data into training and testing sets (Handled in prep phase for this pipeline)
#   2. Train four regression models on the training set
#   3. Evaluate each model on the test set using MAE, MSE, RMSE, R²
#   4. Plot bar charts comparing model performance
#   5. Save the trained models for future deployment (Serialization)
# -----------------------------------------------

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib  # <-- NEW: Used for saving models
import os      # <-- NEW: Used for creating folders

# -----------------------------------------------
# FUNCTION 1: Split Data into Train and Test Sets
# -----------------------------------------------
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# -----------------------------------------------
# FUNCTION 2: Train All Four Models
# -----------------------------------------------
def train_models(X_train, y_train):
    print("Training Linear Regression...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    print("Training Decision Tree...")
    dt_model = DecisionTreeRegressor(random_state=42)
    dt_model.fit(X_train, y_train)

    # Note: n_jobs=-1 uses all CPU cores to speed up Random Forest
    print("Training Random Forest (This may take a moment)...")
    rfr_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    rfr_model.fit(X_train, y_train)

    print("Training XGBoost...")
    xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train)

    return lr_model, dt_model, rfr_model, xgb_model


# -----------------------------------------------
# FUNCTION 3: Evaluate a Single Model
# -----------------------------------------------
def evaluate_model(model, X_test, y_test):
    # Generate predictions on the held-out test set
    y_pred = model.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mae, mse, r2, y_pred


# -----------------------------------------------
# FUNCTION 4: Plot Model Comparison Bar Charts
# -----------------------------------------------
def plot_metrics(models, mae_scores, mse_scores, r2_scores):
    # CHART 1: MAE Comparison
    plt.figure(figsize=(8, 5))
    plt.bar(models, mae_scores, color='skyblue')
    plt.xlabel('Models', fontweight='bold')
    plt.ylabel('Mean Absolute Error (MAE)', fontweight='bold')
    plt.title('Comparison of MAE Scores — Lower is Better', fontweight='bold')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # CHART 2: MSE Comparison
    plt.figure(figsize=(8, 5))
    plt.bar(models, mse_scores, color='lightgreen')
    plt.xlabel('Models', fontweight='bold')
    plt.ylabel('Mean Squared Error (MSE)', fontweight='bold')
    plt.title('Comparison of MSE Scores — Lower is Better', fontweight='bold')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # CHART 3: R² Comparison
    plt.figure(figsize=(8, 5))
    plt.bar(models, r2_scores, color='salmon')
    plt.xlabel('Models', fontweight='bold')
    plt.ylabel('R-squared Score', fontweight='bold')
    plt.title('Comparison of R² Scores — Higher is Better', fontweight='bold')
    plt.axhline(0, color='black', linewidth=1) # Add a zero line just in case of negative R2
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# -----------------------------------------------
# FUNCTION 5: Save Trained Models (NEW)
# -----------------------------------------------
def save_trained_models(lr, dt, rfr, xgb_mod, output_dir="models"):
    # Create the folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nExporting models to '{output_dir}/' directory...")
    
    # Save each model as a .joblib file
    joblib.dump(lr, os.path.join(output_dir, "linear_regression.joblib"))
    joblib.dump(dt, os.path.join(output_dir, "decision_tree.joblib"))
    joblib.dump(rfr, os.path.join(output_dir, "random_forest.joblib"))
    joblib.dump(xgb_mod, os.path.join(output_dir, "xgboost.joblib"))
    
    print("[SUCCESS] All 4 models have been successfully saved!")


# -----------------------------------------------
# EXECUTION BLOCK (Runs when the script is executed)
# -----------------------------------------------
# -----------------------------------------------
# EXECUTION BLOCK (Runs when the script is executed)
# -----------------------------------------------
if __name__ == "__main__":
    print("="*50)
    print(" PHASE 1: LOADING PREPARED DATA & SPLITTING")
    print("="*50)
    
    # 1. Load the single preprocessed dataset (THIS IS THE FIX)
    df = pd.read_csv("data/preprocessed_data.csv", low_memory=False)
    
    # 2. Separate features (X) and target (y)
    target_col = 'TotalClaims'
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 3. Split the data right here before training
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Clean column names for XGBoost
    X_train.columns = X_train.columns.str.replace(r'[\[\]<]', '', regex=True)
    X_test.columns = X_test.columns.str.replace(r'[\[\]<]', '', regex=True)

    print("\n" + "="*50)
    print(" PHASE 2: TRAINING MODELS")
    print("="*50)
    
    lr, dt, rfr, xgb_mod = train_models(X_train, y_train)

    print("\n" + "="*50)
    print(" PHASE 3: EVALUATING MODELS")
    print("="*50)
    
    mae_lr, mse_lr, r2_lr, _ = evaluate_model(lr, X_test, y_test)
    mae_dt, mse_dt, r2_dt, _ = evaluate_model(dt, X_test, y_test)
    mae_rfr, mse_rfr, r2_rfr, _ = evaluate_model(rfr, X_test, y_test)
    mae_xgb, mse_xgb, r2_xgb, _ = evaluate_model(xgb_mod, X_test, y_test)

    models_list = ['Linear Reg', 'Decision Tree', 'Random Forest', 'XGBoost']
    r2_scores = [r2_lr, r2_dt, r2_rfr, r2_xgb]
    mae_scores = [mae_lr, mae_dt, mae_rfr, mae_xgb]
    mse_scores = [mse_lr, mse_dt, mse_rfr, mse_xgb]

    print(f"{'Model':<20} | {'RMSE':<15} | {'R-Squared (R²)'}")
    print("-" * 55)
    for i in range(len(models_list)):
        rmse = np.sqrt(mse_scores[i])
        print(f"{models_list[i]:<20} | ${rmse:<14,.2f} | {r2_scores[i]:.4f}")

    print("\n" + "="*50)
    print(" PHASE 4: SAVING MODELS")
    print("="*50)
    
    # Trigger the saving function
    save_trained_models(lr, dt, rfr, xgb_mod)

    print("\n" + "="*50)
    print(" PHASE 5: VISUALIZING PERFORMANCE")
    print("="*50)
    
    plot_metrics(models_list, mae_scores, mse_scores, r2_scores)