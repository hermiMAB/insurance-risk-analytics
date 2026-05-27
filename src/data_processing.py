import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def handle_missing_data(df: pd.DataFrame, drop_threshold: float = 0.50) -> pd.DataFrame:
    df = df.copy()
    print("\n" + "="*50)
    print(" PHASE 1: MISSING DATA HANDLING")
    print("="*50)
    
    # 1. Drop columns with > 50% missing data
    missing_ratios = df.isnull().mean()
    cols_to_drop = missing_ratios[missing_ratios > drop_threshold].index.tolist()
    
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        print(f"[DROPPED] Columns exceeding {drop_threshold*100}% missing data:")
        for col in cols_to_drop:
            print(f"  - {col} ({missing_ratios[col]*100:.1f}%)")
    
    # 2. Impute remaining missing values
    num_cols = df.select_dtypes(include=['number']).columns
    cat_cols = df.select_dtypes(include=['object', 'category', 'string']).columns
    
    print("\n[IMPUTED] Filling remaining missing values:")
    print(f"  - Numeric columns ({len(num_cols)}): Filled with Median.")
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
            
    print(f"  - Categorical columns ({len(cat_cols)}): Filled with 'Unknown'.")
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna('Unknown')
            
    return df

def engineer_and_drop(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    print("\n" + "="*50)
    print(" PHASE 2: FEATURE ENGINEERING & CLEANUP")
    print("="*50)
    
    # 1. Calculate Vehicle Age
    if 'TransactionMonth' in df.columns and 'RegistrationYear' in df.columns:
        df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'])
        df['VehicleAge'] = df['TransactionMonth'].dt.year - df['RegistrationYear']
        df['VehicleAge'] = df['VehicleAge'].apply(lambda x: max(x, 0)) # Ensure no negative ages
        print("[ENGINEERED] Created new feature: 'VehicleAge'")
        
    # 2. Drop Identifiers, Dates, and High-Cardinality Noise
    unnecessary_cols = [
        'UnderwrittenCoverID', 'PolicyID', 'mmcode',           # Unique Identifiers
        'TransactionMonth', 'VehicleIntroDate',                # Dates (handled or irrelevant)
        'PostalCode', 'make', 'Model', 'SubCrestaZone', 'Bank' # Too many unique values (will crash One-Hot)
    ]
    
    unnecessary_cols.extend(['Has_Claim', 'Margin'])
    
    cols_to_drop = [col for col in unnecessary_cols if col in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)
    
    print(f"[DROPPED] Unnecessary / Identifier columns removed:")
    for col in cols_to_drop:
        print(f"  - {col}")
    
    return df

def encode_and_scale(df: pd.DataFrame, target_col: str = 'TotalClaims') -> pd.DataFrame:
    print("\n" + "="*50)
    print(" PHASE 3: SCALING & ENCODING")
    print("="*50)
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 1. MIN-MAX SCALING
    num_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    print(f"[SCALED] Min-Max Scaling applied to {len(num_cols)} numerical columns.")
    
    scaler = MinMaxScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])
    
    # 2. ONE-HOT ENCODING
    cat_cols = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    print(f"[ENCODED] One-Hot Encoding applied to {len(cat_cols)} categorical columns (drop_first=True).")
    
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=int)
    print(f"  - Total feature dimensions expanded to: {X.shape[1]} columns.")
    
    # 3. RECOMBINE DATASET
    preprocessed_df = pd.concat([X, y], axis=1)
    return preprocessed_df

# === CHANGED output_dir TO "data" ===
def save_preprocessed_data(df: pd.DataFrame, output_dir="data", filename="preprocessed_data.csv"):
    print("\n" + "="*50)
    print(" PHASE 4: SAVING DATA FOR DVC & MODELING")
    print("="*50)
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    df.to_csv(output_path, index=False)
    
    print(f"[SAVED] File successfully exported to '{output_path}':")
    print(f"  - Total Rows: {df.shape[0]:,}")
    print(f"  - Total Columns: {df.shape[1]}")

if __name__ == "__main__":
    print("Loading raw cleaned dataset...")
    raw_df = pd.read_csv("data/MachineLearningRating_v3_clean.csv", low_memory=False)
    
    df_missing_handled = handle_missing_data(raw_df, drop_threshold=0.50)
    df_engineered = engineer_and_drop(df_missing_handled)
    df_preprocessed = encode_and_scale(df_engineered, target_col='TotalClaims')
    
    save_preprocessed_data(df_preprocessed)