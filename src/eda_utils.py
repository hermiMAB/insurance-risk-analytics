import pandas as pd
SEP = "=" * 80
# ════════════════════════════════════════════════════════════════════════════
# 1. DTYPE CASTING
# ════════════════════════════════════════════════════════════════════════════

DATETIME_COLS = ["TransactionMonth"]

INT_COLS = [
    "Cylinders",
    "NumberOfDoors",
    "RegistrationYear"
]

FLOAT_COLS = [
    "cubiccapacity",
    "kilowatts",
    "CustomValueEstimate",
    "CapitalOutstanding",
    "NumberOfVehiclesInFleet",
    "SumInsured",
    "CalculatedPremiumPerTerm",
    "ExcessSelected",
    "TotalPremium",
    "TotalClaims"
]

BOOL_COLS = [
    "IsVATRegistered",
    "AlarmImmobiliser",
    "TrackingDevice",
    "WrittenOff",
    "Rebuilt",
    "Converted",
    "CrossBorder"
]

CAT_COLS = [
    "Citizenship",
    "LegalType",
    "Title",
    "Language",
    "Bank",
    "AccountType",
    "MaritalStatus",
    "Gender",
    "Country",
    "Province",
    "MainCrestaZone",
    "SubCrestaZone",
    "ItemType",
    "VehicleType",
    "make",
    "Model",
    "bodytype",
    "TermFrequency",
    "CoverCategory",
    "CoverType",
    "CoverGroup",
    "Section",
    "Product",
    "StatutoryClass",
    "StatutoryRiskType",
    "NewVehicle"
]

STR_COLS = [
    "UnderwrittenCoverID",
    "PolicyID",
    "PostalCode",
    "mmcode"
]


def cast_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast columns into appropriate datatypes.

    Returns:
        pd.DataFrame: cleaned dataframe with corrected dtypes
    """

    df = df.copy()

    # ════════════════════════════════════════════════════════════════════════
    # DATETIME COLUMNS
    # ════════════════════════════════════════════════════════════════════════
    for col in DATETIME_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(
                df[col],
                errors="coerce"
            )

    # VehicleIntroDate has month/year format like 6/2002
    if "VehicleIntroDate" in df.columns:
        df["VehicleIntroDate"] = pd.to_datetime(
            df["VehicleIntroDate"],
            format="%m/%Y",
            errors="coerce"
        )

    # ════════════════════════════════════════════════════════════════════════
    # INTEGER COLUMNS
    # ════════════════════════════════════════════════════════════════════════
    for col in INT_COLS:
        if col in df.columns:
            df[col] = (
                pd.to_numeric(df[col], errors="coerce")
                .astype("Int64")
            )

    # ════════════════════════════════════════════════════════════════════════
    # FLOAT COLUMNS
    # ════════════════════════════════════════════════════════════════════════
    for col in FLOAT_COLS:
        if col in df.columns:
            df[col] = (
                pd.to_numeric(df[col], errors="coerce")
                .astype("float64")
            )

    # ════════════════════════════════════════════════════════════════════════
    # BOOLEAN COLUMNS
    # ════════════════════════════════════════════════════════════════════════
    bool_map = {
        "Yes": True,
        "No": False,
        "True": True,
        "False": False,
        "1": True,
        "0": False,
        1: True,
        0: False,
        True: True,
        False: False
    }

    for col in BOOL_COLS:
        if col in df.columns:
            df[col] = (
                df[col]
                .map(bool_map)
                .astype("boolean")
            )

    # ════════════════════════════════════════════════════════════════════════
    # CATEGORICAL COLUMNS
    # ════════════════════════════════════════════════════════════════════════
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # ════════════════════════════════════════════════════════════════════════
    # STRING COLUMNS
    # ════════════════════════════════════════════════════════════════════════
    for col in STR_COLS:
        if col in df.columns:
            df[col] = df[col].astype("string")

    return df

def profile_dtypes(df: pd.DataFrame):
    """
    Prints a summary of column data types in the dataframe.

    Groups columns into meaningful categories such as:
    - numeric (int/float)
    - boolean
    - categorical
    - datetime
    - string/object

    Args:
        df (pd.DataFrame): Input dataset

    Returns:
        None (prints output only)
    """

    print(SEP)
    print("1. DTYPE SUMMARY")
    print(SEP)

    groups = {
        "float64": "Numeric (float64)",
        "float32": "Numeric (float32)",
        "Int64": "Numeric (nullable int)",
        "int64": "Numeric (int64)",
        "boolean": "Boolean (nullable)",
        "bool": "Boolean",
        "category": "Categorical",
        "datetime64[ns]": "Datetime",
        "string": "String",
        "object": "Object ← verify/cast"
    }

    for dtype_str, label in groups.items():
        cols = [c for c in df.columns if str(df[c].dtype) == dtype_str]
        if cols:
            print(f"\n  {label.upper()} ({len(cols)})")
            for c in cols:
                print(f"    • {c}")

def profile_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes and prints missing value statistics for each column.

    Shows:
    - total missing values
    - percentage missing
    - flags columns with high missingness (>20%)

    Args:
        df (pd.DataFrame): Input dataset

    Returns:
        pd.DataFrame: Table containing missing counts and percentages
    """

    print("\n" + SEP)
    print("2. MISSING VALUES")
    print(SEP)

    missing = df.isnull().sum().rename("missing").to_frame()
    missing["pct"] = (missing["missing"] / len(df) * 100).round(2)

    missing = missing[missing["missing"] > 0].sort_values("pct", ascending=False)

    if missing.empty:
        print("✓ No missing values found.")
    else:
        print(f"{'Column':<35} {'Missing':>10} {'%':>8}")
        print("-" * 60)

        for col, row in missing.iterrows():
            flag = " ← HIGH" if row["pct"] > 20 else ""
            print(f"{col:<35} {int(row['missing']):>10,} {row['pct']:>7.2f}%{flag}")

    return missing

def profile_numeric(df: pd.DataFrame):
    """
    Generates descriptive statistics for numeric columns.

    Includes:
    - standard descriptive stats (mean, std, quartiles)
    - skewness and kurtosis
    - counts of zeros and negative values

    Args:
        df (pd.DataFrame): Input dataset

    Returns:
        None (prints output only)
    """

    print("\n" + SEP)
    print("3. NUMERIC STATISTICS")
    print(SEP)

    num_df = df.select_dtypes(include="number")

    if num_df.empty:
        print("No numeric columns found.")
        return

    stats = num_df.describe(percentiles=[0.25, 0.5, 0.75, 0.95]).T
    stats["skew"] = num_df.skew(numeric_only=True)
    stats["kurtosis"] = num_df.kurtosis(numeric_only=True)
    stats["zeros"] = (num_df == 0).sum()
    stats["negatives"] = (num_df < 0).sum()

    pd.set_option("display.float_format", "{:,.2f}".format)
    print(stats)
    pd.reset_option("display.float_format")


def profile_categorical(df: pd.DataFrame, top_n: int = 5):
    """
    Analyzes categorical columns by:
    - counting unique values (cardinality)
    - showing most frequent categories

    Args:
        df (pd.DataFrame): Input dataset
        top_n (int): Number of top frequent values to display

    Returns:
        None (prints output only)
    """

    print("\n" + SEP)
    print("4. CATEGORICAL ANALYSIS")
    print(SEP)

    cat_df = df.select_dtypes(include=["category", "string", "object"])

    if cat_df.empty:
        print("No categorical columns found.")
        return

    print(f"{'Column':<30} {'Unique':>8} Top values")
    print("-" * 80)

    for col in cat_df.columns:
        top = df[col].value_counts(dropna=False).head(top_n)
        top_str = ", ".join(f"{v}({c:,})" for v, c in top.items())

        print(f"{col:<30} {df[col].nunique():>8} {top_str}")

def profile_datetime(df: pd.DataFrame):
    """
    Analyzes datetime columns by showing:
    - minimum date
    - maximum date
    - number of missing values

    Args:
        df (pd.DataFrame): Input dataset

    Returns:
        None (prints output only)
    """

    print("\n" + SEP)
    print("5. DATETIME ANALYSIS")
    print(SEP)

    dt_df = df.select_dtypes(include=["datetime64[ns]"])

    if dt_df.empty:
        print("No datetime columns found.")
        return

    for col in dt_df.columns:
        print(f"\n{col}")
        print(f"  min  : {df[col].min()}")
        print(f"  max  : {df[col].max()}")
        print(f"  nulls: {df[col].isnull().sum():,}")


def profile_memory(df: pd.DataFrame):
    """
    Provides dataset-level summary including:
    - number of rows and columns
    - memory usage
    - duplicate row count

    Args:
        df (pd.DataFrame): Input dataset

    Returns:
        None (prints output only)
    """

    print("\n" + SEP)
    print("6. DATASET SUMMARY")
    print(SEP)

    print(f"Rows     : {len(df):,}")
    print(f"Columns  : {df.shape[1]}")

    mem = df.memory_usage(deep=True).sum() / 1e6
    print(f"Memory   : {mem:.2f} MB")

    duplicates = df.duplicated().sum()
    print(f"Duplicates: {duplicates:,}")


def pre_eda_profile(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Full pre-EDA profiling pipeline.

    Executes:
    1. dtype inspection
    2. missing value analysis
    3. numeric statistics
    4. categorical profiling
    5. datetime analysis
    6. dataset summary

    Args:
        df (pd.DataFrame): Input dataset
        top_n (int): Number of top categories to display

    Returns:
        pd.DataFrame: Missing value report for further analysis
    """

    profile_dtypes(df)
    missing = profile_missing_values(df)
    profile_numeric(df)
    profile_categorical(df, top_n=top_n)
    profile_datetime(df)
    profile_memory(df)

    return missing

import pandas as pd
import numpy as np

def handle_missing_values(
    df: pd.DataFrame,
    drop_threshold: float = 0.60
) -> pd.DataFrame:
    """
    Handle missing values using scalable and business-aware
    preprocessing strategies for insurance datasets.

    Strategy Overview
    -----------------
    1. Drop columns with extremely high missingness (>60% missing by default).
    2. Impute continuous numeric features using median.
    3. Impute discrete/count vehicle features using mode.
    4. Infer missing Gender values deterministically from the Title column.
    5. Fill categorical missing values with 'Unknown' while preserving category dtype.
    6. Preserve missing datetime values (NaT) to prevent temporal distortion.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    drop_threshold : float, default=0.60
        Missing value percentage threshold used for dropping columns.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with handled missing values.
    """

    # ============================================================
    # COPY DATAFRAME
    # ============================================================

    # Prevent modifying original raw dataset
    df = df.copy()

    print("=" * 72)
    print("MISSING VALUE HANDLING")
    print("=" * 72)

    # ============================================================
    # 1. DROP EXTREMELY MISSING COLUMNS
    # ============================================================

    drop_cols = [
        col for col in df.columns
        if df[col].isnull().mean() > drop_threshold
    ]

    print("\n1. DROPPING EXTREMELY MISSING COLUMNS")

    if drop_cols:
        for col in drop_cols:
            pct = round(df[col].isnull().mean() * 100, 2)
            print(f"  • {col} ({pct}% missing)")

        df.drop(columns=drop_cols, inplace=True)

    else:
        print("  • No columns exceeded threshold.")

    # ============================================================
    # 2. CONTINUOUS NUMERIC IMPUTATION (MEDIAN)
    # ============================================================

    continuous_cols = [
        "cubiccapacity",
        "kilowatts",
        "CapitalOutstanding",
    ]

    continuous_cols = [
        col for col in continuous_cols
        if col in df.columns
    ]

    print("\n2. MEDIAN IMPUTATION — CONTINUOUS FEATURES")

    if continuous_cols:

        median_map = {
            col: df[col].median()
            for col in continuous_cols
        }

        for col, value in median_map.items():

            missing_before = df[col].isnull().sum()

            print(
                f"  • {col}: "
                f"filled {missing_before:,} missing "
                f"values with median = {round(value, 2)}"
            )

        df.fillna(value=median_map, inplace=True)

    # ============================================================
    # 3. DISCRETE / COUNT FEATURE IMPUTATION (MODE)
    # ============================================================

    discrete_cols = [
        "Cylinders",
        "NumberOfDoors"
    ]

    discrete_cols = [
        col for col in discrete_cols
        if col in df.columns
    ]

    print("\n3. MODE IMPUTATION — DISCRETE FEATURES")

    for col in discrete_cols:

        mode_value = df[col].mode(dropna=True)

        if not mode_value.empty:

            mode_value = mode_value[0]

            missing_before = df[col].isnull().sum()

            df[col] = df[col].fillna(mode_value)

            print(
                f"  • {col}: "
                f"filled {missing_before:,} missing "
                f"values with mode = {mode_value}"
            )

    # ============================================================
    # 3.5 INFER GENDER FROM TITLE
    # ============================================================
    
    # ============================================================
    # 3.5 INFER GENDER FROM TITLE (BUSINESS-AWARE)
    # ============================================================
    
    print("\n3.5. INFERRING GENDER FROM TITLE")
    
    if 'Gender' in df.columns and 'Title' in df.columns and 'LegalType' in df.columns:
        
        # 1. Strip hidden spaces from Title (fixes the "Mr " vs "Mr" bug)
        df['Title'] = df['Title'].astype(str).str.strip()
        
        # 2. Force 'Not specified' to officially be a Pandas NaN
        df['Gender'] = np.where(
            df['Gender'].astype(str).str.contains('not specified', case=False, na=False),
            np.nan,
            df['Gender']
        )
        
        missing_gender_before = df['Gender'].isnull().sum()
        
        # 3. Define what constitutes a "Human" in your LegalType column
        # Update this list based on what your dataset actually calls individuals!
        human_legal_types = ['Individual', 'Sole proprieter'] 
        
        # Create a filter mask: Only rows that are humans AND missing gender
        is_human = df['LegalType'].astype(str).str.strip().isin(human_legal_types)
        is_missing_gender = df['Gender'].isna()
        
        # 4. Define the mapping
        title_gender_map = {
            'Mr': 'Male',
            'Mrs': 'Female',
            'Ms': 'Female',
            'Miss': 'Female'
        }
        
        # 5. Map the titles to genders ONLY for the filtered human rows
        inferred_genders = df.loc[is_human & is_missing_gender, 'Title'].map(title_gender_map)
        
        # Fill only those specific gaps
        df.loc[is_human & is_missing_gender, 'Gender'] = inferred_genders
        
        missing_gender_after = df['Gender'].isnull().sum()
        filled_count = missing_gender_before - missing_gender_after
        
        print(f"  • Gender: Deterministically inferred {filled_count:,} human values from Title.")
        print(f"  • Gender: Left Corporate/Commercial entities as Unknown/NaN.")

    # ============================================================
    # 4. CATEGORICAL IMPUTATION
    # ============================================================

    categorical_cols = [
        "Bank",
        "AccountType",
        "Gender",
        "MaritalStatus",
        "VehicleType",
        "make",
        "Model",
        "bodytype",
        "NewVehicle"
    ]

    categorical_cols = [
        col for col in categorical_cols
        if col in df.columns
    ]

    print("\n4. CATEGORICAL IMPUTATION (FALLBACK)")

    for col in categorical_cols:

        missing_before = df[col].isnull().sum()
        
        if missing_before > 0:
            # Preserve category dtype
            if str(df[col].dtype) == "category":

                if "Unknown" not in df[col].cat.categories:

                    df[col] = df[col].cat.add_categories(
                        ["Unknown"]
                    )

            df[col] = df[col].fillna("Unknown")

            print(
                f"  • {col}: "
                f"filled {missing_before:,} missing "
                f"values with 'Unknown'"
            )

    # ============================================================
    # 5. DATETIME HANDLING
    # ============================================================

    print("\n5. DATETIME HANDLING")

    datetime_cols = [
        "VehicleIntroDate"
    ]

    datetime_cols = [
        col for col in datetime_cols
        if col in df.columns
    ]

    for col in datetime_cols:

        missing_dates = df[col].isnull().sum()

        if missing_dates > 0:
            print(
                f"  • {col}: preserved "
                f"{missing_dates:,} missing values as NaT"
            )

    # ============================================================
    # 6. FINAL VALIDATION
    # ============================================================

    print("\n6. FINAL VALIDATION")

    remaining_missing = (
        df.isnull()
        .sum()
        .sort_values(ascending=False)
    )

    remaining_missing = (
        remaining_missing[remaining_missing > 0]
    )

    if remaining_missing.empty:

        print("  • No remaining missing values.")

    else:

        print("  • Remaining missing values:\n")
        print(remaining_missing)
    # Check how many we are starting with
    missing_mmcode_before = df['mmcode'].isnull().sum()
    print(f"Rows missing 'mmcode' before: {missing_mmcode_before}")

    # Drop the rows where 'mmcode' is NaN
    df.dropna(subset=['mmcode'], inplace=True)

    # Verify they are gone
    missing_mmcode_after = df['mmcode'].isnull().sum()
    print(f"Rows missing 'mmcode' after: {missing_mmcode_after}")
    print(f"Total rows remaining in dataset: {len(df):,}")

    # ============================================================
    # 7. MEMORY SUMMARY
    # ============================================================

    memory_mb = (
        df.memory_usage(deep=True).sum() / 1e6
    )

    print("\n7. MEMORY USAGE")

    print(f"  • Dataset memory usage: {memory_mb:.2f} MB")

    print("\nMissing value handling completed.")
    print("=" * 72)

    return df

if __name__ == "__main__":
    # This only runs when executed via terminal: python src/eda_utils.py
    import pandas as pd
    
    # 1. Load the raw data
    raw_df = pd.read_csv("data/MachineLearningRating_v3.txt", sep="|") 
    
    # 2. Run your cleaning functions
    df_casted = cast_columns(raw_df)
    df_clean = handle_missing_values(df_casted)
    
    # 3. Save the output for DVC to track
    df_clean.to_csv("data/MachineLearningRating_v3_clean.csv", index=False)