import os
import pandas as pd
import numpy as np


# ============================================================
# 1. DATA QUALITY AUDIT
# ============================================================

def audit_dataset(df: pd.DataFrame, name: str) -> None:
    """
    Performs a comprehensive quality audit:
    - dimensions
    - missing values
    - duplicate rows
    - data types
    - invalid values based on BRFSS coding ranges
    """

    print(f"\n{'=' * 70}")
    print(f"DATA QUALITY AUDIT: {name}")
    print(f"{'=' * 70}")

    # Basic structure
    print(f"\nDimensions: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Missing values
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()

    print(f"\nTotal Missing Values: {total_nulls:,}")

    if total_nulls > 0:
        print("\nColumns with missing values:")
        print(null_counts[null_counts > 0])

    # Duplicate rows
    duplicate_count = df.duplicated().sum()

    print(
        f"\nIdentical Survey Profiles: "
        f"{duplicate_count:,} "
        f"({duplicate_count / len(df) * 100:.2f}%)"
    )

    # Data types
    print("\nData Types:")
    print(df.dtypes.value_counts())

    # --------------------------------------------------------
    # Valid BRFSS value ranges
    # --------------------------------------------------------

    valid_ranges = {
        "Diabetes_binary": {0, 1},
        "Diabetes_012": {0, 1, 2},

        "HighBP": {0, 1},
        "HighChol": {0, 1},
        "CholCheck": {0, 1},
        "Smoker": {0, 1},
        "Stroke": {0, 1},
        "HeartDiseaseorAttack": {0, 1},
        "PhysActivity": {0, 1},
        "Fruits": {0, 1},
        "Veggies": {0, 1},
        "HvyAlcoholConsump": {0, 1},
        "AnyHealthcare": {0, 1},
        "NoDocbcCost": {0, 1},
        "DiffWalk": {0, 1},
        "Sex": {0, 1},

        "GenHlth": set(range(1, 6)),
        "Age": set(range(1, 14)),
        "Education": set(range(1, 7)),
        "Income": set(range(1, 9)),

        "MentHlth": set(range(0, 31)),
        "PhysHlth": set(range(0, 31)),
    }

    print("\nInvalid Values:")

    invalid_found = False

    for column, valid_values in valid_ranges.items():

        if column not in df.columns:
            continue

        invalid_mask = ~df[column].isin(valid_values) & df[column].notna()
        invalid_count = invalid_mask.sum()

        if invalid_count > 0:
            invalid_found = True
            print(
                f"  {column}: {invalid_count:,} invalid values"
            )

    # BMI is handled separately because it has a continuous
    # validity rule rather than a finite set of codes.
    if "BMI" in df.columns:

        invalid_bmi = (
            (df["BMI"] < 12) |
            (df["BMI"] >= 100)
        ) & df["BMI"].notna()

        invalid_bmi_count = invalid_bmi.sum()

        if invalid_bmi_count > 0:
            invalid_found = True
            print(
                f"  BMI: {invalid_bmi_count:,} values "
                f"outside CDC-valid range (<12 or >=100)"
            )

    if not invalid_found:
        print("  No invalid coded values detected.")


# ============================================================
# 2. CLEAN INVALID VALUES
# ============================================================

def clean_invalid_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies BRFSS-compatible validity rules.

    Invalid coded values are converted to NaN rather than
    silently deleting observations.
    """

    df_clean = df.copy()

    valid_ranges = {
        "Diabetes_binary": {0, 1},
        "Diabetes_012": {0, 1, 2},

        "HighBP": {0, 1},
        "HighChol": {0, 1},
        "CholCheck": {0, 1},
        "Smoker": {0, 1},
        "Stroke": {0, 1},
        "HeartDiseaseorAttack": {0, 1},
        "PhysActivity": {0, 1},
        "Fruits": {0, 1},
        "Veggies": {0, 1},
        "HvyAlcoholConsump": {0, 1},
        "AnyHealthcare": {0, 1},
        "NoDocbcCost": {0, 1},
        "DiffWalk": {0, 1},
        "Sex": {0, 1},

        "GenHlth": set(range(1, 6)),
        "Age": set(range(1, 14)),
        "Education": set(range(1, 7)),
        "Income": set(range(1, 9)),

        "MentHlth": set(range(0, 31)),
        "PhysHlth": set(range(0, 31)),
    }

    invalid_summary = {}

    for column, valid_values in valid_ranges.items():

        if column not in df_clean.columns:
            continue

        invalid_mask = (
            ~df_clean[column].isin(valid_values)
            & df_clean[column].notna()
        )

        count = invalid_mask.sum()

        if count > 0:
            invalid_summary[column] = count
            df_clean.loc[invalid_mask, column] = np.nan

    # --------------------------------------------------------
    # BMI
    # CDC BRFSS calculated BMI:
    # values <12 or >=100 are treated as missing
    # --------------------------------------------------------

    if "BMI" in df_clean.columns:

        invalid_bmi = (
            (df_clean["BMI"] < 12) |
            (df_clean["BMI"] >= 100)
        ) & df_clean["BMI"].notna()

        bmi_invalid_count = invalid_bmi.sum()

        if bmi_invalid_count > 0:
            invalid_summary["BMI"] = bmi_invalid_count
            df_clean.loc[invalid_bmi, "BMI"] = np.nan

    if invalid_summary:

        print("\nInvalid values converted to missing:")

        for column, count in invalid_summary.items():
            print(f"  {column}: {count:,}")

    else:
        print("\nNo invalid values required correction.")

    return df_clean


# ============================================================
# 3. MEMORY OPTIMIZATION
# ============================================================

def downcast_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduces memory usage while preserving appropriate types.
    """

    initial_mem = (
        df.memory_usage(deep=True).sum()
        / (1024 ** 2)
    )

    # Binary variables
    binary_cols = [
        "Diabetes_binary",
        "HighBP",
        "HighChol",
        "CholCheck",
        "Smoker",
        "Stroke",
        "HeartDiseaseorAttack",
        "PhysActivity",
        "Fruits",
        "Veggies",
        "HvyAlcoholConsump",
        "AnyHealthcare",
        "NoDocbcCost",
        "DiffWalk",
        "Sex",
    ]

    for col in binary_cols:

        if col in df.columns:
            df[col] = df[col].astype("Int8")

    # Ordinal variables
    ordinal_cols = [
        "Diabetes_012",
        "GenHlth",
        "Age",
        "Education",
        "Income",
        "MentHlth",
        "PhysHlth",
    ]

    for col in ordinal_cols:

        if col in df.columns:
            df[col] = df[col].astype("Int8")

    # BMI needs to support missing values
    if "BMI" in df.columns:
        df["BMI"] = df["BMI"].astype("float32")

    final_mem = (
        df.memory_usage(deep=True).sum()
        / (1024 ** 2)
    )

    saving = (
        1 - final_mem / initial_mem
    ) * 100

    print(
        f"\nMemory footprint reduced from "
        f"{initial_mem:.2f} MB to "
        f"{final_mem:.2f} MB "
        f"({saving:.1f}% saving)"
    )

    return df


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates project-defined composite variables.

    Important:
    These are analytical features created for this project,
    not official clinical scores.
    """

    df_feat = df.copy()

    # --------------------------------------------------------
    # 1. Obesity indicator
    # CDC BMI threshold: >= 30
    # --------------------------------------------------------

    df_feat["Is_Obese"] = pd.Series(
        np.where(
            df_feat["BMI"].notna(),
            (df_feat["BMI"] >= 30).astype(float),
            np.nan
        ),
        index=df_feat.index
    ).astype("Int8")

    # --------------------------------------------------------
    # 2. Metabolic factor count
    #
    # Counts presence of:
    # - High blood pressure
    # - High cholesterol
    # - Obesity
    #
    # This is a project-defined count, NOT a clinical score.
    # --------------------------------------------------------

    df_feat["Metabolic_Factor_Count"] = (
        df_feat["HighBP"].astype("float") +
        df_feat["HighChol"].astype("float") +
        df_feat["Is_Obese"].astype("float")
    ).round().astype("Int8")

    # --------------------------------------------------------
    # 3. Non-smoker indicator
    # --------------------------------------------------------

    df_feat["Non_Smoker"] = (
        1 - df_feat["Smoker"]
    ).astype("Int8")

    # --------------------------------------------------------
    # 4. Project-defined healthy habits index
    #
    # Components:
    # - Physical activity
    # - Fruit consumption
    # - Vegetable consumption
    # - Non-smoking
    #
    # Equal weighting is intentional and should be described
    # as a project-defined exploratory index.
    # --------------------------------------------------------

    df_feat["Healthy_Habits"] = (
        df_feat["PhysActivity"].astype("float") +
        df_feat["Fruits"].astype("float") +
        df_feat["Veggies"].astype("float") +
        df_feat["Non_Smoker"].astype("float")
    ).round().astype("Int8")

    # --------------------------------------------------------
    # 5. Combined health burden
    #
    # IMPORTANT:
    # MentHlth and PhysHlth can overlap, so this is NOT the
    # number of unique unhealthy days.
    # It is simply a project-defined sum.
    # --------------------------------------------------------

    df_feat["Combined_Health_Burden"] = (
        df_feat["MentHlth"].astype("float") +
        df_feat["PhysHlth"].astype("float")
    ).astype("Int16")

    return df_feat


# ============================================================
# 5. PROCESS ONE DATASET
# ============================================================

def process_file(filename: str, output_name: str):

    raw_path = os.path.join(
        "data",
        "raw",
        filename
    )

    processed_dir = os.path.join(
        "data",
        "processed"
    )

    os.makedirs(
        processed_dir,
        exist_ok=True
    )

    output_path = os.path.join(
        processed_dir,
        output_name
    )

    print(f"\n\nProcessing: {filename}")

    # Load
    df = pd.read_csv(raw_path)

    # Step 1: Audit raw data
    audit_dataset(
        df,
        filename
    )

    # Step 2: Clean invalid values
    df = clean_invalid_values(df)

    # Step 3: Optimize memory
    df = downcast_types(df)

    # Step 4: Engineer project features
    df = engineer_features(df)

    # Save
    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\n--> Saved curated file: "
        f"{output_path}"
    )

    print(
        f"Final dimensions: "
        f"{df.shape[0]:,} rows × "
        f"{df.shape[1]} columns"
    )


# ============================================================
# 6. PROCESS ALL DATASETS
# ============================================================

def run_all():

    file_manifest = [

        # PRIMARY ANALYTICAL DATASET
        # Preserves:
        # 0 = No diabetes
        # 1 = Prediabetes
        # 2 = Diabetes
        (
            "diabetes_012_health_indicators_BRFSS2015.csv",
            "diabetes_progression_cleaned.csv"
        ),

        # SECONDARY VALIDATION DATASET
        # 0 = No diabetes
        # 1 = Prediabetes or diabetes
        (
            "diabetes_binary_health_indicators_BRFSS2015.csv",
            "diabetes_full_cleaned.csv"
        ),

        # MACHINE-LEARNING BENCHMARK
        # Balanced 50/50 dataset
        (
            "diabetes_binary_5050split_health_indicators_BRFSS2015.csv",
            "diabetes_balanced_cleaned.csv"
        ),
    ]

    for raw_file, output_file in file_manifest:

        process_file(
            raw_file,
            output_file
        )

    print(
        "\n\nAll 3 datasets processed successfully."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    run_all()