import os
import pandas as pd
import numpy as np

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

    print(f"\nDimensions: {df.shape[0]:,} rows × {df.shape[1]} columns")

    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()

    print(f"\nTotal Missing Values: {total_nulls:,}")

    if total_nulls > 0:
        print("\nColumns with missing values:")
        print(null_counts[null_counts > 0])

    duplicate_count = df.duplicated().sum()

    print(
        f"\nIdentical Survey Profiles: "
        f"{duplicate_count:,} "
        f"({duplicate_count / len(df) * 100:.2f}%)"
    )

    print("\nData Types:")
    print(df.dtypes.value_counts())

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

    # BMI
    # CDC BRFSS calculated BMI:
    # values <12 or >=100 are treated as missing

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

def downcast_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduces memory usage while preserving appropriate types.
    """

    initial_mem = (
        df.memory_usage(deep=True).sum()
        / (1024 ** 2)
    )
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

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df_feat = df.copy()
    df_feat["Is_Obese"] = pd.Series(
        np.where(
            df_feat["BMI"].notna(),
            (df_feat["BMI"] >= 30).astype(float),
            np.nan
        ),
        index=df_feat.index
    ).astype("Int8")

    df_feat["Metabolic_Factor_Count"] = (
        df_feat["HighBP"].astype("float") +
        df_feat["HighChol"].astype("float") +
        df_feat["Is_Obese"].astype("float")
    ).round().astype("Int8")

    df_feat["Non_Smoker"] = (
        1 - df_feat["Smoker"]
    ).astype("Int8")

    df_feat["Healthy_Habits"] = (
        df_feat["PhysActivity"].astype("float") +
        df_feat["Fruits"].astype("float") +
        df_feat["Veggies"].astype("float") +
        df_feat["Non_Smoker"].astype("float")
    ).round().astype("Int8")

    df_feat["Combined_Health_Burden"] = (
        df_feat["MentHlth"].astype("float") +
        df_feat["PhysHlth"].astype("float")
    ).astype("Int16")

    return df_feat

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

    df = pd.read_csv(raw_path)
    audit_dataset(
        df,
        filename
    )
    df = clean_invalid_values(df)
    df = downcast_types(df)
    df = engineer_features(df)
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

def run_all():

    file_manifest = [
        (
            "diabetes_012_health_indicators_BRFSS2015.csv",
            "diabetes_progression_cleaned.csv"
        ),
        (
            "diabetes_binary_health_indicators_BRFSS2015.csv",
            "diabetes_full_cleaned.csv"
        ),
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

if __name__ == "__main__":
    run_all()