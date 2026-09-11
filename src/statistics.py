from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "figures" / "statistical_outputs"


def cramers_v(table, chi2):
    n = table.to_numpy().sum()
    rows, cols = table.shape
    return np.sqrt((chi2 / n) / min(rows - 1, cols - 1))


def chi_square_effect(df, feature, target="Diabetes_binary"):
    table = pd.crosstab(df[feature], df[target])
    chi2, p_value, dof, _ = stats.chi2_contingency(table)

    return {
        "Variable": feature,
        "Chi-square": chi2,
        "df": dof,
        "p-value": p_value,
        "Cramer's V": cramers_v(table, chi2),
    }


def wilson_ci(successes, total, z=1.96):
    p = successes / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    margin = (
        z
        * np.sqrt(
            (p * (1 - p) / total) + (z**2 / (4 * total**2))
        )
        / denominator
    )

    return centre - margin, centre + margin


def rank_biserial_correlation(x, y, u_stat):
    return 1 - (2 * u_stat) / (len(x) * len(y))


def run_statistics():
    data_path = PROCESSED_DIR / "diabetes_full_cleaned.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {data_path}. "
            "Run python src/preprocessing.py first."
        )

    df = pd.read_csv(data_path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    categorical_features = [
        "Metabolic_Factor_Count",
        "Income",
        "GenHlth",
        "Healthy_Habits",
        "HighBP",
        "HighChol",
        "Is_Obese",
        "DiffWalk",
    ]

    categorical_results = pd.DataFrame(
        [
            chi_square_effect(df, feature)
            for feature in categorical_features
        ]
    )

    categorical_results.to_csv(
        OUTPUT_DIR / "categorical_association_results.csv",
        index=False,
    )

    health_results = []

    for variable in ["PhysHlth", "MentHlth"]:
        group0 = df.loc[
            df["Diabetes_binary"] == 0, variable
        ].dropna()

        group1 = df.loc[
            df["Diabetes_binary"] == 1, variable
        ].dropna()

        u_stat, p_value = stats.mannwhitneyu(
            group0,
            group1,
            alternative="two-sided",
        )

        effect = rank_biserial_correlation(
            group0,
            group1,
            u_stat,
        )

        health_results.append(
            {
                "Variable": variable,
                "No Prediabetes/Diabetes Median": group0.median(),
                "Prediabetes/Diabetes Median": group1.median(),
                "U statistic": u_stat,
                "p-value": p_value,
                "Rank-biserial correlation": effect,
            }
        )

    pd.DataFrame(health_results).to_csv(
        OUTPUT_DIR / "health_burden_mann_whitney_results.csv",
        index=False,
    )

    metabolic = (
        df.groupby("Metabolic_Factor_Count")["Diabetes_binary"]
        .agg(["sum", "count"])
        .reset_index()
    )

    metabolic["Prevalence (%)"] = (
        metabolic["sum"] / metabolic["count"] * 100
    )

    cis = metabolic.apply(
        lambda row: wilson_ci(row["sum"], row["count"]),
        axis=1,
    )

    metabolic["CI Lower (%)"] = [ci[0] * 100 for ci in cis]
    metabolic["CI Upper (%)"] = [ci[1] * 100 for ci in cis]

    metabolic.to_csv(
        OUTPUT_DIR / "metabolic_factor_prevalence_ci.csv",
        index=False,
    )

    sensitivity_df = df.drop_duplicates().reset_index(drop=True)

    sensitivity_results = pd.DataFrame(
        [
            chi_square_effect(sensitivity_df, feature)
            for feature in categorical_features
        ]
    )

    comparison = (
        categorical_results[["Variable", "Cramer's V"]]
        .rename(columns={"Cramer's V": "Cramer's V (All Rows)"})
        .merge(
            sensitivity_results[["Variable", "Cramer's V"]].rename(
                columns={
                    "Cramer's V": "Cramer's V (Duplicates Removed)"
                }
            ),
            on="Variable",
        )
    )

    comparison["Absolute Difference"] = (
        comparison["Cramer's V (All Rows)"]
        - comparison["Cramer's V (Duplicates Removed)"]
    ).abs()

    comparison["Relative Change (%)"] = (
        comparison["Absolute Difference"]
        / comparison["Cramer's V (All Rows)"]
        * 100
    )

    comparison.to_csv(
        OUTPUT_DIR / "duplicate_sensitivity_comparison.csv",
        index=False,
    )

    print("Statistical outputs saved to:", OUTPUT_DIR)
    print()

    display_results = categorical_results.copy()

    display_results["p-value"] = display_results["p-value"].apply(
        lambda p: "<0.001" if p < 0.001 else f"{p:.4f}"
    )

    print(display_results.to_string(index=False))
    print()

    print(comparison.round(4).to_string(index=False))


if __name__ == "__main__":
    run_statistics()
