"""Comprehensive exploratory data analysis for the BRFSS diabetes datasets.

Run from this folder with:
    py comprehensive_eda.py

The script writes cleaned-data summaries and publication-ready PNG figures to
the ``eda_output`` directory. Use ``--show`` to display figures interactively.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DATASETS: dict[str, dict[str, Any]] = {
    "three_class": {
        "filename": "diabetes_012_health_indicators_BRFSS2015.csv",
        "target": "Diabetes_012",
        "labels": {0.0: "No diabetes", 1.0: "Prediabetes", 2.0: "Diabetes"},
    },
    "balanced_binary": {
        "filename": "diabetes_binary_5050split_health_indicators_BRFSS2015.csv",
        "target": "Diabetes_binary",
        "labels": {0.0: "No diabetes", 1.0: "Diabetes"},
    },
    "imbalanced_binary": {
        "filename": "diabetes_binary_health_indicators_BRFSS2015.csv",
        "target": "Diabetes_binary",
        "labels": {0.0: "No diabetes", 1.0: "Diabetes"},
    },
}

TARGET_LABELS = {
    "No diabetes": "No diabetes",
    "Prediabetes": "Prediabetes",
    "Diabetes": "Diabetes",
}
BALANCED_TARGET_LABELS = {0.0: "No diabetes", 1.0: "Diabetes"}


def load_and_clean_dataset(path: Path, target: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load one CSV, coerce numeric fields, report issues, and remove duplicates."""
    frame = pd.read_csv(path)
    if target not in frame.columns:
        raise ValueError(f"Expected target column {target!r} in {path.name}")

    # All BRFSS columns are numeric indicator or ordinal fields.
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    missing_before = int(frame.isna().sum().sum())
    duplicates_before = int(frame.duplicated().sum())
    if missing_before:
        frame = frame.dropna().reset_index(drop=True)
    if duplicates_before:
        frame = frame.drop_duplicates().reset_index(drop=True)

    quality = {
        "missing_values_removed": missing_before,
        "duplicate_rows_removed": duplicates_before,
    }
    return frame, quality


def print_dataset_summary(
    name: str, frame: pd.DataFrame, target: str, quality: dict[str, int]
) -> None:
    """Print shape, quality checks, target counts, and descriptive statistics."""
    print(f"\n{'=' * 80}\n{name}\n{'=' * 80}")
    print(f"Shape after cleaning: {frame.shape[0]:,} rows x {frame.shape[1]} columns")
    print(f"Missing values removed: {quality['missing_values_removed']:,}")
    print(f"Duplicate rows removed: {quality['duplicate_rows_removed']:,}")
    print("\nTarget distribution:")
    print(frame[target].value_counts(dropna=False).sort_index().to_string())
    print("\nDescriptive statistics:")
    print(frame.describe().T.to_string())


def save_quality_summary(
    summaries: dict[str, tuple[pd.DataFrame, dict[str, int]]], output_dir: Path
) -> None:
    """Save one compact quality and shape summary for later reporting."""
    rows = []
    for name, (frame, quality) in summaries.items():
        rows.append(
            {
                "dataset": name,
                "rows_after_cleaning": len(frame),
                "columns": len(frame.columns),
                **quality,
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "data_quality_summary.csv", index=False)


def plot_target_distributions(
    summaries: dict[str, tuple[pd.DataFrame, dict[str, int]]], output_dir: Path
) -> None:
    """Compare target counts and percentages across all three datasets."""
    distribution_rows = []
    for name, (frame, _) in summaries.items():
        config = DATASETS[name]
        counts = frame[config["target"]].map(config["labels"]).value_counts()
        for label, count in counts.items():
            distribution_rows.append(
                {
                    "Dataset": name.replace("_", " ").title(),
                    "Class": label,
                    "Count": count,
                    "Percentage": count / len(frame) * 100,
                }
            )
    distribution = pd.DataFrame(distribution_rows)
    distribution.to_csv(output_dir / "target_distributions.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.barplot(data=distribution, x="Dataset", y="Count", hue="Class", ax=axes[0], palette="Set2")
    axes[0].set_title("Target Class Counts")
    axes[0].set_xlabel("Dataset")
    axes[0].set_ylabel("Number of records")
    axes[0].tick_params(axis="x", rotation=20)

    sns.barplot(
        data=distribution,
        x="Dataset",
        y="Percentage",
        hue="Class",
        ax=axes[1],
        palette="Set2",
    )
    axes[1].set_title("Target Class Percentages")
    axes[1].set_xlabel("Dataset")
    axes[1].set_ylabel("Percentage of records")
    axes[1].tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure.savefig(output_dir / "01_target_distributions.png", dpi=200)
    plt.close(figure)


def plot_correlation_heatmap(
    frame: pd.DataFrame, target: str, output_dir: Path
) -> None:
    """Plot and save the full numeric correlation matrix for the balanced set."""
    correlations = frame.corr(numeric_only=True)
    correlations[target].drop(target).abs().sort_values(ascending=False).to_csv(
        output_dir / "balanced_target_correlations.csv", header=["absolute_correlation"]
    )

    figure, axis = plt.subplots(figsize=(15, 12))
    sns.heatmap(
        correlations,
        cmap="viridis",
        center=0,
        square=True,
        linewidths=0.25,
        cbar_kws={"label": "Pearson correlation"},
        ax=axis,
    )
    axis.set_title("Correlation Heatmap: Balanced Binary Dataset")
    figure.tight_layout()
    figure.savefig(output_dir / "02_balanced_correlation_heatmap.png", dpi=200)
    plt.close(figure)


def plot_bivariate_distributions(
    frame: pd.DataFrame, target: str, output_dir: Path
) -> None:
    """Compare BMI and reported physical-health days by diabetes status."""
    plot_frame = frame[[target, "BMI", "PhysHlth"]].copy()
    plot_frame["Diabetes status"] = plot_frame[target].map(BALANCED_TARGET_LABELS)
    plot_frame = plot_frame.drop(columns=target).melt(
        id_vars="Diabetes status", var_name="Measure", value_name="Value"
    )
    # Large datasets remain responsive while preserving the distributions.
    if len(plot_frame) > 100_000:
        plot_frame = plot_frame.sample(100_000, random_state=42)

    figure, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.boxplot(data=plot_frame, x="Diabetes status", y="Value", hue="Diabetes status", ax=axes[0], palette="Set2", legend=False)
    axes[0].set_title("BMI and Physical Health by Diabetes Status")
    axes[0].set_xlabel("Diabetes status")
    axes[0].set_ylabel("Value")
    sns.violinplot(data=plot_frame, x="Measure", y="Value", hue="Diabetes status", split=False, inner="quartile", ax=axes[1], palette="Set2")
    axes[1].set_title("Indicator Distributions by Diabetes Status")
    axes[1].set_xlabel("Measure")
    axes[1].set_ylabel("Value")
    figure.tight_layout()
    figure.savefig(output_dir / "03_bivariate_distributions.png", dpi=200)
    plt.close(figure)


def plot_categorical_comparisons(
    frame: pd.DataFrame, target: str, output_dir: Path
) -> None:
    """Plot within-target percentages for key categorical health indicators."""
    variables = ["Smoker", "HighChol", "GenHlth"]
    figure, axes = plt.subplots(1, len(variables), figsize=(18, 5))
    target_labels = frame[target].map(BALANCED_TARGET_LABELS)

    for axis, variable in zip(axes, variables):
        proportions = pd.crosstab(
            frame[variable], target_labels, normalize="index"
        ).mul(100)
        proportions.plot(kind="bar", stacked=True, colormap="Set2", ax=axis)
        axis.set_title(f"{variable} by Diabetes Status")
        axis.set_xlabel(variable)
        axis.set_ylabel("Percentage within category")
        axis.legend(title="Diabetes status", fontsize=8)
        axis.tick_params(axis="x", rotation=0)

    figure.tight_layout()
    figure.savefig(output_dir / "04_categorical_comparisons.png", dpi=200)
    plt.close(figure)


def plot_feature_prevalence(
    frame: pd.DataFrame, target: str, output_dir: Path
) -> None:
    """Compare diabetes prevalence for the binary health indicators."""
    variables = [
        "HighBP",
        "HighChol",
        "Smoker",
        "Stroke",
        "HeartDiseaseorAttack",
        "PhysActivity",
        "Fruits",
        "Veggies",
        "DiffWalk",
    ]
    rows = []
    for variable in variables:
        grouped = frame.groupby(variable, observed=True)[target].agg(
            diabetes_rate="mean", records="size"
        )
        for category, values in grouped.iterrows():
            rows.append(
                {
                    "Feature": variable,
                    "Category": "Yes" if category == 1 else "No",
                    "Diabetes prevalence (%)": values["diabetes_rate"] * 100,
                    "Records": values["records"],
                }
            )
    prevalence = pd.DataFrame(rows)
    prevalence.to_csv(output_dir / "feature_diabetes_prevalence.csv", index=False)

    figure, axis = plt.subplots(figsize=(13, 7))
    sns.barplot(
        data=prevalence[prevalence["Category"] == "Yes"],
        x="Diabetes prevalence (%)",
        y="Feature",
        hue="Feature",
        palette="viridis",
        legend=False,
        ax=axis,
    )
    axis.set_title("Diabetes Prevalence Among Participants With Each Condition")
    axis.set_xlabel("Diabetes prevalence (%)")
    axis.set_ylabel("Health indicator")
    figure.tight_layout()
    figure.savefig(output_dir / "05_feature_prevalence.png", dpi=200)
    plt.close(figure)


def plot_age_and_health_trends(
    frame: pd.DataFrame, target: str, output_dir: Path
) -> None:
    """Show diabetes prevalence across age groups and self-rated general health."""
    age_rates = frame.groupby("Age", observed=True)[target].agg(
        diabetes_prevalence="mean", records="size"
    ).reset_index()
    health_rates = frame.groupby("GenHlth", observed=True)[target].agg(
        diabetes_prevalence="mean", records="size"
    ).reset_index()
    age_rates["diabetes_prevalence"] *= 100
    health_rates["diabetes_prevalence"] *= 100
    age_rates.to_csv(output_dir / "diabetes_prevalence_by_age.csv", index=False)
    health_rates.to_csv(output_dir / "diabetes_prevalence_by_general_health.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.lineplot(data=age_rates, x="Age", y="diabetes_prevalence", marker="o", color="#2a9d8f", ax=axes[0])
    axes[0].set_title("Diabetes Prevalence by Age Group")
    axes[0].set_xlabel("BRFSS age category (1 = youngest)")
    axes[0].set_ylabel("Diabetes prevalence (%)")
    axes[0].set_xticks(sorted(age_rates["Age"].unique()))

    sns.barplot(data=health_rates, x="GenHlth", y="diabetes_prevalence", hue="GenHlth", palette="viridis", legend=False, ax=axes[1])
    axes[1].set_title("Diabetes Prevalence by Self-Rated General Health")
    axes[1].set_xlabel("General health (1 = excellent, 5 = poor)")
    axes[1].set_ylabel("Diabetes prevalence (%)")
    figure.tight_layout()
    figure.savefig(output_dir / "06_age_and_health_trends.png", dpi=200)
    plt.close(figure)


def plot_top_correlations(
    frame: pd.DataFrame, target: str, output_dir: Path
) -> None:
    """Plot the strongest positive and negative Pearson correlations with diabetes."""
    correlations = frame.corr(numeric_only=True)[target].drop(target).sort_values()
    top_correlations = pd.concat([correlations.head(7), correlations.tail(7)]).drop_duplicates()
    top_correlations = top_correlations.sort_values()
    top_correlations.to_csv(output_dir / "top_positive_negative_correlations.csv", header=["correlation"])

    figure, axis = plt.subplots(figsize=(11, 7))
    colors = ["#4575b4" if value < 0 else "#d73027" for value in top_correlations]
    axis.barh(top_correlations.index, top_correlations.values, color=colors)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_title("Strongest Positive and Negative Correlations With Diabetes")
    axis.set_xlabel("Pearson correlation")
    axis.set_ylabel("Health indicator")
    figure.tight_layout()
    figure.savefig(output_dir / "07_top_correlations.png", dpi=200)
    plt.close(figure)


def plot_numeric_distributions(
    frame: pd.DataFrame, target: str, output_dir: Path
) -> None:
    """Compare BMI, mental-health days, and physical-health days by target class."""
    plot_frame = frame[[target, "BMI", "MentHlth", "PhysHlth"]].copy()
    plot_frame["Diabetes status"] = plot_frame[target].map(BALANCED_TARGET_LABELS)
    plot_frame = plot_frame.drop(columns=target).melt(
        id_vars="Diabetes status", var_name="Measure", value_name="Value"
    )
    plot_frame.to_csv(output_dir / "numeric_distributions_long_format.csv", index=False)

    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    for axis, measure in zip(axes, ["BMI", "MentHlth", "PhysHlth"]):
        sns.histplot(
            data=plot_frame[plot_frame["Measure"] == measure],
            x="Value",
            hue="Diabetes status",
            bins=25,
            stat="density",
            common_norm=False,
            element="step",
            palette="Set2",
            ax=axis,
        )
        axis.set_title(f"{measure} Distribution")
        axis.set_xlabel(measure)
        axis.set_ylabel("Density")
    figure.suptitle("Numeric Health Distributions by Diabetes Status", y=1.03)
    figure.tight_layout()
    figure.savefig(output_dir / "08_numeric_distributions.png", dpi=200)
    plt.close(figure)


def plot_health_burden(
    frame: pd.DataFrame, target: str, output_dir: Path
) -> None:
    """Create an interpretable count of selected cardiovascular risk indicators."""
    risk_variables = ["HighBP", "HighChol", "Stroke", "HeartDiseaseorAttack", "DiffWalk"]
    burden_frame = frame[[target, *risk_variables]].copy()
    burden_frame["Risk indicator count"] = burden_frame[risk_variables].sum(axis=1)
    burden_summary = burden_frame.groupby("Risk indicator count", observed=True)[target].agg(
        diabetes_prevalence="mean", records="size"
    ).reset_index()
    burden_summary["diabetes_prevalence"] *= 100
    burden_summary.to_csv(output_dir / "diabetes_prevalence_by_risk_burden.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.barplot(data=burden_summary, x="Risk indicator count", y="diabetes_prevalence", color="#e76f51", ax=axes[0])
    axes[0].set_title("Diabetes Prevalence by Risk-Indicator Burden")
    axes[0].set_xlabel("Number of selected risk indicators")
    axes[0].set_ylabel("Diabetes prevalence (%)")
    sns.boxplot(data=burden_frame, x=target, y="Risk indicator count", hue=target, palette="Set2", legend=False, ax=axes[1])
    axes[1].set_title("Risk-Indicator Burden by Diabetes Status")
    axes[1].set_xlabel("Diabetes status (0 = no, 1 = yes)")
    axes[1].set_ylabel("Number of risk indicators")
    figure.tight_layout()
    figure.savefig(output_dir / "09_risk_burden_analysis.png", dpi=200)
    plt.close(figure)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Folder containing the three BRFSS CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "eda_output",
        help="Folder where figures and summary CSV files will be written.",
    )
    parser.add_argument(
        "--show", action="store_true", help="Display plots instead of closing them immediately."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="Set2")
    plt.rcParams.update({"figure.dpi": 100, "savefig.bbox": "tight"})

    summaries: dict[str, tuple[pd.DataFrame, dict[str, int]]] = {}
    for name, config in DATASETS.items():
        frame, quality = load_and_clean_dataset(
            args.data_dir / config["filename"], config["target"]
        )
        summaries[name] = (frame, quality)
        print_dataset_summary(name, frame, config["target"], quality)
        frame.describe().T.to_csv(args.output_dir / f"{name}_descriptive_statistics.csv")

    save_quality_summary(summaries, args.output_dir)
    plot_target_distributions(summaries, args.output_dir)

    balanced_frame, _ = summaries["balanced_binary"]
    balanced_target = DATASETS["balanced_binary"]["target"]
    plot_correlation_heatmap(balanced_frame, balanced_target, args.output_dir)
    plot_bivariate_distributions(balanced_frame, balanced_target, args.output_dir)
    plot_categorical_comparisons(balanced_frame, balanced_target, args.output_dir)
    plot_feature_prevalence(balanced_frame, balanced_target, args.output_dir)
    plot_age_and_health_trends(balanced_frame, balanced_target, args.output_dir)
    plot_top_correlations(balanced_frame, balanced_target, args.output_dir)
    plot_numeric_distributions(balanced_frame, balanced_target, args.output_dir)
    plot_health_burden(balanced_frame, balanced_target, args.output_dir)

    print(f"\nEDA complete. Outputs saved to: {args.output_dir.resolve()}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    # Avoid GUI backend errors when the script is run on a headless machine.
    if not os.environ.get("DISPLAY") and os.name != "nt":
        import matplotlib

        matplotlib.use("Agg")
    main()