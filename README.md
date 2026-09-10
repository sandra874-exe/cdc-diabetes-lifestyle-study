# Diabetes Health Indicators EDA

A reproducible exploratory data analysis (EDA) script for the 2015 Behavioral Risk Factor Surveillance System (BRFSS) diabetes health-indicator datasets. The analysis compares three-class, balanced binary, and imbalanced binary versions of the data, then creates detailed visualizations from the balanced binary dataset.

## What the script does

`comprehensive_eda.py`:

- Loads the three BRFSS CSV files in this directory.
- Converts all columns to numeric values and reports malformed or missing values.
- Removes rows with missing values and duplicate rows.
- Prints dataset dimensions, cleaning counts, target distributions, and descriptive statistics.
- Compares target class counts and percentages across all datasets.
- Analyzes correlations, BMI, physical and mental health days, categorical indicators, age, general health, and selected cardiovascular risk factors.
- Writes summary tables as CSV files and charts as PNG files.

The detailed plots use `diabetes_binary_5050split_health_indicators_BRFSS2015.csv` because its classes are balanced. The script still summarizes all three datasets so their class distributions can be compared.

## Requirements

- Python 3.10 or newer
- pandas
- matplotlib
- seaborn

Install the Python dependencies with:

```bash
python -m pip install pandas matplotlib seaborn
```

## Input files

Place these files in the same directory as the script, or provide another directory with `--data-dir`:

- `diabetes_012_health_indicators_BRFSS2015.csv` - no diabetes, prediabetes, and diabetes classes
- `diabetes_binary_5050split_health_indicators_BRFSS2015.csv` - balanced binary target
- `diabetes_binary_health_indicators_BRFSS2015.csv` - imbalanced binary target

The files must contain the target columns expected by the script:

- `Diabetes_012` for the three-class dataset
- `Diabetes_binary` for both binary datasets

## Run the analysis

From this directory, run:

```bash
python comprehensive_eda.py
```

Optional arguments:

```text
--data-dir PATH      Folder containing the three input CSV files
--output-dir PATH    Folder for generated CSV summaries and PNG charts
--show               Request display of plots while running
```

For example:

```bash
python comprehensive_eda.py \
  --data-dir ./data \
  --output-dir ./eda_output
```

The output directory is created automatically when it does not exist.

## Generated outputs

### CSV summaries

- `data_quality_summary.csv`
- `target_distributions.csv`
- `three_class_descriptive_statistics.csv`
- `balanced_binary_descriptive_statistics.csv`
- `imbalanced_binary_descriptive_statistics.csv`
- `balanced_target_correlations.csv`
- `top_positive_negative_correlations.csv`
- `feature_diabetes_prevalence.csv`
- `diabetes_prevalence_by_age.csv`
- `diabetes_prevalence_by_general_health.csv`
- `diabetes_prevalence_by_risk_burden.csv`
- `numeric_distributions_long_format.csv`

### PNG charts

- `01_target_distributions.png`
- `02_balanced_correlation_heatmap.png`
- `03_bivariate_distributions.png`
- `04_categorical_comparisons.png`
- `05_feature_prevalence.png`
- `06_age_and_health_trends.png`
- `07_top_correlations.png`
- `08_numeric_distributions.png`
- `09_risk_burden_analysis.png`

## Notes on interpretation

- The binary target is treated as `0 = No diabetes` and `1 = Diabetes`.
- The three-class target is treated as `0 = No diabetes`, `1 = Prediabetes`, and `2 = Diabetes`.
- BRFSS fields such as `Age` and `GenHlth` are ordinal categories, not continuous measurements. Their labels and values should be interpreted according to the dataset documentation.
- Correlation describes linear association and does not establish causation.
- The analysis is descriptive and is not a clinical diagnostic tool.

## License and data attribution

This repository contains analysis code and local copies of the BRFSS-derived CSV files. Check the original dataset source and its terms before redistributing the data or using the results for clinical or production purposes.
