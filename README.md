# Investigating Diabetes Health Patterns in 2015 BRFSS Data

## Project overview

This project uses the 2015 Behavioral Risk Factor Surveillance System (BRFSS) diabetes-health-indicators dataset to explore how diabetes status is associated with metabolic health, lifestyle, socioeconomic conditions, functional health, and healthcare access.

The central question is:

> **What patterns distinguish people with no diabetes, prediabetes, and diabetes, and how do metabolic, lifestyle, socioeconomic, and health-burden factors intersect?**

The analysis is descriptive and inferential rather than predictive. We use Python for data-quality auditing, preprocessing, feature engineering, exploratory analysis, statistical testing, effect-size measurement, confidence intervals, and visualization.

**Scope:** The data represent a U.S. 2015 survey population. The analysis is cross-sectional and therefore identifies associations and prevalence patterns rather than causes or longitudinal disease progression.

---

## Data source

The project uses the **Diabetes Health Indicators Dataset**, derived from the **CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015** and distributed through Kaggle by Alex Teboul.

The source collection contains three related files:

- `diabetes_012_health_indicators_BRFSS2015.csv` — three-class target:
  - `0` = no diabetes
  - `1` = prediabetes
  - `2` = diabetes
- `diabetes_binary_health_indicators_BRFSS2015.csv` — binary target:
  - `0` = no diabetes
  - `1` = prediabetes or diabetes
- `diabetes_binary_5050split_health_indicators_BRFSS2015.csv` — balanced 50/50 binary benchmark:
  - `0` = no diabetes
  - `1` = prediabetes or diabetes

The three-class dataset is treated as the **primary analytical dataset** because it preserves the distinction between no diabetes, prediabetes, and diabetes. The full binary dataset is used for secondary analyses and the balanced dataset is used as a sensitivity/benchmark dataset.

---

## Data preparation

### Quality audit

The preprocessing pipeline checks:

- Dataset dimensions
- Missing values
- Exact duplicate response profiles
- Data types
- Valid ranges for coded BRFSS variables
- BMI validity
- Target-value validity

The current full binary processed dataset contains **253,680 rows and 27 columns**, with **0 missing cells**, **0 invalid target values**, and **0 invalid BMI values** in the final audit.

### Duplicate handling

The audit identified **24,206 exact duplicate response profiles** in the full binary dataset.

These rows are **not automatically removed** from the primary analysis because the dataset does not provide a respondent identifier. An identical set of survey responses does not prove that two rows represent the same respondent.

Instead, duplicate removal is evaluated as a **sensitivity analysis**. The main conclusions remain broadly consistent when exact duplicates are excluded.

### Invalid values

Invalid categorical codes are converted to `NaN` rather than silently deleting observations. BMI follows the BRFSS validity rule used by the project: values below 12 or at/above 100 are treated as invalid/missing.

### Feature engineering

The project creates several analysis features:

**`Is_Obese`**
- `1` when BMI >= 30
- `0` otherwise
- missing when BMI is missing

**`Metabolic_Factor_Count`**
- Sum of:
  - High blood pressure
  - High cholesterol
  - Obesity
- Range: 0–3
- This is a **project-defined exploratory index**, not a clinical score.

**`Non_Smoker`**
- Inverse of the BRFSS smoking indicator.

**`Healthy_Habits`**
- Sum of:
  - Physical activity
  - Fruit consumption
  - Vegetable consumption
  - Non-smoking
- Range: 0–4
- This is a **project-defined exploratory index**, not a validated clinical score.

**`Combined_Health_Burden`**
- `MentHlth + PhysHlth`
- This is an exploratory summary only. The two measures can overlap, so the sum is **not** interpreted as a count of unique unhealthy days.

---

## Statistical approach

The project focuses on interpretable statistical analysis rather than machine learning or regression.

### Categorical associations

Chi-square tests are used for categorical comparisons, with **Cramér's V** reported as an effect-size measure.

Because the sample is very large, statistical significance is not interpreted on its own.

### Continuous/ordinal health measures

For `PhysHlth` and `MentHlth`, group distributions are compared using the **Mann–Whitney U test**. Rank-biserial correlation is used as an effect-size measure.

### Prevalence confidence intervals

Wilson 95% confidence intervals are used for prevalence estimates in the metabolic-factor analysis.

### Sensitivity analysis

Main categorical effect sizes are compared:

1. Using all processed rows
2. After removing exact duplicate response profiles

This checks whether the headline conclusions depend strongly on the duplicate-handling decision.

---

## Main findings

### 1. A clear health gradient appears across diabetes status

Across the three-class dataset, several health characteristics increase from no diabetes to prediabetes to diabetes:

- High blood pressure: **37.1% → 62.9% → 75.3%**
- High cholesterol: **37.9% → 62.1% → 67.0%**
- Obesity: **30.4% → 51.6% → 58.2%**
- Difficulty walking: **13.2% → 27.7% → 37.1%**

These are prevalence differences across cross-sectional groups, not evidence of causal progression.

### 2. Metabolic burden shows the strongest categorical association

In the full binary dataset, prediabetes/diabetes prevalence rises sharply with the project-defined metabolic factor count:

| Metabolic factors | Prevalence |
|---:|---:|
| 0 | 2.88% |
| 1 | 9.10% |
| 2 | 20.94% |
| 3 | 39.31% |

Cramér's V is approximately **0.334**, making this the strongest categorical association among the headline variables tested.

The odds ratio comparing the 3-factor group with the 0-factor group is approximately **21.9 (95% CI: 20.84–22.96)**. This is an odds comparison within the observed cross-sectional data, not a causal risk estimate.

### 3. General health is strongly associated with prediabetes/diabetes status

General health has a Cramér's V of approximately **0.299**.

Difficulty walking also rises sharply as self-reported general health shifts from excellent to poor, and it remains higher among participants with prediabetes/diabetes at every general-health level in the plotted analysis.

### 4. Socioeconomic differences remain visible within obesity groups

Prediabetes/diabetes prevalence decreases across the income categories in both non-obese and obese participants.

In the full binary dataset:

- Non-obese: **16.3% → 5.0%**
- Obese: **35.1% → 15.2%**

This is interpreted as a socioeconomic prevalence gradient, not a causal effect of income.

### 5. Physical health burden differentiates the binary groups more strongly than mental health burden

Mean self-reported physical-health days are approximately:

- No prediabetes/diabetes: **3.6**
- Prediabetes/diabetes: **8.0**

For mental-health days:

- No prediabetes/diabetes: **3.0**
- Prediabetes/diabetes: **4.5**

Both Mann–Whitney tests are statistically significant at **p < 0.001**, but effect sizes differ:

- Physical health rank-biserial correlation: **0.227**
- Mental health rank-biserial correlation: **0.055**

This makes physical-health burden the more meaningful differentiator.

### 6. Healthy-habit count shows a consistent but weaker association

In the full sample, prediabetes/diabetes prevalence falls from **22.4%** among participants reporting zero project-defined healthy habits to **8.9%** among those reporting four.

The direction is also visible in the balanced 50/50 dataset, but absolute prevalence values in the balanced data should not be interpreted as population prevalence because the outcome distribution was deliberately altered.

---

## Visualization set

The final story uses eight primary/supporting visualizations:

1. Health characteristics across no diabetes → prediabetes → diabetes
2. Metabolic factor count with prevalence and 95% CI
3. Income × obesity prevalence gradient
4. Physical and mental health burden
5. Healthy-habit gradient
6. Age × metabolic burden heatmap
7. General health × difficulty walking
8. Healthcare coverage × cost-related care barrier

The visual story moves from broad health differences to compounding metabolic burden, then to demographic and socioeconomic context, lived health burden, lifestyle patterns, and healthcare-access context.

---

## Limitations

This project has several important limitations:

- The data are from the **United States in 2015** and should not be generalized to current populations without additional evidence.
- BRFSS is a **cross-sectional survey**, so the analysis does not establish causation or disease progression over time.
- Several variables are self-reported.
- The project-defined `Metabolic_Factor_Count` and `Healthy_Habits` measures are exploratory indices, not validated clinical instruments.
- The absence of a respondent identifier prevents definitive identification of erroneous duplicate respondents.
- The analysis does not account for the complex BRFSS survey weighting/design in its descriptive estimates or tests.
- The binary target combines prediabetes and diabetes, so binary analyses must be interpreted as **prediabetes/diabetes** rather than diabetes alone.
- The balanced 50/50 dataset is useful for sensitivity/benchmark analysis but should not be treated as a direct estimate of population prevalence.

---

## Reproducibility

From the project root:

```powershell
python -m pip install -r requirements.txt
python src/preprocessing.py
python src/statistics.py
python src/final_story_visualizations.py
```

The main analysis notebook is:

```text
notebooks/02_exploratory_and_storytelling.ipynb
```

---

## Repository structure

```text
cdc-diabetes-lifestyle-study/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_data_cleaning_and_audit.ipynb
│   └── 02_exploratory_and_storytelling.ipynb
├── src/
│   ├── preprocessing.py
│   ├── statistics.py
│   └── final_story_visualizations.py
└── figures/
    ├── fig1_health_characteristics.png
    ├── fig2_metabolic_factor_prevalence.png
    ├── fig3_income_obesity_gradient.png
    ├── fig4_health_burden.png
    ├── fig5_healthy_habits.png
    ├── fig6_age_metabolic_burden_heatmap.png
    ├── fig7_general_health_mobility.png
    ├── fig8_healthcare_access_cost_heatmap.png
    └── statistical_outputs/
```

---

## Dataset and methodology references

- Alex Teboul. *Diabetes Health Indicators Dataset*. Kaggle.
- Centers for Disease Control and Prevention (CDC). *Behavioral Risk Factor Surveillance System (BRFSS) Annual Data*.
- CDC BRFSS 2015 documentation and calculated-variable guidance, including BMI validity rules.
