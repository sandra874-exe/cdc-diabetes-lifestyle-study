
# Investigating the Metabolic Tipping Point: A CDC BRFSS Empirical Study

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Data Source: CDC BRFSS](<https://img.shields.io/badge/Source-CDC%20BRFSS%202015-green.svg>)](https://www.cdc.gov/brfss/annual_data/annual_data.htm)

## Executive Summary

This project investigates the multi-factorial determinants of diabetes prevalence across 253,680 survey participants from the CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015. Rather than evaluating diabetes as an isolated metabolic outcome, this study evaluates the interaction between biometric clinical markers (hypertension, hypercholesterolemia, obesity) and socioeconomic defenses (income, healthcare coverage).

---

## Key Empirical Findings

1. **The Pre-Diabetes Chasm:** Biometric deterioration does not wait for a clinical diagnosis. Hypertension prevalence surges by **+25.7%** and hypercholesterolemia surges by **+24.2%** during the initial shift from healthy to pre-diabetic status.
2. **The Clinical Triad Multiplier:** Co-occurrence of High BP, High Cholesterol, and Obesity ($BMI \ge 30$) increases diabetes risk from **2.9% to 39.3%** ($\text{Odds Ratio} = 21.91, \ p < 0.0001$).
3. **The Socioeconomic Gradient:** High-income individuals with clinical obesity exhibit lower diabetes prevalence (**15.2%**) than low-income individuals of normal weight (**16.3%**).
4. **Comorbid Health Deterioration:** Diagnosed individuals report nearly double the monthly sick days (**12.4 days vs 6.6 days**), with physical impairment rising by **118%**.
5. **Cumulative Lifestyle Shield:** Adopting 4 protective habits (daily exercise, produce intake, non-smoking) provides a **60.4% relative risk reduction**.

---

## Repository Structure

```text
cdc-diabetes-lifestyle-study/
│
├── README.md                          # Project documentation and reproduction guide
├── requirements.txt                   # Pinned project dependencies
├── .gitignore                         # Local environment and data exclusion rules
│
├── data/
│   ├── raw/                           # Raw CDC survey files (excluded from Git)
│   └── processed/                     # Curated, downcasted, and feature-engineered datasets
│
├── notebooks/
│   ├── 01_data_cleaning_and_audit.ipynb   # Audit, downcasting, outlier filter & feature engineering
│   └── 02_exploratory_and_storytelling.ipynb # Statistical hypothesis tests, distributions & charts
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py               # Batch data processing script
│   └── visualization.py               # Production chart generator
│
└── figures/                           # Exported high-resolution visual evidence
    └── diabetes_comprehensive_dashboard.png
```
