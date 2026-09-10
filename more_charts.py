import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ==========================================
# STEP 1: LOAD THE CLEANED DATA
# ==========================================
csv_files = glob.glob("*.csv")
file_to_use = None
for f in csv_files:
    if "5050" in f:
        file_to_use = f
        break
    elif "diabetes" in f.lower():
        file_to_use = f

if not file_to_use:
    file_to_use = csv_files[0]

print(f"Generating charts using: {file_to_use}")
df = pd.read_csv(file_to_use)
df.columns = df.columns.str.strip().str.lower()
df = df.drop_duplicates().dropna()

target_col = [c for c in df.columns if "diabetes" in c][0]
sns.set_theme(style="whitegrid")

# =======================================================
# CHART 1: CHRONIC CARDIOVASCULAR COMORBIDITY PYRAMID
# =======================================================
# Evaluates Diabetes vs. Stroke vs. Heart Disease
cardio_cols = [c for c in ["heartdiseaseorattack", "stroke"] if c in df.columns]

if cardio_cols:
    plt.figure(figsize=(8, 5))
    rates = []
    labels = []
    
    # Baseline
    rates.append(df[target_col].mean() * 100)
    labels.append("Entire Population")
    
    # Heart Disease
    if "heartdiseaseorattack" in df.columns:
        rates.append(df[df["heartdiseaseorattack"] == 1][target_col].mean() * 100)
        labels.append("With Heart Disease/Attack")
        
    # Stroke
    if "stroke" in df.columns:
        rates.append(df[df["stroke"] == 1][target_col].mean() * 100)
        labels.append("With Prior Stroke")
        
    # Both
    if len(cardio_cols) == 2:
        both = df[(df["heartdiseaseorattack"] == 1) & (df["stroke"] == 1)]
        rates.append(both[target_col].mean() * 100)
        labels.append("Both Heart Disease & Stroke")

    palette = sns.color_palette("Reds", len(rates))
    bars = plt.barh(labels, rates, color=palette)
    plt.title("Diabetes Prevalence in Cardiovascular Patient Subsets", fontsize=12, fontweight="bold")
    plt.xlabel("Diabetes Prevalence (%)")
    plt.xlim(0, 100)

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 1.5, bar.get_y() + bar.get_height() / 2, f"{width:.1f}%", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig("chart1_cardio_comorbidities.png", dpi=300)
    plt.close()
    print("Saved: chart1_cardio_comorbidities.png")

# =======================================================
# CHART 2: AGE TRAJECTORY ACROSS BMI CLASSIFICATIONS
# =======================================================
if "age" in df.columns and "bmi" in df.columns:
    # Segment BMI into 3 broad clinical tiers
    df["weight_class"] = pd.cut(
        df["bmi"], 
        bins=[0, 24.9, 29.9, 100], 
        labels=["Normal (<25)", "Overweight (25-29.9)", "Obese (30+)"]
    )
    
    age_bmi_pivot = df.groupby(["age", "weight_class"], observed=False)[target_col].mean().unstack() * 100

    plt.figure(figsize=(10, 5))
    for col in age_bmi_pivot.columns:
        plt.plot(age_bmi_pivot.index, age_bmi_pivot[col], marker="o", lw=2.2, label=col)

    plt.title("Age-Onset Trajectory: Diabetes Risk Separated by Weight Category", fontsize=12, fontweight="bold")
    plt.xlabel("BRFSS Age Category (1: 18-24 ... 13: 80+)")
    plt.ylabel("Diabetes Rate (%)")
    plt.xticks(range(1, 14))
    plt.legend(title="Weight Category")
    plt.tight_layout()
    plt.savefig("chart2_age_bmi_interaction.png", dpi=300)
    plt.close()
    print("Saved: chart2_age_bmi_interaction.png")

# =======================================================
# CHART 3: FUNCTIONAL MOBILITY DISABILITY VS GENERAL HEALTH
# =======================================================
# How physical walking impairment reflects self-rated general health
if "diffwalk" in df.columns and "genhlth" in df.columns:
    plt.figure(figsize=(9, 5))
    mobility_df = df.groupby(["genhlth", "diffwalk"])[target_col].mean().reset_index()
    mobility_df["diffwalk"] = mobility_df["diffwalk"].map({0: "No Difficulty Walking", 1: "Difficulty Walking"})
    mobility_df[target_col] = mobility_df[target_col] * 100

    sns.barplot(
        data=mobility_df, 
        x="genhlth", 
        y=target_col, 
        hue="diffwalk", 
        palette="Blues"
    )
    plt.title("Diabetes Rate: Self-Rated General Health vs. Walking Mobility", fontsize=12, fontweight="bold")
    plt.xlabel("General Health Score (1: Excellent to 5: Poor)")
    plt.ylabel("Diabetes Rate (%)")
    plt.legend(title="")
    plt.tight_layout()
    plt.savefig("chart3_mobility_vs_genhealth.png", dpi=300)
    plt.close()
    print("Saved: chart3_mobility_vs_genhealth.png")

# =======================================================
# CHART 4: FEATURE CORRELATION CLUSTERING HEATMAP
# =======================================================
# Highlights which clusters of variables move together
key_features = [
    c for c in [
        target_col, "highbp", "highchol", "bmi", "smoker", "physactivity", 
        "heartdiseaseorattack", "genhlth", "diffwalk", "age", "income"
    ] if c in df.columns
]

plt.figure(figsize=(10, 8))
corr_matrix = df[key_features].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(
    corr_matrix, 
    mask=mask,
    annot=True, 
    fmt=".2f", 
    cmap="vlag", 
    vmin=-0.4, 
    vmax=0.6,
    cbar_kws={"label": "Pearson Correlation"}
)
plt.title("Correlation Matrix of Key Clinical & Lifestyle Factors", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("chart4_correlation_matrix.png", dpi=300)
plt.close()
print("Saved: chart4_correlation_matrix.png")

print("\nAll 4 charts have been generated and saved successfully!")