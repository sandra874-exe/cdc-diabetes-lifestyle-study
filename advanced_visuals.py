import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ==========================================
# STEP 1: LOAD CLEANED DATASET
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

print(f"Generating advanced plots from: {file_to_use}")
df = pd.read_csv(file_to_use)
df.columns = df.columns.str.strip().str.lower()
df = df.drop_duplicates().dropna()

target_col = [c for c in df.columns if "diabetes" in c][0]
sns.set_theme(style="whitegrid", font_scale=1.0)

# =======================================================
# CHART 5: VIOLIN PLOT - BMI DISTRIBUTION BY HEALTH STATUS
# =======================================================
# Shows kernel density, median, and interquartile range across classes
if "bmi" in df.columns and "genhlth" in df.columns:
    plt.figure(figsize=(10, 6))
    
    # Filter reasonable epidemiological range
    plot_df = df[(df["bmi"] >= 12) & (df["bmi"] <= 60)].copy()
    plot_df["status_label"] = plot_df[target_col].map({0: "Non-Diabetic", 1: "Diabetic"})
    
    sns.violinplot(
        data=plot_df,
        x="genhlth",
        y="bmi",
        hue="status_label",
        split=True,
        inner="quartile",
        palette={"Non-Diabetic": "#4C78A8", "Diabetic": "#E45756"}
    )
    plt.title("BMI Density Across Self-Reported General Health (1=Excellent, 5=Poor)", fontweight="bold")
    plt.xlabel("General Health Score")
    plt.ylabel("Body Mass Index (BMI)")
    plt.legend(title="Condition Status", loc="upper left")
    plt.tight_layout()
    plt.savefig("chart5_bmi_health_violin.png", dpi=300)
    plt.close()
    print("Saved: chart5_bmi_health_violin.png")

# =======================================================
# CHART 6: RADAR / SPIDER CHART - RISK FACTOR PROFILES
# =======================================================
# Compares the average lifestyle and clinical footprint of both groups
radar_features = [
    "highbp", "highchol", "smoker", "diffwalk", 
    "heartdiseaseorattack", "stroke"
]
available_radar = [c for c in radar_features if c in df.columns]

if available_radar:
    # Compute mean indicator values per group (0.0 to 1.0 scale)
    radar_data = df.groupby(target_col)[available_radar].mean()
    
    categories = [col.replace("orattack", "").capitalize() for col in available_radar]
    num_vars = len(categories)
    
    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circular loop
    
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    
    # Non-Diabetic Profile
    vals_0 = radar_data.loc[0].tolist()
    vals_0 += vals_0[:1]
    ax.plot(angles, vals_0, color="#4C78A8", linewidth=2, label="Non-Diabetic")
    ax.fill(angles, vals_0, color="#4C78A8", alpha=0.25)
    
    # Diabetic Profile
    vals_1 = radar_data.loc[1].tolist()
    vals_1 += vals_1[:1]
    ax.plot(angles, vals_1, color="#E45756", linewidth=2, label="Diabetic")
    ax.fill(angles, vals_1, color="#E45756", alpha=0.25)
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories, size=10)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["20%", "40%", "60%", "80%"], color="grey", size=8)
    plt.ylim(0, 1.0)
    
    plt.title("Clinical & Lifestyle Risk Profile Comparison", size=13, fontweight="bold", y=1.08)
    plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    plt.tight_layout()
    plt.savefig("chart6_risk_radar_profile.png", dpi=300)
    plt.close()
    print("Saved: chart6_risk_radar_profile.png")

# =======================================================
# CHART 7: DUAL-AXIS SOCIOECONOMIC HEALTH DIVIDE
# =======================================================
# Visualizes the inverse relationship: Higher Income vs. Less Walking Difficulty & Lower Diabetes
if "income" in df.columns and "diffwalk" in df.columns:
    socio_summary = df.groupby("income").agg(
        diabetes_rate=(target_col, "mean"),
        diffwalk_rate=("diffwalk", "mean")
    ).reset_index()
    
    socio_summary["diabetes_rate"] *= 100
    socio_summary["diffwalk_rate"] *= 100
    
    income_brackets = [
        "<$10k", "$10-15k", "$15-20k", "$20-25k", 
        "$25-35k", "$35-50k", "$50-75k", ">$75k"
    ]
    x_labels = income_brackets[:len(socio_summary)]
    
    fig, ax1 = plt.subplots(figsize=(9, 5))
    
    color = "#E45756"
    ax1.set_xlabel("Household Income Bracket", fontweight="bold")
    ax1.set_ylabel("Diabetes Prevalence (%)", color=color, fontweight="bold")
    line1 = ax1.plot(x_labels, socio_summary["diabetes_rate"], color=color, marker="o", lw=2.5, label="Diabetes Rate")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_ylim(0, max(socio_summary["diabetes_rate"]) + 15)
    
    ax2 = ax1.twinx()
    color = "#4C78A8"
    ax2.set_ylabel("Difficulty Walking Rate (%)", color=color, fontweight="bold")
    line2 = ax2.plot(x_labels, socio_summary["diffwalk_rate"], color=color, marker="s", lw=2.5, linestyle="--", label="DiffWalk Rate")
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylim(0, max(socio_summary["diffwalk_rate"]) + 15)
    
    plt.title("Socioeconomic Gradients: Diabetes Prevalence vs. Physical Mobility", fontweight="bold", fontsize=12)
    fig.tight_layout()
    plt.savefig("chart7_income_gradient_dual_axis.png", dpi=300)
    plt.close()
    print("Saved: chart7_income_gradient_dual_axis.png")

# =======================================================
# CHART 8: HEALTHCARE ACCESS BARRIER MATRIX
# =======================================================
# Examines lack of medical coverage and financial barriers (NoDocBCost)
if "anyhealthcare" in df.columns and "nodocbccost" in df.columns:
    plt.figure(figsize=(8, 5))
    
    access_df = df.groupby(["anyhealthcare", "nodocbccost"])[target_col].mean().reset_index()
    access_df[target_col] *= 100
    access_df["anyhealthcare"] = access_df["anyhealthcare"].map({0: "No Insurance", 1: "Has Insurance"})
    access_df["nodocbccost"] = access_df["nodocbccost"].map({0: "Cost Not a Barrier", 1: "Cost Prevented Care"})
    
    sns.barplot(
        data=access_df,
        x="anyhealthcare",
        y=target_col,
        hue="nodocbccost",
        palette="Blues"
    )
    plt.title("Diabetes Prevalence by Healthcare Coverage and Cost Barriers", fontweight="bold", fontsize=12)
    plt.xlabel("Healthcare Insurance Status")
    plt.ylabel("Diabetes Rate (%)")
    plt.legend(title="Doctor Visit Barrier")
    plt.ylim(0, 100)
    
    for p in plt.gca().patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            plt.gca().annotate(f"{height:.1f}%", (p.get_x() + p.get_width() / 2., height + 1.5), ha="center", fontsize=9)
            
    plt.tight_layout()
    plt.savefig("chart8_healthcare_access_barrier.png", dpi=300)
    plt.close()
    print("Saved: chart8_healthcare_access_barrier.png")

print("\nAll 4 advanced figures successfully generated and saved to your folder!")