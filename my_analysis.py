import glob
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ==========================================
# STEP 1: AUTOMATICALLY FIND THE CSV FILE
# ==========================================
print("Searching for CSV files in your project...")
all_csv_files = glob.glob("**/*.csv", recursive=True)

if not all_csv_files:
    print("\n[ERROR] No CSV file found anywhere in this folder!")
    print(
        "Please check the left sidebar in VS Code to see if a dataset was downloaded or if it needs to be unzipped."
    )
    exit()

print(f"Found the following CSV files: {all_csv_files}")

# Pick a diabetes file from the list, preferring a balanced 50/50 split if present
file_to_use = None
for f in all_csv_files:
    if "5050" in f:
        file_to_use = f
        break
    elif "diabetes" in f.lower():
        file_to_use = f

# If none matched the keywords, default to the first CSV found
if file_to_use is None:
    file_to_use = all_csv_files[0]

print(f"\n--> Loading: {file_to_use}")
df = pd.read_csv(file_to_use)

# Standardize column names to lowercase
df.columns = df.columns.str.strip().str.lower()
print(f"Raw data loaded: {df.shape[0]:,} rows and {df.shape[1]} columns.\n")

# ==========================================
# STEP 2: CLEAN THE DATA
# ==========================================
# 1. Remove duplicate entries
before_dupes = len(df)
df = df.drop_duplicates()
print(f"Removed {before_dupes - len(df):,} duplicate entries.")

# 2. Filter realistic BMI values if BMI column exists
if "bmi" in df.columns:
    df = df[(df["bmi"] >= 12) & (df["bmi"] <= 70)]

# 3. Locate target column
target_candidates = [c for c in df.columns if "diabetes" in c]
if not target_candidates:
    print(
        "[ERROR] Could not find a target column with 'diabetes' in its name."
    )
    print(f"Available columns: {list(df.columns)}")
    exit()

target_col = target_candidates[0]
print(f"Target column identified: '{target_col}'\n")

# ==========================================
# STEP 3: EXTRACT KEY HEALTH INSIGHTS
# ==========================================
print("=" * 45)
print("             KEY HEALTH INSIGHTS            ")
print("=" * 45)

# Insight 1: Blood Pressure
if "highbp" in df.columns:
    bp_data = df.groupby("highbp")[target_col].mean() * 100
    print(f"• Diabetes rate WITH High BP:       {bp_data.get(1, 0):.1f}%")
    print(f"• Diabetes rate WITHOUT High BP:    {bp_data.get(0, 0):.1f}%\n")

# Insight 2: Cholesterol
if "highchol" in df.columns:
    chol_data = df.groupby("highchol")[target_col].mean() * 100
    print(f"• Diabetes rate WITH High Chol:     {chol_data.get(1, 0):.1f}%")
    print(f"• Diabetes rate WITHOUT High Chol:  {chol_data.get(0, 0):.1f}%\n")

# Insight 3: Physical Activity
if "physactivity" in df.columns:
    activity_data = df.groupby("physactivity")[target_col].mean() * 100
    print(f"• Diabetes rate for Active people:  {activity_data.get(1, 0):.1f}%")
    print(
        f"• Diabetes rate for Inactive people:{activity_data.get(0, 0):.1f}%\n"
    )

# ==========================================
# STEP 4: GENERATE AND SAVE A SUMMARY GRAPH
# ==========================================
print("Generating summary chart...")
plt.figure(figsize=(8, 5))

correlations = (
    df.corr(numeric_only=True)[target_col]
    .sort_values(ascending=False)
    .drop(target_col)
)
top_5 = correlations.head(5)

sns.barplot(x=top_5.values, y=top_5.index, palette="Blues_r")
plt.title("Top 5 Risk Factors Most Strongly Linked to Diabetes", fontsize=12)
plt.xlabel("Correlation Strength")
plt.ylabel("Indicator")

plt.tight_layout()
plt.savefig("top_diabetes_risk_factors.png")
print("\n[SUCCESS] Chart saved as 'top_diabetes_risk_factors.png'.")