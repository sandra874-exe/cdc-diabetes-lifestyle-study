from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    path = PROCESSED_DIR / "diabetes_full_cleaned.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {path}\n"
            "Run: python src/preprocessing.py"
        )
    return pd.read_csv(path)

def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.18)
    ax.grid(axis="y", alpha=0.0)

def fig6(df):
    age_labels = {
        1:"18–24",2:"25–29",3:"30–34",4:"35–39",5:"40–44",
        6:"45–49",7:"50–54",8:"55–59",9:"60–64",10:"65–69",
        11:"70–74",12:"75–79",13:"80+"
    }
    work = df.copy()
    work["Age_Group"] = work["Age"].map(age_labels)
    summary = (work.groupby(["Age_Group","Metabolic_Factor_Count"])["Diabetes_binary"]
               .mean().mul(100).unstack()
               .reindex(list(age_labels.values())).reindex(columns=[0,1,2,3]))
    summary.columns = ["0 factors","1 factor","2 factors","3 factors"]
    fig, ax = plt.subplots(figsize=(10.5,8))
    sns.heatmap(summary, annot=True, fmt=".1f", cmap="YlOrRd",
                linewidths=.5,
                cbar_kws={"label":"Prediabetes/diabetes prevalence (%)"}, ax=ax)
    ax.set_title("Metabolic Burden Grows More Informative Across the Age Spectrum",
                 fontsize=16, fontweight="bold", pad=16)
    ax.set_xlabel("Number of metabolic factors\n(High BP + High Cholesterol + Obesity)")
    ax.set_ylabel("Age group")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR/"fig6_age_metabolic_burden_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def fig7(df):
    labels = {1:"Excellent",2:"Very good",3:"Good",4:"Fair",5:"Poor"}
    summary = (df.groupby(["GenHlth","Diabetes_binary"])["DiffWalk"].mean().mul(100)
               .unstack().rename(columns={0:"No Prediabetes/Diabetes",1:"Prediabetes/Diabetes"})
               .reindex([1,2,3,4,5]))
    summary.index = [labels[i] for i in summary.index]
    fig, ax = plt.subplots(figsize=(10.5,6.5))
    for group in summary.columns:
        vals = summary[group].to_numpy()
        ax.plot(summary.index, vals, marker="o", linewidth=3, markersize=8, label=group)
        for x,y in enumerate(vals):
            ax.annotate(f"{y:.1f}%", (x,y), xytext=(0,9),
                        textcoords="offset points", ha="center", fontsize=9)
    ax.set_title("Difficulty Walking Rises With Poorer General Health — and Is Higher With Prediabetes/Diabetes",
                 fontsize=15, fontweight="bold", pad=16)
    ax.set_xlabel("Self-reported general health")
    ax.set_ylabel("Participants reporting difficulty walking (%)")
    ax.set_ylim(0, float(summary.max().max())*1.22)
    style_axis(ax)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR/"fig7_general_health_mobility.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def fig8(df):
    summary = (df.groupby(["AnyHealthcare","NoDocbcCost"])["Diabetes_binary"].mean().mul(100)
               .unstack()
               .rename(index={0:"No healthcare coverage",1:"Has healthcare coverage"},
                       columns={0:"Cost did not prevent care",1:"Cost prevented care"})
               .reindex(index=["No healthcare coverage","Has healthcare coverage"],
                        columns=["Cost did not prevent care","Cost prevented care"]))
    fig, ax = plt.subplots(figsize=(9.5,5.8))
    sns.heatmap(summary, annot=True, fmt=".1f", cmap="Blues", linewidths=.7,
                cbar_kws={"label":"Prediabetes/diabetes prevalence (%)"}, ax=ax)
    ax.set_title("Prediabetes/Diabetes Prevalence Differs Across Healthcare Access Conditions",
                 fontsize=15, fontweight="bold", pad=16)
    ax.set_xlabel("Cost-related barrier to seeing a doctor")
    ax.set_ylabel("Healthcare coverage")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR/"fig8_healthcare_access_cost_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def main():
    sns.set_theme(style="whitegrid", font="DejaVu Sans")
    df = load_data()
    print("Loaded:", PROCESSED_DIR/"diabetes_full_cleaned.csv")
    print("Shape:", df.shape)
    fig6(df); fig7(df); fig8(df)
    print("All three figures generated successfully.")

if __name__ == "__main__":
    main()
