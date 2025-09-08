import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

# === Load CSV ===
df = pd.read_csv("Full text export Robot Planning Survey Release.xlsx - All Data.csv")

# Identify AB Test columns and paired question columns
cols = df.columns
ab_pairs = []
for i, col in enumerate(cols):
    if "AB Test" in str(col):
        if i + 1 < len(cols):
            ab_pairs.append((col, cols[i + 1]))

# Correct answers
true_answers = {1: "Failure", 2: "Failure", 3: "Success", 4: "Failure", 5: "Failure"}

# Convert to long format
def responses_to_long(df):
    rows = []
    for test_idx, (ab_col, q_col) in enumerate(ab_pairs, start=1):
        correct_answer = true_answers[test_idx]
        subset = df[[ab_col, q_col]].dropna()
        for i, row in subset.iterrows():
            ab = row[ab_col]
            ans = row[q_col]
            correct = int(ans == correct_answer)
            category = "Logic (Q1&3)" if test_idx in [1, 3] else "Semantic (Q2,4,5)"
            rows.append({
                "Participant": i,
                "Condition": ab,
                "Category": category,
                "Test": test_idx,
                "Correct": correct
            })
    return pd.DataFrame(rows)

long_df = responses_to_long(df)

# --- Linear Mixed Effects Model ---
model = smf.mixedlm("Correct ~ Condition * Category", data=long_df, groups="Participant")
lme_result = model.fit()
print("\n=== Linear Mixed Effects Model Summary ===")
print(lme_result.summary())

# --- Type III Wald test (ANOVA-like table for fixed effects) ---
from statsmodels.stats.anova import anova_lm
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings

warnings.simplefilter("ignore", ConvergenceWarning)  # suppress warnings

# Fit a regular OLS ignoring random effects for ANOVA-style p-values
ols_model = smf.ols("Correct ~ Condition * Category + C(Participant)", data=long_df).fit()
anova_table = anova_lm(ols_model, typ=3)
print("\n=== ANOVA-style Table for Fixed Effects ===")
print(anova_table)

# --- Plot mean accuracy by condition and category ---
agg = long_df.groupby(["Condition", "Category"])["Correct"].mean().reset_index()
plt.figure(figsize=(8,6))
sns.barplot(
    data=agg,
    x="Category",
    y="Correct",
    hue="Condition",
    palette=["skyblue", "salmon"]
)
plt.ylabel("Accuracy")
plt.title("Accuracy by Condition and Category (LME)")
plt.ylim(0, 1)
plt.axhline(0.5, color="gray", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()
