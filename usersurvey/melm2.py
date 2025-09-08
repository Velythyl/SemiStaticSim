import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
from scipy import stats

# === Load CSV (your .xslt export saved as CSV) ===
df = pd.read_csv("Full text export Robot Planning Survey Release.xlsx - All Data.csv")

# Identify all AB Test columns and their paired question columns
cols = df.columns
ab_pairs = []
for i, col in enumerate(cols):
    if "AB Test" in str(col):
        if i + 1 < len(cols):
            ab_pairs.append((col, cols[i + 1]))

# Define the correct answers for each test
true_answers = {
    1: "Failure",
    2: "Failure",
    3: "Success",
    4: "Failure",
    5: "Failure"
}


# --- Function to compute per-participant correctness ---
def responses_to_long(df):
    rows = []
    for test_idx, (ab_col, q_col) in enumerate(ab_pairs, start=1):
        correct_answer = true_answers[test_idx]
        subset = df[[ab_col, q_col]].dropna()
        for i, row in subset.iterrows():
            ab = row[ab_col]
            ans = row[q_col]
            correct = int(ans == correct_answer)
            if test_idx in [1, 3]:
                category = "Logic (Q1&3)"
            else:
                category = "Semantic (Q2,4,5)"
            rows.append({
                "Participant": i,
                "Condition": ab,
                "Category": category,
                "Test": test_idx,
                "Correct": correct
            })
    return pd.DataFrame(rows)


long_df = responses_to_long(df)

# --- Mixed Effects Model Alternative for Unbalanced Data ---
print("\n=== Mixed Effects Linear Model (Alternative to Repeated Measures ANOVA) ===")

# Option 1: Linear Mixed Effects Model with participant as random effect
model_formula = "Correct ~ Condition * Category"

# Fit the mixed effects model
mixed_model = smf.mixedlm(
    model_formula,
    data=long_df,
    groups=long_df["Participant"],
    re_formula="1"  # Random intercept for each participant
)
mixed_results = mixed_model.fit()
print(mixed_results.summary())

# --- Post-hoc comparisons if significant interactions ---
print("\n=== Post-hoc Comparisons ===")

# Check if interaction is significant
if mixed_results.pvalues['Condition[T.B:]:Category[T.Semantic (Q2,4,5)]'] < 0.05:
    print("Significant interaction found - performing post-hoc tests:")

    # Compare conditions within each category
    for category in long_df['Category'].unique():
        subset = long_df[long_df['Category'] == category]
        cond_a = subset[subset['Condition'] == 'A']['Correct']
        cond_b = subset[subset['Condition'] == 'B']['Correct']

        t_stat, p_value = stats.ttest_ind(cond_a, cond_b, nan_policy='omit')
        print(f"Category {category}: Condition A vs B, t={t_stat:.3f}, p={p_value:.4f}")

# --- Plot mean accuracy by condition and category ---
plt.figure(figsize=(10, 6))

# Create a more detailed plot with individual data points
sns.barplot(
    data=long_df,
    x="Category",
    y="Correct",
    hue="Condition",
    ci=95,
    palette=["skyblue", "salmon"],
    alpha=0.7
)

# Add individual data points with some jitter
sns.stripplot(
    data=long_df,
    x="Category",
    y="Correct",
    hue="Condition",
    dodge=True,
    alpha=0.4,
    jitter=0.2,
    palette=["skyblue", "salmon"],
    legend=False
)

plt.ylabel("Accuracy")
plt.title("Accuracy by Condition and Category (Mixed Effects Model)")
plt.ylim(0, 1)
plt.axhline(0.5, color="gray", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.legend(title="Condition", loc='upper right')
plt.show()

# --- Additional diagnostic plots ---
print("\n=== Model Diagnostics ===")

# Residuals vs fitted values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
residuals = mixed_results.resid
fitted = mixed_results.fittedvalues
plt.scatter(fitted, residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted')

# QQ plot of residuals
plt.subplot(1, 2, 2)
sm.qqplot(residuals, line='s')
plt.title('QQ Plot of Residuals')

plt.tight_layout()
plt.show()

# --- Alternative: Generalized Estimating Equations (GEE) approach ---
print("\n=== Alternative: Generalized Estimating Equations (GEE) ===")

# GEE can also handle correlated data and is robust to misspecification
try:
    # Using an exchangeable correlation structure (appropriate for repeated measures)
    families = sm.families.Binomial()
    gee_model = smf.gee(
        model_formula,
        "Participant",  # grouping variable
        long_df,
        family=families,
        cov_struct=sm.cov_struct.Exchangeable()
    )
    gee_results = gee_model.fit()
    print(gee_results.summary())
except Exception as e:
    print(f"GEE model failed: {e}")
    print("This is common with certain data structures - mixed model is preferred.")

# --- Report model comparison ---
print("\n=== Model Interpretation ===")
print("The mixed effects model accounts for:")
print("1. Fixed effects: Condition and Category (and their interaction)")
print("2. Random effects: Participant-specific intercepts (accounting for repeated measures)")
print("3. Handles unbalanced data (different numbers of observations per participant)")
print("\nKey findings:")
print(f"- Condition effect: p = {mixed_results.pvalues.get('Condition[T.B]', 'N/A')}")
print(f"- Category effect: p = {mixed_results.pvalues.get('Category[T.Semantic (Q2,4,5)]', 'N/A')}")
print(f"- Interaction: p = {mixed_results.pvalues.get('Condition[T.B]:Category[T.Semantic (Q2,4,5)]', 'N/A')}")