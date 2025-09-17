import io
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf

# === Load CSV (your .xslt export saved as CSV) ===
df = pd.read_csv("Full text export Robot Planning Survey Release(2).xlsx - All Data.csv")

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
                category = "Logic"
            else:
                category = "Consistency"
            rows.append({
                "Participant": i,
                "Condition": ab,
                "Category": category,
                "Test": test_idx,
                "Correct": correct
            })
    return pd.DataFrame(rows)

long_df = responses_to_long(df)

long_df = (
    long_df.groupby(["Participant", "Condition", "Category"])["Correct"]
    .mean().apply(lambda x: x*100)
    .reset_index()
)

# --- Rename conditions ---
long_df["Condition"] = long_df["Condition"].replace({
    "A:": "Baseline",
    "B:": "PerceptTwin"
})

# Convert Participant to string to ensure it's treated as categorical
long_df["Participant"] = long_df["Participant"].astype(str)

print("First few rows of the long format data:")
print(long_df.head())
print(f"\nTotal observations: {len(long_df)}")
print(f"Number of unique participants: {long_df['Participant'].nunique()}")

# --- Mixed Effects Binary Logistic Regression ---
print("\n=== Mixed Effects Binary Logistic Regression ===")

# Fit the model with random intercepts for participants
model = smf.glm("Correct ~ Condition * Category",
                   data=long_df,
                   groups="Participant", family=statsmodels.genmod.families.family.Binomial,
                   re_formula="1")  # Random intercept only

result = model.fit(method='powell', maxiter=5000)  # Using BFGS optimizer

print(result.summary())
# === Check Model Assumptions ===
print("\n=== Checking Model Assumptions ===")

# Get predicted values and residuals
fitted_values = result.fittedvalues
residuals = result.resid

# 1. Plot residuals against explanatory variables
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Residual Diagnostics for Mixed Effects Model', fontsize=16)

# Create a combined explanatory variable for plotting
long_df['Condition_Category'] = long_df['Condition'] + '_' + long_df['Category']

# Plot residuals against Condition_Category
sns.boxplot(data=long_df.assign(Residuals=residuals),
            x='Condition_Category', y='Residuals', ax=axes[0,0])
axes[0,0].set_title('Residuals vs Condition × Category')
axes[0,0].tick_params(axis='x', rotation=45)

# 2. Plot residuals in order (for autocorrelation check)
axes[0,1].plot(residuals, 'o-', alpha=0.7)
axes[0,1].axhline(y=0, color='r', linestyle='--')
axes[0,1].set_title('Residuals in Order (Autocorrelation Check)')
axes[0,1].set_xlabel('Observation Order')
axes[0,1].set_ylabel('Residuals')

# 3. Plot residuals against fitted values
axes[1,0].scatter(fitted_values, residuals, alpha=0.7)
axes[1,0].axhline(y=0, color='r', linestyle='--')
axes[1,0].set_title('Residuals vs Fitted Values')
axes[1,0].set_xlabel('Fitted Values')
axes[1,0].set_ylabel('Residuals')

# 4. QQ plot for normality check
sm.qqplot(residuals, line='s', ax=axes[1,1])
axes[1,1].set_title('QQ Plot for Normality Check')

plt.tight_layout()
plt.show()

# Additional diagnostics
print("\n--- Additional Diagnostic Tests ---")

# Test for heteroscedasticity
# Since we have categorical predictors, we can test variance across groups
group_vars = []
for group in long_df['Condition_Category'].unique():
    group_residuals = residuals[long_df['Condition_Category'] == group]
    group_vars.append(np.var(group_residuals))

print(f"Variance of residuals by group: {dict(zip(long_df['Condition_Category'].unique(), group_vars))}")

# Test for normality of residuals
from scipy import stats
shapiro_test = stats.shapiro(residuals)
print(f"Shapiro-Wilk test for normality: W={shapiro_test.statistic:.3f}, p={shapiro_test.pvalue:.3f}")

# Check for influential observations
# Calculate Cook's distance (approximation for mixed models)
influence = result.get_influence()
cooks_d = influence.cooks_distance[0]
print(f"Max Cook's distance: {np.max(cooks_d):.3f}")
print(f"Number of observations with Cook's D > 0.5: {np.sum(cooks_d > 0.5)}")

# Check random effects assumptions
print("\n--- Random Effects Diagnostics ---")
# Get random effects
random_effects = result.random_effects
re_values = list(random_effects.values())
re_values = [list(x.values())[0] for x in re_values if x]

# Check normality of random effects
if re_values:
    re_shapiro = stats.shapiro(re_values)
    print(f"Random effects normality: W={re_shapiro.statistic:.3f}, p={re_shapiro.pvalue:.3f}")
else:
    print("No random effects to test")

# Check for convergence issues
print(f"\nConvergence successful: {result.converged}")
if hasattr(result, 'method'):
    print(f"Optimization method: {result.method}")
print(f"Log-likelihood: {result.llf:.3f}")

# Check model specification
print(f"\nModel formula: {model.formula}")
print(f"Number of groups (participants): {result.ngroups}")
print(f"Average group size: {result.nobs // result.ngroups}")


# --- Additional diagnostics ---
print("\n=== Model Diagnostics ===")
print(f"AIC: {result.aic}")
print(f"BIC: {result.bic}")

# Check the random effects variance
print(f"\nRandom effects variance (participant level): {result.cov_re.iloc[0,0]:.4f}")

# Calculate intraclass correlation coefficient (ICC)
var_random = result.cov_re.iloc[0,0]  # Random effects variance
var_residual = np.pi**2 / 3  # Residual variance for logistic regression (fixed)
icc = var_random / (var_random + var_residual)
print(f"Intraclass Correlation Coefficient (ICC): {icc:.4f}")