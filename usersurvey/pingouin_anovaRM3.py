import copy

import numpy as np

import io
import numpy as np
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg

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
                category = "Logic    "
            else:
                category = "  Consistency"
            rows.append({
                "Participant": i,
                "Condition": ab,
                "Category": category,
                #"Test": test_idx,
                "Correct": correct
            })
    return pd.DataFrame(rows)

long_df = responses_to_long(df)


# --- Rename conditions ---
long_df["Condition"] = long_df["Condition"].replace({
    "A:": "Baseline",
    "B:": "PerceptTwin"
})

# --- Aggregate accuracy at participant × condition × category ---
agg = (
    long_df.groupby(["Participant", "Condition", "Category"])["Correct"]
    .mean().apply(lambda x: x*100)
    .reset_index()
)


# --- Apply ln(x+1) transformation to accuracy data ---
print("\n=== Applying ln(x+1) transformation ===")
agg['Correct_transformed'] = np.sqrt(agg['Correct']+10) #np.log(agg['Correct'] + 1)

# --- Normality tests on original data ---
print("\n=== Shapiro-Wilk Normality Tests (Original Data) ===")
for (cond, cat), sub in agg.groupby(["Condition", "Category"]):
    result = pg.normality(sub["Correct"])
    print(f"{cond}, {cat}:")
    print(result)
    print()

# --- Normality tests on transformed data ---
print("\n=== Shapiro-Wilk Normality Tests (Transformed Data: ln(x+1)) ===")
for (cond, cat), sub in agg.groupby(["Condition", "Category"]):
    result = pg.normality(sub["Correct_transformed"])
    print(f"{cond}, {cat}:")
    print(result)
    print()

tmp = agg[["Participant", "Condition", "Category", "Correct"]]
print(pg.homoscedasticity(tmp, dv="Correct", group="Participant"))
#exit()


# --- Visual comparison of distributions ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Original data distributions
sns.histplot(data=agg, x='Correct', hue='Condition', multiple='dodge',
             ax=axes[0, 0], kde=True)
axes[0, 0].set_title('Original Accuracy Distributions')
axes[0, 0].set_xlabel('Accuracy (%)')

sns.boxplot(data=agg, x='Category', y='Correct', hue='Condition',
            ax=axes[0, 1])
axes[0, 1].set_title('Original Accuracy by Category')
axes[0, 1].set_ylabel('Accuracy (%)')

# Transformed data distributions
sns.histplot(data=agg, x='Correct_transformed', hue='Condition', multiple='dodge',
             ax=axes[1, 0], kde=True)
axes[1, 0].set_title('Transformed Accuracy Distributions (ln(x+1))')
axes[1, 0].set_xlabel('ln(Accuracy + 1)')

sns.boxplot(data=agg, x='Category', y='Correct_transformed', hue='Condition',
            ax=axes[1, 1])
axes[1, 1].set_title('Transformed Accuracy by Category (ln(x+1))')
axes[1, 1].set_ylabel('ln(Accuracy + 1)')

plt.tight_layout()
plt.show()

# --- Q-Q plots for visual normality assessment ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Original data Q-Q plots
for i, ((cond, cat), sub) in enumerate(agg.groupby(["Condition", "Category"])):
    pg.qqplot(sub['Correct'], dist='norm', ax=axes[0, i % 2])
    axes[0, i % 2].set_title(f'Original: {cond}, {cat}')

# Transformed data Q-Q plots
for i, ((cond, cat), sub) in enumerate(agg.groupby(["Condition", "Category"])):
    pg.qqplot(sub['Correct_transformed'], dist='norm', ax=axes[1, i % 2])
    axes[1, i % 2].set_title(f'Transformed: {cond}, {cat}')

plt.tight_layout()
plt.show()

# --- Alternative transformations you might consider ---
print("\n=== Alternative Transformations ===")
agg['Correct_sqrt'] = np.sqrt(agg['Correct']+10)  # Square root transformation
agg['Correct_arcsin'] = np.arcsinh(agg['Correct'] / 100)  # Inverse hyperbolic sine

# Test normality for alternative transformations
transformations = ['Correct_sqrt', 'Correct_arcsin']
for transform in transformations:
    print(f"\n=== Shapiro-Wilk for {transform} ===")
    for (cond, cat), sub in agg.groupby(["Condition", "Category"]):
        print(pg.normality(sub[transform]))
        #print(f"{cond}, {cat}: W={result['W'].values[0]:.4f}, p={result['pval'].values[0]:.4f}")