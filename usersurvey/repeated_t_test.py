import io
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

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

# Get individual test scores for repeated measures
long_df_test_level = long_df.copy()

# Aggregate by participant, condition, and category for overall scores
long_df = (
    long_df.groupby(["Participant", "Condition", "Category"])["Correct"]
    .mean().apply(lambda x: x * 100)
    .reset_index()
)

# --- Rename conditions ---
long_df["Condition"] = long_df["Condition"].replace({
    "A:": "Baseline",
    "B:": "PerceptTwin"
})

long_df_test_level["Condition"] = long_df_test_level["Condition"].replace({
    "A:": "Baseline",
    "B:": "PerceptTwin"
})

# Convert Participant to string to ensure it's treated as categorical
long_df["Participant"] = long_df["Participant"].astype(str)
long_df_test_level["Participant"] = long_df_test_level["Participant"].astype(str)

print("First few rows of the long format data:")
print(long_df.head())
print(f"\nTotal observations: {len(long_df)}")
print(f"Number of unique participants: {long_df['Participant'].nunique()}")

# --- Repeated t-tests with Holm correction ---
print("\n=== Repeated t-tests with Holm correction ===")

# Create all possible comparisons
comparisons = []

# 1. Compare conditions within each category
for category in long_df['Category'].unique():
    baseline_data = long_df[(long_df['Condition'] == 'Baseline') & (long_df['Category'] == category)]['Correct']
    percept_data = long_df[(long_df['Condition'] == 'PerceptTwin') & (long_df['Category'] == category)]['Correct']

    tmp = min(len(baseline_data), len(percept_data))
    # Paired t-test (repeated measures)
    t_stat, p_val = stats.ttest_rel(baseline_data[:tmp], percept_data[:tmp])
    comparisons.append({
        'Comparison': f'{category}: Baseline vs PerceptTwin',
        't-statistic': t_stat,
        'p-value': p_val,
        'Baseline_mean': baseline_data.mean(),
        'PerceptTwin_mean': percept_data.mean()
    })

# 2. Compare categories within each condition
for condition in long_df['Condition'].unique():
    logic_data = long_df[(long_df['Category'] == 'Logic') & (long_df['Condition'] == condition)]['Correct']
    consistency_data = long_df[(long_df['Category'] == 'Consistency') & (long_df['Condition'] == condition)]['Correct']
    tmp = min(len(logic_data), len(consistency_data))
    # Paired t-test (repeated measures)
    t_stat, p_val = stats.ttest_rel(logic_data[:tmp], consistency_data[:tmp])
    comparisons.append({
        'Comparison': f'{condition}: Logic vs Consistency',
        't-statistic': t_stat,
        'p-value': p_val,
        'Logic_mean': logic_data.mean(),
        'Consistency_mean': consistency_data.mean()
    })

# Convert to DataFrame
comparisons_df = pd.DataFrame(comparisons)

# Apply Holm correction
p_values = comparisons_df['p-value'].values
reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='holm')

comparisons_df['p-value_holm'] = pvals_corrected
comparisons_df['significant_holm'] = reject

print("\nResults of repeated t-tests with Holm correction:")
print(comparisons_df.to_string(index=False))

# --- Additional analysis: t-tests for each individual test ---
print("\n=== Individual test analysis with Holm correction ===")

test_comparisons = []
for test_num in range(1, 6):
    for category in long_df_test_level['Category'].unique():
        if test_num in [1, 3] and category == "Logic" or test_num in [2, 4, 5] and category == "Consistency":
            baseline_data = long_df_test_level[(long_df_test_level['Condition'] == 'Baseline') &
                                               (long_df_test_level['Category'] == category) &
                                               (long_df_test_level['Test'] == test_num)]['Correct']
            percept_data = long_df_test_level[(long_df_test_level['Condition'] == 'PerceptTwin') &
                                              (long_df_test_level['Category'] == category) &
                                              (long_df_test_level['Test'] == test_num)]['Correct']

            # Paired t-test
            t_stat, p_val = stats.ttest_rel(baseline_data, percept_data)
            test_comparisons.append({
                'Test': test_num,
                'Category': category,
                't-statistic': t_stat,
                'p-value': p_val,
                'Baseline_mean': baseline_data.mean(),
                'PerceptTwin_mean': percept_data.mean()
            })

# Convert to DataFrame and apply Holm correction
test_comparisons_df = pd.DataFrame(test_comparisons)
p_values_test = test_comparisons_df['p-value'].values
reject_test, pvals_corrected_test, _, _ = multipletests(p_values_test, alpha=0.05, method='holm')

test_comparisons_df['p-value_holm'] = pvals_corrected_test
test_comparisons_df['significant_holm'] = reject_test

print("\nResults of individual test comparisons with Holm correction:")
print(test_comparisons_df.to_string(index=False))

# --- Plotting the results ---
plt.figure(figsize=(12, 6))

# Plot 1: Overall performance by condition and category
plt.subplot(1, 2, 1)
sns.barplot(data=long_df, x='Category', y='Correct', hue='Condition', errorbar='se')
plt.title('Overall Performance by Condition and Category')
plt.ylabel('Correctness (%)')

# Plot 2: Individual test performance
plt.subplot(1, 2, 2)
test_means = long_df_test_level.groupby(['Test', 'Condition', 'Category'])['Correct'].mean().reset_index()
sns.barplot(data=test_means, x='Test', y='Correct', hue='Condition', errorbar='se')
plt.title('Performance by Individual Test')
plt.ylabel('Correctness (%)')

plt.tight_layout()
plt.show()