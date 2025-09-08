import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.anova import AnovaRM
from collections import Counter

# === Load CSV ===
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


# --- Balance participants across all factor combinations ---
def balance_participants(df):
    # Get all unique factor combinations
    factor_combinations = df.groupby(['Condition', 'Category'])['Participant'].apply(set).to_dict()

    # Count participants in each combination
    combination_counts = {combo: len(participants) for combo, participants in factor_combinations.items()}
    print("Initial participant counts per factor combination:")
    for combo, count in combination_counts.items():
        print(f"{combo}: {count} participants")

    # Find the minimum count across all combinations
    min_count = min(combination_counts.values())
    print(f"\nMinimum count across all combinations: {min_count}")

    # Get balanced set of participants
    balanced_participants = set()
    to_remove = set()
    for combo, participants in factor_combinations.items():
        if len(participants) == min_count:
            continue
        to_remove.update(set(list(participants)[min_count:]))

    print(f"\nTotal participants to remove: {len(to_remove)}")
    return to_remove


# Get balanced participants
balanced_participants = balance_participants(long_df)
balanced_long_df = long_df[~long_df['Participant'].isin(balanced_participants)]
balanced_participants = balance_participants(balanced_long_df)
balanced_long_df = balanced_long_df[~balanced_long_df['Participant'].isin(balanced_participants)]
balanced_participants = balance_participants(balanced_long_df)
balanced_long_df = balanced_long_df[~balanced_long_df['Participant'].isin(balanced_participants)]
balanced_participants = balance_participants(balanced_long_df)
balanced_long_df = balanced_long_df[~balanced_long_df['Participant'].isin(balanced_participants)]

# Filter the data to only include balanced participants
balanced_long_df = long_df[long_df['Participant'].isin(balanced_participants)]

# --- Aggregate accuracy at participant × condition × category ---
agg = (
    balanced_long_df.groupby(["Participant", "Condition", "Category"])["Correct"]
    .mean()
    .reset_index()
)

# --- Repeated measures ANOVA ---
print("\n=== Two-way Repeated Measures ANOVA (Balanced Data) ===")
try:
    aovrm = AnovaRM(
        data=agg,
        depvar="Correct",
        subject="Participant",
        within=["Condition", "Category"]
    )
    res = aovrm.fit()
    print(res)
except Exception as e:
    print(f"ANOVA failed with error: {e}")
    print("This might be due to insufficient data after balancing")

# --- Check final balance ---
print("\n=== Final Balance Check ===")
final_counts = balanced_long_df.groupby(['Condition', 'Category'])['Participant'].nunique()
print(final_counts)

# --- Plot mean accuracy by condition and category ---
plt.figure(figsize=(10, 6))
sns.barplot(
    data=agg,
    x="Category",
    y="Correct",
    hue="Condition",
    ci=95,
    palette=["skyblue", "salmon"]
)
plt.ylabel("Accuracy")
plt.title("Accuracy by Condition and Category (Balanced Repeated Measures)")
plt.ylim(0, 1)
plt.axhline(0.5, color="gray", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# --- Additional diagnostic plot to show balance ---
plt.figure(figsize=(10, 6))
balance_check = balanced_long_df.groupby(['Condition', 'Category'])['Participant'].nunique().reset_index()
sns.barplot(
    data=balance_check,
    x="Category",
    y="Participant",
    hue="Condition"
)
plt.ylabel("Number of Participants")
plt.title("Participant Balance Across Conditions and Categories")
plt.tight_layout()
plt.show()