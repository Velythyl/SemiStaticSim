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
def balance_participants(long_df):
    # Get all unique factor combinations
    factor_combinations = long_df[['Condition', 'Category']].drop_duplicates()

    # Count how many observations each participant has for each combination
    participant_counts = long_df.groupby(['Participant', 'Condition', 'Category']).size().reset_index(name='count')

    # Find participants who have all 4 factor combinations
    complete_participants = []
    for participant in long_df['Participant'].unique():
        participant_data = participant_counts[participant_counts['Participant'] == participant]
        if len(participant_data) == 4:  # Should have all 4 combinations
            complete_participants.append(participant)

    print(f"Found {len(complete_participants)} participants with all 4 factor combinations")

    # Filter to only include complete participants
    balanced_long_df = long_df[long_df['Participant'].isin(complete_participants)]

    # Verify balance
    balance_check = balanced_long_df.groupby(['Participant', 'Condition', 'Category']).size().reset_index(name='count')
    print("Balance check - counts per combination:")
    print(balance_check.groupby(['Condition', 'Category']).size())

    return balanced_long_df


# Apply balancing
balanced_long_df = balance_participants(long_df)

# --- Aggregate accuracy at participant × condition × category ---
agg = (
    balanced_long_df.groupby(["Participant", "Condition", "Category"])["Correct"]
    .mean()
    .reset_index()
)

# --- Repeated measures ANOVA ---
print("\n=== Two-way Repeated Measures ANOVA (Balanced Design) ===")
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
    print(f"ANOVA failed: {e}")
    print("This might indicate insufficient data after balancing")

# --- Check if we have enough data for ANOVA ---
if len(agg['Participant'].unique()) < 2:
    print("\nWARNING: Not enough participants remaining after balancing for ANOVA")
    print("Consider alternative analysis methods or collecting more data")
else:
    # --- Plot mean accuracy by condition and category ---
    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=agg,
        x="Category",
        y="Correct",
        hue="Condition",
        ci=95,
        palette=["skyblue", "salmon"]
    )
    plt.ylabel("Accuracy")
    plt.title("Accuracy by Condition and Category (Balanced Repeated Measures ANOVA)")
    plt.ylim(0, 1)
    plt.axhline(0.5, color="gray", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

# --- Alternative: Show participant distribution before and after balancing ---
print("\n=== Participant Distribution ===")
print(f"Original participants: {len(long_df['Participant'].unique())}")
print(f"Balanced participants: {len(balanced_long_df['Participant'].unique())}")

# Show condition distribution
print("\nCondition distribution in balanced data:")
print(balanced_long_df.groupby(['Condition', 'Category']).size())