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

# --- Aggregate accuracy at participant × condition × category ---
agg = (
    long_df.groupby(["Participant", "Condition", "Category"])["Correct"]
    .mean()
    .reset_index()
)

# --- Repeated measures ANOVA using Pingouin ---
aov = pg.rm_anova(
    data=agg,
    dv='Correct',
    within=['Condition', 'Category'],
    subject='Participant',
    detailed=True
)
print("\n=== Two-way Repeated Measures ANOVA (Pingouin) ===")
print(aov)

# --- Plot mean accuracy by condition and category ---
plt.figure(figsize=(8,6))
sns.barplot(
    data=agg,
    x="Category",
    y="Correct",
    hue="Condition",
    ci=95,
    palette=["skyblue", "salmon"]
)
plt.ylabel("Accuracy")
plt.title("Accuracy by Condition and Category (Repeated Measures ANOVA)")
plt.ylim(0, 1)
plt.axhline(0.5, color="gray", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()
