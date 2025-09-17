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
                "Test": test_idx,
                "Correct": correct
            })

    """
    has_all = {}
    for row in rows:
        participant = row["Participant"]
        if participant not in has_all:
            has_all[participant] = set()
        has_all[participant].add((row["Condition"], row["Category"]))

    to_rm = set()
    for participant in has_all:
        if len(has_all[participant]) != 4:
            to_rm.add(participant)

    rows = [row for row in rows if row["Participant"] not in to_rm]"""

    return pd.DataFrame(rows)

long_df = responses_to_long(df)
print(long_df.shape)

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

# --- Repeated measures ANOVA using Pingouin ---
aov = pg.rm_anova(
    data=agg,
    dv='Correct',
    within=['Condition', 'Category'],
    subject='Participant',
    detailed=True
)

pd.set_option("display.max_columns", None)

print("\n=== Two-way Repeated Measures ANOVA (Pingouin) ===")
print(aov)
print("\n=== Two-way Repeated Measures ANOVA (Pingouin) ===")

# --- Paired t-tests per category ---
print("\n=== Paired t-tests (Baseline vs PerceptTwin) by Category ===")
for category in agg["Category"].unique():
    sub = agg[agg["Category"] == category]
    # Pivot so we have Baseline and PerceptTwin columns per participant
    pivoted = sub.pivot(index="Participant", columns="Condition", values="Correct").dropna()

    # Run paired t-test
    ttest_res = pg.ttest(
        pivoted["Baseline"],
        pivoted["PerceptTwin"],
        paired=True
    )

    print(f"\nCategory: {category}")
    print(ttest_res)

FONTSIZE = 18
plt.rcParams.update({
    "axes.titlesize": FONTSIZE,  # Title font size
    "axes.labelsize": FONTSIZE,  # Axis label font size
    "xtick.labelsize": FONTSIZE,  # X-tick font size
    "ytick.labelsize": FONTSIZE,  # Y-tick font size
    "legend.fontsize": 13.5,  # Legend font size
})

# --- Plot mean accuracy by condition and category ---
plt.figure(figsize=(2.5,8))  # square figure

# Use seaborn colourblind palette: blue for Baseline, orange for PerceptTwin
palette = {
    "Baseline": sns.color_palette("colorblind")[0],     # blue
    "PerceptTwin": sns.color_palette("colorblind")[1]   # orange
}

ax = sns.barplot(
    data=agg,
    x="Category",
    y="Correct",
    hue="Condition",
    ci=95,
    palette=palette
)

ax.yaxis.tick_right()
ax.yaxis.set_label_position('right')

plt.ylabel("")
plt.title("Human Accuracy")
plt.ylim(0, 100.05)   # leave headroom for deltas
plt.axhline(50, color="gray", linestyle="--", alpha=0.7)
plt.legend(title="")

# === Compute and annotate deltas ===
cat_means = (
    agg.groupby(["Category", "Condition"])["Correct"]
    .mean()
    .unstack()
)

for i, category in enumerate(cat_means.index):
    baseline = cat_means.loc[category, "Baseline"]
    twin = cat_means.loc[category, "PerceptTwin"]
    delta = twin - baseline

    # Position text above the taller bar
    max_height = max(baseline, twin)
    ax.text(
        (i + 1)  % 2,
        max_height + 10,  # slightly above bar
        f"Δ {delta:.0f}%",
        ha="center",
        va="bottom",
        color="red",
        fontsize=FONTSIZE,
        fontweight="bold"
    )
plt.xlabel("Question Category")


def save_and_trim_plot(scene_name, dpi=300, padding=0):
    """
    Save matplotlib plot as PNG and trim white borders

    Parameters:
    scene_name (str): Name for the output file (without extension)
    dpi (int): DPI for the saved image
    padding (int): Additional padding to keep around the content
    """

    # Create output directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)

    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)

    # Load the image from buffer
    from PIL import Image
    img = Image.open(buf)

    # Convert to numpy array for processing
    img_array = np.array(img)

    if True:
        # Find non-white pixels (where any channel is not 255)
        if img_array.shape[2] == 4:  # RGBA image
            # Consider alpha channel and RGB channels
            non_white = np.any(img_array[:, :, :3] < 255, axis=2)
        else:  # RGB image
            non_white = np.any(img_array < 250, axis=2)

        # Find bounding box of non-white content
        rows = np.any(non_white, axis=1)
        cols = np.any(non_white, axis=0)

        if np.any(rows) and np.any(cols):
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]

            # Add padding
            ymin = max(0, ymin - padding)
            ymax = min(img_array.shape[0], ymax + padding)
            xmin = max(0, xmin - padding)
            xmax = min(img_array.shape[1], xmax + padding)

            # Crop the image
            cropped_img = img.crop((xmin, ymin, xmax, ymax))
        else:
            cropped_img = img  # No content found, return original
    else:
        cropped_img = img

    # Save the cropped image
    scene_name = scene_name.replace('"', '').replace("'", "`")
    output_path = f'plots/{scene_name}.png'
    cropped_img.save(output_path, 'png', dpi=(dpi, dpi))

    # Close the buffer
    buf.close()

    return output_path

#plt.tight_layout()
save_and_trim_plot("anova", dpi=900)
plt.show()


import matplotlib.pyplot as plt

def plot_anova_table(results_df, title="ANOVA Results"):
    """
    Display ANOVA results as a matplotlib table in its own figure.
    results_df: pandas DataFrame with ANOVA results (e.g., p-unc, ng2, etc.)
    """
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis("off")  # Hide axes

    # Create the table
    table = ax.table(
        cellText=results_df.round(4).values,
        colLabels=results_df.columns,
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.4)  # Adjust size scaling

    # Add a title above the table
    plt.title(title, fontsize=12, pad=20)

    plt.tight_layout()
    plt.show()

anova_results = pd.DataFrame({
    "Source": ["Condition", "Category", "Condition * Category"],
    "p": [0.038482, 0.003299, 0.012441],
    "$\eta^2$g": [0.031825, 0.075078, 0.044512],
})


# --- Assumption checks ---

# 1) Normality test (Shapiro-Wilk per cell of Condition × Category)
print("\n=== Shapiro-Wilk Normality Tests ===")
for (cond, cat), sub in agg.groupby(["Condition", "Category"]):
    print(pg.normality(sub["Correct"]))
    #print(f"{cond}, {cat}: W={stat:.4f}, p={pval:.4f}")

# 2) Mauchly’s Test of Sphericity
# Run on the same factors as rm_anova
print("\n=== Mauchly’s Test of Sphericity ===")
spher = pg.sphericity(
    data=agg,
    dv="Correct",
    within=["Condition", "Category"],
    subject="Participant"
)
print(spher)


#plot_anova_table(anova_results, "ANOVA Results Table")
