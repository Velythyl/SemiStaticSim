import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Load CSV
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


# Filter function
def filter_robotics(df, robotics=True):
    if '3. Have you taken classes in robotics or have professional experience working in robotics?' not in df.columns:
        return df
    if robotics:
        return df[
            df['3. Have you taken classes in robotics or have professional experience working in robotics?'] == 'Yes']
    else:
        return df[
            df['3. Have you taken classes in robotics or have professional experience working in robotics?'] == 'No']


# Define the correct order for answers
ANSWER_ORDER = ["Success", "Failure", "I don't know", "I don't understand the plan or the objects in the scene"]

# Define confusion answers
CONFUSION_ANSWERS = ["I don't know", "I don't understand the plan or the objects in the scene"]
DECISIVE_ANSWERS = ["Success", "Failure"]


# Calculate accuracy and confusion for a given test
def calculate_metrics(df, ab_pair, test_idx):
    ab_col, q_col = ab_pair
    subset = df[[ab_col, q_col]].dropna()

    correct_answer = true_answers[test_idx]

    accuracy = {}
    confusion = {}
    individual_responses = {"A:": [], "B:": []}
    individual_confusion = {"A:": [], "B:": []}

    for group in ["A:", "B:"]:
        group_data = subset[subset[ab_col] == group]

        # Accuracy
        correct_count = (group_data[q_col] == correct_answer).sum()
        accuracy[group] = correct_count / len(group_data) * 100 if len(group_data) > 0 else 0
        individual_responses[group] = (group_data[q_col] == correct_answer).astype(int).tolist()

        # Confusion
        confusion_count = group_data[q_col].isin(CONFUSION_ANSWERS).sum()
        confusion[group] = confusion_count / len(group_data) * 100 if len(group_data) > 0 else 0
        individual_confusion[group] = group_data[q_col].isin(CONFUSION_ANSWERS).astype(int).tolist()

    return accuracy, confusion, individual_responses, individual_confusion


# Perform T-test
def perform_t_test(individual_data_a, individual_data_b):
    if len(individual_data_a) > 1 and len(individual_data_b) > 1:
        t_stat, p_value = stats.ttest_ind(individual_data_a, individual_data_b, equal_var=False)
        return t_stat, p_value
    else:
        return None, None


# Plot with AB hue, percentages, accuracy, and confusion
def plot_abtest(df, ab_pair, title, test_idx):
    ab_col, q_col = ab_pair
    subset = df[[ab_col, q_col]].dropna()
    subset[q_col] = pd.Categorical(subset[q_col], categories=ANSWER_ORDER, ordered=True)

    # Normalize counts to percentages
    counts = subset.groupby([ab_col, q_col]).size().reset_index(name="count")
    totals = counts.groupby(ab_col)["count"].transform("sum")
    counts["percent"] = counts["count"] / totals * 100

    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12),
                                        gridspec_kw={'height_ratios': [3, 1, 1]})

    # Plot 1: Main bar chart
    sns.barplot(data=counts, x=q_col, y="percent", hue=ab_col, order=ANSWER_ORDER, ax=ax1)
    ax1.set_ylim(0, 100)

    # Annotate bars with percentages
    for p in ax1.patches:
        height = p.get_height()
        if height > 0:
            ax1.annotate(f"{height:.1f}%", (p.get_x() + p.get_width() / 2., height),
                         ha='center', va='bottom', fontsize=8)

    # Annotate percentage differences
    for answer in ANSWER_ORDER:
        vals_a = counts[(counts[q_col] == answer) & (counts[ab_col] == "A:")]
        vals_b = counts[(counts[q_col] == answer) & (counts[ab_col] == "B:")]

        if len(vals_a) > 0 and len(vals_b) > 0:
            percent_a = vals_a.iloc[0]["percent"]
            percent_b = vals_b.iloc[0]["percent"]
            diff = percent_a - percent_b

            x_pos = ANSWER_ORDER.index(answer)
            max_height = max(percent_a, percent_b) + 5

            ax1.annotate(f"Δ {-diff:+.1f}%", (x_pos, max_height),
                         ha='center', va='bottom', fontsize=9, color='red', fontweight='bold')

    ax1.set_title(title)
    ax1.set_ylabel("Percentage")
    ax1.set_xlabel("")
    ax1.tick_params(axis='x', rotation=45, labelrotation=45)

    # Calculate metrics
    accuracy, confusion, individual_responses, individual_confusion = calculate_metrics(df, ab_pair, test_idx)

    # Perform T-tests
    t_stat_acc, p_value_acc = perform_t_test(individual_responses["A:"], individual_responses["B:"])
    t_stat_conf, p_value_conf = perform_t_test(individual_confusion["A:"], individual_confusion["B:"])

    # Plot 2: Accuracy bar
    groups = ["A:", "B:"]
    acc_values = [accuracy[group] for group in groups]

    bars = ax2.bar(groups, acc_values, color=['skyblue', 'lightcoral'])
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title(
        f"Accuracy by Group\nT-test p-value: {p_value_acc:.3f}" if p_value_acc is not None else "Accuracy by Group")

    for bar, value in zip(bars, acc_values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{value:.1f}%', ha='center', va='bottom')
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.7)

    # Plot 3: Confusion bar
    conf_values = [confusion[group] for group in groups]

    bars = ax3.bar(groups, conf_values, color=['lightgreen', 'lightyellow'])
    ax3.set_ylim(0, 100)
    ax3.set_ylabel("Confusion (%)")
    ax3.set_title(
        f"Confusion by Group\nT-test p-value: {p_value_conf:.3f}" if p_value_conf is not None else "Confusion by Group")

    for bar, value in zip(bars, conf_values):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{value:.1f}%', ha='center', va='bottom')
    ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.7)

    # Print results to console
    print(f"Test {test_idx} - Results:")
    print(
        f"  Group A (n={len(individual_responses['A:'])}): Accuracy={accuracy['A:']:.1f}%, Confusion={confusion['A:']:.1f}%")
    print(
        f"  Group B (n={len(individual_responses['B:'])}): Accuracy={accuracy['B:']:.1f}%, Confusion={confusion['B:']:.1f}%")

    if p_value_acc is not None:
        print(f"  Accuracy T-statistic: {t_stat_acc:.3f}, p-value: {p_value_acc:.3f}")
        if p_value_acc < 0.05:
            print("  * Statistically significant difference in accuracy (p < 0.05)")

    if p_value_conf is not None:
        print(f"  Confusion T-statistic: {t_stat_conf:.3f}, p-value: {p_value_conf:.3f}")
        if p_value_conf < 0.05:
            print("  * Statistically significant difference in confusion (p < 0.05)")
    print()

    plt.tight_layout()
    plt.show()


# Individual AB tests
for idx, pair in enumerate(ab_pairs, start=1):
    plot_abtest(df, pair, f"AB Test {idx}: Do you think the plan will succeed?", idx)


# Group averages with confusion tracking
def plot_average_with_confusion(df, indices, title):
    dfs = []
    all_individual_responses = {"A:": [], "B:": []}
    all_individual_confusion = {"A:": [], "B:": []}

    for i in indices:
        ab_col, q_col = ab_pairs[i - 1]
        sub = df[[ab_col, q_col]].dropna().copy()
        sub["AB"] = sub[ab_col]
        sub["Answer"] = pd.Categorical(sub[q_col], categories=ANSWER_ORDER, ordered=True)
        dfs.append(sub[["AB", "Answer"]])

        # Collect individual responses for T-test
        _, _, individual_responses, individual_confusion = calculate_metrics(df, (ab_col, q_col), i)
        all_individual_responses["A:"].extend(individual_responses["A:"])
        all_individual_responses["B:"].extend(individual_responses["B:"])
        all_individual_confusion["A:"].extend(individual_confusion["A:"])
        all_individual_confusion["B:"].extend(individual_confusion["B:"])

    merged = pd.concat(dfs)
    counts = merged.groupby(["AB", "Answer"]).size().reset_index(name="count")
    totals = counts.groupby("AB")["count"].transform("sum")
    counts["percent"] = counts["count"] / totals * 100

    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12),
                                        gridspec_kw={'height_ratios': [3, 1, 1]})

    # Plot 1: Main bar chart
    sns.barplot(data=counts, x="Answer", y="percent", hue="AB", order=ANSWER_ORDER, ax=ax1)
    ax1.set_ylim(0, 100)

    for p in ax1.patches:
        height = p.get_height()
        if height > 0:
            ax1.annotate(f"{height:.1f}%", (p.get_x() + p.get_width() / 2., height),
                         ha='center', va='bottom', fontsize=8)

    # Annotate percentage differences
    for answer in ANSWER_ORDER:
        vals_a = counts[(counts["Answer"] == answer) & (counts["AB"] == "A:")]
        vals_b = counts[(counts["Answer"] == answer) & (counts["AB"] == "B:")]

        if len(vals_a) > 0 and len(vals_b) > 0:
            percent_a = vals_a.iloc[0]["percent"]
            percent_b = vals_b.iloc[0]["percent"]
            diff = percent_a - percent_b

            x_pos = ANSWER_ORDER.index(answer)
            max_height = max(percent_a, percent_b) + 5

            ax1.annotate(f"Δ {-diff:+.1f}%", (x_pos, max_height),
                         ha='center', va='bottom', fontsize=9, color='red', fontweight='bold')

    ax1.set_title(title)
    ax1.set_ylabel("Percentage")
    ax1.set_xlabel("")
    ax1.tick_params(axis='x', rotation=45, labelrotation=45)

    # Calculate overall accuracy and confusion
    accuracy_a = sum(all_individual_responses["A:"]) / len(all_individual_responses["A:"]) * 100 if \
    all_individual_responses["A:"] else 0
    accuracy_b = sum(all_individual_responses["B:"]) / len(all_individual_responses["B:"]) * 100 if \
    all_individual_responses["B:"] else 0

    confusion_a = sum(all_individual_confusion["A:"]) / len(all_individual_confusion["A:"]) * 100 if \
    all_individual_confusion["A:"] else 0
    confusion_b = sum(all_individual_confusion["B:"]) / len(all_individual_confusion["B:"]) * 100 if \
    all_individual_confusion["B:"] else 0

    # Perform T-tests
    t_stat_acc, p_value_acc = perform_t_test(all_individual_responses["A:"], all_individual_responses["B:"])
    t_stat_conf, p_value_conf = perform_t_test(all_individual_confusion["A:"], all_individual_confusion["B:"])

    # Plot 2: Accuracy bar
    groups = ["A:", "B:"]
    acc_values = [accuracy_a, accuracy_b]

    bars = ax2.bar(groups, acc_values, color=['skyblue', 'lightcoral'])
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title(
        f"Average Accuracy by Group\nT-test p-value: {p_value_acc:.3f}" if p_value_acc is not None else "Average Accuracy by Group")

    for bar, value in zip(bars, acc_values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{value:.1f}%', ha='center', va='bottom')
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.7)

    # Plot 3: Confusion bar
    conf_values = [confusion_a, confusion_b]

    bars = ax3.bar(groups, conf_values, color=['lightgreen', 'lightyellow'])
    ax3.set_ylim(0, 100)
    ax3.set_ylabel("Confusion (%)")
    ax3.set_title(
        f"Average Confusion by Group\nT-test p-value: {p_value_conf:.3f}" if p_value_conf is not None else "Average Confusion by Group")

    for bar, value in zip(bars, conf_values):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{value:.1f}%', ha='center', va='bottom')
    ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.7)

    # Print results to console
    print(f"{title} - Results:")
    print(
        f"  Group A (n={len(all_individual_responses['A:'])}): Accuracy={accuracy_a:.1f}%, Confusion={confusion_a:.1f}%")
    print(
        f"  Group B (n={len(all_individual_responses['B:'])}): Accuracy={accuracy_b:.1f}%, Confusion={confusion_b:.1f}%")

    if p_value_acc is not None:
        print(f"  Accuracy T-statistic: {t_stat_acc:.3f}, p-value: {p_value_acc:.3f}")
        if p_value_acc < 0.05:
            print("  * Statistically significant difference in accuracy (p < 0.05)")

    if p_value_conf is not None:
        print(f"  Confusion T-statistic: {t_stat_conf:.3f}, p-value: {p_value_conf:.3f}")
        if p_value_conf < 0.05:
            print("  * Statistically significant difference in confusion (p < 0.05)")
    print()

    plt.tight_layout()
    plt.show()


# Run all the average plots with confusion tracking
plot_average_with_confusion(df, [1, 2, 3, 4, 5], "Average of All 5 AB Tests")
plot_average_with_confusion(df, [1, 3], "Average of AB Test 1 & 3")
plot_average_with_confusion(df, [2, 4, 5], "Average of AB Test 2, 4 & 5")

plot_average_with_confusion(filter_robotics(df, robotics=True), [1, 3], "Robotics Only: Average of AB Test 1 & 3")
plot_average_with_confusion(filter_robotics(df, robotics=True), [2, 4, 5], "Robotics Only: Average of AB Test 2, 4 & 5")

plot_average_with_confusion(filter_robotics(df, robotics=False), [1, 3], "Non-Robotics Only: Average of AB Test 1 & 3")
plot_average_with_confusion(filter_robotics(df, robotics=False), [2, 4, 5],
                            "Non-Robotics Only: Average of AB Test 2, 4 & 5")

plot_average_with_confusion(filter_robotics(df, robotics=True), [1, 2, 3, 4, 5],
                            "Robotics Only: Average of All 5 AB Tests")
plot_average_with_confusion(filter_robotics(df, robotics=False), [1, 2, 3, 4, 5],
                            "Non-Robotics Only: Average of All 5 AB Tests")