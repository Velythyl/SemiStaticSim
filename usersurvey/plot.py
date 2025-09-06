import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats  # Import scipy for T-test

# Load CSV (your .xslt export saved as CSV)
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
        return df  # no filter column
    if robotics:
        return df[
            df['3. Have you taken classes in robotics or have professional experience working in robotics?'] == 'Yes']
    else:
        return df[
            df['3. Have you taken classes in robotics or have professional experience working in robotics?'] == 'No']


# Define the correct order for answers
ANSWER_ORDER = ["Success", "Failure", "I don't know", "I don't understand the plan or the objects in the scene"]


# Calculate accuracy for a given test and return individual responses for T-test
def calculate_accuracy(df, ab_pair, test_idx):
    ab_col, q_col = ab_pair
    subset = df[[ab_col, q_col]].dropna()

    # Get the correct answer for this test
    correct_answer = true_answers[test_idx]

    # Calculate accuracy for each group (A and B)
    accuracy = {}
    individual_responses = {"A:": [], "B:": []}

    for group in ["A:", "B:"]:
        group_data = subset[subset[ab_col] == group]
        correct_count = (group_data[q_col] == correct_answer).sum()
        accuracy[group] = correct_count / len(group_data) * 100 if len(group_data) > 0 else 0

        # Store individual responses (1 for correct, 0 for incorrect)
        individual_responses[group] = (group_data[q_col] == correct_answer).astype(int).tolist()

    return accuracy, individual_responses


# Perform T-test on accuracy
def perform_t_test(individual_responses):
    group_a = individual_responses["A:"]
    group_b = individual_responses["B:"]

    if len(group_a) > 1 and len(group_b) > 1:  # Need at least 2 samples per group for T-test
        t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)  # Welch's t-test
        return t_stat, p_value
    else:
        return None, None


# Plot with AB hue and percentages
def plot_abtest(df, ab_pair, title, test_idx):
    ab_col, q_col = ab_pair
    subset = df[[ab_col, q_col]].dropna()

    # Ensure answers are in the correct order
    subset[q_col] = pd.Categorical(subset[q_col], categories=ANSWER_ORDER, ordered=True)

    # Normalize counts to percentages
    counts = subset.groupby([ab_col, q_col]).size().reset_index(name="count")
    totals = counts.groupby(ab_col)["count"].transform("sum")
    counts["percent"] = counts["count"] / totals * 100

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                   gridspec_kw={'height_ratios': [3, 1]})

    # Plot the main bar chart
    sns.barplot(data=counts, x=q_col, y="percent", hue=ab_col, order=ANSWER_ORDER, ax=ax1)

    # Set y-axis to always go from 0 to 100
    ax1.set_ylim(0, 100)

    # Annotate bars with percentages
    for p in ax1.patches:
        height = p.get_height()
        if height > 0:  # Only annotate if height is positive
            ax1.annotate(f"{height:.1f}%", (p.get_x() + p.get_width() / 2., height),
                         ha='center', va='bottom', fontsize=8)

    # Annotate percentage difference between A and B for each answer
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

    # Calculate accuracy for the accuracy bar
    accuracy, individual_responses = calculate_accuracy(df, ab_pair, test_idx)

    # Perform T-test
    t_stat, p_value = perform_t_test(individual_responses)

    # Plot accuracy bar
    groups = ["A:", "B:"]
    acc_values = [accuracy[group] for group in groups]

    bars = ax2.bar(groups, acc_values, color=['skyblue', 'lightcoral'])
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title(f"Accuracy by Group\nT-test p-value: {p_value:.3f}" if p_value is not None else "Accuracy by Group")

    # Add value labels on top of bars
    for bar, value in zip(bars, acc_values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{value:.1f}%', ha='center', va='bottom')

    # Add a horizontal line at 50% for reference
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.7)

    # Print T-test results to console
    if p_value is not None:
        print(f"Test {test_idx} - T-test results:")
        print(f"  Group A (n={len(individual_responses['A:'])}): {accuracy['A:']:.1f}%")
        print(f"  Group B (n={len(individual_responses['B:'])}): {accuracy['B:']:.1f}%")
        print(f"  T-statistic: {t_stat:.3f}, p-value: {p_value:.3f}")
        if p_value < 0.05:
            print("  * Statistically significant difference (p < 0.05)")
        else:
            print("  * No statistically significant difference")
        print()

    plt.tight_layout()
    plt.show()


# Individual AB tests
for idx, pair in enumerate(ab_pairs, start=1):
    plot_abtest(df, pair, f"AB Test {idx}: Do you think the plan will succeed?", idx)


# Group averages
def plot_average(df, indices, title):
    dfs = []
    all_individual_responses = {"A:": [], "B:": []}

    for i in indices:
        ab_col, q_col = ab_pairs[i - 1]
        sub = df[[ab_col, q_col]].dropna().copy()
        sub["AB"] = sub[ab_col]  # unify hue column name
        sub["Answer"] = pd.Categorical(sub[q_col], categories=ANSWER_ORDER, ordered=True)
        dfs.append(sub[["AB", "Answer"]])

        # Collect individual responses for T-test
        _, individual_responses = calculate_accuracy(df, (ab_col, q_col), i)
        all_individual_responses["A:"].extend(individual_responses["A:"])
        all_individual_responses["B:"].extend(individual_responses["B:"])

    merged = pd.concat(dfs)

    counts = merged.groupby(["AB", "Answer"]).size().reset_index(name="count")
    totals = counts.groupby("AB")["count"].transform("sum")
    counts["percent"] = counts["count"] / totals * 100

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                   gridspec_kw={'height_ratios': [3, 1]})

    # Plot the main bar chart
    sns.barplot(data=counts, x="Answer", y="percent", hue="AB", order=ANSWER_ORDER, ax=ax1)

    # Set y-axis to always go from 0 to 100
    ax1.set_ylim(0, 100)

    # Annotate bars with percentages
    for p in ax1.patches:
        height = p.get_height()
        if height > 0:  # Only annotate if height is positive
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

    # Calculate overall accuracy for the average
    accuracy_a = 0
    accuracy_b = 0
    count_a = 0
    count_b = 0

    for i in indices:
        ab_col, q_col = ab_pairs[i - 1]
        subset = df[[ab_col, q_col]].dropna()
        correct_answer = true_answers[i]

        for group in ["A:", "B:"]:
            group_data = subset[subset[ab_col] == group]
            correct_count = (group_data[q_col] == correct_answer).sum()

            if group == "A:":
                accuracy_a += correct_count
                count_a += len(group_data)
            else:
                accuracy_b += correct_count
                count_b += len(group_data)

    # Calculate average accuracy
    avg_acc_a = accuracy_a / count_a * 100 if count_a > 0 else 0
    avg_acc_b = accuracy_b / count_b * 100 if count_b > 0 else 0

    # Perform T-test on combined data
    t_stat, p_value = perform_t_test(all_individual_responses)

    # Plot accuracy bar
    groups = ["A:", "B:"]
    acc_values = [avg_acc_a, avg_acc_b]

    bars = ax2.bar(groups, acc_values, color=['skyblue', 'lightcoral'])
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title(
        f"Average Accuracy by Group\nT-test p-value: {p_value:.3f}" if p_value is not None else "Average Accuracy by Group")

    # Add value labels on top of bars
    for bar, value in zip(bars, acc_values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{value:.1f}%', ha='center', va='bottom')

    # Add a horizontal line at 50% for reference
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.7)

    # Print T-test results to console
    if p_value is not None:
        print(f"{title} - T-test results:")
        print(f"  Group A (n={len(all_individual_responses['A:'])}): {avg_acc_a:.1f}%")
        print(f"  Group B (n={len(all_individual_responses['B:'])}): {avg_acc_b:.1f}%")
        print(f"  T-statistic: {t_stat:.3f}, p-value: {p_value:.3f}")
        if p_value < 0.05:
            print("  * Statistically significant difference (p < 0.05)")
        else:
            print("  * No statistically significant difference")
        print()

    plt.tight_layout()
    plt.show()


# Average of all 5 tests
plot_average(df, [1, 2, 3, 4, 5], "Average of All 5 AB Tests")

# Average of groups
plot_average(df, [1, 3], "Average of AB Test 1 & 3")
plot_average(df, [2, 4, 5], "Average of AB Test 2, 4 & 5")

plot_average(filter_robotics(df, robotics=True), [1, 3], "Robotics Only: Average of AB Test 1 & 3")
plot_average(filter_robotics(df, robotics=True), [2, 4, 5], "Robotics Only: Average of AB Test 2, 4 & 5")

plot_average(filter_robotics(df, robotics=False), [1, 3], "Non-Robotics Only: Average of AB Test 1 & 3")
plot_average(filter_robotics(df, robotics=False), [2, 4, 5], "Non-Robotics Only: Average of AB Test 2, 4 & 5")

# Example robotics filter
robotics_df = filter_robotics(df, robotics=True)
plot_average(robotics_df, [1, 2, 3, 4, 5], "Robotics Only: Average of All 5 AB Tests")

robotics_df = filter_robotics(df, robotics=False)
plot_average(robotics_df, [1, 2, 3, 4, 5], "Non-Robotics Only: Average of All 5 AB Tests")