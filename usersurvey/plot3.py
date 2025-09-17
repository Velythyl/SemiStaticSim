import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.stats import f_oneway
import warnings

warnings.filterwarnings('ignore')

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
        return df  # no filter column
    if robotics:
        return df[
            df['3. Have you taken classes in robotics or have professional experience working in robotics?'] == 'Yes']
    else:
        return df[
            df['3. Have you taken classes in robotics or have professional experience working in robotics?'] == 'No']


# Define the correct order for answers
ANSWER_ORDER = ["Success", "Failure", "I don't know", "I don't understand the plan or the objects in the scene"]


# Calculate accuracy for a given test and return individual responses
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


# Perform ANOVA test for multiple groups
def perform_anova_test(groups_data):
    """
    Perform one-way ANOVA test for multiple groups
    groups_data: dictionary with group names as keys and list of accuracy values as values
    """
    valid_groups = {name: data for name, data in groups_data.items() if len(data) >= 2}

    if len(valid_groups) >= 2:
        f_stat, p_value = f_oneway(*valid_groups.values())
        return f_stat, p_value
    else:
        return None, None


# Function to collect accuracy data for ANOVA
def collect_accuracy_data_for_anova(df, test_indices, group_name):
    """
    Collect accuracy data for specified tests to use in ANOVA
    """
    accuracy_data = []

    for test_idx in test_indices:
        ab_col, q_col = ab_pairs[test_idx - 1]
        subset = df[[ab_col, q_col]].dropna()
        correct_answer = true_answers[test_idx]

        group_data = subset[subset[ab_col] == group_name]
        if len(group_data) > 0:
            # Convert to binary accuracy (1 for correct, 0 for incorrect)
            accuracy_binary = (group_data[q_col] == correct_answer).astype(int).tolist()
            accuracy_data.extend(accuracy_binary)

    return accuracy_data


# Function to perform ANOVA comparison between different test groups
def perform_anova_comparison(df, group_definitions):
    """
    Perform ANOVA comparison between different groups of tests
    group_definitions: dictionary with group names as keys and test indices as values
    Example: {"Group1": [1, 3], "Group2": [2, 4, 5]}
    """
    groups_data = {}

    for group_name, test_indices in group_definitions.items():
        # Collect data for both A and B groups
        groups_data[f"A: {group_name}"] = collect_accuracy_data_for_anova(df, test_indices, "A:")
        groups_data[f"B: {group_name}"] = collect_accuracy_data_for_anova(df, test_indices, "B:")

    # Perform ANOVA
    f_stat, p_value = perform_anova_test(groups_data)

    return groups_data, f_stat, p_value


# Plot with ANOVA results
def plot_with_anova(df, group_definitions, title):
    """
    Create a plot showing accuracy for different groups with ANOVA results
    """
    # Collect data and perform ANOVA
    groups_data, f_stat, p_value = perform_anova_comparison(df, group_definitions)

    # Calculate mean accuracy for each group
    group_means = {}
    group_counts = {}
    for group_name, data in groups_data.items():
        group_means[group_name] = np.mean(data) * 100 if len(data) > 0 else 0
        group_counts[group_name] = len(data)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot bars
    groups = list(group_means.keys())
    means = [group_means[group] for group in groups]
    colors = ['skyblue' if 'A:' in group else 'lightcoral' for group in groups]

    bars = ax.bar(groups, means, color=colors, alpha=0.7)

    # Add value labels on bars
    for i, (bar, mean, count) in enumerate(zip(bars, means, group_counts.values())):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{mean:.1f}% (n={count})', ha='center', va='bottom', fontweight='bold')

    # Customize plot
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'{title}\nANOVA F-statistic: {f_stat:.3f}, p-value: {p_value:.3f}' if f_stat is not None else title)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    # Add horizontal line at 50% for reference
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.7)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.show()

    # Print detailed ANOVA results
    print(f"ANOVA Results for {title}:")
    if f_stat is not None and p_value is not None:
        print(f"  F-statistic: {f_stat:.3f}")
        print(f"  p-value: {p_value:.3f}")
        if p_value < 0.05:
            print("  * Statistically significant difference between groups (p < 0.05)")
        else:
            print("  * No statistically significant difference between groups")
    else:
        print("  * Not enough data to perform ANOVA test")

    print("\nGroup details:")
    for group_name, data in groups_data.items():
        mean_acc = np.mean(data) * 100 if len(data) > 0 else 0
        std_acc = np.std(data) * 100 if len(data) > 0 else 0
        print(f"  {group_name}: {mean_acc:.1f}% ± {std_acc:.1f}% (n={len(data)})")

    print("\n" + "=" * 50 + "\n")


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

# Individual AB tests (unchanged)
for idx, pair in enumerate(ab_pairs, start=1):
    plot_abtest(df, pair, f"AB Test {idx}: Do you think the plan will succeed?", idx)


# Group averages (unchanged)

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


# ... (unchanged code from your original function)

# Average of all 5 tests
plot_average(df, [1, 2, 3, 4, 5], "Average of All 5 AB Tests")

# Average of groups
plot_average(df, [1, 3], "Average of AB Test 1 & 3")
plot_average(df, [2, 4, 5], "Average of AB Test 2, 4 & 5")

# ANOVA comparisons for the specific groups you requested
print("ANOVA COMPARISON RESULTS")
print("=" * 50)

# 1. Overall average comparison
group_definitions = {
    "Overall": [1, 2, 3, 4, 5]
}
plot_with_anova(df, group_definitions, "Overall Average: A vs B")

# 2. Average of tests 1 & 3 vs tests 2,4,5
group_definitions = {
    "Tests 1&3": [1, 3],
    "Tests 2,4,5": [2, 4, 5]
}
plot_with_anova(df, group_definitions, "Comparison: Tests 1&3 vs Tests 2,4,5")

# 3. Separate comparison for A and B groups across test types
group_definitions = {
    "A: Tests 1&3": [1, 3],
    "A: Tests 2,4,5": [2, 4, 5],
    "B: Tests 1&3": [1, 3],
    "B: Tests 2,4,5": [2, 4, 5]
}
plot_with_anova(df, group_definitions, "Detailed Comparison: A vs B across Test Types")

# 4. Robotics vs Non-Robotics for the specific test groups
robotics_df = filter_robotics(df, robotics=True)
non_robotics_df = filter_robotics(df, robotics=False)

if len(robotics_df) > 0:
    group_definitions = {
        "Robotics: Tests 1&3": [1, 3],
        "Robotics: Tests 2,4,5": [2, 4, 5]
    }
    plot_with_anova(robotics_df, group_definitions, "Robotics Only: Tests 1&3 vs Tests 2,4,5")

if len(non_robotics_df) > 0:
    group_definitions = {
        "Non-Robotics: Tests 1&3": [1, 3],
        "Non-Robotics: Tests 2,4,5": [2, 4, 5]
    }
    plot_with_anova(non_robotics_df, group_definitions, "Non-Robotics Only: Tests 1&3 vs Tests 2,4,5")


# Post-hoc analysis if ANOVA shows significant differences
def perform_posthoc_analysis(groups_data):
    """
    Perform post-hoc Tukey HSD test if ANOVA shows significant differences
    """
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    # Prepare data for Tukey test
    all_data = []
    group_labels = []

    for group_name, data in groups_data.items():
        if len(data) >= 2:  # Only include groups with sufficient data
            all_data.extend(data)
            group_labels.extend([group_name] * len(data))

    if len(set(group_labels)) >= 2:  # Need at least 2 groups
        tukey = pairwise_tukeyhsd(endog=all_data, groups=group_labels, alpha=0.05)
        return tukey
    return None


# Additional detailed analysis for the main comparison
print("DETAILED POST-HOC ANALYSIS")
print("=" * 50)

# Collect data for the main comparison
main_groups_data, f_stat, p_value = perform_anova_comparison(df, {
    "Tests 1&3": [1, 3],
    "Tests 2,4,5": [2, 4, 5]
})

if f_stat is not None and p_value < 0.05:
    print("Significant difference found in main comparison. Performing post-hoc analysis...")
    tukey_results = perform_posthoc_analysis(main_groups_data)
    if tukey_results:
        print(tukey_results)
else:
    print("No significant difference found in main comparison. No post-hoc analysis needed.")

print("\nANALYSIS COMPLETE")

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
plot_anova_table()