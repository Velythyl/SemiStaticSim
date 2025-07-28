import os
import json
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Global font size
FONT_SIZE = 24  # Change this value to adjust font sizes globally

# Directory containing files
directory = './exp1'

# Variant priority for sorting
variant_priority = {
    "nano": 0,
    "mini": 1,
    "base": 2,
    "": 3,
    "large": 4
}

def extract_sort_key(filename):
    match = re.search(r'gpt-(\d+(?:\.\d+)?)(?:-(\w+))?', filename)
    if not match:
        return (float('inf'), float('inf'))
    version = float(match.group(1))
    variant = match.group(2) or ""
    return (version, variant_priority.get(variant, float('inf')))

def prettify_filename(filename):
    match = re.search(r'gpt-(\d+(?:\.\d+)?)(?:-(\w+))?', filename)
    if not match:
        return filename
    version = match.group(1)
    variant = match.group(2)

    if variant not in variant_priority:
        variant = ""

    if variant:
        return f"{version} {variant.capitalize()}".strip()
    else:
        return f"{version}".strip()

# Load and process data
results = []

for filename in os.listdir(directory):
    if filename.startswith('json_for_gpt'):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)
        recall = np.array(data["anysplit"]["recall"])
        results.append({
            "filename": filename,
            "pretty_name": prettify_filename(filename),
            "mean": np.mean(recall),
            "median": np.median(recall),
            "std": np.std(recall),
            "sort_key": extract_sort_key(filename)
        })

# Sort results
results.sort(key=lambda x: x["sort_key"])
df = pd.DataFrame(results)

# Plot using seaborn with colorblind palette
plt.figure(figsize=(8, 8))
sns.set(style="white", palette="colorblind")

ax = sns.barplot(data=df, x="pretty_name", y="mean", hue="pretty_name", dodge=False)

# Add clipped error bars
x_coords = range(len(df))
y = df["mean"].to_numpy()
yerr = df["std"].to_numpy()
lower = np.clip(y - yerr, a_min=0, a_max=None)
upper = y + yerr
asymmetric_error = [y - lower, upper - y]

ax.errorbar(
    x=x_coords,
    y=y,
    yerr=asymmetric_error,
    fmt='none',
    c='black',
    capsize=5
)

# Tweak labels and fonts
ax.set_xticks(x_coords)
ax.set_xticklabels(df["pretty_name"], rotation=0, ha='center', fontsize=FONT_SIZE)
ax.set_ylabel("Accuracy $\pm$ STD", fontsize=FONT_SIZE)
ax.set_xlabel("GPT Version", fontsize=FONT_SIZE)
ax.set_title("Skill Prediction On HICO-DET", fontsize=FONT_SIZE)
ax.tick_params(axis='both', labelsize=FONT_SIZE)

plt.tight_layout()
plt.savefig("accuracy_per_gpt_model.pdf", format='pdf')
plt.show()
