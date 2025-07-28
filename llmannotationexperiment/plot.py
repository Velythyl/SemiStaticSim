# todo open all files in `./exp1` of format "json_for_gpt*"
# load as json
# get file["anysplit"]["recall"]
# you can get the mean and error bar using
# mean = np.mean(values)
# std = np.std(values)
# plot the results using matplotli
#
# b
import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt

# Directory containing files
directory = './exp1'

# Priority order for variants
variant_priority = {
    "nano": 0,
    "mini": 1,
    "base": 2,
    "": 3,
    "large": 4
}

def extract_sort_key(filename):
    """
    Extract GPT version and variant from filename.
    E.g., "json_for_gpt-3.5-mini.json" -> (3.5, 1)
    """
    match = re.search(r'gpt-(\d+(?:\.\d+)?)(?:-(\w+))?', filename)
    if not match:
        return (float('inf'), float('inf'))  # Push unknowns to the end
    version = float(match.group(1))
    variant = match.group(2) or ""
    return (version, variant_priority.get(variant, float('inf')))

# Collect info
results = []

for filename in os.listdir(directory):
    if filename.startswith('json_for_gpt'):# and filename.endswith('.json'):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)
        recall = np.array(data["anysplit"]["recall"])
        mean = np.mean(recall)
        std = np.std(recall)
        results.append((filename, mean, std))

# Sort by GPT version and variant
results.sort(key=lambda x: extract_sort_key(x[0]))

# Prepare for plotting
labels = [r[0] for r in results]
means = [r[1] for r in results]
stds = [r[2] for r in results]

# Plot
x = np.arange(len(labels))
plt.figure(figsize=(12, 6))
plt.bar(x, means, yerr=stds, capsize=5, color='skyblue')
plt.xticks(x, labels, rotation=45, ha='right')
plt.ylabel("Recall")
plt.title("Recall per GPT Model (Sorted by Version and Variant)")
plt.tight_layout()
plt.show()
