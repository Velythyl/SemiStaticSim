import json
import random
import numpy as np
from datasets import load_dataset
from collections import defaultdict
from pathlib import Path
from llmqueries.llm import set_api_key

set_api_key(Path("/home/charlie/Desktop/Holodeck/hippo/secrets/openai_api_key.txt").read_text())
# Load the HICO-DET dataset
dataset = load_dataset("zhimeng/hico_det", cache_dir="/tmp")
splits = dataset.keys()
print("Available splits:", list(splits))

def loadit(actlist):
    return ast.literal_eval(actlist)


from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import ast


def loadit(actlist):
    return ast.literal_eval(actlist)

def extract_actions_for_sample(sample):
    positive_captions_str = loadit(sample["positive_captions"])
    pos = [act for _, act in positive_captions_str]

    negative_captions_str = loadit(sample["negative_captions"])
    neg = [act for _, act in negative_captions_str]

    amb_captions_str = loadit(sample["ambiguous_captions"])
    amb = [act for _, act in amb_captions_str]
    return pos + neg + amb


def extract_actions_for_sample_idx(dataset_split, idx):
    sample = dataset_split[idx]
    return extract_actions_for_sample(sample)

from diskcache import FanoutCache, Cache
CACHEPATH = "/".join(__file__.split("/")[:-1]) + "/diskcache"
cache = Cache(CACHEPATH)

@cache.memoize()
def process_dataset(dataset_split, desc, max_workers=4):
    actions = []
    total_samples = len(dataset_split)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks with indices instead of samples
        futures = {
            executor.submit(extract_actions_for_sample_idx, dataset_split, i): i
            for i in range(total_samples)
        }

        # Process results as they complete
        for future in tqdm(as_completed(futures), total=total_samples, desc=desc):
            try:
                actions.extend(future.result())
            except Exception as e:
                print(f"Error processing sample {futures[future]}: {str(e)}")

    return actions


# Process datasets
train_actions = process_dataset(dataset["train"], "Processing train set")
test_actions = process_dataset(dataset["test"], "Processing test set")

# Combine results
HOI_CLASSES = set(train_actions + test_actions)#.difference(set("'no_interaction'"))
HOI_CLASSES.remove("no_interaction")
NUM_HOI_CLASSES = len(HOI_CLASSES)


def llm_annotate(sample_name, target_actions):
    from llmannotationexperiment.llm_annotation import LLM_annotate


    # Call the real LLM-based annotator
    return LLM_annotate("gpt-4.1-2025-04-14", sample_name, target_actions)


# Filter samples with at least one target HOI
def augment_sample_with_num_hoi_targets(sample, target_actions):
    sample_s_actions = extract_actions_for_sample(sample)

    COUNT = 0
    for target_action in target_actions:
        if target_action in sample_s_actions:
            COUNT += 1
    sample["NUM_TARGET_HOI"] = COUNT
    return sample

NUM_SAMPLES_TO_EVALUATE = 100
# Evaluate predictions
def evaluate_predictions(samples, target_actions):
    results = []

    idx_to_do = list(map(int, np.random.choice(np.arange(len(samples)), NUM_SAMPLES_TO_EVALUATE)))
    for sample_idx_todo in idx_to_do:

        sample = samples[sample_idx_todo]

        all_possible_actions = loadit(sample["positive_captions"]) + loadit(sample["negative_captions"]) + loadit(sample["ambiguous_captions"])
        sample_object_name = loadit(sample["positive_captions"])[0][0] # only consider the first object in the thing, the rest are treated as confounders
        gt_hois = [act for objname, act in all_possible_actions if objname == sample_object_name]
        gt_hois = list(filter(lambda x: x in target_actions, gt_hois))

        #gt_hois = set([hoi["category_id"] for hoi in sample["hoi_annotation"] if hoi["category_id"] in target_action_ids])
        pred_hois = llm_annotate(sample_object_name, target_actions)

        CORRECT = 0
        assert len(gt_hois) >= 1
        for pred_act in pred_hois:
            if pred_act in gt_hois:
                CORRECT += 1
        acc = float(CORRECT) / len(gt_hois)
        results.append((CORRECT, len(pred_hois), len(gt_hois)))

    total_correct = sum(x[0] for x in results)
    total_pred = sum(x[1] for x in results)
    total_gt = sum(x[2] for x in results)

    precision = total_correct / total_pred if total_pred else 0
    recall = total_correct / total_gt if total_gt else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return precision, recall, f1, len(samples)

# Run the experiment N times
N = 10
all_metrics = {split: {"precision": [], "recall": [], "f1": [], "n_samples": []} for split in splits}

for run in range(N):
    print(f"\n🔁 Run {run + 1}/{N}")

    # Random target actions
    #target_action_ids = random.sample(range(NUM_HOI_CLASSES), 10)
    target_actions = random.sample(HOI_CLASSES, 10)
    print("🎯 Target HOI IDs:", target_actions)

    for split_name in splits:
        split_data = dataset[split_name]

        # Filter retained samples
        retained = split_data.map(lambda s: augment_sample_with_num_hoi_targets(s, target_actions), num_proc=8)
        retained = retained.filter(lambda s: s["NUM_TARGET_HOI"] >= 1, num_proc=8)
        print(len(retained))
        if len(retained) == 0:
            print(f"⚠️  No samples with target actions in split: {split_name}")
            continue

        # Evaluate
        precision, recall, f1, n_samples = evaluate_predictions(retained, target_actions)

        all_metrics[split_name]["precision"].append(precision)
        all_metrics[split_name]["recall"].append(recall)
        all_metrics[split_name]["f1"].append(f1)
        all_metrics[split_name]["n_samples"].append(n_samples)

# Report average and std deviation
print("\n📊 === Summary Across 10 Repeats ===")
for split in splits:
    if not all_metrics[split]["precision"]:
        continue

    print(f"\n📂 Split: {split}")
    for metric in ["precision", "recall", "f1"]:
        values = all_metrics[split][metric]
        mean = np.mean(values)
        std = np.std(values)
        print(f"   {metric.capitalize():<10}: {mean:.3f} ± {std:.3f}")
