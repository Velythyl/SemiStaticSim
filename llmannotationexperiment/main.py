import json
import os
import random
import shutil

import numpy as np
from datasets import load_dataset
from collections import defaultdict
from pathlib import Path
from llmqueries.llm import set_api_key

set_api_key(Path("/home/velythyl/Desktop/Holodeck/hippo/secrets/openai_api_key.txt").read_text())
# Load the HICO-DET dataset
dataset = load_dataset("zhimeng/hico_det", cache_dir="/tmp/llmannotationexperiment")
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


def run_exp(gpt_model):
    # Process datasets
    train_actions = process_dataset(dataset["train"], "Processing train set")
    #test_actions = process_dataset(dataset["test"], "Processing test set")

    # Combine results
    HOI_CLASSES = set(train_actions)# + test_actions)#.difference(set("'no_interaction'"))
    TO_REMOVE = ["no_interaction", "move"]  # too common
    for to_remove in TO_REMOVE:
        if to_remove in HOI_CLASSES:
            HOI_CLASSES.remove(to_remove)
    NUM_HOI_CLASSES = len(HOI_CLASSES)

    import re

    def human_labeler(object_name, actions):
        """Prompt a human user to label an object with available actions."""
        # Create the actions table with aligned numbers
        num_actions = len(actions)
        col_width = max(len(action) for action in actions) + 2  # Add padding for spacing

        print(f"\nLabel: {object_name}")

        # Print numbers centered over each action
        number_row = " | ".join(str(i + 1).center(col_width) for i in range(num_actions))
        action_row = " | ".join(action.ljust(col_width) for action in actions)

        print(number_row)
        print(action_row)

        HUMAN_SELECTED = None
        while True:
            user_input = input("Your answer: ").strip()

            # Parse the input (accept comma or any whitespace-separated values)
            if ',' in user_input:
                selected = [s.strip() for s in user_input.split(',')]
            else:
                selected = re.split(r'\s+', user_input)

            # Validate the input
            try:
                selected_indices = [int(s) for s in selected]
                if all(1 <= idx <= num_actions for idx in selected_indices):
                    HUMAN_SELECTED = [actions[idx - 1] for idx in selected_indices]
                    break
                else:
                    print(f"Please enter numbers between 1 and {num_actions}")
            except ValueError:
                print("Please enter numbers only, separated by commas or whitespace")
        print(f"Human selected: {HUMAN_SELECTED}")
        return HUMAN_SELECTED



    def llm_annotate(sample_name, target_actions):
        from llmannotationexperiment.llm_annotation import LLM_annotate


        # Call the real LLM-based annotator
        return LLM_annotate(gpt_model, sample_name, target_actions)


    # Filter samples with at least one target HOI
    def augment_sample_with_num_hoi_targets(sample, target_actions):
        try:
            sample_s_actions = extract_actions_for_sample(sample)

            COUNT = 0
            for target_action in target_actions:
                if target_action in sample_s_actions:
                    COUNT += 1
            sample["NUM_TARGET_HOI"] = COUNT
            return sample
        except:
            sample["NUM_TARGET_HOI"] = 0
            return sample



    # Run the experiment N times
    N = 20
    all_metrics = {"train": {"precision": [], "recall": [], "f1": [], "n_samples": []}}
    all_metrics["anysplit"] = {"precision": [], "recall": [], "f1": [], "n_samples": []}
    tups = []

    NUM_SAMPLES_TO_EVALUATE = 10

    # Evaluate predictions
    def evaluate_predictions(samples, target_actions):
        results = []

        idx_to_do = list(map(int, np.random.choice(np.arange(len(samples)), len(samples))))
        NUM_DONE = 0
        for sample_idx_todo in idx_to_do:
            if NUM_DONE > NUM_SAMPLES_TO_EVALUATE:
                break
            NUM_DONE += 1

            sample = samples[sample_idx_todo]

            all_possible_actions = loadit(sample["positive_captions"]) + loadit(sample["negative_captions"]) + loadit(
                sample["ambiguous_captions"])
            sample_object_name = loadit(sample["positive_captions"])[0][
                0]  # only consider the first object in the thing, the rest are treated as confounders
            gt_hois = [act for objname, act in all_possible_actions if objname == sample_object_name]
            gt_hois = list(filter(lambda x: x in target_actions, gt_hois))

            if len(gt_hois) == 0:
                continue

            # gt_hois = set([hoi["category_id"] for hoi in sample["hoi_annotation"] if hoi["category_id"] in target_action_ids])
            pred_hois = llm_annotate(sample_object_name, target_actions) if gpt_model != "human" else human_labeler(sample_object_name, target_actions)

            CORRECT = 0
            assert len(gt_hois) >= 1
            for pred_act in pred_hois:
                if pred_act in gt_hois:
                    CORRECT += 1
            acc = float(CORRECT) / len(gt_hois)
            results.append((CORRECT, len(pred_hois), len(gt_hois)))
            tups.append((sample_object_name, gt_hois, pred_hois, CORRECT, len(gt_hois), target_actions))

        total_correct = sum(x[0] for x in results)
        total_pred = sum(x[1] for x in results)
        total_gt = sum(x[2] for x in results)

        precision = total_correct / total_pred if total_pred else 0
        recall = total_correct / total_gt if total_gt else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

        return precision, recall, f1, len(samples)

    for run in range(N):
        print(f"\n🔁 Run {run + 1}/{N}")

        # Random target actions
        #target_action_ids = random.sample(range(NUM_HOI_CLASSES), 10)
        target_actions = random.sample(HOI_CLASSES, 10)
        print("🎯 Target HOI IDs:", target_actions)

        for split_name in splits:
            split_name = "train"

            split_data = dataset[split_name]

            # Filter retained samples
            retained = split_data.map(lambda s: augment_sample_with_num_hoi_targets(s, target_actions))#, num_proc=8)
            retained = retained.filter(lambda s: s["NUM_TARGET_HOI"] >= 1)#, num_proc=8)
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
            all_metrics["anysplit"]["precision"].append(precision)
            all_metrics["anysplit"]["recall"].append(recall)
            all_metrics["anysplit"]["f1"].append(f1)
            all_metrics["anysplit"]["n_samples"].append(n_samples)

            break


        shutil.rmtree("/tmp/llmannotationexperiment")   # HF eats so much disk its insane


    report = """
📊 === Summary Across 10 Repeats ===
""".strip()
    # Report average and std deviation
    #print("\n📊 === Summary Across 10 Repeats ===")
    for k, stats in all_metrics.items():
        report += (f"\n📂 Split: {k}")
        for metric in ["precision", "recall", "f1"]:
            values = stats[metric]
            mean = np.mean(values)
            std = np.std(values)
            report += (f"   {metric.capitalize():<10}: {mean:.3f} ± {std:.3f}")

    with open(f"./json_for_{gpt_model}", "w") as f:
        json.dump(all_metrics, f)

    with open(f"./report_for_{gpt_model}", "w") as f:
        f.write(report)

    print(report)

    with open(f"./tups_{gpt_model}.json", "w") as f:
        json.dump(tups, f)

if __name__ == "__main__":

    """
    gpt-3.5-turbo
        "gpt-4.1-nano-2025-04-14": 200000,
    "gpt-4.1-mini-2025-04-14": 200000,
    "gpt-4.1-2025-04-14": 10000,
    """
    def reset_seeds(seed):
        np.random.seed(seed)
        random.seed(seed)
    reset_seeds(0)
    run_exp("human")
    exit(0)
    reset_seeds(0)
    run_exp("gpt-4")
    reset_seeds(0)
    run_exp("gpt-3.5-turbo")
    reset_seeds(0)
    run_exp("gpt-4.1-nano-2025-04-14")
    reset_seeds(0)
    run_exp("gpt-4.1-mini-2025-04-14")
    reset_seeds(0)
    run_exp("gpt-4.1-2025-04-14")