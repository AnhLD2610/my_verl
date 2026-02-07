# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Preprocess combined dataset: All DAPO-17k + Level 4&5 from MATH-lighteval
"""

import argparse
import json
import os
import datasets
from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None,
                        help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir",
        default="~/data/math_combined",
        help="The save directory for the preprocessed dataset."
    )
    args = parser.parse_args()

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # Load DAPO-17k dataset
    print("Loading DAPO-17k dataset...", flush=True)
    dapo_dataset = datasets.load_dataset("open-r1/DAPO-Math-17k-Processed")
    
    # Load MATH-lighteval dataset
    print("Loading MATH-lighteval dataset...", flush=True)
    math_data_source = "DigitalLearningGmbH/MATH-lighteval"
    
    if args.local_dataset_path is not None:
        math_dataset = datasets.load_dataset(args.local_dataset_path)
    else:
        math_dataset = datasets.load_dataset(math_data_source)

    # Process DAPO dataset
    def process_dapo(example, idx):
        question = example["prompt"] + " " + instruction_following
        solution = example["solution"]
        
        data = {
            "data_source": math_data_source,  # Use MATH as data_source
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {
                "split": "train",
                "index": idx,
                "original_source": "DAPO-17k"
            },
        }
        return data

    # Process MATH dataset (only level 4 and 5)
    def process_math(example, idx, split):
        # Filter by level
        level = example.get("level", "")
        if level not in ["Level 4", "Level 5"]:
            return None
            
        question = example["problem"] + " " + instruction_following
        answer = example["solution"]
        solution = extract_solution(answer)
        
        data = {
            "data_source": math_data_source,
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {
                "split": split,
                "index": idx,
                "level": level,
                "original_source": "MATH-lighteval"
            },
        }
        return data

    # Process all DAPO samples
    print("Processing DAPO-17k samples...", flush=True)
    dapo_processed = dapo_dataset["train"].map(
        function=process_dapo,
        with_indices=True,
        remove_columns=dapo_dataset["train"].column_names
    )

    # Process MATH train samples (level 4 and 5 only)
    print("Processing MATH train samples (Level 4 & 5)...", flush=True)
    math_train_processed = math_dataset["train"].map(
        function=lambda x, idx: process_math(x, idx, "train"),
        with_indices=True,
        remove_columns=math_dataset["train"].column_names
    )
    # Filter out None values (samples that are not level 4 or 5)
    math_train_processed = math_train_processed.filter(lambda x: x is not None)

    # Process MATH test samples (level 4 and 5 only)
    print("Processing MATH test samples (Level 4 & 5)...", flush=True)
    math_test_processed = math_dataset["test"].map(
        function=lambda x, idx: process_math(x, idx, "test"),
        with_indices=True,
        remove_columns=math_dataset["test"].column_names
    )
    # Filter out None values
    math_test_processed = math_test_processed.filter(lambda x: x is not None)

    # Combine datasets
    print("Combining datasets...", flush=True)
    # Combine DAPO + MATH train (level 4&5)
    combined_train = datasets.concatenate_datasets([dapo_processed, math_train_processed])
    # Test set is only MATH level 4&5
    combined_test = math_test_processed

    print(f"Total train samples: {len(combined_train)}")
    print(f"  - DAPO-17k: {len(dapo_processed)}")
    print(f"  - MATH Level 4&5 (train): {len(math_train_processed)}")
    print(f"Total test samples (MATH Level 4&5): {len(combined_test)}")

    # Save to parquet
    local_save_dir = args.local_dir if args.local_dir is not None else args.local_save_dir
    local_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_dir, exist_ok=True)

    print(f"Saving to {local_dir}...", flush=True)
    combined_train.to_parquet(os.path.join(local_dir, "train.parquet"))
    combined_test.to_parquet(os.path.join(local_dir, "test.parquet"))

    # Save examples as JSON for reference
    if len(combined_train) > 0:
        example = combined_train[0]
        with open(os.path.join(local_dir, "train_example.json"), "w") as f:
            json.dump(example, f, indent=2)
    
    if len(combined_test) > 0:
        example = combined_test[0]
        with open(os.path.join(local_dir, "test_example.json"), "w") as f:
            json.dump(example, f, indent=2)

    # Save dataset statistics
    stats = {
        "total_train": len(combined_train),
        "dapo_samples": len(dapo_processed),
        "math_level_4_5_train": len(math_train_processed),
        "total_test": len(combined_test),
        "math_level_4_5_test": len(math_test_processed),
    }
    with open(os.path.join(local_dir, "dataset_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    
    print("Dataset statistics:", stats)

    # Copy to HDFS if specified
    hdfs_dir = args.hdfs_dir
    if hdfs_dir is not None:
        print(f"Copying to HDFS: {hdfs_dir}", flush=True)
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
    
    print("Done!", flush=True)