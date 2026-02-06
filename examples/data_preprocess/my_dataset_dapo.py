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
Preprocess the GSM8k dataset to parquet format
"""

import argparse
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/gsm8k", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()
    # data_path = "BytedTsinghua-SIA/DAPO-Math-17k"
    data_path_1 = "open-r1/DAPO-Math-17k-Processed"
    data_source = "DigitalLearningGmbH/MATH-lighteval"
    if local_dataset_path is not None:
        dataset = datasets.load_dataset(data_path_1, "default")
    else:
        dataset = datasets.load_dataset(data_path_1, "default")

    train_dataset = dataset["train"]

    test_dataset = dataset["test"]

    instruction_following = '''You have TWO tasks to complete:

TASK 1 - REWRITE THE QUESTION:
Rewrite the given question with MAXIMUM DIVERSITY in wording and structure while keeping all numbers exactly the same. Be creative - use different:
- Sentence structures (questions, statements, conditional phrases)
- Vocabulary and expressions
- Lengths (much shorter or much longer)
- Perspectives or framings
The more different from the original, the better. Just keep all numerical values identical.

Place your rewrite between [REWRITE_START] and [REWRITE_END] tokens.

TASK 2 - SOLVE THE PROBLEM:
Let's think step by step and output the final answer within \\boxed{}.

OUTPUT FORMAT:
[REWRITE_START]
[Your creative rewrite here]
[REWRITE_END]
Your thinking and solution:

Let's think step by step and output the final answer within \\boxed{}.

'''

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("prompt")


            # mai check lại cái này 
            question = question_raw + " " + instruction_following

            answer_raw = example.pop("solution")
            solution = answer_raw
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_save_dir, dst=hdfs_dir)