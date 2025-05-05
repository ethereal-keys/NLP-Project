"""
localize: python agentless/fl/localize.py --file_level --output_folder results/swe-bench-lite/file_level --num_threads 1 --target_id astropy__astropy-14995

"""

import json
import os

from datasets import load_dataset
import pandas as pd

def main():
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)

    dataset = load_dataset("princeton-nlp/SWE-bench_Lite")["test"]
    df = dataset.to_pandas()

    for ind, row in df.iterrows():
        json_str = row.to_json()
        json_dict = json.loads(json_str)
        instance_id = json_dict["instance_id"]
        dst = os.path.join(data_dir, f"{instance_id}.json")
        with open(dst, "w") as f:
            json.dump(json_dict, f, indent=4)
            print(f"Saved {instance_id}.json")
    return

if __name__ == "__main__":
    main()
