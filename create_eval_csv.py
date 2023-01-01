import json
import pandas as pd
from glob import glob


def expand_nested_json(json_data):
    expanded_keys = []
    new_values = {}
    for key, value in json_data.items():
        if type(value) is dict:
            for sub_key, sub_value in value.items():
                new_values[f'{key}_{sub_key}'] = sub_value
            expanded_keys.append(key)
    for k in expanded_keys:
        json_data.pop(k)
    json_data.update(new_values)


def get_cleansed_json(fp):
    with open(fp) as f:
        data = json.load(f)
        data.pop('BLEURT scores')
        expand_nested_json(data)
    return data


output_dir_path = "./output"
csv_path = "./automatic_eval_results.csv"

# get file paths
fps = glob(f"{output_dir_path}/*/*_context/*_evaluation.txt")
if len(fps) == 0:
    print("No evaluation data found.")
    exit()

# read json and add to list
eval_results = [
    get_cleansed_json(fp) for fp in fps
]

# transform list of dicts to single dict with lists and create data frame
df = pd.DataFrame.from_dict({
    column: [tmp[column] for tmp in eval_results] for column in list(eval_results[0].keys())
})

# store csv
df.to_csv(csv_path, sep='|', index=False)
