import json
import pandas as pd
from argparse import ArgumentParser
from glob import glob


def expand_nested_json(json_data):
    """
    Function to read the evaluation results that are stored in json data format
    Creates separate entries e.g. for the different ROUGE scores that are part of one json object
    :param json_data:
    :return:
    """
    expanded_keys = []
    new_values = {}
    for key, value in json_data.items():
        if type(value) is dict:
            for sub_key, sub_value in value.items():
                if type(sub_value) is float and sub_value < 1:
                    sub_value = sub_value * 100
                new_values[f'{key}_{sub_key}'] = sub_value
            expanded_keys.append(key)

    for k in expanded_keys:
        json_data.pop(k)

    bleurt_value = json_data['BLEURT Average']

    if bleurt_value:
        json_data.pop('BLEURT Average')
        json_data.update({'BLEURT Average': bleurt_value * 100})
    json_data.update(new_values)


def get_cleansed_json(file_path):
    """
    Reads the evaluation file and returns a json object with all relevant data
    :param file_path:
    :return:
    """
    with open(file_path) as f:
        data = json.load(f)
        data.pop('BLEURT scores')
        expand_nested_json(data)
    return data


def create_evaluation_csv(output_dir_path, csv_path):
    """
    output_dir_path should contain one subfolder per model each with one subfolder per context length
    The function then reads all files ending with "_evaluation.txt" from all this subsubfolders and creates
    a csv file with all evaluation results
    :param output_dir_path: path to the directory with the model outputs;
    :param csv_path: path for the csv file that gets created
    :return:
    """
    # get file paths
    paths = glob(f"{output_dir_path}/*/*_context/*_evaluation.txt")
    if len(paths) == 0:
        print("No evaluation data found.")
        exit()

    # read json and add to list
    eval_results = [
        get_cleansed_json(fp) for fp in paths
    ]

    # transform list of dicts to single dict with lists and create data frame
    df = pd.DataFrame.from_dict({
        column: [tmp[column] for tmp in eval_results] for column in list(eval_results[0].keys())
    })

    # store csv
    df.to_csv(csv_path, sep='|', index=False)


if __name__=="__main__":

    parser = ArgumentParser()
    parser.add_argument("--eval-path", required=True, help="path to the output directory")
    parser.add_argument("--csv-path", required=True, help="path and name for the csv file to create")
    args = parser.parse_args()
    config_file = args.config
    create_evaluation_csv()