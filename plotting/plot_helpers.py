import json
from collections import defaultdict
from typing import Dict, List


def read_eval_output(evaluation_file) -> Dict[str, Dict]:
    """

    :param evaluation_file:
    :return:
    """
    eval_data = dict()

    with open(evaluation_file, 'r', encoding='utf-8') as f:
        current_metric = ''
        for line in f.readlines():
            line = line.strip()

            if current_metric == 'BLEURT Average':
                eval_data[current_metric] = {'score': float(line)}

            # result of a specific metric
            if line.startswith('{'):
                json_line = line.replace("'", '"')
                eval_res = json.loads(json_line)
                assert current_metric
                eval_data[current_metric] = eval_res
                current_metric = ''

            elif line.startswith('['):
                assert current_metric == 'BLEURT scores'
                eval_data[current_metric] = line

            elif line:
                line_list = line.split(': ')
                assert len(line_list) in [1, 2]
                # the name of a metric
                if len(line_list) == 1:
                    current_metric = line_list[0][:-1]
                # name - value pairs of the evaluation set up
                elif len(line_list) == 2:
                    eval_data[line_list[0]] = line_list[1]

    return eval_data


def get_bleu_per_epoch(files_model: list):
    data_to_plot = []
    for f in files_model:
        eval_res = read_eval_output(f)
        model_name = eval_res["Model used"]
        epoch = 5
        if '10ep' in model_name:
            epoch = 10
        elif '30ep' in model_name:
            epoch = 30
        elif '60ep' in model_name:
            epoch = 60
        elif '100ep' in model_name:
            epoch = 100

        bleu_score = eval_res['BLEU Score']['score']
        bleu_score = round(bleu_score, 2)

        data_to_plot.append((epoch, bleu_score))
        data_to_plot.sort()

    return data_to_plot


def get_bleu_per_context(files_model: list):

    data_to_plot = []
    for f in files_model:
        eval_res = read_eval_output(f)
        context = eval_res['Test context length']
        bleu_score = eval_res['BLEU Score']['score']
        bleu_score = round(bleu_score, 2)
        data_to_plot.append((context, bleu_score))
    data_to_plot.sort()

    return data_to_plot


def get_rouge_per_context(files_model: list):

    data_to_plot = defaultdict(list)
    for f in files_model:
        eval_res = read_eval_output(f)
        context = eval_res['Test context length']
        rouge1_score = eval_res['ROUGE score']['rouge1']
        rouge1_score = round(rouge1_score, 4)
        rouge2_score = eval_res['ROUGE score']['rouge2']
        rouge2_score = round(rouge2_score, 4)
        rougel_score = eval_res['ROUGE score']['rougeL']
        rougel_score = round(rougel_score, 4)
        data_to_plot['rouge1'].append((context, rouge1_score))
        data_to_plot['rouge2'].append((context, rouge2_score))
        data_to_plot['rougel'].append((context, rougel_score))

    for key, value in data_to_plot.items():
        value.sort()

    return data_to_plot


def get_meteor_per_context(files_model: list):

    data_to_plot = []
    for f in files_model:
        eval_res = read_eval_output(f)
        context = eval_res['Test context length']
        bleu_score = eval_res['METEOR score']['meteor']
        bleu_score = round(bleu_score, 4)
        data_to_plot.append((context, bleu_score))
    data_to_plot.sort()

    return data_to_plot


def get_bleurt_per_context(files_model: list):

    data_to_plot = []
    for f in files_model:
        eval_res = read_eval_output(f)
        context = eval_res['Test context length']
        bleu_score = eval_res['BLEURT Average']['score']
        bleu_score = round(bleu_score, 4)
        data_to_plot.append((context, bleu_score))
    data_to_plot.sort()

    return data_to_plot

