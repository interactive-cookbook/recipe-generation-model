import json
import os
from pathlib import Path
from statistics import mean

from inference import generate_data_set, RecipeGenerator
from evaluation import read_file_for_eval, compute_meteor, compute_bleurt, compute_rouge, compute_chrf, compute_bleu, compute_chrf_plus


def run_and_save_evaluation(pred_file, ref_file, context_len, model_name, test_data_set,
                            beam_size, linearization, model_config):
    pred_sentences = read_file_for_eval(pred_file)
    ref_sentences = read_file_for_eval(ref_file)

    task_model_config = model_config.task_specific_params['translation_cond_amr_to_text']

    bleurt_score = compute_bleurt(pred_sentences, ref_sentences)
    data = {
        'Model used': model_name, 'Train dataset': task_model_config["corpus_dir"],
        'Train context length': task_model_config.get("context_len", 0),
        'Train linearization': task_model_config.get("linearization", "penman"),
        'Dropout': model_config.dropout_rate, 'Test dataset': test_data_set, 'Test context length': context_len,
        'Beam size': beam_size, 'Test linearization': linearization,
        'BLEU Score': compute_bleu(pred_sentences, ref_sentences),
        'chrF Score': compute_chrf(pred_sentences, ref_sentences),
        'chrF++ Score': compute_chrf_plus(pred_sentences, ref_sentences),
        'ROUGE score': compute_rouge(pred_sentences, ref_sentences),
        'METEOR score': compute_meteor(pred_sentences, ref_sentences), 'BLEURT scores': bleurt_score["scores"],
        'BLEURT Average': bleurt_score["average"]
    }

    output_file = pred_file[:-4] + '_evaluation.txt'
    with open(output_file, 'w') as f:
        json.dump(data, f)


def pred_and_eval(inference_config_file):

    generator: RecipeGenerator = generate_data_set(inference_config_file)

    with open(inference_config_file) as conf:
        config_args = json.load(conf)
    inf_config = config_args['test_args']
    generator_config = config_args['generator_args']
    context_len = inf_config['context_len']
    model_path = generator_config['model_name_or_path']
    model_checkpoint = generator_config.get('checkpoint', None)
    inf_output_file = inf_config['output_file']
    beam_size = generator.num_beams
    test_data_set = f'{inf_config["corpus_dir"]}/{inf_config["test_path"]}'
    linearization = generator.linearization

    pred_file, ref_file, model_name = get_inference_out_path(context_len, model_path, inf_output_file, model_checkpoint)
    run_and_save_evaluation(pred_file, ref_file, context_len, model_name, test_data_set, beam_size, linearization,
                            generator.model.config)


def get_inference_out_path(context_len, model_path, output_file, model_checkpoint=None) -> tuple:

    model_path = os.path.join(Path(model_path))
    model_name = str(model_path.split(os.sep)[-1])
    if model_checkpoint:
        model_name += f'_{model_checkpoint}'
    output_path = os.path.join(Path('./output'), model_name, f'{context_len}_context')

    prediction_path = os.path.join(output_path, output_file)
    ref_file = output_file[:-4] + '_reference.txt'
    reference_path = os.path.join(output_path, ref_file)

    return prediction_path, reference_path, model_name


if __name__=='__main__':

    inf_dir = './inference_configs'
    for model_dir in os.listdir(inf_dir):
        if model_dir == 'old' or model_dir == 'validation_set':
            continue
        for inf_file in os.listdir(os.path.join(inf_dir, model_dir)):
            inf_file_path = os.path.join(inf_dir, model_dir, inf_file)
            pred_and_eval(inf_file_path)




