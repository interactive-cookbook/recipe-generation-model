import json
import os
from pathlib import Path
from statistics import mean

from inference import generate_data_set, RecipeGenerator
from evaluation import read_file_for_eval, compute_meteor, compute_bleurt, compute_rouge, compute_chrf, compute_bleu, compute_chrf_plus


def run_and_save_evaluation(pred_file, ref_file, context_len, model_name, test_data_set,
                            beam_size, linearization):

    pred_sentences = read_file_for_eval(pred_file)
    ref_sentences = read_file_for_eval(ref_file)

    output_file = pred_file[:-4] + '_evaluation.txt'
    with open(output_file, "w", encoding="utf-8") as out:
        out.write(f'Model used: {model_name}\n')
        out.write(f'Test dataset: {test_data_set}\n')
        out.write(f'Test context length: {context_len}\n')
        out.write(f'Beam size: {beam_size}\n')
        out.write(f'Graph linearization: {linearization}\n\n')

        out.write('BLEU Score:\n')
        out.write(f'{compute_bleu(pred_sentences, ref_sentences)}\n')
        out.write('chrF Score:\n')
        out.write(f'{compute_chrf(pred_sentences, ref_sentences)}\n')
        out.write('chrF++ Score:\n')
        out.write(f'{compute_chrf_plus(pred_sentences, ref_sentences)}\n')
        out.write('ROUGE score: \n')
        out.write(f'{compute_rouge(pred_sentences, ref_sentences)}\n')
        out.write('METEOR score: \n')
        out.write(f'{compute_meteor(pred_sentences, ref_sentences)}\n')
        out.write('BLEURT scores: \n')
        bleurt_score = compute_bleurt(pred_sentences, ref_sentences)
        out.write(f'{bleurt_score["scores"]}\n')
        out.write('BLEURT Average: \n')
        out.write(f'{bleurt_score["average"]}\n')


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
    run_and_save_evaluation(pred_file, ref_file, context_len, model_name, test_data_set, beam_size, linearization)


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

    pred_and_eval('inference_configs/t5_amrlib_ara1_split_1/inf_t5_amrlib_ara1_split_1_ara1_split_1.json')
    pred_and_eval('inference_configs/t5_amrlib_ara1_split_0/inf_t5_amrlib_ara1_split_0_ara1_split_0.json')

