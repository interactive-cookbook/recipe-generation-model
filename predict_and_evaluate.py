import json
import os
from pathlib import Path
from argparse import ArgumentParser

from inference import generate_data_set, RecipeGenerator
from evaluation import read_file_for_eval, compute_meteor, compute_bleurt, compute_rouge, compute_chrf, compute_bleu, compute_chrf_plus


def run_and_save_evaluation(pred_file, ref_file, context_len, model_name, test_data_set,
                            beam_size, linearization, model_config):
    """
    Computes BLEU, ROUGE, chrF, chrF++, BLEURT and METEOR scores for the predicted sentences in pred_file
    relative to the reference sentences in ref_file
    All values of the metrics get written into a file named after the pred_file but with suffix "_evaluation.txt"
    Additionally parameters used for creating the predictions are stored in the file
    :param pred_file: path to the file with the predictions, one sentence per line
    :param ref_file: path to the file with the references, one sentence per line, same order as in pred_file
    :param context_len: length of the context used for creating the predictions
    :param model_name: name of the model that was used to obtain the predictions
    :param test_data_set: name of the dataset the predictions belong to
    :param beam_size: beam size used for inference
    :param linearization: linearization type used for inference
    :param model_config: path to the model_config file (i.e. with the training parameters and other model parameters)
    :return:
    """
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
    """
    Runs the inference as specified in the configuration file, computes all automatic metrics and saves them
    into a file with the same name as the output file with the suffix '_evaluation.txt'
    :param inference_config_file:
    :return:
    """
    # runs inference, i.e. creates a file with the predictions
    generator: RecipeGenerator = generate_data_set(inference_config_file)

    # extract all information from the config file that are relevant for creating the output paht / file and
    # that should be saved together with the evaluation results
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
    """

    :param context_len: length of the context
    :param model_path: path to the model
    :param output_file: name of the file with the model output
    :param model_checkpoint: path to the checkpoint folder if different from model_path
    :return: path to the file with the predictions (i.e. output from running inference)
            path to the file with the reference sentences
            name of the model (without the path structure)
    """
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
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the configuration file for generation")
    args = parser.parse_args()
    config_file = args.config
    pred_and_eval(config_file)

    #pred_and_eval('./inference_configs/t5_amrlib_ara1_original_0/inf_t5_amrlib_ara1_original_0_ara1_split_0.json')






