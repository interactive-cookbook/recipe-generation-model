import argparse
import os
import json
import torch
from argparse import ArgumentParser
from typing import List
from transformers import T5ForConditionalGeneration, T5Tokenizer
from dataset_reader import read_data_set, remove_token_alignments
from torch.utils.data import DataLoader
from tqdm import tqdm


def generate_data_set(config_file):
    """
    Generate text from a data set of AMRs
    :param config_file: the json configuration file for generation
    :return:
    """
    with open(config_file) as c:
        config_args = json.load(c)

    torch.cuda.empty_cache()

    _run_inference(config_args['test_args'], config_args['generator_args'])


def _run_inference(inference_config:dict, generator_config: dict):
    """
    Runs inference on the complete test data set defined in the inference config dict and stores
    the generated sentences as well as the references in the folder specified in the inference config dict
    Inference is performed using an instantiation of the RecipeGenerator class, instantiated
    with the parameters in the generator_config
    :param inference_config: parameters related to the data set to generate from
    :param generator_config: parameters for the model to use for generation
    :return:
    """
    corpus_dir = inference_config['corpus_dir']
    test_path = os.path.join(corpus_dir, inference_config['test_path'])
    context_len = inference_config['context_len']

    print("---------- Loading Model and Tokenizer ----------")
    generation_model = RecipeGenerator(generator_config)

    print("---------- Reading the Data ----------")
    test_data_entries = read_data_set(test_path, context_len, generation_model.linearization)
    graphs = test_data_entries['graph']
    contexts = test_data_entries['context']

    print("---------- Starting generation ---------")
    generated_text = generation_model.generate(contexts, graphs)
    print("---------- Finished generation ---------")

    print("---------- Save output ----------")
    with open(inference_config['output_file'], 'w', encoding='utf-8') as f:
        for sentence in generated_text:
            f.write(f'{sentence}\n')
    ref_file = inference_config['output_file'][:-4] + '_reference.txt'
    with open(ref_file, 'w', encoding='utf-8') as f:
        for ref_snt in test_data_entries['sent']:
            f.write(f'{ref_snt}\n')


# Based on amrlib code: https://github.com/bjascob/amrlib/blob/master/amrlib/models/generate_t5/inference.py
class RecipeGenerator:
    """
    A class to hold the fine-tuned generation model and all parameters relevant for running the
    generation as well as the method to run the generation for contextualized AMR graphs
    """
    def __init__(self, configuration: dict):
        self.model = T5ForConditionalGeneration.from_pretrained(configuration['model_name_or_path'])
        self.tokenizer = T5Tokenizer.from_pretrained(configuration['tokenizer_name_or_path'])
        self.max_in_len = self.model.config.task_specific_params['translation_cond_amr_to_text']['max_in_len']
        self.max_out_len = self.model.config.task_specific_params['translation_cond_amr_to_text']['max_out_len']

        self.device = configuration['device'] if torch.cuda.is_available() else 'cpu'
        self.batch_size = configuration.get('batch_size', 1)
        self.num_beams = configuration.get('num_beams', 1)
        self.num_ret_seq = configuration.get('num_ret_seq', 1)
        self.linearization = self.model.config.task_specific_params['translation_cond_amr_to_text'].get('linearization', 'penman')
        self.sep_token = self.model.config.task_specific_params['translation_cond_amr_to_text'].get('sep_token', '')

    # TODO: need to change linearization if implemented
    def generate(self, contexts: List[str], graphs: List[str]) -> List[str]:
        """
        Generates sentences for context-graph input sequences using self.model and self.tokenizer
        graphs should be a list of penman graph strings without metadata, line breaks or indentation
        contexts should contain the corresponding contexts for the graphs in the matching order
        each context at position i in contexts will be prepended to the graph at position i in graphs
        separated by the self.sep_token
        If no context should be preprended, the corresponding context in contexts should be the empty string
        :param contexts: a list of contexts
        :param graphs: a list of graphs
        :return: the list of sentences generated from the graphs and previous context
        """
        # some assertions to identify problem with the input beforehand
        assert isinstance(graphs, list)
        assert isinstance(contexts, list)
        assert len(graphs) == len(contexts)
        for g in graphs:
            assert not g.startswith('# ::'), "AMR graphs include metadata. Please remove the metadata to get the correct" \
                                             "behavior of the model. "

        if self.linearization == 'penman_wo_alignments':
            converted_graphs = [remove_token_alignments(gr_str) for gr_str in graphs]
            graphs = converted_graphs
        elif self.linearization != 'penman':
            raise NotImplemented

        contextualized_graphs = [f'{c} {self.sep_token} {g}' for (c, g) in zip(contexts, graphs)]
        generated_sentences = []

        dataloader = DataLoader(contextualized_graphs, batch_size=self.batch_size)
        for batch in tqdm(dataloader):
            input_str = ['%s' % c_gr for c_gr in batch]
            # encode input
            input_encodings = self.tokenizer.batch_encode_plus(input_str,
                                                               padding=True,
                                                               truncation=True,
                                                               max_length=self.max_in_len,
                                                               return_overflowing_tokens=True)
            input_ids = torch.LongTensor(input_encodings['input_ids']).to(self.device)
            attention_mask = torch.LongTensor(input_encodings['attention_mask']).to(self.device)

            # run inference
            output = self.model.generate(input_ids=input_ids,
                                         attention_mask=attention_mask,
                                         max_length=self.max_out_len,
                                         early_stopping=True,
                                         num_beams=self.num_beams,
                                         num_return_sequences=self.num_ret_seq)

            # decode the output
            decoded_output = [self.tokenizer.decode(out_ids, skip_special_tokens=True) for out_ids in output]
            generated_sentences.extend(decoded_output)

        return generated_sentences


if __name__=='__main__':

    #parser = argparse.ArgumentParser()
    #parser.add_argument("--config", required=True, help="path to the configuration file for generation")
    #args = parser.parse_args()
    #config_file = args.config
    #generate_data_set(config_file)

    #generate_data_set('inference_configs/inference_debug.json')
    #generate_data_set('inference_configs/inference_t5_ms_amr_ara_no_context.json')
    generate_data_set('inference_configs/inference_t5_ara1_orig_no_context.json')

