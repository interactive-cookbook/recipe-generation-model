import os
from nltk.tokenize import workd_tokenize
from nltk.translate.bleu_score import corpus_bleu


# based on the amrlib 14_Score_BLEU.py and bleu_scorer.py scripts

def read_file_for_eval(file_name):

    sentences = []
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().lower()
            if line:
                # TODO: probably replace with nltk.word_tokenize
                tokens = line.split(' ')
                sentences.append(tokens)
    return sentences

def run_bleu_eval(model_output_file):

    generated_file = model_output_file
    gold_file = model_output_file[:-4] + '_reference.txt'

    generated_snts = read_file_for_eval(generated_file)
    gold_snts = read_file_for_eval(gold_file)

    assert len(generated_snts) == len(gold_snts)


def compute_bleu():
    pass

