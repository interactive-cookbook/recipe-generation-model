import os
from typing import List
from nltk.tokenize import word_tokenize
import evaluate
from argparse import ArgumentParser


# amrlib uses nltk.translated.bleu_score.corpus_bleu
# Bevilacqua et al and Ribeiro et al use sacrebleu

"""
Important:
BLEU, SacreBLEU and chrF expect as reference argument a list with one sublist per prediction,
each containing potentially several references for that specific prediction
e.g. preds = [p1, p2] and input_refs = [[ref1a, ref1b], [ref2a]] 
if instance 1 has two references and isntance 2 has only one -> number of references can be different

BLEURT expects as reference argument a list with one reference string for each prediction. 

ROUGE and METEOR can deal with both, a list with one reference string per prediction or a list of lists of reference 


ROUGE and METEOR expects that for each input string, all tokens of the sentence are separated by white spaces
BLEU and SacreBLEU use tokenizer_13a as default tokenizer, 
"""


def read_file_for_eval(file_name) -> List[str]:
    """
    Read in a file with predicted or reference sentences,
    one sentence per line
    :param file_name:
    :return: list of the sentences of the complete text
    """
    sentences = []
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().lower()
            if line:
                sentences.append(line)
    return sentences


def run_eval(model_output_file):
    """
    Evaluates the generated text in the model_output_file against the text in the corresponding references file
    e.g. if the file name is 'model_output.txt' then the references should be in 'model_output_reference.txt'
    Computes BLEU, Sacrebleu, chrF, ROUGE and BLEURT from https://huggingface.co/evaluate-metric
    :param model_output_file: path to a file with the model output
    :return:
    """
    generated_file = model_output_file
    gold_file = model_output_file[:-4] + '_reference.txt'

    generated_snts = read_file_for_eval(generated_file)
    ref_snts = read_file_for_eval(gold_file)
    ref_snts_input = [[r] for r in ref_snts]
    tokenized_generated_snts = [' '.join(word_tokenize(s)) for s in generated_snts]
    tokenized_ref_snts = [' '.join(word_tokenize(r)) for r in ref_snts]

    assert len(generated_snts) == len(ref_snts)

    # https://huggingface.co/spaces/evaluate-metric/bleu
    bleu_scorer= evaluate.load("bleu")
    # https://huggingface.co/spaces/evaluate-metric/sacrebleu
    sacre_bleu_scorer = evaluate.load("sacrebleu")
    # https://huggingface.co/spaces/evaluate-metric/chrf
    chrf_scorer = evaluate.load("chrf")
    # https://huggingface.co/spaces/evaluate-metric/rouge
    rouge_scorer = evaluate.load("rouge")
    # https://huggingface.co/spaces/evaluate-metric/meteor
    meteor_scorer = evaluate.load("meteor")
    # https://huggingface.co/spaces/evaluate-metric/bleurt
    bleurt_scorer = evaluate.load("bleurt", module_type="metric")

    bleu = bleu_scorer.compute(predictions=generated_snts, references=ref_snts_input)
    sacrebleu = sacre_bleu_scorer.compute(predictions=generated_snts, references=ref_snts_input)
    chrf = chrf_scorer.compute(predictions=generated_snts, references=ref_snts_input)
    rouge = rouge_scorer.compute(predictions=tokenized_generated_snts, references=tokenized_ref_snts)
    meteor = meteor_scorer.compute(predictions=tokenized_generated_snts, references=tokenized_ref_snts)
    bleurt = bleurt_scorer.compute(predictions=generated_snts, references=ref_snts)

    print(f'----- Evaluated {model_output_file} -----')
    print(f'BLEU Score: {bleu}')
    print(f'Sacrebleu score: {sacrebleu}')
    print(f'chrF Sacre Score: {chrf}')
    print(f'ROUGE score: {rouge}')
    print(f'METEOR score: {meteor}')
    print(f'BLEURT score: {bleurt}')


if __name__=='__main__':

    #parser = argparse.ArgumentParser()
    #parser.add_argument('--input', required=True)
    #args = parser.parse_args()
    #run_eval(args.input)

    run_eval("./output/0_context/output_ms_amr_t5.txt")



