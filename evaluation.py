import os
from nltk.tokenize import word_tokenize
import sacrebleu
import nltk.translate.bleu_score as nltk_bleu


# based on the amrlib 14_Score_BLEU.py and bleu_scorer.py scripts
# Difference: amrlib uses nltk.translated.bleu_score.corpus_bleu
# Bevilacqua et al and Ribeiro et al use sacrebleu

"""
Important:
nltk bleu expects one sublist for all references of a specific translation
e.g. preds = [p1, p2] and input_refs = [[ref1a, ref1b], [ref2a]] 
if instance 1 has two references and isntance 2 has only one -> number of references can be different

sacrebleu metrics expect on sublist per SET of references, 
e.g. preds = [p1, p2] and input_refs = [[ref1a, ref2a], [ref1b, ref2b]]
-> each instance needs to have the same number of references
"""

def read_file_for_eval(file_name):

    sentences = []
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().lower()
            if line:
                # tokens = line.split(' ')
                tokens = word_tokenize(line)
                sentences.append(tokens)
    return sentences


def run_eval(model_output_file):

    generated_file = model_output_file
    gold_file = model_output_file[:-4] + '_reference.txt'

    generated_snts = read_file_for_eval(generated_file)
    gold_snts = read_file_for_eval(gold_file)

    assert len(generated_snts) == len(gold_snts)

    bleu_nltk = compute_bleu_nltk(gold_snts, generated_snts)
    bleu_sacre = compute_bleu_sacre(gold_snts, generated_snts)
    chrf_sacre = compute_chrf_sacre(gold_snts, generated_snts)

    print(f'----- Evaluated {model_output_file} -----')
    print(f'BLEU NLTK Score: {bleu_nltk}')
    print(f'BLEU Sacre Score: {bleu_sacre}')
    print(f'chrF Sacre Score: {chrf_sacre}')


def compute_bleu_nltk(refs, preds):

    ref_len = sum([len(r) for r in refs])
    pred_len = sum([len(p) for p in preds])

    input_refs = [[r] for r in refs]
    bleu = nltk_bleu.corpus_bleu(input_refs, preds)

    return bleu, ref_len, pred_len


def compute_bleu_sacre(refs, preds):

    refs_text = [' '.join(r) for r in refs]
    preds_text = [' '.join(p) for p in preds]

    input_refs = [refs_text]
    bleu_scorer = sacrebleu.metrics.BLEU()
    bleu = bleu_scorer.corpus_score(preds_text, input_refs)

    return bleu


def compute_chrf_sacre(refs, preds):

    refs_text = [' '.join(r) for r in refs]
    preds_text = [' '.join(p) for p in preds]

    input_refs = [refs_text]
    chrf_scorer = sacrebleu.metrics.CHRF()
    chrf = chrf_scorer.corpus_score(preds_text, input_refs)

    return chrf


if __name__=='__main__':


    #run_eval('./output/0_context_ara2/output_amrlib_t5.txt')
    run_eval('./output/1_context_ara2/output_ms_amr_ara_t5.txt')
    run_eval('./output/0_context_ara2/output_ms_amr_ara_t5.txt')
    #run_eval('./output/0_train_context/output_for_ara1_from_ara1.txt')
    #run_eval('./output/0_train_context/output_for_ara1_from_ara1_orig.txt')


