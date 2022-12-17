import os

import comet_ml
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainer, IntervalStrategy, TrainingArguments, \
    TrainerState, TrainerControl
from transformers import Seq2SeqTrainingArguments
from transformers import TrainerCallback
from transformers.integrations import CometCallback
import json
import torch.cuda
import argparse
import evaluate
import numpy as np
from copy import deepcopy
from comet_ml import Experiment

from dataset_reader import build_dataset


# Taken from amrlib code: https://github.com/bjascob/amrlib/blob/master/amrlib/models/generate_t5/trainer.py
class T2TDataCollator:
    def __call__(self, batch):
        input_ids = torch.stack([example['input_ids'] for example in batch])
        lm_labels = torch.stack([example['target_ids'] for example in batch])
        lm_labels[lm_labels[:, :] == 0] = -100      # -100 is ignored by the loss -> used for padding ids
        attention_mask = torch.stack([example['attention_mask'] for example in batch])
        decoder_attention_mask = torch.stack([example['target_attention_mask'] for example in batch])

        # keys need to match the keys the model expects in the forward method
        collated_data = {'input_ids': input_ids, 'attention_mask': attention_mask,
                         'labels': lm_labels, 'decoder_attention_mask': decoder_attention_mask}

        return collated_data


def train_generation_model(config_file):
    """
    Reads the configuration file and starts the training
    :param config_file: json file with the configurations for the training
    :return:
    """
    with open(config_file) as c:
        config_args = json.load(c)

    torch.cuda.empty_cache()

    training_args = Seq2SeqTrainingArguments(**config_args['train_args'])

    _run_training_loop(config_args['gen_args'], training_args)


def _run_training_loop(general_config: dict, train_config: Seq2SeqTrainingArguments):
    """
    Runs the complete training of a T5 model
    :param general_config: general configuration parameters
    :param train_config: parameters for the Trainer
    :return:
    """

    #comet_ml.init(api_key="O8MdPzcEFBU5cd2o2OtqV6Pfy")
    os.environ["COMET_MODE"] = "ONLINE"
    os.environ["COMET_LOG_ASSETS"] = "True"

    print("---------- Loading Model and Tokenizer ----------")
    model_path = general_config['model_name_or_path']
    # Change dropout rate if provided
    drop_out = general_config.get('dropout_rate', 0.1)  # default value of the T5ForConditionalGeneration is 0.1

    tokenizer_path = general_config.get('tokenizer_name_or_path', general_config['model_name_or_path'])

    model = T5ForConditionalGeneration.from_pretrained(model_path, dropout_rate=drop_out)
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)

    # Add the new special token
    # information from here: https://github.com/huggingface/transformers/issues/8706
    separator_token = general_config['sep_token']
    new_special_token = {'additional_special_tokens': [separator_token]}
    tokenizer.add_special_tokens(new_special_token)
    model.resize_token_embeddings(len(tokenizer))

    # These get saved together with the model in a config.json file to store them
    model.config.task_specific_params = {'translation_cond_amr_to_text': general_config}

    print("---------- Reading the Data ----------")
    train_data_path = os.path.join(general_config['corpus_dir'], general_config['train_path'])
    valid_data_path = os.path.join(general_config['corpus_dir'], general_config['valid_path'])
    train_dataset = build_dataset(tokenizer=tokenizer,
                                  data_path=train_data_path,
                                  context_len=general_config['context_len'],
                                  linearization=general_config['linearization'],
                                  max_in_length=general_config['max_in_len'],
                                  max_out_length=general_config['max_out_len'],
                                  sep_token=separator_token)
    print(f'Loaded Training data set of {len(train_dataset)} instances.')
    valid_dataset = build_dataset(tokenizer=tokenizer,
                                  data_path=valid_data_path,
                                  context_len=general_config['context_len'],
                                  linearization=general_config['linearization'],
                                  max_in_length=general_config['max_in_len'],
                                  max_out_length=general_config['max_out_len'],
                                  sep_token=separator_token)
    print(f'Loaded validation data set of {len(valid_dataset)} instances.')

    def compute_metrics(eval_preds):
        metric = evaluate.load("sacrebleu")
        predictions, labels = eval_preds
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # postprocessing
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_refs = [[l.strip()] for l in decoded_labels]

        result = metric.compute(predictions=decoded_preds, references=decoded_refs)

        return {'bleu': round(result['score'], 4)}

    print("---------- Starting training ----------")
    trainer = Seq2SeqTrainer(model=model, args=train_config, train_dataset=train_dataset,
                      eval_dataset=valid_dataset, data_collator=T2TDataCollator(),
                      compute_metrics=compute_metrics)
    #trainer.add_callback(CustomCallback(trainer))
    trainer.add_callback(ConvergenceStoppingCallback(trainer))
    trainer.add_callback(CometCallback())
    trainer.train()
    print("---------- Finished training ----------")
    print("---------- Saving model and tokenizer ----------")
    trainer.save_model(train_config.output_dir)
    tokenizer.save_pretrained(train_config.output_dir)


class CustomCallback(TrainerCallback):
    """
    Trainer to compute and log BLEU during training
    """
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train", max_length=1024)
            return control_copy


class ConvergenceStoppingCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that handles
    1. computing BLEU on the Train datas set
    2. stopping when model converges based on the training loss
    Args:
       early_stopping_patience (`int`):
            Use with `metric_for_best_model` to stop training when the specified metric worsens for
            `early_stopping_patience` evaluation calls.
       early_stopping_threshold(`float`, *optional*):
            Use with TrainingArguments `metric_for_best_model` and `early_stopping_patience` to denote how much the
            specified metric must improve to satisfy early stopping conditions. `
    This callback depends on [`TrainingArguments`] argument *load_best_model_at_end* functionality to set best_metric
    in [`TrainerState`].
    """

    def __init__(self, trainer:Seq2SeqTrainer, early_stopping_patience: int = 15, early_stopping_threshold = 0.00005):
        super().__init__()
        self._trainer = trainer
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        # early_stopping_patience_counter denotes the number of times validation metrics failed to improve.
        self.early_stopping_patience_counter = 0
        self.best_loss = 100

    def check_metric_value(self, args, state, control):
        # best_metric is fixed to loss here so lower values are better
        # training metrices come before evaluation metrices in state.log_history because first eval
        # on train set takes place
        current_loss = None
        for i in range(len(state.log_history)-1, -1, -1):
            try:
                found_loss = state.log_history[i]['train_loss']
                current_loss = found_loss
                break
            except KeyError:
                continue
        assert current_loss

        operator = np.less
        if (operator(current_loss, self.best_loss)
            and abs(current_loss - self.best_loss) > self.early_stopping_threshold
        ):
            self.early_stopping_patience_counter = 0
            self.best_loss = current_loss
        else:
            self.early_stopping_patience_counter += 1

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train", max_length=1024)

            self.check_metric_value(args, state, control)
            if self.early_stopping_patience_counter >= self.early_stopping_patience:
                control.should_training_stop = True
            return control_copy


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the configuration file for training")
    args = parser.parse_args()
    config_file = args.config
    #config_file = "./training_configs/training_config_ara_dummy.json"
    print(torch.cuda.device_count())
    train_generation_model(config_file)
