import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer
from transformers import TrainingArguments
import json
import torch.cuda

from dataset_reader import build_dataset


# Taken from amrlib code
class T2TDataCollator:
    def __call__(self, batch):
        input_ids = torch.stack([example['input_ids'] for example in batch])
        lm_labels = torch.stack([example['target_ids'] for example in batch])
        lm_labels[lm_labels[:, :] == 0] = -100      # -100 is ignored by the loss -> used for padding ids
        attention_mask = torch.stack([example['attention_mask'] for example in batch])
        decoder_attention_mask = torch.stack([example['target_attention_mask'] for example in batch])

        collated_data = {'input_ids': input_ids, 'attention_mask': attention_mask,
                         'labels': lm_labels, 'decoder_attention_mask': decoder_attention_mask}

        return collated_data


def train_generation_model(config_file):
    with open(config_file) as c:
        config_args = json.load(c)

    torch.cuda.empty_cache()

    training_args = TrainingArguments(**config_args['train_args'])

    _run_training_loop(config_args['gen_args'], training_args)


def _run_training_loop(general_config: dict, train_config: TrainingArguments):

    print("---------- Loading Model and Tokenizer ----------")
    model_path = general_config['model_name_or_path']
    try:
        tokenizer_path = general_config['tokenizer_name_or_path']
    except KeyError:
        tokenizer_path = general_config['model_name_or_path']
    model = T5ForConditionalGeneration.from_pretrained(model_path)
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

    print("---------- Starting training ----------")
    trainer = Trainer(model=model, args=train_config, train_dataset=train_dataset,
                      eval_dataset=valid_dataset, data_collator=T2TDataCollator())
    trainer.train()
    print("---------- Finished training ----------")
    print("---------- Saving model and tokenizer ----------")
    trainer.save_model(train_config['output_dir'])
    tokenizer.save_pretrained(train_config['output_dir'])


if __name__=='__main__':

    c = "./training_configs/training_config_3.json"

    train_generation_model(c)
