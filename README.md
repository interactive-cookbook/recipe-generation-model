# recipe-generation-model

This repository contains part of the code for the master thesis project of generating new recipe texts based on information from two recipes for the same dish. In particular, this repository contains the code to train a text generation model that generates a sentence based on the corresponding amr representation and the n previously generated sentences. 

The generation model is the pretrained t5 model from huggingface which gets fine-tuned on amr-to-text generation. The code is based on and adapted from the scripts from the [amrlib library](https://github.com/bjascob/amrlib).

## Requirements 
The [networkX library](https://networkx.org/documentation/stable/index.html) (really needed?):
* `pip install networkx[default]`
* `pip install graphviz`

The [penman library](https://github.com/goodmami/penman/) (really needed?):
* `pip install penman`

The [pytorch library](https://pytorch.org/get-started/locally/)

[Transformers](https://huggingface.co/docs/transformers/installation#install-with-conda) from Huggingface:
* `pip install transformers` (resulted in an error for me on Windows 10, python 3.6 anaconda environment)
* `conda install -c huggingface transformers` (was successful)

[Sentence Piece](https://github.com/google/sentencepiece#installation)
* `pip install sentencepiece`

[Sacrebleu](https://github.com/mjpost/sacrebleu):
* `pip install sacrebleu`


## Run the training

`python training.py --config [path_to_config_file]`

where the config file should be a .json file including the configuration for the training following the format of the files in the [training_configs](https://github.com/interactive-cookbook/recipe-generation-model/tree/main/training_configs) folder. 

Example: 
```
{"gen_args": {
    "model_name_or_path": "t5-base",
    "tokenizer_name_or_path": "t5-base",
    "corpus_dir": "./data/ara1_amrs",
    "train_path": "train",
    "valid_path": "val",
    "max_in_len": 1024,
    "max_out_len": 1024,
    "context_len": 1,
    "linearization": "penman",
    "sep_token": "<GRAPH>"
  },
    "train_args": {
      "output_dir": "./models/train_t5_ara1_amr/t5_ara1_amr",
      "do_train": true,
      "do_eval": false,
      "overwrite_output_dir": false,
      "prediction_loss_only": true,
      "num_train_epochs": 8,
      "save_steps": 1000,
      "save_total_limit": 2,
      "per_device_train_batch_size": 1,
      "per_device_eval_batch_size": 1,
      "gradient_accumulation_steps": 24,
      "learning_rate": 1e-4,
      "seed": 42,
      "remove_unused_columns": false,
      "no_cuda": false
  }
}
```
The key, value pairs in the scope of "gen_args" are needed to specify the following information:
* "model_name_or_path": path to a the folder of a local model or name of a huggingface model of the type T5ForConditionalGeneration
* "tokenizer_name_or_path": path to the folder containing a trained tokenizer or name of a huggingface tokenizer of type T5Tokenizer; optional, if not provided the same as "model_name_or_path" will be used for loading the tokenizer
* "corpus_dir": path to corpus directory, relative to training.py
* "train_path": path to the file with the complete training data or to a folder with several files for training; path is relative to "corpus_dir"
* "valid_path": path to the file with the complete validation data or to a folder with several files for validation; path is relative to "corpus_dir"
* "max_in_len": maximum input length; tokenizer will truncate longer input sequences
* "max_out_len": maximum output length; tokenizer will truncate longer target sequences
* "context_len": number of previous sentences to preprend to the current input graph
* "linearization": type of linearization to use for the amr graph; currently only "penman" implemented
* "sep_token": the special token that should be added between the current graph and the previous context; will be added as special token to the vocab of the tokenizer

The "train_args" dictionary will be converted into a TrainingArguments object and passed to the transformer [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer#trainer). See the [TrainerArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments) documentation for information about possible parameters and default values. 
**Important:** do not change "remove_unused_columns" to true or the functions will not work any more (see [here](https://github.com/huggingface/transformers/issues/9520) for more information)

## Run prediction

`python inference.py --config [path_to_config_file`

where the config file should be a .json file including the configuration for the inference following the format of the files in the [inference_configs](https://github.com/interactive-cookbook/recipe-generation-model/tree/main/inference_configs) folder. 
