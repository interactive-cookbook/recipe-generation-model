# recipe-generation-model

This repository contains part of the code for the master thesis project of generating new recipe texts based on information from two recipes for the same dish. In particular, this repository contains the code to train a text generation model that generates a sentence based on the corresponding amr representation and the n previously generated sentences. 

The generation model is the pretrained t5 model from huggingface which gets fine-tuned on amr-to-text generation. The code is based on and adapted from the scripts from the [amrlib library](https://github.com/bjascob/amrlib).

## Requirements 
Tested with python 3.7 (and previously also with python 3.6 but huggingface evaluate library requires python 3.7).

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

## Data set preparation


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
* **"model_name_or_path"**: path to a the folder of a local model or name of a huggingface model of the type T5ForConditionalGeneration
* **"tokenizer_name_or_path"**: path to the folder containing a trained tokenizer or name of a huggingface tokenizer of type T5Tokenizer; optional, if not provided the same as "model_name_or_path" will be used for loading the tokenizer
* **"corpus_dir"**: path to corpus directory, relative to training.py
* **"train_path"**: path to the file with the complete training data or to a folder with several files for training; path is relative to "corpus_dir"
* **"valid_path"**: path to the file with the complete validation data or to a folder with several files for validation; path is relative to "corpus_dir"
* **"max_in_len"**: maximum input length; tokenizer will truncate longer input sequences
* **"max_out_len"**: maximum output length; tokenizer will truncate longer target sequences
* **"context_len"**: number of previous sentences of the same document to preprend to the current input graph
* **"linearization"**: type of linearization to use for the amr graph; currently only "penman" implemented
* **"sep_token"**: the special token that should be added between the current graph and the previous context; will be added as special token to the vocab of the tokenizer

**"train_path"** / **"valid_path"**<br>
If "train_path" / "valid_path" is a directory, then each file in the directory is treated as one document if context_len > 0. If "train_path" / "valid_path" is a file, then that file is treated as one single document if context_len > 0.

**train_args**<br>
The "train_args" dictionary will be converted into a TrainingArguments object and passed to the transformer [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer#trainer). See the [TrainerArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments) documentation for information about possible parameters and default values. 

**linearization**<br>
Currently implemented are two options: 
* 'penman': does not make any changes to the input format, i.e. is the same penmanr string representation as in the input files (without the metadata)
* 'penman_wo_alignments': removes the node-to-token alignments from the amr string (i.e. removes all '~e.X' occurences where X is the aligned token ID)

**Important:** do not change **"remove_unused_columns"** to true or the functions will not work any more (see [here](https://github.com/huggingface/transformers/issues/9520) for more information)

**Note**: the transformer Trainer.train() function by default uses all availabel gpu nodes if "no_cuda": false is set. In order to restrict training to a single gpu run e.g. `CUDA_VISIBLE_DEVICES="3", python training.py --config [path_to_config_file]`

## Run prediction

`python inference.py --config [path_to_config_file`

where the config file should be a .json file including the configuration for the inference following the format of the files in the [inference_configs](https://github.com/interactive-cookbook/recipe-generation-model/tree/main/inference_configs) folder. 

Example
```
{
  "generator_args": {
    "model_name_or_path": "./models/t5_amrlib",
    "tokenizer_name_or_path": "t5-base",
    "device": "cpu",
    "batch_size": 4,
    "num_beams": 1,
    "num_ret_seq": 1
  },
  "test_args": {
    "corpus_dir": "./data/ara1_amrs",
    "test_path": "test",
    "context_len": 0,
    "output_file": "./output/output_amrlib_t5.txt"
  }
}
```

"generator_args" are the parameters used for instantiating an RecipeGenerator object 
* "model_name_or_path": path to a the folder of a model trained with the training.py script, folder must also contain the config.json generated during training
* "tokenizer_name_or_path": path to the folder containing the tokenizer used for training the model; all relevant files get saved in the same directory as the model so "model_name_or_path" and "tokenizer_name_or_path" should be identical usually
* "device": "cpu" or e.g. 'cuda:0'
* "batch_size"
* "num_beams": number of beams for the search
* "num_ret_seq": number of sequences to return per input; needs to be smaller or equal to "num_beams"

"test_args" are parameters for testing the model by generating the model predictions for a complete test data set. In case the sentence generation is integrated into another framework, they are not necessary. Instead, a RecipeGenerator object should be instantiated with the parameters of "generator_args" and then the arguments to the RecipeGenerator.generate method can be constructed at at a different place. 
* "corpus_dir": path to corpus directory, relative to inference.py
* "test_path": path to the file with the complete test data or to a folder with several files for testing; path is relative to "corpus_dir"
* "context_len": number of previous sentences of the same document to preprend to the current input graph
* "output_file": path to the outputfile where all generated sentences get written to

When running the inference.py script, this generates one single file for all the model predictions even if "test_path" is a directory with several files. Additionally, a second file is created containing all reference sentences for the generated sentences in the same order. This file has the same name as the one with the predictions, but with '\_references' as suffix. 


## Automatic Evaluation

The script (evaluation.py)[https://github.com/interactive-cookbook/recipe-generation-model/blob/main/evaluation.py] contains functions to compute BLEU and chrF scores for the generated texts. 

Running `python evaluation.py --input [path_input_file]` will compute both scores and print the output to the command line.

The script assumes that the input file contains only the generated text and that there exists a file with the reference text in the same folder that has the same file name with the suffix '_reference'. <br>
For example, if `--input` is `some_folder/output.txt` then there should also be a file `output_reference.txt` in `some_folder`
