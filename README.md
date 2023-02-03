# recipe-generation-model

This repository contains part of the code for the master thesis project of generating new recipe texts based on information from two recipes for the same dish. In particular, this repository contains the code to train a text generation model that generates a sentence based on the corresponding amr representation and the n previously generated sentences. 

The generation model is a pretrained T5 model from huggingface which gets fine-tuned on amr-to-text generation. The code is based on and adapted from the scripts from the [amrlib library](https://github.com/bjascob/amrlib).

The Readme contains information about
* [Preparing data sets](https://github.com/interactive-cookbook/recipe-generation-model#data-set-preparation)
* [Training the model](https://github.com/interactive-cookbook/recipe-generation-model#run-the-training)
* [Generating texts](https://github.com/interactive-cookbook/recipe-generation-model#run-prediction)
* and [evaluating](https://github.com/interactive-cookbook/recipe-generation-model#automatic-evaluation) the generated texts

## Requirements 
Tested with python 3.7 and 3.8 (and previously also with python 3.6 but huggingface evaluate library requires python 3.7) and the library versions listed in brackets below.

The [pytorch library](https://pytorch.org/get-started/locally/) (1.12.1)

[Transformers](https://huggingface.co/docs/transformers/installation#install-with-conda) from Huggingface (version 3 will probably not work) (4.24.0):
* `conda install -c huggingface transformers` (did not work under Linux for me)
* `pip install transformers` (did not work Windows for me)

[Sentence Piece](https://github.com/google/sentencepiece#installation) (0.1.97)
* `pip install sentencepiece`

For evaluation: [evaluate library](https://huggingface.co/docs/evaluate/index) from Huggingface and the dependencies for the computed metrices (0.3.0):
* `pip install evaluate` (0.0.2)
* `pip install sacrebleu` (2.3.1)
* `pip install rouge_score` (0.1.2)
* `pip install git+https://github.com/google-research/bleurt.git` (0.0.2)

nltk (3.7)
* `pip install nltk`

## Data set preparation

### Data set format

The data sets for training and evaluating the generation models should follow the format of the official LDC AMR data sets: <br>
All AMRs are separated from each other by an empty line and in addition to the AMRs, there can be one line at the top of the file starting with '# AMR release' which will be skipped. <br>
Each AMR should at least consist of the metadata '# ::snt' and then the AMR graph in penman notation. All other potentially included metadata information will be ignored and if the penman representation contains token alignments they can be kept or removed by choosing the corresponding linearization during the training process (see below). 

Example file:
```
# ::id waffles_0_instr0
# ::snt Beat eggs .
(b / beat-01~e.1
   :mode imperative~e.3
   :ARG0 (y / you~e.1)
   :ARG1 (e / egg~e.2))

# ::id waffles_0_instr1
# ::snt Mix in remaining ingredients .
(m / mix-01~e.4
   :mode imperative~e.8
   :ARG0 (y / you~e.4)
   :ARG1 (i / ingredient~e.7
            :ARG1-of (r / remain-01~e.6)))

# ::id waffles_0_instr2
# ::snt Cook on hot waffle iron .
(c / cook-01~e.9
   :mode imperative~e.9
   :ARG0 (y / you~e.9)
   :instrument (i / iron~e.13
                  :mod (w / waffle~e.12)
                  :ARG1-of (h / hot-05~e.11)))
```

The functions in the `create_data_splits.py` script can be used to create train, dev and test splits from the complete AMR 3.0 dataset, the multi-sentence subset of AMR 3.0 and from the Ara corpus.

### LDC AMR 3.0 data set

Run the `create_split_full_amr_corpus(amr_corpus_dir)` function with the path to the parent AMR 3.0 folder in order to create a folder `data/amr3_0` in the main repo directory containing a train.txt, valid.txt and test.txt file with the official train-valid-test split of the corpus.

### LDC AMR 3.0 multi-sentence data set

Run the `create_split_ms_corpus(amr_corpus_dir, train_per, val_per, test_per)` function where `amr_corpus_dir` is the path to a folder containing one file per document for which ms-amr annotations exist, each containing all AMRs (empty line separated) for that document in their original order. This will create three folders 'train', 'val' and 'test' in the './data' directory, such that the train folder will contain train_per\*100 percent of the available files, the val folder will contain val\*100 percent of the files and the test folder the rest. The files are randomly assigned to the splits.

### ARA data set

**Random Splits**<br>
Run the `create_split_ara_corpus(amr_corpus, train_per, val_per, test_per)` function where `amr_corpus` is the path a main ara corpus folder, i.e. the folder containing one subfolder per dish with one amr file per recipe. This will create three folders 'train', 'val' and 'test' in the './data' directory, such that the train folder will contain train_per\*100 percent of the available files, the val folder will contain val\*100 percent of the files and the test folder the rest. The files are randomly assigned to the splits.

**Reproducible Splits**<br>
In order to assign each recipe file to a dataset split and additionally save the information about the assignment into a file first run `create_recipe2split_assignment(amr_corpus_dir, split_name, train_per, val_per, test_per)`. This will randomly assign the recipe files from amr_corpus_dir to the train, val and test split according to the specified split proportions. The function does not create these splits, but only saves the assingments into a file './data_splits/[split_name]' with the following format: 
```
train \t waffles/waffles_0.conllu
val \t baked_ziti/baked-ziti_2.conllu
...
```

The actual data set splits can then be created by running the function `create_split_files_from_assignment(assignment_file, corpus_dir, split_dir)` which will copy the files listed in `assignment_file` (paths need to be relative to `corpus_dir`) from `corpus_dir` to the appropriate subdirectory of `split_dir`, i.e. train, val or test directory.


## Run the training

`python training.py --config [path_to_config_file]`

where the config file should be a .json file including the configuration for the training following the format of the files in the [training_configs](https://github.com/interactive-cookbook/recipe-generation-model/tree/main/training_configs) folder. See also the [Wiki](https://github.com/interactive-cookbook/recipe-generation-model/wiki/Training-Configuration-Files#run-training) for more information about the parameters in the configuration files. 

Note that all logging information is only printed to the command line. In order to write the information into a file run <br>
`python training.py --config [path_to_config_file] > path/log_file.txt`

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
    "checkpoint": "checkpoint-3726",
    "device": "cpu",
    "batch_size": 4,
    "num_beams": 1,
    "num_ret_seq": 1,
    "max_in_len": 1024,
    "max_out_len": 1024,
    "linearization": "penman"
  },
  "test_args": {
    "corpus_dir": "./data/ara1_amrs",
    "test_path": "test",
    "context_len": 1,
    "output_file": "output_amrlib_t5.txt"
  }
}
```

"generator_args" are the parameters used for instantiating an RecipeGenerator object 
* **"model_name_or_path"**: path to a the folder of a model trained with the training.py script, folder must also contain the config.json generated during training
* **"tokenizer_name_or_path"**: path to the folder containing the tokenizer used for training the model; all relevant files get saved in the same directory as the model so "model_name_or_path" and "tokenizer_name_or_path" should be identical usually
* **"checkpoint"**: optional, if a specific checkpoint should be used that is stored as a subfolder of the model path then this should be the name of this subfolder 
* **"device"**: "cpu" or e.g. 'cuda:0'
* **"batch_size"**
* **"num_beams"**: number of beams for the search
* **"num_ret_seq"**: number of sequences to return per input; needs to be smaller or equal to "num_beams"
* **"max_in_len"**: maximum input length; tokenizer will truncate longer input sequences; optional, default is set to 1024
* **"max_out_len"**: maximum output length; tokenizer will truncate longer target sequences; optional, default is set to 1024
* **"linearization"**: type of linearization to use (see explanation of training config above for the available options); optional, if not provided then the linearization used for training the model gets used

**"max_in_len"/"max_out_len"**<br>
If not specified in the inference configuration file then both get by default set to 1024. If no limitation and truncation of the input / output sequence should happen, then set the corresponding value to 0. But please note that setting "max_out_len" to 0 will have a negative impact on the inference results. 

"test_args" are parameters for testing the model by generating the model predictions for a complete test data set. In case the sentence generation is integrated into another framework, they are not necessary. Instead, a RecipeGenerator object should be instantiated with the parameters of "generator_args" and then the arguments to the RecipeGenerator.generate method can be constructed at at a different place. 
* **"corpus_dir"**: path to corpus directory, relative to inference.py
* **"test_path"**: path to the file with the complete test data or to a folder with several files for testing; path is relative to "corpus_dir"
* **"context_len"**: number of previous sentences of the same document to preprend to the current input graph
* **"output_file"**: The name of the outputfile where all generated sentences get written to

When running the inference.py script, this generates one single file for all the model predictions even if "test_path" is a directory with several files. Additionally, a second file is created containing all reference sentences for the generated sentences in the same order. This file has the same name as the one with the predictions, but with '\_references' as suffix. The files with the references and the predictions get created in the folder `repo_dir/output/[model_name]/[context_len]_context` where `model_name` is derived from "model_name-or_path"


## Automatic Evaluation

The script (evaluation.py)[https://github.com/interactive-cookbook/recipe-generation-model/blob/main/evaluation.py] contains functions to compute BLEU, chrF, ROUGE, METEOR and BLEURT scores for the generated texts. 

Running `python evaluation.py --input [path_input_file]` will compute all scores and print the output to the command line.

The script assumes that the input file contains only the generated text and that there exists a file with the reference text in the same folder that has the same file name with the suffix '_reference'. <br>
For example, if `--input` is `some_folder/output.txt` then there should also be a file `output_reference.txt` in `some_folder`


## Running Inference plus evaluation

The script `predict_and_evaluate.py ` can be used to generate text using a trained model, directly evaluate the model and save the results into a file. 

To do so, run the `pred_and_eval(inference_config_file)` function with the path to an inference configuration .json file as argument. This will generate the texts as specified in the configuration file and save the predicted and reference, then compute all evaluation metrices and save the output to a file with the same name as the file with the predicted texts plus the suffix '\_evaluation' in the same folder.
