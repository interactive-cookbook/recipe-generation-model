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

## Run prediction

`python inference.py --config [paht_ot_config_file`

where the config file should be a .json file including the configuration for the inference following the format of the files in the [inference_configs](https://github.com/interactive-cookbook/recipe-generation-model/tree/main/inference_configs) folder. 
