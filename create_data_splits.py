import os
import random
from pathlib import Path
import math
import shutil
from typing import List


def create_split_full_amr_corpus(amr_corpus_dir):
    """
    Creates one file for the train split, one for the test split and one for dev split
    Each contains all AMRs from the files in the corresponding split folders of the amr corpus
    :param amr_corpus_dir: the parent folder of the original amr 3.0 corpus
    :return:
    """
    train_folder = os.path.join(amr_corpus_dir, 'data/amrs/split/training')
    valid_folder = os.path.join(amr_corpus_dir, 'data/amrs/split/dev')
    test_folder = os.path.join(amr_corpus_dir, 'data/amrs/split/test')

    Path(os.path.join('./data/amr3_0')).mkdir(exist_ok=True, parents=True)

    train_amrs = []
    valid_amrs = []
    test_amrs = []
    for train_file in os.listdir(train_folder):
        train_amrs.extend(read_amr_file(os.path.join(train_folder, train_file)))
    for valid_file in os.listdir(valid_folder):
        valid_amrs.extend(read_amr_file(os.path.join(valid_folder, valid_file)))
    for test_file in os.listdir(test_folder):
        test_amrs.extend(read_amr_file(os.path.join(test_folder, test_file)))

    with open(os.path.join('./data/amr3_0/train.txt'), 'w', encoding='utf-8') as f:
        for amr in train_amrs:
            f.write(f'{amr}\n\n')
    with open(os.path.join('./data/amr3_0/valid.txt'), 'w', encoding='utf-8') as f:
        for amr in valid_amrs:
            f.write(f'{amr}\n\n')
    with open(os.path.join('./data/amr3_0/test.txt'), 'w', encoding='utf-8') as f:
        for amr in test_amrs:
            f.write(f'{amr}\n\n')


def read_amr_file(file) -> List[str]:
    """
    Reads a file with amrs in penman format and returns a list with the individual amr strings
    :param file: amr file path
    :return: list of the amr strings from the file
    """
    amrs = []

    with open(file, 'r', encoding='utf-8') as f:
        document = f.read()

    for amr_data in document.split('\n\n'):
        if amr_data.startswith('# AMR release'):
            continue
        else:
            amrs.append(amr_data)

    return amrs


def create_split_ms_amr_corpus(amr_corpus_dir, train_per: float = 0.8, val_per: float = 0.1, test_per: float = 0.1):
    """
    Reads all files containing the multi-sentence amrs (one file per document) and copies some of them to a folder
    ./data/ms_amr/train, some to ./data/ms_amr/val and rest to ./data/ms_amr/test
    Number of files copied to each of these folders is specified by train/val/test_per,
    i.e. if train_per = 0.8 then 80% of the files end up in the train folder,
    number of files for train and val split are rounded up; the remaining files get added to the test split
    :param amr_corpus_dir: the parent corpus directory with all ms_amr files
    :param train_per: proportion of files to use for training
    :param val_per: proportion of files to use for validation
    :param test_per: proportion of files to use for testing
    :return:
    """
    assert train_per + val_per + test_per == 1
    ms_amr_files = list(os.listdir(amr_corpus_dir))
    random.shuffle(ms_amr_files)

    split_dir = Path('./data/ms_amr')

    _split_doc_corpus(amr_corpus_dir, ms_amr_files, split_dir, train_per, val_per)


def create_split_ara_corpus(amr_corpus_dir, train_per, val_per, test_per):
    """
    Reads all files containing the ara1 amrs (one file per recipe) and copies some of them to a folder
    ./data/ara1_amrs/train, some to ./data/ara1_amr/val and rest to ./data/ara1_amr/test
    Number of files copied to each of these folders is specified by train/val/test_per,
    i.e. if train_per = 0.8 then 80% of the files end up in the train folder,
    :param amr_corpus_dir: the parent corpus directory with one subfolder per dish
    :param train_per: proportion of files to use for training
    :param val_per: proportion of files to use for validation
    :param test_per: proportion of files to use for testing
    :return:
    """
    assert train_per + val_per + test_per == 1
    recipes_amr_files = []

    for dish in os.listdir(amr_corpus_dir):
        dish_dir = os.path.join(Path(amr_corpus_dir), dish)
        for recipe in os.listdir(dish_dir):
            file_path = os.path.join(dish, recipe)
            recipes_amr_files.append(file_path)

    random.shuffle(recipes_amr_files)
    split_dir = Path('./data/ara1_amrs')

    _split_doc_corpus(amr_corpus_dir, recipes_amr_files, split_dir, train_per, val_per)


def _split_doc_corpus(amr_corpus_dir, amr_doc_files, split_dir, train_per: float, val_per: float):
    """
    Does the actual splitting of the files into train, test and valid split and copies them to the folders
    :param amr_corpus_dir: the parent corpus dir
    :param amr_doc_files: list of all the files (i.e. their paths relative to amr_corpus_dir) to assign to the different
                          splits
    :param split_dir: the name of the parent directory for the 'train', 'test' and 'val' folders
    :param train_per: proportion of files to use for training
    :param val_per: proportion of files to use for validation
    :return:
    """
    n_files = len(amr_doc_files)
    n_train_split = math.ceil(n_files * train_per)
    n_val_split = math.ceil(n_files * val_per)

    train_files = amr_doc_files[:n_train_split]
    val_files = amr_doc_files[n_train_split:n_val_split+n_train_split]
    test_files = amr_doc_files[n_val_split+n_train_split:]
    assert len(train_files) + len(val_files) + len(test_files) == len(amr_doc_files)

    Path(split_dir).mkdir(exist_ok=True, parents=True)
    train_dir = os.path.join(split_dir, 'train')
    val_dir = os.path.join(split_dir, 'val')
    test_dir = os.path.join(split_dir, 'test')
    Path(train_dir).mkdir(exist_ok=True, parents=True)
    Path(val_dir).mkdir(exist_ok=True, parents=True)
    Path(test_dir).mkdir(exist_ok=True, parents=True)

    for file_name in train_files:
        source = os.path.join(amr_corpus_dir, file_name)
        shutil.copy2(source, train_dir)
    for file_name in val_files:
        source = os.path.join(amr_corpus_dir, file_name)
        shutil.copy2(source, val_dir)
    for file_name in test_files:
        source = os.path.join(amr_corpus_dir, file_name)
        shutil.copy2(source, test_dir)




if __name__=='__main__':
    #create_split_ms_amr_corpus('../recipe-generation/training/tuning_data_sets/ms_amr_graphs', 0.8, 0.1, 0.1)

    #create_split_ara_corpus('../recipe-generation/training/tuning_data_sets/ara1_amr_graphs', 0.8, 0.1, 0.1)

    #create_split_full_amr_corpus('../amr_annotation_3.0')

    #create_split_ara_corpus('../recipe-generation/data/recipe_amrs_actions', 0.8, 0.1, 0.1)

    create_split_ara_corpus('../recipe-generation/training/tuning_data_sets/ara2_amr_graphs', 0, 0, 1)

    pass