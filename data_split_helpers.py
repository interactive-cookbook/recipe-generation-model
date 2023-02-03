import os
import random
from pathlib import Path
import math
from typing import List, Tuple
import shutil


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


def assign_recipe2split(amr_doc_files: List[str], train_per: float, val_per: float, test_recipes=None) \
        -> Tuple[List[str], List[str], List[str]]:
    """
    Randomly assigns the files in amr_doc_files to a train, val and test split accordingt to the
    specified proportions
    :param amr_doc_files: list of all the file names to assign to the different splits
    :param train_per: proportion of files to use for training
    :param val_per: proportion of files to use for validation
    :param test_recipes: optional set of the files for the test set
    :return: Tuple(List, List, List)
            List with file names assigned to training split
            List with file names assigned to validation split
            List with file names assigned to test split
    """

    n_files = len(amr_doc_files)
    n_train_split = math.ceil(n_files * train_per)
    n_val_split = math.ceil(n_files * val_per)
    if not test_recipes:
        random.shuffle(amr_doc_files)
        train_files = amr_doc_files[:n_train_split]
        val_files = amr_doc_files[n_train_split:n_val_split + n_train_split]
        test_files = amr_doc_files[n_val_split + n_train_split:]
    else:
        remaining_amr_files = [file for file in amr_doc_files if file not in test_recipes]
        random.shuffle(remaining_amr_files)
        train_files = remaining_amr_files[:n_train_split]
        val_files = remaining_amr_files[n_train_split:n_val_split + n_train_split]
        test_files = list(test_recipes)

    assert len(train_files) + len(val_files) + len(test_files) == len(amr_doc_files)

    return train_files, val_files, test_files


def create_split_files(corpus_dir, split_dir, train_files: List[str], val_files: List[str], test_files: List[str]):
    """
    Copies all files in train_files to a folder split_dir/train, all files from val_files to a folder
    split_dir/val and all files from test_files to split_dir/test
    :param corpus_dir: path to the parent directory of the unsplit corpus
    :param split_dir: path/name of the parent directory for the newly created data split
    :param train_files: list of the file names (relative to corpus_dir) for the training split
    :param val_files: list of the file names (relative to corpus_dir) for the training split
    :param test_files: list of the file names (relative to corpus_dir) for the training split
    :return:
    """
    Path(split_dir).mkdir(exist_ok=True, parents=True)
    train_dir = os.path.join(split_dir, 'train')
    val_dir = os.path.join(split_dir, 'val')
    test_dir = os.path.join(split_dir, 'test')
    Path(train_dir).mkdir(exist_ok=True, parents=True)
    Path(val_dir).mkdir(exist_ok=True, parents=True)
    Path(test_dir).mkdir(exist_ok=True, parents=True)

    for file_name in train_files:
        source = os.path.join(corpus_dir, file_name)
        shutil.copy2(source, train_dir)
    for file_name in val_files:
        source = os.path.join(corpus_dir, file_name)
        shutil.copy2(source, val_dir)
    for file_name in test_files:
        source = os.path.join(corpus_dir, file_name)
        shutil.copy2(source, test_dir)


