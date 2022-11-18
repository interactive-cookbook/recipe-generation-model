import os
from pathlib import Path
from argparse import ArgumentParser
from data_split_helpers import read_amr_file, assign_recipe2split, create_split_files


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


def create_random_split_ms_amr_corpus(amr_corpus_dir, train_per: float = 0.8, val_per: float = 0.1, test_per: float = 0.1):
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

    split_dir = Path('./data/ms_amr')

    train_docs, val_docs, test_docs = assign_recipe2split(ms_amr_files, train_per, val_per)
    create_split_files(amr_corpus_dir, split_dir, train_docs, val_docs, test_docs)


def create_random_split_ara_corpus(amr_corpus_dir, train_per, val_per, test_per):
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

    split_dir = Path('./data/ara_amrs')

    train_docs, val_docs, test_docs = assign_recipe2split(recipes_amr_files, train_per, val_per)
    create_split_files(amr_corpus_dir, split_dir, train_docs, val_docs, test_docs)


def create_recipe2split_assignment(amr_corpus_dir, split_name, train_per=0.8, val_per=0.1, test_per=0.1):
    """
    Traverses all files in all subdirectories of amr_corpus_dir and randomly assigns them to train, val or
    test split.
    Number of files assigned to each split is specified by train/val/test_per,
    i.e. if train_per = 0.8 then 80% of the files are assigned to train.
    The assignment is saved to ./data_splits/split_name;
    one line per file, [split_type]\t[file_name] where [file_name] is relative to amr_corpus_dir
    :param amr_corpus_dir: path to the parent corpus folder
    :param split_name: name for the assignment file
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
            recipe_path = os.path.join(dish, recipe)
            recipes_amr_files.append(recipe_path)

    train_files, val_files, test_files = assign_recipe2split(recipes_amr_files, train_per, val_per)
    Path('./data_splits').mkdir(exist_ok=True, parents=True)
    with open(os.path.join('./data_splits/', split_name + '.tsv'), 'w', encoding='utf-8') as f:
        for train_recipe in train_files:
            f.write(f'train\t{train_recipe}\n')
        for val_recipe in val_files:
            f.write(f'val\t{val_recipe}\n')
        for test_recipe in test_files:
            f.write(f'test\t{test_recipe}\n')


def create_split_files_from_assignment(assignment_file, corpus_dir, split_dir):
    """
    Reads in a file where each line contains the split type (e.g. 'train') and the path of the file
    relative to corpus_dir, separated by \t
    Copies all listed files to the folders split_dir/train, split_dir/val and split_dir/test
    depending on the split type specified for the file
    :param assignment_file: the file with the assignments
    :param corpus_dir: the parent unsplit corpus directory
    :param split_dir: the path to the newly created directory for the split data set
    :return:
    """
    train_files = []
    val_files = []
    test_files = []

    with open(assignment_file, 'r', encoding='utf-8') as a_f:
        for line in a_f.readlines():
            if line:
                split, file_name = line.strip().split('\t')
                if split == 'train':
                    train_files.append(file_name)
                elif split == 'val':
                    val_files.append(file_name)
                else:
                    test_files.append(file_name)

    create_split_files(Path(corpus_dir), split_dir, train_files, val_files, test_files)



if __name__=='__main__':
    """
    create_recipe2split_assignment('../recipe-generation/training/tuning_data_sets/ara1_amr_graphs',
                                   'ara1_data_split')
    create_recipe2split_assignment('../recipe-generation/training/tuning_data_sets/ara2_amr_graphs',
                                   'ara2_data_split')
    """

    create_split_files_from_assignment('./data_splits/ara1_data_split.tsv',
                                       '../recipe-generation/training/tuning_data_sets/ara1_amr_graphs',
                                       './data/ara1_amrs')
    create_split_files_from_assignment('./data_splits/ara2_data_split.tsv',
                                       '../recipe-generation/training/tuning_data_sets/ara2_amr_graphs',
                                       './data/ara2_amrs')
    create_split_files_from_assignment('./data_splits/ara1_data_split.tsv',
                                       '../recipe-generation/training/tuning_data_sets/ara1_amr_graphs',
                                       './data/ara1_2_amrs')
    create_split_files_from_assignment('./data_splits/ara2_data_split.tsv',
                                       '../recipe-generation/training/tuning_data_sets/ara2_amr_graphs',
                                       './data/ara1_2_amrs')
