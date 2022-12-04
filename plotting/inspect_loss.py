import os.path

import matplotlib.pyplot as plt
import json
from pathlib import Path


def plot_train_eval_loss(log_file):

    log_file_name = os.path.join(Path(log_file)).split(os.sep)[-1]

    train_loss_dict, eval_loss_dict = read_loss_from_log(log_file)
    train_loss = [(ep, loss) for (ep, loss) in train_loss_dict.items()]
    train_loss.sort(key=lambda x: x[0])
    eval_loss = [(ep, loss) for (ep, loss) in eval_loss_dict.items()]
    eval_loss.sort(key=lambda x: x[0])

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    ax.plot(*zip(*train_loss), 'r', label='train')
    ax.plot(*zip(*eval_loss), 'b', label='val')
    ax.legend()
    fig.suptitle(f'Loss from training {log_file_name}')
    ax.set_xlabel(f'Epochs')
    ax.set_ylabel(f'Loss')

    """
    plt.plot(*zip(*train_loss), 'r', label='train')
    plt.plot(*zip(*eval_loss), 'b', label='val')
    plt.legend()
    plt.title(f'Loss from training {log_file_name}')
    plt.xlabel(f'Epochs')
    plt.ylabel(f'Loss')
    """

    plt.show()


def read_loss_from_log(log_file):
    train_loss_dict = dict()
    eval_loss_dict = dict()

    with open(log_file, 'r', encoding='utf-8') as log:
        for line in log.readlines():
            if line.startswith('{'):
                line = line.replace("'", '"')
                loss_data = json.loads(line)
                if 'loss' in loss_data.keys():
                    epoch = loss_data['epoch']
                    train_loss_dict[epoch] = loss_data['loss']
                elif 'eval_loss' in loss_data.keys():
                    epoch = loss_data['epoch']
                    eval_loss_dict[epoch] = loss_data['eval_loss']
    return train_loss_dict, eval_loss_dict



if __name__=='__main__':

    #plot_train_eval_loss('./logs/amrlib_ara1_split_1_100_ep.txt')
    #plot_train_eval_loss('./logs/amrlib_ara1_split_1_100ep_drop.txt')
    plot_train_eval_loss('../logs/amrlib_ara1_2_split_1_100ep.txt')
