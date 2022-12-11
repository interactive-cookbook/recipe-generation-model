import matplotlib.pyplot as plt
from plot_helpers import get_bleu_per_epoch


def plot_bleu_per_epoch():

    files_model1 = ['../output/t5_amrlib_ara1_split_1/1_context/output_ara1_split_beam_1_90_evaluation.txt',
                    '../output/t5_amrlib_ara1_split_1_10ep/1_context/output_ara1_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_split_1_30ep/1_context/output_ara1_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_split_1_60ep/1_context/output_ara1_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_split_1_100ep/1_context/output_ara1_split_beam_1_evaluation.txt']

    files_model2 = ['../output/t5_amrlib_ara1_split_1/0_context/output_ara1_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_split_1_10ep/0_context/output_ara1_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_split_1_30ep/0_context/output_ara1_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_split_1_60ep/0_context/output_ara1_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_split_1_100ep/0_context/output_ara1_split_beam_1_evaluation.txt']

    files_model3 = ['../output/t5_amrlib_ara1_2_split_1/1_context/output_ara1_2_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_2_split_1_10ep/1_context/output_ara1_2_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_2_split_1_30ep/1_context/output_ara1_2_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_2_split_1_60ep/1_context/output_ara1_2_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_2_split_1_100ep/1_context/output_ara1_2_split_beam_1_evaluation.txt']

    files_model4 = ['../output/t5_amrlib_ara1_2_split_1/0_context/output_ara1_2_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_2_split_1_10ep/0_context/output_ara1_2_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_2_split_1_30ep/0_context/output_ara1_2_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_2_split_1_60ep/0_context/output_ara1_2_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_2_split_1_100ep/0_context/output_ara1_2_split_beam_1_evaluation.txt']

    files_model5 = ['../output/t5_amrlib_ara1_split_1/1_context/output_ara2_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_split_1_10ep/1_context/output_ara2_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_split_1_30ep/1_context/output_ara2_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_split_1_60ep/1_context/output_ara2_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_split_1_100ep/1_context/output_ara2_split_beam_1_evaluation.txt']

    data_model1 = get_bleu_per_epoch(files_model1)
    data_model2 = get_bleu_per_epoch(files_model2)
    data_model3 = get_bleu_per_epoch(files_model3)
    data_model4 = get_bleu_per_epoch(files_model4)
    data_model5 = get_bleu_per_epoch(files_model5)

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    ax.plot(*zip(*data_model1), 'r+--', label='ara1_split_1_ara1_split_1')
    ax.plot(*zip(*data_model2), 'ro--', label='ara1_split_1_ara1_split_0')
    ax.plot(*zip(*data_model3), 'b+--', label='ara1_2_split_1_ara1_2_split_1')
    ax.plot(*zip(*data_model4), 'bo--', label='ara1_2_split_1_ara1_2_split_0')
    ax.plot(*zip(*data_model5), 'g+--', label='ara1_split_1_ara2_split_1')
    ax.legend(bbox_to_anchor=(1.1, 0.2), loc="upper right", prop={'size': 14})
    ax.set_xlabel(f'Train Epochs')
    ax.set_ylabel(f'BLEU score')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)

    plt.show()
    #fig.savefig('image_output.png', dpi=300, format='png', bbox_extra_artists=(lgd,), bbox_inches='tight')


def plot_bleu_per_epoch_drop():

    files_model1 = ['../output/t5_amrlib_ara1_split_1_60ep/1_context/output_ara1_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_split_1_100ep/1_context/output_ara1_split_beam_1_evaluation.txt']
    files_model1_drop = ['../output/t5_amrlib_ara1_split_1_60ep_drop/1_context/output_ara1_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_split_1_100ep_drop/1_context/output_ara1_split_beam_1_evaluation.txt']

    files_model2 = ['../output/t5_amrlib_ara1_split_1_60ep/0_context/output_ara1_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_split_1_100ep/0_context/output_ara1_split_beam_1_evaluation.txt']
    files_model2_drop = ['../output/t5_amrlib_ara1_split_1_60ep_drop/0_context/output_ara1_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_split_1_100ep_drop/0_context/output_ara1_split_beam_1_evaluation.txt']

    files_model3 = ['../output/t5_amrlib_ara1_2_split_1_60ep/1_context/output_ara1_2_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_2_split_1_100ep/1_context/output_ara1_2_split_beam_1_evaluation.txt']

    files_model3_drop = ['../output/t5_amrlib_ara1_2_split_1_60ep_drop/1_context/output_ara1_2_split_beam_1_evaluation.txt',
                         '../output/t5_amrlib_ara1_2_split_1_100ep_drop/1_context/output_ara1_2_split_beam_1_evaluation.txt']

    files_model4 = ['../output/t5_amrlib_ara1_2_split_1_60ep/0_context/output_ara1_2_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_2_split_1_100ep/0_context/output_ara1_2_split_beam_1_evaluation.txt']

    files_model4_drop = ['../output/t5_amrlib_ara1_2_split_1_60ep_drop/0_context/output_ara1_2_split_beam_1_evaluation.txt',
                         '../output/t5_amrlib_ara1_2_split_1_100ep_drop/0_context/output_ara1_2_split_beam_1_evaluation.txt']

    files_model5 = ['../output/t5_amrlib_ara1_split_1_60ep/1_context/output_ara2_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_split_1_100ep/1_context/output_ara2_split_beam_1_evaluation.txt']

    files_model5_drop = ['../output/t5_amrlib_ara1_split_1_60ep_drop/1_context/output_ara2_split_beam_1_evaluation.txt',
                         '../output/t5_amrlib_ara1_split_1_100ep_drop/1_context/output_ara2_split_beam_1_evaluation.txt']

    data_model1 = get_bleu_per_epoch(files_model1)
    data_model1_drop = get_bleu_per_epoch(files_model1_drop)
    data_model2 = get_bleu_per_epoch(files_model2)
    data_model2_drop = get_bleu_per_epoch(files_model2_drop)
    data_model3 = get_bleu_per_epoch(files_model3)
    data_model3_drop = get_bleu_per_epoch(files_model3_drop)
    data_model4 = get_bleu_per_epoch(files_model4)
    data_model4_drop = get_bleu_per_epoch(files_model4_drop)
    data_model5 = get_bleu_per_epoch(files_model5)
    data_model5_drop = get_bleu_per_epoch(files_model5_drop)

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)

    ax.plot(*zip(*data_model1), 'ro--', label='ara1_split_1_ara1_split_1')
    plt.plot(*zip(*data_model1_drop), 'rv--', label='ara1_split_1_ara1_split_1_drop')

    ax.plot(*zip(*data_model2), 'mo--', label='ara1_split_1_ara1_split_0')
    ax.plot(*zip(*data_model2_drop), 'mv--', label='ara1_split_1_ara1_split_0_drop')

    ax.plot(*zip(*data_model3), 'bo--', label='ara1_2_split_1_ara1_2_split_1')
    ax.plot(*zip(*data_model3_drop), 'bv--', label='ara1_2_split_1_ara1_2_split_1_drop')

    ax.plot(*zip(*data_model4), 'co--', label='ara1_2_split_1_ara1_2_split_0')
    plt.plot(*zip(*data_model4_drop), 'cv--', label='ara1_2_split_1_ara1_2_split_0_drop')

    ax.plot(*zip(*data_model5), 'go--', label='ara1_split_1_ara2_split_1')
    ax.plot(*zip(*data_model5_drop), 'gv--', label='ara1_split_1_ara2_split_1_drop')

    ax.legend(bbox_to_anchor=(1.1, 0.22), loc="upper right", prop={'size': 12})

    ax.set_xlabel(f'Train Epochs')
    ax.set_ylabel(f'BLEU score')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)

    plt.show()


if __name__=='__main__':

    plot_bleu_per_epoch_drop()