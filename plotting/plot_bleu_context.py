import matplotlib.pyplot as plt

from plot_helpers import get_bleu_per_context


def plot_bleu_per_context1():

    files_model1 = ['../output/t5_amrlib_ara1_split_1/0_context/output_ara1_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_split_1/1_context/output_ara1_split_beam_1_evaluation.txt']

    files_model2 = ['../output/t5_amrlib_ara1_split_3/0_context/output_ara1_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_split_3/1_context/output_ara1_split_beam_1_evaluation.txt']

    files_model3 = ['../output/t5_amrlib_ara1_2_split_1/0_context/output_ara1_2_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_2_split_1/1_context/output_ara1_2_split_beam_1_evaluation.txt']

    files_model4 = ['../output/t5_amrlib_ara1_2_split_3/0_context/output_ara1_2_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_2_split_3/1_context/output_ara1_2_split_beam_1_evaluation.txt']

    files_model5 = ['../output/t5_amrlib_ara1_split_1/0_context/output_ara1_orig_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_split_1/1_context/output_ara1_orig_beam_1_evaluation.txt']

    files_model6 = ['../output/t5_amrlib_ara1_split_1_na/0_context/output_ara1_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_split_1_na/1_context/output_ara1_split_beam_1_na_evaluation.txt']

    files_model7 = ['../output/t5_amrlib_ara1_2_split_1_na/0_context/output_ara1_2_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_2_split_1_na/1_context/output_ara1_2_split_beam_1_na_evaluation.txt']

    files_model8 = ['../output/t5_amrlib_ara2_split_1/0_context/output_ara2_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara2_split_1/1_context/output_ara2_split_beam_1_evaluation.txt']

    files_model9 = ['../output/t5_amrlib_ara2_split_1/0_context/output_ara1_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara2_split_1/1_context/output_ara1_split_beam_1_evaluation.txt']

    files_model10 = ['../output/t5_amrlib_ara2_split_1/0_context/output_ara1_2_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara2_split_1/1_context/output_ara1_2_split_beam_1_evaluation.txt']

    data_model1 = get_bleu_per_context(files_model1)
    data_model2 = get_bleu_per_context(files_model2)
    data_model3 = get_bleu_per_context(files_model3)
    data_model4 = get_bleu_per_context(files_model4)
    data_model5 = get_bleu_per_context(files_model5)
    data_model6 = get_bleu_per_context(files_model6)
    data_model7 = get_bleu_per_context(files_model7)
    data_model8 = get_bleu_per_context(files_model8)
    data_model9 = get_bleu_per_context(files_model9)
    data_model10 = get_bleu_per_context(files_model10)

    baseline1 = [(0, 49.37), (1, 49.37)]
    baseline1_0 = [(0, 38.23), (1, 38.23)]
    baseline2 = [(0, 47.85), (1, 47.85)]
    baseline2_0 = [(0, 39.81), (1, 39.81)]
    baseline3_0 = [(0, 41.01), (1, 41.01)]

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(131)
    ax.plot(*zip(*data_model1), '', label='ara1_split_1_ara1_split')
    ax.plot(*zip(*data_model2), '', label='ara1_split_3_ara1_split')
    ax.plot(*zip(*data_model6), '', label='ara1_split_1_na_ara1_split')
    #ax.plot(*zip(*data_model5), '', label='ara1_split_1_ara1_orig')
    ax.plot(*zip(*baseline1), '--', label='ara1_split_0_ara1_split')
    ax.plot(*zip(*baseline1_0), '--', label='no-finetuning_ara1_split')
    ax.legend(bbox_to_anchor=(0.9, 0.5), loc="upper right", prop={'size': 14})
    ax.set_xlabel(f'Test context length')
    ax.set_ylabel(f'BLEU score')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)

    ax2 = fig.add_subplot(132, sharey=ax)
    ax2.plot(*zip(*data_model3), '', label='ara1_2_split_1_ara1_split')
    ax2.plot(*zip(*data_model4), '', label='ara1_2_split_3_ara1_split')
    ax2.plot(*zip(*data_model7), '', label='ara1_2_split_1_na_ara1_2_split')
    ax2.plot(*zip(*baseline2), '--', label='ara1_2_split_0_ara1_2_split')
    ax2.plot(*zip(*baseline2_0), '--', label='no-finetuning_ara1_2_split')
    ax2.legend(bbox_to_anchor=(0.9, 0.4), loc="upper right", prop={'size': 14})
    ax2.set_xlabel(f'Test context length')
    ax2.set_ylabel(f'BLEU score')
    for item in ([ax2.title, ax2.xaxis.label, ax2.yaxis.label] +
                 ax2.get_xticklabels() + ax2.get_yticklabels()):
        item.set_fontsize(15)

    ax3 = fig.add_subplot(133, sharey=ax)
    ax3.plot(*zip(*data_model8), '', label='ara2_split_1_ara2_split')
    ax3.plot(*zip(*data_model9), '', label='ara2_split_1_ara1_split')
    ax3.plot(*zip(*data_model10), '', label='ara2_split_1_ara1_2_split')
    ax3.plot(*zip(*baseline3_0), 'C4--', label='no-finetuning_ara2_split')
    ax3.legend(bbox_to_anchor=(0.9, 0.5), loc="upper right", prop={'size': 14})
    ax3.set_xlabel(f'Test context length')
    ax3.set_ylabel(f'BLEU score')
    for item in ([ax3.title, ax3.xaxis.label, ax3.yaxis.label] +
                 ax3.get_xticklabels() + ax3.get_yticklabels()):
        item.set_fontsize(15)

    plt.show()


def plot_bleu_per_context3():

    files_model1 = ['../output/t5_amrlib_ara1_split_3/0_context/output_ara1_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_split_3/1_context/output_ara1_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_split_3/2_context/output_ara1_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_split_3/3_context/output_ara1_split_beam_1_evaluation.txt']

    files_model2 = ['../output/t5_amrlib_ara1_2_split_3/0_context/output_ara1_2_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_2_split_3/1_context/output_ara1_2_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_2_split_3/2_context/output_ara1_2_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_2_split_3/3_context/output_ara1_2_split_beam_1_evaluation.txt']



    data_model1 = get_bleu_per_context(files_model1)
    data_model2 = get_bleu_per_context(files_model2)

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    ax.plot(*zip(*data_model1), '', label='ara1_split_3_ara1_split')
    ax.plot(*zip(*data_model2), '', label='ara1_2_split_3_ara1_2_split')

    ax.legend(bbox_to_anchor=(0.9, 0.2), loc="upper right", prop={'size': 14})
    ax.set_xlabel(f'Test context length')
    ax.set_ylabel(f'BLEU score')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)

    plt.show()


def plot_bleu_per_context3_epochs():

    files_model1 = ['../output/t5_amrlib_ara1_split_3/0_context/output_ara1_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_split_3/1_context/output_ara1_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_split_3/2_context/output_ara1_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_split_3/3_context/output_ara1_split_beam_1_evaluation.txt']

    files_model2 = ['../output/t5_amrlib_ara1_2_split_3/0_context/output_ara1_2_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_2_split_3/1_context/output_ara1_2_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_2_split_3/2_context/output_ara1_2_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_2_split_3/3_context/output_ara1_2_split_beam_1_evaluation.txt']

    files_model3 = ['../output/t5_amrlib_ara1_split_3_30ep/0_context/output_ara1_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_split_3_30ep/1_context/output_ara1_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_split_3_30ep/2_context/output_ara1_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_split_3_30ep/3_context/output_ara1_split_beam_1_evaluation.txt']

    files_model4 = ['../output/t5_amrlib_ara1_2_split_3_30ep/0_context/output_ara1_2_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_2_split_3_30ep/1_context/output_ara1_2_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_2_split_3_30ep/2_context/output_ara1_2_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_2_split_3_30ep/3_context/output_ara1_2_split_beam_1_evaluation.txt']

    files_model5 = ['../output/t5_amrlib_ara1_split_3_10ep/0_context/output_ara1_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_split_3_10ep/1_context/output_ara1_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_split_3_10ep/2_context/output_ara1_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_split_3_10ep/3_context/output_ara1_split_beam_1_evaluation.txt']

    files_model6 = ['../output/t5_amrlib_ara1_2_split_3_10ep/0_context/output_ara1_2_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_2_split_3_10ep/1_context/output_ara1_2_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_2_split_3_10ep/2_context/output_ara1_2_split_beam_1_evaluation.txt',
                    '../output/t5_amrlib_ara1_2_split_3_10ep/3_context/output_ara1_2_split_beam_1_evaluation.txt']

    data_model1 = get_bleu_per_context(files_model1)
    data_model2 = get_bleu_per_context(files_model2)
    data_model3 = get_bleu_per_context(files_model3)
    data_model4 = get_bleu_per_context(files_model4)
    data_model5 = get_bleu_per_context(files_model5)
    data_model6 = get_bleu_per_context(files_model6)

    baseline1_0 = [(0, 38.23), (1, 38.23), (2, 38.23), (3, 38.23)]
    baseline2_0 = [(0, 39.81), (1, 39.81), (2, 39.81), (3, 39.81)]

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    ax.plot(*zip(*data_model1), 'C0o--', label='ara1_split_3_ara1_split_5ep')
    ax.plot(*zip(*data_model5), 'C0x--', label='ara1_split_3_ara1_split_10ep')
    ax.plot(*zip(*data_model3), 'C0v--', label='ara1_split_3_ara1_split_30ep')
    ax.plot(*zip(*baseline1_0), 'C0--', label='no-finetuning_ara1_split')
    ax.plot(*zip(*data_model2), 'C1o--', label='ara1_2_split_3_ara1_2_split_5ep')
    ax.plot(*zip(*data_model6), 'C1x--', label='ara1_2_split_3_ara1_2_split_10ep')
    ax.plot(*zip(*data_model4), 'C1v--', label='ara1_2_split_3_ara1_2_split_30ep')
    ax.plot(*zip(*baseline2_0), 'C1--', label='no-finetuning_ara1_2_split')

    ax.legend(bbox_to_anchor=(0.9, 0.5), loc="upper right", prop={'size': 14})
    ax.set_xlabel(f'Test context length')
    ax.set_ylabel(f'BLEU score')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)

    plt.show()


if __name__=='__main__':

    #plot_bleu_per_context1()
    #plot_bleu_per_context3()
    plot_bleu_per_context3_epochs()
