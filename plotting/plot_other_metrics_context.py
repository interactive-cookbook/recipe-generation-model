import matplotlib.pyplot as plt
from plot_helpers import get_rouge_per_context, get_meteor_per_context, get_bleurt_per_context


def plot_rouge_per_context3_epochs():

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

    data_model1 = get_rouge_per_context(files_model1)
    data_model2 = get_rouge_per_context(files_model2)
    data_model3 = get_rouge_per_context(files_model3)
    data_model4 = get_rouge_per_context(files_model4)
    data_model5 = get_rouge_per_context(files_model5)
    data_model6 = get_rouge_per_context(files_model6)

    baseline1_0 = {'rouge1': [(0, 0.7195), (1, 0.7195), (2, 0.7195), (3, 0.7195)],
                   'rouge2': [(0, 0.3751), (1, 0.3751), (2, 0.3751), (3, 0.3751)],
                   'rougel': [(0, 0.6479), (1, 0.6479), (2, 0.6479), (3, 0.6479)]}
    baseline2_0 = {'rouge1': [(0, 0.7311), (1, 0.7311), (2, 0.7311), (3, 0.7311)],
                   'rouge2': [(0, 0.4212), (1, 0.4212), (2, 0.4212), (3, 0.4212)],
                   'rougel': [(0, 0.6668), (1, 0.6668), (2, 0.6668), (3, 0.6668)]}

    fig = plt.figure(figsize=(20, 30))
    ax = fig.add_subplot(311)
    ax.plot(*zip(*data_model1['rouge1']), 'C0o--', label='ara1_split_3_ara1_split_5ep')
    ax.plot(*zip(*data_model5['rouge1']), 'C0x--', label='ara1_split_3_ara1_split_10ep')
    ax.plot(*zip(*data_model3['rouge1']), 'C0v--', label='ara1_split_3_ara1_split_30ep')
    ax.plot(*zip(*baseline1_0['rouge1']), 'C0--', label='no-finetuning_ara1_split')
    ax.plot(*zip(*data_model2['rouge1']), 'C1o--', label='ara1_2_split_3_ara1_2_split_5ep')
    ax.plot(*zip(*data_model6['rouge1']), 'C1x--', label='ara1_2_split_3_ara1_2_split_10ep')
    ax.plot(*zip(*data_model4['rouge1']), 'C1v--', label='ara1_2_split_3_ara1_2_split_30ep')
    ax.plot(*zip(*baseline2_0['rouge1']), 'C1--', label='no-finetuning_ara1_2_split')

    ax.legend(bbox_to_anchor=(0.9, 0.5), loc="upper right", prop={'size': 14})
    ax.set_xlabel(f'Test context length')
    ax.set_ylabel(f'ROUGE1 score')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)

    ax2 = fig.add_subplot(312)
    ax2.plot(*zip(*data_model1['rouge2']), 'C0o--', label='ara1_split_3_ara1_split_5ep')
    ax2.plot(*zip(*data_model5['rouge2']), 'C0x--', label='ara1_split_3_ara1_split_10ep')
    ax2.plot(*zip(*data_model3['rouge2']), 'C0v--', label='ara1_split_3_ara1_split_30ep')
    ax2.plot(*zip(*baseline1_0['rouge2']), 'C0--', label='no-finetuning_ara1_split')
    ax2.plot(*zip(*data_model2['rouge2']), 'C1o--', label='ara1_2_split_3_ara1_2_split_5ep')
    ax2.plot(*zip(*data_model6['rouge2']), 'C1x--', label='ara1_2_split_3_ara1_2_split_10ep')
    ax2.plot(*zip(*data_model4['rouge2']), 'C1v--', label='ara1_2_split_3_ara1_2_split_30ep')
    ax2.plot(*zip(*baseline2_0['rouge2']), 'C1--', label='no-finetuning_ara1_2_split')

    ax2.legend(bbox_to_anchor=(0.9, 0.5), loc="upper right", prop={'size': 14})
    ax2.set_xlabel(f'Test context length')
    ax2.set_ylabel(f'ROUGE2 score')
    for item in ([ax2.title, ax2.xaxis.label, ax2.yaxis.label] +
                 ax2.get_xticklabels() + ax2.get_yticklabels()):
        item.set_fontsize(15)

    ax3 = fig.add_subplot(313)
    ax3.plot(*zip(*data_model1['rougel']), 'C0o--', label='ara1_split_3_ara1_split_5ep')
    ax3.plot(*zip(*data_model5['rougel']), 'C0x--', label='ara1_split_3_ara1_split_10ep')
    ax3.plot(*zip(*data_model3['rougel']), 'C0v--', label='ara1_split_3_ara1_split_30ep')
    ax3.plot(*zip(*baseline1_0['rougel']), 'C0--', label='no-finetuning_ara1_split')
    ax3.plot(*zip(*data_model2['rougel']), 'C1o--', label='ara1_2_split_3_ara1_2_split_5ep')
    ax3.plot(*zip(*data_model6['rougel']), 'C1x--', label='ara1_2_split_3_ara1_2_split_10ep')
    ax3.plot(*zip(*data_model4['rougel']), 'C1v--', label='ara1_2_split_3_ara1_2_split_30ep')
    ax3.plot(*zip(*baseline2_0['rougel']), 'C1--', label='no-finetuning_ara1_2_split')

    ax3.legend(bbox_to_anchor=(0.9, 0.5), loc="upper right", prop={'size': 14})
    ax3.set_xlabel(f'Test context length')
    ax3.set_ylabel(f'ROUGEL score')
    for item in ([ax3.title, ax3.xaxis.label, ax3.yaxis.label] +
                 ax3.get_xticklabels() + ax3.get_yticklabels()):
        item.set_fontsize(15)

    plt.show()


def plot_meteor_per_context3_epochs():

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

    data_model1 = get_meteor_per_context(files_model1)
    data_model2 = get_meteor_per_context(files_model2)
    data_model3 = get_meteor_per_context(files_model3)
    data_model4 = get_meteor_per_context(files_model4)
    data_model5 = get_meteor_per_context(files_model5)
    data_model6 = get_meteor_per_context(files_model6)

    baseline1_0 = [(0, 0.6742), (1, 0.6742), (2, 0.6742), (3, 0.6742)]
    baseline2_0 = [(0, 0.7015), (1, 0.7015), (2, 0.7015), (3, 0.7015)]

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
    ax.set_ylabel(f'METEOR score')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)

    plt.show()


def plot_bleurt_per_context3_epochs():

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

    data_model1 = get_bleurt_per_context(files_model1)
    data_model2 = get_bleurt_per_context(files_model2)
    data_model3 = get_bleurt_per_context(files_model3)
    data_model4 = get_bleurt_per_context(files_model4)
    data_model5 = get_bleurt_per_context(files_model5)
    data_model6 = get_bleurt_per_context(files_model6)

    baseline1_0 = [(0, 0.3660), (1, 0.3660), (2, 0.3660), (3, 0.3660)]
    baseline2_0 = [(0, 0.4265), (1, 0.4265), (2, 0.4265), (3, 0.4265)]

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
    ax.set_ylabel(f'BLEURT score')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)

    plt.show()




if __name__=='__main__':

    #plot_rouge_per_context3_epochs()
    plot_bleurt_per_context3_epochs()
