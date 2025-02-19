import matplotlib.pyplot as plt
import seaborn as sns
import read_dir_main
from statannotations.Annotator import Annotator


def get_pairs(layer_list, methods):
    pairs = []
    num_methods = len(methods)
    for layer in layer_list:
        for i in range(num_methods):
            for j in range(i+1, num_methods):
                pairs.append(((layer, methods[i]), (layer, methods[j])))
    return pairs


def subplot(ax, data, layer_list):
    g = sns.boxplot(data=data, x='L', y='loss', hue='method', ax=ax, width=.5)
    methods = ['ZO-$\\mathbf{0}$'.format('{I}'), 'ZO-$\\mathbf{co}$', 'ZO-$\\mathbf{0}$'.format('{\\Sigma}_u')]
    pairs = get_pairs(layer_list, methods)
    annotator = Annotator(ax, pairs, data=data, x='L', y='loss', hue='method')
    annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
    annotator.apply_and_annotate()
    return g


def main():
    _, lastdf = read_dir_main.read_dir()

    plt.rcParams['text.usetex'] = False  # True
    plt.rcParams['pdf.use14corefonts'] = True
    plt.rcParams['font.size'] = 11
    sns.set_style('whitegrid')

    fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(5.4, 10))
    g = subplot(ax[0, 0], lastdf[(lastdf.K==16) & (lastdf.dataset=='MNIST')], [8, 12, 16])
    g.legend(loc='upper right')
    g.set(xlabel='', title='MNIST', yticks=[0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.74])
    g.set_ylabel('Training loss, K=16', fontsize=14)
    g = subplot(ax[1, 0], lastdf[(lastdf.K==32) & (lastdf.dataset=='MNIST')], [16, 24, 32])
    g.legend([],[], frameon=False)
    g.set(xlabel='', yticks=[0.26, 0.28, 0.30, 0.32, 0.34])
    g.set_ylabel('Training loss, K=32', fontsize=14)
    g = subplot(ax[2, 0], lastdf[(lastdf.K==64) & (lastdf.dataset=='MNIST')], [32, 48, 64])
    g.legend([],[], frameon=False)
    g.set_xlabel('L', fontsize=14)
    g.set_ylabel('Training loss, K=64', fontsize=14)
    g.set(yticks=[0.18, 0.20, 0.22])
    g = subplot(ax[0, 1], lastdf[(lastdf.K==16) & (lastdf.dataset=='FMNIST')], [8, 12, 16])
    g.set(ylabel='', xlabel='', title='FashionMNIST', yticks=[1.00, 1.02, 1.04, 1.06, 1.08, 1.10])
    g.legend(loc='upper right')
    g = subplot(ax[1, 1], lastdf[(lastdf.K==32) & (lastdf.dataset=='FMNIST')], [16, 24, 32])
    g.legend([],[], frameon=False)
    g.set(ylabel='', xlabel='', yticks=[0.60, 0.62, 0.64, 0.66, 0.68])
    g = subplot(ax[2, 1], lastdf[(lastdf.K==64) & (lastdf.dataset=='FMNIST')], [32, 48, 64])
    g.set_xlabel('L', fontsize=14)
    g.legend([],[], frameon=False)
    g.set(ylabel='', yticks=[0.46, 0.48, 0.50, 0.52])

    fig.tight_layout()
    plt.savefig('../fig/loss.png', bbox_inches="tight", dpi=500)
    plt.show()


if __name__ == '__main__':
    main()
