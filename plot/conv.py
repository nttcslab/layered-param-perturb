import matplotlib.pyplot as plt
import seaborn as sns
import read_dir_main


def main():
    plt.rcParams['text.usetex'] = False  # True
    plt.rcParams['ps.useafm'] = True
    plt.rcParams['pdf.use14corefonts'] = True
    plt.rcParams['font.size'] = 12
    sns.set_style('whitegrid')

    alldf, _ = read_dir_main.read_dir()
    dataset, se = 'MNIST', 1
    selected = alldf[(alldf.dataset==dataset) & (alldf.seed==se) & (alldf.K==64) & (alldf.L==64)]

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 2.5))
    g = sns.lineplot(data=selected, x='minute', y='loss', hue='method', ax=ax)
    g.set_xlabel('Elapsed time (minute)', fontsize=14)
    g.set(ylim=[0.17, 0.24])
    g.set(yticks=[0.18, 0.2, 0.22, 0.24], yticklabels=[0.18, 0.2, 0.22, 0.24])
    g.legend(loc='upper right')
    g.set_ylabel('Training loss', fontsize=14)
    fig.tight_layout()
    plt.savefig('../fig/conv.png', bbox_inches="tight", dpi=500)
    plt.savefig('../fig/conv.eps', bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    main()
