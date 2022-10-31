import numpy as np
import matplotlib.pyplot as plt

def auc_chart():
    N = 9
    ind = np.arange(N)
    width = 0.25

    xvals = [0.4, 0.5, 0.549, 0.55, 0.65, 0.673, 0.499, 0.4, 0.65]
    bar1 = plt.bar(ind, xvals, width, color='r')

    yvals = [0.71153, 0.847, 0.7832, 0.8286, 0.786, 0.673, 0.708, 0.805, 0.87]
    bar2 = plt.bar(ind + width, yvals, width, color='g')

    zvals = [0.604, 0.8333, 0.7291, 0.8333, 0.75, 0.7083, 0.75, 0.833, 0.916]
    bar3 = plt.bar(ind + width * 2, zvals, width, color='b')

    plt.xlabel("Method")
    plt.ylabel('AUC Score')
    plt.title("Data balancing AUC performance")

    plt.xticks(ind + width, ['NB', 'DT', 'RF', 'XGB', 'ADA', 'LR', 'SVM', 'BM', 'Our'])
    plt.legend((bar1, bar2, bar3), ('Downsampling', 'Oversampling', 'SMOTE'))
    plt.show()


def cohen_chart():
    N = 9
    ind = np.arange(N)
    width = 0.25

    xvals = [0.4368,0.612,0.5744,0.6619,0.577,0.354,0.432,0.618,0.707]
    bar1 = plt.bar(ind, xvals, width, color='r')

    yvals = [0.436,0.7031,0.574,0.661,0.577,0.3548,0.43272,0.618,0.7482]
    bar2 = plt.bar(ind + width, yvals, width, color='g')

    zvals = [0.2083,0.6666,0.458,0.666,0.5,0.41666,0.5,0.666,0.833]
    bar3 = plt.bar(ind + width * 2, zvals, width, color='b')

    plt.xlabel("Method")
    plt.ylabel('Cohen\'s Kapp Score')
    plt.title("Data balancing results using Cohen performance")

    plt.xticks(ind + width, ['NB', 'DT', 'RF', 'XGB', 'ADA', 'LR', 'SVM', 'BM', 'Our'])
    plt.legend((bar1, bar2, bar3), ('Downsampling', 'Oversampling', 'SMOTE'))
    plt.show()

cohen_chart()