import numpy as np
import pandas as pd


def evaluate(predict, label):
    '''
    这是评估函数
    '''
    acc = np.average([1 if i == j else 0 for i, j in zip(predict, label)])
    confusion_matrix = np.array([[0 for i in range(5)] for j in range(5)])
    for i, j in zip(predict, label):
        confusion_matrix[i][j] += 1

    recall = [confusion_matrix[i][i] / np.sum(confusion_matrix, axis=0)[i] for i in range(5)]
    precision = [confusion_matrix[i][i] / np.sum(confusion_matrix, axis=1)[i] for i in range(5)]

    confusion_matrix = pd.DataFrame(confusion_matrix, columns=[i for i in range(5)], index=[i for i in range(5)])
    confusion_matrix.loc[:, 'precision'] = precision
    confusion_matrix.loc['recall', :] = recall + ['/']

    return confusion_matrix, acc

