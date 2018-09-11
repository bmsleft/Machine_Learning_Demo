# -*- coding: UTF-8 -*-
"""
Date: 2018-9-11
Note: KNN classification, basic implement and sk-learn demo
Ref:  《机器学习实战》- https://github.com/apachecn/AiLearning
"""

import numpy as np
import matplotlib.pyplot as plt
import collections
from sklearn import neighbors

datingDataFilePath = "./data/datingTestSet2.txt"


def file2matrix(filename):
    with open(filename) as fr:
        numberOfLines = len(fr.readlines())
        returnMat = np.zeros((numberOfLines, 3))
        classLabelVector = []
        fr.seek(0)
        index = 0
        for line in fr.readlines():
            line = line.strip()
            listFromLine = line.split('\t')
            returnMat[index, :] = listFromLine[0:3]
            classLabelVector.append(int(listFromLine[-1]))
            index += 1

    return returnMat, classLabelVector


def autoNorm(dataSet):
    """
    :param dataSet:
    :return: normDataSet, ranges, minVals
    归一化数据，Y = (X-Xmin)/(Xmax-Xmin)
    """
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    # # the first method
    # normDataSet = np.zeros(dataSet.shape)
    # m = dataSet.shape[0]
    # normDataSet = dataSet - np.tile(minVals, (m, 1))
    # normDataSet = normDataSet / np.tile(ranges, (m, 1))
    # # the first method end

    # the second method
    normDataSet = (dataSet - minVals) / ranges

    return normDataSet, ranges, minVals


def draw_data(dataMat, labels):
    # 按照label画出数据散点图
    # 为了便于查看数据，只取前两个属性作为X Y值
    fig = plt.figure()
    labels_color ={
        1 : 'r',
        2 : 'g',
        3 : 'b'
    }
    colors = list(map(lambda x:labels_color[x], labels))
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 0], dataMat[:, 1], s=15.0, c=colors)
    plt.show()


def knn_classify(inX, dataSet, labels, k):
    # dist: 欧氏距离
    dist = np.sum((inX - dataSet)**2, axis=1) ** 0.5
    k_labels = [labels[index] for index in dist.argsort()[0:k]]
    label = collections.Counter(k_labels).most_common(1)
    # print(label)
    return label[0][0]


def basic_knn_test():
    # using first 20% as test case
    hoRation = 0.2
    datingDataMat, datingLabels = file2matrix(datingDataFilePath)
    # print(datingDataMat, datingLabels)
    datingDataMat, _, _ = autoNorm(datingDataMat)

    # draw_data(datingDataMat, datingLabels)
    m = datingDataMat.shape[0]
    numTestVecs = int(m * hoRation)
    print('Test case count:', numTestVecs)
    errorCnt = 0
    for i in range(numTestVecs):
        classifyResult = knn_classify(datingDataMat[i], datingDataMat[numTestVecs:m], datingLabels[numTestVecs:m], 3)
        # print('calssify result: %d, the real answer: %d' %(classifyResult, datingLabels[i]))
        errorCnt +=  classifyResult != datingLabels[i]
    print('error count :', errorCnt)
    print('the total error rate :', errorCnt / numTestVecs)


def sklearn_knn_test():
    X, Y = file2matrix(datingDataFilePath)
    X, _, _ = autoNorm(X)

    hoRation = 0.2
    m =  X.shape[0]
    numTestVecs = int(m * hoRation)
    print('Test case count:', numTestVecs)

    clf = neighbors.KNeighborsClassifier(n_neighbors=3, weights='distance')
    clf.fit(X[numTestVecs:m], Y[numTestVecs:m])
    errorCnt = 0
    X_pre = clf.predict(X[:numTestVecs])
    for index, value in enumerate(X_pre):
        errorCnt += value != Y[index]
    print('error count :', errorCnt)
    print('the total error rate :', errorCnt / numTestVecs)


if __name__ == '__main__':
    print('--------------------------')
    print('basic knn')
    basic_knn_test()
    print('--------------------------')
    print('sk-learn knn')
    sklearn_knn_test()

