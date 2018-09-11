# -*- coding: UTF-8 -*-

"""
date: 2018-9-11
note: decision tree
Ref:  《机器学习实战》- https://github.com/apachecn/AiLearning
"""

import operator
from math import log
from collections import Counter


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def calShannonEnt(dataSet):
    label_count = Counter(data[-1] for data in dataSet)
    probs = [p[1] / len(dataSet) for p in label_count.items()]
    shannonEnt = sum([-p * log(p, 2) for p in probs])
    return shannonEnt


def splitDataSet(dataSet, feature_index, value):
    retDataSet = [data[:feature_index] + data[feature_index+1: ] for data in dataSet \
                  for i, v in enumerate(data) if i == feature_index and v == value]
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    base_entropy = calShannonEnt(dataSet)
    best_info_gain = 0
    best_feature = -1
    # loop feature
    for i in range(len(dataSet[0]) - 1):
        feature_count = Counter(data[i] for data in dataSet)
        new_entropy = sum(feature[1] / float(len(dataSet)) \
                          * calShannonEnt(splitDataSet(dataSet, i, feature[0])) \
                          for feature in feature_count.items())

        info_gain = base_entropy - new_entropy
        print('No. {0} feature info gain is {1}'.format(i, info_gain))
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def majorityCnt(classList):
    major_label = Counter(classList).most_common(1)[0]
    print('major label', major_label)
    return major_label


def createTree(dataSet, labels):
    '''

    :param dataSet: 训练数据集
    :param labels:  训练数据集中特征对应的含义的labels，不是目标变量
    :return: myTree -- 创建完成的决策树
    '''
    classList = [data[-1] for data in dataSet]

    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    bestFeature = chooseBestFeatureToSplit(dataSet)
    bestFeatureLabel = labels[bestFeature]
    myTree = {bestFeatureLabel: {}}
    del(labels[bestFeature])
    featureValues = [data[bestFeature] for data in dataSet]
    uniqueValues = set(featureValues)
    for value in uniqueValues:
        subLabels = labels[:]
        myTree[bestFeatureLabel][value] = createTree(splitDataSet(dataSet, bestFeature, value), subLabels)
    print('my tree:', myTree)
    return myTree


def classify(inputTree, featureLabels, testVec):
    rootNodeKey = list(inputTree.keys())[0]
    rootNodeValue = inputTree[rootNodeKey]
    featuteIndex = featureLabels.index(rootNodeKey)
    key = testVec[featuteIndex]
    value = rootNodeValue[key]
    print('from ', rootNodeKey, ':', rootNodeValue, '---', key, '>>>', value)
    if isinstance(value, dict):
        classLabel = classify(value, featureLabels, testVec)
    else:
        classLabel = value
    return classLabel


def fishTest():
    import copy
    mydata, labels = createDataSet()
    myTree = createTree(mydata, copy.deepcopy(labels))
    classify(myTree, labels, [1, 1])


if __name__ == '__main__':
    fishTest()