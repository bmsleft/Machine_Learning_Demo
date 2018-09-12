# -*- coding: UTF-8 -*-

"""
date: 2018-9-11
note: decision tree classification sklearn version
Ref:  《机器学习实战》- https://github.com/apachecn/AiLearning
"""

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def trainTree(x_train, y_train):
    tree_clf = DecisionTreeClassifier(criterion='entropy', max_depth=2)
    tree_clf.fit(x_train, y_train)
    # print('feature_importances_: %s' % tree_clf.feature_importances_)
    return tree_clf


def show_precision_recall(clf, x_test, y_test):
    y_pred = clf.predict(x_test)
    print('accuracy:', accuracy_score(y_test, y_pred))
    print('confusion_matrix:\n', confusion_matrix(y_test, y_pred))
    print('classification_report: \n', classification_report(y_test, y_pred))


if __name__ == '__main__':
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
    clf = trainTree(x_train, y_train)
    show_precision_recall(clf, x_test, y_test)


