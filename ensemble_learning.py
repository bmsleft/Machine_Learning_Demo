# -*- coding: UTF-8 -*-
"""
Date: 2018-9-13
Note: ensemble learning
Ref:  https://github.com/killakalle/ageron_handson-ml
"""

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

''' 投票分类'''
X, y = make_moons(n_samples=500, noise=0.3)
X_train, X_test, y_train, y_test = train_test_split(X, y)

log_clf = LogisticRegression(random_state=42)
rnd_clf =  RandomForestClassifier(random_state=42)
svm_clf = SVC(random_state=42)

voting_clf = VotingClassifier(estimators=[
    ('lr', log_clf),
    ('rf', rnd_clf),
    ('svc', svm_clf)], voting='hard')

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))


'''bagging ensembles'''
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(DecisionTreeClassifier(random_state=42), n_estimators=500,
                            max_samples=100, bootstrap=True, n_jobs=1, random_state=42)
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
print('\n bagging: ', accuracy_score(y_test, y_pred))









