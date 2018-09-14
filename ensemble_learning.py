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
from sklearn.metrics import mean_squared_error
import numpy as np

''' 投票分类'''
X, y = make_moons(n_samples=500, noise=0.3)
X_train, X_test, y_train, y_test = train_test_split(X, y)

log_clf = LogisticRegression(random_state=42)
rnd_clf = RandomForestClassifier(random_state=42)
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


'''random forests'''
rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1, random_state=42)
rnd_clf.fit(X_train, y_train)
y_pres_rf = rnd_clf.predict(X_test)

print('\n random forests: ', accuracy_score(y_test, y_pres_rf))


np.random.seed(42)
X = np.random.rand(100, 1) - 0.5
y = 3*X[:, 0]**2 + 0.05 * np.random.randn(100)
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=49)

'''GradientBoosting'''
from sklearn.ensemble import GradientBoostingRegressor
gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, random_state=42)
gbrt.fit(X_train, y_train)


'''early stopping'''
gbrt = GradientBoostingRegressor(max_depth=2, warm_start=True, random_state=42)
min_val_error = float('inf')
error_going_up = 0
for n_nestimators in range(1, 120):
     gbrt.n_estimators = n_nestimators
     gbrt.fit(X_train, y_train)
     y_pred = gbrt.predict(X_val)
     val_error = mean_squared_error(y_val, y_pred)
     if val_error < min_val_error:
         min_val_error = val_error
         error_going_up = 0
     else:
         error_going_up += 1
         if error_going_up == 5:
             break
print('\nearly stopping: gbrt.n_estimators:', gbrt.n_estimators)
print('MSE:', min_val_error)





