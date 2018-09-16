# -*- coding: UTF-8 -*-
"""
Date: 2018-9-13
Note: linear model learn 
Ref:  https://github.com/apachecn/hands_on_Ml_with_Sklearn_and_TF
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# plt.plot(X, y, "b.")
# plt.show()

'''正态方程计算'''
X_b = np.c_[np.ones((100, 1)), X]
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

print('正态方程计算\n theta_best:',theta_best)

X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2,1)), X_new]
print('pre:', X_new_b.dot(theta_best))


'''sklearn '''
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
print('\n\nsklearn 求解\nlin_reg.intercept_, lin_reg.coef_:', lin_reg.intercept_, lin_reg.coef_)
print('lin_reg.predict:', lin_reg.predict(X_new))


'''批量梯度下降'''
eta = 0.1
n_interations = 1000
m = len(X)
theta = np.random.randn(2, 1)

for iter in range(n_interations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients
    
print('\n\n批量梯度下降\n theta:', theta)


'''随机梯度下降'''
n_epochs = 50
t0, t1 = 5, 50    # learning rate schedule


def learning_schedule(t):
    return t0 / (t + t1)

theta = np.random.randn(2, 1)
for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule( epoch * m + i)
        theta = theta - eta * gradients

print('\n随机梯度下降\n theta:', theta)

'''随机梯度 sklearn实现'''
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(max_iter=50, penalty=None, eta0=0.1, random_state=42)
sgd_reg.fit(X, y.ravel())
print('\n随机梯度 sklearn实现\n', sgd_reg.intercept_, sgd_reg.coef_)


'''小批量梯度下降'''
n_interations = 50
minibatch_size = 20
theta = np.random.randn(2, 1)
t0, t1 = 200, 1000
t = 0
for epoch in range(n_interations):
    shuffled_indices = np.random.permutation(m)
    X_b_shuffled = X_b[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    for i in range(0, m, minibatch_size):
        t += 1
        xi = X_b_shuffled[i:i+minibatch_size]
        yi = y_shuffled[i:i+minibatch_size]
        gradients = 2/minibatch_size * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(t)
        theta = theta - eta * gradients

print('\n小批量梯度下降\n theta:', theta)


print('--------------------------------------------------------')
m = 100
X = 3 * np.random.rand(m, 1)
y = 0.5 * X + 2 + np.random.randn(m, 1)/2
X_new = np.linspace(0, 3, 100).reshape(100, 1)

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


def plot_model(model_class, polynomial, alphas, **model_kargs):
    for alpha, style in zip(alphas, ('b-', 'g--', 'r:')):
        model = model_class(alpha, **model_kargs) if alpha > 0 else LinearRegression()
        if polynomial:
            model = Pipeline([
                ('poly_features', PolynomialFeatures(degree=10, include_bias=False)),
                ('std_scalar', StandardScaler()),
                ('regular_reg', model)
            ])
        model.fit(X, y)
        y_new_regul = model.predict(X_new)
        lw = 2 if alpha > 0 else 1
        plt.plot(X_new, y_new_regul, style, linewidth=lw, label=r"$\alpha = {}$".format(alpha))
    plt.plot(X, y, 'b.', linewidth=3)
    plt.legend(loc="upper left", fontsize=5)

plt.figure(figsize=(8,4))
plt.subplot(221)
plot_model(Ridge, polynomial=False, alphas=(0, 10, 100))
plt.subplot(222)
plot_model(Ridge, polynomial=True, alphas=(0, 10**-5, 1))
# plt.show()

from sklearn.linear_model import Lasso

plt.subplot(223)
plot_model(Lasso, polynomial=False, alphas=(0, 0.1, 1), random_state=42)
plt.subplot(224)
plot_model(Lasso, polynomial=True, alphas=(0, 10**-7, 1), tol=1, random_state=42)

plt.show()