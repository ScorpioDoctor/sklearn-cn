"""
===========================================================
绘制 岭系数作为正则化量的函数的曲线图
===========================================================

展示共线性(collinearity)对估计器系数的影响

.. currentmodule:: sklearn.linear_model

这个例子中用到的模型是岭回归估计器(:class:`Ridge`)。
每种颜色表示系数向量的不同特征，并将其显示为正则化参数的函数。

此示例还显示了将岭回归应用于高度病态(ill-conditioned)矩阵的有效性。
对于这样的矩阵，目标变量的微小变化会导致计算出的权重的巨大差异。
在这种情况下，设置一定的正则化(alpha)来减少这种变化(噪声)是很有用的。

当 alpha 很大时，正则化效应将会主导(控制)平方损失函数，线性模型的系数也将趋于零。
在路径的末尾，当 alpha 趋于零时，解趋于普通最小二乘解时，系数会出现很大的振荡。
在实践中，有必要对 alpha 进行调优，以便在两者之间保持平衡。

总共有10个系数，10条曲线，他们是一一对应的。
"""

# Author: Fabian Pedregosa -- <fabian.pedregosa@inria.fr>
# License: BSD 3 clause
# 翻译者：studyai.com的Antares博士

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# X 是一个 10x10 的 希尔伯特矩阵(Hilbert matrix)
X = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
y = np.ones(10)

# #############################################################################
# 计算路径(Compute paths)

n_alphas = 200
alphas = np.logspace(-10, -2, n_alphas)

coefs = []
for a in alphas:
    ridge = linear_model.Ridge(alpha=a, fit_intercept=False)    #每个循环都要重新实例化一个estimator对象
    ridge.fit(X, y)
    coefs.append(ridge.coef_)
# print(coefs)

# #############################################################################
# 展示结果
ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # 反转数轴，越靠左边 alpha 越大，正则化也越厉害
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()
