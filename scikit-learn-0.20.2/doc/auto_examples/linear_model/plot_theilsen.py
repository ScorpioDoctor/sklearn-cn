"""
====================
Theil-Sen Regression
====================

在人工合成的数据集上计算 Theil-Sen 回归。

请看 :ref:`theil_sen_regression` 获得关于回归器的更多信息。

与 OLS (最小二乘) 估计器相比, Theil-Sen 估计器是对离群孤立点比较鲁棒的估计器。
在一个简单2D线性回归问题中，它有大约29.3%的崩溃点(breakdown point)， 这就意味着它可以忍受任意的
损坏数据(outliers)的占比可以达到29.3%。

模型的估计是通过计算 p 个子样本点的所有可能组合的子种群的斜率和截取来完成的。
如果截距(intercept)也被拟合了，则 p 必须大于或等于 n_features + 1。
最终的斜率和截距被定义为这些斜率和截取的空间中值。

在某些情形下 Theil-Sen 比另一个鲁棒算法 :ref:`RANSAC <ransac_regression>` 表现的更好。
这一点在下面的第二个案例中进行了说明，其中相对于x轴的离群点使 RANSAC 不鲁棒。
调整RANSAC的 ``residual_threshold`` 参数可以弥补这一点，但是一般来说，需要对数据和异常值的性质有一个先验的认识。
至于Theil-Sen算法的计算复杂性，建议只能使用在样本数量和特征数量比较小的问题上。
对于大问题，参数 ``max_subpopulation`` 限制p子采样点的所有可能的组合到一个随机选择的子集，并因此会限制运行时间。
因此，Theil-Sen 方法是可以用于更大的问题的，但会损失一些它的数学性质，因为在大问题上它是在随机子集上工作的。
"""

# Author: Florian Wilhelm -- <florian.wilhelm@gmail.com>
# License: BSD 3 clause

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, TheilSenRegressor
from sklearn.linear_model import RANSACRegressor

print(__doc__)

estimators = [('OLS', LinearRegression()),
              ('Theil-Sen', TheilSenRegressor(random_state=42)),
              ('RANSAC', RANSACRegressor(random_state=42)), ]
colors = {'OLS': 'turquoise', 'Theil-Sen': 'gold', 'RANSAC': 'lightgreen'}
lw = 2

# #############################################################################
# 只有 y 方向有离群点

np.random.seed(0)
n_samples = 200
# 线性模型 y = 3*x + N(2, 0.1**2)
x = np.random.randn(n_samples)
w = 3.
c = 2.
noise = 0.1 * np.random.randn(n_samples)
y = w * x + c + noise
# 10% outliers
y[-20:] += -20 * x[-20:]
X = x[:, np.newaxis]

plt.scatter(x, y, color='indigo', marker='x', s=40)
line_x = np.array([-3, 3])
for name, estimator in estimators:
    t0 = time.time()
    estimator.fit(X, y)
    elapsed_time = time.time() - t0
    y_pred = estimator.predict(line_x.reshape(2, 1))
    plt.plot(line_x, y_pred, color=colors[name], linewidth=lw,
             label='%s (fit time: %.2fs)' % (name, elapsed_time))

plt.axis('tight')
plt.legend(loc='upper left')
plt.title("Corrupt y")

# #############################################################################
# X 方向上有离群点

np.random.seed(0)
# 线性模型 y = 3*x + N(2, 0.1**2)
x = np.random.randn(n_samples)
noise = 0.1 * np.random.randn(n_samples)
y = 3 * x + 2 + noise
# 10% outliers
x[-20:] = 9.9
y[-20:] += 22
X = x[:, np.newaxis]

plt.figure()
plt.scatter(x, y, color='indigo', marker='x', s=40)

line_x = np.array([-3, 10])
for name, estimator in estimators:
    t0 = time.time()
    estimator.fit(X, y)
    elapsed_time = time.time() - t0
    y_pred = estimator.predict(line_x.reshape(2, 1))
    plt.plot(line_x, y_pred, color=colors[name], linewidth=lw,
             label='%s (fit time: %.2fs)' % (name, elapsed_time))

plt.axis('tight')
plt.legend(loc='upper left')
plt.title("Corrupt x")
plt.show()
