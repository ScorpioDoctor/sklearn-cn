#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
线性回归案例
=========================================================
此示例仅使用糖尿病(`diabetes`)数据集的第一个特征，以说明此回归技术的二维绘图。
在图中可以看到直线，显示了线性回归如何试图绘制一条直线，
使数据集中观察到的响应与线性近似预测的响应之间的残差平方和最小化。

系数(The coefficients), 残差平方和(the residual sum of squares) 和 方差得分(the variance score)
也被计算出来了。

"""
print(__doc__)


# Code source: Jaques Grobler
# 翻译者: Antares 博士
# License: BSD 3 clause


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# 加载 diabetes 数据集
diabetes = datasets.load_diabetes()


# 仅使用第一个特征
diabetes_X = diabetes.data[:, np.newaxis, 2]

# 把数据划分成训练集和测试集
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# 把目标值划分成对应的训练集和测试集
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# 实例化一个 线性回归 类的对象
regr = linear_model.LinearRegression()

# 在训练集上训练模型
regr.fit(diabetes_X_train, diabetes_y_train)

# 在测试集上进行预测
diabetes_y_pred = regr.predict(diabetes_X_test)

# 线性模型的系数
print('Coefficients: \n', regr.coef_)
# 均方误差
print("Mean squared error: %.2f"
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# 解释方差: 1 代表完美预测
print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))

# 绘制输出结果
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
