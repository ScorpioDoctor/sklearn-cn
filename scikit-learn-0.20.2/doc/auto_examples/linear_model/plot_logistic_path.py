#!/usr/bin/env python
"""
==============================================
L1- Logistic 回归的正则化路径
==============================================


在一个来自于Iris数据集的二分类问题上 训练带有L1惩罚的logistic回归模型。

模型按照从最强的正则化到最弱的正则化进行排序。模型的4个系数被收集起来并作为正则化路径
(regularization path)绘制出来:在图像的左边(强正则化)，所有的系数都是0。当正则化
逐渐变得松弛时，模型的系数就能够一个接一个的获得非零值。

这里我们选择了 SAGA 求解器，因为它可以高效的优化带有非平滑，稀疏诱导的L1惩罚项的Logistic回归损失 。

另外需要注意的是 我们为 tolerance(tol) 设置一个较低的值来确保在收集系数之前模型已经收敛。

我们也使用了 warm_start=True , 这意味着 模型的系数被重复使用来初始化下一个模型从而加速全路径(full-path)的计算。

"""
print(__doc__)

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause
# 翻译者 ： Antares@studyai.com

from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn import datasets
from sklearn.svm import l1_min_c

iris = datasets.load_iris()
X = iris.data
y = iris.target

# 鸢尾花数据集总共有3个类，这里只保留前两个类y=0，1
# 剔除第三个类的样本和类标签
X = X[y != 2]
y = y[y != 2]

X /= X.max()  # 归一化 X 来加速收敛

# #############################################################################
# Demo path functions

cs = l1_min_c(X, y, loss='log') * np.logspace(0, 7, 16)


print("Computing regularization path ...")
start = time()
clf = linear_model.LogisticRegression(penalty='l1', solver='saga',
                                      tol=1e-6, max_iter=int(1e6),
                                      warm_start=True)
coefs_ = []
for c in cs:
    clf.set_params(C=c)
    clf.fit(X, y)
    coefs_.append(clf.coef_.ravel().copy())
print("This took %0.3fs" % (time() - start))

coefs_ = np.array(coefs_)
plt.plot(np.log10(cs), coefs_, marker='o')
ymin, ymax = plt.ylim()
plt.xlabel('log(C)')
plt.ylabel('Coefficients')
plt.title('Logistic Regression Path')
plt.axis('tight')
plt.show()
