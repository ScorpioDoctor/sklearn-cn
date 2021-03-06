"""
=====================================================
在MNIST分类任务中使用 multinomial logistic + L1
=====================================================

这里我们在MNIST手写数字分类任务的一个子集上拟合了一个带有L1惩罚的多项式logistic回归模型。
我们使用 SAGA算法 做求解器的原因是：SAGA 在样本数量显著大于特征数量的时候非常快，
而且可以很好地优化带有L1惩罚项的非平滑目标函数。
测试准确率达到 0.8 以上，而权重向量还可以保持稀疏化，
这样得到的模型就更具有可解释性(*interpretable*)了。

请注意：这个L1惩罚的线性模型的准确率是明显低于L2惩罚的线性模型能够达到的准确率
或是一个非线性多层感知器能够达到的准确率的。

"""
import time
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

print(__doc__)

# Author: Arthur Mensch <arthur.mensch@m4x.org>
# License: BSD 3 clause

# Turn down for faster convergence
t0 = time.time()
train_samples = 5000

# 从这儿 https://www.openml.org/d/554 加载数据
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

# 对数据进行随机洗牌(shuffle data)
random_state = check_random_state(0)
permutation = random_state.permutation(X.shape[0])
X = X[permutation]
y = y[permutation]
X = X.reshape((X.shape[0], -1))

# 分成 训练集 与 测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=train_samples, test_size=10000)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 上调 tolerance 可以更快的收敛
clf = LogisticRegression(C=50. / train_samples,
                         multi_class='multinomial',
                         penalty='l1', solver='saga', tol=0.1)
clf.fit(X_train, y_train)
sparsity = np.mean(clf.coef_ == 0) * 100
score = clf.score(X_test, y_test)
# print('Best C % .4f' % clf.C_)
print("Sparsity with L1 penalty: %.2f%%" % sparsity)
print("Test score with L1 penalty: %.4f" % score)

coef = clf.coef_.copy()
plt.figure(figsize=(10, 5))
scale = np.abs(coef).max()
for i in range(10):
    l1_plot = plt.subplot(2, 5, i + 1)
    l1_plot.imshow(coef[i].reshape(28, 28), interpolation='nearest',
                   cmap=plt.cm.RdBu, vmin=-scale, vmax=scale)
    l1_plot.set_xticks(())
    l1_plot.set_yticks(())
    l1_plot.set_xlabel('Class %i' % i)
plt.suptitle('Classification vector for...')

run_time = time.time() - t0
print('Example run in %.3f s' % run_time)
plt.show()
