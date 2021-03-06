#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
鸢尾花数据集(Iris Dataset)
=========================================================
该数据集包含了三种不同品种的鸢尾花(Setosa, Versicolour, 和 Virginica) 。
每个鸢尾花样本上采集四个特征: 花瓣(petal)的长度,宽度 和 花萼(sepal)的长度,宽度，
总共150个样本，所以最终的数据存储为 150x4 numpy.ndarray。

numpy.ndarray 的每一行对应一个样本，而四个特征构成的四列的顺序为:
Sepal Length, Sepal Width, Petal Length , Petal Width .

下面这个图只是用了前两个特征画出来的。
请看 `这儿 <https://en.wikipedia.org/wiki/Iris_flower_data_set>`_ 
获得此数据集的更多信息。
"""
print(__doc__)


# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# 翻译 和 测试 by Antares博士
# License: BSD 3 clause

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA

# 导入数据用来玩耍
iris = datasets.load_iris()
X = iris.data[:, :2]  # 列切片索引：只取前两个特征.
y = iris.target

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

plt.figure(2, figsize=(8, 6))
plt.clf()

# 绘制训练数据点
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,
            edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

# 为了对数据的各个维度之间的相互作用有一个更好地理解，
# 绘制前三个 PCA dimensions。
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(iris.data)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,
           cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

plt.show()


# #####################################################################################
#                               Antares的测试信息
# 可能会发生 UnicodeDecodeError: 
#     'gbk' codec can't decode byte 0x86 in position 41: illegal multibyte sequence
# 解决方法是 根据 Traceback 信息，找到 base.py 里面的 load_iris() 函数，第390行
# 我们定位到这一行，发现是读取 rst 文件的一段代码，
#             with open(join(module_path, 'descr', 'digits.rst')) as f:
#                   descr = f.read()
# 我们只要将上面这段代码打开文件的格式改成: 'rb' 即以只读二进制方式打开就好了
#          with open(join(module_path, 'descr', 'digits.rst'),'rb') as f:


# ################################################################################################