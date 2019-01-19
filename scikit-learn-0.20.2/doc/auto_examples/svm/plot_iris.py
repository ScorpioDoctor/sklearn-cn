"""
==================================================
在鸢尾花数据集上绘制使用不同类型SVM分类器的分类结果
==================================================

不同线性SVM分类器在鸢尾花数据集的二维投影上的比较。我们只考虑这个数据集的前两个特征：

- 花萼长度(Sepal length)
- 花萼宽度(Sepal width)

这个例子展示了如何绘制四个具有不同核的SVM分类器的决策面。

线性模型  ``LinearSVC()`` 和 ``SVC(kernel='linear')`` 产生的决策边界略有不同。
这可能是以下差异导致的结果：

- ``LinearSVC`` 最小化平方折页损失(squared hinge loss) 而 ``SVC`` 最小化常规的折页损失(regular hinge loss)。

- ``LinearSVC`` 使用 One-vs-All (也被称为 One-vs-Rest) 多类分类策略，而 ``SVC`` 使用 One-vs-One 多类分类策略。

两个线性模型都具有线性决策面(相交的超平面)，而带有非线性核(多项式核与高斯径向基核)的SVM有更加灵活的非线性决策边界，
这些决策边界的shapes依赖于核的种类和其参数。

.. NOTE:: 尽管绘制分类器在2D迷你数据集上的决策函数有助于我们获得对这些分类器的表达能力的直观理解，
   但是这些直观印象不能总是推广到更真实的高维空间中的问题上去。

翻译者：http://studyai.com/antares
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


# import some data to play with
iris = datasets.load_iris()
# Take the first two features. We could avoid this by using a two-dim dataset
X = iris.data[:, :2]
y = iris.target

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
models = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C),
          svm.SVC(kernel='rbf', gamma=0.7, C=C),
          svm.SVC(kernel='poly', degree=3, C=C))
models = (clf.fit(X, y) for clf in models)

# title for the plots
titles = ('SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel')

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()
