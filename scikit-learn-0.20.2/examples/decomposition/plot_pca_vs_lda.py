"""
=======================================================
LDA与PCA在鸢尾花数据集上的二维投影的比较
=======================================================

鸢尾花(Iris)数据集代表了3种鸢尾花(Setosa、Versicolour和Virica)的4个属性：
萼片(sepal)长度、萼片宽度、花瓣(petal)长度和花瓣宽度。

应用于该数据的主成分分析(PCA)确定了属性(主成分，或特征空间中的方向)的组合，
这些属性在数据中的方差最大。在这里，我们以前两个主成分绘制了不同的样本。

线性判别分析(LDA)试图识别 **类间差异最大** 的属性。
特别是，与PCA相比，LDA是一种监督方法，使用已知的类标签。

翻译者：http://www.studyai.com/antares
"""
print(__doc__)

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

iris = datasets.load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names

pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(X, y).transform(X)

# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset')

plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of IRIS dataset')

plt.show()
