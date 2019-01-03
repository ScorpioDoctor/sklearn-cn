.. currentmodule:: sklearn.feature_selection

.. _feature_selection:

===================================
特征选择(Feature selection)
===================================


特征选择模块 :mod:`sklearn.feature_selection` 中的类可以用于样本集的特征选择/降维，
既可以提高估计器的精度得分，也可以提高它们在非常高维数据集上的性能。


.. _variance_threshold:

去除方差比较低的特征
===================================

方差阈值 :class:`VarianceThreshold` 是特征选择的一种简单的基线方法（baseline approach）。
它删除了所有方差不满足某些阈值的特征。默认情况下，它删除所有零方差特征(zero-variance features)，即在所有样本中具有相同值的特征。

例如，假设我们有一个具有若干布尔特征(boolean features)的数据集，并且我们希望删除所有在超过80%的样本中取值都为1或0(ON或OFF)的那些布尔特征。
布尔特征是伯努利随机变量(Bernoulli random variables), 这种类型的随机变量的方差 由下面给出：

.. math:: \mathrm{Var}[X] = p(1 - p)

因此我们可以使用阈值 ``.8 * (1 - .8)`` 进行选择 ::

  >>> from sklearn.feature_selection import VarianceThreshold
  >>> X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
  >>> sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
  >>> sel.fit_transform(X)
  array([[0, 1],
         [1, 0],
         [0, 0],
         [1, 1],
         [1, 0],
         [1, 1]])

就像我希望的那样, ``VarianceThreshold`` 已经将第一列删除了, 因为第一列包含0值的概率(样本比例)是 :math:`p = 5/6 > .8` ，超过了给定的阈值 0.8。

.. _univariate_feature_selection:

单变量特征选择
============================

单变量特征选择是通过选择那些基于单变量统计检验(univariate statistical tests)得出的的最优特征来实现的。
它可以看作是一个预处理步骤。
Scikit-learn 将一系列特征选择程序作为不同的类提供给我们，这些类都实现了  ``transform`` 方法:

 * :class:`SelectKBest` 选择得分最高的 :math:`k` 个特征，删除其余的。

 * :class:`SelectPercentile` 选择得分最高的前百分之几的特征，这个百分比由用户指定，其余的特征全部删除

 * 把常见的 单变量统计检验方法 用到每个特征上：
   false positive rate :class:`SelectFpr`, false discovery rate
   :class:`SelectFdr`, 或 family wise error :class:`SelectFwe`.

 * :class:`GenericUnivariateSelect` 允许使用一个可配置策略(a configurable strategy)进行单变量特征选择。该类允许使用超参数搜索估计器
  (hyper-parameter search estimator)选择最优的单变量选择策略。

举个例子, 我们可以对样本执行一个 :math:`\chi^2` 测试来仅仅挑选出两个最好的特征 ::

  >>> from sklearn.datasets import load_iris
  >>> from sklearn.feature_selection import SelectKBest
  >>> from sklearn.feature_selection import chi2
  >>> iris = load_iris()
  >>> X, y = iris.data, iris.target
  >>> X.shape
  (150, 4)
  >>> X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
  >>> X_new.shape
  (150, 2)

上面这些对象除了 :class:`SelectKBest` 和 :class:`SelectPercentile` 都接受两个参数作为输入：一个是 返回值为单变量之得分的评分函数，另一个是 p-values。
:class:`SelectKBest` 和 :class:`SelectPercentile`对象只接受一个参数：返回值为单变量之得分的评分函数:

 * 对于回归问题: :func:`f_regression`, :func:`mutual_info_regression`

 * 对于分类问题: :func:`chi2`, :func:`f_classif`, :func:`mutual_info_classif`

基于 F-test 的方法 估计 两个随机变量之间的线性依赖度(linear dependency)。 另一方面，基于互信息(mutual information)的方法可以捕捉任何类型的
统计依赖性(statistical dependency), 但由于互信息方法是无参数方法，他们需要更多的样本进行准确的估计。

.. topic:: 稀疏数据的特征选择

   如果你用的是稀疏数据 (i.e. 数据是以稀疏矩阵的形式存放的), :func:`chi2`, :func:`mutual_info_regression`, :func:`mutual_info_classif`
   可以在不用把数据变为稠密矩阵的前提下使用这些稀疏矩阵。

.. warning::

    小心不要把回归评分函数用在分类问题上，你会得到无用的结果。

.. topic:: 案例:

    * :ref:`sphx_glr_auto_examples_feature_selection_plot_feature_selection.py`

    * :ref:`sphx_glr_auto_examples_feature_selection_plot_f_test_vs_mi.py`

.. _rfe:

递归式特征消除
=============================

给定一个可以对特征向量赋予对应权重向量（比如，线性模型的相关系数）的外部估计器，，
recursive feature elimination ( :class:`RFE` ) 通过递归地考虑越来越小的特征集合来选择特征。 
首先，估计器在初始的特征集合上训练并且每一个特征的重要程度是通过一个 ``coef_`` 属性 
或者 ``feature_importances_`` 属性来获得。 然后，从当前的特征集合中移除最不重要的特征。
在特征集合上不断的重复递归这个步骤，直到最终达到所需要的特征数量为止。 
RFECV 在一个交叉验证的循环中执行 RFE 来找到最优的特征数量。

:class:`RFECV` 在一个交叉验证的循环中执行 RFE 来找到最优的特征数量。

.. topic:: 案例:

    * :ref:`sphx_glr_auto_examples_feature_selection_plot_rfe_digits.py`: A recursive feature elimination example
      showing the relevance of pixels in a digit classification task.

    * :ref:`sphx_glr_auto_examples_feature_selection_plot_rfe_with_cross_validation.py`: A recursive feature
      elimination example with automatic tuning of the number of features
      selected with cross-validation.

.. _select_from_model:

使用 SelectFromModel 选取特征
=======================================

:class:`SelectFromModel` 是一个 元变换器(meta-transformer), 可以和任意拟合后具有属性 ``coef_`` 或 ``feature_importances_`` 的估计器一起使用。
如果与某个特征对应的 ``coef_`` 或 ``feature_importances_`` 的值小于某个给定的阈值参数 ``threshold`` ，则认为该特征是不重要的，应该被去除。
除了以数字的方式指定阈值，还有一些内建的启发式方法可以用来寻找合适的阈值，这些方法用一个字符串做参数来指定具体的启发式策略。现在可用的启发式策略有
"mean", "median" 以及 用浮点数乘以字符串的方式，比如 "0.1*mean"。

关于该类的具体使用方法的案例请看下面。

.. topic:: 案例

    * :ref:`sphx_glr_auto_examples_feature_selection_plot_select_from_model_boston.py`: Selecting the two
      most important features from the Boston dataset without knowing the
      threshold beforehand.

.. _l1_feature_selection:

基于 L1 的特征选取
--------------------------

.. currentmodule:: sklearn

用 L1-norm 进行惩罚的线性模型 :ref:`Linear models <linear_model>` 可以获得稀疏解： 它们估计出的模型的很多系数都是0。
当我们的目标是使用另一个分类器对数据进行维数约简的时候，这样的分类器就可以和类 :class:`feature_selection.SelectFromModel` 
一起使用来选择非零的系数。特别的，可以用做这种用途的稀疏估计器(sparse estimators)有这些: :class:`linear_model.Lasso` 用于回归, 
以及 :class:`linear_model.LogisticRegression` 和 :class:`svm.LinearSVC` 用于分类 ::

  >>> from sklearn.svm import LinearSVC
  >>> from sklearn.datasets import load_iris
  >>> from sklearn.feature_selection import SelectFromModel
  >>> iris = load_iris()
  >>> X, y = iris.data, iris.target
  >>> X.shape
  (150, 4)
  >>> lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
  >>> model = SelectFromModel(lsvc, prefit=True)
  >>> X_new = model.transform(X)
  >>> X_new.shape
  (150, 3)

在 SVM 和 logistic-regression 中，参数 C 是用来控制稀疏性的：越小的 C 会导致越少的特征被选择。
在 Lasso 中，参数 alpha 的值越大，越少的特征会被选择。

.. topic:: 案例:

    * :ref:`sphx_glr_auto_examples_text_plot_document_classification_20newsgroups.py`: Comparison
      of different algorithms for document classification including L1-based
      feature selection.

.. _compressive_sensing:

.. topic:: **L1-recovery 和 压缩感知(compressive sensing)**

   当选择了正确的 alpha 值以后，:ref:`lasso` 可以仅通过少量观察点便完整
   的恢复准确的非零变量集合，假设特定的条件可以被满足的话。
   特别的，数据量需要 “足够大(sufficiently large)” ，不然 L1 模型的表现将充满不确定性。 
   “足够大” 的定义取决于非零系数的个数、特征数量的对数值、噪音的数量、非零系数的最小绝对值、 
   以及设计矩阵(design maxtrix) X 的结构。另外，设计矩阵必须有某些特定的性质，如数据不能过度相关。

   关于如何选择 alpha 的值来恢复非零系数并没有通用的规则。alpha 值可以通过交叉验证来确定
   (:class:`LassoCV` 或 :class:`LassoLarsCV`)，
   尽管这可能会导致欠惩罚的模型：包括少量的无关变量对于预测值来说并非致命的。相反的， BIC( :class:`LassoLarsIC` )
   倾向于给定高 alpha 值。
   
   **参考文献** Richard G. Baraniuk "Compressive Sensing", IEEE Signal
   Processing Magazine [120] July 2007
   http://users.isr.ist.utl.pt/~aguiar/CS_notes.pdf


基于树的特征选择
----------------------------

很多基于树的估计器 (请看 :mod:`sklearn.tree` 模块 和 :mod:`sklearn.ensemble` 模块中由树构成的森林那一小节) 可被用来计算特征重要性
(feature importances), 反过来，它们也可以用于丢弃那些无关的特征
(irrelevant features：译者注：这里的无关特征或者翻译为不相关的特征指的是该特征与分类或回归的目标变量没有啥关系，说的并不是各个特征分量之间的关系)
(当于 :class:`sklearn.feature_selection.SelectFromModel` meta-transformer 相耦合的时候)::

  >>> from sklearn.ensemble import ExtraTreesClassifier
  >>> from sklearn.datasets import load_iris
  >>> from sklearn.feature_selection import SelectFromModel
  >>> iris = load_iris()
  >>> X, y = iris.data, iris.target
  >>> X.shape
  (150, 4)
  >>> clf = ExtraTreesClassifier(n_estimators=50)
  >>> clf = clf.fit(X, y)
  >>> clf.feature_importances_  # doctest: +SKIP
  array([ 0.04...,  0.05...,  0.4...,  0.4...])
  >>> model = SelectFromModel(clf, prefit=True)
  >>> X_new = model.transform(X)
  >>> X_new.shape               # doctest: +SKIP
  (150, 2)

.. topic:: 案例:

    * :ref:`sphx_glr_auto_examples_ensemble_plot_forest_importances.py`: example on
      synthetic data showing the recovery of the actually meaningful
      features.

    * :ref:`sphx_glr_auto_examples_ensemble_plot_forest_importances_faces.py`: example
      on face recognition data.

把特征选择作为管道流的一部分
=======================================

在进行实际学习之前，通常使用特征选择作为预处理步骤。在 scikit-learn  中，推荐的方法是使用流水线类 
:class:`sklearn.pipeline.Pipeline` ::

  clf = Pipeline([
    ('feature_selection', SelectFromModel(LinearSVC(penalty="l1"))),
    ('classification', RandomForestClassifier())
  ])
  clf.fit(X, y)

在上面的代码片段中，我们将 :class:`sklearn.svm.LinearSVC` 类 和 
:class:`sklearn.feature_selection.SelectFromModel` 类耦合起来，评估特征重要性并且选择那些最相关的特征(most relevant features)。
然后，:class:`sklearn.ensemble.RandomForestClassifier` 类就紧跟在特征选择的输出端接收数据进行训练(i.e. 随机森林分类器只在最相关的特征上训练)。
在上面的代码片段中你可以选择其他的特征选择器类，也可以选择别的分类器进行特征重要性评估。
请参考 :class:`sklearn.pipeline.Pipeline` 类的更多案例。
