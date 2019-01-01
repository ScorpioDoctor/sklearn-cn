.. _learning_curves:

=====================================================
验证曲线：绘制得分曲线评估模型
=====================================================

.. currentmodule:: sklearn.model_selection

.. topic:: 译者注

    再次献上我做的视频，希望对大家有所帮助。视频地址：
    (`Sklearn模型验证方法 <http://www.studyai.com/course/play/0d82ca1df87d48e181f4c4edbd464ed6>`_)

每种估计器都有其优势和缺陷。它的泛化误差(generalization error)可以用偏差(bias)、
方差(variance)和噪声(noise)来分解。估计器的 **偏差** 是它在不同训练集上的平均误差。
估计器的 **方差** 用来表示它对训练集的变化有多敏感。**噪声** 是数据的一个属性。

在下面的图中，我们可以看到一个函数 :math:`f(x) = \cos (\frac{3}{2} \pi x)` 和这个函数的一些带噪声样本。
我们用三个不同的估计器(estimators)来拟合函数：阶数分别为1,4和15的多项式特征的线性回归。
我们看到，第一个估计器最多只能为样本和真正的函数提供一个很差的拟合 ，因为这个模型太简单了(高偏差），
第二个估计器的预测结果几乎完全逼近真实曲线，最后一个估计器的结果完美逼近所有训练数据点，但不能很好地拟合
真实的函数，即对训练数据的变化非常敏感（高方差）。

.. figure:: ../auto_examples/model_selection/images/sphx_glr_plot_underfitting_overfitting_001.png
   :target: ../auto_examples/model_selection/plot_underfitting_overfitting.html
   :align: center
   :scale: 50%

偏差和方差是估计器所固有的属性，我们通常必须选择合适的学习算法和超参数，以使得偏差和方差都尽可能的低
（参见偏差-方差困境 `Bias-variance dilemma <https://en.wikipedia.org/wiki/Bias-variance_dilemma>`_)。 
另一种降低方差的方法是使用更多的训练数据。不论如何，如果真实函数过于复杂以至于不能用一个方差较小的估计器来逼近，
则只能去收集更多的训练数据。

在一个简单的一维问题中，我们可以很容易看出估计器是否存在偏差或方差。然而，在高维空间中， 
模型可能变得非常难以可视化。 出于这种原因，使用以下工具通常是有帮助的。

.. topic:: 案例:

   * :ref:`sphx_glr_auto_examples_model_selection_plot_underfitting_overfitting.py`
   * :ref:`sphx_glr_auto_examples_model_selection_plot_validation_curve.py`
   * :ref:`sphx_glr_auto_examples_model_selection_plot_learning_curve.py`


.. _validation_curve:

验证曲线(Validation curve)
================

我们需要一个评分函数（参见 :ref:`model_evaluation`）来验证一个模型， 例如分类器的准确性。 
选择估计器的多个超参数的正确方法当然是网格搜索或类似方法 （参见 :ref:`grid_search`），
网格搜索会选择在一个或多个验证集上的分数最高的超参数。 请注意，如果我们基于验证分数优化了超参数，
则验证分数就有偏差了，并且不再是一个良好的泛化估计。 为了得到正确的泛化估计，我们必须在另一个测试集上计算得分。

然而，绘制单个超参数对训练分数和验证分数的影响，有时有助于发现该估计器是否因为某些超参数的值 而出现过拟合或欠拟合。

本例中,下面的方程 :func:`validation_curve` 能起到如下作用::

  >>> import numpy as np
  >>> from sklearn.model_selection import validation_curve
  >>> from sklearn.datasets import load_iris
  >>> from sklearn.linear_model import Ridge

  >>> np.random.seed(0)
  >>> iris = load_iris()
  >>> X, y = iris.data, iris.target
  >>> indices = np.arange(y.shape[0])
  >>> np.random.shuffle(indices)
  >>> X, y = X[indices], y[indices]

  >>> train_scores, valid_scores = validation_curve(Ridge(), X, y, "alpha",
  ...                                               np.logspace(-7, 3, 3),
  ...                                               cv=5)
  >>> train_scores            # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
  array([[0.93..., 0.94..., 0.92..., 0.91..., 0.92...],
         [0.93..., 0.94..., 0.92..., 0.91..., 0.92...],
         [0.51..., 0.52..., 0.49..., 0.47..., 0.49...]])
  >>> valid_scores           # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
  array([[0.90..., 0.84..., 0.94..., 0.96..., 0.93...],
         [0.90..., 0.84..., 0.94..., 0.96..., 0.93...],
         [0.46..., 0.25..., 0.50..., 0.49..., 0.52...]])

如果训练得分和验证得分都很低，则估计器是欠拟合的。如果训练得分高，验证得分低，则估计器发生了过拟合。 
如果两种得分都很高则说明估计器拟合得很好。较低的训练得分和较高的验证得分这种情况通常是不会发生的。
所有三种情况都可以 在下面的图中找到， 其中我们改变了数字数据集上 SVM 的参数 :math:`\gamma` 。

.. figure:: ../auto_examples/model_selection/images/sphx_glr_plot_validation_curve_001.png
   :target: ../auto_examples/model_selection/plot_validation_curve.html
   :align: center
   :scale: 50%


.. _learning_curve:

学习曲线(Learning curve)
==============

学习曲线显示了在不同数量的训练样本下估计器的验证得分和训练得分。它可以帮助我们发现从增加更多的训 
练数据中能获益多少，以及估计是否受到更多来自方差误差或偏差误差的影响。如果在增加训练集大小时，
验证分数和训练分数都收敛到一个很低的值，那么我们即使添加更多的训练数据也很难从中获益。在下面的图中你可以看到一个例子：
朴素贝叶斯大致收敛到一个较低的分数。

.. figure:: ../auto_examples/model_selection/images/sphx_glr_plot_learning_curve_001.png
   :target: ../auto_examples/model_selection/plot_learning_curve.html
   :align: center
   :scale: 50%

我们可能需要使用estimator或者当前estimator的一个参数化形式来学习更复杂概念（i.e. 有一个较低的偏差）。 
如果训练样本的数量达到最大时，训练集得分比验证集得分大得多，那么增加更多的训练样本很可能会增加泛化能力。

.. figure:: ../auto_examples/model_selection/images/sphx_glr_plot_learning_curve_002.png
   :target: ../auto_examples/model_selection/plot_learning_curve.html
   :align: center
   :scale: 50%

我们可以使用 :func:`learning_curve` 函数来产生绘制这样一个学习曲线所需的值
（已使用的样本数量，训练集上的平均得分和验证集上的平均得分） ::

  >>> from sklearn.model_selection import learning_curve
  >>> from sklearn.svm import SVC

  >>> train_sizes, train_scores, valid_scores = learning_curve(
  ...     SVC(kernel='linear'), X, y, train_sizes=[50, 80, 110], cv=5)
  >>> train_sizes            # doctest: +NORMALIZE_WHITESPACE
  array([ 50, 80, 110])
  >>> train_scores           # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
  array([[0.98..., 0.98 , 0.98..., 0.98..., 0.98...],
         [0.98..., 1.   , 0.98..., 0.98..., 0.98...],
         [0.98..., 1.   , 0.98..., 0.98..., 0.99...]])
  >>> valid_scores           # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
  array([[1. ,  0.93...,  1. ,  1. ,  0.96...],
         [1. ,  0.96...,  1. ,  1. ,  0.96...],
         [1. ,  0.96...,  1. ,  1. ,  0.96...]])

