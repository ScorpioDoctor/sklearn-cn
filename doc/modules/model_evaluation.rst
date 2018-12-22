.. currentmodule:: sklearn

.. _model_evaluation:

========================================================
模型评估:对模型的预测进行量化考核
========================================================

有 3 种不同的 API 用于评估模型预测的质量:

* **Estimator score method**: Estimators（估计器）有一个 ``score`` 方法，为其解决的问题提供了默认的评估准则(evaluation criterion) 。 
  在本页面上没有相关讨论，但是在每个 estimator 的文档中会有相关的讨论。

* **Scoring parameter**: 使用了 :ref:`cross-validation <cross_validation>` (比如 :func:`model_selection.cross_val_score` 和 :class:`model_selection.GridSearchCV`)
  的模型评估工具依赖于一个内部评分策略。此参数的用法参考 :ref:`scoring_parameter` 。

* **Metric functions**: 模块 :mod:`metrics` 实现了一些函数用于以某种特殊目的评估模型预测误差。这些测度指标(metrics)的详细介绍在 :ref:`classification_metrics` ，
  :ref:`multilabel_ranking_metrics` , :ref:`regression_metrics` 以及 :ref:`clustering_metrics` 中。

最后, :ref:`dummy_estimators` 可以针对随机预测结果计算那些测度指标的一个基准值。

.. seealso::

   For "pairwise" metrics, between *samples* and not estimators or
   predictions, see the :ref:`metrics` section.

.. _scoring_parameter:

``scoring`` 参数: 定义模型评估准则
==========================================================

模型选择与评估使用的工具，例如 :class:`model_selection.GridSearchCV` 和 :func:`model_selection.cross_val_score`, 
接受一个 ``scoring`` 参数，该参数控制着估计器的评估过程中使用什么样的测度指标(metric)。

一般情况: 使用预定义的值
-------------------------------

对于最常见的用例, 可以使用 ``scoring`` 参数指定一个评分器对象(scorer object); 下表显示了所有可能的值。 
所有 scorer objects 遵循惯例:较高的返回值优于较低的返回值(higher return values are better than lower return values) 。
因此，度量模型和数据之间距离的测度指标(metrics), 如 :func:`metrics.mean_squared_error` 可作为 neg_mean_squared_error, 返回变负的指标值。
(译者注：也就是说 有些 测度指标比如均方误差本来是越小越好，但是为了遵循越大越好的惯例，我们要把这种原来越小越好的指标取个负号，这样就符合惯例啦 )

==============================    =============================================     ==================================
Scoring                           Function                                          Comment
==============================    =============================================     ==================================
**Classification**
'accuracy'                        :func:`metrics.accuracy_score`
'balanced_accuracy'               :func:`metrics.balanced_accuracy_score`           for binary targets
'average_precision'               :func:`metrics.average_precision_score`
'brier_score_loss'                :func:`metrics.brier_score_loss`
'f1'                              :func:`metrics.f1_score`                          for binary targets
'f1_micro'                        :func:`metrics.f1_score`                          micro-averaged
'f1_macro'                        :func:`metrics.f1_score`                          macro-averaged
'f1_weighted'                     :func:`metrics.f1_score`                          weighted average
'f1_samples'                      :func:`metrics.f1_score`                          by multilabel sample
'neg_log_loss'                    :func:`metrics.log_loss`                          requires ``predict_proba`` support
'precision' etc.                  :func:`metrics.precision_score`                   suffixes apply as with 'f1'
'recall' etc.                     :func:`metrics.recall_score`                      suffixes apply as with 'f1'
'roc_auc'                         :func:`metrics.roc_auc_score`

**Clustering**
'adjusted_mutual_info_score'      :func:`metrics.adjusted_mutual_info_score`
'adjusted_rand_score'             :func:`metrics.adjusted_rand_score`
'completeness_score'              :func:`metrics.completeness_score`
'fowlkes_mallows_score'           :func:`metrics.fowlkes_mallows_score`
'homogeneity_score'               :func:`metrics.homogeneity_score`
'mutual_info_score'               :func:`metrics.mutual_info_score`
'normalized_mutual_info_score'    :func:`metrics.normalized_mutual_info_score`
'v_measure_score'                 :func:`metrics.v_measure_score`

**Regression**
'explained_variance'              :func:`metrics.explained_variance_score`
'neg_mean_absolute_error'         :func:`metrics.mean_absolute_error`
'neg_mean_squared_error'          :func:`metrics.mean_squared_error`
'neg_mean_squared_log_error'      :func:`metrics.mean_squared_log_error`
'neg_median_absolute_error'       :func:`metrics.median_absolute_error`
'r2'                              :func:`metrics.r2_score`
==============================    =============================================     ==================================


用法案例:

    >>> from sklearn import svm, datasets
    >>> from sklearn.model_selection import cross_val_score
    >>> iris = datasets.load_iris()
    >>> X, y = iris.data, iris.target
    >>> clf = svm.SVC(gamma='scale', random_state=0)
    >>> cross_val_score(clf, X, y, scoring='recall_macro',
    ...                 cv=5)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    array([0.96..., 1.  ..., 0.96..., 0.96..., 1.        ])
    >>> model = svm.SVC()
    >>> cross_val_score(model, X, y, cv=5, scoring='wrong_choice')
    Traceback (most recent call last):
    ValueError: 'wrong_choice' is not a valid scoring value. Use sorted(sklearn.metrics.SCORERS.keys()) to get valid options.

.. note::

    通过 *ValueError* 异常列举出来的那些值对应于度量预测精度的函数，它们会在下面的小节中介绍。
    用于这些函数的评分器对象(scorer objects) 被存放在 ``sklearn.metrics.SCORERS`` 字典中。

.. currentmodule:: sklearn.metrics

.. _scoring:

利用指标函数 metric 自定义评分策略
-----------------------------------------------------

模块 :mod:`sklearn.metrics` 也暴露了一组简单的函数：当给定真值和预测值的时候用来度量一个预测错误。

- 以 ``_score`` 结尾的函数返回一个值进行最大化，值越高代表预测越好

- 以 ``_error`` 或 ``_loss`` 结尾的函数 返回一个值进行最小化，值越小代表预测越好。当我们使用函数 :func:`make_scorer` 
  把这种越小越好的metric转换成评分对象(scorer object)的时候,就需要设置参数 ``greater_is_better`` 为 False。 (这个参数默认是True,对这个参数下面还会解释)

可用于各种机器学习任务的 Metrics （指标）在下面详细介绍。

许多 metrics 没有被命名以使得它们被用作 ``scoring`` 值，有时是因为它们需要额外的参数，例如 :func:`fbeta_score` 。
在这种情况下，您需要生成一个适当的评分对象(scoring object)。最简单的办法就是利用函数 :func:`make_scorer` 生成一个用于评分的可调用对象
(callable object)。 函数 :func:`make_scorer` 将 metrics 转换为可用于模型评估的可调用对象。
(译者注：可调用对象即callable object是Python的一个知识点，如果你知道这个知识点那么这段话不难理解，如果不知道的话，请自行查一下就会明白啦！)

一个典型的用法是从库中封装一个已经存在的具有非默认值参数的 metric 函数，例如 :func:`fbeta_score` 函数的 ``beta`` 参数 ::

    >>> from sklearn.metrics import fbeta_score, make_scorer
    >>> ftwo_scorer = make_scorer(fbeta_score, beta=2)
    >>> from sklearn.model_selection import GridSearchCV
    >>> from sklearn.svm import LinearSVC
    >>> grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]},
    ...                     scoring=ftwo_scorer, cv=5)

第二个用法是使用 :func:`make_scorer` 从简单的 python 函数构建一个完全自定义的评分对象(scorer object) ，可以接受几个参数 :

* 你想使用的Python函数 (以 ``my_custom_loss_func`` 为例)

* 你的python函数返回值是 score (``greater_is_better=True``, the default) 还是 loss (``greater_is_better=False``)。  
  如果是 loss 的话, python函数的输出就会被 scorer object 取负号，以满足 交叉验证 中关于 评分准则越大越好 的约定惯例。

* 如果你要定义的是一个分类评分指标(classification metrics)，还要确认你的python函数需要连续的 decision certainties (``needs_threshold=True``)，
  默认值是 False。

* 任意的附加参数, 比如 :func:`f1_score` 函数中的 ``beta`` 或 ``labels`` .

下面是一个构建自定义评分器(custom scorers)的例子,并且使用了参数 ``greater_is_better`` ::

    >>> import numpy as np
    >>> def my_custom_loss_func(y_true, y_pred):
    ...     diff = np.abs(y_true - y_pred).max()
    ...     return np.log1p(diff)
    ...
    >>> # score will negate the return value of my_custom_loss_func,
    >>> # which will be np.log(2), 0.693, given the values for X
    >>> # and y defined below.
    >>> score = make_scorer(my_custom_loss_func, greater_is_better=False)
    >>> X = [[1], [1]]
    >>> y = [0, 1]
    >>> from sklearn.dummy import DummyClassifier
    >>> clf = DummyClassifier(strategy='most_frequent', random_state=0)
    >>> clf = clf.fit(X, y)
    >>> my_custom_loss_func(clf.predict(X), y) # doctest: +ELLIPSIS
    0.69...
    >>> score(clf, X, y) # doctest: +ELLIPSIS
    -0.69...


.. _diy_scoring:

实现你自己的 scoring object
------------------------------------
您可以通过从头开始构建自己的 scoring object，而不使用 :func:`make_scorer` 来生成更加灵活的模型评分对象(model scorers)。
如果一个python 可调用对象 被叫做 scorer ，那么它需要符合以下两个规则所指定的协议:

- 可以使用参数 ``(estimator, X, y)`` 来调用它，其中 ``estimator`` 是要被评估的模型，``X`` 是验证数据， ``y`` 是 真实目标变量 (在有监督情况下) 
  或 None (在无监督情况下)。

- 它返回一个浮点数，用于对 ``estimator`` 在 ``X`` 上的预测质量以 ``y`` 为真值参考进行量化。 再就是，按照惯例，越高的数字越好，
  所以如果你的 scorer 返回 loss ，那么这个值应该被取负号 。

.. _multimetric_scoring:

使用多指标评估
--------------------------------

Scikit-learn 还允许在 ``GridSearchCV``, ``RandomizedSearchCV`` 和 ``cross_validate`` 中进行多指标的评估(evaluation of multiple metrics)。

有两种方法可以为 ``scoring`` 参数指定 多个评分指标:

- 把多个metrics的名字以字符串列表的方式传给 ``scoring`` 参数 ::
      >>> scoring = ['accuracy', 'precision']

- 以字典的形式把评分器的名称映射到评分函数上，然后把这字典作为参数传给 ``scoring`` 参数 ::
      >>> from sklearn.metrics import accuracy_score
      >>> from sklearn.metrics import make_scorer
      >>> scoring = {'accuracy': make_scorer(accuracy_score),
      ...            'prec': 'precision'}

要注意的是 字典的值 既可以是 scorer functions 也可以是 sklearn预定义的metric的名字字符串。

目前，只有那些返回单个得分值的 scorer functions 可以被传到 字典中。 那些有多个返回值的 scorer functions 不被允许传入。
如果非要这么干的话，必须对其进行封装使其只有单个返回值 ::

    >>> from sklearn.model_selection import cross_validate
    >>> from sklearn.metrics import confusion_matrix
    >>> # A sample toy binary classification dataset
    >>> X, y = datasets.make_classification(n_classes=2, random_state=0)
    >>> svm = LinearSVC(random_state=0)
    >>> def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
    >>> def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
    >>> def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
    >>> def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]
    >>> scoring = {'tp': make_scorer(tp), 'tn': make_scorer(tn),
    ...            'fp': make_scorer(fp), 'fn': make_scorer(fn)}
    >>> cv_results = cross_validate(svm.fit(X, y), X, y,
    ...                             scoring=scoring, cv=5)
    >>> # Getting the test set true positive scores
    >>> print(cv_results['test_tp'])  # doctest: +NORMALIZE_WHITESPACE
    [10  9  8  7  8]
    >>> # Getting the test set false negative scores
    >>> print(cv_results['test_fn'])  # doctest: +NORMALIZE_WHITESPACE
    [0 1 2 3 2]

.. _classification_metrics:

分类问题的指标
=======================

.. currentmodule:: sklearn.metrics

:mod:`sklearn.metrics` 模块实现了几个 loss, score, 和 utility 函数来度量分类器性能。 
某些测度指标(metrics)可能需要 positive class，confidence values 或 binary decisions values 的概率估计。 
大多数的实现允许每个样本通过 ``sample_weight`` 参数为 整体得分(overall score) 提供 加权贡献(weighted contribution)。

这里面的一部分指标仅仅限于在二分类的情况下使用(binary classification case):

.. autosummary::
   :template: function.rst

   precision_recall_curve
   roc_curve
   balanced_accuracy_score


下面这些既能在二分类中用也能够用于多分类的情况(multiclass case):

.. autosummary::
   :template: function.rst

   cohen_kappa_score
   confusion_matrix
   hinge_loss
   matthews_corrcoef


下面的这些还可以用在多标签分类中(multilabel case):

.. autosummary::
   :template: function.rst

   accuracy_score
   classification_report
   f1_score
   fbeta_score
   hamming_loss
   jaccard_similarity_score
   log_loss
   precision_recall_fscore_support
   precision_score
   recall_score
   zero_one_loss

下面的这些指标可以用在两类多标签问题(不是multiclass而是binary classes喔):

.. autosummary::
   :template: function.rst

   average_precision_score
   roc_auc_score


在下面的小节中，我们会逐个讲解这些函数, 包括一些常用API的注解和metric的数学定义。

从二分类问题到多类或多标签问题
----------------------------------------

有些 metrics 基本上是为 binary classification tasks 定义的 (例如 :func:`f1_score`, :func:`roc_auc_score`) 。
在这些情况下，默认情况下仅评估 positive label （正标签），默认情况下我们假定 positive label （正类） 标记为 1 
(尽管可以通过 ``pos_label`` 参数进行配置)。

.. _average:

将 binary metric （二分指标）扩展为 multiclass （多类）或 multilabel （多标签）问题时，数据将被视为二分问题的集合，每个类都有一个binary metric。
然后可以使用多种策略在整个类中计算所有二分指标的平均值(average binary metric calculations across the set of classes)，
这些不同的计算平均值的策略在某些特定场景中可能会用到。 如果可用，您应该使用 ``average`` 参数来选择某个平均策略。

* ``"macro"``  简单地计算 binary metrics （二分指标）的平均值，赋予每个类别相同的权重。在不常见的类别重要的问题上，
  macro-averaging （宏观平均）可能是突出表现的一种手段。另一方面，所有类别同样重要的假设通常是不真实的，
  因此 macro-averaging （宏观平均）将过度强调不频繁类的典型的低性能。

* ``"weighted"`` 通过计算其在真实数据样本中的存在来对每个类的 score 进行加权的 binary metrics （二分指标）的平均值来计算类不平衡。

* ``"micro"`` 给每个 sample-class pair （样本类对）对 overall metric （总体指数）（sample-class 权重的结果除外） 等同的贡献。
  除了对每个类别的 metric 进行求和之外，这个总和构成每个类别度量的 dividends （除数）和 divisors （除数）计算一个整体商。 
  在 multilabel settings （多标签设置）中， Micro-averaging 可能是优先选择的，包括要忽略 majority class （多数类）的 multiclass classification （多类分类）。

* ``"samples"`` 仅适用于 multilabel problems （多标签问题）。它 does not calculate a per-class measure （不计算每个类别的 measure），而是计算 evaluation data 
  （评估数据）中的每个样本的 true and predicted classes （真实和预测类别）的 metric （指标），并返回 (sample_weight-weighted) 加权平均。

* 选择 ``average=None`` 将返回一个 array 与每个类的 score 。

虽然将 multiclass data 作为 array of class labels 提供给 metric ，就像 binary targets （二分类目标）一样，
multilabel data 被指定为 indicator matrix（标识矩阵），其中如果样本 ``i`` 具有标号 ``j`` ， ``[i, j]`` 具有值 1， 否则为值 0 。

.. _accuracy_score:

Accuracy score
--------------

函数 :func:`accuracy_score` 计算 `accuracy <https://en.wikipedia.org/wiki/Accuracy_and_precision>`_, 
也就是计算正确预测的比例(默认)或数量(normalize=False)。


在多标签分类中，该函数返回子集的准确率(subset accuracy)。对某个样本的预测标签的整个集合与该样本真正的标签集合严格匹配，
那么子集准确率就是1.0,反之 子集准确率为0.0。

如果 :math:`\hat{y}_i` 是 第 :math:`i` 个样本的预测值, 并且 :math:`y_i` 是对应的真实值, 则在 :math:`n_{\text{samples}}` 个样本上估计的
正确预测的比例(the fraction of correct predictions)定义如下：

.. math::

   \texttt{accuracy}(y, \hat{y}) = \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples}-1} 1(\hat{y}_i = y_i)

其中 :math:`1(x)` 是 `indicator function <https://en.wikipedia.org/wiki/Indicator_function>`_。

  >>> import numpy as np
  >>> from sklearn.metrics import accuracy_score
  >>> y_pred = [0, 2, 1, 3]
  >>> y_true = [0, 1, 2, 3]
  >>> accuracy_score(y_true, y_pred)
  0.5
  >>> accuracy_score(y_true, y_pred, normalize=False)
  2

在多标签的情形下，比如 每个样本需要预测两个标签(binary label indicators) ::

  >>> accuracy_score(np.array([[0, 1], [1, 1]]), np.ones((2, 2)))
  0.5

.. topic:: 案例:

  * See :ref:`sphx_glr_auto_examples_feature_selection_plot_permutation_test_for_classification.py`
    for an example of accuracy score usage using permutations of
    the dataset.

.. _balanced_accuracy_score:

Balanced accuracy score
-----------------------

此 :func:`balanced_accuracy_score` 函数计算 `balanced accuracy <https://en.wikipedia.org/wiki/Accuracy_and_precision>`_, 
它可以避免在不平衡数据集上作出夸大的性能估计。It is the macro-average of recall
scores per class or, equivalently, raw accuracy where each sample is weighted
according to the inverse prevalence of its true class.
因此，对均衡数据集， 该函数的得分与准确率得分是相等的。

在二分类情况下, balanced accuracy 等价于 `sensitivity <https://en.wikipedia.org/wiki/Sensitivity_and_specificity>`_
(真正率:true positive rate) 和 `specificity <https://en.wikipedia.org/wiki/Sensitivity_and_specificity>`_ (真负率:true negative
rate) 的算术平均值, 或者 the area under the ROC curve with binary predictions rather than
scores.

如果分类器在两个类上都表现的一样好，该函数就会退化为传统的准确率 (i.e., 正确预测数量除以总的预测数量).

作为对比, 如果传统的准确率(conventional accuracy)比较好，仅仅是因为分类器利用了一个不均衡测试集，此时 balanced_accuracy,将会近似地掉到
:math:`\frac{1}{\text{n\_classes}}`。

得分的范围是 0 到 1, 或者当使用 ``adjusted=True`` 时，得分被缩放到 从 :math:`\frac{1}{1 - \text{n\_classes}}` 到 1, 包括边界的, 随机条件下性能得分为0.

如果 :math:`y_i` 是第 :math:`i` 个样本的真值，并且 :math:`w_i` 是对应的样本权重，然后我们调整样本权重到 :

.. math::

   \hat{w}_i = \frac{w_i}{\sum_j{1(y_j = y_i) w_j}}

其中 :math:`1(x)` 是 `indicator function <https://en.wikipedia.org/wiki/Indicator_function>`_。
给定样本 :math:`i` 的预测值 :math:`\hat{y}_i` , balanced accuracy 如下定义：

.. math::

   \texttt{balanced-accuracy}(y, \hat{y}, w) = \frac{1}{\sum{\hat{w}_i}} \sum_i 1(\hat{y}_i = y_i) \hat{w}_i

With ``adjusted=True``, balanced accuracy reports the relative increase from
:math:`\texttt{balanced-accuracy}(y, \mathbf{0}, w) =
\frac{1}{\text{n\_classes}}`.  In the binary case, this is also known as
`*Youden's J statistic* <https://en.wikipedia.org/wiki/Youden%27s_J_statistic>`_,
or *informedness*.

.. note::

    The multiclass definition here seems the most reasonable extension of the
    metric used in binary classification, though there is no certain consensus
    in the literature:

    * Our definition: [Mosley2013]_, [Kelleher2015]_ and [Guyon2015]_, where
      [Guyon2015]_ adopt the adjusted version to ensure that random predictions
      have a score of :math:`0` and perfect predictions have a score of :math:`1`..
    * Class balanced accuracy as described in [Mosley2013]_: the minimum between the precision
      and the recall for each class is computed. Those values are then averaged over the total
      number of classes to get the balanced accuracy.
    * Balanced Accuracy as described in [Urbanowicz2015]_: the average of sensitivity and specificity
      is computed for each class and then averaged over total number of classes.

.. topic:: 参考文献:

  .. [Guyon2015] I. Guyon, K. Bennett, G. Cawley, H.J. Escalante, S. Escalera, T.K. Ho, N. Macià,
     B. Ray, M. Saeed, A.R. Statnikov, E. Viegas, `Design of the 2015 ChaLearn AutoML Challenge
     <https://ieeexplore.ieee.org/document/7280767>`_,
     IJCNN 2015.
  .. [Mosley2013] L. Mosley, `A balanced approach to the multi-class imbalance problem
     <https://lib.dr.iastate.edu/etd/13537/>`_,
     IJCV 2010.
  .. [Kelleher2015] John. D. Kelleher, Brian Mac Namee, Aoife D'Arcy, `Fundamentals of
     Machine Learning for Predictive Data Analytics: Algorithms, Worked Examples,
     and Case Studies <https://mitpress.mit.edu/books/fundamentals-machine-learning-predictive-data-analytics>`_,
     2015.
  .. [Urbanowicz2015] Urbanowicz R.J.,  Moore, J.H. `ExSTraCS 2.0: description and evaluation of a scalable learning
     classifier system <https://doi.org/10.1007/s12065-015-0128-8>`_, Evol. Intel. (2015) 8: 89.

.. _cohen_kappa:

Cohen's kappa
-------------

函数 :func:`cohen_kappa_score` 计算 `Cohen's kappa <https://en.wikipedia.org/wiki/Cohen%27s_kappa>`_ 统计.
这个度量指标旨在比较由不同的人类标注者给出的标签，而不是去比较分类器预测和真值(ground truth)。 

The kappa score (see docstring) 是一个介于 -1 到 1 之间的数字。得分超过0.8通常被认为是 good agreement; 
得分为0或者小于0意味着 no agreement。

Kappa scores 既可以用于 二分类也可用于多分类，但是 不能用于 多标签问题(except by manually computing a per-label score)。
and not for more than two annotators.

  >>> from sklearn.metrics import cohen_kappa_score
  >>> y_true = [2, 0, 2, 2, 0, 1]
  >>> y_pred = [0, 0, 2, 2, 0, 2]
  >>> cohen_kappa_score(y_true, y_pred)
  0.4285714285714286

.. _confusion_matrix:

Confusion matrix
----------------

函数 :func:`confusion_matrix` 通过计算 混淆矩阵( `confusion matrix <https://en.wikipedia.org/wiki/Confusion_matrix>`_) 
来评估分类准确率。confusion matrix 的每一行对应于真的类。(但是 维基百科和其他引用文献可能会使用不同的axes)。


按照定义, 在 confusion matrix 中，入口 :math:`i, j` 中存储着实际上应该在group :math:`i` 中的观测,
但是却被预测到了group :math:`j` 里面去的这些观测的数量。 这里有一个例子 ::

  >>> from sklearn.metrics import confusion_matrix
  >>> y_true = [2, 0, 2, 2, 0, 1]
  >>> y_pred = [0, 0, 2, 2, 0, 2]
  >>> confusion_matrix(y_true, y_pred)
  array([[2, 0, 0],
         [0, 0, 1],
         [1, 0, 2]])

这里有一个混淆矩阵的可视化表示。 (请看来自于这个例子的图片 :ref:`sphx_glr_auto_examples_model_selection_plot_confusion_matrix.py` ):

.. image:: ../auto_examples/model_selection/images/sphx_glr_plot_confusion_matrix_001.png
   :target: ../auto_examples/model_selection/plot_confusion_matrix.html
   :scale: 75
   :align: center

对于二分类问题, 我们可以得到 真负(true negatives), 假正(false positives), 假负(false negatives) 和 真正(true positives) 的数量 ::

  >>> y_true = [0, 0, 0, 1, 1, 1, 1, 1]
  >>> y_pred = [0, 1, 0, 1, 0, 1, 0, 1]
  >>> tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
  >>> tn, fp, fn, tp
  (2, 1, 2, 3)

.. topic:: 案例:

  * See :ref:`sphx_glr_auto_examples_model_selection_plot_confusion_matrix.py`
    for an example of using a confusion matrix to evaluate classifier output
    quality.

  * See :ref:`sphx_glr_auto_examples_classification_plot_digits_classification.py`
    for an example of using a confusion matrix to classify
    hand-written digits.

  * See :ref:`sphx_glr_auto_examples_text_plot_document_classification_20newsgroups.py`
    for an example of using a confusion matrix to classify text
    documents.

.. _classification_report:

Classification report
----------------------

函数 :func:`classification_report` 会构造一个文本报告展示主要的分类指标。
下面有一个小例子，里面有自定义的 ``target_names`` 和 inferred labels ::

   >>> from sklearn.metrics import classification_report
   >>> y_true = [0, 1, 2, 2, 0]
   >>> y_pred = [0, 0, 2, 1, 0]
   >>> target_names = ['class 0', 'class 1', 'class 2']
   >>> print(classification_report(y_true, y_pred, target_names=target_names))
                 precision    recall  f1-score   support
   <BLANKLINE>
        class 0       0.67      1.00      0.80         2
        class 1       0.00      0.00      0.00         1
        class 2       1.00      0.50      0.67         2
   <BLANKLINE>
      micro avg       0.60      0.60      0.60         5
      macro avg       0.56      0.50      0.49         5
   weighted avg       0.67      0.60      0.59         5
   <BLANKLINE>

.. topic:: 案例:

  * See :ref:`sphx_glr_auto_examples_classification_plot_digits_classification.py`
    for an example of classification report usage for
    hand-written digits.

  * See :ref:`sphx_glr_auto_examples_text_plot_document_classification_20newsgroups.py`
    for an example of classification report usage for text
    documents.

  * See :ref:`sphx_glr_auto_examples_model_selection_plot_grid_search_digits.py`
    for an example of classification report usage for
    grid search with nested cross-validation.

.. _hamming_loss:

Hamming loss
-------------

:func:`hamming_loss` 计算两个样本集合之间的平均 Hamming loss 或 `Hamming distance <https://en.wikipedia.org/wiki/Hamming_distance>`_ 。

如果 :math:`\hat{y}_j` 是给定样本的第 :math:`j` 个标签的预测值，:math:`y_j` 是对应的真值，:math:`n_\text{labels}` 是类(或 标签)的数量，
则两个样本之间的 Hamming loss :math:`L_{Hamming}` 定义如下：

.. math::

   L_{Hamming}(y, \hat{y}) = \frac{1}{n_\text{labels}} \sum_{j=0}^{n_\text{labels} - 1} 1(\hat{y}_j \not= y_j)

其中 :math:`1(x)` 是 `indicator function <https://en.wikipedia.org/wiki/Indicator_function>`_. ::

  >>> from sklearn.metrics import hamming_loss
  >>> y_pred = [1, 2, 3, 4]
  >>> y_true = [2, 2, 3, 4]
  >>> hamming_loss(y_true, y_pred)
  0.25

在多标签情况下，假如每个样本有两个标签(binary label indicators) ::

  >>> hamming_loss(np.array([[0, 1], [1, 1]]), np.zeros((2, 2)))
  0.75

.. note::

    在多类分类任务中, Hamming loss 对应 ``y_true`` 和 ``y_pred`` 之间的 Hamming distance，这与 :ref:`zero_one_loss` 函数是相似的。
    然而，尽管 zero-one loss 惩罚的是不与真值集合严格匹配的预测集合，但是 Hamming loss 惩罚的是独立的标签(individual labels)。
    因此，the Hamming loss, 以 zero-one loss 为上界, 其取值区间在 [0, 1]; 预测真实标签的一个合适的子集或超集将会给出一个范围在(0,1)之间的Hamming loss。

.. _jaccard_similarity_score:

Jaccard similarity coefficient score
-------------------------------------

函数 :func:`jaccard_similarity_score` 计算两个标签集合之间的 `Jaccard similarity coefficients <https://en.wikipedia.org/wiki/Jaccard_index>`_ 
的average(default)或sum, 也被称之为 Jaccard index.

给定 :math:`i`-th samples, 以及关于样本的 真正的标签集合 :math:`y_i` 和 预测出的标签集合 :math:`\hat{y}_i`, 
Jaccard similarity coefficient 是如下定义的：

.. math::

    J(y_i, \hat{y}_i) = \frac{|y_i \cap \hat{y}_i|}{|y_i \cup \hat{y}_i|}.

在两类分类和多类分类中, Jaccard similarity coefficient score 等价于 分类准确率。

::

  >>> import numpy as np
  >>> from sklearn.metrics import jaccard_similarity_score
  >>> y_pred = [0, 2, 1, 3]
  >>> y_true = [0, 1, 2, 3]
  >>> jaccard_similarity_score(y_true, y_pred)
  0.5
  >>> jaccard_similarity_score(y_true, y_pred, normalize=False)
  2

在具有二元标签指示符(binary label indicators)的多标签情况下: ::

  >>> jaccard_similarity_score(np.array([[0, 1], [1, 1]]), np.ones((2, 2)))
  0.75

.. _precision_recall_f_measure_metrics:

Precision, recall and F-measures
---------------------------------

直观地讲, 精度(`precision <https://en.wikipedia.org/wiki/Precision_and_recall#Precision>`_) 指的是分类器不会把负样本标记为正样本的能力；
召回率(或叫 查全率 `recall <https://en.wikipedia.org/wiki/Precision_and_recall#Recall>`_) 指的是分类器找到数据集中所有的正样本的能力。

F-度量(`F-measure <https://en.wikipedia.org/wiki/F1_score>`_ (包括 :math:`F_\beta` 和 :math:`F_1` 度量) ) 可被解释为精度(precision)和
查全率(recall)的加权调和均值(weighted harmonic mean)。 
一个 :math:`F_\beta` measure 在取值为1的时候达到它的最好值，而取值为0的时候达到最差得分。当 :math:`\beta = 1` 时,  :math:`F_\beta` 和 :math:`F_1`
是等价的，而且这时候 recall 和 precision 在 :math:`F_1` 的计算中是同等重要的。

函数 :func:`precision_recall_curve` 通过不断改变决策阈值 (decision threshold) 从真实标签和分类器给出的一个得分中计算一条 precision-recall 曲线。

函数 :func:`average_precision_score` 从预测得分中计算平均精度
( `average precision <https://en.wikipedia.org/w/index.php?title=Information_retrieval&oldid=793358396#Average_precision>`_
(AP))。 它的取值在0到1之间，越高越好。平均精度(AP)是如下定义的：

.. math::
    \text{AP} = \sum_n (R_n - R_{n-1}) P_n

其中 :math:`P_n` 和 :math:`R_n` 是第n个阈值处的precision 和 recall。对于随机预测，AP 是正样本的比例。

参考文献 [Manning2008]_ 和 [Everingham2010]_ 提出了AP的两种可替代变体对precision-recall曲线进行内插。
当前，函数 :func:`average_precision_score` 还没有实现任何具备内插的变体版本。
参考文献 [Davis2006]_ 和 [Flach2015]_ 描述了为什么precision-recall曲线上的点的线性内插提供了一个过于乐观(overly-optimistic)的分类器性能度量。
在函数 :func:`auc` 中使用梯形规则(trapezoidal rule)计算曲线下面积的时候，这个线性内插(linear interpolation)会被使用。 

下面这些函数允许你分析  precision, recall 和 F-measures score:

.. autosummary::
   :template: function.rst

   average_precision_score
   f1_score
   fbeta_score
   precision_recall_curve
   precision_recall_fscore_support
   precision_score
   recall_score

注意 函数 :func:`precision_recall_curve` 只能在二分类的情形下使用。函数 :func:`average_precision_score` 只能工作在 binary classification 和
multilabel indicator 情形下。


.. topic:: 案例:

  * See :ref:`sphx_glr_auto_examples_text_plot_document_classification_20newsgroups.py`
    for an example of :func:`f1_score` usage to classify  text
    documents.

  * See :ref:`sphx_glr_auto_examples_model_selection_plot_grid_search_digits.py`
    for an example of :func:`precision_score` and :func:`recall_score` usage
    to estimate parameters using grid search with nested cross-validation.

  * See :ref:`sphx_glr_auto_examples_model_selection_plot_precision_recall.py`
    for an example of :func:`precision_recall_curve` usage to evaluate
    classifier output quality.


.. topic:: 参考文献:

  .. [Manning2008] C.D. Manning, P. Raghavan, H. Schütze, `Introduction to Information Retrieval
     <http://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html>`_,
     2008.
  .. [Everingham2010] M. Everingham, L. Van Gool, C.K.I. Williams, J. Winn, A. Zisserman,
     `The Pascal Visual Object Classes (VOC) Challenge
     <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.157.5766&rep=rep1&type=pdf>`_,
     IJCV 2010.
  .. [Davis2006] J. Davis, M. Goadrich, `The Relationship Between Precision-Recall and ROC Curves
     <http://www.machinelearning.org/proceedings/icml2006/030_The_Relationship_Bet.pdf>`_,
     ICML 2006.
  .. [Flach2015] P.A. Flach, M. Kull, `Precision-Recall-Gain Curves: PR Analysis Done Right
     <http://papers.nips.cc/paper/5867-precision-recall-gain-curves-pr-analysis-done-right.pdf>`_,
     NIPS 2015.


Binary classification
^^^^^^^^^^^^^^^^^^^^^

在一个二分类任务中，词语 ''positive'' 和 ''negative'' 指的是分类器的预测，词语 ''true'' 和 ''false'' 指的是预测是否和来自外部判断
(external judgment，sometimes known as the ''observation'')相互对应。有了上述词汇的定义，我们就可以给出下面这个表啦：

+-------------------+------------------------------------------------+
|                   |    Actual class (observation)                  |
+-------------------+---------------------+--------------------------+
|   Predicted class | tp (true positive)  | fp (false positive)      |
|   (expectation)   | Correct result      | Unexpected result        |
|                   +---------------------+--------------------------+
|                   | fn (false negative) | tn (true negative)       |
|                   | Missing result      | Correct absence of result|
+-------------------+---------------------+--------------------------+

以此为上下文, 我们可以定义 precision, recall 和 F-measure 如下所示:

.. math::

   \text{precision} = \frac{tp}{tp + fp},

.. math::

   \text{recall} = \frac{tp}{tp + fn},

.. math::

   F_\beta = (1 + \beta^2) \frac{\text{precision} \times \text{recall}}{\beta^2 \text{precision} + \text{recall}}.

下面是一些二分类的例子 ::

  >>> from sklearn import metrics
  >>> y_pred = [0, 1, 0, 0]
  >>> y_true = [0, 1, 0, 1]
  >>> metrics.precision_score(y_true, y_pred)
  1.0
  >>> metrics.recall_score(y_true, y_pred)
  0.5
  >>> metrics.f1_score(y_true, y_pred)  # doctest: +ELLIPSIS
  0.66...
  >>> metrics.fbeta_score(y_true, y_pred, beta=0.5)  # doctest: +ELLIPSIS
  0.83...
  >>> metrics.fbeta_score(y_true, y_pred, beta=1)  # doctest: +ELLIPSIS
  0.66...
  >>> metrics.fbeta_score(y_true, y_pred, beta=2) # doctest: +ELLIPSIS
  0.55...
  >>> metrics.precision_recall_fscore_support(y_true, y_pred, beta=0.5)  # doctest: +ELLIPSIS
  (array([0.66..., 1.        ]), array([1. , 0.5]), array([0.71..., 0.83...]), array([2, 2]))


  >>> import numpy as np
  >>> from sklearn.metrics import precision_recall_curve
  >>> from sklearn.metrics import average_precision_score
  >>> y_true = np.array([0, 0, 1, 1])
  >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
  >>> precision, recall, threshold = precision_recall_curve(y_true, y_scores)
  >>> precision  # doctest: +ELLIPSIS
  array([0.66..., 0.5       , 1.        , 1.        ])
  >>> recall
  array([1. , 0.5, 0.5, 0. ])
  >>> threshold
  array([0.35, 0.4 , 0.8 ])
  >>> average_precision_score(y_true, y_scores)  # doctest: +ELLIPSIS
  0.83...



多类分类和多标签分类
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
在多类和多标签分类任务中，precision, recall, 和 F-measures 的概念可以独立的应用到每一个标签上。
有很多方法可以把所有标签上的结果组合起来，这可以通过设置参数 ``average`` 为 
:func:`average_precision_score` (multilabel only), :func:`f1_score`,
:func:`fbeta_score`, :func:`precision_recall_fscore_support`,
:func:`precision_score` 和 :func:`recall_score` 这些函数来实现，就像在 :ref:`above <average>` 中描述的那样。
请注意 如果所有标签都包括了，在多类分类的设置下取 "micro" 平均策略 将会使产生的 precision, recall 和 :math:`F` 都跟准确率(accuracy)一样。
还要注意的是 "weighted" averaging 策略会产生一个取值范围不在precision 和 recall 之间的 F-score。

要使这更显式更明确，请考虑以下表示法:

* :math:`y` the set of *predicted* :math:`(sample, label)` pairs
* :math:`\hat{y}` the set of *true* :math:`(sample, label)` pairs
* :math:`L` the set of labels
* :math:`S` the set of samples
* :math:`y_s` the subset of :math:`y` with sample :math:`s`,
  i.e. :math:`y_s := \left\{(s', l) \in y | s' = s\right\}`
* :math:`y_l` the subset of :math:`y` with label :math:`l`
* similarly, :math:`\hat{y}_s` and :math:`\hat{y}_l` are subsets of
  :math:`\hat{y}`
* :math:`P(A, B) := \frac{\left| A \cap B \right|}{\left|A\right|}`
* :math:`R(A, B) := \frac{\left| A \cap B \right|}{\left|B\right|}`
  (Conventions vary on handling :math:`B = \emptyset`; this implementation uses
  :math:`R(A, B):=0`, and similar for :math:`P`.)
* :math:`F_\beta(A, B) := \left(1 + \beta^2\right) \frac{P(A, B) \times R(A, B)}{\beta^2 P(A, B) + R(A, B)}`

然后这些指标就可以定义如下：

+---------------+------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------+
|``average``    | Precision                                                                                                        | Recall                                                                                                           | F\_beta                                                                                                              |
+===============+==================================================================================================================+==================================================================================================================+======================================================================================================================+
|``"micro"``    | :math:`P(y, \hat{y})`                                                                                            | :math:`R(y, \hat{y})`                                                                                            | :math:`F_\beta(y, \hat{y})`                                                                                          |
+---------------+------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------+
|``"samples"``  | :math:`\frac{1}{\left|S\right|} \sum_{s \in S} P(y_s, \hat{y}_s)`                                                | :math:`\frac{1}{\left|S\right|} \sum_{s \in S} R(y_s, \hat{y}_s)`                                                | :math:`\frac{1}{\left|S\right|} \sum_{s \in S} F_\beta(y_s, \hat{y}_s)`                                              |
+---------------+------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------+
|``"macro"``    | :math:`\frac{1}{\left|L\right|} \sum_{l \in L} P(y_l, \hat{y}_l)`                                                | :math:`\frac{1}{\left|L\right|} \sum_{l \in L} R(y_l, \hat{y}_l)`                                                | :math:`\frac{1}{\left|L\right|} \sum_{l \in L} F_\beta(y_l, \hat{y}_l)`                                              |
+---------------+------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------+
|``"weighted"`` | :math:`\frac{1}{\sum_{l \in L} \left|\hat{y}_l\right|} \sum_{l \in L} \left|\hat{y}_l\right| P(y_l, \hat{y}_l)`  | :math:`\frac{1}{\sum_{l \in L} \left|\hat{y}_l\right|} \sum_{l \in L} \left|\hat{y}_l\right| R(y_l, \hat{y}_l)`  | :math:`\frac{1}{\sum_{l \in L} \left|\hat{y}_l\right|} \sum_{l \in L} \left|\hat{y}_l\right| F_\beta(y_l, \hat{y}_l)`|
+---------------+------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------+
|``None``       | :math:`\langle P(y_l, \hat{y}_l) | l \in L \rangle`                                                              | :math:`\langle R(y_l, \hat{y}_l) | l \in L \rangle`                                                              | :math:`\langle F_\beta(y_l, \hat{y}_l) | l \in L \rangle`                                                            |
+---------------+------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------+

  >>> from sklearn import metrics
  >>> y_true = [0, 1, 2, 0, 1, 2]
  >>> y_pred = [0, 2, 1, 0, 0, 1]
  >>> metrics.precision_score(y_true, y_pred, average='macro')  # doctest: +ELLIPSIS
  0.22...
  >>> metrics.recall_score(y_true, y_pred, average='micro')
  ... # doctest: +ELLIPSIS
  0.33...
  >>> metrics.f1_score(y_true, y_pred, average='weighted')  # doctest: +ELLIPSIS
  0.26...
  >>> metrics.fbeta_score(y_true, y_pred, average='macro', beta=0.5)  # doctest: +ELLIPSIS
  0.23...
  >>> metrics.precision_recall_fscore_support(y_true, y_pred, beta=0.5, average=None)
  ... # doctest: +ELLIPSIS
  (array([0.66..., 0.        , 0.        ]), array([1., 0., 0.]), array([0.71..., 0.        , 0.        ]), array([2, 2, 2]...))

对于带有一个 "negative class" 的多分类任务, 不包括某些标签是可能的:

  >>> metrics.recall_score(y_true, y_pred, labels=[1, 2], average='micro')
  ... # excluding 0, no labels were correctly recalled
  0.0

类似的, labels not present in the data sample may be accounted for in macro-averaging.

  >>> metrics.precision_score(y_true, y_pred, labels=[0, 1, 2, 3], average='macro')
  ... # doctest: +ELLIPSIS
  0.166...

.. _hinge_loss:

Hinge loss
----------

函数 :func:`hinge_loss` 使用 `hinge loss <https://en.wikipedia.org/wiki/Hinge_loss>`_ 计算模型和数据之间的平均距离。
折叶损失(hinge loss)是一种 单边测度指标(one-sided metric),它仅仅考虑预测误差。
(Hinge loss 被用在最大间隔分类器(maximal margin classifiers)如SVMs中)。

如果类标签被编码为 +1 和 -1， :math:`y`: 是真值，并且 :math:`w` 是用 ``decision_function`` 
预测得到的作为输出的决策，那么 hinge loss 定义如下:

.. math::

  L_\text{Hinge}(y, w) = \max\left\{1 - wy, 0\right\} = \left|1 - wy\right|_+

如果有两个以上的标签, :func:`hinge_loss` 会使用一个多类变种(multiclass variant) 根据 Crammer & Singer 的
论文所描述的：`Here <http://jmlr.csail.mit.edu/papers/volume2/crammer01a/crammer01a.pdf>`_ 。

如果 :math:`y_w` 是对真实标签预测出的决策，并且 :math:`y_t` 是所有其他标签的预测中的最大值，
其中 预测出的决策是决策函数(decision function)的输出，那么多个类的hinge loss定义如下:

.. math::

  L_\text{Hinge}(y_w, y_t) = \max\left\{1 + y_t - y_w, 0\right\}

下面展示了一个例子说明如何使用 :func:`hinge_loss` 函数 将svm分类器用在二分类问题中 ::

  >>> from sklearn import svm
  >>> from sklearn.metrics import hinge_loss
  >>> X = [[0], [1]]
  >>> y = [-1, 1]
  >>> est = svm.LinearSVC(random_state=0)
  >>> est.fit(X, y)
  LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
       intercept_scaling=1, loss='squared_hinge', max_iter=1000,
       multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
       verbose=0)
  >>> pred_decision = est.decision_function([[-2], [3], [0.5]])
  >>> pred_decision  # doctest: +ELLIPSIS
  array([-2.18...,  2.36...,  0.09...])
  >>> hinge_loss([-1, 1, 1], pred_decision)  # doctest: +ELLIPSIS
  0.3...

下面展示了一个例子说明如何使用 :func:`hinge_loss` 函数 将svm分类器用在多类分类问题中 ::

  >>> X = np.array([[0], [1], [2], [3]])
  >>> Y = np.array([0, 1, 2, 3])
  >>> labels = np.array([0, 1, 2, 3])
  >>> est = svm.LinearSVC()
  >>> est.fit(X, Y)
  LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
       intercept_scaling=1, loss='squared_hinge', max_iter=1000,
       multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
       verbose=0)
  >>> pred_decision = est.decision_function([[-1], [2], [3]])
  >>> y_true = [0, 2, 3]
  >>> hinge_loss(y_true, pred_decision, labels)  #doctest: +ELLIPSIS
  0.56...

.. _log_loss:

Log loss
--------

对数损失(Log loss)，又被称为logistic回归损失(logistic regression loss) 或者 交叉熵损失(cross-entropy loss), 
定义在概率估计(probability estimates)上。 它通常用于(多项式)logistic回归 (multinomial logistic regression)
和 神经网络(neural networks) 以及 期望最大化(expectation-maximization)的一些变体中，
并且可用于评估分类器的概率输出(probability outputs: ``predict_proba``)而不是其离散预测值(discrete predictions)。

对于二分类问题，并伴有 真实类标签 :math:`y \in \{0,1\}` 以及 一个概率估计 :math:`p = \operatorname{Pr}(y = 1)`, 
则 每个样本的对数损失是给定真实标签时分类器的负对数似然函数(negative log-likelihood):

.. math::

    L_{\log}(y, p) = -\log \operatorname{Pr}(y|p) = -(y \log (p) + (1 - y) \log (1 - p))

上面的公式可以按下述方法扩展到多类别分类的情形。
首先把一个样本集合的真实类标签编码成 1-of-K 二进制指示矩阵(1-of-K binary indicator matrix): :math:`Y`,
也就是说，如果样本 :math:`i` 有标签 :math:`k`，而 标签 :math:`k` 又是取自于 :math:`K` 个类标签的集合中，那么就让 :math:`y_{i,k} = 1`。
再令 :math:`P` 是概率的估计值的矩阵，并有 :math:`p_{i,k} = \operatorname{Pr}(t_{i,k} = 1)`。
那么 整个样本集上的对数损失就定义如下：

.. math::

    L_{\log}(Y, P) = -\log \operatorname{Pr}(Y|P) = - \frac{1}{N} \sum_{i=0}^{N-1} \sum_{k=0}^{K-1} y_{i,k} \log p_{i,k}

为了让你看清楚上面的公式是如何对二类对数损失(binary log loss)进行推广的，请注意 在二分类情况下，
:math:`p_{i,0} = 1 - p_{i,1}` 和 :math:`y_{i,0} = 1 - y_{i,1}`,
因此在 :math:`y_{i,k} \in \{0,1\}` 上扩展内部核 就可以得到(binary log loss)。

函数 :func:`log_loss` 在真实标签和概率矩阵的列表给定后计算对数损失, as returned by an estimator's ``predict_proba``
method.

    >>> from sklearn.metrics import log_loss
    >>> y_true = [0, 0, 1, 1]
    >>> y_pred = [[.9, .1], [.8, .2], [.3, .7], [.01, .99]]
    >>> log_loss(y_true, y_pred)    # doctest: +ELLIPSIS
    0.1738...

在 ``y_pred`` 中的第一个预测 ``[.9, .1]`` 意味着 第一个样本的标签是0的概率达到了90%。对数损失是非负的。

.. _matthews_corrcoef:

Matthews相关系数
---------------------------------

函数 :func:`matthews_corrcoef` 计算用于binary classes的 `Matthew's correlation coefficient (MCC) <https://en.wikipedia.org/wiki/Matthews_correlation_coefficient>`_
引用维基百科对 Matthews相关系数 的解释：


    "Matthews相关系数(The Matthews correlation coefficient)在机器学习中用作二分类的质量的度量。
    它以正负阴阳(true and false positives and negatives)为考虑， 并且被广泛认为是一个均衡的度量，
    即使是在各个类的样本集大小非常不一样大的时候也可以使用。MCC本质上是一个取值范围在-1到+1的相关系数(correlation coefficient),
    "0" 代表了 平均随机预测(average random prediction)，"-1" 代表了 反转预测(inverse prediction)。
    The statistic is also known as the phi coefficient.



在二分类情况下, :math:`tp`, :math:`tn`, :math:`fp` 和 :math:`fn` 分别指的是 the number of true positives, true negatives, false
positives and false negatives, 那么 MCC 就定义为：

.. math::

  MCC = \frac{tp \times tn - fp \times fn}{\sqrt{(tp + fp)(tp + fn)(tn + fp)(tn + fn)}}.

在多类分类任务中, 给定 :math:`K` 个类的 :func:`confusion_matrix` :math:`C` 以后，Matthews 相关系数可以如此定义 
`defined <http://rk.kvl.dk/introduction/index.html>`_ 。  为了简化它的定义，我们用以下这些中间变量：

* :math:`t_k=\sum_{i}^{K} C_{ik}` the number of times class :math:`k` truly occurred,
* :math:`p_k=\sum_{i}^{K} C_{ki}` the number of times class :math:`k` was predicted,
* :math:`c=\sum_{k}^{K} C_{kk}` the total number of samples correctly predicted,
* :math:`s=\sum_{i}^{K} \sum_{j}^{K} C_{ij}` the total number of samples.

然后，multiclass MCC 的定义如下所示：

.. math::
    MCC = \frac{
        c \times s - \sum_{k}^{K} p_k \times t_k
    }{\sqrt{
        (s^2 - \sum_{k}^{K} p_k^2) \times
        (s^2 - \sum_{k}^{K} t_k^2)
    }}

当有两种以上类标签的时候，MCC的值将不再在-1到+1之间。它的最小值将是一个介于-1到0之间的数，具体数值取决于真实类标签的数量和分布。
它的最大值总是+1。

下面是使用 :func:`matthews_corrcoef` 函数的一个简单小例子::

    >>> from sklearn.metrics import matthews_corrcoef
    >>> y_true = [+1, +1, +1, -1]
    >>> y_pred = [+1, -1, +1, +1]
    >>> matthews_corrcoef(y_true, y_pred)  # doctest: +ELLIPSIS
    -0.33...

.. _roc_metrics:

Receiver operating characteristic (ROC)
---------------------------------------

函数 :func:`roc_curve` 计算 接收机操作特性曲线( `receiver operating characteristic curve, or ROC curve <https://en.wikipedia.org/wiki/Receiver_operating_characteristic>`_)。

引用自维基百科 :

  "一个 接收机操作特性 (ROC), 或简单点叫做 ROC 曲线, 是一幅图，这幅图展示了一个二分类器系统在它的判别阈值(discrimination threshold)
  不断变化的时候的性能。这条曲线上坐标点的纵轴取值是真正的正样本的比例(真阳率: TPR= true positive rate, i.e. the fraction of true positives out of the positives);
  这条曲线上坐标点的横轴取值是负样本中假的正样本的比例(假阳率or虚警率 FPR = false positive rate，i.e.  the fraction of false positives out of the negatives)。 
  当不断改变二分类器的阈值的时候，上述TPR和FPR就会跟着发生变化。这样每一个阈值都会对应一对坐标点(TPR,FPR)，只要不断改变阈值就会产生一条曲线。
  TPR 也被称之为 灵敏度(sensitivity), 而 FPR 是 one minus the specificity or true negative rate."
  (译者注：在二分类问题中，我们有时候会把其中一类特别关注，比如疾病检查的时候各种身体指标有阴性和阳性之分，阳性代表不正常的类是需要被特别关注的；
  再比如在雷达目标检测中，对真正的目标的检出是非常重要的，雷达系统灵敏度越高就代表能够捕捉的真实目标就越多，
  但是灵敏度太高会导致雷达系统把非真实目标看作是真实目标从而报虚警。
  但是报虚警总比漏检真实目标带来的危害小，因为在雷达武器系统中漏检真实目标是致命的错误。这两个例子中的二分类问题都有一个重点关注的类：positive类。
  所以ROC曲线反应的指标也是以positive类为核心的：真阳率(TPR) vs 假阳率(FPR))

This function requires the true binary
value and the target scores, which can either be probability estimates of the
positive class, confidence values, or binary decisions.
下面是函数 :func:`roc_curve` 的一个小例子 ::

    >>> import numpy as np
    >>> from sklearn.metrics import roc_curve
    >>> y = np.array([1, 1, 2, 2])
    >>> scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> fpr, tpr, thresholds = roc_curve(y, scores, pos_label=2)
    >>> fpr
    array([0. , 0. , 0.5, 0.5, 1. ])
    >>> tpr
    array([0. , 0.5, 0.5, 1. , 1. ])
    >>> thresholds
    array([1.8 , 0.8 , 0.4 , 0.35, 0.1 ])

下图展示了这样一个ROC曲线的例子:

.. image:: ../auto_examples/model_selection/images/sphx_glr_plot_roc_001.png
   :target: ../auto_examples/model_selection/plot_roc.html
   :scale: 75
   :align: center

函数 :func:`roc_auc_score` 计算 ROC曲线下的面积, 也被记为 AUC 或 AUROC.  通过计算曲线下的面积，ROC曲线信息被总结到一个数字中。
更多详细信息请参考  `Wikipedia article on AUC
<https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve>`_.

  >>> import numpy as np
  >>> from sklearn.metrics import roc_auc_score
  >>> y_true = np.array([0, 0, 1, 1])
  >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
  >>> roc_auc_score(y_true, y_scores)
  0.75

在多标签分类问题中，函数 :func:`roc_auc_score` 被扩展到计算所有标签上的平均值，就像这个  :ref:`above <average>` 一样。

与这些 the subset accuracy, the Hamming loss, 或 the F1 score 相比, ROC 不需要对每个label都优化一个阈值。
如果预测输出已经被二值化(binarized)，那么函数 :func:`roc_auc_score` 也可被用于多类分类问题中。

在那些高虚警率(false positive rate)不被容忍的情况下，:func:`roc_auc_score` 函数的参数 ``max_fpr`` 可被用来把ROC曲线累加到一个给定的限制( 
can be used to summarize the ROC curve up to the given limit)。


.. image:: ../auto_examples/model_selection/images/sphx_glr_plot_roc_002.png
   :target: ../auto_examples/model_selection/plot_roc.html
   :scale: 75
   :align: center

.. topic:: 案例:

  * See :ref:`sphx_glr_auto_examples_model_selection_plot_roc.py`
    for an example of using ROC to
    evaluate the quality of the output of a classifier.

  * See :ref:`sphx_glr_auto_examples_model_selection_plot_roc_crossval.py`
    for an example of using ROC to
    evaluate classifier output quality, using cross-validation.

  * See :ref:`sphx_glr_auto_examples_applications_plot_species_distribution_modeling.py`
    for an example of using ROC to
    model species distribution.

.. _zero_one_loss:

Zero one loss
--------------

函数 :func:`zero_one_loss` 计算在一个样本上的 0-1 分类损失(:math:`L_{0-1}`) 的和或均值。
默认情况下，该函数会在样本上进行归一化(normalize)。如果想要获得 :math:`L_{0-1}` 的和，
把 ``normalize`` 设为 ``False`` 。

在多标签分类问题中，函数 :func:`zero_one_loss` 给一个子集评分为 1 如果这个子集的真实标签与预测严格匹配，反之如果有任何一处不匹配则评分为 0.
默认情况下，函数返回不完美预测子集的百分比(the percentage of imperfectly predicted subsets)。
如果想要获得这些不完美预测子集的数量，只需要把参数 ``normalize`` 设置成 ``False``。

如果 :math:`\hat{y}_i` 是 第 :math:`i` 个样本的预测值，:math:`y_i` 是对应的真值，那么 0-1损失 :math:`L_{0-1}` 定义如下：

.. math::

   L_{0-1}(y_i, \hat{y}_i) = 1(\hat{y}_i \not= y_i)

其中 :math:`1(x)` 是示性函数(`indicator function <https://en.wikipedia.org/wiki/Indicator_function>`_)。


  >>> from sklearn.metrics import zero_one_loss
  >>> y_pred = [1, 2, 3, 4]
  >>> y_true = [2, 2, 3, 4]
  >>> zero_one_loss(y_true, y_pred)
  0.25
  >>> zero_one_loss(y_true, y_pred, normalize=False)
  1

In the multilabel case with binary label indicators, where the first label
set [0,1] has an error: ::

  >>> zero_one_loss(np.array([[0, 1], [1, 1]]), np.ones((2, 2)))
  0.5

  >>> zero_one_loss(np.array([[0, 1], [1, 1]]), np.ones((2, 2)),  normalize=False)
  1

.. topic:: 示例:

  * See :ref:`sphx_glr_auto_examples_feature_selection_plot_rfe_with_cross_validation.py`
    for an example of zero one loss usage to perform recursive feature
    elimination with cross-validation.

.. _brier_score_loss:

Brier score loss
----------------

:func:`brier_score_loss` 函数计算用于二分类的 `Brier score <https://en.wikipedia.org/wiki/Brier_score>`_，
引用维基百科的话说:

    "The Brier score 是一个用于度量概率性预测(probabilistic predictions)的准确率的合适的评分函数。
    它可以被用到某些任务中，在这些任务里面 预测必须分配概率到一个由互斥离散输出组成的集合上。
    It is applicable to tasks in which predictions
    must assign probabilities to a set of mutually exclusive discrete outcomes."

该函数返回实际输出(actual outcome)和可能输出(possible outcome)的预测概率之间的平均平方差的得分。
实际输出必须是 1 或 0 (true or false)，而实际输出的预测概率可以是一个介于0和1之间的数。

brier score loss 也是一个介于 0 到 1 之间的数，而且得分越低(也就是平均平方误差)则预测越精确。
它可被认为是对一组概率性预测的 "calibration" 的度量。(It can be thought of as a measure of the 
"calibration" of a set of probabilistic predictions.)

.. math::

   BS = \frac{1}{N} \sum_{t=1}^{N}(f_t - o_t)^2

其中 : :math:`N` 是预测的总数, :math:`f_t` 是实际输出 :math:`o_t` 的预测出的概率(predicted probability)。

下面是该函数的用法示例：::

    >>> import numpy as np
    >>> from sklearn.metrics import brier_score_loss
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_true_categorical = np.array(["spam", "ham", "ham", "spam"])
    >>> y_prob = np.array([0.1, 0.9, 0.8, 0.4])
    >>> y_pred = np.array([0, 1, 1, 0])
    >>> brier_score_loss(y_true, y_prob)
    0.055
    >>> brier_score_loss(y_true, 1 - y_prob, pos_label=0)
    0.055
    >>> brier_score_loss(y_true_categorical, y_prob, pos_label="ham")
    0.055
    >>> brier_score_loss(y_true, y_prob > 0.5)
    0.0


.. topic:: 案例:

  * See :ref:`sphx_glr_auto_examples_calibration_plot_calibration.py`
    for an example of Brier score loss usage to perform probability
    calibration of classifiers.

.. topic:: 参考文献:

  * G. Brier, `Verification of forecasts expressed in terms of probability
    <ftp://ftp.library.noaa.gov/docs.lib/htdocs/rescue/mwr/078/mwr-078-01-0001.pdf>`_,
    Monthly weather review 78.1 (1950)

.. _multilabel_ranking_metrics:

多标签排序指标
==========================

.. currentmodule:: sklearn.metrics

在多标签学习中，每个样本可以有任意数量的真实标签与其自身关联。学习的目标是对真实的标签给出高的得分和更好的排序。
(In multilabel learning, each sample can have any number of ground truth labels
associated with it. The goal is to give high scores and better rank to
the ground truth labels.)

.. _coverage_error:

Coverage error
--------------

The :func:`coverage_error` function computes the average number of labels that
have to be included in the final prediction such that all true labels
are predicted. This is useful if you want to know how many top-scored-labels
you have to predict in average without missing any true one. The best value
of this metrics is thus the average number of true labels.

.. note::

    Our implementation's score is 1 greater than the one given in Tsoumakas
    et al., 2010. This extends it to handle the degenerate case in which an
    instance has 0 true labels.

Formally, given a binary indicator matrix of the ground truth labels
:math:`y \in \left\{0, 1\right\}^{n_\text{samples} \times n_\text{labels}}` and the
score associated with each label
:math:`\hat{f} \in \mathbb{R}^{n_\text{samples} \times n_\text{labels}}`,
the coverage is defined as

.. math::
  coverage(y, \hat{f}) = \frac{1}{n_{\text{samples}}}
    \sum_{i=0}^{n_{\text{samples}} - 1} \max_{j:y_{ij} = 1} \text{rank}_{ij}

with :math:`\text{rank}_{ij} = \left|\left\{k: \hat{f}_{ik} \geq \hat{f}_{ij} \right\}\right|`.
Given the rank definition, ties in ``y_scores`` are broken by giving the
maximal rank that would have been assigned to all tied values.

Here is a small example of usage of this function::

    >>> import numpy as np
    >>> from sklearn.metrics import coverage_error
    >>> y_true = np.array([[1, 0, 0], [0, 0, 1]])
    >>> y_score = np.array([[0.75, 0.5, 1], [1, 0.2, 0.1]])
    >>> coverage_error(y_true, y_score)
    2.5

.. _label_ranking_average_precision:

Label ranking average precision
-------------------------------

The :func:`label_ranking_average_precision_score` function
implements label ranking average precision (LRAP). This metric is linked to
the :func:`average_precision_score` function, but is based on the notion of
label ranking instead of precision and recall.

Label ranking average precision (LRAP) averages over the samples the answer to
the following question: for each ground truth label, what fraction of
higher-ranked labels were true labels? This performance measure will be higher
if you are able to give better rank to the labels associated with each sample.
The obtained score is always strictly greater than 0, and the best value is 1.
If there is exactly one relevant label per sample, label ranking average
precision is equivalent to the `mean
reciprocal rank <https://en.wikipedia.org/wiki/Mean_reciprocal_rank>`_.

Formally, given a binary indicator matrix of the ground truth labels
:math:`y \in \left\{0, 1\right\}^{n_\text{samples} \times n_\text{labels}}`
and the score associated with each label
:math:`\hat{f} \in \mathbb{R}^{n_\text{samples} \times n_\text{labels}}`,
the average precision is defined as

.. math::
  LRAP(y, \hat{f}) = \frac{1}{n_{\text{samples}}}
    \sum_{i=0}^{n_{\text{samples}} - 1} \frac{1}{||y_i||_0}
    \sum_{j:y_{ij} = 1} \frac{|\mathcal{L}_{ij}|}{\text{rank}_{ij}}


where
:math:`\mathcal{L}_{ij} = \left\{k: y_{ik} = 1, \hat{f}_{ik} \geq \hat{f}_{ij} \right\}`,
:math:`\text{rank}_{ij} = \left|\left\{k: \hat{f}_{ik} \geq \hat{f}_{ij} \right\}\right|`,
:math:`|\cdot|` computes the cardinality of the set (i.e., the number of
elements in the set), and :math:`||\cdot||_0` is the :math:`\ell_0` "norm"
(which computes the number of nonzero elements in a vector).

Here is a small example of usage of this function::

    >>> import numpy as np
    >>> from sklearn.metrics import label_ranking_average_precision_score
    >>> y_true = np.array([[1, 0, 0], [0, 0, 1]])
    >>> y_score = np.array([[0.75, 0.5, 1], [1, 0.2, 0.1]])
    >>> label_ranking_average_precision_score(y_true, y_score) # doctest: +ELLIPSIS
    0.416...

.. _label_ranking_loss:

Ranking loss
------------

The :func:`label_ranking_loss` function computes the ranking loss which
averages over the samples the number of label pairs that are incorrectly
ordered, i.e. true labels have a lower score than false labels, weighted by
the inverse of the number of ordered pairs of false and true labels.
The lowest achievable ranking loss is zero.

Formally, given a binary indicator matrix of the ground truth labels
:math:`y \in \left\{0, 1\right\}^{n_\text{samples} \times n_\text{labels}}` and the
score associated with each label
:math:`\hat{f} \in \mathbb{R}^{n_\text{samples} \times n_\text{labels}}`,
the ranking loss is defined as

.. math::
  \text{ranking\_loss}(y, \hat{f}) =  \frac{1}{n_{\text{samples}}}
    \sum_{i=0}^{n_{\text{samples}} - 1} \frac{1}{||y_i||_0(n_\text{labels} - ||y_i||_0)}
    \left|\left\{(k, l): \hat{f}_{ik} \leq \hat{f}_{il}, y_{ik} = 1, y_{il} = 0 \right\}\right|

where :math:`|\cdot|` computes the cardinality of the set (i.e., the number of
elements in the set) and :math:`||\cdot||_0` is the :math:`\ell_0` "norm"
(which computes the number of nonzero elements in a vector).

Here is a small example of usage of this function::

    >>> import numpy as np
    >>> from sklearn.metrics import label_ranking_loss
    >>> y_true = np.array([[1, 0, 0], [0, 0, 1]])
    >>> y_score = np.array([[0.75, 0.5, 1], [1, 0.2, 0.1]])
    >>> label_ranking_loss(y_true, y_score) # doctest: +ELLIPSIS
    0.75...
    >>> # With the following prediction, we have perfect and minimal loss
    >>> y_score = np.array([[1.0, 0.1, 0.2], [0.1, 0.2, 0.9]])
    >>> label_ranking_loss(y_true, y_score)
    0.0


.. topic:: 参考文献:

  * Tsoumakas, G., Katakis, I., & Vlahavas, I. (2010). Mining multi-label data. In
    Data mining and knowledge discovery handbook (pp. 667-685). Springer US.

.. _regression_metrics:

回归问题的指标
===================

.. currentmodule:: sklearn.metrics

:mod:`sklearn.metrics` 模块 实现了若干 loss, score, 和 工具函数 来度量回归算法的性能。
他们中的其中一些已经被加强用来处理多输出问题，例如： :func:`mean_squared_error`,
:func:`mean_absolute_error`, :func:`explained_variance_score` 和
:func:`r2_score`.


这些函数都有一个关键字参数 ``multioutput`` 用来指定对每个独立目标的得分或损失进行平均的方式。
默认的参数取值是 ``'uniform_average'`` ，也就是对所有目标输出的得分或损失进行均匀加权后再取平均的方式。
如果一个shape为 ``(n_outputs,)`` 的 ``ndarray`` 被传入，那么它的每个元素被解释为权重，然后指标函数就会返回对应的加权平均值。
are interpreted as weights and an according weighted average is
returned. 如果 参数 ``multioutput`` 被设置为 ``'raw_values'`` , 那么所有未经改变的单独的得分或者损失将会被指标函数作为数组返回，
返回数组的shape是 ``(n_outputs,)``。


函数 :func:`r2_score` 和 :func:`explained_variance_score` 的 ``multioutput`` 参数 还可接受另外的一种取值：``'variance_weighted'``。
这个参数选项会导致每个单独的得分被加权，而权重数值则恰好是对应目标变量的方差。这个参数设置量化了全局捕获的未缩放的方差(globally captured 
unscaled variance)。如果多个目标变量有不同的尺度，那么这个加权得分将会把更多的重要性放在具有良好解释的方差较高的变量上。
为了向后兼容， ``multioutput='variance_weighted'`` 是函数 :func:`r2_score` 的默认取值, 在后面的sklearn版本中将会改为 ``'uniform_average'`` 。

.. _explained_variance_score:

Explained variance score
-------------------------

函数 :func:`explained_variance_score` 计算 `explained variance regression score <https://en.wikipedia.org/wiki/Explained_variation>`_.

如果 :math:`\hat{y}_i` 是 :math:`i`-th 样本的预测值, 并且 :math:`y_i` 是对应的真实目标值,
:math:`Var` 是 `Variance <https://en.wikipedia.org/wiki/Variance>`_, 即 标准差的平方,
则 explained variance 用下面的方法估计得到:

.. math::

  \texttt{explained\_{}variance}(y, \hat{y}) = 1 - \frac{Var\{ y - \hat{y}\}}{Var\{y\}}

最好的得分是 1.0, explained variance 的值越低越不好。

下面是 :func:`explained_variance_score` 函数的用法示例::

    >>> from sklearn.metrics import explained_variance_score
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> explained_variance_score(y_true, y_pred)  # doctest: +ELLIPSIS
    0.957...
    >>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
    >>> y_pred = [[0, 2], [-1, 2], [8, -5]]
    >>> explained_variance_score(y_true, y_pred, multioutput='raw_values')
    ... # doctest: +ELLIPSIS
    array([0.967..., 1.        ])
    >>> explained_variance_score(y_true, y_pred, multioutput=[0.3, 0.7])
    ... # doctest: +ELLIPSIS
    0.990...

.. _mean_absolute_error:

Mean absolute error
-------------------

函数 :func:`mean_absolute_error` 计算平均绝对误差(`mean absolute error <https://en.wikipedia.org/wiki/Mean_absolute_error>`_), 
它是一个对应于 绝对误差损失 或 :math:`l1`-norm损失 的期望值的风险指标。

如果 :math:`\hat{y}_i` 是 :math:`i`-th 样本的预测值, 并且 :math:`y_i` 是对应的真实值, 则在 :math:`n_{\text{samples}}` 个样本上估计的
平均绝对误差(MAE)定义如下：

.. math::

  \text{MAE}(y, \hat{y}) = \frac{1}{n_{\text{samples}}} \sum_{i=0}^{n_{\text{samples}}-1} \left| y_i - \hat{y}_i \right|.

下面是 :func:`mean_absolute_error` 函数的用法示例::

  >>> from sklearn.metrics import mean_absolute_error
  >>> y_true = [3, -0.5, 2, 7]
  >>> y_pred = [2.5, 0.0, 2, 8]
  >>> mean_absolute_error(y_true, y_pred)
  0.5
  >>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
  >>> y_pred = [[0, 2], [-1, 2], [8, -5]]
  >>> mean_absolute_error(y_true, y_pred)
  0.75
  >>> mean_absolute_error(y_true, y_pred, multioutput='raw_values')
  array([0.5, 1. ])
  >>> mean_absolute_error(y_true, y_pred, multioutput=[0.3, 0.7])
  ... # doctest: +ELLIPSIS
  0.85...

.. _mean_squared_error:

Mean squared error
-------------------

函数 :func:`mean_squared_error` 计算均方误差 ( `mean square error <https://en.wikipedia.org/wiki/Mean_squared_error>`_ ), 
是一个对应于平方（二次）误差或损失的期望值的风险度量。

如果 :math:`\hat{y}_i` 是 :math:`i`-th 样本的预测值, 并且 :math:`y_i` 是对应的真实值, 则在 :math:`n_{\text{samples}}` 个样本上估计的
均方误差（MSE）定义如下：

.. math::

  \text{MSE}(y, \hat{y}) = \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples} - 1} (y_i - \hat{y}_i)^2.

下面是函数 :func:`mean_squared_error` 的例子 ::

  >>> from sklearn.metrics import mean_squared_error
  >>> y_true = [3, -0.5, 2, 7]
  >>> y_pred = [2.5, 0.0, 2, 8]
  >>> mean_squared_error(y_true, y_pred)
  0.375
  >>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
  >>> y_pred = [[0, 2], [-1, 2], [8, -5]]
  >>> mean_squared_error(y_true, y_pred)  # doctest: +ELLIPSIS
  0.7083...

.. topic:: 案例:

  * See :ref:`sphx_glr_auto_examples_ensemble_plot_gradient_boosting_regression.py`
    for an example of mean squared error usage to
    evaluate gradient boosting regression.

.. _mean_squared_log_error:

Mean squared logarithmic error
------------------------------

函数 :func:`mean_squared_log_error` 计算一个与平方对数误差(或损失)的期望值相对应的风险指标(risk metric)：

如果 :math:`\hat{y}_i` 是第 :math:`i` 个样本的预测值, 并且 :math:`y_i` 是对应的真值, 在 :math:`n_{\text{samples}}` 
个样本集上估计的MSLE(mean squared logarithmic error) 定义如下：

.. math::

  \text{MSLE}(y, \hat{y}) = \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples} - 1} (\log_e (1 + y_i) - \log_e (1 + \hat{y}_i) )^2.

其中 :math:`\log_e (x)` 是 :math:`x` 的自然对数。 当目标变量(target variable)呈现指数增长的时候(比如 人口数量，商品月平均销量)使用这个测度指标是最好的。This metric
请注意 这个测度指标对under-predicted estimate的惩罚大于over-predicted estimate。

下面是使用函数 :func:`mean_squared_log_error` 的一个小例子 ::

  >>> from sklearn.metrics import mean_squared_log_error
  >>> y_true = [3, 5, 2.5, 7]
  >>> y_pred = [2.5, 5, 4, 8]
  >>> mean_squared_log_error(y_true, y_pred)  # doctest: +ELLIPSIS
  0.039...
  >>> y_true = [[0.5, 1], [1, 2], [7, 6]]
  >>> y_pred = [[0.5, 2], [1, 2.5], [8, 8]]
  >>> mean_squared_log_error(y_true, y_pred)  # doctest: +ELLIPSIS
  0.044...

.. _median_absolute_error:

Median absolute error
---------------------

函数 :func:`median_absolute_error` 相当有意思，因为它对离群点(outliers)比较鲁棒。 损失的计算是通过对所有样本点上的目标值和预测值的绝对误差取中值进行的。

如果 :math:`\hat{y}_i` 是第 :math:`i` 个样本的预测值, 并且 :math:`y_i` 是对应的真值，在 :math:`n_{\text{samples}}` 个样本上估计的
中值绝对误差(MedAE:median absolute error) 如下定义： 

.. math::

  \text{MedAE}(y, \hat{y}) = \text{median}(\mid y_1 - \hat{y}_1 \mid, \ldots, \mid y_n - \hat{y}_n \mid).

:func:`median_absolute_error` 函数不支持 multioutput。

下面是函数 :func:`median_absolute_error` 用法示例 ::

  >>> from sklearn.metrics import median_absolute_error
  >>> y_true = [3, -0.5, 2, 7]
  >>> y_pred = [2.5, 0.0, 2, 8]
  >>> median_absolute_error(y_true, y_pred)
  0.5

.. _r2_score:

R² score, the coefficient of determination
-------------------------------------------

函数 :func:`r2_score` 计算 R², the `coefficient of determination <https://en.wikipedia.org/wiki/Coefficient_of_determination>`_.
它提供了模型对未来样本的预测好坏的度量(It provides a measure of how well future samples are likely to
be predicted by the model.)。 可能的最好得分是 1.0 而且它可以取负值 (因为模型可能要多坏有多坏). 
对于一个常量模型不管输入特征如何变化，它的预测结果总是 y 的期望值，那么这个模型的 R^2 得分将是0.0。

如果 :math:`\hat{y}_i` 是第 :math:`i` 个样本的预测值, 并且 :math:`y_i` 是对应的真值，
那么 :math:`n_{\text{samples}}` 个样本上估计出的 R² score 定义如下： 

.. math::

  R^2(y, \hat{y}) = 1 - \frac{\sum_{i=0}^{n_{\text{samples}} - 1} (y_i - \hat{y}_i)^2}{\sum_{i=0}^{n_\text{samples} - 1} (y_i - \bar{y})^2}

其中 :math:`\bar{y} =  \frac{1}{n_{\text{samples}}} \sum_{i=0}^{n_{\text{samples}} - 1} y_i`.

下面 是一个使用 函数 :func:`r2_score` 的例子::

  >>> from sklearn.metrics import r2_score
  >>> y_true = [3, -0.5, 2, 7]
  >>> y_pred = [2.5, 0.0, 2, 8]
  >>> r2_score(y_true, y_pred)  # doctest: +ELLIPSIS
  0.948...
  >>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
  >>> y_pred = [[0, 2], [-1, 2], [8, -5]]
  >>> r2_score(y_true, y_pred, multioutput='variance_weighted')
  ... # doctest: +ELLIPSIS
  0.938...
  >>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
  >>> y_pred = [[0, 2], [-1, 2], [8, -5]]
  >>> r2_score(y_true, y_pred, multioutput='uniform_average')
  ... # doctest: +ELLIPSIS
  0.936...
  >>> r2_score(y_true, y_pred, multioutput='raw_values')
  ... # doctest: +ELLIPSIS
  array([0.965..., 0.908...])
  >>> r2_score(y_true, y_pred, multioutput=[0.3, 0.7])
  ... # doctest: +ELLIPSIS
  0.925...


.. topic:: 案例:

  * See :ref:`sphx_glr_auto_examples_linear_model_plot_lasso_and_elasticnet.py`
    for an example of R² score usage to
    evaluate Lasso and Elastic Net on sparse signals.

.. _clustering_metrics:

聚类问题的测度
======================

.. currentmodule:: sklearn.metrics

该 :mod:`sklearn.metrics` 模块实现了一些 loss, score 和 utility 函数. 更多信息请参阅
:ref:`clustering_evaluation` 部分, 例如聚类, 以及用于 :ref:`biclustering_evaluation` 的评测.


.. _dummy_estimators:


无实际意义的估计器(Dummy estimators)
=================

.. currentmodule:: sklearn.dummy

在进行监督学习的过程中，简单的 sanity check（理性检查）包括将人的估计与简单的经验法则进行比较. 
:class:`DummyClassifier` 类实现了一些这样的简单分类策略:

- ``stratified`` 根据训练集中类的分布做出随机预测

- ``most_frequent`` 总是以训练集中频率最高的类标签作为预测.

- ``prior`` 总是给出能够最大化类先验概率的预测 (类似于 ``most_frequent``) 并且 ``predict_proba`` 返回类先验概率.

- ``uniform`` 产生均匀随机猜测式的预测结果.

- ``constant`` 预测的类标签是由用户指定的某个固定标签.
   这种方法的主要动机是 F1-scoring, 这种情况下正类比较少.

请注意, 以上这些所有的策略, ``predict`` 方法完全的忽略了输入数据!

为了展示 :class:`DummyClassifier` 的用法, 让我们首先创建一个非平衡数据集 ::

  >>> from sklearn.datasets import load_iris
  >>> from sklearn.model_selection import train_test_split
  >>> iris = load_iris()
  >>> X, y = iris.data, iris.target
  >>> y[y != 1] = -1
  >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

接着，我们比较 ``SVC`` 和 ``most_frequent`` 的准确性 ::

  >>> from sklearn.dummy import DummyClassifier
  >>> from sklearn.svm import SVC
  >>> clf = SVC(kernel='linear', C=1).fit(X_train, y_train)
  >>> clf.score(X_test, y_test) # doctest: +ELLIPSIS
  0.63...
  >>> clf = DummyClassifier(strategy='most_frequent', random_state=0)
  >>> clf.fit(X_train, y_train)
  DummyClassifier(constant=None, random_state=0, strategy='most_frequent')
  >>> clf.score(X_test, y_test)  # doctest: +ELLIPSIS
  0.57...

我们看到 ``SVC`` 做的并不比 dummy classifier 好。现在我们修改核函数 ::

  >>> clf = SVC(gamma='scale', kernel='rbf', C=1).fit(X_train, y_train)
  >>> clf.score(X_test, y_test)  # doctest: +ELLIPSIS
  0.97...

我们看到准确率提升到将近 100%. 建议采用交叉验证策略, 以更好地估计精度, 如果不是太耗 CPU 的话。
更多信息请参阅 :ref:`cross_validation` 部分. 此外，如果要优化参数空间，强烈建议您使用适当的方法; 
更多详情请参阅 :ref:`grid_search` 部分。

通常来说，当分类器的准确度太接近随机情况时，这可能意味着出现了一些问题: 特征没有帮助, 超参数没有正确调整, 类数目不平衡造成分类器有问题等…

:class:`DummyRegressor` 还实现了四个简单的经验法则来进行回归:

- ``mean`` 总是预测训练目标的平均值.
- ``median`` 总是预测训练目标的中值.
- ``quantile`` 总是预测用户提供的训练目标的 quantile（分位数）.
- ``constant`` 总是预测由用户提供的常数值.

在以上所有的策略中, ``predict`` 方法完全忽略了输入数据.
