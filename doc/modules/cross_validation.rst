
.. _cross_validation:

===================================================
交叉验证:评估估计器的性能
===================================================

.. currentmodule:: sklearn.model_selection

学习 预测函数 的参数，并在相同数据集上进行测试是一种错误的做法: 一个仅给出测试用例标签的模型将会获得极高的分数，
但对于尚未出现过的数据它则无法预测出任何有用的信息。 这种情况称为 过拟合（ **overfitting** ）. 
为了避免这种情况，在进行（监督）机器学习实验时，通常取出部分可利用数据作为测试集（ **test set** ）： ``X_test, y_test``。

需要强调的是这里说的 “experiment(实验)” 并不仅限于学术（academic），因为即使是在商业场景下机器学习也往往是从实验开始的。

利用 scikit-learn 包中的 :func:`train_test_split` 辅助函数可以很快地将实验数据集划分为训练集（training sets）和测试集（test sets）。
 下面让我们载入 iris 数据集，并在此数据集上训练出线性支持向量机 ::

  >>> import numpy as np
  >>> from sklearn.model_selection import train_test_split
  >>> from sklearn import datasets
  >>> from sklearn import svm

  >>> iris = datasets.load_iris()
  >>> iris.data.shape, iris.target.shape
  ((150, 4), (150,))

我们能快速采样到原数据集的 40% 作为测试集，从而测试（评估）我们的分类器 ::

  >>> X_train, X_test, y_train, y_test = train_test_split(
  ...     iris.data, iris.target, test_size=0.4, random_state=0)

  >>> X_train.shape, y_train.shape
  ((90, 4), (90,))
  >>> X_test.shape, y_test.shape
  ((60, 4), (60,))

  >>> clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
  >>> clf.score(X_test, y_test)                           # doctest: +ELLIPSIS
  0.96...

当评价估计器的不同设置（”hyperparameters(超参数)”）时，例如手动为 SVM 设置的 ``C`` 参数， 
由于在训练集上，通过调整参数设置使估计器的性能达到了最佳状态；但 在测试集上 可能会出现过拟合的情况。 
此时，测试集上的信息反馈足以颠覆（"leak"）训练好的模型，评估的指标不再有效反映出模型的泛化性能。 
为了解决此类问题，还应该准备另一部分被称为 “validation set(验证集)” 的数据集，模型训练完成以后在验证集上对模型进行评估。 
当验证集上的评估实验比较成功时，在测试集上进行最后的评估。

然而，通过将原始数据分为3个数据集合，我们就大大减少了可用于模型训练的样本数量， 并且得到的结果依赖于集合对（训练，验证）的随机选择。

这个问题可以通过 交叉验证(`cross-validation <https://en.wikipedia.org/wiki/Cross-validation_(statistics)>`_)（CV 缩写） 来解决。 
交叉验证仍需要测试集做最后的模型评估，但不再需要验证集。
在所有CV方法中， 最基本的方法被称之为，*k*-折交叉验证 。 k-折交叉验证将原始的完整训练集划分为 *k* 个较小的集合（其他方法会在下面描述，主要原则基本相同）。 
每一个 *k* "folds" 都会遵循下面的过程 :

 * 将 :math:`k-1` 份训练集子集作为 training data （训练集）训练模型;
 * 将剩余的 1 份训练集子集作为验证集用于模型验证（也就是利用该数据子集计算模型的性能指标，例如准确率）。

*k*-折交叉验证得出的性能指标是循环计算中每个验证集上的值的平均值。 该方法虽然计算代价很高，但是它不会浪费太多的数据
（如固定任意测试集的情况一样）， 在处理样本数据集较少的问题（例如，逆向推理）时比较有优势。


计算交叉验证的测度
=================================

使用交叉验证最简单的方法是在估计器和数据集上调用 :func:`cross_val_score` 辅助函数。

下面的例子展示了如何通过分割数据，拟合模型和计算连续 5 次的分数（每次不同分割）来估计带有线性核函数的支持向量机在 iris 数据集上的精度::

  >>> from sklearn.model_selection import cross_val_score
  >>> clf = svm.SVC(kernel='linear', C=1)
  >>> scores = cross_val_score(clf, iris.data, iris.target, cv=5)
  >>> scores                                              # doctest: +ELLIPSIS
  array([0.96..., 1.  ..., 0.96..., 0.96..., 1.        ])

评分估计的平均得分和 95% 置信区间由此给出 ::

  >>> print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
  Accuracy: 0.98 (+/- 0.03)

默认情况下，每次 CV 迭代计算分数的方法是调用估计器的 ``score`` 方法。可以通过使用 ``scoring`` 参数来改变计算方式如下 ::

  >>> from sklearn import metrics
  >>> scores = cross_val_score(
  ...     clf, iris.data, iris.target, cv=5, scoring='f1_macro')
  >>> scores                                              # doctest: +ELLIPSIS
  array([0.96..., 1.  ..., 0.96..., 0.96..., 1.        ])

``scoring`` 参数的详情请参阅 :ref:`scoring_parameter` 。 
在 Iris 数据集的情形下，样本在各个目标类别之间是平衡的，因此准确率和 F1-score 几乎相等。

当 ``cv`` 参数是一个整数时，:func:`cross_val_score` 默认使用 :class:`KFold` 或 :class:`StratifiedKFold` 策略，
后者会在估计器派生自 :class:`ClassifierMixin <sklearn.base.ClassifierMixin>` 时使用。

也可以通过传入一个交叉验证迭代器来使用其他交叉验证策略，比如 ::

  >>> from sklearn.model_selection import ShuffleSplit
  >>> n_samples = iris.data.shape[0]
  >>> cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
  >>> cross_val_score(clf, iris.data, iris.target, cv=cv)  # doctest: +ELLIPSIS
  array([0.977..., 0.977..., 1.  ..., 0.955..., 1.        ])

另外一种可选方案是使用一个可迭代生成器作为索引数组产生(train, test) 划分，比如::

  >>> def custom_cv_2folds(X):
  ...     n = X.shape[0]
  ...     i = 1
  ...     while i <= 2:
  ...         idx = np.arange(n * (i - 1) / 2, n * i / 2, dtype=int)
  ...         yield idx, idx
  ...         i += 1
  ...
  >>> custom_cv = custom_cv_2folds(iris.data)
  >>> cross_val_score(clf, iris.data, iris.target, cv=custom_cv)
  array([1.        , 0.973...])

.. topic:: Data transformation with held out data

    正如在训练集中保留的数据上测试一个 predictor （预测器）是很重要的一样，预处理（如标准化，特征选择等）和
    类似的 :ref:`data transformations <data-transforms>` 也应该从训练集中学习，并应用于预测数据以进行预测::

      >>> from sklearn import preprocessing
      >>> X_train, X_test, y_train, y_test = train_test_split(
      ...     iris.data, iris.target, test_size=0.4, random_state=0)
      >>> scaler = preprocessing.StandardScaler().fit(X_train)
      >>> X_train_transformed = scaler.transform(X_train)
      >>> clf = svm.SVC(C=1).fit(X_train_transformed, y_train)
      >>> X_test_transformed = scaler.transform(X_test)
      >>> clf.score(X_test_transformed, y_test)  # doctest: +ELLIPSIS
      0.9333...

    :class:`Pipeline <sklearn.pipeline.Pipeline>` 可以更容易地组合估计器，在交叉验证下使用如下 ::

      >>> from sklearn.pipeline import make_pipeline
      >>> clf = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1))
      >>> cross_val_score(clf, iris.data, iris.target, cv=cv)
      ...                                                 # doctest: +ELLIPSIS
      array([0.977..., 0.933..., 0.955..., 0.933..., 0.977...])

    See :ref:`combining_estimators`.


.. _multimetric_cross_validation:

cross_validate 函数 与 多测度评估
----------------------------------------------------------

``cross_validate`` 函数与 ``cross_val_score`` 在下面的两个方面有些不同 -

- 它允许指定多个指标进行评估.

- 除了测试得分之外，它还会返回一个包含训练得分，拟合次数， score-times （得分次数）的一个字典。

对于单个测度(single metric)评估，其中 ``scoring`` 参数是一个字符串，可调用对象 或 None ， keys 将是 - ``['test_score', 'fit_time', 'score_time']``

而对于多测度(multiple metric)评估，返回值是一个带有以下的 keys 的字典 -
``['test_<scorer1_name>', 'test_<scorer2_name>', 'test_<scorer...>', 'fit_time', 'score_time']``

``return_train_score`` 默认设置为 ``True`` 。 它增加了所有 scorers(得分器) 的训练得分 keys 。如果不需要训练 scores ，则应将其明确设置为 ``False`` 。

你还可以通过设置 ``return_estimator=True`` 来保留在所有训练集上拟合好的估计器。

可以将多个测度指标指定为 list ，tuple 或者是 预定义评分器(predefined scorer)的名字的集合 ::

    >>> from sklearn.model_selection import cross_validate
    >>> from sklearn.metrics import recall_score
    >>> scoring = ['precision_macro', 'recall_macro']
    >>> clf = svm.SVC(kernel='linear', C=1, random_state=0)
    >>> scores = cross_validate(clf, iris.data, iris.target, scoring=scoring,
    ...                         cv=5, return_train_score=False)
    >>> sorted(scores.keys())
    ['fit_time', 'score_time', 'test_precision_macro', 'test_recall_macro']
    >>> scores['test_recall_macro']                       # doctest: +ELLIPSIS
    array([0.96..., 1.  ..., 0.96..., 0.96..., 1.        ])

或将多个测度指标指定为一个字典,该字典将评分器名称(scorer name)映射到预定义或自定义的得分函数(scoring function)::

    >>> from sklearn.metrics.scorer import make_scorer
    >>> scoring = {'prec_macro': 'precision_macro',
    ...            'rec_micro': make_scorer(recall_score, average='macro')}
    >>> scores = cross_validate(clf, iris.data, iris.target, scoring=scoring,
    ...                         cv=5, return_train_score=True)
    >>> sorted(scores.keys())                 # doctest: +NORMALIZE_WHITESPACE
    ['fit_time', 'score_time', 'test_prec_macro', 'test_rec_micro',
     'train_prec_macro', 'train_rec_micro']
    >>> scores['train_rec_micro']                         # doctest: +ELLIPSIS
    array([0.97..., 0.97..., 0.99..., 0.98..., 0.98...])

这里是一个使用单一测度指标(single metric)的 ``cross_validate`` 的例子 ::

    >>> scores = cross_validate(clf, iris.data, iris.target,
    ...                         scoring='precision_macro', cv=5,
    ...                         return_estimator=True)
    >>> sorted(scores.keys())
    ['estimator', 'fit_time', 'score_time', 'test_score', 'train_score']


通过cross-validation获得预测
-----------------------------------------

除了返回结果不同，函数 :func:`cross_val_predict` 具有和 :func:`cross_val_score` 相同的接口， 对于每一个输入的元素，
如果其在测试集合中，将会得到预测结果。只有那些将所有元素分配到一个测试集合仅一次的交叉验证策略可以在这个函数中使用（否则会抛出一个异常）。
Only cross-validation strategies that assign all elements to a test set exactly once
can be used (otherwise, an exception is raised).


.. warning:: 交叉预测(cross_val_predict)使用不当的注记

    :func:`cross_val_predict` 函数的结果可能会与 :func:`cross_val_score` 函数的结果不一样，因为在这两种方法中元素的分组方式不一样
    (elements are grouped in different ways) 。
    函数 :func:`cross_val_score` 在所有交叉验证的折子(cross-validation folds)上取平均。但是， 函数 :func:`cross_val_predict` 只是
    简单的返回由若干不同模型预测出的标签或概率(labels or probabilities) 。 因此, :func:`cross_val_predict` 
    不是一种适当的泛化错误(generalisation error)的度量。 


函数 :func:`cross_val_predict` 比较适合做下列事儿:
  - 从不同模型获得的预测结果的可视化。
  - 模型混合: When predictions of one supervised estimator are used to
    train another estimator in ensemble methods.


可用的交叉验证迭代器(cross validation iterators) 会在下面的章节介绍：

.. topic:: 案例

    * :ref:`sphx_glr_auto_examples_model_selection_plot_roc_crossval.py`,
    * :ref:`sphx_glr_auto_examples_feature_selection_plot_rfe_with_cross_validation.py`,
    * :ref:`sphx_glr_auto_examples_model_selection_plot_grid_search_digits.py`,
    * :ref:`sphx_glr_auto_examples_model_selection_grid_search_text_feature_extraction.py`,
    * :ref:`sphx_glr_auto_examples_model_selection_plot_cv_predict.py`,
    * :ref:`sphx_glr_auto_examples_model_selection_plot_nested_cross_validation_iris.py`.

交叉验证迭代器
==========================

接下来的部分列出了一些用于生成索引标号，用于在不同的交叉验证策略中生成数据划分的工具。

.. _iid_cv:

用于独立同分布(i.i.d.)数据的Cross-validation迭代器
------------------------------------------

假设一些数据是独立的和同分布的 (i.i.d); 假定所有的样本来源于相同的生成过程，
并假设生成过程没有记忆过去生成的样本。

在这种情况下可以使用下面的交叉验证器。

**NOTE**

尽管 i.i.d 数据是机器学习理论中的一个常见假设，但在实践中很少成立。如果知道样本是使用时间相关的过程生成的，
则使用 :ref:`time-series aware cross-validation scheme <timeseries_cv>` 更安全。 
同样，如果我们知道生成过程具有分组结构(group structure)（从不同 subjects（主体），experiments（实验），
measurement devices（测量设备）收集的样本），则使用 :ref:`group-wise cross-validation <group_cv>` 更安全。


K-折法(K-fold)
^^^^^^

:class:`KFold` 将所有的样例划分为 :math:`k` 个组，称为折叠(fold)（如果 :math:`k = n`，这等价于 *Leave One Out* 策略），都具有相同的大小（如果可能）。
预测函数学习时使用其中的 :math:`k - 1` 个折叠中的数据，最后一个剩下的折叠会用于测试。

在 4 个样例的数据集上使用 2-fold 交叉验证的例子::

  >>> import numpy as np
  >>> from sklearn.model_selection import KFold

  >>> X = ["a", "b", "c", "d"]
  >>> kf = KFold(n_splits=2)
  >>> for train, test in kf.split(X):
  ...     print("%s %s" % (train, test))
  [2 3] [0 1]
  [0 1] [2 3]

这个例子是关于交叉验证的可视化的。请注意 :class:`KFold` is not affected by classes or groups.

.. figure:: ../auto_examples/model_selection/images/sphx_glr_plot_cv_indices_004.png
   :target: ../auto_examples/model_selection/plot_cv_indices.html
   :align: center
   :scale: 75%

每个折叠由两个 arrays 组成，第一个作为 *training set* ，另一个作为 *test set* 。 由此，可以通过使用 numpy 的索引创建训练/测试集合::

  >>> X = np.array([[0., 0.], [1., 1.], [-1., -1.], [2., 2.]])
  >>> y = np.array([0, 1, 0, 1])
  >>> X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]


重复K-折法
^^^^^^^^^^^^^^^

:class:`RepeatedKFold` 重复 K-Fold n 次。当你需要运行 :class:`KFold` n 次 时可以使用它，在每次重复中产生不同的分割。

2折 K-Fold 重复 2 次的示例 ::

  >>> import numpy as np
  >>> from sklearn.model_selection import RepeatedKFold
  >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
  >>> random_state = 12883823
  >>> rkf = RepeatedKFold(n_splits=2, n_repeats=2, random_state=random_state)
  >>> for train, test in rkf.split(X):
  ...     print("%s %s" % (train, test))
  ...
  [2 3] [0 1]
  [0 1] [2 3]
  [0 2] [1 3]
  [1 3] [0 2]


类似地， :class:`RepeatedStratifiedKFold` 在每个重复中以不同的随机化重复 n 次 Stratified K-Fold 。


留一法(LOO)
^^^^^^^^^^^^^^^^^^^

:class:`LeaveOneOut` (或 LOO) 是一个简单的交叉验证。每个学习集都是通过除去一个样本以外的所有剩余样本创建的，
测试集是被留下的样本。 因此，对于 :math:`n` 个样本，我们有 :math:`n` 个不同的训练集和 :math:`n` 个不同的测试集。
这种交叉验证程序不会浪费太多数据，因为只有一个样本是从训练集中删除掉的 ::

  >>> from sklearn.model_selection import LeaveOneOut

  >>> X = [1, 2, 3, 4]
  >>> loo = LeaveOneOut()
  >>> for train, test in loo.split(X):
  ...     print("%s %s" % (train, test))
  [1 2 3] [0]
  [0 2 3] [1]
  [0 1 3] [2]
  [0 1 2] [3]


将LOO用于选择模型的潜在用户应该权衡一些已知的警告。
当把 LOO 与 :math:`k`-fold CV 进行比较时，可以从 :math:`n` 样本中构建 :math:`n` 模型，而不是 :math:`k` 模型，其中 :math:`n > k` 。 
此外，每个模型在 :math:`n - 1` 个样本而不是在 :math:`(k-1) n / k` 上进行训练。在两种方式中，假设 :math:`k` 不是太大，
并且 :math:`k < n` ， LOO 比 :math:`k`-fold CV 计算开销更加昂贵。

就精度而言，作为测试误差的估计器,LOO 经常导致较高的方差。直观地说，因为 :math:`n` 个样本中的 :math:`n - 1` 个被用来构建每个模型，
所以不同的两个fold包含的 :math:`n - 1` 个训练样本几乎是一样的，所以从这些几乎一样的folds上构建的所有模型相互之间几乎完全相同，
并且这些模型与从整个训练集建立的模型也几乎完全一样。

但是，如果学习曲线对于所讨论的训练大小是陡峭的，那么 5- 或 10- 折交叉验证会过高的估计泛化误差(overestimate the generalization error)。

作为一般规则，大多数作者和经验证据表明， 5-folds 或者 10-folds 交叉验证应该优于 LOO 。


.. topic:: 参考文献:

 * `<http://www.faqs.org/faqs/ai-faq/neural-nets/part3/section-12.html>`_;
 * T. Hastie, R. Tibshirani, J. Friedman,  `The Elements of Statistical Learning
   <https://web.stanford.edu/~hastie/ElemStatLearn/>`_, Springer 2009
 * L. Breiman, P. Spector `Submodel selection and evaluation in regression: The X-random case
   <http://digitalassets.lib.berkeley.edu/sdtr/ucb/text/197.pdf>`_, International Statistical Review 1992;
 * R. Kohavi, `A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection
   <http://web.cs.iastate.edu/~jtian/cs573/Papers/Kohavi-IJCAI-95.pdf>`_, Intl. Jnt. Conf. AI
 * R. Bharat Rao, G. Fung, R. Rosales, `On the Dangers of Cross-Validation. An Experimental Evaluation
   <https://people.csail.mit.edu/romer/papers/CrossVal_SDM08.pdf>`_, SIAM 2008;
 * G. James, D. Witten, T. Hastie, R Tibshirani, `An Introduction to
   Statistical Learning <http://www-bcf.usc.edu/~gareth/ISL>`_, Springer 2013.


留P法(LPO)
^^^^^^^^^^^^^^^^^

:class:`LeavePOut` 与 :class:`LeaveOneOut` 非常相似，因为它通过从整个集合中删除 :math:`p` 个样本来创建所有可能的 训练/测试 集。
对于 :math:`n` 个样本，这产生了 :math:`{n \choose p}` 个 训练-测试 对。与 :class:`LeaveOneOut` 和 :class:`KFold` 不同，
当 :math:`p > 1` 时，测试集会发生重叠。

在有 4 个样例的数据集上使用 Leave-2-Out 的例子 ::

  >>> from sklearn.model_selection import LeavePOut

  >>> X = np.ones(4)
  >>> lpo = LeavePOut(p=2)
  >>> for train, test in lpo.split(X):
  ...     print("%s %s" % (train, test))
  [2 3] [0 1]
  [1 3] [0 2]
  [1 2] [0 3]
  [0 3] [1 2]
  [0 2] [1 3]
  [0 1] [2 3]


.. _ShuffleSplit:

随机置换交叉验证,也就是: 洗牌 & 分割
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`ShuffleSplit`

:class:`ShuffleSplit` 迭代器 将会生成一个用户给定数量的独立的训练/测试数据划分。样例首先被打散然后划分为一对训练测试集合。

可以通过显示的设定 ``random_state`` ，使得伪随机生成器的结果可以重复。

这是一个使用 :class:`ShuffleSplit` 迭代器  的小例子::

  >>> from sklearn.model_selection import ShuffleSplit
  >>> X = np.arange(10)
  >>> ss = ShuffleSplit(n_splits=5, test_size=0.25,
  ...     random_state=0)
  >>> for train_index, test_index in ss.split(X):
  ...     print("%s %s" % (train_index, test_index))
  [9 1 6 7 3 0 5] [2 8 4]
  [2 9 8 0 6 7 4] [3 5 1]
  [4 5 1 0 6 9 7] [2 3 8]
  [2 7 5 8 0 3 4] [6 1 9]
  [4 1 0 6 8 9 3] [5 2 7]

这个例子是关于交叉验证的可视化的。请注意 :class:`ShuffleSplit` is not affected by classes or groups。

.. figure:: ../auto_examples/model_selection/images/sphx_glr_plot_cv_indices_006.png
   :target: ../auto_examples/model_selection/plot_cv_indices.html
   :align: center
   :scale: 75%

:class:`ShuffleSplit` 可以替代 :class:`KFold` 交叉验证，因为其提供了细致的迭代数量，训练/测试划分的数量和 每一个划分中样例所占的比例等的控制。

基于类标签分层的Cross-validation迭代器.
---------------------------------------------------------------------

一些分类问题在目标类别的分布上可能表现出很大的不平衡性：例如，可能会出现比正样本多数倍的负样本。在这种情况下，建议采用如 
StratifiedKFold 和 StratifiedShuffleSplit 中实现的分层抽样方法(stratified sampling)，确保相对的类别频率在每个训练和验证 fold 中大致保留。

Stratified k-fold
^^^^^^^^^^^^^^^^^

:class:`StratifiedKFold` 是 *k-fold* 的变种，会返回 *stratified* folds ：每个小集合中，各个类别的样本比例大致和完整数据集中的样本类别比例相同。

在有10个样本的，有两个略不均衡类别的数据集上进行 stratified 3-fold 交叉验证的例子 ::

  >>> from sklearn.model_selection import StratifiedKFold

  >>> X = np.ones(10)
  >>> y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  >>> skf = StratifiedKFold(n_splits=3)
  >>> for train, test in skf.split(X, y):
  ...     print("%s %s" % (train, test))
  [2 3 6 7 8 9] [0 1 4 5]
  [0 1 3 4 5 8 9] [2 6 7]
  [0 1 2 4 5 6 7] [3 8 9]

Here is a visualization of the cross-validation behavior.

.. figure:: ../auto_examples/model_selection/images/sphx_glr_plot_cv_indices_007.png
   :target: ../auto_examples/model_selection/plot_cv_indices.html
   :align: center
   :scale: 75%

:class:`RepeatedStratifiedKFold` 类可用于在每次重复中用不同的随机化重复stratified k-fold n 次。


Stratified Shuffle Split
^^^^^^^^^^^^^^^^^^^^^^^^

:class:`StratifiedShuffleSplit` 是 ShuffleSplit 的一个变种，会返回 stratified splits，*i.e*：创建一个划分，
但是划分中每个类的样本比例和完整数据集中的相同。

Here is a visualization of the cross-validation behavior.

.. figure:: ../auto_examples/model_selection/images/sphx_glr_plot_cv_indices_009.png
   :target: ../auto_examples/model_selection/plot_cv_indices.html
   :align: center
   :scale: 75%

.. _group_cv:

用于分组数据的交叉验证迭代器.
--------------------------------------------

如果潜在的生成过程产生几组样本，而每个组内的样本是相互依赖的 ，那么 i.i.d. 假设将会被打破。
(The i.i.d. assumption is broken if the underlying generative process yield groups of dependent samples.)

这样的数据分组是与特定领域相关的(domain specific)。一个例子是从多个患者收集医学数据，从每个患者身上采集多个样本。而这样的数据很可能取决于个人群体。
在我们的例子中，每个样本的患者 ID 将是其 group identifier （组标识符）。

在这种情况下，我们想知道在一组特定的 groups 上训练的模型是否能很好地适用于看不见的 group 。为了衡量这一点，
我们需要确保验证对象中的所有样本来自配对训练折叠中完全没有表示的组。

下面的交叉验证分割器(cross-validation splitters)可以用来做到这一点。 样本的 grouping identifier （分组标识符） 通过 ``groups`` 参数指定。


Group k-fold
^^^^^^^^^^^^

:class:`GroupKFold` is a variation of k-fold which ensures that the same group is
not represented in both testing and training sets. For example if the data is
obtained from different subjects with several samples per-subject and if the
model is flexible enough to learn from highly person specific features it
could fail to generalize to new subjects. :class:`GroupKFold` makes it possible
to detect this kind of overfitting situations.

Imagine you have three subjects, each with an associated number from 1 to 3::

  >>> from sklearn.model_selection import GroupKFold

  >>> X = [0.1, 0.2, 2.2, 2.4, 2.3, 4.55, 5.8, 8.8, 9, 10]
  >>> y = ["a", "b", "b", "b", "c", "c", "c", "d", "d", "d"]
  >>> groups = [1, 1, 1, 2, 2, 2, 3, 3, 3, 3]

  >>> gkf = GroupKFold(n_splits=3)
  >>> for train, test in gkf.split(X, y, groups=groups):
  ...     print("%s %s" % (train, test))
  [0 1 2 3 4 5] [6 7 8 9]
  [0 1 2 6 7 8 9] [3 4 5]
  [3 4 5 6 7 8 9] [0 1 2]

Each subject is in a different testing fold, and the same subject is never in
both testing and training. Notice that the folds do not have exactly the same
size due to the imbalance in the data.

Here is a visualization of the cross-validation behavior.

.. figure:: ../auto_examples/model_selection/images/sphx_glr_plot_cv_indices_005.png
   :target: ../auto_examples/model_selection/plot_cv_indices.html
   :align: center
   :scale: 75%

Leave One Group Out
^^^^^^^^^^^^^^^^^^^

:class:`LeaveOneGroupOut` is a cross-validation scheme which holds out
the samples according to a third-party provided array of integer groups. This
group information can be used to encode arbitrary domain specific pre-defined
cross-validation folds.

Each training set is thus constituted by all the samples except the ones
related to a specific group.

For example, in the cases of multiple experiments, :class:`LeaveOneGroupOut`
can be used to create a cross-validation based on the different experiments:
we create a training set using the samples of all the experiments except one::

  >>> from sklearn.model_selection import LeaveOneGroupOut

  >>> X = [1, 5, 10, 50, 60, 70, 80]
  >>> y = [0, 1, 1, 2, 2, 2, 2]
  >>> groups = [1, 1, 2, 2, 3, 3, 3]
  >>> logo = LeaveOneGroupOut()
  >>> for train, test in logo.split(X, y, groups=groups):
  ...     print("%s %s" % (train, test))
  [2 3 4 5 6] [0 1]
  [0 1 4 5 6] [2 3]
  [0 1 2 3] [4 5 6]

Another common application is to use time information: for instance the
groups could be the year of collection of the samples and thus allow
for cross-validation against time-based splits.

Leave P Groups Out
^^^^^^^^^^^^^^^^^^

:class:`LeavePGroupsOut` is similar as :class:`LeaveOneGroupOut`, but removes
samples related to :math:`P` groups for each training/test set.

Example of Leave-2-Group Out::

  >>> from sklearn.model_selection import LeavePGroupsOut

  >>> X = np.arange(6)
  >>> y = [1, 1, 1, 2, 2, 2]
  >>> groups = [1, 1, 2, 2, 3, 3]
  >>> lpgo = LeavePGroupsOut(n_groups=2)
  >>> for train, test in lpgo.split(X, y, groups=groups):
  ...     print("%s %s" % (train, test))
  [4 5] [0 1 2 3]
  [2 3] [0 1 4 5]
  [0 1] [2 3 4 5]

Group Shuffle Split
^^^^^^^^^^^^^^^^^^^

The :class:`GroupShuffleSplit` iterator behaves as a combination of
:class:`ShuffleSplit` and :class:`LeavePGroupsOut`, and generates a
sequence of randomized partitions in which a subset of groups are held
out for each split.

Here is a usage example::

  >>> from sklearn.model_selection import GroupShuffleSplit

  >>> X = [0.1, 0.2, 2.2, 2.4, 2.3, 4.55, 5.8, 0.001]
  >>> y = ["a", "b", "b", "b", "c", "c", "c", "a"]
  >>> groups = [1, 1, 2, 2, 3, 3, 4, 4]
  >>> gss = GroupShuffleSplit(n_splits=4, test_size=0.5, random_state=0)
  >>> for train, test in gss.split(X, y, groups=groups):
  ...     print("%s %s" % (train, test))
  ...
  [0 1 2 3] [4 5 6 7]
  [2 3 6 7] [0 1 4 5]
  [2 3 4 5] [0 1 6 7]
  [4 5 6 7] [0 1 2 3]

Here is a visualization of the cross-validation behavior.

.. figure:: ../auto_examples/model_selection/images/sphx_glr_plot_cv_indices_008.png
   :target: ../auto_examples/model_selection/plot_cv_indices.html
   :align: center
   :scale: 75%

This class is useful when the behavior of :class:`LeavePGroupsOut` is
desired, but the number of groups is large enough that generating all
possible partitions with :math:`P` groups withheld would be prohibitively
expensive.  In such a scenario, :class:`GroupShuffleSplit` provides
a random sample (with replacement) of the train / test splits
generated by :class:`LeavePGroupsOut`.


Predefined Fold-Splits / Validation-Sets
----------------------------------------

For some datasets, a pre-defined split of the data into training- and
validation fold or into several cross-validation folds already
exists. Using :class:`PredefinedSplit` it is possible to use these folds
e.g. when searching for hyperparameters.

For example, when using a validation set, set the ``test_fold`` to 0 for all
samples that are part of the validation set, and to -1 for all other samples.

.. _timeseries_cv:

时间序列数据的交叉验证
------------------------------------

时间序列数据的特点是时间 (*autocorrelation*(自相关性)) 附近的观测之间的相关性。 
然而，传统的交叉验证技术，例如 :class:`KFold` 和 :class:`ShuffleSplit` 假设样本是独立的且分布相同的，
并且在时间序列数据上会导致训练和测试实例之间不合理的相关性（产生广义误差的估计较差）。 
因此，对 “future(未来)” 观测的时间序列数据模型的评估至少与用于训练模型的观测模型非常重要。
为了达到这个目的，一个解决方案是由 :class:`TimeSeriesSplit` 提供的。


时间序列划分
^^^^^^^^^^^^^^^^^

:class:`TimeSeriesSplit` 是 *k-fold* 的一个变体，它首先返回 :math:`k` 折作为训练数据集和 :math:`(k+1)` 折作为测试数据集。 
请注意，与标准的交叉验证方法不同，连续的训练集是超越前者的超集。 另外，它将所有的剩余数据添加到第一个训练分区，它总是用来训练模型。

这个类可以用来交叉验证以固定时间间隔观察到的时间序列数据样本。

对具有 6 个样本的数据集进行 3-split 时间序列交叉验证的示例::

  >>> from sklearn.model_selection import TimeSeriesSplit

  >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
  >>> y = np.array([1, 2, 3, 4, 5, 6])
  >>> tscv = TimeSeriesSplit(n_splits=3)
  >>> print(tscv)  # doctest: +NORMALIZE_WHITESPACE
  TimeSeriesSplit(max_train_size=None, n_splits=3)
  >>> for train, test in tscv.split(X):
  ...     print("%s %s" % (train, test))
  [0 1 2] [3]
  [0 1 2 3] [4]
  [0 1 2 3 4] [5]

Here is a visualization of the cross-validation behavior.

.. figure:: ../auto_examples/model_selection/images/sphx_glr_plot_cv_indices_010.png
   :target: ../auto_examples/model_selection/plot_cv_indices.html
   :align: center
   :scale: 75%

随机洗牌需要注意的地方
===================

(如果数据的顺序不是任意的（比如说，相同标签的样例连续出现），为了获得有意义的交叉验证结果，首先对其进行随机打乱(shuffling)是很有必要的。
然而，当样例不是独立同分布时打乱则是不可行的。例如：样例是相关的文章，以他们发表的时间 进行排序，
这时候如果对数据进行打乱，将会导致模型过拟合，得到一个过高的验证分数：因为验证样例更加相似（在时间上更接近） 于训练数据。

一些交叉验证迭代器， 比如 :class:`KFold` ，有一个内建的在划分数据前进行数据索引打乱的选项。注意: 

* 这种方式仅需要很少的内存就可以打乱数据。
* 默认不会进行打乱，包括设置 ``cv=some_integer`` （直接）k 折叠交叉验证的 :func:`cross_val_score` ，网格搜索等。
  注意 :func:`train_test_split` 会返回一个随机的划分。
* 参数 ``random_state`` 默认设置为 None ，这意为着每次进行 ``KFold(..., shuffle=True)`` 时，打乱都是不同的。 
  然而， ``GridSearchCV`` 通过调用 ``fit`` 方法验证时，将会使用相同的打乱来训练每一组参数。
* 为了保证结果的可重复性（在相同的平台上），应该给 ``random_state`` 设定一个固定的值。

交叉验证与模型选择
====================================

交叉验证迭代器(Cross validation iterators)可以通过网格搜索得到最优的模型超参数，从而直接用于模型选择。 
这是另一部分 :ref:`grid_search` 的主要内容。