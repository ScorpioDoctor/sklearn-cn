.. _svm:

=======================
支持向量机（Support Vector Machines）
=======================

.. currentmodule:: sklearn.svm

**Support vector machines (SVMs)** 是一类监督学习算法，常被用于：:ref:`classification <svm_classification>`,
:ref:`regression <svm_regression>` 和 :ref:`outliers detection <svm_outlier_detection>`.

支持向量机的优点:

    - 在高维空间中非常高效。

    - 即使在数据维度比样本数量大的情况下仍然有效。

    - 在决策函数（称为支持向量）中使用训练集的子集,因此它也是高效利用内存的.

    - 通用性: 不同的核函数与特定的决策函数一一对应.常见的 kernel 已经提供,也可以指定定制的内核.

支持向量机的缺点:

    - 如果特征数量比样本数量大得多, 选择核函数 :ref:`svm_kernels` 和 正则化项以避免过拟合是很关键的。

    - 支持向量机不直接提供概率估计,这些都是使用昂贵的五次交叉验算计算的。 (详情见 :ref:`Scores and probabilities <scores_probabilities>`, 在下文中)

在 scikit-learn 中,支持向量机 支持 稠密样本向量 ( ``numpy.ndarray`` ,可以通过 ``numpy.asarray`` 进行转换) 和 
稀疏样本向量 (任何 ``scipy.sparse`` ) 作为输入。 然而,要使用支持向量机来对稀疏数据作预测,它必须已经在这样的数据上拟合过了。
为了优化性能，使用了 C-ordered ``numpy.ndarray`` (稠密输入) 或者带有 ``dtype=float64`` 的 ``scipy.sparse.csr_matrix`` (稀疏输入) 。


.. _svm_classification:

分类
==============

:class:`SVC`, :class:`NuSVC` 和 :class:`LinearSVC` 能够在指定的数据集上进行多类分类任务的类。


.. figure:: ../auto_examples/svm/images/sphx_glr_plot_iris_001.png
   :target: ../auto_examples/svm/plot_iris.html
   :align: center


:class:`SVC` 和 :class:`NuSVC` 是相似的方法, 但是它们接受的参数集合稍微不同并且数学形式也有所区别
(see section :ref:`svm_mathematical_formulation`)。另一方面， :class:`LinearSVC` 是支持向量机的另一种实现方式，主要用于线性核函数的情况。
注意到 :class:`LinearSVC` 不接受关键参数 ``kernel``, 因为 核函数 已经被假定为线性核啦。它也缺少 :class:`SVC` 和 :class:`NuSVC` 才有的
一些类成员属性, 比如 ``support_``。

和其他分类器一样, :class:`SVC`, :class:`NuSVC` 和 :class:`LinearSVC` 将两个数组作为输入: shape为 ``[n_samples, n_features]`` 的数组 X 作为训练样本,
shape为 ``[n_samples]`` 的数组 y 作为类别标签(字符串或者整数)::


    >>> from sklearn import svm
    >>> X = [[0, 0], [1, 1]]
    >>> y = [0, 1]
    >>> clf = svm.SVC(gamma='scale')
    >>> clf.fit(X, y)  # doctest: +NORMALIZE_WHITESPACE
    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)

在模型拟合好后, 就可以用来预测新的值::

    >>> clf.predict([[2., 2.]])
    array([1])

SVMs的决策函数依赖于训练数据的某些子集，称之为支持向量(support vectors)。这些支持向量的一部分属性可以在成员
``support_vectors_``, ``support_`` 和 ``n_support`` 中找到 ::

    >>> # get support vectors
    >>> clf.support_vectors_
    array([[0., 0.],
           [1., 1.]])
    >>> # get indices of support vectors
    >>> clf.support_ # doctest: +ELLIPSIS
    array([0, 1]...)
    >>> # get number of support vectors for each class
    >>> clf.n_support_ # doctest: +ELLIPSIS
    array([1, 1]...)

.. _svm_multi_class:

多类别分类
--------------------------

:class:`SVC` 和 :class:`NuSVC` 实现了 "one-against-one" 方法 (Knerr et al., 1990) 用于解决多类别分类问题。
如果 ``n_class`` 是类的数量，那么总共需要构建 ``n_class * (n_class - 1) / 2`` 个分类器，其中每一个分类器都是在数据上训练得到的两类分类器。
为了和其他分类器的接口保持一致，``decision_function_shape`` 选项允许把多个"one-against-one"分类器的结果聚集到一个
shape为 ``(n_samples,n_classes)`` 决策函数里边 ::

    >>> X = [[0], [1], [2], [3]]
    >>> Y = [0, 1, 2, 3]
    >>> clf = svm.SVC(gamma='scale', decision_function_shape='ovo')
    >>> clf.fit(X, Y) # doctest: +NORMALIZE_WHITESPACE
    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovo', degree=3, gamma='scale', kernel='rbf',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)
    >>> dec = clf.decision_function([[1]])
    >>> dec.shape[1] # 4 classes: 4*3/2 = 6
    6
    >>> clf.decision_function_shape = "ovr"
    >>> dec = clf.decision_function([[1]])
    >>> dec.shape[1] # 4 classes
    4

另一方面, :class:`LinearSVC` 实现了 "one-vs-the-rest" 多类分类策略, 因此会训练出 n_class 个 模型。 
如果只有两个类，那么就只训练一个模型 ::

    >>> lin_clf = svm.LinearSVC()
    >>> lin_clf.fit(X, Y) # doctest: +NORMALIZE_WHITESPACE
    LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
         intercept_scaling=1, loss='squared_hinge', max_iter=1000,
         multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
         verbose=0)
    >>> dec = lin_clf.decision_function([[1]])
    >>> dec.shape[1]
    4

请参考 :ref:`svm_mathematical_formulation` 小节关于决策函数(decision function)的完整描述。

需要注意的是 :class:`LinearSVC` 也实现了另外一种替代的多类分类策略，称之为 multi-class SVM(由Crammer和Singer提出), 
通过指定参数选项  ``multi_class='crammer_singer'`` 来使用这种策略。
'crammer_singer'多类分类策略的结果具有一致性，而 'one-vs-rest' 多类分类策略的结果却不具有一致性。
但是在实践中，我们通常比较偏爱使用 'one-vs-rest' 策略，因为这两种策略的结果总是非常相似的但是
'one-vs-rest' 策略的执行时间显著少于'crammer_singer'策略。

对于采用 "one-vs-rest" 分类策略的 :class:`LinearSVC` , 属性 ``coef_`` 和 ``intercept_``
的shape分别为 ``[n_class, n_features]`` 和 ``[n_class]`` 。
系数(coefficients)的每一行对应于 ``n_class`` 个 "one-vs-rest" 诸多分类器中的一个。截距(intercepts)的对应规则也是一样的。

对于采用 "one-vs-one" 分类策略的 :class:`SVC`, 属性的布局(the layout of the attributes)更复杂一些。
如果采用了线性内核, 属性 ``coef_`` 和 ``intercept_`` 的shape分别为 ``[n_class * (n_class - 1) / 2, n_features]`` 
和 ``[n_class * (n_class - 1) / 2]`` 。 这与上面描述的 :class:`LinearSVC` 类的布局结构有些相似之处：每一行对应于一个二分类器(binary classifier)。
另一方面， 从 第0类 到 第n类 的顺序为 "0 vs 1", "0 vs 2" , ... "0 vs n", "1 vs 2", "1 vs 3", "1 vs n", . . . "n-1 vs n"。

这段内容说了对偶系数(``dual_coef_``)的shape以及它的含义，这个布局结构有点难以掌握。 ``dual_coef_`` 的shape是 ``[n_class-1, n_SV]`` 。
The columns correspond to the support vectors involved in any
of the ``n_class * (n_class - 1) / 2`` "one-vs-one" classifiers.
Each of the support vectors is used in ``n_class - 1`` classifiers.
The ``n_class - 1`` entries in each row correspond to the dual coefficients
for these classifiers.

我们可以通过一个例子说清楚上面的问题:

Consider a three class problem with class 0 having three support vectors
:math:`v^{0}_0, v^{1}_0, v^{2}_0` and class 1 and 2 having two support vectors
:math:`v^{0}_1, v^{1}_1` and :math:`v^{0}_2, v^{1}_2` respectively.  For each
support vector :math:`v^{j}_i`, there are two dual coefficients.  Let's call
the coefficient of support vector :math:`v^{j}_i` in the classifier between
classes :math:`i` and :math:`k` :math:`\alpha^{j}_{i,k}`.
Then ``dual_coef_`` looks like this:

+------------------------+------------------------+------------------+
|:math:`\alpha^{0}_{0,1}`|:math:`\alpha^{0}_{0,2}`|Coefficients      |
+------------------------+------------------------+for SVs of class 0|
|:math:`\alpha^{1}_{0,1}`|:math:`\alpha^{1}_{0,2}`|                  |
+------------------------+------------------------+                  |
|:math:`\alpha^{2}_{0,1}`|:math:`\alpha^{2}_{0,2}`|                  |
+------------------------+------------------------+------------------+
|:math:`\alpha^{0}_{1,0}`|:math:`\alpha^{0}_{1,2}`|Coefficients      |
+------------------------+------------------------+for SVs of class 1|
|:math:`\alpha^{1}_{1,0}`|:math:`\alpha^{1}_{1,2}`|                  |
+------------------------+------------------------+------------------+
|:math:`\alpha^{0}_{2,0}`|:math:`\alpha^{0}_{2,1}`|Coefficients      |
+------------------------+------------------------+for SVs of class 2|
|:math:`\alpha^{1}_{2,0}`|:math:`\alpha^{1}_{2,1}`|                  |
+------------------------+------------------------+------------------+


.. _scores_probabilities:

得分与概率
------------------------

:class:`SVC` 和 :class:`NuSVC` 的 ``decision_function`` 方法给出了每个样本属于每个类的得分(在二分类问题中每个样本只有一个得分)。
当构造函数选项 ``probability`` 被设置为 ``True``, 类成员概率(class membership probability)估计就被开启了,估计方法 ``predict_proba`` 
和 ``predict_log_proba`` 就会被调用。在二分类问题中，概率以"Platt scaling"的方法被校准(calibrated): logistic regression on the SVM's scores,
fit by an additional cross-validation on the training data.
在多分类情况下, Wu et al. (2004) 对上述方法做出了扩展.

都不用说, 对于大数据集，Platt scaling 方法中使用交叉验证是一个昂贵操作。而且，使用SVM的得分进行概率估计得到的结果是不一致的(inconsistent),
从这个意义上说，得分的最大化并不等价于概率的最大化(the "argmax" of the scores may not be the argmax of the probabilities)。
(比如说, 在二分类问题中, 一个样本可能会被 ``predict`` 标记为属于其中一个根据 ``predict_proba`` 估计出的概率<½的类(a sample may be labeled 
by ``predict`` as belonging to a class that has probability <½ according to ``predict_proba``.)。
Platt的方法还被认为存在一些理论问题。
如果我们需要信任得分(confidence scores), 但是这些信任得分不一定是概率性得分，那么建议设置 ``probability=False`` 并使用 ``decision_function``
而不是 ``predict_proba``。

.. topic:: 参考文献:

 * Wu, Lin and Weng,
   `"Probability estimates for multi-class classification by pairwise coupling"
   <https://www.csie.ntu.edu.tw/~cjlin/papers/svmprob/svmprob.pdf>`_,
   JMLR 5:975-1005, 2004.
 
 
 * Platt
   `"Probabilistic outputs for SVMs and comparisons to regularized likelihood methods"
   <https://www.cs.colorado.edu/~mozer/Teaching/syllabi/6622/papers/Platt1999.pdf>`_.

不均衡问题
--------------------

在某些问题中，我们需要给某些类或个别样本更大的重要性，这时候就要使用关键参数 ``class_weight`` 和 ``sample_weight`` 。

:class:`SVC` (but not :class:`NuSVC`) 在 ``fit`` 方法中实现了关键字参数 ``class_weight``，是一个形式为 ``{class_label : value}`` 的字典,
其中 value 是一个大于0的浮点数，把 ``class_label`` 对应的类的参数 ``C`` 设置为 ``C * value``.

.. figure:: ../auto_examples/svm/images/sphx_glr_plot_separating_hyperplane_unbalanced_001.png
   :target: ../auto_examples/svm/plot_separating_hyperplane_unbalanced.html
   :align: center
   :scale: 75


:class:`SVC`, :class:`NuSVC`, :class:`SVR`, :class:`NuSVR` 和 :class:`OneClassSVM` 在 ``fit`` 方法中
通过关键字参数 ``sample_weight`` 实现了对个别样本进行加权的功能。与关键字参数 ``class_weight`` 类似，``sample_weight`` 
将会为第i个样本把参数 ``C`` 设置成 ``C * sample_weight[i]`` 。

.. figure:: ../auto_examples/svm/images/sphx_glr_plot_weighted_samples_001.png
   :target: ../auto_examples/svm/plot_weighted_samples.html
   :align: center
   :scale: 75


.. topic:: 案例:

 * :ref:`sphx_glr_auto_examples_svm_plot_iris.py`,
 * :ref:`sphx_glr_auto_examples_svm_plot_separating_hyperplane.py`,
 * :ref:`sphx_glr_auto_examples_svm_plot_separating_hyperplane_unbalanced.py`
 * :ref:`sphx_glr_auto_examples_svm_plot_svm_anova.py`,
 * :ref:`sphx_glr_auto_examples_svm_plot_svm_nonlinear.py`
 * :ref:`sphx_glr_auto_examples_svm_plot_weighted_samples.py`,


.. _svm_regression:

回归
==========

支持向量分类方法(The method of Support Vector Classification)可以推广到解决回归问题。
这种方法称为支持向量回归(Support Vector Regression)。

支持向量分类(如上所述)生成的模型仅依赖于训练数据的子集，因为建立模型的代价函数(cost function)不关心超出边际的训练点。
类似地，支持向量回归生成的模型只依赖于训练数据的一个子集，因为用于建立模型的代价函数忽略了任何接近模型预测的训练数据。

Support Vector Regression有三种不同的实现: :class:`SVR`, :class:`NuSVR` 和 :class:`LinearSVR`。 
:class:`LinearSVR` 提供了比 :class:`SVR` 更快的实现但是只能使用 线性核 , 而 :class:`NuSVR` 则实现了一个与 
:class:`SVR` 和 :class:`LinearSVR` 的数学形式稍微不一样的版本， 关于数学形式，请参考 :ref:`svm_implementation_details` 。

与支持向量分类器一样, 在回归问题中，``fit`` 方法也接受向量 X, y 作为参数, 只是这时候 y 是浮点数而不是整数值 ::

    >>> from sklearn import svm
    >>> X = [[0, 0], [2, 2]]
    >>> y = [0.5, 2.5]
    >>> clf = svm.SVR()
    >>> clf.fit(X, y) # doctest: +NORMALIZE_WHITESPACE
    SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
        gamma='auto_deprecated', kernel='rbf', max_iter=-1, shrinking=True,
        tol=0.001, verbose=False)
    >>> clf.predict([[1, 1]])
    array([1.5])


.. topic:: 案列:

 * :ref:`sphx_glr_auto_examples_svm_plot_svm_regression.py`



.. _svm_outlier_detection:

密度估计, 奇异值检测
=======================================

:class:`OneClassSVM` 类实现了一个 One-Class SVM ，它被用来进行离群点检测(outlier detection)。 

请参考 :ref:`outlier_detection` 获得 OneClassSVM 的详细用法。

复杂度
==========

支持向量机是一种强大的工具，但是它们的计算和存储需求随着训练向量的增加而迅速增加。
支持向量机的核心是二次规划问题（quadratic programming problem (QP)）：从训练数据点中剥离出支撑向量(support vectors)。
基于 `libsvm`_ 实现的 QP 求解器 在
:math:`O(n_{features} \times n_{samples}^2)` 和
:math:`O(n_{features} \times n_{samples}^3)` 之间变动，这依赖于在实际计算中如何高效利用 `libsvm`_ 缓存(这是与数据集无关的)。
如果数据非常稀疏， :math:`n_{features}` 应该用样本向量中非零特征的平均数量替换。

值得注意的是，在线性情况下，:class:`LinearSVC` 使用的算法是由 `liblinear`_ 实现的，它比基于`libsvm`_实现的对应的线性 :class:`SVC` 效率更高，而且
其伸缩性在百万级样本和特征下几乎是线性的。


实用小建议
=====================


  * **Avoiding data copy**: For :class:`SVC`, :class:`SVR`, :class:`NuSVC` and
    :class:`NuSVR`, if the data passed to certain methods is not C-ordered
    contiguous, and double precision, it will be copied before calling the
    underlying C implementation. You can check whether a given numpy array is
    C-contiguous by inspecting its ``flags`` attribute.

    For :class:`LinearSVC` (and :class:`LogisticRegression
    <sklearn.linear_model.LogisticRegression>`) any input passed as a numpy
    array will be copied and converted to the liblinear internal sparse data
    representation (double precision floats and int32 indices of non-zero
    components). If you want to fit a large-scale linear classifier without
    copying a dense numpy C-contiguous double precision array as input we
    suggest to use the :class:`SGDClassifier
    <sklearn.linear_model.SGDClassifier>` class instead.  The objective
    function can be configured to be almost the same as the :class:`LinearSVC`
    model.

  * **Kernel cache size**: For :class:`SVC`, :class:`SVR`, :class:`NuSVC` and
    :class:`NuSVR`, the size of the kernel cache has a strong impact on run
    times for larger problems.  If you have enough RAM available, it is
    recommended to set ``cache_size`` to a higher value than the default of
    200(MB), such as 500(MB) or 1000(MB).

  * **Setting C**: ``C`` is ``1`` by default and it's a reasonable default
    choice.  If you have a lot of noisy observations you should decrease it.
    It corresponds to regularize more the estimation.
    
    :class:`LinearSVC` and :class`LinearSVR` are less sensitive to ``C`` when
    it becomes large, and prediction results stop improving after a certain 
    threshold. Meanwhile, larger ``C`` values will take more time to train, 
    sometimes up to 10 times longer, as shown by Fan et al. (2008)

  * Support Vector Machine algorithms are not scale invariant, so **it
    is highly recommended to scale your data**. For example, scale each
    attribute on the input vector X to [0,1] or [-1,+1], or standardize it
    to have mean 0 and variance 1. Note that the *same* scaling must be
    applied to the test vector to obtain meaningful results. See section
    :ref:`preprocessing` for more details on scaling and normalization.

  * Parameter ``nu`` in :class:`NuSVC`/:class:`OneClassSVM`/:class:`NuSVR`
    approximates the fraction of training errors and support vectors.

  * In :class:`SVC`, if data for classification are unbalanced (e.g. many
    positive and few negative), set ``class_weight='balanced'`` and/or try
    different penalty parameters ``C``.

  * **Randomness of the underlying implementations**: The underlying 
    implementations of :class:`SVC` and :class:`NuSVC` use a random number
    generator only to shuffle the data for probability estimation (when
    ``probability`` is set to ``True``). This randomness can be controlled
    with the ``random_state`` parameter. If ``probability`` is set to ``False``
    these estimators are not random and ``random_state`` has no effect on the
    results. The underlying :class:`OneClassSVM` implementation is similar to
    the ones of :class:`SVC` and :class:`NuSVC`. As no probability estimation
    is provided for :class:`OneClassSVM`, it is not random.

    The underlying :class:`LinearSVC` implementation uses a random number
    generator to select features when fitting the model with a dual coordinate
    descent (i.e when ``dual`` is set to ``True``). It is thus not uncommon,
    to have slightly different results for the same input data. If that
    happens, try with a smaller tol parameter. This randomness can also be
    controlled with the ``random_state`` parameter. When ``dual`` is
    set to ``False`` the underlying implementation of :class:`LinearSVC` is
    not random and ``random_state`` has no effect on the results.

  * Using L1 penalization as provided by ``LinearSVC(loss='l2', penalty='l1',
    dual=False)`` yields a sparse solution, i.e. only a subset of feature
    weights is different from zero and contribute to the decision function.
    Increasing ``C`` yields a more complex model (more feature are selected).
    The ``C`` value that yields a "null" model (all weights equal to zero) can
    be calculated using :func:`l1_min_c`.


.. topic:: References:

 * Fan, Rong-En, et al.,
   `"LIBLINEAR: A library for large linear classification."
   <https://www.csie.ntu.edu.tw/~cjlin/papers/liblinear.pdf>`_,
   Journal of machine learning research 9.Aug (2008): 1871-1874.

.. _svm_kernels:

核函数
================

The *kernel function* can be any of the following:

  * linear: :math:`\langle x, x'\rangle`.

  * polynomial: :math:`(\gamma \langle x, x'\rangle + r)^d`.
    :math:`d` is specified by keyword ``degree``, :math:`r` by ``coef0``.

  * rbf: :math:`\exp(-\gamma \|x-x'\|^2)`. :math:`\gamma` is
    specified by keyword ``gamma``, must be greater than 0.

  * sigmoid (:math:`\tanh(\gamma \langle x,x'\rangle + r)`),
    where :math:`r` is specified by ``coef0``.

Different kernels are specified by keyword kernel at initialization::

    >>> linear_svc = svm.SVC(kernel='linear')
    >>> linear_svc.kernel
    'linear'
    >>> rbf_svc = svm.SVC(kernel='rbf')
    >>> rbf_svc.kernel
    'rbf'


自定义核函数
--------------

You can define your own kernels by either giving the kernel as a
python function or by precomputing the Gram matrix.

Classifiers with custom kernels behave the same way as any other
classifiers, except that:

    * Field ``support_vectors_`` is now empty, only indices of support
      vectors are stored in ``support_``

    * A reference (and not a copy) of the first argument in the ``fit()``
      method is stored for future reference. If that array changes between the
      use of ``fit()`` and ``predict()`` you will have unexpected results.


使用Python函数作为核
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also use your own defined kernels by passing a function to the
keyword ``kernel`` in the constructor.

Your kernel must take as arguments two matrices of shape
``(n_samples_1, n_features)``, ``(n_samples_2, n_features)``
and return a kernel matrix of shape ``(n_samples_1, n_samples_2)``.

The following code defines a linear kernel and creates a classifier
instance that will use that kernel::

    >>> import numpy as np
    >>> from sklearn import svm
    >>> def my_kernel(X, Y):
    ...     return np.dot(X, Y.T)
    ...
    >>> clf = svm.SVC(kernel=my_kernel)

.. topic:: Examples:

 * :ref:`sphx_glr_auto_examples_svm_plot_custom_kernel.py`.

使用 Gram matrix
~~~~~~~~~~~~~~~~~~~~~

Set ``kernel='precomputed'`` and pass the Gram matrix instead of X in the fit
method. At the moment, the kernel values between *all* training vectors and the
test vectors must be provided.

    >>> import numpy as np
    >>> from sklearn import svm
    >>> X = np.array([[0, 0], [1, 1]])
    >>> y = [0, 1]
    >>> clf = svm.SVC(kernel='precomputed')
    >>> # linear kernel computation
    >>> gram = np.dot(X, X.T)
    >>> clf.fit(gram, y) # doctest: +NORMALIZE_WHITESPACE
    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
        kernel='precomputed', max_iter=-1, probability=False,
        random_state=None, shrinking=True, tol=0.001, verbose=False)
    >>> # predict on training examples
    >>> clf.predict(gram)
    array([0, 1])

RBF核函数的参数
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When training an SVM with the *Radial Basis Function* (RBF) kernel, two
parameters must be considered: ``C`` and ``gamma``.  The parameter ``C``,
common to all SVM kernels, trades off misclassification of training examples
against simplicity of the decision surface. A low ``C`` makes the decision
surface smooth, while a high ``C`` aims at classifying all training examples
correctly.  ``gamma`` defines how much influence a single training example has.
The larger ``gamma`` is, the closer other examples must be to be affected.

Proper choice of ``C`` and ``gamma`` is critical to the SVM's performance.  One
is advised to use :class:`sklearn.model_selection.GridSearchCV` with 
``C`` and ``gamma`` spaced exponentially far apart to choose good values.

.. topic:: Examples:

 * :ref:`sphx_glr_auto_examples_svm_plot_rbf_parameters.py`

.. _svm_mathematical_formulation:

数学表达形式
========================

A support vector machine constructs a hyper-plane or set of hyper-planes
in a high or infinite dimensional space, which can be used for
classification, regression or other tasks. Intuitively, a good
separation is achieved by the hyper-plane that has the largest distance
to the nearest training data points of any class (so-called functional
margin), since in general the larger the margin the lower the
generalization error of the classifier.


.. figure:: ../auto_examples/svm/images/sphx_glr_plot_separating_hyperplane_001.png
   :align: center
   :scale: 75

SVC
---

Given training vectors :math:`x_i \in \mathbb{R}^p`, i=1,..., n, in two classes, and a
vector :math:`y \in \{1, -1\}^n`, SVC solves the following primal problem:


.. math::

    \min_ {w, b, \zeta} \frac{1}{2} w^T w + C \sum_{i=1}^{n} \zeta_i



    \textrm {subject to } & y_i (w^T \phi (x_i) + b) \geq 1 - \zeta_i,\\
    & \zeta_i \geq 0, i=1, ..., n

Its dual is

.. math::

   \min_{\alpha} \frac{1}{2} \alpha^T Q \alpha - e^T \alpha


   \textrm {subject to } & y^T \alpha = 0\\
   & 0 \leq \alpha_i \leq C, i=1, ..., n

where :math:`e` is the vector of all ones, :math:`C > 0` is the upper bound,
:math:`Q` is an :math:`n` by :math:`n` positive semidefinite matrix,
:math:`Q_{ij} \equiv y_i y_j K(x_i, x_j)`, where :math:`K(x_i, x_j) = \phi (x_i)^T \phi (x_j)`
is the kernel. Here training vectors are implicitly mapped into a higher
(maybe infinite) dimensional space by the function :math:`\phi`.


The decision function is:

.. math:: \operatorname{sgn}(\sum_{i=1}^n y_i \alpha_i K(x_i, x) + \rho)

.. note::

    While SVM models derived from `libsvm`_ and `liblinear`_ use ``C`` as
    regularization parameter, most other estimators use ``alpha``. The exact
    equivalence between the amount of regularization of two models depends on
    the exact objective function optimized by the model. For example, when the
    estimator used is :class:`sklearn.linear_model.Ridge <ridge>` regression,
    the relation between them is given as :math:`C = \frac{1}{alpha}`.

.. TODO multiclass case ?/

This parameters can be accessed through the members ``dual_coef_``
which holds the product :math:`y_i \alpha_i`, ``support_vectors_`` which
holds the support vectors, and ``intercept_`` which holds the independent
term :math:`\rho` :

.. topic:: References:

 * `"Automatic Capacity Tuning of Very Large VC-dimension Classifiers"
   <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.17.7215>`_,
   I. Guyon, B. Boser, V. Vapnik - Advances in neural information
   processing 1993.


 * `"Support-vector networks"
   <https://link.springer.com/article/10.1007%2FBF00994018>`_,
   C. Cortes, V. Vapnik - Machine Learning, 20, 273-297 (1995).



NuSVC
-----

We introduce a new parameter :math:`\nu` which controls the number of
support vectors and training errors. The parameter :math:`\nu \in (0,
1]` is an upper bound on the fraction of training errors and a lower
bound of the fraction of support vectors.

It can be shown that the :math:`\nu`-SVC formulation is a reparameterization
of the :math:`C`-SVC and therefore mathematically equivalent.


SVR
---

Given training vectors :math:`x_i \in \mathbb{R}^p`, i=1,..., n, and a
vector :math:`y \in \mathbb{R}^n` :math:`\varepsilon`-SVR solves the following primal problem:


.. math::

    \min_ {w, b, \zeta, \zeta^*} \frac{1}{2} w^T w + C \sum_{i=1}^{n} (\zeta_i + \zeta_i^*)



    \textrm {subject to } & y_i - w^T \phi (x_i) - b \leq \varepsilon + \zeta_i,\\
                          & w^T \phi (x_i) + b - y_i \leq \varepsilon + \zeta_i^*,\\
                          & \zeta_i, \zeta_i^* \geq 0, i=1, ..., n

Its dual is

.. math::

   \min_{\alpha, \alpha^*} \frac{1}{2} (\alpha - \alpha^*)^T Q (\alpha - \alpha^*) + \varepsilon e^T (\alpha + \alpha^*) - y^T (\alpha - \alpha^*)


   \textrm {subject to } & e^T (\alpha - \alpha^*) = 0\\
   & 0 \leq \alpha_i, \alpha_i^* \leq C, i=1, ..., n

where :math:`e` is the vector of all ones, :math:`C > 0` is the upper bound,
:math:`Q` is an :math:`n` by :math:`n` positive semidefinite matrix,
:math:`Q_{ij} \equiv K(x_i, x_j) = \phi (x_i)^T \phi (x_j)`
is the kernel. Here training vectors are implicitly mapped into a higher
(maybe infinite) dimensional space by the function :math:`\phi`.

The decision function is:

.. math:: \sum_{i=1}^n (\alpha_i - \alpha_i^*) K(x_i, x) + \rho

These parameters can be accessed through the members ``dual_coef_``
which holds the difference :math:`\alpha_i - \alpha_i^*`, ``support_vectors_`` which
holds the support vectors, and ``intercept_`` which holds the independent
term :math:`\rho`

.. topic:: References:

 * `"A Tutorial on Support Vector Regression"
   <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.114.4288>`_,
   Alex J. Smola, Bernhard Schölkopf - Statistics and Computing archive
   Volume 14 Issue 3, August 2004, p. 199-222. 


.. _svm_implementation_details:

算法实现细节
======================

Internally, we use `libsvm`_ and `liblinear`_ to handle all
computations. These libraries are wrapped using C and Cython.

.. _`libsvm`: https://www.csie.ntu.edu.tw/~cjlin/libsvm/
.. _`liblinear`: https://www.csie.ntu.edu.tw/~cjlin/liblinear/

.. topic:: References:

  For a description of the implementation and details of the algorithms
  used, please refer to

    - `LIBSVM: A Library for Support Vector Machines
      <https://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf>`_.

    - `LIBLINEAR -- A Library for Large Linear Classification
      <https://www.csie.ntu.edu.tw/~cjlin/liblinear/>`_.


