.. _introduction:

使用 scikit-learn 介绍机器学习
=====================================================

.. topic:: 内容提要

    在本节中，我们介绍一些在使用 scikit-learn 过程中用到的 `机器学习 <https://en.wikipedia.org/wiki/Machine_learning>`_ 词汇，
    并且给出一些例子阐释它们。

    译者注：这里有一节视频是关于机器学习的介绍，回答了机器学习是什么，机器学习的主要方法论，机器学习的应用领域等等问题，
    是译者自己录制的，希望大家喜欢。视频地址: 
    (`机器学习绪论 <https://v.youku.com/v_show/id_XMjY2MjU1MzczNg==.html?spm=a2h1n.8251843.playList.5~5~A&f=49255928&o=1>`_)


机器学习：问题设置
-------------------------------------

通常，我们的学习问题(learning problem)要考虑一个包含n个样本
(`samples <https://en.wikipedia.org/wiki/Sample_(statistics)>`_)
的数据集合，然后尝试预测未知数据的某个或某些属性。
如果每个样本不止一个数字，则称其具有若干属性(**attributes**)或特征(**features**)。例如，一个多维条目,
又称多变量数据 (`multivariate data <https://en.wikipedia.org/wiki/Multivariate_random_variable>`_)。

学习问题(learning problem)可以被划分为几个大类:

 * 监督学习 (`supervised learning <https://en.wikipedia.org/wiki/Supervised_learning>`_),
   其中数据带有我们想要预测的附加属性
   (:ref:`单击这里 <supervised-learning>` 跳转到 scikit-learn 的监督学习页面). 
   监督学习又可以分为两类:

    * 分类 (`Classification <https://en.wikipedia.org/wiki/Classification_in_machine_learning>`_):
      样本属于两个或多个类，我们希望从已经标记的数据中学习如何预测未标记的数据所属的类。
      分类问题的一个例子就是手写数字识别，其目的是将每个输入向量分配给有限数量的离散类别之一。
      另一种考虑分类的方法是将其作为一种离散的（与连续的相对应）监督学习形式，其中，我们有有限数量的类别，
      对于所提供的n个样本中的每一个，我们试图用正确的类别来标记样本。

    * 回归 (`Regression <https://en.wikipedia.org/wiki/Regression_analysis>`_):
      如果需要的输出由一个或多个连续变量构成，那么这样的任务就被称为 *regression*。
      回归问题的一个例子是预测鲑鱼的长度与其年龄和体重的关系。

 * 无监督学习 (`unsupervised learning <https://en.wikipedia.org/wiki/Unsupervised_learning>`_),
   其中，训练数据由一组输入向量 x 组成，没有任何对应的目标值。此类问题的目标可能是在数据中发现一组相似的样本，
   被称为聚类(`clustering <https://en.wikipedia.org/wiki/Cluster_analysis>`_),
   或者是决定数据在输入空间中的分布, 被称为密度估计(`density estimation <https://en.wikipedia.org/wiki/Density_estimation>`_),
   或者是将高维数据投影到低维空间进行降维以达到可视化的效果*visualization*
   (:ref:`单击这里 <unsupervised-learning>`  跳转到 scikit-learn 的无监督学习页面).

.. topic:: 训练集和测试集

    机器学习就是学习数据集的一些属性，然后根据另一个数据集测试这些属性。机器学习中的一个常见实践是
    通过将数据集分成两个来评估算法。 我们称其中一个集合为**训练集**，我们在其上学习一些属性；
    我们称另一个集合为**测试集**，我们在其上测试所学习的属性。


.. _loading_example_dataset:

加载示例数据集
--------------------------

`scikit-learn` 附带了几个标准数据集，例如用于分类的虹膜(`iris <https://en.wikipedia.org/wiki/Iris_flower_data_set>`_)和
数字(`digits <http://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits>`_)数据集以及
用于回归的波士顿房价(`boston house prices dataset <https://archive.ics.uci.edu/ml/machine-learning-databases/housing/>`_)数据集。

下面，我们从shell启动一个Python解释器，然后加载虹膜 ``iris`` 和数字 ``digits`` 数据集。
我们的符号约定是 ``$`` 表示shell提示，而 ``>>>`` 表示Python解释器的提示::

  $ python
  >>> from sklearn import datasets
  >>> iris = datasets.load_iris()
  >>> digits = datasets.load_digits()

数据集是类似于字典的对象，它保存所有数据和关于数据的一些元数据。数据部分被保存在 ``.data`` 成员中,
它是一个 ``n_samples, n_features`` 的数组。 在监督学习问题中，一个或多个响应变量(response variables)被保存在
``.target`` 成员中. 更多关于数据集的详细信息请参考 :ref:`dedicated section <datasets>`.

比如, 在手写字符数据集中, ``digits.data`` 可以让我们访问特征然后用于分类识别不同的字符样本::

  >>> print(digits.data)  # doctest: +NORMALIZE_WHITESPACE
  [[ 0.   0.   5. ...   0.   0.   0.]
   [ 0.   0.   0. ...  10.   0.   0.]
   [ 0.   0.   0. ...  16.   9.   0.]
   ...
   [ 0.   0.   1. ...   6.   0.   0.]
   [ 0.   0.   2. ...  12.   0.   0.]
   [ 0.   0.  10. ...  12.   1.   0.]]

而 ``digits.target`` 则给出了每一个手写字符样本的真实类别标签, 而这正是我们想要让学习器学习的类标签::

  >>> digits.target
  array([0, 1, 2, ..., 8, 9, 8])

.. topic:: data数组的形状(Shape of the data arrays)

    尽管原始数据可能有各种不同的shape,但是一旦load完毕，data成员 总是一个2D数组, shape 为 ``(n_samples, n_features)`` 。
    在 digits 这个数据集中，每个原始样本是一个shape为 ``(8, 8)`` 的图像，而且可以用以下方式访问::

      >>> digits.images[0]  # doctest: +NORMALIZE_WHITESPACE
      array([[  0.,   0.,   5.,  13.,   9.,   1.,   0.,   0.],
             [  0.,   0.,  13.,  15.,  10.,  15.,   5.,   0.],
             [  0.,   3.,  15.,   2.,   0.,  11.,   8.,   0.],
             [  0.,   4.,  12.,   0.,   0.,   8.,   8.,   0.],
             [  0.,   5.,   8.,   0.,   0.,   9.,   8.,   0.],
             [  0.,   4.,  11.,   0.,   1.,  12.,   7.,   0.],
             [  0.,   2.,  14.,   5.,  10.,  12.,   0.,   0.],
             [  0.,   0.,   6.,  13.,  10.,   0.,   0.,   0.]])

    案列链接 :ref:`simple example on this dataset <sphx_glr_auto_examples_classification_plot_digits_classification.py>`
    举例说明了如何从原始问题出发，形成用于scikit-learning中的数据。

.. topic:: 从外部数据加载

    要想从外部数据集加载数据，请看 :ref:`loading external datasets <external_datasets>`.

学习和预测
------------------------

对于 ``digits`` 数据集，任务是根据图像预测它代表哪个数字。我们给出10个可能的类（从0到9的数字）中的每个类的样本，
我们在这些类上拟合(*fit*)一个估计器(`estimator <https://en.wikipedia.org/wiki/Estimator>`_)，
以便能够预测(*predict*)未知样本所属的类。

在 scikit-learn 中, 一个用于分类的估计器(estimator)是一个 Python 对象，该对象实现了成员方法 ``fit(X, y)`` 和 ``predict(T)``.

分类估计器的一个例子是类 ``sklearn.svm.SVC``, 它实现了支持向量分类器
(`support vector classification <https://en.wikipedia.org/wiki/Support_vector_machine>`_)。
该 estimator 的构造函数(constructor)接受模型参数(model's parameters)作为构造函数的输入参数(arguments)。

现在, 我们实例化SVC的estimator,把它看成个黑盒子::

  >>> from sklearn import svm
  >>> clf = svm.SVC(gamma=0.001, C=100.)

.. topic:: 选择合适的模型参数(model's parameters)

  在本例中，我们手动设置了模型参数 ``gamma`` 的值。如果你想找到一个更合理的模型参数的话，请使用sklearn提供的工具，比如
  :ref:`网格搜索 <grid_search>` 和 :ref:`交叉验证 <cross_validation>`.

上面创建的估计器实例 ``clf`` 首先适合于模型；也就是说，它必须从模型中学习。这是通过把我们的训练集传递给拟合方法 ``fit`` 完成的。
对于训练集，我们将使用数据集中的所有图像，除了最后一个图像，我们将保留这个图像用于预测。我们使用Python语法 ``[:-1]`` 来获取训练集，
该切片操作将返回一个新的array,它包含了来自于 ``digits.data`` 的所有样本除了最后一个 ::

  >>> clf.fit(digits.data[:-1], digits.target[:-1])  # doctest: +NORMALIZE_WHITESPACE
  SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

现在我们可以预测(*predict*)新的值了。 在这个案例中，我们预测一下 ``digits.data`` 中的最后一张图像的类标签。
By predicting, you'll determine the image from the training set that best matches the last image.


  >>> clf.predict(digits.data[-1:])
  array([8])

对应的图像是:

.. image:: /auto_examples/datasets/images/sphx_glr_plot_digits_last_image_001.png
    :target: ../../auto_examples/datasets/plot_digits_last_image.html
    :align: center
    :scale: 50

正如您所看到的，这是一个具有挑战性的任务：毕竟，图像的分辨率很差。你同意分类器的结果吗？

一个分类问题的完整案列:
:ref:`sphx_glr_auto_examples_classification_plot_digits_classification.py`.


模型持久化
-----------------

通过使用Python的内置持久性模块 `pickle <https://docs.python.org/2/library/pickle.html>`_, 可以在scikit-learning中保存模型 ::

  >>> from sklearn import svm
  >>> from sklearn import datasets
  >>> clf = svm.SVC(gamma='scale')
  >>> iris = datasets.load_iris()
  >>> X, y = iris.data, iris.target
  >>> clf.fit(X, y)  # doctest: +NORMALIZE_WHITESPACE
  SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

  >>> import pickle
  >>> s = pickle.dumps(clf)
  >>> clf2 = pickle.loads(s)
  >>> clf2.predict(X[0:1])
  array([0])
  >>> y[0]
  0

对于某些具体情况，使用joblib替代pickle(``joblib.dump`` & ``joblib.load``)可能更有趣，
这对于大数据更有效，但是它只能对磁盘进行pickle而不能对字符串进行pickle::

  >>> from joblib import dump, load
  >>> dump(clf, 'filename.joblib') # doctest: +SKIP

稍后，您可以重新加载pickle模型（可能在另一个Python进程中）::

  >>> clf = load('filename.joblib') # doctest:+SKIP

.. note::

    ``joblib.dump`` and ``joblib.load`` functions also accept file-like object
    instead of filenames. More information on data persistence with Joblib is
    available `here <https://joblib.readthedocs.io/en/latest/persistence.html>`_.

Note that pickle has some security and maintainability issues. Please refer to
section :ref:`model_persistence` for more detailed information about model
persistence with scikit-learn.


规定或约定(Conventions)
-----------

scikit-learning估计器遵循某些规则以使其行为更具预测性。我们可以在这个章节里面看到scikit-learn对机器学习术语的定义 :ref:`glossary`。

类型转换
~~~~~~~~~~~~

除非特别指出，输入将会被转成 ``float64``::

  >>> import numpy as np
  >>> from sklearn import random_projection

  >>> rng = np.random.RandomState(0)
  >>> X = rng.rand(10, 2000)
  >>> X = np.array(X, dtype='float32')
  >>> X.dtype
  dtype('float32')

  >>> transformer = random_projection.GaussianRandomProjection()
  >>> X_new = transformer.fit_transform(X)
  >>> X_new.dtype
  dtype('float64')

在这个例子中, ``X`` 是 ``float32``, 但是被函数 ``fit_transform(X)``转换成 ``float64``.

回归目标值转换为 ``float64`` 以及  分类器的目标值保持不变::

    >>> from sklearn import datasets
    >>> from sklearn.svm import SVC
    >>> iris = datasets.load_iris()
    >>> clf = SVC(gamma='scale')
    >>> clf.fit(iris.data, iris.target)  # doctest: +NORMALIZE_WHITESPACE
    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)

    >>> list(clf.predict(iris.data[:3]))
    [0, 0, 0]

    >>> clf.fit(iris.data, iris.target_names[iris.target])  # doctest: +NORMALIZE_WHITESPACE
    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)

    >>> list(clf.predict(iris.data[:3]))  # doctest: +NORMALIZE_WHITESPACE
    ['setosa', 'setosa', 'setosa']

这里, 第一个 ``predict()`` 返回一个 integer array, 因为 ``iris.target``
(an integer array) 被用在 ``fit``. 第二个 ``predict()`` 返回一个 string
array, since ``iris.target_names`` was for fitting.

再次训练和更新参数
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

当 Estimator 构造好后，它的 Hyper-parameters 还可以用 :term:`set_params()<set_params>` 方法更新。
不止一次的调用 ``fit()`` 将会重新覆盖掉之前学习到的模型 ::

  >>> import numpy as np
  >>> from sklearn.svm import SVC

  >>> rng = np.random.RandomState(0)
  >>> X = rng.rand(100, 10)
  >>> y = rng.binomial(1, 0.5, 100)
  >>> X_test = rng.rand(5, 10)

  >>> clf = SVC()
  >>> clf.set_params(kernel='linear').fit(X, y)  # doctest: +NORMALIZE_WHITESPACE
  SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
    kernel='linear', max_iter=-1, probability=False, random_state=None,
    shrinking=True, tol=0.001, verbose=False)
  >>> clf.predict(X_test)
  array([1, 0, 1, 1, 0])

  >>> clf.set_params(kernel='rbf', gamma='scale').fit(X, y)  # doctest: +NORMALIZE_WHITESPACE
  SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
  >>> clf.predict(X_test)
  array([1, 0, 1, 1, 0])

上面的代码中, 在Estimator创建好以后，默认核函数 ``rbf`` 第一次通过 :func:`SVC.set_params()<sklearn.svm.SVC.set_params>` 被修改成 ``linear``,
然后又改回了默认的 ``rbf`` 进行再次重新拟合 estimator， 然后做第二次预测.

多分类拟合 vs. 多标签拟合
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

当使用多类分类器 :class:`multiclass classifiers <sklearn.multiclass>` 时, 所执行的学习和预测任务取决于符合以下条件的目标数据的格式::

    >>> from sklearn.svm import SVC
    >>> from sklearn.multiclass import OneVsRestClassifier
    >>> from sklearn.preprocessing import LabelBinarizer

    >>> X = [[1, 2], [2, 4], [4, 5], [3, 2], [3, 1]]
    >>> y = [0, 0, 1, 1, 2]

    >>> classif = OneVsRestClassifier(estimator=SVC(gamma='scale',
    ...                                             random_state=0))
    >>> classif.fit(X, y).predict(X)
    array([0, 0, 1, 1, 2])

In the above case, the classifier is fit on a 1d array of multiclass labels and
the ``predict()`` method therefore provides corresponding multiclass predictions.
It is also possible to fit upon a 2d array of binary label indicators::

    >>> y = LabelBinarizer().fit_transform(y)
    >>> classif.fit(X, y).predict(X)
    array([[1, 0, 0],
           [1, 0, 0],
           [0, 1, 0],
           [0, 0, 0],
           [0, 0, 0]])

Here, the classifier is ``fit()``  on a 2d binary label representation of ``y``,
using the :class:`LabelBinarizer <sklearn.preprocessing.LabelBinarizer>`.
In this case ``predict()`` returns a 2d array representing the corresponding
multilabel predictions.

Note that the fourth and fifth instances returned all zeroes, indicating that
they matched none of the three labels ``fit`` upon. With multilabel outputs, it
is similarly possible for an instance to be assigned multiple labels::

  >>> from sklearn.preprocessing import MultiLabelBinarizer
  >>> y = [[0, 1], [0, 2], [1, 3], [0, 2, 3], [2, 4]]
  >>> y = MultiLabelBinarizer().fit_transform(y)
  >>> classif.fit(X, y).predict(X)
  array([[1, 1, 0, 0, 0],
         [1, 0, 1, 0, 0],
         [0, 1, 0, 1, 0],
         [1, 0, 1, 0, 0],
         [1, 0, 1, 0, 0]])

In this case, the classifier is fit upon instances each assigned multiple labels.
The :class:`MultiLabelBinarizer <sklearn.preprocessing.MultiLabelBinarizer>` is
used to binarize the 2d array of multilabels to ``fit`` upon. As a result,
``predict()`` returns a 2d array with multiple predicted labels for each instance.
