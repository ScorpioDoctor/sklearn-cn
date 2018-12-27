.. _metrics:

成对测度, 相似性 和 核
========================================

:mod:`sklearn.metrics.pairwise` 子模块实现了一系列工具用于评估成对距离(pairwise distances)或 样本集合之间的相似性(affinity of sets of samples)。

这个模块包含了距离测度(distance metrics)和核函数(kernel)。这里给出了它们的一个简要的总结。

距离测度是一种函数 ``d(a, b)``,  如果 ``a`` 和 ``b`` 的相似性比 ``a`` 和 ``c`` 的相似性高的话 ，那么 ``d(a, b) < d(a, c)`` ;
如果 ``a`` 和 ``b`` 完全相似的话， 那么 ``d(a, b) = 0`` 。
距离测度中最广为人知的就是 欧氏距离(Euclidean distance)。
要想成为一个真正的测度('true' metric), 它必须遵守以下四个条件 ::

    1. d(a, b) >= 0, 对所有的 a 和 b
    2. d(a, b) == 0, if and only if a = b, 正定性(positive definiteness)
    3. d(a, b) == d(b, a), 对称性(symmetry)
    4. d(a, c) <= d(a, b) + d(b, c), 三角不等式(the triangle inequality)

核(kernel)是相似性的度量, i.e.  如果 ``a`` 和 ``b`` 被认为比 ``a`` 和 ``c`` "更相似"，则 ``s(a, b) > s(a, c)`` 。
核(kernel)也必须是半正定的(positive semi-definite)。

有很多种方法可以在一个距离测度(distance metric)和相似性度量(similarity measure)之间转换，比如用 kenel 进行这种转换。
让 ``D`` 是某种距离, 并且 ``S`` 是一个 kernel:

    1. ``S = np.exp(-D * gamma)``, 其中用于选择 ``gamma`` 的一种启发式方法是 ``1 / num_features``
    2. ``S = 1. / (D / np.max(D))``


.. currentmodule:: sklearn.metrics

``X`` 的行向量和 ``Y`` 的行向量之间的距离可以用函数 :func:`pairwise_distances` 进行计算。
如果 ``Y`` 被忽略，则 ``X`` 的所有行向量的成对距离(pairwise distances)就会被计算。 
类似的，函数 :func:`pairwise.pairwise_kernels` 可以使用不同的核函数(kernel functions)来计算 `X` 和 `Y` 之间的 kernel。
请查看API获得更对详情。::

    >>> import numpy as np
    >>> from sklearn.metrics import pairwise_distances
    >>> from sklearn.metrics.pairwise import pairwise_kernels
    >>> X = np.array([[2, 3], [3, 5], [5, 8]])
    >>> Y = np.array([[1, 0], [2, 1]])
    >>> pairwise_distances(X, Y, metric='manhattan')
    array([[ 4.,  2.],
           [ 7.,  5.],
           [12., 10.]])
    >>> pairwise_distances(X, metric='manhattan')
    array([[0., 3., 8.],
           [3., 0., 5.],
           [8., 5., 0.]])
    >>> pairwise_kernels(X, Y, metric='linear')
    array([[ 2.,  7.],
           [ 3., 11.],
           [ 5., 18.]])


.. currentmodule:: sklearn.metrics.pairwise

.. _cosine_similarity:

余弦相似度
-----------------
函数 :func:`cosine_similarity` 计算向量之间的L2归范化的点积(L2-normalized dot product)。
那就是, 如果 :math:`x` 和 :math:`y` 是两个行向量,则它们的余弦相似度(cosine similarity) 
:math:`k` 定义如下:

.. math::

    k(x, y) = \frac{x y^\top}{\|x\| \|y\|}

之所以被称之为 余弦相似度, 是因为 Euclidean (L2) normalization 把两个向量投影到单位球
(unit sphere),这时它们的点积就是两个向量之间的夹角的余弦值。

kernel 是计算以 tf-idf 向量表示的文档相似度的流行选择。
函数 :func:`cosine_similarity` 接受 ``scipy.sparse`` 矩阵。
(注意 ``sklearn.feature_extraction.text`` 中的 tf-idf 系列函数 能够产生规范化的向量
(normalized vectors), 在这种情况下 函数 :func:`cosine_similarity` 等价于 函数 
:func:`linear_kernel`, 只是较慢一点儿。)

.. topic:: 参考文献:

    * C.D. Manning, P. Raghavan and H. Schütze (2008). Introduction to
      Information Retrieval. Cambridge University Press.
      http://nlp.stanford.edu/IR-book/html/htmledition/the-vector-space-model-for-scoring-1.html

.. _linear_kernel:

线性核
-------------
函数 :func:`linear_kernel` 计算线性核(linear kernel), 这就是说, 这是函数 :func:`polynomial_kernel` 的一种特殊情形，
其中参数取值为：``degree=1`` 并且 ``coef0=0`` (homogeneous)。
若 ``x`` 和 ``y`` 是列向量, 它们的线性核 计算如下:

.. math::

    k(x, y) = x^\top y

.. _polynomial_kernel:

多项式核
-----------------
函数 :func:`polynomial_kernel` 计算两个向量之间的d-阶多项式核(polynomial kernel)。
多项式核表达了两个向量之间的相似度。在概念上，多项式核不仅考虑了同一维下向量之间的相似性，而且考虑了各维之间的相似性。
当用于机器学习算法的时候, 这个特性允许我们把特征之间的相互作用也考虑进去。

多项式核的定义如下:

.. math::

    k(x, y) = (\gamma x^\top y +c_0)^d

其中:

    * ``x``, ``y`` 是输入向量
    * ``d`` 是 核的阶数(kernel degree)

如果 :math:`c_0 = 0` ， 则这个 kernel 就被说成是 同质的(homogeneous)。

.. _sigmoid_kernel:

Sigmoid核
--------------
函数 :func:`sigmoid_kernel` 计算两个向量之间的 sigmoid kernel 。
sigmoid kernel 也被称之为 双曲正切(hyperbolic tangent), 或 Multilayer Perceptron 
(因为在神经网络领域, 它经常被用于 神经元激活函数)。 它的定义如下:

.. math::

    k(x, y) = \tanh( \gamma x^\top y + c_0)

其中:

    * ``x``, ``y`` 时输入向量
    * :math:`\gamma` 是 斜率(slope)
    * :math:`c_0` 是 截距(intercept)

.. _rbf_kernel:

RBF 核
----------
函数 :func:`rbf_kernel` 计算两个向量之间的 径向基函数核(RBF kernel)。
它的定义如下: 

.. math::

    k(x, y) = \exp( -\gamma \| x-y \|^2)

其中 ``x`` 和 ``y`` 是输入向量。 如果 :math:`\gamma = \sigma^{-2}` ，
该 RBF kernel 被称为 方差的高斯核(the Gaussian kernel of variance :math:`\sigma^2`)。

.. _laplacian_kernel:

拉普拉斯核
----------------
函数 :func:`laplacian_kernel` 是 RBF kernel 的一个变体:

.. math::

    k(x, y) = \exp( -\gamma \| x-y \|_1)

其中 ``x`` 和 ``y`` 是输入向量 并且 :math:`\|x-y\|_1` 是输入向量之间的 曼哈顿距离(Manhattan distance) 。

它在ML中被证明是适用于无噪音数据的。
请看 e.g. `Machine learning for quantum mechanics in a nutshell
<http://onlinelibrary.wiley.com/doi/10.1002/qua.24954/abstract/>`_.

.. _chi2_kernel:

Chi-squared kernel
------------------
The chi-squared kernel 是一个在计算机视觉应用中广受欢迎的用于训练 non-linear SVMs 的核。
它使用函数 :func:`chi2_kernel` 进行计算，然后传递给类 :class:`sklearn.svm.SVC` ，并将参数设置为 ``kernel="precomputed"``::

    >>> from sklearn.svm import SVC
    >>> from sklearn.metrics.pairwise import chi2_kernel
    >>> X = [[0, 1], [1, 0], [.2, .8], [.7, .3]]
    >>> y = [0, 1, 0, 1]
    >>> K = chi2_kernel(X, gamma=.5)
    >>> K                        # doctest: +ELLIPSIS
    array([[1.        , 0.36787944, 0.89483932, 0.58364548],
           [0.36787944, 1.        , 0.51341712, 0.83822343],
           [0.89483932, 0.51341712, 1.        , 0.7768366 ],
           [0.58364548, 0.83822343, 0.7768366 , 1.        ]])

    >>> svm = SVC(kernel='precomputed').fit(K, y)
    >>> svm.predict(K)
    array([0, 1, 0, 1])

它也可以直接被用作 ``kernel`` 的参数 ::

    >>> svm = SVC(kernel=chi2_kernel).fit(X, y)
    >>> svm.predict(X)
    array([0, 1, 0, 1])


The chi squared kernel 由下式给出:

.. math::

        k(x, y) = \exp \left (-\gamma \sum_i \frac{(x[i] - y[i]) ^ 2}{x[i] + y[i]} \right )

数据被假定为非负的，并且通常被归一化为 1 的 L1范数 (L1-norm of one)。归一化是通过与 chi squared distance 的连接来合理化的，
chi squared distance 是 两个离散概率分布之间的距离。

chi squared kernel 最常用于 视觉词汇的直方图(词袋)(histograms (bags) of visual words)。

.. topic:: 参考文献:

    * Zhang, J. and Marszalek, M. and Lazebnik, S. and Schmid, C.
      Local features and kernels for classification of texture and object
      categories: A comprehensive study
      International Journal of Computer Vision 2007
      https://research.microsoft.com/en-us/um/people/manik/projects/trade-off/papers/ZhangIJCV06.pdf

