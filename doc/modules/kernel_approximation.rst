.. _kernel_approximation:

核逼近(Kernel Approximation)
====================

该子模块包含一些函数，这些函数逼近(approximate)某些内核对应的特征映射，例如在支持向量机中使用(请看 :ref:`svm`)。
下列特征函数(feature functions)对输入进行非线性变换，可作为线性分类或其他算法的基(basis)。

.. currentmodule:: sklearn.linear_model

与隐式使用特征映射的内核技巧(`kernel trick <https://en.wikipedia.org/wiki/Kernel_trick>`_)相比，
使用近似显式特征映射的优点是 显式映射(explicit mappings)更适合在线学习，并且可以显著降低大数据集上的学习成本。
标准的核化支持向量机(Standard kernelized SVMs)不能很好地扩展到大型数据集，但是使用近似的内核映射(approximate kernel map)
可以使用效率更高的线性支持向量机(linear SVMs)。特别是，核映射逼近与 :class:`SGDClassifier` 的结合可以使大型数据集上的非线性学习成为可能。
由于使用近似嵌入(approximate embeddings)的经验工作不多，所以最好在可能的情况下将结果与精确的核方法(exact kernel methods)进行比较。

.. seealso::

   :ref:`polynomial_regression` 进行精确的多项式变换。

.. currentmodule:: sklearn.kernel_approximation

.. _nystroem_kernel_approx:

核逼近的Nystroem方法
----------------------------------------
在 :class:`Nystroem` 类中实现的Nystroem方法是求解 核的低秩近似(low-rank approximations of kernels) 的一种通用方法。
它实际上是 通过对评估内核的数据进行二次采样(subsampling) 来实现这一点的。默认情况下，:class:`Nystroem` 使用 ``rbf`` 核，
但它可以使用任何核函数(kernel function)或预先计算的核矩阵(kernel matrix)。
使用的样本数，也就是所计算的特征的维数由参数 ``n_components`` 给出。

.. _rbf_kernel_approx:

径向基函数核
----------------------------

:class:`RBFSampler` 类构造了径向基函数核(radial basis function kernel)的近似映射,也被称为 *Random Kitchen Sinks* [RR2007]_。
此转换可用于显式地建模内核映射，在应用线性算法之前，比如 linear SVM :: 

    >>> from sklearn.kernel_approximation import RBFSampler
    >>> from sklearn.linear_model import SGDClassifier
    >>> X = [[0, 0], [1, 1], [1, 0], [0, 1]]
    >>> y = [0, 0, 1, 1]
    >>> rbf_feature = RBFSampler(gamma=1, random_state=1)
    >>> X_features = rbf_feature.fit_transform(X)
    >>> clf = SGDClassifier(max_iter=5)
    >>> clf.fit(X_features, y)   # doctest: +NORMALIZE_WHITESPACE
    SGDClassifier(alpha=0.0001, average=False, class_weight=None,
           early_stopping=False, epsilon=0.1, eta0=0.0, fit_intercept=True,
           l1_ratio=0.15, learning_rate='optimal', loss='hinge', max_iter=5,
           n_iter=None, n_iter_no_change=5, n_jobs=None, penalty='l2',
           power_t=0.5, random_state=None, shuffle=True, tol=None,
           validation_fraction=0.1, verbose=0, warm_start=False)
    >>> clf.score(X_features, y)
    1.0

该映射依赖于对核值的蒙特卡洛近似(Monte Carlo approximation)。 ``fit`` 函数执行蒙特卡洛采样，而 ``transform`` 方法执行数据映射。
由于过程本身的随机性，对 ``fit`` 函数的不同调用可能会导致不同的结果。

``fit`` 函数有两个参数：``n_components`` 是特征变换的目标维数； ``gamma`` 是RBF-kernel的参数。

更高的 ``n_components`` 会使核逼近更好，得到的结果与核支持向量机产生的结果更相似。 
请注意，“拟合”特征函数实际上并不取决于提供给 ``fit`` 函数的数据，只使用数据的维数。有关此方法的详细信息，请参阅 [RR2007]_.

对于 ``n_components`` 的给定值，:class:`RBFSampler` 通常不像 :class:`Nystroem` 那样精确。不过，:class:`RBFSampler` 
使用更大的特征空间更高效，计算成本更低。

.. figure:: ../auto_examples/images/sphx_glr_plot_kernel_approximation_002.png
    :target: ../auto_examples/plot_kernel_approximation.html
    :scale: 50%
    :align: center

    Comparing an exact RBF kernel (left) with the approximation (right)

.. topic:: 案例:

    * :ref:`sphx_glr_auto_examples_plot_kernel_approximation.py`

.. _additive_chi_kernel_approx:

加性 Chi Squared 核
---------------------------

The additive chi squared kernel 是一个在直方图上的内核, 经常在计算机视觉中使用。

The additive chi squared kernel 由下式给出：

.. math::

        k(x, y) = \sum_i \frac{2x_iy_i}{x_i+y_i}

这和 :func:`sklearn.metrics.additive_chi2_kernel` 函数不完全一样。
文献 [VZ2010]_ 的作者们比较喜欢上面这种形式，因为它总是正定的(positive definite)。
由于内核是加性的，因此可以单独处理所有分量 :math:`x_i` 以进行嵌入。
这样就可以在规则的区间内对傅里叶变换进行采样，而不是用蒙特卡罗抽样进行近似。

类 :class:`AdditiveChi2Sampler` 实现了这种按分量进行的确定性采样(component wise deterministic sampling)。
每个分量被采样 :math:`n` 次，每个输入维(两者的乘积来源于傅里叶变换的实部和复部) 产生 :math:`2n+1` 维 。
在文献中，:math:`n` 通常被选择为1或2，把数据集的size变为 ``n_samples * 5 * n_features`` (在 :math:`n=2` 的情况下)。

由 :class:`AdditiveChi2Sampler` 类提供的近似特征映射 可以与 :class:`RBFSampler` 类提供的近似特征映射 组合在一起
去产生一个用于 指数化chi squared kernel 的近似特征映射。
请看文献 [VZ2010]_ 获得更多详情 ，在 [VVZ2010]_ 中介绍了 :class:`AdditiveChi2Sampler` 与 :class:`RBFSampler` 的组合方法。

.. _skewed_chi_kernel_approx:

Skewed Chi Squared Kernel
-------------------------

The skewed chi squared kernel 由下式给出:

.. math::

        k(x,y) = \prod_i \frac{2\sqrt{x_i+c}\sqrt{y_i+c}}{x_i + y_i + 2c}


它具有与计算机视觉中常用的 exponentiated chi squared kernel 相似的性质，但允许特征映射的简单蒙特卡罗近似。

:class:`SkewedChi2Sampler` 类的用法与上面介绍的 :class:`RBFSampler` 类的用法是一样的。
唯一的区别在自由参数, 被称之为 :math:`c` 。 关于此映射的各种详细信息，请参考文献 [LS2010]_.


数学方面的细节
--------------------

核方法，如支持向量机或核化PCA，依赖于再生核Hilbert空间的性质。
对于任何正定核函数 :math:`k` (一个所谓的Mercer核)，保证存在一个映射 :math:`\phi` 到Hilbert空间 :math:`\mathcal{H}`，这样

.. math::

        k(x,y) = \langle \phi(x), \phi(y) \rangle

其中 :math:`\langle \cdot, \cdot \rangle` 表示Hilbert空间中的内积。

如果算法(如linear SVM或PCA)仅依赖于数据点 :math:`x_i` 的标量乘积，则可以使用 :math:`k(x_i, x_j)` 的值，
这相当于将该算法应用于映射得到的数据点 :math:`\phi(x_i)` 。 
使用 :math:`k` 的优点是，不必显式计算映射 :math:`\phi`，从而允许任意大的特征(甚至无限)。

内核方法的一个缺点是，在优化过程中可能需要存储许多内核值 :math:`k(x_i, x_j)` 。
如果将核化分类器应用于新的数据 :math:`y_j` ，则需要计算 :math:`k(x_i, y_j)` 来作出预测，
可能对训练集中的许多不同的 :math:`x_i`  进行预测。

该子模块中的类允许近似 嵌入 :math:`\phi` ，从而显式地使用表示形式 :math:`\phi(x_i)` ，
从而避免了应用内核或存储训练样例的需要。


.. topic:: 参考文献:

    .. [RR2007] `"Random features for large-scale kernel machines"
      <http://www.robots.ox.ac.uk/~vgg/rg/papers/randomfeatures.pdf>`_
      Rahimi, A. and Recht, B. - Advances in neural information processing 2007,
    .. [LS2010] `"Random Fourier approximations for skewed multiplicative histogram kernels"
      <http://www.maths.lth.se/matematiklth/personal/sminchis/papers/lis_dagm10.pdf>`_
      Random Fourier approximations for skewed multiplicative histogram kernels
      - Lecture Notes for Computer Sciencd (DAGM)
    .. [VZ2010] `"Efficient additive kernels via explicit feature maps"
      <https://www.robots.ox.ac.uk/~vgg/publications/2011/Vedaldi11/vedaldi11.pdf>`_
      Vedaldi, A. and Zisserman, A. - Computer Vision and Pattern Recognition 2010
    .. [VVZ2010] `"Generalized RBF feature maps for Efficient Detection"
      <https://www.robots.ox.ac.uk/~vgg/publications/2010/Sreekanth10/sreekanth10.pdf>`_
      Vempati, S. and Vedaldi, A. and Zisserman, A. and Jawahar, CV - 2010
