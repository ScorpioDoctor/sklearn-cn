

.. _gaussian_process:

==================
高斯过程(Gaussian Processes)
==================

.. currentmodule:: sklearn.gaussian_process

高斯过程(**Gaussian Processes (GP)**)是一种通用的监督学习方法，旨在解决*regression*和*probabilistic classification*问题。

高斯过程 的优点包括:

    - 预测值对观测值进行插值(至少对于regular kernels)。

    - 预测结果是概率形式的(Gaussian)。这样的话，可以计算得到经验置信区间(empirical confidence intervals )并且
      据此来判断是否需要修改（在线拟合，自适应拟合）一些感兴趣区域上的预测值。

    - 通用性(Versatile): 可以指定不同的 :ref:`kernels <gp_kernels>` 。 常见的 kernels 都可以使用，还可以自定义kernels.

高斯过程 的缺点包括：

    - 它们不是稀疏的，即它们使用整个样本/特征信息来执行预测。They are not sparse, i.e., they use the whole samples/features information to
      perform the prediction.

    - 高维空间模型会失效，高维也就是指特征的数量超过几十个(a few dozens)。


.. _gpr:

高斯过程回归器 (GPR)
=================================

.. currentmodule:: sklearn.gaussian_process

:class:`GaussianProcessRegressor` 类实现了用于回归问题的高斯过程(GP)模型。 
为此，需要指定GP的先验(prior)。先验均值假定为常数或者零(当参数 ``normalize_y=False`` 时); 
当 ``normalize_y=True`` 时，先验均值通常为训练数据的均值。而先验的方差通过传递 内核(:ref:`kernel <gp_kernels>`) 对象来指定。
通过由参数 ``optimizer`` 指定的优化器 最大化对数边缘似然估计(LML)，内核的超参数可以在 GaussianProcessRegressor 的拟合过程中被优化。
优化器第一次运行的时候总是从一组初始的内核超参数上启动；后续的优化过程从被允许的合理取值范围内随机选择一组超参数开始优化。
如果需要保持初始的超参值固定不变， 那么需要把优化器设置为 `None` 。

目标变量中的噪声级别通过参数 ``alpha`` 来传递并指定，要么全局都是一个常量要么是每个数据点对应一个噪声点。 
请注意，适度的噪声水平也可以有助于处理拟合期间的数值计算问题，
因为它被有效地实现为吉洪诺夫正则化(Tikhonov regularization)， 即通过将其添加到核矩阵(kernel matrix)的对角线。
除了明确指定噪声水平，另外的替代方法是将 WhiteKernel component 包含在内核中， 这样可以从数据中估计全局噪声水平（见下面的示例）。

算法实现是基于 [RW2006]_ 中的算法 2.1 。除了具有标准scikit-learn估计器的 API 之外， GaussianProcessRegressor ：

* 允许预测，无需事先拟合（基于GP先验）

* 提供了一个额外的方法  ``sample_y(X)``, 评估 在给定输入处 从GPR(先验或后验)中抽取的样本

* 暴露了一个方法 ``log_marginal_likelihood(theta)``, 可以在外部使用其他方式选择超参数，例如通过马尔科夫链链蒙特卡罗(Markov chain Monte Carlo)。


GPR 案例
============

带有噪声水平估计的GPR
-------------------------------
该示例说明带有sum-kernel(包括WhiteKernel)的 GPR 可以估计数据的噪声水平。 
对数边缘似然（LML）的景观图示表明存在 LML 的两个局部最大值。

.. figure:: ../auto_examples/gaussian_process/images/sphx_glr_plot_gpr_noisy_000.png
   :target: ../auto_examples/gaussian_process/plot_gpr_noisy.html
   :align: center

第一个对应于具有高噪声水平和大长度尺度的模型，这种情况是 由噪声解释了数据中的所有变化。

.. figure:: ../auto_examples/gaussian_process/images/sphx_glr_plot_gpr_noisy_001.png
   :target: ../auto_examples/gaussian_process/plot_gpr_noisy.html
   :align: center

第二个模型具有较小的噪声水平和较短的长度尺度，这种情况是 由无噪声的函数关系解释了大部分的变化。
第二种模型有较高的似然性(likelihood); 然而，根据超参数的初始值，基于梯度的优化也可能会收敛到高噪声解。 
因此，对于不同的初始化，重复优化多次是很重要的。

.. figure:: ../auto_examples/gaussian_process/images/sphx_glr_plot_gpr_noisy_002.png
   :target: ../auto_examples/gaussian_process/plot_gpr_noisy.html
   :align: center


GPR和KRR的比较
---------------------------------------------

KRR(kernel ridge regression,核岭回归)和 GPR 通过内部使用 “kernel trick(核技巧)” 来学习目标函数。 
KRR学习由相应内核诱导产生的空间中的线性函数，该函数对应于原始空间中的非线性函数。 
核诱导空间中的线性函数是基于带有岭正则化项的均方误差进行选择的。
GPR使用内核来定义目标函数的先验分布的协方差，并使用观察到的训练数据来定义似然函数。 
基于贝叶斯定理，目标函数上的（高斯）后验分布就可以推导出来了，那么其平均值则用于预测。

一个主要区别是，GPR 可以基于边际似然函数上的梯度上升选择内核的超参数， 
而KRR需要在交叉验证的损失函数（均方误差损失）上执行网格搜索。 
另一个区别是，GPR 学习目标函数的生成概率模型，
因此可以提供有意义的置信区间和后验样本以及预测值， 而KRR仅提供预测。

下图展示了KRR和GPR这两种方法在人造数据集上的表现，数据由正弦目标函数和强噪声构成。 
该图比较了在此数据上学习到的 基于ExpSineSquared内核的 KRR 和 GPR 模型，
ExpSineSquared内核适用于学习周期函数，内核的超参数控制内核的平滑度(length_scale)和周期性(periodicity)。 
此外，GPR 通过内核中附带的 WhiteKernel component 显式的学习数据的噪声水平；而 KRR 通过正则化参数 alpha 显式的学习数据的噪声水平。

.. figure:: ../auto_examples/gaussian_process/images/sphx_glr_plot_compare_gpr_krr_001.png
   :target: ../auto_examples/gaussian_process/plot_compare_gpr_krr.html
   :align: center

该图显示，两种方法都可以学习到合理的目标函数模型。 GPR将函数的周期正确地识别为 :math:`2*\pi` (6.28)，而 KRR 选择倍增的周期为 :math:`4*\pi` 。 
此外，GPR为预测提供了合理的置信区间，而KRR不能为预测提供置信区间。 两种方法之间的主要区别是拟合和预测所需的时间： 原则上KRR的拟合速度较快，
但是超参数优化的网格搜索与超参数的数量（ “curse of dimensionality(维数灾难)” ）呈指数级关系。 GPR中的参数基于梯度进行优化不受此指数级增长的影响，
因此在具有三维超参数空间的该示例上相当快。 预测的时间是相似的; 然而，生成 GPR 预测分布的方差需要的时间比仅预测其平均值要长。

GPR 算法在 Mauna Loa CO2 数据上的分析
-------------------------

该示例基于书([RW2006]_) 的第 5.4.3 节。 它演示了使用梯度上升的对数边缘似然性的复杂内核工程和超参数优化的示例。 
数据包括在 1958 年至 1997 年间夏威夷 Mauna Loa 天文台收集的每月平均大气二氧 化碳浓度（以百万分之几（ppmv）计）。
目的是将二氧化碳浓度建模为时间t的函数。

内核由若干项(several terms)组成，负责说明信号的不同属性：

- 一个长期平滑的上升趋势是由一个 RBF 内核来解释的。 具有较大长度尺寸的RBF内核将使该分量平滑; 
  没有强制这种趋势正在上升，这给 GP 带来了可选择性。 具体的长度尺度(length-scale)和振幅(amplitude)是自由的超参数。

- 季节性因素，由周期性的 ExpSineSquared 内核来解释，固定周期为1年。 该周期性分量的长度尺度(length-scale)
  控制其平滑度，是一个自由参数。 
  为了使其具备准确周期性的衰减，将 ExpSineSquared kernel 与 RBF kernel 取乘积(product)。 该RBF component的长度尺度(length-scale)控制衰减时间，
  并且是另一个自由参数。

- 较小的中期不规则性将由 RationalQuadratic kernel component 来解释， RationalQuadratic kernel component 的长度尺度(length-scale)和alpha 
  参数，决定着长度尺度的扩散性，是将要被确定的参数。 根据 [RW2006]_ ，这些不规则性可以更好地由有理二次内核(RationalQuadratic kernel) 来解释， 
  而不是 RBF kernel component，这可能是因为它可以容纳若干个长度尺度(length-scale)。

- 噪声项，由一个 RBF kernel 组成，它将解释相关的噪声分量，如局部天气现象以及 WhiteKernel 对白噪声的贡献。 在这里，
  相对幅度(relative amplitudes)和RBF的length scale 又是自由参数。

减去目标平均值后 最大化 对数边际似然(log-marginal-likelihood)产生下列内核，其中LML为-83.214:

::

   34.4**2 * RBF(length_scale=41.8)
   + 3.27**2 * RBF(length_scale=180) * ExpSineSquared(length_scale=1.44,
                                                      periodicity=1)
   + 0.446**2 * RationalQuadratic(alpha=17.7, length_scale=0.957)
   + 0.197**2 * RBF(length_scale=0.138) + WhiteKernel(noise_level=0.0336)

因此，大多数目标信号（34.4ppm）由长期上升趋势（长度即length-scale为41.8年）解释。 周期分量的振幅为3.27ppm，衰减时间为180年，长度为(length-scale)为1.44。 
长时间的衰变时间表明我们有一个局部非常接近周期性的季节性成分。 相关噪声的幅度为0.197ppm，长度为0.138年，白噪声贡献为0.197ppm。 
因此，整体噪声水平非常小，表明该模型可以很好地解释数据。 该图还显示，该模型直到2015年左右才能做出置信度比较高的预测。

.. figure:: ../auto_examples/gaussian_process/images/sphx_glr_plot_gpr_co2_001.png
   :target: ../auto_examples/gaussian_process/plot_gpr_co2.html
   :align: center

.. _gpc:

高斯过程分类器 (GPC)
=====================================

.. currentmodule:: sklearn.gaussian_process

The :class:`GaussianProcessClassifier` implements Gaussian processes (GP) for
classification purposes, more specifically for probabilistic classification,
where test predictions take the form of class probabilities.
GaussianProcessClassifier places a GP prior on a latent function :math:`f`,
which is then squashed through a link function to obtain the probabilistic
classification. The latent function :math:`f` is a so-called nuisance function,
whose values are not observed and are not relevant by themselves.
Its purpose is to allow a convenient formulation of the model, and :math:`f`
is removed (integrated out) during prediction. GaussianProcessClassifier
implements the logistic link function, for which the integral cannot be
computed analytically but is easily approximated in the binary case.

In contrast to the regression setting, the posterior of the latent function
:math:`f` is not Gaussian even for a GP prior since a Gaussian likelihood is
inappropriate for discrete class labels. Rather, a non-Gaussian likelihood
corresponding to the logistic link function (logit) is used.
GaussianProcessClassifier approximates the non-Gaussian posterior with a
Gaussian based on the Laplace approximation. More details can be found in
Chapter 3 of [RW2006]_.

The GP prior mean is assumed to be zero. The prior's
covariance is specified by passing a :ref:`kernel <gp_kernels>` object. The
hyperparameters of the kernel are optimized during fitting of
GaussianProcessRegressor by maximizing the log-marginal-likelihood (LML) based
on the passed ``optimizer``. As the LML may have multiple local optima, the
optimizer can be started repeatedly by specifying ``n_restarts_optimizer``. The
first run is always conducted starting from the initial hyperparameter values
of the kernel; subsequent runs are conducted from hyperparameter values
that have been chosen randomly from the range of allowed values.
If the initial hyperparameters should be kept fixed, `None` can be passed as
optimizer.

:class:`GaussianProcessClassifier` supports multi-class classification
by performing either one-versus-rest or one-versus-one based training and
prediction.  In one-versus-rest, one binary Gaussian process classifier is
fitted for each class, which is trained to separate this class from the rest.
In "one_vs_one", one binary Gaussian process classifier is fitted for each pair
of classes, which is trained to separate these two classes. The predictions of
these binary predictors are combined into multi-class predictions. See the
section on :ref:`multi-class classification <multiclass>` for more details.

In the case of Gaussian process classification, "one_vs_one" might be
computationally  cheaper since it has to solve many problems involving only a
subset of the whole training set rather than fewer problems on the whole
dataset. Since Gaussian process classification scales cubically with the size
of the dataset, this might be considerably faster. However, note that
"one_vs_one" does not support predicting probability estimates but only plain
predictions. Moreover, note that :class:`GaussianProcessClassifier` does not
(yet) implement a true multi-class Laplace approximation internally, but
as discussed above is based on solving several binary classification tasks
internally, which are combined using one-versus-rest or one-versus-one.

GPC 案列
============

使用 GPC 进行 概率化预测
----------------------------------

This example illustrates the predicted probability of GPC for an RBF kernel
with different choices of the hyperparameters. The first figure shows the
predicted probability of GPC with arbitrarily chosen hyperparameters and with
the hyperparameters corresponding to the maximum log-marginal-likelihood (LML).

While the hyperparameters chosen by optimizing LML have a considerable larger
LML, they perform slightly worse according to the log-loss on test data. The
figure shows that this is because they exhibit a steep change of the class
probabilities at the class boundaries (which is good) but have predicted
probabilities close to 0.5 far away from the class boundaries (which is bad)
This undesirable effect is caused by the Laplace approximation used
internally by GPC.

The second figure shows the log-marginal-likelihood for different choices of
the kernel's hyperparameters, highlighting the two choices of the
hyperparameters used in the first figure by black dots.

.. figure:: ../auto_examples/gaussian_process/images/sphx_glr_plot_gpc_000.png
   :target: ../auto_examples/gaussian_process/plot_gpc.html
   :align: center

.. figure:: ../auto_examples/gaussian_process/images/sphx_glr_plot_gpc_001.png
   :target: ../auto_examples/gaussian_process/plot_gpc.html
   :align: center


GPC分类器在异或问题上的应用
--------------------------------------

.. currentmodule:: sklearn.gaussian_process.kernels

This example illustrates GPC on XOR data. Compared are a stationary, isotropic
kernel (:class:`RBF`) and a non-stationary kernel (:class:`DotProduct`). On this particular
dataset, the `DotProduct` kernel obtains considerably better results because the
class-boundaries are linear and coincide with the coordinate axes. In practice,
however, stationary kernels such as :class:`RBF` often obtain better results.

.. figure:: ../auto_examples/gaussian_process/images/sphx_glr_plot_gpc_xor_001.png
   :target: ../auto_examples/gaussian_process/plot_gpc_xor.html
   :align: center

.. currentmodule:: sklearn.gaussian_process


GPC分类器在鸢尾花数据集上的应用
-----------------------------------------------------

This example illustrates the predicted probability of GPC for an isotropic
and anisotropic RBF kernel on a two-dimensional version for the iris-dataset.
This illustrates the applicability of GPC to non-binary classification.
The anisotropic RBF kernel obtains slightly higher log-marginal-likelihood by
assigning different length-scales to the two feature dimensions.

.. figure:: ../auto_examples/gaussian_process/images/sphx_glr_plot_gpc_iris_001.png
   :target: ../auto_examples/gaussian_process/plot_gpc_iris.html
   :align: center


.. _gp_kernels:

用于高斯过程的核函数
==============================
.. currentmodule:: sklearn.gaussian_process.kernels

Kernels (also called "covariance functions" in the context of GPs) are a crucial
ingredient of GPs which determine the shape of prior and posterior of the GP.
They encode the assumptions on the function being learned by defining the "similarity"
of two datapoints combined with the assumption that similar datapoints should
have similar target values. Two categories of kernels can be distinguished:
stationary kernels depend only on the distance of two datapoints and not on their
absolute values :math:`k(x_i, x_j)= k(d(x_i, x_j))` and are thus invariant to
translations in the input space, while non-stationary kernels
depend also on the specific values of the datapoints. Stationary kernels can further
be subdivided into isotropic and anisotropic kernels, where isotropic kernels are
also invariant to rotations in the input space. For more details, we refer to
Chapter 4 of [RW2006]_.

高斯过程核函数 API
---------------------------
The main usage of a :class:`Kernel` is to compute the GP's covariance between
datapoints. For this, the method ``__call__`` of the kernel can be called. This
method can either be used to compute the "auto-covariance" of all pairs of
datapoints in a 2d array X, or the "cross-covariance" of all combinations
of datapoints of a 2d array X with datapoints in a 2d array Y. The following
identity holds true for all kernels k (except for the :class:`WhiteKernel`):
``k(X) == K(X, Y=X)``

If only the diagonal of the auto-covariance is being used, the method ``diag()``
of a kernel can be called, which is more computationally efficient than the
equivalent call to ``__call__``: ``np.diag(k(X, X)) == k.diag(X)``

Kernels are parameterized by a vector :math:`\theta` of hyperparameters. These
hyperparameters can for instance control length-scales or periodicity of a
kernel (see below). All kernels support computing analytic gradients 
of the kernel's auto-covariance with respect to :math:`\theta` via setting
``eval_gradient=True`` in the ``__call__`` method. This gradient is used by the
Gaussian process (both regressor and classifier) in computing the gradient
of the log-marginal-likelihood, which in turn is used to determine the
value of :math:`\theta`, which maximizes the log-marginal-likelihood,  via
gradient ascent. For each hyperparameter, the initial value and the
bounds need to be specified when creating an instance of the kernel. The
current value of :math:`\theta` can be get and set via the property
``theta`` of the kernel object. Moreover, the bounds of the hyperparameters can be
accessed by the property ``bounds`` of the kernel. Note that both properties
(theta and bounds) return log-transformed values of the internally used values
since those are typically more amenable to gradient-based optimization.
The specification of each hyperparameter is stored in the form of an instance of
:class:`Hyperparameter` in the respective kernel. Note that a kernel using a
hyperparameter with name "x" must have the attributes self.x and self.x_bounds.

The abstract base class for all kernels is :class:`Kernel`. Kernel implements a
similar interface as :class:`Estimator`, providing the methods ``get_params()``,
``set_params()``, and ``clone()``. This allows setting kernel values also via
meta-estimators such as :class:`Pipeline` or :class:`GridSearch`. Note that due to the nested
structure of kernels (by applying kernel operators, see below), the names of
kernel parameters might become relatively complicated. In general, for a
binary kernel operator, parameters of the left operand are prefixed with ``k1__``
and parameters of the right operand with ``k2__``. An additional convenience
method is ``clone_with_theta(theta)``, which returns a cloned version of the
kernel but with the hyperparameters set to ``theta``. An illustrative example:

    >>> from sklearn.gaussian_process.kernels import ConstantKernel, RBF
    >>> kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(0.0, 10.0)) * RBF(length_scale=0.5, length_scale_bounds=(0.0, 10.0)) + RBF(length_scale=2.0, length_scale_bounds=(0.0, 10.0))
    >>> for hyperparameter in kernel.hyperparameters: print(hyperparameter)
    Hyperparameter(name='k1__k1__constant_value', value_type='numeric', bounds=array([[ 0., 10.]]), n_elements=1, fixed=False)
    Hyperparameter(name='k1__k2__length_scale', value_type='numeric', bounds=array([[ 0., 10.]]), n_elements=1, fixed=False)
    Hyperparameter(name='k2__length_scale', value_type='numeric', bounds=array([[ 0., 10.]]), n_elements=1, fixed=False)
    >>> params = kernel.get_params()
    >>> for key in sorted(params): print("%s : %s" % (key, params[key]))
    k1 : 1**2 * RBF(length_scale=0.5)
    k1__k1 : 1**2
    k1__k1__constant_value : 1.0
    k1__k1__constant_value_bounds : (0.0, 10.0)
    k1__k2 : RBF(length_scale=0.5)
    k1__k2__length_scale : 0.5
    k1__k2__length_scale_bounds : (0.0, 10.0)
    k2 : RBF(length_scale=2)
    k2__length_scale : 2.0
    k2__length_scale_bounds : (0.0, 10.0)
    >>> print(kernel.theta)  # Note: log-transformed
    [ 0.         -0.69314718  0.69314718]
    >>> print(kernel.bounds)  # Note: log-transformed
    [[      -inf 2.30258509]
     [      -inf 2.30258509]
     [      -inf 2.30258509]]


All Gaussian process kernels are interoperable with :mod:`sklearn.metrics.pairwise`
and vice versa: instances of subclasses of :class:`Kernel` can be passed as
``metric`` to ``pairwise_kernels`` from :mod:`sklearn.metrics.pairwise`. Moreover,
kernel functions from pairwise can be used as GP kernels by using the wrapper
class :class:`PairwiseKernel`. The only caveat is that the gradient of
the hyperparameters is not analytic but numeric and all those kernels support
only isotropic distances. The parameter ``gamma`` is considered to be a
hyperparameter and may be optimized. The other kernel parameters are set
directly at initialization and are kept fixed.


基本核函数
-------------
The :class:`ConstantKernel` kernel can be used as part of a :class:`Product`
kernel where it scales the magnitude of the other factor (kernel) or as part
of a :class:`Sum` kernel, where it modifies the mean of the Gaussian process.
It depends on a parameter :math:`constant\_value`. It is defined as:

.. math::
   k(x_i, x_j) = constant\_value \;\forall\; x_1, x_2

The main use-case of the :class:`WhiteKernel` kernel is as part of a
sum-kernel where it explains the noise-component of the signal. Tuning its
parameter :math:`noise\_level` corresponds to estimating the noise-level.
It is defined as:

.. math::
    k(x_i, x_j) = noise\_level \text{ if } x_i == x_j \text{ else } 0


核算子(Kernel operators)
----------------
Kernel operators take one or two base kernels and combine them into a new
kernel. The :class:`Sum` kernel takes two kernels :math:`k1` and :math:`k2`
and combines them via :math:`k_{sum}(X, Y) = k1(X, Y) + k2(X, Y)`.
The  :class:`Product` kernel takes two kernels :math:`k1` and :math:`k2`
and combines them via :math:`k_{product}(X, Y) = k1(X, Y) * k2(X, Y)`.
The :class:`Exponentiation` kernel takes one base kernel and a scalar parameter
:math:`exponent` and combines them via
:math:`k_{exp}(X, Y) = k(X, Y)^\text{exponent}`.

径向基核函数 (RBF)
----------------------------------
The :class:`RBF` kernel is a stationary kernel. It is also known as the "squared
exponential" kernel. It is parameterized by a length-scale parameter :math:`l>0`, which
can either be a scalar (isotropic variant of the kernel) or a vector with the same
number of dimensions as the inputs :math:`x` (anisotropic variant of the kernel).
The kernel is given by:

.. math::
   k(x_i, x_j) = \text{exp}\left(-\frac{1}{2} d(x_i / l, x_j / l)^2\right)

This kernel is infinitely differentiable, which implies that GPs with this
kernel as covariance function have mean square derivatives of all orders, and are thus
very smooth. The prior and posterior of a GP resulting from an RBF kernel are shown in
the following figure:

.. figure:: ../auto_examples/gaussian_process/images/sphx_glr_plot_gpr_prior_posterior_000.png
   :target: ../auto_examples/gaussian_process/plot_gpr_prior_posterior.html
   :align: center


Matérn 核
-------------
The :class:`Matern` kernel is a stationary kernel and a generalization of the
:class:`RBF` kernel. It has an additional parameter :math:`\nu` which controls
the smoothness of the resulting function. It is parameterized by a length-scale parameter :math:`l>0`, which can either be a scalar (isotropic variant of the kernel) or a vector with the same number of dimensions as the inputs :math:`x` (anisotropic variant of the kernel). The kernel is given by:

.. math::

    k(x_i, x_j) = \sigma^2\frac{1}{\Gamma(\nu)2^{\nu-1}}\Bigg(\gamma\sqrt{2\nu} d(x_i / l, x_j / l)\Bigg)^\nu K_\nu\Bigg(\gamma\sqrt{2\nu} d(x_i / l, x_j / l)\Bigg),

As :math:`\nu\rightarrow\infty`, the Matérn kernel converges to the RBF kernel.
When :math:`\nu = 1/2`, the Matérn kernel becomes identical to the absolute
exponential kernel, i.e.,

.. math::
    k(x_i, x_j) = \sigma^2 \exp \Bigg(-\gamma d(x_i / l, x_j / l) \Bigg) \quad \quad \nu= \tfrac{1}{2}

In particular, :math:`\nu = 3/2`:

.. math::
    k(x_i, x_j) = \sigma^2 \Bigg(1 + \gamma \sqrt{3} d(x_i / l, x_j / l)\Bigg) \exp \Bigg(-\gamma \sqrt{3}d(x_i / l, x_j / l) \Bigg) \quad \quad \nu= \tfrac{3}{2}

and :math:`\nu = 5/2`:

.. math::
    k(x_i, x_j) = \sigma^2 \Bigg(1 + \gamma \sqrt{5}d(x_i / l, x_j / l) +\frac{5}{3} \gamma^2d(x_i / l, x_j / l)^2 \Bigg) \exp \Bigg(-\gamma \sqrt{5}d(x_i / l, x_j / l) \Bigg) \quad \quad \nu= \tfrac{5}{2}

are popular choices for learning functions that are not infinitely
differentiable (as assumed by the RBF kernel) but at least once (:math:`\nu =
3/2`) or twice differentiable (:math:`\nu = 5/2`).

The flexibility of controlling the smoothness of the learned function via :math:`\nu`
allows adapting to the properties of the true underlying functional relation.
The prior and posterior of a GP resulting from a Matérn kernel are shown in
the following figure:

.. figure:: ../auto_examples/gaussian_process/images/sphx_glr_plot_gpr_prior_posterior_004.png
   :target: ../auto_examples/gaussian_process/plot_gpr_prior_posterior.html
   :align: center

See [RW2006]_, pp84 for further details regarding the
different variants of the Matérn kernel.

有理二次核函数(Rational quadratic kernel)
-------------------------

The :class:`RationalQuadratic` kernel can be seen as a scale mixture (an infinite sum)
of :class:`RBF` kernels with different characteristic length-scales. It is parameterized
by a length-scale parameter :math:`l>0` and a scale mixture parameter  :math:`\alpha>0`
Only the isotropic variant where :math:`l` is a scalar is supported at the moment.
The kernel is given by:

.. math::
   k(x_i, x_j) = \left(1 + \frac{d(x_i, x_j)^2}{2\alpha l^2}\right)^{-\alpha}

The prior and posterior of a GP resulting from a :class:`RationalQuadratic` kernel are shown in
the following figure:

.. figure:: ../auto_examples/gaussian_process/images/sphx_glr_plot_gpr_prior_posterior_001.png
   :target: ../auto_examples/gaussian_process/plot_gpr_prior_posterior.html
   :align: center

Exp-Sine-Squared kernel
-----------------------

The :class:`ExpSineSquared` kernel allows modeling periodic functions.
It is parameterized by a length-scale parameter :math:`l>0` and a periodicity parameter
:math:`p>0`. Only the isotropic variant where :math:`l` is a scalar is supported at the moment.
The kernel is given by:

.. math::
   k(x_i, x_j) = \text{exp}\left(-2 \left(\text{sin}(\pi / p * d(x_i, x_j)) / l\right)^2\right)

The prior and posterior of a GP resulting from an ExpSineSquared kernel are shown in
the following figure:

.. figure:: ../auto_examples/gaussian_process/images/sphx_glr_plot_gpr_prior_posterior_002.png
   :target: ../auto_examples/gaussian_process/plot_gpr_prior_posterior.html
   :align: center

点乘(Dot-Product) kernel
------------------

The :class:`DotProduct` kernel is non-stationary and can be obtained from linear regression
by putting :math:`N(0, 1)` priors on the coefficients of :math:`x_d (d = 1, . . . , D)` and
a prior of :math:`N(0, \sigma_0^2)` on the bias. The :class:`DotProduct` kernel is invariant to a rotation
of the coordinates about the origin, but not translations.
It is parameterized by a parameter :math:`\sigma_0^2`. For :math:`\sigma_0^2 = 0`, the kernel
is called the homogeneous linear kernel, otherwise it is inhomogeneous. The kernel is given by

.. math::
   k(x_i, x_j) = \sigma_0 ^ 2 + x_i \cdot x_j

The :class:`DotProduct` kernel is commonly combined with exponentiation. An example with exponent 2 is
shown in the following figure:

.. figure:: ../auto_examples/gaussian_process/images/sphx_glr_plot_gpr_prior_posterior_003.png
   :target: ../auto_examples/gaussian_process/plot_gpr_prior_posterior.html
   :align: center

参考文献
----------

.. [RW2006] Carl Eduard Rasmussen and Christopher K.I. Williams, "Gaussian Processes for Machine Learning", MIT Press 2006, Link to an official complete PDF version of the book `here <http://www.gaussianprocess.org/gpml/chapters/RW.pdf>`_ .

.. currentmodule:: sklearn.gaussian_process
