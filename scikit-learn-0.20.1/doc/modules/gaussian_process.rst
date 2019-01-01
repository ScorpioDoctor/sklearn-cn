

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

:class:`GaussianProcessClassifier` 类将高斯过程(GP)用于分类，更具体地说是用于概率分类，即 采用类概率的形式进行预测。
GaussianProcessClassifier 把一个GP prior放在隐函数(latent function) :math:`f` 上, 然后通过一个链接函数(link function) 把其压缩来获得
概率性分类(probabilistic classification)。 隐函数 :math:`f` 被称之为 滋扰函数(nuisance function), 其 取值不可被观测，相互之间不相干。
该函数的目的是为了模型的方便的形式，而且 :math:`f` 在预测阶段就被去除了。
GaussianProcessClassifier 实现了 logistic 链接函数，其积分不能被解析的计算但是在二元分类的情况下很容易逼近。

与GP在回归问题中的设置相对比, 隐函数 :math:`f` 的后验概率分布不是高斯分布，甚至对 GP prior 也不是，因为高斯似然函数(Gaussian likelihood)用于离散的类标签
是不合适的。 所以, 一个与logistic链接函数对应的non-Gaussian likelihood 被用作隐函数 :math:`f` 的后验概率分布。
GaussianProcessClassifier使用基于Laplace近似的高斯分布去逼近非高斯后验(non-Gaussian posterior)。 更多详情，请看 [RW2006]_ 的第三章吧！

GP先验的平均值假定为零。先验的协方差是通过传递 内核(:ref:`kernel <gp_kernels>`) 对象来指定的。 
通过由参数 ``optimizer`` 指定的优化器 最大化 对数边缘似然估计(LML)，内核的超参数可以在 GaussianProcessClassifier 的拟合过程中被优化。
由于LML可能具有多个局部最优值， 所以优化器可以通过指定 ``n_restarts_optimizer`` 而重复启动。 
优化器第一次运行的时候总是从一组初始的内核超参数上启动；后续的优化过程从被允许的合理取值范围内随机选择一组超参数开始优化。
如果需要保持初始的超参值固定不变， 那么需要把优化器设置为 `None` 。

:class:`GaussianProcessClassifier` 类通过执行基于OvR(one-versus-rest)或 OvO(one-versus-one)策略的训练和预测来支持多类分类。 
在OvR策略中，每个类都配有一个二元高斯过程分类器，其被训练为将该类与其余类分开。
在OVO策略中，对每两个类拟合一个二元高斯过程分类器，其被训练为将这两个类分开。
最后，这些二元分类器的预测被组合成多类预测。更多详细信息，请参阅 :ref:`multi-class classification <multiclass>` 。

在高斯过程分类的使用中，”one_vs_one” 策略可能在计算上更廉价， 因为它必须解决涉及整个训练集的每一个子集的许多问题， 
而不是整个数据集的较少的问题。由于高斯过程分类随着数据集的大小以立方形式缩放(scales cubically)，所以这可能要快得多。 
但是，请注意，”one_vs_one” 不支持预测概率估计，而只是简单的预测。 
此外，请注意， :class:`GaussianProcessClassifier` 在内部还没有实现真正的多类 Laplace 近似， 
但如上所述，在解决内部二元分类任务的基础上，它们使用 OvR或 OvO 的组合方法。


GPC 案列
============

使用 GPC 进行 概率化预测
----------------------------------

这个例子展示了具有不同超参数选项的RBF内核的GPC预测概率。 
第一幅图显示 具有 任意选择的超参数 的GPC的预测概率 以及 具有 与最大LML对应的超参数 的GPC的预测概率。

虽然通过优化LML选择的超参数具有相当大的LML，但是依据测试数据上的对数损失，它们的表现更差。 该图显示，
这是因为它们在类边界表现出类概率的急剧变化(这是好的表现)， 但在远离类边界的地方 其预测概率却接近0.5（这是坏的表现）
这种不良影响是由于GPC内部使用了拉普拉斯近似(Laplace approximation)。

第二幅图显示了 针对内核超参数的不同选择所对应的LML（对数边缘似然），突出了在第一幅图中使用的通过黑点（训练集）选择的两个超参数。

.. figure:: ../auto_examples/gaussian_process/images/sphx_glr_plot_gpc_000.png
   :target: ../auto_examples/gaussian_process/plot_gpc.html
   :align: center

.. figure:: ../auto_examples/gaussian_process/images/sphx_glr_plot_gpc_001.png
   :target: ../auto_examples/gaussian_process/plot_gpc.html
   :align: center


GPC分类器在异或问题上的应用
--------------------------------------

.. currentmodule:: sklearn.gaussian_process.kernels

此示例展示了将GPC用于XOR数据。参与比较试验的是 平稳的各向同性的核(:class:`RBF`)和非平稳的核(:class:`DotProduct`)。 
在这个特定的数据集上， `DotProduct` kernel 获得了更好的结果，
因为类边界是线性的，并且与坐标轴重合。 然而，实际上，平稳的核 诸如 :class:`RBF` 经常获得更好结果。

.. figure:: ../auto_examples/gaussian_process/images/sphx_glr_plot_gpc_xor_001.png
   :target: ../auto_examples/gaussian_process/plot_gpc_xor.html
   :align: center

.. currentmodule:: sklearn.gaussian_process


GPC分类器在iris数据集上的应用
-----------------------------------------------------

该示例展示了 在iris数据集的二维版本上 各向同性和各向异性RBF核的GPC的预测概率。 
这说明了GPC对多类分类的适用性。 
各向异性RBF内核通过为两个特征维度分配不同的长度尺度(length-scales)来获得稍高的LML(log-marginal-likelihood)。

.. figure:: ../auto_examples/gaussian_process/images/sphx_glr_plot_gpc_iris_001.png
   :target: ../auto_examples/gaussian_process/plot_gpc_iris.html
   :align: center


.. _gp_kernels:

用于高斯过程的核函数
==============================
.. currentmodule:: sklearn.gaussian_process.kernels

Kernels(在GPs的上下文语境中也可以叫做 ”协方差函数”) 是决定 GP的先验形状和后验形状(the shape of prior and posterior of the GP)的关键组成部分。 
它们通过定义两个数据点的“相似性”，并结合相似的数据点应该具有相似的目标值的假设，对所学习的函数进行编码。 
内核可以分为两类：平稳内核(stationary kernels)，只取决于两个数据点的距离，不依赖于它们的绝对值 :math:`k(x_i, x_j)= k(d(x_i, x_j))` ，
因此它们对输入空间中的平移是不变的；非平稳内核(non-stationary kernels)，取决于数据点的具体值。
平稳内核可以进一步细分为各向同性(isotropic)和各向异性(anisotropic)内核，
其中各向同性内核对输入空间中的旋转具有不变性。想要了解更多细节，请参看 [RW2006]_ 的第四章。


高斯过程核函数 API
---------------------------
:class:`Kernel` 类主要用来计算数据点之间的高斯过程的协方差。 为此，内核中 ``__call__`` 方法会被调用。
该方法即可以用于计算 2d数组X中所有数据点对的“自协方差(auto-covariance)”，
也可以计算 2d数组X中的数据点与2d数组Y中的数据点的所有组合的“互协方差(cross-covariance)”。
以下等式对于 所有内核 k（除了 :class:`WhiteKernel` ）都是成立的：``k(X) == K(X, Y=X)`` 。 

如果仅仅是自协方差的对角线元素被使用，那么内核的方法 ``diag()`` 将会被调用， 
该方法比等价的调用 ``__call__`` : ``np.diag(k(X, X)) == k.diag(X)`` 具有更高的计算效率。

内核通过超参数向量 :math:`\theta` 进行参数化。这些超参数可以控制例如内核的长度尺度(length-scales)或周期性(periodicity)（见下文）。
通过设置 ``__call__`` 方法的参数 ``eval_gradient=True`` ，所有的内核都支持计算 内核自协方差 对于 :math:`\theta` 的解析梯度。
该梯度被用来在(回归型和分类型的)高斯过程中计算LML(对数边缘似然)函数的梯度，进而被用来通过梯度上升的方法极大化LML(对数边缘似然)函数，
从而确定 :math:`\theta` 的值。
对于每个超参数，当创建内核的实例时，初始值和边界值需要被指定。 :math:`\theta` 的当前值可以通过内核对象属性 ``theta`` 被获取或者设置。
更重要的是， 超参的边界值可以被内核属性 ``bounds`` 获取。需要注意的是， 
以上两种属性值(theta和bounds)都会返回内部使用值的对数转换值(log-transformed values)，这是因为这两种属性值通常更适合基于梯度的优化。
每个超参数的规格(specification)以 :class:`Hyperparameter` 类的实例的形式被存储在相应内核中。 
请注意使用了以"x"命名的超参的内核必然具有 self.x 和 self.x_bounds 这两种属性。

所有内核的抽象基类为 :class:`Kernel` 。Kernel 基类实现了与 :class:`Estimator` 基类相似的接口，
提供了方法 ``get_params()`` , ``set_params()`` 以及 ``clone()`` 。这样的实现方法也允许通过 meta-estimators 诸如
:class:`Pipeline` or :class:`GridSearch` 来设置内核的values。 
需要注意的是，由于内核的嵌套结构（通过应用内核算子(kernel operators)，如下所见）， 内核参数的名称可能会变得相对复杂些。
通常来说，对于二元内核算子，左运算元的参数以 ``k1__`` 为前缀，而右运算元以 ``k2__`` 为前缀。 
另一个便利方法是 ``clone_with_theta(theta)`` ， 该方法返回克隆版本的内核，但是设置超参数为 ``theta`` 。 
示例如下::

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


所有的 GP kernels 都可以与 :mod:`sklearn.metrics.pairwise` 进行互操作，反之亦然： 
类 :class:`Kernel` 的子类实例可以通过 ``metric`` 参数传给 :mod:`sklearn.metrics.pairwise` 中的 ``pairwise_kernels`` 。
更进一步，来自 pairwise 的核函数可以通过封装类 :class:`PairwiseKernel` 被用作 GP kernels。
唯一警告是，超参数的梯度不是解析的(analytic)，而是数值的(numeric)，并且所有这些内核只支持各向同性距离(isotropic distances)。
该参数 ``gamma`` 被认为是一个超参数，可以进行优化。其他内核参数在初始化时直接设置，并保持固定。



基本核函数
-------------
:class:`ConstantKernel` 内核类 既可以被用作 :class:`Product` 内核类的一部分，在其中，
它可以缩放其他因子（内核）的量级 ； 也可以作为 :class:`Sum` 类的一部分，在其中，它修改高斯过程的均值。
这取决于参数 :math:`constant\_value` 的设置。该方法定义为:

.. math::
   k(x_i, x_j) = constant\_value \;\forall\; x_1, x_2

:class:`WhiteKernel` 核的主要使用情景是作为 sum-kernel 的一部分，它在其中解释信号的噪声成分(noise-component)。 
调节它的参数 :math:`noise\_level` 对应于估计噪声水平。它被定义如下：

.. math::
    k(x_i, x_j) = noise\_level \text{ if } x_i == x_j \text{ else } 0


核算子(Kernel operators)
----------------
内核算子(Kernel operators)接受一到两个基本内核并将它们组合起来变成一个新核。
内核类 :class:`Sum` 接受两个内核 :math:`k1` 和 :math:`k2` 并且通过  :math:`k_{sum}(X, Y) = k1(X, Y) + k2(X, Y)` 
把它们组合起来。 内核类 :class:`Product` 接受两个内核 :math:`k1` 和 :math:`k2` 并且通过 :math:`k_{product}(X, Y) = k1(X, Y) * k2(X, Y)`
把它们组合起来。 内核类 :class:`Exponentiation` 接受一个基本核和一个标量参数 :math:`exponent` 并通过 
:math:`k_{exp}(X, Y) = k(X, Y)^\text{exponent}` 
把它们组合起来。

径向基函数 (RBF)
----------------------------------
:class:`RBF` 核 是一个平稳核(stationary kernel)。 它也被称为 "squared exponential" 核。
它可以通过长度尺度(length-scale)参数 :math:`l>0` 被参数化。该参数既可以是标量(内核的各向同性变体)
或者是 与输入 :math:`x` 具有相同维数的向量(内核的各向异性变体) 。该内核可以被定义为:

.. math::
   k(x_i, x_j) = \text{exp}\left(-\frac{1}{2} d(x_i / l, x_j / l)^2\right)

这个内核是无限可微分的, 这暗含着带有此核的GPs作为协方差函数具有所有阶的平均平方导数(mean square derivatives of all orders)，
并且因此非常平滑。由RBF核产生的GP的先验与后验(prior and posterior)在下图中展示:

.. figure:: ../auto_examples/gaussian_process/images/sphx_glr_plot_gpr_prior_posterior_000.png
   :target: ../auto_examples/gaussian_process/plot_gpr_prior_posterior.html
   :align: center


Matérn 核
-------------
:class:`Matern` 内核是一个平稳内核(stationary kernel)，是 :class:`RBF` 内核的泛化。它有一个附加的参数 :math:`\nu`， 
该参数控制结果函数的平滑程度。它由长度尺度参数(length-scale) :math:`l>0` 来实现参数化。
该参数既可以是标量(内核的各向同性变体) 或者是 与输入 :math:`x` 具有相同维数的向量(内核的各向异性变体) 。
该内核可以被定义为:

.. math::

    k(x_i, x_j) = \sigma^2\frac{1}{\Gamma(\nu)2^{\nu-1}}\Bigg(\gamma\sqrt{2\nu} d(x_i / l, x_j / l)\Bigg)^\nu K_\nu\Bigg(\gamma\sqrt{2\nu} d(x_i / l, x_j / l)\Bigg),

由于 :math:`\nu\rightarrow\infty`, Matérn kernel 收敛到 RBF kernel。
当 :math:`\nu = 1/2` 时, Matérn kernel 变得与绝对指数核(absolute exponential kernel)一模一样啦, i.e.,

.. math::
    k(x_i, x_j) = \sigma^2 \exp \Bigg(-\gamma d(x_i / l, x_j / l) \Bigg) \quad \quad \nu= \tfrac{1}{2}

特别的, :math:`\nu = 3/2`:

.. math::
    k(x_i, x_j) = \sigma^2 \Bigg(1 + \gamma \sqrt{3} d(x_i / l, x_j / l)\Bigg) \exp \Bigg(-\gamma \sqrt{3}d(x_i / l, x_j / l) \Bigg) \quad \quad \nu= \tfrac{3}{2}

和 :math:`\nu = 5/2`:

.. math::
    k(x_i, x_j) = \sigma^2 \Bigg(1 + \gamma \sqrt{5}d(x_i / l, x_j / l) +\frac{5}{3} \gamma^2d(x_i / l, x_j / l)^2 \Bigg) \exp \Bigg(-\gamma \sqrt{5}d(x_i / l, x_j / l) \Bigg) \quad \quad \nu= \tfrac{5}{2}

是不无限可微的(由RBF kernel假定)但至少一阶可微(:math:`\nu = 3/2`)或二阶可微(:math:`\nu = 5/2`)的学习函数的常用选择。

通过 :math:`\nu` 灵活控制学习函数的平滑性可以更加适应真正的底层函数的关联属性。 通过 Matérn 内核产生的GP的先验和后验如下图所示:

.. figure:: ../auto_examples/gaussian_process/images/sphx_glr_plot_gpr_prior_posterior_004.png
   :target: ../auto_examples/gaussian_process/plot_gpr_prior_posterior.html
   :align: center

请看 [RW2006]_, pp84 查找更多关于Matérn核的不同变体的详情。

有理二次内核(Rational quadratic kernel)
-------------------------

:class:`RationalQuadratic` 内核可以被看做不同特征尺度下的 :class:`RBF` 内核的规模混合(一个无穷和) 
它通过长度尺度(length-scales)参数 :math:`l>0` 和比例混合参数 :math:`\alpha>0` 进行参数化。 
当前仅支持各向同性变体(其中 :math:`l` 是标量)。内核公式如下：

.. math::
   k(x_i, x_j) = \left(1 + \frac{d(x_i, x_j)^2}{2\alpha l^2}\right)^{-\alpha}

由 :class:`RationalQuadratic` 核产生的GP的先验和后验在下图中展示:

.. figure:: ../auto_examples/gaussian_process/images/sphx_glr_plot_gpr_prior_posterior_001.png
   :target: ../auto_examples/gaussian_process/plot_gpr_prior_posterior.html
   :align: center

指数正弦平方核(Exp-Sine-Squared kernel)
-----------------------

:class:`ExpSineSquared` 内核类可以用来对周期性函数(periodic functions)进行建模。
它通过长度尺度(length-scale)参数 :math:`l>0` 和 周期性(periodicity)参数 :math:`p>0` 进行参数化。
当前仅支持各向同性变体(其中 :math:`l` 是标量)。内核公式如下：

.. math::
   k(x_i, x_j) = \text{exp}\left(-2 \left(\text{sin}(\pi / p * d(x_i, x_j)) / l\right)^2\right)

由 :class:`ExpSineSquared` 核产生的GP的先验和后验在下图中展示:

.. figure:: ../auto_examples/gaussian_process/images/sphx_glr_plot_gpr_prior_posterior_002.png
   :target: ../auto_examples/gaussian_process/plot_gpr_prior_posterior.html
   :align: center

点乘核(Dot-Product kernel)
------------------

:class:`DotProduct` 内核类 是一个非平稳核(non-stationary kernel) 。
该内核可以从线性回归中得到，而这个线性回归又是通过把 :math:`N(0, 1)` 先验放在 :math:`x_d (d = 1, . . . , D)` 的系数上
以及把 :math:`N(0, \sigma_0^2)` 先验放在偏置上得到的。
:class:`DotProduct` 核对关于原点的坐标旋转是具有不变性的，但是它没有平移不变性。
它通过参数 :math:`\sigma_0^2` 进行参数化。 对于 :math:`\sigma_0^2 = 0`, 该内核又叫做 齐次线性核(homogeneous linear kernel), 
否则它就是非齐次的(inhomogeneous)。 该内核的定义如下：

.. math::
   k(x_i, x_j) = \sigma_0 ^ 2 + x_i \cdot x_j

:class:`DotProduct` 核通常与指数(exponentiation)相结合。下面是指数取2的一个例子:

.. figure:: ../auto_examples/gaussian_process/images/sphx_glr_plot_gpr_prior_posterior_003.png
   :target: ../auto_examples/gaussian_process/plot_gpr_prior_posterior.html
   :align: center

参考文献
----------

.. [RW2006] Carl Eduard Rasmussen and Christopher K.I. Williams, "Gaussian Processes for Machine Learning", MIT Press 2006, Link to an official complete PDF version of the book `here <http://www.gaussianprocess.org/gpml/chapters/RW.pdf>`_ .

.. currentmodule:: sklearn.gaussian_process
