.. _mixture:

.. _gmm:

=======================
高斯混合模型(Gaussian mixture models)
=======================

.. currentmodule:: sklearn.mixture

``sklearn.mixture`` 是一个用于学习高斯混合模型的包(package)，
它支持的混合模型类型有 diagonal, spherical, tied 和 full covariance matrices 。
它还提供了对混合分布进行抽样，以及从数据估计混合模型的功能。它还提供了一些工具帮助我们确定分量的合适数量。

 .. figure:: ../auto_examples/mixture/images/sphx_glr_plot_gmm_pdf_001.png
   :target: ../auto_examples/mixture/plot_gmm_pdf.html
   :align: center
   :scale: 50%

   **二元高斯混合模型(Two-component Gaussian mixture model):** *数据点，以及模型的等概率平面(equi-probability surfaces)。*

高斯混合模型是一种概率性模型(probabilistic model)，它假定所有的数据点都是由有限个参数未知的高斯分布的混合产生的。
可以认为混合模型是k均值聚类的推广，它包含了关于数据的协方差结构以及潜在高斯分布的中心的信息。

Scikit-learn 实现不同的类来估计高斯混合模型，这些模型对应于不同的估计策略，详见下文。

高斯混合
================

:class:`GaussianMixture` 对象实现了拟合混合高斯模型的期望最大化(:ref:`expectation-maximization <expectation_maximization>`:EM)算法.
它还可以绘制多元模型的置信椭球，并计算贝叶斯信息准则(BIC)来评估数据中的聚类数。
该类对象提供了从训练数据中学习高斯混合模型的 :meth:`GaussianMixture.fit` 方法。
给定测试数据，可以用 :meth:`GaussianMixture.predict` 方法给每个样本分配它可能属于的高斯分布。

..
    另外的,每个样本属于各个高斯分布的概率可以使用方法 :meth:`GaussianMixture.predict_proba` 进行检索。

:class:`GaussianMixture` 有不同的选项来约束估计出的不同类的协方差：球面(spherical)、对角线（diagonal）、平移(tied)或完全协方差(full covariance)。

.. figure:: ../auto_examples/mixture/images/sphx_glr_plot_gmm_covariances_001.png
   :target: ../auto_examples/mixture/plot_gmm_covariances.html
   :align: center
   :scale: 75%

.. topic:: 案例:

    * See :ref:`sphx_glr_auto_examples_mixture_plot_gmm_covariances.py` for an example of
      using the Gaussian mixture as clustering on the iris dataset.

    * See :ref:`sphx_glr_auto_examples_mixture_plot_gmm_pdf.py` for an example on plotting the
      density estimation.

类 :class:`GaussianMixture` 的优缺点
-----------------------------------------------

优点(Pros)
....

:速度(Speed): 用于学习混合模型的最快速方法

:Agnostic: 由于此算法仅仅最大化似然函数(likelihood), 因此 它不会把均值偏向到0, 或 使聚类大小偏向于可能适用或者可能不适用的特殊结构。

缺点(Cons)
....

:奇异性(Singularities): 当每个混合模型没有足够多的点时，估算协方差变得困难起来，同时算法会发散并且找具有无穷大似然函数值的解，除非人为地对协方差进行正则化。

:分量的数量(Number of components): 这个算法将会总是用所有它能用的分量，所以在没有外部线索的情况下需要留存数据或者用信息理论准则来决定用多少分量。

选择经典高斯混合模型中分量的数量
------------------------------------------------------------------------

一种高效的方法是利用 BIC（贝叶斯信息准则）来选择高斯混合的分量数。 
理论上，它仅当在渐进状态(asymptotic regime )下可以恢复正确的分量数（即如果有大量数据可用，
并且假设这些数据实际上是一个混合高斯模型独立同分布生成的）。
注意：使用 :ref:`Variational Bayesian Gaussian mixture <bgmm>` 可以避免高斯混合模型中分量数的选择。

.. figure:: ../auto_examples/mixture/images/sphx_glr_plot_gmm_selection_001.png
   :target: ../auto_examples/mixture/plot_gmm_selection.html
   :align: center
   :scale: 50%

.. topic:: 案例:

    * 一个用典型的高斯混合进行模型选择的例子，请看 :ref:`sphx_glr_auto_examples_mixture_plot_gmm_selection.py`  。

.. _expectation_maximization:

估计算法:期望极大化(EM)
-----------------------------------------------

在从无标记的数据中应用高斯混合模型主要的困难在于：通常不知道哪个点来自哪个潜在的分量(component) 
（如果可以获取到这些信息，就可以很容易通过相应的数据点，拟合每个独立的高斯分布）。 
期望最大化(`Expectation-maximization <https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm>`_,EM)
是一个理论完善的统计算法，其通过迭代过程来解决这个问题。
首先，假设我们产生了一些随机分量(随机选择数据点的中心做随机分量，数据点的中心可以用k-means算法得到，或者甚至就让在原点周围正态分布的点做随机分量)，
并且为每个点计算由模型的每个分量生成这个点的概率。
然后，调整模型参数以最大化模型生成这些数据点的可能性。重复这个过程就可以保证总会收敛到局部最优解。

.. _bgmm:

变分贝叶斯高斯混合
=====================================

:class:`BayesianGaussianMixture` 对象实现了 具有变分推理算法的 高斯混合模型的 变体。 这个类的API和 :class:`GaussianMixture` 是类似的。


.. _variational_inference:

估计算法: 变分推理
---------------------------------------------

变分推断是期望最大化(EM)的扩展，它最大化模型证据（包括先验）的下界，而不是最大化数据似然函数。 
变分方法的原理与期望最大化相同(二者都是迭代算法，在 寻找每个混合产生每个点的概率 和 根据所分配的点拟合混合模型 之间两步交替)，
但是变分方法通过整合先验分布信息来增加正则化限制。 这避免了期望最大化解决方案中常出现的奇异性，但是也对模型带来了微小的偏差。 
变分方法计算过程通常明显较慢，但通常不会慢到无法使用。

由于它的贝叶斯特性，变分算法比期望最大化(EM)需要更多的超参数，
其中最重要的就是 浓度参数 ``weight_concentration_prior`` 。
指定一个低浓度先验， 将会使模型将大部分的权重放在少数分量上，其余分量的权重则趋近 0。
而高浓度先验将使混合模型中的大部分分量都有一定的权重。 

:class:`BayesianGaussianMixture` 类的参数实现提出了两种先验权重分布： 
一种是利用狄利克雷分布(Dirichlet distribution)的有限混合模型，
另一种是利用狄利克雷过程(Dirichlet Process)的无限混合模型。 
在实际应用上，狄利克雷过程推理算法是近似的，
并且使用具有固定最大分量数的截尾分布(称之为 Stick-breaking representation)。
使用的分量数实际上几乎总是取决于数据。

下图比较了权重浓度先验的不同类型(参数 ``weight_concentration_prior_type``) 对于
不同的权重浓度先验 ``weight_concentration_prior`` 的取值 所获得的结果。 
在这里，我们可以从图中看到 ``weight_concentration_prior`` 参数的值
对获得的有效的激活分量数（即权重较大的分量的数量）有很大影响。 
我们也能注意到当先验是 'dirichlet_distribution' 类型时，
大的浓度权重先验会导致更均匀的权重，然而 'dirichlet_process' 类型（默认类型）却不是这样。

.. |plot_bgmm| image:: ../auto_examples/mixture/images/sphx_glr_plot_concentration_prior_001.png
   :target: ../auto_examples/mixture/plot_concentration_prior.html
   :scale: 48%

.. |plot_dpgmm| image:: ../auto_examples/mixture/images/sphx_glr_plot_concentration_prior_002.png
   :target: ../auto_examples/mixture/plot_concentration_prior.html
   :scale: 48%

.. centered:: |plot_bgmm| |plot_dpgmm|

下面的例子将 分量数目固定的高斯混合模型 与 带有狄利克雷过程先验(Dirichlet process prior)的变分高斯混合模型 进行比较。 
这里，使用5个分量的典型高斯混合模型在由2个聚类组成的数据集上进行拟合。 
我们可以看到，具有狄利克雷过程先验的变分高斯混合模型可以将自身限制在 2 个分量，
而高斯混合必须按照用户事先设置的固定数量的分量来拟合数据。 
在例子中，用户选择了 ``n_components=5`` ，这不符合该数据集的真正的生成分布(generative distribution)。 
注意到 只有非常少量的观测，带有狄利克雷过程先验的变分高斯混合模型可以采取保守的立场，并且只拟合一个分量。

.. figure:: ../auto_examples/mixture/images/sphx_glr_plot_gmm_001.png
   :target: ../auto_examples/mixture/plot_gmm.html
   :align: center
   :scale: 70%


在下图中，我们将拟合一个并不能被高斯混合模型很好描述的数据集。 调整 ``weight_concentration_prior`` ，
:class:`BayesianGaussianMixture` 类的参数控制着用来拟合数据的分量的数目。
我们在最后两个图上展示了从两个混合模型产生的随机抽样。

.. figure:: ../auto_examples/mixture/images/sphx_glr_plot_gmm_sin_001.png
   :target: ../auto_examples/mixture/plot_gmm_sin.html
   :align: center
   :scale: 65%



.. topic:: 案例:

    * See :ref:`sphx_glr_auto_examples_mixture_plot_gmm.py` for an example on
      plotting the confidence ellipsoids for both :class:`GaussianMixture`
      and :class:`BayesianGaussianMixture`.

    * :ref:`sphx_glr_auto_examples_mixture_plot_gmm_sin.py` shows using
      :class:`GaussianMixture` and :class:`BayesianGaussianMixture` to fit a
      sine wave.

    * See :ref:`sphx_glr_auto_examples_mixture_plot_concentration_prior.py`
      for an example plotting the confidence ellipsoids for the
      :class:`BayesianGaussianMixture` with different
      ``weight_concentration_prior_type`` for different values of the parameter
      ``weight_concentration_prior``.


:class:`BayesianGaussianMixture` 的优缺点
----------------------------------------------------------------------------

优点
.....

:自动选择: 当 ``weight_concentration_prior`` 足够小以及 ``n_components`` 比模型实际需要的更大时，
   变分贝叶斯混合模型有一个天然的趋势就是让一些混合权重值趋近 0。 这让模型可以自动选择合适的有效分量数。
   这仅仅需要提供分量的数量上限。但是请注意，“理想” 的激活分量数只在应用场景中比较明确，在数据挖掘参数设置中通常并不明确。

:对参数的数量不敏感: 在有限模型中，总是用尽可以用的分量，因而将为不同数量的components产生不同的解。
   与上述有限模型不同，带有狄利克雷过程先验的变分推理(``weight_concentration_prior_type='dirichlet_process'``)
   在参数变化的时候结果并不会改变太多，这使之更稳定和需要更少的调优。

:正则化: 由于结合了先验信息，变分的解比期望最大化(EM)的解有更少的病理特征(pathological special cases)。


缺点
.....

:速度: 变分推理所需要的额外参数化使推理速度变慢，尽管并没有慢很多。

:超参数: 这个算法需要一个额外的可能需要通过交叉验证进行实验调优的超参数。 

:有偏的: 在推理算法中存在许多隐含的偏差（如果用到狄利克雷过程也会有偏差），每当这些偏差和数据之间不匹配时，用有限模型可能可以拟合更好的模型。


.. _dirichlet_process:

狄利克雷过程(The Dirichlet Process)
---------------------

这里我们描述了狄利克雷过程混合的变分推理算法。狄利克雷过程是在 *具有无限大，无限制的分区数的聚类* 上的先验概率分布。
相比于有限高斯混合模型，变分技术让我们在推理时间几乎没有惩罚（penalty）的情况下把先验结构纳入到高斯混合模型。

一个重要的问题是狄利克雷过程是如何实现用无限的,无限制的聚类数，并且结果仍然是一致的。 
本文档不做出完整的解释，但是你可以看这里 `stick breaking process <https://en.wikipedia.org/wiki/Dirichlet_process#The_stick-breaking_process>`_ 
来帮助你理解它。
折棍(stick breaking)过程是狄利克雷过程的衍生。我们每次从一个单位长度的 stick 开始，
且每一步都折断剩下的一部分。每次，我们把每个 stick 的长度联想成落入一组混合的点的比例。 
最后，为了表示无限混合，我们联想成最后每个 stick 的剩下的部分到没有落入其他组的点的比例。 
每段的长度是随机变量，概率与浓度参数成比例。较小的浓度值将单位长度分成较大的 stick 段（即定义更集中的分布）。
较高的浓度值将生成更小的 stick 段（即增加非零权重的分量数）。

用于狄利克雷过程的变分推理技术在无限混合模型的一个有限近似上仍然可以工作，
但是与 必须要指定一个关于到底要用多少个components的先验不同，只要指定
浓度参数和一个关于混合分量的数量的上界就可以了。这个上界，假定它比真正的分量数目要高点儿，
仅会影响算法复杂度，而不会影响算法实际用到的component的真实数量。
