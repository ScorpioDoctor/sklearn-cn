.. _outlier_detection:

===================================================
新奇点和孤立点检测(Novelty and Outlier Detection)
===================================================

.. currentmodule:: sklearn

很多机器学习应用都需要有能力判断一个新的观测是否跟已有观测具有相同的分布，或者来自不同的分布。
如果来自于相同的分布，则这个新的观测就是一个 *inlier* ; 如果不同，则这个新的观测被称为 *outlier* 。
这种能力通常用于对真实的数据集进行清洗。首先我们要区分两个重要的概念:

:outlier detection:
  训练数据中包含有一些outliers,它们被定义为远离其他观测值的观测值。所以，outliers应该译为“离群点，孤立点”。
  孤立点检测器(Outlier detection estimators) 因此尝试在训练数据最集中的那些区域上进行拟合，
  而忽略那些异常的观测值(deviant observations)。

:novelty detection:
  已有的训练数据并没有被outliers污染，而我们感兴趣的是去检测一个 **新来的** 观测值是否是outlier。
  在这样的一个语境下，此时的outlier我们称之为 novelty。
  在这儿 ，我把 novelty 译为 "新奇点" ,意为 **新来的奇怪的点** 。

孤立点检测 和 新奇点检测 都被用于异常检测(anomaly detection), 所谓anomaly detection就是
检测反常的的观测或不平常的观测。 
孤立点检测 也被称之为 无监督异常检测; 而 新奇点检测 被称之为 半监督异常检测。
在孤立点检测的语境下, outliers/anomalies 不能够形成一个稠密的聚类簇，因为可用的estimators都假定了
outliers/anomalies 位于低密度区域。相反的，在新奇点检测的语境下， novelties/anomalies 是可以形成
稠密聚类簇(dense cluster)的，只要它们在训练数据的一个低密度区域，这被认为是正常的(normal)。

scikit-learn 提供了一系列机器学习工具可以被用于新奇点或孤立点检测。 
这些检测策略被实现成一些 以无监督学习的方式从数据中学习 的估计器类(estimator)::

    estimator.fit(X_train)

fit好了estimator以后，新的观测数据可以用 ``predict`` 方法判断其是 inliers or outliers？::

    estimator.predict(X_test)

Inliers 被标记为 1, 而 outliers 被标记为 -1。 预测方法使用一个阈值在估计器计算出的原始评分函数上。
这个评分函数可以通过方法 ``score_samples`` 进行访问，而且 这个阈值可以由参数 ``contamination`` 控制。

``decision_function`` 方法也是从评分函数定义的，这样的话，得分为负值的就是 outliers, 得分为非负的就是 inliers::

    estimator.decision_function(X_test)

请注意 :class:`neighbors.LocalOutlierFactor` 类默认不支持 ``predict``, ``decision_function`` 和 
``score_samples`` 方法，而只支持 ``fit_predict`` 方法, 因为这个 estimator 一开始就是要把它用到孤立点检测中去的。
训练样本的异常性得分(scores of abnormality)可以通过 ``negative_outlier_factor_`` 属性来访问获取。

如果你真的特别想用 :class:`neighbors.LocalOutlierFactor` 类进行 新奇点检测(novelty detection), 
i.e. 对新的未见过的样本 预测其标签或计算其异常性得分, 你可以在实例化这个estimator的时候将其
``novelty`` 参数设为 ``True`` ，这一步必须要在拟合之前做。这样的话，``fit_predict`` 方法就不可用了。

.. warning:: **使用局部异常因子(Local Outlier Factor,LOF)进行新奇点检测**

  当 ``novelty`` 参数被设为 ``True`` 时，要当心 你必须只能使用 ``predict``, 
  ``decision_function`` 和 ``score_samples`` 在新的未见过的数据上，而不能把这几个函数用在训练数据上，
  因为这样会导致错误的结果。训练样本的异常性得分总是可以通过 ``negative_outlier_factor_`` 属性来访问获取。

:class:`neighbors.LocalOutlierFactor` 类在孤立点检测和新奇点检测中的行为被总结在下面的表格里啦。

===================== ================================ =====================
Method                Outlier detection                Novelty detection
===================== ================================ =====================
``fit_predict``       可用                             不可用
``predict``           不可用                           只能用于新数据
``decision_function`` 不可用                           只能用于新数据
``score_samples``     用 ``negative_outlier_factor_``  只能用于新数据
===================== ================================ =====================



孤立点检测方法一览
=====================================

下面这个案例针对scikit-learn 中的所有孤立点检测算法进行了对比。
局部异常因子(LOF) 没有在图的背景上画出决策边界，因为在孤立点检测中使用LOF时
它没有 predict 方法可以用在新数据上（见上面表格）。

.. figure:: ../auto_examples/images/sphx_glr_plot_anomaly_comparison_001.png
   :target: ../auto_examples/plot_anomaly_comparison.html
   :align: center
   :scale: 50

:class:`ensemble.IsolationForest` 和 :class:`neighbors.LocalOutlierFactor` 
在这里所用的数据集上表现得相当好。 :class:`svm.OneClassSVM` 类对outliers本来就很敏感，
因此在outlier的检测中表现的不是很好。最后, :class:`covariance.EllipticEnvelope` 类
假定了数据是服从高斯分布的且要学习一个椭圆(ellipse)。关于这个对比试验中各种estimators的更多详细信息
请参考 :ref:`sphx_glr_auto_examples_plot_anomaly_comparison.py` 和后续小节。

.. topic:: 案例:

  * 请看 :ref:`sphx_glr_auto_examples_plot_anomaly_comparison.py`
    对比实验，包括了 :class:`svm.OneClassSVM` 类, :class:`ensemble.IsolationForest` 类, 
    :class:`neighbors.LocalOutlierFactor` 类以及 :class:`covariance.EllipticEnvelope` 类。

新奇点检测(Novelty Detection)
================================

我们现在有一个从相同的分布(该分布由 :math:`p` 个特征分量描述)中得到的 :math:`n` 个观测组成的数据集。
现在我们再往这个观测集合中添加一个新的观测(a new observation)。那么这个新加入的观测是否与该集合中旧有的那些观测
非常不一样以至于我们必须怀疑它是否是正常的(regular or not)? (也就是说这个新的观测是否来自于一个相同的分布？)
或者我们从反向考虑，这个新的观测是否与旧有的那些观测特别的相似以至于我们无法将它与原来的旧观测数据区别开来？
这就是 新奇点检测(novelty detection) 中要强调的问题和它要给我们提供的工具的用途。

一般情况下，新奇点检测器 将学习一个粗糙的、封闭的边界来划分初始观测分布的等高线，绘制在嵌入 :math:`p` 维空间中。
然后，如果新的观测在边界分隔的子空间内(within the frontier-delimited subspace)，他们被认为来自相同的群体，
而不是最初的观测。否则，如果这些新观测数据身处边界之外，我们可以说他们是不正常的(abnormal)，
并且根据我们的评估给出一个置信度。

One-Class SVM 已经被 Schölkopf 引入来完成这样一个目的，而且它被实现在 :ref:`svm` 模块中
的 :class:`svm.OneClassSVM` 类中。 使用该类需要选择一个 kernel 和一个标量参数来定义上面提到的边界(frontier)。
RBF kernel 是通常的选项，尽管没有准确的公式或算法来设置RBF kernel的带宽参数(bandwidth)。
这是 scikit-learn 中的默认实现。 参数 :math:`\nu` , 也被称为One-Class SVM的边界(margin), 
对应于在边界外边(outside frontier)找到一个新的但是正常的观测的概率。

.. topic:: 参考文献:

    * `Estimating the support of a high-dimensional distribution
      <http://dl.acm.org/citation.cfm?id=1119749>`_ Schölkopf,
      Bernhard, et al. Neural computation 13.7 (2001): 1443-1471.

.. topic:: 案例:

   * 请看 :ref:`sphx_glr_auto_examples_svm_plot_oneclass.py` 用于可视化 :class:`svm.OneClassSVM` 类对象
     从数据中学习到的边界(frontier)。
   * :ref:`sphx_glr_auto_examples_applications_plot_species_distribution_modeling.py`

.. figure:: ../auto_examples/svm/images/sphx_glr_plot_oneclass_001.png
   :target: ../auto_examples/svm/plot_oneclass.html
   :align: center
   :scale: 75%


孤立点检测(Outlier Detection)
==================================

孤立点检测与新奇点检测类似，其目的是将常规观测的核心(a core of regular observations)
与一些污染性(polluting)观测分离开来，即所谓的离群点或孤立点(Outliers).
然而，在孤立点检测中，我们并没有一个干净的只包含常规观测的数据集用来训练估计器。


拟合椭圆包络
----------------------------

进行孤立点检测的一种常见方法是假定常规数据(regular data)来自于某个已知的分布(比如高斯分布)。
有了这样一个假定以后，我们试图从广义上定义数据的"shape"，并且定义 **外围观测(outlying observations)**
就是那些足够远离"shape"的观测。

scikit-learn 提供了一个对象 :class:`covariance.EllipticEnvelope` 可以在给定的数据上拟合一个鲁棒的协方差估计
, 并且因此拟合一个椭圆(ellipse)到中央数据点(central data points), 忽略那些远离中央的外围点。

比如, 假定 inlier data 是呈高斯分布的, 上述的类对象将会以鲁棒的方法估计inlier location和协方差
(即 此鲁棒估计不会受到数据集中的outliers的影响)。从这一鲁棒估计获得的Mahalanobis距离用来推导一个
衡量外围程度的度量(a measure of outlyingness)。
下面的图和实例说明了这一策略。

.. figure:: ../auto_examples/covariance/images/sphx_glr_plot_mahalanobis_distances_001.png
   :target: ../auto_examples/covariance/plot_mahalanobis_distances.html
   :align: center
   :scale: 75%

.. topic:: 案例:

   * 请看 :ref:`sphx_glr_auto_examples_covariance_plot_mahalanobis_distances.py` 。这个案例
     用于说明 标准的经验协方差估计(:class:`covariance.EmpiricalCovariance`) 与 
     位置和协方差的鲁棒估计(:class:`covariance.MinCovDet`) 之间在评估一个观测的外围性的程度上的区别

.. topic:: 参考文献:

    * Rousseeuw, P.J., Van Driessen, K. "A fast algorithm for the minimum
      covariance determinant estimator" Technometrics 41(3), 212 (1999)

.. _isolation_forest:

Isolation Forest
----------------------------

在高维数据集上进行孤立点检测的一个有效方法是使用随机森林。
:class:`ensemble.IsolationForest` 通过随机选择一个特征，然后在所选特征的最大值和最小值之间随机选择一个分割值来"分离(isolate)"观测集合。

由于递归划分可以用树结构表示，因此分离一个样本(isolate a sample)所需的分割数相当于从根节点到终止节点的路径长度。

这个路径长度(path length)，是这样一个随机树林中的平均值，是正规性(normality)和我们的决策函数的一种度量
(a measure of normality and our decision function)。

随机划分为异常数据(anomalies)产生明显较短的路径。因此，当随机树木的森林集体地对特定样本产生较短的路径长度时，这些特定样本很可能是异常样本。

该策略已在下面的例子中进行了说明。

.. figure:: ../auto_examples/ensemble/images/sphx_glr_plot_isolation_forest_001.png
   :target: ../auto_examples/ensemble/plot_isolation_forest.html
   :align: center
   :scale: 75%

.. topic:: 案例:

   * 请看 :ref:`sphx_glr_auto_examples_ensemble_plot_isolation_forest.py` 。 这个例子是对类 
    :class:`ensemble.IsolationForest` 的用法的示例说明。

   * 请看 :ref:`sphx_glr_auto_examples_plot_anomaly_comparison.py` 
    类 :class:`ensemble.IsolationForest` ，类 :class:`neighbors.LocalOutlierFactor`,
    类 :class:`svm.OneClassSVM` (调整为像孤立点检测方法一样执行) 以及 
    一个基于协方差的孤立点检测和 :class:`covariance.EllipticEnvelope` 类。

.. topic:: 参考文献:

    * Liu, Fei Tony, Ting, Kai Ming and Zhou, Zhi-Hua. "Isolation forest."
      Data Mining, 2008. ICDM'08. Eighth IEEE International Conference on.


局部离群因子
--------------------
在中高维数据集上进行孤立点检测的另一种有效方法是使用局部离群因子(Local Outlier Factor,LOF)算法。

:class:`neighbors.LocalOutlierFactor` (LOF)算法计算反映观测数据异常程度(degree of abnormality)
的分数(称为局部离群因子)。它测量给定数据点相对于其邻域的局部密度偏差。
这样做的目的是检测那些密度比邻居低得多的样本。

在实践中，局部密度(local density)是从k近邻得到的。一个观测的LOF分数等于他的k近邻的平均局部密度和它自己的局部密度的比率：
一个正常的观测数据被期望有一个与它的邻居相似的局部密度，而异常的观测数据被期望有更小的局部密度。

所考虑的邻居数目k(即参数n_neighbors)通常比cluster必须包含的最小对象数目大，这样的话，
其他对象可以是相对于该cluster的本地异常值，此外，所考虑的邻居数目k小于可能是本地异常值的对象的最大关闭数。
在实践中，这类信息通常是不可得的，而取n_neighbors=20 似乎在一般情况下是很好的。
当异常值的比例很高时(如下面的示例所示，大于10%)，则 n_neighbors 应该更大(下面的示例中n_neighbors=35)。

LOF算法的优点在于它考虑了数据集的局部和全局特性：即使在异常样本具有不同的底层密度时，它表现地也很好。
问题不在于样品有多孤立，而在于它与周围的邻居之间有多孤立。

当使用 LOF 进行孤立点检测的时候，不能使用 ``predict``, ``decision_function`` 和 ``score_samples`` 方法，
只能使用 ``fit_predict`` 方法。训练样本的异常性得分可以通过 ``negative_outlier_factor_`` 属性来获得。
注意当使用LOF算法进行新奇点检测的时候(``novelty`` 设为 ``True``)， ``predict``, ``decision_function`` 和 
``score_samples`` 函数可被用于新的未见过的观测数据。请查看 :ref:`novelty_with_lof` 小节的说明。


该策略已在下面的例子中进行了说明。

.. figure:: ../auto_examples/neighbors/images/sphx_glr_plot_lof_outlier_detection_001.png
   :target: ../auto_examples/neighbors/plot_lof_outlier_detection.html
   :align: center
   :scale: 75%

.. topic:: 案例:

   * 请看 :ref:`sphx_glr_auto_examples_neighbors_plot_lof_outlier_detection.py` 。这个例子展示了 
    :class:`neighbors.LocalOutlierFactor` 类的用法的示例说明。

   * 请看 :ref:`sphx_glr_auto_examples_plot_anomaly_comparison.py` 。这个例子将该算法 
    与其他的异常检测(anomaly detection)算法进行对比。

.. topic:: 参考文献:

   *  Breunig, Kriegel, Ng, and Sander (2000)
      `LOF: identifying density-based local outliers.
      <http://www.dbs.ifi.lmu.de/Publikationen/Papers/LOF.pdf>`_
      Proc. ACM SIGMOD

.. _novelty_with_lof:

使用LOF进行新奇点检测
===========================================

如果要用 :class:`neighbors.LocalOutlierFactor` 类进行 新奇点检测, 
i.e. 对新的未见过的样本 预测其标签或计算其异常性得分, 你可以在实例化这个estimator的时候将其
``novelty`` 参数设为 ``True`` ，这一步必须要在拟合之前做::

  lof = LocalOutlierFactor(novelty=True)
  lof.fit(X_train)

请注意 ``fit_predict`` 方法在这情况下就不可用了。

.. warning:: **使用 局部异常因子(LOF) 进行新奇点检测**

  当 ``novelty`` 参数被设为 ``True`` 时，要当心 你必须只能使用 ``predict``, 
  ``decision_function`` 和 ``score_samples`` 在新的未见过的数据上，而不能把这几个函数用在训练数据上，
  因为这样会导致错误的结果。训练样本的异常性得分总是可以通过 ``negative_outlier_factor_`` 属性来访问获取。

使用 局部异常因子(LOF)进行新奇点检测 的示例见下图。

  .. figure:: ../auto_examples/neighbors/images/sphx_glr_plot_lof_novelty_detection_001.png
     :target: ../auto_examples/neighbors/plot_lof_novelty_detection.html
     :align: center
     :scale: 75%

