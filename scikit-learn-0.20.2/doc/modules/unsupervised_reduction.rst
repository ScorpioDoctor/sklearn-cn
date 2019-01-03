
.. _data_reduction:

=======================================================
无监督维数约减(Unsupervised dimensionality reduction)
=======================================================

如果您的features数量很高，那么在有监督的步骤之前使用一个无监督的步骤来减少它可能是有用的。
很多 :ref:`unsupervised-learning` 方法实现一种可用于降维的 ``transform`` 方法。
下面我们将讨论已经被大量使用的这种无监督降维模式的两个具体示例。

.. topic:: **Pipelining**

    非监督数据约简和监督估计器可以链接起来。 请看 :ref:`pipeline`.

.. currentmodule:: sklearn

PCA: 主成分分析
----------------------------------

:class:`decomposition.PCA` 类寻找能够很好地捕捉原始特征方差的一个特征组合。
请看 :ref:`decompositions`.

.. topic:: **案例**

   * :ref:`sphx_glr_auto_examples_applications_plot_face_recognition.py`

随机投影
-------------------

:mod:`random_projection` 模块提供了若干通过随机投影(random projections)用于数据约简的工具。
请查看相关文档的介绍: :ref:`random_projection`。

.. topic:: **案例**

   * :ref:`sphx_glr_auto_examples_plot_johnson_lindenstrauss_bound.py`

特征集聚
------------------------

:class:`cluster.FeatureAgglomeration` 类 应用 :ref:`hierarchical_clustering` 来将相似的特征分组。

.. topic:: **案例**

   * :ref:`sphx_glr_auto_examples_cluster_plot_feature_agglomeration_vs_univariate_selection.py`
   * :ref:`sphx_glr_auto_examples_cluster_plot_digits_agglomeration.py`

.. topic:: **特征尺度变换(Feature scaling)**

   请注意，如果 features 具有非常不同的缩放或统计属性，:class:`cluster.FeatureAgglomeration` 类 将不能够捕捉相关特征之间的联系。
   在这种情况下，使用 :class:`preprocessing.StandardScaler` 类会非常有用。

