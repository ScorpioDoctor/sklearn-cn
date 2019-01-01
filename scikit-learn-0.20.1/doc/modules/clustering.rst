.. _clustering:

==========
聚类(Clustering)
==========

:mod:`sklearn.cluster` 模块提供了对无标签数据的聚类算法
(`Clustering <https://en.wikipedia.org/wiki/Cluster_analysis>`__)。

该模块中每一个聚类算法都有两个变体: 一个是类(class)另一个是函数(function)。 
类实现了 ``fit`` 方法来从训练数据中学习聚类；函数接受训练数据返回对应于不同聚类的一个整数标签数组。
对类来说，训练过程得到的标签数据可以在属性  ``labels_`` 中找到。

.. currentmodule:: sklearn.cluster

.. topic:: 输入数据

    需要注意的一点是，在该模块中实现的算法可以以不同类型的矩阵作为输入。
    所有方法都接受标准形状的数据矩阵，shape为 ``[n_samples, n_features]`` 。
    这些内容可以从 :mod:`sklearn.feature_extraction` 模块中的类中获得。
    对于 :class:`AffinityPropagation`, :class:`SpectralClustering`
    和 :class:`DBSCAN` 类，你还可以输入shape为 ``[n_samples, n_samples]`` 的相似矩阵。
    这些可以从 :mod:`sklearn.metrics.pairwise` 模块中的函数中获得。

聚类算法一览
===============================

.. figure:: ../auto_examples/cluster/images/sphx_glr_plot_cluster_comparison_001.png
   :target: ../auto_examples/cluster/plot_cluster_comparison.html
   :align: center
   :scale: 50

   scikit-learn中的聚类算法比较


.. list-table::
   :header-rows: 1
   :widths: 14 15 19 25 20

   * - Method name
     - Parameters
     - Scalability
     - Usecase
     - Geometry (metric used)

   * - :ref:`K-Means <k_means>`
     - number of clusters
     - Very large ``n_samples``, medium ``n_clusters`` with
       :ref:`MiniBatch code <mini_batch_kmeans>`
     - General-purpose, even cluster size, flat geometry, not too many clusters
     - Distances between points

   * - :ref:`Affinity propagation <affinity_propagation>`
     - damping, sample preference
     - Not scalable with n_samples
     - Many clusters, uneven cluster size, non-flat geometry
     - Graph distance (e.g. nearest-neighbor graph)

   * - :ref:`Mean-shift <mean_shift>`
     - bandwidth
     - Not scalable with ``n_samples``
     - Many clusters, uneven cluster size, non-flat geometry
     - Distances between points

   * - :ref:`Spectral clustering <spectral_clustering>`
     - number of clusters
     - Medium ``n_samples``, small ``n_clusters``
     - Few clusters, even cluster size, non-flat geometry
     - Graph distance (e.g. nearest-neighbor graph)

   * - :ref:`Ward hierarchical clustering <hierarchical_clustering>`
     - number of clusters
     - Large ``n_samples`` and ``n_clusters``
     - Many clusters, possibly connectivity constraints
     - Distances between points

   * - :ref:`Agglomerative clustering <hierarchical_clustering>`
     - number of clusters, linkage type, distance
     - Large ``n_samples`` and ``n_clusters``
     - Many clusters, possibly connectivity constraints, non Euclidean
       distances
     - Any pairwise distance

   * - :ref:`DBSCAN <dbscan>`
     - neighborhood size
     - Very large ``n_samples``, medium ``n_clusters``
     - Non-flat geometry, uneven cluster sizes
     - Distances between nearest points

   * - :ref:`Gaussian mixtures <mixture>`
     - many
     - Not scalable
     - Flat geometry, good for density estimation
     - Mahalanobis distances to  centers

   * - :ref:`Birch`
     - branching factor, threshold, optional global clusterer.
     - Large ``n_clusters`` and ``n_samples``
     - Large dataset, outlier removal, data reduction.
     - Euclidean distance between points

非平面几何聚类(Non-flat geometry clustering)在集群具有特定形状时非常有用，
i.e. 一个非平坦的流形，而标准的欧氏距离不是正确的度量.这种情况出现在上图中最上面两行中。

用于聚类的高斯混合模型在 :ref:`专门讨论混合模型的文档 <mixture>`中进行了描述。
KMeans可以看做是高斯混合模型的特例，其中每个分量的协方差相等。

.. _k_means:

K-均值
=======

:class:`KMeans` 算法通过尝试将样本分离成n个方差相等的组来对数据进行聚类，最小化了一个称为惯性(`inertia <inertia>`_)或
聚类内和平方(within-cluster sum-of-squares)的准则。
该算法需要指定簇数。它可以很好地扩展到大量的样本，并且已经在许多不同的应用领域得到了广泛的应用。

k-均值算法将 :math:`N` 个样本的集合 :math:`X` 划分为 :math:`K` 个不相交的簇 :math:`C` ，
每个簇由聚类中样本的均值 :math:`\mu_j` 描述。
这些均值通常被称为簇的“质心(centroids)”；请注意，它们通常不是 :math:`X` 内的样本点，尽管它们处在同一个空间。
K-均值算法的目标是选择那些可以最小化惯性(*inertia*)或最小化簇内平方和准则(within-cluster sum of squared criterion) 的质心 :

.. math:: \sum_{i=0}^{n}\min_{\mu_j \in C}(||x_i - \mu_j||^2)

惯性(*inertia*), 或 簇内平方和准则(within-cluster sum of squared criterion) ,
可以被认为是一种对簇内 内相干(internally coherent) 程度的度量。
它有以下缺点:

- 惯性(*inertia*)假设团簇是凸的和各向同性的，但情况并不总是如此。它在细长的团簇或不规则形状的流形上效果很差。

- 惯性(*inertia*)不是一个规范化的度量：我们只知道较低的值更好，而零是最优的。但是在高维空间中，欧几里得距离往往会膨胀。
  (这是 维数灾难("curse of dimensionality")的一个实例)。
  在k-均值聚类前运行 `PCA <PCA>`_ 等降维算法可以缓解这一问题，加快计算速度。

.. image:: ../auto_examples/cluster/images/sphx_glr_plot_kmeans_assumptions_001.png
   :target: ../auto_examples/cluster/plot_kmeans_assumptions.html
   :align: center
   :scale: 50

K-均值常被称为劳埃德算法(Lloyd's algorithm)。基本上讲，该算法分为三个步骤。第一步选择初始质心，
最基本的方法是从数据集 :math:`k` 中选择k个样本。完成了第一步初始化后，K-means将会在接下来的两步中循环执行：
第二步将每个样本分配到最近的质心。
第三步通过获取分配给每个先前质心的所有样本的平均值来创建新的质心。计算旧质心和新质心之间的差值。
算法重复第二、第三这两个步骤，直到新旧质心之间的差值小于一个阈值为止。
换句话说，它会重复，直到质心没有明显的移动。

.. image:: ../auto_examples/cluster/images/sphx_glr_plot_kmeans_digits_001.png
   :target: ../auto_examples/cluster/plot_kmeans_digits.html
   :align: right
   :scale: 35

K-均值等价于具有小的、完全相等的对角协方差矩阵的期望最大化算法。

K-均值算法还可以通过这个 `Voronoi diagrams <https://en.wikipedia.org/wiki/Voronoi_diagram>`_ 概念进行理解。
首先使用当前质心计算点的 Voronoi 图。Voronoi 图中的每个段(segment)都成为一个单独分开的簇(separate cluster)。
其次，质心被更新为每个段(segment)的平均值。然后，该算法重复此操作，直到满足停止条件。 
通常情况下，当每两次迭代之间的目标函数的相对减小小于给定的容忍值(tolerance value)时，算法停止。
我们的实现与上述过程是有点儿区别的: 当质心移动小于容忍值(tolerance)时，迭代停止。

给定足够的时间，K-means 总是能够收敛的，但这可能是局部最小的。
这很大程度上取决于质心的初始化。因此，通常会进行几次不同的质心初始化的计算。
帮助解决这个问题的一种方法是 k-means++ 初始化方案，它已经在 scikit-learn 中实现（使用 ``init='k-means++'`` 参数）。 
这种方法通常将初始化质心彼此远离，导致比随机初始化更好的结果，如参考文献所示。

The algorithm supports sample weights, which can be given by a parameter
``sample_weight``. This allows to assign more weight to some samples when
computing cluster centers and values of inertia. For example, assigning a
weight of 2 to a sample is equivalent to adding a duplicate of that sample
to the dataset :math:`X`.

为了允许 K-means 并行运行，给出一个参数称为 ``n_jobs`` 。给这个参数一个正值可以使用多个处理器（默认值: 1）。
如果给 -1 则使用所有可用的处理器，-2 使用少一个，等等。
并行化通常以增加内存的代价加速计算（在这种情况下，需要存储多个质心副本，每个job使用一个）。


.. warning::

    当 `numpy` 使用 `Accelerate` 框架时，K-Means 的并行版本在 OS X 上会崩溃。
    这是预期的行为(expected behavior): `Accelerate` 可以在 fork 之后调用，
    但是您需要使用 Python binary（该多进程在 posix 下不执行）来执行子进程。

K-means 可用于矢量量化(vector quantization)。这是使用训练好的 :class:`KMeans` 类的 ``transform`` 变换方法获得的结果。

.. topic:: 案例:

 * :ref:`sphx_glr_auto_examples_cluster_plot_kmeans_assumptions.py`: Demonstrating when
   k-means performs intuitively and when it does not
 * :ref:`sphx_glr_auto_examples_cluster_plot_kmeans_digits.py`: 聚类手写数字

.. topic:: 参考文献:

 * `"k-means++: The advantages of careful seeding"
   <http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf>`_
   Arthur, David, and Sergei Vassilvitskii,
   *Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete
   algorithms*, Society for Industrial and Applied Mathematics (2007)

.. _mini_batch_kmeans:

小批量 K-均值
------------------

:class:`MiniBatchKMeans` 是 :class:`KMeans` 算法的一个变体，它使用 mini-batches 来减少计算时间，同时仍然尝试优化相同的目标函数(objective function)。 
小批量样本(Mini-batches)是输入数据的子集，在每个训练迭代中进行随机抽样。
这些mini-batches大大减少了收敛到局部解所需的计算量。 与其他降低 k-means 收敛时间的算法作对比，mini-batch k-means 产生的结果通常只比标准算法略差。

该算法在两个主要步骤之间进行迭代，类似于 vanilla k-means 。 在第一步，从数据集中随机抽取 :math:`b` 个样本形成一个 mini-batch。
然后将它们分配到最近的质心。 在第二步，更新质心。与 k-means 相反，这是在每个样本的基础上完成的。
对 mini-batch 中的每个样本，通过取该样本的流平均值(streaming average)和分配给该质心的所有先前样本来更新分配给该样本的质心。 
这具有随时间降低质心变化率的效果。执行这些步骤直到达到收敛或达到预定次数的迭代。

:class:`MiniBatchKMeans` 收敛速度比 :class:`KMeans` 更快，但是结果的质量会降低。在实践中，质量差异可能相当小，如下面给的案例和引用的参考。

.. figure:: ../auto_examples/cluster/images/sphx_glr_plot_mini_batch_kmeans_001.png
   :target: ../auto_examples/cluster/plot_mini_batch_kmeans.html
   :align: center
   :scale: 100


.. topic:: 案例:

 * :ref:`sphx_glr_auto_examples_cluster_plot_mini_batch_kmeans.py`:  KMeans 与 MiniBatchKMeans 的对比

 * :ref:`sphx_glr_auto_examples_text_plot_document_clustering.py`: 使用 sparse MiniBatchKMeans 进行文档聚类

 * :ref:`sphx_glr_auto_examples_cluster_plot_dict_face_patches.py`


.. topic:: 参考文献:

 * `"Web Scale K-Means clustering" <http://www.eecs.tufts.edu/~dsculley/papers/fastkmeans.pdf>`_
   D. Sculley, *Proceedings of the 19th international conference on World
   wide web* (2010)

.. _affinity_propagation:

吸引子传播
====================

(译者注：Affinity Propagation Clustering 可翻译为：仿射传播聚类，吸引子传播聚类，相似性传播聚类，亲和力传播聚类，以下简称 AP聚类)

:class:`AffinityPropagation` 聚类方法是通过在样本对之间发送消息直到收敛来创建聚类。
然后使用少量示例样本作为聚类中心来描述数据集， 聚类中心是数据集中最能代表一类数据的样本。
在样本对之间发送的消息表示一个样本作为另一个样本的示例样本的 适合程度(suitability)，适合程度值在根据通信的反馈不断更新。
更新迭代直到收敛，完成聚类中心的选取，因此也给出了最终聚类。

.. figure:: ../auto_examples/cluster/images/sphx_glr_plot_affinity_propagation_001.png
   :target: ../auto_examples/cluster/plot_affinity_propagation.html
   :align: center
   :scale: 50


Affinity Propagation 算法比较有趣的是可以根据提供的数据决定聚类的数目。 因此有两个比较重要的参数:
*preference*: 决定使用多少个示例样本; *damping factor*: 阻尼因子,用于减少吸引信息和归属信息以防止更新减少吸引度和归属度信息时数据振荡。

AP聚类算法主要的缺点是算法的复杂度。 AP聚类算法的时间复杂度是 :math:`O(N^2 T)` , 其中 :math:`N` 是样本的个数 ， 
:math:`T` 是收敛之前迭代的次数。如果使用密集的相似性矩阵空间复杂度是 :math:`O(N^2)` ，如果使用稀疏的相似性矩阵空间复杂度可以降低。 
这使得AP聚类最适合中小型数据集(small to medium sized datasets)。

.. topic:: 案例:

 * :ref:`sphx_glr_auto_examples_cluster_plot_affinity_propagation.py`: Affinity
   Propagation on a synthetic 2D datasets with 3 classes.

 * :ref:`sphx_glr_auto_examples_applications_plot_stock_market.py` Affinity Propagation on
   Financial time series to find groups of companies


**算法描述:**
样本之间传递的信息有两种。 第一种是 responsibility :math:`r(i, k)`, 样本 :math:`k` 适合作为样本 :math:`i` 的典型代表(exemplar)的累计证据。
第二种是 availability :math:`a(i, k)`, 样本 :math:`i` 应该选择样本 :math:`k` 作为它的典型代表(exemplar)的累计证据，
并考虑对所有其他样本来说 :math:`k` 作为exemplar的累计证据。
用这种方式, exemplars被选择是因为这些exemplars满足了两个条件: (1)它们与很多样本足够相似，(2)它们被很多样本选择作为自己的代表。

更正式一点, 一个样本 :math:`k` 要成为样本 :math:`i` 的exemplar的 responsibility 由下式给出:

.. math::

    r(i, k) \leftarrow s(i, k) - max [ a(i, k') + s(i, k') \forall k' \neq k ]

其中 :math:`s(i, k)` 是样本 :math:`k` 和样本 :math:`i` 之间的相似度。

样本 :math:`k` 要成为样本 :math:`i` 的exemplar的 availability 由下式给出:

.. math::

    a(i, k) \leftarrow min [0, r(k, k) + \sum_{i'~s.t.~i' \notin \{i, k\}}{r(i', k)}]

在开始的时候, :math:`r` 和 :math:`a` 中的所有值被设为 0, 并且迭代计算到收敛为止。
为了防止更新messages时出现数据振荡，在迭代过程中引入阻尼因子 :math:`\lambda` :

.. math:: r_{t+1}(i, k) = \lambda\cdot r_{t}(i, k) + (1-\lambda)\cdot r_{t+1}(i, k)
.. math:: a_{t+1}(i, k) = \lambda\cdot a_{t}(i, k) + (1-\lambda)\cdot a_{t+1}(i, k)

其中 :math:`t` 是迭代次数。

.. _mean_shift:

均值漂移
==========
:class:`MeanShift` 聚类算法旨在于发现一个样本密度平滑的 *blobs* 。 均值漂移算法是一种基于质心的算法，
其工作原理是更新质心的候选点，使其成为给定区域内点的均值。 
然后，这些候选者在后处理阶段被过滤以消除近似重复(near-duplicates)，从而形成最终质心集合。

给定第 :math:`t` 次迭代中的候选质心 :math:`x_i` , 候选质心的位置将按照如下公式更新:

.. math::

    x_i^{t+1} = m(x_i^t)

其中 :math:`N(x_i)` 是围绕 :math:`x_i` 的给定距离内的样本邻域， :math:`m` 是为指向点密度最大增加的区域的每一个质心计算的均值漂移向量
(*mean shift* vector)。
该均值漂移向量由下面的公式计算, 通过漂移向量可以高效的把一个质心更新到它的邻域内样本均值所在的地方：

.. math::

    m(x_i) = \frac{\sum_{x_j \in N(x_i)}K(x_j - x_i)x_j}{\sum_{x_j \in N(x_i)}K(x_j - x_i)}

该算法自动设置聚类的数量, 而不是依赖参数 ``bandwidth`` 来决定要搜索的区域的大小。
此参数可以手动设置，但可以使用提供的 ``estimate_bandwidth`` 函数进行估计，如果没有设置带宽，则调用该函数。

该算法不具有很高的可扩展性(not highly scalable)，因为在算法执行过程中需要多个最近邻搜索。
该算法保证收敛，但当质心变化较小时，该算法将停止迭代。

标记一个新的样本是通过为给定的样本找到最近的质心来完成的。


.. figure:: ../auto_examples/cluster/images/sphx_glr_plot_mean_shift_001.png
   :target: ../auto_examples/cluster/plot_mean_shift.html
   :align: center
   :scale: 50


.. topic:: 案例:

 * :ref:`sphx_glr_auto_examples_cluster_plot_mean_shift.py`: 带有3个类的合成2D数据集上的均值漂移聚类。

.. topic:: 参考文献:

 * `"Mean shift: A robust approach toward feature space analysis."
   <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.76.8968&rep=rep1&type=pdf>`_
   D. Comaniciu and P. Meer, *IEEE Transactions on Pattern Analysis and Machine Intelligence* (2002)


.. _spectral_clustering:

谱聚类
===================

:class:`SpectralClustering` 是在样本之间进行affinity matrix的低维度嵌入，后面紧跟一个在低维空间中运行的 KMeans。 
如果affinity matrix是稀疏的并且 `pyamg <https://github.com/pyamg/pyamg>`_  模块已经安装好，则这是非常有效的。 
SpectralClustering 需要指定聚类数。这个算法适用于聚类数少时，在聚类数多是不建议使用。

对于两个聚类，它解决了相似图上正规化割集(`normalised cuts <http://people.eecs.berkeley.edu/~malik/papers/SM-ncut.pdf>`_)问题的一个凸松弛问题：
将图形切割成两个，使得切割的边缘的权重比每个簇内的边缘的权重小。
在图像处理时，这个criteria是特别有趣的: 图的顶点是像素，相似图的边缘是图像梯度的函数。


.. |noisy_img| image:: ../auto_examples/cluster/images/sphx_glr_plot_segmentation_toy_001.png
    :target: ../auto_examples/cluster/plot_segmentation_toy.html
    :scale: 50

.. |segmented_img| image:: ../auto_examples/cluster/images/sphx_glr_plot_segmentation_toy_002.png
    :target: ../auto_examples/cluster/plot_segmentation_toy.html
    :scale: 50

.. centered:: |noisy_img| |segmented_img|

.. warning:: Transforming distance to well-behaved similarities

    请注意，如果你的相似矩阵的值分布不均匀，例如:存在负值或者像是一个距离矩阵而不是相似性矩阵，
    那么 spectral problem 将会变得奇异，并且不能解决。 在这种情况下，建议对矩阵的 entries 进行变换。
    比如在有符号的距离矩阵情况下 通常使用 heat kernel::

        similarity = np.exp(-beta * distance / distance.std())

    请看关于这个应用的例子。

.. topic:: 案例:

 * :ref:`sphx_glr_auto_examples_cluster_plot_segmentation_toy.py`: 利用谱聚类从含噪背景中分割目标。

 * :ref:`sphx_glr_auto_examples_cluster_plot_coin_segmentation.py`: 谱聚类分割在区域中的硬币图像。

.. |coin_kmeans| image:: ../auto_examples/cluster/images/sphx_glr_plot_coin_segmentation_001.png
    :target: ../auto_examples/cluster/plot_coin_segmentation.html
    :scale: 65

.. |coin_discretize| image:: ../auto_examples/cluster/images/sphx_glr_plot_coin_segmentation_002.png
    :target: ../auto_examples/cluster/plot_coin_segmentation.html
    :scale: 65

不同的标签分配策略
-------------------------------------

可以使用不同的标签分配策略, 对应于 :class:`SpectralClustering` 类的 ``assign_labels`` 参数。 
``"kmeans"`` 可以匹配更精细的数据细节，但是可能更加不稳定。 特别是，除非你控置 ``random_state``
否则可能无法复现运行的结果 ，因为它取决于随机初始化。另一方面， 使用 ``"discretize"`` 策略是 100% 可以复现的，
但是它往往会产生相当均匀的几何形状的边缘。

=====================================  =====================================
 ``assign_labels="kmeans"``              ``assign_labels="discretize"``
=====================================  =====================================
|coin_kmeans|                          |coin_discretize|
=====================================  =====================================

谱聚类用于图聚类问题
--------------------------

谱聚类还可以通过谱嵌入对图进行聚类。在这种情况下，affinity matrix 是图的邻接矩阵，SpectralClustering 
由 `affinity='precomputed'` 进行初始化 ::

    >>> from sklearn.cluster import SpectralClustering
    >>> sc = SpectralClustering(3, affinity='precomputed', n_init=100,
    ...                         assign_labels='discretize')
    >>> sc.fit_predict(adjacency_matrix)  # doctest: +SKIP

.. topic:: 参考文献:

 * `"A Tutorial on Spectral Clustering"
   <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.165.9323>`_
   Ulrike von Luxburg, 2007

 * `"Normalized cuts and image segmentation"
   <http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.160.2324>`_
   Jianbo Shi, Jitendra Malik, 2000

 * `"A Random Walks View of Spectral Segmentation"
   <http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.33.1501>`_
   Marina Meila, Jianbo Shi, 2001

 * `"On Spectral Clustering: Analysis and an algorithm"
   <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.19.8100>`_
   Andrew Y. Ng, Michael I. Jordan, Yair Weiss, 2001


.. _hierarchical_clustering:

层次聚类(Hierarchical clustering)
=======================

层次聚类是一种通过不断合并或分割嵌套聚类来构建嵌套聚类的通用聚类算法。
聚类的层次结构被表示为一棵树(或树状图)。树的根是收集所有样本的唯一簇，叶是只有一个样本的簇。
请看 `维基百科的相关词条 <https://en.wikipedia.org/wiki/Hierarchical_clustering>`_ 获得更多信息。

聚合聚类(:class:`AgglomerativeClustering`)对象使用自下而上的方法执行分层聚类：每个observation从自己的簇开始，
簇依次合并在一起。链接准则(linkage criteria)确定用于合并策略的度量:

- **Ward** 最小化所有簇内的平方差总和。这是一种 方差最小化(variance-minimizing) 的方法， 
  在这点上，这是与k-means 的目标函数相似的，但是它用了聚合分层(agglomerative hierarchical)的方法处理。
- **Maximum** or **complete linkage** 最小化每两个簇的样本之间的最大距离。
- **Average linkage** 最小化每两个簇的样本之间的平均距离。
- **Single linkage** 最小化每两个簇中最近的样本之间的距离。

:class:`AgglomerativeClustering` 在与连接矩阵(connectivity matrix)联合使用时，也可以扩大到大量的样本，但是 在样本之间没有添加连接约束时，
计算代价很大:每一个步骤都要考虑所有可能的合并。

.. topic:: :class:`FeatureAgglomeration`

   :class:`FeatureAgglomeration` 类使用 agglomerative clustering 将看上去相似的特征组合在一起，
   从而减少特征的数量。这是一个降维工具, 请看 :ref:`data_reduction`。


Different linkage type: Ward, complete, average, and single linkage
-------------------------------------------------------------------

:class:`AgglomerativeClustering` 支持 Ward, single, average, 和 complete linkage 策略。

.. image:: ../auto_examples/cluster/images/sphx_glr_plot_linkage_comparison_001.png
    :target: ../auto_examples/cluster/plot_linkage_comparison.html
    :scale: 43

聚合聚类存在 “rich get richer” 现象导致聚类大小不均匀(uneven cluster sizes)。这方面 single linkage 是最坏的策略，Ward 给出了最规则的大小。
然而，在 Ward 中 affinity (or distance used in clustering) 不能被改变，对于 non Euclidean metrics 来说 average linkage 是一个好的选择。
Single linkage,虽然对噪声数据没有鲁棒性，但可以非常有效地进行计算，因此对于提供更大数据集的分层聚类非常有用。
Single linkage 也可以很好地表现在非球形(non-globular)数据上。

.. topic:: 案例:

 * :ref:`sphx_glr_auto_examples_cluster_plot_digits_linkage.py`: 在一个真实的数据集中探索不同的linkage策略。


添加连通性约束
-------------------------------

:class:`AgglomerativeClustering` 类中一个有趣的特点是可以使用连接矩阵(connectivity matrix)将连接约束添加到算法中(只有相邻的聚类可以合并到一起)，
连接矩阵为每一个样本给定了相邻的样本。 例如，在下面的瑞典卷卷(swiss-roll) 的例子中，连接约束禁止在不相邻的 swiss roll 上合并，
从而防止形成在 roll 上 重复折叠的聚类。

.. |unstructured| image:: ../auto_examples/cluster/images/sphx_glr_plot_ward_structured_vs_unstructured_001.png
        :target: ../auto_examples/cluster/plot_ward_structured_vs_unstructured.html
        :scale: 49

.. |structured| image:: ../auto_examples/cluster/images/sphx_glr_plot_ward_structured_vs_unstructured_002.png
        :target: ../auto_examples/cluster/plot_ward_structured_vs_unstructured.html
        :scale: 49

.. centered:: |unstructured| |structured|

这些约束对于强加一定的局部结构是很有用的，但是这也使得算法更快，特别是当样本数量巨大时。

连通性的限制是通过连接矩阵(connectivity matrix)来实现的:一个 scipy sparse matrix(仅在一行和一列的交集处具有应该连接在一起的数据集的索引)。
这个矩阵可以通过先验信息构建:例如，你可能通过仅仅将从一个连接指向另一个的链接合并页面来聚类页面。也可以从数据中学习到,
例如使用 :func:`sklearn.neighbors.kneighbors_graph` 限制与最近邻的合并，
就像 :ref:`这个例子 <sphx_glr_auto_examples_cluster_plot_agglomerative_clustering.py>` 
中那样, 或者使用 :func:`sklearn.feature_extraction.image.grid_to_graph` 仅合并图像上相邻的像素点，
就像 :ref:`这个例子 <sphx_glr_auto_examples_cluster_plot_coin_ward_segmentation.py>` 。

.. topic:: 案例:

 * :ref:`sphx_glr_auto_examples_cluster_plot_coin_ward_segmentation.py`: 使用 Ward 聚类 分割硬币图像。

 * :ref:`sphx_glr_auto_examples_cluster_plot_ward_structured_vs_unstructured.py`: 瑞士卷上的Ward算法示例，结构化方法与非结构化方法的比较。

 * :ref:`sphx_glr_auto_examples_cluster_plot_feature_agglomeration_vs_univariate_selection.py`:
   基于Ward层次聚类的特征聚类降维实例。

 * :ref:`sphx_glr_auto_examples_cluster_plot_agglomerative_clustering.py`

.. warning:: **Connectivity constraints with single, average and complete linkage**

    连接约束 和 complete or average linkage 可以增强聚合聚类中的 ‘rich getting richer’ 现象。
    特别是，当它们使用函数 :func:`sklearn.neighbors.kneighbors_graph` 进行构建时。 在少量聚类的限制下， 
    更倾向于给出一些 macroscopically occupied clusters 
    并且几乎是空的 (讨论内容请查看 :ref:`sphx_glr_auto_examples_cluster_plot_agglomerative_clustering.py`)。
    在这个问题上，Single linkage 是最脆弱的 linkage 选项。

.. image:: ../auto_examples/cluster/images/sphx_glr_plot_agglomerative_clustering_001.png
    :target: ../auto_examples/cluster/plot_agglomerative_clustering.html
    :scale: 38

.. image:: ../auto_examples/cluster/images/sphx_glr_plot_agglomerative_clustering_002.png
    :target: ../auto_examples/cluster/plot_agglomerative_clustering.html
    :scale: 38

.. image:: ../auto_examples/cluster/images/sphx_glr_plot_agglomerative_clustering_003.png
    :target: ../auto_examples/cluster/plot_agglomerative_clustering.html
    :scale: 38

.. image:: ../auto_examples/cluster/images/sphx_glr_plot_agglomerative_clustering_004.png
    :target: ../auto_examples/cluster/plot_agglomerative_clustering.html
    :scale: 38


改变聚类测度
-------------------

Single, average 和 complete linkage 可以使用各种距离 (or affinities), 特别是 欧氏距离(l2), 
曼哈顿距离(Manhattan distance)(or 城市区块距离(Cityblock), or l1), 余弦距离(cosine distance),
或者 任何预先计算的亲和度(affinity)矩阵。

* *l1* distance 有利于稀疏特征或者稀疏噪声: 例如很多特征都是0，在文本挖掘中统计稀有词汇的出现就会出现这种情况。

* *cosine* distance 非常有趣因为它对信号的全局放缩具有不变性。

选择度量的准则是使用一个准则，使不同类中的样本之间的距离最大化，并使每个类内的距离最小化。

.. image:: ../auto_examples/cluster/images/sphx_glr_plot_agglomerative_clustering_metrics_005.png
    :target: ../auto_examples/cluster/plot_agglomerative_clustering_metrics.html
    :scale: 32

.. image:: ../auto_examples/cluster/images/sphx_glr_plot_agglomerative_clustering_metrics_006.png
    :target: ../auto_examples/cluster/plot_agglomerative_clustering_metrics.html
    :scale: 32

.. image:: ../auto_examples/cluster/images/sphx_glr_plot_agglomerative_clustering_metrics_007.png
    :target: ../auto_examples/cluster/plot_agglomerative_clustering_metrics.html
    :scale: 32

.. topic:: 案例:

 * :ref:`sphx_glr_auto_examples_cluster_plot_agglomerative_clustering_metrics.py`


.. _dbscan:

DBSCAN
======

:class:`DBSCAN` 算法将聚类视为被低密度区域分隔的高密度区域。由于这个相当普遍的观点， 
DBSCAN发现的聚类可以是任何形状的，与假设聚类是 convex shaped 的 K-means 相反。 
DBSCAN 的核心概念是 *core samples* , 是指位于高密度区域的样本。 因此一个聚类是一组核心样本，
每个核心样本彼此靠近（通过一定距离度量测量） 和一组接近核心样本的非核心样本（但本身不是核心样本）。
算法中的两个参数, ``min_samples``  和 ``eps`` ,正式的定义了我们所说的 *dense*（稠密）。
较高的 ``min_samples`` 或者较低的 ``eps`` 表示形成聚类需要较高的密度。

更正式的,我们定义核心样本(core sample)是指数据集中的一个样本，在 ``eps`` 距离范围内存在 ``min_samples`` 个其他样本，
这些样本被定义为核心样本的邻居(*neighbors*) 。 这告诉我们核心样本在向量空间的稠密区域。 一个聚类是一个核心样本的集合，
可以通过递归来构建，选取一个核心样本，寻找它所有的neighbors中的核心样本，然后寻找新获取的核心样本(*their*)的 neighbors中的核心样本，
递归这个过程。 聚类中还具有一组非核心样本，它们是聚类中核心样本的邻居的样本，但本身并不是核心样本。 显然，这些样本位于聚类的边缘。

根据定义，任何核心样本都是聚类的一部分，任何不是核心样本并且和任意一个核心样本距离都至少大于 ``eps`` 的样本被认为是 outliers。

在下图中，颜色表示聚类成员，大圆圈表示算法发现的核心样本。 较小的圆圈是仍然是聚类的一部分的非核心样本。 此外，异常值(outliers)由下面的黑点表示。

.. |dbscan_results| image:: ../auto_examples/cluster/images/sphx_glr_plot_dbscan_001.png
        :target: ../auto_examples/cluster/plot_dbscan.html
        :scale: 50

.. centered:: |dbscan_results|

.. topic:: 案例:

    * :ref:`sphx_glr_auto_examples_cluster_plot_dbscan.py`

.. topic:: 实现

    DBSCAN 算法是具有确定性的，当以相同的顺序给出相同的数据时总是形成相同的聚类。 
    然而，当以不同的顺序提供数据时聚类的结果可能不相同。首先，即使核心样本总是被分配给相同的簇，
    这些簇的标签将取决于数据中遇到这些样本的顺序。第二个更重要的是，给非核心样本分派的聚簇可能因数据顺序而有所不同。 
    当一个非核心样本距离两个核心样本的距离都小于  ``eps`` 时，就会发生这种情况。 
    通过三角不等式可知，这两个核心样本距离一定大于  ``eps`` 或者处于同一个聚类中。 
    非核心样本将被非配到首先查找到改样本的类别，因此结果将取决于数据的顺序。
    
    当前版本使用 ball trees 和 kd-trees 来确定样本点的邻域，这样避免了计算全部的距离矩阵 （0.14 之前的 scikit-learn 版本计算全部的距离矩阵）。
    保留使用自定义指标(custom metrics)的可能性。细节请参照 :class:`NearestNeighbors` 。

.. topic:: 大样本量的内存消耗

    默认的实现方式并不是内存高效的，因为它在不能使用 kd-trees 或者 ball-trees 的情况下构建一个完整的两两相似度矩阵(pairwise similarity matrix)。
    这个矩阵将消耗 n^2 个浮点数。 解决这个问题的几种机制:

    - 稀疏半径邻域图(A sparse radius neighborhood graph)(其中缺少条目被假定为距离超出 eps) 可以以内存高效的方式预先计算，
      并且dbscan(参数设置为 ``metric='precomputed'``)可以在这个图上运行。
      请看 :meth:`sklearn.neighbors.NearestNeighbors.radius_neighbors_graph` 。

    - 通过删除发生在数据中的准确副本或着使用 BIRCH 方法对数据集进行压缩。
      然后，你就可以用一个数量相对少的样本集合来代表大量的样本点。再然后，你还可以在拟合DBSCAN的时候提供 ``sample_weight``。

.. topic:: 参考文献:

 * "A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases
   with Noise"
   Ester, M., H. P. Kriegel, J. Sander, and X. Xu,
   In Proceedings of the 2nd International Conference on Knowledge Discovery
   and Data Mining, Portland, OR, AAAI Press, pp. 226–231. 1996

.. _birch:

Birch
=====

The :class:`Birch` builds a tree called the Characteristic Feature Tree (CFT)
for the given data. The data is essentially lossy compressed to a set of
Characteristic Feature nodes (CF Nodes). The CF Nodes have a number of
subclusters called Characteristic Feature subclusters (CF Subclusters)
and these CF Subclusters located in the non-terminal CF Nodes
can have CF Nodes as children.

The CF Subclusters hold the necessary information for clustering which prevents
the need to hold the entire input data in memory. This information includes:

- Number of samples in a subcluster.
- Linear Sum - A n-dimensional vector holding the sum of all samples
- Squared Sum - Sum of the squared L2 norm of all samples.
- Centroids - To avoid recalculation linear sum / n_samples.
- Squared norm of the centroids.

The Birch algorithm has two parameters, the threshold and the branching factor.
The branching factor limits the number of subclusters in a node and the
threshold limits the distance between the entering sample and the existing
subclusters.

This algorithm can be viewed as an instance or data reduction method,
since it reduces the input data to a set of subclusters which are obtained directly
from the leaves of the CFT. This reduced data can be further processed by feeding
it into a global clusterer. This global clusterer can be set by ``n_clusters``.
If ``n_clusters`` is set to None, the subclusters from the leaves are directly
read off, otherwise a global clustering step labels these subclusters into global
clusters (labels) and the samples are mapped to the global label of the nearest subcluster.

**Algorithm description:**

- A new sample is inserted into the root of the CF Tree which is a CF Node.
  It is then merged with the subcluster of the root, that has the smallest
  radius after merging, constrained by the threshold and branching factor conditions.
  If the subcluster has any child node, then this is done repeatedly till it reaches
  a leaf. After finding the nearest subcluster in the leaf, the properties of this
  subcluster and the parent subclusters are recursively updated.

- If the radius of the subcluster obtained by merging the new sample and the
  nearest subcluster is greater than the square of the threshold and if the
  number of subclusters is greater than the branching factor, then a space is temporarily
  allocated to this new sample. The two farthest subclusters are taken and
  the subclusters are divided into two groups on the basis of the distance
  between these subclusters.

- If this split node has a parent subcluster and there is room
  for a new subcluster, then the parent is split into two. If there is no room,
  then this node is again split into two and the process is continued
  recursively, till it reaches the root.

**Birch or MiniBatchKMeans?**

 - Birch does not scale very well to high dimensional data. As a rule of thumb if
   ``n_features`` is greater than twenty, it is generally better to use MiniBatchKMeans.
 - If the number of instances of data needs to be reduced, or if one wants a
   large number of subclusters either as a preprocessing step or otherwise,
   Birch is more useful than MiniBatchKMeans.


**如何使用partial_fit?**

To avoid the computation of global clustering, for every call of ``partial_fit``
the user is advised

 1. To set ``n_clusters=None`` initially
 2. Train all data by multiple calls to partial_fit.
 3. Set ``n_clusters`` to a required value using
    ``brc.set_params(n_clusters=n_clusters)``.
 4. Call ``partial_fit`` finally with no arguments, i.e. ``brc.partial_fit()``
    which performs the global clustering.

.. image:: ../auto_examples/cluster/images/sphx_glr_plot_birch_vs_minibatchkmeans_001.png
    :target: ../auto_examples/cluster/plot_birch_vs_minibatchkmeans.html

.. topic:: 参考文献:

 * Tian Zhang, Raghu Ramakrishnan, Maron Livny
   BIRCH: An efficient data clustering method for large databases.
   http://www.cs.sfu.ca/CourseCentral/459/han/papers/zhang96.pdf

 * Roberto Perdisci
   JBirch - Java implementation of BIRCH clustering algorithm
   https://code.google.com/archive/p/jbirch


.. _clustering_evaluation:

聚类算法性能评估
=================================

评估聚类算法的性能并不像计算错误数或监督分类算法的精确度和召回率那样简单。
特别是，任何评估指标都不应该考虑聚类标签的绝对值。
In particular any evaluation metric should not
take the absolute values of the cluster labels into account but rather
if this clustering define separations of the data similar to some ground
truth set of classes or satisfying some assumption such that members
belong to the same class are more similar that members of different
classes according to some similarity metric.
(译者注：这句英文简直逆天了，看了五分钟都没看出咋断句，不译了,o(∩_∩)o 哈哈)。

.. currentmodule:: sklearn.metrics

.. _adjusted_rand_score:

Adjusted Rand index
-------------------

给定真实的类分配(ground truth class assignments): ``labels_true`` 和 我们的聚类算法对同样的样本集预测出的类分配：``labels_pred``, 
**adjusted Rand index** 是一个用来度量上述两种分配的相似度(**similarity**)的函数，ignoring permutations 和 **with chance normalization**::

  >>> from sklearn import metrics
  >>> labels_true = [0, 0, 0, 1, 1, 1]
  >>> labels_pred = [0, 0, 1, 1, 2, 2]

  >>> metrics.adjusted_rand_score(labels_true, labels_pred)  # doctest: +ELLIPSIS
  0.24...

可以在预测的标签中排列(permute) 0 和 1，重命名为 2 到 3， 得到相同的分数 ::

  >>> labels_pred = [1, 1, 0, 0, 3, 3]
  >>> metrics.adjusted_rand_score(labels_true, labels_pred)  # doctest: +ELLIPSIS
  0.24...

更进一步, 函数 :func:`adjusted_rand_score` 是 **对称的(symmetric)**: 交换参数(argument)不会改变得分(score)。
它可以作为 **共识度量(consensus measure)** ::

  >>> metrics.adjusted_rand_score(labels_pred, labels_true)  # doctest: +ELLIPSIS
  0.24...

完美的标记(perfect labeling)得分为 1.0 ::

  >>> labels_pred = labels_true[:]
  >>> metrics.adjusted_rand_score(labels_true, labels_pred)
  1.0

坏的标记 (e.g. independent labelings) 具有负值或接近 0.0 的得分::

  >>> labels_true = [0, 1, 2, 0, 3, 4, 5, 1]
  >>> labels_pred = [1, 1, 0, 0, 2, 2, 2, 2]
  >>> metrics.adjusted_rand_score(labels_true, labels_pred)  # doctest: +ELLIPSIS
  -0.12...


优点
~~~~~~~~~~

- **随机(均匀)标签分配的 ARI 得分接近于 0.0**
  对于 ``n_clusters`` 和 ``n_samples`` 的任何值（这不是 raw Rand index 或者 V-measure 的情况）。

- **得分被界定在 [-1, 1] 的区间内**: 负值是坏的(独立性标签), 相似的聚类有一个正的 ARI， 1.0 是完美的匹配得分。

- **没有对簇的结构做任何假定**:  可以用于比较聚类算法，比如 假定了各向同性的blob shapes的k-means方法的结果 和 寻找具有
  "folded"形状的谱聚类方法的结果进行比较。


缺点
~~~~~~~~~

- 与惯性(inertia)方法不同, **ARI 需要 ground truth classes 的相关知识** 
  而在实践中几乎不可得到，或者需要人工标注者手动分配（如在监督学习环境中）。

  然而，ARI 还可以在纯粹无监督的设置中作为可用于 聚类模型选择 的共识索引的构建模块。


.. topic:: 案例:

 * :ref:`sphx_glr_auto_examples_cluster_plot_adjusted_for_chance_measures.py`: 分析数据集大小对随机分配聚类度量值的影响。


数学表达式
~~~~~~~~~~~~~~~~~~~~~~~~

如果 C 是 ground truth class assignment 以及 K 是聚类算法给出的class assignment, 让我们定义 :math:`a` 和 :math:`b` 如下:

- :math:`a`, 在C中相同集合和在K中相同集合的元素对的数量(the number of pairs of elements that are in the same set
  in C and in the same set in K)

- :math:`b`, 在C中不同集合和在K中不同集合的元素对的数量(the number of pairs of elements that are in different sets
  in C and in different sets in K)

那么，原始的 (未调整的,unadjusted) Rand index 由下式给出:

.. math:: \text{RI} = \frac{a + b}{C_2^{n_{samples}}}

其中 :math:`C_2^{n_{samples}}` 是在(未排序的)数据集中所有可能的元素对的总数量。

然而，RI 评分不能保证随机标签分配(random label assignments)将获得接近零的值（特别是如果簇的数量与样本数量有相同的数量级）。

为了抵消这种影响，我们可以通过定义调整后的 Rand index(adjusted Rand index,即ARI) 来 
对随机标签分配的 期望 RI :math:`E[\text{RI}]` 打折(discount), 如下所示:

.. math:: \text{ARI} = \frac{\text{RI} - E[\text{RI}]}{\max(\text{RI}) - E[\text{RI}]}

.. topic:: 参考文献

 * `Comparing Partitions
   <http://link.springer.com/article/10.1007%2FBF01908075>`_
   L. Hubert and P. Arabie, Journal of Classification 1985

 * `Wikipedia entry for the adjusted Rand index
   <https://en.wikipedia.org/wiki/Rand_index#Adjusted_Rand_index>`_

.. _mutual_info_score:

基于互信息的得分
-------------------------------

给定真实的类分配(class assignments): ``labels_true`` 和 我们的聚类算法对同样的样本集预测出的类分配：``labels_pred``, 
**互信息(Mutual Information)** 是一个函数，用于度量两个分配集合的一致性，忽略了排列组合(the **agreement** of the two
assignments, ignoring permutations)。
这种度量方法的两个不同的归一化版本目前可用: **Normalized Mutual Information (NMI)** 和 **Adjusted Mutual Information (AMI)**。 
NMI 在文献中可以经常看到, 而 AMI 最近才被提出 and is **normalized against chance**::

  >>> from sklearn import metrics
  >>> labels_true = [0, 0, 0, 1, 1, 1]
  >>> labels_pred = [0, 0, 1, 1, 2, 2]

  >>> metrics.adjusted_mutual_info_score(labels_true, labels_pred)  # doctest: +SKIP
  0.22504...

可以在预测出的标签(predicted labels)中排列 0 和 1, 重命名为 2 到 3， 并得到相同的得分 ::

  >>> labels_pred = [1, 1, 0, 0, 3, 3]
  >>> metrics.adjusted_mutual_info_score(labels_true, labels_pred)  # doctest: +SKIP
  0.22504...

所有的函数, :func:`mutual_info_score`, :func:`adjusted_mutual_info_score` 和 :func:`normalized_mutual_info_score` 
都是对称的(symmetric): 交换函数的参数不会改变得分。 因此它们可以用作 **consensus measure**::

  >>> metrics.adjusted_mutual_info_score(labels_pred, labels_true)  # doctest: +SKIP
  0.22504...

完美标签分配(Perfect labeling)的得分是 1.0::

  >>> labels_pred = labels_true[:]
  >>> metrics.adjusted_mutual_info_score(labels_true, labels_pred)  # doctest: +SKIP
  1.0

  >>> metrics.normalized_mutual_info_score(labels_true, labels_pred)  # doctest: +SKIP
  1.0

这对于 ``mutual_info_score`` 是不成立的, 因此该得分更难于判断::

  >>> metrics.mutual_info_score(labels_true, labels_pred)  # doctest: +SKIP
  0.69...

坏的标签分配 (e.g. independent labelings) 具有负的得分(non-positive scores)::

  >>> labels_true = [0, 1, 2, 0, 3, 4, 5, 1]
  >>> labels_pred = [1, 1, 0, 0, 2, 2, 2, 2]
  >>> metrics.adjusted_mutual_info_score(labels_true, labels_pred)  # doctest: +SKIP
  -0.10526...


优点
~~~~~~~~~~

- **随机(均匀)标签分配有一个接近于0的 AMI得分。**
  对于 ``n_clusters`` 和 ``n_samples`` 的任何值（这不是 raw Mutual Information 或者 V-measure 的情况）。

- **上界为 1** :  得分值接近于 0 表明两个标签分配集合很大程度上是独立的(largely independent), 而得分值接近于 1 表明两个标签分配集合
  具有很大的一致性(significant agreement)。 更进一步, 正好是1的AMI表示两个标签分配相等。 (with or without permutation).


缺点
~~~~~~~~~

- 与惯性(inertia)方法不同, **基于互信息的度量(MI-based measures) 需要 ground truth classes 的相关知识** 
  而在实践中几乎不可得到，或者需要人工标注者手动分配（如在监督学习环境中）。

  然而，MI-based measures 还可以在纯粹无监督的设置中作为可用于 聚类模型选择 的共识索引的构建模块。

- NMI and MI are not adjusted against chance.


.. topic:: Examples:

 * :ref:`sphx_glr_auto_examples_cluster_plot_adjusted_for_chance_measures.py`: 
  分析数据集大小对随机分配聚类度量值的影响。 此示例还包括 Adjusted Rand Index。


数学表达式
~~~~~~~~~~~~~~~~~~~~~~~~

假定我们有两个标签分配集合(of the same N objects), :math:`U` 和 :math:`V`.
它们的熵(entropy)是划分集(partition set)的不确定性量(the amount of uncertainty), 定义如下:

.. math:: H(U) = - \sum_{i=1}^{|U|}P(i)\log(P(i))

其中 :math:`P(i) = |U_i| / N` 是从 :math:`U` 集合中随机挑选的一个object落到 :math:`U_i` 集合中的概率。 
对于 :math:`V` 集合也是一样的:

.. math:: H(V) = - \sum_{j=1}^{|V|}P'(j)\log(P'(j))

其中 :math:`P'(j) = |V_j| / N`。 :math:`U` 和 :math:`V` 之间的互信息(mutual information)的计算公式如下 :

.. math:: \text{MI}(U, V) = \sum_{i=1}^{|U|}\sum_{j=1}^{|V|}P(i, j)\log\left(\frac{P(i,j)}{P(i)P'(j)}\right)

其中 :math:`P(i, j) = |U_i \cap V_j| / N` 是随机选择的object落到这两类集合 :math:`U_i` 和 :math:`V_j` 中的概率。

互信息还可以用set cardinality的形式来表述 :

.. math:: \text{MI}(U, V) = \sum_{i=1}^{|U|} \sum_{j=1}^{|V|} \frac{|U_i \cap V_j|}{N}\log\left(\frac{N|U_i \cap V_j|}{|U_i||V_j|}\right)

归一化的互信息定义如下:

.. math:: \text{NMI}(U, V) = \frac{\text{MI}(U, V)}{\text{mean}(H(U), H(V))}

不管两个标签分配集合(label assignments)之间的互信息的实际量有多大，互信息的值包括归一化互信息的值没有针对偶然性进行调整(not adjusted for chance) 
而且 倾向于随着不同标签(聚类)的数量的增加而增加。

互信息的期望值可以用等式 [VEB2009]_ 计算。在这个等式中, :math:`a_i = |U_i|` (:math:`U_i` 集合中的元素数量) 和
:math:`b_j = |V_j|` ( :math:`V_j` 集合中的元素数量)。


.. math:: E[\text{MI}(U,V)]=\sum_{i=1}^{|U|} \sum_{j=1}^{|V|} \sum_{n_{ij}=(a_i+b_j-N)^+
   }^{\min(a_i, b_j)} \frac{n_{ij}}{N}\log \left( \frac{ N.n_{ij}}{a_i b_j}\right)
   \frac{a_i!b_j!(N-a_i)!(N-b_j)!}{N!n_{ij}!(a_i-n_{ij})!(b_j-n_{ij})!
   (N-a_i-b_j+n_{ij})!}

使用了互信息期望值后, 经过调整的互信息的计算将使用与 ARI(adjusted Rand index) 类似的形式进行 :

.. math:: \text{AMI} = \frac{\text{MI} - E[\text{MI}]}{\text{mean}(H(U), H(V)) - E[\text{MI}]}

对于归一化互信息和调整后的互信息，归一化值通常是每个聚类的熵的一些广义均值(*generalized* mean)。
有各种广义均值存在，并没有明确的规则说某一个优先于其他的。这个决定很大程度上是取决于各个领域的基础；
例如，在社区检测(community detection)中，算术平均值是最常见的。每一种归一化方法提供 "qualitatively similar behaviours" [YAT2016]_。
在我们的实现中, 这是通过参数 ``average_method`` 进行控制的。

Vinh et al. (2010) 对各种 NMI 和 AMI 的变体 用它们使用的平均方法(averaging method) 进行了命名 [VEB2010]_ 。 
他们在论文里说的 'sqrt' 和 'sum' 平均 分别是 几何 和 算数 平均；我们使用这些更广泛的通用名称。

.. topic:: 参考文献

 * Strehl, Alexander, and Joydeep Ghosh (2002). "Cluster ensembles – a
   knowledge reuse framework for combining multiple partitions". Journal of
   Machine Learning Research 3: 583–617.
   `doi:10.1162/153244303321897735 <http://strehl.com/download/strehl-jmlr02.pdf>`_.

 * [VEB2009] Vinh, Epps, and Bailey, (2009). "Information theoretic measures
   for clusterings comparison". Proceedings of the 26th Annual International
   Conference on Machine Learning - ICML '09.
   `doi:10.1145/1553374.1553511 <https://dl.acm.org/citation.cfm?doid=1553374.1553511>`_.
   ISBN 9781605585161.

 * [VEB2010] Vinh, Epps, and Bailey, (2010). "Information Theoretic Measures for
   Clusterings Comparison: Variants, Properties, Normalization and
   Correction for Chance". JMLR
   <http://jmlr.csail.mit.edu/papers/volume11/vinh10a/vinh10a.pdf>

 * `Wikipedia entry for the (normalized) Mutual Information
   <https://en.wikipedia.org/wiki/Mutual_Information>`_

 * `Wikipedia entry for the Adjusted Mutual Information
   <https://en.wikipedia.org/wiki/Adjusted_Mutual_Information>`_
   
 * [YAT2016] Yang, Algesheimer, and Tessone, (2016). "A comparative analysis of
   community
   detection algorithms on artificial networks". Scientific Reports 6: 30750.
   `doi:10.1038/srep30750 <https://www.nature.com/articles/srep30750>`_.
   
   

.. _homogeneity_completeness:

同质性, 完备性 与 V-测度
---------------------------------------
(译者注：同质性(Homogeneity)、完备性(completeness)与 V-测度(V-measure))

给定样本的真实类分配(ground truth class assignments)的相关知识, 则使用条件熵分析(conditional entropy analysis)来定义某个直观的指标(metric)是可能的。

特别是，Rosenberg和Hirschberg(2007)为任意聚类分配定义了以下两个理想的目标(desirable objectives):

- **同质性(Homogeneity)**: 每个聚类(簇)里面只包含单个类的样本。

- **完备性(completeness)**: 一个给定类的所有样本都被分到了同一个聚类(簇)中。

我们将上述概念转变为函数 :func:`homogeneity_score` 和 :func:`completeness_score` 。 这两个函数的返回值都是介于0到1之间，(返回值越大越好)::

  >>> from sklearn import metrics
  >>> labels_true = [0, 0, 0, 1, 1, 1]
  >>> labels_pred = [0, 0, 1, 1, 2, 2]

  >>> metrics.homogeneity_score(labels_true, labels_pred)  # doctest: +ELLIPSIS
  0.66...

  >>> metrics.completeness_score(labels_true, labels_pred) # doctest: +ELLIPSIS
  0.42...

它们的调和均值被称之为 **V-measure** ，通过函数 :func:`v_measure_score` 来计算::

  >>> metrics.v_measure_score(labels_true, labels_pred)    # doctest: +ELLIPSIS
  0.51...

如果使用的聚合函数(aggregation function)是 算术平均值 [B2011]_ ， V-measure 实际上等价于前面讨论的互信息(NMI)。

Homogeneity, completeness 和 V-measure 可以通过 :func:`homogeneity_completeness_v_measure` 函数一次性的计算出来 ::

  >>> metrics.homogeneity_completeness_v_measure(labels_true, labels_pred)
  ...                                                      # doctest: +ELLIPSIS
  (0.66..., 0.42..., 0.51...)

下面的聚类分配稍微好点儿，因为它是同质的但却不是完备的::

  >>> labels_pred = [0, 0, 0, 1, 2, 2]
  >>> metrics.homogeneity_completeness_v_measure(labels_true, labels_pred)
  ...                                                      # doctest: +ELLIPSIS
  (1.0, 0.68..., 0.81...)

.. note::

  :func:`v_measure_score` 是 **对称的(symmetric)**: 它可被用于在同一个数据集上评估两个independent assignments的**一致性(agreement)**。

  函数 :func:`completeness_score` 和
  :func:`homogeneity_score` 不是这样的: both are bound by the relationship::

    homogeneity_score(a, b) == completeness_score(b, a)


优点
~~~~~~~~~~

- **有界的得分**: 0.0 代表最坏的情况, 1.0 是最完美的得分。

- 直观可解释性: 具有坏的 V-measure 值的聚类可以**从同质性和完备性角度进行定性分析(qualitatively analyzed in terms of homogeneity and completeness)**
  来更好的感受到聚类算法预测标签分配的时候犯了哪种错误。

- **对聚类结构没有做任何假定**: 可以用于比较聚类算法，比如 假定了各向同性的blob shapes的k-means方法的结果 和 寻找具有
  "folded"形状的谱聚类方法的结果进行比较。


缺点
~~~~~~~~~

- 以前引入的度量指标 **并没有对随机标记(random labeling)进行标准化** ：这意味着，依赖于样本数量、簇的数量和真实类的数量，
  一个完全的随机标记对于同质性、完备性和v测度来说并不总是产生相同的值。
  **特别是，随机标记不会产生零得分，尤其是当簇数很大时**。

  当样本数大于1000个，簇数小于10个时，可以安全地忽略这个问题。
  **对于较小的样本大小或较多的簇数，使用调整后的索引比如the Adjusted Rand Index (ARI) 更安全。**

.. figure:: ../auto_examples/cluster/images/sphx_glr_plot_adjusted_for_chance_measures_001.png
   :target: ../auto_examples/cluster/plot_adjusted_for_chance_measures.html
   :align: center
   :scale: 100

- 这些度量指标 **需要 ground truth classes 的相关知识** 
  而在实践中几乎不可得到，或者需要人工标注者手动分配（如在监督学习环境中）。


.. topic:: 案例:

 * :ref:`sphx_glr_auto_examples_cluster_plot_adjusted_for_chance_measures.py`: 
   分析数据集大小对随机分配标签(random assignments)的聚类度量值的影响。


数学表达式
~~~~~~~~~~~~~~~~~~~~~~~~

同质性(Homogeneity) 和 完备性(completeness) 得分正式定义如下:

.. math:: h = 1 - \frac{H(C|K)}{H(C)}

.. math:: c = 1 - \frac{H(K|C)}{H(K)}

其中 :math:`H(C|K)` 是 **给定聚类标签分配以后各个类的条件熵(conditional entropy of the classes given the cluster assignments)** 并且由下式给出:

.. math:: H(C|K) = - \sum_{c=1}^{|C|} \sum_{k=1}^{|K|} \frac{n_{c,k}}{n}
          \cdot \log\left(\frac{n_{c,k}}{n_k}\right)

并且 :math:`H(C)` 是 **各个类的熵(entropy of the classes)** 并且由下式给出:

.. math:: H(C) = - \sum_{c=1}^{|C|} \frac{n_c}{n} \cdot \log\left(\frac{n_c}{n}\right)

公式中 :math:`n` 是样本总量, :math:`n_c` 和 :math:`n_k` 分别是属于 class :math:`c` 和 cluster :math:`k` 的样本的数量，最后
:math:`n_{c,k}` 是从 class :math:`c` 被分配到 cluster :math:`k` 的样本数量。

**给定某个类以后簇的条件熵** :math:`H(K|C)` 和 **各个簇的熵(entropy of clusters)** :math:`H(K)` 以对称方式定义。

Rosenberg 和 Hirschberg 进一步定义了 **V-measure** 作为 **同质性和完备性的调和均值(harmonic mean of homogeneity and completeness)**:

.. math:: v = 2 \cdot \frac{h \cdot c}{h + c}

.. topic:: 参考文献

 * `V-Measure: A conditional entropy-based external cluster evaluation
   measure <http://aclweb.org/anthology/D/D07/D07-1043.pdf>`_
   Andrew Rosenberg and Julia Hirschberg, 2007

 .. [B2011] `Identication and Characterization of Events in Social Media
   <http://www.cs.columbia.edu/~hila/hila-thesis-distributed.pdf>`_, Hila
   Becker, PhD Thesis.

.. _fowlkes_mallows_scores:

Fowlkes-Mallows scores
----------------------

当已知样本的真实类分配时，可以使用 Fowlkes-Mallows index (:func:`sklearn.metrics.fowlkes_mallows_score`)。
Fowlkes-Mlets得分 FMI 被定义为成对精度(pairwise precision)和成对召回率(pairwise recall)的几何均值(geometric mean):

.. math:: \text{FMI} = \frac{\text{TP}}{\sqrt{(\text{TP} + \text{FP}) (\text{TP} + \text{FN})}}

其中 ``TP`` 是 **True Positive** 的数量 (i.e. 在 真实标签集中 和 预测标签集中 属于相同簇的点对的数量), 
``FP`` 是 **False Positive** 的数量(i.e. 在 真实标签集中 但不在 预测标签集中 属于相同簇的点对的数量),
``FN`` 是 **False Negative** 的数量(i.e 不在 真实标签集中 但在 预测标签集中 属于相同簇的点对的数量)。

FMI 得分取值范围在0到1之间。取值越高表明两个簇之间的相似性越好。

  >>> from sklearn import metrics
  >>> labels_true = [0, 0, 0, 1, 1, 1]
  >>> labels_pred = [0, 0, 1, 1, 2, 2]

  >>> metrics.fowlkes_mallows_score(labels_true, labels_pred)  # doctest: +ELLIPSIS
  0.47140...

可以在预测出的标签(predicted labels)中排列 0 和 1, 重命名为 2 到 3， 并得到相同的得分 ::

  >>> labels_pred = [1, 1, 0, 0, 3, 3]

  >>> metrics.fowlkes_mallows_score(labels_true, labels_pred)  # doctest: +ELLIPSIS
  0.47140...

完美标记(Perfect labeling)的得分是 1.0::

  >>> labels_pred = labels_true[:]
  >>> metrics.fowlkes_mallows_score(labels_true, labels_pred)  # doctest: +ELLIPSIS
  1.0

坏的标记 (e.g. independent labelings) 的得分是 0 ::

  >>> labels_true = [0, 1, 2, 0, 3, 4, 5, 1]
  >>> labels_pred = [1, 1, 0, 0, 2, 2, 2, 2]
  >>> metrics.fowlkes_mallows_score(labels_true, labels_pred)  # doctest: +ELLIPSIS
  0.0

优点
~~~~~~~~~~

- **随机(均匀)标签分配有一个接近于0的 FMI 得分。**
  对于 ``n_clusters`` 和 ``n_samples`` 的任何值（这不是 raw Mutual Information 或者 V-measure 的情况）。

- **上界为 1** :  得分值接近于 0 表明两个标签分配集合很大程度上是独立的(largely independent), 而得分值接近于 1 表明两个标签分配集合
  具有很大的一致性(significant agreement)。 更进一步, 正好是 0 的FMI表示两个标签分配纯粹独立(**purely** independent),
  正好是 1 的FMI表示两个标签分配相等。 (with or without permutation).

- **对聚类结构没有做任何限制**: 可以用于比较聚类算法，比如 假定了各向同性的blob shapes的k-means方法的结果 和 寻找具有
  "folded"形状的谱聚类方法的结果进行比较。


缺点
~~~~~~~~~

- 与惯性(inertia)方法不同, **FMI-based measures 需要 ground truth classes 的相关知识** 
  而在实践中几乎不可得到，或者需要人工标注者手动分配（如在监督学习环境中）。

.. topic:: 参考文献

  * E. B. Fowkles and C. L. Mallows, 1983. "A method for comparing two
    hierarchical clusterings". Journal of the American Statistical Association.
    http://wildfire.stat.ucla.edu/pdflibrary/fowlkes.pdf

  * `Wikipedia entry for the Fowlkes-Mallows Index
    <https://en.wikipedia.org/wiki/Fowlkes-Mallows_index>`_

.. _silhouette_coefficient:

Silhouette Coefficient
----------------------

如果不知道ground truth labels，则必须使用模型本身进行评估。
Silhouette Coefficient (:func:`sklearn.metrics.silhouette_score`) 就是这样一种评估的例子，
其中Silhouette Coefficient的得分越高对应于具有更好的聚类能力的模型。
The Silhouette Coefficient 定义在每个样本上 并且由两个得分组成:

- **a**: 在同一个类中一个样本到所有其他样本的平均距离。

- **b**: 在 *next nearest cluster* 中，一个样本到所有其他样本点的平均距离。

那么，对一个单个样本来说，Silhouette Coefficient *s* 由下式给出:

.. math:: s = \frac{b - a}{max(a, b)}

对于一个样本集合，Silhouette Coefficient 是集合中每个样本的Silhouette Coefficient的均值。


  >>> from sklearn import metrics
  >>> from sklearn.metrics import pairwise_distances
  >>> from sklearn import datasets
  >>> dataset = datasets.load_iris()
  >>> X = dataset.data
  >>> y = dataset.target

在正常的使用中, Silhouette Coefficient 会被运用到一个聚类簇的结果的分析中。

  >>> import numpy as np
  >>> from sklearn.cluster import KMeans
  >>> kmeans_model = KMeans(n_clusters=3, random_state=1).fit(X)
  >>> labels = kmeans_model.labels_
  >>> metrics.silhouette_score(X, labels, metric='euclidean')
  ...                                                      # doctest: +ELLIPSIS
  0.55...

.. topic:: 参考文献

 * Peter J. Rousseeuw (1987). "Silhouettes: a Graphical Aid to the
   Interpretation and Validation of Cluster Analysis". Computational
   and Applied Mathematics 20: 53–65.
   `doi:10.1016/0377-0427(87)90125-7 <https://doi.org/10.1016/0377-0427(87)90125-7>`_.


优点
~~~~~~~~~~

- 对于高度稠密的聚类，得分被限制在 -1 (for incorrect clustering) 和 +1 (for highly dense clustering)之间。 
  得分在 0 附近表明是有重叠的聚类(overlapping clusters)。

- 当簇(cluster)密集且分离良好时，得分较高，这与cluster的标准概念有关。


缺点
~~~~~~~~~

- The Silhouette Coefficient 在convex clusters上比在其他一些clusters上一般具有较高的得分，比如 通过DBSCAN获得的基于密度的clusters。

.. topic:: 案例:

 * :ref:`sphx_glr_auto_examples_cluster_plot_kmeans_silhouette_analysis.py` : 在这个例子中，使用剪影分析(silhouette analysis)来选择n_clusters的最优值。

.. _calinski_harabaz_index:

Calinski-Harabaz Index
----------------------

如果 ground truth labels 是未知的, the Calinski-Harabaz index (:func:`sklearn.metrics.calinski_harabaz_score`) - 
也被称之为 方差比率准则(Variance Ratio Criterion) - 可被用于评估模型, 
其中，Calinski-Harabaz 得分越高，与之关联的模型就有更好的聚类。

对于 :math:`k` 个簇，给出了Calinski-Harabaz分数 :math:`s` 作为 簇间分散均值(between-clusters dispersion mean) 和 
簇内分散(within-cluster dispersion)的比值：

.. math::
  s(k) = \frac{\mathrm{Tr}(B_k)}{\mathrm{Tr}(W_k)} \times \frac{N - k}{k - 1}

其中 :math:`B_K` 是簇间分散矩阵 ， :math:`W_K` 是簇内分散矩阵，定义如下:

.. math:: W_k = \sum_{q=1}^k \sum_{x \in C_q} (x - c_q) (x - c_q)^T

.. math:: B_k = \sum_q n_q (c_q - c) (c_q - c)^T

这里， :math:`N` 是我们数据点的数量, :math:`C_q` 是在簇 :math:`q` 中的点的集合, :math:`c_q` 是簇 :math:`q` 的中心, 
:math:`c` 是 :math:`E` 的中心, :math:`n_q` 是簇 :math:`q` 中点的数量::


  >>> from sklearn import metrics
  >>> from sklearn.metrics import pairwise_distances
  >>> from sklearn import datasets
  >>> dataset = datasets.load_iris()
  >>> X = dataset.data
  >>> y = dataset.target

在正常的使用中, Calinski-Harabaz index 会被运用到一个聚类簇的结果的分析中。

  >>> import numpy as np
  >>> from sklearn.cluster import KMeans
  >>> kmeans_model = KMeans(n_clusters=3, random_state=1).fit(X)
  >>> labels = kmeans_model.labels_
  >>> metrics.calinski_harabaz_score(X, labels)  # doctest: +ELLIPSIS
  561.62...


优点
~~~~~~~~~~

- 当簇(cluster)密集且分离良好时，得分较高，这与cluster的标准概念有关。

- 此得分的计算很快


缺点
~~~~~~~~~

- The Calinski-Harabaz index 在convex clusters上比在其他一些clusters上一般具有较高的得分，比如 通过DBSCAN获得的基于密度的clusters。

.. topic:: 参考文献

 *  Caliński, T., & Harabasz, J. (1974). "A dendrite method for cluster
    analysis". Communications in Statistics-theory and Methods 3: 1-27.
    `doi:10.1080/03610926.2011.560741 <https://doi.org/10.1080/03610926.2011.560741>`_.


.. _davies-bouldin_index:

Davies-Bouldin Index
--------------------

如果 ground truth labels 是未知的, the Davies-Bouldin index (:func:`sklearn.metrics.davies_bouldin_score`) 可被用于评估模型, 
其中，较低的Davies-Bouldin指数与团簇间分离程度较高的模型有关。

Davies-Bouldin 指数 被定义为每个团簇 :math:`C_i` (:math:`i=1, ..., k`)和它的最相似的一个 
:math:`C_j` 之间的平均相似度(average similarity)。 在该指数的上下文中，相似度被定义为一个度量 :math:`R_{ij}`,在下面两个中做折中:

- :math:`s_i`, 团簇 :math:`i` 中的每个点与该cluster的质心之间的平均距离-- 也被称为 团簇直径(cluster diameter)。
- :math:`d_{ij}`, 团簇质心 :math:`i` 和 团簇质心 :math:`j` 之间的距离。

使 :math:`R_ij`  非负和对称的 构建 :math:`R_ij` 的一个简单选择就是:

.. math::
   R_{ij} = \frac{s_i + s_j}{d_{ij}}

那么 Davies-Bouldin index 定义如下:

.. math::
   DB = \frac{1}{k} \sum_{i=1}^k \max_{i \neq j} R_{ij}

零分可能是最低的得分。接近于零的值表示更好的划分。

在正常的使用中, Davies-Bouldin index 会被运用到一个聚类簇的结果的分析中,如下所示:

  >>> from sklearn import datasets
  >>> iris = datasets.load_iris()
  >>> X = iris.data
  >>> from sklearn.cluster import KMeans
  >>> from sklearn.metrics import davies_bouldin_score
  >>> kmeans = KMeans(n_clusters=3, random_state=1).fit(X)
  >>> labels = kmeans.labels_
  >>> davies_bouldin_score(X, labels)  # doctest: +ELLIPSIS
  0.6619...


优点
~~~~~~~~~~

- Davies-Bouldin 的计算比Silhouette 得分的计算更简单。

- Davies-Bouldin index 只计算数据集固有的数量和特性。

缺点
~~~~~~~~~

- The Davies-Boulding index 在convex clusters上比在其他一些clusters上一般具有较高的得分，比如 通过DBSCAN获得的基于密度的clusters。

- 质心距离的使用被限制在欧式空间。

- 此方法所报告的好的值不意味着最佳的信息检索(best information retrieval)。

.. topic:: 参考文献

 * Davies, David L.; Bouldin, Donald W. (1979).
   "A Cluster Separation Measure"
   IEEE Transactions on Pattern Analysis and Machine Intelligence.
   PAMI-1 (2): 224-227.
   `doi:10.1109/TPAMI.1979.4766909 <http://dx.doi.org/10.1109/TPAMI.1979.4766909>`_.

 * Halkidi, Maria; Batistakis, Yannis; Vazirgiannis, Michalis (2001).
   "On Clustering Validation Techniques"
   Journal of Intelligent Information Systems, 17(2-3), 107-145.
   `doi:10.1023/A:1012801612483 <http://dx.doi.org/10.1023/A:1012801612483>`_.

 * `Wikipedia entry for Davies-Bouldin index
   <https://en.wikipedia.org/wiki/Davies–Bouldin_index>`_.


.. _contingency_matrix:

Contingency Matrix
------------------

Contingency matrix (:func:`sklearn.metrics.cluster.contingency_matrix`) 报告每个 真实/预测团簇对(true/predicted cluster pair) 
的交集的基数(intersection cardinality)。
The contingency matrix 为所有聚类度量提供了足够的统计信息，其中样本是独立同分布的，并且不需要考虑某些不能被聚类的实例。

这里有个例子::

   >>> from sklearn.metrics.cluster import contingency_matrix
   >>> x = ["a", "a", "a", "b", "b", "b"]
   >>> y = [0, 0, 1, 1, 2, 2]
   >>> contingency_matrix(x, y)
   array([[2, 1, 0],
          [0, 1, 2]])

输出数组的第一行表示有三个样本真正的类标签是“a”。 在它们中间, 有两个在预测出的 cluster 0 里, 有一个在 cluster 1 里面,
没有在cluster 2里的样本点。 第二行表示有三个样本真正的聚类为“b”。 
在它们中间，没有样本被预测到cluster 0中的, 有一个在cluster 1,还有一个在cluster 2里。

分类的混淆矩阵(:ref:`confusion matrix <confusion_matrix>`)是一个方阵，其中行和列的顺序对应于类的列表。


优点
~~~~~~~~~~

- 允许检查每个true cluster在predicted clusters之间的传播，反之亦然。

- 计算出的contingency table通常用于计算两个聚类之间的相似统计量(就像本文档中列出的其他统计指标一样)。

缺点
~~~~~~~~~

- Contingency matrix 对于少量的聚类来说很容易解释，但是对于大量的聚类则变得非常难以解释。

- 它没有给出一个单独的度量来作为聚类优化的目标


.. topic:: 参考文献

 * `Wikipedia entry for contingency matrix <https://en.wikipedia.org/wiki/Contingency_table>`_
