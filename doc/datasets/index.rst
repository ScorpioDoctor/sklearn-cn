.. _datasets:

=========================
数据集加载工具
=========================

.. currentmodule:: sklearn.datasets

``sklearn.datasets`` package 自带了一些迷你数据集(small toy datasets)，
这在 :ref:`Getting Started <loading_example_dataset>` 章节也提到过。

这个 package 也提供了很多帮助函数去获取稍大一点儿的数据集，这些数据集是在机器学习社区中经常用到的
用来测试算法的性能的，他们都是来自 真实世界('real world')的数据集。

为了评估数据集(``n_samples`` 和 ``n_features``) 的规模(scale)对算法模型的影响，同时还能控制数据的某些统计属性
(典型的比如特征的相关性和信息量)，我们还需要人工合成一些数据集(generate synthetic data)。

**译者注：** 由于我之前在人工智能社区网(wwww.studyai.com)做过一些scikit-learn的视频课程，当时对数据集的API这块儿讲的挺详细的，下面我把当时讲课
时自己制作的PPT和视频的链接地址放在这儿，大家可以去听听，看看，我觉得PPT总结的还是不错的，o(∩_∩)o 哈哈  。 
(`Sklearn数据集操作API (一) <http://www.studyai.com/article/ddf094e18e834a95>`_)，
(`Sklearn数据集操作API (三) <http://www.studyai.com/article/dbb7ea7264784b11>`_)，
(`Sklearn数据集操作API (四) <http://www.studyai.com/article/b82ea4b5ae084e8f>`_)，
数据集这一段 对应的视频教程 可以去 
(`我的优酷自频道 <https://list.youku.com/albumlist/show/id_49255928.html?spm=a2h0j.11185381.bpmodule-playpage-righttitle.5~H2~A>`_)
里面找sklearn的播单，感兴趣的可以听听。


通用数据集 API
===================

根据所需数据集的类型，有三种主要类型的数据集API接口可用于获取数据集。
  
**The dataset loaders.** 这些可用来加载小的标准数据集, 在 :ref:`toy_datasets` 中有介绍。  

**The dataset fetchers.** 这些可用来下载并加载大的 real-world 数据集, 在 :ref:`real_world_datasets` 中有介绍。

loaders 和 fetchers 的所有函数都返回一个字典一样的对象，里面至少包含两项 : 
shape 为 ``n_samples`` * ``n_features`` 的数组，对应的字典key是 ``data`` (20newsgroups 数据集除外) 
以及 长度为 ``n_samples`` 的numpy数组, 包含了目标值(target values), 对应的字典key是 ``target``。

通过将 ``return_X_y`` 参数设置为 ``True`` ，几乎所有这些函数都可以将输出约束为只包含数据和目标的元组。

数据集还包含一个完整的 ``DESCR`` 属性描述，一些数据集还包含了 ``feature_names`` 和 ``target_names`` 。有关详细信息，请参阅下面的数据集说明。

**The dataset generation functions.** 它们可以用来生成受控的合成数据集(synthetic datasets), 在 :ref:`sample_generators` 中有介绍。

这些函数返回一个元组 ``(X, y)`` ，该元组由shape为 ``n_samples`` * ``n_features`` 的
numpy数组 ``X`` 和长度为 ``n_samples`` 的包含目标 ``y`` 的数组组成。

此外，还有一些用于加载其他格式或其他位置的数据集的混合工具(miscellanous tools),在 :ref:`loading_other_datasets` 中有介绍。 

.. _toy_datasets:

迷你数据集
============

scikit-learn 附带一些小的标准数据集，这些数据集不需要从某个外部网站下载任何文件。

这些数据集可以用下面的函数加载 :

.. autosummary::

   :toctree: ../modules/generated/
   :template: function.rst

   load_boston
   load_iris
   load_diabetes
   load_digits
   load_linnerud
   load_wine
   load_breast_cancer

这些数据集对于快速说明在scikit-learn中实现的各种算法的行为非常有用。然而，它们往往太小，无法代表现实世界中的机器学习任务。

.. toctree::
    :maxdepth: 2
    :hidden:

    boston_house_prices
    iris
    diabetes
    digits
    linnerud
    wine_data
    breast_cancer

.. include:: ../../sklearn/datasets/descr/boston_house_prices.rst

.. include:: ../../sklearn/datasets/descr/iris.rst

.. include:: ../../sklearn/datasets/descr/diabetes.rst

.. include:: ../../sklearn/datasets/descr/digits.rst

.. include:: ../../sklearn/datasets/descr/linnerud.rst

.. include:: ../../sklearn/datasets/descr/wine_data.rst

.. include:: ../../sklearn/datasets/descr/breast_cancer.rst

.. _real_world_datasets:

真实世界中的数据集
===================

scikit-learn 提供加载较大数据集的工具，并在必要时下载这些数据集。

这些数据集可以用下面的函数加载 :

.. autosummary::

   :toctree: ../modules/generated/
   :template: function.rst

   fetch_olivetti_faces
   fetch_20newsgroups
   fetch_20newsgroups_vectorized
   fetch_lfw_people
   fetch_lfw_pairs
   fetch_covtype
   fetch_rcv1
   fetch_kddcup99
   fetch_california_housing

.. toctree::
    :maxdepth: 2
    :hidden:

    olivetti_faces
    twenty_newsgroups
    labeled_faces
    covtype
    rcv1
    kddcup99
    california_housing

.. include:: ../../sklearn/datasets/descr/olivetti_faces.rst

.. include:: ../../sklearn/datasets/descr/twenty_newsgroups.rst

.. include:: ../../sklearn/datasets/descr/lfw.rst

.. include:: ../../sklearn/datasets/descr/covtype.rst

.. include:: ../../sklearn/datasets/descr/rcv1.rst

.. include:: ../../sklearn/datasets/descr/kddcup99.rst

.. include:: ../../sklearn/datasets/descr/california_housing.rst

.. _sample_generators:

人工合成的数据集
==================

此外，scikit-learn还包括各种随机样本生成器，可用于构建具有可控大小和复杂性的人工数据集(artificial datasets)。


用于分类和回归的数据集的生成器
--------------------------------------------

这些生成器产生一个特征矩阵和相应的离散目标。

单标签
~~~~~~~~~~~~

Both :func:`make_blobs` and :func:`make_classification` create multiclass
datasets by allocating each class one or more normally-distributed clusters of
points.  :func:`make_blobs` provides greater control regarding the centers and
standard deviations of each cluster, and is used to demonstrate clustering.
:func:`make_classification` specialises in introducing noise by way of:
correlated, redundant and uninformative features; multiple Gaussian clusters
per class; and linear transformations of the feature space.

:func:`make_gaussian_quantiles` divides a single Gaussian cluster into
near-equal-size classes separated by concentric hyperspheres.
:func:`make_hastie_10_2` generates a similar binary, 10-dimensional problem.

.. image:: ../auto_examples/datasets/images/sphx_glr_plot_random_dataset_001.png
   :target: ../auto_examples/datasets/plot_random_dataset.html
   :scale: 50
   :align: center

:func:`make_circles` and :func:`make_moons` generate 2d binary classification
datasets that are challenging to certain algorithms (e.g. centroid-based
clustering or linear classification), including optional Gaussian noise.
They are useful for visualisation. produces Gaussian
data with a spherical decision boundary for binary classification.

多标签
~~~~~~~~~~

:func:`make_multilabel_classification` generates random samples with multiple
labels, reflecting a bag of words drawn from a mixture of topics. The number of
topics for each document is drawn from a Poisson distribution, and the topics
themselves are drawn from a fixed random distribution. Similarly, the number of
words is drawn from Poisson, with words drawn from a multinomial, where each
topic defines a probability distribution over words. Simplifications with
respect to true bag-of-words mixtures include:

* Per-topic word distributions are independently drawn, where in reality all
  would be affected by a sparse base distribution, and would be correlated.
* For a document generated from multiple topics, all topics are weighted
  equally in generating its bag of words.
* Documents without labels words at random, rather than from a base
  distribution.

.. image:: ../auto_examples/datasets/images/sphx_glr_plot_random_multilabel_dataset_001.png
   :target: ../auto_examples/datasets/plot_random_multilabel_dataset.html
   :scale: 50
   :align: center

双向聚类
~~~~~~~~~~~~

.. autosummary::

   :toctree: ../modules/generated/
   :template: function.rst

   make_biclusters
   make_checkerboard


用于回归的数据集生成器
-------------------------

:func:`make_regression` produces regression targets as an optionally-sparse
random linear combination of random features, with noise. Its informative
features may be uncorrelated, or low rank (few features account for most of the
variance).

Other regression generators generate functions deterministically from
randomized features.  :func:`make_sparse_uncorrelated` produces a target as a
linear combination of four features with fixed coefficients.
Others encode explicitly non-linear relations:
:func:`make_friedman1` is related by polynomial and sine transforms;
:func:`make_friedman2` includes feature multiplication and reciprocation; and
:func:`make_friedman3` is similar with an arctan transformation on the target.

用于流形学习的数据集生成器
--------------------------------

.. autosummary::

   :toctree: ../modules/generated/
   :template: function.rst

   make_s_curve
   make_swiss_roll

用于信号分解的生成器
----------------------------

.. autosummary::

   :toctree: ../modules/generated/
   :template: function.rst

   make_low_rank_matrix
   make_sparse_coded_signal
   make_spd_matrix
   make_sparse_spd_matrix


.. _loading_other_datasets:

加载其他类型的数据集
======================

.. _sample_images:

示例图像
-------------

Scikit-learn also embed a couple of sample JPEG images published under Creative
Commons license by their authors. Those image can be useful to test algorithms
and pipeline on 2D data.

.. autosummary::

   :toctree: ../modules/generated/
   :template: function.rst

   load_sample_images
   load_sample_image

.. image:: ../auto_examples/cluster/images/sphx_glr_plot_color_quantization_001.png
   :target: ../auto_examples/cluster/plot_color_quantization.html
   :scale: 30
   :align: right


.. warning::

  The default coding of images is based on the ``uint8`` dtype to
  spare memory.  Often machine learning algorithms work best if the
  input is converted to a floating point representation first.  Also,
  if you plan to use ``matplotlib.pyplpt.imshow`` don't forget to scale to the range
  0 - 1 as done in the following example.

.. topic:: Examples:

    * :ref:`sphx_glr_auto_examples_cluster_plot_color_quantization.py`

.. _libsvm_loader:

svmlight 或 libsvm 格式的数据集
------------------------------------

scikit-learn includes utility functions for loading
datasets in the svmlight / libsvm format. In this format, each line
takes the form ``<label> <feature-id>:<feature-value>
<feature-id>:<feature-value> ...``. This format is especially suitable for sparse datasets.
In this module, scipy sparse CSR matrices are used for ``X`` and numpy arrays are used for ``y``.

You may load a dataset like as follows::

  >>> from sklearn.datasets import load_svmlight_file
  >>> X_train, y_train = load_svmlight_file("/path/to/train_dataset.txt")
  ...                                                         # doctest: +SKIP

You may also load two (or more) datasets at once::

  >>> X_train, y_train, X_test, y_test = load_svmlight_files(
  ...     ("/path/to/train_dataset.txt", "/path/to/test_dataset.txt"))
  ...                                                         # doctest: +SKIP

In this case, ``X_train`` and ``X_test`` are guaranteed to have the same number
of features. Another way to achieve the same result is to fix the number of
features::

  >>> X_test, y_test = load_svmlight_file(
  ...     "/path/to/test_dataset.txt", n_features=X_train.shape[1])
  ...                                                         # doctest: +SKIP

.. topic:: Related links:

 _`Public datasets in svmlight / libsvm format`: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets

 _`Faster API-compatible implementation`: https://github.com/mblondel/svmlight-loader

..
    For doctests:

    >>> import numpy as np
    >>> import os

.. _openml:

从 openml.org 库下载数据集
---------------------------------------------------

`openml.org <https://openml.org>`_ is a public repository for machine learning
data and experiments, that allows everybody to upload open datasets.

The ``sklearn.datasets`` package is able to download datasets
from the repository using the function
:func:`sklearn.datasets.fetch_openml`.

For example, to download a dataset of gene expressions in mice brains::

  >>> from sklearn.datasets import fetch_openml
  >>> mice = fetch_openml(name='miceprotein', version=4)

To fully specify a dataset, you need to provide a name and a version, though
the version is optional, see :ref:`openml_versions` below.
The dataset contains a total of 1080 examples belonging to 8 different
classes::

  >>> mice.data.shape
  (1080, 77)
  >>> mice.target.shape
  (1080,)
  >>> np.unique(mice.target) # doctest: +NORMALIZE_WHITESPACE
  array(['c-CS-m', 'c-CS-s', 'c-SC-m', 'c-SC-s', 't-CS-m', 't-CS-s', 't-SC-m', 't-SC-s'], dtype=object)

You can get more information on the dataset by looking at the ``DESCR``
and ``details`` attributes::

  >>> print(mice.DESCR) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS +SKIP
  **Author**: Clara Higuera, Katheleen J. Gardiner, Krzysztof J. Cios
  **Source**: [UCI](https://archive.ics.uci.edu/ml/datasets/Mice+Protein+Expression) - 2015
  **Please cite**: Higuera C, Gardiner KJ, Cios KJ (2015) Self-Organizing
  Feature Maps Identify Proteins Critical to Learning in a Mouse Model of Down
  Syndrome. PLoS ONE 10(6): e0129126...

  >>> mice.details # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS +SKIP
  {'id': '40966', 'name': 'MiceProtein', 'version': '4', 'format': 'ARFF',
  'upload_date': '2017-11-08T16:00:15', 'licence': 'Public',
  'url': 'https://www.openml.org/data/v1/download/17928620/MiceProtein.arff',
  'file_id': '17928620', 'default_target_attribute': 'class',
  'row_id_attribute': 'MouseID',
  'ignore_attribute': ['Genotype', 'Treatment', 'Behavior'],
  'tag': ['OpenML-CC18', 'study_135', 'study_98', 'study_99'],
  'visibility': 'public', 'status': 'active',
  'md5_checksum': '3c479a6885bfa0438971388283a1ce32'}


The ``DESCR`` contains a free-text description of the data, while ``details``
contains a dictionary of meta-data stored by openml, like the dataset id.
For more details, see the `OpenML documentation
<https://docs.openml.org/#data>`_ The ``data_id`` of the mice protein dataset
is 40966, and you can use this (or the name) to get more information on the
dataset on the openml website::

  >>> mice.url
  'https://www.openml.org/d/40966'

The ``data_id`` also uniquely identifies a dataset from OpenML::

  >>> mice = fetch_openml(data_id=40966)
  >>> mice.details # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS +SKIP
  {'id': '4550', 'name': 'MiceProtein', 'version': '1', 'format': 'ARFF',
  'creator': ...,
  'upload_date': '2016-02-17T14:32:49', 'licence': 'Public', 'url':
  'https://www.openml.org/data/v1/download/1804243/MiceProtein.ARFF', 'file_id':
  '1804243', 'default_target_attribute': 'class', 'citation': 'Higuera C,
  Gardiner KJ, Cios KJ (2015) Self-Organizing Feature Maps Identify Proteins
  Critical to Learning in a Mouse Model of Down Syndrome. PLoS ONE 10(6):
  e0129126. [Web Link] journal.pone.0129126', 'tag': ['OpenML100', 'study_14',
  'study_34'], 'visibility': 'public', 'status': 'active', 'md5_checksum':
  '3c479a6885bfa0438971388283a1ce32'}

.. _openml_versions:

数据集版本
~~~~~~~~~~~~~~~~

A dataset is uniquely specified by its ``data_id``, but not necessarily by its
name. Several different "versions" of a dataset with the same name can exist
which can contain entirely different datasets.
If a particular version of a dataset has been found to contain significant
issues, it might be deactivated. Using a name to specify a dataset will yield
the earliest version of a dataset that is still active. That means that
``fetch_openml(name="miceprotein")`` can yield different results at different
times if earlier versions become inactive.
You can see that the dataset with ``data_id`` 40966 that we fetched above is
the version 1 of the "miceprotein" dataset::

  >>> mice.details['version']  #doctest: +SKIP
  '1'

In fact, this dataset only has one version. The iris dataset on the other hand
has multiple versions::

  >>> iris = fetch_openml(name="iris")
  >>> iris.details['version']  #doctest: +SKIP
  '1'
  >>> iris.details['id']  #doctest: +SKIP
  '61'

  >>> iris_61 = fetch_openml(data_id=61)
  >>> iris_61.details['version']
  '1'
  >>> iris_61.details['id']
  '61'

  >>> iris_969 = fetch_openml(data_id=969)
  >>> iris_969.details['version']
  '3'
  >>> iris_969.details['id']
  '969'

Specifying the dataset by the name "iris" yields the lowest version, version 1,
with the ``data_id`` 61. To make sure you always get this exact dataset, it is
safest to specify it by the dataset ``data_id``. The other dataset, with
``data_id`` 969, is version 3 (version 2 has become inactive), and contains a
binarized version of the data::

  >>> np.unique(iris_969.target)
  array(['N', 'P'], dtype=object)

You can also specify both the name and the version, which also uniquely
identifies the dataset::

  >>> iris_version_3 = fetch_openml(name="iris", version=3)
  >>> iris_version_3.details['version']
  '3'
  >>> iris_version_3.details['id']
  '969'


.. topic:: References:

 * Vanschoren, van Rijn, Bischl and Torgo
   `"OpenML: networked science in machine learning"
   <https://arxiv.org/pdf/1407.7722.pdf>`_,
   ACM SIGKDD Explorations Newsletter, 15(2), 49-60, 2014.

.. _external_datasets:

从外部数据集加载数据
------------------------------

scikit-learn works on any numeric data stored as numpy arrays or scipy sparse
matrices. Other types that are convertible to numeric arrays such as pandas
DataFrame are also acceptable.
 
Here are some recommended ways to load standard columnar data into a 
format usable by scikit-learn: 

* `pandas.io <https://pandas.pydata.org/pandas-docs/stable/io.html>`_ 
  provides tools to read data from common formats including CSV, Excel, JSON
  and SQL. DataFrames may also be constructed from lists of tuples or dicts.
  Pandas handles heterogeneous data smoothly and provides tools for
  manipulation and conversion into a numeric array suitable for scikit-learn.
* `scipy.io <https://docs.scipy.org/doc/scipy/reference/io.html>`_ 
  specializes in binary formats often used in scientific computing 
  context such as .mat and .arff
* `numpy/routines.io <https://docs.scipy.org/doc/numpy/reference/routines.io.html>`_
  for standard loading of columnar data into numpy arrays
* scikit-learn's :func:`datasets.load_svmlight_file` for the svmlight or libSVM
  sparse format
* scikit-learn's :func:`datasets.load_files` for directories of text files where
  the name of each directory is the name of each category and each file inside
  of each directory corresponds to one sample from that category

For some miscellaneous data such as images, videos, and audio, you may wish to
refer to:

* `skimage.io <https://scikit-image.org/docs/dev/api/skimage.io.html>`_ or
  `Imageio <https://imageio.readthedocs.io/en/latest/userapi.html>`_ 
  for loading images and videos into numpy arrays
* `scipy.io.wavfile.read 
  <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.io.wavfile.read.html>`_ 
  for reading WAV files into a numpy array

Categorical (or nominal) features stored as strings (common in pandas DataFrames) 
will need converting to numerical features using :class:`sklearn.preprocessing.OneHotEncoder`
or :class:`sklearn.preprocessing.OrdinalEncoder` or similar.
See :ref:`preprocessing`.

Note: if you manage your own numerical data it is recommended to use an 
optimized file format such as HDF5 to reduce data load times. Various libraries
such as H5Py, PyTables and pandas provides a Python interface for reading and 
writing data in that format.
