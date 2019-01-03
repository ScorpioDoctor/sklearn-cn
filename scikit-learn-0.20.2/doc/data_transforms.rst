.. include:: includes/big_toc_css.rst

.. _data-transforms:

数据集变换
-----------------------

scikit-learn 提供了一个变换器(transformers)的库, 它可以完成 清理 (see
:ref:`preprocessing`), 降维 (see :ref:`data_reduction`), 扩展 (see
:ref:`kernel_approximation`) 或 生成 (see :ref:`feature_extraction`)
特征表达(feature representations) 等一系列特征变换任务。

就像其他estimators一样, 这些变换器(transformers)被封装成**类**的形式。
这些变换器类有一个 ``fit`` 方法, 用来从训练数据学习模型参数(e.g. 用于归一化的均值和方差)；
变换器类还有一个  ``transform`` 方法用来把fit好的变换模型用到之前未见过的数据上。
而 ``fit_transform`` 则一次性完成了对训练数据的fit和transform。

把很多个功能相对单一的变换器组合起来, 或并行或串行，在 :ref:`combining_estimators` 
中讨论了。 :ref:`metrics` 包括了把特征变换为affinity矩阵的各种方法。
而 :ref:`preprocessing_targets` 则考虑了对目标值空间(e.g. categorical labels)进行变换的方法。

.. toctree::

    modules/compose
    modules/feature_extraction
    modules/preprocessing
    modules/impute
    modules/unsupervised_reduction
    modules/random_projection
    modules/kernel_approximation
    modules/metrics
    modules/preprocessing_targets
