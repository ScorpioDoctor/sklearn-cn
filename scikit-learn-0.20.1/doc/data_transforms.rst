.. include:: includes/big_toc_css.rst

.. _data-transforms:

数据集变换(Dataset transformations)
-----------------------

scikit-learn provides a library of transformers, which may clean (see
:ref:`preprocessing`), reduce (see :ref:`data_reduction`), expand (see
:ref:`kernel_approximation`) or generate (see :ref:`feature_extraction`)
feature representations.

Like other estimators, these are represented by classes with a ``fit`` method,
which learns model parameters (e.g. mean and standard deviation for
normalization) from a training set, and a ``transform`` method which applies
this transformation model to unseen data. ``fit_transform`` may be more
convenient and efficient for modelling and transforming the training data
simultaneously.

Combining such transformers, either in parallel or series is covered in
:ref:`combining_estimators`. :ref:`metrics` covers transforming feature
spaces into affinity matrices, while :ref:`preprocessing_targets` considers
transformations of the target space (e.g. categorical labels) for use in
scikit-learn.

.. topic:: 译者注

    视频地址: (`SKLearn数据集变换操作--综述 <http://www.studyai.com/course/play/ae8f595cb5af41cba23969a186c19009>`_)


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
