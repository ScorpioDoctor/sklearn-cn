.. currentmodule:: sklearn.preprocessing

.. _preprocessing_targets:

==========================================
变换预测目标 (``y``)
==========================================

本章要介绍的这些变换器(transformers)不是被用于特征的，而是只被用于变换监督学习的目标(targets)。
如果你希望变换预测目标以进行学习，但是在原始（未变换）空间中评估模型，请看 :ref:`transformed_target_regressor` 。

标签二值化
------------------

:class:`LabelBinarizer` 类是一个工具类用于从多类标签的列表创建一个标签指示器矩阵 ::

    >>> from sklearn import preprocessing
    >>> lb = preprocessing.LabelBinarizer()
    >>> lb.fit([1, 2, 6, 4, 2])
    LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)
    >>> lb.classes_
    array([1, 2, 4, 6])
    >>> lb.transform([1, 6])
    array([[1, 0, 0, 0],
           [0, 0, 0, 1]])

对于每个样本实例具有多个标签的情况, 使用 :class:`MultiLabelBinarizer`::

    >>> lb = preprocessing.MultiLabelBinarizer()
    >>> lb.fit_transform([(1, 2), (3,)])
    array([[1, 1, 0],
           [0, 0, 1]])
    >>> lb.classes_
    array([1, 2, 3])


标签编码
--------------

:class:`LabelEncoder` 类 是一个工具类可以用它来归一化标签使得标签只包含从0到n_classes-1的值。
这在有些时候很有用,比如写一个高效的Cython程序。 :class:`LabelEncoder` 类的用法如下所示 ::

    >>> from sklearn import preprocessing
    >>> le = preprocessing.LabelEncoder()
    >>> le.fit([1, 2, 2, 6])
    LabelEncoder()
    >>> le.classes_
    array([1, 2, 6])
    >>> le.transform([1, 1, 2, 6])
    array([0, 0, 1, 2])
    >>> le.inverse_transform([0, 0, 1, 2])
    array([1, 1, 2, 6])

它也可以被用来把 非数值型标签(non-numerical labels) (只要这些非数值型标签是可哈希的和可比较的(hashable and comparable))
变换成 数值型标签(numerical labels) ::

    >>> le = preprocessing.LabelEncoder()
    >>> le.fit(["paris", "paris", "tokyo", "amsterdam"])
    LabelEncoder()
    >>> list(le.classes_)
    ['amsterdam', 'paris', 'tokyo']
    >>> le.transform(["tokyo", "tokyo", "paris"])
    array([2, 2, 1])
    >>> list(le.inverse_transform([2, 2, 1]))
    ['tokyo', 'tokyo', 'paris']
