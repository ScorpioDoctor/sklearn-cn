.. _impute:

============================
缺失值处理(Imputation of missing values)
============================

.. currentmodule:: sklearn.impute

由于各种原因，许多真实世界的数据集包含缺失的值，通常编码为空白、NAN或其他占位符。
然而，这类数据集与scikit-Learn的估计器不兼容，后者假定数组中的所有值都是数字的，
而且都具有并保持着意义。使用不完整数据集的基本策略是丢弃包含缺失值的整行和/或列。
然而，这是以损失可能有价值(即使不完整)的数据为代价的。一个更好的策略是估算缺失的值，
即从数据的已知部分推断它们。请查看 :ref:`glossary` 里面对 imputation 解释。

:class:`SimpleImputer` 类提供了估算缺失值的基本策略。缺失的值可以用提供的常量值来计算，
或使用缺失值所在的每一列的统计数据(平均值、中位数或最频繁)。该类还允许不同的缺失值编码。

下面的代码片段演示如何 使用包含缺失值的列(axis 0)的平均值 替换丢失的值，编码为 ``np.nan`` ::

    >>> import numpy as np
    >>> from sklearn.impute import SimpleImputer
    >>> imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    >>> imp.fit([[1, 2], [np.nan, 3], [7, 6]])       # doctest: +NORMALIZE_WHITESPACE
    SimpleImputer(copy=True, fill_value=None, missing_values=nan, strategy='mean', verbose=0)
    >>> X = [[np.nan, 2], [6, np.nan], [7, 6]]
    >>> print(imp.transform(X))           # doctest: +NORMALIZE_WHITESPACE  +ELLIPSIS
    [[4.          2.        ]
     [6.          3.666...]
     [7.          6.        ]]

:class:`SimpleImputer` 类还支持稀疏矩阵::

    >>> import scipy.sparse as sp
    >>> X = sp.csc_matrix([[1, 2], [0, -1], [8, 4]])
    >>> imp = SimpleImputer(missing_values=-1, strategy='mean')
    >>> imp.fit(X)                  # doctest: +NORMALIZE_WHITESPACE
    SimpleImputer(copy=True, fill_value=None, missing_values=-1, strategy='mean', verbose=0)
    >>> X_test = sp.csc_matrix([[-1, 2], [6, -1], [7, 6]])
    >>> print(imp.transform(X_test).toarray())      # doctest: +NORMALIZE_WHITESPACE
    [[3. 2.]
     [6. 3.]
     [7. 6.]]

请注意，此格式不打算用于隐式存储矩阵中缺少的值，因为它将在转换时对其进行加密。由0编码的缺失值必须与密集输入(dense input)一起使用。

当使用 ``'most_frequent'`` 或 ``'constant'`` 策略时，:class:`SimpleImputer` 类还支持以 
string values 或 pandas categoricals 表示的分类数据(categorical data) ::

    >>> import pandas as pd
    >>> df = pd.DataFrame([["a", "x"],
    ...                    [np.nan, "y"],
    ...                    ["a", np.nan],
    ...                    ["b", "y"]], dtype="category")
    ...
    >>> imp = SimpleImputer(strategy="most_frequent")
    >>> print(imp.fit_transform(df))      # doctest: +NORMALIZE_WHITESPACE
    [['a' 'x']
     ['a' 'y']
     ['a' 'y']
     ['b' 'y']]


:class:`SimpleImputer` 类可以在管道中作为一种方法来构建一个支持imputation的复合估计器。
请看 :ref:`sphx_glr_auto_examples_plot_missing_values.py`.

.. _missing_indicator:

标记缺失值(Marking imputed values)
======================

:class:`MissingIndicator` transformer 用于将数据集转换为相应的二进制矩阵，指示数据集中是否存在缺失值。
这种转换与计算相结合是很有用的。在使用估算时，保存有关哪些值丢失的信息可以提供信息。

``NaN`` 通常用作缺少值的占位符。但是，它强制数据类型为浮点数。参数 ``missing_values`` 允许指定其他占位符，如整数。
在以下示例中，我们将使用-1作为缺失值 ::

  >>> from sklearn.impute import MissingIndicator
  >>> X = np.array([[-1, -1, 1, 3],
  ...               [4, -1, 0, -1],
  ...               [8, -1, 1, 0]])
  >>> indicator = MissingIndicator(missing_values=-1)
  >>> mask_missing_values_only = indicator.fit_transform(X)
  >>> mask_missing_values_only
  array([[ True,  True, False],
         [False,  True,  True],
         [False,  True, False]])

参数 ``features`` 用于选择构造掩码的特征。默认情况下，它是 ``'missing-only'`` ，
它在 ``fit`` 时返回包含缺失值的特征的输入掩码 ::

  >>> indicator.features_
  array([0, 1, 3])

参数 ``features`` 可以设置为 ``'all'`` 以返回所有特征，无论它们是否包含缺失的值 ::
    
  >>> indicator = MissingIndicator(missing_values=-1, features="all")
  >>> mask_all = indicator.fit_transform(X)
  >>> mask_all
  array([[ True,  True, False, False],
         [False,  True, False,  True],
         [False,  True, False, False]])
  >>> indicator.features_
  array([0, 1, 2, 3])

当在 :class:`Pipeline` 中使用 :class:`MissingIndicator` 类时, 务必使用 :class:`FeatureUnion` 类
或 :class:`ColumnTransformer` 类来添加 indicator features 到 regular features. 
首先，我们获得虹膜(`iris`)数据集，并给它添加一些缺失值::

  >>> from sklearn.datasets import load_iris
  >>> from sklearn.impute import SimpleImputer, MissingIndicator
  >>> from sklearn.model_selection import train_test_split
  >>> from sklearn.pipeline import FeatureUnion, make_pipeline
  >>> from sklearn.tree import DecisionTreeClassifier
  >>> X, y = load_iris(return_X_y=True)
  >>> mask = np.random.randint(0, 2, size=X.shape).astype(np.bool)
  >>> X[mask] = np.nan
  >>> X_train, X_test, y_train, _ = train_test_split(X, y, test_size=100,
  ...                                                random_state=0)

现在我们创建一个 :class:`FeatureUnion` 。为了使分类器能够处理这些数据，所有的特征都将使用 :class:`SimpleImputer` 进行估算。
此外，它还从 :class:`MissingIndicator` 中添加指示变量(indicator variables)。

  >>> transformer = FeatureUnion(
  ...     transformer_list=[
  ...         ('features', SimpleImputer(strategy='mean')),
  ...         ('indicators', MissingIndicator())])
  >>> transformer = transformer.fit(X_train, y_train)
  >>> results = transformer.transform(X_test)
  >>> results.shape
  (100, 8)

当然，我们不能用 transformer 来做任何预测。我们应该用分类器(例如，:class:`DecisionTreeClassifier` )将其封装在pipeline中，
以便能够进行预测。

  >>> clf = make_pipeline(transformer, DecisionTreeClassifier())
  >>> clf = clf.fit(X_train, y_train)
  >>> results = clf.predict(X_test)
  >>> results.shape
  (100,)

