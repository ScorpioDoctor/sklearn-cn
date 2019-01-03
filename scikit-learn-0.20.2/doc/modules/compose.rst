
.. _combining_estimators:

========================================================
管道流与复合估计器(Pipelines and composite estimators)
========================================================

.. topic:: 译者注

    Pipelines 可译为 管道流 或 流水线； composite estimators 译为 复合估计器 或 混合估计器
    本章视频地址: (`SKlearn之Pipeline和FeatureUnion <http://www.studyai.com/course/play/d97657eb8a7a45cb96b1808c1d1dc565>`_)


变换器(Transformers)通常与分类器，回归器或其他的学习器组合在一起构建成一个复合式估计器。 
完成这件事的最常用工具是 :ref:`Pipeline <pipeline>`。 Pipeline 经常与 :ref:`FeatureUnion <feature_union>` 结合起来使用。
:ref:`FeatureUnion <feature_union>` 用于将变换器(transformers)的输出串联到复合特征空间(composite feature space)中。
:ref:`TransformedTargetRegressor <transformed_target_regressor>` 用来处理变换 :term:`target` (i.e. 对数变化 :term:`y`)。
作为对比，Pipelines类只用来变换(transform)观测数据(:term:`X`)。

.. _pipeline:

Pipeline: 链式估计器
=============================

.. currentmodule:: sklearn.pipeline

:class:`Pipeline` 类可用于将多个estimators链接为一个。这是有用的，因为在处理数据时通常有固定的步骤序列，例如特征选择、规范化和分类。
:class:`Pipeline` 在这里有多种用途：

便捷性和封装性
    您只需在数据上调用 ``fit`` 和 ``predict`` 一次，就可以对整个估计器序列进行拟合。
联合参数选择
    在 pipline 中，你可以在所有估计器的参数上只进行一次 :ref:`grid search <grid_search>` 。
安全性
    在交叉验证中，Pipelines 有助于避免将测试数据中的统计数据泄漏到经过交叉验证训练的模型中，确保使用相同的样本来训练transformers和predictors。

Pipeline中的所有估计器, 除了最后一个, 必须是变换器(transformers)(i.e. 必须要有 ``transform`` 方法)。
最后一个estimator可以是任意类型的:(transformer, classifier, regressor, etc.).


用法
-----

:class:`Pipeline` 类是使用 ``(key, value)`` 对的列表(Python的list)进行构建的, 其中 ``key`` 是一个包含名字(你想给这个步骤命名的名字)的字符串
； ``value`` 则是一个 estimator 对象::

    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.svm import SVC
    >>> from sklearn.decomposition import PCA
    >>> estimators = [('reduce_dim', PCA()), ('clf', SVC())]
    >>> pipe = Pipeline(estimators)
    >>> pipe # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    Pipeline(memory=None,
             steps=[('reduce_dim', PCA(copy=True,...)),
                    ('clf', SVC(C=1.0,...))])

工具函数 :func:`make_pipeline` 是用于构建Pipeline的快捷方法， 它接受可变数量的estimators然后返回一个pipeline，自动填充 names ::

    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.naive_bayes import MultinomialNB
    >>> from sklearn.preprocessing import Binarizer
    >>> make_pipeline(Binarizer(), MultinomialNB()) # doctest: +NORMALIZE_WHITESPACE
    Pipeline(memory=None,
             steps=[('binarizer', Binarizer(copy=True, threshold=0.0)),
                    ('multinomialnb', MultinomialNB(alpha=1.0,
                                                    class_prior=None,
                                                    fit_prior=True))])

一个pipeline的所有estimators都被存放到 ``steps`` 属性，这是一个python的list ::

    >>> pipe.steps[0]
    ('reduce_dim', PCA(copy=True, iterated_power='auto', n_components=None, random_state=None,
      svd_solver='auto', tol=0.0, whiten=False))

另外 所有estimators也被以 ``dict`` 的形式放在 ``named_steps`` 属性中:: 

    >>> pipe.named_steps['reduce_dim']
    PCA(copy=True, iterated_power='auto', n_components=None, random_state=None,
      svd_solver='auto', tol=0.0, whiten=False)

pipeline中estimators的参数可以使用语法 ``<estimator>__<parameter>`` 获得:: 

    >>> pipe.set_params(clf__C=10) # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    Pipeline(memory=None,
             steps=[('reduce_dim', PCA(copy=True, iterated_power='auto',...)),
                    ('clf', SVC(C=10, cache_size=200, class_weight=None,...))])

named_steps 的属性被映射到Key上, 这样可以在交互使用的环境下使用 tab 键补全 ::

    >>> pipe.named_steps.reduce_dim is pipe.named_steps['reduce_dim']
    True

这对做网格搜索尤其重要::

    >>> from sklearn.model_selection import GridSearchCV
    >>> param_grid = dict(reduce_dim__n_components=[2, 5, 10],
    ...                   clf__C=[0.1, 10, 100])
    >>> grid_search = GridSearchCV(pipe, param_grid=param_grid)

单个的步骤也可以被作为参数进行替换, 而且 非最终步骤(non-final steps)还可以被忽略，只要将其设置为 ``None``::

    >>> from sklearn.linear_model import LogisticRegression
    >>> param_grid = dict(reduce_dim=[None, PCA(5), PCA(10)],
    ...                   clf=[SVC(), LogisticRegression()],
    ...                   clf__C=[0.1, 10, 100])
    >>> grid_search = GridSearchCV(pipe, param_grid=param_grid)

.. topic:: 案例:

 * :ref:`sphx_glr_auto_examples_feature_selection_plot_feature_selection_pipeline.py`
 * :ref:`sphx_glr_auto_examples_model_selection_grid_search_text_feature_extraction.py`
 * :ref:`sphx_glr_auto_examples_compose_plot_digits_pipe.py`
 * :ref:`sphx_glr_auto_examples_plot_kernel_approximation.py`
 * :ref:`sphx_glr_auto_examples_svm_plot_svm_anova.py`
 * :ref:`sphx_glr_auto_examples_compose_plot_compare_reduction.py`

.. topic:: 请看下面的介绍:

 * :ref:`grid_search`


Notes
-----

在pipeline上调用 ``fit`` 与依次对每个估计器调用 ``fit`` 是相同的，``transform`` 输入并将其传递给下一步。
pipeline类的对象实例拥有pipeline中最后一个estimator的所有方法。
i.e. 如果最后一个estimator是个classifier, :class:`Pipeline` 类即可被用做classifier。
如果最后一个estimator是个transformer, 那么, :class:`Pipeline` 类即可被用做transformer。

.. _pipeline_cache:

缓存 transformers: 避免重复计算
-------------------------------------------------

.. currentmodule:: sklearn.pipeline

适配 transformers 是很耗费计算资源的。设置了 ``memory`` 参数， Pipeline 将会在调用 ``fit`` 方法后缓存每个 transformer。 
如果参数和输入数据相同，Pipeline的这个特性用于避免重复计算适配好的transformer。典型的例子是网格搜索transformer，
该transformer只要适配一次就可以多次使用。

为了缓存transformers， 参数 ``memory`` 是必要的。``memory`` 可以是一个指明将在何处缓存transformers的路径字符串 
或 一个 `joblib.Memory <https://pythonhosted.org/joblib/memory.html>`_ 对象 ::

    >>> from tempfile import mkdtemp
    >>> from shutil import rmtree
    >>> from sklearn.decomposition import PCA
    >>> from sklearn.svm import SVC
    >>> from sklearn.pipeline import Pipeline
    >>> estimators = [('reduce_dim', PCA()), ('clf', SVC())]
    >>> cachedir = mkdtemp()
    >>> pipe = Pipeline(estimators, memory=cachedir)
    >>> pipe # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    Pipeline(...,
             steps=[('reduce_dim', PCA(copy=True,...)),
                    ('clf', SVC(C=1.0,...))])
    >>> # Clear the cache directory when you don't need it anymore
    >>> rmtree(cachedir)

.. warning:: ** 缓存 transformers 带来的副作用 **

   如果使用一个没有开启缓存的 :class:`Pipeline` , 可以检查原始实例，例如 ::

     >>> from sklearn.datasets import load_digits
     >>> digits = load_digits()
     >>> pca1 = PCA()
     >>> svm1 = SVC(gamma='scale')
     >>> pipe = Pipeline([('reduce_dim', pca1), ('clf', svm1)])
     >>> pipe.fit(digits.data, digits.target)
     ... # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
     Pipeline(memory=None,
              steps=[('reduce_dim', PCA(...)), ('clf', SVC(...))])
     >>> # The pca instance can be inspected directly
     >>> print(pca1.components_) # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
         [[-1.77484909e-19  ... 4.07058917e-18]]

   开启缓存会在适配前触发transformers的克隆。因此，pipline中的transformer实例不能被直接查看。 
   在下面例子中， 访问 :class:`PCA` 实例 ``pca2`` 将会引发 ``AttributeError``, 因为 ``pca2`` 是一个未适配的transformer。 
   这时应该使用属性 ``named_steps`` 来检查pipline的估计器 ::

     >>> cachedir = mkdtemp()
     >>> pca2 = PCA()
     >>> svm2 = SVC(gamma='scale')
     >>> cached_pipe = Pipeline([('reduce_dim', pca2), ('clf', svm2)],
     ...                        memory=cachedir)
     >>> cached_pipe.fit(digits.data, digits.target)
     ... # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
      Pipeline(memory=...,
               steps=[('reduce_dim', PCA(...)), ('clf', SVC(...))])
     >>> print(cached_pipe.named_steps['reduce_dim'].components_)
     ... # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
         [[-1.77484909e-19  ... 4.07058917e-18]]
     >>> # Remove the cache directory
     >>> rmtree(cachedir)

.. topic:: 案例:

 * :ref:`sphx_glr_auto_examples_compose_plot_compare_reduction.py`

.. _transformed_target_regressor:

变换回归问题的目标
=================================

:class:`TransformedTargetRegressor` 类在拟合回归模型之前会变换目标 ``y`` 。
模型的预测结果会通过一个逆向变换被重新映射回到原始的空间。
该类接受两个参数：一个是用于预测的 regressor，另一个是用于变换目标变量的 transformer::

  >>> import numpy as np
  >>> from sklearn.datasets import load_boston
  >>> from sklearn.compose import TransformedTargetRegressor
  >>> from sklearn.preprocessing import QuantileTransformer
  >>> from sklearn.linear_model import LinearRegression
  >>> from sklearn.model_selection import train_test_split
  >>> boston = load_boston()
  >>> X = boston.data
  >>> y = boston.target
  >>> transformer = QuantileTransformer(output_distribution='normal')
  >>> regressor = LinearRegression()
  >>> regr = TransformedTargetRegressor(regressor=regressor,
  ...                                   transformer=transformer)
  >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
  >>> regr.fit(X_train, y_train) # doctest: +ELLIPSIS
  TransformedTargetRegressor(...)
  >>> print('R2 score: {0:.2f}'.format(regr.score(X_test, y_test)))
  R2 score: 0.67
  >>> raw_target_regr = LinearRegression().fit(X_train, y_train)
  >>> print('R2 score: {0:.2f}'.format(raw_target_regr.score(X_test, y_test)))
  R2 score: 0.64

对一些简单的数据变换需求, 如果不想使用 Transformer object, 可以传入一对函数, 定义了变换与逆变换::

  >>> from __future__ import division
  >>> def func(x):
  ...     return np.log(x)
  >>> def inverse_func(x):
  ...     return np.exp(x)

接下来, 就可以像下面这样创建对象 ::

  >>> regr = TransformedTargetRegressor(regressor=regressor,
  ...                                   func=func,
  ...                                   inverse_func=inverse_func)
  >>> regr.fit(X_train, y_train) # doctest: +ELLIPSIS
  TransformedTargetRegressor(...)
  >>> print('R2 score: {0:.2f}'.format(regr.score(X_test, y_test)))
  R2 score: 0.65

默认情况下, 我们提供的函数会在每一次拟合的时候检查它们的可逆性，
但是，这可以通过把参数 ``check_inverse`` 设置为 ``False`` 绕过可逆性检查 ::

  >>> def inverse_func(x):
  ...     return x
  >>> regr = TransformedTargetRegressor(regressor=regressor,
  ...                                   func=func,
  ...                                   inverse_func=inverse_func,
  ...                                   check_inverse=False)
  >>> regr.fit(X_train, y_train) # doctest: +ELLIPSIS
  TransformedTargetRegressor(...)
  >>> print('R2 score: {0:.2f}'.format(regr.score(X_test, y_test)))
  R2 score: -4.50

.. note::

   变换可以通过设置 ``transformer`` 或者设置 函数对(``func`` 与 ``inverse_func``) 被激发。但是这两个参数不能同时设置，否则会抛出错误。
   

.. topic:: 案例:

 * :ref:`sphx_glr_auto_examples_compose_plot_transformed_target.py`


.. _feature_union:

FeatureUnion: 复合特征空间
======================================

.. currentmodule:: sklearn.pipeline

类 :class:`FeatureUnion` 可以把若干个transformer objects组合起来形成一个新的transformer，
这个新的transformer可以把它们的输出全部组合起来。类 :class:`FeatureUnion` 接受一个transformer objects 构成的python list。
在拟合期间，这些transformers中的每一个都是独立的在数据上适配。这些transformers被并行的使用，
而且他们输出的特征矩阵会被一个挨着一个的串接成更大的矩阵。

如果你想在数据的每个域上应用不同的变换，请看相关类 :class:`sklearn.compose.ColumnTransformer` (请看 :ref:`user guide <column_transformer>`).

:class:`FeatureUnion` 类与 :class:`Pipeline` 类的目的一样 - 都是为了方便性和联合参数估计与验证。

:class:`FeatureUnion` 和 :class:`Pipeline` 可以结合起来创建更复杂的模型。

类 :class:`FeatureUnion` 不会去检查两个transformers是否会产生相同的特征输出，这是调用者的责任。


用法
-----

:class:`FeatureUnion` 类 使用 ``(key, value)`` 对的列表来构造，其中 ``key`` 是你给一个给定的变换所起的名字
(可以是任意的字符串，能够起到一个identifier的作用就可以了)，``value`` 是一个 estimator object::

    >>> from sklearn.pipeline import FeatureUnion
    >>> from sklearn.decomposition import PCA
    >>> from sklearn.decomposition import KernelPCA
    >>> estimators = [('linear_pca', PCA()), ('kernel_pca', KernelPCA())]
    >>> combined = FeatureUnion(estimators)
    >>> combined # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    FeatureUnion(n_jobs=None,
                 transformer_list=[('linear_pca', PCA(copy=True,...)),
                                   ('kernel_pca', KernelPCA(alpha=1.0,...))],
                 transformer_weights=None)


就像 pipelines 一样, feature unions 也有一个快捷的构造器，称之为 :func:`make_union` 。使用这个函数不需要给每个组件显示的命名。


像 ``Pipeline`` 一样, 每个单独的步骤可以通过 ``set_params`` 进行置换, 也可以通过设置 ``'drop'`` 忽略某个步骤 ::

    >>> combined.set_params(kernel_pca='drop')
    ... # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    FeatureUnion(n_jobs=None,
                 transformer_list=[('linear_pca', PCA(copy=True,...)),
                                   ('kernel_pca', 'drop')],
                 transformer_weights=None)

.. topic:: 案列:

 * :ref:`sphx_glr_auto_examples_compose_plot_feature_union.py`


.. _column_transformer:

异构数据变换器:ColumnTransformer
========================================

.. warning::

    类 :class:`compose.ColumnTransformer <sklearn.compose.ColumnTransformer>`
    还在实验中，而且它的 API 可能会变动的。

许多数据集包含不同类型的特性，比如文本、浮点数和日期，每种类型的特征都需要单独的预处理或特征提取步骤。
通常，在应用scikit-learn方法之前，最容易的是对数据进行预处理，例如 `pandas <http://pandas.pydata.org/>`__。
在将数据传递给scikit-learn之前处理数据可能会出现问题，原因如下:

1. 将来自测试数据的统计信息集成到预处理程序中，使得交叉验证分数不可靠(被称为数据泄露 *data leakage*)。
   例如，在尺度变换或计算缺失值的情况下。
2. 你可能想要在 :ref:`parameter search <grid_search>` 中包含预处理器参数。

类 :class:`~sklearn.compose.ColumnTransformer` 对数据的不同列执行不同的变换, 
within a :class:`~sklearn.pipeline.Pipeline` that is safe from data leakage and that can
be parametrized. 类 :class:`~sklearn.compose.ColumnTransformer` 可以在 arrays, sparse matrices, 和
`pandas DataFrames <http://pandas.pydata.org/pandas-docs/stable/>`__ 等各种数据上工作。

对每一列，都会应用一个不同的变换, 比如 preprocessing 或 某个特定的特征抽取方法::

  >>> import pandas as pd
  >>> X = pd.DataFrame(
  ...     {'city': ['London', 'London', 'Paris', 'Sallisaw'],
  ...      'title': ["His Last Bow", "How Watson Learned the Trick",
  ...                "A Moveable Feast", "The Grapes of Wrath"],
  ...      'expert_rating': [5, 3, 4, 5],
  ...      'user_rating': [4, 5, 4, 3]})

针对以上数据, 我们会想把 ``'city'`` 列编码成标称型变量(categorical variable), 
而对 ``'title'`` 列使用类 :class:`feature_extraction.text.CountVectorizer <sklearn.feature_extraction.text.CountVectorizer>` 
进行变换。
由于我们可能会把多个特征抽取器用在同一列上, 我们给每一个变换器 transformer 取一个唯一的名字，称之为 ``'city_category'`` 和 ``'title_bow'``。
默认情况下，剩余的 rating columns 就会被忽略(``remainder='drop'``)。::

  >>> from sklearn.compose import ColumnTransformer
  >>> from sklearn.feature_extraction.text import CountVectorizer
  >>> column_trans = ColumnTransformer(
  ...     [('city_category', CountVectorizer(analyzer=lambda x: [x]), 'city'),
  ...      ('title_bow', CountVectorizer(), 'title')],
  ...     remainder='drop')

  >>> column_trans.fit(X) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
  ColumnTransformer(n_jobs=None, remainder='drop', sparse_threshold=0.3,
      transformer_weights=None,
      transformers=...)

  >>> column_trans.get_feature_names()
  ... # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
  ['city_category__London', 'city_category__Paris', 'city_category__Sallisaw',
  'title_bow__bow', 'title_bow__feast', 'title_bow__grapes', 'title_bow__his',
  'title_bow__how', 'title_bow__last', 'title_bow__learned', 'title_bow__moveable',
  'title_bow__of', 'title_bow__the', 'title_bow__trick', 'title_bow__watson',
  'title_bow__wrath']

  >>> column_trans.transform(X).toarray()
  ... # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
  array([[1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0],
         [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1]]...)

在上面的例子中, 类 :class:`~sklearn.feature_extraction.text.CountVectorizer` 希望接受一个 1D array 作为输入，
并且因此列被指定为一个字符串(``'city'``)。然而其他变换器希望接受 2D 数据，在此情形下，你需要将列指定为字符串组成的列表 (``['city']``)。 

Apart from a scalar or a single item list, the column selection can be specified
as a list of multiple items, an integer array, a slice, or a boolean mask.
Strings can reference columns if the input is a DataFrame, integers are always
interpreted as the positional columns.

We can keep the remaining rating columns by setting
``remainder='passthrough'``. The values are appended to the end of the
transformation::

  >>> column_trans = ColumnTransformer(
  ...     [('city_category', CountVectorizer(analyzer=lambda x: [x]), 'city'),
  ...      ('title_bow', CountVectorizer(), 'title')],
  ...     remainder='passthrough')

  >>> column_trans.fit_transform(X)
  ... # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
  array([[1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 5, 4],
         [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 3, 5],
         [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 4, 4],
         [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 5, 3]]...)

``remainder`` 参数可以被设置成一个estimator来变换 剩余的 rating columns。 变换以后的值被追加到变换的末尾 ::

  >>> from sklearn.preprocessing import MinMaxScaler
  >>> column_trans = ColumnTransformer(
  ...     [('city_category', CountVectorizer(analyzer=lambda x: [x]), 'city'),
  ...      ('title_bow', CountVectorizer(), 'title')],
  ...     remainder=MinMaxScaler())

  >>> column_trans.fit_transform(X)[:, -2:]
  ... # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
  array([[1. , 0.5],
         [0. , 1. ],
         [0.5, 0.5],
         [1. , 0. ]])

函数 :func:`~sklearn.compose.make_column_transformer` 可用来更简单的创建类对象 :class:`~sklearn.compose.ColumnTransformer` 。
特别的，名字将会被自动给出。上面的例子等价于 ::

  >>> from sklearn.compose import make_column_transformer
  >>> column_trans = make_column_transformer(
  ...     (CountVectorizer(analyzer=lambda x: [x]), 'city'),
  ...     (CountVectorizer(), 'title'),
  ...     remainder=MinMaxScaler())
  >>> column_trans # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
  ColumnTransformer(n_jobs=None, remainder=MinMaxScaler(copy=True, ...),
           sparse_threshold=0.3,
           transformer_weights=None,
           transformers=[('countvectorizer-1', ...)

.. topic:: 案例:

 * :ref:`sphx_glr_auto_examples_compose_plot_column_transformer.py`
 * :ref:`sphx_glr_auto_examples_compose_plot_column_transformer_mixed_types.py`
