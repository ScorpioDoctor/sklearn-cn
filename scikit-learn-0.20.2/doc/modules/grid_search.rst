

.. currentmodule:: sklearn.model_selection

.. _grid_search:

===========================================
通过网格搜索调节估计器超参数
===========================================

.. topic:: 译者注

    这一节的重点是Sklearn超参数优化调节方法，再次献上我做的视频，希望对大家有所帮助。视频地址：
    (`Sklearn超参数优化调节方法与网格搜索 <http://www.studyai.com/course/play/41a5ec5505994bf2abeefe02b5cf60ef>`_)

超参数(Hyper-parameters)，即不直接在估计器内学习的参数。在 scikit-learn 包中，它们作为估计器类中构造函数的参数进行传递。
典型的例子有：用于支持向量分类器的 ``C``, ``kernel`` 和 ``gamma`` ，用于Lasso的 ``alpha`` 等。 

搜索超参数空间(hyper-parameter space)以便获得最好 :ref:`交叉验证 <cross_validation>` 分数的方法是可能的而且是值得提倡的。

通过这种方式，构造估计器时被提供的任何参数或许都能被优化。具体来说，要获取到给定估计器的所有参数的名称和当前值，使用 ::

  estimator.get_params()

搜索包括:

- 估计器(回归器或分类器，例如 ``sklearn.svm.SVC()``);
- 参数空间;
- 搜寻或对候选集合采样的方法;
- 交叉验证方案; 和
- :ref:`评分函数 <gridsearch_scoring>` .

有些模型支持专业化的、高效的参数搜索策略, :ref:`罗列如下 <alternative_cv>` 。在 scikit-learn 包中提供了两种采样搜索候选的通用方法:对于给定的值, 
:class:`GridSearchCV` 考虑了所有参数组合；而 :class:`RandomizedSearchCV` 可以从具有指定分布的参数空间中抽取给定数量的候选。介绍完这些工具后，
我们将详细介绍适用于这两种方法的 :ref:`最佳实践 <grid_search_tips>` 。

注意，通常这些参数的一小部分会对模型的预测或计算性能有很大的影响，而其他参数可以保留为其默认值。 
建议阅读估计器类的相关文档，以更好地了解其预期行为，
可能的话还可以阅读下引用的文献。

穷举方式的网格搜索
======================

:class:`GridSearchCV` 类提供的网格搜索从通过 ``param_grid`` 参数确定的网格参数值中穷举生成候选参数组合。例如，下面的 ``param_grid``  ::

  param_grid = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
   ]

探索两个参数网格的详细解释： 一个具有线性内核并且C在[1,10,100,1000]中取值； 
另一个具有RBF内核，C值的交叉乘积范围在[1,10，100,1000]，gamma在[0.001，0.0001]中取值。

:class:`GridSearchCV` 实例实现了常用估计器 API：当在数据集上“拟合”时，所有可能的参数组合都会被评估，
从而计算出最佳的参数组合。

.. currentmodule:: sklearn.model_selection

.. topic:: 案例:

    - See :ref:`sphx_glr_auto_examples_model_selection_plot_grid_search_digits.py` for an example of
      Grid Search computation on the digits dataset.

    - See :ref:`sphx_glr_auto_examples_model_selection_grid_search_text_feature_extraction.py` for an example
      of Grid Search coupling parameters from a text documents feature
      extractor (n-gram count vectorizer and TF-IDF transformer) with a
      classifier (here a linear SVM trained with SGD with either elastic
      net or L2 penalty) using a :class:`pipeline.Pipeline` instance.

    - See :ref:`sphx_glr_auto_examples_model_selection_plot_nested_cross_validation_iris.py`
      for an example of Grid Search within a cross validation loop on the iris
      dataset. This is the best practice for evaluating the performance of a
      model with grid search.

    - See :ref:`sphx_glr_auto_examples_model_selection_plot_multi_metric_evaluation.py`
      for an example of :class:`GridSearchCV` being used to evaluate multiple
      metrics simultaneously.

.. _randomized_parameter_search:

随机化参数优化
=================================
尽管使用参数设置的网格法是目前最广泛使用的参数优化方法, 其他搜索方法也具有更有利的性能。 
:class:`RandomizedSearchCV` 实现了对参数的随机搜索, 其中每个设置都是从可能的参数值的分布中进行取样。 这相对于穷举搜索有两个主要优势:

* 可以选择独立于参数个数和可能值的预算.
* 添加不影响性能的参数不会降低效率

指定参数的抽样方法是使用字典完成的, 非常类似于为 :class:`GridSearchCV` 指定参数。 
此外, 通过 ``n_iter`` 参数指定计算预算, 即取样候选项数或取样迭代次数。 
对于每个参数, 可以指定在可能值上的分布或离散选择的列表 (均匀取样)::

  {'C': scipy.stats.expon(scale=100), 'gamma': scipy.stats.expon(scale=.1),
    'kernel': ['rbf'], 'class_weight':['balanced', None]}

本示例使用 ``scipy.stats`` 模块, 它包含许多用于采样参数的有用分布, 如 ``expon``，``gamma``，``uniform`` 或者 ``randint``。 
原则上, 任何函数都可以通过提供一个 ``rvs`` (random variate sample: 随机变量样本)方法来采样一个值。 
对 ``rvs`` 函数的调用应在连续调用中提供来自可能参数值的独立随机样本。

    .. warning::

        The distributions in ``scipy.stats`` prior to version scipy 0.16
        do not allow specifying a random state. Instead, they use the global
        numpy random state, that can be seeded via ``np.random.seed`` or set
        using ``np.random.set_state``. However, beginning scikit-learn 0.18,
        the :mod:`sklearn.model_selection` module sets the random state provided
        by the user if scipy >= 0.16 is also available.

对于连续参数 (如上面提到的 ``C`` )，指定连续分布以充分利用随机化是很重要的。这样，有助于 ``n_iter`` 总是趋向于更精细的搜索。

.. topic:: 案例:

    * :ref:`sphx_glr_auto_examples_model_selection_plot_randomized_search.py` compares the usage and efficiency
      of randomized search and grid search.

.. topic:: 参考文献:

    * Bergstra, J. and Bengio, Y.,
      Random search for hyper-parameter optimization,
      The Journal of Machine Learning Research (2012)

.. _grid_search_tips:

参数搜索的小技巧
=========================

.. _gridsearch_scoring:

指定一个目标测度
------------------------------

默认情况下, 参数搜索使用估计器的评分函数(``score`` function)来评估（衡量）参数设置。
比如 :func:`sklearn.metrics.accuracy_score` 用于分类和 :func:`sklearn.metrics.r2_score` 用于回归。 
对于一些应用, 其他评分函数将会更加适合 (例如在不平衡的分类问题中, 精度评分往往是信息不足的)。
一个可选的评分函数可以通过评分参数 ``scoring`` 指定给 :class:`GridSearchCV`, :class:`RandomizedSearchCV` 
和许多下文将要描述的、专业化的交叉验证工具。 
有关详细信息, 请参阅 :ref:`scoring_parameter` 。

.. _multimetric_grid_search:

指定多个测度用于评估
------------------------------------------

``GridSearchCV`` 和 ``RandomizedSearchCV`` 允许为评分参数 ``scoring`` 指定多个测度指标(metrics)。

多指标评分(Multimetric scoring)可以被指定为一个预先定义的评分器名称(scorer name)的字符串列表或者
是一个把评分器名称映射到评分函数或预先定义的评分器的字典。
有关详细信息, 请参阅 :ref:`multimetric_scoring` 。

在指定多个指标时,必须将 ``refit`` 参数设置为要在其中找到 ``best_params_``,并用于在整个数据集上构建 ``best_estimator_`` 的度量标准（字符串）。 
如果搜索不应该refit, 则设置 ``refit=False`` 。在使用多指标评分时,如果将 refit 保留为默认值 ``None``, 会导致结果错误。

See :ref:`sphx_glr_auto_examples_model_selection_plot_multi_metric_evaluation.py`
for an example usage.

组合不同估计器和参数空间
-----------------------------------------

:ref:`pipeline` 小节描述了如何使用这些工具搜索参数空间构建组合式评估器。

模型选择: 开发与评估
-------------------------------------------

通过评估各种参数设置，可以将模型选择视为使用标记数据 "训练" 网格参数的一种方法。 

在评估得到的模型时, 重要的是在网格搜索过程中未看到的留出的(held-out)样本数据上执行以下操作: 
建议将数据拆分为开发集 (**development set**,供 ``GridSearchCV`` 实例使用)
和评估集(**evaluation set**)来计算性能指标。

这可以通过使用函数  :func:`train_test_split` 来完成。 

并行化
-----------

:class:`GridSearchCV`  和 :class:`RandomizedSearchCV`  可以独立地评估每个参数设置。
如果您的OS支持,通过使用关键字 ``n_jobs=-1`` 可以使计算并行运行。 
有关详细信息, 请参见函数签名。

对失败保持鲁棒性
---------------------

某些参数设置可能导致无法 ``fit`` 数据的一个或多个folds。 默认情况下, 这将导致整个搜索失败, 即使某些参数设置可以完全计算。 
设置 ``error_score=0`` (或 `=np.NaN` ) 将使程序对此类故障具有鲁棒性,发出警告并将该折叠的分数设置为0(或 `NaN` ), 但可以完成搜索。

.. _alternative_cv:

暴力参数搜索法的替代法
============================================

模型特定的交叉验证
-------------------------------


某些模型可以与参数的单个值的估计值一样有效地适应某一参数范围内的数据。 此功能可用于执行更有效的交叉验证, 用于此参数的模型选择。

该策略最常用的参数是编码正则化因子强度的参数。在这种情况下, 我们称之为, 计算估计器的正则化路径( **regularization path** )。

下面是这些特定模型的列表:

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   linear_model.ElasticNetCV
   linear_model.LarsCV
   linear_model.LassoCV
   linear_model.LassoLarsCV
   linear_model.LogisticRegressionCV
   linear_model.MultiTaskElasticNetCV
   linear_model.MultiTaskLassoCV
   linear_model.OrthogonalMatchingPursuitCV
   linear_model.RidgeCV
   linear_model.RidgeClassifierCV


信息准则
---------------------

一些模型通过计算单个正则化路径 (代替使用交叉验证得出数个参数), 可以给出正则化参数最优估计的信息理论闭包公式。
Some models can offer an information-theoretic closed-form formula of the
optimal estimate of the regularization parameter by computing a single
regularization path (instead of several when using cross-validation).

以下是从 Akaike Information Criterion (AIC) 或 Bayesian Information Criterion(BIC) (可用于自动选择模型) 中受益的模型列表:

.. autosummary::
   :toctree: generated/
   :template: class.rst

   linear_model.LassoLarsIC


.. _out_of_bag:

Out of Bag Estimates
--------------------

当使用基于装袋(bagging)的集成方法(ensemble methods)时，i.e. 使用替换式采样产生新的训练集，部分训练集保持不用。 
对于集合中的每个分类器，训练集的不同部分被忽略。

这个被忽略的部分可以用来估计泛化误差，而不必依靠单独的验证集。 此估计是”免费的”，因为不需要额外的数据，可以用于模型选择。

目前已经实现该方法的类有以下几个:

.. autosummary::
   :toctree: generated/
   :template: class.rst

    ensemble.RandomForestClassifier
    ensemble.RandomForestRegressor
    ensemble.ExtraTreesClassifier
    ensemble.ExtraTreesRegressor
    ensemble.GradientBoostingClassifier
    ensemble.GradientBoostingRegressor
