.. _linear_model:

=========================
广义线性模型(Generalized Linear Models)
=========================

.. currentmodule:: sklearn.linear_model

以下是一组用于回归的方法，其中目标值(target value)被希望(认为)是输入变量的线性组合(input
variables)。 用数学语言来描述是这样的： 假设 :math:`\hat{y}` 是目标值的预测值，则有

.. math::    \hat{y}(w, x) = w_0 + w_1 x_1 + ... + w_p x_p

在整个模块中, 我们把向量 :math:`w = (w_1,..., w_p)` 记作 ``coef_`` (系数)，并把 :math:`w_0` 记作 ``intercept_`` (截距).

如果你想用正则化线性模型求解分类问题, 请参考 :ref:`Logistic_regression`。


.. _ordinary_least_squares:

普通最小二乘法
=======================

:class:`LinearRegression` 拟合一个带有系数 :math:`w = (w_1, ..., w_p)` 的线性模型使得数据集实际观测数据和预测数据（估计值）之间的残差平方和最小。
其数学表达式为:

.. math:: \min_{w} {|| X w - y||_2}^2

.. figure:: ../auto_examples/linear_model/images/sphx_glr_plot_ols_001.png
   :target: ../auto_examples/linear_model/plot_ols.html
   :align: center
   :scale: 50%

:class:`LinearRegression` 会调用 ``fit`` 方法来拟合数组 X， y，并且将线性模型的系数 :math:`w` 存储在其成员变量 ``coef_`` 中::

    >>> from sklearn import linear_model
    >>> reg = linear_model.LinearRegression()
    >>> reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
    ...                                       # doctest: +NORMALIZE_WHITESPACE
    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,
                     normalize=False)
    >>> reg.coef_
    array([0.5, 0.5])

然而，对于普通最小二乘的系数估计问题，其依赖于模型各项的相互独立性。
当各项是相关的，且设计矩阵(design matrix) :math:`X` 的各列近似线性相关，
那么，设计矩阵会趋向于奇异矩阵，这会导致最小二乘估计对于随机误差非常敏感，产生很大的方差。
例如，在没有实验设计的情况下收集到的数据，这种多重共线性（multicollinearity）的情况可能真的会出现。

.. topic:: 案例:

   * :ref:`sphx_glr_auto_examples_linear_model_plot_ols.py`


普通最小二乘法的复杂度
---------------------------------

该方法使用 X 的奇异值分解来计算最小二乘解。如果 X 是一个 size 为 (n, p) 的矩阵，设 :math:`n \geq p` ，则该方法的复杂度为 :math:`O(n p^2)`.

.. _ridge_regression:

岭回归(Ridge Regression)
================

:class:`Ridge` regression 通过对系数的大小施加惩罚来解决 :ref:`ordinary_least_squares` 的一些问题。 
岭系数最小化的是带惩罚项的残差平方和，数学形式如下


.. math::

   \min_{w} {{|| X w - y||_2}^2 + \alpha {||w||_2}^2}


其中, :math:`\alpha \geq 0` 是一个控制缩减量(amount of shrinkage)的复杂度参数:
:math:`\alpha` 的值越大, 缩减量就越大，故而线性模型的系数对共线性(collinearity)就越鲁棒.

.. figure:: ../auto_examples/linear_model/images/sphx_glr_plot_ridge_path_001.png
   :target: ../auto_examples/linear_model/plot_ridge_path.html
   :align: center
   :scale: 50%


与其他线性模型一样, :class:`Ridge` 会调用 ``fit`` 方法来拟合数组 X， y，并且将线性模型的系数 :math:`w` 存储在其成员变量 ``coef_`` 中::

    >>> from sklearn import linear_model
    >>> reg = linear_model.Ridge(alpha=.5)
    >>> reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1]) # doctest: +NORMALIZE_WHITESPACE
    Ridge(alpha=0.5, copy_X=True, fit_intercept=True, max_iter=None,
          normalize=False, random_state=None, solver='auto', tol=0.001)
    >>> reg.coef_
    array([0.34545455, 0.34545455])
    >>> reg.intercept_ #doctest: +ELLIPSIS
    0.13636...


.. topic:: 案例:

   * :ref:`sphx_glr_auto_examples_linear_model_plot_ridge_path.py`
   * :ref:`sphx_glr_auto_examples_text_plot_document_classification_20newsgroups.py`


岭回归的复杂度
----------------

这种方法的复杂度与 :ref:`ordinary_least_squares` 的复杂度是相同的.

.. FIXME:
.. Not completely true: OLS is solved by an SVD, while Ridge is solved by
.. the method of normal equations (Cholesky), there is a big flop difference
.. between these


设置正则化参数: 广义交叉验证
------------------------------------------------------------------

:class:`RidgeCV` 通过内置的 alpha 参数的交叉验证来实现岭回归。 该对象与 GridSearchCV 的使用方法相同，
只是它默认为 Generalized Cross-Validation(广义交叉验证 GCV)，这是一种有效的留一验证方法（LOO-CV）::

    >>> from sklearn import linear_model
    >>> reg = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0], cv=3)
    >>> reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])       # doctest: +SKIP
    RidgeCV(alphas=[0.1, 1.0, 10.0], cv=3, fit_intercept=True, scoring=None,
        normalize=False)
    >>> reg.alpha_                                      # doctest: +SKIP
    0.1

.. topic:: 参考文献

    * "Notes on Regularized Least Squares", Rifkin & Lippert (`technical report
      <http://cbcl.mit.edu/publications/ps/MIT-CSAIL-TR-2007-025.pdf>`_,
      `course slides
      <https://www.mit.edu/~9.520/spring07/Classes/rlsslides.pdf>`_).


.. _lasso:

Lasso
=====

:class:`Lasso` 是一个线性模型，它给出的模型具有稀疏的系数(sparse coefficients)。
它在一些场景中是狠有用的，因为它倾向于使用具有较少参数值的情况，能够有效地减少给定解决方案所依赖变量的数量。 
因此，Lasso 及其变体是压缩感知(compressed sensing)领域的基础。在某些特定条件下, 它能够恢复非零权重的精确解。
(请参见 :ref:`sphx_glr_auto_examples_applications_plot_tomography_l1_reconstruction.py`).

在数学公式表达上，它由一个带有 :math:`\ell_1` 先验的正则项的线性模型组成。 其最小化的目标函数是:

.. math::  \min_{w} { \frac{1}{2n_{samples}} ||X w - y||_2 ^ 2 + \alpha ||w||_1}

lasso estimator 解决了加上惩罚项 :math:`\alpha ||w||_1` 的最小二乘的最小化，其中，
:math:`\alpha` 是一个常数，:math:`||w||_1` 是参数向量的 :math:`\ell_1`-norm 范数。

:class:`Lasso` 类的实现使用了 coordinate descent （坐标下降算法）来拟合系数。 另一种实现方法在 :ref:`least_angle_regression` 中::

    >>> from sklearn import linear_model
    >>> reg = linear_model.Lasso(alpha=0.1)
    >>> reg.fit([[0, 0], [1, 1]], [0, 1])
    Lasso(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000,
       normalize=False, positive=False, precompute=False, random_state=None,
       selection='cyclic', tol=0.0001, warm_start=False)
    >>> reg.predict([[1, 1]])
    array([0.8])

对于较低级别的任务，同样有用的是函数 :func:`lasso_path` 。它能够通过搜索所有可能的路径上的值来计算系数。

.. topic:: 案例:

  * :ref:`sphx_glr_auto_examples_linear_model_plot_lasso_and_elasticnet.py`
  * :ref:`sphx_glr_auto_examples_applications_plot_tomography_l1_reconstruction.py`


.. note:: **使用 Lasso 进行特征选择**

      由于 Lasso regression 能够产生稀疏模型，所以他可以用来执行特征选择，详情可以参见 :ref:`l1_feature_selection`。

下面的两篇参考文章解释了scikit-learn的坐标下降求解器(coordinate descent solver)的迭代过程，以及用于控制收敛的对偶间隙(duality gap)的计算。

.. topic:: 参考文献

    * "Regularization Path For Generalized linear Models by Coordinate Descent",
      Friedman, Hastie & Tibshirani, J Stat Softw, 2010 (`Paper
      <https://www.jstatsoft.org/article/view/v033i01/v33i01.pdf>`_).
    * "An Interior-Point Method for Large-Scale L1-Regularized Least Squares,"
      S. J. Kim, K. Koh, M. Lustig, S. Boyd and D. Gorinevsky,
      in IEEE Journal of Selected Topics in Signal Processing, 2007
      (`Paper <https://web.stanford.edu/~boyd/papers/pdf/l1_ls.pdf>`_)


设置正则化参数
--------------------------------

``alpha`` 参数控制着估计出的模型的系数的稀疏度。 

使用交叉验证
^^^^^^^^^^^^^^^^^^^^^^^

scikit-learn 通过交叉验证来公开设置 Lasso ``alpha`` 参数的对象: :class:`LassoCV` 和 :class:`LassoLarsCV`。 
:class:`LassoLarsCV` 是基于下面解释的 :ref:`least_angle_regression` 算法。

对于带有很多共线回归器(collinear regressors)的高维数据集， :class:`LassoCV` 是经常被选择的模型。 
然而，:class:`LassoLarsCV` 在寻找更有相关性的 `alpha` 参数值上更具有优势，
而且如果样本数量与特征数量相比非常小时，通常 :class:`LassoLarsCV` 比 :class:`LassoCV` 要快。

.. |lasso_cv_1| image:: ../auto_examples/linear_model/images/sphx_glr_plot_lasso_model_selection_002.png
    :target: ../auto_examples/linear_model/plot_lasso_model_selection.html
    :scale: 48%

.. |lasso_cv_2| image:: ../auto_examples/linear_model/images/sphx_glr_plot_lasso_model_selection_003.png
    :target: ../auto_examples/linear_model/plot_lasso_model_selection.html
    :scale: 48%

.. centered:: |lasso_cv_1| |lasso_cv_2|


基于信息标准的模型选择
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

有多种选择时，估计器 :class:`LassoLarsIC` 建议使用 Akaike information criterion （Akaike 信息准则）（AIC）
和 Bayes Information criterion （贝叶斯信息准则）（BIC）。 当使用 k-fold 交叉验证时，正则化路径只计算一次而不是 k + 1 次，
所以找到 alpha 的最优值是一种计算上更便宜的替代方法。 然而，这样的标准需要对解决方案的自由度进行适当的估计，对于大样本（渐近结果）导出，
并假设模型是正确的，即数据实际上是由该模型生成的。 当问题严重受限（比样本更多的特征）时，他们也倾向于打破。

.. figure:: ../auto_examples/linear_model/images/sphx_glr_plot_lasso_model_selection_001.png
    :target: ../auto_examples/linear_model/plot_lasso_model_selection.html
    :align: center
    :scale: 50%


.. topic:: 案例:

  * :ref:`sphx_glr_auto_examples_linear_model_plot_lasso_model_selection.py`

与 SVM 的正则化参数的比较
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``alpha`` 和 SVM 的正则化参数 ``C`` 之间的等式关系是 ``alpha = 1 / C`` 或者 ``alpha = 1 / (n_samples * C)`` ，
并依赖于估计器和模型优化的确切的目标函数。

.. _multi_task_lasso:

多任务 Lasso
================

:class:`MultiTaskLasso` 是一个估计多元回归稀疏系数的线性模型： ``y`` 是一个 ``(n_samples, n_tasks)`` 的2D array，
其约束条件和其他回归问题（也称为任务）是一样的，都是所选的特征值。

下图比较了通过使用简单的 Lasso 或 MultiTaskLasso 得到的 W 中非零的位置。 
Lasso 估计产生分散的非零值，而 MultiTaskLasso 的一整列都是非零的。

.. |multi_task_lasso_1| image:: ../auto_examples/linear_model/images/sphx_glr_plot_multi_task_lasso_support_001.png
    :target: ../auto_examples/linear_model/plot_multi_task_lasso_support.html
    :scale: 48%

.. |multi_task_lasso_2| image:: ../auto_examples/linear_model/images/sphx_glr_plot_multi_task_lasso_support_002.png
    :target: ../auto_examples/linear_model/plot_multi_task_lasso_support.html
    :scale: 48%

.. centered:: |multi_task_lasso_1| |multi_task_lasso_2|

.. centered:: Fitting a time-series model, imposing that any active feature be active at all times.

.. topic:: 案例:

  * :ref:`sphx_glr_auto_examples_linear_model_plot_multi_task_lasso_support.py`


在数学上，它由一个线性模型组成，以混合的 :math:`\ell_1` :math:`\ell_2` 先验 作为正则化器进行训练。最小化的目标函数是：

.. math::  \min_{w} { \frac{1}{2n_{samples}} ||X W - Y||_{Fro} ^ 2 + \alpha ||W||_{21}}

其中 :math:`Fro` indicates the Frobenius norm:

.. math:: ||A||_{Fro} = \sqrt{\sum_{ij} a_{ij}^2}

和 :math:`\ell_1` :math:`\ell_2` reads:

.. math:: ||A||_{2 1} = \sum_i \sqrt{\sum_j a_{ij}^2}


:class:`MultiTaskLasso` 类的实现也使用了坐标下降法(coordinate descent)对系数进行拟合的。


.. _elastic_net:

弹性网(Elastic Net)
===========
:class:`ElasticNet` 是一种使用 L1 和 L2 范数作为先验正则项训练的线性回归模型。
这种正则化项的组合允许学习到一个只有少量参数是非零稀疏的模型，就像 :class:`Lasso` 一样，但是它仍然保持一些像 :class:`Ridge` 的正则性质。
我们可以利用 ``l1_ratio`` 参数控制 L1 和 L2 的凸组合。

弹性网络在很多特征互相联系的情况下是非常有用的。Lasso 很可能只随机考虑这些特征中的一个，而弹性网络更倾向于选择两个。

在实践中，Lasso 和 Ridge 之间权衡的一个优势是它允许Elastic-Net在循环过程（Under rotate）中继承 Ridge 的稳定性。

最小化的目标函数如下所示：

.. math::

    \min_{w} { \frac{1}{2n_{samples}} ||X w - y||_2 ^ 2 + \alpha \rho ||w||_1 +
    \frac{\alpha(1-\rho)}{2} ||w||_2 ^ 2}


.. figure:: ../auto_examples/linear_model/images/sphx_glr_plot_lasso_coordinate_descent_path_001.png
   :target: ../auto_examples/linear_model/plot_lasso_coordinate_descent_path.html
   :align: center
   :scale: 50%

:class:`ElasticNetCV` 类可以被用来通过交叉验证设置 ``alpha`` (:math:`\alpha`) 和 ``l1_ratio`` (:math:`\rho`) 参数。

.. topic:: 案例:

  * :ref:`sphx_glr_auto_examples_linear_model_plot_lasso_and_elasticnet.py`
  * :ref:`sphx_glr_auto_examples_linear_model_plot_lasso_coordinate_descent_path.py`

下面的两篇参考文章解释了scikit-learn 的坐标下降求解器(coordinate descent solver)的迭代过程，以及用于控制收敛的对偶间隙(duality gap)的计算。 

.. topic:: 参考文献

    * "Regularization Path For Generalized linear Models by Coordinate Descent",
      Friedman, Hastie & Tibshirani, J Stat Softw, 2010 (`Paper
      <https://www.jstatsoft.org/article/view/v033i01/v33i01.pdf>`_).
    * "An Interior-Point Method for Large-Scale L1-Regularized Least Squares,"
      S. J. Kim, K. Koh, M. Lustig, S. Boyd and D. Gorinevsky,
      in IEEE Journal of Selected Topics in Signal Processing, 2007
      (`Paper <https://web.stanford.edu/~boyd/papers/pdf/l1_ls.pdf>`_)

.. _multi_task_elastic_net:

多任务弹性网
======================

:class:`MultiTaskElasticNet` 是一个针对多变量回归问题估计其稀疏系数的弹性网模型: ``Y`` 是一个 2D array,
其shape为 ``(n_samples, n_tasks)``。 其约束条件和其他回归问题（也称为任务）是一样的，都是所选的特征值。

数学上, 它是一个混合了 :math:`\ell_1` :math:`\ell_2` 先验 和 :math:`\ell_2` 先验作为正则化项的线性模型。
目标函数的最小化如下所示：

.. math::

    \min_{W} { \frac{1}{2n_{samples}} ||X W - Y||_{Fro}^2 + \alpha \rho ||W||_{2 1} +
    \frac{\alpha(1-\rho)}{2} ||W||_{Fro}^2}

:class:`MultiTaskElasticNet` 类的实现也使用了坐标下降法(coordinate descent)对系数进行拟合的。

:class:`MultiTaskElasticNetCV` 类可以被用来通过交叉验证设置 ``alpha`` (:math:`\alpha`) 和 ``l1_ratio`` (:math:`\rho`) 参数。


.. _least_angle_regression:

最小角回归
======================

最小角回归 （Least-angle regression -- LARS） 是对高维数据的回归算法， 由 Bradley Efron, 
Trevor Hastie, Iain Johnstone 和 Robert Tibshirani 开发完成。 LARS 和前向逐步回归(forward stepwise
regression)很像。在每一步，它寻找与响应最有关联的 预测。当有很多预测有相同的关联时，它没有继续利用相同的预测，
而是在这些预测中找出应该等角的方向。

LARS 的优点如下 :

  - 当 p >> n，该算法数值运算上非常有效。(例如当维度的数目远超点的个数)

  - 它在计算上和前向选择一样快，和普通最小二乘法有相同的运算复杂度。

  - 它产生了一个完整的分段线性的解决路径，在交叉验证或者其他相似的微调模型的方法上非常有用。

  - 如果两个变量对响应几乎有相等的联系，则它们的系数应该有相似的增长率。因此这个算法和我们直觉上的判断一样，而且还更加稳定。

  - 它很容易修改并为其他估算器生成解，比如Lasso。

LARS 的缺点如下 :

  - 因为 LARS 是建立在循环拟合剩余变量上的，所以它对噪声非常敏感。这个问题，在 2004 年统计年鉴的文章由 Weisberg 详细讨论。

LARS 模型可以在 :class:`Lars` ，或者它的底层实现 :func:`lars_path` 中被使用。


LARS Lasso
==========

:class:`LassoLars` 是一个使用 LARS 算法的 lasso 模型，不同于基于坐标下降法的实现，
它可以得到一个精确解，也就是一个关于自身参数标准化后的一个分段线性解。

.. figure:: ../auto_examples/linear_model/images/sphx_glr_plot_lasso_lars_001.png
   :target: ../auto_examples/linear_model/plot_lasso_lars.html
   :align: center
   :scale: 50%

::

   >>> from sklearn import linear_model
   >>> reg = linear_model.LassoLars(alpha=.1)
   >>> reg.fit([[0, 0], [1, 1]], [0, 1])  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
   LassoLars(alpha=0.1, copy_X=True, eps=..., fit_intercept=True,
        fit_path=True, max_iter=500, normalize=True, positive=False,
        precompute='auto', verbose=False)
   >>> reg.coef_    # doctest: +ELLIPSIS
   array([0.717157..., 0.        ])

.. topic:: 案例:

 * :ref:`sphx_glr_auto_examples_linear_model_plot_lasso_lars.py`

Lars 算法提供了一个几乎无代价的沿着正则化参数的系数的完整路径，因此常利用函数 :func:`lars_path` 来取回路径。

数学化表达式
------------------------

该算法和前向逐步回归(forward stepwise regression)非常相似，但是它没有在每一步包含变量，
它估计的参数是根据与其他剩余变量的联系来增加的。

在 LARS 的解中，没有给出一个向量的结果，而是给出一条曲线，显示参数向量的 L1 范式的每个值的解。 
完全的参数路径存在 ``coef_path_`` 下。它的 size 是 (n_features, max_features+1)。 
其中第一列通常是全 0 列。

.. topic:: 参考:

 * Original Algorithm is detailed in the paper `Least Angle Regression
   <http://www-stat.stanford.edu/~hastie/Papers/LARS/LeastAngle_2002.pdf>`_
   by Hastie et al.


.. _omp:

正交匹配追踪法 (OMP)
=================================
:class:`OrthogonalMatchingPursuit` and :func:`orthogonal_mp` implements the OMP
algorithm for approximating the fit of a linear model with constraints imposed
on the number of non-zero coefficients (ie. the L :sub:`0` pseudo-norm).

Being a forward feature selection method like :ref:`least_angle_regression`,
orthogonal matching pursuit can approximate the optimum solution vector with a
fixed number of non-zero elements:

.. math::
    \underset{\gamma}{\operatorname{arg\,min\,}}  ||y - X\gamma||_2^2 \text{ subject to } ||\gamma||_0 \leq n_{nonzero\_coefs}

Alternatively, orthogonal matching pursuit can target a specific error instead
of a specific number of non-zero coefficients. This can be expressed as:

.. math::
    \underset{\gamma}{\operatorname{arg\,min\,}} ||\gamma||_0 \text{ subject to } ||y-X\gamma||_2^2 \leq \text{tol}


OMP is based on a greedy algorithm that includes at each step the atom most
highly correlated with the current residual. It is similar to the simpler
matching pursuit (MP) method, but better in that at each iteration, the
residual is recomputed using an orthogonal projection on the space of the
previously chosen dictionary elements.


.. topic:: 案例:

 * :ref:`sphx_glr_auto_examples_linear_model_plot_omp.py`

.. topic:: 参考文献:

 * http://www.cs.technion.ac.il/~ronrubin/Publications/KSVD-OMP-v2.pdf

 * `Matching pursuits with time-frequency dictionaries
   <http://blanche.polytechnique.fr/~mallat/papiers/MallatPursuit93.pdf>`_,
   S. G. Mallat, Z. Zhang,


.. _bayesian_regression:

贝叶斯回归
===================

贝叶斯回归可以用于在估计阶段的参数正则化: 正则化参数的选择不是通过人为选择进行设置，而是根据手头的数据来调节的。

上述过程可以通过引入 无信息先验(`uninformative priors <https://en.wikipedia.org/wiki/Non-informative_prior#Uninformative_priors>`__) 
于模型中的超参数来完成。 在 `Ridge Regression`_  中使用的 :math:`\ell_{2}` 正则项相当于在 :math:`w` 为高斯先验条件下，
且此先验的精确度为 :math:`\lambda^{-1}` 求最大后验估计。在这里，我们没有手工调参数 `\lambda` ，而是让他作为一个变量，通过数据中估计得到。

为了得到一个全概率模型，输出 :math:`y` 也被认为是关于 :math:`X w` 的高斯分布:

.. math::  p(y|X,w,\alpha) = \mathcal{N}(y|X w,\alpha)

Alpha 在这里也是作为一个变量，通过数据中估计得到。

Bayesian Regression 的优点是:

    - 它能根据已有的数据进行改变。

    - 它能在估计过程中引入正则项的参数。

Bayesian Regression 的缺点是:

    - 它的推断过程是非常耗时的。


.. topic:: 参考文献

 * A good introduction to Bayesian methods is given in C. Bishop: Pattern
   Recognition and Machine learning

 * Original Algorithm is detailed in the  book Bayesian learning for neural
   networks by Radford M. Neal

.. _bayesian_ridge_regression:

贝叶斯岭回归
-------------------------

:class:`BayesianRidge` 类会估计一个求解回归问题的概率模型(probabilistic model)。
参数 :math:`w` 的先验值 通过 球面高斯(spherical Gaussian)给出:

.. math:: p(w|\lambda) =
    \mathcal{N}(w|0,\lambda^{-1}\mathbf{I}_{p})

:math:`\alpha` 和 :math:`\lambda` 的先验分布选择为 `gamma distributions <https://en.wikipedia.org/wiki/Gamma_distribution>`__, 
这个分布与高斯成共轭先验关系。

得到的模型一般称为 贝叶斯岭回归(*Bayesian Ridge Regression*)， 并且这个与传统的 :class:`Ridge` 非常相似。
参数 :math:`w`, :math:`\alpha` 和 :math:`\lambda` 是在模型拟合的时候一起被估算出来的。 
剩下的超参数就是 关于 :math:`\alpha` 和 :math:`\lambda` 的 gamma 分布的先验了。 
它们通常被选择为 无信息先验(*non-informative*) 。模型参数的估计一般利用最大化 边缘对数似然估计(*marginal
log likelihood*)。

默认情况下 :math:`\alpha_1 = \alpha_2 =  \lambda_1 = \lambda_2 = 10^{-6}`.


.. figure:: ../auto_examples/linear_model/images/sphx_glr_plot_bayesian_ridge_001.png
   :target: ../auto_examples/linear_model/plot_bayesian_ridge.html
   :align: center
   :scale: 50%


Bayesian Ridge Regression 用于回归问题 ::

    >>> from sklearn import linear_model
    >>> X = [[0., 0.], [1., 1.], [2., 2.], [3., 3.]]
    >>> Y = [0., 1., 2., 3.]
    >>> reg = linear_model.BayesianRidge()
    >>> reg.fit(X, Y)
    BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, compute_score=False, copy_X=True,
           fit_intercept=True, lambda_1=1e-06, lambda_2=1e-06, n_iter=300,
           normalize=False, tol=0.001, verbose=False)

当模型拟合好以后, 就可以用来预测新的观测值啦::

    >>> reg.predict([[1, 0.]])
    array([0.50000013])


模型的权重 :math:`w` 可以这样获得 ::

    >>> reg.coef_
    array([0.49999993, 0.49999993])

由于贝叶斯框架的缘故，权值与 :ref:`ordinary_least_squares` 产生的不太一样。 
但是，贝叶斯岭回归对病态问题（ill-posed）的鲁棒性要更好。

.. topic:: 案例:

 * :ref:`sphx_glr_auto_examples_linear_model_plot_bayesian_ridge.py`

.. topic:: 参考文献

  * More details can be found in the article `Bayesian Interpolation
    <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.27.9072&rep=rep1&type=pdf>`_
    by MacKay, David J. C.



主动相关决策理论 - ARD
---------------------------------------

:class:`ARDRegression` is very similar to `Bayesian Ridge Regression`_,
but can lead to sparser weights :math:`w` [1]_ [2]_.
:class:`ARDRegression` poses a different prior over :math:`w`, by dropping the
assumption of the Gaussian being spherical.

Instead, the distribution over :math:`w` is assumed to be an axis-parallel,
elliptical Gaussian distribution.

This means each weight :math:`w_{i}` is drawn from a Gaussian distribution,
centered on zero and with a precision :math:`\lambda_{i}`:

.. math:: p(w|\lambda) = \mathcal{N}(w|0,A^{-1})

with :math:`diag \; (A) = \lambda = \{\lambda_{1},...,\lambda_{p}\}`.

In contrast to `Bayesian Ridge Regression`_, each coordinate of :math:`w_{i}`
has its own standard deviation :math:`\lambda_i`. The prior over all
:math:`\lambda_i` is chosen to be the same gamma distribution given by
hyperparameters :math:`\lambda_1` and :math:`\lambda_2`.

.. figure:: ../auto_examples/linear_model/images/sphx_glr_plot_ard_001.png
   :target: ../auto_examples/linear_model/plot_ard.html
   :align: center
   :scale: 50%

ARD is also known in the literature as *Sparse Bayesian Learning* and
*Relevance Vector Machine* [3]_ [4]_.

.. topic:: Examples:

  * :ref:`sphx_glr_auto_examples_linear_model_plot_ard.py`

.. topic:: References:

    .. [1] Christopher M. Bishop: Pattern Recognition and Machine Learning, Chapter 7.2.1

    .. [2] David Wipf and Srikantan Nagarajan: `A new view of automatic relevance determination <http://papers.nips.cc/paper/3372-a-new-view-of-automatic-relevance-determination.pdf>`_

    .. [3] Michael E. Tipping: `Sparse Bayesian Learning and the Relevance Vector Machine <http://www.jmlr.org/papers/volume1/tipping01a/tipping01a.pdf>`_

    .. [4] Tristan Fletcher: `Relevance Vector Machines explained <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.651.8603&rep=rep1&type=pdf>`_



.. _Logistic_regression:

Logistic 回归
===================

Logistic 回归，虽然名字里有 “回归” 二字，但实际上是解决分类问题的一类线性模型。
在某些文献中，logistic 回归又被称作 logit 回归，maximum-entropy classification（MaxEnt，最大熵分类），
或 log-linear classifier（对数线性分类器）。
该模型利用函数 `logistic function <https://en.wikipedia.org/wiki/Logistic_function>`_ 将单次试验（single trial）的可能结果输出为概率。

scikit-learn 中 logistic regression 在 :class:`LogisticRegression` 类中实现了二分类（binary）、
一对多分类（one-vs-rest）及多项式 logistic 回归，并带有可选的 L1 和 L2 正则化。

作为一个优化问题，带 L2 惩罚项的二分类 logistic 回归要最小化以下代价函数（cost function）：

.. math:: \min_{w, c} \frac{1}{2}w^T w + C \sum_{i=1}^n \log(\exp(- y_i (X_i^T w + c)) + 1) .

类似的, 带有 L1 正则化的 logistic regression 求解下面的问题：

.. math:: \min_{w, c} \|w\|_1 + C \sum_{i=1}^n \log(\exp(- y_i (X_i^T w + c)) + 1).

注意, 在这个记法中, 假定了观测 :math:`y_i` 在集合 :math:`{-1, 1}` 中获取值(takes values in the set
:math:`{-1, 1}` at trial :math:`i`)。

在 :class:`LogisticRegression` 类中实现了这些优化算法: "liblinear", "newton-cg", "lbfgs", "sag" 和 "saga"。

The solver "liblinear" uses a coordinate descent (CD) algorithm, and relies
on the excellent C++ `LIBLINEAR library
<https://www.csie.ntu.edu.tw/~cjlin/liblinear/>`_, which is shipped with
scikit-learn. However, the CD algorithm implemented in liblinear cannot learn
a true multinomial (multiclass) model; instead, the optimization problem is
decomposed in a "one-vs-rest" fashion so separate binary classifiers are
trained for all classes. This happens under the hood, so
:class:`LogisticRegression` instances using this solver behave as multiclass
classifiers. For L1 penalization :func:`sklearn.svm.l1_min_c` allows to
calculate the lower bound for C in order to get a non "null" (all feature
weights to zero) model.

The "lbfgs", "sag" and "newton-cg" solvers only support L2 penalization and
are found to converge faster for some high dimensional data. Setting
`multi_class` to "multinomial" with these solvers learns a true multinomial
logistic regression model [5]_, which means that its probability estimates
should be better calibrated than the default "one-vs-rest" setting.

The "sag" solver uses a Stochastic Average Gradient descent [6]_. It is faster
than other solvers for large datasets, when both the number of samples and the
number of features are large.

The "saga" solver [7]_ is a variant of "sag" that also supports the
non-smooth `penalty="l1"` option. This is therefore the solver of choice
for sparse multinomial logistic regression.

In a nutshell, the following table summarizes the penalties supported by each solver:

+------------------------------+-----------------+-------------+-----------------+-----------+------------+
|                              |                       **Solvers**                                        |
+------------------------------+-----------------+-------------+-----------------+-----------+------------+
| **Penalties**                | **'liblinear'** | **'lbfgs'** | **'newton-cg'** | **'sag'** | **'saga'** |
+------------------------------+-----------------+-------------+-----------------+-----------+------------+
| Multinomial + L2 penalty     |       no        |     yes     |       yes       |    yes    |    yes     |
+------------------------------+-----------------+-------------+-----------------+-----------+------------+
| OVR + L2 penalty             |       yes       |     yes     |       yes       |    yes    |    yes     |
+------------------------------+-----------------+-------------+-----------------+-----------+------------+
| Multinomial + L1 penalty     |       no        |     no      |       no        |    no     |    yes     |
+------------------------------+-----------------+-------------+-----------------+-----------+------------+
| OVR + L1 penalty             |       yes       |     no      |       no        |    no     |    yes     |
+------------------------------+-----------------+-------------+-----------------+-----------+------------+
| **Behaviors**                |                                                                          |
+------------------------------+-----------------+-------------+-----------------+-----------+------------+
| Penalize the intercept (bad) |       yes       |     no      |       no        |    no     |    no      |
+------------------------------+-----------------+-------------+-----------------+-----------+------------+
| Faster for large datasets    |       no        |     no      |       no        |    yes    |    yes     |
+------------------------------+-----------------+-------------+-----------------+-----------+------------+
| Robust to unscaled datasets  |       yes       |     yes     |       yes       |    no     |    no      |
+------------------------------+-----------------+-------------+-----------------+-----------+------------+

The "saga" solver is often the best choice but requires scaling. The "liblinear" solver is
used by default for historical reasons.

对于大数据集，还可以用 :class:`SGDClassifier` ，并使用对数损失（'log' loss）。

.. topic:: 案例:

  * :ref:`sphx_glr_auto_examples_linear_model_plot_logistic_l1_l2_sparsity.py`

  * :ref:`sphx_glr_auto_examples_linear_model_plot_logistic_path.py`

  * :ref:`sphx_glr_auto_examples_linear_model_plot_logistic_multinomial.py`

  * :ref:`sphx_glr_auto_examples_linear_model_plot_sparse_logistic_regression_20newsgroups.py`

  * :ref:`sphx_glr_auto_examples_linear_model_plot_sparse_logistic_regression_mnist.py`

.. _liblinear_differences:

.. topic:: 与 liblinear 的不同:

   There might be a difference in the scores obtained between
   :class:`LogisticRegression` with ``solver=liblinear``
   or :class:`LinearSVC` and the external liblinear library directly,
   when ``fit_intercept=False`` and the fit ``coef_`` (or) the data to
   be predicted are zeroes. This is because for the sample(s) with
   ``decision_function`` zero, :class:`LogisticRegression` and :class:`LinearSVC`
   predict the negative class, while liblinear predicts the positive class.
   Note that a model with ``fit_intercept=False`` and having many samples with
   ``decision_function`` zero, is likely to be a underfit, bad model and you are
   advised to set ``fit_intercept=True`` and increase the intercept_scaling.

.. note:: **利用稀疏 logistic 回归进行特征选择**

   带 L1 罚项的 logistic 回归 将得到稀疏模型（sparse model），
   相当于进行了特征选择（feature selection），详情参见 :ref:`l1_feature_selection` 。
   .

:class:`LogisticRegressionCV` implements Logistic Regression with
builtin cross-validation to find out the optimal C parameter.
"newton-cg", "sag", "saga" and "lbfgs" solvers are found to be faster
for high-dimensional dense data, due to warm-starting. For the
multiclass case, if `multi_class` option is set to "ovr", an optimal C
is obtained for each class and if the `multi_class` option is set to
"multinomial", an optimal C is obtained by minimizing the cross-entropy
loss.

.. topic:: 参考文献:

    .. [5] Christopher M. Bishop: Pattern Recognition and Machine Learning, Chapter 4.3.4

    .. [6] Mark Schmidt, Nicolas Le Roux, and Francis Bach: `Minimizing Finite Sums with the Stochastic Average Gradient. <https://hal.inria.fr/hal-00860051/document>`_

    .. [7] Aaron Defazio, Francis Bach, Simon Lacoste-Julien: `SAGA: A Fast Incremental Gradient Method With Support for Non-Strongly Convex Composite Objectives. <https://arxiv.org/abs/1407.0202>`_

随机梯度下降 - SGD
=================================

随机梯度下降(Stochastic gradient descent)是拟合线性模型的一个简单而高效的方法。在样本量（和特征数）很大时尤为有用。 
方法 ``partial_fit`` 可用于 online learning （在线学习）或基于 out-of-core learning （外存的学习）。

:class:`SGDClassifier` 和 :class:`SGDRegressor` 分别用于拟合分类问题和回归问题的线性模型，可使用不同的（凸）损失函数，支持不同的罚项。 
例如，设定 loss="log" ，则 :class:`SGDClassifier` 拟合一个逻辑斯蒂回归模型，而 loss="hinge" 拟合线性支持向量机（SVM）。

.. topic:: 参考

 * :ref:`sgd`

.. _perceptron:

感知器
==========

:class:`Perceptron` 是适用于大规模学习的一种简单算法。默认情况下：

    - 不需要设置学习率（learning rate）。

    - 不需要正则化处理。

    - 仅使用错误样本更新模型。

最后一点表明使用合页损失（hinge loss）的感知机比 使用 hinge loss 的SGD 略快，所得模型更稀疏。

.. _passive_aggressive:

被动攻击算法(Passive Aggressive Algorithms)
=============================

The passive-aggressive algorithms are a family of algorithms for large-scale
learning. They are similar to the Perceptron in that they do not require a
learning rate. However, contrary to the Perceptron, they include a
regularization parameter ``C``.

For classification, :class:`PassiveAggressiveClassifier` can be used with
``loss='hinge'`` (PA-I) or ``loss='squared_hinge'`` (PA-II).  For regression,
:class:`PassiveAggressiveRegressor` can be used with
``loss='epsilon_insensitive'`` (PA-I) or
``loss='squared_epsilon_insensitive'`` (PA-II).

.. topic:: References:


 * `"Online Passive-Aggressive Algorithms"
   <http://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf>`_
   K. Crammer, O. Dekel, J. Keshat, S. Shalev-Shwartz, Y. Singer - JMLR 7 (2006)


鲁棒回归:处理离群点（outliers）和模型错误
=====================================================

Robust regression 特别适用于回归模型包含损坏数据（corrupt data）的情况，如离群点或模型中的错误。

.. figure:: ../auto_examples/linear_model/images/sphx_glr_plot_theilsen_001.png
   :target: ../auto_examples/linear_model/plot_theilsen.html
   :scale: 50%
   :align: center

各种使用场景与相关概念
----------------------------------------

处理包含离群点的数据时牢记以下几点:

.. |y_outliers| image:: ../auto_examples/linear_model/images/sphx_glr_plot_robust_fit_003.png
   :target: ../auto_examples/linear_model/plot_robust_fit.html
   :scale: 60%

.. |X_outliers| image:: ../auto_examples/linear_model/images/sphx_glr_plot_robust_fit_002.png
   :target: ../auto_examples/linear_model/plot_robust_fit.html
   :scale: 60%

.. |large_y_outliers| image:: ../auto_examples/linear_model/images/sphx_glr_plot_robust_fit_005.png
   :target: ../auto_examples/linear_model/plot_robust_fit.html
   :scale: 60%

* **Outliers in X or in y**?

  ==================================== ====================================
  Outliers in the y direction          Outliers in the X direction
  ==================================== ====================================
  |y_outliers|                         |X_outliers|
  ==================================== ====================================

* **Fraction of outliers versus amplitude of error**

  The number of outlying points matters, but also how much they are
  outliers.

  ==================================== ====================================
  Small outliers                       Large outliers
  ==================================== ====================================
  |y_outliers|                         |large_y_outliers|
  ==================================== ====================================

An important notion of robust fitting is that of breakdown point: the
fraction of data that can be outlying for the fit to start missing the
inlying data.

Note that in general, robust fitting in high-dimensional setting (large
`n_features`) is very hard. The robust models here will probably not work
in these settings.


.. topic:: **Trade-offs: which estimator?**

  Scikit-learn provides 3 robust regression estimators:
  :ref:`RANSAC <ransac_regression>`,
  :ref:`Theil Sen <theil_sen_regression>` and
  :ref:`HuberRegressor <huber_regression>`

  * :ref:`HuberRegressor <huber_regression>` should be faster than
    :ref:`RANSAC <ransac_regression>` and :ref:`Theil Sen <theil_sen_regression>`
    unless the number of samples are very large, i.e ``n_samples`` >> ``n_features``.
    This is because :ref:`RANSAC <ransac_regression>` and :ref:`Theil Sen <theil_sen_regression>`
    fit on smaller subsets of the data. However, both :ref:`Theil Sen <theil_sen_regression>`
    and :ref:`RANSAC <ransac_regression>` are unlikely to be as robust as
    :ref:`HuberRegressor <huber_regression>` for the default parameters.

  * :ref:`RANSAC <ransac_regression>` is faster than :ref:`Theil Sen <theil_sen_regression>`
    and scales much better with the number of samples

  * :ref:`RANSAC <ransac_regression>` will deal better with large
    outliers in the y direction (most common situation)

  * :ref:`Theil Sen <theil_sen_regression>` will cope better with
    medium-size outliers in the X direction, but this property will
    disappear in large dimensional settings.

 When in doubt, use :ref:`RANSAC <ransac_regression>`

.. _ransac_regression:

RANSAC:随机抽样一致性算法
--------------------------------

RANSAC (RANdom SAmple Consensus) fits a model from random subsets of
inliers from the complete data set.

RANSAC is a non-deterministic algorithm producing only a reasonable result with
a certain probability, which is dependent on the number of iterations (see
`max_trials` parameter). It is typically used for linear and non-linear
regression problems and is especially popular in the fields of photogrammetric
computer vision.

The algorithm splits the complete input sample data into a set of inliers,
which may be subject to noise, and outliers, which are e.g. caused by erroneous
measurements or invalid hypotheses about the data. The resulting model is then
estimated only from the determined inliers.

.. figure:: ../auto_examples/linear_model/images/sphx_glr_plot_ransac_001.png
   :target: ../auto_examples/linear_model/plot_ransac.html
   :align: center
   :scale: 50%

算法细节
^^^^^^^^^^^^^^^^^^^^^^^^

Each iteration performs the following steps:

1. Select ``min_samples`` random samples from the original data and check
   whether the set of data is valid (see ``is_data_valid``).
2. Fit a model to the random subset (``base_estimator.fit``) and check
   whether the estimated model is valid (see ``is_model_valid``).
3. Classify all data as inliers or outliers by calculating the residuals
   to the estimated model (``base_estimator.predict(X) - y``) - all data
   samples with absolute residuals smaller than the ``residual_threshold``
   are considered as inliers.
4. Save fitted model as best model if number of inlier samples is
   maximal. In case the current estimated model has the same number of
   inliers, it is only considered as the best model if it has better score.

These steps are performed either a maximum number of times (``max_trials``) or
until one of the special stop criteria are met (see ``stop_n_inliers`` and
``stop_score``). The final model is estimated using all inlier samples (consensus
set) of the previously determined best model.

The ``is_data_valid`` and ``is_model_valid`` functions allow to identify and reject
degenerate combinations of random sub-samples. If the estimated model is not
needed for identifying degenerate cases, ``is_data_valid`` should be used as it
is called prior to fitting the model and thus leading to better computational
performance.


.. topic:: Examples:

  * :ref:`sphx_glr_auto_examples_linear_model_plot_ransac.py`
  * :ref:`sphx_glr_auto_examples_linear_model_plot_robust_fit.py`

.. topic:: References:

 * https://en.wikipedia.org/wiki/RANSAC
 * `"Random Sample Consensus: A Paradigm for Model Fitting with Applications to
   Image Analysis and Automated Cartography"
   <https://www.sri.com/sites/default/files/publications/ransac-publication.pdf>`_
   Martin A. Fischler and Robert C. Bolles - SRI International (1981)
 * `"Performance Evaluation of RANSAC Family"
   <http://www.bmva.org/bmvc/2009/Papers/Paper355/Paper355.pdf>`_
   Sunglok Choi, Taemin Kim and Wonpil Yu - BMVC (2009)

.. _theil_sen_regression:

Theil-Sen 估计器: 广义中值估计器
--------------------------------------------------------

The :class:`TheilSenRegressor` estimator uses a generalization of the median in
multiple dimensions. It is thus robust to multivariate outliers. Note however
that the robustness of the estimator decreases quickly with the dimensionality
of the problem. It looses its robustness properties and becomes no
better than an ordinary least squares in high dimension.

.. topic:: Examples:

  * :ref:`sphx_glr_auto_examples_linear_model_plot_theilsen.py`
  * :ref:`sphx_glr_auto_examples_linear_model_plot_robust_fit.py`

.. topic:: References:

 * https://en.wikipedia.org/wiki/Theil%E2%80%93Sen_estimator

理论方面的思考
^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`TheilSenRegressor` is comparable to the :ref:`Ordinary Least Squares
(OLS) <ordinary_least_squares>` in terms of asymptotic efficiency and as an
unbiased estimator. In contrast to OLS, Theil-Sen is a non-parametric
method which means it makes no assumption about the underlying
distribution of the data. Since Theil-Sen is a median-based estimator, it
is more robust against corrupted data aka outliers. In univariate
setting, Theil-Sen has a breakdown point of about 29.3% in case of a
simple linear regression which means that it can tolerate arbitrary
corrupted data of up to 29.3%.

.. figure:: ../auto_examples/linear_model/images/sphx_glr_plot_theilsen_001.png
   :target: ../auto_examples/linear_model/plot_theilsen.html
   :align: center
   :scale: 50%

The implementation of :class:`TheilSenRegressor` in scikit-learn follows a
generalization to a multivariate linear regression model [#f1]_ using the
spatial median which is a generalization of the median to multiple
dimensions [#f2]_.

In terms of time and space complexity, Theil-Sen scales according to

.. math::
    \binom{n_{samples}}{n_{subsamples}}

which makes it infeasible to be applied exhaustively to problems with a
large number of samples and features. Therefore, the magnitude of a
subpopulation can be chosen to limit the time and space complexity by
considering only a random subset of all possible combinations.

.. topic:: Examples:

  * :ref:`sphx_glr_auto_examples_linear_model_plot_theilsen.py`

.. topic:: References:

    .. [#f1] Xin Dang, Hanxiang Peng, Xueqin Wang and Heping Zhang: `Theil-Sen Estimators in a Multiple Linear Regression Model. <http://home.olemiss.edu/~xdang/papers/MTSE.pdf>`_

    .. [#f2] T. Kärkkäinen and S. Äyrämö: `On Computation of Spatial Median for Robust Data Mining. <http://users.jyu.fi/~samiayr/pdf/ayramo_eurogen05.pdf>`_

.. _huber_regression:

Huber 回归
----------------

The :class:`HuberRegressor` is different to :class:`Ridge` because it applies a
linear loss to samples that are classified as outliers.
A sample is classified as an inlier if the absolute error of that sample is
lesser than a certain threshold. It differs from :class:`TheilSenRegressor`
and :class:`RANSACRegressor` because it does not ignore the effect of the outliers
but gives a lesser weight to them.

.. figure:: /auto_examples/linear_model/images/sphx_glr_plot_huber_vs_ridge_001.png
   :target: ../auto_examples/linear_model/plot_huber_vs_ridge.html
   :align: center
   :scale: 50%

The loss function that :class:`HuberRegressor` minimizes is given by

.. math::

  \min_{w, \sigma} {\sum_{i=1}^n\left(\sigma + H_{\epsilon}\left(\frac{X_{i}w - y_{i}}{\sigma}\right)\sigma\right) + \alpha {||w||_2}^2}

where

.. math::

  H_{\epsilon}(z) = \begin{cases}
         z^2, & \text {if } |z| < \epsilon, \\
         2\epsilon|z| - \epsilon^2, & \text{otherwise}
  \end{cases}

It is advised to set the parameter ``epsilon`` to 1.35 to achieve 95% statistical efficiency.

注意
-----
The :class:`HuberRegressor` differs from using :class:`SGDRegressor` with loss set to `huber`
in the following ways.

- :class:`HuberRegressor` is scaling invariant. Once ``epsilon`` is set, scaling ``X`` and ``y``
  down or up by different values would produce the same robustness to outliers as before.
  as compared to :class:`SGDRegressor` where ``epsilon`` has to be set again when ``X`` and ``y`` are
  scaled.

- :class:`HuberRegressor` should be more efficient to use on data with small number of
  samples while :class:`SGDRegressor` needs a number of passes on the training data to
  produce the same robustness.

.. topic:: Examples:

  * :ref:`sphx_glr_auto_examples_linear_model_plot_huber_vs_ridge.py`

.. topic:: References:

  * Peter J. Huber, Elvezio M. Ronchetti: Robust Statistics, Concomitant scale estimates, pg 172

Also, this estimator is different from the R implementation of Robust Regression
(http://www.ats.ucla.edu/stat/r/dae/rreg.htm) because the R implementation does a weighted least
squares implementation with weights given to each sample on the basis of how much the residual is
greater than a certain threshold.

.. _polynomial_regression:

多项式回归:用基函数展开线性模型
===================================================================

.. currentmodule:: sklearn.preprocessing

One common pattern within machine learning is to use linear models trained
on nonlinear functions of the data.  This approach maintains the generally
fast performance of linear methods, while allowing them to fit a much wider
range of data.

For example, a simple linear regression can be extended by constructing
**polynomial features** from the coefficients.  In the standard linear
regression case, you might have a model that looks like this for
two-dimensional data:

.. math::    \hat{y}(w, x) = w_0 + w_1 x_1 + w_2 x_2

If we want to fit a paraboloid to the data instead of a plane, we can combine
the features in second-order polynomials, so that the model looks like this:

.. math::    \hat{y}(w, x) = w_0 + w_1 x_1 + w_2 x_2 + w_3 x_1 x_2 + w_4 x_1^2 + w_5 x_2^2

The (sometimes surprising) observation is that this is *still a linear model*:
to see this, imagine creating a new variable

.. math::  z = [x_1, x_2, x_1 x_2, x_1^2, x_2^2]

With this re-labeling of the data, our problem can be written

.. math::    \hat{y}(w, x) = w_0 + w_1 z_1 + w_2 z_2 + w_3 z_3 + w_4 z_4 + w_5 z_5

We see that the resulting *polynomial regression* is in the same class of
linear models we'd considered above (i.e. the model is linear in :math:`w`)
and can be solved by the same techniques.  By considering linear fits within
a higher-dimensional space built with these basis functions, the model has the
flexibility to fit a much broader range of data.

Here is an example of applying this idea to one-dimensional data, using
polynomial features of varying degrees:

.. figure:: ../auto_examples/linear_model/images/sphx_glr_plot_polynomial_interpolation_001.png
   :target: ../auto_examples/linear_model/plot_polynomial_interpolation.html
   :align: center
   :scale: 50%

This figure is created using the :class:`PolynomialFeatures` preprocessor.
This preprocessor transforms an input data matrix into a new data matrix
of a given degree.  It can be used as follows::

    >>> from sklearn.preprocessing import PolynomialFeatures
    >>> import numpy as np
    >>> X = np.arange(6).reshape(3, 2)
    >>> X
    array([[0, 1],
           [2, 3],
           [4, 5]])
    >>> poly = PolynomialFeatures(degree=2)
    >>> poly.fit_transform(X)
    array([[ 1.,  0.,  1.,  0.,  0.,  1.],
           [ 1.,  2.,  3.,  4.,  6.,  9.],
           [ 1.,  4.,  5., 16., 20., 25.]])

The features of ``X`` have been transformed from :math:`[x_1, x_2]` to
:math:`[1, x_1, x_2, x_1^2, x_1 x_2, x_2^2]`, and can now be used within
any linear model.

This sort of preprocessing can be streamlined with the
:ref:`Pipeline <pipeline>` tools. A single object representing a simple
polynomial regression can be created and used as follows::

    >>> from sklearn.preprocessing import PolynomialFeatures
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.pipeline import Pipeline
    >>> import numpy as np
    >>> model = Pipeline([('poly', PolynomialFeatures(degree=3)),
    ...                   ('linear', LinearRegression(fit_intercept=False))])
    >>> # fit to an order-3 polynomial data
    >>> x = np.arange(5)
    >>> y = 3 - 2 * x + x ** 2 - x ** 3
    >>> model = model.fit(x[:, np.newaxis], y)
    >>> model.named_steps['linear'].coef_
    array([ 3., -2.,  1., -1.])

The linear model trained on polynomial features is able to exactly recover
the input polynomial coefficients.

In some cases it's not necessary to include higher powers of any single feature,
but only the so-called *interaction features*
that multiply together at most :math:`d` distinct features.
These can be gotten from :class:`PolynomialFeatures` with the setting
``interaction_only=True``.

For example, when dealing with boolean features,
:math:`x_i^n = x_i` for all :math:`n` and is therefore useless;
but :math:`x_i x_j` represents the conjunction of two booleans.
This way, we can solve the XOR problem with a linear classifier::

    >>> from sklearn.linear_model import Perceptron
    >>> from sklearn.preprocessing import PolynomialFeatures
    >>> import numpy as np
    >>> X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    >>> y = X[:, 0] ^ X[:, 1]
    >>> y
    array([0, 1, 1, 0])
    >>> X = PolynomialFeatures(interaction_only=True).fit_transform(X).astype(int)
    >>> X
    array([[1, 0, 0, 0],
           [1, 0, 1, 0],
           [1, 1, 0, 0],
           [1, 1, 1, 1]])
    >>> clf = Perceptron(fit_intercept=False, max_iter=10, tol=None,
    ...                  shuffle=False).fit(X, y)

And the classifier "predictions" are perfect::

    >>> clf.predict(X)
    array([0, 1, 1, 0])
    >>> clf.score(X, y)
    1.0
