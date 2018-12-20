.. _sgd:

===========================
随机梯度下降（Stochastic Gradient Descent）
===========================

.. currentmodule:: sklearn.linear_model

**Stochastic Gradient Descent (SGD)** 是一种简单但又非常高效的方法，主要用于凸损失函数下线性分类器的判别式学习，
例如(线性) `支持向量机 <https://en.wikipedia.org/wiki/Support_vector_machine>`_ 和 
`Logistic 回归 <https://en.wikipedia.org/wiki/Logistic_regression>`_ 。 
尽管 SGD 在机器学习社区已经存在了很长时间, 但是最近在 large-scale learning （大规模学习）方面 SGD 获得了相当大的关注。

SGD 已成功应用于在文本分类和自然语言处理中经常遇到的大规模和稀疏的机器学习问题。
对于稀疏数据，本模块的分类器可以轻易的处理超过 10^5 的训练样本和超过 10^5 的特征。

随机梯度下降法 的优势:

    + 高效。

    + 易于实现 (有大量优化代码的机会)。

随机梯度下降法的劣势:

    + SGD 需要一些超参数，例如 regularization （正则化）参数和 number of iterations （迭代次数）。

    + SGD 对 feature scaling （特征缩放）敏感。

分类
==============

.. warning::

  在拟合模型前，确保你重新排列了（打乱）)你的训练数据，或者在每次迭代后用 ``shuffle=True`` 来打乱。

:class:`SGDClassifier` 类实现了一个简单的随机梯度下降学习例程, 支持不同的 loss functions（损失函数）和 penalties for classification（分类惩罚）。

.. figure:: ../auto_examples/linear_model/images/sphx_glr_plot_sgd_separating_hyperplane_001.png
   :target: ../auto_examples/linear_model/plot_sgd_separating_hyperplane.html
   :align: center
   :scale: 75

与其他分类器一样，拟合 SGD 我们需要两个 array （数组）：保存训练样本的 size 为 ``[n_samples, n_features]`` 的数组 X 以及保存训练样本目标值（类标签）
的 size 为 ``[n_samples]`` 的数组 Y ::

    >>> from sklearn.linear_model import SGDClassifier
    >>> X = [[0., 0.], [1., 1.]]
    >>> y = [0, 1]
    >>> clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
    >>> clf.fit(X, y)   # doctest: +NORMALIZE_WHITESPACE
    SGDClassifier(alpha=0.0001, average=False, class_weight=None,
               early_stopping=False, epsilon=0.1, eta0=0.0, fit_intercept=True,
               l1_ratio=0.15, learning_rate='optimal', loss='hinge', max_iter=5,
               n_iter=None, n_iter_no_change=5, n_jobs=None, penalty='l2',
               power_t=0.5, random_state=None, shuffle=True, tol=None,
               validation_fraction=0.1, verbose=0, warm_start=False)


当模型拟合好以后，我们就可以用它对新的观测向量作出预测::

    >>> clf.predict([[2., 2.]])
    array([1])

SGD 会在训练数据上拟合出一个线性模型。 类成员 ``coef_`` 里面保存着学习到的模型参数(线性模型的系数) ::

    >>> clf.coef_                                         # doctest: +ELLIPSIS
    array([[9.9..., 9.9...]])

类成员 ``intercept_`` 保存着学习到的线性模型的截距(intercept) (也叫 偏置(offset or bias))::

    >>> clf.intercept_                                    # doctest: +ELLIPSIS
    array([-9.9...])

模型是否要用到截距(intercept), i.e. 一个有偏置的超平面(a biased hyperplane), 可以通过参数 ``fit_intercept`` 进行控制。

如果要获得某个样本点到超平面的有正负的距离 ， 可以使用 决策函数 :meth:`SGDClassifier.decision_function`::

    >>> clf.decision_function([[2., 2.]])                 # doctest: +ELLIPSIS
    array([29.6...])

模型用到的具体的损失函数可以通过参数 ``loss`` 进行设置。 :class:`SGDClassifier` 类支持以下的损失函数:

  * ``loss="hinge"``: (soft-margin) linear Support Vector Machine,
  * ``loss="modified_huber"``: smoothed hinge loss,
  * ``loss="log"``: logistic regression,
  * and all regression losses below.

前两个损失函数是懒惰的(lazy)，只有当某个/些样本点违反了边界约束(margin constraint），它们才会去更新模型的参数, 这使得训练非常有效率,
即使使用了 L2 penalty（惩罚）我们仍然可能得到稀疏的模型。

使用 ``loss="log"`` 或 ``loss="modified_huber"`` 可以激活 ``predict_proba`` 方法, 
该方法会为每个样本 :math:`x` 估计出它属于每个类的概率向量 :math:`P(y|x)` （译者注：也就是给定一个样本以后预测这个样本属于每个类的概率） ::

    >>> clf = SGDClassifier(loss="log", max_iter=5).fit(X, y)
    >>> clf.predict_proba([[1., 1.]])                      # doctest: +ELLIPSIS
    array([[0.00..., 0.99...]])

具体的惩罚项可以通过参数 ``penalty`` 设置, SGD 支持下列的惩罚项:

  * ``penalty="l2"``: L2 norm penalty on ``coef_``.
  * ``penalty="l1"``: L1 norm penalty on ``coef_``.
  * ``penalty="elasticnet"``:  L2 和 L1 的 凸组合(Convex combination): ``(1 - l1_ratio) * L2 + l1_ratio * L1``.

默认的设置是 ``penalty="l2"``。 L1 penalty 导致稀疏解，使得大多数系数为零。 弹性网（Elastic Net）解决了在特征高相关时 
L1 penalty 的一些不足。 参数 ``l1_ratio`` 用来控制 L1 penalty 和 L2 penalty 的凸组合。

:class:`SGDClassifier` 通过利用 "one versus all" (OVA) 机制来组合多个二分类器，从而实现多分类。
对于每一个 :math:`K` 类, 可以训练一个二分类器来区分自身和其他 :math:`K-1` 个类。在测试阶段，
我们计算每个分类器的 confidence score（置信度分数）（也就是与超平面的距离），并选择置信度最高的分类。
下图阐释了基于 iris（鸢尾花）数据集上的 OVA 方法。虚线表示三个 OVA 分类器; 不同背景色代表由三个分类器产生的决策面。

.. figure:: ../auto_examples/linear_model/images/sphx_glr_plot_sgd_iris_001.png
   :target: ../auto_examples/linear_model/plot_sgd_iris.html
   :align: center
   :scale: 75

在 multi-class classification的情况下， ``coef_`` 是 ``shape=[n_classes, n_features]`` 的一个二维数组， 
``intercept_`` 是 ``shape=[n_classes]`` 的一个一维数组。 ``coef_`` 的第 i 行保存了第 i 类的 OVA 分类器的权重向量；
类以升序索引 （参照属性 ``classes_`` ）。 注意，原则上，由于它们允许创建一个概率模型，所以 ``loss="log"`` 和 ``loss="modified_huber"``
更适合于 one-vs-all 分类。

:class:`SGDClassifier` 类通过参数 ``class_weight`` 和 ``sample_weight`` 来支持 加权类(weighted classes) 和 加权实例(weighted instances)。
更多信息请参照下面的示例和 :meth:`SGDClassifier.fit` 的文档。

.. topic:: 案例:

 - :ref:`sphx_glr_auto_examples_linear_model_plot_sgd_separating_hyperplane.py`,
 - :ref:`sphx_glr_auto_examples_linear_model_plot_sgd_iris.py`
 - :ref:`sphx_glr_auto_examples_linear_model_plot_sgd_weighted_samples.py`
 - :ref:`sphx_glr_auto_examples_linear_model_plot_sgd_comparison.py`
 - :ref:`sphx_glr_auto_examples_svm_plot_separating_hyperplane_unbalanced.py` (See the `Note`)

:class:`SGDClassifier` 支持 averaged SGD (ASGD)。均值化(Averaging)可以通过设置 ``average=True`` 来启用。
ASGD 工作原理是在普通 SGD 的基础上，对每个样本的每次迭代后的系数取均值。
当使用 ASGD 时，学习速率可以更大甚至是恒定，在一些数据集上能够加速训练过程。

对于使用 logistic loss 的分类，在 :class:`LogisticRegression` 中提供了另一个采取 
averaging strategy（平均策略）的 SGD 变体，其使用了随机平均梯度 (SAG) 算法。

回归
==========

:class:`SGDRegressor` 类实现了一个简单的随机梯度下降学习例程，它支持用不同的损失函数和惩罚来拟合线性回归模型。  
:class:`SGDRegressor` 非常适用于有大量训练样本（>10.000)的回归问题，对于其他问题，我们推荐使用 :class:`Ridge`, 
:class:`Lasso`, 或者 :class:`ElasticNet` 。

具体的损失函数可以通过 ``loss`` 参数设置。 :class:`SGDRegressor` 支持以下的损失函数:

  * ``loss="squared_loss"``: Ordinary least squares,
  * ``loss="huber"``: Huber loss for robust regression,
  * ``loss="epsilon_insensitive"``: linear Support Vector Regression.

Huber 和 epsilon-insensitive 损失函数可用于鲁棒回归(robust regression)。
不敏感区域的宽度必须通过参数 ``epsilon`` 来设定。这个参数取决于目标变量的尺度(scale)。

:class:`SGDRegressor` 支持 ASGD（平均随机梯度下降）,就像 :class:`SGDClassifier` 一样。 均值化可以通过设置 ``average=True`` 来启用。

对于利用了平方损失(squared loss) 和 L2 惩罚 的回归算法，在 :class:`Ridge` 中提供了另一个采取平均策略(averaging strategy)的 SGD 变体，
其使用了随机平均梯度 (SAG) 算法。


用于稀疏数据的SGD
===========================================

.. note:: 由于对截距(intercept)使用了收缩的学习速率，导致稀疏实现(sparse implementation)与密集实现(dense implementation)相比产生的结果略有不同。

在 `scipy.sparse <https://docs.scipy.org/doc/scipy/reference/sparse.html>`_ 支持的格式中，任意的矩阵都有对稀疏数据的内置支持方法。
但是，为了获得最高的效率，请使用 `scipy.sparse.csr_matrix <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html>`_ 
中定义的 CSR 矩阵格式。

.. topic:: 案列:

 - :ref:`sphx_glr_auto_examples_text_plot_document_classification_20newsgroups.py`

复杂度
==========

SGD 主要的优势在于它的高效性，对于不同规模的训练样本，处理复杂度基本上是线性的。
假如 X 是 size 为 (n, p) 的矩阵，训练成本为 :math:`O(k n \bar p)`，其中 k 是迭代次数， :math:`\bar p` 是每个样本非零特征的平均数。

但是，最近的理论结果表明，得到我们期望的优化精度的运行时间并不会随着训练集规模扩大而增加。

停止准则
==================

当给定的收敛水平达到以后，:class:`SGDClassifier` 和 :class:`SGDRegressor` 提供了两种停止准则:

  * 如果设定了 ``early_stopping=True``, 输入数据会被划分成训练集和验证集。模型首先在训练集上拟合，
    然后停止准则是基于在验证集上计算出的预测得分进行的。验证集的大小可以由参数 ``validation_fraction`` 来修改。

  * 如果设定了 ``early_stopping=False``, 模型将会在整个输入数据集上进行拟合，此时的停止准则是基于在输入数据上计算出的目标函数。

在上述两种情形下，停止准则会在每一轮(epoch)中评估一次。并且当 "准则不再改进" 这一事件发生了 ``n_iter_no_change`` 次时，该算法就会停止。
"准则的改进" 使用一个容忍度参数(tolerance ``tol``)进行评估。最后，无论何种情况下，只要最大迭代次数 ``max_iter`` 达到以后，算法都会停止。


实用小建议
=====================

  * 随机梯度下降法对特征缩放(feature scaling)很敏感，因此 **强烈建议您缩放您的数据** 。
    例如，将输入向量 X 上的每个特征缩放到 [0,1] 或 [- 1，+1]， 或将其标准化，使其均值为 0，方差为 1。
    请注意，必须将 **相同** 的缩放应用于对应的测试向量中，以获得有意义的结果。使用 :class:`StandardScaler` 很容易做到这一点 ::

      from sklearn.preprocessing import StandardScaler
      scaler = StandardScaler()
      scaler.fit(X_train)  # Don't cheat - fit only on training data
      X_train = scaler.transform(X_train)
      X_test = scaler.transform(X_test)  # apply same transformation to test data

    假如你的 attributes （属性）有一个固有尺度（例如 word frequencies （词频）或 indicator features（指标型特征））就不需要缩放。

  * 最好使用 :class:`GridSearchCV` 找到一个合理的正则化项(regularization term): :math:`\alpha`  ， 它的范围通常在 ``10.0**-np.arange(1,7)``。

  * 经验表明，SGD 在处理约 10^6 训练样本后基本收敛。因此，对于迭代次数第一个合理的猜想是 ``max_iter = np.ceil(10**6 / n)``，其中 ``n`` 是训练集的大小。

  * 如过将 SGD 应用于使用 PCA 做特征提取得到的数据上，我们发现通过某个常数 `c` 来缩放特征值是明智的，这样可以使训练数据的 L2 norm 平均值为 1。

  * 我们发现，当特征很多或 eta0 很大时， ASGD（平均随机梯度下降） 效果更好。

.. topic:: 案例:

 * `"Efficient BackProp" <http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf>`_
   Y. LeCun, L. Bottou, G. Orr, K. Müller - In Neural Networks: Tricks
   of the Trade 1998.

.. _sgd_mathematical_formulation:

数学表达式
========================

给定一组训练样本 :math:`(x_1, y_1), \ldots, (x_n, y_n)` 其中 :math:`x_i \in \mathbf{R}^m` 并且 :math:`y_i \in \{-1,1\}`, 
我们的目标是学习一个线性评分函数(linear scoring function):  :math:`f(x) = w^T x + b` ； :math:`w \in \mathbf{R}^m` 是待学习的模型参数，
:math:`b \in \mathbf{R}` 是待学习的截距。在做预测的时候，我们只需要简单的判断一下 :math:`f(x)` 的正负符号。
寻找模型参数的通常做法就是 最小化一个带有正则化项的训练误差，如下：

.. math::

    E(w,b) = \frac{1}{n}\sum_{i=1}^{n} L(y_i, f(x_i)) + \alpha R(w)

其中 :math:`L` 度量 模型拟合程度的损失函数，:math:`R` 是惩罚模型复杂度的正则化项（也叫作惩罚）; :math:`\alpha > 0` 是一个非负超参数。

:math:`L` 的不同选择会产生本质上不同的分类器，如下所示：

   - Hinge: (soft-margin) Support Vector Machines.
   - Log:   Logistic Regression.
   - Least-Squares: Ridge Regression.
   - Epsilon-Insensitive: (soft-margin) Support Vector Regression.

所有上述损失函数可以看作是错误分类误差（Zero-one loss即0-1损失）的上限，如下图所示：

.. figure:: ../auto_examples/linear_model/images/sphx_glr_plot_sgd_loss_functions_001.png
    :target: ../auto_examples/linear_model/plot_sgd_loss_functions.html
    :align: center
    :scale: 75

比较流行的正则化项 :math:`R` 包括：

   - L2 norm: :math:`R(w) := \frac{1}{2} \sum_{i=1}^{n} w_i^2`,
   - L1 norm: :math:`R(w) := \sum_{i=1}^{n} |w_i|`, 会产生稀疏解，
   - Elastic Net: :math:`R(w) := \frac{\rho}{2} \sum_{i=1}^{n} w_i^2 + (1-\rho) \sum_{i=1}^{n} |w_i|`, L2 和 L1 的凸组合, 其中 :math:`\rho` is given by ``1 - l1_ratio``.

下图显示当 :math:`R(w) = 1` 时参数空间中不同正则项的轮廓。

.. figure:: ../auto_examples/linear_model/images/sphx_glr_plot_sgd_penalties_001.png
    :target: ../auto_examples/linear_model/plot_sgd_penalties.html
    :align: center
    :scale: 75

SGD
---

随机梯度下降法是一种无约束优化问题的优化方法。与（批量）梯度下降法相反，SGD 通过一次只考虑单个训练样本来近似 :math:`E(w,b)` 的真实梯度。

:class:`SGDClassifier` 类实现了一个一阶 SGD 学习程序(first-order SGD learning routine)。 
算法在训练样本上遍历，并且对每个样本根据由以下式子给出的更新规则来更新模型参数:

.. math::

    w \leftarrow w - \eta (\alpha \frac{\partial R(w)}{\partial w}
    + \frac{\partial L(w^T x_i + b, y_i)}{\partial w})

其中 :math:`\eta` 是在参数空间中控制步长的学习速率(learning rate)； 截距(intercept) :math:`b` 的更新方式与权重系数的更新方式类似但不需要正则化。

学习率 :math:`\eta` 可以恒定或者逐渐减小。对于分类来说， 默认的学习率设定方案 （``learning_rate='optimal'``）由下式给出：

.. math::

    \eta^{(t)} = \frac {1}{\alpha  (t_0 + t)}

其中 :math:`t` 是时间步（总共有 `n_samples * n_iter` 个时间步）， :math:`t_0` 是由 Léon Bottou 提出的启发式算法决定的，
这样的话预期的初始更新与预期的权重大小可以保持相当(comparable)（这里假定了训练样本的范数近似为1）。
在 :class:`BaseSGD` 中的 ``_init_t`` 中可以找到确切的定义。

对于回归来说，默认的学习率是逆向缩放(inverse scaling) (``learning_rate='invscaling'``)，由下式给出:

.. math::

    \eta^{(t)} = \frac{eta_0}{t^{power\_t}}

其中 :math:`eta_0` 和 :math:`power\_t` 是用户通过 ``eta0`` 和 ``power_t`` 分别选择的超参数。

如果使用固定的学习速率则设置 ``learning_rate='constant'`` ，或者设置 ``eta0`` 来指定学习速率。

如果要使用自适应下降的学习率, 则设置 ``learning_rate='adaptive'``，并使用 ``eta0`` 指定一个起始学习率。
当停止准则达到以后，学习率会被除以5并且算法不会立刻停止。自适应情况下，算法只有在学习率降低到1e-6时才会停止。

模型参数可以通过成员 ``coef_`` 和 ``intercept_`` 来获得：

     - Member ``coef_`` holds the weights :math:`w`

     - Member ``intercept_`` holds :math:`b`

.. topic:: 参考文献:

 * `"Solving large scale linear prediction problems using stochastic
   gradient descent algorithms"
   <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.58.7377>`_
   T. Zhang - In Proceedings of ICML '04.

 * `"Regularization and variable selection via the elastic net"
   <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.124.4696>`_
   H. Zou, T. Hastie - Journal of the Royal Statistical Society Series B,
   67 (2), 301-320.

 * `"Towards Optimal One Pass Large Scale Learning with
   Averaged Stochastic Gradient Descent"
   <https://arxiv.org/pdf/1107.2490v2.pdf>`_
   Xu, Wei


实现细节
======================

SGD 的实现受到了 Léon Bottou `Stochastic Gradient SVM <https://leon.bottou.org/projects/sgd>`_ 的影响。
类似于 SvmSGD，权重向量表达为一个标量和一个向量的内积，这样保证在使用L2正则项时可以高效更新权重。 
在特征向量稀疏的情况下， intercept （截距）是以更小的学习率（乘以 0.01）更新的，这导致了它的更新更加频繁。
训练样本按顺序选取并且每处理一个样本就要降低学习速率。我们采用了 Shalev-Shwartz 等人2007年提出的的学习速率设定方案。 
对于多类分类，我们使用了 “one versus all” 方法。 我们在 L1 正则化（和 Elastic Net ）中使用 Tsuruoka 等人2009年提出的 
truncated gradient algorithm （截断梯度算法）。代码是用 Cython 编写的。

.. topic:: 参考文献:

 * `"Stochastic Gradient Descent" <https://leon.bottou.org/projects/sgd>`_ L. Bottou - Website, 2010.

 * `"The Tradeoffs of Large Scale Machine Learning" <https://leon.bottou.org/slides/largescale/lstut.pdf>`_ L. Bottou - Website, 2011.

 * `"Pegasos: Primal estimated sub-gradient solver for svm"
   <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.74.8513>`_
   S. Shalev-Shwartz, Y. Singer, N. Srebro - In Proceedings of ICML '07.

 * `"Stochastic gradient descent training for l1-regularized log-linear models with cumulative penalty"
   <https://www.aclweb.org/anthology/P/P09/P09-1054.pdf>`_
   Y. Tsuruoka, J. Tsujii, S. Ananiadou -  In Proceedings of the AFNLP/ACL '09.
