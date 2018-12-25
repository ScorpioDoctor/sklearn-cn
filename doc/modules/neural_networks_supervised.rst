.. _neural_networks_supervised:

==================================
监督神经网络模型(Supervised Neural network models)
==================================

.. currentmodule:: sklearn.neural_network


.. warning::

    此实现不打算用于大规模应用程序。特别是，scikit-learn 没有提供GPU支持。
    为了更快，基于GPU的实现以及框架提供了构建深度学习体系结构的更多灵活性,
    请参考 :ref:`related_projects`。

.. _multilayer_perceptron:

多层感知器
======================

**Multi-layer Perceptron (MLP)** 是一种监督学习算法，通过在数据集上训练学习一个函数 :math:`f(\cdot): R^m \rightarrow R^o` ,
其中 :math:`m` 是输入数据的维数， :math:`o` 是输出数据的维数。给定一个特征向量的集合 :math:`X = {x_1, x_2, ..., x_m}` 和
一个目标 :math:`y` , MLP可以学习一个非线性函数逼近器(non-linear function approximator)用于分类或回归。MLP不同于logistic regression, 
因为MLP在输入层和输出层之间还可以有一个或更多的非线性层，这被称之为 隐藏层(hidden layers)。Figure 1 展示了有一个隐藏层和标量输出的MLP。

.. figure:: ../images/multilayerperceptron_network.png
   :align: center
   :scale: 60%

   **Figure 1 : One hidden layer MLP.**

最左边的层, 称之为输入层(input layer), 由一系列神经元 :math:`\{x_i | x_1, x_2, ..., x_m\}` 构成，表达了输入特征向量(input features)。 
隐藏层的每个神经元使用一个加权线性汇总 :math:`w_1x_1 + w_2x_2 + ... + w_mx_m` 对前面一层的数据进行变换, 变换以后再紧跟着一个非线性激活函数
( :math:`g(\cdot):R \rightarrow R` ) - 比如双曲正切函数(hyperbolic tan function)。 
输出层接受来自最后一个隐藏层的数据然后将其变换为输出值。

该模块有公共属性 ``coefs_`` 和 ``intercepts_`` 。
``coefs_`` 是权重矩阵的列表，其中在第 :math:`i` 个 索引位置上的权重矩阵代表了 :math:`i` 层和 :math:`i+1` 层的权重。
``intercepts_`` 是偏置向量的列表，在第 :math:`i` 个 索引位置上的向量代表了加在 :math:`i+1` 层上的偏置值。

多层感知器的优点如下:

    + 学习非线性模型的能力。

    + 使用 ``partial_fit`` 实时(在线)学习模型的能力。


多层感知器的缺点如下:

    + 带有隐藏层的MLP有非凸(non-convex)损失函数,存在不止一个局部极小值。因此，不同的随机权重初始化会导致不同的验证准确率。

    + MLP 需要调节很多超参数比如 隐藏层的神经元数量，隐藏层的数量以及迭代次数等。

    + MLP 对特征尺度缩放(feature scaling)很敏感。

请查看 :ref:`Tips on Practical Use <mlp_tips>` 小节 ，强调了其中的一些缺点。


分类
==============

类 :class:`MLPClassifier` 实现了一个多层感知器(MLP)算法，使用反向传播进行训练 
`Backpropagation <http://ufldl.stanford.edu/wiki/index.php/Backpropagation_Algorithm>`_ 。

MLP的训练在两个数组上进行：size为 (n_samples, n_features) 的数组 X , 持有以浮点特征向量表示的训练样本；
size为(n_samples,)的数组 y , 持有训练样本的目标值(类标签)::

    >>> from sklearn.neural_network import MLPClassifier
    >>> X = [[0., 0.], [1., 1.]]
    >>> y = [0, 1]
    >>> clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
    ...                     hidden_layer_sizes=(5, 2), random_state=1)
    ...
    >>> clf.fit(X, y)                         # doctest: +NORMALIZE_WHITESPACE
    MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
                  beta_1=0.9, beta_2=0.999, early_stopping=False,
                  epsilon=1e-08, hidden_layer_sizes=(5, 2),
                  learning_rate='constant', learning_rate_init=0.001,
                  max_iter=200, momentum=0.9, n_iter_no_change=10,
                  nesterovs_momentum=True, power_t=0.5, random_state=1,
                  shuffle=True, solver='lbfgs', tol=0.0001,
                  validation_fraction=0.1, verbose=False, warm_start=False)

训练好以后, 模型就可以对新的样本预测其类别标签啦 ::

    >>> clf.predict([[2., 2.], [-1., -2.]])
    array([1, 0])

MLP 可以为给定的训练数据拟合一个非线性模型。 ``clf.coefs_`` 包含构成模型的所有权重矩阵 ::

    >>> [coef.shape for coef in clf.coefs_]
    [(2, 5), (5, 2), (2, 1)]

目前为止, :class:`MLPClassifier` 仅支持交叉熵损失(Cross-Entropy loss)函数, 这就允许我们通过运行 ``predict_proba`` 方法进行概率估计了。

MLP 使用反向传播(Backpropagation)进行训练。更准确的说, 它使用某种形式的梯度下降法进行训练，而梯度的计算用到了反向传播技术。
对于分类来说，它最小化交叉熵损失函数,为每个样本 :math:`x` 给出一个该样本属于各个类别的概率的估计 :math:`P(y|x)` ::

    >>> clf.predict_proba([[2., 2.], [1., 2.]])  # doctest: +ELLIPSIS
    array([[1.967...e-04, 9.998...-01],
           [1.967...e-04, 9.998...-01]])

:class:`MLPClassifier` 类通过使用软最大化(`Softmax <https://en.wikipedia.org/wiki/Softmax_activation_function>`_)
作为输出函数 支持多类别分类。

更进一步, MLP模型还支持多标签分类(:ref:`multi-label classification <multiclass>`),其中一个样本可以属于不止一个类。
对每个类，原始输出会被传递到logistic函数。凡是大于等于0.5的值就被弄成1，反之则为0。对一个样本的一个预测输出(向量)，
值为 `1` 的那个索引位置代表了该样本被分配的类别 ::

    >>> X = [[0., 0.], [1., 1.]]
    >>> y = [[0, 1], [1, 1]]
    >>> clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
    ...                     hidden_layer_sizes=(15,), random_state=1)
    ...
    >>> clf.fit(X, y)                         # doctest: +NORMALIZE_WHITESPACE
    MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
                  beta_1=0.9, beta_2=0.999, early_stopping=False,
                  epsilon=1e-08, hidden_layer_sizes=(15,),
                  learning_rate='constant', learning_rate_init=0.001,
                  max_iter=200, momentum=0.9, n_iter_no_change=10,
                  nesterovs_momentum=True, power_t=0.5,  random_state=1,
                  shuffle=True, solver='lbfgs', tol=0.0001,
                  validation_fraction=0.1, verbose=False, warm_start=False)
    >>> clf.predict([[1., 2.]])
    array([[1, 1]])
    >>> clf.predict([[0., 0.]])
    array([[0, 1]])

请看下面的例子和文档 详细了解 :meth:`MLPClassifier.fit` 的更多信息。

.. topic:: 案例:

 * :ref:`sphx_glr_auto_examples_neural_networks_plot_mlp_training_curves.py`
 * :ref:`sphx_glr_auto_examples_neural_networks_plot_mnist_filters.py`

回归
==========

:class:`MLPRegressor` 类实现了一个多层感知器(MLP)，使用反向传播训练，输出层没有激活函数，也可以认为输出层的激活函数是 identity function。
因此，它使用平方误差(square error)作为损失函数, 输出是一系列连续值。

:class:`MLPRegressor` 类也支持 多输出回归(multi-output regression), 每一个样本有不止一个目标值(target)。

正则化
==============

:class:`MLPRegressor` 和 :class:`MLPClassifier` 使用参数 ``alpha`` 来控制正则化量(L2 正则化)以通过对权重的惩罚避免过拟合发生。
下面的图绘制了 决策函数随着 alpha 值的变化而变化。

.. figure:: ../auto_examples/neural_networks/images/sphx_glr_plot_mlp_alpha_001.png
   :target: ../auto_examples/neural_networks/plot_mlp_alpha.html
   :align: center
   :scale: 75

请看下面的例子获得更多信息。

.. topic:: 案例:

 * :ref:`sphx_glr_auto_examples_neural_networks_plot_mlp_alpha.py`

算法
==========

MLP 使用 `Stochastic Gradient Descent <https://en.wikipedia.org/wiki/Stochastic_gradient_descent>`_, 
`Adam <https://arxiv.org/abs/1412.6980>`_, 或 
`L-BFGS <https://en.wikipedia.org/wiki/Limited-memory_BFGS>`__ 进行训练。
Stochastic Gradient Descent (SGD) 使用损失函数相对需要更新的那个参数的梯度对参数进行更新。如下所示：

.. math::

    w \leftarrow w - \eta (\alpha \frac{\partial R(w)}{\partial w}
    + \frac{\partial Loss}{\partial w})

其中 :math:`\eta` 是控制参数空间搜索步长的学习率。 :math:`Loss` 是网络使用的损失函数。

更多详细信息请参考 `SGD <http://scikit-learn.org/stable/modules/sgd.html>`_

Adam 与 SGD 类似，因为它也是一个随机优化器(stochastic optimizer), 但是它可以根据低阶矩(lower-order moments)的自适应估计，自动调整参数更新量。

使用 SGD 或 Adam, 训练过程支持在线和小批量学习(online and mini-batch learning)。

L-BFGS 是一个逼近Hessian matrix的求解器，Hessian matrix是一个函数的二阶偏导数。
更进一步，它通过逼近 Hessian matrix 的逆来执行参数更新。该算法的实现使用了SciPy的版本 
`L-BFGS <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html>`_ 。

如果选择了 'L-BFGS', 训练过程不支持 在线学习 或 小批量学习 。


复杂度
==========

假定我们有 :math:`n` 个训练样本, :math:`m` 个特征分量, :math:`k` 个隐藏层, 
每个包含 :math:`h` 个神经元(简单点儿), :math:`o` 个输出神经元。反向传播的时间复杂度是 
:math:`O(n\cdot m \cdot h^k \cdot o \cdot i)`, 其中 :math:`i` 是迭代次数。
因为反向传播有很高的时间复杂度，建议大家从一个较小数量的隐藏层以及隐藏层神经元数量 开始设计模型并用于训练。


数学表达式
========================

给定一个训练样本集合 :math:`(x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n)`
其中 :math:`x_i \in \mathbf{R}^n` 和 :math:`y_i \in \{0, 1\}`, 具有一个隐藏层一个隐藏神经元的MLP
学习这样一个函数 :math:`f(x) = W_2 g(W_1^T x + b_1) + b_2`
其中 :math:`W_1 \in \mathbf{R}^m` 和 :math:`W_2, b_1, b_2 \in \mathbf{R}` 是
模型参数。 :math:`W_1, W_2` 分别表示输入层和隐藏层的权重； :math:`b_1, b_2` 分别表示添加到隐藏层和输出层的偏置。
:math:`g(\cdot) : R \rightarrow R` 是激活函数, 默认设为 双曲正切函数(hyperbolic tan)，其表达式如下：

.. math::
      g(z)= \frac{e^z-e^{-z}}{e^z+e^{-z}}

For binary classification, :math:`f(x)` passes through the logistic function
:math:`g(z)=1/(1+e^{-z})` to obtain output values between zero and one. A
threshold, set to 0.5, would assign samples of outputs larger or equal 0.5
to the positive class, and the rest to the negative class.

If there are more than two classes, :math:`f(x)` itself would be a vector of
size (n_classes,). Instead of passing through logistic function, it passes
through the softmax function, which is written as,

.. math::
      \text{softmax}(z)_i = \frac{\exp(z_i)}{\sum_{l=1}^k\exp(z_l)}

where :math:`z_i` represents the :math:`i` th element of the input to softmax,
which corresponds to class :math:`i`, and :math:`K` is the number of classes.
The result is a vector containing the probabilities that sample :math:`x`
belong to each class. The output is the class with the highest probability.

In regression, the output remains as :math:`f(x)`; therefore, output activation
function is just the identity function.

MLP uses different loss functions depending on the problem type. The loss
function for classification is Cross-Entropy, which in binary case is given as,

.. math::

    Loss(\hat{y},y,W) = -y \ln {\hat{y}} - (1-y) \ln{(1-\hat{y})} + \alpha ||W||_2^2

where :math:`\alpha ||W||_2^2` is an L2-regularization term (aka penalty)
that penalizes complex models; and :math:`\alpha > 0` is a non-negative
hyperparameter that controls the magnitude of the penalty.

For regression, MLP uses the Square Error loss function; written as,

.. math::

    Loss(\hat{y},y,W) = \frac{1}{2}||\hat{y} - y ||_2^2 + \frac{\alpha}{2} ||W||_2^2


Starting from initial random weights, multi-layer perceptron (MLP) minimizes
the loss function by repeatedly updating these weights. After computing the
loss, a backward pass propagates it from the output layer to the previous
layers, providing each weight parameter with an update value meant to decrease
the loss.

In gradient descent, the gradient :math:`\nabla Loss_{W}` of the loss with respect
to the weights is computed and deducted from :math:`W`.
More formally, this is expressed as,

.. math::
    W^{i+1} = W^i - \epsilon \nabla {Loss}_{W}^{i}


where :math:`i` is the iteration step, and :math:`\epsilon` is the learning rate
with a value larger than 0.

当达到预设的最大迭代次数时，该算法停止；或 当损失的改善低于某个较小的数字时也会停止。



.. _mlp_tips:

实用小建议
=====================

  * MLP 对特征尺度缩放(feature scaling)很敏感，因此，强烈建议对你的数据进行尺度变换(scale your data) 。
    比如，把输入向量 X 的 每一个属性(特征分量)缩放到 [0, 1] 或 [-1, +1], 或者 将其标准化到 均值为0方差为1。
    请注意，你必须对测试数据也应用 **相同** 的变换操作以获得有意义的结果。
    
    你可以使用 :class:`StandardScaler` 类进行数据标准化变换(standardization)。

      >>> from sklearn.preprocessing import StandardScaler  # doctest: +SKIP
      >>> scaler = StandardScaler()  # doctest: +SKIP
      >>> # Don't cheat - fit only on training data
      >>> scaler.fit(X_train)  # doctest: +SKIP
      >>> X_train = scaler.transform(X_train)  # doctest: +SKIP
      >>> # apply same transformation to test data
      >>> X_test = scaler.transform(X_test)  # doctest: +SKIP

    另一种可替代的推荐方案是在管道流 :class:`Pipeline` 中使用 :class:`StandardScaler` 。

  * 找到一个合理的正则化参数 :math:`\alpha` 的最好方法是使用网格搜索交叉验证类 :class:`GridSearchCV`, 
    这个参数的取值通常在 ``10.0 ** -np.arange(1, 7)`` 区间内。

  * 经验上来说, 我们观察到 `L-BFGS` 优化器 在小数据集上收敛较快结果也更好点儿。对相对较大的数据集， `Adam` 优化器非常鲁棒。
    它通常快速收敛并给出相当好的结果。另一方面，如果学习率被正确的调节的话，带有 momentum 或 nesterov's momentum 的 `SGD` 
    比其他的两种优化器表现更好。

使用warm_start进行更多控制
============================
如果您希望更多地控制SGD中的停止标准或学习速度，或者希望进行额外的监视，
使用 ``warm_start=True`` 和 ``max_iter=1`` 并自己进行迭代过程是很有用的::

    >>> X = [[0., 0.], [1., 1.]]
    >>> y = [0, 1]
    >>> clf = MLPClassifier(hidden_layer_sizes=(15,), random_state=1, max_iter=1, warm_start=True)
    >>> for i in range(10):
    ...     clf.fit(X, y)
    ...     # additional monitoring / inspection # doctest: +ELLIPSIS
    MLPClassifier(...

.. topic:: 参考文献:

    * `"Learning representations by back-propagating errors."
      <https://www.iro.umontreal.ca/~pift6266/A06/refs/backprop_old.pdf>`_
      Rumelhart, David E., Geoffrey E. Hinton, and Ronald J. Williams.

    * `"Stochastic Gradient Descent" <https://leon.bottou.org/projects/sgd>`_ L. Bottou - Website, 2010.

    * `"Backpropagation" <http://ufldl.stanford.edu/wiki/index.php/Backpropagation_Algorithm>`_
      Andrew Ng, Jiquan Ngiam, Chuan Yu Foo, Yifan Mai, Caroline Suen - Website, 2011.

    * `"Efficient BackProp" <http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf>`_
      Y. LeCun, L. Bottou, G. Orr, K. Müller - In Neural Networks: Tricks
      of the Trade 1998.

    *  `"Adam: A method for stochastic optimization."
       <https://arxiv.org/pdf/1412.6980v8.pdf>`_
       Kingma, Diederik, and Jimmy Ba. arXiv preprint arXiv:1412.6980 (2014).
