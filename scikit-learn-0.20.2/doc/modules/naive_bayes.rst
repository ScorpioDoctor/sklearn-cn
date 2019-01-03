.. _naive_bayes:

=========================
朴素贝叶斯(Naive Bayes)
=========================

.. currentmodule:: sklearn.naive_bayes


朴素贝叶斯方法是基于贝叶斯定理的一组有监督学习算法，即“简单”地假设每个类的各个特征分量之间相互条件独立(conditional independence)。
给定一个类别变量 :math:`y` 和一个从 :math:`x_1` 到 :math:`x_n` 的分量之间存在相关关系的特征向量(dependent feature vector)， 贝叶斯定理阐述了以下关系:

.. math::

   P(y \mid x_1, \dots, x_n) = \frac{P(y) P(x_1, \dots x_n \mid y)}
                                    {P(x_1, \dots, x_n)}

根据朴素的条件独立假设(naive conditional independence assumption),下面的这个公式就成立啦：

.. math::

   P(x_i | y, x_1, \dots, x_{i-1}, x_{i+1}, \dots, x_n) = P(x_i | y),

对所有的 :math:`i`, 上面的第一个公式就可以简化成下面这样：

.. math::

   P(y \mid x_1, \dots, x_n) = \frac{P(y) \prod_{i=1}^{n} P(x_i \mid y)}
                                    {P(x_1, \dots, x_n)}

因为给定输入的情况下 :math:`P(x_1, \dots, x_n)` 是一个常量, 我们就可以使用下面的分类规则了啦：

.. math::

   P(y \mid x_1, \dots, x_n) \propto P(y) \prod_{i=1}^{n} P(x_i \mid y)

   \Downarrow

   \hat{y} = \arg\max_y P(y) \prod_{i=1}^{n} P(x_i \mid y),

然后，上面公式里的 :math:`P(y)` 和 :math:`P(x_i \mid y)` 可以用最大后验概率估计方法(Maximum A Posteriori (MAP))估计得到;
类 :math:`y` 的先验概率密度分布 :math:`P(y)` 可以使用 类 :math:`y` 的样本在整个训练集中所占的比率估计出来。

各种朴素贝叶斯方法的主要区别在于它们对 **类条件概率密度分布(class conditional pdf)** :math:`P(x_i \mid y)` 做出的假定不一样。

尽管其假设过于简单，在很多实际情况下，朴素贝叶斯工作得很好，特别是文档分类和垃圾邮件过滤。
这些工作都要求 在一个小的训练集上估计必需的参数。
(至于为什么朴素贝叶斯表现得好的理论原因和它适用于哪些类型的数据，请参见下面的参考文档。)

相比于其他更复杂的方法，朴素贝叶斯学习器和分类器非常快。 **类条件分布的解耦意味着可以独立单独地把每个特征分量的pdf视为一维分布来估计** 。
这样反过来有助于缓解维度灾难带来的问题。

另一方面，尽管朴素贝叶斯被认为是一种相当不错的分类器，但却不是好的估计器(estimator)，所以不能太过于重视从 ``predict_proba`` 输出的概率。

.. topic:: 参考文献:

 * H. Zhang (2004). `The optimality of Naive Bayes.
   <http://www.cs.unb.ca/~hzhang/publications/FLAIRS04ZhangH.pdf>`_
   Proc. FLAIRS.

.. _gaussian_naive_bayes:

高斯朴素贝叶斯
--------------------

:class:`GaussianNB` 实现了运用于分类的高斯朴素贝叶斯算法。每个特征分量的似然函数，也就是类条件概率密度被假设为服从高斯分布:

.. math::

   P(x_i \mid y) = \frac{1}{\sqrt{2\pi\sigma^2_y}} \exp\left(-\frac{(x_i - \mu_y)^2}{2\sigma^2_y}\right)

参数 :math:`\sigma_y` 和 :math:`\mu_y` 可以用极大似然估计法(maximum likelihood)估计出来。

    >>> from sklearn import datasets
    >>> iris = datasets.load_iris()
    >>> from sklearn.naive_bayes import GaussianNB
    >>> gnb = GaussianNB()
    >>> y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
    >>> print("Number of mislabeled points out of a total %d points : %d"
    ...       % (iris.data.shape[0],(iris.target != y_pred).sum()))
    Number of mislabeled points out of a total 150 points : 6

.. _multinomial_naive_bayes:

多项分布朴素贝叶斯
-----------------------

:class:`MultinomialNB` 实现了服从多项分布数据的朴素贝叶斯算法，也是用于文本分类(这个领域中数据往往以词向量表示，
尽管在实践中 tf-idf 向量在预测时表现良好)的两大经典朴素贝叶斯算法之一。 
每个类 :math:`y` 的分布由 :math:`\theta_y = (\theta_{y1},\ldots,\theta_{yn})` 向量进行参数化表示， 
式中 :math:`n` 是特征的数量(对于文本分类，是词汇量的大小) 。
:math:`\theta_{yi}` 是特征 :math:`i` 出现在类 :math:`y` 的样本中的概率 :math:`P(x_i \mid y)` 
(译者注：其实就是类条件概率密度)。

参数向量 :math:`\theta_y` 使用极大似然估计的平滑版本(a smoothed version of maximum likelihood) 进行估计, 
i.e. 相对频率计数(relative frequency counting):

.. math::

    \hat{\theta}_{yi} = \frac{ N_{yi} + \alpha}{N_y + \alpha n}

其中 :math:`N_{yi} = \sum_{x \in T} x_i` 是训练集 :math:`T` 中特征 :math:`i` 出现在类 :math:`y` 的样本中的次数，
:math:`N_{y} = \sum_{i=1}^{n} N_{yi}` 是类 :math:`y` 中出现的所有特征的计数总和。

先验平滑因子 :math:`\alpha \ge 0` 应用于在学习样本中没有出现的特征，以防在将来的计算中出现0概率输出。
如果设置 :math:`\alpha = 1` 则被称为 拉普拉斯平滑(Laplace smoothing),
如果 :math:`\alpha < 1` 则被称为 Lidstone smoothing.

.. _complement_naive_bayes:

Complement Naive Bayes
----------------------

:class:`ComplementNB` implements the complement naive Bayes (CNB) algorithm.
CNB is an adaptation of the standard multinomial naive Bayes (MNB) algorithm
that is particularly suited for imbalanced data sets. Specifically, CNB uses
statistics from the *complement* of each class to compute the model's weights.
The inventors of CNB show empirically that the parameter estimates for CNB are
more stable than those for MNB. Further, CNB regularly outperforms MNB (often
by a considerable margin) on text classification tasks. The procedure for
calculating the weights is as follows:

.. math::

    \hat{\theta}_{ci} = \frac{\alpha_i + \sum_{j:y_j \neq c} d_{ij}}
                             {\alpha + \sum_{j:y_j \neq c} \sum_{k} d_{kj}}

    w_{ci} = \log \hat{\theta}_{ci}

    w_{ci} = \frac{w_{ci}}{\sum_{j} |w_{cj}|}

where the summations are over all documents :math:`j` not in class :math:`c`,
:math:`d_{ij}` is either the count or tf-idf value of term :math:`i` in document
:math:`j`, :math:`\alpha_i` is a smoothing hyperparameter like that found in
MNB, and :math:`\alpha = \sum_{i} \alpha_i`. The second normalization addresses
the tendency for longer documents to dominate parameter estimates in MNB. The
classification rule is:

.. math::

    \hat{c} = \arg\min_c \sum_{i} t_i w_{ci}

i.e., a document is assigned to the class that is the *poorest* complement
match.

.. topic:: References:

 * Rennie, J. D., Shih, L., Teevan, J., & Karger, D. R. (2003).
   `Tackling the poor assumptions of naive bayes text classifiers.
   <https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf>`_
   In ICML (Vol. 3, pp. 616-623).

.. _bernoulli_naive_bayes:

伯努利朴素贝叶斯
---------------------

:class:`BernoulliNB` 实现了用于多变量伯努利分布(multivariate Bernoulli distributions)数据的朴素贝叶斯训练和分类算法，
即有多个特征，但每个特征都假设是一个二元 (Bernoulli, boolean) 变量。 因此，这类算法要求样本以二元化特征向量表示(binary-valued
feature vectors)；如果样本含有其他类型的数据， 一个 ``BernoulliNB`` 类的实例会将其二值化(依赖于 ``binarize`` 参数)。

伯努利朴素贝叶斯的决策规则是基于以下公式：

.. math::

    P(x_i \mid y) = P(i \mid y) x_i + (1 - P(i \mid y)) (1 - x_i)

与多项分布朴素贝叶斯的规则不同 伯努利朴素贝叶斯显式地惩罚作为类 :math:`y` 的指示因子或标识因子(indicator)的不出现，
而多项分布朴素贝叶斯只是简单地忽略没出现的特征。

在文本分类的例子中，词频向量(word occurrence vectors)(而非词数向量(word count vectors))可能用于训练和使用这个分类器。 
``BernoulliNB`` 可能在一些数据集上可能表现得更好，特别是那些更短的文档。 如果时间允许，建议对两个模型都进行评估。

.. topic:: 参考文献:

 * C.D. Manning, P. Raghavan and H. Schütze (2008). Introduction to
   Information Retrieval. Cambridge University Press, pp. 234-265.

 * A. McCallum and K. Nigam (1998).
   `A comparison of event models for Naive Bayes text classification.
   <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.46.1529>`_
   Proc. AAAI/ICML-98 Workshop on Learning for Text Categorization, pp. 41-48.

 * V. Metsis, I. Androutsopoulos and G. Paliouras (2006).
   `Spam filtering with Naive Bayes -- Which Naive Bayes?
   <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.61.5542>`_
   3rd Conf. on Email and Anti-Spam (CEAS).


堆外朴素贝叶斯模型拟合
-------------------------------------

朴素贝叶斯模型可以解决整个训练集不能导入内存的大规模分类问题。 为了解决这个问题，:class:`MultinomialNB`, 
:class:`BernoulliNB`, 和 :class:`GaussianNB` 实现了 ``partial_fit`` 方法，可以动态的增加数据，
使用方法与其他分类器的一样，使用示例见 :ref:`sphx_glr_auto_examples_applications_plot_out_of_core_classification.py` 。
所有的朴素贝叶斯分类器都支持样本权重。

与 ``fit`` 方法不同，首次调用 ``partial_fit`` 方法需要传递一个所有期望的类标签的列表。

对于 scikit-learn 中可用方案的概览，另见 :ref:`out-of-core learning <scaling_strategies>` 文档。

.. note::

   所有朴素贝叶斯模型调用 ``partial_fit`` 都会引入一些计算开销。推荐让数据块(data chunk)越大越好，
   其大小与 RAM 中可用内存大小相同。
