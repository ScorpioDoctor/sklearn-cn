.. _calibration:

=======================
概率校准(Probability calibration)
=======================

.. currentmodule:: sklearn.calibration

在执行分类时，我们不仅想要预测类标签，而且还要获得相应标签的概率。这个概率给了我们一些关于预测的信心。
有些模型可以给出类概率(class probabilities)的糟糕估计，有些甚至不支持概率预测。
校准模块(calibration module)允许我们更好地校准给定模型的概率，或者添加对概率预测(probability prediction)的支持。

经过良好校准的分类器是概率化分类器(probabilistic classifiers)，它的 ``predict_proba`` 方法的输出可以直接解释为置信水平。
例如，经过良好校准的(二元)分类器应该对样本进行分类，以便在它给出的 predict_proba 值接近0.8的样本中，大约80%实际上属于正类。
下面的图比较了不同分类器的概率预测(probabilistic predictions)的校准效果:

.. figure:: ../auto_examples/calibration/images/sphx_glr_plot_compare_calibration_001.png
   :target: ../auto_examples/calibration/plot_compare_calibration.html
   :align: center

.. currentmodule:: sklearn.linear_model

:class:`LogisticRegression` 在默认情况下返回经过良好校准的预测，因为它直接优化对数损失(log-loss)。相反，其他方法返回有偏概率(biased probabilities)；
每种方法有不同的偏差:

.. currentmodule:: sklearn.naive_bayes

*  :class:`GaussianNB` 倾向于将概率推到0或1(注意直方图中的计数)。这主要是因为它假设 在给定某一类的情况下，
   特征分量相互之间是条件独立的(conditionally independent)，而在包含2个冗余特征的数据集中则假设不成立。

.. currentmodule:: sklearn.ensemble

*  :class:`RandomForestClassifier` 显示了相反的行为：直方图在概率约为0.2和0.9的地方出现峰值，而接近0或1的概率非常罕见。
   Niculescu-Mizil和Caruana对此作了解释 [4]_ ：“ 像bagging和random forests这样的通过对一组基本模型的预测取平均的方法很难在0和1附近做出预测，
   因为底层基本模型中的方差会使本应该接近0或1的预测偏离这些值。因为预测被仅限于区间[0，1]，由方差引起的误差往往是近0和1的单边误差。
   例如，如果一个模型应该对一个情况预测p=0，那么bagging可以实现的唯一方法就是袋子里的所有树(all bagged trees)都预测为零。如果我们给装在袋子里的树添加噪声
   则噪声会导致其中的某些树的预测值大于0，因此这就使得bagging的平均预测偏离了0。在随机森林模型中我们可以更加强烈的观察到这些现象，
   因为随机森林中的基本树估计器(都是在全部特征的一个子集上训练的)都具有相对较高的方差。” 。因此，校准曲线(calibration curve)有时候也称之为
   可靠性图(reliability graph, Wilks 1995 [5]_) 展示了一个 characteristic sigmoid shape，表明分类器更应该相信它们的直觉并返回更接近 0 或 1 的概率。

.. currentmodule:: sklearn.svm

* 线性支持向量分类 (:class:`LinearSVC`) 显示了比 RandomForestClassifier 更多的 Sigmoid 曲线, 
  这对于最大边距方法 (比较 Niculescu-Mizil 和 Caruana [4]_) 是很典型的, 
  其重点是聚焦在靠近决策边界的难分样本(hard samples)，也就是 支持向量(support vectors)。

.. currentmodule:: sklearn.calibration

scikit-learn提供了执行概率预测校准的两种方法: 基于Platt的Sigmoid模型的参数化方法和基于保序回归(isotonic regression)
的非参数方法 (:mod:`sklearn.isotonic`)。概率校准应该在新数据上进行而不是在训练数据上。
类 :class:`CalibratedClassifierCV` 使用交叉验证生成器, 对每个拆分，在训练样本上估计模型参数，在测试样本上进行校准。 
然后对所有拆分上预测的概率进行平均。 已经拟合过的分类器可以通过 :class:`CalibratedClassifierCV` 类传递参数 
``cv="prefit"`` 这种方式进行校准. 在这种情况下, 用户必须手动注意模型拟合的数据和校准的数据是不重叠的。

以下图像展示了概率校准的好处。 第一个图像显示一个具有2个类和3个数据块的数据集. 
中间的数据块包含每个类的随机样本. 此数据块中样本的概率应为 0.5.

.. figure:: ../auto_examples/calibration/images/sphx_glr_plot_calibration_001.png
   :target: ../auto_examples/calibration/plot_calibration.html
   :align: center

以下图像使用没有校准的高斯朴素贝叶斯分类器, 
使用 sigmoid 校准和非参数的isotonic校准来显示上述估计概率的数据. 
可以观察到, 非参数模型为中间样本提供最准确的概率估计, 即 0.5。

.. figure:: ../auto_examples/calibration/images/sphx_glr_plot_calibration_002.png
   :target: ../auto_examples/calibration/plot_calibration.html
   :align: center

.. currentmodule:: sklearn.metrics

在具有20个特征的100,000个样本（其中1,000个用于模型拟合）进行二元分类的人造数据集上进行以下实验.
在这20个特征中，只有2个是informative的, 10个是冗余的。
该图显示了使用logistic regression获得的估计概率, 线性支持向量分类器(SVC)
和具有 isotonic 校准和 sigmoid 校准的linear SVC。Brier score 是一个指标,它是 calibration loss 
和 refinement loss 的结合，函数 :func:`brier_score_loss` 来计算, 
请看下面的图例（越小越好）。 校正损失(Calibration loss)定义为从ROC段斜率导出的经验概率的均方偏差。
细化损失（Refinement loss）可以定义为在最优代价曲线下用面积测量的期望最优损失。

.. figure:: ../auto_examples/calibration/images/sphx_glr_plot_calibration_curve_002.png
   :target: ../auto_examples/calibration/plot_calibration_curve.html
   :align: center

One can observe here that logistic regression is well calibrated as its curve is
nearly diagonal. Linear SVC's calibration curve or reliability diagram has a
sigmoid curve, which is typical for an under-confident classifier. In the case of
LinearSVC, this is caused by the margin property of the hinge loss, which lets
the model focus on hard samples that are close to the decision boundary
(the support vectors). Both kinds of calibration can fix this issue and yield
nearly identical results. The next figure shows the calibration curve of
Gaussian naive Bayes on the same data, with both kinds of calibration and also
without calibration.

.. figure:: ../auto_examples/calibration/images/sphx_glr_plot_calibration_curve_001.png
   :target: ../auto_examples/calibration/plot_calibration_curve.html
   :align: center

One can see that Gaussian naive Bayes performs very badly but does so in an
other way than linear SVC: While linear SVC exhibited a sigmoid calibration
curve, Gaussian naive Bayes' calibration curve has a transposed-sigmoid shape.
This is typical for an over-confident classifier. In this case, the classifier's
overconfidence is caused by the redundant features which violate the naive Bayes
assumption of feature-independence.

Calibration of the probabilities of Gaussian naive Bayes with isotonic
regression can fix this issue as can be seen from the nearly diagonal
calibration curve. Sigmoid calibration also improves the brier score slightly,
albeit not as strongly as the non-parametric isotonic calibration. This is an
intrinsic limitation of sigmoid calibration, whose parametric form assumes a
sigmoid rather than a transposed-sigmoid curve. The non-parametric isotonic
calibration model, however, makes no such strong assumptions and can deal with
either shape, provided that there is sufficient calibration data. In general,
sigmoid calibration is preferable in cases where the calibration curve is sigmoid
and where there is limited calibration data, while isotonic calibration is
preferable for non-sigmoid calibration curves and in situations where large
amounts of data are available for calibration.

.. currentmodule:: sklearn.calibration

:class:`CalibratedClassifierCV` can also deal with classification tasks that
involve more than two classes if the base estimator can do so. In this case,
the classifier is calibrated first for each class separately in an one-vs-rest
fashion. When predicting probabilities for unseen data, the calibrated
probabilities for each class are predicted separately. As those probabilities
do not necessarily sum to one, a postprocessing is performed to normalize them.

The next image illustrates how sigmoid calibration changes predicted
probabilities for a 3-class classification problem. Illustrated is the standard
2-simplex, where the three corners correspond to the three classes. Arrows point
from the probability vectors predicted by an uncalibrated classifier to the
probability vectors predicted by the same classifier after sigmoid calibration
on a hold-out validation set. Colors indicate the true class of an instance
(red: class 1, green: class 2, blue: class 3).

.. figure:: ../auto_examples/calibration/images/sphx_glr_plot_calibration_multiclass_000.png
   :target: ../auto_examples/calibration/plot_calibration_multiclass.html
   :align: center

The base classifier is a random forest classifier with 25 base estimators
(trees). If this classifier is trained on all 800 training datapoints, it is
overly confident in its predictions and thus incurs a large log-loss.
Calibrating an identical classifier, which was trained on 600 datapoints, with
method='sigmoid' on the remaining 200 datapoints reduces the confidence of the
predictions, i.e., moves the probability vectors from the edges of the simplex
towards the center:

.. figure:: ../auto_examples/calibration/images/sphx_glr_plot_calibration_multiclass_001.png
   :target: ../auto_examples/calibration/plot_calibration_multiclass.html
   :align: center

This calibration results in a lower log-loss. Note that an alternative would
have been to increase the number of base estimators which would have resulted in
a similar decrease in log-loss.

.. topic:: 参考文献:

    * Obtaining calibrated probability estimates from decision trees
      and naive Bayesian classifiers, B. Zadrozny & C. Elkan, ICML 2001

    * Transforming Classifier Scores into Accurate Multiclass
      Probability Estimates, B. Zadrozny & C. Elkan, (KDD 2002)

    * Probabilistic Outputs for Support Vector Machines and Comparisons to
      Regularized Likelihood Methods, J. Platt, (1999)

    .. [4] Predicting Good Probabilities with Supervised Learning,
           A. Niculescu-Mizil & R. Caruana, ICML 2005

    .. [5] On the combination of forecast probabilities for
           consecutive precipitation periods. Wea. Forecasting, 5, 640–650.,
           Wilks, D. S., 1990a
