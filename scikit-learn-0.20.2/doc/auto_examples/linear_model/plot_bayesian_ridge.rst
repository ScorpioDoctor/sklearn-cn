.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_linear_model_plot_bayesian_ridge.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_linear_model_plot_bayesian_ridge.py:


=========================
贝叶斯岭回归
=========================

在一个人工合成的数据集上计算贝叶斯岭回归(Bayesian Ridge Regression)。

请参考 :ref:`bayesian_ridge_regression` 获得关于此回归算法的更多详细信息。

与OLS(普通最小二乘)估计器相比，系数权值略向零漂移，从而使其稳定。

由于权重上的先验是高斯先验(Gaussian prior), 所以估计出的权重的直方图是个类似高斯分布的直方图。

模型的估计是通过 迭代地最大化 观测值的边际对数似然(marginal log-likelihood of the observations)来实现的。

我们还用多项式特征展开(polynomial feature expansion)绘制了一维回归情形下的贝叶斯岭回归的预测和不确定性图。
注意到，不确定性值在图的右边开始上升。这是因为这些测试样本超出了训练样本的范围。

翻译者：studyai.com的Antares博士




.. rst-class:: sphx-glr-horizontal


    *

      .. image:: /auto_examples/linear_model/images/sphx_glr_plot_bayesian_ridge_001.png
            :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/linear_model/images/sphx_glr_plot_bayesian_ridge_002.png
            :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/linear_model/images/sphx_glr_plot_bayesian_ridge_003.png
            :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/linear_model/images/sphx_glr_plot_bayesian_ridge_004.png
            :class: sphx-glr-multi-img





.. code-block:: python

    print(__doc__)

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats

    from sklearn.linear_model import BayesianRidge, LinearRegression

    # #############################################################################
    # 用高斯权值(Gaussian weights)生成模拟数据
    np.random.seed(0)
    n_samples, n_features = 100, 100
    X = np.random.randn(n_samples, n_features)  # 产生服从高斯分布的数据
    # 用等于 4 的 precision lambda_ 产生权重。
    lambda_ = 4.
    w = np.zeros(n_features)
    # 只保留 10 个感兴趣的权重
    relevant_features = np.random.randint(0, n_features, 10)
    for i in relevant_features:
        w[i] = stats.norm.rvs(loc=0, scale=1. / np.sqrt(lambda_))
    # 用取值为 50 的 precision alpha_ 产生噪声。
    alpha_ = 50.
    noise = stats.norm.rvs(loc=0, scale=1. / np.sqrt(alpha_), size=n_samples)
    # 产生 目标值
    y = np.dot(X, w) + noise

    # #############################################################################
    # 拟合 贝叶斯岭回归模型 和 最小二乘模型　用于比较
    clf = BayesianRidge(compute_score=True)
    clf.fit(X, y)

    ols = LinearRegression()
    ols.fit(X, y)

    # #############################################################################
    # 画出 真正的权重, 估计出的权重, 权重的直方图，和　伴有标准偏差的预测
    lw = 2
    plt.figure(figsize=(6, 5))
    plt.title("Weights of the model")
    plt.plot(clf.coef_, color='lightgreen', linewidth=lw,
             label="Bayesian Ridge estimate")
    plt.plot(w, color='gold', linewidth=lw, label="Ground truth")
    plt.plot(ols.coef_, color='navy', linestyle='--', label="OLS estimate")
    plt.xlabel("Features")
    plt.ylabel("Values of the weights")
    plt.legend(loc="best", prop=dict(size=12))

    plt.figure(figsize=(6, 5))
    plt.title("Histogram of the weights")
    plt.hist(clf.coef_, bins=n_features, color='gold', log=True,
             edgecolor='black')
    plt.scatter(clf.coef_[relevant_features], np.full(len(relevant_features), 5.),
                color='navy', label="Relevant features")
    plt.ylabel("Features")
    plt.xlabel("Values of the weights")
    plt.legend(loc="upper left")

    plt.figure(figsize=(6, 5))
    plt.title("Marginal log-likelihood")
    plt.plot(clf.scores_, color='navy', linewidth=lw)
    plt.ylabel("Score")
    plt.xlabel("Iterations")


    # 绘制一些　多项式回归的预测
    def f(x, noise_amount):
        y = np.sqrt(x) * np.sin(x)
        noise = np.random.normal(0, 1, len(x))
        return y + noise_amount * noise


    degree = 10
    X = np.linspace(0, 10, 100)
    y = f(X, noise_amount=0.1)
    clf_poly = BayesianRidge()
    clf_poly.fit(np.vander(X, degree), y)

    X_plot = np.linspace(0, 11, 25)
    y_plot = f(X_plot, noise_amount=0)
    y_mean, y_std = clf_poly.predict(np.vander(X_plot, degree), return_std=True)
    plt.figure(figsize=(6, 5))
    plt.errorbar(X_plot, y_mean, y_std, color='navy',
                 label="Polynomial Bayesian Ridge Regression", linewidth=lw)
    plt.plot(X_plot, y_plot, color='gold', linewidth=lw,
             label="Ground Truth")
    plt.ylabel("Output y")
    plt.xlabel("Feature X")
    plt.legend(loc="lower left")
    plt.show()

**Total running time of the script:** ( 0 minutes  0.294 seconds)


.. _sphx_glr_download_auto_examples_linear_model_plot_bayesian_ridge.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_bayesian_ridge.py <plot_bayesian_ridge.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_bayesian_ridge.ipynb <plot_bayesian_ridge.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
