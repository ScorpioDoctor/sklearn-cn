.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_linear_model_plot_ard.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_linear_model_plot_ard.py:


==================================================
自动关联确定 回归 (ARD)
==================================================

使用自动关联确定理论拟合回归模型。

请看 自动关联确定(:ref:`automatic_correlation_determination`) 用于回归的更多内容。

与OLS(普通最小二乘)估计器相比，系数权值略向零漂移，从而使其稳定。

估计出的权重的直方图是非常尖峰的(very peaked)，因为在权重上隐含了从稀疏诱导的先验(sparsity-inducing prior)。

模型的估计是通过 迭代地最大化 观测值的边际对数似然(marginal log-likelihood of the observations)来实现的。

我们还用多项式特征展开(polynomial feature expansion)绘制了一维回归情形下的ARD的预测和不确定性图。 
注意到，不确定性值在图的右边开始上升。这是因为这些测试样本超出了训练样本的范围。

翻译者：studyai.com的Antares博士




.. rst-class:: sphx-glr-horizontal


    *

      .. image:: /auto_examples/linear_model/images/sphx_glr_plot_ard_001.png
            :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/linear_model/images/sphx_glr_plot_ard_002.png
            :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/linear_model/images/sphx_glr_plot_ard_003.png
            :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/linear_model/images/sphx_glr_plot_ard_004.png
            :class: sphx-glr-multi-img





.. code-block:: python


    print(__doc__)

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats

    from sklearn.linear_model import ARDRegression, LinearRegression

    # #############################################################################
    # 用高斯权值(Gaussian weights)生成模拟数据

    # 样本集的参数(样本数量，特征数量)
    np.random.seed(0)
    n_samples, n_features = 100, 100
    # 产生服从高斯分布的数据
    X = np.random.randn(n_samples, n_features)
    # Create weights with a precision lambda_ of 4.
    lambda_ = 4.
    w = np.zeros(n_features)
    # 只保留 10 个感兴趣的权重
    relevant_features = np.random.randint(0, n_features, 10)
    for i in relevant_features:
        w[i] = stats.norm.rvs(loc=0, scale=1. / np.sqrt(lambda_))
    # Create noise with a precision alpha of 50.
    alpha_ = 50.
    noise = stats.norm.rvs(loc=0, scale=1. / np.sqrt(alpha_), size=n_samples)
    # 产生 目标值(target)
    y = np.dot(X, w) + noise

    # #############################################################################
    # 拟合 ARD Regression 模型
    clf = ARDRegression(compute_score=True)
    clf.fit(X, y)
    # 拟合 OLS Regression 模型
    ols = LinearRegression()
    ols.fit(X, y)

    # #############################################################################
    # Plot the true weights, the estimated weights, the histogram of the
    # weights, and predictions with standard deviations
    plt.figure(figsize=(6, 5))
    plt.title("Weights of the model")
    plt.plot(clf.coef_, color='darkblue', linestyle='-', linewidth=2,
             label="ARD estimate")
    plt.plot(ols.coef_, color='yellowgreen', linestyle=':', linewidth=2,
             label="OLS estimate")
    plt.plot(w, color='orange', linestyle='-', linewidth=2, label="Ground truth")
    plt.xlabel("Features")
    plt.ylabel("Values of the weights")
    plt.legend(loc=1)

    plt.figure(figsize=(6, 5))
    plt.title("Histogram of the weights")
    plt.hist(clf.coef_, bins=n_features, color='navy', log=True)
    plt.scatter(clf.coef_[relevant_features], np.full(len(relevant_features), 5.),
                color='gold', marker='o', label="Relevant features")
    plt.ylabel("Features")
    plt.xlabel("Values of the weights")
    plt.legend(loc=1)

    plt.figure(figsize=(6, 5))
    plt.title("Marginal log-likelihood")
    plt.plot(clf.scores_, color='navy', linewidth=2)
    plt.ylabel("Score")
    plt.xlabel("Iterations")


    # Plotting some predictions for polynomial regression
    def f(x, noise_amount):
        y = np.sqrt(x) * np.sin(x)
        noise = np.random.normal(0, 1, len(x))
        return y + noise_amount * noise


    degree = 10
    X = np.linspace(0, 10, 100)
    y = f(X, noise_amount=1)
    clf_poly = ARDRegression(threshold_lambda=1e5)
    clf_poly.fit(np.vander(X, degree), y)

    X_plot = np.linspace(0, 11, 25)
    y_plot = f(X_plot, noise_amount=0)
    y_mean, y_std = clf_poly.predict(np.vander(X_plot, degree), return_std=True)
    plt.figure(figsize=(6, 5))
    plt.errorbar(X_plot, y_mean, y_std, color='navy',
                 label="Polynomial ARD", linewidth=2)
    plt.plot(X_plot, y_plot, color='gold', linewidth=2,
             label="Ground Truth")
    plt.ylabel("Output y")
    plt.xlabel("Feature X")
    plt.legend(loc="lower left")
    plt.show()

**Total running time of the script:** ( 0 minutes  1.474 seconds)


.. _sphx_glr_download_auto_examples_linear_model_plot_ard.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_ard.py <plot_ard.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_ard.ipynb <plot_ard.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
