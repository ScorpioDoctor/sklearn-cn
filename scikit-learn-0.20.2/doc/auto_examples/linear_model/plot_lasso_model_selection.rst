.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_linear_model_plot_lasso_model_selection.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_linear_model_plot_lasso_model_selection.py:


===================================================
Lasso 模型选择: 交叉验证 / AIC准则 / BIC准则
===================================================

利用Akaike信息准则(AIC)、Bayes信息准则(BIC)和交叉验证，
选择 :ref:`lasso` 估计器正则化参数 alpha 的最优值。

使用 LassoLarsIC 估计器获得的结果 是基于 AIC/BIC 准则的。

基于信息准则(Information-criterion)的模型选择是非常快速的，但是 这种方法
依赖于对自由度的合理估计。而自由度是 从大量样本(渐近结果)以及假定模型是正确的 的前提下导出的,
也就是说 你的数据恰好(实际上)就是你所选择的模型产生的。
当问题条件不好时(特征多于样本)，这种方法也会崩溃。

对于交叉验证，我们使用2种算法在20-fold上计算Lasso路径(path)：
坐标下降(由LassoCV类实现)和Lars(最小角回归)(由LassoLarsCV类实现)。
这两种算法给出的结果大致相同。它们在执行速度和数值误差来源方面存在差异。

Lars只为路径中的每个扭结(kink)计算其路径解(path solution)。
因此，当只有很少的扭结时，它是非常有效的，如果有很少的特征或样本那么扭结就会很少。
此外，它能够计算完整的路径而不设置任何元参数。
相反，坐标下降法计算预先指定的网格上的路径点(这里我们使用默认值)。
因此，如果网格点的数目小于路径中的扭结数，坐标下降法则效率更高。
如果特征的数量真的非常多，并且有足够的样本来选择大量的特性，那么这样的策略可能会很有趣。
在数值误差方面，对于高度相关的变量，Lars会积累更多的误差，而坐标下降算法只会对网格上的路径进行采样。

注意alpha的最优值在每一个fold上是如何变化的。这说明了为什么在试图评估通过交叉验证选择参数的方法的性能时，
嵌套交叉验证是必要的：对于未见数据，这种参数选择可能不是最优的。




.. rst-class:: sphx-glr-horizontal


    *

      .. image:: /auto_examples/linear_model/images/sphx_glr_plot_lasso_model_selection_001.png
            :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/linear_model/images/sphx_glr_plot_lasso_model_selection_002.png
            :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/linear_model/images/sphx_glr_plot_lasso_model_selection_003.png
            :class: sphx-glr-multi-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Computing regularization path using the coordinate descent lasso...
    Computing regularization path using the Lars lasso...




|


.. code-block:: python

    print(__doc__)

    # Author: Olivier Grisel, Gael Varoquaux, Alexandre Gramfort
    # License: BSD 3 clause
    # 翻译者：studyai.com的Antares博士


    import time

    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC
    from sklearn import datasets

    diabetes = datasets.load_diabetes()
    X = diabetes.data
    y = diabetes.target

    rng = np.random.RandomState(42)
    X = np.c_[X, rng.randn(X.shape[0], 14)]  # 添加一些坏特征

    # normalize data as done by Lars to allow for comparison
    X /= np.sqrt(np.sum(X ** 2, axis=0))

    # #############################################################################
    # LassoLarsIC: 使用 BIC/AIC 准则的最小角回归(Lars)

    model_bic = LassoLarsIC(criterion='bic')
    t1 = time.time()
    model_bic.fit(X, y)
    t_bic = time.time() - t1
    alpha_bic_ = model_bic.alpha_

    model_aic = LassoLarsIC(criterion='aic')
    model_aic.fit(X, y)
    alpha_aic_ = model_aic.alpha_


    def plot_ic_criterion(model, name, color):
        alpha_ = model.alpha_
        alphas_ = model.alphas_
        criterion_ = model.criterion_
        plt.plot(-np.log10(alphas_), criterion_, '--', color=color,
                 linewidth=3, label='%s criterion' % name)
        plt.axvline(-np.log10(alpha_), color=color, linewidth=3,
                    label='alpha: %s estimate' % name)
        plt.xlabel('-log(alpha)')
        plt.ylabel('criterion')

    plt.figure()
    plot_ic_criterion(model_aic, 'AIC', 'b')
    plot_ic_criterion(model_bic, 'BIC', 'r')
    plt.legend()
    plt.title('Information-criterion for model selection (training time %.3fs)' % t_bic)

    # #############################################################################
    # LassoCV: 梯度下降法(coordinate descent)

    # 计算正则化路径
    print("Computing regularization path using the coordinate descent lasso...")
    t1 = time.time()
    model = LassoCV(cv=20).fit(X, y)
    t_lasso_cv = time.time() - t1

    # 展示结果
    m_log_alphas = -np.log10(model.alphas_)

    plt.figure()
    ymin, ymax = 2300, 3800
    plt.plot(m_log_alphas, model.mse_path_, ':')
    plt.plot(m_log_alphas, model.mse_path_.mean(axis=-1), 'k',
             label='Average across the folds', linewidth=2)
    plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
                label='alpha: CV estimate')

    plt.legend()

    plt.xlabel('-log(alpha)')
    plt.ylabel('Mean square error')
    plt.title('Mean square error on each fold: coordinate descent '
              '(train time: %.2fs)' % t_lasso_cv)
    plt.axis('tight')
    plt.ylim(ymin, ymax)

    # #############################################################################
    # LassoLarsCV: 最小角回归(least angle regression)

    # 计算正则化路径
    print("Computing regularization path using the Lars lasso...")
    t1 = time.time()
    model = LassoLarsCV(cv=20).fit(X, y)
    t_lasso_lars_cv = time.time() - t1

    # 展示结果
    m_log_alphas = -np.log10(model.cv_alphas_)

    plt.figure()
    plt.plot(m_log_alphas, model.mse_path_, ':')
    plt.plot(m_log_alphas, model.mse_path_.mean(axis=-1), 'k',
             label='Average across the folds', linewidth=2)
    plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
                label='alpha CV')
    plt.legend()

    plt.xlabel('-log(alpha)')
    plt.ylabel('Mean square error')
    plt.title('Mean square error on each fold: Lars (train time: %.2fs)'
              % t_lasso_lars_cv)
    plt.axis('tight')
    plt.ylim(ymin, ymax)

    plt.show()

**Total running time of the script:** ( 0 minutes  0.677 seconds)


.. _sphx_glr_download_auto_examples_linear_model_plot_lasso_model_selection.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_lasso_model_selection.py <plot_lasso_model_selection.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_lasso_model_selection.ipynb <plot_lasso_model_selection.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
