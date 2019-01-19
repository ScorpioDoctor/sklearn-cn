.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_plot_kernel_ridge_regression.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_plot_kernel_ridge_regression.py:


=============================================
核岭回归(KRR)与支持向量回归(SVR)的比较
=============================================

核岭回归(KRR)与支持向量回归(SVR)都可以利用核技巧来学习非线性函数，也就是说, 
它们可以在相应的核诱导出的空间中学习一个线性函数。而该核诱导空间的线性函数对应于原始空间的非线性函数。
它们的区别在于损失函数(ridge versus epsilon-insensitive loss)。
与SVR相比较, KRR的拟合可以用闭合形式(closed-form)完成，并且在中等规模的数据集上通常更快。
另一方面，KRR学习到的模型是非稀疏的，所以在预测阶段KRR比SVR要慢。

下面这个案例展示了将这两个方法用于人工数据集，它由一个正弦目标函数和每五个数据点所加的强噪声组成。
第一个图比较了在使用网格搜索优化了RBF核的正则化和带宽时，KRR和SVR学习到的模型。它们学习到的函数是非常相似的，
但是，拟合KRR大约比拟合SVR快7倍(都使用了网格搜索)。
然而，使用SVR预测100000个目标值要比KRR快三倍，因为它只使用了100项训练数据中的大约1/3作为支持向量就学到了稀疏模型。

下一个图比较了在不同训练集上KRR和SVR的拟合和预测时间。对于中等规模的训练集(小于1000个样本)，拟合KRR比SVR更快；
然而，对于较大的训练集，SVR的时间弹性更好。在预测时间方面，由于学习到的稀疏解，SVR对于所有训练集的大小都比KRR更快。
请注意，稀疏度和预测时间取决于SVR的参数epsilon和C。

翻译者： http://www.studyai.com/antares




.. rst-class:: sphx-glr-horizontal


    *

      .. image:: /auto_examples/images/sphx_glr_plot_kernel_ridge_regression_001.png
            :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_plot_kernel_ridge_regression_002.png
            :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_plot_kernel_ridge_regression_003.png
            :class: sphx-glr-multi-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    SVR complexity and bandwidth selected and model fitted in 0.586 s
    KRR complexity and bandwidth selected and model fitted in 0.217 s
    Support vector ratio: 0.320
    SVR prediction for 100000 inputs in 0.172 s
    KRR prediction for 100000 inputs in 0.258 s




|


.. code-block:: python


    # Authors: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
    # License: BSD 3 clause


    from __future__ import division
    import time

    import numpy as np

    from sklearn.svm import SVR
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import learning_curve
    from sklearn.kernel_ridge import KernelRidge
    import matplotlib.pyplot as plt

    rng = np.random.RandomState(0)

    # #############################################################################
    # Generate sample data
    X = 5 * rng.rand(10000, 1)
    y = np.sin(X).ravel()

    # Add noise to targets
    y[::5] += 3 * (0.5 - rng.rand(X.shape[0] // 5))

    X_plot = np.linspace(0, 5, 100000)[:, None]

    # #############################################################################
    # Fit regression model
    train_size = 100
    svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
                       param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                                   "gamma": np.logspace(-2, 2, 5)})

    kr = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=5,
                      param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                                  "gamma": np.logspace(-2, 2, 5)})

    t0 = time.time()
    svr.fit(X[:train_size], y[:train_size])
    svr_fit = time.time() - t0
    print("SVR complexity and bandwidth selected and model fitted in %.3f s"
          % svr_fit)

    t0 = time.time()
    kr.fit(X[:train_size], y[:train_size])
    kr_fit = time.time() - t0
    print("KRR complexity and bandwidth selected and model fitted in %.3f s"
          % kr_fit)

    sv_ratio = svr.best_estimator_.support_.shape[0] / train_size
    print("Support vector ratio: %.3f" % sv_ratio)

    t0 = time.time()
    y_svr = svr.predict(X_plot)
    svr_predict = time.time() - t0
    print("SVR prediction for %d inputs in %.3f s"
          % (X_plot.shape[0], svr_predict))

    t0 = time.time()
    y_kr = kr.predict(X_plot)
    kr_predict = time.time() - t0
    print("KRR prediction for %d inputs in %.3f s"
          % (X_plot.shape[0], kr_predict))


    # #############################################################################
    # Look at the results
    sv_ind = svr.best_estimator_.support_
    plt.scatter(X[sv_ind], y[sv_ind], c='r', s=50, label='SVR support vectors',
                zorder=2, edgecolors=(0, 0, 0))
    plt.scatter(X[:100], y[:100], c='k', label='data', zorder=1,
                edgecolors=(0, 0, 0))
    plt.plot(X_plot, y_svr, c='r',
             label='SVR (fit: %.3fs, predict: %.3fs)' % (svr_fit, svr_predict))
    plt.plot(X_plot, y_kr, c='g',
             label='KRR (fit: %.3fs, predict: %.3fs)' % (kr_fit, kr_predict))
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title('SVR versus Kernel Ridge')
    plt.legend()

    # Visualize training and prediction time
    plt.figure()

    # Generate sample data
    X = 5 * rng.rand(10000, 1)
    y = np.sin(X).ravel()
    y[::5] += 3 * (0.5 - rng.rand(X.shape[0] // 5))
    sizes = np.logspace(1, 4, 7).astype(np.int)
    for name, estimator in {"KRR": KernelRidge(kernel='rbf', alpha=0.1,
                                               gamma=10),
                            "SVR": SVR(kernel='rbf', C=1e1, gamma=10)}.items():
        train_time = []
        test_time = []
        for train_test_size in sizes:
            t0 = time.time()
            estimator.fit(X[:train_test_size], y[:train_test_size])
            train_time.append(time.time() - t0)

            t0 = time.time()
            estimator.predict(X_plot[:1000])
            test_time.append(time.time() - t0)

        plt.plot(sizes, train_time, 'o-', color="r" if name == "SVR" else "g",
                 label="%s (train)" % name)
        plt.plot(sizes, test_time, 'o--', color="r" if name == "SVR" else "g",
                 label="%s (test)" % name)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Train size")
    plt.ylabel("Time (seconds)")
    plt.title('Execution Time')
    plt.legend(loc="best")

    # Visualize learning curves
    plt.figure()

    svr = SVR(kernel='rbf', C=1e1, gamma=0.1)
    kr = KernelRidge(kernel='rbf', alpha=0.1, gamma=0.1)
    train_sizes, train_scores_svr, test_scores_svr = \
        learning_curve(svr, X[:100], y[:100], train_sizes=np.linspace(0.1, 1, 10),
                       scoring="neg_mean_squared_error", cv=10)
    train_sizes_abs, train_scores_kr, test_scores_kr = \
        learning_curve(kr, X[:100], y[:100], train_sizes=np.linspace(0.1, 1, 10),
                       scoring="neg_mean_squared_error", cv=10)

    plt.plot(train_sizes, -test_scores_svr.mean(1), 'o-', color="r",
             label="SVR")
    plt.plot(train_sizes, -test_scores_kr.mean(1), 'o-', color="g",
             label="KRR")
    plt.xlabel("Train size")
    plt.ylabel("Mean Squared Error")
    plt.title('Learning curves')
    plt.legend(loc="best")

    plt.show()

**Total running time of the script:** ( 0 minutes  19.992 seconds)


.. _sphx_glr_download_auto_examples_plot_kernel_ridge_regression.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_kernel_ridge_regression.py <plot_kernel_ridge_regression.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_kernel_ridge_regression.ipynb <plot_kernel_ridge_regression.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
