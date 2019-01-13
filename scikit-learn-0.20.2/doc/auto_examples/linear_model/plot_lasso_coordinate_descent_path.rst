.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_linear_model_plot_lasso_coordinate_descent_path.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_linear_model_plot_lasso_coordinate_descent_path.py:


=====================
Lasso 和 Elastic-Net
=====================

Lasso 和 elastic-net (L1与L2惩罚) 是使用坐标下降法来实现的。

模型的系数可以被强制变为正的(通过设置参数 ``positive=True`` )。

(译者注：要注意 在三个图中alpha被取了负对数，所以越靠近图的左边alpha的负对数越小而alpha则越大，
alpha越大则正则化就越厉害，系数就都缩减为0了。所以大家看到在三幅图里面都是
从左到右系数越来越发散，就是因为随着alpha的减小正则化项被逐渐削弱导致的。
但是在Lasso中随着alpha的减小模型系数发散的比ElesticNet要快。)




.. rst-class:: sphx-glr-horizontal


    *

      .. image:: /auto_examples/linear_model/images/sphx_glr_plot_lasso_coordinate_descent_path_001.png
            :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/linear_model/images/sphx_glr_plot_lasso_coordinate_descent_path_002.png
            :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/linear_model/images/sphx_glr_plot_lasso_coordinate_descent_path_003.png
            :class: sphx-glr-multi-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Computing regularization path using the lasso...
    Computing regularization path using the positive lasso...
    Computing regularization path using the elastic net...
    Computing regularization path using the positive elastic net...




|


.. code-block:: python

    print(__doc__)

    # Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
    # License: BSD 3 clause
    # 翻译者： studyai.com 的 Antares 博士

    from itertools import cycle
    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.linear_model import lasso_path, enet_path
    from sklearn import datasets

    diabetes = datasets.load_diabetes()
    X = diabetes.data
    y = diabetes.target

    X /= X.std(axis=0)  # 标准化数据 (更容易设置 l1_ratio 参数)

    # 计算(正则化)路径

    eps = 5e-3  # 此值越小，路径越长

    print("Computing regularization path using the lasso...")
    alphas_lasso, coefs_lasso, _ = lasso_path(X, y, eps, fit_intercept=False)

    print("Computing regularization path using the positive lasso...")
    alphas_positive_lasso, coefs_positive_lasso, _ = lasso_path(
        X, y, eps, positive=True, fit_intercept=False)
    print("Computing regularization path using the elastic net...")
    alphas_enet, coefs_enet, _ = enet_path(
        X, y, eps=eps, l1_ratio=0.8, fit_intercept=False)

    print("Computing regularization path using the positive elastic net...")
    alphas_positive_enet, coefs_positive_enet, _ = enet_path(
        X, y, eps=eps, l1_ratio=0.8, positive=True, fit_intercept=False)

    # 展示结果

    plt.figure(1)
    colors = cycle(['b', 'r', 'g', 'c', 'k'])
    neg_log_alphas_lasso = -np.log10(alphas_lasso)
    neg_log_alphas_enet = -np.log10(alphas_enet)
    for coef_l, coef_e, c in zip(coefs_lasso, coefs_enet, colors):
        l1 = plt.plot(neg_log_alphas_lasso, coef_l, c=c)
        l2 = plt.plot(neg_log_alphas_enet, coef_e, linestyle='--', c=c)

    plt.xlabel('-Log(alpha)')
    plt.ylabel('coefficients')
    plt.title('Lasso and Elastic-Net Paths')
    plt.legend((l1[-1], l2[-1]), ('Lasso', 'Elastic-Net'), loc='lower left')
    plt.axis('tight')


    plt.figure(2)
    neg_log_alphas_positive_lasso = -np.log10(alphas_positive_lasso)
    for coef_l, coef_pl, c in zip(coefs_lasso, coefs_positive_lasso, colors):
        l1 = plt.plot(neg_log_alphas_lasso, coef_l, c=c)
        l2 = plt.plot(neg_log_alphas_positive_lasso, coef_pl, linestyle='--', c=c)

    plt.xlabel('-Log(alpha)')
    plt.ylabel('coefficients')
    plt.title('Lasso and positive Lasso')
    plt.legend((l1[-1], l2[-1]), ('Lasso', 'positive Lasso'), loc='lower left')
    plt.axis('tight')


    plt.figure(3)
    neg_log_alphas_positive_enet = -np.log10(alphas_positive_enet)
    for (coef_e, coef_pe, c) in zip(coefs_enet, coefs_positive_enet, colors):
        l1 = plt.plot(neg_log_alphas_enet, coef_e, c=c)
        l2 = plt.plot(neg_log_alphas_positive_enet, coef_pe, linestyle='--', c=c)

    plt.xlabel('-Log(alpha)')
    plt.ylabel('coefficients')
    plt.title('Elastic-Net and positive Elastic-Net')
    plt.legend((l1[-1], l2[-1]), ('Elastic-Net', 'positive Elastic-Net'),
               loc='lower left')
    plt.axis('tight')
    plt.show()

**Total running time of the script:** ( 0 minutes  0.192 seconds)


.. _sphx_glr_download_auto_examples_linear_model_plot_lasso_coordinate_descent_path.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_lasso_coordinate_descent_path.py <plot_lasso_coordinate_descent_path.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_lasso_coordinate_descent_path.ipynb <plot_lasso_coordinate_descent_path.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
