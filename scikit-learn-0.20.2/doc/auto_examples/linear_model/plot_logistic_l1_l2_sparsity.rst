.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_linear_model_plot_logistic_l1_l2_sparsity.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_linear_model_plot_logistic_l1_l2_sparsity.py:


==============================================
L1 惩罚 与 Logistic回归中的稀疏性
==============================================

对不同的 C 值采用L1和L2惩罚时，解的稀疏性(零系数百分比)的比较。
我们可以看到，较大的 C 值给了模型更多的自由度。
相反，较小的 C 值对模型的约束更大。 L1惩罚导致更稀疏的解。

我们将8x8的数字图像分为两类：0-4对5-9。可视化显示了模型的系数在不断变化的C值下的图像。




.. image:: /auto_examples/linear_model/images/sphx_glr_plot_logistic_l1_l2_sparsity_001.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    C=1.00
    Sparsity with L1 penalty: 6.25%
    score with L1 penalty: 0.9098
    Sparsity with L2 penalty: 4.69%
    score with L2 penalty: 0.9037
    C=0.10
    Sparsity with L1 penalty: 25.00%
    score with L1 penalty: 0.9004
    Sparsity with L2 penalty: 4.69%
    score with L2 penalty: 0.9009
    C=0.01
    Sparsity with L1 penalty: 84.38%
    score with L1 penalty: 0.8625
    Sparsity with L2 penalty: 4.69%
    score with L2 penalty: 0.8893




|


.. code-block:: python


    print(__doc__)

    # Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
    #          Mathieu Blondel <mathieu@mblondel.org>
    #          Andreas Mueller <amueller@ais.uni-bonn.de>
    # 翻译者：studyai.com 的 Antares 博士
    # License: BSD 3 clause

    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.linear_model import LogisticRegression
    from sklearn import datasets
    from sklearn.preprocessing import StandardScaler

    digits = datasets.load_digits()

    X, y = digits.data, digits.target
    X = StandardScaler().fit_transform(X)

    # 把>4的数字看做一类，<=4的数字看做另一类，
    # 就变成了典型的二分类问题
    y = (y > 4).astype(np.int)


    # 设置正则化参数
    for i, C in enumerate((1, 0.1, 0.01)):
        # turn down tolerance for short training time
        clf_l1_LR = LogisticRegression(C=C, penalty='l1', tol=0.01, solver='saga')
        clf_l2_LR = LogisticRegression(C=C, penalty='l2', tol=0.01, solver='saga')
        clf_l1_LR.fit(X, y)
        clf_l2_LR.fit(X, y)

        coef_l1_LR = clf_l1_LR.coef_.ravel()
        coef_l2_LR = clf_l2_LR.coef_.ravel()

        # coef_l1_LR contains zeros due to the
        # L1 sparsity inducing norm

        sparsity_l1_LR = np.mean(coef_l1_LR == 0) * 100
        sparsity_l2_LR = np.mean(coef_l2_LR == 0) * 100

        print("C=%.2f" % C)
        print("Sparsity with L1 penalty: %.2f%%" % sparsity_l1_LR)
        print("score with L1 penalty: %.4f" % clf_l1_LR.score(X, y))
        print("Sparsity with L2 penalty: %.2f%%" % sparsity_l2_LR)
        print("score with L2 penalty: %.4f" % clf_l2_LR.score(X, y))

        l1_plot = plt.subplot(3, 2, 2 * i + 1)
        l2_plot = plt.subplot(3, 2, 2 * (i + 1))
        if i == 0:
            l1_plot.set_title("L1 penalty")
            l2_plot.set_title("L2 penalty")

        l1_plot.imshow(np.abs(coef_l1_LR.reshape(8, 8)), interpolation='nearest',
                       cmap='binary', vmax=1, vmin=0)
        l2_plot.imshow(np.abs(coef_l2_LR.reshape(8, 8)), interpolation='nearest',
                       cmap='binary', vmax=1, vmin=0)
        plt.text(-8, 3, "C = %.2f" % C)

        l1_plot.set_xticks(())
        l1_plot.set_yticks(())
        l2_plot.set_xticks(())
        l2_plot.set_yticks(())

    plt.show()

**Total running time of the script:** ( 0 minutes  0.566 seconds)


.. _sphx_glr_download_auto_examples_linear_model_plot_logistic_l1_l2_sparsity.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_logistic_l1_l2_sparsity.py <plot_logistic_l1_l2_sparsity.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_logistic_l1_l2_sparsity.ipynb <plot_logistic_l1_l2_sparsity.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
