.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_svm_plot_svm_nonlinear.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_svm_plot_svm_nonlinear.py:


==============
非线性 SVM
==============

使用带有RBF核的非线性SVC执行二类分类问题。要预测的目标是输入的XOR。

彩色图展示了SVC学习到的决策函数。




.. image:: /auto_examples/svm/images/sphx_glr_plot_svm_nonlinear_001.png
    :class: sphx-glr-single-img





.. code-block:: python

    print(__doc__)

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import svm

    xx, yy = np.meshgrid(np.linspace(-3, 3, 500),
                         np.linspace(-3, 3, 500))
    np.random.seed(0)
    X = np.random.randn(300, 2)
    Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)

    # fit the model
    clf = svm.NuSVC()
    clf.fit(X, Y)

    # plot the decision function for each datapoint on the grid
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
               origin='lower', cmap=plt.cm.PuOr_r)
    contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2,
                           linetypes='--')
    plt.scatter(X[:, 0], X[:, 1], s=30, c=Y, cmap=plt.cm.Paired,
                edgecolors='k')
    plt.xticks(())
    plt.yticks(())
    plt.axis([-3, 3, -3, 3])
    plt.show()

**Total running time of the script:** ( 0 minutes  0.780 seconds)


.. _sphx_glr_download_auto_examples_svm_plot_svm_nonlinear.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_svm_nonlinear.py <plot_svm_nonlinear.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_svm_nonlinear.ipynb <plot_svm_nonlinear.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
