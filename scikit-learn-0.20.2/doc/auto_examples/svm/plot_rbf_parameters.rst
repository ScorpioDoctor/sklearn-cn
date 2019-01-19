.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_svm_plot_rbf_parameters.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_svm_plot_rbf_parameters.py:


==================
RBF SVM 参数选择
==================

这个例子展示了径向基函数(RBF)核支持向量机的参数 ``gamma`` 和 ``C`` 的影响。

直观地，``gamma`` 参数定义了单个训练样本的影响达到多远，低值意味着“远”，高值意味着“近”。
``gamma``  参数可以看作是模型选取作为支持向量的样本的影响半径的反比。

``C`` 参数在 把尽可能多的训练样本正确分类 与 使决策函数的裕度(margin)最大化 这两个矛盾之间做折中。
对于较大的 ``C`` 值， 如果决策函数能更好地对所有训练点进行正确的分类，则一个较小的裕度将会被接受。
一个较低的 ``C`` 值将鼓励更大的裕度(margin)，因此模型将以牺牲训练的准确性接受一个更简单的决策函数。
换句话说，``C`` 在支持向量机中充当正则化参数： ``C`` 值 越小，正则化程度越强，产生的决策面更简单。

第一个图是对一个简化的分类问题的各种参数值的决策函数的可视化，该分类问题只涉及两个输入特征和两个可能的目标类(二类分类)。
请注意，对于具有更多特征或目标类的问题，这种绘图是不可能的。

第二幅图是分类器的交叉验证准确率作为 ``C`` 和 ``gamma`` 的函数随这两个参数变化的热力图。
在本例中，我们将探索一个相对较大的网格，以进行说明。
在实践中，从 :math:`10^{-3}` 到 :math:`10^3` 的对数网格通常是足够的。
如果最佳参数位于网格的边界上，则可以在随后的搜索中向该方向扩展。

请注意，热力图有一个特别的色度条，其中每个点的颜色值接近不同参数下模型的得分值，
以便在眨眼之间很容易将它们区分开来，哪些表现好，哪些表现不好。

模型的行为对 ``gamma`` 参数非常敏感。如果 ``gamma`` 太大，支持向量的影响范围半径仅包括支持向量本身，
再加上 ``C`` 的正则化也无法防止过拟合。

当 ``gamma`` 非常小时，模型太受约束，无法捕捉数据的复杂性或“形状”。任何选定的支持向量的影响区域将包括整个训练集。
得到的模型将类似于具有一组超平面的线性模型，这些超平面将任意一对两类的高密度中心分离开来。

对于 ``gamma`` 的中等大小的值，我们可以在第二幅图上看到，在 ``C`` 和 ``gamma`` 的对角线上可以找到好的模型。
平滑的模型(对应于较低的 ``gamma`` 值)可以通过增加正确分类每个点的重要性(较大的 ``C`` 值)而变得更加复杂，
从而使性能良好的模型的对角线变得更加复杂。

最后，我们还可以观察到，当 ``C`` 变得非常大时，对于 ``gamma`` 的一些中等大小的值，我们得到了具有相同性能表现的模型：
没有必要通过执行较大的余量(margin)来进行正则化。
RBF核的半径本身就是一个很好的结构化正则化器。然而，在实践中，用较低的 ``C`` 值简化决策函数可能仍然很有趣，
以便更好地支持内存更少、预测速度更快的模型。

我们还应该注意到，分类得分的微小差异是由于交叉验证过程的随机分裂造成的。这些杂散变化可以通过增加CV迭代次数 ``n_splits`` 来平滑，
而牺牲计算时间。增加 ``C_range`` 和 ``gamma_range`` 步长的数值数将提高超参数热力图的分辨率。

翻译者：http://www.studyai.com/antares




.. rst-class:: sphx-glr-horizontal


    *

      .. image:: /auto_examples/svm/images/sphx_glr_plot_rbf_parameters_001.png
            :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/svm/images/sphx_glr_plot_rbf_parameters_002.png
            :class: sphx-glr-multi-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    The best parameters are {'C': 1.0, 'gamma': 0.1} with a score of 0.97




|


.. code-block:: python

    print(__doc__)

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import load_iris
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.model_selection import GridSearchCV


    # Utility function to move the midpoint of a colormap to be around
    # the values of interest.

    class MidpointNormalize(Normalize):

        def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
            self.midpoint = midpoint
            Normalize.__init__(self, vmin, vmax, clip)

        def __call__(self, value, clip=None):
            x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
            return np.ma.masked_array(np.interp(value, x, y))

    # #############################################################################
    # Load and prepare data set
    #
    # dataset for grid search

    iris = load_iris()
    X = iris.data
    y = iris.target

    # Dataset for decision function visualization: we only keep the first two
    # features in X and sub-sample the dataset to keep only 2 classes and
    # make it a binary classification problem.

    X_2d = X[:, :2]
    X_2d = X_2d[y > 0]
    y_2d = y[y > 0]
    y_2d -= 1

    # It is usually a good idea to scale the data for SVM training.
    # We are cheating a bit in this example in scaling all of the data,
    # instead of fitting the transformation on the training set and
    # just applying it on the test set.

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_2d = scaler.fit_transform(X_2d)

    # #############################################################################
    # Train classifiers
    #
    # For an initial search, a logarithmic grid with basis
    # 10 is often helpful. Using a basis of 2, a finer
    # tuning can be achieved but at a much higher cost.

    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
    grid.fit(X, y)

    print("The best parameters are %s with a score of %0.2f"
          % (grid.best_params_, grid.best_score_))

    # Now we need to fit a classifier for all parameters in the 2d version
    # (we use a smaller set of parameters here because it takes a while to train)

    C_2d_range = [1e-2, 1, 1e2]
    gamma_2d_range = [1e-1, 1, 1e1]
    classifiers = []
    for C in C_2d_range:
        for gamma in gamma_2d_range:
            clf = SVC(C=C, gamma=gamma)
            clf.fit(X_2d, y_2d)
            classifiers.append((C, gamma, clf))

    # #############################################################################
    # Visualization
    #
    # draw visualization of parameter effects

    plt.figure(figsize=(8, 6))
    xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
    for (k, (C, gamma, clf)) in enumerate(classifiers):
        # evaluate decision function in a grid
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # visualize decision function for these parameters
        plt.subplot(len(C_2d_range), len(gamma_2d_range), k + 1)
        plt.title("gamma=10^%d, C=10^%d" % (np.log10(gamma), np.log10(C)),
                  size='medium')

        # visualize parameter's effect on decision function
        plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu)
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap=plt.cm.RdBu_r,
                    edgecolors='k')
        plt.xticks(())
        plt.yticks(())
        plt.axis('tight')

    scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),
                                                         len(gamma_range))

    # Draw heatmap of the validation accuracy as a function of gamma and C
    #
    # The score are encoded as colors with the hot colormap which varies from dark
    # red to bright yellow. As the most interesting scores are all located in the
    # 0.92 to 0.97 range we use a custom normalizer to set the mid-point to 0.92 so
    # as to make it easier to visualize the small variations of score values in the
    # interesting range while not brutally collapsing all the low score values to
    # the same color.

    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
               norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title('Validation accuracy')
    plt.show()

**Total running time of the script:** ( 0 minutes  4.514 seconds)


.. _sphx_glr_download_auto_examples_svm_plot_rbf_parameters.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_rbf_parameters.py <plot_rbf_parameters.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_rbf_parameters.ipynb <plot_rbf_parameters.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
