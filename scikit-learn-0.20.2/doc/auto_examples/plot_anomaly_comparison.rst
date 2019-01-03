.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_plot_anomaly_comparison.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_plot_anomaly_comparison.py:


============================================================================
在各种toy数据集上比较用于孤立点检测的异常检测算法
============================================================================

这个例子展示了二维数据集上不同异常检测算法的特点。数据集包含一个或两个模式(高密度区域)
来说明算法处理多模态数据(multimodal data)的能力。

对于每个数据集，15%的样本以随机均匀噪声的形式产生。这个比例是用于 OneClassSVM 的 nu 参数和
其他孤立点检测算法的污染参数的取值。除局部离群因子(Local Outlier Factor，LOF)外，
正常值和异常值之间的决策边界以黑色显示，
因为它在用于异常值检测时没有用于新数据的预测方法。

众所周知， :class:`sklearn.svm.OneClassSVM` 对outliers很敏感，因此对于outliers的检测效果不太好。
当训练集不受outliers污染时，这种估计器最适合于新奇点检测(novelty detection)。
也就是说，在 高维 或 没有任何关于内在数据的分布的假设 的情况下，
孤立点的检测(outlier detection)是非常具有挑战性的，
One-class SVM 在这些情况下可能会根据其超参数的值给出有用的结果。

:class:`sklearn.covariance.EllipticEnvelope` 假设数据是服从高斯分布的，并学习一个椭圆分布。
因此，当数据不是单峰时(unimodal)，它就会退化(degrade)。
但是，请注意，此estimator对outliers是robust的。

:class:`sklearn.ensemble.IsolationForest` 和 :class:`sklearn.neighbors.LocalOutlierFactor` 
似乎在多模式(multi-modal)数据集上表现良好。 :class:`sklearn.neighbors.LocalOutlierFactor` 
在第三个数据集中显示了比其他估计器更优越的优点，其中两种模式具有不同的密度。
这一优势是由 local aspect of LOF 来解释的，这意味着它只将一个样本的异常分数(score of abnormality)
与其邻居的分数进行比较。

最后，对于最后一个数据集，很难说一个样本比另一个样本更不正常(abnormal)，因为它们在超立方体(hypercube)中均匀分布。
除了 :class:`sklearn.svm.OneClassSVM` 有点过拟合外，所有估计器都给出了很好的解决方案。
在这种情况下，更仔细地观察样本的异常分数(scores of abnormality of the samples)是明智的，
因为一个好的估计器应该为所有样本分配相似的分数。

虽然这个例子给出了一些关于这些算法的直觉，但这种直觉可能不适用于非常高维的数据(very high dimensional data)。

最后，请注意，这些模型的参数是在这里精心挑选的，但实际上它们需要调整。
在没有标签数据的情况下，这个问题是完全没有监督的，因此模型的选择可能是一个挑战。




.. image:: /auto_examples/images/sphx_glr_plot_anomaly_comparison_001.png
    :class: sphx-glr-single-img





.. code-block:: python


    # Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
    #         Albert Thomas <albert.thomas@telecom-paristech.fr>
    # License: BSD 3 clause
    # 翻译者 ： Antares博士

    import time

    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt

    from sklearn import svm
    from sklearn.datasets import make_moons, make_blobs
    from sklearn.covariance import EllipticEnvelope
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor

    print(__doc__)

    matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

    # 样本设置

    n_samples = 300
    outliers_fraction = 0.15
    n_outliers = int(outliers_fraction * n_samples)
    n_inliers = n_samples - n_outliers

    # 定义要参与比较的 outlier/anomaly 检测算法

    anomaly_algorithms = [
        ("Robust covariance", EllipticEnvelope(contamination=outliers_fraction)),
        ("One-Class SVM", svm.OneClassSVM(nu=outliers_fraction, kernel="rbf",
                                          gamma=0.1)),
        ("Isolation Forest", IsolationForest(behaviour='new',
                                             contamination=outliers_fraction,
                                             random_state=42)),
        ("Local Outlier Factor", LocalOutlierFactor(
            n_neighbors=35, contamination=outliers_fraction))]

    # 定义数据集

    blobs_params = dict(random_state=0, n_samples=n_inliers, n_features=2)
    datasets = [
        make_blobs(centers=[[0, 0], [0, 0]], cluster_std=0.5,
                   **blobs_params)[0],
        make_blobs(centers=[[2, 2], [-2, -2]], cluster_std=[0.5, 0.5],
                   **blobs_params)[0],
        make_blobs(centers=[[2, 2], [-2, -2]], cluster_std=[1.5, .3],
                   **blobs_params)[0],
        4. * (make_moons(n_samples=n_samples, noise=.05, random_state=0)[0] -
              np.array([0.5, 0.25])),
        14. * (np.random.RandomState(42).rand(n_samples, 2) - 0.5)]

    # 在给定的设置下比较给定的分类器

    xx, yy = np.meshgrid(np.linspace(-7, 7, 150),
                         np.linspace(-7, 7, 150))

    plt.figure(figsize=(len(anomaly_algorithms) * 2 + 3, 12.5))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                        hspace=.01)

    plot_num = 1
    rng = np.random.RandomState(42)

    for i_dataset, X in enumerate(datasets):
        # 添加 outliers
        X = np.concatenate([X, rng.uniform(low=-6, high=6,
                           size=(n_outliers, 2))], axis=0)

        for name, algorithm in anomaly_algorithms:
            t0 = time.time()
            algorithm.fit(X)
            t1 = time.time()
            plt.subplot(len(datasets), len(anomaly_algorithms), plot_num)
            if i_dataset == 0:
                plt.title(name, size=18)

            # 拟合数据 并 标记 outliers
            if name == "Local Outlier Factor":
                y_pred = algorithm.fit_predict(X)
            else:
                y_pred = algorithm.fit(X).predict(X)

            # plot the levels lines and the points
            if name != "Local Outlier Factor":  # LOF 没有实现 predict
                Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')

            colors = np.array(['#377eb8', '#ff7f00'])
            plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[(y_pred + 1) // 2])

            plt.xlim(-7, 7)
            plt.ylim(-7, 7)
            plt.xticks(())
            plt.yticks(())
            plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                     transform=plt.gca().transAxes, size=15,
                     horizontalalignment='right')
            plot_num += 1

    plt.show()

**Total running time of the script:** ( 0 minutes  5.517 seconds)


.. _sphx_glr_download_auto_examples_plot_anomaly_comparison.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_anomaly_comparison.py <plot_anomaly_comparison.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_anomaly_comparison.ipynb <plot_anomaly_comparison.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
