.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_manifold_plot_t_sne_perplexity.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_manifold_plot_t_sne_perplexity.py:


=============================================================================
t-SNE: 不同困惑值对形状的影响
=============================================================================

两个同心圆和S曲线数据集对不同困惑值t-SNE的示例。

我们观察到，随着迷惑性值(preplexity value)的增加，形状越来越清晰。

聚簇(clusters)的大小、距离和形状可能因初始化、困惑值而异，并不总是传达意义。

如下所示，对于较高的困惑值，t-SNE发现了两个同心圆的有意义的拓扑结构，但圆圈的大小和距离与原来的略有不同。
与同心圆数据集相反，即使在较大的困惑值下，t-SNE获得的形状在视觉上也与S曲线数据集上的S曲线拓扑不同。

关于更多详细细节, "How to Use t-SNE Effectively" http://distill.pub/2016/misread-tsne/ 
提供了各种参数的影响的讨论，包括一些交互式绘图来探索这些影响。




.. image:: /auto_examples/manifold/images/sphx_glr_plot_t_sne_perplexity_001.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    circles, perplexity=5 in 1.3 sec
    circles, perplexity=30 in 1.5 sec
    circles, perplexity=50 in 1.5 sec
    circles, perplexity=100 in 1.8 sec
    S-curve, perplexity=5 in 1.4 sec
    S-curve, perplexity=30 in 1.7 sec
    S-curve, perplexity=50 in 2 sec
    S-curve, perplexity=100 in 2.9 sec
    uniform grid, perplexity=5 in 1.3 sec
    uniform grid, perplexity=30 in 1.7 sec
    uniform grid, perplexity=50 in 1.6 sec
    uniform grid, perplexity=100 in 2.4 sec




|


.. code-block:: python


    # Author: Narine Kokhlikyan <narine@slice.com>
    # License: BSD

    print(__doc__)

    import numpy as np
    import matplotlib.pyplot as plt

    from matplotlib.ticker import NullFormatter
    from sklearn import manifold, datasets
    from time import time

    n_samples = 300
    n_components = 2
    (fig, subplots) = plt.subplots(3, 5, figsize=(15, 8))
    perplexities = [5, 30, 50, 100]

    X, y = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)

    red = y == 0
    green = y == 1

    ax = subplots[0][0]
    ax.scatter(X[red, 0], X[red, 1], c="r")
    ax.scatter(X[green, 0], X[green, 1], c="g")
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')

    for i, perplexity in enumerate(perplexities):
        ax = subplots[0][i + 1]

        t0 = time()
        tsne = manifold.TSNE(n_components=n_components, init='random',
                             random_state=0, perplexity=perplexity)
        Y = tsne.fit_transform(X)
        t1 = time()
        print("circles, perplexity=%d in %.2g sec" % (perplexity, t1 - t0))
        ax.set_title("Perplexity=%d" % perplexity)
        ax.scatter(Y[red, 0], Y[red, 1], c="r")
        ax.scatter(Y[green, 0], Y[green, 1], c="g")
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.axis('tight')

    # Another example using s-curve
    X, color = datasets.samples_generator.make_s_curve(n_samples, random_state=0)

    ax = subplots[1][0]
    ax.scatter(X[:, 0], X[:, 2], c=color)
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())

    for i, perplexity in enumerate(perplexities):
        ax = subplots[1][i + 1]

        t0 = time()
        tsne = manifold.TSNE(n_components=n_components, init='random',
                             random_state=0, perplexity=perplexity)
        Y = tsne.fit_transform(X)
        t1 = time()
        print("S-curve, perplexity=%d in %.2g sec" % (perplexity, t1 - t0))

        ax.set_title("Perplexity=%d" % perplexity)
        ax.scatter(Y[:, 0], Y[:, 1], c=color)
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.axis('tight')


    # Another example using a 2D uniform grid
    x = np.linspace(0, 1, int(np.sqrt(n_samples)))
    xx, yy = np.meshgrid(x, x)
    X = np.hstack([
        xx.ravel().reshape(-1, 1),
        yy.ravel().reshape(-1, 1),
    ])
    color = xx.ravel()
    ax = subplots[2][0]
    ax.scatter(X[:, 0], X[:, 1], c=color)
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())

    for i, perplexity in enumerate(perplexities):
        ax = subplots[2][i + 1]

        t0 = time()
        tsne = manifold.TSNE(n_components=n_components, init='random',
                             random_state=0, perplexity=perplexity)
        Y = tsne.fit_transform(X)
        t1 = time()
        print("uniform grid, perplexity=%d in %.2g sec" % (perplexity, t1 - t0))

        ax.set_title("Perplexity=%d" % perplexity)
        ax.scatter(Y[:, 0], Y[:, 1], c=color)
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.axis('tight')


    plt.show()

**Total running time of the script:** ( 0 minutes  21.591 seconds)


.. _sphx_glr_download_auto_examples_manifold_plot_t_sne_perplexity.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_t_sne_perplexity.py <plot_t_sne_perplexity.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_t_sne_perplexity.ipynb <plot_t_sne_perplexity.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
