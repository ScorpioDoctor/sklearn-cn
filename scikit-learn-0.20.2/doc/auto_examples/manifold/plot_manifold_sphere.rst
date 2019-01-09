.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_manifold_plot_manifold_sphere.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_manifold_plot_manifold_sphere.py:


=============================================
在切断球体上的流行学习方法
=============================================

不同的 :ref:`manifold` 技术在球形数据集上的应用。
这里，你可以看到为了获得流行学习方法的一些直观印象使用了维数约减。
关于数据集，两极以及沿着球体一侧的一片薄片被从球体中切下来。
这使得流形学习技术能够“打开”球体，同时将其投射到两个维度。

还有一个类似的例子, 在那里这些流行方法被运用到 S-curve 数据集，请看 
:ref:`sphx_glr_auto_examples_manifold_plot_compare_methods.py`

请注意 :ref:`MDS <multidimensional_scaling>` 的目的是发现数据的一个低维表示(这里是2d表示)，
在这个低维表示下数据点之间的距离
很好地尊重数据点在原始空间中的距离，这不同于其他流形学习算法，它不会在低维空间中寻找数据的
各向同性表示(isotropic representation)。

在这里，流形问题与表示地球平面图的问题相当匹配，就像 `地图投影<https://en.wikipedia.org/wiki/Map_projection>`_ 一样。




.. image:: /auto_examples/manifold/images/sphx_glr_plot_manifold_sphere_001.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    standard: 0.14 sec
    ltsa: 0.16 sec
    hessian: 0.26 sec
    modified: 0.22 sec
    ISO: 0.22 sec
    MDS: 1.5 sec
    Spectral Embedding: 0.09 sec
    t-SNE: 4.2 sec




|


.. code-block:: python


    # Author: Jaques Grobler <jaques.grobler@inria.fr>
    # License: BSD 3 clause

    print(__doc__)

    from time import time

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.ticker import NullFormatter

    from sklearn import manifold
    from sklearn.utils import check_random_state

    # Next line to silence pyflakes.
    Axes3D

    # Variables for manifold learning.
    n_neighbors = 10
    n_samples = 1000

    # Create our sphere.
    random_state = check_random_state(0)
    p = random_state.rand(n_samples) * (2 * np.pi - 0.55)
    t = random_state.rand(n_samples) * np.pi

    # Sever the poles from the sphere.
    indices = ((t < (np.pi - (np.pi / 8))) & (t > ((np.pi / 8))))
    colors = p[indices]
    x, y, z = np.sin(t[indices]) * np.cos(p[indices]), \
        np.sin(t[indices]) * np.sin(p[indices]), \
        np.cos(t[indices])

    # Plot our dataset.
    fig = plt.figure(figsize=(15, 8))
    plt.suptitle("Manifold Learning with %i points, %i neighbors"
                 % (1000, n_neighbors), fontsize=14)

    ax = fig.add_subplot(251, projection='3d')
    ax.scatter(x, y, z, c=p[indices], cmap=plt.cm.rainbow)
    ax.view_init(40, -10)

    sphere_data = np.array([x, y, z]).T

    # Perform Locally Linear Embedding Manifold learning
    methods = ['standard', 'ltsa', 'hessian', 'modified']
    labels = ['LLE', 'LTSA', 'Hessian LLE', 'Modified LLE']

    for i, method in enumerate(methods):
        t0 = time()
        trans_data = manifold\
            .LocallyLinearEmbedding(n_neighbors, 2,
                                    method=method).fit_transform(sphere_data).T
        t1 = time()
        print("%s: %.2g sec" % (methods[i], t1 - t0))

        ax = fig.add_subplot(252 + i)
        plt.scatter(trans_data[0], trans_data[1], c=colors, cmap=plt.cm.rainbow)
        plt.title("%s (%.2g sec)" % (labels[i], t1 - t0))
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.axis('tight')

    # Perform Isomap Manifold learning.
    t0 = time()
    trans_data = manifold.Isomap(n_neighbors, n_components=2)\
        .fit_transform(sphere_data).T
    t1 = time()
    print("%s: %.2g sec" % ('ISO', t1 - t0))

    ax = fig.add_subplot(257)
    plt.scatter(trans_data[0], trans_data[1], c=colors, cmap=plt.cm.rainbow)
    plt.title("%s (%.2g sec)" % ('Isomap', t1 - t0))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')

    # Perform Multi-dimensional scaling.
    t0 = time()
    mds = manifold.MDS(2, max_iter=100, n_init=1)
    trans_data = mds.fit_transform(sphere_data).T
    t1 = time()
    print("MDS: %.2g sec" % (t1 - t0))

    ax = fig.add_subplot(258)
    plt.scatter(trans_data[0], trans_data[1], c=colors, cmap=plt.cm.rainbow)
    plt.title("MDS (%.2g sec)" % (t1 - t0))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')

    # Perform Spectral Embedding.
    t0 = time()
    se = manifold.SpectralEmbedding(n_components=2,
                                    n_neighbors=n_neighbors)
    trans_data = se.fit_transform(sphere_data).T
    t1 = time()
    print("Spectral Embedding: %.2g sec" % (t1 - t0))

    ax = fig.add_subplot(259)
    plt.scatter(trans_data[0], trans_data[1], c=colors, cmap=plt.cm.rainbow)
    plt.title("Spectral Embedding (%.2g sec)" % (t1 - t0))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')

    # Perform t-distributed stochastic neighbor embedding.
    t0 = time()
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    trans_data = tsne.fit_transform(sphere_data).T
    t1 = time()
    print("t-SNE: %.2g sec" % (t1 - t0))

    ax = fig.add_subplot(2, 5, 10)
    plt.scatter(trans_data[0], trans_data[1], c=colors, cmap=plt.cm.rainbow)
    plt.title("t-SNE (%.2g sec)" % (t1 - t0))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')

    plt.show()

**Total running time of the script:** ( 0 minutes  7.020 seconds)


.. _sphx_glr_download_auto_examples_manifold_plot_manifold_sphere.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_manifold_sphere.py <plot_manifold_sphere.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_manifold_sphere.ipynb <plot_manifold_sphere.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
