.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_compose_plot_compare_reduction.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_compose_plot_compare_reduction.py:


=================================================================
Selecting dimensionality reduction with Pipeline and GridSearchCV
=================================================================

This example constructs a pipeline that does dimensionality
reduction followed by prediction with a support vector
classifier. It demonstrates the use of ``GridSearchCV`` and
``Pipeline`` to optimize over different classes of estimators in a
single CV run -- unsupervised ``PCA`` and ``NMF`` dimensionality
reductions are compared to univariate feature selection during
the grid search.

Additionally, ``Pipeline`` can be instantiated with the ``memory``
argument to memoize the transformers within the pipeline, avoiding to fit
again the same transformers over and over.

Note that the use of ``memory`` to enable caching becomes interesting when the
fitting of a transformer is costly.


Illustration of ``Pipeline`` and ``GridSearchCV``
##############################################################################
 This section illustrates the use of a ``Pipeline`` with
 ``GridSearchCV``



.. code-block:: python


    # Authors: Robert McGibbon, Joel Nothman, Guillaume Lemaitre

    from __future__ import print_function, division

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_digits
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC
    from sklearn.decomposition import PCA, NMF
    from sklearn.feature_selection import SelectKBest, chi2

    print(__doc__)

    pipe = Pipeline([
        # the reduce_dim stage is populated by the param_grid
        ('reduce_dim', None),
        ('classify', LinearSVC())
    ])

    N_FEATURES_OPTIONS = [2, 4, 8]
    C_OPTIONS = [1, 10, 100, 1000]
    param_grid = [
        {
            'reduce_dim': [PCA(iterated_power=7), NMF()],
            'reduce_dim__n_components': N_FEATURES_OPTIONS,
            'classify__C': C_OPTIONS
        },
        {
            'reduce_dim': [SelectKBest(chi2)],
            'reduce_dim__k': N_FEATURES_OPTIONS,
            'classify__C': C_OPTIONS
        },
    ]
    reducer_labels = ['PCA', 'NMF', 'KBest(chi2)']

    grid = GridSearchCV(pipe, cv=5, n_jobs=1, param_grid=param_grid)
    digits = load_digits()
    grid.fit(digits.data, digits.target)

    mean_scores = np.array(grid.cv_results_['mean_test_score'])
    # scores are in the order of param_grid iteration, which is alphabetical
    mean_scores = mean_scores.reshape(len(C_OPTIONS), -1, len(N_FEATURES_OPTIONS))
    # select score for best C
    mean_scores = mean_scores.max(axis=0)
    bar_offsets = (np.arange(len(N_FEATURES_OPTIONS)) *
                   (len(reducer_labels) + 1) + .5)

    plt.figure()
    COLORS = 'bgrcmyk'
    for i, (label, reducer_scores) in enumerate(zip(reducer_labels, mean_scores)):
        plt.bar(bar_offsets + i, reducer_scores, label=label, color=COLORS[i])

    plt.title("Comparing feature reduction techniques")
    plt.xlabel('Reduced number of features')
    plt.xticks(bar_offsets + len(reducer_labels) / 2, N_FEATURES_OPTIONS)
    plt.ylabel('Digit classification accuracy')
    plt.ylim((0, 1))
    plt.legend(loc='upper left')

    plt.show()




.. image:: /auto_examples/compose/images/sphx_glr_plot_compare_reduction_001.png
    :class: sphx-glr-single-img




Caching transformers within a ``Pipeline``
##############################################################################
 It is sometimes worthwhile storing the state of a specific transformer
 since it could be used again. Using a pipeline in ``GridSearchCV`` triggers
 such situations. Therefore, we use the argument ``memory`` to enable caching.

 .. warning::
     Note that this example is, however, only an illustration since for this
     specific case fitting PCA is not necessarily slower than loading the
     cache. Hence, use the ``memory`` constructor parameter when the fitting
     of a transformer is costly.



.. code-block:: python


    from tempfile import mkdtemp
    from shutil import rmtree
    from joblib import Memory

    # Create a temporary folder to store the transformers of the pipeline
    cachedir = mkdtemp()
    memory = Memory(cachedir=cachedir, verbose=10)
    cached_pipe = Pipeline([('reduce_dim', PCA()),
                            ('classify', LinearSVC())],
                           memory=memory)

    # This time, a cached pipeline will be used within the grid search
    grid = GridSearchCV(cached_pipe, cv=5, n_jobs=1, param_grid=param_grid)
    digits = load_digits()
    grid.fit(digits.data, digits.target)

    # Delete the temporary cache before exiting
    rmtree(cachedir)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    ________________________________________________________________________________
    [Memory] Calling sklearn.pipeline._fit_transform_one...
    _fit_transform_one(PCA(copy=True, iterated_power=7, n_components=2, random_state=None,
      svd_solver='auto', tol=0.0, whiten=False), 
    array([[0., ..., 0.],
           ...,
           [0., ..., 0.]]), array([0, ..., 8]), None)
    ________________________________________________fit_transform_one - 0.0s, 0.0min
    ________________________________________________________________________________
    [Memory] Calling sklearn.pipeline._fit_transform_one...
    _fit_transform_one(PCA(copy=True, iterated_power=7, n_components=2, random_state=None,
      svd_solver='auto', tol=0.0, whiten=False), 
    array([[0., ..., 0.],
           ...,
           [0., ..., 0.]]), array([0, ..., 8]), None)
    ________________________________________________fit_transform_one - 0.0s, 0.0min
    ________________________________________________________________________________
    [Memory] Calling sklearn.pipeline._fit_transform_one...
    _fit_transform_one(PCA(copy=True, iterated_power=7, n_components=2, random_state=None,
      svd_solver='auto', tol=0.0, whiten=False), 
    array([[0., ..., 0.],
           ...,
           [0., ..., 0.]]), array([0, ..., 8]), None)
    ________________________________________________fit_transform_one - 0.0s, 0.0min
    ________________________________________________________________________________
    [Memory] Calling sklearn.pipeline._fit_transform_one...
    _fit_transform_one(PCA(copy=True, iterated_power=7, n_components=2, random_state=None,
      svd_solver='auto', tol=0.0, whiten=False), 
    array([[0., ..., 0.],
           ...,
           [0., ..., 0.]]), array([0, ..., 8]), None)
    ________________________________________________fit_transform_one - 0.0s, 0.0min
    ________________________________________________________________________________
    [Memory] Calling sklearn.pipeline._fit_transform_one...
    _fit_transform_one(PCA(copy=True, iterated_power=7, n_components=2, random_state=None,
      svd_solver='auto', tol=0.0, whiten=False), 
    array([[0., ..., 0.],
           ...,
           [0., ..., 0.]]), array([0, ..., 9]), None)
    ________________________________________________fit_transform_one - 0.0s, 0.0min
    ________________________________________________________________________________
    [Memory] Calling sklearn.pipeline._fit_transform_one...
    _fit_transform_one(PCA(copy=True, iterated_power=7, n_components=4, random_state=None,
      svd_solver='auto', tol=0.0, whiten=False), 
    array([[0., ..., 0.],
           ...,
           [0., ..., 0.]]), array([0, ..., 8]), None)
    ________________________________________________fit_transform_one - 0.0s, 0.0min
    ________________________________________________________________________________
    [Memory] Calling sklearn.pipeline._fit_transform_one...
    _fit_transform_one(PCA(copy=True, iterated_power=7, n_components=4, random_state=None,
      svd_solver='auto', tol=0.0, whiten=False), 
    array([[0., ..., 0.],
           ...,
           [0., ..., 0.]]), array([0, ..., 8]), None)
    ________________________________________________fit_transform_one - 0.0s, 0.0min
    ________________________________________________________________________________
    [Memory] Calling sklearn.pipeline._fit_transform_one...
    _fit_transform_one(PCA(copy=True, iterated_power=7, n_components=4, random_state=None,
      svd_solver='auto', tol=0.0, whiten=False), 
    array([[0., ..., 0.],
           ...,
           [0., ..., 0.]]), array([0, ..., 8]), None)
    ________________________________________________fit_transform_one - 0.0s, 0.0min
    ________________________________________________________________________________
    [Memory] Calling sklearn.pipeline._fit_transform_one...
    _fit_transform_one(PCA(copy=True, iterated_power=7, n_components=4, random_state=None,
      svd_solver='auto', tol=0.0, whiten=False), 
    array([[0., ..., 0.],
           ...,
           [0., ..., 0.]]), array([0, ..., 8]), None)
    ________________________________________________fit_transform_one - 0.0s, 0.0min
    ________________________________________________________________________________
    [Memory] Calling sklearn.pipeline._fit_transform_one...
    _fit_transform_one(PCA(copy=True, iterated_power=7, n_components=4, random_state=None,
      svd_solver='auto', tol=0.0, whiten=False), 
    array([[0., ..., 0.],
           ...,
           [0., ..., 0.]]), array([0, ..., 9]), None)
    ________________________________________________fit_transform_one - 0.0s, 0.0min
    ________________________________________________________________________________
    [Memory] Calling sklearn.pipeline._fit_transform_one...
    _fit_transform_one(PCA(copy=True, iterated_power=7, n_components=8, random_state=None,
      svd_solver='auto', tol=0.0, whiten=False), 
    array([[0., ..., 0.],
           ...,
           [0., ..., 0.]]), array([0, ..., 8]), None)
    ________________________________________________fit_transform_one - 0.0s, 0.0min
    ________________________________________________________________________________
    [Memory] Calling sklearn.pipeline._fit_transform_one...
    _fit_transform_one(PCA(copy=True, iterated_power=7, n_components=8, random_state=None,
      svd_solver='auto', tol=0.0, whiten=False), 
    array([[0., ..., 0.],
           ...,
           [0., ..., 0.]]), array([0, ..., 8]), None)
    ________________________________________________fit_transform_one - 0.0s, 0.0min
    ________________________________________________________________________________
    [Memory] Calling sklearn.pipeline._fit_transform_one...
    _fit_transform_one(PCA(copy=True, iterated_power=7, n_components=8, random_state=None,
      svd_solver='auto', tol=0.0, whiten=False), 
    array([[0., ..., 0.],
           ...,
           [0., ..., 0.]]), array([0, ..., 8]), None)
    ________________________________________________fit_transform_one - 0.0s, 0.0min
    ________________________________________________________________________________
    [Memory] Calling sklearn.pipeline._fit_transform_one...
    _fit_transform_one(PCA(copy=True, iterated_power=7, n_components=8, random_state=None,
      svd_solver='auto', tol=0.0, whiten=False), 
    array([[0., ..., 0.],
           ...,
           [0., ..., 0.]]), array([0, ..., 8]), None)
    ________________________________________________fit_transform_one - 0.0s, 0.0min
    ________________________________________________________________________________
    [Memory] Calling sklearn.pipeline._fit_transform_one...
    _fit_transform_one(PCA(copy=True, iterated_power=7, n_components=8, random_state=None,
      svd_solver='auto', tol=0.0, whiten=False), 
    array([[0., ..., 0.],
           ...,
           [0., ..., 0.]]), array([0, ..., 9]), None)
    ________________________________________________fit_transform_one - 0.0s, 0.0min
    ________________________________________________________________________________
    [Memory] Calling sklearn.pipeline._fit_transform_one...
    _fit_transform_one(NMF(alpha=0.0, beta_loss='frobenius', init=None, l1_ratio=0.0, max_iter=200,
      n_components=2, random_state=None, shuffle=False, solver='cd',
      tol=0.0001, verbose=0), 
    array([[0., ..., 0.],
           ...,
           [0., ..., 0.]]), array([0, ..., 8]), None)
    ________________________________________________fit_transform_one - 0.0s, 0.0min
    ________________________________________________________________________________
    [Memory] Calling sklearn.pipeline._fit_transform_one...
    _fit_transform_one(NMF(alpha=0.0, beta_loss='frobenius', init=None, l1_ratio=0.0, max_iter=200,
      n_components=2, random_state=None, shuffle=False, solver='cd',
      tol=0.0001, verbose=0), 
    array([[0., ..., 0.],
           ...,
           [0., ..., 0.]]), array([0, ..., 8]), None)
    ________________________________________________fit_transform_one - 0.0s, 0.0min
    ________________________________________________________________________________
    [Memory] Calling sklearn.pipeline._fit_transform_one...
    _fit_transform_one(NMF(alpha=0.0, beta_loss='frobenius', init=None, l1_ratio=0.0, max_iter=200,
      n_components=2, random_state=None, shuffle=False, solver='cd',
      tol=0.0001, verbose=0), 
    array([[0., ..., 0.],
           ...,
           [0., ..., 0.]]), array([0, ..., 8]), None)
    ________________________________________________fit_transform_one - 0.0s, 0.0min
    ________________________________________________________________________________
    [Memory] Calling sklearn.pipeline._fit_transform_one...
    _fit_transform_one(NMF(alpha=0.0, beta_loss='frobenius', init=None, l1_ratio=0.0, max_iter=200,
      n_components=2, random_state=None, shuffle=False, solver='cd',
      tol=0.0001, verbose=0), 
    array([[0., ..., 0.],
           ...,
           [0., ..., 0.]]), array([0, ..., 8]), None)
    ________________________________________________fit_transform_one - 0.0s, 0.0min
    ________________________________________________________________________________
    [Memory] Calling sklearn.pipeline._fit_transform_one...
    _fit_transform_one(NMF(alpha=0.0, beta_loss='frobenius', init=None, l1_ratio=0.0, max_iter=200,
      n_components=2, random_state=None, shuffle=False, solver='cd',
      tol=0.0001, verbose=0), 
    array([[0., ..., 0.],
           ...,
           [0., ..., 0.]]), array([0, ..., 9]), None)
    ________________________________________________fit_transform_one - 0.0s, 0.0min
    ________________________________________________________________________________
    [Memory] Calling sklearn.pipeline._fit_transform_one...
    _fit_transform_one(NMF(alpha=0.0, beta_loss='frobenius', init=None, l1_ratio=0.0, max_iter=200,
      n_components=4, random_state=None, shuffle=False, solver='cd',
      tol=0.0001, verbose=0), 
    array([[0., ..., 0.],
           ...,
           [0., ..., 0.]]), array([0, ..., 8]), None)
    ________________________________________________fit_transform_one - 0.0s, 0.0min
    ________________________________________________________________________________
    [Memory] Calling sklearn.pipeline._fit_transform_one...
    _fit_transform_one(NMF(alpha=0.0, beta_loss='frobenius', init=None, l1_ratio=0.0, max_iter=200,
      n_components=4, random_state=None, shuffle=False, solver='cd',
      tol=0.0001, verbose=0), 
    array([[0., ..., 0.],
           ...,
           [0., ..., 0.]]), array([0, ..., 8]), None)
    ________________________________________________fit_transform_one - 0.0s, 0.0min
    ________________________________________________________________________________
    [Memory] Calling sklearn.pipeline._fit_transform_one...
    _fit_transform_one(NMF(alpha=0.0, beta_loss='frobenius', init=None, l1_ratio=0.0, max_iter=200,
      n_components=4, random_state=None, shuffle=False, solver='cd',
      tol=0.0001, verbose=0), 
    array([[0., ..., 0.],
           ...,
           [0., ..., 0.]]), array([0, ..., 8]), None)
    ________________________________________________fit_transform_one - 0.0s, 0.0min
    ________________________________________________________________________________
    [Memory] Calling sklearn.pipeline._fit_transform_one...
    _fit_transform_one(NMF(alpha=0.0, beta_loss='frobenius', init=None, l1_ratio=0.0, max_iter=200,
      n_components=4, random_state=None, shuffle=False, solver='cd',
      tol=0.0001, verbose=0), 
    array([[0., ..., 0.],
           ...,
           [0., ..., 0.]]), array([0, ..., 8]), None)
    ________________________________________________fit_transform_one - 0.0s, 0.0min
    ________________________________________________________________________________
    [Memory] Calling sklearn.pipeline._fit_transform_one...
    _fit_transform_one(NMF(alpha=0.0, beta_loss='frobenius', init=None, l1_ratio=0.0, max_iter=200,
      n_components=4, random_state=None, shuffle=False, solver='cd',
      tol=0.0001, verbose=0), 
    array([[0., ..., 0.],
           ...,
           [0., ..., 0.]]), array([0, ..., 9]), None)
    ________________________________________________fit_transform_one - 0.0s, 0.0min
    ________________________________________________________________________________
    [Memory] Calling sklearn.pipeline._fit_transform_one...
    _fit_transform_one(NMF(alpha=0.0, beta_loss='frobenius', init=None, l1_ratio=0.0, max_iter=200,
      n_components=8, random_state=None, shuffle=False, solver='cd',
      tol=0.0001, verbose=0), 
    array([[0., ..., 0.],
           ...,
           [0., ..., 0.]]), array([0, ..., 8]), None)
    ________________________________________________fit_transform_one - 0.0s, 0.0min
    ________________________________________________________________________________
    [Memory] Calling sklearn.pipeline._fit_transform_one...
    _fit_transform_one(NMF(alpha=0.0, beta_loss='frobenius', init=None, l1_ratio=0.0, max_iter=200,
      n_components=8, random_state=None, shuffle=False, solver='cd',
      tol=0.0001, verbose=0), 
    array([[0., ..., 0.],
           ...,
           [0., ..., 0.]]), array([0, ..., 8]), None)
    ________________________________________________fit_transform_one - 0.0s, 0.0min
    ________________________________________________________________________________
    [Memory] Calling sklearn.pipeline._fit_transform_one...
    _fit_transform_one(NMF(alpha=0.0, beta_loss='frobenius', init=None, l1_ratio=0.0, max_iter=200,
      n_components=8, random_state=None, shuffle=False, solver='cd',
      tol=0.0001, verbose=0), 
    array([[0., ..., 0.],
           ...,
           [0., ..., 0.]]), array([0, ..., 8]), None)
    ________________________________________________fit_transform_one - 0.0s, 0.0min
    ________________________________________________________________________________
    [Memory] Calling sklearn.pipeline._fit_transform_one...
    _fit_transform_one(NMF(alpha=0.0, beta_loss='frobenius', init=None, l1_ratio=0.0, max_iter=200,
      n_components=8, random_state=None, shuffle=False, solver='cd',
      tol=0.0001, verbose=0), 
    array([[0., ..., 0.],
           ...,
           [0., ..., 0.]]), array([0, ..., 8]), None)
    ________________________________________________fit_transform_one - 0.0s, 0.0min
    ________________________________________________________________________________
    [Memory] Calling sklearn.pipeline._fit_transform_one...
    _fit_transform_one(NMF(alpha=0.0, beta_loss='frobenius', init=None, l1_ratio=0.0, max_iter=200,
      n_components=8, random_state=None, shuffle=False, solver='cd',
      tol=0.0001, verbose=0), 
    array([[0., ..., 0.],
           ...,
           [0., ..., 0.]]), array([0, ..., 9]), None)
    ________________________________________________fit_transform_one - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\7630a23b4a54defac925a56610d8d88a
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\6ee7bf3ad1af81808530d98717aca42c
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\b901760747e46067a8e02e3ce69c4182
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\e7263be6aaa4c6563c386df27e71f6ae
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\b60abd1fe1c3ac52278e8c67fd477907
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\df9f277cbd5d6dedfe65bdc89ffa446b
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\542bc3c0692e7ff7e5a29a9b0b5dfb70
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\e328d07f7d27285b4efa85633dc0ae32
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\d4e36de6392376382e6f22ecb839f806
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\5bda2d847240472971feb73252c26651
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\d9620081177a0c30abf1627057c4b5ff
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\f30cc26efeb2b8374cfa2ea9642a8d5e
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\f0bfdf27e49baf36719029f6ec813b7c
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\f0fc64669f4e0aefad379f9f83552ac1
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\6ab3fc400385f416313a9a36f9e91fcb
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\1e141fe01848dc860fd1d585ca4e8632
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\7e95a4bb3b95f927cdc2cb1e556eb36d
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\663b98eabfd03900b330a9fbfd73204e
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\7d4fefe42a129adb465c2725efcd9e82
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\ebd59b93503292e54249be4a191a8ea7
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\d4226737280c65d0977b02cbad3a5d73
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\3d2e1fb61333f83a20d0d390531ffd12
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\0b676cb2c9df186219115b82b0317447
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\d39c11eb4706a348b80279e00f9faea7
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\005939da9284ba2df9263cab4c201181
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\9abc43f87b2915db099ab9081706b367
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\9b7dae9d74a3749dbfa481ef2b11a404
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\c554ca9aa1090ea76a026cb40586d3e7
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\18e62015558e2a9f8451e7f295da77bd
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\86be68a1562167a4406a050f08a12c8d
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\7630a23b4a54defac925a56610d8d88a
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\6ee7bf3ad1af81808530d98717aca42c
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\b901760747e46067a8e02e3ce69c4182
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\e7263be6aaa4c6563c386df27e71f6ae
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\b60abd1fe1c3ac52278e8c67fd477907
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\df9f277cbd5d6dedfe65bdc89ffa446b
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\542bc3c0692e7ff7e5a29a9b0b5dfb70
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\e328d07f7d27285b4efa85633dc0ae32
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\d4e36de6392376382e6f22ecb839f806
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\5bda2d847240472971feb73252c26651
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\d9620081177a0c30abf1627057c4b5ff
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\f30cc26efeb2b8374cfa2ea9642a8d5e
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\f0bfdf27e49baf36719029f6ec813b7c
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\f0fc64669f4e0aefad379f9f83552ac1
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\6ab3fc400385f416313a9a36f9e91fcb
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\1e141fe01848dc860fd1d585ca4e8632
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\7e95a4bb3b95f927cdc2cb1e556eb36d
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\663b98eabfd03900b330a9fbfd73204e
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\7d4fefe42a129adb465c2725efcd9e82
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\ebd59b93503292e54249be4a191a8ea7
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\d4226737280c65d0977b02cbad3a5d73
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\3d2e1fb61333f83a20d0d390531ffd12
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\0b676cb2c9df186219115b82b0317447
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\d39c11eb4706a348b80279e00f9faea7
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\005939da9284ba2df9263cab4c201181
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\9abc43f87b2915db099ab9081706b367
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\9b7dae9d74a3749dbfa481ef2b11a404
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\c554ca9aa1090ea76a026cb40586d3e7
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\18e62015558e2a9f8451e7f295da77bd
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\86be68a1562167a4406a050f08a12c8d
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\7630a23b4a54defac925a56610d8d88a
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\6ee7bf3ad1af81808530d98717aca42c
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\b901760747e46067a8e02e3ce69c4182
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\e7263be6aaa4c6563c386df27e71f6ae
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\b60abd1fe1c3ac52278e8c67fd477907
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\df9f277cbd5d6dedfe65bdc89ffa446b
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\542bc3c0692e7ff7e5a29a9b0b5dfb70
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\e328d07f7d27285b4efa85633dc0ae32
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\d4e36de6392376382e6f22ecb839f806
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\5bda2d847240472971feb73252c26651
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\d9620081177a0c30abf1627057c4b5ff
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\f30cc26efeb2b8374cfa2ea9642a8d5e
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\f0bfdf27e49baf36719029f6ec813b7c
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\f0fc64669f4e0aefad379f9f83552ac1
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\6ab3fc400385f416313a9a36f9e91fcb
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\1e141fe01848dc860fd1d585ca4e8632
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\7e95a4bb3b95f927cdc2cb1e556eb36d
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\663b98eabfd03900b330a9fbfd73204e
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\7d4fefe42a129adb465c2725efcd9e82
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\ebd59b93503292e54249be4a191a8ea7
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\d4226737280c65d0977b02cbad3a5d73
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\3d2e1fb61333f83a20d0d390531ffd12
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\0b676cb2c9df186219115b82b0317447
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\d39c11eb4706a348b80279e00f9faea7
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\005939da9284ba2df9263cab4c201181
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\9abc43f87b2915db099ab9081706b367
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\9b7dae9d74a3749dbfa481ef2b11a404
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\c554ca9aa1090ea76a026cb40586d3e7
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\18e62015558e2a9f8451e7f295da77bd
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\86be68a1562167a4406a050f08a12c8d
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    ________________________________________________________________________________
    [Memory] Calling sklearn.pipeline._fit_transform_one...
    _fit_transform_one(SelectKBest(k=2, score_func=<function chi2 at 0x0000000008955C80>), array([[0., ..., 0.],
           ...,
           [0., ..., 0.]]), array([0, ..., 8]), None)
    ________________________________________________fit_transform_one - 0.0s, 0.0min
    ________________________________________________________________________________
    [Memory] Calling sklearn.pipeline._fit_transform_one...
    _fit_transform_one(SelectKBest(k=2, score_func=<function chi2 at 0x0000000008955C80>), array([[0., ..., 0.],
           ...,
           [0., ..., 0.]]), array([0, ..., 8]), None)
    ________________________________________________fit_transform_one - 0.0s, 0.0min
    ________________________________________________________________________________
    [Memory] Calling sklearn.pipeline._fit_transform_one...
    _fit_transform_one(SelectKBest(k=2, score_func=<function chi2 at 0x0000000008955C80>), array([[0., ..., 0.],
           ...,
           [0., ..., 0.]]), array([0, ..., 8]), None)
    ________________________________________________fit_transform_one - 0.0s, 0.0min
    ________________________________________________________________________________
    [Memory] Calling sklearn.pipeline._fit_transform_one...
    _fit_transform_one(SelectKBest(k=2, score_func=<function chi2 at 0x0000000008955C80>), array([[0., ..., 0.],
           ...,
           [0., ..., 0.]]), array([0, ..., 8]), None)
    ________________________________________________fit_transform_one - 0.0s, 0.0min
    ________________________________________________________________________________
    [Memory] Calling sklearn.pipeline._fit_transform_one...
    _fit_transform_one(SelectKBest(k=2, score_func=<function chi2 at 0x0000000008955C80>), array([[0., ..., 0.],
           ...,
           [0., ..., 0.]]), array([0, ..., 9]), None)
    ________________________________________________fit_transform_one - 0.0s, 0.0min
    ________________________________________________________________________________
    [Memory] Calling sklearn.pipeline._fit_transform_one...
    _fit_transform_one(SelectKBest(k=4, score_func=<function chi2 at 0x0000000008955C80>), array([[0., ..., 0.],
           ...,
           [0., ..., 0.]]), array([0, ..., 8]), None)
    ________________________________________________fit_transform_one - 0.0s, 0.0min
    ________________________________________________________________________________
    [Memory] Calling sklearn.pipeline._fit_transform_one...
    _fit_transform_one(SelectKBest(k=4, score_func=<function chi2 at 0x0000000008955C80>), array([[0., ..., 0.],
           ...,
           [0., ..., 0.]]), array([0, ..., 8]), None)
    ________________________________________________fit_transform_one - 0.0s, 0.0min
    ________________________________________________________________________________
    [Memory] Calling sklearn.pipeline._fit_transform_one...
    _fit_transform_one(SelectKBest(k=4, score_func=<function chi2 at 0x0000000008955C80>), array([[0., ..., 0.],
           ...,
           [0., ..., 0.]]), array([0, ..., 8]), None)
    ________________________________________________fit_transform_one - 0.0s, 0.0min
    ________________________________________________________________________________
    [Memory] Calling sklearn.pipeline._fit_transform_one...
    _fit_transform_one(SelectKBest(k=4, score_func=<function chi2 at 0x0000000008955C80>), array([[0., ..., 0.],
           ...,
           [0., ..., 0.]]), array([0, ..., 8]), None)
    ________________________________________________fit_transform_one - 0.0s, 0.0min
    ________________________________________________________________________________
    [Memory] Calling sklearn.pipeline._fit_transform_one...
    _fit_transform_one(SelectKBest(k=4, score_func=<function chi2 at 0x0000000008955C80>), array([[0., ..., 0.],
           ...,
           [0., ..., 0.]]), array([0, ..., 9]), None)
    ________________________________________________fit_transform_one - 0.0s, 0.0min
    ________________________________________________________________________________
    [Memory] Calling sklearn.pipeline._fit_transform_one...
    _fit_transform_one(SelectKBest(k=8, score_func=<function chi2 at 0x0000000008955C80>), array([[0., ..., 0.],
           ...,
           [0., ..., 0.]]), array([0, ..., 8]), None)
    ________________________________________________fit_transform_one - 0.0s, 0.0min
    ________________________________________________________________________________
    [Memory] Calling sklearn.pipeline._fit_transform_one...
    _fit_transform_one(SelectKBest(k=8, score_func=<function chi2 at 0x0000000008955C80>), array([[0., ..., 0.],
           ...,
           [0., ..., 0.]]), array([0, ..., 8]), None)
    ________________________________________________fit_transform_one - 0.0s, 0.0min
    ________________________________________________________________________________
    [Memory] Calling sklearn.pipeline._fit_transform_one...
    _fit_transform_one(SelectKBest(k=8, score_func=<function chi2 at 0x0000000008955C80>), array([[0., ..., 0.],
           ...,
           [0., ..., 0.]]), array([0, ..., 8]), None)
    ________________________________________________fit_transform_one - 0.0s, 0.0min
    ________________________________________________________________________________
    [Memory] Calling sklearn.pipeline._fit_transform_one...
    _fit_transform_one(SelectKBest(k=8, score_func=<function chi2 at 0x0000000008955C80>), array([[0., ..., 0.],
           ...,
           [0., ..., 0.]]), array([0, ..., 8]), None)
    ________________________________________________fit_transform_one - 0.0s, 0.0min
    ________________________________________________________________________________
    [Memory] Calling sklearn.pipeline._fit_transform_one...
    _fit_transform_one(SelectKBest(k=8, score_func=<function chi2 at 0x0000000008955C80>), array([[0., ..., 0.],
           ...,
           [0., ..., 0.]]), array([0, ..., 9]), None)
    ________________________________________________fit_transform_one - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\a1b77f931883b53c89c9e06e428f6b54
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\2a9f6dd6266c76d5fe21a16dbd373bf8
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\92c06345967df8cc36d48372806c9e01
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\beb97c1eac9aa73e456f59aa104bdb3e
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\4567718434d5855a81598e706cf74597
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\dfbcc01d9c97f6e8c6c6e1c50150e2af
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\b1069821e36f0d39843c7dba717591cb
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\85d852cf7cb97b38d24451cc6c4736d6
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\bd4113b84943f5dccfe43cb48b400faa
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\28c9a7c94e8dc77d5506e73af50a60cc
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\0f92659eedb07143df5b5932ffe62d89
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\5a6536f77dca05e0e061c651492f8ffc
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\c9793e5f0aa3780a9b590447172b6e21
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\057b9c23db8a0b794a0f8650e5ece577
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\9e8216479fa538dbbed75a16687359ed
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\a1b77f931883b53c89c9e06e428f6b54
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\2a9f6dd6266c76d5fe21a16dbd373bf8
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\92c06345967df8cc36d48372806c9e01
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\beb97c1eac9aa73e456f59aa104bdb3e
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\4567718434d5855a81598e706cf74597
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\dfbcc01d9c97f6e8c6c6e1c50150e2af
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\b1069821e36f0d39843c7dba717591cb
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\85d852cf7cb97b38d24451cc6c4736d6
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\bd4113b84943f5dccfe43cb48b400faa
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\28c9a7c94e8dc77d5506e73af50a60cc
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\0f92659eedb07143df5b5932ffe62d89
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\5a6536f77dca05e0e061c651492f8ffc
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\c9793e5f0aa3780a9b590447172b6e21
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\057b9c23db8a0b794a0f8650e5ece577
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\9e8216479fa538dbbed75a16687359ed
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\a1b77f931883b53c89c9e06e428f6b54
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\2a9f6dd6266c76d5fe21a16dbd373bf8
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\92c06345967df8cc36d48372806c9e01
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\beb97c1eac9aa73e456f59aa104bdb3e
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\4567718434d5855a81598e706cf74597
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\dfbcc01d9c97f6e8c6c6e1c50150e2af
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\b1069821e36f0d39843c7dba717591cb
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\85d852cf7cb97b38d24451cc6c4736d6
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\bd4113b84943f5dccfe43cb48b400faa
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\28c9a7c94e8dc77d5506e73af50a60cc
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\0f92659eedb07143df5b5932ffe62d89
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\5a6536f77dca05e0e061c651492f8ffc
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\c9793e5f0aa3780a9b590447172b6e21
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\057b9c23db8a0b794a0f8650e5ece577
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    [Memory]0.0s, 0.0min    : Loading _fit_transform_one from C:\Users\antares\AppData\Local\Temp\tmp3cs__ib2\joblib\sklearn\pipeline\_fit_transform_one\9e8216479fa538dbbed75a16687359ed
    ___________________________________fit_transform_one cache loaded - 0.0s, 0.0min
    ________________________________________________________________________________
    [Memory] Calling sklearn.pipeline._fit_transform_one...
    _fit_transform_one(NMF(alpha=0.0, beta_loss='frobenius', init=None, l1_ratio=0.0, max_iter=200,
      n_components=8, random_state=None, shuffle=False, solver='cd',
      tol=0.0001, verbose=0), 
    array([[0., ..., 0.],
           ...,
           [0., ..., 0.]]), array([0, ..., 8]), None)
    ________________________________________________fit_transform_one - 0.0s, 0.0min


The ``PCA`` fitting is only computed at the evaluation of the first
configuration of the ``C`` parameter of the ``LinearSVC`` classifier. The
other configurations of ``C`` will trigger the loading of the cached ``PCA``
estimator data, leading to save processing time. Therefore, the use of
caching the pipeline using ``memory`` is highly beneficial when fitting
a transformer is costly.


**Total running time of the script:** ( 2 minutes  5.611 seconds)


.. _sphx_glr_download_auto_examples_compose_plot_compare_reduction.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_compare_reduction.py <plot_compare_reduction.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_compare_reduction.ipynb <plot_compare_reduction.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
