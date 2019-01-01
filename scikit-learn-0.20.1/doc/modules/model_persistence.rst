.. _model_persistence:

=================
模型持久化(Model persistence)
=================

在训练完 scikit-learn 模型之后，最好有一种方法来将模型持久化以备将来使用，而无需重新训练。 
以下部分为您提供了有关如何使用 pickle 来持久化模型的示例。 
在使用 pickle 序列化时，我们还将回顾一些安全性和可维护性方面的问题。


模型持久化案例
-------------------

可以通过使用 Python 的内置持久化模型将训练好的模型保存在 scikit-learn 中，它名为 
`pickle <https://docs.python.org/2/library/pickle.html>`_::

  >>> from sklearn import svm
  >>> from sklearn import datasets
  >>> clf = svm.SVC(gamma='scale')
  >>> iris = datasets.load_iris()
  >>> X, y = iris.data, iris.target
  >>> clf.fit(X, y)  # doctest: +NORMALIZE_WHITESPACE
  SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)

  >>> import pickle
  >>> s = pickle.dumps(clf)
  >>> clf2 = pickle.loads(s)
  >>> clf2.predict(X[0:1])
  array([0])
  >>> y[0]
  0

在这个 scikit-learn 的特殊示例中，使用 joblib 来替换 pickle（joblib.dump & joblib.load）可能会更有意思，
这对于内部带有 numpy 数组的对象来说更为高效， 通常情况下适合 scikit-learn estimators，
但是也只能是 pickle 到硬盘而不是字符串::

  >>> from joblib import dump, load
  >>> dump(clf, 'filename.joblib') # doctest: +SKIP

之后你可以使用以下方式回调 pickled model （可能在另一个 Python 进程中）::

  >>> clf = load('filename.joblib') # doctest:+SKIP

.. note::

   ``dump`` 和 ``load`` 函数也接收类似 file-like 的对象而不是文件名。 
   更多有关使用 Joblib 来持久化数据的信息可以参阅 
   `这里 <https://joblib.readthedocs.io/en/latest/persistence.html>`_。

.. _persistence_limitations:

安全性 & 可维护性限制
--------------------------------------

pickle（和通过扩展的 joblib），在安全性和可维护性方面存在一些问题。 有以下原因,

* 绝对不要使用未经 pickle 的不受信任的数据，因为它可能会在加载时执行恶意代码。
* 虽然一个版本的 scikit-learn 模型可以在其他版本中加载，但这完全不建议并且也是不可取的。 
  还应该了解到，对于这些数据执行的操作可能会产生不同及意想不到的结果。

为了用以后版本的 scikit-learn 来重构类似的模型, 额外的元数据应该随着 pickled model 一起被保存：

* 训练数据，例如：引用不可变的快照
* 用于生成模型的 python 源代码
* scikit-learn 的各版本以及各版本对应的依赖包
* 在训练数据的基础上获得的交叉验证得分

这样可以检查交叉验证得分是否与以前相同。

由于模型内部表示可能在两种不同架构上不一样，因此不支持在一个架构上转储模型并将其加载到另一个体系架构上。

如果您想要了解更多关于这些问题以及其它可能的序列化方法，请参阅这个 Alex Gaynor 的演讲 
`talk by Alex Gaynor <http://pyvideo.org/video/2566/pickles-are-for-delis-not-software>`_.
