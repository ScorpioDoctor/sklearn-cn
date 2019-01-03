
==========================================================================
统计学习: 问题设置以及scikit-learn中的估计器对象
==========================================================================


数据集
=========

Scikit-learn处理来自一个或多个数据集的学习信息，这些信息通常被表示为2D数组。
这个2D数组可以理解成很多的多维观测的一个列表。2D数组的第一个axis是 **samples** axis,是样本索引，
第二个axis是**features** axis，是特征索引。 

.. topic:: scikit-learn自带的一个简单例子: iris dataset

    ::

        >>> from sklearn import datasets
        >>> iris = datasets.load_iris()
        >>> data = iris.data
        >>> data.shape
        (150, 4)

    上面这个数据集包含了对irises的150个样本的观测数据，每个观测数据描述了irise的四个特征：
    their sepal and petal length and width, 每个特征分量的实际意义可以在 ``iris.DESCR`` 中找到。

如果数据的shape一开始不是 ``(n_samples, n_features)`` 这样的2D array, 那么必须要经过预处理才可以在
scikit-learn 中使用.

.. topic:: 下面的例子展示了digits数据集是如何进行reshape操作的。

    .. image:: /auto_examples/datasets/images/sphx_glr_plot_digits_last_image_001.png
        :target: ../../auto_examples/datasets/plot_digits_last_image.html
        :align: right
        :scale: 60

    digits 数据集包含了1797个8x8的手写字符图像 ::

        >>> digits = datasets.load_digits()
        >>> digits.images.shape
        (1797, 8, 8)
        >>> import matplotlib.pyplot as plt #doctest: +SKIP
        >>> plt.imshow(digits.images[-1], cmap=plt.cm.gray_r) #doctest: +SKIP
        <matplotlib.image.AxesImage object at ...>

    要想在scikit-learn中使用digits数据, 我们需要将每一个 8x8 的字符图像变换成一个长度为64的特征向量 ::

        >>> data = digits.images.reshape((digits.images.shape[0], -1))


估计器对象
===================

.. Some code to make the doctests run

   >>> from sklearn.base import BaseEstimator
   >>> class Estimator(BaseEstimator):
   ...      def __init__(self, param1=0, param2=0):
   ...          self.param1 = param1
   ...          self.param2 = param2
   ...      def fit(self, data):
   ...          pass
   >>> estimator = Estimator()

**Fitting data**: scikit-learn 实现的主要API就是 `estimator`。 
估计器是一个可以从数据中学习的任何对象(An estimator is any object 
that learns from data)；它可以是分类器，回归器，聚类器或者是一个从
原始数据中抽取和过滤有用特征的变换器(*transformer*)。

所有的estimator对象都会暴露一个 ``fit`` 方法，该方法接受一个dataset作为参数，
(通常是一个 2-d array):

    >>> estimator.fit(data)

**Estimator parameters**: estimator的所有参数都可以在它被实例化的时候设置，或者通过对应的属性进行修改 ::

    >>> estimator = Estimator(param1=1, param2=2)
    >>> estimator.param1
    1

**Estimated parameters**: 当estimator在数据上进行了拟合以后, 从数据中估计出(学习到)的参数就到手了。
所有学习到的参数都是estimator对象的属性，这些属性的命名有一定的规则 ::

    >>> estimator.estimated_param_ #doctest: +SKIP


scikit-learn的统一API调用接口
=====================================

.. topic:: 译者注:

    你想知道scikit-learn的设计者是怎么把这么多各种各样的机器学习算法设计到一块儿去的吗？
    换句话说，如果你现在对scikit-learn的统一API调用接口有了一个全局性的掌握，那么你就不会在
    后面的学习中面对浩如烟海的机器学习算法和千变万化的参数设置中迷失自己。
    这里有一节视频介绍了scikit-learn的算法库的顶层设计和统一API调用接口，希望对大家有所帮助。视频地址：
    (`scikit-learn的算法库的顶层设计 <http://www.studyai.com/course/play/d11b9c9d12d0452a8e4be6cda74de347>`_)
    (`scikit-learn的统一API调用接口 <http://www.studyai.com/course/play/64e0ac84083745daa894200cc0f04aa6>`_)
