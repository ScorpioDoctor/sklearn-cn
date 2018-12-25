.. _isotonic:

===================
保序回归(Isotonic regression)
===================

.. currentmodule:: sklearn.isotonic

类:class:`IsotonicRegression` 在数据上拟合一个非递减函数(non-decreasing function)。
它求解下面的问题:

  minimize :math:`\sum_i w_i (y_i - \hat{y}_i)^2`

  subject to :math:`\hat{y}_{min} = \hat{y}_1 \le \hat{y}_2 ... \le \hat{y}_n = \hat{y}_{max}`

其中 每一个 :math:`w_i` 是严格正的 并且 每一个 :math:`y_i` 是一个任意的实数。
它生成由非减元素组成的向量，是最接近均方误差的(mean squared error)。 
实际上，这样的元素列表构成了一个分段线性(piecewise linear)的函数。

.. figure:: ../auto_examples/images/sphx_glr_plot_isotonic_regression_001.png
   :target: ../auto_examples/plot_isotonic_regression.html
   :align: center
