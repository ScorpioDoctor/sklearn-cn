#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
手写数字数据集(Digit Dataset)
=========================================================

该数据集由 1797 个  8x8 的图像构成。 每张图像,
就像下面展示的那张一样, 对应着一个手写数字。
为了使用像这样的一个 8x8 图像, 我们要
首先将其变换为长度为 64 的特征向量(feature vector)。

请看 `这里
<http://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits>`_
获得该数据集的更多详情。^_^
"""
print(__doc__)


# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# 翻译 和 测试 by Antares博士
# License: BSD 3 clause

from sklearn import datasets

import matplotlib.pyplot as plt

#加载 digits dataset
digits = datasets.load_digits()

#显示第一个 digit
plt.figure(1, figsize=(3, 3))
plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()



##############################################################################
#                                   Antares的测试信息
# 
# 
#                    可能会发生 UnicodeDecodeError: 
# 'gbk' codec can't decode byte 0x86 in position 41: illegal multibyte sequence
# Traceback (most recent call last):
#   File "c:/GitHub/sklearn-cn/scikit-learn-0.20.2/examples/datasets/plot_digits_last_image.py", line 31, in <module>
#     digits = datasets.load_digits()
#   File "c:\github\sklearn-cn\scikit-learn-0.20.2\sklearn\datasets\base.py", line 549, in load_digits
#     descr = f.read()
# 根据上面的TraceBack信息，我们知道是 sklearn的 load_digits() 这个函数出现问题，而且是 base.py 的 549 行
# 我们定位到这一行，发现是读取 rst 文件的一段代码，
#             with open(join(module_path, 'descr', 'digits.rst')) as f:
#                   descr = f.read()
# 我们只要将上面这段代码打开文件的格式改成: 'rb' 即以只读二进制方式打开就好了
#          with open(join(module_path, 'descr', 'digits.rst'),'rb') as f:

##########################################################################################################