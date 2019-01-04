"""
======================================================================
压缩感知: 使用 L1 prior (Lasso) 进行层析重建
======================================================================

这个例子显示了从一组平行投影中重建图像的过程，这些投影是沿着不同的角度获得的。
这种数据集是在计算机断层扫描(**computed tomography** (CT))中获取的。

在没有关于样本的任何先验信息的情况下，重建图像所需的投影数是图像线性大小 ``l`` 的数量级(以像素为单位)。
为了简单起见，我们在这里考虑稀疏图像，其中只有物体边界上的像素有一个非零值。
例如，这样的数据可以对应于蜂窝材料。 但是，请注意，大多数图像都是在不同的基(basis)上稀疏的，例如Haar小波。
只获得了 ``l/7`` 投影，因此有必要使用现有的样本上的先验信息(其稀疏性)：这是压缩感知(**compressive sensing**)的一个例子。

层析投影操作( tomography projection operation)是一种线性变换。除了对应于线性回归的数据保真度项(data-fidelity term)外，
我们还惩罚图像的L1范数以便把图像的稀疏性也考虑到模型中去。 由此产生的优化问题称为 :ref:`lasso` 问题。
我们使用 :class:`sklearn.linear_model.Lasso` 类，它使用坐标下降算法。
重要的是，这个实现在稀疏矩阵上的计算效率比这里使用的投影算子(projection operator)更高效。

L1惩罚的重建结果为零误差(所有像素都被成功标记为0或1)，即使在投影中添加了噪声。
相比之下，L2惩罚(:class:`sklearn.linear_model.Ridge`)会产生大量的像素标记错误。
在重建图像上观察到重要的伪影(Important artifacts)，这与L1惩罚相反。特别要注意的是，圆形伪影将角上的像素分隔开来，
而与中央圆盘内的像素相比，角点处的像素只贡献了很少的投影。
"""

from __future__ import division

print(__doc__)

# Author: Emmanuelle Gouillart <emmanuelle.gouillart@nsup.org>
# License: BSD 3 clause
# 翻译者：studyai.com的Antares博士

import numpy as np
from scipy import sparse
from scipy import ndimage
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt


def _weights(x, dx=1, orig=0):
    x = np.ravel(x)
    floor_x = np.floor((x - orig) / dx).astype(np.int64)
    alpha = (x - orig - floor_x * dx) / dx
    return np.hstack((floor_x, floor_x + 1)), np.hstack((1 - alpha, alpha))


def _generate_center_coordinates(l_x):
    X, Y = np.mgrid[:l_x, :l_x].astype(np.float64)
    center = l_x / 2.
    X += 0.5 - center
    Y += 0.5 - center
    return X, Y


def build_projection_operator(l_x, n_dir):
    """ 计算层析设计矩阵

    参数
    ----------

    l_x : int
        图像数组的线性长度

    n_dir : int
        获得投影所需的角度的数量.

    Returns
    -------
    p :  shape 为 (n_dir l_x, l_x**2) 的稀疏矩阵
    """
    X, Y = _generate_center_coordinates(l_x)
    angles = np.linspace(0, np.pi, n_dir, endpoint=False)
    data_inds, weights, camera_inds = [], [], []
    data_unravel_indices = np.arange(l_x ** 2)
    data_unravel_indices = np.hstack((data_unravel_indices,
                                      data_unravel_indices))
    for i, angle in enumerate(angles):
        Xrot = np.cos(angle) * X - np.sin(angle) * Y
        inds, w = _weights(Xrot, dx=1, orig=X.min())
        mask = np.logical_and(inds >= 0, inds < l_x)
        weights += list(w[mask])
        camera_inds += list(inds[mask] + i * l_x)
        data_inds += list(data_unravel_indices[mask])
    proj_operator = sparse.coo_matrix((weights, (camera_inds, data_inds)))
    return proj_operator


def generate_synthetic_data():
    """ 合成二进制数据 """
    rs = np.random.RandomState(0)
    n_pts = 36
    x, y = np.ogrid[0:l, 0:l]
    mask_outer = (x - l / 2.) ** 2 + (y - l / 2.) ** 2 < (l / 2.) ** 2
    mask = np.zeros((l, l))
    points = l * rs.rand(2, n_pts)
    mask[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
    mask = ndimage.gaussian_filter(mask, sigma=l / n_pts)
    res = np.logical_and(mask > mask.mean(), mask_outer)
    return np.logical_xor(res, ndimage.binary_erosion(res))


# 生成合成图像和投影
l = 128
proj_operator = build_projection_operator(l, l // 7)
data = generate_synthetic_data()
proj = proj_operator * data.ravel()[:, np.newaxis]
proj += 0.15 * np.random.randn(*proj.shape)

# 使用 L2 (Ridge) 惩罚的重建
rgr_ridge = Ridge(alpha=0.2)
rgr_ridge.fit(proj_operator, proj.ravel())
rec_l2 = rgr_ridge.coef_.reshape(l, l)

# 使用 L1 (Lasso) 惩罚的重建
# alpha 的最佳值 使用交叉验证来确定: LassoCV
rgr_lasso = Lasso(alpha=0.001)
rgr_lasso.fit(proj_operator, proj.ravel())
rec_l1 = rgr_lasso.coef_.reshape(l, l)

plt.figure(figsize=(8, 3.3))
plt.subplot(131)
plt.imshow(data, cmap=plt.cm.gray, interpolation='nearest')
plt.axis('off')
plt.title('original image')
plt.subplot(132)
plt.imshow(rec_l2, cmap=plt.cm.gray, interpolation='nearest')
plt.title('L2 penalization')
plt.axis('off')
plt.subplot(133)
plt.imshow(rec_l1, cmap=plt.cm.gray, interpolation='nearest')
plt.title('L1 penalization')
plt.axis('off')

plt.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0,
                    right=1)

plt.show()
