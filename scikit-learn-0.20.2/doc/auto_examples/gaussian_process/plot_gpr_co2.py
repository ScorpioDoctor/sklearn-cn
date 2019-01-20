"""
========================================================
高斯过程回归 (GPR) 在 Mauna Loa 二氧化碳数据上的应用
========================================================

该示例基于书([RW2006]) 的第 5.4.3 节。 它演示了使用梯度上升的对数边缘似然性的复杂内核工程和超参数优化的示例。 
数据包括在 1958 年至 1997 年间夏威夷 Mauna Loa 天文台收集的每月平均大气二氧 化碳浓度（以百万分之几（ppmv）计）。 
目的是将二氧化碳浓度建模为时间 t 的函数。

内核由若干项组成，负责解释信号的不同属性：

- 一个长期平滑的上升趋势是由一个 RBF 内核来解释的。 具有较大长度尺寸的RBF内核将使该分量平滑; 没有强制这种趋势正在上升，
  这给 GP 带来了可选择性。 具体的长度尺度(length-scale)和振幅(amplitude)是自由的超参数。

- 季节性因素，由周期性的 ExpSineSquared 内核来解释，固定周期为1年。 该周期性分量的长度尺度控制其平滑度，是一个自由参数。 
  为了使其具备准确周期性的衰减，将 ExpSineSquared kernel 与 RBF kernel 取乘积。 
  该RBF component的长度尺度(length-scale)控制衰减时间， 并且是另一个自由参数。

- 较小的中期不规则性将由有理二次(RationalQuadratic)核分量来解释， 有理二次核分量的长度尺度和alpha 参数，
  决定着长度尺度的扩散性，是将要被确定的参数。 根据 [RW2006] ，这些不规则性可以更好地由有理二次内核来解释， 
  而不是 RBF 核分量，这可能是因为它可以容纳若干个长度尺度(length-scale)。

- 噪声项，由一个 RBF 核组成，它将解释相关的噪声分量，如局部天气现象以及 WhiteKernel 对白噪声的贡献。 
  在这里，相对幅度(relative amplitudes)和RBF的长度尺度又是自由参数。

减去目标平均值后 最大化 对数边际似然(log-marginal-likelihood)产生下列内核，其中LML为-83.214:

   34.4**2 * RBF(length_scale=41.8)
   + 3.27**2 * RBF(length_scale=180) * ExpSineSquared(length_scale=1.44,
                                                      periodicity=1)
   + 0.446**2 * RationalQuadratic(alpha=17.7, length_scale=0.957)
   + 0.197**2 * RBF(length_scale=0.138) + WhiteKernel(noise_level=0.0336)

因此，大多数目标信号（34.4ppm）由长期上升趋势（长度尺度即length-scale为41.8年）解释。 
周期分量的振幅为3.27ppm，衰减时间为180年，长度尺度为1.44。 
长时间的衰变时间表明我们有一个局部非常接近周期性的季节性成分。 
相关噪声的幅度为0.197ppm，长度为0.138年，白噪声贡献为0.197ppm。 
因此，整体噪声水平非常小，表明该模型可以很好地解释数据。 
该图还显示，该模型直到2015年左右才能做出置信度比较高的预测。
"""
# Authors: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
#翻译者：http://www.studyai.com/antares
# License: BSD 3 clause

from __future__ import division, print_function

import numpy as np

from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
    import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared

print(__doc__)


def load_mauna_loa_atmospheric_co2():
    ml_data = fetch_openml(data_id=41187)
    months = []
    ppmv_sums = []
    counts = []

    y = ml_data.data[:, 0]
    m = ml_data.data[:, 1]
    month_float = y + (m - 1) / 12
    ppmvs = ml_data.target

    for month, ppmv in zip(month_float, ppmvs):
        if not months or month != months[-1]:
            months.append(month)
            ppmv_sums.append(ppmv)
            counts.append(1)
        else:
            # aggregate monthly sum to produce average
            ppmv_sums[-1] += ppmv
            counts[-1] += 1

    months = np.asarray(months).reshape(-1, 1)
    avg_ppmvs = np.asarray(ppmv_sums) / counts
    return months, avg_ppmvs


X, y = load_mauna_loa_atmospheric_co2()

# Kernel with parameters given in GPML book
k1 = 66.0**2 * RBF(length_scale=67.0)  # long term smooth rising trend
k2 = 2.4**2 * RBF(length_scale=90.0) \
    * ExpSineSquared(length_scale=1.3, periodicity=1.0)  # seasonal component
# medium term irregularity
k3 = 0.66**2 \
    * RationalQuadratic(length_scale=1.2, alpha=0.78)
k4 = 0.18**2 * RBF(length_scale=0.134) \
    + WhiteKernel(noise_level=0.19**2)  # noise terms
kernel_gpml = k1 + k2 + k3 + k4

gp = GaussianProcessRegressor(kernel=kernel_gpml, alpha=0,
                              optimizer=None, normalize_y=True)
gp.fit(X, y)

print("GPML kernel: %s" % gp.kernel_)
print("Log-marginal-likelihood: %.3f"
      % gp.log_marginal_likelihood(gp.kernel_.theta))

# Kernel with optimized parameters
k1 = 50.0**2 * RBF(length_scale=50.0)  # long term smooth rising trend
k2 = 2.0**2 * RBF(length_scale=100.0) \
    * ExpSineSquared(length_scale=1.0, periodicity=1.0,
                     periodicity_bounds="fixed")  # seasonal component
# medium term irregularities
k3 = 0.5**2 * RationalQuadratic(length_scale=1.0, alpha=1.0)
k4 = 0.1**2 * RBF(length_scale=0.1) \
    + WhiteKernel(noise_level=0.1**2,
                  noise_level_bounds=(1e-3, np.inf))  # noise terms
kernel = k1 + k2 + k3 + k4

gp = GaussianProcessRegressor(kernel=kernel, alpha=0,
                              normalize_y=True)
gp.fit(X, y)

print("\nLearned kernel: %s" % gp.kernel_)
print("Log-marginal-likelihood: %.3f"
      % gp.log_marginal_likelihood(gp.kernel_.theta))

X_ = np.linspace(X.min(), X.max() + 30, 1000)[:, np.newaxis]
y_pred, y_std = gp.predict(X_, return_std=True)

# Illustration
plt.scatter(X, y, c='k')
plt.plot(X_, y_pred)
plt.fill_between(X_[:, 0], y_pred - y_std, y_pred + y_std,
                 alpha=0.5, color='k')
plt.xlim(X_.min(), X_.max())
plt.xlabel("Year")
plt.ylabel(r"CO$_2$ in ppm")
plt.title(r"Atmospheric CO$_2$ concentration at Mauna Loa")
plt.tight_layout()
plt.show()
