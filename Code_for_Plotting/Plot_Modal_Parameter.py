# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import solve

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Calibri']

plt.rcParams['font.weight'] = 'normal'  # light normal heavy bold

"""------------------------------FRF data loading------------------------------"""
'x direction'
df_x = pd.read_excel('../Data_Experiment/FRF_x.xlsx', sheet_name='Single value', header=None)
data_x = df_x.values

freq_total_x = data_x[0, 2:]
real_total_x = data_x[1, 2:]
img_total_x = data_x[2, 2:]

# print('signal length:',len(freq_total))
freq_x = freq_total_x[500:12000]
real_acc_x = real_total_x[500:12000]
img_acc_x = img_total_x[500:12000]

real_disp_x = np.empty_like(real_acc_x)
img_disp_x = np.empty_like(img_acc_x)

for i in range(len(freq_x)):
    real_disp_x[i] = real_acc_x[i] / freq_x[i] ** 2
    img_disp_x[i] = img_acc_x[i] / freq_x[i] ** 2

'y direction'
df_y = pd.read_excel('../Data_Experiment/FRF_x.xlsx', sheet_name='Single value', header=None)
data_y = df_y.values

freq_total_y = data_y[0, 2:]
real_total_y = data_y[1, 2:]
img_total_y = data_y[2, 2:]

freq_y = freq_total_y[500:12000]
real_acc_y = real_total_y[500:12000]
img_acc_y = img_total_y[500:12000]

real_disp_y = np.empty_like(real_acc_y)
img_disp_y = np.empty_like(img_acc_y)

for i in range(len(freq_y)):
    real_disp_y[i] = real_acc_y[i] / freq_y[i] ** 2
    img_disp_y[i] = img_acc_y[i] / freq_y[i] ** 2

'''------------------------------def FRF fitting func------------------------------'''


def FRF_numerator_calcu(wn1, wn2, zeta1, zeta2, img1, img2):
    A = np.array([[1 / (2 * 1j * zeta1 * wn1 ** 2), 1 / (wn2 ** 2 - wn1 ** 2 + 2 * 1j * zeta2 * wn1 * wn2)],
                  [1 / (wn1 ** 2 - wn2 ** 2 + 2 * 1j * zeta1 * wn1 * wn2), 1 / (2 * 1j * zeta2 * wn2 ** 2)]])
    b = np.array([img1, img2])
    ans = solve(A, b)
    return ans


def FRF_fitting(w, wn1, wn2, zeta1, zeta2, num1, num2):
    y = num1 / (wn1 ** 2 - w ** 2 + 2 * 1j * wn1 * zeta1 * w) + num2 / (wn2 ** 2 - w ** 2 + 2 * 1j * wn2 * zeta2 * w);
    return y


'''------------------------------FRF fitting------------------------------'''
'x direction'  # data obtained from Step0_Modal_Parameter_Calcu

wn1_x = 1956.5
wn2_x = 4445.5
zeta1_x = 0.0227
zeta2_x = 0.0136
img1_x = 2.329e-06
img2_x = 2.329665361596143e-07

[num1_x, num2_x] = FRF_numerator_calcu(wn1_x * 2 * np.pi, wn2_x * 2 * np.pi, zeta1_x, zeta2_x, img1_x, img2_x)

# FRF_fitted = np.empty_like(real_acc)
real_fitted_x = np.empty_like(real_acc_x)
img_fitted_x = np.empty_like(img_acc_x)

for i in range(len(freq_x)):
    temp = FRF_fitting(freq_x[i], wn1_x, wn2_x, zeta1_x, zeta2_x, num1_x, num2_x)
    real_fitted_x[i] = temp.imag
    img_fitted_x[i] = temp.real

'y direction'  # data obtained from Step0_Modal_Parameter_Calcu
wn1_y = 1954.5
wn2_y = 4445.5
zeta1_y = 0.0228
zeta2_y = 0.0136
img1_y = 2.158e-06
img2_y = 1.422397068233238e-07

[num1_y, num2_y] = FRF_numerator_calcu(wn1_y * 2 * np.pi, wn2_y * 2 * np.pi, zeta1_y, zeta2_y, img1_y, img2_y)

# FRF_fitted = np.empty_like(real_acc)
real_fitted_y = np.empty_like(real_acc_y)
img_fitted_y = np.empty_like(img_acc_y)

for i in range(len(freq_y)):
    temp = FRF_fitting(freq_y[i], wn1_y, wn2_y, zeta1_y, zeta2_y, num1_y, num2_y)
    real_fitted_y[i] = temp.imag
    img_fitted_y[i] = temp.real
'''------------------------------FRF plotting------------------------------'''
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, figsize=(18, 10))

fontsize_axis = 25
fontsize_title = 25
fontsize_legend = 20
fontsize_tick = 20

ax1.plot(freq_x, real_disp_x, color='k', linewidth=2, linestyle=':', label='FRF measured')
ax1.plot(freq_x, real_fitted_x, color='r', linewidth=1.5, label='FRF fitted')
ax1.set_title(r'FRF($\omega$) in X direction', fontsize=fontsize_title)
ax1.set_ylabel(r'Real FRF (m/N)', fontsize=fontsize_axis)
ax1.set_xlim([500, 5500])
ax1.set_xticks(np.arange(1000, 6000, 1000))
ax1.set_ylim([-6 * 1e-5, 6 * 1e-5])
ax1.set_yticks(np.arange(-6 * 1e-5, 8 * 1e-5, 2 * 1e-5))
ax1.tick_params(axis='x', labelsize=fontsize_tick)
ax1.tick_params(axis='y', labelsize=fontsize_tick)
ax1.legend(fontsize=fontsize_legend)

ax2.plot(freq_y, real_disp_y, color='k', linewidth=2, linestyle=':', label='FRF measured')
ax2.plot(freq_y, real_fitted_y, color='r', linewidth=1.5, label='FRF fitted')

ax2.set_title(r'FRF($\omega$) in Y direction', fontsize=fontsize_title)
ax2.set_ylabel(r'Real FRF (m/N)', fontsize=fontsize_axis)
ax2.set_xlim([500, 5500])
ax2.set_xticks(np.arange(1000, 6000, 1000))
ax2.set_ylim([-6 * 1e-5, 6 * 1e-5])
ax2.set_yticks(np.arange(-6 * 1e-5, 8 * 1e-5, 2 * 1e-5))
ax2.tick_params(axis='x', labelsize=fontsize_tick)
ax2.tick_params(axis='y', labelsize=fontsize_tick)
ax2.legend(fontsize=fontsize_legend)

ax3.plot(freq_x, img_disp_x, color='k', linewidth=2, linestyle=':', label='FRF measured')
ax3.plot(freq_x, -1 * img_fitted_x, color='b', linewidth=1.5, label='FRF fitted')

ax3.set_xlabel(r'frequency (Hz)', fontsize=fontsize_axis)
ax3.set_ylabel(r'Imag FRF (m/N)', fontsize=fontsize_axis)
ax3.set_xlim([500, 5500])
ax3.set_xticks([1000, 2000, 3000, 4000, 5000])
ax3.set_ylim([-10 * 1e-5, 2 * 1e-5])
ax3.set_yticks(np.arange(-9 * 1e-5, 3 * 1e-5, 2 * 1e-5))
ax3.tick_params(axis='x', labelsize=fontsize_tick)
ax3.tick_params(axis='y', labelsize=fontsize_tick)
ax3.legend(fontsize=fontsize_legend)

ax4.plot(freq_x, img_disp_x, color='k', linewidth=2, linestyle=':', label='FRF measured')
ax4.plot(freq_x, -1 * img_fitted_x, color='b', linewidth=1.5, label='FRF fitted')

ax4.set_xlabel(r'frequency (Hz)', fontsize=fontsize_axis)
ax4.set_ylabel(r'Imag FRF (m/N)', fontsize=fontsize_axis)
ax4.set_xlim([500, 5500])
ax4.set_xticks([1000, 2000, 3000, 4000, 5000])
ax4.set_ylim([-10 * 1e-5, 2 * 1e-5])
ax4.set_yticks(np.arange(-9 * 1e-5, 3 * 1e-5, 2 * 1e-5))
ax4.tick_params(axis='x', labelsize=fontsize_tick)
ax4.tick_params(axis='y', labelsize=fontsize_tick)
ax4.legend(fontsize=fontsize_legend)

plt.subplots_adjust(wspace=0.15, hspace=0.12)
plt.savefig('../Figures/Fig4_FRFs.png', dpi=300, bbox_inches='tight')
plt.show()
