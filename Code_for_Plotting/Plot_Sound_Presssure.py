# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Calibri']

plt.rcParams['font.weight'] = 'normal'  # light normal heavy bold

'''------------------------------Sound data loading------------------------------'''
signal_stable = pd.read_excel('../Data_Experiment/VOICE_SS7000_Depth1.0_stable.xlsx', sheet_name='Data1')
signal_stable = signal_stable.values[1:, :]

x_stable = np.array(signal_stable[:, 0], dtype=float)
y_stable = np.array(signal_stable[:, 1], dtype=float)

signal_chatter = pd.read_excel('../Data_Experiment/VOICE_SS7000_Depth2.0_chatter.xlsx', sheet_name='Data1')
signal_chatter = signal_chatter.values[1:, :]

x_chatter = np.array(signal_chatter[:, 0], dtype=float)
y_chatter = np.array(signal_chatter[:, 1], dtype=float)

'''------------------------------FFT-----------------------------'''
fs = 8192
T = 1 / fs

N = 200000

yf_stable = np.fft.fft(y_stable)
xf_stable = np.fft.fftfreq(N, T)[:N // 2]

yf_chatter = np.fft.fft(y_chatter)
xf_chatter = np.fft.fftfreq(N, T)[:N // 2]

'''------------------------------Sound Pressure plotting-----------------------------'''
fontsize_axis = 25
fontsize_title = 25
fontsize_tick = 20
x_range = (0, 4000)
small_grid_width = 7000 / 60

num_x = int((x_range[1] - x_range[0]) / small_grid_width)

fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(7, 6))

for i in range(num_x + 1):
    plt.axvline(x=x_range[0] + i * small_grid_width, color='grey', linestyle='--', linewidth=0.7)

ax1.plot(x_stable, y_stable, color='b', linewidth=0.5)
ax1.set_title(r'Point A: $\omega$=7000 rpm, $a_p$= 1 mm', fontsize=fontsize_title)
ax1.set_xlabel(r'time (s)', fontsize=fontsize_axis)
ax1.set_ylabel(r'Sound pressure (Pa)', fontsize=fontsize_axis)
ax1.set_xlim([10, 25])
ax1.set_xticks(np.arange(10, 30, 5))
ax1.set_ylim([-8, 8])
ax1.set_yticks(np.arange(-8, 10, 4))
ax1.tick_params(axis='x', labelsize=fontsize_tick)
ax1.tick_params(axis='y', labelsize=fontsize_tick)

ax2.plot(xf_stable, 2.0 / N * np.abs(yf_stable[0:N // 2]), color='r', linewidth=1.5)

for i in range(num_x + 1):
    ax2.axvline(x=x_range[0] + i * small_grid_width, color='grey', linestyle='--', linewidth=0.7)

ax2.set_xlabel(r'frequency (Hz)', fontsize=fontsize_axis)
ax2.set_ylabel(r'FFT (Pa)', fontsize=fontsize_axis)
ax2.set_xlim([0, 3500])
ax2.set_xticks(np.arange(0, 4000, 1000))
ax2.set_ylim([0, 0.4])
ax2.set_yticks(np.arange(0, 0.5, 0.1))
ax2.tick_params(axis='x', labelsize=fontsize_tick)
ax2.tick_params(axis='y', labelsize=fontsize_tick)

plt.subplots_adjust(hspace=0.4)
plt.savefig('../Figures/Fig6_stable.png', dpi=300, bbox_inches='tight')

fig2, (ax3, ax4) = plt.subplots(nrows=2, ncols=1, figsize=(7, 6))

ax3.plot(x_chatter, y_chatter, color='b', linewidth=0.5)
ax3.set_title(r'Point B: $\omega$=7000 rpm, $a_p$= 2 mm', fontsize=fontsize_title)
ax3.set_xlabel(r'time (s)', fontsize=fontsize_axis)
ax3.set_ylabel(r'Sound pressure (Pa)', fontsize=fontsize_axis)
ax3.set_xlim([10, 25])
ax3.set_xticks(np.arange(10, 30, 5))
ax3.set_ylim([-8, 8])
ax3.set_yticks(np.arange(-8, 10, 4))
ax3.tick_params(axis='x', labelsize=fontsize_tick)
ax3.tick_params(axis='y', labelsize=fontsize_tick)

ax4.plot(xf_chatter, 2.0 / N * np.abs(yf_chatter[0:N // 2]), color='r', linewidth=1.5)

for i in range(num_x + 1):
    ax4.axvline(x=x_range[0] + i * small_grid_width, color='grey', linestyle='--', linewidth=0.7)

ax4.set_xlabel(r'frequency (Hz)', fontsize=fontsize_axis)
ax4.set_ylabel(r'FFT (Pa)', fontsize=fontsize_axis)
ax4.set_xlim([0, 3500])
ax4.set_xticks(np.arange(0, 4000, 1000))
ax4.set_ylim([0, 0.4])
ax4.set_yticks(np.arange(0, 0.5, 0.1))
ax4.tick_params(axis='x', labelsize=fontsize_tick)
ax4.tick_params(axis='y', labelsize=fontsize_tick)

plt.subplots_adjust(hspace=0.4)
plt.savefig('../Figures/Fig6_chatter.png', dpi=300, bbox_inches='tight')
plt.show()

# filter_chatter = 0.3
# yf_chatter_windowed = yf_chatter[0:N // 2]
# greater_than_filter = np.abs(yf_chatter_windowed) > filter_chatter * N / 2
#
# values = yf_chatter_windowed[greater_than_filter]
# indices = np.where(greater_than_filter)[0]
#
# chatter_freq = xf_chatter[indices]
# print('chatter frequency: ', chatter_freq)
