# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt

"""------------------------------force signal loading---------------------------"""

F_total = np.array(pd.read_csv('../Data_Experiment/Cutting_Force_Measurement.csv', sep=','))
Time_total = F_total[:, 0]
Fx_total = F_total[:, 1]
Fy_total = F_total[:, 2]
'''F_total visualization'''
fig0, axs0 = plt.subplots(2)
fig0.suptitle('F_sampled')
axs0[0].plot(Time_total, Fx_total, color='r', linewidth=0.5, label='Fx_sample')
axs0[0].set_title('x direction ')
axs0[0].set_xlabel('time (s)')
axs0[0].set_ylabel('Force (N)')

axs0[1].plot(Time_total, Fy_total, color='b', linewidth=0.5)
axs0[1].set_title('y direction ')
axs0[1].set_xlabel('time (s)')
axs0[1].set_ylabel('Force (N)')
plt.tight_layout()

"""------------------------------low pass filter---------------------------"""
ss = 5000  # (rev/min = 1/60 rev/s)
Fs = 10 * 1000  # (0.001s)
T = 1 / Fs


def butter_lowpass(cutoff, fs, order):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


order = 5
cutoff = 1000
filtered_signal_x = butter_lowpass_filter(Fx_total, cutoff, Fs, order)
filtered_signal_y = butter_lowpass_filter(Fy_total, cutoff, Fs, order)

'''F_filtered visualization'''
fig1, axs1 = plt.subplots(2, figsize=(8, 6))
fig1.suptitle('F_filtered')
axs1[0].plot(Time_total, filtered_signal_x, color='r', linewidth=0.5)
axs1[0].set_title('Fx_filtered')
axs1[0].set_xlabel('time (s)')
axs1[0].set_ylabel('Force (N)')

axs1[1].plot(Time_total, filtered_signal_y, color='b', linewidth=0.5)
axs1[1].set_title('Fy_filtered')
axs1[1].set_xlabel('time (s)')
axs1[1].set_ylabel('Force (N)')
plt.tight_layout()

"""------------------------------force signal FFT---------------------------"""
N_x = len(Fx_total)
X = fft(Fx_total) / N_x
frq_x = fftfreq(N_x, T)

N_y = len(Fy_total)
Y = fft(Fy_total) / N_y
frq_y = fftfreq(N_y, T)

fig2, axs2 = plt.subplots(2, figsize=(8, 6))
fig2.suptitle('FFT')
axs2[0].plot(frq_x[:N_x // 2], np.abs(X[:N_x // 2]), color='r', linewidth=1, label='FFT_Fx')
axs2[0].set_title('FFT of F_x')
axs2[0].set_xlabel('Frequency (Hz)')
axs2[0].set_ylabel('Amplitude')

axs2[1].plot(frq_y[:N_y // 2], np.abs(Y[:N_y // 2]), color='b', linewidth=1, label='FFT_Fy')
axs2[1].set_title('FFT of F_y')
axs2[1].set_xlabel('Frequency (Hz)')
axs2[1].set_ylabel('Amplitude')
axs2[1].set_ylim(0, 5)
plt.tight_layout()

"""------------------------------CFC calculation---------------------------"""
'n_exp signal separation'
# n_exp = 5
# spindle 1 rev -- 120 sample points
# len(Fx_total)//5 = 476999
# 476999 // 50 = 9539

'exp1'
'Fx'
index1 = slice(382000, 765000)
Time_exp1 = Time_total[index1]
Fx_exp1 = Fx_total[index1]
Fy_exp1 = Fy_total[index1]

indexI = slice(190000, 238500)
Fx_exp1_simulation = Fx_exp1[indexI]
Fy_exp1_simulation = Fy_exp1[indexI]
Time_exp1_simulation = Time_exp1[indexI]  # 48500/120 = 400 round

'exp2'
index2 = slice(1200000, 1440000)
Time_exp2 = Time_total[index2]
Fx_exp2 = Fx_total[index2]
Fy_exp2 = Fy_total[index2]

indexII = slice(96000, 144000)
Fx_exp2_simulation = Fx_exp2[indexII]
Fy_exp2_simulation = Fy_exp2[indexII]
Time_exp2_simulation = Time_exp2[indexII]

'exp3'
index3 = slice(1650000, 1800000)
Time_exp3 = Time_total[index3]
Fx_exp3 = Fx_total[index3]
Fy_exp3 = Fy_total[index3]

indexIII = slice(75000, 97500)
Fx_exp3_simulation = Fx_exp3[indexIII]
Fy_exp3_simulation = Fy_exp3[indexIII]
Time_exp3_simulation = Time_exp3[indexIII]

'exp4'
index4 = slice(1960000, 2050000)
Time_exp4 = Time_total[index4]
Fx_exp4 = Fx_total[index4]
Fy_exp4 = Fy_total[index4]

indexIV = slice(27000, 54000)
Fx_exp4_simulation = Fx_exp4[indexIV]
Fy_exp4_simulation = Fy_exp4[indexIV]
Time_exp4_simulation = Time_exp4[indexIV]

'exp5'
index5 = slice(2180000, 2240000)
Time_exp5 = Time_total[index5]
Fx_exp5 = Fx_total[index5]
Fy_exp5 = Fy_total[index5]

indexV = slice(15000, 50000)
Fx_exp5_simulation = Fx_exp5[indexV]
Fy_exp5_simulation = Fy_exp5[indexV]
Time_exp5_simulation = Time_exp5[indexV]

fig3, axs3 = plt.subplots(5, 2, figsize=(10, 15))

axs3[0, 0].plot(Time_exp1, Fx_exp1)
axs3[0, 0].set_title('Fx_exp1')
axs3[0, 1].plot(Time_exp1, Fy_exp1)
axs3[0, 1].set_title('Fy_exp1')
axs3[1, 0].plot(Time_exp2, Fx_exp2)
axs3[1, 0].set_title('Fx_exp2')
axs3[1, 1].plot(Time_exp2, Fy_exp2)
axs3[1, 1].set_title('Fy_exp2')
axs3[2, 0].plot(Time_exp3, Fx_exp3)
axs3[2, 0].set_title('Fx_exp3')
axs3[2, 1].plot(Time_exp3, Fy_exp3)
axs3[2, 1].set_title('Fy_exp3')
axs3[3, 0].plot(Time_exp4, Fx_exp4)
axs3[3, 0].set_title('Fx_exp4')
axs3[3, 1].plot(Time_exp4, Fy_exp4)
axs3[3, 1].set_title('Fy_exp4')
axs3[4, 0].plot(Time_exp5, Fx_exp5)
axs3[4, 0].set_title('Fx_exp5')
axs3[4, 1].plot(Time_exp5, Fy_exp5)
axs3[4, 1].set_title('Fy_exp5')
plt.subplots_adjust(hspace=0.5, wspace=0.2)

'Average cutting force calculation'
n_exp = 5
Fx_final = np.array([np.mean(Fx_exp1_simulation), np.mean(Fx_exp2_simulation),
                     np.mean(Fx_exp3_simulation), np.mean(Fx_exp4_simulation), np.mean(Fx_exp5_simulation)])

Fy_final = np.array([np.mean(Fy_exp1_simulation), np.mean(Fy_exp2_simulation),
                     np.mean(Fy_exp3_simulation), np.mean(Fy_exp4_simulation), np.mean(Fy_exp5_simulation)])

'Cutting force coefficients calculation (Pa = N/m^2)'
'Shearing(feed rate) + Ploughing(discarded)'
'Blade number'
N = 4

'Edge contact length (mm = 1e-3m)'
a = 1e-3  # depth of cut

'Spindle speed (rev/min)'
ss = 5000

'Feed rate (m/tooth)'  # mm/min = mm/tooth * ss * N
n_exp = 5
c_start = 0.016e-3
c_step = 0.016e-3
feed = np.zeros(n_exp)
for i in range(n_exp):
    feed[i] = c_start + c_step * i

'Least square fitting'
fit_coefficients_x = np.polyfit(feed, Fx_final, 1)
fit_function_x = np.poly1d(fit_coefficients_x)

fit_coefficients_y = np.polyfit(feed, Fy_final, 1)
fit_function_y = np.poly1d(fit_coefficients_y)

fig4, axs4 = plt.subplots(2, figsize=(8, 6))
fig4.suptitle('Force_Feed_Relationship')
axs4[0].scatter(feed, Fx_final, color='r', s=100, label='Fx Measurement')
axs4[0].plot(feed, fit_function_x(feed), 'k', linewidth=1, label='Fx Linear Fit')
axs4[0].set_title('Fx_Feed')
axs4[0].set_xlabel('feed (m/tooth)')
axs4[0].set_ylabel('Fx (N)')
axs4[0].legend()

axs4[1].scatter(feed, Fy_final, color='b', s=100, label='Fy Measurement')
axs4[1].plot(feed, fit_function_y(feed), 'k', linewidth=1, label='Fy Linear Fit')
axs4[1].set_title('Fy_Feed')
axs4[1].set_xlabel('feed (m/tooth)')
axs4[1].set_ylabel('Fy (N)')
axs4[1].legend()
plt.tight_layout()


'slope = N * a / 4'
Ktc = (4 * fit_coefficients_y[0]) / (N * a)
Krc = (4 * fit_coefficients_x[0]) / (N * a)

Kte = (np.pi * fit_coefficients_y[1]) / (N * a)
Kre = (np.pi * fit_coefficients_x[1]) / (N * a)

#plt.show()
print("Case 4.1:")
print("Measured Ktc (MPa = 1e6 N/m^2)", Ktc / 1e6)
print("Measured Krc (MPa = 1e6 N/m^2)", Krc / 1e6)