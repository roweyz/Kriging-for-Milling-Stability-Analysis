import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Calibri']

plt.rcParams['font.weight'] = 'normal'  # light normal heavy bold

'''------------------------------OMA data-----------------------------'''
ss = np.linspace(2000, 10000, 100)

ss_wnx = np.array([4000, 5000, 7000, 9000])
wnx = np.array([1779.2, 1853.2, 1665.3, 1524.9])

ss_wny = np.array([4000, 5000, 7000, 8500])
wny = np.array([2018.9, 1947.2, 1992.9, 1846.4])

ss_zetax = np.array([5000, 6000, 7500, 9000])
zetax = np.array([0.0276, 0.0283, 0.0341, 0.0336]) * 100

ss_zetay = np.array([4000, 5000, 6000, 8000])
zetay = np.array([0.0215, 0.0236, 0.0225, 0.0246]) * 100


def quadratic_function(x, coeff):
    return coeff[0] * x ** 2 + coeff[1] * x + coeff[2]


'''------------------------------quadratic fitting-----------------------------'''
'wnx'
coefficients_wnx = np.polyfit(ss_wnx, wnx, 2)
fitting_wnx = quadratic_function(ss, coefficients_wnx)

'wny'
coefficients_wny = np.polyfit(ss_wny, wny, 2)
fitting_wny = quadratic_function(ss, coefficients_wny)

'zetax'
coefficients_zetax = np.polyfit(ss_zetax, zetax, 2)
fitting_zetax = quadratic_function(ss, coefficients_zetax)

'zetay'
coefficients_zetay = np.polyfit(ss_zetay, zetay, 2)
fitting_zetay = quadratic_function(ss, coefficients_zetay)

'''------------------------------trend plotting-----------------------------'''

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, figsize=(18, 12))

datasize = 100
linewidth = 2

fontsize_axis = 25
fontsize_title = 25
fontsize_legend = 18
fontsize_tick = 20

fontsize_text = 35

ax1.plot(ss, fitting_wnx, color='b', linewidth=linewidth, linestyle='--', label='function fitted')
ax1.scatter(ss_wnx, wnx, s=datasize, color='k', label='parameter measured')
ax1.set_title(r'$\omega_{n,x}$', pad=15, fontsize=fontsize_title)
ax1.set_ylabel(r'$\omega_{n,x}$ (Hz)', fontsize=fontsize_axis)
ax1.set_xlim([3500, 9500])
ax1.set_xticks([4000, 5000, 6000, 7000, 8000, 9000])
ax1.set_ylim([1100, 2300])
ax1.set_yticks([1250, 1500, 1750, 2000, 2250])
ax1.legend(loc='upper right', fontsize=fontsize_legend)
ax1.tick_params(axis='x', labelsize=fontsize_tick)
ax1.tick_params(axis='y', labelsize=fontsize_tick)

formula = r'$\omega_{n,x} = 1956.5$ Hz'
ax1.text(0.4, 0.2, formula, transform=ax1.transAxes, horizontalalignment='center', verticalalignment='center',
         fontsize=fontsize_text, color='black')

ax2.plot(ss, fitting_wny, color='b', linewidth=linewidth, linestyle='--', label='function fitted')
ax2.scatter(ss_wny, wny, s=datasize, color='k', label='parameter measured')
ax2.set_title(r'$\omega_{n,y}$', pad=15, fontsize=fontsize_title)
ax2.set_ylabel(r'$\omega_{n,y}$ (Hz)', fontsize=fontsize_axis)
ax2.set_xlim([3500, 9500])
ax2.set_xticks([4000, 5000, 6000, 7000, 8000, 9000])
ax2.set_ylim([1200, 2400])
ax2.set_yticks([1250, 1500, 1750, 2000, 2250])
ax2.legend(loc='upper right', fontsize=fontsize_legend)
ax2.tick_params(axis='x', labelsize=fontsize_tick)
ax2.tick_params(axis='y', labelsize=fontsize_tick)

formula = r'$\omega_{n,y} = 1954.5$ Hz'
ax2.text(0.4, 0.2, formula, transform=ax2.transAxes, horizontalalignment='center', verticalalignment='center',
         fontsize=fontsize_text, color='black')

ax3.plot(ss, fitting_zetax, color='r', linewidth=linewidth, linestyle='--', label='function fitted')
ax3.scatter(ss_zetax, zetax, s=datasize, color='k', label='parameter measured')
ax3.set_title(r'$\zeta_{x}$', pad=15, fontsize=fontsize_title)
ax3.set_xlabel('Spindle speed (rev/min)', fontsize=fontsize_axis)
ax3.set_ylabel(r'$\zeta_{x}$ (%)', fontsize=fontsize_axis)
ax3.set_xlim([3500, 9500])
ax3.set_xticks([4000, 5000, 6000, 7000, 8000, 9000])
ax3.set_ylim([1, 5])
ax3.set_yticks([1, 2, 3, 4, 5])
ax3.legend(loc='upper left', fontsize=fontsize_legend)
ax3.tick_params(axis='x', labelsize=fontsize_tick)
ax3.tick_params(axis='y', labelsize=fontsize_tick)

formula = r'$\zeta_{x} = 2.27$ %'
ax3.text(0.75, 0.2, formula, transform=ax3.transAxes, horizontalalignment='center', verticalalignment='center',
         fontsize=fontsize_text, color='black')

ax4.plot(ss, fitting_zetay, color='r', linewidth=linewidth, linestyle='--', label='function fitted')
ax4.scatter(ss_zetay, zetay, s=datasize, color='k', label='parameter measured')
ax4.set_title(r'$\zeta_{y}$', pad=15, fontsize=fontsize_title)
ax4.set_xlabel('Spindle speed (rev/min)', fontsize=fontsize_axis)
ax4.set_ylabel(r'$\zeta_{y}$ (%)', fontsize=fontsize_axis)
ax4.set_xlim([3500, 9500])
ax4.set_xticks([4000, 5000, 6000, 7000, 8000, 9000])
ax4.set_ylim([0, 4])
ax4.set_yticks([0, 1, 2, 3, 4])
ax4.legend(loc='upper left', fontsize=fontsize_legend)
ax4.tick_params(axis='x', labelsize=fontsize_tick)
ax4.tick_params(axis='y', labelsize=fontsize_tick)

formula = r'$\zeta_{y} = 2.28$ %'
ax4.text(0.75, 0.2, formula, transform=ax4.transAxes, horizontalalignment='center', verticalalignment='center',
         fontsize=fontsize_text, color='black')

plt.subplots_adjust(wspace=0.3, hspace=0.2)

plt.savefig('../Figures//Fig5_OMA_trend.png', dpi=300, bbox_inches='tight')
plt.show()
