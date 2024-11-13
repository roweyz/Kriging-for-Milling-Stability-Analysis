import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import scipy.stats as stats
from scipy.interpolate import interp1d

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Calibri']

plt.rcParams['font.weight'] = 'normal'  # light normal heavy bold

SS_kriging = scipy.io.loadmat('../Data_Essential/SS_Kriging.mat')['SS']
AP_kriging = scipy.io.loadmat('../Data_Essential/AP_Kriging.mat')['AP']
EI_kriging = scipy.io.loadmat('../Data_Essential//EI_Kriging.mat')['lambda']
STD_kriging = scipy.io.loadmat('../Data_Essential//STD_Kriging.mat')['standard']

'''------------------------------confidence interval selection------------------------------'''

df = 500  # degree of freedom for t distribution
alpha = 0.15  # confidence level
t_upper_quantile = stats.t.ppf(1 - alpha, df)  # upper quantile

EI_kriging_lower = EI_kriging + t_upper_quantile * STD_kriging
EI_kriging_upper = EI_kriging - t_upper_quantile * STD_kriging

'''------------------------------confidence interval plot------------------------------'''
fig, ax = plt.subplots(figsize=(9, 6))

contour_SLD1 = ax.contour(SS_kriging, AP_kriging * 1000, EI_kriging, [1, 10], colors='r')
contour_SLD2 = ax.contour(SS_kriging, AP_kriging * 1000, EI_kriging_lower, [1, 10], colors='darkorange', alpha=0.5)
contour_SLD3 = ax.contour(SS_kriging, AP_kriging * 1000, EI_kriging_upper, [1, 10], colors='red', alpha=0.5)

'''------------------------------Cutting Test Result------------------------------'''
kriging_exp = np.array(pd.read_csv('../Data_Experiment/Cutting_Experiment.csv', sep=','))
Stable = []
Unstable = []
Margin = []
for i in range(kriging_exp.shape[0]):
    if int(kriging_exp[i, 2]) == 1:
        Unstable.append(np.array(kriging_exp[i]))
    elif int(kriging_exp[i, 2]) == 0:
        Stable.append(np.array(kriging_exp[i]))
    else:
        Margin.append(np.array(kriging_exp[i]))

Stable = np.array(Stable)
Unstable = np.array(Unstable)
Margin = np.array(Margin)

'''------------------------------Fill between------------------------------'''
'boundary Kstable'
for level, path_collection in zip(contour_SLD2.levels, contour_SLD2.collections):
    for path in path_collection.get_paths():
        boundary_Kstable = path.vertices
        x_points, y_points = boundary_Kstable[:, 0], boundary_Kstable[:, 1]

ss_boundary_Kstable = boundary_Kstable[:, 0]
dc_boundary_Kstable = boundary_Kstable[:, 1]

'boundary Kupper'
for level, path_collection in zip(contour_SLD3.levels, contour_SLD3.collections):
    for path in path_collection.get_paths():
        boundary_Kupper = path.vertices
        x_points, y_points = boundary_Kupper[:, 0], boundary_Kupper[:, 1]

ss_boundary_Kupper = boundary_Kupper[:, 0]
dc_boundary_Kupper = boundary_Kupper[:, 1]

ssinterval = np.arange(0.85, 0.45, -0.01)
ssinterval = np.append(ssinterval, 0.45)
ssinterval_fillbetween = ssinterval * 10000
boundary_Kstable_fillbetween = np.zeros([len(ssinterval), 2])
boundary_Kupper_fillbetween = np.zeros([len(ssinterval), 2])

for i in range(len(ssinterval)):
    temp = ssinterval[i]
    j = 0
    while j < len(boundary_Kstable):
        if abs(boundary_Kstable[j, 0] - temp) < 1e-5:
            boundary_Kstable_fillbetween[i, 0] = boundary_Kstable[j, 0]
            boundary_Kstable_fillbetween[i, 1] = boundary_Kstable[j, 1]
            break
        else:
            j = j + 1
    l = 0
    while l < len(boundary_Kupper):
        if abs(boundary_Kupper[l, 0] - temp) < 1e-5:
            boundary_Kupper_fillbetween[i, 0] = boundary_Kupper[l, 0]
            boundary_Kupper_fillbetween[i, 1] = boundary_Kupper[l, 1]
            break
        else:
            l = l + 1

dc_boundary_Kstable_fillbetween = boundary_Kstable_fillbetween[:, 1]
dc_boundary_Kupper_fillbetween = boundary_Kupper_fillbetween[:, 1]

'''------------------------------labels plotting-----------------------------'''
'SLD labels'
plt.plot([5000, 5000], [0.1, 0.1], 'r', label='Our method')
plt.plot([5000, 5000], [0.1, 0.1], 'darkorange', alpha=0.5, label='lower boundary')
plt.plot([5000, 5000], [0.1, 0.1], 'red', alpha=0.5, label='upper boundary')

'Exp labels'
plt.scatter(Stable[:, 0], Stable[:, 1], c='k', marker='o', s=50, label='Stable')
plt.scatter(Unstable[:, 0], Unstable[:, 1], c='k', marker='x', s=50, label='Unstable')
plt.scatter(Margin[:, 0], Margin[:, 1], c='w', marker='^', edgecolors='k', s=50, label='Margin')

'Fillbetween labels'
plt.fill_between(ss_boundary_Kstable, 0, dc_boundary_Kstable, facecolor='darkorange', alpha=0.15,
                 label='robust stable area')
interp_func = interp1d(ss_boundary_Kstable, dc_boundary_Kstable, bounds_error=False, fill_value="extrapolate")
dc_boundary_Kstable_interpolated = interp_func(ss_boundary_Kupper)
plt.fill_between(ss_boundary_Kupper, dc_boundary_Kstable_interpolated, dc_boundary_Kupper, facecolor='red', alpha=0.15,
                 label='confidence interval({:.0f}%)'.format((1 - alpha) * 100))

'Improvements labels'
'85%'
Improvements_85 = np.array(
    [(4750, 1.25), (5500, 1), (5750, 1.25), (6750, 1,), (7000, 1.25), (7500, 1.25), (7750, 1), (8000, 0.75),
     (8250, 0.75)])
if alpha == 0.15:
    plt.scatter(Improvements_85[:, 0], Improvements_85[:, 1], linewidths=1.5, edgecolors='b', c='None', marker='s',
                s=140, label='Improvements')

'95%'

Improvements_95 = np.vstack((Improvements_85, np.array([(5250, 1), (6000, 1.25)])))
if alpha == 0.05:
    plt.scatter(Improvements_95[:, 0], Improvements_95[:, 1], linewidths=1.5, edgecolors='b', c='None', marker='s',
                s=140, label='Improvements')

'99%'
Improvements_99 = np.vstack((Improvements_95, np.array([(5000, 1.25), (6250, 1), (6500, 1)])))
if alpha == 0.01:
    plt.scatter(Improvements_99[:, 0], Improvements_99[:, 1], linewidths=1.5, edgecolors='b', c='None', marker='s',
                s=140, label='Improvements')

'''------------------------------Figures setting-----------------------------'''

fontsize_axis = 20
fontsize_legend = 13
fontsize_tick = 15
fontsize_title = 20

ax.set_xlabel('Spindle speed (rev/min)', fontsize=fontsize_axis)
ax.set_ylabel('Axial depth (mm)', fontsize=fontsize_axis)

ax.set_xlim([4500, 8500])
ax.set_xticks(np.arange(4500, 9000, 500))
ax.set_ylim([0, 3])
ax.set_yticks(np.arange(0, 3.5, 0.5))
ax.tick_params(axis='x', labelsize=fontsize_tick)
ax.tick_params(axis='y', labelsize=fontsize_tick)

plt.tight_layout()
ax = plt.gca()
plt.legend(loc='upper right', fontsize=fontsize_legend, ncol=3)

# ax.set_title(r'$\gamma$ = {:.2f}, confidence interval = {:.0f}%'.format(alpha, (1 - alpha) * 100),
#              fontsize=fontsize_title)

filename = '../Figures/Fig7_{:.0f}%CI.png'.format((1 - alpha) * 100)
plt.savefig(filename, dpi=300, bbox_inches='tight')

plt.show()
