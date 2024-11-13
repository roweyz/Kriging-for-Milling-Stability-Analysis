# -*- coding: utf-8 -*-
import os
import time

import numpy as np
import scipy.io as sio

import FDM as MF

'data name check'
# LH.npy
data = np.matrix(np.load('./Data_Essential/LH.npy'))

'''------------------------------sample for kriging----------------------------'''
# sample size check: in accordance with step0 sample strategy n_sample
samplesize = 500
filename = 'LH'

# discretization check: in accordance with FDM
ap_step = 30 + 1
ss_step = 40 + 1
gridNumber = ap_step * ss_step

if not os.path.exists('./Data_Generated/TrainingData_' + filename):
    os.makedirs('./Data_Generated/TrainingData_' + filename)

for i in range(samplesize):
    localtime = time.asctime(time.localtime(time.time()))
    print('Index: ', i, ' in ', samplesize, ', Time', localtime)
    _, _, _, spectral_radius = MF.FDM(
        'SampleForCase4.1',
        data[i, 0],
        data[i, 1],
        data[i, 2],
        data[i, 3],
        data[i, 4],
        data[i, 5],
        data[i, 6],
        data[i, 7])

    extra_feature = np.repeat(data[i], len(spectral_radius), axis=0)
    data_temp = np.hstack((extra_feature, spectral_radius))
    if i == 0:
        data_save = data_temp
    if i > 0:
        data_save = np.vstack((data_temp, data_save))
    # sio.savemat('./Data_Generated/TrainingData_' + filename + '/' + str(i) + '_spectral_radius.mat',
    #             {'data': data_temp})

sio.savemat('./Data_Generated/TrainingData_' + filename + '/' + str(samplesize) + '_spectral_radius_for_Kriging.mat',
            {'data': data_save})

alldata = data_save
# alldata = sio.loadmat('./Data_Generated/TrainingData_'+ filename + '/' + str(samplesize) + '_spectral_radius_for_Kriging.mat')['data']

for i in range(0, gridNumber):
    temp = []
    for j in range(0, samplesize):
        if j == 0:
            temp = alldata[gridNumber * j + i, :]
        else:
            temp = np.vstack((temp, alldata[gridNumber * j + i, :]))

    sio.savemat('./Data_Generated/TrainingData_' + filename + '/KrigingData' + str(i) + '.mat', {'data': temp})

print("Step1 Sample and Sort for Kriging is over")
