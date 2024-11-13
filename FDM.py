# -*- coding: utf-8 -*-

import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm


def FDM(purpose, w_x_in, w_y_in, c_x_in, c_y_in, g_x_in, g_y_in, Kt_in, Kn_in):  # Input: Model parameters

    """------------------------------FDM input parameters------------------------------"""
    '''Cutting force coefficients'''
    Kt = Kt_in * 1e8  # Tangential cutting force coefficient (N/m^2)
    Kn = Kn_in * 1e8  # Normal cutting force coefficient (N/m^2)

    '''Dynamical parameters'''
    w_x = w_x_in * 2 * np.pi  # The natural frequency in the X direction, the angular frequency, (rad/s)
    w_y = w_y_in * 2 * np.pi  # The natural frequency in the Y direction, the angular frequency, (rad/s)

    c_x = c_x_in  # The damping ratio in the X direction (1)
    c_y = c_y_in  # The damping ratio in the Y direction (1)

    k_x = g_x_in * 1e6  # Modal stiffness in the X direction (N/m)
    k_y = g_y_in * 1e6  # Modal stiffness in the Y direction (N/m)

    m_x = k_x / (w_x ** 2)  # X direction modal mass (kg), m_t = stiff/(w^2)
    m_y = k_y / (w_y ** 2)  # Y direction modal mass (kg) m_t = stiff/(w^2)

    """------------------------------FDM hidden parameters------------------------------"""
    '''Other cutting information'''
    aD = 0.5  # The ratio of cutting width to tool diameter, '1' denotes full knife
    up_or_down = -1  # '1' denotes up milling, '-1' denote down milling
    if up_or_down == 1:
        phi_start = 0  # Angle of entry
        phi_exit = math.acos(1 - 2 * aD)  # Angle of exit
    if up_or_down == -1:
        phi_start = math.acos(2 * aD - 1)  # Angle of entry
        phi_exit = np.pi  # Angle of exit

    '''Discrete parameter setting'''
    m = 20  # Number of discrete units per cutting cycle
    D = np.eye(2 * m + 4, k=-2)  # The simplified dimension of the 2-DOF transfer matrix is [2*m+4, 2*m+4]
    D[:, 0:4] = 0
    D[4, 0] = 1
    D[5, 1] = 1

    """------------------------------Discretization of model parameter domain------------------------------"""
    if purpose == 'SampleForCase4.1':
        N = 4  # Blade number
        ap_step = 30  # Discrete steps of cutting depth 26*65
        ss_step = 40  # Speed discrete steps
        ap_start = 0e-3  # Start point of cutting depth (m)
        ap_end = 3e-3  # End point of cutting depth (m)
        ss_start = 4.5e3  # Starting point of rotation (rpm)
        ss_end = 8.5e3  # End point of cutting rotation (rpm)
    if purpose == 'SampleForCase4.2':
        N = 2  # Blade number
        ap_step = 26
        ss_step = 84
        ap_start = 0e-3
        ap_end = 4e-3
        ss_start = 4.8e3
        ss_end = 13.2e3

    """------------------------------FDM calculation------------------------------"""
    '''A two-degree-of-freedom modal parameter matrix is constructed'''  # Mq.. + Cq. + Kq = F
    M = np.matrix([[m_x, 0],
                   [0, m_y]])
    C = np.matrix([[2 * m_x * c_x * w_x, 0],
                   [0, 2 * m_y * c_y * w_y]])
    K = np.matrix([[m_x * w_x ** 2, 0],
                   [0, m_y * w_y ** 2]])

    '''Discrete cutting force'''
    h_xx = np.zeros(m + 1)
    h_yy = np.zeros(m + 1)
    h_yx = np.zeros(m + 1)
    h_xy = np.zeros(m + 1)
    for i in range(m + 1):
        delta_t = 2 * np.pi / N / m
        for j in range(N):
            phi = i * delta_t + j * 2 * np.pi / N
            if (phi >= phi_start) * (phi <= phi_exit):
                g = 1  # In the cutting zone
            else:
                g = 0  # Not in the cutting zone
            h_xx[i] = h_xx[i] + g * (Kt * math.cos(phi) + Kn * math.sin(phi)) * math.sin(phi)
            h_xy[i] = h_xy[i] + g * (Kt * math.cos(phi) + Kn * math.sin(phi)) * math.cos(phi)
            h_yx[i] = h_yx[i] + g * (-Kt * math.sin(phi) + Kn * math.cos(phi)) * math.sin(phi)
            h_yy[i] = h_yy[i] + g * (-Kt * math.sin(phi) + Kn * math.cos(phi)) * math.cos(phi)

    '''Construct the coefficient matrix A for the equation of state'''

    A0_11 = -M.I @ C / 2
    A0_12 = M.I
    A0_21 = C @ M.I @ C / 4 - K
    A0_22 = -C @ M.I / 2
    A0_11_12 = np.hstack((A0_11, A0_12))
    A0_21_22 = np.hstack((A0_21, A0_22))
    A0 = np.vstack((A0_11_12, A0_21_22))
    I = np.eye(len(A0))
    invA0 = A0.I

    data = []
    matrix_spindle_speed = np.zeros((ss_step + 1, ap_step + 1))
    matrix_axis_depth = np.zeros((ss_step + 1, ap_step + 1))
    matrix_eigenvalues = np.zeros((ss_step + 1, ap_step + 1))
    for x in range(ss_step + 1):
        ss = ss_start + x * (ss_end - ss_start) / ss_step
        tau = 60 / ss / N
        dt = tau / m

        Fi0 = expm(A0 * dt)
        Fi1 = invA0 @ (Fi0 - I)
        Fi2 = invA0 @ (Fi0 * dt - Fi1)
        Fi3 = invA0 @ (Fi0 * dt * dt - 2 * Fi2)

        for y in range(ap_step + 1):

            ap = ap_start + y * (ap_end - ap_start) / ap_step
            Fi = np.eye(2 * m + 4)
            for i in range(m):
                A0k = np.matrix([[0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [-ap * h_xx[i + 1], -ap * h_xy[i + 1], 0, 0],
                                 [-ap * h_yx[i + 1], -ap * h_yy[i + 1], 0, 0]])
                A1k = np.matrix([[0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [ap * (h_xx[i + 1] - h_xx[i]) / dt, ap * (h_xy[i + 1] - h_xy[i]) / dt, 0, 0],
                                 [ap * (h_yx[i + 1] - h_yx[i]) / dt, ap * (h_yy[i + 1] - h_yy[i]) / dt, 0, 0]])
                F01 = Fi2 @ A0k / dt + Fi3 @ A1k / dt
                Fkp1 = (Fi1 - Fi2 / dt) @ A0k + (Fi2 - Fi3 / dt) @ A1k
                invOfImFkp1 = (I - Fkp1).I
                D[0:4, 0:4] = invOfImFkp1 @ (Fi0 + F01)
                D[0:4, 2 * m: 2 * m + 2] = -invOfImFkp1 @ Fkp1[0:4, 0:2]
                D[0:4, 2 * m + 2: 2 * m + 4] = -invOfImFkp1 @ F01[0:4, 0:2]
                Fi = D @ Fi

            eigenvalues, eigenvectors = np.linalg.eig(Fi)
            matrix_spindle_speed[x, y] = ss
            matrix_axis_depth[x, y] = ap
            matrix_eigenvalues[x, y] = max(abs(eigenvalues))
            data.append([ss, ap, max(abs(eigenvalues))])

    return matrix_spindle_speed, matrix_axis_depth, matrix_eigenvalues, np.matrix(data)


if __name__ == '__main__':
    # (wn; k; c; Kc)
    # (Hz; 1e8 * N/m^2; 1; 1e6 * N/m)
    '''Case selection'''
    case = 'Case4.1'
    purpose = 'SampleFor{}'.format(case)

    if purpose == 'SampleForCase4.1':
        w_x, w_y = 1956.5, 1954.5
        c_x, c_y = 0.0227447, 0.0227680
        k_x, k_y = 9.438878, 10.17640
        Kt, Kn = 11.3088, 4.1039
    if purpose == 'SampleForCase4.2':
        w_x, w_y = 782.7, 752.8,
        c_x, c_y = 0.0184, 0.0186,
        k_x, k_y = 6.5616, 4.8852,
        Kt, Kn = 10.95, 1.76

    fig, ax = plt.subplots(figsize=(9, 6))

    plt.rcParams['font.size'] = 15
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Calibri']

    matrix_SS, matrix_AP, matrix_EI, _ = FDM(purpose, w_x, w_y, c_x, c_y, k_x, k_y, Kt, Kn)

    ax.contour(matrix_SS, matrix_AP * 1000, matrix_EI, [1, 10], colors='k')
    plt.plot([5000, 5000], [0.0, 0.0], 'k', label='FDM')

    fontweight = 'normal'
    fontsize = 15

    plt.yticks(fontsize=fontsize, fontweight=fontweight)
    plt.xticks(fontsize=fontsize, fontweight=fontweight)
    ax.set_xlabel('Spindle speed (rpm)', fontsize=fontsize, fontweight=fontweight)
    ax.set_ylabel('Axis depth (mm)', fontsize=fontsize, fontweight=fontweight)
    plt.legend(loc='upper right', fontsize=fontsize)
    plt.tight_layout()
    plt.show()
