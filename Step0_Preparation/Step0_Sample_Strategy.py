import numpy as np
from scipy.stats import qmc, norm

'''------------------------------sample for kriging----------------------------'''

n_dim = 8
n_samples = 500

mu = [1.9565, 1.9545, 0.0227, 0.0228, 9.44, 10.18, 11.31, 4.10]
sigma = [0.06, 0.06, 0.003, 0.002, 0.3, 0.35, 0.45, 0.15]

matrix = np.array([mu for _ in range(n_samples)])
matrix[:, 0:2] = matrix[:, 0:2] * 1000
'LatinHypercube'
sampler1 = qmc.LatinHypercube(d=n_dim)
lhs_samples = sampler1.random(n=n_samples)

normal_samples1 = norm.ppf(lhs_samples)
data_LH = np.multiply(normal_samples1, sigma) + mu
data_LH[:, 0:2] = data_LH[:, 0:2] * 1000

'Sobol'
sampler2 = qmc.Sobol(d=n_dim)
sobol_samples = sampler2.random(n=n_samples)

normal_samples = norm.ppf(sobol_samples)
data_sobol = np.multiply(normal_samples, sigma) + mu
data_sobol[:, 0:2] = data_sobol[:, 0:2] * 1000

'''------------------------------sample quality check----------------------------'''

data_LH_max = np.zeros(shape=n_dim)
data_LH_min = np.zeros(shape=n_dim)

for i in range(data_LH.shape[1]):
    data_LH_max[i] = np.double(np.max(data_LH[:, i]))
    data_LH_min[i] = np.double(np.min(data_LH[:, i]))

discrepency_LH = matrix - data_LH.copy()

data_sobol_max = np.zeros(shape=n_dim)
data_sobol_min = np.zeros(shape=n_dim)

for i in range(data_sobol.shape[1]):
    data_sobol_max[i] = np.double(np.max(data_sobol[:, i]))
    data_sobol_min[i] = np.double(np.min(data_sobol[:, i]))

discrepency_sobol = matrix - data_sobol.copy()

'''------------------------------sample quality visualization----------------------------'''
print("sobol max:\n", np.round(data_sobol_max, decimals=6))
print("LH max:\n", np.round(data_LH_max, decimals=6))
print("sobol min:\n", np.round(data_sobol_min, decimals=6))
print("LH min:\n", np.round(data_LH_min, decimals=6))

print("\n")
print("sobol mean:\n", np.mean(data_sobol, axis=0))
print("LH mean:\n", np.mean(data_LH, axis=0))

print("\n")
print("sobol var:\n", np.var(data_sobol, axis=0))
print("LH var:\n", np.var(data_LH, axis=0))

'data save'
# directory = '../Data_Essential'
# file_name1 = 'sobol.npy'
# file_name2 = 'LH.npy'
#
# file_path1 = os.path.join(directory, file_name1)
# file_path2 = os.path.join(directory, file_name2)
# np.save(file_path1, data_sobol)
# np.save(file_path2, data_LH)
