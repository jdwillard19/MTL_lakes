import os.path
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../data')
sys.path.append('/home/invyz/workspace/Research/lake_monitoring/src/data')

from rw_data import readMatData, saveMatData
from data_operations import createDensityDeltaMatrix, transformTempToDensity, createTemperatureMatrix

######################################################
# OBSOLETE
# this script was used to create delta(i,t) matrix within the mendota_sampled.mat
#, or the density difference between a depth at a certain time and the depth 
# above it at the same itme. Equation 3.13 in (Karpatne et al 2018). 
#***Moved to data_operations for generalized use***
###########################################################

mendota = readMatData(os.path.abspath('../../../data/processed/mendota_sampled.mat'))

mat = mendota

u_depth_values = np.array(np.sort(list(set(mat['Depth'].flat))))
dd_mat = createDensityDeltaMatrix(mat['Modeled_temp'], mat['Depth'], mat['udates'], mat['datenums'])
mat['Modeled_Density_Delta_int'] = dd_mat
saveMatData(mat, os.path.abspath('../../../data/processed/mendota_sampled.mat'))





