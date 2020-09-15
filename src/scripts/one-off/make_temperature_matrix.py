import os.path
import numpy as np
import sys
sys.path.append('../../data')
sys.path.append('/home/invyz/workspace/Research/lake_monitoring/src/data')

from rw_data import readMatData, saveMatData
from data_operations import createTemperatureMatrix

###################################################################3
# this script was used to make a matrix with depth row and date columns 
# with the modeled temperature value (interpolated)
##################################################################

mendota = readMatData(os.path.abspath('../../../data/processed/mendota_sampled.mat'))
mat = mendota
u_depth_values = np.array(np.sort(list(set(mat['Depth'].flat))))
t_mat = createTemperatureMatrix(mat['Modeled_temp_int'], mat['udates'], u_depth_values, mat['datenums_int'], mat['Depth_int'])
mat['Depth_Time_Temp_int'] = t_mat
saveMatData(mat, os.path.abspath('../../../data/processed/mendota_sampled.mat'))