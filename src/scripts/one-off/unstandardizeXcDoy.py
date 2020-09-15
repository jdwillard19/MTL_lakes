import os.path
import numpy as np
import matplotlib.pyplot as plt
import sys
import csv
import pandas as pd

sys.path.append('../../data')
sys.path.append('/home/invyz/workspace/Research/lake_monitoring/src/data')

from rw_data import readMatData, saveMatData
#######################################################
# This script reverses the standardization done on the 
# intepolated
# physical drivers in 'Xc_doy_int' to create a new data object
# 'Xc_doy_int_unstandardized'
########################################################
mats = readMatData(os.path.abspath('../../../data/processed/mendota_sampled.mat'))
mat = readMatData(os.path.abspath('../../../data/processed/mendota.mat'))

means = mat['meanX_doy']
stds = mat['stdX_doy']
std_XcDoy = mats['Xc_doy_int']
unstd_XcDoy = std_XcDoy

for i in range(0,std_XcDoy.shape[0]):
    # print("old")
    # print(unstd_XcDoy[i,:])
    unstd_XcDoy[i,:] = means + np.multiply(stds, std_XcDoy[i,:])
    # print("new")
    # print(unstd_XcDoy[i,:])
    
mats['Xc_doy_int_unstandardized'] = unstd_XcDoy
saveMatData(mats, os.path.abspath('../../../data/processed/mendota_sampled.mat'))