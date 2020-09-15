import os.path
import numpy as np
import sys
import csv
import pandas as pd

sys.path.append('../../data')
sys.path.append('/home/invyz/workspace/Research/lake_monitoring/src/data')

from rw_data import readMatData, saveMatData

####################################################
# this script was used to take a hypsography data file and 
# save it with the lake dataset at every half meter depth
#######################################################
mat = readMatData(os.path.abspath('../../../data/processed/mendota_sampled.mat'))
depths = np.array(np.sort(list(set(mat['Depth'].flat))))
depth_areas = pd.read_csv('../../../data/raw/Mendota_hypsography.csv', header=0, index_col=0, squeeze=True).to_dict()
tmp = {}

total_area = 0
for key, val in depth_areas.items():
    total_area += val


for depth in depths:
    #find depth with area that is closest
    depth_w_area = min(list(depth_areas.keys()), key=lambda x:abs(x-depth))
    tmp[depth] = depth_areas[depth_w_area]
depth_areas = {}

for k, v in tmp.items():
    total_area += v

for k, v in tmp.items():
    depth_areas[k] = tmp[k] 

#print(tmp.values())
result = np.sort(-np.array([list(depth_areas.values())]))*-1

mat['sortedDepthArea'] = result
#create unique depth 
# u_depth_values = np.array(np.sort(list(set(mat['Depth'].flat))))
# mat['depth_area'] = u_depth_values
saveMatData(mat, os.path.abspath('../../../data/processed/mendota_sampled.mat'))