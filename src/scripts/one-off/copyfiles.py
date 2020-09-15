import pandas as pd
import feather
import numpy as np
import os
import sys
import re
import math
import shutil
from scipy import interpolate
import pdb
###################################################################################
# (Jared) Mar 2020 - Read/format data for many 2700+ lakes with new data release
###################################################################################
first_time = True
directory = '../../data/raw/figure3' #unprocessed data directory
# tgt_directory = '../../data/processed/sixety_nine_lakes_Jan2019'
n_features = 7
# obs_df = pd.read_feather("../../data/raw/additional_lakes/temperature_obs.feather")
# obs_df = pd.read_feather("../../data/raw/usgs_zips/obs/03_temperature_observations.csv")
# ids = np.unique(obs_df['nhdhr_id'].values)
ids = pd.read_csv('../../../metadata/pball_site_ids.csv', header=None)
ids = ids[0].values

# matches = [re.search('nhdhr_(.*)', i) for i in ids]
# ids = [m.group(1) for m in matches]
enable_glm = True
ct = 0

#feature statistics hardcoded
mean_feats = [5.39943134, 1.66547735e2, 2.91895336e2, 6.76191477, 7.37264141e1, 4.79885221e0, 1.83438590e-3, 2.29069199e-3]
std_feats = [3.25563170, 8.53092316e1, 6.09606032e1, 1.27855354e1, 1.29500991e1, 1.69787946, 5.58071004e-3, 1.27507001e-2]

for nid in ids: #for each new additional lake
	name = nid
	shutil.copyfile("../../../data/processed/lake_data/"+name+"/processed_features.npy", "../../../data/processed/lake_data/"+name+"/processed_features_pbAllNorm")