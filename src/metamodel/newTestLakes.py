import pandas as pd
import numpy as np
import pdb
import os
import sys
import re
#script to get new test lakes from previous test lakes but satisfy new criteria

trn_tst_f = pd.read_feather("../../data/processed/lake_splits/trn_tst2.feather")
# trn_tst_f.to_feather("../../data/processed/lake_splits/trn_tst2.feather")
toAdd = []
metadata = pd.read_feather("../../metadata/lake_metadata_wStats4.feather")

glm_results = pd.read_csv("./results/RMSE_transfer_glm_2.csv")
glm_old = pd.read_csv('./results/RMSE_transfer_glm.csv')


train_lakes = trn_tst_f[trn_tst_f['isTrain3_JR']]['site_id'].values
all_lakes = trn_tst_f['site_id'].values
train_lakes_old = trn_tst_f[trn_tst_f['isTrain2']]['id'].values
# test_lakes = trn_tst_f[~trn_tst_f['isTrain2']]['id'].values

# for i_d in trn_tst_f['id'].values:
# 	if np.isin(i_d, train_lakes):
# 		toAdd.append(True)
# 	else:
# 		toAdd.append(False)




all_obs = pd.read_csv("../../data/raw/usgs_zips/obs/03_temperature_observations.csv")
toAdd = []
for i, lake in enumerate(all_lakes):
	print("i, ", i, ": ", lake)

	# if trn_tst_f[trn_tst_f['site_id'] == lake]['isTrain2'].values[0]:
	# 	#skip if train lake
	# 	toAdd.append(False)
	if not trn_tst_f[trn_tst_f['site_id'] == lake]['isTrain2'].values[0]:
		toAdd.append(False)
	else:
		#if possible test lake test that stuff
		obs = all_obs[all_obs['site_id'] == 'nhdhr_'+lake]
		max_d = metadata[metadata['site_id'] == lake]['max_depth']
		n_depths = np.load("../../data/processed/lake_data/"+lake+"/train_b.npy").shape[0]

		u_dates = np.unique(obs['date'])
		qualify = False
		one_per_two = True

		for d in u_dates:
			obs_d = obs[obs['date'] == d]
			rnd = lambda x: np.round(x*2)/2 
			uniq_depths = rnd(np.unique(obs_d['depth']))

			if uniq_depths.shape[0] > 4:
				continue
			else:
				for low_bnd_ind in range(n_depths-5):
					low_bnd = low_bnd_ind / 2
					upper_bnd = low_bnd + 2
					in_bnd_depths = uniq_depths[(uniq_depths >= low_bnd) & (uniq_depths <= upper_bnd)]
					if len(in_bnd_depths) == 0:
						one_per_two = False
						break

		if not one_per_two:
			qualify = False

	if qualify:
		toAdd.append(True)
	else:
		toAdd.append(False)
	#With the threshold of 50 unique observation dates,
	#with at least 1 observation for every 2 meters -or- at least 5 total observations
	# n_depths = 
print(np.array(toAdd).sum())

pdb.set_trace()
tmp = 0