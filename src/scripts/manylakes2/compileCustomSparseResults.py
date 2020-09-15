import numpy as np
import pandas as pd
import pdb
import re
import os
import sys
import time

def mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation 
    """
    arr = np.ma.array(arr).compressed() # should be faster to not use masked arrays.
    med = np.median(arr)
    return np.median(np.abs(arr - med))



ids = pd.read_csv('../../../metadata/pball_site_ids.csv', header=None)
ids = ids[0].values
glm_all_f = pd.read_csv("../../../results/glm_transfer/RMSE_transfer_glm_pball.csv")
train_lakes = [re.search('nhdhr_(.*)', x).group(1) for x in np.unique(glm_all_f['target_id'].values)]
test_lakes = ids[~np.isin(ids, train_lakes)]

n_profiles = np.array([1,2,5,10,15,20,25,30,35,40,45,50])
mean_err_per_sparse = np.empty((test_lakes.shape[0], n_profiles.shape[0]))
std_err_per_sparse = np.empty((test_lakes.shape[0], n_profiles.shape[0]))
med_err_per_sparse = np.empty((test_lakes.shape[0], n_profiles.shape[0]))
mad_err_per_sparse = np.empty((test_lakes.shape[0], n_profiles.shape[0]))
uq_err_per_sparse = np.empty((test_lakes.shape[0], n_profiles.shape[0]))
lq_err_per_sparse = np.empty((test_lakes.shape[0], n_profiles.shape[0]))
df = pd.DataFrame()
df['site_id'] = test_lakes


for ct, site_id in enumerate(test_lakes):
	print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime("../../../results/single_lake_outputs/"+site_id+"/sparseModelResults_July2020.csv"))))
	results = pd.read_csv("../../../results/single_lake_outputs/"+site_id+"/sparseModelResults_July2020.csv")
	print("lake ", ct, "/",test_lakes.shape[0])
	for prof_ct, n_prof in enumerate(n_profiles):
		if results[results['n_profiles'] == n_prof]['rmse'].shape[0] > 0:
			mean_err_per_sparse[ct, prof_ct] = results[results['n_profiles'] == n_prof]['rmse'].values.mean()
			std_err_per_sparse[ct, prof_ct] = results[results['n_profiles'] == n_prof]['rmse'].values.std()
			mad_err_per_sparse[ct, prof_ct] = mad(results[results['n_profiles'] == n_prof]['rmse'].values)
			med_err_per_sparse[ct, prof_ct] = np.median(results[results['n_profiles'] == n_prof]['rmse'])
			# if n_prof == 10 or n_prof == 15:
			# 	pdb.set_trace()
			uq_err_per_sparse[ct, prof_ct] = np.quantile(results[results['n_profiles'] == n_prof]['rmse'],.25)
			lq_err_per_sparse[ct, prof_ct] = np.quantile(results[results['n_profiles'] == n_prof]['rmse'],.75)
			
		else:
			continue

for prof_ct, n_prof in enumerate(n_profiles):
	df[str(n_prof)+" obs median"] = mean_err_per_sparse[:, prof_ct]
	df[str(n_prof)+" obs mean"] = med_err_per_sparse[:, prof_ct]
	df[str(n_prof)+" obs std"] = std_err_per_sparse[:, prof_ct]
	df[str(n_prof)+" obs mad"] = mad_err_per_sparse[:, prof_ct]
	df[str(n_prof)+" obs lq"] = lq_err_per_sparse[:, prof_ct]
	df[str(n_prof)+" obs uq"] = uq_err_per_sparse[:, prof_ct]
pdb.set_trace()
df.to_feather("../../../results/sparseModelResults_July2020.feather")
