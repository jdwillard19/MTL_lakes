import pandas as pd 
import numpy as np 
import re
import os
import sys
import pdb


metadata = pd.read_feather("../../../metadata/lake_metadata_baseJune2020.feather")
glm_all_f = pd.read_csv("../../../results/glm_transfer/RMSE_transfer_glm_pball.csv")
train_lakes = [re.search('nhdhr_(.*)', x).group(1) for x in np.unique(glm_all_f['target_id'].values)]
all_sites = metadata['site_id'].values
test_lakes = all_sites[~np.isin(all_sites, train_lakes)]


ct = 0
old_rmse = np.empty((len(test_lakes)))
new_rmse = np.empty((len(test_lakes)))
old_rmse[:] = np.nan
new_rmse[:] = np.nan

for site_id in test_lakes:
	ct += 1
	print(ct)

	old = pd.read_feather("./old_expanded/mtl_outputs9_expanded/nhdhr_"+site_id+"/9source_ensemble_output.feather")
	new = pd.read_feather("./mtl_outputs9_expanded/mtl_outputs9_expanded/nhdhr_"+site_id+"/9source_ensemble_output.feather")
	lab = pd.read_feather("./mtl_outputs9_expanded/mtl_outputs9_expanded/nhdhr_"+site_id+"/labels.feather")

	date_ind = np.where(np.isfinite(lab.values[:,1:].astype(np.float32)))[0][0]
	first_date = lab.iloc[date_ind,0]
	old = old[old['index'] == first_date]
	new = new[new['index'] == first_date]
	lab = lab[lab['index'] == first_date]

	new_rmse[ct-1] = np.sqrt(np.nanmean((new.iloc[:,1:].values - lab.iloc[:,1:].values) ** 2))
	old_rmse[ct-1] = np.sqrt(np.nanmean((old.iloc[:,1:].values - lab.iloc[:,1:].values) ** 2))
	if np.isnan(old_rmse[ct-1]):
		print("old nan")
		pdb.set_trace()
	if np.isnan(new_rmse[ct-1]):
		print("new nan")
		pdb.set_trace()



print(np.median(old_rmse))
print(np.median(new_rmse))
pdb.set_trace()