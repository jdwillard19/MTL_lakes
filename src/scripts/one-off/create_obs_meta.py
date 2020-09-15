import pandas as pd
import numpy as np
import os
import pdb

metadata = pd.read_csv("../../../metadata/lake_metadata_wNew.csv")
ids = metadata['nhd_id']
to_del = ['120018008', '120020307', '120020636', '32671150', '58125241', '120020800', '91598525']
ids = np.setdiff1d(ids, to_del)
metadata.set_index('nhd_id', inplace=True)

new = pd.DataFrame()
new = metadata
n_lakes = ids.shape[0]
n_obs = np.empty((n_lakes))
n_prof = np.empty((n_lakes))
n_obs[:] = np.nan
n_prof[:] = np.nan

for index, lake in enumerate(ids):
	#load obs data
	obs_f = np.load("../../../data/processed/WRR_69Lake/" + lake + "/full_obs.npy")
	feats_f = np.load("../../../data/processed/WRR_69Lake/" + lake + "/features.npy")
	n_obs[index] = np.count_nonzero(np.isfinite(obs_f))
	n_prof[index] = np.sum(np.count_nonzero(np.isfinite(obs_f), axis=0) > 0)

csv = []

for i in range(n_obs.shape[0]):
	csv.append(",".join([str(n_obs[i]), str(n_prof[i])]))


with open("./obs_and_feats_hist.csv",'w') as file:
	for line in csv:
		file.write(line)
		file.write('\n')