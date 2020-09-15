import pandas as pd
import numpy as np
import os
import pdb
import re

new_lakes = pd.read_feather("../../../metadata/lake_metadata.feather")
new_lakes = new_lakes['site_id']
old_lakes = pd.read_csv("../../../metadata/lake_metadata.csv")['nhd_id']


old_ct = np.empty_like(old_lakes.values)
new_ct = np.empty((new_lakes.values.shape[0] - 5))
old_tot = np.empty_like(old_lakes.values)
new_tot = np.empty((new_lakes.values.shape[0] - 5))
old_ct[:] = np.nan
old_tot[:] = np.nan
new_ct[:] = np.nan
new_tot[:] = np.nan 
for i, lake in enumerate(old_lakes):
	# if lake == '1097324':
		# pdb.set_trace()
	print(lake)
	obs = pd.read_feather("../../../data/raw/figure3/nhd_"+str(lake)+"_test_train.feather")
	n_s = np.sum(obs['depth'] < 0.25)
	n_o = obs.shape[0] - n_s
	old_ct[i] = n_s
	old_tot[i] = n_o
	#total surface obs

ct = -1
for i, lake in enumerate(new_lakes):
	ct += 1
	nid = lake
	match = re.search("nhdhr_(.+)", str(lake))
	if nid == 'nhdhr_120018008' or nid == 'nhdhr_120020307' or nid == 'nhdhr_120020636' or nid == 'nhdhr_32671150' or nid =='nhdhr_58125241':
		ct -= 1
		continue

	lake = match.group(1)

	obs = pd.read_feather("../../../data/raw/figure3/nhd_"+str(lake)+"_test_train.feather")
	n_s = np.sum(obs['depth'] < 0.25)
	n_o = obs.shape[0] - n_s
	new_ct[ct] = n_s
	new_tot[ct] = n_o



print("old=", np.sum(old_ct)/np.sum(old_tot), ", new=", np.sum(new_ct)/np.sum(new_tot))
	#total surface obs


