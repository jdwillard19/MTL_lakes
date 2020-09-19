import pandas as pd 
import numpy as np 
import sys
import os
import pdb
sys.path.append('../../data')
import shutil
import re 

ids = pd.read_csv('../../metadata/pball_site_ids.csv', header=None)
ids = ids[0].values
glm_all_f = pd.read_csv("../../results/glm_transfer/RMSE_transfer_glm_pball.csv")
train_lakes = [re.search('nhdhr_(.*)', x).group(1) for x in np.unique(glm_all_f['target_id'].values)]
n_lakes = len(train_lakes)
test_lakes = ids[~np.isin(ids, train_lakes)]
ids = train_lakes
new_df = pd.DataFrame()
inc = []
for ct, i_d in enumerate(ids):
	#for each target id
	print(ct, ": ", i_d)
	try:
		diffs = pd.read_feather("../../metadata/diffs/target_nhdhr_"+ i_d +".feather")
	except:
		inc.append(i_d)
		continue
	diffs['site_id2'] = ["nhdhr_" + x for x in diffs['site_id'].values]

		#convert
	glm_res = glm_all_f[glm_all_f['target_id'] == 'nhdhr_'+i_d]

	#merge files
	merged_inner = pd.merge(left=glm_res, right=diffs, left_on='source_id', right_on='site_id')
	new_df = pd.concat([new_df, merged_inner], ignore_index=True)
new_df.reset_index().to_feather("../../results/glm_transfer/glm_meta_train_data.feather")