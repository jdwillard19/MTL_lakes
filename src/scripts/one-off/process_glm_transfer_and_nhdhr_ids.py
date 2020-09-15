import pandas as pd 
import numpy as np 
import sys
import os
import pdb
sys.path.append('../../data')
from metadata_ops import nhd2nhdhr
import shutil
import re 

ids = pd.read_csv('../../../metadata/pball_site_ids.csv', header=None)
ids = ids[0].values
# metadata = pd.read_feather("../../../metadata/lake_metadata_012920_wNhdhr.feather")
glm_all_f = pd.read_csv("../../../results/glm_transfer/RMSE_transfer_glm_pball.csv")
# train_df = pd.read_feather("../../../results/transfer_learning/glm/train_rmses_pball.feather")
train_lakes = [re.search('nhdhr_(.*)', x).group(1) for x in np.unique(glm_all_f['target_id'].values)]
n_lakes = len(train_lakes)
test_lakes = ids[~np.isin(ids, train_lakes)]
ids = train_lakes
# temp = pd.read_csv("../../../results/transfer_learning/target_13293262/resultsPGRNNbasic_wNew_norm2")
# conv = pd.read_feather("../../../data/raw/crosswalk/nhd_to_nhdhr.feather")
# trn_tst_f = pd.read_feather("../../../data/processed/lake_splits/trn_tst2.feather")
# ids = trn_tst_f[trn_tst_f['isTrain3_JR']]['site_id'].values
# to_del = ['120018008', '120020307', '120020636', '32671150', '58125241', '120020800', '91598525']
# train_lakes = np.setdiff1d(train_lakes, to_del)
new_df = pd.DataFrame()
inc = []
for ct, i_d in enumerate(ids):
	#for each target id
	print(ct, ": ", i_d)
	#get diffs
	# diffs = pd.read_csv("../../../metadata/diff/target_"+ i_d +"_featsJan20_wHR.csv")
	try:
		diffs = pd.read_feather("../../../metadata/diff/target_"+ i_d +"_pball_Aug2020.feather")
	except:
		inc.append(i_d)
		continue
	# diffs = pd.read_feather("../../../metadata/diff/target_"+ i_d +"_pball_update.feather")
	diffs['site_id2'] = ["nhdhr_" + x for x in diffs['site_id'].values]

		#convert
	glm_res = glm_all_f[glm_all_f['target_id'] == 'nhdhr_'+i_d]

	#merge files
	merged_inner = pd.merge(left=glm_res, right=diffs, left_on='source_id', right_on='site_id2')
	# assert merged_inner.shape[0] == 144
	# if merged_inner.shape[0] == 0:
	# 	inc.append(i_d)
	# 	continue
	new_df = pd.concat([new_df, merged_inner], ignore_index=True)
pdb.set_trace()
new_df.reset_index().to_feather("../../../results/glm_transfer/glm_meta_train_data.feather")