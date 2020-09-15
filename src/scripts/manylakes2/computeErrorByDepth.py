import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import pdb
import sys
sys.path.append('../../data')
from scipy.stats import spearmanr
import re
import os
import glob

metadata = pd.read_feather("../../../metadata/lake_metadata_2700plus.feather")
sites = pd.read_csv('../../../metadata/sites_moreThan10ProfilesWithGLM_Mar2020Update.csv')
ids = pd.read_csv('../../../metadata/pball_site_ids.csv', header=None)
ids = ids[0].values
glm_all_f = pd.read_csv("../../../results/glm_transfer/RMSE_transfer_glm_pball.csv")
train_df = pd.read_feather("../../../results/transfer_learning/glm/train_rmses_pball.feather")
train_lakes = [re.search('nhdhr_(.*)', x).group(1) for x in np.unique(glm_all_f['target_id'].values)]
n_lakes = len(train_lakes)
all_sites = metadata['site_id'].values
all_test_lakes = all_sites[~np.isin(all_sites, train_lakes)]

glm_all_f = pd.read_csv("../../../results/glm_transfer/RMSE_transfer_glm_pball.csv")
train_df = pd.read_feather("../../../results/transfer_learning/glm/train_rmses_pball.feather")
train_lakes = [re.search('nhdhr_(.*)', x).group(1) for x in np.unique(glm_all_f['target_id'].values)]
n_lakes = len(train_lakes)
some_test_lakes = ids[~np.isin(ids, train_lakes)]


glm_err_per_depth = np.empty((550),dtype=np.object)
# glm_err_per_depth = [[] for i in range(550)]
pg_err_per_depth = np.empty((550),dtype=np.object)
# pg_err_per_depth = [[] for i in range(550)]

for i in range(550):
    glm_err_per_depth[i] = []
    pg_err_per_depth[i] = []
# pg_err_per_depth[:] = 0
# glm_err_per_depth[:] = 0
ct_per_depth = np.empty((550))
ct_per_depth[:] = 0
# some_test_lakes = all_test_lakes
# all_test_lakes = np.array(all_test_lakes[0:4])
for site_ct, site_id in enumerate(all_test_lakes):
	print("site ", site_ct,"/",len(all_test_lakes))

	#load output files
	output = pd.read_feather('./mtl_outputs/nhdhr_'+site_id+'/9source_ensemble_output')

	#load glm file
	glm_out = np.load('../../../data/processed/lake_data/'+site_id+'/glm.npy')
	glm_dates = np.load('../../../data/processed/lake_data/'+site_id+'/dates.npy')
	labels_npy = np.load('../../../data/processed/lake_data/'+site_id+'/full.npy')


	#load label
	label = pd.read_feather('./mtl_outputs/nhdhr_'+site_id+'/labels')

	#align glm and output dates
	for d in range(glm_out.shape[0]):
		glm_err = np.sqrt(np.nanmean((glm_out[d,:]-labels_npy[d,:])**2))
		# if site_ct == 2:
			# pdb.set_trace()
		pg_err = np.sqrt(np.nanmean((np.array(output.iloc[d,1:].values,dtype=np.float32) - np.array(label.iloc[d,1:].values,dtype=np.float32))**2))
		if np.isnan(pg_err):
			continue
		elif np.isnan(glm_err):
			print("glm err nan?")
			pdb.set_trace()
			continue
		else:
			pg_err_per_depth[d].append(pg_err)
			# pg_err_per_depth[d] += pg_err
			# glm_err_per_depth[d] += glm_err
			glm_err_per_depth[d].append(glm_err)
			# ct_per_depth[d] += 1
pg_medians = [np.median(np.array(pg_err_per_depth[i])) for i in range(550)]
glm_medians = [np.median(np.array(glm_err_per_depth[i])) for i in range(550)]
# pg_means = (pg_err_per_depth / ct_per_depth)
# glm_means = (glm_err_per_depth / ct_per_depth)
a = pd.DataFrame()
a['pg_median'] = pg_medians
a['glm_median'] = glm_medians
# a['lakes_w_depth'] = ct_per_depth
a.to_csv('./err_by_depth_2233lakes_median.csv')



