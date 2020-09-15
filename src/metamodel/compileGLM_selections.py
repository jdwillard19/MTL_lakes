import pandas as pd
import numpy as np
import pdb
import os
import sys
import re
from scipy.stats import spearmanr
sys.path.append('../data')

from metadata_ops import nhd2nhdhr
conv = pd.read_feather("../../data/raw/crosswalk/nhd_to_nhdhr.feather")
glm_tst = pd.read_csv("RMSE_transfer_test_glm.csv")
glm_trn = pd.read_csv('RMSE_transfer_glm.csv')
pgml_tst = pd.read_csv("pgmtl_results_singleSource012920.csv", names=['target_id','source_id','rmse','spear','glm_uncal_rmse'])
select_f = pd.read_csv("glm_transfer_target_source.csv", names=['target_id', 'source_id','spear'])

ids = np.array([re.search('test_nhdhr_(.+)', txt).group(1) for txt in pgml_tst['target_id']])

glm_source_ids = []
glm_source_rmses = []
glm_min_source_rmses = []
glm_spears = []
new_ids = []
for i_d in ids:
	full_id = 'test_nhdhr_' + i_d
	source_id = re.search('nhdhr_(.+)', pgml_tst[pgml_tst['target_id'] == full_id]['source_id'].values[0]).group(1)
	if np.isin('nhd_'+source_id, conv['WRR_ID']):
		new_id = nhd2nhdhr(source_id)
		#convert
	else:
		new_id = source_id

	new_ids.append(new_id)
	if full_id =='test_nhdhr_120018092':
		pdb.set_trace()
	glm_source_id = str(select_f[select_f['target_id'] == i_d]['source_id'].values[0])
	glm_source_ids.append(glm_source_id)

	query = 'target_id == "test_nhdhr_'+i_d+'" and source_id == "nhdhr_'+glm_source_id+'"'
	glm_source_rmses.append(glm_tst.query(query)['rmse'].values[0])
	glm_min_source_rmse = glm_tst[glm_tst['target_id'] == full_id]['rmse'].min()
	glm_spears.append(select_f[select_f['target_id'] == i_d]['spear'].values[0])


pgml_tst['source_id2'] = new_ids
pgml_tst['glm_source_id'] = np.array(glm_source_ids)
pgml_tst['glm_source_rmse'] = np.array(glm_source_rmses)
pgml_tst['glm_source_spear'] = np.array(glm_spears)


pgml_tst.to_csv("compiledTestResults2.csv")