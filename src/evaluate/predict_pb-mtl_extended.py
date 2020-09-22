import numpy as np
import pdb
import pandas as pd
import sys
sys.path.append('../data')
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import spearmanr
from joblib import dump, load
import re

################################################################################3
# (Sept 2020 - Jared) - evaluate PB-MTL model by predicting best source model for each of 1882 expanded test lakes
# Features and hyperparamters must be manually specified below 
# (e.g. feats = ['dif_max_depth', ....]; n_estimators = 4000, etc)
#############################################################################################################

#file to save results to
save_file_path = '../../results/pbmtl_glm_transfer_results_expanded.csv'

#path to load metamodel from
model_path = "../../models/metamodel_glm_RMSE_GBR.joblib"

#load needed data
metadata = pd.read_feather("../../metadata/lake_metadata_full.feather")
glm_all_f = pd.read_csv("../../results/glm_transfer/RMSE_transfer_glm_pball.csv")
train_lakes = [re.search('nhdhr_(.*)', x).group(1) for x in np.unique(glm_all_f['target_id'].values)]
train_lakes_wp = np.unique(glm_all_f['target_id'].values) #with prefix
n_lakes = len(train_lakes)
all_sites = metadata['site_id'].values
test_lakes = all_sites[~np.isin(all_sites, np.unique(glm_all_f['target_id'].values))]

#########################################################################################
#paste features found in "pbmtl_feature_selection.py" here
feats = ['n_obs_su', 'obs_temp_mean', 'obs_temp_skew', 'obs_temp_kurt',
       'ad_glm_strat_perc', 'obs_temp_mean_airdif', 'dif_max_depth',
       'dif_surface_area', 'dif_sw_mean_au', 'dif_ws_mean_au',
       'dif_lathrop_strat', 'dif_glm_strat_perc', 'perc_dif_max_depth',
       'perc_dif_sqrt_surface_area']
###################################################################################



#load metamodel
model = load(model_path)

########################
##########################
# framework evaluation code
##########################
#######################

#csv header line
csv = [ 'target_id,source_id,predicted_rmse']
for feat in feats:
	csv[0] = csv[0] + ','+str(feat)

for targ_ct, target_id in enumerate(test_lakes): #for each target lake
	print("target lake ", targ_ct, ":", target_id)
	lake_df = pd.read_feather("../../metadata/diffs/target_"+ target_id +".feather")
	lake_df = lake_df[np.isin(lake_df['site_id'], train_lakes_wp)]




	X = pd.DataFrame(lake_df[feats])

	y_pred = model.predict(X)
	lake_df['rmse_pred'] = y_pred
	lake_df.sort_values(by=['rmse_pred'], inplace=True)
	best_predicted = lake_df.iloc[0]['site_id']
	best_predicted_rmse = lake_df.iloc[0]['rmse_pred']
	# lake_df = lake_df[lake_df['rmse_pred'] < lowest_rmse+rel_cut]
	csv.append(",".join([str(target_id), str(best_predicted), str(best_predicted_rmse)] + [str(x) for x in lake_df.iloc[0][feats].values]))




with open(save_file_path,'w') as file:
    for line in csv:
        file.write(line)
        file.write('\n')

