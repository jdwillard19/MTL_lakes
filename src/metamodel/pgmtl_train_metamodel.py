import pandas as pd
import numpy as np
import pdb
import sys
import os
from sklearn.ensemble import GradientBoostingRegressor
from joblib import dump, load
import re

##################################################################3
# (Sept 2020 - Jared) - PG-MTL training script on 145 source lake 
# Features and hyperparamters must be manually specified below 
# (e.g. feats = ['dif_max_depth', ....]; n_estimators = 5500, etc)
####################################################################3

#file to save model  to
save_file_path = '../../models/metamodel_pgdl_RMSE_GBR.joblib'

#########################################################################################
#paste features found in "pbmtl_feature_selection.py" here
feats = ['n_obs_sp', 'n_obs_su', 'dif_max_depth', 'dif_surface_area',
       'dif_glm_strat_perc', 'perc_dif_max_depth', 'perc_dif_surface_area',
       'perc_dif_sqrt_surface_area']
###################################################################################


#######################################################################3
#paste hyperparameters found in "pbmtl_hyperparameter_search.py" here
#
n_estimators = 5500
lr = .05
#####################################################################

ids = pd.read_csv('../../metadata/pball_site_ids.csv', header=None)
ids = ids[0].values
glm_all_f = pd.read_csv("../../results/glm_transfer/RMSE_transfer_glm_pball.csv")
train_lakes = [re.search('nhdhr_(.*)', x).group(1) for x in np.unique(glm_all_f['target_id'].values)]
train_lakes_wp = np.unique(glm_all_f['target_id'].values) #with prefix



#compile training data
train_df = pd.DataFrame()
for _, lake_id in enumerate(train_lakes):
    new_df = pd.DataFrame()

    #get performance results (metatargets), filter out target as source
    lake_df_res = pd.read_csv("../../results/transfer_learning/target_"+lake_id+"/resultsPGRNNbasic_pball",header=None,names=['source_id','rmse'])
    lake_df_res = lake_df_res[lake_df_res.source_id != 'source_id']

    #get metadata differences between target and all the sources
    lake_df = pd.read_feather("../../metadata/diffs/target_nhdhr_"+lake_id+".feather")
    lake_df = lake_df[np.isin(lake_df['site_id'], train_lakes_wp)]
    lake_df_res = lake_df_res[np.isin(lake_df_res['source_id'], train_lakes)]
    lake_df_res['source_id2'] = ['nhdhr_'+str(x) for x in lake_df_res['source_id'].values]
    lake_df = pd.merge(left=lake_df, right=lake_df_res.astype('object'), left_on='site_id', right_on='source_id2')
    new_df = lake_df
    train_df = pd.concat([train_df, new_df], ignore_index=True)



#train model
X_trn = pd.DataFrame(train_df[feats])
y_trn = np.array([float(x) for x in np.ravel(pd.DataFrame(train_df['rmse']))])
model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=lr)
print("Training metamodel...")
model.fit(X_trn, y_trn)
dump(model, save_file_path)
print("Training Complete, saved to ", save_file_path)
