import pandas as pd
import numpy as np
import pdb
import sys
sys.path.append('../../data')
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import re 

glm_all_f = pd.read_csv("../../results/glm_transfer/RMSE_transfer_glm_pball.csv")
train_df = pd.read_feather("../../results/transfer_learning/glm/glm_meta_train_rmses.feather")
train_lakes = [re.search('nhdhr_(.*)', x).group(1) for x in np.unique(glm_all_f['target_id'].values)]
train_lakes_wp = np.unique(glm_all_f['target_id'].values) #with prefix
n_lakes = len(train_lakes)

#cv params
nfolds = 24


# Feats found in "pgmtl_feature_selection.py pasted here"
##################################################################################
feats = ['n_obs_sp', 'n_obs_su', 'dif_max_depth', 'dif_surface_area',
       'dif_glm_strat_perc', 'perc_dif_max_depth', 'perc_dif_surface_area',
       'perc_dif_sqrt_surface_area']

# feats = ['n_obs', 'obs_temp_mean', 'dif_max_depth', 'dif_surface_area',
#        'dif_rh_mean_au', 'dif_lathrop_strat', 'dif_glm_strat_perc',
#        'perc_dif_max_depth', 'perc_dif_surface_area',
#        'perc_dif_sqrt_surface_area']
####################################################################################


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




X = pd.DataFrame(train_df[feats])
y = np.ravel(pd.DataFrame(train_df['rmse']))

def gb_param_selection(X, y, nfolds):
    ests = np.arange(1000,6000,500)
    lrs = [.05,.01]
    # max_d = [3, 5]
    param_grid = {'n_estimators': ests, 'learning_rate' : lrs}
    grid_search = GridSearchCV(GradientBoostingRegressor(), param_grid, cv=nfolds, n_jobs=-1)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_


print(gb_param_selection(X, y, nfolds))

