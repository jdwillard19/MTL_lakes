import pandas as pd
import numpy as np
import os
import pdb


###########################################################################3###############3
# (Sept 2020 - Jared) - this script uses the metadata created in calculateMetadata.py
#                       to calculate the differences between any two lakes and stores that information
#############################################################################################

metadata = pd.read_feather("../../metadata/lake_metadata_full.feather")
meta = metadata
ids = metadata['site_id'].values
if not os.path.exists("../../metadata/diffs"):
    os.mkdir("../../metadata/diffs")
metadata.set_index('site_id', inplace=True)

for i, lake in enumerate(ids):
    print("lake ", i, "/", ids.shape[0], ": ", lake)
    targ = meta.loc[lake][['K_d', 'SDF', 'latitude','longitude', 'max_depth', 'surface_area', \
                           'sw_mean', 'sw_std', 'lw_mean', 'lw_std', 'at_mean', 'at_std', 'rh_mean', 'rh_std', \
                           'ws_mean', 'ws_std', 'rain_mean', 'rain_std', 'snow_mean', 'snow_std', \
                           'sw_mean_sp', 'sw_std_sp', 'lw_mean_sp', 'lw_std_sp', 'at_mean_sp', 'at_std_sp', 'rh_mean_sp', 'rh_std_sp', \
                           'ws_mean_sp', 'ws_std_sp', 'rain_mean_sp', 'rain_std_sp', 'snow_mean_sp', 'snow_std_sp', \
                           'sw_mean_su', 'sw_std_su', 'lw_mean_su', 'lw_std_su', 'at_mean_su', 'at_std_su', 'rh_mean_su', 'rh_std_su', \
                           'ws_mean_su', 'ws_std_su', 'rain_mean_su', 'rain_std_su', 'snow_mean_su', 'snow_std_su', \
                           'sw_mean_au', 'sw_std_au', 'lw_mean_au', 'lw_std_au', 'at_mean_au', 'at_std_au', 'rh_mean_au', 'rh_std_au', \
                           'ws_mean_au', 'ws_std_au', 'rain_mean_au', 'rain_std_au', 'snow_mean_au', 'snow_std_au',\
                           'sw_mean_wi', 'sw_std_wi', 'lw_mean_wi', 'lw_std_wi', 'at_mean_wi', 'at_std_wi', 'rh_mean_wi', 'rh_std_wi', \
                           'ws_mean_wi', 'ws_std_wi', 'rain_mean_wi', 'rain_std_wi', 'snow_mean_wi', 'snow_std_wi', \
                            'n_obs', 'n_prof', 'n_obs_wi', 'n_obs_sp', 'n_obs_su', 'n_obs_au', \
                            'obs_depth_mean_frac', 'obs_temp_mean', 'obs_temp_std', 'obs_temp_skew', 'obs_temp_kurt', \
                            'zero_temp_doy', 'at_amp', 'lathrop_strat', 'glm_strat_perc', 'ws_sp_mix']]
    
    #zero out observation based metadata on target, not a difference just a value!
    targ['n_obs'] = 0
    targ['n_obs_wi'] = 0
    targ['n_obs_sp'] = 0
    targ['n_obs_su'] = 0
    targ['n_obs_au'] = 0
    targ['n_prof'] = 0
    targ['obs_depth_mean_frac'] = 0
    targ['obs_temp_mean'] = 0
    targ['obs_temp_mean_airdif'] = 0
    targ['obs_temp_std'] = 0
    targ['obs_temp_skew'] = 0
    targ['obs_temp_kurt'] = 0

    others = meta.drop(lake, axis=0)
    others.drop(['fullname', 'canopy', 'glm_uncal_rmse_third', 'glm_uncal_rmse_full'], axis=1, inplace=True)


    others['obs_temp_mean_airdif'] = others['at_mean'] - others['obs_temp_mean']
    diff = abs(others - targ)
    diff2 = others - targ

    diff2.drop(['n_obs', 'n_obs_wi', 'n_obs_sp', 'n_obs_su', 'n_obs_au', 'n_prof', 'obs_depth_mean_frac', 'obs_temp_mean', 'obs_temp_std', 'obs_temp_skew', 'obs_temp_kurt', 'obs_temp_mean_airdif'], axis=1, inplace=True)
    
    final_meta = pd.concat([diff, diff2], axis=1)
    final_meta['perc_dif_max_depth'] = (np.abs(targ['max_depth'] - others['max_depth']) / ((targ['max_depth']+others['max_depth'])/2))*100
    final_meta['perc_dif_surface_area'] = (np.abs(targ['surface_area'] - others['surface_area']) / ((targ['surface_area']+others['surface_area'])/2))*100
    final_meta['dif_sqrt_surface_area'] = np.sqrt(targ['surface_area']) - np.sqrt(others['surface_area'])
    #index was after strat_perc
    final_meta['perc_dif_sqrt_surface_area'] = (np.abs(np.sqrt(targ['surface_area']) - np.sqrt(others['surface_area'])) / ((np.sqrt(targ['surface_area']) + np.sqrt(others['surface_area']))/2))*100



    labs = ['ad_k_d', 'ad_SDF', 'ad_lat', 'ad_long','ad_max_depth','ad_surface_area', 'ad_sw_mean', 'ad_sw_std', \
                     'ad_lw_mean', 'ad_lw_std', 'ad_at_mean', 'ad_at_std', 'ad_rh_mean', 'ad_rh_std', 'ad_ws_mean', 'ad_ws_std', \
                     'ad_rain_mean', 'ad_rain_std', 'ad_snow_mean', 'ad_snow_std', \
                     'ad_sw_mean_sp', 'ad_sw_std_sp', \
                     'ad_lw_mean_sp', 'ad_lw_std_sp', 'ad_at_mean_sp', 'ad_at_std_sp', 'ad_rh_mean_sp', 'ad_rh_std_sp', 'ad_ws_mean_sp', 'ad_ws_std_sp', \
                     'ad_rain_mean_sp', 'ad_rain_std_sp', 'ad_snow_mean_sp', 'ad_snow_std_sp', \
                     'ad_sw_mean_su', 'ad_sw_std_su', \
                     'ad_lw_mean_su', 'ad_lw_std_su', 'ad_at_mean_su', 'ad_at_std_su', 'ad_rh_mean_su', 'ad_rh_std_su', 'ad_ws_mean_su', 'ad_ws_std_su', \
                     'ad_rain_mean_su', 'ad_rain_std_su', 'ad_snow_mean_su', 'ad_snow_std_su', \
                     'ad_sw_mean_au', 'ad_sw_std_au', \
                     'ad_lw_mean_au', 'ad_lw_std_au', 'ad_at_mean_au', 'ad_at_std_au', 'ad_rh_mean_au', 'ad_rh_std_au', 'ad_ws_mean_au', 'ad_ws_std_au', \
                     'ad_rain_mean_au', 'ad_rain_std_au', 'ad_snow_mean_au', 'ad_snow_std_au', \
                     'ad_sw_mean_wi', 'ad_sw_std_wi', \
                     'ad_lw_mean_wi', 'ad_lw_std_wi', 'ad_at_mean_wi', 'ad_at_std_wi', 'ad_rh_mean_wi', 'ad_rh_std_wi', 'ad_ws_mean_wi', 'ad_ws_std_wi', \
                     'ad_rain_mean_wi', 'ad_rain_std_wi', 'ad_snow_mean_wi', 'ad_snow_std_wi', \
                     'n_obs', 'n_prof', 'n_obs_wi', 'n_obs_sp', 'n_obs_su', 'n_obs_au', \
                     'obs_depth_frac', 'obs_temp_mean', 'obs_temp_std', 'obs_temp_skew', 'obs_temp_kurt', 
                     'ad_zero_temp_doy', 'ad_at_amp', 'ad_lathrop_strat', 'ad_glm_strat_perc', 'ad_ws_sp_mix', 'obs_temp_mean_airdif', \
                     'dif_SDF', 'dif_k_d', 'dif_lat', 'dif_long','dif_max_depth','dif_surface_area', 'dif_sw_mean', 'dif_sw_std', \
                     'dif_lw_mean', 'dif_lw_std', 'dif_at_mean', 'dif_at_std', 'dif_rh_mean', 'dif_rh_std', 'dif_ws_mean', 'dif_ws_std', \
                     'dif_rain_mean', 'dif_rain_std', 'dif_snow_mean', 'dif_snow_std', \
                     'dif_sw_mean_sp', 'dif_sw_std_sp', \
                     'dif_lw_mean_sp', 'dif_lw_std_sp', 'dif_at_mean_sp', 'dif_at_std_sp', 'dif_rh_mean_sp', 'dif_rh_std_sp', 'dif_ws_mean_sp', 'dif_ws_std_sp', \
                     'dif_rain_mean_sp', 'dif_rain_std_sp', 'dif_snow_mean_sp', 'dif_snow_std_sp', \
                     'dif_sw_mean_su', 'dif_sw_std_su', \
                     'dif_lw_mean_su', 'dif_lw_std_su', 'dif_at_mean_su', 'dif_at_std_su', 'dif_rh_mean_su', 'dif_rh_std_su', 'dif_ws_mean_su', 'dif_ws_std_su', \
                     'dif_rain_mean_su', 'dif_rain_std_su', 'dif_snow_mean_su', 'dif_snow_std_su', \
                     'dif_sw_mean_au', 'dif_sw_std_au', \
                     'dif_lw_mean_au', 'dif_lw_std_au', 'dif_at_mean_au', 'dif_at_std_au', 'dif_rh_mean_au', 'dif_rh_std_au', 'dif_ws_mean_au', 'dif_ws_std_au', \
                     'dif_rain_mean_au', 'dif_rain_std_au', 'dif_snow_mean_au', 'dif_snow_std_au', \
                     'dif_sw_mean_wi', 'dif_sw_std_wi', \
                     'dif_lw_mean_wi', 'dif_lw_std_wi', 'dif_at_mean_wi', 'dif_at_std_wi', 'dif_rh_mean_wi', 'dif_rh_std_wi', 'dif_ws_mean_wi', 'dif_ws_std_wi', \
                     'dif_rain_mean_wi', 'dif_rain_std_wi', 'dif_snow_mean_wi', 'dif_snow_std_wi', 'dif_zero_temp_doy', 'dif_at_amp', 'dif_lathrop_strat', 'dif_glm_strat_perc', \
                     'dif_ws_sp_mix', 'perc_dif_max_depth', 'perc_dif_surface_area', 'dif_sqrt_surface_area', 'perc_dif_sqrt_surface_area']
    final_meta.columns = labs
    final_meta.reset_index(inplace=True)
    if not os.path.exists("../../metadata/diffs/"+lake):
        os.mkdir("../../metadata/diffs/"+lake)
    final_meta.to_feather("../../metadata/diffs/target_"+lake+".feather")



