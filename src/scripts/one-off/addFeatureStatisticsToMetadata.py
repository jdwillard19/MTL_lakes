import pandas as pd
import numpy as np
import os
import pdb
import sys
from scipy.stats import skew, kurtosis
sys.path.append('../../data')
from date_operations import get_season
metadata = pd.read_feather("../../../metadata/lake_metadata_wNew4.feather")
metadata.drop("Unnamed: 0", axis=1, inplace=True)
ids = metadata['site_id']
metadata.set_index('site_id', inplace=True)
# labels =  metadata.columns.tolist() + \
#            ['sw_mean', 'sw_std', 'lw_mean', 'lw_std', 'at_mean', 'at_std', 'rh_mean', 'rh_std', 
#             'ws_mean', 'ws_std', 'rain_mean', 'rain_std', 'snow_mean', 'snow_std',
#             'sw_mean', 'sw_std', 'lw_mean', 'lw_std', 'at_mean', 'at_std', 'rh_mean', 'rh_std', 
#             'ws_mean', 'ws_std', 'rain_mean', 'rain_std', 'snow_mean', 'snow_std',
#             'sw_mean', 'sw_std', 'lw_mean', 'lw_std', 'at_mean', 'at_std', 'rh_mean', 'rh_std', 
#             'ws_mean', 'ws_std', 'rain_mean', 'rain_std', 'snow_mean', 'snow_std',
#             'sw_mean', 'sw_std', 'lw_mean', 'lw_std', 'at_mean', 'at_std', 'rh_mean', 'rh_std', 
#             'ws_mean', 'ws_std', 'rain_mean', 'rain_std', 'snow_mean', 'snow_std',
#             'sw_mean', 'sw_std', 'lw_mean', 'lw_std', 'at_mean', 'at_std', 'rh_mean', 'rh_std', 
#             'ws_mean', 'ws_std', 'rain_mean', 'rain_std', 'snow_mean', 'snow_std',
#              'n_obs', 'n_prof', 'obs_depth_mean_frac', 'obs_temp_mean', 'obs_temp_std']
new_lab = ['sw_mean', 'sw_std', 'lw_mean', 'lw_std', 'at_mean', 'at_std', 'rh_mean', 'rh_std', \
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
            'zero_temp_doy', 'at_amp', 'lathrop_strat', 'glm_strat_perc', 'ws_sp_mix']

meta = pd.concat([metadata, pd.DataFrame(np.nan, metadata.index, new_lab)], axis=1)

for i, lake in enumerate(ids):
    print("lake ", i, ": ", lake)
    name = lake
    #load meteorological data
    if np.isin(lake,df2['site_id'].values):
        continue

    meteo = None
    if os.path.exists("../../../data/raw/figure3/nhd_"+lake+"_meteo.csv"):
        meteo = pd.read_csv("../../../data/raw/figure3/nhd_"+lake+"_meteo.csv")
        meteo.drop("Unnamed: 0", axis=1, inplace=True)

    elif os.path.exists("../../../data/raw/figure3_revised/meteo_data/nhd_"+lake+"_meteo.csv"):
        meteo = pd.read_csv("../../../data/raw/figure3_revised/meteo_data/nhd_"+lake+"_meteo.csv")


#['time', 'ShortWave', 'LongWave', 'AirTemp', 'RelHum', 'WindSpeed',  'Rain', 'Snow']
    seasons = np.array([get_season(pd.Timestamp(t).to_pydatetime()) for t in meteo['time']])

    (sw_m, sw_s) = (meteo['ShortWave'].mean(), meteo['ShortWave'].std())
    (lw_m, lw_s) = (meteo['LongWave'].mean(), meteo['LongWave'].std())
    (at_m, at_s) = (meteo['AirTemp'].mean(), meteo['AirTemp'].std())
    (rh_m, rh_s) = (meteo['RelHum'].mean(), meteo['RelHum'].std())
    (ws_m, ws_s) = (meteo['WindSpeed'].mean(), meteo['WindSpeed'].std())
    (r_m, r_s) = (meteo['Rain'].mean(), meteo['Rain'].std())
    (s_m, s_s) = (meteo['Snow'].mean(), meteo['Snow'].std())

    (sw_m_wi, sw_s_wi) = (meteo.iloc[seasons == 'winter']['ShortWave'].mean(), meteo.iloc[seasons == 'winter']['ShortWave'].std())
    (lw_m_wi, lw_s_wi) = (meteo.iloc[seasons == 'winter']['LongWave'].mean(), meteo.iloc[seasons == 'winter']['LongWave'].std())
    (at_m_wi, at_s_wi) = (meteo.iloc[seasons == 'winter']['AirTemp'].mean(), meteo.iloc[seasons == 'winter']['AirTemp'].std())
    (rh_m_wi, rh_s_wi) = (meteo.iloc[seasons == 'winter']['RelHum'].mean(), meteo.iloc[seasons == 'winter']['RelHum'].std())
    (ws_m_wi, ws_s_wi) = (meteo.iloc[seasons == 'winter']['WindSpeed'].mean(), meteo.iloc[seasons == 'winter']['WindSpeed'].std())
    (r_m_wi, r_s_wi) = (meteo.iloc[seasons == 'winter']['Rain'].mean(), meteo.iloc[seasons == 'winter']['Rain'].std())
    (s_m_wi, s_s_wi) = (meteo.iloc[seasons == 'winter']['Snow'].mean(), meteo.iloc[seasons == 'winter']['Snow'].std())

    (sw_m_sp, sw_s_sp) = (meteo.iloc[seasons == 'spring']['ShortWave'].mean(), meteo.iloc[seasons == 'spring']['ShortWave'].std())
    (lw_m_sp, lw_s_sp) = (meteo.iloc[seasons == 'spring']['LongWave'].mean(), meteo.iloc[seasons == 'spring']['LongWave'].std())
    (at_m_sp, at_s_sp) = (meteo.iloc[seasons == 'spring']['AirTemp'].mean(), meteo.iloc[seasons == 'spring']['AirTemp'].std())
    (rh_m_sp, rh_s_sp) = (meteo.iloc[seasons == 'spring']['RelHum'].mean(), meteo.iloc[seasons == 'spring']['RelHum'].std())
    (ws_m_sp, ws_s_sp) = (meteo.iloc[seasons == 'spring']['WindSpeed'].mean(), meteo.iloc[seasons == 'spring']['WindSpeed'].std())
    (r_m_sp, r_s_sp) = (meteo.iloc[seasons == 'spring']['Rain'].mean(), meteo.iloc[seasons == 'spring']['Rain'].std())
    (s_m_sp, s_s_sp) = (meteo.iloc[seasons == 'spring']['Snow'].mean(), meteo.iloc[seasons == 'spring']['Snow'].std())

    (sw_m_su, sw_s_su) = (meteo.iloc[seasons == 'summer']['ShortWave'].mean(), meteo.iloc[seasons == 'summer']['ShortWave'].std())
    (lw_m_su, lw_s_su) = (meteo.iloc[seasons == 'summer']['LongWave'].mean(), meteo.iloc[seasons == 'summer']['LongWave'].std())
    (at_m_su, at_s_su) = (meteo.iloc[seasons == 'summer']['AirTemp'].mean(), meteo.iloc[seasons == 'summer']['AirTemp'].std())
    (rh_m_su, rh_s_su) = (meteo.iloc[seasons == 'summer']['RelHum'].mean(), meteo.iloc[seasons == 'summer']['RelHum'].std())
    (ws_m_su, ws_s_su) = (meteo.iloc[seasons == 'summer']['WindSpeed'].mean(), meteo.iloc[seasons == 'summer']['WindSpeed'].std())
    (r_m_su, r_s_su) = (meteo.iloc[seasons == 'summer']['Rain'].mean(), meteo.iloc[seasons == 'summer']['Rain'].std())
    (s_m_su, s_s_su) = (meteo.iloc[seasons == 'summer']['Snow'].mean(), meteo.iloc[seasons == 'summer']['Snow'].std())

    (sw_m_au, sw_s_au) = (meteo.iloc[seasons == 'autumn']['ShortWave'].mean(), meteo.iloc[seasons == 'autumn']['ShortWave'].std())
    (lw_m_au, lw_s_au) = (meteo.iloc[seasons == 'autumn']['LongWave'].mean(), meteo.iloc[seasons == 'autumn']['LongWave'].std())
    (at_m_au, at_s_au) = (meteo.iloc[seasons == 'autumn']['AirTemp'].mean(), meteo.iloc[seasons == 'autumn']['AirTemp'].std())
    (rh_m_au, rh_s_au) = (meteo.iloc[seasons == 'autumn']['RelHum'].mean(), meteo.iloc[seasons == 'autumn']['RelHum'].std())
    (ws_m_au, ws_s_au) = (meteo.iloc[seasons == 'autumn']['WindSpeed'].mean(), meteo.iloc[seasons == 'autumn']['WindSpeed'].std())
    (r_m_au, r_s_au) = (meteo.iloc[seasons == 'autumn']['Rain'].mean(), meteo.iloc[seasons == 'autumn']['Rain'].std())
    (s_m_au, s_s_au) = (meteo.iloc[seasons == 'autumn']['Snow'].mean(), meteo.iloc[seasons == 'autumn']['Snow'].std())

    #now do observation data
    obs = None
    if os.path.exists('../../../data/raw/figure3/nhd_'+name+'_test_train.feather'):
        obs = pd.read_feather('../../../data/raw/figure3/nhd_'+name+'_test_train.feather')
    elif os.path.exists('../../../data/raw/figure3_revised/training_data_all_profiles/nhd_'+name+'_train_all_profiles.feather'):
        obs = pd.read_feather('../../../data/raw/figure3_revised/training_data_all_profiles/nhd_'+name+'_train_all_profiles.feather')
    else:
        raise Exception("obs f not found")

    #processed obs file / feats / files
    obs_f = np.load("../../../data/processed/lake_data/" + lake + "/full_obs.npy")
    feats_f = np.load("../../../data/processed/lake_data/" + lake + "/features.npy")

    #number of obs
    n_obs = np.count_nonzero(np.isfinite(obs_f))

    #count seasonal observations
    obs_seasons = np.array([get_season(pd.Timestamp(t).to_pydatetime()) for t in obs['date']])
    n_obs_wi = np.sum(obs_seasons == 'winter')
    n_obs_sp = np.sum(obs_seasons == 'spring')
    n_obs_su = np.sum(obs_seasons == 'summer')
    n_obs_au = np.sum(obs_seasons == 'autumn')

    #count profiles
    n_prof = np.sum(np.count_nonzero(np.isfinite(obs_f), axis=0) > 0)

    #statistics
    max_d = meta.loc[lake]['max_depth']
    mean_obs_frac = obs['depth'].mean() / max_d
    mean_obs_temp = obs['temp'].mean() 
    std_obs_temp = obs['temp'].std()
    skew_obs_temp = skew(obs['temp'])
    kurt_obs_temp = kurtosis(obs['temp'])
    new_feat = [sw_m, sw_s, lw_m, lw_s, at_m, at_s, rh_m, rh_s, ws_m, ws_s, r_m, r_s, s_m, s_s, \
                sw_m_wi, sw_s_wi, lw_m_wi, lw_s_wi, at_m_wi, at_s_wi, rh_m_wi, rh_s_wi, ws_m_wi, ws_s_wi, r_m_wi, r_s_wi, s_m_wi, s_s_wi, \
                sw_m_sp, sw_s_sp, lw_m_sp, lw_s_sp, at_m_sp, at_s_sp, rh_m_sp, rh_s_sp, ws_m_sp, ws_s_sp, r_m_sp, r_s_sp, s_m_sp, s_s_sp, \
                sw_m_su, sw_s_su, lw_m_su, lw_s_su, at_m_su, at_s_su, rh_m_su, rh_s_su, ws_m_su, ws_s_su, r_m_su, r_s_su, s_m_su, s_s_su, \
                sw_m_au, sw_s_au, lw_m_au, lw_s_au, at_m_au, at_s_au, rh_m_au, rh_s_au, ws_m_au, ws_s_au, r_m_au, r_s_au, s_m_au, s_s_au, \
                int(n_obs), int(n_prof), int(n_obs_wi), int(n_obs_sp), int(n_obs_su), int(n_obs_au), mean_obs_frac, mean_obs_temp, std_obs_temp, skew_obs_temp, kurt_obs_temp]

    for j, lab in enumerate(new_lab):
        meta.loc[lake, lab] = new_feat[j]

meta.reset_index(inplace=True)
meta.to_feather("../../../metadata/lake_metadata_wStats2.feather")