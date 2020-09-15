import pandas as pd
import numpy as np
import os
import pdb
import math
import sys
from scipy.stats import skew, kurtosis
sys.path.append('../../data')
from date_operations import get_season

raw_dir = '../../../data/raw/figure3/' #unprocessed data directory

metadata = pd.read_feather("../../../metadata/lake_metadata_wStats2.feather")
# metadata.drop("Unnamed: 0", axis=1, inplace=True)
ids = metadata['nhd_id']
metadata.set_index('nhd_id', inplace=True)
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
new_lab = ['zero_temp_doy', 'at_amp', 'lathrop_strat', 'glm_strat_perc', 'ws_sp_mix']
days_per_year = 366
# (max depth(m) - .01) / (log10(area(ha))) > 3.8
meta = pd.concat([metadata, pd.DataFrame(np.nan, metadata.index, new_lab)], axis=1)
avoid = ['120018008', '120020307', '120020636', '32671150', '58125241', '120020800', '91598525']
# meta = meta.drop(avoid, axis=0)

def findZeroTempDay(arr):
    csd = 0 #consecutive subzero days
    fsdos = np.nan #first subzero day of seq
    for t in range(int(np.round(arr.shape[0]/2)), arr.shape[0]):
        if arr[t] < 0:
            if csd == 0:
                fsdos = t
            csd += 1
        else:
            fsdos = np.nan
            csd = 0
        if csd >= 5:
            return fsdos 


for i, lake in enumerate(ids):
    print("lake ", i, ": ", lake)
    name = lake

    #load meteorological data
    if lake in avoid:
        continue

    meteo = None
    if os.path.exists("../../../data/raw/figure3/nhd_"+lake+"_meteo.csv"):
        meteo = pd.read_csv("../../../data/raw/figure3/nhd_"+lake+"_meteo.csv")
        meteo.drop("Unnamed: 0", axis=1, inplace=True)

    elif os.path.exists("../../../data/raw/figure3_revised/meteo_data/nhd_"+lake+"_meteo.csv"):
        meteo = pd.read_csv("../../../data/raw/figure3_revised/meteo_data/nhd_"+lake+"_meteo.csv")

    dates = [pd.Timestamp(t).to_pydatetime() for t in meteo['time']]
    n_dates_meteo = len(dates)
    air_temps = meteo['AirTemp']
    wind_speeds = meteo['WindSpeed']
    assert n_dates_meteo == air_temps.shape[0]
    assert n_dates_meteo == wind_speeds.shape[0]
    temps_sum = np.zeros((days_per_year)) 
    temps_ct = np.zeros((days_per_year), dtype=np.int16) 
    wspeed_sum = np.zeros((days_per_year)) 
    wspeed_ct = np.zeros((days_per_year), dtype=np.int16) 
    for t in range(n_dates_meteo):
        doy = dates[t].timetuple().tm_yday
        temps_sum[doy-1] += air_temps[t]
        wspeed_sum[doy-1] += wind_speeds[t]
        temps_ct[doy-1] += 1
        wspeed_ct[doy-1] += 1
    temps_per_doy = np.divide(temps_sum, temps_ct)
    ws_per_doy = np.divide(wspeed_sum, wspeed_ct)
    subzero_ind = findZeroTempDay(temps_per_doy)


    #amplitude of avg air temp
    at_amp = (temps_per_doy.max() - temps_per_doy.min()) / 2



    #stratification indices
    area_ha = meta.loc[lake]['surface_area'] / 1.0e4
    max_d = meta.loc[lake]['max_depth']

    lathrop = 1 if ((max_d - 0.1) / np.log10(area_ha)) > 3.8 else 0

    glm = None
    if os.path.exists(raw_dir+"nhd_"+name+"_temperatures.feather"):
        glm = pd.read_feather(raw_dir+"nhd_"+name+"_temperatures.feather")
    else:
        glm = pd.read_feather("../../../data/raw/figure3_revised/pretraining_data/nhd_"+name+"_temperatures.feather")

    # diffs = np.array([(row.values[-4] - row.values[1]) if math.isnan(row.values[-3]) else (row.values[-2] - row.values[1]) for ind, row in glm.iterrows()])
    diffs = np.array([(row.values[-7] - row.values[1]) if len(row) > 6 and math.isnan(row.values[-6]) and type(row.values[-7]) is not pd.Timestamp \
                       else (row.values[-6] - row.values[1]) if len(row) > 5 and math.isnan(row.values[-5]) and type(row.values[-6]) is not pd.Timestamp\
                       else (row.values[-5] - row.values[1]) if len(row) > 4 and math.isnan(row.values[-4]) and type(row.values[-5]) is not pd.Timestamp\
                       else (row.values[-4] - row.values[1]) if math.isnan(row.values[-3]) and type(row.values[-4]) is not pd.Timestamp\
                       else (row.values[-3] - row.values[1]) if math.isnan(row.values[-2]) and type(row.values[-3]) is not pd.Timestamp\
                       else (row.values[-2] - row.values[1]) for ind, row in glm.iterrows()])
    assert np.where(~np.isfinite(diffs))[0].shape[0] == 0
    glm_strat_perc = np.sum(np.abs(diffs) > 1) / diffs.shape[0]

    #45(?) day average of all wind speeds after the date of the air temperature 0 estimate
    ws_ind0 = subzero_ind
    ws_ind1 = subzero_ind + 45
    ws_doy = np.repeat(ws_per_doy, 2)
    ws_sp_mix = ws_doy[ws_ind0:ws_ind1].mean() 
    print("subzero ind: ", subzero_ind, "\namp: ", at_amp, "\nlath: ", lathrop, \
          "\nglm_strat_pec: ", glm_strat_perc, "\nws_sp_mix: ", ws_sp_mix, "\nws_mean: ", ws_per_doy.mean())

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
    mean_obs_frac = obs['depth'].mean() / max_d
    mean_obs_temp = obs['temp'].mean() 
    std_obs_temp = obs['temp'].std()
    skew_obs_temp = skew(obs['temp'])
    kurt_obs_temp = kurtosis(obs['temp'])

    new_feat = [subzero_ind, at_amp, lathrop, glm_strat_perc, ws_sp_mix]

    for j, lab in enumerate(new_lab):
        meta.loc[lake, lab] = new_feat[j]

meta.reset_index(inplace=True)
meta.to_feather("../../../metadata/lake_metadata_wStats3.feather")