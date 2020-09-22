import pandas as pd
import numpy as np
import os
import pdb
import sys
import math
import re
from scipy.stats import skew, kurtosis
sys.path.append('../data')
from date_operations import get_season, findZeroTempDay
from data_operations import rmse

###################################################
# (Sept 2020 - Jared) - clean up for code repo construction, reference new data releaese instead
# (June 2020 - Jared) - file create - create metadata table
###################################################

base_path = "../../data/raw/sb_mtl_data_release/"
all_obs = pd.read_csv(base_path+"obs/temperature_observations.csv")
metadata = pd.read_feather("../../metadata/lake_metadata.feather")
ids = np.unique(all_obs['site_id'].values)
ids_nop = [re.search('nhdhr_(.*)', x).group(1) for x in ids]
days_per_year = 366
usgs_meta = pd.read_csv("../../metadata/lake_metadata_from_data_release.csv")
verbose = True

new_lab = ['site_id', 'K_d', 'SDF', 'canopy', 'fullname', 'glm_uncal_rmse_third', 'glm_uncal_rmse_full',
       'latitude', 'longitude', 'max_depth', 'surface_area', 'sw_mean',
       'sw_std', 'lw_mean', 'lw_std', 'at_mean', 'at_std', 'rh_mean', 'rh_std',
       'ws_mean', 'ws_std', 'rain_mean', 'rain_std', 'snow_mean', 'snow_std',
       'sw_mean_sp', 'sw_std_sp', 'lw_mean_sp', 'lw_std_sp', 'at_mean_sp',
       'at_std_sp', 'rh_mean_sp', 'rh_std_sp', 'ws_mean_sp', 'ws_std_sp',
       'rain_mean_sp', 'rain_std_sp', 'snow_mean_sp', 'snow_std_sp',
       'sw_mean_su', 'sw_std_su', 'lw_mean_su', 'lw_std_su', 'at_mean_su',
       'at_std_su', 'rh_mean_su', 'rh_std_su', 'ws_mean_su', 'ws_std_su',
       'rain_mean_su', 'rain_std_su', 'snow_mean_su', 'snow_std_su',
       'sw_mean_au', 'sw_std_au', 'lw_mean_au', 'lw_std_au', 'at_mean_au',
       'at_std_au', 'rh_mean_au', 'rh_std_au', 'ws_mean_au', 'ws_std_au',
       'rain_mean_au', 'rain_std_au', 'snow_mean_au', 'snow_std_au',
       'sw_mean_wi', 'sw_std_wi', 'lw_mean_wi', 'lw_std_wi', 'at_mean_wi',
       'at_std_wi', 'rh_mean_wi', 'rh_std_wi', 'ws_mean_wi', 'ws_std_wi',
       'rain_mean_wi', 'rain_std_wi', 'snow_mean_wi', 'snow_std_wi', 'n_obs',
       'n_prof', 'n_obs_wi', 'n_obs_sp', 'n_obs_su', 'n_obs_au',
       'obs_depth_mean_frac', 'obs_temp_mean', 'obs_temp_std', 'obs_temp_skew',
       'obs_temp_kurt', 'zero_temp_doy', 'at_amp', 'lathrop_strat',
       'glm_strat_perc', 'ws_sp_mix']
metadata = pd.DataFrame(columns=new_lab)
# metadata = pd.read_feather("../../../metadata/lake_metadata_2700plus_temp.feather")
for i, lake in enumerate(ids):
    print("lake ", i, ": ", lake)
    name = lake
    #load meteorological data
    # if lake in metadata['site_id'].values:
    #     continue

    surf_area = usgs_meta[usgs_meta['site_id']==lake]['surface_area'].values[0]
    max_depth = usgs_meta[usgs_meta['site_id']==lake]['max_depth'].values[0]
    k_d = usgs_meta[usgs_meta['site_id']==lake]['K_d'].values[0]
    lon = usgs_meta[usgs_meta['site_id'] == lake].centroid_lon.values[0]
    lat = usgs_meta[usgs_meta['site_id'] == lake].centroid_lat.values[0]
    sdf = usgs_meta[usgs_meta['site_id'] == lake].SDF.values[0]
    fullname = usgs_meta[usgs_meta['site_id'] == lake].lake_name.values[0]
    if verbose:
      print("surf area ", surf_area)
      print("max d ", max_depth)
      print("kd ", k_d)
      print("lon: ",lon)
      print("lat: ",lat)
      print("sdf: ",sdf)

    meteo = pd.read_csv("../../data/raw/sb_mtl_data_release/meteo/"+lake+"_meteo.csv")

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

    lathrop = 1 if ((max_depth - 0.1) / np.log10(surf_area/1e4)) > 3.8 else 0

    glm = pd.read_csv("../../data/raw/sb_mtl_data_release/predictions/pb0_"+name+"_temperatures.csv")
    first_col = glm.pop('date')
    glm.insert(0, 'date', first_col)

    top_temps = glm.values[:,1]
    bottom_temps = [row[1].iloc[np.where(np.isfinite(np.array(row[1].iloc[1:].values,dtype=np.float32)))[0][-1]+1] for row in glm.iterrows()]


    glm_strat_perc = np.sum(np.abs(top_temps - bottom_temps) > 1) / top_temps.shape[0]
    print("lathrop: ",lathrop," --- sperc: ", glm_strat_perc)
    #.1899, .1541,.5367,.20398

    #45(?) day average of all wind speeds after the date of the air temperature 0 estimate
    ws_ind0 = subzero_ind
    ws_ind1 = subzero_ind + 45
    ws_doy = np.repeat(ws_per_doy, 2)
    ws_sp_mix = ws_doy[ws_ind0:ws_ind1].mean() 
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
    obs = all_obs[all_obs['site_id'] == lake]
    obs = obs[obs['depth'] <= max_depth]

    #processed obs file / feats / files
    obs1 = np.load("../../data/processed/" + ids_nop[i] + "/train.npy")
    obs2 = np.load("../../data/processed/" + ids_nop[i] + "/test.npy")
    shape0 = obs1.shape[0]
    shape1 = obs1.shape[1]
    trn_flt = obs1.flatten()
    tst_flt = obs2.flatten()
    np.put(trn_flt, np.where(np.isfinite(tst_flt))[0], tst_flt[np.isfinite(tst_flt)])
    trn_tst = trn_flt.reshape((shape0, shape1))
    obs_f = trn_tst
    # np.save("../../../data/processed/lake_data/"+lake+"/full_obs.npy", trn_tst)
    feats_f = np.load("../../data/processed/" + ids_nop[i] + "/features.npy")

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
    mean_obs_frac = obs['depth'].mean() / max_depth
    mean_obs_temp = obs['temp'].mean() 
    std_obs_temp = obs['temp'].std()
    skew_obs_temp = skew(obs['temp'])
    kurt_obs_temp = kurtosis(obs['temp'])

    #calculate glm error
    ind_to_del = []
    ind_to_del_full = []
    obs_temps_full = np.array(obs.values[:,3])
    last_tst_ind = math.floor(obs.shape[0]/3)
    obs_temps = np.array(obs.values[:last_tst_ind,3])
    glm_temps = np.empty((last_tst_ind))
    glm_temps_full = np.empty((obs_temps_full.shape[0]))
    glm_temps[:] = np.nan
    glm_temps_full[:] = np.nan
    for t in range(last_tst_ind):
        # if len(np.where(glm['DateTime'] == pd.to_datetime(obs['date'][t]).tz_localize('Etc/GMT+6'))[0]) == 0:
        if np.datetime64(pd.to_datetime(obs['date'].values[t]).tz_localize('Etc/GMT+6')).astype('datetime64[D]') < np.datetime64(pd.to_datetime(glm['date'][0]).tz_localize('Etc/GMT+6')).astype('datetime64[D]'):
            ind_to_del.append(t)
            continue
        # if len(np.where(glm['DateTime'] == np.datetime64(pd.to_datetime(obs['date'][t]).tz_localize('Etc/GMT+6')).astype('datetime64[D]'))[0]) == 0:
        row_ind = np.where(glm['date'] == obs['date'].values[t])[0][0]
        col_ind = int(obs.iloc[t].depth / 0.5) + 1
        if col_ind > glm.shape[1]-1:
            ind_to_del.append(t)
            continue
        elif math.isnan(glm.iloc[row_ind, col_ind]):
            ind_to_del.append(t)
        else:
            glm_temps[t] = glm.iloc[row_ind, col_ind]

    for t in range(obs_temps_full.shape[0]):
        # if len(np.where(glm['DateTime'] == pd.to_datetime(obs['date'][t]).tz_localize('Etc/GMT+6'))[0]) == 0:
        if np.datetime64(pd.to_datetime(obs['date'].values[t]).tz_localize('Etc/GMT+6')).astype('datetime64[D]') < np.datetime64(pd.to_datetime(glm['date'][0]).tz_localize('Etc/GMT+6')).astype('datetime64[D]'):
            ind_to_del_full.append(t)
            continue
        # if len(np.where(glm['DateTime'] == np.datetime64(pd.to_datetime(obs['date'][t]).tz_localize('Etc/GMT+6')).astype('datetime64[D]'))[0]) == 0:
        row_ind = np.where(glm['date'] == obs['date'].values[t])[0][0]
        col_ind = int(obs.iloc[t].depth / 0.5) + 1
        if col_ind > glm.shape[1]-1:
            ind_to_del_full.append(t)
            continue
        elif math.isnan(glm.iloc[row_ind, col_ind]):
            ind_to_del_full.append(t)
        else:
            glm_temps_full[t] = glm.iloc[row_ind, col_ind]

    glm_temps = np.delete(glm_temps, ind_to_del, axis=0)
    obs_temps = np.delete(obs_temps, ind_to_del, axis=0)
    glm_temps_full = np.delete(glm_temps_full, ind_to_del_full, axis=0)
    obs_temps_full = np.delete(obs_temps_full, ind_to_del_full, axis=0)
    glm_uncal_rmse_third = rmse(glm_temps, obs_temps)
    glm_uncal_rmse_full = rmse(glm_temps_full, obs_temps_full)

    new_feat = pd.Series([lake, k_d, sdf, np.nan, fullname, glm_uncal_rmse_third, glm_uncal_rmse_full, lat, lon, max_depth, surf_area, sw_m, sw_s, lw_m, lw_s, at_m, at_s, rh_m, rh_s, ws_m, ws_s, r_m, r_s, s_m, s_s, \
                sw_m_wi, sw_s_wi, lw_m_wi, lw_s_wi, at_m_wi, at_s_wi, rh_m_wi, rh_s_wi, ws_m_wi, ws_s_wi, r_m_wi, r_s_wi, s_m_wi, s_s_wi, \
                sw_m_sp, sw_s_sp, lw_m_sp, lw_s_sp, at_m_sp, at_s_sp, rh_m_sp, rh_s_sp, ws_m_sp, ws_s_sp, r_m_sp, r_s_sp, s_m_sp, s_s_sp, \
                sw_m_su, sw_s_su, lw_m_su, lw_s_su, at_m_su, at_s_su, rh_m_su, rh_s_su, ws_m_su, ws_s_su, r_m_su, r_s_su, s_m_su, s_s_su, \
                sw_m_au, sw_s_au, lw_m_au, lw_s_au, at_m_au, at_s_au, rh_m_au, rh_s_au, ws_m_au, ws_s_au, r_m_au, r_s_au, s_m_au, s_s_au, \
                int(n_obs), int(n_prof), int(n_obs_wi), int(n_obs_sp), int(n_obs_su), int(n_obs_au), mean_obs_frac, mean_obs_temp, std_obs_temp, \
                skew_obs_temp, kurt_obs_temp, subzero_ind, at_amp, lathrop, glm_strat_perc, ws_sp_mix], index=new_lab)
    metadata = metadata.append(new_feat, ignore_index=True)
    metadata.to_feather("../../metadata/lake_metadata_full.feather")