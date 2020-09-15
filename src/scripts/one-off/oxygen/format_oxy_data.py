import glob
import pandas as pd
import numpy as np
import pdb
import datetime
import sys
import os
import math
import re
import shutil
from scipy import interpolate
sys.path.append('../../../data')
from data_operations import getThermoclineVector


#white bear id
site_ids = ['120018046', '34131552', '69886156', '69886228', '69886284', '69886444', '69886510', \
            '74926457', '120018046', '120018084', '120018100', '120018114', '120019354', '120019850', \
            '120019854', '141288680', '143249470', '143249640', '143249705']
n_lakes = len(site_ids)
ct = 0
n_features = 10
n_features_meteo = 7

#get mean/std per feat
means_per_lake = np.zeros((n_lakes,10), dtype=np.float_)
means_per_lake[:] = np.nan
var_per_lake = np.zeros((n_lakes,10),dtype=np.float_)
var_per_lake[:] = np.nan

#get folder name

for lake_ind, name in enumerate(site_ids):
    site_id = name

    #directory stuff one-off
    raw_dir_path = glob.glob('../../../../../lake_oxygen_analysis/data/raw/'+site_id+'*')[0]
    dir_name = re.search('.*/raw/(.*)', raw_dir_path).group(1)
    proc_dir_path = '../../../../../lake_oxygen_analysis/data/processed/'+dir_name
    if not os.path.exists(proc_dir_path):
        os.mkdir(proc_dir_path)
    print("pre ", name)
    ############################################
    #read/format meteorological data for numpy
    #############################################
    if not os.path.exists(raw_dir_path+'/nhdhr_'+name+'_meteo.csv'):
        shutil.copyfile('../../../../data/raw/lakes/meteo/nhd_'+name+'_meteo.csv', raw_dir_path+'/nhdhr_'+name+'_meteo.csv')
    meteo_dates = np.loadtxt(raw_dir_path+'/nhdhr_'+name+'_meteo.csv', delimiter=',', dtype=np.string_ , usecols=1)


    #lower/uppur cutoff indices (to match observations)
    if not os.path.exists(raw_dir_path+'/nhdhr_'+name+'_obs.feather'):
        shutil.copyfile('../../../../data/raw/lakes/obs/nhd_'+name+'_test_train.feather', raw_dir_path+'/nhdhr_'+name+'_obs.feather')
    obs = pd.read_feather(raw_dir_path+'/nhdhr_'+name+'_obs.feather')

    obs['date2'] = pd.to_datetime(obs.date)
    obs.sort_values('date2', inplace=True)
    # print(validate(obs.values[0,1]))
    start_date = []
    end_date = []
    try:
        start_date = "{:%Y-%m-%d}".format(obs.values[0,1]).encode()
    except:
        start_date = obs.values[0,1].encode()
    try:
        end_date = "{:%Y-%m-%d}".format(obs.values[-1,1]).encode()
    except:
        end_date = obs.values[-1,1].encode()
    lower_cutoff = np.where(meteo_dates == start_date)[0][0] #457
    if len(np.where(meteo_dates == end_date)[0]) < 1: 
        print("observation beyond meteorological data! data will only be used up to the end of meteorological data")
        upper_cutoff = meteo_dates.shape[0]
    else:
        upper_cutoff = np.where(meteo_dates == end_date)[0][0]+1 #14233

    meteo_dates = meteo_dates[lower_cutoff:upper_cutoff]


    #read from file and filter dates
    meteo = np.genfromtxt('../../../../data/raw/lakes/meteo/nhd_'+name+'_meteo.csv', delimiter=',', usecols=(2,3,4,5,6,7,8), skip_header=1)
    meteo = meteo[lower_cutoff:upper_cutoff,:]
    means_per_lake[lake_ind,1:8] = [meteo[:,a].mean() for a in range(n_features_meteo)]
    var_per_lake[lake_ind,1:8] = [meteo[:,a].std() ** 2 for a in range(n_features_meteo)]

    if not os.path.exists(raw_dir_path+'/nhdhr_'+name+'_temperatures.feather'):
        shutil.copyfile('../../../../data/raw/lakes/glm/nhd_'+name+'_temperatures.feather', raw_dir_path+'/nhdhr_'+name+'_temperatures.feather')
    glm_temps = pd.read_feather('../../../../data/raw/lakes/glm/nhd_'+name+'_temperatures.feather')
    
    if not os.path.exists(raw_dir_path+'/nhdhr_'+name+'_pgml_temperatures.feather'):
        shutil.copyfile('../../../../results/single_lake_outputs/'+site_id+'/PGRNN_basic_normAllGr10', raw_dir_path+'/nhdhr_'+name+'_pgml_temperatures.feather')
    
    pg_temps = pd.read_feather(raw_dir_path+'/nhdhr_'+name+'_pgml_temperatures.feather').values[:,1:]
    glm_temps = glm_temps.values[:,1:-1]
    means_per_lake[lake_ind,-2] = glm_temps.mean()
    means_per_lake[lake_ind,-1] = pg_temps.mean()
    var_per_lake[lake_ind,-2] = glm_temps.std() ** 2
    var_per_lake[lake_ind,-1] = pg_temps.std() ** 2
    n_total_dates = glm_temps.shape[0]

    #define depths from glm file
    n_depths = glm_temps.shape[1]-2 #minus date and ice flag
    max_depth = 0.5*(n_depths-1)
    depths = np.arange(0, max_depth+0.5, 0.5)
    depths_mean = depths.mean()
    depths_var = depths.std() ** 2
    means_per_lake[lake_ind, 0] = depths_mean
    var_per_lake[lake_ind, 0] = depths_var


mean_feats = np.average(means_per_lake, axis=0)   
std_feats = np.average(var_per_lake ** (.5), axis=0)   
print(mean_feats)
print(std_feats)
sys.exit()

for lake_ind, name in enumerate(site_ids):

    ############################################################################


    pm_data = pd.read_csv("../../../../data/raw/oxy/White_Bear_1.5_oxymodel.txt", sep='\t')

    #####################################################################
    #format process model data


    #format pandas file
    pm_data.columns = ['long_date', 'o2_epi', 'o2_hypo', 'o2_total', 'vol_epi', 'vol_hypo', 'vol_total']
    pm_data.sort_values('long_date', inplace=True)
    pm_data['date'] = [pd.to_datetime(pm_data.iloc[i,0][:10]) for i in range(pm_data.shape[0])] 


    ###############################################################
    #format pgml and glm data



    data = pd.read_csv("../../../../data/raw/oxy/White_Bear_obs.txt", sep='\t')

    data.columns = ['long_date', 'time', 'depth', 'o2']
    data.sort_values('long_date',inplace=True)

    #simplify to date without time
    data['date'] = [pd.to_datetime(data.iloc[i,0][:10]) for i in range(data.shape[0])] 

    #round to half meter depths
    data.depth = (data.depth * 2).round() / 2

    #coalesce same day/depth observations
    data = data.groupby(['date', 'depth']).mean()

    #reindex
    data.reset_index(inplace=True)

    n_features = 7

    meteo_f = pd.read_csv('../../../../data/raw/lakes/meteo/nhd_'+site_id+"_meteo.csv")
    first_meteo_date = datetime.datetime.strptime(meteo_f['time'].values[0], '%Y-%m-%d').date()
    last_meteo_date = datetime.datetime.strptime(meteo_f['time'].values[-1], '%Y-%m-%d').date()

    #filter out data with no corresponding meteorological data
    data = data[data['date'] >= pd.Timestamp(first_meteo_date)]
    data = data[data['date'] <= pd.Timestamp(last_meteo_date)]

    obs_o = data
    obs_t = pd.read_feather('../../../../data/raw/lakes/obs/nhd_'+site_id+"_test_train.feather")

    ############################################
    #read/format meteorological data for numpy
    #############################################
    # meteo_dates = np.loadtxt('../../data/raw/figure3/nhd_'+site_id+'_meteo.csv', delimiter=',', dtype=np.string_ , usecols=0)
    meteo_dates = np.loadtxt('../../../../data/raw/lakes/meteo/nhd_'+site_id+'_meteo.csv', delimiter=',', dtype=np.string_ , usecols=1, skiprows=1)


    obs_t['date2'] = pd.to_datetime(obs_t.date)
    obs_t.sort_values('date2', inplace=True)
    # print(validate(obs.values[0,1]))

    start_date = []
    end_date = []

    if obs_t.date.values[0] < datetime.datetime.utcfromtimestamp(obs_o.date.values[0].astype('O')/1e9).date():
        start_date = obs_o.date.values[0]
    else:
        start_date = obs_t.date.values[0]


    if obs_t.date.values[-1] < datetime.datetime.utcfromtimestamp(obs_o.date.values[-1].astype('O')/1e9).date():
        end_date = obs_t.date.values[-1]
    else:
        end_date = obs_o.date.values[-1]

    start_date = np.datetime64(start_date)
    end_date = np.datetime64(end_date)
    start_date = np.datetime_as_string(start_date)[:10].encode()
    end_date = np.datetime_as_string(end_date)[:10].encode()


    lower_cutoff = np.where(meteo_dates == start_date)[0][0] #457
    if len(np.where(meteo_dates == end_date)[0]) < 1: 
        print("observation beyond meteorological data! data will only be used up to the end of meteorological data")
        upper_cutoff = meteo_dates.shape[0]
    else:
        upper_cutoff = np.where(meteo_dates == end_date)[0][0]+1 #14233

    meteo_dates = meteo_dates[lower_cutoff:upper_cutoff]






    #read from file and filter dates
    # meteo = np.genfromtxt('../../data/raw/figure3/nhd_'+site_id+'_meteo.csv', delimiter=',', usecols=(1,2,3,4,5,6,7))
    meteo = np.genfromtxt('../../../../data/raw/lakes/meteo/nhd_'+site_id+'_meteo.csv', delimiter=',', usecols=(2,3,4,5,6,7,8), skip_header=1)
    meteo = meteo[lower_cutoff:upper_cutoff,:]

    #normalize data
    meteo_means = [meteo[:,a].mean() for a in range(n_features)]
    meteo_std = [meteo[:,a].std() for a in range(n_features)]
    meteo_norm = (meteo - meteo_means[:]) / meteo_std[:]
    # meteo_norm = (meteo - me[1:]) / std_feats[1:]

    #meteo = final features sans depth
    #meteo_norm = normalized final features sans depth

    ################################################################################
    # read/format GLM temperatures and observation data for numpy
    ###################################################################################

    glm_temps = pd.read_feather('../../../../data/raw/lakes/glm/nhd_'+site_id+'_temperatures.feather')
    glm_temps = glm_temps.values[:]
    n_total_dates = glm_temps.shape[0]

    #define depths from glm file
    n_depths = glm_temps.shape[1]-2 #minus date and ice flag
    # print("n_depths: " + str(n_depths))
    max_depth = 0.5*(n_depths-1)
    depths = np.arange(0, max_depth+0.5, 0.5)
    # depths_normalized = np.divide(depths - mean_feats[0], std_feats[0])
    depths_normalized = np.divide(depths - depths.mean(), depths.std())


    #format date to string to match
    try:
        glm_temps[:,0] = np.array([glm_temps[a,0].strftime('%Y-%m-%d') for a in range(n_total_dates)]) 
    except:
        temp = 0
    if len(np.where(glm_temps[:,0] == start_date.decode())[0]) < 1:
        print("shouldnt happen, start date not in glm?")
        sys.exit()
        # print("observations begin at " + start_date.decode() + "which is before GLM data which begins at " + glm_temps[0,0])
        # lower_cutoff = 0
        # new_meteo_lower_cutoff = np.where(meteo_dates == glm_temps[0,0].encode())[0][0]
        # meteo = meteo[new_meteo_lower_cutoff:,:]
        # meteo_norm = meteo_norm[new_meteo_lower_cutoff:,:]
        # meteo_dates = meteo_dates[new_meteo_lower_cutoff:]
    else:
        lower_cutoff = np.where(glm_temps[:,0] == start_date.decode())[0][0] 

    if len(np.where(glm_temps[:,0] == end_date.decode())[0]) < 1: 
        print("shouldnt happen, end date not in glm?")
        sys.exit()
        # print("observations extend to " + end_date.decode() + "which is beyond GLM data which extends to " + glm_temps[-1,0])
        # upper_cutoff = glm_temps[:,0].shape[0]
        # new_meteo_upper_cutoff = np.where(meteo_dates == glm_temps[-1,0].encode())[0][0]
        # meteo = meteo[:new_meteo_upper_cutoff+1,:]
        # meteo_norm = meteo_norm[:new_meteo_upper_cutoff+1,:]
        # meteo_dates = meteo_dates[:new_meteo_upper_cutoff+1]


    else:
        upper_cutoff = np.where(glm_temps[:,0] == end_date.decode())[0][0] 


    # upper_cutoff = np.where(glm_temps[:,0] == end_date.decode())[0][0]

    glm_temps = glm_temps[lower_cutoff:upper_cutoff+1,:]
    n_dates = glm_temps.shape[0]

    if n_dates != meteo.shape[0]:
        print(n_dates)
        print(meteo.shape[0])
    assert n_dates == meteo.shape[0]
    assert n_dates == meteo_norm.shape[0]
    assert n_dates == glm_temps.shape[0]
    #assert dates line up
    assert(glm_temps[0,0] == meteo_dates[0].decode())

    if glm_temps[-1,0] != meteo_dates[-1].decode():
        print(glm_temps[-1,0])
        print(meteo_dates[-1].decode())

    assert(glm_temps[-1,0] == meteo_dates[-1].decode())

    ice_flag = glm_temps[:,-1] 
    glm_temps = glm_temps[:,1:-1]
    n_obs = obs_o.shape[0]


    pg_temps = pd.read_feather('../../../../results/single_lake_outputs/'+site_id+'/PGRNN_basic_normAllGr10')
    new_lower_cut = 0
    new_upper_cut = meteo_dates.shape[0]
    to_cut = False

    pg_temps['date2'] = [str(pg_temps.date[i])[:10] for i in range(pg_temps.shape[0])]
    if pd.Timestamp(pg_temps['date2'].values[0]) > pd.Timestamp(start_date.decode()):
        new_lower_cut = np.where(meteo_dates == pg_temps['date2'].values[0].encode())[0][0]
        to_cut = True

    if pd.Timestamp(pg_temps['date2'].values[-1]) < pd.Timestamp(end_date.decode()):
        new_upper_cut = np.where(meteo_dates == pg_temps['date2'].values[-1].encode())[0][0]
        to_cut = True
    if to_cut:
        meteo = meteo[new_lower_cut:new_upper_cut,:]
        meteo_norm = meteo_norm[new_lower_cut:new_upper_cut,:]
        meteo_dates = meteo_dates[new_lower_cut:new_upper_cut]
        glm_temps = glm_temps[new_lower_cut:new_upper_cut,:]
        ice_flag = ice_flag[new_lower_cut:new_upper_cut]
        start_date = meteo_dates[0]
        end_date = meteo_dates[-1]





    if len(np.where(pg_temps['date2'] == start_date.decode())[0]) < 1:
        print("shouldnt happen, start date not in pg?")
        sys.exit()

    else:
        lower_cutoff = np.where(pg_temps['date2'] == start_date.decode())[0][0] 

    if len(np.where(pg_temps['date2'] == end_date.decode())[0]) < 1: 
        print("shouldnt happen, end date not in pg?")
        sys.exit()

    else:
        upper_cutoff = np.where(pg_temps['date2'] == end_date.decode())[0][0] 

    pg_temps = pg_temps[lower_cutoff:upper_cutoff+1][:]

    #remove date cols
    pg_temps = pg_temps.values[:,1:-1]

    #remove nan rows from pg_temps
    pg_temps = np.array(pg_temps, dtype=np.float64)
    rows_to_del = np.unique(np.where(~np.isfinite(pg_temps))[0])
    if len(rows_to_del) > 1:
        pg_temps = np.delete(pg_temps, rows_to_del, axis=0)
        glm_temps = np.delete(glm_temps, rows_to_del, axis=0)
        meteo = np.delete(meteo, rows_to_del, axis=0)
        meteo_norm = np.delete(meteo_norm, rows_to_del, axis=0)
        meteo_dates = np.delete(meteo_dates, rows_to_del, axis=0)
        ice_flag = np.delete(ice_flag, rows_to_del, axis=0)
        start_date = meteo_dates[0]
        end_date = meteo_dates[-1]


    assert glm_temps.shape == pg_temps.shape
    assert meteo.shape[0] == pg_temps.shape[0]
    assert meteo_dates.shape[0] == pg_temps.shape[0]
    n_dates = pg_temps.shape[0]
    ############################################################
    #fill numpy matrices
    ##################################################################
    feat_mat = np.empty((n_depths, n_dates, n_features+4)) #[depth->7 meteo features-> ice flag]
    feat_mat[:] = np.nan
    feat_norm_mat = np.empty((n_depths, n_dates, n_features+3)) #[standardized depth -> 7 std meteo features -> glm -> pgdl]
    feat_norm_mat[:] = np.nan
    glm_mat = np.empty((n_depths, n_dates))
    glm_mat[:] = np.nan
    obs_trn_mat = np.empty((n_depths, n_dates))
    obs_trn_mat[:] = np.nan
    obs_tst_mat = np.empty((n_depths, n_dates))
    obs_tst_mat[:] = np.nan
    pretrn_mat = np.empty((n_depths, n_dates))
    pretrn_mat[:] = np.nan
    # print("n depths: " + str(n_depths))
    for d in range(n_depths):
        feat_mat[d,:,0] = depths[d]
        feat_norm_mat[d,:,0] = depths_normalized[d]
        glm_mat[d,:] = glm_temps[:,d]
        feat_mat[d,:,1:-3] = meteo[:]
        feat_mat[d,:,-3] = ice_flag[:]
        feat_norm_mat[d,:,1:-2] = meteo_norm[:]



    if True:
        if np.isnan(np.sum(glm_mat)):
            # print("Warning: there is missing data in glm output")
            for i in range(n_depths):
                for t in range(n_dates):
                    if np.isnan(glm_mat[i,t]):
                        x = depths[i]
                        xp = depths[0:(i)]
                        yp = glm_mat[0:(i),t]
                        if xp.shape[0] == 1:
                            glm_mat[i,t] = glm_mat[i-1,t]
                        else:
                            f = interpolate.interp1d(xp, yp,  fill_value="extrapolate")
                            glm_mat[i,t] = f(x) #interp_temp

            assert not np.isnan(np.sum(glm_mat))


    feat_norm_mat[:,:,-2] = (glm_mat[:] - glm_mat.mean()) / glm_mat.std()
    feat_mat[:,:,-2] = glm_mat[:] 
    feat_norm_mat[:,:,-1] = np.transpose((pg_temps[:] - pg_temps.mean()) / pg_temps.std())
    feat_mat[:,:,-1] = np.transpose(pg_temps[:])

    #verify all mats filled
    if np.isnan(np.sum(feat_mat)):
        print("ERROR: Preprocessing failed, there is missing data feat")
        sys.exit()
    if np.isnan(np.sum(feat_norm_mat)):
        print("ERROR: Preprocessing failed, there is missing data feat norm")
        sys.exit() 
    # print(depths)
    obs_g = 0
    obs_d = 0

    #get unique observation days
    obs_o = obs_o.values
    obs_o = obs_o[obs_o[:,0] > pd.Timestamp(start_date.decode()),:]
    n_obs = obs_o.shape[0]

    unq_obs_dates = np.unique(obs_o[:,0])
    n_unq_obs_dates = unq_obs_dates.shape
    first_tst_date = obs_o[0,0]
    last_tst_date = obs_o[math.floor(obs_o.shape[0]/3),0]
    last_tst_obs_ind = np.where(obs_o[:,0] == last_tst_date)[0][-1]

    n_tst = last_tst_obs_ind + 1
    n_trn = obs_o.shape[0] - n_tst

    last_train_date = obs_o[-1,0]
    first_train_date = obs_o[last_tst_obs_ind + 1,0]
    #test data
    n_tst_obs_placed = 0
    n_trn_obs_placed = 0


    for o in range(0,last_tst_obs_ind+1):
        #verify data in depth range
        if obs_o[o,1] > depths[-1]:
            obs_g += 1
            # print("observation depth " + str(obs[o,1]) + " is greater than the max depth of " + str(max_depth))
            continue
        if len(np.where(meteo_dates == str(obs_o[o,0])[:10].encode())[0]) < 1:
            obs_d += 1
            continue
        depth_ind = np.where(depths == obs_o[o,1])[0][0]
        date_ind = np.where(meteo_dates == str(obs_o[o,0])[:10].encode())[0][0]
        obs_tst_mat[depth_ind, date_ind] = obs_o[o,2]
        n_tst_obs_placed += 1

    #train data
    for o in range(last_tst_obs_ind+1, n_obs):
        if obs_o[o,1] > depths[-1]:
            obs_g += 1
            # print("observation depth " + str(obs[o,1]) + " is greater than the max depth of " + str(max_depth))
            continue
        depth_ind = np.where(depths == obs_o[o,1])[0][0]
        if len(np.where(meteo_dates == str(obs_o[o,0])[:10].encode())[0]) < 1:
            obs_d += 1
            continue

        depth_ind = np.where(depths == obs_o[o,1])[0][0]
        date_ind = np.where(meteo_dates == str(obs_o[o,0])[:10].encode())[0][0]

        obs_trn_mat[depth_ind, date_ind] = obs_o[o,2]
        n_trn_obs_placed += 1



    ###################################################################################
    # format process model output data into [depth]*[day] dimensions

    #filter out process model data
    pm_data = pm_data[pm_data['date'] >= pd.Timestamp(meteo_dates[0].decode())]
    pm_data = pm_data[pm_data['date'] <= pd.Timestamp(meteo_dates[-1].decode())]
    pd_meteo_dates = [pd.Timestamp(x.decode()) for x in meteo_dates]
    ind = 0
    for i, row in pm_data.iterrows():
        o2_epi = row['o2_epi'] * 1e-3 / row['vol_epi']
        o2_hypo = row['o2_hypo'] * 1e-3 / row['vol_hypo']

        #date 

        pm_date = row['date']

        # pg_t_col = pg_temps[ind,:]
        glm_t_col = glm_temps[ind,:]
        thermo_ind = getThermoclineVector(depths, glm_t_col)

        pretrn_mat[:thermo_ind,ind] = o2_epi 
        pretrn_mat[thermo_ind:,ind] = o2_hypo
        ind += 1
        #find thermocline
    #############################################################################################3


    d_str = ""
    if obs_d > 0:
        d_str = ", and "+str(obs_d) + " observations outside of combined date range of meteorological and GLM output"
    # if obs_g > 0 or obs_d > 0:
        # continue
    print("lake " + str(ct) + ",  id: " + site_id + ": " + str(obs_g) + "/" + str(n_obs) + " observations greater than max depth " + str(max_depth) + d_str)
    #write features and labels to processed data
    print("training: ", first_train_date, "->", last_train_date, "(", n_trn, ")")
    print("testing: ", first_tst_date, "->", last_tst_date, "(", n_tst, ")")
    if not os.path.exists("../../../../data/processed/o2_lake_data/"+site_id): 
        os.mkdir("../../../../data/processed/o2_lake_data/"+site_id)
    # if not os.path.exists("../../../../models/"+site_id):
        # os.mkdir("../../../../models/"+site_id)
    feat_path = "../../../../data/processed/o2_lake_data/"+site_id+"/features"
    norm_feat_path = "../../../../data/processed/o2_lake_data/"+site_id+"/processed_features"
    # glm_path = "../../../../data/processed/o2_lake_data/"+site_id+"/glm"
    trn_path = "../../../../data/processed/o2_lake_data/"+site_id+"/train"
    tst_path = "../../../../data/processed/o2_lake_data/"+site_id+"/test"
    full_path = "../../../../data/processed/o2_lake_data/"+site_id+"/full"
    pm_path = "../../../../data/processed/o2_lake_data/"+site_id+"/pretrain"

    dates_path = "../../../../data/processed/o2_lake_data/"+site_id+"/dates"
    #geometry
    # shutil.copyfile('../../data/raw/figure3/nhd_'+site_id+'_geometry.csv', "../../data/processed/WRR_69Lake/"+site_id+"/geometry")
    np.save(feat_path, feat_mat)
    np.save(norm_feat_path, feat_norm_mat)
    # np.save(glm_path, glm_mat)
    np.save(dates_path, meteo_dates)
    np.save(trn_path, obs_trn_mat)
    np.save(tst_path, obs_tst_mat)
    np.save(pm_path, pretrn_mat)

    full = obs_trn_mat
    full[np.nonzero(np.isfinite(obs_tst_mat))] = obs_tst_mat[np.isfinite(obs_tst_mat)]
    np.save(full_path, full)
