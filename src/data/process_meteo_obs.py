import pandas as pd
import numpy as np
import re
import pdb
import os 


base_path =  "../../data/raw/sb_mtl_data_release/"
obs_df = pd.read_csv(base_path+"obs/temperature_observations.csv")
metadata = pd.read_csv(base_path+"lake_metadata.csv")
ids = np.unique(obs_df['site_id'].values)
ids = np.array([re.search('nhdhr_(.*)', x).group(1) for x in ids])
# ids_f = pd.DataFrame()
# ids_f['site_id'] = [re.search('nhdhr_(.*)', x).group(1) for x in ids]
# ids_f.to_csv("../../metadata/sites_moreThan10ProfilesWithGLM_June2020Update.csv")
# sys.exit()
# obs_df = pd.read_feather("../../data/raw/additional_lakes/temperature_obs.feather")
# ids_f = pd.read_csv("../../metadata/sites_moreThan10ProfilesWithGLM_June2020Update.csv")
# print(ids_f.shape)
# ids = ids_f['site_id'].values
# ids = np.array([re.search("nhdhr_(.*)", txt).group(1) for txt in ids['site_id'].values])

# matches = [re.search('nhdhr_(.*)', i) for i in ids]
# ids = [m.group(1) for m in matches]
ct = 0
# no_glm_ct = 0
for nid in ids: #for each new additional lake
    ct += 1
    print("processing ",nid, ": ", ct, "/", ids.shape[0])
    obs = obs_df[obs_df['site_id'] == nid]
    iflag_path = base_path+ "ice_flags/pb0_nhdhr_"+nid+"_ice_flag.csv"


    meteo_path = base_path+"meteo/nhdhr_"+nid+"_meteo.csv"
    
    if not os.path.exists(meteo_path):
        print("new meteo file")
        meteo_fn = metadata[metadata['site_id'] == "nhdhr_"+nid]['meteo_filename'].values[0]
        meteo_f = pd.read_csv(base_path+"meteo/"+meteo_fn, dtype='str')
        meteo_f.drop(np.where(meteo_f['Rain'] == 'Rain')[0], inplace=True) #drop dud rows
        meteo_f.reset_index(inplace=True)
        meteo_f.to_csv(meteo_path)


    #obs
    obs_path = base_path+"obs/nhdhr_"+nid+"_obs.feather"
    if not os.path.exists(obs_path):
        print("new obs file")
        obs = obs_df[obs_df['site_id'] == "nhdhr_"+nid]
        if obs.shape[0] == 0:
            print("removing ",nid)
            ct -= 1
            # ids_f = ids_f[ids_f.site_id != nid]
            continue
        obs.sort_values('date', axis=0, ascending=True, inplace=True, kind='quicksort', na_position='last', ignore_index=False)
        obs.reset_index(inplace=True)
        obs.drop("index", axis=1, inplace=True)
        obs.to_feather(obs_path)

print(ct)