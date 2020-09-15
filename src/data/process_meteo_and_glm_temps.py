import pandas as pd
import numpy as np
import re
import pdb
import os 

obs_df = pd.read_csv("../../data/raw/sb_pgdl_data_release/obs/03_temperature_observations.csv")
# obs_df = pd.read_feather("../../data/raw/additional_lakes/temperature_obs.feather")
ids = pd.read_feather("../../metadata/sites_moreThan10Profiles.feather")
ids = np.array([re.search("nhdhr_(.*)", txt).group(1) for txt in ids['site_id'].values])
# matches = [re.search('nhdhr_(.*)', i) for i in ids]
# ids = [m.group(1) for m in matches]
ct = 0
no_glm_ct = 0
for nid in ids: #for each new additional lake
    ct += 1
    print(nid, ": ", ct, "/", ids.shape[0])
    obs = obs_df[obs_df['site_id'] == nid]
    #get temperature file
    # glm_path = "../../data/raw/lakes/glm/nhd_"+nid+"_temperatures.feather"
    glm_path = "../../data/raw/lakes/sb_pgdl_data_release/predictions/pb0_nhdhr_"+nid+"_temperatures.csv"
    # iflag_path = "../../data/raw/sb_pgdl_data_release/nhd_"+nid+"_iceflag.feather"
    iflag_path = "../../data/raw/sb_pgdl_data_release/ice_flags/pb0_nhdhr_"+nid+"_ice_flag.csv"

    # if not os.path.exists(iflag_path):
    #     print("new glm file")
    #     if not os.path.exists("../../data/raw/usgs_zips/glm/pb0_nhdhr_"+nid+"_temperatures.csv"):
    #         print("GLM NOT AVAILABLE")
    #         no_glm_ct += 1
    #         continue
    #     glm = pd.read_csv("../../data/raw/usgs_zips/glm/pb0_nhdhr_"+nid+"_temperatures.csv")
    #     glm2 = pd.read_csv("../../data/raw/usgs_zips/meteo/pb0_nhdhr_"+nid+"_ice_flag.csv")
    #     assert glm['date'].values[0] == glm2['date'].values[0]
    #     assert glm['date'].values[-1] == glm2['date'].values[-1]
    #     glm['ice'] = glm2['ice']
    #     # glm.to_feather(glm_path)
    #     glm2.to_feather(iflag_path)


    #meteo to nhdhr mapping
    meteo_path = "../../data/raw/sb_pgdl_data_release/meteo/nhdhr_"+nid+"_meteo.csv"
    if not os.path.exists(meteo_path):
        print("new meteo file")
        #load nml file
        nml_data = []
        with open('../../data/raw/usgs_zips/cfg/nhdhr_'+nid+'.nml', 'r') as file:
            nml_data = file.read().replace('\n', '')

        meteo_path_str = re.search('meteo_fl\s+=\s+\'(NLDAS.+].csv)\'', nml_data).group(1)
        meteo_f = pd.read_csv("../../data/raw/usgs_zips/meteo/"+meteo_path_str)
    #   met2id = pd.read_feather("../../data/raw/additional_lakes/meteo_to_nhdhr.feather")


    #   # #get meteo file
    #   # if nid == 'nhdhr_120018089':
    #   #   pdb.set_trace()
    #   meteo_path = "../../data/raw/additional_lakes/"+met2id[met2id['site_id'] == nid]['meteo_file'].values[0]
    #   meteo_f = pd.read_csv(meteo_path, dtype='str')
    #   meteo_f.drop(np.where(meteo_f['Rain'] == 'Rain')[0], inplace=True) #drop dud rows
    #   meteo_f.reset_index(inplace=True)
    #   meteo_f.drop(labels='index', axis=1, inplace=True)

        meteo_f.to_csv(meteo_path)



# print(ct, " counted")0