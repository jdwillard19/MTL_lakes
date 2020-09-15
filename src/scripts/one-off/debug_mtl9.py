import pandas as pd
import numpy as np
import pdb
import re


df = pd.read_csv("pgmtl9_debug.csv")
site_ids = df['target_id'].values



nldas = []
most_common_source = []
obs = pd.read_csv("../../../data/raw/sb_pgdl_data_release/obs/03_temperature_observations.csv")

ct = 0
for i_d in site_ids:
    ct += 1
    print(ct, "/",len(site_ids))
    #get nldas
    nml_data = []
    with open('../../../data/raw/sb_pgdl_data_release/cfg/'+i_d+'.nml', 'r') as file:
        nml_data = file.read().replace('\n', '')

    meteo_path_str = re.search('meteo_fl\s+=\s+\'(NLDAS.+].csv)\'', nml_data).group(1)
    nldas.append(meteo_path_str)
    obs_i = obs[obs['site_id'] == i_d]
    most_common_source.append(obs_i['source'].value_counts().idxmax())

df['nldas'] = nldas
df['most_common_source'] = most_common_source

df.to_csv("pgmtl9_debug_res.csv")
