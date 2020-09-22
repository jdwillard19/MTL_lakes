import pandas as pd
import numpy as np
import re
import json
import pdb
######################################################################################
# (Jared - Sept 2020) parse morphometry from USGS data release into geometry files for PGDL
####################################################################################
base_path = "../../data/raw/sb_mtl_data_release/"
cfg_path = base_path+"cfg/pb0_config.json"
obs_df = pd.read_csv(base_path+"obs/temperature_observations.csv")
ids = np.unique(obs_df['site_id'].values)
ids = np.array([re.search('nhdhr_(.*)', x).group(1) for x in ids])

cfg_f = open(cfg_path)
cfg = json.load(cfg_f)

ct = 0

for nid in ids: #for each new additional lake
    ct += 1
    print("processing lake ",ct,": ",nid)
    i_d = nid
    nid = 'nhdhr_'+nid

    csv = []
    csv.append("depths,areas")
    m = re.search('nhdhr_(.*)', nid) 
    name = m.group(1)

    hs = np.array(cfg[nid]['morphometry']['H'])
    As = cfg[nid]['morphometry']['A']

    #get into posi depth form
    hs = -(np.flip(hs,axis=0) - hs.max())
    As = np.flip(As,axis=0)
    assert hs.shape[0] == As.shape[0]

    #back to str form
    hs = [str(i) for i in hs]
    As = [str(i) for i in As]

    for i in range(len(hs)):
        csv.append(",".join([hs[i], As[i]]))


    with open("../../data/processed/"+name+"/geometry",'w') as file:
        for line in csv:
            file.write(line)
            file.write('\n')
