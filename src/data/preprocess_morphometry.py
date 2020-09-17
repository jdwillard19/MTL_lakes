import pandas as pd
import numpy as np
import re
import json
import pdb
######################################################################################
# (Jared - Sept 2020) parse morphometry from USGS data release into geometry files for PGDL
####################################################################################
ids = pd.read_feather("../../metadata/lake_metadata.feather")
cfg_path = "../../data/raw/sb_mtl_data_release/cfg/pb0_config.json"
ids = ids['site_id'].values

cfg_f = open(cfg_path)
cfg = json.load(cfg_f)

ct = 0

for nid in ids: #for each new additional lake
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
