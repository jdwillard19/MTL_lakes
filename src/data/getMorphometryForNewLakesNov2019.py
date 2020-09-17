import pandas as pd
import numpy as np
import re
import json
######################################################################################
# (Jared - Sept 2020) parse morphometry from USGS data release into geometry files for PGDL
####################################################################################
ids = pd.read_csv("../../metadata/lake_metadata.feather")
cfg_path = "../../data/raw/sb_mtl_data_release/pb0_config.json"
ids = ids['site_id'].values


ct = 0
cfg = json.loads(jsondata)

for nid in ids: #for each new additional lake
    i_d = nid
    nid = 'nhdhr_'+nid

    csv = []
    csv.append("depths,areas")
    m = re.search('nhdhr_(.*)', nid) 
    name = m.group(1)
    print(name)

    #load nml file
    f = open("../../../lake_modeling/data/raw/USGS_release/nhdhr_"+name+".nml","r")
    txt = f.read()

    #parse heights and areas
    hs = re.findall('\s+H\s+=\s+(.+)',txt)
    As = re.findall('\s+A\s+=\s+(.+)',txt)
    assert len(hs) == 1
    assert len(As) == 1

    cfg = json.load(cfg_path)
    pdb.set_trace()
    #list2elem
    hs = hs[0]
    As = As[0]

    #str2float
    hs = np.array([float(i) for i in hs.split(",")])
    As = np.array([float(i) for i in As.split(",")])

    #get into posi depth form
    hs = -(np.flip(hs,axis=0) - hs.max())
    As = np.flip(As,axis=0)
    assert hs.shape[0] == As.shape[0]

    #back to str form
    hs = [str(i) for i in hs]
    As = [str(i) for i in As]

    for i in range(len(hs)):
        csv.append(",".join([hs[i], As[i]]))


    with open("../../../data/processed/lake_data/"+name+"/geometry",'w') as file:
        for line in csv:
            file.write(line)
            file.write('\n')
