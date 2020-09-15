import pandas as pd
import numpy as np
import re
######################################################################3
# (Jared - Nov 2019) parse morphometry from USGS data release into geometry files for PGDL
##########################################################################33
#new lake names
# obs_df = pd.read_feather("../../../data/raw/additional_lakes/temperature_obs.feather")
# ids = np.unique(obs_df['nhdhr_id'].values)
ids = pd.read_csv("../../../metadata/sites_moreThan10ProfilesWithGLM.csv")
ids = ids['site_id'].values
# matches = [re.search('nhdhr_(.*)', i) for i in ids]
# ids = [m.group(1) for m in matches]
ct = 0
for nid in ids: #for each new additional lake
    i_d = nid
    nid = 'nhdhr_'+nid
    if not (nid == 'nhdhr_120018008' or nid == 'nhdhr_120020307' or nid == 'nhdhr_120020636' or nid == 'nhdhr_32671150' or nid =='nhdhr_58125241'):
        continue
    csv = []
    csv.append("depths,areas")
    m = re.search('nhdhr_(.*)', nid) 
    name = m.group(1)
    print(name)

    #load nml file
    f = open("../../../data/raw/USGS_release/nhdhr_"+name+".nml","r")
    txt = f.read()

    #parse heights and areas
    hs = re.findall('\s+H\s+=\s+(.+)',txt)
    As = re.findall('\s+A\s+=\s+(.+)',txt)
    assert len(hs) == 1
    assert len(As) == 1

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
