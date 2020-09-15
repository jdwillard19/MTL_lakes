import pandas as pd
import numpy as np
import re
import pdb
import os 

# obs_df = pd.read_csv("../../../data/raw/usgs_zips/obs/03_temperature_observations.csv")
# obs_df = pd.read_feather("../../../data/raw/additional_lakes/temperature_obs.feather")
ids = pd.read_feather("../../../metadata/sites_moreThan10Profiles.feather")
ids = np.array([re.search("nhdhr_(.*)", txt).group(1) for txt in ids['site_id'].values])
# matches = [re.search('nhdhr_(.*)', i) for i in ids]
# ids = [m.group(1) for m in matches]
ct = 0
no_glm_ct = 0
glm_ct = 0
print(ids.shape[0], "lakes before")
new_ids = []
for nid in ids: #for each new additional lake
    ct += 1
    # print(nid, ": ", ct, "/", ids.shape[0])
    # obs = obs_df[obs_df['site_id'] == nid]
    #get temperature file
    glm_path = "../../../data/raw/lakes/glm/nhd_"+nid+"_temperatures.feather"

    if not os.path.exists("../../../data/raw/usgs_zips/glm/pb0_nhdhr_"+nid+"_temperatures.csv"):
        print("GLM NOT AVAILABLE")
        no_glm_ct += 1
        continue
    else:
        glm_ct += 1
        new_ids.append(nid)


print("no ", no_glm_ct)
print("ys ", glm_ct)
df = pd.DataFrame()
df['site_id'] = np.array(new_ids)
df.to_csv("../../../metadata/sites_moreThan10ProfilesWithGLM.csv")


# print(ct, " counted")