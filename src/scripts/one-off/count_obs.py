import pandas as pd
import numpy as np
import sys
import pdb
import os
import re

df = pd.read_csv("../../../data/raw/usgs_zips/obs/03_temperature_observations.csv")
unq, cts = np.unique(df['site_id'], return_counts=True)
n_dates = np.empty_like(unq)
n_dates[:] = np.nan
for ct, i_d in enumerate(unq):
	tmp = df[df['site_id'] == i_d]
	_, ct_per = np.unique(tmp['date'], return_counts=True)
	n_dates[ct] = ct_per


dates_per = np.array([n_dates[i].shape[0] for i in range(n_dates.shape[0])])
ids = unq[dates_per >= 10]
ids = np.array([re.search("nhdhr_(.*)", txt).group(1) for txt in ids])

new_ids = []
ct = 0
no_glm_ct = 0
glm_ct = 0
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
df.to_csv("../../../metadata/sites_moreThan10ProfilesWithGLM_Mar2020Update.csv")
pdb.set_trace()


