import pandas as pd
import numpy as np
import pdb
import re
import os

# obs_df = pd.read_feather("../../../data/raw/additional_lakes/temperature_obs.feather")
obs_df = pd.read_csv("../../../data/raw/usgs_zips/obs/03_temperature_observations.csv")
ids = pd.read_feather("../../../metadata/sites_moreThan10Profiles.feather")
ids = np.array([re.search("nhdhr_(.*)", txt).group(1) for txt in ids['site_id'].values])

# matches = [re.search('nhdhr_(.*)', i) for i in ids]
# ids = [m.group(1) for m in matches]
ct = 0
new_depths = [np.round(d*2)/2 for d in obs_df.depth]
obs_df = obs_df.drop('depth', axis=1)
obs_df['depth'] = new_depths
for nid in ids: #for each new additional lake
	ct += 1
	print(ct, "/", ids.shape[0])
	# if os.path.exists("../../../data/raw/lakes/obs/nhd_"+nid+"obs.feather"):
	# 	print("already exists")
	# 	continue
	# else:

	obs = obs_df[obs_df['site_id'] == "nhdhr_"+nid]
	obs.sort_values('date', axis=0, ascending=True, inplace=True, kind='quicksort', na_position='last', ignore_index=False)
	obs.reset_index(inplace=True)
	obs.drop("index", axis=1, inplace=True)
	obs.to_feather("../../../data/raw/lakes/obs/nhd_"+nid+"obs.feather")

