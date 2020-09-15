import pandas as pd
import numpy as np
import re
import pdb
import os 
import pdb

obs_df = pd.read_csv("../../data/raw/sb_pgdl_data_release/obs/temperature_observations.csv")

site_df = pd.read_csv("../../metadata/sites_moreThan10ProfilesWithGLM_Mar2020Update.csv")
# ids = pd.read_csv("../../metadata/sites_moreThan10ProfilesWithGLM_Mar2020Update.csv")['site_id'].values
ids2 = ['nhdhr_'+str(x) for x in pd.read_csv("../../metadata/sites_moreThan10ProfilesWithGLM_June2020Update.csv")['site_id'].values]
ids = np.unique(obs_df['site_id'].values)
pdb.set_trace()
to_remove = []
for i,i_d in enumerate(ids):
	n_obs = obs_df[obs_df['site_id'] == i_d].values.shape[0]
	if n_obs < 10:
		to_remove.append(i_d)

	print("(",i,"/",len(ids),"): ", n_obs)
print(repr(to_remove))
print(len(to_remove))
sys.exit()
to_remove = ['109981594', '109989488', '123398097', '123398250', '137421002', '143250706', '149094360', '152372580', '157362786', '47484399', '47484724', '69546663', '70330649', '70331069', '75660023', '75661647', '76156225', '91681409']
to_remove = np.array(to_remove)
print(site_df.shape)
site_df.drop(np.where(np.isin(ids,to_remove))[0], inplace=True)
print(site_df.shape)
site_df.to_csv("../../metadata/sites_moreThan10ProfilesWithGLM_June2020Update.csv")