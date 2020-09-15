import pandas as pd
import numpy as np
import pdb
import re

conv = pd.read_feather("../../../metadata/nhd_to_nhdhr.feather")
meta = pd.read_feather("../../../metadata/lake_metadata_wStats3.feather")
usgs = pd.read_csv("../../../metadata/usgs_metadata.csv")
conv['WRR_NUM'] = [re.search(r'nhd_(\d+)', x).group(1) for x in conv['WRR_ID']]
df = pd.read_feather("../../../data/processed/lake_splits/trainTestNewLakes.feather")
df['nhdhr_id'] = df['id']
n_conv = 0
for ind, row in df.iterrows():
	lf = 'nhd_' + row['id']
	if np.isin(lf, conv['WRR_ID']):
		#if wrr_id, convert
		df.loc[row.name,'nhdhr_id'] = conv[conv['WRR_ID'] == lf]['site_id'].values[0]
		n_conv += 1
	else:
		df.loc[row.name,'nhdhr_id'] = 'nhdhr_'+ row['id']


print(n_conv)
df_tst = pd.DataFrame(df[df['isTrain']]['nhdhr_id'])
df_tst.reset_index(inplace=True)
pdb.set_trace()
df_tst.to_feather("./TrainLakes.feather")
# for row in meta.iterrows():
# 	if row[1]['fullname'] == None:
# 		site_id = 'nhdhr_' + str(row[1]['nhd_id'])
# 		name = usgs[usgs['site_id'] == site_id]['lake_name'].values[0]
# 		meta.at[row[0], 'fullname'] = usgs[usgs['site_id'] == site_id]['lake_name'].values[0]

# meta.to_csv("../../../metadata/lake_metadata_wStats4.feather")
# df.reset_index(inplace=True)
# df.drop([''])