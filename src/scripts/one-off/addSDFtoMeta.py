import pandas as pd 
import numpy as np 
import math
import pdb
import re
#add sdf data to metadata files

meta = pd.read_feather("../../../metadata/lake_metadata_wNew2.feather")
sdf = pd.read_csv("../../../metadata/data_release_metadata.csv")
cross = pd.read_feather("../../../metadata/nhd_to_nhdhr.feather")

# ids = meta[~np.isfinite(meta['SDF'])]['nhd_id']
ids = meta['nhd_id']
old_meta = meta
meta.set_index('nhd_id', inplace=True)
nhdhr_ids = np.array(['nhdhr_'+txt for txt in ids])
ct = 0

to_del = ['120018008', '120020307', '120020636', '32671150', '58125241', '120020800', '91598525']
ids = np.setdiff1d(ids, to_del)
for name in ids:
	old_name = None
	ct += 1
	print("target ", name, "        " ,str(ct))
	#set in metadata file
	if math.isnan(meta.loc[name, 'SDF']):
		meta.loc[name, 'SDF'] = sdf[sdf['site_id'] == 'nhdhr_'+name]['SDF'].values[0]



	#set in target results file
	res = pd.read_csv("../../../results/transfer_learning/target_"+name+"/resultsPGRNNbasic_wNew_norm2")
	if len(cross[cross['WRR_ID']=='nhd_'+name]) == 1:
		old_name = name
		name =	re.search("nhdhr_(\d+)", cross[cross['WRR_ID']=='nhd_'+name]['site_id'].values[0]).group(1)

	sdf_curr = sdf[sdf['site_id'] == 'nhdhr_'+name]['SDF'].values[0]
	for index, row in res.iterrows():
		# if index == 203:
		iid = row['source_id']

		#skip if itself
		if iid == name:
			continue

		#if nhdid change to nhdhr id
		if len(cross[cross['WRR_ID']=='nhd_'+iid]) == 1:
			iid = cross[cross['WRR_ID']=='nhd_'+iid]['site_id'].values[0]
			iid = re.search("nhdhr_(\d+)", iid).group(1)


		sdf_oth = sdf[sdf['site_id'] == 'nhdhr_'+iid]['SDF'].values[0]
		res.loc[index, 'SDF'] = abs(sdf_curr - sdf_oth)

	if old_name:
		res.to_csv("../../../results/transfer_learning/target_"+old_name+"/resultsPGRNNbasic_wNew_norm2")
	else:
		res.to_csv("../../../results/transfer_learning/target_"+name+"/resultsPGRNNbasic_wNew_norm2")

# save_meta = meta.reset_index()
# save_meta.to_feather("../../../metadata/lake_metadata_wNew2.feather")
