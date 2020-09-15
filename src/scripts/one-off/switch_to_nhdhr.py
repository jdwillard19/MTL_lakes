import pandas as pd 
import numpy as np 
import sys
import os
import pdb
sys.path.append('../../data')
from metadata_ops import nhd2nhdhr
import shutil

metadata = pd.read_csv("../../../metadata/lake_metadata_wStats4.feather")
# glm_all_f = pd.read_csv("../../../results/glm_transfer/RMSE_transfer_glm.csv")
# temp = pd.read_csv("../../../results/transfer_learning/target_13293262/resultsPGRNNbasic_wNew_norm2")
conv = pd.read_feather("../../../data/raw/crosswalk/nhd_to_nhdhr.feather")
# trn_tst_f = pd.read_feather("../../../data/processed/lake_splits/trainTestNewLakes.feather")

ids = metadata['nhd_id'].values
nhdhr_ids = []
to_del = ['120018008', '120020307', '120020636', '32671150', '58125241', '120020800', '91598525']
# ids = np.setdiff1d(ids, to_del)
for i, num in enumerate(ids):

	print(i)
	old_id = num
	if np.isin('nhd_'+num, conv['WRR_ID']):
		new_id = nhd2nhdhr(num)
		#convert
	else:
		new_id = old_id
	# if not os.path.exists("../../../metadata/diff/target_"+new_id+"_featsJan20_wHR.csv"):
	# 	if not os.path.exists("../../../metadata/diff/target_"+new_id+"_wStats_wSeason_wPercDiffDepth.csv"):
	# 		shutil.copyfile("../../../metadata/diff/target_"+old_id+"_wStats_wSeason_wPercDiffDepth.csv", \
	# 						"../../../metadata/diff/target_"+new_id+"_wStats_wSeason_wPercDiffDepth.csv")
	# 	diff = pd.read_csv("../../../metadata/diff/target_"+new_id+"_wStats_wSeason_wPercDiffDepth.csv")
	# 	nhdhr_ids = []
	# 	num = new_id

	# 	for d_id in diff['nhd_id'].values:
	# 		if np.isin('nhd_'+d_id, conv['WRR_ID']):
	# 			d_id = nhd2nhdhr(d_id)
	# 		nhdhr_ids.append(d_id)
	# 	diff['site_id'] = nhdhr_ids #add col
		
	# 	#save
	# # 	diff.to_csv("../../../metadata/diff/target_"+new_id+"_featsJan20_wHR.csv")

	# if not os.path.exists("../../../data/raw/figure3/nhd_"+new_id+"_meteo.csv"):
	# 	shutil.copyfile("../../../data/raw/figure3/nhd_"+old_id+"_meteo.csv", "../../../data/raw/figure3/nhd_"+new_id+"_meteo.csv")
	# if not os.path.exists("../../../data/raw/figure3/nhd_"+new_id+"_temperatures.feather"):
	# 	shutil.copyfile("../../../data/raw/figure3/nhd_"+old_id+"_temperatures.feather", "../../../data/raw/figure3/nhd_"+new_id+"_temperatures.feather")
	# if not os.path.exists("../../../data/raw/figure3/nhd_"+new_id+"_test_train.feather"):
	# 	if os.path.exists("../../../data/raw/figure3/nhd_"+old_id+"_train_all_profiles.feather"):
	# 		shutil.copyfile("../../../data/raw/figure3/nhd_"+old_id+"_train_all_profiles.feather", "../../../data/raw/figure3/nhd_"+new_id+"_test_train.feather")
	# 	else:
	# 		shutil.copyfile("../../../data/raw/figure3/nhd_"+old_id+"_test_train.feather", "../../../data/raw/figure3/nhd_"+new_id+"_test_train.feather")
	# 	# #copy
		# if not os.path.exists("../../../results/transfer_learning/target_"+new_id):
		# 	os.mkdir("../../../results/transfer_learning/target_"+new_id)
		# if not os.path.exists("../../../results/transfer_learning/target_"+new_id+"/resultsPGRNNbasic_wNew_norm2"):
		# 	shutil.copyfile("../../../results/transfer_learning/target_"+old_id+"/resultsPGRNNbasic_wNew_norm2", \
		# 					"../../../results/transfer_learning/target_"+new_id+"/resultsPGRNNbasic_wNew_norm2")
		
		# if not os.path.exists("../../../data/processed/lake_data/"+new_id):
		# 	shutil.copytree("../../../data/processed/lake_data/"+old_id, "../../../data/processed/lake_data/"+new_id)
		
		# if not os.path.exists("../../../models/single_lake_models/"+new_id):
		# 	os.mkdir("../../../models/single_lake_models/"+new_id)
		# if not os.path.exists("../../../models/single_lake_models/"+new_id+"/PGRNN_basic_normAll"):
		# 	shutil.copyfile("../../../models/single_lake_models/"+old_id+"/PGRNN_basic_normAll", \
		# 					  "../../../models/single_lake_models/"+new_id+"/PGRNN_basic_normAll")
		# if not os.path.exists("../../../results/transfer_learning/target_"+new_id):
		# 	os.mkdir("../../../results/transfer_learning/target_"+new_id)

		# if not os.path.exists("../../../results/transfer_learning/target_"+new_id+"/resultsPGRNNbasic_norm2_bias_wSeason_fix"):
		# 	shutil.copyfile("../../../results/transfer_learning/target_"+old_id+"/resultsPGRNNbasic_norm2_bias_wSeason_fix", \
		# 					  "../../../results/transfer_learning/target_"+new_id+"/resultsPGRNNbasic_norm2_bias_wSeason_fix")

		# if not os.path.exists("../../../results/transfer_learning/target_"+new_id+"/resultsPGRNNbasic_wNew_norm2"):
		# 	shutil.copyfile("../../../results/transfer_learning/target_"+old_id+"/resultsPGRNNbasic_wNew_norm2", \
		# 					  "../../../results/transfer_learning/target_"+new_id+"/resultsPGRNNbasic_wNew_norm2")

# 	new_id = old_id
	nhdhr_ids.append(str(new_id))
metadata['site_id'] = nhdhr_ids
metadata.to_feather("../../../metadata/lake_metadata_wStats4.feather")
