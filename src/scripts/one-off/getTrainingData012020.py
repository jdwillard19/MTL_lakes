import pandas as pd
import numpy as np
from shutil import copyfile, copytree
import os
import pdb
import re


#get names file
lakes = pd.read_feather("./TrainLakes.feather")
lakes = lakes['nhdhr_id']
conv = pd.read_feather("../../../metadata/nhd_to_nhdhr.feather")
if not os.path.exists("./trainingDataJordanConfirm"):
	os.mkdir("./trainingDataJordanConfirm")

to_del = ['120018008', '120020307', '120020636', '32671150', '58125241', '120020800', '91598525']
pdb.set_trace()
ct = 0
for name in lakes:
	fullname = name

	m = re.search("nhdhr_(.*)", name)
	name = m.group(1)
	if np.isin(name, to_del):
		print("skipping ", name)
		continue
	ct += 1 
	print("lake ", ct, ": ", name)
	new_path = "../../../data/processed/lake_data/"+name
	if not os.path.exists(new_path):
		#get new nhdhr id
		search_name = "nhdhr_"+name
		old_id = conv[conv['site_id'] == search_name]['WRR_ID'].values[0]
		m = re.search("nhd_(\d+)", old_id)
		old_id = m.group(1)
		dated_path = "../../../data/processed/lake_data/"+str(old_id)
		copytree(dated_path, new_path)
		send_file = new_path+"/full_obs.npy"
		dates_path = new_path+"/dates.npy"
		data = np.load(send_file)
		dates = np.load(dates_path)

	else:
		send_file = new_path+"/full_obs.npy"
		dates_path = new_path+"/dates.npy"
		data = np.load(send_file)
		dates = np.load(dates_path)
	dates = [x.decode() for x in dates]
	df = pd.DataFrame(data)
	df.columns = dates
	# if ct == 1:
	# 	pdb.set_trace()
	df.to_feather("./trainingDataJordanConfirm/"+name+"_fullData_training_samples.feather")
