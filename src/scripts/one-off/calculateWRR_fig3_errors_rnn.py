import os
import re
import sys
import numpy as np
import pandas as pd
import pdb
#######################################
# Sept 2019
# Jared - 
#######################################

#read lake metadata file to get all the lakenames
metadata = pd.read_csv("../../../metadata/lake_metadata.csv")
lakenames = [str(i) for i in metadata.iloc[:,0].values] # to loop through all lakes
csv = []
csv.append("target_nhd_id,pg1,pg2,pg3,pgAvg,rn1,rn2,rn3,rnAvg")
ct = 0
for target_id in lakenames: 
	ct +=1

	if target_id == '13293262':
		continue
	# target_id = '1097324'
	print("target ",ct,": ", target_id)
	#load output
	# pgtest= np.array(pd.read_feather("../manylakes2/outputs_full/"+target_id+"PGRNN_output_trial0.feather").values)
	# labtest = np.array(pd.read_feather("../manylakes2/labels/"+target_id+"_full_label.feather").values)
	mod = ""
	rn_out0f = pd.read_feather("../manylakes2/outputs_full/rnn/"+target_id+"RNN_output_trial0.feather")
	rn_out0 = rn_out0f.values[:,1:]
	rn_out0 = np.array(rn_out0, dtype=np.float32)
	rn_out1f = pd.read_feather("../manylakes2/outputs_full/rnn/"+target_id+"RNN_output_trial1.feather")
	rn_out1 = rn_out1f.values[:,1:]
	rn_out1 = np.array(rn_out1, dtype=np.float32)
	rn_out2f = pd.read_feather("../manylakes2/outputs_full/rnn/"+target_id+"RNN_output_trial2.feather")
	rn_out2 = rn_out2f.values[:,1:]
	rn_out2 = np.array(rn_out2, dtype=np.float32)	
	rn_out3f = pd.read_feather("../manylakes2/outputs_full/rnn/"+target_id+"RNN_output_trial3.feather")
	rn_out3 = rn_out3f.values[:,1:]
	rn_out3 = np.array(rn_out3, dtype=np.float32)	
	rn_out4f = pd.read_feather("../manylakes2/outputs_full/rnn/"+target_id+"RNN_output_trial4.feather")
	rn_out4 = rn_out4f.values[:,1:]
	rn_out4 = np.array(rn_out4, dtype=np.float32)	
	rn_outAvgf = pd.read_feather("../manylakes2/outputs_full/rnn/"+target_id+"RNN_output_avgT.feather")
	rn_outAvg = rn_outAvgf.values[:,1:]
	rn_outAvg = np.array(rn_outAvg, dtype=np.float32)
	# lab = pd.read_feather("../manylakes2/labels/"+target_id+"_label.feather")
	labf = pd.read_feather("../../../data/processed/labels/"+target_id+"_full.feather")
	plabf = pd.read_feather("../../../data/processed/labels/"+target_id+"_partial.feather")
	#first label date
	lab = labf.values[:,1:]
	plab = plabf.values[:,1:]
	plab_dates = plabf.values[:,0]
	outr0_dates = rn_out0f.values[:,0]
	outr1_dates = rn_out1f.values[:,0]
	outr2_dates = rn_out2f.values[:,0]
	outr2_dates = rn_out3f.values[:,0]
	outr2_dates = rn_out4f.values[:,0]
	outrA_dates = rn_outAvgf.values[:,0]
	#trim labels for dates not in outputs
	plab = plab[np.isin(plab_dates,outr0_dates)]
	plab_dates = plab_dates[np.isin(plab_dates,outr0_dates)]
	# plab = plab[np.isin(plab_dates,outp1_dates)]
	# plab_dates = plab_dates[np.isin(plab_dates,outp1_dates)]
	# plab = plab[np.isin(plab_dates,outp2_dates)]
	# plab_dates = plab_dates[np.isin(plab_dates,outp2_dates)]
	# plab = plab[np.isin(plab_dates,outpA_dates)]
	# plab_dates = plab_dates[np.isin(plab_dates,outpA_dates)]
	# plab = plab[np.isin(plab_dates,outr0_dates)]
	# plab_dates = plab_dates[np.isin(plab_dates,outr0_dates)]
	# plab = plab[np.isin(plab_dates,outr1_dates)]
	# plab_dates = plab_dates[np.isin(plab_dates,outr1_dates)]
	# plab = plab[np.isin(plab_dates,outr2_dates)]
	# plab_dates = plab_dates[np.isin(plab_dates,outr2_dates)]
	# plab = plab[np.isin(plab_dates,outrA_dates)]
	# plab_dates = plab_dates[np.isin(plab_dates,outrA_dates)]

	rn_out0 = rn_out0[np.isin(outr0_dates,plab_dates)]
	rn_out1 = rn_out1[np.isin(outr1_dates,plab_dates)]
	rn_out2 = rn_out2[np.isin(outr2_dates,plab_dates)]
	rn_out3 = rn_out2[np.isin(outr3_dates,plab_dates)]
	rn_out4 = rn_out2[np.isin(outr4_dates,plab_dates)]
	rn_outAvg = rn_outAvg[np.isin(outrA_dates,plab_dates)]
	lab = np.array(lab, dtype=np.float32)
	plab = np.array(plab, dtype=np.float32)
	# full_lab = pd.read_feather("../manylakes2/labels/"+target_id+"_full_label.feather")
	# full_lab = full_lab.values[:,1:]
	# full_lab = np.array(full_lab, dtype=np.float32)
	i = plab.shape[0]
	j = rn_out0.shape[1]
	# k = pg_out0.shape[0]
	# pg_out0 = pg_out0[:i,:]
	# pg_out1 = pg_out1[:i,:]
	# pg_out2 = pg_out2[:i,:]
	# pg_outAvg = pg_outAvg[:i,:]
	# rn_out0 = rn_out0[:i,:]
	# rn_out1 = rn_out1[:i,:]
	# rn_out2 = rn_out2[:i,:]
	# rn_outAvg = rn_outAvg[:i,:]
	# lab = lab[:i,:j]
	# lab = full_lab[:k,:]
	
	lab = plab
	rn_out0 = rn_out0[~np.isnan(lab)]
	rn_out1 = rn_out1[~np.isnan(lab)]
	rn_out2 = rn_out2[~np.isnan(lab)]
	rn_out3 = rn_out3[~np.isnan(lab)]
	rn_out4 = rn_out4[~np.isnan(lab)]
	rn_outAvg = rn_outAvg[~np.isnan(lab)]
	lab = lab[~np.isnan(lab)]

	rn_out1 = rn_out1[~np.isnan(rn_out0)]
	rn_out2 = rn_out2[~np.isnan(rn_out0)]
	rn_out3 = rn_out3[~np.isnan(rn_out0)]
	rn_out4 = rn_out4[~np.isnan(rn_out0)]
	# rn_outAvg = (rn_out0 + rn_out1 + rn_out2) / 3
	rn_outAvg = rn_outAvg[~np.isnan(rn_out0)]
	lab = lab[~np.isnan(rn_out0)]
	rn_out0 = rn_out0[~np.isnan(rn_out0)]

	

	rn0_rmse = str(np.sqrt(((rn_out0 - lab) ** 2).mean()))
	rn1_rmse = str(np.sqrt(((rn_out1 - lab) ** 2).mean()))
	rn2_rmse = str(np.sqrt(((rn_out2 - lab) ** 2).mean()))
	rn3_rmse = str(np.sqrt(((rn_out3 - lab) ** 2).mean()))
	rn4_rmse = str(np.sqrt(((rn_out4 - lab) ** 2).mean()))
	rnA_rmse = str(np.sqrt(((rn_outAvg - lab) ** 2).mean()))
	csv.append(",".join([target_id, rn0_rmse, rn1_rmse, rn2_rmse, rn3_rmse,rn4_rmse, rnA_rmse]))


with open("./rnn_results_1015.csv",'a') as file:
	for line in csv:
		file.write(line)
		file.write('\n')