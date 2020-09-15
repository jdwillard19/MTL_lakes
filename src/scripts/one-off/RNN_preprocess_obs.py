import os.path
import numpy as np
import matplotlib.pyplot as plt
import sys
import getopt
import math
from pandas import DataFrame
sys.path.append('../../data')
sys.path.append('../../../data')
sys.path.append('/home/invyz/workspace/Research/lake_monitoring/src/data')

from rw_data import readMatData, saveMatData

#######################################################
# UNFINISHED - preprocess for observed data experiments
###########################################################

data_fpath = "../../../data/processed/mendota.mat"
data_samp_fpath = "../../../data/processed/mendota_sampled.mat"
#parse command line arguments
def main(argv):
   seq_length = ''
   try:
      opts, args = getopt.getopt(argv,"hl:",["sLength="])
   except getopt.GetoptError:
      print('usage: RNN_proprocess.py -l <seq_length>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('RNN_proprocess.py -l <seq_length>')
         sys.exit()
      elif opt in ("-l", "--seq_length"):
         seq_length = arg
   print ('sequence length is ', seq_length)

seq_length = -1
main(sys.argv[1:])
if(len(sys.argv[1:]) == 2):
	seq_length = int(sys.argv[2])
else:
	print("ERROR, usage: RNN_proprocess.py -l <seq_length>")


#load data
mat = readMatData(os.path.abspath(data_fpath))
mat_s = readMatData(os.path.abspath(data_samp_fpath))


x = mat_s['Xc_doy'][:,0:11]
y = mat['Observed_temp']


std_depth_values = np.array(np.sort(list(set(mat['Xc_doy'][:,2].flat))))

n_depths = len(set(mat_s['Depth'].flat))
u_depth_values = np.array(np.sort(list(set(mat_s['Depth'].flat))))
depths_s = mat_s['Depth']
depths = mat['Depth']
udates = np.array(np.sort(list(set(mat_s['datenums'].flat))))
num_u_dates = np.shape(udates)[0]
datenums = mat['datenum']
datenums_s = mat_s['datenums']
n = np.shape(x)[0]
n_s = np.shape(y)[0]
###########################################3
#create matrix that maps (depth, time) pair to modeled temp
###############################


#TODO generalize these parameters

#parameters
seq_per_depth = math.floor(n / seq_length)
n_features = 11 #hardcoded for now, maybe not in the future?
n_seq = seq_per_depth * n_depths
n_ignore_dates = n - seq_per_depth*seq_length
print(n_ignore_dates)




seq_mat_feat = np.empty(shape=(n_depths, num_u_dates, n_features))
seq_mat_label = np.zeros(shape=(n_depths, num_u_dates, 1))
#ignore remainder dates

ignore_dates = udates[-n_ignore_dates:]

#fill feature matrix, with depth as rows and dates as columns
for i in range(0,n):
	if(i % 50000 == 0):
		print(i, " data processed")
	if mat_s['datenums'][i] in ignore_dates:
		#skip over ignored dates
		continue
	else:
		#get depth and datenum indices for matrix
		depth_ind = np.where(u_depth_values == depths_s[i])[0]
		datenum_ind = np.where(udates == datenums_s[i])[0]

		#place data
		seq_mat_feat[depth_ind,datenum_ind,:] = x[i,:]

		# #find y that matches
		# seq_mat_label[depth_ind,datenum_ind,0] = y[i]

notAdded = 0

for i in range(0,n_s):
	#fill sample labels from observed data
	if mat['datenum'][i] in ignore_dates:
		continue
	else:
		depth_ind = np.where(u_depth_values == depths[i])[0]
		date_ind = np.where(udates == datenums[i])[0]

		if(depth_ind.size == 0 or date_ind.size == 0):
			print("depth ", depths[i], " not in sampled dataset",
				  " or ", datenums[i], " not in it")
			notAdded += 1

		else:
			seq_mat_label[depth_ind, date_ind, 0] = y[i]

print("labels used ", np.count_nonzero(seq_mat_label))
print("labels not used ", notAdded)

#removes depths with not many labels
# depth_observation_cutoff = 35
# depth_rm_ind = []
# for d in range(0,n_depths):
# 	if np.count_nonzero(seq_mat_label[d,:]) < depth_observation_cutoff:
# 		depth_rm_ind.append(d)
# 	#print("depth ", d, ": ", np.count_nonzero(seq_mat_label[d,:]))

# seq_mat_feat = np.delete(seq_mat_feat, depth_rm_ind, axis=0)
# seq_mat_label = np.delete(seq_mat_label, depth_rm_ind, axis=0)


mat['Depth_Time_Series_Features'] = seq_mat_feat
mat['Depth_Time_Series_Labels'] = seq_mat_label
saveMatData(mat, os.path.abspath(data_fpath))


