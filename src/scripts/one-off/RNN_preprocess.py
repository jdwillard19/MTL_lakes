import os.path
import numpy as np
import sys
import getopt
import math
from pandas import DataFrame
sys.path.append('../../data')
sys.path.append('../../../data')
sys.path.append('/home/invyz/workspace/Research/lake_monitoring/src/data')

from rw_data import readMatData, saveMatData

#######################################################
# This is a preprocessing script that persists data from 
# the "Modeled_temp" into a matrices of (depth, time) 
# dimensions that will be used as inputs for an LSTM. 
#
# attempt to create suitable input data for LSTM
# 
###########################################################

data_fpath = "../../../data/processed/mendota_sampled.mat"
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

x = mat['Xc_doy_int'][:,0:11]
x_raw = mat['Xc_doy_int_unstandardized'][:,0:12]
y = mat['Modeled_temp_tuned_int']

std_depth_values = np.array(np.sort(list(set(mat['Xc_doy'][:,2].flat))))

n_depths = len(set(mat['Depth_int'].flat))
u_depth_values = np.array(np.sort(list(set(mat['Depth_int'].flat))))
depths = mat['Depth_int']
udates = np.sort(mat['udates'])
num_u_dates = np.shape(udates)[0]
datenums = mat['datenums_int']
n = np.shape(y)[0]

###########################################3
#create matrix that maps (depth, time) pair to modeled temp
###############################


#TODO generalize these parameters

#parameters
seq_per_depth = math.floor(udates.size / seq_length)
n_features = 11 #hardcoded for now, maybe not in the future?
n_seq = seq_per_depth * n_depths
n_ignore_dates = udates.size - seq_per_depth
print(n_ignore_dates)



#
seq_mat_feat = np.empty(shape=(n_depths, seq_length*seq_per_depth, n_features))
seq_mat_feat_raw = np.empty(shape=(n_depths, seq_length*seq_per_depth, n_features+1))
seq_mat_label = np.empty(shape=(n_depths, seq_length*seq_per_depth, 1))
#ignore remainder dates
ignore_dates = udates[-2:]
ignore_dates = ignore_dates[:,0]
for i in range(0,n):
    if(i % 10000 == 0):
        print(i, " data processed")
    if mat['datenums_int'][i] in ignore_dates:
        #skip over ignored dates
        continue
    else:
        #get depth and datenum indices for matrix
        depth_ind = np.where(u_depth_values == depths[i])[0]
        datenum_ind = np.where(udates == datenums[i])[0]

        #place data
        seq_mat_feat[depth_ind,datenum_ind,:] = x[i,:]
        seq_mat_feat_raw[depth_ind,datenum_ind,:] = x_raw[i,:]
        seq_mat_label[depth_ind,datenum_ind,0] = y[i]


mat['Depth_Time_Series_Features'] = seq_mat_feat
mat['Depth_Time_Series_Features_Raw'] = seq_mat_feat_raw
mat['Depth_Time_Series_Labels'] = seq_mat_label
saveMatData(mat, os.path.abspath(data_fpath))


