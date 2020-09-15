import os.path
import numpy as np
import matplotlib.pyplot as plt
import sys
import math
import csv
from pandas import DataFrame
sys.path.append('../../models')
sys.path.append('../../data')
from rw_data import readMatData
from train_basic_nn_model import nn_model
from sklearn.metrics import mean_squared_error
from keras.models import save_model, load_model
from data_operations import calculateRMSE, calculatePhysicalLossDensityDepth

from rw_data import readMatData, saveMatData
#read data
mat = readMatData(os.path.abspath('../../../data/processed/mendota_sampled.mat'))
x = mat['Xc_doy'][:,0:11]
y = mat['Modeled_temp_tuned'][:,0]
depths = mat['Depth']
dates = mat['datenums']
udates = np.sort(mat['udates'])

#################################################################
# this script runs an experiment for predicting GLM output using
# a basic feed forward 3 layer ANN. it saves the RMSE and depth 
# inconsistency for a variety of training sizes
##############################################################


n = np.shape(mat['Xc_doy'])[0]
n10 = n-math.ceil(np.shape(mat['Xc_doy'])[0]*.10)
n15 = n-math.ceil(np.shape(mat['Xc_doy'])[0]*.15)
n20 = n-math.ceil(np.shape(mat['Xc_doy'])[0]*.2)
n25 = n-math.ceil(np.shape(mat['Xc_doy'])[0]*.25)
n33 = n-math.ceil(np.shape(mat['Xc_doy'])[0]*.33)
n50 = n-math.ceil(np.shape(mat['Xc_doy'])[0]*.50)
n66 = n- math.ceil(np.shape(mat['Xc_doy'])[0]*.66)

y = mat['Modeled_temp'][:,0]
x = mat['Xc_doy'][:,0:11]
nvec = [n66,n50,n33,n25,n20, n15, n10]
NN_PHYS_LOSS = np.empty(shape=len(nvec))
NN_RMSE = np.empty(shape=len(nvec))
n_trials = 2

# for i in range(0,len(perc_train)):
# 	for t in range
#     model = modelvec[i]
#     n = nvec[i]
#     NN_RMSE[i] = math.sqrt(mean_squared_error(model.predict(x[n:,:]),y[n:]))
#     print(model.predict(x).shape, depths.shape, udates.shape, days.shape)
#     NN_PHYS_LOSS[i] = calculatePhysicalLossDensityDepth(model.predict(x), depths, udates, days)

train_percs = [.33, .5, .66, .75, .8, .85, .9]

with open('ANN_results.csv', 'w') as csvfile:
	fieldnames = ['train_percent','mean_rmse', 'std_rmse', 'mean_physloss', 'std_physloss']
	writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
	writer.writeheader()
	for i in range(0,len(train_percs)):
		RMSE = np.empty(n_trials)
		PHYS_LOSS = np.empty(n_trials)
		x_tst = x[(1-nvec[i]):,:]
		y_tst = y[(1-nvec[i]):]
		for t in range(0,n_trials):
			print("trial ", t)
			model = nn_model(x, y, epochs=5,validation_split=(1-train_percs[i]))
			pred_tst = model.predict(x_tst)
			RMSE[t] = math.sqrt(mean_squared_error(pred_tst,y_tst))
			print("RMSE, ",RMSE[t])
			pred = model.predict(x)
			PHYS_LOSS[t] = calculatePhysicalLossDensityDepth(pred, depths, udates, dates)
			print("PHYSLOSS, ",PHYS_LOSS[t])
	    #avg +std
		mean_rmse = np.mean(RMSE, dtype=np.float64)
		std_rmse = np.std(RMSE, dtype=np.float64)
		print(mean_rmse, ":", std_rmse)
		mean_phys = np.mean(PHYS_LOSS, dtype=np.float64)
		std_phys = np.std(PHYS_LOSS, dtype=np.float64)
		writer.writerow({'train_percent':train_percs[i],
						 'mean_rmse': mean_rmse, 'std_rmse': std_rmse, 
						 'mean_physloss': mean_phys, 'std_physloss': std_phys}) 
