import numpy as np
from scipy.spatial.distance import euclidean
import pandas as pd
from fastdtw import fastdtw
import os
import re
import sys
import pdb
from scipy import interpolate
# from dtw import dtw

raw_dir = '../../../data/raw/figure3/' #unprocessed data directory
proc_dir = '../../../data/processed/WRR_69Lake/'
output_dir = '../manylakes2/single_models/'
lnames = set()
mendota_id = "1099282"
lnames.add(mendota_id)
n_features = 7
n_lakes = 0



############################################################################################################################
# July 2019 - calculate dynamic time warping between Mendota GLM energy and energy from other lake models
#####################################################################################################################


#csv to write at end with lake and distance
csv = []
csv.append(mendota_id+"," + "0.0")

#load mendota GLM energy
m_glm_path = raw_dir+"nhd_"+mendota_id+"_temperatures_wEnergy.feather"
m_glm = pd.read_feather(m_glm_path)
m_glm_val = m_glm.values
n_depths = m_glm_val.shape[1]-4 #minus date and ice flag and energies
# print("n_depths: " + str(n_depths))
max_depth = 0.5*(n_depths-1)
depths = np.arange(0, max_depth+0.5, 0.5)


#interpolate missing values
m_glm_val_toInt = np.array(m_glm_val[:,1:-3], dtype=np.float64)
if np.isnan(np.sum(m_glm_val_toInt)):
    # print("Warning: there is missing data in glm output")
    for i in range(depths.shape[0]):
        for t in range(m_glm_val_toInt.shape[0]):
            if np.isnan(m_glm_val_toInt[t,i]):
                x = depths[i]
                xp = depths[0:(i-1)]
                yp = m_glm_val_toInt[t,0:(i-1)]
                f = interpolate.interp1d(xp, yp,  fill_value="extrapolate")
                m_glm_val_toInt[t,i] = f(x) #interp_temp
    assert not np.isnan(np.sum(m_glm_val_toInt))

m_glm_val[:,1:-3] = m_glm_val_toInt

#trim dates to o_en sequence
dates = [np.datetime64(m_glm_val[i,0].date()) for i in range(m_glm_val.shape[0])]

for filename in os.listdir(raw_dir):
    #parse lakename from file,
    m = re.search(r'^nhd_(\d+)_.*', filename)
    if m is None:
        continue
    name = m.group(1)
    if name not in lnames:
        #for each unique lake
        print("lake ", name)
        lnames.add(name)

        #load output energy
        o_en_path = output_dir+name+"PGRNNoutputOn"+name+".feather"
        o_en = pd.read_feather(o_en_path)

        #get rid of "nan" dates
        o_en_values = np.array(o_en.values[:,1:], dtype=np.float64)
        new_val = o_en.values
        new_val = new_val[~np.isnan(o_en_values[:,-1])]
        o_en_vals = new_val[:,-1] #parse energy
        o_t_vals = new_val[:,1:-3]
        o_en_vals = np.array(o_en_vals, dtype=np.float64) #object to float
        o_t_vals = np.array(o_t_vals, dtype=np.float64) #object to float

        #get dates
        first_date = new_val[0][0]
        last_date = new_val[-1][0]


        #trim mendota GLM to dates
        start_ind = np.where(dates == np.datetime64(first_date.date()))[0][0]
        end_ind = np.where(dates == np.datetime64(last_date.date()))[0][0]
        m_glm_val_trimmed = m_glm_val[start_ind:end_ind+1,:]
        m_glm_val_trimmed_npy = np.array(m_glm_val_trimmed[:,1:],dtype=np.float64)

        m_en_vals = m_glm_val_trimmed_npy[:,-1]
        m_t_vals = m_glm_val_trimmed_npy[:,1:-3]

        max_d = np.amin([m_t_vals.shape[1], o_t_vals.shape[1]])

        #calculate dtw
        x = o_en_vals[~np.isnan(o_en_vals)]
        # x = o_t_vals[:,:max_d]
        # x = (x - x.mean()) / x.std()

        y = m_en_vals[~np.isnan(m_en_vals)]
        # y = m_t_vals[:, :max_d]
        # y = (y - y.mean()) / y.std()

        print(x.shape)
        print(y.shape)
        assert o_en_vals.shape == m_en_vals.shape
        dist, path = fastdtw(x, y, dist=euclidean)
        print("dist=",dist)


        # euclidean_norm = lambda x, y: np.abs(x - y)

        # # d, _, _, _ = dtw(x, y, dist=euclidean_norm)
        # dist = np.linalg.norm(x - y)
        # print("dist=",dist)
        # # print("path=", path)
        csv.append(name+","+str(dist))

with open('dtw_results_temps.csv','a') as file:
    for line in csv:
        file.write(line)
        file.write('\n')
